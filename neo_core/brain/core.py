"""
Brain — Agent Orchestrateur
============================
Cortex exécutif du système Neo Core.

Responsabilités :
- Analyser les requêtes transmises par Vox
- Consulter le contexte enrichi par Memory
- Déterminer l'action optimale
- Générer dynamiquement des agents spécialisés (Workers) via Factory
- Apprendre des résultats des Workers

Authentification OAuth (méthode OpenClaw) :
- Méthode 1 : Bearer token + header beta "anthropic-beta: oauth-2025-04-20"
- Méthode 2 : Conversion OAuth → API key via /claude_cli/create_api_key
- Fallback : Clé API classique via LangChain

Stage 3 : Moteur d'Orchestration
Stage 5 : Résilience (retry, circuit breaker, health monitoring)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Optional

from neo_core.validation import validate_message, ValidationError

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from neo_core.config import NeoConfig, default_config, get_agent_model
from neo_core.infra.resilience import (
    RetryConfig,
    RetryableError,
    NonRetryableError,
    retry_with_backoff,
    CircuitBreaker,
    CircuitOpenError,
    HealthMonitor,
    create_resilience_from_config,
)
from neo_core.oauth import (
    is_oauth_token,
    get_valid_access_token,
    get_best_auth,
    get_api_key_from_oauth,
    OAUTH_BETA_HEADER,
)
from neo_core.brain.teams.factory import WorkerFactory, TaskAnalysis
from neo_core.brain.teams.worker import WorkerType, WorkerResult, WorkerState
from neo_core.brain.teams.crew import CrewExecutor, CrewStep, CrewEvent
from neo_core.brain.prompts import BRAIN_SYSTEM_PROMPT, DECOMPOSE_PROMPT, BrainDecision

if TYPE_CHECKING:
    from neo_core.memory.agent import MemoryAgent


class WorkerLifecycleManager:
    """
    Registre centralisé des Workers — garantit leur destruction.

    Thread-safe : toutes les opérations sur _active et _history
    sont protégées par un threading.Lock.

    Responsabilités :
    - Tracker tous les Workers actifs (en cours d'exécution)
    - Conserver l'historique des Workers terminés (pour stats)
    - Garantir que AUCUN Worker ne reste en mémoire sans cleanup
    """

    def __init__(self, max_history: int = 50):
        import threading
        self._lock = threading.Lock()
        self._active: dict[str, object] = {}   # worker_id → Worker (en cours)
        self._history: list[dict] = []          # Historique des Workers terminés
        self._max_history = max_history
        self._total_created = 0
        self._total_cleaned = 0

    def register(self, worker) -> str:
        """Enregistre un Worker dans le registre. Retourne son ID."""
        with self._lock:
            self._active[worker.worker_id] = worker
            self._total_created += 1
        return worker.worker_id

    def unregister(self, worker) -> None:
        """
        Retire un Worker du registre et enregistre son historique.

        IMPORTANT : Doit être appelé APRÈS que Memory a récupéré
        les apprentissages et APRÈS cleanup().
        """
        with self._lock:
            # Sauvegarder les infos avant suppression
            info = worker.get_lifecycle_info()
            self._history.append(info)

            # Limiter l'historique
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            # Retirer du registre actif
            self._active.pop(worker.worker_id, None)
            self._total_cleaned += 1

    def get_active_workers(self) -> list[dict]:
        """Retourne la liste des Workers actuellement actifs."""
        with self._lock:
            return [w.get_lifecycle_info() for w in self._active.values()]

    def get_history(self, limit: int = 10) -> list[dict]:
        """Retourne les N derniers Workers terminés."""
        with self._lock:
            return self._history[-limit:]

    def get_stats(self) -> dict:
        """Statistiques du lifecycle manager."""
        with self._lock:
            return {
                "active_count": len(self._active),
                "total_created": self._total_created,
                "total_cleaned": self._total_cleaned,
                "leaked": self._total_created - self._total_cleaned - len(self._active),
                "history_size": len(self._history),
            }

    def cleanup_all(self) -> int:
        """
        Force le cleanup de TOUS les Workers encore actifs.

        Retourne le nombre de Workers nettoyés.
        Utile lors du shutdown du système.
        """
        with self._lock:
            workers_to_clean = list(self._active.items())

        count = 0
        for worker_id, worker in workers_to_clean:
            try:
                worker.cleanup()
                self.unregister(worker)
                count += 1
            except Exception as e:
                # Force la suppression même si cleanup échoue
                logger.debug("Erreur lors du cleanup du Worker %s: %s", worker_id, e)
                with self._lock:
                    self._active.pop(worker_id, None)
                count += 1
        return count


@dataclass
class Brain:
    """
    Agent Brain — Orchestrateur du système Neo Core.

    Analyse les requêtes, consulte Memory, et détermine
    la meilleure action à entreprendre.

    Stage 3 : Peut créer des Workers spécialisés via Factory
    Stage 5 : Résilience (retry, circuit breaker, health monitoring)
    """
    config: NeoConfig = field(default_factory=lambda: default_config)
    memory: Optional[MemoryAgent] = None
    _llm: Optional[object] = None
    _force_mock: bool = False
    _oauth_mode: bool = False
    _auth_method: str = ""
    _anthropic_client: Optional[object] = None
    _factory: Optional[WorkerFactory] = None
    _health: Optional[HealthMonitor] = None
    _retry_config: Optional[RetryConfig] = None
    _model_config: Optional[object] = None
    _worker_manager: Optional[WorkerLifecycleManager] = None
    _self_patcher: Optional[object] = None

    def __post_init__(self):
        self._model_config = get_agent_model("brain")
        self._worker_manager = WorkerLifecycleManager()

        # Stage 5 : Initialiser la résilience
        retry, circuit, health = create_resilience_from_config(self.config.resilience)
        self._retry_config = retry
        self._health = health

        if not self.config.is_mock_mode():
            self._init_llm()

    @property
    def _mock_mode(self) -> bool:
        """Vérifie dynamiquement si Brain est en mode mock.

        Au lieu de cacher le flag une fois au bootstrap, re-vérifie la config
        à chaque appel. Si la clé API devient disponible après le démarrage
        (ex: vault chargé tardivement), Brain s'auto-initialise.
        """
        if self._force_mock:
            return True
        if self.config.is_mock_mode():
            return True
        # La clé est disponible mais le LLM n'est pas encore initialisé
        if self._llm is None:
            try:
                self._init_llm()
            except Exception:
                self._force_mock = True
                return True
        return False

    @property
    def factory(self) -> WorkerFactory:
        """Accès lazy à la Factory (créée à la demande)."""
        if self._factory is None:
            self._factory = WorkerFactory(config=self.config, memory=self.memory)
        return self._factory

    @property
    def health(self) -> HealthMonitor:
        """Accès au moniteur de santé."""
        if self._health is None:
            _, _, self._health = create_resilience_from_config(self.config.resilience)
        return self._health

    @property
    def worker_manager(self) -> WorkerLifecycleManager:
        """Accès au gestionnaire de cycle de vie des Workers."""
        if self._worker_manager is None:
            self._worker_manager = WorkerLifecycleManager()
        return self._worker_manager

    @property
    def self_patcher(self):
        """Accès lazy au SelfPatcher (Level 3)."""
        if self._self_patcher is None:
            try:
                if self.memory and hasattr(self.memory, 'learning_engine') and self.memory.learning_engine:
                    from neo_core.brain.self_patcher import SelfPatcher
                    self._self_patcher = SelfPatcher(self.config.data_dir, self.memory.learning_engine)
            except Exception as exc:
                logger.debug("SelfPatcher init failed: %s", exc)
        return self._self_patcher

    def get_model_info(self) -> dict:
        """Retourne les infos du modèle utilisé par Brain."""
        return {
            "agent": "Brain",
            "model": self._model_config.model if self._model_config else "unknown",
            "role": "Orchestration, décision, raisonnement",
            "has_llm": not self._mock_mode,
            "auth_method": self._auth_method,
        }

    def get_system_health(self) -> dict:
        """Retourne le rapport de santé complet du système."""
        report = self.health.get_health_report()
        report["brain"] = {
            "mock_mode": self._mock_mode,
            "auth_method": self._auth_method,
            "oauth_mode": self._oauth_mode,
            "model": self._model_config.model if self._model_config else "unknown",
        }
        if self.memory and self.memory.is_initialized:
            stats = self.memory.get_stats()
            report["memory"]["stats"] = stats
            report["memory"]["model"] = self.memory.get_model_info()
            self.health.set_memory_health(True)
        else:
            self.health.set_memory_health(self.memory is not None)

        # Ajouter les infos modèle de chaque agent au rapport
        report["agent_models"] = {
            "brain": self.get_model_info(),
        }
        if self.memory:
            report["agent_models"]["memory"] = self.memory.get_model_info()

        # Worker Lifecycle stats
        report["workers"] = self.worker_manager.get_stats()

        return report

    def get_active_workers(self) -> list[dict]:
        """Retourne les Workers actuellement actifs."""
        return self.worker_manager.get_active_workers()

    def get_worker_history(self, limit: int = 10) -> list[dict]:
        """Retourne l'historique des derniers Workers exécutés."""
        return self.worker_manager.get_history(limit)

    # ─── Initialisation LLM / Auth ──────────────────────────

    def _init_llm(self) -> None:
        """Initialise le LLM avec la meilleure méthode d'auth."""
        try:
            api_key = self.config.llm.api_key
            if is_oauth_token(api_key):
                self._init_oauth(api_key)
            else:
                self._init_langchain(api_key)
        except Exception as e:
            logger.error("Impossible d'initialiser le LLM: %s", e)
            self._force_mock = True
            self._auth_method = "mock"

    def _init_oauth(self, token: str) -> None:
        """Init OAuth avec fallback automatique."""
        converted_key = get_api_key_from_oauth()
        if converted_key:
            logger.info("Clé API convertie depuis OAuth détectée")
            self._init_langchain(converted_key)
            self._auth_method = "converted_api_key"
            return
        self._oauth_mode = True
        self._init_oauth_bearer(token)

    def _init_oauth_bearer(self, token: str) -> None:
        """Init Bearer + beta header (méthode OpenClaw)."""
        import anthropic

        valid_token = get_valid_access_token()
        if not valid_token:
            valid_token = token

        self._anthropic_client = anthropic.AsyncAnthropic(
            api_key="dummy",
            default_headers={
                "Authorization": f"Bearer {valid_token}",
                "anthropic-beta": OAUTH_BETA_HEADER,
            },
        )
        self._current_token = valid_token
        self._auth_method = "oauth_bearer"
        logger.info("Mode OAuth Bearer + beta header activé")

    def _refresh_oauth_client(self) -> bool:
        """Rafraîchit le client OAuth."""
        converted_key = get_api_key_from_oauth()
        if converted_key:
            self._init_langchain(converted_key)
            self._oauth_mode = False
            self._auth_method = "converted_api_key"
            return True
        valid_token = get_valid_access_token()
        if valid_token:
            self._init_oauth_bearer(valid_token)
            return True
        return False

    def _init_langchain(self, api_key: str) -> None:
        """Init LangChain avec clé API classique et modèle dédié Brain."""
        from langchain_anthropic import ChatAnthropic
        self._llm = ChatAnthropic(
            model=self._model_config.model,
            api_key=api_key,
            temperature=self._model_config.temperature,
            max_tokens=self._model_config.max_tokens,
        )
        self._oauth_mode = False
        if not self._auth_method:
            self._auth_method = "langchain"
        logger.info("LLM initialisé : %s", self._model_config.model)

    # ─── Connexions ─────────────────────────────────────────

    def connect_memory(self, memory: MemoryAgent) -> None:
        """Connecte Brain au système mémoire."""
        self.memory = memory
        # Mettre à jour la factory si elle existe déjà
        if self._factory:
            self._factory.memory = memory

    def get_memory_context(
        self, request: str,
        working_context: str = "",
        cached_learning_advice: object | None = None,
    ) -> str:
        """
        Récupère le contexte pertinent depuis Memory.
        Inclut les conseils du LearningEngine si disponibles.

        Optimisation v0.9.2 :
        - working_context pré-chargé par process() (évite double appel)
        - cached_learning_advice réutilisé depuis make_decision (évite double query)
        """
        if not self.memory:
            return "Aucun contexte mémoire disponible."

        # Vider le cache sémantique au début de chaque requête
        try:
            if hasattr(self.memory, '_store') and self.memory._store:
                self.memory._store.clear_semantic_cache()
        except Exception as e:
            logger.debug("Failed to clear semantic cache: %s", e)

        context = self.memory.get_context(request)

        # Ajouter les conseils d'apprentissage au contexte
        # Réutilise le cached_learning_advice si déjà calculé dans make_decision
        try:
            if cached_learning_advice:
                learning_context = cached_learning_advice.to_context_string()
                if learning_context:
                    context += f"\n\n=== Apprentissage ===\n{learning_context}"
            else:
                worker_type = self.factory.classify_task(request)
                if worker_type != WorkerType.GENERIC:
                    advice = self.memory.get_learning_advice(request, worker_type.value)
                    learning_context = advice.to_context_string()
                    if learning_context:
                        context += f"\n\n=== Apprentissage ===\n{learning_context}"
        except Exception as e:
            logger.debug("Impossible de récupérer les conseils d'apprentissage: %s", e)

        # Ajouter le contexte de la mémoire de travail (Working Memory)
        # Réutilise le working_context pré-chargé si disponible
        if working_context:
            context += f"\n\n=== Mémoire de travail ===\n{working_context}"
        else:
            try:
                working_ctx = self.memory.get_working_context()
                if working_ctx:
                    context += f"\n\n=== Mémoire de travail ===\n{working_ctx}"
            except Exception as e:
                logger.debug("Impossible de récupérer la mémoire de travail: %s", e)

        return context

    # ─── Analyse et décision ────────────────────────────────

    # Mots/expressions indiquant une requête dépendante du contexte précédent.
    # Exclut les mots trop génériques ("ça", "cela") qui apparaissent dans
    # des phrases courantes sans lien contextuel ("comment ça va ?").
    _CONTEXT_DEPENDENT_RE = re.compile(
        r"\b(continu|enchaîne|enchaine|la suite|on avance|pareil|"
        r"même chose|fais[- ]le|do it|go ahead|next step|"
        r"le même|la même|les mêmes|refais|recommence)\b",
        re.IGNORECASE,
    )

    # Mots-clés signalant une intention EXPLICITE de projet
    # Très strict : seulement quand l'utilisateur demande clairement un projet multi-étapes
    # Intention de CRÉER un projet — exige un verbe d'action AVANT "projet"
    # Ne matche PAS "projet" tout seul (sinon "pk tu ne met pas dans le projet ?" créerait un projet)
    _EPIC_INTENT_RE = re.compile(
        r"(?:crée|créer|lance|lancer|fais|faire|monte|monter|prépare|préparer|démarre|démarrer|"
        r"nouveau|nouvelle|nouvel|create|start|begin)\s+"
        r"(?:un|une|le|la|mon|ma|moi\s+un|moi\s+une)?\s*"
        r"(?:epic|projet|project|roadmap|feuille de route|plan d[''']action)"
        r"|"
        r"\b(?:multi[- ]?étapes|multi[- ]?step)\b",
        re.IGNORECASE,
    )

    # Mots-clés qui NÉCESSITENT un outil externe (= worker requis)
    # Si ces patterns matchent, Brain ne peut PAS répondre seul
    _NEEDS_TOOL_RE = re.compile(
        r"\b(recherch\w+\s+(?:sur\s+)?(?:internet|le\s+web|google|en\s+ligne)|"
        r"cherch\w+\s+(?:sur\s+)?(?:internet|le\s+web|en\s+ligne)|"
        r"exécute|execute|lance\s+(?:le|un|du)\s+code|run\s+(?:the|this)?\s*code|"
        r"écris\s+(?:un|le|du)\s+(?:fichier|file)|"
        r"lis\s+(?:le|un|du)\s+(?:fichier|file)|"
        r"pip\s+install|npm\s+install|"
        r"scrape|scraping|crawl)\b", re.IGNORECASE,
    )

    # Détection de questions sur un crew actif (bidirectionnel)
    _CREW_QUERY_RE = re.compile(
        r"\b(où en est|avancement|statut|status|briefing|rapport|progrès|progress)\b.*"
        r"\b(crew|epic|projet|mission|équipe)\b"
        r"|\b(crew|epic|projet|mission)\b.*"
        r"\b(où en est|avancement|statut|status|briefing|rapport|progrès)\b",
        re.IGNORECASE,
    )

    # Mots-clés indiquant des sous-tâches implicites (complexité élevée)
    _COMPLEXITY_MULTI_ACTION = re.compile(
        r"\b(puis|ensuite|après|et aussi|également|en plus|aussi|"
        r"then|also|additionally|and then|next)\b", re.IGNORECASE,
    )
    _COMPLEXITY_TECHNICAL = re.compile(
        r"\b(refactor|architect|migrat|optimis|deploy|pipeline|distribu|"
        r"concurren|async|thread|microservic|kubernetes|docker|ci/cd|"
        r"database|schema|api\s+rest|websocket|oauth|encrypt)\b", re.IGNORECASE,
    )
    _COMPLEXITY_ACTION_VERBS = re.compile(
        r"\b(crée|implémente|développe|construi[st]|configure|installe|"
        r"analyse|compare|évalue|audite|corrige|refactorise|"
        r"create|implement|build|setup|configure|install|"
        r"analyze|compare|evaluate|audit|fix|refactor)\b", re.IGNORECASE,
    )

    def analyze_complexity(self, request: str) -> str:
        """
        Analyse la complexité d'une requête via signaux sémantiques.
        Retourne : "simple" | "moderate" | "complex"

        Signaux :
        - Nombre de verbes d'action (sous-tâches implicites)
        - Présence de connecteurs multi-étapes ("puis", "ensuite")
        - Références à des concepts techniques avancés
        - Longueur du message (signal secondaire)

        Note : la complexité n'influe PAS sur la décision direct/worker.
        Elle sert uniquement à calibrer le contexte mémoire chargé.
        """
        word_count = len(request.split())
        score = 0

        # Verbes d'action multiples → sous-tâches implicites
        action_verbs = self._COMPLEXITY_ACTION_VERBS.findall(request)
        score += len(action_verbs)

        # Connecteurs multi-étapes → workflow séquentiel
        multi_steps = self._COMPLEXITY_MULTI_ACTION.findall(request)
        score += len(multi_steps) * 2

        # Concepts techniques avancés
        tech_refs = self._COMPLEXITY_TECHNICAL.findall(request)
        score += len(tech_refs)

        # Longueur comme signal complémentaire
        if word_count > 50:
            score += 4  # Très long → probablement complexe
        elif word_count > 30:
            score += 2
        elif word_count > 12:
            score += 1

        # Décision
        if score >= 4:
            return "complex"
        elif score >= 1:
            return "moderate"
        return "simple"

    # Patterns de requêtes qui sont clairement des conversations/questions simples
    # → JAMAIS de worker/task pour ces patterns
    _SIMPLE_CONVERSATION_RE = re.compile(
        r"^(bonjour|salut|hello|hi|hey|coucou|yo|bonsoir|merci|thanks|"
        r"ok|oui|non|d[''']accord|bien reçu|cool|top|parfait|super|"
        r"comment (ça va|vas[- ]tu)|how are you|ça va|quoi de neuf|"
        r"qu[''']est[- ]ce que tu (peux|sais)|"
        r"/\w+)\s*[\.\!\?]*$",
        re.IGNORECASE,
    )
    _SHORT_QUESTION_RE = re.compile(
        r"^[^\.]{0,80}\??\s*$",
    )

    def make_decision(self, request: str, working_context: str = "") -> BrainDecision:
        """
        Prend une décision stratégique sur la manière de traiter la requête.

        Philosophie v0.9.9 — "Direct by default, task only when needed" :
        Brain répond DIRECTEMENT via Claude Sonnet (avec mémoire + contexte)
        pour la grande majorité des requêtes. Un Worker n'est spawné QUE
        quand un outil externe est EXPLICITEMENT nécessaire (web search,
        exécution de code, accès fichier).

        IMPORTANT: Les conversations simples, questions courtes, salutations,
        et commandes slash ne créent JAMAIS de tâche dans le registre.

        Pipeline :
        0. Conversation simple ? → direct_response (FAST PATH)
        1. Epic explicite ? → delegate_crew
        2. Outil externe nécessaire ? → delegate_worker
        3. Sinon → direct_response (Claude Sonnet répond avec intelligence)

        Le working_context (mémoire de travail) permet de comprendre les requêtes
        courtes qui font référence à une conversation en cours ("continue", "fais pareil").
        """
        # 0. FAST PATH : conversations simples → JAMAIS de worker/task
        if self._is_simple_conversation(request):
            return self._decide_direct(request, "simple")

        # Enrichir la requête pour la classification si contexte de travail disponible.
        classification_input = request
        if working_context and self._CONTEXT_DEPENDENT_RE.search(request):
            classification_input = f"{working_context}\n\nRequête: {request}"

        complexity = self.analyze_complexity(request)

        # 1. Application des patches comportementaux
        worker_type = WorkerType.GENERIC
        patch_overrides = self._apply_behavior_patches(request, worker_type)
        if "override_worker_type" in patch_overrides:
            try:
                worker_type = WorkerType(patch_overrides["override_worker_type"])
            except ValueError:
                pass

        # 2. Intention epic EXPLICITE → delegate_crew
        has_epic_intent = bool(self._EPIC_INTENT_RE.search(request))
        if has_epic_intent:
            worker_type = self.factory.classify_task(classification_input)
            return self._decide_complex_generic(request, worker_type)

        # 3. Besoin d'un outil externe ? → delegate_worker
        needs_tool = bool(self._NEEDS_TOOL_RE.search(request))
        if needs_tool:
            worker_type = self.factory.classify_task(classification_input)
            return self._decide_typed_worker(
                request, complexity, worker_type, patch_overrides
            )

        # 4. Défaut → direct_response (Claude Sonnet + mémoire)
        return self._decide_direct(request, complexity)

    def _is_simple_conversation(self, request: str) -> bool:
        """
        Détecte les requêtes qui sont de simples conversations/questions
        et ne nécessitent JAMAIS un worker ou une task.

        Critères :
        - Salutations, remerciements, acquiescements
        - Commandes slash (/status, /tasks, etc.)
        - Questions très courtes (< 60 chars, pas de verbe d'action fort)
        - Messages de moins de 4 mots
        """
        stripped = request.strip()

        # Commandes slash → toujours direct
        if stripped.startswith("/"):
            return True

        # Conversations simples connues
        if self._SIMPLE_CONVERSATION_RE.match(stripped):
            return True

        # Messages très courts sans verbe d'action → direct
        word_count = len(stripped.split())
        if word_count <= 4 and not self._EPIC_INTENT_RE.search(stripped):
            return True

        # Questions (terminées par ?) sans marqueur d'outil → direct
        if (len(stripped) < 100 and
                stripped.endswith("?") and
                not self._NEEDS_TOOL_RE.search(stripped) and
                not self._EPIC_INTENT_RE.search(stripped)):
            return True

        return False

    def _apply_behavior_patches(self, request: str, worker_type: WorkerType) -> dict:
        """Applique les patches comportementaux du SelfPatcher (Level 3)."""
        if not self.self_patcher:
            return {}
        try:
            return self.self_patcher.apply_patches(
                request, worker_type.value if worker_type != WorkerType.GENERIC else "generic"
            )
        except Exception as exc:
            logger.debug("SelfPatcher apply_patches error: %s", exc)
            return {}

    def _decide_typed_worker(
        self, request: str, complexity: str,
        worker_type: WorkerType, patch_overrides: dict,
    ) -> BrainDecision:
        """
        Décision pour une requête nécessitant un outil externe.

        Appelé UNIQUEMENT quand _NEEDS_TOOL_RE a matché (recherche web,
        exécution de code, accès fichier, etc.).
        """
        # Si la Factory ne trouve pas de type spécifique, utiliser RESEARCHER
        # car _NEEDS_TOOL_RE a déjà validé qu'un outil est nécessaire
        if worker_type == WorkerType.GENERIC:
            worker_type = WorkerType.RESEARCHER

        subtasks = self.factory._basic_decompose(request, worker_type)
        confidence = 0.8

        advice = self._consult_learning(request, worker_type.value)
        if advice:
            confidence = max(0.1, min(1.0, confidence + advice.confidence_adjustment))
            worker_type, subtasks, reasoning = self._apply_learning_advice(
                request, worker_type, subtasks, advice
            )
        else:
            reasoning = f"Outil externe requis → Worker {worker_type.value}"

        metadata = dict(patch_overrides) if patch_overrides else {}
        if advice:
            metadata["_cached_learning_advice"] = advice
        return BrainDecision(
            action="delegate_worker",
            subtasks=subtasks,
            confidence=confidence,
            worker_type=worker_type.value,
            reasoning=reasoning,
            metadata=metadata,
        )

    def _decide_complex_generic(
        self, request: str, worker_type: WorkerType,
    ) -> BrainDecision:
        """Décision pour une requête complexe sans type spécifique.

        Quand une intention epic est détectée, force delegate_crew
        même si la décomposition heuristique ne donne que peu de sous-tâches.
        """
        has_epic_intent = bool(self._EPIC_INTENT_RE.search(request))
        subtasks = self._decompose_task(request)
        confidence = 0.5

        advice = self._consult_learning(request, "generic")
        reasoning = f"Tâche complexe → Worker {worker_type.value}"

        if advice and advice.relevant_skills:
            best_skill = advice.relevant_skills[0]
            if best_skill.worker_type != "generic":
                try:
                    alt_type = WorkerType(best_skill.worker_type)
                    worker_type = alt_type
                    subtasks = self.factory._basic_decompose(request, alt_type)
                    reasoning = (
                        f"Tâche complexe → substitué par {alt_type.value} "
                        f"(compétence acquise: {best_skill.name})"
                    )
                    confidence = 0.7
                except ValueError as e:
                    logger.debug("Invalid worker type for skill %s: %s", best_skill.worker_type, e)

        # Si intention epic ou 3+ sous-tâches → delegate_crew (Epic)
        metadata = {"_cached_learning_advice": advice} if advice else {}
        if has_epic_intent:
            metadata["epic_intent"] = True
        if has_epic_intent or len(subtasks) >= 3:
            return BrainDecision(
                action="delegate_crew",
                subtasks=subtasks,
                confidence=confidence,
                worker_type=worker_type.value,
                reasoning=f"Projet ({len(subtasks)} étapes) → Crew",
                metadata=metadata,
            )

        return BrainDecision(
            action="delegate_worker",
            subtasks=subtasks,
            confidence=confidence,
            worker_type=worker_type.value,
            reasoning=reasoning,
            metadata=metadata,
        )

    def _decide_direct(self, request: str, complexity: str) -> BrainDecision:
        """
        Décision par défaut : réponse directe via Claude Sonnet.

        Brain utilise toute l'intelligence de Claude avec le contexte
        mémoire de Neo pour répondre directement. C'est le chemin
        principal pour la majorité des requêtes.
        """
        advice = self._consult_learning(request, "generic")

        return BrainDecision(
            action="direct_response",
            subtasks=[],
            confidence=0.9 if complexity == "simple" else 0.8,
            reasoning=f"Réponse directe Claude Sonnet (complexité: {complexity})",
            metadata={"_cached_learning_advice": advice} if advice else {},
        )

    def _apply_learning_advice(
        self, request: str, worker_type: WorkerType,
        subtasks: list[str], advice: object,
    ) -> tuple[WorkerType, list[str], str]:
        """Ajuste le worker_type et les subtasks selon les conseils du LearningEngine."""
        reasoning_parts = [f"Type {worker_type.value} détecté"]

        if advice.recommended_worker and advice.recommended_worker != worker_type.value:
            try:
                alt_type = WorkerType(advice.recommended_worker)
                worker_type = alt_type
                subtasks = self.factory._basic_decompose(request, alt_type)
                reasoning_parts.append(f"→ substitué par {alt_type.value} (historique)")
            except ValueError as e:
                logger.debug("Invalid recommended worker %s: %s", advice.recommended_worker, e)

        if advice.warnings:
            reasoning_parts.append(f"⚠ {'; '.join(advice.warnings[:2])}")

        return worker_type, subtasks, " | ".join(reasoning_parts)

    def _consult_learning(self, request: str, worker_type: str) -> object | None:
        """
        Consulte le LearningEngine de Memory pour obtenir des conseils.
        Retourne un LearningAdvice ou None si Memory n'est pas disponible.
        """
        if not self.memory or not self.memory.is_initialized:
            return None

        try:
            return self.memory.get_learning_advice(request, worker_type)
        except Exception as e:
            logger.debug("Impossible de consulter l'apprentissage pour %s: %s", worker_type, e)
            return None

    def _decompose_task(self, request: str) -> list[str]:
        """
        Décompose une tâche complexe en sous-tâches.
        Utilise les heuristiques de la Factory (compatible mock).
        """
        worker_type = self.factory.classify_task(request)
        return self.factory._basic_decompose(request, worker_type)

    async def _decompose_task_with_llm(self, request: str,
                                        memory_context: str) -> TaskAnalysis:
        """
        Décomposition LLM-powered d'une tâche.
        Fallback sur les heuristiques si le LLM échoue.
        """
        if self._mock_mode:
            return self.factory.analyze_task(request)

        try:
            prompt = DECOMPOSE_PROMPT.format(
                request=request,
                memory_context=memory_context[:500],
            )

            response_text = await self._raw_llm_call(prompt)

            # Parser le JSON
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

            data = json.loads(cleaned)

            worker_type_str = data.get("worker_type", "generic")
            try:
                worker_type = WorkerType(worker_type_str)
            except ValueError:
                worker_type = WorkerType.GENERIC

            return TaskAnalysis(
                worker_type=worker_type,
                primary_task=request,
                subtasks=data.get("subtasks", [request]),
                required_tools=[
                    getattr(t, "name", str(t))
                    for t in self.factory._get_tools_for_type(worker_type)
                ] if hasattr(self.factory, '_get_tools_for_type') else [],
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.7)),
            )

        except Exception as e:
            logger.debug("Décomposition LLM échouée, utilisation des heuristiques: %s", e)
            return self.factory.analyze_task(request)

    # ─── Pipeline principal ─────────────────────────────────

    async def process(self, request: str,
                      conversation_history: list[BaseMessage] | None = None,
                      original_request: str = "") -> str:
        """
        Pipeline principal de Brain.

        Stage 5 :
        1. Vérifie la santé du système (circuit breaker)
        2. Récupère le contexte mémoire
        3. Prend une décision (direct / worker)
        4. Si direct → répond via LLM (avec retry)
        5. Si worker → crée un Worker, exécute, apprend du résultat
        """
        # Stage 11: Input validation
        try:
            request = validate_message(request)
        except ValidationError as e:
            logger.warning("Validation du message échouée: %s", e)
            return f"[Brain Erreur] Message invalide: {str(e)[:200]}"

        # ── Optimisation v0.9.2 : working memory AVANT la décision ──
        # La working memory (topic courant, actions en attente) est en RAM (~0ms),
        # elle permet à make_decision de comprendre les requêtes courtes
        # ("continue", "fais pareil") dans leur contexte.
        working_context = ""
        if self.memory and self.memory.is_initialized:
            try:
                working_context = self.memory.get_working_context()
            except Exception as e:
                logger.debug("Failed to load working context for decision: %s", e)

        decision = self.make_decision(request, working_context=working_context)

        # Contexte mémoire chargé pour TOUTES les réponses
        # direct_response est maintenant le chemin principal → toujours enrichir
        needs_context = True
        # Réutilise le working_context et le learning advice déjà calculés
        cached_advice = decision.metadata.pop("_cached_learning_advice", None)
        memory_context = (
            self.get_memory_context(
                request,
                working_context=working_context,
                cached_learning_advice=cached_advice,
            )
            if needs_context
            else ""
        )

        # ── Intercept crew queries (bidirectionnel Brain ↔ Crew) ──
        crew_epic_id = self._detect_crew_query(request)
        if crew_epic_id:
            try:
                executor = CrewExecutor(brain=self)
                briefing = executor.get_briefing(crew_epic_id)
                memory_context += f"\n\n=== CREW ACTIF ===\n{briefing}"
                # Forcer direct_response pour que Brain réponde avec le contexte crew
                decision = BrainDecision(
                    action="direct_response",
                    confidence=0.9,
                    reasoning=f"Question sur crew actif {crew_epic_id[:8]}",
                )
            except Exception as e:
                logger.debug("Injection contexte crew échouée: %s", e)

        # Garder la requête originale (avant reformulation Vox)
        _original_req = original_request or request

        if self._mock_mode:
            if decision.action == "delegate_crew" and decision.subtasks:
                return await self._execute_as_epic(request, decision, memory_context, _original_req)

            if decision.action == "delegate_worker" and decision.worker_type:
                return await self._execute_with_worker(request, decision, memory_context)

            complexity = self.analyze_complexity(request)
            return self._mock_response(request, complexity, memory_context)

        # Stage 5 : Vérifier le circuit breaker
        if not self.health.can_make_api_call():
            return (
                "[Brain] Le système est temporairement indisponible "
                "(trop d'erreurs consécutives). Réessayez dans quelques instants."
            )

        try:
            if decision.action == "delegate_crew" and decision.subtasks:
                return await self._execute_as_epic(request, decision, memory_context, _original_req)

            if decision.action == "delegate_worker" and decision.worker_type:
                # Optimisation v0.9.1 : heuristiques SEULES, plus d'appel LLM redondant
                # L'ancien _decompose_task_with_llm() faisait un appel Sonnet de 2-5s
                # alors que _basic_decompose() dans make_decision l'avait déjà fait.
                analysis = self.factory.analyze_task(request)

                return await self._execute_with_worker(request, decision, memory_context, analysis)

            # Réponse directe
            if self._oauth_mode:
                return await self._oauth_response(request, memory_context, conversation_history)
            else:
                return await self._llm_response(request, memory_context, conversation_history)

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            # Gestion erreur auth OAuth
            if self._oauth_mode and ("authentication" in error_msg.lower()
                                     or "unauthorized" in error_msg.lower()
                                     or "401" in error_msg):
                logger.warning("Erreur auth OAuth, tentative de fallback...")

                converted_key = get_api_key_from_oauth()
                if converted_key:
                    logger.info("Conversion OAuth → API key réussie, retry...")
                    self._init_langchain(converted_key)
                    self._oauth_mode = False
                    self._auth_method = "converted_api_key"
                    try:
                        return await self._llm_response(request, memory_context, conversation_history)
                    except Exception as retry_e:
                        logger.error("Erreur après conversion OAuth: %s", type(retry_e).__name__)
                        return f"[Brain Erreur] Après conversion: {type(retry_e).__name__}: {str(retry_e)[:300]}"

                if self._refresh_oauth_client():
                    try:
                        if self._oauth_mode:
                            return await self._oauth_response(request, memory_context, conversation_history)
                        else:
                            return await self._llm_response(request, memory_context, conversation_history)
                    except Exception as retry_e:
                        logger.error("Erreur après refresh OAuth: %s", type(retry_e).__name__)
                        return f"[Brain Erreur] Après refresh: {type(retry_e).__name__}: {str(retry_e)[:300]}"

            logger.error("Erreur lors du traitement de la requête: %s: %s", error_type, error_msg[:200])
            return f"[Brain Erreur] {error_type}: {error_msg[:500]}"

    async def _execute_with_worker(self, request: str, decision: BrainDecision,
                                    memory_context: str,
                                    analysis: TaskAnalysis | None = None,
                                    existing_task_id: str | None = None) -> str:
        """
        Crée, exécute et détruit un Worker pour une tâche.
        Avec retry persistant (max 3 tentatives) guidé par Memory.

        Args:
            existing_task_id: Si fourni, utilise cette tâche existante dans le
                registre au lieu d'en créer une nouvelle (ex: appelé par heartbeat).

        Cycle de vie garanti par tentative :
        1. Factory crée le Worker
        2. Worker enregistré dans le WorkerLifecycleManager
        3. execute() → Memory récupère l'apprentissage
        4. Brain._learn_from_result() → LearningEngine apprend
        5. Si échec → améliore la stratégie via Memory et retente
        6. Worker.cleanup() → ressources libérées
        """
        max_attempts = 3
        errors_so_far = []

        # Enregistrer la tâche dans le TaskRegistry (sauf si déjà existante)
        task_record = None
        if existing_task_id:
            # Tâche déjà dans le registre (appel depuis heartbeat)
            if self.memory and self.memory.is_initialized:
                try:
                    task_record = self.memory.task_registry.get_task(existing_task_id)
                except Exception:
                    pass
        elif self.memory and self.memory.is_initialized:
            try:
                task_record = self.memory.create_task(
                    description=request[:200],
                    worker_type=decision.worker_type or "generic",
                )
            except Exception as e:
                logger.debug("Impossible de créer le TaskRecord: %s", e)

        for attempt in range(1, max_attempts + 1):
            # Si retry → améliorer la stratégie via Memory
            if attempt > 1:
                decision, analysis = self._improve_strategy(
                    request, decision, errors_so_far, attempt
                )
                logger.info(
                    "[Brain] Retry %d/%d — stratégie: %s (%s)",
                    attempt, max_attempts, decision.worker_type, decision.reasoning,
                )

            try:
                worker_type = WorkerType(decision.worker_type)
            except (ValueError, TypeError):
                worker_type = WorkerType.GENERIC

            if analysis:
                worker = self.factory.create_worker(analysis)
            else:
                worker = self.factory.create_worker_for_type(
                    worker_type=worker_type,
                    task=request,
                    subtasks=decision.subtasks,
                )

            # Stage 5 : Passer le health monitor au Worker
            worker.health_monitor = self._health

            # Mettre à jour la tâche en "in_progress"
            if task_record:
                try:
                    self.memory.update_task_status(task_record.id, "in_progress")
                except Exception as e:
                    logger.debug("Impossible de mettre à jour le statut de la tâche: %s", e)

            # ── Lifecycle géré par context manager ──
            async with worker:
                self.worker_manager.register(worker)

                try:
                    result = await worker.execute()
                    await self._learn_from_result(request, decision, result)

                    if result.success:
                        # Marquer la tâche comme terminée
                        if task_record:
                            try:
                                self.memory.update_task_status(
                                    task_record.id, "done",
                                    result=result.output[:500],
                                )
                            except Exception as e:
                                logger.debug("Impossible de marquer la tâche comme terminée: %s", e)
                        return result.output

                    # Échec → collecter l'erreur pour le prochain essai
                    errors_so_far.append({
                        "attempt": attempt,
                        "worker_type": decision.worker_type,
                        "error": result.output[:200],
                        "errors": result.errors or [],
                    })

                except Exception as e:
                    errors_so_far.append({
                        "attempt": attempt,
                        "worker_type": decision.worker_type,
                        "error": f"{type(e).__name__}: {str(e)[:200]}",
                        "errors": [str(e)],
                    })

                finally:
                    self.worker_manager.unregister(worker)

        # Toutes les tentatives épuisées
        if task_record:
            try:
                last_err = errors_so_far[-1]["error"] if errors_so_far else "inconnu"
                self.memory.update_task_status(
                    task_record.id, "failed",
                    result=f"Échec après {max_attempts} tentatives: {last_err}",
                )
            except Exception as e:
                logger.debug("Impossible de marquer la tâche comme échouée: %s", e)

        last_error = errors_so_far[-1]["error"] if errors_so_far else "erreur inconnue"
        return (
            f"[Brain] Échec après {max_attempts} tentatives. "
            f"Dernière erreur: {last_error}"
        )

    async def _decompose_crew_with_llm(
        self, request: str, memory_context: str,
    ) -> list[CrewStep]:
        """
        Utilise Claude Sonnet pour décomposer une requête epic en
        étapes de crew avec un worker_type par étape.

        Retourne une liste de CrewStep. Fallback sur heuristiques.
        """
        if self._mock_mode:
            subtasks = self._decompose_task(request)
            return [
                CrewStep(index=i, description=st, worker_type=WorkerType.GENERIC)
                for i, st in enumerate(subtasks)
            ]

        decompose_prompt = (
            f"L'utilisateur veut créer un projet :\n"
            f"\"{request}\"\n\n"
            f"Contexte mémoire :\n{memory_context[:500]}\n\n"
            f"Décompose ce projet en 3 à 6 étapes CONCRÈTES et OPÉRATIONNELLES.\n"
            f"Chaque étape doit être une ACTION PRÉCISE qui fait avancer le projet vers son objectif.\n\n"
            f"Types de worker disponibles :\n"
            f"- researcher : collecte d'infos web, veille, recherche de données\n"
            f"- analyst : analyse de données, calculs, modélisation, stratégie\n"
            f"- coder : écriture de code, scripts, automatisation\n"
            f"- writer : rédaction de documents, rapports, synthèses\n"
            f"- summarizer : résumés, briefings\n"
            f"- generic : tâches mixtes\n\n"
            f"Réponds en JSON strict (array) :\n"
            f"[\n"
            f'  {{"description": "Action concrète et spécifique...", "worker_type": "researcher"}},\n'
            f'  {{"description": "Analyser X pour déterminer Y...", "worker_type": "analyst"}}\n'
            f"]\n\n"
            f"RÈGLES CRUCIALES :\n"
            f"- Les descriptions doivent être SPÉCIFIQUES au projet, pas génériques\n"
            f"- VARIE les worker_type selon l'étape — PAS que des researcher\n"
            f"- Chaque étape bâtit sur les résultats de la précédente\n"
            f"- Adapte les étapes à l'OBJECTIF RÉEL du projet (ex: si c'est du trading → analyse cotes, "
            f"modèle de mise, simulation bankroll... PAS juste 'rechercher sur le trading')\n"
            f"- Réponds UNIQUEMENT avec le JSON, rien d'autre."
        )

        try:
            response = await self._raw_llm_call(decompose_prompt)
            # Nettoyer la réponse (enlever ```json ... ``` si présent)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

            data = json.loads(cleaned)
            if isinstance(data, list) and len(data) >= 2:
                steps = []
                for i, item in enumerate(data):
                    desc = item.get("description", "")
                    wt_str = item.get("worker_type", "generic")
                    try:
                        wt = WorkerType(wt_str)
                    except ValueError:
                        wt = WorkerType.GENERIC
                    if desc:
                        steps.append(CrewStep(index=i, description=desc, worker_type=wt))
                if len(steps) >= 2:
                    return steps
        except Exception as e:
            logger.debug("Décomposition crew JSON échouée: %s", e)

        # Fallback : décomposition texte classique
        subtasks = self._decompose_task(request)
        return [
            CrewStep(index=i, description=st, worker_type=WorkerType.GENERIC)
            for i, st in enumerate(subtasks)
        ]

    def _extract_epic_name_and_subject(self, request: str) -> tuple[str, str]:
        """
        Extrait le NOM du projet et le SUJET/DESCRIPTION.

        Le nom est extrait des guillemets, quotes, ou après "appelé/nommé".
        Le sujet est le reste de la requête nettoyée.

        Exemples :
        - "crée un projet 'Smash Gang', objectif : devenir rentable"
          → name="Smash Gang", subject="devenir rentable en pariant..."
        - "lance un projet roadmap pour la refonte du site"
          → name="", subject="refonte du site"

        Returns:
            (name, subject) — name peut être vide si non détecté
        """
        name = ""

        # 1. Chercher un nom entre guillemets/quotes (tous types Unicode)
        # Couvre : 'x' "x" 'x' 'x' "x" "x" «x» ‹x›
        name_match = re.search(r"""['"'\u2018\u2019\u201C\u201D«»‹›]([^'"'\u2018\u2019\u201C\u201D«»‹›]+)['"'\u2018\u2019\u201C\u201D«»‹›]""", request)
        if name_match:
            name = name_match.group(1).strip()

        # 2. Sinon chercher après "appelé/nommé/intitulé"
        if not name:
            name_match = re.search(
                r"(?:appel[ée]|nomm[ée]|intitul[ée])\s+(.+?)(?:[,.\n]|$)",
                request, re.IGNORECASE,
            )
            if name_match:
                name = name_match.group(1).strip()[:60]

        # 3. Extraire le sujet (description) — retirer les préfixes d'intention
        subject = re.sub(
            r"^(?:crée|créer|lance|lancer|fais|faire|monte|prépare)\s+"
            r"(?:un|une|le|la|mon|ma|moi\s+un|moi\s+une)\s+"
            r"(?:epic|epics|projet|project|roadmap|feuille de route|plan)\s+"
            r"(?:pour|sur|de|du|des|d['']\s*)?\s*",
            "", request, flags=re.IGNORECASE,
        ).strip()

        # Retirer le nom entre quotes du sujet pour ne garder que la description
        if name and name_match:
            subject = subject.replace(name_match.group(0), "").strip(" ,;:-")

        # Si le nettoyage a tout supprimé, garder la requête originale
        if len(subject) < 5:
            subject = request

        return name[:100], subject[:1000]

    async def _execute_as_epic(self, request: str, decision: BrainDecision,
                               memory_context: str,
                               original_request: str = "") -> str:
        """
        Crée un Epic et lance un Crew persistant (v2.0).

        Pipeline :
        1. Extraction du sujet réel
        2. Décomposition intelligente avec worker_type par étape (JSON LLM)
        3. Création de l'Epic dans le registre
        4. Création du CrewState persistant en Memory
        5. Exécution IMMÉDIATE de la première étape (feedback rapide)
        6. Les étapes suivantes seront avancées par le heartbeat

        L'utilisateur reçoit un briefing immédiat + la première étape.
        Le heartbeat avance ensuite le crew d'une étape par pulse.
        """
        # 0. Extraire le nom et le sujet
        epic_name, epic_subject = self._extract_epic_name_and_subject(request)

        # 1. Décomposer en étapes crew avec worker_types
        crew_steps = await self._decompose_crew_with_llm(request, memory_context)

        # 2. Créer l'Epic dans le registre
        epic = None
        epic_id = "unknown"
        if self.memory and self.memory.is_initialized:
            try:
                subtask_tuples = [
                    (step.description, step.worker_type.value)
                    for step in crew_steps
                ]
                epic = self.memory.create_epic(
                    description=epic_subject,
                    subtask_descriptions=subtask_tuples,
                    strategy=decision.reasoning,
                    name=epic_name,
                )
                epic_id = epic.id
                logger.info(
                    "Crew Epic créé: %s (%s) avec %d étapes",
                    epic_id[:8], epic_subject[:50], len(crew_steps),
                )
            except Exception as e:
                logger.debug("Impossible de créer l'Epic: %s", e)

        # 3. Créer le CrewState persistant
        try:
            executor = CrewExecutor(brain=self)
            executor.set_event_callback(self._handle_crew_event)
            executor.create_crew_state(
                epic_id=epic_id,
                epic_subject=epic_subject,
                steps=crew_steps,
                memory_context=memory_context,
                original_request=original_request or request,
            )

            # 4. Exécuter la PREMIÈRE étape immédiatement (feedback rapide)
            first_event = await executor.advance_one_step(epic_id)

            # 5. Mettre à jour le statut de l'Epic → in_progress
            if epic and self.memory and self.memory.is_initialized:
                try:
                    self.memory.update_epic_status(epic_id, "in_progress")
                except Exception as e:
                    logger.debug("Impossible de mettre à jour l'Epic: %s", e)

            # 6. Retourner le briefing avec le résultat de la première étape
            briefing = executor.get_briefing(epic_id)

            # Si le crew est déjà terminé (1 seule étape), retourner directement
            if first_event.event_type == "crew_done":
                return first_event.data.get("synthesis", first_event.message)

            return (
                f"{first_event.message}\n\n"
                f"Le crew est en marche. Les prochaines étapes seront "
                f"exécutées automatiquement par le heartbeat.\n\n{briefing}"
            )

        except Exception as e:
            logger.error("Crew creation/execution failed: %s", e)
            if epic and self.memory and self.memory.is_initialized:
                try:
                    self.memory.update_epic_status(epic_id, "failed")
                except Exception:
                    pass
            return f"[Projet échoué — {epic_subject}] {type(e).__name__}: {str(e)[:300]}"

    # ─── Communication bidirectionnelle Crew ↔ Brain ────

    def _detect_crew_query(self, request: str) -> Optional[str]:
        """
        Détecte si la requête concerne un crew actif.
        Retourne l'epic_id ou None.
        """
        if not self._CREW_QUERY_RE.search(request):
            return None
        try:
            executor = CrewExecutor(brain=self)
            active_crews = executor.list_active_crews()
            if not active_crews:
                return None
            if len(active_crews) == 1:
                return active_crews[0].epic_id
            # Plusieurs crews → chercher par similarité dans le sujet
            request_lower = request.lower()
            for crew in active_crews:
                subject_words = crew.epic_subject.lower().split()[:5]
                if any(word in request_lower for word in subject_words if len(word) > 3):
                    return crew.epic_id
            # Fallback: le plus récent
            return active_crews[0].epic_id
        except Exception as e:
            logger.debug("Détection crew query échouée: %s", e)
            return None

    def _handle_crew_event(self, event: CrewEvent) -> None:
        """
        Reçoit les notifications proactives des crews.

        Stocke en mémoire (pour que Brain retrouve l'info sémantiquement)
        et notifie via Telegram.
        """
        # Stocker en mémoire
        if self.memory and self.memory.is_initialized:
            try:
                self.memory.store_memory(
                    content=f"[Crew Event] {event.message}",
                    source=f"crew_event:{event.crew_id}",
                    tags=["crew_event", f"crew:{event.crew_id}", event.event_type],
                    importance=0.7 if event.event_type in ("crew_done", "insight") else 0.5,
                )
            except Exception as e:
                logger.debug("Stockage crew event échoué: %s", e)

        # Notifier via Telegram
        try:
            from neo_core.infra.registry import core_registry
            core_registry.send_telegram(f"🔔 Crew: {event.message}")
        except Exception:
            pass

    def _improve_strategy(
        self,
        request: str,
        current_decision: BrainDecision,
        errors_so_far: list[dict],
        attempt: int,
    ) -> tuple[BrainDecision, TaskAnalysis | None]:
        """
        Améliore la stratégie d'exécution en se basant sur les erreurs passées
        et les conseils de Memory.

        Stratégie progressive :
        - Tentative 2 : changer de worker_type si Memory recommande un alternatif
        - Tentative 3 : simplifier la requête (moins de subtasks)
        """
        new_decision = BrainDecision(
            action=current_decision.action,
            subtasks=list(current_decision.subtasks),
            confidence=current_decision.confidence * 0.8,
            worker_type=current_decision.worker_type,
            reasoning=current_decision.reasoning,
        )
        new_analysis = None

        # Consulter Memory pour des conseils de retry
        retry_advice = None
        if self.memory and self.memory.is_initialized and self.memory.learning:
            try:
                previous_errors = []
                for e in errors_so_far:
                    previous_errors.extend(e.get("errors", []))
                retry_advice = self.memory.learning.get_retry_advice(
                    request,
                    current_decision.worker_type or "generic",
                    previous_errors,
                )
            except Exception as e:
                logger.debug("Impossible de consulter les conseils de retry: %s", e)

        if attempt == 2:
            # Stratégie 2 : Changer de worker_type si recommandé
            if retry_advice and retry_advice.get("recommended_worker"):
                alt_worker = retry_advice["recommended_worker"]
                try:
                    alt_type = WorkerType(alt_worker)
                    new_decision.worker_type = alt_worker
                    new_decision.subtasks = self.factory._basic_decompose(request, alt_type)
                    new_decision.reasoning = (
                        f"Retry: changement {current_decision.worker_type} → {alt_worker} "
                        f"(conseil Memory)"
                    )
                except ValueError as e:
                    logger.debug("Type de worker recommandé invalide %s: %s", alt_worker, e)

            if new_decision.worker_type == current_decision.worker_type:
                # Pas de recommandation → simplifier la requête
                if new_decision.subtasks and len(new_decision.subtasks) > 1:
                    # Garder uniquement la tâche principale
                    new_decision.subtasks = [new_decision.subtasks[0]]
                    new_decision.reasoning = "Retry: simplification (1 seule sous-tâche)"

        elif attempt == 3:
            # Stratégie 3 : Worker generic avec décomposition minimale
            new_decision.worker_type = "generic"
            new_decision.subtasks = [request[:200]]
            new_decision.reasoning = (
                "Retry final: worker generic, requête simplifiée"
            )

        return new_decision, new_analysis

    async def _learn_from_result(self, request: str, decision: BrainDecision,
                                  result: WorkerResult) -> None:
        """
        Apprentissage à partir du résultat d'un Worker.

        Boucle fermée :
        - Enregistre dans le LearningEngine (patterns d'erreur, compétences)
        - Stocke aussi un résumé en mémoire classique pour le contexte
        """
        if not self.memory:
            return

        try:
            # 1. Enregistrer dans le LearningEngine (boucle fermée)
            self.memory.record_execution_result(
                request=request,
                worker_type=result.worker_type,
                success=result.success,
                execution_time=result.execution_time,
                errors=result.errors,
                output=result.output[:500] if result.success else "",
            )

            # 2. Stocker aussi en mémoire classique pour le contexte
            tags = [
                "brain_learning",
                f"worker_type:{result.worker_type}",
                "success" if result.success else "failure",
                f"decision:{decision.action}",
            ]

            content = (
                f"Apprentissage Brain — {result.worker_type}\n"
                f"Requête: {request[:200]}\n"
                f"Décision: {decision.action} (confiance: {decision.confidence:.1f})\n"
                f"Résultat: {'Succès' if result.success else 'Échec'}\n"
                f"Temps: {result.execution_time:.1f}s"
            )

            if result.errors:
                content += f"\nErreurs: {'; '.join(result.errors[:3])}"

            importance = 0.5 if result.success else 0.8

            self.memory.store_memory(
                content=content,
                source="brain_learning",
                tags=tags,
                importance=importance,
                metadata={
                    "worker_type": result.worker_type,
                    "decision_action": decision.action,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "confidence": decision.confidence,
                },
            )
            # 3. Tracker l'usage des outils auto-générés (Level 4)
            if result.worker_type and result.worker_type.startswith("auto_"):
                try:
                    from neo_core.brain.tools.tool_generator import ToolGenerator
                    # Accès via le heartbeat ou lazy init
                    if hasattr(self, '_tool_generator') and self._tool_generator:
                        self._tool_generator.track_usage(
                            result.worker_type,
                            result.success,
                            result.execution_time,
                        )
                except Exception as e:
                    logger.debug("Best-effort tool tracking failed: %s", e)
        except Exception as e:
            logger.debug("Impossible d'enregistrer l'apprentissage: %s", e)

    # ─── Appels LLM ────────────────────────────────────────

    async def _raw_llm_call(self, prompt: str) -> str:
        """
        Appel LLM brut (sans historique ni system prompt Brain).

        Stage 6 : Route via le système multi-provider.
        Fallback automatique vers Anthropic direct.
        """
        try:
            from neo_core.brain.providers.router import route_chat

            response = await route_chat(
                agent_name="brain",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
            )

            if self._health:
                self._health.record_api_call(
                    success=not response.text.startswith("[Erreur")
                )

            if response.text and not response.text.startswith("[Erreur"):
                return response.text

            raise Exception(response.text)

        except Exception as e:
            # Fallback LangChain legacy
            logger.debug("Appel LLM via router échoué, utilisation de LangChain: %s", e)
            if self._llm:
                result = await self._llm.ainvoke(prompt)
                if self._health:
                    self._health.record_api_call(success=True)
                return result.content
            raise

    async def _oauth_response(self, request: str, memory_context: str,
                              conversation_history: list[BaseMessage] | None = None) -> str:
        """
        Génère une réponse Brain complète (avec historique + system prompt).

        Stage 6 : Route via le système multi-provider.
        Compatible OAuth, API key, et providers alternatifs.
        """
        messages = []
        if conversation_history:
            for msg in conversation_history:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})
        messages.append({"role": "user", "content": request})

        from datetime import datetime
        now = datetime.now()

        # Stage 9 — Injection du contexte utilisateur
        user_context = ""
        if self.memory and self.memory.persona_engine and self.memory.persona_engine.is_initialized:
            try:
                user_context = self.memory.persona_engine.get_brain_injection()
            except Exception as e:
                logger.debug("Impossible de récupérer le contexte utilisateur: %s", e)

        system_prompt = BRAIN_SYSTEM_PROMPT.format(
            memory_context=memory_context,
            current_date=now.strftime("%A %d %B %Y"),
            current_time=now.strftime("%H:%M"),
            user_context=user_context,
        )

        try:
            from neo_core.brain.providers.router import route_chat

            response = await route_chat(
                agent_name="brain",
                messages=messages,
                system=system_prompt,
                max_tokens=self._model_config.max_tokens,
                temperature=self._model_config.temperature,
            )

            if self._health:
                self._health.record_api_call(
                    success=not response.text.startswith("[Erreur")
                )

            if response.text and not response.text.startswith("[Erreur"):
                return response.text

            return f"[Brain Erreur] {response.text}"

        except Exception as e:
            logger.error("Erreur dans _oauth_response: %s: %s", type(e).__name__, str(e)[:200])
            if self._health:
                self._health.record_api_call(success=False)
            return f"[Brain Erreur] {type(e).__name__}: {str(e)[:200]}"

    async def _llm_response(self, request: str, memory_context: str,
                            conversation_history: list[BaseMessage] | None = None) -> str:
        """Génère une réponse via LangChain."""
        from datetime import datetime
        now = datetime.now()

        # Stage 9 — Injection du contexte utilisateur
        user_context = ""
        if self.memory and self.memory.persona_engine and self.memory.persona_engine.is_initialized:
            try:
                user_context = self.memory.persona_engine.get_brain_injection()
            except Exception as e:
                logger.debug("Impossible de récupérer le contexte utilisateur: %s", e)

        prompt = ChatPromptTemplate.from_messages([
            ("system", BRAIN_SYSTEM_PROMPT),
            MessagesPlaceholder("conversation_history", optional=True),
            ("human", "{request}"),
        ])
        chain = prompt | self._llm
        result = await chain.ainvoke({
            "memory_context": memory_context,
            "user_context": user_context,
            "current_date": now.strftime("%A %d %B %Y"),
            "current_time": now.strftime("%H:%M"),
            "conversation_history": conversation_history or [],
            "request": request,
        })
        if self._health:
            self._health.record_api_call(success=True)
        return result.content

    def _mock_response(self, request: str, complexity: str, context: str) -> str:
        """Réponse mock pour les tests sans clé API."""
        return (
            f"[Brain Mock] Requête reçue (complexité: {complexity}). "
            f"Contexte: {context[:100]}. "
            f"Analyse de: '{request[:80]}...'" if len(request) > 80 else
            f"[Brain Mock] Requête reçue (complexité: {complexity}). "
            f"Contexte: {context[:100]}. "
            f"Analyse de: '{request}'"
        )
