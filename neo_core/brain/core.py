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
        self._action_result_callback = None  # Callable[[str], None] | None

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

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """Parse un JSON depuis une réponse LLM (gère les backticks markdown)."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Retirer la ligne ```json ou ``` initiale
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        return json.loads(cleaned)

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

        # Injecter le contexte du TaskRegistry (projets & tâches actifs)
        # Permet au LLM Brain de connaître les projets quand il génère sa réponse
        try:
            registry_ctx = self._build_task_registry_context(request)
            if registry_ctx:
                context += f"\n\n=== Projets & Tâches ===\n{registry_ctx}"
        except Exception as e:
            logger.debug("Impossible de récupérer le contexte TaskRegistry: %s", e)

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
        r"mets?\s+en\s+place|construi[st]|build|setup|met[st]?\s+en\s+place|"
        r"nouveau|nouvelle|nouvel|create|start|begin)\s+"
        r"(?:un|une|le|la|mon|ma|moi\s+un|moi\s+une|nous\s+un|nous\s+une)?\s*"
        r"(?:epic|projet|project|roadmap|feuille de route|plan d[''']action|"
        r"système|systeme|system|truc|outil|tool|bot|app|programme|stratégie|strategie|strategy)"
        r"|"
        r"(?:j['''](?:aimerais?|veu[xt]|voudrais?|souhaite)\s+(?:qu[''']on\s+)?(?:crée|monte|lance|fasse|prépare|construise|mette en place))"
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

    # Filet de sécurité : verbes d'action clairs qui ne devraient PAS rester en "conversation"
    # Si le classifier dit "conversation" mais que la requête contient ces verbes → promote
    _LOOKS_LIKE_ACTION_RE = re.compile(
        r"(?:^|\b)(?:"
        # Verbes d'action FR
        r"cherche|trouve|recherche|regarde|vérifie|check|analyse|calcule|"
        r"écris|rédige|traduis|résume|crée|génère|fabrique|"
        r"installe|configure|déploie|teste|debug|corrige|fixe|"
        r"télécharge|download|upload|envoie|"
        r"scrape|crawl|fetch|parse|"
        # Verbes d'action EN
        r"find|search|look\s*up|write|create|build|make|run|execute|"
        r"install|deploy|test|fix|generate|analyze|summarize|translate"
        r")\b",
        re.IGNORECASE,
    )

    def _looks_like_action(self, request: str) -> bool:
        """
        Détecte si une requête classée 'conversation' contient en réalité
        un verbe d'action qui devrait déclencher un worker.

        Filet de sécurité pour éviter que le classifier ne laisse passer
        des demandes d'action comme de simples conversations.
        """
        # Ne pas promouvoir les questions pures (qui est, qu'est-ce que, c'est quoi)
        if re.search(r"^\s*(?:qui|que|quel|comment|pourquoi|c[''']est\s+quoi|qu[''']est)", request, re.IGNORECASE):
            return False
        # Ne pas promouvoir les messages très courts (<4 mots) sauf impératif clair
        words = request.split()
        if len(words) < 4 and not re.match(r"(?:cherche|trouve|crée|écris|lance|fais)\b", request, re.IGNORECASE):
            return False
        return bool(self._LOOKS_LIKE_ACTION_RE.search(request))

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

    # (Supprimé : _SIMPLE_CONVERSATION_RE et _SHORT_QUESTION_RE)
    # La classification est désormais gérée par le LLM dans _classify_intent()

    # ─── Prompt de classification LLM ────────────────────
    _CLASSIFY_PROMPT = """Classifie cette requête utilisateur. Réponds UNIQUEMENT en JSON strict.

Requête : {request}
{original_block}{context_block}{task_registry_block}

Types d'intent (du plus passif au plus structuré) :
- "conversation" : UNIQUEMENT si l'utilisateur pose une question pure, discute, ou demande une info sans vouloir qu'on FASSE quoi que ce soit. Ex: "c'est quoi le Kelly criterion ?", "comment ça va ?", "explique-moi X"
- "smart_action" : l'utilisateur demande de FAIRE quelque chose de concret mais en une seule action. Recherche, code, analyse, rédaction, calcul, traduction, vérification — tout ce qui demande d'AGIR sans être un gros projet. C'est l'intent le PLUS COURANT. Ex: "cherche les résultats de Roland Garros", "écris un script Python pour X", "analyse ce CSV", "traduis ce texte", "calcule le rendement de...", "vérifie si...", "résume cet article"
- "tool_use" : synonyme de smart_action (traité de la même façon). Utilise smart_action de préférence.
- "project" : l'utilisateur veut lancer un GROS travail structuré en plusieurs étapes. Un projet = une mission complexe avec planification. Ex: "monte-moi un bot de trading", "crée une app web pour gérer...", "mets en place un système de..."
- "crew_directive" : l'utilisateur donne un ORDRE à un projet/crew EXISTANT. Ex: "change l'approche du P1", "pause le projet", "ajoute une étape", "reprends le P2"

RÈGLE D'OR : en cas de doute, choisis "smart_action". C'est TOUJOURS mieux d'agir que de juste parler.
- Si l'utilisateur dit "cherche X" → smart_action (PAS conversation)
- Si l'utilisateur dit "fais X" → smart_action (PAS conversation)
- Si l'utilisateur dit "écris X" → smart_action (PAS conversation)
- "conversation" = ZÉRO action demandée, juste une question ou une discussion
- "project" = UNIQUEMENT si c'est clairement un gros travail multi-étapes

Pour "crew_directive", indique aussi :
- "directive_type" : "send_instruction|pause|resume|add_step|modify_step"
- "target_project" : le short_id du projet concerné (P1, P2, etc.)

Types de worker (pour smart_action/tool_use/project) :
- "researcher" : recherche web, collecte d'infos, actualité
- "coder" : écriture/exécution de code, debug, scripts
- "analyst" : analyse de données, calculs, stratégie, finance
- "writer" : rédaction de textes, rapports, emails
- "summarizer" : résumés, synthèses
- "translator" : traduction
- "generic" : tâches mixtes ou inclassables

Réponds UNIQUEMENT avec ce JSON :
{{"intent": "conversation|smart_action|tool_use|project|crew_directive", "worker_type": "researcher|coder|analyst|writer|summarizer|translator|generic", "confidence": 0.0-1.0, "reasoning": "explication courte", "target_project": "P1|P2|...|null", "directive_type": "send_instruction|pause|resume|add_step|modify_step|null"}}"""

    def _build_task_registry_context(self, request: str) -> str:
        """
        Construit un contexte compact du TaskRegistry pour injection dans les prompts.

        - Détecte les références P{N} / T{N} → lookup spécifique
        - Sinon → liste les projets/tâches actifs (max 5)
        """
        if not self.memory or not self.memory.task_registry:
            return ""

        try:
            lines = []

            # Détecter les références explicites (P1, P2, T3, etc.)
            refs = re.findall(r'\b[Pp](\d+)\b|\b[Tt](\d+)\b', request)
            if refs:
                for p_num, t_num in refs:
                    if p_num:
                        epic = self.memory.task_registry.find_epic_by_short_id(f"P{p_num}")
                        if epic:
                            tasks = self.memory.task_registry.get_epic_tasks(epic.id)
                            done = sum(1 for t in tasks if t.status == "done")
                            lines.append(
                                f"#{epic.short_id} {epic.display_name} ({epic.status}) "
                                f"— {done}/{len(tasks)} tâche(s) terminée(s)"
                            )
                            for t in tasks[:5]:
                                lines.append(f"  └ #{t.short_id} {t.description[:60]} ({t.status})")
                    elif t_num:
                        task = self.memory.task_registry.find_task_by_short_id(f"T{t_num}")
                        if task:
                            lines.append(
                                f"#{task.short_id} {task.description[:80]} ({task.status}) "
                                f"— worker: {task.worker_type}"
                            )

            # Si pas de référence explicite, lister les projets/tâches actifs
            if not lines:
                active_epics = [
                    e for e in self.memory.task_registry.get_all_epics(limit=5)
                    if e.status in ("pending", "in_progress")
                ]
                for e in active_epics:
                    tasks = self.memory.task_registry.get_epic_tasks(e.id)
                    done = sum(1 for t in tasks if t.status == "done")
                    lines.append(f"#{e.short_id} {e.display_name} ({e.status}) — {done}/{len(tasks)}")

                active_tasks = [
                    t for t in self.memory.task_registry.get_all_tasks(limit=5)
                    if t.status in ("pending", "in_progress") and not t.epic_id
                ]
                for t in active_tasks:
                    lines.append(f"#{t.short_id} {t.description[:60]} ({t.status})")

            return "\n".join(lines) if lines else ""

        except Exception as e:
            logger.debug("Failed to build task registry context: %s", e)
            return ""

    def _build_recent_sessions_context(self) -> str:
        """
        Construit un résumé des sessions de conversation récentes.

        Ouvre le ConversationStore et charge les résumés de session_summaries.
        Retourne un texte formaté pour injection dans le system prompt.
        """
        try:
            from neo_core.memory.conversation import ConversationStore

            db_path = self.config.memory.storage_path / "conversations.db"
            if not db_path.exists():
                return "(aucune session précédente)"

            store = ConversationStore(db_path)
            summaries = store.get_recent_summaries(days=7, limit=5)
            if summaries:
                return "\n".join(summaries)

            return "(aucune session précédente)"
        except Exception as e:
            logger.debug("Failed to build recent sessions context: %s", e)
            return "(aucune session précédente)"

    def _build_full_projects_context(self) -> str:
        """
        Construit un contexte COMPLET des projets pour injection dans le system prompt.

        Plus riche que _build_task_registry_context() (qui est pour le classify prompt).
        Celui-ci est injecté en haut du system prompt pour que Brain
        connaisse TOUJOURS ses projets, même après un redémarrage.
        """
        if not self.memory or not self.memory.task_registry:
            return "Aucun projet en cours."

        try:
            registry = self.memory.task_registry
            epics = registry.get_all_epics(limit=10)

            if not epics:
                return "Aucun projet en cours."

            lines = []
            active = [e for e in epics if e.status in ("pending", "in_progress")]
            done = [e for e in epics if e.status == "done"]
            failed = [e for e in epics if e.status == "failed"]

            if active:
                for e in active:
                    tasks = registry.get_epic_tasks(e.id)
                    tasks.sort(key=lambda t: t.created_at)
                    done_count = sum(1 for t in tasks if t.status == "done")
                    total = len(tasks)
                    pct = f"{done_count * 100 // total}%" if total > 0 else "0%"

                    # Essayer de charger le CrewState pour le statut crew (paused, etc.)
                    crew_status = ""
                    try:
                        executor = CrewExecutor(brain=self)
                        cs = executor.load_state(e.id)
                        if cs and cs.status == "paused":
                            crew_status = " [EN PAUSE]"
                    except Exception:
                        pass

                    lines.append(
                        f"#{e.short_id} « {e.display_name} » — {e.status}{crew_status} "
                        f"— {done_count}/{total} tâches ({pct})"
                    )
                    if e.strategy:
                        lines.append(f"  Stratégie : {e.strategy[:100]}")

                    for t in tasks:
                        status_icon = {"pending": "⏳", "in_progress": "🔄", "done": "✅", "failed": "❌"}.get(t.status, "?")
                        result_preview = ""
                        if t.status == "done" and t.result:
                            result_preview = f" → {t.result[:80]}"
                        lines.append(
                            f"  {status_icon} #{t.short_id} [{t.worker_type}] {t.description[:60]}{result_preview}"
                        )
                    lines.append("")

            if done:
                lines.append(f"Projets terminés : {len(done)}")
                for e in done[-3:]:
                    lines.append(f"  ✅ #{e.short_id} {e.display_name[:50]}")

            if failed:
                lines.append(f"Projets échoués : {len(failed)}")
                for e in failed[-2:]:
                    lines.append(f"  ❌ #{e.short_id} {e.display_name[:50]}")

            # Tâches standalone (non rattachées à un projet)
            standalone = [
                t for t in registry.get_all_tasks(limit=10)
                if not t.epic_id and t.status in ("pending", "in_progress")
            ]
            if standalone:
                lines.append(f"\nTâches indépendantes : {len(standalone)}")
                for t in standalone[:5]:
                    lines.append(f"  #{t.short_id} [{t.worker_type}] {t.description[:60]} ({t.status})")

            return "\n".join(lines) if lines else "Aucun projet en cours."

        except Exception as e:
            logger.debug("Failed to build full projects context: %s", e)
            return "Aucun projet en cours."

    async def _classify_intent(self, request: str, original_request: str = "",
                                context: str = "") -> dict:
        """
        Classifie l'intention via un appel LLM Haiku (rapide, ~200 tokens).

        Retourne un dict avec : intent, worker_type, confidence, reasoning.
        Fallback sur les regex en cas d'erreur/timeout.
        """
        import asyncio as _aio

        original_block = ""
        if original_request and original_request != request:
            original_block = f"Message original (avant reformulation) : {original_request}\n"
        context_block = ""
        if context:
            context_block = f"Contexte récent : {context[:300]}\n"

        # Injection du contexte TaskRegistry (projets/tâches actifs)
        task_registry_block = ""
        registry_ctx = self._build_task_registry_context(
            original_request if original_request else request
        )
        if registry_ctx:
            task_registry_block = f"\nProjets et tâches actifs :\n{registry_ctx}\n"

        prompt = self._CLASSIFY_PROMPT.format(
            request=request,
            original_block=original_block,
            context_block=context_block,
            task_registry_block=task_registry_block,
        )

        try:
            from neo_core.brain.providers.router import route_chat
            response = await _aio.wait_for(
                route_chat(
                    agent_name="vox",  # Utilise Haiku (rapide)
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.1,
                ),
                timeout=8.0,
            )

            if response.text and not response.text.startswith("[Erreur"):
                data = self._parse_json_response(response.text)
                logger.info(
                    "[Brain] LLM classifier: intent=%s, worker=%s, confidence=%.1f — %s",
                    data.get("intent"), data.get("worker_type"),
                    data.get("confidence", 0), data.get("reasoning", "")[:60],
                )
                return data

        except _aio.TimeoutError:
            logger.warning("[Brain] LLM classifier timeout (>8s), fallback regex")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("[Brain] LLM classifier parse error: %s", e)
        except Exception as e:
            logger.warning("[Brain] LLM classifier error: %s", e)

        # Fallback: classification par regex (ancien système)
        return self._classify_intent_regex(request, original_request)

    def _classify_intent_regex(self, request: str, original_request: str = "") -> dict:
        """Fallback regex pour la classification (ancien système amélioré)."""
        # Epic intent → project
        has_epic = bool(self._EPIC_INTENT_RE.search(request))
        if not has_epic and original_request:
            has_epic = bool(self._EPIC_INTENT_RE.search(original_request))
        if has_epic:
            return {"intent": "project", "worker_type": "generic", "confidence": 0.7,
                    "reasoning": "regex: epic intent detected"}

        # Outil explicitement requis → smart_action
        if self._NEEDS_TOOL_RE.search(request):
            worker_type = self.factory.classify_task(request)
            return {"intent": "smart_action", "worker_type": worker_type.value, "confidence": 0.7,
                    "reasoning": "regex: tool keyword detected"}

        # Verbe d'action détecté → smart_action (filet de sécurité)
        if self._looks_like_action(request):
            worker_type = self.factory.classify_task(request)
            return {"intent": "smart_action", "worker_type": worker_type.value, "confidence": 0.6,
                    "reasoning": "regex: action verb detected"}

        # Default → conversation (vraiment aucun signal d'action)
        return {"intent": "conversation", "worker_type": "generic", "confidence": 0.6,
                "reasoning": "regex: pure conversation"}

    def make_decision(self, request: str, working_context: str = "",
                      original_request: str = "",
                      llm_classification: dict | None = None) -> BrainDecision:
        """
        Prend une décision stratégique sur la manière de traiter la requête.

        v4.0 — Classification fluide avec smart_action :
        0. Commandes slash → fast-path direct
        1. Classification LLM (Haiku, 8s timeout) → intent routing
        2. Fallback regex amélioré si le LLM échoue

        Pipeline (du plus structuré au plus simple) :
        1. "project" → delegate_crew (multi-étapes)
        2. "crew_directive" → piloter un projet existant
        3. "smart_action" / "tool_use" → delegate_worker (action concrète)
        4. "conversation" → direct_response (avec promotion si verbe d'action)
        """
        # 0. Commandes slash → fast-path direct (pas besoin de LLM)
        if request.strip().startswith("/"):
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

        # 2. Classification LLM (ou fallback regex)
        classification = llm_classification or self._classify_intent_regex(request, original_request)
        intent = classification.get("intent", "conversation")
        llm_worker_type = classification.get("worker_type", "generic")

        # 3. Routing basé sur l'intent
        if intent == "project":
            try:
                worker_type = WorkerType(llm_worker_type)
            except ValueError:
                worker_type = self.factory.classify_task(classification_input)

            # Si la classification vient de la LLM (pas du fallback regex),
            # on fait confiance et on force delegate_crew directement.
            # La LLM comprend le langage naturel — pas besoin de re-vérifier
            # avec _EPIC_INTENT_RE qui rate les formulations libres.
            if llm_classification and llm_classification.get("intent") == "project":
                confidence = float(classification.get("confidence", 0.7))
                subtasks = self._decompose_task(request)
                advice = self._consult_learning(request, "generic")
                metadata = {"_cached_learning_advice": advice} if advice else {}
                metadata["epic_intent"] = True
                metadata["llm_classified"] = True
                return BrainDecision(
                    action="delegate_crew",
                    subtasks=subtasks if subtasks else [request],
                    confidence=max(confidence, 0.7),
                    worker_type=worker_type.value,
                    reasoning=f"Projet détecté par LLM ({classification.get('reasoning', '')[:60]}) → Crew",
                    metadata=metadata,
                )

            # Fallback regex : on passe par _decide_complex_generic
            # qui re-vérifie avec _EPIC_INTENT_RE
            return self._decide_complex_generic(request, worker_type)

        if intent == "crew_directive":
            target_project = classification.get("target_project")
            directive_type = classification.get("directive_type", "send_instruction")
            metadata = {
                "crew_directive": True,
                "directive_type": directive_type if directive_type and directive_type != "null" else "send_instruction",
            }
            if target_project and target_project != "null":
                metadata["target_project"] = target_project
            return BrainDecision(
                action="crew_directive",
                confidence=float(classification.get("confidence", 0.8)),
                reasoning=f"Directive crew détectée: {directive_type} → {target_project or 'auto'}",
                metadata=metadata,
            )

        if intent == "tool_use":
            try:
                worker_type = WorkerType(llm_worker_type)
            except ValueError:
                worker_type = self.factory.classify_task(classification_input)
            decision = self._decide_typed_worker(
                request, complexity, worker_type, patch_overrides
            )
            # Propager le target_project du LLM pour rattacher la tâche à un projet
            target_project = classification.get("target_project")
            if target_project and target_project != "null":
                decision.metadata["target_project"] = target_project
            return decision

        # 4. Intent "smart_action" → Brain décide dynamiquement
        if intent == "smart_action":
            try:
                worker_type = WorkerType(llm_worker_type)
            except ValueError:
                worker_type = self.factory.classify_task(classification_input)
            # smart_action = le LLM pense qu'il faut agir mais pas forcément un projet
            # → on délègue à un worker unique (pas un crew complet)
            decision = self._decide_typed_worker(
                request, complexity, worker_type, patch_overrides
            )
            decision.reasoning = f"Smart action ({classification.get('reasoning', '')[:60]}) → {worker_type.value}"
            return decision

        # 5. Défaut → direct_response (Claude Sonnet + mémoire)
        # Filet de sécurité : si la requête contient un verbe d'action clair
        # et que le classifier a mis "conversation" par défaut, on promeut en tool_use
        if intent == "conversation" and self._looks_like_action(request):
            logger.info("[Brain] Promotion conversation → tool_use (verbe d'action détecté)")
            worker_type = self.factory.classify_task(classification_input)
            return self._decide_typed_worker(
                request, complexity, worker_type, patch_overrides
            )

        return self._decide_direct(request, complexity)

    # (Supprimé : _is_simple_conversation())
    # Toutes les décisions sont désormais prises par le LLM via _classify_intent()
    # Seules les commandes slash (/status, /tasks) restent en fast-path dans process()

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
            data = self._parse_json_response(response_text)

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

    # ─── Callbacks pour résultats asynchrones ───────────────

    def set_action_result_callback(self, callback):
        """Enregistre un callback pour recevoir les résultats d'actions async."""
        self._action_result_callback = callback

    def _deliver_action_result(self, message: str):
        """Délivre un résultat d'action via callback (thread-safe)."""
        if self._action_result_callback:
            try:
                self._action_result_callback(message)
            except Exception as e:
                logger.warning("[Brain] Action result callback failed: %s", e)

    # ─── Smart Response : parse action JSON ──────────────
    def _parse_smart_response(self, raw_response: str) -> tuple:
        """
        Parse une réponse Brain pour extraire le texte et une éventuelle action JSON.

        Brain peut ajouter un bloc JSON en fin de réponse pour déclencher une action.
        Format attendu : texte libre suivi de ```neo-action\\n{...}\\n```

        Returns:
            (text_part, action_dict) — action_dict est None si pas d'action
        """
        import re as _re

        # Cherche un bloc ```neo-action ... ``` en fin de réponse
        pattern = _re.compile(
            r'```neo-action\s*\n(\{.*?\})\s*\n```\s*$',
            _re.DOTALL,
        )
        match = pattern.search(raw_response)
        if match:
            try:
                action = json.loads(match.group(1))
                text_part = raw_response[:match.start()].strip()
                logger.info("[Brain] Smart action detected: %s", action.get("action", "?"))
                return text_part, action
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("[Brain] Failed to parse action JSON: %s", e)

        # Fallback : cherche un JSON brut en dernière ligne
        lines = raw_response.strip().split("\n")
        if lines:
            last_line = lines[-1].strip()
            if last_line.startswith("{") and last_line.endswith("}"):
                try:
                    action = json.loads(last_line)
                    if "action" in action:
                        text_part = "\n".join(lines[:-1]).strip()
                        logger.info("[Brain] Smart action (inline): %s", action.get("action", "?"))
                        return text_part, action
                except (json.JSONDecodeError, KeyError):
                    pass

        return raw_response, None

    async def _execute_action(self, action: dict, request: str,
                               memory_context: str,
                               original_request: str = "") -> None:
        """
        Exécute une action détectée dans la réponse Brain (en arrière-plan).

        Appelé via asyncio.create_task() — ne bloque pas le pipeline principal.
        Les résultats sont délivrés via _action_result_callback.
        """
        action_type = action.get("action", "")
        logger.info("[Brain] Executing async action: %s", action_type)

        try:
            if action_type in ("search", "delegate"):
                # Déléguer à un Worker
                worker_type_str = action.get("worker", "researcher")
                task_desc = action.get("task") or action.get("query") or request
                try:
                    wt = WorkerType(worker_type_str)
                except ValueError:
                    wt = WorkerType.RESEARCHER if action_type == "search" else WorkerType.GENERIC

                decision = BrainDecision(
                    action="delegate_worker",
                    subtasks=[task_desc],
                    confidence=0.8,
                    worker_type=wt.value,
                    reasoning=f"Smart action → {wt.value}",
                )
                analysis = self.factory.analyze_task(task_desc)
                result = await self._execute_with_worker(task_desc, decision, memory_context, analysis)
                self._deliver_action_result(f"✅ {result[:1000]}")

            elif action_type == "code":
                # Exécuter du code via un Worker Coder
                code = action.get("code", "")
                task_desc = f"Exécute ce code Python et retourne le résultat :\n```python\n{code}\n```"
                decision = BrainDecision(
                    action="delegate_worker",
                    subtasks=[task_desc],
                    confidence=0.9,
                    worker_type="coder",
                    reasoning="Smart action → code execution",
                )
                analysis = self.factory.analyze_task(task_desc)
                result = await self._execute_with_worker(task_desc, decision, memory_context, analysis)
                self._deliver_action_result(f"✅ {result[:1000]}")

            elif action_type == "create_project":
                # Créer un projet (Crew)
                name = action.get("name", "Projet")
                steps = action.get("steps", [request])
                decision = BrainDecision(
                    action="delegate_crew",
                    subtasks=steps if steps else [request],
                    confidence=0.8,
                    worker_type="generic",
                    reasoning=f"Smart action → Projet '{name}'",
                    metadata={"epic_intent": True, "llm_classified": True},
                )
                _orig = original_request or request
                result = await self._execute_as_epic(request, decision, memory_context, _orig)
                self._deliver_action_result(f"🏁 Projet terminé : {result[:1000]}")

            elif action_type == "crew_directive":
                # Diriger un crew existant
                project_id = action.get("project", "")
                directive_type = action.get("type", "send_instruction")
                detail = action.get("detail", request)
                decision = BrainDecision(
                    action="crew_directive",
                    confidence=0.8,
                    reasoning=f"Smart action → directive {directive_type} sur {project_id}",
                    metadata={
                        "crew_directive": True,
                        "directive_type": directive_type,
                        "target_project": project_id,
                    },
                )
                result = await self._execute_crew_directive(detail, decision)
                self._deliver_action_result(f"📋 {result[:1000]}")

            else:
                logger.warning("[Brain] Unknown action type: %s", action_type)
                self._deliver_action_result(f"⚠️ Action inconnue : {action_type}")

        except Exception as e:
            logger.error("[Brain] Action execution failed: %s — %s", action_type, e)
            self._deliver_action_result(f"❌ Erreur ({action_type}): {str(e)[:300]}")

    async def process(self, request: str,
                      conversation_history: list[BaseMessage] | None = None,
                      original_request: str = "") -> str:
        """
        Pipeline principal de Brain — v5.0 Smart Pipeline.

        UN SEUL appel Sonnet qui comprend + décide + répond.
        Si Brain décide d'agir, il ajoute un bloc JSON action en fin de réponse.
        Les actions sont exécutées en arrière-plan (asyncio.create_task).

        Fallback sur l'ancien pipeline (classify → make_decision) si smart échoue.
        """
        import asyncio as _aio

        # Input validation
        try:
            request = validate_message(request)
        except ValidationError as e:
            logger.warning("Validation du message échouée: %s", e)
            return f"[Brain Erreur] Message invalide: {str(e)[:200]}"

        # Slash commands → fast-path direct (pas de LLM)
        if request.strip().startswith("/"):
            return await self._process_legacy(request, conversation_history, original_request)

        # Mock mode → ancien pipeline
        if self._mock_mode:
            return await self._process_legacy(request, conversation_history, original_request)

        # Circuit breaker
        if not self.health.can_make_api_call():
            return (
                "[Brain] Le système est temporairement indisponible "
                "(trop d'erreurs consécutives). Réessayez dans quelques instants."
            )

        # ── Charger le contexte ──
        working_context = ""
        if self.memory and self.memory.is_initialized:
            try:
                working_context = self.memory.get_working_context()
            except Exception as e:
                logger.debug("Failed to load working context: %s", e)

        memory_context = self.get_memory_context(
            request, working_context=working_context,
        )

        # Enrichir avec le contexte crew si question sur un projet
        crew_epic_id = self._detect_crew_query(request)
        if crew_epic_id:
            try:
                executor = CrewExecutor(brain=self)
                briefing = executor.get_briefing(crew_epic_id)
                memory_context += f"\n\n=== CREW ACTIF ===\n{briefing}"
            except Exception as e:
                logger.debug("Injection contexte crew échouée: %s", e)

        # ── UN SEUL appel Sonnet — comprend + décide + répond ──
        try:
            if self._oauth_mode:
                raw_response = await self._oauth_response(request, memory_context, conversation_history)
            else:
                raw_response = await self._llm_response(request, memory_context, conversation_history)

            # Parser : texte + éventuelle action JSON
            text_part, action = self._parse_smart_response(raw_response)

            if action:
                # Spawner l'action en arrière-plan — ne bloque pas
                _aio.create_task(
                    self._execute_action(action, request, memory_context, original_request)
                )
                logger.info("[Brain] Action '%s' spawned in background", action.get("action"))
                return text_part or "C'est parti, je m'en occupe."

            # Pas d'action → réponse directe (conversation)
            return text_part

        except Exception as e:
            logger.warning("[Brain] Smart pipeline failed, falling back to legacy: %s", e)
            return await self._process_legacy(request, conversation_history, original_request)

    async def _process_legacy(self, request: str,
                               conversation_history: list[BaseMessage] | None = None,
                               original_request: str = "") -> str:
        """
        Ancien pipeline (classify → make_decision → execute).
        Utilisé comme fallback si le smart pipeline échoue,
        et pour les slash commands / mock mode.
        """
        # Working memory
        working_context = ""
        if self.memory and self.memory.is_initialized:
            try:
                working_context = self.memory.get_working_context()
            except Exception as e:
                logger.debug("Failed to load working context for decision: %s", e)

        # Classification LLM async (Haiku, 8s timeout)
        llm_classification = None
        if not request.strip().startswith("/"):
            try:
                llm_classification = await self._classify_intent(
                    request, original_request=original_request,
                    context=working_context[:300] if working_context else "",
                )
            except Exception as e:
                logger.warning("[Brain] LLM classification failed, using regex: %s", e)

        decision = self.make_decision(
            request, working_context=working_context,
            original_request=original_request,
            llm_classification=llm_classification,
        )

        cached_advice = decision.metadata.pop("_cached_learning_advice", None)
        memory_context = self.get_memory_context(
            request, working_context=working_context,
            cached_learning_advice=cached_advice,
        )

        # Intercept crew directives
        if decision.action == "crew_directive":
            try:
                return await self._execute_crew_directive(request, decision)
            except Exception as e:
                logger.error("Crew directive failed: %s", e)
                decision = BrainDecision(
                    action="direct_response", confidence=0.7,
                    reasoning=f"Crew directive fallback: {e}",
                )

        # Intercept crew queries
        crew_epic_id = self._detect_crew_query(request)
        if crew_epic_id:
            try:
                executor = CrewExecutor(brain=self)
                briefing = executor.get_briefing(crew_epic_id)
                memory_context += f"\n\n=== CREW ACTIF ===\n{briefing}"
                decision = BrainDecision(
                    action="direct_response", confidence=0.9,
                    reasoning=f"Question sur crew actif {crew_epic_id[:8]}",
                )
            except Exception as e:
                logger.debug("Injection contexte crew échouée: %s", e)

        _original_req = original_request or request

        if self._mock_mode:
            if decision.action == "delegate_crew" and decision.subtasks:
                return await self._execute_as_epic(request, decision, memory_context, _original_req)
            if decision.action == "delegate_worker" and decision.worker_type:
                return await self._execute_with_worker(request, decision, memory_context)
            complexity = self.analyze_complexity(request)
            return self._mock_response(request, complexity, memory_context)

        if not self.health.can_make_api_call():
            return "[Brain] Système temporairement indisponible."

        try:
            if decision.action == "delegate_crew" and decision.subtasks:
                return await self._execute_as_epic(request, decision, memory_context, _original_req)
            if decision.action == "delegate_worker" and decision.worker_type:
                analysis = self.factory.analyze_task(request)
                return await self._execute_with_worker(request, decision, memory_context, analysis)
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
                # Résoudre le projet cible si spécifié par le LLM
                epic_id = None
                target_project = decision.metadata.get("target_project")
                if target_project and self.memory.task_registry:
                    epic = self.memory.task_registry.find_epic_by_short_id(target_project)
                    if epic:
                        epic_id = epic.id
                        logger.info(
                            "[Brain] Tâche rattachée au projet #%s (%s)",
                            epic.short_id, epic.display_name[:30],
                        )

                task_record = self.memory.create_task(
                    description=request[:200],
                    worker_type=decision.worker_type or "generic",
                    epic_id=epic_id,
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
            f'  {{"description": "Action concrète...", "worker_type": "researcher", "depends_on": []}},\n'
            f'  {{"description": "Analyser les données collectées...", "worker_type": "analyst", "depends_on": [0]}},\n'
            f'  {{"description": "Coder la simulation...", "worker_type": "coder", "depends_on": [0]}},\n'
            f'  {{"description": "Rédiger le rapport final...", "worker_type": "writer", "depends_on": [1, 2]}}\n'
            f"]\n\n"
            f"RÈGLES CRUCIALES :\n"
            f"- Les descriptions doivent être SPÉCIFIQUES au projet, pas génériques\n"
            f"- VARIE les worker_type selon l'étape — PAS que des researcher\n"
            f"- depends_on : liste des INDEX des étapes qui doivent être terminées AVANT cette étape\n"
            f"- Les étapes SANS dépendance commune peuvent s'exécuter EN PARALLÈLE\n"
            f"- La première étape a toujours depends_on: []\n"
            f"- L'étape finale dépend de toutes les étapes précédentes nécessaires\n"
            f"- Adapte les étapes à l'OBJECTIF RÉEL du projet\n"
            f"- Réponds UNIQUEMENT avec le JSON, rien d'autre."
        )

        try:
            response = await self._raw_llm_call(decompose_prompt)
            data = self._parse_json_response(response)
            if isinstance(data, list) and len(data) >= 2:
                steps = []
                for i, item in enumerate(data):
                    desc = item.get("description", "")
                    wt_str = item.get("worker_type", "generic")
                    depends_on = item.get("depends_on", [])
                    # Valider les dépendances (doivent être des indices < i)
                    depends_on = [d for d in depends_on if isinstance(d, int) and 0 <= d < i]
                    try:
                        wt = WorkerType(wt_str)
                    except ValueError:
                        wt = WorkerType.GENERIC
                    if desc:
                        steps.append(CrewStep(
                            index=i, description=desc, worker_type=wt,
                            depends_on=depends_on,
                        ))
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

    async def _extract_epic_name_and_subject(self, request: str) -> tuple[str, str]:
        """
        Extrait le NOM du projet et le SUJET/DESCRIPTION via la LLM.

        La LLM comprend le langage naturel et extrait correctement le nom
        du projet quelle que soit la formulation (avec ou sans guillemets,
        en français, avec des contractions, etc.).

        Exemples :
        - "crée un projet 'Smash Gang', objectif : devenir rentable"
          → name="Smash Gang", subject="devenir rentable"
        - "lance un projet roadmap pour la refonte du site"
          → name="Roadmap Refonte Site", subject="refonte du site"
        - "Neo j'ai besoin que tu creer un nouveau projet Smash gang"
          → name="Smash Gang", subject="créer le projet Smash Gang"

        Returns:
            (name, subject) — name peut être vide si non détecté
        """
        extraction_prompt = (
            "Extrait le NOM du projet et sa DESCRIPTION à partir de cette demande utilisateur.\n\n"
            f"Demande : \"{request}\"\n\n"
            "RÈGLES :\n"
            "- 'name' : le nom court du projet tel que l'utilisateur le souhaite "
            "(ex: 'Smash Gang', 'Alpha Team', 'Roadmap Q1'). "
            "Capitalise chaque mot.\n"
            "- 'subject' : une description concise de l'objectif/sujet du projet.\n"
            "- Si l'utilisateur ne donne pas de nom explicite, génère un nom court et pertinent "
            "basé sur le sujet.\n"
            "- Ignore les formules de politesse, les préfixes comme 'Neo', "
            "'j\\'ai besoin que tu', etc.\n"
            "- NE PAS inclure de préfixes comme 'Projet' dans le nom.\n\n"
            "Réponds UNIQUEMENT en JSON strict :\n"
            '{"name": "Nom Du Projet", "subject": "Description courte du projet"}'
        )

        try:
            response = await self._raw_llm_call(extraction_prompt)
            data = self._parse_json_response(response)
            name = str(data.get("name", "")).strip()[:100]
            subject = str(data.get("subject", "")).strip()[:1000]

            if name:
                logger.info("Extraction LLM — name=%r, subject=%r", name, subject[:50])
                return name, subject if len(subject) >= 5 else request

        except Exception as e:
            logger.debug("Extraction LLM du nom échouée: %s — fallback requête brute", e)

        # Fallback minimal : retourner la requête comme sujet
        return "", request[:1000]

    async def _execute_as_epic(self, request: str, decision: BrainDecision,
                               memory_context: str,
                               original_request: str = "") -> str:
        """
        Crée un Epic et lance un Chef de Projet (v3.0).

        Pipeline :
        1. Extraction du sujet réel
        2. Décomposition intelligente avec worker_type par étape (JSON LLM)
        3. Création de l'Epic dans le registre
        4. Spawn du ProjectManagerWorker (Chef de Projet)
        5. Le PM exécute TOUTES les étapes en parallèle
        6. Retourne la synthèse complète

        Le PM gère les dépendances entre étapes et spawn les workers
        par batch parallèle. Plus besoin du heartbeat pour avancer.
        """
        # 0. Extraire le nom et le sujet (préférer le message original pour les quotes)
        source_for_name = original_request if original_request else request
        epic_name, epic_subject = await self._extract_epic_name_and_subject(source_for_name)
        # Si le sujet extrait est trop court, enrichir avec la reformulation
        if len(epic_subject) < 20 and request != source_for_name:
            _, alt_subject = await self._extract_epic_name_and_subject(request)
            if len(alt_subject) > len(epic_subject):
                epic_subject = alt_subject

        # 1. Décomposer en étapes crew avec worker_types
        decompose_input = original_request if original_request else request
        crew_steps = await self._decompose_crew_with_llm(decompose_input, memory_context)

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
                    "Epic créé: %s (%s) avec %d étapes",
                    epic_id[:8], epic_subject[:50], len(crew_steps),
                )
            except Exception as e:
                logger.debug("Impossible de créer l'Epic: %s", e)

        # 3. Mettre à jour le statut → in_progress
        if epic and self.memory and self.memory.is_initialized:
            try:
                self.memory.update_epic_status(epic_id, "in_progress")
            except Exception:
                pass

        # 4. Spawn le Chef de Projet (ProjectManagerWorker)
        try:
            from neo_core.brain.teams.project_manager import ProjectManagerWorker

            pm = ProjectManagerWorker(
                brain=self,
                epic_id=epic_id,
                epic_subject=epic_subject,
                steps=crew_steps,
                memory_context=memory_context,
                original_request=original_request or request,
                event_callback=self._handle_crew_event,
            )

            # 5. Le PM exécute tout — workers en parallèle
            pm_result = await pm.execute()

            # 6. Mettre à jour le statut de l'Epic
            if self.memory and self.memory.is_initialized:
                try:
                    final_status = "done" if pm_result.success else "failed"
                    self.memory.update_epic_status(epic_id, final_status)
                except Exception:
                    pass

            # 7. Retourner la synthèse
            return pm_result.synthesis

        except Exception as e:
            logger.error("ProjectManager execution failed: %s", e)
            if epic and self.memory and self.memory.is_initialized:
                try:
                    self.memory.update_epic_status(epic_id, "failed")
                except Exception:
                    pass
            return f"[Projet échoué — {epic_subject}] {type(e).__name__}: {str(e)[:300]}"

    # ─── Pilotage interactif Brain → Crew ──────────────

    async def _execute_crew_directive(self, request: str, decision: BrainDecision) -> str:
        """
        Exécute une directive de pilotage sur un crew actif.

        Appelé quand le LLM classifie l'intent comme "crew_directive".
        Résout le projet cible, puis dispatch selon le directive_type :
        - send_instruction : injecte une instruction libre
        - pause : met le crew en pause
        - resume : reprend le crew
        - add_step : ajoute une étape
        - modify_step : modifie une étape existante
        """
        directive_type = decision.metadata.get("directive_type", "send_instruction")
        target_project = decision.metadata.get("target_project")

        # Résoudre l'epic_id depuis le short_id ou trouver le crew actif unique
        epic_id = None
        executor = CrewExecutor(brain=self)
        executor.set_event_callback(self._handle_crew_event)

        if target_project and self.memory and self.memory.task_registry:
            epic = self.memory.task_registry.find_epic_by_short_id(target_project)
            if epic:
                epic_id = epic.id

        # Si pas de target explicite, chercher le crew actif unique
        # Si aucun crew actif, chercher aussi les projets terminés (pour réouverture)
        if not epic_id:
            active_crews = executor.list_active_crews()
            if active_crews:
                if len(active_crews) == 1:
                    epic_id = active_crews[0].epic_id
                else:
                    # Plusieurs crews → essayer de matcher par sujet
                    request_lower = request.lower()
                    for crew in active_crews:
                        subject_words = crew.epic_subject.lower().split()[:5]
                        if any(word in request_lower for word in subject_words if len(word) > 3):
                            epic_id = crew.epic_id
                            break
                    if not epic_id:
                        crew_list = "\n".join(
                            f"  • #{c.epic_id[:8]} — {c.epic_subject[:60]}"
                            for c in active_crews
                        )
                        return (
                            f"Plusieurs projets actifs détectés. Précise lequel :\n{crew_list}"
                        )
            else:
                # Pas de crew actif → chercher le projet le plus récent (done/failed)
                # pour permettre la réouverture avec de nouvelles étapes
                recent_epic = self._find_most_recent_epic()
                if recent_epic:
                    epic_id = recent_epic.id
                    # Rouvrir le crew en "active" pour accepter les nouvelles étapes
                    state = executor.load_state(epic_id)
                    if state and state.status in ("done", "failed"):
                        state.status = "active"
                        executor.save_state(state)
                        logger.info("[Brain] Réouverture du projet %s pour directive", epic_id[:8])
                else:
                    return (
                        "Aucun projet trouvé. "
                        "Crée d'abord un projet avec une demande comme "
                        "\"crée un projet Smash Gang pour les paris ATP\"."
                    )

        # Dispatch selon le type de directive
        if directive_type == "pause":
            event = executor.pause_crew(epic_id, reason=request[:200])
            return f"⏸️ {event.message}"

        if directive_type == "resume":
            event = executor.resume_crew(epic_id)
            return f"▶️ {event.message}"

        if directive_type == "add_step":
            # Extraire les étapes à ajouter via LLM (supporte plusieurs d'un coup)
            steps_info = await self._extract_steps_from_directive(request, epic_id)
            if steps_info:
                added = []
                for desc, wtype in steps_info:
                    try:
                        worker_type = WorkerType(wtype)
                    except ValueError:
                        worker_type = WorkerType.GENERIC
                    success = executor.add_step(epic_id, desc, worker_type)
                    if success:
                        added.append(f"  • {desc[:80]} ({worker_type.value})")
                if added:
                    summary = "\n".join(added)
                    # Relancer le PM automatiquement sur les nouvelles étapes
                    relaunch_msg = await self._relaunch_pm_on_new_steps(epic_id, executor)
                    return f"➕ {len(added)} étape(s) ajoutée(s) :\n{summary}\n\n{relaunch_msg}"
                return "Impossible d'ajouter les étapes (crew introuvable)."
            return "Je n'ai pas compris quelles étapes ajouter. Peux-tu préciser ?"

        if directive_type == "modify_step":
            # Extraire l'index et les modifications via LLM
            mod_info = await self._extract_step_modification(request)
            if mod_info:
                step_idx, new_desc, new_wtype = mod_info
                event = executor.modify_step(
                    epic_id, step_idx,
                    new_description=new_desc,
                    new_worker_type=new_wtype,
                )
                return f"✏️ {event.message}"
            return "Je n'ai pas compris quelle étape modifier. Peux-tu préciser l'index et les changements ?"

        # Default: send_instruction (directive libre)
        event = executor.send_directive(epic_id, request)
        # Aussi enrichir la réponse avec le briefing actuel
        briefing = executor.get_briefing(epic_id)
        return f"📋 {event.message}\n\nÉtat actuel :\n{briefing}"

    async def _extract_steps_from_directive(
        self, request: str, epic_id: str,
    ) -> list[tuple[str, str]]:
        """
        Extrait UNE ou PLUSIEURS étapes d'une directive utilisateur.

        L'utilisateur peut dire :
        - "ajoute une étape de scraping" → 1 étape
        - "scrapper les données, créer une BDD, analyser, placer un pari" → 4 étapes

        Retourne une liste de (description, worker_type).
        """
        # Injecter le contexte du projet pour que le LLM comprenne
        project_ctx = ""
        try:
            executor = CrewExecutor(brain=self)
            state = executor.load_state(epic_id)
            if state:
                project_ctx = (
                    f"\nProjet : « {state.epic_subject} »\n"
                    f"Étapes existantes : {len(state.steps)}\n"
                )
        except Exception:
            pass

        prompt = (
            f"L'utilisateur veut ajouter des étapes à un projet.\n"
            f"Requête : {request}\n{project_ctx}\n"
            f"Extrais TOUTES les étapes décrites (1 ou plusieurs).\n"
            f"Chaque étape doit être une ACTION CONCRÈTE.\n\n"
            f"Types de worker :\n"
            f"- researcher : collecte web, scraping, veille\n"
            f"- coder : code, scripts, BDD, automatisation\n"
            f"- analyst : analyse données, calculs, stratégie\n"
            f"- writer : rédaction, rapports\n"
            f"- generic : tâches mixtes\n\n"
            f"Réponds en JSON (array) :\n"
            f'[{{"description": "...", "worker_type": "researcher|coder|analyst|writer|generic"}}]\n\n'
            f"IMPORTANT : même si l'utilisateur décrit les étapes informellement "
            f"(\"scrapper, créer une bdd, analyser...\"), extrais CHAQUE action "
            f"comme une étape séparée avec un worker_type adapté."
        )
        try:
            response = await self._raw_llm_call(prompt)
            data = self._parse_json_response(response)
            if isinstance(data, list):
                return [
                    (item.get("description", ""), item.get("worker_type", "generic"))
                    for item in data
                    if item.get("description")
                ]
            # Fallback si le LLM retourne un dict (1 étape)
            if isinstance(data, dict) and data.get("description"):
                return [(data["description"], data.get("worker_type", "generic"))]
        except Exception as e:
            logger.warning("Extraction steps from directive failed: %s", e)
        return []

    async def _relaunch_pm_on_new_steps(self, epic_id: str, executor: CrewExecutor) -> str:
        """
        Relance le ProjectManager sur les étapes non-complétées d'un crew.

        Appelé après add_step pour exécuter immédiatement les nouvelles étapes
        au lieu d'attendre le heartbeat.
        """
        state = executor.load_state(epic_id)
        if not state:
            return ""

        completed_set = set(state.completed_indices)
        pending_steps = [s for s in state.steps if s.index not in completed_set]

        if not pending_steps:
            return "Toutes les étapes sont déjà terminées."

        try:
            from neo_core.brain.teams.project_manager import ProjectManagerWorker

            pm = ProjectManagerWorker(
                brain=self,
                epic_id=epic_id,
                epic_subject=state.epic_subject,
                steps=pending_steps,
                memory_context=state.memory_context,
                original_request=state.original_request,
                event_callback=self._handle_crew_event,
            )

            logger.info(
                "[Brain] Relance PM sur %d étapes pending pour %s",
                len(pending_steps), epic_id[:8],
            )

            pm_result = await pm.execute()

            if self.memory and self.memory.is_initialized:
                try:
                    final_status = "done" if pm_result.success else "failed"
                    self.memory.update_epic_status(epic_id, final_status)
                except Exception:
                    pass

            return pm_result.synthesis

        except Exception as e:
            logger.error("PM relaunch failed: %s", e)
            return f"[Relance échouée] {type(e).__name__}: {str(e)[:200]}"

    async def _extract_step_modification(self, request: str) -> Optional[tuple[int, str | None, str | None]]:
        """Extrait l'index, nouvelle description et nouveau worker_type d'une modification."""
        prompt = (
            f"L'utilisateur veut modifier une étape d'un projet.\n"
            f"Requête : {request}\n\n"
            f"Extrais l'index de l'étape (0-based), la nouvelle description (ou null), "
            f"et le nouveau worker_type (ou null).\n"
            f"Réponds en JSON : {{\"step_index\": 0, \"new_description\": \"...\"|null, \"new_worker_type\": \"...\"|null}}"
        )
        try:
            response = await self._raw_llm_call(prompt)
            data = self._parse_json_response(response)
            idx = data.get("step_index")
            if idx is not None:
                new_desc = data.get("new_description")
                new_wtype = data.get("new_worker_type")
                if new_desc == "null":
                    new_desc = None
                if new_wtype == "null":
                    new_wtype = None
                return int(idx), new_desc, new_wtype
        except Exception as e:
            logger.debug("Extraction step modification failed: %s", e)
        return None

    def _find_most_recent_epic(self):
        """Trouve l'epic le plus récent (tous statuts) dans le TaskRegistry."""
        if not self.memory or not self.memory.task_registry:
            return None
        try:
            epics = self.memory.task_registry.get_all_epics(limit=5)
            if epics:
                # Trier par date de création décroissante
                epics.sort(key=lambda e: e.created_at, reverse=True)
                return epics[0]
        except Exception as e:
            logger.debug("Find most recent epic failed: %s", e)
        return None

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

    def set_crew_progress_callback(self, callback) -> None:
        """
        Définit un callback pour remonter les événements crew vers l'UI (CLI/API).

        Le callback reçoit un string formaté pour affichage utilisateur.
        Permet de voir la progression en temps réel au lieu d'attendre
        la synthèse finale.
        """
        self._crew_progress_callback = callback

    def _handle_crew_event(self, event: CrewEvent) -> None:
        """
        Reçoit les notifications proactives des crews.

        Stocke en mémoire, notifie Telegram, ET remonte vers le CLI/API
        via le callback de progression.
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

        # Remonter vers le CLI/API (progression temps réel)
        progress_cb = getattr(self, "_crew_progress_callback", None)
        if progress_cb:
            try:
                # Formater un message court pour l'UI
                icon = {
                    "step_completed": "✅",
                    "step_failed": "❌",
                    "crew_done": "🏁",
                    "crew_paused": "⏸️",
                    "crew_resumed": "▶️",
                    "directive_received": "📋",
                    "orchestrator_replan": "🔄",
                }.get(event.event_type, "📌")
                progress_cb(f"{icon} {event.message}")
            except Exception as e:
                logger.debug("Crew progress callback failed: %s", e)

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

        # Injection proéminente des projets actifs (séparée du memory_context)
        projects_context = self._build_full_projects_context()

        recent_sessions = self._build_recent_sessions_context()

        system_prompt = BRAIN_SYSTEM_PROMPT.format(
            memory_context=memory_context,
            projects_context=projects_context,
            current_date=now.strftime("%A %d %B %Y"),
            current_time=now.strftime("%H:%M"),
            user_context=user_context,
            recent_sessions=recent_sessions,
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

        projects_context = self._build_full_projects_context()

        recent_sessions = self._build_recent_sessions_context()

        prompt = ChatPromptTemplate.from_messages([
            ("system", BRAIN_SYSTEM_PROMPT),
            MessagesPlaceholder("conversation_history", optional=True),
            ("human", "{request}"),
        ])
        chain = prompt | self._llm
        result = await chain.ainvoke({
            "memory_context": memory_context,
            "projects_context": projects_context,
            "user_context": user_context,
            "recent_sessions": recent_sessions,
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
