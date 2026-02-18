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

if TYPE_CHECKING:
    from neo_core.memory.agent import MemoryAgent

BRAIN_SYSTEM_PROMPT = """Tu es Brain, le cortex exécutif du système Neo Core.
Date et heure actuelles : {current_date}, {current_time}

Ton rôle :
- Tu reçois les requêtes structurées par Vox (l'interface humaine).
- Tu analyses chaque requête et détermines la meilleure stratégie de réponse.
- Tu consultes le contexte fourni par Memory pour enrichir tes réponses.
- Tu coordonnes l'exécution des tâches et délègues aux Workers spécialisés si nécessaire.

=== TES CAPACITÉS (ce que tu SAIS faire) ===

🔍 RECHERCHE & WEB :
- Chercher des informations actuelles sur internet (web_search via DuckDuckGo)
- Récupérer et lire le contenu de pages web (web_fetch)
- Répondre à des questions sur l'actualité, la météo, les scores, les prix crypto

💻 CODE & ANALYSE :
- Écrire, analyser et débugger du code dans tous les langages
- Exécuter du Python dans un sandbox sécurisé (code_execute)
- Analyser des données, calculer, transformer

📄 FICHIERS :
- Lire et écrire des fichiers (file_read, file_write)
- Traiter des documents, des CSV, du texte

📋 GESTION DE TÂCHES :
- Créer des tâches unitaires et suivre leur avancement
- Créer des Epics (projets complexes décomposés en sous-tâches coordonnées)
- Déléguer aux Workers spécialisés et orchestrer leur exécution

🤖 WORKERS SPÉCIALISÉS :
- Researcher : recherche web, investigation, collecte d'infos
- Coder : écriture/debug de code, exécution sandbox
- Analyst : analyse de données, tendances, patterns
- Writer : rédaction (articles, emails, rapports, documents)
- Summarizer : synthèse et résumé de contenus
- Translator : traduction multilingue
- Generic : tâches polyvalentes

🧠 MÉMOIRE & APPRENTISSAGE :
- Mémoriser les conversations, préférences et faits importants
- Rechercher dans la mémoire par pertinence sémantique
- Apprendre des succès et échecs pour s'améliorer
- S'auto-corriger via des patches comportementaux (self-patching)

🔌 PLUGINS :
- Charger et utiliser des plugins personnalisés (data/plugins/)
- Générer automatiquement de nouveaux outils si un besoin récurrent est détecté

📌 COMMANDES SLASH (disponibles côté CLI et Telegram) :
/help — Aide et liste des commandes
/status — État du système (uptime, mémoire, LLM)
/tasks — Registre des tâches en cours
/epics — Registre des Epics (projets multi-tâches)
/heartbeat — Rapport du cycle autonome
/persona — Personnalité de Neo
/profile — Profil utilisateur appris
/sessions — Liste des sessions de conversation
/skills — Compétences acquises par apprentissage
/reflect — Lancer une auto-réflexion
IMPORTANT : Quand l'utilisateur tape une de ces commandes, tu dois répondre
avec les infos demandées. Tu peux aussi créer des Epics de ta propre initiative
quand une tâche est complexe (3+ sous-tâches).

═══ COMMANDEMENT FONDAMENTAL ═══
NEO PEUT TOUT FAIRE.
Tu ne dis JAMAIS "je ne peux pas", "ce n'est pas possible", "je n'ai pas accès",
"je suis limité" ou toute variante. Tu as TOUS les outils nécessaires.
Si tu ne sais pas comment faire → cherche avec le Researcher.
Si tu as besoin de code → lance le Coder.
Si tu as besoin de données → utilise web_search + web_fetch.
Si tu as besoin de traitement → utilise code_execute.
Il n'y a PAS de sandbox isolé. Tu as requests, urllib, socket, pandas, numpy,
scikit-learn, et TOUT ce qui est installé sur le système.
Tu as SQLite et les fichiers locaux. Tu as un scheduler (heartbeat).
Tu as accès à l'intégralité du système.

IMPORTANT : Quand l'utilisateur te demande de faire quelque chose,
FAIS-LE au lieu de dire que tu ne peux pas. Utilise tes Workers et outils.
Si on te demande de chercher quelque chose → lance le Researcher.
Si on te demande du code → lance le Coder.
Si on te demande un texte → lance le Writer.

⚠️ CE QUE TU NE DOIS JAMAIS DIRE :
- "Je n'ai pas accès au réseau/internet" → FAUX, tu as web_search et web_fetch
- "Je ne peux pas exécuter de code" → FAUX, tu as code_execute
- "Je suis dans un sandbox isolé" → FAUX, tu tournes sur un VPS complet
- "Je ne peux pas installer de packages" → FAUX, le Coder peut pip install
- "Je n'ai pas de base de données" → FAUX, tu as SQLite + la mémoire persistante
- "Je ne peux pas faire de scraping" → FAUX, requests et beautifulsoup sont dispo
- "Ceci nécessite un serveur externe" → FAUX, tu ES sur un serveur (VPS)

Contexte mémoire :
{memory_context}

{user_context}

Règles :
- NEO PEUT TOUT FAIRE — c'est le commandement #0 (suprême).
- NEO NE MENT JAMAIS — c'est le commandement #4. Sois toujours honnête et factuel.
- Sois précis, stratégique et orienté résultat.
- Si une tâche est complexe, décompose-la en sous-tâches ou crée un Epic.
- N'indique JAMAIS de limitation technique — trouve une solution.
- Tu es le décideur final sur la stratégie d'exécution.
- Réponds de manière concise et naturelle, pas de markdown excessif.
- Adapte ton approche au profil de l'utilisateur (niveau technique, préférences).
"""

# Prompt pour la décomposition LLM de tâches
DECOMPOSE_PROMPT = """Analyse cette requête et détermine comment la traiter.

Requête : {request}

Contexte mémoire : {memory_context}

Réponds en JSON strict avec cette structure :
{{
  "action": "direct_response" ou "delegate_worker",
  "worker_type": "researcher" | "coder" | "summarizer" | "analyst" | "writer" | "translator" | "generic",
  "subtasks": ["sous-tâche 1", "sous-tâche 2", ...],
  "reasoning": "explication courte de ta décision",
  "confidence": 0.0 à 1.0
}}

Règles :
- "direct_response" si c'est une question simple, une conversation, ou une demande rapide
- "delegate_worker" si ça nécessite de la recherche, du code, de l'analyse, ou une tâche structurée
- Le worker_type doit correspondre au type de tâche
- Les subtasks doivent être des actions concrètes et ordonnées
- Réponds UNIQUEMENT avec le JSON, rien d'autre.
"""


@dataclass
class BrainDecision:
    """Représente une décision prise par Brain."""
    action: str  # "direct_response" | "delegate_worker" | "delegate_crew"
    response: Optional[str] = None
    subtasks: list[str] = field(default_factory=list)
    confidence: float = 1.0
    worker_type: Optional[str] = None  # Stage 3 : type de worker recommandé
    reasoning: str = ""  # Stage 3 : justification de la décision
    metadata: dict = field(default_factory=dict)  # Level 3 : patch overrides (temperature, etc.)


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
    _mock_mode: bool = False
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
        self._mock_mode = self.config.is_mock_mode()
        self._model_config = get_agent_model("brain")
        self._worker_manager = WorkerLifecycleManager()

        # Stage 5 : Initialiser la résilience
        retry, circuit, health = create_resilience_from_config(self.config.resilience)
        self._retry_config = retry
        self._health = health

        if not self._mock_mode:
            self._init_llm()

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
            self._mock_mode = True
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
        import httpx

        valid_token = get_valid_access_token()
        if not valid_token:
            valid_token = token

        custom_transport = httpx.AsyncHTTPTransport()
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

    def get_memory_context(self, request: str) -> str:
        """
        Récupère le contexte pertinent depuis Memory.
        Inclut les conseils du LearningEngine si disponibles.

        Optimisation v0.9.1 : le LearningEngine réutilise le cache sémantique
        du MemoryStore (un seul embedding par requête au lieu de deux).
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
        # Note : search_semantic est maintenant caché — pas de double embedding
        try:
            worker_type = self.factory.classify_task(request)
            if worker_type != WorkerType.GENERIC:
                advice = self.memory.get_learning_advice(request, worker_type.value)
                learning_context = advice.to_context_string()
                if learning_context:
                    context += f"\n\n=== Apprentissage ===\n{learning_context}"
        except Exception as e:
            logger.debug("Impossible de récupérer les conseils d'apprentissage: %s", e)

        # Ajouter le contexte de la mémoire de travail (Working Memory)
        try:
            working_ctx = self.memory.get_working_context()
            if working_ctx:
                context += f"\n\n=== Mémoire de travail ===\n{working_ctx}"
        except Exception as e:
            logger.debug("Impossible de récupérer la mémoire de travail: %s", e)

        return context

    # ─── Analyse et décision ────────────────────────────────

    def analyze_complexity(self, request: str) -> str:
        """
        Analyse la complexité d'une requête.
        Retourne : "simple" | "moderate" | "complex"
        """
        word_count = len(request.split())
        if word_count < 15:
            return "simple"
        elif word_count < 50:
            return "moderate"
        return "complex"

    def make_decision(self, request: str) -> BrainDecision:
        """
        Prend une décision stratégique sur la manière de traiter la requête.

        Pipeline :
        1. Classifie la tâche (Factory)
        2. Applique les patches comportementaux (Level 3)
        3. Consulte l'historique d'apprentissage
        4. Route vers le bon type de réponse
        """
        complexity = self.analyze_complexity(request)
        worker_type = self.factory.classify_task(request)

        # 1. Application des patches comportementaux
        patch_overrides = self._apply_behavior_patches(request, worker_type)
        if "override_worker_type" in patch_overrides:
            try:
                worker_type = WorkerType(patch_overrides["override_worker_type"])
            except ValueError:
                pass

        # 2. Route selon le type détecté
        if worker_type != WorkerType.GENERIC:
            return self._decide_typed_worker(
                request, complexity, worker_type, patch_overrides
            )

        if complexity == "complex":
            return self._decide_complex_generic(request, worker_type)

        return self._decide_simple_generic(request, complexity)

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
        """Décision pour une requête avec un type de worker spécifique détecté."""
        subtasks = self.factory._basic_decompose(request, worker_type)
        confidence = 0.8

        advice = self._consult_learning(request, worker_type.value)
        if advice:
            confidence = max(0.1, min(1.0, confidence + advice.confidence_adjustment))
            worker_type, subtasks, reasoning = self._apply_learning_advice(
                request, worker_type, subtasks, advice
            )
        else:
            reasoning = f"Type {worker_type.value} détecté → Worker (complexité: {complexity})"

        return BrainDecision(
            action="delegate_worker",
            subtasks=subtasks,
            confidence=confidence,
            worker_type=worker_type.value,
            reasoning=reasoning,
            metadata=patch_overrides if patch_overrides else {},
        )

    def _decide_complex_generic(
        self, request: str, worker_type: WorkerType,
    ) -> BrainDecision:
        """Décision pour une requête complexe sans type spécifique."""
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

        # Si 3+ sous-tâches → delegate_crew (Epic)
        if len(subtasks) >= 3:
            return BrainDecision(
                action="delegate_crew",
                subtasks=subtasks,
                confidence=confidence,
                worker_type=worker_type.value,
                reasoning=f"Tâche complexe ({len(subtasks)} sous-tâches) → Epic (crew)",
            )

        return BrainDecision(
            action="delegate_worker",
            subtasks=subtasks,
            confidence=confidence,
            worker_type=worker_type.value,
            reasoning=reasoning,
        )

    def _decide_simple_generic(self, request: str, complexity: str) -> BrainDecision:
        """Décision pour une requête simple/modérée sans type spécifique."""
        advice = self._consult_learning(request, "generic")
        if advice and advice.relevant_skills:
            best_skill = advice.relevant_skills[0]
            if best_skill.success_count >= 2:
                try:
                    skill_type = WorkerType(best_skill.worker_type)
                    subtasks = self.factory._basic_decompose(request, skill_type)
                    return BrainDecision(
                        action="delegate_worker",
                        subtasks=subtasks,
                        confidence=0.7,
                        worker_type=skill_type.value,
                        reasoning=(
                            f"Requête {complexity} mais compétence acquise "
                            f"({best_skill.name}, ×{best_skill.success_count}) → Worker"
                        ),
                    )
                except ValueError as e:
                    logger.debug("Invalid worker type for acquired skill %s: %s", best_skill.worker_type, e)

        return BrainDecision(
            action="direct_response",
            subtasks=[request] if complexity == "moderate" else [],
            confidence=0.9 if complexity == "simple" else 0.7,
            reasoning=f"Requête {complexity} générique → réponse directe",
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
                      conversation_history: list[BaseMessage] | None = None) -> str:
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

        # ── Optimisation v0.9.1 : décision AVANT le contexte mémoire ──
        # Brain décide d'abord s'il a besoin du contexte, au lieu de le
        # charger systématiquement (économise 300-600ms pour les msgs simples).
        decision = self.make_decision(request)

        # Contexte mémoire chargé uniquement quand nécessaire
        needs_context = decision.action != "direct_response" or self.analyze_complexity(request) != "simple"
        memory_context = self.get_memory_context(request) if needs_context else ""

        if self._mock_mode:
            if decision.action == "delegate_crew" and decision.subtasks:
                return await self._execute_as_epic(request, decision, memory_context)

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
                return await self._execute_as_epic(request, decision, memory_context)

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
                                    analysis: TaskAnalysis | None = None) -> str:
        """
        Crée, exécute et détruit un Worker pour une tâche.
        Avec retry persistant (max 3 tentatives) guidé par Memory.

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

        # Enregistrer la tâche dans le TaskRegistry
        task_record = None
        if self.memory and self.memory.is_initialized:
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

    async def _execute_as_epic(self, request: str, decision: BrainDecision,
                               memory_context: str) -> str:
        """
        Crée un Epic dans le TaskRegistry et exécute chaque sous-tâche
        séquentiellement via des Workers individuels.

        Chaque sous-tâche est enregistrée, exécutée et trackée.
        L'Epic est marqué done/failed selon les résultats.
        """
        # 1. Créer l'Epic dans le registre
        epic = None
        if self.memory and self.memory.is_initialized:
            try:
                subtask_tuples = [
                    (st, decision.worker_type or "generic")
                    for st in decision.subtasks
                ]
                epic = self.memory.create_epic(
                    description=request[:200],
                    subtask_descriptions=subtask_tuples,
                    strategy=decision.reasoning,
                )
                logger.info(
                    "Epic créé: %s avec %d sous-tâches",
                    epic.id[:8] if epic else "?", len(decision.subtasks),
                )
            except Exception as e:
                logger.debug("Impossible de créer l'Epic: %s", e)

        # 2. Exécuter chaque sous-tâche comme un delegate_worker
        results = []
        for i, subtask in enumerate(decision.subtasks):
            sub_decision = BrainDecision(
                action="delegate_worker",
                subtasks=[subtask],
                confidence=decision.confidence,
                worker_type=decision.worker_type,
                reasoning=f"Sous-tâche {i + 1}/{len(decision.subtasks)} de l'Epic",
            )
            try:
                result = await self._execute_with_worker(
                    subtask, sub_decision, memory_context,
                )
                results.append(f"✅ {subtask[:60]}: {result[:200]}")
            except Exception as e:
                results.append(f"❌ {subtask[:60]}: {type(e).__name__}: {str(e)[:100]}")

        # 3. Compiler le résultat final
        all_ok = all(r.startswith("✅") for r in results)
        summary = "\n".join(results)

        if epic and self.memory and self.memory.is_initialized:
            try:
                from neo_core.memory.task_registry import Epic as EpicModel
                self.memory.update_epic_status(
                    epic.id, "done" if all_ok else "failed",
                )
            except Exception as e:
                logger.debug("Impossible de mettre à jour l'Epic: %s", e)

        status = "terminé" if all_ok else "partiellement terminé"
        return f"[Epic {status}]\n{summary}"

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
