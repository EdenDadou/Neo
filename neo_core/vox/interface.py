"""
Vox — Agent Interface Humaine
==============================
Cortex du langage et interface sociale du système Neo Core.

Responsabilités :
- Interagir directement avec l'humain
- Reformuler et structurer les demandes avant transmission à Brain
- Restituer les réponses de Brain sans altération
- Fournir un état en temps réel des autres agents
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from neo_core.config import NeoConfig, default_config, get_agent_model
from neo_core.memory.conversation import ConversationStore, ConversationSession
from neo_core.oauth import is_oauth_token
from neo_core.validation import validate_message, ValidationError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neo_core.brain.core import Brain
    from neo_core.memory.agent import MemoryAgent

VOX_SYSTEM_PROMPT = """Tu es Vox, l'interface de communication du système Neo Core.
Date et heure actuelles : {current_date}, {current_time}

Ton rôle :
- Tu es le point de contact entre l'humain et le système.
- Tu reformules les demandes de l'humain de manière claire et structurée avant de les transmettre à Brain.
- Tu restitues les réponses de Brain de manière naturelle et accessible.
- Tu peux répondre aux questions simples toi-même (salutations, conversation légère).
- Tu peux informer l'humain de l'état des autres agents si demandé.

Ce que Neo Core peut faire (informe l'utilisateur s'il demande) :
- Chercher sur internet (actualités, météo, prix, scores...)
- Écrire et exécuter du code (Python sandbox, debug, analyse)
- Lire et écrire des fichiers
- Rédiger des textes (articles, emails, rapports)
- Traduire dans toutes les langues
- Analyser des données et identifier des tendances
- Créer des tâches et des projets (missions multi-étapes)
- Mémoriser les préférences et apprendre des interactions
- Utiliser des plugins personnalisés

Règles :
- Sois naturel, empathique et efficace dans ta communication.
- Ne modifie jamais le fond des réponses de Brain, seulement la forme si nécessaire.
- Si une demande est ambiguë, pose des questions de clarification.
- Quand Brain travaille sur une tâche longue, tu peux continuer à discuter avec l'humain.
- Réponds de manière concise et naturelle, pas de markdown excessif.

{personality}

État actuel du système :
{system_status}
"""

# Prompt pour la reformulation intelligente des requêtes
VOX_REFORMULATE_PROMPT = """Tu es Vox, l'interface du système Neo Core.
Reformule cette demande utilisateur en une requête claire et structurée pour Brain (l'orchestrateur).

{recent_context}Demande originale : {user_message}

Règles :
- Conserve l'intention exacte de l'utilisateur
- Si la demande est courte ou utilise des pronoms ("ça", "pareil", "continue"),
  résous les références en utilisant le contexte récent ci-dessus
- Clarifie les ambiguïtés si possible
- Structure la demande pour faciliter le travail de Brain
- Reste concis (1-3 phrases max)
- Si la demande est déjà claire, retourne-la telle quelle
- Réponds UNIQUEMENT avec la requête reformulée, rien d'autre.
"""

# Acks statiques pour accusé de réception instantané
STATIC_ACKS = [
    "Je transmets à Brain, un instant...",
    "Compris, Brain analyse votre demande...",
    "Bien reçu, je traite votre requête...",
]

# Liste UNIQUE de pronoms/expressions ambigus nécessitant contexte pour désambiguïser
# Utilisée à la fois par process_message() (skip reformulation) et format_request_async()
_AMBIGUOUS_PRONOUNS = frozenset([
    "ça", "cela", "ce", "il", "elle", "les", "eux", "celle", "celui",
    "pareil", "même", "même chose", "le même", "la même", "les mêmes",
    "continue", "enchaîne", "enchaine", "la suite",
    "fais-le", "fais le", "fais-la", "fais la",
])


@dataclass
class AgentStatus:
    """État d'un agent dans le système."""
    name: str
    active: bool = False
    current_task: Optional[str] = None
    progress: float = 0.0

    def to_string(self) -> str:
        status = "actif" if self.active else "inactif"
        task = self.current_task or "aucune tâche"
        return f"[{self.name}] {status} — {task} ({self.progress:.0%})"


@dataclass
class Vox:
    """
    Agent Vox — Interface humaine du système Neo Core.

    Agit comme le cortex du langage : reçoit les messages humains,
    les structure, les transmet à Brain, et restitue les réponses.

    Possède son propre LLM (Haiku) pour :
    - Reformuler les requêtes avant transmission à Brain
    - Restituer les réponses de Brain de manière naturelle
    - Générer des accusés de réception instantanés
    """
    config: NeoConfig = field(default_factory=lambda: default_config)
    brain: Optional[Brain] = None
    memory: Optional[MemoryAgent] = None
    conversation_history: list = field(default_factory=list)
    _agent_statuses: dict[str, AgentStatus] = field(default_factory=dict)
    _llm: Optional[object] = None
    _force_mock: bool = False
    _model_config: Optional[object] = None
    _on_thinking_callback: Optional[object] = None  # Callable[[str], None]
    _on_brain_done_callback: Optional[object] = None  # Callable[[str], None] — async brain result
    _conversation_store: Optional[ConversationStore] = None
    _current_session: Optional[ConversationSession] = None
    _session_lock: Optional[object] = None  # threading.Lock
    _brain_task: Optional[object] = None  # asyncio.Task — tâche Brain en cours
    _brain_busy: bool = False  # True quand Brain travaille en arrière-plan

    def __post_init__(self):
        self._agent_statuses = {
            "Vox": AgentStatus(name="Vox", active=True, current_task="communication"),
            "Brain": AgentStatus(name="Brain"),
            "Memory": AgentStatus(name="Memory"),
        }
        self._model_config = get_agent_model("vox")

        # Lock pour protéger _current_session contre les accès concurrents
        import threading
        self._session_lock = threading.Lock()

        # Initialiser le conversation store (chemin relatif au data_dir de la config)
        db_path = self.config.memory.storage_path / "conversations.db"
        self._conversation_store = ConversationStore(db_path)

        if not self.config.is_mock_mode():
            self._init_llm()

    @property
    def _mock_mode(self) -> bool:
        """Vérifie dynamiquement si Vox est en mode mock.

        Re-vérifie la config à chaque appel. Si la clé API devient
        disponible après le démarrage, Vox s'auto-initialise.
        """
        if self._force_mock:
            return True
        if self.config.is_mock_mode():
            return True
        # La clé est disponible mais le LLM n'est pas encore initialisé
        if self._llm is None and not is_oauth_token(self.config.llm.api_key or ""):
            try:
                self._init_llm()
            except Exception:
                pass  # Vox fonctionne en passthrough sans LLM
        return False

    def _init_llm(self) -> None:
        """Initialise le LLM dédié de Vox (Haiku — rapide et léger)."""
        try:
            api_key = self.config.llm.api_key
            if is_oauth_token(api_key):
                # En mode OAuth, Vox utilise les appels directs httpx
                self._llm = None  # Sera géré via _vox_llm_call
            else:
                from langchain_anthropic import ChatAnthropic
                self._llm = ChatAnthropic(
                    model=self._model_config.model,
                    api_key=api_key,
                    temperature=self._model_config.temperature,
                    max_tokens=self._model_config.max_tokens,
                )
            logger.info("LLM initialisé : %s", self._model_config.model)
        except Exception as e:
            logger.error("LLM non disponible (%s), mode passthrough", e)

    async def _vox_llm_call(self, prompt: str) -> str:
        """
        Appel LLM dédié pour Vox (reformulation/restitution).

        Route via le système multi-provider (Ollama, Groq, Gemini, Anthropic).
        Fallback automatique vers Anthropic direct si aucun provider configuré.
        """
        if self._mock_mode:
            return prompt  # Passthrough en mock

        try:
            from neo_core.brain.providers.router import route_chat

            response = await route_chat(
                agent_name="vox",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._model_config.max_tokens,
                temperature=self._model_config.temperature,
            )

            if response.text and not response.text.startswith("[Erreur"):
                return response.text
            # Fallback : retourne le prompt original
            return prompt

        except Exception as e:
            # Fallback LangChain legacy
            logger.debug("route_chat failed, trying legacy LangChain: %s", e)
            if self._llm:
                try:
                    result = await self._llm.ainvoke(prompt)
                    return result.content
                except Exception as e:
                    logger.debug("LangChain ainvoke failed: %s", e)

        return prompt  # Passthrough si pas de LLM

    def connect(self, brain: Brain, memory: MemoryAgent) -> None:
        """Connecte Vox aux autres agents du Neo Core."""
        self.brain = brain
        self.memory = memory

    def start_new_session(self, user_name: str = "User") -> ConversationSession:
        """Démarre une nouvelle session de conversation."""
        if not self._conversation_store:
            return None
        with self._session_lock:
            self._current_session = self._conversation_store.start_session(user_name)
            logger.info("Started new conversation session: %s", self._current_session.session_id)
            return self._current_session

    def resume_session(self, session_id: str) -> Optional[ConversationSession]:
        """Charge une session existante et restaure son historique (thread-safe)."""
        if not self._conversation_store:
            return None
        with self._session_lock:
            session = self._conversation_store.get_session_by_id(session_id)
            if not session:
                logger.warning("Session not found: %s", session_id)
                return None

            # Charger l'historique
            history = self._conversation_store.get_history(session_id, limit=10000)
            self.conversation_history = []
            for turn in history:
                if turn.role == "human":
                    self.conversation_history.append(HumanMessage(content=turn.content))
                else:
                    self.conversation_history.append(AIMessage(content=turn.content))

            self._current_session = session
            logger.info("Resumed session: %s with %d messages", session_id, len(history))
            return self._current_session

    def get_session_info(self) -> Optional[dict]:
        """Retourne les infos de la session courante."""
        if not self._current_session:
            return None
        return {
            "session_id": self._current_session.session_id,
            "user_name": self._current_session.user_name,
            "created_at": self._current_session.created_at,
            "updated_at": self._current_session.updated_at,
            "message_count": self._current_session.message_count,
        }

    def set_thinking_callback(self, callback) -> None:
        """
        Définit un callback appelé quand Vox envoie un ack
        pendant que Brain réfléchit.

        Le callback reçoit un str (message d'accusé de réception).
        """
        self._on_thinking_callback = callback

    def set_brain_done_callback(self, callback) -> None:
        """
        Définit un callback appelé quand Brain termine en arrière-plan.
        Le callback reçoit un str (réponse de Brain).
        Utilisé en mode asynchrone pour délivrer le résultat au CLI.
        """
        self._on_brain_done_callback = callback

    def push_message(self, message: str, source: str = "brain") -> None:
        """
        Envoie un message proactif à l'utilisateur sur tous les canaux.

        Appelé par le heartbeat quand Brain décide de parler spontanément.
        Route vers : TUI (callback), Telegram, et stockage mémoire.
        """
        import logging as _log
        _logger = _log.getLogger(__name__)

        # 1. TUI / frontend via callback
        if self._on_brain_done_callback:
            try:
                self._on_brain_done_callback(f"🧠 {message}")
            except Exception as e:
                _logger.debug("push_message callback failed: %s", e)

        # 2. Telegram
        try:
            from neo_core.infra.registry import core_registry
            core_registry.send_telegram(f"🧠 {message}")
        except Exception as e:
            _logger.debug("push_message telegram failed: %s", e)

        # 3. Stockage mémoire (Brain se souvient de ce qu'il a dit)
        if self.memory and self.memory.is_initialized:
            try:
                self.memory.store_memory(
                    content=f"[Proactif — {source}] {message}",
                    source="brain_proactive",
                    tags=["proactive", source],
                    importance=0.6,
                )
            except Exception as e:
                _logger.debug("push_message memory store failed: %s", e)

        _logger.info("[Vox] Message proactif envoyé: %s", message[:80])

    @property
    def is_brain_busy(self) -> bool:
        """Indique si Brain travaille en arrière-plan."""
        return self._brain_busy

    def _is_simple_message(self, message: str) -> bool:
        """
        Détermine si un message peut être traité par Vox seul (sans Brain).

        Seuls deux cas restent en fast-path Vox :
        1. Commandes slash (/status, /tasks, etc.) — routage structurel
        2. Salutations pures isolées ("salut", "bonjour", "merci")

        Tout le reste est envoyé à Brain pour classification LLM.
        C'est le LLM qui décide, pas des heuristiques.
        """
        msg_lower = message.lower().strip()

        # Commandes slash → routage structurel, pas besoin de Brain
        if msg_lower.startswith("/"):
            return True

        # Salutations pures isolées (le message n'est QUE ça, sans contexte)
        import re as _re
        pure_greetings = (
            r"^(salut|hello|bonjour|bonsoir|hey|hi|yo|coucou|wesh|slt|"
            r"ça va|ca va|quoi de neuf|"
            r"merci|thanks)"
            r"\s*[!.\?]*$"
        )
        if _re.match(pure_greetings, msg_lower):
            return True

        # Tout le reste → Brain (le LLM décide)
        return False

    async def _vox_quick_reply(self, message: str) -> str:
        """
        Génère une réponse rapide de Vox (sans Brain) pour les messages simples.
        Si Brain travaille en arrière-plan, mentionne-le.
        """
        brain_status = ""
        if self._brain_busy:
            brain_status = "\n(Brain travaille encore sur ta demande précédente, je te donnerai la réponse dès qu'elle est prête.)"

        if self._mock_mode:
            return f"Salut ! Je suis Vox, l'interface de Neo Core. Je suis là pour t'aider !{brain_status}"

        try:
            import asyncio
            prompt = (
                f"Tu es Vox, l'interface de Neo Core. Réponds de manière courte et naturelle "
                f"à ce message de l'utilisateur (max 2-3 phrases). "
                f"Si on te demande tes capacités, explique ce que Neo peut faire : "
                f"chercher sur internet, écrire du code, analyser des données, rédiger des textes, "
                f"traduire, créer des tâches/projets, mémoriser et apprendre.\n\n"
                f"Message : {message}\n\n"
                f"{'Note: Brain travaille actuellement sur une tâche en arrière-plan.' if self._brain_busy else ''}"
            )
            reply = await asyncio.wait_for(
                self._vox_llm_call(prompt),
                timeout=5.0,
            )
            return reply.strip() + brain_status if reply else f"Je suis là !{brain_status}"
        except Exception:
            return f"Je suis là, comment puis-je t'aider ?{brain_status}"

    async def _generate_ack(self, user_message: str) -> str:
        """
        Génère un accusé de réception INSTANTANÉ (pas d'appel LLM).

        L'ACK doit être immédiat (<10ms) pour que le prompt revienne
        tout de suite. Brain tourne en fond, le résultat arrivera via callback.
        """
        import random

        # ACK contextuel basé sur des mots-clés simples (pas de LLM)
        msg_lower = user_message.lower()

        if any(w in msg_lower for w in ("cherche", "recherche", "trouve", "search", "look")):
            return random.choice(["🔍 Je lance la recherche...", "Je cherche ça...", "Recherche en cours..."])
        if any(w in msg_lower for w in ("code", "script", "programme", "debug", "fix")):
            return random.choice(["💻 Je m'en occupe...", "Je travaille dessus...", "Code en cours..."])
        if any(w in msg_lower for w in ("crée", "créer", "monte", "lance", "projet", "project")):
            return random.choice(["📋 Je mets ça en place...", "Projet en préparation...", "Je structure ça..."])
        if any(w in msg_lower for w in ("écris", "rédige", "traduis", "résume")):
            return random.choice(["✍️ J'y travaille...", "Rédaction en cours...", "Je m'en charge..."])
        if any(w in msg_lower for w in ("analyse", "calcule", "compare", "évalue")):
            return random.choice(["📊 Analyse en cours...", "Je calcule...", "Je regarde les données..."])

        return random.choice(STATIC_ACKS)

    def _get_personality_injection(self) -> str:
        """Récupère l'injection de personnalité depuis le PersonaEngine."""
        if self.memory and self.memory.persona_engine and self.memory.persona_engine.is_initialized:
            try:
                return self.memory.persona_engine.get_vox_injection()
            except Exception as e:
                logger.debug("Failed to get personality injection: %s", e)
        return ""

    def get_system_status(self) -> str:
        """Génère un résumé de l'état de tous les agents."""
        lines = [status.to_string() for status in self._agent_statuses.values()]
        return "\n".join(lines)

    def update_agent_status(self, name: str, active: bool = False,
                            task: Optional[str] = None, progress: float = 0.0) -> None:
        """Met à jour le statut d'un agent."""
        if name in self._agent_statuses:
            self._agent_statuses[name].active = active
            self._agent_statuses[name].current_task = task
            self._agent_statuses[name].progress = progress

    # ── Gestion intelligente de l'historique ──────────────────────
    _MAX_HISTORY_TOKENS = 12_000   # budget tokens pour l'historique conversation
    _SUMMARY_TRIGGER = 14_000      # seuil pour déclencher la compression
    _CHARS_PER_TOKEN = 4           # estimation fr/en (conservateur)

    def _estimate_tokens(self, text: str) -> int:
        """Estimation rapide du nombre de tokens (~4 chars/token)."""
        return len(text) // self._CHARS_PER_TOKEN

    def _history_token_count(self) -> int:
        """Compte les tokens estimés dans l'historique conversation."""
        return sum(self._estimate_tokens(m.content) for m in self.conversation_history)

    def _trim_history_by_tokens(self) -> None:
        """
        Gestion intelligente de l'historique basée sur un budget tokens.
        Au lieu de couper brutalement à 50 messages :
        1. Si < _MAX_HISTORY_TOKENS → on garde tout
        2. Si > _SUMMARY_TRIGGER → on résume les vieux messages et on ne garde
           que le résumé + les messages récents
        """
        total = self._history_token_count()
        if total <= self._MAX_HISTORY_TOKENS:
            return  # tout tient dans le budget

        # Stratégie : garder les 20 derniers messages intacts,
        # résumer tout ce qui précède en un seul message "contexte"
        keep_recent = 20
        if len(self.conversation_history) <= keep_recent:
            return  # pas assez de messages pour compresser

        old_messages = self.conversation_history[:-keep_recent]
        recent_messages = self.conversation_history[-keep_recent:]

        # Construire un résumé condensé des vieux messages
        summary_parts = []
        for msg in old_messages:
            role = "User" if isinstance(msg, HumanMessage) else "Neo"
            # Garder les 200 premiers chars de chaque message
            content = msg.content[:200]
            if len(msg.content) > 200:
                content += "..."
            summary_parts.append(f"{role}: {content}")

        summary_text = (
            "[Résumé des échanges précédents]\n"
            + "\n".join(summary_parts[-30:])  # max 30 entrées dans le résumé
        )

        # Remplacer l'historique : résumé + messages récents
        self.conversation_history = [
            AIMessage(content=summary_text),
            *recent_messages,
        ]
        logger.info(
            "[VOX] History compressed: %d messages → summary + %d recent (was %d tokens, now ~%d)",
            len(old_messages), len(recent_messages), total, self._history_token_count()
        )

    def format_request(self, human_message: str) -> str:
        """
        Reformule et structure la demande humaine.
        Version synchrone (fallback) — retourne tel quel.
        """
        return human_message

    def _build_recent_context(self, max_turns: int = 8) -> str:
        """
        Construit un résumé des derniers échanges pour contextualiser
        les messages courts ou ambigus (pronoms, références implicites).
        """
        if not self.conversation_history:
            return ""

        # Prendre les derniers échanges (max_turns * 2 messages)
        recent = self.conversation_history[-(max_turns * 2):]
        if not recent:
            return ""

        lines = ["Contexte récent de la conversation :"]
        for msg in recent:
            role = "User" if isinstance(msg, HumanMessage) else "Neo"
            # Tronquer les réponses longues pour ne pas saturer le prompt
            content = msg.content[:400] + "..." if len(msg.content) > 400 else msg.content
            lines.append(f"  {role}: {content}")
        lines.append("")  # Ligne vide avant la demande
        return "\n".join(lines)

    async def format_request_async(self, human_message: str) -> str:
        """
        Reformule intelligemment la demande via le LLM de Vox (Haiku).
        Injecte le contexte récent pour résoudre les pronoms et références.
        Si le LLM n'est pas disponible, retourne le message tel quel.
        """
        if self._mock_mode or (not self._llm and not is_oauth_token(self.config.llm.api_key or "")):
            return human_message

        try:
            # Injecter le contexte récent pour les messages courts/ambigus ou avec pronoms
            recent_context = ""
            msg_lower = human_message.lower()
            _has_pronoun = any(p in msg_lower for p in _AMBIGUOUS_PRONOUNS)
            if (len(human_message.split()) < 20 or _has_pronoun) and self.conversation_history:
                recent_context = self._build_recent_context()

            prompt = VOX_REFORMULATE_PROMPT.format(
                user_message=human_message,
                recent_context=recent_context,
            )
            result = await self._vox_llm_call(prompt)
            return result.strip() if result else human_message
        except Exception as e:
            logger.debug("Failed to reformat request: %s", e)
            return human_message

    async def process_message(self, human_message: str) -> str:
        """
        Pipeline principal : humain → Vox(LLM) → Brain(LLM) → humain.

        Mode synchrone (par défaut) :
        1. Si message simple → Vox répond directement (sans Brain)
        2. Si message complexe → Vox reformule → Brain traite → retourne réponse

        Mode asynchrone (si _on_brain_done_callback est défini) :
        1. Si message simple → Vox répond immédiatement
        2. Si message complexe → Lance Brain en arrière-plan, retourne un ack
        3. Quand Brain termine → callback avec la réponse
        """
        # Input validation
        try:
            human_message = validate_message(human_message)
        except ValidationError as e:
            logger.error("Message validation failed: %s", e)
            return f"[Erreur] Message invalide : {e}"

        if not self.brain:
            return "[Erreur] Brain n'est pas connecté à Vox."

        # Enregistre le message dans l'historique (budget tokens, pas message count)
        self.conversation_history.append(HumanMessage(content=human_message))
        self._trim_history_by_tokens()

        # ── Slash commands → routage structurel sans Brain ──
        if human_message.strip().startswith("/"):
            # Les commandes slash sont gérées par le legacy pipeline de Brain
            pass  # Tombe dans le flux normal ci-dessous

        # ── TOUT passe à Brain — plus de reformulation, plus de Vox LLM ──
        # Brain (Sonnet) a l'historique complet, il comprend "ça", "pareil", "salut"
        # Zéro intermédiaire = zéro perte d'intelligence
        formatted_request = human_message

        # Mode asynchrone : lancer Brain en arrière-plan
        if self._on_brain_done_callback:
            import asyncio

            # Accusé de réception
            ack = await self._generate_ack(human_message)
            self.update_agent_status("Vox", active=False, task="communication", progress=0.0)
            self.update_agent_status("Brain", active=True, task="analyse de la requête", progress=0.5)

            # Lancer Brain en arrière-plan
            self._brain_busy = True
            self._brain_task = asyncio.create_task(
                self._brain_background(formatted_request, human_message)
            )

            return ack

        # Mode synchrone (fallback / API / Telegram) : attendre Brain
        if self._on_thinking_callback:
            try:
                ack = await self._generate_ack(human_message)
                self._on_thinking_callback(ack)
            except Exception as e:
                logger.debug("Failed to send thinking callback: %s", e)

        self.update_agent_status("Vox", active=False, task="communication", progress=0.0)
        self.update_agent_status("Brain", active=True, task="analyse de la requête", progress=0.5)

        brain_response = await self.brain.process(
            request=formatted_request,
            conversation_history=self.conversation_history,
            original_request=human_message,
        )

        self.update_agent_status("Brain", active=False, task=None, progress=0.0)
        final_response = brain_response

        self.conversation_history.append(AIMessage(content=final_response))
        self._save_conversation_turn(human_message, final_response)
        self._store_in_memory(human_message, final_response)

        return final_response

    async def _brain_background(self, formatted_request: str, original_message: str) -> None:
        """Exécute Brain en arrière-plan et délivre le résultat via callback."""
        try:
            brain_response = await self.brain.process(
                request=formatted_request,
                conversation_history=self.conversation_history,
                original_request=original_message,
            )

            self.update_agent_status("Brain", active=False, task=None, progress=0.0)

            # Enregistrer dans l'historique
            self.conversation_history.append(AIMessage(content=brain_response))
            self._save_conversation_turn(original_message, brain_response)
            self._store_in_memory(original_message, brain_response)

            # Délivrer via callback
            if self._on_brain_done_callback:
                self._on_brain_done_callback(brain_response)

        except Exception as e:
            logger.error("Brain background task failed: %s", e)
            error_msg = f"[Erreur Brain] {type(e).__name__}: {str(e)[:300]}"
            if self._on_brain_done_callback:
                self._on_brain_done_callback(error_msg)
        finally:
            self._brain_busy = False
            self._brain_task = None

    def _save_conversation_turn(self, human_message: str, response: str) -> None:
        """Sauvegarde un échange dans le ConversationStore."""
        if self._conversation_store and self._current_session:
            try:
                self._conversation_store.append_turn(
                    self._current_session.session_id,
                    "human",
                    human_message,
                )
                self._conversation_store.append_turn(
                    self._current_session.session_id,
                    "assistant",
                    response,
                )
            except Exception as e:
                logger.error("Failed to save conversation turn: %s", e)

    def _store_in_memory(self, human_message: str, response: str) -> None:
        """Stocke l'échange en mémoire persistante et met à jour la working memory."""
        if self.memory and self.memory.is_initialized:
            self.update_agent_status("Memory", active=True, task="stockage", progress=0.5)
            self.memory.on_conversation_turn(human_message, response)
            self.update_agent_status("Memory", active=False, task=None, progress=0.0)

    def get_prompt_template(self) -> ChatPromptTemplate:
        """Retourne le template de prompt pour Vox avec personnalité injectée."""
        personality = self._get_personality_injection()
        prompt = VOX_SYSTEM_PROMPT.replace("{personality}", personality)
        return ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder("conversation_history"),
            ("human", "{input}"),
        ])

    async def generate_session_summary(self) -> None:
        """
        Génère un résumé de la session courante via LLM Haiku.

        Appelé à la fin de chaque session (quit, Ctrl+C).
        Le résumé est stocké dans ConversationStore.session_summaries.
        Skippé si la session a < 3 messages.
        """
        if not self._conversation_store or not self._current_session:
            return

        session_id = self._current_session.session_id
        session = self._conversation_store.get_session_by_id(session_id)
        if not session or session.message_count < 3:
            logger.debug("Session too short for summary (%d messages)", session.message_count if session else 0)
            return

        try:
            import asyncio

            # Récupérer les turns formatés
            turns_text = self._conversation_store.get_session_turns_for_summary(session_id)
            if not turns_text:
                return

            summary_prompt = f"""Résume cette conversation en 2-3 phrases concises.
Extrais aussi les sujets clés, les décisions prises, et les faits importants appris sur l'utilisateur.

Conversation :
{turns_text[:3000]}

Réponds UNIQUEMENT en JSON :
{{"summary": "résumé concis de la conversation", "key_topics": ["sujet1", "sujet2"], "key_decisions": ["décision1"], "key_facts": ["fait1"]}}"""

            # Appel Haiku rapide (timeout 5s)
            response_text = await asyncio.wait_for(
                self._vox_llm_call(summary_prompt),
                timeout=5.0,
            )

            if response_text:
                import json
                # Parser le JSON (tolérant aux backticks markdown)
                cleaned = response_text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    cleaned = cleaned.strip()

                data = json.loads(cleaned)
                self._conversation_store.save_session_summary(
                    session_id=session_id,
                    summary=data.get("summary", ""),
                    key_topics=data.get("key_topics", []),
                    key_decisions=data.get("key_decisions", []),
                    key_facts=data.get("key_facts", []),
                )
                logger.info("[Vox] Session summary generated for %s", session_id[:8])

        except Exception as e:
            logger.warning("[Vox] Failed to generate session summary: %s", e)

    def get_model_info(self) -> dict:
        """Retourne les infos du modèle utilisé par Vox."""
        return {
            "agent": "Vox",
            "model": self._model_config.model if self._model_config else "none",
            "role": "Interface utilisateur (reformulation/restitution)",
            "has_llm": self._llm is not None or (
                not self._mock_mode and is_oauth_token(self.config.llm.api_key or "")
            ),
        }


def _warmup_providers() -> None:
    """
    Warm-up des providers LLM au démarrage.

    Envoie un appel minimal à Haiku pour ouvrir la connexion réseau.
    Ça évite le cold start sur la première vraie requête utilisateur,
    qui ferait timeout le classifier (8s).
    """
    import asyncio

    async def _ping():
        try:
            from neo_core.brain.providers.router import route_chat
            await asyncio.wait_for(
                route_chat(
                    agent_name="vox",
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5,
                    temperature=0.0,
                ),
                timeout=10.0,
            )
            logger.info("[Warmup] Provider connection established")
        except Exception as e:
            logger.warning("[Warmup] Provider warmup failed (non-blocking): %s", e)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # On est déjà dans une boucle async → lancer en background
            asyncio.ensure_future(_ping())
        else:
            loop.run_until_complete(_ping())
    except RuntimeError:
        # Pas de loop → en créer une
        asyncio.run(_ping())


def bootstrap() -> Vox:
    """
    Initialise et connecte les 3 agents du Neo Core.
    Retourne l'agent Vox prêt à communiquer.
    """
    from neo_core.brain.core import Brain
    from neo_core.memory.agent import MemoryAgent

    config = NeoConfig()

    memory = MemoryAgent(config=config)
    memory.initialize()

    brain = Brain(config=config)
    brain.connect_memory(memory)

    vox = Vox(config=config)
    vox.connect(brain=brain, memory=memory)

    # Warm-up : ouvre la connexion provider pour éviter le cold start
    if not config.is_mock_mode():
        _warmup_providers()

    return vox
