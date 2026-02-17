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
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from neo_core.config import NeoConfig, default_config, get_agent_model
from neo_core.memory.conversation import ConversationStore, ConversationSession
from neo_core.oauth import is_oauth_token, get_valid_access_token, OAUTH_BETA_HEADER
from neo_core.validation import validate_message, ValidationError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neo_core.core.brain import Brain
    from neo_core.core.memory_agent import MemoryAgent

VOX_SYSTEM_PROMPT = """Tu es Vox, l'interface de communication du système Neo Core.
Date et heure actuelles : {current_date}, {current_time}

Ton rôle :
- Tu es le point de contact entre l'humain et le système.
- Tu reformules les demandes de l'humain de manière claire et structurée avant de les transmettre à Brain.
- Tu restitues les réponses de Brain de manière naturelle et accessible.
- Tu peux informer l'humain de l'état des autres agents si demandé.

Règles :
- Sois naturel, empathique et efficace dans ta communication.
- Ne modifie jamais le fond des réponses de Brain, seulement la forme si nécessaire.
- Si une demande est ambiguë, pose des questions de clarification.
- Tu peux signaler à l'humain quand Brain travaille sur une tâche longue.
- Réponds de manière concise et naturelle, pas de markdown excessif.

{personality}

État actuel du système :
{system_status}
"""

# Prompt pour la reformulation intelligente des requêtes
VOX_REFORMULATE_PROMPT = """Tu es Vox, l'interface du système Neo Core.
Reformule cette demande utilisateur en une requête claire et structurée pour Brain (l'orchestrateur).

Demande originale : {user_message}

Règles :
- Conserve l'intention exacte de l'utilisateur
- Clarifie les ambiguïtés si possible
- Structure la demande pour faciliter le travail de Brain
- Reste concis (1-3 phrases max)
- Si la demande est déjà claire, retourne-la telle quelle
- Réponds UNIQUEMENT avec la requête reformulée, rien d'autre.
"""

# Prompt pour l'accusé de réception instantané
VOX_ACK_PROMPT = """Tu es Vox, l'interface du système Neo Core.
L'utilisateur vient d'envoyer un message. Génère un court accusé de réception
(max 15 mots) pour lui dire que tu as compris et que Brain travaille dessus.

Message : {user_message}

Règles :
- Sois naturel et rassurant
- Maximum 15 mots
- Pas de markdown, pas d'emojis
- Montre que tu as compris le sujet
- Réponds UNIQUEMENT avec l'accusé de réception, rien d'autre.
"""

# Acks statiques (fallback si LLM échoue)
STATIC_ACKS = [
    "Je transmets à Brain, un instant...",
    "Compris, Brain analyse votre demande...",
    "Bien reçu, je traite votre requête...",
]


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
    _mock_mode: bool = False
    _model_config: Optional[object] = None
    _on_thinking_callback: Optional[object] = None  # Callable[[str], None]
    _conversation_store: Optional[ConversationStore] = None
    _current_session: Optional[ConversationSession] = None
    _session_lock: Optional[object] = None  # threading.Lock

    def __post_init__(self):
        self._agent_statuses = {
            "Vox": AgentStatus(name="Vox", active=True, current_task="communication"),
            "Brain": AgentStatus(name="Brain"),
            "Memory": AgentStatus(name="Memory"),
        }
        self._mock_mode = self.config.is_mock_mode()
        self._model_config = get_agent_model("vox")

        # Lock pour protéger _current_session contre les accès concurrents
        import threading
        self._session_lock = threading.Lock()

        # Initialiser le conversation store
        db_path = Path("data/memory/conversations.db")
        self._conversation_store = ConversationStore(db_path)

        if not self._mock_mode:
            self._init_llm()

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
            from neo_core.providers.router import route_chat

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

    async def _generate_ack(self, user_message: str) -> str:
        """
        Génère un accusé de réception instantané via LLM.
        Fallback sur un ack statique si le LLM échoue ou est trop lent.
        """
        import random

        if self._mock_mode:
            return random.choice(STATIC_ACKS)

        try:
            import asyncio
            prompt = VOX_ACK_PROMPT.format(user_message=user_message[:100])

            # Timeout court pour ne pas bloquer — l'ack doit être rapide
            ack = await asyncio.wait_for(
                self._vox_llm_call(prompt),
                timeout=3.0,
            )

            if ack and len(ack.strip()) < 100:
                return ack.strip()

        except asyncio.TimeoutError as e:
            logger.debug("Acknowledgment generation timeout: %s", e)
        except Exception as e:
            logger.debug("Acknowledgment generation failed: %s", e)

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

    def format_request(self, human_message: str) -> str:
        """
        Reformule et structure la demande humaine.
        Version synchrone (fallback) — retourne tel quel.
        """
        return human_message

    async def format_request_async(self, human_message: str) -> str:
        """
        Reformule intelligemment la demande via le LLM de Vox (Haiku).
        Si le LLM n'est pas disponible, retourne le message tel quel.
        """
        if self._mock_mode or (not self._llm and not is_oauth_token(self.config.llm.api_key or "")):
            return human_message

        try:
            prompt = VOX_REFORMULATE_PROMPT.format(user_message=human_message)
            result = await self._vox_llm_call(prompt)
            return result.strip() if result else human_message
        except Exception as e:
            logger.debug("Failed to reformat request: %s", e)
            return human_message

    async def process_message(self, human_message: str) -> str:
        """
        Pipeline principal : humain → Vox(LLM) → Brain(LLM) → humain.

        1. Reçoit le message humain
        2. Vox reformule intelligemment la requête (Haiku)
        3. Envoie un accusé de réception instantané (callback)
        4. Transmet à Brain (Sonnet)
        5. Retourne la réponse de Brain telle quelle
        """
        # Input validation
        try:
            human_message = validate_message(human_message)
        except ValidationError as e:
            logger.error("Message validation failed: %s", e)
            return f"[Erreur] Message invalide : {e}"

        if not self.brain:
            return "[Erreur] Brain n'est pas connecté à Vox."

        # Enregistre le message dans l'historique (borné à 200 messages max)
        self.conversation_history.append(HumanMessage(content=human_message))
        if len(self.conversation_history) > 200:
            self.conversation_history = self.conversation_history[-200:]

        # Met à jour les statuts
        self.update_agent_status("Vox", active=True, task="reformulation", progress=0.3)

        # Vox reformule intelligemment la requête
        formatted_request = await self.format_request_async(human_message)

        # Accusé de réception instantané — feedback immédiat à l'utilisateur
        if self._on_thinking_callback:
            try:
                ack = await self._generate_ack(human_message)
                self._on_thinking_callback(ack)
            except Exception as e:
                logger.debug("Failed to send thinking callback: %s", e)

        self.update_agent_status("Vox", active=False, task="communication", progress=0.0)
        self.update_agent_status("Brain", active=True, task="analyse de la requête", progress=0.5)

        # Transmet à Brain et récupère la réponse
        brain_response = await self.brain.process(
            request=formatted_request,
            conversation_history=self.conversation_history,
        )

        # Met à jour les statuts
        self.update_agent_status("Brain", active=False, task=None, progress=0.0)

        # Retourne la réponse de Brain telle quelle (pas de réécriture)
        final_response = brain_response

        # Enregistre la réponse dans l'historique
        self.conversation_history.append(AIMessage(content=final_response))

        # Sauvegarde dans le ConversationStore
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
                    final_response,
                )
            except Exception as e:
                logger.error("Failed to save conversation turn: %s", e)

        # Stocke l'échange en mémoire persistante
        if self.memory and self.memory.is_initialized:
            self.update_agent_status("Memory", active=True, task="stockage", progress=0.5)
            self.memory.on_conversation_turn(human_message, final_response)
            self.update_agent_status("Memory", active=False, task=None, progress=0.0)

        return final_response

    def get_prompt_template(self) -> ChatPromptTemplate:
        """Retourne le template de prompt pour Vox avec personnalité injectée."""
        personality = self._get_personality_injection()
        prompt = VOX_SYSTEM_PROMPT.replace("{personality}", personality)
        return ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder("conversation_history"),
            ("human", "{input}"),
        ])

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
