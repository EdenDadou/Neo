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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from neo_core.config import NeoConfig, default_config, get_agent_model
from neo_core.oauth import is_oauth_token, get_valid_access_token, OAUTH_BETA_HEADER

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
    """
    config: NeoConfig = field(default_factory=lambda: default_config)
    brain: Optional[Brain] = None
    memory: Optional[MemoryAgent] = None
    conversation_history: list = field(default_factory=list)
    _agent_statuses: dict[str, AgentStatus] = field(default_factory=dict)
    _llm: Optional[object] = None
    _mock_mode: bool = False
    _model_config: Optional[object] = None

    def __post_init__(self):
        self._agent_statuses = {
            "Vox": AgentStatus(name="Vox", active=True, current_task="communication"),
            "Brain": AgentStatus(name="Brain"),
            "Memory": AgentStatus(name="Memory"),
        }
        self._mock_mode = self.config.is_mock_mode()
        self._model_config = get_agent_model("vox")

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
            print(f"[Vox] LLM initialisé : {self._model_config.model}")
        except Exception as e:
            print(f"[Vox] LLM non disponible ({e}), mode passthrough")
            self._llm = None

    async def _vox_llm_call(self, prompt: str) -> str:
        """Appel LLM dédié pour Vox (reformulation/restitution)."""
        if self._mock_mode:
            return prompt  # Passthrough en mock

        api_key = self.config.llm.api_key

        if is_oauth_token(api_key):
            # Mode OAuth direct
            import httpx
            valid_token = get_valid_access_token() or api_key
            headers = {
                "Authorization": f"Bearer {valid_token}",
                "anthropic-beta": OAUTH_BETA_HEADER,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": self._model_config.model,
                "max_tokens": self._model_config.max_tokens,
                "temperature": self._model_config.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                )
            if response.status_code == 200:
                data = response.json()
                return data["content"][0]["text"]
            # Fallback : retourne le prompt original
            return prompt

        elif self._llm:
            result = await self._llm.ainvoke(prompt)
            return result.content

        return prompt  # Passthrough si pas de LLM

    def connect(self, brain: Brain, memory: MemoryAgent) -> None:
        """Connecte Vox aux autres agents du Neo Core."""
        self.brain = brain
        self.memory = memory

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
        except Exception:
            return human_message

    async def process_message(self, human_message: str) -> str:
        """
        Pipeline principal : humain → Vox(LLM) → Brain(LLM) → humain.

        1. Reçoit le message humain
        2. Vox reformule intelligemment la requête (Haiku)
        3. Transmet à Brain (Sonnet)
        4. Retourne la réponse de Brain telle quelle
        """
        if not self.brain:
            return "[Erreur] Brain n'est pas connecté à Vox."

        # Enregistre le message dans l'historique
        self.conversation_history.append(HumanMessage(content=human_message))

        # Met à jour les statuts
        self.update_agent_status("Vox", active=True, task="reformulation", progress=0.3)

        # Vox reformule intelligemment la requête
        formatted_request = await self.format_request_async(human_message)

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

        # Stocke l'échange en mémoire persistante
        if self.memory and self.memory.is_initialized:
            self.update_agent_status("Memory", active=True, task="stockage", progress=0.5)
            self.memory.on_conversation_turn(human_message, final_response)
            self.update_agent_status("Memory", active=False, task=None, progress=0.0)

        return final_response

    def get_prompt_template(self) -> ChatPromptTemplate:
        """Retourne le template de prompt pour Vox."""
        return ChatPromptTemplate.from_messages([
            ("system", VOX_SYSTEM_PROMPT),
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
