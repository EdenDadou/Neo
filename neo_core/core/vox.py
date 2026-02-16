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

from neo_core.config import NeoConfig, default_config

if TYPE_CHECKING:
    from neo_core.core.brain import Brain
    from neo_core.core.memory_agent import MemoryAgent

VOX_SYSTEM_PROMPT = """Tu es Vox, l'interface de communication du système Neo Core.

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

État actuel du système :
{system_status}
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
    """
    config: NeoConfig = field(default_factory=lambda: default_config)
    brain: Optional[Brain] = None
    memory: Optional[MemoryAgent] = None
    conversation_history: list = field(default_factory=list)
    _agent_statuses: dict[str, AgentStatus] = field(default_factory=dict)

    def __post_init__(self):
        self._agent_statuses = {
            "Vox": AgentStatus(name="Vox", active=True, current_task="communication"),
            "Brain": AgentStatus(name="Brain"),
            "Memory": AgentStatus(name="Memory"),
        }

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
        En étape 1, on transmet tel quel. Sera enrichi avec Memory en étape 2.
        """
        return human_message

    async def process_message(self, human_message: str) -> str:
        """
        Pipeline principal : humain → Vox → Brain → Vox → humain.

        1. Reçoit le message humain
        2. Le formate/structure
        3. Le transmet à Brain
        4. Restitue la réponse
        """
        if not self.brain:
            return "[Erreur] Brain n'est pas connecté à Vox."

        # Enregistre le message dans l'historique
        self.conversation_history.append(HumanMessage(content=human_message))

        # Met à jour les statuts
        self.update_agent_status("Brain", active=True, task="analyse de la requête", progress=0.5)

        # Formate la requête
        formatted_request = self.format_request(human_message)

        # Transmet à Brain et récupère la réponse
        brain_response = await self.brain.process(
            request=formatted_request,
            conversation_history=self.conversation_history,
        )

        # Met à jour les statuts
        self.update_agent_status("Brain", active=False, task=None, progress=0.0)

        # Enregistre la réponse dans l'historique
        self.conversation_history.append(AIMessage(content=brain_response))

        # Stocke l'échange en mémoire persistante (Étape 2)
        if self.memory and self.memory.is_initialized:
            self.update_agent_status("Memory", active=True, task="stockage", progress=0.5)
            self.memory.on_conversation_turn(human_message, brain_response)
            self.update_agent_status("Memory", active=False, task=None, progress=0.0)

        return brain_response

    def get_prompt_template(self) -> ChatPromptTemplate:
        """Retourne le template de prompt pour Vox."""
        return ChatPromptTemplate.from_messages([
            ("system", VOX_SYSTEM_PROMPT),
            MessagesPlaceholder("conversation_history"),
            ("human", "{input}"),
        ])
