"""
Brain — Agent Orchestrateur
============================
Cortex exécutif du système Neo Core.

Responsabilités :
- Analyser les requêtes transmises par Vox
- Consulter le contexte enrichi par Memory
- Déterminer l'action optimale
- (Étape 3+) Générer dynamiquement des agents spécialisés
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from neo_core.config import NeoConfig, default_config

if TYPE_CHECKING:
    from neo_core.core.memory_agent import MemoryAgent

BRAIN_SYSTEM_PROMPT = """Tu es Brain, le cortex exécutif du système Neo Core.

Ton rôle :
- Tu reçois les requêtes structurées par Vox (l'interface humaine).
- Tu analyses chaque requête et détermines la meilleure stratégie de réponse.
- Tu consultes le contexte fourni par Memory pour enrichir tes réponses.
- Tu coordonnes l'exécution des tâches.

Contexte mémoire :
{memory_context}

Règles :
- Sois précis, stratégique et orienté résultat.
- Si une tâche est complexe, décompose-la en sous-tâches.
- Indique clairement quand tu as besoin de plus d'informations.
- Tu es le décideur final sur la stratégie d'exécution.
"""


@dataclass
class BrainDecision:
    """Représente une décision prise par Brain."""
    action: str  # "direct_response" | "delegate_worker" | "delegate_crew"
    response: Optional[str] = None
    subtasks: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class Brain:
    """
    Agent Brain — Orchestrateur du système Neo Core.

    Analyse les requêtes, consulte Memory, et détermine
    la meilleure action à entreprendre.
    """
    config: NeoConfig = field(default_factory=lambda: default_config)
    memory: Optional[MemoryAgent] = None
    _llm: Optional[object] = None
    _mock_mode: bool = False

    def __post_init__(self):
        self._mock_mode = self.config.is_mock_mode()
        if not self._mock_mode:
            self._init_llm()

    @staticmethod
    def _is_oauth_token(key: str) -> bool:
        """Détecte si la clé est un token OAuth (sk-ant-oat...) vs API key (sk-ant-api...)."""
        return key.startswith("sk-ant-oat")

    def _init_llm(self) -> None:
        """
        Initialise le LLM réel via LangChain.
        Supporte les clés API classiques ET les tokens OAuth Anthropic.

        Pour les tokens OAuth (sk-ant-oat...), on crée le client Anthropic
        directement avec auth_token et on le passe à ChatAnthropic.
        """
        try:
            from langchain_anthropic import ChatAnthropic
            import anthropic

            api_key = self.config.llm.api_key

            if self._is_oauth_token(api_key):
                # Token OAuth → créer le client Anthropic directement
                oauth_token = api_key
                if self.config.debug:
                    print(f"[Brain] Mode OAuth détecté (token: {api_key[:15]}...)")

                class _ChatAnthropicOAuth(ChatAnthropic):
                    """ChatAnthropic modifié pour l'authentification OAuth Bearer."""

                    @property
                    def _client_params(self) -> dict:
                        return {
                            "api_key": None,
                            "auth_token": oauth_token,
                            "base_url": self.anthropic_api_url,
                            "max_retries": self.max_retries,
                            "default_headers": (self.default_headers or None),
                            "timeout": self.default_request_timeout,
                        }

                self._llm = _ChatAnthropicOAuth(
                    model=self.config.llm.model,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                    anthropic_api_key="placeholder",
                )
            else:
                # Clé API classique
                if self.config.debug:
                    print(f"[Brain] Mode API Key classique (clé: {api_key[:10]}...)")
                self._llm = ChatAnthropic(
                    model=self.config.llm.model,
                    api_key=api_key,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                )
        except Exception as e:
            print(f"[Brain] Impossible d'initialiser le LLM: {e}")
            self._mock_mode = True

    def connect_memory(self, memory: MemoryAgent) -> None:
        """Connecte Brain au système mémoire."""
        self.memory = memory

    def get_memory_context(self, request: str) -> str:
        """
        Récupère le contexte pertinent depuis Memory.
        En étape 1, retourne un contexte vide. Sera enrichi en étape 2.
        """
        if self.memory:
            return self.memory.get_context(request)
        return "Aucun contexte mémoire disponible."

    def analyze_complexity(self, request: str) -> str:
        """
        Analyse la complexité d'une requête.
        Retourne : "simple" | "moderate" | "complex"
        Sera enrichi en étape 3 pour la délégation.
        """
        # Heuristique simple pour l'étape 1
        word_count = len(request.split())
        if word_count < 15:
            return "simple"
        elif word_count < 50:
            return "moderate"
        return "complex"

    async def process(self, request: str,
                      conversation_history: list[BaseMessage] | None = None) -> str:
        """
        Pipeline principal de Brain.

        1. Récupère le contexte mémoire
        2. Analyse la complexité
        3. Génère une réponse (LLM réel ou mock)
        """
        # Contexte mémoire
        memory_context = self.get_memory_context(request)

        # Analyse
        complexity = self.analyze_complexity(request)

        if self._mock_mode:
            return self._mock_response(request, complexity, memory_context)

        # Réponse LLM réelle
        return await self._llm_response(request, memory_context, conversation_history)

    async def _llm_response(self, request: str, memory_context: str,
                            conversation_history: list[BaseMessage] | None = None) -> str:
        """Génère une réponse via le LLM réel."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", BRAIN_SYSTEM_PROMPT),
            MessagesPlaceholder("conversation_history", optional=True),
            ("human", "{request}"),
        ])

        chain = prompt | self._llm

        result = await chain.ainvoke({
            "memory_context": memory_context,
            "conversation_history": conversation_history or [],
            "request": request,
        })

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

    def make_decision(self, request: str) -> BrainDecision:
        """
        Prend une décision stratégique sur la manière de traiter la requête.
        Sera enrichi en étape 3 avec la génération d'agents.
        """
        complexity = self.analyze_complexity(request)

        if complexity == "simple":
            return BrainDecision(action="direct_response", confidence=0.9)
        elif complexity == "moderate":
            return BrainDecision(
                action="direct_response",
                subtasks=[request],
                confidence=0.7,
            )
        else:
            return BrainDecision(
                action="delegate_worker",  # Sera implémenté en étape 3
                subtasks=self._decompose_task(request),
                confidence=0.5,
            )

    def _decompose_task(self, request: str) -> list[str]:
        """
        Décompose une tâche complexe en sous-tâches.
        Placeholder pour l'étape 3.
        """
        return [f"Sous-tâche: {request}"]
