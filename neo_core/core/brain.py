"""
Brain — Agent Orchestrateur
============================
Cortex exécutif du système Neo Core.

Responsabilités :
- Analyser les requêtes transmises par Vox
- Consulter le contexte enrichi par Memory
- Déterminer l'action optimale
- (Étape 3+) Générer dynamiquement des agents spécialisés

Authentification OAuth (méthode OpenClaw) :
- Méthode 1 : Bearer token + header beta "anthropic-beta: oauth-2025-04-20"
- Méthode 2 : Conversion OAuth → API key via /claude_cli/create_api_key
- Fallback : Clé API classique via LangChain
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from neo_core.config import NeoConfig, default_config
from neo_core.oauth import (
    is_oauth_token,
    get_valid_access_token,
    get_best_auth,
    get_api_key_from_oauth,
    OAUTH_BETA_HEADER,
)

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

    Supporte 3 modes d'authentification :
    1. OAuth Bearer + beta header (méthode OpenClaw directe)
    2. Clé API convertie depuis OAuth (méthode OpenClaw/pi-mono)
    3. Clé API classique via LangChain
    """
    config: NeoConfig = field(default_factory=lambda: default_config)
    memory: Optional[MemoryAgent] = None
    _llm: Optional[object] = None
    _mock_mode: bool = False
    _oauth_mode: bool = False
    _auth_method: str = ""  # "oauth_bearer" | "converted_api_key" | "langchain" | "mock"
    _anthropic_client: Optional[object] = None  # SDK Anthropic direct pour OAuth

    def __post_init__(self):
        self._mock_mode = self.config.is_mock_mode()
        if not self._mock_mode:
            self._init_llm()

    def _init_llm(self) -> None:
        """
        Initialise le LLM avec la meilleure méthode d'auth disponible.

        Ordre de priorité pour les tokens OAuth :
        1. Clé API déjà convertie (cache) → LangChain classique
        2. Bearer + beta header → SDK Anthropic direct + httpx custom
        3. Conversion OAuth → API Key → LangChain classique
        4. Clé API classique → LangChain

        Pour les clés API classiques :
        → LangChain ChatAnthropic directement
        """
        try:
            api_key = self.config.llm.api_key

            if is_oauth_token(api_key):
                self._init_oauth(api_key)
            else:
                # Clé API classique → LangChain
                self._init_langchain(api_key)
        except Exception as e:
            print(f"[Brain] Impossible d'initialiser le LLM: {e}")
            self._mock_mode = True
            self._auth_method = "mock"

    def _init_oauth(self, token: str) -> None:
        """
        Initialise l'authentification OAuth avec fallback automatique.

        Essaie dans l'ordre :
        1. Clé API convertie (cache) → LangChain
        2. Bearer + beta header → httpx direct
        3. Conversion OAuth → API Key → LangChain
        """
        # Essayer d'abord de récupérer une clé API convertie
        converted_key = get_api_key_from_oauth()
        if converted_key:
            print(f"[Brain] Clé API convertie depuis OAuth détectée")
            self._init_langchain(converted_key)
            self._auth_method = "converted_api_key"
            return

        # Pas de clé convertie → Bearer + beta header (méthode OpenClaw)
        self._oauth_mode = True
        self._init_oauth_bearer(token)

    def _init_oauth_bearer(self, token: str) -> None:
        """
        Initialise le client Anthropic avec Bearer auth + beta header.

        C'est la méthode OpenClaw : on utilise le SDK Anthropic avec
        auth_token et on ajoute le header beta "oauth-2025-04-20"
        via un httpx custom qui injecte les headers nécessaires.
        """
        import anthropic
        import httpx

        # Récupérer un token valide (refresh si expiré)
        valid_token = get_valid_access_token()
        if not valid_token:
            valid_token = token

        # Créer un httpx client custom qui ajoute le beta header OAuth
        custom_transport = httpx.AsyncHTTPTransport()

        self._anthropic_client = anthropic.AsyncAnthropic(
            api_key="dummy",  # Sera overridé par le header Authorization
            default_headers={
                "Authorization": f"Bearer {valid_token}",
                "anthropic-beta": OAUTH_BETA_HEADER,
            },
        )
        # Supprimer x-api-key pour éviter le conflit
        # Le SDK va utiliser notre header Authorization à la place
        self._current_token = valid_token
        self._auth_method = "oauth_bearer"
        print(f"[Brain] Mode OAuth Bearer + beta header activé")

    def _refresh_oauth_client(self) -> bool:
        """Rafraîchit le client OAuth si le token a expiré."""
        # D'abord, vérifier si une clé API convertie est apparue
        converted_key = get_api_key_from_oauth()
        if converted_key:
            self._init_langchain(converted_key)
            self._oauth_mode = False
            self._auth_method = "converted_api_key"
            return True

        # Sinon, rafraîchir le token Bearer
        valid_token = get_valid_access_token()
        if valid_token:
            self._init_oauth_bearer(valid_token)
            return True
        return False

    def _init_langchain(self, api_key: str) -> None:
        """Initialise LangChain avec une clé API classique."""
        from langchain_anthropic import ChatAnthropic
        self._llm = ChatAnthropic(
            model=self.config.llm.model,
            api_key=api_key,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )
        self._oauth_mode = False
        if not self._auth_method:
            self._auth_method = "langchain"

    def connect_memory(self, memory: MemoryAgent) -> None:
        """Connecte Brain au système mémoire."""
        self.memory = memory

    def get_memory_context(self, request: str) -> str:
        """Récupère le contexte pertinent depuis Memory."""
        if self.memory:
            return self.memory.get_context(request)
        return "Aucun contexte mémoire disponible."

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

    async def process(self, request: str,
                      conversation_history: list[BaseMessage] | None = None) -> str:
        """
        Pipeline principal de Brain.

        1. Récupère le contexte mémoire
        2. Analyse la complexité
        3. Génère une réponse (LLM réel ou mock)
        """
        memory_context = self.get_memory_context(request)
        complexity = self.analyze_complexity(request)

        if self._mock_mode:
            return self._mock_response(request, complexity, memory_context)

        try:
            if self._oauth_mode:
                return await self._oauth_response(request, memory_context, conversation_history)
            else:
                return await self._llm_response(request, memory_context, conversation_history)
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            # Si erreur d'auth en mode OAuth → tenter refresh/conversion et retry
            if self._oauth_mode and ("authentication" in error_msg.lower()
                                     or "unauthorized" in error_msg.lower()
                                     or "401" in error_msg):
                print(f"[Brain] Erreur auth OAuth, tentative de fallback...")

                # Fallback 1 : Essayer la conversion en API key
                converted_key = get_api_key_from_oauth()
                if converted_key:
                    print(f"[Brain] Conversion OAuth → API key réussie, retry...")
                    self._init_langchain(converted_key)
                    self._oauth_mode = False
                    self._auth_method = "converted_api_key"
                    try:
                        return await self._llm_response(request, memory_context, conversation_history)
                    except Exception as retry_e:
                        return f"[Brain Erreur] Après conversion: {type(retry_e).__name__}: {str(retry_e)[:300]}"

                # Fallback 2 : Refresh du token et retry Bearer
                if self._refresh_oauth_client():
                    try:
                        if self._oauth_mode:
                            return await self._oauth_response(request, memory_context, conversation_history)
                        else:
                            return await self._llm_response(request, memory_context, conversation_history)
                    except Exception as retry_e:
                        return f"[Brain Erreur] Après refresh: {type(retry_e).__name__}: {str(retry_e)[:300]}"

            return f"[Brain Erreur] {error_type}: {error_msg[:500]}"

    async def _oauth_response(self, request: str, memory_context: str,
                              conversation_history: list[BaseMessage] | None = None) -> str:
        """
        Génère une réponse via le SDK Anthropic avec OAuth Bearer + beta header.

        C'est la méthode OpenClaw : Authorization: Bearer <token>
        avec le header anthropic-beta: oauth-2025-04-20
        """
        import httpx as _httpx

        # Construire les messages
        messages = []

        # Historique de conversation
        if conversation_history:
            for msg in conversation_history:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})

        # Message actuel
        messages.append({"role": "user", "content": request})

        # System prompt avec contexte mémoire
        system_prompt = BRAIN_SYSTEM_PROMPT.replace("{memory_context}", memory_context)

        # Rafraîchir le token si nécessaire
        valid_token = get_valid_access_token()
        if valid_token:
            token = valid_token
        else:
            token = getattr(self, "_current_token", None)
            if not token:
                return "[Brain Erreur] Aucun token OAuth valide disponible"

        # Appel API direct via httpx (contournement complet du SDK pour contrôle total)
        headers = {
            "Authorization": f"Bearer {token}",
            "anthropic-beta": OAUTH_BETA_HEADER,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self.config.llm.model,
            "max_tokens": self.config.llm.max_tokens,
            "temperature": self.config.llm.temperature,
            "system": system_prompt,
            "messages": messages,
        }

        async with _httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=120,
            )

        if response.status_code == 200:
            data = response.json()
            return data["content"][0]["text"]
        else:
            # Erreur → lever une exception pour déclencher le fallback
            error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            error_msg = error_data.get("error", {}).get("message", response.text[:300])
            raise Exception(f"API {response.status_code}: {error_msg}")

    async def _llm_response(self, request: str, memory_context: str,
                            conversation_history: list[BaseMessage] | None = None) -> str:
        """Génère une réponse via LangChain (clé API classique ou convertie)."""
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
                action="delegate_worker",
                subtasks=self._decompose_task(request),
                confidence=0.5,
            )

    def _decompose_task(self, request: str) -> list[str]:
        """
        Décompose une tâche complexe en sous-tâches.
        Placeholder pour l'étape 3.
        """
        return [f"Sous-tâche: {request}"]
