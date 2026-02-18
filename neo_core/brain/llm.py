"""
Brain LLM — Initialisation & Génération de réponses
=====================================================
Fonctions d'initialisation LLM (OAuth, LangChain) et de
génération de réponses (router, OAuth, mock).

Chaque fonction reçoit l'instance Brain en premier argument.
Extrait de brain.py pour séparer l'infra LLM de l'orchestration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from neo_core.brain.prompts import BRAIN_SYSTEM_PROMPT
from neo_core.oauth import (
    is_oauth_token,
    get_valid_access_token,
    get_api_key_from_oauth,
    OAUTH_BETA_HEADER,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ─── Initialisation LLM / Auth ──────────────────────────

def init_llm(brain) -> None:
    """Initialise le LLM avec la meilleure méthode d'auth."""
    try:
        api_key = brain.config.llm.api_key
        if is_oauth_token(api_key):
            init_oauth(brain, api_key)
        else:
            init_langchain(brain, api_key)
    except Exception as e:
        logger.error("Impossible d'initialiser le LLM: %s", e)
        brain._mock_mode = True
        brain._auth_method = "mock"


def init_oauth(brain, token: str) -> None:
    """Init OAuth avec fallback automatique."""
    converted_key = get_api_key_from_oauth()
    if converted_key:
        logger.info("Clé API convertie depuis OAuth détectée")
        init_langchain(brain, converted_key)
        brain._auth_method = "converted_api_key"
        return
    brain._oauth_mode = True
    init_oauth_bearer(brain, token)


def init_oauth_bearer(brain, token: str) -> None:
    """Init Bearer + beta header (méthode OpenClaw)."""
    import anthropic
    import httpx  # noqa: F401

    valid_token = get_valid_access_token()
    if not valid_token:
        valid_token = token

    brain._anthropic_client = anthropic.AsyncAnthropic(
        api_key="dummy",
        default_headers={
            "Authorization": f"Bearer {valid_token}",
            "anthropic-beta": OAUTH_BETA_HEADER,
        },
    )
    brain._current_token = valid_token
    brain._auth_method = "oauth_bearer"
    logger.info("Mode OAuth Bearer + beta header activé")


def refresh_oauth_client(brain) -> bool:
    """Rafraîchit le client OAuth."""
    converted_key = get_api_key_from_oauth()
    if converted_key:
        init_langchain(brain, converted_key)
        brain._oauth_mode = False
        brain._auth_method = "converted_api_key"
        return True
    valid_token = get_valid_access_token()
    if valid_token:
        init_oauth_bearer(brain, valid_token)
        return True
    return False


def init_langchain(brain, api_key: str) -> None:
    """Init LangChain avec clé API classique et modèle dédié Brain."""
    from langchain_anthropic import ChatAnthropic
    brain._llm = ChatAnthropic(
        model=brain._model_config.model,
        api_key=api_key,
        temperature=brain._model_config.temperature,
        max_tokens=brain._model_config.max_tokens,
    )
    brain._oauth_mode = False
    if not brain._auth_method:
        brain._auth_method = "langchain"
    logger.info("LLM initialisé : %s", brain._model_config.model)


# ─── Appels LLM ────────────────────────────────────────

async def raw_llm_call(brain, prompt: str) -> str:
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

        if brain._health:
            brain._health.record_api_call(
                success=not response.text.startswith("[Erreur")
            )

        if response.text and not response.text.startswith("[Erreur"):
            return response.text

        raise Exception(response.text)

    except Exception as e:
        # Fallback LangChain legacy
        logger.debug("Appel LLM via router échoué, utilisation de LangChain: %s", e)
        if brain._llm:
            result = await brain._llm.ainvoke(prompt)
            if brain._health:
                brain._health.record_api_call(success=True)
            return result.content
        raise


async def oauth_response(brain, request: str, memory_context: str,
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
    if brain.memory and brain.memory.persona_engine and brain.memory.persona_engine.is_initialized:
        try:
            user_context = brain.memory.persona_engine.get_brain_injection()
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
            max_tokens=brain._model_config.max_tokens,
            temperature=brain._model_config.temperature,
        )

        if brain._health:
            brain._health.record_api_call(
                success=not response.text.startswith("[Erreur")
            )

        if response.text and not response.text.startswith("[Erreur"):
            return response.text

        return f"[Brain Erreur] {response.text}"

    except Exception as e:
        logger.error("Erreur dans oauth_response: %s: %s", type(e).__name__, str(e)[:200])
        if brain._health:
            brain._health.record_api_call(success=False)
        return f"[Brain Erreur] {type(e).__name__}: {str(e)[:200]}"


async def llm_response(brain, request: str, memory_context: str,
                       conversation_history: list[BaseMessage] | None = None) -> str:
    """Génère une réponse via LangChain."""
    from datetime import datetime
    now = datetime.now()

    # Stage 9 — Injection du contexte utilisateur
    user_context = ""
    if brain.memory and brain.memory.persona_engine and brain.memory.persona_engine.is_initialized:
        try:
            user_context = brain.memory.persona_engine.get_brain_injection()
        except Exception as e:
            logger.debug("Impossible de récupérer le contexte utilisateur: %s", e)

    prompt = ChatPromptTemplate.from_messages([
        ("system", BRAIN_SYSTEM_PROMPT),
        MessagesPlaceholder("conversation_history", optional=True),
        ("human", "{request}"),
    ])
    chain = prompt | brain._llm
    result = await chain.ainvoke({
        "memory_context": memory_context,
        "user_context": user_context,
        "current_date": now.strftime("%A %d %B %Y"),
        "current_time": now.strftime("%H:%M"),
        "conversation_history": conversation_history or [],
        "request": request,
    })
    if brain._health:
        brain._health.record_api_call(success=True)
    return result.content


def mock_response(request: str, complexity: str, context: str) -> str:
    """Réponse mock pour les tests sans clé API."""
    return (
        f"[Brain Mock] Requête reçue (complexité: {complexity}). "
        f"Contexte: {context[:100]}. "
        f"Analyse de: '{request[:80]}...'" if len(request) > 80 else
        f"[Brain Mock] Requête reçue (complexité: {complexity}). "
        f"Contexte: {context[:100]}. "
        f"Analyse de: '{request}'"
    )
