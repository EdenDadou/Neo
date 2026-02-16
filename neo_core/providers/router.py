"""
Neo Core — LLM Router : Couche unifiée de routing des appels LLM
================================================================
Point d'entrée unique pour tous les agents (Brain, Worker, Vox, Memory).

Le router :
1. Consulte le ModelRegistry pour la chaîne de fallback ordonnée
2. Tente chaque provider dans l'ordre de priorité
3. Si un provider échoue (rate limit, timeout, erreur) → passe au suivant
4. Dernier recours : fallback Anthropic direct (OAuth/API key)
5. Retourne un format unifié (ChatResponse)
"""

from __future__ import annotations

import logging
from typing import Optional

from neo_core.providers.base import ChatResponse

logger = logging.getLogger(__name__)


async def route_chat(
    agent_name: str,
    messages: list[dict],
    system: str = "",
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> ChatResponse:
    """
    Route un appel LLM vers le meilleur provider disponible.

    Chaîne de fallback :
    1. Provider prioritaire selon le registry (ex: Ollama pour Workers simples)
    2. Provider suivant dans la chaîne (ex: Groq)
    3. Provider suivant (ex: Gemini)
    4. Provider suivant (ex: Anthropic via registry)
    5. Dernier recours : Anthropic direct (OAuth/API key)

    Chaque échec est loggué et le provider suivant est tenté.
    """
    errors = []

    # Obtenir la chaîne de fallback depuis le registry
    try:
        from neo_core.providers.registry import get_model_registry

        registry = get_model_registry()
        require_tools = bool(tools)
        fallback_chain = registry.get_fallback_chain(
            agent_name, require_tools=require_tools
        )

        # Tenter chaque modèle dans l'ordre
        for model in fallback_chain:
            provider = registry.get_provider(model.provider)
            if not provider:
                continue

            try:
                logger.debug(
                    f"[Router] {agent_name} → {model.model_id} "
                    f"({model.provider})"
                )
                response = await provider.chat(
                    messages=messages,
                    model=model.model_name,
                    system=system,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                # Vérifier que la réponse est valide (pas une erreur)
                if response.text and not response.text.startswith("[Erreur"):
                    return response

                # Réponse d'erreur du provider → noter et continuer
                error_msg = f"{model.model_id}: {response.text[:100]}"
                errors.append(error_msg)
                logger.warning(f"[Router] Échec {error_msg}, fallback...")

            except Exception as e:
                error_msg = f"{model.model_id}: {type(e).__name__}: {str(e)[:100]}"
                errors.append(error_msg)
                logger.warning(f"[Router] Échec {error_msg}, fallback...")
                continue

    except Exception as e:
        errors.append(f"Registry: {e}")
        logger.debug(f"[Router] Registry indisponible: {e}")

    # Dernier recours : Anthropic direct (OAuth ou API key)
    logger.debug(f"[Router] Tous les providers ont échoué, fallback Anthropic direct")
    response = await _fallback_anthropic(
        agent_name=agent_name,
        messages=messages,
        system=system,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Si même Anthropic direct échoue, enrichir le message d'erreur
    if response.text.startswith("[Erreur"):
        all_errors = " | ".join(errors) if errors else "aucun provider disponible"
        response.text = f"[Erreur] Chaîne de fallback épuisée: {all_errors}"

    return response


async def route_chat_raw(
    agent_name: str,
    messages: list[dict],
    system: str = "",
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> dict | None:
    """
    Comme route_chat(), mais retourne le dict brut de la réponse.

    Utilisé par Worker pour sa boucle tool_use native qui a besoin
    du format brut (content blocks, stop_reason, etc.).

    Returns:
        Dict brut de la réponse, ou None si échec.
    """
    response = await route_chat(
        agent_name=agent_name,
        messages=messages,
        system=system,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Convertir ChatResponse → format dict brut Anthropic-like
    content = []

    # Texte
    if response.text:
        content.append({"type": "text", "text": response.text})

    # Tool calls
    for tc in response.tool_calls:
        content.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": tc.get("name", ""),
            "input": tc.get("input", {}),
        })

    return {
        "content": content,
        "stop_reason": response.stop_reason,
        "model": response.model,
        "usage": response.usage,
    }


async def _fallback_anthropic(
    agent_name: str,
    messages: list[dict],
    system: str = "",
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> ChatResponse:
    """
    Dernier recours : Anthropic direct via httpx.

    Supporte OAuth Bearer et API key classique.
    Utilisé uniquement si TOUS les providers du registry ont échoué.
    """
    import httpx
    import os

    from neo_core.config import get_agent_model

    model_config = get_agent_model(agent_name)

    # Récupérer la clé API
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ChatResponse(
            text="[Erreur] Aucune clé API disponible",
            model=model_config.model,
            provider="anthropic",
            stop_reason="error",
        )

    # Headers
    headers = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        from neo_core.oauth import is_oauth_token, get_valid_access_token, OAUTH_BETA_HEADER

        if is_oauth_token(api_key):
            valid_token = get_valid_access_token() or api_key
            headers["Authorization"] = f"Bearer {valid_token}"
            headers["anthropic-beta"] = OAUTH_BETA_HEADER
        else:
            headers["x-api-key"] = api_key
    except ImportError:
        headers["x-api-key"] = api_key

    # Payload
    payload = {
        "model": model_config.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }

    if system:
        payload["system"] = system

    if tools:
        payload["tools"] = tools

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=60.0,
            )

        if response.status_code == 200:
            data = response.json()
            return _parse_anthropic_response(data, model_config.model)

        return ChatResponse(
            text=f"[Erreur API] HTTP {response.status_code}",
            model=model_config.model,
            provider="anthropic",
            stop_reason="error",
        )

    except Exception as e:
        return ChatResponse(
            text=f"[Erreur] {type(e).__name__}: {str(e)[:200]}",
            model=model_config.model,
            provider="anthropic",
            stop_reason="error",
        )


def _parse_anthropic_response(data: dict, model: str) -> ChatResponse:
    """Parse une réponse brute Anthropic → ChatResponse."""
    text_parts = []
    tool_calls = []

    for block in data.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "input": block.get("input", {}),
            })

    return ChatResponse(
        text="\n".join(text_parts),
        model=data.get("model", model),
        provider="anthropic",
        stop_reason=data.get("stop_reason", "end_turn"),
        tool_calls=tool_calls,
        usage=data.get("usage", {}),
        raw_response=data,
    )
