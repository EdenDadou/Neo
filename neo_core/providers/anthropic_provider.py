"""
Neo Core — AnthropicProvider : Provider Claude (Anthropic)
==========================================================
Provider payant — Claude Sonnet, Haiku, Opus.
Supporte le tool_use natif.

Auth : API Key classique ou OAuth Bearer (Claude Code).
"""

from __future__ import annotations

import os
from typing import Optional

from neo_core.providers.base import (
    LLMProvider,
    ProviderType,
    ModelInfo,
    ModelCapability,
    ChatResponse,
)


class AnthropicProvider(LLMProvider):
    """Provider pour l'API Anthropic (Claude)."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.CLOUD_PAID

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_id="anthropic:claude-sonnet-4-5-20250929",
                provider="anthropic",
                model_name="claude-sonnet-4-5-20250929",
                display_name="Claude Sonnet 4.5",
                capability=ModelCapability.ADVANCED,
                context_window=200_000,
                max_output_tokens=8192,
                is_free=False,
                is_local=False,
                supports_tools=True,
                supports_streaming=True,
            ),
            ModelInfo(
                model_id="anthropic:claude-haiku-4-5-20251001",
                provider="anthropic",
                model_name="claude-haiku-4-5-20251001",
                display_name="Claude Haiku 4.5",
                capability=ModelCapability.STANDARD,
                context_window=200_000,
                max_output_tokens=8192,
                is_free=False,
                is_local=False,
                supports_tools=True,
                supports_streaming=True,
            ),
        ]

    async def chat(
        self,
        messages: list[dict],
        model: str,
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> ChatResponse:
        import httpx
        from neo_core.oauth import is_oauth_token, get_valid_access_token, OAUTH_BETA_HEADER

        headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        api_key = self._api_key
        if is_oauth_token(api_key):
            valid_token = get_valid_access_token() or api_key
            headers["Authorization"] = f"Bearer {valid_token}"
            headers["anthropic-beta"] = OAUTH_BETA_HEADER
        else:
            headers["x-api-key"] = api_key

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=60,
            )

        if response.status_code != 200:
            error_data = {}
            try:
                error_data = response.json()
            except Exception:
                pass
            error_msg = error_data.get("error", {}).get("message", response.text[:300])
            raise RuntimeError(f"Anthropic HTTP {response.status_code}: {error_msg}")

        data = response.json()
        return self._parse_response(data, model)

    def _parse_response(self, data: dict, model: str) -> ChatResponse:
        """Parse la réponse Anthropic en format unifié."""
        text_parts = []
        tool_calls = []

        for block in data.get("content", []):
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "name": block["name"],
                    "input": block["input"],
                    "id": block["id"],
                })

        return ChatResponse(
            text="\n".join(text_parts),
            model=model,
            provider="anthropic",
            stop_reason=data.get("stop_reason", "end_turn"),
            tool_calls=tool_calls,
            usage=data.get("usage", {}),
            raw_response=data,
        )
