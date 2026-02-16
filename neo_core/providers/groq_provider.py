"""
Neo Core — GroqProvider : Provider cloud gratuit (Groq)
========================================================
Gratuit, 14 400 req/jour, inférence ultra-rapide (LPU).
API compatible OpenAI — facile à intégrer.

Modèles : Llama 3.3 70B, Llama 3 8B, etc.
Auth : Clé API gratuite via console.groq.com
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


class GroqProvider(LLMProvider):
    """Provider pour l'API Groq (cloud gratuit, ultra-rapide)."""

    API_BASE = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")

    @property
    def name(self) -> str:
        return "groq"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.CLOUD_FREE

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_id="groq:llama-3.3-70b-versatile",
                provider="groq",
                model_name="llama-3.3-70b-versatile",
                display_name="Llama 3.3 70B (Groq)",
                capability=ModelCapability.ADVANCED,
                context_window=128_000,
                max_output_tokens=32_768,
                is_free=True,
                is_local=False,
                supports_tools=True,
                supports_streaming=True,
            ),
            ModelInfo(
                model_id="groq:llama3-8b-8192",
                provider="groq",
                model_name="llama3-8b-8192",
                display_name="Llama 3 8B (Groq)",
                capability=ModelCapability.STANDARD,
                context_window=8_192,
                max_output_tokens=8_192,
                is_free=True,
                is_local=False,
                supports_tools=True,
                supports_streaming=True,
            ),
            ModelInfo(
                model_id="groq:gemma2-9b-it",
                provider="groq",
                model_name="gemma2-9b-it",
                display_name="Gemma 2 9B (Groq)",
                capability=ModelCapability.STANDARD,
                context_window=8_192,
                max_output_tokens=8_192,
                is_free=True,
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
        """
        Appel chat via l'API Groq (format OpenAI).

        L'API Groq est 100% compatible OpenAI :
        - /chat/completions avec messages, tools, etc.
        """
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Format OpenAI pour les messages
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(messages)

        payload = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Convertir les tools Anthropic → format OpenAI
        if tools:
            payload["tools"] = self._convert_tools_to_openai(tools)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.API_BASE}/chat/completions",
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
            raise RuntimeError(f"Groq HTTP {response.status_code}: {error_msg}")

        data = response.json()
        return self._parse_openai_response(data, model)

    def _parse_openai_response(self, data: dict, model: str) -> ChatResponse:
        """Parse une réponse format OpenAI en format unifié."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        text = message.get("content", "") or ""
        finish_reason = choice.get("finish_reason", "stop")

        # Tool calls (format OpenAI)
        tool_calls = []
        for tc in message.get("tool_calls", []):
            import json
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}

            tool_calls.append({
                "name": func.get("name", ""),
                "input": args,
                "id": tc.get("id", f"groq_{func.get('name', 'unknown')}"),
            })

        stop_reason = "end_turn"
        if finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif finish_reason == "length":
            stop_reason = "max_tokens"

        usage = data.get("usage", {})

        return ChatResponse(
            text=text,
            model=model,
            provider="groq",
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
            raw_response=data,
        )

    def _convert_tools_to_openai(self, anthropic_tools: list[dict]) -> list[dict]:
        """
        Convertit les schémas tool_use Anthropic → OpenAI function calling.

        Anthropic : {"name": "...", "description": "...", "input_schema": {...}}
        OpenAI    : {"type": "function", "function": {"name": "...", "parameters": {...}}}
        """
        openai_tools = []
        for tool in anthropic_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        return openai_tools
