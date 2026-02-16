"""
Neo Core — GeminiProvider : Provider cloud gratuit (Google Gemini)
==================================================================
Gratuit, contexte 1M tokens, bon pour l'analyse de gros documents.
Rate limit réduit sur le free tier (10-15 RPM, 250-1000 RPD).

Modèles : Gemini 2.5 Flash, Flash-Lite, Pro
Auth : Clé API gratuite via Google AI Studio (aistudio.google.com)
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


class GeminiProvider(LLMProvider):
    """Provider pour l'API Google Gemini (cloud gratuit)."""

    API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.CLOUD_FREE

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_id="gemini:gemini-2.5-flash",
                provider="gemini",
                model_name="gemini-2.5-flash",
                display_name="Gemini 2.5 Flash",
                capability=ModelCapability.ADVANCED,
                context_window=1_000_000,
                max_output_tokens=8_192,
                is_free=True,
                is_local=False,
                supports_tools=True,
                supports_streaming=True,
            ),
            ModelInfo(
                model_id="gemini:gemini-2.0-flash-lite",
                provider="gemini",
                model_name="gemini-2.0-flash-lite",
                display_name="Gemini 2.0 Flash-Lite",
                capability=ModelCapability.STANDARD,
                context_window=1_000_000,
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
        Appel chat via l'API REST Gemini.

        L'API Gemini utilise un format propre :
        - /models/{model}:generateContent
        - Contents avec "parts" au lieu de "content"
        """
        import httpx

        url = f"{self.API_BASE}/models/{model}:generateContent?key={self._api_key}"

        # Convertir messages au format Gemini
        gemini_contents = self._convert_messages(messages, system)

        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        # System instruction (Gemini a un champ dédié)
        if system:
            payload["systemInstruction"] = {
                "parts": [{"text": system}]
            }

        # Tools
        if tools:
            payload["tools"] = self._convert_tools_to_gemini(tools)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
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
            raise RuntimeError(f"Gemini HTTP {response.status_code}: {error_msg}")

        data = response.json()
        return self._parse_response(data, model)

    def _convert_messages(self, messages: list[dict], system: str = "") -> list[dict]:
        """Convertit les messages format standard → format Gemini."""
        gemini_contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Gemini n'a pas de rôle "system" dans les contents
            if role == "system":
                continue

            # Gemini utilise "user" et "model" (pas "assistant")
            gemini_role = "model" if role == "assistant" else "user"

            if isinstance(content, str):
                gemini_contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}]
                })
            elif isinstance(content, list):
                # Messages multi-parts (tool results, etc.)
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append({"text": block["text"]})
                        elif block.get("type") == "tool_result":
                            parts.append({
                                "functionResponse": {
                                    "name": block.get("tool_use_id", "unknown"),
                                    "response": {"content": block.get("content", "")},
                                }
                            })
                if parts:
                    gemini_contents.append({"role": gemini_role, "parts": parts})

        return gemini_contents

    def _parse_response(self, data: dict, model: str) -> ChatResponse:
        """Parse la réponse Gemini en format unifié."""
        candidates = data.get("candidates", [])
        if not candidates:
            return ChatResponse(
                text="",
                model=model,
                provider="gemini",
                stop_reason="end_turn",
            )

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        text_parts = []
        tool_calls = []

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "name": fc.get("name", ""),
                    "input": fc.get("args", {}),
                    "id": f"gemini_{fc.get('name', 'unknown')}",
                })

        finish_reason = candidate.get("finishReason", "STOP")
        stop_reason = "end_turn"
        if tool_calls:
            stop_reason = "tool_use"
        elif finish_reason == "MAX_TOKENS":
            stop_reason = "max_tokens"

        usage = data.get("usageMetadata", {})

        return ChatResponse(
            text="\n".join(text_parts),
            model=model,
            provider="gemini",
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            usage={
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0),
            },
            raw_response=data,
        )

    def _convert_tools_to_gemini(self, anthropic_tools: list[dict]) -> list[dict]:
        """
        Convertit les schémas tool_use Anthropic → format Gemini.

        Anthropic : {"name": "...", "description": "...", "input_schema": {...}}
        Gemini    : {"functionDeclarations": [{"name": "...", "parameters": {...}}]}
        """
        declarations = []
        for tool in anthropic_tools:
            declarations.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            })

        return [{"functionDeclarations": declarations}]
