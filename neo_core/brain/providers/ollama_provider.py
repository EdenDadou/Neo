"""
Neo Core — OllamaProvider : Provider local (Ollama)
=====================================================
100% gratuit, illimité, privé.
Tourne sur la machine — limité par le hardware (RAM/GPU).

API : REST sur localhost:11434 (natif) ou /v1/ (OpenAI compat)
Modèles : Llama 3.1/3.2, DeepSeek-R1, Mistral, Gemma, Qwen, etc.
"""

from __future__ import annotations

import os
from typing import Optional

from neo_core.brain.providers.base import (
    LLMProvider,
    ProviderType,
    ModelInfo,
    ModelCapability,
    ChatResponse,
)


# Catalogue des modèles Ollama connus avec leurs capabilities
_OLLAMA_MODEL_CATALOG = {
    # DeepSeek-R1 — raisonnement
    "deepseek-r1:1.5b": {
        "display_name": "DeepSeek-R1 1.5B",
        "capability": ModelCapability.BASIC,
        "context_window": 64_000,
        "supports_tools": False,
    },
    "deepseek-r1:8b": {
        "display_name": "DeepSeek-R1 8B",
        "capability": ModelCapability.STANDARD,
        "context_window": 64_000,
        "supports_tools": False,
    },
    "deepseek-r1:14b": {
        "display_name": "DeepSeek-R1 14B",
        "capability": ModelCapability.ADVANCED,
        "context_window": 64_000,
        "supports_tools": False,
    },
    "deepseek-r1:32b": {
        "display_name": "DeepSeek-R1 32B",
        "capability": ModelCapability.ADVANCED,
        "context_window": 64_000,
        "supports_tools": False,
    },
    # Llama 3.x — général
    "llama3.2:3b": {
        "display_name": "Llama 3.2 3B",
        "capability": ModelCapability.BASIC,
        "context_window": 128_000,
        "supports_tools": True,
    },
    "llama3.1:8b": {
        "display_name": "Llama 3.1 8B",
        "capability": ModelCapability.STANDARD,
        "context_window": 128_000,
        "supports_tools": True,
    },
    # Mistral
    "mistral:7b": {
        "display_name": "Mistral 7B",
        "capability": ModelCapability.STANDARD,
        "context_window": 32_000,
        "supports_tools": True,
    },
    # Gemma
    "gemma2:2b": {
        "display_name": "Gemma 2 2B",
        "capability": ModelCapability.BASIC,
        "context_window": 8_000,
        "supports_tools": False,
    },
    "gemma2:9b": {
        "display_name": "Gemma 2 9B",
        "capability": ModelCapability.STANDARD,
        "context_window": 8_000,
        "supports_tools": False,
    },
    # Qwen
    "qwen2.5:0.5b": {
        "display_name": "Qwen 2.5 0.5B",
        "capability": ModelCapability.BASIC,
        "context_window": 32_000,
        "supports_tools": False,
    },
    "qwen2.5:7b": {
        "display_name": "Qwen 2.5 7B",
        "capability": ModelCapability.STANDARD,
        "context_window": 128_000,
        "supports_tools": True,
    },
}


class OllamaProvider(LLMProvider):
    """Provider pour Ollama (modèles locaux)."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.LOCAL

    def is_configured(self) -> bool:
        """Ollama est configuré si le serveur répond."""
        try:
            import httpx
            response = httpx.get(f"{self._base_url}/api/tags", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[ModelInfo]:
        """Liste les modèles installés localement."""
        try:
            import httpx
            response = httpx.get(f"{self._base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return []

            data = response.json()
            models = []

            for m in data.get("models", []):
                name = m.get("name", "")
                # Chercher dans le catalogue, sinon créer une entrée générique
                catalog_entry = _OLLAMA_MODEL_CATALOG.get(name, None)

                if catalog_entry:
                    models.append(ModelInfo(
                        model_id=f"ollama:{name}",
                        provider="ollama",
                        model_name=name,
                        display_name=catalog_entry["display_name"],
                        capability=catalog_entry["capability"],
                        context_window=catalog_entry["context_window"],
                        max_output_tokens=4096,
                        is_free=True,
                        is_local=True,
                        supports_tools=catalog_entry.get("supports_tools", False),
                        supports_streaming=True,
                    ))
                else:
                    # Modèle non catalogué → capability standard par défaut
                    models.append(ModelInfo(
                        model_id=f"ollama:{name}",
                        provider="ollama",
                        model_name=name,
                        display_name=f"Ollama {name}",
                        capability=ModelCapability.STANDARD,
                        context_window=4096,
                        max_output_tokens=4096,
                        is_free=True,
                        is_local=True,
                        supports_tools=False,
                        supports_streaming=True,
                    ))

            return models

        except Exception:
            return []

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
        Appel chat via l'API native Ollama.

        L'API /api/chat accepte :
        - model, messages, system, stream, options
        - tools (depuis Ollama 0.4+)
        """
        import httpx

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system:
            # Injecter le system message en premier
            payload["messages"] = [
                {"role": "system", "content": system},
                *messages,
            ]

        # Tool use si supporté par le modèle
        if tools:
            ollama_tools = self._convert_tools_to_ollama(tools)
            if ollama_tools:
                payload["tools"] = ollama_tools

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=120,  # Les modèles locaux peuvent être lents
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama HTTP {response.status_code}: {response.text[:300]}"
            )

        data = response.json()
        return self._parse_response(data, model)

    def _parse_response(self, data: dict, model: str) -> ChatResponse:
        """Parse la réponse Ollama en format unifié."""
        message = data.get("message", {})
        text = message.get("content", "")

        # Tool calls (format Ollama)
        tool_calls = []
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            tool_calls.append({
                "name": func.get("name", ""),
                "input": func.get("arguments", {}),
                "id": f"ollama_{func.get('name', 'unknown')}",
            })

        stop_reason = "tool_use" if tool_calls else "end_turn"
        if data.get("done_reason") == "length":
            stop_reason = "max_tokens"

        return ChatResponse(
            text=text,
            model=model,
            provider="ollama",
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            usage={
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            },
            raw_response=data,
        )

    def _convert_tools_to_ollama(self, anthropic_tools: list[dict]) -> list[dict]:
        """
        Convertit les schémas tool_use Anthropic en format Ollama.

        Anthropic : {"name": "...", "description": "...", "input_schema": {...}}
        Ollama    : {"type": "function", "function": {"name": "...", "parameters": {...}}}
        """
        ollama_tools = []
        for tool in anthropic_tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        return ollama_tools
