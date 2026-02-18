"""
Neo Core — LLMProvider : Interface abstraite pour tous les providers LLM
========================================================================
Chaque provider (Anthropic, Ollama, Groq, Gemini) implémente cette interface.
Le ModelRegistry utilise ces providers pour router les appels vers le bon LLM.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ModelCapability(str, Enum):
    """Niveau de capacité d'un modèle."""
    BASIC = "basic"          # Reformulation, résumé, tâches simples
    STANDARD = "standard"    # Recherche, rédaction, traduction
    ADVANCED = "advanced"    # Orchestration, code complexe, raisonnement


class ProviderType(str, Enum):
    """Type de provider."""
    LOCAL = "local"          # Ollama — tourne sur la machine
    CLOUD_FREE = "cloud_free"    # Groq, Gemini — gratuit
    CLOUD_PAID = "cloud_paid"    # Anthropic — payant


@dataclass
class ModelInfo:
    """
    Informations sur un modèle disponible.

    Identifié de manière unique par `model_id` au format "provider:model".
    Exemple : "ollama:deepseek-r1:8b", "groq:llama-3.3-70b-versatile"
    """
    model_id: str               # "provider:model_name"
    provider: str               # "ollama", "groq", "gemini", "anthropic"
    model_name: str             # Nom du modèle chez le provider
    display_name: str           # Nom lisible pour l'UI
    capability: ModelCapability
    context_window: int = 4096  # Tokens max en entrée
    max_output_tokens: int = 4096
    is_free: bool = True
    is_local: bool = False
    supports_tools: bool = False   # Tool use / function calling
    supports_streaming: bool = False

    # ── État (rempli par le ModelRegistry après test) ──
    status: str = "untested"       # "untested" | "available" | "failed" | "rate_limited"
    last_test: Optional[datetime] = None
    avg_latency_ms: Optional[float] = None
    test_error: Optional[str] = None

    def __str__(self) -> str:
        status_icon = {
            "untested": "?",
            "available": "✓",
            "failed": "✗",
            "rate_limited": "⚠",
        }.get(self.status, "?")
        local_tag = " (local)" if self.is_local else ""
        return f"[{status_icon}] {self.display_name}{local_tag}"


@dataclass
class TestResult:
    """Résultat du test d'un modèle."""
    model_id: str
    success: bool
    latency_ms: float = 0.0
    response_text: str = ""
    error: str = ""
    tested_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "tested_at": self.tested_at.isoformat(),
        }


@dataclass
class ChatMessage:
    """Message dans une conversation LLM (format unifié)."""
    role: str       # "system", "user", "assistant", "tool"
    content: str
    tool_use_id: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)


@dataclass
class ChatResponse:
    """
    Réponse unifiée d'un provider LLM.

    Tous les providers retournent ce format, quelle que soit leur API native.
    """
    text: str                           # Texte de la réponse
    model: str                          # Modèle utilisé
    provider: str                       # Provider source
    stop_reason: str = "end_turn"       # "end_turn" | "tool_use" | "max_tokens"
    tool_calls: list[dict] = field(default_factory=list)  # [{name, input, id}]
    usage: dict = field(default_factory=dict)  # {input_tokens, output_tokens}
    raw_response: Any = None            # Réponse brute du provider


class LLMProvider(ABC):
    """
    Interface abstraite pour un provider LLM.

    Chaque provider (Anthropic, Ollama, Groq, Gemini) implémente cette interface.
    Le ModelRegistry utilise ces providers pour :
    1. Lister les modèles disponibles
    2. Tester chaque modèle
    3. Router les appels chat vers le bon provider
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom du provider : "anthropic", "ollama", "groq", "gemini"."""
        ...

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Type : local, cloud_free, cloud_paid."""
        ...

    @abstractmethod
    def is_configured(self) -> bool:
        """True si le provider est configuré (clé API, serveur local, etc.)."""
        ...

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """Liste les modèles disponibles chez ce provider."""
        ...

    @abstractmethod
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
        Envoie un message au LLM et retourne la réponse.

        Format des messages : [{"role": "user", "content": "..."}]
        Format des tools : schéma Anthropic tool_use (le provider traduit si besoin)
        """
        ...

    async def test_model(self, model: str) -> TestResult:
        """
        Teste un modèle avec un message simple.

        Vérifie que le modèle répond correctement et mesure la latence.
        Le modèle doit être testé AVANT d'être marqué "available".
        """
        start = time.time()
        try:
            response = await self.chat(
                messages=[{"role": "user", "content": "Réponds uniquement 'ok'."}],
                model=model,
                max_tokens=10,
                temperature=0.0,
            )
            latency = (time.time() - start) * 1000

            return TestResult(
                model_id=f"{self.name}:{model}",
                success=True,
                latency_ms=latency,
                response_text=response.text[:100],
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return TestResult(
                model_id=f"{self.name}:{model}",
                success=False,
                latency_ms=latency,
                error=f"{type(e).__name__}: {str(e)[:200]}",
            )
