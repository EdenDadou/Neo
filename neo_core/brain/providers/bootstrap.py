"""
Neo Core — Provider Bootstrap : Initialisation des providers au démarrage
==========================================================================
Connecte les providers configurés (Anthropic, Groq, Gemini, Ollama)
au ModelRegistry global et lance la découverte des modèles.

Usage :
    from neo_core.brain.providers.bootstrap import bootstrap_providers
    registry = bootstrap_providers(config)
    print(f"Providers actifs : {registry.get_configured_providers()}")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from neo_core.config import NeoConfig
from neo_core.brain.providers.registry import (
    ModelRegistry,
    get_model_registry,
    set_model_registry,
)

logger = logging.getLogger(__name__)

# Chemin config : neo_core/brain/providers/bootstrap.py → 4 niveaux
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CONFIG_FILE = _PROJECT_ROOT / "data" / "neo_config.json"


def bootstrap_providers(config: NeoConfig = None) -> ModelRegistry:
    """
    Enregistre les providers configurés et lance la découverte de modèles.

    Ordre de détection :
    1. Anthropic (ANTHROPIC_API_KEY ou config.llm.api_key)
    2. Groq (GROQ_API_KEY)
    3. Gemini (GEMINI_API_KEY)
    4. Ollama (serveur local sur localhost:11434)

    Args:
        config: NeoConfig instance (uses default if None)

    Returns:
        ModelRegistry initialisé avec les providers et modèles découverts
    """
    config = config or NeoConfig()

    registry = ModelRegistry(config_path=_CONFIG_FILE)

    providers_found = []

    # 1. Anthropic (payant)
    anthropic_key = _get_anthropic_key(config)
    if anthropic_key:
        try:
            from neo_core.brain.providers.anthropic_provider import AnthropicProvider
            registry.register_provider(AnthropicProvider(api_key=anthropic_key))
            providers_found.append("anthropic")
        except Exception as e:
            logger.warning("Failed to register Anthropic provider: %s", e)

    # 2. Groq (cloud gratuit)
    groq_key = config.groq_api_key or os.getenv("GROQ_API_KEY", "")
    if groq_key:
        try:
            from neo_core.brain.providers.groq_provider import GroqProvider
            registry.register_provider(GroqProvider(api_key=groq_key))
            providers_found.append("groq")
        except Exception as e:
            logger.warning("Failed to register Groq provider: %s", e)

    # 3. Gemini (cloud gratuit)
    gemini_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        try:
            from neo_core.brain.providers.gemini_provider import GeminiProvider
            registry.register_provider(GeminiProvider(api_key=gemini_key))
            providers_found.append("gemini")
        except Exception as e:
            logger.warning("Failed to register Gemini provider: %s", e)

    # 4. Ollama (local, gratuit)
    if _is_ollama_running():
        try:
            from neo_core.brain.providers.ollama_provider import OllamaProvider
            ollama = OllamaProvider()
            if ollama.is_configured():
                registry.register_provider(ollama)
                providers_found.append("ollama")
        except Exception as e:
            logger.warning("Failed to register Ollama provider: %s", e)

    # Découvrir les modèles
    models = registry.discover_models()

    # Installer comme registry global
    set_model_registry(registry)

    # Log
    stats = registry.get_stats()
    logger.info(
        "Providers bootstrap: %s actifs, %d modèles découverts (%d disponibles)",
        ", ".join(providers_found) or "aucun",
        stats["total_models"],
        stats["available_models"],
    )

    return registry


def _get_anthropic_key(config: NeoConfig) -> str:
    """Récupère la clé Anthropic depuis config ou env."""
    key = config.llm.api_key or os.getenv("ANTHROPIC_API_KEY", "")
    return key.strip() if key else ""


def _is_ollama_running() -> bool:
    """Vérifie si le serveur Ollama est accessible."""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def get_provider_summary(registry: ModelRegistry = None) -> dict:
    """
    Résumé des providers pour l'affichage CLI/API.

    Returns:
        Dict avec providers, modèles, routing info
    """
    if registry is None:
        registry = get_model_registry()

    stats = registry.get_stats()
    configured = registry.get_configured_providers()
    routing = registry.get_current_routing()

    providers_info = {}
    for name in configured:
        provider = registry.get_provider(name)
        if provider:
            provider_models = [
                m for m in registry.get_all_models()
                if m.provider == name
            ]
            available = [m for m in provider_models if m.status == "available"]
            providers_info[name] = {
                "type": provider.provider_type.value,
                "models_total": len(provider_models),
                "models_available": len(available),
                "models": [
                    {
                        "model_id": m.model_id,
                        "display_name": m.display_name,
                        "capability": m.capability.value,
                        "status": m.status,
                        "latency_ms": m.avg_latency_ms,
                    }
                    for m in provider_models
                ],
            }

    return {
        "providers": providers_info,
        "total_providers": stats["total_providers"],
        "configured_providers": stats["configured_providers"],
        "total_models": stats["total_models"],
        "available_models": stats["available_models"],
        "local_models": stats["local_models"],
        "free_models": stats["free_models"],
        "routing": routing,
    }
