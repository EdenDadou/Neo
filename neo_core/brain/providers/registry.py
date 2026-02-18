"""
Neo Core — ModelRegistry : Catalogue, test et routing des modèles LLM
======================================================================
Point d'entrée unique pour tous les providers et modèles.

Responsabilités :
1. Enregistrer les providers (Anthropic, Ollama, Groq, Gemini)
2. Découvrir automatiquement les modèles disponibles
3. Tester chaque modèle AVANT qu'il soit utilisable par Brain
4. Router Brain vers le meilleur modèle selon le besoin (capability)
5. Gérer le fallback si un modèle échoue (rate limit, timeout)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from neo_core.brain.providers.base import (
    LLMProvider,
    ModelInfo,
    ModelCapability,
    ModelTestResult,
    ChatResponse,
)


# ─── Stratégie de routing par agent ────────────────────────
# Chaque agent a un niveau de capability requis.
# Le registry sélectionne le meilleur modèle testé pour ce niveau.

AGENT_REQUIREMENTS: dict[str, ModelCapability] = {
    # Core agents — le cœur du système, toujours sur Claude
    "vox": ModelCapability.STANDARD,        # Reformulation → Claude Haiku (fiable)
    "brain": ModelCapability.ADVANCED,      # Orchestration → Claude Sonnet (le meilleur)
    "memory": ModelCapability.STANDARD,     # Apprentissage + synthèse → Claude Haiku (fiable)
    # Workers — routés dynamiquement par Brain selon la tâche
    "worker:researcher": ModelCapability.STANDARD,
    "worker:coder": ModelCapability.ADVANCED,
    "worker:analyst": ModelCapability.ADVANCED,
    "worker:summarizer": ModelCapability.BASIC,
    "worker:writer": ModelCapability.STANDARD,
    "worker:translator": ModelCapability.BASIC,
    "worker:generic": ModelCapability.BASIC,
}

# Agents du Core : TOUJOURS sur Anthropic (Claude)
# Vox, Brain, Memory = le cœur, pas de compromis sur la qualité
# Les Workers sont routés dynamiquement (Ollama, Groq, Gemini, Anthropic)
PREFER_CLOUD_AGENTS: set[str] = {
    "vox",
    "brain",
    "memory",
    "worker:coder",
    "worker:analyst",
}

# Priorité des providers (gratuit local > gratuit cloud > payant)
PROVIDER_PRIORITY = {
    "ollama": 0,       # Gratuit, illimité, pas de rate limit
    "groq": 1,         # Gratuit, rapide, 14 400 req/jour
    "gemini": 2,       # Gratuit, 250 req/jour
    "anthropic": 3,    # Payant
}

# Mapping capability → priorité de sélection
CAPABILITY_ORDER = {
    ModelCapability.BASIC: 0,
    ModelCapability.STANDARD: 1,
    ModelCapability.ADVANCED: 2,
}


class ModelRegistry:
    """
    Catalogue centralisé de tous les modèles LLM disponibles.

    Usage :
        registry = ModelRegistry()
        registry.register_provider(OllamaProvider())
        registry.register_provider(GroqProvider(api_key="..."))
        registry.discover_models()
        await registry.test_all()

        # Brain demande le meilleur modèle pour un agent
        model = registry.get_best_for("brain")  # → modèle ADVANCED testé
        provider = registry.get_provider(model.provider)
        response = await provider.chat(...)
    """

    def __init__(self, config_path: Path | None = None):
        self._providers: dict[str, LLMProvider] = {}
        self._models: dict[str, ModelInfo] = {}  # model_id → ModelInfo
        self._config_path = config_path
        # Cache du routing (agent_name → model_id)
        self._routing_cache: dict[str, str] = {}

    # ─── Gestion des providers ──────────────────────────────

    def register_provider(self, provider: LLMProvider) -> None:
        """Enregistre un provider."""
        self._providers[provider.name] = provider
        logger.debug("Provider registered: %s (%s)", provider.name, provider.provider_type.value)

    def get_provider(self, provider_name: str) -> LLMProvider | None:
        """Récupère un provider par son nom."""
        return self._providers.get(provider_name)

    def get_provider_for_model(self, model_id: str) -> LLMProvider | None:
        """Récupère le provider d'un modèle à partir de son model_id."""
        model = self._models.get(model_id)
        if model:
            return self._providers.get(model.provider)
        # Fallback : extraire le provider du model_id
        parts = model_id.split(":", 1)
        if parts:
            return self._providers.get(parts[0])
        return None

    def get_configured_providers(self) -> list[str]:
        """Liste les providers correctement configurés."""
        return [
            name for name, provider in self._providers.items()
            if provider.is_configured()
        ]

    # ─── Découverte des modèles ─────────────────────────────

    def discover_models(self) -> list[ModelInfo]:
        """
        Découvre tous les modèles de tous les providers configurés.

        Fusionne avec les résultats de tests précédents (si config persistée).
        """
        self._models.clear()

        for name, provider in self._providers.items():
            if not provider.is_configured():
                continue

            try:
                models = provider.list_models()
                for model in models:
                    self._models[model.model_id] = model
                logger.debug("Provider %s: %d models discovered", name, len(models))
            except Exception as e:
                logger.warning("Provider %s discovery failed: %s", name, e)

        # Charger les résultats de tests précédents
        self._load_test_results()

        logger.info("Model discovery complete: %d models from %d providers",
                    len(self._models), len(self._providers))

        return list(self._models.values())

    # ─── Test des modèles ───────────────────────────────────

    async def test_model(self, model_id: str) -> ModelTestResult:
        """
        Teste un modèle spécifique.

        Le modèle DOIT être testé avec succès avant d'être marqué "available".
        """
        model = self._models.get(model_id)
        if not model:
            return ModelTestResult(
                model_id=model_id,
                success=False,
                error=f"Modèle {model_id} non trouvé dans le catalogue",
            )

        provider = self._providers.get(model.provider)
        if not provider:
            return ModelTestResult(
                model_id=model_id,
                success=False,
                error=f"Provider {model.provider} non enregistré",
            )

        result = await provider.test_model(model.model_name)

        # Mettre à jour le statut du modèle
        if result.success:
            model.status = "available"
            model.avg_latency_ms = result.latency_ms
        else:
            model.status = "failed"
            model.test_error = result.error
        model.last_test = result.tested_at

        # Persister le résultat
        self._save_test_results()

        return result

    async def test_all(self) -> dict[str, ModelTestResult]:
        """Teste tous les modèles découverts."""
        results = {}
        for model_id in list(self._models.keys()):
            results[model_id] = await self.test_model(model_id)
        return results

    # ─── Routing : sélection du meilleur modèle ────────────

    def get_best_for(self, agent_name: str) -> ModelInfo | None:
        """
        Retourne le meilleur modèle disponible pour un agent donné.

        Stratégie :
        1. Déterminer le niveau de capability requis
        2. Filtrer les modèles testés et disponibles
        3. Parmi ceux qui matchent, prendre le plus prioritaire
           (local gratuit > cloud gratuit > cloud payant)
        4. Si rien ne matche, fallback au niveau inférieur
        """
        # Vérifier le cache du routing (override manuel)
        if agent_name in self._routing_cache:
            cached_id = self._routing_cache[agent_name]
            cached_model = self._models.get(cached_id)
            if cached_model and cached_model.status == "available":
                return cached_model

        required = AGENT_REQUIREMENTS.get(agent_name, ModelCapability.STANDARD)
        prefer_cloud = agent_name in PREFER_CLOUD_AGENTS
        return self._find_best_model(required, prefer_cloud=prefer_cloud)

    def _find_best_model(
        self,
        min_capability: ModelCapability,
        require_tools: bool = False,
        prefer_cloud: bool = False,
    ) -> ModelInfo | None:
        """
        Trouve le meilleur modèle disponible pour une capability donnée.

        Priorité normale (économe) :
            local gratuit (Ollama) → cloud gratuit (Groq, Gemini) → cloud payant (Anthropic)

        Priorité cloud (qualité, pour Brain/Coder/Analyst) :
            cloud payant (Anthropic) → cloud gratuit (Groq, Gemini) → local (Ollama)
        """
        available = [
            m for m in self._models.values()
            if m.status == "available"
        ]

        if not available:
            return None

        # Filtrer par capability minimum
        min_level = CAPABILITY_ORDER[min_capability]
        matching = [
            m for m in available
            if CAPABILITY_ORDER.get(m.capability, 0) >= min_level
        ]

        # Filtrer par support tools si requis
        if require_tools:
            matching = [m for m in matching if m.supports_tools]

        # Si rien ne matche, élargir la recherche
        if not matching:
            matching = available

        if not matching:
            return None

        if prefer_cloud:
            # Brain, Coder, Analyst : priorité qualité (cloud d'abord)
            # Inverser les priorités : Anthropic (3→0), Gemini (2→1), Groq (1→2), Ollama (0→3)
            cloud_priority = {
                "anthropic": 0,
                "gemini": 1,
                "groq": 2,
                "ollama": 3,
            }
            def sort_key(m: ModelInfo) -> tuple:
                return (
                    cloud_priority.get(m.provider, 99),
                    -CAPABILITY_ORDER.get(m.capability, 0),  # Meilleur modèle d'abord
                    m.avg_latency_ms or 9999,
                )
        else:
            # Vox, Memory, Workers simples : priorité économie (local d'abord)
            def sort_key(m: ModelInfo) -> tuple:
                return (
                    PROVIDER_PRIORITY.get(m.provider, 99),
                    CAPABILITY_ORDER.get(m.capability, 0),
                    m.avg_latency_ms or 9999,
                )

        matching.sort(key=sort_key)
        return matching[0]

    def get_fallback_chain(
        self,
        agent_name: str,
        require_tools: bool = False,
    ) -> list[ModelInfo]:
        """
        Retourne une chaîne ordonnée de modèles pour un agent.

        Si le premier modèle échoue (rate limit, timeout, erreur),
        le router essaie le suivant dans la liste.

        Retourne tous les modèles disponibles, triés par priorité.
        """
        required = AGENT_REQUIREMENTS.get(agent_name, ModelCapability.STANDARD)
        prefer_cloud = agent_name in PREFER_CLOUD_AGENTS

        available = [
            m for m in self._models.values()
            if m.status == "available"
        ]

        if not available:
            return []

        # Filtrer par capability minimum
        min_level = CAPABILITY_ORDER[required]
        matching = [
            m for m in available
            if CAPABILITY_ORDER.get(m.capability, 0) >= min_level
        ]

        if require_tools:
            matching = [m for m in matching if m.supports_tools]

        # Si rien ne matche, tout prendre
        if not matching:
            matching = available

        if prefer_cloud:
            cloud_priority = {
                "anthropic": 0,
                "gemini": 1,
                "groq": 2,
                "ollama": 3,
            }
            def sort_key(m: ModelInfo) -> tuple:
                return (
                    cloud_priority.get(m.provider, 99),
                    -CAPABILITY_ORDER.get(m.capability, 0),
                    m.avg_latency_ms or 9999,
                )
        else:
            def sort_key(m: ModelInfo) -> tuple:
                return (
                    PROVIDER_PRIORITY.get(m.provider, 99),
                    CAPABILITY_ORDER.get(m.capability, 0),
                    m.avg_latency_ms or 9999,
                )

        matching.sort(key=sort_key)
        return matching

    def get_model_for_worker_with_tools(self, agent_name: str) -> ModelInfo | None:
        """
        Comme get_best_for, mais exige le support tool_use.

        Utilisé par les Workers qui ont besoin d'outils (researcher, coder, etc.).
        """
        required = AGENT_REQUIREMENTS.get(agent_name, ModelCapability.STANDARD)
        return self._find_best_model(required, require_tools=True)

    # ─── Routing override ───────────────────────────────────

    def set_routing(self, agent_name: str, model_id: str) -> None:
        """Force le routing d'un agent vers un modèle spécifique."""
        self._routing_cache[agent_name] = model_id

    def set_routing_from_config(self, routing: dict[str, str]) -> None:
        """Charge le routing depuis la config (neo_config.json)."""
        self._routing_cache.update(routing)

    def get_current_routing(self) -> dict[str, str]:
        """Retourne le routing actuel pour tous les agents."""
        routing = {}
        for agent_name in AGENT_REQUIREMENTS:
            model = self.get_best_for(agent_name)
            if model:
                routing[agent_name] = model.model_id
        return routing

    # ─── Informations ───────────────────────────────────────

    def get_available_models(self) -> list[ModelInfo]:
        """Retourne uniquement les modèles testés et disponibles."""
        return [m for m in self._models.values() if m.status == "available"]

    def get_all_models(self) -> list[ModelInfo]:
        """Retourne tous les modèles découverts (quel que soit le statut)."""
        return list(self._models.values())

    def get_stats(self) -> dict:
        """Statistiques du registry."""
        models = list(self._models.values())
        return {
            "total_providers": len(self._providers),
            "configured_providers": len(self.get_configured_providers()),
            "total_models": len(models),
            "available_models": len([m for m in models if m.status == "available"]),
            "failed_models": len([m for m in models if m.status == "failed"]),
            "untested_models": len([m for m in models if m.status == "untested"]),
            "local_models": len([m for m in models if m.is_local and m.status == "available"]),
            "free_models": len([m for m in models if m.is_free and m.status == "available"]),
        }

    # ─── Persistance ────────────────────────────────────────

    def _load_test_results(self) -> None:
        """Charge les résultats de tests précédents depuis neo_config.json."""
        if not self._config_path or not self._config_path.exists():
            return

        try:
            with open(self._config_path) as f:
                config = json.load(f)

            tested = config.get("tested_models", {})
            for model_id, result in tested.items():
                if model_id in self._models:
                    model = self._models[model_id]
                    model.status = result.get("status", "untested")
                    model.avg_latency_ms = result.get("latency_ms")
                    if result.get("tested_at"):
                        try:
                            model.last_test = datetime.fromisoformat(result["tested_at"])
                        except (ValueError, TypeError):
                            pass

            # Charger le routing override
            routing = config.get("model_routing", {})
            self.set_routing_from_config(routing)

        except (json.JSONDecodeError, IOError):
            pass

    def _save_test_results(self) -> None:
        """Sauvegarde les résultats de tests dans neo_config.json."""
        if not self._config_path:
            return

        try:
            config = {}
            if self._config_path.exists():
                with open(self._config_path) as f:
                    config = json.load(f)

            # Sauvegarder les résultats de tests
            tested = {}
            for model_id, model in self._models.items():
                if model.status != "untested":
                    tested[model_id] = {
                        "status": model.status,
                        "latency_ms": model.avg_latency_ms,
                        "tested_at": model.last_test.isoformat() if model.last_test else None,
                    }
            config["tested_models"] = tested

            # Sauvegarder le routing
            config["model_routing"] = self.get_current_routing()

            # Sauvegarder les providers configurés
            providers_config = {}
            for name, provider in self._providers.items():
                providers_config[name] = {
                    "enabled": provider.is_configured(),
                    "models": [
                        m.model_name for m in self._models.values()
                        if m.provider == name and m.status == "available"
                    ],
                }
            config["providers"] = providers_config

            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        except Exception:
            pass


# ─── Singleton global ──────────────────────────────────────

_global_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """
    Retourne le ModelRegistry global.

    Le registry est initialisé lazy — les providers sont enregistrés
    au premier appel en fonction des variables d'environnement.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = _create_default_registry()
    return _global_registry


def set_model_registry(registry: ModelRegistry) -> None:
    """Remplace le registry global (utile pour les tests)."""
    global _global_registry
    _global_registry = registry


def _create_default_registry() -> ModelRegistry:
    """
    Crée un registry par défaut avec les providers disponibles.

    Enregistre automatiquement :
    - Anthropic si ANTHROPIC_API_KEY est défini
    - Groq si GROQ_API_KEY est défini
    - Gemini si GEMINI_API_KEY est défini
    - Ollama si le serveur est accessible
    """
    from pathlib import Path

    # Config path
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "data" / "neo_config.json"

    registry = ModelRegistry(config_path=config_path)

    # Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        from neo_core.brain.providers.anthropic_provider import AnthropicProvider
        registry.register_provider(AnthropicProvider(api_key=anthropic_key))

    # Groq
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        from neo_core.brain.providers.groq_provider import GroqProvider
        registry.register_provider(GroqProvider(api_key=groq_key))

    # Gemini
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        from neo_core.brain.providers.gemini_provider import GeminiProvider
        registry.register_provider(GeminiProvider(api_key=gemini_key))

    # Ollama (local — pas besoin de clé)
    from neo_core.brain.providers.ollama_provider import OllamaProvider
    ollama = OllamaProvider()
    if ollama.is_configured():
        registry.register_provider(ollama)

    # Découvrir les modèles
    registry.discover_models()

    return registry
