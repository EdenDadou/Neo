"""
Tests Étape 6 — Multi-Provider LLM (Local + Cloud)
====================================================
Vérifie le système de providers, le hardware detector,
le model registry, et le routing dynamique.

Tous les tests fonctionnent en mode mock (sans clé API, sans Ollama).
"""

import pytest
import asyncio

from neo_core.config import NeoConfig, LLMConfig, AgentModelConfig, get_agent_model
from neo_core.providers.base import (
    LLMProvider,
    ProviderType,
    ModelInfo,
    ModelCapability,
    TestResult,
    ChatResponse,
)
from neo_core.providers.hardware import HardwareDetector, HardwareProfile, GPUInfo
from neo_core.providers.registry import (
    ModelRegistry,
    AGENT_REQUIREMENTS,
    PROVIDER_PRIORITY,
)


# ═══════════════════════════════════════════════════════════
# HARDWARE DETECTOR
# ═══════════════════════════════════════════════════════════

class TestHardwareDetector:
    """Tests du détecteur de hardware."""

    def test_detect_returns_profile(self):
        """detect() retourne un HardwareProfile valide."""
        profile = HardwareDetector.detect()
        assert isinstance(profile, HardwareProfile)
        assert profile.total_ram_gb > 0
        assert profile.cpu_cores > 0
        assert profile.cpu_threads >= profile.cpu_cores

    def test_hardware_profile_properties(self):
        """Les propriétés can_run_* suivent la hiérarchie de RAM."""
        # Simuler un profil avec 8 GB RAM, pas de GPU
        profile = HardwareProfile(
            total_ram_gb=8.0,
            available_ram_gb=4.0,
            cpu_cores=4,
            cpu_threads=8,
        )
        assert profile.can_run_1_5b is True
        assert profile.can_run_3b is True
        assert profile.can_run_7b is True  # 8GB ≥ 6GB min
        assert profile.can_run_14b is False  # 8GB < 10GB min
        assert profile.max_model_size() == "7b"

    def test_hardware_profile_with_gpu(self):
        """Avec GPU, effective_memory utilise le VRAM."""
        profile = HardwareProfile(
            total_ram_gb=8.0,
            available_ram_gb=4.0,
            cpu_cores=4,
            cpu_threads=8,
            gpu=GPUInfo(name="RTX 3090", vram_mb=24576, vram_gb=24.0),
        )
        assert profile.has_gpu is True
        assert profile.effective_memory_gb == 24.0
        assert profile.can_run_14b is True
        assert profile.max_model_size() == "32b"  # 24GB ≥ 20GB → peut tourner 32b

    def test_hardware_profile_no_gpu(self):
        """Sans GPU, effective_memory utilise la RAM."""
        profile = HardwareProfile(
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            cpu_cores=4,
            cpu_threads=8,
        )
        assert profile.has_gpu is False
        assert profile.effective_memory_gb == 16.0
        assert profile.can_run_14b is True

    def test_recommend_ollama_models_8gb(self):
        """8GB RAM recommande DeepSeek-R1 8B + Llama 3.1 8B."""
        profile = HardwareProfile(
            total_ram_gb=8.0,
            available_ram_gb=4.0,
            cpu_cores=4,
            cpu_threads=8,
        )
        models = profile.recommend_ollama_models()
        assert len(models) >= 2
        model_names = [m["model"] for m in models]
        assert "deepseek-r1:8b" in model_names
        assert "llama3.1:8b" in model_names

    def test_recommend_ollama_models_4gb(self):
        """4GB RAM recommande des modèles 3B max."""
        profile = HardwareProfile(
            total_ram_gb=4.0,
            available_ram_gb=2.0,
            cpu_cores=2,
            cpu_threads=2,
        )
        models = profile.recommend_ollama_models()
        assert len(models) >= 1
        # Pas de modèles 7B+
        for m in models:
            assert "8b" not in m["model"] and "7b" not in m["model"]

    def test_recommend_ollama_models_2gb(self):
        """2GB RAM recommande uniquement ultra-légers."""
        profile = HardwareProfile(
            total_ram_gb=2.0,
            available_ram_gb=1.0,
            cpu_cores=1,
            cpu_threads=1,
        )
        models = profile.recommend_ollama_models()
        assert len(models) >= 1
        assert any("1.5b" in m["model"] or "0.5b" in m["model"] for m in models)

    def test_recommend_ollama_models_1gb(self):
        """1GB RAM : aucun modèle recommandé."""
        profile = HardwareProfile(
            total_ram_gb=1.0,
            available_ram_gb=0.5,
            cpu_cores=1,
            cpu_threads=1,
        )
        assert profile.max_model_size() == "none"
        models = profile.recommend_ollama_models()
        assert models == []

    def test_hardware_summary(self):
        """summary() retourne un string lisible."""
        profile = HardwareProfile(
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            cpu_cores=4,
            cpu_threads=8,
        )
        summary = profile.summary()
        assert "16.0 GB" in summary
        assert "4 cores" in summary
        assert "Aucun" in summary  # Pas de GPU


# ═══════════════════════════════════════════════════════════
# PROVIDER BASE
# ═══════════════════════════════════════════════════════════

class TestModelInfo:
    """Tests de ModelInfo et ModelCapability."""

    def test_model_info_creation(self):
        """ModelInfo est créé avec les bons champs."""
        model = ModelInfo(
            model_id="groq:llama-3.3-70b",
            provider="groq",
            model_name="llama-3.3-70b",
            display_name="Llama 3.3 70B",
            capability=ModelCapability.ADVANCED,
        )
        assert model.model_id == "groq:llama-3.3-70b"
        assert model.status == "untested"
        assert model.is_free is True

    def test_model_info_str(self):
        """__str__ affiche le status et le nom."""
        model = ModelInfo(
            model_id="test:model",
            provider="test",
            model_name="model",
            display_name="Test Model",
            capability=ModelCapability.BASIC,
            status="available",
        )
        assert "✓" in str(model)
        assert "Test Model" in str(model)

    def test_capability_ordering(self):
        """Les capabilities sont ordonnées."""
        assert ModelCapability.BASIC.value == "basic"
        assert ModelCapability.STANDARD.value == "standard"
        assert ModelCapability.ADVANCED.value == "advanced"


# ═══════════════════════════════════════════════════════════
# PROVIDERS
# ═══════════════════════════════════════════════════════════

class TestProviders:
    """Tests des providers (sans appels réels)."""

    def test_anthropic_provider_not_configured_without_key(self):
        """AnthropicProvider n'est pas configuré sans clé."""
        from neo_core.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(api_key="")
        assert provider.name == "anthropic"
        assert provider.provider_type == ProviderType.CLOUD_PAID
        assert provider.is_configured() is False

    def test_anthropic_provider_configured_with_key(self):
        """AnthropicProvider est configuré avec une clé."""
        from neo_core.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(api_key="sk-ant-test-key")
        assert provider.is_configured() is True
        models = provider.list_models()
        assert len(models) >= 2
        assert any("sonnet" in m.model_name for m in models)
        assert any("haiku" in m.model_name for m in models)
        assert all(m.supports_tools for m in models)

    def test_groq_provider_lists_models(self):
        """GroqProvider liste ses modèles."""
        from neo_core.providers.groq_provider import GroqProvider
        provider = GroqProvider(api_key="gsk_test")
        assert provider.name == "groq"
        assert provider.provider_type == ProviderType.CLOUD_FREE
        models = provider.list_models()
        assert len(models) >= 2
        assert all(m.is_free for m in models)
        assert any("70b" in m.model_name for m in models)

    def test_gemini_provider_lists_models(self):
        """GeminiProvider liste ses modèles."""
        from neo_core.providers.gemini_provider import GeminiProvider
        provider = GeminiProvider(api_key="AIza_test")
        assert provider.name == "gemini"
        assert provider.provider_type == ProviderType.CLOUD_FREE
        models = provider.list_models()
        assert len(models) >= 1
        assert all(m.is_free for m in models)
        assert any(m.context_window >= 1_000_000 for m in models)

    def test_ollama_provider_not_configured_without_server(self):
        """OllamaProvider n'est pas configuré si le serveur ne répond pas."""
        from neo_core.providers.ollama_provider import OllamaProvider
        provider = OllamaProvider(base_url="http://localhost:99999")
        assert provider.name == "ollama"
        assert provider.provider_type == ProviderType.LOCAL
        assert provider.is_configured() is False
        assert provider.list_models() == []


# ═══════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════

class MockProvider(LLMProvider):
    """Provider mock pour les tests du registry."""

    def __init__(self, name: str = "mock", models: list[ModelInfo] | None = None):
        self._name = name
        self._models = models or []
        self._configured = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.LOCAL

    def is_configured(self) -> bool:
        return self._configured

    def list_models(self) -> list[ModelInfo]:
        return self._models

    async def chat(self, messages, model, **kwargs) -> ChatResponse:
        return ChatResponse(text="mock response", model=model, provider=self._name)


class TestModelRegistry:
    """Tests du ModelRegistry."""

    @pytest.fixture
    def registry(self):
        return ModelRegistry()

    @pytest.fixture
    def mock_models(self):
        return [
            ModelInfo(
                model_id="mock:basic-model",
                provider="mock",
                model_name="basic-model",
                display_name="Mock Basic",
                capability=ModelCapability.BASIC,
                is_free=True,
                is_local=True,
            ),
            ModelInfo(
                model_id="mock:advanced-model",
                provider="mock",
                model_name="advanced-model",
                display_name="Mock Advanced",
                capability=ModelCapability.ADVANCED,
                is_free=True,
                is_local=True,
                supports_tools=True,
            ),
        ]

    def test_register_provider(self, registry):
        """Enregistrer un provider fonctionne."""
        provider = MockProvider("test")
        registry.register_provider(provider)
        assert registry.get_provider("test") is provider

    def test_discover_models(self, registry, mock_models):
        """discover_models() trouve les modèles des providers enregistrés."""
        provider = MockProvider("mock", models=mock_models)
        registry.register_provider(provider)
        models = registry.discover_models()
        assert len(models) == 2

    def test_get_stats(self, registry, mock_models):
        """get_stats() retourne les bonnes statistiques."""
        provider = MockProvider("mock", models=mock_models)
        registry.register_provider(provider)
        registry.discover_models()
        stats = registry.get_stats()
        assert stats["total_providers"] == 1
        assert stats["total_models"] == 2
        assert stats["untested_models"] == 2

    @pytest.mark.asyncio
    async def test_test_model_success(self, registry, mock_models):
        """Tester un modèle mock retourne success."""
        provider = MockProvider("mock", models=mock_models)
        registry.register_provider(provider)
        registry.discover_models()

        result = await registry.test_model("mock:basic-model")
        assert result.success is True
        assert result.latency_ms >= 0

        # Le modèle doit être marqué "available"
        model = registry._models["mock:basic-model"]
        assert model.status == "available"

    @pytest.mark.asyncio
    async def test_test_all(self, registry, mock_models):
        """test_all() teste tous les modèles."""
        provider = MockProvider("mock", models=mock_models)
        registry.register_provider(provider)
        registry.discover_models()

        results = await registry.test_all()
        assert len(results) == 2
        assert all(r.success for r in results.values())

    @pytest.mark.asyncio
    async def test_routing_best_for_agent(self, registry, mock_models):
        """get_best_for() retourne le bon modèle selon la capability."""
        provider = MockProvider("mock", models=mock_models)
        registry.register_provider(provider)
        registry.discover_models()
        await registry.test_all()

        # vox → STANDARD, prefer_cloud mais "mock" n'est pas dans cloud_priority
        # Donc fallback sur le modèle disponible le plus adapté
        vox_model = registry.get_best_for("vox")
        assert vox_model is not None

        # brain → ADVANCED → devrait obtenir advanced-model
        advanced_model = registry.get_best_for("brain")
        assert advanced_model is not None
        assert advanced_model.capability == ModelCapability.ADVANCED

    @pytest.mark.asyncio
    async def test_routing_core_agents_prefer_cloud(self, registry):
        """Le Core (Vox, Brain, Memory) reste sur Anthropic. Workers simples → Ollama."""
        local_model = ModelInfo(
            model_id="ollama:test",
            provider="ollama",
            model_name="test",
            display_name="Local Test",
            capability=ModelCapability.STANDARD,
            is_free=True,
            is_local=True,
        )
        cloud_model = ModelInfo(
            model_id="anthropic:test",
            provider="anthropic",
            model_name="test",
            display_name="Cloud Test",
            capability=ModelCapability.STANDARD,
            is_free=False,
            is_local=False,
        )

        local_provider = MockProvider("ollama", models=[local_model])
        cloud_provider = MockProvider("anthropic", models=[cloud_model])

        registry.register_provider(local_provider)
        registry.register_provider(cloud_provider)
        registry.discover_models()
        await registry.test_all()

        # Vox (STANDARD, prefer_cloud) → Anthropic
        best_vox = registry.get_best_for("vox")
        assert best_vox.provider == "anthropic"

        # Memory (STANDARD, prefer_cloud) → Anthropic
        best_memory = registry.get_best_for("memory")
        assert best_memory.provider == "anthropic"

        # Brain (ADVANCED, prefer_cloud) → fallback Anthropic (STANDARD >= rien de mieux)
        best_brain = registry.get_best_for("brain")
        assert best_brain.provider == "anthropic"

        # Worker:summarizer (BASIC, PAS prefer_cloud) → Ollama local
        best_summarizer = registry.get_best_for("worker:summarizer")
        assert best_summarizer.provider == "ollama"

        # Worker:translator (BASIC, PAS prefer_cloud) → Ollama local
        best_translator = registry.get_best_for("worker:translator")
        assert best_translator.provider == "ollama"

    @pytest.mark.asyncio
    async def test_routing_with_tools_requirement(self, registry):
        """get_model_for_worker_with_tools() filtre par support tools."""
        no_tools = ModelInfo(
            model_id="mock:no-tools",
            provider="mock",
            model_name="no-tools",
            display_name="No Tools",
            capability=ModelCapability.ADVANCED,
            supports_tools=False,
        )
        with_tools = ModelInfo(
            model_id="mock:with-tools",
            provider="mock",
            model_name="with-tools",
            display_name="With Tools",
            capability=ModelCapability.STANDARD,
            supports_tools=True,
        )

        provider = MockProvider("mock", models=[no_tools, with_tools])
        registry.register_provider(provider)
        registry.discover_models()
        await registry.test_all()

        best = registry.get_model_for_worker_with_tools("worker:researcher")
        assert best.supports_tools is True
        assert best.model_name == "with-tools"

    def test_routing_override(self, registry, mock_models):
        """set_routing() force le routing vers un modèle spécifique."""
        provider = MockProvider("mock", models=mock_models)
        registry.register_provider(provider)
        registry.discover_models()

        # Marquer comme available manuellement
        for m in registry._models.values():
            m.status = "available"

        registry.set_routing("brain", "mock:basic-model")
        best = registry.get_best_for("brain")
        assert best.model_id == "mock:basic-model"

    def test_no_available_models_returns_none(self, registry):
        """Sans modèles disponibles, get_best_for retourne None."""
        assert registry.get_best_for("brain") is None


# ═══════════════════════════════════════════════════════════
# CONFIG INTEGRATION
# ═══════════════════════════════════════════════════════════

class TestConfigIntegration:
    """Tests de l'intégration config + providers."""

    def test_agent_model_config_has_provider_fields(self):
        """AgentModelConfig a les champs provider et model_id."""
        config = AgentModelConfig(
            model="llama-3.3-70b",
            provider="groq",
            model_id="groq:llama-3.3-70b",
        )
        assert config.provider == "groq"
        assert config.model_id == "groq:llama-3.3-70b"

    def test_agent_model_config_defaults_to_anthropic(self):
        """Par défaut, le provider est anthropic."""
        config = AgentModelConfig()
        assert config.provider == "anthropic"

    def test_get_agent_model_fallback(self):
        """get_agent_model() retourne un fallback sans registry."""
        # En mode test (pas de providers configurés), fallback au hardcodé
        model = get_agent_model("brain")
        assert model.model is not None
        assert model.temperature > 0

    def test_agent_requirements_complete(self):
        """AGENT_REQUIREMENTS couvre tous les agents et worker types."""
        assert "vox" in AGENT_REQUIREMENTS
        assert "brain" in AGENT_REQUIREMENTS
        assert "memory" in AGENT_REQUIREMENTS
        assert "worker:researcher" in AGENT_REQUIREMENTS
        assert "worker:coder" in AGENT_REQUIREMENTS
        assert "worker:generic" in AGENT_REQUIREMENTS

    def test_provider_priority_ordering(self):
        """Les providers sont ordonnés : local < cloud gratuit < cloud payant."""
        assert PROVIDER_PRIORITY["ollama"] < PROVIDER_PRIORITY["groq"]
        assert PROVIDER_PRIORITY["groq"] < PROVIDER_PRIORITY["anthropic"]
