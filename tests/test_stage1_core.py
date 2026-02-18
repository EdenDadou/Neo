"""
Tests Étape 1 — Fondation Neo Core
====================================
Vérifie que les 3 agents existent, se connectent et communiquent.
Tous les tests fonctionnent en mode mock (sans clé API).
"""

import pytest
import asyncio

from neo_core.config import NeoConfig, LLMConfig, MemoryConfig
from neo_core.vox.interface import Vox, AgentStatus
from neo_core.brain.core import Brain, BrainDecision
from neo_core.memory.agent import MemoryAgent
from neo_core.vox.interface import bootstrap


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path, monkeypatch):
    """Config sans clé API → mode mock, avec stockage temporaire."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return NeoConfig(
        llm=LLMConfig(api_key=None),
        memory=MemoryConfig(storage_path=tmp_path / "test_memory"),
    )


@pytest.fixture
def memory(config):
    mem = MemoryAgent(config=config)
    mem.initialize()
    yield mem
    mem.close()


@pytest.fixture
def brain(config, memory):
    b = Brain(config=config)
    b.connect_memory(memory)
    return b


@pytest.fixture
def vox(config, brain, memory):
    v = Vox(config=config)
    v.connect(brain=brain, memory=memory)
    return v


# ─── Tests Config ────────────────────────────────────────────────────

class TestConfig:
    def test_default_config_is_mock(self, config):
        """Sans clé API, on doit être en mode mock."""
        assert config.is_mock_mode() is True

    def test_config_with_key_is_not_mock(self):
        cfg = NeoConfig(llm=LLMConfig(api_key="test-key-123"))
        assert cfg.is_mock_mode() is False

    def test_config_validate_without_key(self, config):
        assert config.validate() is False

    def test_config_agent_names(self, config):
        assert config.vox_name == "Vox"
        assert config.brain_name == "Brain"
        assert config.memory_agent_name == "Memory"


# ─── Tests Memory Agent (Stub) ──────────────────────────────────────

class TestMemoryAgent:
    def test_initialization(self, memory):
        assert memory.is_initialized is True

    def test_store_and_retrieve(self, memory):
        memory.store_memory("L'utilisateur s'appelle Eden.", source="conversation")
        context = memory.get_context("Comment s'appelle l'utilisateur ?")
        assert "Eden" in context

    def test_empty_context(self, config):
        mem = MemoryAgent(config=config)
        context = mem.get_context("test")
        assert "Aucun contexte" in context

    def test_stats(self, memory):
        initial_count = memory.get_stats()["total_entries"]
        memory.store_memory("info 1")
        memory.store_memory("info 2", source="system")
        stats = memory.get_stats()
        assert stats["total_entries"] == initial_count + 2
        assert "conversation" in stats["sources"]
        assert "system" in stats["sources"]

    def test_clear(self, memory):
        memory.store_memory("data")
        memory.clear()
        assert memory.get_stats()["total_entries"] == 0


# ─── Tests Brain ─────────────────────────────────────────────────────

class TestBrain:
    def test_mock_mode(self, brain):
        assert brain._mock_mode is True

    def test_complexity_simple(self, brain):
        assert brain.analyze_complexity("Bonjour") == "simple"

    def test_complexity_moderate(self, brain):
        request = "Peux-tu m'aider à comprendre comment fonctionne le système de mémoire du projet Neo Core avec ChromaDB ?"
        assert brain.analyze_complexity(request) == "moderate"

    def test_complexity_complex(self, brain):
        request = " ".join(["mot"] * 60)
        assert brain.analyze_complexity(request) == "complex"

    @pytest.mark.asyncio
    async def test_process_mock(self, brain):
        response = await brain.process("Bonjour, comment ça va ?")
        assert "[Brain Mock]" in response
        assert "simple" in response

    def test_memory_context(self, brain, memory):
        memory.store_memory("Le projet s'appelle Neo Core.")
        context = brain.get_memory_context("Quel est le nom du projet ?")
        assert "Neo Core" in context

    def test_decision_simple(self, brain):
        decision = brain.make_decision("Salut")
        assert decision.action == "direct_response"
        assert decision.confidence >= 0.8

    def test_decision_complex(self, brain):
        request = " ".join(["mot"] * 60)
        decision = brain.make_decision(request)
        assert decision.action == "delegate_worker"
        assert len(decision.subtasks) > 0


# ─── Tests Vox ───────────────────────────────────────────────────────

class TestVox:
    def test_connection(self, vox, brain, memory):
        assert vox.brain is brain
        assert vox.memory is memory

    def test_system_status(self, vox):
        status = vox.get_system_status()
        assert "Vox" in status
        assert "Brain" in status
        assert "Memory" in status

    def test_update_agent_status(self, vox):
        vox.update_agent_status("Brain", active=True, task="test", progress=0.5)
        status = vox.get_system_status()
        assert "test" in status
        assert "actif" in status

    @pytest.mark.asyncio
    async def test_process_message(self, vox):
        """Test du pipeline complet : humain → Vox → Brain → Vox → humain."""
        response = await vox.process_message("Bonjour Neo Core !")
        assert response  # Non vide
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_conversation_history(self, vox):
        await vox.process_message("Premier message")
        await vox.process_message("Deuxième message")
        assert len(vox.conversation_history) == 4  # 2 human + 2 AI

    def test_not_connected(self):
        """Vox sans Brain doit retourner une erreur."""
        vox = Vox()
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(vox.process_message("test"))
        assert "Erreur" in response


# ─── Tests Bootstrap (Intégration) ───────────────────────────────────

class TestBootstrap:
    def test_bootstrap_creates_connected_system(self):
        vox = bootstrap()
        assert vox.brain is not None
        assert vox.memory is not None
        assert vox.brain.memory is not None

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test end-to-end : bootstrap → message → réponse."""
        vox = bootstrap()
        response = await vox.process_message("Quel est ton rôle dans Neo Core ?")
        assert response
        assert isinstance(response, str)
        assert len(response) > 10
