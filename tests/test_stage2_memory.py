"""
Tests Étape 2 — Système Mémoire Persistant
=============================================
Vérifie le stockage, la recherche sémantique, l'injection de contexte
et la consolidation de la mémoire.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from neo_core.config import NeoConfig, LLMConfig, MemoryConfig
from neo_core.memory.store import MemoryStore
from neo_core.memory.context import ContextEngine, ContextBlock
from neo_core.memory.consolidator import MemoryConsolidator
from neo_core.core.memory_agent import MemoryAgent
from neo_core.core.brain import Brain
from neo_core.core.vox import Vox
from neo_core.main import bootstrap


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def tmp_storage(tmp_path):
    """Crée un répertoire temporaire pour le stockage mémoire."""
    storage_path = tmp_path / "test_memory"
    storage_path.mkdir()
    yield storage_path
    shutil.rmtree(storage_path, ignore_errors=True)


@pytest.fixture
def memory_config(tmp_storage):
    """Config mémoire pointant vers le stockage temporaire."""
    return MemoryConfig(storage_path=tmp_storage)


@pytest.fixture
def config(memory_config):
    """Config complète sans clé API (mode mock)."""
    return NeoConfig(
        llm=LLMConfig(api_key=None),
        memory=memory_config,
    )


@pytest.fixture
def store(memory_config):
    """MemoryStore initialisé."""
    s = MemoryStore(memory_config)
    s.initialize()
    yield s
    s.close()


@pytest.fixture
def context_engine(store, memory_config):
    """ContextEngine initialisé."""
    return ContextEngine(store, memory_config)


@pytest.fixture
def consolidator(store, config):
    """MemoryConsolidator initialisé."""
    return MemoryConsolidator(store, config)


@pytest.fixture
def memory_agent(config):
    """MemoryAgent complet initialisé."""
    agent = MemoryAgent(config=config)
    agent.initialize()
    yield agent
    agent.close()


# ─── Tests MemoryStore ───────────────────────────────────────────────

class TestMemoryStore:
    def test_initialization(self, store):
        assert store.is_initialized is True

    def test_store_and_count(self, store):
        store.store("Bonjour le monde", source="test")
        assert store.count() == 1

    def test_store_multiple(self, store):
        store.store("Premier souvenir")
        store.store("Deuxième souvenir")
        store.store("Troisième souvenir")
        assert store.count() == 3

    def test_get_recent(self, store):
        store.store("Ancien", importance=0.3)
        store.store("Récent", importance=0.8)
        records = store.get_recent(limit=2)
        assert len(records) == 2

    def test_get_important(self, store):
        store.store("Peu important", importance=0.2)
        store.store("Très important", importance=0.9)
        store.store("Moyennement important", importance=0.5)
        records = store.get_important(min_importance=0.7)
        assert len(records) == 1
        assert "Très important" in records[0].content

    def test_search_by_source(self, store):
        store.store("Conversation 1", source="conversation")
        store.store("Système info", source="system")
        store.store("Conversation 2", source="conversation")
        records = store.search_by_source("conversation")
        assert len(records) == 2

    def test_search_by_tags(self, store):
        store.store("Info perso", tags=["identité", "préférence"])
        store.store("Task info", tags=["tâche"])
        records = store.search_by_tags(["identité"])
        assert len(records) == 1

    def test_update_importance(self, store):
        record_id = store.store("Test", importance=0.3)
        store.update_importance(record_id, 0.9)
        records = store.get_important(min_importance=0.8)
        assert len(records) == 1

    def test_delete(self, store):
        record_id = store.store("À supprimer")
        assert store.count() == 1
        store.delete(record_id)
        assert store.count() == 0

    def test_stats(self, store):
        store.store("A", source="conversation")
        store.store("B", source="system")
        stats = store.get_stats()
        assert stats["total_entries"] == 2
        assert "conversation" in stats["sources"]
        assert "system" in stats["sources"]

    def test_semantic_search(self, store):
        """Teste la recherche sémantique (si ChromaDB est disponible)."""
        store.store("Je m'appelle Eden et je suis développeur.", importance=0.9)
        store.store("Le projet Neo Core utilise Python.", importance=0.7)
        store.store("Il fait beau aujourd'hui.", importance=0.3)

        results = store.search_semantic("Comment s'appelle l'utilisateur ?")
        # Au minimum, on doit avoir des résultats
        assert len(results) > 0
        # Le premier résultat devrait être le plus pertinent
        assert "Eden" in results[0].content


# ─── Tests ContextEngine ─────────────────────────────────────────────

class TestContextEngine:
    def test_build_empty_context(self, context_engine):
        block = context_engine.build_context("test")
        assert block.is_empty

    def test_build_context_with_data(self, context_engine, store):
        store.store("L'utilisateur s'appelle Eden.", importance=0.9, tags=["identité"])
        store.store("Le projet utilise Python.", importance=0.7, tags=["projet"])

        block = context_engine.build_context("Qui est l'utilisateur ?")
        assert not block.is_empty
        text = block.to_string()
        assert "Eden" in text

    def test_context_to_string_format(self, context_engine, store):
        store.store("Info importante", importance=0.9)
        block = context_engine.build_context("test")
        text = block.to_string()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_store_conversation_turn(self, context_engine, store):
        context_engine.store_conversation_turn(
            "Je m'appelle Eden",
            "Bonjour Eden !"
        )
        assert store.count() == 2

    def test_importance_estimation_personal(self, context_engine):
        importance = context_engine._estimate_importance("Je m'appelle Eden")
        assert importance >= 0.8

    def test_importance_estimation_question(self, context_engine):
        importance = context_engine._estimate_importance("Ça va ?")
        assert importance <= 0.4

    def test_importance_estimation_task(self, context_engine):
        importance = context_engine._estimate_importance(
            "Mon projet urgent a une deadline demain"
        )
        assert importance >= 0.7

    def test_tag_extraction(self, context_engine):
        tags = context_engine._extract_tags("Je m'appelle Eden et je préfère Python")
        assert "identité" in tags
        assert "préférence" in tags


# ─── Tests MemoryConsolidator ────────────────────────────────────────

class TestMemoryConsolidator:
    def test_cleanup_removes_old_low_importance(self, consolidator, store):
        # Crée une entrée avec un vieux timestamp
        record_id = store.store("Vieux souvenir", importance=0.1)
        # Modifie le timestamp manuellement en SQLite
        store._db_conn.execute(
            "UPDATE memories SET timestamp = '2020-01-01T00:00:00' WHERE id = ?",
            (record_id,)
        )
        store._db_conn.commit()

        report = consolidator.cleanup(max_age_days=30, min_importance=0.2)
        assert report.entries_deleted == 1

    def test_cleanup_keeps_important(self, consolidator, store):
        record_id = store.store("Souvenir important", importance=0.9)
        store._db_conn.execute(
            "UPDATE memories SET timestamp = '2020-01-01T00:00:00' WHERE id = ?",
            (record_id,)
        )
        store._db_conn.commit()

        report = consolidator.cleanup(max_age_days=30)
        assert report.entries_deleted == 0

    def test_promote_important(self, consolidator, store):
        store.store("Souvenir accédé", importance=0.5, tags=["accessed"])
        report = consolidator.promote_important(boost=0.1)
        assert report.entries_promoted == 1

    def test_full_consolidation(self, consolidator, store):
        store.store("Entrée 1", importance=0.5)
        store.store("Entrée 2", importance=0.8)
        report = consolidator.full_consolidation()
        assert report.entries_before >= 2

    def test_summarize_conversation(self, consolidator, store):
        from neo_core.memory.store import MemoryRecord
        entries = [
            MemoryRecord(id="1", content="Bonjour, je cherche de l'aide.", importance=0.5),
            MemoryRecord(id="2", content="Bien sûr, comment puis-je vous aider ?", importance=0.4),
            MemoryRecord(id="3", content="Je veux créer un système IA.", importance=0.8),
        ]
        summary = consolidator.summarize_conversation(entries)
        assert summary is not None
        assert "Synthèse" in summary


# ─── Tests MemoryAgent (Intégration) ────────────────────────────────

class TestMemoryAgent:
    def test_initialization(self, memory_agent):
        assert memory_agent.is_initialized is True

    def test_store_and_retrieve(self, memory_agent):
        memory_agent.store_memory(
            "L'utilisateur s'appelle Eden.",
            source="conversation",
            tags=["identité"],
            importance=0.9,
        )
        context = memory_agent.get_context("Comment s'appelle l'utilisateur ?")
        assert "Eden" in context

    def test_conversation_turn(self, memory_agent):
        before = memory_agent.get_stats()["total_entries"]
        memory_agent.on_conversation_turn(
            "Je m'appelle Eden",
            "Bonjour Eden, enchanté !"
        )
        stats = memory_agent.get_stats()
        assert stats["total_entries"] >= before + 2

    def test_search(self, memory_agent):
        memory_agent.store_memory("Python est mon langage préféré.", importance=0.8)
        memory_agent.store_memory("J'habite à Paris.", importance=0.7)
        results = memory_agent.search("langage de programmation")
        assert len(results) > 0

    def test_stats(self, memory_agent):
        stats = memory_agent.get_stats()
        assert stats["initialized"] is True
        assert "total_entries" in stats

    def test_clear(self, memory_agent):
        memory_agent.store_memory("Test")
        memory_agent.clear()
        stats = memory_agent.get_stats()
        assert stats["total_entries"] == 0

    def test_context_block(self, memory_agent):
        memory_agent.store_memory("Info test", importance=0.9)
        block = memory_agent.get_context_block("test")
        assert isinstance(block, ContextBlock)

    def test_not_initialized_returns_empty(self):
        agent = MemoryAgent()
        assert agent.get_context("test") == "Aucun contexte mémoire disponible."
        assert agent.search("test") == []


# ─── Tests Pipeline Complet (Vox → Memory → Brain) ──────────────────

class TestPipelineIntegration:
    @pytest.mark.asyncio
    async def test_conversation_stores_to_memory(self, config):
        """Un échange via Vox doit être stocké en mémoire."""
        memory = MemoryAgent(config=config)
        memory.initialize()

        brain = Brain(config=config)
        brain.connect_memory(memory)

        vox = Vox(config=config)
        vox.connect(brain=brain, memory=memory)

        await vox.process_message("Je m'appelle Eden")
        stats = memory.get_stats()
        assert stats["total_entries"] >= 2  # user msg + ai response

        memory.close()

    @pytest.mark.asyncio
    async def test_memory_persists_across_messages(self, config):
        """Le contexte d'un message précédent est disponible pour le suivant."""
        memory = MemoryAgent(config=config)
        memory.initialize()

        brain = Brain(config=config)
        brain.connect_memory(memory)

        vox = Vox(config=config)
        vox.connect(brain=brain, memory=memory)

        # Premier message
        await vox.process_message("Je m'appelle Eden et je suis développeur Python")

        # Le contexte devrait contenir "Eden" pour le message suivant
        context = memory.get_context("Comment s'appelle l'utilisateur ?")
        assert "Eden" in context

        memory.close()

    @pytest.mark.asyncio
    async def test_full_bootstrap_with_memory(self):
        """Bootstrap crée un système avec mémoire fonctionnelle."""
        vox = bootstrap()
        await vox.process_message("Test de mémoire persistante")

        stats = vox.memory.get_stats()
        assert stats["initialized"] is True
        assert stats["total_entries"] >= 2

        vox.memory.close()
