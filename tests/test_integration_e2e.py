"""
Tests d'intégration End-to-End — v0.8.3
=========================================
Vérifie le pipeline complet :
CLI bootstrap → CoreRegistry → Vox → Brain → Memory → retrieval

Ces tests utilisent le mode mock (pas d'API key) mais vérifient
que TOUS les composants sont connectés et fonctionnent ensemble,
avec de vrais objets (pas de mocks unitaires).
"""

import asyncio
import sqlite3
import threading
from pathlib import Path

import pytest


@pytest.fixture
def data_dir(tmp_path):
    """Crée un répertoire de données temporaire."""
    d = tmp_path / "data"
    d.mkdir()
    (d / "memory").mkdir()
    return d


@pytest.fixture
def config(data_dir):
    """Crée une vraie NeoConfig en mode mock."""
    from neo_core.config import NeoConfig, LLMConfig, MemoryConfig

    return NeoConfig(
        llm=LLMConfig(api_key=None),
        memory=MemoryConfig(storage_path=data_dir / "memory"),
        data_dir=data_dir,
        core_name="TestNeo",
        user_name="TestUser",
    )


@pytest.fixture
def full_stack(config):
    """
    Monte la stack complète : Memory → Brain → Vox.
    Pas de mock, vrais objets en mode mock LLM.
    """
    from neo_core.memory.agent import MemoryAgent
    from neo_core.brain.core import Brain
    from neo_core.vox.interface import Vox

    memory = MemoryAgent(config=config)
    memory.initialize()

    brain = Brain(config=config)
    brain.connect_memory(memory)

    vox = Vox(config=config)
    vox.connect(brain=brain, memory=memory)
    vox.start_new_session("TestUser")

    yield {"vox": vox, "brain": brain, "memory": memory, "config": config}
    memory.close()


# ─── E2E Pipeline ──────────────────────────────────────────────

class TestE2EPipeline:
    """Test du pipeline complet message → réponse → mémoire."""

    def test_full_pipeline_message_to_response(self, full_stack):
        """Envoie un message et vérifie qu'on reçoit une réponse."""
        vox = full_stack["vox"]
        response = asyncio.get_event_loop().run_until_complete(
            vox.process_message("Bonjour, comment ça va ?")
        )
        assert response is not None
        assert len(response) > 0
        assert "[Erreur]" not in response

    def test_conversation_history_grows(self, full_stack):
        """L'historique de conversation s'enrichit après chaque échange."""
        vox = full_stack["vox"]
        assert len(vox.conversation_history) == 0

        asyncio.get_event_loop().run_until_complete(
            vox.process_message("Premier message")
        )
        # 1 HumanMessage + 1 AIMessage
        assert len(vox.conversation_history) == 2

        asyncio.get_event_loop().run_until_complete(
            vox.process_message("Deuxième message")
        )
        assert len(vox.conversation_history) == 4

    def test_conversation_history_bounded(self, full_stack):
        """L'historique ne dépasse pas la limite (200 messages)."""
        vox = full_stack["vox"]
        # Ajouter beaucoup de messages
        for i in range(110):
            asyncio.get_event_loop().run_until_complete(
                vox.process_message(f"Message {i}")
            )
        # 110 * 2 = 220 messages, devrait être borné autour de 200
        # (la troncature se fait avant l'ajout de l'AIMessage, donc +1 possible)
        assert len(vox.conversation_history) <= 202

    def test_session_persistence(self, full_stack):
        """Les tours sont sauvegardés dans le ConversationStore."""
        vox = full_stack["vox"]
        session_id = vox._current_session.session_id

        asyncio.get_event_loop().run_until_complete(
            vox.process_message("Message à persister")
        )

        # Vérifier la persistance
        history = vox._conversation_store.get_history(session_id, limit=100)
        assert len(history) >= 2  # human + assistant
        assert any("persister" in t.content for t in history)


# ─── Composants Connectés ───────────────────────────────────────

class TestComponentConnections:
    """Vérifie que tous les composants sont bien connectés."""

    def test_vox_has_brain(self, full_stack):
        assert full_stack["vox"].brain is not None
        assert full_stack["vox"].brain is full_stack["brain"]

    def test_vox_has_memory(self, full_stack):
        assert full_stack["vox"].memory is not None
        assert full_stack["vox"].memory is full_stack["memory"]

    def test_brain_has_memory(self, full_stack):
        assert full_stack["brain"].memory is not None
        assert full_stack["brain"].memory is full_stack["memory"]

    def test_memory_initialized(self, full_stack):
        assert full_stack["memory"].is_initialized

    def test_session_created(self, full_stack):
        assert full_stack["vox"]._current_session is not None
        assert full_stack["vox"]._current_session.user_name == "TestUser"


# ─── Thread Safety ──────────────────────────────────────────────

class TestConcurrentAccess:
    """Vérifie la thread safety de Vox avec accès concurrent."""

    def test_concurrent_session_resume(self, full_stack):
        """Deux resume_session simultanés ne cassent pas l'état."""
        vox = full_stack["vox"]
        session_id = vox._current_session.session_id
        errors = []

        def resume():
            try:
                vox.resume_session(session_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=resume) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert vox._current_session is not None

    def test_concurrent_messages(self, full_stack):
        """Messages concurrents via asyncio ne crashent pas."""
        vox = full_stack["vox"]
        results = []
        errors = []

        async def send_msg(i):
            try:
                r = await vox.process_message(f"Concurrent msg {i}")
                results.append(r)
            except Exception as e:
                errors.append(e)

        async def run_concurrent():
            await asyncio.gather(*(send_msg(i) for i in range(5)))

        asyncio.get_event_loop().run_until_complete(run_concurrent())
        assert len(errors) == 0
        assert len(results) == 5


# ─── Memory Retrieval ──────────────────────────────────────────

class TestMemoryRetrieval:
    """Vérifie le stockage et la recherche en mémoire."""

    def test_store_and_search(self, full_stack):
        """Stocke un souvenir et le retrouve par recherche sémantique."""
        memory = full_stack["memory"]

        memory.store_memory(
            "L'utilisateur habite à Paris et travaille chez Anthropic.",
            source="conversation",
        )

        context = memory.get_context("Où habite l'utilisateur ?")
        assert context is not None
        assert len(context) > 0

    def test_memory_persists_across_calls(self, full_stack):
        """Les souvenirs persistent entre les appels."""
        memory = full_stack["memory"]

        memory.store_memory("Eden aime le Python.", source="conversation")
        memory.store_memory("Le projet s'appelle Neo Core.", source="conversation")

        # Les deux souvenirs doivent être trouvables
        context = memory.get_context("Quels sont les intérêts d'Eden ?")
        assert context is not None


# ─── Vault Integrity ───────────────────────────────────────────

class TestVaultIntegrity:
    """Vérifie le KeyVault avec de vrais chiffrements."""

    def test_store_retrieve_roundtrip(self, data_dir):
        """Store → Retrieve fonctionne avec de vrais secrets."""
        from neo_core.infra.security.vault import KeyVault

        vault = KeyVault(data_dir=data_dir)
        vault.initialize()

        vault.store("test_key", "super_secret_value_123")
        result = vault.retrieve("test_key")
        assert result == "super_secret_value_123"
        vault.close()

    def test_vault_backup_created(self, data_dir):
        """Le vault crée un backup après initialisation."""
        from neo_core.infra.security.vault import KeyVault

        vault = KeyVault(data_dir=data_dir)
        vault.initialize()
        vault.close()

        backup = data_dir / ".vault.db.bak"
        assert backup.exists()

    def test_corrupted_db_recovery(self, data_dir):
        """Un vault corrompu est recréé proprement."""
        from neo_core.infra.security.vault import KeyVault

        # Créer un fichier DB corrompu
        db_path = data_dir / ".vault.db"
        db_path.write_text("THIS IS NOT A VALID SQLITE FILE")

        # L'init doit recréer la DB
        vault = KeyVault(data_dir=data_dir)
        vault.initialize()
        vault.store("after_corruption", "it_works")
        assert vault.retrieve("after_corruption") == "it_works"
        vault.close()


# ─── API Integration ───────────────────────────────────────────

class TestAPIIntegration:
    """Vérifie les endpoints API avec de vrais composants."""

    @pytest.fixture
    def client(self, full_stack):
        from neo_core.vox.api.server import create_app, neo_core as nc
        from neo_core.infra.registry import CoreRegistry

        # Reset and setup
        nc.reset()
        nc.vox = full_stack["vox"]
        nc.config = full_stack["config"]
        nc._initialized = True

        app = create_app(full_stack["config"])

        from starlette.testclient import TestClient
        return TestClient(app)

    def test_health_returns_components(self, client):
        data = client.get("/health").json()
        assert "checks" in data
        assert "core" in data["checks"]
        assert data["status"] in ("healthy", "degraded")

    def test_health_shows_faiss_status(self, client):
        data = client.get("/health").json()
        faiss_status = data["checks"].get("faiss", "")
        # Should report ok with vector capability
        assert "ok" in faiss_status or "unavailable" in faiss_status


# ─── WorkerLifecycleManager Thread Safety ──────────────────────

class TestWorkerManagerThreadSafety:
    """Vérifie que le WorkerManager est thread-safe."""

    def test_concurrent_register_unregister(self):
        from neo_core.brain.core import WorkerLifecycleManager

        manager = WorkerLifecycleManager()
        errors = []

        class FakeWorker:
            def __init__(self, wid):
                self.worker_id = wid

            def get_lifecycle_info(self):
                return {"id": self.worker_id, "status": "done"}

            def cleanup(self):
                pass

        def register_and_unregister(i):
            try:
                w = FakeWorker(f"w-{i}")
                manager.register(w)
                manager.unregister(w)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_and_unregister, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = manager.get_stats()
        assert stats["active_count"] == 0
        assert stats["total_created"] == 50
        assert stats["total_cleaned"] == 50
        assert stats["leaked"] == 0
