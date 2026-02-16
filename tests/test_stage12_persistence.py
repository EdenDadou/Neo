"""
Tests Stage 12 — Conversation Persistence
===========================================
Vérifie que les conversations sont sauvegardées et restaurées correctement.

30+ tests au total.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neo_core.memory.conversation import (
    ConversationStore,
    ConversationSession,
    ConversationTurn,
)


# ─── Fixtures ────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """Base de données temporaire."""
    return tmp_path / "test_conversations.db"


@pytest.fixture
def store(tmp_db):
    """ConversationStore prêt à l'emploi."""
    s = ConversationStore(tmp_db)
    yield s
    s.close()


# ══════════════════════════════════════════════════════
# 1. TestConversationStore — Stockage de base (~12 tests)
# ══════════════════════════════════════════════════════

class TestConversationStore:
    """Tests du ConversationStore."""

    def test_init_creates_db(self, tmp_db):
        """L'initialisation crée la base de données."""
        store = ConversationStore(tmp_db)
        assert tmp_db.exists()
        store.close()

    def test_start_session(self, store):
        """Démarrer une session crée un ID unique."""
        session = store.start_session("Eden")
        assert session.session_id
        assert session.user_name == "Eden"
        assert session.message_count == 0

    def test_start_multiple_sessions(self, store):
        """Plusieurs sessions ont des IDs différents."""
        s1 = store.start_session("Eden")
        s2 = store.start_session("Eden")
        assert s1.session_id != s2.session_id

    def test_append_turn_human(self, store):
        """Ajouter un message humain."""
        session = store.start_session("Eden")
        turn = store.append_turn(session.session_id, "human", "Bonjour Neo")
        assert turn.role == "human"
        assert turn.content == "Bonjour Neo"
        assert turn.turn_number == 1

    def test_append_turn_assistant(self, store):
        """Ajouter un message assistant."""
        session = store.start_session("Eden")
        store.append_turn(session.session_id, "human", "Hello")
        turn = store.append_turn(session.session_id, "assistant", "Bonjour !")
        assert turn.role == "assistant"
        assert turn.turn_number == 2

    def test_append_updates_message_count(self, store):
        """Le compteur de messages est mis à jour."""
        session = store.start_session("Eden")
        store.append_turn(session.session_id, "human", "msg1")
        store.append_turn(session.session_id, "assistant", "resp1")
        store.append_turn(session.session_id, "human", "msg2")

        sessions = store.get_sessions()
        assert sessions[0].message_count == 3

    def test_get_history(self, store):
        """Récupérer l'historique dans l'ordre."""
        session = store.start_session("Eden")
        store.append_turn(session.session_id, "human", "Q1")
        store.append_turn(session.session_id, "assistant", "A1")
        store.append_turn(session.session_id, "human", "Q2")
        store.append_turn(session.session_id, "assistant", "A2")

        history = store.get_history(session.session_id)
        assert len(history) == 4
        assert history[0].role == "human"
        assert history[0].content == "Q1"
        assert history[3].role == "assistant"
        assert history[3].content == "A2"

    def test_get_history_with_limit(self, store):
        """Pagination avec limit."""
        session = store.start_session("Eden")
        for i in range(10):
            store.append_turn(session.session_id, "human", f"msg{i}")

        history = store.get_history(session.session_id, limit=5)
        assert len(history) == 5

    def test_get_history_with_offset(self, store):
        """Pagination avec offset."""
        session = store.start_session("Eden")
        for i in range(10):
            store.append_turn(session.session_id, "human", f"msg{i}")

        history = store.get_history(session.session_id, limit=3, offset=5)
        assert len(history) == 3
        assert history[0].content == "msg5"

    def test_get_sessions(self, store):
        """Lister les sessions."""
        store.start_session("Eden")
        store.start_session("Other")

        sessions = store.get_sessions()
        assert len(sessions) == 2

    def test_get_last_session(self, store):
        """Récupérer la dernière session."""
        store.start_session("First")
        s2 = store.start_session("Last")

        last = store.get_last_session()
        assert last is not None
        assert last.session_id == s2.session_id

    def test_get_last_session_empty(self, store):
        """Pas de session → None."""
        assert store.get_last_session() is None


# ══════════════════════════════════════════════════════
# 2. TestExportImport — Export JSON (~4 tests)
# ══════════════════════════════════════════════════════

class TestExportImport:
    """Tests d'export/import."""

    def test_export_json(self, store):
        """Export JSON contient session + turns."""
        session = store.start_session("Eden")
        store.append_turn(session.session_id, "human", "Bonjour")
        store.append_turn(session.session_id, "assistant", "Salut !")

        data = store.export_json(session.session_id)
        assert "session" in data
        assert "turns" in data
        assert data["session"]["user_name"] == "Eden"
        assert len(data["turns"]) == 2

    def test_export_json_valid(self, store):
        """Export est sérialisable en JSON."""
        session = store.start_session("Eden")
        store.append_turn(session.session_id, "human", "Test")

        data = store.export_json(session.session_id)
        json_str = json.dumps(data)
        assert json_str  # Pas d'erreur de sérialisation

    def test_export_nonexistent_session(self, store):
        """Export d'une session inexistante → dict vide ou erreur gérée."""
        data = store.export_json("nonexistent-id")
        # Devrait retourner un dict vide ou avec session=None
        assert data is not None

    def test_export_json_preserves_unicode(self, store):
        """Export préserve les caractères Unicode."""
        session = store.start_session("Eden")
        store.append_turn(session.session_id, "human", "Café ☕ et crêpes 🥞")

        data = store.export_json(session.session_id)
        assert data["turns"][0]["content"] == "Café ☕ et crêpes 🥞"


# ══════════════════════════════════════════════════════
# 3. TestContextManager — Context manager (~2 tests)
# ══════════════════════════════════════════════════════

class TestContextManager:
    """Tests du context manager."""

    def test_context_manager(self, tmp_db):
        """Utilisation avec 'with'."""
        with ConversationStore(tmp_db) as store:
            session = store.start_session("Eden")
            store.append_turn(session.session_id, "human", "Test")
        # Pas d'exception → OK

    def test_context_manager_persists(self, tmp_db):
        """Les données survivent après fermeture."""
        session_id = None
        with ConversationStore(tmp_db) as store:
            session = store.start_session("Eden")
            session_id = session.session_id
            store.append_turn(session_id, "human", "Persisté")

        # Ré-ouvrir
        with ConversationStore(tmp_db) as store:
            history = store.get_history(session_id)
            assert len(history) == 1
            assert history[0].content == "Persisté"


# ══════════════════════════════════════════════════════
# 4. TestVoxIntegration — Intégration Vox (~6 tests)
# ══════════════════════════════════════════════════════

class TestVoxIntegration:
    """Tests de l'intégration dans Vox."""

    def _make_vox(self):
        """Crée un Vox en mode mock."""
        from neo_core.config import NeoConfig
        from neo_core.core.vox import Vox
        from neo_core.core.brain import Brain
        from neo_core.core.memory_agent import MemoryAgent

        config = NeoConfig()
        memory = MemoryAgent(config=config)
        memory.initialize()
        brain = Brain(config=config)
        brain.connect_memory(memory)
        vox = Vox(config=config)
        vox.connect(brain=brain, memory=memory)
        return vox

    def test_vox_has_conversation_store(self):
        """Vox a un ConversationStore."""
        vox = self._make_vox()
        assert hasattr(vox, '_conversation_store')

    def test_vox_start_new_session(self):
        """Vox peut démarrer une session."""
        vox = self._make_vox()
        vox.start_new_session("Eden")
        assert vox._current_session is not None
        assert vox._current_session.user_name == "Eden"

    def test_vox_get_session_info(self):
        """get_session_info retourne les infos de session."""
        vox = self._make_vox()
        vox.start_new_session("Eden")
        info = vox.get_session_info()
        assert info is not None
        assert info["user_name"] == "Eden"
        assert "session_id" in info

    @pytest.mark.asyncio
    async def test_vox_persists_conversation(self):
        """Les messages sont persistés automatiquement."""
        vox = self._make_vox()
        vox.start_new_session("Eden")
        session_id = vox._current_session.session_id

        await vox.process_message("Bonjour Neo")

        # Vérifier que les turns sont persistés
        if vox._conversation_store:
            history = vox._conversation_store.get_history(session_id)
            assert len(history) >= 2  # human + assistant

    def test_vox_resume_session(self):
        """Vox peut reprendre une session existante."""
        vox = self._make_vox()
        vox.start_new_session("Eden")
        session_id = vox._current_session.session_id

        # Ajouter des turns directement
        if vox._conversation_store:
            vox._conversation_store.append_turn(session_id, "human", "Message 1")
            vox._conversation_store.append_turn(session_id, "assistant", "Réponse 1")

        # Reset conversation history
        vox.conversation_history = []

        # Resume
        vox.resume_session(session_id)
        assert len(vox.conversation_history) >= 2

    def test_vox_resume_nonexistent_session(self):
        """Resume d'une session inexistante → pas de crash."""
        vox = self._make_vox()
        vox.resume_session("nonexistent-id")
        # Pas d'exception → OK


# ══════════════════════════════════════════════════════
# 5. TestGuardianIntegration — StateSnapshot + session (~4 tests)
# ══════════════════════════════════════════════════════

class TestGuardianIntegration:
    """Tests de l'intégration Guardian."""

    def test_state_snapshot_has_session_id(self):
        """StateSnapshot a un champ session_id."""
        from neo_core.core.guardian import StateSnapshot
        snapshot = StateSnapshot(session_id="test-123")
        assert snapshot.session_id == "test-123"

    def test_state_snapshot_to_dict_includes_session(self):
        """to_dict inclut session_id."""
        from neo_core.core.guardian import StateSnapshot
        snapshot = StateSnapshot(session_id="sess-abc")
        d = snapshot.to_dict()
        assert d["session_id"] == "sess-abc"

    def test_state_snapshot_from_dict_restores_session(self):
        """from_dict restaure session_id."""
        from neo_core.core.guardian import StateSnapshot
        data = {"session_id": "sess-xyz", "shutdown_reason": "crash"}
        snapshot = StateSnapshot.from_dict(data)
        assert snapshot.session_id == "sess-xyz"

    def test_state_snapshot_save_load_session(self, tmp_path):
        """Save/load round-trip avec session_id."""
        from neo_core.core.guardian import StateSnapshot
        state_dir = tmp_path / "guardian"
        state_dir.mkdir()

        original = StateSnapshot(session_id="round-trip-id", shutdown_reason="test")
        original.save(state_dir)

        loaded = StateSnapshot.load(state_dir)
        assert loaded is not None
        assert loaded.session_id == "round-trip-id"


# ══════════════════════════════════════════════════════
# 6. TestCLIIntegration — Commandes CLI (~4 tests)
# ══════════════════════════════════════════════════════

class TestCLIIntegration:
    """Tests des commandes CLI."""

    def test_chat_imports_history_functions(self):
        """chat.py a les fonctions d'historique."""
        from neo_core.cli import chat
        assert hasattr(chat, 'print_history')
        assert hasattr(chat, 'print_sessions')

    def test_cli_has_history_command(self):
        """Le CLI supporte 'neo history'."""
        import neo_core.cli
        import inspect
        source = inspect.getsource(neo_core.cli.main)
        assert "history" in source

    def test_help_includes_new_commands(self):
        """print_help() mentionne /history et /sessions."""
        from neo_core.cli.chat import print_help
        import io
        from unittest.mock import patch
        from rich.console import Console

        # Capturer la sortie
        output = io.StringIO()
        test_console = Console(file=output)

        with patch('neo_core.cli.chat.console', test_console):
            print_help()

        text = output.getvalue()
        assert "/history" in text
        assert "/sessions" in text

    def test_conversation_store_file_creation(self):
        """ConversationStore crée le fichier DB au bon endroit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = ConversationStore(db_path)
            store.start_session("Test")
            store.close()
            assert db_path.exists()
