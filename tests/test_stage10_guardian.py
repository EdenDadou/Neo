"""
Tests Stage 10 — Guardian (Auto-restart + Persistence)
========================================================
Vérifie le mécanisme de supervision et d'auto-restart de Neo.

Classes testées :
- GuardianConfig — Configuration du guardian
- StateSnapshot — Sauvegarde/restauration d'état
- GracefulShutdown — Gestion des signaux
- Guardian — Superviseur de process
- Intégrations chat.py / heartbeat

28 tests au total.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neo_core.core.guardian import (
    EXIT_CODE_NORMAL,
    EXIT_CODE_RESTART,
    GracefulShutdown,
    Guardian,
    GuardianConfig,
    StateSnapshot,
)


# ─── Fixtures ────────────────────────────────────────

@pytest.fixture
def tmp_state_dir(tmp_path):
    """Répertoire temporaire pour les state snapshots."""
    state_dir = tmp_path / "guardian"
    state_dir.mkdir()
    return state_dir


@pytest.fixture
def config(tmp_state_dir):
    """Configuration Guardian avec répertoire temporaire."""
    return GuardianConfig(
        state_dir=tmp_state_dir,
        max_restarts_per_hour=5,
        base_restart_delay=0.01,  # Très court pour les tests
        max_restart_delay=0.1,
        stable_threshold_seconds=0.5,
    )


@pytest.fixture
def guardian(config):
    """Instance Guardian pour les tests."""
    return Guardian(config)


# ══════════════════════════════════════════════════════
# 1. TestGuardianConfig — Configuration (~4 tests)
# ══════════════════════════════════════════════════════

class TestGuardianConfig:
    """Tests de la configuration du Guardian."""

    def test_default_values(self):
        """Les valeurs par défaut sont raisonnables."""
        config = GuardianConfig()
        assert config.max_restarts_per_hour == 10
        assert config.base_restart_delay == 1.0
        assert config.max_restart_delay == 60.0
        assert config.stable_threshold_seconds == 300.0
        assert config.state_snapshot_interval == 600.0
        assert config.enabled is True
        assert config.state_dir == Path("data/guardian")

    def test_to_dict(self):
        """Sérialisation en dictionnaire."""
        config = GuardianConfig()
        d = config.to_dict()
        assert d["max_restarts_per_hour"] == 10
        assert d["base_restart_delay"] == 1.0
        assert d["enabled"] is True
        assert isinstance(d["state_dir"], str)

    def test_from_dict(self):
        """Désérialisation depuis un dictionnaire."""
        data = {
            "max_restarts_per_hour": 20,
            "base_restart_delay": 2.0,
            "state_dir": "/tmp/test",
            "enabled": False,
        }
        config = GuardianConfig.from_dict(data)
        assert config.max_restarts_per_hour == 20
        assert config.base_restart_delay == 2.0
        assert config.state_dir == Path("/tmp/test")
        assert config.enabled is False

    def test_from_dict_ignores_unknown_keys(self):
        """Les clés inconnues sont ignorées."""
        data = {
            "max_restarts_per_hour": 5,
            "unknown_key": "should be ignored",
            "another_unknown": 42,
        }
        config = GuardianConfig.from_dict(data)
        assert config.max_restarts_per_hour == 5
        assert not hasattr(config, "unknown_key")


# ══════════════════════════════════════════════════════
# 2. TestStateSnapshot — Sauvegarde d'état (~6 tests)
# ══════════════════════════════════════════════════════

class TestStateSnapshot:
    """Tests de la sauvegarde/restauration d'état."""

    def test_default_values(self):
        """Valeurs par défaut correctes."""
        snapshot = StateSnapshot()
        assert snapshot.heartbeat_pulse_count == 0
        assert snapshot.turn_count == 0
        assert snapshot.active_tasks == []
        assert snapshot.shutdown_reason == "unknown"
        assert snapshot.uptime_seconds == 0.0
        assert snapshot.restart_count == 0
        assert snapshot.timestamp  # Non vide

    def test_to_dict(self):
        """Sérialisation correcte."""
        snapshot = StateSnapshot(
            heartbeat_pulse_count=42,
            turn_count=10,
            active_tasks=["task-1", "task-2"],
            shutdown_reason="crash",
            uptime_seconds=3600.5,
            restart_count=3,
        )
        d = snapshot.to_dict()
        assert d["heartbeat_pulse_count"] == 42
        assert d["turn_count"] == 10
        assert d["active_tasks"] == ["task-1", "task-2"]
        assert d["shutdown_reason"] == "crash"
        assert d["uptime_seconds"] == 3600.5
        assert d["restart_count"] == 3

    def test_from_dict(self):
        """Désérialisation round-trip."""
        original = StateSnapshot(
            heartbeat_pulse_count=99,
            shutdown_reason="signal",
            uptime_seconds=1234.5,
        )
        restored = StateSnapshot.from_dict(original.to_dict())
        assert restored.heartbeat_pulse_count == 99
        assert restored.shutdown_reason == "signal"
        assert restored.uptime_seconds == 1234.5

    def test_save_and_load(self, tmp_state_dir):
        """Sauvegarde sur disque et rechargement."""
        snapshot = StateSnapshot(
            heartbeat_pulse_count=15,
            shutdown_reason="crash",
            active_tasks=["abc-123"],
        )
        snapshot.save(tmp_state_dir)

        # Vérifier que le fichier existe
        assert (tmp_state_dir / "state.json").exists()

        # Recharger
        loaded = StateSnapshot.load(tmp_state_dir)
        assert loaded is not None
        assert loaded.heartbeat_pulse_count == 15
        assert loaded.shutdown_reason == "crash"
        assert loaded.active_tasks == ["abc-123"]

    def test_load_missing_file(self, tmp_state_dir):
        """Fichier manquant → retourne None."""
        loaded = StateSnapshot.load(tmp_state_dir)
        assert loaded is None

    def test_clear(self, tmp_state_dir):
        """Suppression du fichier state."""
        snapshot = StateSnapshot(shutdown_reason="test")
        snapshot.save(tmp_state_dir)
        assert (tmp_state_dir / "state.json").exists()

        StateSnapshot.clear(tmp_state_dir)
        assert not (tmp_state_dir / "state.json").exists()


# ══════════════════════════════════════════════════════
# 3. TestGracefulShutdown — Gestion des signaux (~6 tests)
# ══════════════════════════════════════════════════════

class TestGracefulShutdown:
    """Tests de la gestion gracieuse des signaux."""

    def test_initial_state(self):
        """État initial correct."""
        gs = GracefulShutdown()
        assert gs.shutdown_requested is False
        assert gs.uptime_seconds >= 0

    def test_uptime_increases(self):
        """L'uptime augmente avec le temps."""
        gs = GracefulShutdown()
        time.sleep(0.05)
        assert gs.uptime_seconds > 0

    def test_add_cleanup_callback(self):
        """Les callbacks de cleanup sont enregistrés."""
        gs = GracefulShutdown()
        callbacks_called = []

        gs.add_cleanup_callback(lambda: callbacks_called.append("cb1"))
        gs.add_cleanup_callback(lambda: callbacks_called.append("cb2"))

        # Simuler un cleanup manuel
        gs._run_cleanup(shutdown_reason="test")

        assert callbacks_called == ["cb1", "cb2"]

    def test_cleanup_runs_once(self):
        """Le cleanup ne s'exécute qu'une seule fois."""
        gs = GracefulShutdown()
        call_count = [0]

        gs.add_cleanup_callback(lambda: call_count.__setitem__(0, call_count[0] + 1))

        gs._run_cleanup(shutdown_reason="test")
        gs._run_cleanup(shutdown_reason="test")

        assert call_count[0] == 1

    def test_save_state(self, tmp_state_dir):
        """Sauvegarde manuelle du state."""
        gs = GracefulShutdown()
        gs.set_state_dir(tmp_state_dir)

        gs.save_state(shutdown_reason="user_quit", turn_count=5)

        loaded = StateSnapshot.load(tmp_state_dir)
        assert loaded is not None
        assert loaded.shutdown_reason == "user_quit"
        assert loaded.turn_count == 5

    def test_clear_state(self, tmp_state_dir):
        """Nettoyage du state."""
        gs = GracefulShutdown()
        gs.set_state_dir(tmp_state_dir)

        gs.save_state(shutdown_reason="test")
        assert (tmp_state_dir / "state.json").exists()

        gs.clear_state()
        assert not (tmp_state_dir / "state.json").exists()


# ══════════════════════════════════════════════════════
# 4. TestGuardian — Superviseur de process (~9 tests)
# ══════════════════════════════════════════════════════

class TestGuardian:
    """Tests du superviseur de process."""

    def test_initial_state(self, guardian):
        """État initial correct."""
        assert guardian._running is False
        assert guardian._restart_count == 0
        assert guardian._total_restarts == 0
        assert guardian._child_process is None

    def test_disabled_guardian(self, config):
        """Guardian désactivé ne fait rien."""
        config.enabled = False
        g = Guardian(config)
        g.run()  # Ne devrait pas bloquer
        assert g._running is False

    def test_find_neo_command(self, guardian):
        """La commande Neo est correcte."""
        cmd = guardian._find_neo_command()
        assert cmd == [sys.executable, "-m", "neo_core.cli", "chat"]

    def test_should_restart_within_limit(self, guardian):
        """Restart autorisé quand sous la limite."""
        assert guardian._should_restart() is True
        assert guardian._should_restart() is True
        assert guardian._should_restart() is True

    def test_should_restart_exceeds_limit(self, config):
        """Restart refusé quand la limite est atteinte."""
        config.max_restarts_per_hour = 3
        g = Guardian(config)

        assert g._should_restart() is True
        assert g._should_restart() is True
        assert g._should_restart() is True
        # 4ème restart → refusé (max=3)
        assert g._should_restart() is False

    def test_backoff_exponential(self, guardian):
        """Le backoff augmente exponentiellement."""
        initial = guardian._current_delay
        guardian._wait_backoff()  # 0.01
        assert guardian._current_delay == initial * 2

        guardian._wait_backoff()  # 0.02
        assert guardian._current_delay == initial * 4

    def test_backoff_capped(self, config):
        """Le backoff est plafonné."""
        config.max_restart_delay = 0.05
        g = Guardian(config)

        for _ in range(20):
            g._wait_backoff()

        assert g._current_delay <= config.max_restart_delay

    def test_get_status(self, guardian):
        """get_status retourne les bonnes infos."""
        status = guardian.get_status()
        assert "running" in status
        assert "total_restarts" in status
        assert "current_backoff_delay" in status
        assert "restarts_last_hour" in status
        assert "max_restarts_per_hour" in status
        assert "child_pid" in status
        assert "uptime" in status
        assert status["running"] is False
        assert status["child_pid"] is None

    def test_stop(self, guardian):
        """Stop arrête le guardian."""
        guardian._running = True
        guardian.stop()
        assert guardian._running is False

    @patch("subprocess.Popen")
    def test_run_child_success(self, mock_popen, guardian):
        """_run_child lance un subprocess et retourne le exit code."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        exit_code = guardian._run_child()

        assert exit_code == 0
        mock_popen.assert_called_once()
        # Vérifier que NEO_GUARDIAN_MODE est set dans l'env
        call_kwargs = mock_popen.call_args
        env = call_kwargs[1].get("env", {})
        assert env.get("NEO_GUARDIAN_MODE") == "1"

    @patch("subprocess.Popen")
    def test_run_child_crash(self, mock_popen, guardian):
        """_run_child retourne le exit code du crash."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process

        exit_code = guardian._run_child()
        assert exit_code == 1

    @patch("subprocess.Popen")
    def test_run_child_restart_code(self, mock_popen, guardian):
        """_run_child retourne le code 42 (restart)."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.wait.return_value = EXIT_CODE_RESTART
        mock_popen.return_value = mock_process

        exit_code = guardian._run_child()
        assert exit_code == 42


# ══════════════════════════════════════════════════════
# 5. TestGuardianRun — Boucle principale (~4 tests)
# ══════════════════════════════════════════════════════

class TestGuardianRun:
    """Tests de la boucle principale du Guardian."""

    @patch.object(Guardian, "_run_child")
    def test_normal_exit_no_restart(self, mock_run_child, config):
        """Exit code 0 → pas de restart."""
        mock_run_child.return_value = EXIT_CODE_NORMAL
        g = Guardian(config)
        g.run()
        assert mock_run_child.call_count == 1
        assert g._total_restarts == 0

    @patch.object(Guardian, "_run_child")
    def test_restart_code_immediate(self, mock_run_child, config):
        """Exit code 42 → restart immédiat, puis exit 0 pour arrêter."""
        mock_run_child.side_effect = [EXIT_CODE_RESTART, EXIT_CODE_NORMAL]
        g = Guardian(config)
        g.run()
        assert mock_run_child.call_count == 2
        assert g._total_restarts == 1

    @patch.object(Guardian, "_run_child")
    def test_crash_with_backoff(self, mock_run_child, config):
        """Crash → restart avec backoff, puis exit 0."""
        mock_run_child.side_effect = [1, EXIT_CODE_NORMAL]
        g = Guardian(config)
        g.run()
        assert mock_run_child.call_count == 2
        assert g._total_restarts == 1

    @patch.object(Guardian, "_run_child")
    def test_max_restarts_protection(self, mock_run_child, config):
        """Protection contre boucle infinie de restarts."""
        config.max_restarts_per_hour = 3
        # Toujours crasher
        mock_run_child.return_value = 1
        g = Guardian(config)
        g.run()
        # Ne devrait pas dépasser max_restarts_per_hour + 1 appels
        # (3 restarts autorisés + le crash initial qui déclenche le check)
        assert mock_run_child.call_count <= config.max_restarts_per_hour + 1

    @patch.object(Guardian, "_run_child")
    def test_guardian_log_created(self, mock_run_child, config):
        """Le Guardian écrit un log."""
        mock_run_child.return_value = EXIT_CODE_NORMAL
        g = Guardian(config)
        g.run()

        log_file = config.state_dir / "guardian.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "Guardian démarré" in content
        assert "Guardian arrêté" in content


# ══════════════════════════════════════════════════════
# 6. TestExitCodes — Codes de sortie (~3 tests)
# ══════════════════════════════════════════════════════

class TestExitCodes:
    """Tests des codes de sortie."""

    def test_exit_code_normal(self):
        """Code de sortie normal."""
        assert EXIT_CODE_NORMAL == 0

    def test_exit_code_restart(self):
        """Code de sortie restart."""
        assert EXIT_CODE_RESTART == 42

    def test_exit_codes_different(self):
        """Les codes sont différents."""
        assert EXIT_CODE_NORMAL != EXIT_CODE_RESTART


# ══════════════════════════════════════════════════════
# 7. TestIntegration — Intégrations (~3 tests)
# ══════════════════════════════════════════════════════

class TestIntegration:
    """Tests d'intégration avec les autres modules."""

    def test_chat_imports_guardian(self):
        """chat.py importe correctement les composants Guardian."""
        from neo_core.cli.chat import GracefulShutdown, StateSnapshot, EXIT_CODE_RESTART
        assert GracefulShutdown is not None
        assert StateSnapshot is not None
        assert EXIT_CODE_RESTART == 42

    def test_guardian_state_dir_creation(self, tmp_path):
        """Le Guardian crée le répertoire state si nécessaire."""
        state_dir = tmp_path / "new" / "guardian"
        snapshot = StateSnapshot(shutdown_reason="test")
        snapshot.save(state_dir)
        assert state_dir.exists()
        assert (state_dir / "state.json").exists()

    def test_systemd_service_exists(self):
        """Le fichier systemd existe."""
        service_file = Path(__file__).parent.parent / "deploy" / "neo-guardian.service"
        assert service_file.exists()
        content = service_file.read_text()
        assert "neo guardian" in content
        assert "[Service]" in content
        assert "[Unit]" in content
