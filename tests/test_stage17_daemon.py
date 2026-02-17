"""
Tests Stage 17 — Daemon & systemd
====================================
Vérifie le daemon manager, le PID file, le systemd service, la CLI.

~25 tests au total.
"""

import os
import signal
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import psutil

from neo_core.core.daemon import (
    _read_pid,
    _write_pid,
    _remove_pid,
    _get_pid_file,
    _get_log_file,
    is_running,
    get_status,
    start,
    stop,
    generate_systemd_service,
    install_service,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════
# 1. TestPidFile — Gestion du PID file (~6 tests)
# ══════════════════════════════════════════════════════

class TestPidFile:
    """Tests de la gestion du PID file."""

    @pytest.fixture(autouse=True)
    def clean_pid(self, tmp_path):
        """Utilise un PID file temporaire."""
        with patch("neo_core.core.daemon._PID_FILE", tmp_path / ".neo.pid"):
            yield

    def test_write_and_read_pid(self, tmp_path):
        """write_pid + read_pid round-trip."""
        with patch("neo_core.core.daemon._PID_FILE", tmp_path / ".neo.pid"):
            _write_pid(12345)
            assert _read_pid() == 12345

    def test_read_pid_no_file(self, tmp_path):
        """read_pid retourne None si pas de fichier."""
        with patch("neo_core.core.daemon._PID_FILE", tmp_path / "nonexistent.pid"):
            assert _read_pid() is None

    def test_remove_pid(self, tmp_path):
        """remove_pid supprime le fichier."""
        with patch("neo_core.core.daemon._PID_FILE", tmp_path / ".neo.pid"):
            _write_pid(99999)
            _remove_pid()
            assert _read_pid() is None

    def test_remove_pid_no_file(self, tmp_path):
        """remove_pid ne crash pas si pas de fichier."""
        with patch("neo_core.core.daemon._PID_FILE", tmp_path / "nonexistent.pid"):
            _remove_pid()  # Ne devrait pas lever d'exception

    def test_pid_file_creates_parent(self, tmp_path):
        """write_pid crée le répertoire parent si nécessaire."""
        pid_path = tmp_path / "subdir" / ".neo.pid"
        with patch("neo_core.core.daemon._PID_FILE", pid_path):
            _write_pid(11111)
            assert pid_path.exists()

    def test_read_pid_invalid_content(self, tmp_path):
        """read_pid retourne None si contenu invalide."""
        pid_file = tmp_path / ".neo.pid"
        pid_file.write_text("not_a_number")
        with patch("neo_core.core.daemon._PID_FILE", pid_file):
            assert _read_pid() is None


# ══════════════════════════════════════════════════════
# 2. TestIsRunning — Détection de process (~5 tests)
# ══════════════════════════════════════════════════════

class TestIsRunning:
    """Tests de la détection de processus."""

    def test_not_running_no_pid(self, tmp_path):
        """is_running retourne False sans PID file."""
        with patch("neo_core.core.daemon._PID_FILE", tmp_path / ".neo.pid"):
            assert is_running() is False

    def test_not_running_stale_pid(self, tmp_path):
        """is_running retourne False avec un PID périmé."""
        with patch("neo_core.core.daemon._PID_FILE", tmp_path / ".neo.pid"):
            _write_pid(999999999)  # PID très probablement inexistant
            assert is_running() is False

    def test_running_current_process(self, tmp_path):
        """is_running retourne True pour le processus courant."""
        with patch("neo_core.core.daemon._PID_FILE", tmp_path / ".neo.pid"):
            _write_pid(os.getpid())
            assert is_running() is True

    def test_get_status_not_running(self, tmp_path):
        """get_status retourne running=False si daemon inactif."""
        with patch("neo_core.core.daemon._PID_FILE", tmp_path / ".neo.pid"):
            status = get_status()
            assert status["running"] is False
            assert status["pid"] is None

    def test_get_status_running(self, tmp_path):
        """get_status retourne des infos si daemon actif."""
        with patch("neo_core.core.daemon._PID_FILE", tmp_path / ".neo.pid"):
            _write_pid(os.getpid())
            status = get_status()
            assert status["running"] is True
            assert status["pid"] == os.getpid()
            assert "uptime_seconds" in status
            assert "memory_mb" in status


# ══════════════════════════════════════════════════════
# 3. TestStartStop — Démarrage/arrêt (~4 tests)
# ══════════════════════════════════════════════════════

class TestStartStop:
    """Tests de start/stop (sans fork réel)."""

    def test_start_already_running(self, tmp_path):
        """start() retourne erreur si déjà en cours."""
        with patch("neo_core.core.daemon._PID_FILE", tmp_path / ".neo.pid"):
            _write_pid(os.getpid())
            result = start(foreground=False)
            assert result["success"] is False
            assert "déjà" in result["message"]

    def test_stop_not_running(self, tmp_path):
        """stop() retourne erreur si pas de daemon."""
        with patch("neo_core.core.daemon._PID_FILE", tmp_path / ".neo.pid"):
            result = stop()
            assert result["success"] is False
            assert "pas en cours" in result["message"].lower() or "n'est pas" in result["message"].lower()

    def test_stop_stale_pid(self, tmp_path):
        """stop() gère un PID périmé."""
        with patch("neo_core.core.daemon._PID_FILE", tmp_path / ".neo.pid"):
            _write_pid(999999999)
            result = stop()
            assert result["success"] is False

    def test_start_returns_dict(self, tmp_path):
        """start() retourne toujours un dict avec success et message."""
        with patch("neo_core.core.daemon._PID_FILE", tmp_path / ".neo.pid"):
            _write_pid(os.getpid())
            result = start()
            assert isinstance(result, dict)
            assert "success" in result
            assert "message" in result


# ══════════════════════════════════════════════════════
# 4. TestSystemd — Génération de service (~5 tests)
# ══════════════════════════════════════════════════════

class TestSystemd:
    """Tests de la génération du fichier systemd."""

    def test_generate_service_content(self):
        """generate_systemd_service() retourne un contenu valide."""
        content = generate_systemd_service(user="eden", python_path="/usr/bin/python3")
        assert "[Unit]" in content
        assert "[Service]" in content
        assert "[Install]" in content
        assert "User=eden" in content
        assert "ExecStart=" in content

    def test_generate_service_has_restart(self):
        """Le service a Restart=always."""
        content = generate_systemd_service()
        assert "Restart=always" in content

    def test_generate_service_has_security(self):
        """Le service a les directives de sécurité."""
        content = generate_systemd_service()
        assert "NoNewPrivileges=true" in content
        assert "ProtectSystem=strict" in content

    def test_generate_service_has_env_file(self):
        """Le service charge .env."""
        content = generate_systemd_service()
        assert "EnvironmentFile" in content
        assert ".env" in content

    def test_install_service_no_sudo(self, tmp_path):
        """install_service() gère l'absence de sudo."""
        with patch("neo_core.core.daemon._PROJECT_ROOT", tmp_path):
            with patch("neo_core.core.daemon._DATA_DIR", tmp_path / "data"):
                result = install_service()
                # Sans sudo, on devrait avoir un fichier local ou une erreur gérée
                assert isinstance(result, dict)
                assert "commands" in result


# ══════════════════════════════════════════════════════
# 5. TestInfra — Structure du Stage 17 (~5 tests)
# ══════════════════════════════════════════════════════

class TestDaemonInfra:
    """Tests de la structure du Stage 17."""

    def test_daemon_module_exists(self):
        """neo_core/core/daemon.py existe."""
        assert (PROJECT_ROOT / "neo_core" / "core" / "daemon.py").exists()

    def test_cli_has_start_command(self):
        """La CLI a la commande start."""
        import inspect
        import neo_core.cli
        source = inspect.getsource(neo_core.cli.main)
        assert '"start"' in source or "'start'" in source

    def test_cli_has_stop_command(self):
        """La CLI a la commande stop."""
        import inspect
        import neo_core.cli
        source = inspect.getsource(neo_core.cli.main)
        assert '"stop"' in source or "'stop'" in source

    def test_cli_has_install_service(self):
        """La CLI a la commande install-service."""
        import inspect
        import neo_core.cli
        source = inspect.getsource(neo_core.cli.main)
        assert "install-service" in source or "install_service" in source

    def test_version_updated(self):
        """La version est correcte."""
        content = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert 'version = "0.' in content
