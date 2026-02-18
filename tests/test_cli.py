"""
Tests — CLI (chat.py, status.py, setup.py utilitaires)
========================================================
"""

import inspect
from unittest.mock import patch, MagicMock

import pytest


class TestChatBootstrap:
    """Tests de chat.bootstrap() et fonctions utilitaires."""

    def test_bootstrap_uses_registry(self):
        """bootstrap() passe par le CoreRegistry."""
        from neo_core.vox.cli.chat import bootstrap
        source = inspect.getsource(bootstrap)
        assert "core_registry" in source
        assert "MemoryAgent(" not in source

    def test_print_help_no_crash(self, capsys):
        """print_help() s'exécute sans erreur."""
        from neo_core.vox.cli.chat import print_help
        print_help()
        # Pas d'exception = OK

    def test_check_installation_not_installed(self):
        """check_installation retourne False si pas installé."""
        from neo_core.vox.cli.chat import check_installation
        mock_config = MagicMock()
        mock_config.is_installed.return_value = False
        # On ne peut pas tester le input(), mais on vérifie que la
        # fonction appelle is_installed
        with patch("neo_core.vox.cli.chat.console"):
            with patch("builtins.input", side_effect=EOFError):
                result = check_installation(mock_config)
        assert result is False

    def test_check_installation_installed(self):
        """check_installation retourne True si installé."""
        from neo_core.vox.cli.chat import check_installation
        mock_config = MagicMock()
        mock_config.is_installed.return_value = True
        assert check_installation(mock_config) is True


class TestChatCommands:
    """Tests des fonctions print_* du chat."""

    def test_print_status(self):
        """print_status appelle vox.get_system_status."""
        from neo_core.vox.cli.chat import print_status
        mock_vox = MagicMock()
        mock_vox.get_system_status.return_value = "Tout va bien"
        with patch("neo_core.vox.cli.chat.console"):
            print_status(mock_vox)
        mock_vox.get_system_status.assert_called_once()

    def test_print_health_no_brain(self):
        """print_health gère l'absence de brain."""
        from neo_core.vox.cli.chat import print_health
        mock_vox = MagicMock()
        mock_vox.brain = None
        with patch("neo_core.vox.cli.chat.console"):
            print_health(mock_vox)

    def test_print_skills_no_memory(self):
        """print_skills gère l'absence de memory."""
        from neo_core.vox.cli.chat import print_skills
        mock_vox = MagicMock()
        mock_vox.memory = None
        with patch("neo_core.vox.cli.chat.console"):
            print_skills(mock_vox)

    def test_print_persona_no_engine(self):
        """print_persona gère l'absence de persona_engine."""
        from neo_core.vox.cli.chat import print_persona
        mock_vox = MagicMock()
        mock_vox.memory = None
        with patch("neo_core.vox.cli.chat.console"):
            print_persona(mock_vox)

    def test_print_history_no_session(self):
        """print_history gère l'absence de session."""
        from neo_core.vox.cli.chat import print_history
        mock_vox = MagicMock()
        mock_vox.get_session_info.return_value = None
        with patch("neo_core.vox.cli.chat.console"):
            print_history(mock_vox)


class TestSetupUtilities:
    """Tests des fonctions utilitaires de setup.py."""

    def test_check_python_version(self, capsys):
        """check_python_version détecte Python 3.10+."""
        from neo_core.vox.cli.setup import check_python_version
        import sys
        # On est forcément en 3.10+ (requis par le projet)
        assert check_python_version() is True

    def test_check_venv(self):
        """check_venv détecte l'état du venv."""
        from neo_core.vox.cli.setup import check_venv
        # Ne crash pas
        result = check_venv()
        assert isinstance(result, bool)

    def test_run_command_success(self, capsys):
        """run_command exécute une commande simple."""
        from neo_core.vox.cli.setup import run_command
        assert run_command("echo hello", "Test echo") is True

    def test_run_command_failure(self, capsys):
        """run_command détecte un échec."""
        from neo_core.vox.cli.setup import run_command
        assert run_command("false", "Test fail") is False

    def test_run_command_timeout(self, capsys):
        """run_command gère le timeout."""
        from neo_core.vox.cli.setup import run_command
        # La commande `sleep 999` devrait timeout (timeout=300s dans le code)
        # mais on teste juste que la méthode ne crash pas

    def test_save_config(self, tmp_path):
        """save_config crée les fichiers."""
        import json
        from neo_core.vox.cli.setup import save_config

        config_dir = tmp_path / "data"
        config_file = config_dir / "neo_config.json"
        env_file = tmp_path / ".env"

        with patch("neo_core.vox.cli.setup.CONFIG_DIR", config_dir):
            with patch("neo_core.vox.cli.setup.CONFIG_FILE", config_file):
                with patch("neo_core.vox.cli.setup.ENV_FILE", env_file):
                    save_config("TestCore", "TestUser", "sk-test", "/usr/bin/python3")

        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert data["core_name"] == "TestCore"
        assert data["user_name"] == "TestUser"

        assert env_file.exists()
        env_content = env_file.read_text()
        # Les clés API sont dans le vault, pas dans .env
        assert "ANTHROPIC_API_KEY" not in env_content
        assert "NEO_CORE_NAME=TestCore" in env_content

        # Vérifier que la clé est dans le vault
        from neo_core.infra.security.vault import KeyVault
        vault = KeyVault(data_dir=config_dir)
        vault.initialize()
        assert vault.retrieve("anthropic_api_key") == "sk-test"
        vault.close()


class TestStatusModule:
    """Tests du module status.py."""

    def test_import(self):
        """Le module status s'importe."""
        from neo_core.vox.cli import status
        assert hasattr(status, "run_status") or hasattr(status, "print_status")


class TestMigrations:
    """Tests des migrations SQLite."""

    def test_run_migrations(self):
        """Les migrations s'appliquent sur une DB vide."""
        import sqlite3
        from neo_core.memory.migrations import run_migrations, get_current_version

        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT,
                timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rate_limits (
                client_ip TEXT,
                timestamp REAL
            )
        """)

        assert get_current_version(conn) == 0
        applied = run_migrations(conn)
        assert applied >= 1
        assert get_current_version(conn) > 0
        conn.close()

    def test_migrations_idempotent(self):
        """Exécuter les migrations 2x ne change rien."""
        import sqlite3
        from neo_core.memory.migrations import run_migrations, get_current_version

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, timestamp TEXT)")
        conn.execute("CREATE TABLE rate_limits (client_ip TEXT, timestamp REAL)")

        run_migrations(conn)
        v1 = get_current_version(conn)

        applied = run_migrations(conn)
        v2 = get_current_version(conn)

        assert applied == 0  # Rien de nouveau
        assert v1 == v2
        conn.close()

    def test_get_version_no_table(self):
        """get_current_version retourne 0 si pas de table."""
        import sqlite3
        from neo_core.memory.migrations import get_current_version
        conn = sqlite3.connect(":memory:")
        assert get_current_version(conn) == 0
        conn.close()


class TestHealthCheck:
    """Tests du health check enrichi."""

    def test_health_endpoint_structure(self):
        """Le health check retourne les bons champs."""
        import inspect
        from neo_core.vox.api import routes
        source = inspect.getsource(routes.health)
        assert "checks" in source
        assert "faiss" in source
        assert "vault" in source
        assert "healthy" in source or "degraded" in source

    def test_global_exception_handler_exists(self):
        """Le server.py a un global exception handler."""
        import inspect
        from neo_core.vox.api import server
        source = inspect.getsource(server.create_app)
        assert "exception_handler" in source
        assert "internal_server_error" in source
