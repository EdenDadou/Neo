"""
Tests — CoreRegistry : Singleton du Core
==========================================
Vérifie qu'il n'existe qu'une seule instance de Vox/Brain/Memory.
"""

import threading
from unittest.mock import patch, MagicMock

import pytest


class TestCoreRegistrySingleton:
    """Le CoreRegistry est un vrai singleton."""

    def test_singleton_identity(self):
        """Deux instanciations retournent le même objet."""
        from neo_core.infra.registry import CoreRegistry
        CoreRegistry._reset_singleton()
        a = CoreRegistry()
        b = CoreRegistry()
        assert a is b
        CoreRegistry._reset_singleton()

    def test_singleton_across_imports(self):
        """L'import du module retourne toujours la même instance."""
        from neo_core.infra.registry import core_registry as r1
        from neo_core.infra.registry import core_registry as r2
        assert r1 is r2

    def test_reset_singleton(self):
        """_reset_singleton crée une nouvelle instance."""
        from neo_core.infra.registry import CoreRegistry
        a = CoreRegistry()
        CoreRegistry._reset_singleton()
        b = CoreRegistry()
        assert a is not b
        CoreRegistry._reset_singleton()


class TestCoreRegistryBootstrap:
    """Le bootstrap crée une seule instance partagée."""

    @patch("neo_core.memory.agent.MemoryAgent")
    @patch("neo_core.brain.core.Brain")
    @patch("neo_core.vox.interface.Vox")
    @patch("neo_core.config.NeoConfig")
    def test_bootstrap_creates_once(self, MockConfig, MockVox, MockBrain, MockMemory):
        """Bootstrap ne crée les agents qu'une seule fois."""
        from neo_core.infra.registry import CoreRegistry
        CoreRegistry._reset_singleton()
        reg = CoreRegistry()

        vox1 = reg.get_vox()
        assert vox1 is not None

        # Deuxième appel → même instance
        vox2 = reg.get_vox()
        assert vox1 is vox2
        CoreRegistry._reset_singleton()

    @patch("neo_core.memory.agent.MemoryAgent")
    @patch("neo_core.brain.core.Brain")
    @patch("neo_core.vox.interface.Vox")
    @patch("neo_core.config.NeoConfig")
    def test_all_getters_same_instance(self, MockConfig, MockVox, MockBrain, MockMemory):
        """get_vox, get_brain, get_memory retournent les mêmes objets."""
        from neo_core.infra.registry import CoreRegistry
        CoreRegistry._reset_singleton()
        reg = CoreRegistry()

        vox = reg.get_vox()
        brain = reg.get_brain()
        memory = reg.get_memory()

        assert reg.get_vox() is vox
        assert reg.get_brain() is brain
        assert reg.get_memory() is memory
        CoreRegistry._reset_singleton()

    @patch("neo_core.memory.agent.MemoryAgent")
    @patch("neo_core.brain.core.Brain")
    @patch("neo_core.vox.interface.Vox")
    @patch("neo_core.config.NeoConfig")
    def test_reset_allows_rebootstrap(self, MockConfig, MockVox, MockBrain, MockMemory):
        """reset() remet is_bootstrapped à False et re-bootstrap."""
        from neo_core.infra.registry import CoreRegistry
        CoreRegistry._reset_singleton()
        reg = CoreRegistry()

        reg.get_vox()
        assert reg.is_bootstrapped

        reg.reset()
        assert not reg.is_bootstrapped

        # Après reset, un nouvel appel re-bootstrap
        MockVox.reset_mock()
        reg.get_vox()
        assert reg.is_bootstrapped
        assert MockVox.called  # Vox() a été appelé à nouveau
        CoreRegistry._reset_singleton()

    @patch("neo_core.memory.agent.MemoryAgent")
    @patch("neo_core.brain.core.Brain")
    @patch("neo_core.vox.interface.Vox")
    @patch("neo_core.config.NeoConfig")
    def test_is_bootstrapped(self, MockConfig, MockVox, MockBrain, MockMemory):
        """is_bootstrapped reflète l'état."""
        from neo_core.infra.registry import CoreRegistry
        CoreRegistry._reset_singleton()
        reg = CoreRegistry()

        assert not reg.is_bootstrapped
        reg.get_vox()
        assert reg.is_bootstrapped
        reg.reset()
        assert not reg.is_bootstrapped
        CoreRegistry._reset_singleton()

    @patch("neo_core.memory.agent.MemoryAgent")
    @patch("neo_core.brain.core.Brain")
    @patch("neo_core.vox.interface.Vox")
    @patch("neo_core.config.NeoConfig")
    def test_get_config(self, MockConfig, MockVox, MockBrain, MockMemory):
        """get_config retourne la config."""
        from neo_core.infra.registry import CoreRegistry
        CoreRegistry._reset_singleton()
        reg = CoreRegistry()

        config = reg.get_config()
        assert config is not None
        assert reg.get_config() is config
        CoreRegistry._reset_singleton()


class TestCoreRegistryThreadSafety:
    """Le registry est thread-safe."""

    @patch("neo_core.memory.agent.MemoryAgent")
    @patch("neo_core.brain.core.Brain")
    @patch("neo_core.vox.interface.Vox")
    @patch("neo_core.config.NeoConfig")
    def test_concurrent_get_vox(self, MockConfig, MockVox, MockBrain, MockMemory):
        """Plusieurs threads obtiennent la même instance."""
        from neo_core.infra.registry import CoreRegistry
        CoreRegistry._reset_singleton()
        reg = CoreRegistry()

        results = []

        def get_vox():
            v = reg.get_vox()
            results.append(id(v))

        threads = [threading.Thread(target=get_vox) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1
        CoreRegistry._reset_singleton()


class TestCoreRegistryIntegration:
    """Tests d'intégration avec les vrais consommateurs."""

    def test_chat_bootstrap_uses_registry(self):
        """chat.bootstrap() utilise le CoreRegistry."""
        import inspect
        from neo_core.vox.cli import chat
        source = inspect.getsource(chat.bootstrap)
        assert "core_registry" in source
        # Ne crée plus directement les agents
        assert "MemoryAgent(" not in source
        assert "Brain(" not in source

    def test_api_server_uses_registry(self):
        """api/server.py utilise le CoreRegistry."""
        import inspect
        from neo_core.vox.api import server
        source = inspect.getsource(server.NeoCore.initialize)
        assert "core_registry" in source
        assert "MemoryAgent(" not in source

    def test_daemon_uses_registry(self):
        """daemon.py utilise le CoreRegistry."""
        import inspect
        from neo_core.infra import daemon
        source = inspect.getsource(daemon._run_daemon)
        assert "core_registry" in source
        # Ne fait plus bootstrap() depuis chat.py
        assert "from neo_core.vox.cli.chat import bootstrap" not in source

    def test_module_exports(self):
        """Le module core exporte core_registry."""
        from neo_core.infra import core_registry
        assert hasattr(core_registry, "get_vox")
        assert hasattr(core_registry, "get_brain")
        assert hasattr(core_registry, "get_memory")
        assert hasattr(core_registry, "get_config")
        assert hasattr(core_registry, "reset")

    def test_no_duplicate_creation_in_sources(self):
        """Aucun consommateur ne crée MemoryAgent/Brain/Vox directement."""
        import inspect
        from neo_core.vox.cli import chat
        from neo_core.vox.api import server

        # chat.bootstrap
        chat_src = inspect.getsource(chat.bootstrap)
        assert "MemoryAgent(" not in chat_src

        # server.NeoCore.initialize
        server_src = inspect.getsource(server.NeoCore.initialize)
        assert "MemoryAgent(" not in server_src
