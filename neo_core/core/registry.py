"""
Neo Core — CoreRegistry : Singleton du Core
==============================================
Garantit qu'il n'existe qu'une seule instance de Vox/Brain/Memory
dans tout le processus, quel que soit le consommateur
(CLI chat, daemon, API, Telegram).

Usage :
    from neo_core.core.registry import core_registry

    # Premier appel → crée les instances
    vox = core_registry.get_vox()

    # Appels suivants → retourne la MÊME instance
    vox2 = core_registry.get_vox()
    assert vox is vox2  # True
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class CoreRegistry:
    """
    Singleton registry pour les agents Neo Core.

    Garantit une seule instance de Memory, Brain et Vox
    par processus. Thread-safe via un lock.
    """

    _instance: Optional["CoreRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CoreRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._memory = None
        self._brain = None
        self._vox = None
        self._config = None
        self._bootstrap_lock = threading.Lock()
        self._initialized = True

    def _bootstrap(self) -> None:
        """Bootstrap les 3 agents une seule fois."""
        if self._vox is not None:
            return

        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*RuntimeWarning.*")
        warnings.filterwarnings("ignore", message=".*renamed.*")

        from neo_core.config import NeoConfig
        from neo_core.core.brain import Brain
        from neo_core.core.memory_agent import MemoryAgent
        from neo_core.core.vox import Vox

        self._config = NeoConfig()

        # Bootstrap providers multi-LLM (best-effort)
        try:
            from neo_core.providers.bootstrap import bootstrap_providers
            bootstrap_providers(self._config)
        except Exception:
            pass

        # Instanciation unique des 3 agents
        self._memory = MemoryAgent(config=self._config)
        self._memory.initialize()

        self._brain = Brain(config=self._config)
        self._brain.connect_memory(self._memory)

        self._vox = Vox(config=self._config)
        self._vox.connect(brain=self._brain, memory=self._memory)

        logger.info(
            "CoreRegistry bootstrap — single instance created "
            "(Memory=%s, Brain=%s, Vox=%s)",
            id(self._memory), id(self._brain), id(self._vox),
        )

    @property
    def is_bootstrapped(self) -> bool:
        """True si les agents ont été initialisés."""
        return self._vox is not None

    def get_vox(self):
        """Retourne l'instance unique de Vox (bootstrap si nécessaire)."""
        with self._bootstrap_lock:
            self._bootstrap()
        return self._vox

    def get_brain(self):
        """Retourne l'instance unique de Brain."""
        with self._bootstrap_lock:
            self._bootstrap()
        return self._brain

    def get_memory(self):
        """Retourne l'instance unique de Memory."""
        with self._bootstrap_lock:
            self._bootstrap()
        return self._memory

    def get_config(self):
        """Retourne la config NeoConfig."""
        with self._bootstrap_lock:
            self._bootstrap()
        return self._config

    def reset(self) -> None:
        """
        Reset complet (pour les tests uniquement).

        Remet le registry à l'état initial.
        """
        with self._bootstrap_lock:
            self._memory = None
            self._brain = None
            self._vox = None
            self._config = None
            logger.info("CoreRegistry reset")

    @classmethod
    def _reset_singleton(cls) -> None:
        """Reset le singleton lui-même (tests uniquement)."""
        with cls._lock:
            cls._instance = None


# Instance globale unique
core_registry = CoreRegistry()
