"""
Neo Core — Shared Test Fixtures
=================================
Fixtures communes utilisées par l'ensemble de la suite de tests.
Fournit des instances pré-configurées de Config, MemoryAgent, Brain, Vox.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from neo_core.config import NeoConfig


@pytest.fixture
def config(tmp_path):
    """NeoConfig en mode mock (pas de clé API)."""
    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "",
        "GROQ_API_KEY": "",
        "GEMINI_API_KEY": "",
    }, clear=False):
        cfg = NeoConfig()
        # Override data_dir pour isolation
        cfg.data_dir = tmp_path / "data"
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        cfg.memory.storage_path = tmp_path / "memory"
        cfg.memory.storage_path.mkdir(parents=True, exist_ok=True)
        yield cfg


@pytest.fixture
def memory_agent(config):
    """MemoryAgent initialisé avec stockage temporaire."""
    from neo_core.memory.agent import MemoryAgent
    agent = MemoryAgent(config=config)
    agent.initialize()
    return agent


@pytest.fixture
def brain(config, memory_agent):
    """Brain connecté à la mémoire."""
    from neo_core.brain.core import Brain
    b = Brain(config=config)
    b.connect_memory(memory_agent)
    return b


@pytest.fixture
def vox(config, brain, memory_agent):
    """Vox connecté à Brain et Memory."""
    from neo_core.vox.interface import Vox
    v = Vox(config=config)
    v.connect(brain=brain, memory=memory_agent)
    return v
