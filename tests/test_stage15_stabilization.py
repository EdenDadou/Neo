"""
Tests Stage 15 — Stabilisation & Optimisations
=================================================
Vérifie le jitter, le batch query mémoire, les deps, conftest et CI/CD.

~20 tests au total.
"""

import json
import os
import random
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from neo_core.infra.resilience import (
    RetryConfig,
    compute_backoff_delay,
)
from neo_core.memory.store import MemoryStore, MemoryRecord
from neo_core.config import NeoConfig, MemoryConfig


# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════
# 1. TestJitter — Jitter dans le backoff (~5 tests)
# ══════════════════════════════════════════════════════

class TestJitter:
    """Tests du jitter dans compute_backoff_delay."""

    def test_jitter_varies_delay(self):
        """Le jitter produit des délais différents entre deux appels."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=30.0)
        delays = {compute_backoff_delay(1, config) for _ in range(20)}
        # Avec jitter, on devrait avoir plusieurs valeurs distinctes
        assert len(delays) > 1

    def test_jitter_within_bounds(self):
        """Le délai avec jitter reste dans ±20% du délai de base."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=30.0)
        for _ in range(100):
            delay = compute_backoff_delay(0, config)
            # attempt=0 → base_delay * 2^0 = 1.0
            # jitter ±20% → [0.8, 1.2]
            assert 0.8 <= delay <= 1.2, f"Delay {delay} hors bornes [0.8, 1.2]"

    def test_jitter_attempt_2_bounds(self):
        """Le délai attempt=2 est dans les bonnes bornes."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=30.0)
        for _ in range(100):
            delay = compute_backoff_delay(2, config)
            # attempt=2 → 1.0 * 2^2 = 4.0, jitter → [3.2, 4.8]
            assert 3.2 <= delay <= 4.8, f"Delay {delay} hors bornes [3.2, 4.8]"

    def test_jitter_respects_max_delay(self):
        """Le jitter n'explose pas au-delà de max_delay * 1.2."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=10.0)
        for _ in range(100):
            delay = compute_backoff_delay(10, config)
            # 2^10 = 1024 → min(1024, 10) = 10, jitter → [8.0, 12.0]
            assert delay <= 12.0, f"Delay {delay} dépasse max_delay * 1.2"
            assert delay >= 8.0, f"Delay {delay} trop bas"

    def test_jitter_deterministic_with_seed(self):
        """Avec un seed fixe, le jitter est reproductible."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=30.0)
        random.seed(42)
        d1 = compute_backoff_delay(1, config)
        random.seed(42)
        d2 = compute_backoff_delay(1, config)
        assert d1 == d2


# ══════════════════════════════════════════════════════
# 2. TestMemoryBatch — Optimisation batch query (~5 tests)
# ══════════════════════════════════════════════════════

class TestMemoryBatch:
    """Tests du batch query dans search_semantic."""

    @pytest.fixture
    def store(self, tmp_path):
        """MemoryStore initialisé avec stockage temporaire."""
        cfg = MemoryConfig(storage_path=tmp_path / "memory")
        s = MemoryStore(cfg)
        s.initialize()
        yield s
        s.close()

    def test_batch_query_returns_results(self, store):
        """search_semantic retourne des résultats avec batch query."""
        # Stocker quelques souvenirs
        store.store("Le Python est un langage de programmation", source="test")
        store.store("La mémoire fonctionne bien", source="test")
        store.store("Les tests sont importants", source="test")

        results = store.search_semantic("programmation Python", n_results=3)
        assert len(results) >= 1
        assert all(isinstance(r, MemoryRecord) for r in results)

    def test_batch_preserves_order(self, store):
        """Le batch query préserve l'ordre de pertinence ChromaDB."""
        store.store("Recette de gâteau au chocolat", source="test")
        store.store("Python est un langage de programmation", source="test")
        store.store("JavaScript est aussi un langage", source="test")

        results = store.search_semantic("langage de programmation", n_results=3)
        # Le premier résultat devrait être le plus pertinent
        assert len(results) >= 1
        assert "langage" in results[0].content.lower() or "python" in results[0].content.lower()

    def test_batch_empty_collection(self, store):
        """search_semantic sur collection vide retourne []."""
        results = store.search_semantic("test query")
        assert results == []

    def test_batch_filters_importance(self, store):
        """search_semantic filtre par importance minimale."""
        store.store("Souvenir important", source="test", importance=0.9)
        store.store("Souvenir banal", source="test", importance=0.1)

        results = store.search_semantic("souvenir", n_results=5, min_importance=0.5)
        for r in results:
            assert r.importance >= 0.5

    def test_batch_single_query(self, store):
        """Le batch query utilise IN (...) au lieu de N requêtes individuelles."""
        # Stocker des souvenirs
        for i in range(10):
            store.store(f"Souvenir numéro {i} pour tester le batch", source="test")

        # Vérifier que le code source utilise IN (...) et non une boucle
        import inspect
        source = inspect.getsource(store.search_semantic)
        assert "IN (" in source, "search_semantic devrait utiliser SELECT ... IN (...)"
        assert "for i, record_id" not in source, "Pas de boucle N+1"


# ══════════════════════════════════════════════════════
# 3. TestDependencies — Imports et extras (~4 tests)
# ══════════════════════════════════════════════════════

class TestDependencies:
    """Tests des dépendances ajoutées."""

    def test_import_httpx(self):
        """httpx est importable."""
        import httpx
        assert hasattr(httpx, "get")
        assert hasattr(httpx, "AsyncClient")

    def test_import_psutil(self):
        """psutil est importable."""
        import psutil
        assert hasattr(psutil, "cpu_percent")
        assert hasattr(psutil, "virtual_memory")

    def test_pyproject_has_httpx(self):
        """pyproject.toml contient httpx dans les deps."""
        toml_path = PROJECT_ROOT / "pyproject.toml"
        content = toml_path.read_text()
        assert "httpx" in content

    def test_pyproject_has_psutil(self):
        """pyproject.toml contient psutil dans les deps."""
        toml_path = PROJECT_ROOT / "pyproject.toml"
        content = toml_path.read_text()
        assert "psutil" in content


# ══════════════════════════════════════════════════════
# 4. TestConftest — Fixtures partagées (~3 tests)
# ══════════════════════════════════════════════════════

class TestConftest:
    """Tests des fixtures conftest."""

    def test_config_fixture(self, config):
        """La fixture config fournit un NeoConfig."""
        assert isinstance(config, NeoConfig)
        assert config.data_dir.exists()

    def test_memory_agent_fixture(self, memory_agent):
        """La fixture memory_agent fournit un agent initialisé."""
        assert memory_agent.is_initialized
        stats = memory_agent.get_stats()
        assert "total_entries" in stats

    def test_brain_fixture(self, brain):
        """La fixture brain fournit un Brain connecté."""
        assert brain is not None
        assert brain.memory is not None


# ══════════════════════════════════════════════════════
# 5. TestInfra — CI/CD et README (~3 tests)
# ══════════════════════════════════════════════════════

class TestInfra:
    """Tests de l'infrastructure projet."""

    def test_readme_exists(self):
        """README.md existe à la racine."""
        readme = PROJECT_ROOT / "README.md"
        assert readme.exists()
        content = readme.read_text()
        assert "Neo Core" in content
        assert "Installation" in content

    def test_ci_workflow_exists(self):
        """Le workflow GitHub Actions existe."""
        workflow = PROJECT_ROOT / ".github" / "workflows" / "test.yml"
        assert workflow.exists()
        content = workflow.read_text()
        assert "pytest" in content

    def test_conftest_exists(self):
        """conftest.py existe dans tests/."""
        conftest = PROJECT_ROOT / "tests" / "conftest.py"
        assert conftest.exists()
        content = conftest.read_text()
        assert "config" in content
        assert "memory_agent" in content

    def test_pyproject_version_updated(self):
        """pyproject.toml a une version récente."""
        toml_path = PROJECT_ROOT / "pyproject.toml"
        content = toml_path.read_text()
        assert 'version = "0.' in content
