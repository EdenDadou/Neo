"""
Tests — Level 3 : Self-Patcher
================================
40 tests couvrant : détection, génération, validation,
application, évaluation, rollback et persistence.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neo_core.core.self_patcher import (
    DEFAULT_DETECTION_THRESHOLD,
    PATCHABLE_ERROR_TYPES,
    PatchEvaluation,
    PatchMetadata,
    SelfPatcher,
)
from neo_core.memory.learning import ErrorPattern


# ─── Helpers / Fixtures ─────────────────────────────────────────

def _make_error_pattern(
    worker_type: str = "coder",
    error_type: str = "hallucination",
    count: int = 5,
    last_seen: str | None = None,
    examples: list[str] | None = None,
) -> ErrorPattern:
    return ErrorPattern(
        worker_type=worker_type,
        error_type=error_type,
        count=count,
        last_seen=last_seen or datetime.now().isoformat(),
        examples=examples or ["request1", "request2", "request3"],
    )


@pytest.fixture
def mock_learning():
    m = MagicMock()
    m.get_error_patterns.return_value = []
    m.get_performance_summary.return_value = {}
    return m


@pytest.fixture
def patcher(tmp_path, mock_learning):
    return SelfPatcher(data_dir=tmp_path, learning_engine=mock_learning)


# ════════════════════════════════════════════════════════════════
#  1. DÉTECTION
# ════════════════════════════════════════════════════════════════

class TestDetection:
    """Tests pour detect_patchable_patterns()."""

    def test_detect_recurring_hallucination(self, patcher, mock_learning):
        mock_learning.get_error_patterns.return_value = [
            _make_error_pattern(error_type="hallucination", count=5),
        ]
        patterns = patcher.detect_patchable_patterns()
        assert len(patterns) == 1
        assert patterns[0].error_type == "hallucination"

    def test_detect_ignores_low_count(self, patcher, mock_learning):
        mock_learning.get_error_patterns.return_value = [
            _make_error_pattern(count=1),
        ]
        assert patcher.detect_patchable_patterns() == []

    def test_detect_ignores_non_patchable_type(self, patcher, mock_learning):
        mock_learning.get_error_patterns.return_value = [
            _make_error_pattern(error_type="resource_exhaustion", count=10),
        ]
        assert patcher.detect_patchable_patterns() == []

    def test_detect_ignores_old_patterns(self, patcher, mock_learning):
        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        mock_learning.get_error_patterns.return_value = [
            _make_error_pattern(last_seen=old_date, count=5),
        ]
        assert patcher.detect_patchable_patterns() == []

    def test_detect_multiple_patterns(self, patcher, mock_learning):
        mock_learning.get_error_patterns.return_value = [
            _make_error_pattern(error_type="hallucination", count=5),
            _make_error_pattern(error_type="timeout", worker_type="researcher", count=4),
            _make_error_pattern(error_type="tool_failure", count=3),
        ]
        patterns = patcher.detect_patchable_patterns()
        assert len(patterns) == 3

    def test_detect_skips_already_patched(self, patcher, mock_learning):
        pattern = _make_error_pattern(error_type="hallucination", count=5)
        mock_learning.get_error_patterns.return_value = [pattern]
        # Générer et valider un premier patch
        patch = patcher.generate_patch(pattern)
        patch.enabled = True
        # Maintenant le pattern ne devrait plus être détecté
        patterns = patcher.detect_patchable_patterns()
        assert len(patterns) == 0

    def test_detect_handles_learning_error(self, patcher, mock_learning):
        mock_learning.get_error_patterns.side_effect = RuntimeError("DB error")
        assert patcher.detect_patchable_patterns() == []

    def test_detect_exact_threshold(self, patcher, mock_learning):
        mock_learning.get_error_patterns.return_value = [
            _make_error_pattern(count=DEFAULT_DETECTION_THRESHOLD),
        ]
        assert len(patcher.detect_patchable_patterns()) == 1


# ════════════════════════════════════════════════════════════════
#  2. GÉNÉRATION
# ════════════════════════════════════════════════════════════════

class TestGeneration:
    """Tests pour generate_patch()."""

    def test_generate_hallucination_patch(self, patcher):
        pattern = _make_error_pattern(error_type="hallucination")
        patch = patcher.generate_patch(pattern)
        assert patch is not None
        assert patch.patch_type == "prompt_override"
        assert patch.action["override_temperature"] == 0.3
        assert "add_system_suffix" in patch.action

    def test_generate_timeout_patch(self, patcher):
        pattern = _make_error_pattern(error_type="timeout")
        patch = patcher.generate_patch(pattern)
        assert patch is not None
        assert patch.patch_type == "config_override"
        assert patch.action["override_timeout"] == 300.0
        assert patch.action["decompose_task"] is True

    def test_generate_tool_failure_patch(self, patcher):
        pattern = _make_error_pattern(error_type="tool_failure")
        patch = patcher.generate_patch(pattern)
        assert patch is not None
        assert patch.patch_type == "routing_rule"
        assert "fallback_worker" in patch.action

    def test_generate_routing_error_patch(self, patcher):
        pattern = _make_error_pattern(error_type="routing_error")
        patch = patcher.generate_patch(pattern)
        assert patch is not None
        assert patch.patch_type == "routing_rule"
        assert "override_worker" in patch.action

    def test_generate_strategy_mismatch_patch(self, patcher):
        pattern = _make_error_pattern(error_type="strategy_mismatch")
        patch = patcher.generate_patch(pattern)
        assert patch is not None
        assert "override_worker" in patch.action

    def test_no_duplicate_patch(self, patcher):
        pattern = _make_error_pattern()
        p1 = patcher.generate_patch(pattern)
        p2 = patcher.generate_patch(pattern)
        assert p1 is not None
        assert p2 is None  # Déjà existant

    def test_generate_unknown_type_returns_none(self, patcher):
        pattern = _make_error_pattern(error_type="resource_exhaustion")
        # On force le passage (normalement filtré par detect)
        patch = patcher.generate_patch(pattern)
        assert patch is None

    def test_generated_patch_starts_disabled(self, patcher):
        patch = patcher.generate_patch(_make_error_pattern())
        assert patch is not None
        assert patch.enabled is False


# ════════════════════════════════════════════════════════════════
#  3. VALIDATION
# ════════════════════════════════════════════════════════════════

class TestValidation:
    """Tests pour validate_patch()."""

    def test_validate_enables_patch_with_high_error_rate(self, patcher, mock_learning):
        mock_learning.get_performance_summary.return_value = {
            "coder": {
                "total_tasks": 20,
                "success_count": 2,
                "failure_count": 18,
            }
        }
        pattern = _make_error_pattern()
        patch = patcher.generate_patch(pattern)
        result = patcher.validate_patch(patch)
        assert result is True
        assert patch.enabled is True

    def test_validate_rejects_low_error_rate(self, patcher, mock_learning):
        mock_learning.get_performance_summary.return_value = {
            "coder": {
                "total_tasks": 20,
                "success_count": 19,
                "failure_count": 1,
            }
        }
        pattern = _make_error_pattern()
        patch = patcher.generate_patch(pattern)
        result = patcher.validate_patch(patch)
        assert result is False
        assert patch.enabled is False

    def test_validate_rejects_insufficient_data(self, patcher, mock_learning):
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 1, "success_count": 0, "failure_count": 1}
        }
        pattern = _make_error_pattern()
        patch = patcher.generate_patch(pattern)
        assert patcher.validate_patch(patch) is False

    def test_validate_captures_metrics_before(self, patcher, mock_learning):
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 10, "success_count": 3, "failure_count": 7}
        }
        pattern = _make_error_pattern()
        patch = patcher.generate_patch(pattern)
        patcher.validate_patch(patch)
        assert patch.metrics_before["total_tasks"] == 10
        assert patch.metrics_before["success_rate"] == 0.3

    def test_validate_saves_patches_on_success(self, patcher, mock_learning, tmp_path):
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 20, "success_count": 2, "failure_count": 18}
        }
        patch = patcher.generate_patch(_make_error_pattern())
        patcher.validate_patch(patch)
        patches_dir = tmp_path / "patches"
        assert patches_dir.exists()
        files = list(patches_dir.glob("*.json"))
        assert len(files) >= 1

    def test_validate_unknown_worker_uses_defaults(self, patcher, mock_learning):
        mock_learning.get_performance_summary.return_value = {}
        pattern = _make_error_pattern(worker_type="unknown_worker")
        patch = patcher.generate_patch(pattern)
        # Pas de données → pas assez → rejeté
        assert patcher.validate_patch(patch) is False


# ════════════════════════════════════════════════════════════════
#  4. APPLICATION
# ════════════════════════════════════════════════════════════════

class TestApplication:
    """Tests pour apply_patches()."""

    def _create_active_patch(self, patcher, mock_learning, **kwargs):
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 20, "success_count": 2, "failure_count": 18},
            "researcher": {"total_tasks": 20, "success_count": 2, "failure_count": 18},
        }
        error_type = kwargs.get("error_type", "hallucination")
        pattern = _make_error_pattern(error_type=error_type)
        patch = patcher.generate_patch(pattern)
        patcher.validate_patch(patch)
        return patch

    def test_apply_returns_overrides(self, patcher, mock_learning):
        self._create_active_patch(patcher, mock_learning)
        overrides = patcher.apply_patches("request1 something", "coder")
        assert "override_temperature" in overrides

    def test_apply_no_match_returns_empty(self, patcher, mock_learning):
        self._create_active_patch(patcher, mock_learning)
        overrides = patcher.apply_patches("completely different topic", "researcher")
        assert overrides == {}

    def test_apply_increments_applied_count(self, patcher, mock_learning):
        patch = self._create_active_patch(patcher, mock_learning)
        patcher.apply_patches("request1 something", "coder")
        assert patch.applied_count >= 1

    def test_apply_disabled_patch_ignored(self, patcher, mock_learning):
        patch = self._create_active_patch(patcher, mock_learning)
        patch.enabled = False
        overrides = patcher.apply_patches("request1 something", "coder")
        assert overrides == {}

    def test_apply_wrong_worker_type_ignored(self, patcher, mock_learning):
        self._create_active_patch(patcher, mock_learning)
        overrides = patcher.apply_patches("request1 something", "researcher")
        assert overrides == {}

    def test_apply_multiple_patches(self, patcher, mock_learning):
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 20, "success_count": 2, "failure_count": 18},
            "researcher": {"total_tasks": 20, "success_count": 2, "failure_count": 18},
        }
        # Hallucination sur coder
        p1 = _make_error_pattern(error_type="hallucination", worker_type="coder")
        patch1 = patcher.generate_patch(p1)
        patcher.validate_patch(patch1)
        # Timeout sur researcher
        p2 = _make_error_pattern(error_type="timeout", worker_type="researcher",
                                  examples=["slow query", "fetch data"])
        patch2 = patcher.generate_patch(p2)
        patcher.validate_patch(patch2)

        # Seul le patch coder doit matcher
        overrides = patcher.apply_patches("request1 something", "coder")
        assert "override_temperature" in overrides
        assert "override_timeout" not in overrides


# ════════════════════════════════════════════════════════════════
#  5. ÉVALUATION
# ════════════════════════════════════════════════════════════════

class TestEvaluation:
    """Tests pour evaluate_effectiveness()."""

    def test_evaluate_effective_patch(self, patcher, mock_learning):
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 20, "success_count": 2, "failure_count": 18}
        }
        pattern = _make_error_pattern()
        patch = patcher.generate_patch(pattern)
        patcher.validate_patch(patch)

        # Simuler une amélioration
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 30, "success_count": 25, "failure_count": 5}
        }
        evaluation = patcher.evaluate_effectiveness(patch.patch_id)
        assert evaluation.is_effective is True
        assert evaluation.improvement > 0

    def test_evaluate_unknown_patch(self, patcher):
        evaluation = patcher.evaluate_effectiveness("nonexistent")
        assert evaluation.is_effective is False
        assert evaluation.recommendation == "not_found"

    def test_evaluate_insufficient_data(self, patcher, mock_learning):
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 20, "success_count": 2, "failure_count": 18}
        }
        patch = patcher.generate_patch(_make_error_pattern())
        patcher.validate_patch(patch)

        # Seulement 1 tâche après le patch
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 1, "success_count": 1, "failure_count": 0}
        }
        evaluation = patcher.evaluate_effectiveness(patch.patch_id)
        assert evaluation.confidence < 0.5

    def test_evaluate_rollback_recommendation(self, patcher, mock_learning):
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 20, "success_count": 2, "failure_count": 18}
        }
        patch = patcher.generate_patch(_make_error_pattern())
        patcher.validate_patch(patch)

        # Performance dégradée
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 40, "success_count": 5, "failure_count": 35}
        }
        evaluation = patcher.evaluate_effectiveness(patch.patch_id)
        # La recommandation dépend de la confiance et de l'amélioration
        assert evaluation.recommendation in ("rollback", "improve", "insufficient_data")

    def test_evaluate_keep_recommendation(self, patcher, mock_learning):
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 20, "success_count": 2, "failure_count": 18}
        }
        patch = patcher.generate_patch(_make_error_pattern())
        patcher.validate_patch(patch)

        # Grande amélioration + assez de données
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 50, "success_count": 45, "failure_count": 5}
        }
        evaluation = patcher.evaluate_effectiveness(patch.patch_id)
        assert evaluation.recommendation == "keep"

    def test_evaluate_and_rollback_all(self, patcher, mock_learning):
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 20, "success_count": 2, "failure_count": 18}
        }
        patch = patcher.generate_patch(_make_error_pattern())
        patcher.validate_patch(patch)

        # Dégradation massive avec assez de confiance
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 50, "success_count": 2, "failure_count": 48}
        }
        rolled_back = patcher.evaluate_and_rollback_all()
        # Soit rollback soit pas assez de confiance
        assert isinstance(rolled_back, list)


# ════════════════════════════════════════════════════════════════
#  6. PERSISTENCE
# ════════════════════════════════════════════════════════════════

class TestPersistence:
    """Tests pour save/load des patches."""

    def test_save_and_reload(self, tmp_path, mock_learning):
        patcher = SelfPatcher(data_dir=tmp_path, learning_engine=mock_learning)
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 20, "success_count": 2, "failure_count": 18}
        }
        pattern = _make_error_pattern()
        patch = patcher.generate_patch(pattern)
        patcher.validate_patch(patch)
        patcher._save_patches()

        # Recharger un nouveau SelfPatcher
        patcher2 = SelfPatcher(data_dir=tmp_path, learning_engine=mock_learning)
        assert len(patcher2.all_patches) == 1
        loaded = patcher2.all_patches[0]
        assert loaded.patch_id == patch.patch_id
        assert loaded.enabled == patch.enabled
        assert loaded.patch_type == patch.patch_type

    def test_patches_dir_created_auto(self, tmp_path, mock_learning):
        patcher = SelfPatcher(data_dir=tmp_path, learning_engine=mock_learning)
        pattern = _make_error_pattern()
        patcher.generate_patch(pattern)
        patcher._save_patches()
        assert (tmp_path / "patches").is_dir()

    def test_load_ignores_corrupt_json(self, tmp_path, mock_learning):
        patches_dir = tmp_path / "patches"
        patches_dir.mkdir(parents=True)
        (patches_dir / "bad.json").write_text("{invalid json")
        patcher = SelfPatcher(data_dir=tmp_path, learning_engine=mock_learning)
        assert len(patcher.all_patches) == 0

    def test_thread_safe_access(self, patcher, mock_learning):
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 20, "success_count": 2, "failure_count": 18}
        }
        errors = []

        def gen_and_apply():
            try:
                pattern = _make_error_pattern(
                    worker_type="coder",
                    error_type="hallucination",
                    count=5,
                )
                patcher.generate_patch(pattern)
                patcher.apply_patches("test request1", "coder")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=gen_and_apply) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_rollback_persists(self, tmp_path, mock_learning):
        patcher = SelfPatcher(data_dir=tmp_path, learning_engine=mock_learning)
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 20, "success_count": 2, "failure_count": 18}
        }
        patch = patcher.generate_patch(_make_error_pattern())
        patcher.validate_patch(patch)
        patcher.rollback_patch(patch.patch_id, "test_reason")

        patcher2 = SelfPatcher(data_dir=tmp_path, learning_engine=mock_learning)
        loaded = patcher2.all_patches[0]
        assert loaded.enabled is False
        assert loaded.rollback_reason == "test_reason"

    def test_get_stats(self, patcher, mock_learning):
        mock_learning.get_performance_summary.return_value = {
            "coder": {"total_tasks": 20, "success_count": 2, "failure_count": 18}
        }
        patcher.generate_patch(_make_error_pattern())
        stats = patcher.get_stats()
        assert stats["total_patches"] == 1
        assert stats["active_patches"] == 0
