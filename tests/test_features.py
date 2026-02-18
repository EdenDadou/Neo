"""
Tests — Feature Flags System
==============================
Tests pour le système de feature flags (neo_core/features.py).
"""

import json
from pathlib import Path

import pytest

from neo_core.features import (
    feature_enabled,
    set_feature,
    get_all_features,
    reset_features,
    save_features,
    set_config_path,
    _FEATURE_DEFAULTS,
)


@pytest.fixture(autouse=True)
def clean_features():
    """Reset les features avant et après chaque test."""
    reset_features()
    yield
    reset_features()


class TestFeatureDefaults:
    def test_all_defaults_enabled(self):
        for name, default in _FEATURE_DEFAULTS.items():
            assert feature_enabled(name) == default

    def test_unknown_feature_returns_true(self):
        """Feature inconnue → fail-open (True)."""
        assert feature_enabled("nonexistent_feature") is True


class TestFeatureOverrides:
    def test_set_feature_disables(self):
        assert feature_enabled("tool_generation") is True
        set_feature("tool_generation", False)
        assert feature_enabled("tool_generation") is False

    def test_set_feature_enables(self):
        set_feature("tool_generation", False)
        assert feature_enabled("tool_generation") is False
        set_feature("tool_generation", True)
        assert feature_enabled("tool_generation") is True

    def test_runtime_override_takes_priority(self):
        set_feature("working_memory", False)
        assert feature_enabled("working_memory") is False


class TestFeatureFile:
    def test_load_from_file(self, tmp_path):
        flags_file = tmp_path / "feature_flags.json"
        flags_file.write_text(json.dumps({
            "tool_generation": False,
            "self_patching": False,
        }))
        set_config_path(flags_file)
        assert feature_enabled("tool_generation") is False
        assert feature_enabled("self_patching") is False
        assert feature_enabled("working_memory") is True  # Non overridden

    def test_invalid_json_uses_defaults(self, tmp_path):
        flags_file = tmp_path / "feature_flags.json"
        flags_file.write_text("not json!")
        set_config_path(flags_file)
        assert feature_enabled("tool_generation") is True

    def test_missing_file_uses_defaults(self, tmp_path):
        set_config_path(tmp_path / "nonexistent.json")
        assert feature_enabled("tool_generation") is True

    def test_save_features(self, tmp_path):
        set_feature("tool_generation", False)
        save_features(tmp_path)
        data = json.loads((tmp_path / "feature_flags.json").read_text())
        assert data["tool_generation"] is False


class TestGetAllFeatures:
    def test_returns_all(self):
        features = get_all_features()
        assert len(features) == len(_FEATURE_DEFAULTS)
        for name in _FEATURE_DEFAULTS:
            assert name in features

    def test_reflects_overrides(self):
        set_feature("auto_tuning", False)
        features = get_all_features()
        assert features["auto_tuning"] is False
        assert features["working_memory"] is True


class TestResetFeatures:
    def test_reset_clears_overrides(self):
        set_feature("tool_generation", False)
        assert feature_enabled("tool_generation") is False
        reset_features()
        assert feature_enabled("tool_generation") is True
