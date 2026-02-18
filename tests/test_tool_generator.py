"""
Tests — Level 4 : Tool Generator
===================================
35 tests couvrant : détection, génération, validation,
déploiement, tracking et pruning.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neo_core.memory.learning import ErrorPattern
from neo_core.brain.tools.tool_generator import (
    FORBIDDEN_IMPORTS,
    GeneratedToolMeta,
    ToolGenerator,
    ToolPattern,
)


# ─── Fixtures ───────────────────────────────────────────────────

def _make_error_pattern(
    error_type: str = "tool_not_found",
    worker_type: str = "researcher",
    count: int = 5,
    examples: list[str] | None = None,
) -> ErrorPattern:
    return ErrorPattern(
        worker_type=worker_type,
        error_type=error_type,
        count=count,
        last_seen=datetime.now().isoformat(),
        examples=examples or ["extract data from CSV", "parse CSV file", "read CSV columns"],
    )


def _make_tool_pattern(name: str = "csv_parser", **kwargs) -> ToolPattern:
    return ToolPattern(
        pattern_id=f"pattern_test_{name}",
        name=name,
        description=f"Auto-detected: {name}",
        detected_at=datetime.now().isoformat(),
        occurrence_count=kwargs.get("occurrence_count", 5),
        trigger_keywords=kwargs.get("keywords", ["csv", "parse", "extract"]),
        request_examples=kwargs.get("examples", ["parse csv", "extract csv data"]),
        step_sequence=kwargs.get("steps", ["csv_parser"]),
        proposed_tool_name=name,
        proposed_description=f"Auto-generated: {name}",
        estimated_complexity=kwargs.get("complexity", "simple"),
    )


@pytest.fixture
def mock_learning():
    m = MagicMock()
    m.get_error_patterns.return_value = []
    m.get_performance_summary.return_value = {}
    return m


@pytest.fixture
def mock_plugin_loader():
    m = MagicMock()
    m.load_plugin.return_value = MagicMock(name="loaded_plugin")
    m.reload_all.return_value = {"reloaded": ["csv_parser"]}
    return m


@pytest.fixture
def generator(tmp_path, mock_learning, mock_plugin_loader):
    return ToolGenerator(
        data_dir=tmp_path,
        learning_engine=mock_learning,
        plugin_loader=mock_plugin_loader,
    )


# ════════════════════════════════════════════════════════════════
#  1. DÉTECTION
# ════════════════════════════════════════════════════════════════

class TestDetection:
    """Tests pour detect_opportunities()."""

    def test_detect_tool_not_found(self, generator, mock_learning):
        mock_learning.get_error_patterns.return_value = [
            _make_error_pattern(error_type="tool_not_found", count=5),
        ]
        patterns = generator.detect_opportunities()
        assert len(patterns) == 1
        assert patterns[0].occurrence_count == 5

    def test_detect_ignores_low_count(self, generator, mock_learning):
        mock_learning.get_error_patterns.return_value = [
            _make_error_pattern(count=1),
        ]
        assert generator.detect_opportunities() == []

    def test_detect_ignores_non_tool_errors(self, generator, mock_learning):
        mock_learning.get_error_patterns.return_value = [
            _make_error_pattern(error_type="hallucination", count=10),
        ]
        assert generator.detect_opportunities() == []

    def test_detect_multiple_patterns(self, generator, mock_learning):
        mock_learning.get_error_patterns.return_value = [
            _make_error_pattern(count=5, examples=["parse json", "read json", "load json"]),
            _make_error_pattern(count=3, examples=["convert pdf", "extract pdf", "read pdf"]),
        ]
        patterns = generator.detect_opportunities()
        assert len(patterns) == 2

    def test_detect_sorted_by_occurrence(self, generator, mock_learning):
        mock_learning.get_error_patterns.return_value = [
            _make_error_pattern(count=3, examples=["small pattern", "small"]),
            _make_error_pattern(count=10, examples=["big pattern", "big"]),
        ]
        patterns = generator.detect_opportunities()
        assert patterns[0].occurrence_count >= patterns[-1].occurrence_count

    def test_detect_handles_learning_error(self, generator, mock_learning):
        mock_learning.get_error_patterns.side_effect = RuntimeError("DB error")
        assert generator.detect_opportunities() == []


# ════════════════════════════════════════════════════════════════
#  2. GÉNÉRATION
# ════════════════════════════════════════════════════════════════

class TestGeneration:
    """Tests pour generate_plugin()."""

    def test_generate_valid_plugin(self, generator, tmp_path):
        pattern = _make_tool_pattern("csv_parser")
        tool = generator.generate_plugin(pattern)
        assert tool is not None
        assert tool.name == "csv_parser"
        assert tool.enabled is False

    def test_generate_creates_file(self, generator, tmp_path):
        pattern = _make_tool_pattern("csv_parser")
        tool = generator.generate_plugin(pattern)
        filepath = Path(tool.filepath)
        assert filepath.exists()
        content = filepath.read_text()
        assert "PLUGIN_META" in content
        assert "def execute" in content

    def test_generate_plugin_meta_valid(self, generator, tmp_path):
        pattern = _make_tool_pattern("csv_parser")
        tool = generator.generate_plugin(pattern)
        filepath = Path(tool.filepath)
        code = filepath.read_text()

        # Parse and verify PLUGIN_META
        ns: dict = {}
        exec(compile(code, str(filepath), "exec"), ns)
        meta = ns["PLUGIN_META"]
        assert meta["name"] == "csv_parser"
        assert "input_schema" in meta
        assert "worker_types" in meta
        assert isinstance(meta["worker_types"], list)

    def test_generate_execute_callable(self, generator, tmp_path):
        pattern = _make_tool_pattern("csv_parser")
        tool = generator.generate_plugin(pattern)
        filepath = Path(tool.filepath)
        code = filepath.read_text()

        ns: dict = {}
        exec(compile(code, str(filepath), "exec"), ns)
        assert callable(ns["execute"])

    def test_generate_execute_returns_string(self, generator, tmp_path):
        pattern = _make_tool_pattern("csv_parser")
        tool = generator.generate_plugin(pattern)
        filepath = Path(tool.filepath)
        code = filepath.read_text()

        ns: dict = {}
        exec(compile(code, str(filepath), "exec"), ns)
        result = ns["execute"](input="test data")
        assert isinstance(result, str)

    def test_generate_multi_step_plugin(self, generator, tmp_path):
        pattern = _make_tool_pattern(
            "fetch_and_parse",
            steps=["fetch_data", "parse_html", "extract_text"],
        )
        tool = generator.generate_plugin(pattern)
        filepath = Path(tool.filepath)
        code = filepath.read_text()
        assert "Step 1" in code
        assert "Step 2" in code
        assert "Step 3" in code

    def test_no_duplicate_generation(self, generator, tmp_path):
        pattern = _make_tool_pattern("csv_parser")
        t1 = generator.generate_plugin(pattern)
        assert t1 is not None
        # Simuler le déploiement pour enregistrer dans _tools
        t1.enabled = True
        generator._tools[t1.tool_id] = t1
        t2 = generator.generate_plugin(pattern)
        assert t2 is None  # Déjà existant

    def test_generate_sanitizes_name(self, generator, tmp_path):
        pattern = _make_tool_pattern("My Tool!@#$%")
        tool = generator.generate_plugin(pattern)
        assert tool is not None
        assert " " not in tool.name
        assert "!" not in tool.name


# ════════════════════════════════════════════════════════════════
#  3. VALIDATION
# ════════════════════════════════════════════════════════════════

class TestValidation:
    """Tests pour validate_plugin()."""

    def test_validate_good_plugin(self, generator, tmp_path):
        pattern = _make_tool_pattern("good_tool")
        tool = generator.generate_plugin(pattern)
        result = generator.validate_plugin(tool)
        assert result is True
        assert tool.enabled is True

    def test_validate_syntax_error(self, generator, tmp_path):
        # Créer un fichier avec erreur de syntaxe
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir(parents=True)
        bad_file = plugins_dir / "auto_bad.py"
        bad_file.write_text("def execute(:\n  pass")

        tool = GeneratedToolMeta(
            tool_id="auto_bad",
            name="bad",
            description="bad plugin",
            filepath=str(bad_file),
        )
        assert generator.validate_plugin(tool) is False

    def test_validate_forbidden_import_os(self, generator, tmp_path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir(parents=True)
        bad_file = plugins_dir / "auto_dangerous.py"
        bad_file.write_text("""
import os
PLUGIN_META = {"name": "dangerous", "description": "x", "version": "1.0",
               "input_schema": {}, "worker_types": ["generic"]}
def execute(**kwargs): return os.listdir(".")
""")
        tool = GeneratedToolMeta(
            tool_id="auto_dangerous",
            name="dangerous",
            description="dangerous plugin",
            filepath=str(bad_file),
        )
        result = generator.validate_plugin(tool)
        assert result is False
        assert not bad_file.exists()  # Fichier supprimé

    def test_validate_forbidden_import_subprocess(self, generator, tmp_path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir(parents=True)
        bad_file = plugins_dir / "auto_sub.py"
        bad_file.write_text("""
import subprocess
PLUGIN_META = {"name": "sub", "description": "x", "version": "1.0",
               "input_schema": {}, "worker_types": ["generic"]}
def execute(**kwargs): return subprocess.run(["ls"]).stdout
""")
        tool = GeneratedToolMeta(
            tool_id="auto_sub",
            name="sub",
            description="sub plugin",
            filepath=str(bad_file),
        )
        assert generator.validate_plugin(tool) is False

    def test_validate_nonexistent_file(self, generator):
        tool = GeneratedToolMeta(
            tool_id="auto_ghost",
            name="ghost",
            description="ghost plugin",
            filepath="/nonexistent/path.py",
        )
        assert generator.validate_plugin(tool) is False

    def test_validate_no_execute_function(self, generator, tmp_path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir(parents=True)
        bad_file = plugins_dir / "auto_noexec.py"
        bad_file.write_text("""
PLUGIN_META = {"name": "noexec", "description": "x", "version": "1.0",
               "input_schema": {}, "worker_types": ["generic"]}
# No execute function
""")
        tool = GeneratedToolMeta(
            tool_id="auto_noexec",
            name="noexec",
            description="noexec plugin",
            filepath=str(bad_file),
        )
        # Plugin loader might fail, but exec test should catch it
        assert generator.validate_plugin(tool) is False


# ════════════════════════════════════════════════════════════════
#  4. DÉPLOIEMENT
# ════════════════════════════════════════════════════════════════

class TestDeployment:
    """Tests pour deploy_plugin()."""

    def test_deploy_validated_plugin(self, generator, mock_plugin_loader, tmp_path):
        pattern = _make_tool_pattern("deploy_test")
        tool = generator.generate_plugin(pattern)
        generator.validate_plugin(tool)
        result = generator.deploy_plugin(tool)
        assert result is True
        mock_plugin_loader.reload_all.assert_called()

    def test_deploy_unvalidated_fails(self, generator, tmp_path):
        pattern = _make_tool_pattern("not_validated")
        tool = generator.generate_plugin(pattern)
        # Ne pas valider
        result = generator.deploy_plugin(tool)
        assert result is False

    def test_deploy_saves_metadata(self, generator, tmp_path):
        pattern = _make_tool_pattern("meta_test")
        tool = generator.generate_plugin(pattern)
        generator.validate_plugin(tool)
        generator.deploy_plugin(tool)
        meta_dir = tmp_path / "tool_metadata"
        assert meta_dir.exists()
        files = list(meta_dir.glob("*.json"))
        assert len(files) >= 1

    def test_deploy_without_plugin_loader(self, tmp_path, mock_learning):
        gen = ToolGenerator(data_dir=tmp_path, learning_engine=mock_learning, plugin_loader=None)
        pattern = _make_tool_pattern("no_loader")
        tool = gen.generate_plugin(pattern)
        gen.validate_plugin(tool)
        result = gen.deploy_plugin(tool)
        assert result is True

    def test_deploy_tool_accessible(self, generator, tmp_path):
        pattern = _make_tool_pattern("accessible_tool")
        tool = generator.generate_plugin(pattern)
        generator.validate_plugin(tool)
        generator.deploy_plugin(tool)
        assert len(generator.active_tools) == 1
        assert generator.active_tools[0].name == "accessible_tool"


# ════════════════════════════════════════════════════════════════
#  5. TRACKING
# ════════════════════════════════════════════════════════════════

class TestTracking:
    """Tests pour track_usage()."""

    def _deploy_tool(self, generator, name="track_test"):
        pattern = _make_tool_pattern(name)
        tool = generator.generate_plugin(pattern)
        generator.validate_plugin(tool)
        generator.deploy_plugin(tool)
        return tool

    def test_track_increments_usage(self, generator):
        tool = self._deploy_tool(generator)
        generator.track_usage("track_test", success=True, execution_time=1.0)
        assert tool.usage_count == 1

    def test_track_increments_success(self, generator):
        tool = self._deploy_tool(generator)
        generator.track_usage("track_test", success=True, execution_time=1.0)
        generator.track_usage("track_test", success=False, execution_time=2.0)
        assert tool.usage_count == 2
        assert tool.success_count == 1

    def test_track_updates_avg_time(self, generator):
        tool = self._deploy_tool(generator)
        generator.track_usage("track_test", success=True, execution_time=2.0)
        generator.track_usage("track_test", success=True, execution_time=4.0)
        assert tool.avg_execution_time == pytest.approx(3.0)

    def test_track_updates_last_used(self, generator):
        tool = self._deploy_tool(generator)
        assert tool.last_used is None
        generator.track_usage("track_test", success=True, execution_time=1.0)
        assert tool.last_used is not None

    def test_track_unknown_tool_noop(self, generator):
        # Should not crash
        generator.track_usage("nonexistent", success=True, execution_time=1.0)


# ════════════════════════════════════════════════════════════════
#  6. PRUNING
# ════════════════════════════════════════════════════════════════

class TestPruning:
    """Tests pour prune_unused()."""

    def _deploy_tool(self, generator, name="prune_test"):
        pattern = _make_tool_pattern(name)
        tool = generator.generate_plugin(pattern)
        generator.validate_plugin(tool)
        generator.deploy_plugin(tool)
        return tool

    def test_prune_unused_after_7_days(self, generator):
        tool = self._deploy_tool(generator)
        # Simuler dernière utilisation il y a 10 jours
        tool.last_used = (datetime.now() - timedelta(days=10)).isoformat()
        pruned = generator.prune_unused()
        assert "prune_test" in pruned
        assert tool.deprecated is True

    def test_prune_never_used_after_7_days(self, generator):
        tool = self._deploy_tool(generator)
        # Simuler création il y a 10 jours, jamais utilisé
        tool.created_at = (datetime.now() - timedelta(days=10)).isoformat()
        tool.last_used = None
        pruned = generator.prune_unused()
        assert "prune_test" in pruned

    def test_prune_low_success_rate(self, generator):
        tool = self._deploy_tool(generator)
        tool.usage_count = 10
        tool.success_count = 1  # 10% success
        tool.last_used = datetime.now().isoformat()  # Used recently
        pruned = generator.prune_unused()
        assert "prune_test" in pruned
        assert "low_success_rate" in tool.deprecation_reason

    def test_prune_keeps_healthy_tools(self, generator):
        tool = self._deploy_tool(generator)
        tool.usage_count = 10
        tool.success_count = 8  # 80% success
        tool.last_used = datetime.now().isoformat()
        pruned = generator.prune_unused()
        assert pruned == []
        assert tool.deprecated is False

    def test_prune_removes_file(self, generator, tmp_path):
        tool = self._deploy_tool(generator)
        filepath = Path(tool.filepath)
        assert filepath.exists()
        tool.last_used = (datetime.now() - timedelta(days=10)).isoformat()
        generator.prune_unused()
        assert not filepath.exists()


# ════════════════════════════════════════════════════════════════
#  7. PERSISTENCE & STATS
# ════════════════════════════════════════════════════════════════

class TestPersistence:
    """Tests pour save/load et stats."""

    def test_save_and_reload(self, tmp_path, mock_learning, mock_plugin_loader):
        gen1 = ToolGenerator(data_dir=tmp_path, learning_engine=mock_learning,
                             plugin_loader=mock_plugin_loader)
        pattern = _make_tool_pattern("persist_test")
        tool = gen1.generate_plugin(pattern)
        gen1.validate_plugin(tool)
        gen1.deploy_plugin(tool)

        # Recharger
        gen2 = ToolGenerator(data_dir=tmp_path, learning_engine=mock_learning,
                             plugin_loader=mock_plugin_loader)
        assert len(gen2.all_tools) == 1
        loaded = gen2.all_tools[0]
        assert loaded.tool_id == tool.tool_id
        assert loaded.name == tool.name

    def test_get_stats(self, generator):
        pattern = _make_tool_pattern("stats_test")
        tool = generator.generate_plugin(pattern)
        generator.validate_plugin(tool)
        generator.deploy_plugin(tool)

        stats = generator.get_stats()
        assert stats["total_tools"] == 1
        assert stats["active_tools"] == 1

    def test_thread_safe_tracking(self, generator):
        pattern = _make_tool_pattern("thread_test")
        tool = generator.generate_plugin(pattern)
        generator.validate_plugin(tool)
        generator.deploy_plugin(tool)

        errors = []

        def track():
            try:
                for _ in range(10):
                    generator.track_usage("thread_test", True, 1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=track) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tool.usage_count == 50
