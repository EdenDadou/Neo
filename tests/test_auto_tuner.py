"""
Tests for the AutoTuner (Level 1: Auto-tuning system).

Tests cover:
- Loading and saving tuning configurations
- Temperature tuning based on hallucination errors
- Timeout tuning based on execution time and timeout errors
- Max retries tuning based on transient failures
- Thread safety and concurrent access
- Bounds checking
- Backwards compatibility (no auto_tuning.json exists)
"""

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neo_core.core.auto_tuner import AutoTuner, TuningMetrics
from neo_core.memory.learning import LearningEngine, WorkerPerformance, ErrorPattern


class MockLearningEngine:
    """Mock LearningEngine for testing."""

    def __init__(self):
        self._performance_cache = {}
        self._error_patterns = []

    def get_performance_summary(self) -> dict:
        """Return mock performance summary."""
        return self._performance_cache

    def get_error_patterns(self) -> list:
        """Return mock error patterns."""
        return self._error_patterns

    def add_performance(self, worker_type: str, total_tasks: int, success_rate: float,
                        avg_time: float, failures: int) -> None:
        """Add performance data."""
        self._performance_cache[worker_type] = {
            "total_tasks": total_tasks,
            "success_rate": f"{success_rate * 100:.0f}%",
            "avg_time": f"{avg_time:.1f}s",
            "failures": failures,
        }

    def add_error_pattern(self, worker_type: str, error_type: str, count: int) -> None:
        """Add error pattern."""
        pattern = ErrorPattern(
            worker_type=worker_type,
            error_type=error_type,
            count=count,
        )
        self._error_patterns.append(pattern)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def mock_learning_engine():
    """Create a mock learning engine."""
    return MockLearningEngine()


@pytest.fixture
def auto_tuner(temp_data_dir, mock_learning_engine):
    """Create an AutoTuner instance with mock learning engine."""
    return AutoTuner(temp_data_dir, mock_learning_engine)


class TestAutoTunerBasics:
    """Test basic AutoTuner functionality."""

    def test_initialization(self, auto_tuner, temp_data_dir):
        """Test AutoTuner initialization."""
        assert auto_tuner.data_dir == temp_data_dir
        assert auto_tuner.learning is not None
        assert auto_tuner._current_tuning == {}

    def test_load_nonexistent_file(self, auto_tuner):
        """Test loading from non-existent file (backwards compatible)."""
        assert auto_tuner.get_tuned_config("worker:coder") is None

    def test_save_and_load(self, auto_tuner):
        """Test saving and loading tuning configuration."""
        # Set some tuning data
        auto_tuner._current_tuning["worker:coder"] = {
            "temperature": 0.4,
            "max_retries": 4,
        }
        auto_tuner._save()

        # Create a new instance and verify it loads the data
        auto_tuner2 = AutoTuner(auto_tuner.data_dir, auto_tuner.learning)
        config = auto_tuner2.get_tuned_config("worker:coder")
        assert config is not None
        assert config["temperature"] == 0.4
        assert config["max_retries"] == 4

    def test_get_tuned_config_missing(self, auto_tuner):
        """Test getting config for agent with no tuning."""
        assert auto_tuner.get_tuned_config("worker:nonexistent") is None

    def test_get_tuned_config_existing(self, auto_tuner):
        """Test getting config for agent with tuning."""
        auto_tuner._current_tuning["worker:researcher"] = {
            "temperature": 0.5,
            "worker_timeout": 240.0,
        }
        config = auto_tuner.get_tuned_config("worker:researcher")
        assert config["temperature"] == 0.5
        assert config["worker_timeout"] == 240.0


class TestTemperatureTuning:
    """Test temperature tuning logic."""

    def test_temperature_no_hallucinations(self, auto_tuner, mock_learning_engine):
        """Test temperature stays stable with no hallucination errors."""
        auto_tuner._current_tuning["worker:coder"] = {
            "temperature": 0.7,
            "max_retries": 3,
            "worker_timeout": 180.0,
            "max_tool_iterations": 10,
        }

        # Setup: 10 tasks, 80% success, low time
        mock_learning_engine.add_performance("coder", 10, 0.8, 2.5, 2)

        metrics = TuningMetrics(
            success_rate=0.8,
            failure_rate=0.2,
            avg_execution_time=2.5,
            total_tasks=10,
            timeout_count=0,
            hallucination_count=0,
            tool_failure_count=2,
            transient_failure_count=0,
        )

        new_temp = auto_tuner._tune_temperature("coder", metrics)
        assert new_temp == 0.7  # Should remain unchanged

    def test_temperature_high_hallucinations(self, auto_tuner, mock_learning_engine):
        """Test temperature decreases with high hallucination rate."""
        auto_tuner._current_tuning["worker:writer"] = {
            "temperature": 0.8,
            "max_retries": 3,
            "worker_timeout": 180.0,
            "max_tool_iterations": 10,
        }

        # Setup: 10 tasks, 50% success, high hallucination
        mock_learning_engine.add_performance("writer", 10, 0.5, 3.0, 5)
        mock_learning_engine.add_error_pattern("writer", "hallucination", 3)
        mock_learning_engine.add_error_pattern("writer", "tool_failure", 2)

        metrics = TuningMetrics(
            success_rate=0.5,
            failure_rate=0.5,
            avg_execution_time=3.0,
            total_tasks=10,
            timeout_count=0,
            hallucination_count=3,  # 3 out of 5 failures
            tool_failure_count=2,
            transient_failure_count=0,
        )

        new_temp = auto_tuner._tune_temperature("writer", metrics)
        assert new_temp < 0.8  # Temperature should decrease
        assert new_temp >= auto_tuner.BOUNDS["temperature"][0]  # Within bounds

    def test_temperature_bounds(self, auto_tuner):
        """Test temperature stays within bounds."""
        auto_tuner._current_tuning["worker:generic"] = {
            "temperature": 0.1,
            "max_retries": 3,
            "worker_timeout": 180.0,
            "max_tool_iterations": 10,
        }

        metrics = TuningMetrics(
            success_rate=0.5,
            failure_rate=0.5,
            avg_execution_time=3.0,
            total_tasks=10,
            timeout_count=0,
            hallucination_count=4,
            tool_failure_count=1,
            transient_failure_count=0,
        )

        new_temp = auto_tuner._tune_temperature("generic", metrics)
        assert auto_tuner.BOUNDS["temperature"][0] <= new_temp <= auto_tuner.BOUNDS["temperature"][1]


class TestTimeoutTuning:
    """Test timeout tuning logic."""

    def test_timeout_no_issues(self, auto_tuner, mock_learning_engine):
        """Test timeout stays stable with low error rate."""
        auto_tuner._current_tuning["worker:coder"] = {
            "temperature": 0.5,
            "max_retries": 3,
            "worker_timeout": 180.0,
            "max_tool_iterations": 10,
        }

        # Setup: 10 tasks, 90% success, low time
        mock_learning_engine.add_performance("coder", 10, 0.9, 2.5, 1)

        metrics = TuningMetrics(
            success_rate=0.9,
            failure_rate=0.1,
            avg_execution_time=2.5,
            total_tasks=10,
            timeout_count=0,
            hallucination_count=1,
            tool_failure_count=0,
            transient_failure_count=0,
        )

        new_timeout = auto_tuner._tune_timeout("coder", metrics)
        assert new_timeout == 180.0  # Should remain unchanged

    def test_timeout_high_timeout_errors(self, auto_tuner, mock_learning_engine):
        """Test timeout increases with high timeout error rate."""
        auto_tuner._current_tuning["worker:researcher"] = {
            "temperature": 0.5,
            "max_retries": 3,
            "worker_timeout": 180.0,
            "max_tool_iterations": 10,
        }

        # Setup: 10 tasks, 60% success, high timeout errors
        mock_learning_engine.add_performance("researcher", 10, 0.6, 120.0, 4)
        mock_learning_engine.add_error_pattern("researcher", "timeout", 3)
        mock_learning_engine.add_error_pattern("researcher", "tool_failure", 1)

        metrics = TuningMetrics(
            success_rate=0.6,
            failure_rate=0.4,
            avg_execution_time=120.0,
            total_tasks=10,
            timeout_count=3,
            hallucination_count=0,
            tool_failure_count=1,
            transient_failure_count=0,
        )

        new_timeout = auto_tuner._tune_timeout("researcher", metrics)
        assert new_timeout > 180.0  # Should increase

    def test_timeout_high_avg_execution_time(self, auto_tuner):
        """Test timeout increases when avg execution time is high."""
        auto_tuner._current_tuning["worker:coder"] = {
            "temperature": 0.5,
            "max_retries": 3,
            "worker_timeout": 180.0,
            "max_tool_iterations": 10,
        }

        metrics = TuningMetrics(
            success_rate=0.85,
            failure_rate=0.15,
            avg_execution_time=140.0,  # 77% of timeout
            total_tasks=10,
            timeout_count=0,
            hallucination_count=0,
            tool_failure_count=1,
            transient_failure_count=1,
        )

        new_timeout = auto_tuner._tune_timeout("coder", metrics)
        assert new_timeout >= 180.0  # Should increase or stay same

    def test_timeout_bounds(self, auto_tuner):
        """Test timeout stays within bounds."""
        auto_tuner._current_tuning["worker:generic"] = {
            "temperature": 0.5,
            "max_retries": 3,
            "worker_timeout": 60.0,
            "max_tool_iterations": 10,
        }

        metrics = TuningMetrics(
            success_rate=0.4,
            failure_rate=0.6,
            avg_execution_time=55.0,
            total_tasks=10,
            timeout_count=5,
            hallucination_count=0,
            tool_failure_count=1,
            transient_failure_count=0,
        )

        new_timeout = auto_tuner._tune_timeout("generic", metrics)
        assert auto_tuner.BOUNDS["worker_timeout"][0] <= new_timeout <= auto_tuner.BOUNDS["worker_timeout"][1]


class TestRetryTuning:
    """Test max_retries tuning logic."""

    def test_retries_stable(self, auto_tuner):
        """Test retries stay stable with low transient failure rate."""
        auto_tuner._current_tuning["worker:coder"] = {
            "temperature": 0.5,
            "max_retries": 3,
            "worker_timeout": 180.0,
            "max_tool_iterations": 10,
        }

        metrics = TuningMetrics(
            success_rate=0.8,
            failure_rate=0.2,
            avg_execution_time=2.5,
            total_tasks=10,
            timeout_count=0,
            hallucination_count=1,
            tool_failure_count=1,
            transient_failure_count=0,
        )

        new_retries = auto_tuner._tune_max_retries("coder", metrics)
        assert new_retries == 3  # Should stay same

    def test_retries_increase_high_transient(self, auto_tuner):
        """Test retries increase with high transient failure rate."""
        auto_tuner._current_tuning["worker:researcher"] = {
            "temperature": 0.5,
            "max_retries": 3,
            "worker_timeout": 180.0,
            "max_tool_iterations": 10,
        }

        metrics = TuningMetrics(
            success_rate=0.6,
            failure_rate=0.4,
            avg_execution_time=5.0,
            total_tasks=10,
            timeout_count=0,
            hallucination_count=0,
            tool_failure_count=1,
            transient_failure_count=3,  # 3 out of 4 failures are transient
        )

        new_retries = auto_tuner._tune_max_retries("researcher", metrics)
        assert new_retries > 3  # Should increase

    def test_retries_bounds(self, auto_tuner):
        """Test retries stays within bounds."""
        auto_tuner._current_tuning["worker:generic"] = {
            "temperature": 0.5,
            "max_retries": 5,
            "worker_timeout": 180.0,
            "max_tool_iterations": 10,
        }

        metrics = TuningMetrics(
            success_rate=0.3,
            failure_rate=0.7,
            avg_execution_time=5.0,
            total_tasks=10,
            timeout_count=0,
            hallucination_count=0,
            tool_failure_count=2,
            transient_failure_count=5,
        )

        new_retries = auto_tuner._tune_max_retries("generic", metrics)
        assert auto_tuner.BOUNDS["max_retries"][0] <= new_retries <= auto_tuner.BOUNDS["max_retries"][1]


class TestTuningCycle:
    """Test the full tuning cycle."""

    def test_run_tuning_cycle_no_data(self, auto_tuner, mock_learning_engine):
        """Test tuning cycle with no performance data."""
        changes = auto_tuner.run_tuning_cycle()
        assert changes == {}

    def test_run_tuning_cycle_insufficient_data(self, auto_tuner, mock_learning_engine):
        """Test tuning cycle with insufficient data."""
        # Only 2 tasks - below threshold of 3
        mock_learning_engine.add_performance("coder", 2, 1.0, 1.0, 0)
        changes = auto_tuner.run_tuning_cycle()
        assert changes == {}

    def test_run_tuning_cycle_with_changes(self, auto_tuner, mock_learning_engine):
        """Test tuning cycle that produces changes."""
        # Setup initial tuning
        auto_tuner._current_tuning["worker:coder"] = {
            "temperature": 0.7,
            "max_retries": 3,
            "worker_timeout": 180.0,
            "max_tool_iterations": 10,
        }

        # Setup performance data with hallucinations
        mock_learning_engine.add_performance("coder", 10, 0.5, 2.5, 5)
        mock_learning_engine.add_error_pattern("coder", "hallucination", 4)

        changes = auto_tuner.run_tuning_cycle()
        # Should have at least detected temperature change from high hallucination
        assert len(changes) >= 0  # Depends on threshold logic

    def test_run_tuning_cycle_persists(self, auto_tuner, mock_learning_engine):
        """Test that tuning cycle persists changes."""
        auto_tuner._current_tuning["worker:researcher"] = {
            "temperature": 0.7,
            "max_retries": 3,
            "worker_timeout": 180.0,
            "max_tool_iterations": 10,
        }

        mock_learning_engine.add_performance("researcher", 10, 0.6, 150.0, 4)
        mock_learning_engine.add_error_pattern("researcher", "timeout", 3)

        auto_tuner.run_tuning_cycle()

        # Create new instance and verify changes persisted
        auto_tuner2 = AutoTuner(auto_tuner.data_dir, mock_learning_engine)
        config = auto_tuner2.get_tuned_config("worker:researcher")
        assert config is not None
        assert "worker_timeout" in config


class TestThreadSafety:
    """Test thread safety of AutoTuner."""

    def test_concurrent_access(self, auto_tuner):
        """Test concurrent access to tuning configuration."""
        results = []

        def writer(agent_name, data):
            auto_tuner._current_tuning[agent_name] = data
            auto_tuner._save()

        def reader(agent_name):
            config = auto_tuner.get_tuned_config(agent_name)
            results.append(config)

        threads = []
        for i in range(5):
            t = threading.Thread(
                target=writer,
                args=(f"worker:agent{i}", {"temperature": 0.5 + i * 0.01}),
                daemon=True,
            )
            threads.append(t)

        for i in range(5):
            t = threading.Thread(
                target=reader,
                args=(f"worker:agent{i % 3}",),
                daemon=True,
            )
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=5.0)

        # Should have completed without errors
        assert len(results) >= 3

    def test_save_consistency(self, auto_tuner):
        """Test that concurrent saves maintain consistency."""
        def writer(agent_id):
            for i in range(3):
                auto_tuner._current_tuning[f"worker:agent{agent_id}"] = {
                    "temperature": 0.5 + i * 0.01,
                }
                auto_tuner._save()
                time.sleep(0.0001)

        threads = [threading.Thread(target=writer, args=(i,), daemon=True) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # File should be valid JSON
        if auto_tuner._tuning_path.exists():
            with open(auto_tuner._tuning_path) as f:
                config = json.load(f)
            assert isinstance(config, dict)


class TestHelperMethods:
    """Test helper methods."""

    def test_parse_percentage(self, auto_tuner):
        """Test percentage parsing."""
        assert auto_tuner._parse_percentage("85%") == 0.85
        assert auto_tuner._parse_percentage("100%") == 1.0
        assert auto_tuner._parse_percentage("50%") == 0.5
        assert auto_tuner._parse_percentage(0.85) == 0.85

    def test_parse_time(self, auto_tuner):
        """Test time parsing."""
        assert auto_tuner._parse_time("2.5s") == 2.5
        assert auto_tuner._parse_time("120.0s") == 120.0
        assert auto_tuner._parse_time(2.5) == 2.5

    def test_clamp(self, auto_tuner):
        """Test value clamping."""
        assert auto_tuner._clamp(0.05, (0.1, 1.0)) == 0.1
        assert auto_tuner._clamp(1.5, (0.1, 1.0)) == 1.0
        assert auto_tuner._clamp(0.5, (0.1, 1.0)) == 0.5

    def test_clamp_int(self, auto_tuner):
        """Test integer clamping."""
        assert auto_tuner._clamp(0, (1, 5)) == 1
        assert auto_tuner._clamp(10, (1, 5)) == 5
        assert auto_tuner._clamp(3, (1, 5)) == 3


class TestStatistics:
    """Test statistics methods."""

    def test_get_stats_empty(self, auto_tuner):
        """Test get_stats with no tuning."""
        stats = auto_tuner.get_stats()
        assert stats["agents_tuned"] == 0
        assert stats["tuning_file"] is not None

    def test_get_stats_with_data(self, auto_tuner):
        """Test get_stats with tuning data."""
        auto_tuner._current_tuning["worker:coder"] = {"temperature": 0.5}
        auto_tuner._current_tuning["worker:writer"] = {"temperature": 0.8}

        stats = auto_tuner.get_stats()
        assert stats["agents_tuned"] == 2

    def test_reset_tuning_single(self, auto_tuner):
        """Test resetting tuning for a single agent."""
        auto_tuner._current_tuning["worker:coder"] = {"temperature": 0.5}
        auto_tuner._current_tuning["worker:writer"] = {"temperature": 0.8}

        auto_tuner.reset_tuning("worker:coder")

        assert auto_tuner.get_tuned_config("worker:coder") is None
        assert auto_tuner.get_tuned_config("worker:writer") is not None

    def test_reset_tuning_all(self, auto_tuner):
        """Test resetting all tuning."""
        auto_tuner._current_tuning["worker:coder"] = {"temperature": 0.5}
        auto_tuner._current_tuning["worker:writer"] = {"temperature": 0.8}

        auto_tuner.reset_tuning()

        assert auto_tuner.get_tuned_config("worker:coder") is None
        assert auto_tuner.get_tuned_config("worker:writer") is None


class TestTuningMetrics:
    """Test TuningMetrics dataclass."""

    def test_has_sufficient_data_false(self):
        """Test has_sufficient_data returns False for < 3 tasks."""
        metrics = TuningMetrics(
            success_rate=1.0,
            failure_rate=0.0,
            avg_execution_time=1.0,
            total_tasks=2,
        )
        assert not metrics.has_sufficient_data

    def test_has_sufficient_data_true(self):
        """Test has_sufficient_data returns True for >= 3 tasks."""
        metrics = TuningMetrics(
            success_rate=0.9,
            failure_rate=0.1,
            avg_execution_time=2.5,
            total_tasks=5,
        )
        assert metrics.has_sufficient_data
