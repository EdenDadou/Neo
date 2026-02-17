"""
Auto-Tuning System for Neo Core
================================
Level 1: Automatic parameter adjustment based on performance data.

Analyzes performance metrics from LearningEngine and adjusts:
- temperature: Lower if hallucination errors are high, raise if outputs are repetitive
- max_retries: Increase for workers with high transient failure rates
- worker_timeout: Increase if timeout errors are frequent
- max_tool_iterations: Adjust based on average tool calls per task

Persists tuned parameters to data/auto_tuning.json for automatic loading
by get_agent_model() in config.py.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TuningMetrics:
    """Metrics used for auto-tuning decisions."""
    success_rate: float  # 0.0-1.0
    failure_rate: float  # 0.0-1.0
    avg_execution_time: float  # seconds
    total_tasks: int
    timeout_count: int = 0
    hallucination_count: int = 0
    tool_failure_count: int = 0
    transient_failure_count: int = 0  # Auth, rate limit, server errors

    @property
    def has_sufficient_data(self) -> bool:
        """Returns True if we have enough data to make tuning decisions."""
        return self.total_tasks >= 3


class AutoTuner:
    """
    Niveau 1 d'auto-amélioration : ajustement automatique des paramètres.

    Analyse les performances passées (via LearningEngine) et ajuste:
    - temperature par worker type
    - max_retries
    - worker_timeout
    - max_tool_iterations

    Les paramètres sont sauvegardés dans data/auto_tuning.json
    et chargés automatiquement par get_agent_model().
    """

    TUNING_FILE = "auto_tuning.json"

    # Bornes de sécurité (ne jamais dépasser)
    BOUNDS = {
        "temperature": (0.1, 1.0),
        "max_retries": (1, 5),
        "worker_timeout": (60.0, 600.0),
        "max_tool_iterations": (5, 25),
    }

    # Thresholds for triggering tuning adjustments
    HALLUCINATION_THRESHOLD = 0.3  # If >= 30% of failures are hallucinations
    TIMEOUT_THRESHOLD = 0.4  # If >= 40% of failures are timeouts
    TRANSIENT_FAILURE_THRESHOLD = 0.5  # If >= 50% of failures are transient

    def __init__(self, data_dir: Path, learning_engine):
        """
        Initialize the AutoTuner.

        Args:
            data_dir: Path to data directory (where auto_tuning.json is stored)
            learning_engine: LearningEngine instance for reading performance data
        """
        self.data_dir = data_dir
        self.learning = learning_engine
        self._tuning_path = data_dir / self.TUNING_FILE
        self._current_tuning: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """Load existing tuning from JSON file."""
        with self._lock:
            if self._tuning_path.exists():
                try:
                    with open(self._tuning_path, "r") as f:
                        self._current_tuning = json.load(f)
                    logger.debug(
                        "Loaded auto-tuning from %s (%d agents)",
                        self._tuning_path,
                        len(self._current_tuning),
                    )
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(
                        "Failed to load auto-tuning file %s: %s",
                        self._tuning_path,
                        e,
                    )
                    self._current_tuning = {}
            else:
                self._current_tuning = {}

    def _save(self) -> None:
        """Save current tuning to JSON file (thread-safe)."""
        with self._lock:
            try:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                with open(self._tuning_path, "w") as f:
                    json.dump(self._current_tuning, f, indent=2)
                logger.debug(
                    "Saved auto-tuning to %s (%d agents)",
                    self._tuning_path,
                    len(self._current_tuning),
                )
            except IOError as e:
                logger.error("Failed to save auto-tuning file: %s", e)

    def run_tuning_cycle(self) -> dict:
        """
        Analyze performance data and compute new parameters.
        Called periodically (e.g., every N tasks or on daemon restart).

        Returns:
            dict of changes made: {"agent_key.param": new_value, ...}
        """
        changes = {}

        # Get performance data from learning engine
        perf_summary = self.learning.get_performance_summary()

        for worker_type, perf_data in perf_summary.items():
            agent_key = f"worker:{worker_type}"

            # Parse performance summary data
            try:
                total_tasks = int(perf_data.get("total_tasks", 0))
                if total_tasks < 3:
                    continue  # Not enough data

                success_rate = self._parse_percentage(perf_data.get("success_rate", "100%"))
                avg_time_str = perf_data.get("avg_time", "0s")
                avg_time = self._parse_time(avg_time_str)
                failures = int(perf_data.get("failures", 0))

                metrics = self._build_metrics(
                    worker_type,
                    total_tasks,
                    success_rate,
                    avg_time,
                    failures,
                )
            except (ValueError, TypeError) as e:
                logger.debug(
                    "Failed to parse performance for %s: %s",
                    worker_type,
                    e,
                )
                continue

            if not metrics.has_sufficient_data:
                continue

            current = self._current_tuning.get(agent_key, {})

            # Temperature tuning
            new_temp = self._tune_temperature(worker_type, metrics)
            old_temp = current.get("temperature")
            if old_temp is not None and new_temp != old_temp:
                changes[f"{agent_key}.temperature"] = new_temp
                current["temperature"] = new_temp

            # Timeout tuning
            new_timeout = self._tune_timeout(worker_type, metrics)
            old_timeout = current.get("worker_timeout")
            if old_timeout is not None and new_timeout != old_timeout:
                changes[f"{agent_key}.worker_timeout"] = new_timeout
                current["worker_timeout"] = new_timeout

            # Max retries tuning
            new_retries = self._tune_max_retries(worker_type, metrics)
            old_retries = current.get("max_retries")
            if old_retries is not None and new_retries != old_retries:
                changes[f"{agent_key}.max_retries"] = new_retries
                current["max_retries"] = new_retries

            # Tool iterations tuning
            new_tool_iters = self._tune_max_tool_iterations(worker_type, metrics)
            old_tool_iters = current.get("max_tool_iterations")
            if old_tool_iters is not None and new_tool_iters != old_tool_iters:
                changes[f"{agent_key}.max_tool_iterations"] = new_tool_iters
                current["max_tool_iterations"] = new_tool_iters

            # Update or create the tuning entry
            if current or agent_key not in self._current_tuning:
                self._current_tuning[agent_key] = current

        if changes:
            self._save()
            logger.info(
                "Auto-tuning cycle complete: %d parameter(s) adjusted",
                len(changes),
            )

        return changes

    def get_tuned_config(self, agent_name: str) -> Optional[dict]:
        """
        Returns tuned params for an agent, or None if no tuning exists.

        Args:
            agent_name: Agent identifier (e.g., "worker:coder")

        Returns:
            dict with tuned parameters or None
        """
        with self._lock:
            return self._current_tuning.get(agent_name)

    # ─── Private tuning methods ────────────────────────────────

    def _tune_temperature(self, worker_type: str, metrics: TuningMetrics) -> float:
        """
        Tune temperature based on failure patterns.
        Lower if hallucination is high, raise if repetitive (low diversity).
        """
        current = self._current_tuning.get(f"worker:{worker_type}", {})
        base_temp = current.get("temperature", 0.5)  # fallback

        if metrics.failure_rate == 0:
            return base_temp  # No failures, keep it stable

        failure_count = metrics.total_tasks - int(metrics.total_tasks * metrics.success_rate)

        # Calculate hallucination ratio
        if failure_count > 0:
            hallucination_ratio = metrics.hallucination_count / failure_count
        else:
            hallucination_ratio = 0.0

        # If high hallucination → lower temperature
        if hallucination_ratio >= self.HALLUCINATION_THRESHOLD:
            new_temp = base_temp - 0.15
            logger.info(
                "Worker %s: high hallucination (%.0f%%) → lowering temperature %.2f → %.2f",
                worker_type,
                hallucination_ratio * 100,
                base_temp,
                new_temp,
            )
        else:
            # Keep current temperature
            new_temp = base_temp

        return self._clamp(new_temp, self.BOUNDS["temperature"])

    def _tune_timeout(self, worker_type: str, metrics: TuningMetrics) -> float:
        """
        Tune worker timeout based on execution time and timeout errors.
        Increase if timeouts are frequent or avg execution time is high.
        """
        current = self._current_tuning.get(f"worker:{worker_type}", {})
        base_timeout = current.get("worker_timeout", 180.0)  # fallback

        failure_count = metrics.total_tasks - int(metrics.total_tasks * metrics.success_rate)

        # Calculate timeout ratio
        if failure_count > 0:
            timeout_ratio = metrics.timeout_count / failure_count
        else:
            timeout_ratio = 0.0

        new_timeout = base_timeout

        # If high timeout rate → increase timeout
        if timeout_ratio >= self.TIMEOUT_THRESHOLD:
            increase = base_timeout * 0.3  # Add 30%
            new_timeout = base_timeout + increase
            logger.info(
                "Worker %s: high timeout rate (%.0f%%) → increasing timeout %.1f → %.1f",
                worker_type,
                timeout_ratio * 100,
                base_timeout,
                new_timeout,
            )

        # Also check if avg execution time is close to timeout
        if metrics.avg_execution_time > base_timeout * 0.7:
            increase = base_timeout * 0.25
            new_timeout = max(new_timeout, base_timeout + increase)
            logger.info(
                "Worker %s: avg time %.1fs is %.0f%% of timeout → increasing timeout",
                worker_type,
                metrics.avg_execution_time,
                (metrics.avg_execution_time / base_timeout) * 100,
            )

        return self._clamp(new_timeout, self.BOUNDS["worker_timeout"])

    def _tune_max_retries(self, worker_type: str, metrics: TuningMetrics) -> int:
        """
        Tune max_retries based on transient failure rate.
        Increase if many failures are transient (auth, rate limit, server errors).
        """
        current = self._current_tuning.get(f"worker:{worker_type}", {})
        base_retries = current.get("max_retries", 3)  # fallback

        failure_count = metrics.total_tasks - int(metrics.total_tasks * metrics.success_rate)

        if failure_count == 0:
            return base_retries

        # Calculate transient failure ratio
        transient_ratio = metrics.transient_failure_count / failure_count

        new_retries = base_retries

        # If high transient failure rate → increase retries
        if transient_ratio >= self.TRANSIENT_FAILURE_THRESHOLD:
            new_retries = base_retries + 1
            logger.info(
                "Worker %s: high transient failure rate (%.0f%%) → increasing retries %d → %d",
                worker_type,
                transient_ratio * 100,
                base_retries,
                new_retries,
            )

        return self._clamp(new_retries, self.BOUNDS["max_retries"])

    def _tune_max_tool_iterations(self, worker_type: str, metrics: TuningMetrics) -> int:
        """
        Tune max_tool_iterations based on average execution patterns.
        For now, keep it stable since we don't track iteration counts in performance data.
        """
        current = self._current_tuning.get(f"worker:{worker_type}", {})
        return current.get("max_tool_iterations", 10)  # Keep baseline

    # ─── Helper methods ───────────────────────────────────────

    def _build_metrics(
        self,
        worker_type: str,
        total_tasks: int,
        success_rate: float,
        avg_time: float,
        failures: int,
    ) -> TuningMetrics:
        """
        Build TuningMetrics from performance summary.
        Analyzes error patterns from LearningEngine to categorize failures.
        """
        failure_rate = 1.0 - success_rate

        # Get error patterns for this worker type
        error_patterns = self.learning.get_error_patterns()
        timeout_count = 0
        hallucination_count = 0
        tool_failure_count = 0
        transient_failure_count = 0

        for pattern in error_patterns:
            if pattern.worker_type == worker_type:
                if pattern.error_type == "timeout":
                    timeout_count = pattern.count
                elif pattern.error_type == "hallucination":
                    hallucination_count = pattern.count
                elif pattern.error_type == "tool_failure":
                    tool_failure_count = pattern.count
                elif pattern.error_type in ("auth_failure", "rate_limit", "server_error"):
                    transient_failure_count += pattern.count

        return TuningMetrics(
            success_rate=success_rate,
            failure_rate=failure_rate,
            avg_execution_time=avg_time,
            total_tasks=total_tasks,
            timeout_count=timeout_count,
            hallucination_count=hallucination_count,
            tool_failure_count=tool_failure_count,
            transient_failure_count=transient_failure_count,
        )

    def _parse_percentage(self, perf_str: str) -> float:
        """Parse percentage string like '85%' to 0.85."""
        if isinstance(perf_str, float):
            return perf_str
        s = str(perf_str).strip().rstrip("%")
        return float(s) / 100.0

    def _parse_time(self, time_str: str) -> float:
        """Parse time string like '2.5s' to 2.5 seconds."""
        if isinstance(time_str, (int, float)):
            return float(time_str)
        s = str(time_str).strip().rstrip("s")
        return float(s)

    def _clamp(self, value: float | int, bounds: tuple) -> float | int:
        """Clamp value to bounds (min, max)."""
        return max(bounds[0], min(bounds[1], value))

    def get_stats(self) -> dict:
        """Return statistics about the current tuning state."""
        with self._lock:
            last_modified = None
            if self._tuning_path.exists():
                try:
                    last_modified = datetime.fromtimestamp(
                        self._tuning_path.stat().st_mtime
                    ).isoformat()
                except (OSError, FileNotFoundError):
                    pass

            return {
                "agents_tuned": len(self._current_tuning),
                "tuning_file": str(self._tuning_path),
                "tuning_data": self._current_tuning,
                "last_modified": last_modified,
            }

    def reset_tuning(self, agent_name: Optional[str] = None) -> None:
        """
        Reset tuning for a specific agent or all agents.

        Args:
            agent_name: Agent to reset, or None to reset all
        """
        with self._lock:
            if agent_name:
                if agent_name in self._current_tuning:
                    del self._current_tuning[agent_name]
                    logger.info("Reset tuning for %s", agent_name)
            else:
                self._current_tuning.clear()
                logger.info("Reset all tuning")

        # Save outside the lock to avoid deadlock
        self._save()
