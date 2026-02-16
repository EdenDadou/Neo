"""
Neo Core — Module de Résilience
=================================
Retry avec exponential backoff, circuit breaker, et monitoring de santé.

Utilisé par Brain et Worker pour gérer les erreurs transitoires
(rate limits, timeouts, erreurs serveur) sans crasher.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Set

from neo_core.config import ResilienceConfig


# ─── Retry avec Exponential Backoff ────────────────────────

@dataclass
class RetryConfig:
    """Configuration du retry."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    retryable_status_codes: set = field(
        default_factory=lambda: {429, 500, 502, 503, 529}
    )

    @classmethod
    def from_resilience_config(cls, rc: ResilienceConfig) -> RetryConfig:
        """Crée un RetryConfig depuis la config globale."""
        return cls(
            max_retries=rc.max_retries,
            base_delay=rc.base_delay,
            max_delay=rc.max_delay,
            exponential_base=rc.exponential_base,
        )


def compute_backoff_delay(attempt: int, config: RetryConfig) -> float:
    """Calcule le délai de backoff pour une tentative donnée."""
    delay = config.base_delay * (config.exponential_base ** attempt)
    return min(delay, config.max_delay)


class RetryableError(Exception):
    """Erreur qui peut être retentée (ex: rate limit, timeout)."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class NonRetryableError(Exception):
    """Erreur qui ne doit PAS être retentée (ex: bad request, auth)."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


async def retry_with_backoff(
    func: Callable,
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> Any:
    """
    Exécute une fonction async avec retry et exponential backoff.

    Args:
        func: Coroutine à exécuter (sans arguments — utiliser functools.partial)
        config: Configuration du retry
        on_retry: Callback optionnel (attempt, error, delay) appelé avant chaque retry

    Returns:
        Le résultat de func()

    Raises:
        NonRetryableError: Si l'erreur n'est pas retryable
        RetryableError: Si toutes les tentatives ont échoué
        Exception: Toute autre exception non gérée
    """
    if config is None:
        config = RetryConfig()

    last_error: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func()

        except NonRetryableError:
            raise  # Ne pas retenter

        except RetryableError as e:
            last_error = e
            if attempt < config.max_retries:
                delay = compute_backoff_delay(attempt, config)
                if on_retry:
                    on_retry(attempt, e, delay)
                await asyncio.sleep(delay)
            # Sinon, on sort de la boucle et raise

        except asyncio.TimeoutError as e:
            last_error = RetryableError(f"Timeout: {e}", status_code=0)
            if attempt < config.max_retries:
                delay = compute_backoff_delay(attempt, config)
                if on_retry:
                    on_retry(attempt, last_error, delay)
                await asyncio.sleep(delay)

        except Exception as e:
            # Exception inattendue — ne pas retenter
            raise

    # Toutes les tentatives épuisées
    raise last_error or RetryableError("Toutes les tentatives ont échoué")


# ─── Circuit Breaker ──────────────────────────────────────

@dataclass
class CircuitBreaker:
    """
    Circuit breaker pour protéger contre les cascades d'échecs.

    États :
    - closed : fonctionnement normal
    - open : bloque les appels (trop d'échecs)
    - half_open : laisse passer un appel pour tester la recovery
    """
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    state: str = "closed"
    _consecutive_failures: int = 0
    _last_failure_time: float = 0.0
    _total_successes: int = 0
    _total_failures: int = 0

    def can_execute(self) -> bool:
        """Vérifie si un appel est autorisé."""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Vérifier si le timeout de recovery est passé
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self.state = "half_open"
                return True
            return False

        if self.state == "half_open":
            return True  # Laisser passer un appel pour tester

        return False

    def record_success(self) -> None:
        """Enregistre un succès."""
        self._consecutive_failures = 0
        self._total_successes += 1

        if self.state == "half_open":
            self.state = "closed"  # Recovery réussie

    def record_failure(self) -> None:
        """Enregistre un échec."""
        self._consecutive_failures += 1
        self._total_failures += 1
        self._last_failure_time = time.time()

        if self.state == "half_open":
            self.state = "open"  # Recovery échouée
        elif self._consecutive_failures >= self.failure_threshold:
            self.state = "open"

    def reset(self) -> None:
        """Remet le circuit breaker à zéro."""
        self.state = "closed"
        self._consecutive_failures = 0
        self._last_failure_time = 0.0

    def get_stats(self) -> dict:
        """Retourne les statistiques du circuit breaker."""
        return {
            "state": self.state,
            "consecutive_failures": self._consecutive_failures,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "failure_threshold": self.failure_threshold,
        }


class CircuitOpenError(Exception):
    """Levée quand le circuit breaker est ouvert."""
    pass


# ─── Health Monitor ───────────────────────────────────────

@dataclass
class HealthMonitor:
    """
    Moniteur de santé du système Neo Core.

    Collecte des métriques sur les appels API, les erreurs,
    et l'état des composants.
    """
    api_circuit: CircuitBreaker = field(default_factory=CircuitBreaker)
    _recent_calls: deque = field(default_factory=lambda: deque(maxlen=100))
    _start_time: float = field(default_factory=time.time)
    total_calls: int = 0
    total_errors: int = 0
    total_successes: int = 0
    _memory_healthy: bool = True

    def record_api_call(self, success: bool, duration: float = 0.0,
                        error: str = "") -> None:
        """Enregistre un appel API."""
        self.total_calls += 1
        entry = {
            "time": time.time(),
            "success": success,
            "duration": duration,
            "error": error,
        }
        self._recent_calls.append(entry)

        if success:
            self.total_successes += 1
            self.api_circuit.record_success()
        else:
            self.total_errors += 1
            self.api_circuit.record_failure()

    def set_memory_health(self, healthy: bool) -> None:
        """Met à jour l'état de santé de la mémoire."""
        self._memory_healthy = healthy

    @property
    def error_rate(self) -> float:
        """Taux d'erreur sur les 100 derniers appels."""
        if not self._recent_calls:
            return 0.0
        errors = sum(1 for c in self._recent_calls if not c["success"])
        return errors / len(self._recent_calls)

    @property
    def avg_response_time(self) -> float:
        """Temps de réponse moyen sur les appels récents."""
        if not self._recent_calls:
            return 0.0
        durations = [c["duration"] for c in self._recent_calls if c["duration"] > 0]
        return sum(durations) / len(durations) if durations else 0.0

    @property
    def uptime(self) -> float:
        """Temps de fonctionnement en secondes."""
        return time.time() - self._start_time

    def can_make_api_call(self) -> bool:
        """Vérifie si un appel API est autorisé (circuit breaker)."""
        return self.api_circuit.can_execute()

    def get_health_report(self) -> dict:
        """Retourne un rapport de santé complet."""
        return {
            "status": self._compute_overall_status(),
            "uptime_seconds": round(self.uptime, 1),
            "api": {
                "circuit_state": self.api_circuit.state,
                "total_calls": self.total_calls,
                "total_errors": self.total_errors,
                "error_rate": round(self.error_rate, 3),
                "avg_response_time": round(self.avg_response_time, 3),
            },
            "memory": {
                "healthy": self._memory_healthy,
            },
        }

    def _compute_overall_status(self) -> str:
        """Calcule le statut global du système."""
        if self.api_circuit.state == "open":
            return "degraded"
        if not self._memory_healthy:
            return "degraded"
        if self.error_rate > 0.5:
            return "warning"
        if self.error_rate > 0.1:
            return "ok_with_errors"
        return "healthy"


# ─── Factory helper ───────────────────────────────────────

def create_resilience_from_config(rc: ResilienceConfig) -> tuple[RetryConfig, CircuitBreaker, HealthMonitor]:
    """Crée les composants de résilience depuis la config globale."""
    retry = RetryConfig.from_resilience_config(rc)
    circuit = CircuitBreaker(
        failure_threshold=rc.circuit_breaker_threshold,
        recovery_timeout=rc.circuit_breaker_recovery,
    )
    health = HealthMonitor(api_circuit=circuit)
    return retry, circuit, health
