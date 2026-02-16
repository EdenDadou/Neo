"""
Guardian — Supervision et Persistance du Process Neo Core
==========================================================

Garantit que Neo ne s'éteint jamais (Commandement #1).

Composants :
- GracefulShutdown : gestion des signaux (SIGTERM, SIGINT, SIGHUP)
- StateSnapshot : sauvegarde/restauration d'état pour recovery
- Guardian : superviseur de process (auto-restart avec backoff)
- GuardianConfig : configuration du guardian

Architecture :
- Guardian (process parent) lance `neo chat` (process enfant)
- Si l'enfant crash → Guardian le relance avec backoff exponentiel
- Exit code 0 = quit normal → pas de restart
- Exit code 42 = restart demandé → restart immédiat
- Autre code = crash → restart avec backoff (1s → 2s → 4s → ... → 60s max)
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Code de sortie spécial pour demander un restart
EXIT_CODE_RESTART = 42
EXIT_CODE_NORMAL = 0


# ---------------------------------------------------------------------------
# 1. GuardianConfig
# ---------------------------------------------------------------------------

@dataclass
class GuardianConfig:
    """Configuration du Guardian."""

    max_restarts_per_hour: int = 10
    base_restart_delay: float = 1.0       # Délai initial entre restarts (secondes)
    max_restart_delay: float = 60.0       # Délai max entre restarts
    stable_threshold_seconds: float = 300.0  # 5 min sans crash = stable → reset backoff
    state_snapshot_interval: float = 600.0   # Sauvegarder le state toutes les 10 min
    state_dir: Path = field(default_factory=lambda: Path("data/guardian"))
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "max_restarts_per_hour": self.max_restarts_per_hour,
            "base_restart_delay": self.base_restart_delay,
            "max_restart_delay": self.max_restart_delay,
            "stable_threshold_seconds": self.stable_threshold_seconds,
            "state_snapshot_interval": self.state_snapshot_interval,
            "state_dir": str(self.state_dir),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GuardianConfig:
        if "state_dir" in data:
            data = {**data, "state_dir": Path(data["state_dir"])}
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


# ---------------------------------------------------------------------------
# 2. StateSnapshot — Sauvegarde d'état
# ---------------------------------------------------------------------------

@dataclass
class StateSnapshot:
    """
    Snapshot de l'état du système pour recovery après restart.

    Sérialisé en JSON dans data/guardian/state.json.
    Chargé au redémarrage pour restaurer le contexte.
    """

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    heartbeat_pulse_count: int = 0
    turn_count: int = 0
    active_tasks: list[str] = field(default_factory=list)
    shutdown_reason: str = "unknown"  # signal | crash | user_quit | guardian_restart
    uptime_seconds: float = 0.0
    restart_count: int = 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "heartbeat_pulse_count": self.heartbeat_pulse_count,
            "turn_count": self.turn_count,
            "active_tasks": self.active_tasks,
            "shutdown_reason": self.shutdown_reason,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "restart_count": self.restart_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StateSnapshot:
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def save(self, state_dir: Path) -> None:
        """Sauvegarde le snapshot sur disque."""
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / "state.json"
        try:
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            logger.debug("[Guardian] State snapshot sauvegardé: %s", state_file)
        except Exception as e:
            logger.error("[Guardian] Erreur sauvegarde state: %s", e)

    @classmethod
    def load(cls, state_dir: Path) -> Optional[StateSnapshot]:
        """Charge le dernier snapshot depuis le disque."""
        state_file = state_dir / "state.json"
        if not state_file.exists():
            return None
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.warning("[Guardian] Erreur chargement state: %s", e)
            return None

    @classmethod
    def clear(cls, state_dir: Path) -> None:
        """Supprime le fichier state (après quit normal)."""
        state_file = state_dir / "state.json"
        if state_file.exists():
            try:
                state_file.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 3. GracefulShutdown — Gestion des signaux
# ---------------------------------------------------------------------------

class GracefulShutdown:
    """
    Gère l'arrêt gracieux du process Neo.

    Intercepte SIGTERM, SIGINT, SIGHUP et exécute les callbacks
    de cleanup avant de laisser le process se terminer.
    """

    def __init__(self):
        self._shutdown_requested: bool = False
        self._callbacks: list[Callable] = []
        self._cleanup_done: bool = False
        self._start_time: float = time.time()
        self._state_dir: Path = Path("data/guardian")

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def install_handlers(self) -> None:
        """Installe les handlers de signaux."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        # SIGHUP existe sur Linux/Mac mais pas sur Windows
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self._handle_signal)
        logger.info("[Guardian] Signal handlers installés (SIGTERM, SIGINT, SIGHUP)")

    def add_cleanup_callback(self, callback: Callable) -> None:
        """Ajoute un callback exécuté lors du shutdown gracieux."""
        self._callbacks.append(callback)

    def set_state_dir(self, state_dir: Path) -> None:
        """Définit le répertoire pour les state snapshots."""
        self._state_dir = state_dir

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handler appelé par les signaux OS."""
        sig_name = signal.Signals(signum).name if signum in signal.Signals.__members__.values() else str(signum)
        logger.info("[Guardian] Signal reçu: %s", sig_name)

        if self._shutdown_requested:
            # Deuxième signal → forcer la sortie
            logger.warning("[Guardian] Deuxième signal — forçage de la sortie")
            sys.exit(1)

        self._shutdown_requested = True
        self._run_cleanup(shutdown_reason="signal")

    def _run_cleanup(self, shutdown_reason: str = "unknown") -> None:
        """Exécute tous les callbacks de cleanup (une seule fois)."""
        if self._cleanup_done:
            return
        self._cleanup_done = True

        logger.info("[Guardian] Cleanup en cours (%d callbacks)...", len(self._callbacks))

        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                logger.error("[Guardian] Erreur callback cleanup: %s", e)

        # Sauvegarder le state snapshot
        snapshot = StateSnapshot(
            shutdown_reason=shutdown_reason,
            uptime_seconds=self.uptime_seconds,
        )
        snapshot.save(self._state_dir)

        logger.info("[Guardian] Cleanup terminé (uptime: %.1fs)", self.uptime_seconds)

    def save_state(self, shutdown_reason: str = "user_quit", **kwargs) -> None:
        """Sauvegarde un state snapshot manuellement."""
        snapshot = StateSnapshot(
            shutdown_reason=shutdown_reason,
            uptime_seconds=self.uptime_seconds,
            **kwargs,
        )
        snapshot.save(self._state_dir)

    def clear_state(self) -> None:
        """Supprime le state (après quit normal)."""
        StateSnapshot.clear(self._state_dir)


# ---------------------------------------------------------------------------
# 4. Guardian — Superviseur de process
# ---------------------------------------------------------------------------

class Guardian:
    """
    Superviseur de process qui garantit que Neo reste vivant.

    Architecture :
    - Process parent léger (le Guardian)
    - Lance `neo chat` comme subprocess
    - Monitore et relance en cas de crash
    - Backoff exponentiel entre restarts
    - Protection contre les boucles infinies
    """

    def __init__(self, config: Optional[GuardianConfig] = None):
        self.config = config or GuardianConfig()
        self._child_process: Optional[subprocess.Popen] = None
        self._restart_count: int = 0
        self._restart_times: list[float] = []
        self._current_delay: float = self.config.base_restart_delay
        self._running: bool = False
        self._start_time: float = 0.0
        self._total_restarts: int = 0

    def run(self) -> None:
        """
        Boucle principale du Guardian.

        Lance Neo et le relance en cas de crash.
        """
        if not self.config.enabled:
            logger.info("[Guardian] Désactivé par configuration")
            return

        self._running = True
        self._start_time = time.time()

        # Installer le handler pour arrêter proprement le Guardian
        signal.signal(signal.SIGTERM, self._stop_signal)
        signal.signal(signal.SIGINT, self._stop_signal)

        logger.info("[Guardian] Démarré — surveillance active")
        self._log("Guardian démarré")

        while self._running:
            # Lancer Neo
            child_start = time.time()
            exit_code = self._run_child()
            child_uptime = time.time() - child_start

            if not self._running:
                break

            # Analyser le code de sortie
            if exit_code == EXIT_CODE_NORMAL:
                logger.info("[Guardian] Neo s'est arrêté normalement (code 0)")
                self._log(f"Neo arrêté normalement (uptime: {child_uptime:.0f}s)")
                break

            if exit_code == EXIT_CODE_RESTART:
                logger.info("[Guardian] Restart demandé (code 42)")
                self._log(f"Restart demandé (uptime: {child_uptime:.0f}s)")
                self._total_restarts += 1
                # Restart immédiat, pas de backoff
                continue

            # Crash — vérifier si on doit relancer
            logger.warning("[Guardian] Neo a crashé (code %d, uptime: %.0fs)",
                          exit_code, child_uptime)
            self._log(f"Crash détecté (code: {exit_code}, uptime: {child_uptime:.0f}s)")

            if not self._should_restart():
                logger.error("[Guardian] Trop de restarts — abandon")
                self._log("ABANDON: trop de restarts en 1 heure")
                break

            # Reset backoff si le process était stable
            if child_uptime >= self.config.stable_threshold_seconds:
                self._current_delay = self.config.base_restart_delay
                logger.info("[Guardian] Process était stable — reset backoff")

            # Attendre avant de relancer (backoff)
            self._wait_backoff()
            self._total_restarts += 1

        logger.info("[Guardian] Arrêté (total restarts: %d)", self._total_restarts)
        self._log(f"Guardian arrêté (total restarts: {self._total_restarts})")

    def _run_child(self) -> int:
        """Lance le process Neo et attend sa fin. Retourne le exit code."""
        try:
            # Trouver l'exécutable neo
            neo_cmd = self._find_neo_command()

            self._child_process = subprocess.Popen(
                neo_cmd,
                env={**os.environ, "NEO_GUARDIAN_MODE": "1"},
            )

            logger.info("[Guardian] Neo lancé (PID: %d)", self._child_process.pid)

            # Attendre la fin du process
            exit_code = self._child_process.wait()
            self._child_process = None
            return exit_code

        except FileNotFoundError:
            logger.error("[Guardian] Commande 'neo' non trouvée")
            return 1
        except Exception as e:
            logger.error("[Guardian] Erreur lancement: %s", e)
            return 1

    def _find_neo_command(self) -> list[str]:
        """Détermine la commande pour lancer Neo."""
        # Option 1 : neo est dans le PATH
        # Option 2 : python -m neo_core.cli chat
        return [sys.executable, "-m", "neo_core.cli", "chat"]

    def _should_restart(self) -> bool:
        """Vérifie si un restart est autorisé (protection boucle infinie)."""
        now = time.time()

        # Enregistrer ce restart
        self._restart_times.append(now)

        # Nettoyer les restarts de plus d'1 heure
        one_hour_ago = now - 3600
        self._restart_times = [t for t in self._restart_times if t > one_hour_ago]

        # Vérifier la limite
        if len(self._restart_times) > self.config.max_restarts_per_hour:
            return False

        return True

    def _wait_backoff(self) -> None:
        """Attend avec backoff exponentiel avant de relancer."""
        delay = self._current_delay
        logger.info("[Guardian] Backoff: %.1fs avant restart", delay)
        self._log(f"Backoff: {delay:.1f}s")

        time.sleep(delay)

        # Augmenter le délai pour le prochain crash (exponentiel)
        self._current_delay = min(
            self._current_delay * 2,
            self.config.max_restart_delay,
        )

    def _stop_signal(self, signum: int, frame: Any) -> None:
        """Arrête le Guardian proprement."""
        logger.info("[Guardian] Signal d'arrêt reçu")
        self._running = False

        # Transmettre le signal au process enfant
        if self._child_process and self._child_process.poll() is None:
            try:
                self._child_process.send_signal(signum)
                self._child_process.wait(timeout=10)
            except (subprocess.TimeoutExpired, Exception):
                self._child_process.kill()

    def _log(self, message: str) -> None:
        """Écrit un message dans le log du Guardian."""
        try:
            self.config.state_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.config.state_dir / "guardian.log"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception:
            pass

    def stop(self) -> None:
        """Arrête le Guardian et le process enfant."""
        self._running = False
        if self._child_process and self._child_process.poll() is None:
            try:
                self._child_process.terminate()
                self._child_process.wait(timeout=10)
            except (subprocess.TimeoutExpired, Exception):
                self._child_process.kill()

    def get_status(self) -> dict:
        """Retourne le statut du Guardian."""
        return {
            "running": self._running,
            "total_restarts": self._total_restarts,
            "current_backoff_delay": self._current_delay,
            "restarts_last_hour": len(self._restart_times),
            "max_restarts_per_hour": self.config.max_restarts_per_hour,
            "child_pid": self._child_process.pid if self._child_process else None,
            "uptime": time.time() - self._start_time if self._start_time else 0,
        }
