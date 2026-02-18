"""
Neo Core — Daemon Manager
============================
Gère le cycle de vie du processus Neo en arrière-plan.

Commandes :
    neo start     → Démarre Neo en arrière-plan (heartbeat + API)
    neo stop      → Arrête le processus
    neo restart   → stop + start
    neo status    → Vérifie si Neo tourne

Architecture :
    - PID file dans data/.neo.pid
    - Logs dans data/neo.log
    - Le daemon lance la boucle asyncio avec heartbeat + API server
    - Gestion propre des signaux SIGTERM/SIGINT
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import psutil

logger = logging.getLogger(__name__)

# Chemins par défaut
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_PID_FILE = _DATA_DIR / ".neo.pid"
_LOG_FILE = _DATA_DIR / "neo.log"


def _get_pid_file() -> Path:
    """Retourne le chemin du PID file."""
    return _PID_FILE


def _get_log_file() -> Path:
    """Retourne le chemin du fichier de log."""
    return _LOG_FILE


def _read_pid() -> Optional[int]:
    """Lit le PID depuis le fichier. Retourne None si absent ou invalide."""
    pid_file = _get_pid_file()
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
        return pid
    except (ValueError, OSError):
        return None


def _write_pid(pid: int) -> None:
    """Écrit le PID dans le fichier."""
    pid_file = _get_pid_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid))


def _remove_pid() -> None:
    """Supprime le PID file."""
    pid_file = _get_pid_file()
    if pid_file.exists():
        pid_file.unlink()


def is_running() -> bool:
    """Vérifie si le daemon Neo est en cours d'exécution."""
    pid = _read_pid()
    if pid is None:
        return False
    try:
        proc = psutil.Process(pid)
        # Vérifier que c'est bien un process Python (pas un PID recyclé)
        return proc.is_running() and "python" in proc.name().lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # PID file périmé → nettoyer
        _remove_pid()
        return False


def get_status() -> dict:
    """Retourne le statut du daemon."""
    pid = _read_pid()
    running = is_running()

    info = {
        "running": running,
        "pid": pid if running else None,
        "pid_file": str(_get_pid_file()),
        "log_file": str(_get_log_file()),
    }

    if running and pid:
        try:
            proc = psutil.Process(pid)
            info["uptime_seconds"] = time.time() - proc.create_time()
            info["memory_mb"] = round(proc.memory_info().rss / 1024 / 1024, 1)
            info["cpu_percent"] = proc.cpu_percent(interval=0.1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return info


def _setup_logging() -> None:
    """Configure le logging pour le daemon (fichier + stdout)."""
    log_file = _get_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Handler fichier
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)

    # Handler console (pour neo start --foreground)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(console_handler)


async def _run_daemon(host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    Boucle principale du daemon.

    Lance en parallèle :
    1. Le serveur API REST (uvicorn)
    2. Le heartbeat (pulse périodique)
    """
    from neo_core.infra.registry import core_registry
    from neo_core.vox.api.server import create_app

    logger.info("Neo Core daemon starting — host=%s port=%d", host, port)

    # Bootstrap unique via le CoreRegistry
    # (la même instance sera utilisée par API, heartbeat et Telegram)
    vox = None
    try:
        vox = core_registry.get_vox()
        config = core_registry.get_config()
        vox.start_new_session(config.user_name)
        logger.info(
            "CoreRegistry bootstrap OK — single Vox=%s, Brain=%s, Memory=%s",
            id(vox), id(core_registry.get_brain()), id(core_registry.get_memory()),
        )
    except Exception as e:
        logger.warning("CoreRegistry bootstrap failed: %s — Telegram disabled", e)
        from neo_core.config import NeoConfig
        config = NeoConfig()

    # Créer l'app FastAPI (utilise aussi le CoreRegistry)
    app = create_app()

    # Lancer uvicorn en mode programmatique
    import uvicorn
    uvi_config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(uvi_config)

    # Signal handler pour shutdown propre (async-safe via loop.call_soon_threadsafe)
    shutdown_event = asyncio.Event()
    loop = asyncio.get_event_loop()

    def _signal_handler(signum, frame):
        # Signal handlers ne doivent pas appeler directement des méthodes asyncio
        # call_soon_threadsafe est la seule méthode asyncio safe depuis un signal handler
        try:
            loop.call_soon_threadsafe(shutdown_event.set)
        except RuntimeError:
            # Loop fermée — fallback direct
            shutdown_event.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # SIGHUP → reload config à chaud (async-safe)
    def _sighup_handler(signum, frame):
        def _do_reload():
            try:
                from neo_core.infra.registry import core_registry
                cfg = core_registry.get_config()
                cfg.reload()
                logger.info("Configuration rechargée avec succès")
            except Exception as e:
                logger.error("Échec du rechargement de config: %s", e)
        try:
            loop.call_soon_threadsafe(_do_reload)
        except RuntimeError:
            _do_reload()

    signal.signal(signal.SIGHUP, _sighup_handler)

    # Lancer les tâches
    async def run_server():
        await server.serve()

    async def run_heartbeat():
        """Lance le heartbeat si disponible."""
        try:
            from neo_core.infra.heartbeat import HeartbeatManager, HeartbeatConfig

            # Récupérer brain et memory depuis le CoreRegistry
            brain = core_registry.get_brain()
            memory = core_registry.get_memory()

            hb_config = HeartbeatConfig(interval_seconds=1800.0)
            heartbeat = HeartbeatManager(
                brain=brain,
                memory=memory,
                config=hb_config,
            )
            heartbeat.start()
            core_registry.set_heartbeat_manager(heartbeat)
            logger.info("Heartbeat démarré — interval=%ds", hb_config.interval_seconds)

            # Attendre le shutdown
            await shutdown_event.wait()

            # Arrêt propre
            heartbeat.stop()
            logger.info("Heartbeat arrêté")

        except ImportError as e:
            logger.warning("Heartbeat non disponible (%s) — daemon en mode API seul", e)
            await shutdown_event.wait()
        except Exception as e:
            logger.error("Heartbeat fatal: %s", e)
            await shutdown_event.wait()

    telegram_bot = None

    async def run_telegram():
        """Lance le bot Telegram si configuré."""
        nonlocal telegram_bot
        try:
            from neo_core.vox.integrations.telegram import (
                TelegramBot,
                load_telegram_config,
            )

            tg_config = load_telegram_config(config.data_dir)
            if not tg_config.bot_token or not tg_config.allowed_user_ids:
                logger.info("Telegram bot not configured — skipping")
                await shutdown_event.wait()
                return

            telegram_bot = TelegramBot(config=tg_config, vox=vox)
            # Enregistrer dans le CoreRegistry pour l'envoi proactif
            from neo_core.infra.registry import core_registry
            core_registry.set_telegram_bot(telegram_bot)
            await telegram_bot.start_polling()

            # Attendre le shutdown
            await shutdown_event.wait()
            await telegram_bot.stop()

        except ImportError:
            logger.info(
                "python-telegram-bot not installed — Telegram disabled. "
                "Install: pip install 'python-telegram-bot>=21.0'"
            )
            await shutdown_event.wait()
        except Exception as e:
            logger.error("Telegram bot error: %s", e)
            await shutdown_event.wait()

    async def watch_shutdown():
        """Attend le signal de shutdown et arrête le serveur."""
        await shutdown_event.wait()
        logger.info("Shutdown en cours...")
        server.should_exit = True

    # Exécuter tout en parallèle
    await asyncio.gather(
        run_server(),
        run_heartbeat(),
        run_telegram(),
        watch_shutdown(),
        return_exceptions=True,
    )

    logger.info("Neo Core daemon arrêté proprement")


def start(foreground: bool = False, host: str = "127.0.0.1", port: int = 8000) -> dict:
    """
    Démarre le daemon Neo Core.

    Args:
        foreground: Si True, tourne au premier plan (bloquant).
                    Si False, fork en arrière-plan.
        host: Adresse d'écoute de l'API.
        port: Port de l'API.

    Returns:
        dict avec le statut.
    """
    if is_running():
        pid = _read_pid()
        return {"success": False, "message": f"Neo est déjà en cours d'exécution (PID {pid})"}

    if foreground:
        # Mode foreground : bloquant
        _setup_logging()
        _write_pid(os.getpid())
        logger.info("Neo Core daemon PID %d (foreground)", os.getpid())
        try:
            asyncio.run(_run_daemon(host=host, port=port))
        finally:
            _remove_pid()
        return {"success": True, "message": "Neo Core daemon arrêté"}

    # Mode background : fork
    try:
        pid = os.fork()
    except OSError as e:
        return {"success": False, "message": f"Fork failed: {e}"}

    if pid > 0:
        # Reap le premier fils (double-fork) pour éviter les zombies
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass

        # Parent — attendre que le deuxième fils (vrai daemon) écrive le PID file.
        # Le PID est écrit juste après le double-fork, AVANT le chargement des modèles,
        # donc ça devrait être quasi instantané. On laisse 15s de marge au cas où.
        for _ in range(30):  # 30 x 0.5s = 15s max
            time.sleep(0.5)
            if is_running():
                daemon_pid = _read_pid() or pid
                return {"success": True, "pid": daemon_pid, "message": f"Neo démarré (PID {daemon_pid})"}
        # Vérifier si le PID file existe même si is_running() échoue
        daemon_pid = _read_pid()
        if daemon_pid:
            return {"success": True, "pid": daemon_pid, "message": f"Neo démarré (PID {daemon_pid}) — initialisation en cours"}
        return {"success": False, "message": "Le processus a démarré puis s'est arrêté — vérifiez neo logs"}

    # Fils — devenir un daemon
    os.setsid()  # Nouvelle session (détaché du terminal)

    # Double fork pour éviter les zombie processes
    try:
        pid2 = os.fork()
    except OSError:
        sys.exit(1)

    if pid2 > 0:
        sys.exit(0)  # Premier fils quitte

    # Deuxième fils = le vrai daemon
    _setup_logging()
    _write_pid(os.getpid())
    logger.info("Neo Core daemon PID %d (background)", os.getpid())

    # Rediriger stdin/stdout/stderr — fermer les anciens pour éviter les fuites de FD
    _old_stdin = sys.stdin
    _old_stdout = sys.stdout
    _old_stderr = sys.stderr

    _new_stdin = open(os.devnull, "r")
    _new_stdout = open(str(_get_log_file()), "a")
    _new_stderr = open(str(_get_log_file()), "a")

    sys.stdin = _new_stdin
    sys.stdout = _new_stdout
    sys.stderr = _new_stderr

    # Fermer les anciens descripteurs (hérités du parent)
    try:
        _old_stdin.close()
    except Exception as e:
        logger.debug("Failed to close old stdin: %s", e)
    try:
        _old_stdout.close()
    except Exception as e:
        logger.debug("Failed to close old stdout: %s", e)
    try:
        _old_stderr.close()
    except Exception as e:
        logger.debug("Failed to close old stderr: %s", e)

    try:
        asyncio.run(_run_daemon(host=host, port=port))
    except Exception as e:
        logger.error("Daemon crash: %s", e)
    finally:
        _remove_pid()
        # Fermer proprement les nouveaux descripteurs
        for f in (_new_stdin, _new_stdout, _new_stderr):
            try:
                f.close()
            except Exception as e:
                logger.debug("Failed to close daemon FD: %s", e)
        sys.exit(0)


def stop() -> dict:
    """Arrête le daemon Neo Core."""
    if not is_running():
        return {"success": False, "message": "Neo n'est pas en cours d'exécution"}

    pid = _read_pid()
    try:
        os.kill(pid, signal.SIGTERM)

        # Attendre l'arrêt (max 10s)
        for _ in range(20):
            time.sleep(0.5)
            if not is_running():
                _remove_pid()
                return {"success": True, "message": f"Neo arrêté (PID {pid})"}

        # Force kill si toujours vivant
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        _remove_pid()
        return {"success": True, "message": f"Neo forcé à s'arrêter (PID {pid})"}

    except ProcessLookupError:
        _remove_pid()
        return {"success": True, "message": "Neo n'était plus en cours d'exécution"}
    except PermissionError:
        return {"success": False, "message": f"Permission refusée pour arrêter PID {pid}"}


def restart(host: str = "127.0.0.1", port: int = 8000) -> dict:
    """Redémarre le daemon Neo Core."""
    if is_running():
        result = stop()
        if not result["success"]:
            return result
        time.sleep(1)

    return start(foreground=False, host=host, port=port)


def generate_systemd_service(
    user: Optional[str] = None,
    python_path: Optional[str] = None,
    working_dir: Optional[str] = None,
) -> str:
    """
    Génère le contenu du fichier neo.service pour systemd.

    Usage :
        neo install-service  → crée /etc/systemd/system/neo.service
    """
    user = user or os.getenv("USER", "neo")
    python_path = python_path or sys.executable
    working_dir = working_dir or str(_PROJECT_ROOT)

    return f"""[Unit]
Description=Neo Core — Écosystème IA Multi-Agents Autonome
Documentation=https://github.com/EdenDadou/Neo
After=network.target

[Service]
Type=simple
User={user}
Group={user}
WorkingDirectory={working_dir}
ExecStart={python_path} -m neo_core.vox.cli start --foreground
ExecStop=/bin/kill -SIGTERM $MAINPID
Restart=always
RestartSec=10
StandardOutput=append:{_DATA_DIR}/neo.log
StandardError=append:{_DATA_DIR}/neo.log

# Sécurité
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths={_DATA_DIR}

# Limites
LimitNOFILE=65536
TimeoutStopSec=30

# Environnement
EnvironmentFile=-{_PROJECT_ROOT}/.env

[Install]
WantedBy=multi-user.target
"""


def install_service() -> dict:
    """
    Installe le service systemd.

    Nécessite les droits sudo.
    """
    service_content = generate_systemd_service()
    service_path = Path("/etc/systemd/system/neo.service")

    try:
        service_path.write_text(service_content)
        import subprocess
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "enable", "neo.service"], check=True)
        return {
            "success": True,
            "message": f"Service installé: {service_path}",
            "commands": [
                "sudo systemctl start neo",
                "sudo systemctl status neo",
                "sudo journalctl -u neo -f",
            ],
        }
    except (PermissionError, FileNotFoundError):
        # Écrire dans un fichier local pour que l'utilisateur puisse l'installer
        local_path = _PROJECT_ROOT / "neo.service"
        local_path.write_text(service_content)
        return {
            "success": False,
            "message": f"Pas les droits sudo. Fichier généré: {local_path}",
            "commands": [
                f"sudo cp {local_path} /etc/systemd/system/neo.service",
                "sudo systemctl daemon-reload",
                "sudo systemctl enable neo",
                "sudo systemctl start neo",
            ],
        }
