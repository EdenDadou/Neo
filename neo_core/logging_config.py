"""
Neo Core — Configuration centralisée du logging
=================================================
Fournit un logging structuré pour tout le projet.

Usage :
    from neo_core.logging_config import setup_logging
    setup_logging()  # Appeler une fois au démarrage

Chaque module utilise :
    import logging
    logger = logging.getLogger(__name__)
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


# ─── Constantes ──────────────────────────────────────

DEFAULT_LOG_DIR = Path("data")
DEFAULT_LOG_FILE = "neo.log"
DEFAULT_LOG_LEVEL = logging.INFO
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

# Format console (lisible, avec couleurs ANSI)
CONSOLE_FORMAT = "%(asctime)s [%(name)s] %(levelname)s — %(message)s"
CONSOLE_DATE_FORMAT = "%H:%M:%S"

# Format fichier (JSON-like, pour parsing)
FILE_FORMAT = '{"time":"%(asctime)s","logger":"%(name)s","level":"%(levelname)s","message":"%(message)s"}'
FILE_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Niveaux par défaut par module
MODULE_LEVELS = {
    "neo_core.core.vox": logging.INFO,
    "neo_core.core.brain": logging.INFO,
    "neo_core.core.memory_agent": logging.INFO,
    "neo_core.core.heartbeat": logging.INFO,
    "neo_core.core.guardian": logging.INFO,
    "neo_core.core.persona": logging.INFO,
    "neo_core.core.resilience": logging.WARNING,
    "neo_core.teams.worker": logging.INFO,
    "neo_core.teams.factory": logging.INFO,
    "neo_core.memory.store": logging.INFO,
    "neo_core.memory.learning": logging.INFO,
    "neo_core.memory.consolidator": logging.INFO,
    "neo_core.providers": logging.INFO,
}


# ─── Setup ───────────────────────────────────────────

_logging_configured = False


def setup_logging(
    level: int = DEFAULT_LOG_LEVEL,
    log_file: Optional[str] = DEFAULT_LOG_FILE,
    log_dir: Optional[Path] = None,
    console: bool = True,
    file_logging: bool = True,
) -> None:
    """
    Configure le logging pour l'ensemble du projet Neo Core.

    Args:
        level: Niveau de log global (default: INFO)
        log_file: Nom du fichier de log (default: neo.log)
        log_dir: Répertoire des logs (default: data/)
        console: Activer les logs console (default: True)
        file_logging: Activer les logs fichier (default: True)
    """
    global _logging_configured
    if _logging_configured:
        return

    root_logger = logging.getLogger("neo_core")
    root_logger.setLevel(level)

    # Supprimer les handlers existants
    root_logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter(CONSOLE_FORMAT, datefmt=CONSOLE_DATE_FORMAT)
        )
        root_logger.addHandler(console_handler)

    # File handler avec rotation
    if file_logging and log_file:
        _log_dir = log_dir or DEFAULT_LOG_DIR
        _log_dir.mkdir(parents=True, exist_ok=True)
        log_path = _log_dir / log_file

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(FILE_FORMAT, datefmt=FILE_DATE_FORMAT)
        )
        root_logger.addHandler(file_handler)

    # Appliquer les niveaux par module
    for module_name, module_level in MODULE_LEVELS.items():
        logging.getLogger(module_name).setLevel(module_level)

    # Réduire le bruit des dépendances
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)

    _logging_configured = True


def reset_logging() -> None:
    """Reset le logging (utile pour les tests)."""
    global _logging_configured
    root_logger = logging.getLogger("neo_core")
    root_logger.handlers.clear()
    _logging_configured = False


def get_logger(name: str) -> logging.Logger:
    """Raccourci pour obtenir un logger Neo Core."""
    return logging.getLogger(name)
