"""
Neo Core — Feature Flags
==========================
Système de feature flags pour contrôler l'activation des fonctionnalités.

Permet :
- Kill switch d'urgence (désactiver une feature en prod)
- Rollout progressif (activer par config)
- A/B testing (via overrides)

Fichier de config : data/feature_flags.json
Si absent → toutes les features sont activées par défaut.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Définition des features ──────────────────────────────

# Chaque feature a un nom, une description, et un état par défaut.
# Les features sont activées par défaut sauf indication contraire.
_FEATURE_DEFAULTS: dict[str, bool] = {
    # Level 1 — Auto-tuning des hyperparamètres
    "auto_tuning": True,
    # Level 3 — Self-patching comportemental
    "self_patching": True,
    # Level 4 — Création autonome d'outils
    "tool_generation": True,
    # Working Memory — scratchpad contextuel
    "working_memory": True,
    # Vox — reformulation LLM des messages
    "vox_reformat": True,
    # Brain — consultation du LearningEngine
    "learning_loop": True,
    # Heartbeat — cycle autonome
    "heartbeat": True,
    # PersonaEngine — empathie et profil utilisateur
    "persona_engine": True,
    # Telegram — notifications bot
    "telegram_notifications": True,
}

# ─── Singleton ────────────────────────────────────────────

_overrides: dict[str, bool] = {}
_loaded = False
_config_path: Optional[Path] = None


def _ensure_loaded() -> None:
    """Charge les overrides depuis le fichier de config (lazy, une seule fois)."""
    global _loaded, _overrides, _config_path
    if _loaded:
        return
    _loaded = True

    # Trouver le chemin du fichier de config
    if _config_path is None:
        _config_path = Path(__file__).resolve().parent.parent / "data" / "feature_flags.json"

    if not _config_path.exists():
        return

    try:
        data = json.loads(_config_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _overrides = {k: bool(v) for k, v in data.items() if k in _FEATURE_DEFAULTS}
            logger.info("Feature flags loaded: %d override(s)", len(_overrides))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load feature flags: %s — using defaults", e)


# ─── API publique ─────────────────────────────────────────


def feature_enabled(name: str) -> bool:
    """
    Vérifie si une feature est activée.

    Priorité :
    1. Override runtime (set_feature)
    2. Override fichier (data/feature_flags.json)
    3. Valeur par défaut (_FEATURE_DEFAULTS)

    Retourne True si la feature n'est pas définie (fail-open).
    """
    _ensure_loaded()
    if name in _overrides:
        return _overrides[name]
    return _FEATURE_DEFAULTS.get(name, True)


def set_feature(name: str, enabled: bool) -> None:
    """Override runtime d'une feature (en mémoire uniquement)."""
    _overrides[name] = enabled
    logger.info("Feature '%s' %s (runtime override)", name, "enabled" if enabled else "disabled")


def save_features(data_dir: Path | None = None) -> None:
    """Persiste les overrides actuels sur disque."""
    path = data_dir / "feature_flags.json" if data_dir else _config_path
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_overrides, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Feature flags saved to %s", path)
    except OSError as e:
        logger.warning("Failed to save feature flags: %s", e)


def get_all_features() -> dict[str, bool]:
    """Retourne l'état de toutes les features."""
    _ensure_loaded()
    return {name: feature_enabled(name) for name in _FEATURE_DEFAULTS}


def reset_features() -> None:
    """Réinitialise les overrides runtime (pour les tests)."""
    global _overrides, _loaded
    _overrides = {}
    _loaded = False


def set_config_path(path: Path) -> None:
    """Définit le chemin du fichier de config (pour les tests)."""
    global _config_path, _loaded
    _config_path = path
    _loaded = False
