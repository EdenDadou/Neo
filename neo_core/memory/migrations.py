"""
Neo Core — Migrations SQLite
===============================
Système de migration simple pour les bases SQLite de Neo.

Chaque migration est un tuple (version, description, sql).
Le schéma courant est stocké dans la table `schema_version`.

Usage :
    from neo_core.memory.migrations import run_migrations
    run_migrations(conn)  # Applique les migrations manquantes
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

# ─── Migrations ──────────────────────────────────────
# Chaque entrée : (version: int, description: str, sql: str)
# Les migrations sont cumulatives et idempotentes.

MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "Création table schema_version",
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
    ),
    (
        2,
        "Index sur memories.timestamp",
        """
        CREATE INDEX IF NOT EXISTS idx_memories_timestamp
        ON memories(timestamp);
        """,
    ),
    # Migration v3 removed: rate_limits table lives in brain.db, not memory_meta.db.
    # We keep v3 as a no-op to avoid re-applying on existing DBs that already recorded v3.
    (
        3,
        "No-op (rate_limits index moved to brain migrations)",
        """
        SELECT 1;
        """,
    ),
]


def get_current_version(conn: sqlite3.Connection) -> int:
    """Retourne la version courante du schéma (0 si jamais migré)."""
    try:
        cursor = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        )
        row = cursor.fetchone()
        return row[0] if row and row[0] else 0
    except sqlite3.OperationalError:
        # Table schema_version n'existe pas encore
        return 0


def run_migrations(conn: sqlite3.Connection) -> int:
    """
    Applique les migrations manquantes.

    Returns:
        Nombre de migrations appliquées.
    """
    current = get_current_version(conn)
    applied = 0

    for version, description, sql in MIGRATIONS:
        if version <= current:
            continue

        try:
            conn.executescript(sql)
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version, description) VALUES (?, ?)",
                (version, description),
            )
            conn.commit()
            applied += 1
            logger.info("Migration v%d applied: %s", version, description)
        except sqlite3.Error as e:
            logger.warning("Migration v%d failed (%s): %s — skipping", version, description, e)

    if applied:
        logger.info("Migrations: %d applied (now at v%d)", applied, get_current_version(conn))

    return applied
