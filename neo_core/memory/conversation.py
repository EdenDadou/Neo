"""
Conversation Persistence System — SQLite-based conversation store
===================================================================

Système de persistance des conversations pour Neo Core.
Stocke les messages humain-assistant dans une base SQLite.

Responsabilités :
- Créer et gérer les sessions de conversation
- Enregistrer chaque tour (human/assistant)
- Récupérer l'historique avec pagination
- Exporter les conversations en JSON
- Gestion de contexte (context manager)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Un tour de conversation (human ou assistant)."""
    role: str          # "human" | "assistant"
    content: str
    timestamp: str     # ISO format
    turn_number: int


@dataclass
class ConversationSession:
    """Une session de conversation."""
    session_id: str    # UUID
    user_name: str
    created_at: str    # ISO format
    updated_at: str    # ISO format
    message_count: int


class ConversationStore:
    """
    Magasin de conversations basé sur SQLite.

    Tables :
    - sessions(session_id TEXT PK, user_name TEXT, created_at TEXT, updated_at TEXT, message_count INTEGER)
    - turns(id INTEGER PK AUTOINCREMENT, session_id TEXT FK, turn_number INTEGER, role TEXT, content TEXT, timestamp TEXT)
    """

    def __init__(self, db_path: Path):
        """Initialise le store avec un chemin DB."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Crée les tables si nécessaire."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Table sessions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0
                )
            """)

            # Table turns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id),
                    UNIQUE(session_id, turn_number)
                )
            """)

            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON turns(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON turns(timestamp)")

            conn.commit()
            logger.debug(f"Database initialized: {self.db_path}")

    def start_session(self, user_name: str = "User") -> ConversationSession:
        """Démarre une nouvelle session de conversation."""
        session_id = str(uuid4())
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (session_id, user_name, created_at, updated_at, message_count)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, user_name, now, now, 0))
            conn.commit()

        logger.debug(f"Started session: {session_id}")
        return ConversationSession(
            session_id=session_id,
            user_name=user_name,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def append_turn(self, session_id: str, role: str, content: str) -> ConversationTurn:
        """Ajoute un tour à une session."""
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Récupérer le numéro du prochain tour
            cursor.execute(
                "SELECT COALESCE(MAX(turn_number), 0) FROM turns WHERE session_id = ?",
                (session_id,)
            )
            result = cursor.fetchone()
            next_turn = (result[0] if result else 0) + 1

            # Insérer le turn
            cursor.execute("""
                INSERT INTO turns (session_id, turn_number, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, next_turn, role, content, now))

            # Mettre à jour la session
            cursor.execute("""
                UPDATE sessions
                SET updated_at = ?, message_count = message_count + 1
                WHERE session_id = ?
            """, (now, session_id))

            conn.commit()

        logger.debug(f"Appended {role} turn to session {session_id}")
        return ConversationTurn(
            role=role,
            content=content,
            timestamp=now,
            turn_number=next_turn,
        )

    def get_history(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ConversationTurn]:
        """Récupère l'historique avec pagination."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content, timestamp, turn_number
                FROM turns
                WHERE session_id = ?
                ORDER BY turn_number ASC
                LIMIT ? OFFSET ?
            """, (session_id, limit, offset))

            rows = cursor.fetchall()

        return [
            ConversationTurn(role=row[0], content=row[1], timestamp=row[2], turn_number=row[3])
            for row in rows
        ]

    def get_sessions(self, limit: int = 20) -> list[ConversationSession]:
        """Récupère les sessions récentes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, user_name, created_at, updated_at, message_count
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()

        return [
            ConversationSession(
                session_id=row[0],
                user_name=row[1],
                created_at=row[2],
                updated_at=row[3],
                message_count=row[4],
            )
            for row in rows
        ]

    def get_last_session(self) -> Optional[ConversationSession]:
        """Récupère la dernière session."""
        sessions = self.get_sessions(limit=1)
        return sessions[0] if sessions else None

    def get_session_by_id(self, session_id: str) -> Optional[ConversationSession]:
        """Récupère une session par ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, user_name, created_at, updated_at, message_count
                FROM sessions
                WHERE session_id = ?
            """, (session_id,))

            row = cursor.fetchone()

        if not row:
            return None

        return ConversationSession(
            session_id=row[0],
            user_name=row[1],
            created_at=row[2],
            updated_at=row[3],
            message_count=row[4],
        )

    def export_json(self, session_id: str) -> dict:
        """Exporte une session au format JSON."""
        session = self.get_session_by_id(session_id)
        if not session:
            return {}

        history = self.get_history(session_id, limit=10000)  # Tout charger

        return {
            "session": {
                "session_id": session.session_id,
                "user_name": session.user_name,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "message_count": session.message_count,
            },
            "turns": [
                {
                    "turn_number": turn.turn_number,
                    "role": turn.role,
                    "content": turn.content,
                    "timestamp": turn.timestamp,
                }
                for turn in history
            ],
        }

    def close(self) -> None:
        """Ferme les connexions (si nécessaire)."""
        # SQLite gère les connexions automatiquement avec context managers
        logger.debug("ConversationStore closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
