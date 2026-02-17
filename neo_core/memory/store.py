"""
Memory Store — Stockage Persistant
====================================
Double système de stockage :
- ChromaDB : stockage vectoriel pour la recherche sémantique
- SQLite : métadonnées structurées (timestamps, tags, sources, importance)

Les embeddings sont générés par ChromaDB (modèle par défaut all-MiniLM-L6-v2).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from neo_core.config import MemoryConfig

logger = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    """Un enregistrement complet en mémoire."""
    id: str
    content: str
    source: str = "conversation"
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


class MemoryStore:
    """
    Stockage persistant double couche : ChromaDB + SQLite.

    ChromaDB gère les embeddings et la recherche sémantique.
    SQLite gère les métadonnées structurées et les requêtes exactes.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._chroma_client = None
        self._collection = None
        self._db_conn: Optional[sqlite3.Connection] = None
        self._initialized = False
        # Cache d'embeddings pour éviter les doubles recherches sémantiques
        self._semantic_cache: dict[str, list] = {}  # query → results
        self._semantic_cache_hits: int = 0

    def initialize(self) -> None:
        """Initialise les deux systèmes de stockage."""
        # Crée le répertoire de stockage
        self.config.storage_path.mkdir(parents=True, exist_ok=True)

        self._init_sqlite()
        self._init_chromadb()
        self._initialized = True

    def _init_sqlite(self) -> None:
        """Initialise la base SQLite pour les métadonnées."""
        db_path = self.config.storage_path / "memory_meta.db"
        self._db_conn = sqlite3.connect(str(db_path))

        # Vérifier l'intégrité
        try:
            result = self._db_conn.execute("PRAGMA integrity_check").fetchone()
            if result and result[0] != "ok":
                logger.error("Memory DB integrity check FAILED: %s — recreating", result[0])
                self._db_conn.close()
                db_path.unlink(missing_ok=True)
                self._db_conn = sqlite3.connect(str(db_path))
        except sqlite3.DatabaseError as e:
            logger.error("Memory DB corrupted: %s — recreating", e)
            self._db_conn.close()
            db_path.unlink(missing_ok=True)
            self._db_conn = sqlite3.connect(str(db_path))

        self._db_conn.execute("PRAGMA journal_mode=WAL")
        self._db_conn.row_factory = sqlite3.Row

        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT DEFAULT 'conversation',
                tags TEXT DEFAULT '[]',
                importance REAL DEFAULT 0.5,
                timestamp TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self._db_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source)
        """)
        self._db_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)
        """)
        self._db_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)
        """)

        self._db_conn.commit()

        # Appliquer les migrations
        try:
            from neo_core.memory.migrations import run_migrations
            run_migrations(self._db_conn)
        except Exception as e:
            logger.debug("Migrations skipped: %s", e)

    def _init_chromadb(self) -> None:
        """Initialise ChromaDB pour le stockage vectoriel."""
        try:
            # Désactiver la télémétrie chromadb AVANT tout import/init
            # Corrige: "Failed to send telemetry event: capture() takes 1 positional argument but 3 were given"
            import os as _os
            _os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

            import chromadb
            from chromadb.config import Settings as _ChromaSettings

            chroma_path = self.config.storage_path / "chroma"
            self._chroma_client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=_ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name="neo_memory",
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            # Fallback sans ChromaDB — utilise uniquement SQLite
            logger.info("ChromaDB non installé — mémoire vectorielle désactivée")
            self._chroma_client = None
            self._collection = None
        except PermissionError as e:
            # chromadb/pydantic tries to read .env in cwd — may fail with wrong perms
            logger.warning("ChromaDB init failed (permission denied: %s) — mémoire vectorielle désactivée", e)
            self._chroma_client = None
            self._collection = None
        except Exception as e:
            # Any other error — degrade gracefully instead of crashing
            logger.warning("ChromaDB init failed (%s) — mémoire vectorielle désactivée", e)
            self._chroma_client = None
            self._collection = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def has_vector_search(self) -> bool:
        """Indique si la recherche vectorielle est disponible."""
        return self._collection is not None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper cleanup."""
        self.close()
        return False

    def store(self, content: str, source: str = "conversation",
              tags: list[str] | None = None, importance: float = 0.5,
              metadata: dict | None = None) -> str:
        """
        Stocke un souvenir dans les deux systèmes.
        Retourne l'ID du souvenir créé.
        """
        record_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        tags = tags or []
        metadata = metadata or {}

        # SQLite
        try:
            self._db_conn.execute(
                "INSERT INTO memories (id, content, source, tags, importance, timestamp, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (record_id, content, source, json.dumps(tags), importance, timestamp, json.dumps(metadata))
            )
            self._db_conn.commit()
        except (sqlite3.OperationalError, OSError) as e:
            err = str(e).lower()
            if "disk" in err or "no space" in err or "i/o" in err:
                logger.critical("DISK FULL — memory store write failed: %s", e)
                return record_id  # Graceful degradation: return ID but don't crash
            raise

        # ChromaDB (si disponible)
        if self._collection is not None:
            try:
                self._collection.add(
                    ids=[record_id],
                    documents=[content],
                    metadatas=[{
                        "source": source,
                        "tags": json.dumps(tags),
                        "importance": importance,
                        "timestamp": timestamp,
                    }],
                )
            except OSError as e:
                logger.critical("DISK FULL — ChromaDB write failed: %s", e)

        return record_id

    def clear_semantic_cache(self) -> None:
        """Vide le cache sémantique (à appeler entre les requêtes utilisateur)."""
        self._semantic_cache.clear()

    def search_semantic(self, query: str, n_results: int = 5,
                        min_importance: float = 0.0) -> list[MemoryRecord]:
        """
        Recherche sémantique via ChromaDB.
        Retourne les souvenirs les plus pertinents pour la requête.
        Utilise un cache par requête pour éviter les doubles embeddings.
        """
        if not self._collection or self._collection.count() == 0:
            return []

        # Cache hit — même query + même n_results
        cache_key = f"{query}::{n_results}"
        if cache_key in self._semantic_cache:
            self._semantic_cache_hits += 1
            cached = self._semantic_cache[cache_key]
            if min_importance > 0.0:
                return [r for r in cached if r.importance >= min_importance]
            return list(cached)

        n_results = min(n_results, self._collection.count())
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        records = []
        if results and results["ids"] and results["ids"][0]:
            record_ids = results["ids"][0]

            # Batch query : un seul SELECT IN au lieu de N requêtes individuelles
            placeholders = ",".join("?" * len(record_ids))
            rows = self._db_conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})",
                record_ids,
            ).fetchall()
            id_to_row = {row["id"]: row for row in rows}

            # Reconstituer dans l'ordre ChromaDB (pertinence décroissante)
            for record_id in record_ids:
                row = id_to_row.get(record_id)
                if row:
                    records.append(MemoryRecord(
                        id=row["id"],
                        content=row["content"],
                        source=row["source"],
                        tags=json.loads(row["tags"]),
                        importance=row["importance"],
                        timestamp=row["timestamp"],
                        metadata=json.loads(row["metadata"]),
                    ))

        # Stocker dans le cache AVANT filtrage par importance
        self._semantic_cache[cache_key] = records

        # Filtrer par importance si demandé
        if min_importance > 0.0:
            return [r for r in records if r.importance >= min_importance]
        return records

    def search_by_source(self, source: str, limit: int = 20) -> list[MemoryRecord]:
        """Recherche par source (conversation, system, agent, etc.)."""
        rows = self._db_conn.execute(
            "SELECT * FROM memories WHERE source = ? ORDER BY timestamp DESC LIMIT ?",
            (source, limit)
        ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def search_by_tags(self, tags: list[str], limit: int = 20) -> list[MemoryRecord]:
        """
        Recherche par tags.
        Optimisation v0.9.1 : utilise LIKE en SQL pour pré-filtrer,
        avec fallback Python pour les tags unicode (json.dumps encode en \\uXXXX).
        """
        if not tags:
            return []

        # Construire les patterns LIKE — inclure aussi la forme json-escaped
        like_conditions = []
        like_params: list = []
        for tag in tags:
            # Forme brute (si ensure_ascii=False a été utilisé)
            like_conditions.append("tags LIKE ?")
            like_params.append(f"%{tag}%")
            # Forme json-escaped (json.dumps par défaut encode les accents)
            escaped = json.dumps(tag)[1:-1]  # Enlever les guillemets
            if escaped != tag:
                like_conditions.append("tags LIKE ?")
                like_params.append(f"%{escaped}%")

        conditions = " OR ".join(like_conditions)
        like_params.append(limit * 3)  # Marge pour faux positifs

        rows = self._db_conn.execute(
            f"SELECT * FROM memories WHERE ({conditions}) ORDER BY timestamp DESC LIMIT ?",
            like_params,
        ).fetchall()

        # Vérification exacte en Python
        records = []
        for row in rows:
            row_tags = json.loads(row["tags"])
            if any(tag in row_tags for tag in tags):
                records.append(self._row_to_record(row))
                if len(records) >= limit:
                    break

        return records

    def get_recent(self, limit: int = 10) -> list[MemoryRecord]:
        """Retourne les souvenirs les plus récents."""
        rows = self._db_conn.execute(
            "SELECT * FROM memories ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_important(self, min_importance: float = 0.7, limit: int = 10) -> list[MemoryRecord]:
        """Retourne les souvenirs les plus importants."""
        rows = self._db_conn.execute(
            "SELECT * FROM memories WHERE importance >= ? ORDER BY importance DESC LIMIT ?",
            (min_importance, limit)
        ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def update_importance(self, record_id: str, importance: float) -> None:
        """Met à jour l'importance d'un souvenir."""
        self._db_conn.execute(
            "UPDATE memories SET importance = ? WHERE id = ?",
            (importance, record_id)
        )
        self._db_conn.commit()

        if self._collection is not None:
            # Met à jour aussi dans ChromaDB
            try:
                existing = self._collection.get(ids=[record_id])
                if existing and existing["metadatas"]:
                    meta = existing["metadatas"][0]
                    meta["importance"] = importance
                    self._collection.update(ids=[record_id], metadatas=[meta])
            except Exception as e:
                logger.debug("ChromaDB update importance failed: %s", e)

    def delete(self, record_id: str) -> None:
        """Supprime un souvenir."""
        self._db_conn.execute("DELETE FROM memories WHERE id = ?", (record_id,))
        self._db_conn.commit()

        if self._collection is not None:
            try:
                self._collection.delete(ids=[record_id])
            except Exception as e:
                logger.debug("ChromaDB delete failed: %s", e)

    def count(self) -> int:
        """Retourne le nombre total de souvenirs."""
        row = self._db_conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
        return row["cnt"] if row else 0

    def get_stats(self) -> dict:
        """Retourne des statistiques sur la mémoire."""
        total = self.count()
        sources = self._db_conn.execute(
            "SELECT source, COUNT(*) as cnt FROM memories GROUP BY source"
        ).fetchall()

        return {
            "total_entries": total,
            "has_vector_search": self.has_vector_search,
            "sources": {row["source"]: row["cnt"] for row in sources},
        }

    def close(self) -> None:
        """Ferme les connexions."""
        if self._db_conn:
            self._db_conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        """Convertit une row SQLite en MemoryRecord."""
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            source=row["source"],
            tags=json.loads(row["tags"]),
            importance=row["importance"],
            timestamp=row["timestamp"],
            metadata=json.loads(row["metadata"]),
        )
