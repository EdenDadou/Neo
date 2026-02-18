"""
Memory Store — Stockage Persistant
====================================
Double système de stockage :
- FAISS : index vectoriel pour la recherche sémantique (remplace ChromaDB v0.9.3)
- SQLite : métadonnées structurées (timestamps, tags, sources, importance)

Les embeddings sont générés par sentence-transformers (all-MiniLM-L6-v2, dim=384).
L'index FAISS utilise IndexFlatIP (produit scalaire = cosine sur vecteurs normalisés).
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

import numpy as np

from neo_core.config import MemoryConfig

logger = logging.getLogger(__name__)

# Dimension du modèle all-MiniLM-L6-v2
EMBEDDING_DIM = 384
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Singleton process-level pour le modèle d'embedding (évite de recharger ~2s à chaque init)
_EMBEDDING_MODEL_CACHE = None


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
    Stockage persistant double couche : FAISS + SQLite.

    FAISS gère les embeddings et la recherche sémantique.
    SQLite gère les métadonnées structurées et les requêtes exactes.

    Persistence : l'index FAISS est sauvegardé sur disque après chaque
    modification (write/delete). L'index est rechargé au démarrage.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._faiss_index = None
        self._embedding_model = None
        self._using_fallback_embeddings = False
        self._id_map: list[str] = []  # position FAISS → memory ID
        self._db_conn: Optional[sqlite3.Connection] = None
        self._initialized = False
        # Cache d'embeddings pour éviter les doubles recherches sémantiques
        self._semantic_cache: dict[str, list] = {}  # query → results
        self._semantic_cache_hits: int = 0
        self._needs_rebuild: bool = False

    def initialize(self) -> None:
        """Initialise les deux systèmes de stockage."""
        # Crée le répertoire de stockage
        self.config.storage_path.mkdir(parents=True, exist_ok=True)

        self._init_sqlite()
        self._init_faiss()
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

    def _init_faiss(self) -> None:
        """
        Initialise FAISS pour le stockage vectoriel.

        Charge le modèle d'embedding (all-MiniLM-L6-v2) et l'index FAISS.
        Si un index persisté existe, le recharge. Sinon, crée un index vide
        et le peuple à partir des données SQLite existantes.
        """
        try:
            import faiss

            index_path = self.config.storage_path / "faiss_index.bin"
            id_map_path = self.config.storage_path / "faiss_id_map.json"

            # Charger le modèle d'embedding (lazy — une seule fois par process)
            self._load_embedding_model()

            # Recharger l'index persisté s'il existe
            if index_path.exists() and id_map_path.exists():
                self._faiss_index = faiss.read_index(str(index_path))
                self._id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
                logger.info(
                    "FAISS index loaded: %d vectors (dim=%d)",
                    self._faiss_index.ntotal, EMBEDDING_DIM,
                )
            else:
                # Index cosine = IndexFlatIP sur vecteurs L2-normalisés
                self._faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
                self._id_map = []
                self._needs_rebuild = True  # Rebuild lazy au premier search

        except ImportError:
            logger.info("FAISS non installé — mémoire vectorielle désactivée")
            self._faiss_index = None
        except Exception as e:
            logger.warning("FAISS init failed (%s) — mémoire vectorielle désactivée", e)
            self._faiss_index = None

    def _load_embedding_model(self) -> None:
        """
        Charge le modèle d'embedding sentence-transformers.

        Singleton process-level : chargé une seule fois, réutilisé par toutes
        les instances de MemoryStore. Temps de chargement : ~200ms au premier appel.

        Fallback : si sentence-transformers échoue (HuggingFace 429, pas de réseau),
        utilise un HashingVectorizer local (pas de téléchargement nécessaire).
        Moins précis mais toujours fonctionnel.
        """
        global _EMBEDDING_MODEL_CACHE

        if self._embedding_model is not None:
            return

        if _EMBEDDING_MODEL_CACHE is not None:
            self._embedding_model = _EMBEDDING_MODEL_CACHE
            self._using_fallback_embeddings = getattr(
                _EMBEDDING_MODEL_CACHE, "_is_fallback", False
            )
            return

        # Tenter sentence-transformers (meilleure qualité)
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL_CACHE = SentenceTransformer(EMBEDDING_MODEL)
            self._embedding_model = _EMBEDDING_MODEL_CACHE
            self._using_fallback_embeddings = False
            logger.info("Embedding model loaded: %s", EMBEDDING_MODEL)
            return
        except ImportError:
            logger.warning("sentence-transformers non installé")
        except Exception as e:
            logger.warning("sentence-transformers échoué (%s) — fallback local", e)

        # Fallback : HashingVectorizer (sklearn) — pas de téléchargement
        try:
            from sklearn.feature_extraction.text import HashingVectorizer
            vectorizer = HashingVectorizer(
                n_features=EMBEDDING_DIM,
                alternate_sign=False,
                norm="l2",
            )
            vectorizer._is_fallback = True
            _EMBEDDING_MODEL_CACHE = vectorizer
            self._embedding_model = vectorizer
            self._using_fallback_embeddings = True
            logger.info("Fallback embedding loaded: HashingVectorizer (dim=%d)", EMBEDDING_DIM)
            return
        except ImportError:
            logger.warning("sklearn non disponible — embeddings désactivés")

        self._embedding_model = None
        self._using_fallback_embeddings = False

    def _embed(self, texts: list[str]) -> np.ndarray:
        """
        Génère des embeddings normalisés pour une liste de textes.

        Retourne un array numpy (n_texts, 384) normalisé L2.
        La normalisation permet d'utiliser le produit scalaire (IP)
        comme distance cosine dans FAISS.

        Supporte deux backends :
        - sentence-transformers (qualité sémantique)
        - HashingVectorizer fallback (bag-of-words, pas de téléchargement)
        """
        if not self._embedding_model:
            return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

        if self._using_fallback_embeddings:
            # HashingVectorizer retourne une sparse matrix → dense + normalize
            sparse = self._embedding_model.transform(texts)
            embeddings = sparse.toarray().astype(np.float32)
            # Normaliser L2 (certaines lignes peuvent être 0)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms
            return embeddings

        embeddings = self._embedding_model.encode(
            texts,
            normalize_embeddings=True,  # Cosine similarity via IP
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _rebuild_index_from_sqlite(self) -> None:
        """
        Rebuilde l'index FAISS à partir des données SQLite existantes.

        Utilisé lors de la migration ChromaDB → FAISS ou si l'index
        est corrompu/manquant.
        """
        if not self._faiss_index or not self._embedding_model:
            return

        rows = self._db_conn.execute(
            "SELECT id, content FROM memories ORDER BY timestamp ASC"
        ).fetchall()

        if not rows:
            return

        ids = [row["id"] for row in rows]
        contents = [row["content"] for row in rows]

        # Embedder par batch de 256
        batch_size = 256
        all_embeddings = []
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            emb = self._embed(batch)
            all_embeddings.append(emb)

        embeddings = np.vstack(all_embeddings)
        self._faiss_index.add(embeddings)
        self._id_map = ids

        self._save_faiss_index()
        logger.info("FAISS index rebuilt from SQLite: %d vectors", len(ids))

    def _save_faiss_index(self) -> None:
        """Persiste l'index FAISS et le mapping ID sur disque."""
        if not self._faiss_index:
            return

        try:
            import faiss
            index_path = self.config.storage_path / "faiss_index.bin"
            id_map_path = self.config.storage_path / "faiss_id_map.json"

            faiss.write_index(self._faiss_index, str(index_path))
            id_map_path.write_text(
                json.dumps(self._id_map), encoding="utf-8"
            )
        except Exception as e:
            logger.debug("Failed to save FAISS index: %s", e)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def has_vector_search(self) -> bool:
        """Indique si la recherche vectorielle est disponible."""
        return self._faiss_index is not None and self._embedding_model is not None

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

        # FAISS (si disponible)
        if self._faiss_index is not None and self._embedding_model is not None:
            try:
                embedding = self._embed([content])
                self._faiss_index.add(embedding)
                self._id_map.append(record_id)
                self._save_faiss_index()
            except OSError as e:
                logger.critical("DISK FULL — FAISS write failed: %s", e)

        return record_id

    def clear_semantic_cache(self) -> None:
        """Vide le cache sémantique (à appeler entre les requêtes utilisateur)."""
        self._semantic_cache.clear()

    def search_semantic(self, query: str, n_results: int = 5,
                        min_importance: float = 0.0) -> list[MemoryRecord]:
        """
        Recherche sémantique via FAISS.
        Retourne les souvenirs les plus pertinents pour la requête.
        Utilise un cache par requête pour éviter les doubles embeddings.
        """
        if not self._faiss_index or not self._embedding_model:
            return []

        # Lazy rebuild: construire l'index au premier search (pas au boot)
        if self._needs_rebuild:
            self._needs_rebuild = False
            self._rebuild_index_from_sqlite()

        if self._faiss_index.ntotal == 0:
            return []

        # Cache hit — même query + même n_results
        cache_key = f"{query}::{n_results}"
        if cache_key in self._semantic_cache:
            self._semantic_cache_hits += 1
            cached = self._semantic_cache[cache_key]
            if min_importance > 0.0:
                return [r for r in cached if r.importance >= min_importance]
            return list(cached)

        n_results = min(n_results, self._faiss_index.ntotal)
        query_embedding = self._embed([query])

        # Recherche FAISS — retourne les n_results plus proches
        scores, indices = self._faiss_index.search(query_embedding, n_results)

        # Mapper les indices FAISS vers les IDs mémoire
        record_ids = []
        for idx in indices[0]:
            if 0 <= idx < len(self._id_map):
                record_ids.append(self._id_map[idx])

        if not record_ids:
            return []

        # Batch query SQLite
        placeholders = ",".join("?" * len(record_ids))
        rows = self._db_conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            record_ids,
        ).fetchall()
        id_to_row = {row["id"]: row for row in rows}

        # Reconstituer dans l'ordre FAISS (pertinence décroissante)
        records = []
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

    def delete(self, record_id: str) -> None:
        """
        Supprime un souvenir de SQLite.

        Note : FAISS IndexFlatIP ne supporte pas la suppression individuelle.
        L'entrée reste dans l'index mais sera ignorée car absente de SQLite
        lors de la reconstitution des résultats. L'index est rebuildd
        périodiquement par le heartbeat pour nettoyer les entrées orphelines.
        """
        self._db_conn.execute("DELETE FROM memories WHERE id = ?", (record_id,))
        self._db_conn.commit()

        # Marquer comme supprimé dans le id_map (sera nettoyé au rebuild)
        if record_id in self._id_map:
            idx = self._id_map.index(record_id)
            self._id_map[idx] = "__deleted__"

    def rebuild_index(self) -> None:
        """
        Rebuilde l'index FAISS en éliminant les entrées supprimées.

        Appelé périodiquement par le heartbeat ou manuellement.
        """
        if not self._faiss_index or not self._embedding_model:
            return

        import faiss

        deleted_count = self._id_map.count("__deleted__")
        if deleted_count == 0:
            return

        # Recréer l'index propre
        self._faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._id_map = []
        self._rebuild_index_from_sqlite()
        logger.info("FAISS index rebuilt: removed %d deleted entries", deleted_count)

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

        faiss_stats = {}
        if self._faiss_index:
            faiss_stats = {
                "faiss_vectors": self._faiss_index.ntotal,
                "faiss_deleted": self._id_map.count("__deleted__") if self._id_map else 0,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": EMBEDDING_DIM,
            }

        return {
            "total_entries": total,
            "has_vector_search": self.has_vector_search,
            "sources": {row["source"]: row["cnt"] for row in sources},
            "semantic_cache_hits": self._semantic_cache_hits,
            **faiss_stats,
        }

    def close(self) -> None:
        """Ferme les connexions."""
        # Sauvegarder l'index FAISS avant de fermer
        self._save_faiss_index()
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
