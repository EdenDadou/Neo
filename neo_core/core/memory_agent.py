"""
Memory — Agent Bibliothécaire (Étape 2 — Complet)
====================================================
Hippocampe et système de consolidation des connaissances.

Agent LangChain complet qui pilote le module memory/ :
- MemoryStore : stockage persistant (ChromaDB + SQLite)
- ContextEngine : injection de contexte intelligent
- MemoryConsolidator : nettoyage et synthèse

Responsabilités :
- Organiser, nettoyer et synthétiser la mémoire
- Archiver succès et échecs des agents
- Référencer les nouvelles compétences
- Injecter du contexte pertinent
- Mémoire long terme persistante
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from neo_core.config import NeoConfig, default_config
from neo_core.memory.store import MemoryStore, MemoryRecord
from neo_core.memory.context import ContextEngine, ContextBlock
from neo_core.memory.consolidator import MemoryConsolidator


@dataclass
class MemoryAgent:
    """
    Agent Memory — Bibliothécaire du système Neo Core.

    Pilote le système mémoire complet :
    - Stockage persistant (ChromaDB + SQLite)
    - Injection de contexte intelligent
    - Consolidation périodique
    """
    config: NeoConfig = field(default_factory=lambda: default_config)
    _store: Optional[MemoryStore] = field(default=None, init=False)
    _context_engine: Optional[ContextEngine] = field(default=None, init=False)
    _consolidator: Optional[MemoryConsolidator] = field(default=None, init=False)
    _initialized: bool = field(default=False, init=False)
    _turn_count: int = field(default=0, init=False)
    _consolidation_interval: int = 50  # Consolide tous les N tours

    def initialize(self) -> None:
        """Initialise le système mémoire complet."""
        self._store = MemoryStore(self.config.memory)
        self._store.initialize()

        self._context_engine = ContextEngine(self._store, self.config.memory)
        self._consolidator = MemoryConsolidator(self._store, self.config)

        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def store(self) -> MemoryStore:
        """Accès direct au store (pour les tests et usages avancés)."""
        return self._store

    def store_memory(self, content: str, source: str = "conversation",
                     tags: list[str] | None = None, importance: float = 0.5,
                     metadata: dict | None = None) -> str:
        """
        Stocke un souvenir en mémoire persistante.
        Retourne l'ID du souvenir.
        """
        if not self._initialized:
            raise RuntimeError("Memory n'est pas initialisé. Appelez initialize() d'abord.")

        return self._store.store(
            content=content,
            source=source,
            tags=tags,
            importance=importance,
            metadata=metadata,
        )

    def get_context(self, query: str) -> str:
        """
        Retourne le contexte pertinent pour une requête sous forme de texte.
        Interface principale utilisée par Brain.
        """
        if not self._initialized:
            return "Aucun contexte mémoire disponible."

        block = self._context_engine.build_context(query)
        return block.to_string()

    def get_context_block(self, query: str) -> ContextBlock:
        """
        Retourne le contexte sous forme structurée (ContextBlock).
        Pour les usages avancés nécessitant un accès aux records individuels.
        """
        if not self._initialized:
            return ContextBlock()

        return self._context_engine.build_context(query)

    def on_conversation_turn(self, user_message: str, ai_response: str) -> None:
        """
        Appelé après chaque échange conversationnel.
        Stocke l'échange et déclenche la consolidation si nécessaire.
        """
        if not self._initialized:
            return

        self._context_engine.store_conversation_turn(user_message, ai_response)

        self._turn_count += 1
        if self._turn_count % self._consolidation_interval == 0:
            self.consolidate()

    def consolidate(self) -> dict:
        """
        Lance une consolidation complète de la mémoire.
        Retourne un rapport de consolidation.
        """
        if not self._initialized:
            return {}

        report = self._consolidator.full_consolidation()
        return {
            "entries_before": report.entries_before,
            "entries_after": report.entries_after,
            "deleted": report.entries_deleted,
            "merged": report.entries_merged,
            "promoted": report.entries_promoted,
        }

    def search(self, query: str, n_results: int = 5) -> list[MemoryRecord]:
        """Recherche sémantique dans la mémoire."""
        if not self._initialized:
            return []
        return self._store.search_semantic(query, n_results=n_results)

    def get_stats(self) -> dict:
        """Retourne des statistiques sur la mémoire."""
        if not self._initialized:
            return {
                "total_entries": 0,
                "initialized": False,
                "has_vector_search": False,
            }

        store_stats = self._store.get_stats()
        return {
            **store_stats,
            "initialized": True,
            "turn_count": self._turn_count,
            "next_consolidation_in": self._consolidation_interval - (self._turn_count % self._consolidation_interval),
        }

    def clear(self) -> None:
        """Vide toute la mémoire. À utiliser avec précaution."""
        if self._initialized and self._store:
            # Récupère tous les records et les supprime un par un
            records = self._store.get_recent(limit=10000)
            for record in records:
                self._store.delete(record.id)

    def close(self) -> None:
        """Ferme proprement les connexions."""
        if self._store:
            self._store.close()
