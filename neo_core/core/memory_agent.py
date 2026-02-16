"""
Memory — Agent Bibliothécaire (Stub Étape 1)
==============================================
Hippocampe et système de consolidation des connaissances.

Ce module est un stub minimal pour l'étape 1.
Il sera complètement développé à l'étape 2 avec :
- Stockage vectoriel (ChromaDB)
- SQLite pour métadonnées
- Consolidation et synthèse
- Injection de contexte enrichi

Responsabilités futures :
- Organiser, nettoyer et synthétiser la mémoire
- Archiver succès et échecs des agents
- Référencer les nouvelles compétences
- Injecter du contexte pertinent
- Mémoire long terme (jusqu'à 10 ans)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from neo_core.config import NeoConfig, default_config


@dataclass
class MemoryEntry:
    """Une entrée en mémoire."""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "conversation"
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5  # 0.0 à 1.0


@dataclass
class MemoryAgent:
    """
    Agent Memory — Bibliothécaire du système Neo Core.

    Stub minimal pour l'étape 1.
    Stocke les conversations en mémoire volatile.
    Sera remplacé par un système persistant en étape 2.
    """
    config: NeoConfig = field(default_factory=lambda: default_config)
    _short_term: list[MemoryEntry] = field(default_factory=list)
    _initialized: bool = False

    def initialize(self) -> None:
        """Initialise le système mémoire."""
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def store(self, content: str, source: str = "conversation",
              tags: list[str] | None = None, importance: float = 0.5) -> None:
        """
        Stocke une entrée en mémoire court terme.
        Sera étendu avec ChromaDB en étape 2.
        """
        entry = MemoryEntry(
            content=content,
            source=source,
            tags=tags or [],
            importance=importance,
        )
        self._short_term.append(entry)

    def get_context(self, query: str) -> str:
        """
        Retourne le contexte pertinent pour une requête.
        En étape 1 : retourne les dernières entrées.
        En étape 2 : recherche vectorielle sémantique.
        """
        if not self._short_term:
            return "Aucun contexte mémoire disponible."

        # Retourne les 5 dernières entrées comme contexte
        recent = self._short_term[-5:]
        context_parts = [f"- [{e.source}] {e.content}" for e in recent]
        return "Contexte récent :\n" + "\n".join(context_parts)

    def get_stats(self) -> dict:
        """Retourne des statistiques sur la mémoire."""
        return {
            "total_entries": len(self._short_term),
            "initialized": self._initialized,
            "sources": list(set(e.source for e in self._short_term)),
        }

    def clear(self) -> None:
        """Vide la mémoire court terme."""
        self._short_term.clear()
