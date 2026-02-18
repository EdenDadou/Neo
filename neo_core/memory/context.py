"""
Context Engine — Injection de Contexte
========================================
Moteur d'injection de contexte intelligent.

Responsabilités :
- Récupérer les souvenirs pertinents pour une requête
- Construire un bloc de contexte structuré
- Prioriser par pertinence sémantique et importance
- Respecter la limite de tokens du contexte
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from neo_core.config import MemoryConfig
from neo_core.memory.store import MemoryStore, MemoryRecord


@dataclass
class ContextBlock:
    """Bloc de contexte structuré pour injection dans les prompts."""
    relevant_memories: list[MemoryRecord] = field(default_factory=list)
    important_memories: list[MemoryRecord] = field(default_factory=list)
    recent_memories: list[MemoryRecord] = field(default_factory=list)
    total_tokens_estimate: int = 0

    def to_string(self) -> str:
        """Formate le contexte en texte pour injection dans un prompt."""
        sections = []

        if self.relevant_memories:
            sections.append("=== Souvenirs pertinents ===")
            for mem in self.relevant_memories:
                tag_str = f" [{', '.join(mem.tags)}]" if mem.tags else ""
                sections.append(f"- [{mem.source}]{tag_str} {mem.content}")

        if self.important_memories:
            # Évite les doublons avec les relevant
            relevant_ids = {m.id for m in self.relevant_memories}
            unique_important = [m for m in self.important_memories if m.id not in relevant_ids]
            if unique_important:
                sections.append("\n=== Informations importantes ===")
                for mem in unique_important:
                    sections.append(f"- [{mem.source}] {mem.content}")

        if self.recent_memories:
            # Évite les doublons
            seen_ids = {m.id for m in self.relevant_memories + self.important_memories}
            unique_recent = [m for m in self.recent_memories if m.id not in seen_ids]
            if unique_recent:
                sections.append("\n=== Contexte récent ===")
                for mem in unique_recent:
                    sections.append(f"- {mem.content}")

        if not sections:
            return "Aucun contexte mémoire disponible."

        return "\n".join(sections)

    @property
    def is_empty(self) -> bool:
        return not (self.relevant_memories or self.important_memories or self.recent_memories)


class ContextEngine:
    """
    Moteur d'injection de contexte.

    Combine recherche sémantique, importance et récence
    pour construire le meilleur contexte possible pour Brain.
    """

    def __init__(self, store: MemoryStore, config: MemoryConfig):
        self.store = store
        self.config = config

    def build_context(self, query: str) -> ContextBlock:
        """
        Construit un bloc de contexte pour une requête.

        Stratégie :
        1. Recherche sémantique pour les souvenirs pertinents
        2. Récupère les souvenirs importants (toujours inclus)
        3. Ajoute les souvenirs récents pour le contexte conversationnel
        4. Tronque si nécessaire pour respecter la limite de tokens
        """
        block = ContextBlock()

        # 1. Recherche sémantique
        if self.store.has_vector_search:
            block.relevant_memories = self.store.search_semantic(
                query,
                n_results=self.config.max_results,
            )

        # 2. Souvenirs importants (max 2 — optimisation perf)
        block.important_memories = self.store.get_important(
            min_importance=0.7,
            limit=2,
        )

        # 3. Souvenirs récents (max 2 — optimisation perf)
        block.recent_memories = self.store.get_recent(limit=2)

        # 4. Estimation des tokens et troncature
        block.total_tokens_estimate = self._estimate_tokens(block)
        if block.total_tokens_estimate > self.config.max_context_tokens:
            block = self._truncate(block)

        # 5. Marquer les souvenirs retournés comme "accessed"
        # pour la promotion par le Consolidator (promote_important)
        accessed_ids = [
            m.id for m in block.relevant_memories + block.important_memories
            if m.id
        ]
        if accessed_ids:
            try:
                self.store.mark_accessed(accessed_ids)
            except Exception:
                pass  # Non-critique, ne pas bloquer la réponse

        return block

    def store_conversation_turn(self, user_message: str, ai_response: str) -> None:
        """
        Stocke un échange conversationnel en mémoire.
        Analyse l'importance automatiquement.
        """
        # Stocke le message utilisateur
        importance = self._estimate_importance(user_message)
        self.store.store(
            content=f"[Utilisateur] {user_message}",
            source="conversation",
            tags=self._extract_tags(user_message),
            importance=importance,
        )

        # Stocke la réponse (importance légèrement plus basse)
        self.store.store(
            content=f"[Assistant] {ai_response[:500]}",
            source="conversation",
            tags=["response"],
            importance=max(0.3, importance - 0.1),
        )

    def _estimate_importance(self, text: str) -> float:
        """
        Estime l'importance d'un texte.
        Heuristique basée sur des indicateurs de contenu.
        """
        importance = 0.5

        # Indicateurs d'information personnelle
        personal_keywords = [
            "je m'appelle", "mon nom", "j'habite", "je travaille",
            "ma profession", "mon email", "mon numéro",
            "je suis", "mon entreprise", "ma société",
        ]
        if any(kw in text.lower() for kw in personal_keywords):
            importance = 0.9

        # Indicateurs de préférence
        pref_keywords = [
            "je préfère", "j'aime", "je n'aime pas", "je veux",
            "mon favori", "toujours", "jamais",
        ]
        if any(kw in text.lower() for kw in pref_keywords):
            importance = max(importance, 0.8)

        # Indicateurs de tâche/projet
        task_keywords = [
            "projet", "objectif", "deadline", "priorité",
            "important", "urgent", "rappelle",
        ]
        if any(kw in text.lower() for kw in task_keywords):
            importance = max(importance, 0.7)

        # Questions courtes = moins important
        if len(text.split()) < 5 and "?" in text:
            importance = min(importance, 0.3)

        return importance

    def _extract_tags(self, text: str) -> list[str]:
        """Extrait des tags automatiques depuis le texte."""
        tags = []

        tag_mapping = {
            "identité": ["je m'appelle", "mon nom", "je suis"],
            "préférence": ["je préfère", "j'aime", "je n'aime pas"],
            "projet": ["projet", "développement", "code"],
            "tâche": ["faire", "créer", "implémenter", "corriger"],
            "question": ["?", "comment", "pourquoi", "quand"],
        }

        text_lower = text.lower()
        for tag, keywords in tag_mapping.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(tag)

        return tags

    def _estimate_tokens(self, block: ContextBlock) -> int:
        """Estime le nombre de tokens d'un bloc de contexte (~4 chars/token)."""
        text = block.to_string()
        return len(text) // 4

    def _truncate(self, block: ContextBlock) -> ContextBlock:
        """Tronque le contexte pour respecter la limite de tokens."""
        # Priorité : relevant > important > recent
        # On réduit d'abord les récents, puis les importants
        while (self._estimate_tokens(block) > self.config.max_context_tokens
               and block.recent_memories):
            block.recent_memories.pop()

        while (self._estimate_tokens(block) > self.config.max_context_tokens
               and len(block.relevant_memories) > 2):
            block.relevant_memories.pop()

        return block
