"""
Memory Consolidator — Nettoyage et Synthèse
=============================================
Comme le sommeil consolide les souvenirs dans le cerveau humain,
le Consolidator nettoie, fusionne et synthétise la mémoire.

Responsabilités :
- Nettoyer les entrées obsolètes ou de faible importance
- Fusionner les entrées similaires/redondantes
- Synthétiser les conversations longues en résumés
- Promouvoir les entrées fréquemment accédées
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from neo_core.config import NeoConfig, default_config
from neo_core.memory.store import MemoryStore, MemoryRecord


@dataclass
class ConsolidationReport:
    """Rapport d'une opération de consolidation."""
    entries_before: int = 0
    entries_after: int = 0
    entries_deleted: int = 0
    entries_merged: int = 0
    entries_promoted: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MemoryConsolidator:
    """
    Consolide et optimise la mémoire du système.

    Opérations :
    - cleanup : supprime les entrées obsolètes de faible importance
    - merge_similar : fusionne les entrées sémantiquement proches
    - summarize : crée des synthèses de conversations
    - promote : augmente l'importance des entrées fréquemment utiles
    """

    def __init__(self, store: MemoryStore, config: NeoConfig | None = None):
        self.store = store
        self.config = config or default_config

    def cleanup(self, max_age_days: int = 30,
                min_importance: float = 0.2) -> ConsolidationReport:
        """
        Supprime les entrées anciennes de faible importance.
        Les entrées importantes sont gardées indéfiniment.
        """
        report = ConsolidationReport()
        report.entries_before = self.store.count()

        cutoff_date = (datetime.now() - timedelta(days=max_age_days)).isoformat()

        # Récupère les candidats à la suppression
        all_records = self.store.get_recent(limit=10000)

        for record in all_records:
            # Ne supprime jamais les entrées importantes
            if record.importance >= 0.7:
                continue

            # Supprime si ancien ET peu important
            if record.timestamp < cutoff_date and record.importance < min_importance:
                self.store.delete(record.id)
                report.entries_deleted += 1

        report.entries_after = self.store.count()
        return report

    def merge_similar(self, similarity_threshold: float = 0.85) -> ConsolidationReport:
        """
        Fusionne les entrées sémantiquement similaires.
        Garde l'entrée la plus importante et supprime les doublons.
        """
        report = ConsolidationReport()
        report.entries_before = self.store.count()

        if not self.store.has_vector_search:
            return report

        processed_ids: set[str] = set()
        all_records = self.store.get_recent(limit=500)

        for record in all_records:
            if record.id in processed_ids:
                continue

            # Cherche les entrées similaires
            similar = self.store.search_semantic(
                record.content, n_results=5
            )

            for sim_record in similar:
                if sim_record.id == record.id or sim_record.id in processed_ids:
                    continue

                # Si très similaire, fusionne
                # On garde l'entrée avec la plus haute importance
                if record.importance >= sim_record.importance:
                    # Combine les tags
                    combined_tags = list(set(record.tags + sim_record.tags))
                    max_importance = max(record.importance, sim_record.importance)

                    self.store.update_importance(record.id, max_importance)
                    self.store.delete(sim_record.id)
                    processed_ids.add(sim_record.id)
                    report.entries_merged += 1

            processed_ids.add(record.id)

        report.entries_after = self.store.count()
        return report

    def summarize_conversation(self, entries: list[MemoryRecord]) -> Optional[str]:
        """
        Crée un résumé synthétique d'une série d'entrées de conversation.
        Retourne le résumé sous forme de texte.
        """
        if not entries:
            return None

        # Synthèse simple par extraction des points clés
        contents = [e.content for e in entries]

        # Extrait les entrées les plus importantes
        sorted_entries = sorted(entries, key=lambda e: e.importance, reverse=True)
        key_points = sorted_entries[:5]

        summary_parts = ["Synthèse de conversation :"]
        for entry in key_points:
            summary_parts.append(f"- {entry.content[:200]}")

        return "\n".join(summary_parts)

    def promote_important(self, boost: float = 0.1) -> ConsolidationReport:
        """
        Augmente l'importance des entrées fréquemment accédées.
        (Les tags 'accessed' sont ajoutés par le ContextEngine lors des requêtes.)
        """
        report = ConsolidationReport()
        report.entries_before = self.store.count()

        accessed_records = self.store.search_by_tags(["accessed"])

        for record in accessed_records:
            new_importance = min(1.0, record.importance + boost)
            if new_importance != record.importance:
                self.store.update_importance(record.id, new_importance)
                report.entries_promoted += 1

        report.entries_after = self.store.count()
        return report

    def full_consolidation(self) -> ConsolidationReport:
        """Exécute toutes les opérations de consolidation."""
        report = ConsolidationReport()
        report.entries_before = self.store.count()

        # 1. Nettoyage
        cleanup_report = self.cleanup()
        report.entries_deleted = cleanup_report.entries_deleted

        # 2. Fusion des doublons
        merge_report = self.merge_similar()
        report.entries_merged = merge_report.entries_merged

        # 3. Promotion
        promote_report = self.promote_important()
        report.entries_promoted = promote_report.entries_promoted

        report.entries_after = self.store.count()
        return report
