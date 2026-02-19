"""
Task Registry — Registre de Tâches et Epics
=============================================
Gère le cycle de vie des tâches dans le système Neo Core.

Deux types d'entrées :
- Task : problème simple → 1 seul agent (worker)
- Epic : problème complexe → équipe d'agents coordonnés

Le Brain décide du type (Task vs Epic) selon la complexité.
Persistance via MemoryStore.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from neo_core.memory.store import MemoryStore


# ─── Structures de données ──────────────────────────────

@dataclass
class Task:
    """Une tâche simple nécessitant un seul agent."""
    id: str
    description: str
    worker_type: str
    status: str = "pending"  # pending | in_progress | done | failed
    result: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""
    epic_id: Optional[str] = None  # Si cette tâche fait partie d'un Epic
    attempt_count: int = 0  # Nombre de tentatives
    context_notes: list[str] = field(default_factory=list)  # Notes de contexte ajoutées par Memory

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "worker_type": self.worker_type,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "epic_id": self.epic_id,
            "attempt_count": self.attempt_count,
            "context_notes": self.context_notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Task:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def is_terminal(self) -> bool:
        """True si la tâche est dans un état final."""
        return self.status in ("done", "failed")

    def __str__(self) -> str:
        status_icons = {
            "pending": "⏳",
            "in_progress": "🔄",
            "done": "✅",
            "failed": "❌",
        }
        icon = status_icons.get(self.status, "?")
        epic_tag = f" [Epic:{self.epic_id[:8]}]" if self.epic_id else ""
        attempts = f" (×{self.attempt_count})" if self.attempt_count > 1 else ""
        return f"{icon} [{self.id[:8]}] {self.description[:60]}{epic_tag}{attempts} — {self.worker_type}"


@dataclass
class Epic:
    """Un problème complexe nécessitant une équipe d'agents coordonnés."""
    id: str
    description: str
    task_ids: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | in_progress | done | failed
    strategy: str = ""  # Stratégie de coordination
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""
    context_notes: list[str] = field(default_factory=list)  # Notes de contexte ajoutées par Memory

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "task_ids": self.task_ids,
            "status": self.status,
            "strategy": self.strategy,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "context_notes": self.context_notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Epic:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def is_terminal(self) -> bool:
        return self.status in ("done", "failed")

    def __str__(self) -> str:
        status_icons = {
            "pending": "⏳",
            "in_progress": "🔄",
            "done": "✅",
            "failed": "❌",
        }
        icon = status_icons.get(self.status, "?")
        n_tasks = len(self.task_ids)
        return f"{icon} [{self.id[:8]}] {self.description[:60]} — {n_tasks} tâche(s)"


# ─── Task Registry ──────────────────────────────────────

class TaskRegistry:
    """
    Registre persistant de tâches et epics.

    Stocke dans le MemoryStore avec des sources dédiées :
    - task_registry:task
    - task_registry:epic
    """

    SOURCE_TASK = "task_registry:task"
    SOURCE_EPIC = "task_registry:epic"

    def __init__(self, store: MemoryStore):
        self.store = store

    # ─── Création ────────────────────────────────────

    def create_task(
        self,
        description: str,
        worker_type: str,
        epic_id: Optional[str] = None,
    ) -> Task:
        """Crée et persiste une nouvelle Task."""
        task = Task(
            id=str(uuid.uuid4()),
            description=description,
            worker_type=worker_type,
            epic_id=epic_id,
        )
        self._persist_task(task)
        return task

    def create_epic(
        self,
        description: str,
        subtask_descriptions: list[tuple[str, str]],
        strategy: str = "",
    ) -> Epic:
        """
        Crée un Epic avec ses sous-tâches.

        Args:
            description: Description globale de l'epic
            subtask_descriptions: Liste de (description, worker_type) pour chaque sous-tâche
            strategy: Stratégie de coordination des agents
        """
        epic = Epic(
            id=str(uuid.uuid4()),
            description=description,
            strategy=strategy,
        )

        # Créer les sous-tâches liées à l'epic
        for task_desc, task_worker_type in subtask_descriptions:
            task = self.create_task(
                description=task_desc,
                worker_type=task_worker_type,
                epic_id=epic.id,
            )
            epic.task_ids.append(task.id)

        self._persist_epic(epic)
        return epic

    # ─── Mise à jour ─────────────────────────────────

    def update_task_status(
        self,
        task_id: str,
        status: str,
        result: str = "",
    ) -> Optional[Task]:
        """Met à jour le statut d'une Task."""
        task, record_id = self._find_task_with_id(task_id)
        if not task or not record_id:
            return None

        task.status = status
        if result:
            task.result = result
        if status in ("done", "failed"):
            task.completed_at = datetime.now().isoformat()
        if status == "in_progress":
            task.attempt_count += 1

        # Supprimer l'ancien record et persister le nouveau
        try:
            self.store.delete(record_id)
        except Exception:
            pass
        self._persist_task(task)

        # Si la tâche fait partie d'un epic, vérifier si l'epic est terminé
        if task.epic_id:
            self._check_epic_completion(task.epic_id)

        return task

    def update_epic_status(
        self,
        epic_id: str,
        status: str,
    ) -> Optional[Epic]:
        """Met à jour le statut d'un Epic."""
        epic, record_id = self._find_epic_with_id(epic_id)
        if not epic or not record_id:
            return None

        epic.status = status
        if status in ("done", "failed"):
            epic.completed_at = datetime.now().isoformat()

        try:
            self.store.delete(record_id)
        except Exception:
            pass
        self._persist_epic(epic)
        return epic

    def add_task_context(self, task_id: str, note: str) -> Optional[Task]:
        """
        Ajoute une note de contexte à une Task.
        Appelé par Memory lorsqu'il collecte des informations pertinentes.
        """
        task, record_id = self._find_task_with_id(task_id)
        if not task or not record_id:
            return None

        # Limiter à 20 notes pour éviter l'inflation
        task.context_notes.append(note[:300])
        if len(task.context_notes) > 20:
            task.context_notes = task.context_notes[-20:]

        try:
            self.store.delete(record_id)
        except Exception:
            pass
        self._persist_task(task)
        return task

    def add_epic_context(self, epic_id: str, note: str) -> Optional[Epic]:
        """
        Ajoute une note de contexte à un Epic.
        Appelé par Memory lorsqu'il collecte des informations pertinentes.
        """
        epic, record_id = self._find_epic_with_id(epic_id)
        if not epic or not record_id:
            return None

        epic.context_notes.append(note[:300])
        if len(epic.context_notes) > 20:
            epic.context_notes = epic.context_notes[-20:]

        try:
            self.store.delete(record_id)
        except Exception:
            pass
        self._persist_epic(epic)
        return epic

    # ─── Consultation ────────────────────────────────

    def get_task(self, task_id: str) -> Optional[Task]:
        """Récupère une Task par son ID."""
        task, _ = self._find_task_with_id(task_id)
        return task

    def get_epic(self, epic_id: str) -> Optional[Epic]:
        """Récupère un Epic par son ID."""
        epic, _ = self._find_epic_with_id(epic_id)
        return epic

    def get_pending_tasks(self) -> list[Task]:
        """Retourne toutes les tâches en attente."""
        all_tasks = self.get_all_tasks()
        return [t for t in all_tasks if t.status == "pending"]

    def get_active_tasks(self) -> list[Task]:
        """Retourne toutes les tâches en cours."""
        all_tasks = self.get_all_tasks()
        return [t for t in all_tasks if t.status == "in_progress"]

    def get_all_tasks(self, limit: int = 50) -> list[Task]:
        """Retourne toutes les tâches, triées par date de création."""
        records = self.store.search_by_source(self.SOURCE_TASK, limit=limit)
        tasks = []
        for record in records:
            try:
                data = json.loads(record.content)
                tasks.append(Task.from_dict(data))
            except (json.JSONDecodeError, TypeError):
                pass
        # Trier par date de création (plus récent d'abord)
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks

    def get_all_epics(self, limit: int = 20) -> list[Epic]:
        """Retourne tous les epics, triés par date de création."""
        records = self.store.search_by_source(self.SOURCE_EPIC, limit=limit)
        epics = []
        for record in records:
            try:
                data = json.loads(record.content)
                epics.append(Epic.from_dict(data))
            except (json.JSONDecodeError, TypeError):
                pass
        epics.sort(key=lambda e: e.created_at, reverse=True)
        return epics

    def get_epic_tasks(self, epic_id: str) -> list[Task]:
        """Retourne toutes les tâches d'un Epic."""
        all_tasks = self.get_all_tasks(limit=200)
        return [t for t in all_tasks if t.epic_id == epic_id]

    def get_summary(self) -> dict:
        """Retourne un résumé du registre."""
        tasks = self.get_all_tasks()
        epics = self.get_all_epics()

        task_by_status = {}
        for t in tasks:
            task_by_status[t.status] = task_by_status.get(t.status, 0) + 1

        epic_by_status = {}
        for e in epics:
            epic_by_status[e.status] = epic_by_status.get(e.status, 0) + 1

        return {
            "total_tasks": len(tasks),
            "tasks_by_status": task_by_status,
            "total_epics": len(epics),
            "epics_by_status": epic_by_status,
        }

    def get_organized_summary(self) -> dict:
        """
        Retourne un résumé ORGANISÉ du registre, groupé par Epic.

        Structure :
        {
            "epics": [
                {
                    "epic": Epic,
                    "tasks": [Task, ...],
                    "progress": "3/5",
                },
            ],
            "standalone_tasks": [Task, ...],  # Tâches hors epic
            "stats": {...},
        }
        """
        tasks = self.get_all_tasks(limit=100)
        epics = self.get_all_epics(limit=20)

        # Grouper les tâches par epic
        epic_tasks_map: dict[str, list[Task]] = {}
        standalone: list[Task] = []

        for t in tasks:
            if t.epic_id:
                epic_tasks_map.setdefault(t.epic_id, []).append(t)
            else:
                standalone.append(t)

        # Construire le résumé par epic
        epic_summaries = []
        for epic in epics:
            etasks = epic_tasks_map.get(epic.id, [])
            etasks.sort(key=lambda t: t.created_at)
            done_count = sum(1 for t in etasks if t.status == "done")
            total = len(etasks)
            epic_summaries.append({
                "epic": epic,
                "tasks": etasks,
                "progress": f"{done_count}/{total}",
                "done_count": done_count,
                "total": total,
            })

        return {
            "epics": epic_summaries,
            "standalone_tasks": standalone,
            "stats": self.get_summary(),
        }

    def cleanup_completed(self, max_age_hours: float = 48.0) -> int:
        """
        Supprime les tâches terminées (done/failed) plus anciennes que max_age_hours.
        Garde les tâches des Epics actifs intactes.

        Retourne le nombre de tâches supprimées.
        """
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        deleted = 0

        # Identifier les epics actifs
        active_epic_ids = set()
        for epic in self.get_all_epics():
            if epic.status in ("pending", "in_progress"):
                active_epic_ids.add(epic.id)

        tasks = self.get_all_tasks(limit=200)
        for task in tasks:
            if not task.is_terminal:
                continue
            # Ne pas supprimer les tâches d'Epics actifs
            if task.epic_id and task.epic_id in active_epic_ids:
                continue
            try:
                completed = datetime.fromisoformat(task.completed_at) if task.completed_at else None
                created = datetime.fromisoformat(task.created_at)
                ref_time = completed or created
                if ref_time < cutoff:
                    _, record_id = self._find_task_with_id(task.id)
                    if record_id:
                        self.store.delete(record_id)
                        deleted += 1
            except (ValueError, TypeError):
                pass

        # Nettoyer aussi les Epics terminés
        for epic in self.get_all_epics():
            if not epic.is_terminal:
                continue
            try:
                completed = datetime.fromisoformat(epic.completed_at) if epic.completed_at else None
                created = datetime.fromisoformat(epic.created_at)
                ref_time = completed or created
                if ref_time < cutoff:
                    _, record_id = self._find_epic_with_id(epic.id)
                    if record_id:
                        self.store.delete(record_id)
                        deleted += 1
            except (ValueError, TypeError):
                pass

        return deleted

    # ─── Méthodes internes ───────────────────────────

    def _persist_task(self, task: Task) -> None:
        """Persiste une Task dans le MemoryStore."""
        self.store.store(
            content=json.dumps(task.to_dict()),
            source=self.SOURCE_TASK,
            tags=[
                f"task:{task.id}",
                f"status:{task.status}",
                f"worker:{task.worker_type}",
            ] + ([f"epic:{task.epic_id}"] if task.epic_id else []),
            importance=0.6 if task.status == "pending" else 0.4,
            metadata=task.to_dict(),
        )

    def _persist_epic(self, epic: Epic) -> None:
        """Persiste un Epic dans le MemoryStore."""
        self.store.store(
            content=json.dumps(epic.to_dict()),
            source=self.SOURCE_EPIC,
            tags=[
                f"epic:{epic.id}",
                f"status:{epic.status}",
            ],
            importance=0.7 if epic.status == "pending" else 0.5,
            metadata=epic.to_dict(),
        )

    def _find_task_with_id(self, task_id: str) -> tuple[Optional[Task], Optional[str]]:
        """Cherche une Task par son ID et retourne aussi le record ID du store."""
        records = self.store.search_by_tags(
            [f"task:{task_id}"],
            limit=5,
        )
        for record in records:
            if record.source == self.SOURCE_TASK:
                try:
                    data = json.loads(record.content)
                    return Task.from_dict(data), record.id
                except (json.JSONDecodeError, TypeError):
                    pass
        return None, None

    def _find_epic_with_id(self, epic_id: str) -> tuple[Optional[Epic], Optional[str]]:
        """Cherche un Epic par son ID et retourne aussi le record ID du store."""
        records = self.store.search_by_tags(
            [f"epic:{epic_id}"],
            limit=5,
        )
        for record in records:
            if record.source == self.SOURCE_EPIC:
                try:
                    data = json.loads(record.content)
                    return Epic.from_dict(data), record.id
                except (json.JSONDecodeError, TypeError):
                    pass
        return None, None

    def _check_epic_completion(self, epic_id: str) -> None:
        """Vérifie si toutes les tâches d'un Epic sont terminées."""
        epic_tasks = self.get_epic_tasks(epic_id)
        if not epic_tasks:
            return

        all_done = all(t.status == "done" for t in epic_tasks)
        any_failed = any(t.status == "failed" for t in epic_tasks)

        if all_done:
            self.update_epic_status(epic_id, "done")
        elif any_failed and all(t.is_terminal for t in epic_tasks):
            self.update_epic_status(epic_id, "failed")
