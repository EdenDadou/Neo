"""
Task Registry — Registre de Tâches et Projets
==============================================
Gère le cycle de vie des tâches dans le système Neo Core.

Deux types d'entrées :
- Task : action simple → 1 seul agent (worker)
- Epic (Projet) : mission complexe → équipe d'agents coordonnés (Crew)

Le Brain décide du type (Task vs Projet) selon la complexité.
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
    short_id: str = ""  # ID court affiché (ex: "T1", "T2")
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
            "short_id": self.short_id,
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
        sid = f"#{self.short_id}" if self.short_id else f"[{self.id[:8]}]"
        epic_tag = f" [P:{self.epic_id[:8]}]" if self.epic_id else ""
        attempts = f" (×{self.attempt_count})" if self.attempt_count > 1 else ""
        return f"{icon} {sid} {self.description[:60]}{epic_tag}{attempts} — {self.worker_type}"


@dataclass
class Epic:
    """Un projet complexe nécessitant une équipe d'agents coordonnés (Crew)."""
    id: str
    description: str
    name: str = ""  # Nom court du projet (donné par l'utilisateur)
    short_id: str = ""  # ID court affiché (ex: "P1", "P2")
    task_ids: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | in_progress | done | failed
    strategy: str = ""  # Stratégie de coordination
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""
    context_notes: list[str] = field(default_factory=list)  # Notes de contexte ajoutées par Memory
    # ── Projet cyclique (récurrent) ──
    recurring: bool = False  # True = projet récurrent (relancé automatiquement)
    schedule_cron: str = ""  # Cron expression ("0 10 * * *" = chaque jour 10h)
    schedule_interval_minutes: int = 0  # Alternative au cron : intervalle en minutes
    cycle_count: int = 0  # Nombre de cycles exécutés
    max_cycles: int = 0  # 0 = illimité, sinon s'arrête après N cycles
    last_cycle_at: str = ""  # ISO timestamp du dernier cycle
    next_cycle_at: str = ""  # ISO timestamp du prochain cycle prévu
    cycle_template: list[dict] = field(default_factory=list)  # Template des étapes à reproduire
    goal: str = ""  # Objectif mesurable ("doubler 300€ en simulation")
    goal_reached: bool = False  # True quand l'objectif est atteint
    accumulated_results: list[str] = field(default_factory=list)  # Résumés des cycles précédents (derniers 10)

    @property
    def display_name(self) -> str:
        """Retourne le nom affiché : name si dispo, sinon description tronquée."""
        return self.name if self.name else self.description[:60]

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "description": self.description,
            "name": self.name,
            "short_id": self.short_id,
            "task_ids": self.task_ids,
            "status": self.status,
            "strategy": self.strategy,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "context_notes": self.context_notes,
        }
        # Champs récurrents (seulement si activé, pour ne pas polluer les projets classiques)
        if self.recurring:
            d.update({
                "recurring": self.recurring,
                "schedule_cron": self.schedule_cron,
                "schedule_interval_minutes": self.schedule_interval_minutes,
                "cycle_count": self.cycle_count,
                "max_cycles": self.max_cycles,
                "last_cycle_at": self.last_cycle_at,
                "next_cycle_at": self.next_cycle_at,
                "cycle_template": self.cycle_template,
                "goal": self.goal,
                "goal_reached": self.goal_reached,
                "accumulated_results": self.accumulated_results,
            })
        return d

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
        sid = f"#{self.short_id}" if self.short_id else f"[{self.id[:8]}]"
        n_tasks = len(self.task_ids)
        recurring_tag = f" 🔁 cycle {self.cycle_count}" if self.recurring else ""
        goal_tag = f" 🎯 {self.goal[:30]}" if self.goal else ""
        return f"{icon} {sid} {self.display_name} — {n_tasks} tâche(s){recurring_tag}{goal_tag}"


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
        # Compteurs pour les short_ids — initialisés paresseusement
        self._next_task_num: int | None = None
        self._next_epic_num: int | None = None

    def _init_counters(self) -> None:
        """Initialise les compteurs short_id à partir des records existants."""
        if self._next_task_num is not None:
            return  # Déjà initialisé
        max_t, max_e = 0, 0
        for record in self.store.search_by_source(self.SOURCE_TASK, limit=500):
            try:
                data = json.loads(record.content)
                sid = data.get("short_id", "")
                if sid and sid.startswith("T"):
                    num = int(sid[1:])
                    max_t = max(max_t, num)
            except (json.JSONDecodeError, ValueError):
                pass
        for record in self.store.search_by_source(self.SOURCE_EPIC, limit=100):
            try:
                data = json.loads(record.content)
                sid = data.get("short_id", "")
                if sid and sid.startswith("P"):
                    num = int(sid[1:])
                    max_e = max(max_e, num)
            except (json.JSONDecodeError, ValueError):
                pass
        self._next_task_num = max_t + 1
        self._next_epic_num = max_e + 1

    def _next_task_short_id(self) -> str:
        self._init_counters()
        sid = f"T{self._next_task_num}"
        self._next_task_num += 1
        return sid

    def _next_epic_short_id(self) -> str:
        self._init_counters()
        sid = f"P{self._next_epic_num}"
        self._next_epic_num += 1
        return sid

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
            short_id=self._next_task_short_id(),
            epic_id=epic_id,
        )
        self._persist_task(task)
        return task

    def create_epic(
        self,
        description: str,
        subtask_descriptions: list[tuple[str, str]],
        strategy: str = "",
        name: str = "",
    ) -> Epic:
        """
        Crée un Epic avec ses sous-tâches.

        Args:
            description: Description globale de l'epic
            subtask_descriptions: Liste de (description, worker_type) pour chaque sous-tâche
            strategy: Stratégie de coordination des agents
            name: Nom court du projet (donné par l'utilisateur)
        """
        epic = Epic(
            id=str(uuid.uuid4()),
            description=description,
            name=name,
            short_id=self._next_epic_short_id(),
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

    # ─── Recherche par short_id ───────────────────────

    def find_task_by_short_id(self, short_id: str) -> Optional[Task]:
        """Trouve une tâche par son short_id (ex: 'T3' ou '3')."""
        # Normaliser : "3" → "T3", "t3" → "T3"
        sid = short_id.strip().upper()
        if sid.isdigit():
            sid = f"T{sid}"
        for task in self.get_all_tasks(limit=200):
            if task.short_id == sid:
                return task
        return None

    def find_epic_by_short_id(self, short_id: str) -> Optional[Epic]:
        """Trouve un projet par son short_id (ex: 'P1' ou '1')."""
        sid = short_id.strip().upper()
        if sid.isdigit():
            sid = f"P{sid}"
        for epic in self.get_all_epics(limit=100):
            if epic.short_id == sid:
                return epic
        return None

    # ─── Suppression individuelle ─────────────────────

    def delete_task(self, short_id: str) -> Optional[Task]:
        """Supprime une tâche par son short_id. Retourne la tâche supprimée ou None."""
        task = self.find_task_by_short_id(short_id)
        if not task:
            return None
        _, record_id = self._find_task_with_id(task.id)
        if record_id:
            self.store.delete(record_id)
        return task

    def delete_epic(self, short_id: str) -> tuple[Optional[Epic], int]:
        """
        Supprime un projet et toutes ses tâches liées.
        Retourne (epic, nombre_tâches_supprimées) ou (None, 0).
        """
        epic = self.find_epic_by_short_id(short_id)
        if not epic:
            return None, 0
        # Supprimer les tâches liées
        tasks_deleted = 0
        for task in self.get_epic_tasks(epic.id):
            _, record_id = self._find_task_with_id(task.id)
            if record_id:
                self.store.delete(record_id)
                tasks_deleted += 1
        # Supprimer l'epic
        _, record_id = self._find_epic_with_id(epic.id)
        if record_id:
            self.store.delete(record_id)
        return epic, tasks_deleted

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

    def reset_all_tasks(self) -> int:
        """Supprime TOUTES les tâches (standalone, pas celles liées aux projets)."""
        deleted = 0
        records = self.store.search_by_source(self.SOURCE_TASK, limit=500)
        for record in records:
            try:
                data = json.loads(record.content)
                # Ne supprimer que les tâches sans epic_id (standalone)
                if not data.get("epic_id"):
                    self.store.delete(record.id)
                    deleted += 1
            except (json.JSONDecodeError, TypeError):
                self.store.delete(record.id)
                deleted += 1
        return deleted

    def reset_all_epics(self) -> int:
        """Supprime TOUS les projets (epics) et leurs tâches liées."""
        deleted = 0
        # 1. Supprimer toutes les tâches liées à un epic
        records = self.store.search_by_source(self.SOURCE_TASK, limit=500)
        for record in records:
            try:
                data = json.loads(record.content)
                if data.get("epic_id"):
                    self.store.delete(record.id)
                    deleted += 1
            except (json.JSONDecodeError, TypeError):
                pass
        # 2. Supprimer tous les epics
        records = self.store.search_by_source(self.SOURCE_EPIC, limit=500)
        for record in records:
            self.store.delete(record.id)
            deleted += 1
        return deleted

    def reset_all(self) -> int:
        """Supprime TOUT (tâches + projets). Remet le registre à zéro."""
        deleted = 0
        for source in (self.SOURCE_TASK, self.SOURCE_EPIC):
            records = self.store.search_by_source(source, limit=500)
            for record in records:
                self.store.delete(record.id)
                deleted += 1
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

    # ─── Projets récurrents ─────────────────────────────

    def get_recurring_epics_due(self) -> list[Epic]:
        """Retourne les projets récurrents dont le prochain cycle est dû."""
        now = datetime.now()
        due = []
        for epic in self.get_all_epics(limit=50):
            if not epic.recurring:
                continue
            if epic.goal_reached:
                continue
            if epic.max_cycles > 0 and epic.cycle_count >= epic.max_cycles:
                continue
            # Vérifier si le cycle est dû
            if epic.next_cycle_at:
                try:
                    next_at = datetime.fromisoformat(epic.next_cycle_at)
                    if now >= next_at:
                        due.append(epic)
                except (ValueError, TypeError):
                    due.append(epic)  # Date invalide → relancer
            elif not epic.last_cycle_at:
                # Jamais exécuté → dû immédiatement
                due.append(epic)
        return due

    def advance_recurring_cycle(self, epic_id: str, cycle_summary: str) -> Epic | None:
        """
        Avance un projet récurrent au cycle suivant.
        - Stocke le résumé du cycle terminé
        - Calcule le prochain cycle
        - Remet le status à 'pending' pour le prochain cycle
        - Crée de nouvelles tâches à partir du template
        """
        epic_data, record_id = self._find_epic_with_id(epic_id)
        if not epic_data or not record_id:
            return None

        epic = Epic.from_dict(epic_data)

        # Stocker le résumé du cycle (garder les 10 derniers)
        epic.accumulated_results.append(cycle_summary[:500])
        if len(epic.accumulated_results) > 10:
            epic.accumulated_results = epic.accumulated_results[-10:]

        # Avancer le compteur
        epic.cycle_count += 1
        epic.last_cycle_at = datetime.now().isoformat()

        # Calculer le prochain cycle
        if epic.schedule_interval_minutes > 0:
            from datetime import timedelta
            next_dt = datetime.now() + timedelta(minutes=epic.schedule_interval_minutes)
            epic.next_cycle_at = next_dt.isoformat()
        elif epic.schedule_cron:
            epic.next_cycle_at = self._next_cron_time(epic.schedule_cron)
        else:
            # Pas de schedule → un cycle par jour par défaut
            from datetime import timedelta
            next_dt = datetime.now() + timedelta(hours=24)
            epic.next_cycle_at = next_dt.isoformat()

        # Remettre le status à pending pour le prochain cycle
        epic.status = "pending"
        epic.completed_at = ""

        # Supprimer les anciennes tâches (elles sont dans accumulated_results)
        for task in self.get_epic_tasks(epic.id):
            _, task_record_id = self._find_task_with_id(task.id)
            if task_record_id:
                self.store.delete(task_record_id)
        epic.task_ids = []

        # Créer de nouvelles tâches à partir du template
        if epic.cycle_template:
            for step in epic.cycle_template:
                task = Task(
                    id=str(uuid.uuid4()),
                    description=step.get("description", ""),
                    worker_type=step.get("worker_type", "generic"),
                    short_id=f"T{self._next_task_counter()}",
                    epic_id=epic.id,
                )
                epic.task_ids.append(task.id)
                self.store.add(
                    content=json.dumps(task.to_dict()),
                    source="task_registry:task",
                    tags=["task", f"epic:{epic.id}"],
                )

        # Persister l'epic mis à jour
        self.store.update(record_id, content=json.dumps(epic.to_dict()))

        return epic

    def _next_cron_time(self, cron_expr: str) -> str:
        """Calcule le prochain déclenchement cron (simplifié : heure quotidienne)."""
        # Format supporté simplifié : "HH:MM" ou "0 HH * * *"
        try:
            parts = cron_expr.strip().split()
            if len(parts) == 5:
                minute, hour = int(parts[0]), int(parts[1])
            elif ":" in cron_expr:
                hour, minute = map(int, cron_expr.strip().split(":"))
            else:
                return (datetime.now() + __import__('datetime').timedelta(hours=24)).isoformat()

            from datetime import timedelta
            now = datetime.now()
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return target.isoformat()
        except Exception:
            from datetime import timedelta
            return (datetime.now() + timedelta(hours=24)).isoformat()

    def _next_task_counter(self) -> int:
        """Retourne le prochain numéro de tâche disponible."""
        all_tasks = self.get_all_tasks(limit=500)
        max_n = 0
        for t in all_tasks:
            if t.short_id and t.short_id.startswith("T"):
                try:
                    n = int(t.short_id[1:])
                    if n > max_n:
                        max_n = n
                except ValueError:
                    pass
        return max_n + 1
