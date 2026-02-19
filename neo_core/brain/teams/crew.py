"""
Neo Core — CrewExecutor : Équipe coordonnée de Workers (v2 — Persistant)
=========================================================================
Orchestre un pipeline de Workers spécialisés pour les Epics.

v1 : Pipeline one-shot (toutes les étapes d'un coup)
v2 : Crew persistant piloté par le heartbeat (1 étape par pulse)
     + Communication bidirectionnelle Brain ↔ Crew

Architecture :
  Brain ←→ CrewExecutor ←→ CrewState (persistant en Memory)
                 ↑
           Heartbeat (advance_one_step à chaque pulse)

Exemple : "devenir parieur pro sur le tennis"
  Step 0 (RESEARCHER) → recherche ATP, résultats
  Step 1 (RESEARCHER) → recherche cotes, stratégies
  Step 2 (ANALYST)    → analyse patterns, calculs
  Step 3 (WRITER)     → rédige guide complet
  → Synthèse Claude Sonnet → rapport final cohérent
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional

from neo_core.brain.teams.worker import WorkerType

if TYPE_CHECKING:
    from neo_core.brain.core import Brain

logger = logging.getLogger(__name__)

# Limite de chars par résultat d'étape injecté dans le contexte
_MAX_STEP_OUTPUT_IN_CONTEXT = 3000

# Source Memory pour la persistance des CrewState
_CREW_STATE_SOURCE_PREFIX = "crew_state:"


# ─── Dataclasses ────────────────────────────────────────


@dataclass
class CrewStep:
    """Une étape du pipeline crew."""

    index: int
    description: str
    worker_type: WorkerType

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "description": self.description,
            "worker_type": self.worker_type.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CrewStep:
        wt = data.get("worker_type", "generic")
        try:
            worker_type = WorkerType(wt)
        except ValueError:
            worker_type = WorkerType.GENERIC
        return cls(
            index=data["index"],
            description=data["description"],
            worker_type=worker_type,
        )


@dataclass
class CrewStepResult:
    """Résultat persisté d'une étape exécutée."""

    index: int
    worker_type: str
    output: str
    success: bool
    executed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "worker_type": self.worker_type,
            "output": self.output,
            "success": self.success,
            "executed_at": self.executed_at,
            "execution_time": self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CrewStepResult:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CrewEvent:
    """Message d'un crew vers Brain (notification proactive)."""

    crew_id: str
    event_type: str  # "step_completed" | "step_failed" | "crew_done" | "crew_blocked" | "insight"
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "crew_id": self.crew_id,
            "event_type": self.event_type,
            "message": self.message,
            "timestamp": self.timestamp,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CrewEvent:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CrewState:
    """
    État complet et persistable d'un crew.

    Sérialisé en Memory avec source="crew_state:{epic_id}".
    Rechargé par le heartbeat ou Brain à chaque interaction.
    """

    epic_id: str
    epic_subject: str
    steps: list[CrewStep] = field(default_factory=list)
    results: list[CrewStepResult] = field(default_factory=list)
    current_step_index: int = 0
    status: str = "active"  # "active" | "paused" | "done" | "failed"
    memory_context: str = ""
    original_request: str = ""  # Requête utilisateur ORIGINALE — jamais tronquée
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    events: list[CrewEvent] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.current_step_index >= len(self.steps)

    @property
    def progress_pct(self) -> float:
        if not self.steps:
            return 0.0
        return (self.current_step_index / len(self.steps)) * 100

    def get_accumulated_text(self) -> str:
        """Résultats accumulés des étapes terminées (pour injection contexte)."""
        if not self.results:
            return ""
        parts: list[str] = []
        for res in self.results:
            status_icon = "✅" if res.success else "❌"
            parts.append(f"[Étape {res.index + 1} — {res.worker_type}] {status_icon}")
            parts.append(res.output[:_MAX_STEP_OUTPUT_IN_CONTEXT])
            if len(res.output) > _MAX_STEP_OUTPUT_IN_CONTEXT:
                parts.append("(...tronqué)")
            parts.append("")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "epic_id": self.epic_id,
            "epic_subject": self.epic_subject,
            "original_request": self.original_request[:2000],  # Requête complète
            "steps": [s.to_dict() for s in self.steps],
            "results": [r.to_dict() for r in self.results],
            "current_step_index": self.current_step_index,
            "status": self.status,
            "memory_context": self.memory_context[:2000],  # Limiter la taille
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "events": [e.to_dict() for e in self.events[-20:]],  # Garder les 20 derniers
        }

    @classmethod
    def from_dict(cls, data: dict) -> CrewState:
        return cls(
            epic_id=data["epic_id"],
            epic_subject=data["epic_subject"],
            original_request=data.get("original_request", ""),
            steps=[CrewStep.from_dict(s) for s in data.get("steps", [])],
            results=[CrewStepResult.from_dict(r) for r in data.get("results", [])],
            current_step_index=data.get("current_step_index", 0),
            status=data.get("status", "active"),
            memory_context=data.get("memory_context", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            events=[CrewEvent.from_dict(e) for e in data.get("events", [])],
        )


# ─── Legacy : CrewContext (compat v1 execute()) ─────────


@dataclass
class CrewContext:
    """Contexte accumulé au fil des étapes du crew (compat v1)."""

    epic_id: str
    epic_subject: str
    step_results: list[tuple[int, str, str]] = field(default_factory=list)

    def add_step_result(self, index: int, worker_type: str, output: str) -> None:
        """Ajoute le résultat d'une étape."""
        self.step_results.append((index, worker_type, output))

    def get_accumulated_text(self) -> str:
        """Retourne le texte accumulé de toutes les étapes."""
        if not self.step_results:
            return ""
        parts = []
        for idx, wtype, output in self.step_results:
            parts.append(f"[Étape {idx + 1} — {wtype}]")
            parts.append(output[:_MAX_STEP_OUTPUT_IN_CONTEXT])
            if len(output) > _MAX_STEP_OUTPUT_IN_CONTEXT:
                parts.append("(...tronqué)")
            parts.append("")
        return "\n".join(parts)


# ─── CrewExecutor ───────────────────────────────────────


class CrewExecutor:
    """
    Orchestre une équipe de Workers en pipeline coordonné.

    v1 : execute() — pipeline one-shot (toutes les étapes d'un coup)
    v2 : advance_one_step() — exécution progressive (1 étape par appel)
         + persistance CrewState en Memory
         + communication bidirectionnelle via CrewEvent
    """

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self.memory = brain.memory
        self.factory = brain.factory
        self.worker_manager = brain.worker_manager
        self._event_callback: Optional[Callable[[CrewEvent], None]] = None

    def set_event_callback(self, callback: Callable[[CrewEvent], None]) -> None:
        """Définit le callback pour les événements crew → Brain."""
        self._event_callback = callback

    # ─── Persistance ────────────────────────────────

    def create_crew_state(
        self,
        epic_id: str,
        epic_subject: str,
        steps: list[CrewStep],
        memory_context: str,
        original_request: str = "",
    ) -> CrewState:
        """Crée et persiste un nouvel état de crew."""
        state = CrewState(
            epic_id=epic_id,
            epic_subject=epic_subject,
            original_request=original_request,
            steps=steps,
            memory_context=memory_context,
        )
        self.save_state(state)
        logger.info(
            "[Crew] État créé: %s (%s) — %d étapes",
            epic_id[:8], epic_subject[:50], len(steps),
        )
        return state

    def save_state(self, state: CrewState) -> None:
        """Sauvegarde l'état en Memory (source='crew_state:{epic_id}')."""
        if not self.memory or not self.memory.is_initialized:
            return
        try:
            state.updated_at = datetime.now().isoformat()
            source = f"{_CREW_STATE_SOURCE_PREFIX}{state.epic_id}"

            # Supprimer l'ancien état s'il existe
            existing = self.memory._store.search_by_source(source, limit=1)
            for record in existing:
                try:
                    self.memory._store.delete(record.id)
                except Exception:
                    pass

            # Persister le nouvel état
            self.memory.store_memory(
                content=json.dumps(state.to_dict()),
                source=source,
                tags=[
                    f"crew:{state.epic_id}",
                    "crew_state",
                    f"crew_status:{state.status}",
                ],
                importance=0.9,
                metadata={
                    "epic_id": state.epic_id,
                    "status": state.status,
                    "progress": state.progress_pct,
                    "step_count": len(state.steps),
                },
            )
        except Exception as e:
            logger.error("[Crew] Sauvegarde état échouée: %s", e)

    def load_state(self, epic_id: str) -> Optional[CrewState]:
        """Charge l'état depuis Memory."""
        if not self.memory or not self.memory.is_initialized:
            return None
        try:
            source = f"{_CREW_STATE_SOURCE_PREFIX}{epic_id}"
            records = self.memory._store.search_by_source(source, limit=1)
            for record in records:
                try:
                    data = json.loads(record.content)
                    return CrewState.from_dict(data)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.debug("[Crew] Parsing état échoué: %s", e)
        except Exception as e:
            logger.debug("[Crew] Chargement état échoué: %s", e)
        return None

    def list_active_crews(self) -> list[CrewState]:
        """Liste tous les crews actifs."""
        if not self.memory or not self.memory.is_initialized:
            return []
        try:
            records = self.memory._store.search_by_tags(
                ["crew_state", "crew_status:active"], limit=20,
            )
            crews: list[CrewState] = []
            for record in records:
                try:
                    data = json.loads(record.content)
                    crews.append(CrewState.from_dict(data))
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
            return crews
        except Exception as e:
            logger.debug("[Crew] Listing crews actifs échoué: %s", e)
            return []

    # ─── Exécution progressive (heartbeat) ──────────

    async def advance_one_step(self, epic_id: str) -> CrewEvent:
        """
        Avance le crew d'UNE étape. Appelé par le heartbeat.

        1. Charge l'état depuis Memory
        2. Construit le task enrichi (résultats précédents + mission)
        3. Exécute le Worker pour l'étape courante
        4. Stocke le résultat dans CrewState + Memory
        5. Sauvegarde l'état
        6. Émet un CrewEvent
        7. Si dernière étape → synthèse finale + event "crew_done"
        """
        state = self.load_state(epic_id)
        if not state:
            return self._make_event(
                epic_id, "crew_blocked",
                f"Impossible de charger l'état du crew {epic_id[:8]}",
            )

        if state.is_complete or state.status != "active":
            return self._make_event(
                epic_id, "crew_blocked",
                f"Crew {epic_id[:8]} déjà terminé ou inactif (status={state.status})",
            )

        step = state.steps[state.current_step_index]
        step_num = state.current_step_index + 1
        total = len(state.steps)

        logger.info(
            "[Crew %s] Avancement étape %d/%d — %s (%s)",
            epic_id[:8], step_num, total,
            step.description[:60], step.worker_type.value,
        )

        # 1. Construire le task enrichi
        enriched_task = self._build_enriched_task_from_state(step, state)

        # 2. Exécuter le Worker
        t0 = time.time()
        step_output, success = await self._execute_step(step, enriched_task)
        elapsed = time.time() - t0

        # 3. Enregistrer le résultat
        result = CrewStepResult(
            index=step.index,
            worker_type=step.worker_type.value,
            output=step_output,
            success=success,
            execution_time=elapsed,
        )
        state.results.append(result)
        state.current_step_index += 1

        # 4. Stocker en mémoire (pour recherche sémantique)
        self._store_step_result(epic_id, step, step_output)

        # 5. Vérifier si c'est la dernière étape
        if state.is_complete:
            # Synthèse finale
            synthesis = await self._synthesize(state.epic_subject, state)
            state.status = "done"

            # Stocker la synthèse en mémoire
            self._store_step_result_raw(
                epic_id, "synthesis", synthesis,
                tags=["crew_synthesis", f"crew:{epic_id}"],
            )

            event = self._make_event(
                epic_id, "crew_done",
                f"Crew « {state.epic_subject[:50]} » terminé ({total} étapes). "
                f"Synthèse: {synthesis[:200]}...",
                data={"synthesis": synthesis[:2000], "total_steps": total},
            )
        elif success:
            event = self._make_event(
                epic_id, "step_completed",
                f"Étape {step_num}/{total} terminée — "
                f"{step.worker_type.value}: {step.description[:60]}",
                data={"step_index": step.index, "progress_pct": state.progress_pct},
            )
        else:
            event = self._make_event(
                epic_id, "step_failed",
                f"Étape {step_num}/{total} échouée — "
                f"{step.worker_type.value}: {step_output[:100]}",
                data={"step_index": step.index, "error": step_output[:300]},
            )

        # 6. Sauvegarder l'état + émettre l'événement
        state.events.append(event)
        self.save_state(state)
        self._emit_event(event)

        # 7. Synchroniser avec le TaskRegistry (epic tasks)
        self._sync_task_registry(epic_id, step.index, success, step_output)

        logger.info(
            "[Crew %s] Étape %d/%d %s — %.1fs",
            epic_id[:8], step_num, total,
            "✅" if success else "❌", elapsed,
        )

        return event

    # ─── Interface Brain → Crew (channel) ───────────

    def get_briefing(self, epic_id: str) -> str:
        """
        Retourne un briefing lisible de l'état du crew.
        Appelé quand Brain détecte une question sur un crew actif.
        """
        state = self.load_state(epic_id)
        if not state:
            return f"Aucun crew trouvé pour {epic_id[:8]}."

        total = len(state.steps)
        done = state.current_step_index
        pct = state.progress_pct

        lines: list[str] = []
        lines.append(
            f"Projet « {state.epic_subject} » — "
            f"Étape {done}/{total} ({pct:.0f}%) — {state.status}"
        )

        # Dernière étape exécutée
        if state.results:
            last = state.results[-1]
            icon = "✅" if last.success else "❌"
            lines.append(
                f"Dernière: {icon} {state.steps[last.index].description[:60]} "
                f"({last.worker_type}, {last.execution_time:.0f}s)"
            )

        # Prochaine étape
        if not state.is_complete and state.status == "active":
            next_step = state.steps[state.current_step_index]
            lines.append(
                f"Prochaine: {next_step.description[:60]} ({next_step.worker_type.value})"
            )

        # Résumé des résultats (extraits courts)
        if state.results:
            lines.append("Résultats clés:")
            for res in state.results[-3:]:  # 3 derniers
                icon = "✅" if res.success else "❌"
                summary = res.output[:120].replace("\n", " ")
                lines.append(f"  {icon} Étape {res.index + 1}: {summary}...")

        return "\n".join(lines)

    def add_step(
        self,
        epic_id: str,
        description: str,
        worker_type: WorkerType,
        position: int = -1,
    ) -> bool:
        """Ajoute dynamiquement une étape au plan du crew."""
        state = self.load_state(epic_id)
        if not state or state.status != "active":
            return False

        new_step = CrewStep(
            index=len(state.steps),
            description=description,
            worker_type=worker_type,
        )

        if position < 0 or position >= len(state.steps):
            state.steps.append(new_step)
        else:
            state.steps.insert(position, new_step)
            # Ré-indexer
            for i, step in enumerate(state.steps):
                step.index = i

        self.save_state(state)
        logger.info(
            "[Crew %s] Étape ajoutée: %s (%s)",
            epic_id[:8], description[:50], worker_type.value,
        )
        return True

    # ─── Synchronisation TaskRegistry ────────────────

    def _sync_task_registry(
        self, epic_id: str, step_index: int,
        success: bool, output: str,
    ) -> None:
        """
        Synchronise l'avancement du CrewState avec le TaskRegistry.

        Quand une étape crew avance, la Task correspondante dans le
        registre est mise à jour (pending → done/failed).
        Cela permet à l'affichage (/tasks, TUI sidebar) de refléter
        le vrai progrès de l'epic.
        """
        if not self.memory or not self.memory.is_initialized:
            return

        try:
            registry = self.memory.task_registry
            if not registry:
                return

            epic_tasks = registry.get_epic_tasks(epic_id)
            if not epic_tasks:
                return

            # Les tasks sont dans l'ordre de création = l'ordre des steps
            # On trie par created_at pour s'assurer de l'ordre
            epic_tasks.sort(key=lambda t: t.created_at)

            if step_index < len(epic_tasks):
                task = epic_tasks[step_index]
                new_status = "done" if success else "failed"
                registry.update_task_status(
                    task.id, new_status,
                    result=output[:500],
                )
                logger.debug(
                    "[Crew] TaskRegistry sync: task %s → %s",
                    task.id[:8], new_status,
                )

                # Marquer la prochaine task comme in_progress si elle existe
                next_idx = step_index + 1
                if next_idx < len(epic_tasks):
                    registry.update_task_status(epic_tasks[next_idx].id, "in_progress")

        except Exception as e:
            logger.debug("[Crew] Sync TaskRegistry échouée: %s", e)

    # ─── Pipeline one-shot (compat v1) ──────────────

    async def execute(
        self,
        epic_id: str,
        epic_subject: str,
        steps: list[CrewStep],
        memory_context: str,
    ) -> str:
        """
        Exécute le pipeline crew complet en une passe (v1 compat).

        Pour chaque étape :
        1. Construit le task enrichi (résultats précédents + mission)
        2. Crée et exécute le Worker avec le bon type
        3. Stocke le résultat en mémoire (taggé crew)
        4. Accumule dans le CrewContext

        À la fin : synthèse Claude Sonnet.
        """
        crew_ctx = CrewContext(epic_id=epic_id, epic_subject=epic_subject)

        for step in steps:
            logger.info(
                "[Crew %s] Étape %d/%d — %s (%s)",
                epic_id[:8], step.index + 1, len(steps),
                step.description[:60], step.worker_type.value,
            )

            enriched_task = self._build_enriched_task(step, crew_ctx, memory_context)
            step_output, _ = await self._execute_step(step, enriched_task)

            self._store_step_result(epic_id, step, step_output)
            crew_ctx.add_step_result(
                step.index, step.worker_type.value, step_output,
            )

            logger.info(
                "[Crew %s] Étape %d terminée — %d chars",
                epic_id[:8], step.index + 1, len(step_output),
            )

        return await self._synthesize_from_context(epic_subject, crew_ctx)

    # ─── Méthodes internes ──────────────────────────

    def _build_enriched_task_from_state(
        self, step: CrewStep, state: CrewState,
    ) -> str:
        """Construit le prompt enrichi depuis l'état persistant."""
        parts: list[str] = []

        # TOUJOURS inclure la demande originale de l'utilisateur
        if state.original_request:
            parts.append("=== DEMANDE ORIGINALE DE L'UTILISATEUR ===")
            parts.append(state.original_request)

        accumulated = state.get_accumulated_text()
        if accumulated:
            parts.append("=== RÉSULTATS DES ÉTAPES PRÉCÉDENTES ===")
            parts.append(accumulated)

        if state.memory_context:
            ctx_limit = 800 if accumulated else 2000
            parts.append("=== CONTEXTE MÉMOIRE ===")
            parts.append(state.memory_context[:ctx_limit])

        parts.append(f"=== TA MISSION (Étape {step.index + 1}/{len(state.steps)}) ===")
        parts.append(step.description)
        parts.append(
            f"\nPROJET : « {state.epic_subject} »\n"
            "IMPORTANT : Ta mission s'inscrit dans le projet ci-dessus. "
            "Utilise les résultats des étapes précédentes comme base. "
            "Ne répète pas ce qui a déjà été fait. Avance le projet "
            "en restant fidèle à la DEMANDE ORIGINALE de l'utilisateur."
        )

        return "\n\n".join(parts)

    def _build_enriched_task(
        self,
        step: CrewStep,
        crew_ctx: CrewContext,
        base_memory_context: str,
    ) -> str:
        """Construit le prompt task enrichi pour un Worker (v1 compat)."""
        parts: list[str] = []

        accumulated = crew_ctx.get_accumulated_text()
        if accumulated:
            parts.append("=== RÉSULTATS DES ÉTAPES PRÉCÉDENTES ===")
            parts.append(accumulated)

        if base_memory_context:
            ctx_limit = 800 if accumulated else 2000
            parts.append("=== CONTEXTE MÉMOIRE ===")
            parts.append(base_memory_context[:ctx_limit])

        parts.append(f"=== TA MISSION (Étape {step.index + 1}) ===")
        parts.append(step.description)
        parts.append(
            "\nIMPORTANT : Utilise les résultats des étapes précédentes comme base. "
            "Ne répète pas ce qui a déjà été fait. Avance le projet."
        )

        return "\n\n".join(parts)

    async def _execute_step(
        self, step: CrewStep, enriched_task: str,
    ) -> tuple[str, bool]:
        """Crée et exécute un Worker pour une étape du crew. Retourne (output, success)."""
        worker = self.factory.create_worker_for_type(
            worker_type=step.worker_type,
            task=enriched_task,
            subtasks=[step.description],
        )

        if self.brain._health:
            worker.health_monitor = self.brain._health

        async with worker:
            self.worker_manager.register(worker)
            try:
                result = await worker.execute()

                # Apprentissage
                from neo_core.brain.prompts import BrainDecision
                decision = BrainDecision(
                    action="delegate_worker",
                    worker_type=step.worker_type.value,
                    reasoning=f"Crew step {step.index + 1}",
                )
                await self.brain._learn_from_result(
                    step.description, decision, result,
                )

                if result.success:
                    return result.output, True
                return f"[Échec étape {step.index + 1}] {result.output[:500]}", False

            except Exception as e:
                logger.error(
                    "[Crew] Étape %d échouée: %s", step.index + 1, e,
                )
                return f"[Erreur étape {step.index + 1}] {type(e).__name__}: {str(e)[:300]}", False

            finally:
                self.worker_manager.unregister(worker)

    def _store_step_result(
        self, epic_id: str, step: CrewStep, output: str,
    ) -> None:
        """Stocke le résultat d'une étape en mémoire pour les Workers suivants."""
        self._store_step_result_raw(
            epic_id, step.worker_type.value, output,
            tags=[
                f"crew:{epic_id}",
                f"step_{step.index}",
                step.worker_type.value,
                "crew_result",
            ],
            metadata={
                "epic_id": epic_id,
                "step_index": step.index,
                "worker_type": step.worker_type.value,
                "timestamp": time.time(),
            },
        )

    def _store_step_result_raw(
        self, epic_id: str, label: str, output: str,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Stocke un résultat brut en mémoire."""
        if not self.memory or not self.memory.is_initialized:
            return
        try:
            self.memory.store_memory(
                content=output[:5000],
                source=f"crew:{epic_id}",
                tags=tags or [f"crew:{epic_id}", label],
                importance=0.8,
                metadata=metadata or {"epic_id": epic_id, "label": label},
            )
        except Exception as e:
            logger.debug("Stockage crew result échoué: %s", e)

    async def _synthesize(
        self, epic_subject: str, state: CrewState,
    ) -> str:
        """Synthèse finale par Claude Sonnet depuis l'état persistant."""
        if not state.results:
            return "[Crew] Aucun résultat produit."

        accumulated = state.get_accumulated_text()
        return await self._run_synthesis(
            epic_subject, len(state.results), accumulated,
            original_request=state.original_request,
        )

    async def _synthesize_from_context(
        self, epic_subject: str, crew_ctx: CrewContext,
    ) -> str:
        """Synthèse finale par Claude Sonnet depuis le contexte v1."""
        if not crew_ctx.step_results:
            return "[Crew] Aucun résultat produit."

        accumulated = crew_ctx.get_accumulated_text()
        return await self._run_synthesis(
            epic_subject, len(crew_ctx.step_results), accumulated,
        )

    async def _run_synthesis(
        self, epic_subject: str, step_count: int, accumulated: str,
        original_request: str = "",
    ) -> str:
        """Appel LLM pour la synthèse finale."""
        original_block = ""
        if original_request:
            original_block = (
                f"=== DEMANDE ORIGINALE DE L'UTILISATEUR ===\n"
                f"{original_request}\n\n"
            )
        synthesis_prompt = (
            f"Tu es le synthétiseur du projet « {epic_subject} ».\n\n"
            f"{original_block}"
            f"Une équipe de {step_count} agents spécialisés "
            f"a travaillé sur ce projet. Voici leurs résultats :\n\n"
            f"{accumulated}\n\n"
            f"Ta mission :\n"
            f"1. Intègre les découvertes de chaque étape en un rapport COHÉRENT "
            f"qui répond à la DEMANDE ORIGINALE de l'utilisateur\n"
            f"2. Ne liste PAS les étapes — fusionne les résultats en un texte naturel\n"
            f"3. Mets en avant les insights clés, patterns, et recommandations\n"
            f"4. Si des étapes ont échoué, mentionne-le brièvement\n"
            f"5. Conclus avec les prochaines actions recommandées\n\n"
            f"Réponds directement avec le rapport, sans préambule."
        )

        try:
            return await self.brain._raw_llm_call(synthesis_prompt)
        except Exception as e:
            logger.error("[Crew] Synthèse LLM échouée: %s", e)
            header = f"[Crew — {epic_subject}] {step_count} étapes complétées\n"
            return header + accumulated

    def _make_event(
        self, crew_id: str, event_type: str, message: str,
        data: dict | None = None,
    ) -> CrewEvent:
        """Crée un CrewEvent."""
        return CrewEvent(
            crew_id=crew_id,
            event_type=event_type,
            message=message,
            data=data or {},
        )

    def _emit_event(self, event: CrewEvent) -> None:
        """Émet un événement via le callback si défini."""
        if self._event_callback:
            try:
                self._event_callback(event)
            except Exception as e:
                logger.debug("[Crew] Event callback échoué: %s", e)
