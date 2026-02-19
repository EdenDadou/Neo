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
    """Une étape du pipeline crew avec dépendances optionnelles."""

    index: int
    description: str
    worker_type: WorkerType
    depends_on: list[int] = field(default_factory=list)  # indices des steps requis avant exécution

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "description": self.description,
            "worker_type": self.worker_type.value,
            "depends_on": self.depends_on,
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
            depends_on=data.get("depends_on", []),
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
    completed_indices: list[int] = field(default_factory=list)  # v2: indices des steps terminés
    status: str = "active"  # "active" | "paused" | "done" | "failed"
    memory_context: str = ""
    original_request: str = ""  # Requête utilisateur ORIGINALE — jamais tronquée
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    events: list[CrewEvent] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        if not self.steps:
            return True
        # v2: terminé quand tous les steps sont dans completed_indices
        if self.completed_indices:
            return len(self.completed_indices) >= len(self.steps)
        # Compat v1: fallback sur current_step_index
        return self.current_step_index >= len(self.steps)

    @property
    def progress_pct(self) -> float:
        if not self.steps:
            return 0.0
        # Utiliser le max entre completed_indices et current_step_index pour compat v1/v2
        done = max(len(self.completed_indices), self.current_step_index)
        return (done / len(self.steps)) * 100

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
            "completed_indices": self.completed_indices,
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
            completed_indices=data.get("completed_indices", []),
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

    async def advance_ready_steps(self, epic_id: str) -> list[CrewEvent]:
        """
        Avance TOUS les steps prêts en parallèle. Appelé par le heartbeat.

        Un step est "prêt" quand :
        - Il n'est pas encore terminé (pas dans completed_indices)
        - Toutes ses dépendances sont terminées

        Si des steps n'ont pas de depends_on (ancien format), fallback
        sur l'exécution séquentielle classique via advance_one_step().
        """
        import asyncio as _aio

        state = self.load_state(epic_id)
        if not state:
            return [self._make_event(
                epic_id, "crew_blocked",
                f"Impossible de charger l'état du crew {epic_id[:8]}",
            )]

        if state.is_complete or state.status != "active":
            return [self._make_event(
                epic_id, "crew_blocked",
                f"Crew {epic_id[:8]} déjà terminé ou inactif (status={state.status})",
            )]

        # Vérifier si les steps utilisent le format v2 (completed_indices renseigné
        # OU au moins un step a des depends_on non-vides)
        is_v2 = bool(state.completed_indices) or any(len(step.depends_on) > 0 for step in state.steps)
        if not is_v2:
            # Fallback séquentiel (compat ancien format v1)
            event = await self.advance_one_step(epic_id)
            return [event]

        completed_set = set(state.completed_indices)

        # Trouver les steps prêts (dépendances satisfaites, pas encore terminés)
        ready_steps = [
            step for step in state.steps
            if step.index not in completed_set
            and all(d in completed_set for d in step.depends_on)
        ]

        if not ready_steps:
            return [self._make_event(
                epic_id, "crew_blocked",
                f"Crew {epic_id[:8]} — aucun step prêt (dépendances non satisfaites)",
            )]

        logger.info(
            "[Crew %s] Exécution parallèle de %d steps: %s",
            epic_id[:8], len(ready_steps),
            [f"#{s.index}({s.worker_type.value})" for s in ready_steps],
        )

        # Exécuter tous les steps prêts en parallèle
        events = []
        total = len(state.steps)

        async def _run_step(step: CrewStep) -> tuple[CrewStep, str, bool, float]:
            enriched_task = self._build_enriched_task_from_state(step, state)
            t0 = time.time()
            output, success = await self._execute_step(step, enriched_task)
            return step, output, success, time.time() - t0

        results = await _aio.gather(*[_run_step(s) for s in ready_steps], return_exceptions=True)

        # Filtrer les exceptions : les transformer en résultats d'échec
        processed_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                step = ready_steps[i]
                logger.error("[Crew %s] Step %d exception: %s", epic_id[:8], step.index, res)
                processed_results.append((step, f"[Exception] {type(res).__name__}: {str(res)[:300]}", False, 0.0))
            else:
                processed_results.append(res)
        results = processed_results

        for step, output, success, elapsed in results:
            # Enregistrer le résultat
            result = CrewStepResult(
                index=step.index,
                worker_type=step.worker_type.value,
                output=output,
                success=success,
                execution_time=elapsed,
            )
            state.results.append(result)
            state.completed_indices.append(step.index)
            state.current_step_index = max(state.current_step_index, step.index + 1)

            # Stocker en mémoire
            self._store_step_result(epic_id, step, output)
            self._sync_task_registry(epic_id, step.index, success, output)

            if success:
                event = self._make_event(
                    epic_id, "step_completed",
                    f"Étape {step.index + 1}/{total} terminée — "
                    f"{step.worker_type.value}: {step.description[:60]}",
                    data={"step_index": step.index, "progress_pct": state.progress_pct},
                )
            else:
                event = self._make_event(
                    epic_id, "step_failed",
                    f"Étape {step.index + 1}/{total} échouée — "
                    f"{step.worker_type.value}: {output[:100]}",
                    data={"step_index": step.index, "error": output[:300]},
                )
            state.events.append(event)
            events.append(event)

            logger.info(
                "[Crew %s] Étape %d/%d %s — %.1fs",
                epic_id[:8], step.index + 1, total,
                "✅" if success else "❌", elapsed,
            )

        # ── Orchestrateur : réévaluer le plan après ce batch ──
        # Si il reste des steps non terminés, l'orchestrateur peut adapter le plan
        if not state.is_complete:
            orchestrator_events = await self._orchestrator_replan(state)
            events.extend(orchestrator_events)
            # Recalculer total après possible ajout de steps
            total = len(state.steps)

        # Vérifier si tout est terminé → synthèse
        if state.is_complete:
            synthesis = await self._synthesize(state.epic_subject, state)
            state.status = "done"
            self._store_step_result_raw(
                epic_id, "synthesis", synthesis,
                tags=["crew_synthesis", f"crew:{epic_id}"],
            )
            done_event = self._make_event(
                epic_id, "crew_done",
                f"Crew « {state.epic_subject[:50]} » terminé ({total} étapes). "
                f"Synthèse: {synthesis[:200]}...",
                data={"synthesis": synthesis[:2000], "total_steps": total},
            )
            state.events.append(done_event)
            events.append(done_event)

        self.save_state(state)
        for event in events:
            self._emit_event(event)

        return events

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
        # Synchroniser completed_indices (compat v2)
        if step.index not in state.completed_indices:
            state.completed_indices.append(step.index)

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

    # ─── Orchestrateur : réévaluation du plan ───────

    _ORCHESTRATOR_PROMPT = """Tu es l'orchestrateur du projet « {epic_subject} ».

DEMANDE ORIGINALE : {original_request}

PLAN ACTUEL ({total_steps} étapes) :
{plan_summary}

RÉSULTATS DES ÉTAPES TERMINÉES :
{results_summary}

ÉTAPES RESTANTES (pas encore exécutées) :
{remaining_summary}

Ta mission : analyser les résultats et décider si le plan doit être adapté.

Réponds en JSON strict :
{{
  "action": "continue|add_steps|remove_steps|replace_steps",
  "reasoning": "explication courte de ta décision",
  "new_steps": [
    {{"description": "...", "worker_type": "researcher|coder|analyst|writer|summarizer|generic", "depends_on": [indices]}}
  ],
  "remove_indices": [indices des steps à supprimer],
  "replace": {{old_index: {{"description": "...", "worker_type": "...", "depends_on": [...]}} }}
}}

RÈGLES :
- "continue" : le plan est bon, on continue tel quel (new_steps=[], remove_indices=[], replace={{}})
- "add_steps" : les résultats révèlent un besoin imprévu → ajouter des étapes
- "remove_steps" : certaines étapes restantes sont devenues inutiles
- "replace_steps" : une étape restante doit être redéfinie vu les nouveaux résultats
- Les depends_on des new_steps font référence aux indices EXISTANTS ou aux nouveaux (ajoutés à la suite)
- Maximum 3 étapes ajoutées à la fois
- Si tout va bien, préfère "continue" — ne change le plan que si c'est VRAIMENT nécessaire

Réponds UNIQUEMENT avec le JSON."""

    async def _orchestrator_replan(self, state: CrewState) -> list[CrewEvent]:
        """
        Agent orchestrateur : réévalue le plan après chaque batch de steps.

        Analyse les résultats des étapes terminées et décide si le plan
        doit être adapté (ajouter/supprimer/remplacer des étapes).

        Retourne des CrewEvents décrivant les modifications.
        """
        events: list[CrewEvent] = []
        completed_set = set(state.completed_indices)
        remaining = [s for s in state.steps if s.index not in completed_set]

        # Pas besoin de replan si tout est terminé ou s'il ne reste qu'1 step
        if not remaining or len(remaining) <= 1:
            return events

        # Construire les résumés pour le prompt
        plan_lines = []
        for s in state.steps:
            status = "✅" if s.index in completed_set else "⏳"
            deps = f" (après {s.depends_on})" if s.depends_on else ""
            plan_lines.append(f"  {status} Step {s.index}: {s.worker_type.value} — {s.description[:80]}{deps}")

        results_lines = []
        for r in state.results:
            icon = "✅" if r.success else "❌"
            results_lines.append(f"  {icon} Step {r.index} ({r.worker_type}): {r.output[:200]}")

        remaining_lines = []
        for s in remaining:
            deps = f" (après {s.depends_on})" if s.depends_on else ""
            remaining_lines.append(f"  Step {s.index}: {s.worker_type.value} — {s.description[:80]}{deps}")

        prompt = self._ORCHESTRATOR_PROMPT.format(
            epic_subject=state.epic_subject,
            original_request=state.original_request[:500],
            total_steps=len(state.steps),
            plan_summary="\n".join(plan_lines),
            results_summary="\n".join(results_lines) or "  (aucun résultat)",
            remaining_summary="\n".join(remaining_lines),
        )

        try:
            response = await self.brain._raw_llm_call(prompt)

            # Parser le JSON via le helper de Brain
            data = self.brain._parse_json_response(response)
            action = data.get("action", "continue")
            reasoning = data.get("reasoning", "")

            logger.info(
                "[Crew Orchestrator %s] Décision: %s — %s",
                state.epic_id[:8], action, reasoning[:80],
            )

            if action == "continue":
                # Plan inchangé
                return events

            if action == "add_steps":
                new_steps_data = data.get("new_steps", [])
                if new_steps_data:
                    added = self._apply_add_steps(state, new_steps_data)
                    if added:
                        event = self._make_event(
                            state.epic_id, "orchestrator_replan",
                            f"Orchestrateur: +{len(added)} étapes ajoutées — {reasoning[:100]}",
                            data={"action": "add_steps", "added_count": len(added),
                                  "reasoning": reasoning},
                        )
                        state.events.append(event)
                        events.append(event)

            elif action == "remove_steps":
                remove_indices = data.get("remove_indices", [])
                if remove_indices:
                    removed = self._apply_remove_steps(state, remove_indices)
                    if removed:
                        event = self._make_event(
                            state.epic_id, "orchestrator_replan",
                            f"Orchestrateur: -{len(removed)} étapes supprimées — {reasoning[:100]}",
                            data={"action": "remove_steps", "removed_count": len(removed),
                                  "reasoning": reasoning},
                        )
                        state.events.append(event)
                        events.append(event)

            elif action == "replace_steps":
                replacements = data.get("replace", {})
                if replacements:
                    replaced = self._apply_replace_steps(state, replacements)
                    if replaced:
                        event = self._make_event(
                            state.epic_id, "orchestrator_replan",
                            f"Orchestrateur: {len(replaced)} étapes modifiées — {reasoning[:100]}",
                            data={"action": "replace_steps", "replaced_count": len(replaced),
                                  "reasoning": reasoning},
                        )
                        state.events.append(event)
                        events.append(event)

        except Exception as e:
            logger.debug("[Crew Orchestrator] Replan failed: %s", e)

        # Sauvegarder l'état après replan (même si continue, pour les events)
        if events:
            self.save_state(state)

        return events

    def _apply_add_steps(self, state: CrewState, new_steps_data: list[dict]) -> list[CrewStep]:
        """Ajoute de nouvelles étapes au plan. Max 3."""
        added = []
        base_index = len(state.steps)

        for i, item in enumerate(new_steps_data[:3]):  # Max 3
            desc = item.get("description", "")
            wt_str = item.get("worker_type", "generic")
            depends_on = item.get("depends_on", [])
            # Valider les dépendances
            max_valid = base_index + i
            depends_on = [d for d in depends_on if isinstance(d, int) and 0 <= d < max_valid]

            try:
                wt = WorkerType(wt_str)
            except ValueError:
                wt = WorkerType.GENERIC

            if desc:
                step = CrewStep(
                    index=base_index + i,
                    description=desc,
                    worker_type=wt,
                    depends_on=depends_on,
                )
                state.steps.append(step)
                added.append(step)

                # Synchroniser avec le TaskRegistry
                self._register_new_step_task(state.epic_id, step, state.epic_subject)

        if added:
            logger.info(
                "[Crew Orchestrator %s] %d étapes ajoutées: %s",
                state.epic_id[:8], len(added),
                [f"#{s.index}({s.worker_type.value})" for s in added],
            )
        return added

    def _apply_remove_steps(self, state: CrewState, remove_indices: list[int]) -> list[int]:
        """Marque des étapes comme 'terminées' (skip) sans les exécuter."""
        completed_set = set(state.completed_indices)
        removed = []

        for idx in remove_indices:
            if isinstance(idx, int) and idx not in completed_set:
                # Vérifier que l'index existe
                if any(s.index == idx for s in state.steps):
                    state.completed_indices.append(idx)
                    # Ajouter un résultat "skipped"
                    state.results.append(CrewStepResult(
                        index=idx,
                        worker_type="skipped",
                        output="[Orchestrateur] Étape supprimée du plan — jugée non nécessaire",
                        success=True,
                        execution_time=0.0,
                    ))
                    removed.append(idx)

        if removed:
            logger.info(
                "[Crew Orchestrator %s] %d étapes supprimées: %s",
                state.epic_id[:8], len(removed), removed,
            )
        return removed

    def _apply_replace_steps(self, state: CrewState, replacements: dict) -> list[int]:
        """Remplace la description/type de steps non encore exécutés."""
        completed_set = set(state.completed_indices)
        replaced = []

        for idx_str, new_data in replacements.items():
            try:
                idx = int(idx_str)
            except (ValueError, TypeError):
                continue

            if idx in completed_set:
                continue  # Ne pas modifier un step déjà terminé

            # Trouver le step
            for step in state.steps:
                if step.index == idx:
                    if "description" in new_data:
                        step.description = new_data["description"]
                    if "worker_type" in new_data:
                        try:
                            step.worker_type = WorkerType(new_data["worker_type"])
                        except ValueError:
                            pass
                    if "depends_on" in new_data:
                        step.depends_on = [
                            d for d in new_data["depends_on"]
                            if isinstance(d, int) and 0 <= d < len(state.steps) and d != idx
                        ]
                    replaced.append(idx)
                    break

        if replaced:
            logger.info(
                "[Crew Orchestrator %s] %d étapes remplacées: %s",
                state.epic_id[:8], len(replaced), replaced,
            )
        return replaced

    def _register_new_step_task(self, epic_id: str, step: CrewStep, epic_subject: str) -> None:
        """Enregistre une nouvelle étape dans le TaskRegistry."""
        if not self.memory or not self.memory.is_initialized:
            return
        try:
            registry = self.memory.task_registry
            if registry:
                registry.create_task(
                    description=step.description[:200],
                    worker_type=step.worker_type.value,
                    epic_id=epic_id,
                )
        except Exception as e:
            logger.debug("[Crew Orchestrator] Task registration failed: %s", e)

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
