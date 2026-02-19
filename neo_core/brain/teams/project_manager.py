"""
Neo Core — ProjectManagerWorker : Chef de Projet autonome
==========================================================
Worker spécial qui orchestre l'exécution parallèle d'un projet.

Architecture :
  Brain → crée Epic + étapes → spawn ProjectManagerWorker
  ProjectManagerWorker → spawn les workers en parallèle → collecte → synthèse

Le PM n'est PAS un Worker LLM classique. C'est un orchestrateur qui :
1. Reçoit les CrewSteps à exécuter
2. Résout les dépendances (DAG)
3. Spawn les workers par batch parallèle
4. Collecte les résultats
5. Fait la synthèse finale
6. Met à jour le TaskRegistry
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from neo_core.brain.teams.crew import (
    CrewEvent,
    CrewState,
    CrewStep,
    CrewStepResult,
    _CREW_STATE_SOURCE_PREFIX,
    _MAX_STEP_OUTPUT_IN_CONTEXT,
)
from neo_core.brain.teams.worker import WorkerType

if TYPE_CHECKING:
    from neo_core.brain.core import Brain
    from neo_core.brain.teams.crew import CrewExecutor

logger = logging.getLogger(__name__)


@dataclass
class PMResult:
    """Résultat du ProjectManagerWorker."""

    success: bool
    synthesis: str
    step_results: list[CrewStepResult] = field(default_factory=list)
    events: list[CrewEvent] = field(default_factory=list)
    total_time: float = 0.0
    steps_completed: int = 0
    steps_failed: int = 0


class ProjectManagerWorker:
    """
    Chef de Projet — orchestre l'exécution parallèle des étapes d'un projet.

    Spawn par Brain après la création de l'Epic.
    Gère tout le cycle de vie : exécution, suivi, synthèse.
    """

    def __init__(
        self,
        brain: Brain,
        epic_id: str,
        epic_subject: str,
        steps: list[CrewStep],
        memory_context: str = "",
        original_request: str = "",
        event_callback=None,
    ) -> None:
        self.brain = brain
        self.memory = brain.memory
        self.factory = brain.factory
        self.worker_manager = brain.worker_manager
        self.epic_id = epic_id
        self.epic_subject = epic_subject
        self.steps = steps
        self.memory_context = memory_context
        self.original_request = original_request
        self._event_callback = event_callback

        # État interne
        self._results: list[CrewStepResult] = []
        self._completed_indices: set[int] = set()
        self._events: list[CrewEvent] = []
        self._started_at: float = 0.0

    # ─── Exécution principale ───────────────────────────

    async def execute(self) -> PMResult:
        """
        Exécute toutes les étapes du projet en parallèle par batch.

        Pipeline :
        1. Identifier les steps prêts (dépendances satisfaites)
        2. Spawn les workers en parallèle pour ce batch
        3. Collecter les résultats
        4. Répéter jusqu'à tout terminé
        5. Synthèse finale
        """
        self._started_at = time.time()
        total = len(self.steps)

        logger.info(
            "[PM %s] Démarrage — %d étapes pour « %s »",
            self.epic_id[:8], total, self.epic_subject[:50],
        )

        # ── Créer le CrewState pour le suivi ──
        state = CrewState(
            epic_id=self.epic_id,
            epic_subject=self.epic_subject,
            original_request=self.original_request,
            steps=self.steps,
            memory_context=self.memory_context,
            status="active",
        )
        self._save_state(state)

        # ── Boucle par batch ──
        max_rounds = total + 3  # sécurité anti-boucle infinie
        round_num = 0

        while round_num < max_rounds:
            round_num += 1

            # Trouver les steps prêts
            ready = self._get_ready_steps()
            if not ready:
                # Soit tout est terminé, soit deadlock
                if len(self._completed_indices) >= total:
                    break  # Tout terminé
                # Deadlock — steps restants avec dépendances non satisfaites
                remaining = [s for s in self.steps if s.index not in self._completed_indices]
                logger.warning(
                    "[PM %s] Deadlock — %d steps bloqués: %s",
                    self.epic_id[:8], len(remaining),
                    [s.index for s in remaining],
                )
                # Forcer l'exécution des steps bloqués
                ready = remaining[:3]

            logger.info(
                "[PM %s] Batch %d — %d workers en parallèle: %s",
                self.epic_id[:8], round_num, len(ready),
                [f"#{s.index}({s.worker_type.value})" for s in ready],
            )

            # ── Spawn les workers en parallèle ──
            batch_results = await self._execute_batch(ready, state)

            # ── Mettre à jour l'état ──
            for step, output, success, elapsed in batch_results:
                result = CrewStepResult(
                    index=step.index,
                    worker_type=step.worker_type.value,
                    output=output,
                    success=success,
                    execution_time=elapsed,
                )
                self._results.append(result)
                self._completed_indices.add(step.index)
                state.results.append(result)
                state.completed_indices.append(step.index)
                state.current_step_index = max(
                    state.current_step_index, step.index + 1,
                )

                # Stocker en mémoire pour les workers suivants
                self._store_step_result(step, output)
                # Sync TaskRegistry
                self._sync_task_registry(step.index, success, output)

                # Émettre événement
                if success:
                    event = self._make_event(
                        "step_completed",
                        f"✅ Étape {step.index + 1}/{total} — "
                        f"{step.worker_type.value}: {step.description[:60]}",
                        data={"step_index": step.index, "elapsed": elapsed},
                    )
                else:
                    event = self._make_event(
                        "step_failed",
                        f"❌ Étape {step.index + 1}/{total} — "
                        f"{step.worker_type.value}: {output[:100]}",
                        data={"step_index": step.index, "error": output[:300]},
                    )
                self._events.append(event)
                state.events.append(event)
                self._emit_event(event)

                logger.info(
                    "[PM %s] Étape %d/%d %s — %.1fs",
                    self.epic_id[:8], step.index + 1, total,
                    "✅" if success else "❌", elapsed,
                )

            # Sauvegarder l'état après chaque batch
            self._save_state(state)

        # ── Synthèse finale ──
        total_time = time.time() - self._started_at
        steps_ok = sum(1 for r in self._results if r.success)
        steps_ko = sum(1 for r in self._results if not r.success)

        synthesis = await self._synthesize(state)

        # Marquer le projet comme terminé
        state.status = "done"
        done_event = self._make_event(
            "crew_done",
            f"Projet « {self.epic_subject[:50]} » terminé — "
            f"{steps_ok}/{total} étapes réussies en {total_time:.0f}s",
            data={"synthesis": synthesis[:2000], "total_steps": total,
                  "total_time": total_time},
        )
        self._events.append(done_event)
        state.events.append(done_event)
        self._save_state(state)
        self._emit_event(done_event)

        logger.info(
            "[PM %s] Projet terminé — %d/%d OK, %.0fs total",
            self.epic_id[:8], steps_ok, total, total_time,
        )

        return PMResult(
            success=steps_ko == 0,
            synthesis=synthesis,
            step_results=self._results,
            events=self._events,
            total_time=total_time,
            steps_completed=steps_ok,
            steps_failed=steps_ko,
        )

    # ─── Exécution d'un batch en parallèle ──────────────

    async def _execute_batch(
        self, steps: list[CrewStep], state: CrewState,
    ) -> list[tuple[CrewStep, str, bool, float]]:
        """Exécute un batch de steps en parallèle via asyncio.gather."""

        async def _run_one(step: CrewStep) -> tuple[CrewStep, str, bool, float]:
            enriched_task = self._build_enriched_task(step, state)
            t0 = time.time()
            output, success = await self._execute_step(step, enriched_task)
            return step, output, success, time.time() - t0

        raw_results = await asyncio.gather(
            *[_run_one(s) for s in steps],
            return_exceptions=True,
        )

        # Convertir les exceptions en résultats d'échec
        results = []
        for i, res in enumerate(raw_results):
            if isinstance(res, Exception):
                step = steps[i]
                logger.error(
                    "[PM %s] Step %d exception: %s",
                    self.epic_id[:8], step.index, res,
                )
                results.append((
                    step,
                    f"[Exception] {type(res).__name__}: {str(res)[:300]}",
                    False,
                    0.0,
                ))
            else:
                results.append(res)

        return results

    # ─── Résolution des dépendances ─────────────────────

    def _get_ready_steps(self) -> list[CrewStep]:
        """Retourne les steps dont toutes les dépendances sont satisfaites."""
        ready = []
        for step in self.steps:
            if step.index in self._completed_indices:
                continue  # Déjà terminé
            # Vérifier les dépendances
            if all(d in self._completed_indices for d in step.depends_on):
                ready.append(step)
        return ready

    # ─── Exécution d'un step individuel ─────────────────

    async def _execute_step(
        self, step: CrewStep, enriched_task: str,
    ) -> tuple[str, bool]:
        """Spawn un Worker pour une étape. Retourne (output, success)."""
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
                    reasoning=f"PM step {step.index + 1}",
                )
                await self.brain._learn_from_result(
                    step.description, decision, result,
                )

                if result.success:
                    return result.output, True
                return f"[Échec étape {step.index + 1}] {result.output[:500]}", False

            except Exception as e:
                logger.error(
                    "[PM %s] Step %d failed: %s",
                    self.epic_id[:8], step.index + 1, e,
                )
                return (
                    f"[Erreur étape {step.index + 1}] "
                    f"{type(e).__name__}: {str(e)[:300]}",
                    False,
                )
            finally:
                self.worker_manager.unregister(worker)

    # ─── Construction du prompt enrichi ─────────────────

    def _build_enriched_task(self, step: CrewStep, state: CrewState) -> str:
        """Construit le prompt enrichi pour un worker."""
        parts: list[str] = []

        # Demande originale
        if self.original_request:
            parts.append("=== DEMANDE ORIGINALE DE L'UTILISATEUR ===")
            parts.append(self.original_request)

        # Résultats des étapes précédentes (dont ce step dépend)
        dep_results = [
            r for r in self._results
            if r.index in step.depends_on and r.success
        ]
        if dep_results:
            parts.append("=== RÉSULTATS DES ÉTAPES DONT TU DÉPENDS ===")
            for r in dep_results:
                parts.append(
                    f"[Étape {r.index + 1} — {r.worker_type}]"
                )
                parts.append(r.output[:_MAX_STEP_OUTPUT_IN_CONTEXT])

        # S'il n'y a pas de dépendances explicites, injecter tous les résultats
        if not dep_results and self._results:
            parts.append("=== RÉSULTATS DES ÉTAPES PRÉCÉDENTES ===")
            for r in self._results:
                if r.success:
                    parts.append(
                        f"[Étape {r.index + 1} — {r.worker_type}]"
                    )
                    parts.append(r.output[:_MAX_STEP_OUTPUT_IN_CONTEXT])

        # Contexte mémoire
        if self.memory_context:
            ctx_limit = 800 if self._results else 2000
            parts.append("=== CONTEXTE MÉMOIRE ===")
            parts.append(self.memory_context[:ctx_limit])

        # Mission
        total = len(self.steps)
        parts.append(f"=== TA MISSION (Étape {step.index + 1}/{total}) ===")
        parts.append(step.description)
        parts.append(
            f"\nPROJET : « {self.epic_subject} »\n"
            "IMPORTANT : Ta mission s'inscrit dans le projet ci-dessus. "
            "Utilise les résultats des étapes précédentes comme base. "
            "Ne répète pas ce qui a déjà été fait. Avance le projet "
            "en restant fidèle à la DEMANDE ORIGINALE de l'utilisateur."
        )

        return "\n\n".join(parts)

    # ─── Synthèse finale ────────────────────────────────

    async def _synthesize(self, state: CrewState) -> str:
        """Synthèse finale par LLM depuis les résultats."""
        if not self._results:
            return "[Chef de Projet] Aucun résultat produit."

        # Construire le texte accumulé
        parts: list[str] = []
        for r in self._results:
            icon = "✅" if r.success else "❌"
            parts.append(f"[Étape {r.index + 1} — {r.worker_type}] {icon}")
            parts.append(r.output[:_MAX_STEP_OUTPUT_IN_CONTEXT])
            parts.append("")
        accumulated = "\n".join(parts)

        original_block = ""
        if self.original_request:
            original_block = (
                f"=== DEMANDE ORIGINALE DE L'UTILISATEUR ===\n"
                f"{self.original_request}\n\n"
            )

        synthesis_prompt = (
            f"Tu es le synthétiseur du projet « {self.epic_subject} ».\n\n"
            f"{original_block}"
            f"Une équipe de {len(self._results)} agents spécialisés "
            f"a travaillé EN PARALLÈLE sur ce projet. "
            f"Voici leurs résultats :\n\n"
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
            logger.error("[PM %s] Synthèse LLM échouée: %s", self.epic_id[:8], e)
            header = f"[Projet — {self.epic_subject}] {len(self._results)} étapes\n\n"
            return header + accumulated

    # ─── Utilitaires ────────────────────────────────────

    def _save_state(self, state: CrewState) -> None:
        """Sauvegarde l'état du crew en Memory."""
        if not self.memory or not self.memory.is_initialized:
            return
        try:
            import json
            state.updated_at = datetime.now().isoformat()
            source = f"{_CREW_STATE_SOURCE_PREFIX}{self.epic_id}"

            # Supprimer l'ancien
            existing = self.memory._store.search_by_source(source, limit=1)
            for record in existing:
                try:
                    self.memory._store.delete(record.id)
                except Exception:
                    pass

            self.memory.store_memory(
                content=json.dumps(state.to_dict()),
                source=source,
                tags=[
                    f"crew:{self.epic_id}",
                    "crew_state",
                    f"crew_status:{state.status}",
                ],
                importance=0.9,
                metadata={
                    "epic_id": self.epic_id,
                    "status": state.status,
                    "progress": state.progress_pct,
                    "step_count": len(state.steps),
                },
            )
        except Exception as e:
            logger.error("[PM] Save state failed: %s", e)

    def _store_step_result(self, step: CrewStep, output: str) -> None:
        """Stocke un résultat d'étape en mémoire."""
        if not self.memory or not self.memory.is_initialized:
            return
        try:
            self.memory.store_memory(
                content=output[:5000],
                source=f"crew:{self.epic_id}",
                tags=[
                    f"crew:{self.epic_id}",
                    f"step_{step.index}",
                    step.worker_type.value,
                    "crew_result",
                ],
                importance=0.8,
                metadata={
                    "epic_id": self.epic_id,
                    "step_index": step.index,
                    "worker_type": step.worker_type.value,
                    "timestamp": time.time(),
                },
            )
        except Exception as e:
            logger.debug("[PM] Store step result failed: %s", e)

    def _sync_task_registry(
        self, step_index: int, success: bool, output: str,
    ) -> None:
        """Synchronise avec le TaskRegistry."""
        if not self.memory or not self.memory.is_initialized:
            return
        try:
            registry = self.memory.task_registry
            if not registry:
                return
            epic_tasks = registry.get_epic_tasks(self.epic_id)
            if not epic_tasks:
                return
            epic_tasks.sort(key=lambda t: t.created_at)
            if step_index < len(epic_tasks):
                task = epic_tasks[step_index]
                new_status = "done" if success else "failed"
                registry.update_task_status(
                    task.id, new_status, result=output[:500],
                )
        except Exception as e:
            logger.debug("[PM] Sync TaskRegistry failed: %s", e)

    def _make_event(
        self, event_type: str, message: str, data: dict | None = None,
    ) -> CrewEvent:
        return CrewEvent(
            crew_id=self.epic_id,
            event_type=event_type,
            message=message,
            data=data or {},
        )

    def _emit_event(self, event: CrewEvent) -> None:
        if self._event_callback:
            try:
                self._event_callback(event)
            except Exception as e:
                logger.debug("[PM] Event callback failed: %s", e)
