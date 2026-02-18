"""
Brain Execution — Workers, Epics, Retry & Apprentissage
=========================================================
Pipeline d'exécution des Workers et Epics, avec retry
intelligent et boucle d'apprentissage fermée.

Chaque fonction reçoit l'instance Brain en premier argument.
Extrait de brain.py pour séparer l'exécution de l'orchestration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from neo_core.brain.prompts import BrainDecision
from neo_core.brain.teams.worker import WorkerType, WorkerResult

if TYPE_CHECKING:
    from neo_core.brain.teams.factory import TaskAnalysis

logger = logging.getLogger(__name__)


async def execute_with_worker(brain, request: str, decision: BrainDecision,
                              memory_context: str,
                              analysis: "TaskAnalysis | None" = None) -> str:
    """
    Crée, exécute et détruit un Worker pour une tâche.
    Avec retry persistant (max 3 tentatives) guidé par Memory.

    Cycle de vie garanti par tentative :
    1. Factory crée le Worker
    2. Worker enregistré dans le WorkerLifecycleManager
    3. execute() → Memory récupère l'apprentissage
    4. Brain._learn_from_result() → LearningEngine apprend
    5. Si échec → améliore la stratégie via Memory et retente
    6. Worker.cleanup() → ressources libérées
    """
    max_attempts = 3
    errors_so_far = []

    # Enregistrer la tâche dans le TaskRegistry
    task_record = None
    if brain.memory and brain.memory.is_initialized:
        try:
            task_record = brain.memory.create_task(
                description=request[:200],
                worker_type=decision.worker_type or "generic",
            )
        except Exception as e:
            logger.debug("Impossible de créer le TaskRecord: %s", e)

    for attempt in range(1, max_attempts + 1):
        # Si retry → améliorer la stratégie via Memory
        if attempt > 1:
            decision, analysis = improve_strategy(
                brain, request, decision, errors_so_far, attempt
            )
            logger.info(
                "[Brain] Retry %d/%d — stratégie: %s (%s)",
                attempt, max_attempts, decision.worker_type, decision.reasoning,
            )

        try:
            worker_type = WorkerType(decision.worker_type)
        except (ValueError, TypeError):
            worker_type = WorkerType.GENERIC

        if analysis:
            worker = brain.factory.create_worker(analysis)
        else:
            worker = brain.factory.create_worker_for_type(
                worker_type=worker_type,
                task=request,
                subtasks=decision.subtasks,
            )

        # Stage 5 : Passer le health monitor au Worker
        worker.health_monitor = brain._health

        # Mettre à jour la tâche en "in_progress"
        if task_record:
            try:
                brain.memory.update_task_status(task_record.id, "in_progress")
            except Exception as e:
                logger.debug("Impossible de mettre à jour le statut de la tâche: %s", e)

        # ── Lifecycle géré par context manager ──
        async with worker:
            brain.worker_manager.register(worker)

            try:
                result = await worker.execute()
                await learn_from_result(brain, request, decision, result)

                if result.success:
                    # Marquer la tâche comme terminée
                    if task_record:
                        try:
                            brain.memory.update_task_status(
                                task_record.id, "done",
                                result=result.output[:500],
                            )
                        except Exception as e:
                            logger.debug("Impossible de marquer la tâche comme terminée: %s", e)
                    return result.output

                # Échec → collecter l'erreur pour le prochain essai
                errors_so_far.append({
                    "attempt": attempt,
                    "worker_type": decision.worker_type,
                    "error": result.output[:200],
                    "errors": result.errors or [],
                })

            except Exception as e:
                errors_so_far.append({
                    "attempt": attempt,
                    "worker_type": decision.worker_type,
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                    "errors": [str(e)],
                })

            finally:
                brain.worker_manager.unregister(worker)

    # Toutes les tentatives épuisées
    if task_record:
        try:
            last_err = errors_so_far[-1]["error"] if errors_so_far else "inconnu"
            brain.memory.update_task_status(
                task_record.id, "failed",
                result=f"Échec après {max_attempts} tentatives: {last_err}",
            )
        except Exception as e:
            logger.debug("Impossible de marquer la tâche comme échouée: %s", e)

    last_error = errors_so_far[-1]["error"] if errors_so_far else "erreur inconnue"
    return (
        f"[Brain] Échec après {max_attempts} tentatives. "
        f"Dernière erreur: {last_error}"
    )


async def execute_as_epic(brain, request: str, decision: BrainDecision,
                          memory_context: str) -> str:
    """
    Crée un Epic dans le TaskRegistry et exécute chaque sous-tâche
    séquentiellement via des Workers individuels.

    Chaque sous-tâche est enregistrée, exécutée et trackée.
    L'Epic est marqué done/failed selon les résultats.
    """
    # 1. Créer l'Epic dans le registre
    epic = None
    if brain.memory and brain.memory.is_initialized:
        try:
            subtask_tuples = [
                (st, decision.worker_type or "generic")
                for st in decision.subtasks
            ]
            epic = brain.memory.create_epic(
                description=request[:200],
                subtask_descriptions=subtask_tuples,
                strategy=decision.reasoning,
            )
            logger.info(
                "Epic créé: %s avec %d sous-tâches",
                epic.id[:8] if epic else "?", len(decision.subtasks),
            )
        except Exception as e:
            logger.debug("Impossible de créer l'Epic: %s", e)

    # 2. Exécuter chaque sous-tâche comme un delegate_worker
    results = []
    for i, subtask in enumerate(decision.subtasks):
        sub_decision = BrainDecision(
            action="delegate_worker",
            subtasks=[subtask],
            confidence=decision.confidence,
            worker_type=decision.worker_type,
            reasoning=f"Sous-tâche {i + 1}/{len(decision.subtasks)} de l'Epic",
        )
        try:
            result = await execute_with_worker(
                brain, subtask, sub_decision, memory_context,
            )
            results.append(f"✅ {subtask[:60]}: {result[:200]}")
        except Exception as e:
            results.append(f"❌ {subtask[:60]}: {type(e).__name__}: {str(e)[:100]}")

    # 3. Compiler le résultat final
    all_ok = all(r.startswith("✅") for r in results)
    summary = "\n".join(results)

    if epic and brain.memory and brain.memory.is_initialized:
        try:
            brain.memory.update_epic_status(
                epic.id, "done" if all_ok else "failed",
            )
        except Exception as e:
            logger.debug("Impossible de mettre à jour l'Epic: %s", e)

    status = "terminé" if all_ok else "partiellement terminé"
    return f"[Epic {status}]\n{summary}"


def improve_strategy(
    brain,
    request: str,
    current_decision: BrainDecision,
    errors_so_far: list[dict],
    attempt: int,
) -> tuple[BrainDecision, Optional["TaskAnalysis"]]:
    """
    Améliore la stratégie d'exécution en se basant sur les erreurs passées
    et les conseils de Memory.

    Stratégie progressive :
    - Tentative 2 : changer de worker_type si Memory recommande un alternatif
    - Tentative 3 : simplifier la requête (moins de subtasks)
    """
    new_decision = BrainDecision(
        action=current_decision.action,
        subtasks=list(current_decision.subtasks),
        confidence=current_decision.confidence * 0.8,
        worker_type=current_decision.worker_type,
        reasoning=current_decision.reasoning,
    )
    new_analysis = None

    # Consulter Memory pour des conseils de retry
    retry_advice = None
    if brain.memory and brain.memory.is_initialized and brain.memory.learning:
        try:
            previous_errors = []
            for e in errors_so_far:
                previous_errors.extend(e.get("errors", []))
            retry_advice = brain.memory.learning.get_retry_advice(
                request,
                current_decision.worker_type or "generic",
                previous_errors,
            )
        except Exception as e:
            logger.debug("Impossible de consulter les conseils de retry: %s", e)

    if attempt == 2:
        # Stratégie 2 : Changer de worker_type si recommandé
        if retry_advice and retry_advice.get("recommended_worker"):
            alt_worker = retry_advice["recommended_worker"]
            try:
                alt_type = WorkerType(alt_worker)
                new_decision.worker_type = alt_worker
                new_decision.subtasks = brain.factory._basic_decompose(request, alt_type)
                new_decision.reasoning = (
                    f"Retry: changement {current_decision.worker_type} → {alt_worker} "
                    f"(conseil Memory)"
                )
            except ValueError as e:
                logger.debug("Type de worker recommandé invalide %s: %s", alt_worker, e)

        if new_decision.worker_type == current_decision.worker_type:
            # Pas de recommandation → simplifier la requête
            if new_decision.subtasks and len(new_decision.subtasks) > 1:
                # Garder uniquement la tâche principale
                new_decision.subtasks = [new_decision.subtasks[0]]
                new_decision.reasoning = "Retry: simplification (1 seule sous-tâche)"

    elif attempt == 3:
        # Stratégie 3 : Worker generic avec décomposition minimale
        new_decision.worker_type = "generic"
        new_decision.subtasks = [request[:200]]
        new_decision.reasoning = (
            "Retry final: worker generic, requête simplifiée"
        )

    return new_decision, new_analysis


async def learn_from_result(brain, request: str, decision: BrainDecision,
                            result: WorkerResult) -> None:
    """
    Apprentissage à partir du résultat d'un Worker.

    Boucle fermée :
    - Enregistre dans le LearningEngine (patterns d'erreur, compétences)
    - Stocke aussi un résumé en mémoire classique pour le contexte
    """
    if not brain.memory:
        return

    try:
        # 1. Enregistrer dans le LearningEngine (boucle fermée)
        brain.memory.record_execution_result(
            request=request,
            worker_type=result.worker_type,
            success=result.success,
            execution_time=result.execution_time,
            errors=result.errors,
            output=result.output[:500] if result.success else "",
        )

        # 2. Stocker aussi en mémoire classique pour le contexte
        tags = [
            "brain_learning",
            f"worker_type:{result.worker_type}",
            "success" if result.success else "failure",
            f"decision:{decision.action}",
        ]

        content = (
            f"Apprentissage Brain — {result.worker_type}\n"
            f"Requête: {request[:200]}\n"
            f"Décision: {decision.action} (confiance: {decision.confidence:.1f})\n"
            f"Résultat: {'Succès' if result.success else 'Échec'}\n"
            f"Temps: {result.execution_time:.1f}s"
        )

        if result.errors:
            content += f"\nErreurs: {'; '.join(result.errors[:3])}"

        importance = 0.5 if result.success else 0.8

        brain.memory.store_memory(
            content=content,
            source="brain_learning",
            tags=tags,
            importance=importance,
            metadata={
                "worker_type": result.worker_type,
                "decision_action": decision.action,
                "success": result.success,
                "execution_time": result.execution_time,
                "confidence": decision.confidence,
            },
        )
        # 3. Tracker l'usage des outils auto-générés (Level 4)
        if result.worker_type and result.worker_type.startswith("auto_"):
            try:
                if hasattr(brain, '_tool_generator') and brain._tool_generator:
                    brain._tool_generator.track_usage(
                        result.worker_type,
                        result.success,
                        result.execution_time,
                    )
            except Exception:
                pass  # Best-effort tracking
    except Exception as e:
        logger.debug("Impossible d'enregistrer l'apprentissage: %s", e)
