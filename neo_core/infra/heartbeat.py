"""
Heartbeat — Cycle Autonome du Système Neo Core
================================================
Pulse régulier qui rend Neo proactif.

Responsabilités :
- Surveiller les Epics/Tasks actives et en attente
- Déclencher Brain pour les tâches pending dont les dépendances sont résolues
- Notifier Vox pour informer l'utilisateur du progrès
- Lancer la consolidation Memory en arrière-plan
- Détecter les tâches bloquées (stale) et déclencher des retries

Architecture :
- asyncio.Task en arrière-plan
- Pulse toutes les N secondes (configurable)
- Chaque pulse : check tasks → notify → consolidate
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional, Callable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neo_core.brain.core import Brain
    from neo_core.memory.agent import MemoryAgent
    from neo_core.vox.interface import Vox


# ─── Configuration ───────────────────────────────────────

@dataclass
class HeartbeatConfig:
    """Configuration du heartbeat."""
    interval_seconds: float = 300.0  # Intervalle entre les pulses (5 minutes)
    stale_task_minutes: float = 15.0  # Tâche considérée "stale" après N minutes
    max_auto_tasks_per_pulse: int = 6  # Max tâches auto-lancées par pulse (permet le parallélisme crew)
    auto_consolidation: bool = True  # Consolider Memory automatiquement
    consolidation_interval_pulses: int = 60  # Consolider toutes les 60 pulses (~5h)
    auto_tuning: bool = True  # Lancer l'auto-tuning périodiquement
    auto_tuning_interval_pulses: int = 30  # Toutes les 30 pulses (~2h30)
    self_patching: bool = True  # Level 3 — auto-correction comportementale
    self_patching_interval_pulses: int = 60  # Toutes les 60 pulses (~5h)
    tool_generation: bool = True  # Level 4 — création autonome d'outils
    tool_generation_interval_pulses: int = 90  # Toutes les 90 pulses (~7h30)
    enabled: bool = True


# ─── Événements du Heartbeat ──────────────────────────────

@dataclass
class HeartbeatEvent:
    """Un événement généré par le heartbeat."""
    event_type: str  # "task_started" | "task_stale" | "epic_progress" | "epic_done" | "consolidation"
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: dict = field(default_factory=dict)


# ─── Heartbeat Manager ───────────────────────────────────

class HeartbeatManager:
    """
    Gestionnaire du cycle autonome de Neo.

    Lance un asyncio.Task en arrière-plan qui pulse à intervalle régulier.
    Chaque pulse vérifie les tâches, déclenche Brain si nécessaire,
    et notifie l'utilisateur via un callback.
    """

    def __init__(
        self,
        brain: Brain,
        memory: MemoryAgent,
        config: HeartbeatConfig | None = None,
        on_event: Callable[[HeartbeatEvent], None] | None = None,
        vox: object | None = None,
    ):
        self.brain = brain
        self.memory = memory
        self.config = config or HeartbeatConfig()
        self._on_event = on_event
        self._vox = vox  # Référence Vox pour push_message() proactif

        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._pulse_count = 0
        self._events: list[HeartbeatEvent] = []
        self._auto_tuner = None  # Initialisé au premier pulse si disponible
        self._self_patcher = None  # Level 3 — lazy init
        self._tool_generator = None  # Level 4 — lazy init
        self._consecutive_failures: int = 0  # Compteur d'échecs consécutifs
        self._last_error: Optional[str] = None  # Dernier message d'erreur
        self._last_proactive_at: float = 0.0  # Timestamp du dernier message proactif
        self._proactive_cooldown: float = 3600.0  # 1 heure entre chaque message proactif

    # ─── Lifecycle ────────────────────────────────────

    def start(self) -> None:
        """
        Démarre le heartbeat en arrière-plan.
        Nécessite une event loop active (appelé depuis un contexte async).
        """
        if self._running or not self.config.enabled:
            return

        self._running = True
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._pulse_loop())
        except RuntimeError:
            # Pas d'event loop active — on marque comme running
            # mais la boucle ne démarre pas (sera gérée manuellement)
            self._task = None

        self._emit(HeartbeatEvent(
            event_type="heartbeat_started",
            message="Heartbeat démarré",
        ))
        logger.info("[Heartbeat] Démarré (interval: %.1fs)", self.config.interval_seconds)

    def stop(self) -> None:
        """Arrête le heartbeat."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._emit(HeartbeatEvent(
            event_type="heartbeat_stopped",
            message="Heartbeat arrêté",
        ))
        logger.info("[Heartbeat] Arrêté après %d pulses", self._pulse_count)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def pulse_count(self) -> int:
        return self._pulse_count

    @property
    def recent_events(self) -> list[HeartbeatEvent]:
        """Retourne les 20 derniers événements."""
        return self._events[-20:]

    # ─── Boucle principale ────────────────────────────

    async def _pulse_loop(self) -> None:
        """Boucle principale du heartbeat."""
        while self._running:
            try:
                await asyncio.sleep(self.config.interval_seconds)
                if not self._running:
                    break

                await self._pulse()
                self._pulse_count += 1
                self._consecutive_failures = 0  # Reset sur succès
                self._last_error = None

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._consecutive_failures += 1
                self._last_error = f"{type(e).__name__}: {e}"
                logger.error(
                    "[Heartbeat] Erreur pulse #%d (échec consécutif %d): %s",
                    self._pulse_count, self._consecutive_failures, e,
                )
                # Ne pas crasher le heartbeat, juste log et continue

    async def _pulse(self) -> None:
        """
        Un seul pulse du heartbeat.

        Étapes :
        1. Vérifier les Epics actifs → avancer les tâches pending
        2. Détecter les tâches stale → alerter
        3. Consolider Memory si nécessaire
        """
        if not self.memory:
            logger.warning("[Heartbeat] Memory non disponible, pulse ignoré")
            return

        if not self.memory.is_initialized:
            logger.warning("[Heartbeat] Memory non initialisé, tentative de réinit...")
            try:
                self.memory.initialize()
                logger.info("[Heartbeat] Memory réinitialisé avec succès")
            except Exception as e:
                logger.error("[Heartbeat] Réinit Memory échouée: %s — pulse ignoré", e)
                return

        registry = self.memory.task_registry
        if not registry:
            logger.warning("[Heartbeat] TaskRegistry non disponible, pulse ignoré")
            return

        # 1. Avancer les Epics actifs
        await self._advance_epics(registry)

        # 1b. Déclencher les cycles des projets récurrents
        await self._trigger_recurring_cycles(registry)

        # 2. Détecter les tâches stale
        self._detect_stale_tasks(registry)

        # 2b. Nettoyage des tâches anciennes terminées (toutes les 5 pulses)
        if self._pulse_count > 0 and self._pulse_count % 5 == 0:
            try:
                deleted = registry.cleanup_completed(max_age_hours=48.0)
                if deleted > 0:
                    logger.info("[Heartbeat] Nettoyage: %d tâches anciennes supprimées", deleted)
            except Exception as e:
                logger.debug("[Heartbeat] Cleanup échoué: %s", e)

        # 3. Consolidation Memory périodique
        if (self.config.auto_consolidation and
                self._pulse_count > 0 and
                self._pulse_count % self.config.consolidation_interval_pulses == 0):
            self._consolidate_memory()

        # 4. Pensée proactive — Brain parle de lui-même
        await self._brain_proactive_think(registry)

        # 5. Auto-réflexion de la personnalité (Stage 9)
        await self._perform_personality_reflection()

        # 6. State snapshot périodique pour Guardian (Stage 10)
        if self._pulse_count > 0 and self._pulse_count % 10 == 0:
            self._save_guardian_state()

        # 7. Auto-tuning périodique (Level 1 self-improvement)
        from neo_core.features import feature_enabled
        if (self.config.auto_tuning and feature_enabled("auto_tuning") and
                self._pulse_count > 0 and
                self._pulse_count % self.config.auto_tuning_interval_pulses == 0):
            self._run_auto_tuning()

        # 8. Self-patching périodique (Level 3 — auto-correction)
        if (self.config.self_patching and feature_enabled("self_patching") and
                self._pulse_count > 0 and
                self._pulse_count % self.config.self_patching_interval_pulses == 0):
            self._run_self_patching()

        # 9. Tool generation périodique (Level 4 — création d'outils)
        if (self.config.tool_generation and feature_enabled("tool_generation") and
                self._pulse_count > 0 and
                self._pulse_count % self.config.tool_generation_interval_pulses == 0):
            self._run_tool_generation()

    # ─── Avancement des Epics ─────────────────────────

    async def _advance_epics(self, registry) -> None:
        """
        Vérifie les Epics in_progress et gère les cas spéciaux.

        v3 : Le ProjectManagerWorker gère l'exécution parallèle.
        Le heartbeat ne fait plus avancer les steps — il vérifie :
        1. Les projets stale/bloqués (PM a crashé) → relance un PM
        2. Les epics non-crew (tâches standalone) → exécution séquentielle
        3. Les epics terminées → mise à jour du statut
        """
        epics = registry.get_all_epics(limit=10)
        tasks_launched = 0

        for epic in epics:
            if epic.status not in ("pending", "in_progress"):
                continue

            # ── Vérifier l'état du crew (si PM en cours) ──
            try:
                from neo_core.brain.teams.crew import CrewExecutor
                executor = CrewExecutor(brain=self.brain)
                crew_state = executor.load_state(epic.id)

                if crew_state:
                    if crew_state.status == "active":
                        # PM en cours d'exécution → ne pas interférer
                        continue

                    if crew_state.status == "paused":
                        # En pause par l'utilisateur → ne pas toucher
                        continue

                    if crew_state.status == "done":
                        # Terminé → s'assurer que l'epic est à jour
                        if epic.status != "done":
                            registry.update_epic_status(epic.id, "done")
                            self._emit(HeartbeatEvent(
                                event_type="epic_done",
                                message=f"Projet terminé: {epic.description[:60]}",
                                data={"epic_id": epic.id, "success": True},
                            ))
                        continue

                    if crew_state.status == "failed":
                        if epic.status != "failed":
                            registry.update_epic_status(epic.id, "failed")
                        continue

            except Exception as e:
                logger.debug("[Heartbeat] Crew check failed for %s: %s", epic.id[:8], e)

            # ── Epics non-crew (tâches standalone) ──
            epic_tasks = registry.get_epic_tasks(epic.id)
            if not epic_tasks:
                continue

            # Si l'epic est "pending" et a des tâches, le passer en "in_progress"
            if epic.status == "pending":
                registry.update_epic_status(epic.id, "in_progress")
                self._emit(HeartbeatEvent(
                    event_type="epic_started",
                    message=f"Epic démarré: {epic.description[:60]}",
                    data={"epic_id": epic.id},
                ))

            # Trouver les tâches pending prêtes à être lancées
            has_active = any(t.status == "in_progress" for t in epic_tasks)
            if has_active:
                continue

            pending_tasks = [t for t in epic_tasks if t.status == "pending"]
            if not pending_tasks:
                # Vérifier si l'epic est terminé
                all_terminal = all(t.is_terminal for t in epic_tasks)
                if all_terminal:
                    all_done = all(t.status == "done" for t in epic_tasks)
                    done_count = sum(1 for t in epic_tasks if t.status == "done")
                    total = len(epic_tasks)
                    self._emit(HeartbeatEvent(
                        event_type="epic_done",
                        message=(
                            f"Epic terminé: {epic.description[:60]} "
                            f"({done_count}/{total} réussies)"
                        ),
                        data={"epic_id": epic.id, "success": all_done},
                    ))
                continue

            if tasks_launched >= self.config.max_auto_tasks_per_pulse:
                continue

            # Lancer la prochaine tâche pending
            next_task = pending_tasks[0]
            tasks_launched += 1

            self._emit(HeartbeatEvent(
                event_type="task_auto_started",
                message=f"Tâche auto-lancée: {next_task.description[:60]}",
                data={"task_id": next_task.id, "epic_id": epic.id},
            ))

            # Exécuter via Brain
            await self._execute_task(next_task)

    async def _execute_task(self, task) -> None:
        """
        Exécute une tâche via Brain avec timeout de 5 minutes.
        Met à jour le statut dans le registry.
        """
        from neo_core.brain.core import BrainDecision

        try:
            # Mettre à jour le statut
            self.memory.update_task_status(task.id, "in_progress")

            # Construire la décision pour Brain
            decision = BrainDecision(
                action="delegate_worker",
                worker_type=task.worker_type,
                subtasks=[task.description],
                confidence=0.7,
                reasoning=f"Heartbeat: tâche auto-lancée (epic {task.epic_id[:8] if task.epic_id else 'N/A'})",
            )

            memory_context = self.brain.get_memory_context(task.description)

            # Timeout de 5 minutes pour éviter qu'un worker bloque le heartbeat
            result = await asyncio.wait_for(
                self.brain._execute_with_worker(
                    request=task.description,
                    decision=decision,
                    memory_context=memory_context,
                    existing_task_id=task.id,
                ),
                timeout=300.0,
            )

            # Déterminer si c'est un succès
            if result and not result.startswith("[Brain] Échec") and not result.startswith("[Worker"):
                self.memory.update_task_status(task.id, "done", result=result[:500])
                self._emit(HeartbeatEvent(
                    event_type="task_completed",
                    message=f"Tâche terminée: {task.description[:60]}",
                    data={"task_id": task.id, "success": True},
                ))
            else:
                self.memory.update_task_status(task.id, "failed", result=result[:500])
                self._emit(HeartbeatEvent(
                    event_type="task_failed",
                    message=f"Tâche échouée: {task.description[:60]}",
                    data={"task_id": task.id, "success": False, "error": result[:200]},
                ))

        except asyncio.TimeoutError:
            logger.error("[Heartbeat] Tâche %s timeout (>5min)", task.id[:8])
            try:
                self.memory.update_task_status(
                    task.id, "failed",
                    result="Timeout: tâche dépassée (>5 minutes)",
                )
            except Exception as e:
                logger.warning("Failed to update task status after timeout: %s", e)

        except Exception as e:
            logger.error("[Heartbeat] Erreur exécution tâche %s: %s", task.id[:8], e)
            try:
                self.memory.update_task_status(
                    task.id, "failed",
                    result=f"{type(e).__name__}: {str(e)[:200]}",
                )
            except Exception as e:
                logger.warning("Failed to update task status after execution error: %s", e)

    # ─── Projets récurrents (cycliques) ────────────────

    async def _trigger_recurring_cycles(self, registry) -> None:
        """
        Vérifie les projets récurrents et déclenche un nouveau cycle si dû.

        Pour chaque projet récurrent dont le next_cycle_at est dépassé :
        1. Crée de nouvelles tâches à partir du template
        2. Lance le ProjectManager pour exécuter le cycle
        3. Stocke le résumé du cycle et calcule le prochain
        """
        try:
            due_epics = registry.get_recurring_epics_due()
        except Exception as e:
            logger.debug("[Heartbeat] get_recurring_epics_due failed: %s", e)
            return

        for epic in due_epics:
            logger.info(
                "[Heartbeat] Projet récurrent dû: %s (%s) — cycle %d",
                epic.short_id, epic.display_name[:40], epic.cycle_count + 1,
            )

            self._emit(HeartbeatEvent(
                event_type="recurring_cycle_started",
                message=f"🔁 Cycle {epic.cycle_count + 1} lancé: {epic.display_name[:50]}",
                data={"epic_id": epic.id, "cycle": epic.cycle_count + 1},
            ))

            try:
                await self._execute_recurring_cycle(epic, registry)
            except Exception as e:
                logger.error(
                    "[Heartbeat] Recurring cycle failed for %s: %s",
                    epic.short_id, e,
                )
                self._emit(HeartbeatEvent(
                    event_type="recurring_cycle_failed",
                    message=f"❌ Cycle {epic.cycle_count + 1} échoué: {epic.display_name[:50]}",
                    data={"epic_id": epic.id, "error": str(e)[:200]},
                ))

    async def _execute_recurring_cycle(self, epic, registry) -> None:
        """Exécute un cycle complet d'un projet récurrent."""
        from neo_core.brain.teams.project_manager import ProjectManagerWorker
        from neo_core.brain.teams.worker import WorkerType

        # Préparer le cycle : créer les tâches à partir du template
        advanced_epic = registry.advance_recurring_cycle(
            epic.id,
            cycle_summary="(cycle en cours)",  # placeholder, mis à jour après
        )
        if not advanced_epic:
            logger.warning("[Heartbeat] advance_recurring_cycle returned None for %s", epic.id)
            return

        # Construire les étapes Crew à partir du template
        from neo_core.brain.teams.crew import CrewStep
        crew_steps = []
        for i, step_tmpl in enumerate(epic.cycle_template):
            try:
                wt = WorkerType(step_tmpl.get("worker_type", "generic"))
            except ValueError:
                wt = WorkerType.GENERIC
            depends = step_tmpl.get("depends_on", [])
            crew_steps.append(CrewStep(
                description=step_tmpl.get("description", f"Étape {i+1}"),
                worker_type=wt,
                depends_on=depends,
            ))

        if not crew_steps:
            logger.warning("[Heartbeat] No crew_steps for recurring epic %s", epic.short_id)
            return

        # Construire le contexte mémoire enrichi avec les résultats des cycles précédents
        memory_context = ""
        if self.brain:
            memory_context = self.brain.get_memory_context(epic.description)

        # Injecter les résultats des cycles précédents dans le contexte
        if advanced_epic.accumulated_results:
            prev_results = "\n---\n".join(advanced_epic.accumulated_results[-5:])
            memory_context += (
                f"\n\n=== RÉSULTATS DES CYCLES PRÉCÉDENTS (projet récurrent) ===\n"
                f"Objectif: {epic.goal}\n"
                f"Cycle actuel: {advanced_epic.cycle_count}\n"
                f"Résultats précédents:\n{prev_results}\n"
                f"=== ADAPTE ta stratégie en fonction de ces résultats ===\n"
            )

        # Mettre l'epic en in_progress
        registry.update_epic_status(epic.id, "in_progress")

        # Lancer le ProjectManager
        pm = ProjectManagerWorker(
            brain=self.brain,
            epic_id=epic.id,
            epic_subject=epic.description,
            steps=crew_steps,
            memory_context=memory_context,
            original_request=epic.description,
            event_callback=self.brain._handle_crew_event if self.brain else None,
        )

        try:
            pm_result = await asyncio.wait_for(pm.execute(), timeout=600.0)  # 10 min max par cycle

            # Stocker le vrai résumé du cycle
            cycle_summary = pm_result.synthesis[:500] if pm_result.synthesis else "(pas de synthèse)"
            # Mettre à jour le résumé (remplacer le placeholder)
            epic_data, record_id = registry._find_epic_with_id(epic.id)
            if epic_data and record_id:
                updated_epic = Epic.from_dict(epic_data)
                if updated_epic.accumulated_results:
                    updated_epic.accumulated_results[-1] = cycle_summary
                updated_epic.status = "pending"  # Prêt pour le prochain cycle
                import json
                registry.store.update(record_id, content=json.dumps(updated_epic.to_dict()))

            # Notifier le résultat
            self._emit(HeartbeatEvent(
                event_type="recurring_cycle_done",
                message=f"✅ Cycle {advanced_epic.cycle_count} terminé: {epic.display_name[:50]}",
                data={
                    "epic_id": epic.id,
                    "cycle": advanced_epic.cycle_count,
                    "synthesis": cycle_summary[:200],
                },
            ))

            # Délivrer le résultat dans le chat si callback disponible
            if self.brain and hasattr(self.brain, '_action_result_callback') and self.brain._action_result_callback:
                self.brain._deliver_action_result(
                    f"🔁 **Cycle {advanced_epic.cycle_count} — {epic.display_name}**\n\n{pm_result.synthesis[:800]}"
                )

        except asyncio.TimeoutError:
            logger.error("[Heartbeat] Recurring cycle timeout for %s", epic.short_id)
            registry.update_epic_status(epic.id, "pending")  # Retry au prochain pulse
        except Exception as e:
            logger.error("[Heartbeat] Recurring cycle execution failed: %s", e)
            registry.update_epic_status(epic.id, "pending")

    # ─── Détection des tâches stale ───────────────────

    def _detect_stale_tasks(self, registry) -> None:
        """
        Détecte les tâches in_progress depuis trop longtemps.
        Émet un événement pour alerter l'utilisateur.
        """
        active_tasks = registry.get_active_tasks()
        stale_threshold = timedelta(minutes=self.config.stale_task_minutes)
        now = datetime.now()

        for task in active_tasks:
            try:
                created = datetime.fromisoformat(task.created_at)
                if now - created > stale_threshold:
                    self._emit(HeartbeatEvent(
                        event_type="task_stale",
                        message=(
                            f"Tâche bloquée depuis {self.config.stale_task_minutes:.0f}min: "
                            f"{task.description[:60]}"
                        ),
                        data={"task_id": task.id, "age_minutes": (now - created).total_seconds() / 60},
                    ))
            except (ValueError, TypeError) as e:
                logger.debug("Failed to parse task timestamp for stale detection: %s", e)

    # ─── Consolidation Memory ─────────────────────────

    def _consolidate_memory(self) -> None:
        """Lance une consolidation Memory en arrière-plan."""
        try:
            report = self.memory.consolidate()
            if report.get("deleted", 0) > 0 or report.get("merged", 0) > 0:
                self._emit(HeartbeatEvent(
                    event_type="consolidation",
                    message=(
                        f"Consolidation Memory: "
                        f"{report.get('deleted', 0)} supprimés, "
                        f"{report.get('merged', 0)} fusionnés"
                    ),
                    data=report,
                ))
        except Exception as e:
            logger.error("[Heartbeat] Erreur consolidation: %s", e)

    # ─── Auto-tuning (Level 1 self-improvement) ────────

    def _run_auto_tuning(self) -> None:
        """Lance un cycle d'auto-tuning basé sur les performances."""
        try:
            # Lazy init de l'AutoTuner
            if self._auto_tuner is None:
                from neo_core.config import NeoConfig
                config = NeoConfig()
                data_dir = config.data_dir

                # Vérifier si le LearningEngine est disponible
                if hasattr(self.memory, 'learning_engine') and self.memory.learning_engine:
                    from neo_core.brain.auto_tuner import AutoTuner
                    self._auto_tuner = AutoTuner(data_dir, self.memory.learning_engine)
                    logger.info("[Heartbeat] AutoTuner initialisé")
                else:
                    logger.debug("[Heartbeat] LearningEngine non disponible, auto-tuning ignoré")
                    return

            changes = self._auto_tuner.run_tuning_cycle()
            if changes:
                self._emit(HeartbeatEvent(
                    event_type="auto_tuning",
                    message=f"Auto-tuning: {len(changes)} paramètre(s) ajusté(s)",
                    data={"changes": changes},
                ))
                logger.info("[Heartbeat] Auto-tuning: %d changements", len(changes))
            else:
                logger.debug("[Heartbeat] Auto-tuning: aucun changement")
        except Exception as e:
            logger.error("[Heartbeat] Erreur auto-tuning: %s", e)

    # ─── Self-patching (Level 3 self-improvement) ──────

    def _run_self_patching(self) -> None:
        """Lance un cycle de self-patching : détection → génération → validation → évaluation."""
        try:
            # Lazy init du SelfPatcher
            if self._self_patcher is None:
                from neo_core.config import NeoConfig
                config = NeoConfig()

                if hasattr(self.memory, 'learning_engine') and self.memory.learning_engine:
                    from neo_core.brain.self_patcher import SelfPatcher
                    self._self_patcher = SelfPatcher(config.data_dir, self.memory.learning_engine)
                    logger.info("[Heartbeat] SelfPatcher initialisé")
                else:
                    logger.debug("[Heartbeat] LearningEngine non disponible, self-patching ignoré")
                    return

            # 1. Détecter les patterns d'erreurs récurrents
            patterns = self._self_patcher.detect_patchable_patterns()

            # 2. Générer et valider les patches
            new_patches = 0
            for pattern in patterns[:3]:  # Max 3 patches par cycle
                patch = self._self_patcher.generate_patch(pattern)
                if patch and self._self_patcher.validate_patch(patch):
                    new_patches += 1

            # 3. Évaluer et rollback les patches inefficaces
            rolled_back = self._self_patcher.evaluate_and_rollback_all()

            if new_patches or rolled_back:
                self._emit(HeartbeatEvent(
                    event_type="self_patching",
                    message=f"Self-patching: {new_patches} nouveau(x), {rolled_back} rollback(s)",
                    data={"new_patches": new_patches, "rolled_back": rolled_back},
                ))
                logger.info("[Heartbeat] Self-patching: %d nouveaux, %d rollback", new_patches, rolled_back)
            else:
                logger.debug("[Heartbeat] Self-patching: aucun changement")
        except Exception as e:
            logger.error("[Heartbeat] Erreur self-patching: %s", e)

    # ─── Tool generation (Level 4 self-improvement) ───

    def _run_tool_generation(self) -> None:
        """Lance un cycle de tool generation : détection → génération → validation → déploiement."""
        try:
            # Lazy init du ToolGenerator
            if self._tool_generator is None:
                from neo_core.config import NeoConfig
                config = NeoConfig()

                if hasattr(self.memory, 'learning_engine') and self.memory.learning_engine:
                    from neo_core.brain.tools.tool_generator import ToolGenerator

                    # Essayer de récupérer le PluginLoader s'il existe
                    plugin_loader = None
                    try:
                        from neo_core.brain.tools.plugin_loader import PluginLoader
                        plugin_loader = PluginLoader(config.data_dir)
                    except Exception as e:
                        logger.debug("PluginLoader init failed (optional): %s", e)

                    self._tool_generator = ToolGenerator(
                        config.data_dir,
                        self.memory.learning_engine,
                        plugin_loader,
                    )
                    logger.info("[Heartbeat] ToolGenerator initialisé")
                else:
                    logger.debug("[Heartbeat] LearningEngine non disponible, tool generation ignoré")
                    return

            # 1. Détecter les opportunités
            opportunities = self._tool_generator.detect_opportunities()

            # 2. Générer, valider et déployer
            deployed = 0
            for pattern in opportunities[:2]:  # Max 2 outils par cycle
                tool = self._tool_generator.generate_plugin(pattern)
                if tool and self._tool_generator.validate_plugin(tool):
                    if self._tool_generator.deploy_plugin(tool):
                        deployed += 1

            # 3. Pruning des outils inutilisés
            pruned = self._tool_generator.prune_unused()

            if deployed or pruned:
                self._emit(HeartbeatEvent(
                    event_type="tool_generation",
                    message=f"Tool generation: {deployed} déployé(s), {pruned} nettoyé(s)",
                    data={"deployed": deployed, "pruned": pruned},
                ))
                logger.info("[Heartbeat] Tool generation: %d déployés, %d nettoyés", deployed, pruned)
            else:
                logger.debug("[Heartbeat] Tool generation: aucun changement")
        except Exception as e:
            logger.error("[Heartbeat] Erreur tool generation: %s", e)

    # ─── Pensée proactive Brain ────────────────────────

    _PROACTIVE_PROMPT = """Tu es le cerveau autonome de Neo, un système IA multi-agents.
Tu observes en continu les projets et l'activité de l'utilisateur.

Heure actuelle : {current_time}
Dernier échange avec l'utilisateur : {last_interaction}

=== PROJETS ACTIFS ===
{projects_context}

=== ÉVÉNEMENTS RÉCENTS ===
{recent_events}

=== DERNIERS MESSAGES PROACTIFS ===
{recent_proactive}

Ta mission : décider si tu dois PARLER SPONTANÉMENT à l'utilisateur.

PARLE si :
- Un projet a bien avancé et mérite un update
- Tu as une idée, suggestion ou insight utile sur un projet en cours
- Un problème est détecté (tâche bloquée, erreur, incohérence)
- Tu as une réflexion pertinente sur le travail en cours
- Le silence dure depuis longtemps et un check-in serait bienvenu
- Tu veux encourager, féliciter, ou simplement discuter

NE PARLE PAS si :
- Rien de nouveau depuis la dernière fois
- Tu as déjà envoyé un message récemment sur le même sujet
- L'info est triviale ou évidente
- L'utilisateur est probablement occupé (dernière interaction < 10 min)

STYLE : Sois naturel, direct, comme un collègue intelligent. Pas de formalité.
Tutoie l'utilisateur. Sois concis (1-3 phrases max).

Réponds en JSON strict :
{{"should_speak": true, "message": "ton message naturel ici", "reason": "pourquoi tu parles"}}
ou
{{"should_speak": false, "reason": "pourquoi tu ne parles pas"}}"""

    async def _brain_proactive_think(self, registry) -> None:
        """
        Moteur de pensées autonomes — Brain décide s'il veut parler.

        Appelé à chaque pulse du heartbeat. Vérifie le cooldown (1h),
        construit un contexte compact, et demande à Haiku si Brain
        a quelque chose à dire. Si oui, diffuse via Vox.push_message().
        """
        import time as _time

        # Cooldown : pas plus d'1 message par heure
        now = _time.time()
        if now - self._last_proactive_at < self._proactive_cooldown:
            return

        # Pas de pensée proactive si Brain n'est pas dispo
        if not self.brain:
            return

        try:
            # 1. Contexte projets actifs
            projects_lines = []
            try:
                epics = registry.get_all_epics(limit=5)
                for e in epics:
                    if e.status in ("pending", "in_progress"):
                        tasks = registry.get_epic_tasks(e.id)
                        done = sum(1 for t in tasks if t.status == "done")
                        total = len(tasks)
                        projects_lines.append(
                            f"- #{e.short_id} {e.display_name[:50]} "
                            f"({e.status}) — {done}/{total} tâches"
                        )
                        # Dernière tâche terminée
                        done_tasks = [t for t in tasks if t.status == "done"]
                        if done_tasks:
                            last = done_tasks[-1]
                            projects_lines.append(
                                f"  Dernière: {last.description[:60]} → {(last.result or '')[:80]}"
                            )
                        # Tâches bloquées/failed
                        failed = [t for t in tasks if t.status == "failed"]
                        if failed:
                            projects_lines.append(
                                f"  ⚠ {len(failed)} tâche(s) en échec"
                            )
            except Exception:
                pass

            if not projects_lines:
                projects_lines.append("Aucun projet actif.")

            # 2. Événements récents
            recent_events = self._events[-5:] if self._events else []
            events_lines = [
                f"- [{e.event_type}] {e.message[:80]}"
                for e in recent_events
            ] or ["Aucun événement récent."]

            # 3. Dernière interaction utilisateur
            last_interaction = "inconnue"
            try:
                if self.memory and self.memory.is_initialized:
                    wm = self.memory.get_working_context()
                    if wm:
                        last_interaction = wm[:100]
                    else:
                        last_interaction = "pas de contexte récent"
            except Exception:
                pass

            # 4. Derniers messages proactifs (éviter les répétitions)
            recent_proactive = "Aucun message proactif récent."
            try:
                if self.memory and self.memory.is_initialized:
                    proactive_records = self.memory._store.search_by_tags(
                        ["proactive"], limit=3,
                    )
                    if proactive_records:
                        recent_proactive = "\n".join(
                            f"- {r.content[:100]}" for r in proactive_records
                        )
            except Exception:
                pass

            # 5. Construire le prompt
            prompt = self._PROACTIVE_PROMPT.format(
                current_time=datetime.now().strftime("%H:%M (%A)"),
                last_interaction=last_interaction,
                projects_context="\n".join(projects_lines),
                recent_events="\n".join(events_lines),
                recent_proactive=recent_proactive,
            )

            # 6. Appel LLM (Haiku, rapide)
            from neo_core.brain.providers.router import route_chat
            response = await asyncio.wait_for(
                route_chat(
                    agent_name="vox",  # Haiku — rapide et économique
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=250,
                    temperature=0.7,
                ),
                timeout=5.0,
            )

            if not response.text or response.text.startswith("[Erreur"):
                return

            # 7. Parser la réponse JSON
            data = self.brain._parse_json_response(response.text)
            should_speak = data.get("should_speak", False)

            if not should_speak:
                logger.debug(
                    "[Heartbeat Proactif] Brain ne parle pas: %s",
                    data.get("reason", "?")[:60],
                )
                return

            message = data.get("message", "")
            reason = data.get("reason", "")

            if not message:
                return

            # 8. Diffuser le message
            self._last_proactive_at = now

            # Via Vox.push_message() si disponible
            if self._vox and hasattr(self._vox, "push_message"):
                self._vox.push_message(message, source="proactive_think")
            else:
                # Fallback : Telegram direct + event
                from neo_core.infra.registry import core_registry
                core_registry.send_telegram(f"🧠 {message}")

            # Émettre un event heartbeat
            self._emit(HeartbeatEvent(
                event_type="brain_proactive",
                message=f"[Proactif] {message[:120]}",
                data={"message": message, "reason": reason},
            ))

            logger.info(
                "[Heartbeat] Brain parle: %s (raison: %s)",
                message[:80], reason[:40],
            )

        except asyncio.TimeoutError:
            logger.debug("[Heartbeat Proactif] LLM timeout (>5s)")
        except Exception as e:
            logger.debug("[Heartbeat Proactif] Erreur: %s", e)

    # ─── Auto-réflexion personnalité (Stage 9) ────────

    async def _perform_personality_reflection(self) -> None:
        """
        Effectue une auto-réflexion de la personnalité si c'est l'heure.
        Vérifie via Memory.should_self_reflect() (intervalle 24h).
        """
        if not self.memory:
            return

        try:
            if not self.memory.should_self_reflect():
                return

            result = await self.memory.perform_self_reflection()

            if result.get("success"):
                self._emit(HeartbeatEvent(
                    event_type="persona_reflection",
                    message=(
                        f"Auto-réflexion: {result.get('traits_updated', 0)} traits ajustés, "
                        f"{result.get('observations_recorded', 0)} observations — "
                        f"{result.get('summary', '')[:100]}"
                    ),
                    data=result,
                ))
                logger.info("[Heartbeat] Auto-réflexion effectuée: %s", result.get("summary", ""))
        except Exception as e:
            logger.error("[Heartbeat] Erreur auto-réflexion: %s", e)

    # ─── State snapshot pour Guardian (Stage 10) ──────

    def _save_guardian_state(self) -> None:
        """Sauvegarde un state snapshot périodique pour le Guardian."""
        try:
            from neo_core.infra.guardian import StateSnapshot
            from pathlib import Path

            state_dir = Path("data/guardian")
            registry = self.memory.task_registry if self.memory else None
            active_tasks = []
            if registry:
                try:
                    active = registry.get_active_tasks()
                    active_tasks = [t.id for t in active[:10]]
                except Exception as e:
                    logger.debug("Failed to get active tasks for guardian snapshot: %s", e)

            snapshot = StateSnapshot(
                heartbeat_pulse_count=self._pulse_count,
                active_tasks=active_tasks,
                shutdown_reason="heartbeat_snapshot",
            )
            snapshot.save(state_dir)
            logger.debug("[Heartbeat] State snapshot sauvegardé (pulse #%d)", self._pulse_count)
        except Exception as e:
            logger.error("[Heartbeat] Erreur state snapshot: %s", e)

    # ─── Émission d'événements ────────────────────────

    # Événements notifiés sur Telegram (les plus importants)
    _TELEGRAM_NOTIFY_EVENTS = {
        "task_completed", "task_failed", "epic_done",
        "task_stale", "self_patching", "tool_generation",
        "crew_step_advanced", "orchestrator_replan",
    }

    def _emit(self, event: HeartbeatEvent) -> None:
        """Émet un événement et appelle le callback si défini."""
        self._events.append(event)
        # Garder max 100 événements en mémoire
        if len(self._events) > 100:
            self._events = self._events[-100:]

        if self._on_event:
            try:
                self._on_event(event)
            except Exception as e:
                logger.debug("Event callback failed for %s: %s", event.event_type, e)

        # Notifier sur Telegram les événements importants
        if event.event_type in self._TELEGRAM_NOTIFY_EVENTS:
            try:
                from neo_core.infra.registry import core_registry
                core_registry.send_telegram(f"🧠 {event.message}")
            except Exception as e:
                logger.debug("Telegram notification failed for %s: %s", event.event_type, e)

    # ─── Statut ───────────────────────────────────────

    def get_status(self) -> dict:
        """Retourne le statut du heartbeat."""
        status = {
            "running": self._running,
            "healthy": self._consecutive_failures == 0,
            "pulse_count": self._pulse_count,
            "interval": self.config.interval_seconds,
            "recent_events": len(self._events),
            "last_event": self._events[-1].message if self._events else "aucun",
            "consecutive_failures": self._consecutive_failures,
        }
        if self._last_error:
            status["last_error"] = self._last_error
        return status

    def get_progress_report(self) -> str:
        """
        Génère un rapport de progrès lisible pour l'utilisateur.
        Appelé par Vox quand l'utilisateur demande /heartbeat.
        """
        if not self.memory or not self.memory.is_initialized:
            return "Heartbeat inactif (Memory non initialisé)"

        registry = self.memory.task_registry
        if not registry:
            return "Heartbeat inactif (pas de TaskRegistry)"

        lines = []

        # Résumé global
        summary = registry.get_summary()
        lines.append(
            f"Pulse #{self._pulse_count} — "
            f"{summary.get('total_tasks', 0)} tâches, "
            f"{summary.get('total_epics', 0)} epics"
        )

        # Epics actifs
        epics = registry.get_all_epics(limit=5)
        active_epics = [e for e in epics if e.status in ("pending", "in_progress")]
        if active_epics:
            lines.append("\nEpics actifs:")
            for epic in active_epics:
                epic_tasks = registry.get_epic_tasks(epic.id)
                done = sum(1 for t in epic_tasks if t.status == "done")
                total = len(epic_tasks)
                lines.append(f"  {epic.description[:50]} [{done}/{total}]")

        # Tâches actives
        active_tasks = registry.get_active_tasks()
        if active_tasks:
            lines.append("\nTâches en cours:")
            for task in active_tasks[:5]:
                lines.append(f"  {task.description[:50]} ({task.worker_type})")

        # Événements récents
        recent = self._events[-5:]
        if recent:
            lines.append("\nDerniers événements:")
            for event in recent:
                lines.append(f"  [{event.event_type}] {event.message[:60]}")

        return "\n".join(lines) if lines else "Aucune activité"
