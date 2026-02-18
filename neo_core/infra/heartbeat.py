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
    interval_seconds: float = 1800.0  # Intervalle entre les pulses (30 minutes)
    stale_task_minutes: float = 10.0  # Tâche considérée "stale" après N minutes
    max_auto_tasks_per_pulse: int = 1  # Max tâches auto-lancées par pulse
    auto_consolidation: bool = True  # Consolider Memory automatiquement
    consolidation_interval_pulses: int = 10  # Consolider toutes les N pulses
    auto_tuning: bool = True  # Lancer l'auto-tuning périodiquement
    auto_tuning_interval_pulses: int = 5  # Toutes les 5 pulses (~2h30)
    self_patching: bool = True  # Level 3 — auto-correction comportementale
    self_patching_interval_pulses: int = 10  # Toutes les 10 pulses (~5h)
    tool_generation: bool = True  # Level 4 — création autonome d'outils
    tool_generation_interval_pulses: int = 15  # Toutes les 15 pulses (~7h30)
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
    ):
        self.brain = brain
        self.memory = memory
        self.config = config or HeartbeatConfig()
        self._on_event = on_event

        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._pulse_count = 0
        self._events: list[HeartbeatEvent] = []
        self._auto_tuner = None  # Initialisé au premier pulse si disponible
        self._self_patcher = None  # Level 3 — lazy init
        self._tool_generator = None  # Level 4 — lazy init

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

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[Heartbeat] Erreur pulse: %s", e)
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

        # 2. Détecter les tâches stale
        self._detect_stale_tasks(registry)

        # 3. Consolidation Memory périodique
        if (self.config.auto_consolidation and
                self._pulse_count > 0 and
                self._pulse_count % self.config.consolidation_interval_pulses == 0):
            self._consolidate_memory()

        # 4. Auto-réflexion de la personnalité (Stage 9)
        await self._perform_personality_reflection()

        # 5. State snapshot périodique pour Guardian (Stage 10)
        if self._pulse_count > 0 and self._pulse_count % 10 == 0:
            self._save_guardian_state()

        # 6. Auto-tuning périodique (Level 1 self-improvement)
        from neo_core.features import feature_enabled
        if (self.config.auto_tuning and feature_enabled("auto_tuning") and
                self._pulse_count > 0 and
                self._pulse_count % self.config.auto_tuning_interval_pulses == 0):
            self._run_auto_tuning()

        # 7. Self-patching périodique (Level 3 — auto-correction)
        if (self.config.self_patching and feature_enabled("self_patching") and
                self._pulse_count > 0 and
                self._pulse_count % self.config.self_patching_interval_pulses == 0):
            self._run_self_patching()

        # 8. Tool generation périodique (Level 4 — création d'outils)
        if (self.config.tool_generation and feature_enabled("tool_generation") and
                self._pulse_count > 0 and
                self._pulse_count % self.config.tool_generation_interval_pulses == 0):
            self._run_tool_generation()

    # ─── Avancement des Epics ─────────────────────────

    async def _advance_epics(self, registry) -> None:
        """
        Vérifie les Epics in_progress et lance les tâches pending
        dont les prédécesseurs sont terminés.
        """
        epics = registry.get_all_epics(limit=10)
        tasks_launched = 0

        for epic in epics:
            if epic.status not in ("pending", "in_progress"):
                continue

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
            # (= pas de tâche in_progress pour cet epic → on peut en lancer une)
            has_active = any(t.status == "in_progress" for t in epic_tasks)
            if has_active:
                continue  # Attendre que la tâche active se termine

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
        return {
            "running": self._running,
            "pulse_count": self._pulse_count,
            "interval": self.config.interval_seconds,
            "recent_events": len(self._events),
            "last_event": self._events[-1].message if self._events else "aucun",
        }

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
