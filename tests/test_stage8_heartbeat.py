"""
Tests Étape 8 — Heartbeat (Cycle Autonome)
============================================
Vérifie le HeartbeatManager :
1. Lifecycle (start/stop)
2. Avancement automatique des Epics
3. Détection des tâches stale
4. Événements et reporting

Tous les tests fonctionnent en mode mock (sans clé API).
"""

import pytest
import asyncio

from neo_core.config import NeoConfig, LLMConfig, MemoryConfig
from neo_core.brain.core import Brain
from neo_core.memory.agent import MemoryAgent
from neo_core.infra.heartbeat import HeartbeatManager, HeartbeatConfig, HeartbeatEvent
from neo_core.memory.task_registry import TaskRegistry
from neo_core.brain.tools.base_tools import set_mock_mode


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path):
    return NeoConfig(
        llm=LLMConfig(api_key=None),
        memory=MemoryConfig(storage_path=tmp_path / "test_memory"),
    )


@pytest.fixture
def memory(config):
    mem = MemoryAgent(config=config)
    mem.initialize()
    yield mem
    mem.close()


@pytest.fixture
def brain(config, memory):
    b = Brain(config=config)
    b.connect_memory(memory)
    return b


@pytest.fixture
def heartbeat_config():
    return HeartbeatConfig(
        interval_seconds=0.1,  # Très rapide pour les tests
        stale_task_minutes=0.01,  # 0.6 secondes pour les tests
        max_auto_tasks_per_pulse=1,
        auto_consolidation=False,  # Désactiver pour les tests
    )


@pytest.fixture
def events():
    """Collecteur d'événements."""
    return []


@pytest.fixture
def heartbeat(brain, memory, heartbeat_config, events):
    def on_event(event):
        events.append(event)

    hb = HeartbeatManager(
        brain=brain,
        memory=memory,
        config=heartbeat_config,
        on_event=on_event,
    )
    yield hb
    # Cleanup
    if hb.is_running:
        hb.stop()


@pytest.fixture(autouse=True)
def enable_mock_mode():
    set_mock_mode(True)
    yield
    set_mock_mode(False)


# ═══════════════════════════════════════════════════════════════════════
# Lifecycle
# ═══════════════════════════════════════════════════════════════════════

class TestHeartbeatLifecycle:
    """Tests pour le cycle de vie du heartbeat."""

    def test_initial_state(self, heartbeat):
        """Le heartbeat démarre inactif."""
        assert not heartbeat.is_running
        assert heartbeat.pulse_count == 0

    def test_start_stop(self, heartbeat):
        """Le heartbeat peut être démarré et arrêté."""
        heartbeat.start()
        assert heartbeat.is_running

        heartbeat.stop()
        assert not heartbeat.is_running

    def test_double_start(self, heartbeat):
        """Démarrer deux fois ne crée pas deux boucles."""
        heartbeat.start()
        heartbeat.start()  # Pas d'erreur
        assert heartbeat.is_running
        heartbeat.stop()

    def test_double_stop(self, heartbeat):
        """Arrêter deux fois ne pose pas de problème."""
        heartbeat.start()
        heartbeat.stop()
        heartbeat.stop()  # Pas d'erreur
        assert not heartbeat.is_running

    @pytest.mark.asyncio
    async def test_pulse_increments(self, heartbeat):
        """Chaque pulse incrémente le compteur."""
        heartbeat.start()
        await asyncio.sleep(0.35)  # ~3 pulses à 0.1s
        heartbeat.stop()

        assert heartbeat.pulse_count >= 2

    def test_get_status(self, heartbeat):
        """get_status retourne les infos correctes."""
        status = heartbeat.get_status()
        assert status["running"] is False
        assert status["pulse_count"] == 0
        assert "interval" in status

    def test_start_emits_event(self, heartbeat, events):
        """Démarrer le heartbeat émet un événement."""
        heartbeat.start()
        heartbeat.stop()

        event_types = [e.event_type for e in events]
        assert "heartbeat_started" in event_types
        assert "heartbeat_stopped" in event_types

    def test_disabled_config(self, brain, memory):
        """Un heartbeat désactivé ne démarre pas."""
        config = HeartbeatConfig(enabled=False)
        hb = HeartbeatManager(brain=brain, memory=memory, config=config)
        hb.start()
        assert not hb.is_running


# ═══════════════════════════════════════════════════════════════════════
# Avancement des Epics
# ═══════════════════════════════════════════════════════════════════════

class TestEpicAdvancement:
    """Tests pour l'avancement automatique des Epics."""

    @pytest.mark.asyncio
    async def test_epic_auto_starts(self, heartbeat, memory, events):
        """Le heartbeat passe un Epic pending en in_progress."""
        epic = memory.create_epic(
            "Test epic auto",
            [("Tâche 1", "generic"), ("Tâche 2", "generic")],
        )
        assert epic.status == "pending"

        heartbeat.start()
        await asyncio.sleep(0.35)
        heartbeat.stop()

        # L'epic doit être passé en in_progress
        event_types = [e.event_type for e in events]
        assert "epic_started" in event_types

    @pytest.mark.asyncio
    async def test_task_auto_executed(self, heartbeat, memory, events):
        """Le heartbeat lance automatiquement la première tâche pending."""
        epic = memory.create_epic(
            "Epic auto exec",
            [("Tâche simple test", "generic")],
        )

        heartbeat.start()
        await asyncio.sleep(0.5)
        heartbeat.stop()

        # La tâche doit avoir été lancée
        event_types = [e.event_type for e in events]
        assert "task_auto_started" in event_types

    @pytest.mark.asyncio
    async def test_no_double_execution(self, heartbeat, memory):
        """Le heartbeat ne lance pas une tâche si une autre est déjà in_progress."""
        epic = memory.create_epic(
            "Epic no double",
            [("Tâche 1", "generic"), ("Tâche 2", "generic")],
        )

        # Mettre la première tâche en cours manuellement
        memory.update_task_status(epic.task_ids[0], "in_progress")

        # Pulse unique
        registry = memory.task_registry
        await heartbeat._advance_epics(registry)

        # La deuxième tâche doit rester pending
        task2 = registry.get_task(epic.task_ids[1])
        assert task2.status == "pending"

    @pytest.mark.asyncio
    async def test_sequential_task_execution(self, heartbeat, memory, events):
        """Quand une tâche est done, le heartbeat lance la suivante."""
        epic = memory.create_epic(
            "Epic sequential",
            [("Tâche A", "generic"), ("Tâche B", "generic")],
        )

        # Terminer la première tâche manuellement
        memory.update_task_status(epic.task_ids[0], "done", "OK")

        # Un pulse devrait lancer Tâche B
        registry = memory.task_registry
        await heartbeat._advance_epics(registry)

        event_types = [e.event_type for e in events]
        # Soit task_auto_started (la tâche B a été lancée)
        # Soit epic_started (l'epic passe en in_progress)
        assert any(et in event_types for et in ["task_auto_started", "epic_started"])


# ═══════════════════════════════════════════════════════════════════════
# Détection des tâches stale
# ═══════════════════════════════════════════════════════════════════════

class TestStaleDetection:
    """Tests pour la détection des tâches bloquées."""

    def test_detect_stale_task(self, heartbeat, memory, events):
        """Une tâche in_progress depuis trop longtemps est détectée."""
        from datetime import datetime, timedelta

        task = memory.create_task("Tâche vieille", "generic")
        memory.update_task_status(task.id, "in_progress")

        # Tricher : modifier le created_at pour simuler une vieille tâche
        registry = memory.task_registry
        old_task, record_id = registry._find_task_with_id(task.id)
        old_task.created_at = (datetime.now() - timedelta(minutes=30)).isoformat()

        # Supprimer et re-persister avec la date modifiée
        try:
            registry.store.delete(record_id)
        except Exception:
            pass
        registry._persist_task(old_task)

        # Détecter
        heartbeat._detect_stale_tasks(registry)

        event_types = [e.event_type for e in events]
        assert "task_stale" in event_types

    def test_no_false_stale(self, heartbeat, memory, events):
        """Les tâches récentes ne sont pas détectées comme stale."""
        task = memory.create_task("Tâche récente", "generic")
        memory.update_task_status(task.id, "in_progress")

        # Le config a stale_task_minutes=0.01 (0.6s), mais la tâche
        # vient d'être créée → pas stale
        # Note: on a besoin d'attendre un peu moins que le seuil
        heartbeat._detect_stale_tasks(memory.task_registry)

        # Pas d'événement stale (la tâche est trop récente)
        stale_events = [e for e in events if e.event_type == "task_stale"]
        assert len(stale_events) == 0


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════

class TestHeartbeatReporting:
    """Tests pour le reporting du heartbeat."""

    def test_progress_report_empty(self, heartbeat):
        """Le rapport est lisible même sans données."""
        report = heartbeat.get_progress_report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_progress_report_with_data(self, heartbeat, memory):
        """Le rapport inclut les epics et tâches actives."""
        epic = memory.create_epic(
            "Epic pour le rapport",
            [("Tâche de test", "coder")],
        )
        memory.update_task_status(epic.task_ids[0], "in_progress")

        report = heartbeat.get_progress_report()
        assert "Epic" in report or "epic" in report
        assert "1" in report  # Au moins un chiffre

    def test_recent_events(self, heartbeat, events):
        """recent_events retourne les derniers événements."""
        heartbeat.start()
        heartbeat.stop()

        recent = heartbeat.recent_events
        assert len(recent) >= 2  # started + stopped

    def test_get_status_after_pulses(self, heartbeat):
        """get_status reflète l'état après des pulses."""
        heartbeat.start()
        heartbeat.stop()

        status = heartbeat.get_status()
        assert status["running"] is False
        assert "last_event" in status
