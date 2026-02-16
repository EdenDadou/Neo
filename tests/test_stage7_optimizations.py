"""
Tests Étape 7 — Optimisations Core (Vox Ack, Memory Registry, Brain Retry)
===========================================================================
Vérifie les 3 optimisations :
1. Vox : Accusé de réception immédiat (callback)
2. Memory : Skills consultables + Task/Epic Registry
3. Brain : Retry persistant avec amélioration via Memory

Tous les tests fonctionnent en mode mock (sans clé API).
"""

import pytest
import asyncio

from neo_core.config import NeoConfig, LLMConfig, MemoryConfig
from neo_core.core.brain import Brain, BrainDecision
from neo_core.core.memory_agent import MemoryAgent
from neo_core.core.vox import Vox
from neo_core.memory.task_registry import TaskRegistry, Task, Epic
from neo_core.memory.learning import LearningEngine, LearnedSkill, ErrorPattern
from neo_core.memory.store import MemoryStore
from neo_core.tools.base_tools import set_mock_mode


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path):
    """Config sans clé API → mode mock, avec stockage temporaire."""
    return NeoConfig(
        llm=LLMConfig(api_key=None),
        memory=MemoryConfig(storage_path=tmp_path / "test_memory"),
    )


@pytest.fixture
def memory(config):
    mem = MemoryAgent(config=config)
    mem.initialize()
    return mem


@pytest.fixture
def brain(config, memory):
    b = Brain(config=config)
    b.connect_memory(memory)
    return b


@pytest.fixture
def vox(config, brain, memory):
    v = Vox(config=config)
    v.connect(brain=brain, memory=memory)
    return v


@pytest.fixture
def store(config):
    s = MemoryStore(config.memory)
    s.initialize()
    return s


@pytest.fixture
def task_registry(store):
    return TaskRegistry(store)


@pytest.fixture
def learning_engine(store):
    return LearningEngine(store)


@pytest.fixture(autouse=True)
def enable_mock_mode():
    set_mock_mode(True)
    yield
    set_mock_mode(False)


# ═══════════════════════════════════════════════════════════════════════
# OPTIM 1 — Vox : Accusé de réception immédiat
# ═══════════════════════════════════════════════════════════════════════

class TestVoxAck:
    """Tests pour l'accusé de réception instantané de Vox."""

    def test_vox_has_thinking_callback(self, vox):
        """Vox doit avoir un attribut callback."""
        assert hasattr(vox, '_on_thinking_callback')
        assert vox._on_thinking_callback is None

    def test_set_thinking_callback(self, vox):
        """set_thinking_callback doit définir le callback."""
        callback = lambda msg: None
        vox.set_thinking_callback(callback)
        assert vox._on_thinking_callback is callback

    @pytest.mark.asyncio
    async def test_generate_ack_mock_mode(self, vox):
        """En mode mock, _generate_ack retourne un ack statique."""
        ack = await vox._generate_ack("Bonjour comment ça va ?")
        assert isinstance(ack, str)
        assert len(ack) > 0
        assert len(ack) < 200

    @pytest.mark.asyncio
    async def test_process_message_calls_callback(self, vox):
        """process_message doit appeler le callback avec un ack."""
        ack_messages = []

        def on_thinking(msg):
            ack_messages.append(msg)

        vox.set_thinking_callback(on_thinking)
        await vox.process_message("Qu'est-ce que Python ?")

        # Le callback doit avoir été appelé au moins une fois
        assert len(ack_messages) >= 1
        assert isinstance(ack_messages[0], str)
        assert len(ack_messages[0]) > 0

    @pytest.mark.asyncio
    async def test_process_message_without_callback(self, vox):
        """process_message fonctionne même sans callback défini."""
        response = await vox.process_message("Hello")
        assert isinstance(response, str)
        assert len(response) > 0


# ═══════════════════════════════════════════════════════════════════════
# OPTIM 2a — Memory : Skills consultables
# ═══════════════════════════════════════════════════════════════════════

class TestSkillsConsultable:
    """Tests pour les compétences consultables dans Memory."""

    def test_get_skills_report_empty(self, memory):
        """get_skills_report retourne un rapport vide si pas de skills."""
        report = memory.get_skills_report()
        assert report["total_skills"] == 0
        assert report["skills"] == []
        assert isinstance(report["performance"], dict)

    def test_get_skills_report_with_data(self, memory):
        """get_skills_report retourne les skills après apprentissage."""
        # Simuler un apprentissage réussi
        memory.record_execution_result(
            request="Cherche des informations sur Python",
            worker_type="researcher",
            success=True,
            execution_time=2.5,
            output="Python est un langage de programmation",
        )

        report = memory.get_skills_report()
        assert report["total_skills"] >= 1
        assert len(report["skills"]) >= 1

        # Vérifier la structure d'une skill
        skill = report["skills"][0]
        assert "name" in skill
        assert "worker_type" in skill
        assert skill["worker_type"] == "researcher"
        assert skill["success_count"] >= 1

    def test_get_skills_report_with_errors(self, memory):
        """get_skills_report inclut les patterns d'erreur."""
        memory.record_execution_result(
            request="Cherche quelque chose",
            worker_type="researcher",
            success=False,
            errors=["timeout: requête trop longue"],
        )

        report = memory.get_skills_report()
        assert report["total_error_patterns"] >= 1

    def test_skills_report_performance(self, memory):
        """get_skills_report inclut les métriques de performance."""
        # Enregistrer plusieurs résultats
        for i in range(3):
            memory.record_execution_result(
                request=f"Tâche {i}",
                worker_type="coder",
                success=i < 2,  # 2 succès, 1 échec
                execution_time=1.0 + i,
                errors=["erreur de format"] if i >= 2 else None,
            )

        report = memory.get_skills_report()
        assert "coder" in report["performance"]
        perf = report["performance"]["coder"]
        assert perf["total_tasks"] == 3


# ═══════════════════════════════════════════════════════════════════════
# OPTIM 2b — Memory : Task/Epic Registry
# ═══════════════════════════════════════════════════════════════════════

class TestTaskRegistry:
    """Tests pour le registre de tâches."""

    def test_create_task(self, task_registry):
        """Création d'une Task basique."""
        task = task_registry.create_task("Analyser le code Python", "coder")
        assert task.id is not None
        assert task.description == "Analyser le code Python"
        assert task.worker_type == "coder"
        assert task.status == "pending"
        assert task.epic_id is None

    def test_task_to_dict_and_back(self):
        """Sérialisation/désérialisation d'une Task."""
        task = Task(
            id="test-123",
            description="Test task",
            worker_type="coder",
            status="done",
            result="OK",
        )
        d = task.to_dict()
        restored = Task.from_dict(d)
        assert restored.id == "test-123"
        assert restored.status == "done"
        assert restored.result == "OK"

    def test_task_str(self):
        """Représentation string d'une Task."""
        task = Task(id="abc12345", description="Test task", worker_type="coder")
        s = str(task)
        assert "abc12345"[:8] in s
        assert "coder" in s

    def test_update_task_status(self, task_registry):
        """Mise à jour du statut d'une Task."""
        task = task_registry.create_task("Test", "generic")

        # Passer en cours
        updated = task_registry.update_task_status(task.id, "in_progress")
        assert updated is not None
        assert updated.status == "in_progress"
        assert updated.attempt_count == 1

        # Terminer
        updated = task_registry.update_task_status(
            task.id, "done", result="Résultat OK"
        )
        assert updated.status == "done"
        assert updated.result == "Résultat OK"
        assert updated.completed_at != ""

    def test_get_all_tasks(self, task_registry):
        """Récupération de toutes les tâches."""
        task_registry.create_task("Task 1", "coder")
        task_registry.create_task("Task 2", "researcher")
        task_registry.create_task("Task 3", "writer")

        tasks = task_registry.get_all_tasks()
        assert len(tasks) == 3

    def test_get_pending_tasks(self, task_registry):
        """Récupération des tâches en attente."""
        t1 = task_registry.create_task("Task 1", "coder")
        task_registry.create_task("Task 2", "researcher")

        task_registry.update_task_status(t1.id, "done")

        pending = task_registry.get_pending_tasks()
        assert len(pending) == 1
        assert pending[0].description == "Task 2"

    def test_create_epic(self, task_registry):
        """Création d'un Epic avec sous-tâches."""
        epic = task_registry.create_epic(
            description="Refactoring complet du système",
            subtask_descriptions=[
                ("Analyser le code existant", "analyst"),
                ("Écrire le nouveau code", "coder"),
                ("Écrire les tests", "coder"),
            ],
            strategy="séquentiel: analyser → coder → tester",
        )

        assert epic.id is not None
        assert len(epic.task_ids) == 3
        assert epic.strategy != ""
        assert epic.status == "pending"

    def test_epic_str(self):
        """Représentation string d'un Epic."""
        epic = Epic(
            id="epic-123",
            description="Grand epic",
            task_ids=["t1", "t2", "t3"],
        )
        s = str(epic)
        assert "3 tâche(s)" in s

    def test_get_epic_tasks(self, task_registry):
        """Récupération des tâches d'un Epic."""
        epic = task_registry.create_epic(
            description="Epic test",
            subtask_descriptions=[
                ("Sous-tâche 1", "coder"),
                ("Sous-tâche 2", "writer"),
            ],
        )

        epic_tasks = task_registry.get_epic_tasks(epic.id)
        assert len(epic_tasks) == 2
        assert all(t.epic_id == epic.id for t in epic_tasks)

    def test_epic_auto_completion(self, task_registry):
        """Un Epic se termine automatiquement quand toutes ses tâches sont done."""
        epic = task_registry.create_epic(
            description="Epic auto",
            subtask_descriptions=[
                ("Tâche 1", "coder"),
                ("Tâche 2", "writer"),
            ],
        )

        # Terminer les deux tâches
        for task_id in epic.task_ids:
            task_registry.update_task_status(task_id, "done")

        # L'epic doit être automatiquement "done"
        updated_epic = task_registry.get_epic(epic.id)
        assert updated_epic is not None
        assert updated_epic.status == "done"

    def test_epic_auto_failure(self, task_registry):
        """Un Epic échoue si une tâche échoue et les autres sont terminales."""
        epic = task_registry.create_epic(
            description="Epic fail",
            subtask_descriptions=[
                ("Tâche 1", "coder"),
                ("Tâche 2", "writer"),
            ],
        )

        task_registry.update_task_status(epic.task_ids[0], "done")
        task_registry.update_task_status(epic.task_ids[1], "failed")

        updated_epic = task_registry.get_epic(epic.id)
        assert updated_epic is not None
        assert updated_epic.status == "failed"

    def test_get_summary(self, task_registry):
        """Résumé global du registre."""
        task_registry.create_task("T1", "coder")
        t2 = task_registry.create_task("T2", "writer")
        task_registry.update_task_status(t2.id, "done")

        summary = task_registry.get_summary()
        assert summary["total_tasks"] == 2
        assert summary["tasks_by_status"]["pending"] == 1
        assert summary["tasks_by_status"]["done"] == 1


class TestTaskRegistryInMemoryAgent:
    """Tests pour l'intégration du TaskRegistry dans MemoryAgent."""

    def test_memory_has_task_registry(self, memory):
        """MemoryAgent doit avoir un TaskRegistry après init."""
        assert memory.task_registry is not None

    def test_memory_create_task(self, memory):
        """MemoryAgent.create_task() fonctionne."""
        task = memory.create_task("Test task", "coder")
        assert task is not None
        assert task.status == "pending"

    def test_memory_create_epic(self, memory):
        """MemoryAgent.create_epic() fonctionne."""
        epic = memory.create_epic(
            "Epic test",
            [("Sub 1", "coder"), ("Sub 2", "writer")],
        )
        assert epic is not None
        assert len(epic.task_ids) == 2

    def test_memory_update_task_status(self, memory):
        """MemoryAgent.update_task_status() fonctionne."""
        task = memory.create_task("Test", "generic")
        updated = memory.update_task_status(task.id, "done", "OK")
        assert updated.status == "done"

    def test_memory_get_tasks_report(self, memory):
        """MemoryAgent.get_tasks_report() retourne un rapport."""
        memory.create_task("T1", "coder")
        memory.create_task("T2", "writer")

        report = memory.get_tasks_report()
        assert report["summary"]["total_tasks"] == 2
        assert len(report["tasks"]) == 2


# ═══════════════════════════════════════════════════════════════════════
# OPTIM 3 — Brain : Retry persistant
# ═══════════════════════════════════════════════════════════════════════

class TestBrainRetry:
    """Tests pour le retry persistant de Brain."""

    @pytest.mark.asyncio
    async def test_brain_creates_task_on_worker_execution(self, brain):
        """Brain crée une Task dans le registry avant d'exécuter un Worker."""
        decision = BrainDecision(
            action="delegate_worker",
            worker_type="generic",
            subtasks=["Tâche simple"],
            confidence=0.8,
        )

        result = await brain._execute_with_worker(
            request="Fais un test simple",
            decision=decision,
            memory_context="",
        )

        # Vérifier qu'une tâche a été créée
        assert brain.memory is not None
        report = brain.memory.get_tasks_report()
        assert report["summary"]["total_tasks"] >= 1

    def test_improve_strategy_attempt_2(self, brain):
        """_improve_strategy change la stratégie en tentative 2."""
        decision = BrainDecision(
            action="delegate_worker",
            worker_type="coder",
            subtasks=["Sub 1", "Sub 2", "Sub 3"],
            confidence=0.8,
        )

        errors = [{
            "attempt": 1,
            "worker_type": "coder",
            "error": "timeout: requête trop longue",
            "errors": ["timeout"],
        }]

        new_decision, new_analysis = brain._improve_strategy(
            "Écris du code Python",
            decision,
            errors,
            attempt=2,
        )

        # La confiance doit être réduite
        assert new_decision.confidence < decision.confidence

        # Soit le worker a changé, soit les subtasks ont été simplifiées
        changed = (
            new_decision.worker_type != decision.worker_type or
            len(new_decision.subtasks) < len(decision.subtasks)
        )
        assert changed

    def test_improve_strategy_attempt_3(self, brain):
        """_improve_strategy passe en generic en tentative 3."""
        decision = BrainDecision(
            action="delegate_worker",
            worker_type="coder",
            subtasks=["Sub 1", "Sub 2"],
            confidence=0.8,
        )

        errors = [
            {"attempt": 1, "worker_type": "coder", "error": "fail 1", "errors": []},
            {"attempt": 2, "worker_type": "coder", "error": "fail 2", "errors": []},
        ]

        new_decision, _ = brain._improve_strategy(
            "Écris du code",
            decision,
            errors,
            attempt=3,
        )

        assert new_decision.worker_type == "generic"
        assert len(new_decision.subtasks) == 1


class TestLearningRetryAdvice:
    """Tests pour get_retry_advice dans le LearningEngine."""

    def test_retry_advice_empty(self, learning_engine):
        """get_retry_advice retourne des conseils même sans historique."""
        advice = learning_engine.get_retry_advice(
            "Une requête",
            "generic",
            [],
        )
        assert isinstance(advice, dict)
        assert "recommended_worker" in advice
        assert "simplify" in advice
        assert "warnings" in advice

    def test_retry_advice_timeout(self, learning_engine):
        """get_retry_advice recommande de simplifier sur timeout."""
        advice = learning_engine.get_retry_advice(
            "Une requête longue",
            "researcher",
            ["timeout: timed out after 60s"],
        )
        assert advice["simplify"] is True

    def test_retry_advice_tool_failure(self, learning_engine):
        """get_retry_advice signale les tool failures."""
        advice = learning_engine.get_retry_advice(
            "Utilise un outil",
            "coder",
            ["tool failure: outil non trouvé"],
        )
        assert len(advice["warnings"]) > 0

    def test_retry_advice_recommends_better_worker(self, learning_engine):
        """get_retry_advice recommande un worker avec meilleur taux de succès."""
        # Simuler un historique où 'writer' réussit mieux que 'generic'
        for i in range(3):
            learning_engine.record_result(
                request=f"Écrire texte {i}",
                worker_type="writer",
                success=True,
                execution_time=1.0,
            )

        for i in range(3):
            learning_engine.record_result(
                request=f"Tâche générique {i}",
                worker_type="generic",
                success=False,
                errors=["erreur"],
            )

        advice = learning_engine.get_retry_advice(
            "Écrire un résumé",
            "generic",
            ["erreur précédente"],
        )

        # Devrait recommander 'writer' qui a un meilleur taux de succès
        assert advice["recommended_worker"] == "writer"


# ═══════════════════════════════════════════════════════════════════════
# Tests d'intégration
# ═══════════════════════════════════════════════════════════════════════

class TestTaskContextEnrichment:
    """Tests pour l'enrichissement automatique des tâches par Memory."""

    def test_add_context_note_to_task(self, task_registry):
        """Ajouter une note de contexte à une Task."""
        task = task_registry.create_task("Analyser le code Python", "coder")
        updated = task_registry.add_task_context(task.id, "L'utilisateur préfère Python 3.12")
        assert updated is not None
        assert len(updated.context_notes) == 1
        assert "Python 3.12" in updated.context_notes[0]

    def test_add_context_note_to_epic(self, task_registry):
        """Ajouter une note de contexte à un Epic."""
        epic = task_registry.create_epic(
            "Refactoring",
            [("Analyser", "analyst"), ("Coder", "coder")],
        )
        updated = task_registry.add_epic_context(epic.id, "Architecture modulaire choisie")
        assert updated is not None
        assert len(updated.context_notes) == 1

    def test_context_notes_limit(self, task_registry):
        """Les notes de contexte sont limitées à 20 max."""
        task = task_registry.create_task("Test", "generic")
        for i in range(25):
            task_registry.add_task_context(task.id, f"Note {i}")

        fetched = task_registry.get_task(task.id)
        assert len(fetched.context_notes) == 20
        # Les notes les plus récentes sont conservées
        assert "Note 24" in fetched.context_notes[-1]

    def test_memory_enriches_active_tasks(self, memory):
        """Memory enrichit les tâches actives lors des conversations."""
        # Créer une tâche et la passer en "in_progress"
        task = memory.create_task("Analyser le code Python", "coder")
        memory.update_task_status(task.id, "in_progress")

        # Simuler un échange conversationnel pertinent
        memory.on_conversation_turn(
            "Je veux analyser le code Python pour trouver les bugs",
            "D'accord, je vais analyser le code Python et identifier les bugs.",
        )

        # La tâche doit avoir été enrichie
        updated = memory.task_registry.get_task(task.id)
        assert len(updated.context_notes) >= 1

    def test_memory_does_not_enrich_irrelevant_tasks(self, memory):
        """Memory n'enrichit pas les tâches non pertinentes."""
        task = memory.create_task("Analyser le code Python", "coder")
        memory.update_task_status(task.id, "in_progress")

        # Échange sans rapport avec la tâche
        memory.on_conversation_turn(
            "Quel temps fait-il aujourd'hui ?",
            "Je ne suis pas en mesure de vérifier la météo.",
        )

        updated = memory.task_registry.get_task(task.id)
        assert len(updated.context_notes) == 0

    def test_memory_enriches_epic_through_task(self, memory):
        """Memory enrichit aussi l'epic quand une de ses tâches est enrichie."""
        epic = memory.create_epic(
            "Refactoring du module Python",
            [("Analyser le code Python", "analyst")],
        )

        # Passer la tâche en in_progress
        task_id = epic.task_ids[0]
        memory.update_task_status(task_id, "in_progress")

        # Échange pertinent
        memory.on_conversation_turn(
            "Le code Python a beaucoup de dette technique",
            "Je vais analyser le code Python et proposer un plan de refactoring.",
        )

        # Vérifier que l'epic a été enrichi aussi
        updated_epic = memory.task_registry.get_epic(epic.id)
        assert len(updated_epic.context_notes) >= 1


class TestIntegration:
    """Tests d'intégration vérifiant que tout fonctionne ensemble."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_ack(self, vox):
        """Pipeline complet avec ack → brain → réponse."""
        acks = []
        vox.set_thinking_callback(lambda msg: acks.append(msg))

        response = await vox.process_message("Explique-moi Python")

        assert isinstance(response, str)
        assert len(response) > 0
        assert len(acks) >= 1

    def test_memory_agent_reports_both(self, memory):
        """MemoryAgent expose skills et tasks dans des rapports séparés."""
        # Créer des données
        memory.record_execution_result(
            request="Test skill",
            worker_type="coder",
            success=True,
            execution_time=1.0,
        )
        memory.create_task("Test task", "coder")

        skills_report = memory.get_skills_report()
        tasks_report = memory.get_tasks_report()

        assert skills_report["total_skills"] >= 1
        assert tasks_report["summary"]["total_tasks"] >= 1
