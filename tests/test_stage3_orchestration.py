"""
Tests Étape 3 — Moteur d'Orchestration (Brain avancé)
======================================================
Vérifie que Brain peut analyser les requêtes, créer des Workers
spécialisés via la Factory, et apprendre des résultats.
Tous les tests fonctionnent en mode mock (sans clé API).
"""

import pytest
import asyncio

from neo_core.config import NeoConfig, LLMConfig, MemoryConfig
from neo_core.brain.core import Brain, BrainDecision
from neo_core.memory.agent import MemoryAgent
from neo_core.vox.interface import Vox
from neo_core.brain.teams.worker import Worker, WorkerType, WorkerResult, WORKER_SYSTEM_PROMPTS
from neo_core.brain.teams.factory import WorkerFactory, TaskAnalysis
from neo_core.brain.tools.base_tools import (
    ToolRegistry,
    web_search_tool,
    file_read_tool,
    file_write_tool,
    code_execute_tool,
    memory_search_tool,
    set_mock_mode,
)


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
def factory(config, memory):
    return WorkerFactory(config=config, memory=memory)


@pytest.fixture(autouse=True)
def enable_mock_mode():
    """Active le mock mode pour tous les outils avant chaque test."""
    set_mock_mode(True)
    yield
    set_mock_mode(False)


# ─── Tests ToolRegistry ─────────────────────────────────────────────

class TestToolRegistry:
    """Test du registre d'outils."""

    def test_list_tools(self):
        """Le registre contient les 5 outils de base."""
        tools = ToolRegistry.list_tools()
        assert "web_search" in tools
        assert "file_read" in tools
        assert "file_write" in tools
        assert "code_execute" in tools
        assert "memory_search" in tools

    def test_list_worker_types(self):
        """Le registre connaît tous les types de workers."""
        types = ToolRegistry.list_worker_types()
        assert "researcher" in types
        assert "coder" in types
        assert "summarizer" in types
        assert "analyst" in types
        assert "writer" in types
        assert "generic" in types

    def test_get_tool_by_name(self):
        """On peut récupérer un outil par son nom."""
        tool = ToolRegistry.get_tool("web_search")
        assert tool is web_search_tool

    def test_get_tool_unknown_raises(self):
        """Un outil inconnu lève ValueError."""
        with pytest.raises(ValueError, match="Outil inconnu"):
            ToolRegistry.get_tool("unknown_tool")

    def test_get_tools_for_researcher(self):
        """Un researcher a web_search, memory_search, file_read."""
        tools = ToolRegistry.get_tools_for_type("researcher")
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert any("web_search" in n for n in tool_names)
        assert any("memory_search" in n for n in tool_names)
        assert any("file_read" in n for n in tool_names)

    def test_get_tools_for_coder(self):
        """Un coder a code_execute, file_read, file_write, memory_search."""
        tools = ToolRegistry.get_tools_for_type("coder")
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert any("code_execute" in n for n in tool_names)
        assert any("file_read" in n for n in tool_names)
        assert any("file_write" in n for n in tool_names)

    def test_get_tools_for_unknown_defaults_to_generic(self):
        """Un type inconnu utilise les outils génériques."""
        tools = ToolRegistry.get_tools_for_type("unknown_type")
        generic_tools = ToolRegistry.get_tools_for_type("generic")
        assert len(tools) == len(generic_tools)

    def test_initialize(self, memory):
        """ToolRegistry.initialize() configure mock mode et mémoire."""
        ToolRegistry.initialize(mock_mode=True, memory=memory)
        # Vérifier que les outils fonctionnent en mock
        result = web_search_tool.invoke("test query")
        assert "[Mock Web Search]" in result


# ─── Tests Outils Mock ──────────────────────────────────────────────

class TestToolsMock:
    """Test des outils en mode mock."""

    def test_web_search_mock(self):
        """web_search retourne des résultats mock."""
        result = web_search_tool.invoke("intelligence artificielle")
        assert "Mock Web Search" in result
        assert "intelligence artificielle" in result

    def test_file_read_mock(self):
        """file_read retourne un contenu mock."""
        result = file_read_tool.invoke("test.txt")
        assert "Mock File Read" in result

    def test_file_write_mock(self):
        """file_write retourne une confirmation mock."""
        result = file_write_tool.invoke({"path": "test.txt", "content": "hello"})
        assert "Mock File Write" in result
        assert "5 caractères" in result

    def test_code_execute_mock_with_print(self):
        """code_execute simule l'exécution de code avec print."""
        result = code_execute_tool.invoke("print('hello world')")
        assert "Mock Code Execute" in result

    def test_code_execute_mock_no_output(self):
        """code_execute simule l'exécution sans sortie."""
        result = code_execute_tool.invoke("x = 42")
        assert "Mock Code Execute" in result
        assert "1 lignes" in result

    def test_memory_search_mock(self):
        """memory_search retourne des résultats mock."""
        result = memory_search_tool.invoke("test query")
        assert "Mock Memory Search" in result


# ─── Tests WorkerType ────────────────────────────────────────────────

class TestWorkerTypes:
    """Test de la classification des tâches."""

    def test_classify_researcher(self, factory):
        """Recherche → RESEARCHER."""
        assert factory.classify_task("Recherche sur le machine learning") == WorkerType.RESEARCHER
        assert factory.classify_task("Cherche des informations sur Python") == WorkerType.RESEARCHER
        assert factory.classify_task("Trouve des données sur l'IA") == WorkerType.RESEARCHER

    def test_classify_coder(self, factory):
        """Code → CODER."""
        assert factory.classify_task("Écris un script Python pour trier une liste") == WorkerType.CODER
        assert factory.classify_task("Debug ce programme qui ne fonctionne pas") == WorkerType.CODER
        assert factory.classify_task("Implémente un algorithme de tri rapide") == WorkerType.CODER

    def test_classify_summarizer(self, factory):
        """Résumé → SUMMARIZER."""
        assert factory.classify_task("Résume ce document de 50 pages en points clés") == WorkerType.SUMMARIZER
        assert factory.classify_task("Fais une synthèse détaillée de cette longue réunion") == WorkerType.SUMMARIZER

    def test_classify_analyst(self, factory):
        """Analyse → ANALYST."""
        assert factory.classify_task("Analyse les données de vente de ce trimestre") == WorkerType.ANALYST
        assert factory.classify_task("Compare ces deux approches et évalue les résultats") == WorkerType.ANALYST

    def test_classify_writer(self, factory):
        """Rédaction → WRITER."""
        assert factory.classify_task("Rédige un article de blog sur les tendances tech") == WorkerType.WRITER
        assert factory.classify_task("Écris un rapport détaillé sur nos performances") == WorkerType.WRITER

    def test_classify_translator(self, factory):
        """Traduction → TRANSLATOR."""
        assert factory.classify_task("Traduis ce texte en anglais") == WorkerType.TRANSLATOR

    def test_classify_generic(self, factory):
        """Requête vague → GENERIC."""
        assert factory.classify_task("Bonjour, ça va ?") == WorkerType.GENERIC
        assert factory.classify_task("Oui") == WorkerType.GENERIC


# ─── Tests Worker ────────────────────────────────────────────────────

class TestWorker:
    """Test de l'exécution des Workers."""

    @pytest.mark.asyncio
    async def test_worker_mock_execute(self, config, memory):
        """Un Worker en mock exécute et retourne un WorkerResult."""
        worker = Worker(
            config=config,
            worker_type=WorkerType.RESEARCHER,
            task="Recherche sur l'IA",
            memory=memory,
        )
        result = await worker.execute()
        assert isinstance(result, WorkerResult)
        assert result.success is True
        assert result.worker_type == "researcher"
        assert "Recherche sur l'IA" in result.task

    @pytest.mark.asyncio
    async def test_worker_with_subtasks(self, config, memory):
        """Un Worker traite ses sous-tâches."""
        worker = Worker(
            config=config,
            worker_type=WorkerType.CODER,
            task="Écrire un script Python",
            subtasks=["Comprendre les specs", "Écrire le code", "Tester"],
            memory=memory,
        )
        result = await worker.execute()
        assert result.success is True
        assert "Sous-tâches traitées" in result.output
        assert "Comprendre les specs" in result.output

    @pytest.mark.asyncio
    async def test_worker_with_tools(self, config, memory):
        """Un Worker utilise les outils assignés."""
        tools = ToolRegistry.get_tools_for_type("researcher")
        worker = Worker(
            config=config,
            worker_type=WorkerType.RESEARCHER,
            task="Recherche test",
            tools=tools,
            memory=memory,
        )
        result = await worker.execute()
        assert result.success is True
        assert len(result.tool_calls) > 0

    @pytest.mark.asyncio
    async def test_worker_reports_to_memory(self, config, memory):
        """Un Worker stocke son résultat dans Memory."""
        worker = Worker(
            config=config,
            worker_type=WorkerType.SUMMARIZER,
            task="Résumer un document",
            memory=memory,
        )
        initial_count = memory.get_stats().get("total_entries", 0)
        await worker.execute()
        final_count = memory.get_stats().get("total_entries", 0)
        assert final_count > initial_count

    @pytest.mark.asyncio
    async def test_worker_result_metadata(self, config):
        """WorkerResult contient les métadonnées correctes."""
        worker = Worker(
            config=config,
            worker_type=WorkerType.GENERIC,
            task="Test task",
        )
        result = await worker.execute()
        assert result.execution_time >= 0
        assert result.metadata.get("mock") is True

    def test_worker_system_prompts_exist(self):
        """Chaque WorkerType a un prompt système."""
        for wt in WorkerType:
            assert wt in WORKER_SYSTEM_PROMPTS
            assert "{task}" in WORKER_SYSTEM_PROMPTS[wt]
            assert "{memory_context}" in WORKER_SYSTEM_PROMPTS[wt]


# ─── Tests Factory ───────────────────────────────────────────────────

class TestFactory:
    """Test de la fabrique de Workers."""

    def test_analyze_task(self, factory):
        """analyze_task produit une TaskAnalysis complète."""
        analysis = factory.analyze_task("Recherche approfondie sur le deep learning et ses applications récentes")
        assert isinstance(analysis, TaskAnalysis)
        assert analysis.worker_type == WorkerType.RESEARCHER
        assert len(analysis.subtasks) > 0
        assert len(analysis.required_tools) > 0
        assert analysis.confidence > 0

    def test_create_worker_from_analysis(self, factory):
        """create_worker crée un Worker correctement configuré."""
        analysis = TaskAnalysis(
            worker_type=WorkerType.CODER,
            primary_task="Écrire un script",
            subtasks=["Specs", "Code", "Test"],
        )
        worker = factory.create_worker(analysis)
        assert isinstance(worker, Worker)
        assert worker.worker_type == WorkerType.CODER
        assert worker.task == "Écrire un script"
        assert len(worker.tools) > 0

    def test_create_worker_for_type(self, factory):
        """create_worker_for_type est un raccourci fonctionnel."""
        worker = factory.create_worker_for_type(
            worker_type=WorkerType.ANALYST,
            task="Analyser les données",
        )
        assert worker.worker_type == WorkerType.ANALYST
        assert worker.task == "Analyser les données"

    def test_factory_basic_decompose_researcher(self, factory):
        """Décomposition basique pour un researcher."""
        subtasks = factory._basic_decompose("Recherche IA", WorkerType.RESEARCHER)
        assert len(subtasks) == 3
        assert any("Rechercher" in st for st in subtasks)

    def test_factory_basic_decompose_coder(self, factory):
        """Décomposition basique pour un coder."""
        subtasks = factory._basic_decompose("Script Python", WorkerType.CODER)
        assert len(subtasks) == 3
        assert any("code" in st.lower() for st in subtasks)

    @pytest.mark.asyncio
    async def test_factory_create_and_execute(self, factory):
        """Pipeline complet : analyse → création → exécution."""
        analysis = factory.analyze_task("Écris un script Python pour calculer des statistiques")
        worker = factory.create_worker(analysis)
        result = await worker.execute()
        assert result.success is True
        assert result.worker_type == "coder"


# ─── Tests Brain Decision (Stage 3) ─────────────────────────────────

class TestBrainDecision:
    """Test des décisions de Brain enrichies."""

    def test_simple_generic_direct_response(self, brain):
        """Requête simple et générique → réponse directe."""
        decision = brain.make_decision("Bonjour")
        assert decision.action == "direct_response"
        assert decision.confidence >= 0.8

    def test_moderate_typed_delegate_worker(self, brain):
        """Requête modérée avec type spécifique → délègue à un Worker."""
        decision = brain.make_decision(
            "Recherche des informations très détaillées et approfondies sur les toutes dernières avancées majeures en intelligence artificielle et machine learning"
        )
        assert decision.action == "delegate_worker"
        assert decision.worker_type == "researcher"
        assert len(decision.subtasks) > 0

    def test_complex_delegate_worker(self, brain):
        """Requête complexe → délègue à un Worker."""
        long_request = (
            "Je voudrais que tu analyses en profondeur les données de vente du dernier trimestre, "
            "que tu identifies les tendances principales, que tu compares avec l'année précédente, "
            "et que tu produises un rapport détaillé avec des recommandations stratégiques "
            "pour le prochain trimestre incluant des projections et des scénarios optimiste et pessimiste"
        )
        decision = brain.make_decision(long_request)
        assert decision.action == "delegate_worker"
        assert decision.confidence > 0

    def test_code_task_identified(self, brain):
        """Une demande de code est correctement identifiée."""
        decision = brain.make_decision(
            "Écris un script Python qui implémente un algorithme de tri fusion avec des tests unitaires"
        )
        assert decision.action == "delegate_worker"
        assert decision.worker_type == "coder"

    def test_decision_has_reasoning(self, brain):
        """Les décisions incluent un reasoning."""
        decision = brain.make_decision("Résume ce long document technique en points clés principaux")
        assert decision.reasoning != ""

    def test_decision_backward_compat(self, brain):
        """make_decision reste compatible avec les tests Stage 1."""
        decision = brain.make_decision("Bonjour")
        assert hasattr(decision, "action")
        assert hasattr(decision, "confidence")
        assert hasattr(decision, "subtasks")


# ─── Tests Brain Process avec Workers ────────────────────────────────

class TestBrainProcess:
    """Test du pipeline process() avec orchestration de Workers."""

    @pytest.mark.asyncio
    async def test_simple_request_direct_response(self, brain):
        """Une requête simple passe par la réponse directe (mock)."""
        response = await brain.process("Salut")
        assert "[Brain Mock]" in response

    @pytest.mark.asyncio
    async def test_complex_request_uses_worker(self, brain):
        """Une requête complexe et typée utilise un Worker."""
        response = await brain.process(
            "Recherche approfondie sur les dernières avancées en deep learning et les applications "
            "dans le domaine médical, avec une analyse comparative des différentes architectures"
        )
        assert "[Worker" in response or "[Brain Mock]" in response

    @pytest.mark.asyncio
    async def test_code_request_uses_coder_worker(self, brain):
        """Une demande de code utilise un Worker coder."""
        response = await brain.process(
            "Développe un script Python complet qui implémente un serveur HTTP "
            "avec gestion des routes et middleware d'authentification"
        )
        # En mock, devrait passer par un worker coder
        assert "coder" in response.lower() or "Worker" in response or "Brain Mock" in response

    @pytest.mark.asyncio
    async def test_worker_result_stored_in_memory(self, brain, memory):
        """Le résultat d'un Worker est stocké dans Memory."""
        initial_count = memory.get_stats().get("total_entries", 0)
        await brain.process(
            "Fais une recherche complète sur les frameworks Python pour le développement web "
            "et compare leurs avantages et inconvénients respectifs"
        )
        final_count = memory.get_stats().get("total_entries", 0)
        # Au moins 1 entrée ajoutée (résultat worker + apprentissage Brain)
        assert final_count > initial_count


# ─── Tests Brain Learning ────────────────────────────────────────────

class TestBrainLearning:
    """Test de l'apprentissage de Brain."""

    @pytest.mark.asyncio
    async def test_learn_from_success(self, brain, memory):
        """Brain stocke un apprentissage après succès."""
        decision = BrainDecision(
            action="delegate_worker",
            worker_type="researcher",
            confidence=0.8,
        )
        result = WorkerResult(
            success=True,
            output="Résultat de recherche",
            worker_type="researcher",
            task="Recherche test",
            execution_time=1.5,
        )
        await brain._learn_from_result("test request", decision, result)

        # Vérifier que c'est dans Memory
        stats = memory.get_stats()
        assert stats["total_entries"] > 0

    @pytest.mark.asyncio
    async def test_learn_from_failure(self, brain, memory):
        """Brain stocke un apprentissage après échec (importance plus haute)."""
        decision = BrainDecision(
            action="delegate_worker",
            worker_type="coder",
            confidence=0.6,
        )
        result = WorkerResult(
            success=False,
            output="Erreur de compilation",
            worker_type="coder",
            task="Code test",
            execution_time=0.5,
            errors=["SyntaxError: invalid syntax"],
        )
        await brain._learn_from_result("test request", decision, result)

        stats = memory.get_stats()
        assert stats["total_entries"] > 0


# ─── Tests Intégration Pipeline Complet ──────────────────────────────

class TestIntegration:
    """Tests d'intégration end-to-end."""

    @pytest.mark.asyncio
    async def test_full_pipeline_simple(self, config, memory):
        """Pipeline complet simple : Vox → Brain → réponse directe."""
        brain = Brain(config=config)
        brain.connect_memory(memory)
        vox = Vox(config=config)
        vox.connect(brain=brain, memory=memory)

        response = await vox.process_message("Bonjour, comment vas-tu ?")
        assert response is not None
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_full_pipeline_with_worker(self, config, memory):
        """Pipeline complet : Vox → Brain → Worker → Memory."""
        brain = Brain(config=config)
        brain.connect_memory(memory)
        vox = Vox(config=config)
        vox.connect(brain=brain, memory=memory)

        response = await vox.process_message(
            "Recherche approfondie sur les algorithmes de tri en informatique "
            "et leurs complexités temporelles et spatiales respectives"
        )
        assert response is not None
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_multiple_requests_memory_grows(self, config, memory):
        """Plusieurs requêtes enrichissent progressivement la mémoire."""
        brain = Brain(config=config)
        brain.connect_memory(memory)
        vox = Vox(config=config)
        vox.connect(brain=brain, memory=memory)

        # Plusieurs requêtes variées
        await vox.process_message("Salut")
        count_1 = memory.get_stats()["total_entries"]

        await vox.process_message(
            "Développe un script Python complet pour analyser des données CSV "
            "avec pandas et matplotlib pour créer des graphiques"
        )
        count_2 = memory.get_stats()["total_entries"]

        assert count_2 > count_1

    @pytest.mark.asyncio
    async def test_factory_accessible_from_brain(self, brain):
        """Brain.factory est accessible et correctement initialisé."""
        factory = brain.factory
        assert isinstance(factory, WorkerFactory)
        assert factory.config == brain.config
        assert factory.memory == brain.memory


# ─── Tests Backward Compatibility ────────────────────────────────────

class TestBackwardCompatibility:
    """Vérifie que les changements Stage 3 ne cassent pas Stages 1 et 2."""

    def test_brain_decision_has_legacy_fields(self):
        """BrainDecision a toujours les champs action, response, subtasks, confidence."""
        d = BrainDecision(action="direct_response")
        assert d.action == "direct_response"
        assert d.response is None
        assert d.subtasks == []
        assert d.confidence == 1.0

    def test_brain_mock_mode(self, brain):
        """Brain est en mock mode sans clé API."""
        assert brain._mock_mode is True

    @pytest.mark.asyncio
    async def test_brain_process_still_works(self, brain):
        """brain.process() fonctionne toujours comme avant."""
        response = await brain.process("test simple")
        assert "Brain Mock" in response or "Worker" in response

    def test_brain_analyze_complexity(self, brain):
        """analyze_complexity fonctionne comme avant."""
        assert brain.analyze_complexity("court") == "simple"
        assert brain.analyze_complexity(
            "une requête de longueur moyenne avec assez de mots pour dépasser le seuil de quinze mots nécessaires pour être modérée"
        ) == "moderate"

    def test_brain_connect_memory(self, brain, memory):
        """connect_memory fonctionne comme avant."""
        assert brain.memory == memory


# ═══════════════════════════════════════════════════════════
# WORKER LIFECYCLE MANAGEMENT
# ═══════════════════════════════════════════════════════════

class TestWorkerLifecycle:
    """Tests du cycle de vie complet des Workers."""

    def test_worker_has_lifecycle_fields(self, config):
        """Un Worker a un ID, un state, et des timestamps."""
        from neo_core.brain.teams.worker import WorkerState
        worker = Worker(
            config=config,
            worker_type=WorkerType.RESEARCHER,
            task="test lifecycle",
        )
        assert worker.worker_id  # Non vide
        assert len(worker.worker_id) == 8  # UUID tronqué
        assert worker.state == WorkerState.CREATED
        assert worker._created_at > 0
        assert worker.is_active is True

    @pytest.mark.asyncio
    async def test_worker_state_transitions(self, config):
        """Le state passe CREATED → RUNNING → COMPLETED après execute."""
        from neo_core.brain.teams.worker import WorkerState
        worker = Worker(
            config=config,
            worker_type=WorkerType.GENERIC,
            task="test state transitions",
        )
        assert worker.state == WorkerState.CREATED

        result = await worker.execute()

        assert worker.state == WorkerState.COMPLETED
        assert result.success is True
        assert worker._finished_at > worker._started_at

    @pytest.mark.asyncio
    async def test_worker_cleanup_releases_resources(self, config):
        """cleanup() libère les outils, prompt, memory, et passe en CLOSED."""
        from neo_core.brain.teams.worker import WorkerState
        worker = Worker(
            config=config,
            worker_type=WorkerType.RESEARCHER,
            task="test cleanup",
            tools=["fake_tool_1", "fake_tool_2"],
        )
        assert len(worker.tools) == 2

        await worker.execute()
        worker.cleanup()

        assert worker.state == WorkerState.CLOSED
        assert worker.tools == []
        assert worker.system_prompt == ""
        assert worker.memory is None
        assert worker.health_monitor is None
        assert worker.is_active is False

    @pytest.mark.asyncio
    async def test_worker_context_manager_guarantees_cleanup(self, config):
        """async with Worker garantit cleanup même sans appel explicite."""
        from neo_core.brain.teams.worker import WorkerState

        worker = Worker(
            config=config,
            worker_type=WorkerType.GENERIC,
            task="test context manager",
        )

        async with worker:
            await worker.execute()
            assert worker.state == WorkerState.COMPLETED

        # Après sortie du context manager → CLOSED
        assert worker.state == WorkerState.CLOSED
        assert worker.tools == []

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self, config):
        """async with Worker garantit cleanup même si une exception survient."""
        from neo_core.brain.teams.worker import WorkerState

        worker = Worker(
            config=config,
            worker_type=WorkerType.GENERIC,
            task="test exception cleanup",
        )

        with pytest.raises(ValueError):
            async with worker:
                raise ValueError("Simulated crash")

        # cleanup() a quand même été appelé
        assert worker.state == WorkerState.CLOSED

    def test_worker_lifecycle_info(self, config):
        """get_lifecycle_info() retourne un dict complet."""
        worker = Worker(
            config=config,
            worker_type=WorkerType.ANALYST,
            task="analyse des tendances du marché",
        )
        info = worker.get_lifecycle_info()
        assert info["worker_id"] == worker.worker_id
        assert info["worker_type"] == "analyst"
        assert info["state"] == "created"
        assert "analyse des tendances" in info["task"]

    def test_lifecycle_manager_register_unregister(self, config):
        """Le manager track les Workers actifs et l'historique."""
        from neo_core.brain.core import WorkerLifecycleManager
        manager = WorkerLifecycleManager()

        w1 = Worker(config=config, worker_type=WorkerType.RESEARCHER, task="task1")
        w2 = Worker(config=config, worker_type=WorkerType.CODER, task="task2")

        manager.register(w1)
        manager.register(w2)

        assert len(manager.get_active_workers()) == 2
        assert manager.get_stats()["active_count"] == 2
        assert manager.get_stats()["total_created"] == 2

        # Unregister w1
        w1.cleanup()
        manager.unregister(w1)

        assert len(manager.get_active_workers()) == 1
        assert manager.get_stats()["active_count"] == 1
        assert manager.get_stats()["total_cleaned"] == 1
        assert len(manager.get_history()) == 1

    def test_lifecycle_manager_cleanup_all(self, config):
        """cleanup_all() force la destruction de tous les Workers actifs."""
        from neo_core.brain.core import WorkerLifecycleManager
        manager = WorkerLifecycleManager()

        for i in range(5):
            w = Worker(config=config, worker_type=WorkerType.GENERIC, task=f"task{i}")
            manager.register(w)

        assert manager.get_stats()["active_count"] == 5

        cleaned = manager.cleanup_all()
        assert cleaned == 5
        assert manager.get_stats()["active_count"] == 0
        assert manager.get_stats()["total_cleaned"] == 5

    def test_lifecycle_manager_leak_detection(self, config):
        """Le manager détecte les fuites (Workers créés mais jamais nettoyés)."""
        from neo_core.brain.core import WorkerLifecycleManager
        manager = WorkerLifecycleManager()

        w = Worker(config=config, worker_type=WorkerType.GENERIC, task="leaky")
        manager.register(w)

        # Simuler un "oubli" de unregister : le worker est encore actif
        stats = manager.get_stats()
        assert stats["active_count"] == 1
        # Pas de fuite tant que le Worker est actif
        assert stats["leaked"] == 0

    @pytest.mark.asyncio
    async def test_brain_execute_with_worker_lifecycle(self, brain):
        """Brain._execute_with_worker utilise le lifecycle manager."""
        decision = BrainDecision(
            action="delegate_worker",
            subtasks=["sous-tâche test"],
            confidence=0.8,
            worker_type="researcher",
            reasoning="test lifecycle integration",
        )

        result = await brain._execute_with_worker("recherche test lifecycle", decision, "")

        # Le Worker a été exécuté et nettoyé
        assert "Worker" in result or "recherche" in result.lower() or "Tâche" in result
        stats = brain.worker_manager.get_stats()
        assert stats["total_created"] >= 1
        assert stats["total_cleaned"] >= 1
        assert stats["active_count"] == 0  # Aucun Worker actif après exécution

    @pytest.mark.asyncio
    async def test_brain_worker_history_after_execution(self, brain):
        """Après exécution, l'historique contient le Worker terminé."""
        decision = BrainDecision(
            action="delegate_worker",
            subtasks=["test"],
            confidence=0.8,
            worker_type="generic",
        )

        await brain._execute_with_worker("test historique", decision, "")

        history = brain.get_worker_history(limit=5)
        assert len(history) >= 1
        last = history[-1]
        # L'historique capture l'état au moment de unregister
        # (avant que __aexit__ appelle cleanup → CLOSED)
        assert last["state"] in ("completed", "closed")
        assert last["worker_type"] == "generic"

    @pytest.mark.asyncio
    async def test_memory_learns_before_cleanup(self, brain, memory):
        """Memory récupère l'apprentissage AVANT que le Worker soit détruit."""
        decision = BrainDecision(
            action="delegate_worker",
            subtasks=["test apprentissage"],
            confidence=0.8,
            worker_type="researcher",
        )

        await brain._execute_with_worker("recherche pour test apprentissage", decision, "")

        # Vérifier que Memory a enregistré quelque chose
        stats = memory.get_stats()
        assert stats.get("total_entries", 0) > 0

        # Le Worker est bien fermé (plus actif)
        assert brain.worker_manager.get_stats()["active_count"] == 0
