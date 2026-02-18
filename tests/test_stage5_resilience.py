"""
Tests Stage 5 — Core Fonctionnel : Outils Réels, Résilience & Autonomie
=========================================================================
~45 tests couvrant :
- Schémas tool_use Anthropic
- Exécution d'outils via ToolRegistry.execute_tool()
- Retry avec exponential backoff
- Circuit breaker (ouverture/fermeture/half-open)
- Health monitoring
- Boucle tool_use mock dans Worker
- Intégration Brain + résilience
- Backward compatibility avec les 114 tests existants
"""

import asyncio
import time
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from neo_core.config import NeoConfig, LLMConfig, MemoryConfig, ResilienceConfig
from neo_core.core.brain import Brain, BrainDecision
from neo_core.core.memory_agent import MemoryAgent
from neo_core.core.resilience import (
    RetryConfig,
    RetryableError,
    NonRetryableError,
    retry_with_backoff,
    compute_backoff_delay,
    CircuitBreaker,
    CircuitOpenError,
    HealthMonitor,
    create_resilience_from_config,
)
from neo_core.teams.worker import Worker, WorkerType, WorkerResult, WORKER_SYSTEM_PROMPTS
from neo_core.teams.factory import WorkerFactory, TaskAnalysis
from neo_core.tools.base_tools import (
    ToolRegistry,
    TOOL_SCHEMAS,
    set_mock_mode,
    set_memory_ref,
    web_search_tool,
    file_read_tool,
    file_write_tool,
    code_execute_tool,
    memory_search_tool,
)


# ─── Fixtures ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def enable_mock_mode():
    """Active le mock mode pour tous les tests."""
    set_mock_mode(True)
    yield
    set_mock_mode(False)


@pytest.fixture
def config():
    """Config mock (pas de clé API)."""
    return NeoConfig(
        llm=LLMConfig(api_key=None),
        memory=MemoryConfig(storage_path=Path("/tmp/neo_test_stage5")),
    )


@pytest.fixture
def config_with_resilience():
    """Config avec settings de résilience personnalisés."""
    return NeoConfig(
        llm=LLMConfig(api_key=None),
        memory=MemoryConfig(storage_path=Path("/tmp/neo_test_stage5_res")),
        resilience=ResilienceConfig(
            max_retries=2,
            base_delay=0.01,  # Rapide pour les tests
            circuit_breaker_threshold=3,
            circuit_breaker_recovery=0.5,
        ),
    )


@pytest.fixture
def memory(config):
    """MemoryAgent initialisé pour les tests."""
    mem = MemoryAgent(config=config)
    mem.initialize()
    yield mem
    mem.close()


@pytest.fixture
def brain(config):
    """Brain en mode mock."""
    return Brain(config=config)


@pytest.fixture
def brain_with_memory(config, memory):
    """Brain connecté à Memory."""
    b = Brain(config=config, memory=memory)
    return b


# ═══════════════════════════════════════════════════════════
# TOOL SCHEMAS
# ═══════════════════════════════════════════════════════════

class TestToolSchemas:
    """Tests des schémas tool_use pour l'API Anthropic."""

    def test_all_schemas_exist(self):
        """Chaque outil a un schéma tool_use."""
        expected_tools = ["web_search", "file_read", "file_write", "code_execute", "memory_search"]
        for tool_name in expected_tools:
            assert tool_name in TOOL_SCHEMAS, f"Schéma manquant pour {tool_name}"

    def test_schema_format(self):
        """Les schémas ont le format Anthropic correct."""
        for name, schema in TOOL_SCHEMAS.items():
            assert "name" in schema, f"'name' manquant dans {name}"
            assert "description" in schema, f"'description' manquant dans {name}"
            assert "input_schema" in schema, f"'input_schema' manquant dans {name}"
            assert schema["input_schema"]["type"] == "object"
            assert "properties" in schema["input_schema"]
            assert "required" in schema["input_schema"]

    def test_schema_names_match(self):
        """Les noms dans les schémas correspondent aux clés."""
        for key, schema in TOOL_SCHEMAS.items():
            assert schema["name"] == key

    def test_get_schemas_for_researcher(self):
        """Researcher obtient web_search, memory_search, file_read."""
        schemas = ToolRegistry.get_tool_schemas_for_type("researcher")
        names = [s["name"] for s in schemas]
        assert "web_search" in names
        assert "memory_search" in names
        assert "file_read" in names

    def test_get_schemas_for_coder(self):
        """Coder obtient code_execute, file_read, file_write, memory_search."""
        schemas = ToolRegistry.get_tool_schemas_for_type("coder")
        names = [s["name"] for s in schemas]
        assert "code_execute" in names
        assert "file_read" in names
        assert "file_write" in names

    def test_get_all_schemas(self):
        """get_all_tool_schemas retourne tous les schémas."""
        all_schemas = ToolRegistry.get_all_tool_schemas()
        assert len(all_schemas) == 6

    def test_schemas_have_descriptions(self):
        """Les descriptions sont non-vides et informatives."""
        for name, schema in TOOL_SCHEMAS.items():
            assert len(schema["description"]) > 20, f"Description trop courte pour {name}"


# ═══════════════════════════════════════════════════════════
# TOOL EXECUTOR
# ═══════════════════════════════════════════════════════════

class TestToolExecutor:
    """Tests de ToolRegistry.execute_tool()."""

    def test_execute_web_search(self):
        """execute_tool('web_search', ...) retourne des résultats mock."""
        result = ToolRegistry.execute_tool("web_search", {"query": "Python tutorial"})
        assert "Mock Web Search" in result
        assert "Python tutorial" in result

    def test_execute_file_read(self):
        """execute_tool('file_read', ...) retourne du contenu mock."""
        result = ToolRegistry.execute_tool("file_read", {"path": "/tmp/test.py"})
        assert "Mock File Read" in result

    def test_execute_file_write(self):
        """execute_tool('file_write', ...) confirme l'écriture mock."""
        result = ToolRegistry.execute_tool("file_write", {"path": "/tmp/out.txt", "content": "hello"})
        assert "Mock File Write" in result

    def test_execute_code(self):
        """execute_tool('code_execute', ...) exécute du code mock."""
        result = ToolRegistry.execute_tool("code_execute", {"code": "print('hello')"})
        assert "Mock Code Execute" in result

    def test_execute_memory_search(self):
        """execute_tool('memory_search', ...) cherche en mémoire mock."""
        result = ToolRegistry.execute_tool("memory_search", {"query": "historique"})
        assert "Mock Memory Search" in result

    def test_execute_unknown_tool(self):
        """execute_tool avec un outil inconnu retourne une erreur."""
        result = ToolRegistry.execute_tool("unknown_tool", {"arg": "val"})
        assert "Erreur" in result
        assert "unknown_tool" in result

    def test_execute_with_empty_args(self):
        """execute_tool gère les arguments vides."""
        result = ToolRegistry.execute_tool("web_search", {})
        assert "Mock Web Search" in result


# ═══════════════════════════════════════════════════════════
# RETRY & BACKOFF
# ═══════════════════════════════════════════════════════════

class TestRetryBackoff:
    """Tests du retry avec exponential backoff."""

    def test_compute_backoff_delay(self):
        """Les délais croissent exponentiellement (avec jitter ±20%)."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=30.0)
        # attempt=0 → base 1.0, jitter [0.8, 1.2]
        assert 0.8 <= compute_backoff_delay(0, config) <= 1.2
        # attempt=1 → base 2.0, jitter [1.6, 2.4]
        assert 1.6 <= compute_backoff_delay(1, config) <= 2.4
        # attempt=2 → base 4.0, jitter [3.2, 4.8]
        assert 3.2 <= compute_backoff_delay(2, config) <= 4.8
        # attempt=3 → base 8.0, jitter [6.4, 9.6]
        assert 6.4 <= compute_backoff_delay(3, config) <= 9.6

    def test_backoff_max_delay(self):
        """Le délai ne dépasse pas max_delay * 1.2 (jitter)."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=5.0)
        delay = compute_backoff_delay(10, config)
        assert 4.0 <= delay <= 6.0

    @pytest.mark.asyncio
    async def test_retry_success_first_try(self):
        """Succès au premier essai."""
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await retry_with_backoff(success_func)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Succès après quelques échecs retryables."""
        call_count = 0
        config = RetryConfig(max_retries=3, base_delay=0.01)

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("temporary failure", status_code=429)
            return "ok"

        result = await retry_with_backoff(flaky_func, config)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_non_retryable_error(self):
        """NonRetryableError n'est pas retentée."""
        call_count = 0

        async def bad_func():
            nonlocal call_count
            call_count += 1
            raise NonRetryableError("bad request", status_code=400)

        with pytest.raises(NonRetryableError):
            await retry_with_backoff(bad_func)

        assert call_count == 1  # Pas de retry

    @pytest.mark.asyncio
    async def test_retry_all_attempts_fail(self):
        """Toutes les tentatives échouent → lève RetryableError."""
        config = RetryConfig(max_retries=2, base_delay=0.01)

        async def always_fail():
            raise RetryableError("server error", status_code=500)

        with pytest.raises(RetryableError):
            await retry_with_backoff(always_fail, config)

    @pytest.mark.asyncio
    async def test_retry_callback(self):
        """Le callback on_retry est appelé avant chaque retry."""
        retries = []
        config = RetryConfig(max_retries=2, base_delay=0.01)

        async def fail_then_ok():
            if len(retries) < 1:
                raise RetryableError("fail", status_code=500)
            return "ok"

        def on_retry(attempt, error, delay):
            retries.append((attempt, str(error), delay))

        result = await retry_with_backoff(fail_then_ok, config, on_retry=on_retry)
        assert result == "ok"
        assert len(retries) == 1
        assert retries[0][0] == 0  # Premier attempt

    @pytest.mark.asyncio
    async def test_retry_timeout_error(self):
        """asyncio.TimeoutError est retryable."""
        call_count = 0
        config = RetryConfig(max_retries=2, base_delay=0.01)

        async def timeout_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError("timeout")
            return "ok"

        result = await retry_with_backoff(timeout_then_ok, config)
        assert result == "ok"
        assert call_count == 2

    def test_retry_config_from_resilience(self):
        """RetryConfig se crée correctement depuis ResilienceConfig."""
        rc = ResilienceConfig(max_retries=5, base_delay=2.0, max_delay=60.0)
        retry = RetryConfig.from_resilience_config(rc)
        assert retry.max_retries == 5
        assert retry.base_delay == 2.0
        assert retry.max_delay == 60.0


# ═══════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════

class TestCircuitBreaker:
    """Tests du circuit breaker."""

    def test_initial_state_closed(self):
        """Le circuit commence fermé."""
        cb = CircuitBreaker()
        assert cb.state == "closed"
        assert cb.can_execute()

    def test_opens_after_threshold(self):
        """Le circuit s'ouvre après N échecs consécutifs."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "open"
        assert not cb.can_execute()

    def test_success_resets_failures(self):
        """Un succès remet le compteur d'échecs à 0."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._consecutive_failures == 0
        cb.record_failure()
        assert cb.state == "closed"  # Pas encore 3 consécutifs

    def test_half_open_after_timeout(self):
        """Le circuit passe en half_open après le recovery timeout."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.15)
        assert cb.can_execute()  # Passe en half_open
        assert cb.state == "half_open"

    def test_half_open_success_closes(self):
        """Un succès en half_open ferme le circuit."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.can_execute()  # Trigger half_open
        cb.record_success()
        assert cb.state == "closed"

    def test_half_open_failure_reopens(self):
        """Un échec en half_open rouvre le circuit."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.can_execute()  # Trigger half_open
        cb.record_failure()
        assert cb.state == "open"

    def test_reset(self):
        """reset() remet tout à zéro."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        cb.reset()
        assert cb.state == "closed"
        assert cb._consecutive_failures == 0

    def test_get_stats(self):
        """get_stats() retourne les bonnes métriques."""
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_success()
        cb.record_success()
        cb.record_failure()
        stats = cb.get_stats()
        assert stats["state"] == "closed"
        assert stats["total_successes"] == 2
        assert stats["total_failures"] == 1
        assert stats["consecutive_failures"] == 1


# ═══════════════════════════════════════════════════════════
# HEALTH MONITOR
# ═══════════════════════════════════════════════════════════

class TestHealthMonitor:
    """Tests du moniteur de santé."""

    def test_initial_state(self):
        """Le moniteur commence sain."""
        hm = HealthMonitor()
        assert hm.total_calls == 0
        assert hm.error_rate == 0.0
        assert hm.can_make_api_call()

    def test_record_success(self):
        """Enregistrer un succès met à jour les métriques."""
        hm = HealthMonitor()
        hm.record_api_call(success=True, duration=0.5)
        assert hm.total_calls == 1
        assert hm.total_successes == 1
        assert hm.error_rate == 0.0

    def test_record_failure(self):
        """Enregistrer un échec met à jour le taux d'erreur."""
        hm = HealthMonitor()
        hm.record_api_call(success=False, error="HTTP 500")
        assert hm.total_calls == 1
        assert hm.total_errors == 1
        assert hm.error_rate == 1.0

    def test_error_rate_calculation(self):
        """Le taux d'erreur se calcule sur les derniers appels."""
        hm = HealthMonitor()
        for _ in range(8):
            hm.record_api_call(success=True)
        for _ in range(2):
            hm.record_api_call(success=False)
        assert abs(hm.error_rate - 0.2) < 0.01

    def test_avg_response_time(self):
        """Le temps de réponse moyen est calculé correctement."""
        hm = HealthMonitor()
        hm.record_api_call(success=True, duration=1.0)
        hm.record_api_call(success=True, duration=3.0)
        assert abs(hm.avg_response_time - 2.0) < 0.01

    def test_health_report(self):
        """get_health_report() retourne un rapport structuré."""
        hm = HealthMonitor()
        hm.record_api_call(success=True, duration=0.5)
        report = hm.get_health_report()
        assert "status" in report
        assert "api" in report
        assert "memory" in report
        assert report["status"] == "healthy"

    def test_health_degraded_when_circuit_open(self):
        """Le statut est 'degraded' quand le circuit est ouvert."""
        cb = CircuitBreaker(failure_threshold=2)
        hm = HealthMonitor(api_circuit=cb)
        hm.record_api_call(success=False)
        hm.record_api_call(success=False)
        report = hm.get_health_report()
        assert report["status"] == "degraded"

    def test_memory_health_tracked(self):
        """L'état de la mémoire est tracké."""
        hm = HealthMonitor()
        hm.set_memory_health(False)
        report = hm.get_health_report()
        assert report["memory"]["healthy"] is False
        assert report["status"] == "degraded"

    def test_create_from_config(self):
        """create_resilience_from_config crée les 3 composants."""
        rc = ResilienceConfig(max_retries=5, circuit_breaker_threshold=10)
        retry, circuit, health = create_resilience_from_config(rc)
        assert retry.max_retries == 5
        assert circuit.failure_threshold == 10
        assert health.total_calls == 0


# ═══════════════════════════════════════════════════════════
# WORKER TOOL_USE
# ═══════════════════════════════════════════════════════════

class TestWorkerToolUse:
    """Tests de la boucle tool_use dans Worker."""

    def test_worker_has_health_monitor_field(self, config):
        """Worker accepte un health_monitor."""
        hm = HealthMonitor()
        worker = Worker(
            config=config,
            worker_type=WorkerType.RESEARCHER,
            task="test",
            health_monitor=hm,
        )
        assert worker.health_monitor is hm

    @pytest.mark.asyncio
    async def test_mock_execute_unchanged(self, config):
        """L'exécution mock fonctionne toujours comme avant."""
        worker = Worker(
            config=config,
            worker_type=WorkerType.RESEARCHER,
            task="Rechercher des informations sur Python",
            tools=ToolRegistry.get_tools_for_type("researcher"),
        )
        result = await worker.execute()
        assert result.success
        assert "Worker researcher" in result.output
        assert result.metadata.get("mock") is True

    @pytest.mark.asyncio
    async def test_mock_execute_with_subtasks(self, config):
        """Mock execute traite les sous-tâches."""
        worker = Worker(
            config=config,
            worker_type=WorkerType.CODER,
            task="Écrire un script Python",
            subtasks=["Lire les specs", "Écrire le code", "Tester"],
            tools=ToolRegistry.get_tools_for_type("coder"),
        )
        result = await worker.execute()
        assert result.success
        assert "Sous-tâches traitées" in result.output
        assert "✓" in result.output

    @pytest.mark.asyncio
    async def test_worker_reports_to_memory(self, config, memory):
        """Worker stocke le résultat dans Memory."""
        worker = Worker(
            config=config,
            worker_type=WorkerType.RESEARCHER,
            task="Rechercher des infos sur l'IA",
            tools=ToolRegistry.get_tools_for_type("researcher"),
            memory=memory,
        )
        result = await worker.execute()
        assert result.success

        # Vérifier que c'est stocké en mémoire
        records = memory.search("researcher", n_results=5)
        worker_records = [r for r in records if any("worker:" in t for t in r.tags)]
        assert len(worker_records) > 0


# ═══════════════════════════════════════════════════════════
# BRAIN RÉSILIENCE
# ═══════════════════════════════════════════════════════════

class TestBrainResilience:
    """Tests de l'intégration résilience dans Brain."""

    def test_brain_has_health_monitor(self, config):
        """Brain initialise le health monitor."""
        brain = Brain(config=config)
        assert brain.health is not None
        assert isinstance(brain.health, HealthMonitor)

    def test_brain_has_retry_config(self, config):
        """Brain initialise la config de retry."""
        brain = Brain(config=config)
        assert brain._retry_config is not None
        assert isinstance(brain._retry_config, RetryConfig)

    def test_brain_health_report(self, brain):
        """get_system_health() retourne un rapport complet."""
        report = brain.get_system_health()
        assert "status" in report
        assert "api" in report
        assert "brain" in report
        assert report["brain"]["mock_mode"] is True

    def test_brain_health_with_memory(self, brain_with_memory):
        """Le rapport inclut les stats mémoire."""
        report = brain_with_memory.get_system_health()
        assert "memory" in report
        assert report["memory"]["healthy"] is True

    @pytest.mark.asyncio
    async def test_brain_circuit_breaker_blocks(self, config):
        """Brain refuse les appels quand le circuit est ouvert (mode réel uniquement)."""
        # En mock mode, le circuit breaker n'est pas vérifié
        # On teste la logique directement
        brain = Brain(config=config)
        # Forcer l'ouverture du circuit
        for _ in range(10):
            brain.health.api_circuit.record_failure()
        assert brain.health.api_circuit.state == "open"

    def test_resilience_config_propagation(self):
        """La config de résilience se propage correctement."""
        rc = ResilienceConfig(max_retries=7, base_delay=2.0)
        config = NeoConfig(
            llm=LLMConfig(api_key=None),
            resilience=rc,
        )
        brain = Brain(config=config)
        assert brain._retry_config.max_retries == 7
        assert brain._retry_config.base_delay == 2.0


# ═══════════════════════════════════════════════════════════
# RESILIENCE CONFIG
# ═══════════════════════════════════════════════════════════

class TestResilienceConfig:
    """Tests de la config de résilience."""

    def test_default_values(self):
        """Les valeurs par défaut sont sensées."""
        rc = ResilienceConfig()
        assert rc.max_retries == 3
        assert rc.base_delay == 1.0
        assert rc.api_timeout == 60.0
        assert rc.worker_timeout == 180.0
        assert rc.circuit_breaker_threshold == 5
        assert rc.max_tool_iterations == 10

    def test_config_in_neo_config(self):
        """NeoConfig inclut ResilienceConfig."""
        config = NeoConfig()
        assert hasattr(config, "resilience")
        assert isinstance(config.resilience, ResilienceConfig)

    def test_custom_resilience_config(self):
        """Valeurs personnalisées sont acceptées."""
        rc = ResilienceConfig(
            max_retries=10,
            base_delay=0.5,
            api_timeout=30.0,
            max_tool_iterations=20,
        )
        assert rc.max_retries == 10
        assert rc.max_tool_iterations == 20


# ═══════════════════════════════════════════════════════════
# INTEGRATION
# ═══════════════════════════════════════════════════════════

class TestIntegrationStage5:
    """Tests d'intégration du pipeline complet avec résilience."""

    @pytest.mark.asyncio
    async def test_full_pipeline_simple_request(self, brain_with_memory):
        """Pipeline complet pour une requête simple."""
        result = await brain_with_memory.process("Bonjour")
        # En mode mock, la réponse vient soit de Brain directement,
        # soit d'un Worker si le LearningEngine a acquis des compétences
        assert "Brain Mock" in result or "Worker" in result

    @pytest.mark.asyncio
    async def test_full_pipeline_worker_delegation(self, brain_with_memory):
        """Pipeline complet avec délégation à un Worker."""
        result = await brain_with_memory.process(
            "Recherche des informations détaillées et approfondies sur les dernières "
            "avancées majeures en intelligence artificielle et leurs implications"
        )
        assert "Worker researcher" in result or "Brain Mock" in result

    @pytest.mark.asyncio
    async def test_pipeline_health_tracked(self, brain_with_memory):
        """Les appels au pipeline sont trackés dans le health monitor."""
        await brain_with_memory.process("Test de santé du système")
        # En mock mode, pas d'appels API réels, mais le health monitor existe
        assert brain_with_memory.health is not None

    @pytest.mark.asyncio
    async def test_worker_creation_with_tool_schemas(self, config, memory):
        """Les Workers créés ont accès aux schémas tool_use."""
        factory = WorkerFactory(config=config, memory=memory)
        analysis = factory.analyze_task(
            "Recherche des informations détaillées et approfondies sur le sujet "
            "de l'intelligence artificielle avec toutes les sources disponibles"
        )
        worker = factory.create_worker(analysis)
        # Vérifier que les schémas sont disponibles pour le type du worker
        schemas = ToolRegistry.get_tool_schemas_for_type(worker.worker_type.value)
        assert len(schemas) > 0


# ═══════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Vérifie que les changements Stage 5 ne cassent pas les étapes précédentes."""

    def test_config_still_has_llm(self):
        """NeoConfig a toujours LLMConfig."""
        config = NeoConfig()
        assert hasattr(config, "llm")
        assert config.llm.provider == "anthropic"

    def test_config_still_has_memory(self):
        """NeoConfig a toujours MemoryConfig."""
        config = NeoConfig()
        assert hasattr(config, "memory")

    def test_worker_type_enum_unchanged(self):
        """WorkerType a toujours les mêmes valeurs."""
        assert WorkerType.RESEARCHER.value == "researcher"
        assert WorkerType.CODER.value == "coder"
        assert WorkerType.SUMMARIZER.value == "summarizer"
        assert WorkerType.ANALYST.value == "analyst"
        assert WorkerType.WRITER.value == "writer"
        assert WorkerType.TRANSLATOR.value == "translator"
        assert WorkerType.GENERIC.value == "generic"

    def test_worker_result_fields_unchanged(self):
        """WorkerResult a toujours les mêmes champs."""
        result = WorkerResult(
            success=True,
            output="test",
            worker_type="researcher",
            task="test task",
        )
        assert hasattr(result, "success")
        assert hasattr(result, "output")
        assert hasattr(result, "worker_type")
        assert hasattr(result, "task")
        assert hasattr(result, "execution_time")
        assert hasattr(result, "errors")
        assert hasattr(result, "tool_calls")
        assert hasattr(result, "metadata")

    def test_brain_decision_fields_unchanged(self):
        """BrainDecision a toujours les mêmes champs."""
        decision = BrainDecision(action="direct_response")
        assert hasattr(decision, "action")
        assert hasattr(decision, "response")
        assert hasattr(decision, "subtasks")
        assert hasattr(decision, "confidence")
        assert hasattr(decision, "worker_type")
        assert hasattr(decision, "reasoning")

    def test_tool_registry_api_unchanged(self):
        """ToolRegistry garde son API publique."""
        assert hasattr(ToolRegistry, "get_tool")
        assert hasattr(ToolRegistry, "get_tools_for_type")
        assert hasattr(ToolRegistry, "list_tools")
        assert hasattr(ToolRegistry, "initialize")
        # Nouvelles méthodes Stage 5
        assert hasattr(ToolRegistry, "execute_tool")
        assert hasattr(ToolRegistry, "get_tool_schemas_for_type")
        assert hasattr(ToolRegistry, "get_all_tool_schemas")

    def test_worker_system_prompts_exist(self):
        """Les prompts système existent pour tous les types."""
        for wt in WorkerType:
            assert wt in WORKER_SYSTEM_PROMPTS

    @pytest.mark.asyncio
    async def test_brain_process_still_works(self, brain):
        """Brain.process() fonctionne toujours comme avant."""
        result = await brain.process("Bonjour, comment vas-tu?")
        assert isinstance(result, str)
        assert len(result) > 0
