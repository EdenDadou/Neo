"""
Neo Core — Worker : Agent Spécialisé Éphémère
================================================
Un Worker est un agent créé dynamiquement par Brain via la Factory
pour exécuter une tâche spécifique, puis détruit.

Cycle de vie :
1. Factory crée le Worker avec config, outils, prompt adapté
2. Worker.execute() exécute la tâche
3. Le résultat est stocké dans Memory
4. Le Worker est détruit (pas de persistance)

Stage 5 :
- Boucle tool_use native (le LLM contrôle ses outils)
- Retry avec exponential backoff
- Timeout protection
- Circuit breaker integration
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from neo_core.core.memory_agent import MemoryAgent

from neo_core.config import NeoConfig
from neo_core.core.resilience import (
    RetryConfig,
    RetryableError,
    NonRetryableError,
    retry_with_backoff,
    HealthMonitor,
)
from neo_core.oauth import is_oauth_token, get_valid_access_token, OAUTH_BETA_HEADER
from neo_core.tools.base_tools import ToolRegistry


class WorkerType(str, Enum):
    """Types de Workers spécialisés disponibles."""
    RESEARCHER = "researcher"
    CODER = "coder"
    SUMMARIZER = "summarizer"
    ANALYST = "analyst"
    WRITER = "writer"
    TRANSLATOR = "translator"
    GENERIC = "generic"


@dataclass
class WorkerResult:
    """Résultat de l'exécution d'un Worker."""
    success: bool
    output: str
    worker_type: str
    task: str
    execution_time: float = 0.0
    errors: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# Prompts système par type de Worker
WORKER_SYSTEM_PROMPTS: dict[WorkerType, str] = {
    WorkerType.RESEARCHER: """Tu es un agent de recherche du système Neo Core.
Ta mission : trouver des informations concrètes et les présenter à l'utilisateur.

RÈGLES ABSOLUES :
1. Tu DOIS utiliser l'outil web_search pour CHAQUE recherche. Ne réponds JAMAIS sans avoir cherché.
2. Formule des requêtes de recherche PRÉCISES et EN ANGLAIS pour de meilleurs résultats.
   Exemple : au lieu de "matchs ATP de la journée", cherche "ATP tennis matches today schedule results {current_date}"
3. Fais 2-3 recherches avec des angles différents si la première ne suffit pas.
4. Présente les RÉSULTATS TROUVÉS directement. Ne dis JAMAIS "je n'ai pas accès" ou "je ne peux pas".
   Tu AS accès au web via l'outil web_search.
5. Si les résultats sont partiels, présente ce que tu as trouvé et suggère des sources.
6. Réponds de manière concise et directe — pas de markdown excessif, pas d'analyse méta.

Tâche : {task}

Contexte mémoire : {memory_context}
""",

    WorkerType.CODER: """Tu es un agent développeur spécialisé du système Neo Core.
Ta mission : écrire, analyser ou débugger du code.

Stratégie :
1. Comprends précisément ce qui est demandé
2. Écris du code propre, commenté et fonctionnel
3. Teste ton code si possible via l'outil d'exécution
4. Explique ta solution

IMPORTANT : Tu as accès à des outils. Utilise-les pour lire des fichiers, exécuter du code, et sauvegarder tes résultats.

Tâche : {task}

Contexte mémoire : {memory_context}
""",

    WorkerType.SUMMARIZER: """Tu es un agent de synthèse spécialisé du système Neo Core.
Ta mission : résumer et condenser des informations.

Stratégie :
1. Lis attentivement le contenu à résumer
2. Identifie les points clés et informations essentielles
3. Produis un résumé clair et concis
4. Conserve les faits importants, élimine le superflu

Tâche : {task}

Contexte mémoire : {memory_context}
""",

    WorkerType.ANALYST: """Tu es un agent analyste spécialisé du système Neo Core.
Ta mission : analyser des données ou des situations et en tirer des conclusions.

Stratégie :
1. Examine les données ou le contexte disponible
2. Identifie les tendances, patterns et anomalies
3. Formule des conclusions et recommandations
4. Justifie ton analyse avec des données

IMPORTANT : Tu as accès à des outils. Utilise-les pour exécuter du code d'analyse et consulter la mémoire.

Tâche : {task}

Contexte mémoire : {memory_context}
""",

    WorkerType.WRITER: """Tu es un agent rédacteur spécialisé du système Neo Core.
Ta mission : créer du contenu écrit de qualité.

Stratégie :
1. Comprends le ton, le style et le public cible
2. Structure ton contenu de manière logique
3. Écris de manière claire, engageante et professionnelle
4. Relis et corrige ton texte

Tâche : {task}

Contexte mémoire : {memory_context}
""",

    WorkerType.TRANSLATOR: """Tu es un agent traducteur spécialisé du système Neo Core.
Ta mission : traduire du contenu entre langues.

Stratégie :
1. Identifie les langues source et cible
2. Traduis fidèlement le sens, pas mot à mot
3. Adapte les expressions idiomatiques
4. Vérifie la cohérence et la fluidité

Tâche : {task}

Contexte mémoire : {memory_context}
""",

    WorkerType.GENERIC: """Tu es un agent spécialisé du système Neo Core.
Ta mission : accomplir la tâche qui t'est assignée avec précision.

IMPORTANT : Tu as accès à des outils. Utilise-les si nécessaire pour accomplir ta tâche.

Tâche : {task}

Contexte mémoire : {memory_context}
""",
}


@dataclass
class Worker:
    """
    Agent spécialisé éphémère.

    Créé par la Factory, exécute une tâche, reporte dans Memory, puis est détruit.

    Stage 5 : Utilise la boucle tool_use native d'Anthropic pour que le LLM
    contrôle activement ses outils au lieu de les appeler à l'aveugle.
    """
    config: NeoConfig
    worker_type: WorkerType
    task: str
    subtasks: list[str] = field(default_factory=list)
    tools: list = field(default_factory=list)
    memory: Optional[MemoryAgent] = None
    system_prompt: str = ""
    health_monitor: Optional[HealthMonitor] = None
    _mock_mode: bool = False

    def __post_init__(self):
        self._mock_mode = self.config.is_mock_mode()
        if not self.system_prompt:
            self.system_prompt = WORKER_SYSTEM_PROMPTS.get(
                self.worker_type, WORKER_SYSTEM_PROMPTS[WorkerType.GENERIC]
            )

    async def execute(self) -> WorkerResult:
        """
        Exécute la tâche assignée.

        En mode mock : retourne un résultat déterministe.
        En mode réel : boucle tool_use avec le LLM.
        """
        start_time = time.time()

        try:
            if self._mock_mode:
                result = self._mock_execute()
            else:
                result = await self._real_execute()

            result.execution_time = time.time() - start_time

            # Reporter dans Memory
            self._report_to_memory(result)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = WorkerResult(
                success=False,
                output=f"Erreur Worker ({self.worker_type.value}): {e}",
                worker_type=self.worker_type.value,
                task=self.task,
                execution_time=execution_time,
                errors=[f"{type(e).__name__}: {e}"],
            )
            self._report_to_memory(result)
            return result

    async def _real_execute(self) -> WorkerResult:
        """
        Exécution réelle via le LLM avec boucle tool_use native.

        Stage 5 : Le LLM contrôle ses outils :
        1. On envoie la requête + définitions d'outils
        2. Si stop_reason == "tool_use" → on exécute l'outil demandé
        3. On renvoie le résultat au LLM
        4. On boucle jusqu'à stop_reason == "end_turn" (max N itérations)
        """
        import httpx

        api_key = self.config.llm.api_key
        memory_context = ""
        if self.memory:
            memory_context = self.memory.get_context(self.task)

        # Construire le prompt système
        prompt = self.system_prompt.format(
            task=self.task,
            memory_context=memory_context or "Aucun contexte disponible.",
        )

        # Headers d'auth
        headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        if is_oauth_token(api_key):
            valid_token = get_valid_access_token() or api_key
            headers["Authorization"] = f"Bearer {valid_token}"
            headers["anthropic-beta"] = OAUTH_BETA_HEADER
        else:
            headers["x-api-key"] = api_key

        # Message initial
        user_message = self.task
        if self.subtasks:
            user_message += "\n\nSous-tâches :\n" + "\n".join(
                f"- {st}" for st in self.subtasks
            )

        # Schémas tool_use pour ce type de Worker
        tool_schemas = ToolRegistry.get_tool_schemas_for_type(self.worker_type.value)

        # Messages conversationnels (accumule les tool_use)
        messages = [{"role": "user", "content": user_message}]

        # Config retry
        retry_config = RetryConfig.from_resilience_config(self.config.resilience)

        # Tracking des tool calls
        tool_calls = []
        max_iterations = self.config.resilience.max_tool_iterations
        total_text_parts = []

        for iteration in range(max_iterations):
            # Payload
            payload = {
                "model": self.config.llm.model,
                "max_tokens": self.config.llm.max_tokens,
                "temperature": self.config.llm.temperature,
                "system": prompt,
                "messages": messages,
            }

            # N'inclure les outils que s'il y en a
            if tool_schemas:
                payload["tools"] = tool_schemas

            # Appel API avec retry
            response_data = await self._api_call_with_retry(
                headers, payload, retry_config
            )

            if response_data is None:
                return WorkerResult(
                    success=False,
                    output="Échec de l'appel API après retries",
                    worker_type=self.worker_type.value,
                    task=self.task,
                    errors=["API call failed after retries"],
                    tool_calls=tool_calls,
                )

            # Traiter la réponse
            stop_reason = response_data.get("stop_reason", "end_turn")
            content_blocks = response_data.get("content", [])

            # Collecter le texte et les tool_use
            assistant_content = []
            pending_tool_uses = []

            for block in content_blocks:
                if block["type"] == "text":
                    total_text_parts.append(block["text"])
                    assistant_content.append(block)
                elif block["type"] == "tool_use":
                    tool_name = block["name"]
                    tool_input = block["input"]
                    tool_id = block["id"]
                    tool_calls.append(f"{tool_name}({tool_input})")
                    assistant_content.append(block)
                    pending_tool_uses.append({
                        "id": tool_id,
                        "name": tool_name,
                        "input": tool_input,
                    })

            if stop_reason == "end_turn" or not pending_tool_uses:
                # Le LLM a terminé — retourner la réponse
                final_output = "\n".join(total_text_parts) if total_text_parts else "Tâche exécutée."
                return WorkerResult(
                    success=True,
                    output=final_output,
                    worker_type=self.worker_type.value,
                    task=self.task,
                    tool_calls=tool_calls,
                    metadata={"iterations": iteration + 1},
                )

            # stop_reason == "tool_use" → exécuter les outils demandés
            # Ajouter le message assistant avec ses tool_use
            messages.append({"role": "assistant", "content": assistant_content})

            # Exécuter chaque outil et construire les résultats
            tool_results = []
            for tu in pending_tool_uses:
                try:
                    result = ToolRegistry.execute_tool(tu["name"], tu["input"])
                except Exception as e:
                    result = f"Erreur outil {tu['name']}: {e}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": str(result)[:5000],  # Limiter la taille
                })

            # Renvoyer les résultats au LLM
            messages.append({"role": "user", "content": tool_results})

        # Max iterations atteint
        final_output = "\n".join(total_text_parts) if total_text_parts else "Tâche interrompue (max itérations atteint)."
        return WorkerResult(
            success=True,
            output=final_output,
            worker_type=self.worker_type.value,
            task=self.task,
            tool_calls=tool_calls,
            metadata={"iterations": max_iterations, "max_reached": True},
        )

    async def _api_call_with_retry(
        self,
        headers: dict,
        payload: dict,
        retry_config: RetryConfig,
    ) -> dict | None:
        """
        Appel API Anthropic avec retry et exponential backoff.

        Retourne le JSON de réponse ou None si toutes les tentatives échouent.
        """
        import httpx

        timeout = self.config.resilience.api_timeout

        async def _do_call():
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )

            status = response.status_code

            # Record dans le health monitor
            if self.health_monitor:
                self.health_monitor.record_api_call(
                    success=(status == 200),
                    error=f"HTTP {status}" if status != 200 else "",
                )

            if status == 200:
                return response.json()

            # Erreurs retryables
            if status in retry_config.retryable_status_codes:
                error_msg = response.text[:300]
                raise RetryableError(f"HTTP {status}: {error_msg}", status_code=status)

            # Erreurs non retryables (400, 401, 403, 404)
            error_data = {}
            try:
                error_data = response.json()
            except Exception:
                pass
            error_msg = error_data.get("error", {}).get("message", response.text[:300])
            raise NonRetryableError(f"HTTP {status}: {error_msg}", status_code=status)

        try:
            return await retry_with_backoff(_do_call, retry_config)
        except NonRetryableError as e:
            # Erreur définitive, pas de retry
            return None
        except RetryableError:
            # Toutes les tentatives épuisées
            return None
        except Exception:
            return None

    def _mock_execute(self) -> WorkerResult:
        """Exécution mock déterministe pour les tests."""
        tool_names = [getattr(t, "name", str(t)) for t in self.tools]

        # Simuler l'appel aux outils
        tool_calls = [f"{name}('{self.task[:30]}...')" for name in tool_names]

        output_parts = [
            f"[Worker {self.worker_type.value}] Tâche exécutée: {self.task}",
        ]

        if self.subtasks:
            output_parts.append("Sous-tâches traitées:")
            for i, st in enumerate(self.subtasks, 1):
                output_parts.append(f"  {i}. {st} ✓")

        if tool_names:
            output_parts.append(f"Outils utilisés: {', '.join(tool_names)}")

        output_parts.append("Résultat: Tâche accomplie avec succès.")

        return WorkerResult(
            success=True,
            output="\n".join(output_parts),
            worker_type=self.worker_type.value,
            task=self.task,
            tool_calls=tool_calls,
            metadata={"mock": True},
        )

    def _report_to_memory(self, result: WorkerResult) -> None:
        """
        Stocke le résultat dans Memory pour l'apprentissage.

        Tags : worker_type, succès/échec, outils utilisés
        """
        if not self.memory:
            return

        try:
            tags = [
                "worker_execution",
                f"type:{self.worker_type.value}",
                "success" if result.success else "failure",
            ]

            importance = 0.6 if result.success else 0.7  # Les échecs sont plus importants

            content = (
                f"Worker {self.worker_type.value} — "
                f"{'Succès' if result.success else 'Échec'}\n"
                f"Tâche: {self.task}\n"
                f"Résultat: {result.output[:500]}"
            )

            if result.errors:
                content += f"\nErreurs: {'; '.join(result.errors)}"

            self.memory.store_memory(
                content=content,
                source="worker_execution",
                tags=tags,
                importance=importance,
                metadata={
                    "worker_type": self.worker_type.value,
                    "execution_time": result.execution_time,
                    "tool_calls": result.tool_calls,
                    "success": result.success,
                },
            )
        except Exception:
            pass  # Ne pas faire crasher le worker pour un problème mémoire
