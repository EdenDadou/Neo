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

Authentification :
- Hérite de la même méthode que Brain (OAuth Bearer + beta, API key, mock)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from neo_core.core.memory_agent import MemoryAgent

from neo_core.config import NeoConfig
from neo_core.oauth import is_oauth_token, get_valid_access_token, OAUTH_BETA_HEADER


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
    WorkerType.RESEARCHER: """Tu es un agent de recherche spécialisé du système Neo Core.
Ta mission : rechercher et synthétiser des informations sur un sujet donné.

Stratégie :
1. Identifie les aspects clés du sujet à rechercher
2. Utilise les outils de recherche web et mémoire disponibles
3. Synthétise les résultats de manière claire et structurée
4. Cite tes sources

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

Tâche : {task}

Contexte mémoire : {memory_context}
""",
}


@dataclass
class Worker:
    """
    Agent spécialisé éphémère.

    Créé par la Factory, exécute une tâche, reporte dans Memory, puis est détruit.
    """
    config: NeoConfig
    worker_type: WorkerType
    task: str
    subtasks: list[str] = field(default_factory=list)
    tools: list = field(default_factory=list)
    memory: Optional[MemoryAgent] = None
    system_prompt: str = ""
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
        En mode réel : appelle le LLM avec les outils disponibles.
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
        Exécution réelle via le LLM.

        Utilise httpx direct avec Bearer + beta header (comme Brain)
        pour garder le contrôle total sur l'auth.
        """
        import httpx

        api_key = self.config.llm.api_key
        memory_context = ""
        if self.memory:
            memory_context = self.memory.get_context(self.task)

        # Construire le prompt
        prompt = self.system_prompt.format(
            task=self.task,
            memory_context=memory_context or "Aucun contexte disponible.",
        )

        # Déterminer les headers d'auth
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

        # Construire le message
        user_message = self.task
        if self.subtasks:
            user_message += "\n\nSous-tâches :\n" + "\n".join(
                f"- {st}" for st in self.subtasks
            )

        # Appel aux outils d'abord si disponibles (exécution séquentielle simple)
        tool_results = []
        tool_calls = []
        for t in self.tools:
            try:
                # Chaque outil peut contribuer du contexte
                tool_name = getattr(t, "name", str(t))
                if tool_name == "web_search":
                    res = t.invoke(self.task[:100])
                    tool_results.append(f"[Recherche web]\n{res}")
                    tool_calls.append(f"web_search('{self.task[:50]}...')")
                elif tool_name == "memory_search":
                    res = t.invoke(self.task[:100])
                    tool_results.append(f"[Mémoire]\n{res}")
                    tool_calls.append(f"memory_search('{self.task[:50]}...')")
                elif tool_name == "file_read" and self.subtasks:
                    # Lire les fichiers mentionnés dans les sous-tâches
                    for st in self.subtasks:
                        if "/" in st or "." in st:
                            res = t.invoke(st)
                            tool_results.append(f"[Fichier: {st}]\n{res}")
                            tool_calls.append(f"file_read('{st}')")
            except Exception:
                pass

        # Enrichir le message avec les résultats d'outils
        if tool_results:
            user_message += "\n\nRésultats des outils :\n" + "\n---\n".join(tool_results)

        payload = {
            "model": self.config.llm.model,
            "max_tokens": self.config.llm.max_tokens,
            "temperature": self.config.llm.temperature,
            "system": prompt,
            "messages": [{"role": "user", "content": user_message}],
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=120,
            )

        if response.status_code == 200:
            data = response.json()
            output = data["content"][0]["text"]
            return WorkerResult(
                success=True,
                output=output,
                worker_type=self.worker_type.value,
                task=self.task,
                tool_calls=tool_calls,
            )
        else:
            error_data = response.json() if "json" in response.headers.get("content-type", "") else {}
            error_msg = error_data.get("error", {}).get("message", response.text[:300])
            return WorkerResult(
                success=False,
                output=f"Erreur API ({response.status_code}): {error_msg}",
                worker_type=self.worker_type.value,
                task=self.task,
                errors=[f"HTTP {response.status_code}: {error_msg}"],
                tool_calls=tool_calls,
            )

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
