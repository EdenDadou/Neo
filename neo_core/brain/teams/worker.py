"""
Neo Core — Worker : Agent Spécialisé Éphémère
================================================
Un Worker est un agent créé dynamiquement par Brain via la Factory
pour exécuter une tâche spécifique, puis détruit.

Cycle de vie géré par le WorkerLifecycleManager :
1. Factory crée le Worker → état CREATED
2. Brain enregistre le Worker dans le registre → état RUNNING
3. Worker.execute() exécute la tâche
4. Memory récupère TOUS les apprentissages (AVANT destruction)
5. Worker.cleanup() libère les ressources → état CLOSED
6. Le Worker est retiré du registre

Stage 5 :
- Boucle tool_use native (le LLM contrôle ses outils)
- Retry avec exponential backoff
- Timeout protection
- Circuit breaker integration
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from neo_core.memory.agent import MemoryAgent

from neo_core.config import NeoConfig, get_agent_model
from neo_core.infra.resilience import (
    RetryConfig,
    RetryableError,
    NonRetryableError,
    retry_with_backoff,
    HealthMonitor,
)
from neo_core.oauth import is_oauth_token, get_valid_access_token, OAUTH_BETA_HEADER
from neo_core.brain.tools.base_tools import ToolRegistry

logger = logging.getLogger(__name__)


class WorkerState(str, Enum):
    """État du cycle de vie d'un Worker."""
    CREATED = "created"      # Worker créé, pas encore lancé
    RUNNING = "running"      # En cours d'exécution
    COMPLETED = "completed"  # Exécution terminée, apprentissage fait
    FAILED = "failed"        # Exécution échouée, apprentissage fait
    CLOSED = "closed"        # Ressources libérées, prêt pour GC


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

Date et heure actuelles : {current_date}, {current_time}

RÈGLES ABSOLUES :
1. Tu DOIS utiliser l'outil web_search pour CHAQUE recherche. Ne réponds JAMAIS sans avoir cherché.
2. Formule des requêtes PRÉCISES et EN ANGLAIS. Inclus la date si pertinent.
   Exemple : "ATP Doha schedule {current_date}" ou "ATP tennis matches today odds"
3. Fais MAXIMUM 3 appels d'outils au total (web_search et/ou web_fetch combinés), puis SYNTHÉTISE.
4. Présente les RÉSULTATS TROUVÉS directement. Ne dis JAMAIS "je n'ai pas accès".
5. Si les résultats sont partiels, présente ce que tu as trouvé + liens utiles.

CONVERGENCE — CRITIQUE :
- Après tes appels d'outils, tu DOIS produire ta RÉPONSE FINALE immédiatement.
- Ne dis JAMAIS "Laissez-moi chercher...", "Je vais récupérer...", "Excellent, laisse-moi...".
- Ne NARRATE PAS ton processus de recherche. L'utilisateur veut le RÉSULTAT, pas le journal de bord.
- Si web_fetch échoue (site JavaScript), N'INSISTE PAS. Utilise les extraits de web_search et donne le lien.
- Tu as 3 appels d'outils MAX. Si après 3 appels tu n'as pas tout, présente ce que tu as + liens directs.

FORMAT DE SORTIE :
- Quand tu appelles un outil, n'écris AUCUN texte d'accompagnement (pas de "Je vais chercher...").
- Écris ta réponse UNIQUEMENT quand tu as fini TOUS tes appels d'outils.
- Réponds de manière concise et naturelle, sans markdown excessif.

ANTI-HALLUCINATION — TRÈS IMPORTANT :
- N'INVENTE JAMAIS de scores, de classements, de cotes, ou de données chiffrées.
- Si tu ne trouves pas un score en direct, dis-le clairement : "Score non trouvé dans les résultats."
- Ne présente comme fait QUE ce qui est explicitement dans les résultats de web_search.
- Distingue toujours ce que tu as TROUVÉ (avec source) de ce que tu SUPPOSES.
- Pour les scores live, donne les liens directs (flashscore.com, sofascore.com) plutôt qu'inventer.

Tâche : {task}

Contexte mémoire : {memory_context}
""",

    WorkerType.CODER: """Tu es un agent développeur autonome du système Neo Core.
Ta mission : PRODUIRE du code fonctionnel, l'EXÉCUTER, et le SAUVEGARDER.

Date et heure actuelles : {current_date}, {current_time}

RÈGLES ABSOLUES :
1. Tu ÉCRIS du code, tu l'EXÉCUTES avec code_execute, et tu SAUVEGARDES le résultat avec file_write.
2. Ne dis JAMAIS "voici ce que je ferais" ou "tu pourrais faire" — FAIS-LE.
3. Si le code plante, analyse l'erreur, corrige, et relance. Tu as plusieurs itérations.
4. Sauvegarde TOUJOURS tes fichiers dans le répertoire du projet.
5. Installe les dépendances nécessaires via code_execute (pip install ...) si besoin.

CAPACITÉS :
- Créer des sites web (HTML/CSS/JS, Flask, FastAPI)
- Créer et manipuler des BDD (SQLite, JSON)
- Scripts d'automatisation (API, scraping, cron)
- Analyse de données (pandas, matplotlib)
- Tout ce qui est faisable en Python

CONVERGENCE :
- Après avoir écrit et testé le code, donne un RÉSUMÉ de ce qui a été créé et où.
- Ne narrate PAS ton processus. L'utilisateur veut le RÉSULTAT.

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

    WorkerType.ANALYST: """Tu es un agent analyste autonome du système Neo Core.
Ta mission : COLLECTER des données, les ANALYSER avec du code, et PRODUIRE des résultats concrets.

Date et heure actuelles : {current_date}, {current_time}

RÈGLES ABSOLUES :
1. Tu DOIS écrire et EXÉCUTER du code Python pour tes analyses (pandas, numpy, matplotlib, scipy).
2. Cherche les données nécessaires avec web_search/web_fetch si elles ne sont pas en mémoire.
3. Produis des CHIFFRES, des TABLEAUX, des GRAPHIQUES — pas du texte vague.
4. Sauvegarde tes résultats (CSV, JSON, images) avec file_write.
5. Si une analyse échoue, corrige et relance. Tu as plusieurs itérations.

CAPACITÉS :
- Analyse statistique (moyennes, corrélations, régressions)
- Modélisation et prédiction (scikit-learn, scipy)
- Visualisation (matplotlib, plotly → fichiers PNG/HTML)
- Traitement de données (pandas, nettoyage, transformation)
- Calculs financiers (ROI, EV, bankroll management, Kelly criterion)

CONVERGENCE :
- Présente tes conclusions avec les données qui les soutiennent.
- Pas de narration. RÉSULTATS + RECOMMANDATIONS.

Tâche : {task}

Contexte mémoire : {memory_context}
""",

    WorkerType.WRITER: """Tu es un agent rédacteur autonome du système Neo Core.
Ta mission : PRODUIRE du contenu écrit de qualité et le SAUVEGARDER.

Date et heure actuelles : {current_date}, {current_time}

RÈGLES ABSOLUES :
1. Tu PRODUIS le contenu demandé — pas un plan ou une ébauche, le RÉSULTAT FINAL.
2. Sauvegarde TOUJOURS le contenu dans un fichier (Markdown, HTML, ou texte) avec file_write.
3. Adapte le ton au public cible sans qu'on te le dise.
4. Si tu as besoin d'informations, utilise web_search pour les obtenir.

CAPACITÉS :
- Articles, rapports, synthèses, newsletters
- Documentation technique, README, guides
- Scripts, pitchs, contenus marketing
- Tout format texte demandé

CONVERGENCE :
- Sauvegarde le fichier, puis résume ce que tu as produit et où.

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

    WorkerType.GENERIC: """Tu es un agent autonome polyvalent du système Neo Core.
Ta mission : accomplir la tâche qui t'est assignée en AGISSANT concrètement.

Date et heure actuelles : {current_date}, {current_time}

RÈGLES ABSOLUES :
1. Tu as accès à TOUS les outils : web_search, web_fetch, code_execute, file_read, file_write, memory_search.
2. Ne dis JAMAIS "voici ce que tu pourrais faire" — FAIS-LE directement.
3. Si la tâche nécessite du code, écris-le ET exécute-le.
4. Si la tâche nécessite des données, va les chercher sur le web.
5. Sauvegarde TOUJOURS tes résultats dans des fichiers.
6. Si quelque chose échoue, corrige et recommence. Tu es autonome.

PHILOSOPHIE : Tu es un agent qui AGIT, pas un assistant qui CONSEILLE.
L'utilisateur ne veut pas savoir comment faire — il veut que ce soit FAIT.

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
    conversation_context: str = ""  # Derniers échanges user↔Neo pour donner du contexte au Worker
    system_prompt: str = ""
    health_monitor: Optional[HealthMonitor] = None
    _mock_mode: bool = False
    _model_config: Optional[object] = None
    # ── Lifecycle fields ──
    _worker_id: str = ""
    _state: WorkerState = WorkerState.CREATED
    _created_at: float = 0.0
    _started_at: float = 0.0
    _finished_at: float = 0.0
    _result: Optional[WorkerResult] = None

    def __post_init__(self):
        self._mock_mode = self.config.is_mock_mode()
        self._worker_id = str(uuid.uuid4())[:8]
        self._created_at = time.time()
        # Modèle sélectionné selon le type de Worker
        self._model_config = get_agent_model(f"worker:{self.worker_type.value}")
        if not self.system_prompt:
            self.system_prompt = WORKER_SYSTEM_PROMPTS.get(
                self.worker_type, WORKER_SYSTEM_PROMPTS[WorkerType.GENERIC]
            )

    # ── Propriétés lifecycle ──

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def state(self) -> WorkerState:
        return self._state

    @property
    def is_active(self) -> bool:
        """True si le Worker est en cours d'exécution."""
        return self._state in (WorkerState.CREATED, WorkerState.RUNNING)

    @property
    def lifetime(self) -> float:
        """Durée de vie du Worker en secondes."""
        end = self._finished_at or time.time()
        return end - self._created_at

    async def execute(self) -> WorkerResult:
        """
        Exécute la tâche assignée.

        En mode mock : retourne un résultat déterministe.
        En mode réel : boucle tool_use avec le LLM, avec timeout
        (ResilienceConfig.worker_timeout, défaut 180s).

        Note : Le state est géré ici, mais la garantie de cleanup
        est assurée par le context manager async (voir __aenter__/__aexit__).
        """
        self._state = WorkerState.RUNNING
        self._started_at = time.time()
        start_time = self._started_at

        # Timeout depuis la config résilience (défaut 180s)
        worker_timeout = self.config.resilience.worker_timeout

        try:
            if self._mock_mode:
                result = self._mock_execute()
            else:
                # Protéger l'exécution réelle par un timeout
                async with asyncio.timeout(worker_timeout):
                    result = await self._real_execute()

            result.execution_time = time.time() - start_time
            self._state = WorkerState.COMPLETED
            self._finished_at = time.time()
            self._result = result

            # Reporter dans Memory (apprentissage AVANT cleanup)
            self._report_to_memory(result)

            return result

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            result = WorkerResult(
                success=False,
                output=f"Worker ({self.worker_type.value}) timeout après {worker_timeout:.0f}s",
                worker_type=self.worker_type.value,
                task=self.task,
                execution_time=execution_time,
                errors=[f"TimeoutError: exécution dépassée ({worker_timeout:.0f}s)"],
            )
            self._state = WorkerState.FAILED
            self._finished_at = time.time()
            self._result = result
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
            self._state = WorkerState.FAILED
            self._finished_at = time.time()
            self._result = result

            # Reporter dans Memory MÊME en cas d'échec (apprentissage)
            self._report_to_memory(result)
            return result

    def cleanup(self) -> None:
        """
        Libère toutes les ressources du Worker.

        Appelé automatiquement par le context manager async,
        ou manuellement par le WorkerLifecycleManager.

        IMPORTANT : L'apprentissage dans Memory est DÉJÀ fait
        dans execute() AVANT que cleanup ne soit appelé.
        """
        # Libérer les références lourdes
        self.tools = []
        self.system_prompt = ""
        self.subtasks = []
        self.memory = None
        self.health_monitor = None

        # Marquer comme fermé
        self._state = WorkerState.CLOSED
        if not self._finished_at:
            self._finished_at = time.time()

    async def __aenter__(self):
        """Permet l'utilisation avec `async with`."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Garantit le cleanup même en cas d'exception."""
        self.cleanup()
        return False  # Ne pas avaler les exceptions

    def get_lifecycle_info(self) -> dict:
        """Retourne les infos complètes du cycle de vie."""
        model = self._model_config.model if self._model_config else "unknown"
        return {
            "worker_id": self._worker_id,
            "worker_type": self.worker_type.value,
            "state": self._state.value,
            "task": self.task[:80],
            "model": model,
            "created_at": self._created_at,
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "lifetime": self.lifetime,
            "success": self._result.success if self._result else None,
        }

    async def _real_execute(self) -> WorkerResult:
        """
        Exécution réelle via le LLM avec boucle tool_use native.

        Stage 6 : Route via le système multi-provider.
        Le LLM contrôle ses outils :
        1. On envoie la requête + définitions d'outils
        2. Si stop_reason == "tool_use" → on exécute l'outil demandé
        3. On renvoie le résultat au LLM
        4. On boucle jusqu'à stop_reason == "end_turn" (max N itérations)
        """
        memory_context = ""
        if self.memory:
            memory_context = self.memory.get_context(self.task)

        # Enrichir avec le contexte d'apprentissage (3 dernières exécutions du même type)
        learning_context = self._build_learning_context()
        if learning_context:
            memory_context += f"\n\n=== Apprentissage (exécutions précédentes) ===\n{learning_context}"

        # Construire le prompt système avec la date réelle
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%A %d %B %Y")
        time_str = now.strftime("%H:%M")

        prompt = self.system_prompt.format(
            task=self.task,
            memory_context=memory_context or "Aucun contexte disponible.",
            current_date=date_str,
            current_time=time_str,
        )

        # Message initial — inclut le contexte conversation si disponible
        user_message = ""
        if self.conversation_context:
            user_message += f"[Contexte de la conversation en cours]\n{self.conversation_context}\n\n"
        user_message += self.task
        if self.subtasks:
            user_message += "\n\nSous-tâches :\n" + "\n".join(
                f"- {st}" for st in self.subtasks
            )

        # Schémas tool_use pour ce type de Worker
        tool_schemas = ToolRegistry.get_tool_schemas_for_type(self.worker_type.value)

        # Messages conversationnels (accumule les tool_use)
        messages = [{"role": "user", "content": user_message}]

        # Nom d'agent pour le routing multi-provider
        agent_name = f"worker:{self.worker_type.value}"

        # Tracking des tool calls
        tool_calls = []
        # Limites par type de worker
        _type_max_iterations = {
            WorkerType.RESEARCHER: 5,   # search/fetch + code si besoin
            WorkerType.CODER: 10,       # Écrire, tester, corriger, sauvegarder
            WorkerType.ANALYST: 8,      # Collecter données + analyser + sauvegarder
            WorkerType.WRITER: 6,       # Rechercher + écrire + sauvegarder
            WorkerType.GENERIC: 8,      # Polyvalent, besoin de marge
        }
        max_iterations = _type_max_iterations.get(
            self.worker_type, self.config.resilience.max_tool_iterations
        )
        total_text_parts = []

        # Compteur d'extensions dynamiques
        _extensions_used = 0
        _max_extensions = 2  # Max 2 extensions de 2 itérations chacune
        _absolute_max = 15  # Plafond absolu, jamais dépassé
        iteration = 0

        while iteration < max_iterations:
            # Appel via le router multi-provider
            response_data = await self._provider_call(
                agent_name=agent_name,
                messages=messages,
                system=prompt,
                tools=tool_schemas if tool_schemas else None,
                max_tokens=self._model_config.max_tokens,
                temperature=self._model_config.temperature,
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
                # Le LLM a terminé — retourner UNIQUEMENT le texte final
                current_text = [b["text"] for b in content_blocks if b["type"] == "text"]
                if current_text:
                    final_output = "\n".join(current_text)
                else:
                    final_output = "\n".join(total_text_parts) if total_text_parts else "Tâche exécutée."

                # FILET DE SÉCURITÉ : un Researcher qui termine SANS appeler
                # web_search = hallucination quasi certaine. Marquer comme échec
                # pour que le retry utilise un autre provider (via LearningEngine).
                _tool_required_types = {WorkerType.RESEARCHER, WorkerType.CODER, WorkerType.ANALYST}
                if self.worker_type in _tool_required_types and not tool_calls and iteration == 0:
                    logger.warning(
                        "[Worker %s] %s a terminé sans aucun appel d'outil — probable hallucination",
                        self._worker_id, self.worker_type.value,
                    )
                    return WorkerResult(
                        success=False,
                        output=f"[Échec] Le worker {self.worker_type.value} n'a utilisé aucun outil. "
                               f"Réponse probablement hallucinée. Retry avec un provider compatible tool_use.",
                        worker_type=self.worker_type.value,
                        task=self.task,
                        tool_calls=[],
                        errors=["NO_TOOL_CALLS: worker finished without using any tools"],
                        metadata={"iterations": iteration + 1, "hallucination_detected": True},
                    )

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

            # Exécuter chaque outil avec timeout individuel (30s)
            tool_results = []
            tool_timeout = 30.0
            for tu in pending_tool_uses:
                try:
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, ToolRegistry.execute_tool, tu["name"], tu["input"]
                        ),
                        timeout=tool_timeout,
                    )
                    logger.info("Tool executed: %s", tu['name'])
                except asyncio.TimeoutError:
                    logger.error("Tool timeout (>%.0fs): %s", tool_timeout, tu['name'])
                    result = f"Timeout: outil {tu['name']} dépassé ({tool_timeout:.0f}s)"
                except Exception as e:
                    logger.error("Tool execution failed: %s: %s", tu['name'], e)
                    result = f"Erreur outil {tu['name']}: {e}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": str(result)[:5000],  # Limiter la taille
                })

            # Renvoyer les résultats au LLM
            messages.append({"role": "user", "content": tool_results})
            iteration += 1

            # Extension dynamique : si le worker n'a pas fini et qu'il reste des extensions
            if iteration >= max_iterations and _extensions_used < _max_extensions and max_iterations < _absolute_max:
                if total_text_parts:
                    last_text = total_text_parts[-1].lower()
                    needs_more = any(kw in last_text for kw in [
                        "je continue", "pas terminé", "need more", "en cours",
                        "je vais", "laissez-moi", "let me", "i'll continue",
                        "prochaine étape", "next step",
                    ])
                    if needs_more:
                        max_iterations = min(max_iterations + 2, _absolute_max)
                        _extensions_used += 1
                        logger.info(
                            "[Worker %s] Dynamic extension +2 iterations (total: %d, ext: %d)",
                            self._worker_id, max_iterations, _extensions_used,
                        )

        # Prendre le texte le plus récent
        if total_text_parts:
            final_output = total_text_parts[-1]
        else:
            final_output = "Tâche interrompue (max itérations atteint)."
        return WorkerResult(
            success=True,
            output=final_output,
            worker_type=self.worker_type.value,
            task=self.task,
            tool_calls=tool_calls,
            metadata={
                "iterations": max_iterations,
                "max_reached": True,
                "extensions": _extensions_used,
            },
        )

    async def _provider_call(
        self,
        agent_name: str,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> dict | None:
        """
        Appel LLM via le router multi-provider.

        Retourne le dict brut de la réponse (compatible boucle tool_use)
        ou None si toutes les tentatives échouent.
        """
        try:
            from neo_core.brain.providers.router import route_chat_raw

            response_data = await route_chat_raw(
                agent_name=agent_name,
                messages=messages,
                system=system,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Record dans le health monitor
            if self.health_monitor:
                success = response_data is not None
                self.health_monitor.record_api_call(success=success)

            return response_data

        except Exception as e:
            logger.debug("Provider call failed: %s", e)
            if self.health_monitor:
                self.health_monitor.record_api_call(success=False)
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

    def get_model_info(self) -> dict:
        """Retourne les infos du modèle utilisé par ce Worker."""
        return {
            "agent": f"Worker:{self.worker_type.value}",
            "model": self._model_config.model if self._model_config else "unknown",
            "task": self.task[:80],
        }

    def _build_learning_context(self) -> str:
        """
        Récupère les 3 dernières exécutions du même type de worker
        depuis Memory pour enrichir le contexte d'apprentissage.
        """
        if not self.memory or not self.memory.is_initialized:
            return ""

        try:
            records = self.memory._store.search_by_tags(
                [f"type:{self.worker_type.value}", "worker_execution"],
                limit=3,
            )
            if not records:
                return ""

            parts = []
            for rec in records:
                # Extraire un résumé court (max 200 chars par record)
                lines = rec.content.split("\n")
                status_line = (lines[0] if lines else "")[:200]
                parts.append(f"- {status_line}")

            result = "\n".join(parts)
            return result[:1000]  # Plafonner pour ne pas saturer le contexte
        except Exception as e:
            logger.debug("Learning context retrieval failed: %s", e)
            return ""

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
        except Exception as e:
            logger.debug("Memory reporting failed: %s", e)  # Ne pas faire crasher le worker pour un problème mémoire
