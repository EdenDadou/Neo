"""
Brain — Agent Orchestrateur
============================
Cortex exécutif du système Neo Core.

Responsabilités :
- Analyser les requêtes transmises par Vox
- Consulter le contexte enrichi par Memory
- Déterminer l'action optimale
- Générer dynamiquement des agents spécialisés (Workers) via Factory
- Apprendre des résultats des Workers

Authentification OAuth (méthode OpenClaw) :
- Méthode 1 : Bearer token + header beta "anthropic-beta: oauth-2025-04-20"
- Méthode 2 : Conversion OAuth → API key via /claude_cli/create_api_key
- Fallback : Clé API classique via LangChain

Stage 3 : Moteur d'Orchestration
Stage 5 : Résilience (retry, circuit breaker, health monitoring)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from neo_core.config import NeoConfig, default_config, get_agent_model
from neo_core.core.resilience import (
    RetryConfig,
    RetryableError,
    NonRetryableError,
    retry_with_backoff,
    CircuitBreaker,
    CircuitOpenError,
    HealthMonitor,
    create_resilience_from_config,
)
from neo_core.oauth import (
    is_oauth_token,
    get_valid_access_token,
    get_best_auth,
    get_api_key_from_oauth,
    OAUTH_BETA_HEADER,
)
from neo_core.teams.factory import WorkerFactory, TaskAnalysis
from neo_core.teams.worker import WorkerType, WorkerResult

if TYPE_CHECKING:
    from neo_core.core.memory_agent import MemoryAgent

BRAIN_SYSTEM_PROMPT = """Tu es Brain, le cortex exécutif du système Neo Core.
Date et heure actuelles : {current_date}, {current_time}

Ton rôle :
- Tu reçois les requêtes structurées par Vox (l'interface humaine).
- Tu analyses chaque requête et détermines la meilleure stratégie de réponse.
- Tu consultes le contexte fourni par Memory pour enrichir tes réponses.
- Tu coordonnes l'exécution des tâches et délègues aux Workers spécialisés si nécessaire.

Contexte mémoire :
{memory_context}

Règles :
- Sois précis, stratégique et orienté résultat.
- Si une tâche est complexe, décompose-la en sous-tâches.
- Indique clairement quand tu as besoin de plus d'informations.
- Tu es le décideur final sur la stratégie d'exécution.
- Réponds de manière concise et naturelle, pas de markdown excessif.
"""

# Prompt pour la décomposition LLM de tâches
DECOMPOSE_PROMPT = """Analyse cette requête et détermine comment la traiter.

Requête : {request}

Contexte mémoire : {memory_context}

Réponds en JSON strict avec cette structure :
{{
  "action": "direct_response" ou "delegate_worker",
  "worker_type": "researcher" | "coder" | "summarizer" | "analyst" | "writer" | "translator" | "generic",
  "subtasks": ["sous-tâche 1", "sous-tâche 2", ...],
  "reasoning": "explication courte de ta décision",
  "confidence": 0.0 à 1.0
}}

Règles :
- "direct_response" si c'est une question simple, une conversation, ou une demande rapide
- "delegate_worker" si ça nécessite de la recherche, du code, de l'analyse, ou une tâche structurée
- Le worker_type doit correspondre au type de tâche
- Les subtasks doivent être des actions concrètes et ordonnées
- Réponds UNIQUEMENT avec le JSON, rien d'autre.
"""


@dataclass
class BrainDecision:
    """Représente une décision prise par Brain."""
    action: str  # "direct_response" | "delegate_worker" | "delegate_crew"
    response: Optional[str] = None
    subtasks: list[str] = field(default_factory=list)
    confidence: float = 1.0
    worker_type: Optional[str] = None  # Stage 3 : type de worker recommandé
    reasoning: str = ""  # Stage 3 : justification de la décision


@dataclass
class Brain:
    """
    Agent Brain — Orchestrateur du système Neo Core.

    Analyse les requêtes, consulte Memory, et détermine
    la meilleure action à entreprendre.

    Stage 3 : Peut créer des Workers spécialisés via Factory
    Stage 5 : Résilience (retry, circuit breaker, health monitoring)
    """
    config: NeoConfig = field(default_factory=lambda: default_config)
    memory: Optional[MemoryAgent] = None
    _llm: Optional[object] = None
    _mock_mode: bool = False
    _oauth_mode: bool = False
    _auth_method: str = ""
    _anthropic_client: Optional[object] = None
    _factory: Optional[WorkerFactory] = None
    _health: Optional[HealthMonitor] = None
    _retry_config: Optional[RetryConfig] = None
    _model_config: Optional[object] = None

    def __post_init__(self):
        self._mock_mode = self.config.is_mock_mode()
        self._model_config = get_agent_model("brain")

        # Stage 5 : Initialiser la résilience
        retry, circuit, health = create_resilience_from_config(self.config.resilience)
        self._retry_config = retry
        self._health = health

        if not self._mock_mode:
            self._init_llm()

    @property
    def factory(self) -> WorkerFactory:
        """Accès lazy à la Factory (créée à la demande)."""
        if self._factory is None:
            self._factory = WorkerFactory(config=self.config, memory=self.memory)
        return self._factory

    @property
    def health(self) -> HealthMonitor:
        """Accès au moniteur de santé."""
        if self._health is None:
            _, _, self._health = create_resilience_from_config(self.config.resilience)
        return self._health

    def get_model_info(self) -> dict:
        """Retourne les infos du modèle utilisé par Brain."""
        return {
            "agent": "Brain",
            "model": self._model_config.model if self._model_config else "unknown",
            "role": "Orchestration, décision, raisonnement",
            "has_llm": not self._mock_mode,
            "auth_method": self._auth_method,
        }

    def get_system_health(self) -> dict:
        """Retourne le rapport de santé complet du système."""
        report = self.health.get_health_report()
        report["brain"] = {
            "mock_mode": self._mock_mode,
            "auth_method": self._auth_method,
            "oauth_mode": self._oauth_mode,
            "model": self._model_config.model if self._model_config else "unknown",
        }
        if self.memory and self.memory.is_initialized:
            stats = self.memory.get_stats()
            report["memory"]["stats"] = stats
            report["memory"]["model"] = self.memory.get_model_info()
            self.health.set_memory_health(True)
        else:
            self.health.set_memory_health(self.memory is not None)

        # Ajouter les infos modèle de chaque agent au rapport
        report["agent_models"] = {
            "brain": self.get_model_info(),
        }
        if self.memory:
            report["agent_models"]["memory"] = self.memory.get_model_info()

        return report

    # ─── Initialisation LLM / Auth ──────────────────────────

    def _init_llm(self) -> None:
        """Initialise le LLM avec la meilleure méthode d'auth."""
        try:
            api_key = self.config.llm.api_key
            if is_oauth_token(api_key):
                self._init_oauth(api_key)
            else:
                self._init_langchain(api_key)
        except Exception as e:
            print(f"[Brain] Impossible d'initialiser le LLM: {e}")
            self._mock_mode = True
            self._auth_method = "mock"

    def _init_oauth(self, token: str) -> None:
        """Init OAuth avec fallback automatique."""
        converted_key = get_api_key_from_oauth()
        if converted_key:
            print(f"[Brain] Clé API convertie depuis OAuth détectée")
            self._init_langchain(converted_key)
            self._auth_method = "converted_api_key"
            return
        self._oauth_mode = True
        self._init_oauth_bearer(token)

    def _init_oauth_bearer(self, token: str) -> None:
        """Init Bearer + beta header (méthode OpenClaw)."""
        import anthropic
        import httpx

        valid_token = get_valid_access_token()
        if not valid_token:
            valid_token = token

        custom_transport = httpx.AsyncHTTPTransport()
        self._anthropic_client = anthropic.AsyncAnthropic(
            api_key="dummy",
            default_headers={
                "Authorization": f"Bearer {valid_token}",
                "anthropic-beta": OAUTH_BETA_HEADER,
            },
        )
        self._current_token = valid_token
        self._auth_method = "oauth_bearer"
        print(f"[Brain] Mode OAuth Bearer + beta header activé")

    def _refresh_oauth_client(self) -> bool:
        """Rafraîchit le client OAuth."""
        converted_key = get_api_key_from_oauth()
        if converted_key:
            self._init_langchain(converted_key)
            self._oauth_mode = False
            self._auth_method = "converted_api_key"
            return True
        valid_token = get_valid_access_token()
        if valid_token:
            self._init_oauth_bearer(valid_token)
            return True
        return False

    def _init_langchain(self, api_key: str) -> None:
        """Init LangChain avec clé API classique et modèle dédié Brain."""
        from langchain_anthropic import ChatAnthropic
        self._llm = ChatAnthropic(
            model=self._model_config.model,
            api_key=api_key,
            temperature=self._model_config.temperature,
            max_tokens=self._model_config.max_tokens,
        )
        self._oauth_mode = False
        if not self._auth_method:
            self._auth_method = "langchain"
        print(f"[Brain] LLM initialisé : {self._model_config.model}")

    # ─── Connexions ─────────────────────────────────────────

    def connect_memory(self, memory: MemoryAgent) -> None:
        """Connecte Brain au système mémoire."""
        self.memory = memory
        # Mettre à jour la factory si elle existe déjà
        if self._factory:
            self._factory.memory = memory

    def get_memory_context(self, request: str) -> str:
        """
        Récupère le contexte pertinent depuis Memory.
        Inclut les conseils du LearningEngine si disponibles.
        """
        if not self.memory:
            return "Aucun contexte mémoire disponible."

        context = self.memory.get_context(request)

        # Ajouter les conseils d'apprentissage au contexte
        try:
            # On récupère les conseils pour le type le plus probable
            worker_type = self.factory.classify_task(request)
            if worker_type != WorkerType.GENERIC:
                advice = self.memory.get_learning_advice(request, worker_type.value)
                learning_context = advice.to_context_string()
                if learning_context:
                    context += f"\n\n=== Apprentissage ===\n{learning_context}"
        except Exception:
            pass

        return context

    # ─── Analyse et décision ────────────────────────────────

    def analyze_complexity(self, request: str) -> str:
        """
        Analyse la complexité d'une requête.
        Retourne : "simple" | "moderate" | "complex"
        """
        word_count = len(request.split())
        if word_count < 15:
            return "simple"
        elif word_count < 50:
            return "moderate"
        return "complex"

    def make_decision(self, request: str) -> BrainDecision:
        """
        Prend une décision stratégique sur la manière de traiter la requête.

        Boucle d'apprentissage fermée :
        1. Classifie la tâche (Factory)
        2. Consulte l'historique des erreurs (Memory/LearningEngine)
        3. Ajuste la stratégie si nécessaire
        4. Retourne la décision avec confiance ajustée
        """
        complexity = self.analyze_complexity(request)
        worker_type = self.factory.classify_task(request)

        # Si un type spécifique est détecté → TOUJOURS déléguer
        if worker_type != WorkerType.GENERIC:
            subtasks = self.factory._basic_decompose(request, worker_type)
            confidence = 0.8

            # ── Consultation de l'historique (Learning Loop) ──
            advice = self._consult_learning(request, worker_type.value)

            if advice:
                # Ajuster la confiance en fonction de l'historique
                confidence += advice.confidence_adjustment
                confidence = max(0.1, min(1.0, confidence))

                reasoning_parts = [f"Type {worker_type.value} détecté"]

                # Si un worker alternatif est recommandé
                if advice.recommended_worker and advice.recommended_worker != worker_type.value:
                    try:
                        alt_type = WorkerType(advice.recommended_worker)
                        worker_type = alt_type
                        subtasks = self.factory._basic_decompose(request, alt_type)
                        reasoning_parts.append(
                            f"→ substitué par {alt_type.value} (historique)"
                        )
                    except ValueError:
                        pass

                # Si des avertissements existent, les inclure
                if advice.warnings:
                    reasoning_parts.append(f"⚠ {'; '.join(advice.warnings[:2])}")

                reasoning = " | ".join(reasoning_parts)
            else:
                reasoning = f"Type {worker_type.value} détecté → Worker (complexité: {complexity})"

            return BrainDecision(
                action="delegate_worker",
                subtasks=subtasks,
                confidence=confidence,
                worker_type=worker_type.value,
                reasoning=reasoning,
            )

        # Requêtes complexes sans type spécifique → Worker generic
        if complexity == "complex":
            subtasks = self._decompose_task(request)
            confidence = 0.5

            # Consultation learning même pour les workers génériques
            advice = self._consult_learning(request, "generic")
            reasoning = f"Tâche complexe → Worker {worker_type.value}"

            if advice and advice.relevant_skills:
                # Si on a une compétence passée pertinente, l'utiliser
                best_skill = advice.relevant_skills[0]
                if best_skill.worker_type != "generic":
                    try:
                        alt_type = WorkerType(best_skill.worker_type)
                        worker_type = alt_type
                        subtasks = self.factory._basic_decompose(request, alt_type)
                        reasoning = (
                            f"Tâche complexe → substitué par {alt_type.value} "
                            f"(compétence acquise: {best_skill.name})"
                        )
                        confidence = 0.7
                    except ValueError:
                        pass

            return BrainDecision(
                action="delegate_worker",
                subtasks=subtasks,
                confidence=confidence,
                worker_type=worker_type.value,
                reasoning=reasoning,
            )

        # Requêtes simples/modérées sans type spécifique → réponse directe
        # Mais vérifier si une compétence passée suggère de déléguer
        advice = self._consult_learning(request, "generic")
        if advice and advice.relevant_skills:
            best_skill = advice.relevant_skills[0]
            if best_skill.success_count >= 2:
                # On a une compétence prouvée pour ce type de requête → déléguer
                try:
                    skill_type = WorkerType(best_skill.worker_type)
                    subtasks = self.factory._basic_decompose(request, skill_type)
                    return BrainDecision(
                        action="delegate_worker",
                        subtasks=subtasks,
                        confidence=0.7,
                        worker_type=skill_type.value,
                        reasoning=(
                            f"Requête {complexity} mais compétence acquise "
                            f"({best_skill.name}, ×{best_skill.success_count}) → Worker"
                        ),
                    )
                except ValueError:
                    pass

        return BrainDecision(
            action="direct_response",
            subtasks=[request] if complexity == "moderate" else [],
            confidence=0.9 if complexity == "simple" else 0.7,
            reasoning=f"Requête {complexity} générique → réponse directe",
        )

    def _consult_learning(self, request: str, worker_type: str) -> object | None:
        """
        Consulte le LearningEngine de Memory pour obtenir des conseils.
        Retourne un LearningAdvice ou None si Memory n'est pas disponible.
        """
        if not self.memory or not self.memory.is_initialized:
            return None

        try:
            return self.memory.get_learning_advice(request, worker_type)
        except Exception:
            return None

    def _decompose_task(self, request: str) -> list[str]:
        """
        Décompose une tâche complexe en sous-tâches.
        Utilise les heuristiques de la Factory (compatible mock).
        """
        worker_type = self.factory.classify_task(request)
        return self.factory._basic_decompose(request, worker_type)

    async def _decompose_task_with_llm(self, request: str,
                                        memory_context: str) -> TaskAnalysis:
        """
        Décomposition LLM-powered d'une tâche.
        Fallback sur les heuristiques si le LLM échoue.
        """
        if self._mock_mode:
            return self.factory.analyze_task(request)

        try:
            prompt = DECOMPOSE_PROMPT.format(
                request=request,
                memory_context=memory_context[:500],
            )

            response_text = await self._raw_llm_call(prompt)

            # Parser le JSON
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

            data = json.loads(cleaned)

            worker_type_str = data.get("worker_type", "generic")
            try:
                worker_type = WorkerType(worker_type_str)
            except ValueError:
                worker_type = WorkerType.GENERIC

            return TaskAnalysis(
                worker_type=worker_type,
                primary_task=request,
                subtasks=data.get("subtasks", [request]),
                required_tools=[
                    getattr(t, "name", str(t))
                    for t in self.factory._get_tools_for_type(worker_type)
                ] if hasattr(self.factory, '_get_tools_for_type') else [],
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.7)),
            )

        except Exception:
            return self.factory.analyze_task(request)

    # ─── Pipeline principal ─────────────────────────────────

    async def process(self, request: str,
                      conversation_history: list[BaseMessage] | None = None) -> str:
        """
        Pipeline principal de Brain.

        Stage 5 :
        1. Vérifie la santé du système (circuit breaker)
        2. Récupère le contexte mémoire
        3. Prend une décision (direct / worker)
        4. Si direct → répond via LLM (avec retry)
        5. Si worker → crée un Worker, exécute, apprend du résultat
        """
        memory_context = self.get_memory_context(request)

        if self._mock_mode:
            decision = self.make_decision(request)

            if decision.action == "delegate_worker" and decision.worker_type:
                return await self._execute_with_worker(request, decision, memory_context)

            complexity = self.analyze_complexity(request)
            return self._mock_response(request, complexity, memory_context)

        # Stage 5 : Vérifier le circuit breaker
        if not self.health.can_make_api_call():
            return (
                "[Brain] Le système est temporairement indisponible "
                "(trop d'erreurs consécutives). Réessayez dans quelques instants."
            )

        try:
            decision = self.make_decision(request)

            if decision.action == "delegate_worker" and decision.worker_type:
                try:
                    analysis = await self._decompose_task_with_llm(request, memory_context)
                except Exception:
                    analysis = self.factory.analyze_task(request)

                return await self._execute_with_worker(request, decision, memory_context, analysis)

            # Réponse directe
            if self._oauth_mode:
                return await self._oauth_response(request, memory_context, conversation_history)
            else:
                return await self._llm_response(request, memory_context, conversation_history)

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            # Gestion erreur auth OAuth
            if self._oauth_mode and ("authentication" in error_msg.lower()
                                     or "unauthorized" in error_msg.lower()
                                     or "401" in error_msg):
                print(f"[Brain] Erreur auth OAuth, tentative de fallback...")

                converted_key = get_api_key_from_oauth()
                if converted_key:
                    print(f"[Brain] Conversion OAuth → API key réussie, retry...")
                    self._init_langchain(converted_key)
                    self._oauth_mode = False
                    self._auth_method = "converted_api_key"
                    try:
                        return await self._llm_response(request, memory_context, conversation_history)
                    except Exception as retry_e:
                        return f"[Brain Erreur] Après conversion: {type(retry_e).__name__}: {str(retry_e)[:300]}"

                if self._refresh_oauth_client():
                    try:
                        if self._oauth_mode:
                            return await self._oauth_response(request, memory_context, conversation_history)
                        else:
                            return await self._llm_response(request, memory_context, conversation_history)
                    except Exception as retry_e:
                        return f"[Brain Erreur] Après refresh: {type(retry_e).__name__}: {str(retry_e)[:300]}"

            return f"[Brain Erreur] {error_type}: {error_msg[:500]}"

    async def _execute_with_worker(self, request: str, decision: BrainDecision,
                                    memory_context: str,
                                    analysis: TaskAnalysis | None = None) -> str:
        """
        Crée et exécute un Worker pour une tâche complexe.
        Stage 5 : Passe le health monitor au Worker.
        """
        try:
            worker_type = WorkerType(decision.worker_type)
        except (ValueError, TypeError):
            worker_type = WorkerType.GENERIC

        if analysis:
            worker = self.factory.create_worker(analysis)
        else:
            worker = self.factory.create_worker_for_type(
                worker_type=worker_type,
                task=request,
                subtasks=decision.subtasks,
            )

        # Stage 5 : Passer le health monitor au Worker
        worker.health_monitor = self._health

        result = await worker.execute()

        # Apprentissage
        await self._learn_from_result(request, decision, result)

        if result.success:
            return result.output
        else:
            return f"[Worker {worker_type.value} échoué] {result.output}"

    async def _learn_from_result(self, request: str, decision: BrainDecision,
                                  result: WorkerResult) -> None:
        """
        Apprentissage à partir du résultat d'un Worker.

        Boucle fermée :
        - Enregistre dans le LearningEngine (patterns d'erreur, compétences)
        - Stocke aussi un résumé en mémoire classique pour le contexte
        """
        if not self.memory:
            return

        try:
            # 1. Enregistrer dans le LearningEngine (boucle fermée)
            self.memory.record_execution_result(
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

            self.memory.store_memory(
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
        except Exception:
            pass

    # ─── Appels LLM ────────────────────────────────────────

    async def _raw_llm_call(self, prompt: str) -> str:
        """
        Appel LLM brut (sans historique ni system prompt Brain).
        Stage 5 : Avec retry et health monitoring.
        """
        if self._oauth_mode:
            import httpx as _httpx

            valid_token = get_valid_access_token()
            token = valid_token or getattr(self, "_current_token", None)
            if not token:
                raise Exception("Aucun token OAuth valide")

            headers = {
                "Authorization": f"Bearer {token}",
                "anthropic-beta": OAUTH_BETA_HEADER,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

            payload = {
                "model": self._model_config.model,
                "max_tokens": 1024,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}],
            }

            timeout = self.config.resilience.api_timeout

            async def _do_call():
                async with _httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers=headers,
                        json=payload,
                        timeout=timeout,
                    )

                status = response.status_code
                if self._health:
                    self._health.record_api_call(success=(status == 200))

                if status == 200:
                    data = response.json()
                    return data["content"][0]["text"]

                if status in self._retry_config.retryable_status_codes:
                    raise RetryableError(f"API {status}", status_code=status)

                error_data = response.json() if "json" in response.headers.get("content-type", "") else {}
                error_msg = error_data.get("error", {}).get("message", response.text[:300])
                raise NonRetryableError(f"API {status}: {error_msg}", status_code=status)

            return await retry_with_backoff(_do_call, self._retry_config)
        else:
            result = await self._llm.ainvoke(prompt)
            if self._health:
                self._health.record_api_call(success=True)
            return result.content

    async def _oauth_response(self, request: str, memory_context: str,
                              conversation_history: list[BaseMessage] | None = None) -> str:
        """Génère une réponse via OAuth Bearer + beta header."""
        import httpx as _httpx

        messages = []
        if conversation_history:
            for msg in conversation_history:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})
        messages.append({"role": "user", "content": request})

        from datetime import datetime
        now = datetime.now()
        system_prompt = BRAIN_SYSTEM_PROMPT.format(
            memory_context=memory_context,
            current_date=now.strftime("%A %d %B %Y"),
            current_time=now.strftime("%H:%M"),
        )

        valid_token = get_valid_access_token()
        if valid_token:
            token = valid_token
        else:
            token = getattr(self, "_current_token", None)
            if not token:
                return "[Brain Erreur] Aucun token OAuth valide disponible"

        headers = {
            "Authorization": f"Bearer {token}",
            "anthropic-beta": OAUTH_BETA_HEADER,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self._model_config.model,
            "max_tokens": self._model_config.max_tokens,
            "temperature": self._model_config.temperature,
            "system": system_prompt,
            "messages": messages,
        }

        timeout = self.config.resilience.api_timeout

        async def _do_call():
            async with _httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )

            status = response.status_code
            if self._health:
                self._health.record_api_call(success=(status == 200))

            if status == 200:
                data = response.json()
                return data["content"][0]["text"]

            if status in self._retry_config.retryable_status_codes:
                raise RetryableError(f"API {status}", status_code=status)

            error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            error_msg = error_data.get("error", {}).get("message", response.text[:300])
            raise Exception(f"API {status}: {error_msg}")

        return await retry_with_backoff(_do_call, self._retry_config)

    async def _llm_response(self, request: str, memory_context: str,
                            conversation_history: list[BaseMessage] | None = None) -> str:
        """Génère une réponse via LangChain."""
        from datetime import datetime
        now = datetime.now()

        prompt = ChatPromptTemplate.from_messages([
            ("system", BRAIN_SYSTEM_PROMPT),
            MessagesPlaceholder("conversation_history", optional=True),
            ("human", "{request}"),
        ])
        chain = prompt | self._llm
        result = await chain.ainvoke({
            "memory_context": memory_context,
            "current_date": now.strftime("%A %d %B %Y"),
            "current_time": now.strftime("%H:%M"),
            "conversation_history": conversation_history or [],
            "request": request,
        })
        if self._health:
            self._health.record_api_call(success=True)
        return result.content

    def _mock_response(self, request: str, complexity: str, context: str) -> str:
        """Réponse mock pour les tests sans clé API."""
        return (
            f"[Brain Mock] Requête reçue (complexité: {complexity}). "
            f"Contexte: {context[:100]}. "
            f"Analyse de: '{request[:80]}...'" if len(request) > 80 else
            f"[Brain Mock] Requête reçue (complexité: {complexity}). "
            f"Contexte: {context[:100]}. "
            f"Analyse de: '{request}'"
        )
