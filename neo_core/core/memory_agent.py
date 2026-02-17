"""
Memory — Agent Bibliothécaire (Étape 2 — Complet)
====================================================
Hippocampe et système de consolidation des connaissances.

Agent LangChain complet qui pilote le module memory/ :
- MemoryStore : stockage persistant (ChromaDB + SQLite)
- ContextEngine : injection de contexte intelligent
- MemoryConsolidator : nettoyage et synthèse

Responsabilités :
- Organiser, nettoyer et synthétiser la mémoire
- Archiver succès et échecs des agents
- Référencer les nouvelles compétences
- Injecter du contexte pertinent
- Mémoire long terme persistante
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from neo_core.config import NeoConfig, default_config, get_agent_model
from neo_core.memory.store import MemoryStore, MemoryRecord
from neo_core.memory.context import ContextEngine, ContextBlock
from neo_core.memory.consolidator import MemoryConsolidator
from neo_core.memory.learning import LearningEngine, LearningAdvice
from neo_core.memory.task_registry import TaskRegistry, Task, Epic
from neo_core.core.persona import PersonaEngine
from neo_core.oauth import is_oauth_token, get_valid_access_token, OAUTH_BETA_HEADER

logger = logging.getLogger(__name__)

# Prompt pour la synthèse intelligente de conversations
MEMORY_SUMMARIZE_PROMPT = """Tu es Memory, le bibliothécaire du système Neo Core.
Synthétise ces échanges conversationnels en un résumé concis et utile.

Échanges :
{conversations}

Règles :
- Identifie les informations clés, préférences utilisateur, et décisions prises
- Conserve les faits importants, élimine le superflu
- Structure le résumé en points essentiels
- Maximum 5-8 points
- Réponds UNIQUEMENT avec le résumé, rien d'autre.
"""

# Prompt pour l'enrichissement de contexte
MEMORY_CONTEXT_PROMPT = """Tu es Memory, le bibliothécaire du système Neo Core.
À partir des souvenirs retrouvés, synthétise un contexte pertinent pour la requête.

Requête : {query}

Souvenirs retrouvés :
{memories}

Règles :
- Sélectionne uniquement les informations pertinentes pour la requête
- Organise-les de manière logique
- Sois concis (max 200 mots)
- Si aucun souvenir n'est pertinent, dis-le clairement
- Réponds UNIQUEMENT avec le contexte synthétisé, rien d'autre.
"""


@dataclass
class MemoryAgent:
    """
    Agent Memory — Bibliothécaire du système Neo Core.

    Pilote le système mémoire complet :
    - Stockage persistant (ChromaDB + SQLite)
    - Injection de contexte intelligent
    - Consolidation périodique
    """
    config: NeoConfig = field(default_factory=lambda: default_config)
    _store: Optional[MemoryStore] = field(default=None, init=False)
    _context_engine: Optional[ContextEngine] = field(default=None, init=False)
    _consolidator: Optional[MemoryConsolidator] = field(default=None, init=False)
    _learning: Optional[LearningEngine] = field(default=None, init=False)
    _task_registry: Optional[TaskRegistry] = field(default=None, init=False)
    _persona_engine: Optional[PersonaEngine] = field(default=None, init=False)
    _initialized: bool = field(default=False, init=False)
    _turn_count: int = field(default=0, init=False)
    _consolidation_interval: int = 50  # Consolide tous les N tours
    _llm: Optional[object] = field(default=None, init=False)
    _mock_mode: bool = field(default=False, init=False)
    _model_config: Optional[object] = field(default=None, init=False)

    def initialize(self) -> None:
        """Initialise le système mémoire complet + LLM dédié."""
        self._store = MemoryStore(self.config.memory)
        self._store.initialize()

        self._context_engine = ContextEngine(self._store, self.config.memory)
        self._consolidator = MemoryConsolidator(self._store, self.config)
        self._learning = LearningEngine(self._store)
        self._task_registry = TaskRegistry(self._store)

        # Stage 9 — PersonaEngine (identité + profil utilisateur)
        self._init_persona_engine()

        self._initialized = True
        self._mock_mode = self.config.is_mock_mode()
        self._model_config = get_agent_model("memory")

        if not self._mock_mode:
            self._init_llm()

        # Charger la documentation système en mémoire (auto-connaissance)
        self._load_system_docs()

    def _init_llm(self) -> None:
        """Initialise le LLM dédié de Memory (Haiku — économique)."""
        try:
            api_key = self.config.llm.api_key
            if is_oauth_token(api_key):
                self._llm = None  # Sera géré via _memory_llm_call
            else:
                from langchain_anthropic import ChatAnthropic
                self._llm = ChatAnthropic(
                    model=self._model_config.model,
                    api_key=api_key,
                    temperature=self._model_config.temperature,
                    max_tokens=self._model_config.max_tokens,
                )
            logger.info("[Memory] LLM initialisé : %s", self._model_config.model)
        except Exception as e:
            logger.error("[Memory] LLM non disponible (%s), mode heuristique", e)

    def _init_persona_engine(self) -> None:
        """Initialise le moteur de personnalité et d'empathie (Stage 9)."""
        try:
            self._persona_engine = PersonaEngine(store=self._store)
            self._persona_engine.initialize()
            logger.info("[Memory] PersonaEngine initialisé")
        except Exception as e:
            logger.error("[Memory] PersonaEngine non disponible (%s)", e)

    def _load_system_docs(self) -> None:
        """
        Charge la documentation système en mémoire pour l'auto-connaissance.

        Lit les fichiers Markdown de data/system_docs/ et les stocke
        avec une importance haute. Vérifie le hash pour ne pas recharger
        si le contenu n'a pas changé.
        """
        import hashlib
        from pathlib import Path

        docs_dir = self.config.data_dir / "system_docs"
        if not docs_dir.exists():
            return

        for doc_path in sorted(docs_dir.glob("*.md")):
            try:
                content = doc_path.read_text(encoding="utf-8")
                if not content.strip():
                    continue

                # Hash pour éviter les duplicats
                doc_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                tag = f"system_doc:{doc_path.stem}"

                # Vérifier si déjà chargé (même hash)
                existing = self._store.search_by_tags([tag], limit=1)
                if existing:
                    existing_hash = existing[0].metadata.get("doc_hash", "")
                    if existing_hash == doc_hash:
                        logger.debug("[Memory] System doc '%s' already loaded (hash match)", doc_path.stem)
                        continue
                    # Hash différent → supprimer l'ancien et recharger
                    for old in existing:
                        self._store.delete(old.id)

                # Découper en sections si le document est long (>2000 chars)
                sections = self._split_doc_sections(content)
                for i, section in enumerate(sections):
                    section_tag = f"{tag}:part{i}" if len(sections) > 1 else tag
                    self._store.store(
                        content=section,
                        source="system_documentation",
                        tags=["system_doc", tag, "architecture", "capacités"],
                        importance=0.95,
                        metadata={"doc_hash": doc_hash, "filename": doc_path.name, "part": i},
                    )

                logger.info("[Memory] System doc loaded: %s (%d sections)", doc_path.stem, len(sections))

            except Exception as e:
                logger.debug("[Memory] Failed to load system doc %s: %s", doc_path.name, e)

    @staticmethod
    def _split_doc_sections(content: str, max_chars: int = 2000) -> list[str]:
        """Découpe un document en sections basées sur les titres ## ou par taille."""
        import re
        # Découper sur les titres de niveau 2 (##)
        parts = re.split(r'\n(?=## )', content)
        sections = []
        current = ""

        for part in parts:
            if len(current) + len(part) > max_chars and current:
                sections.append(current.strip())
                current = part
            else:
                current += ("\n" if current else "") + part

        if current.strip():
            sections.append(current.strip())

        return sections if sections else [content]

    async def _memory_llm_call(self, prompt: str) -> Optional[str]:
        """
        Appel LLM dédié pour Memory (synthèse/contexte).

        Route via le système multi-provider (Ollama, Groq, Gemini, Anthropic).
        Fallback automatique vers Anthropic direct si aucun provider configuré.
        """
        if self._mock_mode:
            return None

        try:
            from neo_core.providers.router import route_chat

            response = await route_chat(
                agent_name="memory",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._model_config.max_tokens,
                temperature=self._model_config.temperature,
            )

            if response.text and not response.text.startswith("[Erreur"):
                return response.text
            return None

        except Exception as e:
            logger.debug("Memory LLM call via route_chat failed: %s", e)
            # Fallback LangChain legacy
            if self._llm:
                try:
                    result = await self._llm.ainvoke(prompt)
                    return result.content
                except Exception as e:
                    logger.debug("Memory LLM ainvoke fallback failed: %s", e)

        return None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def store(self) -> MemoryStore:
        """Accès direct au store (pour les tests et usages avancés)."""
        return self._store

    def store_memory(self, content: str, source: str = "conversation",
                     tags: list[str] | None = None, importance: float = 0.5,
                     metadata: dict | None = None) -> str:
        """
        Stocke un souvenir en mémoire persistante.
        Retourne l'ID du souvenir.
        """
        if not self._initialized:
            raise RuntimeError("Memory n'est pas initialisé. Appelez initialize() d'abord.")

        return self._store.store(
            content=content,
            source=source,
            tags=tags,
            importance=importance,
            metadata=metadata,
        )

    def get_context(self, query: str) -> str:
        """
        Retourne le contexte pertinent pour une requête sous forme de texte.
        Interface principale utilisée par Brain.
        """
        if not self._initialized:
            return "Aucun contexte mémoire disponible."

        block = self._context_engine.build_context(query)
        return block.to_string()

    def get_context_block(self, query: str) -> ContextBlock:
        """
        Retourne le contexte sous forme structurée (ContextBlock).
        Pour les usages avancés nécessitant un accès aux records individuels.
        """
        if not self._initialized:
            return ContextBlock()

        return self._context_engine.build_context(query)

    def on_conversation_turn(self, user_message: str, ai_response: str) -> None:
        """
        Appelé après chaque échange conversationnel.
        Stocke l'échange, enrichit les tâches actives, et déclenche
        la consolidation si nécessaire.

        Memory est le garant du savoir et du progrès sur les missions.
        """
        if not self._initialized:
            return

        self._context_engine.store_conversation_turn(user_message, ai_response)

        # Enrichir les tâches actives avec le contexte de la conversation
        self._enrich_active_tasks(user_message, ai_response)

        # Stage 9 — Analyse automatique pour apprentissage persona/user
        self.analyze_conversation(user_message, ai_response)

        self._turn_count += 1
        if self._turn_count % self._consolidation_interval == 0:
            self.consolidate()

    def _enrich_active_tasks(self, user_message: str, ai_response: str) -> None:
        """
        Enrichit automatiquement les tâches in_progress avec le contexte
        des conversations. Memory collecte les informations pertinentes
        et les associe aux missions en cours.
        """
        if not self._task_registry:
            return

        try:
            active_tasks = self._task_registry.get_active_tasks()
            if not active_tasks:
                return

            # Extraire un résumé court de l'échange pour le contexte
            exchange_summary = (
                f"[{self._turn_count}] Q: {user_message[:100]} "
                f"→ R: {ai_response[:150]}"
            )

            for task in active_tasks:
                # Vérifier si l'échange est pertinent pour cette tâche
                # (heuristique simple : mots clés en commun)
                task_words = set(task.description.lower().split())
                msg_words = set(user_message.lower().split())
                common = task_words & msg_words

                # Au moins 2 mots en commun (hors mots vides) → pertinent
                stop_words = {"le", "la", "les", "de", "du", "des", "un", "une",
                              "et", "est", "en", "pour", "dans", "sur", "avec",
                              "que", "qui", "je", "tu", "il", "on", "ce", "ça"}
                meaningful_common = common - stop_words

                if len(meaningful_common) >= 2:
                    self._task_registry.add_task_context(
                        task.id, exchange_summary
                    )

                    # Si la tâche fait partie d'un epic, enrichir l'epic aussi
                    if task.epic_id:
                        self._task_registry.add_epic_context(
                            task.epic_id,
                            f"[Tâche {task.id[:8]}] {exchange_summary[:200]}"
                        )

        except Exception as e:
            logger.debug("Task enrichment failed: %s", e)  # Ne jamais bloquer le flux principal

    async def smart_summarize(self, entries: list[MemoryRecord]) -> Optional[str]:
        """
        Synthèse intelligente via LLM (Haiku).
        Fallback sur la synthèse heuristique si LLM indisponible.
        """
        if not entries:
            return None

        # Tenter la synthèse LLM
        if not self._mock_mode and (self._llm or is_oauth_token(self.config.llm.api_key or "")):
            try:
                conversations = "\n\n".join(
                    f"[{e.source}] {e.content[:300]}" for e in entries[:20]
                )
                prompt = MEMORY_SUMMARIZE_PROMPT.format(conversations=conversations)
                result = await self._memory_llm_call(prompt)
                if result:
                    return result.strip()
            except Exception as e:
                logger.debug("Smart summarize LLM call failed: %s", e)

        # Fallback heuristique
        return self._consolidator.summarize_conversation(entries)

    def consolidate(self) -> dict:
        """
        Lance une consolidation complète de la mémoire.
        Retourne un rapport de consolidation.
        """
        if not self._initialized:
            return {}

        report = self._consolidator.full_consolidation()
        return {
            "entries_before": report.entries_before,
            "entries_after": report.entries_after,
            "deleted": report.entries_deleted,
            "merged": report.entries_merged,
            "promoted": report.entries_promoted,
        }

    def search(self, query: str, n_results: int = 5) -> list[MemoryRecord]:
        """Recherche sémantique dans la mémoire."""
        if not self._initialized:
            return []
        return self._store.search_semantic(query, n_results=n_results)

    def get_stats(self) -> dict:
        """Retourne des statistiques sur la mémoire."""
        if not self._initialized:
            return {
                "total_entries": 0,
                "initialized": False,
                "has_vector_search": False,
            }

        store_stats = self._store.get_stats()
        return {
            **store_stats,
            "initialized": True,
            "turn_count": self._turn_count,
            "next_consolidation_in": self._consolidation_interval - (self._turn_count % self._consolidation_interval),
        }

    # ─── Learning Engine (boucle d'apprentissage fermée) ───

    @property
    def learning(self) -> Optional[LearningEngine]:
        """Accès au moteur d'apprentissage."""
        return self._learning

    def record_execution_result(
        self,
        request: str,
        worker_type: str,
        success: bool,
        execution_time: float = 0.0,
        errors: list[str] | None = None,
        output: str = "",
    ) -> None:
        """
        Enregistre le résultat d'une exécution pour apprentissage.
        Interface principale appelée par Brain après chaque Worker.
        """
        if not self._initialized or not self._learning:
            return

        self._learning.record_result(
            request=request,
            worker_type=worker_type,
            success=success,
            execution_time=execution_time,
            errors=errors,
            output=output,
        )

    def get_learning_advice(self, request: str, proposed_worker_type: str) -> LearningAdvice:
        """
        Obtient les conseils du LearningEngine AVANT de créer un Worker.
        Appelé par Brain.make_decision() pour ajuster la stratégie.
        """
        if not self._initialized or not self._learning:
            return LearningAdvice()

        return self._learning.get_advice(request, proposed_worker_type)

    def get_learning_stats(self) -> dict:
        """Retourne les statistiques d'apprentissage."""
        if not self._initialized or not self._learning:
            return {"skills": 0, "error_patterns": 0, "performance": {}}

        return {
            "skills": len(self._learning.get_learned_skills()),
            "error_patterns": len(self._learning.get_error_patterns()),
            "performance": self._learning.get_performance_summary(),
        }

    def clear(self) -> None:
        """Vide toute la mémoire. À utiliser avec précaution."""
        if self._initialized and self._store:
            # Récupère tous les records et les supprime un par un
            records = self._store.get_recent(limit=10000)
            for record in records:
                self._store.delete(record.id)

    def close(self) -> None:
        """Ferme proprement les connexions."""
        if self._store:
            self._store.close()

    def get_model_info(self) -> dict:
        """Retourne les infos du modèle utilisé par Memory."""
        return {
            "agent": "Memory",
            "model": self._model_config.model if self._model_config else "none",
            "role": "Consolidation et synthèse intelligente",
            "has_llm": self._llm is not None or (
                not self._mock_mode and is_oauth_token(self.config.llm.api_key or "")
            ),
        }

    # ─── Skills consultables ─────────────────────────────

    def get_skills_report(self) -> dict:
        """
        Retourne un rapport complet des compétences acquises.

        Inclut :
        - Liste des skills avec stats (success_count, avg_time, best_approach)
        - Patterns d'erreur identifiés
        - Performance par type de worker
        """
        if not self._initialized or not self._learning:
            return {
                "skills": [],
                "error_patterns": [],
                "performance": {},
                "total_skills": 0,
            }

        skills = self._learning.get_learned_skills()
        error_patterns = self._learning.get_error_patterns()
        performance = self._learning.get_performance_summary()

        return {
            "skills": [s.to_dict() for s in skills],
            "error_patterns": [e.to_dict() for e in error_patterns],
            "performance": performance,
            "total_skills": len(skills),
            "total_error_patterns": len(error_patterns),
        }

    # ─── Task Registry (Tâches et Epics) ─────────────────

    @property
    def task_registry(self) -> Optional[TaskRegistry]:
        """Accès au registre de tâches."""
        return self._task_registry

    def create_task(self, description: str, worker_type: str,
                    epic_id: str | None = None) -> Task | None:
        """Crée une nouvelle tâche dans le registre."""
        if not self._initialized or not self._task_registry:
            return None
        return self._task_registry.create_task(description, worker_type, epic_id)

    def create_epic(self, description: str,
                    subtask_descriptions: list[tuple[str, str]],
                    strategy: str = "") -> Epic | None:
        """Crée un Epic avec ses sous-tâches."""
        if not self._initialized or not self._task_registry:
            return None
        return self._task_registry.create_epic(description, subtask_descriptions, strategy)

    def update_task_status(self, task_id: str, status: str,
                           result: str = "") -> Task | None:
        """Met à jour le statut d'une tâche."""
        if not self._initialized or not self._task_registry:
            return None
        return self._task_registry.update_task_status(task_id, status, result)

    def update_epic_status(self, epic_id: str, status: str) -> Epic | None:
        """Met à jour le statut d'un Epic."""
        if not self._initialized or not self._task_registry:
            return None
        return self._task_registry.update_epic_status(epic_id, status)

    def add_task_context(self, task_id: str, note: str) -> Task | None:
        """Ajoute une note de contexte à une tâche active."""
        if not self._initialized or not self._task_registry:
            return None
        return self._task_registry.add_task_context(task_id, note)

    def add_epic_context(self, epic_id: str, note: str) -> Epic | None:
        """Ajoute une note de contexte à un epic actif."""
        if not self._initialized or not self._task_registry:
            return None
        return self._task_registry.add_epic_context(epic_id, note)

    def get_tasks_report(self) -> dict:
        """Retourne un rapport du registre de tâches."""
        if not self._initialized or not self._task_registry:
            return {"tasks": [], "epics": [], "summary": {}}

        tasks = self._task_registry.get_all_tasks(limit=30)
        epics = self._task_registry.get_all_epics(limit=10)
        summary = self._task_registry.get_summary()

        return {
            "tasks": [str(t) for t in tasks],
            "epics": [str(e) for e in epics],
            "summary": summary,
        }

    # ─── Persona Engine (Stage 9 — Empathie) ─────────────

    @property
    def persona_engine(self) -> Optional[PersonaEngine]:
        """Accès au moteur de personnalité."""
        return self._persona_engine

    def get_neo_persona(self) -> Optional[dict]:
        """Retourne la personnalité actuelle de Neo."""
        if not self._persona_engine or not self._persona_engine.is_initialized:
            return None
        return self._persona_engine.persona.to_dict()

    def get_user_profile(self) -> Optional[dict]:
        """Retourne le profil utilisateur appris."""
        if not self._persona_engine or not self._persona_engine.is_initialized:
            return None
        return self._persona_engine.user_profile.to_dict()

    def record_user_observation(self, observation_type: str, content: str,
                                polarity: str = "neutral") -> None:
        """Enregistre une observation sur l'utilisateur."""
        if not self._persona_engine or not self._persona_engine.is_initialized:
            return
        self._persona_engine.record_user_observation(observation_type, content, polarity)

    def analyze_conversation(self, user_message: str, neo_response: str) -> dict:
        """Analyse automatiquement une conversation pour apprentissage."""
        if not self._persona_engine or not self._persona_engine.is_initialized:
            return {}
        return self._persona_engine.analyze_conversation(user_message, neo_response)

    def should_self_reflect(self) -> bool:
        """Indique si Neo devrait effectuer une auto-réflexion."""
        if not self._persona_engine or not self._persona_engine.is_initialized:
            return False
        return self._persona_engine.should_reflect()

    async def perform_self_reflection(self, llm_call=None) -> dict:
        """Lance une auto-réflexion de la personnalité."""
        if not self._persona_engine or not self._persona_engine.is_initialized:
            return {"success": False, "reason": "PersonaEngine non disponible"}
        call = llm_call or self._memory_llm_call
        return await self._persona_engine.perform_self_reflection(call)
