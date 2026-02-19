"""
Neo Core — Factory : Fabrique de Workers
==========================================
Analyse les requêtes, détermine le type de Worker optimal,
et crée des agents spécialisés à la demande.

La Factory est stateless — elle ne conserve aucun état entre les créations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

from neo_core.config import NeoConfig
from neo_core.brain.teams.worker import Worker, WorkerType
from neo_core.brain.tools.base_tools import ToolRegistry

if TYPE_CHECKING:
    from neo_core.memory.agent import MemoryAgent


@dataclass
class TaskAnalysis:
    """
    Résultat de l'analyse d'une tâche par Brain/Factory.

    Produit par Brain._decompose_task() ou Factory.analyze_task().
    Consommé par Factory.create_worker().
    """
    worker_type: WorkerType
    primary_task: str
    subtasks: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.8


# ─── Patterns de classification par mots-clés ─────────────

_CLASSIFICATION_PATTERNS: dict[WorkerType, list[str]] = {
    WorkerType.RESEARCHER: [
        # Verbes de recherche explicites
        r"recherch", r"cherch", r"trouv", r"investig", r"explor",
        r"search", r"find", r"look\s*up", r"discover",
        r"renseign", r"inform", r"document",
        # Sport / résultats / classements
        r"\bmatchs?\b", r"\bscores?\b", r"\brésultats?\b", r"\bclassement",
        r"\batp\b", r"\bfifa\b", r"\bligue\b", r"\bchampion", r"\btournoi",
        r"\bfoot\b", r"\btennis\b", r"\bbasket\b", r"\bf1\b", r"\bmoto\s*gp",
        r"\béquipe", r"\bjoueur", r"\bvainqueur", r"\bgagn",
        # Météo
        r"mét[ée]o", r"quel\s*temps", r"temp[ée]rature", r"pr[ée]vision",
        # Actualité / news
        r"\bactualit", r"\bnews\b", r"\bnouvelles?\b", r"\binfos?\b",
        # Finance / prix
        r"\bcours\b", r"\bbourse\b", r"\bcrypto", r"\bbitcoin",
        r"\bprix\b",
        # Marqueurs temporels (= besoin de données fraîches)
        r"aujourd[''h]?hui", r"\bce\s*(soir|matin)", r"\bdemain\b",
        r"en\s*ce\s*moment", r"actuellement", r"récemment",
        r"de\s*la\s*journée", r"\bdu\s*jour\b", r"\bcette\s*semaine",
        # Questions implicites sur l'état du monde
        r"\bqui\s*(a\s*gagn|est\s*premier|mène|joue)",
        r"\boù\s*(en\s*est|se\s*passe|joue)",
        r"\bquand\b.{0,20}\b(est|commence|finit|a\s*lieu)",
        r"\bc['']est\s*quoi", r"\bqu['']est[- ]ce\s*que",
    ],
    WorkerType.CODER: [
        r"code", r"script", r"program", r"développ", r"debug",
        r"python", r"javascript", r"function", r"class\b",
        r"algorith", r"implémen", r"compil", r"exécut",
        r"bug", r"fix", r"corrig", r"refactor",
    ],
    WorkerType.SUMMARIZER: [
        r"résum", r"synth[eè]s", r"condens", r"abrég",
        r"summariz", r"summary", r"digest", r"shorten",
        r"point[s]?\s*cl[ée]", r"essentiel",
    ],
    WorkerType.ANALYST: [
        r"analy[sz]", r"examin", r"éval[u]", r"compar",
        r"statistiq", r"donn[ée]", r"tendanc", r"pattern",
        r"assess", r"evaluat", r"metric", r"data",
    ],
    WorkerType.WRITER: [
        r"écri[st]", r"rédig", r"compos", r"rédact",
        r"write", r"draft", r"crée?\s*(un|une|le|la)?\s*(texte|article|rapport|email|lettre)",
        r"article", r"blog", r"contenu", r"rapport",
    ],
    WorkerType.TRANSLATOR: [
        r"tradui[st]", r"translat", r"traduc",
        r"en\s+(anglais|français|espagnol|allemand|italien|chinois|japonais|arabe)",
        r"from\s+\w+\s+to\s+\w+",
    ],
}


@dataclass
class WorkerFactory:
    """
    Fabrique de Workers — crée des agents spécialisés à la demande.

    Responsabilités :
    1. Classifier les requêtes par type de Worker
    2. Sélectionner les outils appropriés
    3. Créer le Worker avec config adaptée
    """
    config: NeoConfig
    memory: Optional[MemoryAgent] = None

    # Seuil minimum de confiance pour la classification par heuristiques.
    # En dessous de ce score, on tente un fallback LLM (si disponible).
    CLASSIFICATION_CONFIDENCE_THRESHOLD = 2

    def classify_task(self, request: str) -> WorkerType:
        """
        Classifie une requête en type de Worker.

        Pipeline :
        1. Heuristiques par mots-clés (instantané, mock-compatible)
        2. Si le score est trop faible (< seuil) et qu'un LLM est disponible,
           tente une classification LLM en fallback
        3. Sinon → GENERIC

        Returns:
            WorkerType correspondant à la requête
        """
        request_lower = request.lower()

        # Score par type via heuristiques
        scores: dict[WorkerType, int] = {}

        for worker_type, patterns in _CLASSIFICATION_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, request_lower):
                    score += 1
            if score > 0:
                scores[worker_type] = score

        if scores:
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]

            # Confiance suffisante → retourner directement
            if best_score >= self.CLASSIFICATION_CONFIDENCE_THRESHOLD:
                return best_type

            # Score faible → stocker comme candidat, on tentera le LLM
            self._last_heuristic_candidate = (best_type, best_score)
            return best_type

        # Aucun match heuristique → GENERIC
        self._last_heuristic_candidate = None
        return WorkerType.GENERIC

    async def classify_task_with_llm(self, request: str, llm_call) -> WorkerType:
        """
        Classification LLM en fallback quand les heuristiques sont insuffisantes.

        Appelé par Brain quand classify_task retourne GENERIC ou un score faible.
        Le LLM choisit parmi les types de workers disponibles.
        """
        worker_types_str = ", ".join(wt.value for wt in WorkerType if wt != WorkerType.GENERIC)

        prompt = (
            f"Classifie cette requête utilisateur dans exactement UN type de worker.\n"
            f"Types disponibles : {worker_types_str}\n"
            f"Requête : {request[:300]}\n\n"
            f"Réponds avec UNIQUEMENT le type (un seul mot, ex: researcher). "
            f"Si aucun type ne correspond, réponds 'generic'."
        )

        try:
            result = await llm_call(prompt)
            result_clean = result.strip().lower().replace("'", "").replace('"', '')
            return WorkerType(result_clean)
        except (ValueError, TypeError):
            logger.debug("LLM classification returned invalid type: %s", result if 'result' in dir() else "error")
        except Exception as e:
            logger.debug("LLM classification failed: %s", e)

        return WorkerType.GENERIC

    def analyze_task(self, request: str) -> TaskAnalysis:
        """
        Analyse complète d'une requête : type, sous-tâches, outils.

        Version heuristique (mock-compatible).
        Pour la version LLM, voir Brain._decompose_task().
        """
        worker_type = self.classify_task(request)
        tool_names = [
            getattr(t, "name", str(t))
            for t in ToolRegistry.get_tools_for_type(worker_type.value)
        ]

        # Décomposition simple en sous-tâches
        subtasks = self._basic_decompose(request, worker_type)

        return TaskAnalysis(
            worker_type=worker_type,
            primary_task=request,
            subtasks=subtasks,
            required_tools=tool_names,
            reasoning=f"Classification par mots-clés → {worker_type.value}",
            confidence=0.7,
        )

    def _basic_decompose(self, request: str, worker_type: WorkerType) -> list[str]:
        """
        Décomposition basique d'une tâche en sous-tâches.

        Retourne UNE SEULE sous-tâche descriptive.
        Les décompositions fines sont gérées par le Worker lui-même
        ou par la décomposition LLM (Brain._decompose_crew_with_llm).

        Principe : une Task dans le registre = une action concrète,
        pas 3 étapes génériques qui polluent l'affichage.
        """
        # Une seule sous-tâche descriptive par type
        desc = request[:150]
        if worker_type == WorkerType.RESEARCHER:
            return [f"Rechercher: {desc}"]
        elif worker_type == WorkerType.CODER:
            return [f"Coder: {desc}"]
        elif worker_type == WorkerType.SUMMARIZER:
            return [f"Résumer: {desc}"]
        elif worker_type == WorkerType.ANALYST:
            return [f"Analyser: {desc}"]
        elif worker_type == WorkerType.WRITER:
            return [f"Rédiger: {desc}"]
        elif worker_type == WorkerType.TRANSLATOR:
            return [f"Traduire: {desc}"]
        else:
            return [f"Traiter: {desc}"]

    def create_worker(self, analysis: TaskAnalysis) -> Worker:
        """
        Crée un Worker spécialisé à partir d'une TaskAnalysis.

        Le Worker reçoit :
        - La config globale (pour l'auth LLM)
        - Les outils appropriés pour son type
        - Le prompt système spécialisé
        - La connexion à Memory
        - Le contexte d'apprentissage (erreurs passées, compétences)
        """
        # Récupérer les outils
        tools = ToolRegistry.get_tools_for_type(analysis.worker_type.value)

        # Initialiser les outils avec la mémoire
        ToolRegistry.initialize(
            mock_mode=self.config.is_mock_mode(),
            memory=self.memory,
        )

        # Enrichir les subtasks avec les conseils d'apprentissage
        enriched_subtasks = list(analysis.subtasks)
        if self.memory and self.memory.is_initialized:
            try:
                advice = self.memory.get_learning_advice(
                    analysis.primary_task, analysis.worker_type.value
                )
                learning_context = advice.to_context_string()
                if learning_context:
                    enriched_subtasks.insert(0,
                        f"[Mémoire] {learning_context}"
                    )
            except Exception as e:
                logger.debug("Learning context enrichment failed: %s", e)

        return Worker(
            config=self.config,
            worker_type=analysis.worker_type,
            task=analysis.primary_task,
            subtasks=enriched_subtasks,
            tools=tools,
            memory=self.memory,
        )

    def create_worker_for_type(self, worker_type: WorkerType, task: str,
                                subtasks: list[str] | None = None) -> Worker:
        """
        Raccourci : crée directement un Worker d'un type donné.

        Utile quand Brain a déjà déterminé le type via LLM.
        """
        analysis = TaskAnalysis(
            worker_type=worker_type,
            primary_task=task,
            subtasks=subtasks or [],
            required_tools=[
                getattr(t, "name", str(t))
                for t in ToolRegistry.get_tools_for_type(worker_type.value)
            ],
            reasoning="Création directe par Brain",
            confidence=0.9,
        )
        return self.create_worker(analysis)
