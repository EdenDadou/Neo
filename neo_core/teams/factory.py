"""
Neo Core — Factory : Fabrique de Workers
==========================================
Analyse les requêtes, détermine le type de Worker optimal,
et crée des agents spécialisés à la demande.

La Factory est stateless — elle ne conserve aucun état entre les créations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from neo_core.config import NeoConfig
from neo_core.teams.worker import Worker, WorkerType
from neo_core.tools.base_tools import ToolRegistry

if TYPE_CHECKING:
    from neo_core.core.memory_agent import MemoryAgent


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

    def classify_task(self, request: str) -> WorkerType:
        """
        Classifie une requête en type de Worker.

        Utilise des heuristiques par mots-clés (fonctionne en mock ET réel).
        En mode réel, pourrait être enrichi par le LLM (via Brain).

        Returns:
            WorkerType correspondant à la requête
        """
        request_lower = request.lower()

        # Score par type
        scores: dict[WorkerType, int] = {}

        for worker_type, patterns in _CLASSIFICATION_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, request_lower):
                    score += 1
            if score > 0:
                scores[worker_type] = score

        if scores:
            # Retourner le type avec le score le plus élevé
            best_type = max(scores, key=scores.get)
            return best_type

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

        Heuristique simple — la version LLM est dans Brain.
        """
        subtasks = []

        if worker_type == WorkerType.RESEARCHER:
            subtasks = [
                f"Rechercher des informations sur: {request[:100]}",
                "Synthétiser les résultats trouvés",
                "Vérifier la fiabilité des sources",
            ]
        elif worker_type == WorkerType.CODER:
            subtasks = [
                f"Comprendre les spécifications: {request[:100]}",
                "Écrire le code",
                "Tester le code",
            ]
        elif worker_type == WorkerType.SUMMARIZER:
            subtasks = [
                f"Lire le contenu à résumer",
                "Identifier les points clés",
                "Rédiger le résumé",
            ]
        elif worker_type == WorkerType.ANALYST:
            subtasks = [
                f"Collecter les données pertinentes",
                "Analyser les tendances et patterns",
                "Formuler les conclusions",
            ]
        elif worker_type == WorkerType.WRITER:
            subtasks = [
                f"Planifier la structure du contenu",
                "Rédiger le premier jet",
                "Relire et améliorer",
            ]
        elif worker_type == WorkerType.TRANSLATOR:
            subtasks = [
                "Identifier les langues source et cible",
                "Traduire le contenu",
                "Vérifier la qualité de la traduction",
            ]
        else:
            subtasks = [f"Traiter: {request[:100]}"]

        return subtasks

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
            except Exception:
                pass

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
