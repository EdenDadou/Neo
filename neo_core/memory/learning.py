"""
Learning Engine — Apprentissage et Adaptation
===============================================
Ferme la boucle d'apprentissage de Memory.

Responsabilités :
- Tracker les patterns d'erreur par worker type et domaine
- Stocker les compétences acquises (stratégies qui marchent)
- Générer des règles d'évitement (ne pas reproduire les erreurs)
- Fournir des recommandations à Brain pour ses décisions

Le LearningEngine travaille avec le MemoryStore existant
et ajoute une couche d'intelligence au-dessus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from neo_core.memory.store import MemoryStore, MemoryRecord


# ─── Structures de données ──────────────────────────────

@dataclass
class ErrorPattern:
    """Un pattern d'erreur récurrent identifié."""
    worker_type: str
    error_type: str  # ex: "timeout", "hallucination", "tool_failure"
    count: int = 1
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    examples: list[str] = field(default_factory=list)  # Requêtes qui ont échoué
    avoidance_rule: str = ""  # Règle pour éviter cette erreur

    def to_dict(self) -> dict:
        return {
            "worker_type": self.worker_type,
            "error_type": self.error_type,
            "count": self.count,
            "last_seen": self.last_seen,
            "examples": self.examples[:5],
            "avoidance_rule": self.avoidance_rule,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ErrorPattern:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LearnedSkill:
    """Une compétence acquise par le système."""
    name: str
    description: str
    worker_type: str
    success_count: int = 1
    avg_execution_time: float = 0.0
    best_approach: str = ""  # La meilleure stratégie trouvée
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "worker_type": self.worker_type,
            "success_count": self.success_count,
            "avg_execution_time": self.avg_execution_time,
            "best_approach": self.best_approach,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LearnedSkill:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class WorkerPerformance:
    """Métriques de performance d'un type de Worker."""
    worker_type: str
    total_tasks: int = 0
    successes: int = 0
    failures: int = 0
    total_time: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 1.0
        return self.successes / self.total_tasks

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.success_rate

    @property
    def avg_time(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.total_time / self.total_tasks


@dataclass
class LearningAdvice:
    """Conseil du LearningEngine pour une requête donnée."""
    recommended_worker: Optional[str] = None
    avoid_workers: list[str] = field(default_factory=list)
    relevant_errors: list[ErrorPattern] = field(default_factory=list)
    relevant_skills: list[LearnedSkill] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence_adjustment: float = 0.0  # Ajustement de confiance (-0.3 à +0.3)

    def to_context_string(self) -> str:
        """Formate les conseils pour injection dans un prompt."""
        parts = []

        if self.warnings:
            parts.append("⚠ Avertissements mémoire :")
            for w in self.warnings:
                parts.append(f"  - {w}")

        if self.avoid_workers:
            parts.append(f"⚠ Workers à éviter : {', '.join(self.avoid_workers)}")

        if self.relevant_skills:
            parts.append("✓ Compétences acquises pertinentes :")
            for skill in self.relevant_skills[:3]:
                parts.append(f"  - {skill.name}: {skill.best_approach}")

        if self.relevant_errors:
            parts.append("✗ Erreurs passées similaires :")
            for err in self.relevant_errors[:3]:
                rule = f" → {err.avoidance_rule}" if err.avoidance_rule else ""
                parts.append(f"  - {err.worker_type}/{err.error_type} (×{err.count}){rule}")

        if not parts:
            return ""

        return "\n".join(parts)


# ─── Learning Engine ─────────────────────────────────────

class LearningEngine:
    """
    Moteur d'apprentissage du système Neo Core.

    Analyse les succès et échecs pour :
    1. Identifier les patterns d'erreur récurrents
    2. Stocker les compétences acquises
    3. Recommander des stratégies pour les nouvelles requêtes
    4. Éviter de reproduire les mêmes erreurs
    """

    # Sources de données dans le store
    SOURCE_ERROR_PATTERN = "learning:error_pattern"
    SOURCE_SKILL = "learning:skill"
    SOURCE_PERFORMANCE = "learning:performance"

    def __init__(self, store: MemoryStore):
        self.store = store
        self._performance_cache: dict[str, WorkerPerformance] = {}
        # Recharger les performances depuis le store (survit au redémarrage)
        self._reload_performance_cache()

    def _reload_performance_cache(self) -> None:
        """
        Recharge les métriques de performance depuis le store.
        Appelé à l'initialisation pour survivre aux redémarrages.
        """
        try:
            records = self.store.search_by_source(self.SOURCE_PERFORMANCE, limit=500)
            for record in records:
                try:
                    data = json.loads(record.content)
                    wtype = data.get("worker_type", "")
                    if not wtype:
                        continue

                    if wtype not in self._performance_cache:
                        self._performance_cache[wtype] = WorkerPerformance(
                            worker_type=wtype
                        )

                    perf = self._performance_cache[wtype]
                    perf.total_tasks += 1
                    perf.total_time += data.get("execution_time", 0.0)
                    if data.get("success", False):
                        perf.successes += 1
                    else:
                        perf.failures += 1
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception:
            pass  # Ne pas crasher si le store est vide

    # ─── Enregistrement des résultats ────────────────────

    def record_result(
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
        Appelé par Brain après chaque Worker.
        """
        # 1. Mettre à jour les métriques de performance
        self._update_performance(worker_type, success, execution_time)

        # 2. Si échec → identifier et stocker le pattern d'erreur
        if not success and errors:
            self._record_error_pattern(request, worker_type, errors)

        # 3. Si succès → renforcer la compétence
        if success:
            self._record_success(request, worker_type, execution_time, output)

    def _update_performance(self, worker_type: str, success: bool, time: float) -> None:
        """Met à jour les métriques de performance d'un worker type."""
        if worker_type not in self._performance_cache:
            self._performance_cache[worker_type] = WorkerPerformance(
                worker_type=worker_type
            )

        perf = self._performance_cache[worker_type]
        perf.total_tasks += 1
        perf.total_time += time
        if success:
            perf.successes += 1
        else:
            perf.failures += 1

        # Persister dans le store
        self.store.store(
            content=json.dumps({
                "worker_type": worker_type,
                "success": success,
                "execution_time": time,
                "success_rate": perf.success_rate,
                "total_tasks": perf.total_tasks,
            }),
            source=self.SOURCE_PERFORMANCE,
            tags=[f"worker:{worker_type}", "success" if success else "failure"],
            importance=0.4,
            metadata={"worker_type": worker_type, "success": success},
        )

    def _record_error_pattern(self, request: str, worker_type: str, errors: list[str]) -> None:
        """Identifie et stocke un pattern d'erreur (sans duplication)."""
        error_type = self._classify_error(errors)

        # Chercher si ce pattern existe déjà
        existing, existing_record_id = self._find_error_pattern_with_id(worker_type, error_type)

        if existing and existing_record_id:
            # Incrémenter le compteur
            existing.count += 1
            existing.last_seen = datetime.now().isoformat()
            if request[:100] not in existing.examples:
                existing.examples.append(request[:100])
                existing.examples = existing.examples[-5:]

            # Générer une règle d'évitement si pattern récurrent (≥ 3)
            if existing.count >= 3 and not existing.avoidance_rule:
                existing.avoidance_rule = self._generate_avoidance_rule(existing)

            # Remplacer l'ancien record (évite la duplication)
            try:
                self.store.delete(existing_record_id)
            except Exception:
                pass

            self.store.store(
                content=json.dumps(existing.to_dict()),
                source=self.SOURCE_ERROR_PATTERN,
                tags=[
                    f"worker:{worker_type}",
                    f"error:{error_type}",
                    "recurring" if existing.count >= 3 else "sporadic",
                ],
                importance=min(0.9, 0.6 + existing.count * 0.05),
                metadata=existing.to_dict(),
            )
        else:
            # Nouveau pattern
            pattern = ErrorPattern(
                worker_type=worker_type,
                error_type=error_type,
                examples=[request[:100]],
            )
            self.store.store(
                content=json.dumps(pattern.to_dict()),
                source=self.SOURCE_ERROR_PATTERN,
                tags=[f"worker:{worker_type}", f"error:{error_type}", "new"],
                importance=0.6,
                metadata=pattern.to_dict(),
            )

    def _record_success(self, request: str, worker_type: str,
                        execution_time: float, output: str) -> None:
        """Enregistre un succès comme compétence acquise."""
        skill_name = self._extract_skill_name(request, worker_type)

        # Chercher si cette compétence existe déjà
        existing_skill, existing_record_id = self._find_skill_with_id(skill_name, worker_type)

        if existing_skill and existing_record_id:
            # Mettre à jour la compétence existante
            existing_skill.success_count += 1
            existing_skill.avg_execution_time = (
                (existing_skill.avg_execution_time * (existing_skill.success_count - 1)
                 + execution_time) / existing_skill.success_count
            )
            # Persister la mise à jour : supprimer l'ancien, créer le nouveau
            try:
                self.store.delete(existing_record_id)
            except Exception:
                pass
            self.store.store(
                content=json.dumps(existing_skill.to_dict()),
                source=self.SOURCE_SKILL,
                tags=[f"worker:{worker_type}", f"skill:{skill_name}"],
                importance=min(0.8, 0.5 + existing_skill.success_count * 0.05),
                metadata=existing_skill.to_dict(),
            )
        else:
            # Nouvelle compétence
            skill = LearnedSkill(
                name=skill_name,
                description=request[:200],
                worker_type=worker_type,
                avg_execution_time=execution_time,
                best_approach=f"Worker {worker_type} exécuté avec succès en {execution_time:.1f}s",
            )
            self.store.store(
                content=json.dumps(skill.to_dict()),
                source=self.SOURCE_SKILL,
                tags=[f"worker:{worker_type}", f"skill:{skill_name}"],
                importance=0.5,
                metadata=skill.to_dict(),
            )

    # ─── Consultation (pour Brain) ───────────────────────

    def get_advice(self, request: str, proposed_worker_type: str) -> LearningAdvice:
        """
        Consulte l'historique pour donner des conseils à Brain.
        Appelé AVANT de créer un Worker.
        """
        advice = LearningAdvice()

        # 1. Vérifier les erreurs passées pour ce type de worker
        error_patterns = self._get_error_patterns_for_type(proposed_worker_type)
        if error_patterns:
            advice.relevant_errors = error_patterns

            # Si taux d'erreur élevé → avertissement
            perf = self._performance_cache.get(proposed_worker_type)
            if perf and perf.failure_rate > 0.5 and perf.total_tasks >= 3:
                advice.warnings.append(
                    f"Worker '{proposed_worker_type}' a un taux d'échec de "
                    f"{perf.failure_rate:.0%} ({perf.failures}/{perf.total_tasks})"
                )
                advice.confidence_adjustment = -0.2

            # Patterns récurrents → réduire la confiance
            recurring = [p for p in error_patterns if p.count >= 3]
            if recurring:
                for p in recurring:
                    if p.avoidance_rule:
                        advice.warnings.append(p.avoidance_rule)
                advice.confidence_adjustment -= 0.1

        # 2. Chercher les compétences pertinentes
        skills = self._find_relevant_skills(request)
        if skills:
            advice.relevant_skills = skills
            # Si on a une compétence qui a bien marché avec un autre worker
            for skill in skills:
                if skill.worker_type != proposed_worker_type and skill.success_count >= 2:
                    advice.recommended_worker = skill.worker_type
                    advice.confidence_adjustment += 0.1

        # 3. Vérifier si des workers sont à éviter pour ce type de requête
        for wtype, perf in self._performance_cache.items():
            if perf.failure_rate > 0.7 and perf.total_tasks >= 3:
                advice.avoid_workers.append(wtype)

        return advice

    def get_performance_summary(self) -> dict:
        """Retourne un résumé des performances de tous les workers."""
        summary = {}
        for wtype, perf in self._performance_cache.items():
            summary[wtype] = {
                "total_tasks": perf.total_tasks,
                "success_rate": f"{perf.success_rate:.0%}",
                "avg_time": f"{perf.avg_time:.1f}s",
                "failures": perf.failures,
            }
        return summary

    def get_learned_skills(self) -> list[LearnedSkill]:
        """Retourne toutes les compétences acquises."""
        records = self.store.search_by_source(self.SOURCE_SKILL, limit=50)
        skills = []
        for record in records:
            try:
                data = json.loads(record.content)
                skills.append(LearnedSkill.from_dict(data))
            except (json.JSONDecodeError, TypeError):
                pass
        return skills

    def get_error_patterns(self) -> list[ErrorPattern]:
        """Retourne tous les patterns d'erreur identifiés."""
        records = self.store.search_by_source(self.SOURCE_ERROR_PATTERN, limit=50)
        patterns = []
        for record in records:
            try:
                data = json.loads(record.content)
                patterns.append(ErrorPattern.from_dict(data))
            except (json.JSONDecodeError, TypeError):
                pass
        return patterns

    # ─── Méthodes internes ───────────────────────────────

    def _classify_error(self, errors: list[str]) -> str:
        """Classifie une erreur en catégorie."""
        error_text = " ".join(errors).lower()

        if "timeout" in error_text or "timed out" in error_text:
            return "timeout"
        if "401" in error_text or "auth" in error_text or "unauthorized" in error_text:
            return "auth_failure"
        if "429" in error_text or "rate limit" in error_text:
            return "rate_limit"
        if "500" in error_text or "502" in error_text or "503" in error_text:
            return "server_error"
        if "tool" in error_text and ("fail" in error_text or "error" in error_text):
            return "tool_failure"
        if "hallucin" in error_text or "invent" in error_text:
            return "hallucination"
        if "format" in error_text or "json" in error_text or "parse" in error_text:
            return "format_error"
        if "key" in error_text and "error" in error_text:
            return "key_error"

        return "unknown"

    def _generate_avoidance_rule(self, pattern: ErrorPattern) -> str:
        """Génère une règle d'évitement pour un pattern récurrent."""
        rules = {
            "timeout": f"Le worker '{pattern.worker_type}' a tendance à timeout. "
                       f"Réduire la complexité de la requête ou augmenter le timeout.",
            "auth_failure": f"Problème d'authentification récurrent avec '{pattern.worker_type}'. "
                           f"Vérifier les tokens.",
            "rate_limit": f"Rate limiting fréquent pour '{pattern.worker_type}'. "
                         f"Espacer les requêtes.",
            "tool_failure": f"Les outils du worker '{pattern.worker_type}' échouent souvent. "
                           f"Considérer un worker alternatif.",
            "hallucination": f"Le worker '{pattern.worker_type}' invente des données. "
                            f"Renforcer les règles anti-hallucination.",
            "server_error": f"Erreurs serveur récurrentes pour '{pattern.worker_type}'. "
                           f"Réessayer plus tard ou utiliser un fallback.",
            "format_error": f"Erreurs de format récurrentes avec '{pattern.worker_type}'. "
                           f"Simplifier les instructions de formatage.",
        }
        return rules.get(pattern.error_type,
                        f"Erreur récurrente ({pattern.error_type}) avec '{pattern.worker_type}' "
                        f"(×{pattern.count}). Considérer une approche alternative.")

    def _find_error_pattern(self, worker_type: str, error_type: str) -> Optional[ErrorPattern]:
        """Cherche un pattern d'erreur existant."""
        result = self._find_error_pattern_with_id(worker_type, error_type)
        return result[0]

    def _find_error_pattern_with_id(
        self, worker_type: str, error_type: str
    ) -> tuple[Optional[ErrorPattern], Optional[str]]:
        """Cherche un pattern d'erreur existant et retourne aussi son record ID."""
        records = self.store.search_by_tags(
            [f"worker:{worker_type}", f"error:{error_type}"],
            limit=5,
        )
        for record in records:
            if record.source == self.SOURCE_ERROR_PATTERN:
                try:
                    data = json.loads(record.content)
                    return ErrorPattern.from_dict(data), record.id
                except (json.JSONDecodeError, TypeError):
                    pass
        return None, None

    def _find_skill(self, skill_name: str, worker_type: str) -> Optional[LearnedSkill]:
        """Cherche une compétence existante."""
        result = self._find_skill_with_id(skill_name, worker_type)
        return result[0]

    def _find_skill_with_id(
        self, skill_name: str, worker_type: str
    ) -> tuple[Optional[LearnedSkill], Optional[str]]:
        """Cherche une compétence existante et retourne aussi son record ID."""
        records = self.store.search_by_tags(
            [f"worker:{worker_type}", f"skill:{skill_name}"],
            limit=5,
        )
        for record in records:
            if record.source == self.SOURCE_SKILL:
                try:
                    data = json.loads(record.content)
                    return LearnedSkill.from_dict(data), record.id
                except (json.JSONDecodeError, TypeError):
                    pass
        return None, None

    def _find_relevant_skills(self, request: str) -> list[LearnedSkill]:
        """Cherche les compétences pertinentes par recherche sémantique."""
        if not self.store.has_vector_search:
            return []

        records = self.store.search_semantic(request, n_results=5)
        skills = []
        for record in records:
            if record.source == self.SOURCE_SKILL:
                try:
                    data = json.loads(record.content)
                    skills.append(LearnedSkill.from_dict(data))
                except (json.JSONDecodeError, TypeError):
                    pass
        return skills

    def _get_error_patterns_for_type(self, worker_type: str) -> list[ErrorPattern]:
        """Récupère tous les patterns d'erreur pour un type de worker."""
        records = self.store.search_by_tags(
            [f"worker:{worker_type}"],
            limit=20,
        )
        patterns = []
        for record in records:
            if record.source == self.SOURCE_ERROR_PATTERN:
                try:
                    data = json.loads(record.content)
                    patterns.append(ErrorPattern.from_dict(data))
                except (json.JSONDecodeError, TypeError):
                    pass
        return patterns

    def _extract_skill_name(self, request: str, worker_type: str) -> str:
        """Extrait un nom de compétence depuis la requête."""
        # Simplification : prend les 5 premiers mots significatifs
        words = request.lower().split()[:5]
        # Retire les mots vides
        stop_words = {"le", "la", "les", "de", "du", "des", "un", "une", "et",
                      "est", "en", "pour", "dans", "sur", "avec", "que", "qui"}
        significant = [w for w in words if w not in stop_words]
        name = "_".join(significant[:3]) if significant else worker_type
        return f"{worker_type}:{name}"
