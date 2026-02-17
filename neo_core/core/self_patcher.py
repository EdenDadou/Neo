"""
Neo Core — Self-Patcher (Level 3)
====================================
Détecte les patterns d'erreurs récurrents et génère des "patches comportementaux"
qui modifient les décisions du Brain à runtime.

Les patches ne modifient JAMAIS les fichiers source de neo_core/.
Ils opèrent au niveau config/décision et sont stockés en JSON dans data/patches/.

Lifecycle :
    1. DÉTECTION  — analyse les ErrorPattern du LearningEngine
    2. GÉNÉRATION — crée un patch JSON selon le type d'erreur
    3. VALIDATION — vérifie sur données historiques (amélioration ≥ 50%)
    4. APPLICATION — applique les overrides lors de make_decision()
    5. MONITORING — évalue l'efficacité avant/après
    6. ROLLBACK   — désactive si la confiance tombe sous le seuil
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─── Erreurs corrigeables par un patch ─────────────────────────
PATCHABLE_ERROR_TYPES = frozenset({
    "hallucination",
    "tool_failure",
    "timeout",
    "routing_error",
    "strategy_mismatch",
})

# ─── Workers disponibles pour fallback ─────────────────────────
_WORKER_FALLBACKS: dict[str, str] = {
    "coder": "analyst",
    "analyst": "coder",
    "researcher": "generic",
    "writer": "generic",
    "summarizer": "generic",
    "translator": "generic",
    "generic": "researcher",
}

# ─── Seuils par défaut ─────────────────────────────────────────
DEFAULT_DETECTION_THRESHOLD = 3     # Min occurrences pour déclencher
DEFAULT_VALIDATION_THRESHOLD = 0.5  # Amélioration requise (50 %)
DEFAULT_ROLLBACK_THRESHOLD = -0.1   # Dégradation tolérée avant rollback
RECENT_WINDOW_DAYS = 7              # Fenêtre de recency


# ════════════════════════════════════════════════════════════════
#  Data structures
# ════════════════════════════════════════════════════════════════

@dataclass
class PatchMetadata:
    """Métadonnées d'un patch comportemental."""

    patch_id: str
    name: str
    description: str
    created_at: str

    # Patch definition
    patch_type: str       # "prompt_override" | "routing_rule" | "config_override"
    target: str           # worker_type visé (ex. "coder")
    condition: dict       # Quand appliquer : {"error_type": "...", ...}
    action: dict          # Quoi faire       : {"override_temperature": 0.3, ...}

    # État
    enabled: bool = False
    version: int = 1

    # Métriques A/B
    metrics_before: dict = field(default_factory=dict)
    metrics_after: dict = field(default_factory=dict)
    confidence: float = 0.0
    applied_count: int = 0

    # Rollback
    auto_rollback_threshold: float = DEFAULT_ROLLBACK_THRESHOLD
    rollback_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PatchMetadata:
        # Accepter les clés manquantes avec des valeurs par défaut
        expected = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in expected}
        return cls(**filtered)


@dataclass
class PatchEvaluation:
    """Résultat de l'évaluation d'un patch."""

    patch_id: str
    is_effective: bool
    confidence: float
    before_success_rate: float
    after_success_rate: float
    improvement: float        # Points de pourcentage
    sample_size: int
    recommendation: str       # "keep" | "improve" | "rollback" | "insufficient_data"


# ════════════════════════════════════════════════════════════════
#  SelfPatcher
# ════════════════════════════════════════════════════════════════

class SelfPatcher:
    """
    Détecte les erreurs récurrentes et génère des patches comportementaux.
    Thread-safe — toutes les mutations sont protégées par un Lock.
    """

    def __init__(
        self,
        data_dir: Path,
        learning_engine: Any,
        *,
        detection_threshold: int = DEFAULT_DETECTION_THRESHOLD,
        validation_threshold: float = DEFAULT_VALIDATION_THRESHOLD,
    ):
        self._data_dir = Path(data_dir)
        self._patches_dir = self._data_dir / "patches"
        self._learning = learning_engine

        self._detection_threshold = detection_threshold
        self._validation_threshold = validation_threshold

        self._patches: dict[str, PatchMetadata] = {}
        self._lock = threading.Lock()

        self._load_patches()

    # ──────────────────────────────────────────────
    #  1. DÉTECTION
    # ──────────────────────────────────────────────

    def detect_patchable_patterns(self) -> list:
        """
        Analyse les ErrorPattern du LearningEngine et retourne ceux qui
        sont corrigeables par un patch comportemental.

        Critères :
          - count >= detection_threshold
          - error_type dans PATCHABLE_ERROR_TYPES
          - last_seen récent (< RECENT_WINDOW_DAYS jours)
          - Pas de patch actif existant pour ce pattern
        """
        try:
            all_patterns = self._learning.get_error_patterns()
        except Exception as exc:
            logger.debug("SelfPatcher: impossible de lire les error patterns: %s", exc)
            return []

        patchable = []
        for pattern in all_patterns:
            if (
                pattern.count >= self._detection_threshold
                and pattern.error_type in PATCHABLE_ERROR_TYPES
                and self._is_recent(pattern.last_seen)
                and not self._has_active_patch_for(pattern)
            ):
                patchable.append(pattern)

        return patchable

    # ──────────────────────────────────────────────
    #  2. GÉNÉRATION
    # ──────────────────────────────────────────────

    def generate_patch(self, error_pattern) -> Optional[PatchMetadata]:
        """
        Génère un patch à partir d'un ErrorPattern détecté.
        Retourne None si le pattern ne peut pas être patché ou s'il existe déjà.
        """
        patch_id = self._make_patch_id(error_pattern)

        with self._lock:
            if patch_id in self._patches:
                return None

        patch_type, action = self._plan_action(error_pattern)
        if patch_type is None:
            return None

        patch = PatchMetadata(
            patch_id=patch_id,
            name=f"auto_{error_pattern.error_type}_{error_pattern.worker_type}",
            description=(
                f"Auto-patch for {error_pattern.error_type} on "
                f"{error_pattern.worker_type} (count={error_pattern.count})"
            ),
            created_at=datetime.now().isoformat(),
            patch_type=patch_type,
            target=error_pattern.worker_type,
            condition={
                "error_type": error_pattern.error_type,
                "worker_type": error_pattern.worker_type,
                "trigger_examples": error_pattern.examples[:3],
            },
            action=action,
            enabled=False,          # Désactivé jusqu'à validation
        )

        with self._lock:
            self._patches[patch_id] = patch

        logger.info(
            "SelfPatcher: patch généré — %s (type=%s, target=%s)",
            patch_id, patch_type, error_pattern.worker_type,
        )
        return patch

    def _plan_action(self, pattern) -> tuple[Optional[str], dict]:
        """Planifie l'action corrective selon le type d'erreur."""

        etype = pattern.error_type

        if etype == "hallucination":
            return ("prompt_override", {
                "override_temperature": 0.3,
                "add_system_suffix": (
                    "Double-check your response for accuracy. "
                    "Cite sources when possible."
                ),
            })

        if etype == "tool_failure":
            fallback = _WORKER_FALLBACKS.get(pattern.worker_type, "generic")
            return ("routing_rule", {
                "fallback_worker": fallback,
                "retry_with_fallback": True,
            })

        if etype == "timeout":
            return ("config_override", {
                "override_timeout": 300.0,
                "decompose_task": True,
            })

        if etype == "routing_error":
            fallback = _WORKER_FALLBACKS.get(pattern.worker_type, "generic")
            return ("routing_rule", {
                "override_worker": fallback,
            })

        if etype == "strategy_mismatch":
            fallback = _WORKER_FALLBACKS.get(pattern.worker_type, "generic")
            return ("routing_rule", {
                "override_worker": fallback,
                "add_system_suffix": "Simplify your approach step by step.",
            })

        return (None, {})

    # ──────────────────────────────────────────────
    #  3. VALIDATION
    # ──────────────────────────────────────────────

    def validate_patch(self, patch: PatchMetadata) -> bool:
        """
        Valide un patch sur les données historiques.
        Active le patch si l'amélioration estimée ≥ validation_threshold.
        """
        # Capturer les métriques « avant »
        metrics_before = self._capture_metrics(patch.target, patch.condition.get("error_type", ""))
        patch.metrics_before = metrics_before

        # Estimer si le patch aurait aidé
        error_count = metrics_before.get("error_count", 0)
        total = metrics_before.get("total_tasks", 0)

        if total < 2:
            logger.info("SelfPatcher: pas assez de données pour valider %s", patch.patch_id)
            return False

        # Heuristique : un patch hallucination avec temp réduite améliore ~ 60 %
        # Un patch routing améliore si le fallback worker a un meilleur taux
        estimated_improvement = self._estimate_improvement(patch, metrics_before)

        if estimated_improvement >= self._validation_threshold:
            patch.enabled = True
            self._save_patches()
            logger.info(
                "SelfPatcher: patch %s validé (amélioration estimée=%.0f%%)",
                patch.patch_id, estimated_improvement * 100,
            )
            return True

        logger.info(
            "SelfPatcher: patch %s rejeté (amélioration=%.0f%% < seuil=%.0f%%)",
            patch.patch_id, estimated_improvement * 100, self._validation_threshold * 100,
        )
        return False

    def _estimate_improvement(self, patch: PatchMetadata, metrics: dict) -> float:
        """Estime l'amélioration qu'un patch apporterait."""
        error_rate = metrics.get("error_rate", 0.0)
        if error_rate == 0:
            return 0.0

        # Facteurs d'amélioration par type de patch
        improvement_factors = {
            "prompt_override": 0.6,    # Réduire temp aide ~60% des cas
            "routing_rule": 0.5,       # Changer de worker aide ~50%
            "config_override": 0.4,    # Augmenter timeout aide ~40%
        }

        factor = improvement_factors.get(patch.patch_type, 0.3)
        return error_rate * factor

    # ──────────────────────────────────────────────
    #  4. APPLICATION
    # ──────────────────────────────────────────────

    def apply_patches(self, request: str, worker_type: str) -> dict[str, Any]:
        """
        Cherche les patches actifs qui matchent la requête et le worker_type.
        Retourne un dict d'overrides à appliquer au BrainDecision.

        Overrides possibles :
          - "override_temperature": float
          - "override_worker": str
          - "override_timeout": float
          - "add_system_suffix": str
          - "decompose_task": bool
          - "retry_with_fallback": bool
          - "fallback_worker": str
        """
        overrides: dict[str, Any] = {}

        with self._lock:
            for patch in self._patches.values():
                if not patch.enabled:
                    continue

                if self._patch_matches(patch, request, worker_type):
                    # Fusionner les actions du patch dans les overrides
                    overrides.update(patch.action)
                    patch.applied_count += 1
                    logger.debug(
                        "SelfPatcher: patch %s appliqué (count=%d)",
                        patch.patch_id, patch.applied_count,
                    )

        return overrides

    def _patch_matches(self, patch: PatchMetadata, request: str, worker_type: str) -> bool:
        """Vérifie si un patch s'applique au contexte courant."""
        cond = patch.condition

        # Le worker_type doit correspondre
        if cond.get("worker_type") and cond["worker_type"] != worker_type:
            return False

        # Match sur les exemples trigger (recherche de similarité simple)
        trigger_examples = cond.get("trigger_examples", [])
        if trigger_examples:
            request_lower = request.lower()
            request_words = set(request_lower.split())
            for example in trigger_examples:
                # Vérifier si des mots-clés de l'exemple sont dans la requête
                example_words = set(example.lower().split())
                common = example_words & request_words
                # Seuil adaptatif : au moins 1 mot pour les courts, 1/3 pour les longs
                threshold = max(1, len(example_words) // 3)
                if len(common) >= threshold:
                    return True
            return False

        # Si pas de trigger_examples, match sur le worker_type seul
        return True

    # ──────────────────────────────────────────────
    #  5. MONITORING
    # ──────────────────────────────────────────────

    def evaluate_effectiveness(self, patch_id: str) -> PatchEvaluation:
        """
        Évalue l'efficacité d'un patch en comparant les métriques avant/après.
        """
        with self._lock:
            patch = self._patches.get(patch_id)

        if not patch:
            return PatchEvaluation(
                patch_id=patch_id,
                is_effective=False,
                confidence=0.0,
                before_success_rate=0.0,
                after_success_rate=0.0,
                improvement=0.0,
                sample_size=0,
                recommendation="not_found",
            )

        # Métriques actuelles
        metrics_now = self._capture_metrics(
            patch.target, patch.condition.get("error_type", "")
        )

        before_sr = patch.metrics_before.get("success_rate", 0.5)
        after_sr = metrics_now.get("success_rate", before_sr)
        sample = metrics_now.get("total_tasks", 0)

        improvement = after_sr - before_sr
        confidence = min(sample / 20.0, 1.0)  # Confiance basée sur la taille

        recommendation = self._recommend(improvement, confidence, patch.applied_count)

        evaluation = PatchEvaluation(
            patch_id=patch_id,
            is_effective=improvement > 0,
            confidence=confidence,
            before_success_rate=before_sr,
            after_success_rate=after_sr,
            improvement=improvement,
            sample_size=sample,
            recommendation=recommendation,
        )

        # Mettre à jour les métriques du patch
        with self._lock:
            patch.metrics_after = metrics_now
            patch.confidence = confidence

        return evaluation

    def _recommend(self, improvement: float, confidence: float, applied_count: int) -> str:
        """Génère une recommandation basée sur l'évaluation."""
        if confidence < 0.3:
            return "insufficient_data"
        if improvement >= 0.15:
            return "keep"
        if improvement >= 0.0:
            return "improve"
        return "rollback"

    # ──────────────────────────────────────────────
    #  6. ROLLBACK
    # ──────────────────────────────────────────────

    def rollback_patch(self, patch_id: str, reason: str = "manual") -> bool:
        """Désactive un patch."""
        with self._lock:
            patch = self._patches.get(patch_id)
            if not patch:
                return False
            patch.enabled = False
            patch.rollback_reason = reason

        self._save_patches()
        logger.warning("SelfPatcher: rollback du patch %s (raison=%s)", patch_id, reason)
        return True

    def evaluate_and_rollback_all(self) -> list[str]:
        """
        Évalue tous les patches actifs et rollback ceux qui sont inefficaces.
        Retourne la liste des patch_id rollbackés.
        """
        rolled_back = []

        with self._lock:
            active_ids = [
                pid for pid, p in self._patches.items() if p.enabled
            ]

        for pid in active_ids:
            try:
                evaluation = self.evaluate_effectiveness(pid)
                if (
                    evaluation.confidence >= 0.5
                    and evaluation.recommendation == "rollback"
                ):
                    self.rollback_patch(pid, reason=f"auto_rollback_improvement={evaluation.improvement:.2f}")
                    rolled_back.append(pid)
            except Exception as exc:
                logger.debug("SelfPatcher: erreur évaluation %s: %s", pid, exc)

        return rolled_back

    # ──────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────

    def _is_recent(self, last_seen: str) -> bool:
        """Vérifie si une date est récente (< RECENT_WINDOW_DAYS)."""
        try:
            dt = datetime.fromisoformat(last_seen)
            return (datetime.now() - dt).days < RECENT_WINDOW_DAYS
        except (ValueError, TypeError):
            return False

    def _has_active_patch_for(self, pattern) -> bool:
        """Vérifie s'il existe déjà un patch actif pour ce pattern."""
        pid = self._make_patch_id(pattern)
        with self._lock:
            existing = self._patches.get(pid)
        return existing is not None and existing.enabled

    def _make_patch_id(self, pattern) -> str:
        """Génère un ID déterministe pour un ErrorPattern."""
        key = f"{pattern.error_type}:{pattern.worker_type}"
        return f"patch_{hashlib.md5(key.encode()).hexdigest()[:12]}"

    def _capture_metrics(self, worker_type: str, error_type: str) -> dict:
        """Capture les métriques actuelles pour un worker_type donné."""
        try:
            summary = self._learning.get_performance_summary()
            perf = summary.get(worker_type, {})
            total = perf.get("total_tasks", 0)
            success = perf.get("success_count", 0)
            errors = perf.get("failure_count", 0)

            success_rate = success / total if total > 0 else 0.5
            error_rate = errors / total if total > 0 else 0.0

            return {
                "total_tasks": total,
                "success_count": success,
                "error_count": errors,
                "success_rate": success_rate,
                "error_rate": error_rate,
            }
        except Exception as exc:
            logger.debug("SelfPatcher: erreur capture métriques: %s", exc)
            return {"total_tasks": 0, "success_rate": 0.5, "error_rate": 0.0}

    # ──────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────

    def _save_patches(self) -> None:
        """Persiste tous les patches en JSON."""
        self._patches_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            for patch_id, patch in self._patches.items():
                path = self._patches_dir / f"{patch_id}.json"
                try:
                    with open(path, "w") as fh:
                        json.dump(patch.to_dict(), fh, indent=2, ensure_ascii=False)
                except OSError as exc:
                    logger.error("SelfPatcher: impossible d'écrire %s: %s", path, exc)

    def _load_patches(self) -> None:
        """Charge les patches depuis data/patches/*.json."""
        if not self._patches_dir.exists():
            return

        with self._lock:
            for path in self._patches_dir.glob("*.json"):
                try:
                    with open(path) as fh:
                        data = json.load(fh)
                    patch = PatchMetadata.from_dict(data)
                    self._patches[patch.patch_id] = patch
                except Exception as exc:
                    logger.warning("SelfPatcher: impossible de charger %s: %s", path, exc)

    # ──────────────────────────────────────────────
    #  Stats
    # ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Retourne les statistiques du SelfPatcher."""
        with self._lock:
            total = len(self._patches)
            active = sum(1 for p in self._patches.values() if p.enabled)
            rolled_back = sum(1 for p in self._patches.values() if p.rollback_reason)
            total_applied = sum(p.applied_count for p in self._patches.values())

        return {
            "total_patches": total,
            "active_patches": active,
            "rolled_back_patches": rolled_back,
            "total_applied": total_applied,
        }

    @property
    def active_patches(self) -> list[PatchMetadata]:
        """Liste des patches actifs."""
        with self._lock:
            return [p for p in self._patches.values() if p.enabled]

    @property
    def all_patches(self) -> list[PatchMetadata]:
        """Liste de tous les patches."""
        with self._lock:
            return list(self._patches.values())
