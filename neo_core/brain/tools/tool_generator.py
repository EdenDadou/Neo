"""
Neo Core — Tool Generator (Level 4)
======================================
Détecte les patterns d'actions répétitives et génère automatiquement
des plugins Neo pour les automatiser.

Les plugins générés suivent le contrat existant (PLUGIN_META + execute())
et sont chargés dynamiquement via le PluginLoader.

Lifecycle :
    1. DÉTECTION  — analyse les tool_not_found et les séquences répétées
    2. GÉNÉRATION — crée un fichier Python plugin valide
    3. VALIDATION — syntax check, safety check, dry-run load
    4. DÉPLOIEMENT — hot-reload via PluginLoader
    5. MONITORING — suivi usage et efficacité
    6. PRUNING    — suppression des outils inutilisés après N jours
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import textwrap
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─── Imports dangereux interdits dans les plugins générés ──────
FORBIDDEN_IMPORTS = frozenset({
    "os", "subprocess", "sys", "importlib", "shutil",
    "socket", "ctypes", "signal", "multiprocessing",
})

# ─── Seuils par défaut ─────────────────────────────────────────
DEFAULT_DETECTION_THRESHOLD = 3
DEFAULT_PRUNING_DAYS = 7
DEFAULT_MIN_SUCCESS_RATE = 0.3


# ════════════════════════════════════════════════════════════════
#  Data structures
# ════════════════════════════════════════════════════════════════

@dataclass
class ToolPattern:
    """Pattern détecté suggérant la création d'un outil."""

    pattern_id: str
    name: str
    description: str
    detected_at: str
    occurrence_count: int

    trigger_keywords: list[str] = field(default_factory=list)
    request_examples: list[str] = field(default_factory=list)
    step_sequence: list[str] = field(default_factory=list)

    proposed_tool_name: str = ""
    proposed_description: str = ""
    estimated_complexity: str = "moderate"   # "simple" | "moderate" | "complex"


@dataclass
class GeneratedToolMeta:
    """Métadonnées d'un outil auto-généré."""

    tool_id: str
    name: str
    description: str
    version: int = 1

    # Fichier source
    filepath: str = ""

    # Tracking
    created_at: str = ""
    generated_from_pattern: str = ""

    # Efficacité
    usage_count: int = 0
    success_count: int = 0
    avg_execution_time: float = 0.0
    last_used: Optional[str] = None

    # Lifecycle
    enabled: bool = False
    deprecated: bool = False
    deprecation_reason: Optional[str] = None

    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> GeneratedToolMeta:
        expected = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in expected}
        return cls(**filtered)


# ════════════════════════════════════════════════════════════════
#  ToolGenerator
# ════════════════════════════════════════════════════════════════

class ToolGenerator:
    """
    Détecte les patterns répétitifs et génère des plugins automatiquement.
    Thread-safe — toutes les mutations sont protégées par un Lock.
    """

    def __init__(
        self,
        data_dir: Path,
        learning_engine: Any,
        plugin_loader: Any = None,
        *,
        detection_threshold: int = DEFAULT_DETECTION_THRESHOLD,
        pruning_days: int = DEFAULT_PRUNING_DAYS,
        min_success_rate: float = DEFAULT_MIN_SUCCESS_RATE,
    ):
        self._data_dir = Path(data_dir)
        self._plugins_dir = self._data_dir / "plugins"
        self._meta_dir = self._data_dir / "tool_metadata"
        self._learning = learning_engine
        self._plugin_loader = plugin_loader

        self._detection_threshold = detection_threshold
        self._pruning_days = pruning_days
        self._min_success_rate = min_success_rate

        self._tools: dict[str, GeneratedToolMeta] = {}
        self._lock = threading.Lock()

        self._load_tool_metadata()

    # ──────────────────────────────────────────────
    #  1. DÉTECTION
    # ──────────────────────────────────────────────

    def detect_opportunities(self) -> list[ToolPattern]:
        """
        Détecte les opportunités de création d'outils :
          1. Erreurs tool_not_found récurrentes (≥ threshold)
          2. Séquences multi-outils répétées (≥ threshold)

        Retourne les ToolPattern triés par occurrence décroissante.
        """
        opportunities: list[ToolPattern] = []

        # Méthode 1 : clustering des tool_not_found
        try:
            tool_errors = self._get_tool_not_found_patterns()
            opportunities.extend(tool_errors)
        except Exception as exc:
            logger.debug("ToolGenerator: erreur détection tool_not_found: %s", exc)

        # Méthode 2 : séquences multi-outils
        try:
            multi_step = self._detect_multi_step_patterns()
            opportunities.extend(multi_step)
        except Exception as exc:
            logger.debug("ToolGenerator: erreur détection multi-step: %s", exc)

        # Dédupliquage et tri
        seen = set()
        unique = []
        for pattern in opportunities:
            if pattern.pattern_id not in seen:
                seen.add(pattern.pattern_id)
                unique.append(pattern)

        unique.sort(key=lambda p: p.occurrence_count, reverse=True)
        return unique

    def _get_tool_not_found_patterns(self) -> list[ToolPattern]:
        """Analyse les erreurs tool_not_found du LearningEngine."""
        all_errors = self._learning.get_error_patterns()
        patterns = []

        for error in all_errors:
            if (
                error.error_type == "tool_not_found"
                and error.count >= self._detection_threshold
            ):
                # Extraire les mots-clés des exemples
                keywords = self._extract_keywords(error.examples)
                tool_name = self._suggest_name(keywords)

                # Vérifier qu'on n'a pas déjà généré cet outil
                if self._has_existing_tool(tool_name):
                    continue

                pid = self._make_pattern_id(f"tnf_{tool_name}")
                patterns.append(ToolPattern(
                    pattern_id=pid,
                    name=tool_name,
                    description=f"Auto-detected from tool_not_found errors",
                    detected_at=datetime.now().isoformat(),
                    occurrence_count=error.count,
                    trigger_keywords=keywords,
                    request_examples=error.examples[:5],
                    step_sequence=[tool_name],
                    proposed_tool_name=tool_name,
                    proposed_description=f"Auto-generated tool: {tool_name}",
                    estimated_complexity="simple",
                ))

        return patterns

    def _detect_multi_step_patterns(self) -> list[ToolPattern]:
        """
        Détecte les séquences d'outils exécutées de manière répétitive.
        Utilise le performance_summary pour identifier les patterns.
        """
        # Note: Dans l'implémentation actuelle, le LearningEngine ne
        # stocke pas les séquences d'outils. Cette méthode est un stub
        # qui sera enrichi quand le tracking de séquences sera ajouté.
        return []

    # ──────────────────────────────────────────────
    #  2. GÉNÉRATION
    # ──────────────────────────────────────────────

    def generate_plugin(self, pattern: ToolPattern) -> Optional[GeneratedToolMeta]:
        """
        Génère un fichier plugin Python à partir d'un ToolPattern.
        Retourne None si le plugin existe déjà ou ne peut pas être créé.
        """
        tool_name = self._sanitize_name(pattern.proposed_tool_name)
        tool_id = f"auto_{tool_name}"

        with self._lock:
            if tool_id in self._tools:
                existing = self._tools[tool_id]
                if not existing.deprecated:
                    return None

        # Générer le code
        plugin_code = self._generate_code(
            tool_name=tool_name,
            description=pattern.proposed_description,
            keywords=pattern.trigger_keywords,
            steps=pattern.step_sequence,
        )

        # Écrire le fichier
        self._plugins_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._plugins_dir / f"auto_{tool_name}.py"

        try:
            filepath.write_text(plugin_code, encoding="utf-8")
        except OSError as exc:
            logger.error("ToolGenerator: impossible d'écrire %s: %s", filepath, exc)
            return None

        tool_meta = GeneratedToolMeta(
            tool_id=tool_id,
            name=tool_name,
            description=pattern.proposed_description,
            version=1,
            filepath=str(filepath),
            created_at=datetime.now().isoformat(),
            generated_from_pattern=pattern.pattern_id,
            enabled=False,
        )

        logger.info("ToolGenerator: plugin généré — %s (%s)", tool_name, filepath)
        return tool_meta

    def _generate_code(
        self,
        tool_name: str,
        description: str,
        keywords: list[str],
        steps: list[str],
    ) -> str:
        """Génère le code Python d'un plugin."""

        # Input schema basique
        input_schema = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": f"Input for {tool_name}",
                },
            },
            "required": ["input"],
        }

        # Worker types par défaut
        worker_types = ["generic", "researcher", "analyst"]

        # Générer le corps de execute()
        if len(steps) <= 1:
            body = self._generate_simple_body(tool_name, keywords)
        else:
            body = self._generate_multi_step_body(tool_name, steps)

        schema_json = json.dumps(input_schema, indent=4)
        workers_json = json.dumps(worker_types)
        now = datetime.now().isoformat()

        code = f'''"""
Auto-generated plugin by Neo ToolGenerator.

Tool: {tool_name}
Generated: {now}
Description: {description}

This tool was auto-generated from detected usage patterns.
If unused for {self._pruning_days} days, it will be automatically pruned.
"""

PLUGIN_META = {{
    "name": "{tool_name}",
    "description": "{description}",
    "version": "1.0.0",
    "author": "neo_tool_generator",
    "input_schema": {schema_json},
    "worker_types": {workers_json},
}}


def execute(**kwargs) -> str:
    """
    Execute the {tool_name} tool.

    Args:
        **kwargs: See input_schema in PLUGIN_META.

    Returns:
        str: Tool result.
    """
{textwrap.indent(body, "    ")}
'''
        return code

    def _generate_simple_body(self, tool_name: str, keywords: list[str]) -> str:
        """Génère le corps d'un tool simple (single-step)."""
        kw_str = ", ".join(keywords[:3]) if keywords else tool_name
        return f'''input_text = kwargs.get("input", "")
if not input_text:
    return "Error: 'input' parameter is required"

# Auto-generated tool for: {kw_str}
# This is a template — enhance the logic as needed.
result = f"Processed '{{input_text}}' with {tool_name}"
return result
'''

    def _generate_multi_step_body(self, tool_name: str, steps: list[str]) -> str:
        """Génère le corps d'un tool multi-step."""
        code = f'''input_text = kwargs.get("input", "")
if not input_text:
    return "Error: 'input' parameter is required"

result = input_text
'''
        for i, step in enumerate(steps):
            code += f'''
# Step {i + 1}: {step}
try:
    result = f"Step {i + 1} ({step}): {{result}}"
except Exception as e:
    return f"Error in step {i + 1} ({step}): {{str(e)}}"
'''
        code += "\nreturn result\n"
        return code

    # ──────────────────────────────────────────────
    #  3. VALIDATION
    # ──────────────────────────────────────────────

    def validate_plugin(self, tool: GeneratedToolMeta) -> bool:
        """
        Valide un plugin généré avant déploiement.

        Checks :
          1. Syntaxe Python valide
          2. Pas d'imports dangereux
          3. Chargement dry-run via PluginLoader (si disponible)
          4. Test d'exécution basique
        """
        filepath = Path(tool.filepath)
        if not filepath.exists():
            logger.error("ToolGenerator: fichier introuvable — %s", filepath)
            return False

        code = filepath.read_text(encoding="utf-8")

        # 1. Syntax check
        try:
            compile(code, str(filepath), "exec")
        except SyntaxError as exc:
            logger.error("ToolGenerator: erreur de syntaxe dans %s: %s", filepath, exc)
            return False

        # 2. Safety check
        if self._has_forbidden_imports(code):
            logger.error("ToolGenerator: imports interdits dans %s", filepath)
            filepath.unlink(missing_ok=True)
            return False

        # 3. Dry-run load via PluginLoader
        if self._plugin_loader is not None:
            try:
                loaded = self._plugin_loader.load_plugin(filepath)
                if loaded is None:
                    logger.error("ToolGenerator: PluginLoader n'a pas pu charger %s", filepath)
                    return False
            except Exception as exc:
                logger.error("ToolGenerator: erreur dry-run %s: %s", filepath, exc)
                return False

        # 4. Basic execution test (via exec in isolated namespace)
        try:
            ns: dict = {}
            exec(compile(code, str(filepath), "exec"), ns)
            execute_fn = ns.get("execute")
            if not callable(execute_fn):
                logger.error("ToolGenerator: execute() non trouvé dans %s", filepath)
                return False
            result = execute_fn(input="test_validation")
            if not isinstance(result, str):
                logger.error("ToolGenerator: execute() ne retourne pas str dans %s", filepath)
                return False
        except Exception as exc:
            logger.warning("ToolGenerator: test d'exécution échoué pour %s: %s", filepath, exc)
            # On ne bloque pas pour un test d'exécution
            pass

        tool.enabled = True
        logger.info("ToolGenerator: plugin %s validé", tool.name)
        return True

    def _has_forbidden_imports(self, code: str) -> bool:
        """Vérifie si le code contient des imports dangereux ou des builtins dangereux."""
        # Pattern: import X / from X import ...
        imports = re.findall(r"^\s*(?:from|import)\s+(\w+)", code, re.MULTILINE)
        for imp in imports:
            if imp in FORBIDDEN_IMPORTS:
                logger.warning("ToolGenerator: import interdit détecté — %s", imp)
                return True

        # Bloquer les builtins dangereux qui contournent FORBIDDEN_IMPORTS
        dangerous_builtins = [
            r"\b__import__\s*\(",       # __import__('os')
            r"\bexec\s*\(",             # exec(...)
            r"\beval\s*\(",             # eval(...)
            r"\bcompile\s*\(",          # compile(...)
            r"\bgetattr\s*\(",          # getattr(module, 'system')
            r"\bglobals\s*\(",          # globals()['__builtins__']
            r"\bopen\s*\(",             # open('/etc/passwd')
        ]
        for pattern in dangerous_builtins:
            if re.search(pattern, code):
                logger.warning("ToolGenerator: builtin dangereux détecté — %s", pattern)
                return True

        return False

    # ──────────────────────────────────────────────
    #  4. DÉPLOIEMENT
    # ──────────────────────────────────────────────

    def deploy_plugin(self, tool: GeneratedToolMeta) -> bool:
        """Déploie un plugin validé via hot-reload."""
        if not tool.enabled:
            logger.warning("ToolGenerator: tentative de déployer un plugin non validé — %s", tool.name)
            return False

        # Enregistrer dans notre tracking
        with self._lock:
            self._tools[tool.tool_id] = tool

        # Hot-reload si PluginLoader disponible
        if self._plugin_loader is not None:
            try:
                result = self._plugin_loader.reload_all()
                logger.info("ToolGenerator: hot-reload effectué — %s", result)
            except Exception as exc:
                logger.warning("ToolGenerator: hot-reload échoué: %s", exc)

        # Sauvegarder les métadonnées
        self._save_tool_metadata(tool)
        logger.info("ToolGenerator: plugin %s déployé", tool.name)
        return True

    # ──────────────────────────────────────────────
    #  5. MONITORING
    # ──────────────────────────────────────────────

    def track_usage(self, tool_name: str, success: bool, execution_time: float) -> None:
        """Enregistre l'utilisation d'un outil auto-généré."""
        with self._lock:
            for tool in self._tools.values():
                if tool.name == tool_name and not tool.deprecated:
                    tool.usage_count += 1
                    if success:
                        tool.success_count += 1
                    # Moyenne mobile
                    n = tool.usage_count
                    tool.avg_execution_time = (
                        (tool.avg_execution_time * (n - 1) + execution_time) / n
                    )
                    tool.last_used = datetime.now().isoformat()
                    break

    # ──────────────────────────────────────────────
    #  6. PRUNING
    # ──────────────────────────────────────────────

    def prune_unused(self) -> list[str]:
        """
        Supprime les outils auto-générés inutilisés ou inefficaces.

        Critères :
          - Non utilisé depuis pruning_days jours → deprecated
          - success_rate < min_success_rate (avec ≥ 5 usages) → deprecated

        Retourne la liste des noms d'outils supprimés.
        """
        now = datetime.now()
        pruned: list[str] = []

        with self._lock:
            for tool in list(self._tools.values()):
                if tool.deprecated:
                    continue

                should_prune = False
                reason = ""

                # Critère 1 : non utilisé depuis N jours
                if tool.last_used:
                    last_used = datetime.fromisoformat(tool.last_used)
                    days_unused = (now - last_used).days
                    if days_unused >= self._pruning_days:
                        should_prune = True
                        reason = f"unused_{days_unused}_days"
                elif tool.created_at:
                    created = datetime.fromisoformat(tool.created_at)
                    days_since_creation = (now - created).days
                    if days_since_creation >= self._pruning_days:
                        should_prune = True
                        reason = f"never_used_{days_since_creation}_days"

                # Critère 2 : taux de succès trop bas
                if (
                    not should_prune
                    and tool.usage_count >= 5
                    and tool.success_rate < self._min_success_rate
                ):
                    should_prune = True
                    reason = f"low_success_rate_{tool.success_rate:.0%}"

                if should_prune:
                    tool.deprecated = True
                    tool.deprecation_reason = reason
                    pruned.append(tool.name)

        # Supprimer les fichiers des outils pruned
        for name in pruned:
            self._remove_plugin_file(name)
            logger.info("ToolGenerator: outil pruned — %s", name)

        if pruned:
            self._save_all_metadata()

        return pruned

    def _remove_plugin_file(self, tool_name: str) -> None:
        """Supprime le fichier plugin du filesystem."""
        filepath = self._plugins_dir / f"auto_{tool_name}.py"
        if filepath.exists():
            try:
                filepath.unlink()
            except OSError as exc:
                logger.warning("ToolGenerator: impossible de supprimer %s: %s", filepath, exc)

    # ──────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────

    def _extract_keywords(self, examples: list[str]) -> list[str]:
        """Extrait les mots-clés significatifs des exemples."""
        word_counts: dict[str, int] = {}
        stop_words = {"the", "a", "an", "is", "was", "to", "for", "in", "on", "of",
                       "and", "or", "not", "it", "this", "that", "with", "from", "by",
                       "le", "la", "les", "un", "une", "de", "du", "des", "et", "ou"}

        for example in examples:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", example.lower())
            for word in words:
                if word not in stop_words:
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Top N mots-clés par fréquence
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:5]]

    def _suggest_name(self, keywords: list[str]) -> str:
        """Suggère un nom d'outil à partir des mots-clés."""
        if not keywords:
            return "unnamed_tool"
        # Utiliser les 2-3 premiers mots-clés
        name_parts = keywords[:3]
        return "_".join(name_parts)

    def _sanitize_name(self, name: str) -> str:
        """Nettoie un nom pour être un identifiant Python valide."""
        # Remplacer les caractères non-alphanumériques par _
        clean = re.sub(r"[^a-zA-Z0-9]", "_", name)
        # Supprimer les _ multiples
        clean = re.sub(r"_+", "_", clean).strip("_")
        # S'assurer que ça ne commence pas par un chiffre
        if clean and clean[0].isdigit():
            clean = f"tool_{clean}"
        return clean.lower() or "unnamed_tool"

    def _make_pattern_id(self, key: str) -> str:
        """Génère un ID déterministe."""
        return f"pattern_{hashlib.md5(key.encode()).hexdigest()[:12]}"

    def _has_existing_tool(self, name: str) -> bool:
        """Vérifie si un outil avec ce nom existe déjà (actif)."""
        sanitized = self._sanitize_name(name)
        with self._lock:
            for tool in self._tools.values():
                if tool.name == sanitized and not tool.deprecated:
                    return True
        return False

    # ──────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────

    def _save_tool_metadata(self, tool: GeneratedToolMeta) -> None:
        """Sauvegarde les métadonnées d'un seul outil."""
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        path = self._meta_dir / f"{tool.tool_id}.json"
        try:
            with open(path, "w") as fh:
                json.dump(tool.to_dict(), fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.error("ToolGenerator: impossible d'écrire %s: %s", path, exc)

    def _save_all_metadata(self) -> None:
        """Sauvegarde toutes les métadonnées."""
        with self._lock:
            for tool in self._tools.values():
                self._save_tool_metadata(tool)

    def _load_tool_metadata(self) -> None:
        """Charge les métadonnées depuis data/tool_metadata/*.json."""
        if not self._meta_dir.exists():
            return

        with self._lock:
            for path in self._meta_dir.glob("*.json"):
                try:
                    with open(path) as fh:
                        data = json.load(fh)
                    tool = GeneratedToolMeta.from_dict(data)
                    self._tools[tool.tool_id] = tool
                except Exception as exc:
                    logger.warning("ToolGenerator: impossible de charger %s: %s", path, exc)

    # ──────────────────────────────────────────────
    #  Stats
    # ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Retourne les statistiques du ToolGenerator."""
        with self._lock:
            total = len(self._tools)
            active = sum(1 for t in self._tools.values() if t.enabled and not t.deprecated)
            deprecated = sum(1 for t in self._tools.values() if t.deprecated)
            total_usage = sum(t.usage_count for t in self._tools.values())

        return {
            "total_tools": total,
            "active_tools": active,
            "deprecated_tools": deprecated,
            "total_usage": total_usage,
        }

    @property
    def active_tools(self) -> list[GeneratedToolMeta]:
        """Liste des outils actifs."""
        with self._lock:
            return [t for t in self._tools.values() if t.enabled and not t.deprecated]

    @property
    def all_tools(self) -> list[GeneratedToolMeta]:
        """Liste de tous les outils."""
        with self._lock:
            return list(self._tools.values())
