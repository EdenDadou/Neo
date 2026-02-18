"""
Neo Core — Plugin Loader
=========================
Charge et gère les plugins dynamiques depuis data/plugins/.

Chaque plugin est un fichier .py avec:
- PLUGIN_META dict (name, description, input_schema, worker_types)
- execute(**kwargs) function

Sécurité:
- Validation du PLUGIN_META avant chargement
- Les plugins tournent dans le même process mais sont validés
- Un plugin qui crashe ne fait pas tomber le système
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# Timeout max pour l'exécution d'un plugin (secondes)
PLUGIN_EXECUTION_TIMEOUT = 30

logger = logging.getLogger(__name__)


@dataclass
class LoadedPlugin:
    """Représente un plugin chargé."""
    name: str
    description: str
    version: str
    input_schema: dict
    worker_types: list[str]
    execute_fn: Callable
    filepath: Path
    loaded_at: str

    def to_dict(self) -> dict:
        """Convertit le plugin en dictionnaire pour serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "input_schema": self.input_schema,
            "worker_types": self.worker_types,
            "filepath": str(self.filepath),
            "loaded_at": self.loaded_at,
        }


class PluginLoader:
    """
    Charge et gère les plugins dynamiques depuis data/plugins/.
    Thread-safe pour les opérations de chargement/déchargement.
    """

    REQUIRED_META_FIELDS = ["name", "description", "version", "input_schema", "worker_types"]

    def __init__(self, plugins_dir: Path):
        self.plugins_dir = Path(plugins_dir)
        self._plugins: dict[str, LoadedPlugin] = {}
        self._lock = threading.Lock()
        self._loaded_modules: dict[str, object] = {}  # Pour tracking des modules

    def discover(self) -> dict[str, str]:
        """
        Scan plugins_dir for .py files et charge les plugins valides.

        Returns:
            dict avec keys:
            - "loaded": list de noms de plugins chargés
            - "errors": dict[name -> error_msg] pour les plugins échoués
        """
        results = {"loaded": [], "errors": {}}

        # Créer le répertoire s'il n'existe pas
        if not self.plugins_dir.exists():
            logger.info("Plugin directory does not exist: %s", self.plugins_dir)
            return results

        if not self.plugins_dir.is_dir():
            logger.warning("Plugin path is not a directory: %s", self.plugins_dir)
            return results

        # Scanner les fichiers .py
        plugin_files = sorted(self.plugins_dir.glob("*.py"))
        if not plugin_files:
            logger.debug("No plugin files found in %s", self.plugins_dir)
            return results

        for filepath in plugin_files:
            # Ignorer les fichiers privés
            if filepath.name.startswith("_"):
                continue

            plugin_name = filepath.stem
            try:
                loaded_plugin = self.load_plugin(filepath)
                if loaded_plugin:
                    results["loaded"].append(loaded_plugin.name)
                    logger.info("Plugin loaded: %s (v%s)", loaded_plugin.name, loaded_plugin.version)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                results["errors"][plugin_name] = error_msg
                logger.error("Failed to load plugin %s: %s", plugin_name, error_msg)

        return results

    def load_plugin(self, filepath: Path) -> LoadedPlugin | None:
        """
        Load a single plugin file, validate, and register.

        Returns:
            LoadedPlugin instance or None if validation failed

        Raises:
            Various exceptions during loading (caught by discover())
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Plugin file not found: {filepath}")

        if not filepath.is_file():
            raise ValueError(f"Plugin path is not a file: {filepath}")

        # Pré-validation : scanner les imports dangereux AVANT d'exécuter le code
        import re as _re
        source_code = filepath.read_text(encoding="utf-8")
        _DANGEROUS_IMPORTS = frozenset({
            "os", "subprocess", "sys", "importlib", "shutil",
            "socket", "ctypes", "signal", "multiprocessing",
        })
        _DANGEROUS_BUILTINS = [
            r"\b__import__\s*\(", r"\bexec\s*\(", r"\beval\s*\(",
            r"\bcompile\s*\(", r"\bglobals\s*\(",
        ]
        found_imports = _re.findall(r"^\s*(?:from|import)\s+(\w+)", source_code, _re.MULTILINE)
        for imp in found_imports:
            if imp in _DANGEROUS_IMPORTS:
                raise ValueError(f"Plugin contains forbidden import: {imp}")
        for pattern in _DANGEROUS_BUILTINS:
            if _re.search(pattern, source_code):
                raise ValueError(f"Plugin contains dangerous builtin: {pattern}")

        # Charger le module Python avec namespace isolé
        module_name = f"neo_plugin_{filepath.stem}"
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from {filepath}")

        module = importlib.util.module_from_spec(spec)

        # Namespace isolé pour éviter les collisions
        sys.modules[module_name] = module
        self._loaded_modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            if module_name in sys.modules:
                del sys.modules[module_name]
            raise RuntimeError(f"Failed to execute module: {e}") from e

        # Valider PLUGIN_META
        plugin_meta = getattr(module, "PLUGIN_META", None)
        if plugin_meta is None:
            raise ValueError("Plugin must define PLUGIN_META")

        if not isinstance(plugin_meta, dict):
            raise ValueError("PLUGIN_META must be a dict")

        # Vérifier les champs requis
        missing_fields = [f for f in self.REQUIRED_META_FIELDS if f not in plugin_meta]
        if missing_fields:
            raise ValueError(f"PLUGIN_META missing required fields: {missing_fields}")

        # Valider les types
        for field in ["name", "description", "version"]:
            if not isinstance(plugin_meta[field], str):
                raise ValueError(f"PLUGIN_META.{field} must be a string")

        if not isinstance(plugin_meta["input_schema"], dict):
            raise ValueError("PLUGIN_META.input_schema must be a dict")

        if not isinstance(plugin_meta["worker_types"], (list, tuple)):
            raise ValueError("PLUGIN_META.worker_types must be a list or tuple")

        # Vérifier la fonction execute
        execute_fn = getattr(module, "execute", None)
        if execute_fn is None:
            raise ValueError("Plugin must define an execute() function")

        if not callable(execute_fn):
            raise ValueError("execute must be callable")

        # Créer le plugin chargé
        loaded_plugin = LoadedPlugin(
            name=plugin_meta["name"],
            description=plugin_meta["description"],
            version=plugin_meta["version"],
            input_schema=plugin_meta["input_schema"],
            worker_types=list(plugin_meta["worker_types"]),
            execute_fn=execute_fn,
            filepath=filepath,
            loaded_at=datetime.now(timezone.utc).isoformat(),
        )

        # Enregistrer
        with self._lock:
            self._plugins[loaded_plugin.name] = loaded_plugin

        return loaded_plugin

    def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin by name.

        Returns:
            True if unloaded, False if not found
        """
        with self._lock:
            if name not in self._plugins:
                return False

            plugin = self._plugins[name]

            # Nettoyer le module de sys.modules (namespace isolé)
            module_name = f"neo_plugin_{plugin.filepath.stem}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            if module_name in self._loaded_modules:
                del self._loaded_modules[module_name]

            del self._plugins[name]
            logger.info("Plugin unloaded: %s", name)
            return True

    def reload_all(self) -> dict:
        """
        Reload all plugins (hot-reload support).

        Returns:
            dict with "reloaded" and "errors" lists
        """
        # Sauvegarder les chemins AVANT de verrouiller
        with self._lock:
            filepaths = [p.filepath for p in self._plugins.values()]
            # Décharger tous les plugins
            names_to_unload = list(self._plugins.keys())

        # Décharger en dehors de la boucle de chargement pour éviter deadlock
        for name in names_to_unload:
            self.unload_plugin(name)

        # Recharger tous
        results = {"reloaded": [], "errors": {}}
        for filepath in filepaths:
            try:
                loaded_plugin = self.load_plugin(filepath)
                if loaded_plugin:
                    results["reloaded"].append(loaded_plugin.name)
            except Exception as e:
                results["errors"][filepath.stem] = str(e)

        return results

    def get_plugin(self, name: str) -> LoadedPlugin | None:
        """Get a loaded plugin by name."""
        with self._lock:
            return self._plugins.get(name)

    def execute_plugin(self, name: str, args: dict) -> str:
        """
        Execute a plugin safely with error handling.

        Args:
            name: Plugin name
            args: Arguments dict matching input_schema

        Returns:
            Result as string or error message
        """
        plugin = self.get_plugin(name)
        if plugin is None:
            return f"Error: Plugin '{name}' not found"

        try:
            # Exécuter avec timeout via thread pour ne pas bloquer
            result_container = [None]
            error_container = [None]

            def _run():
                try:
                    result_container[0] = plugin.execute_fn(**args)
                except Exception as exc:
                    error_container[0] = exc

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=PLUGIN_EXECUTION_TIMEOUT)

            if t.is_alive():
                logger.error("Plugin '%s' execution timed out after %ds", name, PLUGIN_EXECUTION_TIMEOUT)
                return f"Error: Plugin '{name}' timed out after {PLUGIN_EXECUTION_TIMEOUT}s"

            if error_container[0] is not None:
                raise error_container[0]

            result = result_container[0]

            # Convertir le résultat en string
            if isinstance(result, str):
                return result
            else:
                return str(result)
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error("Plugin '%s' execution failed:\n%s", name, error_msg)
            return f"Error executing plugin '{name}': {type(e).__name__}: {e}"

    def list_plugins(self) -> list[dict]:
        """
        List all loaded plugins with their metadata.

        Returns:
            List of plugin dicts (name, description, version, etc.)
        """
        with self._lock:
            return [p.to_dict() for p in self._plugins.values()]

    def get_plugins_for_worker_type(self, worker_type: str) -> list[LoadedPlugin]:
        """
        Get all plugins available for a worker type.

        Args:
            worker_type: e.g., "researcher", "coder", etc.

        Returns:
            List of LoadedPlugin matching the worker type
        """
        with self._lock:
            return [
                p for p in self._plugins.values()
                if worker_type in p.worker_types
            ]

    def get_plugin_schemas(self) -> dict[str, dict]:
        """
        Get schemas for all loaded plugins (tool_use format).

        Returns:
            dict mapping plugin_name -> schema
        """
        schemas = {}
        with self._lock:
            for plugin in self._plugins.values():
                schemas[plugin.name] = {
                    "name": plugin.name,
                    "description": plugin.description,
                    "input_schema": plugin.input_schema,
                }
        return schemas
