"""
Neo Core — Outils de base pour les Workers
=============================================
Outils LangChain utilisés par les agents spécialisés (Workers).

Chaque outil :
- Est un LangChain BaseTool
- Supporte un mode mock (retourne des résultats déterministes)
- A des protections de sécurité intégrées

Le ToolRegistry gère le chargement et la sélection d'outils par type de Worker.
"""

from __future__ import annotations

import io
import os
import sys
import textwrap
import time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from langchain_core.tools import tool

if TYPE_CHECKING:
    from neo_core.core.memory_agent import MemoryAgent


# ─── Variable globale pour le mode mock ───────────────────
_MOCK_MODE = False
_MEMORY_REF: Optional[MemoryAgent] = None


def set_mock_mode(enabled: bool) -> None:
    """Active/désactive le mode mock pour tous les outils."""
    global _MOCK_MODE
    _MOCK_MODE = enabled


def set_memory_ref(memory: Optional[MemoryAgent]) -> None:
    """Définit la référence mémoire pour l'outil memory_search."""
    global _MEMORY_REF
    _MEMORY_REF = memory


def is_mock_mode() -> bool:
    """Vérifie si les outils sont en mode mock."""
    return _MOCK_MODE


# ─── Outil : Recherche web ────────────────────────────────

@tool
def web_search_tool(query: str) -> str:
    """
    Recherche sur le web. Retourne des résultats pertinents.

    Args:
        query: Termes de recherche

    Returns:
        Résultats formatés (titre, URL, extrait)
    """
    if _MOCK_MODE:
        return (
            f"[Mock Web Search] Résultats pour '{query}':\n"
            f"1. «{query} - Guide complet» — https://example.com/guide\n"
            f"   Un guide détaillé couvrant tous les aspects de {query}.\n"
            f"2. «{query} - Dernières avancées» — https://example.com/news\n"
            f"   Les développements récents dans le domaine.\n"
            f"3. «{query} - Tutoriel pratique» — https://example.com/tutorial\n"
            f"   Un tutoriel étape par étape pour maîtriser le sujet."
        )

    # Mode réel — utilise httpx pour une recherche basique
    # (peut être étendu avec Google Custom Search, Brave API, etc.)
    try:
        import httpx
        # Fallback : DuckDuckGo lite (pas de clé API requise)
        response = httpx.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            timeout=15,
            follow_redirects=True,
        )
        if response.status_code == 200:
            # Parser les résultats basiques
            text = response.text[:2000]
            return f"Résultats web pour '{query}':\n{text}"
        return f"Recherche web échouée (HTTP {response.status_code})"
    except Exception as e:
        return f"Erreur recherche web: {e}"


# ─── Outil : Lecture de fichier ───────────────────────────

# Chemins autorisés pour la lecture
_ALLOWED_READ_ROOTS: list[Path] = []


def set_allowed_read_roots(roots: list[str | Path]) -> None:
    """Définit les répertoires racines autorisés pour la lecture."""
    global _ALLOWED_READ_ROOTS
    _ALLOWED_READ_ROOTS = [Path(r).resolve() for r in roots]


def _is_path_safe(path: str, roots: list[Path]) -> bool:
    """Vérifie qu'un chemin est dans les répertoires autorisés."""
    resolved = Path(path).resolve()
    if not roots:
        return True  # Pas de restriction si aucune racine définie
    return any(str(resolved).startswith(str(root)) for root in roots)


@tool
def file_read_tool(path: str) -> str:
    """
    Lit le contenu d'un fichier.

    Args:
        path: Chemin du fichier à lire

    Returns:
        Contenu du fichier ou message d'erreur
    """
    if _MOCK_MODE:
        return f"[Mock File Read] Contenu de '{path}':\nCeci est le contenu simulé du fichier."

    try:
        # Sécurité : pas de traversal
        if ".." in path:
            return "Erreur: Traversal de répertoire interdit (..)"

        if not _is_path_safe(path, _ALLOWED_READ_ROOTS):
            return f"Erreur: Chemin non autorisé: {path}"

        file_path = Path(path)
        if not file_path.exists():
            return f"Erreur: Fichier introuvable: {path}"

        if file_path.stat().st_size > 1_000_000:  # 1MB max
            return f"Erreur: Fichier trop volumineux ({file_path.stat().st_size} octets, max 1MB)"

        return file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Erreur lecture fichier: {e}"


# ─── Outil : Écriture de fichier ─────────────────────────

_ALLOWED_WRITE_ROOTS: list[Path] = []


def set_allowed_write_roots(roots: list[str | Path]) -> None:
    """Définit les répertoires racines autorisés pour l'écriture."""
    global _ALLOWED_WRITE_ROOTS
    _ALLOWED_WRITE_ROOTS = [Path(r).resolve() for r in roots]


@tool
def file_write_tool(path: str, content: str) -> str:
    """
    Écrit du contenu dans un fichier.

    Args:
        path: Chemin du fichier à écrire
        content: Contenu à écrire

    Returns:
        Message de confirmation ou d'erreur
    """
    if _MOCK_MODE:
        return f"[Mock File Write] Fichier '{path}' écrit ({len(content)} caractères)"

    try:
        if ".." in path:
            return "Erreur: Traversal de répertoire interdit (..)"

        if not _is_path_safe(path, _ALLOWED_WRITE_ROOTS):
            return f"Erreur: Chemin d'écriture non autorisé: {path}"

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

        return f"Fichier '{path}' écrit avec succès ({len(content)} caractères)"
    except Exception as e:
        return f"Erreur écriture fichier: {e}"


# ─── Outil : Exécution de code Python ────────────────────

@tool
def code_execute_tool(code: str) -> str:
    """
    Exécute du code Python dans un environnement contrôlé.

    Args:
        code: Code Python à exécuter

    Returns:
        Sortie standard ou message d'erreur
    """
    if _MOCK_MODE:
        # En mock, on simule une exécution réussie
        lines = code.strip().split("\n")
        last_line = lines[-1].strip() if lines else ""
        if last_line.startswith("print("):
            # Simuler un print basique
            content = last_line[6:-1].strip("'\"")
            return f"[Mock Code Execute] Output:\n{content}"
        return f"[Mock Code Execute] Code exécuté ({len(lines)} lignes), aucune sortie."

    try:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Environnement restreint
        safe_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "bool": bool,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "reversed": reversed,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "isinstance": isinstance,
                "type": type,
                "True": True,
                "False": False,
                "None": None,
            }
        }

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, safe_globals)

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        result = ""
        if output:
            result += output
        if errors:
            result += f"\nStderr:\n{errors}"
        if not result.strip():
            result = "Code exécuté avec succès (aucune sortie)."

        return result[:5000]  # Limiter la sortie
    except Exception as e:
        return f"Erreur d'exécution: {type(e).__name__}: {e}"


# ─── Outil : Recherche sémantique dans Memory ────────────

@tool
def memory_search_tool(query: str) -> str:
    """
    Recherche dans la mémoire du système pour trouver des informations pertinentes.

    Args:
        query: Requête de recherche

    Returns:
        Résultats pertinents depuis la mémoire
    """
    if _MOCK_MODE:
        return (
            f"[Mock Memory Search] Résultats pour '{query}':\n"
            f"- Souvenir pertinent 1: Information liée à '{query}'\n"
            f"- Souvenir pertinent 2: Contexte historique sur le sujet"
        )

    if _MEMORY_REF is None:
        return "Mémoire non disponible."

    try:
        context = _MEMORY_REF.get_context(query)
        if context and context != "Aucun contexte mémoire disponible.":
            return f"Résultats mémoire pour '{query}':\n{context}"
        return f"Aucun résultat en mémoire pour '{query}'."
    except Exception as e:
        return f"Erreur recherche mémoire: {e}"


# ─── Registre d'outils ───────────────────────────────────

class ToolRegistry:
    """
    Registre centralisé des outils disponibles.

    Gère la sélection d'outils par type de Worker
    et l'initialisation globale (mock mode, mémoire, etc.)
    """

    # Mapping type de worker → noms d'outils
    WORKER_TOOLS: dict[str, list[str]] = {
        "researcher": ["web_search", "memory_search", "file_read"],
        "coder": ["code_execute", "file_read", "file_write", "memory_search"],
        "summarizer": ["file_read", "memory_search"],
        "analyst": ["code_execute", "file_read", "memory_search"],
        "writer": ["file_read", "file_write", "memory_search"],
        "translator": ["memory_search"],
        "generic": ["web_search", "file_read", "memory_search"],
    }

    # Mapping nom → instance d'outil
    _TOOL_MAP: dict[str, object] = {
        "web_search": web_search_tool,
        "file_read": file_read_tool,
        "file_write": file_write_tool,
        "code_execute": code_execute_tool,
        "memory_search": memory_search_tool,
    }

    @classmethod
    def get_tool(cls, name: str) -> object:
        """Retourne un outil par son nom."""
        tool_obj = cls._TOOL_MAP.get(name)
        if tool_obj is None:
            raise ValueError(f"Outil inconnu: {name}")
        return tool_obj

    @classmethod
    def get_tools_for_type(cls, worker_type: str) -> list:
        """Retourne la liste d'outils pour un type de Worker."""
        tool_names = cls.WORKER_TOOLS.get(worker_type, cls.WORKER_TOOLS["generic"])
        return [cls._TOOL_MAP[name] for name in tool_names if name in cls._TOOL_MAP]

    @classmethod
    def list_tools(cls) -> list[str]:
        """Liste tous les outils disponibles."""
        return list(cls._TOOL_MAP.keys())

    @classmethod
    def list_worker_types(cls) -> list[str]:
        """Liste tous les types de workers supportés."""
        return list(cls.WORKER_TOOLS.keys())

    @classmethod
    def initialize(cls, mock_mode: bool = False,
                   memory: Optional[MemoryAgent] = None) -> None:
        """
        Initialise tous les outils.

        Args:
            mock_mode: Active le mode mock pour tous les outils
            memory: Référence vers MemoryAgent pour l'outil memory_search
        """
        set_mock_mode(mock_mode)
        set_memory_ref(memory)
