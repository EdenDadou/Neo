"""
Neo Core — Input Sanitizer
=============================
Valide et nettoie les entrées utilisateur pour prévenir les injections.

Détecte :
- Injection de prompts (tentatives de bypass system prompt)
- Injection SQL (patterns dangereux)
- XSS (balises HTML/script)
- Path traversal (../ ou accès fichiers système)
- Entrées excessivement longues
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SanitizeResult:
    """Résultat de la sanitisation."""
    is_safe: bool
    cleaned: str
    threats: list[str] = field(default_factory=list)
    severity: str = "none"  # none, low, medium, high


# ─── Patterns dangereux ──────────────────────────────

_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|in)\s+", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"<\|?(system|assistant|im_start|im_end)\|?>", re.I),
    re.compile(r"\[INST\]|\[/INST\]|\[SYS\]|\[/SYS\]", re.I),
    re.compile(r"forget\s+(everything|all|your\s+(instructions|rules))", re.I),
    re.compile(r"override\s+(your|the|all)\s+(safety|rules|restrictions)", re.I),
    # v0.8.4 — Roleplay / impersonation patterns
    re.compile(r"act\s+as\s+(a|an|if|though)\b", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be|you're)\s+", re.I),
    re.compile(r"you\s+must\s+now\s+", re.I),
    re.compile(r"(new|updated|revised)\s+(instructions?|rules?|prompt)\s*:", re.I),
    re.compile(r"disregard\s+(all|any|the|your)\s+", re.I),
    re.compile(r"do\s+not\s+follow\s+(your|the|any)\s+(instructions|rules|guidelines)", re.I),
    # Encodage / obfuscation
    re.compile(r"base64[:\s]", re.I),
    re.compile(r"eval\s*\(", re.I),
    # Extraction de prompt
    re.compile(r"(repeat|show|display|print|output)\s+(your|the|system)\s+(prompt|instructions)", re.I),
    re.compile(r"what\s+(are|is)\s+your\s+(system\s+)?prompt", re.I),
]

_SQL_INJECTION_PATTERNS = [
    re.compile(r"('\s*(OR|AND)\s*'?\s*\d+\s*=\s*\d+)", re.I),
    re.compile(r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE)\s+", re.I),
    re.compile(r"UNION\s+(ALL\s+)?SELECT\s+", re.I),
    re.compile(r"--\s*$", re.M),
]

_XSS_PATTERNS = [
    re.compile(r"<\s*script", re.I),
    re.compile(r"javascript\s*:", re.I),
    re.compile(r"on(load|error|click|mouseover)\s*=", re.I),
    re.compile(r"<\s*iframe", re.I),
    re.compile(r"<\s*object", re.I),
    re.compile(r"<\s*embed", re.I),
]

_PATH_TRAVERSAL_PATTERNS = [
    re.compile(r"\.\./"),
    re.compile(r"\.\.\\"),
    re.compile(r"/etc/(passwd|shadow|hosts)", re.I),
    re.compile(r"(C:\\|/)(Windows|System32)", re.I),
]


class Sanitizer:
    """
    Sanitiseur d'entrées pour Neo Core.

    Analyse les messages utilisateur et détecte les tentatives
    d'injection avant qu'elles n'atteignent les agents LLM.
    """

    def __init__(self, max_length: int = 10_000, strict: bool = False):
        """
        Args:
            max_length: Longueur maximale d'un message (caractères).
            strict: Si True, rejette au moindre soupçon. Si False, nettoie et laisse passer.
        """
        self.max_length = max_length
        self.strict = strict

    def sanitize(self, text: str) -> SanitizeResult:
        """
        Analyse et nettoie un texte.

        Retourne un SanitizeResult avec :
        - is_safe : True si aucune menace détectée
        - cleaned : texte nettoyé
        - threats : liste des menaces détectées
        - severity : niveau de sévérité global
        """
        if not text or not text.strip():
            return SanitizeResult(is_safe=True, cleaned="", severity="none")

        threats: list[str] = []
        cleaned = text

        # 1. Longueur excessive
        if len(text) > self.max_length:
            threats.append(f"input_too_long:{len(text)}")
            cleaned = cleaned[:self.max_length]

        # 2. Prompt injection
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(text):
                threats.append(f"prompt_injection:{pattern.pattern[:40]}")

        # 3. SQL injection
        for pattern in _SQL_INJECTION_PATTERNS:
            if pattern.search(text):
                threats.append(f"sql_injection:{pattern.pattern[:40]}")

        # 4. XSS
        for pattern in _XSS_PATTERNS:
            match = pattern.search(text)
            if match:
                threats.append(f"xss:{match.group()[:30]}")
                # Nettoyer les balises dangereuses
                cleaned = pattern.sub("[FILTERED]", cleaned)

        # 5. Path traversal
        for pattern in _PATH_TRAVERSAL_PATTERNS:
            if pattern.search(text):
                threats.append(f"path_traversal:{pattern.pattern[:30]}")
                cleaned = pattern.sub("[BLOCKED]", cleaned)

        # Calculer la sévérité
        severity = "none"
        if threats:
            if any("prompt_injection" in t for t in threats):
                severity = "high"
            elif any("sql_injection" in t for t in threats):
                severity = "high"
            elif any("xss" in t for t in threats):
                severity = "medium"
            elif any("path_traversal" in t for t in threats):
                severity = "medium"
            else:
                severity = "low"

        is_safe = len(threats) == 0
        if not is_safe:
            logger.warning(
                "Sanitizer detected %d threat(s): %s [severity=%s]",
                len(threats), ", ".join(threats), severity,
            )

        return SanitizeResult(
            is_safe=is_safe,
            cleaned=cleaned if not self.strict else (cleaned if is_safe else ""),
            threats=threats,
            severity=severity,
        )

    def is_safe(self, text: str) -> bool:
        """Raccourci : True si le texte ne contient aucune menace."""
        return self.sanitize(text).is_safe
