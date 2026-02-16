"""
Neo Core — Validation d'inputs
================================
Valide les entrées utilisateur aux points d'entrée critiques.

Usage :
    from neo_core.validation import validate_message, validate_task_description
    clean_msg = validate_message(user_input)
"""

from __future__ import annotations

# ─── Constantes ──────────────────────────────────────

MAX_MESSAGE_LENGTH = 10_000    # 10K caractères max par message
MAX_TASK_LENGTH = 5_000        # 5K caractères max par tâche
MIN_MESSAGE_LENGTH = 1         # Au moins 1 caractère


# ─── Exceptions ──────────────────────────────────────

class ValidationError(ValueError):
    """Erreur de validation d'input."""
    pass


class InputTooLongError(ValidationError):
    """Input dépasse la longueur maximale."""
    def __init__(self, length: int, max_length: int):
        self.length = length
        self.max_length = max_length
        super().__init__(
            f"Input trop long ({length} caractères, max {max_length})"
        )


class EmptyInputError(ValidationError):
    """Input vide ou whitespace uniquement."""
    def __init__(self):
        super().__init__("Input vide")


# ─── Fonctions de validation ────────────────────────

def validate_message(
    msg: str,
    max_length: int = MAX_MESSAGE_LENGTH,
    min_length: int = MIN_MESSAGE_LENGTH,
) -> str:
    """
    Valide et nettoie un message utilisateur.

    Args:
        msg: Le message brut
        max_length: Longueur maximale autorisée
        min_length: Longueur minimale requise

    Returns:
        Le message nettoyé (stripped)

    Raises:
        EmptyInputError: Si le message est vide après strip
        InputTooLongError: Si le message dépasse max_length
        ValidationError: Si le type n'est pas str
    """
    if not isinstance(msg, str):
        raise ValidationError(f"Type invalide: attendu str, reçu {type(msg).__name__}")

    cleaned = msg.strip()

    if len(cleaned) < min_length:
        raise EmptyInputError()

    if len(cleaned) > max_length:
        raise InputTooLongError(len(cleaned), max_length)

    return cleaned


def validate_task_description(
    desc: str,
    max_length: int = MAX_TASK_LENGTH,
) -> str:
    """
    Valide une description de tâche.

    Args:
        desc: La description brute
        max_length: Longueur maximale autorisée

    Returns:
        La description nettoyée

    Raises:
        EmptyInputError: Si la description est vide
        InputTooLongError: Si elle dépasse max_length
    """
    return validate_message(desc, max_length=max_length)


def validate_session_id(session_id: str) -> str:
    """
    Valide un identifiant de session (UUID-like).

    Args:
        session_id: L'identifiant brut

    Returns:
        L'identifiant nettoyé

    Raises:
        ValidationError: Si le format est invalide
    """
    if not isinstance(session_id, str):
        raise ValidationError(f"session_id doit être str, reçu {type(session_id).__name__}")

    cleaned = session_id.strip()

    if not cleaned:
        raise EmptyInputError()

    if len(cleaned) > 100:
        raise ValidationError(f"session_id trop long ({len(cleaned)} > 100)")

    # Autoriser UUID, UUID-hex, ou identifiants alphanumériques avec tirets
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', cleaned):
        raise ValidationError(f"session_id contient des caractères invalides: {cleaned[:20]}")

    return cleaned
