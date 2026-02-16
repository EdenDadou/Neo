"""
Neo Core — Gestionnaire OAuth Anthropic
=========================================
Gère l'authentification OAuth pour les tokens Claude Code (sk-ant-oat...).

Le flow OAuth Anthropic :
1. L'utilisateur obtient un access_token + refresh_token via `claude setup-token`
2. L'access_token (sk-ant-oat01-...) expire après ~8h
3. Le refresh_token (sk-ant-ort01-...) permet de renouveler l'access_token
4. L'API accepte le Bearer token sur api.anthropic.com/v1/messages

Ce module gère le stockage et le rafraîchissement automatique des tokens.
"""

import json
import time
from pathlib import Path
from typing import Optional

import httpx

# Client ID officiel du Claude Code CLI
CLAUDE_CODE_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
OAUTH_TOKEN_ENDPOINT = "https://console.anthropic.com/api/oauth/token"

# Fichier de stockage des credentials OAuth
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = _PROJECT_ROOT / "data" / ".oauth_credentials.json"


def is_oauth_token(key: str) -> bool:
    """Détecte si une clé est un token OAuth (sk-ant-oat...) vs API key (sk-ant-api...)."""
    return key.startswith("sk-ant-oat")


def is_refresh_token(key: str) -> bool:
    """Détecte si c'est un refresh token (sk-ant-ort...)."""
    return key.startswith("sk-ant-ort")


def load_credentials() -> dict:
    """Charge les credentials OAuth depuis le fichier."""
    if CREDENTIALS_FILE.exists():
        try:
            with open(CREDENTIALS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_credentials(access_token: str, refresh_token: str, expires_at: float):
    """Sauvegarde les credentials OAuth."""
    CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    creds = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
    }
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(creds, f, indent=2)
    # Sécurise le fichier
    import os
    os.chmod(CREDENTIALS_FILE, 0o600)


def is_token_expired(creds: dict) -> bool:
    """Vérifie si l'access token a expiré (avec 5 min de marge)."""
    expires_at = creds.get("expires_at", 0)
    return time.time() > (expires_at - 300)  # 5 min de marge


def refresh_access_token(refresh_token: str) -> Optional[dict]:
    """
    Rafraîchit l'access token via le refresh token.

    Retourne un dict avec access_token, refresh_token, expires_at
    ou None si le refresh échoue.
    """
    try:
        response = httpx.post(
            OAUTH_TOKEN_ENDPOINT,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLAUDE_CODE_CLIENT_ID,
            },
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            new_access = data.get("access_token")
            # Le refresh token peut aussi être renouvelé
            new_refresh = data.get("refresh_token", refresh_token)
            expires_in = data.get("expires_in", 28800)  # 8h par défaut
            expires_at = time.time() + expires_in

            if new_access:
                save_credentials(new_access, new_refresh, expires_at)
                return {
                    "access_token": new_access,
                    "refresh_token": new_refresh,
                    "expires_at": expires_at,
                }

        return None
    except Exception:
        return None


def get_valid_access_token() -> Optional[str]:
    """
    Retourne un access token valide.

    1. Charge les credentials
    2. Si le token est expiré, tente un refresh
    3. Retourne le token ou None si impossible
    """
    creds = load_credentials()

    if not creds:
        return None

    access_token = creds.get("access_token")
    refresh_token = creds.get("refresh_token")

    # Si le token n'est pas expiré, on le retourne
    if access_token and not is_token_expired(creds):
        return access_token

    # Token expiré → tenter un refresh
    if refresh_token:
        result = refresh_access_token(refresh_token)
        if result:
            return result["access_token"]

    return None


def setup_oauth_from_token(token: str) -> dict:
    """
    Configure l'OAuth à partir d'un token fourni par l'utilisateur.

    Si c'est un access token (sk-ant-oat...) : on le stocke directement
    Si c'est un refresh token (sk-ant-ort...) : on l'utilise pour obtenir un access token

    Retourne un dict avec le statut.
    """
    if is_refresh_token(token):
        # Refresh token → obtenir un access token
        result = refresh_access_token(token)
        if result:
            return {
                "success": True,
                "type": "refresh_token",
                "access_token": result["access_token"],
                "message": "Token rafraîchi avec succès",
            }
        else:
            return {
                "success": False,
                "type": "refresh_token",
                "message": "Impossible de rafraîchir le token. Vérifiez qu'il est valide.",
            }

    elif is_oauth_token(token):
        # Access token → stocker directement (sans refresh, expiration inconnue)
        # On essaie d'abord de récupérer les credentials Claude Code existantes
        claude_creds = _load_claude_code_credentials()
        refresh = claude_creds.get("refreshToken") if claude_creds else None

        # Stocker avec une expiration de 8h par défaut
        expires_at = time.time() + 28800
        save_credentials(token, refresh or "", expires_at)

        if refresh:
            return {
                "success": True,
                "type": "oauth_with_refresh",
                "access_token": token,
                "message": "Token OAuth stocké avec refresh token de Claude Code",
            }
        else:
            return {
                "success": True,
                "type": "oauth_no_refresh",
                "access_token": token,
                "message": "Token OAuth stocké (sans refresh — expirera dans ~8h)",
            }

    else:
        return {
            "success": False,
            "type": "unknown",
            "message": "Format de token non reconnu",
        }


def _load_claude_code_credentials() -> Optional[dict]:
    """
    Tente de charger les credentials depuis le fichier Claude Code CLI.
    (~/.claude/.credentials.json)
    """
    creds_path = Path.home() / ".claude" / ".credentials.json"
    if creds_path.exists():
        try:
            with open(creds_path) as f:
                data = json.load(f)
            oauth = data.get("claudeAiOauth", {})
            if oauth:
                return oauth
        except (json.JSONDecodeError, IOError):
            pass
    return None


def import_claude_code_credentials() -> Optional[dict]:
    """
    Importe automatiquement les credentials depuis Claude Code CLI.
    Utile si l'utilisateur a déjà `claude login` configuré.

    Retourne un dict avec access_token et refresh_token ou None.
    """
    claude_creds = _load_claude_code_credentials()
    if not claude_creds:
        return None

    access = claude_creds.get("accessToken")
    refresh = claude_creds.get("refreshToken")
    expires_at = claude_creds.get("expiresAt", 0)

    if not access:
        return None

    # Convertir expiresAt (peut être en ms ou en s)
    if expires_at > 1e12:  # Probablement en millisecondes
        expires_at = expires_at / 1000

    save_credentials(access, refresh or "", expires_at)

    # Si expiré, tenter un refresh immédiat
    if time.time() > expires_at and refresh:
        result = refresh_access_token(refresh)
        if result:
            return result

    return {
        "access_token": access,
        "refresh_token": refresh or "",
        "expires_at": expires_at,
    }
