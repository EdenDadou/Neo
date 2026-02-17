"""
Neo Core — Gestionnaire OAuth Anthropic
=========================================
Gère l'authentification OAuth pour les tokens Claude Code (sk-ant-oat...).

Deux méthodes d'authentification (inspiré d'OpenClaw/pi-mono) :

Méthode 1 — Bearer + Beta Header :
    L'API Anthropic accepte les tokens OAuth via Bearer auth
    SI le header beta "anthropic-beta: oauth-2025-04-20" est présent.
    C'est la méthode la plus directe.

Méthode 2 — Conversion OAuth → API Key :
    Utilise le token OAuth pour créer une clé API permanente via
    POST https://console.anthropic.com/api/v1/claude_cli/create_api_key
    avec le token OAuth en Bearer auth.
    Le résultat est une clé sk-ant-api... classique.

Le flow OAuth Anthropic :
1. L'utilisateur obtient un access_token + refresh_token via `claude setup-token`
2. L'access_token (sk-ant-oat01-...) expire après ~8h
3. Le refresh_token (sk-ant-ort01-...) permet de renouveler l'access_token
4. On utilise Méthode 1 (Bearer + beta) ou Méthode 2 (conversion → API key)
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Client ID officiel du Claude Code CLI
CLAUDE_CODE_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
OAUTH_TOKEN_ENDPOINT = "https://console.anthropic.com/api/oauth/token"

# Endpoint pour convertir OAuth → API Key (méthode OpenClaw/pi-mono)
API_KEY_CREATE_ENDPOINT = "https://console.anthropic.com/api/v1/claude_cli/create_api_key"

# Header beta requis pour l'auth OAuth directe
OAUTH_BETA_HEADER = "oauth-2025-04-20"

# Fichier de stockage des credentials OAuth
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
CREDENTIALS_FILE = _DATA_DIR / ".oauth_credentials.json"


def _get_vault():
    """Retourne une instance du KeyVault (best-effort)."""
    try:
        from neo_core.security.vault import KeyVault
        vault = KeyVault(data_dir=_DATA_DIR)
        vault.initialize()
        return vault
    except Exception:
        return None


def is_oauth_token(key: str) -> bool:
    """Détecte si une clé est un token OAuth (sk-ant-oat...) vs API key (sk-ant-api...)."""
    if not key:
        return False
    return key.startswith("sk-ant-oat")


def is_refresh_token(key: str) -> bool:
    """Détecte si c'est un refresh token (sk-ant-ort...)."""
    if not key:
        return False
    return key.startswith("sk-ant-ort")


def load_credentials() -> dict:
    """
    Charge les credentials OAuth.

    Priorité :
    1. KeyVault (chiffré AES) pour access_token et refresh_token
    2. Fallback fichier JSON legacy (.oauth_credentials.json)
    """
    # 1. Essayer le vault
    vault = _get_vault()
    if vault:
        try:
            access = vault.retrieve("oauth_access_token")
            refresh = vault.retrieve("oauth_refresh_token")
            meta_raw = vault.retrieve("oauth_meta")
            vault.close()

            if access:
                meta = json.loads(meta_raw) if meta_raw else {}
                return {
                    "access_token": access,
                    "refresh_token": refresh or "",
                    "expires_at": meta.get("expires_at", 0),
                    "api_key": meta.get("api_key", ""),
                }
        except Exception as e:
            logger.debug("Vault read for OAuth failed: %s", e)
            try:
                vault.close()
            except Exception:
                pass

    # 2. Fallback : fichier JSON legacy
    if CREDENTIALS_FILE.exists():
        try:
            with open(CREDENTIALS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_credentials(access_token: str, refresh_token: str, expires_at: float,
                     api_key: str = ""):
    """
    Sauvegarde les credentials OAuth.

    Les tokens sont chiffrés dans le KeyVault (AES).
    Un fichier JSON legacy est aussi écrit en fallback.
    """
    # 1. Vault (chiffré)
    vault = _get_vault()
    if vault:
        try:
            vault.store("oauth_access_token", access_token)
            if refresh_token:
                vault.store("oauth_refresh_token", refresh_token)
            meta = {"expires_at": expires_at}
            if api_key:
                meta["api_key"] = api_key
            vault.store("oauth_meta", json.dumps(meta))
            vault.close()
            logger.debug("OAuth credentials saved to vault (encrypted)")
        except Exception as e:
            logger.debug("Vault write for OAuth failed: %s — fallback to JSON", e)
            try:
                vault.close()
            except Exception:
                pass

    # 2. Fichier JSON legacy (backward compat + fallback)
    CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    creds = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
    }
    if api_key:
        creds["api_key"] = api_key
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(creds, f, indent=2)
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


def convert_oauth_to_api_key(access_token: str) -> Optional[str]:
    """
    Méthode 2 — Convertit un token OAuth en clé API permanente.

    Appelle POST https://console.anthropic.com/api/v1/claude_cli/create_api_key
    avec le token OAuth en Bearer auth.

    Retourne la clé API (sk-ant-api...) ou None si la conversion échoue.
    """
    try:
        response = httpx.post(
            API_KEY_CREATE_ENDPOINT,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={},
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            # Réponse attendue: {"type": "success", "key": {"raw_key": "sk-ant-api03-..."}}
            if data.get("type") == "success":
                key_data = data.get("key", {})
                raw_key = key_data.get("raw_key")
                if raw_key:
                    # Sauvegarder la clé dans les credentials
                    creds = load_credentials()
                    creds["api_key"] = raw_key
                    with open(CREDENTIALS_FILE, "w") as f:
                        json.dump(creds, f, indent=2)
                    os.chmod(CREDENTIALS_FILE, 0o600)
                    return raw_key

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


def get_api_key_from_oauth() -> Optional[str]:
    """
    Retourne une clé API permanente depuis les credentials OAuth.

    1. Vérifie si une clé API convertie existe déjà
    2. Sinon, obtient un token valide et tente la conversion
    3. Retourne la clé API ou None
    """
    creds = load_credentials()

    # Vérifier si on a déjà une clé API convertie
    existing_key = creds.get("api_key")
    if existing_key and existing_key.startswith("sk-ant-api"):
        return existing_key

    # Pas de clé API → tenter la conversion
    access_token = get_valid_access_token()
    if access_token:
        api_key = convert_oauth_to_api_key(access_token)
        if api_key:
            return api_key

    return None


def get_best_auth() -> dict:
    """
    Détermine la meilleure méthode d'authentification disponible.

    Retourne un dict avec :
    - method: "api_key" | "oauth_bearer" | "converted_api_key" | None
    - key: la clé ou le token à utiliser
    - beta_header: le header beta si nécessaire (pour oauth_bearer)
    - message: description de la méthode utilisée

    Ordre de priorité :
    1. Clé API convertie (déjà en cache) → utilise x-api-key classique
    2. Token OAuth valide → Bearer + beta header
    3. Conversion OAuth → API Key (appel réseau)
    4. Aucune auth disponible
    """
    creds = load_credentials()

    # 1. Clé API déjà convertie en cache ?
    existing_key = creds.get("api_key")
    if existing_key and existing_key.startswith("sk-ant-api"):
        return {
            "method": "converted_api_key",
            "key": existing_key,
            "beta_header": None,
            "message": "Clé API convertie depuis OAuth (cache)",
        }

    # 2. Token OAuth valide → Bearer direct avec beta header
    access_token = get_valid_access_token()
    if access_token:
        return {
            "method": "oauth_bearer",
            "key": access_token,
            "beta_header": OAUTH_BETA_HEADER,
            "message": "Token OAuth Bearer + beta header",
        }

    # 3. Pas de token valide → tenter refresh puis conversion
    refresh_token = creds.get("refresh_token")
    if refresh_token:
        result = refresh_access_token(refresh_token)
        if result:
            # Token rafraîchi → essayer Bearer d'abord
            return {
                "method": "oauth_bearer",
                "key": result["access_token"],
                "beta_header": OAUTH_BETA_HEADER,
                "message": "Token OAuth rafraîchi + beta header",
            }

    return {
        "method": None,
        "key": None,
        "beta_header": None,
        "message": "Aucune authentification OAuth disponible",
    }


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
        # Access token → stocker directement
        claude_creds = _load_claude_code_credentials()
        refresh = claude_creds.get("refreshToken") if claude_creds else None

        # Stocker avec une expiration de 8h par défaut
        expires_at = time.time() + 28800
        save_credentials(token, refresh or "", expires_at)

        # Tenter la conversion en clé API (méthode 2 — OpenClaw)
        api_key = convert_oauth_to_api_key(token)

        if api_key:
            return {
                "success": True,
                "type": "oauth_converted",
                "access_token": token,
                "api_key": api_key,
                "message": f"Token OAuth converti en clé API : {api_key[:12]}...{api_key[-4:]}",
            }
        elif refresh:
            return {
                "success": True,
                "type": "oauth_with_refresh",
                "access_token": token,
                "message": "Token OAuth stocké avec refresh (Bearer + beta header)",
            }
        else:
            return {
                "success": True,
                "type": "oauth_bearer",
                "access_token": token,
                "message": "Token OAuth stocké (Bearer + beta header, expirera dans ~8h)",
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
