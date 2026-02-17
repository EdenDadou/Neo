"""
Tests — OAuth (authentification Anthropic)
=============================================
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from neo_core.oauth import (
    is_oauth_token,
    is_refresh_token,
    is_token_expired,
    load_credentials,
    save_credentials,
    get_valid_access_token,
    get_best_auth,
    setup_oauth_from_token,
    CREDENTIALS_FILE,
    OAUTH_BETA_HEADER,
)


class TestTokenDetection:
    def test_oauth_token(self):
        assert is_oauth_token("sk-ant-oat01-abc") is True

    def test_api_key_not_oauth(self):
        assert is_oauth_token("sk-ant-api03-xyz") is False

    def test_empty(self):
        assert is_oauth_token("") is False
        assert is_oauth_token(None) is False

    def test_refresh_token(self):
        assert is_refresh_token("sk-ant-ort01-abc") is True

    def test_not_refresh(self):
        assert is_refresh_token("sk-ant-oat01-abc") is False
        assert is_refresh_token("") is False


class TestTokenExpiration:
    def test_not_expired(self):
        creds = {"expires_at": time.time() + 3600}
        assert is_token_expired(creds) is False

    def test_expired(self):
        creds = {"expires_at": time.time() - 100}
        assert is_token_expired(creds) is True

    def test_expires_within_margin(self):
        """Expire dans 4 minutes (< 5 min de marge) → considéré expiré."""
        creds = {"expires_at": time.time() + 240}
        assert is_token_expired(creds) is True

    def test_no_expires_at(self):
        assert is_token_expired({}) is True


class TestCredentialsPersistence:
    def test_save_and_load(self, tmp_path):
        """save_credentials crée un marqueur (pas de tokens en clair)."""
        creds_file = tmp_path / "data" / ".oauth_credentials.json"

        with patch("neo_core.oauth.CREDENTIALS_FILE", creds_file):
            with patch("neo_core.oauth._get_vault", return_value=None):
                save_credentials("access-123", "refresh-456", 99999.0, "sk-ant-api-key")

                assert creds_file.exists()

                # Le fichier JSON ne contient plus les tokens en clair (v0.8.3)
                import json
                raw = json.loads(creds_file.read_text())
                assert raw.get("has_credentials") is True
                assert "access_token" not in raw  # Pas de tokens en clair!
                assert raw.get("expires_at") == 99999.0

                # Sans vault, load_credentials retourne dict vide (tokens absents du JSON)
                data = load_credentials()
                assert data == {} or data.get("has_credentials") is True

    def test_load_empty(self, tmp_path):
        """Sans fichier → dict vide."""
        with patch("neo_core.oauth.CREDENTIALS_FILE", tmp_path / "nonexistent.json"):
            with patch("neo_core.oauth._get_vault", return_value=None):
                assert load_credentials() == {}

    def test_save_with_vault(self, tmp_path):
        """save_credentials utilise le vault si disponible."""
        mock_vault = MagicMock()
        creds_file = tmp_path / "data" / ".oauth_credentials.json"

        with patch("neo_core.oauth._get_vault", return_value=mock_vault):
            with patch("neo_core.oauth.CREDENTIALS_FILE", creds_file):
                save_credentials("tok-abc", "ref-xyz", 12345.0)

        mock_vault.store.assert_any_call("oauth_access_token", "tok-abc")
        mock_vault.store.assert_any_call("oauth_refresh_token", "ref-xyz")
        mock_vault.close.assert_called()

    def test_load_from_vault(self):
        """load_credentials lit du vault en priorité."""
        mock_vault = MagicMock()
        mock_vault.retrieve.side_effect = lambda key: {
            "oauth_access_token": "vault-access",
            "oauth_refresh_token": "vault-refresh",
            "oauth_meta": json.dumps({"expires_at": 999, "api_key": "sk-vault"}),
        }.get(key)

        with patch("neo_core.oauth._get_vault", return_value=mock_vault):
            creds = load_credentials()

        assert creds["access_token"] == "vault-access"
        assert creds["refresh_token"] == "vault-refresh"
        assert creds["api_key"] == "sk-vault"


class TestGetValidAccessToken:
    def test_valid_token(self):
        creds = {
            "access_token": "valid-tok",
            "expires_at": time.time() + 3600,
        }
        with patch("neo_core.oauth.load_credentials", return_value=creds):
            assert get_valid_access_token() == "valid-tok"

    def test_expired_triggers_refresh(self):
        creds = {
            "access_token": "expired-tok",
            "refresh_token": "ref-tok",
            "expires_at": time.time() - 1000,
        }
        with patch("neo_core.oauth.load_credentials", return_value=creds):
            with patch("neo_core.oauth.refresh_access_token", return_value={
                "access_token": "new-tok",
            }):
                assert get_valid_access_token() == "new-tok"

    def test_no_credentials(self):
        with patch("neo_core.oauth.load_credentials", return_value={}):
            assert get_valid_access_token() is None


class TestGetBestAuth:
    def test_converted_api_key(self):
        creds = {"api_key": "sk-ant-api03-converted"}
        with patch("neo_core.oauth.load_credentials", return_value=creds):
            result = get_best_auth()
            assert result["method"] == "converted_api_key"
            assert result["key"] == "sk-ant-api03-converted"

    def test_oauth_bearer(self):
        creds = {
            "access_token": "sk-ant-oat-valid",
            "expires_at": time.time() + 3600,
        }
        with patch("neo_core.oauth.load_credentials", return_value=creds):
            result = get_best_auth()
            assert result["method"] == "oauth_bearer"
            assert result["beta_header"] == OAUTH_BETA_HEADER

    def test_no_auth(self):
        with patch("neo_core.oauth.load_credentials", return_value={}):
            result = get_best_auth()
            assert result["method"] is None


class TestSetupOAuthFromToken:
    def test_access_token_stored(self):
        with patch("neo_core.oauth.save_credentials"):
            with patch("neo_core.oauth._load_claude_code_credentials", return_value=None):
                with patch("neo_core.oauth.convert_oauth_to_api_key", return_value=None):
                    result = setup_oauth_from_token("sk-ant-oat01-mytoken")
                    assert result["success"] is True
                    assert result["type"] == "oauth_bearer"

    def test_unknown_format(self):
        result = setup_oauth_from_token("random-string")
        assert result["success"] is False
        assert result["type"] == "unknown"

    def test_refresh_token_triggers_refresh(self):
        with patch("neo_core.oauth.refresh_access_token", return_value={
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_at": time.time() + 3600,
        }):
            result = setup_oauth_from_token("sk-ant-ort01-myrefresh")
            assert result["success"] is True
            assert result["type"] == "refresh_token"

    def test_refresh_token_fails(self):
        with patch("neo_core.oauth.refresh_access_token", return_value=None):
            result = setup_oauth_from_token("sk-ant-ort01-bad")
            assert result["success"] is False
