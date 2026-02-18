"""
Tests Stage 13 — API REST (FastAPI)
=====================================
Vérifie les endpoints REST, WebSocket, auth et errors.

30+ tests au total.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from neo_core.vox.api.schemas import (
    ChatRequest,
    ChatResponse,
    StatusResponse,
    SessionInfo,
    HistoryTurn,
    ErrorResponse,
)
from neo_core.vox.api.server import create_app, NeoCore, neo_core


# ─── Fixtures ────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_neo_core():
    """Reset NeoCore singleton between tests."""
    neo_core.reset()
    yield
    neo_core.reset()


@pytest.fixture
def app():
    """Crée une app FastAPI de test avec NeoCore initialisé via lifespan."""
    test_app = create_app()
    return test_app


@pytest.fixture
def client(app):
    """Client de test HTTP (lifespan triggers initialize)."""
    with TestClient(app) as c:
        yield c


# ══════════════════════════════════════════════════════
# 1. TestSchemas — Modèles Pydantic (~6 tests)
# ══════════════════════════════════════════════════════

class TestSchemas:
    """Tests des schémas Pydantic."""

    def test_chat_request_valid(self):
        """ChatRequest avec message valide."""
        req = ChatRequest(message="Hello Neo")
        assert req.message == "Hello Neo"
        assert req.session_id is None

    def test_chat_request_with_session(self):
        """ChatRequest avec session_id."""
        req = ChatRequest(message="Hello", session_id="abc-123")
        assert req.session_id == "abc-123"

    def test_chat_response_valid(self):
        """ChatResponse sérialisable."""
        resp = ChatResponse(
            response="Bonjour !",
            session_id="sess-1",
            turn_number=1,
            timestamp="2025-01-01T00:00:00",
        )
        assert resp.response == "Bonjour !"

    def test_status_response(self):
        """StatusResponse complète."""
        status = StatusResponse(
            status="ready",
            core_name="Neo",
            uptime_seconds=42.0,
            agents={"vox": {"connected": True}},
            guardian_mode=False,
        )
        assert status.status == "ready"

    def test_session_info(self):
        """SessionInfo sérialisable."""
        info = SessionInfo(
            session_id="s1",
            user_name="Eden",
            created_at="2025-01-01",
            updated_at="2025-01-01",
            message_count=5,
        )
        assert info.message_count == 5

    def test_history_turn(self):
        """HistoryTurn sérialisable."""
        turn = HistoryTurn(
            role="human",
            content="Bonjour",
            timestamp="2025-01-01",
            turn_number=1,
        )
        assert turn.role == "human"


# ══════════════════════════════════════════════════════
# 2. TestHealthEndpoint — Health check (~3 tests)
# ══════════════════════════════════════════════════════

class TestHealthEndpoint:
    """Tests du health check."""

    def test_health_returns_200(self, client):
        """GET /health retourne 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status(self, client):
        """GET /health contient status=ok."""
        data = client.get("/health").json()
        assert data["status"] in ("ok", "healthy", "degraded")

    def test_health_has_timestamp(self, client):
        """GET /health contient un timestamp."""
        data = client.get("/health").json()
        assert "timestamp" in data


# ══════════════════════════════════════════════════════
# 3. TestStatusEndpoint — Status système (~3 tests)
# ══════════════════════════════════════════════════════

class TestStatusEndpoint:
    """Tests du status endpoint."""

    def test_status_returns_200(self, client):
        """GET /status retourne 200."""
        response = client.get("/status")
        assert response.status_code == 200

    def test_status_has_agents(self, client):
        """GET /status contient la liste des agents."""
        data = client.get("/status").json()
        assert "agents" in data
        assert "vox" in data["agents"]
        assert "brain" in data["agents"]
        assert "memory" in data["agents"]

    def test_status_has_uptime(self, client):
        """GET /status contient l'uptime."""
        data = client.get("/status").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


# ══════════════════════════════════════════════════════
# 4. TestChatEndpoint — Chat (~6 tests)
# ══════════════════════════════════════════════════════

class TestChatEndpoint:
    """Tests du chat endpoint."""

    def test_chat_returns_200(self, client):
        """POST /chat avec message valide → 200."""
        with patch.object(neo_core.vox, "process_message", new_callable=AsyncMock, return_value="Bonjour !"):
            response = client.post("/chat", json={"message": "Bonjour Neo"})
            assert response.status_code == 200

    def test_chat_has_response(self, client):
        """POST /chat retourne une réponse."""
        with patch.object(neo_core.vox, "process_message", new_callable=AsyncMock, return_value="Hello back!"):
            data = client.post("/chat", json={"message": "Hello"}).json()
            assert "response" in data
            assert len(data["response"]) > 0

    def test_chat_has_session_id(self, client):
        """POST /chat retourne un session_id."""
        with patch.object(neo_core.vox, "process_message", new_callable=AsyncMock, return_value="Hi"):
            data = client.post("/chat", json={"message": "Hello"}).json()
            assert "session_id" in data
            assert len(data["session_id"]) > 0

    def test_chat_has_timestamp(self, client):
        """POST /chat retourne un timestamp."""
        with patch.object(neo_core.vox, "process_message", new_callable=AsyncMock, return_value="Hi"):
            data = client.post("/chat", json={"message": "Hello"}).json()
            assert "timestamp" in data

    def test_chat_empty_message_rejected(self, client):
        """POST /chat avec message vide → erreur."""
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 422  # Validation error

    def test_chat_missing_message_rejected(self, client):
        """POST /chat sans message → erreur."""
        response = client.post("/chat", json={})
        assert response.status_code == 422


# ══════════════════════════════════════════════════════
# 5. TestSessionsEndpoint — Sessions (~4 tests)
# ══════════════════════════════════════════════════════

class TestSessionsEndpoint:
    """Tests du sessions endpoint."""

    def test_sessions_returns_200(self, client):
        """GET /sessions retourne 200."""
        response = client.get("/sessions")
        assert response.status_code == 200

    def test_sessions_returns_list(self, client):
        """GET /sessions retourne une liste."""
        data = client.get("/sessions").json()
        assert isinstance(data, list)

    def test_sessions_has_initial_session(self, client):
        """GET /sessions contient au moins la session initiale."""
        data = client.get("/sessions").json()
        assert len(data) >= 1

    def test_sessions_with_limit(self, client):
        """GET /sessions?limit=1 respecte la limite."""
        data = client.get("/sessions?limit=1").json()
        assert len(data) <= 1


# ══════════════════════════════════════════════════════
# 6. TestHistoryEndpoint — Historique (~4 tests)
# ══════════════════════════════════════════════════════

class TestHistoryEndpoint:
    """Tests du history endpoint."""

    def test_history_returns_200(self, client):
        """GET /sessions/{id}/history retourne 200."""
        # Use session from startup
        sessions = client.get("/sessions").json()
        if sessions:
            session_id = sessions[0]["session_id"]
            response = client.get(f"/sessions/{session_id}/history")
            assert response.status_code == 200

    def test_history_returns_list(self, client):
        """L'historique retourne une liste."""
        sessions = client.get("/sessions").json()
        if sessions:
            session_id = sessions[0]["session_id"]
            data = client.get(f"/sessions/{session_id}/history").json()
            assert isinstance(data, list)

    def test_history_pagination(self, client):
        """Pagination de l'historique."""
        sessions = client.get("/sessions").json()
        if sessions:
            session_id = sessions[0]["session_id"]
            data = client.get(f"/sessions/{session_id}/history?limit=2").json()
            assert len(data) <= 2

    def test_history_nonexistent_session(self, client):
        """Historique d'une session inexistante → liste vide."""
        data = client.get("/sessions/nonexistent/history").json()
        assert isinstance(data, list)
        assert len(data) == 0


# ══════════════════════════════════════════════════════
# 7. TestPersonaEndpoint — Persona (~2 tests)
# ══════════════════════════════════════════════════════

class TestPersonaEndpoint:
    """Tests du persona endpoint."""

    def test_persona_returns_200(self, client):
        """GET /persona retourne 200."""
        response = client.get("/persona")
        assert response.status_code == 200

    def test_persona_has_data(self, client):
        """GET /persona contient persona et user_profile."""
        data = client.get("/persona").json()
        assert "persona" in data
        assert "user_profile" in data


# ══════════════════════════════════════════════════════
# 8. TestWebSocket — WebSocket chat (~3 tests)
# ══════════════════════════════════════════════════════

class TestWebSocket:
    """Tests du WebSocket."""

    def test_websocket_connect(self, client):
        """WebSocket se connecte."""
        neo_core.config = None
        with client.websocket_connect("/ws/chat") as ws:
            assert ws is not None

    def test_websocket_send_receive(self, client):
        """WebSocket envoie et reçoit un message."""
        neo_core.config = None
        with patch.object(neo_core.vox, "process_message", new_callable=AsyncMock, return_value="WS response"):
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_json({"message": "Hello WS"})
                data = ws.receive_json()
                assert "response" in data
                assert "session_id" in data

    def test_websocket_empty_message(self, client):
        """WebSocket rejette un message vide."""
        neo_core.config = None
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": ""})
            data = ws.receive_json()
            assert "error" in data


# ══════════════════════════════════════════════════════
# 9. TestMiddleware — Auth et Rate limiting (~4 tests)
# ══════════════════════════════════════════════════════

class TestMiddleware:
    """Tests du middleware."""

    def test_api_key_middleware_exists(self):
        """Le middleware APIKey existe."""
        from neo_core.vox.api.middleware import APIKeyMiddleware
        assert APIKeyMiddleware is not None

    def test_rate_limit_middleware_exists(self):
        """Le middleware RateLimit existe."""
        from neo_core.vox.api.middleware import RateLimitMiddleware
        assert RateLimitMiddleware is not None

    def test_api_key_rejects_invalid(self):
        """Avec API key configurée, une mauvaise clé est rejetée."""
        from neo_core.vox.api.middleware import APIKeyMiddleware

        app = create_app()
        app.add_middleware(APIKeyMiddleware, api_key="secret-key-123")
        with TestClient(app) as c:
            response = c.get("/health")  # health is exempt
            assert response.status_code == 200
            response = c.get("/status", headers={"X-Neo-Key": "wrong-key"})
            assert response.status_code == 401

    def test_api_key_accepts_valid(self):
        """Avec API key configurée, la bonne clé passe."""
        from neo_core.vox.api.middleware import APIKeyMiddleware

        # Reset before creating new app
        neo_core.reset()
        app = create_app()
        app.add_middleware(APIKeyMiddleware, api_key="secret-key-123")
        with TestClient(app) as c:
            response = c.get("/status", headers={"X-Neo-Key": "secret-key-123"})
            assert response.status_code == 200


# ══════════════════════════════════════════════════════
# 10. TestCLIIntegration — Commande neo api (~2 tests)
# ══════════════════════════════════════════════════════

class TestCLIIntegration:
    """Tests de la commande CLI."""

    def test_cli_has_api_command(self):
        """Le CLI a la commande 'api'."""
        import neo_core.vox.cli
        import inspect
        source = inspect.getsource(neo_core.vox.cli.main)
        assert "api" in source

    def test_create_app_returns_fastapi(self):
        """create_app() retourne une app FastAPI."""
        from fastapi import FastAPI
        neo_core.reset()
        app = create_app()
        assert isinstance(app, FastAPI)
