"""
Neo Core API Server — FastAPI Application
==========================================

Creates and manages the FastAPI application.
Uses CoreRegistry for a single shared instance of Vox/Brain/Memory.
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class NeoCore:
    """Wrapper API autour du CoreRegistry partagé."""

    def __init__(self):
        import asyncio
        self.vox = None
        self.config = None
        self._lock = asyncio.Lock()
        self._start_time = time.time()
        self._initialized = False

    def initialize(self, config=None):
        """Bootstrap via le CoreRegistry (instance unique partagée)."""
        if self._initialized:
            return

        from neo_core.core.registry import core_registry

        self.vox = core_registry.get_vox()
        self.config = core_registry.get_config()
        self._initialized = True
        logger.info("NeoCore API initialized (via CoreRegistry)")

    def reset(self):
        """Reset state (for testing)."""
        self.vox = None
        self.config = None
        self._initialized = False
        self._start_time = time.time()

    @property
    def uptime(self):
        """Return uptime in seconds."""
        return time.time() - self._start_time


# Global singleton
neo_core = NeoCore()


def create_app(config=None) -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        FastAPI application instance
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan: initialize NeoCore on startup."""
        neo_core.initialize(config)
        if neo_core.vox and not neo_core.vox._current_session:
            neo_core.vox.start_new_session(neo_core.config.user_name)
        yield

    app = FastAPI(
        title="Neo Core API",
        description="API REST pour Neo Core — Écosystème IA Multi-Agents",
        version="0.8.5",
        lifespan=lifespan,
    )

    # Global exception handler — retourne du JSON propre au lieu d'un stacktrace
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again.",
            },
        )

    # CORS middleware — restreint aux origines locales et au VPS
    # Note : si tu ajoutes un front-end externe, ajoute son domaine ici.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["X-Neo-Key", "Content-Type", "Authorization"],
    )

    # --- Security middlewares ---
    from neo_core.api.middleware import APIKeyMiddleware, RateLimitMiddleware, SanitizerMiddleware

    # 1. API Key authentication (protège /chat, /sessions, /persona, /ws/chat)
    #    /health et /docs restent ouverts (gérés dans le middleware)
    api_key = None
    if neo_core.config:
        api_key = getattr(neo_core.config.llm, "api_key", None)
    if not api_key:
        import os
        api_key = os.getenv("NEO_API_KEY", "")
    if api_key:
        app.add_middleware(APIKeyMiddleware, api_key=api_key)
        logger.info("API key authentication enabled")
    else:
        logger.warning("No API key configured — API endpoints are UNPROTECTED")

    # 2. Rate limiting (persistant SQLite)
    data_dir = neo_core.config.data_dir if neo_core.config else None
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60, data_dir=data_dir)

    # 3. Input sanitization
    app.add_middleware(SanitizerMiddleware, max_length=10_000)

    # Import and include routes
    from neo_core.api.routes import router
    app.include_router(router)

    return app
