"""
Neo Core API Server — FastAPI Application
==========================================

Creates and manages the FastAPI application.
Uses CoreRegistry for a single shared instance of Vox/Brain/Memory.
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
        version="0.8.1",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and include routes
    from neo_core.api.routes import router
    app.include_router(router)

    return app
