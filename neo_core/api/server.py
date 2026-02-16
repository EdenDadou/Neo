"""
Neo Core API Server — FastAPI Application
==========================================

Creates and manages the FastAPI application and NeoCore singleton.
"""

import asyncio
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from neo_core.config import NeoConfig

logger = logging.getLogger(__name__)


class NeoCore:
    """Singleton wrapping Vox + Brain + Memory for the API."""

    def __init__(self):
        self.vox = None
        self.config = None
        self._lock = asyncio.Lock()
        self._start_time = time.time()
        self._initialized = False

    def initialize(self, config: NeoConfig = None):
        """Bootstrap agents (same as chat.py bootstrap)."""
        if self._initialized:
            return

        from neo_core.core.brain import Brain
        from neo_core.core.memory_agent import MemoryAgent
        from neo_core.core.vox import Vox

        self.config = config or NeoConfig()
        memory = MemoryAgent(config=self.config)
        memory.initialize()
        brain = Brain(config=self.config)
        brain.connect_memory(memory)
        self.vox = Vox(config=self.config)
        self.vox.connect(brain=brain, memory=memory)
        self._initialized = True
        logger.info("NeoCore API initialized")

    def reset(self):
        """Reset singleton state (for testing)."""
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


def create_app(config: NeoConfig = None) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        config: Optional NeoConfig instance

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
        version="1.0.0",
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
