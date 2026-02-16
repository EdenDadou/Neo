"""
Neo Core API Middleware — Authentication & Rate Limiting
=========================================================

Optional middleware for API key authentication and rate limiting.
"""

import time
import logging
from collections import defaultdict
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Optional API key authentication middleware."""

    def __init__(self, app, api_key: str = None):
        """
        Initialize API key middleware.

        Args:
            app: FastAPI application
            api_key: API key to validate (if None, auth is disabled)
        """
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        """
        Validate API key before processing request.

        Skips auth for health endpoint and docs.
        """
        # Skip auth for health endpoint and docs
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        if self.api_key:
            key = request.headers.get("X-Neo-Key", "")
            if key != self.api_key:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Unauthorized",
                        "detail": "Invalid or missing API key",
                    },
                )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter."""

    def __init__(self, app, requests_per_minute: int = 60):
        """
        Initialize rate limit middleware.

        Args:
            app: FastAPI application
            requests_per_minute: Max requests per minute per IP
        """
        super().__init__(app)
        self.rpm = requests_per_minute
        self._requests = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        """
        Rate limit requests by client IP.

        Tracks requests per minute and returns 429 if exceeded.
        """
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean up old requests (older than 60 seconds)
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if now - t < 60
        ]

        if len(self._requests[client_ip]) >= self.rpm:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limited",
                    "detail": f"Max {self.rpm} requests/minute",
                },
            )

        self._requests[client_ip].append(now)
        return await call_next(request)
