"""
Neo Core API Middleware — Authentication & Rate Limiting
=========================================================

Sécurité renforcée :
- APIKeyMiddleware : comparaison timing-safe (hmac.compare_digest)
- RateLimitMiddleware : stockage SQLite persistant au lieu de dict en mémoire
- SanitizerMiddleware : valide les entrées utilisateur via le Sanitizer
"""

import hmac
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    API key authentication middleware with timing-safe comparison.

    Utilise hmac.compare_digest pour empêcher les timing attacks
    (un attaquant ne peut pas deviner la clé caractère par caractère
    en mesurant le temps de réponse).
    """

    def __init__(self, app, api_key: str = None):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health endpoint and docs
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        if self.api_key:
            key = request.headers.get("X-Neo-Key", "")
            # Timing-safe comparison : empêche les timing attacks
            if not hmac.compare_digest(key.encode("utf-8"), self.api_key.encode("utf-8")):
                logger.warning("Unauthorized API access from %s", request.client.host if request.client else "unknown")
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Unauthorized",
                        "detail": "Invalid or missing API key",
                    },
                )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiter persistant basé sur SQLite.

    Avantages vs dict en mémoire :
    - Survit aux redémarrages du serveur
    - Pas de fuite mémoire sur les IPs
    - Nettoyage automatique des anciennes entrées
    """

    def __init__(self, app, requests_per_minute: int = 60, data_dir: Optional[Path] = None):
        super().__init__(app)
        self.rpm = requests_per_minute
        self._conn: Optional[sqlite3.Connection] = None
        self._db_lock = threading.Lock()

        if data_dir:
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / ".rate_limit.db"
        else:
            db_path = ":memory:"

        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS rate_limits (
                client_ip TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rate_ip_ts
            ON rate_limits(client_ip, timestamp)
        """)
        self._conn.commit()

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60

        with self._db_lock:
            try:
                # Nettoyer les anciennes entrées (> 60s)
                self._conn.execute(
                    "DELETE FROM rate_limits WHERE timestamp < ?", (window_start,)
                )

                # Compter les requêtes dans la fenêtre
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM rate_limits WHERE client_ip = ? AND timestamp >= ?",
                    (client_ip, window_start),
                ).fetchone()
                count = row[0] if row else 0

                if count >= self.rpm:
                    logger.warning("Rate limit exceeded for %s (%d/%d)", client_ip, count, self.rpm)
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Rate limited",
                            "detail": f"Max {self.rpm} requests/minute",
                        },
                    )

                # Enregistrer la requête
                self._conn.execute(
                    "INSERT INTO rate_limits (client_ip, timestamp) VALUES (?, ?)",
                    (client_ip, now),
                )
                self._conn.commit()
            except sqlite3.OperationalError as e:
                logger.warning("Rate limit DB error: %s — allowing request", e)

        return await call_next(request)


class SanitizerMiddleware(BaseHTTPMiddleware):
    """
    Middleware de sanitisation des entrées.

    Vérifie les corps JSON des requêtes POST/PUT pour détecter
    les tentatives d'injection avant qu'elles n'atteignent les agents.
    """

    def __init__(self, app, strict: bool = False, max_length: int = 10_000):
        super().__init__(app)
        from neo_core.security.sanitizer import Sanitizer
        self.sanitizer = Sanitizer(max_length=max_length, strict=strict)

    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    body = await request.body()
                    text = body.decode("utf-8", errors="replace")

                    result = self.sanitizer.sanitize(text)
                    if not result.is_safe and result.severity == "high":
                        logger.warning(
                            "Blocked %s request from %s: %s",
                            request.url.path,
                            request.client.host if request.client else "unknown",
                            ", ".join(result.threats),
                        )
                        return JSONResponse(
                            status_code=400,
                            content={
                                "error": "Invalid input",
                                "detail": "Request contains potentially harmful content",
                            },
                        )
                except Exception as e:
                    logger.warning("Sanitizer body parse failed (allowing request): %s", e)

        return await call_next(request)
