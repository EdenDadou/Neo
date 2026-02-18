"""
Neo Core API — Pydantic Models
==============================

Request/Response schemas for REST API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    session_id: str
    turn_number: int
    timestamp: str


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    status: str  # "ready", "busy", "error"
    core_name: str
    uptime_seconds: float
    agents: dict
    guardian_mode: bool
    heartbeat: dict = {}


class SessionInfo(BaseModel):
    """Information about a conversation session."""
    session_id: str
    user_name: str
    created_at: str
    updated_at: str
    message_count: int


class HistoryTurn(BaseModel):
    """A single turn in conversation history."""
    role: str
    content: str
    timestamp: str
    turn_number: int


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: str
