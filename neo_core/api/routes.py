"""
Neo Core API Routes — FastAPI Endpoints
========================================

REST endpoints for interacting with Neo Core.
"""

import asyncio
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import Optional

from neo_core.api.schemas import (
    ChatRequest,
    ChatResponse,
    StatusResponse,
    SessionInfo,
    HistoryTurn,
    ErrorResponse,
)
from neo_core.api.server import neo_core

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to Neo and get a response.

    Args:
        request: ChatRequest with message and optional session_id

    Returns:
        ChatResponse with response, session_id, turn_number, timestamp
    """
    if not neo_core._initialized:
        raise HTTPException(503, "Neo Core not initialized")

    # Handle session
    if request.session_id and neo_core.vox._current_session:
        if request.session_id != neo_core.vox._current_session.session_id:
            neo_core.vox.resume_session(request.session_id)

    async with neo_core._lock:
        response = await neo_core.vox.process_message(request.message)

    session = neo_core.vox._current_session
    return ChatResponse(
        response=response,
        session_id=session.session_id if session else "",
        turn_number=session.message_count if session else 0,
        timestamp=datetime.now().isoformat(),
    )


@router.get("/health")
async def health():
    """
    Health check endpoint.

    Returns:
        Status and timestamp
    """
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@router.get("/status", response_model=StatusResponse)
async def status():
    """
    Get system status.

    Returns:
        StatusResponse with core name, agents, uptime, etc.

    Raises:
        HTTPException: If Neo Core is not initialized
    """
    if not neo_core._initialized:
        raise HTTPException(503, "Neo Core not initialized")

    vox = neo_core.vox
    agents = {
        "vox": {"connected": vox is not None},
        "brain": {"connected": vox.brain is not None if vox else False},
        "memory": {
            "connected": vox.memory is not None if vox else False,
            "initialized": vox.memory.is_initialized if vox and vox.memory else False,
        },
    }

    return StatusResponse(
        status="ready",
        core_name=neo_core.config.core_name,
        uptime_seconds=neo_core.uptime,
        agents=agents,
        guardian_mode=False,
    )


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(limit: int = 20):
    """
    List recent conversation sessions.

    Args:
        limit: Maximum number of sessions to return

    Returns:
        List of SessionInfo objects
    """
    if not neo_core.vox or not neo_core.vox._conversation_store:
        return []
    sessions = neo_core.vox._conversation_store.get_sessions(limit=limit)
    return [
        SessionInfo(
            session_id=s.session_id,
            user_name=s.user_name,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=s.message_count,
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}/history", response_model=list[HistoryTurn])
async def get_history(session_id: str, limit: int = 50, offset: int = 0):
    """
    Get conversation history for a session.

    Args:
        session_id: Session ID to retrieve history from
        limit: Maximum number of turns to return
        offset: Number of turns to skip

    Returns:
        List of HistoryTurn objects (empty list if session not found)
    """
    if not neo_core.vox or not neo_core.vox._conversation_store:
        return []
    turns = neo_core.vox._conversation_store.get_history(
        session_id, limit=limit, offset=offset
    )
    return [
        HistoryTurn(
            role=t.role,
            content=t.content,
            timestamp=t.timestamp,
            turn_number=t.turn_number,
        )
        for t in turns
    ]


@router.get("/persona")
async def get_persona():
    """
    Get Neo's current persona and user profile.

    Returns:
        Dictionary with persona and user_profile data

    Raises:
        HTTPException: If agents are not initialized
    """
    if not neo_core.vox or not neo_core.vox.memory:
        raise HTTPException(503, "Agents not initialized")
    persona = neo_core.vox.memory.get_neo_persona()
    profile = neo_core.vox.memory.get_user_profile()
    return {"persona": persona, "user_profile": profile}


@router.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    """
    WebSocket endpoint for real-time chat.

    Accepts JSON messages with "message" field and responds with
    "response", "session_id", and "timestamp".

    Raises:
        WebSocketDisconnect: When client disconnects
    """
    await ws.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            data = await ws.receive_json()
            message = data.get("message", "")
            if not message:
                await ws.send_json({"error": "Empty message"})
                continue

            async with neo_core._lock:
                response = await neo_core.vox.process_message(message)

            session = neo_core.vox._current_session
            await ws.send_json(
                {
                    "response": response,
                    "session_id": session.session_id if session else "",
                    "timestamp": datetime.now().isoformat(),
                }
            )
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
