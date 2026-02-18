"""
Neo Core API Routes — FastAPI Endpoints
========================================

REST endpoints for interacting with Neo Core.
"""

import asyncio
import hmac
import json
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse
from typing import Optional

from neo_core.vox.api.schemas import (
    ChatRequest,
    ChatResponse,
    StatusResponse,
    SessionInfo,
    HistoryTurn,
    ErrorResponse,
)
from neo_core.vox.api.server import neo_core

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

    try:
        async with neo_core._lock:
            response = await asyncio.wait_for(
                neo_core.vox.process_message(request.message),
                timeout=120.0,
            )
    except asyncio.TimeoutError:
        raise HTTPException(504, "Request timeout — Brain did not respond in time")

    session = neo_core.vox._current_session
    return ChatResponse(
        response=response,
        session_id=session.session_id if session else "",
        turn_number=session.message_count if session else 0,
        timestamp=datetime.now().isoformat(),
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Chat avec streaming SSE — envoie un accusé de réception immédiat
    puis la réponse finale quand Brain termine.

    Events SSE :
      event: ack     → {"text": "Je réfléchis..."}
      event: response → {"text": "...", "session_id": "...", "turn_number": N}
      event: error    → {"text": "..."}
    """
    if not neo_core._initialized:
        raise HTTPException(503, "Neo Core not initialized")

    if request.session_id and neo_core.vox._current_session:
        if request.session_id != neo_core.vox._current_session.session_id:
            neo_core.vox.resume_session(request.session_id)

    message = request.message

    async def event_generator():
        # Queue pour recevoir ack et résultat depuis les callbacks Vox
        result_queue: asyncio.Queue = asyncio.Queue()

        # Sauvegarder les callbacks existants
        old_thinking_cb = neo_core.vox._on_thinking_callback
        old_brain_done_cb = neo_core.vox._on_brain_done_callback

        def on_ack(ack_text: str):
            """Callback ack — Vox envoie l'accusé de réception."""
            result_queue.put_nowait(("ack", ack_text))

        def on_brain_done(brain_response: str):
            """Callback quand Brain termine en arrière-plan."""
            result_queue.put_nowait(("response", brain_response))

        # Installer les callbacks pour le streaming
        neo_core.vox.set_thinking_callback(on_ack)
        neo_core.vox.set_brain_done_callback(on_brain_done)

        try:
            async with neo_core._lock:
                # Lancer process_message — en mode async, retourne immédiatement l'ack
                try:
                    ack_or_response = await asyncio.wait_for(
                        neo_core.vox.process_message(message),
                        timeout=120.0,
                    )
                except asyncio.TimeoutError:
                    yield f"event: error\ndata: {json.dumps({'text': 'Timeout — Brain n\\'a pas répondu'})}\n\n"
                    return

            # Drainer tous les events de la queue (ack envoyé pendant process_message)
            while not result_queue.empty():
                try:
                    event_type, text = result_queue.get_nowait()
                    yield f"event: {event_type}\ndata: {json.dumps({'text': text})}\n\n"
                except asyncio.QueueEmpty:
                    break

            # Si brain_done_callback est actif, process_message a retourné l'ack
            # et Brain tourne en arrière-plan → attendre la réponse
            if neo_core.vox._brain_busy:
                # Envoyer l'ack retourné par process_message
                yield f"event: ack\ndata: {json.dumps({'text': ack_or_response})}\n\n"

                # Attendre le résultat Brain via la queue
                try:
                    event_type, brain_response = await asyncio.wait_for(
                        result_queue.get(), timeout=120.0,
                    )
                except asyncio.TimeoutError:
                    yield f"event: error\ndata: {json.dumps({'text': 'Timeout — Brain n\\'a pas répondu'})}\n\n"
                    return

                session = neo_core.vox._current_session
                yield f"event: response\ndata: {json.dumps({'text': brain_response, 'session_id': session.session_id if session else '', 'turn_number': session.message_count if session else 0})}\n\n"
            else:
                # Mode synchrone — process_message a retourné la réponse complète
                session = neo_core.vox._current_session
                yield f"event: response\ndata: {json.dumps({'text': ack_or_response, 'session_id': session.session_id if session else '', 'turn_number': session.message_count if session else 0})}\n\n"

        finally:
            # Restaurer les callbacks
            neo_core.vox._on_thinking_callback = old_thinking_cb
            neo_core.vox._on_brain_done_callback = old_brain_done_cb

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health")
async def health():
    """
    Health check enrichi — vérifie tous les composants.

    Returns:
        Status, composants, et timestamp.
    """
    checks = {}

    # 1. Core initialisé ?
    checks["core"] = "ok" if neo_core._initialized else "not_initialized"

    # 2. Vox/Brain/Memory
    if neo_core.vox:
        checks["vox"] = "ok"
        checks["brain"] = "ok" if neo_core.vox.brain else "disconnected"
        checks["memory"] = (
            "ok" if neo_core.vox.memory and neo_core.vox.memory.is_initialized
            else "not_initialized"
        )
    else:
        checks["vox"] = "not_initialized"

    # 3. FAISS (vector search)
    try:
        if neo_core.vox and neo_core.vox.memory and neo_core.vox.memory._store:
            store = neo_core.vox.memory._store
            if store.has_vector_search:
                n_vectors = store._faiss_index.ntotal if store._faiss_index else 0
                checks["faiss"] = f"ok ({n_vectors} vectors)"
            else:
                checks["faiss"] = "no_index"
        else:
            checks["faiss"] = "unavailable"
    except Exception as e:
        checks["faiss"] = f"error: {e}"

    # 4. KeyVault
    try:
        from neo_core.infra.security.vault import KeyVault
        data_dir = neo_core.config.data_dir if neo_core.config else None
        if data_dir:
            vault = KeyVault(data_dir=data_dir)
            vault.initialize()
            vault.close()
            checks["vault"] = "ok"
        else:
            checks["vault"] = "no_config"
    except Exception as e:
        checks["vault"] = f"error: {e}"

    # Statut global
    all_ok = all(
        v == "ok" or v.startswith("ok ")
        for k, v in checks.items()
        if k in ("core", "vox", "brain", "memory")
    )

    return {
        "status": "healthy" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.now().isoformat(),
    }


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


@router.get("/tasks")
async def get_tasks():
    """Get task registry report."""
    if not neo_core._initialized:
        raise HTTPException(503, "Neo Core not initialized")
    vox = neo_core.vox
    if not vox or not vox.memory or not vox.memory.is_initialized:
        raise HTTPException(503, "Memory not initialized")
    return vox.memory.get_tasks_report()


@router.get("/epics")
async def get_epics():
    """Get active epics."""
    if not neo_core._initialized:
        raise HTTPException(503, "Neo Core not initialized")
    vox = neo_core.vox
    if not vox or not vox.memory or not vox.memory.is_initialized:
        raise HTTPException(503, "Memory not initialized")
    registry = vox.memory.task_registry
    if not registry:
        return {"epics": []}
    epics = registry.get_all_epics(limit=15)
    result = []
    for epic in epics:
        tasks = registry.get_epic_tasks(epic.id)
        done = sum(1 for t in tasks if t.status == "done")
        total = len(tasks)
        result.append({
            "id": epic.id,
            "description": epic.description,
            "status": epic.status,
            "strategy": getattr(epic, "strategy", ""),
            "progress": f"{done}/{total}",
        })
    return {"epics": result}


@router.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket, token: str = Query(default="")):
    """
    WebSocket endpoint for real-time chat.

    Authentication via query param: /ws/chat?token=<api_key>
    (WebSocket ne supporte pas les headers custom côté navigateur)

    Accepts JSON messages with "message" field and responds with
    "response", "session_id", and "timestamp".
    """
    # Auth WebSocket — vérifie le token query param
    if neo_core.config:
        api_key = getattr(neo_core.config.llm, "api_key", None)
        if not api_key:
            import os
            api_key = os.getenv("NEO_API_KEY", "")
        if api_key and not hmac.compare_digest(token.encode(), api_key.encode()):
            await ws.close(code=4001, reason="Unauthorized")
            logger.warning("WebSocket auth failed")
            return

    await ws.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            data = await ws.receive_json()
            message = data.get("message", "")
            if not message:
                await ws.send_json({"error": "Empty message"})
                continue

            try:
                async with neo_core._lock:
                    response = await asyncio.wait_for(
                        neo_core.vox.process_message(message),
                        timeout=120.0,
                    )
            except asyncio.TimeoutError:
                await ws.send_json({"error": "Request timeout"})
                continue

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
        try:
            await ws.close(code=1011, reason="Internal error")
        except Exception:
            pass  # Client déjà déconnecté
