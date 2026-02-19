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

    Architecture v2 — Mode synchrone avec ack isolé :
    Chaque requête SSE est autonome (pas de callbacks globaux partagés).
    L'ack est généré directement, puis Brain est appelé synchroniquement.
    Cela élimine la race condition entre requêtes concurrentes.

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
        vox = neo_core.vox

        # ── Phase 1 : Ack rapide (hors lock, sans toucher Brain) ──
        try:
            ack_text = await asyncio.wait_for(
                vox._generate_ack(message),
                timeout=4.0,
            )
        except Exception:
            import random
            ack_text = random.choice([
                "🧠 Brain analyse la situation et revient vers toi très vite.",
                "Je traite ta demande...",
                "C'est noté, je m'en occupe.",
            ])

        yield f"event: ack\ndata: {json.dumps({'text': ack_text})}\n\n"

        # ── Phase 2 : Traitement synchrone complet (sous lock) ──
        # On désactive temporairement les callbacks async pour forcer
        # process_message à retourner la réponse complète (pas un ack).
        old_brain_done_cb = vox._on_brain_done_callback
        old_thinking_cb = vox._on_thinking_callback

        try:
            # Forcer le mode synchrone : pas de callback → process_message
            # attend Brain et retourne la réponse complète.
            vox._on_brain_done_callback = None
            vox._on_thinking_callback = None

            async with neo_core._lock:
                try:
                    full_response = await asyncio.wait_for(
                        vox.process_message(message),
                        timeout=300.0,  # 5 min pour les projets longs
                    )
                except asyncio.TimeoutError:
                    timeout_msg = json.dumps({"text": "Timeout — Brain n'a pas répondu dans le délai imparti."})
                    yield f"event: error\ndata: {timeout_msg}\n\n"
                    return

            session = vox._current_session
            yield f"event: response\ndata: {json.dumps({'text': full_response, 'session_id': session.session_id if session else '', 'turn_number': session.message_count if session else 0})}\n\n"

        except Exception as e:
            error_msg = json.dumps({"text": f"{type(e).__name__}: {str(e)[:300]}"})
            yield f"event: error\ndata: {error_msg}\n\n"

        finally:
            # Restaurer les callbacks
            vox._on_brain_done_callback = old_brain_done_cb
            vox._on_thinking_callback = old_thinking_cb

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

    # Agent statuses — expose les statuts vivants de Vox (_agent_statuses)
    # + infos structurelles (connected, initialized)
    agents = {
        "vox": {"connected": vox is not None},
        "brain": {"connected": vox.brain is not None if vox else False},
        "memory": {
            "connected": vox.memory is not None if vox else False,
            "initialized": vox.memory.is_initialized if vox and vox.memory else False,
        },
    }

    # Injecter les statuts temps-réel depuis Vox._agent_statuses
    if vox and hasattr(vox, "_agent_statuses"):
        for name, status in vox._agent_statuses.items():
            key = name  # "Vox", "Brain", "Memory" — majuscule
            agents[key] = {
                **agents.get(key.lower(), {}),  # Conserver connected/initialized
                "active": status.active,
                "task": status.current_task,
                "progress": status.progress,
            }

    # Ajouter les stats Memory (entries, turn_count, etc.)
    if vox and vox.memory and vox.memory.is_initialized:
        try:
            mem_stats = vox.memory.get_stats()
            agents["Memory"]["stats"] = {
                "total_entries": mem_stats.get("total_entries", 0),
                "turn_count": mem_stats.get("turn_count", 0),
            }
        except Exception:
            pass

    # Workers actifs
    workers_info = {"active": [], "stats": {}}
    try:
        if vox and vox.brain and hasattr(vox.brain, "worker_manager"):
            wm = vox.brain.worker_manager
            workers_info["active"] = wm.get_active_workers()
            workers_info["stats"] = wm.get_stats()
    except Exception:
        pass

    # Modèles assignés aux agents principaux
    agent_models = {}
    try:
        from neo_core.config import AGENT_MODELS
        for key, cfg in AGENT_MODELS.items():
            agent_models[key] = cfg.model
    except Exception:
        pass

    # Heartbeat info
    heartbeat_info = {}
    try:
        from neo_core.infra.registry import core_registry
        hb = core_registry.get_heartbeat_manager()
        if hb:
            hb_status = hb.get_status()
            heartbeat_info = {
                "running": hb_status.get("running", False),
                "pulse_count": hb_status.get("pulse_count", 0),
                "interval": hb_status.get("interval", 0),
                "last_event": hb_status.get("last_event", ""),
            }
    except Exception:
        pass

    # Projets actifs (résumé compact pour /status)
    projects_summary = {}
    try:
        registry = vox.memory.task_registry if vox and vox.memory and vox.memory.is_initialized else None
        if registry:
            all_epics = registry.get_all_epics(limit=10)
            active_epics = [e for e in all_epics if e.status in ("pending", "in_progress")]
            done_epics = [e for e in all_epics if e.status == "done"]
            failed_epics = [e for e in all_epics if e.status == "failed"]

            projects_list = []
            for e in active_epics:
                tasks = registry.get_epic_tasks(e.id)
                done_t = sum(1 for t in tasks if t.status == "done")
                total_t = len(tasks)
                # Crew status
                crew_status = None
                try:
                    if vox.brain:
                        from neo_core.brain.teams.crew import CrewExecutor
                        executor = CrewExecutor(brain=vox.brain)
                        cs = executor.load_state(e.id)
                        if cs:
                            crew_status = cs.status
                except Exception:
                    pass
                # Elapsed
                elapsed = ""
                try:
                    from datetime import datetime as _dt
                    started = _dt.fromisoformat(e.created_at)
                    delta = _dt.now() - started
                    total_secs = int(delta.total_seconds())
                    if total_secs < 60:
                        elapsed = f"{total_secs}s"
                    elif total_secs < 3600:
                        elapsed = f"{total_secs // 60}m{total_secs % 60:02d}s"
                    else:
                        h = total_secs // 3600
                        m = (total_secs % 3600) // 60
                        elapsed = f"{h}h{m:02d}m"
                except Exception:
                    pass

                projects_list.append({
                    "short_id": e.short_id,
                    "name": e.display_name,
                    "status": e.status,
                    "crew_status": crew_status,
                    "progress": f"{done_t}/{total_t}",
                    "elapsed": elapsed,
                })

            all_tasks = registry.get_all_tasks(limit=50)
            standalone = [t for t in all_tasks if not t.epic_id]
            standalone_active = [t for t in standalone if t.status in ("pending", "in_progress")]

            projects_summary = {
                "active": projects_list,
                "done_count": len(done_epics),
                "failed_count": len(failed_epics),
                "standalone_tasks": len(standalone_active),
            }
    except Exception:
        pass

    return StatusResponse(
        status="ready",
        core_name=neo_core.config.core_name,
        uptime_seconds=neo_core.uptime,
        agents=agents,
        guardian_mode=False,
        heartbeat=heartbeat_info,
        workers=workers_info,
        agent_models=agent_models,
        projects=projects_summary,
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
    """Get task registry — structured data with epic_id."""
    if not neo_core._initialized:
        raise HTTPException(503, "Neo Core not initialized")
    vox = neo_core.vox
    if not vox or not vox.memory or not vox.memory.is_initialized:
        raise HTTPException(503, "Memory not initialized")
    registry = vox.memory.task_registry
    if not registry:
        return {"tasks": [], "summary": {}}
    tasks = registry.get_all_tasks(limit=50)
    # /tasks ne retourne que les tâches standalone (sans projet)
    # Les tâches liées à un projet sont visibles via /project
    standalone = [t for t in tasks if not t.epic_id]
    return {
        "tasks": [
            {
                "id": t.id,
                "short_id": t.short_id,
                "description": t.description,
                "status": t.status,
                "worker_type": t.worker_type,
            }
            for t in standalone
        ],
        "summary": registry.get_summary(),
    }


@router.get("/project")
@router.get("/epics")
async def get_epics():
    """Get active projects (formerly epics)."""
    if not neo_core._initialized:
        raise HTTPException(503, "Neo Core not initialized")
    vox = neo_core.vox
    if not vox or not vox.memory or not vox.memory.is_initialized:
        raise HTTPException(503, "Memory not initialized")
    registry = vox.memory.task_registry
    if not registry:
        return {"epics": []}
    epics = registry.get_all_epics(limit=15)

    # Map worker_type → model name (for display)
    from neo_core.config import AGENT_MODELS
    def _model_for_worker(wtype: str) -> str:
        key = f"worker:{wtype}"
        cfg = AGENT_MODELS.get(key, AGENT_MODELS.get("brain"))
        if cfg:
            name = cfg.model
            # Shorten: "claude-sonnet-4-6" → "Sonnet 4.6"
            if "sonnet" in name:
                return "Sonnet"
            if "haiku" in name:
                return "Haiku"
            if "opus" in name:
                return "Opus"
            return name
        return "?"

    # Try loading CrewState for richer data (execution times, etc.)
    crew_states = {}
    try:
        from neo_core.brain.teams.crew import CrewExecutor
        brain = getattr(vox, "brain", None)
        if brain:
            executor = CrewExecutor(brain=brain)
            for epic in epics:
                cs = executor.load_state(epic.id)
                if cs:
                    crew_states[epic.id] = cs
    except Exception:
        pass

    result = []
    for epic in epics:
        tasks = registry.get_epic_tasks(epic.id)
        tasks.sort(key=lambda t: t.created_at)
        done = sum(1 for t in tasks if t.status == "done")
        total = len(tasks)

        crew_state = crew_states.get(epic.id)

        # Compute elapsed time
        elapsed_str = ""
        try:
            from datetime import datetime as _dt
            started = _dt.fromisoformat(epic.created_at)
            if epic.status in ("done", "failed") and epic.completed_at:
                ended = _dt.fromisoformat(epic.completed_at)
            else:
                ended = _dt.now()
            delta = ended - started
            total_secs = int(delta.total_seconds())
            if total_secs < 60:
                elapsed_str = f"{total_secs}s"
            elif total_secs < 3600:
                elapsed_str = f"{total_secs // 60}m{total_secs % 60:02d}s"
            else:
                h = total_secs // 3600
                m = (total_secs % 3600) // 60
                elapsed_str = f"{h}h{m:02d}m"
        except Exception:
            pass

        # Build enriched task list
        enriched_tasks = []
        for i, t in enumerate(tasks):
            model = _model_for_worker(t.worker_type)
            # Try to get execution time from CrewState
            exec_time = ""
            if crew_state:
                for r in crew_state.results:
                    if r.index == i:
                        if r.execution_time > 0:
                            et = r.execution_time
                            exec_time = f"{et:.1f}s" if et < 60 else f"{et / 60:.1f}m"
                        break

            enriched_tasks.append({
                "short_id": t.short_id,
                "description": t.description,
                "status": t.status,
                "worker_type": t.worker_type,
                "model": model,
                "created_at": t.created_at,
                "completed_at": t.completed_at,
                "result_preview": t.result[:120] if t.result else "",
                "execution_time": exec_time,
                "attempt_count": t.attempt_count,
            })

        result.append({
            "id": epic.id,
            "short_id": epic.short_id,
            "name": epic.display_name,
            "description": epic.description,
            "status": epic.status,
            "strategy": getattr(epic, "strategy", ""),
            "progress": f"{done}/{total}",
            "created_at": epic.created_at,
            "completed_at": getattr(epic, "completed_at", ""),
            "elapsed": elapsed_str,
            "crew_status": crew_state.status if crew_state else None,
            "tasks": enriched_tasks,
        })
    return {"epics": result}


@router.delete("/tasks/reset")
async def reset_tasks():
    """Reset all standalone tasks."""
    if not neo_core._initialized:
        raise HTTPException(503, "Neo Core not initialized")
    vox = neo_core.vox
    if not vox or not vox.memory or not vox.memory.is_initialized:
        raise HTTPException(503, "Memory not initialized")
    registry = vox.memory.task_registry
    if not registry:
        raise HTTPException(503, "TaskRegistry not available")
    deleted = registry.reset_all_tasks()
    return {"deleted": deleted, "message": f"{deleted} tâche(s) supprimée(s)"}


@router.delete("/tasks/{short_id}")
async def delete_task(short_id: str):
    """Delete a single task by short ID (e.g. T3 or 3)."""
    if short_id == "reset":
        return await reset_tasks()
    if not neo_core._initialized:
        raise HTTPException(503, "Neo Core not initialized")
    vox = neo_core.vox
    if not vox or not vox.memory or not vox.memory.is_initialized:
        raise HTTPException(503, "Memory not initialized")
    registry = vox.memory.task_registry
    if not registry:
        raise HTTPException(503, "TaskRegistry not available")
    task = registry.delete_task(short_id)
    if not task:
        raise HTTPException(404, f"Task '{short_id}' not found")
    return {
        "deleted": True,
        "short_id": task.short_id,
        "description": task.description,
        "message": f"Tâche #{task.short_id} supprimée",
    }


@router.delete("/project/{short_id}")
@router.delete("/epics/{short_id}")
async def delete_epic(short_id: str):
    """Delete a project and its tasks by short ID (e.g. P1 or 1)."""
    if short_id == "reset":
        return await _reset_epics_impl()
    if not neo_core._initialized:
        raise HTTPException(503, "Neo Core not initialized")
    vox = neo_core.vox
    if not vox or not vox.memory or not vox.memory.is_initialized:
        raise HTTPException(503, "Memory not initialized")
    registry = vox.memory.task_registry
    if not registry:
        raise HTTPException(503, "TaskRegistry not available")
    epic, tasks_deleted = registry.delete_epic(short_id)
    if not epic:
        raise HTTPException(404, f"Project '{short_id}' not found")
    return {
        "deleted": True,
        "short_id": epic.short_id,
        "name": epic.display_name,
        "tasks_deleted": tasks_deleted,
        "message": f"Projet #{epic.short_id} supprimé ({tasks_deleted} tâches)",
    }


@router.delete("/project/reset")
async def reset_projects():
    """Reset all projects and their tasks."""
    return await _reset_epics_impl()


@router.delete("/epics/reset")
async def reset_epics():
    """Reset all projects and their tasks (alias)."""
    return await _reset_epics_impl()


async def _reset_epics_impl():
    """Reset all projects and their tasks."""
    if not neo_core._initialized:
        raise HTTPException(503, "Neo Core not initialized")
    vox = neo_core.vox
    if not vox or not vox.memory or not vox.memory.is_initialized:
        raise HTTPException(503, "Memory not initialized")
    registry = vox.memory.task_registry
    if not registry:
        raise HTTPException(503, "TaskRegistry not available")
    deleted = registry.reset_all_epics()
    # Also clean CrewStates
    try:
        records = vox.memory._store.search_by_tags(["crew_state"], limit=100)
        for record in records:
            vox.memory._store.delete(record.id)
            deleted += 1
    except Exception:
        pass
    return {"deleted": deleted, "message": f"{deleted} entrée(s) supprimée(s)"}


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
