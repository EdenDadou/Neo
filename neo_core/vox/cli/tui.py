"""
Neo Core — TUI : Interface Textual Full-Dashboard
===================================================
Interface terminal riche basée sur Textual.
Dashboard avec chat scrollable, sidebar agents/heartbeat, et input async.

Supporte deux modes :
- daemon (default) : API SSE streaming vers le daemon FastAPI
- local : Vox direct avec callbacks Brain + Guardian + Heartbeat

Usage : neo chat  (auto-détecte le mode)
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import sys
from typing import Optional

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Footer,
    Header,
    Input,
    RichLog,
    Static,
)

from neo_core.config import NeoConfig

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Custom Messages (thread-safe callback → UI)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BrainResult(Message):
    """Brain a terminé : afficher la réponse."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class AckReceived(Message):
    """Vox a envoyé un accusé de réception."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class ThinkingStarted(Message):
    """Brain est en train de réfléchir."""
    pass


class HeartbeatEvent(Message):
    """Événement heartbeat important."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class ErrorOccurred(Message):
    """Erreur à afficher dans le chat."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class StatusUpdate(Message):
    """Mise à jour de la sidebar."""
    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Sidebar Widget
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _short_model(model: str) -> str:
    """Transforme un nom de modèle long en version courte lisible."""
    if not model or model == "unknown":
        return "?"
    # "claude-sonnet-4-5-20250929" → "sonnet-4.5"
    # "claude-haiku-4-5-20251001" → "haiku-4.5"
    # "claude-opus-4-5-20251101" → "opus-4.5"
    for family in ("sonnet", "haiku", "opus"):
        if family in model.lower():
            # Extraire version (ex: "4-5" → "4.5")
            import re
            ver = re.search(r"(\d+)-(\d+)", model.split(family)[-1] if family in model.lower() else "")
            if ver:
                return f"{family}-{ver.group(1)}.{ver.group(2)}"
            return family
    # Modèle non-Anthropic : garder court
    parts = model.split("/")[-1].split(":")[-1]
    return parts[:20] if len(parts) > 20 else parts


class Sidebar(Static):
    """Panneau latéral : agents, heartbeat, session info."""

    mode_label: reactive[str] = reactive("daemon")
    session_id: reactive[str] = reactive("—")
    brain_status: reactive[str] = reactive("idle")
    memory_status: reactive[str] = reactive("idle")
    memory_entries: reactive[int] = reactive(0)
    memory_turns: reactive[int] = reactive(0)
    vox_status: reactive[str] = reactive("actif")
    heartbeat_active: reactive[bool] = reactive(False)
    heartbeat_pulses: reactive[int] = reactive(0)
    brain_pending: reactive[int] = reactive(0)
    active_tasks: reactive[list] = reactive(list, always_update=True)
    active_epics: reactive[list] = reactive(list, always_update=True)
    active_workers: reactive[list] = reactive(list, always_update=True)
    worker_stats: reactive[dict] = reactive(dict, always_update=True)
    agent_models: reactive[dict] = reactive(dict, always_update=True)

    def render(self) -> Text:
        t = Text()
        models = self.agent_models

        # Compter les agents actifs
        core_active = sum(1 for s in [self.brain_status, self.memory_status, self.vox_status]
                          if s not in ("idle", "inactif"))
        workers = self.active_workers
        total_active = core_active + len(workers)
        stats = self.worker_stats
        total_created = stats.get("total_created", 0) if stats else 0

        t.append(f"◆ Agents ", style="bold cyan")
        t.append(f"({total_active} actif{'s' if total_active > 1 else ''})", style="bold green" if total_active > 0 else "dim")
        if total_created > 0:
            t.append(f"  Σ {total_created}", style="dim")
        t.append("\n")

        # Brain
        brain_icon = "●" if self.brain_status != "idle" else "○"
        brain_style = "bold yellow" if self.brain_status != "idle" else "dim"
        brain_label = self.brain_status
        if self.brain_pending > 0:
            brain_label = f"⟳ {self.brain_pending} req"
            brain_style = "bold cyan"
        brain_model = _short_model(models.get("brain", ""))
        t.append(f"  Brain   {brain_icon} ", style=brain_style)
        t.append(f"{brain_label}", style=brain_style)
        if brain_model:
            t.append(f"  {brain_model}", style="dim italic")
        t.append("\n")

        # Memory
        mem_active = self.memory_status != "idle"
        mem_icon = "●" if mem_active else "○"
        mem_style = "green" if mem_active else "dim"
        mem_model = _short_model(models.get("memory", ""))
        t.append(f"  Memory  {mem_icon} ", style=mem_style)
        t.append(f"{self.memory_status}", style=mem_style)
        if mem_model:
            t.append(f"  {mem_model}", style="dim italic")
        t.append("\n")
        if self.memory_entries > 0:
            t.append(f"          {self.memory_entries} entries", style="dim")
            if self.memory_turns > 0:
                t.append(f" · {self.memory_turns} tours", style="dim")
            t.append("\n", style="dim")

        # Vox
        vox_model = _short_model(models.get("vox", ""))
        t.append(f"  Vox     ● ", style="green")
        t.append(f"{self.vox_status}", style="green")
        if vox_model:
            t.append(f"  {vox_model}", style="dim italic")
        t.append("\n")

        # Workers actifs
        if workers:
            t.append("\n")
            t.append(f"  ◇ Workers ({len(workers)})\n", style="bold magenta")
            for w in workers[:6]:
                wtype = w.get("worker_type", "?")
                wmodel = _short_model(w.get("model", ""))
                wtask = w.get("task", "")
                if len(wtask) > 22:
                    wtask = wtask[:21] + "…"
                t.append(f"    🔄 {wtype}", style="white")
                if wmodel:
                    t.append(f"  {wmodel}", style="dim italic")
                t.append("\n")
                if wtask:
                    t.append(f"       {wtask}\n", style="dim")
            if len(workers) > 6:
                t.append(f"    … +{len(workers) - 6} autres\n", style="dim")
        t.append("\n")

        # Projets en cours (avec tâches groupées)
        t.append("◆ Projets\n", style="bold cyan")
        epics = self.active_epics
        epic_ids = set()
        if epics:
            status_icons = {"pending": "⏳", "in_progress": "🔄", "done": "✅", "failed": "❌"}
            for epic in epics[:5]:
                epic_ids.add(epic.get("id", ""))
                icon = status_icons.get(epic.get("status", ""), "?")
                # Use name (display_name) first, fallback to description
                name = epic.get("name") or epic.get("description", "?")
                sid = epic.get("short_id", "")
                if len(name) > 24:
                    name = name[:23] + "…"
                progress = epic.get("progress", "")
                sid_str = f"#{sid} " if sid else ""
                t.append(f"  {icon} {sid_str}{name}", style="white")
                if progress:
                    t.append(f"  {progress}", style="dim")
                t.append("\n")
                # Show project tasks inline
                epic_tasks = epic.get("tasks", [])
                for et in epic_tasks[:4]:
                    et_status = et.get("status", "")
                    et_icon = status_icons.get(et_status, "⏳")
                    et_desc = et.get("description", "?")
                    if len(et_desc) > 24:
                        et_desc = et_desc[:23] + "…"
                    et_style = "white" if et_status == "in_progress" else "dim"
                    t.append(f"    {et_icon} {et_desc}\n", style=et_style)
                if len(epic_tasks) > 4:
                    t.append(f"    … +{len(epic_tasks) - 4}\n", style="dim")
        else:
            t.append("  aucun\n", style="dim")
        t.append("\n")

        # Tâches indépendantes (pas liées à un projet)
        all_tasks = self.active_tasks
        standalone = [t_ for t_ in all_tasks if not t_.get("epic_id")]
        if standalone:
            t.append("◆ Tâches\n", style="bold cyan")
            for task in standalone[:8]:
                status = task.get("status", "")
                desc = task.get("description", "?")
                if len(desc) > 30:
                    desc = desc[:29] + "…"
                icon = "🔄" if status == "in_progress" else "⏳"
                t.append(f"  {icon} {desc}\n", style="white" if status == "in_progress" else "dim")
            t.append("\n")

        # Heartbeat
        t.append("◆ Heartbeat\n", style="bold cyan")
        hb_state = "actif" if self.heartbeat_active else "inactif"
        hb_style = "green" if self.heartbeat_active else "dim red"
        t.append(f"  ♥ {hb_state}", style=hb_style)
        t.append(f"  Pulse: {self.heartbeat_pulses}\n\n", style="dim")

        # Info
        t.append("◆ Info\n", style="bold cyan")
        t.append(f"  Mode: {self.mode_label}\n", style="dim")
        sid = self.session_id[:12] if len(self.session_id) > 12 else self.session_id
        t.append(f"  Session: {sid}\n", style="dim")
        return t


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main TUI App
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class NeoTUI(App):
    """Neo Core — Terminal User Interface."""

    CSS = """
    #main-row {
        height: 1fr;
        min-height: 20;
    }

    #chat-area {
        width: 3fr;
        height: 100%;
        border: solid $primary;
        padding: 0 1;
        scrollbar-size: 1 1;
    }

    #sidebar {
        width: 1fr;
        height: 100%;
        border: solid $secondary;
        padding: 1;
        min-width: 28;
    }

    #input-field {
        height: 3;
        margin: 0 0;
    }
    """

    TITLE = "Neo Core"
    SUB_TITLE = "Écosystème IA Multi-Agents"

    BINDINGS = [
        Binding("ctrl+c", "quit_app", "Quitter", show=True, priority=True),
        Binding("ctrl+r", "refresh_sidebar", "Refresh", show=True),
    ]

    def __init__(
        self,
        config: NeoConfig,
        mode: str = "daemon",
        api_url: str = "http://localhost:8000",
        vox: object | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.mode = mode  # "daemon" or "local"
        self.api_url = api_url
        self.vox = vox  # Vox instance (local mode only)

        # API mode state
        self._api_key = ""
        self._api_headers: dict = {}
        self._brain_pending = 0
        self._http_client = None

        # Local mode state
        self._brain_results: asyncio.Queue | None = None
        self._heartbeat_manager = None
        self._shutdown_handler = None
        self._queue_watcher_task = None

    # ── Compose ──────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-row"):
            yield RichLog(id="chat-area", wrap=True, highlight=True, markup=True)
            yield Sidebar(id="sidebar")
        yield Input(
            placeholder=f"  {self.config.user_name} > tapez votre message...",
            id="input-field",
        )
        yield Footer()

    # ── Mount : initialisation ───────────────────────────────

    async def on_mount(self) -> None:
        """Initialise le mode choisi et affiche le banner."""
        chat = self.query_one("#chat-area", RichLog)
        sidebar = self.query_one("#sidebar", Sidebar)

        # Banner
        banner = Text()
        banner.append("╔══════════════════════════════════════════╗\n", style="bold cyan")
        name = self.config.core_name.upper()
        banner.append(f"║  {name:^38}  ║\n", style="bold cyan")
        banner.append("║  Écosystème IA Multi-Agents              ║\n", style="cyan")
        banner.append("╚══════════════════════════════════════════╝", style="bold cyan")
        chat.write(banner)

        if self.mode == "daemon":
            sidebar.mode_label = "daemon"
            await self._init_daemon_mode(chat, sidebar)
        else:
            sidebar.mode_label = "local"
            await self._init_local_mode(chat, sidebar)

        # Focus sur l'input
        self.query_one("#input-field", Input).focus()

        # Refresh sidebar toutes les 10s
        self.set_interval(10.0, self._periodic_sidebar_refresh)

    # ── Daemon mode init ─────────────────────────────────────

    async def _init_daemon_mode(self, chat: RichLog, sidebar: Sidebar) -> None:
        """Initialise la connexion API SSE."""
        import httpx

        self._api_key = _get_api_key(self.config)
        if self._api_key:
            self._api_headers = {"X-Neo-Key": self._api_key}

        self._http_client = httpx.AsyncClient(timeout=130.0)

        chat.write(Text(f"\n  Bienvenue {self.config.user_name} — {self.config.core_name} est prêt.", style="bold"))
        chat.write(Text(f"  ✓ Connecté au daemon ({self.api_url})", style="dim green"))
        chat.write(Text(f"  Conversation unifiée avec Telegram.", style="dim"))
        chat.write(Text(f"  Tapez /help pour les commandes.\n", style="dim"))

    # ── Local mode init ──────────────────────────────────────

    async def _init_local_mode(self, chat: RichLog, sidebar: Sidebar) -> None:
        """Initialise Vox local + Guardian + Heartbeat."""
        self._brain_results = asyncio.Queue()

        if self.vox:
            # Callbacks Brain → Message TUI
            def on_brain_done(result: str):
                self.post_message(BrainResult(result))

            def on_thinking(ack_text: str):
                self.post_message(AckReceived(ack_text))

            self.vox.set_brain_done_callback(on_brain_done)
            self.vox.set_thinking_callback(on_thinking)

            # Heartbeat
            try:
                from neo_core.infra.heartbeat import HeartbeatManager, HeartbeatConfig

                def on_heartbeat_event(event):
                    important = {"task_completed", "task_failed", "epic_done", "task_stale",
                                 "persona_reflection", "crew_step_advanced", "orchestrator_replan"}
                    if event.event_type in important:
                        self.post_message(HeartbeatEvent(event.message))

                self._heartbeat_manager = HeartbeatManager(
                    brain=self.vox.brain,
                    memory=self.vox.memory,
                    config=HeartbeatConfig(interval_seconds=300.0),  # 5 minutes
                    on_event=on_heartbeat_event,
                )
                self._heartbeat_manager.start()
                sidebar.heartbeat_active = True
            except Exception as e:
                logger.warning("Heartbeat init failed: %s", e)

            # Session
            info = self.vox.get_session_info()
            if info and info.get("session_id"):
                sidebar.session_id = info["session_id"]

        chat.write(Text(f"\n  Bienvenue {self.config.user_name} — mode local.", style="bold"))
        chat.write(Text(f"  Tapez /help pour les commandes.\n", style="dim"))

    # ── Input handling ───────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Traite la saisie utilisateur."""
        user_input = event.value.strip()
        if not user_input:
            return

        event.input.clear()
        chat = self.query_one("#chat-area", RichLog)

        # Afficher le message utilisateur
        user_text = Text()
        user_text.append(f"  {self.config.user_name} > ", style="bold red")
        user_text.append(user_input)
        chat.write(user_text)

        # Slash commands
        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "quit", "exit", "q"):
            await self._cleanup()
            self.exit()
            return

        if cmd in ("/help", "help"):
            self._show_help(chat)
            return

        if cmd in ("/status", "status"):
            await self._cmd_status(chat)
            return

        if cmd in ("/health", "health"):
            await self._cmd_health(chat)
            return

        if cmd in ("/tasks reset", "tasks reset"):
            await self._cmd_tasks_reset(chat)
            return

        if cmd.startswith(("/tasks delete ", "tasks delete ")):
            sid = cmd.split("delete", 1)[1].strip()
            await self._cmd_tasks_delete(chat, sid)
            return

        if cmd in ("/tasks", "tasks"):
            await self._cmd_tasks(chat)
            return

        if cmd in ("/project reset", "/epics reset", "project reset"):
            await self._cmd_epics_reset(chat)
            return

        if cmd.startswith(("/project delete ", "/epics delete ", "project delete ")):
            sid = cmd.split("delete", 1)[1].strip()
            await self._cmd_epics_delete(chat, sid)
            return

        if cmd in ("/project", "/epics", "project", "epics"):
            await self._cmd_epics(chat)
            return

        if cmd in ("/heartbeat", "heartbeat"):
            self._cmd_heartbeat(chat)
            return

        if cmd in ("/skills", "skills"):
            self._cmd_skills(chat)
            return

        if cmd in ("/persona", "persona"):
            self._cmd_persona(chat)
            return

        if cmd in ("/profile", "profile"):
            self._cmd_profile(chat)
            return

        if cmd in ("/history", "history"):
            self._cmd_history(chat)
            return

        if cmd in ("/sessions", "sessions"):
            self._cmd_sessions(chat)
            return

        if cmd in ("/restart", "restart"):
            await self._cmd_restart()
            return

        if cmd in ("/reflect", "reflect"):
            await self._cmd_reflect(chat)
            return

        # ── Message normal → envoyer au backend ──
        if self.mode == "daemon":
            self._send_api_message(user_input)
        else:
            self._send_local_message(user_input)

    # ── API mode : SSE streaming ─────────────────────────────

    @work(exclusive=False, thread=False)
    async def _send_api_message(self, message: str) -> None:
        """Envoie un message via SSE /chat/stream (daemon mode)."""
        if not self._http_client:
            return

        self._brain_pending += 1
        self._update_sidebar_brain()

        try:
            async with self._http_client.stream(
                "POST",
                f"{self.api_url}/chat/stream",
                json={"message": message},
                headers=self._api_headers,
                timeout=130.0,
            ) as resp:
                if resp.status_code == 401:
                    self.post_message(ErrorOccurred("Accès refusé — clé API invalide."))
                    return
                if resp.status_code == 503:
                    self.post_message(ErrorOccurred("Neo Core pas encore initialisé — réessayez."))
                    return
                if resp.status_code != 200:
                    self.post_message(ErrorOccurred(f"Erreur API ({resp.status_code})"))
                    return

                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    while "\n\n" in buffer:
                        block, buffer = buffer.split("\n\n", 1)
                        event_type = "message"
                        data_str = ""
                        for line in block.split("\n"):
                            if line.startswith("event: "):
                                event_type = line[7:].strip()
                            elif line.startswith("data: "):
                                data_str = line[6:]

                        if not data_str:
                            continue
                        try:
                            payload = _json.loads(data_str)
                        except _json.JSONDecodeError:
                            continue

                        text = payload.get("text", "")

                        if event_type == "ack":
                            self.post_message(AckReceived(text))
                            self.post_message(ThinkingStarted())
                        elif event_type == "response":
                            self.post_message(BrainResult(text))
                        elif event_type == "error":
                            self.post_message(ErrorOccurred(text))

        except Exception as e:
            self.post_message(ErrorOccurred(f"{type(e).__name__}: {e}"))
        finally:
            self._brain_pending -= 1
            self._update_sidebar_brain()

    # ── Local mode : Vox direct ──────────────────────────────

    @work(exclusive=False, thread=False)
    async def _send_local_message(self, message: str) -> None:
        """Envoie un message via Vox local."""
        if not self.vox:
            self.post_message(ErrorOccurred("Vox non initialisé."))
            return

        sidebar = self.query_one("#sidebar", Sidebar)
        sidebar.brain_status = "analyse..."

        try:
            response = await self.vox.process_message(message)
            self.post_message(BrainResult(response))
        except Exception as e:
            self.post_message(ErrorOccurred(f"{type(e).__name__}: {e}"))
        finally:
            sidebar.brain_status = "idle"

    # ── Message handlers (affichage dans le chat) ────────────

    def on_brain_result(self, event: BrainResult) -> None:
        chat = self.query_one("#chat-area", RichLog)
        response_text = Text()
        response_text.append("\n  Vox > ", style="bold cyan")
        response_text.append(event.text)
        response_text.append("")  # blank line
        chat.write(response_text)
        chat.write(Text(""))

    def on_ack_received(self, event: AckReceived) -> None:
        chat = self.query_one("#chat-area", RichLog)
        ack_text = Text()
        ack_text.append("  Vox > ", style="dim cyan")
        ack_text.append(event.text, style="dim")
        chat.write(ack_text)

    def on_thinking_started(self, event: ThinkingStarted) -> None:
        chat = self.query_one("#chat-area", RichLog)
        chat.write(Text("  ⟳ Brain réfléchit...", style="bold cyan"))

    def on_heartbeat_event(self, event: HeartbeatEvent) -> None:
        chat = self.query_one("#chat-area", RichLog)
        chat.write(Text(f"  ♥ {event.text}", style="dim magenta"))
        sidebar = self.query_one("#sidebar", Sidebar)
        sidebar.heartbeat_pulses += 1

    def on_error_occurred(self, event: ErrorOccurred) -> None:
        chat = self.query_one("#chat-area", RichLog)
        err_text = Text()
        err_text.append("  Erreur > ", style="bold red")
        err_text.append(event.text, style="red")
        chat.write(err_text)

    # ── Sidebar helpers ──────────────────────────────────────

    def _update_sidebar_brain(self) -> None:
        try:
            sidebar = self.query_one("#sidebar", Sidebar)
            sidebar.brain_pending = self._brain_pending
            if self._brain_pending > 0:
                sidebar.brain_status = f"⟳ {self._brain_pending} req"
            else:
                sidebar.brain_status = "idle"
        except NoMatches:
            pass

    async def _periodic_sidebar_refresh(self) -> None:
        """Refresh périodique de la sidebar."""
        try:
            sidebar = self.query_one("#sidebar", Sidebar)

            if self.mode == "daemon" and self._http_client:
                try:
                    resp = await self._http_client.get(
                        f"{self.api_url}/status",
                        headers=self._api_headers,
                        timeout=5.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        agents = data.get("agents", {})

                        # Brain status (clé "Brain" majuscule depuis les _agent_statuses)
                        brain_info = agents.get("Brain", agents.get("brain", {}))
                        if isinstance(brain_info, dict):
                            if brain_info.get("active"):
                                sidebar.brain_status = brain_info.get("task") or "travaille..."
                            elif sidebar.brain_status not in ("idle",):
                                sidebar.brain_status = "idle"

                        # Memory status + stats
                        mem_info = agents.get("Memory", agents.get("memory", {}))
                        if isinstance(mem_info, dict):
                            if mem_info.get("active"):
                                sidebar.memory_status = mem_info.get("task") or "actif"
                            elif mem_info.get("initialized"):
                                sidebar.memory_status = "prêt"
                            else:
                                sidebar.memory_status = "idle"
                            # Stats Memory
                            mem_stats = mem_info.get("stats", {})
                            if mem_stats:
                                sidebar.memory_entries = mem_stats.get("total_entries", 0)
                                sidebar.memory_turns = mem_stats.get("turn_count", 0)

                        # Heartbeat info from status if available
                        hb = data.get("heartbeat", {})
                        if hb:
                            sidebar.heartbeat_active = hb.get("running", True)
                            sidebar.heartbeat_pulses = hb.get("pulse_count", 0)
                        else:
                            # Daemon actif = heartbeat actif
                            sidebar.heartbeat_active = True

                        # Workers actifs + modèles
                        workers_data = data.get("workers", {})
                        sidebar.active_workers = workers_data.get("active", [])
                        sidebar.worker_stats = workers_data.get("stats", {})
                        sidebar.agent_models = data.get("agent_models", {})
                except Exception:
                    pass

                # Charger epics via API
                try:
                    resp = await self._http_client.get(
                        f"{self.api_url}/epics",
                        headers=self._api_headers,
                        timeout=5.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        epics = data.get("epics", [])
                        sidebar.active_epics = [
                            e for e in epics
                            if e.get("status") in ("pending", "in_progress")
                        ]
                except Exception:
                    pass

                # Charger tâches via API (structured with epic_id)
                try:
                    resp = await self._http_client.get(
                        f"{self.api_url}/tasks",
                        headers=self._api_headers,
                        timeout=5.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        sidebar.active_tasks = data.get("tasks", [])
                except Exception:
                    pass

            elif self.mode == "local" and self.vox:
                if hasattr(self.vox, "is_brain_busy") and self.vox.is_brain_busy:
                    sidebar.brain_status = "travaille..."
                elif self._brain_pending == 0:
                    sidebar.brain_status = "idle"

                # Memory status — lire depuis Vox._agent_statuses
                if hasattr(self.vox, "_agent_statuses"):
                    mem_agent = self.vox._agent_statuses.get("Memory")
                    if mem_agent:
                        if mem_agent.active:
                            sidebar.memory_status = mem_agent.current_task or "actif"
                        elif self.vox.memory and self.vox.memory.is_initialized:
                            sidebar.memory_status = "prêt"
                        else:
                            sidebar.memory_status = "idle"

                # Memory stats
                if self.vox.memory and self.vox.memory.is_initialized:
                    try:
                        mem_stats = self.vox.memory.get_stats()
                        sidebar.memory_entries = mem_stats.get("total_entries", 0)
                        sidebar.memory_turns = mem_stats.get("turn_count", 0)
                    except Exception:
                        pass

                if self._heartbeat_manager:
                    status = self._heartbeat_manager.get_status()
                    sidebar.heartbeat_active = status.get("running", False)
                    sidebar.heartbeat_pulses = status.get("pulse_count", 0)

                # Workers actifs + modèles (local)
                try:
                    if self.vox.brain and hasattr(self.vox.brain, "worker_manager"):
                        wm = self.vox.brain.worker_manager
                        sidebar.active_workers = wm.get_active_workers()
                        sidebar.worker_stats = wm.get_stats()
                    from neo_core.config import AGENT_MODELS
                    sidebar.agent_models = {k: cfg.model for k, cfg in AGENT_MODELS.items()}
                except Exception:
                    pass

                # Charger epics/tâches en local
                if self.vox.memory and self.vox.memory.is_initialized:
                    try:
                        registry = self.vox.memory.task_registry
                        if registry:
                            epics = registry.get_all_epics(limit=10)
                            epic_list = []
                            for e in epics:
                                if e.status not in ("pending", "in_progress"):
                                    continue
                                e_tasks = registry.get_epic_tasks(e.id)
                                done = sum(1 for t in e_tasks if t.status == "done")
                                epic_list.append({
                                    "id": e.id, "short_id": e.short_id,
                                    "name": e.display_name, "status": e.status,
                                    "progress": f"{done}/{len(e_tasks)}",
                                    "tasks": [
                                        {"short_id": t.short_id, "description": t.description,
                                         "status": t.status, "worker_type": t.worker_type}
                                        for t in e_tasks
                                    ],
                                })
                            sidebar.active_epics = epic_list
                    except Exception:
                        pass
                    try:
                        registry = self.vox.memory.task_registry
                        if registry:
                            all_tasks = registry.get_all_tasks(limit=30)
                            sidebar.active_tasks = [
                                {"description": t.description, "status": t.status,
                                 "epic_id": t.epic_id or "", "short_id": t.short_id}
                                for t in all_tasks
                            ]
                    except Exception:
                        pass

        except NoMatches:
            pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Slash Commands
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _show_help(self, chat: RichLog) -> None:
        table = Table(title="Commandes", title_style="bold cyan", border_style="dim", padding=(0, 2))
        table.add_column("Commande", style="bold")
        table.add_column("Description")
        commands = [
            ("/status", "État des agents"),
            ("/health", "Rapport de santé"),
            ("/tasks", "Registre des tâches"),
            ("/tasks reset", "Supprimer toutes les tâches"),
            ("/project", "Projets en cours"),
            ("/project reset", "Supprimer tous les projets"),
            ("/skills", "Compétences acquises"),
            ("/heartbeat", "État du heartbeat"),
            ("/persona", "Personnalité de Neo"),
            ("/profile", "Profil utilisateur"),
            ("/history", "Historique récent"),
            ("/sessions", "Sessions précédentes"),
            ("/reflect", "Lancer auto-réflexion"),
            ("/restart", "Redémarrer Neo"),
            ("/quit", "Quitter"),
        ]
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        chat.write(table)

    async def _cmd_status(self, chat: RichLog) -> None:
        if self.mode == "daemon" and self._http_client:
            try:
                resp = await self._http_client.get(
                    f"{self.api_url}/status",
                    headers=self._api_headers,
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    lines = [
                        f"  Core    : {data.get('core_name', '?')}",
                        f"  Status  : {data.get('status', '?')}",
                        f"  Uptime  : {data.get('uptime_seconds', 0):.0f}s",
                        f"  Guardian: {'oui' if data.get('guardian_mode') else 'non'}",
                    ]
                    if self._brain_pending > 0:
                        lines.append(f"  Brain   : [bold cyan]⟳ {self._brain_pending} requête(s)[/bold cyan]")
                    agents = data.get("agents", {})
                    for agent_name, info in agents.items():
                        lines.append(f"  {agent_name:10}: {info}")
                    chat.write(Panel("\n".join(lines), title="[bold]État du système[/bold]", border_style="dim"))
                else:
                    chat.write(Text(f"  ⚠ API error: {resp.status_code}", style="yellow"))
            except Exception as e:
                chat.write(Text(f"  Erreur: {e}", style="red"))
        elif self.vox:
            status = self.vox.get_system_status()
            chat.write(Panel(status, title="[bold]État du système[/bold]", border_style="dim"))

    async def _cmd_health(self, chat: RichLog) -> None:
        if self.mode == "daemon" and self._http_client:
            try:
                resp = await self._http_client.get(f"{self.api_url}/health", timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json()
                    checks = data.get("checks", {})
                    lines = [f"  {k}: {v}" for k, v in checks.items()]
                    chat.write(Panel("\n".join(lines), title="[bold]Health[/bold]", border_style="dim"))
            except Exception as e:
                chat.write(Text(f"  Erreur: {e}", style="red"))
        elif self.vox and self.vox.brain:
            try:
                health = self.vox.brain.get_system_health()
                lines = [
                    f"  État global    : {health.get('status', 'inconnu')}",
                    f"  Appels API     : {health.get('total_calls', 0)}",
                    f"  Taux d'erreur  : {health.get('error_rate', 0):.1%}",
                    f"  Temps moyen    : {health.get('avg_response_time', 0):.2f}s",
                    f"  Circuit breaker: {health.get('circuit_state', 'inconnu')}",
                ]
                chat.write(Panel("\n".join(lines), title="[bold cyan]Health Report[/bold cyan]", border_style="cyan"))
            except Exception as e:
                chat.write(Text(f"  Erreur health: {e}", style="red"))

    async def _cmd_tasks(self, chat: RichLog) -> None:
        if self.mode == "daemon" and self._http_client:
            try:
                resp = await self._http_client.get(
                    f"{self.api_url}/tasks",
                    headers=self._api_headers,
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self._render_tasks_organized(chat, data)
            except Exception as e:
                chat.write(Text(f"  Erreur: {e}", style="red"))
        elif self.vox:
            if self.vox.memory and self.vox.memory.is_initialized:
                registry = self.vox.memory.task_registry
                if registry:
                    organized = registry.get_organized_summary()
                    self._render_tasks_local(chat, organized)
                else:
                    chat.write(Text("  ⚠ TaskRegistry non disponible", style="yellow"))
            else:
                chat.write(Text("  ⚠ Memory non initialisé", style="yellow"))

    def _render_tasks_organized(self, chat: RichLog, data: dict) -> None:
        """Rendu organisé des tâches (mode daemon, depuis API), groupé par statut."""
        tasks = data.get("tasks", [])
        status_icons = {"pending": "⏳", "in_progress": "🔄", "done": "✅", "failed": "❌"}
        lines: list[str] = []

        if not tasks:
            lines.append("[dim]Aucune tâche indépendante.[/dim]")
            lines.append("[dim]Les tâches liées à des projets sont dans /project[/dim]")
        else:
            def _render_task_line(t, indent="  "):
                status = t.get("status", "pending")
                icon = status_icons.get(status, "?")
                sid = t.get("short_id", "")
                sid_tag = f"[bold cyan]#{sid}[/bold cyan] " if sid else ""
                desc = t.get("description", "")[:55]
                wtype = t.get("worker_type", "")
                if status == "done":
                    desc = f"[dim]{desc}[/dim]"
                elif status == "failed":
                    desc = f"[red]{desc}[/red]"
                elif status == "in_progress":
                    desc = f"[bold yellow]{desc}[/bold yellow]"
                return f"{indent}{icon} {sid_tag}{desc}  [dim]{wtype}[/dim]"

            in_progress = [t for t in tasks if t.get("status") == "in_progress"]
            pending = [t for t in tasks if t.get("status") == "pending"]
            done = [t for t in tasks if t.get("status") == "done"]
            failed = [t for t in tasks if t.get("status") == "failed"]

            if in_progress:
                lines.append("[bold green]▶ En cours[/bold green]")
                for t in in_progress:
                    lines.append(_render_task_line(t))
                lines.append("")
            if pending:
                lines.append("[bold yellow]◻ À faire[/bold yellow]")
                for t in pending[:10]:
                    lines.append(_render_task_line(t))
                lines.append("")
            if done:
                lines.append("[bold dim]✓ Terminées[/bold dim]")
                for t in done[-5:]:
                    lines.append(_render_task_line(t))
                if len(done) > 5:
                    lines.append(f"  [dim]... et {len(done) - 5} autres[/dim]")
                lines.append("")
            if failed:
                lines.append("[bold red]✗ Échouées[/bold red]")
                for t in failed[-3:]:
                    lines.append(_render_task_line(t))
                lines.append("")

            lines.append(f"[dim]{len(tasks)} tâche(s) · /tasks reset[/dim]")

        chat.write(Panel("\n".join(lines), title="[bold cyan]Tâches[/bold cyan]", border_style="cyan"))

    def _render_tasks_local(self, chat: RichLog, organized: dict) -> None:
        """Rendu organisé des tâches (mode local), groupé par statut."""
        status_icons = {"pending": "⏳", "in_progress": "🔄", "done": "✅", "failed": "❌"}
        lines: list[str] = []

        # Collecter toutes les tâches standalone
        standalone = organized.get("standalone_tasks", [])

        # Grouper par statut
        in_progress = [t for t in standalone if t.status == "in_progress"]
        pending = [t for t in standalone if t.status == "pending"]
        done = [t for t in standalone if t.status == "done"]
        failed = [t for t in standalone if t.status == "failed"]

        def _sid(t):
            return f"[bold cyan]#{t.short_id}[/bold cyan]" if t.short_id else ""

        # ── En cours ──
        if in_progress:
            lines.append("[bold green]▶ En cours[/bold green]")
            for t in in_progress:
                lines.append(f"  🔄 {_sid(t)} {t.description[:50]}  [dim]{t.worker_type}[/dim]")
            lines.append("")

        # ── À faire ──
        if pending:
            lines.append("[bold yellow]◻ À faire[/bold yellow]")
            for t in pending[:10]:
                lines.append(f"  ⏳ {_sid(t)} {t.description[:50]}  [dim]{t.worker_type}[/dim]")
            if len(pending) > 10:
                lines.append(f"  [dim]... et {len(pending) - 10} autres[/dim]")
            lines.append("")

        # ── Terminées ──
        if done:
            lines.append("[bold dim]✓ Terminées[/bold dim]")
            for t in done[-5:]:
                lines.append(f"  [dim]✅ {_sid(t)} {t.description[:50]}  {t.worker_type}[/dim]")
            if len(done) > 5:
                lines.append(f"  [dim]... et {len(done) - 5} autres[/dim]")
            lines.append("")

        # ── Échouées ──
        if failed:
            lines.append("[bold red]✗ Échouées[/bold red]")
            for t in failed[-3:]:
                lines.append(f"  [dim red]❌ {_sid(t)} {t.description[:50]}  {t.worker_type}[/dim red]")
            lines.append("")

        # Stats
        total = len(standalone)
        if total > 0:
            lines.append(f"[dim]{total} tâche(s) · /tasks delete {{id}} | /tasks reset[/dim]")
        else:
            lines.append("[dim]Aucune tâche indépendante.[/dim]")

        chat.write(Panel("\n".join(lines), title="[bold cyan]Tâches[/bold cyan]", border_style="cyan"))

    async def _cmd_epics(self, chat: RichLog) -> None:
        """Commande /project — affiche les projets groupés par statut."""
        if self.mode == "daemon" and self._http_client:
            try:
                resp = await self._http_client.get(
                    f"{self.api_url}/epics",
                    headers=self._api_headers,
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    epics = data.get("epics", [])
                    if not epics:
                        chat.write(Panel("[dim]Aucun projet.[/dim]\n[dim]/project reset pour tout supprimer[/dim]", title="[bold cyan]Projets[/bold cyan]", border_style="cyan"))
                    else:
                        self._render_projects_by_status(chat, epics, from_api=True)
            except Exception as e:
                chat.write(Text(f"  Erreur: {e}", style="red"))
        elif self.vox:
            if self.vox.memory and self.vox.memory.is_initialized:
                registry = self.vox.memory.task_registry
                if registry:
                    epics = registry.get_all_epics(limit=20)
                    if not epics:
                        chat.write(Panel("[dim]Aucun projet.[/dim]", title="[bold cyan]Projets[/bold cyan]", border_style="cyan"))
                    else:
                        self._render_projects_by_status(chat, epics, registry=registry)

    def _render_projects_by_status(self, chat: RichLog, epics, from_api=False, registry=None) -> None:
        """Rendu des projets groupés par statut avec détails complets."""
        status_icons = {"pending": "⏳", "in_progress": "🔄", "done": "✅", "failed": "❌", "paused": "⏸️"}
        worker_icons = {
            "researcher": "🔍", "coder": "💻", "analyst": "📊",
            "writer": "✍️", "summarizer": "📝", "generic": "⚙️",
            "translator": "🌐", "skipped": "⏭️",
        }

        # Model mapping for local (non-API) mode
        _model_map = {}
        try:
            from neo_core.config import AGENT_MODELS
            for key, cfg in AGENT_MODELS.items():
                if key.startswith("worker:"):
                    wt = key.split(":", 1)[1]
                    name = cfg.model
                    if "sonnet" in name:
                        _model_map[wt] = "Sonnet"
                    elif "haiku" in name:
                        _model_map[wt] = "Haiku"
                    elif "opus" in name:
                        _model_map[wt] = "Opus"
                    else:
                        _model_map[wt] = name
        except Exception:
            pass

        lines: list[str] = []

        # Helper: compute elapsed time string
        def _elapsed(created_at_str, completed_at_str=""):
            try:
                from datetime import datetime as _dt
                started = _dt.fromisoformat(created_at_str)
                if completed_at_str:
                    ended = _dt.fromisoformat(completed_at_str)
                else:
                    ended = _dt.now()
                total_secs = int((ended - started).total_seconds())
                if total_secs < 60:
                    return f"{total_secs}s"
                elif total_secs < 3600:
                    return f"{total_secs // 60}m{total_secs % 60:02d}s"
                else:
                    h = total_secs // 3600
                    m = (total_secs % 3600) // 60
                    return f"{h}h{m:02d}m"
            except Exception:
                return ""

        # Grouper par statut
        if from_api:
            active = [e for e in epics if e.get("status") in ("in_progress",)]
            paused = [e for e in epics if e.get("crew_status") == "paused"]
            pending = [e for e in epics if e.get("status") == "pending"]
            done = [e for e in epics if e.get("status") == "done"]
            failed = [e for e in epics if e.get("status") == "failed"]
            # Remove paused from active (they may have status=in_progress but crew paused)
            paused_ids = {e.get("id") for e in paused}
            active = [e for e in active if e.get("id") not in paused_ids]
        else:
            active = [e for e in epics if e.status == "in_progress"]
            paused = []  # Can't detect pause without CrewState in local mode
            pending = [e for e in epics if e.status == "pending"]
            done = [e for e in epics if e.status == "done"]
            failed = [e for e in epics if e.status == "failed"]

        def _render_epic(epic, icon_override=None):
            if from_api:
                status = epic.get("status", "")
                crew_status = epic.get("crew_status", "")
                icon = icon_override or status_icons.get(crew_status or status, "?")
                sid = epic.get("short_id", "")
                sid_tag = f"[bold cyan]#{sid}[/bold cyan] " if sid else ""
                name = epic.get("name", "") or epic.get("description", "")[:50]
                progress = epic.get("progress", "0/0")
                elapsed = epic.get("elapsed", "")
                elapsed_tag = f"  [dim]({elapsed})[/dim]" if elapsed else ""

                # Progress bar
                parts = progress.split("/")
                try:
                    d, tot = int(parts[0]), int(parts[1])
                    pct = d * 100 // tot if tot > 0 else 0
                    bar_len = 12
                    filled = d * bar_len // tot if tot > 0 else 0
                    bar = f"[green]{'█' * filled}[/green][dim]{'░' * (bar_len - filled)}[/dim]"
                except (ValueError, IndexError):
                    pct = 0
                    bar = ""

                lines.append(f"  {icon} {sid_tag}[bold]{name[:40]}[/bold]  {bar} {progress} ({pct}%){elapsed_tag}")

                # Strategy line (if present)
                strategy = epic.get("strategy", "")
                if strategy:
                    lines.append(f"    [dim italic]Stratégie : {strategy[:70]}[/dim italic]")

                # Sous-tâches enrichies
                api_tasks = epic.get("tasks", [])
                for t in api_tasks:
                    t_status = t.get("status", "pending")
                    t_icon = status_icons.get(t_status, "?")
                    t_sid = t.get("short_id", "")
                    t_sid_tag = f"[cyan]#{t_sid}[/cyan] " if t_sid else ""
                    desc = t.get("description", "")[:38]
                    wtype = t.get("worker_type", "generic")
                    model = t.get("model", _model_map.get(wtype, "?"))
                    w_icon = worker_icons.get(wtype, "⚙️")
                    exec_time = t.get("execution_time", "")
                    exec_tag = f" [green]{exec_time}[/green]" if exec_time else ""

                    # Style selon statut
                    if t_status == "done":
                        desc = f"[dim]{desc}[/dim]"
                        agent_info = f"[dim]{w_icon} {wtype}[/dim] [dim italic]({model})[/dim italic]{exec_tag}"
                    elif t_status == "failed":
                        desc = f"[red]{desc}[/red]"
                        agent_info = f"[red]{w_icon} {wtype}[/red] [dim italic]({model})[/dim italic]"
                    elif t_status == "in_progress":
                        desc = f"[bold yellow]{desc}[/bold yellow]"
                        agent_info = f"[yellow]{w_icon} {wtype}[/yellow] [italic]({model})[/italic]"
                    else:
                        agent_info = f"[dim]{w_icon} {wtype} ({model})[/dim]"

                    lines.append(f"      {t_icon} {t_sid_tag}{desc}  {agent_info}")

                    # Result preview for done tasks
                    if t_status == "done":
                        preview = t.get("result_preview", "")
                        if preview:
                            lines.append(f"         [dim]→ {preview[:65]}…[/dim]")

            else:
                # Local mode (non-API)
                icon = icon_override or status_icons.get(epic.status, "?")
                sid_tag = f"[bold cyan]#{epic.short_id}[/bold cyan] " if epic.short_id else ""
                epic_tasks = registry.get_epic_tasks(epic.id) if registry else []
                epic_tasks.sort(key=lambda t: t.created_at)
                d = sum(1 for t in epic_tasks if t.status == "done")
                tot = len(epic_tasks)
                pct = f"{d * 100 // tot}%" if tot > 0 else "—"
                elapsed = _elapsed(epic.created_at, getattr(epic, "completed_at", ""))
                elapsed_tag = f"  [dim]({elapsed})[/dim]" if elapsed else ""

                # Progress bar
                bar_len = 12
                filled = d * bar_len // tot if tot > 0 else 0
                bar = f"[green]{'█' * filled}[/green][dim]{'░' * (bar_len - filled)}[/dim]"

                lines.append(f"  {icon} {sid_tag}[bold]{epic.display_name[:40]}[/bold]  {bar} {d}/{tot} ({pct}){elapsed_tag}")

                # Strategy
                if hasattr(epic, "strategy") and epic.strategy:
                    lines.append(f"    [dim italic]Stratégie : {epic.strategy[:70]}[/dim italic]")

                # Sous-tâches
                for t in epic_tasks:
                    t_icon = status_icons.get(t.status, "?")
                    t_sid = f"[cyan]#{t.short_id}[/cyan] " if t.short_id else ""
                    desc = t.description[:38]
                    wtype = t.worker_type or "generic"
                    model = _model_map.get(wtype, "?")
                    w_icon = worker_icons.get(wtype, "⚙️")

                    if t.status == "done":
                        desc = f"[dim]{desc}[/dim]"
                        agent_info = f"[dim]{w_icon} {wtype}[/dim] [dim italic]({model})[/dim italic]"
                    elif t.status == "failed":
                        desc = f"[red]{desc}[/red]"
                        agent_info = f"[red]{w_icon} {wtype}[/red] [dim italic]({model})[/dim italic]"
                    elif t.status == "in_progress":
                        desc = f"[bold yellow]{desc}[/bold yellow]"
                        agent_info = f"[yellow]{w_icon} {wtype}[/yellow] [italic]({model})[/italic]"
                    else:
                        agent_info = f"[dim]{w_icon} {wtype} ({model})[/dim]"

                    lines.append(f"      {t_icon} {t_sid}{desc}  {agent_info}")

                    # Result preview
                    if t.status == "done" and t.result:
                        lines.append(f"         [dim]→ {t.result[:65]}…[/dim]")

        # ── En cours ──
        if active:
            lines.append("[bold green]▶ En cours[/bold green]")
            for e in active:
                _render_epic(e)
                lines.append("")

        # ── En pause ──
        if paused:
            lines.append("[bold magenta]⏸ En pause[/bold magenta]")
            for e in paused:
                _render_epic(e, icon_override="⏸️")
                lines.append("")

        # ── À faire ──
        if pending:
            lines.append("[bold yellow]◻ À faire[/bold yellow]")
            for e in pending:
                _render_epic(e)
                lines.append("")

        # ── Terminés ──
        if done:
            lines.append("[bold dim]✓ Terminés[/bold dim]")
            for e in done[-5:]:
                _render_epic(e)
            if len(done) > 5:
                lines.append(f"  [dim]... et {len(done) - 5} autres[/dim]")
            lines.append("")

        # ── Échoués ──
        if failed:
            lines.append("[bold red]✗ Échoués[/bold red]")
            for e in failed[-3:]:
                _render_epic(e)
            lines.append("")

        total = len(epics)
        lines.append(f"[dim]{total} projet(s) · /project delete {{id}} | /project reset[/dim]")

        chat.write(Panel("\n".join(lines), title="[bold cyan]Projets[/bold cyan]", border_style="cyan"))

    async def _cmd_tasks_reset(self, chat: RichLog) -> None:
        """Commande /tasks reset — supprime toutes les tâches standalone."""
        if self.mode == "daemon" and self._http_client:
            try:
                resp = await self._http_client.request(
                    "DELETE",
                    f"{self.api_url}/tasks/reset",
                    headers=self._api_headers,
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    deleted = data.get("deleted", 0)
                    chat.write(Panel(
                        f"[green]✅ {deleted} tâche(s) supprimée(s).[/green]\nLe registre de tâches est remis à zéro.",
                        title="[bold cyan]Tasks Reset[/bold cyan]",
                        border_style="cyan",
                    ))
                else:
                    chat.write(Text(f"  Erreur API: {resp.status_code}", style="red"))
            except Exception as e:
                chat.write(Text(f"  Erreur: {e}", style="red"))
        elif self.vox and self.vox.memory and self.vox.memory.is_initialized:
            registry = self.vox.memory.task_registry
            if registry:
                deleted = registry.reset_all_tasks()
                chat.write(Panel(
                    f"[green]✅ {deleted} tâche(s) supprimée(s).[/green]\nLe registre de tâches est remis à zéro.",
                    title="[bold cyan]Tasks Reset[/bold cyan]",
                    border_style="cyan",
                ))
                return
        else:
            chat.write(Text("  ⚠ Memory non disponible", style="yellow"))

    async def _cmd_epics_reset(self, chat: RichLog) -> None:
        """Commande /project reset — supprime tous les projets et leurs tâches."""
        if self.mode == "daemon" and self._http_client:
            try:
                resp = await self._http_client.request(
                    "DELETE",
                    f"{self.api_url}/project/reset",
                    headers=self._api_headers,
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    deleted = data.get("deleted", 0)
                    chat.write(Panel(
                        f"[green]✅ {deleted} entrée(s) supprimée(s).[/green]\nTous les projets sont remis à zéro.",
                        title="[bold cyan]Projects Reset[/bold cyan]",
                        border_style="cyan",
                    ))
                else:
                    chat.write(Text(f"  Erreur API: {resp.status_code}", style="red"))
            except Exception as e:
                chat.write(Text(f"  Erreur: {e}", style="red"))
        elif self.vox and self.vox.memory and self.vox.memory.is_initialized:
            registry = self.vox.memory.task_registry
            if registry:
                deleted = registry.reset_all_epics()
                try:
                    records = self.vox.memory._store.search_by_tags(["crew_state"], limit=100)
                    for record in records:
                        self.vox.memory._store.delete(record.id)
                        deleted += 1
                except Exception:
                    pass
                chat.write(Panel(
                    f"[green]✅ {deleted} entrée(s) supprimée(s).[/green]\nTous les projets sont remis à zéro.",
                    title="[bold cyan]Projects Reset[/bold cyan]",
                    border_style="cyan",
                ))
                return
        else:
            chat.write(Text("  ⚠ Memory non disponible", style="yellow"))

    async def _cmd_tasks_delete(self, chat: RichLog, short_id: str) -> None:
        """Commande /tasks delete {id} — supprime une tâche spécifique."""
        if self.mode == "daemon" and self._http_client:
            try:
                resp = await self._http_client.request(
                    "DELETE",
                    f"{self.api_url}/tasks/{short_id}",
                    headers=self._api_headers,
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    chat.write(Text(f"  ✅ Tâche #{data.get('short_id', short_id)} supprimée.", style="green"))
                else:
                    chat.write(Text(f"  ⚠ Tâche '{short_id}' non trouvée.", style="yellow"))
            except Exception as e:
                chat.write(Text(f"  Erreur: {e}", style="red"))
        elif self.vox and self.vox.memory and self.vox.memory.is_initialized:
            registry = self.vox.memory.task_registry
            if registry:
                task = registry.delete_task(short_id)
                if task:
                    chat.write(Text(f"  ✅ Tâche #{task.short_id} supprimée : {task.description[:50]}", style="green"))
                else:
                    chat.write(Text(f"  ⚠ Tâche '{short_id}' non trouvée.", style="yellow"))
                return
        else:
            chat.write(Text("  ⚠ Memory non disponible", style="yellow"))

    async def _cmd_epics_delete(self, chat: RichLog, short_id: str) -> None:
        """Commande /project delete {id} — supprime un projet et ses tâches."""
        if self.mode == "daemon" and self._http_client:
            try:
                resp = await self._http_client.request(
                    "DELETE",
                    f"{self.api_url}/project/{short_id}",
                    headers=self._api_headers,
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    chat.write(Text(
                        f"  ✅ Projet #{data.get('short_id', short_id)} supprimé "
                        f"({data.get('tasks_deleted', 0)} tâches liées).",
                        style="green",
                    ))
                else:
                    chat.write(Text(f"  ⚠ Projet '{short_id}' non trouvé.", style="yellow"))
            except Exception as e:
                chat.write(Text(f"  Erreur: {e}", style="red"))
        elif self.vox and self.vox.memory and self.vox.memory.is_initialized:
            registry = self.vox.memory.task_registry
            if registry:
                epic, tasks_deleted = registry.delete_epic(short_id)
                if epic:
                    chat.write(Text(
                        f"  ✅ Projet #{epic.short_id} '{epic.display_name[:40]}' supprimé "
                        f"({tasks_deleted} tâches liées).",
                        style="green",
                    ))
                else:
                    chat.write(Text(f"  ⚠ Projet '{short_id}' non trouvé.", style="yellow"))
                return
        else:
            chat.write(Text("  ⚠ Memory non disponible", style="yellow"))

    def _cmd_heartbeat(self, chat: RichLog) -> None:
        if self._heartbeat_manager:
            status = self._heartbeat_manager.get_status()
            report = self._heartbeat_manager.get_progress_report()
            hb_state = "actif" if status["running"] else "inactif"
            lines = [
                f"[bold]Heartbeat[/bold] — {'[green]actif[/green]' if status['running'] else '[red]inactif[/red]'}",
                f"  Pulses: {status['pulse_count']} | Intervalle: {status['interval']:.0f}s",
                f"  Dernier événement: {status['last_event']}",
                "",
                report,
            ]
            chat.write(Panel("\n".join(lines), title="[bold cyan]Heartbeat[/bold cyan]", border_style="cyan"))
        else:
            chat.write(Text("  ⚠ Heartbeat non disponible en mode daemon", style="yellow"))

    def _cmd_skills(self, chat: RichLog) -> None:
        if not self.vox or not self.vox.memory or not self.vox.memory.is_initialized:
            chat.write(Text("  ⚠ Non disponible", style="yellow"))
            return
        report = self.vox.memory.get_skills_report()
        lines = [f"[bold]Compétences[/bold] ({report['total_skills']})\n"]
        if report["skills"]:
            for s in report["skills"][:15]:
                lines.append(f"  [green]✓[/green] {s['name']} [dim]({s['worker_type']})[/dim]")
        else:
            lines.append("  [dim]Aucune compétence acquise.[/dim]")
        chat.write(Panel("\n".join(lines), title="[bold cyan]Skills[/bold cyan]", border_style="cyan"))

    def _cmd_persona(self, chat: RichLog) -> None:
        if not self.vox or not self.vox.memory or not self.vox.memory.persona_engine:
            chat.write(Text("  ⚠ Non disponible", style="yellow"))
            return
        persona = self.vox.memory.get_neo_persona()
        if not persona:
            chat.write(Text("  ⚠ Persona non initialisée", style="yellow"))
            return
        lines = [f"[bold]Identité de Neo[/bold]\n"]
        if persona.get("commandments"):
            lines.append("[bold cyan]Commandements :[/bold cyan]")
            for cmd in persona["commandments"]:
                lines.append(f"  ⚡ {cmd['french']}")
        if persona.get("traits"):
            lines.append(f"\n[bold cyan]Traits :[/bold cyan]")
            for name, trait in persona["traits"].items():
                bar_len = 15
                filled = int(bar_len * trait["value"])
                bar = "█" * filled + "░" * (bar_len - filled)
                lines.append(f"  {name:20} [{bar}] {trait['value']:.2f}")
        chat.write(Panel("\n".join(lines), title="[bold cyan]Neo Persona[/bold cyan]", border_style="cyan"))

    def _cmd_profile(self, chat: RichLog) -> None:
        if not self.vox or not self.vox.memory or not self.vox.memory.persona_engine:
            chat.write(Text("  ⚠ Non disponible", style="yellow"))
            return
        profile = self.vox.memory.get_user_profile()
        if not profile:
            chat.write(Text("  ⚠ Profil non initialisé", style="yellow"))
            return
        prefs = profile.get("preferences", {})
        lines = [
            f"[bold]Profil Utilisateur[/bold]\n",
            f"  Langue      : {prefs.get('language', 'auto')}",
            f"  Longueur    : {prefs.get('response_length', 'medium')}",
            f"  Niveau tech : {prefs.get('technical_level', 'intermediate')}",
            f"  Ton         : {prefs.get('tone', 'professional')}",
        ]
        chat.write(Panel("\n".join(lines), title="[bold cyan]User Profile[/bold cyan]", border_style="cyan"))

    def _cmd_history(self, chat: RichLog) -> None:
        if not self.vox:
            chat.write(Text("  ⚠ Non disponible", style="yellow"))
            return
        info = self.vox.get_session_info()
        if not info:
            chat.write(Text("  ⚠ Pas de session active", style="yellow"))
            return
        lines = [f"[bold]Session active[/bold]\n"]
        lines.append(f"  ID       : {info.get('session_id', '?')}")
        lines.append(f"  Début    : {info.get('started_at', '?')}")
        lines.append(f"  Messages : {info.get('message_count', 0)}")
        chat.write(Panel("\n".join(lines), title="[bold cyan]Historique[/bold cyan]", border_style="cyan"))

    def _cmd_sessions(self, chat: RichLog) -> None:
        if not self.vox:
            chat.write(Text("  ⚠ Non disponible", style="yellow"))
            return
        try:
            sessions = self.vox.get_recent_sessions(limit=10)
            if not sessions:
                chat.write(Text("  Aucune session précédente.", style="dim"))
                return
            table = Table(title="Sessions récentes", title_style="bold cyan", border_style="cyan")
            table.add_column("ID", style="dim", width=12)
            table.add_column("Début", width=19)
            table.add_column("Msgs", justify="center", width=5)
            for s in sessions:
                table.add_row(
                    s.get("session_id", "?")[:10],
                    s.get("started_at", "?")[:19],
                    str(s.get("message_count", 0)),
                )
            chat.write(table)
        except Exception as e:
            chat.write(Text(f"  Erreur: {e}", style="red"))

    async def _cmd_restart(self) -> None:
        from neo_core.infra.guardian import EXIT_CODE_RESTART
        await self._cleanup()
        sys.exit(EXIT_CODE_RESTART)

    async def _cmd_reflect(self, chat: RichLog) -> None:
        if not self.vox or not self.vox.memory:
            chat.write(Text("  ⚠ Non disponible", style="yellow"))
            return
        chat.write(Text("  Lancement de l'auto-réflexion...", style="dim"))
        try:
            result = await self.vox.memory.perform_self_reflection()
            if result.get("success"):
                chat.write(Text(
                    f"  ✓ Réflexion : {result.get('traits_updated', 0)} traits, "
                    f"{result.get('observations_recorded', 0)} observations",
                    style="green",
                ))
            else:
                chat.write(Text(f"  ⚠ {result.get('reason', 'Erreur')}", style="yellow"))
        except Exception as e:
            chat.write(Text(f"  Erreur: {e}", style="red"))

    # ── Cleanup ──────────────────────────────────────────────

    async def _cleanup(self) -> None:
        """Nettoyage avant exit."""
        if self._heartbeat_manager:
            try:
                self._heartbeat_manager.stop()
            except Exception:
                pass
        if self._http_client:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
        if self._queue_watcher_task:
            self._queue_watcher_task.cancel()

    # ── Actions (bindings) ───────────────────────────────────

    async def action_quit_app(self) -> None:
        await self._cleanup()
        self.exit()

    async def action_refresh_sidebar(self) -> None:
        await self._periodic_sidebar_refresh()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _get_api_key(config: NeoConfig) -> str:
    """Récupère la clé API pour se connecter au daemon."""
    import os
    key = getattr(config.llm, "api_key", None) or ""
    if not key:
        key = os.environ.get("NEO_API_KEY", "")
    if not key:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
    return key


def run_tui():
    """
    Point d'entrée du TUI.

    Auto-détecte si le daemon tourne (mode API) ou non (mode local).
    Lance l'application Textual.
    """
    from rich.console import Console
    console = Console()
    config = NeoConfig()

    # Check installation
    from neo_core.vox.cli.chat import check_installation
    if not check_installation(config):
        return

    # Détecter le daemon
    daemon_running = False
    api_url = "http://localhost:8000"
    try:
        from neo_core.infra.daemon import is_running
        if is_running():
            daemon_running = True
        else:
            import httpx
            try:
                resp = httpx.get(f"{api_url}/health", timeout=5.0)
                if resp.status_code == 200:
                    daemon_running = True
            except Exception:
                pass
    except Exception:
        pass

    if daemon_running:
        # Attendre que l'API soit prête
        console.print("[dim]  Daemon détecté — connexion...[/dim]", end="")
        import httpx
        import time
        api_ready = False
        for _ in range(24):
            try:
                resp = httpx.get(f"{api_url}/health", timeout=5.0)
                if resp.status_code == 200:
                    api_ready = True
                    break
            except Exception:
                pass
            time.sleep(5)
            console.print(".", end="")

        if api_ready:
            console.print(f"\r[dim]  Daemon prêt — lancement TUI...                    [/dim]")
            app = NeoTUI(config=config, mode="daemon", api_url=api_url)
            app.run()
        else:
            console.print(f"\r[yellow]  ⚠ API pas prête — mode local.                      [/yellow]")
            from neo_core.vox.cli.chat import bootstrap
            vox = bootstrap()
            app = NeoTUI(config=config, mode="local", vox=vox)
            app.run()
    else:
        console.print("[dim]  Mode local — lancement TUI...[/dim]")
        from neo_core.vox.cli.chat import bootstrap
        vox = bootstrap()
        app = NeoTUI(config=config, mode="local", vox=vox)
        app.run()
