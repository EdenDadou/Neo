"""
Neo Core — Chat : Boucle conversationnelle
============================================
Interface utilisateur avec Rich.
Bootstrap les 3 agents (Memory, Brain, Vox) puis lance la boucle de chat.

Usage : neo chat
"""

import asyncio
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from neo_core.config import NeoConfig
from neo_core.infra.guardian import GracefulShutdown, StateSnapshot, EXIT_CODE_RESTART

console = Console()


def print_banner(config: NeoConfig):
    """Affiche le banner personnalisé."""
    name = config.core_name.upper()
    banner = Text()
    banner.append("╔══════════════════════════════════════════╗\n", style="bold cyan")
    banner.append(f"║  {name:^38}  ║\n", style="bold cyan")
    banner.append("║  Écosystème IA Multi-Agents              ║\n", style="cyan")
    banner.append("║                                          ║\n", style="cyan")
    banner.append("║  [Vox] Interface   [Brain] Cortex        ║\n", style="dim cyan")
    banner.append("║  [Memory] Hippocampe                     ║\n", style="dim cyan")
    banner.append("╚══════════════════════════════════════════╝", style="bold cyan")
    console.print(banner)


def print_status(vox):
    """Affiche le statut des agents."""
    status = vox.get_system_status()
    console.print(Panel(status, title="[bold]État du système[/bold]", border_style="dim"))


def print_health(vox):
    """Affiche le rapport de santé détaillé (Stage 5)."""
    if not vox.brain:
        console.print("[yellow]  ⚠ Brain non connecté[/yellow]")
        return

    try:
        health = vox.brain.get_system_health()
        lines = [
            f"[bold]Santé du système[/bold]",
            f"",
            f"  État global    : {health.get('status', 'inconnu')}",
            f"  Appels API     : {health.get('total_calls', 0)}",
            f"  Taux d'erreur  : {health.get('error_rate', 0):.1%}",
            f"  Temps moyen    : {health.get('avg_response_time', 0):.2f}s",
            f"  Circuit breaker: {health.get('circuit_state', 'inconnu')}",
        ]
        console.print(Panel(
            "\n".join(lines),
            title="[bold cyan]Health Report[/bold cyan]",
            border_style="cyan",
        ))
    except Exception as e:
        console.print(f"[red]  Erreur health check: {e}[/red]")


def print_skills(vox):
    """Affiche les compétences acquises par le système."""
    if not vox.memory or not vox.memory.is_initialized:
        console.print("[yellow]  ⚠ Memory non initialisé[/yellow]")
        return

    report = vox.memory.get_skills_report()

    lines = [f"[bold]Compétences acquises[/bold] ({report['total_skills']})\n"]

    if report["skills"]:
        for s in report["skills"][:15]:
            count = s.get("success_count", 0)
            avg_t = s.get("avg_execution_time", 0)
            lines.append(
                f"  [green]✓[/green] {s['name']} "
                f"[dim]({s['worker_type']}, ×{count}, {avg_t:.1f}s)[/dim]"
            )
    else:
        lines.append("  [dim]Aucune compétence acquise pour le moment.[/dim]")

    if report["error_patterns"]:
        lines.append(f"\n[bold]Patterns d'erreur[/bold] ({report['total_error_patterns']})\n")
        for e in report["error_patterns"][:10]:
            lines.append(
                f"  [red]✗[/red] {e['worker_type']}/{e['error_type']} "
                f"[dim](×{e['count']})[/dim]"
            )
            if e.get("avoidance_rule"):
                lines.append(f"    [dim]→ {e['avoidance_rule'][:80]}[/dim]")

    if report["performance"]:
        lines.append(f"\n[bold]Performance par worker[/bold]\n")
        for wtype, perf in report["performance"].items():
            lines.append(
                f"  [{wtype}] {perf['success_rate']} succès "
                f"({perf['total_tasks']} tâches, avg {perf['avg_time']})"
            )

    console.print(Panel(
        "\n".join(lines),
        title="[bold cyan]Skills & Learning[/bold cyan]",
        border_style="cyan",
    ))


def print_tasks(vox):
    """Affiche le registre des tâches, groupées par statut."""
    if not vox.memory or not vox.memory.is_initialized:
        console.print("[yellow]  ⚠ Memory non initialisé[/yellow]")
        return

    registry = vox.memory.task_registry
    if not registry:
        console.print("[yellow]  ⚠ TaskRegistry non disponible[/yellow]")
        return

    tasks = registry.get_all_tasks(limit=50)

    if not tasks:
        console.print(Panel(
            "[dim]Aucune tâche enregistrée.[/dim]",
            title="[bold cyan]Tâches[/bold cyan]",
            border_style="cyan",
        ))
        return

    # Grouper par statut
    in_progress = [t for t in tasks if t.status == "in_progress"]
    pending = [t for t in tasks if t.status == "pending"]
    done = [t for t in tasks if t.status == "done"]
    failed = [t for t in tasks if t.status == "failed"]

    lines = []

    if in_progress:
        lines.append("[bold yellow]🔄 En cours[/bold yellow]")
        for t in in_progress:
            lines.append(f"  {t}")
        lines.append("")

    if pending:
        lines.append("[bold blue]⏳ À faire[/bold blue]")
        for t in pending:
            lines.append(f"  {t}")
        lines.append("")

    if done:
        lines.append("[bold green]✅ Terminées[/bold green]")
        for t in done[:10]:
            lines.append(f"  {t}")
        if len(done) > 10:
            lines.append(f"  [dim]... et {len(done) - 10} autres[/dim]")
        lines.append("")

    if failed:
        lines.append("[bold red]❌ Échouées[/bold red]")
        for t in failed[:5]:
            lines.append(f"  {t}")
        if len(failed) > 5:
            lines.append(f"  [dim]... et {len(failed) - 5} autres[/dim]")
        lines.append("")

    # Résumé
    summary = registry.get_summary()
    lines.append(
        f"[dim]Total : {summary['total_tasks']} tâches, "
        f"{summary['total_epics']} projets[/dim]"
    )

    console.print(Panel(
        "\n".join(lines).rstrip(),
        title="[bold cyan]Tâches[/bold cyan]",
        border_style="cyan",
    ))


def print_epics(vox):
    """Affiche le registre des projets avec sous-tâches et barre de progression."""
    if not vox.memory or not vox.memory.is_initialized:
        console.print("[yellow]  ⚠ Memory non initialisé[/yellow]")
        return

    registry = vox.memory.task_registry
    if not registry:
        console.print("[yellow]  ⚠ TaskRegistry non disponible[/yellow]")
        return

    epics = registry.get_all_epics(limit=15)

    if not epics:
        console.print(Panel(
            "[dim]Aucun projet enregistré.[/dim]",
            title="[bold cyan]Projets[/bold cyan]",
            border_style="cyan",
        ))
        return

    status_icons = {
        "pending": "⏳",
        "in_progress": "🔄",
        "done": "✅",
        "failed": "❌",
    }

    # Séparer actifs et terminés
    active_epics = [e for e in epics if e.status in ("pending", "in_progress")]
    completed_epics = [e for e in epics if e.status in ("done", "failed")]

    lines: list[str] = []

    def _render_epic(epic, show_subtasks=True):
        """Rend un epic avec sa barre de progression et ses sous-tâches."""
        icon = status_icons.get(epic.status, "?")
        epic_tasks = registry.get_epic_tasks(epic.id)
        epic_tasks.sort(key=lambda t: t.created_at)

        done_count = sum(1 for t in epic_tasks if t.status == "done")
        failed_count = sum(1 for t in epic_tasks if t.status == "failed")
        in_progress_count = sum(1 for t in epic_tasks if t.status == "in_progress")
        total = len(epic_tasks)
        pct = (done_count * 100 // total) if total > 0 else 0

        # Titre du projet
        sid_tag = f"[bold cyan]#{epic.short_id}[/bold cyan] " if epic.short_id else ""
        lines.append(f"{icon} {sid_tag}[bold]{epic.display_name[:70]}[/bold]")

        # Barre de progression visuelle
        if total > 0:
            bar_width = 20
            filled = int(bar_width * done_count / total)
            bar = "[bold green]" + "█" * filled + "[/bold green]" + "[dim]░[/dim]" * (bar_width - filled)
            lines.append(f"  {bar} {pct}%  ({done_count}/{total} tâches)")
        else:
            lines.append("  [dim]Aucune sous-tâche[/dim]")

        if epic.strategy:
            lines.append(f"  [dim italic]{epic.strategy[:80]}[/dim italic]")

        # Sous-tâches détaillées
        if show_subtasks and epic_tasks:
            for t in epic_tasks:
                t_icon = status_icons.get(t.status, "?")
                t_sid = f"[cyan]#{t.short_id}[/cyan] " if t.short_id else ""
                # Couleur selon le statut
                if t.status == "done":
                    style_open, style_close = "[dim]", "[/dim]"
                elif t.status == "failed":
                    style_open, style_close = "[red]", "[/red]"
                elif t.status == "in_progress":
                    style_open, style_close = "[bold yellow]", "[/bold yellow]"
                else:
                    style_open, style_close = "", ""
                lines.append(
                    f"    {t_icon} {t_sid}{style_open}{t.description[:60]}{style_close}"
                    f"  [dim]{t.worker_type}[/dim]"
                )

        lines.append("")

    # Projets actifs (avec sous-tâches)
    if active_epics:
        lines.append("[bold yellow]── Projets en cours ──[/bold yellow]\n")
        for epic in active_epics:
            _render_epic(epic, show_subtasks=True)

    # Projets terminés (avec sous-tâches résumées)
    if completed_epics:
        lines.append("[bold green]── Projets terminés ──[/bold green]\n")
        for epic in completed_epics[:5]:
            _render_epic(epic, show_subtasks=True)
        if len(completed_epics) > 5:
            lines.append(f"  [dim]... et {len(completed_epics) - 5} autres projets terminés[/dim]\n")

    console.print(Panel(
        "\n".join(lines).rstrip(),
        title="[bold cyan]Projets[/bold cyan]",
        border_style="cyan",
    ))


def reset_tasks(vox):
    """Supprime toutes les tâches standalone."""
    if not vox.memory or not vox.memory.is_initialized:
        console.print("[yellow]  ⚠ Memory non initialisé[/yellow]")
        return
    registry = vox.memory.task_registry
    if not registry:
        console.print("[yellow]  ⚠ TaskRegistry non disponible[/yellow]")
        return
    deleted = registry.reset_all_tasks()
    console.print(f"[green]  ✅ {deleted} tâche(s) supprimée(s). Registre remis à zéro.[/green]")


def reset_epics(vox):
    """Supprime tous les projets et leurs tâches liées."""
    if not vox.memory or not vox.memory.is_initialized:
        console.print("[yellow]  ⚠ Memory non initialisé[/yellow]")
        return
    registry = vox.memory.task_registry
    if not registry:
        console.print("[yellow]  ⚠ TaskRegistry non disponible[/yellow]")
        return
    deleted = registry.reset_all_epics()
    # Aussi nettoyer les CrewStates
    try:
        records = vox.memory._store.search_by_tags(["crew_state"], limit=100)
        for record in records:
            vox.memory._store.delete(record.id)
            deleted += 1
    except Exception:
        pass
    console.print(f"[green]  ✅ {deleted} entrée(s) supprimée(s). Tous les projets remis à zéro.[/green]")


def delete_task(vox, short_id: str):
    """Supprime une tâche par son short_id."""
    if not vox.memory or not vox.memory.is_initialized:
        console.print("[yellow]  ⚠ Memory non initialisé[/yellow]")
        return
    registry = vox.memory.task_registry
    if not registry:
        console.print("[yellow]  ⚠ TaskRegistry non disponible[/yellow]")
        return
    task = registry.delete_task(short_id)
    if task:
        console.print(f"[green]  ✅ Tâche #{task.short_id} supprimée : {task.description[:50]}[/green]")
    else:
        console.print(f"[yellow]  ⚠ Tâche '{short_id}' non trouvée.[/yellow]")


def delete_epic(vox, short_id: str):
    """Supprime un projet et ses tâches liées."""
    if not vox.memory or not vox.memory.is_initialized:
        console.print("[yellow]  ⚠ Memory non initialisé[/yellow]")
        return
    registry = vox.memory.task_registry
    if not registry:
        console.print("[yellow]  ⚠ TaskRegistry non disponible[/yellow]")
        return
    epic, tasks_deleted = registry.delete_epic(short_id)
    if epic:
        console.print(
            f"[green]  ✅ Projet #{epic.short_id} '{epic.display_name[:40]}' supprimé "
            f"({tasks_deleted} tâches liées).[/green]"
        )
    else:
        console.print(f"[yellow]  ⚠ Projet '{short_id}' non trouvé.[/yellow]")


def print_heartbeat(heartbeat_manager):
    """Affiche le rapport du heartbeat."""
    if not heartbeat_manager:
        console.print("[yellow]  ⚠ Heartbeat non initialisé[/yellow]")
        return

    status = heartbeat_manager.get_status()
    report = heartbeat_manager.get_progress_report()

    lines = [
        f"[bold]Heartbeat[/bold] — {'[green]actif[/green]' if status['running'] else '[red]inactif[/red]'}",
        f"  Pulses: {status['pulse_count']} | Intervalle: {status['interval']:.0f}s",
        f"  Dernier événement: {status['last_event']}",
        "",
        report,
    ]

    console.print(Panel(
        "\n".join(lines),
        title="[bold cyan]Heartbeat[/bold cyan]",
        border_style="cyan",
    ))


def print_persona(vox):
    """Affiche la personnalité actuelle de Neo."""
    if not vox.memory or not vox.memory.persona_engine:
        console.print("[yellow]  ⚠ PersonaEngine non disponible[/yellow]")
        return

    persona_data = vox.memory.get_neo_persona()
    if not persona_data:
        console.print("[yellow]  ⚠ Persona non initialisée[/yellow]")
        return

    lines = [f"[bold]Identité de Neo[/bold]\n"]

    # Commandements
    if persona_data.get("commandments"):
        lines.append("[bold cyan]Commandements (immuables) :[/bold cyan]")
        for cmd in persona_data["commandments"]:
            lines.append(f"  ⚡ {cmd['french']}  [dim]({cmd['english']})[/dim]")

    # Traits évolutifs
    if persona_data.get("traits"):
        lines.append(f"\n[bold cyan]Traits évolutifs :[/bold cyan]")
        for name, trait in persona_data["traits"].items():
            bar_len = 15
            filled = int(bar_len * trait["value"])
            bar = "█" * filled + "░" * (bar_len - filled)
            lines.append(
                f"  {name:22} [{bar}] {trait['value']:.2f} "
                f"[dim](conf: {trait['confidence']:.2f})[/dim]"
            )

    # Dernière réflexion
    if persona_data.get("last_reflection"):
        lines.append(f"\n[dim]Dernière réflexion: {persona_data['last_reflection'][:19]}[/dim]")

    console.print(Panel(
        "\n".join(lines),
        title="[bold cyan]Neo Persona[/bold cyan]",
        border_style="cyan",
    ))


def print_user_profile(vox):
    """Affiche le profil utilisateur appris par Neo."""
    if not vox.memory or not vox.memory.persona_engine:
        console.print("[yellow]  ⚠ PersonaEngine non disponible[/yellow]")
        return

    profile_data = vox.memory.get_user_profile()
    if not profile_data:
        console.print("[yellow]  ⚠ Profil non initialisé[/yellow]")
        return

    lines = [f"[bold]Profil Utilisateur[/bold]\n"]

    # Préférences
    prefs = profile_data.get("preferences", {})
    if prefs:
        lines.append("[bold cyan]Préférences détectées :[/bold cyan]")
        lines.append(f"  Langue        : {prefs.get('language', 'auto')}")
        lines.append(f"  Longueur rép. : {prefs.get('response_length', 'medium')}")
        lines.append(f"  Niveau tech.  : {prefs.get('technical_level', 'intermediate')}")
        lines.append(f"  Ton           : {prefs.get('tone', 'professional')}")

        topics = prefs.get("preferred_topics", [])
        if topics:
            lines.append(f"  Sujets        : {', '.join(topics[:5])}")

    # Patterns
    patterns = profile_data.get("patterns", {})
    if patterns:
        lines.append(f"\n[bold cyan]Patterns observés :[/bold cyan]")
        avg_len = patterns.get("average_message_length", 0)
        total = patterns.get("total_messages", 0)
        lines.append(f"  Messages total  : {total}")
        lines.append(f"  Longueur moy.   : {avg_len:.0f} caractères")

        interests = patterns.get("topic_interests", {})
        if interests:
            top = sorted(interests.items(), key=lambda x: x[1], reverse=True)[:5]
            topics_str = ", ".join(f"{t} (×{c})" for t, c in top)
            lines.append(f"  Sujets favoris  : {topics_str}")

        langs = patterns.get("languages_used", {})
        if langs:
            lines.append(f"  Langues         : {', '.join(f'{l} (×{c})' for l, c in langs.items())}")

    # Satisfaction
    observations = profile_data.get("observations", [])
    if observations:
        positive = sum(1 for o in observations if o.get("polarity") == "positive")
        negative = sum(1 for o in observations if o.get("polarity") == "negative")
        total_obs = len(observations)
        lines.append(f"\n[bold cyan]Satisfaction :[/bold cyan]")
        lines.append(f"  Score : {positive}/{total_obs} positif ({positive/max(total_obs,1):.0%})")
        if negative > 0:
            lines.append(f"  [dim]({negative} observations négatives)[/dim]")

    lines.append(f"\n[dim]Interactions: {profile_data.get('interaction_count', 0)}[/dim]")

    console.print(Panel(
        "\n".join(lines),
        title="[bold cyan]User Profile[/bold cyan]",
        border_style="cyan",
    ))


def print_history(vox, limit: int = 10):
    """Affiche les derniers messages de la conversation courante."""
    session_info = vox.get_session_info()
    if not session_info:
        console.print("[yellow]  ⚠ Pas de session active[/yellow]")
        return

    if not vox._conversation_store:
        console.print("[yellow]  ⚠ Store de conversation non disponible[/yellow]")
        return

    try:
        history = vox._conversation_store.get_history(
            session_info["session_id"],
            limit=limit,
        )
        if not history:
            console.print("[dim]  Aucun message dans cette session.[/dim]")
            return

        lines = [f"[bold]Historique - Derniers {limit} messages[/bold]\n"]
        for turn in history:
            role_label = "[bold cyan]Vous[/bold cyan]" if turn.role == "human" else "[bold yellow]Neo[/bold yellow]"
            timestamp = turn.timestamp.split("T")[1][:8] if "T" in turn.timestamp else turn.timestamp
            lines.append(f"[dim]{timestamp}[/dim] {role_label}")
            # Truncate long messages
            content = turn.content[:200] + "..." if len(turn.content) > 200 else turn.content
            lines.append(f"  {content}\n")

        console.print(Panel(
            "\n".join(lines),
            title="[bold cyan]Historique[/bold cyan]",
            border_style="cyan",
        ))
    except Exception as e:
        console.print(f"[red]  Erreur: {e}[/red]")


def print_sessions(vox):
    """Affiche les sessions récentes."""
    if not vox._conversation_store:
        console.print("[yellow]  ⚠ Store de conversation non disponible[/yellow]")
        return

    try:
        sessions = vox._conversation_store.get_sessions(limit=10)
        if not sessions:
            console.print("[dim]  Aucune session enregistrée.[/dim]")
            return

        lines = [f"[bold]Sessions récentes ({len(sessions)})[/bold]\n"]
        for session in sessions:
            created = session.created_at.split("T")[0] if "T" in session.created_at else session.created_at
            updated = session.updated_at.split("T")[1][:8] if "T" in session.updated_at else ""
            is_current = " [green]←[/green]" if session.session_id == vox.get_session_info().get("session_id") else ""
            lines.append(
                f"  [cyan]{session.session_id[:8]}...[/cyan] "
                f"{session.user_name} | {session.message_count} msgs | {created} {updated}{is_current}"
            )

        console.print(Panel(
            "\n".join(lines),
            title="[bold cyan]Historique des sessions[/bold cyan]",
            border_style="cyan",
        ))
    except Exception as e:
        console.print(f"[red]  Erreur: {e}[/red]")


def print_help():
    """Affiche les commandes disponibles."""
    console.print(Panel(
        "[bold]Commandes disponibles :[/bold]\n\n"
        "  [cyan]/help[/cyan]      — Affiche cette aide\n"
        "  [cyan]/status[/cyan]    — État des agents\n"
        "  [cyan]/health[/cyan]    — Rapport de santé détaillé\n"
        "  [cyan]/skills[/cyan]    — Compétences acquises\n"
        "  [cyan]/tasks[/cyan]     — Registre des tâches\n"
        "  [cyan]/project[/cyan]   — Projets en cours\n"
        "  [cyan]/heartbeat[/cyan] — Statut du heartbeat\n"
        "  [cyan]/persona[/cyan]   — Personnalité de Neo\n"
        "  [cyan]/profile[/cyan]   — Profil utilisateur\n"
        "  [cyan]/history[/cyan]   — Historique de la conversation\n"
        "  [cyan]/sessions[/cyan]  — Lister les sessions\n"
        "  [cyan]/reflect[/cyan]   — Force une auto-réflexion\n"
        "  [cyan]/restart[/cyan]   — Redémarrer Neo (Guardian)\n"
        "  [cyan]/quit[/cyan]      — Quitter le chat\n",
        title="[bold]Aide[/bold]",
        border_style="dim",
    ))


def check_installation(config: NeoConfig) -> bool:
    """Vérifie si le setup a été exécuté, sinon propose de le lancer."""
    if config.is_installed():
        return True

    console.print(
        "\n[yellow]⚠ Neo Core n'a pas encore été configuré.[/yellow]"
    )
    console.print(
        "[dim]Lancez le setup :[/dim] "
        "[bold cyan]neo setup[/bold cyan]\n"
    )

    response = console.input(
        "[bold]Lancer le setup maintenant ? [/bold][dim][O/n][/dim] "
    ).strip().lower()

    if response in ("", "o", "oui", "y", "yes"):
        from neo_core.vox.cli.setup import run_setup
        run_setup()
        return False  # Le setup lance lui-même le chat à la fin

    return False


def bootstrap():
    """
    Retourne l'instance unique de Vox via le CoreRegistry.

    Le registry garantit qu'il n'existe qu'une seule instance
    de Memory, Brain et Vox par processus — partagée entre
    CLI chat, daemon, API et Telegram.
    """
    from neo_core.infra.registry import core_registry

    vox = core_registry.get_vox()

    # Afficher les providers actifs (première fois uniquement)
    try:
        from neo_core.brain.providers.bootstrap import bootstrap_providers
        config = core_registry.get_config()
        registry = bootstrap_providers(config)
        configured = registry.get_configured_providers()
        if configured:
            console.print(
                f"  [dim]Providers actifs : {', '.join(configured)} "
                f"({registry.get_stats()['total_models']} modèles)[/dim]"
            )
    except Exception:
        pass

    return vox


async def _background_queue_watcher(brain_results: asyncio.Queue) -> None:
    """
    Surveille la queue Brain en continu et affiche les résultats
    dès qu'ils arrivent — même si l'utilisateur est en train de taper.

    Fonctionne grâce à patch_stdout() de prompt_toolkit qui permet
    d'imprimer au-dessus du prompt sans corrompre l'input en cours.
    """
    while True:
        try:
            while not brain_results.empty():
                result = brain_results.get_nowait()
                console.print(f"\n  [bold cyan]Vox >[/bold cyan] {result}\n")
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            # Drain remaining items before exiting
            while not brain_results.empty():
                try:
                    result = brain_results.get_nowait()
                    console.print(f"\n  [bold cyan]Vox >[/bold cyan] {result}\n")
                except asyncio.QueueEmpty:
                    break
            break
        except Exception:
            await asyncio.sleep(0.1)


async def conversation_loop(vox):
    """Boucle principale de conversation."""
    config = vox.config
    heartbeat_manager = None
    guardian_mode = os.environ.get("NEO_GUARDIAN_MODE") == "1"
    state_dir = config.data_dir / "guardian" if hasattr(config, "data_dir") else __import__("pathlib").Path("data/guardian")

    # --- Guardian : GracefulShutdown ---
    shutdown_handler = GracefulShutdown()
    shutdown_handler.set_state_dir(state_dir)
    shutdown_handler.install_handlers()

    # --- Guardian : Charger le state snapshot précédent ---
    previous_state = StateSnapshot.load(state_dir)
    recovery_message = None
    if previous_state:
        reason_labels = {
            "signal": "signal système",
            "crash": "crash",
            "user_quit": "arrêt utilisateur",
            "guardian_restart": "restart Guardian",
        }
        reason = reason_labels.get(previous_state.shutdown_reason, previous_state.shutdown_reason)
        recovery_message = (
            f"Redémarré après {reason} "
            f"(uptime précédent: {previous_state.uptime_seconds:.0f}s, "
            f"restarts: {previous_state.restart_count})"
        )
        # Nettoyer le state après chargement
        StateSnapshot.clear(state_dir)

    # --- Conversation : Démarrer ou reprendre une session ---
    if previous_state and previous_state.session_id:
        # Reprendre la session précédente
        vox.resume_session(previous_state.session_id)
    else:
        # Nouvelle session
        vox.start_new_session(config.user_name)

    # Démarrer le heartbeat en arrière-plan
    try:
        from neo_core.infra.heartbeat import HeartbeatManager, HeartbeatConfig

        def on_heartbeat_event(event):
            """Affiche les événements importants du heartbeat."""
            important = {"task_completed", "task_failed", "epic_done", "task_stale",
                         "persona_reflection", "crew_step_advanced", "orchestrator_replan"}
            if event.event_type in important:
                console.print(
                    f"\n  [dim magenta]♥ {event.message}[/dim magenta]"
                )

        heartbeat_manager = HeartbeatManager(
            brain=vox.brain,
            memory=vox.memory,
            config=HeartbeatConfig(interval_seconds=300.0),  # 5 minutes
            on_event=on_heartbeat_event,
        )
        heartbeat_manager.start()
    except Exception:
        pass

    # --- Guardian : Callbacks de cleanup ---
    def cleanup_heartbeat():
        if heartbeat_manager:
            heartbeat_manager.stop()

    shutdown_handler.add_cleanup_callback(cleanup_heartbeat)

    # ── Mode asynchrone : queue pour les résultats Brain en arrière-plan ──
    brain_results: asyncio.Queue = asyncio.Queue()

    def on_brain_done(result: str):
        """Callback appelé quand Brain termine en arrière-plan."""
        brain_results.put_nowait(result)

    vox.set_brain_done_callback(on_brain_done)

    print_banner(config)

    # Message de recovery si redémarrage
    if recovery_message:
        console.print(
            f"\n  [bold yellow]⟳ {recovery_message}[/bold yellow]"
        )

    if guardian_mode:
        console.print(
            "  [dim green]✓ Mode Guardian actif — auto-restart en cas de crash[/dim green]"
        )

    # Message d'accueil personnalisé
    console.print(
        f"\n  [bold]Bienvenue {config.user_name}[/bold] — "
        f"[dim]{config.core_name} est prêt.[/dim]\n"
    )

    if config.is_mock_mode():
        console.print(
            "[yellow]  ⚠ Mode mock actif (pas de clé API). "
            "Les réponses sont simulées.[/yellow]\n"
        )
    else:
        auth_method = getattr(vox.brain, "_auth_method", "unknown") if vox.brain else "unknown"
        auth_labels = {
            "oauth_bearer": "OAuth Bearer + beta header",
            "converted_api_key": "Clé API convertie depuis OAuth",
            "langchain": "API Key classique",
        }
        auth_label = auth_labels.get(auth_method, auth_method)
        console.print(
            f"[green]  ✓ Connecté à Anthropic ({auth_label})[/green]\n"
        )

    console.print(
        "[dim]  Tapez /help pour les commandes disponibles.[/dim]\n"
    )

    # ── prompt_toolkit : input async + affichage Brain en temps réel ──
    prompt_session = PromptSession()
    queue_watcher_task = None

    async def _cleanup_and_exit(code: int = 0):
        """Arrêt propre : cancel le watcher, stop heartbeat, exit."""
        if queue_watcher_task:
            queue_watcher_task.cancel()
            try:
                await queue_watcher_task
            except asyncio.CancelledError:
                pass
        if heartbeat_manager:
            heartbeat_manager.stop()

    with patch_stdout():
        queue_watcher_task = asyncio.create_task(
            _background_queue_watcher(brain_results)
        )

        while True:
            # ── Lecture input async (Brain peut afficher pendant qu'on tape) ──
            try:
                user_input = await prompt_session.prompt_async(
                    f"  {config.user_name} > ",
                )
            except (KeyboardInterrupt, EOFError):
                await _cleanup_and_exit()
                shutdown_handler.save_state(shutdown_reason="signal")
                shutdown_handler.clear_state()
                console.print("\n[dim]  Au revoir ![/dim]")
                sys.exit(0)

            user_input = user_input.strip()
            if not user_input:
                continue

            # Commandes spéciales (avec ou sans /)
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "quit", "exit", "q"):
                await _cleanup_and_exit()
                # Sauvegarder le state et nettoyer
                session_id = vox.get_session_info().get("session_id", "") if vox.get_session_info() else ""
                shutdown_handler.save_state(
                    shutdown_reason="user_quit",
                    turn_count=0,
                    heartbeat_pulse_count=heartbeat_manager.get_status()["pulse_count"] if heartbeat_manager else 0,
                    session_id=session_id,
                )
                shutdown_handler.clear_state()  # Quit normal → pas besoin du state
                console.print("[dim]  Au revoir ![/dim]")
                sys.exit(0)

            if cmd in ("/restart", "restart"):
                console.print("[bold yellow]  ⟳ Redémarrage de Neo...[/bold yellow]")
                await _cleanup_and_exit()
                session_id = vox.get_session_info().get("session_id", "") if vox.get_session_info() else ""
                shutdown_handler.save_state(
                    shutdown_reason="guardian_restart",
                    turn_count=0,
                    heartbeat_pulse_count=heartbeat_manager.get_status()["pulse_count"] if heartbeat_manager else 0,
                    session_id=session_id,
                )
                sys.exit(EXIT_CODE_RESTART)  # Code 42 → Guardian relance immédiatement

            if cmd in ("/status", "status"):
                print_status(vox)
                continue

            if cmd in ("/health", "health"):
                print_health(vox)
                continue

            if cmd in ("/help", "help"):
                print_help()
                continue

            if cmd in ("/history", "history"):
                print_history(vox, limit=10)
                continue

            if cmd in ("/sessions", "sessions"):
                print_sessions(vox)
                continue

            if cmd in ("/skills", "skills"):
                print_skills(vox)
                continue

            if cmd in ("/tasks reset", "tasks reset"):
                reset_tasks(vox)
                continue

            if cmd.startswith(("/tasks delete ", "tasks delete ")):
                sid = cmd.split("delete", 1)[1].strip()
                delete_task(vox, sid)
                continue

            if cmd in ("/tasks", "tasks"):
                print_tasks(vox)
                continue

            if cmd in ("/project reset", "/epics reset", "project reset"):
                reset_epics(vox)
                continue

            if cmd.startswith(("/project delete ", "/epics delete ", "project delete ")):
                sid = cmd.split("delete", 1)[1].strip()
                delete_epic(vox, sid)
                continue

            if cmd in ("/project", "/epics", "project", "epics"):
                print_epics(vox)
                continue

            if cmd in ("/heartbeat", "heartbeat"):
                print_heartbeat(heartbeat_manager)
                continue

            if cmd in ("/persona", "persona"):
                print_persona(vox)
                continue

            if cmd in ("/profile", "profile"):
                print_user_profile(vox)
                continue

            if cmd in ("/reflect", "reflect"):
                console.print("[dim]  Lancement de l'auto-réflexion...[/dim]")
                try:
                    result = await vox.memory.perform_self_reflection()
                    if result.get("success"):
                        console.print(
                            f"[green]  ✓ Réflexion effectuée : "
                            f"{result.get('traits_updated', 0)} traits ajustés, "
                            f"{result.get('observations_recorded', 0)} observations[/green]"
                        )
                        if result.get("summary"):
                            console.print(f"  [dim]{result['summary'][:200]}[/dim]")
                    else:
                        console.print(f"[yellow]  ⚠ {result.get('reason', 'Erreur')}[/yellow]")
                except Exception as e:
                    console.print(f"[red]  Erreur: {e}[/red]")
                continue

            # ── Callback ack : Vox envoie un accusé de réception pendant que Brain réfléchit ──
            ack_displayed = False

            def on_thinking(ack_text: str):
                nonlocal ack_displayed
                if not ack_displayed:
                    console.print(f"\n  [dim cyan]Vox >[/dim cyan] [dim]{ack_text}[/dim]")
                    ack_displayed = True

            vox.set_thinking_callback(on_thinking)

            # Process via Vox → Brain → Vox
            try:
                with console.status("[bold cyan]  Brain analyse...[/bold cyan]"):
                    response = await vox.process_message(user_input)

                console.print(f"\n  [bold cyan]Vox >[/bold cyan] {response}\n")
            except Exception as e:
                console.print(f"\n  [bold red]Erreur >[/bold red] {type(e).__name__}: {e}\n")


def _get_api_key(config: NeoConfig) -> str:
    """Récupère la clé API pour se connecter au daemon."""
    key = getattr(config.llm, "api_key", None) or ""
    if not key:
        key = os.environ.get("NEO_API_KEY", "")
    if not key:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
    return key


async def api_conversation_loop(config: NeoConfig, api_url: str):
    """
    Boucle de conversation via l'API du daemon.

    Mode asynchrone : envoie les messages au daemon via SSE (/chat/stream).
    L'ack arrive immédiatement, le prompt revient, et la réponse Brain
    s'affiche en arrière-plan quand elle est prête.
    L'utilisateur peut continuer à discuter pendant que Brain réfléchit.
    """
    import httpx
    import json as _json

    api_key = _get_api_key(config)
    headers = {}
    if api_key:
        headers["X-Neo-Key"] = api_key

    print_banner(config)

    console.print(
        f"\n  [bold]Bienvenue {config.user_name}[/bold] — "
        f"[dim]{config.core_name} est prêt.[/dim]"
    )
    console.print(
        f"  [dim green]✓ Connecté au daemon ({api_url})[/dim green]"
    )
    console.print(
        f"  [dim]  Conversation unifiée avec Telegram.[/dim]\n"
    )
    console.print(
        "[dim]  Tapez /help pour les commandes disponibles.[/dim]\n"
    )

    # ── Queue pour les résultats Brain en arrière-plan ──
    brain_results: asyncio.Queue = asyncio.Queue()
    brain_pending = 0  # nombre de requêtes Brain en cours

    async def _stream_chat(client: httpx.AsyncClient, message: str):
        """
        Envoie un message via SSE /chat/stream.
        Affiche l'ack dès réception, puis pousse la réponse Brain dans la queue.
        """
        nonlocal brain_pending
        brain_pending += 1
        try:
            async with client.stream(
                "POST",
                f"{api_url}/chat/stream",
                json={"message": message},
                headers=headers,
                timeout=130.0,
            ) as resp:
                if resp.status_code == 401:
                    console.print(
                        "\n[red]  ⚠ Accès refusé — clé API invalide. "
                        "Vérifiez NEO_API_KEY ou ANTHROPIC_API_KEY.[/red]\n"
                    )
                    return
                if resp.status_code == 503:
                    console.print("\n[yellow]  ⚠ Neo Core pas encore initialisé — réessayez.[/yellow]\n")
                    return
                if resp.status_code != 200:
                    console.print(f"\n[red]  Erreur API ({resp.status_code})[/red]\n")
                    return

                # Parser les events SSE
                buffer = ""
                current_event = "message"
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
                            # Afficher l'ack immédiatement
                            console.print(
                                f"\n  [dim cyan]Vox >[/dim cyan] [dim]{text}[/dim]"
                            )
                            console.print(
                                "  [bold cyan]⟳ Brain réfléchit...[/bold cyan]"
                            )
                        elif event_type == "response":
                            # Pousser la réponse dans la queue
                            brain_results.put_nowait(text)
                        elif event_type == "error":
                            console.print(f"\n  [bold red]Erreur >[/bold red] {text}\n")

        except httpx.ConnectError:
            console.print(
                "\n[red]  ⚠ Connexion au daemon perdue. "
                "Vérifiez avec: neo status[/red]\n"
            )
        except Exception as e:
            console.print(f"\n  [bold red]Erreur >[/bold red] {type(e).__name__}: {e}\n")
        finally:
            brain_pending -= 1

    # ── prompt_toolkit : input async + affichage Brain en temps réel ──
    prompt_session = PromptSession()
    queue_watcher_task = None

    async def _cleanup_api():
        """Arrêt propre : cancel le watcher."""
        if queue_watcher_task:
            queue_watcher_task.cancel()
            try:
                await queue_watcher_task
            except asyncio.CancelledError:
                pass

    async with httpx.AsyncClient(timeout=130.0) as client:
        with patch_stdout():
            queue_watcher_task = asyncio.create_task(
                _background_queue_watcher(brain_results)
            )

            while True:
                try:
                    user_input = await prompt_session.prompt_async(
                        f"  {config.user_name} > ",
                    )
                except (KeyboardInterrupt, EOFError):
                    await _cleanup_api()
                    console.print("\n[dim]  Au revoir ![/dim]")
                    sys.exit(0)

                user_input = user_input.strip()
                if not user_input:
                    continue

                cmd = user_input.lower()
                if cmd in ("/quit", "/exit", "quit", "exit", "q"):
                    await _cleanup_api()
                    console.print("[dim]  Au revoir ![/dim]")
                    sys.exit(0)

                if cmd in ("/help", "help"):
                    print_help()
                    continue

                # Commandes /status, /health → appeler les endpoints API
                if cmd in ("/status", "status"):
                    try:
                        resp = await client.get(f"{api_url}/status", headers=headers)
                        if resp.status_code == 200:
                            data = resp.json()
                            lines = [
                                f"  Core    : {data.get('core_name', '?')}",
                                f"  Status  : {data.get('status', '?')}",
                                f"  Uptime  : {data.get('uptime_seconds', 0):.0f}s",
                                f"  Guardian: {'oui' if data.get('guardian_mode') else 'non'}",
                            ]
                            if brain_pending > 0:
                                lines.append(f"  Brain   : [bold cyan]⟳ {brain_pending} requête(s) en cours[/bold cyan]")
                            agents = data.get("agents", {})
                            for name, info in agents.items():
                                lines.append(f"  {name:10}: {info}")
                            console.print(Panel("\n".join(lines), title="[bold]État du système[/bold]", border_style="dim"))
                        else:
                            console.print(f"[yellow]  ⚠ API error: {resp.status_code}[/yellow]")
                    except Exception as e:
                        console.print(f"[red]  Erreur: {e}[/red]")
                    continue

                if cmd in ("/health", "health"):
                    try:
                        resp = await client.get(f"{api_url}/health")
                        if resp.status_code == 200:
                            data = resp.json()
                            checks = data.get("checks", {})
                            lines = [f"  {k}: {v}" for k, v in checks.items()]
                            console.print(Panel("\n".join(lines), title="[bold]Health[/bold]", border_style="dim"))
                    except Exception as e:
                        console.print(f"[red]  Erreur: {e}[/red]")
                    continue

                if cmd in ("/tasks", "tasks"):
                    try:
                        resp = await client.get(f"{api_url}/tasks", headers=headers)
                        if resp.status_code == 200:
                            data = resp.json()
                            summary = data.get("summary", {})
                            lines = ["[bold]Registre des tâches[/bold]\n"]
                            if data.get("tasks"):
                                for t_str in data["tasks"][:20]:
                                    lines.append(f"  {t_str}")
                            else:
                                lines.append("  [dim]Aucune tâche enregistrée.[/dim]")
                            if summary:
                                lines.append(f"\n[bold]Résumé[/bold]")
                                lines.append(
                                    f"  Total : {summary.get('total_tasks', 0)} tâches, "
                                    f"{summary.get('total_epics', 0)} projets"
                                )
                                if summary.get("tasks_by_status"):
                                    parts = [f"{k}: {v}" for k, v in summary["tasks_by_status"].items()]
                                    lines.append(f"  Statuts : {', '.join(parts)}")
                            console.print(Panel("\n".join(lines), title="[bold cyan]Tâches[/bold cyan]", border_style="cyan"))
                        else:
                            console.print(f"[yellow]  ⚠ API error: {resp.status_code}[/yellow]")
                    except Exception as e:
                        console.print(f"[red]  Erreur: {e}[/red]")
                    continue

                if cmd in ("/project", "/epics", "project", "epics"):
                    try:
                        resp = await client.get(f"{api_url}/epics", headers=headers)
                        if resp.status_code == 200:
                            data = resp.json()
                            epics = data.get("epics", [])
                            if not epics:
                                console.print(Panel("[dim]Aucun projet actif.[/dim]", title="[bold cyan]Projets[/bold cyan]", border_style="cyan"))
                            else:
                                status_icons = {"pending": "⏳", "in_progress": "🔄", "done": "✅", "failed": "❌"}
                                table = Table(title="Projets", title_style="bold cyan", border_style="cyan", show_header=True, header_style="bold white", padding=(0, 1))
                                table.add_column("", justify="center", width=3)
                                table.add_column("ID", style="dim", width=10)
                                table.add_column("Projet", min_width=30)
                                table.add_column("Progrès", justify="center", width=12)
                                for epic in epics:
                                    icon = status_icons.get(epic.get("status", ""), "?")
                                    table.add_row(
                                        icon,
                                        epic.get("id", "")[:10],
                                        epic.get("description", "")[:50],
                                        epic.get("progress", "0/0"),
                                    )
                                console.print(table)
                        else:
                            console.print(f"[yellow]  ⚠ API error: {resp.status_code}[/yellow]")
                    except Exception as e:
                        console.print(f"[red]  Erreur: {e}[/red]")
                    continue

                # ── Envoyer le message en arrière-plan via SSE ──
                # Le prompt revient immédiatement, la réponse Brain arrive dans la queue
                asyncio.create_task(_stream_chat(client, user_input))


def run_chat():
    """
    Point d'entrée du chat — lance le TUI Textual.

    Délègue au module tui.py qui auto-détecte daemon/local.
    Les anciennes boucles conversation_loop / api_conversation_loop
    sont conservées comme fallback.
    """
    try:
        from neo_core.vox.cli.tui import run_tui
        run_tui()
    except ImportError:
        # Fallback si textual n'est pas installé
        console.print("[yellow]  ⚠ textual non installé — mode legacy.[/yellow]")
        console.print("[dim]  Installez avec : pip install 'textual>=1.0'[/dim]\n")
        _run_chat_legacy()
    except Exception as e:
        console.print(f"[red]  Erreur TUI: {e}[/red]")
        console.print("[dim]  Fallback mode legacy...[/dim]\n")
        _run_chat_legacy()


def _run_chat_legacy():
    """Fallback : ancien mode prompt_toolkit (si textual indisponible)."""
    config = NeoConfig()

    if not check_installation(config):
        return

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
        console.print("[dim]  Daemon détecté — connexion à l'API...[/dim]", end="")
        import httpx
        api_ready = False
        for _ in range(24):
            try:
                resp = httpx.get(f"{api_url}/health", timeout=5.0)
                if resp.status_code == 200:
                    api_ready = True
                    break
            except Exception:
                pass
            import time
            time.sleep(5)
            console.print(".", end="")

        if api_ready:
            console.print(f"\r[dim]  Daemon détecté — mode unifié.[/dim]")
            asyncio.run(api_conversation_loop(config, api_url))
        else:
            console.print(f"\r[yellow]  ⚠ API pas prête — mode local.[/yellow]")
            vox = bootstrap()
            asyncio.run(conversation_loop(vox))
    else:
        console.print("[dim]  Mode local.[/dim]")
        vox = bootstrap()
        asyncio.run(conversation_loop(vox))
