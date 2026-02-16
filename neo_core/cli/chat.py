"""
Neo Core — Chat : Boucle conversationnelle
============================================
Interface utilisateur avec Rich.
Bootstrap les 3 agents (Memory, Brain, Vox) puis lance la boucle de chat.

Usage : python3 neo.py chat
"""

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from neo_core.config import NeoConfig

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


def print_help():
    """Affiche les commandes disponibles."""
    console.print(Panel(
        "[bold]Commandes disponibles :[/bold]\n\n"
        "  [cyan]/help[/cyan]     — Affiche cette aide\n"
        "  [cyan]/status[/cyan]   — État des agents\n"
        "  [cyan]/health[/cyan]   — Rapport de santé détaillé\n"
        "  [cyan]/quit[/cyan]     — Quitter le chat\n",
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
        "[bold cyan]python3 neo.py setup[/bold cyan]\n"
    )

    response = console.input(
        "[bold]Lancer le setup maintenant ? [/bold][dim][O/n][/dim] "
    ).strip().lower()

    if response in ("", "o", "oui", "y", "yes"):
        from neo_core.cli.setup import run_setup
        run_setup()
        return False  # Le setup lance lui-même le chat à la fin

    return False


def bootstrap():
    """
    Initialise et connecte les 3 agents du Neo Core.
    Retourne l'agent Vox prêt à communiquer.
    """
    from neo_core.core.brain import Brain
    from neo_core.core.memory_agent import MemoryAgent
    from neo_core.core.vox import Vox

    config = NeoConfig()

    # Instanciation des 3 agents
    memory = MemoryAgent(config=config)
    memory.initialize()

    brain = Brain(config=config)
    brain.connect_memory(memory)

    vox = Vox(config=config)
    vox.connect(brain=brain, memory=memory)

    return vox


async def conversation_loop(vox):
    """Boucle principale de conversation."""
    config = vox.config

    print_banner(config)

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

    while True:
        try:
            user_input = console.input(
                f"[bold green]  {config.user_name} >[/bold green] "
            )
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]  Au revoir ![/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Commandes spéciales (avec ou sans /)
        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "quit", "exit", "q"):
            console.print("[dim]  Au revoir ![/dim]")
            break

        if cmd in ("/status", "status"):
            print_status(vox)
            continue

        if cmd in ("/health", "health"):
            print_health(vox)
            continue

        if cmd in ("/help", "help"):
            print_help()
            continue

        # Process via Vox → Brain → Vox
        try:
            with console.status("[bold cyan]  Brain analyse...[/bold cyan]"):
                response = await vox.process_message(user_input)
            console.print(f"\n  [bold cyan]Vox >[/bold cyan] {response}\n")
        except Exception as e:
            console.print(f"\n  [bold red]Erreur >[/bold red] {type(e).__name__}: {e}\n")


def run_chat():
    """Point d'entrée du chat."""
    config = NeoConfig()

    if not check_installation(config):
        return

    vox = bootstrap()
    asyncio.run(conversation_loop(vox))
