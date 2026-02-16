"""
Neo Core — Point d'entrée principal
=====================================
Boucle conversationnelle CLI pour interagir avec le système Neo Core.
Utilise Rich pour un affichage terminal soigné.
"""

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from neo_core.config import NeoConfig
from neo_core.core.brain import Brain
from neo_core.core.memory_agent import MemoryAgent
from neo_core.core.vox import Vox

console = Console()


def print_banner():
    """Affiche le banner Neo Core."""
    banner = Text()
    banner.append("╔══════════════════════════════════════╗\n", style="bold cyan")
    banner.append("║          NEO CORE v0.1               ║\n", style="bold cyan")
    banner.append("║   Écosystème IA Multi-Agents         ║\n", style="cyan")
    banner.append("║                                      ║\n", style="cyan")
    banner.append("║   [Vox] Interface  [Brain] Cortex    ║\n", style="dim cyan")
    banner.append("║   [Memory] Hippocampe                ║\n", style="dim cyan")
    banner.append("╚══════════════════════════════════════╝", style="bold cyan")
    console.print(banner)


def print_status(vox: Vox):
    """Affiche le statut des agents."""
    status = vox.get_system_status()
    console.print(Panel(status, title="[bold]État du système[/bold]", border_style="dim"))


def bootstrap() -> Vox:
    """
    Initialise et connecte les 3 agents du Neo Core.
    Retourne l'agent Vox prêt à communiquer.
    """
    config = NeoConfig()

    # Instanciation des 3 agents
    memory = MemoryAgent(config=config)
    memory.initialize()

    brain = Brain(config=config)
    brain.connect_memory(memory)

    vox = Vox(config=config)
    vox.connect(brain=brain, memory=memory)

    return vox


async def conversation_loop(vox: Vox):
    """Boucle principale de conversation."""
    print_banner()

    if vox.config.is_mock_mode():
        console.print(
            "[yellow]⚠ Mode mock actif (pas de clé API). "
            "Les réponses sont simulées.[/yellow]\n"
        )

    console.print("[dim]Tapez 'quit' pour quitter, 'status' pour l'état du système.[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold green]Vous >[/bold green] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Au revoir ![/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Au revoir ![/dim]")
            break

        if user_input.lower() == "status":
            print_status(vox)
            continue

        # Process via Vox → Brain → Vox
        with console.status("[bold cyan]Brain analyse...[/bold cyan]"):
            response = await vox.process_message(user_input)

        console.print(f"\n[bold cyan]Vox >[/bold cyan] {response}\n")


def main():
    """Point d'entrée."""
    vox = bootstrap()
    asyncio.run(conversation_loop(vox))


if __name__ == "__main__":
    main()
