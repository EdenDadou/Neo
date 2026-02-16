"""
Neo Core — Point d'entrée principal
=====================================
Boucle conversationnelle CLI pour interagir avec le système Neo Core.
Utilise Rich pour un affichage terminal soigné.
Détecte si le wizard d'installation a été exécuté.
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


def print_status(vox: Vox):
    """Affiche le statut des agents."""
    status = vox.get_system_status()
    console.print(Panel(status, title="[bold]État du système[/bold]", border_style="dim"))


def check_installation(config: NeoConfig) -> bool:
    """Vérifie si le wizard a été exécuté, sinon propose de le lancer."""
    if config.is_installed():
        return True

    console.print(
        "\n[yellow]⚠ Neo Core n'a pas encore été configuré.[/yellow]"
    )
    console.print(
        "[dim]Lancez le wizard d'installation :[/dim] "
        "[bold cyan]python3 setup_wizard.py[/bold cyan]\n"
    )

    response = console.input(
        "[bold]Continuer sans configuration ? [/bold][dim][o/N][/dim] "
    ).strip().lower()

    return response in ("o", "oui", "y", "yes")


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
    config = vox.config

    # Vérification installation
    if not check_installation(config):
        console.print("[dim]Au revoir ![/dim]")
        return

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
        key = config.llm.api_key
        key_type = "OAuth" if key and key.startswith("sk-ant-oat") else "API Key"
        console.print(
            f"[green]  ✓ Connecté à Anthropic ({key_type})[/green]\n"
        )

    console.print(
        "[dim]  Commandes : 'quit' pour quitter, "
        "'status' pour l'état du système.[/dim]\n"
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

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]  Au revoir ![/dim]")
            break

        if user_input.lower() == "status":
            print_status(vox)
            continue

        # Process via Vox → Brain → Vox
        with console.status("[bold cyan]  Brain analyse...[/bold cyan]"):
            response = await vox.process_message(user_input)

        console.print(f"\n  [bold cyan]Vox >[/bold cyan] {response}\n")


def main():
    """Point d'entrée."""
    vox = bootstrap()
    asyncio.run(conversation_loop(vox))


if __name__ == "__main__":
    main()
