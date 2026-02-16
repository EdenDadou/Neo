"""
Neo Core — History : Afficher les sessions de conversation
============================================================

Commande `neo history` pour lister les sessions enregistrées.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from neo_core.memory.conversation import ConversationStore
from pathlib import Path

console = Console()


def run_history():
    """Affiche les sessions de conversation enregistrées."""
    db_path = Path("data/memory/conversations.db")

    if not db_path.exists():
        console.print("\n  [yellow]Aucune session enregistrée.[/yellow]\n")
        return

    try:
        store = ConversationStore(db_path)
        sessions = store.get_sessions(limit=50)

        if not sessions:
            console.print("\n  [yellow]Aucune session enregistrée.[/yellow]\n")
            return

        # Créer une table
        table = Table(title="Sessions de conversation", border_style="cyan")
        table.add_column("Session ID", style="cyan", width=12)
        table.add_column("Utilisateur", style="green")
        table.add_column("Messages", justify="right")
        table.add_column("Créée", style="dim")
        table.add_column("Mise à jour", style="dim")

        for session in sessions:
            created = session.created_at.split("T")[0] if "T" in session.created_at else session.created_at
            updated = session.updated_at.split("T")[1][:8] if "T" in session.updated_at else session.updated_at
            table.add_row(
                session.session_id[:8] + "...",
                session.user_name,
                str(session.message_count),
                created,
                updated,
            )

        console.print("\n")
        console.print(table)
        console.print(f"\n  [dim]Total : {len(sessions)} sessions[/dim]\n")

        store.close()

    except Exception as e:
        console.print(f"\n  [red]Erreur : {e}[/red]\n")
