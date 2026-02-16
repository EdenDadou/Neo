"""
Neo Core — Status : Health Check rapide
=========================================
Affiche un dashboard de santé du système.

Usage : neo status
"""

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from neo_core.config import NeoConfig

console = Console()

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = PROJECT_ROOT / "data" / "neo_config.json"
ENV_FILE = PROJECT_ROOT / ".env"


def run_status():
    """Affiche le statut complet du système."""
    console.print("\n[bold cyan]  Neo Core — Status[/bold cyan]\n")

    # ─── Config ────────────────────────────────────────────────
    config_ok = CONFIG_FILE.exists()
    env_ok = ENV_FILE.exists()

    table = Table(title="Configuration", show_header=False, border_style="dim")
    table.add_column("Élément", style="bold")
    table.add_column("Statut")

    table.add_row(
        "neo_config.json",
        f"[green]✓ Trouvé[/green]" if config_ok else "[red]✗ Manquant[/red]",
    )
    table.add_row(
        ".env",
        f"[green]✓ Trouvé[/green]" if env_ok else "[red]✗ Manquant[/red]",
    )

    if config_ok:
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            table.add_row("Core Name", f"[cyan]{cfg.get('core_name', '?')}[/cyan]")
            table.add_row("User Name", f"[cyan]{cfg.get('user_name', '?')}[/cyan]")
            table.add_row("Version", f"[dim]{cfg.get('version', '?')}[/dim]")
            table.add_row("Stage", f"[dim]{cfg.get('stage', '?')}[/dim]")
        except Exception as e:
            table.add_row("Erreur lecture", f"[red]{e}[/red]")

    console.print(table)

    if not config_ok:
        console.print(
            "\n[yellow]  ⚠ Setup non effectué.[/yellow]"
            "\n[dim]  Lancez : neo setup[/dim]\n"
        )
        return

    # ─── Test des agents ───────────────────────────────────────
    console.print()

    try:
        config = NeoConfig()

        from neo_core.core.memory_agent import MemoryAgent
        memory = MemoryAgent(config=config)
        memory.initialize()
        memory_status = "[green]✓ Opérationnel[/green]"
        stats = memory.get_stats()
        memory_detail = f"[dim]{stats.get('total_entries', 0)} souvenirs[/dim]"
    except Exception as e:
        memory_status = f"[red]✗ Erreur[/red]"
        memory_detail = f"[dim]{e}[/dim]"

    try:
        from neo_core.core.brain import Brain
        brain = Brain(config=config)
        brain.connect_memory(memory)
        brain_status = "[green]✓ Opérationnel[/green]"
        auth_method = getattr(brain, "_auth_method", "unknown")
        auth_labels = {
            "oauth_bearer": "OAuth Bearer",
            "converted_api_key": "API Key (converti)",
            "langchain": "API Key classique",
        }
        brain_detail = f"[dim]{auth_labels.get(auth_method, auth_method)}[/dim]"

        if config.is_mock_mode():
            brain_status = "[yellow]⚠ Mode mock[/yellow]"
            brain_detail = "[dim]Réponses simulées[/dim]"
    except Exception as e:
        brain_status = f"[red]✗ Erreur[/red]"
        brain_detail = f"[dim]{e}[/dim]"

    try:
        from neo_core.core.vox import Vox
        vox = Vox(config=config)
        vox.connect(brain=brain, memory=memory)
        vox_status = "[green]✓ Opérationnel[/green]"
        vox_detail = "[dim]Interface active[/dim]"
    except Exception as e:
        vox_status = f"[red]✗ Erreur[/red]"
        vox_detail = f"[dim]{e}[/dim]"

    agents_table = Table(title="Agents", show_header=True, border_style="dim")
    agents_table.add_column("Agent", style="bold")
    agents_table.add_column("Modèle", style="cyan")
    agents_table.add_column("Statut")
    agents_table.add_column("Détails")

    # Récupérer les infos modèle
    from neo_core.config import get_agent_model
    vox_model = get_agent_model("vox").model.split("-")[1] if get_agent_model("vox") else "?"
    brain_model = get_agent_model("brain").model.split("-")[1] if get_agent_model("brain") else "?"
    memory_model = get_agent_model("memory").model.split("-")[1] if get_agent_model("memory") else "?"

    agents_table.add_row("Vox", f"[cyan]{vox_model}[/cyan]", vox_status, vox_detail)
    agents_table.add_row("Brain", f"[cyan]{brain_model}[/cyan]", brain_status, brain_detail)
    agents_table.add_row("Memory", f"[cyan]{memory_model}[/cyan]", memory_status, memory_detail)

    console.print(agents_table)

    # ─── Worker Models ─────────────────────────────────────────
    console.print()
    worker_table = Table(title="Worker Models", show_header=True, border_style="dim")
    worker_table.add_column("Type", style="bold")
    worker_table.add_column("Modèle", style="cyan")
    worker_table.add_column("Rôle")

    worker_types = {
        "researcher": "Recherche web, données temps réel",
        "coder": "Code, debug, développement",
        "analyst": "Analyse de données, tendances",
        "summarizer": "Résumé, synthèse",
        "writer": "Rédaction, contenu",
        "translator": "Traduction",
        "generic": "Tâches générales",
    }
    for wtype, role in worker_types.items():
        wmodel = get_agent_model(f"worker:{wtype}")
        model_name = wmodel.model.split("-")[1] if wmodel else "?"
        worker_table.add_row(wtype, f"[cyan]{model_name}[/cyan]", f"[dim]{role}[/dim]")

    console.print(worker_table)

    # ─── Worker Lifecycle ──────────────────────────────────────
    console.print()
    try:
        wm_stats = brain.worker_manager.get_stats()
        lifecycle_table = Table(title="Worker Lifecycle", show_header=False, border_style="dim")
        lifecycle_table.add_column("Métrique", style="bold")
        lifecycle_table.add_column("Valeur")

        active_count = wm_stats.get("active_count", 0)
        active_color = "yellow" if active_count > 0 else "green"
        lifecycle_table.add_row(
            "Workers actifs",
            f"[{active_color}]{active_count}[/{active_color}]"
        )
        lifecycle_table.add_row("Total créés", str(wm_stats.get("total_created", 0)))
        lifecycle_table.add_row("Total nettoyés", str(wm_stats.get("total_cleaned", 0)))

        leaked = wm_stats.get("leaked", 0)
        if leaked > 0:
            lifecycle_table.add_row("⚠ Fuites mémoire", f"[red]{leaked}[/red]")

        console.print(lifecycle_table)

        # Historique des derniers Workers
        history = brain.get_worker_history(limit=5)
        if history:
            hist_table = Table(title="Derniers Workers", show_header=True, border_style="dim")
            hist_table.add_column("ID", style="dim")
            hist_table.add_column("Type", style="bold")
            hist_table.add_column("État")
            hist_table.add_column("Durée")
            hist_table.add_column("Tâche")

            for w in reversed(history):
                state = w.get("state", "?")
                state_color = {
                    "closed": "green",
                    "completed": "cyan",
                    "failed": "red",
                    "running": "yellow",
                }.get(state, "dim")

                lifetime = w.get("lifetime", 0)
                success = w.get("success")
                result_icon = "✓" if success else "✗" if success is False else "?"

                hist_table.add_row(
                    w.get("worker_id", "?"),
                    w.get("worker_type", "?"),
                    f"[{state_color}]{state} {result_icon}[/{state_color}]",
                    f"{lifetime:.1f}s",
                    (w.get("task", "")[:40] + "...") if len(w.get("task", "")) > 40 else w.get("task", ""),
                )

            console.print()
            console.print(hist_table)

    except Exception:
        pass

    # ─── Health Monitor (si disponible) ────────────────────────
    try:
        health = brain.get_system_health()
        if health:
            console.print()
            health_table = Table(title="Health Monitor", show_header=False, border_style="dim")
            health_table.add_column("Métrique", style="bold")
            health_table.add_column("Valeur")

            status_color = "green" if health.get("status") == "healthy" else "yellow"
            health_table.add_row("État", f"[{status_color}]{health.get('status', '?')}[/{status_color}]")
            health_table.add_row("Appels API", str(health.get("total_calls", 0)))
            health_table.add_row("Taux d'erreur", f"{health.get('error_rate', 0):.1%}")
            health_table.add_row("Temps moyen", f"{health.get('avg_response_time', 0):.2f}s")
            health_table.add_row("Circuit Breaker", health.get("circuit_state", "?"))

            console.print(health_table)
    except Exception:
        pass

    # ─── Providers LLM ──────────────────────────────────────
    console.print()
    try:
        from neo_core.providers.bootstrap import bootstrap_providers, get_provider_summary

        registry = bootstrap_providers(config)
        summary = get_provider_summary(registry)

        prov_table = Table(title="Providers LLM", show_header=True, border_style="dim")
        prov_table.add_column("Provider", style="bold")
        prov_table.add_column("Type")
        prov_table.add_column("Modèles")
        prov_table.add_column("Statut")

        for name, info in summary["providers"].items():
            type_labels = {
                "local": "[green]Local[/green]",
                "cloud_free": "[cyan]Cloud gratuit[/cyan]",
                "cloud_paid": "[yellow]Cloud payant[/yellow]",
            }
            type_label = type_labels.get(info["type"], info["type"])
            avail = info["models_available"]
            total = info["models_total"]

            if avail > 0:
                status_str = f"[green]✓ {avail}/{total} disponibles[/green]"
            else:
                status_str = f"[red]✗ 0/{total}[/red]"

            prov_table.add_row(name.capitalize(), type_label, str(total), status_str)

        if not summary["providers"]:
            prov_table.add_row(
                "[dim]Aucun[/dim]",
                "[dim]—[/dim]",
                "[dim]0[/dim]",
                "[yellow]Mode mock (Anthropic fallback)[/yellow]",
            )

        console.print(prov_table)

        # Routing actuel
        if summary["routing"]:
            console.print()
            route_table = Table(title="Routing actuel", show_header=True, border_style="dim")
            route_table.add_column("Agent", style="bold")
            route_table.add_column("Modèle", style="cyan")

            for agent, model_id in summary["routing"].items():
                route_table.add_row(agent, f"[cyan]{model_id}[/cyan]")

            console.print(route_table)

    except Exception:
        console.print("[dim]  Providers : non configurés[/dim]")

    console.print(
        f"\n[dim]  Pour lancer le chat : neo chat[/dim]\n"
    )
