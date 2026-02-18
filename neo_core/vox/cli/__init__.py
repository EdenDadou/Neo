"""
Neo Core — CLI : Point d'entrée principal
===========================================
Commande `neo` installée via pip install -e .

Usage :
    neo setup     Onboarding complet
    neo chat      Lancer le chat
    neo status    Health check
    neo start     Démarrer le daemon
    neo stop      Arrêter le daemon
"""

import sys

CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_usage():
    print(f"""
{CYAN}{BOLD}  Neo Core — CLI{RESET}

  {BOLD}Usage :{RESET}
    neo {CYAN}<commande>{RESET}

  {BOLD}Commandes :{RESET}
    {CYAN}setup{RESET}             Onboarding complet (install deps + config + lance le chat)
    {CYAN}setup --auto{RESET}      Installation rapide (minimum de questions)
    {CYAN}chat{RESET}              Lancer le chat directement
    {CYAN}api{RESET}               Lancer le serveur REST API

  {BOLD}Daemon :{RESET}
    {CYAN}start{RESET}             Démarrer Neo en arrière-plan (heartbeat + API)
    {CYAN}start --foreground{RESET} Démarrer au premier plan (bloquant)
    {CYAN}stop{RESET}              Arrêter le daemon
    {CYAN}restart{RESET}           Redémarrer le daemon
    {CYAN}status{RESET}            Afficher la santé du système + état daemon
    {CYAN}logs{RESET}              Afficher les logs du daemon

  {BOLD}Système :{RESET}
    {CYAN}update{RESET}            Mettre à jour Neo (git pull + deps + restart)
    {CYAN}guardian{RESET}          Lancer Neo avec le Guardian (auto-restart)
    {CYAN}history{RESET}           Lister les sessions de conversation
    {CYAN}providers{RESET}         Afficher les providers LLM configurés
    {CYAN}plugins{RESET}           Afficher les plugins dynamiques chargés
    {CYAN}install-service{RESET}   Générer/installer le service systemd
    {CYAN}telegram-setup{RESET}   Configurer le bot Telegram
    {CYAN}version{RESET}           Afficher la version

  {BOLD}Première utilisation :{RESET}
    {DIM}git clone https://github.com/EdenDadou/Neo.git && cd Neo{RESET}
    {DIM}pip install -e .{RESET}
    {DIM}neo setup{RESET}
""")


def main():
    """Point d'entrée CLI — commande `neo`."""
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower().strip("-")

    if command == "setup":
        from neo_core.vox.cli.setup import run_setup
        auto_mode = "--auto" in sys.argv or "-a" in sys.argv
        run_setup(auto_mode=auto_mode)

    elif command == "chat":
        from neo_core.vox.cli.chat import run_chat
        run_chat()

    elif command == "api":
        import uvicorn
        from neo_core.vox.api.server import create_app
        host = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
        print(f"\n  {CYAN}{BOLD}Neo Core API{RESET} — http://{host}:{port}")
        print(f"  {DIM}Docs: http://{host}:{port}/docs{RESET}\n")
        app = create_app()
        uvicorn.run(app, host=host, port=port)

    # ─── Daemon commands ──────────────────────────────
    elif command == "start":
        from neo_core.infra.daemon import start
        foreground = "--foreground" in sys.argv or "-f" in sys.argv
        host = "0.0.0.0"
        port = 8000
        # Parser --port et --host
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
            elif arg == "--host" and i + 1 < len(sys.argv):
                host = sys.argv[i + 1]

        if foreground:
            print(f"\n  {CYAN}{BOLD}Neo Core{RESET} — Démarrage en mode foreground")
            print(f"  {DIM}Ctrl+C pour arrêter.{RESET}\n")
        else:
            print(f"\n  {CYAN}{BOLD}Neo Core{RESET} — Démarrage du daemon...")

        result = start(foreground=foreground, host=host, port=port)
        if result["success"]:
            print(f"  {GREEN}✓{RESET} {result['message']}")
        else:
            print(f"  {RED}✗{RESET} {result['message']}")

    elif command == "stop":
        from neo_core.infra.daemon import stop
        print(f"\n  {CYAN}{BOLD}Neo Core{RESET} — Arrêt du daemon...")
        result = stop()
        if result["success"]:
            print(f"  {GREEN}✓{RESET} {result['message']}")
        else:
            print(f"  {RED}✗{RESET} {result['message']}")

    elif command == "restart":
        from neo_core.infra.daemon import restart
        print(f"\n  {CYAN}{BOLD}Neo Core{RESET} — Redémarrage du daemon...")
        result = restart()
        if result["success"]:
            print(f"  {GREEN}✓{RESET} {result['message']}")
        else:
            print(f"  {RED}✗{RESET} {result['message']}")

    elif command == "logs":
        from neo_core.infra.daemon import _get_log_file
        log_file = _get_log_file()
        if log_file.exists():
            # Afficher les N dernières lignes
            n = 50
            for i, arg in enumerate(sys.argv):
                if arg == "-n" and i + 1 < len(sys.argv):
                    n = int(sys.argv[i + 1])
            lines = log_file.read_text().splitlines()
            for line in lines[-n:]:
                print(line)
        else:
            print(f"  {DIM}Aucun log — Neo n'a jamais été lancé en daemon.{RESET}")

    elif command in ("install-service", "install_service"):
        from neo_core.infra.daemon import install_service, generate_systemd_service
        if "--dry-run" in sys.argv:
            print(generate_systemd_service())
        else:
            result = install_service()
            if result["success"]:
                print(f"  {GREEN}✓{RESET} {result['message']}")
            else:
                print(f"  {RED}✗{RESET} {result['message']}")
            print(f"\n  {BOLD}Commandes :{RESET}")
            for cmd in result.get("commands", []):
                print(f"    {DIM}{cmd}{RESET}")

    # ─── Existing commands ────────────────────────────
    elif command == "status":
        from neo_core.vox.cli.status import run_status
        from neo_core.infra.daemon import get_status
        run_status()
        # Ajouter le statut daemon
        daemon = get_status()
        if daemon["running"]:
            uptime = daemon.get("uptime_seconds", 0)
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            print(f"\n  {GREEN}●{RESET} Daemon actif — PID {daemon['pid']} — uptime {hours}h{minutes:02d}m — {daemon.get('memory_mb', '?')} MB RAM")
        else:
            print(f"\n  {DIM}○ Daemon inactif — `neo start` pour lancer{RESET}")

    elif command == "history":
        from neo_core.vox.cli.history import run_history
        run_history()

    elif command in ("help", "h"):
        print_usage()

    elif command == "guardian":
        from neo_core.infra.guardian import Guardian, GuardianConfig
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
        print(f"\n  {CYAN}{BOLD}Neo Guardian{RESET} — Surveillance active")
        print(f"  {DIM}Neo sera relancé automatiquement en cas de crash.{RESET}")
        print(f"  {DIM}Ctrl+C pour arrêter le Guardian.{RESET}\n")
        guardian = Guardian(GuardianConfig())
        guardian.run()

    elif command == "providers":
        from neo_core.brain.providers.bootstrap import bootstrap_providers, get_provider_summary
        from neo_core.config import NeoConfig
        from rich.console import Console as RichConsole
        from rich.table import Table as RichTable
        rc = RichConsole()
        config = NeoConfig()
        registry = bootstrap_providers(config)
        summary = get_provider_summary(registry)
        rc.print(f"\n[bold cyan]  Neo Core — Providers LLM[/bold cyan]\n")
        for name, info in summary["providers"].items():
            rc.print(f"  [bold]{name}[/bold] ({info['type']}) — {info['models_available']}/{info['models_total']} modèles")
            for m in info["models"]:
                status_icon = "✓" if m["status"] == "available" else "✗" if m["status"] == "failed" else "?"
                latency = f" ({m['latency_ms']:.0f}ms)" if m["latency_ms"] else ""
                rc.print(f"    [{status_icon}] {m['display_name']} — {m['capability']}{latency}")
        if not summary["providers"]:
            rc.print("  [dim]Aucun provider configuré — mode mock[/dim]")
        rc.print()

    elif command in ("telegram-setup", "telegram_setup"):
        from neo_core.vox.integrations.telegram import save_telegram_config
        from neo_core.config import NeoConfig
        cfg = NeoConfig()

        print(f"\n  {CYAN}{BOLD}Neo Core — Configuration Telegram{RESET}\n")
        print(f"  {DIM}1. Ouvrez Telegram et cherchez @BotFather{RESET}")
        print(f"  {DIM}2. Envoyez /newbot et suivez les instructions{RESET}")
        print(f"  {DIM}3. Copiez le token du bot ici{RESET}\n")

        token = input(f"  {GREEN}▸{RESET} Token du bot : ").strip()
        if not token:
            print(f"  {RED}✗ Token requis{RESET}")
            return

        print(f"\n  {DIM}Pour trouver votre user_id Telegram :{RESET}")
        print(f"  {DIM}  → Envoyez /start à @userinfobot{RESET}\n")

        ids_input = input(f"  {GREEN}▸{RESET} User IDs autorisés (séparés par des virgules) : ").strip()
        try:
            user_ids = [int(x.strip()) for x in ids_input.split(",") if x.strip()]
        except ValueError:
            print(f"  {RED}✗ IDs invalides (doivent être des nombres){RESET}")
            return

        if not user_ids:
            print(f"  {RED}✗ Au moins un user_id requis{RESET}")
            return

        save_telegram_config(cfg.data_dir, token, user_ids)
        print(f"\n  {GREEN}✓{RESET} Token chiffré dans le vault")
        print(f"  {GREEN}✓{RESET} {len(user_ids)} utilisateur(s) autorisé(s)")
        print(f"\n  {DIM}Redémarrez le daemon pour activer Telegram :{RESET}")
        print(f"  {CYAN}neo restart{RESET}\n")

    elif command == "plugins":
        from neo_core.brain.tools.plugin_loader import PluginLoader
        from neo_core.config import NeoConfig
        from rich.console import Console as RichConsole
        from rich.table import Table as RichTable

        cfg = NeoConfig()
        plugins_dir = cfg.data_dir / "plugins"

        rc = RichConsole()
        rc.print(f"\n[bold cyan]  Neo Core — Plugins Dynamiques[/bold cyan]\n")

        if not plugins_dir.exists():
            rc.print(f"  [dim]Aucun répertoire de plugins ({plugins_dir})[/dim]")
            rc.print()
            return

        loader = PluginLoader(plugins_dir)
        result = loader.discover()

        if not result["loaded"] and not result["errors"]:
            rc.print(f"  [dim]Aucun plugin dans {plugins_dir}[/dim]")
            rc.print()
            return

        if result["loaded"]:
            rc.print(f"  [bold green]Plugins chargés ({len(result['loaded'])}):[/bold green]")
            plugins_list = loader.list_plugins()
            for plugin in plugins_list:
                rc.print(f"    [cyan]{plugin['name']}[/cyan] v{plugin['version']}")
                rc.print(f"      {plugin['description']}")
                rc.print(f"      Workers: {', '.join(plugin['worker_types'])}")
            rc.print()

        if result["errors"]:
            rc.print(f"  [bold red]Erreurs de chargement:[/bold red]")
            for name, error in result["errors"].items():
                rc.print(f"    [red]✗[/red] {name}: {error}")
            rc.print()

    elif command == "update":
        import subprocess
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent.parent

        print(f"\n  {CYAN}{BOLD}Neo Core — Mise à jour{RESET}\n")

        # 1. Git pull
        print(f"  {DIM}⧗ Récupération de la dernière version...{RESET}", end="", flush=True)
        try:
            # Ajouter safe.directory au cas où
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", str(project_root)],
                cwd=str(project_root), capture_output=True, timeout=10,
            )
            result = subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=str(project_root), capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                if "Already up to date" in output or "Already up-to-date" in output:
                    print(f"\r  {GREEN}✓{RESET} Déjà à jour                    ")
                else:
                    print(f"\r  {GREEN}✓{RESET} Code mis à jour                 ")
                    # Afficher les fichiers changés
                    for line in output.splitlines():
                        if line.strip():
                            print(f"    {DIM}{line.strip()}{RESET}")
            else:
                print(f"\r  {RED}✗{RESET} git pull échoué : {result.stderr[:100]}")
                sys.exit(1)
        except Exception as e:
            print(f"\r  {RED}✗{RESET} Erreur git : {e}")
            sys.exit(1)

        # 2. Pip install (met à jour les deps si pyproject.toml a changé)
        print(f"  {DIM}⧗ Mise à jour des dépendances...{RESET}", end="", flush=True)
        try:
            pip_result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", ".", "-q", "--no-cache-dir"],
                cwd=str(project_root), capture_output=True, text=True, timeout=300,
            )
            if pip_result.returncode == 0:
                print(f"\r  {GREEN}✓{RESET} Dépendances à jour              ")
            else:
                print(f"\r  {RED}✗{RESET} pip install échoué              ")
                print(f"    {DIM}{pip_result.stderr[:200]}{RESET}")
        except Exception as e:
            print(f"\r  {RED}✗{RESET} Erreur pip : {e}")

        # 3. Redémarrer le daemon/service
        print(f"  {DIM}⧗ Redémarrage de Neo...{RESET}", end="", flush=True)
        try:
            # Essayer systemd d'abord
            svc = subprocess.run(
                ["systemctl", "is-active", "neo-guardian"],
                capture_output=True, text=True, timeout=5,
            )
            if svc.stdout.strip() == "active":
                subprocess.run(
                    ["sudo", "systemctl", "daemon-reload"],
                    capture_output=True, timeout=10,
                )
                subprocess.run(
                    ["sudo", "systemctl", "restart", "neo-guardian"],
                    capture_output=True, timeout=30,
                )
                print(f"\r  {GREEN}✓{RESET} Service neo-guardian redémarré   ")
            else:
                # Fallback : daemon interne
                from neo_core.infra.daemon import restart as daemon_restart
                res = daemon_restart()
                if res["success"]:
                    print(f"\r  {GREEN}✓{RESET} Daemon redémarré                ")
                else:
                    print(f"\r  {DIM}○{RESET} Aucun daemon actif — lancez : neo start")
        except Exception:
            print(f"\r  {DIM}○{RESET} Pas de service actif — lancez : neo start")

        # 4. Afficher la version
        try:
            from importlib.metadata import version as pkg_version
            v = pkg_version("neo-core")
            print(f"\n  {GREEN}{BOLD}Neo Core v{v} — à jour !{RESET}\n")
        except Exception:
            print(f"\n  {GREEN}{BOLD}Mise à jour terminée !{RESET}\n")

    elif command == "version":
        try:
            from importlib.metadata import version as pkg_version
            print(f"Neo Core v{pkg_version('neo-core')}")
        except Exception:
            print("Neo Core (version inconnue)")

    else:
        print(f"\n  Commande inconnue : '{sys.argv[1]}'")
        print_usage()
