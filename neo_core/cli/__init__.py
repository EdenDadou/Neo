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
    {CYAN}guardian{RESET}          Lancer Neo avec le Guardian (auto-restart)
    {CYAN}history{RESET}           Lister les sessions de conversation
    {CYAN}providers{RESET}         Afficher les providers LLM configurés
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
        from neo_core.cli.setup import run_setup
        run_setup()

    elif command == "chat":
        from neo_core.cli.chat import run_chat
        run_chat()

    elif command == "api":
        import uvicorn
        from neo_core.api.server import create_app
        host = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
        print(f"\n  {CYAN}{BOLD}Neo Core API{RESET} — http://{host}:{port}")
        print(f"  {DIM}Docs: http://{host}:{port}/docs{RESET}\n")
        app = create_app()
        uvicorn.run(app, host=host, port=port)

    # ─── Daemon commands ──────────────────────────────
    elif command == "start":
        from neo_core.core.daemon import start
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
        from neo_core.core.daemon import stop
        print(f"\n  {CYAN}{BOLD}Neo Core{RESET} — Arrêt du daemon...")
        result = stop()
        if result["success"]:
            print(f"  {GREEN}✓{RESET} {result['message']}")
        else:
            print(f"  {RED}✗{RESET} {result['message']}")

    elif command == "restart":
        from neo_core.core.daemon import restart
        print(f"\n  {CYAN}{BOLD}Neo Core{RESET} — Redémarrage du daemon...")
        result = restart()
        if result["success"]:
            print(f"  {GREEN}✓{RESET} {result['message']}")
        else:
            print(f"  {RED}✗{RESET} {result['message']}")

    elif command == "logs":
        from neo_core.core.daemon import _get_log_file
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
        from neo_core.core.daemon import install_service, generate_systemd_service
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
        from neo_core.cli.status import run_status
        from neo_core.core.daemon import get_status
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
        from neo_core.cli.history import run_history
        run_history()

    elif command in ("help", "h"):
        print_usage()

    elif command == "guardian":
        from neo_core.core.guardian import Guardian, GuardianConfig
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
        print(f"\n  {CYAN}{BOLD}Neo Guardian{RESET} — Surveillance active")
        print(f"  {DIM}Neo sera relancé automatiquement en cas de crash.{RESET}")
        print(f"  {DIM}Ctrl+C pour arrêter le Guardian.{RESET}\n")
        guardian = Guardian(GuardianConfig())
        guardian.run()

    elif command == "providers":
        from neo_core.providers.bootstrap import bootstrap_providers, get_provider_summary
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
        from neo_core.integrations.telegram import save_telegram_config
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

    elif command == "version":
        print("Neo Core v0.8.4")

    else:
        print(f"\n  Commande inconnue : '{sys.argv[1]}'")
        print_usage()
