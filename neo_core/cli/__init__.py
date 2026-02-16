"""
Neo Core — CLI : Point d'entrée principal
===========================================
Commande `neo` installée via pip install -e .

Usage :
    neo setup     Onboarding complet
    neo chat      Lancer le chat
    neo status    Health check
"""

import sys

CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_usage():
    print(f"""
{CYAN}{BOLD}  Neo Core — CLI{RESET}

  {BOLD}Usage :{RESET}
    neo {CYAN}<commande>{RESET}

  {BOLD}Commandes :{RESET}
    {CYAN}setup{RESET}      Onboarding complet (install deps + config + lance le chat)
    {CYAN}chat{RESET}       Lancer le chat directement
    {CYAN}status{RESET}     Afficher la santé du système
    {CYAN}guardian{RESET}    Lancer Neo avec le Guardian (auto-restart)
    {CYAN}version{RESET}    Afficher la version

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

    elif command == "status":
        from neo_core.cli.status import run_status
        run_status()

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

    elif command == "version":
        print("Neo Core v0.7.1 — Stage 10")

    else:
        print(f"\n  Commande inconnue : '{sys.argv[1]}'")
        print_usage()
