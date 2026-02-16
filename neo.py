#!/usr/bin/env python3
"""
Neo Core — Point d'entrée CLI unifié
======================================
Usage :
    python3 neo.py setup     Onboarding complet (install + config + chat)
    python3 neo.py chat      Lancer le chat directement
    python3 neo.py status    Afficher la santé du système
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
    python3 neo.py {CYAN}<commande>{RESET}

  {BOLD}Commandes :{RESET}
    {CYAN}setup{RESET}    Onboarding complet (install deps + config + lance le chat)
    {CYAN}chat{RESET}     Lancer le chat directement
    {CYAN}status{RESET}   Afficher la santé du système

  {BOLD}Première utilisation :{RESET}
    {DIM}git clone https://github.com/EdenDadou/Neo.git && cd Neo{RESET}
    {DIM}python3 neo.py setup{RESET}
""")


def main():
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

    elif command == "version":
        print("Neo Core v0.5 — Stage 5")

    else:
        print(f"\n  Commande inconnue : '{sys.argv[1]}'")
        print_usage()


if __name__ == "__main__":
    main()
