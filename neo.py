#!/usr/bin/env python3
"""
Neo Core — Fallback CLI
========================
Si `neo` n'est pas installé (pip install -e .), ce script sert de fallback.
Préférer : pip install -e . && neo setup

Usage :
    python3 neo.py setup
    python3 neo.py chat
    python3 neo.py status
"""

from neo_core.cli import main

if __name__ == "__main__":
    main()
