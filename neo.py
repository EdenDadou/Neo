#!/usr/bin/env python3
"""
Neo Core — Bootstrap CLI
=========================
Premier point d'entrée après un git clone.
S'auto-installe puis délègue à la commande `neo`.

Usage :
    git clone https://github.com/EdenDadou/Neo.git && cd Neo
    python3 neo.py setup     ← première fois uniquement
    neo chat                 ← ensuite, toujours `neo`
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

PROJECT_ROOT = Path(__file__).resolve().parent


def neo_is_installed() -> bool:
    """Vérifie si la commande `neo` est déjà disponible."""
    return shutil.which("neo") is not None


def get_pip_cmd() -> str:
    """Retourne la commande pip à utiliser."""
    return f"{sys.executable} -m pip"


def bootstrap_install():
    """Installe Neo Core en mode éditable pour rendre `neo` disponible."""
    print(f"\n{CYAN}{BOLD}  Neo Core — Bootstrap{RESET}\n")
    print(f"  {DIM}Installation de la commande `neo`...{RESET}\n")

    pip_cmd = get_pip_cmd()

    # Upgrade pip
    subprocess.run(
        f"{pip_cmd} install --upgrade pip -q",
        shell=True, capture_output=True,
    )

    # Install en mode éditable
    result = subprocess.run(
        f"{pip_cmd} install -e '{PROJECT_ROOT}[dev]' -q",
        shell=True, capture_output=True, text=True,
        timeout=300,
    )

    if result.returncode != 0:
        print(f"  {RED}✗ Erreur d'installation :{RESET}")
        if result.stderr:
            print(f"  {DIM}{result.stderr[:300]}{RESET}")
        print(f"\n  {DIM}Essayez manuellement : {pip_cmd} install -e .[dev]{RESET}\n")
        sys.exit(1)

    # Vérifier que neo est maintenant dispo
    # Rafraîchir le PATH pour trouver le nouveau binaire
    bin_dir = Path(sys.executable).parent
    neo_path = bin_dir / "neo"

    if neo_path.exists():
        print(f"  {GREEN}✓{RESET} Commande {CYAN}neo{RESET} installée !")
        print(f"  {DIM}  → {neo_path}{RESET}\n")
        return str(neo_path)
    else:
        # Chercher dans les paths courants
        for p in [
            Path.home() / ".local" / "bin" / "neo",
            bin_dir / "neo",
        ]:
            if p.exists():
                print(f"  {GREEN}✓{RESET} Commande {CYAN}neo{RESET} installée !")
                print(f"  {DIM}  → {p}{RESET}\n")
                return str(p)

        # Fallback : pas trouvé mais installé quand même
        print(f"  {GREEN}✓{RESET} Neo Core installé")
        print(f"  {YELLOW}⚠{RESET} Commande `neo` non trouvée dans le PATH")
        print(f"  {DIM}  Ajoutez au PATH : export PATH=\"{bin_dir}:$PATH\"{RESET}\n")
        return None


def main():
    # Si `neo` est déjà installé, déléguer directement
    if neo_is_installed():
        from neo_core.cli import main as cli_main
        cli_main()
        return

    # Sinon, bootstrap : installer puis relancer avec `neo`
    if len(sys.argv) < 2:
        print(f"\n{CYAN}{BOLD}  Neo Core{RESET}\n")
        print(f"  Première utilisation détectée.")
        print(f"  Lancez : {CYAN}python3 neo.py setup{RESET}\n")
        return

    neo_path = bootstrap_install()

    # Relancer la commande via `neo` maintenant qu'il est installé
    if neo_path:
        print(f"  {DIM}Lancement de : neo {' '.join(sys.argv[1:])}{RESET}\n")
        os.execv(neo_path, ["neo"] + sys.argv[1:])
    else:
        # Fallback : import direct
        from neo_core.cli import main as cli_main
        cli_main()


if __name__ == "__main__":
    main()
