#!/usr/bin/env python3
"""
Neo Core — Wizard d'Installation
==================================
Assistant interactif pour configurer et installer Neo Core.

Demande :
- Le nom du Core (personnalisation du système)
- Le nom de l'utilisateur
- La clé API Anthropic
- Installe toutes les dépendances automatiquement

Usage : python3 setup_wizard.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# ─── Couleurs ANSI (pas de dépendance externe) ────────────────────────

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

CONFIG_DIR = Path(__file__).parent / "data"
CONFIG_FILE = CONFIG_DIR / "neo_config.json"
ENV_FILE = Path(__file__).parent / ".env"


def clear_screen():
    os.system("clear" if os.name != "nt" else "cls")


def print_banner():
    clear_screen()
    print(f"""
{CYAN}{BOLD}
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║             ███╗   ██╗███████╗ ██████╗            ║
    ║             ████╗  ██║██╔════╝██╔═══██╗           ║
    ║             ██╔██╗ ██║█████╗  ██║   ██║           ║
    ║             ██║╚██╗██║██╔══╝  ██║   ██║           ║
    ║             ██║ ╚████║███████╗╚██████╔╝           ║
    ║             ╚═╝  ╚═══╝╚══════╝ ╚═════╝           ║
    ║                                                   ║
    ║         Écosystème IA Multi-Agents Autonome       ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
{RESET}""")


def print_step(step: int, total: int, title: str):
    print(f"\n{CYAN}{BOLD}  [{step}/{total}] {title}{RESET}")
    print(f"  {DIM}{'─' * 45}{RESET}\n")


def ask(prompt: str, default: str = "", secret: bool = False) -> str:
    """Demande une entrée utilisateur avec valeur par défaut."""
    if default:
        display = f"  {GREEN}▸{RESET} {prompt} {DIM}[{default}]{RESET}: "
    else:
        display = f"  {GREEN}▸{RESET} {prompt}: "

    if secret:
        import getpass
        value = getpass.getpass(display)
    else:
        value = input(display)

    return value.strip() or default


def ask_confirm(prompt: str, default: bool = True) -> bool:
    """Demande une confirmation oui/non."""
    suffix = "[O/n]" if default else "[o/N]"
    response = input(f"  {GREEN}▸{RESET} {prompt} {DIM}{suffix}{RESET}: ").strip().lower()
    if not response:
        return default
    return response in ("o", "oui", "y", "yes")


def run_command(cmd: str, description: str) -> bool:
    """Exécute une commande système avec affichage."""
    print(f"  {DIM}⧗ {description}...{RESET}", end="", flush=True)
    try:
        result = subprocess.run(
            cmd, shell=True,
            capture_output=True, text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print(f"\r  {GREEN}✓ {description}{RESET}          ")
            return True
        else:
            print(f"\r  {RED}✗ {description}{RESET}          ")
            if result.stderr:
                print(f"    {DIM}{result.stderr[:200]}{RESET}")
            return False
    except subprocess.TimeoutExpired:
        print(f"\r  {RED}✗ {description} (timeout){RESET}")
        return False
    except Exception as e:
        print(f"\r  {RED}✗ {description}: {e}{RESET}")
        return False


def check_python_version() -> bool:
    """Vérifie la version de Python."""
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 10:
        print(f"  {GREEN}✓{RESET} Python {major}.{minor} détecté")
        return True
    print(f"  {RED}✗{RESET} Python 3.10+ requis (version actuelle: {major}.{minor})")
    return False


def check_venv() -> bool:
    """Vérifie si on est dans un virtual environment."""
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


def setup_venv() -> str:
    """Configure le virtual environment si nécessaire."""
    if check_venv():
        print(f"  {GREEN}✓{RESET} Virtual environment actif")
        return sys.executable

    print(f"  {YELLOW}⚠{RESET} Pas de virtual environment détecté")

    venv_path = Path(__file__).parent / ".venv"

    if venv_path.exists():
        print(f"  {GREEN}✓{RESET} Virtual environment existant trouvé: {venv_path}")
    else:
        if ask_confirm("Créer un virtual environment (.venv) ?"):
            if not run_command(
                f"{sys.executable} -m venv {venv_path}",
                "Création du virtual environment"
            ):
                print(f"\n  {RED}Erreur: impossible de créer le venv.{RESET}")
                print(f"  {DIM}Essayez: sudo apt install python3-full{RESET}")
                sys.exit(1)
        else:
            print(f"\n  {YELLOW}⚠ Sans venv, l'installation pourrait échouer sur certains systèmes.{RESET}")
            return sys.executable

    # Retourne le chemin du python dans le venv
    venv_python = venv_path / "bin" / "python3"
    return str(venv_python)


def install_dependencies(python_path: str) -> bool:
    """Installe les dépendances depuis requirements.txt."""
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print(f"  {RED}✗ requirements.txt introuvable{RESET}")
        return False

    pip_cmd = f"{python_path} -m pip"

    # Upgrade pip d'abord
    run_command(f"{pip_cmd} install --upgrade pip -q", "Mise à jour de pip")

    # Install dependencies
    return run_command(
        f"{pip_cmd} install -r {req_file} -q",
        "Installation des dépendances"
    )


def save_config(core_name: str, user_name: str, api_key: str, python_path: str):
    """Sauvegarde la configuration."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "core_name": core_name,
        "user_name": user_name,
        "python_path": python_path,
        "version": "0.2",
        "stage": 2,
    }

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  {GREEN}✓{RESET} Configuration sauvegardée: {CONFIG_FILE}")

    # Fichier .env pour la clé API
    env_content = f"""# Neo Core — Configuration Environnement
# Généré par le wizard d'installation

ANTHROPIC_API_KEY={api_key}
NEO_CORE_NAME={core_name}
NEO_USER_NAME={user_name}
NEO_DEBUG=false
NEO_LOG_LEVEL=INFO
"""
    with open(ENV_FILE, "w") as f:
        f.write(env_content)

    # Sécurise le fichier .env
    os.chmod(ENV_FILE, 0o600)
    print(f"  {GREEN}✓{RESET} Variables d'environnement: {ENV_FILE}")


def print_summary(core_name: str, user_name: str, python_path: str):
    """Affiche le résumé de l'installation."""
    venv_active = ".venv" in python_path

    print(f"""
{CYAN}{BOLD}  ╔═══════════════════════════════════════════════╗
  ║          Installation terminée !               ║
  ╚═══════════════════════════════════════════════╝{RESET}

  {BOLD}Configuration :{RESET}
    Nom du Core  : {GREEN}{core_name}{RESET}
    Utilisateur  : {GREEN}{user_name}{RESET}
    Python       : {DIM}{python_path}{RESET}
    Venv         : {GREEN if venv_active else YELLOW}{'oui' if venv_active else 'non'}{RESET}

  {BOLD}Pour lancer {core_name} :{RESET}
""")

    if venv_active and not check_venv():
        print(f"    {CYAN}source .venv/bin/activate{RESET}")

    print(f"    {CYAN}python3 -m neo_core.main{RESET}")
    print(f"""
  {BOLD}Pour lancer les tests :{RESET}
    {CYAN}python3 -m pytest tests/ -v{RESET}

  {DIM}─────────────────────────────────────────────────{RESET}
  {DIM}Fichiers de configuration :{RESET}
  {DIM}  .env              → Clés API et variables{RESET}
  {DIM}  data/neo_config.json → Paramètres du Core{RESET}
  {DIM}─────────────────────────────────────────────────{RESET}
""")


def main():
    print_banner()

    print(f"  {BOLD}Bienvenue dans le wizard d'installation de Neo Core.{RESET}")
    print(f"  {DIM}Ce wizard va configurer votre système en quelques étapes.{RESET}\n")

    # ─── Étape 1 : Vérifications système ─────────────────────────

    print_step(1, 4, "Vérifications système")

    if not check_python_version():
        sys.exit(1)

    python_path = setup_venv()

    # ─── Étape 2 : Identité du Core ─────────────────────────────

    print_step(2, 4, "Identité du Core")

    print(f"  {DIM}Donnez un nom à votre système IA.{RESET}")
    print(f"  {DIM}Ce nom sera utilisé par les agents pour se référencer.{RESET}\n")

    core_name = ask("Nom du Core", default="Neo")
    user_name = ask("Votre nom / pseudonyme")

    if not user_name:
        print(f"  {RED}Le nom d'utilisateur est requis.{RESET}")
        sys.exit(1)

    print(f"\n  {GREEN}✓{RESET} Core: {BOLD}{core_name}{RESET} — Utilisateur: {BOLD}{user_name}{RESET}")

    # ─── Étape 3 : Clé API ──────────────────────────────────────

    print_step(3, 4, "Connexion Anthropic")

    print(f"  {DIM}La clé API Anthropic permet à Brain de fonctionner avec Claude.{RESET}")
    print(f"  {DIM}Sans clé, le système tourne en mode mock (réponses simulées).{RESET}")
    print(f"  {DIM}Obtenez votre clé sur : https://console.anthropic.com/keys{RESET}\n")

    api_key = ask("Clé API Anthropic (laisser vide pour mode mock)", secret=True)

    if api_key:
        # Vérification basique du format
        if api_key.startswith("sk-ant-") or api_key.startswith("sk-"):
            print(f"  {GREEN}✓{RESET} Clé API enregistrée")
        else:
            print(f"  {YELLOW}⚠{RESET} Le format de la clé semble inhabituel, mais on continue.")
    else:
        print(f"  {YELLOW}⚠{RESET} Mode mock activé — Brain simulera les réponses")

    # ─── Étape 4 : Installation des dépendances ─────────────────

    print_step(4, 4, "Installation des dépendances")

    if not install_dependencies(python_path):
        print(f"\n  {RED}⚠ L'installation a rencontré des erreurs.{RESET}")
        print(f"  {DIM}Vous pouvez réessayer manuellement :{RESET}")
        print(f"  {CYAN}{python_path} -m pip install -r requirements.txt{RESET}")
        if not ask_confirm("Continuer malgré les erreurs ?", default=False):
            sys.exit(1)

    # ─── Sauvegarde ──────────────────────────────────────────────

    save_config(core_name, user_name, api_key, python_path)

    # ─── Résumé ──────────────────────────────────────────────────

    print_summary(core_name, user_name, python_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n  {DIM}Installation annulée.{RESET}\n")
        sys.exit(0)
