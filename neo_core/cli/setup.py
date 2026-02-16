"""
Neo Core — Setup : Onboarding complet
=======================================
Installe les dépendances, configure le système, et lance le chat.

Usage : python3 neo.py setup
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

# Racine du projet (parent de neo_core/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "data"
CONFIG_FILE = CONFIG_DIR / "neo_config.json"
ENV_FILE = PROJECT_ROOT / ".env"


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
        try:
            import getpass
            value = getpass.getpass(display)
            if not value.strip():
                print(f"  {DIM}(saisie non détectée, réessai en mode visible){RESET}")
                value = input(display)
        except (EOFError, OSError):
            value = input(display)
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

    venv_path = PROJECT_ROOT / ".venv"

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

    venv_python = venv_path / "bin" / "python3"
    return str(venv_python)


def install_dependencies(python_path: str) -> bool:
    """Installe les dépendances depuis requirements.txt."""
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        print(f"  {RED}✗ requirements.txt introuvable{RESET}")
        return False

    pip_cmd = f"{python_path} -m pip"

    run_command(f"{pip_cmd} install --upgrade pip -q", "Mise à jour de pip")

    return run_command(
        f"{pip_cmd} install -r {req_file} -q",
        "Installation des dépendances"
    )


def configure_auth() -> str:
    """Configure l'authentification Anthropic. Retourne la clé/token."""
    print(f"  {DIM}La clé Anthropic permet à Brain de fonctionner avec Claude.{RESET}")
    print(f"  {DIM}Sans clé, le système tourne en mode mock (réponses simulées).{RESET}")
    print()
    print(f"  {DIM}Trois méthodes supportées :{RESET}")
    print(f"  {DIM}  1. Claude Code : si vous avez 'claude' installé (auto-import){RESET}")
    print(f"  {DIM}  2. API Key     : sk-ant-api... (console.anthropic.com/keys){RESET}")
    print(f"  {DIM}  3. Token OAuth : sk-ant-oat... (claude setup-token){RESET}")
    print()

    api_key = None

    # Tenter l'import automatique depuis Claude Code
    try:
        from neo_core.oauth import import_claude_code_credentials, setup_oauth_from_token

        claude_creds = import_claude_code_credentials()
        if claude_creds:
            print(f"  {GREEN}✓{RESET} Credentials Claude Code détectées automatiquement !")
            token = claude_creds["access_token"]
            print(f"  {DIM}  Token: {token[:12]}...{token[-4:]}{RESET}")
            if ask_confirm("Utiliser ces credentials ?"):
                api_key = token
                print(f"  {GREEN}✓{RESET} Credentials Claude Code importées (avec refresh automatique)")
    except Exception:
        pass

    if not api_key:
        api_key = ask("Clé Anthropic (laisser vide pour mode mock)", secret=True)

    if api_key:
        masked = api_key[:12] + "..." + api_key[-4:]
        if api_key.startswith("sk-ant-oat"):
            try:
                from neo_core.oauth import setup_oauth_from_token
                result = setup_oauth_from_token(api_key)
                if result["success"]:
                    print(f"  {GREEN}✓{RESET} {result['message']} : {DIM}{masked}{RESET}")
                else:
                    print(f"  {YELLOW}⚠{RESET} {result['message']}")
            except Exception:
                print(f"  {GREEN}✓{RESET} Token OAuth stocké : {DIM}{masked}{RESET}")
        elif api_key.startswith("sk-ant-api") or api_key.startswith("sk-"):
            print(f"  {GREEN}✓{RESET} Clé API classique : {DIM}{masked}{RESET}")
        else:
            print(f"  {YELLOW}⚠{RESET} Format non reconnu : {DIM}{masked}{RESET}")
    else:
        print(f"  {YELLOW}⚠{RESET} Mode mock activé — Brain simulera les réponses")

    return api_key or ""


def test_connection(api_key: str) -> bool:
    """Teste la connexion à l'API Anthropic."""
    if not api_key:
        print(f"  {DIM}⊘ Pas de clé API — test ignoré (mode mock){RESET}")
        return True

    print(f"  {DIM}⧗ Test de connexion à Anthropic...{RESET}", end="", flush=True)
    try:
        import httpx
        from neo_core.oauth import is_oauth_token, get_valid_access_token, OAUTH_BETA_HEADER

        headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        if is_oauth_token(api_key):
            valid_token = get_valid_access_token() or api_key
            headers["Authorization"] = f"Bearer {valid_token}"
            headers["anthropic-beta"] = OAUTH_BETA_HEADER
        else:
            headers["x-api-key"] = api_key

        payload = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Dis 'ok'"}],
        }

        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=15,
        )

        if response.status_code == 200:
            print(f"\r  {GREEN}✓ Connexion à Anthropic réussie !{RESET}          ")
            return True
        else:
            error = response.json().get("error", {}).get("message", response.text[:100])
            print(f"\r  {YELLOW}⚠ Connexion échouée (HTTP {response.status_code}): {error}{RESET}")
            return False
    except ImportError:
        print(f"\r  {DIM}⊘ httpx non installé — test ignoré{RESET}          ")
        return True
    except Exception as e:
        print(f"\r  {YELLOW}⚠ Test échoué: {e}{RESET}          ")
        return False


def save_config(core_name: str, user_name: str, api_key: str, python_path: str):
    """Sauvegarde la configuration."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "core_name": core_name,
        "user_name": user_name,
        "python_path": python_path,
        "version": "0.5",
        "stage": 5,
    }

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  {GREEN}✓{RESET} Configuration sauvegardée: {CONFIG_FILE}")

    env_content = f"""# Neo Core — Configuration Environnement
# Généré par neo.py setup

ANTHROPIC_API_KEY={api_key}
NEO_CORE_NAME={core_name}
NEO_USER_NAME={user_name}
NEO_DEBUG=false
NEO_LOG_LEVEL=INFO
"""
    with open(ENV_FILE, "w") as f:
        f.write(env_content)

    os.chmod(ENV_FILE, 0o600)
    print(f"  {GREEN}✓{RESET} Variables d'environnement: {ENV_FILE}")


def run_setup():
    """Point d'entrée du setup complet."""
    print_banner()

    print(f"  {BOLD}Bienvenue dans le setup de Neo Core.{RESET}")
    print(f"  {DIM}Ce wizard va tout configurer en quelques étapes.{RESET}\n")

    # ─── Étape 1 : Vérifications système ─────────────────────────
    print_step(1, 5, "Vérifications système")

    if not check_python_version():
        sys.exit(1)

    python_path = setup_venv()

    # ─── Étape 2 : Installation des dépendances ─────────────────
    print_step(2, 5, "Installation des dépendances")

    if not install_dependencies(python_path):
        print(f"\n  {RED}⚠ L'installation a rencontré des erreurs.{RESET}")
        print(f"  {DIM}Vous pouvez réessayer manuellement :{RESET}")
        print(f"  {CYAN}{python_path} -m pip install -r requirements.txt{RESET}")
        if not ask_confirm("Continuer malgré les erreurs ?", default=False):
            sys.exit(1)

    # ─── Étape 3 : Identité du Core ─────────────────────────────
    print_step(3, 5, "Identité du Core")

    print(f"  {DIM}Donnez un nom à votre système IA.{RESET}")
    print(f"  {DIM}Ce nom sera utilisé par les agents pour se référencer.{RESET}\n")

    core_name = ask("Nom du Core", default="Neo")
    user_name = ask("Votre nom / pseudonyme")

    if not user_name:
        print(f"  {RED}Le nom d'utilisateur est requis.{RESET}")
        sys.exit(1)

    print(f"\n  {GREEN}✓{RESET} Core: {BOLD}{core_name}{RESET} — Utilisateur: {BOLD}{user_name}{RESET}")

    # ─── Étape 4 : Connexion Anthropic ───────────────────────────
    print_step(4, 5, "Connexion Anthropic")

    api_key = configure_auth()

    # ─── Étape 5 : Sauvegarde et vérification ────────────────────
    print_step(5, 5, "Finalisation")

    save_config(core_name, user_name, api_key, python_path)

    # Test de connexion
    test_connection(api_key)

    # ─── Résumé + lancement du chat ──────────────────────────────
    print(f"""
{CYAN}{BOLD}  ╔═══════════════════════════════════════════════╗
  ║          Installation terminée !               ║
  ╚═══════════════════════════════════════════════╝{RESET}

  {BOLD}Configuration :{RESET}
    Nom du Core  : {GREEN}{core_name}{RESET}
    Utilisateur  : {GREEN}{user_name}{RESET}
    Auth         : {GREEN}{'API Key / OAuth' if api_key else 'Mode mock'}{RESET}
""")

    if ask_confirm(f"Lancer {core_name} maintenant ?"):
        print(f"\n  {CYAN}Démarrage de {core_name}...{RESET}\n")
        # Reload la config depuis le .env qu'on vient de créer
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=True)

        from neo_core.cli.chat import run_chat
        run_chat()
    else:
        print(f"""
  {BOLD}Pour lancer {core_name} :{RESET}
    {CYAN}python3 neo.py chat{RESET}

  {BOLD}Pour lancer les tests :{RESET}
    {CYAN}python3 -m pytest tests/ -v{RESET}
""")
