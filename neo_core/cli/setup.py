"""
Neo Core — Setup : Onboarding complet
=======================================
Installe les dépendances, configure le système, et lance le chat.

Usage : neo setup
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
    """Installe Neo Core en mode éditable (pip install -e .)."""
    pip_cmd = f"{python_path} -m pip"

    run_command(f"{pip_cmd} install --upgrade pip -q", "Mise à jour de pip")

    # Install en mode éditable — installe le package + toutes les deps
    success = run_command(
        f"{pip_cmd} install -e '{PROJECT_ROOT}[dev]' -q",
        "Installation de Neo Core + dépendances"
    )

    if success:
        # Vérifier que la commande neo est dispo
        neo_path = Path(python_path).parent / "neo"
        if neo_path.exists():
            print(f"  {GREEN}✓{RESET} Commande {CYAN}neo{RESET} installée !")
        else:
            print(f"  {DIM}  (utilisez {CYAN}{python_path} -m neo_core.cli{RESET} comme fallback){RESET}")

    return success


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


def save_config(core_name: str, user_name: str, api_key: str,
                python_path: str, provider_keys: dict | None = None):
    """Sauvegarde la configuration."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "core_name": core_name,
        "user_name": user_name,
        "python_path": python_path,
        "version": "0.6",
        "stage": 6,
    }

    # Sauvegarder les providers configurés (sans les clés)
    if provider_keys:
        config["providers"] = {
            k: {"enabled": bool(v)}
            for k, v in provider_keys.items()
        }

    # Fusionner avec la config existante (tests, routing)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                existing = json.load(f)
            # Garder les résultats de tests et le routing existants
            for key in ("tested_models", "model_routing"):
                if key in existing and key not in config:
                    config[key] = existing[key]
        except (json.JSONDecodeError, IOError):
            pass

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  {GREEN}✓{RESET} Configuration sauvegardée: {CONFIG_FILE}")

    # .env avec toutes les clés
    env_lines = [
        "# Neo Core — Configuration Environnement",
        "# Généré par neo.py setup",
        "",
        f"ANTHROPIC_API_KEY={api_key}",
    ]

    if provider_keys:
        if provider_keys.get("groq") and provider_keys["groq"] != "local":
            env_lines.append(f"GROQ_API_KEY={provider_keys['groq']}")
        if provider_keys.get("gemini") and provider_keys["gemini"] != "local":
            env_lines.append(f"GEMINI_API_KEY={provider_keys['gemini']}")

    env_lines.extend([
        "",
        f"NEO_CORE_NAME={core_name}",
        f"NEO_USER_NAME={user_name}",
        "NEO_DEBUG=false",
        "NEO_LOG_LEVEL=INFO",
        "",
    ])

    with open(ENV_FILE, "w") as f:
        f.write("\n".join(env_lines))

    os.chmod(ENV_FILE, 0o600)
    print(f"  {GREEN}✓{RESET} Variables d'environnement: {ENV_FILE}")


def configure_hardware_and_providers(api_key: str) -> dict:
    """
    Étape 5 du wizard : détection hardware + configuration des providers.

    Retourne un dict avec les clés API et modèles configurés.
    """
    provider_keys = {"anthropic": api_key}

    # ── 5a. Détection hardware ──
    print(f"  {DIM}⧗ Détection du hardware...{RESET}", end="", flush=True)
    try:
        from neo_core.providers.hardware import HardwareDetector
        profile = HardwareDetector.detect()
        print(f"\r  {GREEN}✓{RESET} {profile.summary()}")
    except Exception as e:
        print(f"\r  {YELLOW}⚠{RESET} Impossible de détecter le hardware: {e}")
        profile = None

    # ── 5b. Modèles locaux (Ollama) ──
    print()
    if profile and profile.max_model_size() != "none":
        models = profile.recommend_ollama_models()
        model_names = ", ".join(m["model"] for m in models[:2])
        print(f"  {BOLD}Modèles locaux (Ollama) :{RESET}")
        print(f"  {DIM}Votre machine peut faire tourner des modèles jusqu'à {profile.max_model_size()}{RESET}")
        print(f"  {DIM}Recommandé : {model_names}{RESET}")
        print()

        if ask_confirm("Installer Ollama et les modèles recommandés ?"):
            # Vérifier si Ollama est déjà installé
            ollama_installed = HardwareDetector.is_ollama_installed()
            if not ollama_installed:
                print(f"  {DIM}⧗ Installation d'Ollama...{RESET}", end="", flush=True)
                success = run_command(
                    "curl -fsSL https://ollama.com/install.sh | sh",
                    "Installation d'Ollama"
                )
                if not success:
                    print(f"  {YELLOW}⚠{RESET} Installation Ollama échouée. Vous pouvez l'installer manuellement :")
                    print(f"  {CYAN}curl -fsSL https://ollama.com/install.sh | sh{RESET}")
                    ollama_installed = False
                else:
                    ollama_installed = True

            if ollama_installed:
                # Démarrer Ollama en arrière-plan si pas encore lancé
                if not HardwareDetector.is_ollama_running():
                    run_command("ollama serve &", "Démarrage du serveur Ollama")
                    import time
                    time.sleep(2)  # Attendre le démarrage

                # Télécharger les modèles recommandés
                for m in models[:2]:
                    print(f"  {DIM}⧗ Téléchargement de {m['model']}...{RESET}", end="", flush=True)
                    success = run_command(
                        f"ollama pull {m['model']}",
                        f"Téléchargement de {m['model']}"
                    )
                    if success:
                        print(f"\r  {GREEN}✓{RESET} {m['model']} prêt ({m['size']}, {m['role']})")
                    else:
                        print(f"\r  {YELLOW}⚠{RESET} Échec du téléchargement de {m['model']}")

                provider_keys["ollama"] = "local"  # Marqueur pour Ollama
    else:
        print(f"  {DIM}Modèles locaux : hardware insuffisant, utilisation cloud uniquement.{RESET}")

    # ── 5c. Groq (cloud gratuit) ──
    print()
    print(f"  {BOLD}Modèles cloud gratuits :{RESET}")
    print()
    print(f"  {DIM}Groq — Llama 3.3 70B, ultra-rapide, 14 400 req/jour{RESET}")
    print(f"  {DIM}Clé gratuite : https://console.groq.com/keys{RESET}")

    if ask_confirm("Configurer Groq (gratuit) ?"):
        groq_key = ask("Clé Groq (gsk_...)", secret=True)
        if groq_key and groq_key.startswith("gsk_") and len(groq_key) > 10:
            provider_keys["groq"] = groq_key
            print(f"  {GREEN}✓{RESET} Groq configuré")
        elif groq_key:
            print(f"  {YELLOW}⚠{RESET} Clé Groq invalide (doit commencer par gsk_) — ignoré")
        else:
            print(f"  {DIM}  (ignoré){RESET}")

    # ── 5d. Gemini (cloud gratuit) ──
    print()
    print(f"  {DIM}Gemini — 1M tokens contexte, 250 req/jour{RESET}")
    print(f"  {DIM}Clé gratuite : https://aistudio.google.com/apikey{RESET}")

    if ask_confirm("Configurer Gemini (gratuit) ?"):
        gemini_key = ask("Clé Gemini (AIza...)", secret=True)
        if gemini_key and gemini_key.startswith("AIza") and len(gemini_key) > 10:
            provider_keys["gemini"] = gemini_key
            print(f"  {GREEN}✓{RESET} Gemini configuré")
        elif gemini_key:
            print(f"  {YELLOW}⚠{RESET} Clé Gemini invalide (doit commencer par AIza) — ignoré")
        else:
            print(f"  {DIM}  (ignoré){RESET}")

    # ── 5e. Test des modèles configurés ──
    print()
    print(f"  {DIM}⧗ Test des modèles configurés...{RESET}")

    test_results = _test_providers(provider_keys)

    for model_id, result in test_results.items():
        if result["success"]:
            print(f"  {GREEN}✓{RESET} {model_id} → {result['latency_ms']:.0f}ms")
        else:
            print(f"  {RED}✗{RESET} {model_id} → {result['error'][:60]}")

    return provider_keys


def _test_providers(provider_keys: dict) -> dict:
    """Teste les providers configurés (sync wrapper)."""
    import asyncio

    async def _run_tests():
        results = {}

        try:
            from neo_core.providers.registry import ModelRegistry
            from neo_core.providers.anthropic_provider import AnthropicProvider
            from neo_core.providers.groq_provider import GroqProvider
            from neo_core.providers.gemini_provider import GeminiProvider
            from neo_core.providers.ollama_provider import OllamaProvider

            # IMPORTANT : passer config_path pour que les résultats de tests
            # soient persistés dans neo_config.json et rechargés au runtime
            registry = ModelRegistry(config_path=CONFIG_FILE)

            if provider_keys.get("anthropic"):
                registry.register_provider(AnthropicProvider(api_key=provider_keys["anthropic"]))
            if provider_keys.get("groq"):
                registry.register_provider(GroqProvider(api_key=provider_keys["groq"]))
            if provider_keys.get("gemini"):
                registry.register_provider(GeminiProvider(api_key=provider_keys["gemini"]))
            if provider_keys.get("ollama"):
                ollama = OllamaProvider()
                if ollama.is_configured():
                    registry.register_provider(ollama)

            registry.discover_models()

            for model_id, model in registry._models.items():
                try:
                    result = await registry.test_model(model_id)
                    results[model_id] = result.to_dict()
                except Exception as e:
                    results[model_id] = {
                        "success": False,
                        "error": str(e),
                        "latency_ms": 0,
                    }

        except Exception as e:
            results["__error__"] = {
                "success": False,
                "error": f"Erreur d'initialisation: {e}",
                "latency_ms": 0,
            }

        return results

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return {}
        return loop.run_until_complete(_run_tests())
    except RuntimeError:
        return asyncio.run(_run_tests())


def run_setup():
    """Point d'entrée du setup complet."""
    print_banner()

    print(f"  {BOLD}Bienvenue dans le setup de Neo Core.{RESET}")
    print(f"  {DIM}Ce wizard va tout configurer en quelques étapes.{RESET}\n")

    # ─── Étape 1 : Vérifications système ─────────────────────────
    print_step(1, 8, "Vérifications système")

    if not check_python_version():
        sys.exit(1)

    python_path = setup_venv()

    # ─── Étape 2 : Installation des dépendances ─────────────────
    print_step(2, 8, "Installation des dépendances")

    if not install_dependencies(python_path):
        print(f"\n  {RED}⚠ L'installation a rencontré des erreurs.{RESET}")
        print(f"  {DIM}Vous pouvez réessayer manuellement :{RESET}")
        print(f"  {CYAN}{python_path} -m pip install -r requirements.txt{RESET}")
        if not ask_confirm("Continuer malgré les erreurs ?", default=False):
            sys.exit(1)

    # ─── Étape 3 : Identité du Core ─────────────────────────────
    print_step(3, 8, "Identité du Core")

    print(f"  {DIM}Donnez un nom à votre système IA.{RESET}")
    print(f"  {DIM}Ce nom sera utilisé par les agents pour se référencer.{RESET}\n")

    core_name = ask("Nom du Core", default="Neo")
    user_name = ask("Votre nom / pseudonyme")

    if not user_name:
        print(f"  {RED}Le nom d'utilisateur est requis.{RESET}")
        sys.exit(1)

    print(f"\n  {GREEN}✓{RESET} Core: {BOLD}{core_name}{RESET} — Utilisateur: {BOLD}{user_name}{RESET}")

    # ─── Étape 4 : Connexion Anthropic (optionnel) ───────────────
    print_step(4, 8, "Connexion Anthropic (payant, optionnel)")

    api_key = configure_auth()

    # ─── Étape 5 : Modèles LLM (hardware + providers gratuits) ───
    print_step(5, 8, "Configuration des modèles LLM")

    provider_keys = configure_hardware_and_providers(api_key)

    # ─── Étape 6 : Sauvegarde ────────────────────────────────────
    print_step(6, 8, "Sauvegarde")

    save_config(core_name, user_name, api_key, python_path, provider_keys)

    # ─── Étape 7 : Test final ────────────────────────────────────
    print_step(7, 8, "Vérification finale")

    if api_key:
        test_connection(api_key)

    # Compter les providers configurés
    active_providers = [k for k, v in provider_keys.items() if v]
    provider_list = ", ".join(active_providers) if active_providers else "Mode mock"

    # ─── Étape 8 : Démarrage du daemon ───────────────────────────
    print_step(8, 8, "Démarrage du daemon Neo")

    # Reload la config depuis le .env qu'on vient de créer
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE, override=True)

    # Installer les dépendances finales si nécessaire
    print(f"  {DIM}⧗ Vérification des dépendances du daemon...{RESET}", end="", flush=True)
    try:
        import psutil
        import cryptography
        print(f"\r  {GREEN}✓{RESET} Dépendances daemon OK (psutil, cryptography)")
    except ImportError as e:
        print(f"\r  {YELLOW}⚠{RESET} Dépendance manquante: {e}")
        run_command(
            f"{python_path} -m pip install psutil cryptography -q",
            "Installation des dépendances daemon"
        )

    # Démarrer le daemon
    from neo_core.core.daemon import start, is_running, get_status

    if is_running():
        print(f"  {GREEN}✓{RESET} Neo daemon déjà en cours d'exécution")
        status = get_status()
        if status.get("pid"):
            print(f"  {DIM}  PID {status['pid']} — {status.get('memory_mb', '?')} MB RAM{RESET}")
    else:
        print(f"  {DIM}⧗ Démarrage du daemon Neo (heartbeat + API)...{RESET}")
        result = start(foreground=False)
        if result["success"]:
            print(f"  {GREEN}✓{RESET} {result['message']}")
            print(f"  {DIM}  API disponible sur http://0.0.0.0:8000{RESET}")
            print(f"  {DIM}  Logs : neo logs{RESET}")
        else:
            print(f"  {YELLOW}⚠{RESET} {result['message']}")
            print(f"  {DIM}  Vous pouvez démarrer manuellement : neo start{RESET}")

    # Proposer l'installation du service systemd
    import platform
    if platform.system() == "Linux":
        print()
        if ask_confirm("Installer le service systemd (démarrage automatique au boot) ?", default=False):
            from neo_core.core.daemon import install_service
            svc_result = install_service()
            if svc_result["success"]:
                print(f"  {GREEN}✓{RESET} {svc_result['message']}")
            else:
                print(f"  {YELLOW}⚠{RESET} {svc_result['message']}")
                for cmd in svc_result.get("commands", []):
                    print(f"    {DIM}{cmd}{RESET}")

    # ─── Résumé final ─────────────────────────────────────────────
    print(f"""
{CYAN}{BOLD}  ╔═══════════════════════════════════════════════╗
  ║          Installation terminée !               ║
  ╚═══════════════════════════════════════════════╝{RESET}

  {BOLD}Configuration :{RESET}
    Nom du Core   : {GREEN}{core_name}{RESET}
    Utilisateur   : {GREEN}{user_name}{RESET}
    Providers LLM : {GREEN}{provider_list}{RESET}

  {BOLD}Commandes utiles :{RESET}
    {CYAN}neo chat{RESET}        Discuter avec {core_name}
    {CYAN}neo status{RESET}      État du système + daemon
    {CYAN}neo logs{RESET}        Voir les logs du daemon
    {CYAN}neo stop{RESET}        Arrêter le daemon
    {CYAN}neo restart{RESET}     Redémarrer le daemon
""")

    # Lancer le chat
    print(f"  {CYAN}Démarrage du chat avec {core_name}...{RESET}\n")
    from neo_core.cli.chat import run_chat
    run_chat()
