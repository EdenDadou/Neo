"""
Neo Core — Setup : Configuration uniquement
=============================================
Configure l'identité, les clés API, les providers et Telegram.
L'installation système est gérée par install.sh.

Usage : neo setup
"""

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Couleurs ANSI (pas de dépendance externe) ────────────────────────

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Racine du projet : neo_core/vox/cli/setup.py → 4 niveaux pour atteindre /opt/neo-core/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "data"
CONFIG_FILE = CONFIG_DIR / "neo_config.json"
ENV_FILE = PROJECT_ROOT / ".env"


def _load_existing_env() -> dict:
    """Charge les clés existantes depuis le vault + .env (si présent)."""
    existing = {}
    # 1. Lire le .env (variables non-sensibles + fallback legacy)
    if ENV_FILE.exists():
        try:
            for line in ENV_FILE.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if value:
                        existing[key] = value
        except OSError:
            pass
    # 2. Lire le vault (clés chiffrées — prioritaire sur .env)
    try:
        from neo_core.infra.security.vault import KeyVault
        vault_dir = CONFIG_DIR
        if (vault_dir / ".vault.db").exists():
            vault = KeyVault(data_dir=vault_dir)
            vault.initialize()
            _vault_map = {
                "anthropic_api_key": "ANTHROPIC_API_KEY",
                "groq_api_key": "GROQ_API_KEY",
                "gemini_api_key": "GEMINI_API_KEY",
                "hf_token": "HF_TOKEN",
            }
            for vault_name, env_name in _vault_map.items():
                val = vault.retrieve(vault_name)
                if val:
                    existing[env_name] = val
            vault.close()
    except Exception:
        pass
    return existing


def _mask_key(key: str) -> str:
    """Masque une clé en gardant début et fin visibles."""
    if len(key) <= 12:
        return key[:4] + "..." + key[-2:]
    return key[:12] + "..." + key[-4:]


def clear_screen():
    import subprocess
    subprocess.run(["clear" if os.name != "nt" else "cls"], check=False)


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


def _download_embedding_model(provider_keys: dict | None = None) -> bool:
    """
    Télécharge et vérifie le modèle d'embedding all-MiniLM-L6-v2 via FastEmbed.

    FastEmbed utilise ONNX Runtime (pas PyTorch) et gère son propre cache.
    Pas de dépendance à l'API HuggingFace Hub → pas de rate-limit 429.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print(f"\n  {DIM}⧗ Vérification du modèle d'embedding (all-MiniLM-L6-v2)...{RESET}")

    try:
        from fastembed import TextEmbedding
    except ImportError:
        print(f"  {YELLOW}⚠{RESET} fastembed non installé — mémoire en mode dégradé")
        print(f"  {DIM}  Fix: pip install fastembed{RESET}")
        return False

    try:
        # FastEmbed télécharge automatiquement si pas en cache (~80 Mo ONNX)
        model = TextEmbedding(model_name=model_name)
        result = list(model.embed(["test de vérification"]))
        if result and len(result) > 0:
            dim = len(result[0])
            print(f"  {GREEN}✓{RESET} Modèle all-MiniLM-L6-v2 prêt (dim={dim}, ONNX)")
            del model
            return True
    except Exception as e:
        print(f"  {RED}✗{RESET} Erreur FastEmbed: {str(e)[:200]}")

    print(f"  {YELLOW}⚠{RESET} Modèle d'embedding non disponible — mémoire en mode dégradé (bag-of-words)")
    print(f"  {DIM}  La recherche mémoire fonctionnera par mots-clés au lieu de sémantique.{RESET}")
    return False


def configure_auth(existing_env: dict | None = None) -> str:
    """Configure l'authentification Anthropic. Retourne la clé/token."""
    existing_env = existing_env or {}

    # Si une clé existe déjà, proposer de la garder
    existing_key = existing_env.get("ANTHROPIC_API_KEY", "")
    if existing_key:
        print(f"  {GREEN}✓{RESET} Clé Anthropic déjà configurée : {DIM}{_mask_key(existing_key)}{RESET}")
        if not ask_confirm("Changer cette clé ?", default=False):
            return existing_key
        print()

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
    except Exception as e:
        logger.debug("Claude Code credentials import failed: %s", e)

    if not api_key:
        api_key = ask("Clé Anthropic (laisser vide pour mode mock)", secret=True)

    if api_key:
        masked = _mask_key(api_key)
        if api_key.startswith("sk-ant-oat"):
            try:
                from neo_core.oauth import setup_oauth_from_token
                result = setup_oauth_from_token(api_key)
                if result["success"]:
                    print(f"  {GREEN}✓{RESET} {result['message']} : {DIM}{masked}{RESET}")
                else:
                    print(f"  {YELLOW}⚠{RESET} {result['message']}")
            except Exception as e:
                logger.debug("OAuth setup failed, storing token raw: %s", e)
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


def _write_env_fallback(api_key: str, provider_keys: dict | None = None):
    """Fallback : écrit les clés en .env si le vault est indisponible."""
    lines = []
    if api_key:
        lines.append(f"ANTHROPIC_API_KEY={api_key}")
    if provider_keys:
        if provider_keys.get("huggingface"):
            lines.append(f"HF_TOKEN={provider_keys['huggingface']}")
        if provider_keys.get("groq") and provider_keys["groq"] != "local":
            lines.append(f"GROQ_API_KEY={provider_keys['groq']}")
        if provider_keys.get("gemini") and provider_keys["gemini"] != "local":
            lines.append(f"GEMINI_API_KEY={provider_keys['gemini']}")
    if lines:
        with open(ENV_FILE, "a") as f:
            f.write("\n".join(lines) + "\n")


def save_config(core_name: str, user_name: str, api_key: str,
                provider_keys: dict | None = None):
    """Sauvegarde la configuration."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "core_name": core_name,
        "user_name": user_name,
        "python_path": sys.executable,
        "version": "0.9.7",
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

    # ── Stocker les clés API dans le KeyVault (chiffré) ──
    try:
        from neo_core.infra.security.vault import KeyVault
        vault = KeyVault(data_dir=CONFIG_DIR)
        vault.initialize()
        if api_key:
            vault.store("anthropic_api_key", api_key)
        if provider_keys:
            if provider_keys.get("groq") and provider_keys["groq"] != "local":
                vault.store("groq_api_key", provider_keys["groq"])
            if provider_keys.get("gemini") and provider_keys["gemini"] != "local":
                vault.store("gemini_api_key", provider_keys["gemini"])
            if provider_keys.get("huggingface"):
                vault.store("hf_token", provider_keys["huggingface"])
        vault.close()
        print(f"  {GREEN}✓{RESET} Clés API chiffrées dans le vault")
    except Exception as e:
        print(f"  {YELLOW}⚠{RESET} Vault indisponible ({e}) — clés en fallback .env")
        # Fallback : écrire dans .env si le vault échoue
        _write_env_fallback(api_key, provider_keys)

    # ── .env : uniquement les variables non-sensibles ──
    env_lines = [
        "# Neo Core — Configuration Environnement",
        "# Les clés API sont stockées dans le vault chiffré (data/.vault.db)",
        "",
        f"NEO_CORE_NAME={core_name}",
        f"NEO_USER_NAME={user_name}",
        "NEO_DEBUG=false",
        "NEO_LOG_LEVEL=INFO",
        "",
    ]

    with open(ENV_FILE, "w") as f:
        f.write("\n".join(env_lines))

    os.chmod(ENV_FILE, 0o600)
    print(f"  {GREEN}✓{RESET} Variables d'environnement: {ENV_FILE}")


def configure_hardware_and_providers(api_key: str, existing_env: dict | None = None) -> dict:
    """
    Détection hardware + configuration des providers LLM.

    Retourne un dict avec les clés API et modèles configurés.
    """
    existing_env = existing_env or {}
    provider_keys = {"anthropic": api_key}

    # ── Détection hardware ──
    print(f"  {DIM}⧗ Détection du hardware...{RESET}", end="", flush=True)
    try:
        from neo_core.brain.providers.hardware import HardwareDetector
        profile = HardwareDetector.detect()
        print(f"\r  {GREEN}✓{RESET} {profile.summary()}")
    except Exception as e:
        print(f"\r  {YELLOW}⚠{RESET} Impossible de détecter le hardware: {e}")
        profile = None

    # ── Modèles locaux (Ollama) ──
    print()
    if profile and profile.max_model_size() != "none":
        models = profile.recommend_ollama_models()
        model_names = ", ".join(m["model"] for m in models[:2])
        print(f"  {BOLD}Modèles locaux (Ollama) :{RESET}")
        print(f"  {DIM}Votre machine peut faire tourner des modèles jusqu'à {profile.max_model_size()}{RESET}")
        print(f"  {DIM}Recommandé : {model_names}{RESET}")
        print()

        if ask_confirm("Installer Ollama et les modèles recommandés ?"):
            ollama_installed = HardwareDetector.is_ollama_installed()
            if not ollama_installed:
                print(f"  {DIM}⧗ Installation d'Ollama...{RESET}")
                try:
                    import subprocess
                    result = subprocess.run(
                        "curl -fsSL https://ollama.com/install.sh | sh",
                        shell=True, capture_output=True, text=True, timeout=300,
                    )
                    if result.returncode == 0:
                        print(f"  {GREEN}✓{RESET} Ollama installé")
                        ollama_installed = True
                    else:
                        print(f"  {YELLOW}⚠{RESET} Installation Ollama échouée")
                        print(f"  {CYAN}curl -fsSL https://ollama.com/install.sh | sh{RESET}")
                except Exception:
                    print(f"  {YELLOW}⚠{RESET} Installation Ollama échouée")

            if ollama_installed:
                if not HardwareDetector.is_ollama_running():
                    try:
                        import subprocess
                        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        import time
                        time.sleep(2)
                    except Exception:
                        pass

                for m in models[:2]:
                    print(f"  {DIM}⧗ Téléchargement de {m['model']}...{RESET}", end="", flush=True)
                    try:
                        import subprocess
                        result = subprocess.run(
                            ["ollama", "pull", m["model"]],
                            capture_output=True, text=True, timeout=600,
                        )
                        if result.returncode == 0:
                            print(f"\r  {GREEN}✓{RESET} {m['model']} prêt ({m['size']}, {m['role']})")
                        else:
                            print(f"\r  {YELLOW}⚠{RESET} Échec du téléchargement de {m['model']}")
                    except Exception:
                        print(f"\r  {YELLOW}⚠{RESET} Échec du téléchargement de {m['model']}")

                provider_keys["ollama"] = "local"
    else:
        print(f"  {DIM}Modèles locaux : hardware insuffisant, utilisation cloud uniquement.{RESET}")

    # ── Groq (cloud gratuit) ──
    print()
    print(f"  {BOLD}Modèles cloud gratuits :{RESET}")

    existing_groq = existing_env.get("GROQ_API_KEY", "")
    if existing_groq:
        print(f"\n  {GREEN}✓{RESET} Groq déjà configuré : {DIM}{_mask_key(existing_groq)}{RESET}")
        provider_keys["groq"] = existing_groq
    else:
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

    # ── Gemini (cloud gratuit) ──
    existing_gemini = existing_env.get("GEMINI_API_KEY", "")
    if existing_gemini:
        print(f"\n  {GREEN}✓{RESET} Gemini déjà configuré : {DIM}{_mask_key(existing_gemini)}{RESET}")
        provider_keys["gemini"] = existing_gemini
    else:
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

    # ── Test des modèles configurés ──
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
            from neo_core.brain.providers.registry import ModelRegistry
            from neo_core.brain.providers.anthropic_provider import AnthropicProvider
            from neo_core.brain.providers.groq_provider import GroqProvider
            from neo_core.brain.providers.gemini_provider import GeminiProvider
            from neo_core.brain.providers.ollama_provider import OllamaProvider

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


def _configure_telegram():
    """Configure le bot Telegram (optionnel). Skip si déjà configuré."""

    # Vérifier si Telegram est déjà configuré dans le vault
    already_configured = False
    try:
        from neo_core.infra.security.vault import KeyVault
        if (CONFIG_DIR / ".vault.db").exists():
            vault = KeyVault(data_dir=CONFIG_DIR)
            vault.initialize()
            tg_token_existing = vault.retrieve("telegram_bot_token")
            vault.close()
            if tg_token_existing:
                already_configured = True
                masked = tg_token_existing[:8] + "..." + tg_token_existing[-4:] if len(tg_token_existing) > 12 else "***"
                print(f"  {GREEN}✓{RESET} Bot Telegram déjà configuré : {DIM}{masked}{RESET}")
                if not ask_confirm("Reconfigurer Telegram ?", default=False):
                    return
    except Exception:
        pass

    if not already_configured:
        print(f"  {DIM}Connectez Neo à Telegram pour discuter via votre téléphone.{RESET}")
        print(f"  {DIM}Le bot sera lancé automatiquement avec le daemon.{RESET}")
        print()

        if not ask_confirm("Configurer le bot Telegram ?", default=False):
            print(f"  {DIM}  (ignoré — configurable plus tard avec neo telegram-setup){RESET}")
            return

    print()
    print(f"  {DIM}1. Ouvrez Telegram et cherchez @BotFather{RESET}")
    print(f"  {DIM}2. Envoyez /newbot et suivez les instructions{RESET}")
    print(f"  {DIM}3. Copiez le token du bot ici{RESET}")
    print()

    tg_token = ask("Token du bot Telegram", secret=False)
    if not tg_token:
        print(f"  {DIM}  (ignoré — configurable plus tard avec neo telegram-setup){RESET}")
        return

    print()
    print(f"  {DIM}Pour trouver votre user_id Telegram :{RESET}")
    print(f"  {DIM}  → Envoyez /start à @userinfobot{RESET}")
    print()

    tg_ids_input = ask("User IDs autorisés (séparés par des virgules)")
    try:
        tg_user_ids = [int(x.strip()) for x in tg_ids_input.split(",") if x.strip()]
    except ValueError:
        tg_user_ids = []
        print(f"  {YELLOW}⚠{RESET} IDs invalides — Telegram non configuré")

    if tg_user_ids:
        try:
            from neo_core.vox.integrations.telegram import save_telegram_config
            save_telegram_config(CONFIG_DIR, tg_token, tg_user_ids)
            print(f"  {GREEN}✓{RESET} Token chiffré dans le vault")
            print(f"  {GREEN}✓{RESET} {len(tg_user_ids)} utilisateur(s) autorisé(s)")
        except Exception as e:
            print(f"  {YELLOW}⚠{RESET} Erreur config Telegram: {e}")


def run_setup():
    """
    Point d'entrée du setup — configuration uniquement.

    L'installation système (deps, venv, pip) est gérée par install.sh.
    Cette commande configure : identité, clés API, providers, Telegram.
    """
    print_banner()

    existing_env = _load_existing_env()

    TOTAL_STEPS = 5

    print(f"  {BOLD}Configuration de Neo Core{RESET}")
    print(f"  {DIM}Ce wizard configure votre système en quelques étapes.{RESET}")
    print(f"  {DIM}L'installation des dépendances est gérée par install.sh.{RESET}\n")

    # ─── Étape 1 : Identité ──────────────────────────────────────
    print_step(1, TOTAL_STEPS, "Identité")

    print(f"  {DIM}Donnez un nom à votre système IA.{RESET}")
    print(f"  {DIM}Ce nom sera utilisé par les agents pour se référencer.{RESET}\n")

    existing_core = existing_env.get("NEO_CORE_NAME", "")
    existing_user = existing_env.get("NEO_USER_NAME", "")

    core_name = ask("Nom du Core", default=existing_core or "Neo")
    user_name = ask("Votre nom / pseudonyme", default=existing_user)

    if not user_name:
        user_name = "User"
        print(f"  {DIM}  (nom par défaut: User){RESET}")

    print(f"\n  {GREEN}✓{RESET} Core: {BOLD}{core_name}{RESET} — Utilisateur: {BOLD}{user_name}{RESET}")

    # ─── Étape 2 : Clés API ──────────────────────────────────────
    print_step(2, TOTAL_STEPS, "Connexion Anthropic (optionnel)")

    api_key = configure_auth(existing_env)

    # ─── Étape 3 : Providers LLM + hardware ──────────────────────
    print_step(3, TOTAL_STEPS, "Configuration des modèles LLM")

    provider_keys = configure_hardware_and_providers(api_key, existing_env)

    # ─── Étape 4 : Sauvegarde + embedding ────────────────────────
    print_step(4, TOTAL_STEPS, "Sauvegarde et téléchargement")

    save_config(core_name, user_name, api_key, provider_keys)

    _download_embedding_model(provider_keys)

    # Test connexion
    if api_key:
        test_connection(api_key)

    # ─── Étape 5 : Telegram (optionnel) ──────────────────────────
    print_step(5, TOTAL_STEPS, "Bot Telegram (optionnel)")

    _configure_telegram()

    # ─── Résumé final ─────────────────────────────────────────────
    active_providers = [k for k, v in provider_keys.items() if v]
    provider_list = ", ".join(active_providers) if active_providers else "Mode mock"

    print(f"""
{CYAN}{BOLD}  ╔═══════════════════════════════════════════════╗
  ║          Configuration terminée !              ║
  ╚═══════════════════════════════════════════════╝{RESET}

  {BOLD}Configuration :{RESET}
    Nom du Core   : {GREEN}{core_name}{RESET}
    Utilisateur   : {GREEN}{user_name}{RESET}
    Providers LLM : {GREEN}{provider_list}{RESET}

  {BOLD}Commandes utiles :{RESET}
    {CYAN}neo chat{RESET}            Discuter avec {core_name}
    {CYAN}neo status{RESET}          État du système + daemon
    {CYAN}neo logs{RESET}            Voir les logs du daemon
    {CYAN}neo stop{RESET}            Arrêter le daemon
    {CYAN}neo restart{RESET}         Redémarrer le daemon
    {CYAN}neo telegram-setup{RESET}  Configurer le bot Telegram
""")
