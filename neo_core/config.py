"""
Neo Core — Configuration Globale
================================
Gestion centralisée de la configuration : LLM, modèles, chemins, paramètres.
Supporte .env pour les clés API et data/neo_config.json pour les paramètres du wizard.
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Chemin racine du projet (parent de neo_core/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Charge les variables d'environnement depuis .env à la racine du projet
try:
    load_dotenv(_PROJECT_ROOT / ".env")
except PermissionError:
    logger.warning("Cannot read .env (permission denied) — using existing environment")
except Exception as e:
    logger.warning("Failed to load .env: %s — using existing environment", e)

# Chemin vers la config du wizard
_CONFIG_FILE = _PROJECT_ROOT / "data" / "neo_config.json"


def _load_wizard_config() -> dict:
    """Charge la configuration créée par le wizard d'installation."""
    try:
        if _CONFIG_FILE.exists():
            try:
                with open(_CONFIG_FILE) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.debug("Failed to load wizard config: %s", e)
    except PermissionError:
        logger.warning("Cannot read %s (permission denied) — using defaults", _CONFIG_FILE)
    return {}


# ─── Lecture des secrets depuis le KeyVault ───────────────

_vault_cache: dict[str, Optional[str]] = {}
_vault_loaded = False


def _load_vault_secrets() -> None:
    """Charge tous les secrets du vault en mémoire (une seule fois)."""
    global _vault_loaded
    if _vault_loaded:
        return
    _vault_loaded = True
    try:
        from neo_core.infra.security.vault import KeyVault
        data_dir = _PROJECT_ROOT / "data"
        if not (data_dir / ".vault.db").exists():
            return
        vault = KeyVault(data_dir=data_dir)
        vault.initialize()
        for name in ("anthropic_api_key", "groq_api_key", "gemini_api_key", "hf_token"):
            val = vault.retrieve(name)
            if val:
                _vault_cache[name] = val
        vault.close()
    except Exception as e:
        logger.debug("Vault load skipped: %s", e)


def _get_secret(vault_name: str, env_name: str) -> Optional[str]:
    """
    Récupère un secret : vault d'abord, puis fallback os.getenv().

    Args:
        vault_name: Nom du secret dans le vault (ex: "anthropic_api_key")
        env_name: Variable d'environnement fallback (ex: "ANTHROPIC_API_KEY")
    """
    _load_vault_secrets()
    val = _vault_cache.get(vault_name)
    if val:
        return val
    return os.getenv(env_name) or None


@dataclass
class AgentModelConfig:
    """Configuration du modèle pour un agent spécifique."""
    model: str = "claude-sonnet-4-5-20250929"
    temperature: float = 0.7
    max_tokens: int = 4096
    # ── Champs multi-provider (remplis par le ModelRegistry) ──
    provider: str = "anthropic"      # "anthropic" | "ollama" | "groq" | "gemini"
    model_id: str = ""               # "provider:model_name" (ex: "groq:llama-3.3-70b")


# ─── Modèles par agent ──────────────────────────────────
# Chaque agent a un modèle choisi selon ses besoins et les ressources.
#
# Haiku  = rapide, économique, idéal pour les tâches simples
# Sonnet = puissant, pour l'orchestration et les tâches complexes
# Opus   = premium, réservé aux tâches critiques (non utilisé par défaut)

AGENT_MODELS = {
    # Vox : interface utilisateur — besoin de rapidité, pas de raisonnement lourd
    "vox": AgentModelConfig(
        model="claude-haiku-4-5-20251001",
        temperature=0.6,
        max_tokens=2048,
    ),
    # Brain : orchestrateur — besoin de raisonnement, planification, décision
    "brain": AgentModelConfig(
        model="claude-sonnet-4-5-20250929",
        temperature=0.7,
        max_tokens=4096,
    ),
    # Memory : consolidation intelligente — tâches courtes, besoin d'efficacité
    "memory": AgentModelConfig(
        model="claude-haiku-4-5-20251001",
        temperature=0.3,
        max_tokens=2048,
    ),
    # Workers par type — les tâches complexes méritent Sonnet
    "worker:researcher": AgentModelConfig(
        model="claude-sonnet-4-5-20250929",
        temperature=0.5,
        max_tokens=4096,
    ),
    "worker:coder": AgentModelConfig(
        model="claude-sonnet-4-5-20250929",
        temperature=0.3,
        max_tokens=4096,
    ),
    "worker:analyst": AgentModelConfig(
        model="claude-sonnet-4-5-20250929",
        temperature=0.5,
        max_tokens=4096,
    ),
    # Workers légers — Haiku suffit
    "worker:summarizer": AgentModelConfig(
        model="claude-haiku-4-5-20251001",
        temperature=0.5,
        max_tokens=2048,
    ),
    "worker:writer": AgentModelConfig(
        model="claude-haiku-4-5-20251001",
        temperature=0.8,
        max_tokens=4096,
    ),
    "worker:translator": AgentModelConfig(
        model="claude-haiku-4-5-20251001",
        temperature=0.3,
        max_tokens=2048,
    ),
    "worker:generic": AgentModelConfig(
        model="claude-haiku-4-5-20251001",
        temperature=0.5,
        max_tokens=2048,
    ),
}


def _load_auto_tuning() -> dict:
    """
    Load auto-tuning overrides from data/auto_tuning.json.
    Returns empty dict if file doesn't exist (backwards compatible).
    """
    tuning_file = _PROJECT_ROOT / "data" / "auto_tuning.json"
    if not tuning_file.exists():
        return {}

    try:
        with open(tuning_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to load auto-tuning: %s", e)
        return {}


# Bornes de sécurité pour les paramètres auto-tuned
_TUNING_BOUNDS = {
    "temperature": (0.1, 1.0),
    "max_tokens": (256, 8192),
}


def _apply_tuning_overrides(config: AgentModelConfig, tuning: dict, agent_name: str) -> None:
    """Apply auto-tuning overrides with bounds checking (mutates config in-place)."""
    if agent_name not in tuning:
        return
    overrides = tuning[agent_name]
    if "temperature" in overrides:
        try:
            val = float(overrides["temperature"])
            lo, hi = _TUNING_BOUNDS["temperature"]
            config.temperature = max(lo, min(hi, val))
        except (ValueError, TypeError):
            logger.warning("Invalid auto-tuning temperature for %s: %s", agent_name, overrides["temperature"])
    if "max_tokens" in overrides:
        try:
            val = int(overrides["max_tokens"])
            lo, hi = _TUNING_BOUNDS["max_tokens"]
            config.max_tokens = max(lo, min(hi, val))
        except (ValueError, TypeError):
            logger.warning("Invalid auto-tuning max_tokens for %s: %s", agent_name, overrides["max_tokens"])


def get_agent_model(agent_name: str) -> AgentModelConfig:
    """
    Récupère la config modèle pour un agent donné.

    Stratégie :
    1. Si le ModelRegistry a des modèles testés → routing dynamique
       (local gratuit > cloud gratuit > cloud payant)
    2. Sinon → fallback aux modèles hardcodés (Anthropic)
    3. Appliquer les surcharges d'auto-tuning si disponibles
    """
    # Load tuning once for both code paths
    tuning = _load_auto_tuning()

    try:
        from neo_core.brain.providers.registry import get_model_registry
        registry = get_model_registry()
        stats = registry.get_stats()

        if stats.get("available_models", 0) > 0:
            model = registry.get_best_for(agent_name)
            if model:
                # Récupérer la temperature/max_tokens de la config hardcodée
                defaults = AGENT_MODELS.get(agent_name, AGENT_MODELS["brain"])
                config = AgentModelConfig(
                    model=model.model_name,
                    temperature=defaults.temperature,
                    max_tokens=min(defaults.max_tokens, model.max_output_tokens),
                    provider=model.provider,
                    model_id=model.model_id,
                )

                # Appliquer les surcharges d'auto-tuning avec bounds checking
                _apply_tuning_overrides(config, tuning, agent_name)

                return config
    except Exception as e:
        logger.debug("Provider registry lookup failed for '%s': %s", agent_name, e)

    # Fallback : modèles hardcodés (Anthropic)
    config = AGENT_MODELS.get(agent_name, AGENT_MODELS["brain"])

    # Appliquer les surcharges d'auto-tuning avec bounds checking
    _apply_tuning_overrides(config, tuning, agent_name)

    return config


@dataclass
class LLMConfig:
    """Configuration du modèle de langage (legacy — utilisée pour l'auth globale)."""
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5-20250929"
    api_key: Optional[str] = field(default=None, repr=False)
    temperature: float = 0.7
    max_tokens: int = 4096

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = _get_secret("anthropic_api_key", "ANTHROPIC_API_KEY")
        # Toujours strip la clé (espaces, newlines, guillemets résiduels)
        if self.api_key:
            self.api_key = self.api_key.strip().strip('"').strip("'")


@dataclass
class MemoryConfig:
    """Configuration du système mémoire."""
    storage_path: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "memory")
    vector_db: str = "faiss"
    max_context_tokens: int = 1024
    max_results: int = 3
    similarity_threshold: float = 0.3


@dataclass
class ResilienceConfig:
    """Configuration de la résilience (retry, circuit breaker, timeouts)."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    api_timeout: float = 60.0
    worker_timeout: float = 180.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery: float = 60.0
    max_tool_iterations: int = 10


@dataclass
class SelfImprovementConfig:
    """Configuration des niveaux d'auto-amélioration (Levels 1-4)."""
    # Level 1 — Auto-tuning (géré aussi dans HeartbeatConfig)
    auto_tuning_enabled: bool = True
    # Level 2 — Plugin system (toujours actif)
    # Level 3 — Self-patching
    self_patching_enabled: bool = True
    patch_detection_threshold: int = 3   # Nb erreurs min avant de générer un patch
    patch_validation_min_improvement: float = 0.5  # Amélioration min pour valider
    # Level 4 — Autonomous tool creation
    tool_generation_enabled: bool = True
    tool_pruning_days: int = 7           # Supprimer outils inutilisés après N jours
    tool_min_success_rate: float = 0.3   # Taux succès min (sinon deprecated)


@dataclass
class NeoConfig:
    """Configuration principale de Neo Core."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    self_improvement: SelfImprovementConfig = field(default_factory=SelfImprovementConfig)
    debug: bool = field(default_factory=lambda: os.getenv("NEO_DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("NEO_LOG_LEVEL", "INFO"))

    # Paramètres du wizard (personnalisation)
    core_name: str = field(default_factory=lambda: os.getenv("NEO_CORE_NAME", "Neo"))
    user_name: str = field(default_factory=lambda: os.getenv("NEO_USER_NAME", "Utilisateur"))

    # Noms des agents
    vox_name: str = "Vox"
    brain_name: str = "Brain"
    memory_agent_name: str = "Memory"

    # API configuration
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("NEO_API_KEY", None))
    api_host: str = field(default_factory=lambda: os.getenv("NEO_API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("NEO_API_PORT", "8000")))

    # Provider configuration (vault → env fallback)
    groq_api_key: Optional[str] = field(default_factory=lambda: _get_secret("groq_api_key", "GROQ_API_KEY"))
    gemini_api_key: Optional[str] = field(default_factory=lambda: _get_secret("gemini_api_key", "GEMINI_API_KEY"))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    provider_mode: str = field(default_factory=lambda: os.getenv("NEO_PROVIDER_MODE", "economic"))

    # Data directory
    data_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")

    def __post_init__(self):
        # Lock pour protéger reload() contre les accès concurrents
        self._reload_lock = threading.Lock()
        # Surcharge avec la config du wizard si disponible
        wizard = _load_wizard_config()
        if wizard:
            if "core_name" in wizard and self.core_name == "Neo":
                self.core_name = wizard["core_name"]
            if "user_name" in wizard and self.user_name == "Utilisateur":
                self.user_name = wizard["user_name"]

    def validate(self) -> bool:
        """Vérifie que la configuration est valide pour une utilisation avec un LLM réel."""
        if not self.llm.api_key:
            return False
        return True

    def is_provider_configured(self, name: str) -> bool:
        """Vérifie si un provider spécifique est configuré."""
        checks = {
            "anthropic": bool(self.llm.api_key),
            "groq": bool(self.groq_api_key),
            "gemini": bool(self.gemini_api_key),
            "ollama": True,  # Toujours "configuré", la connectivité est vérifiée au runtime
        }
        return checks.get(name, False)

    def is_mock_mode(self) -> bool:
        """
        Indique si on fonctionne en mode mock (réponses simulées).

        Le mode mock est désactivé si au moins un provider est disponible :
        - Anthropic (ANTHROPIC_API_KEY)
        - Groq (GROQ_API_KEY)
        - Gemini (GEMINI_API_KEY)
        - Ollama (serveur local)
        """
        # Si une clé Anthropic existe, pas mock
        if self.llm.api_key:
            return False
        # Vérifier les autres providers (vault → env)
        if self.groq_api_key:
            return False
        if self.gemini_api_key:
            return False
        # Pas de vérification Ollama ici (trop lent au startup)
        return True

    def reload(self) -> None:
        """Recharge la configuration depuis le vault, .env et neo_config.json (thread-safe)."""
        global _vault_loaded
        with self._reload_lock:
            from dotenv import load_dotenv
            load_dotenv(override=True)
            # Forcer le rechargement du vault
            _vault_loaded = False
            _vault_cache.clear()
            # Re-read env vars
            self.debug = os.getenv("NEO_DEBUG", "false").lower() == "true"
            self.log_level = os.getenv("NEO_LOG_LEVEL", "INFO")
            self.groq_api_key = _get_secret("groq_api_key", "GROQ_API_KEY")
            self.gemini_api_key = _get_secret("gemini_api_key", "GEMINI_API_KEY")
            self.provider_mode = os.getenv("NEO_PROVIDER_MODE", "economic")
            # Re-read LLM config (vault → env fallback)
            api_key = _get_secret("anthropic_api_key", "ANTHROPIC_API_KEY")
            if api_key:
                self.llm.api_key = api_key.strip().strip("'\"")
            # Re-read wizard config
            wizard = _load_wizard_config()
            if wizard:
                if "core_name" in wizard:
                    self.core_name = wizard["core_name"]
                if "user_name" in wizard:
                    self.user_name = wizard["user_name"]
            logger.info("Configuration reloaded from vault, .env and neo_config.json")

    def is_installed(self) -> bool:
        """Vérifie si le wizard d'installation a été exécuté."""
        return _CONFIG_FILE.exists()


# Instance globale par défaut
default_config = NeoConfig()
