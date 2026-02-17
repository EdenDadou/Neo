"""
Neo Core — Configuration Globale
================================
Gestion centralisée de la configuration : LLM, modèles, chemins, paramètres.
Supporte .env pour les clés API et data/neo_config.json pour les paramètres du wizard.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Chemin racine du projet (parent de neo_core/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Charge les variables d'environnement depuis .env à la racine du projet
load_dotenv(_PROJECT_ROOT / ".env")

# Chemin vers la config du wizard
_CONFIG_FILE = _PROJECT_ROOT / "data" / "neo_config.json"


def _load_wizard_config() -> dict:
    """Charge la configuration créée par le wizard d'installation."""
    if _CONFIG_FILE.exists():
        try:
            with open(_CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.debug("Failed to load wizard config: %s", e)
    return {}


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


def get_agent_model(agent_name: str) -> AgentModelConfig:
    """
    Récupère la config modèle pour un agent donné.

    Stratégie :
    1. Si le ModelRegistry a des modèles testés → routing dynamique
       (local gratuit > cloud gratuit > cloud payant)
    2. Sinon → fallback aux modèles hardcodés (Anthropic)
    """
    try:
        from neo_core.providers.registry import get_model_registry
        registry = get_model_registry()
        stats = registry.get_stats()

        if stats.get("available_models", 0) > 0:
            model = registry.get_best_for(agent_name)
            if model:
                # Récupérer la temperature/max_tokens de la config hardcodée
                defaults = AGENT_MODELS.get(agent_name, AGENT_MODELS["brain"])
                return AgentModelConfig(
                    model=model.model_name,
                    temperature=defaults.temperature,
                    max_tokens=min(defaults.max_tokens, model.max_output_tokens),
                    provider=model.provider,
                    model_id=model.model_id,
                )
    except Exception as e:
        logger.debug("Provider registry lookup failed for '%s': %s", agent_name, e)

    # Fallback : modèles hardcodés (Anthropic)
    return AGENT_MODELS.get(agent_name, AGENT_MODELS["brain"])


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
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        # Toujours strip la clé (espaces, newlines, guillemets résiduels)
        if self.api_key:
            self.api_key = self.api_key.strip().strip('"').strip("'")


@dataclass
class MemoryConfig:
    """Configuration du système mémoire."""
    storage_path: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "memory")
    vector_db: str = "chromadb"
    max_context_tokens: int = 2048
    max_results: int = 5
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
class NeoConfig:
    """Configuration principale de Neo Core."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
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

    # Provider configuration
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY", None))
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", None))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    provider_mode: str = field(default_factory=lambda: os.getenv("NEO_PROVIDER_MODE", "economic"))

    # Data directory
    data_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")

    def __post_init__(self):
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
        # Vérifier les autres providers via le .env
        if os.getenv("GROQ_API_KEY"):
            return False
        if os.getenv("GEMINI_API_KEY"):
            return False
        # Pas de vérification Ollama ici (trop lent au startup)
        return True

    def reload(self) -> None:
        """Recharge la configuration depuis le .env et neo_config.json."""
        from dotenv import load_dotenv
        load_dotenv(override=True)
        # Re-read env vars
        self.debug = os.getenv("NEO_DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("NEO_LOG_LEVEL", "INFO")
        self.groq_api_key = os.getenv("GROQ_API_KEY", None)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", None)
        self.provider_mode = os.getenv("NEO_PROVIDER_MODE", "economic")
        # Re-read LLM config
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if api_key:
            self.llm.api_key = api_key.strip().strip("'\"")
        # Re-read wizard config
        wizard = _load_wizard_config()
        if wizard:
            if "core_name" in wizard:
                self.core_name = wizard["core_name"]
            if "user_name" in wizard:
                self.user_name = wizard["user_name"]
        logger.info("Configuration reloaded from .env and neo_config.json")

    def is_installed(self) -> bool:
        """Vérifie si le wizard d'installation a été exécuté."""
        return _CONFIG_FILE.exists()


# Instance globale par défaut
default_config = NeoConfig()
