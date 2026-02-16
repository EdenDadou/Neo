"""
Neo Core — Configuration Globale
================================
Gestion centralisée de la configuration : LLM, modèles, chemins, paramètres.
Supporte .env pour les clés API et data/neo_config.json pour les paramètres du wizard.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

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
        except (json.JSONDecodeError, IOError):
            pass
    return {}


@dataclass
class LLMConfig:
    """Configuration du modèle de langage."""
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

    def is_mock_mode(self) -> bool:
        """Indique si on fonctionne sans clé API (mode mock pour les tests)."""
        return not self.llm.api_key

    def is_installed(self) -> bool:
        """Vérifie si le wizard d'installation a été exécuté."""
        return _CONFIG_FILE.exists()


# Instance globale par défaut
default_config = NeoConfig()
