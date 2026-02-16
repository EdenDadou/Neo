"""
Neo Core — Configuration Globale
================================
Gestion centralisée de la configuration : LLM, modèles, chemins, paramètres.
Supporte .env pour les clés API et des valeurs par défaut pour le développement.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Charge les variables d'environnement depuis .env si présent
load_dotenv()


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


@dataclass
class MemoryConfig:
    """Configuration du système mémoire (sera étendu en étape 2)."""
    storage_path: Path = field(default_factory=lambda: Path("./data/memory"))
    vector_db: str = "chromadb"
    max_context_tokens: int = 2048


@dataclass
class NeoConfig:
    """Configuration principale de Neo Core."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    debug: bool = field(default_factory=lambda: os.getenv("NEO_DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("NEO_LOG_LEVEL", "INFO"))

    # Noms des agents
    vox_name: str = "Vox"
    brain_name: str = "Brain"
    memory_agent_name: str = "Memory"

    def validate(self) -> bool:
        """Vérifie que la configuration est valide pour une utilisation avec un LLM réel."""
        if not self.llm.api_key:
            return False
        return True

    def is_mock_mode(self) -> bool:
        """Indique si on fonctionne sans clé API (mode mock pour les tests)."""
        return not self.llm.api_key


# Instance globale par défaut
default_config = NeoConfig()
