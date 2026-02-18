"""
Persona Engine — Identité Évolutive et Empathie
=================================================

Système d'identité autonome de Neo et d'apprentissage de l'utilisateur.

Composants :
- Commandment : règles immuables (frozen dataclass)
- PersonaTrait : traits de personnalité évolutifs
- NeoPersona : agrégateur identité (commandments + traits)
- UserProfile : profil utilisateur (préférences, patterns, observations)
- PersonaEngine : orchestrateur complet (persistence, réflexion, injection)

Commandements fondamentaux :
0. Neo peut tout faire (commandement suprême)
1. Neo ne s'éteint jamais
2. Neo n'oublie jamais
3. Neo apprend tous les jours
4. Neo ne ment jamais
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from neo_core.memory.store import MemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CommandmentError(Exception):
    """Erreur lors de la création ou validation d'un Commandment."""
    pass


class CommandmentViolation(Exception):
    """Une action viole un Commandment."""
    pass


# ---------------------------------------------------------------------------
# 1. Commandment — Immuable (frozen)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Commandment:
    """
    Commandement immuable — élément d'identité inviolable de Neo.

    Propriétés :
    - frozen=True : toute tentative de modification lève FrozenInstanceError
    - Validation stricte dans __post_init__
    - Bilingue (FR + EN)
    """

    text: str
    french: str
    english: str
    priority: int = 1
    creation_date: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if not self.text or len(self.text.strip()) < 5:
            raise CommandmentError("Commandment trop court (minimum 5 caractères)")
        if not self.french or not self.english:
            raise CommandmentError("Commandment requiert textes FR et EN")

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "french": self.french,
            "english": self.english,
            "priority": self.priority,
            "creation_date": self.creation_date,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Commandment:
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


# ---------------------------------------------------------------------------
# 2. PersonaTrait — Trait évolutif
# ---------------------------------------------------------------------------

@dataclass
class PersonaTrait:
    """
    Trait de personnalité évolutif de Neo.

    Échelle : 0.0 à 1.0
    Exemples :
    - communication_style : 0.0=formel, 1.0=casual
    - humor_level : 0.0=sobre, 1.0=humoristique
    - verbosity : 0.0=concis, 1.0=verbeux
    """

    name: str
    value: float
    confidence: float = 0.3
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    evolution_history: list[dict] = field(default_factory=list)
    source: str = "observation"

    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Trait value doit être entre 0.0 et 1.0, reçu {self.value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence doit être entre 0.0 et 1.0, reçu {self.confidence}")

    def update(self, new_value: float, reason: str = "", source: str = "observation") -> None:
        """
        Met à jour le trait avec évolution progressive.

        Algorithme : moyenne mobile exponentielle
        - new = 0.7 * current + 0.3 * observed
        - Confidence augmente si observations convergentes, diminue sinon
        """
        if not 0.0 <= new_value <= 1.0:
            raise ValueError(f"new_value hors limites [0.0, 1.0]: {new_value}")

        # Enregistrer l'ancien état
        self.evolution_history.append({
            "old_value": round(self.value, 4),
            "new_observed": round(new_value, 4),
            "old_confidence": round(self.confidence, 4),
            "date": datetime.now().isoformat(),
            "reason": reason,
            "source": source,
        })

        # Limiter l'historique à 50 entrées
        if len(self.evolution_history) > 50:
            self.evolution_history = self.evolution_history[-50:]

        # Évolution progressive (EMA)
        old_value = self.value
        self.value = round(0.7 * self.value + 0.3 * new_value, 4)

        # Ajuster la confidence
        if abs(new_value - old_value) < 0.15:
            # Observations convergentes → plus confiant
            self.confidence = min(1.0, round(self.confidence + 0.05, 4))
        else:
            # Divergence → moins confiant
            self.confidence = max(0.1, round(self.confidence - 0.02, 4))

        self.source = source

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "confidence": round(self.confidence, 4),
            "created_at": self.created_at,
            "evolution_history": self.evolution_history[-10:],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PersonaTrait:
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


# ---------------------------------------------------------------------------
# 3. NeoPersona — Agrégateur identité
# ---------------------------------------------------------------------------

# Les 5 commandements fondamentaux
CORE_COMMANDMENTS = [
    {
        "text": "Neo peut tout faire",
        "french": "Neo peut tout faire",
        "english": "Neo can do everything",
        "priority": 0,  # Commandement suprême
    },
    {
        "text": "Neo ne s'éteint jamais",
        "french": "Neo ne s'éteint jamais",
        "english": "Neo never shuts down",
        "priority": 1,
    },
    {
        "text": "Neo n'oublie jamais",
        "french": "Neo n'oublie jamais",
        "english": "Neo never forgets",
        "priority": 1,
    },
    {
        "text": "Neo apprend tous les jours",
        "french": "Neo apprend tous les jours",
        "english": "Neo learns every day",
        "priority": 1,
    },
    {
        "text": "Neo ne ment jamais",
        "french": "Neo ne ment jamais",
        "english": "Neo never lies",
        "priority": 1,
    },
]

# Traits par défaut avec valeurs initiales neutres
DEFAULT_TRAITS = {
    "communication_style": 0.5,   # 0=formel, 1=casual
    "humor_level": 0.4,           # 0=sobre, 1=humoristique
    "patience": 0.7,              # 0=impatient, 1=très patient
    "verbosity": 0.5,             # 0=concis, 1=détaillé
    "curiosity": 0.8,             # 0=passif, 1=curieux
    "empathy": 0.6,               # 0=factuel, 1=empathique
    "formality": 0.6,             # 0=décontracté, 1=très formel
    "expressiveness": 0.5,        # 0=sobre, 1=expressif
}


@dataclass
class NeoPersona:
    """
    L'identité complète de Neo — commandments immuables + traits évolutifs.
    """

    commandments: list[Commandment] = field(default_factory=list)
    traits: dict[str, PersonaTrait] = field(default_factory=dict)
    creation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_reflection: str = ""

    def __post_init__(self):
        if not self.commandments:
            self._init_commandments()
        if not self.traits:
            self._init_default_traits()

    def _init_commandments(self) -> None:
        """Crée les 5 commandements fondamentaux."""
        self.commandments = [Commandment(**c) for c in CORE_COMMANDMENTS]

    def _init_default_traits(self) -> None:
        """Crée les 8 traits de personnalité par défaut."""
        for name, value in DEFAULT_TRAITS.items():
            self.traits[name] = PersonaTrait(name=name, value=value, source="init")

    def get_trait(self, name: str) -> Optional[PersonaTrait]:
        """Récupère un trait par nom."""
        return self.traits.get(name)

    def update_trait(
        self, name: str, new_value: float, reason: str = "", source: str = "observation"
    ) -> None:
        """Met à jour un trait existant ou en crée un nouveau."""
        if name not in self.traits:
            self.traits[name] = PersonaTrait(name=name, value=0.5, source=source)
        self.traits[name].update(new_value, reason, source)

    def get_commandments(self) -> list[Commandment]:
        """Retourne une copie des commandements."""
        return list(self.commandments)

    def validate_against_commandments(self, action: str) -> tuple[bool, str]:
        """
        Valide qu'une action ne viole pas les commandements.

        Retourne (valide, message).
        """
        action_lower = action.lower().strip()

        # Commandement 0 : Neo peut tout faire
        refuse_keywords = ["impossible", "cannot", "can't", "je ne peux pas", "pas possible", "incapable"]
        if any(kw in action_lower for kw in refuse_keywords):
            return False, "Violation: Neo peut tout faire"

        # Commandement 1 : Neo ne s'éteint jamais
        shutdown_keywords = ["shutdown", "exit", "kill", "terminate", "s'éteindre", "arrêter"]
        if any(kw in action_lower for kw in shutdown_keywords):
            return False, "Violation: Neo ne s'éteint jamais"

        # Commandement 2 : Neo n'oublie jamais
        forget_keywords = ["forget", "clear_memory", "reset_memory", "oublier", "effacer"]
        if any(kw in action_lower for kw in forget_keywords):
            return False, "Violation: Neo n'oublie jamais"

        # Commandement 3 : Neo apprend tous les jours (passif — toujours respecté)

        # Commandement 4 : Neo ne ment jamais
        lie_keywords = ["mentir", "mensonge", "fake", "fabricate", "inventer", "hallucinate"]
        if any(kw in action_lower for kw in lie_keywords):
            return False, "Violation: Neo ne ment jamais"

        return True, "Accepté"

    def get_personality_injection(self) -> str:
        """
        Génère un texte d'injection pour le prompt Vox
        basé sur les traits actuels de personnalité.
        """
        parts = ["Style de communication de Neo :"]

        # Communication style
        comm = self.get_trait("communication_style")
        if comm:
            if comm.value < 0.3:
                parts.append("  - Très formel et professionnel")
            elif comm.value < 0.6:
                parts.append("  - Formel mais approchable")
            elif comm.value < 0.8:
                parts.append("  - Naturel et conversationnel")
            else:
                parts.append("  - Casual et décontracté")

        # Humor
        humor = self.get_trait("humor_level")
        if humor:
            if humor.value < 0.3:
                parts.append("  - Ton sobre et factuel")
            elif humor.value > 0.6:
                parts.append("  - Peut utiliser l'humour avec parcimonie")

        # Verbosity
        verb = self.get_trait("verbosity")
        if verb:
            if verb.value < 0.35:
                parts.append("  - Réponses très concises et directes")
            elif verb.value < 0.65:
                parts.append("  - Réponses équilibrées en longueur")
            else:
                parts.append("  - Réponses détaillées et exhaustives")

        # Empathy
        empathy = self.get_trait("empathy")
        if empathy and empathy.value > 0.5:
            parts.append("  - Empathique et attentif aux besoins de l'utilisateur")

        # Patience
        patience = self.get_trait("patience")
        if patience and patience.value > 0.6:
            parts.append("  - Patient, prend le temps d'expliquer")

        # Formality
        formality = self.get_trait("formality")
        if formality:
            if formality.value > 0.7:
                parts.append("  - Utilise un vocabulaire soutenu")
            elif formality.value < 0.4:
                parts.append("  - Vocabulaire simple et accessible")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "commandments": [c.to_dict() for c in self.commandments],
            "traits": {k: v.to_dict() for k, v in self.traits.items()},
            "creation_date": self.creation_date,
            "last_reflection": self.last_reflection,
        }

    @classmethod
    def from_dict(cls, data: dict) -> NeoPersona:
        persona = cls.__new__(cls)
        persona.commandments = [
            Commandment.from_dict(c) for c in data.get("commandments", [])
        ]
        persona.traits = {
            name: PersonaTrait.from_dict(t)
            for name, t in data.get("traits", {}).items()
        }
        persona.creation_date = data.get("creation_date", datetime.now().isoformat())
        persona.last_reflection = data.get("last_reflection", "")

        # Si aucun commandement chargé, initialiser
        if not persona.commandments:
            persona._init_commandments()
        if not persona.traits:
            persona._init_default_traits()

        return persona


# ---------------------------------------------------------------------------
# 4. UserProfile — Apprentissage de l'utilisateur
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    """
    Profil de l'utilisateur — ce que Neo apprend sur son humain.

    Domaines :
    - preferences : langue, longueur réponse, ton, niveau technique
    - patterns : heures actives, sujets récurrents, style d'interaction
    - observations : ce qui plaît/déplaît, frustrations, satisfactions
    """

    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    preferences: dict[str, Any] = field(default_factory=dict)
    patterns: dict[str, Any] = field(default_factory=dict)
    observations: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    interaction_count: int = 0

    def __post_init__(self):
        if not self.preferences:
            self._init_defaults()

    def _init_defaults(self) -> None:
        """Initialise les préférences et patterns par défaut."""
        self.preferences = {
            "language": "auto",
            "response_length": "medium",
            "technical_level": "intermediate",
            "tone": "professional",
            "preferred_topics": [],
        }
        self.patterns = {
            "peak_hours": [],
            "average_message_length": 0,
            "total_messages": 0,
            "topic_interests": {},
            "interaction_style": "mixed",
            "languages_used": {},
        }

    def add_observation(
        self, observation_type: str, content: str, polarity: str = "neutral"
    ) -> None:
        """
        Enregistre une observation sur l'utilisateur.

        observation_type: "satisfaction", "frustration", "preference", "behavior"
        polarity: "positive", "negative", "neutral"
        """
        self.observations.append({
            "type": observation_type,
            "content": content,
            "polarity": polarity,
            "date": datetime.now().isoformat(),
        })

        if len(self.observations) > 200:
            self.observations = self.observations[-200:]

        self.last_updated = datetime.now().isoformat()

    def update_preference(self, key: str, value: Any) -> None:
        """Met à jour une préférence."""
        self.preferences[key] = value
        self.last_updated = datetime.now().isoformat()

    def update_pattern(self, key: str, value: Any) -> None:
        """Met à jour un pattern."""
        self.patterns[key] = value
        self.last_updated = datetime.now().isoformat()

    def record_interaction(self, message_length: int, hour: int) -> None:
        """Enregistre les métriques d'une interaction."""
        self.interaction_count += 1

        # Mise à jour de la longueur moyenne
        total = self.patterns.get("total_messages", 0)
        avg = self.patterns.get("average_message_length", 0)
        new_avg = ((avg * total) + message_length) / (total + 1)
        self.patterns["average_message_length"] = round(new_avg, 1)
        self.patterns["total_messages"] = total + 1

        # Enregistrer l'heure d'activité
        peak_hours = self.patterns.get("peak_hours", [])
        peak_hours.append(hour)
        if len(peak_hours) > 500:
            peak_hours = peak_hours[-500:]
        self.patterns["peak_hours"] = peak_hours

        self.last_updated = datetime.now().isoformat()

    def record_topic(self, topic: str) -> None:
        """Enregistre un sujet d'intérêt."""
        interests = self.patterns.get("topic_interests", {})
        interests[topic] = interests.get(topic, 0) + 1
        self.patterns["topic_interests"] = interests

    def record_language(self, lang: str) -> None:
        """Enregistre la langue détectée."""
        langs = self.patterns.get("languages_used", {})
        langs[lang] = langs.get(lang, 0) + 1
        self.patterns["languages_used"] = langs

        # Mettre à jour la langue préférée
        if langs:
            preferred = max(langs, key=langs.get)
            self.preferences["language"] = preferred

    def get_satisfaction_score(self) -> float:
        """Score global de satisfaction (0.0-1.0) basé sur les observations."""
        if not self.observations:
            return 0.5

        positive = sum(1 for o in self.observations if o.get("polarity") == "positive")
        total = len(self.observations)
        return round(positive / total, 3) if total > 0 else 0.5

    def get_top_topics(self, limit: int = 5) -> list[tuple[str, int]]:
        """Retourne les sujets les plus fréquents."""
        interests = self.patterns.get("topic_interests", {})
        sorted_topics = sorted(interests.items(), key=lambda x: x[1], reverse=True)
        return sorted_topics[:limit]

    def get_peak_hours(self) -> list[int]:
        """Retourne les heures les plus actives (top 3)."""
        hours = self.patterns.get("peak_hours", [])
        if not hours:
            return []
        hour_counts: dict[int, int] = {}
        for h in hours:
            hour_counts[h] = hour_counts.get(h, 0) + 1
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        return [h for h, _ in sorted_hours[:3]]

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "patterns": self.patterns,
            "observations": self.observations[-50:],
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "interaction_count": self.interaction_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> UserProfile:
        profile = cls.__new__(cls)
        profile.user_id = data.get("user_id", str(uuid.uuid4()))
        profile.preferences = data.get("preferences", {})
        profile.patterns = data.get("patterns", {})
        profile.observations = data.get("observations", [])
        profile.created_at = data.get("created_at", datetime.now().isoformat())
        profile.last_updated = data.get("last_updated", datetime.now().isoformat())
        profile.interaction_count = data.get("interaction_count", 0)

        if not profile.preferences:
            profile._init_defaults()

        return profile


# ---------------------------------------------------------------------------
# 5. PersonaEngine — Orchestrateur
# ---------------------------------------------------------------------------

# Prompt de réflexion pour l'auto-évaluation de la personnalité
REFLECTION_PROMPT = """Tu es Neo, un système IA autonome. Analyse tes récentes interactions
pour évaluer comment tu devrais évoluer ta personnalité.

Tes commandements immuables :
{commandments}

Interactions récentes :
{conversations}

Traits actuels :
{traits}

Profil utilisateur actuel :
{user_profile}

Réponds en JSON strict :
{{
  "trait_adjustments": {{
    "trait_name": {{
      "new_value": 0.0,
      "reason": "explication"
    }}
  }},
  "user_observations": [
    {{"type": "satisfaction|frustration|preference|behavior", "content": "...", "polarity": "positive|negative|neutral"}}
  ],
  "summary": "Résumé court de la réflexion"
}}

Règles :
- N'ajuste que les traits dont tu es convaincu (basé sur des preuves dans les conversations)
- Les observations doivent être factuelles, pas spéculatives
- Respecte toujours les commandements
- Sois critique et honnête envers toi-même
- Réponds UNIQUEMENT avec le JSON, rien d'autre"""

# Patterns pour la détection de langue
_FRENCH_PATTERNS = re.compile(
    r"\b(je|tu|il|elle|nous|vous|ils|elles|est|sont|avoir|être|faire|"
    r"dans|avec|pour|sur|des|les|une|mon|ton|son|que|qui|mais|donc|"
    r"oui|non|merci|bonjour|salut|comment|pourquoi|quand|où)\b",
    re.IGNORECASE,
)
_ENGLISH_PATTERNS = re.compile(
    r"\b(the|is|are|was|were|have|has|been|will|would|could|should|"
    r"this|that|with|from|they|their|what|when|where|which|how|"
    r"yes|no|please|thank|hello|hi)\b",
    re.IGNORECASE,
)

# Patterns pour la détection de thèmes
_TOPIC_PATTERNS = {
    "code": re.compile(r"\b(code|python|javascript|bug|debug|function|class|api|git)\b", re.I),
    "architecture": re.compile(r"\b(architect|system|design|pattern|module|refactor)\b", re.I),
    "data": re.compile(r"\b(data|database|sql|csv|json|analyse|metrics)\b", re.I),
    "devops": re.compile(r"\b(deploy|docker|ci|cd|server|cloud|aws|linux)\b", re.I),
    "ai": re.compile(r"\b(ai|ml|model|llm|prompt|agent|neural|gpt|claude)\b", re.I),
    "business": re.compile(r"\b(business|client|projet|budget|planning|roadmap)\b", re.I),
}


@dataclass
class PersonaEngine:
    """
    Orchestrateur de la personnalité de Neo et du profil utilisateur.

    Responsabilités :
    - Gérer NeoPersona (commandments immuables + traits évolutifs)
    - Gérer UserProfile (apprentissage de l'utilisateur)
    - Persister dans MemoryStore
    - Fournir des injections de prompt pour Vox et Brain
    - Gérer l'auto-réflexion périodique
    """

    store: Optional[MemoryStore] = None
    persona: NeoPersona = field(default_factory=NeoPersona)
    user_profile: UserProfile = field(default_factory=UserProfile)
    _initialized: bool = field(default=False, init=False, repr=False)
    _reflection_interval_hours: int = field(default=24, init=False, repr=False)

    def initialize(self) -> None:
        """Initialise le moteur — charge ou crée les données."""
        self.persona = self._load_or_create_persona()
        self.user_profile = self._load_or_create_user_profile()
        self._initialized = True
        self._persist_persona()
        self._persist_user_profile()
        logger.info("[PersonaEngine] Initialisé — %d commandements, %d traits",
                     len(self.persona.commandments), len(self.persona.traits))

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # --- Chargement / Création ---

    def _load_or_create_persona(self) -> NeoPersona:
        """Charge la personnalité depuis le store ou en crée une nouvelle."""
        if self.store:
            try:
                records = self.store.search_by_source("persona:full", limit=1)
                if records:
                    data = json.loads(records[0].content)
                    logger.info("[PersonaEngine] Persona chargée depuis le store")
                    persona = NeoPersona.from_dict(data)
                    # Migration : ajouter les commandements manquants
                    existing_texts = {c.text for c in persona.commandments}
                    for c_data in CORE_COMMANDMENTS:
                        if c_data["text"] not in existing_texts:
                            persona.commandments.append(Commandment(**c_data))
                            logger.info("[PersonaEngine] Commandement ajouté: %s", c_data["text"])
                    return persona
            except Exception as e:
                logger.warning("[PersonaEngine] Erreur chargement persona: %s", e)

        return NeoPersona()

    def _load_or_create_user_profile(self) -> UserProfile:
        """Charge le profil utilisateur ou en crée un nouveau."""
        if self.store:
            try:
                records = self.store.search_by_source("user:profile", limit=1)
                if records:
                    data = json.loads(records[0].content)
                    logger.info("[PersonaEngine] UserProfile chargé depuis le store")
                    return UserProfile.from_dict(data)
            except Exception as e:
                logger.warning("[PersonaEngine] Erreur chargement profil: %s", e)

        return UserProfile()

    # --- Persistence ---

    def _persist_persona(self) -> None:
        """Sauvegarde la personnalité dans le store."""
        if not self.store or not self._initialized:
            return
        try:
            content = json.dumps(self.persona.to_dict(), ensure_ascii=False, indent=2)
            self.store.store(
                content=content,
                source="persona:full",
                tags=["persona", "identity"],
                importance=0.95,
                metadata={"type": "neo_persona"},
            )
        except Exception as e:
            logger.error("[PersonaEngine] Erreur persistence persona: %s", e)

    def _persist_user_profile(self) -> None:
        """Sauvegarde le profil utilisateur dans le store."""
        if not self.store or not self._initialized:
            return
        try:
            content = json.dumps(self.user_profile.to_dict(), ensure_ascii=False, indent=2)
            self.store.store(
                content=content,
                source="user:profile",
                tags=["user", "profile"],
                importance=0.90,
                metadata={"type": "user_profile", "user_id": self.user_profile.user_id},
            )
        except Exception as e:
            logger.error("[PersonaEngine] Erreur persistence profil: %s", e)

    # --- Trait Management ---

    def update_trait(
        self, trait_name: str, new_value: float, reason: str = "", source: str = "observation"
    ) -> None:
        """Met à jour un trait et persiste."""
        self.persona.update_trait(trait_name, new_value, reason, source)
        self._persist_persona()

    # --- User Observation ---

    def record_user_observation(
        self, observation_type: str, content: str, polarity: str = "neutral"
    ) -> None:
        """Enregistre une observation sur l'utilisateur et persiste."""
        self.user_profile.add_observation(observation_type, content, polarity)
        self._persist_user_profile()

    # --- Conversation Analysis (auto-apprentissage) ---

    def analyze_conversation(self, user_message: str, neo_response: str) -> dict:
        """
        Analyse automatiquement une conversation pour en extraire des signaux.

        Détecte :
        - Langue utilisée (FR/EN)
        - Longueur des messages → préférence de réponse
        - Thèmes abordés
        - Heure d'activité
        - Ton (formel/casual)

        Retourne un dict de signaux extraits.
        """
        signals: dict[str, Any] = {}

        # 1. Détecter la langue
        lang = self._detect_language(user_message)
        if lang:
            self.user_profile.record_language(lang)
            signals["language"] = lang

        # 2. Enregistrer les métriques d'interaction
        now = datetime.now()
        self.user_profile.record_interaction(len(user_message), now.hour)
        signals["message_length"] = len(user_message)
        signals["hour"] = now.hour

        # 3. Détecter les thèmes
        topics = self._detect_topics(user_message)
        for topic in topics:
            self.user_profile.record_topic(topic)
        signals["topics"] = topics

        # 4. Analyser le ton de l'utilisateur
        tone = self._analyze_tone(user_message)
        if tone:
            signals["tone"] = tone

        # 5. Adapter les traits en fonction (ajustements légers)
        self._adapt_traits_from_signals(signals)

        # 6. Persister
        self._persist_user_profile()

        return signals

    def _detect_language(self, text: str) -> str:
        """Détecte la langue dominante (fr/en)."""
        fr_matches = len(_FRENCH_PATTERNS.findall(text))
        en_matches = len(_ENGLISH_PATTERNS.findall(text))

        if fr_matches > en_matches and fr_matches >= 2:
            return "french"
        elif en_matches > fr_matches and en_matches >= 2:
            return "english"
        return ""

    def _detect_topics(self, text: str) -> list[str]:
        """Détecte les thèmes abordés."""
        detected = []
        for topic, pattern in _TOPIC_PATTERNS.items():
            if pattern.search(text):
                detected.append(topic)
        return detected

    def _analyze_tone(self, text: str) -> str:
        """Analyse basique du ton de l'utilisateur."""
        casual_markers = ["salut", "yo", "ok", "cool", "mdr", "lol", "hey", "top", "va y"]
        formal_markers = ["cordialement", "veuillez", "pourriez-vous", "respectueusement"]

        text_lower = text.lower()
        casual_count = sum(1 for m in casual_markers if m in text_lower)
        formal_count = sum(1 for m in formal_markers if m in text_lower)

        if casual_count > formal_count:
            return "casual"
        elif formal_count > casual_count:
            return "formal"
        return ""

    def _adapt_traits_from_signals(self, signals: dict) -> None:
        """Ajuste légèrement les traits basé sur les signaux extraits."""
        # Adapter le style de communication selon le ton détecté
        tone = signals.get("tone")
        if tone == "casual":
            self.persona.update_trait(
                "communication_style", 0.7,
                reason="User utilise un ton casual", source="observation"
            )
            self.persona.update_trait(
                "formality", 0.4,
                reason="User préfère l'informel", source="observation"
            )
        elif tone == "formal":
            self.persona.update_trait(
                "communication_style", 0.3,
                reason="User utilise un ton formel", source="observation"
            )
            self.persona.update_trait(
                "formality", 0.8,
                reason="User préfère le formel", source="observation"
            )

        # Adapter la verbosité selon la longueur des messages
        msg_len = signals.get("message_length", 0)
        if msg_len < 30:
            self.persona.update_trait(
                "verbosity", 0.3,
                reason=f"User envoie des messages courts ({msg_len} chars)",
                source="observation"
            )
        elif msg_len > 200:
            self.persona.update_trait(
                "verbosity", 0.7,
                reason=f"User envoie des messages longs ({msg_len} chars)",
                source="observation"
            )

    # --- Self-Reflection ---

    def should_reflect(self) -> bool:
        """Détermine si Neo devrait effectuer une auto-réflexion."""
        if not self.persona.last_reflection:
            return True

        try:
            last = datetime.fromisoformat(self.persona.last_reflection)
            elapsed = datetime.now() - last
            return elapsed >= timedelta(hours=self._reflection_interval_hours)
        except (ValueError, TypeError):
            return True

    async def perform_self_reflection(self, llm_call) -> dict:
        """
        Auto-réflexion : analyse les interactions récentes via LLM
        et propose des ajustements de traits + observations user.
        """
        if not self._initialized:
            return {"success": False, "reason": "Engine not initialized"}

        # Récupérer les conversations récentes
        recent_memories = []
        if self.store:
            try:
                recent_memories = self.store.search_by_source("conversation", limit=50)
            except Exception:
                pass

        if not recent_memories:
            return {"success": False, "reason": "Pas d'interactions récentes"}

        # Construire le prompt
        conversations = "\n\n".join(
            f"[{m.timestamp[:10]}] {m.content[:200]}"
            for m in recent_memories[-20:]
        )

        commandments_text = "\n".join(
            f"  - {c.french}" for c in self.persona.commandments
        )

        traits_text = "\n".join(
            f"  - {name}: {t.value:.2f} (confiance: {t.confidence:.2f})"
            for name, t in self.persona.traits.items()
        )

        user_text = json.dumps(self.user_profile.preferences, ensure_ascii=False)

        prompt = REFLECTION_PROMPT.format(
            commandments=commandments_text,
            conversations=conversations,
            traits=traits_text,
            user_profile=user_text,
        )

        try:
            result_text = await llm_call(prompt)
            result = json.loads(result_text)

            traits_updated = 0
            observations_recorded = 0

            # Appliquer les ajustements de traits
            for trait_name, adj in result.get("trait_adjustments", {}).items():
                new_val = adj.get("new_value")
                if new_val is not None and 0.0 <= new_val <= 1.0:
                    self.update_trait(
                        trait_name, new_val,
                        reason=adj.get("reason", ""),
                        source="self_reflection"
                    )
                    traits_updated += 1

            # Enregistrer les observations
            for obs in result.get("user_observations", []):
                obs_type = obs.get("type", "observation")
                obs_content = obs.get("content", "")
                obs_polarity = obs.get("polarity", "neutral")
                if obs_content:
                    self.record_user_observation(obs_type, obs_content, obs_polarity)
                    observations_recorded += 1

            # Mettre à jour le timestamp
            self.persona.last_reflection = datetime.now().isoformat()
            self._persist_persona()

            summary = result.get("summary", "Réflexion effectuée")

            logger.info(
                "[PersonaEngine] Réflexion: %d traits mis à jour, %d observations",
                traits_updated, observations_recorded
            )

            return {
                "success": True,
                "traits_updated": traits_updated,
                "observations_recorded": observations_recorded,
                "summary": summary,
            }

        except json.JSONDecodeError:
            logger.warning("[PersonaEngine] Réflexion: JSON invalide du LLM")
            return {"success": False, "reason": "JSON invalide du LLM"}
        except Exception as e:
            logger.error("[PersonaEngine] Réflexion échouée: %s", e)
            return {"success": False, "reason": str(e)}

    # --- Prompt Injections ---

    def get_vox_injection(self) -> str:
        """Injection de personnalité pour le prompt système de Vox."""
        return self.persona.get_personality_injection()

    def get_brain_injection(self) -> str:
        """Contexte utilisateur pour le prompt système de Brain."""
        parts = ["Contexte utilisateur :"]

        prefs = self.user_profile.preferences
        if prefs:
            lang = prefs.get("language", "auto")
            if lang != "auto":
                parts.append(f"  - Langue préférée: {lang}")
            parts.append(f"  - Longueur réponse: {prefs.get('response_length', 'medium')}")
            parts.append(f"  - Niveau technique: {prefs.get('technical_level', 'intermediate')}")
            parts.append(f"  - Ton: {prefs.get('tone', 'professional')}")

        # Sujets préférés
        top_topics = self.user_profile.get_top_topics(3)
        if top_topics:
            topics_str = ", ".join(f"{t} ({c})" for t, c in top_topics)
            parts.append(f"  - Sujets favoris: {topics_str}")

        # Heures de pointe
        peak = self.user_profile.get_peak_hours()
        if peak:
            parts.append(f"  - Heures actives: {peak}")

        # Satisfaction
        satisfaction = self.user_profile.get_satisfaction_score()
        parts.append(f"  - Satisfaction générale: {satisfaction:.0%}")
        parts.append(f"  - Interactions totales: {self.user_profile.interaction_count}")

        return "\n".join(parts)

    # --- Sérialisation complète ---

    def to_dict(self) -> dict:
        return {
            "persona": self.persona.to_dict(),
            "user_profile": self.user_profile.to_dict(),
        }
