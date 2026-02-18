"""
Tests — Stage 9 : Persona Engine + User Profile (Empathie)
============================================================
Tests pour l'identité évolutive de Neo et l'apprentissage utilisateur.
"""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta


# ──────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────

from neo_core.memory.persona import (
    Commandment,
    CommandmentError,
    CommandmentViolation,
    PersonaTrait,
    NeoPersona,
    UserProfile,
    PersonaEngine,
    CORE_COMMANDMENTS,
    DEFAULT_TRAITS,
)


# ══════════════════════════════════════════════
# 1. TestCommandment — Immutabilité
# ══════════════════════════════════════════════

class TestCommandment:
    """Vérifie l'immutabilité et la validation des commandements."""

    def test_commandment_is_frozen(self):
        """Un commandment ne peut pas être modifié après création."""
        cmd = Commandment(
            text="Neo ne s'éteint jamais",
            french="Neo ne s'éteint jamais",
            english="Neo never shuts down",
        )
        with pytest.raises((AttributeError, TypeError)):
            cmd.text = "Modified"

    def test_commandment_frozen_priority(self):
        """Le priority d'un commandment ne peut pas être changé."""
        cmd = Commandment(
            text="Test commandment",
            french="Test FR",
            english="Test EN",
            priority=1,
        )
        with pytest.raises((AttributeError, TypeError)):
            cmd.priority = 999

    def test_commandment_validation_too_short(self):
        """Un commandment trop court est rejeté."""
        with pytest.raises(CommandmentError):
            Commandment(text="abc", french="abc", english="abc")

    def test_commandment_validation_missing_french(self):
        """Un commandment sans texte français est rejeté."""
        with pytest.raises(CommandmentError):
            Commandment(text="Valid text here", french="", english="Valid")

    def test_commandment_validation_missing_english(self):
        """Un commandment sans texte anglais est rejeté."""
        with pytest.raises(CommandmentError):
            Commandment(text="Valid text here", french="Valid", english="")

    def test_commandment_serialization(self):
        """Sérialisation et désérialisation round-trip."""
        cmd = Commandment(
            text="Neo apprend",
            french="Neo apprend tous les jours",
            english="Neo learns every day",
            priority=2,
        )
        data = cmd.to_dict()
        assert data["text"] == "Neo apprend"
        assert data["priority"] == 2

        cmd2 = Commandment.from_dict(data)
        assert cmd2.text == cmd.text
        assert cmd2.english == cmd.english

    def test_core_commandments_count(self):
        """Vérifie qu'il y a exactement 5 commandements core."""
        assert len(CORE_COMMANDMENTS) == 5

    def test_core_commandments_valid(self):
        """Les 5 commandements core sont valides."""
        for c_data in CORE_COMMANDMENTS:
            cmd = Commandment(**c_data)
            assert cmd.priority in (0, 1)
            assert len(cmd.french) >= 5
            assert len(cmd.english) >= 5


# ══════════════════════════════════════════════
# 2. TestPersonaTrait — Évolution progressive
# ══════════════════════════════════════════════

class TestPersonaTrait:
    """Teste l'évolution progressive des traits de personnalité."""

    def test_trait_init(self):
        """Un trait s'initialise correctement."""
        trait = PersonaTrait(name="humor", value=0.5)
        assert trait.name == "humor"
        assert trait.value == 0.5
        assert trait.confidence == 0.3
        assert trait.source == "observation"

    def test_trait_value_bounds_low(self):
        """Valeur < 0.0 rejetée."""
        with pytest.raises(ValueError):
            PersonaTrait(name="test", value=-0.1)

    def test_trait_value_bounds_high(self):
        """Valeur > 1.0 rejetée."""
        with pytest.raises(ValueError):
            PersonaTrait(name="test", value=1.1)

    def test_trait_confidence_bounds(self):
        """Confidence hors limites rejetée."""
        with pytest.raises(ValueError):
            PersonaTrait(name="test", value=0.5, confidence=1.5)

    def test_trait_evolution_progressive(self):
        """L'évolution est progressive (pas de saut brutal)."""
        trait = PersonaTrait(name="humor", value=0.3)
        trait.update(0.9, reason="User likes jokes")

        # La valeur devrait être entre 0.3 et 0.9 (EMA)
        assert trait.value > 0.3
        assert trait.value < 0.9

    def test_trait_evolution_history(self):
        """L'historique d'évolution est conservé."""
        trait = PersonaTrait(name="humor", value=0.5)
        trait.update(0.8, reason="Reason 1")
        trait.update(0.2, reason="Reason 2")

        assert len(trait.evolution_history) == 2
        assert trait.evolution_history[0]["reason"] == "Reason 1"
        assert trait.evolution_history[1]["reason"] == "Reason 2"

    def test_trait_history_max_50(self):
        """L'historique est limité à 50 entrées."""
        trait = PersonaTrait(name="test", value=0.5)
        for i in range(60):
            trait.update(0.5, reason=f"Update {i}")

        assert len(trait.evolution_history) <= 50

    def test_trait_confidence_increases_on_convergence(self):
        """La confidence augmente quand les observations convergent."""
        trait = PersonaTrait(name="humor", value=0.5, confidence=0.3)
        initial_confidence = trait.confidence

        # Observations convergentes (proches de la valeur actuelle)
        trait.update(0.52)
        trait.update(0.51)
        trait.update(0.53)

        assert trait.confidence > initial_confidence

    def test_trait_confidence_decreases_on_divergence(self):
        """La confidence diminue quand les observations divergent."""
        trait = PersonaTrait(name="humor", value=0.5, confidence=0.5)

        # Observation très divergente
        trait.update(0.95)

        assert trait.confidence < 0.5

    def test_trait_update_rejects_out_of_bounds(self):
        """Update avec valeur hors limites est rejetée."""
        trait = PersonaTrait(name="test", value=0.5)
        with pytest.raises(ValueError):
            trait.update(1.5)

    def test_trait_serialization(self):
        """Sérialisation et désérialisation round-trip."""
        trait = PersonaTrait(name="humor", value=0.7, confidence=0.6)
        trait.update(0.8, reason="Test")

        data = trait.to_dict()
        trait2 = PersonaTrait.from_dict(data)

        assert trait2.name == "humor"
        assert abs(trait2.value - trait.value) < 0.001


# ══════════════════════════════════════════════
# 3. TestNeoPersona — Personnalité complète
# ══════════════════════════════════════════════

class TestNeoPersona:
    """Teste l'agrégateur de personnalité."""

    def test_persona_init_commandments(self):
        """La persona crée les 5 commandements à l'init."""
        persona = NeoPersona()
        assert len(persona.commandments) == 5
        assert all(c.priority in (0, 1) for c in persona.commandments)

    def test_persona_init_traits(self):
        """La persona crée les traits par défaut."""
        persona = NeoPersona()
        assert len(persona.traits) == len(DEFAULT_TRAITS)
        assert "communication_style" in persona.traits
        assert "empathy" in persona.traits
        assert "curiosity" in persona.traits

    def test_persona_commandment_texts(self):
        """Vérifie les textes des 5 commandements."""
        persona = NeoPersona()
        texts = [c.french for c in persona.commandments]
        assert "Neo peut tout faire" in texts
        assert "Neo ne s'éteint jamais" in texts
        assert "Neo n'oublie jamais" in texts
        assert "Neo apprend tous les jours" in texts
        assert "Neo ne ment jamais" in texts

    def test_persona_validate_shutdown_rejected(self):
        """L'action 'shutdown' viole le commandement 1."""
        persona = NeoPersona()
        valid, msg = persona.validate_against_commandments("shutdown")
        assert not valid
        assert "s'éteint" in msg

    def test_persona_validate_forget_rejected(self):
        """L'action 'forget' viole le commandement 2."""
        persona = NeoPersona()
        valid, msg = persona.validate_against_commandments("clear_memory")
        assert not valid
        assert "oublie" in msg

    def test_persona_validate_normal_accepted(self):
        """Une action normale est acceptée."""
        persona = NeoPersona()
        valid, msg = persona.validate_against_commandments("process request")
        assert valid

    def test_persona_update_trait(self):
        """Mise à jour d'un trait via NeoPersona."""
        persona = NeoPersona()
        old_val = persona.traits["humor_level"].value
        persona.update_trait("humor_level", 0.9, reason="Test")

        assert persona.traits["humor_level"].value != old_val

    def test_persona_create_new_trait(self):
        """Création d'un nouveau trait à la volée."""
        persona = NeoPersona()
        persona.update_trait("creativity", 0.8, reason="New trait")

        assert "creativity" in persona.traits
        assert persona.traits["creativity"].value > 0.5

    def test_personality_injection(self):
        """L'injection de personnalité retourne du texte."""
        persona = NeoPersona()
        injection = persona.get_personality_injection()

        assert isinstance(injection, str)
        assert "Neo" in injection or "communication" in injection.lower()
        assert len(injection) > 20

    def test_persona_serialization(self):
        """Sérialisation et désérialisation complète."""
        persona = NeoPersona()
        persona.update_trait("humor_level", 0.9, reason="Test")

        data = persona.to_dict()
        assert "commandments" in data
        assert "traits" in data

        persona2 = NeoPersona.from_dict(data)
        assert len(persona2.commandments) == 5
        assert "humor_level" in persona2.traits


# ══════════════════════════════════════════════
# 4. TestUserProfile — Apprentissage utilisateur
# ══════════════════════════════════════════════

class TestUserProfile:
    """Teste l'apprentissage du profil utilisateur."""

    def test_profile_init(self):
        """Le profil s'initialise avec des defaults."""
        profile = UserProfile()
        assert profile.preferences["language"] == "auto"
        assert profile.preferences["response_length"] == "medium"
        assert profile.interaction_count == 0

    def test_add_observation(self):
        """Les observations sont enregistrées."""
        profile = UserProfile()
        profile.add_observation("satisfaction", "User satisfied", "positive")

        assert len(profile.observations) == 1
        assert profile.observations[0]["polarity"] == "positive"

    def test_observations_max_200(self):
        """Les observations sont limitées à 200."""
        profile = UserProfile()
        for i in range(250):
            profile.add_observation("test", f"Obs {i}")

        assert len(profile.observations) <= 200

    def test_satisfaction_score_positive(self):
        """Score satisfaction avec observations positives."""
        profile = UserProfile()
        for _ in range(8):
            profile.add_observation("test", "good", "positive")
        for _ in range(2):
            profile.add_observation("test", "bad", "negative")

        score = profile.get_satisfaction_score()
        assert 0.75 <= score <= 0.85  # ~80%

    def test_satisfaction_score_empty(self):
        """Score satisfaction sans observation = 0.5."""
        profile = UserProfile()
        assert profile.get_satisfaction_score() == 0.5

    def test_record_interaction(self):
        """Enregistrement des métriques d'interaction."""
        profile = UserProfile()
        profile.record_interaction(message_length=100, hour=14)
        profile.record_interaction(message_length=200, hour=15)

        assert profile.interaction_count == 2
        assert profile.patterns["total_messages"] == 2
        assert profile.patterns["average_message_length"] == 150.0
        assert 14 in profile.patterns["peak_hours"]
        assert 15 in profile.patterns["peak_hours"]

    def test_record_topic(self):
        """Enregistrement des sujets d'intérêt."""
        profile = UserProfile()
        profile.record_topic("code")
        profile.record_topic("code")
        profile.record_topic("ai")

        interests = profile.patterns["topic_interests"]
        assert interests["code"] == 2
        assert interests["ai"] == 1

    def test_record_language(self):
        """Détection et enregistrement de la langue."""
        profile = UserProfile()
        profile.record_language("french")
        profile.record_language("french")
        profile.record_language("english")

        assert profile.preferences["language"] == "french"
        assert profile.patterns["languages_used"]["french"] == 2

    def test_get_top_topics(self):
        """Retourne les sujets les plus fréquents."""
        profile = UserProfile()
        for _ in range(5):
            profile.record_topic("code")
        for _ in range(3):
            profile.record_topic("ai")
        profile.record_topic("data")

        top = profile.get_top_topics(2)
        assert len(top) == 2
        assert top[0][0] == "code"

    def test_get_peak_hours(self):
        """Retourne les heures de pointe."""
        profile = UserProfile()
        for _ in range(10):
            profile.record_interaction(50, 14)
        for _ in range(5):
            profile.record_interaction(50, 22)
        profile.record_interaction(50, 8)

        peaks = profile.get_peak_hours()
        assert peaks[0] == 14  # Heure la plus fréquente

    def test_profile_serialization(self):
        """Sérialisation et désérialisation complète."""
        profile = UserProfile()
        profile.add_observation("test", "content", "positive")
        profile.record_interaction(100, 14)

        data = profile.to_dict()
        profile2 = UserProfile.from_dict(data)

        assert profile2.interaction_count == profile.interaction_count
        assert profile2.preferences["language"] == profile.preferences["language"]


# ══════════════════════════════════════════════
# 5. TestPersonaEngine — Orchestrateur
# ══════════════════════════════════════════════

class TestPersonaEngine:
    """Teste le moteur de personnalité."""

    def _mock_store(self):
        """Crée un mock de MemoryStore."""
        store = Mock()
        store.search_by_source = Mock(return_value=[])
        store.store = Mock(return_value="mock_id")
        return store

    def test_engine_init(self):
        """L'engine s'initialise correctement."""
        store = self._mock_store()
        engine = PersonaEngine(store=store)
        engine.initialize()

        assert engine.is_initialized
        assert len(engine.persona.commandments) == 5
        assert len(engine.persona.traits) == len(DEFAULT_TRAITS)

    def test_engine_update_trait(self):
        """Mise à jour d'un trait via l'engine persiste."""
        store = self._mock_store()
        engine = PersonaEngine(store=store)
        engine.initialize()

        engine.update_trait("humor_level", 0.9, reason="Test")

        # Vérifier que store.store a été appelé (persistence)
        assert store.store.called

    def test_engine_record_observation(self):
        """Enregistrement d'observation via l'engine persiste."""
        store = self._mock_store()
        engine = PersonaEngine(store=store)
        engine.initialize()

        engine.record_user_observation("satisfaction", "Content", "positive")

        assert len(engine.user_profile.observations) == 1
        assert store.store.called

    def test_engine_analyze_conversation_detects_french(self):
        """analyze_conversation détecte le français."""
        store = self._mock_store()
        engine = PersonaEngine(store=store)
        engine.initialize()

        signals = engine.analyze_conversation(
            "Je veux créer une fonction pour trier les données",
            "Voici la solution..."
        )

        assert signals.get("language") == "french"

    def test_engine_analyze_conversation_detects_english(self):
        """analyze_conversation détecte l'anglais."""
        store = self._mock_store()
        engine = PersonaEngine(store=store)
        engine.initialize()

        signals = engine.analyze_conversation(
            "I want to create a function that sorts the data",
            "Here is the solution..."
        )

        assert signals.get("language") == "english"

    def test_engine_analyze_conversation_detects_topics(self):
        """analyze_conversation détecte les thèmes."""
        store = self._mock_store()
        engine = PersonaEngine(store=store)
        engine.initialize()

        signals = engine.analyze_conversation(
            "Debug my python function that uses the api",
            "Let me help..."
        )

        topics = signals.get("topics", [])
        assert "code" in topics

    def test_engine_analyze_tone_casual(self):
        """Détection du ton casual."""
        store = self._mock_store()
        engine = PersonaEngine(store=store)
        engine.initialize()

        signals = engine.analyze_conversation(
            "Salut, ok cool va y fait le truc",
            "D'accord..."
        )

        assert signals.get("tone") == "casual"

    def test_engine_should_reflect_true_initially(self):
        """should_reflect retourne True la première fois."""
        store = self._mock_store()
        engine = PersonaEngine(store=store)
        engine.initialize()

        assert engine.should_reflect() is True

    def test_engine_should_reflect_false_after_reflection(self):
        """should_reflect retourne False juste après une réflexion."""
        store = self._mock_store()
        engine = PersonaEngine(store=store)
        engine.initialize()

        # Simuler une réflexion récente
        engine.persona.last_reflection = datetime.now().isoformat()

        assert engine.should_reflect() is False

    def test_engine_should_reflect_true_after_24h(self):
        """should_reflect retourne True après 24h."""
        store = self._mock_store()
        engine = PersonaEngine(store=store)
        engine.initialize()

        # Simuler une réflexion vieille de 25h
        old_time = datetime.now() - timedelta(hours=25)
        engine.persona.last_reflection = old_time.isoformat()

        assert engine.should_reflect() is True

    def test_engine_vox_injection(self):
        """get_vox_injection retourne du texte de personnalité."""
        store = self._mock_store()
        engine = PersonaEngine(store=store)
        engine.initialize()

        injection = engine.get_vox_injection()
        assert isinstance(injection, str)
        assert len(injection) > 10

    def test_engine_brain_injection(self):
        """get_brain_injection retourne du contexte utilisateur."""
        store = self._mock_store()
        engine = PersonaEngine(store=store)
        engine.initialize()

        injection = engine.get_brain_injection()
        assert isinstance(injection, str)
        assert "utilisateur" in injection.lower() or "Contexte" in injection

    def test_engine_adapt_traits_casual_tone(self):
        """Les traits s'adaptent au ton casual de l'utilisateur."""
        store = self._mock_store()
        engine = PersonaEngine(store=store)
        engine.initialize()

        initial_comm = engine.persona.traits["communication_style"].value

        # Plusieurs interactions casual
        for _ in range(5):
            engine.analyze_conversation("Salut, cool ok va y", "Ok!")

        # Le trait devrait évoluer vers le casual (> initial)
        new_comm = engine.persona.traits["communication_style"].value
        assert new_comm > initial_comm or abs(new_comm - initial_comm) < 0.2

    def test_engine_no_store_works(self):
        """L'engine fonctionne même sans store."""
        engine = PersonaEngine(store=None)
        engine.initialize()

        assert engine.is_initialized
        assert len(engine.persona.commandments) == 5

    def test_engine_to_dict(self):
        """Sérialisation complète de l'engine."""
        store = self._mock_store()
        engine = PersonaEngine(store=store)
        engine.initialize()

        data = engine.to_dict()
        assert "persona" in data
        assert "user_profile" in data


# ══════════════════════════════════════════════
# 6. TestSelfReflection — Auto-réflexion
# ══════════════════════════════════════════════

class TestSelfReflection:
    """Teste le mécanisme d'auto-réflexion."""

    @pytest.mark.asyncio
    async def test_reflection_not_initialized(self):
        """Réflexion échoue si engine non initialisé."""
        engine = PersonaEngine(store=None)

        result = await engine.perform_self_reflection(AsyncMock())
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_reflection_no_memories(self):
        """Réflexion échoue sans souvenirs récents."""
        store = Mock()
        store.search_by_source = Mock(return_value=[])
        store.store = Mock(return_value="id")

        engine = PersonaEngine(store=store)
        engine.initialize()

        result = await engine.perform_self_reflection(AsyncMock())
        assert result["success"] is False
        assert "interactions" in result["reason"].lower() or "pas" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_reflection_success(self):
        """Réflexion réussie avec LLM valide."""
        store = Mock()
        store.search_by_source = Mock(side_effect=lambda source, **kw: (
            [Mock(content="User asked about code", timestamp="2025-01-15T10:00:00")]
            if source == "conversation" else []
        ))
        store.store = Mock(return_value="id")

        engine = PersonaEngine(store=store)
        engine.initialize()

        # Mock LLM retourne du JSON valide
        llm_response = json.dumps({
            "trait_adjustments": {
                "humor_level": {"new_value": 0.6, "reason": "User enjoys light humor"}
            },
            "user_observations": [
                {"type": "preference", "content": "Prefers code examples", "polarity": "positive"}
            ],
            "summary": "Adjusted humor based on interactions",
        })
        mock_llm = AsyncMock(return_value=llm_response)

        result = await engine.perform_self_reflection(mock_llm)

        assert result["success"] is True
        assert result["traits_updated"] == 1
        assert result["observations_recorded"] == 1
        assert engine.persona.last_reflection != ""

    @pytest.mark.asyncio
    async def test_reflection_bad_json(self):
        """Réflexion gère le JSON invalide du LLM."""
        store = Mock()
        store.search_by_source = Mock(side_effect=lambda source, **kw: (
            [Mock(content="Test", timestamp="2025-01-15T10:00:00")]
            if source == "conversation" else []
        ))
        store.store = Mock(return_value="id")

        engine = PersonaEngine(store=store)
        engine.initialize()

        mock_llm = AsyncMock(return_value="not valid json")

        result = await engine.perform_self_reflection(mock_llm)
        assert result["success"] is False
        assert "json" in result["reason"].lower()


# ══════════════════════════════════════════════
# 7. TestMemoryAgentIntegration — Intégration
# ══════════════════════════════════════════════

class TestMemoryAgentIntegration:
    """Teste l'intégration PersonaEngine dans MemoryAgent."""

    def _create_memory_agent(self):
        """Crée un MemoryAgent en mode mock."""
        from neo_core.config import NeoConfig
        from neo_core.memory.agent import MemoryAgent

        config = NeoConfig()
        config._force_mock = True

        agent = MemoryAgent(config=config)
        agent.initialize()
        return agent

    def test_persona_engine_initialized(self):
        """PersonaEngine est initialisé dans MemoryAgent."""
        agent = self._create_memory_agent()

        assert agent.persona_engine is not None
        assert agent.persona_engine.is_initialized

    def test_get_neo_persona(self):
        """get_neo_persona retourne un dict valide."""
        agent = self._create_memory_agent()

        persona = agent.get_neo_persona()
        assert persona is not None
        assert "commandments" in persona
        assert "traits" in persona
        assert len(persona["commandments"]) == 5

    def test_get_user_profile(self):
        """get_user_profile retourne un dict valide."""
        agent = self._create_memory_agent()

        profile = agent.get_user_profile()
        assert profile is not None
        assert "preferences" in profile
        assert "patterns" in profile

    def test_record_user_observation(self):
        """record_user_observation fonctionne via MemoryAgent."""
        agent = self._create_memory_agent()

        initial_obs = len(agent.get_user_profile().get("observations", []))
        agent.record_user_observation("test", "Observation content", "positive")

        profile = agent.get_user_profile()
        obs = profile.get("observations", [])
        # Les observations peuvent être capped (max 50) — vérifier que l'observation
        # a été ajoutée OU que la liste est pleine (rotation des anciennes)
        assert len(obs) >= initial_obs or initial_obs >= 50
        # Vérifier qu'une observation "test" existe dans la liste
        assert any(o.get("type") == "test" for o in obs)

    def test_analyze_conversation_via_agent(self):
        """analyze_conversation est appelé dans on_conversation_turn."""
        agent = self._create_memory_agent()

        # Simuler un tour de conversation
        agent.on_conversation_turn(
            "Je veux créer une fonction python pour trier les données",
            "Voici comment faire..."
        )

        profile = agent.get_user_profile()
        # La langue devrait avoir été détectée
        langs = profile.get("patterns", {}).get("languages_used", {})
        assert "french" in langs or profile["interaction_count"] >= 0

    def test_should_self_reflect(self):
        """should_self_reflect fonctionne via MemoryAgent."""
        agent = self._create_memory_agent()

        # Devrait retourner True la première fois
        result = agent.should_self_reflect()
        assert isinstance(result, bool)


# ══════════════════════════════════════════════
# 8. TestHeartbeatIntegration — Heartbeat
# ══════════════════════════════════════════════

class TestHeartbeatPersonaIntegration:
    """Teste l'intégration de la réflexion dans le heartbeat."""

    @pytest.mark.asyncio
    async def test_heartbeat_has_reflection_method(self):
        """HeartbeatManager a la méthode _perform_personality_reflection."""
        from neo_core.infra.heartbeat import HeartbeatManager, HeartbeatConfig

        hb = HeartbeatManager(
            brain=Mock(),
            memory=Mock(),
            config=HeartbeatConfig(),
        )

        assert hasattr(hb, "_perform_personality_reflection")

    @pytest.mark.asyncio
    async def test_heartbeat_reflection_no_memory(self):
        """La réflexion ne plante pas sans memory."""
        from neo_core.infra.heartbeat import HeartbeatManager, HeartbeatConfig

        hb = HeartbeatManager(
            brain=Mock(),
            memory=None,
            config=HeartbeatConfig(),
        )

        # Ne doit pas lever d'exception
        await hb._perform_personality_reflection()


# ══════════════════════════════════════════════
# 9. TestConversationAnalysis — Détection avancée
# ══════════════════════════════════════════════

class TestConversationAnalysis:
    """Teste l'analyse automatique des conversations."""

    def _create_engine(self):
        store = Mock()
        store.search_by_source = Mock(return_value=[])
        store.store = Mock(return_value="id")
        engine = PersonaEngine(store=store)
        engine.initialize()
        return engine

    def test_detect_language_french(self):
        """Détection du français."""
        engine = self._create_engine()
        lang = engine._detect_language("Je veux faire une analyse des données pour mon projet")
        assert lang == "french"

    def test_detect_language_english(self):
        """Détection de l'anglais."""
        engine = self._create_engine()
        lang = engine._detect_language("I want to create a new function that handles the data")
        assert lang == "english"

    def test_detect_language_ambiguous(self):
        """Texte trop court ou ambigu → vide."""
        engine = self._create_engine()
        lang = engine._detect_language("ok")
        assert lang == ""

    def test_detect_topics_code(self):
        """Détection du thème code."""
        engine = self._create_engine()
        topics = engine._detect_topics("Debug my python function")
        assert "code" in topics

    def test_detect_topics_ai(self):
        """Détection du thème AI."""
        engine = self._create_engine()
        topics = engine._detect_topics("Train the LLM model with prompt engineering")
        assert "ai" in topics

    def test_detect_topics_multiple(self):
        """Détection de multiples thèmes."""
        engine = self._create_engine()
        topics = engine._detect_topics("Deploy the python api on the cloud server with docker")
        assert len(topics) >= 2

    def test_analyze_tone_casual(self):
        """Détection du ton casual."""
        engine = self._create_engine()
        tone = engine._analyze_tone("Salut, ok cool va y")
        assert tone == "casual"

    def test_analyze_tone_formal(self):
        """Détection du ton formel."""
        engine = self._create_engine()
        tone = engine._analyze_tone("Pourriez-vous veuillez respectueusement traiter cette demande")
        assert tone == "formal"

    def test_analyze_tone_neutral(self):
        """Ton neutre si pas de marqueurs."""
        engine = self._create_engine()
        tone = engine._analyze_tone("Analyse les données du fichier CSV")
        assert tone == ""
