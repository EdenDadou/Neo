"""
Tests — Working Memory (Mémoire de Travail)
=============================================
~40 tests couvrant l'intégralité du module working_memory.py.
"""

import json
import threading
from pathlib import Path

import pytest

from neo_core.memory.working_memory import (
    WorkingMemory,
    WorkingMemoryState,
    MAX_TOPIC_HISTORY,
    MAX_PENDING_ACTIONS,
    MAX_DECISIONS,
    MAX_KEY_FACTS,
    MAX_CONTEXT_CHARS,
)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Répertoire temporaire pour les tests."""
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def wm(tmp_data_dir):
    """Instance WorkingMemory initialisée."""
    m = WorkingMemory(tmp_data_dir)
    m.initialize()
    return m


# ─── WorkingMemoryState ──────────────────────────────────


class TestWorkingMemoryState:
    def test_default_state(self):
        state = WorkingMemoryState()
        assert state.current_topic == ""
        assert state.pending_actions == []
        assert state.user_mood == "neutral"
        assert state.turn_count == 0

    def test_to_dict_roundtrip(self):
        state = WorkingMemoryState(
            current_topic="test topic",
            topic_history=["prev1", "prev2"],
            pending_actions=["action1"],
            recent_decisions=["decision1"],
            key_facts=["fact1"],
            user_mood="positive",
            turn_count=5,
        )
        d = state.to_dict()
        restored = WorkingMemoryState.from_dict(d)
        assert restored.current_topic == "test topic"
        assert restored.topic_history == ["prev1", "prev2"]
        assert restored.pending_actions == ["action1"]
        assert restored.recent_decisions == ["decision1"]
        assert restored.key_facts == ["fact1"]
        assert restored.user_mood == "positive"
        assert restored.turn_count == 5

    def test_from_dict_missing_fields(self):
        state = WorkingMemoryState.from_dict({})
        assert state.current_topic == ""
        assert state.pending_actions == []
        assert state.turn_count == 0

    def test_to_context_string_empty(self):
        state = WorkingMemoryState()
        assert state.to_context_string() == ""

    def test_to_context_string_with_data(self):
        state = WorkingMemoryState(
            current_topic="Déploiement v0.9.5",
            pending_actions=["Pousser sur GitHub", "Tester en prod"],
            recent_decisions=["Utiliser FAISS pour les embeddings"],
            key_facts=["VPS sur OVH"],
            user_mood="positive",
        )
        ctx = state.to_context_string()
        assert "Déploiement v0.9.5" in ctx
        assert "Pousser sur GitHub" in ctx
        assert "FAISS" in ctx
        assert "VPS sur OVH" in ctx
        assert "positive" in ctx

    def test_to_context_string_neutral_mood_hidden(self):
        state = WorkingMemoryState(
            current_topic="Test",
            user_mood="neutral",
        )
        ctx = state.to_context_string()
        assert "Humeur" not in ctx

    def test_to_context_string_truncated(self):
        state = WorkingMemoryState(
            current_topic="A" * 1000,
            key_facts=["B" * 500 for _ in range(10)],
        )
        ctx = state.to_context_string()
        assert len(ctx) <= MAX_CONTEXT_CHARS


# ─── WorkingMemory — Initialization & Persistence ────────


class TestWorkingMemoryInit:
    def test_initialize_creates_dir(self, tmp_path):
        d = tmp_path / "nonexistent" / "data"
        wm = WorkingMemory(d)
        wm.initialize()
        assert d.exists()

    def test_initialize_loads_existing(self, tmp_data_dir):
        # Save some state first
        state = WorkingMemoryState(current_topic="saved topic", turn_count=42)
        file_path = tmp_data_dir / "working_memory.json"
        file_path.write_text(json.dumps(state.to_dict()), encoding="utf-8")

        wm = WorkingMemory(tmp_data_dir)
        wm.initialize()
        assert wm.state.current_topic == "saved topic"
        assert wm.state.turn_count == 42

    def test_corrupted_file_resets(self, tmp_data_dir):
        file_path = tmp_data_dir / "working_memory.json"
        file_path.write_text("not valid json!!!", encoding="utf-8")

        wm = WorkingMemory(tmp_data_dir)
        wm.initialize()
        assert wm.state.current_topic == ""
        assert wm.state.turn_count == 0

    def test_save_and_reload(self, tmp_data_dir):
        wm1 = WorkingMemory(tmp_data_dir)
        wm1.initialize()
        wm1.update("message long pour changer le sujet", "réponse de Neo",
                    topic="nouveau sujet")

        wm2 = WorkingMemory(tmp_data_dir)
        wm2.initialize()
        assert wm2.state.current_topic == "nouveau sujet"
        assert wm2.state.turn_count == 1


# ─── WorkingMemory — Update Logic ────────────────────────


class TestWorkingMemoryUpdate:
    def test_basic_update_increments_turn(self, wm):
        wm.update("bonjour", "salut !")
        assert wm.state.turn_count == 1

    def test_explicit_topic(self, wm):
        wm.update("msg", "resp", topic="Mon nouveau projet")
        assert wm.state.current_topic == "Mon nouveau projet"

    def test_topic_change_archives_old(self, wm):
        wm.update("msg", "resp", topic="Sujet A")
        wm.update("msg", "resp", topic="Sujet B")
        assert wm.state.current_topic == "Sujet B"
        assert "Sujet A" in wm.state.topic_history

    def test_topic_history_capped(self, wm):
        for i in range(MAX_TOPIC_HISTORY + 5):
            wm.update("msg", "resp", topic=f"Topic {i}")
        assert len(wm.state.topic_history) <= MAX_TOPIC_HISTORY

    def test_pending_action_explicit(self, wm):
        wm.update("msg", "resp", pending_action="Déployer en production")
        assert "Déployer en production" in wm.state.pending_actions

    def test_pending_actions_capped(self, wm):
        for i in range(MAX_PENDING_ACTIONS + 5):
            wm.update("msg", "resp", pending_action=f"Action {i}")
        assert len(wm.state.pending_actions) <= MAX_PENDING_ACTIONS

    def test_completed_action_removes(self, wm):
        wm.update("msg", "resp", pending_action="Déployer")
        assert len(wm.state.pending_actions) == 1
        wm.update("msg", "resp", completed_action="déployer")
        assert len(wm.state.pending_actions) == 0

    def test_decision_stored(self, wm):
        wm.update("msg", "resp", decision="Utiliser PostgreSQL")
        assert "Utiliser PostgreSQL" in wm.state.recent_decisions

    def test_decisions_capped(self, wm):
        for i in range(MAX_DECISIONS + 5):
            wm.update("msg", "resp", decision=f"Decision {i}")
        assert len(wm.state.recent_decisions) <= MAX_DECISIONS

    def test_key_fact_stored(self, wm):
        wm.update("msg", "resp", key_fact="Le serveur est en France")
        assert "Le serveur est en France" in wm.state.key_facts

    def test_key_facts_capped(self, wm):
        for i in range(MAX_KEY_FACTS + 5):
            wm.update("msg", "resp", key_fact=f"Fact {i}")
        assert len(wm.state.key_facts) <= MAX_KEY_FACTS

    def test_explicit_mood(self, wm):
        wm.update("msg", "resp", user_mood="positive")
        assert wm.state.user_mood == "positive"


# ─── WorkingMemory — Heuristic Detection ────────────────


class TestHeuristicDetection:
    def test_extract_topic_short_message(self):
        assert WorkingMemory._extract_topic("oui") == ""
        assert WorkingMemory._extract_topic("ok merci") == ""

    def test_extract_topic_long_message(self):
        topic = WorkingMemory._extract_topic(
            "Je voudrais déployer l'application sur le serveur de production"
        )
        assert topic  # Non-empty
        assert "déployer" in topic.lower() or "application" in topic.lower()

    def test_detect_pending_action_proposal_fr(self):
        action = WorkingMemory._detect_pending_action(
            "Tu veux que je lance le déploiement maintenant ?"
        )
        assert action
        assert "déploiement" in action.lower()

    def test_detect_pending_action_proposal_en(self):
        action = WorkingMemory._detect_pending_action(
            "Should I run the tests before deploying?"
        )
        assert action
        assert "tests" in action.lower() or "deploying" in action.lower()

    def test_detect_pending_action_no_proposal(self):
        action = WorkingMemory._detect_pending_action(
            "Voilà le résultat de l'analyse."
        )
        assert action == ""

    def test_detect_mood_positive(self):
        assert WorkingMemory._detect_mood("Super, merci beaucoup !") == "positive"

    def test_detect_mood_negative(self):
        assert WorkingMemory._detect_mood("Il y a un problème avec le code") == "negative"

    def test_detect_mood_urgent(self):
        assert WorkingMemory._detect_mood("C'est urgent, fais-le maintenant") == "urgent"

    def test_detect_mood_neutral(self):
        assert WorkingMemory._detect_mood("Peux-tu vérifier ce fichier ?") == ""

    def test_auto_detect_pending_from_response(self, wm):
        wm.update(
            "Déploie le projet",
            "Je peux lancer le déploiement sur le VPS. Tu confirmes ?",
        )
        assert len(wm.state.pending_actions) >= 1

    def test_auto_resolve_actions(self, wm):
        wm.update("msg", "resp", pending_action="Faire le test")
        assert len(wm.state.pending_actions) == 1
        wm.update("msg", "C'est fait, les tests passent tous.")
        assert len(wm.state.pending_actions) == 0


# ─── WorkingMemory — Clear & Helpers ────────────────────


class TestWorkingMemoryClear:
    def test_clear_resets_all(self, wm):
        wm.update("msg", "resp", topic="Topic", key_fact="Fact",
                   pending_action="Action", decision="Decision")
        wm.clear()
        assert wm.state.current_topic == ""
        assert wm.state.pending_actions == []
        assert wm.state.key_facts == []
        assert wm.state.recent_decisions == []
        assert wm.state.turn_count == 0

    def test_clear_pending_actions(self, wm):
        wm.update("msg", "resp", pending_action="A1")
        wm.update("msg", "resp", pending_action="A2")
        wm.clear_pending_actions()
        assert wm.state.pending_actions == []
        # Other state preserved
        assert wm.state.turn_count == 2

    def test_add_key_fact_external(self, wm):
        wm.add_key_fact("User préfère Python")
        assert "User préfère Python" in wm.state.key_facts

    def test_set_topic_external(self, wm):
        wm.set_topic("Nouveau sujet forcé")
        assert wm.state.current_topic == "Nouveau sujet forcé"

    def test_set_topic_archives_old(self, wm):
        wm.set_topic("A")
        wm.set_topic("B")
        assert wm.state.current_topic == "B"
        assert "A" in wm.state.topic_history


# ─── WorkingMemory — Context Injection ──────────────────


class TestContextInjection:
    def test_empty_injection(self, wm):
        assert wm.get_context_injection() == ""

    def test_injection_with_data(self, wm):
        wm.update("msg", "resp", topic="Audit sécurité",
                   pending_action="Fixer les vulnérabilités",
                   key_fact="905 tests passent")
        ctx = wm.get_context_injection()
        assert "Audit sécurité" in ctx
        assert "Fixer les vulnérabilités" in ctx
        assert "905 tests" in ctx


# ─── WorkingMemory — Thread Safety ──────────────────────


class TestThreadSafety:
    def test_concurrent_updates(self, wm):
        errors = []

        def worker(thread_id):
            try:
                for i in range(20):
                    wm.update(
                        f"Thread {thread_id} msg {i}",
                        f"Response {i}",
                        key_fact=f"Fact from thread {thread_id} iteration {i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert wm.state.turn_count == 80  # 4 threads × 20
        assert len(wm.state.key_facts) <= MAX_KEY_FACTS


# ─── WorkingMemory — Stats ──────────────────────────────


class TestStats:
    def test_stats_empty(self, wm):
        stats = wm.get_stats()
        assert stats["turn_count"] == 0
        assert stats["pending_actions"] == 0
        assert stats["current_topic"] == ""

    def test_stats_after_updates(self, wm):
        wm.update("msg", "resp", topic="Test", pending_action="A1",
                   key_fact="F1", decision="D1")
        stats = wm.get_stats()
        assert stats["turn_count"] == 1
        assert stats["pending_actions"] == 1
        assert stats["key_facts"] == 1
        assert stats["recent_decisions"] == 1
        assert stats["current_topic"] == "Test"
