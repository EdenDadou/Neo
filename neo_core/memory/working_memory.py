"""
Working Memory — Mémoire de Travail en Temps Réel
===================================================
Scratchpad contextuel maintenu en continu pendant les conversations.

Contrairement à la mémoire long-terme (MemoryStore/FAISS), la Working Memory
est un résumé structuré et compact de la conversation EN COURS, injecté
directement dans le system prompt de Brain.

Objectifs :
- Éviter la perte de contexte sur les réponses courtes ("oui", "ok")
- Réduire les tokens nécessaires (résumé vs historique brut)
- Maintenir la continuité thématique entre les échanges
- Tracer les décisions prises et les actions en attente

Persistence : fichier JSON dans data/working_memory.json
(léger, rechargeable au redémarrage, effaçable sans risque).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Limites pour garder la working memory compacte
MAX_TOPIC_HISTORY = 5          # Derniers sujets abordés
MAX_PENDING_ACTIONS = 10       # Actions en attente
MAX_DECISIONS = 10             # Décisions récentes
MAX_KEY_FACTS = 15             # Faits clés extraits
MAX_CONTEXT_CHARS = 2000       # Taille max de l'injection dans le prompt


@dataclass
class WorkingMemoryState:
    """
    État structuré de la mémoire de travail.

    Champs :
    - current_topic : sujet principal de la conversation en cours
    - topic_history : derniers sujets abordés (pour détecter les changements)
    - pending_actions : actions que l'utilisateur attend de Neo
    - recent_decisions : décisions prises récemment
    - key_facts : faits importants extraits de la conversation
    - user_mood : ton/humeur détecté de l'utilisateur
    - last_updated : timestamp de la dernière mise à jour
    """
    current_topic: str = ""
    topic_history: list[str] = field(default_factory=list)
    pending_actions: list[str] = field(default_factory=list)
    recent_decisions: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    user_mood: str = "neutral"
    last_updated: str = ""
    turn_count: int = 0

    def to_dict(self) -> dict:
        return {
            "current_topic": self.current_topic,
            "topic_history": self.topic_history,
            "pending_actions": self.pending_actions,
            "recent_decisions": self.recent_decisions,
            "key_facts": self.key_facts,
            "user_mood": self.user_mood,
            "last_updated": self.last_updated,
            "turn_count": self.turn_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> WorkingMemoryState:
        return cls(
            current_topic=data.get("current_topic", ""),
            topic_history=data.get("topic_history", []),
            pending_actions=data.get("pending_actions", []),
            recent_decisions=data.get("recent_decisions", []),
            key_facts=data.get("key_facts", []),
            user_mood=data.get("user_mood", "neutral"),
            last_updated=data.get("last_updated", ""),
            turn_count=data.get("turn_count", 0),
        )

    def to_context_string(self) -> str:
        """
        Génère un résumé compact pour injection dans le system prompt de Brain.

        Format concis pour minimiser les tokens tout en maximisant le contexte.
        """
        parts = []

        if self.current_topic:
            parts.append(f"Sujet en cours : {self.current_topic}")

        if self.pending_actions:
            actions = " ; ".join(self.pending_actions[:5])
            parts.append(f"Actions en attente : {actions}")

        if self.recent_decisions:
            decisions = " ; ".join(self.recent_decisions[:3])
            parts.append(f"Décisions récentes : {decisions}")

        if self.key_facts:
            facts = " ; ".join(self.key_facts[:5])
            parts.append(f"Faits clés : {facts}")

        if self.user_mood and self.user_mood != "neutral":
            parts.append(f"Humeur utilisateur : {self.user_mood}")

        result = "\n".join(parts)
        # Tronquer si trop long
        if len(result) > MAX_CONTEXT_CHARS:
            result = result[:MAX_CONTEXT_CHARS - 3] + "..."
        return result


class WorkingMemory:
    """
    Mémoire de travail persistante — scratchpad contextuel.

    Maintenue en temps réel à chaque échange conversationnel.
    Injectée dans le system prompt de Brain pour une compréhension
    contextuelle maximale.

    Thread-safe : toutes les mutations de state sont protégées par un lock.
    """

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._state = WorkingMemoryState()
        self._lock = threading.Lock()
        self._file_path = data_dir / "working_memory.json"

    def initialize(self) -> None:
        """Charge l'état depuis le disque (s'il existe)."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    @property
    def state(self) -> WorkingMemoryState:
        """Accès en lecture à l'état (snapshot thread-safe)."""
        with self._lock:
            return WorkingMemoryState(**self._state.to_dict())

    def get_context_injection(self) -> str:
        """
        Retourne le texte à injecter dans le system prompt de Brain.

        Appelé par Brain.get_memory_context() à chaque requête.
        Retourne "" si la working memory est vide.
        """
        with self._lock:
            text = self._state.to_context_string()
        return text

    def update(
        self,
        user_message: str,
        ai_response: str,
        *,
        topic: str = "",
        pending_action: str = "",
        decision: str = "",
        completed_action: str = "",
        key_fact: str = "",
        user_mood: str = "",
    ) -> None:
        """
        Met à jour la working memory après un échange.

        Peut être appelé avec des champs explicites (topic, pending_action, etc.)
        ou simplement avec user_message + ai_response pour l'analyse heuristique.
        """
        with self._lock:
            self._state.turn_count += 1
            self._state.last_updated = datetime.now().isoformat()

            # ── Analyse heuristique du message ──
            if not topic:
                topic = self._extract_topic(user_message)

            if topic and topic != self._state.current_topic:
                # Nouveau sujet → archiver l'ancien
                if self._state.current_topic:
                    self._state.topic_history.append(self._state.current_topic)
                    self._state.topic_history = self._state.topic_history[-MAX_TOPIC_HISTORY:]
                self._state.current_topic = topic

            # Ajouter une action en attente
            if pending_action:
                self._state.pending_actions.append(pending_action)
                self._state.pending_actions = self._state.pending_actions[-MAX_PENDING_ACTIONS:]

            # Auto-détecter les propositions dans la réponse AI
            if not pending_action:
                detected = self._detect_pending_action(ai_response)
                if detected:
                    self._state.pending_actions.append(detected)
                    self._state.pending_actions = self._state.pending_actions[-MAX_PENDING_ACTIONS:]

            # Marquer une action comme terminée
            if completed_action:
                self._state.pending_actions = [
                    a for a in self._state.pending_actions
                    if completed_action.lower() not in a.lower()
                ]

            # Auto-détecter les confirmations qui résolvent des actions
            self._resolve_actions_from_response(user_message, ai_response)

            # Enregistrer une décision
            if decision:
                self._state.recent_decisions.append(decision)
                self._state.recent_decisions = self._state.recent_decisions[-MAX_DECISIONS:]

            # Extraire un fait clé
            if key_fact:
                self._state.key_facts.append(key_fact)
                self._state.key_facts = self._state.key_facts[-MAX_KEY_FACTS:]

            # Humeur
            if user_mood:
                self._state.user_mood = user_mood
            elif not user_mood:
                detected_mood = self._detect_mood(user_message)
                if detected_mood:
                    self._state.user_mood = detected_mood

            # Persistance
            self._save()

    def clear(self) -> None:
        """Réinitialise la working memory."""
        with self._lock:
            self._state = WorkingMemoryState()
            self._save()

    def clear_pending_actions(self) -> None:
        """Vide les actions en attente (utile en fin de session)."""
        with self._lock:
            self._state.pending_actions = []
            self._save()

    def add_key_fact(self, fact: str) -> None:
        """Ajoute un fait clé (externe, thread-safe)."""
        with self._lock:
            self._state.key_facts.append(fact)
            self._state.key_facts = self._state.key_facts[-MAX_KEY_FACTS:]
            self._save()

    def set_topic(self, topic: str) -> None:
        """Force le sujet courant (override externe)."""
        with self._lock:
            if self._state.current_topic and self._state.current_topic != topic:
                self._state.topic_history.append(self._state.current_topic)
                self._state.topic_history = self._state.topic_history[-MAX_TOPIC_HISTORY:]
            self._state.current_topic = topic
            self._save()

    # ─── Heuristiques d'extraction ───────────────────────

    @staticmethod
    def _extract_topic(message: str) -> str:
        """
        Extrait le sujet principal d'un message utilisateur.

        Heuristique simple basée sur la longueur et les mots-clés.
        Pour les messages courts (<5 mots), on ne change pas de sujet.
        """
        msg = message.strip()
        words = msg.split()

        # Messages trop courts → pas de changement de sujet
        if len(words) < 5:
            return ""

        # Supprimer les mots vides pour extraire l'essence
        stop_words = {
            "le", "la", "les", "de", "du", "des", "un", "une", "et", "est",
            "en", "pour", "dans", "sur", "avec", "que", "qui", "je", "tu",
            "il", "on", "ce", "ça", "ne", "pas", "se", "me", "te", "nous",
            "vous", "ils", "au", "aux", "mon", "ton", "son", "ma", "ta",
            "sa", "mes", "tes", "ses", "être", "avoir", "faire", "dire",
            "the", "a", "an", "is", "are", "was", "were", "to", "of",
            "and", "in", "for", "on", "with", "at", "by", "from", "i",
            "you", "he", "she", "it", "we", "they", "can", "do", "does",
        }

        meaningful = [w for w in words if w.lower() not in stop_words]
        if not meaningful:
            return ""

        # Prendre les 6 premiers mots significatifs comme sujet
        topic = " ".join(meaningful[:6])
        return topic[:100]  # Cap à 100 chars

    @staticmethod
    def _detect_pending_action(ai_response: str) -> str:
        """
        Détecte si la réponse AI propose une action à l'utilisateur.

        Cherche les patterns de proposition/question.
        """
        response_lower = ai_response.lower()

        proposal_markers = [
            "tu veux que je", "veux-tu que je", "voulez-vous que je",
            "je peux", "on fait", "je lance", "je démarre", "je crée",
            "shall i", "should i", "want me to", "i can",
        ]

        for marker in proposal_markers:
            idx = response_lower.find(marker)
            if idx >= 0:
                # Extraire la phrase contenant le marker
                # Trouver la fin de la phrase (. ? \n)
                end = len(ai_response)
                for delim in [".", "?", "\n"]:
                    pos = ai_response.find(delim, idx)
                    if pos >= 0:
                        end = min(end, pos + 1)
                action = ai_response[idx:end].strip()
                return action[:150]  # Cap

        return ""

    @staticmethod
    def _detect_mood(message: str) -> str:
        """
        Détection basique de l'humeur de l'utilisateur.

        Retourne : "positive", "negative", "urgent", "neutral", ou "".
        """
        msg_lower = message.lower()

        positive_markers = [
            "super", "génial", "parfait", "excellent", "merci", "bravo",
            "bien joué", "great", "awesome", "perfect", "thanks", "nice",
            "cool", "top", "👍", "🎉",
        ]
        negative_markers = [
            "problème", "bug", "erreur", "marche pas", "fonctionne pas",
            "nul", "merde", "putain", "frustrant", "broken", "not working",
            "error", "issue", "failed", "wrong",
        ]
        urgent_markers = [
            "urgent", "vite", "rapidement", "asap", "maintenant",
            "tout de suite", "immédiat", "hurry", "quick", "now",
        ]

        if any(m in msg_lower for m in urgent_markers):
            return "urgent"
        if any(m in msg_lower for m in negative_markers):
            return "negative"
        if any(m in msg_lower for m in positive_markers):
            return "positive"

        return ""

    def _resolve_actions_from_response(self, user_message: str, ai_response: str) -> None:
        """
        Si la réponse AI confirme qu'une action a été réalisée,
        retire les actions correspondantes de la liste pending.
        """
        completion_markers = [
            "c'est fait", "terminé", "j'ai", "voilà", "done",
            "completed", "finished", "effectué", "réalisé",
        ]

        response_lower = ai_response.lower()
        if any(marker in response_lower for marker in completion_markers):
            # Si ≤3 actions, on les considère toutes résolues
            # (heuristique : si Neo dit "c'est fait", il a probablement tout fait)
            if len(self._state.pending_actions) <= 3:
                self._state.pending_actions = []

    # ─── Persistence ─────────────────────────────────────

    def _save(self) -> None:
        """Persiste l'état sur disque."""
        try:
            self._file_path.write_text(
                json.dumps(self._state.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Failed to save working memory: %s", e)

    def _load(self) -> None:
        """Charge l'état depuis le disque."""
        if not self._file_path.exists():
            return

        try:
            data = json.loads(self._file_path.read_text(encoding="utf-8"))
            self._state = WorkingMemoryState.from_dict(data)
            logger.info(
                "Working memory loaded: topic='%s', %d pending actions, %d facts",
                self._state.current_topic[:50],
                len(self._state.pending_actions),
                len(self._state.key_facts),
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load working memory (resetting): %s", e)
            self._state = WorkingMemoryState()

    def get_stats(self) -> dict:
        """Retourne des statistiques sur la working memory."""
        with self._lock:
            return {
                "current_topic": self._state.current_topic,
                "topic_history_count": len(self._state.topic_history),
                "pending_actions": len(self._state.pending_actions),
                "recent_decisions": len(self._state.recent_decisions),
                "key_facts": len(self._state.key_facts),
                "user_mood": self._state.user_mood,
                "turn_count": self._state.turn_count,
                "last_updated": self._state.last_updated,
            }
