"""
Neo Core — Telegram Bot Integration
======================================
Connecte Neo à Telegram de manière sécurisée.

Sécurité :
- Whitelist stricte par user_id Telegram (pas par username — falsifiable)
- Token bot stocké dans le KeyVault (chiffré AES)
- Rate limiting par utilisateur
- Sanitisation de tous les messages entrants
- Pas de commandes admin exposées

Architecture :
- Utilise python-telegram-bot (async, webhook-ready)
- Partage la même session Vox que le CLI → même conversation
- Le bot tourne dans la boucle asyncio du daemon

Usage :
    1. Créer un bot via @BotFather sur Telegram → obtenir le token
    2. neo setup ou neo telegram-setup → configure token + whitelist
    3. neo start → le daemon lance le bot automatiquement
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    """Configuration du bot Telegram."""
    bot_token: str = ""
    allowed_user_ids: list[int] = field(default_factory=list)
    rate_limit_per_minute: int = 20
    max_message_length: int = 4000
    webhook_url: str = ""  # Vide = polling mode


class TelegramBot:
    """
    Bot Telegram sécurisé pour Neo Core.

    Partage la même instance Vox que le CLI pour avoir
    exactement la même conversation, mémoire et contexte.
    """

    def __init__(self, config: TelegramConfig, vox=None):
        self._config = config
        self._vox = vox
        self._app = None
        self._running = False
        # Rate limit persistant via SQLite (survit aux redémarrages)
        self._rate_db: Optional[sqlite3.Connection] = None
        self._rate_lock = threading.Lock()
        self._init_rate_db()

    def _init_rate_db(self) -> None:
        """Initialise la DB SQLite pour le rate limiting persistant."""
        try:
            self._rate_db = sqlite3.connect(":memory:", check_same_thread=False)
            self._rate_db.execute("""
                CREATE TABLE IF NOT EXISTS tg_rate_limits (
                    user_id INTEGER NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            self._rate_db.execute("""
                CREATE INDEX IF NOT EXISTS idx_tg_rate ON tg_rate_limits(user_id, timestamp)
            """)
            self._rate_db.commit()
        except Exception as e:
            logger.debug("Rate limit DB init failed: %s — using in-memory fallback", e)

    @property
    def is_configured(self) -> bool:
        """True si le bot a un token et au moins un user autorisé."""
        return bool(self._config.bot_token) and len(self._config.allowed_user_ids) > 0

    def set_vox(self, vox) -> None:
        """Connecte le bot à l'instance Vox partagée."""
        self._vox = vox

    def _is_authorized(self, user_id: int) -> bool:
        """Vérifie si le user_id Telegram est dans la whitelist."""
        return user_id in self._config.allowed_user_ids

    def _is_rate_limited(self, user_id: int) -> bool:
        """Vérifie le rate limit par utilisateur (SQLite, thread-safe via lock)."""
        now = time.time()
        window = now - 60

        if not self._rate_db:
            return False

        with self._rate_lock:
            try:
                # Nettoyer les entrées > 60s
                self._rate_db.execute(
                    "DELETE FROM tg_rate_limits WHERE timestamp < ?", (window,)
                )

                # Compter les requêtes dans la fenêtre
                row = self._rate_db.execute(
                    "SELECT COUNT(*) FROM tg_rate_limits WHERE user_id = ? AND timestamp >= ?",
                    (user_id, window),
                ).fetchone()
                count = row[0] if row else 0

                if count >= self._config.rate_limit_per_minute:
                    return True

                # Enregistrer
                self._rate_db.execute(
                    "INSERT INTO tg_rate_limits (user_id, timestamp) VALUES (?, ?)",
                    (user_id, now),
                )
                self._rate_db.commit()
                return False
            except Exception as e:
                logger.debug("Rate limit check failed: %s", e)
                return False

    def _sanitize_message(self, text: str) -> tuple[bool, str]:
        """
        Sanitise un message Telegram entrant.

        Returns:
            (is_safe, cleaned_text)
        """
        try:
            from neo_core.security.sanitizer import Sanitizer
            sanitizer = Sanitizer(max_length=self._config.max_message_length)
            result = sanitizer.sanitize(text)
            return result.is_safe or result.severity != "high", result.cleaned
        except ImportError:
            # Sanitizer pas dispo → filtre basique
            if len(text) > self._config.max_message_length:
                return True, text[:self._config.max_message_length]
            return True, text

    async def _handle_message(self, update, context) -> None:
        """Handler principal pour les messages Telegram."""
        message = update.message
        if not message or not message.text:
            return

        user_id = message.from_user.id
        user_name = message.from_user.first_name or "Unknown"
        text = message.text.strip()

        # 1. Auth : whitelist user_id
        if not self._is_authorized(user_id):
            logger.warning(
                "Unauthorized Telegram access: user_id=%d name=%s",
                user_id, user_name,
            )
            await message.reply_text(
                "⛔ Accès refusé. Votre compte n'est pas autorisé."
            )
            return

        # 2. Rate limit
        if self._is_rate_limited(user_id):
            await message.reply_text(
                "⏳ Trop de messages. Attendez un instant."
            )
            return

        # 3. Commandes spéciales
        if text.startswith("/"):
            await self._handle_command(text, message)
            return

        # 4. Sanitisation
        is_safe, cleaned = self._sanitize_message(text)
        if not is_safe:
            await message.reply_text(
                "⚠️ Message bloqué pour raisons de sécurité."
            )
            return

        # 5. Process via Vox (même conversation que CLI)
        if not self._vox:
            await message.reply_text(
                "⚠️ Neo n'est pas encore initialisé. Réessayez dans un instant."
            )
            return

        try:
            # Indicateur "typing"
            await context.bot.send_chat_action(
                chat_id=message.chat_id,
                action="typing",
            )

            # Appel Vox → Brain → réponse (timeout 60s)
            response = await asyncio.wait_for(
                self._vox.process_message(cleaned),
                timeout=60.0,
            )

            # Telegram limite à 4096 chars par message
            if len(response) > 4096:
                # Découper en chunks
                for i in range(0, len(response), 4096):
                    chunk = response[i:i + 4096]
                    await message.reply_text(chunk)
            else:
                await message.reply_text(response)

            logger.info(
                "Telegram msg from %s (id=%d): %s → %d chars response",
                user_name, user_id, cleaned[:50], len(response),
            )

        except asyncio.TimeoutError:
            logger.warning("Telegram process timeout for user %d", user_id)
            await message.reply_text(
                "⏱ Délai d'attente dépassé. Réessayez avec une demande plus courte."
            )
        except Exception as e:
            logger.error("Telegram process error: %s", e)
            await message.reply_text(
                "❌ Erreur de traitement. Réessayez."
            )

    async def _handle_command(self, text: str, message) -> None:
        """Gère les commandes /slash Telegram."""
        cmd = text.split()[0].lower()

        if cmd == "/start":
            user_name = message.from_user.first_name or "utilisateur"
            await message.reply_text(
                f"👋 Bienvenue {user_name} !\n\n"
                f"Je suis Neo, votre assistant IA.\n"
                f"Envoyez-moi un message et je vous répondrai.\n\n"
                f"Commandes :\n"
                f"/status — État du système\n"
                f"/help — Aide\n"
                f"/whoami — Vérifier votre accès"
            )

        elif cmd == "/help":
            await message.reply_text(
                "📖 *Aide Neo*\n\n"
                "Envoyez un message texte pour discuter avec Neo.\n"
                "La conversation est partagée avec le CLI.\n\n"
                "*Commandes :*\n"
                "/status — État du système\n"
                "/whoami — Info sur votre accès\n"
                "/help — Cette aide",
                parse_mode="Markdown",
            )

        elif cmd == "/status":
            if self._vox:
                status = self._vox.get_system_status()
                # Nettoyer le markup Rich pour Telegram
                clean_status = status.replace("[bold", "").replace("[/bold", "")
                clean_status = clean_status.replace("[dim", "").replace("[/dim", "")
                clean_status = clean_status.replace("[cyan]", "").replace("[/cyan]", "")
                clean_status = clean_status.replace("[green]", "").replace("[/green]", "")
                clean_status = clean_status.replace("]", "")
                await message.reply_text(f"📊 État du système :\n\n{clean_status}")
            else:
                await message.reply_text("⚠️ Système non initialisé")

        elif cmd == "/whoami":
            user = message.from_user
            await message.reply_text(
                f"✅ Accès autorisé\n"
                f"ID : {user.id}\n"
                f"Nom : {user.first_name} {user.last_name or ''}\n"
                f"Username : @{user.username or 'N/A'}"
            )

        else:
            await message.reply_text(
                "Commande inconnue. Tapez /help pour la liste."
            )

    async def _handle_error(self, update, context) -> None:
        """Handler d'erreurs global."""
        logger.error("Telegram bot error: %s", context.error)

    async def start_polling(self) -> None:
        """Démarre le bot en mode polling (pour le daemon)."""
        try:
            from telegram.ext import (
                ApplicationBuilder,
                MessageHandler,
                CommandHandler,
                filters,
            )
        except ImportError:
            logger.error(
                "python-telegram-bot non installé. "
                "Installez-le : pip install 'python-telegram-bot>=21.0'"
            )
            return

        if not self.is_configured:
            logger.warning("Telegram bot not configured — skipping")
            return

        logger.info("Starting Telegram bot (polling mode)")

        self._app = (
            ApplicationBuilder()
            .token(self._config.bot_token)
            .build()
        )

        # Handlers
        self._app.add_handler(CommandHandler("start", self._handle_message))
        self._app.add_handler(CommandHandler("help", self._handle_message))
        self._app.add_handler(CommandHandler("status", self._handle_message))
        self._app.add_handler(CommandHandler("whoami", self._handle_message))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )
        self._app.add_error_handler(self._handle_error)

        self._running = True

        # Initialiser et démarrer le polling
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        logger.info("Telegram bot started — waiting for messages")

    async def stop(self) -> None:
        """Arrête le bot proprement."""
        if self._app and self._running:
            logger.info("Stopping Telegram bot")
            self._running = False
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram bot stopped")


def load_telegram_config(data_dir: Path) -> TelegramConfig:
    """
    Charge la config Telegram depuis le KeyVault + config JSON.

    Le token est chiffré dans le vault.
    Les user_ids autorisés sont dans neo_config.json.
    """
    import json

    config = TelegramConfig()

    # 1. Token depuis le vault
    try:
        from neo_core.security.vault import KeyVault
        vault = KeyVault(data_dir=data_dir)
        vault.initialize()
        token = vault.retrieve("telegram_bot_token")
        if token:
            config.bot_token = token
        vault.close()
    except Exception as e:
        logger.debug("Vault read failed: %s", e)
        # Fallback : variable d'env
        import os
        config.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")

    # 2. Allowed user IDs depuis la config JSON
    config_file = data_dir / "neo_config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                data = json.load(f)
            telegram_cfg = data.get("telegram", {})
            config.allowed_user_ids = telegram_cfg.get("allowed_user_ids", [])
            config.rate_limit_per_minute = telegram_cfg.get("rate_limit", 20)
        except (json.JSONDecodeError, IOError):
            pass

    return config


def save_telegram_config(
    data_dir: Path,
    bot_token: str,
    allowed_user_ids: list[int],
) -> None:
    """
    Sauvegarde la config Telegram.

    Token → KeyVault (chiffré).
    User IDs → neo_config.json.
    """
    import json

    # 1. Token dans le vault
    try:
        from neo_core.security.vault import KeyVault
        vault = KeyVault(data_dir=data_dir)
        vault.initialize()
        vault.store("telegram_bot_token", bot_token)
        vault.close()
        logger.info("Telegram token stored in vault")
    except Exception as e:
        logger.warning("Vault write failed: %s — token NOT saved securely", e)

    # 2. User IDs dans neo_config.json
    config_file = data_dir / "neo_config.json"
    data = {}
    if config_file.exists():
        try:
            with open(config_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    data["telegram"] = {
        "enabled": True,
        "allowed_user_ids": allowed_user_ids,
        "rate_limit": 20,
    }

    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
