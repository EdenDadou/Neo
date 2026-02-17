"""
Tests Stage 18 — Intégration Telegram
========================================
Vérifie le bot Telegram : auth, rate limiting, sanitization,
config load/save, commandes, CLI.

~25 tests.
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neo_core.integrations.telegram import (
    TelegramBot,
    TelegramConfig,
    load_telegram_config,
    save_telegram_config,
)


# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def tg_config():
    """Config Telegram de test."""
    return TelegramConfig(
        bot_token="test-token-123",
        allowed_user_ids=[111, 222],
        rate_limit_per_minute=3,
        max_message_length=500,
    )


@pytest.fixture
def bot(tg_config):
    """Bot Telegram avec config de test."""
    vox = MagicMock()
    vox.process_message = AsyncMock(return_value="Réponse test")
    vox.get_system_status = MagicMock(return_value="Système OK")
    return TelegramBot(config=tg_config, vox=vox)


@pytest.fixture
def make_update():
    """Factory pour créer des updates Telegram simulés."""
    def _make(user_id=111, first_name="Eden", text="Bonjour", username="eden"):
        update = MagicMock()
        update.message.from_user.id = user_id
        update.message.from_user.first_name = first_name
        update.message.from_user.last_name = None
        update.message.from_user.username = username
        update.message.text = text
        update.message.chat_id = 99999
        update.message.reply_text = AsyncMock()
        return update
    return _make


@pytest.fixture
def context():
    """Context Telegram simulé."""
    ctx = MagicMock()
    ctx.bot.send_chat_action = AsyncMock()
    return ctx


# ═══════════════════════════════════════════════════════════════
#  Tests — Authorization (whitelist user_id)
# ═══════════════════════════════════════════════════════════════


class TestTelegramAuth:
    """Tests d'autorisation par whitelist user_id."""

    def test_authorized_user(self, bot):
        """User dans la whitelist → autorisé."""
        assert bot._is_authorized(111) is True
        assert bot._is_authorized(222) is True

    def test_unauthorized_user(self, bot):
        """User hors whitelist → refusé."""
        assert bot._is_authorized(999) is False
        assert bot._is_authorized(0) is False

    @pytest.mark.asyncio
    async def test_unauthorized_message_rejected(self, bot, make_update, context):
        """Message d'un user non autorisé → réponse refusée."""
        update = make_update(user_id=999)
        await bot._handle_message(update, context)
        update.message.reply_text.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "refusé" in reply.lower() or "⛔" in reply

    @pytest.mark.asyncio
    async def test_authorized_message_processed(self, bot, make_update, context):
        """Message d'un user autorisé → traité par Vox."""
        update = make_update(user_id=111, text="Salut Neo")
        await bot._handle_message(update, context)
        bot._vox.process_message.assert_called_once_with("Salut Neo")


# ═══════════════════════════════════════════════════════════════
#  Tests — Rate Limiting
# ═══════════════════════════════════════════════════════════════


class TestTelegramRateLimit:
    """Tests rate limiting par utilisateur."""

    def test_under_limit(self, bot):
        """Sous la limite → pas bloqué."""
        assert bot._is_rate_limited(111) is False
        assert bot._is_rate_limited(111) is False
        assert bot._is_rate_limited(111) is False

    def test_over_limit(self, bot):
        """Au-dessus de la limite → bloqué."""
        for _ in range(3):
            bot._is_rate_limited(111)
        assert bot._is_rate_limited(111) is True

    def test_per_user_isolation(self, bot):
        """Rate limit isolé par utilisateur."""
        for _ in range(3):
            bot._is_rate_limited(111)
        # User 111 est limité
        assert bot._is_rate_limited(111) is True
        # User 222 n'est pas affecté
        assert bot._is_rate_limited(222) is False

    @pytest.mark.asyncio
    async def test_rate_limited_message_rejected(self, bot, make_update, context):
        """Message rate-limité → réponse d'attente."""
        # Saturer le rate limit
        for _ in range(3):
            bot._is_rate_limited(111)

        update = make_update(user_id=111, text="Encore")
        await bot._handle_message(update, context)
        reply = update.message.reply_text.call_args[0][0]
        assert "⏳" in reply or "trop" in reply.lower()


# ═══════════════════════════════════════════════════════════════
#  Tests — Sanitization
# ═══════════════════════════════════════════════════════════════


class TestTelegramSanitize:
    """Tests sanitisation des messages."""

    def test_safe_message(self, bot):
        """Message normal → safe."""
        is_safe, cleaned = bot._sanitize_message("Bonjour Neo !")
        assert is_safe is True
        assert "Bonjour" in cleaned

    def test_long_message_truncated(self, bot):
        """Message trop long → tronqué."""
        long_msg = "A" * 1000
        is_safe, cleaned = bot._sanitize_message(long_msg)
        assert len(cleaned) <= bot._config.max_message_length


# ═══════════════════════════════════════════════════════════════
#  Tests — Commands (/start, /help, /status, /whoami)
# ═══════════════════════════════════════════════════════════════


class TestTelegramCommands:
    """Tests des commandes /slash."""

    @pytest.mark.asyncio
    async def test_start_command(self, bot, make_update):
        """Commande /start → message de bienvenue."""
        update = make_update(text="/start")
        await bot._handle_command("/start", update.message)
        reply = update.message.reply_text.call_args[0][0]
        assert "Bienvenue" in reply or "👋" in reply

    @pytest.mark.asyncio
    async def test_help_command(self, bot, make_update):
        """Commande /help → aide."""
        update = make_update(text="/help")
        await bot._handle_command("/help", update.message)
        reply = update.message.reply_text.call_args[0][0]
        assert "Aide" in reply or "📖" in reply

    @pytest.mark.asyncio
    async def test_status_command(self, bot, make_update):
        """Commande /status → statut système."""
        update = make_update(text="/status")
        await bot._handle_command("/status", update.message)
        update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_whoami_command(self, bot, make_update):
        """Commande /whoami → info utilisateur."""
        update = make_update(text="/whoami")
        await bot._handle_command("/whoami", update.message)
        reply = update.message.reply_text.call_args[0][0]
        assert "111" in reply or "autorisé" in reply.lower() or "✅" in reply

    @pytest.mark.asyncio
    async def test_unknown_command(self, bot, make_update):
        """Commande inconnue → message d'erreur."""
        update = make_update(text="/foo")
        await bot._handle_command("/foo", update.message)
        reply = update.message.reply_text.call_args[0][0]
        assert "inconnue" in reply.lower() or "help" in reply.lower()


# ═══════════════════════════════════════════════════════════════
#  Tests — Config load/save
# ═══════════════════════════════════════════════════════════════


class TestTelegramConfig:
    """Tests chargement/sauvegarde config Telegram."""

    def test_save_config(self, tmp_path):
        """Sauvegarde crée le fichier JSON avec les user IDs."""
        save_telegram_config(tmp_path, "token-abc", [111, 222])
        config_file = tmp_path / "neo_config.json"
        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert data["telegram"]["enabled"] is True
        assert data["telegram"]["allowed_user_ids"] == [111, 222]

    def test_load_config_with_json(self, tmp_path):
        """Charge la config depuis le JSON."""
        config_file = tmp_path / "neo_config.json"
        config_file.write_text(json.dumps({
            "telegram": {
                "allowed_user_ids": [333, 444],
                "rate_limit": 10,
            }
        }))
        config = load_telegram_config(tmp_path)
        assert config.allowed_user_ids == [333, 444]
        assert config.rate_limit_per_minute == 10

    def test_load_config_no_file(self, tmp_path):
        """Sans fichier → config vide."""
        config = load_telegram_config(tmp_path)
        assert config.bot_token == ""
        assert config.allowed_user_ids == []

    def test_save_preserves_existing_config(self, tmp_path):
        """Sauvegarde préserve les autres clés du JSON."""
        config_file = tmp_path / "neo_config.json"
        config_file.write_text(json.dumps({"other_key": "keep_me"}))
        save_telegram_config(tmp_path, "token-xyz", [555])
        data = json.loads(config_file.read_text())
        assert data["other_key"] == "keep_me"
        assert data["telegram"]["allowed_user_ids"] == [555]


# ═══════════════════════════════════════════════════════════════
#  Tests — Bot properties & lifecycle
# ═══════════════════════════════════════════════════════════════


class TestTelegramBot:
    """Tests propriétés et cycle de vie du bot."""

    def test_is_configured_true(self, bot):
        """Bot configuré → True."""
        assert bot.is_configured is True

    def test_is_configured_no_token(self):
        """Bot sans token → False."""
        config = TelegramConfig(allowed_user_ids=[111])
        bot = TelegramBot(config=config)
        assert bot.is_configured is False

    def test_is_configured_no_users(self):
        """Bot sans users → False."""
        config = TelegramConfig(bot_token="token")
        bot = TelegramBot(config=config)
        assert bot.is_configured is False

    def test_set_vox(self, bot):
        """set_vox connecte le Vox."""
        new_vox = MagicMock()
        bot.set_vox(new_vox)
        assert bot._vox is new_vox

    @pytest.mark.asyncio
    async def test_no_vox_message(self, make_update, context):
        """Message sans Vox → avertissement."""
        config = TelegramConfig(
            bot_token="token",
            allowed_user_ids=[111],
        )
        bot = TelegramBot(config=config, vox=None)
        update = make_update(user_id=111, text="Salut")
        await bot._handle_message(update, context)
        reply = update.message.reply_text.call_args[0][0]
        assert "initialisé" in reply.lower() or "⚠️" in reply


# ═══════════════════════════════════════════════════════════════
#  Tests — Infrastructure
# ═══════════════════════════════════════════════════════════════


class TestTelegramInfra:
    """Tests infrastructure."""

    def test_module_import(self):
        """Le module integrations s'importe."""
        import neo_core.integrations
        assert hasattr(neo_core.integrations, "__doc__")

    def test_telegram_module_import(self):
        """Le module telegram s'importe."""
        from neo_core.integrations import telegram
        assert hasattr(telegram, "TelegramBot")
        assert hasattr(telegram, "TelegramConfig")
        assert hasattr(telegram, "load_telegram_config")
        assert hasattr(telegram, "save_telegram_config")

    def test_version_tag(self):
        """La version est correcte."""
        from pathlib import Path
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        assert 'version = "0.9.2"' in content

    def test_telegram_optional_dep(self):
        """python-telegram-bot est dans les extras."""
        from pathlib import Path
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        assert "python-telegram-bot" in content

    def test_daemon_has_telegram_task(self):
        """Le daemon a une tâche run_telegram."""
        import inspect
        from neo_core.core import daemon
        source = inspect.getsource(daemon)
        assert "run_telegram" in source
        assert "TelegramBot" in source
