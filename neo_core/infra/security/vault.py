"""
Neo Core — KeyVault
=====================
Chiffrement symétrique des secrets (clés API, tokens OAuth, credentials).

Utilise Fernet (AES-128-CBC + HMAC-SHA256) avec une clé dérivée via
PBKDF2 à partir d'un identifiant machine + salt persistant.

Usage :
    vault = KeyVault(data_dir=Path("data"))
    vault.store("anthropic_key", "sk-ant-api03-...")
    key = vault.retrieve("anthropic_key")
    vault.delete("anthropic_key")
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import platform
import sqlite3
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


def _get_machine_id() -> str:
    """
    Retourne un identifiant stable de la machine.

    Tente /etc/machine-id (Linux), puis IOPlatformUUID (macOS),
    puis fallback sur le hostname.
    """
    # Linux
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            mid = Path(path).read_text().strip()
            if mid:
                return mid
        except (OSError, PermissionError):
            continue

    # macOS
    if platform.system() == "Darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if "IOPlatformUUID" in line:
                    return line.split('"')[-2]
        except Exception as e:
            logger.debug("macOS IOPlatformUUID read failed: %s", e)

    # Fallback : hostname + user
    return f"{platform.node()}-{os.getenv('USER', 'neo')}"


class KeyVault:
    """
    Coffre-fort chiffré pour les secrets Neo Core.

    Stocke les secrets dans un fichier SQLite avec chiffrement Fernet.
    La clé de chiffrement est dérivée de :
    - L'identifiant machine (facteur automatique)
    - Un master password optionnel (facteur utilisateur)

    Si un master password est défini (recommandé), la clé est dérivée
    de la concaténation machine_id + master_password, rendant le vault
    inaccessible même si le fichier est copié sur une autre machine
    ou si le machine-id est deviné.
    """

    def __init__(self, data_dir: Path, master_password: Optional[str] = None):
        self._data_dir = data_dir
        self._db_path = data_dir / ".vault.db"
        self._salt_path = data_dir / ".vault.salt"
        self._master_hash_path = data_dir / ".vault.master"
        self._master_password = master_password
        self._fernet: Optional[Fernet] = None
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """Initialise le vault : crée/charge le salt, dérive la clé, ouvre le DB (idempotent)."""
        if self._conn is not None:
            return  # Déjà initialisé — évite les fuites de connexion
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Salt persistant (généré une seule fois)
        if self._salt_path.exists():
            salt = self._salt_path.read_bytes()
        else:
            salt = os.urandom(16)
            self._salt_path.write_bytes(salt)
            os.chmod(self._salt_path, 0o600)

        # Construire le matériau de clé : machine_id + master_password (si fourni)
        machine_id = _get_machine_id().encode("utf-8")
        key_material = machine_id

        if self._master_password:
            key_material += b":" + self._master_password.encode("utf-8")
            # Stocker un hash du master password pour vérification future
            self._store_master_hash(self._master_password, salt)
        elif self._master_hash_path.exists():
            # Un master password était configuré mais n'est pas fourni
            logger.warning(
                "Vault has a master password configured but none was provided. "
                "Secrets may not be decryptable."
            )

        # Aussi lire le master password depuis env si non fourni explicitement
        if not self._master_password:
            env_master = os.getenv("NEO_VAULT_PASSWORD", "")
            if env_master:
                key_material = machine_id + b":" + env_master.encode("utf-8")

        # Dérivation PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480_000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(key_material))
        self._fernet = Fernet(key)

        # SQLite
        self._conn = sqlite3.connect(str(self._db_path))

        # Vérifier l'intégrité de la base de données
        try:
            result = self._conn.execute("PRAGMA integrity_check").fetchone()
            if result and result[0] != "ok":
                logger.error("KeyVault DB integrity check FAILED: %s", result[0])
                self._conn.close()
                # Tenter de recréer la DB depuis un backup ou from scratch
                backup = self._db_path.with_suffix(".db.bak")
                if backup.exists():
                    logger.info("Restoring vault from backup: %s", backup)
                    import shutil
                    self._db_path.unlink(missing_ok=True)
                    shutil.copy2(backup, self._db_path)
                    self._conn = sqlite3.connect(str(self._db_path))
                else:
                    logger.warning("No backup available — creating fresh vault DB")
                    self._db_path.unlink(missing_ok=True)
                    self._conn = sqlite3.connect(str(self._db_path))
        except sqlite3.DatabaseError as e:
            logger.error("KeyVault DB corrupted: %s — recreating", e)
            self._conn.close()
            self._db_path.unlink(missing_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS secrets (
                name TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()
        os.chmod(self._db_path, 0o600)

        # Créer un backup périodique
        backup_path = self._db_path.with_suffix(".db.bak")
        try:
            import shutil
            shutil.copy2(self._db_path, backup_path)
            os.chmod(backup_path, 0o600)
        except OSError:
            pass  # Pas grave si le backup échoue

        logger.info("KeyVault initialized at %s", self._db_path)

    def _store_master_hash(self, password: str, salt: bytes) -> None:
        """Stocke un hash du master password pour vérification future."""
        try:
            pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
            self._master_hash_path.write_bytes(pw_hash)
            os.chmod(self._master_hash_path, 0o600)
        except OSError as e:
            logger.debug("Could not store master hash: %s", e)

    def verify_master_password(self, password: str) -> bool:
        """Vérifie si le master password est correct (sans ouvrir le vault)."""
        if not self._master_hash_path.exists():
            return True  # Pas de master password configuré
        if not self._salt_path.exists():
            return False
        try:
            salt = self._salt_path.read_bytes()
            stored_hash = self._master_hash_path.read_bytes()
            computed_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
            return hmac.compare_digest(stored_hash, computed_hash)
        except Exception:
            return False

    @property
    def has_master_password(self) -> bool:
        """True si un master password est configuré."""
        return self._master_hash_path.exists()

    @property
    def is_initialized(self) -> bool:
        return self._fernet is not None and self._conn is not None

    def store(self, name: str, value: str) -> None:
        """Chiffre et stocke un secret."""
        if not self.is_initialized:
            raise RuntimeError("KeyVault not initialized")

        encrypted = self._fernet.encrypt(value.encode("utf-8"))
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO secrets (name, value, updated_at) "
                "VALUES (?, ?, CURRENT_TIMESTAMP)",
                (name, encrypted),
            )
            self._conn.commit()
        except (sqlite3.OperationalError, OSError) as e:
            err = str(e).lower()
            if "disk" in err or "no space" in err or "i/o" in err:
                logger.critical("DISK FULL — vault write failed for '%s': %s", name, e)
                raise RuntimeError(f"Disk full — cannot store secret '{name}'") from e
            raise
        logger.debug("Secret '%s' stored", name)

    def retrieve(self, name: str) -> Optional[str]:
        """
        Déchiffre et retourne un secret, ou None si inexistant.

        Si la clé machine a changé (VM clonée, changement hardware),
        le déchiffrement échoue. Le secret corrompu est supprimé et None
        est retourné pour que l'utilisateur reconfigure via `neo setup`.
        """
        if not self.is_initialized:
            raise RuntimeError("KeyVault not initialized")

        row = self._conn.execute(
            "SELECT value FROM secrets WHERE name = ?", (name,)
        ).fetchone()

        if not row:
            return None

        try:
            return self._fernet.decrypt(row[0]).decode("utf-8")
        except InvalidToken:
            logger.warning(
                "Cannot decrypt '%s' — machine_id may have changed. "
                "Secret invalidated. Reconfigure with `neo setup`.",
                name,
            )
            # Supprimer le secret indéchiffrable pour éviter les erreurs en boucle
            try:
                self._conn.execute("DELETE FROM secrets WHERE name = ?", (name,))
                self._conn.commit()
            except Exception:
                pass
            return None

    def delete(self, name: str) -> bool:
        """Supprime un secret. Retourne True si le secret existait."""
        if not self.is_initialized:
            raise RuntimeError("KeyVault not initialized")

        cursor = self._conn.execute("DELETE FROM secrets WHERE name = ?", (name,))
        self._conn.commit()
        return cursor.rowcount > 0

    def list_secrets(self) -> list[str]:
        """Retourne la liste des noms de secrets (pas les valeurs)."""
        if not self.is_initialized:
            raise RuntimeError("KeyVault not initialized")

        rows = self._conn.execute("SELECT name FROM secrets ORDER BY name").fetchall()
        return [row[0] for row in rows]

    def has(self, name: str) -> bool:
        """Vérifie si un secret existe."""
        if not self.is_initialized:
            raise RuntimeError("KeyVault not initialized")

        row = self._conn.execute(
            "SELECT 1 FROM secrets WHERE name = ?", (name,)
        ).fetchone()
        return row is not None

    def close(self) -> None:
        """Ferme la connexion."""
        if self._conn:
            self._conn.close()
            self._conn = None
