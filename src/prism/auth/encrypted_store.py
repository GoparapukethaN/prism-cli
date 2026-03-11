"""Fernet-encrypted file-backed credential store.

API keys are kept as a JSON dictionary, encrypted with a Fernet key that
is derived from a user-supplied passphrase via PBKDF2-HMAC-SHA256 (600 000
iterations, random 16-byte salt).

File layout of ``~/.prism/credentials.enc``::

    [ 16-byte salt ][ Fernet-encrypted JSON payload ]

The ``cryptography`` package is an **optional** dependency (``pip install
prism-cli[crypto]``).  All public methods raise ``AuthError`` (not raw
``ImportError``) when the library is missing.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import structlog

from prism.exceptions import AuthError

logger = structlog.get_logger(__name__)

_SALT_LENGTH = 16
_KDF_ITERATIONS = 600_000
_DEFAULT_CREDENTIALS_PATH = Path.home() / ".prism" / "credentials.enc"


class EncryptedStore:
    """Encrypted on-disk credential store secured by a user passphrase."""

    def __init__(self, path: Path | None = None) -> None:
        """Initialise the store.

        Args:
            path: Location of the encrypted credentials file.  Defaults to
                  ``~/.prism/credentials.enc``.
        """
        self._path = path if path is not None else _DEFAULT_CREDENTIALS_PATH

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return *True* if the ``cryptography`` package is importable."""
        try:
            import cryptography  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get_key(self, provider: str, passphrase: str) -> str | None:
        """Retrieve an API key for *provider*.

        Args:
            provider: Canonical provider name.
            passphrase: User passphrase used to derive the decryption key.

        Returns:
            The stored key, or ``None`` if no key is stored for *provider*.

        Raises:
            AuthError: On decryption failure or missing ``cryptography`` dep.
        """
        data = self._load(passphrase)
        value = data.get(provider)
        if value is not None:
            logger.debug(
                "encrypted_store_key_found",
                provider=provider,
                key_hint=_mask_key(value),
            )
        else:
            logger.debug("encrypted_store_key_not_found", provider=provider)
        return value

    def set_key(self, provider: str, key: str, passphrase: str) -> None:
        """Store (or overwrite) an API key for *provider*.

        Args:
            provider: Canonical provider name.
            key: The API key. **Never logged.**
            passphrase: User passphrase used to derive the encryption key.

        Raises:
            AuthError: On encryption failure or missing ``cryptography`` dep.
        """
        data = self._load_or_empty(passphrase)
        data[provider] = key
        self._save(data, passphrase)
        logger.info(
            "encrypted_store_key_stored",
            provider=provider,
            key_hint=_mask_key(key),
        )

    def delete_key(self, provider: str, passphrase: str) -> None:
        """Remove the API key for *provider*, if present.

        Args:
            provider: Canonical provider name.
            passphrase: User passphrase used to derive the encryption key.

        Raises:
            AuthError: On encryption/decryption failure or missing dep.
        """
        data = self._load_or_empty(passphrase)
        if provider in data:
            del data[provider]
            self._save(data, passphrase)
            logger.info("encrypted_store_key_deleted", provider=provider)
        else:
            logger.debug("encrypted_store_key_not_found_for_delete", provider=provider)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_fernet_key(self, passphrase: str, salt: bytes) -> bytes:
        """Derive a Fernet key from *passphrase* and *salt*."""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        except ImportError as exc:
            raise AuthError(
                "The 'cryptography' package is required for the encrypted "
                "credential store.  Install it with:  pip install prism-cli[crypto]"
            ) from exc

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=_KDF_ITERATIONS,
        )
        raw_key = kdf.derive(passphrase.encode("utf-8"))
        return base64.urlsafe_b64encode(raw_key)

    def _load(self, passphrase: str) -> dict[str, str]:
        """Load and decrypt the credentials file.

        Returns:
            A dictionary mapping provider names to API keys.

        Raises:
            AuthError: If decryption fails or the file is corrupt.
        """
        try:
            from cryptography.fernet import Fernet, InvalidToken
        except ImportError as exc:
            raise AuthError(
                "The 'cryptography' package is required for the encrypted "
                "credential store.  Install it with:  pip install prism-cli[crypto]"
            ) from exc

        if not self._path.is_file():
            return {}

        try:
            raw = self._path.read_bytes()
        except OSError as exc:
            raise AuthError(f"Cannot read credentials file: {exc}") from exc

        if len(raw) < _SALT_LENGTH:
            raise AuthError("Credentials file is corrupt (too short).")

        salt = raw[:_SALT_LENGTH]
        encrypted = raw[_SALT_LENGTH:]

        fernet_key = self._derive_fernet_key(passphrase, salt)
        fernet = Fernet(fernet_key)

        try:
            decrypted = fernet.decrypt(encrypted)
        except InvalidToken as exc:
            raise AuthError(
                "Decryption failed — wrong passphrase or corrupt file."
            ) from exc

        try:
            data = json.loads(decrypted)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AuthError(
                "Credentials file contains invalid JSON after decryption."
            ) from exc

        if not isinstance(data, dict):
            raise AuthError("Credentials file has unexpected format.")

        return data

    def _load_or_empty(self, passphrase: str) -> dict[str, str]:
        """Load existing credentials, returning an empty dict on first use."""
        return self._load(passphrase)

    def _save(self, data: dict[str, str], passphrase: str) -> None:
        """Encrypt and persist *data* to disk.

        A fresh random salt is generated on every save so that re-encrypting
        with the same passphrase produces different ciphertext.

        Raises:
            AuthError: On encryption failure or missing ``cryptography`` dep.
        """
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise AuthError(
                "The 'cryptography' package is required for the encrypted "
                "credential store.  Install it with:  pip install prism-cli[crypto]"
            ) from exc

        salt = os.urandom(_SALT_LENGTH)
        fernet_key = self._derive_fernet_key(passphrase, salt)
        fernet = Fernet(fernet_key)

        payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
        encrypted = fernet.encrypt(payload)

        # Ensure parent directory exists.
        self._path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._path.write_bytes(salt + encrypted)
        except OSError as exc:
            raise AuthError(f"Cannot write credentials file: {exc}") from exc

        logger.debug(
            "encrypted_store_saved",
            path=str(self._path),
            providers=sorted(data.keys()),
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mask_key(key: str) -> str:
    """Return a masked representation of *key* showing only the last 4 chars."""
    if len(key) <= 4:
        return "****"
    return f"****{key[-4:]}"
