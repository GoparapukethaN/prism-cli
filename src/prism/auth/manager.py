"""High-level authentication manager.

:class:`AuthManager` wraps the three credential stores (keyring, env,
encrypted) behind a single interface and resolves keys in priority order:

1. **OS keyring** (if available)
2. **Environment variables**
3. **Encrypted file store** (if ``cryptography`` is installed and a
   passphrase is provided)

It also exposes key validation and a per-provider status report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from prism.auth.encrypted_store import EncryptedStore
from prism.auth.env_store import EnvStore
from prism.auth.keyring_store import KeyringStore
from prism.auth.validator import KeyValidator
from prism.exceptions import (
    AuthError,
    KeyInvalidError,
    KeyNotFoundError,
    KeyringUnavailableError,
)

logger = structlog.get_logger(__name__)

# All providers Prism knows about (superset of any single store's mapping).
KNOWN_PROVIDERS: frozenset[str] = frozenset(
    {
        "anthropic",
        "openai",
        "google",
        "deepseek",
        "groq",
        "mistral",
    }
)


@dataclass(frozen=True)
class ProviderAuthStatus:
    """Status of a single provider's authentication credentials."""

    provider: str
    has_key: bool
    source: str | None = None  # "keyring", "env", "encrypted", or None
    is_valid_format: bool | None = None  # None if no key found
    errors: list[str] = field(default_factory=list)


class AuthManager:
    """Unified credential manager across all stores.

    Args:
        keyring_store: Keyring store instance.  Created automatically if
            ``None``.
        env_store: Env store instance.  Created automatically if ``None``.
        encrypted_store: Encrypted store instance.  Created automatically if
            ``None``.
        passphrase: Passphrase for the encrypted store.  If ``None``, the
            encrypted store will not be consulted for reads, but can still
            be used for explicit writes that pass a passphrase.
    """

    def __init__(
        self,
        keyring_store: KeyringStore | None = None,
        env_store: EnvStore | None = None,
        encrypted_store: EncryptedStore | None = None,
        passphrase: str | None = None,
    ) -> None:
        self._keyring = keyring_store or KeyringStore()
        self._env = env_store or EnvStore()
        self._encrypted = encrypted_store or EncryptedStore()
        self._passphrase = passphrase
        self._validator = KeyValidator()

    # ------------------------------------------------------------------
    # Key resolution
    # ------------------------------------------------------------------

    def get_key(self, provider: str) -> str:
        """Retrieve an API key for *provider*, searching all stores.

        Resolution order: keyring -> env -> encrypted.

        Args:
            provider: Canonical provider name.

        Returns:
            The API key string.

        Raises:
            KeyNotFoundError: If no key is found in any store.
        """
        # 1. Keyring
        key = self._try_keyring(provider)
        if key is not None:
            logger.debug("auth_key_resolved", provider=provider, source="keyring")
            return key

        # 2. Environment
        key = self._env.get_key(provider)
        if key is not None:
            logger.debug("auth_key_resolved", provider=provider, source="env")
            return key

        # 3. Encrypted store
        key = self._try_encrypted(provider)
        if key is not None:
            logger.debug("auth_key_resolved", provider=provider, source="encrypted")
            return key

        raise KeyNotFoundError(provider)

    # ------------------------------------------------------------------
    # Key storage
    # ------------------------------------------------------------------

    def store_key(
        self,
        provider: str,
        key: str,
        *,
        validate: bool = True,
        passphrase: str | None = None,
    ) -> str:
        """Store an API key, preferring the keyring, falling back to encrypted.

        Args:
            provider: Canonical provider name.
            key: The API key. **Never logged.**
            validate: Whether to run format validation before storing.
            passphrase: Passphrase for the encrypted store fallback.  Falls
                back to the instance-level passphrase if not supplied.

        Returns:
            The name of the store where the key was saved (``"keyring"`` or
            ``"encrypted"``).

        Raises:
            KeyInvalidError: If *validate* is True and the key format is bad.
            AuthError: If no writable store is available.
        """
        if validate and not self._validator.validate_key(provider, key):
            raise KeyInvalidError(provider)

        # Try keyring first.
        if KeyringStore.is_available():
            try:
                KeyringStore.set_key(provider, key)
                logger.info(
                    "auth_key_stored",
                    provider=provider,
                    store="keyring",
                    key_hint=_mask_key(key),
                )
                return "keyring"
            except KeyringUnavailableError:
                logger.debug(
                    "auth_keyring_fallback",
                    provider=provider,
                    reason="keyring write failed",
                )

        # Fall back to encrypted store.
        effective_passphrase = passphrase or self._passphrase
        if effective_passphrase is not None and EncryptedStore.is_available():
            self._encrypted.set_key(provider, key, effective_passphrase)
            logger.info(
                "auth_key_stored",
                provider=provider,
                store="encrypted",
                key_hint=_mask_key(key),
            )
            return "encrypted"

        raise AuthError(
            f"No writable credential store available for provider '{provider}'. "
            "Install 'keyring' or 'cryptography' and supply a passphrase."
        )

    # ------------------------------------------------------------------
    # Key removal
    # ------------------------------------------------------------------

    def remove_key(self, provider: str, *, passphrase: str | None = None) -> None:
        """Remove an API key from all writable stores.

        Args:
            provider: Canonical provider name.
            passphrase: Passphrase for the encrypted store.

        Environment variables are *not* modified (read-only by design).
        """
        # Keyring
        if KeyringStore.is_available():
            try:
                KeyringStore.delete_key(provider)
            except KeyringUnavailableError:
                logger.debug("auth_remove_keyring_skip", provider=provider)

        # Encrypted store
        effective_passphrase = passphrase or self._passphrase
        if effective_passphrase is not None and EncryptedStore.is_available():
            try:
                self._encrypted.delete_key(provider, effective_passphrase)
            except AuthError:
                logger.debug("auth_remove_encrypted_skip", provider=provider)

        logger.info("auth_key_removed", provider=provider)

    # ------------------------------------------------------------------
    # Listing / status
    # ------------------------------------------------------------------

    def list_configured(self) -> list[ProviderAuthStatus]:
        """Return authentication status for every known provider.

        Returns:
            A list of :class:`ProviderAuthStatus` objects, one per known
            provider, sorted by provider name.
        """
        statuses: list[ProviderAuthStatus] = []
        for provider in sorted(KNOWN_PROVIDERS):
            status = self._provider_status(provider)
            statuses.append(status)
        return statuses

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_key(self, provider: str, key: str) -> bool:
        """Run offline format validation on *key*.

        This is a convenience proxy for :meth:`KeyValidator.validate_key`.

        Args:
            provider: Canonical provider name.
            key: The API key. **Never logged in full.**

        Returns:
            ``True`` if the key looks structurally valid.
        """
        return self._validator.validate_key(provider, key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_keyring(self, provider: str) -> str | None:
        """Attempt to read from the keyring, returning ``None`` on failure."""
        if not KeyringStore.is_available():
            return None
        try:
            return KeyringStore.get_key(provider)
        except KeyringUnavailableError:
            logger.debug("auth_keyring_read_failed", provider=provider)
            return None

    def _try_encrypted(self, provider: str) -> str | None:
        """Attempt to read from the encrypted store, returning ``None``."""
        if self._passphrase is None:
            return None
        if not EncryptedStore.is_available():
            return None
        try:
            return self._encrypted.get_key(provider, self._passphrase)
        except AuthError:
            logger.debug("auth_encrypted_read_failed", provider=provider)
            return None

    def _provider_status(self, provider: str) -> ProviderAuthStatus:
        """Build a :class:`ProviderAuthStatus` for *provider*."""
        errors: list[str] = []

        # Try each source in priority order.
        key: str | None = None
        source: str | None = None

        # Keyring
        kr_key = self._try_keyring(provider)
        if kr_key is not None:
            key, source = kr_key, "keyring"

        # Env
        if key is None:
            env_key = self._env.get_key(provider)
            if env_key is not None:
                key, source = env_key, "env"

        # Encrypted
        if key is None:
            enc_key = self._try_encrypted(provider)
            if enc_key is not None:
                key, source = enc_key, "encrypted"

        # Validate
        is_valid: bool | None = None
        if key is not None:
            is_valid = self._validator.validate_key(provider, key)
            if not is_valid:
                errors.append(f"Key from {source} has invalid format")

        return ProviderAuthStatus(
            provider=provider,
            has_key=key is not None,
            source=source,
            is_valid_format=is_valid,
            errors=errors,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mask_key(key: str) -> str:
    """Return a masked representation of *key* showing only the last 4 chars."""
    if len(key) <= 4:
        return "****"
    return f"****{key[-4:]}"
