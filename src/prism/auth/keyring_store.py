"""OS keyring-backed credential store.

Uses the ``keyring`` library to store and retrieve API keys from the
operating system's native credential manager (macOS Keychain, Windows
Credential Vault, GNOME/KDE secret services, etc.).

All keys are stored under the service name ``prism-cli``.
"""

from __future__ import annotations

import structlog

from prism.exceptions import KeyringUnavailableError

logger = structlog.get_logger(__name__)

SERVICE_NAME = "prism-cli"


class KeyringStore:
    """Store and retrieve API keys via the OS keyring."""

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return *True* if the OS keyring backend is usable.

        Returns ``False`` when:
        * the ``keyring`` package is not installed,
        * only the ``fail`` backend is active (headless Linux, Docker), or
        * any ``KeyringError`` is raised during a probe write.
        """
        try:
            import keyring
            from keyring.errors import KeyringError
        except ImportError:
            logger.debug("keyring_unavailable", reason="keyring package not installed")
            return False

        backend = keyring.get_keyring()
        # Check for the fail backend safely — try isinstance first, fall
        # back to class-name heuristic when the import is unavailable.
        try:
            from keyring.backends.fail import Keyring as FailKeyring

            _is_fail = isinstance(backend, FailKeyring)
        except (ImportError, TypeError):
            backend_cls = type(backend).__name__
            backend_mod = getattr(type(backend), "__module__", "") or ""
            _is_fail = backend_cls == "Keyring" and "fail" in backend_mod
        if _is_fail:
            logger.debug(
                "keyring_unavailable",
                reason="fail backend active (no usable keyring)",
            )
            return False

        # Probe: attempt a harmless read to verify the backend works.
        try:
            keyring.get_password(SERVICE_NAME, "__prism_probe__")
        except KeyringError as exc:
            logger.debug("keyring_unavailable", reason=str(exc))
            return False
        except Exception as exc:
            logger.debug("keyring_unavailable", reason=str(exc))
            return False

        return True

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    @staticmethod
    def get_key(provider: str) -> str | None:
        """Retrieve an API key for *provider* from the OS keyring.

        Args:
            provider: Canonical provider name (e.g. ``"anthropic"``).

        Returns:
            The stored key, or ``None`` if no key is stored.

        Raises:
            KeyringUnavailableError: If the keyring backend is broken.
        """
        try:
            import keyring
            from keyring.errors import KeyringError
        except ImportError as exc:
            raise KeyringUnavailableError("keyring package not installed") from exc

        try:
            value = keyring.get_password(SERVICE_NAME, provider)
            if value is not None:
                masked = _mask_key(value)
                logger.debug("keyring_get_key", provider=provider, key_hint=masked)
            else:
                logger.debug("keyring_key_not_found", provider=provider)
            return value
        except KeyringError as exc:
            raise KeyringUnavailableError(str(exc)) from exc

    @staticmethod
    def set_key(provider: str, key: str) -> None:
        """Store an API key for *provider* in the OS keyring.

        Args:
            provider: Canonical provider name.
            key: The API key value. **Never logged.**

        Raises:
            KeyringUnavailableError: If the keyring backend is broken.
        """
        try:
            import keyring
            from keyring.errors import KeyringError
        except ImportError as exc:
            raise KeyringUnavailableError("keyring package not installed") from exc

        try:
            keyring.set_password(SERVICE_NAME, provider, key)
            logger.info(
                "keyring_key_stored",
                provider=provider,
                key_hint=_mask_key(key),
            )
        except KeyringError as exc:
            raise KeyringUnavailableError(str(exc)) from exc

    @staticmethod
    def delete_key(provider: str) -> None:
        """Remove the API key for *provider* from the OS keyring.

        Args:
            provider: Canonical provider name.

        Raises:
            KeyringUnavailableError: If the keyring backend is broken.
        """
        try:
            import keyring
            from keyring.errors import KeyringError, PasswordDeleteError
        except ImportError as exc:
            raise KeyringUnavailableError("keyring package not installed") from exc

        try:
            keyring.delete_password(SERVICE_NAME, provider)
            logger.info("keyring_key_deleted", provider=provider)
        except PasswordDeleteError:
            # Key was not present — not an error.
            logger.debug("keyring_key_not_found_for_delete", provider=provider)
        except KeyringError as exc:
            raise KeyringUnavailableError(str(exc)) from exc


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mask_key(key: str) -> str:
    """Return a masked representation of *key* showing only the last 4 chars."""
    if len(key) <= 4:
        return "****"
    return f"****{key[-4:]}"
