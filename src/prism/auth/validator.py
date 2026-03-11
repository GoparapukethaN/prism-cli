"""API key format validation.

IMPORTANT: This module performs **offline format checks only**.  It never
makes real API calls.  Live validation (hitting the provider's ``/models``
or ``/me`` endpoint) is deferred to a future implementation.  The format
rules here catch obvious typos and copy-paste errors before any network
call is attempted.
"""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger(__name__)


# Maps provider name -> (human-readable rule, compiled regex or callable).
_FORMAT_RULES: dict[str, tuple[str, re.Pattern[str] | None]] = {
    "anthropic": (
        "must start with 'sk-ant-'",
        re.compile(r"^sk-ant-.+"),
    ),
    "openai": (
        "must start with 'sk-'",
        re.compile(r"^sk-.+"),
    ),
    "google": (
        "must be non-empty",
        None,  # any non-empty string
    ),
    "deepseek": (
        "must start with 'sk-'",
        re.compile(r"^sk-.+"),
    ),
    "groq": (
        "must start with 'gsk_'",
        re.compile(r"^gsk_.+"),
    ),
    "mistral": (
        "must be non-empty",
        None,
    ),
}


class KeyValidator:
    """Offline format validator for API keys.

    Validates keys against known format patterns *without* making any
    network requests.
    """

    @staticmethod
    def validate_key(provider: str, key: str) -> bool:
        """Check whether *key* matches the expected format for *provider*.

        Args:
            provider: Canonical provider name.
            key: The API key to validate.  **Never logged in full.**

        Returns:
            ``True`` if the key looks structurally valid, ``False`` otherwise.
        """
        if not key:
            logger.debug("key_validation_failed", provider=provider, reason="empty key")
            return False

        rule = _FORMAT_RULES.get(provider)
        if rule is None:
            # Unknown provider — accept any non-empty key.
            logger.debug(
                "key_validation_unknown_provider",
                provider=provider,
                result="accepted (no rule)",
            )
            return True

        description, pattern = rule

        if pattern is None:
            # Rule is "must be non-empty" — already passed the empty check.
            logger.debug(
                "key_validation_passed",
                provider=provider,
                key_hint=_mask_key(key),
            )
            return True

        if pattern.match(key):
            logger.debug(
                "key_validation_passed",
                provider=provider,
                key_hint=_mask_key(key),
            )
            return True

        logger.debug(
            "key_validation_failed",
            provider=provider,
            reason=description,
            key_hint=_mask_key(key),
        )
        return False

    @staticmethod
    def known_providers() -> list[str]:
        """Return sorted list of providers with known format rules."""
        return sorted(_FORMAT_RULES.keys())


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mask_key(key: str) -> str:
    """Return a masked representation of *key* showing only the last 4 chars."""
    if len(key) <= 4:
        return "****"
    return f"****{key[-4:]}"
