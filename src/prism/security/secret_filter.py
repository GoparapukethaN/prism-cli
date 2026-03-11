"""Filter sensitive environment variables from subprocess environments."""

from __future__ import annotations

import fnmatch
import os
import re
from typing import TYPE_CHECKING

import structlog

from prism.config.defaults import SENSITIVE_ENV_PATTERNS

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = structlog.get_logger(__name__)

# Sentinel value used to replace filtered variables when we want to keep
# the key visible but mask the value (e.g. for audit logging).
REDACTED = "***REDACTED***"


class SecretFilter:
    """Filters sensitive environment variables from dictionaries.

    By default the filter uses the patterns from
    ``prism.config.defaults.SENSITIVE_ENV_PATTERNS`` (e.g. ``*_API_KEY``,
    ``*_SECRET``, ``*_TOKEN``, ``*_PASSWORD``).  Additional patterns can be
    supplied at construction time.

    Usage::

        sf = SecretFilter()
        clean_env = sf.filter_env(os.environ)
        # clean_env has all sensitive vars removed
    """

    def __init__(
        self,
        extra_patterns: list[str] | None = None,
    ) -> None:
        """Initialise the filter.

        Args:
            extra_patterns: Additional fnmatch patterns to treat as sensitive,
                            *on top of* the defaults.
        """
        self._patterns: list[str] = list(SENSITIVE_ENV_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def patterns(self) -> list[str]:
        """Return a copy of the active pattern list."""
        return list(self._patterns)

    def is_sensitive(self, key: str) -> bool:
        """Return ``True`` if *key* matches any sensitive pattern."""
        upper_key = key.upper()
        return any(fnmatch.fnmatch(upper_key, pattern.upper()) for pattern in self._patterns)

    def filter_env(
        self,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Return a *new* dict with all sensitive variables removed.

        Args:
            env: The environment mapping to filter.  Defaults to
                 ``os.environ`` when ``None``.

        Returns:
            A new ``dict[str, str]`` with sensitive entries stripped out.
        """
        source: Mapping[str, str] = env if env is not None else os.environ
        filtered: dict[str, str] = {}
        removed_keys: list[str] = []

        for key, value in source.items():
            if self.is_sensitive(key):
                removed_keys.append(key)
            else:
                filtered[key] = value

        if removed_keys:
            logger.debug(
                "env_vars_filtered",
                count=len(removed_keys),
                keys=removed_keys,
            )

        return filtered

    def redact_env(
        self,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Return a *new* dict with sensitive values replaced by a redacted marker.

        Unlike :meth:`filter_env` the keys are preserved so the caller can
        see *which* variables were present without leaking their values.

        Args:
            env: The environment mapping to redact.  Defaults to
                 ``os.environ`` when ``None``.

        Returns:
            A new ``dict[str, str]`` with sensitive values set to
            ``***REDACTED***``.
        """
        source: Mapping[str, str] = env if env is not None else os.environ
        redacted: dict[str, str] = {}

        for key, value in source.items():
            if self.is_sensitive(key):
                redacted[key] = REDACTED
            else:
                redacted[key] = value

        return redacted

    # Patterns matching common API key formats embedded in string values.
    _API_KEY_VALUE_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}"),      # Anthropic
        re.compile(r"sk-[A-Za-z0-9]{20,}"),             # OpenAI
        re.compile(r"gsk_[A-Za-z0-9]{20,}"),            # Groq
        re.compile(r"AIza[A-Za-z0-9_-]{30,}"),          # Google
        re.compile(r"xai-[A-Za-z0-9]{20,}"),            # xAI
    ]

    def scrub_value(self, value: str) -> str:
        """Replace API key patterns embedded in a string value."""
        for pattern in self._API_KEY_VALUE_PATTERNS:
            value = pattern.sub(REDACTED, value)
        return value

    def sanitize_dict(self, data: dict[str, object]) -> dict[str, object]:
        """Recursively sanitize a dict by redacting values whose keys look sensitive.

        This operates on arbitrary dicts (not just ``str→str`` env maps).
        String values under sensitive keys are replaced with the
        ``REDACTED`` sentinel; nested dicts are processed recursively.
        Additionally, all string values are scanned for embedded API key patterns.

        Args:
            data: The dictionary to sanitize.

        Returns:
            A *new* dictionary with sensitive values redacted.
        """
        result: dict[str, object] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = self.sanitize_dict(value)  # type: ignore[arg-type]
            elif isinstance(value, str) and self.is_sensitive(key):
                result[key] = REDACTED
            elif isinstance(value, str):
                result[key] = self.scrub_value(value)
            else:
                result[key] = value
        return result
