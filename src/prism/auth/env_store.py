"""Environment-variable credential store.

Reads API keys from well-known environment variables. This is read-only
by design: environment variables are set outside Prism (shell profile,
``.env`` loaders, CI secrets, etc.).
"""

from __future__ import annotations

import os

import structlog

logger = structlog.get_logger(__name__)

# Mapping: canonical provider name  ->  environment variable name.
PROVIDER_ENV_MAP: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


class EnvStore:
    """Read-only store that resolves API keys from environment variables."""

    def __init__(self, env_map: dict[str, str] | None = None) -> None:
        """Initialise the store.

        Args:
            env_map: Optional override of the provider -> env-var mapping.
                     Defaults to :data:`PROVIDER_ENV_MAP`.
        """
        self._env_map = env_map if env_map is not None else PROVIDER_ENV_MAP

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_key(self, provider: str) -> str | None:
        """Retrieve an API key for *provider* from the environment.

        Args:
            provider: Canonical provider name (e.g. ``"openai"``).

        Returns:
            The environment variable value, or ``None`` if the variable is
            unset **or** the provider is not in the mapping.
        """
        env_var = self._env_map.get(provider)
        if env_var is None:
            logger.debug("env_store_unknown_provider", provider=provider)
            return None

        value = os.environ.get(env_var)
        if value is not None:
            masked = _mask_key(value)
            logger.debug(
                "env_store_key_found",
                provider=provider,
                env_var=env_var,
                key_hint=masked,
            )
        else:
            logger.debug(
                "env_store_key_not_found",
                provider=provider,
                env_var=env_var,
            )
        return value

    def list_configured(self) -> list[str]:
        """Return a list of providers whose env var is currently set.

        Returns:
            Sorted list of provider names that have a non-empty key in the
            environment.
        """
        configured: list[str] = []
        for provider, env_var in self._env_map.items():
            value = os.environ.get(env_var)
            if value:
                configured.append(provider)
        configured.sort()
        return configured

    @property
    def env_map(self) -> dict[str, str]:
        """Return a copy of the provider -> env-var mapping."""
        return dict(self._env_map)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mask_key(key: str) -> str:
    """Return a masked representation of *key* showing only the last 4 chars."""
    if len(key) <= 4:
        return "****"
    return f"****{key[-4:]}"
