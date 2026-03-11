"""Proxy and network configuration — HTTP/HTTPS/SOCKS5 proxy support.

Manages proxy settings from environment variables, per-provider config,
and global defaults.  Produces ready-to-use keyword arguments for both
``httpx`` and ``LiteLLM`` clients.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ProxyConfig:
    """Proxy configuration for a provider or globally.

    Attributes:
        http_proxy:      HTTP proxy URL (e.g. ``http://proxy:8080``).
        https_proxy:     HTTPS proxy URL.
        socks5_proxy:    SOCKS5 proxy URL (e.g. ``socks5://proxy:1080``).
        no_proxy:        Hostnames / patterns excluded from proxying.
        ssl_verify:      Whether to verify SSL certificates.
        ssl_ca_bundle:   Path to a custom CA certificate bundle.
        connect_timeout: TCP connect timeout in seconds.
        read_timeout:    Read timeout in seconds.
    """

    http_proxy: str | None = None
    https_proxy: str | None = None
    socks5_proxy: str | None = None
    no_proxy: list[str] = field(default_factory=list)
    ssl_verify: bool = True
    ssl_ca_bundle: str | None = None
    connect_timeout: float = 10.0
    read_timeout: float = 60.0


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ProxyManager:
    """Manages proxy configuration from env vars and per-provider overrides.

    On construction the manager reads ``HTTP_PROXY``, ``HTTPS_PROXY``,
    ``SOCKS5_PROXY``, ``NO_PROXY``, ``PRISM_SSL_VERIFY``, and
    ``PRISM_CA_BUNDLE`` from the environment and stores them as the
    global (fallback) configuration.

    Per-provider configs can be set via :meth:`set_provider_config` and
    are looked up by :meth:`get_config`.

    Usage::

        pm = ProxyManager()
        kwargs = pm.get_httpx_kwargs("openai")
        async with httpx.AsyncClient(**kwargs) as client:
            ...
    """

    def __init__(
        self,
        provider_configs: dict[str, ProxyConfig] | None = None,
    ) -> None:
        """Initialise proxy manager.

        Args:
            provider_configs: Optional mapping of provider name to
                :class:`ProxyConfig`.  When a provider is not in this
                mapping the global config derived from env vars is used.
        """
        self._global: ProxyConfig = self._from_env()
        self._provider_configs: dict[str, ProxyConfig] = dict(
            provider_configs or {}
        )
        if self.has_proxy:
            logger.info(
                "proxy_configured",
                http=bool(self._global.http_proxy),
                https=bool(self._global.https_proxy),
                socks5=bool(self._global.socks5_proxy),
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _from_env() -> ProxyConfig:
        """Build a :class:`ProxyConfig` from environment variables.

        Reads both upper- and lower-case variants of each variable
        (upper-case takes precedence).
        """
        no_proxy_raw = os.environ.get(
            "NO_PROXY", os.environ.get("no_proxy", "")
        )
        no_proxy = [h.strip() for h in no_proxy_raw.split(",") if h.strip()]

        ssl_verify_raw = os.environ.get("PRISM_SSL_VERIFY", "true")
        ssl_verify = ssl_verify_raw.lower() not in ("false", "0", "no")

        connect_timeout_raw = os.environ.get("PRISM_CONNECT_TIMEOUT")
        read_timeout_raw = os.environ.get("PRISM_READ_TIMEOUT")

        connect_timeout = 10.0
        if connect_timeout_raw is not None:
            try:
                connect_timeout = float(connect_timeout_raw)
                if connect_timeout <= 0:
                    connect_timeout = 10.0
            except ValueError:
                connect_timeout = 10.0

        read_timeout = 60.0
        if read_timeout_raw is not None:
            try:
                read_timeout = float(read_timeout_raw)
                if read_timeout <= 0:
                    read_timeout = 60.0
            except ValueError:
                read_timeout = 60.0

        return ProxyConfig(
            http_proxy=os.environ.get(
                "HTTP_PROXY", os.environ.get("http_proxy")
            ),
            https_proxy=os.environ.get(
                "HTTPS_PROXY", os.environ.get("https_proxy")
            ),
            socks5_proxy=(
                os.environ.get("SOCKS5_PROXY")
                or os.environ.get("ALL_PROXY")
            ),
            no_proxy=no_proxy,
            ssl_verify=ssl_verify,
            ssl_ca_bundle=os.environ.get("PRISM_CA_BUNDLE"),
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_config(self, provider: str | None = None) -> ProxyConfig:
        """Return proxy config for *provider*, falling back to global.

        Args:
            provider: Provider name (e.g. ``"openai"``).  ``None``
                always returns the global config.

        Returns:
            The resolved :class:`ProxyConfig`.
        """
        if provider and provider in self._provider_configs:
            return self._provider_configs[provider]
        return self._global

    def set_provider_config(self, provider: str, config: ProxyConfig) -> None:
        """Set a per-provider proxy configuration.

        Args:
            provider: Provider name.
            config: The :class:`ProxyConfig` to use for this provider.

        Raises:
            ValueError: If *provider* is empty.
        """
        if not provider or not provider.strip():
            raise ValueError("Provider name must not be empty")
        self._provider_configs[provider.strip()] = config
        logger.debug("provider_proxy_set", provider=provider.strip())

    def remove_provider_config(self, provider: str) -> bool:
        """Remove a per-provider proxy configuration.

        Args:
            provider: Provider name.

        Returns:
            ``True`` if a config was removed, ``False`` if none existed.
        """
        removed = self._provider_configs.pop(provider, None)
        return removed is not None

    # ------------------------------------------------------------------
    # Client builder helpers
    # ------------------------------------------------------------------

    def get_httpx_kwargs(self, provider: str | None = None) -> dict[str, Any]:
        """Build keyword arguments for ``httpx.AsyncClient``.

        Includes proxy, timeout, and SSL verification settings.

        Args:
            provider: Optional provider name for per-provider config.

        Returns:
            A dictionary suitable for ``httpx.AsyncClient(**kwargs)``.
        """
        config = self.get_config(provider)
        kwargs: dict[str, Any] = {}

        # --- Proxy ---
        proxy_url = self._resolve_proxy_url(config)
        if proxy_url:
            kwargs["proxy"] = proxy_url

        # --- Timeouts ---
        kwargs["timeout"] = {
            "connect": config.connect_timeout,
            "read": config.read_timeout,
            "write": config.read_timeout,
            "pool": config.connect_timeout,
        }

        # --- SSL ---
        if not config.ssl_verify:
            kwargs["verify"] = False
        elif config.ssl_ca_bundle:
            kwargs["verify"] = config.ssl_ca_bundle

        return kwargs

    def get_litellm_kwargs(self, provider: str | None = None) -> dict[str, Any]:
        """Build keyword arguments suitable for ``litellm.completion()``.

        Args:
            provider: Optional provider name for per-provider config.

        Returns:
            A dictionary of extra kwargs for LiteLLM calls.
        """
        config = self.get_config(provider)
        kwargs: dict[str, Any] = {}

        # LiteLLM honours the timeout kwarg
        kwargs["timeout"] = config.read_timeout

        if not config.ssl_verify:
            kwargs["ssl_verify"] = False

        # If a proxy is configured, pass it as api_base override hint
        proxy_url = self._resolve_proxy_url(config)
        if proxy_url:
            kwargs["proxy"] = proxy_url

        return kwargs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_proxy(self) -> bool:
        """Return ``True`` if any proxy is configured globally."""
        g = self._global
        return bool(g.http_proxy or g.https_proxy or g.socks5_proxy)

    @property
    def global_config(self) -> ProxyConfig:
        """Return the global (environment-derived) proxy config."""
        return self._global

    @property
    def provider_names(self) -> list[str]:
        """Return sorted list of providers with custom proxy configs."""
        return sorted(self._provider_configs.keys())

    # ------------------------------------------------------------------
    # No-proxy check
    # ------------------------------------------------------------------

    def is_no_proxy(self, hostname: str) -> bool:
        """Check if *hostname* should bypass the proxy.

        Matches against the ``no_proxy`` list from the global config.
        Supports wildcard (``*``), suffix matching (``.example.com``),
        and exact matching.

        Args:
            hostname: The hostname to check.

        Returns:
            ``True`` if the hostname is in the no-proxy list.
        """
        if not hostname:
            return False

        config = self._global
        hostname_lower = hostname.lower().strip()

        for pattern in config.no_proxy:
            pattern_lower = pattern.lower().strip()
            if not pattern_lower:
                continue

            # Wildcard matches everything
            if pattern_lower == "*":
                return True

            # Exact match
            if hostname_lower == pattern_lower:
                return True

            # Suffix match: ".example.com" matches "foo.example.com"
            if pattern_lower.startswith("."):
                if hostname_lower.endswith(pattern_lower):
                    return True
                # Also match the bare domain: ".example.com" matches "example.com"
                if hostname_lower == pattern_lower.lstrip("."):
                    return True
            elif hostname_lower.endswith("." + pattern_lower):
                # Implicit suffix match: "example.com" matches
                # "foo.example.com"
                return True

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_proxy_url(config: ProxyConfig) -> str | None:
        """Resolve the single proxy URL to use from a config.

        SOCKS5 takes precedence over HTTPS, which takes precedence
        over HTTP.

        Args:
            config: The proxy config to resolve from.

        Returns:
            The proxy URL string, or ``None`` if no proxy is configured.
        """
        if config.socks5_proxy:
            return config.socks5_proxy
        if config.https_proxy:
            return config.https_proxy
        if config.http_proxy:
            return config.http_proxy
        return None

    def reload_from_env(self) -> None:
        """Re-read proxy configuration from environment variables.

        Useful when environment variables have been modified at runtime
        (e.g. in tests or after loading a ``.env`` file).
        """
        self._global = self._from_env()
        logger.debug("proxy_config_reloaded")
