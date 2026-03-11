"""Internet connectivity detection and offline request routing."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from prism.providers.registry import ProviderRegistry

logger = structlog.get_logger(__name__)


class ConnectivityChecker:
    """Detects internet connectivity status.

    Caches the result for ``check_interval`` seconds to avoid repeated
    network probes.  Call :meth:`force_offline` or :meth:`force_online` to
    override detection.
    """

    # Well-known endpoints used for connectivity checks.
    _CHECK_URLS: list[str] = [
        "https://dns.google/resolve?name=example.com",
        "https://1.1.1.1/dns-query",
    ]

    def __init__(self, check_interval: float = 30.0) -> None:
        self._last_check: float | None = None
        self._is_online: bool = True
        self._check_interval: float = check_interval
        self._forced: bool | None = None  # None = auto, True = online, False = offline

    def is_online(self) -> bool:
        """Check if internet is available.  Caches result for check_interval."""
        # If forced, return forced value
        if self._forced is not None:
            return self._forced

        now = time.monotonic()
        if (
            self._last_check is not None
            and (now - self._last_check) < self._check_interval
        ):
            return self._is_online

        self._is_online = self._do_check()
        self._last_check = now

        logger.debug("connectivity_check", online=self._is_online)
        return self._is_online

    def _do_check(self) -> bool:
        """Actual connectivity check.  MUST be mocked in tests.

        Attempts to connect to well-known DNS endpoints.  Returns True
        if any endpoint is reachable.
        """
        import urllib.request

        for url in self._CHECK_URLS:
            try:
                req = urllib.request.Request(url, method="HEAD")  # noqa: S310
                with urllib.request.urlopen(req, timeout=3) as resp:  # noqa: S310
                    if resp.status == 200:
                        return True
            except Exception:
                continue
        return False

    def force_offline(self) -> None:
        """Force offline mode (for testing or user preference)."""
        self._forced = False
        self._is_online = False
        logger.info("connectivity_forced_offline")

    def force_online(self) -> None:
        """Force online mode."""
        self._forced = True
        self._is_online = True
        logger.info("connectivity_forced_online")

    def reset(self) -> None:
        """Reset to auto-detection mode."""
        self._forced = None
        self._last_check = None


class OfflineRouter:
    """Routes requests to local models when offline.

    Maintains a retry queue for requests that failed due to connectivity
    issues so they can be replayed when the connection is restored.
    """

    def __init__(
        self,
        provider_registry: ProviderRegistry,
        connectivity_checker: ConnectivityChecker,
    ) -> None:
        self._registry = provider_registry
        self._connectivity = connectivity_checker
        self._retry_queue: list[dict[str, object]] = []

    def get_available_model(self) -> str | None:
        """Get the best available local model (Ollama).

        Returns:
            Model ID string, or ``None`` if no local model is available.
        """
        provider = self._registry.get_provider("ollama")
        if provider is None:
            return None

        for model in provider.models:
            return model.id  # Return first available Ollama model

        return None

    def should_route_locally(self) -> bool:
        """Return ``True`` if offline or forced-local mode."""
        return not self._connectivity.is_online()

    def queue_for_retry(self, request: dict[str, object]) -> None:
        """Queue a failed request for retry when online.

        Args:
            request: The request data to queue.
        """
        self._retry_queue.append(request)
        logger.debug(
            "request_queued_for_retry",
            queue_size=len(self._retry_queue),
        )

    def get_queued_requests(self) -> list[dict[str, object]]:
        """Get requests queued for retry.

        Returns:
            Shallow copy of the queue.
        """
        return list(self._retry_queue)

    def clear_queue(self) -> None:
        """Clear the retry queue."""
        count = len(self._retry_queue)
        self._retry_queue.clear()
        logger.debug("retry_queue_cleared", cleared=count)
