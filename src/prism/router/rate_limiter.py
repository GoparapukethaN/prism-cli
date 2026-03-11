"""Sliding-window rate limiter for per-provider request tracking.

Tracks request timestamps in memory and exposes helpers to check
whether a provider is currently rate-limited and how long to wait
before retrying.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

# Default limits used when a provider has no explicit configuration.
_DEFAULT_REQUESTS_PER_MINUTE: int = 60
_DEFAULT_WINDOW_SECONDS: float = 60.0


@dataclass
class _ProviderWindow:
    """Internal state for one provider's sliding window."""

    timestamps: deque[float] = field(default_factory=deque)
    max_requests: int = _DEFAULT_REQUESTS_PER_MINUTE
    window_seconds: float = _DEFAULT_WINDOW_SECONDS


class RateLimiter:
    """In-memory sliding-window rate limiter.

    Maintains per-provider request counts using a sliding time window.
    This is a *local* tracker — it does **not** communicate with provider
    APIs and should be used in conjunction with HTTP 429 handling.
    """

    def __init__(
        self,
        provider_limits: dict[str, int] | None = None,
        *,
        window_seconds: float = _DEFAULT_WINDOW_SECONDS,
    ) -> None:
        """Initialise the rate limiter.

        Args:
            provider_limits: Mapping of ``provider_name → max_requests``
                per window.  Providers not listed use the default limit.
            window_seconds: Length of the sliding window in seconds.
        """
        self._window_seconds = window_seconds
        self._provider_limits: dict[str, int] = provider_limits or {}
        self._windows: dict[str, _ProviderWindow] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_request(self, provider: str) -> None:
        """Record a request for *provider* at the current time.

        Args:
            provider: Provider name (e.g. ``"anthropic"``).
        """
        window = self._get_window(provider)
        window.timestamps.append(time.monotonic())

    def is_rate_limited(self, provider: str) -> bool:
        """Check whether *provider* has hit its per-window limit.

        Args:
            provider: Provider name.

        Returns:
            ``True`` if the number of requests within the current window
            meets or exceeds the configured limit.
        """
        window = self._get_window(provider)
        self._prune(window)
        return len(window.timestamps) >= window.max_requests

    def get_wait_time(self, provider: str) -> float:
        """Seconds until at least one request slot becomes available.

        Args:
            provider: Provider name.

        Returns:
            Wait time in seconds.  ``0.0`` if the provider is not
            currently rate-limited.
        """
        window = self._get_window(provider)
        self._prune(window)

        if len(window.timestamps) < window.max_requests:
            return 0.0

        # Oldest request in the window — slot frees when it expires
        oldest = window.timestamps[0]
        wait = (oldest + window.window_seconds) - time.monotonic()
        return max(0.0, wait)

    def get_remaining_requests(self, provider: str) -> int:
        """Number of requests remaining before the limit is reached.

        Args:
            provider: Provider name.

        Returns:
            Non-negative integer of remaining requests.
        """
        window = self._get_window(provider)
        self._prune(window)
        return max(0, window.max_requests - len(window.timestamps))

    def reset(self, provider: str) -> None:
        """Clear the window for *provider* (e.g. after a long pause).

        Args:
            provider: Provider name.
        """
        window = self._get_window(provider)
        window.timestamps.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_window(self, provider: str) -> _ProviderWindow:
        """Get or create the sliding window for a provider."""
        if provider not in self._windows:
            limit = self._provider_limits.get(provider, _DEFAULT_REQUESTS_PER_MINUTE)
            self._windows[provider] = _ProviderWindow(
                max_requests=limit,
                window_seconds=self._window_seconds,
            )
        return self._windows[provider]

    def _prune(self, window: _ProviderWindow) -> None:
        """Remove timestamps outside the current window."""
        cutoff = time.monotonic() - window.window_seconds
        while window.timestamps and window.timestamps[0] < cutoff:
            window.timestamps.popleft()
