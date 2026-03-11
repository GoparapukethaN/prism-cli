"""Provider health-check utilities."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger(__name__)


@dataclass
class HealthStatus:
    """Result of a single provider health check."""

    provider: str
    available: bool
    latency_ms: float | None = None
    error: str | None = None
    models_available: list[str] = field(default_factory=list)


class HealthChecker:
    """Checks provider availability, optionally with a lightweight API ping.

    The actual check logic is injected via ``check_fn`` so that tests can
    provide a mock without touching real endpoints.

    Usage::

        checker = HealthChecker(check_fn=my_ping_function)
        statuses = await checker.check_all(["openai", "anthropic"])
    """

    def __init__(
        self,
        check_fn: Callable[..., Any] | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        """Initialise the health checker.

        Args:
            check_fn: An async callable ``(provider: str) -> list[str]``
                that returns available model IDs.  If ``None``, every
                provider is assumed healthy.
            timeout_s: Per-provider timeout in seconds.
        """
        self._check_fn = check_fn
        self._timeout_s = timeout_s

    async def check_all(
        self,
        providers: list[str],
    ) -> dict[str, HealthStatus]:
        """Ping all *providers* concurrently.

        Args:
            providers: List of provider names to check.

        Returns:
            Dict mapping provider name to :class:`HealthStatus`.
        """
        tasks = [self.check_one(p) for p in providers]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return {status.provider: status for status in results}

    async def check_one(self, provider: str) -> HealthStatus:
        """Check a single provider.

        Args:
            provider: Canonical provider name.

        Returns:
            :class:`HealthStatus` for the provider.
        """
        if self._check_fn is None:
            return HealthStatus(provider=provider, available=True)

        start = time.perf_counter()
        try:
            models = await asyncio.wait_for(
                self._check_fn(provider),
                timeout=self._timeout_s,
            )
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                "health_check_ok",
                provider=provider,
                latency_ms=round(elapsed, 1),
                models=len(models),
            )
            return HealthStatus(
                provider=provider,
                available=True,
                latency_ms=elapsed,
                models_available=list(models) if models else [],
            )
        except TimeoutError:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("health_check_timeout", provider=provider)
            return HealthStatus(
                provider=provider,
                available=False,
                latency_ms=elapsed,
                error="Health check timed out",
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(
                "health_check_failed",
                provider=provider,
                error=str(exc),
            )
            return HealthStatus(
                provider=provider,
                available=False,
                latency_ms=elapsed,
                error=str(exc),
            )
