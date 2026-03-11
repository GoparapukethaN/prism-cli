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


@dataclass
class ProviderDashboardEntry:
    """Entry in the provider availability dashboard."""

    provider: str
    display_name: str
    status: str  # "healthy", "degraded", "down", "unconfigured"
    latency_ms: float | None = None
    models_count: int = 0
    error: str | None = None
    has_api_key: bool = False
    free_tier: bool = False
    free_tier_remaining: int | None = None


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

    async def generate_dashboard(
        self,
        providers: list[str],
        registry: Any = None,
    ) -> list[ProviderDashboardEntry]:
        """Generate a provider availability dashboard.

        Runs health checks on all providers and enriches results with
        registry metadata (model counts, free tier info, etc.).

        Args:
            providers: Provider names to check.
            registry: Optional ProviderRegistry for enrichment.

        Returns:
            List of ProviderDashboardEntry, one per provider.
        """
        statuses = await self.check_all(providers)
        entries: list[ProviderDashboardEntry] = []

        for provider_name in providers:
            health = statuses.get(provider_name)
            status_str = "healthy"
            latency = None
            error = None
            models_count = 0
            display_name = provider_name.replace("_", " ").title()
            has_key = False
            free_tier = False
            free_remaining = None

            if health is not None:
                latency = health.latency_ms
                error = health.error
                if not health.available:
                    status_str = "down"
                elif health.latency_ms is not None and health.latency_ms > 5000:
                    status_str = "degraded"
                models_count = len(health.models_available)

            if registry is not None:
                provider_cfg = registry.get_provider(provider_name)
                if provider_cfg is not None:
                    display_name = provider_cfg.display_name
                    models_count = len(provider_cfg.models)
                    free_tier = provider_cfg.free_tier is not None
                    free_remaining = registry.get_free_tier_remaining(
                        provider_name,
                    )

                has_key = registry.is_provider_available(provider_name)
                if not has_key and provider_name != "ollama":
                    status_str = "unconfigured"

            entries.append(
                ProviderDashboardEntry(
                    provider=provider_name,
                    display_name=display_name,
                    status=status_str,
                    latency_ms=latency,
                    models_count=models_count,
                    error=error,
                    has_api_key=has_key,
                    free_tier=free_tier,
                    free_tier_remaining=free_remaining,
                )
            )

        return entries
