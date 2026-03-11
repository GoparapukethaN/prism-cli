"""Provider health-check utilities.

Provides :class:`HealthChecker` for startup/periodic provider pings,
:class:`HealthCheckResult` for per-provider results, and a
:class:`ProviderDashboard` for Rich terminal display.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

try:
    import httpx as _httpx_mod
except ImportError:  # pragma: no cover
    _httpx_mod = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Callable

    from prism.auth.manager import AuthManager
    from prism.providers.registry import ProviderRegistry

logger = structlog.get_logger(__name__)

# Cheapest model per provider for health-check pings.
_CHEAPEST_MODEL: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "google": "gemini/gemini-2.0-flash",
    "deepseek": "deepseek/deepseek-chat",
    "groq": "groq/mixtral-8x7b-32768",
    "mistral": "mistral/mistral-small-latest",
    "ollama": "ollama/qwen2.5-coder:7b",
    "kimi": "moonshot/moonshot-v1-8k",
    "perplexity": "perplexity/llama-3.1-sonar-small-128k-online",
    "qwen": "qwen/qwen-turbo",
    "cohere": "cohere/command-r",
    "together_ai": "together_ai/meta-llama/Llama-3-70b-chat-hf",
    "fireworks_ai": "fireworks_ai/llama-v3p1-70b-instruct",
}

# Display names for providers with free tiers.
_FREE_TIER_PROVIDERS: frozenset[str] = frozenset({
    "google", "groq",
})

# Local providers (no API key needed).
_LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama"})

_OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
_HEALTH_CHECK_TIMEOUT = 5.0


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class HealthStatus:
    """Result of a single provider health check (legacy)."""

    provider: str
    available: bool
    latency_ms: float | None = None
    error: str | None = None
    models_available: list[str] = field(default_factory=list)


@dataclass
class HealthCheckResult:
    """Detailed result of a single provider health check."""

    provider: str
    display_name: str
    available: bool
    latency_ms: float
    model_used: str
    checked_at: datetime
    error: str | None = None
    is_free_tier: bool = False
    is_local: bool = False


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


# ======================================================================
# HealthChecker
# ======================================================================


class HealthChecker:
    """Checks provider availability on startup and periodically.

    Supports two initialization patterns:

    **New-style** (recommended)::

        checker = HealthChecker(
            auth_manager=auth,
            provider_registry=registry,
            litellm_backend=mock_backend,
        )
        results = await checker.check_all()

    **Legacy-style** (backward compatible)::

        checker = HealthChecker(check_fn=my_ping_function)
        statuses = await checker.check_all(["openai", "anthropic"])
    """

    def __init__(
        self,
        auth_manager: AuthManager | None = None,
        provider_registry: ProviderRegistry | None = None,
        litellm_backend: Any | None = None,
        *,
        check_fn: Callable[..., Any] | None = None,
        timeout_s: float = _HEALTH_CHECK_TIMEOUT,
    ) -> None:
        """Initialise the health checker.

        Args:
            auth_manager: AuthManager for checking API key
                availability.
            provider_registry: Registry for provider/model metadata.
            litellm_backend: Object with ``acompletion`` method.
                **Tests MUST inject a mock here.**
            check_fn: Legacy async callable
                ``(provider: str) -> list[str]``.
            timeout_s: Per-provider timeout in seconds.
        """
        self._auth = auth_manager
        self._registry = provider_registry
        self._backend = litellm_backend
        self._check_fn = check_fn
        self._timeout_s = timeout_s

        # State
        self._results: dict[str, HealthCheckResult] = {}
        self._lock = threading.Lock()
        self._bg_thread: threading.Thread | None = None
        self._bg_stop = threading.Event()

    @property
    def _is_legacy_mode(self) -> bool:
        """True when using the legacy check_fn interface."""
        return (
            self._auth is None
            and self._registry is None
            and self._backend is None
        )

    # ------------------------------------------------------------------
    # Public API — new-style
    # ------------------------------------------------------------------

    async def check_all(
        self,
        providers: list[str] | None = None,
    ) -> list[HealthCheckResult] | dict[str, HealthStatus]:
        """Ping all configured providers concurrently.

        In new-style mode (auth_manager/registry/backend injected),
        returns ``list[HealthCheckResult]`` with a 5s timeout per
        provider.

        In legacy mode (check_fn), returns
        ``dict[str, HealthStatus]`` for backward compatibility.

        Args:
            providers: Explicit list of provider names.
                In new-style mode, defaults to all registered
                providers. In legacy mode, this is required.

        Returns:
            Results of the health checks.
        """
        if self._is_legacy_mode:
            return await self._legacy_check_all(providers or [])

        if providers is None and self._registry is not None:
            providers = list(
                self._registry.all_providers.keys()
            )
        elif providers is None:
            providers = []

        tasks = [
            self.check_provider(p) for p in providers
        ]
        results = await asyncio.gather(*tasks)
        result_list = list(results)
        with self._lock:
            for r in result_list:
                self._results[r.provider] = r
        return result_list

    async def check_provider(
        self,
        provider_name: str,
    ) -> HealthCheckResult:
        """Check a single provider's health.

        Uses the cheapest model for the provider and sends a
        minimal completion request ("Hi", max_tokens=1) with a
        5-second timeout.

        Args:
            provider_name: Canonical provider name.

        Returns:
            :class:`HealthCheckResult` for the provider.
        """
        display_name = self._get_display_name(provider_name)
        model_used = _CHEAPEST_MODEL.get(provider_name, "unknown")
        is_free = provider_name in _FREE_TIER_PROVIDERS
        is_local = provider_name in _LOCAL_PROVIDERS

        # If provider has models in registry, pick cheapest.
        if self._registry is not None:
            cfg = self._registry.get_provider(provider_name)
            if cfg and cfg.models:
                cheapest = min(
                    cfg.models,
                    key=lambda m: (
                        m.input_cost_per_1m + m.output_cost_per_1m
                    ),
                )
                model_used = cheapest.id
                display_name = cfg.display_name
                if cfg.free_tier is not None:
                    is_free = True

        # Check API key availability (skip for local providers).
        if not is_local and self._auth is not None:
            try:
                key = self._auth.get_key(provider_name)
                if key is None:
                    return HealthCheckResult(
                        provider=provider_name,
                        display_name=display_name,
                        available=False,
                        latency_ms=0.0,
                        model_used=model_used,
                        error="API key not configured",
                        is_free_tier=is_free,
                        is_local=is_local,
                        checked_at=datetime.now(UTC),
                    )
            except Exception:
                return HealthCheckResult(
                    provider=provider_name,
                    display_name=display_name,
                    available=False,
                    latency_ms=0.0,
                    model_used=model_used,
                    error="API key not configured",
                    is_free_tier=is_free,
                    is_local=is_local,
                    checked_at=datetime.now(UTC),
                )

        # Ollama: check local server first.
        if is_local:
            return await self._check_ollama(
                provider_name, display_name, model_used,
            )

        # General provider: send minimal completion.
        return await self._ping_provider(
            provider_name, display_name, model_used,
            is_free, is_local,
        )

    def start_background_checks(
        self,
        interval_seconds: int = 300,
    ) -> None:
        """Start background re-checks every *interval_seconds*.

        Spawns a daemon thread that re-runs ``check_all`` at the
        specified interval until :meth:`stop_background_checks`
        is called.

        Args:
            interval_seconds: Seconds between re-checks.
        """
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return  # Already running

        self._bg_stop.clear()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            try:
                while not self._bg_stop.is_set():
                    if self._bg_stop.wait(interval_seconds):
                        break
                    try:
                        loop.run_until_complete(self.check_all())
                    except Exception as exc:
                        logger.warning(
                            "bg_health_check_error",
                            error=str(exc),
                        )
            finally:
                loop.close()

        self._bg_thread = threading.Thread(
            target=_run, daemon=True, name="prism-health-bg",
        )
        self._bg_thread.start()
        logger.info(
            "bg_health_checks_started",
            interval=interval_seconds,
        )

    def stop_background_checks(self) -> None:
        """Stop the background checking thread."""
        self._bg_stop.set()
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=5.0)
            self._bg_thread = None
        logger.info("bg_health_checks_stopped")

    def get_available_providers(self) -> list[str]:
        """Return list of providers that passed health check.

        Returns:
            Provider names that are currently available.
        """
        with self._lock:
            return [
                name for name, r in self._results.items()
                if r.available
            ]

    def get_unavailable_providers(self) -> list[str]:
        """Return list of providers that failed health check.

        Returns:
            Provider names that are currently unavailable.
        """
        with self._lock:
            return [
                name for name, r in self._results.items()
                if not r.available
            ]

    def get_results(self) -> dict[str, HealthCheckResult]:
        """Return a snapshot of all stored results.

        Returns:
            Dict mapping provider name to HealthCheckResult.
        """
        with self._lock:
            return dict(self._results)

    def format_startup_summary(
        self,
        results: list[HealthCheckResult],
    ) -> str:
        """Format results for terminal display.

        Example output::

            Providers:
              [check] Anthropic     (claude-haiku-4-5) - 234ms
              [check] Gemini        (flash, free) - 189ms  [FREE]
              [check] Groq          (mixtral-8x7b) - 89ms  [FREE]
              [check] Ollama        (qwen2.5-coder:7b) - 12ms  [LOCAL]
              [cross] DeepSeek      - API key not configured
              [cross] OpenAI        - Connection timeout

        Args:
            results: List of HealthCheckResult to format.

        Returns:
            Formatted multi-line string.
        """
        lines: list[str] = []
        # Sort: available first, then alphabetically.
        sorted_results = sorted(
            results,
            key=lambda r: (not r.available, r.display_name),
        )

        for r in sorted_results:
            icon = "\u2713" if r.available else "\u2717"
            name = r.display_name.ljust(16)

            if r.available:
                # Shorten model name for display.
                short_model = self._short_model_name(
                    r.model_used
                )
                latency_str = f"{r.latency_ms:.0f}ms"
                suffix = ""
                if r.is_local:
                    suffix = "  [LOCAL]"
                elif r.is_free_tier:
                    suffix = "  [FREE]"
                lines.append(
                    f"  {icon} {name}({short_model})"
                    f" - {latency_str}{suffix}"
                )
            else:
                error_msg = r.error or "Unknown error"
                lines.append(
                    f"  {icon} {name}- {error_msg}"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Legacy API (backward-compatible)
    # ------------------------------------------------------------------

    async def check_one(
        self,
        provider: str,
    ) -> HealthStatus:
        """Check a single provider (legacy API).

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
                models_available=(
                    list(models) if models else []
                ),
            )
        except TimeoutError:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(
                "health_check_timeout",
                provider=provider,
            )
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

        Runs health checks on all providers and enriches results
        with registry metadata (model counts, free tier info, etc.).

        Args:
            providers: Provider names to check.
            registry: Optional ProviderRegistry for enrichment.

        Returns:
            List of ProviderDashboardEntry, one per provider.
        """
        statuses = await self._legacy_check_all(providers)
        entries: list[ProviderDashboardEntry] = []

        for provider_name in providers:
            health = statuses.get(provider_name)
            status_str = "healthy"
            latency = None
            error = None
            models_count = 0
            display_name = (
                provider_name.replace("_", " ").title()
            )
            has_key = False
            free_tier = False
            free_remaining = None

            if health is not None:
                latency = health.latency_ms
                error = health.error
                if not health.available:
                    status_str = "down"
                elif (
                    health.latency_ms is not None
                    and health.latency_ms > 5000
                ):
                    status_str = "degraded"
                models_count = len(health.models_available)

            if registry is not None:
                provider_cfg = registry.get_provider(
                    provider_name
                )
                if provider_cfg is not None:
                    display_name = provider_cfg.display_name
                    models_count = len(provider_cfg.models)
                    free_tier = (
                        provider_cfg.free_tier is not None
                    )
                    free_remaining = (
                        registry.get_free_tier_remaining(
                            provider_name,
                        )
                    )

                has_key = registry.is_provider_available(
                    provider_name
                )
                if (
                    not has_key
                    and provider_name != "ollama"
                ):
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

    def generate_status_table(
        self,
        results: list[HealthCheckResult] | None = None,
    ) -> Any:
        """Return a Rich Table for the /status REPL command.

        Args:
            results: Health check results to display. If ``None``,
                uses the last stored results.

        Returns:
            A ``rich.table.Table`` instance.
        """
        from rich.table import Table

        if results is None:
            with self._lock:
                results = list(self._results.values())

        table = Table(
            title="Provider Status",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Status", width=6)
        table.add_column("Provider", min_width=16)
        table.add_column("Model", min_width=20)
        table.add_column("Latency", justify="right")
        table.add_column("Notes")

        sorted_results = sorted(
            results,
            key=lambda r: (not r.available, r.display_name),
        )

        for r in sorted_results:
            if r.available:
                status = "[green]\u2713[/green]"
                model = self._short_model_name(r.model_used)
                latency = f"{r.latency_ms:.0f}ms"
                notes_parts: list[str] = []
                if r.is_local:
                    notes_parts.append("[blue]LOCAL[/blue]")
                elif r.is_free_tier:
                    notes_parts.append(
                        "[green]FREE[/green]"
                    )
                notes = " ".join(notes_parts)
            else:
                status = "[red]\u2717[/red]"
                model = "-"
                latency = "-"
                notes = f"[red]{r.error or 'Error'}[/red]"

            table.add_row(
                status, r.display_name, model,
                latency, notes,
            )

        return table

    # ------------------------------------------------------------------
    # Internal: new-style checks
    # ------------------------------------------------------------------

    async def _ping_provider(
        self,
        provider_name: str,
        display_name: str,
        model_used: str,
        is_free: bool,
        is_local: bool,
    ) -> HealthCheckResult:
        """Send a minimal completion to the provider.

        Args:
            provider_name: Provider name.
            display_name: Human-readable name.
            model_used: Model ID for the ping.
            is_free: Whether the provider has a free tier.
            is_local: Whether the provider is local.

        Returns:
            HealthCheckResult with latency and status.
        """
        if self._backend is None:
            return HealthCheckResult(
                provider=provider_name,
                display_name=display_name,
                available=False,
                latency_ms=0.0,
                model_used=model_used,
                error="No LiteLLM backend configured",
                is_free_tier=is_free,
                is_local=is_local,
                checked_at=datetime.now(UTC),
            )

        start = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "model": model_used,
                "messages": [
                    {"role": "user", "content": "Hi"},
                ],
                "max_tokens": 1,
                "timeout": self._timeout_s,
            }
            # Inject API key if auth is available.
            if (
                self._auth is not None
                and not is_local
            ):
                try:
                    key = self._auth.get_key(provider_name)
                    if key is not None:
                        kwargs["api_key"] = key
                except Exception:
                    logger.debug(
                        "health_key_lookup_failed",
                        provider=provider_name,
                    )

            await asyncio.wait_for(
                self._backend.acompletion(**kwargs),
                timeout=self._timeout_s,
            )
            elapsed = (
                (time.perf_counter() - start) * 1000
            )

            logger.info(
                "health_ping_ok",
                provider=provider_name,
                model=model_used,
                latency_ms=round(elapsed, 1),
            )

            return HealthCheckResult(
                provider=provider_name,
                display_name=display_name,
                available=True,
                latency_ms=elapsed,
                model_used=model_used,
                is_free_tier=is_free,
                is_local=is_local,
                checked_at=datetime.now(UTC),
            )
        except TimeoutError:
            elapsed = (
                (time.perf_counter() - start) * 1000
            )
            logger.warning(
                "health_ping_timeout",
                provider=provider_name,
            )
            return HealthCheckResult(
                provider=provider_name,
                display_name=display_name,
                available=False,
                latency_ms=elapsed,
                model_used=model_used,
                error="Connection timeout",
                is_free_tier=is_free,
                is_local=is_local,
                checked_at=datetime.now(UTC),
            )
        except Exception as exc:
            elapsed = (
                (time.perf_counter() - start) * 1000
            )
            error_msg = str(exc) or type(exc).__name__
            logger.warning(
                "health_ping_failed",
                provider=provider_name,
                error=error_msg,
            )
            return HealthCheckResult(
                provider=provider_name,
                display_name=display_name,
                available=False,
                latency_ms=elapsed,
                model_used=model_used,
                error=error_msg,
                is_free_tier=is_free,
                is_local=is_local,
                checked_at=datetime.now(UTC),
            )

    async def _check_ollama(
        self,
        provider_name: str,
        display_name: str,
        model_used: str,
    ) -> HealthCheckResult:
        """Check Ollama by hitting the local tags endpoint.

        Falls back to an acompletion ping if httpx is not available.

        Args:
            provider_name: Provider name (always "ollama").
            display_name: Display name for Ollama.
            model_used: Default model to report.

        Returns:
            HealthCheckResult for Ollama.
        """
        if _httpx_mod is None:
            # httpx not available, fall back to ping
            return await self._ping_provider(
                provider_name, display_name, model_used,
                is_free=False, is_local=True,
            )

        start = time.perf_counter()
        try:
            async with _httpx_mod.AsyncClient(
                timeout=self._timeout_s,
            ) as client:
                resp = await asyncio.wait_for(
                    client.get(_OLLAMA_TAGS_URL),
                    timeout=self._timeout_s,
                )
            elapsed = (
                (time.perf_counter() - start) * 1000
            )

            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                if models:
                    model_used = (
                        f"ollama/"
                        f"{models[0].get('name', model_used)}"
                    )
                logger.info(
                    "ollama_health_ok",
                    models_found=len(models),
                    latency_ms=round(elapsed, 1),
                )
                return HealthCheckResult(
                    provider=provider_name,
                    display_name=display_name,
                    available=True,
                    latency_ms=elapsed,
                    model_used=model_used,
                    is_free_tier=False,
                    is_local=True,
                    checked_at=datetime.now(UTC),
                )
            else:
                return HealthCheckResult(
                    provider=provider_name,
                    display_name=display_name,
                    available=False,
                    latency_ms=elapsed,
                    model_used=model_used,
                    error=(
                        f"Ollama returned HTTP "
                        f"{resp.status_code}"
                    ),
                    is_free_tier=False,
                    is_local=True,
                    checked_at=datetime.now(UTC),
                )
        except TimeoutError:
            elapsed = (
                (time.perf_counter() - start) * 1000
            )
            return HealthCheckResult(
                provider=provider_name,
                display_name=display_name,
                available=False,
                latency_ms=elapsed,
                model_used=model_used,
                error="Ollama not responding (timeout)",
                is_free_tier=False,
                is_local=True,
                checked_at=datetime.now(UTC),
            )
        except Exception as exc:
            elapsed = (
                (time.perf_counter() - start) * 1000
            )
            error_msg = str(exc) or type(exc).__name__
            logger.warning(
                "ollama_health_failed",
                error=error_msg,
            )
            return HealthCheckResult(
                provider=provider_name,
                display_name=display_name,
                available=False,
                latency_ms=elapsed,
                model_used=model_used,
                error=error_msg,
                is_free_tier=False,
                is_local=True,
                checked_at=datetime.now(UTC),
            )

    # ------------------------------------------------------------------
    # Internal: legacy checks
    # ------------------------------------------------------------------

    async def _legacy_check_all(
        self,
        providers: list[str],
    ) -> dict[str, HealthStatus]:
        """Run legacy check_fn-based checks.

        Args:
            providers: Provider names to check.

        Returns:
            Dict mapping provider name to HealthStatus.
        """
        tasks = [self.check_one(p) for p in providers]
        results = await asyncio.gather(
            *tasks, return_exceptions=False,
        )
        return {
            status.provider: status for status in results
        }

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _get_display_name(self, provider: str) -> str:
        """Get human-readable display name for a provider.

        Args:
            provider: Canonical provider name.

        Returns:
            Display name string.
        """
        if self._registry is not None:
            cfg = self._registry.get_provider(provider)
            if cfg is not None:
                return cfg.display_name
        return provider.replace("_", " ").title()

    @staticmethod
    def _short_model_name(model_id: str) -> str:
        """Shorten a model ID for display.

        Args:
            model_id: Full model identifier.

        Returns:
            Shortened model name.
        """
        # Strip provider prefix.
        if "/" in model_id:
            model_id = model_id.split("/", 1)[-1]
        # Shorten known long names.
        replacements = {
            "claude-haiku-4-5-20251001": "claude-haiku-4-5",
            "claude-sonnet-4-20250514": "claude-sonnet-4",
            "claude-opus-4-20250514": "claude-opus-4",
            "gemini-2.0-flash": "flash-2.0",
            "gemini-2.5-flash": "flash-2.5",
            "gemini-2.5-pro": "gemini-2.5-pro",
            "llama-3.3-70b-versatile": "llama-3.3-70b",
            "mixtral-8x7b-32768": "mixtral-8x7b",
            "mistral-small-latest": "mistral-small",
            "llama-3.1-sonar-small-128k-online": "sonar-small",
            "llama-3.1-sonar-large-128k-online": "sonar-large",
        }
        return replacements.get(model_id, model_id)
