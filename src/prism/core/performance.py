"""Performance optimization — lazy loading, connection pooling, and benchmarking.

Provides utilities for deferring expensive imports, reusing HTTP connections
across providers, tracking startup latency, and running benchmark suites to
verify performance requirements are met.
"""

from __future__ import annotations

import importlib
import importlib.util
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lazy import registry — heavy modules loaded only when needed
# ---------------------------------------------------------------------------

_LAZY_MODULES: dict[str, str] = {
    "playwright": "playwright.sync_api",
    "tree_sitter": "tree_sitter",
    "chromadb": "chromadb",
    "beautifulsoup4": "bs4",
}

_loaded_modules: dict[str, Any] = {}


def lazy_import(module_name: str) -> Any:
    """Import a module on first access and cache the result.

    For modules registered in ``_LAZY_MODULES``, the actual import path
    is resolved from the mapping. All other names are imported directly.

    Args:
        module_name: Logical module name (e.g. ``"playwright"``).

    Returns:
        The imported module, or None if the module is not installed.
    """
    if module_name in _loaded_modules:
        return _loaded_modules[module_name]

    actual_name = _LAZY_MODULES.get(module_name, module_name)

    try:
        start = time.monotonic()
        mod = importlib.import_module(actual_name)
        elapsed_ms = (time.monotonic() - start) * 1000
        _loaded_modules[module_name] = mod
        logger.debug("lazy_import", module=actual_name, time_ms=f"{elapsed_ms:.1f}")
        return mod
    except ImportError:
        logger.debug("lazy_import_unavailable", module=actual_name)
        return None


def is_module_available(module_name: str) -> bool:
    """Check whether a module can be imported without actually importing it.

    Args:
        module_name: Logical module name.

    Returns:
        True if the module is installed and importable.
    """
    if module_name in _loaded_modules:
        return True
    actual_name = _LAZY_MODULES.get(module_name, module_name)
    try:
        spec = importlib.util.find_spec(actual_name)
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


def clear_lazy_cache() -> None:
    """Clear the lazy import cache. Primarily for testing."""
    _loaded_modules.clear()


# ---------------------------------------------------------------------------
# Benchmark types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkResult:
    """Result of a single performance benchmark run.

    Attributes:
        name: Human-readable name for the benchmark.
        duration_ms: Total wall-clock time across all iterations.
        iterations: Number of iterations executed.
        avg_ms: Average time per iteration.
        min_ms: Fastest iteration.
        max_ms: Slowest iteration.
        passed: Whether the average was within the threshold.
        threshold_ms: Maximum allowed average time per iteration.
    """

    name: str
    duration_ms: float
    iterations: int
    avg_ms: float
    min_ms: float
    max_ms: float
    passed: bool
    threshold_ms: float


@dataclass
class BenchmarkSuite:
    """Aggregated collection of benchmark results.

    Attributes:
        results: List of individual benchmark results.
        total_duration_ms: Sum of all benchmark durations.
    """

    results: list[BenchmarkResult] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def all_passed(self) -> bool:
        """Return True if every benchmark in the suite passed."""
        if not self.results:
            return True
        return all(r.passed for r in self.results)

    @property
    def failed_count(self) -> int:
        """Count of benchmarks that exceeded their threshold."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def passed_count(self) -> int:
        """Count of benchmarks that passed their threshold."""
        return sum(1 for r in self.results if r.passed)


class PerformanceBenchmark:
    """Performance benchmark suite for Prism.

    Run individual benchmarks, accumulate results, and inspect the
    overall suite status.

    Example::

        bench = PerformanceBenchmark()
        result = bench.benchmark("json_parse", json.loads, args=('{"a":1}',))
        assert result.passed
    """

    def __init__(self) -> None:
        self._suite = BenchmarkSuite()

    def benchmark(
        self,
        name: str,
        func: Callable[..., Any],
        iterations: int = 10,
        threshold_ms: float = 1000.0,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Run a benchmark and record the result.

        Args:
            name: Descriptive name for the benchmark.
            func: Callable to benchmark.
            iterations: Number of times to invoke ``func``.
            threshold_ms: Maximum acceptable average time (ms).
            args: Positional arguments forwarded to ``func``.
            kwargs: Keyword arguments forwarded to ``func``.

        Returns:
            The BenchmarkResult for this run.

        Raises:
            ValueError: If iterations is less than 1.
        """
        if iterations < 1:
            raise ValueError("iterations must be at least 1")

        resolved_kwargs = kwargs or {}
        times: list[float] = []

        for _ in range(iterations):
            start = time.monotonic()
            func(*args, **resolved_kwargs)
            elapsed_ms = (time.monotonic() - start) * 1000
            times.append(elapsed_ms)

        total = sum(times)
        avg = total / len(times)
        result = BenchmarkResult(
            name=name,
            duration_ms=total,
            iterations=iterations,
            avg_ms=avg,
            min_ms=min(times),
            max_ms=max(times),
            passed=avg <= threshold_ms,
            threshold_ms=threshold_ms,
        )

        self._suite.results.append(result)
        self._suite.total_duration_ms += result.duration_ms
        logger.debug(
            "benchmark_complete",
            name=name,
            avg_ms=f"{avg:.2f}",
            passed=result.passed,
        )
        return result

    def get_suite(self) -> BenchmarkSuite:
        """Return the accumulated benchmark suite."""
        return self._suite

    def reset(self) -> None:
        """Discard all recorded benchmark results."""
        self._suite = BenchmarkSuite()


# ---------------------------------------------------------------------------
# Connection pooling
# ---------------------------------------------------------------------------


class ConnectionPool:
    """HTTP connection pool for reusing connections across providers.

    Wraps ``httpx.Client`` and ``httpx.AsyncClient`` instances, keyed
    by base URL.  All connections share configurable limits.

    Args:
        max_connections: Maximum total connections per client.
        max_keepalive: Maximum keep-alive connections per client.
    """

    def __init__(
        self,
        max_connections: int = 20,
        max_keepalive: int = 10,
    ) -> None:
        if max_connections < 1:
            raise ValueError("max_connections must be at least 1")
        if max_keepalive < 0:
            raise ValueError("max_keepalive must be non-negative")
        self._max_connections = max_connections
        self._max_keepalive = max_keepalive
        self._clients: dict[str, Any] = {}

    @property
    def max_connections(self) -> int:
        """Return the maximum connections setting."""
        return self._max_connections

    @property
    def max_keepalive(self) -> int:
        """Return the maximum keep-alive connections setting."""
        return self._max_keepalive

    def get_client(self, base_url: str = "") -> Any:
        """Get or create a synchronous ``httpx.Client`` for a base URL.

        Args:
            base_url: Optional base URL. Clients are keyed by this value.

        Returns:
            An httpx.Client instance.
        """
        import httpx

        key = base_url or "default"
        if key not in self._clients:
            client_kwargs: dict[str, Any] = {
                "timeout": httpx.Timeout(30.0, connect=10.0),
                "limits": httpx.Limits(
                    max_connections=self._max_connections,
                    max_keepalive_connections=self._max_keepalive,
                ),
                "follow_redirects": True,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            self._clients[key] = httpx.Client(**client_kwargs)
            logger.debug("connection_pool_create", key=key, type="sync")
        return self._clients[key]

    def get_async_client(self, base_url: str = "") -> Any:
        """Get or create an asynchronous ``httpx.AsyncClient`` for a base URL.

        Args:
            base_url: Optional base URL. Clients are keyed by this value.

        Returns:
            An httpx.AsyncClient instance.
        """
        import httpx

        key = f"async_{base_url or 'default'}"
        if key not in self._clients:
            client_kwargs: dict[str, Any] = {
                "timeout": httpx.Timeout(30.0, connect=10.0),
                "limits": httpx.Limits(
                    max_connections=self._max_connections,
                    max_keepalive_connections=self._max_keepalive,
                ),
                "follow_redirects": True,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            self._clients[key] = httpx.AsyncClient(**client_kwargs)
            logger.debug("connection_pool_create", key=key, type="async")
        return self._clients[key]

    def close_all(self) -> None:
        """Close all pooled connections and clear the pool."""
        for key, client in list(self._clients.items()):
            try:
                client.close()
                logger.debug("connection_pool_close", key=key)
            except Exception:  # noqa: S110
                pass
        self._clients.clear()

    @property
    def active_connections(self) -> int:
        """Return the number of active connection pools."""
        return len(self._clients)

    def has_client(self, base_url: str = "") -> bool:
        """Check whether a client for the given base URL already exists.

        Args:
            base_url: The base URL to check.

        Returns:
            True if a client is already pooled for this URL.
        """
        key = base_url or "default"
        return key in self._clients or f"async_{key}" in self._clients


# ---------------------------------------------------------------------------
# Startup timer
# ---------------------------------------------------------------------------


@dataclass
class StartupTimer:
    """Tracks startup performance via named checkpoints.

    Call ``start()`` to begin timing, then ``checkpoint(name)`` at
    each significant point during initialization. Retrieve a report
    of all checkpoints via ``get_report()``.

    Attributes:
        checkpoints: List of (name, elapsed_ms) tuples.
        start_time: Monotonic time when ``start()`` was called.
    """

    checkpoints: list[tuple[str, float]] = field(default_factory=list)
    start_time: float = 0.0

    def start(self) -> None:
        """Record the start time."""
        self.start_time = time.monotonic()
        self.checkpoints = []

    def checkpoint(self, name: str) -> float:
        """Record a named checkpoint and return elapsed milliseconds.

        Args:
            name: Descriptive label for this checkpoint.

        Returns:
            Milliseconds elapsed since ``start()`` was called.

        Raises:
            RuntimeError: If ``start()`` was not called first.
        """
        if self.start_time == 0.0:
            raise RuntimeError("StartupTimer.start() must be called before checkpoint()")
        elapsed_ms = (time.monotonic() - self.start_time) * 1000
        self.checkpoints.append((name, elapsed_ms))
        return elapsed_ms

    def get_report(self) -> dict[str, float]:
        """Return a mapping of checkpoint names to elapsed milliseconds.

        Returns:
            Ordered dict of {checkpoint_name: elapsed_ms}.
        """
        return {name: ms for name, ms in self.checkpoints}

    @property
    def total_ms(self) -> float:
        """Return the elapsed time of the last checkpoint, or 0.0 if none."""
        if not self.checkpoints:
            return 0.0
        return self.checkpoints[-1][1]

    @property
    def checkpoint_count(self) -> int:
        """Return the number of recorded checkpoints."""
        return len(self.checkpoints)
