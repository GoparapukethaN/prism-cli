"""Tests for prism.core.performance — lazy loading, connection pooling, and benchmarking."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from prism.core.performance import (
    _LAZY_MODULES,
    BenchmarkResult,
    BenchmarkSuite,
    ConnectionPool,
    PerformanceBenchmark,
    StartupTimer,
    _loaded_modules,
    clear_lazy_cache,
    is_module_available,
    lazy_import,
)

# ---------------------------------------------------------------------------
# lazy_import
# ---------------------------------------------------------------------------


class TestLazyImport:
    """Tests for the lazy_import function."""

    def setup_method(self) -> None:
        clear_lazy_cache()

    def teardown_method(self) -> None:
        clear_lazy_cache()

    def test_import_known_stdlib_module(self) -> None:
        mod = lazy_import("json")
        assert mod is not None
        assert hasattr(mod, "dumps")

    def test_import_caches_result(self) -> None:
        mod1 = lazy_import("json")
        mod2 = lazy_import("json")
        assert mod1 is mod2

    def test_import_unknown_module_returns_none(self) -> None:
        result = lazy_import("totally_nonexistent_module_xyz_123")
        assert result is None

    def test_import_from_lazy_modules_mapping(self) -> None:
        # "beautifulsoup4" maps to "bs4" in _LAZY_MODULES
        # If bs4 is installed we get a module, if not we get None
        mod = lazy_import("beautifulsoup4")
        if mod is not None:
            assert hasattr(mod, "BeautifulSoup")

    def test_import_stores_in_loaded_modules(self) -> None:
        clear_lazy_cache()
        lazy_import("os")
        assert "os" in _loaded_modules

    def test_cached_module_returned_without_reimport(self) -> None:
        clear_lazy_cache()
        sentinel = MagicMock()
        _loaded_modules["fake_module"] = sentinel
        result = lazy_import("fake_module")
        assert result is sentinel

    def test_import_populates_cache_on_success(self) -> None:
        clear_lazy_cache()
        mod = lazy_import("sys")
        assert mod is not None
        assert _loaded_modules["sys"] is mod


# ---------------------------------------------------------------------------
# is_module_available
# ---------------------------------------------------------------------------


class TestIsModuleAvailable:
    """Tests for the is_module_available function."""

    def setup_method(self) -> None:
        clear_lazy_cache()

    def teardown_method(self) -> None:
        clear_lazy_cache()

    def test_available_stdlib_module(self) -> None:
        assert is_module_available("json") is True

    def test_unavailable_module(self) -> None:
        assert is_module_available("nonexistent_module_xyz_999") is False

    def test_already_loaded_module(self) -> None:
        _loaded_modules["cached_test"] = MagicMock()
        assert is_module_available("cached_test") is True

    def test_mapped_module(self) -> None:
        # "beautifulsoup4" -> "bs4" in the mapping; check it doesn't crash
        result = is_module_available("beautifulsoup4")
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# clear_lazy_cache
# ---------------------------------------------------------------------------


class TestClearLazyCache:
    """Tests for the clear_lazy_cache function."""

    def test_clears_all_cached_modules(self) -> None:
        _loaded_modules["test_a"] = MagicMock()
        _loaded_modules["test_b"] = MagicMock()
        clear_lazy_cache()
        assert len(_loaded_modules) == 0


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    """Tests for the BenchmarkResult dataclass."""

    def test_fields_set_correctly(self) -> None:
        result = BenchmarkResult(
            name="test",
            duration_ms=100.0,
            iterations=10,
            avg_ms=10.0,
            min_ms=8.0,
            max_ms=15.0,
            passed=True,
            threshold_ms=20.0,
        )
        assert result.name == "test"
        assert result.duration_ms == 100.0
        assert result.iterations == 10
        assert result.avg_ms == 10.0
        assert result.min_ms == 8.0
        assert result.max_ms == 15.0
        assert result.passed is True
        assert result.threshold_ms == 20.0

    def test_passed_is_true_when_under_threshold(self) -> None:
        result = BenchmarkResult(
            name="fast",
            duration_ms=50.0,
            iterations=5,
            avg_ms=10.0,
            min_ms=9.0,
            max_ms=11.0,
            passed=True,
            threshold_ms=100.0,
        )
        assert result.passed is True

    def test_passed_is_false_when_over_threshold(self) -> None:
        result = BenchmarkResult(
            name="slow",
            duration_ms=5000.0,
            iterations=5,
            avg_ms=1000.0,
            min_ms=900.0,
            max_ms=1100.0,
            passed=False,
            threshold_ms=500.0,
        )
        assert result.passed is False

    def test_frozen_dataclass(self) -> None:
        result = BenchmarkResult(
            name="test",
            duration_ms=10.0,
            iterations=1,
            avg_ms=10.0,
            min_ms=10.0,
            max_ms=10.0,
            passed=True,
            threshold_ms=20.0,
        )
        with pytest.raises(AttributeError):
            result.name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------


class TestBenchmarkSuite:
    """Tests for the BenchmarkSuite dataclass."""

    def test_empty_suite_all_passed(self) -> None:
        suite = BenchmarkSuite()
        assert suite.all_passed is True
        assert suite.failed_count == 0
        assert suite.passed_count == 0

    def test_all_passed_when_all_benchmarks_pass(self) -> None:
        suite = BenchmarkSuite(
            results=[
                BenchmarkResult("a", 10.0, 1, 10.0, 10.0, 10.0, True, 100.0),
                BenchmarkResult("b", 20.0, 1, 20.0, 20.0, 20.0, True, 100.0),
            ],
            total_duration_ms=30.0,
        )
        assert suite.all_passed is True
        assert suite.failed_count == 0
        assert suite.passed_count == 2

    def test_not_all_passed_when_one_fails(self) -> None:
        suite = BenchmarkSuite(
            results=[
                BenchmarkResult("a", 10.0, 1, 10.0, 10.0, 10.0, True, 100.0),
                BenchmarkResult("b", 200.0, 1, 200.0, 200.0, 200.0, False, 100.0),
            ],
            total_duration_ms=210.0,
        )
        assert suite.all_passed is False
        assert suite.failed_count == 1
        assert suite.passed_count == 1

    def test_multiple_failures(self) -> None:
        suite = BenchmarkSuite(
            results=[
                BenchmarkResult("a", 200.0, 1, 200.0, 200.0, 200.0, False, 100.0),
                BenchmarkResult("b", 300.0, 1, 300.0, 300.0, 300.0, False, 100.0),
                BenchmarkResult("c", 10.0, 1, 10.0, 10.0, 10.0, True, 100.0),
            ],
            total_duration_ms=510.0,
        )
        assert suite.failed_count == 2
        assert suite.passed_count == 1

    def test_total_duration(self) -> None:
        suite = BenchmarkSuite(total_duration_ms=42.5)
        assert suite.total_duration_ms == 42.5


# ---------------------------------------------------------------------------
# PerformanceBenchmark
# ---------------------------------------------------------------------------


class TestPerformanceBenchmark:
    """Tests for the PerformanceBenchmark class."""

    def test_benchmark_fast_function_passes(self) -> None:
        bench = PerformanceBenchmark()
        result = bench.benchmark(
            name="fast_op",
            func=lambda: None,
            iterations=5,
            threshold_ms=1000.0,
        )
        assert result.passed is True
        assert result.name == "fast_op"
        assert result.iterations == 5
        assert result.avg_ms < 1000.0

    def test_benchmark_with_args(self) -> None:
        bench = PerformanceBenchmark()
        result = bench.benchmark(
            name="add_test",
            func=lambda x, y: x + y,
            iterations=3,
            threshold_ms=1000.0,
            args=(1, 2),
        )
        assert result.passed is True

    def test_benchmark_with_kwargs(self) -> None:
        bench = PerformanceBenchmark()
        result = bench.benchmark(
            name="kw_test",
            func=lambda a=0: a * 2,
            iterations=3,
            threshold_ms=1000.0,
            kwargs={"a": 5},
        )
        assert result.passed is True

    def test_benchmark_slow_function_fails_threshold(self) -> None:
        def slow_func() -> None:
            time.sleep(0.02)

        bench = PerformanceBenchmark()
        result = bench.benchmark(
            name="slow_op",
            func=slow_func,
            iterations=3,
            threshold_ms=1.0,  # 1ms threshold — will fail
        )
        assert result.passed is False
        assert result.avg_ms > 1.0

    def test_benchmark_accumulates_in_suite(self) -> None:
        bench = PerformanceBenchmark()
        bench.benchmark("a", lambda: None, iterations=2, threshold_ms=1000.0)
        bench.benchmark("b", lambda: None, iterations=2, threshold_ms=1000.0)
        suite = bench.get_suite()
        assert len(suite.results) == 2

    def test_get_suite_returns_suite(self) -> None:
        bench = PerformanceBenchmark()
        suite = bench.get_suite()
        assert isinstance(suite, BenchmarkSuite)
        assert len(suite.results) == 0

    def test_reset_clears_results(self) -> None:
        bench = PerformanceBenchmark()
        bench.benchmark("a", lambda: None, iterations=2, threshold_ms=1000.0)
        assert len(bench.get_suite().results) == 1
        bench.reset()
        assert len(bench.get_suite().results) == 0
        assert bench.get_suite().total_duration_ms == 0.0

    def test_benchmark_min_max(self) -> None:
        bench = PerformanceBenchmark()
        result = bench.benchmark(
            name="minmax",
            func=lambda: None,
            iterations=10,
            threshold_ms=1000.0,
        )
        assert result.min_ms <= result.avg_ms
        assert result.max_ms >= result.avg_ms

    def test_benchmark_duration_is_sum(self) -> None:
        bench = PerformanceBenchmark()
        result = bench.benchmark(
            name="duration",
            func=lambda: None,
            iterations=5,
            threshold_ms=1000.0,
        )
        # duration_ms should be approximately avg_ms * iterations
        assert result.duration_ms >= 0.0
        assert abs(result.duration_ms - result.avg_ms * result.iterations) < 1.0

    def test_benchmark_invalid_iterations_raises(self) -> None:
        bench = PerformanceBenchmark()
        with pytest.raises(ValueError, match="iterations must be at least 1"):
            bench.benchmark("bad", lambda: None, iterations=0, threshold_ms=1000.0)

    def test_total_duration_accumulates(self) -> None:
        bench = PerformanceBenchmark()
        bench.benchmark("a", lambda: None, iterations=1, threshold_ms=1000.0)
        bench.benchmark("b", lambda: None, iterations=1, threshold_ms=1000.0)
        suite = bench.get_suite()
        assert suite.total_duration_ms > 0.0


# ---------------------------------------------------------------------------
# ConnectionPool
# ---------------------------------------------------------------------------


class TestConnectionPool:
    """Tests for the ConnectionPool class."""

    def test_init_defaults(self) -> None:
        pool = ConnectionPool()
        assert pool.max_connections == 20
        assert pool.max_keepalive == 10
        assert pool.active_connections == 0

    def test_init_custom_values(self) -> None:
        pool = ConnectionPool(max_connections=50, max_keepalive=25)
        assert pool.max_connections == 50
        assert pool.max_keepalive == 25

    def test_init_invalid_max_connections(self) -> None:
        with pytest.raises(ValueError, match="max_connections"):
            ConnectionPool(max_connections=0)

    def test_init_invalid_max_keepalive(self) -> None:
        with pytest.raises(ValueError, match="max_keepalive"):
            ConnectionPool(max_keepalive=-1)

    def test_get_client_creates_httpx_client(self) -> None:
        pool = ConnectionPool()
        client = pool.get_client()
        assert client is not None
        assert pool.active_connections == 1
        pool.close_all()

    def test_get_client_reuses_existing(self) -> None:
        pool = ConnectionPool()
        client1 = pool.get_client()
        client2 = pool.get_client()
        assert client1 is client2
        assert pool.active_connections == 1
        pool.close_all()

    def test_get_client_with_base_url(self) -> None:
        pool = ConnectionPool()
        client = pool.get_client(base_url="https://api.example.com")
        assert client is not None
        assert pool.active_connections == 1
        pool.close_all()

    def test_different_base_urls_get_different_clients(self) -> None:
        pool = ConnectionPool()
        client1 = pool.get_client(base_url="https://api1.example.com")
        client2 = pool.get_client(base_url="https://api2.example.com")
        assert client1 is not client2
        assert pool.active_connections == 2
        pool.close_all()

    def test_get_async_client(self) -> None:
        pool = ConnectionPool()
        client = pool.get_async_client()
        assert client is not None
        assert pool.active_connections == 1
        pool.close_all()

    def test_get_async_client_reuses_existing(self) -> None:
        pool = ConnectionPool()
        client1 = pool.get_async_client()
        client2 = pool.get_async_client()
        assert client1 is client2
        assert pool.active_connections == 1
        pool.close_all()

    def test_get_async_client_with_base_url(self) -> None:
        pool = ConnectionPool()
        client = pool.get_async_client(base_url="https://api.example.com")
        assert client is not None
        pool.close_all()

    def test_sync_and_async_are_separate(self) -> None:
        pool = ConnectionPool()
        sync = pool.get_client()
        async_client = pool.get_async_client()
        assert sync is not async_client
        assert pool.active_connections == 2
        pool.close_all()

    def test_close_all_clears_pool(self) -> None:
        pool = ConnectionPool()
        pool.get_client()
        pool.get_async_client()
        assert pool.active_connections == 2
        pool.close_all()
        assert pool.active_connections == 0

    def test_close_all_handles_error_gracefully(self) -> None:
        pool = ConnectionPool()
        pool.get_client()
        # Replace client with a mock that raises on close
        mock_client = MagicMock()
        mock_client.close.side_effect = RuntimeError("close failed")
        pool._clients["default"] = mock_client
        # Should not raise
        pool.close_all()
        assert pool.active_connections == 0

    def test_has_client_false_initially(self) -> None:
        pool = ConnectionPool()
        assert pool.has_client() is False
        assert pool.has_client("https://api.example.com") is False

    def test_has_client_true_after_creation(self) -> None:
        pool = ConnectionPool()
        pool.get_client()
        assert pool.has_client() is True
        pool.close_all()

    def test_has_client_with_base_url(self) -> None:
        pool = ConnectionPool()
        pool.get_client(base_url="https://api.test.com")
        assert pool.has_client("https://api.test.com") is True
        assert pool.has_client("https://other.test.com") is False
        pool.close_all()


# ---------------------------------------------------------------------------
# StartupTimer
# ---------------------------------------------------------------------------


class TestStartupTimer:
    """Tests for the StartupTimer dataclass."""

    def test_initial_state(self) -> None:
        timer = StartupTimer()
        assert timer.start_time == 0.0
        assert timer.checkpoints == []
        assert timer.total_ms == 0.0
        assert timer.checkpoint_count == 0

    def test_start_sets_time(self) -> None:
        timer = StartupTimer()
        timer.start()
        assert timer.start_time > 0.0

    def test_start_clears_previous_checkpoints(self) -> None:
        timer = StartupTimer()
        timer.start()
        timer.checkpoint("a")
        timer.start()
        assert timer.checkpoints == []

    def test_checkpoint_records_elapsed(self) -> None:
        timer = StartupTimer()
        timer.start()
        time.sleep(0.01)
        elapsed = timer.checkpoint("step1")
        assert elapsed > 0.0
        assert len(timer.checkpoints) == 1
        assert timer.checkpoints[0][0] == "step1"

    def test_multiple_checkpoints(self) -> None:
        timer = StartupTimer()
        timer.start()
        timer.checkpoint("step1")
        time.sleep(0.01)
        timer.checkpoint("step2")
        assert len(timer.checkpoints) == 2
        assert timer.checkpoints[1][1] > timer.checkpoints[0][1]

    def test_get_report(self) -> None:
        timer = StartupTimer()
        timer.start()
        timer.checkpoint("init")
        timer.checkpoint("load_config")
        report = timer.get_report()
        assert "init" in report
        assert "load_config" in report
        assert isinstance(report["init"], float)

    def test_get_report_empty(self) -> None:
        timer = StartupTimer()
        assert timer.get_report() == {}

    def test_total_ms_returns_last_checkpoint(self) -> None:
        timer = StartupTimer()
        timer.start()
        timer.checkpoint("first")
        time.sleep(0.01)
        timer.checkpoint("last")
        assert timer.total_ms == timer.checkpoints[-1][1]

    def test_total_ms_zero_when_no_checkpoints(self) -> None:
        timer = StartupTimer()
        timer.start()
        assert timer.total_ms == 0.0

    def test_checkpoint_count(self) -> None:
        timer = StartupTimer()
        timer.start()
        timer.checkpoint("a")
        timer.checkpoint("b")
        timer.checkpoint("c")
        assert timer.checkpoint_count == 3

    def test_checkpoint_without_start_raises(self) -> None:
        timer = StartupTimer()
        with pytest.raises(RuntimeError, match="start.*must be called"):
            timer.checkpoint("fail")

    def test_checkpoint_returns_increasing_values(self) -> None:
        timer = StartupTimer()
        timer.start()
        t1 = timer.checkpoint("a")
        t2 = timer.checkpoint("b")
        assert t2 >= t1


# ---------------------------------------------------------------------------
# Module-level mapping
# ---------------------------------------------------------------------------


class TestLazyModulesMapping:
    """Tests for the _LAZY_MODULES registry."""

    def test_playwright_mapped(self) -> None:
        assert _LAZY_MODULES["playwright"] == "playwright.sync_api"

    def test_tree_sitter_mapped(self) -> None:
        assert _LAZY_MODULES["tree_sitter"] == "tree_sitter"

    def test_chromadb_mapped(self) -> None:
        assert _LAZY_MODULES["chromadb"] == "chromadb"

    def test_beautifulsoup4_mapped(self) -> None:
        assert _LAZY_MODULES["beautifulsoup4"] == "bs4"
