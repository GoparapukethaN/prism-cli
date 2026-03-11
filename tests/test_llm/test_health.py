"""Tests for HealthChecker — 55+ tests, fully offline.

All tests use mocks — absolutely no real API calls.
Covers: check_all, check_provider, format_startup_summary,
background checks, get_available/unavailable_providers,
Ollama special handling, no-API-key scenarios, thread safety,
legacy API backward compatibility, and ProviderDashboard.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC, datetime
from typing import Any
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import pytest

from prism.llm.health import (
    _CHEAPEST_MODEL,
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    ProviderDashboardEntry,
)
from prism.llm.mock import MockLiteLLM, MockResponse

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def mock_auth() -> MagicMock:
    """Mock AuthManager returning keys for most providers."""
    mgr = MagicMock()
    _keys: dict[str, str | None] = {
        "anthropic": "sk-ant-test-1234",
        "openai": "sk-test-1234",
        "google": "goog-test-1234",
        "deepseek": "sk-ds-test-1234",
        "groq": "gsk-test-1234",
        "mistral": "mist-test-1234",
    }

    def _get_key(provider: str) -> str | None:
        if provider in _keys:
            return _keys[provider]
        from prism.exceptions import KeyNotFoundError
        raise KeyNotFoundError(provider)

    mgr.get_key.side_effect = _get_key
    return mgr


@pytest.fixture
def mock_auth_no_keys() -> MagicMock:
    """Mock AuthManager that has no keys for any provider."""
    mgr = MagicMock()

    def _get_key(provider: str) -> str | None:
        from prism.exceptions import KeyNotFoundError
        raise KeyNotFoundError(provider)

    mgr.get_key.side_effect = _get_key
    return mgr


@pytest.fixture
def mock_registry() -> MagicMock:
    """Mock ProviderRegistry with basic provider configs."""
    registry = MagicMock()

    def _make_provider(
        name: str,
        display: str,
        model_id: str,
        cost: float = 0.10,
        free_tier: Any = None,
    ) -> MagicMock:
        cfg = MagicMock()
        cfg.name = name
        cfg.display_name = display
        model = MagicMock()
        model.id = model_id
        model.input_cost_per_1m = cost
        model.output_cost_per_1m = cost
        cfg.models = [model]
        cfg.free_tier = free_tier
        return cfg

    configs = {
        "anthropic": _make_provider(
            "anthropic", "Anthropic",
            "claude-haiku-4-5-20251001", 0.80,
        ),
        "openai": _make_provider(
            "openai", "OpenAI",
            "gpt-4o-mini", 0.15,
        ),
        "google": _make_provider(
            "google", "Google AI Studio",
            "gemini/gemini-2.0-flash", 0.10,
            free_tier=MagicMock(),
        ),
        "groq": _make_provider(
            "groq", "Groq",
            "groq/mixtral-8x7b-32768", 0.24,
            free_tier=MagicMock(),
        ),
        "ollama": _make_provider(
            "ollama", "Ollama (Local)",
            "ollama/qwen2.5-coder:7b", 0.00,
        ),
        "deepseek": _make_provider(
            "deepseek", "DeepSeek",
            "deepseek/deepseek-chat", 0.27,
        ),
    }

    def _get_provider(name: str) -> MagicMock | None:
        return configs.get(name)

    registry.get_provider.side_effect = _get_provider
    registry.all_providers = {
        k: v for k, v in configs.items()
    }

    return registry


@pytest.fixture
def mock_backend() -> MockLiteLLM:
    """MockLiteLLM backend for health pings."""
    backend = MockLiteLLM()
    backend.set_default_response(
        MockResponse(content="", output_tokens=1)
    )
    return backend


@pytest.fixture
def checker(
    mock_auth: MagicMock,
    mock_registry: MagicMock,
    mock_backend: MockLiteLLM,
) -> HealthChecker:
    """Standard HealthChecker wired to all mocks."""
    return HealthChecker(
        auth_manager=mock_auth,
        provider_registry=mock_registry,
        litellm_backend=mock_backend,
    )


@pytest.fixture
def checker_no_keys(
    mock_auth_no_keys: MagicMock,
    mock_registry: MagicMock,
    mock_backend: MockLiteLLM,
) -> HealthChecker:
    """HealthChecker with no API keys configured."""
    return HealthChecker(
        auth_manager=mock_auth_no_keys,
        provider_registry=mock_registry,
        litellm_backend=mock_backend,
    )


# ======================================================================
# HealthCheckResult dataclass
# ======================================================================


class TestHealthCheckResult:
    """HealthCheckResult creation and fields."""

    def test_default_fields(self) -> None:
        r = HealthCheckResult(
            provider="test",
            display_name="Test",
            available=True,
            latency_ms=42.5,
            model_used="test-model",
            checked_at=datetime.now(UTC),
        )
        assert r.error is None
        assert r.is_free_tier is False
        assert r.is_local is False

    def test_all_fields(self) -> None:
        ts = datetime.now(UTC)
        r = HealthCheckResult(
            provider="google",
            display_name="Google AI Studio",
            available=True,
            latency_ms=189.3,
            model_used="gemini/gemini-2.0-flash",
            checked_at=ts,
            is_free_tier=True,
            is_local=False,
        )
        assert r.provider == "google"
        assert r.display_name == "Google AI Studio"
        assert r.available is True
        assert r.latency_ms == 189.3
        assert r.model_used == "gemini/gemini-2.0-flash"
        assert r.checked_at == ts
        assert r.is_free_tier is True

    def test_error_fields(self) -> None:
        r = HealthCheckResult(
            provider="deepseek",
            display_name="DeepSeek",
            available=False,
            latency_ms=0.0,
            model_used="deepseek/deepseek-chat",
            error="API key not configured",
            checked_at=datetime.now(UTC),
        )
        assert not r.available
        assert r.error == "API key not configured"

    def test_local_provider(self) -> None:
        r = HealthCheckResult(
            provider="ollama",
            display_name="Ollama (Local)",
            available=True,
            latency_ms=12.0,
            model_used="ollama/qwen2.5-coder:7b",
            is_local=True,
            checked_at=datetime.now(UTC),
        )
        assert r.is_local is True
        assert r.is_free_tier is False


# ======================================================================
# check_all — mixed results
# ======================================================================


class TestCheckAllMixed:
    """Test check_all with mixed pass/fail results."""

    async def test_all_up(
        self,
        checker: HealthChecker,
    ) -> None:
        results = await checker.check_all(
            ["anthropic", "openai"]
        )
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(r.available for r in results)

    async def test_one_down_one_up(
        self,
        checker: HealthChecker,
        mock_backend: MockLiteLLM,
    ) -> None:
        mock_backend.set_error(
            "gpt-4o-mini",
            ConnectionError("Connection refused"),
        )
        results = await checker.check_all(
            ["anthropic", "openai"]
        )
        assert isinstance(results, list)
        available = [r for r in results if r.available]
        unavailable = [r for r in results if not r.available]
        assert len(available) == 1
        assert len(unavailable) == 1
        assert unavailable[0].provider == "openai"
        assert "refused" in (unavailable[0].error or "")

    async def test_all_down(
        self,
        checker: HealthChecker,
        mock_backend: MockLiteLLM,
    ) -> None:
        mock_backend.set_error(
            "claude-haiku-4-5-20251001",
            ConnectionError("down"),
        )
        mock_backend.set_error(
            "gpt-4o-mini",
            ConnectionError("down"),
        )
        results = await checker.check_all(
            ["anthropic", "openai"]
        )
        assert all(not r.available for r in results)

    async def test_defaults_to_all_registered(
        self,
        checker: HealthChecker,
    ) -> None:
        """check_all with no args uses all registered providers."""
        results = await checker.check_all()
        assert isinstance(results, list)
        providers = {r.provider for r in results}
        assert "anthropic" in providers
        assert "openai" in providers

    async def test_results_stored_in_memory(
        self,
        checker: HealthChecker,
    ) -> None:
        """Results are cached in checker._results."""
        await checker.check_all(["anthropic"])
        stored = checker.get_results()
        assert "anthropic" in stored


# ======================================================================
# check_provider — success, timeout, auth, network
# ======================================================================


class TestCheckProviderSuccess:
    """Test check_provider with successful responses."""

    async def test_basic_success(
        self,
        checker: HealthChecker,
    ) -> None:
        result = await checker.check_provider("anthropic")
        assert result.available is True
        assert result.provider == "anthropic"
        assert result.display_name == "Anthropic"
        assert result.latency_ms > 0
        assert result.error is None

    async def test_latency_recorded(
        self,
        checker: HealthChecker,
    ) -> None:
        result = await checker.check_provider("openai")
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms >= 0

    async def test_model_used_set(
        self,
        checker: HealthChecker,
    ) -> None:
        result = await checker.check_provider("openai")
        assert result.model_used == "gpt-4o-mini"

    async def test_checked_at_set(
        self,
        checker: HealthChecker,
    ) -> None:
        before = datetime.now(UTC)
        result = await checker.check_provider("anthropic")
        after = datetime.now(UTC)
        assert before <= result.checked_at <= after

    async def test_free_tier_detected(
        self,
        checker: HealthChecker,
    ) -> None:
        result = await checker.check_provider("google")
        assert result.is_free_tier is True

    async def test_groq_free_tier(
        self,
        checker: HealthChecker,
    ) -> None:
        result = await checker.check_provider("groq")
        assert result.is_free_tier is True


class TestCheckProviderTimeout:
    """Test check_provider with timeout."""

    async def test_timeout_returns_unavailable(
        self,
        mock_auth: MagicMock,
        mock_registry: MagicMock,
    ) -> None:
        backend = MagicMock()

        async def slow_completion(**kwargs: Any) -> None:
            await asyncio.sleep(10)

        backend.acompletion = slow_completion

        chk = HealthChecker(
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=backend,
            timeout_s=0.05,
        )
        result = await chk.check_provider("openai")
        assert result.available is False
        assert "timeout" in (result.error or "").lower()


class TestCheckProviderAuthError:
    """Test check_provider when API key is missing."""

    async def test_no_key_returns_not_configured(
        self,
        checker_no_keys: HealthChecker,
    ) -> None:
        result = await checker_no_keys.check_provider(
            "anthropic"
        )
        assert result.available is False
        assert "not configured" in (result.error or "").lower()

    async def test_no_key_zero_latency(
        self,
        checker_no_keys: HealthChecker,
    ) -> None:
        result = await checker_no_keys.check_provider(
            "openai"
        )
        assert result.latency_ms == 0.0


class TestCheckProviderNetworkError:
    """Test check_provider with network errors."""

    async def test_connection_error(
        self,
        checker: HealthChecker,
        mock_backend: MockLiteLLM,
    ) -> None:
        mock_backend.set_error(
            "claude-haiku-4-5-20251001",
            ConnectionError("Connection refused"),
        )
        result = await checker.check_provider("anthropic")
        assert result.available is False
        assert "refused" in (result.error or "")

    async def test_generic_exception(
        self,
        checker: HealthChecker,
        mock_backend: MockLiteLLM,
    ) -> None:
        mock_backend.set_error(
            "gpt-4o-mini",
            RuntimeError("Internal server error 500"),
        )
        result = await checker.check_provider("openai")
        assert result.available is False
        assert "500" in (result.error or "")

    async def test_os_error(
        self,
        checker: HealthChecker,
        mock_backend: MockLiteLLM,
    ) -> None:
        mock_backend.set_error(
            "claude-haiku-4-5-20251001",
            OSError("Network unreachable"),
        )
        result = await checker.check_provider("anthropic")
        assert result.available is False
        assert "unreachable" in (result.error or "").lower()


# ======================================================================
# format_startup_summary
# ======================================================================


class TestFormatStartupSummary:
    """Test format_startup_summary output format."""

    def test_basic_format(
        self,
        checker: HealthChecker,
    ) -> None:
        results = [
            HealthCheckResult(
                provider="anthropic",
                display_name="Anthropic",
                available=True,
                latency_ms=234.0,
                model_used="claude-haiku-4-5-20251001",
                checked_at=datetime.now(UTC),
            ),
        ]
        output = checker.format_startup_summary(results)
        assert "\u2713" in output
        assert "Anthropic" in output
        assert "234ms" in output

    def test_unavailable_format(
        self,
        checker: HealthChecker,
    ) -> None:
        results = [
            HealthCheckResult(
                provider="deepseek",
                display_name="DeepSeek",
                available=False,
                latency_ms=0.0,
                model_used="deepseek/deepseek-chat",
                error="API key not configured",
                checked_at=datetime.now(UTC),
            ),
        ]
        output = checker.format_startup_summary(results)
        assert "\u2717" in output
        assert "DeepSeek" in output
        assert "API key not configured" in output

    def test_free_tier_tag(
        self,
        checker: HealthChecker,
    ) -> None:
        results = [
            HealthCheckResult(
                provider="groq",
                display_name="Groq",
                available=True,
                latency_ms=89.0,
                model_used="groq/mixtral-8x7b-32768",
                is_free_tier=True,
                checked_at=datetime.now(UTC),
            ),
        ]
        output = checker.format_startup_summary(results)
        assert "[FREE]" in output

    def test_local_tag(
        self,
        checker: HealthChecker,
    ) -> None:
        results = [
            HealthCheckResult(
                provider="ollama",
                display_name="Ollama (Local)",
                available=True,
                latency_ms=12.0,
                model_used="ollama/qwen2.5-coder:7b",
                is_local=True,
                checked_at=datetime.now(UTC),
            ),
        ]
        output = checker.format_startup_summary(results)
        assert "[LOCAL]" in output

    def test_mixed_results_sorted(
        self,
        checker: HealthChecker,
    ) -> None:
        results = [
            HealthCheckResult(
                provider="deepseek",
                display_name="DeepSeek",
                available=False,
                latency_ms=0.0,
                model_used="deepseek/deepseek-chat",
                error="API key not configured",
                checked_at=datetime.now(UTC),
            ),
            HealthCheckResult(
                provider="anthropic",
                display_name="Anthropic",
                available=True,
                latency_ms=234.0,
                model_used="claude-haiku-4-5-20251001",
                checked_at=datetime.now(UTC),
            ),
        ]
        output = checker.format_startup_summary(results)
        lines = output.strip().split("\n")
        # Available providers come first.
        assert "\u2713" in lines[0]
        assert "\u2717" in lines[1]

    def test_model_name_shortened(
        self,
        checker: HealthChecker,
    ) -> None:
        results = [
            HealthCheckResult(
                provider="anthropic",
                display_name="Anthropic",
                available=True,
                latency_ms=234.0,
                model_used="claude-haiku-4-5-20251001",
                checked_at=datetime.now(UTC),
            ),
        ]
        output = checker.format_startup_summary(results)
        assert "claude-haiku-4-5" in output
        # Full datestamp should not appear.
        assert "20251001" not in output

    def test_empty_results(
        self,
        checker: HealthChecker,
    ) -> None:
        output = checker.format_startup_summary([])
        assert output == ""


# ======================================================================
# Background checks start/stop
# ======================================================================


class TestBackgroundChecks:
    """Test start_background_checks and stop_background_checks."""

    async def test_start_creates_thread(
        self,
        checker: HealthChecker,
    ) -> None:
        checker.start_background_checks(interval_seconds=60)
        try:
            assert checker._bg_thread is not None
            assert checker._bg_thread.is_alive()
            assert checker._bg_thread.daemon is True
        finally:
            checker.stop_background_checks()

    async def test_stop_terminates_thread(
        self,
        checker: HealthChecker,
    ) -> None:
        checker.start_background_checks(interval_seconds=60)
        checker.stop_background_checks()
        assert (
            checker._bg_thread is None
            or not checker._bg_thread.is_alive()
        )

    async def test_double_start_is_idempotent(
        self,
        checker: HealthChecker,
    ) -> None:
        checker.start_background_checks(interval_seconds=60)
        thread1 = checker._bg_thread
        checker.start_background_checks(interval_seconds=60)
        thread2 = checker._bg_thread
        assert thread1 is thread2
        checker.stop_background_checks()

    async def test_stop_without_start_is_safe(
        self,
        checker: HealthChecker,
    ) -> None:
        # Should not raise.
        checker.stop_background_checks()


# ======================================================================
# get_available / get_unavailable providers
# ======================================================================


class TestAvailableUnavailableProviders:
    """Test get_available_providers and get_unavailable_providers."""

    async def test_available_after_check(
        self,
        checker: HealthChecker,
    ) -> None:
        await checker.check_all(["anthropic", "openai"])
        avail = checker.get_available_providers()
        assert "anthropic" in avail
        assert "openai" in avail

    async def test_unavailable_after_failure(
        self,
        checker: HealthChecker,
        mock_backend: MockLiteLLM,
    ) -> None:
        mock_backend.set_error(
            "gpt-4o-mini",
            ConnectionError("down"),
        )
        await checker.check_all(
            ["anthropic", "openai"]
        )
        unavail = checker.get_unavailable_providers()
        assert "openai" in unavail
        avail = checker.get_available_providers()
        assert "openai" not in avail

    async def test_empty_before_any_check(
        self,
        checker: HealthChecker,
    ) -> None:
        assert checker.get_available_providers() == []
        assert checker.get_unavailable_providers() == []

    async def test_mixed_results(
        self,
        checker: HealthChecker,
        mock_backend: MockLiteLLM,
    ) -> None:
        mock_backend.set_error(
            "gpt-4o-mini",
            RuntimeError("error"),
        )
        await checker.check_all(
            ["anthropic", "openai"]
        )
        assert "anthropic" in checker.get_available_providers()
        assert "openai" in checker.get_unavailable_providers()


# ======================================================================
# Ollama special handling
# ======================================================================


class TestOllamaHandling:
    """Test Ollama special handling (local HTTP check)."""

    async def test_ollama_success_via_httpx(
        self,
        checker: HealthChecker,
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen2.5-coder:7b"},
                {"name": "llama3.2:3b"},
            ]
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(
            return_value=mock_client
        )
        mock_client.__aexit__ = AsyncMock(
            return_value=False
        )

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client

        with patch(
            "prism.llm.health._httpx_mod", mock_httpx,
        ):
            result = await checker.check_provider("ollama")

        assert result.available is True
        assert result.is_local is True
        assert "qwen2.5-coder:7b" in result.model_used

    async def test_ollama_server_not_running(
        self,
        checker: HealthChecker,
    ) -> None:
        mock_client = AsyncMock()
        mock_client.get.side_effect = ConnectionError(
            "Connection refused"
        )
        mock_client.__aenter__ = AsyncMock(
            return_value=mock_client
        )
        mock_client.__aexit__ = AsyncMock(
            return_value=False
        )

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client

        with patch(
            "prism.llm.health._httpx_mod", mock_httpx,
        ):
            result = await checker.check_provider("ollama")

        assert result.available is False
        assert result.is_local is True
        assert "refused" in (result.error or "").lower()

    async def test_ollama_timeout(
        self,
        mock_auth: MagicMock,
        mock_registry: MagicMock,
        mock_backend: MockLiteLLM,
    ) -> None:
        chk = HealthChecker(
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_backend,
            timeout_s=0.05,
        )

        async def slow_get(*a: Any, **kw: Any) -> None:
            await asyncio.sleep(10)

        mock_client = AsyncMock()
        mock_client.get = slow_get
        mock_client.__aenter__ = AsyncMock(
            return_value=mock_client
        )
        mock_client.__aexit__ = AsyncMock(
            return_value=False
        )

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client

        with patch(
            "prism.llm.health._httpx_mod", mock_httpx,
        ):
            result = await chk.check_provider("ollama")

        assert result.available is False
        assert "timeout" in (result.error or "").lower()

    async def test_ollama_bad_status_code(
        self,
        checker: HealthChecker,
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 503

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(
            return_value=mock_client
        )
        mock_client.__aexit__ = AsyncMock(
            return_value=False
        )

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client

        with patch(
            "prism.llm.health._httpx_mod", mock_httpx,
        ):
            result = await checker.check_provider("ollama")

        assert result.available is False
        assert "503" in (result.error or "")

    async def test_ollama_no_api_key_needed(
        self,
        checker_no_keys: HealthChecker,
    ) -> None:
        """Ollama should never fail due to missing API key."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2:3b"}]
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(
            return_value=mock_client
        )
        mock_client.__aexit__ = AsyncMock(
            return_value=False
        )

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client

        with patch(
            "prism.llm.health._httpx_mod", mock_httpx,
        ):
            result = (
                await checker_no_keys.check_provider(
                    "ollama"
                )
            )

        assert result.available is True
        assert result.is_local is True

    async def test_ollama_httpx_not_installed(
        self,
        checker: HealthChecker,
        mock_backend: MockLiteLLM,
    ) -> None:
        """If httpx not installed, falls back to acompletion."""
        with patch(
            "prism.llm.health._httpx_mod", None,
        ):
            result = await checker.check_provider("ollama")

        # Should still work via acompletion fallback.
        assert isinstance(result, HealthCheckResult)
        assert result.is_local is True
        assert result.available is True


# ======================================================================
# Provider with no API key
# ======================================================================


class TestNoApiKey:
    """Test providers with no API key configured."""

    async def test_anthropic_no_key(
        self,
        checker_no_keys: HealthChecker,
    ) -> None:
        result = await checker_no_keys.check_provider(
            "anthropic"
        )
        assert result.available is False
        assert "not configured" in (result.error or "").lower()

    async def test_openai_no_key(
        self,
        checker_no_keys: HealthChecker,
    ) -> None:
        result = await checker_no_keys.check_provider(
            "openai"
        )
        assert result.available is False

    async def test_deepseek_no_key(
        self,
        checker_no_keys: HealthChecker,
    ) -> None:
        result = await checker_no_keys.check_provider(
            "deepseek"
        )
        assert result.available is False
        assert result.latency_ms == 0.0

    async def test_all_no_keys(
        self,
        checker_no_keys: HealthChecker,
    ) -> None:
        results = await checker_no_keys.check_all(
            ["anthropic", "openai", "deepseek"]
        )
        assert isinstance(results, list)
        assert all(not r.available for r in results)


# ======================================================================
# Thread safety
# ======================================================================


class TestThreadSafety:
    """Test thread-safe access to results."""

    async def test_concurrent_check_all(
        self,
        checker: HealthChecker,
    ) -> None:
        """Multiple concurrent check_all should not corrupt state."""
        tasks = [
            checker.check_all(["anthropic", "openai"])
            for _ in range(5)
        ]
        all_results = await asyncio.gather(*tasks)
        for results in all_results:
            assert isinstance(results, list)
            assert len(results) == 2

    async def test_read_during_write(
        self,
        checker: HealthChecker,
    ) -> None:
        """Reading results while writing should not crash."""
        await checker.check_all(["anthropic"])

        results: list[dict[str, HealthCheckResult]] = []
        errors: list[Exception] = []

        def read_loop() -> None:
            for _ in range(100):
                try:
                    r = checker.get_results()
                    results.append(r)
                except Exception as e:
                    errors.append(e)

        thread = threading.Thread(target=read_loop)
        thread.start()

        # Write concurrently.
        for _ in range(10):
            await checker.check_all(["anthropic"])

        thread.join(timeout=5.0)
        assert len(errors) == 0
        assert len(results) > 0

    async def test_get_available_threadsafe(
        self,
        checker: HealthChecker,
    ) -> None:
        await checker.check_all(["anthropic", "openai"])

        errors: list[Exception] = []

        def read_available() -> None:
            for _ in range(50):
                try:
                    checker.get_available_providers()
                    checker.get_unavailable_providers()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=read_available)
            for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0


# ======================================================================
# No backend configured
# ======================================================================


class TestNoBackend:
    """Test when no litellm_backend is injected."""

    async def test_no_backend_returns_error(
        self,
        mock_auth: MagicMock,
        mock_registry: MagicMock,
    ) -> None:
        chk = HealthChecker(
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=None,
        )
        result = await chk.check_provider("anthropic")
        assert result.available is False
        assert "backend" in (result.error or "").lower()


# ======================================================================
# Legacy API backward compatibility
# ======================================================================


class TestLegacyAllHealthy:
    """All providers healthy (legacy API)."""

    async def test_all_healthy(self) -> None:
        async def ok_check(
            provider: str,
        ) -> list[str]:
            return [
                f"{provider}/model-a",
                f"{provider}/model-b",
            ]

        chk = HealthChecker(check_fn=ok_check)
        statuses = await chk.check_all(["openai", "anthropic"])
        assert isinstance(statuses, dict)
        assert len(statuses) == 2
        assert statuses["openai"].available is True
        assert statuses["anthropic"].available is True

    async def test_models_reported(self) -> None:
        async def ok_check(
            provider: str,
        ) -> list[str]:
            return ["model-1"]

        chk = HealthChecker(check_fn=ok_check)
        statuses = await chk.check_all(["openai"])
        assert statuses["openai"].models_available == [
            "model-1"
        ]

    async def test_latency_recorded(self) -> None:
        async def ok_check(
            provider: str,
        ) -> list[str]:
            return []

        chk = HealthChecker(check_fn=ok_check)
        statuses = await chk.check_all(["openai"])
        assert statuses["openai"].latency_ms is not None
        assert statuses["openai"].latency_ms >= 0


class TestLegacyOneDown:
    """Mixed availability (legacy API)."""

    async def test_one_down(self) -> None:
        async def mixed_check(
            provider: str,
        ) -> list[str]:
            if provider == "openai":
                raise ConnectionError("refused")
            return ["model-1"]

        chk = HealthChecker(check_fn=mixed_check)
        statuses = await chk.check_all(
            ["openai", "anthropic"]
        )
        assert statuses["openai"].available is False
        assert "refused" in (statuses["openai"].error or "")
        assert statuses["anthropic"].available is True

    async def test_error_message_captured(self) -> None:
        async def fail_check(
            provider: str,
        ) -> list[str]:
            raise RuntimeError("server error 500")

        chk = HealthChecker(check_fn=fail_check)
        statuses = await chk.check_all(["openai"])
        assert (
            statuses["openai"].error == "server error 500"
        )


class TestLegacyTimeout:
    """Timeout during legacy health check."""

    async def test_timeout(self) -> None:
        async def slow_check(
            provider: str,
        ) -> list[str]:
            await asyncio.sleep(10)
            return []

        chk = HealthChecker(
            check_fn=slow_check, timeout_s=0.05,
        )
        statuses = await chk.check_all(["openai"])
        assert statuses["openai"].available is False
        assert "timed out" in (
            statuses["openai"].error or ""
        ).lower()


class TestLegacyConcurrent:
    """Checks run concurrently (legacy API)."""

    async def test_concurrent_execution(self) -> None:
        async def timed_check(
            provider: str,
        ) -> list[str]:
            await asyncio.sleep(0.05)
            return ["model-1"]

        chk = HealthChecker(
            check_fn=timed_check, timeout_s=5.0,
        )
        start = time.perf_counter()
        statuses = await chk.check_all(["a", "b", "c"])
        elapsed = time.perf_counter() - start

        assert len(statuses) == 3
        assert elapsed < 0.15


class TestLegacyNoCheckFn:
    """No check_fn → all assumed healthy (legacy API)."""

    async def test_no_check_fn_all_available(self) -> None:
        chk = HealthChecker(check_fn=None)
        statuses = await chk.check_all(
            ["openai", "anthropic"]
        )
        for status in statuses.values():
            assert status.available is True

    async def test_check_one_no_fn(self) -> None:
        chk = HealthChecker(check_fn=None)
        status = await chk.check_one("openai")
        assert status.available is True
        assert status.error is None


# ======================================================================
# HealthStatus dataclass (legacy)
# ======================================================================


class TestHealthStatusDataclass:
    """HealthStatus creation and fields."""

    def test_default_fields(self) -> None:
        status = HealthStatus(
            provider="test", available=True,
        )
        assert status.latency_ms is None
        assert status.error is None
        assert status.models_available == []

    def test_full_fields(self) -> None:
        status = HealthStatus(
            provider="openai",
            available=False,
            latency_ms=42.5,
            error="503",
            models_available=["gpt-4o"],
        )
        assert status.provider == "openai"
        assert not status.available
        assert status.latency_ms == 42.5
        assert status.error == "503"
        assert status.models_available == ["gpt-4o"]


# ======================================================================
# ProviderDashboardEntry dataclass
# ======================================================================


class TestProviderDashboardEntry:
    """ProviderDashboardEntry creation and fields."""

    def test_default_fields(self) -> None:
        entry = ProviderDashboardEntry(
            provider="test",
            display_name="Test",
            status="healthy",
        )
        assert entry.latency_ms is None
        assert entry.models_count == 0
        assert entry.error is None
        assert entry.has_api_key is False
        assert entry.free_tier is False
        assert entry.free_tier_remaining is None

    def test_all_fields(self) -> None:
        entry = ProviderDashboardEntry(
            provider="google",
            display_name="Google AI Studio",
            status="healthy",
            latency_ms=189.3,
            models_count=3,
            has_api_key=True,
            free_tier=True,
            free_tier_remaining=1450,
        )
        assert entry.free_tier_remaining == 1450


# ======================================================================
# generate_dashboard (legacy)
# ======================================================================


class TestGenerateDashboard:
    """Test generate_dashboard backward compatibility."""

    async def test_dashboard_basic(self) -> None:
        async def check_fn(
            provider: str,
        ) -> list[str]:
            return ["model-a", "model-b"]

        chk = HealthChecker(check_fn=check_fn)
        entries = await chk.generate_dashboard(
            ["openai", "anthropic"]
        )
        assert len(entries) == 2
        for entry in entries:
            assert isinstance(entry, ProviderDashboardEntry)
            assert entry.status == "healthy"

    async def test_dashboard_down_provider(self) -> None:
        async def check_fn(
            provider: str,
        ) -> list[str]:
            if provider == "openai":
                raise ConnectionError("down")
            return ["model-a"]

        chk = HealthChecker(check_fn=check_fn)
        entries = await chk.generate_dashboard(
            ["openai", "anthropic"]
        )
        openai_entry = next(
            e for e in entries if e.provider == "openai"
        )
        assert openai_entry.status == "down"
        assert openai_entry.error is not None


# ======================================================================
# generate_status_table
# ======================================================================


class TestGenerateStatusTable:
    """Test generate_status_table returns a Rich Table."""

    def test_returns_table(
        self,
        checker: HealthChecker,
    ) -> None:
        from rich.table import Table

        results = [
            HealthCheckResult(
                provider="anthropic",
                display_name="Anthropic",
                available=True,
                latency_ms=200.0,
                model_used="claude-haiku-4-5-20251001",
                checked_at=datetime.now(UTC),
            ),
        ]
        table = checker.generate_status_table(results)
        assert isinstance(table, Table)

    def test_uses_stored_results_if_none(
        self,
        checker: HealthChecker,
    ) -> None:
        """generate_status_table(None) uses stored results."""
        from rich.table import Table

        # Pre-populate results.
        checker._results["anthropic"] = HealthCheckResult(
            provider="anthropic",
            display_name="Anthropic",
            available=True,
            latency_ms=200.0,
            model_used="claude-haiku-4-5-20251001",
            checked_at=datetime.now(UTC),
        )
        table = checker.generate_status_table()
        assert isinstance(table, Table)

    def test_empty_results(
        self,
        checker: HealthChecker,
    ) -> None:
        from rich.table import Table

        table = checker.generate_status_table([])
        assert isinstance(table, Table)


# ======================================================================
# Short model name helper
# ======================================================================


class TestShortModelName:
    """Test _short_model_name static method."""

    def test_strips_provider_prefix(self) -> None:
        assert (
            HealthChecker._short_model_name(
                "groq/mixtral-8x7b-32768"
            )
            == "mixtral-8x7b"
        )

    def test_shortens_claude_haiku(self) -> None:
        assert (
            HealthChecker._short_model_name(
                "claude-haiku-4-5-20251001"
            )
            == "claude-haiku-4-5"
        )

    def test_unknown_model_passthrough(self) -> None:
        assert (
            HealthChecker._short_model_name(
                "some/new-model"
            )
            == "new-model"
        )

    def test_no_prefix(self) -> None:
        assert (
            HealthChecker._short_model_name("gpt-4o")
            == "gpt-4o"
        )


# ======================================================================
# _CHEAPEST_MODEL mapping
# ======================================================================


class TestCheapestModelMapping:
    """Verify _CHEAPEST_MODEL has expected entries."""

    def test_has_anthropic(self) -> None:
        assert "anthropic" in _CHEAPEST_MODEL

    def test_has_openai(self) -> None:
        assert "openai" in _CHEAPEST_MODEL

    def test_has_google(self) -> None:
        assert "google" in _CHEAPEST_MODEL

    def test_has_ollama(self) -> None:
        assert "ollama" in _CHEAPEST_MODEL

    def test_has_groq(self) -> None:
        assert "groq" in _CHEAPEST_MODEL


# ======================================================================
# get_results snapshot
# ======================================================================


class TestGetResults:
    """Test get_results returns a copy."""

    async def test_returns_dict(
        self,
        checker: HealthChecker,
    ) -> None:
        await checker.check_all(["anthropic"])
        results = checker.get_results()
        assert isinstance(results, dict)
        assert "anthropic" in results

    async def test_returns_copy(
        self,
        checker: HealthChecker,
    ) -> None:
        """Modifying the returned dict should not affect state."""
        await checker.check_all(["anthropic"])
        results = checker.get_results()
        results.clear()
        assert len(checker.get_results()) > 0
