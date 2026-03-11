"""Tests for Phase 3 LiteLLM live integration enhancements.

All tests use MockLiteLLM — absolutely no real API calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from prism.cost.pricing import get_provider_for_model
from prism.exceptions import (
    AllProvidersFailedError,
    BudgetExceededError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from prism.llm.completion import (
    _DEFAULT_TIMEOUT,
    PROVIDER_TIMEOUTS,
    CompletionEngine,
)
from prism.llm.health import HealthChecker, ProviderDashboardEntry
from prism.llm.mock import MockLiteLLM, MockResponse
from prism.llm.provider_config import (
    EXTENDED_PRICING,
    KIMI_CONFIG,
    PERPLEXITY_CONFIG,
    register_extended_providers,
)

if TYPE_CHECKING:
    from prism.providers.registry import ProviderRegistry


@pytest.fixture(autouse=True)
def _ensure_extended_registered() -> None:
    """Ensure extended providers are registered for all tests in this module.

    ``register_extended_providers`` is idempotent, so calling it repeatedly
    is harmless but guarantees that MODEL_PRICING has the moonshot / perplexity
    entries before any test runs.
    """
    register_extended_providers()


# ======================================================================
# Provider Config Updates
# ======================================================================


class TestKimiModels:
    """Test Kimi/Moonshot provider has all expected models."""

    def test_kimi_has_three_models(self) -> None:
        assert len(KIMI_CONFIG.models) == 3

    def test_moonshot_128k_exists(self) -> None:
        ids = {m.id for m in KIMI_CONFIG.models}
        assert "moonshot/moonshot-v1-128k" in ids

    def test_moonshot_128k_context_window(self) -> None:
        model = next(m for m in KIMI_CONFIG.models if m.id == "moonshot/moonshot-v1-128k")
        assert model.context_window == 128_000

    def test_moonshot_128k_pricing(self) -> None:
        assert "moonshot/moonshot-v1-128k" in EXTENDED_PRICING
        pricing = EXTENDED_PRICING["moonshot/moonshot-v1-128k"]
        assert pricing.input_cost_per_1m == 0.48
        assert pricing.output_cost_per_1m == 0.48


class TestPerplexityModels:
    """Test Perplexity provider updated model IDs."""

    def test_perplexity_has_two_models(self) -> None:
        assert len(PERPLEXITY_CONFIG.models) == 2

    def test_sonar_large_exists(self) -> None:
        ids = {m.id for m in PERPLEXITY_CONFIG.models}
        assert "perplexity/llama-3.1-sonar-large-128k-online" in ids

    def test_sonar_small_exists(self) -> None:
        ids = {m.id for m in PERPLEXITY_CONFIG.models}
        assert "perplexity/llama-3.1-sonar-small-128k-online" in ids

    def test_old_models_removed_from_pricing(self) -> None:
        assert "perplexity/pplx-70b-online" not in EXTENDED_PRICING
        assert "perplexity/pplx-sonar-medium" not in EXTENDED_PRICING

    def test_new_models_in_pricing(self) -> None:
        assert "perplexity/llama-3.1-sonar-large-128k-online" in EXTENDED_PRICING
        assert "perplexity/llama-3.1-sonar-small-128k-online" in EXTENDED_PRICING


# ======================================================================
# Provider Identification
# ======================================================================


class TestProviderIdentification:
    """Test get_provider_for_model recognizes new prefixes."""

    def test_moonshot_returns_kimi(self) -> None:
        # After register_extended_providers(), the pricing table contains
        # moonshot/moonshot-v1-128k with provider="kimi", so the pricing-based
        # lookup returns "kimi".
        assert get_provider_for_model("moonshot/moonshot-v1-128k") == "kimi"

    def test_perplexity_prefix(self) -> None:
        result = get_provider_for_model("perplexity/llama-3.1-sonar-large-128k-online")
        assert result == "perplexity"

    def test_cohere_prefix(self) -> None:
        result = get_provider_for_model("cohere/command-r-plus")
        assert result == "cohere"

    def test_qwen_prefix(self) -> None:
        result = get_provider_for_model("qwen/qwen-max")
        assert result == "qwen"

    def test_together_ai_prefix(self) -> None:
        result = get_provider_for_model("together_ai/meta-llama/Llama-3-70b-chat-hf")
        assert result == "together_ai"

    def test_fireworks_ai_prefix(self) -> None:
        result = get_provider_for_model("fireworks_ai/llama-v3p1-70b-instruct")
        assert result == "fireworks_ai"


# ======================================================================
# Provider Timeouts
# ======================================================================


class TestProviderTimeouts:
    """Test per-provider timeout configuration."""

    def test_anthropic_timeout(self) -> None:
        assert PROVIDER_TIMEOUTS["anthropic"] == 120.0

    def test_groq_fastest(self) -> None:
        assert PROVIDER_TIMEOUTS["groq"] == 30.0

    def test_ollama_slowest(self) -> None:
        assert PROVIDER_TIMEOUTS["ollama"] == 300.0

    def test_default_timeout(self) -> None:
        assert _DEFAULT_TIMEOUT == 60.0

    def test_get_provider_timeout_known(self) -> None:
        assert CompletionEngine.get_provider_timeout("anthropic") == 120.0

    def test_get_provider_timeout_unknown(self) -> None:
        assert CompletionEngine.get_provider_timeout("unknown_provider") == 60.0

    def test_timeout_injected_in_kwargs(
        self,
        completion_engine: CompletionEngine,
    ) -> None:
        """Verify timeout is set in kwargs built for LiteLLM."""
        kwargs = completion_engine._build_litellm_kwargs(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "timeout" in kwargs
        assert kwargs["timeout"] == PROVIDER_TIMEOUTS["openai"]


# ======================================================================
# Prompt Caching
# ======================================================================


class TestPromptCaching:
    """Test Anthropic prompt caching headers."""

    def test_caching_applied_to_anthropic(
        self,
        completion_engine: CompletionEngine,
    ) -> None:
        """System messages get cache_control for Anthropic models."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "A" * 200},
            {"role": "user", "content": "B" * 300},
        ]
        result = completion_engine._apply_prompt_caching(
            "claude-sonnet-4-20250514", messages,
        )
        # System message should be converted to list format with cache_control
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"] == {"type": "ephemeral"}

    def test_caching_not_applied_to_openai(
        self,
        completion_engine: CompletionEngine,
    ) -> None:
        """Non-Anthropic models should not get cache_control."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "A" * 200},
            {"role": "user", "content": "B" * 300},
        ]
        result = completion_engine._apply_prompt_caching("gpt-4o", messages)
        # Should return messages unchanged
        assert result[0]["content"] == "A" * 200

    def test_short_system_not_cached(
        self,
        completion_engine: CompletionEngine,
    ) -> None:
        """Short system messages (<=100 chars) should not be cached."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "B" * 300},
        ]
        result = completion_engine._apply_prompt_caching(
            "claude-sonnet-4-20250514", messages,
        )
        assert isinstance(result[0]["content"], str)

    def test_short_user_not_cached(
        self,
        completion_engine: CompletionEngine,
    ) -> None:
        """Short user messages (<=200 chars) should not have cache_control."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "A" * 200},
            {"role": "user", "content": "short"},
        ]
        result = completion_engine._apply_prompt_caching(
            "claude-sonnet-4-20250514", messages,
        )
        # User message should remain a string
        assert isinstance(result[1]["content"], str)

    def test_last_user_message_cached(
        self,
        completion_engine: CompletionEngine,
    ) -> None:
        """Only the last user message gets cache_control."""
        messages: list[dict[str, str]] = [
            {"role": "user", "content": "X" * 300},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "Y" * 300},
        ]
        result = completion_engine._apply_prompt_caching(
            "claude-sonnet-4-20250514", messages,
        )
        # First user message should NOT be cached
        assert isinstance(result[0]["content"], str)
        # Last user message should be cached
        assert isinstance(result[2]["content"], list)

    def test_caching_does_not_mutate_original(
        self,
        completion_engine: CompletionEngine,
    ) -> None:
        """Prompt caching should not mutate the input messages."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "A" * 200},
            {"role": "user", "content": "B" * 300},
        ]
        original_sys = messages[0]["content"]
        completion_engine._apply_prompt_caching(
            "claude-sonnet-4-20250514", messages,
        )
        assert messages[0]["content"] == original_sys


# ======================================================================
# Provider Error Handling
# ======================================================================


class TestProviderErrorHandling:
    """Test _handle_provider_error updates registry status correctly."""

    def test_rate_limit_marks_provider(
        self,
        completion_engine: CompletionEngine,
        mock_registry: ProviderRegistry,
    ) -> None:
        exc = ProviderRateLimitError("openai", retry_after=30.0)
        completion_engine._handle_provider_error(exc, "gpt-4o", "openai")
        status = mock_registry.get_status("openai")
        assert status is not None
        assert status.is_rate_limited

    def test_auth_error_marks_unavailable(
        self,
        completion_engine: CompletionEngine,
        mock_registry: ProviderRegistry,
    ) -> None:
        exc = ProviderAuthError("openai")
        completion_engine._handle_provider_error(exc, "gpt-4o", "openai")
        status = mock_registry.get_status("openai")
        assert status is not None
        assert not status.is_available

    def test_unavailable_marks_down(
        self,
        completion_engine: CompletionEngine,
        mock_registry: ProviderRegistry,
    ) -> None:
        exc = ProviderUnavailableError("openai", "503 Service Unavailable")
        completion_engine._handle_provider_error(exc, "gpt-4o", "openai")
        status = mock_registry.get_status("openai")
        assert status is not None
        assert not status.is_available

    def test_connection_error_marks_unavailable(
        self,
        completion_engine: CompletionEngine,
        mock_registry: ProviderRegistry,
    ) -> None:
        exc = ConnectionError("Connection refused")
        completion_engine._handle_provider_error(exc, "gpt-4o", "openai")
        status = mock_registry.get_status("openai")
        assert status is not None
        assert not status.is_available

    def test_unknown_provider_no_crash(
        self,
        completion_engine: CompletionEngine,
    ) -> None:
        """Error handling for unknown provider should not crash."""
        exc = TimeoutError("timed out")
        # Should not raise
        completion_engine._handle_provider_error(exc, "unknown-model", "nonexistent")


# ======================================================================
# Fallback Chain
# ======================================================================


class TestFallbackChain:
    """Test complete_with_fallback tries models in order."""

    async def test_first_model_succeeds(
        self,
        completion_engine: CompletionEngine,
        mock_litellm: MockLiteLLM,
    ) -> None:
        mock_litellm.set_response("gpt-4o", MockResponse(content="success"))
        result = await completion_engine.complete_with_fallback(
            messages=[{"role": "user", "content": "test"}],
            models=["gpt-4o", "gpt-4o-mini"],
            session_id="s1",
        )
        assert result.content == "success"

    async def test_fallback_to_second_model(
        self,
        completion_engine: CompletionEngine,
        mock_litellm: MockLiteLLM,
    ) -> None:
        mock_litellm.set_error("gpt-4o", ProviderUnavailableError("openai"))
        mock_litellm.set_response("gpt-4o-mini", MockResponse(content="fallback"))
        result = await completion_engine.complete_with_fallback(
            messages=[{"role": "user", "content": "test"}],
            models=["gpt-4o", "gpt-4o-mini"],
            session_id="s1",
        )
        assert result.content == "fallback"

    async def test_all_fail_raises(
        self,
        completion_engine: CompletionEngine,
        mock_litellm: MockLiteLLM,
    ) -> None:
        mock_litellm.set_error("gpt-4o", ProviderUnavailableError("openai"))
        mock_litellm.set_error("gpt-4o-mini", ProviderUnavailableError("openai"))
        with pytest.raises(AllProvidersFailedError) as exc_info:
            await completion_engine.complete_with_fallback(
                messages=[{"role": "user", "content": "test"}],
                models=["gpt-4o", "gpt-4o-mini"],
                session_id="s1",
            )
        assert "gpt-4o" in exc_info.value.tried_models
        assert "gpt-4o-mini" in exc_info.value.tried_models

    async def test_budget_error_not_caught_by_fallback(
        self,
        mock_litellm: MockLiteLLM,
        mock_cost_tracker_blocked: MagicMock,
        llm_settings,
        mock_auth,
        mock_registry,
    ) -> None:
        """BudgetExceededError should propagate, not trigger fallback."""
        engine = CompletionEngine(
            settings=llm_settings,
            cost_tracker=mock_cost_tracker_blocked,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm,
        )
        mock_litellm.set_response("gpt-4o", MockResponse(content="ok"))
        with pytest.raises(BudgetExceededError):
            await engine.complete_with_fallback(
                messages=[{"role": "user", "content": "test"}],
                models=["gpt-4o", "gpt-4o-mini"],
                session_id="s1",
            )


# ======================================================================
# Parallel Completion
# ======================================================================


class TestParallelCompletion:
    """Test complete_parallel runs models concurrently."""

    async def test_parallel_returns_all_results(
        self,
        completion_engine: CompletionEngine,
        mock_litellm: MockLiteLLM,
    ) -> None:
        mock_litellm.set_response("gpt-4o", MockResponse(content="openai"))
        mock_litellm.set_response("gpt-4o-mini", MockResponse(content="mini"))
        results = await completion_engine.complete_parallel(
            messages=[{"role": "user", "content": "test"}],
            models=["gpt-4o", "gpt-4o-mini"],
            session_id="s1",
        )
        assert len(results) == 2
        contents = {r.content for r in results}
        assert "openai" in contents
        assert "mini" in contents

    async def test_parallel_excludes_failures(
        self,
        completion_engine: CompletionEngine,
        mock_litellm: MockLiteLLM,
    ) -> None:
        mock_litellm.set_response("gpt-4o", MockResponse(content="ok"))
        mock_litellm.set_error("gpt-4o-mini", ProviderUnavailableError("openai"))
        results = await completion_engine.complete_parallel(
            messages=[{"role": "user", "content": "test"}],
            models=["gpt-4o", "gpt-4o-mini"],
            session_id="s1",
        )
        assert len(results) == 1
        assert results[0].content == "ok"

    async def test_parallel_all_fail_returns_empty(
        self,
        completion_engine: CompletionEngine,
        mock_litellm: MockLiteLLM,
    ) -> None:
        mock_litellm.set_error("gpt-4o", ProviderUnavailableError("openai"))
        mock_litellm.set_error("gpt-4o-mini", ProviderUnavailableError("openai"))
        results = await completion_engine.complete_parallel(
            messages=[{"role": "user", "content": "test"}],
            models=["gpt-4o", "gpt-4o-mini"],
            session_id="s1",
        )
        assert len(results) == 0


# ======================================================================
# Provider Dashboard
# ======================================================================


class TestProviderDashboard:
    """Test HealthChecker.generate_dashboard."""

    async def test_dashboard_basic(self) -> None:
        async def check_fn(provider: str) -> list[str]:
            return ["model-a", "model-b"]

        checker = HealthChecker(check_fn=check_fn)
        entries = await checker.generate_dashboard(["openai", "anthropic"])
        assert len(entries) == 2
        for entry in entries:
            assert isinstance(entry, ProviderDashboardEntry)
            assert entry.status == "healthy"

    async def test_dashboard_down_provider(self) -> None:
        async def check_fn(provider: str) -> list[str]:
            if provider == "openai":
                raise ConnectionError("down")
            return ["model-a"]

        checker = HealthChecker(check_fn=check_fn)
        entries = await checker.generate_dashboard(["openai", "anthropic"])
        openai_entry = next(e for e in entries if e.provider == "openai")
        assert openai_entry.status == "down"
        assert openai_entry.error is not None
        anthropic_entry = next(e for e in entries if e.provider == "anthropic")
        assert anthropic_entry.status == "healthy"

    async def test_dashboard_with_registry(
        self,
        mock_registry: ProviderRegistry,
    ) -> None:
        checker = HealthChecker(check_fn=None)  # All assumed healthy
        entries = await checker.generate_dashboard(
            ["openai", "ollama"],
            registry=mock_registry,
        )
        assert len(entries) == 2
        openai_entry = next(e for e in entries if e.provider == "openai")
        assert openai_entry.display_name == "OpenAI"
        assert openai_entry.models_count > 0

    async def test_dashboard_unconfigured_provider(self) -> None:
        """Provider without API key should show unconfigured."""
        registry_mock = MagicMock()
        provider_cfg = MagicMock()
        provider_cfg.display_name = "Test Provider"
        provider_cfg.models = []
        provider_cfg.free_tier = None
        registry_mock.get_provider.return_value = provider_cfg
        registry_mock.is_provider_available.return_value = False
        registry_mock.get_free_tier_remaining.return_value = None

        checker = HealthChecker(check_fn=None)
        entries = await checker.generate_dashboard(
            ["testprov"],
            registry=registry_mock,
        )
        assert entries[0].status == "unconfigured"


# ======================================================================
# Context Window Verification
# ======================================================================


class TestContextWindows:
    """Verify context windows match the Phase 3 spec."""

    def test_claude_200k(self, mock_registry: ProviderRegistry) -> None:
        for model_id in [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-haiku-4-5-20251001",
        ]:
            info = mock_registry.get_model_info(model_id)
            assert info is not None, f"{model_id} not found"
            assert info.context_window == 200_000, f"{model_id}: {info.context_window}"

    def test_gemini_1m(self, mock_registry: ProviderRegistry) -> None:
        for model_id in ["gemini/gemini-2.5-pro", "gemini/gemini-2.0-flash"]:
            info = mock_registry.get_model_info(model_id)
            assert info is not None, f"{model_id} not found"
            assert info.context_window == 1_000_000, f"{model_id}: {info.context_window}"

    def test_gpt4o_128k(self, mock_registry: ProviderRegistry) -> None:
        for model_id in ["gpt-4o", "gpt-4o-mini"]:
            info = mock_registry.get_model_info(model_id)
            assert info is not None
            assert info.context_window == 128_000

    def test_o3_o4_200k(self, mock_registry: ProviderRegistry) -> None:
        for model_id in ["o3", "o4-mini"]:
            info = mock_registry.get_model_info(model_id)
            assert info is not None
            assert info.context_window == 200_000

    def test_deepseek_64k(self, mock_registry: ProviderRegistry) -> None:
        for model_id in ["deepseek/deepseek-chat", "deepseek/deepseek-reasoner"]:
            info = mock_registry.get_model_info(model_id)
            assert info is not None
            assert info.context_window == 64_000

    def test_groq_128k(self, mock_registry: ProviderRegistry) -> None:
        info = mock_registry.get_model_info("groq/llama-3.3-70b-versatile")
        assert info is not None
        assert info.context_window == 128_000

    def test_mistral_128k(self, mock_registry: ProviderRegistry) -> None:
        info = mock_registry.get_model_info("mistral/mistral-small-latest")
        assert info is not None
        assert info.context_window == 128_000
