"""Tests for cost estimation and pricing."""

from __future__ import annotations

import pytest

from prism.cost.pricing import (
    MODEL_PRICING,
    calculate_cost,
    estimate_input_tokens,
    estimate_output_tokens,
    get_model_pricing,
    get_provider_for_model,
)


class TestGetModelPricing:
    def test_known_model(self) -> None:
        pricing = get_model_pricing("gpt-4o")
        assert pricing.provider == "openai"
        assert pricing.input_cost_per_1m == 2.50
        assert pricing.output_cost_per_1m == 10.00

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_pricing("nonexistent-model-xyz")

    def test_ollama_is_free(self) -> None:
        pricing = get_model_pricing("ollama/qwen2.5-coder:7b")
        assert pricing.input_cost_per_1m == 0.00
        assert pricing.output_cost_per_1m == 0.00

    def test_all_models_have_provider(self) -> None:
        for model_id, pricing in MODEL_PRICING.items():
            assert pricing.provider, f"Model {model_id} has no provider"

    def test_all_costs_non_negative(self) -> None:
        for model_id, pricing in MODEL_PRICING.items():
            assert pricing.input_cost_per_1m >= 0, f"{model_id} has negative input cost"
            assert pricing.output_cost_per_1m >= 0, f"{model_id} has negative output cost"


class TestCalculateCost:
    def test_zero_tokens_zero_cost(self) -> None:
        cost = calculate_cost("gpt-4o", 0, 0)
        assert cost == 0.0

    def test_basic_cost_calculation(self) -> None:
        # GPT-4o: $2.50 input, $10.00 output per 1M tokens
        cost = calculate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == pytest.approx(12.50, abs=0.01)

    def test_small_request_cost(self) -> None:
        # 500 input + 200 output tokens with GPT-4o
        cost = calculate_cost("gpt-4o", 500, 200)
        expected = (500 / 1_000_000) * 2.50 + (200 / 1_000_000) * 10.00
        assert cost == pytest.approx(expected, abs=0.0001)

    def test_cached_tokens_discounted(self) -> None:
        # 1000 input tokens, 500 cached, 500 output
        full_cost = calculate_cost("gpt-4o", 1000, 500, cached_tokens=0)
        cached_cost = calculate_cost("gpt-4o", 1000, 500, cached_tokens=500)
        assert cached_cost < full_cost

    def test_ollama_always_free(self) -> None:
        cost = calculate_cost("ollama/qwen2.5-coder:7b", 10000, 5000)
        assert cost == 0.0

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError):
            calculate_cost("nonexistent-model", 100, 100)

    def test_cost_never_negative(self) -> None:
        for model_id in MODEL_PRICING:
            cost = calculate_cost(model_id, 100, 100)
            assert cost >= 0.0, f"Negative cost for {model_id}"

    def test_deepseek_cache_discount(self) -> None:
        # DeepSeek has very aggressive cache discount
        full = calculate_cost("deepseek/deepseek-chat", 10000, 1000, cached_tokens=0)
        cached = calculate_cost("deepseek/deepseek-chat", 10000, 1000, cached_tokens=8000)
        assert cached < full


class TestEstimateInputTokens:
    def test_empty_string(self) -> None:
        assert estimate_input_tokens("") == 0

    def test_single_word(self) -> None:
        tokens = estimate_input_tokens("hello")
        assert tokens >= 1

    def test_code_snippet(self) -> None:
        code = "def calculate_cost(model_id: str, input_tokens: int) -> float:\n    return 0.0"
        tokens = estimate_input_tokens(code)
        assert tokens > 10

    def test_longer_text_more_tokens(self) -> None:
        short = estimate_input_tokens("fix typo")
        long = estimate_input_tokens("refactor the entire authentication module to use async")
        assert long > short

    def test_never_returns_zero_for_non_empty(self) -> None:
        assert estimate_input_tokens("x") >= 1


class TestEstimateOutputTokens:
    def test_explain_task(self) -> None:
        tokens = estimate_output_tokens("explain what this function does")
        assert tokens == 300

    def test_test_generation(self) -> None:
        tokens = estimate_output_tokens("write test cases for this module")
        assert tokens == 1500

    def test_architecture_task(self) -> None:
        tokens = estimate_output_tokens("design a new architecture")
        assert tokens == 3000

    def test_default_for_unknown(self) -> None:
        tokens = estimate_output_tokens("do something interesting and unique")
        assert tokens == 1000

    def test_empty_prompt(self) -> None:
        tokens = estimate_output_tokens("")
        assert tokens == 1000  # Default


class TestGetProviderForModel:
    def test_known_model(self) -> None:
        assert get_provider_for_model("gpt-4o") == "openai"
        assert get_provider_for_model("claude-sonnet-4-20250514") == "anthropic"
        assert get_provider_for_model("deepseek/deepseek-chat") == "deepseek"
        assert get_provider_for_model("ollama/qwen2.5-coder:7b") == "ollama"

    def test_unknown_model_with_prefix(self) -> None:
        assert get_provider_for_model("groq/some-new-model") == "groq"

    def test_claude_prefix_inference(self) -> None:
        assert get_provider_for_model("claude-some-new-model") == "anthropic"

    def test_gpt_prefix_inference(self) -> None:
        assert get_provider_for_model("gpt-5-turbo") == "openai"

    def test_completely_unknown(self) -> None:
        result = get_provider_for_model("totally-unknown-model")
        assert result == "unknown"
