"""Model pricing data and cost calculation functions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Pricing information for a single model."""

    provider: str
    input_cost_per_1m: float  # USD per 1M input tokens
    output_cost_per_1m: float  # USD per 1M output tokens
    cache_discount: float = 0.0  # Fraction of input price for cached tokens (0-1)


# Cache discount rates by provider
CACHE_DISCOUNTS: dict[str, float] = {
    "anthropic": 0.10,  # 10% of input price for cached tokens
    "openai": 0.50,  # 50%
    "google": 0.25,  # 25%
    "deepseek": 0.07,  # ~7% (cache hit at $0.07/1M vs $0.27/1M)
}


MODEL_PRICING: dict[str, ModelPricing] = {
    # Tier 1: Premium Cloud Models
    "claude-sonnet-4-20250514": ModelPricing(
        provider="anthropic",
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        cache_discount=0.10,
    ),
    "claude-opus-4-20250514": ModelPricing(
        provider="anthropic",
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
        cache_discount=0.10,
    ),
    "claude-haiku-4-5-20251001": ModelPricing(
        provider="anthropic",
        input_cost_per_1m=0.80,
        output_cost_per_1m=4.00,
        cache_discount=0.10,
    ),
    "gpt-4o": ModelPricing(
        provider="openai",
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
        cache_discount=0.50,
    ),
    "gpt-4o-mini": ModelPricing(
        provider="openai",
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
        cache_discount=0.50,
    ),
    "o3": ModelPricing(
        provider="openai",
        input_cost_per_1m=10.00,
        output_cost_per_1m=40.00,
        cache_discount=0.50,
    ),
    "o4-mini": ModelPricing(
        provider="openai",
        input_cost_per_1m=1.10,
        output_cost_per_1m=4.40,
        cache_discount=0.50,
    ),
    "gemini/gemini-2.5-pro": ModelPricing(
        provider="google",
        input_cost_per_1m=1.25,
        output_cost_per_1m=5.00,
        cache_discount=0.25,
    ),
    "gemini/gemini-2.0-flash": ModelPricing(
        provider="google",
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.40,
        cache_discount=0.25,
    ),
    # Tier 2: Cost-Efficient Cloud Models
    "deepseek/deepseek-chat": ModelPricing(
        provider="deepseek",
        input_cost_per_1m=0.27,
        output_cost_per_1m=1.10,
        cache_discount=0.07,
    ),
    "deepseek/deepseek-reasoner": ModelPricing(
        provider="deepseek",
        input_cost_per_1m=0.55,
        output_cost_per_1m=2.19,
        cache_discount=0.07,
    ),
    "groq/llama-3.3-70b-versatile": ModelPricing(
        provider="groq",
        input_cost_per_1m=0.59,
        output_cost_per_1m=0.79,
    ),
    "groq/mixtral-8x7b-32768": ModelPricing(
        provider="groq",
        input_cost_per_1m=0.24,
        output_cost_per_1m=0.24,
    ),
    "mistral/mistral-small-latest": ModelPricing(
        provider="mistral",
        input_cost_per_1m=0.20,
        output_cost_per_1m=0.60,
    ),
    "mistral/codestral-latest": ModelPricing(
        provider="mistral",
        input_cost_per_1m=0.30,
        output_cost_per_1m=0.90,
    ),
    # Tier 3: Free / Local
    "ollama/qwen2.5-coder:7b": ModelPricing(
        provider="ollama",
        input_cost_per_1m=0.00,
        output_cost_per_1m=0.00,
    ),
    "ollama/llama3.2:3b": ModelPricing(
        provider="ollama",
        input_cost_per_1m=0.00,
        output_cost_per_1m=0.00,
    ),
    "ollama/deepseek-coder-v2:16b": ModelPricing(
        provider="ollama",
        input_cost_per_1m=0.00,
        output_cost_per_1m=0.00,
    ),
}


def get_model_pricing(model_id: str) -> ModelPricing:
    """Get pricing for a model.

    Args:
        model_id: LiteLLM model identifier.

    Returns:
        ModelPricing for the model.

    Raises:
        ValueError: If model_id is not in the pricing table.
    """
    pricing = MODEL_PRICING.get(model_id)
    if pricing is None:
        msg = f"Unknown model for pricing: {model_id}"
        raise ValueError(msg)
    return pricing


def calculate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Calculate cost in USD for an API call.

    Args:
        model_id: LiteLLM model identifier.
        input_tokens: Total input tokens (including cached).
        output_tokens: Output tokens generated.
        cached_tokens: Input tokens served from cache.

    Returns:
        Cost in USD (always >= 0).

    Raises:
        ValueError: If model_id is not in the pricing table.
    """
    pricing = get_model_pricing(model_id)

    non_cached_input = max(0, input_tokens - cached_tokens)
    input_cost = (non_cached_input / 1_000_000) * pricing.input_cost_per_1m
    cached_cost = (cached_tokens / 1_000_000) * pricing.input_cost_per_1m * pricing.cache_discount
    output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_1m

    return max(0.0, input_cost + cached_cost + output_cost)


def estimate_input_tokens(text: str) -> int:
    """Estimate token count for input text.

    Uses a heuristic blend of word-based and character-based estimation.
    Actual tokenization varies by model, so this is an approximation.

    Args:
        text: The input text to estimate tokens for.

    Returns:
        Estimated token count (always >= 0).
    """
    if not text:
        return 0

    word_count = len(text.split())
    char_count = len(text)

    # English text: ~0.75 tokens per word
    # Code: ~0.4 tokens per character
    word_estimate = int(word_count * 0.75)
    char_estimate = int(char_count * 0.4)

    # Use the higher estimate for safety margin
    return max(1, word_estimate, char_estimate)


# Output token estimates by task type keyword
OUTPUT_TOKEN_ESTIMATES: dict[str, int] = {
    "explain": 300,
    "what does": 300,
    "describe": 300,
    "rename": 150,
    "fix typo": 100,
    "typo": 100,
    "format": 200,
    "indent": 150,
    "comment": 200,
    "test": 1500,
    "write test": 1500,
    "refactor": 2000,
    "restructure": 2000,
    "debug": 1000,
    "error": 1000,
    "fix bug": 1000,
    "implement": 2500,
    "add feature": 2500,
    "create": 2500,
    "design": 3000,
    "architect": 3000,
    "plan": 3000,
}

DEFAULT_OUTPUT_ESTIMATE: int = 1000


def estimate_output_tokens(prompt: str) -> int:
    """Estimate expected output tokens for a prompt.

    Uses keyword matching to guess the task type and expected output size.

    Args:
        prompt: The user's input prompt.

    Returns:
        Estimated output token count.
    """
    if not prompt:
        return DEFAULT_OUTPUT_ESTIMATE

    prompt_lower = prompt.lower()
    for keyword, estimate in OUTPUT_TOKEN_ESTIMATES.items():
        if keyword in prompt_lower:
            return estimate

    return DEFAULT_OUTPUT_ESTIMATE


def get_provider_for_model(model_id: str) -> str:
    """Extract provider name from a model ID.

    Args:
        model_id: LiteLLM model identifier.

    Returns:
        Provider name string.
    """
    pricing = MODEL_PRICING.get(model_id)
    if pricing:
        return pricing.provider

    # Infer from model ID prefix
    if "/" in model_id:
        prefix = model_id.split("/", maxsplit=1)[0]
        known_prefixes = {
            "gemini", "deepseek", "groq", "mistral", "ollama", "openai",
            "perplexity", "cohere", "together_ai", "fireworks_ai", "moonshot",
            "qwen",
        }
        if prefix in known_prefixes:
            return prefix

    # Default inference for known patterns
    if model_id.startswith("claude"):
        return "anthropic"
    if model_id.startswith("gpt") or model_id.startswith("o3") or model_id.startswith("o4"):
        return "openai"
    if model_id.startswith("moonshot"):
        return "kimi"

    return "unknown"
