"""Base provider definitions and model metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ComplexityTier(StrEnum):
    """Task complexity classification tiers."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass(frozen=True)
class ModelInfo:
    """Metadata about a specific AI model."""

    id: str  # LiteLLM model identifier
    display_name: str
    provider: str
    tier: ComplexityTier
    input_cost_per_1m: float  # USD per 1M input tokens
    output_cost_per_1m: float  # USD per 1M output tokens
    context_window: int  # Maximum tokens
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    max_output_tokens: int = 4096


@dataclass(frozen=True)
class FreeTierConfig:
    """Free tier rate limits for a provider."""

    requests_per_day: int
    requests_per_minute: int
    tokens_per_minute: int | None = None
    requires_credit_card: bool = False


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an AI provider."""

    name: str
    display_name: str
    api_key_env: str  # Environment variable name for API key
    api_base: str | None = None  # Custom endpoint URL (None = provider default)
    models: list[ModelInfo] = field(default_factory=list)
    free_tier: FreeTierConfig | None = None
    enabled: bool = True


# --- Built-in Provider Configurations ---

ANTHROPIC_CONFIG = ProviderConfig(
    name="anthropic",
    display_name="Anthropic",
    api_key_env="ANTHROPIC_API_KEY",
    models=[
        ModelInfo(
            id="claude-sonnet-4-20250514",
            display_name="Claude Sonnet 4",
            provider="anthropic",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=3.00,
            output_cost_per_1m=15.00,
            context_window=200_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=8192,
        ),
        ModelInfo(
            id="claude-opus-4-20250514",
            display_name="Claude Opus 4",
            provider="anthropic",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=15.00,
            output_cost_per_1m=75.00,
            context_window=200_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=8192,
        ),
        ModelInfo(
            id="claude-haiku-4-5-20251001",
            display_name="Claude Haiku 4.5",
            provider="anthropic",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.80,
            output_cost_per_1m=4.00,
            context_window=200_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=8192,
        ),
    ],
)

OPENAI_CONFIG = ProviderConfig(
    name="openai",
    display_name="OpenAI",
    api_key_env="OPENAI_API_KEY",
    models=[
        ModelInfo(
            id="gpt-4o",
            display_name="GPT-4o",
            provider="openai",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=2.50,
            output_cost_per_1m=10.00,
            context_window=128_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=16384,
        ),
        ModelInfo(
            id="gpt-4o-mini",
            display_name="GPT-4o Mini",
            provider="openai",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.15,
            output_cost_per_1m=0.60,
            context_window=128_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=16384,
        ),
        ModelInfo(
            id="o3",
            display_name="o3",
            provider="openai",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=10.00,
            output_cost_per_1m=40.00,
            context_window=200_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=100_000,
        ),
        ModelInfo(
            id="o4-mini",
            display_name="o4-mini",
            provider="openai",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=1.10,
            output_cost_per_1m=4.40,
            context_window=200_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=100_000,
        ),
    ],
)

GOOGLE_CONFIG = ProviderConfig(
    name="google",
    display_name="Google AI Studio",
    api_key_env="GOOGLE_API_KEY",
    models=[
        ModelInfo(
            id="gemini/gemini-2.5-pro",
            display_name="Gemini 2.5 Pro",
            provider="google",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=1.25,
            output_cost_per_1m=5.00,
            context_window=2_097_152,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=8192,
        ),
        ModelInfo(
            id="gemini/gemini-2.0-flash",
            display_name="Gemini 2.0 Flash",
            provider="google",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.10,
            output_cost_per_1m=0.40,
            context_window=1_048_576,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=8192,
        ),
    ],
    free_tier=FreeTierConfig(
        requests_per_day=1500,
        requests_per_minute=15,
        tokens_per_minute=1_000_000,
    ),
)

DEEPSEEK_CONFIG = ProviderConfig(
    name="deepseek",
    display_name="DeepSeek",
    api_key_env="DEEPSEEK_API_KEY",
    models=[
        ModelInfo(
            id="deepseek/deepseek-chat",
            display_name="DeepSeek V3",
            provider="deepseek",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.27,
            output_cost_per_1m=1.10,
            context_window=64_000,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
        ModelInfo(
            id="deepseek/deepseek-reasoner",
            display_name="DeepSeek R1",
            provider="deepseek",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=0.55,
            output_cost_per_1m=2.19,
            context_window=64_000,
            supports_tools=False,
            supports_vision=False,
            max_output_tokens=8192,
        ),
    ],
)

GROQ_CONFIG = ProviderConfig(
    name="groq",
    display_name="Groq",
    api_key_env="GROQ_API_KEY",
    models=[
        ModelInfo(
            id="groq/llama-3.3-70b-versatile",
            display_name="Llama 3.3 70B",
            provider="groq",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.59,
            output_cost_per_1m=0.79,
            context_window=128_000,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
        ModelInfo(
            id="groq/mixtral-8x7b-32768",
            display_name="Mixtral 8x7B",
            provider="groq",
            tier=ComplexityTier.SIMPLE,
            input_cost_per_1m=0.24,
            output_cost_per_1m=0.24,
            context_window=32_768,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
    ],
    free_tier=FreeTierConfig(
        requests_per_day=14400,
        requests_per_minute=30,
        tokens_per_minute=20_000,
    ),
)

MISTRAL_CONFIG = ProviderConfig(
    name="mistral",
    display_name="Mistral",
    api_key_env="MISTRAL_API_KEY",
    models=[
        ModelInfo(
            id="mistral/mistral-small-latest",
            display_name="Mistral Small",
            provider="mistral",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.20,
            output_cost_per_1m=0.60,
            context_window=128_000,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
        ModelInfo(
            id="mistral/codestral-latest",
            display_name="Codestral",
            provider="mistral",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.20,
            output_cost_per_1m=0.60,
            context_window=200_000,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
    ],
)

OPENROUTER_CONFIG = ProviderConfig(
    name="openrouter",
    display_name="OpenRouter",
    api_key_env="OPENROUTER_API_KEY",
    api_base="https://openrouter.ai/api/v1",
    models=[
        ModelInfo(
            id="openrouter/meta-llama/llama-3.3-70b-instruct:free",
            display_name="Llama 3.3 70B (Free)",
            provider="openrouter",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.00,
            output_cost_per_1m=0.00,
            context_window=128_000,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
        ModelInfo(
            id="openrouter/qwen/qwen3-coder:free",
            display_name="Qwen3 Coder (Free)",
            provider="openrouter",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=0.00,
            output_cost_per_1m=0.00,
            context_window=262_000,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
        ModelInfo(
            id="openrouter/google/gemma-3-27b-it:free",
            display_name="Gemma 3 27B (Free)",
            provider="openrouter",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.00,
            output_cost_per_1m=0.00,
            context_window=131_072,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
        ModelInfo(
            id="openrouter/nousresearch/hermes-3-llama-3.1-405b:free",
            display_name="Hermes 3 405B (Free)",
            provider="openrouter",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=0.00,
            output_cost_per_1m=0.00,
            context_window=131_072,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
        ModelInfo(
            id="openrouter/mistralai/mistral-small-3.1-24b-instruct:free",
            display_name="Mistral Small 3.1 (Free)",
            provider="openrouter",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.00,
            output_cost_per_1m=0.00,
            context_window=128_000,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
    ],
    free_tier=FreeTierConfig(
        requests_per_day=50,
        requests_per_minute=10,
        tokens_per_minute=100_000,
    ),
)

OLLAMA_CONFIG = ProviderConfig(
    name="ollama",
    display_name="Ollama (Local)",
    api_key_env="",  # No key needed
    api_base="http://localhost:11434",
    models=[
        ModelInfo(
            id="ollama/qwen2.5-coder:7b",
            display_name="Qwen 2.5 Coder 7B",
            provider="ollama",
            tier=ComplexityTier.SIMPLE,
            input_cost_per_1m=0.00,
            output_cost_per_1m=0.00,
            context_window=32_768,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=4096,
        ),
        ModelInfo(
            id="ollama/llama3.2:3b",
            display_name="Llama 3.2 3B",
            provider="ollama",
            tier=ComplexityTier.SIMPLE,
            input_cost_per_1m=0.00,
            output_cost_per_1m=0.00,
            context_window=32_768,
            supports_tools=False,
            supports_vision=False,
            max_output_tokens=4096,
        ),
        ModelInfo(
            id="ollama/deepseek-coder-v2:16b",
            display_name="DeepSeek Coder V2 16B",
            provider="ollama",
            tier=ComplexityTier.SIMPLE,
            input_cost_per_1m=0.00,
            output_cost_per_1m=0.00,
            context_window=32_768,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=4096,
        ),
    ],
)

# All built-in provider configs
BUILTIN_PROVIDERS: list[ProviderConfig] = [
    ANTHROPIC_CONFIG,
    OPENAI_CONFIG,
    GOOGLE_CONFIG,
    DEEPSEEK_CONFIG,
    GROQ_CONFIG,
    MISTRAL_CONFIG,
    OPENROUTER_CONFIG,
    OLLAMA_CONFIG,
]
