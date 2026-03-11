"""Extended provider configurations — new providers and model updates.

This module registers additional providers and models that extend the
built-in set defined in :mod:`prism.providers.base`.
"""

from __future__ import annotations

from prism.cost.pricing import MODEL_PRICING, ModelPricing
from prism.providers.base import (
    BUILTIN_PROVIDERS,
    ComplexityTier,
    ModelInfo,
    ProviderConfig,
)

# ======================================================================
# New providers
# ======================================================================

KIMI_CONFIG = ProviderConfig(
    name="kimi",
    display_name="Kimi (Moonshot AI)",
    api_key_env="MOONSHOT_API_KEY",
    api_base="https://api.moonshot.cn/v1",
    models=[
        ModelInfo(
            id="moonshot/moonshot-v1-8k",
            display_name="Moonshot V1 8K",
            provider="kimi",
            tier=ComplexityTier.SIMPLE,
            input_cost_per_1m=0.12,
            output_cost_per_1m=0.12,
            context_window=8_192,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=4096,
        ),
        ModelInfo(
            id="moonshot/moonshot-v1-32k",
            display_name="Moonshot V1 32K",
            provider="kimi",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.24,
            output_cost_per_1m=0.24,
            context_window=32_768,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=4096,
        ),
        ModelInfo(
            id="moonshot/moonshot-v1-128k",
            display_name="Moonshot V1 128K",
            provider="kimi",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=0.48,
            output_cost_per_1m=0.48,
            context_window=128_000,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=4096,
        ),
    ],
)

PERPLEXITY_CONFIG = ProviderConfig(
    name="perplexity",
    display_name="Perplexity",
    api_key_env="PERPLEXITY_API_KEY",
    models=[
        ModelInfo(
            id="perplexity/llama-3.1-sonar-large-128k-online",
            display_name="Sonar Large 128K Online",
            provider="perplexity",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=1.00,
            output_cost_per_1m=1.00,
            context_window=127_072,
            supports_tools=False,
            supports_vision=False,
            max_output_tokens=4096,
        ),
        ModelInfo(
            id="perplexity/llama-3.1-sonar-small-128k-online",
            display_name="Sonar Small 128K Online",
            provider="perplexity",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.20,
            output_cost_per_1m=0.20,
            context_window=127_072,
            supports_tools=False,
            supports_vision=False,
            max_output_tokens=4096,
        ),
    ],
)

QWEN_CONFIG = ProviderConfig(
    name="qwen",
    display_name="Qwen (Alibaba)",
    api_key_env="DASHSCOPE_API_KEY",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    models=[
        ModelInfo(
            id="qwen/qwen-max",
            display_name="Qwen Max",
            provider="qwen",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=2.40,
            output_cost_per_1m=9.60,
            context_window=32_768,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
        ModelInfo(
            id="qwen/qwen-turbo",
            display_name="Qwen Turbo",
            provider="qwen",
            tier=ComplexityTier.SIMPLE,
            input_cost_per_1m=0.30,
            output_cost_per_1m=0.60,
            context_window=131_072,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
        ModelInfo(
            id="qwen/qwen-plus",
            display_name="Qwen Plus",
            provider="qwen",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.80,
            output_cost_per_1m=2.00,
            context_window=131_072,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
    ],
)

COHERE_CONFIG = ProviderConfig(
    name="cohere",
    display_name="Cohere",
    api_key_env="COHERE_API_KEY",
    models=[
        ModelInfo(
            id="cohere/command-r-plus",
            display_name="Command R+",
            provider="cohere",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=2.50,
            output_cost_per_1m=10.00,
            context_window=128_000,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=4096,
        ),
        ModelInfo(
            id="cohere/command-r",
            display_name="Command R",
            provider="cohere",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.15,
            output_cost_per_1m=0.60,
            context_window=128_000,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=4096,
        ),
    ],
)

TOGETHER_AI_CONFIG = ProviderConfig(
    name="together_ai",
    display_name="Together AI",
    api_key_env="TOGETHER_API_KEY",
    models=[
        ModelInfo(
            id="together_ai/meta-llama/Llama-3-70b-chat-hf",
            display_name="Llama 3 70B (Together)",
            provider="together_ai",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=0.90,
            output_cost_per_1m=0.90,
            context_window=8_192,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=4096,
        ),
    ],
)

FIREWORKS_AI_CONFIG = ProviderConfig(
    name="fireworks_ai",
    display_name="Fireworks AI",
    api_key_env="FIREWORKS_API_KEY",
    models=[
        ModelInfo(
            id="fireworks_ai/llama-v3p1-70b-instruct",
            display_name="Llama 3.1 70B (Fireworks)",
            provider="fireworks_ai",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=0.90,
            output_cost_per_1m=0.90,
            context_window=131_072,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=4096,
        ),
    ],
)

# ======================================================================
# Additional models for existing providers (Gemini 2.5 Flash)
# ======================================================================

_GEMINI_2_5_FLASH = ModelInfo(
    id="gemini/gemini-2.5-flash",
    display_name="Gemini 2.5 Flash",
    provider="google",
    tier=ComplexityTier.MEDIUM,
    input_cost_per_1m=0.15,
    output_cost_per_1m=0.60,
    context_window=1_000_000,
    supports_tools=True,
    supports_vision=True,
    max_output_tokens=8192,
)

# ======================================================================
# All new provider configs collected for registration
# ======================================================================

EXTENDED_PROVIDERS: list[ProviderConfig] = [
    KIMI_CONFIG,
    PERPLEXITY_CONFIG,
    QWEN_CONFIG,
    COHERE_CONFIG,
    TOGETHER_AI_CONFIG,
    FIREWORKS_AI_CONFIG,
]

# Additional models to be appended to existing providers
EXTRA_MODELS_FOR_EXISTING: dict[str, list[ModelInfo]] = {
    "google": [_GEMINI_2_5_FLASH],
}

# ======================================================================
# Extended pricing table entries
# ======================================================================

EXTENDED_PRICING: dict[str, ModelPricing] = {
    # Kimi / Moonshot
    "moonshot/moonshot-v1-8k": ModelPricing(
        provider="kimi", input_cost_per_1m=0.12, output_cost_per_1m=0.12,
    ),
    "moonshot/moonshot-v1-32k": ModelPricing(
        provider="kimi", input_cost_per_1m=0.24, output_cost_per_1m=0.24,
    ),
    "moonshot/moonshot-v1-128k": ModelPricing(
        provider="kimi", input_cost_per_1m=0.48, output_cost_per_1m=0.48,
    ),
    # Perplexity
    "perplexity/llama-3.1-sonar-large-128k-online": ModelPricing(
        provider="perplexity", input_cost_per_1m=1.00, output_cost_per_1m=1.00,
    ),
    "perplexity/llama-3.1-sonar-small-128k-online": ModelPricing(
        provider="perplexity", input_cost_per_1m=0.20, output_cost_per_1m=0.20,
    ),
    # Qwen
    "qwen/qwen-max": ModelPricing(
        provider="qwen", input_cost_per_1m=2.40, output_cost_per_1m=9.60,
    ),
    "qwen/qwen-turbo": ModelPricing(
        provider="qwen", input_cost_per_1m=0.30, output_cost_per_1m=0.60,
    ),
    "qwen/qwen-plus": ModelPricing(
        provider="qwen", input_cost_per_1m=0.80, output_cost_per_1m=2.00,
    ),
    # Cohere
    "cohere/command-r-plus": ModelPricing(
        provider="cohere", input_cost_per_1m=2.50, output_cost_per_1m=10.00,
    ),
    "cohere/command-r": ModelPricing(
        provider="cohere", input_cost_per_1m=0.15, output_cost_per_1m=0.60,
    ),
    # Together AI
    "together_ai/meta-llama/Llama-3-70b-chat-hf": ModelPricing(
        provider="together_ai", input_cost_per_1m=0.90, output_cost_per_1m=0.90,
    ),
    # Fireworks AI
    "fireworks_ai/llama-v3p1-70b-instruct": ModelPricing(
        provider="fireworks_ai", input_cost_per_1m=0.90, output_cost_per_1m=0.90,
    ),
    # Gemini 2.5 Flash (extends existing Google)
    "gemini/gemini-2.5-flash": ModelPricing(
        provider="google", input_cost_per_1m=0.15, output_cost_per_1m=0.60,
        cache_discount=0.25,
    ),
}


def register_extended_providers() -> None:
    """Add the extended providers to ``BUILTIN_PROVIDERS`` and ``MODEL_PRICING``.

    This is idempotent — calling it multiple times is safe.
    """
    existing_names = {p.name for p in BUILTIN_PROVIDERS}
    for provider in EXTENDED_PROVIDERS:
        if provider.name not in existing_names:
            BUILTIN_PROVIDERS.append(provider)
            existing_names.add(provider.name)

    # Add extra models to existing providers
    for provider_name, models in EXTRA_MODELS_FOR_EXISTING.items():
        for provider_cfg in BUILTIN_PROVIDERS:
            if provider_cfg.name == provider_name:
                existing_ids = {m.id for m in provider_cfg.models}
                for model in models:
                    if model.id not in existing_ids:
                        provider_cfg.models.append(model)
                break

    # Extend pricing
    for model_id, pricing in EXTENDED_PRICING.items():
        if model_id not in MODEL_PRICING:
            MODEL_PRICING[model_id] = pricing
