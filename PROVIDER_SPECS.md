# PROVIDER_SPECS.md — Prism Provider Specifications

## Provider Configuration Format

Each provider is defined by:
```python
@dataclass
class ProviderConfig:
    name: str                          # e.g., "anthropic"
    display_name: str                  # e.g., "Anthropic"
    api_key_env: str                   # e.g., "ANTHROPIC_API_KEY"
    api_base: str | None               # Custom endpoint URL (None = default)
    models: list[ModelConfig]          # Available models
    free_tier: FreeTierConfig | None   # Free tier limits
    enabled: bool = True               # User can disable

@dataclass
class ModelConfig:
    id: str                            # LiteLLM model identifier
    display_name: str
    tier: ComplexityTier               # Which tier this model serves
    input_cost_per_1m: float           # USD per 1M input tokens
    output_cost_per_1m: float          # USD per 1M output tokens
    context_window: int                # Max tokens
    supports_tools: bool = True        # Function calling support
    supports_vision: bool = False      # Image input support
    supports_streaming: bool = True
    max_output_tokens: int = 4096

@dataclass
class FreeTierConfig:
    requests_per_day: int
    requests_per_minute: int
    tokens_per_minute: int | None = None
    requires_credit_card: bool = False
```

## Tier 1: Premium Cloud Models

### Anthropic
```python
ProviderConfig(
    name="anthropic",
    display_name="Anthropic",
    api_key_env="ANTHROPIC_API_KEY",
    api_base=None,  # Default: https://api.anthropic.com
    models=[
        ModelConfig(
            id="claude-sonnet-4-20250514",
            display_name="Claude Sonnet 4",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=3.00,
            output_cost_per_1m=15.00,
            context_window=200_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=8192,
        ),
        ModelConfig(
            id="claude-opus-4-20250514",
            display_name="Claude Opus 4",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=15.00,
            output_cost_per_1m=75.00,
            context_window=200_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=8192,
        ),
        ModelConfig(
            id="claude-haiku-4-5-20251001",
            display_name="Claude Haiku 4.5",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.80,
            output_cost_per_1m=4.00,
            context_window=200_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=8192,
        ),
    ],
    free_tier=None,
)
```

### OpenAI
```python
ProviderConfig(
    name="openai",
    display_name="OpenAI",
    api_key_env="OPENAI_API_KEY",
    api_base=None,  # Default: https://api.openai.com/v1
    models=[
        ModelConfig(
            id="gpt-4o",
            display_name="GPT-4o",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=2.50,
            output_cost_per_1m=10.00,
            context_window=128_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=16384,
        ),
        ModelConfig(
            id="gpt-4o-mini",
            display_name="GPT-4o Mini",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.15,
            output_cost_per_1m=0.60,
            context_window=128_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=16384,
        ),
        ModelConfig(
            id="o3",
            display_name="o3",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=10.00,
            output_cost_per_1m=40.00,
            context_window=200_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=100_000,
        ),
        ModelConfig(
            id="o4-mini",
            display_name="o4-mini",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=1.10,
            output_cost_per_1m=4.40,
            context_window=200_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=100_000,
        ),
    ],
    free_tier=None,
)
```

### Google (AI Studio)
```python
ProviderConfig(
    name="google",
    display_name="Google AI Studio",
    api_key_env="GOOGLE_API_KEY",
    api_base=None,  # Default: Gemini API
    models=[
        ModelConfig(
            id="gemini/gemini-2.5-pro",
            display_name="Gemini 2.5 Pro",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=1.25,
            output_cost_per_1m=5.00,
            context_window=1_000_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=8192,
        ),
        ModelConfig(
            id="gemini/gemini-2.0-flash",
            display_name="Gemini 2.0 Flash",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.10,
            output_cost_per_1m=0.40,
            context_window=1_000_000,
            supports_tools=True,
            supports_vision=True,
            max_output_tokens=8192,
        ),
    ],
    free_tier=FreeTierConfig(
        requests_per_day=1500,
        requests_per_minute=15,
        tokens_per_minute=1_000_000,
        requires_credit_card=False,
    ),
)
```

## Tier 2: Cost-Efficient Cloud Models

### DeepSeek
```python
ProviderConfig(
    name="deepseek",
    display_name="DeepSeek",
    api_key_env="DEEPSEEK_API_KEY",
    api_base=None,  # Default: https://api.deepseek.com
    models=[
        ModelConfig(
            id="deepseek/deepseek-chat",
            display_name="DeepSeek V3",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.27,
            output_cost_per_1m=1.10,
            context_window=64_000,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
        ModelConfig(
            id="deepseek/deepseek-reasoner",
            display_name="DeepSeek R1",
            tier=ComplexityTier.COMPLEX,
            input_cost_per_1m=0.55,
            output_cost_per_1m=2.19,
            context_window=64_000,
            supports_tools=False,
            supports_vision=False,
            max_output_tokens=8192,
        ),
    ],
    free_tier=None,
)
```

### Groq
```python
ProviderConfig(
    name="groq",
    display_name="Groq",
    api_key_env="GROQ_API_KEY",
    api_base=None,  # Default: https://api.groq.com/openai/v1
    models=[
        ModelConfig(
            id="groq/llama-3.3-70b-versatile",
            display_name="Llama 3.3 70B",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.59,
            output_cost_per_1m=0.79,
            context_window=128_000,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
        ModelConfig(
            id="groq/mixtral-8x7b-32768",
            display_name="Mixtral 8x7B",
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
        requires_credit_card=False,
    ),
)
```

### Mistral
```python
ProviderConfig(
    name="mistral",
    display_name="Mistral",
    api_key_env="MISTRAL_API_KEY",
    api_base=None,  # Default: https://api.mistral.ai/v1
    models=[
        ModelConfig(
            id="mistral/mistral-small-latest",
            display_name="Mistral Small",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.20,
            output_cost_per_1m=0.60,
            context_window=128_000,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
        ModelConfig(
            id="mistral/codestral-latest",
            display_name="Codestral",
            tier=ComplexityTier.MEDIUM,
            input_cost_per_1m=0.30,
            output_cost_per_1m=0.90,
            context_window=32_000,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=8192,
        ),
    ],
    free_tier=None,
)
```

## Tier 3: Free and Local Models

### Ollama (Local)
```python
ProviderConfig(
    name="ollama",
    display_name="Ollama (Local)",
    api_key_env="",  # No key needed
    api_base="http://localhost:11434",
    models=[
        ModelConfig(
            id="ollama/qwen2.5-coder:7b",
            display_name="Qwen 2.5 Coder 7B",
            tier=ComplexityTier.SIMPLE,
            input_cost_per_1m=0.00,
            output_cost_per_1m=0.00,
            context_window=32_768,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=4096,
        ),
        ModelConfig(
            id="ollama/llama3.2:3b",
            display_name="Llama 3.2 3B",
            tier=ComplexityTier.SIMPLE,
            input_cost_per_1m=0.00,
            output_cost_per_1m=0.00,
            context_window=32_768,
            supports_tools=False,
            supports_vision=False,
            max_output_tokens=4096,
        ),
        ModelConfig(
            id="ollama/deepseek-coder-v2:16b",
            display_name="DeepSeek Coder V2 16B",
            tier=ComplexityTier.SIMPLE,
            input_cost_per_1m=0.00,
            output_cost_per_1m=0.00,
            context_window=32_768,
            supports_tools=True,
            supports_vision=False,
            max_output_tokens=4096,
        ),
    ],
    free_tier=None,  # Always free
)
```

## Custom Provider Template

For any OpenAI-compatible endpoint:
```yaml
# In ~/.prism/config.yaml
providers:
  my-custom-provider:
    api_base: "https://api.example.com/v1"
    api_key_env: "CUSTOM_API_KEY"
    models:
      - id: "openai/custom-model"
        display_name: "Custom Model"
        tier: "medium"
        input_cost_per_1m: 1.00
        output_cost_per_1m: 3.00
        context_window: 32768
        supports_tools: true
        supports_vision: false
```

## Provider Health Check Protocol

Each provider health check does the minimum possible:

| Provider | Health Check Method | Expected Response Time |
|----------|-------------------|----------------------|
| Anthropic | `GET /v1/models` with API key header | < 2s |
| OpenAI | `GET /v1/models` with API key header | < 2s |
| Google | `GET /v1beta/models` with API key param | < 2s |
| DeepSeek | `GET /v1/models` with API key header | < 3s |
| Groq | `GET /openai/v1/models` with API key header | < 2s |
| Mistral | `GET /v1/models` with API key header | < 2s |
| Ollama | `GET /api/tags` (no auth) | < 1s |
| Custom | `GET /v1/models` with API key header | < 5s |

Health checks are:
- Cached for 60 seconds
- Run in parallel on `prism status`
- Run in background on session start (non-blocking)
- **Mocked in all tests** — never make real API calls

## LiteLLM Model ID Mapping

Prism uses LiteLLM's model naming convention:

| Prism Display | LiteLLM Model ID |
|--------------|------------------|
| Claude Sonnet 4 | `claude-sonnet-4-20250514` |
| Claude Opus 4 | `claude-opus-4-20250514` |
| GPT-4o | `gpt-4o` |
| GPT-4o Mini | `gpt-4o-mini` |
| Gemini 2.0 Flash | `gemini/gemini-2.0-flash` |
| Gemini 2.5 Pro | `gemini/gemini-2.5-pro` |
| DeepSeek V3 | `deepseek/deepseek-chat` |
| DeepSeek R1 | `deepseek/deepseek-reasoner` |
| Llama 3.3 70B (Groq) | `groq/llama-3.3-70b-versatile` |
| Mistral Small | `mistral/mistral-small-latest` |
| Qwen 2.5 Coder 7B | `ollama/qwen2.5-coder:7b` |
| Llama 3.2 3B | `ollama/llama3.2:3b` |
