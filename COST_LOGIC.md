# COST_LOGIC.md — Prism Cost Estimation, Tracking, and Budget Enforcement

## Overview

Every API call in Prism has a cost lifecycle:
1. **Estimate** (before call): predict cost based on input/output token estimates
2. **Budget check** (before call): verify estimated cost fits within limits
3. **Track** (after call): record actual cost from real token counts
4. **Report** (on demand): display cost dashboard via `/cost`

## Token Estimation

### Input Token Estimation
```python
def estimate_input_tokens(text: str) -> int:
    """Estimate tokens for input text.

    Rough approximation — actual tokenization varies by model:
    - English text: ~0.75 tokens per word
    - Code: ~0.4 tokens per character
    - JSON/structured: ~0.5 tokens per character
    """
    word_count = len(text.split())
    char_count = len(text)

    # Heuristic: blend word-based and char-based estimates
    word_estimate = int(word_count * 0.75)
    char_estimate = int(char_count * 0.4)

    # Use the higher estimate for safety
    return max(word_estimate, char_estimate)
```

### Output Token Estimation by Task Type
| Task Type | Detection Keywords | Estimated Output Tokens |
|-----------|-------------------|------------------------|
| Explain | "explain", "what does", "describe" | 300 |
| Rename/Fix | "rename", "fix", "typo" | 150 |
| Format | "format", "indent", "style" | 200 |
| Write test | "test", "write test" | 1,500 |
| Refactor | "refactor", "restructure" | 2,000 |
| Debug | "debug", "error", "fix bug" | 1,000 |
| Implement | "implement", "add feature", "create" | 2,500 |
| Architecture | "design", "architect", "plan" | 3,000 |
| Unknown | (default) | 1,000 |

### Context Token Budget
```
System prompt + tools schema:     ~2,000 tokens
Repo map (if present):            ~1,000-5,000 tokens
Project memory (.prism.md):       ~500-1,000 tokens
Active file contents:             measured per file
Conversation history:             measured cumulatively

Total input = system + repo_map + memory + files + history + user_prompt
```

## Cost Calculation

### Per-Request Cost Formula
```python
def calculate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Calculate cost in USD for a model call.

    Cached tokens are charged at a reduced rate (varies by provider):
    - Anthropic: cached tokens at 10% of input price
    - OpenAI: cached tokens at 50% of input price
    - Others: cached tokens at 0% (no cache support)
    """
    pricing = get_model_pricing(model_id)

    non_cached_input = input_tokens - cached_tokens
    cache_discount = get_cache_discount(pricing.provider)

    input_cost = (non_cached_input / 1_000_000) * pricing.input_cost_per_1m
    cached_cost = (cached_tokens / 1_000_000) * pricing.input_cost_per_1m * cache_discount
    output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_1m

    return input_cost + cached_cost + output_cost
```

### Cache Discount Rates
| Provider | Cache Discount (% of input price) |
|----------|----------------------------------|
| Anthropic | 10% |
| OpenAI | 50% |
| Google | 25% |
| DeepSeek | 10% (cache hit at $0.07/1M) |
| Others | 0% (no caching) |

## Model Pricing Table

Maintained in `src/prism/cost/pricing.py`:

```python
MODEL_PRICING: dict[str, ModelPricing] = {
    # Tier 1: Premium
    "claude-sonnet-4-20250514": ModelPricing(
        provider="anthropic",
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
    ),
    "claude-opus-4-20250514": ModelPricing(
        provider="anthropic",
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
    ),
    "gpt-4o": ModelPricing(
        provider="openai",
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
    ),
    "o3": ModelPricing(
        provider="openai",
        input_cost_per_1m=10.00,
        output_cost_per_1m=40.00,
    ),
    "gemini/gemini-2.5-pro": ModelPricing(
        provider="google",
        input_cost_per_1m=1.25,
        output_cost_per_1m=5.00,
    ),

    # Tier 2: Cost-efficient
    "deepseek/deepseek-chat": ModelPricing(
        provider="deepseek",
        input_cost_per_1m=0.27,
        output_cost_per_1m=1.10,
    ),
    "deepseek/deepseek-reasoner": ModelPricing(
        provider="deepseek",
        input_cost_per_1m=0.55,
        output_cost_per_1m=2.19,
    ),
    "groq/llama-3.3-70b-versatile": ModelPricing(
        provider="groq",
        input_cost_per_1m=0.59,
        output_cost_per_1m=0.79,
    ),
    "mistral/mistral-small-latest": ModelPricing(
        provider="mistral",
        input_cost_per_1m=0.20,
        output_cost_per_1m=0.60,
    ),
    "gemini/gemini-2.0-flash": ModelPricing(
        provider="google",
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.40,
    ),
    "gpt-4o-mini": ModelPricing(
        provider="openai",
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
    ),
    "o4-mini": ModelPricing(
        provider="openai",
        input_cost_per_1m=1.10,
        output_cost_per_1m=4.40,
    ),
    "claude-haiku-4-5-20251001": ModelPricing(
        provider="anthropic",
        input_cost_per_1m=0.80,
        output_cost_per_1m=4.00,
    ),

    # Tier 3: Free/Local
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
}
```

## Budget Enforcement

### Budget Configuration
```yaml
# ~/.prism/config.yaml
budget:
  daily_limit: 5.00       # USD per day (null = unlimited)
  monthly_limit: 50.00    # USD per month (null = unlimited)
  warn_at_percent: 80     # Warn when 80% of budget consumed
```

### Budget Check Flow
```
1. Get estimated cost for the request
2. Get current daily/monthly totals from SQLite
3. Check daily limit:
   a. If daily_total + estimated > daily_limit → BLOCK
   b. If daily_total + estimated > daily_limit * warn_percent → WARN
4. Check monthly limit:
   a. If monthly_total + estimated > monthly_limit → BLOCK
   b. If monthly_total + estimated > monthly_limit * warn_percent → WARN
5. Return action: PROCEED, WARN, or BLOCK
```

### User Actions on Budget Block
When budget is exceeded:
```
⚠ Daily budget limit reached: $4.87 of $5.00 used
  Estimated cost of this request: $0.25

  Options:
  1. Increase daily limit: prism config set budget.daily_limit 10
  2. Use a cheaper model: /model ollama/qwen2.5-coder:7b
  3. Wait until tomorrow (budget resets at midnight UTC)
  4. Continue anyway: type 'yes' to override once
```

## Cost Dashboard (`/cost`)

### Display Format
```
┌─────────────────────────────────────────────────────────┐
│                    Cost Dashboard                         │
├─────────────────────────────────────────────────────────┤
│ Session:      $0.42   (23 requests)                      │
│ Today:        $1.87   (89 requests)                      │
│ This month:   $12.34  (1,247 requests)                   │
│ Budget:       $37.66 remaining / $50.00 monthly          │
├─────────────────────────────────────────────────────────┤
│ Model Breakdown (this month):                            │
│                                                          │
│ Ollama/qwen2.5:7b  │ 847 req │   $0.00 │ ████████████ 68% │
│ Gemini 2.0 Flash   │ 289 req │   $0.94 │ ████░░░░░░░ 23%  │
│ Claude Sonnet 4    │  87 req │   $9.12 │ █░░░░░░░░░░  7%  │
│ DeepSeek V3        │  24 req │   $0.28 │ ░░░░░░░░░░░  2%  │
├─────────────────────────────────────────────────────────┤
│ Savings estimate:                                        │
│ If all routed to Claude Sonnet: ~$89.23                  │
│ Actual cost with Prism routing:   $12.34                 │
│ You saved: ~$76.89 (86%)                                 │
└─────────────────────────────────────────────────────────┘
```

### Savings Calculation
```python
def calculate_savings(cost_entries: list[CostEntry]) -> float:
    """Calculate how much the user saved vs. using only premium models."""
    premium_model = "claude-sonnet-4-20250514"
    premium_pricing = get_model_pricing(premium_model)

    hypothetical_cost = sum(
        (e.input_tokens / 1_000_000) * premium_pricing.input_cost_per_1m +
        (e.output_tokens / 1_000_000) * premium_pricing.output_cost_per_1m
        for e in cost_entries
    )

    actual_cost = sum(e.cost_usd for e in cost_entries)

    return hypothetical_cost - actual_cost
```

## Cost Tracking Implementation

### Write Path
Every API call → `CostTracker.track()`:
```python
def track(self, result: CompletionResult, session_id: str, tier: ComplexityTier) -> CostEntry:
    cost = calculate_cost(
        model_id=result.model_id,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached_tokens=result.cached_tokens,
    )

    entry = CostEntry(
        id=str(uuid4()),
        timestamp=datetime.utcnow(),
        session_id=session_id,
        model_id=result.model_id,
        provider=get_provider(result.model_id),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached_tokens=result.cached_tokens,
        cost_usd=cost,
        complexity_tier=tier,
    )

    self.db.save_cost_entry(entry)
    self.db.update_session(session_id, cost_delta=cost, request_delta=1)

    return entry
```

### Read Path
Budget checks and dashboard use aggregation queries:
```sql
-- Daily total
SELECT COALESCE(SUM(cost_usd), 0.0) FROM cost_entries
WHERE created_date = date('now');

-- Monthly total
SELECT COALESCE(SUM(cost_usd), 0.0) FROM cost_entries
WHERE created_date >= date('now', 'start of month');

-- Model breakdown
SELECT model_id, COUNT(*) as count, SUM(cost_usd) as total
FROM cost_entries
WHERE created_date >= date('now', 'start of month')
GROUP BY model_id
ORDER BY total DESC;
```
