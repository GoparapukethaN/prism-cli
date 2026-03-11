# ROUTING_LOGIC.md — Prism Intelligent Routing System

## Overview

The routing engine is Prism's core differentiator. It classifies every user request by complexity, estimates token costs across candidate models, and dispatches to the cheapest model likely to succeed — learning from outcomes over time.

## Task Classification Algorithm

### Feature Extraction

For every user prompt, extract these features:

| Feature | Type | Description | Weight |
|---------|------|-------------|--------|
| `prompt_token_count` | int | Estimated tokens in the prompt | 0.15 |
| `files_referenced` | int | Number of files in active context | 0.15 |
| `estimated_output_tokens` | int | Expected output size by task type | 0.10 |
| `complexity_keywords` | float | Score from keyword matching (0-1) | 0.25 |
| `requires_reasoning` | bool→float | Does task need multi-step reasoning? | 0.20 |
| `scope` | float | Single-file (0) to architecture (1) | 0.15 |

### Keyword Scoring

#### Simple Keywords (reduce score)
```python
SIMPLE_KEYWORDS = {
    "fix": -0.15, "rename": -0.2, "format": -0.2, "explain": -0.15,
    "what does": -0.15, "typo": -0.25, "syntax": -0.2, "indent": -0.25,
    "comment": -0.15, "print": -0.2, "log": -0.15, "import": -0.2,
    "delete": -0.15, "remove": -0.15, "add a line": -0.2,
    "spelling": -0.25, "whitespace": -0.25,
}
```

#### Complex Keywords (increase score)
```python
COMPLEX_KEYWORDS = {
    "design": 0.25, "architect": 0.3, "from scratch": 0.25,
    "optimize": 0.2, "performance": 0.15, "security": 0.2,
    "migrate": 0.2, "refactor entire": 0.25, "redesign": 0.3,
    "system": 0.15, "scalable": 0.2, "microservice": 0.25,
    "algorithm": 0.2, "concurrent": 0.2, "distributed": 0.25,
    "abstract": 0.15, "pattern": 0.15, "trade-off": 0.2,
    "why": 0.1, "compare": 0.1, "evaluate": 0.15,
}
```

#### Medium Keywords (moderate increase)
```python
MEDIUM_KEYWORDS = {
    "refactor": 0.1, "test": 0.05, "debug": 0.1, "implement": 0.1,
    "feature": 0.1, "module": 0.1, "class": 0.05, "function": 0.0,
    "api": 0.1, "endpoint": 0.1, "database": 0.1, "query": 0.05,
    "validate": 0.05, "error handling": 0.1, "async": 0.1,
}
```

### Reasoning Detection

A task requires reasoning if:
- Prompt length > 200 tokens AND contains conditional logic keywords ("if", "when", "depends", "either")
- Prompt references > 3 files
- Prompt contains comparison/evaluation language ("better", "vs", "which approach")
- Prompt describes a multi-step process ("first... then... finally")

### Scope Assessment

| Signal | Scope Score |
|--------|-------------|
| Single file mentioned | 0.1 |
| 2-3 files mentioned | 0.3 |
| "module" or "package" mentioned | 0.5 |
| "codebase" or "project" mentioned | 0.7 |
| "architecture" or "system" mentioned | 0.9 |
| No file context, vague request | 0.6 |

### Scoring Formula

```python
def compute_score(features: dict[str, float]) -> float:
    weights = {
        "prompt_token_count": 0.15,
        "files_referenced": 0.15,
        "estimated_output_tokens": 0.10,
        "complexity_keywords": 0.25,
        "requires_reasoning": 0.20,
        "scope": 0.15,
    }

    # Normalize features to 0-1 range
    normalized = {
        "prompt_token_count": min(features["prompt_token_count"] / 2000, 1.0),
        "files_referenced": min(features["files_referenced"] / 10, 1.0),
        "estimated_output_tokens": min(features["estimated_output_tokens"] / 5000, 1.0),
        "complexity_keywords": features["complexity_keywords"],  # already 0-1
        "requires_reasoning": features["requires_reasoning"],    # already 0-1
        "scope": features["scope"],                              # already 0-1
    }

    score = sum(normalized[k] * weights[k] for k in weights)
    return max(0.0, min(1.0, score))  # Clamp to [0, 1]
```

### Classification Thresholds

```python
SIMPLE_THRESHOLD = 0.3    # score < 0.3 → SIMPLE
MEDIUM_THRESHOLD = 0.7    # score 0.3-0.7 → MEDIUM
                          # score > 0.7 → COMPLEX
```

Both thresholds are configurable in `~/.prism/config.yaml`.

## Model Selection Algorithm

### Step 1: Get Candidate Models

For each tier, the candidate models (in priority order):

#### Simple Tier Candidates
1. Ollama local (qwen2.5-coder:7b, llama3.2:3b) — $0.00
2. Google AI Studio free tier (Gemini 2.0 Flash) — $0.00
3. Groq free tier (Llama 3.3 70B) — $0.00
4. GPT-4o-mini — $0.15/$0.60 per 1M tokens

#### Medium Tier Candidates
1. DeepSeek V3 — $0.27/$1.10 per 1M tokens
2. Groq (Llama 3.3 70B, paid) — $0.59/$0.79
3. Mistral Small — $0.20/$0.60
4. Gemini 2.0 Flash (paid) — $0.10/$0.40
5. GPT-4o-mini — $0.15/$0.60
6. Ollama large (if available) — $0.00

#### Complex Tier Candidates
1. Claude Sonnet 4 — $3.00/$15.00
2. GPT-4o — $2.50/$10.00
3. Gemini 2.5 Pro — $1.25/$5.00
4. DeepSeek R1 — $0.55/$2.19
5. o3 — $10.00/$40.00 (for reasoning-heavy tasks)
6. Claude Opus 4 — $15.00/$75.00 (for the hardest tasks)

### Step 2: Filter by Availability

Remove candidates where:
- API key not configured
- Provider currently rate-limited (known from recent 429s)
- Provider known to be down (health check failed in last 60s)
- User has excluded the provider in config

### Step 3: Estimate Costs

For each remaining candidate:
```python
estimated_cost = (
    (estimated_input_tokens / 1_000_000) * model.input_cost_per_1m +
    (estimated_output_tokens / 1_000_000) * model.output_cost_per_1m
)
```

### Step 4: Check Budget

Remove candidates whose estimated cost exceeds remaining budget.
If no candidates remain → raise `BudgetExceededError`.

### Step 5: Quality-Cost Ranking

```python
def rank_candidate(model, estimated_cost, success_rate):
    """Higher is better."""
    if estimated_cost == 0:
        # Free models: rank by success rate alone
        return success_rate * 100

    # Paid models: quality per dollar
    quality_weight = 0.7  # Configurable: higher = prefer quality over cost
    cost_weight = 1.0 - quality_weight

    quality_score = success_rate * quality_weight
    cost_score = (1.0 / (estimated_cost + 0.0001)) * cost_weight

    return quality_score * cost_score
```

### Step 6: Build Fallback Chain

```
Primary: Best-ranked candidate
Fallback 1: Second-best in same tier
Fallback 2: Best in next-cheaper tier
Fallback 3: Ollama local (always available if installed)
```

### Step 7: User Overrides

User preferences take priority:
- `prism config set routing.prefer anthropic` → Anthropic always first when available
- `prism config set routing.exclude deepseek` → Never route to DeepSeek
- `prism config set routing.pin_model claude-sonnet-4` → Always use this model
- `/model deepseek-v3` → Override for this session only
- `--model gpt-4o` → Override for this single request

## Adaptive Learning

### Phase 1: Rule-Based (0-100 interactions)
Use the keyword/heuristic classifier described above. No learning.

### Phase 2: Data Collection (100-500 interactions)
Continue using rule-based classifier. Log outcomes for every interaction:
- Prompt features (vector of 6 floats)
- Model used
- Outcome signal:
  - **Accepted**: User didn't re-run or edit the output
  - **Rejected**: User immediately re-ran the same prompt or manually edited
  - **Corrected**: User made minor tweaks to the output

### Phase 3: Learned Routing (500+ interactions)
Train a logistic regression model mapping features → optimal model per tier:

```python
from sklearn.linear_model import LogisticRegression

def retrain_classifier(self):
    data = self.db.get_learning_data(min_entries=100)

    X = [d.features_vector for d in data]  # Feature vectors
    y = [1 if d.outcome == Outcome.ACCEPTED else 0 for d in data]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Save model weights locally
    self._save_model(model)
```

### Exploration Rounds
- 10% of requests randomly routed to a non-default model
- This prevents the system from getting stuck in local optima
- Exploration data feeds back into retraining
- User sees: "Exploring: routed to GPT-4o-mini (usually uses DeepSeek V3)"

## Architect Mode Routing

For complex tasks, split into two phases:

1. **Planning phase**: Route to premium model (Claude Sonnet, GPT-4o)
   - Model creates a step-by-step implementation plan
   - Cost: ~$0.02-0.10 (short output)

2. **Execution phase**: Route each step to cheap model (DeepSeek V3, GPT-4o-mini)
   - Each step executed independently with plan context
   - Cost: ~$0.001-0.005 per step

**Total cost reduction**: 60-80% compared to using premium model for everything.

**Trigger**: Automatically activated when:
- Task classified as COMPLEX
- Estimated output > 3,000 tokens
- User hasn't pinned a specific model

**Override**: `prism config set routing.architect_mode false`

## Rate Limit Handling

### Per-Provider Tracking
```python
@dataclass
class RateLimitState:
    provider: str
    limited_at: datetime
    retry_after: float | None     # From 429 response header
    consecutive_limits: int        # How many times in a row
    backoff_until: datetime        # When to try again
```

### Backoff Strategy
| Consecutive Limits | Backoff Duration |
|-------------------|-----------------|
| 1 | retry_after or 5s |
| 2 | retry_after or 15s |
| 3 | retry_after or 60s |
| 4+ | 300s (5 minutes) |

### Free Tier Awareness
Track remaining quota for free tiers:
- Google AI Studio: 1,500 req/day → track count, don't route when near limit
- Groq free: 14,400 req/day, 30 RPM → track both, respect RPM limit
- When free tier exhausted → automatically upgrade to paid tier for that provider
