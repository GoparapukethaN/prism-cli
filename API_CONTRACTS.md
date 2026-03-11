# API_CONTRACTS.md — Prism Internal API Contracts

## Overview

This document defines the contracts between Prism's internal modules. Every cross-module function call must conform to these interfaces. This ensures modules can be developed, tested, and refactored independently.

## Core Data Types

### ComplexityTier (Enum)
```python
class ComplexityTier(str, Enum):
    SIMPLE = "simple"      # Score < 0.3
    MEDIUM = "medium"      # Score 0.3-0.7
    COMPLEX = "complex"    # Score > 0.7
```

### TaskContext (Dataclass)
```python
@dataclass(frozen=True)
class TaskContext:
    prompt: str
    active_files: list[str]            # Paths relative to project root
    conversation_history: list[Message] # Previous turns
    project_root: Path
    repo_map: str | None = None        # Compressed codebase view
    project_memory: str | None = None  # .prism.md contents
```

### Message (Dataclass)
```python
@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_results: list[ToolResult] | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    model: str | None = None           # Which model generated this
    cost: float | None = None          # Cost of this message
```

### ModelSelection (Dataclass)
```python
@dataclass(frozen=True)
class ModelSelection:
    model_id: str                      # LiteLLM model identifier
    provider: str                      # Provider name
    tier: ComplexityTier               # Why this model was chosen
    estimated_cost: float              # Estimated cost in USD
    fallback_chain: list[str]          # Ordered list of fallback model_ids
    reasoning: str                     # Human-readable explanation
```

### CompletionResult (Dataclass)
```python
@dataclass
class CompletionResult:
    content: str                       # Model's text response
    tool_calls: list[ToolCall]         # Tool calls requested by model
    model_id: str                      # Actual model used (may differ from requested if fallback)
    input_tokens: int
    output_tokens: int
    cost: float                        # Actual cost in USD
    latency_ms: float                  # Time to complete
    cached_tokens: int = 0             # Tokens served from cache
    finish_reason: str = "stop"
```

### ToolCall (Dataclass)
```python
@dataclass(frozen=True)
class ToolCall:
    id: str                            # Unique tool call ID
    name: str                          # Tool name (e.g., "read_file")
    arguments: dict[str, Any]          # Tool parameters
```

### ToolResult (Dataclass)
```python
@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str                  # Matches ToolCall.id
    content: str                       # Tool output
    success: bool
    error: str | None = None
```

### CostEntry (Dataclass)
```python
@dataclass(frozen=True)
class CostEntry:
    id: str                            # UUID
    timestamp: datetime
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    session_id: str
    complexity_tier: ComplexityTier
    cached_tokens: int = 0
```

### RoutingDecision (Dataclass)
```python
@dataclass(frozen=True)
class RoutingDecision:
    id: str                            # UUID
    timestamp: datetime
    prompt_hash: str                   # SHA-256 of prompt (not raw prompt)
    complexity_tier: ComplexityTier
    complexity_score: float
    model_selected: str
    fallback_chain: list[str]
    estimated_cost: float
    actual_cost: float | None = None   # Filled after completion
    outcome: Outcome | None = None     # Filled after user feedback
    features: dict[str, float] = field(default_factory=dict)
```

### Outcome (Enum)
```python
class Outcome(str, Enum):
    ACCEPTED = "accepted"              # User accepted output
    REJECTED = "rejected"              # User re-ran or manually edited
    CORRECTED = "corrected"            # User made minor corrections
    UNKNOWN = "unknown"                # No signal collected
```

## Module Interfaces

### TaskClassifier
```python
class TaskClassifier:
    def __init__(self, config: Settings) -> None: ...

    def classify(self, prompt: str, context: TaskContext) -> ComplexityTier:
        """Classify a prompt into a complexity tier.

        Args:
            prompt: The user's input text.
            context: Current task context.

        Returns:
            The classified complexity tier.

        Raises:
            Never raises — defaults to MEDIUM on any internal error.
        """

    def extract_features(self, prompt: str, context: TaskContext) -> dict[str, float]:
        """Extract feature vector from prompt for classification.

        Returns dict with keys:
            prompt_token_count, files_referenced, estimated_output_tokens,
            complexity_keywords, requires_reasoning, scope
        """

    def get_score(self, prompt: str, context: TaskContext) -> float:
        """Get raw complexity score (0.0 to 1.0) for a prompt."""
```

### ModelSelector
```python
class ModelSelector:
    def __init__(
        self,
        config: Settings,
        provider_registry: ProviderRegistry,
        cost_estimator: CostEstimator,
        db: Database,
    ) -> None: ...

    async def select(
        self,
        tier: ComplexityTier,
        context: TaskContext,
        budget_remaining: float,
    ) -> ModelSelection:
        """Select the optimal model for a task.

        Args:
            tier: Classified complexity tier.
            context: Current task context.
            budget_remaining: Remaining budget in USD.

        Returns:
            ModelSelection with chosen model and fallback chain.

        Raises:
            BudgetExceededError: No model fits within budget.
            RoutingError: No models available for the tier.
        """
```

### CostEstimator
```python
class CostEstimator:
    def __init__(self, pricing: dict[str, ModelPricing]) -> None: ...

    def estimate(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost in USD for a model call.

        Returns:
            Estimated cost in USD (always >= 0).

        Raises:
            ValueError: Unknown model_id.
        """

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""

    def estimate_output_tokens(self, prompt: str, task_type: str) -> int:
        """Estimate expected output tokens for a prompt."""
```

### ProviderRegistry
```python
class ProviderRegistry:
    def __init__(self, config: Settings, auth: AuthManager) -> None: ...

    def get_available_models(self, tier: ComplexityTier) -> list[ModelInfo]:
        """Get models available for a tier (API key configured, not rate-limited).

        Returns:
            List of ModelInfo sorted by default preference.
        """

    def is_available(self, provider: str) -> bool:
        """Check if a provider is configured and reachable."""

    async def health_check(self, provider: str) -> ProviderStatus:
        """Test provider connectivity with a minimal API call."""

    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get metadata for a specific model.

        Raises:
            ValueError: Unknown model_id.
        """
```

### Tool Interface
```python
class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]: ...

    @property
    @abstractmethod
    def permission_level(self) -> PermissionLevel: ...

    @abstractmethod
    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult: ...

class PermissionLevel(str, Enum):
    AUTO = "auto"           # No confirmation needed
    CONFIRM = "confirm"     # Show action, ask yes/no
    DANGEROUS = "dangerous" # Always require explicit confirmation
```

### Database
```python
class Database:
    def __init__(self, path: Path) -> None: ...

    def initialize(self) -> None:
        """Create tables and run migrations."""

    def save_routing_decision(self, decision: RoutingDecision) -> None: ...
    def save_cost_entry(self, entry: CostEntry) -> None: ...
    def update_outcome(self, decision_id: str, outcome: Outcome) -> None: ...

    def get_cost_summary(
        self, period: Literal["session", "day", "month"]
    ) -> CostSummary: ...

    def get_model_success_rate(
        self, model_id: str, tier: ComplexityTier
    ) -> float: ...

    def get_routing_history(self, limit: int = 100) -> list[RoutingDecision]: ...

    def get_learning_data(self, min_entries: int = 100) -> list[RoutingDecision]: ...
```

### AuthManager
```python
class AuthManager:
    def __init__(self, config: Settings) -> None: ...

    def get_key(self, provider: str) -> str | None:
        """Get API key for a provider. Checks keyring → env → encrypted file.

        Returns:
            The API key string, or None if not configured.

        Never raises — returns None on any error.
        """

    async def validate_key(self, provider: str, key: str) -> bool:
        """Validate an API key with a minimal API call.

        NOTE: In development, this always returns True (mocked).
        Only real validation happens in production with user's explicit consent.
        """

    def store_key(self, provider: str, key: str) -> None:
        """Store an API key in the preferred storage backend.

        Raises:
            AuthError: If storage fails.
        """

    def remove_key(self, provider: str) -> None:
        """Remove a stored API key."""

    def list_configured(self) -> list[ProviderAuthStatus]: ...
```

### PathGuard
```python
class PathGuard:
    def __init__(self, project_root: Path, excluded_patterns: list[str]) -> None: ...

    def validate(self, path: str | Path) -> Path:
        """Validate and resolve a path within the project root.

        Args:
            path: The path to validate (relative or absolute).

        Returns:
            The resolved absolute Path.

        Raises:
            SecurityError: If path escapes project root or matches excluded pattern.
        """

    def is_excluded(self, path: Path) -> bool:
        """Check if a path matches any excluded pattern."""
```

### CostTracker
```python
class CostTracker:
    def __init__(self, db: Database, config: Settings) -> None: ...

    def track(self, result: CompletionResult, session_id: str, tier: ComplexityTier) -> CostEntry:
        """Record a completion's cost. Returns the created entry."""

    def get_session_cost(self, session_id: str) -> float: ...
    def get_daily_cost(self) -> float: ...
    def get_monthly_cost(self) -> float: ...
    def get_budget_remaining(self) -> float | None:
        """Returns remaining budget, or None if no budget set."""

    def format_dashboard(self) -> str:
        """Format the cost dashboard for display."""
```

## Error Contract

All module errors inherit from `PrismError`. Modules MUST:
1. Catch provider-specific exceptions and wrap in Prism exceptions
2. Include context in error messages (model, provider, path, etc.)
3. Never include secrets in error messages
4. Use exception chaining (`raise ... from e`)

```python
# Example error flow:
# LiteLLM raises litellm.RateLimitError
# → router catches, wraps in ProviderError with context
# → CLI catches ProviderError, shows user-friendly message + triggers fallback
```

## Thread Safety

- `Database`: thread-safe (one connection per thread via threading.local)
- `CostTracker`: thread-safe (uses Database)
- `ProviderRegistry`: read-only after init, thread-safe
- `TaskClassifier`: stateless, thread-safe
- `PathGuard`: stateless, thread-safe
- `AuthManager`: thread-safe (keyring is thread-safe)
