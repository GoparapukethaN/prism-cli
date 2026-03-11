# DATA_MODELS.md — Prism Data Models and Database Schema

## SQLite Database Location
`~/.prism/prism.db`

## Schema Version
Current: 1

## Tables

### routing_decisions
Tracks every routing decision made by the classifier and selector.

```sql
CREATE TABLE routing_decisions (
    id TEXT PRIMARY KEY,                    -- UUID
    created_at TEXT NOT NULL,               -- ISO 8601 timestamp
    session_id TEXT NOT NULL,               -- Session identifier
    prompt_hash TEXT NOT NULL,              -- SHA-256 hash of prompt (not raw prompt)
    complexity_tier TEXT NOT NULL,          -- 'simple', 'medium', 'complex'
    complexity_score REAL NOT NULL,         -- 0.0 to 1.0
    model_selected TEXT NOT NULL,           -- LiteLLM model ID
    model_actual TEXT,                      -- Actual model used (if fallback)
    fallback_chain TEXT NOT NULL,           -- JSON array of model IDs
    estimated_cost REAL NOT NULL,           -- Estimated cost in USD
    actual_cost REAL,                       -- Actual cost after completion
    input_tokens INTEGER,                   -- Actual input tokens
    output_tokens INTEGER,                  -- Actual output tokens
    cached_tokens INTEGER DEFAULT 0,        -- Tokens from cache
    latency_ms REAL,                        -- Total latency in milliseconds
    outcome TEXT DEFAULT 'unknown',         -- 'accepted', 'rejected', 'corrected', 'unknown'
    features TEXT NOT NULL,                 -- JSON object of extracted features
    error TEXT,                             -- Error message if failed
    created_date TEXT GENERATED ALWAYS AS (date(created_at)) STORED  -- For daily aggregation
);

CREATE INDEX idx_routing_created_at ON routing_decisions(created_at);
CREATE INDEX idx_routing_created_date ON routing_decisions(created_date);
CREATE INDEX idx_routing_model ON routing_decisions(model_selected);
CREATE INDEX idx_routing_tier ON routing_decisions(complexity_tier);
CREATE INDEX idx_routing_session ON routing_decisions(session_id);
CREATE INDEX idx_routing_outcome ON routing_decisions(outcome);
```

### cost_entries
Individual cost records for every API call.

```sql
CREATE TABLE cost_entries (
    id TEXT PRIMARY KEY,                    -- UUID
    created_at TEXT NOT NULL,               -- ISO 8601 timestamp
    session_id TEXT NOT NULL,
    model_id TEXT NOT NULL,                 -- LiteLLM model ID
    provider TEXT NOT NULL,                 -- Provider name
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cached_tokens INTEGER DEFAULT 0,
    cost_usd REAL NOT NULL,                -- Calculated cost in USD
    complexity_tier TEXT NOT NULL,
    created_date TEXT GENERATED ALWAYS AS (date(created_at)) STORED
);

CREATE INDEX idx_cost_created_at ON cost_entries(created_at);
CREATE INDEX idx_cost_created_date ON cost_entries(created_date);
CREATE INDEX idx_cost_session ON cost_entries(session_id);
CREATE INDEX idx_cost_model ON cost_entries(model_id);
CREATE INDEX idx_cost_provider ON cost_entries(provider);
```

### sessions
Session metadata.

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,                    -- UUID
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    project_root TEXT NOT NULL,             -- Project directory path
    total_cost REAL DEFAULT 0.0,
    total_requests INTEGER DEFAULT 0,
    summary TEXT,                           -- Compressed summary (after /compact)
    active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_sessions_created_at ON sessions(created_at);
CREATE INDEX idx_sessions_project ON sessions(project_root);
```

### provider_status
Provider health and rate limit tracking.

```sql
CREATE TABLE provider_status (
    provider TEXT PRIMARY KEY,
    last_check_at TEXT,
    is_available BOOLEAN DEFAULT TRUE,
    last_error TEXT,
    rate_limited_until TEXT,                -- ISO 8601 timestamp
    consecutive_failures INTEGER DEFAULT 0,
    free_tier_requests_today INTEGER DEFAULT 0,
    free_tier_reset_at TEXT                 -- When daily counter resets
);
```

### budget_config
Budget configuration and tracking.

```sql
CREATE TABLE budget_config (
    id TEXT PRIMARY KEY DEFAULT 'default',
    daily_limit REAL,                       -- NULL = no limit
    monthly_limit REAL,                     -- NULL = no limit
    warn_at_percent REAL DEFAULT 80.0,
    updated_at TEXT NOT NULL
);
```

### tool_executions
Audit log of all tool executions (mirrors audit.log but queryable).

```sql
CREATE TABLE tool_executions (
    id TEXT PRIMARY KEY,                    -- UUID
    created_at TEXT NOT NULL,
    session_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments TEXT NOT NULL,                -- JSON (sanitized — no secrets)
    result_success BOOLEAN NOT NULL,
    result_error TEXT,
    duration_ms REAL,
    metadata TEXT                           -- JSON (extra info per tool type)
);

CREATE INDEX idx_tools_created_at ON tool_executions(created_at);
CREATE INDEX idx_tools_session ON tool_executions(session_id);
CREATE INDEX idx_tools_name ON tool_executions(tool_name);
```

### schema_version
Database migration tracking.

```sql
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT
);
```

## Pydantic Models (Application Layer)

### Configuration Models

```python
class Settings(BaseModel):
    """Global Prism settings."""
    prism_home: Path = Path.home() / ".prism"
    db_path: Path | None = None  # Default: prism_home / "prism.db"

    # Routing
    simple_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    medium_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    exploration_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    architect_mode: bool = True

    # Budget
    daily_budget: float | None = None
    monthly_budget: float | None = None
    budget_warn_percent: float = Field(default=80.0, ge=0.0, le=100.0)

    # Tools
    web_enabled: bool = False
    auto_approve: bool = False
    command_timeout: int = Field(default=30, ge=1, le=300)
    allowed_commands: list[str] = Field(default_factory=list)

    # Providers
    excluded_providers: list[str] = Field(default_factory=list)
    preferred_provider: str | None = None
    pinned_model: str | None = None

    @model_validator(mode="after")
    def validate_thresholds(self) -> "Settings":
        if self.simple_threshold >= self.medium_threshold:
            raise ValueError("simple_threshold must be less than medium_threshold")
        return self
```

### Provider Models

```python
class ModelInfo(BaseModel):
    """Information about a specific AI model."""
    id: str
    display_name: str
    provider: str
    tier: ComplexityTier
    input_cost_per_1m: float = Field(ge=0.0)
    output_cost_per_1m: float = Field(ge=0.0)
    context_window: int = Field(gt=0)
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    max_output_tokens: int = 4096

class ProviderAuthStatus(BaseModel):
    """Status of a provider's authentication."""
    provider: str
    display_name: str
    configured: bool
    validated: bool | None = None  # None = not checked yet
    available_models: list[str] = Field(default_factory=list)
    error: str | None = None
```

### Cost Models

```python
class CostSummary(BaseModel):
    """Aggregated cost data for a time period."""
    period: Literal["session", "day", "month"]
    total_cost: float
    total_requests: int
    model_breakdown: list[ModelCostBreakdown]
    budget_limit: float | None = None
    budget_remaining: float | None = None

class ModelCostBreakdown(BaseModel):
    """Cost breakdown for a single model."""
    model_id: str
    display_name: str
    request_count: int
    total_cost: float
    percentage: float  # Of total cost
```

### Routing Models

```python
class ClassificationResult(BaseModel):
    """Result of task classification."""
    tier: ComplexityTier
    score: float = Field(ge=0.0, le=1.0)
    features: dict[str, float]
    reasoning: str  # Human-readable explanation

class ModelCandidate(BaseModel):
    """A candidate model during selection."""
    model_info: ModelInfo
    estimated_cost: float
    success_rate: float  # Historical success rate (0-1)
    rank_score: float    # Combined quality/cost score
    available: bool
    unavailable_reason: str | None = None
```

## Migration Strategy

### Forward-Only Migrations
Migrations are numbered sequentially and run exactly once:

```python
MIGRATIONS = {
    1: Migration(
        version=1,
        description="Initial schema",
        sql="""
            CREATE TABLE routing_decisions (...);
            CREATE TABLE cost_entries (...);
            CREATE TABLE sessions (...);
            CREATE TABLE provider_status (...);
            CREATE TABLE budget_config (...);
            CREATE TABLE tool_executions (...);
            CREATE TABLE schema_version (...);
        """,
    ),
    # Future migrations:
    # 2: Migration(version=2, description="Add learning weights table", sql="..."),
}
```

### Migration Rules
- Never modify existing migrations
- Never delete columns (add new ones, deprecate old)
- Always add indexes for columns used in WHERE/ORDER BY
- Test migration on a copy of production data before release
- Backup database before running migrations
