"""Default configuration values for Prism."""

from __future__ import annotations

from prism.config.schema import (
    BudgetConfig,
    ExcludedPatternsConfig,
    PrismConfig,
    RoutingConfig,
    ToolsConfig,
)

DEFAULT_ROUTING = RoutingConfig(
    simple_threshold=0.3,
    medium_threshold=0.7,
    exploration_rate=0.1,
    architect_mode=True,
    quality_weight=0.7,
)

DEFAULT_BUDGET = BudgetConfig(
    daily_limit=None,
    monthly_limit=None,
    warn_at_percent=80.0,
)

DEFAULT_TOOLS = ToolsConfig(
    web_enabled=False,
    auto_approve=False,
    command_timeout=30,
    max_output_bytes=102400,
    max_error_bytes=10240,
    allowed_commands=[],
)

DEFAULT_EXCLUDED_PATTERNS = ExcludedPatternsConfig()

DEFAULT_SETTINGS = PrismConfig(
    routing=DEFAULT_ROUTING,
    budget=DEFAULT_BUDGET,
    tools=DEFAULT_TOOLS,
    excluded_patterns=DEFAULT_EXCLUDED_PATTERNS,
)

# Blocked commands that can NEVER be overridden
BLOCKED_COMMAND_PATTERNS: list[str] = [
    r"^rm\s+-rf\s+/$",
    r"^rm\s+-rf\s+~",
    r"^rm\s+-rf\s+/\*",
    r"^:\(\)\{.*\}",
    r"^mkfs\.",
    r"^dd\s+if=/dev/zero",
    r"^chmod\s+-R\s+777",
    r"^sudo\s+rm",
    r"\|\s*sh\s*$",
    r"\|\s*bash\s*$",
]

# Environment variable names for sensitive data that must be filtered
SENSITIVE_ENV_PATTERNS: list[str] = [
    "*API_KEY*",
    "*_SECRET",
    "*SECRET*",
    "*_TOKEN",
    "*TOKEN*",
    "*_PASSWORD",
    "*PASSWORD*",
    "*_CREDENTIAL",
    "*_PRIVATE_KEY",
    "DATABASE_URL",
    "REDIS_URL",
]

# Files that are ALWAYS blocked from tool access regardless of config
ALWAYS_BLOCKED_PATTERNS: list[str] = [
    "**/.ssh/id_*",
    "**/.ssh/known_hosts",
    "**/.aws/credentials",
    "**/.azure/credentials",
    "**/.gcloud/credentials.db",
    "**/.gnupg/**",
]

# Maximum file size for read operations (bytes)
MAX_FILE_READ_BYTES: int = 1_048_576  # 1MB

# Maximum entries in repo map
MAX_REPO_MAP_ENTRIES: int = 5000

# Maximum search results
MAX_SEARCH_RESULTS: int = 50

# Session and data retention
SESSION_RETENTION_DAYS: int = 30
ROUTING_RETENTION_DAYS: int = 90
COST_RETENTION_DAYS: int = 365

# Audit log rotation
AUDIT_LOG_MAX_BYTES: int = 10_000_000  # 10MB
AUDIT_LOG_BACKUP_COUNT: int = 5

# Free tier limits
FREE_TIER_LIMITS: dict[str, dict[str, int]] = {
    "google": {
        "requests_per_day": 1500,
        "requests_per_minute": 15,
    },
    "groq": {
        "requests_per_day": 14400,
        "requests_per_minute": 30,
    },
}

# Provider health check timeout (seconds)
HEALTH_CHECK_TIMEOUT: float = 5.0

# Provider health check cache duration (seconds)
HEALTH_CHECK_CACHE_SECONDS: int = 60
