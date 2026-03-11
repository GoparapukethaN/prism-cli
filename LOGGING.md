# LOGGING.md — Prism Logging Strategy

## Logging Library
**structlog** — structured, key-value logging that's both human-readable and machine-parseable.

## Logger Setup

```python
import structlog

def configure_logging(level: str = "WARNING", log_file: str | None = None) -> None:
    """Configure structlog for Prism."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if log_file:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
```

## Log Levels

| Level | When | Example |
|-------|------|---------|
| **DEBUG** | Internal state, detailed operation flow | Feature vector values, cache hits, DB queries |
| **INFO** | Significant events, routing decisions | Model selected, cost tracked, tool executed |
| **WARNING** | Degraded but recoverable conditions | Rate limit hit, fallback activated, budget warning |
| **ERROR** | Failed operations that affect the user | Provider down, tool failed, security violation |
| **CRITICAL** | Unrecoverable conditions | DB corruption, all providers failed |

## Log Categories and Examples

### Router Logs
```python
logger.info("task_classified", tier="medium", score=0.54, features=features)
logger.info("model_selected", model="deepseek-v3", tier="medium", cost_est=0.0018, fallback=["gpt-4o-mini"])
logger.debug("candidates_ranked", candidates=[{"model": "deepseek-v3", "rank": 0.87}, ...])
logger.warning("budget_warning", daily_used=4.12, daily_limit=5.00, percent=82.4)
logger.info("fallback_triggered", from_model="groq/llama-3.3-70b", to_model="deepseek-v3", reason="rate_limited")
```

### Provider Logs
```python
logger.info("provider_healthy", provider="anthropic", latency_ms=234)
logger.warning("provider_rate_limited", provider="groq", retry_after=42.0)
logger.error("provider_auth_failed", provider="openai", status=401)
logger.info("provider_recovered", provider="groq", downtime_seconds=45)
```

### Tool Logs
```python
logger.info("tool_executed", tool="read_file", path="src/main.py", size=4823, duration_ms=12)
logger.info("tool_executed", tool="execute_command", cmd="npm test", exit_code=0, duration_ms=12300)
logger.warning("tool_denied", tool="write_file", path="src/main.py", reason="user_rejected")
logger.error("tool_failed", tool="execute_command", cmd="npm test", error="timeout after 30s")
logger.error("tool_blocked", tool="execute_command", cmd="rm -rf /", reason="dangerous_command")
```

### Cost Logs
```python
logger.info("cost_tracked", model="deepseek-v3", input_tokens=487, output_tokens=1834, cost=0.0023)
logger.info("session_cost", session_id="abc123", total=0.42, requests=23)
logger.warning("budget_exceeded", remaining=0.13, estimated=0.25, action="blocked")
```

### Security Logs
```python
logger.error("path_traversal_attempt", path="../../../etc/passwd", resolved="/etc/passwd")
logger.error("blocked_command", cmd="rm -rf /", pattern="^rm\\s+-rf\\s+/")
logger.warning("excluded_file_access", path=".env", tool="read_file")
logger.info("secret_filtered", env_var="ANTHROPIC_API_KEY", context="subprocess")
```

### Context Logs
```python
logger.info("context_assembled", system_tokens=2000, repo_map_tokens=3500, files_tokens=8000, history_tokens=15000)
logger.info("context_compacted", before_tokens=45000, after_tokens=3000)
logger.debug("repo_map_generated", files=234, entries=1847, duration_ms=3200, cached=False)
logger.debug("repo_map_cached", files=234, cache_age_seconds=45)
```

## CRITICAL: What NEVER to Log

```python
# NEVER log API keys
logger.info("auth", key=api_key)                    # NEVER
logger.info("auth", provider="anthropic")            # OK

# NEVER log passwords
logger.info("decrypt", passphrase=passphrase)        # NEVER
logger.info("decrypt", success=True)                 # OK

# NEVER log raw prompts (privacy)
logger.info("classify", prompt=user_prompt)          # NEVER
logger.info("classify", prompt_hash=hash, tokens=n)  # OK

# NEVER log file contents
logger.info("read", content=file_content)            # NEVER
logger.info("read", path=path, size=len(content))    # OK

# NEVER log sensitive env vars
logger.debug("env", env=os.environ)                  # NEVER
logger.debug("env", filtered_count=5)                # OK
```

## Audit Log

Separate from application logging. Dedicated to tracking tool executions for accountability.

### Location
`~/.prism/audit.log`

### Format
```
[2026-03-10T14:23:01Z] TOOL=read_file PATH=src/main.py RESULT=success SIZE=4823
[2026-03-10T14:23:05Z] TOOL=execute_command CMD="npm test" RESULT=success EXIT=0 DURATION=12.3s
[2026-03-10T14:23:18Z] TOOL=write_file PATH=src/main.py RESULT=success DIFF=+5/-3
[2026-03-10T14:23:20Z] TOOL=browse_web URL=https://docs.python.org RESULT=success SIZE=45KB
[2026-03-10T14:23:25Z] TOOL=write_file PATH=.env RESULT=blocked REASON=excluded_pattern
[2026-03-10T14:24:01Z] ROUTE MODEL=deepseek-v3 TIER=medium COST=$0.0012 IN=450 OUT=1200
```

### Rotation
- Max file size: 10MB
- Keep 5 rotated files: `audit.log`, `audit.log.1`, ..., `audit.log.5`
- Total max audit storage: 60MB

### Implementation
```python
class AuditLogger:
    def __init__(self, log_path: Path, max_bytes: int = 10_000_000, backup_count: int = 5):
        self.handler = RotatingFileHandler(
            str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
        )

    def log_tool(self, tool_name: str, **kwargs) -> None:
        """Log a tool execution."""
        sanitized = self._sanitize(kwargs)
        entry = f"TOOL={tool_name} " + " ".join(f"{k}={v}" for k, v in sanitized.items())
        self._write(entry)

    def log_route(self, model: str, tier: str, cost: float, in_tokens: int, out_tokens: int) -> None:
        """Log a routing decision."""
        self._write(f"ROUTE MODEL={model} TIER={tier} COST=${cost:.4f} IN={in_tokens} OUT={out_tokens}")

    def _sanitize(self, data: dict) -> dict:
        """Remove any sensitive data before logging."""
        sanitized = {}
        for k, v in data.items():
            if any(secret in k.lower() for secret in ["key", "secret", "token", "password"]):
                continue
            sanitized[k.upper()] = v
        return sanitized
```

## User-Facing Output vs Logs

| Information | User sees (Rich console) | Logged (structlog) | Audited |
|-------------|-------------------------|-------------------|---------|
| Model selection | "Using DeepSeek V3" | Full routing details | Yes |
| Tool execution | Diff display, command output | Tool name + metadata | Yes |
| Cost | Dashboard on `/cost` | Every cost entry | Yes |
| Errors | Friendly message + suggestion | Full error + stack trace | No |
| Rate limits | "Switched to DeepSeek" | Full rate limit details | Yes |
| Security blocks | "Cannot access that file" | Full security event | Yes |

## Debugging with Logs

```bash
# See verbose output in terminal
prism --verbose

# See full debug output
prism --debug

# Write debug logs to file (doesn't affect terminal output)
PRISM_LOG_FILE=~/.prism/debug.log PRISM_LOG_LEVEL=DEBUG prism

# Review audit trail
cat ~/.prism/audit.log | tail -50

# Search for specific events
grep "TOOL=execute_command" ~/.prism/audit.log
grep "ROUTE.*TIER=complex" ~/.prism/audit.log
```
