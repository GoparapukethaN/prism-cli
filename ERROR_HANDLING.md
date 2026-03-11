# ERROR_HANDLING.md — Prism Error Handling Strategy

## Exception Hierarchy

```python
class PrismError(Exception):
    """Base exception for all Prism errors. All custom exceptions inherit from this."""

# Configuration
class ConfigError(PrismError):
    """Invalid configuration, missing config file, schema violation."""

class ConfigNotFoundError(ConfigError):
    """Config file not found at expected path."""

# Authentication
class AuthError(PrismError):
    """Authentication/credential errors."""

class KeyNotFoundError(AuthError):
    """API key not found for a provider."""

class KeyInvalidError(AuthError):
    """API key failed validation."""

class KeyringUnavailableError(AuthError):
    """OS keyring not available (headless Linux, etc.)."""

# Providers
class ProviderError(PrismError):
    """Provider communication errors."""

class ProviderUnavailableError(ProviderError):
    """Provider API unreachable."""

class ProviderRateLimitError(ProviderError):
    """Rate limit exceeded. Includes retry_after if available."""
    def __init__(self, provider: str, retry_after: float | None = None):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(f"Rate limited by {provider}" +
                        (f", retry after {retry_after}s" if retry_after else ""))

class ProviderAuthError(ProviderError):
    """Provider rejected the API key (401/403)."""

class ProviderQuotaError(ProviderError):
    """Provider quota exhausted (different from rate limit — billing issue)."""

class ModelNotFoundError(ProviderError):
    """Requested model not found or not available."""

# Routing
class RoutingError(PrismError):
    """Routing decision errors."""

class NoModelsAvailableError(RoutingError):
    """No models available for the requested tier."""

class BudgetExceededError(RoutingError):
    """All candidate models exceed remaining budget."""
    def __init__(self, budget_remaining: float, cheapest_estimate: float):
        self.budget_remaining = budget_remaining
        self.cheapest_estimate = cheapest_estimate
        super().__init__(
            f"Budget remaining: ${budget_remaining:.2f}, "
            f"cheapest model: ${cheapest_estimate:.4f}"
        )

class AllProvidersFailed(RoutingError):
    """All models in fallback chain failed."""

# Tools
class ToolError(PrismError):
    """Tool execution errors."""

class ToolNotFoundError(ToolError):
    """Unknown tool name."""

class ToolPermissionDeniedError(ToolError):
    """User denied permission for tool execution."""

class ToolTimeoutError(ToolError):
    """Tool execution exceeded timeout."""

class ToolExecutionError(ToolError):
    """Tool execution failed (non-zero exit, file not found, etc.)."""

# Security
class SecurityError(PrismError):
    """Security violation detected."""

class PathTraversalError(SecurityError):
    """Path escapes project root."""

class BlockedCommandError(SecurityError):
    """Command matches blocked command pattern."""

class ExcludedFileError(SecurityError):
    """File matches excluded pattern (secrets, credentials)."""

# Database
class DatabaseError(PrismError):
    """Database operation errors."""

class MigrationError(DatabaseError):
    """Schema migration failed."""

# Context
class ContextError(PrismError):
    """Context management errors."""

class ContextWindowExceededError(ContextError):
    """Prompt + context exceeds model's context window."""

# Git
class GitError(PrismError):
    """Git operation errors."""

class NotAGitRepoError(GitError):
    """Project directory is not a git repository."""
```

## Error Handling Patterns

### Pattern 1: Provider Fallback
```python
async def route_with_fallback(self, selection: ModelSelection, messages: list) -> CompletionResult:
    """Try primary model, fall back through chain on failure."""
    models = [selection.model_id] + selection.fallback_chain
    last_error: Exception | None = None

    for model_id in models:
        try:
            result = await self._complete(model_id, messages)
            if model_id != selection.model_id:
                logger.info("fallback_used", primary=selection.model_id, fallback=model_id)
                self.ui.notify(f"Routed to {model_id} ({selection.model_id} unavailable)")
            return result

        except ProviderRateLimitError as e:
            logger.warning("rate_limited", model=model_id, retry_after=e.retry_after)
            last_error = e
            continue

        except ProviderUnavailableError as e:
            logger.warning("provider_unavailable", model=model_id, error=str(e))
            last_error = e
            continue

        except ProviderAuthError as e:
            logger.error("auth_failed", model=model_id)
            last_error = e
            continue  # Try next model, this provider's key is bad

        except ProviderQuotaError as e:
            logger.error("quota_exceeded", model=model_id)
            last_error = e
            continue

    raise AllProvidersFailed(
        f"All {len(models)} models failed. Last error: {last_error}"
    ) from last_error
```

### Pattern 2: Budget Guard
```python
async def check_budget(self, estimated_cost: float) -> BudgetAction:
    """Check if a request fits within budget. Returns action to take."""
    remaining = self.cost_tracker.get_budget_remaining()

    if remaining is None:
        return BudgetAction.PROCEED  # No budget set

    if estimated_cost <= remaining:
        return BudgetAction.PROCEED

    if estimated_cost <= remaining * 1.5:  # Within 150%
        return BudgetAction.WARN  # Warn user but allow

    return BudgetAction.BLOCK  # Hard block

class BudgetAction(Enum):
    PROCEED = "proceed"
    WARN = "warn"
    BLOCK = "block"
```

### Pattern 3: Tool Execution with Security
```python
async def execute_tool(self, tool_call: ToolCall, context: ToolContext) -> ToolResult:
    """Execute a tool call with full security checks."""
    try:
        tool = self.registry.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                content="",
                success=False,
                error=f"Unknown tool: {tool_call.name}",
            )

        # Permission check
        if tool.permission_level == PermissionLevel.CONFIRM:
            approved = await self.ui.confirm_tool(tool_call)
            if not approved:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    content="",
                    success=False,
                    error="User denied permission",
                )

        # Execute with timeout
        result = await asyncio.wait_for(
            tool.execute(tool_call.arguments, context),
            timeout=context.timeout,
        )

        # Audit log
        self.audit.log_tool_execution(tool_call, result)

        return result

    except asyncio.TimeoutError:
        return ToolResult(
            tool_call_id=tool_call.id,
            content="",
            success=False,
            error=f"Tool execution timed out after {context.timeout}s",
        )
    except SecurityError as e:
        logger.error("security_violation", tool=tool_call.name, error=str(e))
        return ToolResult(
            tool_call_id=tool_call.id,
            content="",
            success=False,
            error=f"Security violation: {e}",
        )
    except Exception as e:
        logger.error("tool_error", tool=tool_call.name, error=str(e))
        return ToolResult(
            tool_call_id=tool_call.id,
            content="",
            success=False,
            error=f"Tool execution failed: {e}",
        )
```

### Pattern 4: Graceful Degradation
```python
def get_repo_map(self) -> str | None:
    """Get repo map, gracefully degrade if tree-sitter unavailable."""
    try:
        return self.repo_mapper.generate()
    except ImportError:
        logger.warning("tree_sitter_unavailable", msg="Falling back to directory listing")
        return self._simple_directory_listing()
    except Exception as e:
        logger.warning("repo_map_failed", error=str(e))
        return None  # Proceed without repo map
```

## User-Facing Error Messages

### Rules
1. **Never show stack traces** to users (only in --debug mode)
2. **Never show API keys** in error messages
3. **Always suggest a next action** the user can take
4. **Use Rich formatting** for clear, readable errors

### Examples
```python
# Provider error
console.print("[red]Error:[/] Could not connect to Anthropic API.")
console.print("  Check your API key: [cyan]prism auth test anthropic[/]")
console.print("  Check Anthropic status: [cyan]https://status.anthropic.com[/]")

# Budget exceeded
console.print(f"[yellow]Budget limit reached:[/] ${remaining:.2f} remaining of ${limit:.2f}/day")
console.print("  Options:")
console.print("  1. Increase budget: [cyan]prism config set budget.daily_limit 10[/]")
console.print("  2. Use a cheaper model: [cyan]/model ollama/qwen2.5-coder:7b[/]")
console.print("  3. Wait until tomorrow (budget resets at midnight)")

# Security violation
console.print("[red]Security:[/] Cannot access file outside project root.")
console.print(f"  Requested: {path}")
console.print(f"  Project root: {project_root}")

# All providers failed
console.print("[red]All providers failed.[/] Run [cyan]prism status[/] to diagnose.")
```

## Retry Strategy

| Error Type | Retry? | Strategy |
|-----------|--------|----------|
| Rate limit (429) | Yes | Exponential backoff: 1s, 2s, 4s, then fallback |
| Server error (500/502/503) | Yes | 1 retry after 2s, then fallback |
| Auth error (401/403) | No | Immediate fallback to next provider |
| Quota exceeded | No | Immediate fallback |
| Timeout | Yes | 1 retry with 2x timeout, then fallback |
| Network error | Yes | 1 retry after 1s, then fallback |
| Invalid response | No | Log and fallback |
| Budget exceeded | No | User decision required |

## Critical vs Non-Critical Errors

### Critical (stop execution, inform user)
- All providers in fallback chain failed
- Budget exceeded with no cheaper alternative
- Database corruption
- Security violation
- Missing project root

### Non-Critical (log, continue with degradation)
- tree-sitter unavailable → use simple directory listing
- ChromaDB unavailable → skip RAG
- Prompt cache miss → proceed without cache
- Git not available → skip auto-commit
- Single provider rate limit → fallback to next
- Audit log write failed → continue without audit

## Logging Requirements for Errors

Every error MUST be logged with:
1. **Module**: Which module raised the error
2. **Operation**: What was being attempted
3. **Error type**: The exception class name
4. **Error message**: The exception message (sanitized — no secrets)
5. **Context**: Relevant identifiers (model_id, provider, path, etc.)
6. **Action taken**: What happened next (fallback, retry, abort)
