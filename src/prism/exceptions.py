"""Prism exception hierarchy.

All custom exceptions inherit from PrismError.
Modules catch provider-specific exceptions and wrap them in Prism exceptions.
"""

from __future__ import annotations


class PrismError(Exception):
    """Base exception for all Prism errors."""


# --- Configuration ---


class ConfigError(PrismError):
    """Invalid configuration, missing config file, or schema violation."""


class ConfigNotFoundError(ConfigError):
    """Configuration file not found at expected path."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Config file not found: {path}")


# --- Authentication ---


class AuthError(PrismError):
    """Authentication or credential errors."""


class KeyNotFoundError(AuthError):
    """API key not found for a provider."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"No API key found for provider: {provider}")


class KeyInvalidError(AuthError):
    """API key failed validation."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"API key validation failed for provider: {provider}")


class KeyringUnavailableError(AuthError):
    """OS keyring not available (headless Linux, Docker, etc.)."""

    def __init__(self, reason: str = "") -> None:
        self.reason = reason
        msg = "OS keyring is not available"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


# --- Providers ---


class ProviderError(PrismError):
    """Provider communication errors."""


class ProviderUnavailableError(ProviderError):
    """Provider API is unreachable."""

    def __init__(self, provider: str, reason: str = "") -> None:
        self.provider = provider
        msg = f"Provider unavailable: {provider}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class ProviderRateLimitError(ProviderError):
    """Rate limit exceeded for a provider."""

    def __init__(self, provider: str, retry_after: float | None = None) -> None:
        self.provider = provider
        self.retry_after = retry_after
        msg = f"Rate limited by {provider}"
        if retry_after is not None:
            msg += f", retry after {retry_after:.1f}s"
        super().__init__(msg)


class ProviderAuthError(ProviderError):
    """Provider rejected the API key (401/403)."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"Authentication failed for provider: {provider}")


class ProviderQuotaError(ProviderError):
    """Provider quota exhausted (billing issue, not rate limit)."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"Quota exhausted for provider: {provider}")


class ModelNotFoundError(ProviderError):
    """Requested model not found or not available."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"Model not found: {model_id}")


# --- Routing ---


class RoutingError(PrismError):
    """Routing decision errors."""


class NoModelsAvailableError(RoutingError):
    """No models available for the requested tier."""

    def __init__(self, tier: str) -> None:
        self.tier = tier
        super().__init__(f"No models available for tier: {tier}")


class BudgetExceededError(RoutingError):
    """All candidate models exceed remaining budget."""

    def __init__(self, budget_remaining: float, cheapest_estimate: float) -> None:
        self.budget_remaining = budget_remaining
        self.cheapest_estimate = cheapest_estimate
        super().__init__(
            f"Budget remaining: ${budget_remaining:.2f}, "
            f"cheapest model estimate: ${cheapest_estimate:.4f}"
        )


class AllProvidersFailedError(RoutingError):
    """All models in the fallback chain failed."""

    def __init__(self, tried_models: list[str], last_error: str = "") -> None:
        self.tried_models = tried_models
        self.last_error = last_error
        models_str = ", ".join(tried_models)
        msg = f"All models failed: [{models_str}]"
        if last_error:
            msg += f". Last error: {last_error}"
        super().__init__(msg)


# --- Tools ---


class ToolError(PrismError):
    """Tool execution errors."""


class ToolNotFoundError(ToolError):
    """Unknown tool name."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Unknown tool: {tool_name}")


class ToolPermissionDeniedError(ToolError):
    """User denied permission for tool execution."""

    def __init__(self, tool_name: str, action: str = "") -> None:
        self.tool_name = tool_name
        self.action = action
        msg = f"Permission denied for tool: {tool_name}"
        if action:
            msg += f" (action: {action})"
        super().__init__(msg)


class ToolTimeoutError(ToolError):
    """Tool execution exceeded timeout."""

    def __init__(self, tool_name: str, timeout: float) -> None:
        self.tool_name = tool_name
        self.timeout = timeout
        super().__init__(f"Tool '{tool_name}' timed out after {timeout:.1f}s")


class ToolExecutionError(ToolError):
    """Tool execution failed."""

    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Tool '{tool_name}' failed: {reason}")


# --- Security ---


class SecurityError(PrismError):
    """Security violation detected."""


class PathTraversalError(SecurityError):
    """Path escapes project root."""

    def __init__(self, path: str, project_root: str) -> None:
        self.path = path
        self.project_root = project_root
        super().__init__(f"Path '{path}' escapes project root '{project_root}'")


class BlockedCommandError(SecurityError):
    """Command matches a blocked pattern."""

    def __init__(self, command: str, pattern: str) -> None:
        self.command = command
        self.pattern = pattern
        super().__init__(f"Command blocked by security rule: {pattern}")


class ExcludedFileError(SecurityError):
    """File matches an excluded pattern (secrets, credentials)."""

    def __init__(self, path: str, pattern: str) -> None:
        self.path = path
        self.pattern = pattern
        super().__init__(f"File '{path}' matches excluded pattern: {pattern}")


# --- Database ---


class DatabaseError(PrismError):
    """Database operation errors."""


class MigrationError(DatabaseError):
    """Schema migration failed."""

    def __init__(self, version: int, reason: str) -> None:
        self.version = version
        self.reason = reason
        super().__init__(f"Migration to version {version} failed: {reason}")


# --- Context ---


class ContextError(PrismError):
    """Context management errors."""


class ContextWindowExceededError(ContextError):
    """Prompt plus context exceeds model's context window."""

    def __init__(self, total_tokens: int, max_tokens: int) -> None:
        self.total_tokens = total_tokens
        self.max_tokens = max_tokens
        super().__init__(
            f"Context window exceeded: {total_tokens} tokens "
            f"(max: {max_tokens})"
        )


# --- Git ---


class GitError(PrismError):
    """Git operation errors."""


class NotAGitRepoError(GitError):
    """Project directory is not a git repository."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Not a git repository: {path}")
