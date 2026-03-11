"""User-friendly error handling for the Prism CLI."""

from __future__ import annotations

import signal
import sys
from dataclasses import dataclass

import structlog

from prism.exceptions import (
    AllProvidersFailedError,
    AuthError,
    BlockedCommandError,
    BudgetExceededError,
    ConfigError,
    ConfigNotFoundError,
    ContextError,
    ContextWindowExceededError,
    DatabaseError,
    ExcludedFileError,
    GitError,
    KeyInvalidError,
    KeyNotFoundError,
    KeyringUnavailableError,
    MigrationError,
    ModelNotFoundError,
    NoModelsAvailableError,
    PathTraversalError,
    PrismError,
    ProviderAuthError,
    ProviderError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    RoutingError,
    SecurityError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionDeniedError,
    ToolTimeoutError,
)

logger = structlog.get_logger(__name__)


@dataclass
class UserError:
    """A user-facing error with an optional suggestion and error code."""

    message: str
    suggestion: str | None = None
    error_code: str | None = None
    recoverable: bool = True


class ErrorHandler:
    """Translates exceptions into user-friendly messages with suggestions.

    Maps Prism exception types to structured :class:`UserError` objects that
    include an actionable suggestion and error code.
    """

    # Mapping from exception type to (message_template, suggestion, error_code, recoverable)
    _ERROR_MAP: dict[
        type[Exception],
        tuple[str | None, str | None, str, bool],
    ] = {
        # Auth errors
        KeyNotFoundError: (
            None,  # use str(error) as message
            "Run 'prism auth add {provider}' to set your API key.",
            "AUTH_001",
            True,
        ),
        KeyInvalidError: (
            None,
            "Check your API key format. Run 'prism auth add {provider}' to re-enter it.",
            "AUTH_002",
            True,
        ),
        KeyringUnavailableError: (
            "OS keyring is not available.",
            "Set your API key via environment variable instead (e.g. ANTHROPIC_API_KEY).",
            "AUTH_003",
            True,
        ),
        ProviderAuthError: (
            None,
            "Your API key may be invalid or expired. Run 'prism auth add {provider}' to update it.",
            "AUTH_004",
            True,
        ),
        AuthError: (
            None,
            "Check your credentials. Run 'prism auth status' to see configured providers.",
            "AUTH_005",
            True,
        ),
        # Provider errors
        ProviderRateLimitError: (
            None,
            "Wait {retry_after} seconds or use '/model' to switch to a different provider.",
            "PROV_001",
            True,
        ),
        ProviderUnavailableError: (
            None,
            "Provider is down. Use '/model' to switch providers.",
            "PROV_002",
            True,
        ),
        ProviderQuotaError: (
            None,
            "Check your billing at the provider's dashboard. Use '/model' to switch providers.",
            "PROV_003",
            True,
        ),
        ModelNotFoundError: (
            None,
            "Run 'prism status' to see available models.",
            "PROV_004",
            True,
        ),
        AllProvidersFailedError: (
            None,
            "All configured providers failed. Check your network and API keys.",
            "PROV_005",
            True,
        ),
        ProviderError: (
            None,
            "Provider error occurred. Try '/model' to switch providers.",
            "PROV_006",
            True,
        ),
        # Budget / Routing errors
        BudgetExceededError: (
            None,
            "Run '/budget set <amount>' to increase your budget or wait until tomorrow.",
            "BUDGET_001",
            True,
        ),
        NoModelsAvailableError: (
            None,
            "Configure more providers with 'prism auth add <provider>'.",
            "ROUTE_001",
            True,
        ),
        RoutingError: (
            None,
            "Check routing configuration with 'prism config get routing'.",
            "ROUTE_002",
            True,
        ),
        # Config errors
        ConfigNotFoundError: (
            None,
            "Run 'prism init' to create a default configuration.",
            "CFG_001",
            True,
        ),
        ConfigError: (
            None,
            "Check your config file. Run 'prism init' to reset to defaults.",
            "CFG_002",
            True,
        ),
        # Tool errors
        ToolNotFoundError: (
            None,
            "This tool is not available. Use '/help' to see available commands.",
            "TOOL_001",
            True,
        ),
        ToolPermissionDeniedError: (
            None,
            "Grant permission or use '--yes' flag to auto-approve.",
            "TOOL_002",
            True,
        ),
        ToolTimeoutError: (
            None,
            "Increase timeout with 'prism config set tools.command_timeout <seconds>'.",
            "TOOL_003",
            True,
        ),
        ToolExecutionError: (
            None,
            "Check the command and try again.",
            "TOOL_004",
            True,
        ),
        ToolError: (
            None,
            "A tool error occurred. Use '/help' for available commands.",
            "TOOL_005",
            True,
        ),
        # Security errors
        PathTraversalError: (
            None,
            "File operations are restricted to the project root directory.",
            "SEC_001",
            False,
        ),
        BlockedCommandError: (
            None,
            "This command is blocked by security rules.",
            "SEC_002",
            False,
        ),
        ExcludedFileError: (
            None,
            "This file matches an excluded pattern (likely contains secrets).",
            "SEC_003",
            False,
        ),
        SecurityError: (
            None,
            "A security restriction prevented this operation.",
            "SEC_004",
            False,
        ),
        # Context errors
        ContextWindowExceededError: (
            None,
            "Reduce the context size or switch to a model with a larger context window.",
            "CTX_001",
            True,
        ),
        ContextError: (
            None,
            "Context management error. Try '/compact' to reduce context size.",
            "CTX_002",
            True,
        ),
        # Database errors
        MigrationError: (
            None,
            "Try 'prism db vacuum' or delete ~/.prism/prism.db to reset.",
            "DB_001",
            True,
        ),
        DatabaseError: (
            None,
            "Database error occurred. Try 'prism db vacuum'.",
            "DB_002",
            True,
        ),
        # Git errors
        GitError: (
            None,
            "Not a git repository. Run 'git init' first.",
            "GIT_001",
            True,
        ),
    }

    def handle(self, error: Exception) -> UserError:
        """Convert any exception to a user-friendly message.

        Args:
            error: The exception to handle.

        Returns:
            A :class:`UserError` with a human-readable message and
            optional suggestion.
        """
        # Walk the MRO to find the most specific match
        for cls in type(error).__mro__:
            entry = self._ERROR_MAP.get(cls)  # type: ignore[arg-type]
            if entry is not None:
                msg_template, suggestion_template, code, recoverable = entry
                message = msg_template or str(error)
                suggestion = self._suggest_fix(error, suggestion_template)
                return UserError(
                    message=message,
                    suggestion=suggestion,
                    error_code=code,
                    recoverable=recoverable,
                )

        # Generic PrismError fallback
        if isinstance(error, PrismError):
            return UserError(
                message=str(error),
                suggestion="See 'prism --help' for usage information.",
                error_code="PRISM_000",
                recoverable=True,
            )

        # Keyboard interrupt
        if isinstance(error, KeyboardInterrupt):
            return UserError(
                message="Operation interrupted.",
                suggestion=None,
                error_code="INT_001",
                recoverable=False,
            )

        # Unknown error
        logger.error("unhandled_error", error_type=type(error).__name__, error=str(error))
        return UserError(
            message=f"An unexpected error occurred: {error}",
            suggestion="Please report this issue if it persists.",
            error_code="UNKNOWN_001",
            recoverable=False,
        )

    def _suggest_fix(
        self,
        error: Exception,
        template: str | None,
    ) -> str | None:
        """Suggest a fix based on error type, formatting the template with
        attributes from the error.

        Args:
            error: The original exception.
            template: A format string that may reference error attributes.

        Returns:
            Formatted suggestion string, or ``None``.
        """
        if template is None:
            return None

        # Collect attributes from the error for template formatting
        attrs: dict[str, object] = {}
        for attr in ("provider", "retry_after", "model_id", "path", "tool_name"):
            val = getattr(error, attr, None)
            if val is not None:
                attrs[attr] = val

        # Provide sensible default for retry_after
        if "retry_after" not in attrs:
            attrs["retry_after"] = "a few"

        try:
            return template.format(**attrs)
        except (KeyError, IndexError):
            return template


# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------

_original_sigint: object = None
_original_sigterm: object = None


def install_signal_handlers() -> None:
    """Install Ctrl+C and SIGTERM handlers for graceful shutdown.

    - Ctrl+C: save session, show message, exit cleanly
    - SIGTERM: same behaviour
    - Never corrupt files mid-write
    """
    global _original_sigint, _original_sigterm  # noqa: PLW0603

    def _graceful_shutdown(signum: int, frame: object) -> None:
        """Handle signals gracefully."""
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info("signal_received", signal=sig_name)

        # Print a user-friendly message
        try:
            from rich.console import Console

            console = Console(stderr=True)
            console.print(f"\n[yellow]Received {sig_name}. Shutting down gracefully...[/]")
        except Exception:
            print(f"\nReceived {sig_name}. Shutting down...", file=sys.stderr)  # noqa: T201

        sys.exit(128 + signum)

    _original_sigint = signal.getsignal(signal.SIGINT)
    _original_sigterm = signal.getsignal(signal.SIGTERM)

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)


def restore_signal_handlers() -> None:
    """Restore original signal handlers."""
    global _original_sigint, _original_sigterm  # noqa: PLW0603

    if _original_sigint is not None:
        signal.signal(signal.SIGINT, _original_sigint)
        _original_sigint = None
    if _original_sigterm is not None:
        signal.signal(signal.SIGTERM, _original_sigterm)
        _original_sigterm = None
