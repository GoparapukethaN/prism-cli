"""Tests for the ErrorHandler and signal handling."""

from __future__ import annotations

import signal

import pytest

from prism.cli.error_handler import (
    ErrorHandler,
    UserError,
    install_signal_handlers,
    restore_signal_handlers,
)
from prism.exceptions import (
    AuthError,
    BudgetExceededError,
    ContextWindowExceededError,
    KeyNotFoundError,
    ModelNotFoundError,
    NotAGitRepoError,
    PathTraversalError,
    PrismError,
    ProviderError,
    ProviderRateLimitError,
    ToolExecutionError,
    ToolNotFoundError,
)


@pytest.fixture
def handler() -> ErrorHandler:
    """An ErrorHandler instance."""
    return ErrorHandler()


# ---------------------------------------------------------------------------
# Auth errors
# ---------------------------------------------------------------------------


class TestAuthErrors:
    def test_auth_error_message(self, handler: ErrorHandler) -> None:
        """AuthError produces a UserError with AUTH error code."""
        error = AuthError("Something went wrong with auth")
        result = handler.handle(error)

        assert isinstance(result, UserError)
        assert "AUTH" in (result.error_code or "")
        assert result.recoverable is True
        assert result.suggestion is not None

    def test_key_not_found_message(self, handler: ErrorHandler) -> None:
        """KeyNotFoundError includes the provider in the suggestion."""
        error = KeyNotFoundError("anthropic")
        result = handler.handle(error)

        assert "anthropic" in result.message
        assert result.error_code == "AUTH_001"
        assert "prism auth add" in (result.suggestion or "")
        assert "anthropic" in (result.suggestion or "")


# ---------------------------------------------------------------------------
# Provider errors
# ---------------------------------------------------------------------------


class TestProviderErrors:
    def test_rate_limit_with_suggestion(self, handler: ErrorHandler) -> None:
        """ProviderRateLimitError includes retry info in suggestion."""
        error = ProviderRateLimitError("openai", retry_after=30.0)
        result = handler.handle(error)

        assert result.error_code == "PROV_001"
        assert "30" in (result.suggestion or "")
        assert result.recoverable is True

    def test_rate_limit_no_retry_after(self, handler: ErrorHandler) -> None:
        """ProviderRateLimitError without retry_after still works."""
        error = ProviderRateLimitError("openai", retry_after=None)
        result = handler.handle(error)

        assert result.error_code == "PROV_001"
        assert result.suggestion is not None

    def test_provider_error_suggestion(self, handler: ErrorHandler) -> None:
        """Generic ProviderError gets a suggestion to switch."""
        error = ProviderError("Something broke")
        result = handler.handle(error)

        assert "PROV" in (result.error_code or "")
        assert result.suggestion is not None

    def test_model_not_found(self, handler: ErrorHandler) -> None:
        """ModelNotFoundError suggests checking available models."""
        error = ModelNotFoundError("gpt-99")
        result = handler.handle(error)

        assert result.error_code == "PROV_004"
        assert "status" in (result.suggestion or "").lower()


# ---------------------------------------------------------------------------
# Budget / Routing errors
# ---------------------------------------------------------------------------


class TestBudgetErrors:
    def test_budget_exceeded_message(self, handler: ErrorHandler) -> None:
        """BudgetExceededError includes budget info and suggestion."""
        error = BudgetExceededError(budget_remaining=0.50, cheapest_estimate=1.00)
        result = handler.handle(error)

        assert result.error_code == "BUDGET_001"
        assert "budget" in (result.suggestion or "").lower()
        assert result.recoverable is True


# ---------------------------------------------------------------------------
# Security errors
# ---------------------------------------------------------------------------


class TestSecurityErrors:
    def test_path_traversal_message(self, handler: ErrorHandler) -> None:
        """PathTraversalError is not recoverable."""
        error = PathTraversalError(path="/etc/passwd", project_root="/home/user/project")
        result = handler.handle(error)

        assert result.error_code == "SEC_001"
        assert result.recoverable is False
        assert "project root" in (result.suggestion or "").lower()


# ---------------------------------------------------------------------------
# Tool errors
# ---------------------------------------------------------------------------


class TestToolErrors:
    def test_tool_error_message(self, handler: ErrorHandler) -> None:
        """ToolNotFoundError gets a suggestion."""
        error = ToolNotFoundError("nonexistent_tool")
        result = handler.handle(error)

        assert result.error_code == "TOOL_001"
        assert result.suggestion is not None

    def test_tool_execution_error(self, handler: ErrorHandler) -> None:
        """ToolExecutionError gets TOOL_004 code."""
        error = ToolExecutionError("bash", "command failed")
        result = handler.handle(error)

        assert result.error_code == "TOOL_004"


# ---------------------------------------------------------------------------
# Context errors
# ---------------------------------------------------------------------------


class TestContextErrors:
    def test_context_error_message(self, handler: ErrorHandler) -> None:
        """ContextWindowExceededError suggests reducing context."""
        error = ContextWindowExceededError(total_tokens=300_000, max_tokens=200_000)
        result = handler.handle(error)

        assert result.error_code == "CTX_001"
        assert "context" in (result.suggestion or "").lower()


# ---------------------------------------------------------------------------
# Git errors
# ---------------------------------------------------------------------------


class TestGitErrors:
    def test_git_error_suggestion(self, handler: ErrorHandler) -> None:
        """GitError suggests git init."""
        error = NotAGitRepoError("/some/path")
        result = handler.handle(error)

        assert "GIT" in (result.error_code or "")
        assert "git init" in (result.suggestion or "").lower()


# ---------------------------------------------------------------------------
# UserError dataclass
# ---------------------------------------------------------------------------


class TestUserError:
    def test_user_error_has_code(self, handler: ErrorHandler) -> None:
        """All mapped errors get an error code."""
        errors = [
            KeyNotFoundError("test"),
            ProviderRateLimitError("test"),
            BudgetExceededError(0.0, 1.0),
        ]
        for err in errors:
            result = handler.handle(err)
            assert result.error_code is not None

    def test_recoverable_flag(self, handler: ErrorHandler) -> None:
        """Security errors are not recoverable; auth errors are."""
        sec_error = PathTraversalError("/etc/passwd", "/home")
        auth_error = KeyNotFoundError("test")

        sec_result = handler.handle(sec_error)
        auth_result = handler.handle(auth_error)

        assert sec_result.recoverable is False
        assert auth_result.recoverable is True


# ---------------------------------------------------------------------------
# Unknown / fallback errors
# ---------------------------------------------------------------------------


class TestFallbackErrors:
    def test_unknown_error_fallback(self, handler: ErrorHandler) -> None:
        """Unknown exception gets a generic message."""
        error = RuntimeError("something unexpected")
        result = handler.handle(error)

        assert result.error_code == "UNKNOWN_001"
        assert result.recoverable is False
        assert "unexpected" in result.message.lower()

    def test_prism_error_fallback(self, handler: ErrorHandler) -> None:
        """PrismError base class gets a generic Prism fallback."""
        error = PrismError("generic prism error")
        result = handler.handle(error)

        assert result.error_code == "PRISM_000"
        assert result.recoverable is True


# ---------------------------------------------------------------------------
# Keyboard interrupt
# ---------------------------------------------------------------------------


class TestKeyboardInterrupt:
    def test_keyboard_interrupt_handled(self, handler: ErrorHandler) -> None:
        """KeyboardInterrupt is converted to a UserError."""
        error = KeyboardInterrupt()
        result = handler.handle(error)

        assert result.error_code == "INT_001"
        assert result.recoverable is False
        assert "interrupted" in result.message.lower()


# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------


class TestSignalHandlers:
    def test_install_and_restore_signal_handlers(self) -> None:
        """Signal handlers can be installed and restored."""
        original_int = signal.getsignal(signal.SIGINT)
        signal.getsignal(signal.SIGTERM)

        install_signal_handlers()

        # Handlers should be changed
        current_int = signal.getsignal(signal.SIGINT)
        assert current_int != original_int

        restore_signal_handlers()

        # Handlers should be restored
        restored_int = signal.getsignal(signal.SIGINT)
        assert restored_int == original_int
