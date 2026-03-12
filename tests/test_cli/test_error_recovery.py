"""Tests for prism.cli.error_recovery — smart error recovery engine."""

from __future__ import annotations

import pytest

from prism.cli.error_recovery import (
    ErrorClassifier,
    ErrorRecoveryEngine,
    ErrorType,
    RecoveryStrategy,
    format_tool_error,
)

# ---------------------------------------------------------------------------
# ErrorType enum
# ---------------------------------------------------------------------------


class TestErrorType:
    """Basic invariants for the ErrorType enum."""

    def test_all_members_have_string_values(self) -> None:
        for member in ErrorType:
            assert isinstance(member.value, str)
            assert len(member.value) > 0

    def test_member_count(self) -> None:
        assert len(ErrorType) == 11

    def test_unknown_exists(self) -> None:
        assert ErrorType.UNKNOWN in ErrorType


# ---------------------------------------------------------------------------
# RecoveryStrategy dataclass
# ---------------------------------------------------------------------------


class TestRecoveryStrategy:
    """Verify RecoveryStrategy defaults and immutability."""

    def test_defaults(self) -> None:
        s = RecoveryStrategy(suggestion="do something")
        assert s.suggestion == "do something"
        assert s.auto_fix_command is None
        assert s.max_retries == 3
        assert s.should_backoff is False

    def test_custom_values(self) -> None:
        s = RecoveryStrategy(
            suggestion="wait",
            auto_fix_command="pip install foo",
            max_retries=5,
            should_backoff=True,
        )
        assert s.auto_fix_command == "pip install foo"
        assert s.max_retries == 5
        assert s.should_backoff is True

    def test_frozen(self) -> None:
        s = RecoveryStrategy(suggestion="x")
        with pytest.raises(AttributeError):
            s.suggestion = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ErrorClassifier
# ---------------------------------------------------------------------------


class TestErrorClassifier:
    """Exercise every ErrorType classification path."""

    @pytest.fixture()
    def classifier(self) -> ErrorClassifier:
        return ErrorClassifier()

    # --- FILE_NOT_FOUND ---

    @pytest.mark.parametrize(
        "msg",
        [
            "No such file or directory: '/tmp/missing.py'",
            "FileNotFoundError: [Errno 2] No such file",
            "File not found: config.yaml",
            "The path /foo/bar does not exist",
        ],
    )
    def test_file_not_found(self, classifier: ErrorClassifier, msg: str) -> None:
        assert classifier.classify("read_file", msg) == ErrorType.FILE_NOT_FOUND

    # --- PERMISSION_DENIED ---

    @pytest.mark.parametrize(
        "msg",
        [
            "PermissionError: [Errno 13] Permission denied",
            "permission denied for /etc/shadow",
            "Access is denied",
            "Operation not permitted",
        ],
    )
    def test_permission_denied(self, classifier: ErrorClassifier, msg: str) -> None:
        assert classifier.classify("read_file", msg) == ErrorType.PERMISSION_DENIED

    # --- EDIT_MISMATCH ---

    @pytest.mark.parametrize(
        "msg",
        [
            "Search string 'foo bar' not found in file",
            "old_string not found in the file",
            "No match found in target file",
            "Text not found in file /src/main.py",
        ],
    )
    def test_edit_mismatch(self, classifier: ErrorClassifier, msg: str) -> None:
        assert classifier.classify("edit_file", msg) == ErrorType.EDIT_MISMATCH

    # --- SEARCH_NO_MATCH ---

    @pytest.mark.parametrize(
        "msg",
        [
            "No results found for pattern 'foobar'",
            "No matches in directory /src",
            "Pattern not found in any files",
            "Search returned 0 results",
        ],
    )
    def test_search_no_match(self, classifier: ErrorClassifier, msg: str) -> None:
        assert classifier.classify("search_codebase", msg) == ErrorType.SEARCH_NO_MATCH

    # --- SYNTAX_ERROR ---

    @pytest.mark.parametrize(
        "msg",
        [
            "SyntaxError: invalid syntax (line 42)",
            "syntax error near unexpected token 'fi'",
            "Unexpected token { at line 10",
            "Unexpected indent at line 5",
            "Parsing error in module.py",
        ],
    )
    def test_syntax_error(self, classifier: ErrorClassifier, msg: str) -> None:
        assert classifier.classify("execute_command", msg) == ErrorType.SYNTAX_ERROR

    # --- IMPORT_ERROR ---

    @pytest.mark.parametrize(
        "msg",
        [
            "ModuleNotFoundError: No module named 'numpy'",
            "ImportError: cannot import name 'foo' from 'bar'",
            "No module named 'requests'",
            "Import error while loading conftest",
        ],
    )
    def test_import_error(self, classifier: ErrorClassifier, msg: str) -> None:
        assert classifier.classify("execute_command", msg) == ErrorType.IMPORT_ERROR

    # --- TIMEOUT ---

    @pytest.mark.parametrize(
        "msg",
        [
            "Operation timed out after 30s",
            "TimeoutError: deadline exceeded",
            "Command timed out",
            "Operation exceeded allowed time limit",
        ],
    )
    def test_timeout(self, classifier: ErrorClassifier, msg: str) -> None:
        assert classifier.classify("execute_command", msg) == ErrorType.TIMEOUT

    # --- RATE_LIMITED ---

    @pytest.mark.parametrize(
        "msg",
        [
            "Rate limit exceeded, retry after 60s",
            "HTTP 429 Too Many Requests",
            "Quota exceeded for this API key",
            "Request throttled by provider",
        ],
    )
    def test_rate_limited(self, classifier: ErrorClassifier, msg: str) -> None:
        assert classifier.classify("browse_web", msg) == ErrorType.RATE_LIMITED

    # --- NETWORK_ERROR ---

    @pytest.mark.parametrize(
        "msg",
        [
            "ConnectionRefused: could not connect to server",
            "Connection reset by peer",
            "Network error: host unreachable",
            "DNS resolution failed",
            "Could not resolve hostname api.example.com",
            "SSL error: certificate verify failed",
            "ConnectionError: Remote end closed connection",
            "httpx.ConnectError: connection refused",
        ],
    )
    def test_network_error(self, classifier: ErrorClassifier, msg: str) -> None:
        assert classifier.classify("browse_web", msg) == ErrorType.NETWORK_ERROR

    # --- COMMAND_FAILED ---

    @pytest.mark.parametrize(
        "msg",
        [
            "Non-zero exit code 1",
            "Exit code 127",
            "Command failed with exit status 2",
            "Process returned 1",
            "Process exited with code 1",
        ],
    )
    def test_command_failed(self, classifier: ErrorClassifier, msg: str) -> None:
        assert classifier.classify("execute_command", msg) == ErrorType.COMMAND_FAILED

    # --- UNKNOWN ---

    def test_unknown_for_empty_message(self, classifier: ErrorClassifier) -> None:
        assert classifier.classify("read_file", "") == ErrorType.UNKNOWN

    def test_unknown_for_unrecognised(self, classifier: ErrorClassifier) -> None:
        assert classifier.classify("some_tool", "something went sideways") == ErrorType.UNKNOWN

    # --- Tool heuristic fallback ---

    def test_heuristic_edit_file(self, classifier: ErrorClassifier) -> None:
        # Generic message for edit_file falls back to EDIT_MISMATCH
        assert classifier.classify("edit_file", "something failed") == ErrorType.EDIT_MISMATCH

    def test_heuristic_search_codebase(
        self, classifier: ErrorClassifier,
    ) -> None:
        result = classifier.classify("search_codebase", "something failed")
        assert result == ErrorType.SEARCH_NO_MATCH

    def test_heuristic_execute_command(
        self, classifier: ErrorClassifier,
    ) -> None:
        result = classifier.classify("execute_command", "something failed")
        assert result == ErrorType.COMMAND_FAILED


# ---------------------------------------------------------------------------
# ErrorRecoveryEngine
# ---------------------------------------------------------------------------


class TestErrorRecoveryEngine:
    """Full integration tests for the engine."""

    @pytest.fixture()
    def engine(self) -> ErrorRecoveryEngine:
        return ErrorRecoveryEngine()

    # -- classify_error delegates to classifier --

    def test_classify_delegates(self, engine: ErrorRecoveryEngine) -> None:
        result = engine.classify_error("read_file", "No such file or directory")
        assert result == ErrorType.FILE_NOT_FOUND

    # -- get_recovery_strategy base cases --

    def test_strategy_file_not_found(self, engine: ErrorRecoveryEngine) -> None:
        s = engine.get_recovery_strategy(ErrorType.FILE_NOT_FOUND, "read_file")
        assert "does not exist" in s.suggestion
        assert s.auto_fix_command is None

    def test_strategy_permission_denied(self, engine: ErrorRecoveryEngine) -> None:
        s = engine.get_recovery_strategy(ErrorType.PERMISSION_DENIED, "read_file")
        assert "denied" in s.suggestion.lower() or "permission" in s.suggestion.lower()

    def test_strategy_search_no_match(self, engine: ErrorRecoveryEngine) -> None:
        s = engine.get_recovery_strategy(ErrorType.SEARCH_NO_MATCH, "search_codebase")
        assert "no results" in s.suggestion.lower() or "broader" in s.suggestion.lower()

    def test_strategy_edit_mismatch(self, engine: ErrorRecoveryEngine) -> None:
        s = engine.get_recovery_strategy(ErrorType.EDIT_MISMATCH, "edit_file")
        assert "read_file" in s.suggestion

    def test_strategy_command_failed(self, engine: ErrorRecoveryEngine) -> None:
        s = engine.get_recovery_strategy(ErrorType.COMMAND_FAILED, "execute_command")
        assert "failed" in s.suggestion.lower() or "command" in s.suggestion.lower()

    def test_strategy_syntax_error(self, engine: ErrorRecoveryEngine) -> None:
        s = engine.get_recovery_strategy(ErrorType.SYNTAX_ERROR, "execute_command")
        assert "syntax" in s.suggestion.lower()

    def test_strategy_import_error(self, engine: ErrorRecoveryEngine) -> None:
        s = engine.get_recovery_strategy(ErrorType.IMPORT_ERROR, "execute_command")
        assert "install" in s.suggestion.lower() or "module" in s.suggestion.lower()

    def test_strategy_timeout(self, engine: ErrorRecoveryEngine) -> None:
        s = engine.get_recovery_strategy(ErrorType.TIMEOUT, "execute_command")
        assert "timed out" in s.suggestion.lower() or "smaller" in s.suggestion.lower()
        assert s.should_backoff is True

    def test_strategy_rate_limited(self, engine: ErrorRecoveryEngine) -> None:
        s = engine.get_recovery_strategy(ErrorType.RATE_LIMITED, "browse_web")
        assert "rate limit" in s.suggestion.lower() or "wait" in s.suggestion.lower()
        assert s.should_backoff is True

    def test_strategy_network_error(self, engine: ErrorRecoveryEngine) -> None:
        s = engine.get_recovery_strategy(ErrorType.NETWORK_ERROR, "browse_web")
        assert "network" in s.suggestion.lower() or "connection" in s.suggestion.lower()
        assert s.should_backoff is True

    def test_strategy_unknown(self, engine: ErrorRecoveryEngine) -> None:
        s = engine.get_recovery_strategy(ErrorType.UNKNOWN, "some_tool")
        assert "unexpected" in s.suggestion.lower() or "different approach" in s.suggestion.lower()

    # -- context-aware specializations --

    def test_strategy_file_not_found_with_path(self, engine: ErrorRecoveryEngine) -> None:
        ctx = {
            "error_message": "No such file or directory",
            "arguments": {"file_path": "/tmp/missing.py"},
        }
        s = engine.get_recovery_strategy(ErrorType.FILE_NOT_FOUND, "read_file", ctx)
        assert "/tmp/missing.py" in s.suggestion

    def test_strategy_edit_mismatch_with_path(self, engine: ErrorRecoveryEngine) -> None:
        ctx = {
            "error_message": "Search string not found",
            "arguments": {"file_path": "/src/main.py"},
        }
        s = engine.get_recovery_strategy(ErrorType.EDIT_MISMATCH, "edit_file", ctx)
        assert "/src/main.py" in s.suggestion
        assert "read_file" in s.suggestion

    def test_strategy_import_error_extracts_package(self, engine: ErrorRecoveryEngine) -> None:
        ctx = {
            "error_message": "ModuleNotFoundError: No module named 'requests'",
            "arguments": {},
        }
        s = engine.get_recovery_strategy(ErrorType.IMPORT_ERROR, "execute_command", ctx)
        assert "requests" in s.suggestion
        assert s.auto_fix_command is not None
        assert "pip install requests" in s.auto_fix_command

    def test_strategy_import_error_dotted_module(self, engine: ErrorRecoveryEngine) -> None:
        ctx = {
            "error_message": "No module named 'sklearn.ensemble'",
            "arguments": {},
        }
        s = engine.get_recovery_strategy(ErrorType.IMPORT_ERROR, "execute_command", ctx)
        assert "sklearn" in s.suggestion
        assert s.auto_fix_command is not None
        assert "pip install sklearn" in s.auto_fix_command

    def test_strategy_command_not_found(self, engine: ErrorRecoveryEngine) -> None:
        ctx = {
            "error_message": "bash: cargo: command not found",
            "arguments": {},
        }
        s = engine.get_recovery_strategy(ErrorType.COMMAND_FAILED, "execute_command", ctx)
        assert "not installed" in s.suggestion.lower() or "not in PATH" in s.suggestion

    def test_strategy_npm_not_found(self, engine: ErrorRecoveryEngine) -> None:
        ctx = {
            "error_message": "npm: not found",
            "arguments": {},
        }
        s = engine.get_recovery_strategy(ErrorType.COMMAND_FAILED, "execute_command", ctx)
        assert "npm" in s.suggestion.lower() or "node" in s.suggestion.lower()

    def test_strategy_disk_full(self, engine: ErrorRecoveryEngine) -> None:
        ctx = {
            "error_message": "OSError: No space left on device",
            "arguments": {},
        }
        s = engine.get_recovery_strategy(ErrorType.COMMAND_FAILED, "execute_command", ctx)
        assert "space" in s.suggestion.lower() or "disk" in s.suggestion.lower()

    def test_strategy_oom_killed(self, engine: ErrorRecoveryEngine) -> None:
        ctx = {
            "error_message": "Process killed by OOM killer",
            "arguments": {},
        }
        s = engine.get_recovery_strategy(ErrorType.COMMAND_FAILED, "execute_command", ctx)
        assert "memory" in s.suggestion.lower() or "killed" in s.suggestion.lower()

    # -- format_recovery_prompt --

    def test_format_prompt_basic(self, engine: ErrorRecoveryEngine) -> None:
        strategy = RecoveryStrategy(suggestion="Try again differently")
        result = engine.format_recovery_prompt("read_file", "file missing", strategy)
        assert "ERROR in read_file:" in result
        assert "file missing" in result
        assert "Try again differently" in result
        assert "different approach" in result

    def test_format_prompt_with_auto_fix(self, engine: ErrorRecoveryEngine) -> None:
        strategy = RecoveryStrategy(
            suggestion="Install the package",
            auto_fix_command="pip install foo",
        )
        result = engine.format_recovery_prompt("execute_command", "err", strategy)
        assert "pip install foo" in result

    def test_format_prompt_with_backoff(self, engine: ErrorRecoveryEngine) -> None:
        strategy = RecoveryStrategy(suggestion="wait", should_backoff=True)
        result = engine.format_recovery_prompt("browse_web", "rate limited", strategy)
        assert "Wait before retrying" in result

    def test_format_prompt_consecutive_warnings(self, engine: ErrorRecoveryEngine) -> None:
        engine.track_error("read_file", ErrorType.FILE_NOT_FOUND)
        engine.track_error("read_file", ErrorType.FILE_NOT_FOUND)
        strategy = RecoveryStrategy(suggestion="check path")
        result = engine.format_recovery_prompt("read_file", "not found", strategy)
        assert "failed 2 times in a row" in result
        assert "fundamentally different approach" in result

    # -- track_error / should_abort / reset --

    def test_track_error_increments(self, engine: ErrorRecoveryEngine) -> None:
        engine.track_error("read_file", ErrorType.FILE_NOT_FOUND)
        assert not engine.should_abort("read_file")
        engine.track_error("read_file", ErrorType.FILE_NOT_FOUND)
        assert not engine.should_abort("read_file")
        engine.track_error("read_file", ErrorType.FILE_NOT_FOUND)
        assert engine.should_abort("read_file")

    def test_should_abort_false_for_unknown_tool(self, engine: ErrorRecoveryEngine) -> None:
        assert not engine.should_abort("never_called_tool")

    def test_reset_clears_history(self, engine: ErrorRecoveryEngine) -> None:
        engine.track_error("read_file", ErrorType.FILE_NOT_FOUND)
        engine.track_error("read_file", ErrorType.FILE_NOT_FOUND)
        engine.track_error("read_file", ErrorType.FILE_NOT_FOUND)
        assert engine.should_abort("read_file")
        engine.reset("read_file")
        assert not engine.should_abort("read_file")

    def test_reset_noop_for_unknown_tool(self, engine: ErrorRecoveryEngine) -> None:
        engine.reset("nonexistent")  # should not raise

    def test_different_tools_tracked_independently(self, engine: ErrorRecoveryEngine) -> None:
        engine.track_error("read_file", ErrorType.FILE_NOT_FOUND)
        engine.track_error("read_file", ErrorType.FILE_NOT_FOUND)
        engine.track_error("read_file", ErrorType.FILE_NOT_FOUND)
        engine.track_error("edit_file", ErrorType.EDIT_MISMATCH)
        assert engine.should_abort("read_file")
        assert not engine.should_abort("edit_file")


# ---------------------------------------------------------------------------
# format_tool_error convenience function
# ---------------------------------------------------------------------------


class TestFormatToolError:
    """Tests for the drop-in replacement function."""

    def test_basic_call(self) -> None:
        result = format_tool_error("read_file", "No such file or directory")
        assert "ERROR in read_file:" in result
        assert "No such file or directory" in result
        assert "does not exist" in result or "different approach" in result

    def test_with_arguments(self) -> None:
        result = format_tool_error(
            "read_file",
            "No such file or directory",
            arguments={"file_path": "/tmp/foo.py"},
        )
        assert "/tmp/foo.py" in result

    def test_with_engine_tracks_errors(self) -> None:
        engine = ErrorRecoveryEngine()
        format_tool_error("read_file", "No such file", engine=engine)
        format_tool_error("read_file", "No such file", engine=engine)
        format_tool_error("read_file", "No such file", engine=engine)
        assert engine.should_abort("read_file")

    def test_without_engine_creates_temporary(self) -> None:
        # Should not raise even without an engine
        result = format_tool_error("edit_file", "old_string not found in the file")
        assert "ERROR in edit_file:" in result

    def test_edit_mismatch_with_engine(self) -> None:
        engine = ErrorRecoveryEngine()
        result = format_tool_error(
            "edit_file",
            "Search string 'class Foo' not found in file",
            arguments={"file_path": "/src/models.py"},
            engine=engine,
        )
        assert "/src/models.py" in result
        assert "read_file" in result

    def test_import_error_with_package(self) -> None:
        result = format_tool_error(
            "execute_command",
            "ModuleNotFoundError: No module named 'pandas'",
        )
        assert "pandas" in result
        assert "pip install" in result

    def test_timeout_has_backoff_note(self) -> None:
        result = format_tool_error(
            "execute_command",
            "TimeoutError: operation timed out after 60s",
        )
        assert "Wait before retrying" in result

    def test_unknown_error(self) -> None:
        result = format_tool_error("some_tool", "xyzzy happened")
        assert "ERROR in some_tool:" in result
        assert "different approach" in result


# ---------------------------------------------------------------------------
# Edge cases and regression tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Cover tricky inputs and boundary conditions."""

    @pytest.fixture()
    def engine(self) -> ErrorRecoveryEngine:
        return ErrorRecoveryEngine()

    def test_empty_error_message(self, engine: ErrorRecoveryEngine) -> None:
        etype = engine.classify_error("read_file", "")
        assert etype == ErrorType.UNKNOWN

    def test_none_arguments(self, engine: ErrorRecoveryEngine) -> None:
        etype = engine.classify_error("read_file", "No such file", arguments=None)
        assert etype == ErrorType.FILE_NOT_FOUND

    def test_strategy_with_empty_context(self, engine: ErrorRecoveryEngine) -> None:
        s = engine.get_recovery_strategy(ErrorType.IMPORT_ERROR, "execute_command", {})
        assert "install" in s.suggestion.lower() or "module" in s.suggestion.lower()

    def test_strategy_with_none_context(self, engine: ErrorRecoveryEngine) -> None:
        s = engine.get_recovery_strategy(ErrorType.IMPORT_ERROR, "execute_command", None)
        assert "install" in s.suggestion.lower()

    def test_multiple_resets_are_safe(self, engine: ErrorRecoveryEngine) -> None:
        engine.reset("foo")
        engine.reset("foo")
        engine.reset("foo")
        assert not engine.should_abort("foo")

    def test_mixed_error_types_still_count(self, engine: ErrorRecoveryEngine) -> None:
        engine.track_error("tool", ErrorType.FILE_NOT_FOUND)
        engine.track_error("tool", ErrorType.PERMISSION_DENIED)
        engine.track_error("tool", ErrorType.UNKNOWN)
        assert engine.should_abort("tool")

    def test_classify_prefers_specific_over_generic(self) -> None:
        classifier = ErrorClassifier()
        # "permission denied" matches both PERMISSION_DENIED and COMMAND_FAILED patterns
        # but PERMISSION_DENIED should win because it comes first in the rule list
        result = classifier.classify("execute_command", "permission denied: /etc/shadow")
        assert result == ErrorType.PERMISSION_DENIED

    def test_classify_import_error_with_cannot_import(self) -> None:
        classifier = ErrorClassifier()
        result = classifier.classify(
            "execute_command",
            "ImportError: cannot import name 'Foo' from 'bar.baz'",
        )
        assert result == ErrorType.IMPORT_ERROR

    def test_strategy_import_cannot_import_extracts_name(
        self, engine: ErrorRecoveryEngine,
    ) -> None:
        ctx = {
            "error_message": "ImportError: cannot import name 'Foo' from 'bar.baz'",
            "arguments": {},
        }
        s = engine.get_recovery_strategy(ErrorType.IMPORT_ERROR, "execute_command", ctx)
        # Should extract "Foo" as the name from the "cannot import name" pattern
        assert "Foo" in s.suggestion or "bar" in s.suggestion

    def test_format_prompt_no_consecutive_warning_for_single_error(
        self, engine: ErrorRecoveryEngine,
    ) -> None:
        engine.track_error("read_file", ErrorType.FILE_NOT_FOUND)
        strategy = RecoveryStrategy(suggestion="fix it")
        result = engine.format_recovery_prompt("read_file", "err", strategy)
        assert "failed" not in result.lower() or "1 times" not in result

    def test_command_failed_permission_denied_subpattern(
        self, engine: ErrorRecoveryEngine,
    ) -> None:
        ctx = {
            "error_message": "exit code 1: permission denied for /usr/local/bin/foo",
            "arguments": {},
        }
        s = engine.get_recovery_strategy(ErrorType.COMMAND_FAILED, "execute_command", ctx)
        assert "permission" in s.suggestion.lower()

    def test_file_not_found_path_from_path_key(
        self, engine: ErrorRecoveryEngine,
    ) -> None:
        ctx = {
            "error_message": "No such file",
            "arguments": {"path": "/home/user/missing.txt"},
        }
        s = engine.get_recovery_strategy(ErrorType.FILE_NOT_FOUND, "list_directory", ctx)
        assert "/home/user/missing.txt" in s.suggestion

    def test_edit_mismatch_path_from_path_key(
        self, engine: ErrorRecoveryEngine,
    ) -> None:
        ctx = {
            "error_message": "old_string not found",
            "arguments": {"path": "/src/app.py"},
        }
        s = engine.get_recovery_strategy(ErrorType.EDIT_MISMATCH, "edit_file", ctx)
        assert "/src/app.py" in s.suggestion
