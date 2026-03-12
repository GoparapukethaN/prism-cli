"""Smart error recovery for the agentic tool-execution loop.

Classifies tool errors into known categories, generates targeted recovery
strategies, and provides structured prompts that help the LLM self-correct
instead of blindly retrying.

Typical usage inside the REPL::

    engine = ErrorRecoveryEngine()
    error_type = engine.classify_error("edit_file", error_msg, args)
    strategy = engine.get_recovery_strategy(error_type, "edit_file", args)
    prompt = engine.format_recovery_prompt("edit_file", error_msg, strategy)
    # feed *prompt* back to the LLM as the tool-call result

After a successful tool call, call ``engine.reset(tool_name)`` so the
consecutive-error counter restarts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Error type taxonomy
# ---------------------------------------------------------------------------


@unique
class ErrorType(Enum):
    """Categorisation of tool errors.

    Each variant maps to a family of root causes so the recovery engine can
    pick the right strategy without parsing ad-hoc substrings everywhere.
    """

    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    SEARCH_NO_MATCH = "search_no_match"
    EDIT_MISMATCH = "edit_mismatch"
    COMMAND_FAILED = "command_failed"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Recovery strategy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryStrategy:
    """A concrete suggestion that the LLM can act on.

    Attributes:
        suggestion: Human-readable description of what to do differently.
        auto_fix_command: Optional tool call (as a string) that might fix
            the underlying problem automatically (e.g. ``pip install foo``).
        max_retries: How many times this class of error should be retried
            before giving up.
        should_backoff: Whether to pause before retrying (useful for
            rate-limits and network errors).
    """

    suggestion: str
    auto_fix_command: str | None = None
    max_retries: int = 3
    should_backoff: bool = False


# ---------------------------------------------------------------------------
# Classification rules (compiled once at import time)
# ---------------------------------------------------------------------------

# Each rule is a tuple of (regex_pattern, error_type).  Rules are evaluated
# top-to-bottom; the first match wins.

_CLASSIFICATION_RULES: list[
    tuple[re.Pattern[str], ErrorType]
] = [
    # --- File / path errors ---
    (
        re.compile(
            r"no such file|file not found"
            r"|does not exist|filenotfounderror",
            re.I,
        ),
        ErrorType.FILE_NOT_FOUND,
    ),
    (
        re.compile(
            r"permission denied|permissionerror"
            r"|access is denied|operation not permitted",
            re.I,
        ),
        ErrorType.PERMISSION_DENIED,
    ),
    # --- Edit / search mismatches ---
    (
        re.compile(
            r"search string.*not found|old_string.*not found"
            r"|no match found|text not found in file",
            re.I,
        ),
        ErrorType.EDIT_MISMATCH,
    ),
    (
        re.compile(
            r"no results|no matches|pattern not found"
            r"|nothing matched|returned 0 results",
            re.I,
        ),
        ErrorType.SEARCH_NO_MATCH,
    ),
    # --- Syntax ---
    (
        re.compile(
            r"syntaxerror|syntax error|invalid syntax"
            r"|unexpected token|unexpected indent"
            r"|parsing error",
            re.I,
        ),
        ErrorType.SYNTAX_ERROR,
    ),
    # --- Import / package ---
    (
        re.compile(
            r"modulenotfounderror|no module named"
            r"|importerror|import error|cannot import",
            re.I,
        ),
        ErrorType.IMPORT_ERROR,
    ),
    # --- Timeout ---
    (
        re.compile(
            r"timed? ?out|timeout|deadline exceeded"
            r"|operation.*exceeded.*time",
            re.I,
        ),
        ErrorType.TIMEOUT,
    ),
    # --- Rate limiting ---
    (
        re.compile(
            r"rate.?limit|too many requests"
            r"|429|quota exceeded|throttled",
            re.I,
        ),
        ErrorType.RATE_LIMITED,
    ),
    # --- Network ---
    (
        re.compile(
            r"connection.?(?:refused|reset|error|timeout)"
            r"|network.?(?:error|unreachable)"
            r"|dns.?(?:resolution|lookup)"
            r"|could not resolve|ssl.?error"
            r"|connectionerror|urllib3|httpx.*connect",
            re.I,
        ),
        ErrorType.NETWORK_ERROR,
    ),
    # --- Command failures (broad, must come last) ---
    (
        re.compile(
            r"non-?zero exit|exit code|exit status"
            r"|command failed|returned \d+|exited with",
            re.I,
        ),
        ErrorType.COMMAND_FAILED,
    ),
]

# Secondary rules that use *tool_name* context to refine the classification
# when the generic regex didn't match but the tool is well-known.
_TOOL_HEURISTICS: dict[str, ErrorType] = {
    "edit_file": ErrorType.EDIT_MISMATCH,
    "search_codebase": ErrorType.SEARCH_NO_MATCH,
    "execute_command": ErrorType.COMMAND_FAILED,
}

# Sub-patterns for COMMAND_FAILED to detect common root causes
_COMMAND_SUB_PATTERNS: list[
    tuple[re.Pattern[str], str, str | None]
] = [
    # Specific tool-not-found patterns (must come before generic)
    (
        re.compile(r"npm.*not found|npx.*not found", re.I),
        "npm/npx is not installed. Install Node.js first.",
        "execute_command: npm install",
    ),
    (
        re.compile(r"pip.*not found|python.*not found", re.I),
        "Python/pip is not available. Check your Python installation.",
        None,
    ),
    # Generic command-not-found (broad catch, must come after specific)
    (
        re.compile(r"command not found|not recognized|not found", re.I),
        "The command is not installed or not in PATH. "
        "Install the missing tool first.",
        None,
    ),
    (
        re.compile(r"permission denied", re.I),
        "Command failed due to permission denied. "
        "Try running with appropriate permissions "
        "or use a different approach.",
        None,
    ),
    (
        re.compile(r"no space left|disk full", re.I),
        "Disk is full. Free up space before retrying.",
        None,
    ),
    (
        re.compile(r"killed|oom|out of memory", re.I),
        "Process was killed, possibly due to memory exhaustion. "
        "Try a smaller operation.",
        None,
    ),
]

# Sub-patterns for IMPORT_ERROR to extract the package name
_IMPORT_PACKAGE_RE = re.compile(
    r"no module named ['\"]([^'\"]+)['\"]"
    r"|no module named (\S+)"
    r"|cannot import name ['\"]([^'\"]+)['\"]"
    r"|cannot import name (\S+)",
    re.I,
)


# ---------------------------------------------------------------------------
# ErrorClassifier
# ---------------------------------------------------------------------------


class ErrorClassifier:
    """Categorises a raw error message into an :class:`ErrorType`.

    The classifier runs a series of compiled regex rules against the error
    message, falling back to tool-specific heuristics when no regex matches.
    """

    def classify(
        self,
        tool_name: str,
        error_message: str,
        arguments: dict[str, Any] | None = None,
    ) -> ErrorType:
        """Classify an error into an :class:`ErrorType`.

        Args:
            tool_name: The tool that produced the error.
            error_message: The raw error string from tool execution.
            arguments: The arguments that were passed to the tool (used for
                context-aware hinting but not required).

        Returns:
            The best-matching :class:`ErrorType`.
        """
        if not error_message:
            return ErrorType.UNKNOWN

        # First pass: regex rules (most specific first)
        for pattern, error_type in _CLASSIFICATION_RULES:
            if pattern.search(error_message):
                logger.debug(
                    "error_classified",
                    tool=tool_name,
                    error_type=error_type.value,
                    pattern=pattern.pattern[:60],
                )
                return error_type

        # Second pass: tool-specific heuristic (broad catch)
        heuristic = _TOOL_HEURISTICS.get(tool_name)
        if heuristic is not None:
            logger.debug(
                "error_classified_heuristic",
                tool=tool_name,
                error_type=heuristic.value,
            )
            return heuristic

        return ErrorType.UNKNOWN


# ---------------------------------------------------------------------------
# ErrorRecoveryEngine
# ---------------------------------------------------------------------------


class ErrorRecoveryEngine:
    """Orchestrates classification, strategy selection, prompt formatting,
    and consecutive-error tracking for the agentic tool loop.

    Instantiate one per REPL session and call :meth:`reset` after each
    successful tool execution.
    """

    def __init__(self) -> None:
        self._classifier = ErrorClassifier()
        # tool_name -> list of consecutive ErrorType values
        self._error_history: dict[str, list[ErrorType]] = {}

    # -- public API --------------------------------------------------------

    def classify_error(
        self,
        tool_name: str,
        error_message: str,
        arguments: dict[str, Any] | None = None,
    ) -> ErrorType:
        """Classify an error and record it in the history.

        Args:
            tool_name: The tool that failed.
            error_message: Raw error string.
            arguments: Tool arguments (optional, for context).

        Returns:
            The classified :class:`ErrorType`.
        """
        return self._classifier.classify(tool_name, error_message, arguments)

    def get_recovery_strategy(
        self,
        error_type: ErrorType,
        tool_name: str,
        context: dict[str, Any] | None = None,
    ) -> RecoveryStrategy:
        """Select a recovery strategy for the given error type.

        The strategy is context-aware: for ``COMMAND_FAILED`` it inspects
        the error message (passed via *context*) to detect sub-causes like
        "command not found" vs. "permission denied".  For ``IMPORT_ERROR``
        it tries to extract the missing package name.

        Args:
            error_type: The classified error type.
            tool_name: The tool that produced the error.
            context: Additional context.  The key ``"error_message"`` is
                used for sub-pattern matching.  The key ``"arguments"`` may
                contain the original tool arguments.

        Returns:
            A :class:`RecoveryStrategy` tailored to the error.
        """
        ctx = context or {}
        error_message: str = ctx.get("error_message", "")
        arguments: dict[str, Any] = ctx.get("arguments", {})

        strategy = _BASE_STRATEGIES.get(error_type)
        if strategy is None:
            strategy = _BASE_STRATEGIES[ErrorType.UNKNOWN]

        # --- Specialise for COMMAND_FAILED sub-causes ---
        if error_type == ErrorType.COMMAND_FAILED and error_message:
            for pattern, suggestion, auto_fix in _COMMAND_SUB_PATTERNS:
                if pattern.search(error_message):
                    return RecoveryStrategy(
                        suggestion=suggestion,
                        auto_fix_command=auto_fix,
                        max_retries=strategy.max_retries,
                        should_backoff=strategy.should_backoff,
                    )

        # --- Specialise for IMPORT_ERROR: extract package name ---
        if error_type == ErrorType.IMPORT_ERROR and error_message:
            match = _IMPORT_PACKAGE_RE.search(error_message)
            if match:
                pkg = (
                    match.group(1) or match.group(2)
                    or match.group(3) or match.group(4) or ""
                )
                pkg = pkg.strip("'\"").split(".")[0]
                if pkg:
                    return RecoveryStrategy(
                        suggestion=(
                            f"The module '{pkg}' is not installed. "
                            f"Install it with: pip install {pkg}"
                        ),
                        auto_fix_command=f"execute_command: pip install {pkg}",
                        max_retries=strategy.max_retries,
                        should_backoff=strategy.should_backoff,
                    )

        # --- Specialise for FILE_NOT_FOUND: include the attempted path ---
        if error_type == ErrorType.FILE_NOT_FOUND:
            attempted_path = arguments.get("path") or arguments.get("file_path", "")
            if attempted_path:
                return RecoveryStrategy(
                    suggestion=(
                        f"The file '{attempted_path}' does not exist. "
                        "Use list_directory to find the correct path, "
                        "or check for typos in the filename."
                    ),
                    auto_fix_command=None,
                    max_retries=strategy.max_retries,
                    should_backoff=strategy.should_backoff,
                )

        # --- Specialise for EDIT_MISMATCH: remind to read first ---
        if error_type == ErrorType.EDIT_MISMATCH:
            file_path = arguments.get("path") or arguments.get("file_path", "")
            if file_path:
                return RecoveryStrategy(
                    suggestion=(
                        f"The search string was not found in '{file_path}'. "
                        "Use read_file first to see the actual current content, "
                        "then copy the exact text (including whitespace and "
                        "indentation) into old_string."
                    ),
                    auto_fix_command=None,
                    max_retries=strategy.max_retries,
                    should_backoff=strategy.should_backoff,
                )

        return strategy

    def format_recovery_prompt(
        self,
        tool_name: str,
        error_message: str,
        strategy: RecoveryStrategy,
    ) -> str:
        """Build a structured prompt to feed back to the LLM.

        The prompt tells the model *what* went wrong, *why*, and *how* to
        fix it, reducing aimless retries.

        Args:
            tool_name: The tool that failed.
            error_message: The raw error string.
            strategy: The selected recovery strategy.

        Returns:
            A multi-line string ready to be used as the tool-call result.
        """
        parts: list[str] = [
            f"ERROR in {tool_name}:",
            error_message,
            "",
            f"Recovery suggestion: {strategy.suggestion}",
        ]

        if strategy.auto_fix_command:
            parts.append(f"Possible auto-fix: {strategy.auto_fix_command}")

        history = self._error_history.get(tool_name, [])
        consecutive = len(history)
        if consecutive >= 2:
            parts.append(
                f"Warning: {tool_name} has failed {consecutive} times in a row. "
                "Try a fundamentally different approach."
            )

        if strategy.should_backoff:
            parts.append("Note: Wait before retrying — this error may be transient.")

        parts.append("")
        parts.append("Please analyze this error and try a different approach.")

        return "\n".join(parts)

    def track_error(self, tool_name: str, error_type: ErrorType) -> None:
        """Record a consecutive error for *tool_name*.

        Args:
            tool_name: The tool that just failed.
            error_type: The classified error type.
        """
        self._error_history.setdefault(tool_name, []).append(error_type)
        logger.debug(
            "error_tracked",
            tool=tool_name,
            error_type=error_type.value,
            consecutive=len(self._error_history[tool_name]),
        )

    def should_abort(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* has failed 3+ consecutive times.

        Args:
            tool_name: The tool to check.

        Returns:
            Whether the loop should stop retrying this tool.
        """
        history = self._error_history.get(tool_name, [])
        return len(history) >= 3

    def reset(self, tool_name: str) -> None:
        """Clear the consecutive-error counter for *tool_name*.

        Call this after a successful execution so the next failure starts
        fresh.

        Args:
            tool_name: The tool that succeeded.
        """
        if tool_name in self._error_history:
            del self._error_history[tool_name]


# ---------------------------------------------------------------------------
# Base strategies (one per ErrorType)
# ---------------------------------------------------------------------------

_BASE_STRATEGIES: dict[ErrorType, RecoveryStrategy] = {
    ErrorType.FILE_NOT_FOUND: RecoveryStrategy(
        suggestion=(
            "The file does not exist. Use list_directory to find the correct "
            "path, or check for typos in the filename."
        ),
        auto_fix_command=None,
        max_retries=3,
        should_backoff=False,
    ),
    ErrorType.PERMISSION_DENIED: RecoveryStrategy(
        suggestion=(
            "Access was denied. Check file/directory permissions, or try "
            "a different path. Avoid operations that require elevated "
            "privileges."
        ),
        auto_fix_command=None,
        max_retries=2,
        should_backoff=False,
    ),
    ErrorType.SEARCH_NO_MATCH: RecoveryStrategy(
        suggestion=(
            "The search returned no results. Try broader search terms, "
            "check the regex syntax, or use list_directory to confirm "
            "the file exists in the expected location."
        ),
        auto_fix_command=None,
        max_retries=3,
        should_backoff=False,
    ),
    ErrorType.EDIT_MISMATCH: RecoveryStrategy(
        suggestion=(
            "The search string was not found in the file. Use read_file "
            "first to see the actual content, then use the exact text "
            "including whitespace and indentation."
        ),
        auto_fix_command=None,
        max_retries=3,
        should_backoff=False,
    ),
    ErrorType.COMMAND_FAILED: RecoveryStrategy(
        suggestion=(
            "The command failed. Check the error output, verify the command "
            "syntax, and ensure required tools are installed."
        ),
        auto_fix_command=None,
        max_retries=3,
        should_backoff=False,
    ),
    ErrorType.SYNTAX_ERROR: RecoveryStrategy(
        suggestion=(
            "The code has a syntax error. Carefully review the error line "
            "and column, fix the syntax issue, and retry."
        ),
        auto_fix_command=None,
        max_retries=3,
        should_backoff=False,
    ),
    ErrorType.IMPORT_ERROR: RecoveryStrategy(
        suggestion=(
            "A required Python module is not installed. Install it with "
            "pip install <package_name>."
        ),
        auto_fix_command=None,
        max_retries=2,
        should_backoff=False,
    ),
    ErrorType.TIMEOUT: RecoveryStrategy(
        suggestion=(
            "The operation timed out. Try breaking it into smaller steps, "
            "reduce the scope of the operation, or increase the timeout."
        ),
        auto_fix_command=None,
        max_retries=2,
        should_backoff=True,
    ),
    ErrorType.RATE_LIMITED: RecoveryStrategy(
        suggestion=(
            "The API rate limit was hit. Wait a moment before retrying, "
            "or switch to a different provider with '/model'."
        ),
        auto_fix_command=None,
        max_retries=3,
        should_backoff=True,
    ),
    ErrorType.NETWORK_ERROR: RecoveryStrategy(
        suggestion=(
            "A network error occurred. Check your internet connection. "
            "The service may be temporarily unavailable — retry in a moment."
        ),
        auto_fix_command=None,
        max_retries=3,
        should_backoff=True,
    ),
    ErrorType.UNKNOWN: RecoveryStrategy(
        suggestion=(
            "An unexpected error occurred. Analyze the error message "
            "carefully and try a different approach."
        ),
        auto_fix_command=None,
        max_retries=2,
        should_backoff=False,
    ),
}


# ---------------------------------------------------------------------------
# Convenience helper (drop-in replacement for _format_tool_error)
# ---------------------------------------------------------------------------


def format_tool_error(
    tool_name: str,
    error_message: str,
    arguments: dict[str, Any] | None = None,
    engine: ErrorRecoveryEngine | None = None,
) -> str:
    """Drop-in replacement for the old ``_format_tool_error`` in repl.py.

    If no *engine* is supplied a temporary one is created (without history
    tracking).  For full benefit, pass the session-wide engine so that
    consecutive-error tracking and abort detection work.

    Args:
        tool_name: The tool that failed.
        error_message: Raw error string.
        arguments: The arguments that were passed to the tool.
        engine: An :class:`ErrorRecoveryEngine` (optional).

    Returns:
        A structured error prompt for the LLM.
    """
    eng = engine or ErrorRecoveryEngine()
    error_type = eng.classify_error(tool_name, error_message, arguments)
    eng.track_error(tool_name, error_type)

    context: dict[str, Any] = {
        "error_message": error_message,
        "arguments": arguments or {},
    }
    strategy = eng.get_recovery_strategy(error_type, tool_name, context)
    return eng.format_recovery_prompt(tool_name, error_message, strategy)
