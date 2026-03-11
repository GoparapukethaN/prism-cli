"""Centralized logging configuration using structlog.

Usage::

    from prism.logging_config import configure_logging, get_logger

    configure_logging(level="INFO", log_dir=Path("~/.prism/logs"))
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import logging.handlers
import re
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Secret scrubber — structlog processor
# ---------------------------------------------------------------------------


class SecretScrubber:
    """Structlog processor that scrubs API keys from all log entries.

    Scans every string value in the event dict and replaces patterns that
    look like API keys with ``***REDACTED***``.
    """

    PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"sk-[a-zA-Z0-9\-_]{20,}"),       # OpenAI / Anthropic
        re.compile(r"gsk_[a-zA-Z0-9]{20,}"),          # Groq
        re.compile(r"AIzaSy[a-zA-Z0-9\-_]{30,}"),     # Google
        re.compile(r"sk-ant-[a-zA-Z0-9\-_]{20,}"),    # Anthropic legacy
        re.compile(r"xai-[a-zA-Z0-9\-_]{20,}"),       # xAI
        re.compile(r"key-[a-zA-Z0-9\-_]{20,}"),       # Generic key prefix
        re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*"),  # Bearer tokens
    ]

    _REDACTED: str = "***REDACTED***"

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Scrub secrets from all string values in the event dict."""
        return self._scrub_dict(event_dict)

    # -- helpers -----------------------------------------------------------

    def _scrub_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively scrub a dict."""
        cleaned: dict[str, Any] = {}
        for key, value in data.items():
            cleaned[key] = self._scrub_value(value)
        return cleaned

    def _scrub_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._scrub_string(value)
        if isinstance(value, dict):
            return self._scrub_dict(value)
        if isinstance(value, (list, tuple)):
            return type(value)(self._scrub_value(v) for v in value)
        return value

    def _scrub_string(self, text: str) -> str:
        result = text
        for pattern in self.PATTERNS:
            result = pattern.sub(self._REDACTED, result)
        return result


# ---------------------------------------------------------------------------
# File handler factories
# ---------------------------------------------------------------------------


def _create_file_handler(
    log_dir: Path,
    max_bytes: int,
    backup_count: int,
) -> logging.Handler:
    """Create a rotating JSON file handler for application logs.

    Args:
        log_dir: Directory to write log files into.
        max_bytes: Maximum size per log file before rotation.
        backup_count: Number of rotated backup files to keep.

    Returns:
        A configured ``RotatingFileHandler``.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        filename=str(log_dir / "prism.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def _create_audit_handler(log_dir: Path) -> logging.Handler:
    """Create a separate audit log handler for file/terminal operations.

    The audit log rotates at 10 MB and keeps 5 backups (matching the
    settings in ``prism.config.defaults``).

    Args:
        log_dir: Directory to write the audit log into.

    Returns:
        A configured ``RotatingFileHandler`` for audit entries.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        filename=str(log_dir / "audit.log"),
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    console_output: bool = True,
    json_file: bool = True,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> None:
    """Configure structlog with rotation, never logging secrets.

    Args:
        level: Root log level (``DEBUG``, ``INFO``, ``WARNING``, etc.).
        log_dir: Directory for log files. ``None`` disables file logging.
        console_output: Whether to emit logs to the console (stderr).
        json_file: Whether to write JSON-formatted logs to a file.
        max_bytes: Maximum size per log file before rotation (default 10 MB).
        backup_count: Number of rotated backup files to keep.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # -- shared structlog processors --------------------------------------
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        SecretScrubber(),
    ]

    # -- stdlib logging config --------------------------------------------
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    # Remove pre-existing handlers to avoid duplicates on reconfigure
    root_logger.handlers.clear()

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(console_handler)

    if json_file and log_dir is not None:
        file_handler = _create_file_handler(log_dir, max_bytes, backup_count)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)

    # -- structlog wiring --------------------------------------------------
    renderer: Any
    if json_file and log_dir is not None:
        renderer = structlog.processors.JSONRenderer()
    elif console_output:
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Attach a ProcessorFormatter to every handler so that structlog's
    # processors are applied uniformly.
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger for *name*.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A structlog bound logger.
    """
    return structlog.get_logger(name)
