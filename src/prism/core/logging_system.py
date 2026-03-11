"""Enhanced logging system — structured JSON logs with rotation and secret scrubbing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 — used at runtime in dataclass fields and methods

import structlog

logger = structlog.get_logger(__name__)

MAX_LOG_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_ROTATIONS = 5

# Secret patterns to never log
SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI keys
    re.compile(r"sk-ant-[a-zA-Z0-9\-]{20,}"),  # Anthropic keys
    re.compile(r"gsk_[a-zA-Z0-9]{20,}"),  # Groq keys
    re.compile(r"AI[a-zA-Z0-9]{35,}"),  # Google AI keys
    re.compile(
        r"(password|secret|token|key|credential)\s*[=:]\s*\S+",
        re.IGNORECASE,
    ),
]


@dataclass
class LogConfig:
    """Logging configuration.

    Args:
        log_dir: Directory where log files are stored.
        general_log: Filename for the general application log.
        audit_log: Filename for audit events (file/terminal ops).
        cost_log: Filename for cost tracking events.
        routing_log: Filename for routing decision events.
        max_size_bytes: Maximum single log file size before rotation.
        max_rotations: Number of rotated files to keep.
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        json_format: Whether to write JSON-structured log lines.
    """

    log_dir: Path
    general_log: str = "prism.log"
    audit_log: str = "audit.log"
    cost_log: str = "cost.log"
    routing_log: str = "routing.log"
    max_size_bytes: int = MAX_LOG_SIZE_BYTES
    max_rotations: int = MAX_ROTATIONS
    level: str = "INFO"
    json_format: bool = True


class LogRotator:
    """Manages log file rotation.

    When a log file exceeds ``max_size_bytes``, it is renamed with a
    numeric suffix and older rotations are shifted up.  The oldest file
    beyond ``max_rotations`` is deleted.

    Args:
        config: The logging configuration to use.
    """

    def __init__(self, config: LogConfig) -> None:
        self._config = config
        config.log_dir.mkdir(parents=True, exist_ok=True)

    def rotate_if_needed(self, log_name: str) -> bool:
        """Rotate a log file if it exceeds max size.

        Args:
            log_name: Filename (not path) of the log to check.

        Returns:
            ``True`` if the file was rotated, ``False`` otherwise.
        """
        log_path = self._config.log_dir / log_name

        if not log_path.is_file():
            return False

        if log_path.stat().st_size < self._config.max_size_bytes:
            return False

        # Rotate: .log.4 -> .log.5, .log.3 -> .log.4, etc.
        for i in range(self._config.max_rotations - 1, 0, -1):
            src = self._config.log_dir / f"{log_name}.{i}"
            dst = self._config.log_dir / f"{log_name}.{i + 1}"
            if src.is_file():
                src.rename(dst)

        # Current -> .1
        rotated = self._config.log_dir / f"{log_name}.1"
        log_path.rename(rotated)

        # Delete oldest if exceeds max rotations
        oldest = self._config.log_dir / f"{log_name}.{self._config.max_rotations + 1}"
        oldest.unlink(missing_ok=True)

        logger.debug("log_rotated", log=log_name)
        return True

    def get_log_size(self, log_name: str) -> int:
        """Get current log file size in bytes.

        Args:
            log_name: Filename (not path) of the log.

        Returns:
            File size in bytes, or ``0`` if the file does not exist.
        """
        log_path = self._config.log_dir / log_name
        if log_path.is_file():
            return log_path.stat().st_size
        return 0


class SecretScrubber:
    """Scrubs secrets from log output before writing.

    Applies a set of regular expressions to detect and redact API keys,
    tokens, and other sensitive values from text and dictionaries.

    Args:
        extra_patterns: Additional regex patterns to apply on top of the
            built-in ``SECRET_PATTERNS``.
    """

    def __init__(self, extra_patterns: list[re.Pattern[str]] | None = None) -> None:
        self._patterns: list[re.Pattern[str]] = list(SECRET_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def scrub(self, text: str) -> str:
        """Remove secrets from text.

        Args:
            text: Input string that may contain secrets.

        Returns:
            Sanitised string with secrets replaced by ``[REDACTED]``.
        """
        result = text
        for pattern in self._patterns:
            result = pattern.sub("[REDACTED]", result)
        return result

    def scrub_dict(self, data: dict) -> dict:
        """Scrub secrets from a dictionary (for JSON logging).

        Keys whose lowercase name matches a known sensitive keyword are
        unconditionally redacted.  String values are run through
        :meth:`scrub`.  Nested dicts are processed recursively.

        Args:
            data: Dictionary to sanitise.

        Returns:
            A new dictionary with secrets redacted.
        """
        scrubbed: dict = {}
        sensitive_keys = {
            "api_key",
            "token",
            "secret",
            "password",
            "credential",
            "authorization",
        }

        for key, value in data.items():
            if key.lower() in sensitive_keys:
                scrubbed[key] = "[REDACTED]"
            elif isinstance(value, str):
                scrubbed[key] = self.scrub(value)
            elif isinstance(value, dict):
                scrubbed[key] = self.scrub_dict(value)
            else:
                scrubbed[key] = value

        return scrubbed


class PrismLogger:
    """Structured logger with categorized outputs and secret scrubbing.

    Writes log entries as JSON lines to separate files per category
    (general, audit, cost, routing).  All entries are scrubbed for
    secrets before being written, and log files are automatically
    rotated when they exceed the configured size limit.

    Args:
        config: Logging configuration.
    """

    def __init__(self, config: LogConfig) -> None:
        self._config = config
        self._rotator = LogRotator(config)
        self._scrubber = SecretScrubber()
        config.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, category: str, level: str, message: str, **kwargs: object) -> None:
        """Write a log entry to the appropriate log file.

        Args:
            category: Log category (``"general"``, ``"audit"``,
                ``"cost"``, ``"routing"``).
            level: Log level string (``"DEBUG"``, ``"INFO"``, etc.).
            message: Human-readable log message.
            **kwargs: Additional structured fields to include in the
                JSON entry.
        """
        log_name = self._get_log_file(category)

        entry: dict = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "category": category,
            "message": message,
            **kwargs,
        }

        # Scrub secrets
        entry = self._scrubber.scrub_dict(entry)

        # Rotate if needed
        self._rotator.rotate_if_needed(log_name)

        # Write
        log_path = self._config.log_dir / log_name
        with log_path.open("a") as f:
            if self._config.json_format:
                f.write(json.dumps(entry) + "\n")
            else:
                f.write(f"[{entry['timestamp']}] [{level}] {message}\n")

    def log_audit(self, operation: str, target: str, **kwargs: object) -> None:
        """Log an audit event (file/terminal operations).

        Args:
            operation: The operation performed (e.g., ``"file_read"``).
            target: The target of the operation (e.g., a file path).
            **kwargs: Extra structured fields.
        """
        self.log(
            "audit",
            "INFO",
            f"{operation}: {target}",
            operation=operation,
            target=target,
            **kwargs,
        )

    def log_cost(self, model: str, cost: float, tokens: int, **kwargs: object) -> None:
        """Log a cost event.

        Args:
            model: Model identifier that was used.
            cost: Cost in USD.
            tokens: Number of tokens consumed.
            **kwargs: Extra structured fields.
        """
        self.log(
            "cost",
            "INFO",
            f"${cost:.6f} for {model}",
            model=model,
            cost=cost,
            tokens=tokens,
            **kwargs,
        )

    def log_routing(
        self, task: str, model: str, reason: str, **kwargs: object
    ) -> None:
        """Log a routing decision.

        Args:
            task: Description of the task being routed.
            model: The model that was selected.
            reason: Why this model was chosen.
            **kwargs: Extra structured fields.
        """
        self.log(
            "routing",
            "INFO",
            f"{task} -> {model}: {reason}",
            task=task,
            model=model,
            reason=reason,
            **kwargs,
        )

    def read_logs(
        self,
        category: str = "general",
        lines: int = 50,
        level: str | None = None,
    ) -> list[str]:
        """Read recent log lines.

        Args:
            category: Log category to read from.
            lines: Maximum number of lines to return (from the end).
            level: If given, only return lines matching this level.

        Returns:
            List of raw log line strings (newest last).
        """
        log_name = self._get_log_file(category)
        log_path = self._config.log_dir / log_name

        if not log_path.is_file():
            return []

        all_lines = log_path.read_text().strip().split("\n")

        # Filter empty lines that result from stripping an empty file
        all_lines = [line for line in all_lines if line]

        if level:
            filtered: list[str] = []
            for line in all_lines:
                try:
                    entry = json.loads(line)
                    if entry.get("level", "").upper() == level.upper():
                        filtered.append(line)
                except json.JSONDecodeError:
                    if level.upper() in line.upper():
                        filtered.append(line)
            all_lines = filtered

        return all_lines[-lines:]

    def clear_logs(self, category: str | None = None) -> int:
        """Clear log files.

        Args:
            category: Specific category to clear. ``None`` clears all.

        Returns:
            Number of files cleared.
        """
        cleared = 0
        if category:
            log_name = self._get_log_file(category)
            log_path = self._config.log_dir / log_name
            if log_path.is_file():
                log_path.write_text("")
                cleared = 1
        else:
            for log_name in [
                self._config.general_log,
                self._config.audit_log,
                self._config.cost_log,
                self._config.routing_log,
            ]:
                log_path = self._config.log_dir / log_name
                if log_path.is_file():
                    log_path.write_text("")
                    cleared += 1
        return cleared

    def _get_log_file(self, category: str) -> str:
        """Map category to log file name.

        Args:
            category: Log category string.

        Returns:
            The corresponding log filename.
        """
        mapping = {
            "general": self._config.general_log,
            "audit": self._config.audit_log,
            "cost": self._config.cost_log,
            "routing": self._config.routing_log,
        }
        return mapping.get(category, self._config.general_log)
