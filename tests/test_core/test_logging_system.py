"""Tests for the enhanced logging system."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from prism.core.logging_system import (
    MAX_LOG_SIZE_BYTES,
    MAX_ROTATIONS,
    SECRET_PATTERNS,
    LogConfig,
    LogRotator,
    PrismLogger,
    SecretScrubber,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# LogConfig
# ---------------------------------------------------------------------------


class TestLogConfig:
    """Tests for the LogConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        config = LogConfig(log_dir=tmp_path)
        assert config.general_log == "prism.log"
        assert config.audit_log == "audit.log"
        assert config.cost_log == "cost.log"
        assert config.routing_log == "routing.log"
        assert config.max_size_bytes == MAX_LOG_SIZE_BYTES
        assert config.max_rotations == MAX_ROTATIONS
        assert config.level == "INFO"
        assert config.json_format is True

    def test_custom_values(self, tmp_path: Path) -> None:
        config = LogConfig(
            log_dir=tmp_path,
            general_log="app.log",
            max_size_bytes=1024,
            max_rotations=3,
            level="DEBUG",
            json_format=False,
        )
        assert config.general_log == "app.log"
        assert config.max_size_bytes == 1024
        assert config.max_rotations == 3
        assert config.level == "DEBUG"
        assert config.json_format is False


# ---------------------------------------------------------------------------
# LogRotator
# ---------------------------------------------------------------------------


class TestLogRotator:
    """Tests for the LogRotator class."""

    def test_no_rotation_when_file_missing(self, tmp_path: Path) -> None:
        config = LogConfig(log_dir=tmp_path)
        rotator = LogRotator(config)

        assert rotator.rotate_if_needed("prism.log") is False

    def test_no_rotation_when_under_limit(self, tmp_path: Path) -> None:
        config = LogConfig(log_dir=tmp_path, max_size_bytes=1024)
        rotator = LogRotator(config)

        log_file = tmp_path / "prism.log"
        log_file.write_text("small content")

        assert rotator.rotate_if_needed("prism.log") is False
        assert log_file.is_file()

    def test_rotation_when_over_limit(self, tmp_path: Path) -> None:
        config = LogConfig(log_dir=tmp_path, max_size_bytes=100)
        rotator = LogRotator(config)

        log_file = tmp_path / "prism.log"
        log_file.write_text("x" * 200)

        assert rotator.rotate_if_needed("prism.log") is True
        assert not log_file.is_file()
        rotated = tmp_path / "prism.log.1"
        assert rotated.is_file()
        assert rotated.read_text() == "x" * 200

    def test_rotation_shifts_existing_files(self, tmp_path: Path) -> None:
        config = LogConfig(log_dir=tmp_path, max_size_bytes=100, max_rotations=3)
        rotator = LogRotator(config)

        # Create existing rotated files
        (tmp_path / "prism.log.1").write_text("old1")
        (tmp_path / "prism.log.2").write_text("old2")
        log_file = tmp_path / "prism.log"
        log_file.write_text("x" * 200)

        rotator.rotate_if_needed("prism.log")

        assert (tmp_path / "prism.log.1").read_text() == "x" * 200
        assert (tmp_path / "prism.log.2").read_text() == "old1"
        assert (tmp_path / "prism.log.3").read_text() == "old2"

    def test_rotation_deletes_oldest_beyond_max(self, tmp_path: Path) -> None:
        config = LogConfig(log_dir=tmp_path, max_size_bytes=100, max_rotations=2)
        rotator = LogRotator(config)

        (tmp_path / "prism.log.1").write_text("old1")
        (tmp_path / "prism.log.2").write_text("old2")
        log_file = tmp_path / "prism.log"
        log_file.write_text("x" * 200)

        rotator.rotate_if_needed("prism.log")

        # .3 should be deleted (max_rotations=2)
        assert not (tmp_path / "prism.log.3").is_file()

    def test_get_log_size_existing_file(self, tmp_path: Path) -> None:
        config = LogConfig(log_dir=tmp_path)
        rotator = LogRotator(config)

        log_file = tmp_path / "prism.log"
        content = "test log content\n"
        log_file.write_text(content)

        size = rotator.get_log_size("prism.log")
        assert size == len(content)

    def test_get_log_size_missing_file(self, tmp_path: Path) -> None:
        config = LogConfig(log_dir=tmp_path)
        rotator = LogRotator(config)

        assert rotator.get_log_size("nonexistent.log") == 0

    def test_creates_log_dir(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "subdir" / "logs"
        config = LogConfig(log_dir=log_dir)
        LogRotator(config)

        assert log_dir.is_dir()


# ---------------------------------------------------------------------------
# SecretScrubber
# ---------------------------------------------------------------------------


class TestSecretScrubber:
    """Tests for the SecretScrubber class."""

    def test_scrub_openai_key(self) -> None:
        scrubber = SecretScrubber()
        text = "Using key sk-abcdefghijklmnopqrstuvwxyz"
        result = scrubber.scrub(text)
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in result
        assert "[REDACTED]" in result

    def test_scrub_anthropic_key(self) -> None:
        scrubber = SecretScrubber()
        text = "Key: sk-ant-api03-xxxxxxxxxxxxxxxxxxxx"
        result = scrubber.scrub(text)
        assert "sk-ant-" not in result
        assert "[REDACTED]" in result

    def test_scrub_groq_key(self) -> None:
        scrubber = SecretScrubber()
        text = "Groq: gsk_abcdefghijklmnopqrstuvwxyz"
        result = scrubber.scrub(text)
        assert "gsk_" not in result
        assert "[REDACTED]" in result

    def test_scrub_google_ai_key(self) -> None:
        scrubber = SecretScrubber()
        text = "Google: AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
        result = scrubber.scrub(text)
        assert "AIzaSy" not in result
        assert "[REDACTED]" in result

    def test_scrub_password_assignment(self) -> None:
        scrubber = SecretScrubber()
        text = "password = my_super_secret_password"
        result = scrubber.scrub(text)
        assert "my_super_secret" not in result
        assert "[REDACTED]" in result

    def test_scrub_token_assignment(self) -> None:
        scrubber = SecretScrubber()
        text = "token: bearer_abc123_secret_value"
        result = scrubber.scrub(text)
        assert "bearer_abc123" not in result

    def test_scrub_preserves_safe_text(self) -> None:
        scrubber = SecretScrubber()
        text = "Hello, this is a normal log message about routing."
        result = scrubber.scrub(text)
        assert result == text

    def test_scrub_dict_redacts_sensitive_keys(self) -> None:
        scrubber = SecretScrubber()
        data = {
            "api_key": "sk-secret-key-12345678901234567890",
            "model": "gpt-4",
            "token": "bearer-abc123",
            "message": "normal text",
        }
        result = scrubber.scrub_dict(data)
        assert result["api_key"] == "[REDACTED]"
        assert result["token"] == "[REDACTED]"
        assert result["model"] == "gpt-4"
        assert result["message"] == "normal text"

    def test_scrub_dict_redacts_password_key(self) -> None:
        scrubber = SecretScrubber()
        data = {"password": "super_secret", "username": "admin"}
        result = scrubber.scrub_dict(data)
        assert result["password"] == "[REDACTED]"
        assert result["username"] == "admin"

    def test_scrub_dict_redacts_credential_key(self) -> None:
        scrubber = SecretScrubber()
        data = {"credential": "my_cred", "data": "safe"}
        result = scrubber.scrub_dict(data)
        assert result["credential"] == "[REDACTED]"

    def test_scrub_dict_redacts_authorization_key(self) -> None:
        scrubber = SecretScrubber()
        data = {"authorization": "Bearer xyz", "status": "ok"}
        result = scrubber.scrub_dict(data)
        assert result["authorization"] == "[REDACTED]"

    def test_scrub_dict_nested(self) -> None:
        scrubber = SecretScrubber()
        data = {
            "config": {
                "api_key": "secret_value",
                "model": "claude-3",
            },
            "level": "INFO",
        }
        result = scrubber.scrub_dict(data)
        assert result["config"]["api_key"] == "[REDACTED]"
        assert result["config"]["model"] == "claude-3"

    def test_scrub_dict_non_string_values_preserved(self) -> None:
        scrubber = SecretScrubber()
        data = {"count": 42, "ratio": 3.14, "active": True, "tags": ["a", "b"]}
        result = scrubber.scrub_dict(data)
        assert result["count"] == 42
        assert result["ratio"] == 3.14
        assert result["active"] is True
        assert result["tags"] == ["a", "b"]

    def test_scrub_with_extra_patterns(self) -> None:
        custom_pattern = re.compile(r"CUSTOM-[A-Z0-9]{10,}")
        scrubber = SecretScrubber(extra_patterns=[custom_pattern])
        text = "Token: CUSTOM-ABCDEF1234567890"
        result = scrubber.scrub(text)
        assert "CUSTOM-ABCDEF" not in result
        assert "[REDACTED]" in result

    def test_secret_patterns_are_compiled(self) -> None:
        for pattern in SECRET_PATTERNS:
            assert isinstance(pattern, re.Pattern)

    def test_scrub_secret_key_in_value(self) -> None:
        scrubber = SecretScrubber()
        data = {"secret": "should_be_redacted"}
        result = scrubber.scrub_dict(data)
        assert result["secret"] == "[REDACTED]"


# ---------------------------------------------------------------------------
# PrismLogger
# ---------------------------------------------------------------------------


class TestPrismLogger:
    """Tests for the PrismLogger class."""

    def _make_logger(self, tmp_path: Path) -> PrismLogger:
        config = LogConfig(log_dir=tmp_path)
        return PrismLogger(config)

    def test_log_creates_file(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.log("general", "INFO", "test message")

        log_file = tmp_path / "prism.log"
        assert log_file.is_file()
        content = log_file.read_text()
        assert "test message" in content

    def test_log_writes_json(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.log("general", "INFO", "json test")

        log_file = tmp_path / "prism.log"
        line = log_file.read_text().strip()
        entry = json.loads(line)
        assert entry["message"] == "json test"
        assert entry["level"] == "INFO"
        assert entry["category"] == "general"
        assert "timestamp" in entry

    def test_log_plain_text_format(self, tmp_path: Path) -> None:
        config = LogConfig(log_dir=tmp_path, json_format=False)
        logger = PrismLogger(config)
        logger.log("general", "WARNING", "plain test")

        log_file = tmp_path / "prism.log"
        content = log_file.read_text().strip()
        assert "[WARNING]" in content
        assert "plain test" in content

    def test_log_audit(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.log_audit("file_read", "/path/to/file.py", user="test")

        log_file = tmp_path / "audit.log"
        assert log_file.is_file()
        line = log_file.read_text().strip()
        entry = json.loads(line)
        assert entry["operation"] == "file_read"
        assert entry["target"] == "/path/to/file.py"
        assert "file_read: /path/to/file.py" in entry["message"]

    def test_log_cost(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.log_cost("gpt-4", 0.03, 1500, provider="openai")

        log_file = tmp_path / "cost.log"
        assert log_file.is_file()
        line = log_file.read_text().strip()
        entry = json.loads(line)
        assert entry["model"] == "gpt-4"
        assert entry["cost"] == 0.03
        assert entry["tokens"] == 1500
        assert "$0.030000" in entry["message"]

    def test_log_routing(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.log_routing("code_edit", "claude-3-haiku", "cheapest for simple task")

        log_file = tmp_path / "routing.log"
        assert log_file.is_file()
        line = log_file.read_text().strip()
        entry = json.loads(line)
        assert entry["task"] == "code_edit"
        assert entry["model"] == "claude-3-haiku"
        assert entry["reason"] == "cheapest for simple task"

    def test_log_scrubs_secrets(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.log("general", "INFO", "test", api_key="sk-secret12345678901234567890")

        log_file = tmp_path / "prism.log"
        line = log_file.read_text().strip()
        entry = json.loads(line)
        assert entry["api_key"] == "[REDACTED]"

    def test_log_to_correct_category_files(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)

        logger.log("general", "INFO", "general msg")
        logger.log("audit", "INFO", "audit msg")
        logger.log("cost", "INFO", "cost msg")
        logger.log("routing", "INFO", "routing msg")

        assert (tmp_path / "prism.log").is_file()
        assert (tmp_path / "audit.log").is_file()
        assert (tmp_path / "cost.log").is_file()
        assert (tmp_path / "routing.log").is_file()

    def test_log_unknown_category_uses_general(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.log("unknown_category", "INFO", "fallback test")

        log_file = tmp_path / "prism.log"
        assert log_file.is_file()
        content = log_file.read_text()
        assert "fallback test" in content

    def test_read_logs_empty(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        result = logger.read_logs("general")
        assert result == []

    def test_read_logs_returns_lines(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.log("general", "INFO", "line1")
        logger.log("general", "WARNING", "line2")
        logger.log("general", "ERROR", "line3")

        result = logger.read_logs("general")
        assert len(result) == 3

    def test_read_logs_limits_lines(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        for i in range(10):
            logger.log("general", "INFO", f"message {i}")

        result = logger.read_logs("general", lines=3)
        assert len(result) == 3
        # Should be the last 3 lines
        last_entry = json.loads(result[-1])
        assert last_entry["message"] == "message 9"

    def test_read_logs_filter_by_level(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.log("general", "INFO", "info msg")
        logger.log("general", "WARNING", "warn msg")
        logger.log("general", "ERROR", "error msg")
        logger.log("general", "INFO", "info msg 2")

        result = logger.read_logs("general", level="WARNING")
        assert len(result) == 1
        entry = json.loads(result[0])
        assert entry["level"] == "WARNING"

    def test_read_logs_filter_by_level_case_insensitive(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.log("general", "ERROR", "error msg")
        logger.log("general", "INFO", "info msg")

        result = logger.read_logs("general", level="error")
        assert len(result) == 1

    def test_read_logs_nonexistent_category(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        result = logger.read_logs("cost")
        assert result == []

    def test_clear_logs_specific_category(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.log("general", "INFO", "test")
        logger.log("audit", "INFO", "audit test")

        cleared = logger.clear_logs("general")
        assert cleared == 1
        assert (tmp_path / "prism.log").read_text() == ""
        # audit should still have content
        assert (tmp_path / "audit.log").read_text().strip() != ""

    def test_clear_logs_all_categories(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.log("general", "INFO", "test")
        logger.log("audit", "INFO", "test")
        logger.log("cost", "INFO", "test")
        logger.log("routing", "INFO", "test")

        cleared = logger.clear_logs()
        assert cleared == 4
        assert (tmp_path / "prism.log").read_text() == ""
        assert (tmp_path / "audit.log").read_text() == ""
        assert (tmp_path / "cost.log").read_text() == ""
        assert (tmp_path / "routing.log").read_text() == ""

    def test_clear_logs_missing_files(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        cleared = logger.clear_logs("cost")
        assert cleared == 0

    def test_clear_all_when_no_files_exist(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        cleared = logger.clear_logs()
        assert cleared == 0

    def test_creates_log_dir_on_init(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "deep" / "nested" / "logs"
        config = LogConfig(log_dir=log_dir)
        PrismLogger(config)
        assert log_dir.is_dir()

    def test_log_with_extra_kwargs(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.log("general", "INFO", "custom", request_id="abc-123", duration_ms=42)

        log_file = tmp_path / "prism.log"
        entry = json.loads(log_file.read_text().strip())
        assert entry["request_id"] == "abc-123"
        assert entry["duration_ms"] == 42

    def test_multiple_logs_append(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.log("general", "INFO", "first")
        logger.log("general", "INFO", "second")

        log_file = tmp_path / "prism.log"
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["message"] == "first"
        assert json.loads(lines[1])["message"] == "second"


class TestPrismLoggerGetLogFile:
    """Tests for the _get_log_file mapping."""

    def test_general_category(self, tmp_path: Path) -> None:
        config = LogConfig(log_dir=tmp_path)
        logger = PrismLogger(config)
        assert logger._get_log_file("general") == "prism.log"

    def test_audit_category(self, tmp_path: Path) -> None:
        config = LogConfig(log_dir=tmp_path)
        logger = PrismLogger(config)
        assert logger._get_log_file("audit") == "audit.log"

    def test_cost_category(self, tmp_path: Path) -> None:
        config = LogConfig(log_dir=tmp_path)
        logger = PrismLogger(config)
        assert logger._get_log_file("cost") == "cost.log"

    def test_routing_category(self, tmp_path: Path) -> None:
        config = LogConfig(log_dir=tmp_path)
        logger = PrismLogger(config)
        assert logger._get_log_file("routing") == "routing.log"

    def test_unknown_category_falls_back_to_general(self, tmp_path: Path) -> None:
        config = LogConfig(log_dir=tmp_path)
        logger = PrismLogger(config)
        assert logger._get_log_file("xyz") == "prism.log"
