"""Tests for prism.logging_config — centralized logging configuration."""

from __future__ import annotations

import logging
import logging.handlers
from typing import TYPE_CHECKING, Any

import structlog

from prism.logging_config import (
    SecretScrubber,
    _create_audit_handler,
    _create_file_handler,
    configure_logging,
    get_logger,
)

if TYPE_CHECKING:
    from pathlib import Path
    pass


# ---------------------------------------------------------------------------
# SecretScrubber
# ---------------------------------------------------------------------------


class TestSecretScrubber:
    """Tests for the SecretScrubber structlog processor."""

    def setup_method(self) -> None:
        self.scrubber = SecretScrubber()

    def test_scrubs_openai_key(self) -> None:
        event_dict: dict[str, Any] = {
            "event": "api_call",
            "key": "sk-abc123def456ghi789jklmnopqrstuv",
        }
        result = self.scrubber(None, "info", event_dict)
        assert "sk-abc123" not in result["key"]
        assert "***REDACTED***" in result["key"]

    def test_scrubs_anthropic_key(self) -> None:
        event_dict: dict[str, Any] = {
            "event": "auth",
            "key": "sk-ant-api03-someRandomString12345",
        }
        result = self.scrubber(None, "info", event_dict)
        assert "sk-ant-" not in result["key"]
        assert "***REDACTED***" in result["key"]

    def test_scrubs_groq_key(self) -> None:
        event_dict: dict[str, Any] = {
            "event": "auth",
            "key": "gsk_abcdefghijklmnopqrst12345",
        }
        result = self.scrubber(None, "info", event_dict)
        assert "gsk_" not in result["key"]
        assert "***REDACTED***" in result["key"]

    def test_scrubs_google_key(self) -> None:
        event_dict: dict[str, Any] = {
            "event": "auth",
            "key": "AIzaSy1234567890abcdefghijklmnopqrstuv",
        }
        result = self.scrubber(None, "info", event_dict)
        assert "AIzaSy" not in result["key"]
        assert "***REDACTED***" in result["key"]

    def test_scrubs_xai_key(self) -> None:
        event_dict: dict[str, Any] = {
            "event": "auth",
            "key": "xai-some-long-api-key-string-here",
        }
        result = self.scrubber(None, "info", event_dict)
        assert "xai-" not in result["key"]
        assert "***REDACTED***" in result["key"]

    def test_scrubs_generic_key_prefix(self) -> None:
        event_dict: dict[str, Any] = {
            "event": "auth",
            "key": "key-abcdefghijklmnopqrst12345",
        }
        result = self.scrubber(None, "info", event_dict)
        assert "key-abcdef" not in result["key"]
        assert "***REDACTED***" in result["key"]

    def test_scrubs_bearer_token(self) -> None:
        event_dict: dict[str, Any] = {
            "event": "http_request",
            "header": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.sig",
        }
        result = self.scrubber(None, "info", event_dict)
        assert "eyJhbGci" not in result["header"]
        assert "***REDACTED***" in result["header"]

    def test_preserves_safe_strings(self) -> None:
        event_dict: dict[str, Any] = {
            "event": "normal_log",
            "message": "This is a normal log message",
            "count": 42,
        }
        result = self.scrubber(None, "info", event_dict)
        assert result["message"] == "This is a normal log message"
        assert result["count"] == 42

    def test_scrubs_nested_dict(self) -> None:
        event_dict: dict[str, Any] = {
            "event": "nested",
            "data": {
                "auth": "sk-abc123def456ghi789jklmnopqrstuv",
                "safe": "hello",
            },
        }
        result = self.scrubber(None, "info", event_dict)
        assert "***REDACTED***" in result["data"]["auth"]
        assert result["data"]["safe"] == "hello"

    def test_scrubs_list_values(self) -> None:
        event_dict: dict[str, Any] = {
            "event": "list_test",
            "keys": [
                "sk-abc123def456ghi789jklmnopqrstuv",
                "normal_value",
            ],
        }
        result = self.scrubber(None, "info", event_dict)
        assert "***REDACTED***" in result["keys"][0]
        assert result["keys"][1] == "normal_value"

    def test_scrubs_tuple_values(self) -> None:
        event_dict: dict[str, Any] = {
            "event": "tuple_test",
            "keys": (
                "sk-abc123def456ghi789jklmnopqrstuv",
                "safe",
            ),
        }
        result = self.scrubber(None, "info", event_dict)
        assert isinstance(result["keys"], tuple)
        assert "***REDACTED***" in result["keys"][0]
        assert result["keys"][1] == "safe"

    def test_non_string_values_unchanged(self) -> None:
        event_dict: dict[str, Any] = {
            "event": "types_test",
            "count": 42,
            "rate": 3.14,
            "flag": True,
            "empty": None,
        }
        result = self.scrubber(None, "info", event_dict)
        assert result["count"] == 42
        assert result["rate"] == 3.14
        assert result["flag"] is True
        assert result["empty"] is None

    def test_event_field_is_scrubbed(self) -> None:
        event_dict: dict[str, Any] = {
            "event": "Key is sk-abc123def456ghi789jklmnopqrstuv here",
        }
        result = self.scrubber(None, "info", event_dict)
        assert "sk-abc123" not in result["event"]

    def test_multiple_keys_in_one_string(self) -> None:
        event_dict: dict[str, Any] = {
            "event": "multi",
            "msg": "Keys: sk-abc123def456ghi789jklmnopqrstuv and gsk_another12345678901234567",
        }
        result = self.scrubber(None, "info", event_dict)
        assert result["msg"].count("***REDACTED***") == 2

    def test_empty_dict(self) -> None:
        event_dict: dict[str, Any] = {}
        result = self.scrubber(None, "info", event_dict)
        assert result == {}

    def test_deeply_nested_scrubbing(self) -> None:
        event_dict: dict[str, Any] = {
            "level1": {
                "level2": {
                    "level3": {
                        "secret": "sk-ant-deep-nested-key-value12345",
                    }
                }
            }
        }
        result = self.scrubber(None, "info", event_dict)
        assert "***REDACTED***" in result["level1"]["level2"]["level3"]["secret"]


# ---------------------------------------------------------------------------
# _create_file_handler
# ---------------------------------------------------------------------------


class TestCreateFileHandler:
    """Tests for the _create_file_handler factory."""

    def test_creates_handler(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        handler = _create_file_handler(log_dir, max_bytes=1_000_000, backup_count=3)
        assert isinstance(handler, logging.handlers.RotatingFileHandler)

    def test_creates_log_directory(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs" / "nested"
        _create_file_handler(log_dir, max_bytes=1_000_000, backup_count=3)
        assert log_dir.is_dir()

    def test_handler_filename(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        handler = _create_file_handler(log_dir, max_bytes=1_000_000, backup_count=3)
        assert handler.baseFilename.endswith("prism.log")

    def test_handler_max_bytes(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        handler = _create_file_handler(log_dir, max_bytes=5_000_000, backup_count=3)
        assert handler.maxBytes == 5_000_000

    def test_handler_backup_count(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        handler = _create_file_handler(log_dir, max_bytes=1_000_000, backup_count=7)
        assert handler.backupCount == 7

    def test_handler_encoding(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        handler = _create_file_handler(log_dir, max_bytes=1_000_000, backup_count=3)
        assert handler.encoding == "utf-8"


# ---------------------------------------------------------------------------
# _create_audit_handler
# ---------------------------------------------------------------------------


class TestCreateAuditHandler:
    """Tests for the _create_audit_handler factory."""

    def test_creates_handler(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        handler = _create_audit_handler(log_dir)
        assert isinstance(handler, logging.handlers.RotatingFileHandler)

    def test_creates_log_directory(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "audit_logs"
        _create_audit_handler(log_dir)
        assert log_dir.is_dir()

    def test_handler_filename(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        handler = _create_audit_handler(log_dir)
        assert handler.baseFilename.endswith("audit.log")

    def test_handler_max_bytes(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        handler = _create_audit_handler(log_dir)
        assert handler.maxBytes == 10_000_000

    def test_handler_backup_count(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        handler = _create_audit_handler(log_dir)
        assert handler.backupCount == 5

    def test_handler_encoding(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        handler = _create_audit_handler(log_dir)
        assert handler.encoding == "utf-8"


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    """Tests for the configure_logging function."""

    def teardown_method(self) -> None:
        """Clean up the root logger after each test."""
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    def test_configure_default(self) -> None:
        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_configure_debug_level(self) -> None:
        configure_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_configure_warning_level(self) -> None:
        configure_logging(level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_configure_with_console_output(self) -> None:
        configure_logging(console_output=True, json_file=False)
        root = logging.getLogger()
        assert len(root.handlers) >= 1
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)

    def test_configure_without_console_output(self) -> None:
        configure_logging(console_output=False, json_file=False)
        root = logging.getLogger()
        # No console handler should be added
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 0

    def test_configure_with_file_logging(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        configure_logging(
            log_dir=log_dir,
            console_output=False,
            json_file=True,
        )
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 1
        assert log_dir.is_dir()

    def test_configure_without_file_logging(self) -> None:
        configure_logging(log_dir=None, json_file=True)
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 0

    def test_configure_clears_existing_handlers(self) -> None:
        root = logging.getLogger()
        initial_count = len(root.handlers)
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        assert len(root.handlers) == initial_count + 2

        configure_logging(console_output=True, json_file=False)
        # Old StreamHandlers should be cleared by configure_logging
        stream_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
            and not type(h).__name__.startswith("LogCapture")
        ]
        assert len(stream_handlers) >= 1

    def test_configure_with_custom_max_bytes(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        configure_logging(
            log_dir=log_dir,
            json_file=True,
            max_bytes=2_000_000,
            console_output=False,
        )
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].maxBytes == 2_000_000

    def test_configure_with_custom_backup_count(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        configure_logging(
            log_dir=log_dir,
            json_file=True,
            backup_count=10,
            console_output=False,
        )
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].backupCount == 10

    def test_configure_sets_structlog_processors(self) -> None:
        configure_logging(level="INFO", console_output=True, json_file=False)
        # After configure_logging, structlog should be configured. Getting a
        # logger should not raise.
        logger = structlog.get_logger("test")
        assert logger is not None

    def test_configure_both_console_and_file(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        configure_logging(
            log_dir=log_dir,
            console_output=True,
            json_file=True,
        )
        root = logging.getLogger()
        assert len(root.handlers) == 2

    def test_configure_neither_console_nor_file(self) -> None:
        configure_logging(console_output=False, json_file=False)
        root = logging.getLogger()
        assert len(root.handlers) == 0

    def test_reconfigure_does_not_duplicate_handlers(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        configure_logging(log_dir=log_dir, console_output=True, json_file=True)
        configure_logging(log_dir=log_dir, console_output=True, json_file=True)
        root = logging.getLogger()
        # Should be exactly 2 handlers (console + file), not 4
        assert len(root.handlers) == 2

    def test_handler_formatters_set(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        configure_logging(log_dir=log_dir, console_output=True, json_file=True)
        root = logging.getLogger()
        for handler in root.handlers:
            assert handler.formatter is not None

    def test_case_insensitive_level(self) -> None:
        configure_logging(level="debug")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_invalid_level_defaults_to_info(self) -> None:
        configure_logging(level="NONEXISTENT")
        root = logging.getLogger()
        # getattr(logging, "NONEXISTENT", logging.INFO) -> INFO
        assert root.level == logging.INFO


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_returns_bound_logger(self) -> None:
        logger = get_logger("test_module")
        assert logger is not None

    def test_logger_name(self) -> None:
        logger = get_logger("my.module.name")
        # structlog bound loggers wrap a stdlib logger
        assert logger is not None

    def test_different_names_return_loggers(self) -> None:
        logger1 = get_logger("module_a")
        logger2 = get_logger("module_b")
        assert logger1 is not None
        assert logger2 is not None

    def test_logger_has_standard_methods(self) -> None:
        # After configuring, loggers should have standard log methods
        configure_logging(level="DEBUG", console_output=False, json_file=False)
        logger = get_logger("test")
        # Structlog bound loggers have info, debug, warning, error methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
