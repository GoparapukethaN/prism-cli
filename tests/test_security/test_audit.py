"""Tests for prism.security.audit.AuditLogger."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from prism.security.audit import AuditLogger
from prism.security.secret_filter import REDACTED

if TYPE_CHECKING:
    from pathlib import Path

# =====================================================================
# Log format and structure
# =====================================================================


class TestAuditLogFormat:
    """Verify the JSON-line log format is correct and complete."""

    def test_tool_execution_log_entry_structure(
        self, audit_logger: AuditLogger
    ) -> None:
        audit_logger.log_tool_execution(
            tool_name="bash",
            command="ls -la",
            exit_code=0,
            duration_ms=42.5,
            success=True,
        )
        # Force flush by accessing the handler
        for handler in audit_logger._logger.handlers:
            handler.flush()

        lines = audit_logger.log_path.read_text().strip().splitlines()
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert "timestamp" in entry
        assert entry["event"] == "tool_execution"
        assert entry["data"]["tool_name"] == "bash"
        assert entry["data"]["command"] == "ls -la"
        assert entry["data"]["exit_code"] == 0
        assert entry["data"]["duration_ms"] == 42.5
        assert entry["data"]["success"] is True

    def test_routing_decision_log_entry(
        self, audit_logger: AuditLogger
    ) -> None:
        audit_logger.log_routing_decision(
            task_tier="complex",
            selected_model="gpt-4o",
            candidates=["gpt-4o", "claude-3-opus", "gemini-pro"],
            scores={"gpt-4o": 0.95, "claude-3-opus": 0.90, "gemini-pro": 0.80},
            reason="highest_quality_score",
            budget_remaining=4.50,
            estimated_cost=0.0123,
        )
        for handler in audit_logger._logger.handlers:
            handler.flush()

        lines = audit_logger.log_path.read_text().strip().splitlines()
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["event"] == "routing_decision"
        data = entry["data"]
        assert data["task_tier"] == "complex"
        assert data["selected_model"] == "gpt-4o"
        assert data["candidates"] == ["gpt-4o", "claude-3-opus", "gemini-pro"]
        assert data["scores"]["gpt-4o"] == 0.95
        assert data["reason"] == "highest_quality_score"
        assert data["budget_remaining"] == 4.50
        assert data["estimated_cost"] == 0.0123

    def test_security_event_log_entry(
        self, audit_logger: AuditLogger
    ) -> None:
        audit_logger.log_security_event(
            event_type="path_traversal",
            detail="Attempted directory traversal detected",
            path="../../../etc/passwd",
            pattern=None,
        )
        for handler in audit_logger._logger.handlers:
            handler.flush()

        lines = audit_logger.log_path.read_text().strip().splitlines()
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["event"] == "security_event"
        assert entry["data"]["event_type"] == "path_traversal"
        assert entry["data"]["path"] == "../../../etc/passwd"
        assert "pattern" not in entry["data"]

    def test_timestamp_is_iso8601_utc(
        self, audit_logger: AuditLogger
    ) -> None:
        audit_logger.log_tool_execution(tool_name="test")
        for handler in audit_logger._logger.handlers:
            handler.flush()

        lines = audit_logger.log_path.read_text().strip().splitlines()
        entry = json.loads(lines[0])
        ts = entry["timestamp"]
        # Must end with +00:00 (UTC)
        assert ts.endswith("+00:00")

    def test_multiple_entries_on_separate_lines(
        self, audit_logger: AuditLogger
    ) -> None:
        audit_logger.log_tool_execution(tool_name="bash", command="echo 1")
        audit_logger.log_tool_execution(tool_name="bash", command="echo 2")
        audit_logger.log_tool_execution(tool_name="bash", command="echo 3")
        for handler in audit_logger._logger.handlers:
            handler.flush()

        lines = audit_logger.log_path.read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            entry = json.loads(line)
            assert entry["event"] == "tool_execution"

    def test_raw_log_event(self, audit_logger: AuditLogger) -> None:
        audit_logger.log_raw("custom_event", {"key": "value", "count": 42})
        for handler in audit_logger._logger.handlers:
            handler.flush()

        lines = audit_logger.log_path.read_text().strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["event"] == "custom_event"
        assert entry["data"]["key"] == "value"
        assert entry["data"]["count"] == 42


# =====================================================================
# Sensitive data sanitization
# =====================================================================


class TestAuditSanitization:
    """Verify that sensitive data is redacted before being written to the log."""

    def test_api_key_in_args_is_redacted(
        self, audit_logger: AuditLogger
    ) -> None:
        audit_logger.log_tool_execution(
            tool_name="http_request",
            args={
                "url": "https://api.openai.com/v1/chat",
                "openai_api_key": "sk-very-secret",
                "model": "gpt-4",
            },
        )
        for handler in audit_logger._logger.handlers:
            handler.flush()

        lines = audit_logger.log_path.read_text().strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["data"]["args"]["openai_api_key"] == REDACTED
        assert entry["data"]["args"]["url"] == "https://api.openai.com/v1/chat"
        assert entry["data"]["args"]["model"] == "gpt-4"

    def test_raw_log_sanitizes_data(
        self, audit_logger: AuditLogger
    ) -> None:
        audit_logger.log_raw(
            "provider_call",
            {"auth_token": "ghp_xxxx", "status": 200},
        )
        for handler in audit_logger._logger.handlers:
            handler.flush()

        lines = audit_logger.log_path.read_text().strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["data"]["auth_token"] == REDACTED
        assert entry["data"]["status"] == 200

    def test_nested_secrets_are_redacted(
        self, audit_logger: AuditLogger
    ) -> None:
        audit_logger.log_raw(
            "nested",
            {
                "provider": {
                    "name": "aws",
                    "aws_secret": "wJalrX...",
                    "region": "us-east-1",
                }
            },
        )
        for handler in audit_logger._logger.handlers:
            handler.flush()

        lines = audit_logger.log_path.read_text().strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["data"]["provider"]["aws_secret"] == REDACTED
        assert entry["data"]["provider"]["name"] == "aws"
        assert entry["data"]["provider"]["region"] == "us-east-1"


# =====================================================================
# File rotation
# =====================================================================


class TestLogRotation:
    """Verify that the rotating file handler configuration is correct."""

    def test_log_file_is_created(self, audit_logger: AuditLogger) -> None:
        audit_logger.log_tool_execution(tool_name="test")
        for handler in audit_logger._logger.handlers:
            handler.flush()
        assert audit_logger.log_path.exists()

    def test_parent_directory_is_created(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "c" / "audit.log"
        logger = AuditLogger(log_path=deep_path)
        logger.log_tool_execution(tool_name="test")
        for handler in logger._logger.handlers:
            handler.flush()
        assert deep_path.exists()

    def test_rotation_on_size_exceeded(self, tmp_path: Path) -> None:
        log_path = tmp_path / "small_audit.log"
        small_logger = AuditLogger(
            log_path=log_path,
            max_bytes=500,
            backup_count=2,
        )
        # Write enough entries to trigger rotation
        for i in range(50):
            small_logger.log_tool_execution(
                tool_name="bash",
                command=f"echo {'x' * 50} iteration {i}",
                exit_code=0,
                duration_ms=1.0,
            )
        for handler in small_logger._logger.handlers:
            handler.flush()

        # At least one backup file should have been created
        backups = list(tmp_path.glob("small_audit.log.*"))
        assert len(backups) >= 1


# =====================================================================
# Optional fields
# =====================================================================


class TestOptionalFields:
    """Verify that optional fields are omitted from the log when not provided."""

    def test_tool_execution_minimal(
        self, audit_logger: AuditLogger
    ) -> None:
        audit_logger.log_tool_execution(tool_name="read_file")
        for handler in audit_logger._logger.handlers:
            handler.flush()

        lines = audit_logger.log_path.read_text().strip().splitlines()
        entry = json.loads(lines[0])
        data = entry["data"]
        assert data["tool_name"] == "read_file"
        assert data["success"] is True
        assert "command" not in data
        assert "args" not in data
        assert "exit_code" not in data
        assert "duration_ms" not in data
        assert "error" not in data
        assert "user" not in data

    def test_tool_execution_with_error(
        self, audit_logger: AuditLogger
    ) -> None:
        audit_logger.log_tool_execution(
            tool_name="bash",
            command="bad_cmd",
            success=False,
            error="command not found",
            exit_code=127,
            duration_ms=5.0,
            user="testuser",
        )
        for handler in audit_logger._logger.handlers:
            handler.flush()

        lines = audit_logger.log_path.read_text().strip().splitlines()
        entry = json.loads(lines[0])
        data = entry["data"]
        assert data["success"] is False
        assert data["error"] == "command not found"
        assert data["exit_code"] == 127
        assert data["user"] == "testuser"

    def test_routing_decision_minimal(
        self, audit_logger: AuditLogger
    ) -> None:
        audit_logger.log_routing_decision(
            task_tier="simple",
            selected_model="gemini-flash",
        )
        for handler in audit_logger._logger.handlers:
            handler.flush()

        lines = audit_logger.log_path.read_text().strip().splitlines()
        entry = json.loads(lines[0])
        data = entry["data"]
        assert data["task_tier"] == "simple"
        assert data["selected_model"] == "gemini-flash"
        assert "candidates" not in data
        assert "scores" not in data
        assert "reason" not in data

    def test_security_event_with_command_and_pattern(
        self, audit_logger: AuditLogger
    ) -> None:
        audit_logger.log_security_event(
            event_type="blocked_command",
            detail="Destructive command blocked",
            command="rm -rf /",
            pattern=r"^rm\s+-rf\s+/$",
        )
        for handler in audit_logger._logger.handlers:
            handler.flush()

        lines = audit_logger.log_path.read_text().strip().splitlines()
        entry = json.loads(lines[0])
        data = entry["data"]
        assert data["event_type"] == "blocked_command"
        assert data["command"] == "rm -rf /"
        assert data["pattern"] == r"^rm\s+-rf\s+/$"
        assert "path" not in data
