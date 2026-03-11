"""Tests for ResponseValidator — 8+ tests, fully offline."""

from __future__ import annotations

import warnings

import pytest

from prism.llm.result import CompletionResult
from prism.llm.validation import ResponseValidator


def _make_result(
    content: str = "All good.",
    finish_reason: str = "stop",
    tool_calls: list[dict] | None = None,
    **kwargs: object,
) -> CompletionResult:
    """Helper to build a CompletionResult with sensible defaults."""
    defaults = {
        "model": "gpt-4o",
        "provider": "openai",
        "input_tokens": 10,
        "output_tokens": 5,
        "cached_tokens": 0,
        "cost_usd": 0.001,
        "latency_ms": 100.0,
    }
    defaults.update(kwargs)
    return CompletionResult(
        content=content,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        **defaults,
    )


class TestValidResponse:
    """Normal responses pass through unchanged."""

    def test_valid_response_passes(self) -> None:
        v = ResponseValidator()
        result = _make_result(content="Perfectly fine response.")
        validated = v.validate(result)
        assert validated.content == "Perfectly fine response."
        assert validated is result  # same object, no modification

    def test_valid_with_tool_calls(self) -> None:
        v = ResponseValidator()
        result = _make_result(
            content="Using a tool.",
            tool_calls=[{"id": "c1", "type": "function"}],
        )
        validated = v.validate(result)
        assert validated.tool_calls is not None


class TestEmptyContent:
    """Empty content with no tool calls is invalid."""

    def test_empty_content_raises(self) -> None:
        v = ResponseValidator()
        result = _make_result(content="")
        with pytest.raises(ValueError, match="Empty response"):
            v.validate(result)

    def test_empty_content_with_tool_calls_ok(self) -> None:
        """Empty content is OK when tool calls are present."""
        v = ResponseValidator()
        result = _make_result(
            content="",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "x"}}],
        )
        validated = v.validate(result)
        assert validated.content == ""
        assert validated.tool_calls is not None


class TestFinishReason:
    """finish_reason analysis."""

    def test_length_emits_warning(self) -> None:
        v = ResponseValidator()
        result = _make_result(content="Partial output...", finish_reason="length")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validated = v.validate(result)
            assert len(w) == 1
            assert "truncated" in str(w[0].message)
        assert validated.content == "Partial output..."

    def test_content_filter_emits_warning(self) -> None:
        v = ResponseValidator()
        result = _make_result(content="Filtered.", finish_reason="content_filter")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v.validate(result)
            assert any("content filter" in str(x.message) for x in w)

    def test_stop_no_warning(self) -> None:
        v = ResponseValidator()
        result = _make_result(content="Fine.", finish_reason="stop")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v.validate(result)
            assert len(w) == 0


class TestSecretRedaction:
    """API key / secret leak detection in response content."""

    def test_openai_key_redacted(self) -> None:
        v = ResponseValidator()
        result = _make_result(
            content="Here is your key: sk-1234567890abcdefghijklmnopqr",
        )
        validated = v.validate(result)
        assert "sk-1234567890abcdefghijklmnopqr" not in validated.content
        assert "[REDACTED]" in validated.content

    def test_anthropic_key_redacted(self) -> None:
        v = ResponseValidator()
        result = _make_result(
            content="The Anthropic key is sk-ant-abc123456789012345678901",
        )
        validated = v.validate(result)
        assert "sk-ant-" not in validated.content
        assert "[REDACTED]" in validated.content

    def test_google_key_redacted(self) -> None:
        v = ResponseValidator()
        result = _make_result(
            content="Google key: AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        validated = v.validate(result)
        assert "AIzaSy" not in validated.content
        assert "[REDACTED]" in validated.content

    def test_aws_key_redacted(self) -> None:
        v = ResponseValidator()
        result = _make_result(
            content="AWS access: AKIAIOSFODNN7EXAMPLE",
        )
        validated = v.validate(result)
        assert "AKIAIOSFODNN7EXAMPLE" not in validated.content

    def test_no_secrets_untouched(self) -> None:
        v = ResponseValidator()
        result = _make_result(content="No secrets here, just normal text.")
        validated = v.validate(result)
        assert validated is result  # exact same object
