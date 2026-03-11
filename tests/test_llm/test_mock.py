"""Tests for MockLiteLLM — 10+ tests, verifying the mock layer itself."""

from __future__ import annotations

import pytest

from prism.exceptions import ProviderUnavailableError
from prism.llm.mock import MockLiteLLM, MockResponse, MockStreamChunk


class TestDefaultResponse:
    """Default response when no model-specific response is set."""

    async def test_default_content(self) -> None:
        mock = MockLiteLLM()
        result = await mock.acompletion(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
        )
        content = result["choices"][0]["message"]["content"]
        assert content == "Mock response from the AI model."

    def test_sync_default_content(self) -> None:
        mock = MockLiteLLM()
        result = mock.completion(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
        )
        content = result["choices"][0]["message"]["content"]
        assert content == "Mock response from the AI model."

    async def test_custom_default_response(self) -> None:
        mock = MockLiteLLM()
        mock.set_default_response(MockResponse(content="Custom default"))
        result = await mock.acompletion(
            model="anything", messages=[{"role": "user", "content": "hi"}],
        )
        assert result["choices"][0]["message"]["content"] == "Custom default"


class TestPreprogrammedResponses:
    """Model-specific pre-programmed responses."""

    async def test_set_response(self) -> None:
        mock = MockLiteLLM()
        mock.set_response("gpt-4o", MockResponse(content="GPT response"))
        mock.set_response("claude-sonnet-4-20250514", MockResponse(content="Claude response"))

        gpt = await mock.acompletion(
            model="gpt-4o", messages=[{"role": "user", "content": "test"}],
        )
        claude = await mock.acompletion(
            model="claude-sonnet-4-20250514", messages=[{"role": "user", "content": "test"}],
        )
        assert gpt["choices"][0]["message"]["content"] == "GPT response"
        assert claude["choices"][0]["message"]["content"] == "Claude response"

    async def test_response_includes_usage(self) -> None:
        mock = MockLiteLLM()
        mock.set_response(
            "gpt-4o",
            MockResponse(input_tokens=200, output_tokens=100),
        )
        result = await mock.acompletion(
            model="gpt-4o", messages=[{"role": "user", "content": "test"}],
        )
        assert result["usage"]["prompt_tokens"] == 200
        assert result["usage"]["completion_tokens"] == 100

    async def test_response_with_tool_calls(self) -> None:
        mock = MockLiteLLM()
        tools = [{"id": "c1", "type": "function", "function": {"name": "search"}}]
        mock.set_response("gpt-4o", MockResponse(content="Found it", tool_calls=tools))
        result = await mock.acompletion(
            model="gpt-4o", messages=[{"role": "user", "content": "search"}],
        )
        msg = result["choices"][0]["message"]
        assert msg["tool_calls"] == tools


class TestPreprogrammedErrors:
    """Model-specific errors."""

    async def test_set_error_async(self) -> None:
        mock = MockLiteLLM()
        mock.set_error("gpt-4o", ProviderUnavailableError("openai", "down"))
        with pytest.raises(ProviderUnavailableError, match="down"):
            await mock.acompletion(
                model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
            )

    def test_set_error_sync(self) -> None:
        mock = MockLiteLLM()
        mock.set_error("gpt-4o", ValueError("bad"))
        with pytest.raises(ValueError, match="bad"):
            mock.completion(
                model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
            )

    async def test_error_only_for_specific_model(self) -> None:
        mock = MockLiteLLM()
        mock.set_error("gpt-4o", RuntimeError("fail"))
        # Other models should still work.
        result = await mock.acompletion(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}],
        )
        assert result["choices"][0]["message"]["content"]


class TestCallLog:
    """Call recording."""

    async def test_call_log_recorded(self) -> None:
        mock = MockLiteLLM()
        await mock.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.5,
        )
        assert len(mock.call_log) == 1
        assert mock.call_log[0]["model"] == "gpt-4o"
        assert mock.call_log[0]["messages"][0]["content"] == "hello"
        assert mock.call_log[0]["kwargs"]["temperature"] == 0.5

    async def test_call_log_does_not_record_api_key(self) -> None:
        mock = MockLiteLLM()
        await mock.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-secret-12345",
        )
        # api_key must NOT appear in call log.
        assert "api_key" not in mock.call_log[0]["kwargs"]

    async def test_multiple_calls_logged(self) -> None:
        mock = MockLiteLLM()
        for i in range(5):
            await mock.acompletion(
                model=f"model-{i}", messages=[{"role": "user", "content": str(i)}],
            )
        assert len(mock.call_log) == 5

    def test_reset_clears_everything(self) -> None:
        mock = MockLiteLLM()
        mock.set_response("gpt-4o", MockResponse(content="hi"))
        mock.set_error("claude", ValueError("err"))
        mock.completion(model="gpt-4o", messages=[])
        assert len(mock.call_log) == 1

        mock.reset()
        assert len(mock.call_log) == 0
        assert len(mock.responses) == 0
        assert len(mock.errors) == 0


class TestStreamingMock:
    """Streaming mock produces chunks."""

    async def test_default_streaming(self) -> None:
        mock = MockLiteLLM()
        mock.set_response("gpt-4o", MockResponse(content="Hello world"))
        chunks: list[dict] = []
        async for chunk in mock.acompletion_streaming(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
        ):
            chunks.append(chunk)
        # Default splits into 3 chunks: first half, second half, finish
        assert len(chunks) == 3
        # Reconstruct content.
        content = ""
        for c in chunks:
            delta = c["choices"][0]["delta"]
            if "content" in delta:
                content += delta["content"]
        assert "Hello" in content
        assert "world" in content

    async def test_custom_stream_chunks(self) -> None:
        mock = MockLiteLLM()
        mock.set_stream_chunks("gpt-4o", [
            MockStreamChunk(content="A"),
            MockStreamChunk(content="B"),
            MockStreamChunk(finish_reason="stop"),
        ])
        chunks: list[dict] = []
        async for chunk in mock.acompletion_streaming(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
        ):
            chunks.append(chunk)
        assert len(chunks) == 3

    async def test_streaming_error(self) -> None:
        mock = MockLiteLLM()
        mock.set_error("gpt-4o", ConnectionError("lost"))
        with pytest.raises(ConnectionError, match="lost"):
            async for _ in mock.acompletion_streaming(
                model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
            ):
                pass
