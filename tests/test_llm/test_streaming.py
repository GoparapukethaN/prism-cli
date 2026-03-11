"""Tests for StreamingHandler — 8+ tests, fully offline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prism.llm.mock import MockLiteLLM, MockResponse, MockStreamChunk
from prism.llm.streaming import StreamingHandler

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


async def _async_iter(items: list[dict]) -> AsyncIterator[dict]:
    """Helper to create an async iterator from a list of dicts."""
    for item in items:
        yield item


class TestStreamAccumulation:
    """Content is correctly assembled from chunks."""

    async def test_basic_stream(self) -> None:
        handler = StreamingHandler()
        chunks = [
            {
                "choices": [{"delta": {"content": "Hello "}, "finish_reason": None}],
            },
            {
                "choices": [{"delta": {"content": "world!"}, "finish_reason": None}],
            },
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            },
        ]
        result = await handler.handle_stream(
            stream=_async_iter(chunks),
            model="gpt-4o",
            input_text="Hi",
        )
        assert result.content == "Hello world!"
        assert result.finish_reason == "stop"
        assert result.model == "gpt-4o"
        assert result.provider == "openai"

    async def test_empty_stream(self) -> None:
        handler = StreamingHandler()
        result = await handler.handle_stream(
            stream=_async_iter([]),
            model="gpt-4o",
        )
        assert result.content == ""
        assert result.finish_reason == "stop"  # default

    async def test_single_chunk(self) -> None:
        handler = StreamingHandler()
        chunks = [
            {
                "choices": [{"delta": {"content": "one"}, "finish_reason": "stop"}],
            },
        ]
        result = await handler.handle_stream(
            stream=_async_iter(chunks), model="gpt-4o",
        )
        assert result.content == "one"
        assert result.finish_reason == "stop"


class TestOnTokenCallback:
    """The on_token callback is invoked for each content delta."""

    async def test_callback_called_per_chunk(self) -> None:
        handler = StreamingHandler()
        tokens: list[str] = []

        chunks = [
            {"choices": [{"delta": {"content": "A "}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "B "}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "C"}, "finish_reason": "stop"}]},
        ]
        result = await handler.handle_stream(
            stream=_async_iter(chunks),
            model="gpt-4o",
            on_token=tokens.append,
        )
        assert tokens == ["A ", "B ", "C"]
        assert result.content == "A B C"

    async def test_no_callback_still_works(self) -> None:
        handler = StreamingHandler()
        chunks = [
            {"choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}]},
        ]
        result = await handler.handle_stream(
            stream=_async_iter(chunks), model="gpt-4o",
        )
        assert result.content == "hello"


class TestStreamWithToolCalls:
    """Tool-call deltas in stream."""

    async def test_tool_calls_accumulated(self) -> None:
        handler = StreamingHandler()
        chunks = [
            {
                "choices": [{
                    "delta": {
                        "content": "Calling tool...",
                        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search"}}],
                    },
                    "finish_reason": None,
                }],
            },
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            },
        ]
        result = await handler.handle_stream(
            stream=_async_iter(chunks), model="gpt-4o",
        )
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1


class TestStreamTokenCounting:
    """Token counting at end of stream."""

    async def test_usage_from_final_chunk(self) -> None:
        handler = StreamingHandler()
        chunks = [
            {"choices": [{"delta": {"content": "result"}, "finish_reason": None}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 42,
                    "completion_tokens": 15,
                    "cached_tokens": 5,
                },
            },
        ]
        result = await handler.handle_stream(
            stream=_async_iter(chunks), model="gpt-4o", input_text="test",
        )
        assert result.input_tokens == 42
        assert result.output_tokens == 15
        assert result.cached_tokens == 5

    async def test_estimated_tokens_when_no_usage(self) -> None:
        handler = StreamingHandler()
        chunks = [
            {"choices": [{"delta": {"content": "some response text"}, "finish_reason": "stop"}]},
        ]
        result = await handler.handle_stream(
            stream=_async_iter(chunks),
            model="gpt-4o",
            input_text="count my tokens",
        )
        # Without usage data, tokens are estimated.
        assert result.input_tokens > 0
        assert result.output_tokens > 0


class TestMockLiteLLMStreaming:
    """Use MockLiteLLM streaming directly."""

    async def test_mock_streaming_default_chunks(self) -> None:
        mock = MockLiteLLM()
        mock.set_response("gpt-4o", MockResponse(content="Two words"))
        handler = StreamingHandler()

        result = await handler.handle_stream(
            stream=mock.acompletion_streaming(
                model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
            ),
            model="gpt-4o",
        )
        assert "Two" in result.content
        assert "words" in result.content

    async def test_mock_streaming_custom_chunks(self) -> None:
        mock = MockLiteLLM()
        mock.set_stream_chunks("gpt-4o", [
            MockStreamChunk(content="chunk1 "),
            MockStreamChunk(content="chunk2"),
            MockStreamChunk(finish_reason="stop"),
        ])
        handler = StreamingHandler()
        result = await handler.handle_stream(
            stream=mock.acompletion_streaming(
                model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
            ),
            model="gpt-4o",
        )
        assert result.content == "chunk1 chunk2"
