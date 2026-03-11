"""Full mock layer for LiteLLM — used in ALL tests.

No real API calls are ever made.  This module provides drop-in replacements
for ``litellm.acompletion`` and ``litellm.completion`` that return pre-
programmed responses or raise pre-programmed errors.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass
class MockResponse:
    """A pre-programmed response for the mock LiteLLM layer."""

    content: str = "Mock response from the AI model."
    model: str = "gpt-4o"
    input_tokens: int = 100
    output_tokens: int = 50
    finish_reason: str = "stop"
    tool_calls: list[dict] | None = None


@dataclass
class MockStreamChunk:
    """A single chunk in a mock streaming response."""

    content: str | None = None
    finish_reason: str | None = None
    tool_calls: list[dict] | None = None


class MockLiteLLM:
    """Complete mock for ``litellm.completion`` and ``litellm.acompletion``.

    **CRITICAL**: This is used for ALL testing.  No real API calls ever.

    Usage::

        mock = MockLiteLLM()
        mock.set_response("gpt-4o", MockResponse(content="hello"))
        result = await mock.acompletion(model="gpt-4o", messages=[...])
    """

    def __init__(self) -> None:
        self.responses: dict[str, MockResponse] = {}
        self.errors: dict[str, Exception] = {}
        self.call_log: list[dict[str, Any]] = []
        self._default_response = MockResponse()
        self._stream_chunks: dict[str, list[MockStreamChunk]] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_response(self, model: str, response: MockResponse) -> None:
        """Pre-program a response for a specific model."""
        self.responses[model] = response

    def set_error(self, model: str, error: Exception) -> None:
        """Pre-program an error for a specific model."""
        self.errors[model] = error

    def set_stream_chunks(self, model: str, chunks: list[MockStreamChunk]) -> None:
        """Pre-program streaming chunks for a specific model."""
        self._stream_chunks[model] = chunks

    def set_default_response(self, response: MockResponse) -> None:
        """Override the default response returned for unconfigured models."""
        self._default_response = response

    def reset(self) -> None:
        """Clear all pre-programmed responses, errors, and call log."""
        self.responses.clear()
        self.errors.clear()
        self.call_log.clear()
        self._stream_chunks.clear()
        self._default_response = MockResponse()

    # ------------------------------------------------------------------
    # Mock API surface
    # ------------------------------------------------------------------

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock async completion — mirrors ``litellm.acompletion``."""
        self._record_call(model, messages, kwargs)
        self._maybe_raise(model)
        response = self._resolve_response(model)
        return self._build_response_dict(response, model)

    def completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock sync completion — mirrors ``litellm.completion``."""
        self._record_call(model, messages, kwargs)
        self._maybe_raise(model)
        response = self._resolve_response(model)
        return self._build_response_dict(response, model)

    async def acompletion_streaming(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Mock streaming async completion yielding chunks.

        If :meth:`set_stream_chunks` has been called for *model*, those
        chunks are yielded.  Otherwise a default two-chunk stream is produced
        from the resolved :class:`MockResponse`.
        """
        self._record_call(model, messages, {**kwargs, "_streaming": True})
        self._maybe_raise(model)

        chunks = self._stream_chunks.get(model)
        if chunks is None:
            response = self._resolve_response(model)
            chunks = self._default_stream_chunks(response)

        for chunk in chunks:
            yield self._build_stream_chunk_dict(chunk, model)
            await asyncio.sleep(0)  # yield control

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_call(
        self,
        model: str,
        messages: list[dict[str, str]],
        kwargs: dict[str, Any],
    ) -> None:
        self.call_log.append({
            "model": model,
            "messages": copy.deepcopy(messages),
            "kwargs": {k: v for k, v in kwargs.items() if k != "api_key"},
        })

    def _maybe_raise(self, model: str) -> None:
        if model in self.errors:
            raise self.errors[model]

    def _resolve_response(self, model: str) -> MockResponse:
        return self.responses.get(model, self._default_response)

    @staticmethod
    def _build_response_dict(resp: MockResponse, model: str) -> dict[str, Any]:
        """Build a dict that mirrors the LiteLLM ``ModelResponse`` shape."""
        message: dict[str, Any] = {"role": "assistant", "content": resp.content}
        if resp.tool_calls:
            message["tool_calls"] = resp.tool_calls

        return {
            "id": "mock-completion-id",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": resp.finish_reason,
                },
            ],
            "usage": {
                "prompt_tokens": resp.input_tokens,
                "completion_tokens": resp.output_tokens,
                "total_tokens": resp.input_tokens + resp.output_tokens,
                "cached_tokens": 0,
            },
        }

    @staticmethod
    def _build_stream_chunk_dict(chunk: MockStreamChunk, model: str) -> dict[str, Any]:
        delta: dict[str, Any] = {}
        if chunk.content is not None:
            delta["content"] = chunk.content
        if chunk.tool_calls is not None:
            delta["tool_calls"] = chunk.tool_calls

        return {
            "id": "mock-stream-id",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": chunk.finish_reason,
                },
            ],
        }

    @staticmethod
    def _default_stream_chunks(resp: MockResponse) -> list[MockStreamChunk]:
        """Split a MockResponse into two streaming chunks."""
        words = resp.content.split()
        mid = max(1, len(words) // 2)
        return [
            MockStreamChunk(content=" ".join(words[:mid]) + " "),
            MockStreamChunk(content=" ".join(words[mid:])),
            MockStreamChunk(finish_reason=resp.finish_reason),
        ]
