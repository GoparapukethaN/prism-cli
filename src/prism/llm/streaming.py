"""Streaming response handler for LiteLLM async streams."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from prism.cost.pricing import calculate_cost, estimate_input_tokens, get_provider_for_model
from prism.llm.result import CompletionResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

logger = structlog.get_logger(__name__)


class StreamingHandler:
    """Processes streaming chunks from LiteLLM, accumulating the full result.

    Usage::

        handler = StreamingHandler()
        result = await handler.handle_stream(
            stream=litellm_stream,
            model="gpt-4o",
            input_text="Hello",
            on_token=print,
        )
    """

    async def handle_stream(
        self,
        stream: AsyncIterator[Any],
        model: str,
        input_text: str = "",
        on_token: Callable[[str], None] | None = None,
    ) -> CompletionResult:
        """Consume an async stream of chunks and assemble a :class:`CompletionResult`.

        Args:
            stream: Async iterator yielding LiteLLM-style ``ModelResponse`` chunks.
            model: The model identifier (needed for pricing).
            input_text: Original input text (for token estimation).
            on_token: Optional callback invoked with each content delta.

        Returns:
            A fully assembled :class:`CompletionResult`.
        """
        start = time.perf_counter()
        content_parts: list[str] = []
        tool_calls: list[dict] = []
        finish_reason = "stop"
        _raw_usage: dict[str, int] = {}

        async for chunk in stream:
            # Handle both dict-like and attribute-style chunks.
            choices = _get(chunk, "choices", [])
            if not choices:
                continue
            delta = _get(choices[0], "delta", {})
            chunk_finish = _get(choices[0], "finish_reason", None)

            if chunk_finish is not None:
                finish_reason = chunk_finish

            # Content delta
            text = _get(delta, "content", None)
            if text:
                content_parts.append(text)
                if on_token is not None:
                    on_token(text)

            # Tool-call deltas (accumulated)
            tc = _get(delta, "tool_calls", None)
            if tc:
                for call in tc:
                    tool_calls.append(_to_dict(call))

            # Some providers include usage on the final chunk.
            usage = _get(chunk, "usage", None)
            if usage is not None:
                _raw_usage = _to_dict(usage)

        elapsed_ms = (time.perf_counter() - start) * 1000
        full_content = "".join(content_parts)

        # Token counting: prefer reported usage, fall back to estimates.
        input_tokens = int(_raw_usage.get("prompt_tokens", 0))
        output_tokens = int(
            _raw_usage.get("completion_tokens", 0)
        )
        cached_tokens = int(_raw_usage.get("cached_tokens", 0))

        if input_tokens == 0 and input_text:
            input_tokens = estimate_input_tokens(input_text)
        if output_tokens == 0 and full_content:
            # Rough estimate: ~0.75 tokens per word
            output_tokens = max(1, int(len(full_content.split()) * 0.75))

        try:
            cost = calculate_cost(model, input_tokens, output_tokens, cached_tokens)
        except ValueError:
            cost = 0.0

        provider = get_provider_for_model(model)

        return CompletionResult(
            content=full_content,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            latency_ms=elapsed_ms,
            finish_reason=finish_reason,
            tool_calls=tool_calls if tool_calls else None,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Retrieve *key* from *obj* whether it is a dict or has attributes."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_dict(obj: Any) -> dict:
    """Coerce *obj* to a plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": obj}
