"""Completion result dataclass — the standard return type for all LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompletionResult:
    """Immutable result of a completion request.

    Every LLM call (streaming or non-streaming) ultimately returns one of
    these.  The ``raw_response`` is only populated when debugging is enabled.
    """

    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float
    latency_ms: float
    finish_reason: str
    tool_calls: list[dict] | None = field(default=None)
    raw_response: dict | None = field(default=None)
