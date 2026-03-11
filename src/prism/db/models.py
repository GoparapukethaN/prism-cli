"""Pydantic data models for the Prism database layer.

These models represent rows stored in or retrieved from SQLite tables.
IDs are UUID strings. Timestamps are ISO 8601 strings.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ComplexityTier(StrEnum):
    """Task complexity classification tier."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class Outcome(StrEnum):
    """Outcome of a routing decision after the user sees the response."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CORRECTED = "corrected"
    UNKNOWN = "unknown"


class RoutingDecision(BaseModel):
    """A single routing decision record (routing_decisions table)."""

    id: str = Field(description="UUID primary key")
    created_at: str = Field(description="ISO 8601 timestamp")
    session_id: str = Field(description="Session identifier")
    prompt_hash: str = Field(description="SHA-256 hash of the prompt (never raw prompt)")
    complexity_tier: ComplexityTier = Field(description="Classified tier")
    complexity_score: float = Field(ge=0.0, le=1.0, description="Complexity score 0..1")
    model_selected: str = Field(description="LiteLLM model ID chosen by router")
    model_actual: str | None = Field(default=None, description="Actual model used (if fallback)")
    fallback_chain: str = Field(description="JSON array of model IDs in fallback order")
    estimated_cost: float = Field(ge=0.0, description="Estimated cost in USD")
    actual_cost: float | None = Field(default=None, description="Actual cost after completion")
    input_tokens: int | None = Field(default=None, ge=0, description="Actual input tokens")
    output_tokens: int | None = Field(default=None, ge=0, description="Actual output tokens")
    cached_tokens: int = Field(default=0, ge=0, description="Tokens served from cache")
    latency_ms: float | None = Field(default=None, ge=0.0, description="Total latency in ms")
    outcome: Outcome = Field(default=Outcome.UNKNOWN, description="User outcome")
    features: str = Field(description="JSON object of extracted features")
    error: str | None = Field(default=None, description="Error message if failed")


class CostEntry(BaseModel):
    """A single cost tracking record (cost_entries table)."""

    id: str = Field(description="UUID primary key")
    created_at: str = Field(description="ISO 8601 timestamp")
    session_id: str = Field(description="Session identifier")
    model_id: str = Field(description="LiteLLM model ID")
    provider: str = Field(description="Provider name")
    input_tokens: int = Field(ge=0, description="Input token count")
    output_tokens: int = Field(ge=0, description="Output token count")
    cached_tokens: int = Field(default=0, ge=0, description="Cached token count")
    cost_usd: float = Field(ge=0.0, description="Calculated cost in USD")
    complexity_tier: ComplexityTier = Field(description="Complexity tier for this call")


class Session(BaseModel):
    """Session metadata record (sessions table)."""

    id: str = Field(description="UUID primary key")
    created_at: str = Field(description="ISO 8601 timestamp")
    updated_at: str = Field(description="ISO 8601 timestamp of last update")
    project_root: str = Field(description="Project directory path")
    total_cost: float = Field(default=0.0, ge=0.0, description="Cumulative session cost")
    total_requests: int = Field(default=0, ge=0, description="Total requests in session")
    summary: str | None = Field(default=None, description="Compressed conversation summary")
    active: bool = Field(default=True, description="Whether the session is still active")


class ProviderStatus(BaseModel):
    """Provider health and rate-limit status (provider_status table)."""

    provider: str = Field(description="Provider name (primary key)")
    last_check_at: str | None = Field(default=None, description="ISO 8601 last health check")
    is_available: bool = Field(default=True, description="Currently reachable")
    last_error: str | None = Field(default=None, description="Last error message")
    rate_limited_until: str | None = Field(
        default=None,
        description="ISO 8601 timestamp until rate limit expires",
    )
    consecutive_failures: int = Field(default=0, ge=0, description="Consecutive failure count")
    free_tier_requests_today: int = Field(
        default=0,
        ge=0,
        description="Free-tier requests used today",
    )
    free_tier_reset_at: str | None = Field(
        default=None,
        description="ISO 8601 timestamp when daily counter resets",
    )


class ToolExecution(BaseModel):
    """Audit record for a tool execution (tool_executions table)."""

    id: str = Field(description="UUID primary key")
    created_at: str = Field(description="ISO 8601 timestamp")
    session_id: str = Field(description="Session identifier")
    tool_name: str = Field(description="Name of the tool executed")
    arguments: str = Field(description="JSON of sanitized arguments (no secrets)")
    result_success: bool = Field(description="Whether the tool call succeeded")
    result_error: str | None = Field(default=None, description="Error message on failure")
    duration_ms: float | None = Field(default=None, ge=0.0, description="Execution time in ms")
    metadata: str | None = Field(default=None, description="JSON of extra per-tool metadata")
