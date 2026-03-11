"""Configuration schema definitions using Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class RoutingConfig(BaseModel):
    """Routing engine configuration."""

    simple_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Score below which tasks are classified as SIMPLE",
    )
    medium_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Score below which tasks are classified as MEDIUM (above = COMPLEX)",
    )
    exploration_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of requests routed to non-default models for exploration",
    )
    architect_mode: bool = Field(
        default=True,
        description="Enable architect mode: premium plans, cheap executes",
    )
    quality_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for quality vs cost in model ranking (higher = prefer quality)",
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> RoutingConfig:
        if self.simple_threshold >= self.medium_threshold:
            msg = (
                f"simple_threshold ({self.simple_threshold}) must be less than "
                f"medium_threshold ({self.medium_threshold})"
            )
            raise ValueError(msg)
        return self


class BudgetConfig(BaseModel):
    """Budget enforcement configuration."""

    daily_limit: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum daily spend in USD (None = unlimited)",
    )
    monthly_limit: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum monthly spend in USD (None = unlimited)",
    )
    warn_at_percent: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Warn user when this percentage of budget is consumed",
    )


class ToolsConfig(BaseModel):
    """Tool execution configuration."""

    web_enabled: bool = Field(
        default=False,
        description="Enable web browsing and screenshot tools",
    )
    auto_approve: bool = Field(
        default=False,
        description="Auto-approve file writes and command execution (--yes mode)",
    )
    command_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Default command execution timeout in seconds",
    )
    max_output_bytes: int = Field(
        default=102400,
        ge=1024,
        description="Maximum stdout capture size in bytes (default 100KB)",
    )
    max_error_bytes: int = Field(
        default=10240,
        ge=1024,
        description="Maximum stderr capture size in bytes (default 10KB)",
    )
    allowed_commands: list[str] = Field(
        default_factory=list,
        description="Commands that skip confirmation prompts",
    )


class ProviderOverride(BaseModel):
    """Per-provider configuration overrides."""

    enabled: bool = Field(default=True, description="Whether this provider is active")
    api_base: str | None = Field(
        default=None,
        description="Custom API endpoint URL",
    )
    api_key_env: str | None = Field(
        default=None,
        description="Custom environment variable name for API key",
    )
    preferred_models: list[str] = Field(
        default_factory=list,
        description="Preferred models from this provider (ordered)",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra provider-specific configuration",
    )


class ExcludedPatternsConfig(BaseModel):
    """File patterns excluded from tool operations."""

    patterns: list[str] = Field(
        default_factory=lambda: [
            ".env",
            ".env.*",
            "*.env",
            ".git/credentials",
            "**/node_modules/**",
            "**/.ssh/**",
            "**/*.pem",
            "**/*.key",
            "**/*.p12",
            "**/*.pfx",
            "**/credentials.json",
            "**/service-account*.json",
            "**/.aws/credentials",
            "**/.azure/credentials",
            "**/secrets.yaml",
            "**/secrets.yml",
        ],
        description="Glob patterns for files excluded from tool operations",
    )


class CustomProviderConfig(BaseModel):
    """Configuration for a custom OpenAI-compatible provider."""

    api_base: str = Field(description="API endpoint URL")
    api_key_env: str = Field(description="Environment variable name for API key")
    models: list[CustomModelConfig] = Field(
        default_factory=list,
        description="Available models",
    )


class CustomModelConfig(BaseModel):
    """Configuration for a custom model."""

    id: str = Field(description="Model identifier")
    display_name: str = Field(default="", description="Human-readable model name")
    tier: str = Field(
        default="medium",
        description="Complexity tier: simple, medium, or complex",
    )
    input_cost_per_1m: float = Field(
        default=0.0,
        ge=0.0,
        description="Input cost per 1M tokens in USD",
    )
    output_cost_per_1m: float = Field(
        default=0.0,
        ge=0.0,
        description="Output cost per 1M tokens in USD",
    )
    context_window: int = Field(
        default=32768,
        gt=0,
        description="Maximum context window in tokens",
    )
    supports_tools: bool = Field(default=True, description="Supports function calling")
    supports_vision: bool = Field(default=False, description="Supports image inputs")


class PrismConfig(BaseModel):
    """Complete Prism configuration file schema (~/.prism/config.yaml)."""

    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    providers: dict[str, ProviderOverride] = Field(
        default_factory=dict,
        description="Per-provider configuration overrides",
    )
    custom_providers: dict[str, CustomProviderConfig] = Field(
        default_factory=dict,
        description="Custom OpenAI-compatible providers",
    )
    excluded_patterns: ExcludedPatternsConfig = Field(
        default_factory=ExcludedPatternsConfig,
    )
    preferred_provider: str | None = Field(
        default=None,
        description="Always try this provider first when available",
    )
    pinned_model: str | None = Field(
        default=None,
        description="Force all requests to this model",
    )
    excluded_providers: list[str] = Field(
        default_factory=list,
        description="Providers that should never receive requests",
    )
    prism_home: Path = Field(
        default_factory=lambda: Path.home() / ".prism",
        description="Prism data directory",
    )
