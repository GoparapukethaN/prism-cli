"""Provider registry — manages all configured providers and their models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from prism.providers.base import (
    BUILTIN_PROVIDERS,
    ComplexityTier,
    ModelInfo,
    ProviderConfig,
)

if TYPE_CHECKING:
    from prism.auth.manager import AuthManager
    from prism.config.settings import Settings

logger = structlog.get_logger(__name__)


class ProviderStatus:
    """Tracks runtime status of a provider."""

    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name
        self.is_available: bool = True
        self.last_check: datetime | None = None
        self.last_error: str | None = None
        self.rate_limited_until: datetime | None = None
        self.consecutive_failures: int = 0
        self.free_tier_requests_today: int = 0
        self.free_tier_reset_date: str = ""

    @property
    def is_rate_limited(self) -> bool:
        """Check if provider is currently rate-limited."""
        if self.rate_limited_until is None:
            return False
        return datetime.now(UTC) < self.rate_limited_until

    def mark_rate_limited(self, until: datetime) -> None:
        """Mark this provider as rate-limited until a specific time."""
        self.rate_limited_until = until
        self.consecutive_failures += 1
        logger.warning(
            "provider_rate_limited",
            provider=self.provider_name,
            until=until.isoformat(),
            consecutive=self.consecutive_failures,
        )

    def mark_available(self) -> None:
        """Mark this provider as available (recovered from failure)."""
        if not self.is_available or self.consecutive_failures > 0:
            logger.info(
                "provider_recovered",
                provider=self.provider_name,
                previous_failures=self.consecutive_failures,
            )
        self.is_available = True
        self.rate_limited_until = None
        self.consecutive_failures = 0
        self.last_error = None
        self.last_check = datetime.now(UTC)

    def mark_unavailable(self, error: str) -> None:
        """Mark this provider as unavailable."""
        self.is_available = False
        self.last_error = error
        self.consecutive_failures += 1
        self.last_check = datetime.now(UTC)
        logger.warning(
            "provider_unavailable",
            provider=self.provider_name,
            error=error,
            consecutive=self.consecutive_failures,
        )

    def increment_free_tier_usage(self) -> None:
        """Increment the free tier request counter for today."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if self.free_tier_reset_date != today:
            self.free_tier_requests_today = 0
            self.free_tier_reset_date = today
        self.free_tier_requests_today += 1


class ProviderRegistry:
    """Central registry of all configured AI providers and their models.

    Manages provider availability, model metadata, and runtime status.
    Thread-safe for read operations after initialization.
    """

    def __init__(self, settings: Settings, auth_manager: AuthManager) -> None:
        """Initialize the provider registry.

        Args:
            settings: Application settings.
            auth_manager: AuthManager for checking API key availability.
        """
        self._settings = settings
        self._auth = auth_manager
        self._providers: dict[str, ProviderConfig] = {}
        self._models: dict[str, ModelInfo] = {}
        self._status: dict[str, ProviderStatus] = {}
        self._initialize()

    def _initialize(self) -> None:
        """Load all built-in and custom providers."""
        excluded = set(self._settings.config.excluded_providers)

        for provider_config in BUILTIN_PROVIDERS:
            if provider_config.name in excluded:
                logger.info("provider_excluded", provider=provider_config.name)
                continue

            # Check provider override in settings
            override = self._settings.config.providers.get(provider_config.name)
            if override and not override.enabled:
                logger.info("provider_disabled", provider=provider_config.name)
                continue

            self._register_provider(provider_config)

        # Register custom providers from settings
        for name, custom_config in self._settings.config.custom_providers.items():
            if name in excluded:
                continue
            self._register_custom_provider(name, custom_config)

        logger.info(
            "registry_initialized",
            providers=list(self._providers.keys()),
            total_models=len(self._models),
        )

    def _register_provider(self, config: ProviderConfig) -> None:
        """Register a built-in provider and its models."""
        self._providers[config.name] = config
        self._status[config.name] = ProviderStatus(config.name)

        for model in config.models:
            self._models[model.id] = model

    def _register_custom_provider(self, name: str, config: object) -> None:
        """Register a custom provider from settings."""
        from prism.config.schema import CustomProviderConfig

        if not isinstance(config, CustomProviderConfig):
            logger.warning("invalid_custom_provider", name=name)
            return

        models: list[ModelInfo] = []
        for model_config in config.models:
            tier = ComplexityTier(model_config.tier) if model_config.tier else ComplexityTier.MEDIUM
            model = ModelInfo(
                id=model_config.id,
                display_name=model_config.display_name or model_config.id,
                provider=name,
                tier=tier,
                input_cost_per_1m=model_config.input_cost_per_1m,
                output_cost_per_1m=model_config.output_cost_per_1m,
                context_window=model_config.context_window,
                supports_tools=model_config.supports_tools,
                supports_vision=model_config.supports_vision,
            )
            models.append(model)
            self._models[model.id] = model

        provider_config = ProviderConfig(
            name=name,
            display_name=name.replace("-", " ").replace("_", " ").title(),
            api_key_env=config.api_key_env,
            api_base=config.api_base,
            models=models,
        )
        self._providers[name] = provider_config
        self._status[name] = ProviderStatus(name)

    def get_provider(self, name: str) -> ProviderConfig | None:
        """Get a provider configuration by name."""
        return self._providers.get(name)

    def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get metadata for a specific model.

        Args:
            model_id: LiteLLM model identifier.

        Returns:
            ModelInfo or None if not found.
        """
        return self._models.get(model_id)

    def get_status(self, provider_name: str) -> ProviderStatus | None:
        """Get the runtime status of a provider."""
        return self._status.get(provider_name)

    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is configured, has a key, and is not rate-limited.

        Args:
            provider_name: Provider name.

        Returns:
            True if the provider is available for requests.
        """
        provider = self._providers.get(provider_name)
        if provider is None:
            return False

        status = self._status.get(provider_name)
        if status and (not status.is_available or status.is_rate_limited):
            return False

        # Ollama doesn't need an API key
        if provider_name == "ollama":
            return True

        # Check if API key is configured
        key = self._auth.get_key(provider_name)
        return key is not None

    def get_available_models(self, tier: ComplexityTier | None = None) -> list[ModelInfo]:
        """Get all models from available providers, optionally filtered by tier.

        Args:
            tier: Filter by complexity tier. None returns all models.

        Returns:
            List of ModelInfo for available models, sorted by cost (cheapest first).
        """
        available: list[ModelInfo] = []

        for provider_name, provider_config in self._providers.items():
            if not self.is_provider_available(provider_name):
                continue

            for model in provider_config.models:
                if tier is not None and model.tier != tier:
                    continue
                available.append(model)

        # Sort by cost (cheapest first within each tier)
        available.sort(key=lambda m: m.input_cost_per_1m + m.output_cost_per_1m)
        return available

    def get_models_for_tier(self, tier: ComplexityTier) -> list[ModelInfo]:
        """Get candidate models for a specific complexity tier.

        Includes models from the target tier AND models from adjacent tiers
        that could serve as fallbacks.

        Args:
            tier: The target complexity tier.

        Returns:
            List of candidate models, ordered by preference.
        """
        # Primary: exact tier match
        primary = self.get_available_models(tier)

        # For SIMPLE tier, don't add higher-tier fallbacks by default
        if tier == ComplexityTier.SIMPLE:
            return primary

        # For MEDIUM, add SIMPLE models as fallbacks
        if tier == ComplexityTier.MEDIUM:
            fallbacks = self.get_available_models(ComplexityTier.SIMPLE)
            return primary + [m for m in fallbacks if m not in primary]

        # For COMPLEX, add MEDIUM models as fallbacks
        if tier == ComplexityTier.COMPLEX:
            fallbacks = self.get_available_models(ComplexityTier.MEDIUM)
            return primary + [m for m in fallbacks if m not in primary]

        return primary

    def get_free_tier_remaining(self, provider_name: str) -> int | None:
        """Get remaining free tier requests for today.

        Args:
            provider_name: Provider name.

        Returns:
            Remaining requests, or None if provider has no free tier.
        """
        provider = self._providers.get(provider_name)
        if provider is None or provider.free_tier is None:
            return None

        status = self._status.get(provider_name)
        if status is None:
            return provider.free_tier.requests_per_day

        # Reset counter if it's a new day
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if status.free_tier_reset_date != today:
            status.free_tier_requests_today = 0
            status.free_tier_reset_date = today

        return max(0, provider.free_tier.requests_per_day - status.free_tier_requests_today)

    def list_providers(self) -> list[dict[str, object]]:
        """List all registered providers with their status.

        Returns:
            List of provider info dicts.
        """
        result: list[dict[str, object]] = []
        for name, config in self._providers.items():
            status = self._status.get(name)
            has_key = name == "ollama" or self._auth.get_key(name) is not None

            info: dict[str, object] = {
                "name": name,
                "display_name": config.display_name,
                "configured": has_key,
                "available": self.is_provider_available(name),
                "model_count": len(config.models),
                "models": [m.display_name for m in config.models],
                "free_tier": config.free_tier is not None,
            }

            if status:
                info["rate_limited"] = status.is_rate_limited
                info["last_error"] = status.last_error

            free_remaining = self.get_free_tier_remaining(name)
            if free_remaining is not None:
                info["free_tier_remaining"] = free_remaining

            result.append(info)
        return result

    @property
    def all_models(self) -> dict[str, ModelInfo]:
        """Get all registered models."""
        return dict(self._models)

    @property
    def all_providers(self) -> dict[str, ProviderConfig]:
        """Get all registered provider configs."""
        return dict(self._providers)
