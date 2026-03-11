"""Unified completion engine wrapping LiteLLM."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from prism.cost.pricing import (
    calculate_cost,
    estimate_input_tokens,
    get_provider_for_model,
)
from prism.cost.tracker import BudgetAction
from prism.exceptions import (
    BudgetExceededError,
    ModelNotFoundError,
)
from prism.llm.result import CompletionResult
from prism.llm.retry import RetryPolicy
from prism.llm.streaming import StreamingHandler
from prism.llm.validation import ResponseValidator

if TYPE_CHECKING:
    from collections.abc import Callable

    from prism.auth.manager import AuthManager
    from prism.config.settings import Settings
    from prism.cost.tracker import CostTracker
    from prism.providers.registry import ProviderRegistry

logger = structlog.get_logger(__name__)

# Fallback context windows for models not in the registry.
_DEFAULT_CONTEXT_WINDOW = 32_768


class CompletionEngine:
    """Unified completion interface wrapping LiteLLM.

    Manages budget enforcement, retries, cost tracking, and response
    validation for every LLM call.
    """

    def __init__(
        self,
        settings: Settings,
        cost_tracker: CostTracker,
        auth_manager: AuthManager,
        provider_registry: ProviderRegistry,
        retry_policy: RetryPolicy | None = None,
        litellm_backend: Any | None = None,
    ) -> None:
        """Initialise the completion engine.

        Args:
            settings: Application settings.
            cost_tracker: For budget checks and cost recording.
            auth_manager: For retrieving API keys.
            provider_registry: For model metadata look-ups.
            retry_policy: Optional custom retry policy.
            litellm_backend: Object with ``acompletion`` / ``completion``
                methods.  If ``None``, the real ``litellm`` package is
                imported lazily.  **Tests MUST inject a mock here.**
        """
        self._settings = settings
        self._cost_tracker = cost_tracker
        self._auth = auth_manager
        self._registry = provider_registry
        self._retry = retry_policy or RetryPolicy()
        self._validator = ResponseValidator()
        self._streaming_handler = StreamingHandler()
        self._litellm = litellm_backend  # None → lazy import on first call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
        tools: list[dict] | None = None,
        session_id: str = "",
        complexity_tier: str = "medium",
    ) -> CompletionResult:
        """Send a completion request.

        Enforces budget, handles retry, tracks cost, and validates the
        response.

        Args:
            messages: Chat messages in OpenAI format.
            model: LiteLLM model identifier.  If ``None``, the caller
                should have already selected a model via the router.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            stream: If ``True``, use streaming under the hood but still
                return a full :class:`CompletionResult`.
            tools: Optional tool/function definitions.
            session_id: Current session id for cost tracking.
            complexity_tier: Tier label for cost tracking.

        Returns:
            :class:`CompletionResult` with content, cost, and metadata.

        Raises:
            BudgetExceededError: Remaining budget too low.
            ModelNotFoundError: Model identifier not recognised.
            ProviderError: Any provider-level failure after retries.
        """
        if model is None:
            msg = "model must be specified (use the router to select one first)"
            raise ModelNotFoundError(msg)

        # 1. Estimate input tokens
        input_text = " ".join(m.get("content", "") for m in messages)
        estimated_input = estimate_input_tokens(input_text)

        # 2. Budget check
        self._check_budget(model, estimated_input)

        # 3. Context-window enforcement
        model_info = self._registry.get_model_info(model)
        context_window = model_info.context_window if model_info else _DEFAULT_CONTEXT_WINDOW
        messages = self._enforce_context_window(model, messages, context_window)

        # 4. Build kwargs and call
        kwargs = self._build_litellm_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            tools=tools,
        )

        start = time.perf_counter()
        raw = await self._retry.execute(self._call_litellm, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 5. Parse response
        result = self._parse_response(raw, model, elapsed_ms)

        # 6. Track cost
        if session_id:
            self._cost_tracker.track(
                model_id=model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                session_id=session_id,
                complexity_tier=complexity_tier,
                cached_tokens=result.cached_tokens,
            )

        # 7. Validate
        result = self._validator.validate(result)
        return result

    async def complete_streaming(
        self,
        messages: list[dict[str, str]],
        model: str,
        on_token: Callable[[str], None] | None = None,
        session_id: str = "",
        complexity_tier: str = "medium",
        **kwargs: Any,
    ) -> CompletionResult:
        """Streaming completion with real-time token callback.

        Args:
            messages: Chat messages.
            model: LiteLLM model identifier.
            on_token: Callback invoked for each content delta.
            session_id: For cost tracking.
            complexity_tier: For cost tracking.
            **kwargs: Forwarded to ``_build_litellm_kwargs``.

        Returns:
            :class:`CompletionResult` assembled from the full stream.
        """
        input_text = " ".join(m.get("content", "") for m in messages)
        self._check_budget(model, estimate_input_tokens(input_text))

        model_info = self._registry.get_model_info(model)
        context_window = model_info.context_window if model_info else _DEFAULT_CONTEXT_WINDOW
        messages = self._enforce_context_window(model, messages, context_window)

        build_kwargs = self._build_litellm_kwargs(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        backend = self._get_backend()
        stream = backend.acompletion_streaming(**build_kwargs)

        result = await self._streaming_handler.handle_stream(
            stream=stream,
            model=model,
            input_text=input_text,
            on_token=on_token,
        )

        if session_id:
            self._cost_tracker.track(
                model_id=model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                session_id=session_id,
                complexity_tier=complexity_tier,
                cached_tokens=result.cached_tokens,
            )

        result = self._validator.validate(result)
        return result

    # ------------------------------------------------------------------
    # LiteLLM call (mocked in tests)
    # ------------------------------------------------------------------

    async def _call_litellm(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the actual LiteLLM async completion call.

        **MUST be mocked in tests** — inject a mock backend via
        ``litellm_backend`` in the constructor.
        """
        backend = self._get_backend()
        return await backend.acompletion(**kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_backend(self) -> Any:
        """Return the LiteLLM backend (real or mock)."""
        if self._litellm is not None:
            return self._litellm
        import litellm  # pragma: no cover — never reached in tests

        self._litellm = litellm  # pragma: no cover
        return litellm  # pragma: no cover

    def _check_budget(self, model: str, estimated_input_tokens: int) -> None:
        """Raise :class:`BudgetExceededError` if budget is insufficient."""
        try:
            estimated_cost = calculate_cost(model, estimated_input_tokens, 500)
        except ValueError:
            estimated_cost = 0.0

        action = self._cost_tracker.check_budget(estimated_cost)
        if action == BudgetAction.BLOCK:
            remaining = self._cost_tracker.get_budget_remaining()
            raise BudgetExceededError(remaining or 0.0, estimated_cost)

    def _enforce_context_window(
        self,
        model: str,
        messages: list[dict[str, str]],
        context_window: int,
    ) -> list[dict[str, str]]:
        """Trim messages to fit within the model's context window.

        Keeps the system message (if present) and the last user message,
        then fills backwards with as many older messages as possible.
        """
        total = sum(
            estimate_input_tokens(m.get("content", "")) for m in messages
        )
        # Leave headroom for output tokens (~10%)
        limit = int(context_window * 0.9)
        if total <= limit:
            return messages

        logger.warning(
            "context_trimmed",
            model=model,
            original_tokens=total,
            limit=limit,
            messages_before=len(messages),
        )

        # Always keep system message (index 0 if role == system) and
        # the last message.
        system_msgs: list[dict[str, str]] = []
        remaining: list[dict[str, str]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                remaining.append(msg)

        # Start with system + last user message, then add backwards.
        kept: list[dict[str, str]] = list(system_msgs)
        if remaining:
            kept.append(remaining[-1])

        budget = limit - sum(
            estimate_input_tokens(m.get("content", "")) for m in kept
        )

        # Add older messages from most recent to oldest.
        for msg in reversed(remaining[:-1]):
            tokens = estimate_input_tokens(msg.get("content", ""))
            if tokens <= budget:
                kept.insert(len(system_msgs), msg)
                budget -= tokens
            else:
                break

        logger.info(
            "context_trimmed_result",
            messages_after=len(kept),
        )
        return kept

    def _build_litellm_kwargs(
        self,
        model: str,
        messages: list[dict[str, str]],
        **overrides: Any,
    ) -> dict[str, Any]:
        """Build the kwargs dict for ``litellm.acompletion``.

        Injects the API key from the auth manager and any provider-
        specific settings.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        # Standard optional params
        for key in ("temperature", "max_tokens", "tools", "stream"):
            if key in overrides and overrides[key] is not None:
                kwargs[key] = overrides[key]

        # Resolve API key (never logged/printed)
        provider = get_provider_for_model(model)
        if provider not in ("ollama", "unknown"):
            try:
                api_key = self._auth.get_key(provider)
                kwargs["api_key"] = api_key
            except Exception:
                logger.debug("auth_key_not_found", provider=provider)

        # Provider-specific base URL
        model_info = self._registry.get_model_info(model)
        if model_info:
            provider_cfg = self._registry.get_provider(model_info.provider)
            if provider_cfg and provider_cfg.api_base:
                kwargs["api_base"] = provider_cfg.api_base

        return kwargs

    @staticmethod
    def _parse_response(
        raw: dict[str, Any],
        model: str,
        latency_ms: float,
    ) -> CompletionResult:
        """Convert a raw LiteLLM response dict into a :class:`CompletionResult`."""
        choices = raw.get("choices", [])
        if not choices:
            return CompletionResult(
                content="",
                model=model,
                provider=get_provider_for_model(model),
                input_tokens=0,
                output_tokens=0,
                cached_tokens=0,
                cost_usd=0.0,
                latency_ms=latency_ms,
                finish_reason="error",
            )

        message = choices[0].get("message", {})
        content = message.get("content", "") or ""
        finish_reason = choices[0].get("finish_reason", "stop")
        tool_calls = message.get("tool_calls")

        usage = raw.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cached_tokens = usage.get("cached_tokens", 0)

        try:
            cost = calculate_cost(model, input_tokens, output_tokens, cached_tokens)
        except ValueError:
            cost = 0.0

        provider = get_provider_for_model(model)

        return CompletionResult(
            content=content,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw_response=raw,
        )
