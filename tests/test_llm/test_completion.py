"""Tests for CompletionEngine — 15+ tests, all using MockLiteLLM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.exceptions import BudgetExceededError, ModelNotFoundError
from prism.llm.completion import CompletionEngine
from prism.llm.mock import MockLiteLLM, MockResponse

if TYPE_CHECKING:
    from unittest.mock import MagicMock


class TestBasicCompletion:
    """Core happy-path tests."""

    async def test_basic_completion(
        self,
        completion_engine: CompletionEngine,
        mock_litellm: MockLiteLLM,
        simple_messages: list[dict[str, str]],
    ) -> None:
        mock_litellm.set_response(
            "gpt-4o",
            MockResponse(content="Hello!", input_tokens=10, output_tokens=5),
        )
        result = await completion_engine.complete(
            messages=simple_messages, model="gpt-4o",
        )
        assert result.content == "Hello!"
        assert result.model == "gpt-4o"
        assert result.provider == "openai"
        assert result.finish_reason == "stop"

    async def test_model_none_raises(
        self,
        completion_engine: CompletionEngine,
        simple_messages: list[dict[str, str]],
    ) -> None:
        with pytest.raises(ModelNotFoundError):
            await completion_engine.complete(messages=simple_messages, model=None)

    async def test_completion_records_call_in_mock(
        self,
        completion_engine: CompletionEngine,
        mock_litellm: MockLiteLLM,
        simple_messages: list[dict[str, str]],
    ) -> None:
        await completion_engine.complete(messages=simple_messages, model="gpt-4o")
        assert len(mock_litellm.call_log) == 1
        assert mock_litellm.call_log[0]["model"] == "gpt-4o"

    async def test_default_mock_response(
        self,
        completion_engine: CompletionEngine,
        simple_messages: list[dict[str, str]],
    ) -> None:
        """When no model-specific response is set, the default is used."""
        result = await completion_engine.complete(
            messages=simple_messages, model="gpt-4o",
        )
        assert result.content == "Mock response from the AI model."


class TestBudgetEnforcement:
    """Budget checks before calling."""

    async def test_budget_block_raises(
        self,
        llm_settings: object,
        mock_cost_tracker_blocked: MagicMock,
        mock_auth: MagicMock,
        mock_registry: object,
        mock_litellm: MockLiteLLM,
        simple_messages: list[dict[str, str]],
    ) -> None:
        engine = CompletionEngine(
            settings=llm_settings,
            cost_tracker=mock_cost_tracker_blocked,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm,
        )
        with pytest.raises(BudgetExceededError):
            await engine.complete(messages=simple_messages, model="gpt-4o")

    async def test_budget_proceed_succeeds(
        self,
        completion_engine: CompletionEngine,
        simple_messages: list[dict[str, str]],
    ) -> None:
        result = await completion_engine.complete(
            messages=simple_messages, model="gpt-4o",
        )
        assert result.content


class TestContextWindow:
    """Context window enforcement (message trimming)."""

    async def test_messages_within_window_untouched(
        self,
        completion_engine: CompletionEngine,
        mock_litellm: MockLiteLLM,
        simple_messages: list[dict[str, str]],
    ) -> None:
        await completion_engine.complete(
            messages=simple_messages, model="gpt-4o",
        )
        sent_messages = mock_litellm.call_log[0]["messages"]
        assert len(sent_messages) == len(simple_messages)

    async def test_long_messages_trimmed(
        self,
        completion_engine: CompletionEngine,
        mock_litellm: MockLiteLLM,
        long_messages: list[dict[str, str]],
    ) -> None:
        """Very long message lists get trimmed to fit the context window."""
        await completion_engine.complete(
            messages=long_messages, model="gpt-4o",
        )
        sent_messages = mock_litellm.call_log[0]["messages"]
        # The system message and last message should always be kept.
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[-1] == long_messages[-1]
        # Should have fewer messages than the original.
        assert len(sent_messages) < len(long_messages)


class TestCostTracking:
    """Cost is tracked after successful calls."""

    async def test_cost_tracked_with_session(
        self,
        completion_engine: CompletionEngine,
        mock_cost_tracker: MagicMock,
        simple_messages: list[dict[str, str]],
    ) -> None:
        await completion_engine.complete(
            messages=simple_messages,
            model="gpt-4o",
            session_id="sess-123",
        )
        mock_cost_tracker.track.assert_called_once()
        call_kwargs = mock_cost_tracker.track.call_args
        assert call_kwargs.kwargs["model_id"] == "gpt-4o"
        assert call_kwargs.kwargs["session_id"] == "sess-123"

    async def test_cost_not_tracked_without_session(
        self,
        completion_engine: CompletionEngine,
        mock_cost_tracker: MagicMock,
        simple_messages: list[dict[str, str]],
    ) -> None:
        await completion_engine.complete(
            messages=simple_messages, model="gpt-4o",
        )
        mock_cost_tracker.track.assert_not_called()


class TestErrorHandling:
    """Provider errors, timeouts, auth failures."""

    async def test_provider_error_propagates(
        self,
        completion_engine: CompletionEngine,
        mock_litellm: MockLiteLLM,
        simple_messages: list[dict[str, str]],
    ) -> None:
        from prism.exceptions import ProviderUnavailableError

        mock_litellm.set_error(
            "gpt-4o", ProviderUnavailableError("openai", "server down"),
        )
        with pytest.raises(ProviderUnavailableError):
            await completion_engine.complete(
                messages=simple_messages, model="gpt-4o",
            )

    async def test_auth_error_not_retried(
        self,
        mock_litellm: MockLiteLLM,
        llm_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: object,
        simple_messages: list[dict[str, str]],
    ) -> None:
        from prism.exceptions import ProviderAuthError
        from prism.llm.retry import RetryPolicy

        mock_litellm.set_error("gpt-4o", ProviderAuthError("openai"))
        engine = CompletionEngine(
            settings=llm_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm,
            retry_policy=RetryPolicy(max_retries=3),
        )
        with pytest.raises(ProviderAuthError):
            await engine.complete(messages=simple_messages, model="gpt-4o")
        # Auth error is NOT retryable, so only 1 call.
        assert len(mock_litellm.call_log) == 1

    async def test_timeout_error_retried(
        self,
        mock_litellm: MockLiteLLM,
        llm_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: object,
        simple_messages: list[dict[str, str]],
    ) -> None:
        from prism.llm.retry import RetryPolicy

        # Set a TimeoutError — should be retried
        mock_litellm.set_error("gpt-4o", TimeoutError("request timed out"))
        engine = CompletionEngine(
            settings=llm_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm,
            retry_policy=RetryPolicy(max_retries=2, base_delay=0.001),
        )
        with pytest.raises(TimeoutError):
            await engine.complete(messages=simple_messages, model="gpt-4o")
        # 1 original + 2 retries = 3 calls
        assert len(mock_litellm.call_log) == 3


class TestToolCalls:
    """Tool calls in response."""

    async def test_tool_calls_in_result(
        self,
        completion_engine: CompletionEngine,
        mock_litellm: MockLiteLLM,
        simple_messages: list[dict[str, str]],
    ) -> None:
        mock_litellm.set_response(
            "gpt-4o",
            MockResponse(
                content="Let me search that for you.",
                tool_calls=[{
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }],
            ),
        )
        result = await completion_engine.complete(
            messages=simple_messages, model="gpt-4o",
        )
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "search"


class TestParameterPassthrough:
    """Temperature, max_tokens, tools forwarded correctly."""

    async def test_temperature_passthrough(
        self,
        completion_engine: CompletionEngine,
        mock_litellm: MockLiteLLM,
        simple_messages: list[dict[str, str]],
    ) -> None:
        await completion_engine.complete(
            messages=simple_messages, model="gpt-4o", temperature=0.2,
        )
        kwargs = mock_litellm.call_log[0]["kwargs"]
        assert kwargs.get("temperature") == 0.2

    async def test_max_tokens_passthrough(
        self,
        completion_engine: CompletionEngine,
        mock_litellm: MockLiteLLM,
        simple_messages: list[dict[str, str]],
    ) -> None:
        await completion_engine.complete(
            messages=simple_messages, model="gpt-4o", max_tokens=512,
        )
        kwargs = mock_litellm.call_log[0]["kwargs"]
        assert kwargs.get("max_tokens") == 512

    async def test_tools_passthrough(
        self,
        completion_engine: CompletionEngine,
        mock_litellm: MockLiteLLM,
        simple_messages: list[dict[str, str]],
    ) -> None:
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        await completion_engine.complete(
            messages=simple_messages, model="gpt-4o", tools=tools,
        )
        kwargs = mock_litellm.call_log[0]["kwargs"]
        assert kwargs.get("tools") == tools
