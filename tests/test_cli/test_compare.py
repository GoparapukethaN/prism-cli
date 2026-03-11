"""Tests for prism.cli.compare — model comparison mode."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest
from rich.console import Console

from prism.cli.compare import (
    _MAX_MODELS,
    _MIN_MODELS,
    _PANEL_COLORS,
    _SIDE_BY_SIDE_MAX_PANELS,
    _SIDE_BY_SIDE_MIN_WIDTH,
    DEFAULT_COMPARISON_MODELS,
    MODEL_DISPLAY_NAMES,
    ComparisonResult,
    ComparisonSession,
    ModelComparator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeCompletionResult:
    """Lightweight stand-in for CompletionResult used by the mock engine."""

    content: str = "Hello from the model."
    input_tokens: int = 50
    output_tokens: int = 100
    cost_usd: float = 0.0025
    latency_ms: float = 500.0


def _make_console(width: int = 200) -> Console:
    """Create a console writing to an in-memory buffer with a configurable width."""
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, no_color=True, width=width)


def _get_output(console: Console) -> str:
    """Extract rendered text from an in-memory console."""
    assert isinstance(console.file, io.StringIO)
    return console.file.getvalue()


def _make_engine(
    content: str = "Test response",
    input_tokens: int = 50,
    output_tokens: int = 100,
    cost_usd: float = 0.0025,
) -> AsyncMock:
    """Create a mock completion engine returning fixed results."""
    engine = AsyncMock()
    engine.complete.return_value = FakeCompletionResult(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )
    return engine


def _make_session(
    prompt: str = "What is 2+2?",
    num_results: int = 3,
    include_error: bool = False,
) -> ComparisonSession:
    """Build a pre-populated ComparisonSession for display tests."""
    models = DEFAULT_COMPARISON_MODELS[:num_results]
    results: list[ComparisonResult] = []
    for i, model in enumerate(models):
        if include_error and i == num_results - 1:
            results.append(ComparisonResult(
                model=model,
                display_name=MODEL_DISPLAY_NAMES.get(model, model),
                content="",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=0.0,
                error="Provider rate-limited",
            ))
        else:
            results.append(ComparisonResult(
                model=model,
                display_name=MODEL_DISPLAY_NAMES.get(model, model),
                content=f"Response from model {i + 1}",
                input_tokens=50 + i * 10,
                output_tokens=100 + i * 20,
                cost_usd=0.001 * (i + 1),
                latency_ms=300.0 + i * 100,
            ))
    return ComparisonSession(
        prompt=prompt,
        results=results,
        created_at="2026-03-11T00:00:00+00:00",
    )


# ===========================================================================
# TestComparisonResult
# ===========================================================================


class TestComparisonResult:
    """Tests for the ComparisonResult dataclass."""

    def test_basic_fields(self) -> None:
        r = ComparisonResult(
            model="gpt-4o",
            display_name="GPT-4o",
            content="Hi",
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.001,
            latency_ms=250.0,
        )
        assert r.model == "gpt-4o"
        assert r.display_name == "GPT-4o"
        assert r.content == "Hi"
        assert r.input_tokens == 10
        assert r.output_tokens == 20
        assert r.cost_usd == 0.001
        assert r.latency_ms == 250.0
        assert r.error is None

    def test_succeeded_true_when_no_error(self) -> None:
        r = ComparisonResult(
            model="m", display_name="M", content="ok",
            input_tokens=1, output_tokens=1, cost_usd=0.0, latency_ms=0.0,
        )
        assert r.succeeded is True

    def test_succeeded_false_when_error(self) -> None:
        r = ComparisonResult(
            model="m", display_name="M", content="",
            input_tokens=0, output_tokens=0, cost_usd=0.0, latency_ms=0.0,
            error="timeout",
        )
        assert r.succeeded is False

    def test_total_tokens(self) -> None:
        r = ComparisonResult(
            model="m", display_name="M", content="",
            input_tokens=30, output_tokens=70, cost_usd=0.0, latency_ms=0.0,
        )
        assert r.total_tokens == 100

    def test_total_tokens_zero(self) -> None:
        r = ComparisonResult(
            model="m", display_name="M", content="",
            input_tokens=0, output_tokens=0, cost_usd=0.0, latency_ms=0.0,
        )
        assert r.total_tokens == 0

    def test_error_field(self) -> None:
        r = ComparisonResult(
            model="m", display_name="M", content="",
            input_tokens=0, output_tokens=0, cost_usd=0.0, latency_ms=0.0,
            error="API key invalid",
        )
        assert r.error == "API key invalid"
        assert r.succeeded is False


# ===========================================================================
# TestComparisonSession
# ===========================================================================


class TestComparisonSession:
    """Tests for the ComparisonSession dataclass."""

    def test_creation_defaults(self) -> None:
        s = ComparisonSession(prompt="hello")
        assert s.prompt == "hello"
        assert s.results == []
        assert s.winner_index is None
        assert s.created_at == ""
        assert s.system_prompt == ""

    def test_has_winner_false_initially(self) -> None:
        s = ComparisonSession(prompt="test")
        assert s.has_winner is False

    def test_has_winner_true_after_set(self) -> None:
        s = _make_session()
        s.winner_index = 1
        assert s.has_winner is True

    def test_winner_returns_none_initially(self) -> None:
        s = ComparisonSession(prompt="test")
        assert s.winner is None

    def test_winner_returns_correct_result(self) -> None:
        s = _make_session(num_results=3)
        s.winner_index = 1
        assert s.winner is not None
        assert s.winner.model == s.results[1].model

    def test_winner_out_of_range_returns_none(self) -> None:
        s = _make_session(num_results=2)
        s.winner_index = 99
        assert s.winner is None

    def test_winner_negative_returns_none(self) -> None:
        s = _make_session(num_results=2)
        s.winner_index = -1
        assert s.winner is None

    def test_total_cost(self) -> None:
        s = _make_session(num_results=3)
        expected = sum(r.cost_usd for r in s.results)
        assert abs(s.total_cost - expected) < 1e-10

    def test_total_cost_empty(self) -> None:
        s = ComparisonSession(prompt="x")
        assert s.total_cost == 0.0

    def test_successful_count(self) -> None:
        s = _make_session(num_results=3, include_error=True)
        assert s.successful_count == 2

    def test_successful_count_all_ok(self) -> None:
        s = _make_session(num_results=3, include_error=False)
        assert s.successful_count == 3

    def test_system_prompt_stored(self) -> None:
        s = ComparisonSession(prompt="q", system_prompt="You are a helpful assistant.")
        assert s.system_prompt == "You are a helpful assistant."


# ===========================================================================
# TestModelComparatorInit
# ===========================================================================


class TestModelComparatorInit:
    """Tests for ModelComparator construction and property access."""

    def test_default_models(self) -> None:
        comp = ModelComparator(_make_engine())
        assert comp.models == DEFAULT_COMPARISON_MODELS

    def test_custom_models(self) -> None:
        custom = ["gpt-4o", "deepseek/deepseek-chat"]
        comp = ModelComparator(_make_engine(), models=custom)
        assert comp.models == custom

    def test_models_are_copied(self) -> None:
        """Mutating the original list must not affect the comparator."""
        original = ["gpt-4o", "deepseek/deepseek-chat"]
        comp = ModelComparator(_make_engine(), models=original)
        original.append("extra-model")
        assert len(comp.models) == 2

    def test_models_getter_returns_copy(self) -> None:
        comp = ModelComparator(_make_engine())
        m = comp.models
        m.clear()
        assert len(comp.models) == len(DEFAULT_COMPARISON_MODELS)

    def test_init_too_few_models_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            ModelComparator(_make_engine(), models=["gpt-4o"])

    def test_init_empty_models_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            ModelComparator(_make_engine(), models=[])

    def test_init_too_many_models_raises(self) -> None:
        many = [f"model-{i}" for i in range(6)]
        with pytest.raises(ValueError, match="Maximum 5"):
            ModelComparator(_make_engine(), models=many)

    def test_custom_console(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        assert comp.console is console

    def test_history_empty_initially(self) -> None:
        comp = ModelComparator(_make_engine())
        assert comp.history == []


# ===========================================================================
# TestModelsSetter
# ===========================================================================


class TestModelsSetter:
    """Tests for the ``models`` property setter."""

    def test_set_valid_models(self) -> None:
        comp = ModelComparator(_make_engine())
        comp.models = ["a", "b"]
        assert comp.models == ["a", "b"]

    def test_set_max_models(self) -> None:
        comp = ModelComparator(_make_engine())
        comp.models = ["a", "b", "c", "d", "e"]
        assert len(comp.models) == 5

    def test_set_too_few_raises(self) -> None:
        comp = ModelComparator(_make_engine())
        with pytest.raises(ValueError, match="at least 2"):
            comp.models = ["only-one"]

    def test_set_too_many_raises(self) -> None:
        comp = ModelComparator(_make_engine())
        with pytest.raises(ValueError, match="Maximum 5"):
            comp.models = [f"m{i}" for i in range(6)]

    def test_set_not_a_list_raises(self) -> None:
        comp = ModelComparator(_make_engine())
        with pytest.raises(TypeError, match="must be a list"):
            comp.models = ("a", "b")  # type: ignore[assignment]

    def test_setter_copies(self) -> None:
        comp = ModelComparator(_make_engine())
        new = ["a", "b"]
        comp.models = new
        new.append("c")
        assert len(comp.models) == 2


# ===========================================================================
# TestCompare
# ===========================================================================


class TestCompare:
    """Tests for the async ``compare`` method."""

    @pytest.mark.asyncio
    async def test_compare_returns_session(self) -> None:
        comp = ModelComparator(_make_engine())
        session = await comp.compare("What is Python?")
        assert isinstance(session, ComparisonSession)
        assert session.prompt == "What is Python?"
        assert len(session.results) == len(DEFAULT_COMPARISON_MODELS)

    @pytest.mark.asyncio
    async def test_compare_all_succeed(self) -> None:
        comp = ModelComparator(_make_engine())
        session = await comp.compare("Hello")
        for result in session.results:
            assert result.succeeded
            assert result.content == "Test response"
            assert result.input_tokens == 50
            assert result.output_tokens == 100

    @pytest.mark.asyncio
    async def test_compare_records_latency(self) -> None:
        comp = ModelComparator(_make_engine())
        session = await comp.compare("Hi")
        for result in session.results:
            assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_compare_one_model_fails(self) -> None:
        engine = AsyncMock()
        call_count = 0

        async def _side_effect(messages: Any, model: str, **kw: Any) -> FakeCompletionResult:
            nonlocal call_count
            call_count += 1
            if model == "gpt-4o":
                raise RuntimeError("API key expired")
            return FakeCompletionResult()

        engine.complete.side_effect = _side_effect

        comp = ModelComparator(engine)
        session = await comp.compare("test prompt")

        assert len(session.results) == 3
        gpt_result = next(r for r in session.results if r.model == "gpt-4o")
        assert not gpt_result.succeeded
        assert "API key expired" in (gpt_result.error or "")

        ok_results = [r for r in session.results if r.succeeded]
        assert len(ok_results) == 2

    @pytest.mark.asyncio
    async def test_compare_all_models_fail(self) -> None:
        engine = AsyncMock()
        engine.complete.side_effect = ConnectionError("no network")

        comp = ModelComparator(engine)
        session = await comp.compare("test")

        assert len(session.results) == 3
        for result in session.results:
            assert not result.succeeded
            assert "no network" in (result.error or "")
        assert session.successful_count == 0

    @pytest.mark.asyncio
    async def test_compare_with_system_prompt(self) -> None:
        engine = _make_engine()
        comp = ModelComparator(engine)
        session = await comp.compare("hi", system_prompt="Be concise.")

        assert session.system_prompt == "Be concise."
        # The engine should have received system + user messages
        for call_args in engine.complete.call_args_list:
            messages = call_args.kwargs.get("messages") or call_args[1].get("messages", [])
            if not messages:
                messages = call_args[0][0] if call_args[0] else []
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_compare_without_system_prompt(self) -> None:
        engine = _make_engine()
        comp = ModelComparator(engine)
        await comp.compare("hi")

        for call_args in engine.complete.call_args_list:
            messages = call_args.kwargs.get("messages") or call_args[1].get("messages", [])
            if not messages:
                messages = call_args[0][0] if call_args[0] else []
            assert len(messages) == 1
            assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_compare_empty_prompt_raises(self) -> None:
        comp = ModelComparator(_make_engine())
        with pytest.raises(ValueError, match="must not be empty"):
            await comp.compare("")

    @pytest.mark.asyncio
    async def test_compare_whitespace_only_prompt_raises(self) -> None:
        comp = ModelComparator(_make_engine())
        with pytest.raises(ValueError, match="must not be empty"):
            await comp.compare("   ")

    @pytest.mark.asyncio
    async def test_compare_parallel_execution(self) -> None:
        """Verify all models are called concurrently, not sequentially."""
        call_order: list[str] = []

        async def _track(messages: Any, model: str, **kw: Any) -> FakeCompletionResult:
            call_order.append(model)
            # All tasks should start before any finishes
            if len(call_order) < 3:
                await asyncio.sleep(0.01)
            return FakeCompletionResult()

        engine = AsyncMock()
        engine.complete.side_effect = _track

        comp = ModelComparator(engine)
        session = await comp.compare("parallel test")

        assert len(session.results) == 3
        assert engine.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_compare_sets_created_at(self) -> None:
        comp = ModelComparator(_make_engine())
        session = await comp.compare("timestamp test")
        assert session.created_at != ""
        assert "T" in session.created_at  # ISO format

    @pytest.mark.asyncio
    async def test_compare_uses_display_names(self) -> None:
        comp = ModelComparator(_make_engine())
        session = await comp.compare("names test")
        for result in session.results:
            expected_name = MODEL_DISPLAY_NAMES.get(result.model, result.model)
            assert result.display_name == expected_name

    @pytest.mark.asyncio
    async def test_compare_exception_propagated_as_error(self) -> None:
        """BaseException from gather is captured, not re-raised."""
        engine = AsyncMock()

        async def _boom(messages: Any, model: str, **kw: Any) -> FakeCompletionResult:
            if model == DEFAULT_COMPARISON_MODELS[0]:
                raise ValueError("bad model config")
            return FakeCompletionResult()

        engine.complete.side_effect = _boom

        comp = ModelComparator(engine)
        session = await comp.compare("exc test")

        failed = [r for r in session.results if not r.succeeded]
        assert len(failed) == 1
        assert "bad model config" in (failed[0].error or "")


# ===========================================================================
# TestHistory
# ===========================================================================


class TestHistory:
    """Tests for history tracking."""

    @pytest.mark.asyncio
    async def test_sessions_appended(self) -> None:
        comp = ModelComparator(_make_engine())
        await comp.compare("first")
        await comp.compare("second")
        assert len(comp.history) == 2

    @pytest.mark.asyncio
    async def test_history_preserves_order(self) -> None:
        comp = ModelComparator(_make_engine())
        await comp.compare("alpha")
        await comp.compare("beta")
        assert comp.history[0].prompt == "alpha"
        assert comp.history[1].prompt == "beta"

    @pytest.mark.asyncio
    async def test_history_is_copy(self) -> None:
        comp = ModelComparator(_make_engine())
        await comp.compare("test")
        history = comp.history
        history.clear()
        assert len(comp.history) == 1


# ===========================================================================
# TestDisplayResults
# ===========================================================================


class TestDisplayResults:
    """Tests for display_results rendering."""

    def test_display_with_results(self) -> None:
        console = _make_console(width=200)
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=3)
        comp.display_results(session)
        output = _get_output(console)
        assert "Claude Sonnet 4" in output
        assert "GPT-4o" in output
        assert "DeepSeek V3" in output

    def test_display_shows_token_counts(self) -> None:
        console = _make_console(width=200)
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=2)
        comp.display_results(session)
        output = _get_output(console)
        assert "Tokens:" in output

    def test_display_shows_cost(self) -> None:
        console = _make_console(width=200)
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=2)
        comp.display_results(session)
        output = _get_output(console)
        assert "$" in output

    def test_display_shows_latency(self) -> None:
        console = _make_console(width=200)
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=2)
        comp.display_results(session)
        output = _get_output(console)
        assert "ms" in output

    def test_display_error_result(self) -> None:
        console = _make_console(width=200)
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=3, include_error=True)
        comp.display_results(session)
        output = _get_output(console)
        assert "Error:" in output
        assert "Provider rate-limited" in output

    def test_display_stacked_on_narrow_terminal(self) -> None:
        """With a narrow terminal, panels are stacked vertically."""
        console = _make_console(width=80)
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=3)
        comp.display_results(session)
        output = _get_output(console)
        # All model names should still appear
        assert "Claude Sonnet 4" in output
        assert "GPT-4o" in output

    def test_display_stacked_with_many_panels(self) -> None:
        """More than 3 panels force stacked layout even on wide terminal."""
        console = _make_console(width=200)
        models = ["gpt-4o", "deepseek/deepseek-chat", "gemini/gemini-1.5-pro", "gpt-4o-mini"]
        comp = ModelComparator(_make_engine(), models=models, console=console)
        session = ComparisonSession(prompt="test", results=[
            ComparisonResult(
                model=m,
                display_name=MODEL_DISPLAY_NAMES.get(m, m),
                content=f"Response from {m}",
                input_tokens=50, output_tokens=100, cost_usd=0.001, latency_ms=300.0,
            )
            for m in models
        ])
        comp.display_results(session)
        output = _get_output(console)
        assert "GPT-4o" in output
        assert "GPT-4o Mini" in output

    def test_display_empty_session(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        session = ComparisonSession(prompt="empty")
        comp.display_results(session)
        output = _get_output(console)
        assert "No results" in output

    def test_display_numbered_panels(self) -> None:
        console = _make_console(width=200)
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=3)
        comp.display_results(session)
        output = _get_output(console)
        assert "[1]" in output
        assert "[2]" in output
        assert "[3]" in output

    def test_display_empty_response_content(self) -> None:
        console = _make_console(width=200)
        comp = ModelComparator(_make_engine(), console=console)
        session = ComparisonSession(prompt="test", results=[
            ComparisonResult(
                model="gpt-4o", display_name="GPT-4o", content="",
                input_tokens=10, output_tokens=0, cost_usd=0.0, latency_ms=100.0,
            ),
            ComparisonResult(
                model="deepseek/deepseek-chat", display_name="DeepSeek V3",
                content="Some text",
                input_tokens=10, output_tokens=50, cost_usd=0.001, latency_ms=200.0,
            ),
        ])
        comp.display_results(session)
        output = _get_output(console)
        assert "(empty response)" in output


# ===========================================================================
# TestDisplayCostTable
# ===========================================================================


class TestDisplayCostTable:
    """Tests for display_cost_table rendering."""

    def test_table_header(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=2)
        comp.display_cost_table(session)
        output = _get_output(console)
        assert "Cost Comparison" in output
        assert "Model" in output
        assert "Input Tokens" in output
        assert "Output Tokens" in output
        assert "Status" in output

    def test_table_rows(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=3)
        comp.display_cost_table(session)
        output = _get_output(console)
        assert "Claude Sonnet 4" in output
        assert "GPT-4o" in output
        assert "DeepSeek V3" in output

    def test_table_status_ok(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=2, include_error=False)
        comp.display_cost_table(session)
        output = _get_output(console)
        assert "OK" in output

    def test_table_status_fail(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=3, include_error=True)
        comp.display_cost_table(session)
        output = _get_output(console)
        assert "FAIL" in output

    def test_table_total_row(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=3)
        comp.display_cost_table(session)
        output = _get_output(console)
        assert "Total" in output

    def test_table_empty_session(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        session = ComparisonSession(prompt="empty")
        comp.display_cost_table(session)
        output = _get_output(console)
        assert "No results" in output

    def test_table_total_tokens_column(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=2)
        comp.display_cost_table(session)
        output = _get_output(console)
        assert "Total Tokens" in output


# ===========================================================================
# TestRecordWinner
# ===========================================================================


class TestRecordWinner:
    """Tests for record_winner."""

    def test_valid_choice_first(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=3)
        comp.record_winner(session, 1)
        assert session.winner_index == 0
        assert session.has_winner

    def test_valid_choice_last(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=3)
        comp.record_winner(session, 3)
        assert session.winner_index == 2

    def test_valid_choice_middle(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=3)
        comp.record_winner(session, 2)
        assert session.winner_index == 1
        assert session.winner is not None
        assert session.winner.model == session.results[1].model

    def test_invalid_choice_zero(self) -> None:
        comp = ModelComparator(_make_engine())
        session = _make_session(num_results=3)
        with pytest.raises(ValueError, match="Invalid choice: 0"):
            comp.record_winner(session, 0)

    def test_invalid_choice_negative(self) -> None:
        comp = ModelComparator(_make_engine())
        session = _make_session(num_results=3)
        with pytest.raises(ValueError, match="Invalid choice: -1"):
            comp.record_winner(session, -1)

    def test_invalid_choice_too_large(self) -> None:
        comp = ModelComparator(_make_engine())
        session = _make_session(num_results=3)
        with pytest.raises(ValueError, match="Invalid choice: 4"):
            comp.record_winner(session, 4)

    def test_empty_session_raises(self) -> None:
        comp = ModelComparator(_make_engine())
        session = ComparisonSession(prompt="test")
        with pytest.raises(ValueError, match="no results"):
            comp.record_winner(session, 1)

    def test_winner_recorded_prints_confirmation(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=2)
        comp.record_winner(session, 1)
        output = _get_output(console)
        assert "Winner recorded" in output
        assert session.results[0].display_name in output


# ===========================================================================
# TestDisplayPromptHint
# ===========================================================================


class TestDisplayPromptHint:
    """Tests for display_prompt_hint."""

    def test_shows_numbered_choices(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=3)
        comp.display_prompt_hint(session)
        output = _get_output(console)
        assert "1/2/3" in output

    def test_two_models(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        session = _make_session(num_results=2)
        comp.display_prompt_hint(session)
        output = _get_output(console)
        assert "1/2" in output

    def test_empty_session_no_output(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        session = ComparisonSession(prompt="empty")
        comp.display_prompt_hint(session)
        output = _get_output(console)
        assert output == ""


# ===========================================================================
# TestGetConfigSummary
# ===========================================================================


class TestGetConfigSummary:
    """Tests for configuration display."""

    def test_summary_lists_all_models(self) -> None:
        comp = ModelComparator(_make_engine())
        summary = comp.get_config_summary()
        for model in DEFAULT_COMPARISON_MODELS:
            assert model in summary

    def test_summary_includes_display_names(self) -> None:
        comp = ModelComparator(_make_engine())
        summary = comp.get_config_summary()
        assert "Claude Sonnet 4" in summary
        assert "GPT-4o" in summary
        assert "DeepSeek V3" in summary

    def test_summary_numbered(self) -> None:
        comp = ModelComparator(_make_engine())
        summary = comp.get_config_summary()
        assert "1." in summary
        assert "2." in summary
        assert "3." in summary

    def test_display_config_outputs_to_console(self) -> None:
        console = _make_console()
        comp = ModelComparator(_make_engine(), console=console)
        comp.display_config()
        output = _get_output(console)
        assert "Comparison models" in output
        assert "Claude Sonnet 4" in output

    def test_summary_custom_models(self) -> None:
        comp = ModelComparator(
            _make_engine(),
            models=["gpt-4o", "gpt-4o-mini"],
        )
        summary = comp.get_config_summary()
        assert "GPT-4o" in summary
        assert "GPT-4o Mini" in summary

    def test_summary_unknown_model_uses_id(self) -> None:
        comp = ModelComparator(
            _make_engine(),
            models=["gpt-4o", "custom/my-model"],
        )
        summary = comp.get_config_summary()
        assert "custom/my-model" in summary


# ===========================================================================
# TestDefaultModels
# ===========================================================================


class TestDefaultModels:
    """Tests for the module-level defaults."""

    def test_default_models_count(self) -> None:
        assert len(DEFAULT_COMPARISON_MODELS) == 3

    def test_default_models_contains_claude(self) -> None:
        assert any("claude" in m for m in DEFAULT_COMPARISON_MODELS)

    def test_default_models_contains_gpt(self) -> None:
        assert any("gpt" in m for m in DEFAULT_COMPARISON_MODELS)

    def test_default_models_contains_deepseek(self) -> None:
        assert any("deepseek" in m for m in DEFAULT_COMPARISON_MODELS)

    def test_all_defaults_have_display_names(self) -> None:
        for model in DEFAULT_COMPARISON_MODELS:
            assert model in MODEL_DISPLAY_NAMES

    def test_display_names_is_dict(self) -> None:
        assert isinstance(MODEL_DISPLAY_NAMES, dict)

    def test_panel_colors_has_enough(self) -> None:
        assert len(_PANEL_COLORS) >= _MAX_MODELS

    def test_min_max_constants(self) -> None:
        assert _MIN_MODELS == 2
        assert _MAX_MODELS == 5
        assert _MIN_MODELS < _MAX_MODELS

    def test_side_by_side_constants(self) -> None:
        assert _SIDE_BY_SIDE_MIN_WIDTH == 120
        assert _SIDE_BY_SIDE_MAX_PANELS == 3
