"""Tests for prism.intelligence.debate — Multi-Model Debate Mode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from prism.intelligence.debate import (
    DEFAULT_DEBATE_MODELS,
    DebateCritique,
    DebatePosition,
    DebateSession,
    DebateSynthesis,
    MultiModelDebate,
)

# ======================================================================
# Helpers — mock completion engine
# ======================================================================


@dataclass
class MockCompletionResult:
    """Minimal mock for the CompletionResult protocol."""

    content: str = "Mock response"
    input_tokens: int = 100
    output_tokens: int = 200
    cost_usd: float = 0.005


class MockCompletionEngine:
    """Async completion engine that returns canned responses."""

    def __init__(
        self,
        default_content: str = "Mock response",
        fail_on_model: str | None = None,
    ) -> None:
        self._default_content = default_content
        self._fail_on_model = fail_on_model
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
    ) -> MockCompletionResult:
        self.calls.append({"messages": messages, "model": model})
        if self._fail_on_model and model == self._fail_on_model:
            raise RuntimeError(f"Simulated failure for {model}")
        return MockCompletionResult(content=f"Response from {model}")


# ======================================================================
# TestDebatePosition
# ======================================================================


class TestDebatePosition:
    """Tests for the DebatePosition dataclass."""

    def test_fields(self) -> None:
        pos = DebatePosition(
            model="gpt-4o",
            content="I think X is best",
            tokens_used=300,
            cost_usd=0.01,
            round_number=1,
        )
        assert pos.model == "gpt-4o"
        assert pos.content == "I think X is best"
        assert pos.tokens_used == 300
        assert pos.cost_usd == 0.01
        assert pos.round_number == 1

    def test_error_position(self) -> None:
        pos = DebatePosition(
            model="bad-model",
            content="Error: connection refused",
            tokens_used=0,
            cost_usd=0.0,
            round_number=1,
        )
        assert pos.tokens_used == 0
        assert pos.cost_usd == 0.0


# ======================================================================
# TestDebateCritique
# ======================================================================


class TestDebateCritique:
    """Tests for the DebateCritique dataclass."""

    def test_fields(self) -> None:
        critique = DebateCritique(
            model="claude-sonnet-4-20250514",
            target_model="all",
            strengths="Good reasoning",
            weaknesses="Missing edge case",
            risks_missed="Security risk",
            updated_position="Updated view",
            tokens_used=500,
            cost_usd=0.02,
        )
        assert critique.model == "claude-sonnet-4-20250514"
        assert critique.target_model == "all"
        assert critique.strengths == "Good reasoning"
        assert critique.weaknesses == "Missing edge case"
        assert critique.risks_missed == "Security risk"
        assert critique.updated_position == "Updated view"
        assert critique.tokens_used == 500
        assert critique.cost_usd == 0.02

    def test_error_critique(self) -> None:
        critique = DebateCritique(
            model="bad-model",
            target_model="all",
            strengths="",
            weaknesses="",
            risks_missed="",
            updated_position="Error: timeout",
            tokens_used=0,
            cost_usd=0.0,
        )
        assert critique.tokens_used == 0
        assert "Error" in critique.updated_position


# ======================================================================
# TestDebateSynthesis
# ======================================================================


class TestDebateSynthesis:
    """Tests for the DebateSynthesis dataclass."""

    def test_fields(self) -> None:
        synth = DebateSynthesis(
            consensus_points=["Point A", "Point B"],
            disagreements=["Disagree on X"],
            recommendation="Go with approach A",
            confidence=0.85,
            what_each_missed={"gpt-4o": "missed Y"},
            synthesizer_model="claude-sonnet-4-20250514",
            tokens_used=800,
            cost_usd=0.03,
        )
        assert len(synth.consensus_points) == 2
        assert len(synth.disagreements) == 1
        assert synth.recommendation == "Go with approach A"
        assert synth.confidence == 0.85
        assert synth.what_each_missed["gpt-4o"] == "missed Y"
        assert synth.synthesizer_model == "claude-sonnet-4-20250514"
        assert synth.tokens_used == 800
        assert synth.cost_usd == 0.03


# ======================================================================
# TestDebateSession
# ======================================================================


class TestDebateSession:
    """Tests for the DebateSession dataclass."""

    def test_fields(self) -> None:
        session = DebateSession(
            question="Should we use React or Vue?",
            models=["gpt-4o", "claude-sonnet-4-20250514"],
        )
        assert session.question == "Should we use React or Vue?"
        assert len(session.models) == 2
        assert session.round1_positions == []
        assert session.round2_critiques == []
        assert session.synthesis is None
        assert session.total_cost == 0.0
        assert session.total_tokens == 0
        assert session.quick_mode is False

    def test_is_complete_false(self) -> None:
        session = DebateSession(question="Q?", models=["m1", "m2"])
        assert session.is_complete is False

    def test_is_complete_true(self) -> None:
        session = DebateSession(question="Q?", models=["m1", "m2"])
        session.synthesis = DebateSynthesis(
            consensus_points=[],
            disagreements=[],
            recommendation="Do X",
            confidence=0.9,
            what_each_missed={},
            synthesizer_model="m1",
            tokens_used=100,
            cost_usd=0.01,
        )
        assert session.is_complete is True

    def test_total_tracking(self) -> None:
        session = DebateSession(
            question="Q?",
            models=["m1", "m2"],
            total_cost=0.05,
            total_tokens=1000,
        )
        assert session.total_cost == 0.05
        assert session.total_tokens == 1000

    def test_quick_mode_flag(self) -> None:
        session = DebateSession(question="Q?", models=["m1", "m2"], quick_mode=True)
        assert session.quick_mode is True

    def test_created_at(self) -> None:
        session = DebateSession(
            question="Q?", models=["m1"], created_at="2025-01-01T00:00:00+00:00",
        )
        assert session.created_at == "2025-01-01T00:00:00+00:00"


# ======================================================================
# TestMultiModelDebate
# ======================================================================


class TestMultiModelDebate:
    """Tests for the MultiModelDebate class."""

    # --- Init ---

    def test_init_defaults(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine)
        assert debate.models == list(DEFAULT_DEBATE_MODELS)
        assert debate._synthesizer == DEFAULT_DEBATE_MODELS[0]

    def test_init_custom_models(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        assert debate.models == ["m1", "m2"]

    def test_init_custom_synthesizer(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, synthesizer="custom-synth")
        assert debate._synthesizer == "custom-synth"

    # --- Models property ---

    def test_models_getter_returns_copy(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        models = debate.models
        models.append("m3")
        assert len(debate.models) == 2  # original unchanged

    def test_models_setter_valid(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        debate.models = ["a", "b", "c"]
        assert debate.models == ["a", "b", "c"]

    def test_models_setter_too_few(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        with pytest.raises(ValueError, match="at least 2"):
            debate.models = ["only-one"]

    def test_models_setter_empty(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        with pytest.raises(ValueError, match="at least 2"):
            debate.models = []

    # --- Estimate cost ---

    def test_estimate_cost_default_models(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine)
        cost = debate.estimate_cost()
        # 3 models: (3*2+1) * 0.005 = 7 * 0.005 = 0.035
        assert cost == pytest.approx(0.035)

    def test_estimate_cost_two_models(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        cost = debate.estimate_cost()
        # 2 models: (2*2+1) * 0.005 = 5 * 0.005 = 0.025
        assert cost == pytest.approx(0.025)

    def test_estimate_cost_with_prompt_length(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        # prompt_length is reserved for future use, doesn't affect result yet
        cost = debate.estimate_cost(prompt_length=10000)
        assert cost == pytest.approx(0.025)

    # --- Debate full (async) ---

    @pytest.mark.asyncio
    async def test_debate_full(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        session = await debate.debate("Which database should we use?")

        assert session.is_complete
        assert session.question == "Which database should we use?"
        assert session.quick_mode is False
        assert len(session.round1_positions) == 2
        assert len(session.round2_critiques) == 2
        assert session.synthesis is not None
        assert session.total_cost > 0
        assert session.total_tokens > 0
        assert session.created_at  # non-empty

    @pytest.mark.asyncio
    async def test_debate_quick_mode(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        session = await debate.debate("Quick question?", quick=True)

        assert session.is_complete
        assert session.quick_mode is True
        assert len(session.round1_positions) == 2
        assert len(session.round2_critiques) == 0  # skipped in quick mode
        assert session.synthesis is not None

    @pytest.mark.asyncio
    async def test_debate_empty_question(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        with pytest.raises(ValueError, match="must not be empty"):
            await debate.debate("")

    @pytest.mark.asyncio
    async def test_debate_whitespace_question(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        with pytest.raises(ValueError, match="must not be empty"):
            await debate.debate("   ")

    @pytest.mark.asyncio
    async def test_debate_history_tracked(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        assert len(debate.history) == 0

        await debate.debate("Q1?")
        assert len(debate.history) == 1

        await debate.debate("Q2?")
        assert len(debate.history) == 2

    @pytest.mark.asyncio
    async def test_debate_history_returns_copy(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        await debate.debate("Q1?")
        h1 = debate.history
        h2 = debate.history
        assert h1 is not h2

    # --- _get_position ---

    @pytest.mark.asyncio
    async def test_get_position_success(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        pos = await debate._get_position("m1", "What is best?")

        assert isinstance(pos, DebatePosition)
        assert pos.model == "m1"
        assert "Response from m1" in pos.content
        assert pos.tokens_used == 300  # 100 + 200
        assert pos.cost_usd == 0.005
        assert pos.round_number == 1

    @pytest.mark.asyncio
    async def test_get_position_error(self) -> None:
        engine = MockCompletionEngine(fail_on_model="bad-model")
        debate = MultiModelDebate(engine, models=["bad-model", "m2"])
        pos = await debate._get_position("bad-model", "What?")

        assert isinstance(pos, DebatePosition)
        assert pos.model == "bad-model"
        assert "Error:" in pos.content
        assert pos.tokens_used == 0
        assert pos.cost_usd == 0.0

    # --- _get_critique ---

    @pytest.mark.asyncio
    async def test_get_critique_success(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        all_positions = [
            DebatePosition("m1", "Position A", 300, 0.005, 1),
            DebatePosition("m2", "Position B", 300, 0.005, 1),
        ]
        others = [all_positions[1]]

        critique = await debate._get_critique("m1", "Q?", all_positions, others)

        assert isinstance(critique, DebateCritique)
        assert critique.model == "m1"
        assert critique.target_model == "all"
        assert "Response from m1" in critique.updated_position
        assert critique.tokens_used == 300
        assert critique.cost_usd == 0.005

    @pytest.mark.asyncio
    async def test_get_critique_error(self) -> None:
        engine = MockCompletionEngine(fail_on_model="bad-model")
        debate = MultiModelDebate(engine, models=["bad-model", "m2"])
        all_positions = [
            DebatePosition("bad-model", "Pos A", 300, 0.005, 1),
            DebatePosition("m2", "Pos B", 300, 0.005, 1),
        ]
        others = [all_positions[1]]

        critique = await debate._get_critique("bad-model", "Q?", all_positions, others)

        assert critique.tokens_used == 0
        assert critique.cost_usd == 0.0
        assert "Error:" in critique.updated_position

    # --- _synthesize ---

    @pytest.mark.asyncio
    async def test_synthesize_success(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        session = DebateSession(question="Q?", models=["m1", "m2"])
        session.round1_positions = [
            DebatePosition("m1", "Position A", 300, 0.005, 1),
            DebatePosition("m2", "Position B", 300, 0.005, 1),
        ]

        synth = await debate._synthesize("Q?", session)

        assert synth is not None
        assert isinstance(synth, DebateSynthesis)
        assert synth.synthesizer_model == "m1"
        assert synth.tokens_used == 300
        assert synth.cost_usd == 0.005
        assert synth.confidence == 0.7

    @pytest.mark.asyncio
    async def test_synthesize_with_critiques(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        session = DebateSession(question="Q?", models=["m1", "m2"])
        session.round1_positions = [
            DebatePosition("m1", "Position A", 300, 0.005, 1),
        ]
        session.round2_critiques = [
            DebateCritique("m2", "all", "", "", "", "Critique text", 400, 0.008),
        ]

        synth = await debate._synthesize("Q?", session)
        assert synth is not None

    @pytest.mark.asyncio
    async def test_synthesize_error_returns_none(self) -> None:
        engine = MockCompletionEngine(fail_on_model="m1")
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        session = DebateSession(question="Q?", models=["m1", "m2"])
        session.round1_positions = [
            DebatePosition("m1", "Pos A", 300, 0.005, 1),
        ]

        synth = await debate._synthesize("Q?", session)
        assert synth is None

    # --- Error handling in full debate ---

    @pytest.mark.asyncio
    async def test_debate_partial_position_failure(self) -> None:
        """If one model fails in R1, the rest should still work."""
        engine = MockCompletionEngine(fail_on_model="bad-model")
        debate = MultiModelDebate(engine, models=["bad-model", "good-model"])
        session = await debate.debate("Test?")

        # Both positions returned (one with error content)
        assert len(session.round1_positions) == 2
        error_pos = next(p for p in session.round1_positions if p.model == "bad-model")
        assert "Error:" in error_pos.content
        good_pos = next(p for p in session.round1_positions if p.model == "good-model")
        assert "Response from good-model" in good_pos.content

    @pytest.mark.asyncio
    async def test_debate_all_positions_have_correct_models(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2", "m3"])
        session = await debate.debate("Multi-model test?")

        r1_models = {p.model for p in session.round1_positions}
        assert r1_models == {"m1", "m2", "m3"}

    # --- save_debate ---

    def test_save_debate_creates_file(self, tmp_path: Path) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        session = DebateSession(
            question="Save test?",
            models=["m1", "m2"],
            created_at="2025-06-15T10:30:00+00:00",
        )
        session.round1_positions = [
            DebatePosition("m1", "Pos A", 300, 0.005, 1),
        ]
        session.synthesis = DebateSynthesis(
            consensus_points=["A"],
            disagreements=[],
            recommendation="Do A",
            confidence=0.9,
            what_each_missed={},
            synthesizer_model="m1",
            tokens_used=100,
            cost_usd=0.01,
        )

        path = debate.save_debate(session, tmp_path / "debates")
        assert path.exists()
        assert path.suffix == ".json"

        data = json.loads(path.read_text())
        assert data["question"] == "Save test?"
        assert len(data["round1_positions"]) == 1

    def test_save_debate_creates_directory(self, tmp_path: Path) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        session = DebateSession(
            question="Dir test?",
            models=["m1", "m2"],
            created_at="2025-01-01T00:00:00+00:00",
        )

        save_dir = tmp_path / "nested" / "debates"
        assert not save_dir.exists()
        path = debate.save_debate(session, save_dir)
        assert save_dir.exists()
        assert path.exists()

    def test_save_debate_no_created_at(self, tmp_path: Path) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        session = DebateSession(
            question="No timestamp?",
            models=["m1", "m2"],
            created_at="",
        )

        path = debate.save_debate(session, tmp_path / "debates")
        assert path.exists()
        assert "unknown" in path.name

    # --- Engine call verification ---

    @pytest.mark.asyncio
    async def test_engine_receives_correct_messages_r1(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        await debate._get_position("m1", "What is 2+2?")

        assert len(engine.calls) == 1
        call = engine.calls[0]
        assert call["model"] == "m1"
        assert call["messages"][0]["role"] == "system"
        assert call["messages"][1]["role"] == "user"
        assert "2+2" in call["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_engine_receives_correct_messages_r2(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        positions = [
            DebatePosition("m1", "Answer A", 300, 0.005, 1),
            DebatePosition("m2", "Answer B", 300, 0.005, 1),
        ]
        await debate._get_critique("m1", "What is best?", positions, [positions[1]])

        assert len(engine.calls) == 1
        call = engine.calls[0]
        assert call["model"] == "m1"
        assert "Round 2" in call["messages"][0]["content"]
        assert "Answer A" in call["messages"][1]["content"]
        assert "Answer B" in call["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_cost_tracking_across_rounds(self) -> None:
        """Verify that total_cost and total_tokens accumulate correctly."""
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        session = await debate.debate("Cost tracking test?")

        # R1: 2 positions * (0.005 each) = 0.01
        # R2: 2 critiques * (0.005 each) = 0.01
        # R3: 1 synthesis * 0.005 = 0.005
        # Total = 0.025
        assert session.total_cost == pytest.approx(0.025)
        # Each call: 300 tokens, 5 calls total = 1500
        assert session.total_tokens == 1500

    @pytest.mark.asyncio
    async def test_cost_tracking_quick_mode(self) -> None:
        engine = MockCompletionEngine()
        debate = MultiModelDebate(engine, models=["m1", "m2"])
        session = await debate.debate("Quick cost test?", quick=True)

        # R1: 2 * 0.005 = 0.01
        # R3: 1 * 0.005 = 0.005
        # Total = 0.015
        assert session.total_cost == pytest.approx(0.015)
        assert session.total_tokens == 900  # 3 calls * 300
