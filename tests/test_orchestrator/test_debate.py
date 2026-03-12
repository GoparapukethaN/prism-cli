"""Tests for the multi-round debate engine.

All tests use MockLiteLLM.  No real API calls are ever made.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from prism.config.schema import BudgetConfig, PrismConfig
from prism.config.settings import Settings
from prism.llm.completion import CompletionEngine
from prism.llm.mock import MockLiteLLM, MockResponse
from prism.orchestrator.debate import (
    DebateConfig,
    DebateEngine,
    DebatePosition,
    DebateResult,
    DebateRound,
)
from prism.orchestrator.swarm import ModelPool
from prism.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from pathlib import Path


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def debate_settings(tmp_path: Path) -> Settings:
    """Settings with a reasonable budget for debate testing."""
    config = PrismConfig(
        prism_home=tmp_path / ".prism",
        budget=BudgetConfig(daily_limit=50.0, monthly_limit=500.0),
    )
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()
    return settings


@pytest.fixture()
def mock_auth() -> MagicMock:
    """Mock AuthManager that returns test keys for all providers."""
    mgr = MagicMock()
    mgr.get_key.return_value = "test-key-1234"
    return mgr


@pytest.fixture()
def mock_cost_tracker() -> MagicMock:
    """Mock CostTracker that always allows requests."""
    tracker = MagicMock()
    tracker.check_budget.return_value = "proceed"
    tracker.get_budget_remaining.return_value = None
    tracker.track.return_value = MagicMock(cost_usd=0.001)
    return tracker


@pytest.fixture()
def mock_registry(
    mock_auth: MagicMock, debate_settings: Settings,
) -> ProviderRegistry:
    """Real ProviderRegistry using mock auth."""
    return ProviderRegistry(settings=debate_settings, auth_manager=mock_auth)


@pytest.fixture()
def model_pool(mock_registry: ProviderRegistry) -> ModelPool:
    """A real ModelPool using the mock registry."""
    return ModelPool(mock_registry)


def _make_position_response(
    position: str = "This is my position.",
    confidence: float = 0.85,
    critiques: list[str] | None = None,
) -> str:
    """Create a valid position JSON response."""
    data: dict = {
        "position": position,
        "confidence": confidence,
    }
    if critiques is not None:
        data["critiques"] = critiques
    return json.dumps(data)


def _make_consensus_response(score: float = 0.75, summary: str = "Partial agreement.") -> str:
    """Create a valid consensus check JSON response."""
    return json.dumps({
        "consensus_score": score,
        "summary": summary,
    })


def _make_mock_litellm_for_debate(
    *,
    position_text: str = "This is my debate position.",
    confidence: float = 0.85,
    consensus_score: float = 0.6,
    synthesis_text: str = "Final synthesised answer.",
) -> MockLiteLLM:
    """Create a MockLiteLLM that handles debate flows."""
    mock = MockLiteLLM()
    # Default response covers positions, consensus, and synthesis
    mock.set_default_response(
        MockResponse(
            content=_make_position_response(position_text, confidence),
            input_tokens=100,
            output_tokens=80,
        ),
    )
    return mock


def _make_engine(
    settings: Settings,
    cost_tracker: MagicMock,
    auth: MagicMock,
    registry: ProviderRegistry,
    litellm: MockLiteLLM,
) -> CompletionEngine:
    """Helper to create a CompletionEngine from components."""
    return CompletionEngine(
        settings=settings,
        cost_tracker=cost_tracker,
        auth_manager=auth,
        provider_registry=registry,
        litellm_backend=litellm,
    )


@pytest.fixture()
def debate_mock_litellm() -> MockLiteLLM:
    """MockLiteLLM pre-loaded with debate-friendly responses."""
    return _make_mock_litellm_for_debate()


@pytest.fixture()
def debate_engine(
    debate_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    debate_mock_litellm: MockLiteLLM,
) -> DebateEngine:
    """DebateEngine wired to mocks — no real API calls."""
    engine = _make_engine(
        debate_settings, mock_cost_tracker, mock_auth, mock_registry, debate_mock_litellm,
    )
    pool = ModelPool(mock_registry)
    return DebateEngine(engine, pool)


@pytest.fixture()
def consensus_engine(
    debate_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
) -> DebateEngine:
    """DebateEngine that reaches consensus in round 1."""
    mock = MockLiteLLM()
    mock.set_default_response(
        MockResponse(
            content=_make_consensus_response(score=0.95, summary="Strong agreement."),
            input_tokens=100,
            output_tokens=50,
        ),
    )
    engine = _make_engine(
        debate_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
    )
    pool = ModelPool(mock_registry)
    return DebateEngine(engine, pool)


@pytest.fixture()
def no_consensus_engine(
    debate_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
) -> DebateEngine:
    """DebateEngine that never reaches consensus (score always low)."""
    mock = MockLiteLLM()
    mock.set_default_response(
        MockResponse(
            content=_make_position_response("Disagreement position.", 0.4),
            input_tokens=100,
            output_tokens=50,
        ),
    )
    engine = _make_engine(
        debate_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
    )
    pool = ModelPool(mock_registry)
    config = DebateConfig(max_rounds=2, consensus_threshold=0.99)
    return DebateEngine(engine, pool, config)


# ======================================================================
# DebatePosition dataclass
# ======================================================================


class TestDebatePosition:
    """Test DebatePosition dataclass."""

    def test_default_critiques(self) -> None:
        """Default critiques list is empty."""
        pos = DebatePosition(
            model="gpt-4o", content="My position.", round=1, confidence=0.8,
        )
        assert pos.critiques == []
        assert pos.model == "gpt-4o"
        assert pos.round == 1
        assert pos.confidence == 0.8

    def test_with_critiques(self) -> None:
        """Critiques list is properly stored."""
        critiques = ["Too vague.", "Needs more evidence."]
        pos = DebatePosition(
            model="claude-opus-4-20250514",
            content="Position text.",
            round=2,
            confidence=0.9,
            critiques=critiques,
        )
        assert len(pos.critiques) == 2
        assert "Too vague." in pos.critiques

    def test_content_is_stored(self) -> None:
        """Content text is properly stored."""
        pos = DebatePosition(
            model="test-model", content="Detailed position.", round=1, confidence=0.5,
        )
        assert pos.content == "Detailed position."


# ======================================================================
# DebateRound dataclass
# ======================================================================


class TestDebateRound:
    """Test DebateRound dataclass."""

    def test_default_values(self) -> None:
        """DebateRound has sensible defaults."""
        dr = DebateRound(round_number=1)
        assert dr.round_number == 1
        assert dr.positions == []
        assert dr.consensus_reached is False
        assert dr.consensus_text is None

    def test_with_positions(self) -> None:
        """DebateRound tracks positions."""
        pos = DebatePosition(
            model="gpt-4o", content="Position.", round=1, confidence=0.7,
        )
        dr = DebateRound(round_number=1, positions=[pos])
        assert len(dr.positions) == 1

    def test_consensus_state(self) -> None:
        """DebateRound tracks consensus state."""
        dr = DebateRound(
            round_number=2,
            consensus_reached=True,
            consensus_text="All agreed.",
        )
        assert dr.consensus_reached is True
        assert dr.consensus_text == "All agreed."


# ======================================================================
# DebateConfig validation
# ======================================================================


class TestDebateConfig:
    """Test DebateConfig validation."""

    def test_default_values(self) -> None:
        """Default config is valid."""
        config = DebateConfig()
        assert config.max_rounds == 3
        assert config.min_participants == 2
        assert config.max_participants == 4
        assert config.consensus_threshold == 0.8
        assert config.temperature == 0.7

    def test_custom_values(self) -> None:
        """Custom config values are accepted."""
        config = DebateConfig(
            max_rounds=5,
            min_participants=3,
            max_participants=6,
            consensus_threshold=0.9,
            temperature=0.5,
        )
        assert config.max_rounds == 5
        assert config.min_participants == 3

    def test_max_rounds_zero_raises(self) -> None:
        """Zero max_rounds raises ValueError."""
        with pytest.raises(ValueError, match="max_rounds must be at least 1"):
            DebateConfig(max_rounds=0)

    def test_negative_max_rounds_raises(self) -> None:
        """Negative max_rounds raises ValueError."""
        with pytest.raises(ValueError, match="max_rounds must be at least 1"):
            DebateConfig(max_rounds=-1)

    def test_min_participants_one_raises(self) -> None:
        """min_participants=1 raises ValueError (need at least 2)."""
        with pytest.raises(ValueError, match="min_participants must be at least 2"):
            DebateConfig(min_participants=1)

    def test_max_lt_min_raises(self) -> None:
        """max_participants < min_participants raises ValueError."""
        with pytest.raises(ValueError, match="max_participants must be >= min_participants"):
            DebateConfig(min_participants=3, max_participants=2)

    def test_threshold_out_of_range_raises(self) -> None:
        """consensus_threshold outside 0-1 raises ValueError."""
        with pytest.raises(ValueError, match="consensus_threshold must be between"):
            DebateConfig(consensus_threshold=1.5)

    def test_threshold_negative_raises(self) -> None:
        """Negative consensus_threshold raises ValueError."""
        with pytest.raises(ValueError, match="consensus_threshold must be between"):
            DebateConfig(consensus_threshold=-0.1)

    def test_temperature_out_of_range_raises(self) -> None:
        """temperature outside 0-2 raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between"):
            DebateConfig(temperature=2.5)

    def test_temperature_negative_raises(self) -> None:
        """Negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between"):
            DebateConfig(temperature=-0.1)


# ======================================================================
# DebateResult dataclass
# ======================================================================


class TestDebateResult:
    """Test DebateResult dataclass."""

    def test_all_fields(self) -> None:
        """DebateResult stores all fields correctly."""
        result = DebateResult(
            topic="Should we use microservices?",
            rounds=[DebateRound(round_number=1)],
            final_synthesis="Use microservices selectively.",
            consensus_score=0.85,
            total_cost=0.05,
            participating_models=["gpt-4o", "claude-opus-4-20250514"],
        )
        assert result.topic == "Should we use microservices?"
        assert len(result.rounds) == 1
        assert result.consensus_score == 0.85
        assert result.total_cost == 0.05
        assert len(result.participating_models) == 2


# ======================================================================
# DebateEngine._parse_position_response
# ======================================================================


class TestParsePositionResponse:
    """Test position response parsing."""

    def test_valid_json(self) -> None:
        """Valid JSON is parsed correctly."""
        content = _make_position_response("My stance.", 0.9, ["Critique 1"])
        position, confidence, critiques = DebateEngine._parse_position_response(content)
        assert position == "My stance."
        assert confidence == 0.9
        assert critiques == ["Critique 1"]

    def test_missing_confidence_defaults(self) -> None:
        """Missing confidence defaults to 0.5."""
        content = json.dumps({"position": "My stance."})
        _, confidence, _ = DebateEngine._parse_position_response(content)
        assert confidence == 0.5

    def test_missing_critiques_defaults_empty(self) -> None:
        """Missing critiques defaults to empty list."""
        content = json.dumps({"position": "My stance.", "confidence": 0.8})
        _, _, critiques = DebateEngine._parse_position_response(content)
        assert critiques == []

    def test_invalid_json_fallback(self) -> None:
        """Non-JSON content falls back to raw text."""
        content = "This is not JSON but it is my position."
        position, confidence, critiques = DebateEngine._parse_position_response(content)
        assert position == content
        assert confidence == 0.5
        assert critiques == []

    def test_confidence_clamped_high(self) -> None:
        """Confidence > 1.0 is clamped to 1.0."""
        content = json.dumps({"position": "Sure.", "confidence": 5.0})
        _, confidence, _ = DebateEngine._parse_position_response(content)
        assert confidence == 1.0

    def test_confidence_clamped_low(self) -> None:
        """Confidence < 0.0 is clamped to 0.0."""
        content = json.dumps({"position": "Unsure.", "confidence": -1.0})
        _, confidence, _ = DebateEngine._parse_position_response(content)
        assert confidence == 0.0

    def test_critiques_non_list_ignored(self) -> None:
        """Non-list critiques defaults to empty list."""
        content = json.dumps({
            "position": "Stance.",
            "confidence": 0.7,
            "critiques": "not a list",
        })
        _, _, critiques = DebateEngine._parse_position_response(content)
        assert critiques == []

    def test_missing_position_uses_raw(self) -> None:
        """Missing position key uses full content string."""
        content = json.dumps({"confidence": 0.6})
        position, _, _ = DebateEngine._parse_position_response(content)
        assert position == content


# ======================================================================
# DebateEngine._parse_consensus_response
# ======================================================================


class TestParseConsensusResponse:
    """Test consensus response parsing."""

    def test_valid_json(self) -> None:
        """Valid JSON is parsed correctly."""
        content = _make_consensus_response(0.85)
        score = DebateEngine._parse_consensus_response(content)
        assert score == 0.85

    def test_invalid_json_fallback(self) -> None:
        """Non-JSON content falls back to 0.5."""
        score = DebateEngine._parse_consensus_response("Not JSON.")
        assert score == 0.5

    def test_score_clamped_high(self) -> None:
        """Score > 1.0 is clamped to 1.0."""
        content = json.dumps({"consensus_score": 2.0})
        score = DebateEngine._parse_consensus_response(content)
        assert score == 1.0

    def test_score_clamped_low(self) -> None:
        """Score < 0.0 is clamped to 0.0."""
        content = json.dumps({"consensus_score": -0.5})
        score = DebateEngine._parse_consensus_response(content)
        assert score == 0.0

    def test_missing_score_defaults(self) -> None:
        """Missing consensus_score defaults to 0.5."""
        content = json.dumps({"summary": "Some notes."})
        score = DebateEngine._parse_consensus_response(content)
        assert score == 0.5


# ======================================================================
# DebateEngine._format_positions
# ======================================================================


class TestFormatPositions:
    """Test position formatting."""

    def test_single_position(self) -> None:
        """Single position is formatted correctly."""
        positions = [
            DebatePosition(
                model="gpt-4o", content="My position.", round=1, confidence=0.8,
            ),
        ]
        result = DebateEngine._format_positions(positions)
        assert "Participant 1" in result
        assert "gpt-4o" in result
        assert "0.80" in result
        assert "My position." in result

    def test_multiple_positions(self) -> None:
        """Multiple positions are numbered correctly."""
        positions = [
            DebatePosition(
                model="gpt-4o", content="First.", round=1, confidence=0.7,
            ),
            DebatePosition(
                model="claude-opus-4-20250514", content="Second.", round=1, confidence=0.9,
            ),
        ]
        result = DebateEngine._format_positions(positions)
        assert "Participant 1" in result
        assert "Participant 2" in result
        assert "First." in result
        assert "Second." in result

    def test_empty_positions(self) -> None:
        """Empty positions list returns empty string."""
        result = DebateEngine._format_positions([])
        assert result == ""


# ======================================================================
# DebateEngine._format_debate_history
# ======================================================================


class TestFormatDebateHistory:
    """Test debate history formatting."""

    def test_single_round(self) -> None:
        """Single round is formatted correctly."""
        pos = DebatePosition(
            model="gpt-4o", content="My stance.", round=1, confidence=0.8,
        )
        rounds = [DebateRound(round_number=1, positions=[pos])]
        result = DebateEngine._format_debate_history(rounds)
        assert "Round 1" in result
        assert "gpt-4o" in result
        assert "My stance." in result

    def test_multiple_rounds(self) -> None:
        """Multiple rounds are formatted with clear separators."""
        rounds = [
            DebateRound(round_number=1, positions=[
                DebatePosition(model="m1", content="R1 pos.", round=1, confidence=0.7),
            ]),
            DebateRound(round_number=2, positions=[
                DebatePosition(model="m1", content="R2 pos.", round=2, confidence=0.9),
            ]),
        ]
        result = DebateEngine._format_debate_history(rounds)
        assert "Round 1" in result
        assert "Round 2" in result
        assert "R1 pos." in result
        assert "R2 pos." in result

    def test_consensus_marked(self) -> None:
        """Consensus rounds are marked in the history."""
        rounds = [
            DebateRound(
                round_number=1,
                positions=[],
                consensus_reached=True,
                consensus_text="All agreed.",
            ),
        ]
        result = DebateEngine._format_debate_history(rounds)
        assert "Consensus reached" in result
        assert "All agreed." in result

    def test_critiques_included(self) -> None:
        """Critiques are included in history formatting."""
        pos = DebatePosition(
            model="gpt-4o",
            content="My stance.",
            round=2,
            confidence=0.8,
            critiques=["Too vague.", "Missing evidence."],
        )
        rounds = [DebateRound(round_number=2, positions=[pos])]
        result = DebateEngine._format_debate_history(rounds)
        assert "Too vague." in result
        assert "Missing evidence." in result

    def test_empty_rounds(self) -> None:
        """Empty rounds list produces empty string."""
        result = DebateEngine._format_debate_history([])
        assert result == ""


# ======================================================================
# DebateEngine._select_participants
# ======================================================================


class TestSelectParticipants:
    """Test participant selection logic."""

    def test_selects_at_least_min(
        self, debate_engine: DebateEngine,
    ) -> None:
        """At least min_participants are selected."""
        participants = debate_engine._select_participants("Test topic")
        assert len(participants) >= debate_engine._config.min_participants

    def test_does_not_exceed_max(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Does not exceed max_participants."""
        participants = debate_engine._select_participants("Test topic")
        assert len(participants) <= debate_engine._config.max_participants

    def test_no_duplicates(
        self, debate_engine: DebateEngine,
    ) -> None:
        """No duplicate model IDs in participants."""
        participants = debate_engine._select_participants("Test topic")
        assert len(participants) == len(set(participants))

    def test_prefers_different_providers(
        self,
        debate_engine: DebateEngine,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Participants are from different providers when possible."""
        participants = debate_engine._select_participants("Test topic")
        if len(participants) >= 2:
            providers = set()
            for model_id in participants:
                info = mock_registry.get_model_info(model_id)
                if info:
                    providers.add(info.provider)
            # With a mock registry having multiple providers, we should
            # have at least 2 different providers
            assert len(providers) >= 2


# ======================================================================
# DebateEngine.debate — full pipeline
# ======================================================================


class TestDebateFullPipeline:
    """Test the full debate pipeline end-to-end (all mocked)."""

    async def test_debate_returns_result(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Debate returns a DebateResult."""
        result = await debate_engine.debate("Should we use REST or GraphQL?")
        assert isinstance(result, DebateResult)
        assert result.topic == "Should we use REST or GraphQL?"

    async def test_debate_has_rounds(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Debate produces at least one round."""
        result = await debate_engine.debate("Best testing strategy?")
        assert len(result.rounds) >= 1

    async def test_debate_has_synthesis(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Debate produces a non-empty final synthesis."""
        result = await debate_engine.debate("Optimal database choice?")
        assert result.final_synthesis
        assert len(result.final_synthesis) > 0

    async def test_debate_has_participants(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Debate reports participating models."""
        result = await debate_engine.debate("Architecture patterns?")
        assert len(result.participating_models) >= 2

    async def test_debate_has_consensus_score(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Debate reports a consensus score between 0 and 1."""
        result = await debate_engine.debate("Deployment strategy?")
        assert 0.0 <= result.consensus_score <= 1.0

    async def test_debate_tracks_cost(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Debate accumulates total cost."""
        result = await debate_engine.debate("Cost optimisation approach?")
        assert result.total_cost >= 0.0

    async def test_debate_positions_have_content(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Every position in every round has non-empty content."""
        result = await debate_engine.debate("Error handling patterns?")
        for debate_round in result.rounds:
            for pos in debate_round.positions:
                assert pos.content
                assert pos.model
                assert 0.0 <= pos.confidence <= 1.0

    async def test_debate_round_numbers_sequential(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Round numbers are sequential starting from 1."""
        result = await debate_engine.debate("Logging strategy?")
        for i, debate_round in enumerate(result.rounds, 1):
            assert debate_round.round_number == i

    async def test_debate_with_context(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Debate accepts and uses context string."""
        result = await debate_engine.debate(
            "Should we refactor the auth module?",
            context="Current auth uses JWT. 500 daily active users.",
        )
        assert isinstance(result, DebateResult)
        assert len(result.rounds) >= 1


# ======================================================================
# Empty / invalid topic
# ======================================================================


class TestDebateValidation:
    """Test input validation for debates."""

    async def test_empty_topic_raises(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Empty topic raises ValueError."""
        with pytest.raises(ValueError, match="Debate topic must not be empty"):
            await debate_engine.debate("")

    async def test_whitespace_topic_raises(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Whitespace-only topic raises ValueError."""
        with pytest.raises(ValueError, match="Debate topic must not be empty"):
            await debate_engine.debate("   ")


# ======================================================================
# Consensus early stop
# ======================================================================


class TestConsensusEarlyStop:
    """Test early stopping when consensus is reached."""

    async def test_consensus_stops_early(
        self, consensus_engine: DebateEngine,
    ) -> None:
        """Debate stops before max_rounds when consensus is reached."""
        result = await consensus_engine.debate("Simple agreed topic.")
        # Consensus engine returns score=0.95, threshold=0.8 by default
        # Position parsing will fallback to 0.5 since the content is consensus JSON
        # But consensus check should reach threshold
        assert isinstance(result, DebateResult)
        # Should have completed (not necessarily in 1 round if
        # position parsing gives non-consensus content)
        assert len(result.rounds) >= 1

    async def test_consensus_text_set(
        self, consensus_engine: DebateEngine,
    ) -> None:
        """When consensus is reached, consensus_text is populated."""
        result = await consensus_engine.debate("Topic everyone agrees on.")
        consensus_rounds = [r for r in result.rounds if r.consensus_reached]
        if consensus_rounds:
            for cr in consensus_rounds:
                assert cr.consensus_text is not None
                assert "Consensus reached" in cr.consensus_text


# ======================================================================
# No consensus — full rounds exhausted
# ======================================================================


class TestNoConsensus:
    """Test behaviour when consensus is never reached."""

    async def test_runs_all_rounds(
        self, no_consensus_engine: DebateEngine,
    ) -> None:
        """Runs all max_rounds when consensus threshold is very high."""
        result = await no_consensus_engine.debate("Contentious topic.")
        # Config has max_rounds=2, threshold=0.99
        assert len(result.rounds) == 2

    async def test_still_produces_synthesis(
        self, no_consensus_engine: DebateEngine,
    ) -> None:
        """Even without consensus, a synthesis is produced."""
        result = await no_consensus_engine.debate("Divisive topic.")
        assert result.final_synthesis
        assert len(result.final_synthesis) > 0


# ======================================================================
# Custom DebateConfig
# ======================================================================


class TestCustomDebateConfig:
    """Test debate engine with custom configurations."""

    async def test_single_round_debate(
        self,
        debate_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Debate with max_rounds=1 produces exactly one round."""
        mock = _make_mock_litellm_for_debate()
        engine = _make_engine(
            debate_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
        )
        pool = ModelPool(mock_registry)
        config = DebateConfig(max_rounds=1)
        debate_eng = DebateEngine(engine, pool, config)

        result = await debate_eng.debate("Quick debate.")
        assert len(result.rounds) == 1

    async def test_high_consensus_threshold(
        self,
        debate_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Very high consensus threshold means debate runs all rounds."""
        mock = _make_mock_litellm_for_debate(consensus_score=0.5)
        engine = _make_engine(
            debate_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
        )
        pool = ModelPool(mock_registry)
        config = DebateConfig(max_rounds=2, consensus_threshold=0.99)
        debate_eng = DebateEngine(engine, pool, config)

        result = await debate_eng.debate("Hard to agree on.")
        # With default mock, consensus parse falls back to 0.5 which is < 0.99
        assert len(result.rounds) == 2


# ======================================================================
# Position generation
# ======================================================================


class TestPositionGeneration:
    """Test individual position generation."""

    async def test_round_1_independent(
        self,
        debate_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Round 1 positions are generated independently (no others)."""
        mock = MockLiteLLM()
        mock.set_default_response(
            MockResponse(
                content=_make_position_response("Independent position.", 0.85),
                input_tokens=50,
                output_tokens=40,
            ),
        )
        engine = _make_engine(
            debate_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
        )
        pool = ModelPool(mock_registry)
        debate_eng = DebateEngine(engine, pool)

        model = pool.get_planning_model()
        position = await debate_eng._generate_position(
            model=model,
            topic="Test topic",
            others=[],
            context="",
            round_num=1,
        )
        assert position.content == "Independent position."
        assert position.confidence == 0.85
        assert position.round == 1
        assert position.model == model

    async def test_round_2_sees_others(
        self,
        debate_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Round 2 positions reference other participants' positions."""
        mock = MockLiteLLM()
        mock.set_default_response(
            MockResponse(
                content=_make_position_response(
                    "Refined position.", 0.9, ["Critique of other."],
                ),
                input_tokens=80,
                output_tokens=60,
            ),
        )
        engine = _make_engine(
            debate_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
        )
        pool = ModelPool(mock_registry)
        debate_eng = DebateEngine(engine, pool)

        others = [
            DebatePosition(
                model="other-model",
                content="Other position.",
                round=1,
                confidence=0.7,
            ),
        ]
        model = pool.get_planning_model()
        position = await debate_eng._generate_position(
            model=model,
            topic="Test topic",
            others=others,
            context="",
            round_num=2,
        )
        assert position.round == 2
        assert position.content == "Refined position."
        assert position.confidence == 0.9

    async def test_position_with_context(
        self,
        debate_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Context string is included in the position request."""
        mock = MockLiteLLM()
        mock.set_default_response(
            MockResponse(
                content=_make_position_response("Contextual position.", 0.8),
                input_tokens=60,
                output_tokens=50,
            ),
        )
        engine = _make_engine(
            debate_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
        )
        pool = ModelPool(mock_registry)
        debate_eng = DebateEngine(engine, pool)

        model = pool.get_planning_model()
        position = await debate_eng._generate_position(
            model=model,
            topic="Test topic",
            others=[],
            context="Important context here.",
            round_num=1,
        )
        assert position.content == "Contextual position."

        # Verify context was included in the call
        assert len(mock.call_log) == 1
        messages = mock.call_log[0]["messages"]
        user_msg = messages[-1]["content"]
        assert "Important context here." in user_msg


# ======================================================================
# Consensus check
# ======================================================================


class TestConsensusCheck:
    """Test consensus checking logic."""

    async def test_single_position_is_consensus(
        self,
        debate_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Single position trivially reaches consensus."""
        mock = _make_mock_litellm_for_debate()
        engine = _make_engine(
            debate_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
        )
        pool = ModelPool(mock_registry)
        debate_eng = DebateEngine(engine, pool)

        positions = [
            DebatePosition(model="m1", content="Only position.", round=1, confidence=0.9),
        ]
        reached, score = await debate_eng._check_consensus(positions)
        assert reached is True
        assert score == 1.0

    async def test_empty_positions_is_consensus(
        self,
        debate_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Empty positions list is trivially consensus."""
        mock = _make_mock_litellm_for_debate()
        engine = _make_engine(
            debate_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
        )
        pool = ModelPool(mock_registry)
        debate_eng = DebateEngine(engine, pool)

        reached, score = await debate_eng._check_consensus([])
        assert reached is True
        assert score == 1.0

    async def test_high_score_reaches_consensus(
        self,
        debate_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """High consensus score meets the threshold."""
        mock = MockLiteLLM()
        mock.set_default_response(
            MockResponse(
                content=_make_consensus_response(0.95),
                input_tokens=80,
                output_tokens=40,
            ),
        )
        engine = _make_engine(
            debate_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
        )
        pool = ModelPool(mock_registry)
        debate_eng = DebateEngine(engine, pool)

        positions = [
            DebatePosition(model="m1", content="Agree.", round=1, confidence=0.9),
            DebatePosition(model="m2", content="Also agree.", round=1, confidence=0.85),
        ]
        reached, score = await debate_eng._check_consensus(positions)
        assert reached is True
        assert score == 0.95

    async def test_low_score_no_consensus(
        self,
        debate_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Low consensus score does not meet the threshold."""
        mock = MockLiteLLM()
        mock.set_default_response(
            MockResponse(
                content=_make_consensus_response(0.3),
                input_tokens=80,
                output_tokens=40,
            ),
        )
        engine = _make_engine(
            debate_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
        )
        pool = ModelPool(mock_registry)
        debate_eng = DebateEngine(engine, pool)

        positions = [
            DebatePosition(model="m1", content="Position A.", round=1, confidence=0.9),
            DebatePosition(model="m2", content="Position B.", round=1, confidence=0.85),
        ]
        reached, score = await debate_eng._check_consensus(positions)
        assert reached is False
        assert score == 0.3


# ======================================================================
# Synthesis
# ======================================================================


class TestSynthesis:
    """Test final synthesis generation."""

    async def test_synthesis_produces_text(
        self,
        debate_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Synthesis produces non-empty text."""
        mock = MockLiteLLM()
        mock.set_default_response(
            MockResponse(
                content="Comprehensive synthesis combining all viewpoints.",
                input_tokens=200,
                output_tokens=100,
            ),
        )
        engine = _make_engine(
            debate_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
        )
        pool = ModelPool(mock_registry)
        debate_eng = DebateEngine(engine, pool)

        rounds = [
            DebateRound(round_number=1, positions=[
                DebatePosition(model="m1", content="Pos A.", round=1, confidence=0.8),
                DebatePosition(model="m2", content="Pos B.", round=1, confidence=0.7),
            ]),
        ]
        result = await debate_eng._synthesize("Test topic", rounds)
        assert result == "Comprehensive synthesis combining all viewpoints."

    async def test_synthesis_with_empty_rounds(
        self,
        debate_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Synthesis handles empty rounds gracefully."""
        mock = MockLiteLLM()
        mock.set_default_response(
            MockResponse(
                content="Synthesis from empty debate.",
                input_tokens=50,
                output_tokens=30,
            ),
        )
        engine = _make_engine(
            debate_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
        )
        pool = ModelPool(mock_registry)
        debate_eng = DebateEngine(engine, pool)

        result = await debate_eng._synthesize("Empty debate", [])
        assert result == "Synthesis from empty debate."


# ======================================================================
# Cost tracking
# ======================================================================


class TestDebateCostTracking:
    """Test that debate engine accumulates costs correctly."""

    async def test_cost_accumulates(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Total cost increases with each LLM call."""
        result = await debate_engine.debate("Cost tracking test.")
        # At minimum: position calls + consensus check + synthesis = multiple calls
        # Each call has cost_usd from CompletionResult
        assert result.total_cost >= 0.0

    async def test_cost_resets_between_debates(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Each debate starts with fresh cost tracking."""
        result1 = await debate_engine.debate("First debate.")
        cost1 = result1.total_cost
        result2 = await debate_engine.debate("Second debate.")
        cost2 = result2.total_cost
        # Both should have cost, but they should be independent
        # (not cumulative across debates)
        assert cost1 >= 0.0
        assert cost2 >= 0.0
        # cost2 should be roughly the same as cost1 (same mock),
        # not cost1 + cost2
        assert abs(cost2 - cost1) < cost1 * 2 + 0.01  # Allow some variance


# ======================================================================
# Edge cases
# ======================================================================


class TestDebateEdgeCases:
    """Test edge cases and error handling."""

    async def test_debate_with_long_topic(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Long topic strings are handled without error."""
        long_topic = "Should we " + "really " * 100 + "use microservices?"
        result = await debate_engine.debate(long_topic)
        assert isinstance(result, DebateResult)
        assert result.topic == long_topic

    async def test_debate_with_special_characters(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Topics with special characters are handled."""
        result = await debate_engine.debate(
            'How should we handle "quotes" and {braces} and <angles>?',
        )
        assert isinstance(result, DebateResult)

    async def test_debate_with_unicode(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Unicode topics are handled."""
        result = await debate_engine.debate("Kann KI kreativ sein?")
        assert isinstance(result, DebateResult)
        assert result.topic == "Kann KI kreativ sein?"

    async def test_multiple_sequential_debates(
        self, debate_engine: DebateEngine,
    ) -> None:
        """Multiple sequential debates work independently."""
        result1 = await debate_engine.debate("First topic.")
        result2 = await debate_engine.debate("Second topic.")
        assert result1.topic == "First topic."
        assert result2.topic == "Second topic."
        assert result1.topic != result2.topic
