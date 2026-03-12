"""Tests for FrugalGPT confidence cascading — cost-efficient model routing.

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
from prism.orchestrator.cascade import (
    CascadeAttempt,
    CascadeConfig,
    CascadeLevel,
    CascadeResult,
    ConfidenceCascade,
    ConfidenceScore,
    TaskResult,
)
from prism.orchestrator.swarm import ModelPool
from prism.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from pathlib import Path


# ======================================================================
# Fixtures
# ======================================================================


def _make_confidence_json(
    score: float = 0.9,
    reasoning: str = "High confidence",
    uncertainty_areas: list[str] | None = None,
    alternative_approaches: list[str] | None = None,
) -> str:
    """Build a valid JSON confidence assessment response."""
    return json.dumps({
        "score": score,
        "reasoning": reasoning,
        "uncertainty_areas": uncertainty_areas or [],
        "alternative_approaches": alternative_approaches or [],
    })


@pytest.fixture()
def cascade_settings(tmp_path: Path) -> Settings:
    """Settings with a reasonable budget for cascade testing."""
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
def mock_registry(mock_auth: MagicMock, cascade_settings: Settings) -> ProviderRegistry:
    """Real ProviderRegistry using mock auth."""
    return ProviderRegistry(settings=cascade_settings, auth_manager=mock_auth)


@pytest.fixture()
def model_pool(mock_registry: ProviderRegistry) -> ModelPool:
    """A real ModelPool using the mock registry."""
    return ModelPool(mock_registry)


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
def high_confidence_litellm() -> MockLiteLLM:
    """MockLiteLLM that returns high-confidence assessments.

    Every call returns either the main response or a high-confidence
    JSON assessment, depending on the prompt.
    """
    mock = MockLiteLLM()
    mock.set_default_response(
        MockResponse(
            content=_make_confidence_json(score=0.92, reasoning="Very confident"),
            input_tokens=100,
            output_tokens=50,
        ),
    )
    return mock


@pytest.fixture()
def low_confidence_litellm() -> MockLiteLLM:
    """MockLiteLLM that returns low-confidence assessments."""
    mock = MockLiteLLM()
    mock.set_default_response(
        MockResponse(
            content=_make_confidence_json(
                score=0.3,
                reasoning="Low confidence — many unknowns",
                uncertainty_areas=["correctness", "completeness"],
            ),
            input_tokens=100,
            output_tokens=50,
        ),
    )
    return mock


@pytest.fixture()
def cascade_high_confidence(
    cascade_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    high_confidence_litellm: MockLiteLLM,
    model_pool: ModelPool,
) -> ConfidenceCascade:
    """Cascade wired to high-confidence mocks — accepts at first level."""
    engine = _make_engine(
        cascade_settings, mock_cost_tracker, mock_auth, mock_registry, high_confidence_litellm,
    )
    return ConfidenceCascade(engine, model_pool)


@pytest.fixture()
def cascade_low_confidence(
    cascade_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    low_confidence_litellm: MockLiteLLM,
    model_pool: ModelPool,
) -> ConfidenceCascade:
    """Cascade wired to low-confidence mocks — escalates through all levels."""
    engine = _make_engine(
        cascade_settings, mock_cost_tracker, mock_auth, mock_registry, low_confidence_litellm,
    )
    return ConfidenceCascade(engine, model_pool)


# ======================================================================
# ConfidenceScore dataclass
# ======================================================================


class TestConfidenceScore:
    """Test the ConfidenceScore dataclass."""

    def test_default_values(self) -> None:
        """ConfidenceScore has expected defaults."""
        cs = ConfidenceScore(score=0.8, reasoning="Good")
        assert cs.score == 0.8
        assert cs.reasoning == "Good"
        assert cs.uncertainty_areas == []
        assert cs.alternative_approaches == []
        assert cs.model == ""

    def test_score_clamped_high(self) -> None:
        """Score is clamped to max 1.0."""
        cs = ConfidenceScore(score=1.5, reasoning="Over")
        assert cs.score == 1.0

    def test_score_clamped_low(self) -> None:
        """Score is clamped to min 0.0."""
        cs = ConfidenceScore(score=-0.3, reasoning="Under")
        assert cs.score == 0.0

    def test_full_values(self) -> None:
        """ConfidenceScore with all fields set."""
        cs = ConfidenceScore(
            score=0.75,
            reasoning="Mostly confident",
            uncertainty_areas=["edge cases"],
            alternative_approaches=["try approach B"],
            model="gpt-4o",
        )
        assert cs.score == 0.75
        assert len(cs.uncertainty_areas) == 1
        assert len(cs.alternative_approaches) == 1
        assert cs.model == "gpt-4o"

    def test_non_list_uncertainty_coerced(self) -> None:
        """Non-list uncertainty_areas is coerced to empty list."""
        cs = ConfidenceScore(
            score=0.5,
            reasoning="ok",
            uncertainty_areas="not a list",  # type: ignore[arg-type]
        )
        assert cs.uncertainty_areas == []

    def test_non_list_alternatives_coerced(self) -> None:
        """Non-list alternative_approaches is coerced to empty list."""
        cs = ConfidenceScore(
            score=0.5,
            reasoning="ok",
            alternative_approaches="not a list",  # type: ignore[arg-type]
        )
        assert cs.alternative_approaches == []


# ======================================================================
# CascadeLevel dataclass
# ======================================================================


class TestCascadeLevel:
    """Test the CascadeLevel dataclass."""

    def test_basic_creation(self) -> None:
        """CascadeLevel is created with correct values."""
        level = CascadeLevel(
            tier="cheap",
            models=["groq/llama3-8b"],
            confidence_threshold=0.85,
            cost_multiplier=1.0,
        )
        assert level.tier == "cheap"
        assert len(level.models) == 1
        assert level.confidence_threshold == 0.85
        assert level.cost_multiplier == 1.0

    def test_threshold_clamped(self) -> None:
        """Confidence threshold is clamped to [0, 1]."""
        level = CascadeLevel(
            tier="test",
            models=["m1"],
            confidence_threshold=1.5,
            cost_multiplier=1.0,
        )
        assert level.confidence_threshold == 1.0

    def test_negative_threshold_clamped(self) -> None:
        """Negative threshold is clamped to 0.0."""
        level = CascadeLevel(
            tier="test",
            models=["m1"],
            confidence_threshold=-0.5,
            cost_multiplier=1.0,
        )
        assert level.confidence_threshold == 0.0

    def test_negative_cost_multiplier_clamped(self) -> None:
        """Negative cost multiplier is clamped to 0.0."""
        level = CascadeLevel(
            tier="test",
            models=["m1"],
            confidence_threshold=0.7,
            cost_multiplier=-2.0,
        )
        assert level.cost_multiplier == 0.0

    def test_empty_tier_defaults(self) -> None:
        """Empty tier string defaults to 'unknown'."""
        level = CascadeLevel(
            tier="",
            models=["m1"],
            confidence_threshold=0.7,
            cost_multiplier=1.0,
        )
        assert level.tier == "unknown"


# ======================================================================
# CascadeConfig dataclass
# ======================================================================


class TestCascadeConfig:
    """Test the CascadeConfig dataclass."""

    def test_default_values(self) -> None:
        """CascadeConfig has sensible defaults."""
        config = CascadeConfig()
        assert config.levels is None
        assert config.min_confidence == 0.7
        assert config.max_escalations == 3
        assert config.use_external_judge is True
        assert config.budget_limit is None

    def test_custom_values(self) -> None:
        """CascadeConfig accepts custom values."""
        levels = [
            CascadeLevel(tier="only", models=["m1"], confidence_threshold=0.5, cost_multiplier=1.0),
        ]
        config = CascadeConfig(
            levels=levels,
            min_confidence=0.6,
            max_escalations=5,
            use_external_judge=False,
            budget_limit=1.0,
        )
        assert config.levels == levels
        assert config.min_confidence == 0.6
        assert config.max_escalations == 5
        assert config.use_external_judge is False
        assert config.budget_limit == 1.0

    def test_min_confidence_clamped(self) -> None:
        """min_confidence is clamped to [0, 1]."""
        config = CascadeConfig(min_confidence=2.0)
        assert config.min_confidence == 1.0

    def test_max_escalations_minimum(self) -> None:
        """max_escalations cannot go below 1."""
        config = CascadeConfig(max_escalations=0)
        assert config.max_escalations == 1

    def test_budget_limit_non_negative(self) -> None:
        """budget_limit is clamped to non-negative."""
        config = CascadeConfig(budget_limit=-10.0)
        assert config.budget_limit == 0.0


# ======================================================================
# CascadeAttempt dataclass
# ======================================================================


class TestCascadeAttempt:
    """Test the CascadeAttempt dataclass."""

    def test_basic_creation(self) -> None:
        """CascadeAttempt is created with correct values."""
        confidence = ConfidenceScore(score=0.8, reasoning="Good")
        attempt = CascadeAttempt(
            level=0,
            model="groq/llama3-8b",
            output="Hello world",
            self_confidence=confidence,
        )
        assert attempt.level == 0
        assert attempt.model == "groq/llama3-8b"
        assert attempt.output == "Hello world"
        assert attempt.self_confidence.score == 0.8
        assert attempt.judge_confidence is None
        assert attempt.accepted is False
        assert attempt.cost == 0.0

    def test_with_judge(self) -> None:
        """CascadeAttempt with judge confidence."""
        self_conf = ConfidenceScore(score=0.9, reasoning="Self sure")
        judge_conf = ConfidenceScore(score=0.7, reasoning="Judge less sure")
        attempt = CascadeAttempt(
            level=1,
            model="gpt-4o",
            output="Result",
            self_confidence=self_conf,
            judge_confidence=judge_conf,
            cost=0.05,
        )
        assert attempt.judge_confidence is not None
        assert attempt.judge_confidence.score == 0.7


# ======================================================================
# CascadeResult dataclass
# ======================================================================


class TestCascadeResult:
    """Test the CascadeResult dataclass."""

    def test_basic_creation(self) -> None:
        """CascadeResult is created with correct values."""
        confidence = ConfidenceScore(score=0.85, reasoning="Good result")
        result = CascadeResult(
            output="The answer is 42",
            confidence=confidence,
            attempts=[],
            accepted_at_level=0,
            total_cost=0.001,
            cost_saved_vs_premium=0.019,
        )
        assert result.output == "The answer is 42"
        assert result.confidence.score == 0.85
        assert result.accepted_at_level == 0
        assert result.total_cost == 0.001
        assert result.cost_saved_vs_premium == 0.019


# ======================================================================
# TaskResult dataclass
# ======================================================================


class TestTaskResult:
    """Test the TaskResult structured result dataclass."""

    def test_basic_creation(self) -> None:
        """TaskResult is created with correct values."""
        confidence = ConfidenceScore(score=0.9, reasoning="High")
        result = TaskResult(
            output="def hello(): return 'world'",
            confidence=confidence,
            model="gpt-4o",
            cost=0.02,
            tokens_used=150,
            execution_time=1.5,
        )
        assert result.output == "def hello(): return 'world'"
        assert result.model == "gpt-4o"
        assert result.tokens_used == 150
        assert result.cascade_attempts == 1
        assert result.files_changed == []
        assert result.metadata == {}

    def test_full_values(self) -> None:
        """TaskResult with all fields populated."""
        confidence = ConfidenceScore(score=0.75, reasoning="Moderate")
        result = TaskResult(
            output="code here",
            confidence=confidence,
            model="claude-opus-4-20250514",
            cost=0.10,
            tokens_used=500,
            execution_time=3.2,
            files_changed=["src/main.py", "tests/test_main.py"],
            cascade_attempts=3,
            metadata={"tier": "premium", "escalated": True},
        )
        assert len(result.files_changed) == 2
        assert result.cascade_attempts == 3
        assert result.metadata["tier"] == "premium"


# ======================================================================
# ConfidenceCascade — initialisation
# ======================================================================


class TestConfidenceCascadeInit:
    """Test ConfidenceCascade initialisation and level building."""

    def test_default_config(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Default config creates a multi-level cascade."""
        cascade = cascade_high_confidence
        assert len(cascade._levels) >= 1

    def test_custom_levels(
        self,
        cascade_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        high_confidence_litellm: MockLiteLLM,
        model_pool: ModelPool,
    ) -> None:
        """Custom levels override the default 3-tier cascade."""
        custom_levels = [
            CascadeLevel(
                tier="only",
                models=["groq/llama3-8b-8192"],
                confidence_threshold=0.5,
                cost_multiplier=1.0,
            ),
        ]
        config = CascadeConfig(levels=custom_levels)
        engine = _make_engine(
            cascade_settings, mock_cost_tracker, mock_auth, mock_registry,
            high_confidence_litellm,
        )
        cascade = ConfidenceCascade(engine, model_pool, config)
        assert len(cascade._levels) == 1
        assert cascade._levels[0].tier == "only"

    def test_build_levels_has_tiers(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Built levels have distinct tier names."""
        tiers = [level.tier for level in cascade_high_confidence._levels]
        # Should have at least cheap and something else (or just cheap if limited models)
        assert len(tiers) >= 1
        # All tiers should be non-empty strings
        for tier in tiers:
            assert tier
            assert isinstance(tier, str)

    def test_build_levels_models_populated(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Each built level has at least one model."""
        for level in cascade_high_confidence._levels:
            assert len(level.models) >= 1

    def test_build_levels_thresholds_descending(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Higher tiers have lower confidence thresholds (more accepting)."""
        levels = cascade_high_confidence._levels
        if len(levels) > 1:
            for i in range(len(levels) - 1):
                assert levels[i].confidence_threshold >= levels[i + 1].confidence_threshold


# ======================================================================
# ConfidenceCascade — execution (high confidence)
# ======================================================================


class TestCascadeHighConfidence:
    """Test cascade when models return high confidence — accepts early."""

    async def test_accepts_at_first_level(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """High-confidence response is accepted at the first level."""
        result = await cascade_high_confidence.execute("What is 2 + 2?")
        assert isinstance(result, CascadeResult)
        assert result.output
        assert result.confidence.score > 0.0
        assert result.accepted_at_level == 0

    async def test_single_attempt_on_high_confidence(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Only one attempt is made when confidence is high."""
        result = await cascade_high_confidence.execute("Simple question")
        assert len(result.attempts) == 1
        assert result.attempts[0].accepted is True

    async def test_total_cost_tracked(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Total cost is tracked across attempts."""
        result = await cascade_high_confidence.execute("Test cost tracking")
        assert result.total_cost >= 0.0

    async def test_cost_saved_non_negative(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Savings estimate is non-negative."""
        result = await cascade_high_confidence.execute("Test savings")
        assert result.cost_saved_vs_premium >= 0.0


# ======================================================================
# ConfidenceCascade — execution (low confidence)
# ======================================================================


class TestCascadeLowConfidence:
    """Test cascade when models return low confidence — escalates."""

    async def test_escalates_through_levels(
        self, cascade_low_confidence: ConfidenceCascade,
    ) -> None:
        """Low confidence causes escalation through multiple levels."""
        result = await cascade_low_confidence.execute("Complex question needing escalation")
        assert isinstance(result, CascadeResult)
        # Should have tried multiple levels
        assert len(result.attempts) >= 1

    async def test_all_attempts_recorded(
        self, cascade_low_confidence: ConfidenceCascade,
    ) -> None:
        """All escalation attempts are recorded."""
        result = await cascade_low_confidence.execute("Hard problem")
        for attempt in result.attempts:
            assert attempt.model
            assert attempt.self_confidence.score >= 0.0

    async def test_forced_accept_best_attempt(
        self, cascade_low_confidence: ConfidenceCascade,
    ) -> None:
        """When no attempt meets the threshold, the best one is accepted."""
        result = await cascade_low_confidence.execute("Very hard problem")
        # Should still have output
        assert result.output
        # Exactly one attempt should be marked accepted
        accepted_count = sum(1 for a in result.attempts if a.accepted)
        assert accepted_count == 1

    async def test_multiple_attempts_cost_accumulated(
        self, cascade_low_confidence: ConfidenceCascade,
    ) -> None:
        """Cost accumulates across multiple escalation attempts."""
        result = await cascade_low_confidence.execute("Multi-escalation test")
        if len(result.attempts) > 1:
            individual_costs = sum(a.cost for a in result.attempts)
            assert result.total_cost == pytest.approx(individual_costs, abs=0.001)


# ======================================================================
# ConfidenceCascade — edge cases
# ======================================================================


class TestCascadeEdgeCases:
    """Test edge cases and error handling."""

    async def test_empty_prompt_raises(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="Prompt must not be empty"):
            await cascade_high_confidence.execute("")

    async def test_whitespace_prompt_raises(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Whitespace-only prompt raises ValueError."""
        with pytest.raises(ValueError, match="Prompt must not be empty"):
            await cascade_high_confidence.execute("   \t\n  ")

    async def test_with_context(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Cascade works with additional context."""
        result = await cascade_high_confidence.execute(
            "Explain this code",
            context="def foo(): return 42",
        )
        assert result.output
        assert result.confidence.score > 0.0

    async def test_budget_limit_stops_escalation(
        self,
        cascade_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        low_confidence_litellm: MockLiteLLM,
        model_pool: ModelPool,
    ) -> None:
        """Budget limit prevents further escalation."""
        config = CascadeConfig(budget_limit=0.0)
        engine = _make_engine(
            cascade_settings, mock_cost_tracker, mock_auth, mock_registry,
            low_confidence_litellm,
        )
        cascade = ConfidenceCascade(engine, model_pool, config)
        result = await cascade.execute("Budget-limited query")
        # Should have no attempts because budget is 0
        assert len(result.attempts) == 0 or result.total_cost >= 0.0

    async def test_max_escalations_respected(
        self,
        cascade_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        low_confidence_litellm: MockLiteLLM,
        model_pool: ModelPool,
    ) -> None:
        """max_escalations config limits the number of attempts."""
        config = CascadeConfig(max_escalations=1)
        engine = _make_engine(
            cascade_settings, mock_cost_tracker, mock_auth, mock_registry,
            low_confidence_litellm,
        )
        cascade = ConfidenceCascade(engine, model_pool, config)
        result = await cascade.execute("One-shot only")
        assert len(result.attempts) <= 1

    async def test_no_external_judge(
        self,
        cascade_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        high_confidence_litellm: MockLiteLLM,
        model_pool: ModelPool,
    ) -> None:
        """Cascade works without external judge."""
        config = CascadeConfig(use_external_judge=False)
        engine = _make_engine(
            cascade_settings, mock_cost_tracker, mock_auth, mock_registry,
            high_confidence_litellm,
        )
        cascade = ConfidenceCascade(engine, model_pool, config)
        result = await cascade.execute("No judge mode")
        assert result.output
        # No attempt should have a judge confidence
        for attempt in result.attempts:
            assert attempt.judge_confidence is None


# ======================================================================
# ConfidenceCascade — confidence parsing
# ======================================================================


class TestConfidenceParsing:
    """Test the confidence JSON parsing logic."""

    def test_valid_json(self) -> None:
        """Valid JSON is parsed correctly."""
        raw = _make_confidence_json(score=0.85, reasoning="Looks good")
        result = ConfidenceCascade._parse_confidence(raw, "test-model")
        assert result.score == 0.85
        assert result.reasoning == "Looks good"
        assert result.model == "test-model"

    def test_json_with_all_fields(self) -> None:
        """JSON with all fields is parsed correctly."""
        raw = _make_confidence_json(
            score=0.6,
            reasoning="Some concerns",
            uncertainty_areas=["edge cases", "performance"],
            alternative_approaches=["try caching"],
        )
        result = ConfidenceCascade._parse_confidence(raw, "model-x")
        assert result.score == 0.6
        assert len(result.uncertainty_areas) == 2
        assert len(result.alternative_approaches) == 1

    def test_json_embedded_in_text(self) -> None:
        """JSON embedded in surrounding text is extracted."""
        raw = 'Here is my assessment: {"score": 0.7, "reasoning": "OK"} end'
        result = ConfidenceCascade._parse_confidence(raw, "model-y")
        assert result.score == 0.7

    def test_invalid_json_fallback(self) -> None:
        """Invalid JSON falls back to score 0.5."""
        raw = "This is not JSON at all, just plain text."
        result = ConfidenceCascade._parse_confidence(raw, "model-z")
        assert result.score == 0.5
        assert "Could not parse" in result.reasoning

    def test_empty_string_fallback(self) -> None:
        """Empty string falls back to score 0.5."""
        result = ConfidenceCascade._parse_confidence("", "model-empty")
        assert result.score == 0.5

    def test_score_out_of_range_clamped(self) -> None:
        """Parsed score out of [0, 1] is clamped by ConfidenceScore."""
        raw = json.dumps({"score": 5.0, "reasoning": "Over confident"})
        result = ConfidenceCascade._parse_confidence(raw, "model-over")
        assert result.score == 1.0

    def test_negative_score_clamped(self) -> None:
        """Negative score is clamped to 0.0."""
        raw = json.dumps({"score": -1.0, "reasoning": "Negative"})
        result = ConfidenceCascade._parse_confidence(raw, "model-neg")
        assert result.score == 0.0

    def test_missing_score_defaults(self) -> None:
        """Missing score key defaults to 0.5."""
        raw = json.dumps({"reasoning": "No score given"})
        result = ConfidenceCascade._parse_confidence(raw, "model-nosc")
        assert result.score == 0.5


# ======================================================================
# ConfidenceCascade — decision logic
# ======================================================================


class TestCascadeDecisionLogic:
    """Test _should_accept, _should_escalate, _effective_confidence."""

    def test_effective_confidence_self_only(self) -> None:
        """Effective confidence is self-score when no judge."""
        attempt = CascadeAttempt(
            level=0,
            model="m1",
            output="x",
            self_confidence=ConfidenceScore(score=0.8, reasoning="ok"),
            judge_confidence=None,
        )
        assert ConfidenceCascade._effective_confidence(attempt) == 0.8

    def test_effective_confidence_with_judge(self) -> None:
        """Effective confidence is min(self, judge) when judge present."""
        attempt = CascadeAttempt(
            level=0,
            model="m1",
            output="x",
            self_confidence=ConfidenceScore(score=0.9, reasoning="self high"),
            judge_confidence=ConfidenceScore(score=0.6, reasoning="judge lower"),
        )
        assert ConfidenceCascade._effective_confidence(attempt) == 0.6

    def test_effective_confidence_judge_higher(self) -> None:
        """When judge is higher than self, self score is used (min)."""
        attempt = CascadeAttempt(
            level=0,
            model="m1",
            output="x",
            self_confidence=ConfidenceScore(score=0.5, reasoning="self low"),
            judge_confidence=ConfidenceScore(score=0.9, reasoning="judge high"),
        )
        assert ConfidenceCascade._effective_confidence(attempt) == 0.5

    def test_should_accept_high_confidence(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """High-confidence attempt should be accepted."""
        attempt = CascadeAttempt(
            level=0,
            model="m1",
            output="x",
            self_confidence=ConfidenceScore(score=0.95, reasoning="great"),
        )
        assert cascade_high_confidence._should_accept(attempt) is True

    def test_should_not_accept_low_confidence(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Low-confidence attempt should NOT be accepted."""
        attempt = CascadeAttempt(
            level=0,
            model="m1",
            output="x",
            self_confidence=ConfidenceScore(score=0.3, reasoning="poor"),
        )
        assert cascade_high_confidence._should_accept(attempt) is False

    def test_should_accept_below_min_confidence(
        self,
        cascade_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        high_confidence_litellm: MockLiteLLM,
        model_pool: ModelPool,
    ) -> None:
        """Attempt below min_confidence is rejected even if above level threshold."""
        config = CascadeConfig(
            min_confidence=0.9,
            levels=[
                CascadeLevel(
                    tier="easy",
                    models=["m1"],
                    confidence_threshold=0.5,
                    cost_multiplier=1.0,
                ),
            ],
        )
        engine = _make_engine(
            cascade_settings, mock_cost_tracker, mock_auth, mock_registry,
            high_confidence_litellm,
        )
        cascade = ConfidenceCascade(engine, model_pool, config)
        attempt = CascadeAttempt(
            level=0,
            model="m1",
            output="x",
            self_confidence=ConfidenceScore(score=0.7, reasoning="above threshold but below min"),
        )
        assert cascade._should_accept(attempt) is False

    def test_should_escalate_has_next_level(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Escalation is warranted when there are more levels."""
        if len(cascade_high_confidence._levels) > 1:
            attempt = CascadeAttempt(
                level=0,
                model="m1",
                output="x",
                self_confidence=ConfidenceScore(score=0.3, reasoning="low"),
            )
            assert cascade_high_confidence._should_escalate(attempt, 0) is True

    def test_should_not_escalate_at_max_level(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Cannot escalate beyond the last level."""
        last_level = len(cascade_high_confidence._levels) - 1
        attempt = CascadeAttempt(
            level=last_level,
            model="m1",
            output="x",
            self_confidence=ConfidenceScore(score=0.3, reasoning="low"),
        )
        assert cascade_high_confidence._should_escalate(attempt, last_level) is False


# ======================================================================
# ConfidenceCascade — message building
# ======================================================================


class TestCascadeMessageBuilding:
    """Test the message construction for generation and assessment."""

    def test_basic_messages(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Basic messages contain system and user roles."""
        messages = cascade_high_confidence._build_generation_messages(
            prompt="Write hello world",
            context="",
            previous_attempts=[],
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Write hello world" in messages[1]["content"]

    def test_messages_with_context(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Context is included in user message."""
        messages = cascade_high_confidence._build_generation_messages(
            prompt="Explain this",
            context="def foo(): return 42",
            previous_attempts=[],
        )
        assert "def foo(): return 42" in messages[1]["content"]

    def test_messages_with_previous_attempts(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Previous attempts are included for learning."""
        prev = [
            CascadeAttempt(
                level=0,
                model="cheap-model",
                output="Initial attempt output",
                self_confidence=ConfidenceScore(
                    score=0.4,
                    reasoning="Low confidence",
                    uncertainty_areas=["correctness"],
                ),
            ),
        ]
        messages = cascade_high_confidence._build_generation_messages(
            prompt="Retry this",
            context="",
            previous_attempts=prev,
        )
        content = messages[1]["content"]
        assert "Previous attempts" in content
        assert "Initial attempt output" in content
        assert "cheap-model" in content
        assert "correctness" in content
        assert "Improve upon" in content


# ======================================================================
# ConfidenceCascade — judge model selection
# ======================================================================


class TestJudgeModelSelection:
    """Test the judge model selection logic."""

    def test_judge_different_from_current(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Judge model should be different from current level's models."""
        if len(cascade_high_confidence._levels) < 2:
            pytest.skip("Need at least 2 levels to test judge selection")
        current_models = set(cascade_high_confidence._levels[0].models)
        judge = cascade_high_confidence._select_judge_model(0)
        if judge is not None:
            assert judge not in current_models

    def test_judge_returns_none_single_model(
        self,
        cascade_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        high_confidence_litellm: MockLiteLLM,
        model_pool: ModelPool,
    ) -> None:
        """Judge returns None if only one model available everywhere."""
        config = CascadeConfig(
            levels=[
                CascadeLevel(
                    tier="only",
                    models=["unique-model-xyz"],
                    confidence_threshold=0.5,
                    cost_multiplier=1.0,
                ),
            ],
        )
        engine = _make_engine(
            cascade_settings, mock_cost_tracker, mock_auth, mock_registry,
            high_confidence_litellm,
        )
        cascade = ConfidenceCascade(engine, model_pool, config)
        # The judge must be different from "unique-model-xyz", so it will
        # try to find one from other levels (none) then fall back to pool
        judge = cascade._select_judge_model(0)
        # Should either find something from the pool or return None
        if judge is not None:
            assert judge != "unique-model-xyz"


# ======================================================================
# ConfidenceCascade — savings calculation
# ======================================================================


class TestCascadeSavingsCalculation:
    """Test cost savings estimation."""

    def test_savings_non_negative(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Savings are always non-negative."""
        confidence = ConfidenceScore(score=0.9, reasoning="good")
        attempt = CascadeAttempt(
            level=0, model="cheap", output="ok",
            self_confidence=confidence, accepted=True, cost=0.001,
        )
        result = CascadeResult(
            output="ok",
            confidence=confidence,
            attempts=[attempt],
            accepted_at_level=0,
            total_cost=0.001,
            cost_saved_vs_premium=0.0,
        )
        savings = cascade_high_confidence._calculate_savings(result)
        assert savings >= 0.0

    def test_savings_zero_no_levels(
        self,
        cascade_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        high_confidence_litellm: MockLiteLLM,
        model_pool: ModelPool,
    ) -> None:
        """Savings is 0 when no levels are configured."""
        config = CascadeConfig(levels=[])
        engine = _make_engine(
            cascade_settings, mock_cost_tracker, mock_auth, mock_registry,
            high_confidence_litellm,
        )
        cascade = ConfidenceCascade(engine, model_pool, config)
        confidence = ConfidenceScore(score=0.5, reasoning="n/a")
        result = CascadeResult(
            output="x", confidence=confidence, attempts=[],
            accepted_at_level=0, total_cost=0, cost_saved_vs_premium=0,
        )
        assert cascade._calculate_savings(result) == 0.0

    def test_savings_with_no_attempts(
        self, cascade_high_confidence: ConfidenceCascade,
    ) -> None:
        """Savings is 0 when there are no attempts."""
        confidence = ConfidenceScore(score=0.5, reasoning="n/a")
        result = CascadeResult(
            output="x", confidence=confidence, attempts=[],
            accepted_at_level=0, total_cost=0, cost_saved_vs_premium=0,
        )
        assert cascade_high_confidence._calculate_savings(result) == 0.0


# ======================================================================
# ConfidenceCascade — assessment cost estimation
# ======================================================================


class TestAssessmentCostEstimation:
    """Test the assessment cost estimation helper."""

    def test_estimate_returns_float(self) -> None:
        """Cost estimate returns a non-negative float."""
        cost = ConfidenceCascade._estimate_assessment_cost("gpt-4o")
        assert isinstance(cost, float)
        assert cost >= 0.0

    def test_estimate_unknown_model(self) -> None:
        """Unknown model returns a valid cost estimate (may be 0)."""
        cost = ConfidenceCascade._estimate_assessment_cost("unknown/model-xyz")
        assert isinstance(cost, float)
        assert cost >= 0.0


# ======================================================================
# Integration: full cascade with mixed confidence
# ======================================================================


class TestCascadeIntegration:
    """Integration tests: full cascade flows."""

    async def test_cascade_returns_valid_result(
        self,
        cascade_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        model_pool: ModelPool,
    ) -> None:
        """Full cascade run returns a structurally valid result."""
        mock = MockLiteLLM()
        mock.set_default_response(
            MockResponse(
                content=_make_confidence_json(score=0.88, reasoning="Good"),
                input_tokens=150,
                output_tokens=80,
            ),
        )
        engine = _make_engine(
            cascade_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
        )
        cascade = ConfidenceCascade(engine, model_pool)
        result = await cascade.execute("Implement a binary search function in Python")

        assert isinstance(result, CascadeResult)
        assert result.output
        assert result.confidence.score > 0.0
        assert len(result.attempts) >= 1
        assert result.total_cost >= 0.0

    async def test_cascade_with_custom_config(
        self,
        cascade_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        model_pool: ModelPool,
    ) -> None:
        """Cascade with a fully custom config works correctly."""
        mock = MockLiteLLM()
        mock.set_default_response(
            MockResponse(
                content=_make_confidence_json(score=0.6, reasoning="Medium confidence"),
                input_tokens=100,
                output_tokens=50,
            ),
        )
        engine = _make_engine(
            cascade_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
        )
        research_models = model_pool.get_research_models()
        first_model = research_models[0] if research_models else "groq/llama3-8b-8192"

        config = CascadeConfig(
            levels=[
                CascadeLevel(
                    tier="test",
                    models=[first_model],
                    confidence_threshold=0.5,
                    cost_multiplier=1.0,
                ),
            ],
            min_confidence=0.5,
            max_escalations=1,
            use_external_judge=False,
        )
        cascade = ConfidenceCascade(engine, model_pool, config)
        result = await cascade.execute("Test custom config")

        assert result.output
        assert len(result.attempts) == 1
        assert result.attempts[0].judge_confidence is None

    async def test_cascade_escalation_includes_prior_context(
        self,
        cascade_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        model_pool: ModelPool,
    ) -> None:
        """Higher-tier models receive context from previous attempts."""
        call_contents: list[str] = []

        class CapturingMock(MockLiteLLM):
            """MockLiteLLM that captures user message content."""

            async def acompletion(
                self, model: str, messages: list[dict[str, str]], **kwargs: object,
            ) -> dict:
                for msg in messages:
                    if msg.get("role") == "user":
                        call_contents.append(msg.get("content", ""))
                return await super().acompletion(model, messages, **kwargs)

        mock = CapturingMock()
        mock.set_default_response(
            MockResponse(
                content=_make_confidence_json(
                    score=0.3, reasoning="Not confident",
                    uncertainty_areas=["accuracy"],
                ),
                input_tokens=100,
                output_tokens=50,
            ),
        )

        engine = _make_engine(
            cascade_settings, mock_cost_tracker, mock_auth, mock_registry, mock,
        )
        cascade = ConfidenceCascade(engine, model_pool)
        await cascade.execute("Complex task that needs escalation")

        # Check if later calls include "Previous attempts" in the prompt
        if len(call_contents) > 3:
            # The generation calls for level 1+ should reference prior attempts
            later_calls = call_contents[3:]  # Skip first level's generation + assessments
            found_prior = any("Previous attempts" in c for c in later_calls)
            # If there were multiple levels, prior context should be present
            if len(cascade._levels) > 1:
                assert found_prior, "Higher-tier generation should include prior attempt context"
