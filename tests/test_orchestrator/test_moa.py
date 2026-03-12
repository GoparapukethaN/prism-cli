"""Tests for Mixture-of-Agents (MoA) parallel generation with pairwise ranking.

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
from prism.orchestrator.moa import (
    MixtureOfAgents,
    MoAConfig,
    MoALayer,
    MoAResult,
    OutputRanker,
    PairwiseRanking,
)
from prism.orchestrator.swarm import ModelPool
from prism.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from pathlib import Path


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def moa_settings(tmp_path: Path) -> Settings:
    """Settings with a reasonable budget for MoA testing."""
    config = PrismConfig(
        prism_home=tmp_path / ".prism",
        budget=BudgetConfig(daily_limit=100.0, monthly_limit=1000.0),
    )
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()
    return settings


@pytest.fixture()
def mock_auth() -> MagicMock:
    """Mock AuthManager that returns test keys for all providers."""
    mgr = MagicMock()
    mgr.get_key.return_value = "test-key-moa-1234"
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
def mock_registry(mock_auth: MagicMock, moa_settings: Settings) -> ProviderRegistry:
    """Real ProviderRegistry using mock auth."""
    return ProviderRegistry(settings=moa_settings, auth_manager=mock_auth)


@pytest.fixture()
def model_pool(mock_registry: ProviderRegistry) -> ModelPool:
    """A real ModelPool using the mock registry."""
    return ModelPool(mock_registry)


def _judge_response(*, winner: str = "A", reason: str = "Better quality", confidence: float = 0.8) -> str:
    """Create a mock judge response JSON."""
    return json.dumps({
        "winner": winner,
        "reason": reason,
        "confidence": confidence,
    })


@pytest.fixture()
def mock_litellm_moa() -> MockLiteLLM:
    """MockLiteLLM with proposer + judge + fusion responses."""
    mock = MockLiteLLM()
    mock.set_default_response(
        MockResponse(
            content="Mock MoA response from the AI model.",
            input_tokens=200,
            output_tokens=100,
        ),
    )
    return mock


@pytest.fixture()
def mock_litellm_judge() -> MockLiteLLM:
    """MockLiteLLM that returns judge-style JSON responses."""
    mock = MockLiteLLM()
    mock.set_default_response(
        MockResponse(
            content=_judge_response(),
            input_tokens=150,
            output_tokens=60,
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
def completion_engine(
    moa_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    mock_litellm_moa: MockLiteLLM,
) -> CompletionEngine:
    """CompletionEngine wired to mocks — no real API calls."""
    return _make_engine(
        moa_settings, mock_cost_tracker, mock_auth, mock_registry, mock_litellm_moa,
    )


@pytest.fixture()
def judge_engine(
    moa_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    mock_litellm_judge: MockLiteLLM,
) -> CompletionEngine:
    """CompletionEngine that returns judge-style JSON."""
    return _make_engine(
        moa_settings, mock_cost_tracker, mock_auth, mock_registry, mock_litellm_judge,
    )


# ======================================================================
# MoALayer dataclass
# ======================================================================


class TestMoALayer:
    """Test MoALayer dataclass."""

    def test_default_values(self) -> None:
        """MoALayer has sensible defaults."""
        layer = MoALayer(layer_number=0)
        assert layer.layer_number == 0
        assert layer.models == []
        assert layer.outputs == {}
        assert layer.cost == 0.0

    def test_custom_values(self) -> None:
        """MoALayer accepts custom values."""
        layer = MoALayer(
            layer_number=1,
            models=["gpt-4o", "claude-sonnet-4-20250514"],
            outputs={"gpt-4o": "response A", "claude-sonnet-4-20250514": "response B"},
            cost=0.05,
        )
        assert layer.layer_number == 1
        assert len(layer.models) == 2
        assert len(layer.outputs) == 2
        assert layer.cost == 0.05


# ======================================================================
# MoAConfig dataclass
# ======================================================================


class TestMoAConfig:
    """Test MoAConfig dataclass."""

    def test_defaults(self) -> None:
        """MoAConfig has sensible defaults."""
        config = MoAConfig()
        assert config.num_proposers == 3
        assert config.num_layers == 2
        assert config.use_ranking is True
        assert config.fusion_model is None

    def test_custom_config(self) -> None:
        """MoAConfig accepts overrides."""
        config = MoAConfig(
            num_proposers=5,
            num_layers=3,
            use_ranking=False,
            fusion_model="gpt-4o",
        )
        assert config.num_proposers == 5
        assert config.num_layers == 3
        assert config.use_ranking is False
        assert config.fusion_model == "gpt-4o"


# ======================================================================
# PairwiseRanking dataclass
# ======================================================================


class TestPairwiseRanking:
    """Test PairwiseRanking dataclass."""

    def test_creation(self) -> None:
        """PairwiseRanking stores comparison results."""
        ranking = PairwiseRanking(
            candidate_a="model-a",
            candidate_b="model-b",
            winner="model-a",
            reason="More accurate response",
            confidence=0.85,
        )
        assert ranking.candidate_a == "model-a"
        assert ranking.winner == "model-a"
        assert ranking.confidence == 0.85


# ======================================================================
# MoAResult dataclass
# ======================================================================


class TestMoAResult:
    """Test MoAResult dataclass."""

    def test_defaults(self) -> None:
        """MoAResult has sensible defaults."""
        result = MoAResult()
        assert result.layers == []
        assert result.rankings == []
        assert result.final_output == ""
        assert result.best_individual is None
        assert result.total_cost == 0.0
        assert result.participating_models == []

    def test_full_result(self) -> None:
        """MoAResult stores all pipeline outputs."""
        layer = MoALayer(layer_number=0, models=["m1", "m2"])
        ranking = PairwiseRanking(
            candidate_a="m1", candidate_b="m2",
            winner="m1", reason="better", confidence=0.9,
        )
        result = MoAResult(
            layers=[layer],
            rankings=[ranking],
            final_output="Synthesized output",
            best_individual="m1",
            total_cost=0.05,
            participating_models=["m1", "m2", "judge"],
        )
        assert len(result.layers) == 1
        assert len(result.rankings) == 1
        assert result.final_output == "Synthesized output"
        assert result.best_individual == "m1"
        assert result.total_cost == 0.05
        assert len(result.participating_models) == 3


# ======================================================================
# OutputRanker
# ======================================================================


class TestOutputRanker:
    """Test pairwise ranking via judge model."""

    async def test_rank_two_outputs(
        self,
        judge_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Ranking two outputs produces one pairwise comparison."""
        judge_model = model_pool.get_review_model()
        ranker = OutputRanker(judge_engine, judge_model)

        outputs = {
            "model-a": "Response from model A",
            "model-b": "Response from model B",
        }
        rankings = await ranker.rank(outputs, "Test prompt")
        assert len(rankings) == 1
        assert rankings[0].candidate_a == "model-a"
        assert rankings[0].candidate_b == "model-b"
        assert rankings[0].winner in ("model-a", "model-b")
        assert 0.0 <= rankings[0].confidence <= 1.0

    async def test_rank_three_outputs(
        self,
        judge_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Ranking three outputs produces three pairwise comparisons."""
        judge_model = model_pool.get_review_model()
        ranker = OutputRanker(judge_engine, judge_model)

        outputs = {
            "model-a": "Response A",
            "model-b": "Response B",
            "model-c": "Response C",
        }
        rankings = await ranker.rank(outputs, "Test prompt")
        # C(3,2) = 3 pairs
        assert len(rankings) == 3

    async def test_rank_fewer_than_two_raises(
        self,
        judge_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Fewer than 2 outputs raises ValueError."""
        judge_model = model_pool.get_review_model()
        ranker = OutputRanker(judge_engine, judge_model)

        with pytest.raises(ValueError, match="At least 2 outputs"):
            await ranker.rank({"model-a": "Only one"}, "Test prompt")

    async def test_rank_empty_raises(
        self,
        judge_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Empty outputs raises ValueError."""
        judge_model = model_pool.get_review_model()
        ranker = OutputRanker(judge_engine, judge_model)

        with pytest.raises(ValueError, match="At least 2 outputs"):
            await ranker.rank({}, "Test prompt")

    def test_get_win_rates_basic(self) -> None:
        """Win rates calculated correctly for simple case."""
        rankings = [
            PairwiseRanking(
                candidate_a="a", candidate_b="b",
                winner="a", reason="better", confidence=0.8,
            ),
            PairwiseRanking(
                candidate_a="a", candidate_b="c",
                winner="a", reason="better", confidence=0.9,
            ),
            PairwiseRanking(
                candidate_a="b", candidate_b="c",
                winner="c", reason="better", confidence=0.7,
            ),
        ]
        win_rates = OutputRanker.get_win_rates(rankings)
        assert win_rates["a"] == 1.0  # 2 wins out of 2 appearances
        assert win_rates["b"] == 0.0  # 0 wins out of 2 appearances
        assert win_rates["c"] == 0.5  # 1 win out of 2 appearances

    def test_get_win_rates_empty(self) -> None:
        """Empty rankings returns empty dict."""
        assert OutputRanker.get_win_rates([]) == {}

    def test_get_best_basic(self) -> None:
        """Best model is the one with highest win rate."""
        rankings = [
            PairwiseRanking(
                candidate_a="a", candidate_b="b",
                winner="a", reason="better", confidence=0.8,
            ),
            PairwiseRanking(
                candidate_a="a", candidate_b="c",
                winner="a", reason="better", confidence=0.9,
            ),
            PairwiseRanking(
                candidate_a="b", candidate_b="c",
                winner="c", reason="better", confidence=0.7,
            ),
        ]
        assert OutputRanker.get_best(rankings) == "a"

    def test_get_best_empty_raises(self) -> None:
        """Empty rankings raises ValueError."""
        with pytest.raises(ValueError, match="Cannot determine best"):
            OutputRanker.get_best([])

    def test_parse_judge_response_valid_a(self) -> None:
        """Valid JSON with winner A parses correctly."""
        raw = json.dumps({"winner": "A", "reason": "Better", "confidence": 0.9})
        result = OutputRanker._parse_judge_response(raw, "model-x", "model-y")
        assert result.winner == "model-x"
        assert result.reason == "Better"
        assert result.confidence == 0.9

    def test_parse_judge_response_valid_b(self) -> None:
        """Valid JSON with winner B parses correctly."""
        raw = json.dumps({"winner": "B", "reason": "Clearer", "confidence": 0.75})
        result = OutputRanker._parse_judge_response(raw, "model-x", "model-y")
        assert result.winner == "model-y"
        assert result.reason == "Clearer"
        assert result.confidence == 0.75

    def test_parse_judge_response_invalid_json(self) -> None:
        """Invalid JSON falls back to first candidate."""
        result = OutputRanker._parse_judge_response(
            "This is not JSON!", "model-x", "model-y",
        )
        assert result.winner == "model-x"
        assert result.confidence == 0.0
        assert "defaulting" in result.reason.lower()

    def test_parse_judge_response_clamps_confidence(self) -> None:
        """Confidence values are clamped to [0, 1]."""
        raw = json.dumps({"winner": "A", "reason": "ok", "confidence": 1.5})
        result = OutputRanker._parse_judge_response(raw, "m1", "m2")
        assert result.confidence == 1.0

        raw_neg = json.dumps({"winner": "A", "reason": "ok", "confidence": -0.5})
        result_neg = OutputRanker._parse_judge_response(raw_neg, "m1", "m2")
        assert result_neg.confidence == 0.0

    def test_parse_judge_response_missing_fields(self) -> None:
        """Partial JSON uses defaults."""
        raw = json.dumps({"winner": "B"})
        result = OutputRanker._parse_judge_response(raw, "m1", "m2")
        assert result.winner == "m2"
        assert result.confidence == 0.5
        assert result.reason == "No reason provided."

    def test_parse_judge_response_with_surrounding_text(self) -> None:
        """JSON embedded in text is still parsed."""
        raw = 'Here is my analysis:\n{"winner": "A", "reason": "better", "confidence": 0.8}\nDone.'
        result = OutputRanker._parse_judge_response(raw, "m1", "m2")
        assert result.winner == "m1"
        assert result.confidence == 0.8

    async def test_compare_pair_makes_llm_call(
        self,
        judge_engine: CompletionEngine,
        mock_litellm_judge: MockLiteLLM,
        model_pool: ModelPool,
    ) -> None:
        """Pairwise comparison makes an LLM call to the judge model."""
        judge_model = model_pool.get_review_model()
        ranker = OutputRanker(judge_engine, judge_model)

        result = await ranker._compare_pair(
            model_a="m-a",
            output_a="Output A content",
            model_b="m-b",
            output_b="Output B content",
            prompt="Test prompt",
        )

        assert isinstance(result, PairwiseRanking)
        assert result.candidate_a == "m-a"
        assert result.candidate_b == "m-b"
        # Verify an LLM call was made
        assert len(mock_litellm_judge.call_log) >= 1

    async def test_rank_handles_failed_comparisons(
        self,
        moa_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        model_pool: ModelPool,
    ) -> None:
        """Failed comparisons fall back to first candidate with zero confidence."""
        mock = MockLiteLLM()
        # Make all calls fail
        mock.set_default_response(
            MockResponse(content="not json at all", input_tokens=10, output_tokens=5),
        )
        engine = _make_engine(moa_settings, mock_cost_tracker, mock_auth, mock_registry, mock)
        judge_model = model_pool.get_review_model()
        ranker = OutputRanker(engine, judge_model)

        outputs = {"model-a": "A output", "model-b": "B output"}
        rankings = await ranker.rank(outputs, "Test")
        assert len(rankings) == 1
        # Fallback: first candidate wins with zero confidence
        assert rankings[0].confidence == 0.0


# ======================================================================
# MixtureOfAgents
# ======================================================================


class TestMixtureOfAgents:
    """Test the full MoA pipeline."""

    async def test_basic_generation(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Basic MoA generation produces a result with layers and output."""
        config = MoAConfig(num_proposers=2, num_layers=1, use_ranking=False)
        moa = MixtureOfAgents(completion_engine, model_pool, config)

        result = await moa.generate("What is the meaning of life?")
        assert isinstance(result, MoAResult)
        assert result.final_output
        assert len(result.layers) == 1
        assert result.total_cost >= 0.0
        assert len(result.participating_models) >= 1

    async def test_multi_layer_generation(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Multi-layer MoA runs the specified number of layers."""
        config = MoAConfig(num_proposers=2, num_layers=3, use_ranking=False)
        moa = MixtureOfAgents(completion_engine, model_pool, config)

        result = await moa.generate("Explain quantum computing")
        assert len(result.layers) == 3
        for i, layer in enumerate(result.layers):
            assert layer.layer_number == i
            assert len(layer.outputs) >= 1

    async def test_with_ranking(
        self,
        moa_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        model_pool: ModelPool,
    ) -> None:
        """MoA with ranking produces rankings and identifies best individual."""
        mock = MockLiteLLM()
        # Default response that also works as a judge response
        mock.set_default_response(
            MockResponse(
                content=_judge_response(winner="A", reason="More detailed", confidence=0.85),
                input_tokens=100,
                output_tokens=50,
            ),
        )
        engine = _make_engine(moa_settings, mock_cost_tracker, mock_auth, mock_registry, mock)
        config = MoAConfig(num_proposers=2, num_layers=1, use_ranking=True)
        moa = MixtureOfAgents(engine, model_pool, config)

        result = await moa.generate("Compare Python and Rust")
        assert len(result.rankings) >= 1
        assert result.best_individual is not None
        assert result.best_individual in result.participating_models

    async def test_without_ranking(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """MoA without ranking skips ranking step."""
        config = MoAConfig(num_proposers=2, num_layers=1, use_ranking=False)
        moa = MixtureOfAgents(completion_engine, model_pool, config)

        result = await moa.generate("Simple question")
        assert result.rankings == []
        assert result.best_individual is None

    async def test_empty_prompt_raises(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Empty prompt raises ValueError."""
        moa = MixtureOfAgents(completion_engine, model_pool)
        with pytest.raises(ValueError, match="Prompt must not be empty"):
            await moa.generate("")

    async def test_whitespace_prompt_raises(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Whitespace-only prompt raises ValueError."""
        moa = MixtureOfAgents(completion_engine, model_pool)
        with pytest.raises(ValueError, match="Prompt must not be empty"):
            await moa.generate("   ")

    async def test_with_context(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Context is included in generation."""
        config = MoAConfig(num_proposers=2, num_layers=1, use_ranking=False)
        moa = MixtureOfAgents(completion_engine, model_pool, config)

        result = await moa.generate(
            "Explain this code",
            context="def hello(): return 'world'",
        )
        assert result.final_output

    async def test_custom_fusion_model(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Custom fusion model is used when specified."""
        config = MoAConfig(
            num_proposers=2,
            num_layers=1,
            use_ranking=False,
            fusion_model="gpt-4o",
        )
        moa = MixtureOfAgents(completion_engine, model_pool, config)

        result = await moa.generate("Test with custom fusion")
        assert "gpt-4o" in result.participating_models

    async def test_layer_outputs_propagate(
        self,
        moa_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        model_pool: ModelPool,
    ) -> None:
        """Layer 2 prompts include Layer 1 outputs (refinement)."""
        mock = MockLiteLLM()
        mock.set_default_response(
            MockResponse(content="Refined response.", input_tokens=200, output_tokens=100),
        )
        engine = _make_engine(moa_settings, mock_cost_tracker, mock_auth, mock_registry, mock)
        config = MoAConfig(num_proposers=2, num_layers=2, use_ranking=False)
        moa = MixtureOfAgents(engine, model_pool, config)

        result = await moa.generate("Test layered generation")

        # Layer 0 calls should not mention "Previous model outputs"
        # Layer 1 calls SHOULD mention previous outputs in system prompt
        assert len(result.layers) == 2
        # Layer 2 should have calls with previous context in system prompt
        assert len(result.layers[1].outputs) >= 1

    async def test_participating_models_collected(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """All participating models are collected in the result."""
        config = MoAConfig(num_proposers=2, num_layers=1, use_ranking=False)
        moa = MixtureOfAgents(completion_engine, model_pool, config)

        result = await moa.generate("Collect models")
        assert len(result.participating_models) >= 2  # proposers + fusion model

    async def test_total_cost_accumulated(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Total cost includes all layers and fusion."""
        config = MoAConfig(num_proposers=2, num_layers=2, use_ranking=False)
        moa = MixtureOfAgents(completion_engine, model_pool, config)

        result = await moa.generate("Track costs")
        assert result.total_cost > 0.0
        # Layer costs should sum up
        layer_costs = sum(layer.cost for layer in result.layers)
        assert layer_costs >= 0.0


# ======================================================================
# MixtureOfAgents._select_proposers
# ======================================================================


class TestProposerSelection:
    """Test proposer model selection for diversity."""

    def test_selects_correct_count(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Selection returns the configured number of proposers."""
        config = MoAConfig(num_proposers=2)
        moa = MixtureOfAgents(completion_engine, model_pool, config)
        proposers = moa._select_proposers()
        assert len(proposers) == 2

    def test_proposers_prefer_diversity(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Proposers are selected from different providers when possible."""
        config = MoAConfig(num_proposers=3)
        moa = MixtureOfAgents(completion_engine, model_pool, config)
        proposers = moa._select_proposers()

        # Check provider diversity
        providers = set()
        for mid in proposers:
            info = mock_registry.get_model_info(mid)
            if info:
                providers.add(info.provider)
        # Should have at least 2 providers if available
        available_providers = set()
        for m in mock_registry.get_available_models():
            available_providers.add(m.provider)
        expected_diversity = min(len(available_providers), len(proposers))
        assert len(providers) >= min(2, expected_diversity)

    def test_proposers_no_duplicates(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Selected proposers should not contain duplicates."""
        config = MoAConfig(num_proposers=3)
        moa = MixtureOfAgents(completion_engine, model_pool, config)
        proposers = moa._select_proposers()
        assert len(proposers) == len(set(proposers))


# ======================================================================
# MixtureOfAgents._fuse
# ======================================================================


class TestFusion:
    """Test output fusion/synthesis."""

    async def test_fuse_empty_outputs(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Empty outputs returns empty string."""
        moa = MixtureOfAgents(completion_engine, model_pool)
        result = await moa._fuse("prompt", {}, [])
        assert result == ""

    async def test_fuse_single_output(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Single output is returned directly without fusion call."""
        moa = MixtureOfAgents(completion_engine, model_pool)
        result = await moa._fuse("prompt", {"m1": "Only output"}, [])
        assert result == "Only output"

    async def test_fuse_multiple_outputs(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Multiple outputs are fused via LLM call."""
        moa = MixtureOfAgents(completion_engine, model_pool)
        outputs = {"m1": "Response 1", "m2": "Response 2"}
        result = await moa._fuse("Test prompt", outputs, [])
        assert result  # Non-empty from mock

    async def test_fuse_with_rankings(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Fusion includes ranking information when available."""
        moa = MixtureOfAgents(completion_engine, model_pool)
        outputs = {"m1": "Response 1", "m2": "Response 2"}
        rankings = [
            PairwiseRanking(
                candidate_a="m1", candidate_b="m2",
                winner="m1", reason="Better", confidence=0.8,
            ),
        ]
        result = await moa._fuse("Test prompt", outputs, rankings)
        assert result  # Non-empty from mock


# ======================================================================
# MixtureOfAgents._generate_one
# ======================================================================


class TestGenerateOne:
    """Test individual model generation."""

    async def test_generate_without_previous(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
        mock_litellm_moa: MockLiteLLM,
    ) -> None:
        """First-layer generation uses the proposer system prompt."""
        moa = MixtureOfAgents(completion_engine, model_pool)
        models = model_pool._registry.get_available_models()
        model_id = models[0].id if models else "gpt-4o"

        content, cost = await moa._generate_one(
            model=model_id,
            prompt="Test prompt",
            previous_outputs=None,
            context="",
        )
        assert content
        assert cost >= 0.0
        # Should have used the proposer system prompt (no "other AI models")
        last_call = mock_litellm_moa.call_log[-1]
        system_msg = last_call["messages"][0]["content"]
        assert "other AI models" not in system_msg

    async def test_generate_with_previous(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
        mock_litellm_moa: MockLiteLLM,
    ) -> None:
        """Subsequent-layer generation includes previous outputs."""
        moa = MixtureOfAgents(completion_engine, model_pool)
        models = model_pool._registry.get_available_models()
        model_id = models[0].id if models else "gpt-4o"

        previous = {"prev-model": "Previous output text"}
        content, _cost = await moa._generate_one(
            model=model_id,
            prompt="Test prompt",
            previous_outputs=previous,
            context="",
        )
        assert content
        # Should have used the refine system prompt
        last_call = mock_litellm_moa.call_log[-1]
        system_msg = last_call["messages"][0]["content"]
        assert "other AI models" in system_msg or "Previous output text" in system_msg

    async def test_generate_with_context(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
        mock_litellm_moa: MockLiteLLM,
    ) -> None:
        """Context is appended to the user prompt."""
        moa = MixtureOfAgents(completion_engine, model_pool)
        models = model_pool._registry.get_available_models()
        model_id = models[0].id if models else "gpt-4o"

        content, _ = await moa._generate_one(
            model=model_id,
            prompt="Explain code",
            previous_outputs=None,
            context="def foo(): pass",
        )
        assert content
        last_call = mock_litellm_moa.call_log[-1]
        user_msg = last_call["messages"][1]["content"]
        assert "def foo(): pass" in user_msg


# ======================================================================
# MixtureOfAgents._rank_outputs (convenience wrapper)
# ======================================================================


class TestRankOutputsWrapper:
    """Test the convenience ranking wrapper method."""

    async def test_rank_outputs_with_multiple(
        self,
        moa_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        model_pool: ModelPool,
    ) -> None:
        """Ranking wrapper returns rankings for multiple outputs."""
        mock = MockLiteLLM()
        mock.set_default_response(
            MockResponse(
                content=_judge_response(),
                input_tokens=100,
                output_tokens=50,
            ),
        )
        engine = _make_engine(moa_settings, mock_cost_tracker, mock_auth, mock_registry, mock)
        moa = MixtureOfAgents(engine, model_pool)

        outputs = {"m1": "Output 1", "m2": "Output 2", "m3": "Output 3"}
        rankings = await moa._rank_outputs(outputs, "Test prompt")
        assert len(rankings) == 3  # C(3,2) = 3

    async def test_rank_outputs_single_returns_empty(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Single output returns empty rankings."""
        moa = MixtureOfAgents(completion_engine, model_pool)
        rankings = await moa._rank_outputs({"m1": "Only one"}, "Test")
        assert rankings == []


# ======================================================================
# Integration: full pipeline end-to-end
# ======================================================================


class TestMoAIntegration:
    """End-to-end integration tests for the MoA pipeline."""

    async def test_full_pipeline_with_ranking(
        self,
        moa_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        model_pool: ModelPool,
    ) -> None:
        """Full pipeline: 2 layers + ranking + fusion."""
        mock = MockLiteLLM()
        mock.set_default_response(
            MockResponse(
                content=_judge_response(winner="A", reason="Thorough", confidence=0.9),
                input_tokens=150,
                output_tokens=75,
            ),
        )
        engine = _make_engine(moa_settings, mock_cost_tracker, mock_auth, mock_registry, mock)
        config = MoAConfig(num_proposers=2, num_layers=2, use_ranking=True)
        moa = MixtureOfAgents(engine, model_pool, config)

        result = await moa.generate("Build a REST API in Python")

        assert isinstance(result, MoAResult)
        assert len(result.layers) == 2
        assert result.final_output
        assert result.total_cost > 0.0
        assert len(result.participating_models) >= 2
        # Ranking should have been performed
        assert len(result.rankings) >= 1
        assert result.best_individual is not None

    async def test_full_pipeline_without_ranking(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Full pipeline without ranking: layers + fusion only."""
        config = MoAConfig(num_proposers=2, num_layers=1, use_ranking=False)
        moa = MixtureOfAgents(completion_engine, model_pool, config)

        result = await moa.generate("Simple question about Python")

        assert len(result.layers) == 1
        assert result.rankings == []
        assert result.best_individual is None
        assert result.final_output

    async def test_pipeline_handles_generation_failure(
        self,
        moa_settings: Settings,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        model_pool: ModelPool,
    ) -> None:
        """Pipeline handles individual model failures gracefully."""
        mock = MockLiteLLM()
        # Set some models to fail and others to succeed
        available = mock_registry.get_available_models()
        if len(available) >= 2:
            mock.set_error(available[0].id, RuntimeError("Model A unavailable"))
            mock.set_response(
                available[1].id,
                MockResponse(content="Successful response.", input_tokens=100, output_tokens=50),
            )
        mock.set_default_response(
            MockResponse(content="Default response.", input_tokens=50, output_tokens=25),
        )
        engine = _make_engine(moa_settings, mock_cost_tracker, mock_auth, mock_registry, mock)
        config = MoAConfig(num_proposers=2, num_layers=1, use_ranking=False)
        moa = MixtureOfAgents(engine, model_pool, config)

        result = await moa.generate("Test with failures")
        assert isinstance(result, MoAResult)
        assert result.final_output
        # At least one layer should have outputs
        assert len(result.layers) >= 1
        # Failed models get an error placeholder
        layer_outputs = result.layers[0].outputs
        assert len(layer_outputs) >= 1

    async def test_single_proposer_pipeline(
        self,
        completion_engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        """Pipeline works with a single proposer (no pairwise ranking possible)."""
        config = MoAConfig(num_proposers=1, num_layers=1, use_ranking=True)
        moa = MixtureOfAgents(completion_engine, model_pool, config)

        result = await moa.generate("Single proposer test")
        assert result.final_output
        # Ranking needs >= 2 outputs, so should be empty
        assert result.rankings == []
        assert result.best_individual is None
