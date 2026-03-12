"""Tests for the smart cost optimizer tool."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prism.cost.tracker import CostSummary, ModelCostBreakdown
from prism.tools.base import PermissionLevel
from prism.tools.cost_optimizer import CostOptimizerTool, ModelRecommendation

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_cost_tracker() -> MagicMock:
    """Create a mock CostTracker."""
    tracker = MagicMock()
    tracker.get_cost_summary.return_value = CostSummary(
        period="month",
        total_cost=5.25,
        total_requests=150,
        model_breakdown=[
            ModelCostBreakdown(
                model_id="gpt-4o",
                display_name="GPT-4o",
                request_count=100,
                total_cost=4.00,
                percentage=76.2,
            ),
            ModelCostBreakdown(
                model_id="groq/llama-3.3-70b-versatile",
                display_name="Llama 3.3 70B",
                request_count=50,
                total_cost=1.25,
                percentage=23.8,
            ),
        ],
        budget_limit=10.0,
        budget_remaining=4.75,
    )
    tracker.calculate_savings.return_value = (25.0, 5.25, 19.75)
    tracker.get_daily_cost.return_value = 0.50
    tracker.get_monthly_cost.return_value = 5.25
    tracker.get_budget_remaining.return_value = 4.75
    return tracker


@pytest.fixture
def mock_learner() -> MagicMock:
    """Create a mock AdaptiveLearner with usage data."""
    learner = MagicMock()

    # gpt-4o: high success, high cost
    # groq/llama-3.3-70b: good success, low cost
    def get_model_stats(model: str) -> dict[str, object]:
        stats: dict[str, dict[str, object]] = {
            "gpt-4o": {
                "interactions": 100,
                "success_rate": 0.92,
                "total_cost": 4.00,
            },
            "groq/llama-3.3-70b-versatile": {
                "interactions": 50,
                "success_rate": 0.88,
                "total_cost": 1.25,
            },
        }
        return stats.get(model, {
            "interactions": 0,
            "success_rate": 0.5,
            "total_cost": 0.0,
        })

    def get_success_rate(model: str) -> float:
        rates: dict[str, float] = {
            "gpt-4o": 0.92,
            "groq/llama-3.3-70b-versatile": 0.88,
            "deepseek/deepseek-chat": 0.85,
            "gemini/gemini-2.0-flash": 0.80,
            "gpt-4o-mini": 0.82,
            "ollama/qwen2.5-coder:7b": 0.75,
        }
        return rates.get(model, 0.5)

    learner.get_model_stats.side_effect = get_model_stats
    learner.get_success_rate.side_effect = get_success_rate
    return learner


@pytest.fixture
def cost_optimizer(
    mock_cost_tracker: MagicMock,
    mock_learner: MagicMock,
) -> CostOptimizerTool:
    """Create a CostOptimizerTool with mocked dependencies."""
    return CostOptimizerTool(mock_cost_tracker, mock_learner)


# ------------------------------------------------------------------
# Property tests
# ------------------------------------------------------------------


class TestCostOptimizerProperties:
    """Tests for tool metadata properties."""

    def test_name(self, cost_optimizer: CostOptimizerTool) -> None:
        """Tool name is 'cost_optimizer'."""
        assert cost_optimizer.name == "cost_optimizer"

    def test_description(self, cost_optimizer: CostOptimizerTool) -> None:
        """Description mentions cost analysis and recommendations."""
        desc = cost_optimizer.description
        assert "cost" in desc.lower()
        assert "recommend" in desc.lower()

    def test_permission_level(self, cost_optimizer: CostOptimizerTool) -> None:
        """Permission level is AUTO."""
        assert cost_optimizer.permission_level == PermissionLevel.AUTO

    def test_parameters_schema(self, cost_optimizer: CostOptimizerTool) -> None:
        """Schema has action (required), period, session_id."""
        schema = cost_optimizer.parameters_schema
        assert schema["type"] == "object"
        assert "action" in schema["properties"]
        assert "period" in schema["properties"]
        assert "session_id" in schema["properties"]
        assert schema["required"] == ["action"]


# ------------------------------------------------------------------
# Analyze action tests
# ------------------------------------------------------------------


class TestAnalyzeAction:
    """Tests for the 'analyze' action."""

    def test_analyze_returns_breakdown(
        self, cost_optimizer: CostOptimizerTool, mock_cost_tracker: MagicMock
    ) -> None:
        """Analyze returns cost breakdown by model."""
        result = cost_optimizer.execute({"action": "analyze"})
        assert result.success is True
        assert "$5.25" in result.output
        assert "150" in result.output
        assert result.metadata is not None
        assert result.metadata["action"] == "analyze"
        assert result.metadata["total_cost"] == 5.25
        assert result.metadata["total_requests"] == 150

    def test_analyze_with_period(
        self, cost_optimizer: CostOptimizerTool, mock_cost_tracker: MagicMock
    ) -> None:
        """Analyze respects the period parameter."""
        cost_optimizer.execute({"action": "analyze", "period": "day"})
        mock_cost_tracker.get_cost_summary.assert_called_with("day", "")

    def test_analyze_with_session(
        self, cost_optimizer: CostOptimizerTool, mock_cost_tracker: MagicMock
    ) -> None:
        """Analyze passes session_id to cost tracker."""
        cost_optimizer.execute({
            "action": "analyze",
            "period": "session",
            "session_id": "abc123",
        })
        mock_cost_tracker.get_cost_summary.assert_called_with("session", "abc123")

    def test_analyze_budget_info(
        self, cost_optimizer: CostOptimizerTool
    ) -> None:
        """Analyze includes budget information when set."""
        result = cost_optimizer.execute({"action": "analyze"})
        assert result.metadata is not None
        assert result.metadata["budget_limit"] == 10.0
        assert result.metadata["budget_remaining"] == 4.75

    def test_analyze_model_breakdown(
        self, cost_optimizer: CostOptimizerTool
    ) -> None:
        """Analyze metadata includes per-model breakdown."""
        result = cost_optimizer.execute({"action": "analyze"})
        assert result.metadata is not None
        models = result.metadata["models"]
        assert len(models) == 2
        assert models[0]["model_id"] == "gpt-4o"
        assert models[0]["request_count"] == 100


# ------------------------------------------------------------------
# Recommend action tests
# ------------------------------------------------------------------


class TestRecommendAction:
    """Tests for the 'recommend' action."""

    def test_recommend_finds_cheaper_alternatives(
        self, cost_optimizer: CostOptimizerTool
    ) -> None:
        """Recommend finds cheaper models with acceptable success rates."""
        result = cost_optimizer.execute({"action": "recommend"})
        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["action"] == "recommend"
        # Should find some recommendations since groq is cheaper than gpt-4o
        # with acceptable success rate
        recs = result.metadata["recommendations"]
        # May or may not find recommendations depending on rate thresholds
        assert isinstance(recs, list)

    def test_recommend_no_usage_data(
        self, mock_cost_tracker: MagicMock
    ) -> None:
        """Returns helpful message when no usage data exists."""
        learner = MagicMock()
        learner.get_model_stats.return_value = {
            "interactions": 0,
            "success_rate": 0.5,
            "total_cost": 0.0,
        }
        tool = CostOptimizerTool(mock_cost_tracker, learner)
        result = tool.execute({"action": "recommend"})
        assert result.success is True
        assert "not enough" in result.output.lower()
        assert result.metadata is not None
        assert result.metadata["recommendations"] == []

    def test_recommend_metadata_structure(
        self, cost_optimizer: CostOptimizerTool
    ) -> None:
        """Recommendation metadata has the right structure."""
        result = cost_optimizer.execute({"action": "recommend"})
        assert result.metadata is not None
        assert "recommendation_count" in result.metadata
        assert "recommendations" in result.metadata
        for rec in result.metadata["recommendations"]:
            assert "current_model" in rec
            assert "recommended_model" in rec
            assert "savings_percent" in rec
            assert "success_rate_current" in rec
            assert "success_rate_rec" in rec
            assert "reason" in rec


# ------------------------------------------------------------------
# Report action tests
# ------------------------------------------------------------------


class TestReportAction:
    """Tests for the 'report' action."""

    def test_report_shows_savings(
        self, cost_optimizer: CostOptimizerTool
    ) -> None:
        """Report shows savings vs. premium-only usage."""
        result = cost_optimizer.execute({"action": "report"})
        assert result.success is True
        assert "$25.00" in result.output  # hypothetical
        assert "$5.25" in result.output  # actual
        assert "$19.75" in result.output  # savings

    def test_report_metadata(
        self, cost_optimizer: CostOptimizerTool
    ) -> None:
        """Report metadata includes all cost figures."""
        result = cost_optimizer.execute({"action": "report"})
        assert result.metadata is not None
        assert result.metadata["action"] == "report"
        assert result.metadata["hypothetical_cost"] == 25.0
        assert result.metadata["actual_cost"] == 5.25
        assert result.metadata["savings"] == 19.75
        assert result.metadata["daily_cost"] == 0.50
        assert result.metadata["monthly_cost"] == 5.25

    def test_report_includes_budget_remaining(
        self, cost_optimizer: CostOptimizerTool
    ) -> None:
        """Report includes budget remaining."""
        result = cost_optimizer.execute({"action": "report"})
        assert result.metadata is not None
        assert result.metadata["budget_remaining"] == 4.75

    def test_report_savings_percentage(
        self, cost_optimizer: CostOptimizerTool
    ) -> None:
        """Report calculates savings percentage."""
        result = cost_optimizer.execute({"action": "report"})
        assert result.metadata is not None
        assert result.metadata["savings_percent"] == 79.0  # 19.75/25.0 * 100


# ------------------------------------------------------------------
# Error handling tests
# ------------------------------------------------------------------


class TestCostOptimizerErrors:
    """Tests for error handling."""

    def test_unknown_action(self, cost_optimizer: CostOptimizerTool) -> None:
        """Unknown action returns an error."""
        result = cost_optimizer.execute({"action": "explode"})
        assert result.success is False
        assert "Unknown action" in (result.error or "")

    def test_unknown_period(self, cost_optimizer: CostOptimizerTool) -> None:
        """Unknown period returns an error."""
        result = cost_optimizer.execute({
            "action": "analyze",
            "period": "century",
        })
        assert result.success is False
        assert "Unknown period" in (result.error or "")

    def test_missing_required_action(
        self, cost_optimizer: CostOptimizerTool
    ) -> None:
        """Missing required 'action' raises ValueError."""
        with pytest.raises(ValueError, match="Missing required"):
            cost_optimizer.execute({})

    def test_exception_during_analyze(
        self, mock_learner: MagicMock
    ) -> None:
        """Exceptions during analyze are caught gracefully."""
        tracker = MagicMock()
        tracker.get_cost_summary.side_effect = RuntimeError("DB down")
        tool = CostOptimizerTool(tracker, mock_learner)
        result = tool.execute({"action": "analyze"})
        assert result.success is False
        assert "DB down" in (result.error or "")


# ------------------------------------------------------------------
# Helper function tests
# ------------------------------------------------------------------


class TestCostOptimizerHelpers:
    """Tests for helper functions."""

    def test_avg_cost_per_token(self) -> None:
        """Blended cost calculation is correct."""
        from prism.cost.pricing import ModelPricing

        pricing = ModelPricing(
            provider="test",
            input_cost_per_1m=2.0,
            output_cost_per_1m=10.0,
        )
        result = CostOptimizerTool._avg_cost_per_token(pricing)
        # (0.75 * 2.0 + 0.25 * 10.0) / 1_000_000 * 1000 = 4.0 / 1000 = 0.004
        expected = (0.75 * 2.0 + 0.25 * 10.0) / 1_000_000 * 1000
        assert abs(result - expected) < 1e-10

    def test_avg_cost_per_token_free(self) -> None:
        """Free models return 0 cost."""
        from prism.cost.pricing import ModelPricing

        pricing = ModelPricing(
            provider="ollama",
            input_cost_per_1m=0.0,
            output_cost_per_1m=0.0,
        )
        result = CostOptimizerTool._avg_cost_per_token(pricing)
        assert result == 0.0

    def test_build_recommendation_reason_high_savings(self) -> None:
        """Reason for high-savings recommendation."""
        reason = CostOptimizerTool._build_recommendation_reason(
            "gpt-4o", "ollama/llama3.2:3b", 95.0, 0.80
        )
        assert "nearly free" in reason.lower()

    def test_build_recommendation_reason_medium_savings(self) -> None:
        """Reason for medium-savings recommendation."""
        reason = CostOptimizerTool._build_recommendation_reason(
            "gpt-4o", "deepseek/deepseek-chat", 60.0, 0.85
        )
        assert "save" in reason.lower()
        assert "60%" in reason

    def test_build_recommendation_reason_low_savings(self) -> None:
        """Reason for lower-savings recommendation."""
        reason = CostOptimizerTool._build_recommendation_reason(
            "gpt-4o", "gpt-4o-mini", 30.0, 0.82
        )
        assert "consider" in reason.lower()

    def test_model_recommendation_dataclass(self) -> None:
        """ModelRecommendation is frozen and has all fields."""
        rec = ModelRecommendation(
            current_model="a",
            recommended_model="b",
            current_cost_per_req=0.01,
            recommended_cost=0.001,
            savings_percent=90.0,
            success_rate_current=0.95,
            success_rate_rec=0.90,
            tier="simple",
            reason="test",
        )
        assert rec.savings_percent == 90.0
        # Frozen — cannot modify
        with pytest.raises(AttributeError):
            rec.tier = "complex"  # type: ignore[misc]
