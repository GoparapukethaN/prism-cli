"""Tests for Phase 5 AEI enhancements — new strategies, escalation rules, explain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from prism.intelligence.aei import (
    ESCALATION_THRESHOLD,
    STRATEGY_ORDER,
    AdaptiveExecutionIntelligence,
    ErrorFingerprint,
    FixStrategy,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Return a temporary database path."""
    return tmp_path / "aei_phase5" / "test.db"


@pytest.fixture
def aei(db_path: Path) -> AdaptiveExecutionIntelligence:
    """Create a fresh AEI instance."""
    engine = AdaptiveExecutionIntelligence(db_path=db_path, repo="test-repo")
    yield engine
    engine.close()


@pytest.fixture
def sample_fingerprint() -> ErrorFingerprint:
    """A reusable error fingerprint for tests."""
    return AdaptiveExecutionIntelligence.fingerprint_error(
        error_type="TypeError",
        stack_trace='File "src/main.py", line 42, in process\n  return x + y',
        file_path="src/main.py",
        function_name="process",
    )


# ======================================================================
# TestNewFixStrategyValues
# ======================================================================


class TestNewFixStrategyValues:
    """Tests for the two new FixStrategy enum members."""

    def test_add_defensive_code_exists(self) -> None:
        """ADD_DEFENSIVE_CODE is a valid FixStrategy member."""
        assert FixStrategy.ADD_DEFENSIVE_CODE.value == "add_defensive_code"

    def test_revert_and_redesign_exists(self) -> None:
        """REVERT_AND_REDESIGN is a valid FixStrategy member."""
        assert FixStrategy.REVERT_AND_REDESIGN.value == "revert_and_redesign"

    def test_add_defensive_code_constructible_from_value(self) -> None:
        """ADD_DEFENSIVE_CODE can be reconstructed from its string value."""
        assert FixStrategy("add_defensive_code") is FixStrategy.ADD_DEFENSIVE_CODE

    def test_revert_and_redesign_constructible_from_value(self) -> None:
        """REVERT_AND_REDESIGN can be reconstructed from its string value."""
        assert FixStrategy("revert_and_redesign") is FixStrategy.REVERT_AND_REDESIGN

    def test_total_strategy_count(self) -> None:
        """There are now 9 strategies in total."""
        assert len(FixStrategy) == 9


# ======================================================================
# TestStrategyOrder
# ======================================================================


class TestStrategyOrderEnhancements:
    """Tests for STRATEGY_ORDER with new strategies in correct position."""

    def test_strategy_order_includes_all(self) -> None:
        """STRATEGY_ORDER covers every FixStrategy member."""
        assert set(STRATEGY_ORDER) == set(FixStrategy)

    def test_strategy_order_length(self) -> None:
        """STRATEGY_ORDER has 9 entries."""
        assert len(STRATEGY_ORDER) == 9

    def test_add_defensive_code_after_context_expand(self) -> None:
        """ADD_DEFENSIVE_CODE comes right after CONTEXT_EXPAND."""
        idx_ctx = STRATEGY_ORDER.index(FixStrategy.CONTEXT_EXPAND)
        idx_def = STRATEGY_ORDER.index(FixStrategy.ADD_DEFENSIVE_CODE)
        assert idx_def == idx_ctx + 1

    def test_add_defensive_code_before_model_escalate(self) -> None:
        """ADD_DEFENSIVE_CODE comes before MODEL_ESCALATE."""
        idx_def = STRATEGY_ORDER.index(FixStrategy.ADD_DEFENSIVE_CODE)
        idx_esc = STRATEGY_ORDER.index(FixStrategy.MODEL_ESCALATE)
        assert idx_def < idx_esc

    def test_revert_and_redesign_after_decompose(self) -> None:
        """REVERT_AND_REDESIGN comes right after DECOMPOSE_SUBTASKS."""
        idx_dec = STRATEGY_ORDER.index(FixStrategy.DECOMPOSE_SUBTASKS)
        idx_rev = STRATEGY_ORDER.index(FixStrategy.REVERT_AND_REDESIGN)
        assert idx_rev == idx_dec + 1

    def test_revert_and_redesign_before_debate(self) -> None:
        """REVERT_AND_REDESIGN comes before MULTI_MODEL_DEBATE."""
        idx_rev = STRATEGY_ORDER.index(FixStrategy.REVERT_AND_REDESIGN)
        idx_deb = STRATEGY_ORDER.index(FixStrategy.MULTI_MODEL_DEBATE)
        assert idx_rev < idx_deb

    def test_starts_with_regex_patch(self) -> None:
        """STRATEGY_ORDER still starts with REGEX_PATCH."""
        assert STRATEGY_ORDER[0] == FixStrategy.REGEX_PATCH

    def test_ends_with_multi_model_debate(self) -> None:
        """STRATEGY_ORDER still ends with MULTI_MODEL_DEBATE."""
        assert STRATEGY_ORDER[-1] == FixStrategy.MULTI_MODEL_DEBATE


# ======================================================================
# TestEscalationFullRewrite
# ======================================================================


class TestEscalationFullRewrite:
    """3 full_rewrite failures -> context_expand + add_defensive_code."""

    def test_full_rewrite_escalation_to_defensive_code(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """After 3 full_rewrite failures, recommends add_defensive_code."""
        # Exhaust regex_patch, ast_diff first so full_rewrite is reached
        for s in [
            FixStrategy.REGEX_PATCH,
            FixStrategy.AST_DIFF,
        ]:
            for _ in range(ESCALATION_THRESHOLD):
                aei.record_attempt(
                    sample_fingerprint, s, "gpt-4o-mini", 2000, "failure"
                )

        # Now exhaust full_rewrite
        for _ in range(ESCALATION_THRESHOLD):
            aei.record_attempt(
                sample_fingerprint,
                FixStrategy.FULL_REWRITE,
                "gpt-4o-mini",
                2000,
                "failure",
            )

        rec = aei.recommend_strategy(sample_fingerprint)
        assert rec.strategy == FixStrategy.ADD_DEFENSIVE_CODE

    def test_full_rewrite_escalation_includes_context_expand(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """After 3 full_rewrite failures, context multiplier is >= 2x."""
        for s in [
            FixStrategy.REGEX_PATCH,
            FixStrategy.AST_DIFF,
        ]:
            for _ in range(ESCALATION_THRESHOLD):
                aei.record_attempt(
                    sample_fingerprint, s, "gpt-4o-mini", 2000, "failure"
                )

        for _ in range(ESCALATION_THRESHOLD):
            aei.record_attempt(
                sample_fingerprint,
                FixStrategy.FULL_REWRITE,
                "gpt-4o-mini",
                2000,
                "failure",
            )

        rec = aei.recommend_strategy(sample_fingerprint)
        assert rec.context_multiplier >= 2.0

    def test_full_rewrite_escalation_reasoning(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Reasoning mentions full_rewrite -> add_defensive_code."""
        for s in [
            FixStrategy.REGEX_PATCH,
            FixStrategy.AST_DIFF,
        ]:
            for _ in range(ESCALATION_THRESHOLD):
                aei.record_attempt(
                    sample_fingerprint, s, "gpt-4o-mini", 2000, "failure"
                )

        for _ in range(ESCALATION_THRESHOLD):
            aei.record_attempt(
                sample_fingerprint,
                FixStrategy.FULL_REWRITE,
                "gpt-4o-mini",
                2000,
                "failure",
            )

        rec = aei.recommend_strategy(sample_fingerprint)
        assert "full_rewrite" in rec.reasoning.lower()
        assert "add_defensive_code" in rec.reasoning.lower()


# ======================================================================
# TestEscalationDecompose
# ======================================================================


class TestEscalationDecompose:
    """3 decompose failures -> revert_and_redesign before debate."""

    def test_decompose_escalation_to_revert_and_redesign(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """After 3 decompose failures, recommends revert_and_redesign."""
        # Exhaust all strategies up to decompose
        for s in STRATEGY_ORDER:
            if s == FixStrategy.DECOMPOSE_SUBTASKS:
                break
            for _ in range(ESCALATION_THRESHOLD):
                aei.record_attempt(
                    sample_fingerprint, s, "gpt-4o-mini", 2000, "failure"
                )

        for _ in range(ESCALATION_THRESHOLD):
            aei.record_attempt(
                sample_fingerprint,
                FixStrategy.DECOMPOSE_SUBTASKS,
                "gpt-4o-mini",
                2000,
                "failure",
            )

        rec = aei.recommend_strategy(sample_fingerprint)
        assert rec.strategy == FixStrategy.REVERT_AND_REDESIGN

    def test_decompose_escalation_reasoning(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Reasoning mentions decompose -> revert_and_redesign."""
        for s in STRATEGY_ORDER:
            if s == FixStrategy.DECOMPOSE_SUBTASKS:
                break
            for _ in range(ESCALATION_THRESHOLD):
                aei.record_attempt(
                    sample_fingerprint, s, "gpt-4o-mini", 2000, "failure"
                )

        for _ in range(ESCALATION_THRESHOLD):
            aei.record_attempt(
                sample_fingerprint,
                FixStrategy.DECOMPOSE_SUBTASKS,
                "gpt-4o-mini",
                2000,
                "failure",
            )

        rec = aei.recommend_strategy(sample_fingerprint)
        assert "decompose" in rec.reasoning.lower()
        assert "revert_and_redesign" in rec.reasoning.lower()


# ======================================================================
# TestExplainMethod
# ======================================================================


class TestExplainMethod:
    """Tests for the explain() method."""

    def test_explain_no_history(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Explain returns meaningful text even with no history."""
        result = aei.explain(sample_fingerprint)
        assert "Past attempts: 0" in result
        assert "Successes: 0" in result
        assert "Failures: 0" in result
        assert "Current recommendation:" in result
        assert "Confidence:" in result

    def test_explain_includes_fingerprint_details(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Explain includes the fingerprint hash and error type."""
        result = aei.explain(sample_fingerprint)
        assert sample_fingerprint.fingerprint_hash in result
        assert sample_fingerprint.error_type in result

    def test_explain_with_attempts(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Explain reflects recorded attempts."""
        aei.record_attempt(
            sample_fingerprint,
            FixStrategy.REGEX_PATCH,
            "gpt-4o-mini",
            2000,
            "failure",
        )
        aei.record_attempt(
            sample_fingerprint,
            FixStrategy.AST_DIFF,
            "gpt-4o",
            3000,
            "success",
        )
        result = aei.explain(sample_fingerprint)
        assert "Past attempts: 2" in result
        assert "Successes: 1" in result
        assert "Failures: 1" in result
        assert "Strategy history:" in result
        assert "regex_patch" in result
        assert "ast_diff" in result

    def test_explain_shows_confidence_level(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Explain includes a confidence level label."""
        result = aei.explain(sample_fingerprint)
        assert any(
            level in result for level in ("high", "moderate", "low")
        )

    def test_explain_shows_model_tier(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Explain includes the model tier."""
        result = aei.explain(sample_fingerprint)
        assert "Model tier:" in result

    def test_explain_shows_reasoning(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Explain includes the reasoning string."""
        result = aei.explain(sample_fingerprint)
        assert "Reasoning:" in result

    def test_explain_multiline_output(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Explain returns a multi-line string."""
        result = aei.explain(sample_fingerprint)
        lines = result.strip().split("\n")
        assert len(lines) >= 8


# ======================================================================
# TestBackwardsCompatibility
# ======================================================================


class TestBackwardsCompatibility:
    """Verify that new strategies do not break existing behaviour."""

    def test_first_attempt_still_cheapest(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """First attempt still defaults to REGEX_PATCH."""
        rec = aei.recommend_strategy(sample_fingerprint)
        assert rec.strategy == FixStrategy.REGEX_PATCH

    def test_regex_escalation_still_works(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """3 regex failures still escalate to AST_DIFF."""
        for _ in range(ESCALATION_THRESHOLD):
            aei.record_attempt(
                sample_fingerprint,
                FixStrategy.REGEX_PATCH,
                "gpt-4o-mini",
                2000,
                "failure",
            )
        rec = aei.recommend_strategy(sample_fingerprint)
        assert rec.strategy == FixStrategy.AST_DIFF

    def test_success_reuse_still_works(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """A past success is still re-used."""
        aei.record_attempt(
            sample_fingerprint,
            FixStrategy.FULL_REWRITE,
            "gpt-4o",
            5000,
            "success",
        )
        rec = aei.recommend_strategy(sample_fingerprint)
        assert rec.strategy == FixStrategy.FULL_REWRITE
        assert rec.confidence == 0.8

    def test_all_exhausted_still_debate(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """When every strategy has 3+ failures, MULTI_MODEL_DEBATE is used."""
        for s in STRATEGY_ORDER:
            for _ in range(ESCALATION_THRESHOLD):
                aei.record_attempt(
                    sample_fingerprint, s, "gpt-4o-mini", 2000, "failure"
                )
        rec = aei.recommend_strategy(sample_fingerprint)
        assert rec.strategy == FixStrategy.MULTI_MODEL_DEBATE

    def test_record_new_strategies(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """New strategies can be recorded without errors."""
        rec1 = aei.record_attempt(
            sample_fingerprint,
            FixStrategy.ADD_DEFENSIVE_CODE,
            "gpt-4o",
            4000,
            "success",
        )
        rec2 = aei.record_attempt(
            sample_fingerprint,
            FixStrategy.REVERT_AND_REDESIGN,
            "gpt-4o",
            5000,
            "failure",
        )
        assert rec1.strategy == "add_defensive_code"
        assert rec2.strategy == "revert_and_redesign"
