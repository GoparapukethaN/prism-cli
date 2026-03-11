"""Tests for prism.intelligence.aei — Adaptive Execution Intelligence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from prism.intelligence.aei import (
    ESCALATION_THRESHOLD,
    MODEL_TIERS,
    STRATEGY_ORDER,
    AdaptiveExecutionIntelligence,
    AEIStats,
    AttemptRecord,
    ErrorFingerprint,
    FixStrategy,
    StrategyRecommendation,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Return a temporary database path."""
    return tmp_path / "aei" / "test.db"


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


@pytest.fixture
def other_fingerprint() -> ErrorFingerprint:
    """A distinct fingerprint for comparison tests."""
    return AdaptiveExecutionIntelligence.fingerprint_error(
        error_type="KeyError",
        stack_trace='File "src/config.py", line 10, in load\n  return d["key"]',
        file_path="src/config.py",
        function_name="load",
    )


# ======================================================================
# TestFixStrategy
# ======================================================================


class TestFixStrategy:
    """Tests for the FixStrategy enum."""

    def test_all_values_exist(self) -> None:
        """All expected strategy values are present."""
        expected = {
            "regex_patch",
            "ast_diff",
            "full_rewrite",
            "context_expand",
            "add_defensive_code",
            "model_escalate",
            "decompose_subtasks",
            "revert_and_redesign",
            "multi_model_debate",
        }
        actual = {s.value for s in FixStrategy}
        assert actual == expected

    def test_enum_member_count(self) -> None:
        """There are exactly 9 strategies."""
        assert len(FixStrategy) == 9

    def test_strategy_order_contains_all(self) -> None:
        """STRATEGY_ORDER covers every FixStrategy member."""
        assert set(STRATEGY_ORDER) == set(FixStrategy)

    def test_strategy_order_starts_with_cheapest(self) -> None:
        """The cheapest strategy is first in the order."""
        assert STRATEGY_ORDER[0] == FixStrategy.REGEX_PATCH

    def test_strategy_order_ends_with_debate(self) -> None:
        """Multi-Model Debate is the last resort."""
        assert STRATEGY_ORDER[-1] == FixStrategy.MULTI_MODEL_DEBATE

    def test_strategy_from_value(self) -> None:
        """Strategies can be reconstructed from their string value."""
        for s in FixStrategy:
            assert FixStrategy(s.value) is s


# ======================================================================
# TestErrorFingerprint
# ======================================================================


class TestErrorFingerprint:
    """Tests for ErrorFingerprint creation and hashing."""

    def test_hash_determinism(self) -> None:
        """Same inputs produce the same fingerprint hash."""
        fp1 = ErrorFingerprint(
            error_type="TypeError",
            stack_pattern="line N in foo",
            file_path="a.py",
            function_name="foo",
        )
        fp2 = ErrorFingerprint(
            error_type="TypeError",
            stack_pattern="line N in foo",
            file_path="a.py",
            function_name="foo",
        )
        assert fp1.fingerprint_hash == fp2.fingerprint_hash
        assert len(fp1.fingerprint_hash) == 16

    def test_different_inputs_different_hashes(self) -> None:
        """Different error signatures produce different hashes."""
        fp1 = ErrorFingerprint(
            error_type="TypeError",
            stack_pattern="pattern_a",
            file_path="a.py",
            function_name="foo",
        )
        fp2 = ErrorFingerprint(
            error_type="KeyError",
            stack_pattern="pattern_a",
            file_path="a.py",
            function_name="foo",
        )
        assert fp1.fingerprint_hash != fp2.fingerprint_hash

    def test_different_file_path_different_hash(self) -> None:
        """Different file paths produce different hashes."""
        fp1 = ErrorFingerprint(
            error_type="TypeError",
            stack_pattern="p",
            file_path="a.py",
            function_name="foo",
        )
        fp2 = ErrorFingerprint(
            error_type="TypeError",
            stack_pattern="p",
            file_path="b.py",
            function_name="foo",
        )
        assert fp1.fingerprint_hash != fp2.fingerprint_hash

    def test_different_function_different_hash(self) -> None:
        """Different function names produce different hashes."""
        fp1 = ErrorFingerprint(
            error_type="TypeError",
            stack_pattern="p",
            file_path="a.py",
            function_name="foo",
        )
        fp2 = ErrorFingerprint(
            error_type="TypeError",
            stack_pattern="p",
            file_path="a.py",
            function_name="bar",
        )
        assert fp1.fingerprint_hash != fp2.fingerprint_hash

    def test_frozen_dataclass(self) -> None:
        """ErrorFingerprint is immutable."""
        fp = ErrorFingerprint(
            error_type="TypeError",
            stack_pattern="p",
            file_path="a.py",
            function_name="foo",
        )
        with pytest.raises(AttributeError):
            fp.error_type = "KeyError"  # type: ignore[misc]

    def test_explicit_hash_not_overridden(self) -> None:
        """An explicitly provided hash is preserved."""
        fp = ErrorFingerprint(
            error_type="TypeError",
            stack_pattern="p",
            file_path="a.py",
            function_name="foo",
            fingerprint_hash="explicit_hash_val",
        )
        assert fp.fingerprint_hash == "explicit_hash_val"

    def test_hash_is_hex(self) -> None:
        """The auto-computed hash is a valid hex string."""
        fp = ErrorFingerprint(
            error_type="ValueError",
            stack_pattern="trace",
            file_path="x.py",
            function_name="run",
        )
        int(fp.fingerprint_hash, 16)  # Raises if not valid hex


# ======================================================================
# TestStrategyRecommendation
# ======================================================================


class TestStrategyRecommendation:
    """Tests for the StrategyRecommendation dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible."""
        rec = StrategyRecommendation(
            strategy=FixStrategy.REGEX_PATCH,
            model_tier="cheap",
            context_multiplier=1.0,
            reasoning="test",
            past_attempts=0,
            past_successes=0,
            confidence=0.5,
        )
        assert rec.strategy == FixStrategy.REGEX_PATCH
        assert rec.model_tier == "cheap"
        assert rec.context_multiplier == 1.0
        assert rec.reasoning == "test"
        assert rec.past_attempts == 0
        assert rec.past_successes == 0
        assert rec.confidence == 0.5

    def test_confidence_range(self) -> None:
        """Confidence can be set to boundary values."""
        low = StrategyRecommendation(
            strategy=FixStrategy.AST_DIFF,
            model_tier="mid",
            context_multiplier=2.0,
            reasoning="low",
            past_attempts=10,
            past_successes=0,
            confidence=0.0,
        )
        high = StrategyRecommendation(
            strategy=FixStrategy.AST_DIFF,
            model_tier="mid",
            context_multiplier=1.0,
            reasoning="high",
            past_attempts=10,
            past_successes=10,
            confidence=1.0,
        )
        assert low.confidence == 0.0
        assert high.confidence == 1.0


# ======================================================================
# TestAttemptRecord
# ======================================================================


class TestAttemptRecord:
    """Tests for the AttemptRecord dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible and defaults are applied."""
        rec = AttemptRecord(
            id=1,
            repo="my-repo",
            fingerprint="abc123",
            strategy="regex_patch",
            model="gpt-4o-mini",
            context_size=2000,
            outcome="success",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        assert rec.id == 1
        assert rec.repo == "my-repo"
        assert rec.error_type == ""
        assert rec.reasoning == ""

    def test_optional_fields(self) -> None:
        """Optional fields can be set explicitly."""
        rec = AttemptRecord(
            id=2,
            repo="other",
            fingerprint="def456",
            strategy="ast_diff",
            model="gpt-4o",
            context_size=4000,
            outcome="failure",
            timestamp="2025-06-15T12:00:00+00:00",
            error_type="KeyError",
            reasoning="Escalated from regex",
        )
        assert rec.error_type == "KeyError"
        assert rec.reasoning == "Escalated from regex"


# ======================================================================
# TestAEIStats
# ======================================================================


class TestAEIStats:
    """Tests for the AEIStats dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible."""
        stats = AEIStats(
            total_attempts=10,
            total_successes=7,
            total_failures=3,
            success_rate=0.7,
            strategies_used={"regex_patch": 5, "ast_diff": 5},
            escalation_count=1,
            top_error_types=[("TypeError", 4), ("KeyError", 3)],
        )
        assert stats.total_attempts == 10
        assert stats.success_rate == pytest.approx(0.7)
        assert len(stats.strategies_used) == 2
        assert stats.top_error_types[0] == ("TypeError", 4)


# ======================================================================
# TestAdaptiveExecutionIntelligence
# ======================================================================


class TestAEIInit:
    """Tests for AEI initialisation."""

    def test_creates_db_file(self, db_path: Path) -> None:
        """Database file is created on init."""
        engine = AdaptiveExecutionIntelligence(db_path=db_path, repo="r")
        try:
            assert db_path.exists()
        finally:
            engine.close()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        deep_path = tmp_path / "a" / "b" / "c" / "test.db"
        engine = AdaptiveExecutionIntelligence(db_path=deep_path, repo="r")
        try:
            assert deep_path.parent.exists()
        finally:
            engine.close()

    def test_table_exists(self, aei: AdaptiveExecutionIntelligence) -> None:
        """The attempt_log table is created."""
        conn = aei._get_conn()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='attempt_log'"
        )
        assert cursor.fetchone() is not None

    def test_indexes_exist(self, aei: AdaptiveExecutionIntelligence) -> None:
        """Required indexes are created."""
        conn = aei._get_conn()
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        index_names = {r["name"] for r in rows}
        assert "idx_attempt_fingerprint" in index_names
        assert "idx_attempt_repo" in index_names
        assert "idx_attempt_outcome" in index_names


class TestFingerprintError:
    """Tests for the static fingerprint_error method."""

    def test_normalises_line_numbers(self) -> None:
        """Line numbers are replaced with 'line N'."""
        fp = AdaptiveExecutionIntelligence.fingerprint_error(
            "TypeError",
            'File "foo.py", line 42, in bar',
            "foo.py",
            "bar",
        )
        assert "line 42" not in fp.stack_pattern
        assert "line N" in fp.stack_pattern

    def test_normalises_addresses(self) -> None:
        """Memory addresses are replaced with '0xADDR'."""
        fp = AdaptiveExecutionIntelligence.fingerprint_error(
            "RuntimeError",
            "object at 0x7f3abc123def",
            "x.py",
            "run",
        )
        assert "0x7f3abc123def" not in fp.stack_pattern
        assert "0xADDR" in fp.stack_pattern

    def test_normalises_dates(self) -> None:
        """Date stamps are replaced with 'DATE'."""
        fp = AdaptiveExecutionIntelligence.fingerprint_error(
            "OSError",
            "log entry 2025-03-11 failed",
            "log.py",
            "write",
        )
        assert "2025-03-11" not in fp.stack_pattern
        assert "DATE" in fp.stack_pattern

    def test_truncates_long_stack(self) -> None:
        """Stack patterns longer than 500 chars are truncated."""
        long_trace = "x" * 1000
        fp = AdaptiveExecutionIntelligence.fingerprint_error(
            "Error", long_trace, "f.py", "fn"
        )
        assert len(fp.stack_pattern) <= 500

    def test_returns_fingerprint_with_hash(self) -> None:
        """The returned fingerprint has a non-empty hash."""
        fp = AdaptiveExecutionIntelligence.fingerprint_error(
            "TypeError", "trace", "f.py", "fn"
        )
        assert fp.fingerprint_hash
        assert len(fp.fingerprint_hash) == 16


class TestRecordAttempt:
    """Tests for recording attempts."""

    def test_records_success(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """A successful attempt is persisted."""
        rec = aei.record_attempt(
            sample_fingerprint,
            FixStrategy.REGEX_PATCH,
            "gpt-4o-mini",
            2000,
            "success",
        )
        assert rec.id > 0
        assert rec.outcome == "success"
        assert rec.strategy == "regex_patch"
        assert rec.model == "gpt-4o-mini"

    def test_records_failure(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """A failed attempt is persisted."""
        rec = aei.record_attempt(
            sample_fingerprint,
            FixStrategy.AST_DIFF,
            "gpt-4o",
            4000,
            "failure",
            reasoning="AST parse failed",
        )
        assert rec.outcome == "failure"
        assert rec.reasoning == "AST parse failed"

    def test_rejects_invalid_outcome(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Invalid outcomes are rejected."""
        with pytest.raises(ValueError, match="outcome must be"):
            aei.record_attempt(
                sample_fingerprint,
                FixStrategy.REGEX_PATCH,
                "gpt-4o-mini",
                2000,
                "maybe",
            )

    def test_stores_error_type(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """The error_type from the fingerprint is stored."""
        rec = aei.record_attempt(
            sample_fingerprint,
            FixStrategy.REGEX_PATCH,
            "gpt-4o-mini",
            2000,
            "success",
        )
        assert rec.error_type == "TypeError"

    def test_auto_increments_id(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Each record gets a unique auto-incremented ID."""
        r1 = aei.record_attempt(
            sample_fingerprint, FixStrategy.REGEX_PATCH, "m", 1000, "success"
        )
        r2 = aei.record_attempt(
            sample_fingerprint, FixStrategy.AST_DIFF, "m", 1000, "failure"
        )
        assert r2.id > r1.id

    def test_timestamp_is_set(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Recorded attempts have a non-empty timestamp."""
        rec = aei.record_attempt(
            sample_fingerprint, FixStrategy.REGEX_PATCH, "m", 1000, "success"
        )
        assert rec.timestamp
        assert "T" in rec.timestamp  # ISO format


class TestGetPastAttempts:
    """Tests for retrieving past attempts."""

    def test_empty_for_unseen_fingerprint(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """No results for a fingerprint with no history."""
        assert aei.get_past_attempts(sample_fingerprint) == []

    def test_returns_recorded_attempts(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Previously recorded attempts are returned."""
        aei.record_attempt(
            sample_fingerprint, FixStrategy.REGEX_PATCH, "m", 1000, "failure"
        )
        aei.record_attempt(
            sample_fingerprint, FixStrategy.AST_DIFF, "m", 2000, "success"
        )
        attempts = aei.get_past_attempts(sample_fingerprint)
        assert len(attempts) == 2

    def test_ordered_newest_first(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Attempts are returned newest first."""
        aei.record_attempt(
            sample_fingerprint, FixStrategy.REGEX_PATCH, "m", 1000, "failure"
        )
        aei.record_attempt(
            sample_fingerprint, FixStrategy.AST_DIFF, "m", 2000, "success"
        )
        attempts = aei.get_past_attempts(sample_fingerprint)
        assert attempts[0].strategy == "ast_diff"

    def test_does_not_return_other_fingerprints(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
        other_fingerprint: ErrorFingerprint,
    ) -> None:
        """Attempts for different fingerprints are isolated."""
        aei.record_attempt(
            sample_fingerprint, FixStrategy.REGEX_PATCH, "m", 1000, "failure"
        )
        aei.record_attempt(
            other_fingerprint, FixStrategy.AST_DIFF, "m", 2000, "success"
        )
        attempts = aei.get_past_attempts(sample_fingerprint)
        assert len(attempts) == 1
        assert attempts[0].strategy == "regex_patch"


class TestRecommendStrategy:
    """Tests for strategy recommendation."""

    def test_first_attempt_defaults_to_cheapest(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """First attempt uses REGEX_PATCH with cheap tier."""
        rec = aei.recommend_strategy(sample_fingerprint)
        assert rec.strategy == FixStrategy.REGEX_PATCH
        assert rec.model_tier == "cheap"
        assert rec.past_attempts == 0
        assert rec.confidence == 0.5

    def test_reuses_successful_strategy(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """If a strategy previously succeeded, it's recommended again."""
        aei.record_attempt(
            sample_fingerprint,
            FixStrategy.AST_DIFF,
            "gpt-4o",
            3000,
            "success",
        )
        rec = aei.recommend_strategy(sample_fingerprint)
        assert rec.strategy == FixStrategy.AST_DIFF
        assert rec.confidence == 0.8
        assert rec.past_successes == 1

    def test_escalates_after_failures(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """After 3 REGEX_PATCH failures, recommends AST_DIFF."""
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

    def test_model_tier_escalation_cheap_to_mid(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """After 3 failures with cheap models, escalates to mid tier."""
        for _ in range(ESCALATION_THRESHOLD):
            aei.record_attempt(
                sample_fingerprint,
                FixStrategy.REGEX_PATCH,
                "gpt-4o-mini",  # cheap tier
                2000,
                "failure",
            )
        rec = aei.recommend_strategy(sample_fingerprint)
        assert rec.model_tier == "mid"

    def test_model_tier_escalation_mid_to_premium(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """After failures with both cheap and mid, escalates to premium."""
        # 3 cheap failures
        for _ in range(ESCALATION_THRESHOLD):
            aei.record_attempt(
                sample_fingerprint,
                FixStrategy.REGEX_PATCH,
                "gpt-4o-mini",
                2000,
                "failure",
            )
        # 3 mid failures
        for _ in range(ESCALATION_THRESHOLD):
            aei.record_attempt(
                sample_fingerprint,
                FixStrategy.AST_DIFF,
                "gpt-4o",  # mid tier
                3000,
                "failure",
            )
        rec = aei.recommend_strategy(sample_fingerprint)
        assert rec.model_tier == "premium"

    def test_context_expansion(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """After 3 small-context failures, context multiplier is 2x."""
        for _ in range(ESCALATION_THRESHOLD):
            aei.record_attempt(
                sample_fingerprint,
                FixStrategy.REGEX_PATCH,
                "gpt-4o-mini",
                2000,  # < 4000
                "failure",
            )
        rec = aei.recommend_strategy(sample_fingerprint)
        assert rec.context_multiplier == 2.0

    def test_no_context_expansion_for_large_context(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Failures with large context do not trigger expansion."""
        for _ in range(ESCALATION_THRESHOLD):
            aei.record_attempt(
                sample_fingerprint,
                FixStrategy.REGEX_PATCH,
                "gpt-4o-mini",
                8000,  # >= 4000
                "failure",
            )
        rec = aei.recommend_strategy(sample_fingerprint)
        assert rec.context_multiplier == 1.0

    def test_multi_model_debate_when_all_exhausted(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """When all strategies have 3+ failures, Multi-Model Debate is used."""
        for s in STRATEGY_ORDER:
            for _ in range(ESCALATION_THRESHOLD):
                aei.record_attempt(
                    sample_fingerprint, s, "gpt-4o-mini", 2000, "failure"
                )
        rec = aei.recommend_strategy(sample_fingerprint)
        assert rec.strategy == FixStrategy.MULTI_MODEL_DEBATE

    def test_confidence_decreases_with_failures(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Confidence drops as failures accumulate."""
        first_rec = aei.recommend_strategy(sample_fingerprint)
        for _ in range(5):
            aei.record_attempt(
                sample_fingerprint,
                FixStrategy.REGEX_PATCH,
                "gpt-4o-mini",
                2000,
                "failure",
            )
        second_rec = aei.recommend_strategy(sample_fingerprint)
        assert second_rec.confidence < first_rec.confidence

    def test_confidence_floor(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Confidence does not drop below 0.1."""
        for _ in range(20):
            aei.record_attempt(
                sample_fingerprint,
                FixStrategy.REGEX_PATCH,
                "gpt-4o-mini",
                2000,
                "failure",
            )
        rec = aei.recommend_strategy(sample_fingerprint)
        assert rec.confidence >= 0.1

    def test_reasoning_explains_escalation(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Reasoning string mentions the escalation trigger."""
        for _ in range(ESCALATION_THRESHOLD):
            aei.record_attempt(
                sample_fingerprint,
                FixStrategy.REGEX_PATCH,
                "gpt-4o-mini",
                2000,
                "failure",
            )
        rec = aei.recommend_strategy(sample_fingerprint)
        assert "cheap" in rec.reasoning.lower() or "mid" in rec.reasoning.lower()


class TestCrossRepoLearning:
    """Tests for cross-repo learning."""

    def test_uses_cross_repo_success(self, db_path: Path) -> None:
        """A success in repo A is suggested for the same error in repo B."""
        engine_a = AdaptiveExecutionIntelligence(db_path=db_path, repo="repo-a")
        engine_b = AdaptiveExecutionIntelligence(db_path=db_path, repo="repo-b")
        try:
            fp_a = AdaptiveExecutionIntelligence.fingerprint_error(
                "TypeError", "trace at line 1", "f.py", "fn"
            )
            engine_a.record_attempt(
                fp_a, FixStrategy.AST_DIFF, "gpt-4o", 3000, "success"
            )

            # repo-b has a different fingerprint hash but same error_type
            fp_b = AdaptiveExecutionIntelligence.fingerprint_error(
                "TypeError", "trace at line 99", "g.py", "gn"
            )
            rec = engine_b.recommend_strategy(fp_b)
            assert rec.strategy == FixStrategy.AST_DIFF
            assert "Cross-repo" in rec.reasoning
            assert rec.confidence == 0.6
        finally:
            engine_a.close()
            engine_b.close()

    def test_no_cross_repo_when_local_exists(self, db_path: Path) -> None:
        """Local history takes priority over cross-repo data."""
        engine_a = AdaptiveExecutionIntelligence(db_path=db_path, repo="repo-a")
        engine_b = AdaptiveExecutionIntelligence(db_path=db_path, repo="repo-b")
        try:
            fp_a = engine_a.fingerprint_error(
                "TypeError", "trace x", "f.py", "fn"
            )
            engine_a.record_attempt(
                fp_a, FixStrategy.AST_DIFF, "gpt-4o", 3000, "success"
            )

            # Record local success in repo-b with same fingerprint
            fp_b = engine_b.fingerprint_error(
                "TypeError", "trace x", "f.py", "fn"
            )
            engine_b.record_attempt(
                fp_b, FixStrategy.FULL_REWRITE, "gpt-4o", 5000, "success"
            )
            rec = engine_b.recommend_strategy(fp_b)
            # Should use local success, not cross-repo
            assert rec.strategy == FixStrategy.FULL_REWRITE
            assert rec.confidence == 0.8
        finally:
            engine_a.close()
            engine_b.close()


class TestModelTierDetection:
    """Tests for the _get_model_tier static method."""

    def test_cheap_models(self) -> None:
        """All cheap models are classified correctly."""
        for model in MODEL_TIERS["cheap"]:
            assert AdaptiveExecutionIntelligence._get_model_tier(model) == "cheap"

    def test_mid_models(self) -> None:
        """All mid models are classified correctly."""
        for model in MODEL_TIERS["mid"]:
            assert AdaptiveExecutionIntelligence._get_model_tier(model) == "mid"

    def test_premium_models(self) -> None:
        """All premium models are classified correctly."""
        for model in MODEL_TIERS["premium"]:
            assert AdaptiveExecutionIntelligence._get_model_tier(model) == "premium"

    def test_unknown_model_defaults_to_cheap(self) -> None:
        """Unknown models default to the cheap tier."""
        assert AdaptiveExecutionIntelligence._get_model_tier("unknown-model") == "cheap"


class TestGetStats:
    """Tests for the get_stats method."""

    def test_empty_stats(self, aei: AdaptiveExecutionIntelligence) -> None:
        """Stats from an empty database are all zeros."""
        stats = aei.get_stats()
        assert stats.total_attempts == 0
        assert stats.total_successes == 0
        assert stats.total_failures == 0
        assert stats.success_rate == 0.0
        assert stats.strategies_used == {}
        assert stats.escalation_count == 0
        assert stats.top_error_types == []

    def test_stats_with_data(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Stats reflect recorded attempts accurately."""
        aei.record_attempt(
            sample_fingerprint, FixStrategy.REGEX_PATCH, "m", 1000, "success"
        )
        aei.record_attempt(
            sample_fingerprint, FixStrategy.REGEX_PATCH, "m", 1000, "failure"
        )
        aei.record_attempt(
            sample_fingerprint, FixStrategy.AST_DIFF, "m", 2000, "success"
        )
        stats = aei.get_stats()
        assert stats.total_attempts == 3
        assert stats.total_successes == 2
        assert stats.total_failures == 1
        assert stats.success_rate == pytest.approx(2 / 3)
        assert stats.strategies_used["regex_patch"] == 2
        assert stats.strategies_used["ast_diff"] == 1

    def test_stats_filtered_by_repo(self, db_path: Path) -> None:
        """Stats can be filtered by repository."""
        engine = AdaptiveExecutionIntelligence(db_path=db_path, repo="repo-a")
        try:
            fp = engine.fingerprint_error("E", "t", "f.py", "fn")
            engine.record_attempt(
                fp, FixStrategy.REGEX_PATCH, "m", 1000, "success"
            )

            # Switch repo context manually for a second record
            engine._repo = "repo-b"
            engine.record_attempt(
                fp, FixStrategy.AST_DIFF, "m", 2000, "failure"
            )

            stats_a = engine.get_stats(repo="repo-a")
            assert stats_a.total_attempts == 1
            assert stats_a.total_successes == 1

            stats_b = engine.get_stats(repo="repo-b")
            assert stats_b.total_attempts == 1
            assert stats_b.total_failures == 1

            stats_all = engine.get_stats()
            assert stats_all.total_attempts == 2
        finally:
            engine.close()

    def test_escalation_count(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Escalation count includes MODEL_ESCALATE and MULTI_MODEL_DEBATE."""
        aei.record_attempt(
            sample_fingerprint,
            FixStrategy.MODEL_ESCALATE,
            "m",
            1000,
            "failure",
        )
        aei.record_attempt(
            sample_fingerprint,
            FixStrategy.MULTI_MODEL_DEBATE,
            "m",
            2000,
            "success",
        )
        aei.record_attempt(
            sample_fingerprint,
            FixStrategy.REGEX_PATCH,
            "m",
            1000,
            "success",
        )
        stats = aei.get_stats()
        assert stats.escalation_count == 2

    def test_top_error_types(
        self,
        aei: AdaptiveExecutionIntelligence,
    ) -> None:
        """Top error types are sorted by frequency."""
        fp_type = AdaptiveExecutionIntelligence.fingerprint_error(
            "TypeError", "t1", "f.py", "fn"
        )
        fp_key = AdaptiveExecutionIntelligence.fingerprint_error(
            "KeyError", "t2", "g.py", "gn"
        )
        for _ in range(5):
            aei.record_attempt(
                fp_type, FixStrategy.REGEX_PATCH, "m", 1000, "failure"
            )
        for _ in range(3):
            aei.record_attempt(
                fp_key, FixStrategy.REGEX_PATCH, "m", 1000, "failure"
            )
        stats = aei.get_stats()
        assert stats.top_error_types[0] == ("TypeError", 5)
        assert stats.top_error_types[1] == ("KeyError", 3)


class TestReset:
    """Tests for the reset method."""

    def test_reset_all(
        self,
        aei: AdaptiveExecutionIntelligence,
        sample_fingerprint: ErrorFingerprint,
    ) -> None:
        """Reset with no repo filter clears everything."""
        aei.record_attempt(
            sample_fingerprint, FixStrategy.REGEX_PATCH, "m", 1000, "success"
        )
        aei.record_attempt(
            sample_fingerprint, FixStrategy.AST_DIFF, "m", 2000, "failure"
        )
        deleted = aei.reset()
        assert deleted == 2
        assert aei.get_stats().total_attempts == 0

    def test_reset_by_repo(self, db_path: Path) -> None:
        """Reset with a repo filter only deletes that repo's records."""
        engine = AdaptiveExecutionIntelligence(db_path=db_path, repo="repo-a")
        try:
            fp = engine.fingerprint_error("E", "t", "f.py", "fn")
            engine.record_attempt(
                fp, FixStrategy.REGEX_PATCH, "m", 1000, "success"
            )

            engine._repo = "repo-b"
            engine.record_attempt(
                fp, FixStrategy.AST_DIFF, "m", 2000, "failure"
            )

            deleted = engine.reset(repo="repo-a")
            assert deleted == 1
            assert engine.get_stats().total_attempts == 1
        finally:
            engine.close()

    def test_reset_empty_returns_zero(
        self, aei: AdaptiveExecutionIntelligence
    ) -> None:
        """Resetting an empty database returns 0."""
        assert aei.reset() == 0


class TestClose:
    """Tests for closing the AEI engine."""

    def test_close_clears_connection(self, db_path: Path) -> None:
        """After close, the connection is set to None."""
        engine = AdaptiveExecutionIntelligence(db_path=db_path, repo="r")
        engine.close()
        assert not hasattr(engine._local, "conn") or engine._local.conn is None

    def test_double_close_is_safe(self, db_path: Path) -> None:
        """Calling close twice does not raise."""
        engine = AdaptiveExecutionIntelligence(db_path=db_path, repo="r")
        engine.close()
        engine.close()  # Should not raise
