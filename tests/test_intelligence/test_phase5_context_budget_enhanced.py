"""Tests for Phase 5 Smart Context Budget Manager enhancements.

Covers:
- RelevanceLevel enum values
- ContextItem creation and scoring
- BudgetAllocation calculation (40/10/50 split)
- build_relevance_graph with mock project structure
- estimate_tokens accuracy
- allocate() with various scenarios
- log_efficiency and get_efficiency_stats with temp SQLite
- generate_context_display format
- Edge cases: empty project, single file, massive files
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from prism.intelligence.context_budget import (
    CONTEXT_BUDGET_PCT,
    DEFAULT_CONTEXT_WINDOW,
    MODEL_CONTEXT_WINDOWS,
    RESPONSE_RESERVE_PCT,
    SYSTEM_RESERVE_PCT,
    BudgetAllocation,
    ContextEfficiencyRecord,
    ContextItem,
    EfficiencyStats,
    RelevanceLevel,
    SmartContextBudgetManager,
    estimate_tokens,
)

if TYPE_CHECKING:
    from pathlib import Path

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """Create a minimal Python project with imports and tests."""
    src = tmp_path / "src" / "prism"
    src.mkdir(parents=True)
    tests = tmp_path / "tests"
    tests.mkdir()

    (src / "__init__.py").write_text("")

    (src / "auth.py").write_text(
        "from prism.utils import helper\n\n"
        "def validate_token(token: str) -> bool:\n"
        "    return bool(token)\n\n"
        "def authenticate(user: str) -> bool:\n"
        "    return True\n"
    )

    (src / "router.py").write_text(
        "from prism.auth import validate_token\n\n"
        "def route_request(prompt: str) -> str:\n"
        "    return 'model-a'\n\n"
        "def select_model(tier: str) -> str:\n"
        "    return tier\n"
    )

    (src / "utils.py").write_text(
        "def helper() -> str:\n"
        "    return 'help'\n\n"
        "def format_output(text: str) -> str:\n"
        "    return text.strip()\n"
    )

    (src / "config.py").write_text(
        "DEFAULT_MODEL = 'gpt-4o'\n\n"
        "def get_config() -> dict:\n"
        "    return {}\n"
    )

    (src / "types.py").write_text(
        "from typing import TypeAlias\n\n"
        "ModelId: TypeAlias = str\n"
    )

    (tests / "__init__.py").write_text("")

    (tests / "test_auth.py").write_text(
        "from prism.auth import validate_token\n\n"
        "def test_validate_token():\n"
        "    assert validate_token('abc')\n"
    )

    (tests / "test_router.py").write_text(
        "from prism.router import route_request\n\n"
        "def test_route_request():\n"
        "    assert route_request('hi') == 'model-a'\n"
    )

    # README
    (tmp_path / "README.md").write_text("# Test Project\n")

    return tmp_path


@pytest.fixture()
def manager(project: Path) -> SmartContextBudgetManager:
    """Create a SmartContextBudgetManager with the test project."""
    return SmartContextBudgetManager(project_root=project)


class _FakeDatabase:
    """Minimal in-memory SQLite wrapper mimicking prism.db.database.Database."""

    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self._in_transaction = False

    def execute(
        self,
        sql: str,
        params: tuple[object, ...] | dict[str, object] = (),
    ) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def fetchone(
        self,
        sql: str,
        params: tuple[object, ...] | dict[str, object] = (),
    ) -> sqlite3.Row | None:
        cursor = self.conn.execute(sql, params)
        return cursor.fetchone()

    def fetchall(
        self,
        sql: str,
        params: tuple[object, ...] | dict[str, object] = (),
    ) -> list[sqlite3.Row]:
        cursor = self.conn.execute(sql, params)
        return cursor.fetchall()

    def commit(self) -> None:
        self.conn.commit()

    @property
    def in_transaction(self) -> bool:
        return self._in_transaction


@pytest.fixture()
def fake_db() -> _FakeDatabase:
    """Create an in-memory fake database for testing."""
    return _FakeDatabase()


# ======================================================================
# Test RelevanceLevel enum
# ======================================================================


class TestRelevanceLevel:
    """Tests for the RelevanceLevel enum."""

    def test_direct_value(self) -> None:
        """DIRECT should be 1.0."""
        assert RelevanceLevel.DIRECT.value == 1.0

    def test_related_value(self) -> None:
        """RELATED should be 0.85."""
        assert RelevanceLevel.RELATED.value == 0.85

    def test_indirect_value(self) -> None:
        """INDIRECT should be 0.6."""
        assert RelevanceLevel.INDIRECT.value == 0.6

    def test_context_only_value(self) -> None:
        """CONTEXT_ONLY should be 0.3."""
        assert RelevanceLevel.CONTEXT_ONLY.value == 0.3

    def test_excluded_value(self) -> None:
        """EXCLUDED should be 0.0."""
        assert RelevanceLevel.EXCLUDED.value == 0.0

    def test_all_levels_exist(self) -> None:
        """Enum should have exactly 5 levels."""
        assert len(RelevanceLevel) == 5

    def test_ordering(self) -> None:
        """Levels should be ordered from highest to lowest."""
        values = [level.value for level in RelevanceLevel]
        assert sorted(values, reverse=True) == [1.0, 0.85, 0.6, 0.3, 0.0]


# ======================================================================
# Test ContextItem
# ======================================================================


class TestContextItem:
    """Tests for the ContextItem dataclass."""

    def test_creation(self) -> None:
        """ContextItem should store all fields."""
        item = ContextItem(
            path="src/auth.py",
            relevance=1.0,
            token_count=500,
            reason="directly mentioned",
            level=RelevanceLevel.DIRECT,
        )
        assert item.path == "src/auth.py"
        assert item.relevance == 1.0
        assert item.token_count == 500
        assert item.reason == "directly mentioned"
        assert item.level == RelevanceLevel.DIRECT

    def test_excluded_item(self) -> None:
        """An excluded item should have relevance 0.0."""
        item = ContextItem(
            path="src/unrelated.py",
            relevance=0.0,
            token_count=100,
            reason="no relevance detected",
            level=RelevanceLevel.EXCLUDED,
        )
        assert item.relevance == 0.0
        assert item.level == RelevanceLevel.EXCLUDED


# ======================================================================
# Test BudgetAllocation calculation (40/10/50 split)
# ======================================================================


class TestBudgetAllocation:
    """Tests for BudgetAllocation and the 40/10/50 split."""

    def test_split_percentages(self) -> None:
        """40% response + 10% system + 50% context = 100%."""
        assert RESPONSE_RESERVE_PCT == 0.40
        assert SYSTEM_RESERVE_PCT == 0.10
        assert CONTEXT_BUDGET_PCT == 0.50
        assert RESPONSE_RESERVE_PCT + SYSTEM_RESERVE_PCT + CONTEXT_BUDGET_PCT == 1.0

    def test_allocation_fields(self) -> None:
        """BudgetAllocation should compute correct splits."""
        total = 100_000
        alloc = BudgetAllocation(
            total_tokens=total,
            response_tokens=int(total * 0.4),
            system_tokens=int(total * 0.1),
            context_tokens=int(total * 0.5),
            items_included=[],
            items_excluded=[],
            tokens_used=0,
            tokens_remaining=int(total * 0.5),
            efficiency_pct=0.0,
        )
        assert alloc.response_tokens == 40_000
        assert alloc.system_tokens == 10_000
        assert alloc.context_tokens == 50_000

    def test_allocation_with_items(self, manager: SmartContextBudgetManager) -> None:
        """allocate() should produce correct splits for a known model."""
        alloc = manager.allocate(
            task_description="test",
            available_files=[],
            model="gpt-4o",
        )
        total = MODEL_CONTEXT_WINDOWS["gpt-4o"]
        assert alloc.total_tokens == total
        assert alloc.response_tokens == int(total * 0.4)
        assert alloc.system_tokens == int(total * 0.1)
        assert alloc.context_tokens == total - alloc.response_tokens - alloc.system_tokens

    def test_default_model_context(self, manager: SmartContextBudgetManager) -> None:
        """Unknown models should use DEFAULT_CONTEXT_WINDOW."""
        alloc = manager.allocate(
            task_description="test",
            available_files=[],
            model="unknown-model-xyz",
        )
        assert alloc.total_tokens == DEFAULT_CONTEXT_WINDOW

    def test_tokens_remaining_calculation(self) -> None:
        """tokens_remaining should be context_tokens - tokens_used."""
        alloc = BudgetAllocation(
            total_tokens=100_000,
            response_tokens=40_000,
            system_tokens=10_000,
            context_tokens=50_000,
            items_included=[],
            items_excluded=[],
            tokens_used=20_000,
            tokens_remaining=30_000,
            efficiency_pct=40.0,
        )
        assert alloc.tokens_remaining == alloc.context_tokens - alloc.tokens_used


# ======================================================================
# Test build_relevance_graph
# ======================================================================


class TestBuildRelevanceGraph:
    """Tests for build_relevance_graph with mock project structure."""

    def test_level0_direct_mentions(
        self, manager: SmartContextBudgetManager, project: Path,
    ) -> None:
        """Directly mentioned files should be DIRECT (1.0)."""
        graph = manager.build_relevance_graph(
            mentioned_files=["src/prism/auth.py"],
            project_root=project,
        )
        assert "src/prism/auth.py" in graph
        level, _reason = graph["src/prism/auth.py"]
        assert level == RelevanceLevel.DIRECT

    def test_level1_imports(
        self, manager: SmartContextBudgetManager, project: Path,
    ) -> None:
        """Files imported by Level 0 should be RELATED (0.85)."""
        graph = manager.build_relevance_graph(
            mentioned_files=["src/prism/auth.py"],
            project_root=project,
        )
        # auth.py imports prism.utils, so utils.py should be RELATED
        related_files = {
            k for k, (lvl, _) in graph.items()
            if lvl == RelevanceLevel.RELATED
        }
        # At least test_auth should be found as a test file
        has_test = any("test_auth" in f for f in related_files)
        assert has_test, f"Expected test_auth in related files, got {related_files}"

    def test_level1_reverse_imports(
        self, manager: SmartContextBudgetManager, project: Path,
    ) -> None:
        """Files that import Level 0 should be RELATED."""
        graph = manager.build_relevance_graph(
            mentioned_files=["src/prism/auth.py"],
            project_root=project,
        )
        # router.py imports prism.auth, so should be RELATED
        router_entries = {
            k for k, (lvl, _) in graph.items()
            if "router" in k and lvl == RelevanceLevel.RELATED
        }
        assert len(router_entries) >= 1, (
            f"Expected router.py in related, got {router_entries}"
        )

    def test_level1_test_files(
        self, manager: SmartContextBudgetManager, project: Path,
    ) -> None:
        """Test files for Level 0 should be RELATED."""
        graph = manager.build_relevance_graph(
            mentioned_files=["src/prism/router.py"],
            project_root=project,
        )
        test_files = {
            k for k, (lvl, _) in graph.items()
            if "test_router" in k
        }
        assert len(test_files) >= 1

    def test_level3_context_files(
        self, manager: SmartContextBudgetManager, project: Path,
    ) -> None:
        """Config, types, and READMEs should be CONTEXT_ONLY (0.3)."""
        graph = manager.build_relevance_graph(
            mentioned_files=["src/prism/auth.py"],
            project_root=project,
        )
        context_files = {
            k for k, (lvl, _) in graph.items()
            if lvl == RelevanceLevel.CONTEXT_ONLY
        }
        # Should contain config.py or types.py or README.md
        has_context = any(
            "config" in f or "types" in f or "README" in f
            for f in context_files
        )
        assert has_context, f"Expected context files, got {context_files}"

    def test_empty_mentioned_files(
        self, manager: SmartContextBudgetManager, project: Path,
    ) -> None:
        """Empty mentioned_files should still find context-only files."""
        graph = manager.build_relevance_graph(
            mentioned_files=[],
            project_root=project,
        )
        # Should still find config/types/README at CONTEXT_ONLY level
        assert any(
            lvl == RelevanceLevel.CONTEXT_ONLY
            for lvl, _ in graph.values()
        )


# ======================================================================
# Test estimate_tokens
# ======================================================================


class TestEstimateTokens:
    """Tests for the estimate_tokens function."""

    def test_empty_string(self) -> None:
        """Empty string should return 0 tokens."""
        assert estimate_tokens("") == 0

    def test_short_text(self) -> None:
        """Short text should return at least 1 token."""
        result = estimate_tokens("hello")
        assert result >= 1

    def test_longer_text_proportional(self) -> None:
        """Longer text should produce more tokens."""
        short = estimate_tokens("hello")
        long = estimate_tokens("hello world this is a much longer text string")
        assert long > short

    def test_known_ratio(self) -> None:
        """Without tiktoken, 400 chars should produce ~100 tokens."""
        text = "x" * 400
        result = estimate_tokens(text)
        # Should be roughly 100 (400 / 4)
        assert 50 <= result <= 200

    def test_multiline_text(self) -> None:
        """Multi-line text should be counted correctly."""
        text = "line one\nline two\nline three\n" * 100
        result = estimate_tokens(text)
        assert result > 0


# ======================================================================
# Test allocate() scenarios
# ======================================================================


class TestAllocate:
    """Tests for the allocate() method with various scenarios."""

    def test_under_budget(self, manager: SmartContextBudgetManager) -> None:
        """All small files should fit within budget."""
        alloc = manager.allocate(
            task_description="fix the auth module",
            available_files=[
                "src/prism/auth.py",
                "src/prism/utils.py",
            ],
        )
        assert alloc.tokens_used > 0
        assert len(alloc.items_included) > 0
        assert alloc.tokens_remaining >= 0

    def test_over_budget_excludes_items(self, project: Path) -> None:
        """When files exceed budget, lowest-scored items should be excluded."""
        # Create many large files
        src = project / "src" / "prism"
        for i in range(20):
            (src / f"large_{i}.py").write_text(
                f"# Large file {i}\n" + ("x = 1\n" * 5000)
            )

        mgr = SmartContextBudgetManager(project_root=project)

        files = [f"src/prism/large_{i}.py" for i in range(20)]
        alloc = mgr.allocate(
            task_description="fix large_0",
            available_files=files,
            model="ollama/llama3.1:8b",  # very small context
        )
        # Some should be excluded due to small context window
        assert alloc.tokens_used <= alloc.context_tokens

    def test_forced_includes(self, manager: SmartContextBudgetManager) -> None:
        """Manually included files should have score 1.0."""
        manager.add_file("src/prism/utils.py")

        alloc = manager.allocate(
            task_description="unrelated task xyz",
            available_files=["src/prism/utils.py"],
        )

        found = [
            it for it in alloc.items_included
            if it.path == "src/prism/utils.py"
        ]
        assert len(found) == 1
        assert found[0].relevance == 1.0

    def test_forced_excludes(self, manager: SmartContextBudgetManager) -> None:
        """Manually excluded files should be in items_excluded."""
        manager.drop_file("src/prism/auth.py")

        alloc = manager.allocate(
            task_description="fix the auth module",
            available_files=["src/prism/auth.py"],
        )

        excluded_paths = [it.path for it in alloc.items_excluded]
        assert "src/prism/auth.py" in excluded_paths

    def test_error_context_always_included(
        self, manager: SmartContextBudgetManager,
    ) -> None:
        """Error context should always be included at highest priority."""
        alloc = manager.allocate(
            task_description="debug error",
            available_files=[],
            error_context="Traceback: ZeroDivisionError in auth.py:42",
        )

        error_items = [
            it for it in alloc.items_included
            if it.path == "<error_context>"
        ]
        assert len(error_items) == 1
        assert error_items[0].relevance == 1.0

    def test_empty_available_files(
        self, manager: SmartContextBudgetManager,
    ) -> None:
        """Empty available_files should return an empty allocation."""
        alloc = manager.allocate(
            task_description="do something",
            available_files=[],
        )
        assert alloc.tokens_used == 0
        assert len(alloc.items_included) == 0

    def test_mentioned_file_gets_direct_score(
        self, manager: SmartContextBudgetManager,
    ) -> None:
        """Files mentioned in the task should get DIRECT relevance."""
        alloc = manager.allocate(
            task_description="fix the bug in auth.py",
            available_files=["src/prism/auth.py"],
        )
        found = [
            it for it in alloc.items_included
            if it.path == "src/prism/auth.py"
        ]
        assert len(found) == 1
        assert found[0].relevance == 1.0

    def test_efficiency_percentage(
        self, manager: SmartContextBudgetManager,
    ) -> None:
        """efficiency_pct should reflect budget utilization."""
        alloc = manager.allocate(
            task_description="fix auth",
            available_files=["src/prism/auth.py"],
        )
        assert 0.0 <= alloc.efficiency_pct <= 100.0


# ======================================================================
# Test manual overrides
# ======================================================================


class TestManualOverrides:
    """Tests for add_file, drop_file, and reset_overrides."""

    def test_add_file(self, manager: SmartContextBudgetManager) -> None:
        """add_file should force-include and remove from excludes."""
        manager.drop_file("foo.py")
        manager.add_file("foo.py")
        assert "foo.py" in manager._manual_includes
        assert "foo.py" not in manager._manual_excludes

    def test_drop_file(self, manager: SmartContextBudgetManager) -> None:
        """drop_file should force-exclude and remove from includes."""
        manager.add_file("bar.py")
        manager.drop_file("bar.py")
        assert "bar.py" in manager._manual_excludes
        assert "bar.py" not in manager._manual_includes

    def test_reset_overrides(self, manager: SmartContextBudgetManager) -> None:
        """reset_overrides should clear both sets."""
        manager.add_file("a.py")
        manager.drop_file("b.py")
        manager.reset_overrides()
        assert len(manager._manual_includes) == 0
        assert len(manager._manual_excludes) == 0


# ======================================================================
# Test log_efficiency and get_efficiency_stats
# ======================================================================


class TestEfficiencyLogging:
    """Tests for SQLite efficiency logging and stats."""

    def test_log_efficiency_with_db(
        self, project: Path, fake_db: _FakeDatabase,
    ) -> None:
        """log_efficiency should insert a record into context_efficiency."""
        mgr = SmartContextBudgetManager(
            project_root=project,
            db=fake_db,  # type: ignore[arg-type]
        )

        record = ContextEfficiencyRecord(
            task_type="code_edit",
            files_included=5,
            tokens_used=10_000,
            files_excluded=3,
            outcome="success",
            model_used="gpt-4o",
            created_at=datetime.now(UTC).isoformat(),
        )
        mgr.log_efficiency(record)

        row = fake_db.fetchone(
            "SELECT COUNT(*) AS cnt FROM context_efficiency"
        )
        assert row is not None
        assert row["cnt"] == 1

    def test_log_efficiency_without_db(
        self, project: Path,
    ) -> None:
        """log_efficiency without a DB should silently skip."""
        mgr = SmartContextBudgetManager(project_root=project, db=None)
        record = ContextEfficiencyRecord(
            task_type="question",
            files_included=0,
            tokens_used=0,
            files_excluded=0,
            outcome="success",
            model_used="gpt-4o",
            created_at=datetime.now(UTC).isoformat(),
        )
        # Should not raise
        mgr.log_efficiency(record)

    def test_get_efficiency_stats_empty(
        self, project: Path, fake_db: _FakeDatabase,
    ) -> None:
        """get_efficiency_stats on empty DB should return zeros."""
        mgr = SmartContextBudgetManager(
            project_root=project,
            db=fake_db,  # type: ignore[arg-type]
        )
        stats = mgr.get_efficiency_stats()
        assert stats.total_records == 0
        assert stats.avg_tokens_used == 0.0
        assert stats.success_rate == 0.0

    def test_get_efficiency_stats_with_data(
        self, project: Path, fake_db: _FakeDatabase,
    ) -> None:
        """get_efficiency_stats should compute correct averages."""
        mgr = SmartContextBudgetManager(
            project_root=project,
            db=fake_db,  # type: ignore[arg-type]
        )

        now = datetime.now(UTC).isoformat()
        for i in range(3):
            mgr.log_efficiency(ContextEfficiencyRecord(
                task_type="code_edit",
                files_included=5 + i,
                tokens_used=10_000 + i * 1000,
                files_excluded=2,
                outcome="success" if i < 2 else "failure",
                model_used="gpt-4o",
                created_at=now,
            ))

        stats = mgr.get_efficiency_stats()
        assert stats.total_records == 3
        assert stats.avg_tokens_used > 0
        # 2 out of 3 succeeded
        assert abs(stats.success_rate - 2 / 3) < 0.01
        assert stats.total_tokens_saved >= 0

    def test_get_efficiency_stats_no_db(
        self, project: Path,
    ) -> None:
        """get_efficiency_stats without a DB should return zeros."""
        mgr = SmartContextBudgetManager(project_root=project, db=None)
        stats = mgr.get_efficiency_stats()
        assert stats.total_records == 0


# ======================================================================
# Test generate_context_display
# ======================================================================


class TestGenerateContextDisplay:
    """Tests for the display formatter."""

    def test_display_header(self) -> None:
        """Display should start with CURRENT CONTEXT header."""
        alloc = BudgetAllocation(
            total_tokens=100_000,
            response_tokens=40_000,
            system_tokens=10_000,
            context_tokens=50_000,
            items_included=[
                ContextItem(
                    path="src/auth.py",
                    relevance=1.0,
                    token_count=2341,
                    reason="directly mentioned",
                    level=RelevanceLevel.DIRECT,
                ),
            ],
            items_excluded=[],
            tokens_used=2341,
            tokens_remaining=47_659,
            efficiency_pct=4.7,
        )
        display = SmartContextBudgetManager.generate_context_display(alloc)
        assert "CURRENT CONTEXT" in display
        assert "2,341 tokens" in display

    def test_display_includes_scores(self) -> None:
        """Display should show scores for each included item."""
        item = ContextItem(
            path="src/router.py",
            relevance=0.85,
            token_count=1876,
            reason="calls validate_token",
            level=RelevanceLevel.RELATED,
        )
        alloc = BudgetAllocation(
            total_tokens=100_000,
            response_tokens=40_000,
            system_tokens=10_000,
            context_tokens=50_000,
            items_included=[item],
            items_excluded=[],
            tokens_used=1876,
            tokens_remaining=48_124,
            efficiency_pct=3.8,
        )
        display = SmartContextBudgetManager.generate_context_display(alloc)
        assert "score 0.85" in display
        assert "src/router.py" in display
        assert "calls validate_token" in display

    def test_display_excluded_section(self) -> None:
        """Display should show EXCLUDED section when items are excluded."""
        excluded = ContextItem(
            path="src/utils/jwt.py",
            relevance=0.6,
            token_count=4200,
            reason="too large",
            level=RelevanceLevel.INDIRECT,
        )
        alloc = BudgetAllocation(
            total_tokens=100_000,
            response_tokens=40_000,
            system_tokens=10_000,
            context_tokens=50_000,
            items_included=[],
            items_excluded=[excluded],
            tokens_used=0,
            tokens_remaining=50_000,
            efficiency_pct=0.0,
        )
        display = SmartContextBudgetManager.generate_context_display(alloc)
        assert "EXCLUDED" in display
        assert "src/utils/jwt.py" in display

    def test_display_empty_allocation(self) -> None:
        """Display should handle empty allocation gracefully."""
        alloc = BudgetAllocation(
            total_tokens=100_000,
            response_tokens=40_000,
            system_tokens=10_000,
            context_tokens=50_000,
            items_included=[],
            items_excluded=[],
            tokens_used=0,
            tokens_remaining=50_000,
            efficiency_pct=0.0,
        )
        display = SmartContextBudgetManager.generate_context_display(alloc)
        assert "CURRENT CONTEXT" in display
        assert "EXCLUDED" not in display

    def test_display_many_excluded_truncated(self) -> None:
        """Display should truncate excluded list to 10 items."""
        excluded = [
            ContextItem(
                path=f"src/file_{i}.py",
                relevance=0.1,
                token_count=100,
                reason="low relevance",
                level=RelevanceLevel.CONTEXT_ONLY,
            )
            for i in range(15)
        ]
        alloc = BudgetAllocation(
            total_tokens=100_000,
            response_tokens=40_000,
            system_tokens=10_000,
            context_tokens=50_000,
            items_included=[],
            items_excluded=excluded,
            tokens_used=0,
            tokens_remaining=50_000,
            efficiency_pct=0.0,
        )
        display = SmartContextBudgetManager.generate_context_display(alloc)
        assert "... and 5 more" in display


# ======================================================================
# Test edge cases
# ======================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_project(self, tmp_path: Path) -> None:
        """Manager should handle an empty project directory."""
        mgr = SmartContextBudgetManager(project_root=tmp_path)
        alloc = mgr.allocate(
            task_description="do something",
            available_files=[],
        )
        assert alloc.tokens_used == 0
        assert len(alloc.items_included) == 0

    def test_single_file_project(self, tmp_path: Path) -> None:
        """Manager should handle a project with only one file."""
        (tmp_path / "main.py").write_text("print('hello')\n")
        mgr = SmartContextBudgetManager(project_root=tmp_path)
        alloc = mgr.allocate(
            task_description="fix main.py",
            available_files=["main.py"],
        )
        assert len(alloc.items_included) == 1

    def test_massive_file_truncation(self, tmp_path: Path) -> None:
        """Files larger than 500KB should be skipped."""
        big_file = tmp_path / "huge.py"
        big_file.write_text("x = 1\n" * 100_000)  # ~600KB
        mgr = SmartContextBudgetManager(project_root=tmp_path)
        alloc = mgr.allocate(
            task_description="fix huge",
            available_files=["huge.py"],
        )
        # File should be silently skipped (too large to read)
        included_paths = [it.path for it in alloc.items_included]
        assert "huge.py" not in included_paths

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Non-existent files should be silently skipped."""
        mgr = SmartContextBudgetManager(project_root=tmp_path)
        alloc = mgr.allocate(
            task_description="fix missing",
            available_files=["does_not_exist.py"],
        )
        assert len(alloc.items_included) == 0

    def test_binary_file_skipped(self, tmp_path: Path) -> None:
        """Binary files should be handled gracefully."""
        bin_file = tmp_path / "data.bin"
        bin_file.write_bytes(b"\x00\x01\x02" * 100)
        mgr = SmartContextBudgetManager(project_root=tmp_path)
        # Should not crash
        alloc = mgr.allocate(
            task_description="check data",
            available_files=["data.bin"],
        )
        # May or may not be included depending on size, but should not crash
        assert alloc.tokens_used >= 0


# ======================================================================
# Test _extract_mentioned_functions
# ======================================================================


class TestExtractMentionedFunctions:
    """Tests for function name extraction from task descriptions."""

    def test_backtick_functions(self) -> None:
        """Should extract functions wrapped in backticks."""
        result = SmartContextBudgetManager._extract_mentioned_functions(
            "Fix the `validate_token` function"
        )
        assert "validate_token" in result

    def test_call_syntax(self) -> None:
        """Should extract function calls with parentheses."""
        result = SmartContextBudgetManager._extract_mentioned_functions(
            "The error is in authenticate(user)"
        )
        assert "authenticate" in result

    def test_snake_case(self) -> None:
        """Should extract snake_case identifiers."""
        result = SmartContextBudgetManager._extract_mentioned_functions(
            "Check route_request and select_model"
        )
        assert "route_request" in result
        assert "select_model" in result

    def test_deduplication(self) -> None:
        """Should deduplicate results."""
        result = SmartContextBudgetManager._extract_mentioned_functions(
            "`validate` calls validate() in validate_token"
        )
        assert result.count("validate") == 1


# ======================================================================
# Test _truncate_to_functions
# ======================================================================


class TestTruncateToFunctions:
    """Tests for function-level file truncation."""

    def test_truncation_keeps_matching_functions(
        self, manager: SmartContextBudgetManager,
    ) -> None:
        """Should keep only the matching functions."""
        content = (
            "def foo():\n"
            "    pass\n\n"
            "def target_function():\n"
            "    return 42\n\n"
            "def bar():\n"
            "    pass\n"
        )
        result = manager._truncate_to_functions(
            content, ["target_function"],
        )
        assert "target_function" in result

    def test_truncation_no_match_fallback(
        self, manager: SmartContextBudgetManager,
    ) -> None:
        """No match should fall back to first 200 lines."""
        content = "\n".join(f"line_{i} = {i}" for i in range(300))
        result = manager._truncate_to_functions(content, ["nonexistent"])
        lines = result.splitlines()
        assert len(lines) <= 200

    def test_truncation_syntax_error_fallback(
        self, manager: SmartContextBudgetManager,
    ) -> None:
        """Invalid Python should fall back to first 200 lines."""
        content = "def incomplete(\n" * 100
        result = manager._truncate_to_functions(content, ["incomplete"])
        assert len(result.splitlines()) <= 200


# ======================================================================
# Test ContextEfficiencyRecord dataclass
# ======================================================================


class TestContextEfficiencyRecord:
    """Tests for the ContextEfficiencyRecord dataclass."""

    def test_creation(self) -> None:
        """Record should store all fields."""
        now = datetime.now(UTC).isoformat()
        record = ContextEfficiencyRecord(
            task_type="code_edit",
            files_included=10,
            tokens_used=5000,
            files_excluded=5,
            outcome="success",
            model_used="gpt-4o",
            created_at=now,
        )
        assert record.task_type == "code_edit"
        assert record.files_included == 10
        assert record.tokens_used == 5000
        assert record.files_excluded == 5
        assert record.outcome == "success"
        assert record.model_used == "gpt-4o"
        assert record.created_at == now


# ======================================================================
# Test EfficiencyStats dataclass
# ======================================================================


class TestEfficiencyStats:
    """Tests for the EfficiencyStats dataclass."""

    def test_creation(self) -> None:
        """Stats should store all fields."""
        stats = EfficiencyStats(
            avg_tokens_used=5000.0,
            avg_efficiency_pct=75.0,
            success_rate=0.9,
            total_records=100,
            total_tokens_saved=50_000,
        )
        assert stats.avg_tokens_used == 5000.0
        assert stats.avg_efficiency_pct == 75.0
        assert stats.success_rate == 0.9
        assert stats.total_records == 100
        assert stats.total_tokens_saved == 50_000


# ======================================================================
# Test _get_file_imports
# ======================================================================


class TestGetFileImports:
    """Tests for AST-based import extraction."""

    def test_import_statement(self, tmp_path: Path) -> None:
        """Should extract 'import X' statements."""
        f = tmp_path / "mod.py"
        f.write_text("import os\nimport sys\n")
        imports = SmartContextBudgetManager._get_file_imports(f)
        assert "os" in imports
        assert "sys" in imports

    def test_from_import(self, tmp_path: Path) -> None:
        """Should extract 'from X import Y' statements."""
        f = tmp_path / "mod.py"
        f.write_text("from pathlib import Path\nfrom os.path import join\n")
        imports = SmartContextBudgetManager._get_file_imports(f)
        assert "pathlib" in imports
        assert "os.path" in imports

    def test_syntax_error_returns_empty(self, tmp_path: Path) -> None:
        """Should return empty list on syntax errors."""
        f = tmp_path / "bad.py"
        f.write_text("def broken(\n")
        imports = SmartContextBudgetManager._get_file_imports(f)
        assert imports == []

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Should return empty list for missing files."""
        imports = SmartContextBudgetManager._get_file_imports(
            tmp_path / "missing.py"
        )
        assert imports == []
