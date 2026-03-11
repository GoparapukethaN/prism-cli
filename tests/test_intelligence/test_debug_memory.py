"""Tests for prism.intelligence.debug_memory — Cross-Session Debugging Memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.intelligence.debug_memory import (
    BugFingerprint,
    DebugMemory,
    FixRecord,
    FixSuggestion,
)

if TYPE_CHECKING:
    from pathlib import Path


# =========================================================================
# Data-class unit tests
# =========================================================================


class TestBugFingerprint:
    """Tests for the BugFingerprint dataclass."""

    def test_auto_hash(self) -> None:
        fp = BugFingerprint(
            error_type="TypeError",
            stack_pattern="File x.py, line 10",
            affected_files=["x.py"],
            affected_functions=["foo"],
            language="python",
            framework="django",
        )
        assert fp.fingerprint_hash != ""
        assert len(fp.fingerprint_hash) == 16

    def test_different_inputs_different_hashes(self) -> None:
        fp1 = BugFingerprint(
            error_type="TypeError",
            stack_pattern="File a.py",
            affected_files=["a.py"],
            affected_functions=[],
            language="python",
            framework="",
        )
        fp2 = BugFingerprint(
            error_type="ValueError",
            stack_pattern="File b.py",
            affected_files=["b.py"],
            affected_functions=[],
            language="python",
            framework="",
        )
        assert fp1.fingerprint_hash != fp2.fingerprint_hash

    def test_same_inputs_same_hash(self) -> None:
        kwargs = {
            "error_type": "KeyError",
            "stack_pattern": "line 5",
            "affected_files": ["f.py"],
            "affected_functions": ["g"],
            "language": "python",
            "framework": "",
        }
        fp1 = BugFingerprint(**kwargs)
        fp2 = BugFingerprint(**kwargs)
        assert fp1.fingerprint_hash == fp2.fingerprint_hash

    def test_explicit_hash_preserved(self) -> None:
        fp = BugFingerprint(
            error_type="X",
            stack_pattern="Y",
            affected_files=[],
            affected_functions=[],
            language="python",
            framework="",
            fingerprint_hash="custom_hash_1234",
        )
        assert fp.fingerprint_hash == "custom_hash_1234"

    def test_hash_deterministic(self) -> None:
        fp = BugFingerprint(
            error_type="RuntimeError",
            stack_pattern="traceback",
            affected_files=["z.py", "a.py"],
            affected_functions=["run"],
            language="python",
            framework="flask",
        )
        # Calling __post_init__ again with same data produces same hash
        raw = f"{fp.error_type}:{fp.stack_pattern}:{','.join(sorted(fp.affected_files))}"
        import hashlib
        expected = hashlib.sha256(raw.encode()).hexdigest()[:16]
        assert fp.fingerprint_hash == expected

    def test_fields(self) -> None:
        fp = BugFingerprint(
            error_type="IOError",
            stack_pattern="open()",
            affected_files=["io.py"],
            affected_functions=["read"],
            language="python",
            framework="aiohttp",
        )
        assert fp.error_type == "IOError"
        assert fp.stack_pattern == "open()"
        assert fp.affected_files == ["io.py"]
        assert fp.affected_functions == ["read"]
        assert fp.language == "python"
        assert fp.framework == "aiohttp"


class TestFixRecord:
    """Tests for the FixRecord dataclass."""

    def test_fields(self) -> None:
        r = FixRecord(
            id=1,
            fingerprint="abc123",
            error_type="TypeError",
            stack_pattern="line 5",
            fix_pattern="Add None check",
            fix_diff="+ if x is None:",
            confidence=0.9,
            project="myapp",
            model_used="gpt-4o",
            timestamp="2026-01-01T00:00:00",
            language="python",
            framework="fastapi",
            affected_files_json='["app.py"]',
            affected_functions_json='["handler"]',
        )
        assert r.id == 1
        assert r.fingerprint == "abc123"
        assert r.error_type == "TypeError"
        assert r.fix_pattern == "Add None check"
        assert r.confidence == 0.9
        assert r.project == "myapp"
        assert r.model_used == "gpt-4o"

    def test_default_fields(self) -> None:
        r = FixRecord(
            id=0,
            fingerprint="x",
            error_type="E",
            stack_pattern="",
            fix_pattern="fix",
            fix_diff="",
            confidence=0.5,
            project="",
            model_used="",
            timestamp="t",
        )
        assert r.language == ""
        assert r.framework == ""
        assert r.affected_files_json == "[]"
        assert r.affected_functions_json == "[]"


class TestFixSuggestion:
    """Tests for the FixSuggestion dataclass."""

    def test_fields(self) -> None:
        record = FixRecord(
            id=1, fingerprint="h", error_type="E", stack_pattern="",
            fix_pattern="fix it", fix_diff="", confidence=0.8,
            project="proj", model_used="m", timestamp="t",
        )
        s = FixSuggestion(
            original_fix=record,
            similarity=0.95,
            adapted_description="Exact match: fix it",
            original_context="Project: proj",
        )
        assert s.original_fix is record
        assert s.similarity == 0.95
        assert s.adapted_description == "Exact match: fix it"
        assert s.original_context == "Project: proj"

    def test_similarity_range(self) -> None:
        record = FixRecord(
            id=0, fingerprint="", error_type="", stack_pattern="",
            fix_pattern="", fix_diff="", confidence=0.0,
            project="", model_used="", timestamp="",
        )
        for sim in (0.0, 0.5, 1.0):
            s = FixSuggestion(
                original_fix=record, similarity=sim,
                adapted_description="d", original_context="c",
            )
            assert s.similarity == sim


# =========================================================================
# DebugMemory tests
# =========================================================================


class TestDebugMemory:
    """Tests for the DebugMemory class."""

    @pytest.fixture()
    def db(self, tmp_path: Path) -> DebugMemory:
        """Create a fresh DebugMemory with a temporary database."""
        return DebugMemory(db_path=tmp_path / "debug_memory.db")

    @pytest.fixture()
    def sample_fingerprint(self) -> BugFingerprint:
        return BugFingerprint(
            error_type="TypeError",
            stack_pattern="File app.py, line 42, in handler\n  TypeError: NoneType",
            affected_files=["app.py"],
            affected_functions=["handler"],
            language="python",
            framework="fastapi",
        )

    @pytest.fixture()
    def alt_fingerprint(self) -> BugFingerprint:
        return BugFingerprint(
            error_type="TypeError",
            stack_pattern="File views.py, line 10\n  TypeError: cannot unpack",
            affected_files=["views.py"],
            affected_functions=["index"],
            language="python",
            framework="django",
        )

    # --- init ---

    def test_init_creates_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "sub" / "test.db"
        mem = DebugMemory(db_path=db_path)
        assert db_path.exists()
        mem.close()

    def test_init_creates_tables(self, db: DebugMemory) -> None:
        conn = db._get_conn()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fix_records'"
        )
        assert cursor.fetchone() is not None

    def test_init_idempotent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        mem1 = DebugMemory(db_path=db_path)
        mem1.close()
        # Opening again should not fail
        mem2 = DebugMemory(db_path=db_path)
        mem2.close()

    # --- store_fix ---

    def test_store_fix_returns_record(
        self, db: DebugMemory, sample_fingerprint: BugFingerprint
    ) -> None:
        record = db.store_fix(
            fingerprint=sample_fingerprint,
            fix_pattern="Add None check before accessing attribute",
            fix_diff="+ if obj is None:\n+     return",
            project="myapp",
            model_used="gpt-4o",
            confidence=0.85,
        )
        assert record.id > 0
        assert record.fingerprint == sample_fingerprint.fingerprint_hash
        assert record.error_type == "TypeError"
        assert record.fix_pattern == "Add None check before accessing attribute"
        assert record.confidence == 0.85
        assert record.project == "myapp"
        assert record.model_used == "gpt-4o"
        assert record.timestamp != ""

    def test_store_fix_default_confidence(
        self, db: DebugMemory, sample_fingerprint: BugFingerprint
    ) -> None:
        record = db.store_fix(
            fingerprint=sample_fingerprint,
            fix_pattern="fix",
            fix_diff="diff",
            project="p",
            model_used="m",
        )
        assert record.confidence == 0.5

    def test_store_fix_persists(
        self, db: DebugMemory, sample_fingerprint: BugFingerprint
    ) -> None:
        db.store_fix(
            fingerprint=sample_fingerprint,
            fix_pattern="fix1",
            fix_diff="diff1",
            project="proj",
            model_used="model",
        )
        fixes = db.browse_fixes()
        assert len(fixes) == 1
        assert fixes[0].fix_pattern == "fix1"

    def test_store_multiple_fixes(
        self, db: DebugMemory, sample_fingerprint: BugFingerprint
    ) -> None:
        for i in range(5):
            db.store_fix(
                fingerprint=sample_fingerprint,
                fix_pattern=f"fix_{i}",
                fix_diff=f"diff_{i}",
                project="proj",
                model_used="model",
            )
        fixes = db.browse_fixes()
        assert len(fixes) == 5

    # --- search_similar ---

    def test_search_similar_exact_match(
        self, db: DebugMemory, sample_fingerprint: BugFingerprint
    ) -> None:
        db.store_fix(
            fingerprint=sample_fingerprint,
            fix_pattern="Add None check",
            fix_diff="diff",
            project="p",
            model_used="m",
            confidence=0.9,
        )
        suggestions = db.search_similar(sample_fingerprint)
        assert len(suggestions) == 1
        assert suggestions[0].similarity == 1.0
        assert "Exact match" in suggestions[0].adapted_description

    def test_search_similar_same_error_type(
        self,
        db: DebugMemory,
        sample_fingerprint: BugFingerprint,
        alt_fingerprint: BugFingerprint,
    ) -> None:
        # Store a fix with sample fingerprint
        db.store_fix(
            fingerprint=sample_fingerprint,
            fix_pattern="Add None check",
            fix_diff="diff",
            project="proj_a",
            model_used="m",
        )
        # Search with alt fingerprint (same error_type, different hash)
        suggestions = db.search_similar(alt_fingerprint)
        assert len(suggestions) >= 1
        # Should be a "Similar error" match, not exact
        assert any("Similar error" in s.adapted_description for s in suggestions)

    def test_search_similar_no_results(self, db: DebugMemory) -> None:
        fp = BugFingerprint(
            error_type="UnknownError",
            stack_pattern="nowhere",
            affected_files=[],
            affected_functions=[],
            language="rust",
            framework="",
        )
        suggestions = db.search_similar(fp)
        assert suggestions == []

    def test_search_similar_respects_limit(
        self, db: DebugMemory, sample_fingerprint: BugFingerprint
    ) -> None:
        for i in range(10):
            fp = BugFingerprint(
                error_type="TypeError",
                stack_pattern=f"pattern_{i}",
                affected_files=[f"file_{i}.py"],
                affected_functions=[],
                language="python",
                framework="",
            )
            db.store_fix(
                fingerprint=fp,
                fix_pattern=f"fix_{i}",
                fix_diff="",
                project="proj",
                model_used="m",
            )
        suggestions = db.search_similar(sample_fingerprint, limit=3)
        assert len(suggestions) <= 3

    def test_search_similar_sorted_by_similarity(
        self,
        db: DebugMemory,
        sample_fingerprint: BugFingerprint,
        alt_fingerprint: BugFingerprint,
    ) -> None:
        # Exact match
        db.store_fix(
            fingerprint=sample_fingerprint,
            fix_pattern="exact fix",
            fix_diff="",
            project="p",
            model_used="m",
        )
        # Same error type but different fingerprint
        db.store_fix(
            fingerprint=alt_fingerprint,
            fix_pattern="similar fix",
            fix_diff="",
            project="p",
            model_used="m",
        )
        suggestions = db.search_similar(sample_fingerprint, limit=10)
        assert len(suggestions) >= 1
        # First should be the exact match (similarity=1.0)
        assert suggestions[0].similarity == 1.0

    # --- browse_fixes ---

    def test_browse_fixes_empty(self, db: DebugMemory) -> None:
        fixes = db.browse_fixes()
        assert fixes == []

    def test_browse_fixes_all(
        self, db: DebugMemory, sample_fingerprint: BugFingerprint
    ) -> None:
        db.store_fix(
            fingerprint=sample_fingerprint,
            fix_pattern="fix_a",
            fix_diff="",
            project="proj_a",
            model_used="m",
        )
        db.store_fix(
            fingerprint=sample_fingerprint,
            fix_pattern="fix_b",
            fix_diff="",
            project="proj_b",
            model_used="m",
        )
        fixes = db.browse_fixes()
        assert len(fixes) == 2

    def test_browse_fixes_by_project(
        self, db: DebugMemory, sample_fingerprint: BugFingerprint
    ) -> None:
        db.store_fix(
            fingerprint=sample_fingerprint,
            fix_pattern="fix_a",
            fix_diff="",
            project="alpha",
            model_used="m",
        )
        db.store_fix(
            fingerprint=sample_fingerprint,
            fix_pattern="fix_b",
            fix_diff="",
            project="beta",
            model_used="m",
        )
        fixes = db.browse_fixes(project="alpha")
        assert len(fixes) == 1
        assert fixes[0].project == "alpha"

    def test_browse_fixes_limit(
        self, db: DebugMemory, sample_fingerprint: BugFingerprint
    ) -> None:
        for i in range(10):
            db.store_fix(
                fingerprint=sample_fingerprint,
                fix_pattern=f"fix_{i}",
                fix_diff="",
                project="proj",
                model_used="m",
            )
        fixes = db.browse_fixes(limit=3)
        assert len(fixes) == 3

    # --- search_by_description ---

    def test_search_by_description_match(
        self, db: DebugMemory, sample_fingerprint: BugFingerprint
    ) -> None:
        db.store_fix(
            fingerprint=sample_fingerprint,
            fix_pattern="Add None check before attribute access",
            fix_diff="",
            project="p",
            model_used="m",
        )
        results = db.search_by_description("None check")
        assert len(results) == 1
        assert "None check" in results[0].fix_pattern

    def test_search_by_description_no_match(self, db: DebugMemory) -> None:
        results = db.search_by_description("xyzzy_nonexistent")
        assert results == []

    def test_search_by_description_matches_error_type(
        self, db: DebugMemory, sample_fingerprint: BugFingerprint
    ) -> None:
        db.store_fix(
            fingerprint=sample_fingerprint,
            fix_pattern="unrelated pattern",
            fix_diff="",
            project="p",
            model_used="m",
        )
        # Search by error_type "TypeError" which is in the record
        results = db.search_by_description("TypeError")
        assert len(results) == 1

    def test_search_by_description_limit(
        self, db: DebugMemory, sample_fingerprint: BugFingerprint
    ) -> None:
        for i in range(10):
            db.store_fix(
                fingerprint=sample_fingerprint,
                fix_pattern=f"fix_common_{i}",
                fix_diff="",
                project="p",
                model_used="m",
            )
        results = db.search_by_description("fix_common", limit=3)
        assert len(results) == 3

    # --- forget ---

    def test_forget_existing(
        self, db: DebugMemory, sample_fingerprint: BugFingerprint
    ) -> None:
        record = db.store_fix(
            fingerprint=sample_fingerprint,
            fix_pattern="bad fix",
            fix_diff="",
            project="p",
            model_used="m",
        )
        assert db.forget(record.id) is True
        fixes = db.browse_fixes()
        assert len(fixes) == 0

    def test_forget_nonexistent(self, db: DebugMemory) -> None:
        assert db.forget(9999) is False

    def test_forget_only_removes_target(
        self, db: DebugMemory, sample_fingerprint: BugFingerprint
    ) -> None:
        r1 = db.store_fix(
            fingerprint=sample_fingerprint,
            fix_pattern="keep",
            fix_diff="",
            project="p",
            model_used="m",
        )
        r2 = db.store_fix(
            fingerprint=sample_fingerprint,
            fix_pattern="remove",
            fix_diff="",
            project="p",
            model_used="m",
        )
        db.forget(r2.id)
        fixes = db.browse_fixes()
        assert len(fixes) == 1
        assert fixes[0].id == r1.id

    # --- get_stats ---

    def test_get_stats_empty(self, db: DebugMemory) -> None:
        stats = db.get_stats()
        assert stats["total_fixes"] == 0
        assert stats["projects"] == 0
        assert stats["error_types"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_get_stats_populated(self, db: DebugMemory) -> None:
        fp1 = BugFingerprint(
            error_type="TypeError",
            stack_pattern="",
            affected_files=[],
            affected_functions=[],
            language="python",
            framework="",
        )
        fp2 = BugFingerprint(
            error_type="ValueError",
            stack_pattern="",
            affected_files=[],
            affected_functions=[],
            language="python",
            framework="",
        )
        db.store_fix(
            fingerprint=fp1, fix_pattern="fix1", fix_diff="",
            project="alpha", model_used="m", confidence=0.8,
        )
        db.store_fix(
            fingerprint=fp2, fix_pattern="fix2", fix_diff="",
            project="beta", model_used="m", confidence=0.6,
        )
        stats = db.get_stats()
        assert stats["total_fixes"] == 2
        assert stats["projects"] == 2
        assert stats["error_types"] == 2
        assert abs(stats["avg_confidence"] - 0.7) < 0.01

    # --- _compute_similarity ---

    def test_compute_similarity_same_error_type(self) -> None:
        fp = BugFingerprint(
            error_type="TypeError",
            stack_pattern="",
            affected_files=[],
            affected_functions=[],
            language="",
            framework="",
        )
        record = FixRecord(
            id=1, fingerprint="x", error_type="TypeError", stack_pattern="",
            fix_pattern="", fix_diff="", confidence=0.5,
            project="", model_used="", timestamp="",
        )
        sim = DebugMemory._compute_similarity(fp, record)
        assert sim >= 0.4

    def test_compute_similarity_different_error_type(self) -> None:
        fp = BugFingerprint(
            error_type="TypeError",
            stack_pattern="",
            affected_files=[],
            affected_functions=[],
            language="",
            framework="",
        )
        record = FixRecord(
            id=1, fingerprint="x", error_type="ValueError", stack_pattern="",
            fix_pattern="", fix_diff="", confidence=0.5,
            project="", model_used="", timestamp="",
        )
        sim = DebugMemory._compute_similarity(fp, record)
        assert sim < 0.4

    def test_compute_similarity_language_match(self) -> None:
        fp = BugFingerprint(
            error_type="TypeError",
            stack_pattern="",
            affected_files=[],
            affected_functions=[],
            language="python",
            framework="",
        )
        record = FixRecord(
            id=1, fingerprint="x", error_type="TypeError", stack_pattern="",
            fix_pattern="", fix_diff="", confidence=0.5,
            project="", model_used="", timestamp="",
            language="python",
        )
        sim = DebugMemory._compute_similarity(fp, record)
        assert sim >= 0.5  # 0.4 (error) + 0.1 (language)

    def test_compute_similarity_framework_match(self) -> None:
        fp = BugFingerprint(
            error_type="TypeError",
            stack_pattern="",
            affected_files=[],
            affected_functions=[],
            language="python",
            framework="django",
        )
        record = FixRecord(
            id=1, fingerprint="x", error_type="TypeError", stack_pattern="",
            fix_pattern="", fix_diff="", confidence=0.5,
            project="", model_used="", timestamp="",
            language="python", framework="django",
        )
        sim = DebugMemory._compute_similarity(fp, record)
        assert sim >= 0.6  # 0.4 + 0.1 + 0.1

    def test_compute_similarity_file_overlap(self) -> None:
        fp = BugFingerprint(
            error_type="TypeError",
            stack_pattern="",
            affected_files=["app.py", "utils.py"],
            affected_functions=[],
            language="",
            framework="",
        )
        record = FixRecord(
            id=1, fingerprint="x", error_type="TypeError", stack_pattern="",
            fix_pattern="", fix_diff="", confidence=0.5,
            project="", model_used="", timestamp="",
            affected_files_json='["app.py", "models.py"]',
        )
        sim = DebugMemory._compute_similarity(fp, record)
        # error_type=0.4, files: 1 overlap / 3 union * 0.2 ≈ 0.067
        assert sim > 0.4

    def test_compute_similarity_stack_pattern_overlap(self) -> None:
        fp = BugFingerprint(
            error_type="TypeError",
            stack_pattern="File app.py line 42 handler",
            affected_files=[],
            affected_functions=[],
            language="",
            framework="",
        )
        record = FixRecord(
            id=1, fingerprint="x", error_type="TypeError",
            stack_pattern="File app.py line 99 handler",
            fix_pattern="", fix_diff="", confidence=0.5,
            project="", model_used="", timestamp="",
        )
        sim = DebugMemory._compute_similarity(fp, record)
        # error_type=0.4, stack overlap: {"file","app.py","handler","line"}/total * 0.2
        assert sim > 0.4

    def test_compute_similarity_max_one(self) -> None:
        fp = BugFingerprint(
            error_type="TypeError",
            stack_pattern="same words",
            affected_files=["same.py"],
            affected_functions=[],
            language="python",
            framework="django",
        )
        record = FixRecord(
            id=1, fingerprint="x", error_type="TypeError",
            stack_pattern="same words",
            fix_pattern="", fix_diff="", confidence=0.5,
            project="", model_used="", timestamp="",
            language="python", framework="django",
            affected_files_json='["same.py"]',
        )
        sim = DebugMemory._compute_similarity(fp, record)
        assert sim <= 1.0

    def test_compute_similarity_empty_everything(self) -> None:
        fp = BugFingerprint(
            error_type="",
            stack_pattern="",
            affected_files=[],
            affected_functions=[],
            language="",
            framework="",
        )
        record = FixRecord(
            id=1, fingerprint="x", error_type="other", stack_pattern="",
            fix_pattern="", fix_diff="", confidence=0.5,
            project="", model_used="", timestamp="",
        )
        sim = DebugMemory._compute_similarity(fp, record)
        assert sim == 0.0

    # --- cross-project search ---

    def test_cross_project_search(self, db: DebugMemory) -> None:
        fp1 = BugFingerprint(
            error_type="ImportError",
            stack_pattern="ModuleNotFoundError: No module named 'foo'",
            affected_files=["main.py"],
            affected_functions=["start"],
            language="python",
            framework="",
        )
        fp2 = BugFingerprint(
            error_type="ImportError",
            stack_pattern="ModuleNotFoundError: No module named 'bar'",
            affected_files=["entry.py"],
            affected_functions=["init"],
            language="python",
            framework="",
        )
        db.store_fix(
            fingerprint=fp1,
            fix_pattern="pip install foo",
            fix_diff="",
            project="project_one",
            model_used="claude",
        )
        # Search from a different project context
        suggestions = db.search_similar(fp2)
        assert len(suggestions) >= 1
        assert suggestions[0].original_fix.project == "project_one"

    # --- close ---

    def test_close(self, db: DebugMemory) -> None:
        db.close()
        # After close, conn should be None
        conn = getattr(db._local, "conn", None)
        assert conn is None

    def test_close_idempotent(self, db: DebugMemory) -> None:
        db.close()
        db.close()  # should not raise

    # --- edge cases ---

    def test_store_fix_with_empty_strings(self, db: DebugMemory) -> None:
        fp = BugFingerprint(
            error_type="",
            stack_pattern="",
            affected_files=[],
            affected_functions=[],
            language="",
            framework="",
        )
        record = db.store_fix(
            fingerprint=fp,
            fix_pattern="",
            fix_diff="",
            project="",
            model_used="",
        )
        assert record.id > 0

    def test_store_fix_unicode(self, db: DebugMemory) -> None:
        fp = BugFingerprint(
            error_type="UnicodeError",
            stack_pattern="Traceback: \u2603 snowman",
            affected_files=["\u00e9.py"],
            affected_functions=["handle_\u00fc"],
            language="python",
            framework="",
        )
        db.store_fix(
            fingerprint=fp,
            fix_pattern="Encode \u00e9 properly",
            fix_diff="",
            project="intl_app",
            model_used="m",
        )
        fixes = db.browse_fixes()
        assert len(fixes) == 1
        assert "\u2603" in fixes[0].stack_pattern

    def test_browse_fixes_nonexistent_project(self, db: DebugMemory) -> None:
        fixes = db.browse_fixes(project="doesnt_exist")
        assert fixes == []
