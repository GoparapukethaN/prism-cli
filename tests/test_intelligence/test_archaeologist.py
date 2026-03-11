"""Tests for prism.intelligence.archaeologist — Temporal Code Archaeologist."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prism.intelligence.archaeologist import (
    AuthorContribution,
    CodeArchaeologist,
    CodeEvolution,
    CommitEvent,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create a minimal project directory with a fake .git."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def run(): pass\n")
    return tmp_path


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    """Provide a temporary reports directory."""
    d = tmp_path / "archaeology"
    d.mkdir()
    return d


@pytest.fixture
def archaeologist(
    project_root: Path, reports_dir: Path
) -> CodeArchaeologist:
    """Create a CodeArchaeologist with temporary paths."""
    return CodeArchaeologist(
        project_root=project_root, reports_dir=reports_dir
    )


# ======================================================================
# TestCommitEvent
# ======================================================================


class TestCommitEvent:
    """Tests for the CommitEvent dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible and correctly typed."""
        event = CommitEvent(
            commit_hash="abc123def456789012345678901234567890abcd",
            short_hash="abc123d",
            author="Alice",
            date="2025-01-15T10:00:00+00:00",
            message="feat: add startup logic",
            event_type="feature",
            files_changed=3,
            insertions=42,
            deletions=10,
        )
        assert event.commit_hash.startswith("abc123")
        assert event.short_hash == "abc123d"
        assert event.author == "Alice"
        assert event.date == "2025-01-15T10:00:00+00:00"
        assert event.message == "feat: add startup logic"
        assert event.event_type == "feature"
        assert event.files_changed == 3
        assert event.insertions == 42
        assert event.deletions == 10

    def test_event_type_created(self) -> None:
        """Event type 'created' is valid."""
        event = CommitEvent(
            commit_hash="a" * 40,
            short_hash="a" * 7,
            author="X",
            date="2025-01-01",
            message="initial",
            event_type="created",
            files_changed=1,
            insertions=10,
            deletions=0,
        )
        assert event.event_type == "created"

    def test_event_type_bugfix(self) -> None:
        """Event type 'bugfix' is valid."""
        event = CommitEvent(
            commit_hash="b" * 40,
            short_hash="b" * 7,
            author="Y",
            date="2025-02-01",
            message="fix: crash",
            event_type="bugfix",
            files_changed=1,
            insertions=5,
            deletions=3,
        )
        assert event.event_type == "bugfix"

    def test_event_type_refactored(self) -> None:
        """Event type 'refactored' is valid."""
        event = CommitEvent(
            commit_hash="c" * 40,
            short_hash="c" * 7,
            author="Z",
            date="2025-03-01",
            message="refactor: cleanup",
            event_type="refactored",
            files_changed=2,
            insertions=20,
            deletions=15,
        )
        assert event.event_type == "refactored"

    def test_event_type_modified(self) -> None:
        """Event type 'modified' is valid for generic changes."""
        event = CommitEvent(
            commit_hash="d" * 40,
            short_hash="d" * 7,
            author="W",
            date="2025-04-01",
            message="update config",
            event_type="modified",
            files_changed=1,
            insertions=2,
            deletions=1,
        )
        assert event.event_type == "modified"

    def test_zero_changes(self) -> None:
        """Commit with zero insertions/deletions is valid."""
        event = CommitEvent(
            commit_hash="e" * 40,
            short_hash="e" * 7,
            author="V",
            date="2025-05-01",
            message="chore: empty",
            event_type="modified",
            files_changed=0,
            insertions=0,
            deletions=0,
        )
        assert event.files_changed == 0
        assert event.insertions == 0
        assert event.deletions == 0


# ======================================================================
# TestAuthorContribution
# ======================================================================


class TestAuthorContribution:
    """Tests for the AuthorContribution dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible and correctly typed."""
        contrib = AuthorContribution(
            name="Alice",
            commits=15,
            lines_added=300,
            lines_removed=100,
            first_commit="2024-01-01T00:00:00+00:00",
            last_commit="2025-03-01T00:00:00+00:00",
            expertise_score=0.75,
        )
        assert contrib.name == "Alice"
        assert contrib.commits == 15
        assert contrib.lines_added == 300
        assert contrib.lines_removed == 100
        assert contrib.first_commit == "2024-01-01T00:00:00+00:00"
        assert contrib.last_commit == "2025-03-01T00:00:00+00:00"
        assert contrib.expertise_score == 0.75

    def test_expertise_score_range(self) -> None:
        """Expertise score should be between 0.0 and 1.0."""
        contrib = AuthorContribution(
            name="Bob",
            commits=5,
            lines_added=50,
            lines_removed=10,
            first_commit="2025-01-01",
            last_commit="2025-03-01",
            expertise_score=0.25,
        )
        assert 0.0 <= contrib.expertise_score <= 1.0

    def test_single_commit_author(self) -> None:
        """Author with a single commit."""
        contrib = AuthorContribution(
            name="Charlie",
            commits=1,
            lines_added=10,
            lines_removed=0,
            first_commit="2025-06-01",
            last_commit="2025-06-01",
            expertise_score=0.1,
        )
        assert contrib.commits == 1
        assert contrib.first_commit == contrib.last_commit


# ======================================================================
# TestCodeEvolution
# ======================================================================


class TestCodeEvolution:
    """Tests for the CodeEvolution dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible."""
        evo = CodeEvolution(
            target="src/main.py:42",
            file_path="src/main.py",
            timeline=[],
            authors=[],
            narrative="No history.",
            total_commits=0,
            age_days=0,
            stability_score=1.0,
            risk_assessment="Low risk.",
            created_at="2025-03-11T00:00:00+00:00",
        )
        assert evo.target == "src/main.py:42"
        assert evo.file_path == "src/main.py"
        assert evo.timeline == []
        assert evo.authors == []
        assert evo.total_commits == 0
        assert evo.age_days == 0
        assert evo.stability_score == 1.0
        assert evo.risk_assessment == "Low risk."
        assert evo.created_at.startswith("2025")

    def test_stability_score_range(self) -> None:
        """Stability score should be between 0.0 and 1.0."""
        evo = CodeEvolution(
            target="x",
            file_path="x.py",
            timeline=[],
            authors=[],
            narrative="",
            total_commits=0,
            age_days=0,
            stability_score=0.5,
            risk_assessment="",
            created_at="",
        )
        assert 0.0 <= evo.stability_score <= 1.0

    def test_with_timeline_and_authors(self) -> None:
        """Evolution with populated timeline and authors."""
        event = CommitEvent(
            commit_hash="a" * 40,
            short_hash="a" * 7,
            author="Alice",
            date="2025-01-01",
            message="initial",
            event_type="created",
            files_changed=1,
            insertions=100,
            deletions=0,
        )
        author = AuthorContribution(
            name="Alice",
            commits=1,
            lines_added=100,
            lines_removed=0,
            first_commit="2025-01-01",
            last_commit="2025-01-01",
            expertise_score=1.0,
        )
        evo = CodeEvolution(
            target="main.py",
            file_path="main.py",
            timeline=[event],
            authors=[author],
            narrative="Created by Alice.",
            total_commits=1,
            age_days=70,
            stability_score=1.0,
            risk_assessment="Single maintainer.",
            created_at="2025-03-11T00:00:00+00:00",
        )
        assert len(evo.timeline) == 1
        assert len(evo.authors) == 1
        assert evo.total_commits == 1


# ======================================================================
# TestCodeArchaeologist — init
# ======================================================================


class TestArchaeologistInit:
    """Tests for CodeArchaeologist initialisation."""

    def test_resolves_project_root(self, project_root: Path) -> None:
        """Project root is resolved to an absolute path."""
        a = CodeArchaeologist(project_root=project_root)
        assert a._root.is_absolute()

    def test_creates_reports_dir(
        self, project_root: Path, tmp_path: Path
    ) -> None:
        """Reports directory is created if it doesn't exist."""
        rd = tmp_path / "new_archaeology"
        CodeArchaeologist(project_root=project_root, reports_dir=rd)
        assert rd.exists()
        assert rd.is_dir()

    def test_default_reports_dir(self, project_root: Path) -> None:
        """Without explicit reports_dir, uses ~/.prism/archaeology/."""
        a = CodeArchaeologist(project_root=project_root)
        assert a._reports_dir == Path.home() / ".prism" / "archaeology"

    def test_existing_reports_dir_is_fine(
        self, project_root: Path, reports_dir: Path
    ) -> None:
        """No error when reports_dir already exists."""
        a = CodeArchaeologist(
            project_root=project_root, reports_dir=reports_dir
        )
        assert a._reports_dir == reports_dir


# ======================================================================
# TestCodeArchaeologist — _parse_target
# ======================================================================


class TestParseTarget:
    """Tests for the _parse_target method."""

    def test_file_with_line(self, archaeologist: CodeArchaeologist) -> None:
        """'file.py:42' parses correctly."""
        path, line = archaeologist._parse_target("src/main.py:42")
        assert path == "src/main.py"
        assert line == 42

    def test_file_only(self, archaeologist: CodeArchaeologist) -> None:
        """'file.py' returns the file with no line number."""
        path, line = archaeologist._parse_target("src/main.py")
        assert path == "src/main.py"
        assert line is None

    def test_file_with_slash(self, archaeologist: CodeArchaeologist) -> None:
        """Path with slashes is recognised as a file."""
        path, line = archaeologist._parse_target("src/utils/helpers.py")
        assert path == "src/utils/helpers.py"
        assert line is None

    def test_function_name(self, archaeologist: CodeArchaeologist) -> None:
        """A bare function name returns empty string for file_path."""
        path, line = archaeologist._parse_target("my_function")
        assert path == ""
        assert line is None

    def test_file_with_invalid_line(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """'file.py:abc' — non-numeric line returns None for line."""
        path, line = archaeologist._parse_target("file.py:abc")
        assert path == "file.py"
        assert line is None

    def test_file_with_directory_and_line(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """'dir/file.py:100' parses correctly."""
        path, line = archaeologist._parse_target("dir/file.py:100")
        assert path == "dir/file.py"
        assert line == 100

    def test_py_extension_recognised(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Files ending in .py are recognised as file paths."""
        path, line = archaeologist._parse_target("config.py")
        assert path == "config.py"
        assert line is None


# ======================================================================
# TestCodeArchaeologist — _find_function
# ======================================================================


class TestFindFunction:
    """Tests for the _find_function method."""

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_finds_function(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """Finds a function definition via git grep."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="src/main.py:10:def my_function():\n"
        )
        path, line = archaeologist._find_function("my_function")
        assert path == "src/main.py"
        assert line == 10

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_function_not_found(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """Returns empty tuple when function is not found."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        path, line = archaeologist._find_function("nonexistent")
        assert path == ""
        assert line is None

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_multiple_matches_returns_first(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """When multiple matches exist, the first is returned."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "src/a.py:5:def helper():\n"
                "src/b.py:20:def helper():\n"
            ),
        )
        path, line = archaeologist._find_function("helper")
        assert path == "src/a.py"
        assert line == 5

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_match_with_non_numeric_line(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """If the line number part is not numeric, returns None for line."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="src/a.py:xyz:def func():\n"
        )
        path, line = archaeologist._find_function("func")
        assert path == "src/a.py"
        assert line is None


# ======================================================================
# TestCodeArchaeologist — _git_blame
# ======================================================================


class TestGitBlame:
    """Tests for the _git_blame method."""

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_parses_blame_output(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """Porcelain blame output is parsed correctly."""
        blame_output = (
            "abc123def456789012345678901234567890abcd 1 1 1\n"
            "author Alice\n"
            "author-time 1700000000\n"
            "summary feat: initial implementation\n"
            "\tdef run():\n"
        )
        mock_run.return_value = MagicMock(
            returncode=0, stdout=blame_output
        )
        entries = archaeologist._git_blame("src/main.py")
        assert len(entries) == 1
        assert entries[0]["hash"] == (
            "abc123def456789012345678901234567890abcd"
        )
        assert entries[0]["author"] == "Alice"
        assert entries[0]["timestamp"] == "1700000000"
        assert entries[0]["message"] == "feat: initial implementation"
        assert entries[0]["content"] == "def run():"

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_blame_with_line_number(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """Line number is passed as -L argument."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        archaeologist._git_blame("src/main.py", line=42)
        cmd = mock_run.call_args[0][0]
        assert "-L" in cmd
        assert "42,62" in cmd

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_blame_without_line_number(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """Without line number, no -L argument is passed."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        archaeologist._git_blame("src/main.py")
        cmd = mock_run.call_args[0][0]
        assert "-L" not in cmd

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_empty_blame(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """Empty blame output returns empty list."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        entries = archaeologist._git_blame("nonexistent.py")
        assert entries == []

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_multiple_blame_entries(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """Multiple blame entries are parsed."""
        blame_output = (
            "aaaa" + "0" * 36 + " 1 1 1\n"
            "author Alice\n"
            "author-time 1700000000\n"
            "summary first commit\n"
            "\tline one\n"
            "bbbb" + "0" * 36 + " 2 2 1\n"
            "author Bob\n"
            "author-time 1700100000\n"
            "summary second commit\n"
            "\tline two\n"
        )
        mock_run.return_value = MagicMock(
            returncode=0, stdout=blame_output
        )
        entries = archaeologist._git_blame("src/main.py")
        assert len(entries) == 2
        assert entries[0]["author"] == "Alice"
        assert entries[1]["author"] == "Bob"


# ======================================================================
# TestCodeArchaeologist — _git_log_follow
# ======================================================================


class TestGitLogFollow:
    """Tests for the _git_log_follow method."""

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_parses_log_output(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """Log output with numstat is parsed correctly."""
        log_output = (
            "abc123|abc123d|Alice|2025-01-15T10:00:00+00:00|feat: init\n"
            "10\t5\tsrc/main.py\n"
        )
        mock_run.return_value = MagicMock(
            returncode=0, stdout=log_output
        )
        commits = archaeologist._git_log_follow("src/main.py")
        assert len(commits) == 1
        assert commits[0]["hash"] == "abc123"
        assert commits[0]["short_hash"] == "abc123d"
        assert commits[0]["author"] == "Alice"
        assert commits[0]["message"] == "feat: init"
        assert commits[0]["insertions"] == 10
        assert commits[0]["deletions"] == 5
        assert commits[0]["files_changed"] == 1

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_multiple_commits(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """Multiple commits in log output are parsed."""
        log_output = (
            "aaa|aaa|Alice|2025-01-01|first\n"
            "5\t2\tsrc/main.py\n"
            "\n"
            "bbb|bbb|Bob|2025-02-01|second\n"
            "3\t1\tsrc/main.py\n"
        )
        mock_run.return_value = MagicMock(
            returncode=0, stdout=log_output
        )
        commits = archaeologist._git_log_follow("src/main.py")
        assert len(commits) == 2
        assert commits[0]["author"] == "Alice"
        assert commits[1]["author"] == "Bob"

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_empty_log(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """Empty log returns empty list."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        commits = archaeologist._git_log_follow("nonexistent.py")
        assert commits == []

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_binary_numstat_dash(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """Binary files show '-' for insertions/deletions."""
        log_output = (
            "aaa|aaa|Alice|2025-01-01|add binary\n"
            "-\t-\timage.png\n"
        )
        mock_run.return_value = MagicMock(
            returncode=0, stdout=log_output
        )
        commits = archaeologist._git_log_follow("image.png")
        assert len(commits) == 1
        assert commits[0]["insertions"] == 0
        assert commits[0]["deletions"] == 0
        assert commits[0]["files_changed"] == 1

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_commit_with_multiple_files(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """A commit changing multiple files accumulates stats."""
        log_output = (
            "aaa|aaa|Alice|2025-01-01|multi-file\n"
            "10\t5\tsrc/a.py\n"
            "20\t3\tsrc/b.py\n"
        )
        mock_run.return_value = MagicMock(
            returncode=0, stdout=log_output
        )
        commits = archaeologist._git_log_follow("src/a.py")
        assert commits[0]["insertions"] == 30
        assert commits[0]["deletions"] == 8
        assert commits[0]["files_changed"] == 2


# ======================================================================
# TestCodeArchaeologist — _build_timeline
# ======================================================================


class TestBuildTimeline:
    """Tests for the _build_timeline method."""

    def test_builds_timeline_from_log(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Timeline is built in chronological order."""
        log_data = [
            {
                "hash": "bbb",
                "short_hash": "bbb",
                "author": "Bob",
                "date": "2025-02-01",
                "message": "fix: crash",
                "insertions": 5,
                "deletions": 3,
                "files_changed": 1,
            },
            {
                "hash": "aaa",
                "short_hash": "aaa",
                "author": "Alice",
                "date": "2025-01-01",
                "message": "feat: initial",
                "insertions": 100,
                "deletions": 0,
                "files_changed": 1,
            },
        ]
        timeline = archaeologist._build_timeline(log_data)
        # Reversed → chronological: Alice first, Bob second
        assert len(timeline) == 2
        assert timeline[0].author == "Alice"
        assert timeline[0].event_type == "created"
        assert timeline[1].author == "Bob"
        assert timeline[1].event_type == "bugfix"

    def test_empty_log_data(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Empty log data produces empty timeline."""
        timeline = archaeologist._build_timeline([])
        assert timeline == []

    def test_single_commit(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Single commit is classified as 'created'."""
        log_data = [
            {
                "hash": "aaa",
                "short_hash": "aaa",
                "author": "Alice",
                "date": "2025-01-01",
                "message": "anything",
                "insertions": 10,
                "deletions": 0,
                "files_changed": 1,
            },
        ]
        timeline = archaeologist._build_timeline(log_data)
        assert len(timeline) == 1
        assert timeline[0].event_type == "created"


# ======================================================================
# TestCodeArchaeologist — _classify_commit
# ======================================================================


class TestClassifyCommit:
    """Tests for the static _classify_commit method."""

    def test_first_commit_is_created(self) -> None:
        """The first commit is always classified as 'created'."""
        assert CodeArchaeologist._classify_commit("anything", True) == "created"

    def test_fix_keyword(self) -> None:
        """'fix' in message maps to 'bugfix'."""
        assert CodeArchaeologist._classify_commit("fix: crash", False) == "bugfix"

    def test_bug_keyword(self) -> None:
        """'bug' in message maps to 'bugfix'."""
        assert CodeArchaeologist._classify_commit("fix bug #123", False) == "bugfix"

    def test_patch_keyword(self) -> None:
        """'patch' in message maps to 'bugfix'."""
        assert CodeArchaeologist._classify_commit("patch: security", False) == "bugfix"

    def test_hotfix_keyword(self) -> None:
        """'hotfix' in message maps to 'bugfix'."""
        assert CodeArchaeologist._classify_commit("hotfix: urgent", False) == "bugfix"

    def test_issue_keyword(self) -> None:
        """'issue' in message maps to 'bugfix'."""
        assert CodeArchaeologist._classify_commit("resolve issue #42", False) == "bugfix"

    def test_refactor_keyword(self) -> None:
        """'refactor' in message maps to 'refactored'."""
        assert CodeArchaeologist._classify_commit("refactor: cleanup", False) == "refactored"

    def test_cleanup_keyword(self) -> None:
        """'cleanup' in message maps to 'refactored'."""
        assert CodeArchaeologist._classify_commit("cleanup dead code", False) == "refactored"

    def test_reorganize_keyword(self) -> None:
        """'reorganize' in message maps to 'refactored'."""
        assert CodeArchaeologist._classify_commit("reorganize modules", False) == "refactored"

    def test_rename_keyword(self) -> None:
        """'rename' in message maps to 'refactored'."""
        assert CodeArchaeologist._classify_commit("rename variable", False) == "refactored"

    def test_restructure_keyword(self) -> None:
        """'restructure' in message maps to 'refactored'."""
        assert CodeArchaeologist._classify_commit("restructure project", False) == "refactored"

    def test_feat_keyword(self) -> None:
        """'feat' in message maps to 'feature'."""
        assert CodeArchaeologist._classify_commit("feat: new widget", False) == "feature"

    def test_add_keyword(self) -> None:
        """'add' in message maps to 'feature'."""
        assert CodeArchaeologist._classify_commit("add search bar", False) == "feature"

    def test_implement_keyword(self) -> None:
        """'implement' in message maps to 'feature'."""
        assert CodeArchaeologist._classify_commit("implement caching", False) == "feature"

    def test_new_keyword(self) -> None:
        """'new' in message maps to 'feature'."""
        assert CodeArchaeologist._classify_commit("new endpoint", False) == "feature"

    def test_introduce_keyword(self) -> None:
        """'introduce' in message maps to 'feature'."""
        assert CodeArchaeologist._classify_commit("introduce rate limit", False) == "feature"

    def test_generic_message(self) -> None:
        """A message without recognised keywords maps to 'modified'."""
        assert CodeArchaeologist._classify_commit("update config values", False) == "modified"

    def test_case_insensitive(self) -> None:
        """Classification is case-insensitive."""
        assert CodeArchaeologist._classify_commit("FIX: Crash", False) == "bugfix"
        assert CodeArchaeologist._classify_commit("REFACTOR: code", False) == "refactored"
        assert CodeArchaeologist._classify_commit("FEAT: thing", False) == "feature"

    def test_bugfix_takes_priority_over_feature(self) -> None:
        """When both fix and feat keywords present, bugfix wins."""
        result = CodeArchaeologist._classify_commit("fix: add error handling", False)
        assert result == "bugfix"


# ======================================================================
# TestCodeArchaeologist — _analyze_authors
# ======================================================================


class TestAnalyzeAuthors:
    """Tests for the _analyze_authors method."""

    def test_single_author(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Single author gets 100% expertise."""
        log_data = [
            {
                "hash": "aaa",
                "author": "Alice",
                "date": "2025-01-01",
                "insertions": 50,
                "deletions": 10,
            },
            {
                "hash": "bbb",
                "author": "Alice",
                "date": "2025-02-01",
                "insertions": 30,
                "deletions": 5,
            },
        ]
        authors = archaeologist._analyze_authors(log_data)
        assert len(authors) == 1
        assert authors[0].name == "Alice"
        assert authors[0].commits == 2
        assert authors[0].lines_added == 80
        assert authors[0].lines_removed == 15
        assert authors[0].expertise_score == 1.0

    def test_multiple_authors(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Multiple authors are sorted by commit count."""
        log_data = [
            {"hash": "1", "author": "Alice", "date": "2025-01-01", "insertions": 10, "deletions": 0},
            {"hash": "2", "author": "Bob", "date": "2025-01-02", "insertions": 20, "deletions": 5},
            {"hash": "3", "author": "Bob", "date": "2025-01-03", "insertions": 15, "deletions": 3},
            {"hash": "4", "author": "Alice", "date": "2025-01-04", "insertions": 5, "deletions": 1},
            {"hash": "5", "author": "Alice", "date": "2025-01-05", "insertions": 8, "deletions": 2},
        ]
        authors = archaeologist._analyze_authors(log_data)
        assert len(authors) == 2
        assert authors[0].name == "Alice"  # 3 commits > Bob's 2
        assert authors[0].commits == 3
        assert authors[1].name == "Bob"
        assert authors[1].commits == 2

    def test_expertise_scores_sum_to_one(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """All expertise scores sum to approximately 1.0."""
        log_data = [
            {"hash": "1", "author": "A", "date": "2025-01-01", "insertions": 0, "deletions": 0},
            {"hash": "2", "author": "B", "date": "2025-01-02", "insertions": 0, "deletions": 0},
            {"hash": "3", "author": "A", "date": "2025-01-03", "insertions": 0, "deletions": 0},
            {"hash": "4", "author": "C", "date": "2025-01-04", "insertions": 0, "deletions": 0},
        ]
        authors = archaeologist._analyze_authors(log_data)
        total = sum(a.expertise_score for a in authors)
        assert abs(total - 1.0) < 0.05  # Allow small rounding error

    def test_empty_log(self, archaeologist: CodeArchaeologist) -> None:
        """Empty log returns empty authors list."""
        authors = archaeologist._analyze_authors([])
        assert authors == []

    def test_first_and_last_commit_dates(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """First and last commit dates are tracked correctly."""
        log_data = [
            {"hash": "1", "author": "Alice", "date": "2025-01-01", "insertions": 0, "deletions": 0},
            {"hash": "2", "author": "Alice", "date": "2025-03-01", "insertions": 0, "deletions": 0},
            {"hash": "3", "author": "Alice", "date": "2025-06-01", "insertions": 0, "deletions": 0},
        ]
        authors = archaeologist._analyze_authors(log_data)
        assert authors[0].first_commit == "2025-01-01"
        assert authors[0].last_commit == "2025-06-01"


# ======================================================================
# TestCodeArchaeologist — _build_narrative
# ======================================================================


class TestBuildNarrative:
    """Tests for the _build_narrative method."""

    def _make_event(
        self,
        author: str = "Alice",
        date: str = "2025-01-01",
        message: str = "initial",
        event_type: str = "created",
    ) -> CommitEvent:
        """Helper to create a CommitEvent."""
        return CommitEvent(
            commit_hash="a" * 40,
            short_hash="a" * 7,
            author=author,
            date=date,
            message=message,
            event_type=event_type,
            files_changed=1,
            insertions=10,
            deletions=0,
        )

    def _make_author(
        self,
        name: str = "Alice",
        commits: int = 5,
        lines_added: int = 100,
        expertise: float = 0.8,
    ) -> AuthorContribution:
        """Helper to create an AuthorContribution."""
        return AuthorContribution(
            name=name,
            commits=commits,
            lines_added=lines_added,
            lines_removed=20,
            first_commit="2025-01-01",
            last_commit="2025-03-01",
            expertise_score=expertise,
        )

    def test_empty_timeline(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Empty timeline produces a 'no history' message."""
        narrative = archaeologist._build_narrative(
            "my_func", "main.py", [], [], []
        )
        assert "No git history found" in narrative

    def test_includes_target_heading(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Narrative heading includes the target."""
        event = self._make_event()
        narrative = archaeologist._build_narrative(
            "main.py:42", "main.py", [event], [], []
        )
        assert "main.py:42" in narrative

    def test_includes_creator(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Narrative mentions the original author."""
        event = self._make_event(author="Charlie")
        narrative = archaeologist._build_narrative(
            "f.py", "f.py", [event], [], []
        )
        assert "Charlie" in narrative
        assert "**Created**" in narrative

    def test_includes_creation_date(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Narrative mentions the creation date."""
        event = self._make_event(date="2024-06-15T10:00:00+00:00")
        narrative = archaeologist._build_narrative(
            "f.py", "f.py", [event], [], []
        )
        assert "2024-06-15" in narrative

    def test_feature_count(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Feature additions are counted in the narrative."""
        events = [
            self._make_event(event_type="created"),
            self._make_event(event_type="feature"),
            self._make_event(event_type="feature"),
        ]
        narrative = archaeologist._build_narrative(
            "f.py", "f.py", events, [], []
        )
        assert "2 commits added new functionality" in narrative

    def test_bugfix_count(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Bug fixes are counted in the narrative."""
        events = [
            self._make_event(event_type="created"),
            self._make_event(event_type="bugfix"),
        ]
        narrative = archaeologist._build_narrative(
            "f.py", "f.py", events, [], []
        )
        assert "1 commits fixed bugs" in narrative

    def test_refactor_count(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Refactors are counted in the narrative."""
        events = [
            self._make_event(event_type="created"),
            self._make_event(event_type="refactored"),
            self._make_event(event_type="refactored"),
            self._make_event(event_type="refactored"),
        ]
        narrative = archaeologist._build_narrative(
            "f.py", "f.py", events, [], []
        )
        assert "3 commits restructured" in narrative

    def test_primary_maintainer(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Primary maintainer is mentioned."""
        event = self._make_event()
        author = self._make_author(name="Dave", commits=10, expertise=0.85)
        narrative = archaeologist._build_narrative(
            "f.py", "f.py", [event], [author], []
        )
        assert "Dave" in narrative
        assert "Primary maintainer" in narrative

    def test_latest_change(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Latest change is mentioned at the end."""
        events = [
            self._make_event(
                author="Alice", date="2025-01-01", message="first",
                event_type="created",
            ),
            self._make_event(
                author="Bob", date="2025-06-15", message="latest update",
                event_type="modified",
            ),
        ]
        narrative = archaeologist._build_narrative(
            "f.py", "f.py", events, [], []
        )
        assert "Latest change" in narrative
        assert "2025-06-15" in narrative
        assert "Bob" in narrative
        assert "latest update" in narrative


# ======================================================================
# TestCodeArchaeologist — _calculate_age
# ======================================================================


class TestCalculateAge:
    """Tests for the static _calculate_age method."""

    def test_empty_timeline(self) -> None:
        """Empty timeline returns 0 days."""
        assert CodeArchaeologist._calculate_age([]) == 0

    def test_recent_commit(self) -> None:
        """A very recent commit returns a small age."""
        now = datetime.now(UTC)
        yesterday = (now - timedelta(days=1)).isoformat()
        event = CommitEvent(
            commit_hash="a" * 40,
            short_hash="a" * 7,
            author="A",
            date=yesterday,
            message="x",
            event_type="created",
            files_changed=1,
            insertions=1,
            deletions=0,
        )
        age = CodeArchaeologist._calculate_age([event])
        assert age >= 1
        assert age <= 2

    def test_old_commit(self) -> None:
        """An old commit returns a large age."""
        event = CommitEvent(
            commit_hash="a" * 40,
            short_hash="a" * 7,
            author="A",
            date="2020-01-01T00:00:00+00:00",
            message="x",
            event_type="created",
            files_changed=1,
            insertions=1,
            deletions=0,
        )
        age = CodeArchaeologist._calculate_age([event])
        assert age > 365  # More than a year old

    def test_invalid_date_returns_zero(self) -> None:
        """Invalid date string returns 0."""
        event = CommitEvent(
            commit_hash="a" * 40,
            short_hash="a" * 7,
            author="A",
            date="not-a-date",
            message="x",
            event_type="created",
            files_changed=1,
            insertions=1,
            deletions=0,
        )
        assert CodeArchaeologist._calculate_age([event]) == 0

    def test_z_suffix_handled(self) -> None:
        """Dates with 'Z' suffix are handled correctly."""
        event = CommitEvent(
            commit_hash="a" * 40,
            short_hash="a" * 7,
            author="A",
            date="2024-01-01T00:00:00Z",
            message="x",
            event_type="created",
            files_changed=1,
            insertions=1,
            deletions=0,
        )
        age = CodeArchaeologist._calculate_age([event])
        assert age > 0


# ======================================================================
# TestCodeArchaeologist — _calculate_stability
# ======================================================================


class TestCalculateStability:
    """Tests for the static _calculate_stability method."""

    def test_empty_timeline(self) -> None:
        """Empty timeline is maximally stable."""
        assert CodeArchaeologist._calculate_stability([]) == 1.0

    def test_single_commit(self) -> None:
        """Single commit is maximally stable."""
        event = CommitEvent(
            commit_hash="a" * 40, short_hash="a", author="A",
            date="2025-01-01", message="x", event_type="created",
            files_changed=1, insertions=1, deletions=0,
        )
        assert CodeArchaeologist._calculate_stability([event]) == 1.0

    def test_few_commits_high_stability(self) -> None:
        """2-5 commits yield 0.9 stability."""
        events = [
            CommitEvent(
                commit_hash=f"{i}" * 40, short_hash=str(i), author="A",
                date="2025-01-01", message="x", event_type="modified",
                files_changed=1, insertions=1, deletions=0,
            )
            for i in range(3)
        ]
        assert CodeArchaeologist._calculate_stability(events) == 0.9

    def test_moderate_commits(self) -> None:
        """6-10 commits yield 0.8 stability."""
        events = [
            CommitEvent(
                commit_hash=f"{i}" * 40, short_hash=str(i), author="A",
                date="2025-01-01", message="x", event_type="modified",
                files_changed=1, insertions=1, deletions=0,
            )
            for i in range(8)
        ]
        assert CodeArchaeologist._calculate_stability(events) == 0.8

    def test_many_commits(self) -> None:
        """11-20 commits yield 0.6 stability."""
        events = [
            CommitEvent(
                commit_hash=f"{i}" * 40, short_hash=str(i), author="A",
                date="2025-01-01", message="x", event_type="modified",
                files_changed=1, insertions=1, deletions=0,
            )
            for i in range(15)
        ]
        assert CodeArchaeologist._calculate_stability(events) == 0.6

    def test_frequent_changes(self) -> None:
        """21-50 commits yield 0.4 stability."""
        events = [
            CommitEvent(
                commit_hash=f"{i}" * 40, short_hash=str(i), author="A",
                date="2025-01-01", message="x", event_type="modified",
                files_changed=1, insertions=1, deletions=0,
            )
            for i in range(30)
        ]
        assert CodeArchaeologist._calculate_stability(events) == 0.4

    def test_very_volatile(self) -> None:
        """50+ commits yield 0.2 stability."""
        events = [
            CommitEvent(
                commit_hash=f"{i}" * 40, short_hash=str(i), author="A",
                date="2025-01-01", message="x", event_type="modified",
                files_changed=1, insertions=1, deletions=0,
            )
            for i in range(60)
        ]
        assert CodeArchaeologist._calculate_stability(events) == 0.2


# ======================================================================
# TestCodeArchaeologist — _assess_risk
# ======================================================================


class TestAssessRisk:
    """Tests for the static _assess_risk method."""

    def _make_events(
        self, count: int, bugfix_count: int = 0
    ) -> list[CommitEvent]:
        """Helper to create a list of events."""
        events = []
        for i in range(count):
            event_type = "bugfix" if i < bugfix_count else "modified"
            events.append(CommitEvent(
                commit_hash=f"{i}" * 40,
                short_hash=str(i),
                author="A",
                date="2025-01-01",
                message="x",
                event_type=event_type,
                files_changed=1,
                insertions=1,
                deletions=0,
            ))
        return events

    def _make_authors(self, *names: str) -> list[AuthorContribution]:
        """Helper to create author contributions."""
        return [
            AuthorContribution(
                name=n, commits=5, lines_added=50, lines_removed=10,
                first_commit="2025-01-01", last_commit="2025-03-01",
                expertise_score=1.0 / len(names),
            )
            for n in names
        ]

    def test_low_risk(self) -> None:
        """Stable code with multiple maintainers is low risk."""
        events = self._make_events(3)
        authors = self._make_authors("Alice", "Bob")
        risk = CodeArchaeologist._assess_risk(events, authors, 0.9)
        assert "Low risk" in risk

    def test_volatile_code(self) -> None:
        """Volatile code (stability < 0.5) is flagged."""
        events = self._make_events(30)
        authors = self._make_authors("Alice", "Bob")
        risk = CodeArchaeologist._assess_risk(events, authors, 0.3)
        assert "volatile" in risk.lower()

    def test_single_maintainer(self) -> None:
        """Single maintainer is flagged as bus factor = 1."""
        events = self._make_events(3)
        authors = self._make_authors("Alice")
        risk = CodeArchaeologist._assess_risk(events, authors, 0.9)
        assert "bus factor" in risk.lower()
        assert "Alice" in risk

    def test_high_bug_rate(self) -> None:
        """More than 30% bugfix commits triggers warning."""
        events = self._make_events(10, bugfix_count=5)
        authors = self._make_authors("Alice", "Bob")
        risk = CodeArchaeologist._assess_risk(events, authors, 0.8)
        assert "bug rate" in risk.lower()

    def test_multiple_risks(self) -> None:
        """Multiple risks are combined with semicolons."""
        events = self._make_events(10, bugfix_count=5)
        authors = self._make_authors("Alice")
        risk = CodeArchaeologist._assess_risk(events, authors, 0.3)
        assert "Risks identified:" in risk
        assert ";" in risk

    def test_no_bugfix_below_threshold(self) -> None:
        """Less than 30% bugfix does not trigger warning."""
        events = self._make_events(10, bugfix_count=2)
        authors = self._make_authors("Alice", "Bob")
        risk = CodeArchaeologist._assess_risk(events, authors, 0.8)
        assert "Low risk" in risk


# ======================================================================
# TestCodeArchaeologist — _save_report and list_reports
# ======================================================================


class TestSaveAndListReports:
    """Tests for report persistence."""

    def _make_evolution(self, target: str = "main.py") -> CodeEvolution:
        """Helper to create a minimal CodeEvolution."""
        return CodeEvolution(
            target=target,
            file_path=f"src/{target}",
            timeline=[],
            authors=[],
            narrative="Test narrative.",
            total_commits=0,
            age_days=0,
            stability_score=1.0,
            risk_assessment="Low risk.",
            created_at="2025-03-11T00:00:00+00:00",
        )

    def test_save_creates_json_file(
        self,
        archaeologist: CodeArchaeologist,
        reports_dir: Path,
    ) -> None:
        """_save_report creates a JSON file in the reports directory."""
        evo = self._make_evolution()
        path = archaeologist._save_report(evo)
        assert path.exists()
        assert path.suffix == ".json"
        assert path.parent == reports_dir

    def test_save_file_contents(
        self,
        archaeologist: CodeArchaeologist,
    ) -> None:
        """Saved file contains valid JSON with expected fields."""
        evo = self._make_evolution("utils.py")
        path = archaeologist._save_report(evo)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["target"] == "utils.py"
        assert data["file_path"] == "src/utils.py"
        assert data["narrative"] == "Test narrative."
        assert data["stability_score"] == 1.0

    def test_save_filename_contains_target(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Filename contains a sanitised version of the target."""
        evo = self._make_evolution("main.py:42")
        path = archaeologist._save_report(evo)
        assert "why_" in path.name
        assert "main" in path.name

    def test_save_filename_sanitised(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Special characters in target are sanitised in filename."""
        evo = self._make_evolution("dir/sub/file.py:100")
        path = archaeologist._save_report(evo)
        # No slashes or colons in filename
        assert "/" not in path.name
        assert ":" not in path.name

    def test_list_reports_sorted(
        self,
        archaeologist: CodeArchaeologist,
        reports_dir: Path,
    ) -> None:
        """Reports are listed newest first."""
        (reports_dir / "why_aaa_2025-01-01.json").write_text("{}")
        (reports_dir / "why_bbb_2025-01-02.json").write_text("{}")
        reports = archaeologist.list_reports()
        assert len(reports) == 2
        assert "bbb" in reports[0].name

    def test_list_reports_empty(
        self, archaeologist: CodeArchaeologist, reports_dir: Path
    ) -> None:
        """Empty reports directory returns empty list."""
        # Ensure directory is empty
        for f in reports_dir.iterdir():
            f.unlink()
        assert archaeologist.list_reports() == []

    def test_list_reports_ignores_non_why_files(
        self,
        archaeologist: CodeArchaeologist,
        reports_dir: Path,
    ) -> None:
        """Only files matching 'why_*.json' are listed."""
        (reports_dir / "why_main_2025-01-01.json").write_text("{}")
        (reports_dir / "other_report.json").write_text("{}")
        (reports_dir / "readme.txt").write_text("hello")
        reports = archaeologist.list_reports()
        assert len(reports) == 1
        assert "why_" in reports[0].name

    def test_load_report(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """A saved report can be loaded back."""
        event = CommitEvent(
            commit_hash="a" * 40,
            short_hash="a" * 7,
            author="Alice",
            date="2025-01-01",
            message="initial",
            event_type="created",
            files_changed=1,
            insertions=10,
            deletions=0,
        )
        author = AuthorContribution(
            name="Alice",
            commits=1,
            lines_added=10,
            lines_removed=0,
            first_commit="2025-01-01",
            last_commit="2025-01-01",
            expertise_score=1.0,
        )
        evo = CodeEvolution(
            target="main.py",
            file_path="src/main.py",
            timeline=[event],
            authors=[author],
            narrative="Test.",
            total_commits=1,
            age_days=5,
            stability_score=1.0,
            risk_assessment="Low risk.",
            created_at="2025-03-11T00:00:00+00:00",
        )
        path = archaeologist._save_report(evo)
        loaded = archaeologist.load_report(path)
        assert loaded.target == "main.py"
        assert len(loaded.timeline) == 1
        assert loaded.timeline[0].author == "Alice"
        assert len(loaded.authors) == 1
        assert loaded.authors[0].name == "Alice"
        assert loaded.stability_score == 1.0

    def test_load_report_file_not_found(
        self, archaeologist: CodeArchaeologist
    ) -> None:
        """Loading a nonexistent report raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            archaeologist.load_report(Path("/nonexistent/why_report.json"))


# ======================================================================
# TestCodeArchaeologist — full investigate
# ======================================================================


class TestInvestigate:
    """Tests for the full investigate method with all git commands mocked."""

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_investigate_file(
        self,
        mock_run: MagicMock,
        archaeologist: CodeArchaeologist,
    ) -> None:
        """Investigate a file target."""
        def side_effect(*args, **kwargs):
            cmd = args[0]
            if cmd[1] == "blame":
                return MagicMock(
                    returncode=0,
                    stdout=(
                        "a" * 40 + " 1 1 1\n"
                        "author Alice\n"
                        "author-time 1700000000\n"
                        "summary feat: initial\n"
                        "\tdef run():\n"
                    ),
                )
            if cmd[1] == "log":
                return MagicMock(
                    returncode=0,
                    stdout=(
                        "aaa|aaa|Alice|2025-01-15T10:00:00+00:00|"
                        "feat: initial\n"
                        "10\t0\tsrc/main.py\n"
                    ),
                )
            return MagicMock(returncode=0, stdout="")

        mock_run.side_effect = side_effect
        evo = archaeologist.investigate("src/main.py")
        assert evo.target == "src/main.py"
        assert evo.file_path == "src/main.py"
        assert evo.total_commits == 1
        assert evo.stability_score == 1.0
        # Single author triggers bus factor warning
        assert "bus factor" in evo.risk_assessment.lower()
        assert evo.created_at

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_investigate_file_with_line(
        self,
        mock_run: MagicMock,
        archaeologist: CodeArchaeologist,
    ) -> None:
        """Investigate a file:line target."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        # Even with empty git output, it should complete
        evo = archaeologist.investigate("src/main.py:10")
        assert evo.target == "src/main.py:10"
        assert evo.file_path == "src/main.py"

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_investigate_function(
        self,
        mock_run: MagicMock,
        archaeologist: CodeArchaeologist,
    ) -> None:
        """Investigate a function name target."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            cmd = args[0]
            if cmd[1] == "grep":
                return MagicMock(
                    returncode=0,
                    stdout="src/main.py:5:def my_function():\n",
                )
            if cmd[1] == "blame":
                return MagicMock(returncode=0, stdout="")
            if cmd[1] == "log":
                return MagicMock(
                    returncode=0,
                    stdout=(
                        "abc|abc|Alice|2025-01-01|"
                        "feat: add my_function\n"
                        "20\t0\tsrc/main.py\n"
                    ),
                )
            return MagicMock(returncode=0, stdout="")

        mock_run.side_effect = side_effect
        evo = archaeologist.investigate("my_function")
        assert evo.target == "my_function"
        assert evo.file_path == "src/main.py"

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_investigate_not_found(
        self,
        mock_run: MagicMock,
        archaeologist: CodeArchaeologist,
    ) -> None:
        """Investigating a nonexistent function raises ValueError."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        with pytest.raises(ValueError, match="Could not find target"):
            archaeologist.investigate("nonexistent_function")

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_investigate_saves_report(
        self,
        mock_run: MagicMock,
        archaeologist: CodeArchaeologist,
        reports_dir: Path,
    ) -> None:
        """Investigate saves a report file."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        archaeologist.investigate("src/main.py")
        reports = list(reports_dir.glob("why_*.json"))
        assert len(reports) == 1

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_investigate_report_has_created_at(
        self,
        mock_run: MagicMock,
        archaeologist: CodeArchaeologist,
    ) -> None:
        """The evolution has a non-empty created_at timestamp."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        evo = archaeologist.investigate("src/main.py")
        assert evo.created_at
        assert "T" in evo.created_at

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_investigate_with_rich_history(
        self,
        mock_run: MagicMock,
        archaeologist: CodeArchaeologist,
    ) -> None:
        """Investigate with multiple commits and authors."""
        def side_effect(*args, **kwargs):
            cmd = args[0]
            if cmd[1] == "blame":
                return MagicMock(returncode=0, stdout="")
            if cmd[1] == "log":
                return MagicMock(
                    returncode=0,
                    stdout=(
                        "ccc|ccc|Bob|2025-03-01|fix: crash\n"
                        "5\t3\tsrc/main.py\n"
                        "\n"
                        "bbb|bbb|Alice|2025-02-01|feat: add widget\n"
                        "30\t0\tsrc/main.py\n"
                        "\n"
                        "aaa|aaa|Alice|2025-01-01|feat: initial\n"
                        "100\t0\tsrc/main.py\n"
                    ),
                )
            return MagicMock(returncode=0, stdout="")

        mock_run.side_effect = side_effect
        evo = archaeologist.investigate("src/main.py")
        assert evo.total_commits == 3
        assert len(evo.timeline) == 3
        assert evo.timeline[0].event_type == "created"
        assert evo.timeline[2].event_type == "bugfix"
        assert len(evo.authors) == 2
        # Alice has 2 commits, Bob has 1
        assert evo.authors[0].name == "Alice"
        assert evo.authors[0].commits == 2
        assert evo.authors[1].name == "Bob"
        assert evo.authors[1].commits == 1


# ======================================================================
# TestCodeArchaeologist — _git helper
# ======================================================================


class TestGitHelper:
    """Tests for the _git helper method."""

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_passes_args_to_git(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """Arguments are forwarded to the git command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="output")
        result = archaeologist._git("status", "--short")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd == ["git", "status", "--short"]
        assert result == "output"

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_uses_project_root_as_cwd(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """Git commands run from the project root directory."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        archaeologist._git("log")
        kwargs = mock_run.call_args[1]
        assert kwargs["cwd"] == archaeologist._root

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_returns_empty_on_failure(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """Non-zero return code still returns stdout (may be empty)."""
        mock_run.return_value = MagicMock(returncode=128, stdout="")
        result = archaeologist._git("log", "-1")
        assert result == ""

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_timeout_is_set(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """A timeout is set on subprocess.run."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        archaeologist._git("log")
        kwargs = mock_run.call_args[1]
        assert kwargs["timeout"] == 30

    @patch("prism.intelligence.archaeologist.subprocess.run")
    def test_capture_output_enabled(
        self, mock_run: MagicMock, archaeologist: CodeArchaeologist
    ) -> None:
        """capture_output is True."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        archaeologist._git("log")
        kwargs = mock_run.call_args[1]
        assert kwargs["capture_output"] is True
