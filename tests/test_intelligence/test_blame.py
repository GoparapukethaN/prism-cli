"""Tests for prism.intelligence.blame — Causal Blame Tracer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prism.intelligence.blame import (
    BisectResult,
    BlameReport,
    CausalBlameTracer,
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
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("def test_run(): pass\n")
    return tmp_path


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    """Provide a temporary reports directory."""
    d = tmp_path / "blame_reports"
    d.mkdir()
    return d


@pytest.fixture
def tracer(project_root: Path, reports_dir: Path) -> CausalBlameTracer:
    """Create a CausalBlameTracer with temporary paths."""
    return CausalBlameTracer(
        project_root=project_root, reports_dir=reports_dir
    )


# ======================================================================
# TestBlameReport
# ======================================================================


class TestBlameReport:
    """Tests for the BlameReport dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible."""
        report = BlameReport(
            bug_description="Crash on startup",
            breaking_commit="abc123def456",
            breaking_author="Alice",
            breaking_date="2025-01-15T10:00:00+00:00",
            breaking_message="feat: add startup logic",
            affected_files=["src/main.py"],
            affected_lines=["src/main.py: import sys"],
            causal_narrative="The crash was introduced ...",
            confidence=0.7,
            proposed_fix="Revert the import",
            related_tests=["tests/test_main.py"],
            bisect_steps=5,
            created_at="2025-03-11T00:00:00+00:00",
        )
        assert report.bug_description == "Crash on startup"
        assert report.breaking_commit == "abc123def456"
        assert report.breaking_author == "Alice"
        assert report.confidence == 0.7
        assert report.bisect_steps == 5
        assert len(report.affected_files) == 1
        assert len(report.related_tests) == 1

    def test_empty_fields(self) -> None:
        """Report can be created with empty collections."""
        report = BlameReport(
            bug_description="",
            breaking_commit="",
            breaking_author="",
            breaking_date="",
            breaking_message="",
            affected_files=[],
            affected_lines=[],
            causal_narrative="",
            confidence=0.0,
            proposed_fix="",
            related_tests=[],
            bisect_steps=0,
            created_at="",
        )
        assert report.affected_files == []
        assert report.confidence == 0.0


# ======================================================================
# TestBisectResult
# ======================================================================


class TestBisectResult:
    """Tests for the BisectResult dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible."""
        result = BisectResult(
            breaking_commit="abc123",
            total_steps=7,
            good_commit="def456",
            bad_commit="HEAD",
        )
        assert result.breaking_commit == "abc123"
        assert result.total_steps == 7
        assert result.good_commit == "def456"
        assert result.bad_commit == "HEAD"

    def test_zero_steps(self) -> None:
        """Bisect can complete in 0 steps (adjacent commits)."""
        result = BisectResult(
            breaking_commit="x", total_steps=0, good_commit="a", bad_commit="b"
        )
        assert result.total_steps == 0


# ======================================================================
# TestCausalBlameTracer — init
# ======================================================================


class TestTracerInit:
    """Tests for CausalBlameTracer initialisation."""

    def test_resolves_project_root(self, project_root: Path) -> None:
        """Project root is resolved to an absolute path."""
        t = CausalBlameTracer(project_root=project_root)
        assert t._root.is_absolute()

    def test_creates_reports_dir(self, project_root: Path, tmp_path: Path) -> None:
        """Reports directory is created if it doesn't exist."""
        rd = tmp_path / "new_reports"
        CausalBlameTracer(project_root=project_root, reports_dir=rd)
        assert rd.exists()

    def test_default_reports_dir(self, project_root: Path) -> None:
        """Without explicit reports_dir, uses ~/.prism/blame_reports/."""
        t = CausalBlameTracer(project_root=project_root)
        assert t._reports_dir == Path.home() / ".prism" / "blame_reports"


# ======================================================================
# TestCausalBlameTracer — _find_relevant_code
# ======================================================================


class TestFindRelevantCode:
    """Tests for the _find_relevant_code method."""

    @patch("prism.intelligence.blame.subprocess.run")
    def test_searches_keywords(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Keywords from the description are searched with git grep."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="src/main.py\nsrc/utils.py\n"
        )
        result = tracer._find_relevant_code("crash in startup handler")
        assert "src/main.py" in result
        assert "src/utils.py" in result

    @patch("prism.intelligence.blame.subprocess.run")
    def test_skips_short_words(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Words with 3 or fewer characters are skipped."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        tracer._find_relevant_code("a is not working")
        # Only "working" (and "not" which is 3 chars, excluded) should be searched
        calls = mock_run.call_args_list
        keywords_searched = []
        for c in calls:
            args = c[0][0]  # first positional arg (the command list)
            keywords_searched.append(args[-1])  # last arg is the keyword
        assert "a" not in keywords_searched
        assert "is" not in keywords_searched

    @patch("prism.intelligence.blame.subprocess.run")
    def test_limits_to_20_files(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Results are capped at 20 files."""
        files = "\n".join(f"file_{i}.py" for i in range(30))
        mock_run.return_value = MagicMock(returncode=0, stdout=files)
        result = tracer._find_relevant_code("crash problem error failure debug")
        assert len(result) <= 20

    @patch("prism.intelligence.blame.subprocess.run")
    def test_handles_git_grep_failure(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Git grep returning non-zero is handled gracefully."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = tracer._find_relevant_code("nonexistent module")
        assert result == []

    @patch("prism.intelligence.blame.subprocess.run")
    def test_deduplicates_files(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Duplicate file results from multiple keywords are deduplicated."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="src/main.py\nsrc/main.py\n"
        )
        result = tracer._find_relevant_code("crash startup handler")
        assert result.count("src/main.py") == 1


# ======================================================================
# TestCausalBlameTracer — _get_commit_info
# ======================================================================


class TestGetCommitInfo:
    """Tests for the _get_commit_info method."""

    @patch("prism.intelligence.blame.subprocess.run")
    def test_parses_commit_info(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Commit metadata is parsed correctly from git log output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "abc123def456\n"
                "Alice\n"
                "2025-01-15T10:00:00+00:00\n"
                "feat: add startup logic\n"
                "Extended description here\n"
            ),
        )
        info = tracer._get_commit_info("abc123")
        assert info["hash"] == "abc123def456"
        assert info["author"] == "Alice"
        assert info["date"] == "2025-01-15T10:00:00+00:00"
        assert info["message"] == "feat: add startup logic"
        assert "Extended description" in info["body"]

    @patch("prism.intelligence.blame.subprocess.run")
    def test_handles_empty_output(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Empty git log output returns a dict with empty strings."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        info = tracer._get_commit_info("HEAD")
        assert info["hash"] == ""

    @patch("prism.intelligence.blame.subprocess.run")
    def test_handles_no_body(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Commits without a body still return a valid dict."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc\nAlice\n2025-01-15\nfeat: stuff\n",
        )
        info = tracer._get_commit_info("abc")
        assert info["body"] == ""


# ======================================================================
# TestCausalBlameTracer — _get_changed_files
# ======================================================================


class TestGetChangedFiles:
    """Tests for the _get_changed_files method."""

    @patch("prism.intelligence.blame.subprocess.run")
    def test_returns_file_list(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Changed files are returned as a list."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="src/main.py\nsrc/utils.py\n"
        )
        files = tracer._get_changed_files("abc123")
        assert files == ["src/main.py", "src/utils.py"]

    @patch("prism.intelligence.blame.subprocess.run")
    def test_empty_diff(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Empty diff returns empty list."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        files = tracer._get_changed_files("abc123")
        assert files == []


# ======================================================================
# TestCausalBlameTracer — _extract_changed_lines
# ======================================================================


class TestExtractChangedLines:
    """Tests for the static _extract_changed_lines method."""

    def test_extracts_additions(self) -> None:
        """Added lines are prefixed with the file name."""
        diff = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "+new_line = True\n"
        )
        lines = CausalBlameTracer._extract_changed_lines(diff)
        assert len(lines) == 1
        assert "foo.py: new_line = True" in lines[0]

    def test_extracts_removals(self) -> None:
        """Removed lines are marked with '(removed)'."""
        diff = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "-old_line = False\n"
        )
        lines = CausalBlameTracer._extract_changed_lines(diff)
        assert len(lines) == 1
        assert "(removed)" in lines[0]

    def test_ignores_headers(self) -> None:
        """'---' and '+++' header lines are not included."""
        diff = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
        )
        lines = CausalBlameTracer._extract_changed_lines(diff)
        assert lines == []

    def test_limits_to_50_lines(self) -> None:
        """Output is capped at 50 lines."""
        additions = "\n".join(f"+line_{i}" for i in range(60))
        diff = f"diff --git a/f.py b/f.py\n{additions}"
        lines = CausalBlameTracer._extract_changed_lines(diff)
        assert len(lines) <= 50

    def test_multiple_files(self) -> None:
        """Changed lines from multiple files have correct file prefixes."""
        diff = (
            "diff --git a/a.py b/a.py\n"
            "+alpha\n"
            "diff --git a/b.py b/b.py\n"
            "+beta\n"
        )
        lines = CausalBlameTracer._extract_changed_lines(diff)
        assert any("a.py" in ln for ln in lines)
        assert any("b.py" in ln for ln in lines)

    def test_empty_diff(self) -> None:
        """An empty diff returns no lines."""
        assert CausalBlameTracer._extract_changed_lines("") == []


# ======================================================================
# TestCausalBlameTracer — _build_narrative
# ======================================================================


class TestBuildNarrative:
    """Tests for the static _build_narrative method."""

    def test_includes_description(self) -> None:
        """The narrative includes the bug description."""
        narrative = CausalBlameTracer._build_narrative(
            description="Widget crashes",
            info={"hash": "abc12345", "author": "Bob", "date": "2025-01-01", "message": "fix"},
            diff="line1\nline2\nline3\n",
            files=["widget.py"],
        )
        assert "Widget crashes" in narrative

    def test_includes_commit_hash(self) -> None:
        """The narrative mentions the commit hash prefix."""
        narrative = CausalBlameTracer._build_narrative(
            description="bug",
            info={"hash": "abcdef1234567890", "author": "A", "date": "D", "message": "M"},
            diff="",
            files=[],
        )
        assert "abcdef12" in narrative

    def test_includes_author(self) -> None:
        """The narrative mentions the author."""
        narrative = CausalBlameTracer._build_narrative(
            description="bug",
            info={"hash": "abc", "author": "Charlie", "date": "D", "message": "M"},
            diff="",
            files=[],
        )
        assert "Charlie" in narrative

    def test_includes_file_count(self) -> None:
        """The narrative mentions the number of files changed."""
        narrative = CausalBlameTracer._build_narrative(
            description="bug",
            info={"hash": "", "author": "", "date": "", "message": ""},
            diff="l1\nl2\n",
            files=["a.py", "b.py", "c.py"],
        )
        assert "3 file(s)" in narrative

    def test_empty_files_no_error(self) -> None:
        """An empty file list does not cause an error."""
        narrative = CausalBlameTracer._build_narrative(
            description="bug",
            info={"hash": "", "author": "", "date": "", "message": ""},
            diff="",
            files=[],
        )
        assert "0 file(s)" in narrative


# ======================================================================
# TestCausalBlameTracer — _find_related_tests
# ======================================================================


class TestFindRelatedTests:
    """Tests for the _find_related_tests method."""

    @patch("prism.intelligence.blame.subprocess.run")
    def test_finds_matching_tests(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Test files matching 'test_<module>.py' pattern are returned."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="tests/test_main.py\n"
        )
        tests = tracer._find_related_tests(["src/main.py"])
        assert "tests/test_main.py" in tests

    @patch("prism.intelligence.blame.subprocess.run")
    def test_deduplicates(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Duplicate test paths are removed."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="tests/test_main.py\ntests/test_main.py\n"
        )
        tests = tracer._find_related_tests(["src/main.py"])
        assert tests.count("tests/test_main.py") == 1

    @patch("prism.intelligence.blame.subprocess.run")
    def test_no_tests_found(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Returns empty list when no test files match."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        tests = tracer._find_related_tests(["src/obscure.py"])
        assert tests == []

    @patch("prism.intelligence.blame.subprocess.run")
    def test_multiple_affected_files(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Tests from multiple affected files are aggregated."""
        def side_effect(*args, **kwargs):
            cmd = args[0]
            name_arg = cmd[3]  # -name arg
            if "main" in name_arg:
                return MagicMock(returncode=0, stdout="tests/test_main.py\n")
            if "utils" in name_arg:
                return MagicMock(returncode=0, stdout="tests/test_utils.py\n")
            return MagicMock(returncode=0, stdout="")

        mock_run.side_effect = side_effect
        tests = tracer._find_related_tests(["src/main.py", "src/utils.py"])
        assert "tests/test_main.py" in tests
        assert "tests/test_utils.py" in tests


# ======================================================================
# TestCausalBlameTracer — _save_report and list_reports
# ======================================================================


class TestSaveAndListReports:
    """Tests for report persistence."""

    def test_save_creates_json_file(
        self, tracer: CausalBlameTracer, reports_dir: Path
    ) -> None:
        """_save_report creates a JSON file in the reports directory."""
        report = BlameReport(
            bug_description="test",
            breaking_commit="abc12345",
            breaking_author="A",
            breaking_date="2025-01-01",
            breaking_message="msg",
            affected_files=[],
            affected_lines=[],
            causal_narrative="narrative",
            confidence=0.5,
            proposed_fix="",
            related_tests=[],
            bisect_steps=0,
            created_at="2025-03-11T00:00:00+00:00",
        )
        path = tracer._save_report(report)
        assert path.exists()
        assert path.suffix == ".json"
        data = json.loads(path.read_text())
        assert data["bug_description"] == "test"
        assert data["breaking_commit"] == "abc12345"

    def test_list_reports_sorted(
        self, tracer: CausalBlameTracer, reports_dir: Path
    ) -> None:
        """Reports are listed newest first."""
        (reports_dir / "blame_aaa_2025-01-01.json").write_text("{}")
        (reports_dir / "blame_bbb_2025-01-02.json").write_text("{}")
        reports = tracer.list_reports()
        assert len(reports) == 2
        # 'bbb' > 'aaa' alphabetically, so it comes first in reverse sort
        assert "bbb" in reports[0].name

    def test_list_reports_empty(self, tracer: CausalBlameTracer) -> None:
        """Empty reports directory returns empty list."""
        # Clear reports_dir
        for f in tracer._reports_dir.iterdir():
            f.unlink()
        assert tracer.list_reports() == []

    def test_load_report(
        self, tracer: CausalBlameTracer, reports_dir: Path
    ) -> None:
        """A saved report can be loaded back."""
        report = BlameReport(
            bug_description="test load",
            breaking_commit="xyz789",
            breaking_author="Bob",
            breaking_date="2025-02-01",
            breaking_message="fix: thing",
            affected_files=["a.py"],
            affected_lines=["a.py: x = 1"],
            causal_narrative="narrative",
            confidence=0.9,
            proposed_fix="do Y",
            related_tests=["tests/test_a.py"],
            bisect_steps=3,
            created_at="2025-03-11T12:00:00+00:00",
        )
        path = tracer._save_report(report)
        loaded = tracer.load_report(path)
        assert loaded.bug_description == "test load"
        assert loaded.breaking_commit == "xyz789"
        assert loaded.confidence == 0.9

    def test_load_report_file_not_found(
        self, tracer: CausalBlameTracer
    ) -> None:
        """Loading a nonexistent report raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            tracer.load_report(Path("/nonexistent/blame_report.json"))


# ======================================================================
# TestCausalBlameTracer — _run_bisect
# ======================================================================


class TestRunBisect:
    """Tests for the _run_bisect method."""

    @patch("prism.intelligence.blame.subprocess.run")
    def test_bisect_finds_breaking_commit(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Bisect output is parsed to find the first bad commit."""
        def side_effect(*args, **kwargs):
            cmd = args[0]
            if cmd[1] == "bisect" and cmd[2] == "run":
                return MagicMock(
                    returncode=0,
                    stdout="abc123def456 is the first bad commit\nBisect took 5 steps\n",
                )
            return MagicMock(returncode=0, stdout="")

        mock_run.side_effect = side_effect
        result = tracer._run_bisect("good_sha", "bad_sha", "pytest tests/")
        assert result.breaking_commit == "abc123def456"
        assert result.total_steps == 5
        assert result.good_commit == "good_sha"
        assert result.bad_commit == "bad_sha"

    @patch("prism.intelligence.blame.subprocess.run")
    def test_bisect_always_resets(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """git bisect reset is always called, even on failure."""
        call_log: list[list[str]] = []

        def side_effect(*args, **kwargs):
            cmd = args[0]
            call_log.append(list(cmd))
            if cmd[1] == "bisect" and cmd[2] == "run":
                raise subprocess.TimeoutExpired(cmd="git bisect run", timeout=300)
            return MagicMock(returncode=0, stdout="")

        mock_run.side_effect = side_effect
        with pytest.raises(subprocess.TimeoutExpired):
            tracer._run_bisect("good", "bad", "false")

        # Verify reset was called
        reset_calls = [c for c in call_log if c[1:] == ["bisect", "reset"]]
        assert len(reset_calls) == 1

    @patch("prism.intelligence.blame.subprocess.run")
    def test_bisect_fallback_to_bad_commit(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """If bisect output doesn't contain a clear result, falls back to bad commit."""
        mock_run.return_value = MagicMock(returncode=0, stdout="no useful output\n")
        result = tracer._run_bisect("good", "HEAD", "pytest")
        assert result.breaking_commit == "HEAD"


# ======================================================================
# TestCausalBlameTracer — full trace
# ======================================================================


class TestFullTrace:
    """Tests for the full trace method with all git commands mocked."""

    @patch("prism.intelligence.blame.subprocess.run")
    def test_trace_without_bisect(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Trace without test_command skips bisect."""
        def side_effect(*args, **kwargs):
            cmd = args[0]
            if cmd[0] == "git" and cmd[1] == "grep":
                return MagicMock(returncode=0, stdout="src/main.py\n")
            if cmd[0] == "git" and cmd[1] == "log":
                return MagicMock(
                    returncode=0,
                    stdout="abc123\nAlice\n2025-01-15\nfeat: thing\n",
                )
            if cmd[0] == "git" and cmd[1] == "diff" and "--name-only" in cmd:
                return MagicMock(returncode=0, stdout="src/main.py\n")
            if cmd[0] == "git" and cmd[1] == "diff":
                return MagicMock(
                    returncode=0,
                    stdout="diff --git a/src/main.py b/src/main.py\n+new line\n",
                )
            if cmd[0] == "find":
                return MagicMock(returncode=0, stdout="tests/test_main.py\n")
            return MagicMock(returncode=0, stdout="")

        mock_run.side_effect = side_effect
        report = tracer.trace(bug_description="Widget crashes on startup")
        assert report.bug_description == "Widget crashes on startup"
        assert report.breaking_commit == "HEAD"
        assert report.confidence == 0.4  # No bisect = lower confidence
        assert report.bisect_steps == 0

    @patch("prism.intelligence.blame.subprocess.run")
    def test_trace_with_bisect(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Trace with test_command runs bisect and has higher confidence."""
        def side_effect(*args, **kwargs):
            cmd = args[0]
            if cmd[0] == "git" and cmd[1] == "grep":
                return MagicMock(returncode=0, stdout="src/main.py\n")
            if cmd[0] == "git" and cmd[1] == "bisect" and cmd[2] == "run":
                return MagicMock(
                    returncode=0,
                    stdout="deadbeef12345678 is the first bad commit\n3 steps\n",
                )
            if cmd[0] == "git" and cmd[1] == "log":
                return MagicMock(
                    returncode=0,
                    stdout="deadbeef\nBob\n2025-02-01\nfix: break\n",
                )
            if cmd[0] == "git" and cmd[1] == "diff" and "--name-only" in cmd:
                return MagicMock(returncode=0, stdout="src/main.py\n")
            if cmd[0] == "git" and cmd[1] == "diff":
                return MagicMock(returncode=0, stdout="diff ...\n+bad\n")
            if cmd[0] == "find":
                return MagicMock(returncode=0, stdout="")
            return MagicMock(returncode=0, stdout="")

        mock_run.side_effect = side_effect
        report = tracer.trace(
            bug_description="test failure",
            test_command="pytest tests/",
            good_commit="good_sha",
            bad_commit="bad_sha",
        )
        assert report.breaking_commit == "deadbeef12345678"
        assert report.confidence == 0.7  # Bisect = higher confidence
        assert report.bisect_steps == 3

    @patch("prism.intelligence.blame.subprocess.run")
    def test_trace_saves_report(
        self,
        mock_run: MagicMock,
        tracer: CausalBlameTracer,
        reports_dir: Path,
    ) -> None:
        """The trace method saves a report to disk."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        tracer.trace(bug_description="saves report test")
        reports = list(reports_dir.glob("blame_*.json"))
        assert len(reports) == 1

    @patch("prism.intelligence.blame.subprocess.run")
    def test_trace_report_has_created_at(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """The report has a non-empty created_at timestamp."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        report = tracer.trace(bug_description="timestamp test")
        assert report.created_at
        assert "T" in report.created_at


# ======================================================================
# TestCausalBlameTracer — _git helper
# ======================================================================


class TestGitHelper:
    """Tests for the _git helper method."""

    @patch("prism.intelligence.blame.subprocess.run")
    def test_passes_args_to_git(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Arguments are forwarded to the git command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="output")
        result = tracer._git("status", "--short")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd == ["git", "status", "--short"]
        assert result == "output"

    @patch("prism.intelligence.blame.subprocess.run")
    def test_uses_project_root_as_cwd(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Git commands run from the project root directory."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        tracer._git("log")
        kwargs = mock_run.call_args[1]
        assert kwargs["cwd"] == tracer._root

    @patch("prism.intelligence.blame.subprocess.run")
    def test_returns_empty_on_failure(
        self, mock_run: MagicMock, tracer: CausalBlameTracer
    ) -> None:
        """Non-zero return code still returns stdout (may be empty)."""
        mock_run.return_value = MagicMock(returncode=128, stdout="")
        result = tracer._git("log", "-1")
        assert result == ""
