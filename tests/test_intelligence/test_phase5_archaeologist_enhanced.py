"""Tests for Phase 5 enhanced temporal code archaeologist — Item 27.

Covers:
- CommitInfo and ArchaeologyReport dataclasses
- _git_blame parsing
- _git_log parsing
- _analyze_co_evolution
- _calculate_stability
- _identify_primary_author
- _build_author_distribution
- _generate_narrative
- _identify_risks
- generate_report_text format
- save_report and list_reports
- investigate() with mock git runner
- Edge cases: no history, single commit, empty input
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.intelligence.archaeologist import (
    ArchaeologyReport,
    CommitInfo,
    _analyze_co_evolution,
    _build_author_distribution,
    _calculate_stability,
    _generate_narrative,
    _git_blame,
    _git_log,
    _identify_primary_author,
    _identify_risks,
    _is_significant_commit,
    _parse_investigation_target,
    generate_report_text,
    investigate,
    list_reports,
    save_report,
)

if TYPE_CHECKING:
    from pathlib import Path

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def tmp_reports_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for report storage."""
    d = tmp_path / "archaeology"
    d.mkdir()
    return d


@pytest.fixture
def sample_commits() -> list[CommitInfo]:
    """Create a list of sample CommitInfo objects (newest first)."""
    return [
        CommitInfo(
            hash="abc123def456789012345678901234567890abcd",
            author="Alice",
            email="alice@example.com",
            date="2025-06-15T10:00:00+00:00",
            subject="fix: handle edge case in parser",
            body="",
            files_changed=["src/parser.py", "tests/test_parser.py"],
            is_significant=True,
        ),
        CommitInfo(
            hash="def456abc789012345678901234567890abcdef12",
            author="Bob",
            email="bob@example.com",
            date="2025-06-10T10:00:00+00:00",
            subject="feat: add new parsing mode",
            body="Adds streaming parser support.",
            files_changed=["src/parser.py", "src/stream.py"],
            is_significant=True,
        ),
        CommitInfo(
            hash="789012def456abc345678901234567890abcdef34",
            author="Alice",
            email="alice@example.com",
            date="2025-05-01T10:00:00+00:00",
            subject="feat: initial parser implementation",
            body="",
            files_changed=["src/parser.py"],
            is_significant=True,
        ),
    ]


@pytest.fixture
def sample_report(sample_commits: list[CommitInfo]) -> ArchaeologyReport:
    """Create a sample ArchaeologyReport."""
    return ArchaeologyReport(
        target="src/parser.py:42",
        timeline=sample_commits,
        primary_author="Alice",
        author_distribution={"Alice": 2, "Bob": 1},
        co_evolution=[("tests/test_parser.py", 0.67), ("src/stream.py", 0.33)],
        stability_score=0.9,
        narrative="## History of src/parser.py\n\nCreated by Alice...",
        risks=["Low risk: stable code with good maintainer coverage."],
        created_at="2025-06-15T12:00:00+00:00",
    )


def _make_mock_git_runner(responses: dict[str, str] | None = None):
    """Create a mock git runner that returns canned responses.

    Args:
        responses: A dict mapping partial command strings to output.
            For example, {"blame": "blame output", "log": "log output"}.
    """
    default_responses = responses or {}

    def runner(args: tuple[str, ...], cwd: Path) -> str:
        args_str = " ".join(args)
        for key, value in default_responses.items():
            if key in args_str:
                return value
        return ""

    return runner


# ======================================================================
# TestCommitInfo
# ======================================================================


class TestCommitInfo:
    """Tests for CommitInfo dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """CommitInfo can be created with all fields."""
        ci = CommitInfo(
            hash="abc123",
            author="Alice",
            email="alice@example.com",
            date="2025-06-15",
            subject="feat: add feature",
            body="Detailed body",
            files_changed=["a.py", "b.py"],
            is_significant=True,
        )
        assert ci.hash == "abc123"
        assert ci.author == "Alice"
        assert ci.email == "alice@example.com"
        assert ci.subject == "feat: add feature"
        assert len(ci.files_changed) == 2
        assert ci.is_significant is True

    def test_default_body_empty(self) -> None:
        """Body defaults to empty string."""
        ci = CommitInfo(
            hash="x", author="A", email="a@b.c",
            date="2025-01-01", subject="test",
        )
        assert ci.body == ""

    def test_default_files_changed_empty(self) -> None:
        """files_changed defaults to empty list."""
        ci = CommitInfo(
            hash="x", author="A", email="a@b.c",
            date="2025-01-01", subject="test",
        )
        assert ci.files_changed == []

    def test_default_is_significant_true(self) -> None:
        """is_significant defaults to True."""
        ci = CommitInfo(
            hash="x", author="A", email="a@b.c",
            date="2025-01-01", subject="test",
        )
        assert ci.is_significant is True


# ======================================================================
# TestArchaeologyReport
# ======================================================================


class TestArchaeologyReport:
    """Tests for ArchaeologyReport dataclass."""

    def test_default_values(self) -> None:
        """Default values are empty/zero."""
        report = ArchaeologyReport(target="test.py")
        assert report.target == "test.py"
        assert report.timeline == []
        assert report.primary_author == ""
        assert report.author_distribution == {}
        assert report.co_evolution == []
        assert report.stability_score == 1.0
        assert report.narrative == ""
        assert report.risks == []
        assert report.created_at == ""

    def test_full_report(self, sample_report: ArchaeologyReport) -> None:
        """Full report has all fields populated."""
        assert sample_report.target == "src/parser.py:42"
        assert len(sample_report.timeline) == 3
        assert sample_report.primary_author == "Alice"
        assert sample_report.author_distribution["Alice"] == 2
        assert len(sample_report.co_evolution) == 2
        assert sample_report.stability_score == 0.9


# ======================================================================
# TestGitBlame
# ======================================================================


class TestGitBlame:
    """Tests for _git_blame parsing."""

    def test_parse_porcelain_output(self, tmp_path: Path) -> None:
        """Porcelain blame output is parsed correctly."""
        blame_output = (
            "abc123def456789012345678901234567890abcd 1 1 1\n"
            "author Alice\n"
            "author-mail <alice@example.com>\n"
            "author-time 1718451600\n"
            "summary feat: initial commit\n"
            "\tdef hello():\n"
            "def456abc789012345678901234567890abcdef12 2 2 1\n"
            "author Bob\n"
            "author-mail <bob@example.com>\n"
            "author-time 1718538000\n"
            "summary fix: typo\n"
            "\t    return 42\n"
        )
        runner = _make_mock_git_runner({"blame": blame_output})
        entries = _git_blame("test.py", None, None, runner, tmp_path)
        assert len(entries) == 2
        assert entries[0]["author"] == "Alice"
        assert entries[0]["content"] == "def hello():"
        assert entries[1]["author"] == "Bob"

    def test_blame_with_line_range(self, tmp_path: Path) -> None:
        """Blame with line range passes -L flag."""
        called_args: list[tuple[str, ...]] = []

        def capture_runner(args: tuple[str, ...], cwd: Path) -> str:
            called_args.append(args)
            return ""

        _git_blame("test.py", 10, 30, capture_runner, tmp_path)
        assert len(called_args) == 1
        args_str = " ".join(called_args[0])
        assert "-L" in args_str
        assert "10,30" in args_str

    def test_blame_without_line_range(self, tmp_path: Path) -> None:
        """Blame without line range does not pass -L flag."""
        called_args: list[tuple[str, ...]] = []

        def capture_runner(args: tuple[str, ...], cwd: Path) -> str:
            called_args.append(args)
            return ""

        _git_blame("test.py", None, None, capture_runner, tmp_path)
        args_str = " ".join(called_args[0])
        assert "-L" not in args_str

    def test_blame_empty_output(self, tmp_path: Path) -> None:
        """Empty blame output returns empty list."""
        runner = _make_mock_git_runner({"blame": ""})
        entries = _git_blame("test.py", None, None, runner, tmp_path)
        assert entries == []


# ======================================================================
# TestGitLog
# ======================================================================


class TestGitLog:
    """Tests for _git_log parsing."""

    def test_parse_log_output(self, tmp_path: Path) -> None:
        """Git log output is parsed into CommitInfo list."""
        log_output = (
            "abc123|Alice|alice@example.com|2025-06-15T10:00:00+00:00|feat: add feature|\n"
            "5\t2\tsrc/parser.py\n"
            "3\t1\ttests/test_parser.py\n"
            "\n"
            "def456|Bob|bob@example.com|2025-06-10T10:00:00+00:00|fix: bug|\n"
            "2\t1\tsrc/parser.py\n"
        )
        runner = _make_mock_git_runner({"log": log_output})
        commits = _git_log("src/parser.py", runner, tmp_path)
        assert len(commits) == 2
        assert commits[0].author == "Alice"
        assert commits[0].subject == "feat: add feature"
        assert "src/parser.py" in commits[0].files_changed
        assert "tests/test_parser.py" in commits[0].files_changed
        assert commits[1].author == "Bob"

    def test_empty_log(self, tmp_path: Path) -> None:
        """Empty log output returns empty list."""
        runner = _make_mock_git_runner({"log": ""})
        commits = _git_log("test.py", runner, tmp_path)
        assert commits == []

    def test_log_single_commit(self, tmp_path: Path) -> None:
        """Single commit is parsed correctly."""
        log_output = "aaa|Dev|dev@x.com|2025-01-01T00:00:00|initial|\n"
        runner = _make_mock_git_runner({"log": log_output})
        commits = _git_log("test.py", runner, tmp_path)
        assert len(commits) == 1
        assert commits[0].author == "Dev"


# ======================================================================
# TestIsSignificantCommit
# ======================================================================


class TestIsSignificantCommit:
    """Tests for _is_significant_commit."""

    def test_normal_commit_significant(self) -> None:
        """Normal feature commit is significant."""
        assert _is_significant_commit("feat: add parser", 50, 10) is True

    def test_style_small_change_insignificant(self) -> None:
        """Style commit with small change is insignificant."""
        assert _is_significant_commit("style: fix whitespace", 2, 2) is False

    def test_style_large_change_significant(self) -> None:
        """Style commit with large change is still significant."""
        assert _is_significant_commit("style: reformat entire file", 100, 100) is True

    def test_fix_typo_small_insignificant(self) -> None:
        """Fix typo with small change is insignificant."""
        assert _is_significant_commit("fix typo in docstring", 1, 1) is False

    def test_bugfix_commit_significant(self) -> None:
        """Bugfix commit is significant."""
        assert _is_significant_commit("fix: null pointer in parser", 10, 5) is True


# ======================================================================
# TestAnalyzeCoEvolution
# ======================================================================


class TestAnalyzeCoEvolution:
    """Tests for _analyze_co_evolution."""

    def test_co_evolution_found(
        self, sample_commits: list[CommitInfo], tmp_path: Path,
    ) -> None:
        """Co-evolving files are identified."""
        runner = _make_mock_git_runner()
        result = _analyze_co_evolution(
            "src/parser.py", sample_commits, runner, tmp_path,
        )
        # tests/test_parser.py appears in 1/3 commits with src/parser.py
        # src/stream.py appears in 1/3
        # Only files with > 20% co-change rate are included
        file_names = [f for f, _ in result]
        assert isinstance(result, list)
        # All 3 commits change src/parser.py,
        # tests/test_parser.py in 1/3 = 33%
        # src/stream.py in 1/3 = 33%
        assert "tests/test_parser.py" in file_names
        assert "src/stream.py" in file_names

    def test_co_evolution_empty_commits(self, tmp_path: Path) -> None:
        """Empty commit list returns empty co-evolution."""
        runner = _make_mock_git_runner()
        result = _analyze_co_evolution("test.py", [], runner, tmp_path)
        assert result == []

    def test_co_evolution_excludes_target(
        self, sample_commits: list[CommitInfo], tmp_path: Path,
    ) -> None:
        """Target file is excluded from co-evolution results."""
        runner = _make_mock_git_runner()
        result = _analyze_co_evolution(
            "src/parser.py", sample_commits, runner, tmp_path,
        )
        file_names = [f for f, _ in result]
        assert "src/parser.py" not in file_names

    def test_co_evolution_sorted_descending(self, tmp_path: Path) -> None:
        """Results are sorted by co-change percentage descending."""
        commits = [
            CommitInfo(
                hash="a", author="A", email="a@b.c", date="2025-01-01",
                subject="c1",
                files_changed=["target.py", "always.py", "sometimes.py"],
            ),
            CommitInfo(
                hash="b", author="A", email="a@b.c", date="2025-01-02",
                subject="c2",
                files_changed=["target.py", "always.py"],
            ),
            CommitInfo(
                hash="c", author="A", email="a@b.c", date="2025-01-03",
                subject="c3",
                files_changed=["target.py", "always.py"],
            ),
        ]
        runner = _make_mock_git_runner()
        result = _analyze_co_evolution("target.py", commits, runner, tmp_path)
        # always.py: 3/3 = 100%, sometimes.py: 1/3 = 33%
        assert result[0][0] == "always.py"
        assert result[0][1] >= result[-1][1]


# ======================================================================
# TestCalculateStability
# ======================================================================


class TestCalculateStability:
    """Tests for _calculate_stability."""

    def test_single_commit_stable(self) -> None:
        """Single commit yields 1.0 stability."""
        commits = [CommitInfo(
            hash="a", author="A", email="a@b.c",
            date="2025-01-01", subject="init",
        )]
        assert _calculate_stability(commits) == 1.0

    def test_empty_commits_stable(self) -> None:
        """Empty list yields 1.0 stability."""
        assert _calculate_stability([]) == 1.0

    def test_few_commits_high_stability(self) -> None:
        """3 commits yields 0.9 stability."""
        commits = [
            CommitInfo(hash=f"h{i}", author="A", email="a@b.c",
                       date=f"2025-01-{i+1:02d}", subject=f"c{i}")
            for i in range(3)
        ]
        assert _calculate_stability(commits) == 0.9

    def test_many_commits_low_stability(self) -> None:
        """51+ commits yields 0.2 stability."""
        commits = [
            CommitInfo(hash=f"h{i}", author="A", email="a@b.c",
                       date="2025-01-01", subject=f"c{i}")
            for i in range(51)
        ]
        assert _calculate_stability(commits) == 0.2

    def test_medium_commits_stability(self) -> None:
        """6-10 commits yields 0.8 stability."""
        commits = [
            CommitInfo(hash=f"h{i}", author="A", email="a@b.c",
                       date="2025-01-01", subject=f"c{i}")
            for i in range(8)
        ]
        assert _calculate_stability(commits) == 0.8


# ======================================================================
# TestIdentifyPrimaryAuthor
# ======================================================================


class TestIdentifyPrimaryAuthor:
    """Tests for _identify_primary_author."""

    def test_identifies_most_frequent(self, sample_commits: list[CommitInfo]) -> None:
        """Primary author is the one with most commits."""
        primary = _identify_primary_author(sample_commits)
        assert primary == "Alice"  # Alice has 2 commits, Bob has 1

    def test_empty_commits_returns_unknown(self) -> None:
        """Empty commit list returns 'unknown'."""
        assert _identify_primary_author([]) == "unknown"

    def test_single_author(self) -> None:
        """Single author is identified as primary."""
        commits = [
            CommitInfo(hash="a", author="Solo", email="s@x.com",
                       date="2025-01-01", subject="init"),
        ]
        assert _identify_primary_author(commits) == "Solo"


# ======================================================================
# TestBuildAuthorDistribution
# ======================================================================


class TestBuildAuthorDistribution:
    """Tests for _build_author_distribution."""

    def test_correct_counts(self, sample_commits: list[CommitInfo]) -> None:
        """Author counts are correct."""
        dist = _build_author_distribution(sample_commits)
        assert dist["Alice"] == 2
        assert dist["Bob"] == 1

    def test_empty_commits(self) -> None:
        """Empty commits yields empty distribution."""
        assert _build_author_distribution([]) == {}


# ======================================================================
# TestGenerateNarrative
# ======================================================================


class TestGenerateNarrative:
    """Tests for _generate_narrative."""

    def test_narrative_with_commits(self, sample_commits: list[CommitInfo]) -> None:
        """Narrative includes creation info and categories."""
        narrative = _generate_narrative(
            "src/parser.py", [], sample_commits, [],
        )
        assert "History of" in narrative
        assert "Alice" in narrative
        assert "Latest change" in narrative

    def test_narrative_empty_commits(self) -> None:
        """Empty commits yields 'No git history' message."""
        narrative = _generate_narrative("test.py", [], [], [])
        assert "No git history" in narrative

    def test_narrative_includes_co_evolution(
        self, sample_commits: list[CommitInfo],
    ) -> None:
        """Narrative includes co-evolving files section."""
        co_evolution = [("tests/test_parser.py", 0.67)]
        narrative = _generate_narrative(
            "src/parser.py", [], sample_commits, co_evolution,
        )
        assert "Co-evolving files" in narrative
        assert "tests/test_parser.py" in narrative

    def test_narrative_includes_blame_summary(
        self, sample_commits: list[CommitInfo],
    ) -> None:
        """Narrative includes blame attribution summary."""
        blame_data = [
            {"author": "Alice", "content": "def foo():"},
            {"author": "Bob", "content": "return 1"},
        ]
        narrative = _generate_narrative(
            "src/parser.py", blame_data, sample_commits, [],
        )
        assert "author" in narrative.lower()


# ======================================================================
# TestIdentifyRisks
# ======================================================================


class TestIdentifyRisks:
    """Tests for _identify_risks."""

    def test_low_risk_stable(self) -> None:
        """Stable code with multiple authors is low risk."""
        commits = [
            CommitInfo(hash="a", author="Alice", email="a@b.c",
                       date="2025-01-01", subject="feat: add"),
            CommitInfo(hash="b", author="Bob", email="b@b.c",
                       date="2025-01-02", subject="feat: extend"),
        ]
        risks = _identify_risks(commits, 0.9, {"Alice": 1, "Bob": 1})
        assert any("Low risk" in r for r in risks)

    def test_volatile_code_flagged(self) -> None:
        """Volatile code (stability < 0.5) is flagged."""
        commits = [
            CommitInfo(hash=f"h{i}", author="A", email="a@b.c",
                       date="2025-01-01", subject="mod")
            for i in range(25)
        ]
        risks = _identify_risks(commits, 0.3, {"A": 25})
        assert any("volatile" in r.lower() for r in risks)

    def test_single_maintainer_flagged(self) -> None:
        """Single maintainer is flagged as bus factor = 1."""
        commits = [
            CommitInfo(hash="a", author="Solo", email="s@b.c",
                       date="2025-01-01", subject="feat: x"),
        ]
        risks = _identify_risks(commits, 0.9, {"Solo": 1})
        assert any("bus factor" in r.lower() for r in risks)

    def test_high_bugfix_ratio_flagged(self) -> None:
        """High bugfix ratio (>30%) is flagged."""
        commits = [
            CommitInfo(hash=f"h{i}", author="A", email="a@b.c",
                       date="2025-01-01",
                       subject="fix: bug" if i < 4 else "feat: add")
            for i in range(10)
        ]
        risks = _identify_risks(commits, 0.6, {"A": 10})
        assert any("bug rate" in r.lower() for r in risks)


# ======================================================================
# TestGenerateReportText
# ======================================================================


class TestGenerateReportText:
    """Tests for generate_report_text output format."""

    def test_report_contains_header(self, sample_report: ArchaeologyReport) -> None:
        """Report contains CODE ARCHAEOLOGY REPORT header."""
        text = generate_report_text(sample_report)
        assert "CODE ARCHAEOLOGY REPORT" in text

    def test_report_contains_target(self, sample_report: ArchaeologyReport) -> None:
        """Report contains the target."""
        text = generate_report_text(sample_report)
        assert "src/parser.py:42" in text

    def test_report_contains_stability(self, sample_report: ArchaeologyReport) -> None:
        """Report contains stability score."""
        text = generate_report_text(sample_report)
        assert "Stability:" in text
        assert "90%" in text

    def test_report_contains_timeline(self, sample_report: ArchaeologyReport) -> None:
        """Report contains timeline section."""
        text = generate_report_text(sample_report)
        assert "Timeline" in text
        assert "Alice" in text

    def test_report_contains_author_distribution(
        self, sample_report: ArchaeologyReport,
    ) -> None:
        """Report contains author distribution section."""
        text = generate_report_text(sample_report)
        assert "Author Distribution" in text
        assert "Alice" in text
        assert "Bob" in text

    def test_report_contains_co_evolution(
        self, sample_report: ArchaeologyReport,
    ) -> None:
        """Report contains co-evolving files section."""
        text = generate_report_text(sample_report)
        assert "Co-evolving Files" in text
        assert "tests/test_parser.py" in text

    def test_report_contains_risks(self, sample_report: ArchaeologyReport) -> None:
        """Report contains risks section."""
        text = generate_report_text(sample_report)
        assert "Risks" in text

    def test_empty_report(self) -> None:
        """Empty report is handled gracefully."""
        report = ArchaeologyReport(target="empty.py")
        text = generate_report_text(report)
        assert "CODE ARCHAEOLOGY REPORT" in text
        assert "empty.py" in text


# ======================================================================
# TestSaveAndListReports
# ======================================================================


class TestSaveAndListReports:
    """Tests for save_report and list_reports."""

    def test_save_creates_file(
        self, sample_report: ArchaeologyReport, tmp_reports_dir: Path,
    ) -> None:
        """Saving a report creates a .md file."""
        path = save_report(sample_report, save_dir=tmp_reports_dir)
        assert path.exists()
        assert path.suffix == ".md"

    def test_save_content(
        self, sample_report: ArchaeologyReport, tmp_reports_dir: Path,
    ) -> None:
        """Saved file contains report text."""
        path = save_report(sample_report, save_dir=tmp_reports_dir)
        content = path.read_text()
        assert "src/parser.py" in content

    def test_list_reports_returns_saved(
        self, sample_report: ArchaeologyReport, tmp_reports_dir: Path,
    ) -> None:
        """list_reports returns saved files."""
        save_report(sample_report, save_dir=tmp_reports_dir)
        reports = list_reports(tmp_reports_dir)
        assert len(reports) >= 1

    def test_list_reports_empty_dir(self, tmp_path: Path) -> None:
        """list_reports returns empty list for empty directory."""
        empty = tmp_path / "empty_arch"
        empty.mkdir()
        assert list_reports(empty) == []

    def test_list_reports_nonexistent(self, tmp_path: Path) -> None:
        """list_reports returns empty list for nonexistent directory."""
        assert list_reports(tmp_path / "nonexistent") == []

    def test_save_creates_dir(
        self, sample_report: ArchaeologyReport, tmp_path: Path,
    ) -> None:
        """save_report creates directory if it doesn't exist."""
        target = tmp_path / "new" / "nested"
        path = save_report(sample_report, save_dir=target)
        assert path.exists()


# ======================================================================
# TestInvestigate
# ======================================================================


class TestInvestigate:
    """Tests for the investigate() function with mock git runner."""

    def test_investigate_file_target(self, tmp_path: Path) -> None:
        """investigate() works with a file path target."""
        blame_output = (
            "abc123def456789012345678901234567890abcd 1 1 1\n"
            "author Alice\n"
            "author-time 1718451600\n"
            "summary feat: init\n"
            "\tcode here\n"
        )
        log_output = (
            "abc123|Alice|alice@x.com|2025-06-15T10:00:00+00:00|feat: init|\n"
            "10\t0\tsrc/main.py\n"
        )
        runner = _make_mock_git_runner({
            "blame": blame_output,
            "log": log_output,
        })
        report = investigate(
            "src/main.py",
            project_root=tmp_path,
            run_git=runner,
        )
        assert report.target == "src/main.py"
        assert report.primary_author == "Alice"
        assert len(report.timeline) == 1

    def test_investigate_file_with_line(self, tmp_path: Path) -> None:
        """investigate() works with file:line target."""
        runner = _make_mock_git_runner({
            "blame": (
                "abc123def456789012345678901234567890abcd 42 42 1\n"
                "author Bob\n"
                "summary fix: typo\n"
                "\tcode\n"
            ),
            "log": (
                "abc123|Bob|bob@x.com|2025-01-01T00:00:00+00:00|fix: typo|\n"
            ),
        })
        report = investigate("main.py:42", project_root=tmp_path, run_git=runner)
        assert report.target == "main.py:42"

    def test_investigate_function_name(self, tmp_path: Path) -> None:
        """investigate() works with a function name target."""
        runner = _make_mock_git_runner({
            "grep": "src/utils.py:10:def my_func():\n",
            "blame": (
                "abc123def456789012345678901234567890abcd 10 10 1\n"
                "author Dev\n"
                "summary feat: add my_func\n"
                "\tdef my_func():\n"
            ),
            "log": (
                "abc123|Dev|dev@x.com|2025-01-01T00:00:00+00:00|feat: add my_func|\n"
            ),
        })
        report = investigate("my_func", project_root=tmp_path, run_git=runner)
        assert report.target == "my_func"

    def test_investigate_not_found_raises(self, tmp_path: Path) -> None:
        """investigate() raises ValueError when target not found."""
        runner = _make_mock_git_runner({})  # Everything returns ""
        with pytest.raises(ValueError, match="Could not find"):
            investigate("nonexistent_func", project_root=tmp_path, run_git=runner)

    def test_investigate_saves_report(self, tmp_path: Path) -> None:
        """investigate() auto-saves a report."""
        runner = _make_mock_git_runner({
            "blame": "",
            "log": (
                "abc|A|a@x.com|2025-01-01T00:00:00|init|\n"
            ),
        })
        report = investigate("test.py", project_root=tmp_path, run_git=runner)
        # Check the default save dir was used
        assert report.created_at != ""

    def test_investigate_stability_score(self, tmp_path: Path) -> None:
        """investigate() correctly calculates stability."""
        # Create log with many commits for low stability
        log_lines = []
        for i in range(25):
            log_lines.append(
                f"hash{i:04d}|Dev|d@x.com|2025-01-{(i%28)+1:02d}T00:00:00|commit {i}|\n"
            )
        runner = _make_mock_git_runner({
            "blame": "",
            "log": "".join(log_lines),
        })
        report = investigate("busy.py", project_root=tmp_path, run_git=runner)
        assert report.stability_score < 0.5  # 25 commits = 0.4


# ======================================================================
# TestParseInvestigationTarget
# ======================================================================


class TestParseInvestigationTarget:
    """Tests for _parse_investigation_target."""

    def test_file_with_line(self, tmp_path: Path) -> None:
        """Parses file:line correctly."""
        runner = _make_mock_git_runner()
        file_path, start, end = _parse_investigation_target(
            "main.py:42", runner, tmp_path,
        )
        assert file_path == "main.py"
        assert start == 42
        assert end == 62  # 42 + 20

    def test_file_without_line(self, tmp_path: Path) -> None:
        """Parses file path without line number."""
        runner = _make_mock_git_runner()
        file_path, start, end = _parse_investigation_target(
            "src/parser.py", runner, tmp_path,
        )
        assert file_path == "src/parser.py"
        assert start is None
        assert end is None

    def test_function_name(self, tmp_path: Path) -> None:
        """Function name triggers git grep search."""
        runner = _make_mock_git_runner({
            "grep": "src/utils.py:15:def my_func():\n",
        })
        file_path, start, _end = _parse_investigation_target(
            "my_func", runner, tmp_path,
        )
        assert file_path == "src/utils.py"
        assert start == 15

    def test_function_not_found(self, tmp_path: Path) -> None:
        """Unknown function returns empty file path."""
        runner = _make_mock_git_runner({})
        file_path, _start, _end = _parse_investigation_target(
            "nonexistent", runner, tmp_path,
        )
        assert file_path == ""
