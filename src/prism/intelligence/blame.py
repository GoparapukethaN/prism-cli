"""Causal Blame Tracer — automated git bisect with causal analysis.

Given a bug description (and optionally a test command), the tracer:

1. Searches the codebase for files related to the bug.
2. Runs ``git bisect`` (if a test command and good commit are provided) to
   pinpoint the breaking commit.
3. Analyses the breaking commit's diff, author, and message.
4. Builds a causal narrative explaining *why* the regression happened.
5. Identifies related test files.
6. Persists the report as JSON under ``~/.prism/blame_reports/``.

Slash-command hooks:
    /blame <description>             — run a full trace
    /blame --test "pytest ..." --good abc123 <description>
    /blame list                      — list saved reports
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Default timeout (seconds) for individual git sub-commands.
_GIT_TIMEOUT: int = 30

# Timeout for git bisect run which may invoke a test suite multiple times.
_BISECT_TIMEOUT: int = 300


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class BlameReport:
    """The output of a blame trace.

    Attributes:
        bug_description: User-provided description of the regression.
        breaking_commit: SHA of the commit that introduced the bug.
        breaking_author: Author of the breaking commit.
        breaking_date: ISO-8601 date of the breaking commit.
        breaking_message: Commit subject line.
        affected_files: Files changed in the breaking commit.
        affected_lines: Individual added/removed lines from the diff.
        causal_narrative: Human-readable explanation of the regression.
        confidence: Estimated confidence (higher when bisect was used).
        proposed_fix: Placeholder for an LLM-generated fix suggestion.
        related_tests: Test files that may cover the affected code.
        bisect_steps: Number of bisect steps taken (0 if bisect was skipped).
        created_at: ISO-8601 timestamp when the report was generated.
    """

    bug_description: str
    breaking_commit: str
    breaking_author: str
    breaking_date: str
    breaking_message: str
    affected_files: list[str]
    affected_lines: list[str]
    causal_narrative: str
    confidence: float
    proposed_fix: str
    related_tests: list[str]
    bisect_steps: int
    created_at: str


@dataclass
class BisectResult:
    """Result of an automated ``git bisect run``.

    Attributes:
        breaking_commit: SHA identified as the first bad commit.
        total_steps: Number of bisect iterations.
        good_commit: The known-good commit supplied by the user.
        bad_commit: The known-bad commit supplied by the user.
    """

    breaking_commit: str
    total_steps: int
    good_commit: str
    bad_commit: str


# ======================================================================
# Main tracer
# ======================================================================


class CausalBlameTracer:
    """Automated git bisect with causal analysis for regression hunting.

    Args:
        project_root: Path to the git repository root.
        reports_dir: Directory to persist JSON reports.  Defaults to
            ``~/.prism/blame_reports/``.
    """

    def __init__(
        self,
        project_root: Path,
        reports_dir: Path | None = None,
    ) -> None:
        self._root = project_root.resolve()
        self._reports_dir = (
            reports_dir or Path.home() / ".prism" / "blame_reports"
        )
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trace(
        self,
        bug_description: str,
        test_command: str | None = None,
        good_commit: str | None = None,
        bad_commit: str = "HEAD",
    ) -> BlameReport:
        """Run the full blame trace: search -> bisect -> analyse -> report.

        Args:
            bug_description: Natural-language description of the bug.
            test_command: Shell command that exits 0 on *good* and non-zero
                on *bad*.  If ``None``, bisect is skipped.
            good_commit: A known-good commit SHA (required with *test_command*).
            bad_commit: The known-bad commit (default ``"HEAD"``).

        Returns:
            A :class:`BlameReport` with all findings.
        """
        logger.info(
            "blame_trace_start",
            description=bug_description[:80],
            has_test=test_command is not None,
        )

        # Phase 1: Find relevant code (results used for future LLM context)
        self._find_relevant_code(bug_description)

        # Phase 2: Bisect if a test command is available
        bisect_result: BisectResult | None = None
        if test_command and good_commit:
            bisect_result = self._run_bisect(
                good_commit, bad_commit, test_command
            )

        breaking_commit = (
            bisect_result.breaking_commit if bisect_result else bad_commit
        )

        # Phase 3: Analyse the breaking commit
        commit_info = self._get_commit_info(breaking_commit)
        diff = self._get_commit_diff(breaking_commit)
        affected_files = self._get_changed_files(breaking_commit)

        # Phase 4: Build the report
        narrative = self._build_narrative(
            bug_description, commit_info, diff, affected_files
        )
        related_tests = self._find_related_tests(affected_files)

        report = BlameReport(
            bug_description=bug_description,
            breaking_commit=breaking_commit,
            breaking_author=commit_info.get("author", "unknown"),
            breaking_date=commit_info.get("date", ""),
            breaking_message=commit_info.get("message", ""),
            affected_files=affected_files,
            affected_lines=self._extract_changed_lines(diff),
            causal_narrative=narrative,
            confidence=0.7 if bisect_result else 0.4,
            proposed_fix="",
            related_tests=related_tests,
            bisect_steps=(
                bisect_result.total_steps if bisect_result else 0
            ),
            created_at=datetime.now(UTC).isoformat(),
        )

        self._save_report(report)
        logger.info(
            "blame_trace_complete",
            commit=breaking_commit[:8],
            confidence=report.confidence,
        )
        return report

    def list_reports(self) -> list[Path]:
        """List all saved blame reports, newest first.

        Returns:
            Sorted list of JSON report file paths.
        """
        return sorted(self._reports_dir.glob("blame_*.json"), reverse=True)

    def load_report(self, path: Path) -> BlameReport:
        """Load a blame report from a JSON file.

        Args:
            path: Path to the JSON report file.

        Returns:
            The deserialized :class:`BlameReport`.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        return BlameReport(**data)

    # ------------------------------------------------------------------
    # Phase 1: Codebase search
    # ------------------------------------------------------------------

    def _find_relevant_code(self, description: str) -> list[str]:
        """Search the codebase for files related to the bug description.

        Extracts keywords (words > 3 chars) from the description and uses
        ``git grep`` to find matching files.

        Args:
            description: The bug description to extract keywords from.

        Returns:
            Up to 20 matching file paths.
        """
        keywords = [w for w in description.lower().split() if len(w) > 3]
        relevant: set[str] = set()

        for keyword in keywords[:5]:
            result = subprocess.run(
                ["git", "grep", "-l", keyword],
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    stripped = line.strip()
                    if stripped:
                        relevant.add(stripped)

        return list(relevant)[:20]

    # ------------------------------------------------------------------
    # Phase 2: Git bisect
    # ------------------------------------------------------------------

    def _run_bisect(
        self, good: str, bad: str, test_cmd: str
    ) -> BisectResult:
        """Run ``git bisect run`` with the supplied test command.

        Always resets bisect state in a ``finally`` block to avoid
        leaving the repo in bisect mode.

        Args:
            good: Known-good commit SHA.
            bad: Known-bad commit SHA.
            test_cmd: Shell command to test each commit.

        Returns:
            A :class:`BisectResult` with the breaking commit.
        """
        logger.info(
            "blame_bisect_start", good=good[:8], bad=bad[:8]
        )
        try:
            self._git("bisect", "start")
            self._git("bisect", "bad", bad)
            self._git("bisect", "good", good)

            result = subprocess.run(
                ["git", "bisect", "run", "sh", "-c", test_cmd],
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=_BISECT_TIMEOUT,
                check=False,
            )

            breaking = ""
            steps = 0
            for line in result.stdout.split("\n"):
                if "is the first bad commit" in line:
                    parts = line.split()
                    if parts:
                        breaking = parts[0]
                if "steps" in line.lower():
                    digits = "".join(c for c in line if c.isdigit())
                    if digits:
                        with contextlib.suppress(ValueError):
                            steps = int(digits[:3])

            return BisectResult(
                breaking_commit=breaking or bad,
                total_steps=steps,
                good_commit=good,
                bad_commit=bad,
            )
        finally:
            self._git("bisect", "reset")

    # ------------------------------------------------------------------
    # Phase 3: Commit analysis
    # ------------------------------------------------------------------

    def _get_commit_info(self, commit: str) -> dict[str, str]:
        """Retrieve metadata for a single commit.

        Args:
            commit: Commit SHA (or ref like ``HEAD``).

        Returns:
            Dict with keys ``hash``, ``author``, ``date``, ``message``,
            ``body``.
        """
        fmt = "%H%n%an%n%aI%n%s%n%b"
        result = self._git("log", "-1", f"--format={fmt}", commit)
        lines = result.strip().split("\n")
        return {
            "hash": lines[0] if len(lines) > 0 else "",
            "author": lines[1] if len(lines) > 1 else "",
            "date": lines[2] if len(lines) > 2 else "",
            "message": lines[3] if len(lines) > 3 else "",
            "body": "\n".join(lines[4:]) if len(lines) > 4 else "",
        }

    def _get_commit_diff(self, commit: str) -> str:
        """Get the full unified diff for a commit.

        Args:
            commit: Commit SHA.

        Returns:
            The raw diff string.
        """
        return self._git("diff", f"{commit}~1..{commit}")

    def _get_changed_files(self, commit: str) -> list[str]:
        """Get the list of files changed in a commit.

        Args:
            commit: Commit SHA.

        Returns:
            List of file paths.
        """
        result = self._git("diff", "--name-only", f"{commit}~1..{commit}")
        return [f for f in result.strip().split("\n") if f.strip()]

    # ------------------------------------------------------------------
    # Phase 4: Narrative and tests
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_changed_lines(diff: str) -> list[str]:
        """Extract individual changed lines from a unified diff.

        Lines prefixed with ``+`` (additions) and ``-`` (removals) are
        collected with their file context.  Header lines (``+++``, ``---``)
        are excluded.

        Args:
            diff: A unified diff string.

        Returns:
            Up to 50 annotated change lines.
        """
        lines: list[str] = []
        current_file = ""
        for line in diff.split("\n"):
            if line.startswith("diff --git"):
                parts = line.split(" b/")
                current_file = parts[-1] if len(parts) > 1 else ""
            elif line.startswith("+") and not line.startswith("+++"):
                lines.append(f"{current_file}: {line[1:]}")
            elif line.startswith("-") and not line.startswith("---"):
                lines.append(f"{current_file}: (removed) {line[1:]}")
        return lines[:50]

    @staticmethod
    def _build_narrative(
        description: str,
        info: dict[str, str],
        diff: str,
        files: list[str],
    ) -> str:
        """Build a causal narrative explaining the regression.

        Args:
            description: Bug description from the user.
            info: Commit metadata dict.
            diff: The full diff.
            files: List of affected files.

        Returns:
            A multi-paragraph narrative string.
        """
        commit_hash = info.get("hash", "")[:8]
        author = info.get("author", "unknown")
        date = info.get("date", "")
        message = info.get("message", "")
        diff_lines = diff.count("\n")
        files_str = ", ".join(files) if files else "(none)"

        return (
            f"Bug: {description}\n\n"
            f"The breaking change was introduced in commit {commit_hash} "
            f"by {author} on {date}.\n\n"
            f"Commit message: {message}\n\n"
            f"Files affected: {files_str}\n\n"
            f"The change modified {len(files)} file(s). "
            f"The diff shows {diff_lines} lines of changes."
        )

    def _find_related_tests(self, affected_files: list[str]) -> list[str]:
        """Find test files that might cover the affected source files.

        Searches the ``tests/`` directory for files named
        ``test_<module>.py`` where ``<module>`` is the stem of each
        affected file.

        Args:
            affected_files: Source files changed in the breaking commit.

        Returns:
            Deduplicated list of test file paths.
        """
        tests: list[str] = []
        for f in affected_files:
            name = Path(f).stem
            result = subprocess.run(
                [
                    "find",
                    "tests",
                    "-name",
                    f"test_{name}.py",
                    "-o",
                    "-name",
                    f"test_*{name}*.py",
                ],
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    stripped = line.strip()
                    if stripped and stripped not in tests:
                        tests.append(stripped)
        return tests

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_report(self, report: BlameReport) -> Path:
        """Persist a blame report as JSON.

        Args:
            report: The report to save.

        Returns:
            Path to the saved JSON file.
        """
        commit_prefix = report.breaking_commit[:8] if report.breaking_commit else "unknown"
        date_prefix = report.created_at[:10] if report.created_at else "undated"
        filename = f"blame_{commit_prefix}_{date_prefix}.json"
        path = self._reports_dir / filename
        path.write_text(
            json.dumps(asdict(report), indent=2), encoding="utf-8"
        )
        logger.info("blame_report_saved", path=str(path))
        return path

    # ------------------------------------------------------------------
    # Git helper
    # ------------------------------------------------------------------

    def _git(self, *args: str) -> str:
        """Run a git command and return stdout.

        Args:
            *args: Arguments to pass to ``git``.

        Returns:
            The stdout of the git command as a string.
        """
        result = subprocess.run(
            ["git", *args],
            cwd=self._root,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
        return result.stdout
