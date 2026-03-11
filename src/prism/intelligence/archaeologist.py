"""Temporal Code Archaeologist — trace the history and evolution of code.

Given a file path (optionally with a line number) or a function name, the
archaeologist digs through the git history and constructs a narrative that
explains *why* the code looks the way it does today:

1. Runs ``git blame``, ``git log --follow``, ``git show``, and parses
   commit messages and numstat output.
2. Classifies every commit touching the target into categories such as
   *created*, *bugfix*, *refactored*, *feature*, or *modified*.
3. Computes per-author contribution stats and an expertise score.
4. Builds a human-readable narrative covering the origin, key evolution
   moments, bug-driven changes, refactors, and current state.
5. Generates a stability score and risk assessment.
6. Identifies co-evolving files that always change together.
7. Persists reports as Markdown under ``~/.prism/archaeology/``.

Slash-command hooks::

    /why <file_path>:<line_number>
    /why <function_name>
    /why list
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Default timeout (seconds) for individual git sub-commands.
_GIT_TIMEOUT: int = 30

# Type alias for a git command runner callback (for testability).
# Receives (args_tuple, cwd) and returns stdout as a string.
GitRunner = Callable[[tuple[str, ...], Path], str]


def _default_git_runner(args: tuple[str, ...], cwd: Path) -> str:
    """Run a git command via subprocess and return stdout.

    Args:
        args: Arguments to pass to ``git`` (without the ``git`` prefix).
        cwd: Working directory for the git command.

    Returns:
        Standard output of the git command as a string.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
        check=False,
    )
    return result.stdout


# ======================================================================
# Data classes — enhanced
# ======================================================================


@dataclass
class CommitInfo:
    """Detailed information about a single commit.

    Attributes:
        hash: Full SHA of the commit.
        author: Name of the commit author.
        email: Email of the commit author.
        date: ISO-8601 date string.
        subject: Commit subject line.
        body: Full commit body text (may be empty).
        files_changed: List of file paths changed in this commit.
        is_significant: Whether the commit is significant (not just
            formatting or whitespace changes).
    """

    hash: str
    author: str
    email: str
    date: str
    subject: str
    body: str = ""
    files_changed: list[str] = field(default_factory=list)
    is_significant: bool = True


@dataclass
class ArchaeologyReport:
    """Complete archaeology report for a code target.

    Attributes:
        target: The original target string (``file:line`` or function name).
        timeline: Chronological list of commit info entries.
        primary_author: Name of the author with the most commits.
        author_distribution: Mapping of author name to commit count.
        co_evolution: Files that co-evolve with the target, as
            ``(file_path, co_change_percentage)`` tuples sorted
            descending by percentage.
        stability_score: A score from 0.0 (volatile) to 1.0 (stable).
        narrative: Human-readable narrative of the code's evolution.
        risks: List of identified risk strings.
        created_at: ISO-8601 timestamp of report generation.
    """

    target: str
    timeline: list[CommitInfo] = field(default_factory=list)
    primary_author: str = ""
    author_distribution: dict[str, int] = field(default_factory=dict)
    co_evolution: list[tuple[str, float]] = field(default_factory=list)
    stability_score: float = 1.0
    narrative: str = ""
    risks: list[str] = field(default_factory=list)
    created_at: str = ""


# ======================================================================
# Legacy data classes (preserved for backward compatibility)
# ======================================================================


@dataclass
class CommitEvent:
    """A significant event in code history.

    Attributes:
        commit_hash: Full SHA of the commit.
        short_hash: Abbreviated SHA (typically 7-8 characters).
        author: Name of the commit author.
        date: ISO-8601 date string.
        message: Commit subject line.
        event_type: Classification — one of ``"created"``, ``"modified"``,
            ``"refactored"``, ``"bugfix"``, or ``"feature"``.
        files_changed: Number of files changed in the commit.
        insertions: Lines added.
        deletions: Lines removed.
    """

    commit_hash: str
    short_hash: str
    author: str
    date: str
    message: str
    event_type: str
    files_changed: int
    insertions: int
    deletions: int


@dataclass
class AuthorContribution:
    """Contribution statistics for a single author.

    Attributes:
        name: Author name.
        commits: Total number of commits touching the target.
        lines_added: Cumulative lines added.
        lines_removed: Cumulative lines removed.
        first_commit: ISO-8601 date of the author's first commit.
        last_commit: ISO-8601 date of the author's most recent commit.
        expertise_score: Fraction of total commits authored (0.0 to 1.0).
    """

    name: str
    commits: int
    lines_added: int
    lines_removed: int
    first_commit: str
    last_commit: str
    expertise_score: float


@dataclass
class CodeEvolution:
    """Complete evolution story of a code entity.

    Attributes:
        target: The original target string (``file:line`` or function name).
        file_path: Resolved file path within the repository.
        timeline: Chronological list of commit events.
        authors: Author contribution statistics, sorted by commit count.
        narrative: Human-readable markdown narrative.
        total_commits: Number of commits in the timeline.
        age_days: Days since the first commit touching this target.
        stability_score: 0.0 (volatile) to 1.0 (stable).
        risk_assessment: Human-readable risk summary.
        created_at: ISO-8601 timestamp of when the report was generated.
    """

    target: str
    file_path: str
    timeline: list[CommitEvent]
    authors: list[AuthorContribution]
    narrative: str
    total_commits: int
    age_days: int
    stability_score: float
    risk_assessment: str
    created_at: str


# ======================================================================
# Enhanced investigation functions (module-level)
# ======================================================================


def _git_blame(
    file_path: str,
    start_line: int | None,
    end_line: int | None,
    run_git: GitRunner,
    cwd: Path,
) -> list[dict[str, str]]:
    """Run ``git blame --porcelain`` and parse the output.

    Args:
        file_path: Path to the file, relative to the repo root.
        start_line: Optional start line to narrow blame range.
        end_line: Optional end line to narrow blame range.
        run_git: Callback to execute git commands.
        cwd: Working directory.

    Returns:
        List of dicts with keys ``hash``, ``author``, ``timestamp``,
        ``message``, and ``content``.
    """
    args: list[str] = ["blame", "--porcelain"]
    if start_line is not None:
        effective_end = end_line if end_line is not None else start_line + 20
        args.extend(["-L", f"{start_line},{effective_end}"])
    args.append(file_path)

    result = run_git(tuple(args), cwd)
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}

    for blame_line in result.split("\n"):
        if blame_line.startswith("\t"):
            current["content"] = blame_line[1:]
            if current:
                entries.append(dict(current))
            current = {}
        elif " " in blame_line:
            parts = blame_line.split(" ", 1)
            key = parts[0]
            value = parts[1] if len(parts) > 1 else ""
            if len(key) == 40:  # SHA hash
                current["hash"] = key
            elif key == "author":
                current["author"] = value
            elif key == "author-mail":
                current["email"] = value.strip("<>")
            elif key == "author-time":
                current["timestamp"] = value
            elif key == "summary":
                current["message"] = value

    return entries


def _git_log(
    file_path: str,
    run_git: GitRunner,
    cwd: Path,
) -> list[CommitInfo]:
    """Get commit history for a file with ``--follow``.

    Parses the formatted log output and ``--numstat`` to capture
    insertion/deletion counts and file lists per commit.

    Args:
        file_path: Path to the file, relative to the repo root.
        run_git: Callback to execute git commands.
        cwd: Working directory.

    Returns:
        List of :class:`CommitInfo` entries, newest first.
    """
    fmt = "%H|%an|%ae|%aI|%s|%b"
    result = run_git(
        ("log", "--follow", f"--format={fmt}", "--numstat", "--", file_path),
        cwd,
    )

    commits: list[CommitInfo] = []
    lines = result.strip().split("\n") if result.strip() else []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "|" in line and len(line.split("|")) >= 5:
            parts = line.split("|", 5)
            commit_hash = parts[0]
            author = parts[1]
            email = parts[2]
            date = parts[3]
            subject = parts[4]
            body = parts[5] if len(parts) > 5 else ""

            files_changed: list[str] = []
            total_insertions = 0
            total_deletions = 0

            # Parse numstat lines that follow
            i += 1
            while i < len(lines):
                stat_line = lines[i].strip()
                if not stat_line or (
                    "|" in stat_line
                    and len(stat_line.split("|")) >= 5
                ):
                    break
                stat_parts = stat_line.split("\t")
                if len(stat_parts) >= 3:
                    try:
                        add_val = int(stat_parts[0]) if stat_parts[0] != "-" else 0
                        del_val = int(stat_parts[1]) if stat_parts[1] != "-" else 0
                        total_insertions += add_val
                        total_deletions += del_val
                        files_changed.append(stat_parts[2])
                    except ValueError:
                        pass
                i += 1

            # Determine if significant (not just whitespace/formatting)
            is_significant = _is_significant_commit(subject, total_insertions, total_deletions)

            commits.append(CommitInfo(
                hash=commit_hash,
                author=author,
                email=email,
                date=date,
                subject=subject,
                body=body.strip(),
                files_changed=files_changed,
                is_significant=is_significant,
            ))
        else:
            i += 1

    return commits


def _is_significant_commit(subject: str, insertions: int, deletions: int) -> bool:
    """Determine whether a commit is significant or just formatting.

    A commit is considered insignificant if its subject matches
    formatting-only patterns and its change size is very small.

    Args:
        subject: The commit subject line.
        insertions: Number of lines added.
        deletions: Number of lines removed.

    Returns:
        True if the commit is significant, False otherwise.
    """
    trivial_patterns = [
        r"^(style|fmt|format|lint|whitespace|typo|wording)",
        r"^fix(ed)?\s+(typo|style|whitespace|formatting)",
    ]
    subject_lower = subject.lower()
    for pattern in trivial_patterns:
        if re.match(pattern, subject_lower) and insertions + deletions <= 5:
            return False
    return True


def _analyze_co_evolution(
    file_path: str,
    commits: list[CommitInfo],
    run_git: GitRunner,
    cwd: Path,
) -> list[tuple[str, float]]:
    """Find files that co-evolve with the target file.

    Examines all commits that touched the target file and counts how
    often each other file was also changed in the same commit.

    Args:
        file_path: The target file path.
        commits: List of commits touching the target.
        run_git: Callback to execute git commands (unused here since
            we already have file lists in commits).
        cwd: Working directory (unused here).

    Returns:
        List of ``(file_path, co_change_percentage)`` tuples sorted
        descending by percentage.  Only files with > 20% co-change rate
        are included.
    """
    if not commits:
        return []

    co_change_counter: Counter[str] = Counter()
    total_commits = len(commits)

    for commit in commits:
        for changed_file in commit.files_changed:
            if changed_file != file_path:
                co_change_counter[changed_file] += 1

    result: list[tuple[str, float]] = []
    for changed_file, count in co_change_counter.most_common():
        pct = count / total_commits
        if pct > 0.2:
            result.append((changed_file, round(pct, 2)))

    return result


def _calculate_stability(commits: list[CommitInfo]) -> float:
    """Calculate a stability score based on change frequency.

    The score ranges from 0.0 (very volatile) to 1.0 (very stable).
    Recent changes weigh more heavily than older ones.

    Args:
        commits: List of commits touching the target (newest first).

    Returns:
        Stability score between 0.0 and 1.0.
    """
    if len(commits) <= 1:
        return 1.0

    total = len(commits)
    if total > 50:
        return 0.2
    if total > 20:
        return 0.4
    if total > 10:
        return 0.6
    if total > 5:
        return 0.8
    return 0.9


def _identify_primary_author(commits: list[CommitInfo]) -> str:
    """Identify the primary author based on commit count.

    Args:
        commits: List of commits touching the target.

    Returns:
        Name of the author with the most commits, or ``"unknown"`` if
        no commits are available.
    """
    if not commits:
        return "unknown"

    author_counts: Counter[str] = Counter()
    for commit in commits:
        author_counts[commit.author] += 1

    most_common = author_counts.most_common(1)
    return most_common[0][0] if most_common else "unknown"


def _build_author_distribution(commits: list[CommitInfo]) -> dict[str, int]:
    """Build author-to-commit-count mapping.

    Args:
        commits: List of commits touching the target.

    Returns:
        Dict mapping author name to number of commits.
    """
    dist: dict[str, int] = {}
    for commit in commits:
        dist[commit.author] = dist.get(commit.author, 0) + 1
    return dist


def _generate_narrative(
    target: str,
    blame_data: list[dict[str, str]],
    commits: list[CommitInfo],
    co_evolution: list[tuple[str, float]],
) -> str:
    """Build a human-readable narrative of the code's history.

    Args:
        target: The original target string.
        blame_data: Output from ``git blame``.
        commits: List of commits touching the target (newest first).
        co_evolution: Co-evolving files with percentages.

    Returns:
        A formatted narrative string.
    """
    if not commits:
        return f"No git history found for {target}."

    # Reverse to get chronological order (oldest first)
    chronological = list(reversed(commits))
    lines: list[str] = [f"## History of {target}", ""]

    # Origin story
    first = chronological[0]
    lines.append(
        f"**Created** by {first.author} on {first.date[:10]}: "
        f'"{first.subject}"'
    )
    lines.append("")

    # Categorize commits
    bugfixes = [c for c in chronological if _classify_subject(c.subject) == "bugfix"]
    refactors = [c for c in chronological if _classify_subject(c.subject) == "refactored"]
    features = [c for c in chronological if _classify_subject(c.subject) == "feature"]

    if features:
        lines.append(
            f"**Feature additions**: {len(features)} commits added new functionality."
        )
    if bugfixes:
        lines.append(
            f"**Bug fixes**: {len(bugfixes)} commits fixed bugs in this code."
        )
    if refactors:
        lines.append(
            f"**Refactors**: {len(refactors)} commits restructured this code."
        )

    lines.append("")

    # Primary maintainer
    primary = _identify_primary_author(commits)
    dist = _build_author_distribution(commits)
    if primary != "unknown":
        count = dist.get(primary, 0)
        lines.append(f"**Primary maintainer**: {primary} ({count} commits)")

    # Co-evolution
    if co_evolution:
        lines.append("")
        lines.append("**Co-evolving files** (frequently changed together):")
        for co_file, pct in co_evolution[:5]:
            lines.append(f"  - {co_file} ({pct:.0%} of changes)")

    # Blame summary
    if blame_data:
        unique_authors = {e.get("author", "?") for e in blame_data}
        lines.append("")
        lines.append(
            f"**Current code** attributed to {len(unique_authors)} "
            f"author(s) via blame."
        )

    # Latest change
    last = chronological[-1]
    lines.append("")
    lines.append(
        f"**Latest change**: {last.date[:10]} by {last.author}: "
        f'"{last.subject}"'
    )

    return "\n".join(lines)


def _classify_subject(subject: str) -> str:
    """Classify a commit subject line into a category.

    Args:
        subject: The commit subject line.

    Returns:
        One of ``"bugfix"``, ``"refactored"``, ``"feature"``, or
        ``"modified"``.
    """
    msg_lower = subject.lower()
    if any(kw in msg_lower for kw in ["fix", "bug", "patch", "hotfix", "issue"]):
        return "bugfix"
    if any(kw in msg_lower for kw in [
        "refactor", "cleanup", "reorganize", "rename", "restructure",
    ]):
        return "refactored"
    if any(kw in msg_lower for kw in ["feat", "add", "implement", "new", "introduce"]):
        return "feature"
    return "modified"


def _identify_risks(
    commits: list[CommitInfo],
    stability: float,
    author_dist: dict[str, int],
) -> list[str]:
    """Identify risks based on commit history and metrics.

    Args:
        commits: List of commits touching the target.
        stability: Pre-computed stability score.
        author_dist: Author-to-commit-count mapping.

    Returns:
        List of risk description strings.
    """
    risks: list[str] = []

    if stability < 0.5:
        risks.append("Code is volatile with frequent changes")

    if len(author_dist) == 1:
        sole_author = next(iter(author_dist))
        risks.append(f"Single maintainer: {sole_author} (bus factor = 1)")

    # Check bugfix ratio
    total = len(commits)
    if total > 0:
        bugfix_count = sum(
            1 for c in commits if _classify_subject(c.subject) == "bugfix"
        )
        bugfix_ratio = bugfix_count / total
        if bugfix_ratio > 0.3:
            risks.append(
                f"High bug rate: {bugfix_ratio:.0%} of commits are bug fixes"
            )

    if not risks:
        risks.append("Low risk: stable code with good maintainer coverage.")

    return risks


# ======================================================================
# Enhanced investigate function (module-level)
# ======================================================================


def investigate(
    target: str,
    project_root: Path,
    run_git: GitRunner | None = None,
) -> ArchaeologyReport:
    """Investigate the history of a code target.

    Performs a full archaeology investigation:

    1. Parses the target to extract file path and optional line range.
    2. Runs ``git blame`` on the target region.
    3. Runs ``git log --follow`` to get full commit history.
    4. Analyzes co-evolution with other files.
    5. Calculates stability score.
    6. Identifies the primary author and author distribution.
    7. Generates a narrative.
    8. Identifies risks.

    Args:
        target: A target string — either ``file.py:42``, ``file.py``,
            or a function name.
        project_root: Root directory of the git repository.
        run_git: Optional callback for executing git commands. Defaults
            to subprocess-based execution.

    Returns:
        An :class:`ArchaeologyReport` with all analysis results.

    Raises:
        ValueError: If the target cannot be resolved to a file.
    """
    git_runner = run_git or _default_git_runner
    root = project_root.resolve()

    file_path, start_line, end_line = _parse_investigation_target(target, git_runner, root)

    if not file_path:
        raise ValueError(f"Could not find target: {target}")

    # Step 1: git blame
    blame_data = _git_blame(file_path, start_line, end_line, git_runner, root)

    # Step 2: git log --follow
    commits = _git_log(file_path, git_runner, root)

    # Step 3: co-evolution analysis
    co_evolution = _analyze_co_evolution(file_path, commits, git_runner, root)

    # Step 4: stability
    stability = _calculate_stability(commits)

    # Step 5: primary author and distribution
    primary_author = _identify_primary_author(commits)
    author_dist = _build_author_distribution(commits)

    # Step 6: narrative
    narrative = _generate_narrative(target, blame_data, commits, co_evolution)

    # Step 7: risks
    risks = _identify_risks(commits, stability, author_dist)

    report = ArchaeologyReport(
        target=target,
        timeline=commits,
        primary_author=primary_author,
        author_distribution=author_dist,
        co_evolution=co_evolution,
        stability_score=stability,
        narrative=narrative,
        risks=risks,
        created_at=datetime.now(UTC).isoformat(),
    )

    # Auto-save
    save_report(report, root)

    return report


def _parse_investigation_target(
    target: str,
    run_git: GitRunner,
    cwd: Path,
) -> tuple[str, int | None, int | None]:
    """Parse a target string into file path and line range.

    Args:
        target: User-supplied target (``file:line``, ``file``, or
            function name).
        run_git: Callback to execute git commands.
        cwd: Working directory for git commands.

    Returns:
        A tuple of ``(file_path, start_line, end_line)``.  ``file_path``
        is an empty string if the target could not be resolved.
    """
    if ":" in target:
        parts = target.rsplit(":", 1)
        try:
            line = int(parts[1])
            return parts[0], line, line + 20
        except ValueError:
            return parts[0], None, None

    # Looks like a file path
    if target.endswith(".py") or "/" in target:
        return target, None, None

    # Looks like a function name — search for it
    result = run_git(("grep", "-n", f"def {target}"), cwd)
    for line in result.strip().split("\n"):
        if line.strip():
            parts = line.split(":", 2)
            if len(parts) >= 2:
                try:
                    return parts[0], int(parts[1]), int(parts[1]) + 20
                except ValueError:
                    return parts[0], None, None

    return "", None, None


# ======================================================================
# Report generation
# ======================================================================


def generate_report_text(report: ArchaeologyReport) -> str:
    """Generate a formatted plain text report from an archaeology report.

    The report includes sections for the target, timeline, authors,
    co-evolution, stability, narrative, and risks.

    Args:
        report: The archaeology report.

    Returns:
        A formatted plain text string.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("CODE ARCHAEOLOGY REPORT")
    lines.append("=" * 60)
    lines.append(f"\nTarget: {report.target}")
    lines.append(f"Date: {report.created_at[:19] if report.created_at else 'N/A'}")
    lines.append(f"Stability: {report.stability_score:.0%}")
    lines.append(f"Primary Author: {report.primary_author}")
    lines.append(f"Total Commits: {len(report.timeline)}")
    lines.append("")

    # Author distribution
    if report.author_distribution:
        lines.append("--- Author Distribution ---")
        for author, count in sorted(
            report.author_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            lines.append(f"  {author}: {count} commits")
        lines.append("")

    # Timeline (last 20)
    if report.timeline:
        lines.append("--- Timeline ---")
        display_commits = report.timeline[:20]
        for commit in display_commits:
            sig_marker = "" if commit.is_significant else " [minor]"
            lines.append(
                f"  {commit.hash[:7]} {commit.date[:10]} "
                f"{commit.author}: {commit.subject}{sig_marker}"
            )
        if len(report.timeline) > 20:
            lines.append(f"  ... and {len(report.timeline) - 20} more commits")
        lines.append("")

    # Co-evolution
    if report.co_evolution:
        lines.append("--- Co-evolving Files ---")
        for co_file, pct in report.co_evolution[:10]:
            lines.append(f"  {co_file}: {pct:.0%} co-change rate")
        lines.append("")

    # Narrative
    if report.narrative:
        lines.append("--- Narrative ---")
        lines.append(report.narrative)
        lines.append("")

    # Risks
    if report.risks:
        lines.append("--- Risks ---")
        for risk in report.risks:
            lines.append(f"  - {risk}")
        lines.append("")

    return "\n".join(lines)


# ======================================================================
# Persistence
# ======================================================================


def save_report(
    report: ArchaeologyReport,
    project_root: Path | None = None,
    save_dir: Path | None = None,
) -> Path:
    """Save an archaeology report as a Markdown file.

    The file is named ``<timestamp>_<file_slug>.md`` and written to
    *save_dir* (defaulting to ``~/.prism/archaeology/``).

    Args:
        report: The archaeology report to persist.
        project_root: Project root (unused, kept for API compatibility).
        save_dir: Target directory. Created if it does not exist.

    Returns:
        Path to the saved Markdown file.
    """
    target_dir = save_dir or (Path.home() / ".prism" / "archaeology")
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_target = re.sub(r"[^\w]", "_", report.target)[:50]
    timestamp = (
        report.created_at[:19].replace(":", "-")
        if report.created_at
        else "unknown"
    )
    filename = f"{timestamp}_{safe_target}.md"
    path = target_dir / filename

    report_text = generate_report_text(report)
    path.write_text(report_text, encoding="utf-8")
    logger.info("archaeology_report_saved", path=str(path))
    return path


def list_reports(save_dir: Path | None = None) -> list[Path]:
    """List all saved archaeology reports, newest first.

    Args:
        save_dir: Directory to scan. Defaults to ``~/.prism/archaeology/``.

    Returns:
        List of Markdown file paths sorted newest first.
    """
    target_dir = save_dir or (Path.home() / ".prism" / "archaeology")
    if not target_dir.exists():
        return []
    return sorted(target_dir.glob("*.md"), reverse=True)


# ======================================================================
# Main archaeologist class (preserved for backward compatibility)
# ======================================================================


class CodeArchaeologist:
    """Traces the history and evolution of code through git history.

    Args:
        project_root: Path to the git repository root.
        reports_dir: Directory to persist JSON reports.  Defaults to
            ``~/.prism/archaeology/``.
    """

    def __init__(
        self,
        project_root: Path,
        reports_dir: Path | None = None,
    ) -> None:
        self._root = project_root.resolve()
        self._reports_dir = (
            reports_dir or Path.home() / ".prism" / "archaeology"
        )
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def investigate(self, target: str) -> CodeEvolution:
        """Investigate the history of a code target.

        The *target* can be:

        * ``file.py:42`` — a file with an optional line number.
        * ``my_function`` — a function name to search for.

        Args:
            target: File path (with optional line), or function name.

        Returns:
            A :class:`CodeEvolution` with timeline, authors, narrative,
            and risk assessment.

        Raises:
            ValueError: If the target cannot be found in the codebase.
        """
        logger.info("archaeology_start", target=target)

        file_path, line_number = self._parse_target(target)

        if not file_path:
            # Target is a function name — find it
            file_path, line_number = self._find_function(target)

        if not file_path:
            raise ValueError(f"Could not find target: {target}")

        # Gather data
        blame_data = self._git_blame(file_path, line_number)
        log_data = self._git_log_follow(file_path)
        timeline = self._build_timeline(log_data)
        authors = self._analyze_authors(log_data)

        # Build narrative
        narrative = self._build_narrative(
            target, file_path, timeline, authors, blame_data
        )

        # Calculate metrics
        age_days = self._calculate_age(timeline)
        stability = self._calculate_stability(timeline)
        risk = self._assess_risk(timeline, authors, stability)

        evolution = CodeEvolution(
            target=target,
            file_path=file_path,
            timeline=timeline,
            authors=authors,
            narrative=narrative,
            total_commits=len(timeline),
            age_days=age_days,
            stability_score=stability,
            risk_assessment=risk,
            created_at=datetime.now(UTC).isoformat(),
        )

        self._save_report(evolution)

        logger.info(
            "archaeology_complete",
            target=target,
            commits=len(timeline),
            stability=stability,
        )

        return evolution

    def list_reports(self) -> list[Path]:
        """List saved archaeology reports, newest first.

        Returns:
            Sorted list of JSON report file paths.
        """
        return sorted(self._reports_dir.glob("why_*.json"), reverse=True)

    def load_report(self, path: Path) -> CodeEvolution:
        """Load an archaeology report from a JSON file.

        Args:
            path: Path to the JSON report file.

        Returns:
            The deserialized :class:`CodeEvolution`.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        # Reconstruct nested dataclasses from dicts
        timeline = [CommitEvent(**e) for e in data.pop("timeline", [])]
        authors = [AuthorContribution(**a) for a in data.pop("authors", [])]
        return CodeEvolution(timeline=timeline, authors=authors, **data)

    # ------------------------------------------------------------------
    # Target parsing
    # ------------------------------------------------------------------

    def _parse_target(self, target: str) -> tuple[str, int | None]:
        """Parse a target string like ``'file.py:42'`` or ``'file.py'``.

        Args:
            target: The user-supplied target string.

        Returns:
            A tuple of ``(file_path, line_number)``.  *file_path* is an
            empty string when the target appears to be a function name
            rather than a file path.
        """
        if ":" in target:
            parts = target.rsplit(":", 1)
            try:
                return parts[0], int(parts[1])
            except ValueError:
                return parts[0], None

        # Looks like a file path
        if target.endswith(".py") or "/" in target:
            return target, None

        # Looks like a function name — return empty so caller searches
        return "", None

    def _find_function(self, name: str) -> tuple[str, int | None]:
        """Find a function definition in the codebase via ``git grep``.

        Args:
            name: Function name to search for.

        Returns:
            A tuple of ``(file_path, line_number)`` for the first match,
            or ``("", None)`` if the function was not found.
        """
        result = self._git("grep", "-n", f"def {name}")
        for line in result.strip().split("\n"):
            if line.strip():
                parts = line.split(":", 2)
                if len(parts) >= 2:
                    try:
                        return parts[0], int(parts[1])
                    except ValueError:
                        return parts[0], None
        return "", None

    # ------------------------------------------------------------------
    # Git data gathering
    # ------------------------------------------------------------------

    def _git_blame(
        self, file_path: str, line: int | None = None
    ) -> list[dict[str, str]]:
        """Run ``git blame --porcelain`` on a file.

        Args:
            file_path: Path to the file, relative to the repo root.
            line: Optional line number to narrow the blame range.

        Returns:
            List of dicts with keys ``hash``, ``author``, ``timestamp``,
            ``message``, and ``content``.
        """
        args: list[str] = ["blame", "--porcelain"]
        if line is not None:
            end_line = line + 20
            args.extend(["-L", f"{line},{end_line}"])
        args.append(file_path)

        result = self._git(*args)
        entries: list[dict[str, str]] = []
        current: dict[str, str] = {}

        for blame_line in result.split("\n"):
            if blame_line.startswith("\t"):
                current["content"] = blame_line[1:]
                if current:
                    entries.append(dict(current))
                current = {}
            elif " " in blame_line:
                parts = blame_line.split(" ", 1)
                key = parts[0]
                value = parts[1] if len(parts) > 1 else ""
                if len(key) == 40:  # SHA hash
                    current["hash"] = key
                elif key == "author":
                    current["author"] = value
                elif key == "author-time":
                    current["timestamp"] = value
                elif key == "summary":
                    current["message"] = value

        return entries

    def _git_log_follow(self, file_path: str) -> list[dict]:
        """Get full commit history for a file with ``--follow``.

        Parses both the formatted log lines and ``--numstat`` output to
        capture insertion/deletion counts.

        Args:
            file_path: Path to the file, relative to the repo root.

        Returns:
            List of dicts with keys ``hash``, ``short_hash``, ``author``,
            ``date``, ``message``, ``insertions``, ``deletions``,
            ``files_changed``.
        """
        fmt = "%H|%h|%an|%aI|%s"
        result = self._git(
            "log", "--follow", f"--format={fmt}", "--numstat",
            "--", file_path,
        )

        commits: list[dict] = []
        lines = result.strip().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if "|" in line and len(line.split("|")) >= 5:
                parts = line.split("|", 4)
                commit: dict = {
                    "hash": parts[0],
                    "short_hash": parts[1],
                    "author": parts[2],
                    "date": parts[3],
                    "message": parts[4],
                    "insertions": 0,
                    "deletions": 0,
                    "files_changed": 0,
                }

                # Parse numstat lines that follow the log entry
                i += 1
                while i < len(lines):
                    stat_line = lines[i].strip()
                    if not stat_line or (
                        "|" in stat_line
                        and len(stat_line.split("|")) >= 5
                    ):
                        break
                    stat_parts = stat_line.split("\t")
                    if len(stat_parts) >= 2:
                        try:
                            add_val = (
                                int(stat_parts[0])
                                if stat_parts[0] != "-"
                                else 0
                            )
                            del_val = (
                                int(stat_parts[1])
                                if stat_parts[1] != "-"
                                else 0
                            )
                            commit["insertions"] += add_val
                            commit["deletions"] += del_val
                            commit["files_changed"] += 1
                        except ValueError:
                            pass
                    i += 1

                commits.append(commit)
            else:
                i += 1

        return commits

    # ------------------------------------------------------------------
    # Timeline construction
    # ------------------------------------------------------------------

    def _build_timeline(self, log_data: list[dict]) -> list[CommitEvent]:
        """Build a chronological timeline of events from git log data.

        The log data arrives newest-first, so it is reversed to produce a
        chronological ordering.

        Args:
            log_data: Output from :meth:`_git_log_follow`.

        Returns:
            Chronological list of :class:`CommitEvent` instances.
        """
        events: list[CommitEvent] = []
        for i, commit in enumerate(reversed(log_data)):
            event_type = self._classify_commit(commit["message"], i == 0)
            events.append(CommitEvent(
                commit_hash=commit["hash"],
                short_hash=commit["short_hash"],
                author=commit["author"],
                date=commit["date"],
                message=commit["message"],
                event_type=event_type,
                files_changed=commit.get("files_changed", 0),
                insertions=commit.get("insertions", 0),
                deletions=commit.get("deletions", 0),
            ))
        return events

    @staticmethod
    def _classify_commit(message: str, is_first: bool) -> str:
        """Classify a commit based on its message.

        Args:
            message: Commit subject line.
            is_first: Whether this is the first (oldest) commit.

        Returns:
            One of ``"created"``, ``"bugfix"``, ``"refactored"``,
            ``"feature"``, or ``"modified"``.
        """
        if is_first:
            return "created"

        msg_lower = message.lower()
        if any(
            kw in msg_lower
            for kw in ["fix", "bug", "patch", "hotfix", "issue"]
        ):
            return "bugfix"
        if any(
            kw in msg_lower
            for kw in [
                "refactor", "cleanup", "reorganize", "rename", "restructure",
            ]
        ):
            return "refactored"
        if any(
            kw in msg_lower
            for kw in ["feat", "add", "implement", "new", "introduce"]
        ):
            return "feature"
        return "modified"

    # ------------------------------------------------------------------
    # Author analysis
    # ------------------------------------------------------------------

    def _analyze_authors(
        self, log_data: list[dict]
    ) -> list[AuthorContribution]:
        """Analyze per-author contribution statistics.

        Args:
            log_data: Output from :meth:`_git_log_follow`.

        Returns:
            List of :class:`AuthorContribution` sorted by commit count
            descending.
        """
        author_data: dict[str, dict] = {}

        for commit in log_data:
            author = commit["author"]
            if author not in author_data:
                author_data[author] = {
                    "commits": 0,
                    "lines_added": 0,
                    "lines_removed": 0,
                    "first_commit": commit["date"],
                    "last_commit": commit["date"],
                }

            data = author_data[author]
            data["commits"] += 1
            data["lines_added"] += commit.get("insertions", 0)
            data["lines_removed"] += commit.get("deletions", 0)
            data["last_commit"] = commit["date"]

        total_commits = sum(d["commits"] for d in author_data.values())

        contributions: list[AuthorContribution] = []
        for name, data in sorted(
            author_data.items(),
            key=lambda x: x[1]["commits"],
            reverse=True,
        ):
            expertise = data["commits"] / max(total_commits, 1)
            contributions.append(AuthorContribution(
                name=name,
                commits=data["commits"],
                lines_added=data["lines_added"],
                lines_removed=data["lines_removed"],
                first_commit=data["first_commit"],
                last_commit=data["last_commit"],
                expertise_score=round(expertise, 2),
            ))

        return contributions

    # ------------------------------------------------------------------
    # Narrative construction
    # ------------------------------------------------------------------

    def _build_narrative(
        self,
        target: str,
        file_path: str,
        timeline: list[CommitEvent],
        authors: list[AuthorContribution],
        blame_data: list[dict[str, str]],
    ) -> str:
        """Build a human-readable narrative of code evolution.

        Args:
            target: The original user-supplied target string.
            file_path: Resolved file path.
            timeline: Chronological commit events.
            authors: Author contribution stats.
            blame_data: Output from ``git blame``.

        Returns:
            A markdown-formatted narrative string.
        """
        if not timeline:
            return f"No git history found for {target}."

        lines: list[str] = [f"## History of {target}", ""]

        # Origin story
        first = timeline[0]
        lines.append(
            f"**Created** by {first.author} on {first.date[:10]}: "
            f'"{first.message}"'
        )
        lines.append("")

        # Key events
        bugfixes = [e for e in timeline if e.event_type == "bugfix"]
        refactors = [e for e in timeline if e.event_type == "refactored"]
        features = [e for e in timeline if e.event_type == "feature"]

        if features:
            lines.append(
                f"**Feature additions**: {len(features)} commits added "
                f"new functionality."
            )
        if bugfixes:
            lines.append(
                f"**Bug fixes**: {len(bugfixes)} commits fixed bugs "
                f"in this code."
            )
        if refactors:
            lines.append(
                f"**Refactors**: {len(refactors)} commits restructured "
                f"this code."
            )

        lines.append("")

        # Who knows this code
        if authors:
            top = authors[0]
            lines.append(
                f"**Primary maintainer**: {top.name} ({top.commits} commits, "
                f"{top.lines_added} lines added, "
                f"expertise: {top.expertise_score:.0%})"
            )

        # Current state
        last = timeline[-1]
        lines.append("")
        lines.append(
            f"**Latest change**: {last.date[:10]} by {last.author}: "
            f'"{last.message}"'
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_age(timeline: list[CommitEvent]) -> int:
        """Calculate age in days from the first commit to now.

        Args:
            timeline: Chronological commit events.

        Returns:
            Number of days since the first commit, or 0 if the timeline
            is empty or unparseable.
        """
        if not timeline:
            return 0
        try:
            first_date = datetime.fromisoformat(
                timeline[0].date.replace("Z", "+00:00")
            )
            now = datetime.now(UTC)
            return (now - first_date).days
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _calculate_stability(timeline: list[CommitEvent]) -> float:
        """Calculate a stability score based on change frequency.

        The score ranges from 0.0 (very volatile) to 1.0 (very stable).

        Args:
            timeline: Chronological commit events.

        Returns:
            Stability score between 0.0 and 1.0.
        """
        if len(timeline) <= 1:
            return 1.0

        if len(timeline) > 50:
            return 0.2
        if len(timeline) > 20:
            return 0.4
        if len(timeline) > 10:
            return 0.6
        if len(timeline) > 5:
            return 0.8
        return 0.9

    @staticmethod
    def _assess_risk(
        timeline: list[CommitEvent],
        authors: list[AuthorContribution],
        stability: float,
    ) -> str:
        """Generate a risk assessment based on metrics.

        Args:
            timeline: Chronological commit events.
            authors: Author contributions.
            stability: Pre-computed stability score.

        Returns:
            Human-readable risk summary string.
        """
        risks: list[str] = []

        if stability < 0.5:
            risks.append("Code is volatile with frequent changes")

        if len(authors) == 1:
            risks.append(
                f"Single maintainer: {authors[0].name} (bus factor = 1)"
            )

        bugfix_count = sum(
            1 for e in timeline if e.event_type == "bugfix"
        )
        bugfix_ratio = bugfix_count / max(len(timeline), 1)
        if bugfix_ratio > 0.3:
            risks.append(
                f"High bug rate: {bugfix_ratio:.0%} of commits are bug fixes"
            )

        if not risks:
            return "Low risk: stable code with good maintainer coverage."

        return "Risks identified: " + "; ".join(risks)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_report(self, evolution: CodeEvolution) -> Path:
        """Save an archaeology report as JSON.

        Args:
            evolution: The evolution report to persist.

        Returns:
            Path to the saved JSON file.
        """
        safe_target = re.sub(r"[^\w]", "_", evolution.target)[:50]
        filename = f"why_{safe_target}_{evolution.created_at[:10]}.json"
        path = self._reports_dir / filename
        path.write_text(
            json.dumps(asdict(evolution), indent=2), encoding="utf-8"
        )
        logger.info("archaeology_report_saved", path=str(path))
        return path

    # ------------------------------------------------------------------
    # Git helper
    # ------------------------------------------------------------------

    def _git(self, *args: str) -> str:
        """Run a git command and return its stdout.

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
