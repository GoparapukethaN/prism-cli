"""Smart Context Budget Manager — intelligent context selection with relevance graphs.

Provides AST-based import analysis, multi-level relevance scoring, token budget
allocation with 40/10/50 splits (response/system/context), SQLite efficiency
logging, and a rich display formatter.

Slash-command hook:
    /context             — show current context allocation
    /context add <file>  — force-include a file
    /context drop <file> — force-exclude a file
    /context stats       — show efficiency metrics

CLI hook:
    prism context [show|stats]
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from prism.db.database import Database

logger = structlog.get_logger(__name__)


# ======================================================================
# Constants
# ======================================================================

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-sonnet-4-20250514": 200_000,
    "claude-3-opus-20240229": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "deepseek/deepseek-chat": 128_000,
    "gemini/gemini-1.5-pro": 1_000_000,
    "gemini/gemini-1.5-flash": 1_000_000,
    "groq/llama-3.1-70b-versatile": 131_072,
    "ollama/llama3.1:8b": 8_192,
    "ollama/qwen2.5-coder:7b": 32_768,
}

DEFAULT_CONTEXT_WINDOW = 128_000
TOKEN_CHARS_RATIO = 4  # approximate chars per token

# Budget split percentages
RESPONSE_RESERVE_PCT = 0.40
SYSTEM_RESERVE_PCT = 0.10
CONTEXT_BUDGET_PCT = 0.50


# ======================================================================
# Enums
# ======================================================================


class RelevanceLevel(Enum):
    """Relevance scoring tiers for context items.

    Each level has a default score that can be overridden by manual
    include/exclude directives.
    """

    DIRECT = 1.0
    RELATED = 0.85
    INDIRECT = 0.6
    CONTEXT_ONLY = 0.3
    EXCLUDED = 0.0


# ======================================================================
# Data models
# ======================================================================


@dataclass
class ContextItem:
    """A single item considered for context inclusion.

    Attributes:
        path: File path (relative to project root).
        relevance: Numeric relevance score (0.0 to 1.0).
        token_count: Estimated token count for this item.
        reason: Human-readable explanation of why this score was assigned.
        level: The :class:`RelevanceLevel` that determined the score.
    """

    path: str
    relevance: float
    token_count: int
    reason: str
    level: RelevanceLevel


@dataclass
class BudgetAllocation:
    """Result of a context budget allocation.

    Attributes:
        total_tokens: The model's full context window.
        response_tokens: Tokens reserved for the response (40%).
        system_tokens: Tokens reserved for the system prompt (10%).
        context_tokens: Tokens available for context (50%).
        items_included: Context items selected for inclusion.
        items_excluded: Context items that did not fit.
        tokens_used: Total tokens consumed by included items.
        tokens_remaining: Tokens still available within context budget.
        efficiency_pct: Percentage of context budget actually used.
    """

    total_tokens: int
    response_tokens: int
    system_tokens: int
    context_tokens: int
    items_included: list[ContextItem]
    items_excluded: list[ContextItem]
    tokens_used: int
    tokens_remaining: int
    efficiency_pct: float


@dataclass
class ContextEfficiencyRecord:
    """Record for SQLite efficiency logging.

    Attributes:
        task_type: Category of the task (e.g. "code_edit", "question").
        files_included: Number of files included in context.
        tokens_used: Total tokens consumed.
        files_excluded: Number of files excluded.
        outcome: Result of the task ("success" or "failure").
        model_used: The model identifier used.
        created_at: ISO-8601 timestamp.
    """

    task_type: str
    files_included: int
    tokens_used: int
    files_excluded: int
    outcome: str
    model_used: str
    created_at: str


@dataclass
class EfficiencyStats:
    """Aggregate efficiency statistics from SQLite.

    Attributes:
        avg_tokens_used: Mean tokens used per task.
        avg_efficiency_pct: Mean efficiency percentage.
        success_rate: Fraction of tasks that succeeded (0.0 to 1.0).
        total_records: Number of records in the database.
        total_tokens_saved: Estimated tokens saved vs naive approach.
    """

    avg_tokens_used: float
    avg_efficiency_pct: float
    success_rate: float
    total_records: int
    total_tokens_saved: int


# ======================================================================
# Smart Context Budget Manager
# ======================================================================


class SmartContextBudgetManager:
    """Manages context budget allocation with relevance-graph scoring.

    Given a task description and a project root, this manager:
    1. Parses the task to extract mentioned files/functions/errors
    2. Builds a relevance graph using AST-based import analysis
    3. Scores items by relevance level
    4. Fills the context budget in score order until the limit is reached

    Special rules:
    - Error messages, the file being modified, and test files for modified
      files are always included (forced to score 1.0).
    - Large files are truncated to relevant sections (functions matching
      the query) when they exceed 50% of the remaining budget.

    Args:
        project_root: Absolute path to the project root directory.
        db: Optional :class:`Database` instance for efficiency logging.
    """

    def __init__(
        self,
        project_root: Path,
        db: Database | None = None,
    ) -> None:
        self._root: Path = project_root.resolve()
        self._db = db
        self._manual_includes: set[str] = set()
        self._manual_excludes: set[str] = set()

        if self._db is not None:
            self._ensure_table()

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        """Create the context_efficiency table if it does not exist."""
        if self._db is None:
            return
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS context_efficiency (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                files_included INTEGER NOT NULL,
                tokens_used INTEGER NOT NULL,
                files_excluded INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                model_used TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._db.commit()

    # ------------------------------------------------------------------
    # Public API — allocation
    # ------------------------------------------------------------------

    def allocate(
        self,
        task_description: str,
        available_files: list[str],
        model: str = "claude-sonnet-4-20250514",
        error_context: str | None = None,
    ) -> BudgetAllocation:
        """Allocate context budget for *task_description*.

        Args:
            task_description: Natural-language task description.
            available_files: Candidate file paths (relative to project root).
            model: Model identifier for context window lookup.
            error_context: Optional error/stack-trace text (always included).

        Returns:
            A :class:`BudgetAllocation` describing the allocation.
        """
        total_tokens = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
        response_tokens = int(total_tokens * RESPONSE_RESERVE_PCT)
        system_tokens = int(total_tokens * SYSTEM_RESERVE_PCT)
        context_tokens = total_tokens - response_tokens - system_tokens

        # Parse task to find mentioned files and functions
        mentioned_files = self._extract_mentioned_files(
            task_description, available_files,
        )
        mentioned_functions = self._extract_mentioned_functions(task_description)

        # Build relevance graph
        relevance_map = self.build_relevance_graph(
            mentioned_files, self._root,
        )

        # Score all available files
        items: list[ContextItem] = []
        for file_path in available_files:
            if file_path in self._manual_excludes:
                items.append(ContextItem(
                    path=file_path,
                    relevance=0.0,
                    token_count=0,
                    reason="manually excluded",
                    level=RelevanceLevel.EXCLUDED,
                ))
                continue

            level, reason = self._score_file(
                file_path,
                task_description,
                mentioned_files,
                mentioned_functions,
                relevance_map,
            )

            # Force-included files
            if file_path in self._manual_includes:
                level = RelevanceLevel.DIRECT
                reason = "manually included"

            content = self._read_file_safe(file_path)
            if not content and level != RelevanceLevel.EXCLUDED:
                continue

            token_count = estimate_tokens(content)

            # Truncate large files to relevant sections
            if (
                token_count > context_tokens * 0.5
                and level != RelevanceLevel.DIRECT
                and mentioned_functions
            ):
                content = self._truncate_to_functions(
                    content, mentioned_functions,
                )
                token_count = estimate_tokens(content)
                reason += " (truncated to relevant functions)"

            items.append(ContextItem(
                path=file_path,
                relevance=level.value,
                token_count=token_count,
                reason=reason,
                level=level,
            ))

        # Add error context as highest-priority item
        if error_context:
            err_tokens = estimate_tokens(error_context)
            items.append(ContextItem(
                path="<error_context>",
                relevance=1.0,
                token_count=err_tokens,
                reason="error/stack trace — always included",
                level=RelevanceLevel.DIRECT,
            ))

        # Sort by relevance descending
        items.sort(key=lambda it: it.relevance, reverse=True)

        # Greedily fill budget
        included: list[ContextItem] = []
        excluded: list[ContextItem] = []
        used = 0

        for item in items:
            if item.relevance <= 0.0:
                excluded.append(item)
                continue

            if used + item.token_count <= context_tokens:
                included.append(item)
                used += item.token_count
            else:
                excluded.append(item)

        remaining = context_tokens - used
        efficiency = (used / max(context_tokens, 1)) * 100.0

        logger.debug(
            "context_allocated",
            model=model,
            included=len(included),
            excluded=len(excluded),
            tokens_used=used,
            context_budget=context_tokens,
            efficiency_pct=f"{efficiency:.1f}",
        )

        return BudgetAllocation(
            total_tokens=total_tokens,
            response_tokens=response_tokens,
            system_tokens=system_tokens,
            context_tokens=context_tokens,
            items_included=included,
            items_excluded=excluded,
            tokens_used=used,
            tokens_remaining=remaining,
            efficiency_pct=efficiency,
        )

    # ------------------------------------------------------------------
    # Public API — manual overrides
    # ------------------------------------------------------------------

    def add_file(self, file_path: str) -> None:
        """Force-include a file in context (set score to 1.0).

        Args:
            file_path: Path of the file to force-include.
        """
        self._manual_includes.add(file_path)
        self._manual_excludes.discard(file_path)

    def drop_file(self, file_path: str) -> None:
        """Force-exclude a file from context (set score to 0.0).

        Args:
            file_path: Path of the file to force-exclude.
        """
        self._manual_excludes.add(file_path)
        self._manual_includes.discard(file_path)

    def reset_overrides(self) -> None:
        """Clear all manual includes and excludes."""
        self._manual_includes.clear()
        self._manual_excludes.clear()

    # ------------------------------------------------------------------
    # Public API — relevance graph
    # ------------------------------------------------------------------

    def build_relevance_graph(
        self,
        mentioned_files: list[str],
        project_root: Path,
    ) -> dict[str, tuple[RelevanceLevel, str]]:
        """Build a relevance graph from mentioned files using import analysis.

        Levels:
        - Level 0 (DIRECT, 1.0): directly mentioned files
        - Level 1 (RELATED, 0.85): files imported by Level 0, files that
          import Level 0, test files for Level 0
        - Level 2 (INDIRECT, 0.6): files imported by Level 1
        - Level 3 (CONTEXT_ONLY, 0.3): READMEs, type defs, config files

        Args:
            mentioned_files: Files directly mentioned in the task.
            project_root: Absolute path to the project root.

        Returns:
            A dict mapping file path to ``(RelevanceLevel, reason)`` tuples.
        """
        graph: dict[str, tuple[RelevanceLevel, str]] = {}

        # Level 0: directly mentioned
        for f in mentioned_files:
            graph[f] = (RelevanceLevel.DIRECT, "directly mentioned in request")

        # Level 1: imports and reverse imports of Level 0
        level0_stems = {Path(f).stem for f in mentioned_files}
        level1_files: set[str] = set()

        for f in mentioned_files:
            imports = self._get_file_imports(project_root / f)
            for imp in imports:
                imp_file = self._resolve_import_to_file(imp, project_root)
                if imp_file and imp_file not in graph:
                    graph[imp_file] = (
                        RelevanceLevel.RELATED,
                        f"imported by {f}",
                    )
                    level1_files.add(imp_file)

        # Find files that import Level 0 files
        try:
            py_files = list(project_root.rglob("*.py"))
        except OSError:
            py_files = []

        for py_file in py_files:
            rel = self._relative_path(py_file, project_root)
            if rel in graph:
                continue
            imports = self._get_file_imports(py_file)
            for imp in imports:
                imp_stem = imp.split(".")[-1] if "." in imp else imp
                if imp_stem in level0_stems:
                    graph[rel] = (
                        RelevanceLevel.RELATED,
                        f"imports {imp_stem}",
                    )
                    level1_files.add(rel)
                    break

        # Test files for Level 0
        for f in mentioned_files:
            stem = Path(f).stem
            test_name = f"test_{stem}"
            for py_file in py_files:
                if (
                    py_file.stem == test_name
                    and self._relative_path(py_file, project_root)
                    not in graph
                ):
                    rel = self._relative_path(py_file, project_root)
                    graph[rel] = (
                        RelevanceLevel.RELATED,
                        f"test file for {stem}",
                    )
                    level1_files.add(rel)

        # Level 2: imports of Level 1 files
        for f in level1_files:
            full_path = project_root / f
            imports = self._get_file_imports(full_path)
            for imp in imports:
                imp_file = self._resolve_import_to_file(imp, project_root)
                if imp_file and imp_file not in graph:
                    graph[imp_file] = (
                        RelevanceLevel.INDIRECT,
                        f"imported by {f} (2nd-degree)",
                    )

        # Level 3: READMEs, type defs, config files
        context_patterns = {
            "readme": "README file",
            "types": "type definitions",
            "typing": "type definitions",
            "config": "configuration file",
            "settings": "settings file",
            "constants": "constants file",
            "defaults": "defaults file",
            "schema": "schema definitions",
        }

        for py_file in py_files:
            rel = self._relative_path(py_file, project_root)
            if rel in graph:
                continue
            stem_lower = py_file.stem.lower()
            for pattern, desc in context_patterns.items():
                if pattern in stem_lower:
                    graph[rel] = (
                        RelevanceLevel.CONTEXT_ONLY,
                        desc,
                    )
                    break

        # READMEs (markdown)
        try:
            md_files = list(project_root.rglob("README*.md"))
        except OSError:
            md_files = []

        for md_file in md_files:
            rel = self._relative_path(md_file, project_root)
            if rel not in graph:
                graph[rel] = (
                    RelevanceLevel.CONTEXT_ONLY,
                    "README documentation",
                )

        return graph

    # ------------------------------------------------------------------
    # Public API — efficiency logging
    # ------------------------------------------------------------------

    def log_efficiency(self, record: ContextEfficiencyRecord) -> None:
        """Log a context efficiency record to SQLite.

        Args:
            record: The :class:`ContextEfficiencyRecord` to store.
        """
        if self._db is None:
            logger.debug("log_efficiency_skipped", reason="no database")
            return

        self._db.execute(
            """
            INSERT INTO context_efficiency
                (task_type, files_included, tokens_used,
                 files_excluded, outcome, model_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.task_type,
                record.files_included,
                record.tokens_used,
                record.files_excluded,
                record.outcome,
                record.model_used,
                record.created_at,
            ),
        )
        self._db.commit()

    def get_efficiency_stats(self) -> EfficiencyStats:
        """Query SQLite for aggregate efficiency statistics.

        Returns:
            An :class:`EfficiencyStats` with averages, success rate, and
            estimated savings.
        """
        if self._db is None:
            return EfficiencyStats(
                avg_tokens_used=0.0,
                avg_efficiency_pct=0.0,
                success_rate=0.0,
                total_records=0,
                total_tokens_saved=0,
            )

        row = self._db.fetchone(
            """
            SELECT
                COUNT(*) AS cnt,
                COALESCE(AVG(tokens_used), 0) AS avg_tokens,
                COALESCE(SUM(tokens_used), 0) AS total_tokens,
                COALESCE(
                    SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END)
                    * 1.0 / NULLIF(COUNT(*), 0),
                    0
                ) AS success_rate
            FROM context_efficiency
            """
        )

        if row is None or row["cnt"] == 0:
            return EfficiencyStats(
                avg_tokens_used=0.0,
                avg_efficiency_pct=0.0,
                success_rate=0.0,
                total_records=0,
                total_tokens_saved=0,
            )

        cnt = int(row["cnt"])
        avg_tokens = float(row["avg_tokens"])
        total_tokens = int(row["total_tokens"])
        success_rate = float(row["success_rate"])

        # Estimate savings: naive approach would use full context window
        # (DEFAULT_CONTEXT_WINDOW * 0.5 per task)
        naive_per_task = DEFAULT_CONTEXT_WINDOW * CONTEXT_BUDGET_PCT
        total_naive = int(naive_per_task * cnt)
        total_saved = max(0, total_naive - total_tokens)
        avg_eff = (1.0 - avg_tokens / max(naive_per_task, 1)) * 100.0

        return EfficiencyStats(
            avg_tokens_used=avg_tokens,
            avg_efficiency_pct=max(0.0, avg_eff),
            success_rate=success_rate,
            total_records=cnt,
            total_tokens_saved=total_saved,
        )

    # ------------------------------------------------------------------
    # Public API — display
    # ------------------------------------------------------------------

    @staticmethod
    def generate_context_display(allocation: BudgetAllocation) -> str:
        """Format a :class:`BudgetAllocation` for terminal display.

        Args:
            allocation: The allocation to render.

        Returns:
            A multi-line string with scores, file paths, and token counts.
        """
        header = (
            f"CURRENT CONTEXT ({allocation.tokens_used:,} tokens, "
            f"{allocation.efficiency_pct:.0f}% of budget)"
        )
        separator = "\u2501" * max(len(header), 44)

        lines: list[str] = [header, separator]

        for item in allocation.items_included:
            lines.append(
                f"score {item.relevance:.2f} \u2502 {item.path} "
                f"({item.token_count:,} tokens) - {item.reason}"
            )

        if allocation.items_excluded:
            lines.append("")
            lines.append("EXCLUDED (would exceed budget):")
            for item in allocation.items_excluded[:10]:
                lines.append(
                    f"score {item.relevance:.2f} \u2502 {item.path} "
                    f"({item.token_count:,} tokens) - {item.reason}"
                )
            remaining = len(allocation.items_excluded) - 10
            if remaining > 0:
                lines.append(f"  ... and {remaining} more")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal — parsing / scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_mentioned_files(
        task_description: str,
        available_files: list[str],
    ) -> list[str]:
        """Extract file paths that appear in *task_description*.

        Checks both full paths and stem names.
        """
        mentioned: list[str] = []
        task_lower = task_description.lower()

        for f in available_files:
            stem = Path(f).stem.lower()
            name = Path(f).name.lower()
            if f in task_description or name in task_lower or stem in task_lower:
                mentioned.append(f)

        return mentioned

    @staticmethod
    def _extract_mentioned_functions(task_description: str) -> list[str]:
        """Extract likely function/method names from *task_description*.

        Looks for identifiers in backticks, ``()`` suffixes, and
        snake_case patterns.
        """
        functions: list[str] = []

        # Backtick-wrapped identifiers
        backtick_matches = re.findall(r"`(\w+)`", task_description)
        functions.extend(backtick_matches)

        # Identifiers followed by parentheses
        call_matches = re.findall(r"\b(\w+)\s*\(", task_description)
        functions.extend(call_matches)

        # snake_case identifiers (at least two segments)
        snake_matches = re.findall(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b", task_description)
        functions.extend(snake_matches)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for fn in functions:
            if fn not in seen:
                seen.add(fn)
                unique.append(fn)

        return unique

    def _score_file(
        self,
        file_path: str,
        task_description: str,
        mentioned_files: list[str],
        mentioned_functions: list[str],
        relevance_map: dict[str, tuple[RelevanceLevel, str]],
    ) -> tuple[RelevanceLevel, str]:
        """Score a file based on the relevance graph and task analysis.

        Returns:
            ``(RelevanceLevel, reason)`` tuple.
        """
        # Check relevance graph first
        if file_path in relevance_map:
            return relevance_map[file_path]

        # Direct mention
        if file_path in mentioned_files:
            return RelevanceLevel.DIRECT, "directly mentioned in request"

        # Test file for a mentioned file
        stem = Path(file_path).stem
        if stem.startswith("test_"):
            tested = stem[5:]  # strip "test_"
            for mf in mentioned_files:
                if Path(mf).stem == tested:
                    return RelevanceLevel.RELATED, f"test file for {tested}"

        # Keyword matching against task
        task_lower = task_description.lower()
        path_lower = file_path.lower()
        path_words = set(
            re.findall(
                r"\b\w{3,}\b",
                path_lower.replace("/", " ").replace("_", " "),
            )
        )
        task_words = set(re.findall(r"\b\w{3,}\b", task_lower))
        overlap = path_words & task_words

        if len(overlap) >= 2:
            return RelevanceLevel.INDIRECT, f"keyword match: {', '.join(sorted(overlap)[:3])}"

        if overlap:
            return RelevanceLevel.CONTEXT_ONLY, f"weak keyword match: {next(iter(overlap))}"

        return RelevanceLevel.EXCLUDED, "no relevance detected"

    # ------------------------------------------------------------------
    # Internal — AST import analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _get_file_imports(file_path: Path) -> list[str]:
        """Parse imports from a Python file using AST.

        Returns a list of module names (dotted). Silently returns an empty
        list on any parse error.
        """
        try:
            if not file_path.is_file():
                return []
            source = file_path.read_text(errors="replace")
            tree = ast.parse(source, filename=str(file_path))
        except (OSError, SyntaxError, ValueError):
            return []

        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

        return imports

    @staticmethod
    def _resolve_import_to_file(
        module_name: str,
        project_root: Path,
    ) -> str | None:
        """Attempt to resolve a dotted module name to a project-relative file path.

        Checks ``<root>/<module/as/path>.py`` and
        ``<root>/<module/as/path>/__init__.py``.

        Returns ``None`` if no matching file is found.
        """
        parts = module_name.replace(".", "/")
        candidates = [
            project_root / f"{parts}.py",
            project_root / parts / "__init__.py",
        ]

        # Also try under src/
        candidates.extend([
            project_root / "src" / f"{parts}.py",
            project_root / "src" / parts / "__init__.py",
        ])

        for candidate in candidates:
            if candidate.is_file():
                try:
                    return str(candidate.relative_to(project_root))
                except ValueError:
                    return str(candidate)

        return None

    def _truncate_to_functions(
        self,
        content: str,
        function_names: list[str],
    ) -> str:
        """Truncate *content* to only functions matching *function_names*.

        Falls back to a simple head truncation if AST parsing fails.
        """
        try:
            tree = ast.parse(content)
        except (SyntaxError, ValueError):
            # Fall back to first 200 lines
            lines = content.splitlines()
            return "\n".join(lines[:200])

        matching_ranges: list[tuple[int, int]] = []
        lines = content.splitlines()

        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in function_names
            ):
                start = node.lineno - 1
                end = node.end_lineno if node.end_lineno else start + 20
                # Include 3 lines of context before
                matching_ranges.append((max(0, start - 3), end))

        if not matching_ranges:
            # No matches — return first 200 lines
            return "\n".join(lines[:200])

        # Merge overlapping ranges
        matching_ranges.sort()
        merged: list[tuple[int, int]] = [matching_ranges[0]]
        for start, end in matching_ranges[1:]:
            if start <= merged[-1][1] + 3:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Extract lines
        result_lines: list[str] = []
        for start, end in merged:
            if result_lines:
                result_lines.append("# ... (truncated) ...")
            result_lines.extend(lines[start:end])

        return "\n".join(result_lines)

    @staticmethod
    def _relative_path(full_path: Path, root: Path) -> str:
        """Return the path relative to *root*, or the full path as a string."""
        try:
            return str(full_path.relative_to(root))
        except ValueError:
            return str(full_path)

    def _read_file_safe(self, file_path: str) -> str:
        """Read a file safely, returning empty string on error.

        Files larger than 500 KB are silently skipped.
        """
        try:
            full_path = (
                self._root / file_path
                if not Path(file_path).is_absolute()
                else Path(file_path)
            )
            if full_path.is_file() and full_path.stat().st_size < 500_000:
                return full_path.read_text(errors="replace")
        except OSError:
            pass
        return ""


# ======================================================================
# Module-level utility functions
# ======================================================================


def estimate_tokens(content: str) -> int:
    """Estimate token count for *content*.

    Uses tiktoken if available, otherwise falls back to ``len(content) / 4``.

    Args:
        content: The text to estimate tokens for.

    Returns:
        Estimated token count (always >= 0).
    """
    if not content:
        return 0

    try:
        import tiktoken  # type: ignore[import-untyped]

        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(content))
    except (ImportError, KeyError):
        pass

    return max(1, len(content) // TOKEN_CHARS_RATIO)
