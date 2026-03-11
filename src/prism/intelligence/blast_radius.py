"""Predictive Blast Radius Analysis — impact assessment for planned changes.

Builds import and call graphs via AST parsing, then walks outward from
target files to find every direct and indirect dependency (two levels deep).
Each affected file is scored for risk based on its domain (auth/security =
HIGH, core logic = MEDIUM, utilities = LOW) and whether tests exist.

Results are persisted as JSON reports under ``~/.prism/impact_reports/`` so
they can be reviewed, compared, or fed into CI gates.

Slash-command hook:
    /impact "description of planned change"
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


# ======================================================================
# Risk-level constants
# ======================================================================


class RiskLevel:
    """String constants for blast-radius risk levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ======================================================================
# Data models
# ======================================================================


@dataclass
class AffectedFile:
    """A single file affected by a planned change."""

    path: str
    risk_level: str
    reason: str
    affected_functions: list[str]
    has_tests: bool
    test_files: list[str]
    depth: int  # 0 = direct target, 1 = direct caller/importer, 2 = indirect


@dataclass
class ImpactReport:
    """Full blast-radius report for a planned change."""

    description: str
    risk_score: int  # 0-100
    affected_files: list[AffectedFile]
    missing_tests: list[str]
    recommended_test_order: list[str]
    execution_order: list[str]
    estimated_complexity: str  # "trivial" | "simple" | "moderate" | "complex"
    critical_paths: list[str]
    created_at: str

    @property
    def high_risk_count(self) -> int:
        """Number of HIGH-risk affected files."""
        return sum(1 for f in self.affected_files if f.risk_level == RiskLevel.HIGH)

    @property
    def file_count(self) -> int:
        """Total number of affected files."""
        return len(self.affected_files)


# ======================================================================
# Analyzer
# ======================================================================


class BlastRadiusAnalyzer:
    """Analyzes the blast radius of planned code changes.

    Args:
        project_root: Absolute path to the repository root.
        src_dir: Relative name of the source directory (default ``"src"``).
        reports_dir: Where to persist JSON reports (default ``~/.prism/impact_reports/``).
    """

    def __init__(
        self,
        project_root: Path,
        src_dir: str = "src",
        reports_dir: Path | None = None,
    ) -> None:
        self._root = project_root.resolve()
        self._src = self._root / src_dir
        self._tests = self._root / "tests"
        self._reports_dir = reports_dir or Path.home() / ".prism" / "impact_reports"
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._call_graph: dict[str, set[str]] = {}
        self._import_graph: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        description: str,
        target_files: list[str] | None = None,
    ) -> ImpactReport:
        """Analyze blast radius for a planned change.

        Args:
            description: Human-readable description of the change.
            target_files: Explicit list of relative paths; inferred from
                *description* when ``None``.

        Returns:
            A complete :class:`ImpactReport`.
        """
        # Build graphs
        self._build_import_graph()
        self._build_call_graph()

        # Find target files from description if not provided
        if not target_files:
            target_files = self._find_targets(description)

        # Find affected files at each depth
        affected = self._find_affected(target_files)

        # Score risk, locate tests
        for af in affected:
            af.risk_level = self._assess_risk(af)
            af.test_files = self._find_tests_for(af.path)
            af.has_tests = bool(af.test_files)

        # Identify missing tests
        missing_tests = [af.path for af in affected if not af.has_tests]

        # Calculate overall risk
        risk_score = self._calculate_risk_score(affected)

        # Determine complexity
        complexity = self._estimate_complexity(affected)

        # Find critical paths
        critical = [af.path for af in affected if af.risk_level == RiskLevel.HIGH]

        # Build execution order (test-covered first, then outward)
        exec_order = self._build_execution_order(affected)
        test_order = [f for f in exec_order if self._find_tests_for(f)]

        report = ImpactReport(
            description=description,
            risk_score=risk_score,
            affected_files=affected,
            missing_tests=missing_tests,
            recommended_test_order=test_order,
            execution_order=exec_order,
            estimated_complexity=complexity,
            critical_paths=critical,
            created_at=datetime.now(UTC).isoformat(),
        )

        self._save_report(report)
        return report

    def list_reports(self) -> list[Path]:
        """List saved impact reports, newest first."""
        return sorted(self._reports_dir.glob("impact_*.json"), reverse=True)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_import_graph(self) -> None:
        """Build import dependency graph from source files."""
        self._import_graph.clear()
        if not self._src.is_dir():
            return

        for py_file in self._src.rglob("*.py"):
            rel = str(py_file.relative_to(self._root))
            self._import_graph[rel] = set()

            try:
                tree = ast.parse(py_file.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        # Convert dotted module path to possible file paths
                        parts = node.module.replace(".", "/")
                        candidates = [
                            f"src/{parts}.py",
                            f"src/{parts}/__init__.py",
                        ]
                        for c in candidates:
                            if (self._root / c).is_file():
                                self._import_graph[rel].add(c)
            except (SyntaxError, OSError):
                logger.debug("import_graph_parse_error", file=rel)

    def _build_call_graph(self) -> None:
        """Build a simplified call graph from function references.

        For every source file, record which *other* source files define a
        function that is called by name inside the file.  Attribute-style
        calls (``obj.method()``) are intentionally skipped because the
        receiver type cannot be resolved without a full type checker.
        """
        self._call_graph.clear()
        if not self._src.is_dir():
            return

        # Pre-build a map: function_name -> set of files defining it
        func_definitions: dict[str, set[str]] = {}
        all_files: list[tuple[str, Path]] = []
        for py_file in self._src.rglob("*.py"):
            rel = str(py_file.relative_to(self._root))
            all_files.append((rel, py_file))
            try:
                tree = ast.parse(py_file.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_definitions.setdefault(node.name, set()).add(rel)
            except (SyntaxError, OSError):
                pass

        for rel, py_file in all_files:
            self._call_graph[rel] = set()
            try:
                tree = ast.parse(py_file.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        for definer in func_definitions.get(func_name, set()):
                            if definer != rel:
                                self._call_graph[rel].add(definer)
            except (SyntaxError, OSError):
                pass

    # ------------------------------------------------------------------
    # Target discovery
    # ------------------------------------------------------------------

    def _find_targets(self, description: str) -> list[str]:
        """Find target files whose stem matches keywords in *description*."""
        keywords = [w.lower() for w in description.split() if len(w) > 3]
        targets: list[str] = []

        if not self._src.is_dir():
            return targets

        for py_file in self._src.rglob("*.py"):
            rel = str(py_file.relative_to(self._root))
            name_lower = py_file.stem.lower()

            for kw in keywords:
                if kw in name_lower:
                    targets.append(rel)
                    break

        return targets[:10]

    # ------------------------------------------------------------------
    # Affected-file expansion
    # ------------------------------------------------------------------

    def _find_affected(self, targets: list[str]) -> list[AffectedFile]:
        """Find all affected files at depths 0, 1, and 2."""
        affected: dict[str, AffectedFile] = {}

        # Depth 0: direct targets
        for t in targets:
            affected[t] = AffectedFile(
                path=t,
                risk_level=RiskLevel.MEDIUM,
                reason="Direct target",
                affected_functions=[],
                has_tests=False,
                test_files=[],
                depth=0,
            )

        # Depth 1: direct importers / callers
        for file_path, deps in self._import_graph.items():
            for t in targets:
                if t in deps and file_path not in affected:
                    affected[file_path] = AffectedFile(
                        path=file_path,
                        risk_level=RiskLevel.MEDIUM,
                        reason=f"Imports {t}",
                        affected_functions=[],
                        has_tests=False,
                        test_files=[],
                        depth=1,
                    )

        # Depth 2: indirect (importers of depth-1 files)
        depth1_files = [f for f, af in affected.items() if af.depth == 1]
        for file_path, deps in self._import_graph.items():
            for d1 in depth1_files:
                if d1 in deps and file_path not in affected:
                    affected[file_path] = AffectedFile(
                        path=file_path,
                        risk_level=RiskLevel.LOW,
                        reason=f"Indirect dependency via {d1}",
                        affected_functions=[],
                        has_tests=False,
                        test_files=[],
                        depth=2,
                    )

        return list(affected.values())

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------

    def _assess_risk(self, af: AffectedFile) -> str:
        """Assess risk level for an affected file."""
        path_lower = af.path.lower()

        # High risk patterns
        high_patterns = [
            "auth",
            "security",
            "secret",
            "credential",
            "payment",
            "billing",
            "migration",
        ]
        for p in high_patterns:
            if p in path_lower:
                return RiskLevel.HIGH

        # Medium risk patterns
        medium_patterns = ["router", "completion", "provider", "database", "config"]
        for p in medium_patterns:
            if p in path_lower:
                return RiskLevel.MEDIUM

        # Depth-based fallback
        if af.depth <= 1:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def _find_tests_for(self, file_path: str) -> list[str]:
        """Find test files covering a source file."""
        if not self._tests.is_dir():
            return []

        stem = Path(file_path).stem
        tests: list[str] = []

        for test_file in self._tests.rglob(f"test_{stem}.py"):
            tests.append(str(test_file.relative_to(self._root)))

        # Also look for partial matches
        for test_file in self._tests.rglob(f"test_*{stem}*.py"):
            rel = str(test_file.relative_to(self._root))
            if rel not in tests:
                tests.append(rel)

        return tests

    # ------------------------------------------------------------------
    # Scoring & estimation
    # ------------------------------------------------------------------

    def _calculate_risk_score(self, affected: list[AffectedFile]) -> int:
        """Calculate overall risk score 0-100."""
        if not affected:
            return 0

        weights = {RiskLevel.HIGH: 30, RiskLevel.MEDIUM: 15, RiskLevel.LOW: 5}
        score = sum(weights.get(af.risk_level, 5) for af in affected)

        # Untested files add risk
        untested = sum(1 for af in affected if not af.has_tests)
        score += untested * 10

        return min(score, 100)

    def _estimate_complexity(self, affected: list[AffectedFile]) -> str:
        """Estimate change complexity from affected-file count and risk."""
        count = len(affected)
        high_count = sum(1 for af in affected if af.risk_level == RiskLevel.HIGH)

        if count <= 2 and high_count == 0:
            return "trivial"
        if count <= 5 and high_count <= 1:
            return "simple"
        if count <= 15 or high_count <= 3:
            return "moderate"
        return "complex"

    # ------------------------------------------------------------------
    # Execution ordering
    # ------------------------------------------------------------------

    def _build_execution_order(self, affected: list[AffectedFile]) -> list[str]:
        """Build safe execution order (leaves first, then inward)."""
        return sorted(
            [af.path for af in affected],
            key=lambda p: next(
                (af.depth for af in affected if af.path == p), 0
            ),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_report(self, report: ImpactReport) -> Path:
        """Save impact report to disk as JSON."""
        timestamp = report.created_at[:19].replace(":", "-")
        filename = f"impact_{timestamp}_{report.risk_score}.json"
        path = self._reports_dir / filename
        path.write_text(json.dumps(asdict(report), indent=2))
        logger.info("impact_report_saved", path=str(path))
        return path
