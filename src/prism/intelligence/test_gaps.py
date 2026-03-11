"""Intelligent Test Gap Hunter — find untested code with risk-based prioritization.

Scans the project's source tree via AST, collects every public function and
method, cross-references against the test tree, and reports untested code
ordered by business risk.  Security / auth gaps are ``CRITICAL``, data-layer
gaps are ``HIGH``, core logic is ``MEDIUM``, and utilities are ``LOW``.

Each gap includes a ready-to-paste test stub so developers can start writing
coverage immediately.

Slash-command hook:
    /test-gaps           — full analysis
    /test-gaps critical  — show only critical gaps
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


# ======================================================================
# Risk-level constants
# ======================================================================


class GapRisk:
    """String constants for test-gap risk levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ======================================================================
# Data models
# ======================================================================


@dataclass
class TestGap:
    """A single untested function or method."""

    function_name: str
    file_path: str
    line_number: int
    risk_level: str
    reason: str
    description: str
    test_stub: str
    estimated_effort: str  # "minutes" | "hours"


@dataclass
class TestGapReport:
    """Summary report of test-coverage gaps."""

    total_functions: int
    tested_functions: int
    untested_functions: int
    coverage_percent: float
    gaps: list[TestGap]
    critical_count: int
    high_count: int

    @property
    def has_critical_gaps(self) -> bool:
        """Whether any CRITICAL-risk gaps exist."""
        return self.critical_count > 0


@dataclass
class FunctionInfo:
    """Metadata about a single function or method found in source."""

    name: str
    file_path: str
    line_number: int
    is_async: bool
    has_error_handling: bool
    parameters: list[str]
    decorators: list[str]
    docstring: str | None
    complexity: int  # rough cyclomatic complexity


# ======================================================================
# Hunter
# ======================================================================


class TestGapHunter:
    """Finds untested code and prioritises by business risk.

    Args:
        project_root: Absolute path to the repository root.
        src_dir: Relative name of the source directory (default ``"src"``).
        tests_dir: Relative name of the test directory (default ``"tests"``).
    """

    def __init__(
        self,
        project_root: Path,
        src_dir: str = "src",
        tests_dir: str = "tests",
    ) -> None:
        self._root = project_root.resolve()
        self._src = self._root / src_dir
        self._tests = self._root / tests_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> TestGapReport:
        """Analyze the codebase for test gaps.

        Returns:
            A :class:`TestGapReport` with every gap sorted by risk.
        """
        functions = self._scan_functions()
        tested = self._find_tested_functions()

        gaps: list[TestGap] = []
        for func in functions:
            if not self._is_tested(func, tested):
                risk = self._assess_risk(func)
                gap = TestGap(
                    function_name=func.name,
                    file_path=func.file_path,
                    line_number=func.line_number,
                    risk_level=risk,
                    reason=self._get_risk_reason(func, risk),
                    description=(
                        f"Function '{func.name}' in {func.file_path} "
                        f"has no test coverage"
                    ),
                    test_stub=self._generate_test_stub(func),
                    estimated_effort=self._estimate_effort(func),
                )
                gaps.append(gap)

        # Sort by risk severity
        risk_order = {
            GapRisk.CRITICAL: 0,
            GapRisk.HIGH: 1,
            GapRisk.MEDIUM: 2,
            GapRisk.LOW: 3,
        }
        gaps.sort(key=lambda g: risk_order.get(g.risk_level, 4))

        tested_count = len(functions) - len(gaps)
        total = len(functions)

        return TestGapReport(
            total_functions=total,
            tested_functions=tested_count,
            untested_functions=len(gaps),
            coverage_percent=(tested_count / total * 100) if total > 0 else 100.0,
            gaps=gaps,
            critical_count=sum(1 for g in gaps if g.risk_level == GapRisk.CRITICAL),
            high_count=sum(1 for g in gaps if g.risk_level == GapRisk.HIGH),
        )

    # ------------------------------------------------------------------
    # Source scanning
    # ------------------------------------------------------------------

    def _scan_functions(self) -> list[FunctionInfo]:
        """Scan source tree for all public functions and methods."""
        functions: list[FunctionInfo] = []
        if not self._src.is_dir():
            return functions

        for py_file in self._src.rglob("*.py"):
            rel = str(py_file.relative_to(self._root))

            try:
                content = py_file.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue

                    # Skip private helpers (single underscore), keep dunders
                    if node.name.startswith("_") and not node.name.startswith("__"):
                        continue

                    params = [
                        a.arg for a in node.args.args if a.arg != "self"
                    ]

                    decorators: list[str] = []
                    for d in node.decorator_list:
                        if isinstance(d, ast.Name):
                            decorators.append(d.id)
                        elif isinstance(d, ast.Attribute):
                            decorators.append(d.attr)

                    has_error = any(
                        isinstance(child, ast.ExceptHandler)
                        for child in ast.walk(node)
                    )

                    docstring = ast.get_docstring(node)
                    complexity = self._estimate_complexity(node)

                    functions.append(
                        FunctionInfo(
                            name=node.name,
                            file_path=rel,
                            line_number=node.lineno,
                            is_async=isinstance(node, ast.AsyncFunctionDef),
                            has_error_handling=has_error,
                            parameters=params,
                            decorators=decorators,
                            docstring=docstring,
                            complexity=complexity,
                        )
                    )
            except (SyntaxError, OSError):
                logger.debug("scan_functions_parse_error", file=rel)

        return functions

    # ------------------------------------------------------------------
    # Test detection
    # ------------------------------------------------------------------

    def _find_tested_functions(self) -> set[str]:
        """Collect function names that appear in test files."""
        tested: set[str] = set()
        if not self._tests.is_dir():
            return tested

        for test_file in self._tests.rglob("test_*.py"):
            try:
                content = test_file.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    # Function calls in tests
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Attribute):
                            tested.add(node.func.attr)
                        elif isinstance(node.func, ast.Name):
                            tested.add(node.func.id)
                    # Imported names
                    if isinstance(node, ast.ImportFrom):
                        for alias in node.names or []:
                            tested.add(alias.name)
            except (SyntaxError, OSError):
                pass

        return tested

    def _is_tested(self, func: FunctionInfo, tested: set[str]) -> bool:
        """Check if a function has test coverage."""
        if func.name in tested:
            return True
        # Check for a matching test file by stem
        stem = Path(func.file_path).stem
        for _test_file in self._tests.rglob(f"test_{stem}.py"):
            return True
        return False

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------

    def _assess_risk(self, func: FunctionInfo) -> str:
        """Assess risk level for an untested function."""
        path_lower = func.file_path.lower()
        name_lower = func.name.lower()

        # Critical: security / auth / payment
        critical_patterns = [
            "auth",
            "security",
            "secret",
            "credential",
            "encrypt",
            "decrypt",
            "password",
            "token",
            "permission",
            "validate_api_key",
        ]
        for p in critical_patterns:
            if p in path_lower or p in name_lower:
                return GapRisk.CRITICAL

        # High: data persistence, configuration, billing
        high_patterns = [
            "database",
            "migration",
            "save",
            "delete",
            "update",
            "config",
            "payment",
            "billing",
            "cost",
        ]
        for p in high_patterns:
            if p in path_lower or p in name_lower:
                return GapRisk.HIGH

        # Medium: business logic, routing, tools
        medium_patterns = [
            "router",
            "classify",
            "select",
            "execute",
            "process",
            "handle",
        ]
        for p in medium_patterns:
            if p in path_lower or p in name_lower:
                return GapRisk.MEDIUM

        return GapRisk.LOW

    def _get_risk_reason(self, func: FunctionInfo, risk: str) -> str:
        """Generate human-readable risk reason."""
        reasons = {
            GapRisk.CRITICAL: "Security/authentication code must be tested",
            GapRisk.HIGH: "Data persistence or core configuration needs coverage",
            GapRisk.MEDIUM: "Business logic should have test coverage",
            GapRisk.LOW: "Utility function, lower risk but still worth testing",
        }
        extra = ""
        if func.has_error_handling:
            extra = " (contains error handling paths)"
        if func.complexity > 5:
            extra += f" (cyclomatic complexity: {func.complexity})"
        return reasons.get(risk, "Unknown risk") + extra

    # ------------------------------------------------------------------
    # Test-stub generation
    # ------------------------------------------------------------------

    def _generate_test_stub(self, func: FunctionInfo) -> str:
        """Generate a test stub for an untested function."""
        test_name = f"test_{func.name}"
        params_str = ", ".join(func.parameters) if func.parameters else ""

        if func.is_async:
            return (
                f"async def {test_name}():\n"
                f'    """Test {func.name}."""\n'
                f"    # TODO: Implement test for {func.file_path}:{func.name}\n"
                f"    result = await {func.name}({params_str})\n"
                f"    assert result is not None"
            )

        return (
            f"def {test_name}():\n"
            f'    """Test {func.name}."""\n'
            f"    # TODO: Implement test for {func.file_path}:{func.name}\n"
            f"    result = {func.name}({params_str})\n"
            f"    assert result is not None"
        )

    # ------------------------------------------------------------------
    # Effort estimation
    # ------------------------------------------------------------------

    def _estimate_effort(self, func: FunctionInfo) -> str:
        """Estimate testing effort based on complexity."""
        if func.complexity <= 3 and not func.has_error_handling:
            return "minutes"
        return "hours"

    # ------------------------------------------------------------------
    # Complexity estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_complexity(node: ast.AST) -> int:
        """Estimate cyclomatic complexity of a function AST node."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
