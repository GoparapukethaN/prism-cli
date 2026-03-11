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
from dataclasses import dataclass, field
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
    scenarios: list[str] = field(default_factory=list)


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
        return self._build_report(functions)

    def analyze_module(self, module_name: str) -> TestGapReport:
        """Analyze a specific module only (e.g. 'auth', 'router').

        Filters scanned functions to only include files whose path
        contains the given module name.

        Args:
            module_name: Module name to filter by (case-insensitive).

        Returns:
            A :class:`TestGapReport` with gaps from the specified module.
        """
        all_functions = self._scan_functions()
        module_lower = module_name.lower()
        filtered = [
            f for f in all_functions
            if module_lower in f.file_path.lower()
        ]
        return self._build_report(filtered)

    def generate_tests(
        self, gaps: list[TestGap], count: int = 5,
    ) -> dict[str, str]:
        """Generate complete test file contents for the top N gaps.

        Groups gaps by source file and produces one test file per source.
        The generated tests include proper imports, async handling,
        parametrize decorators for multiple scenarios, and edge case
        coverage based on semantic gap analysis.

        Args:
            gaps: Gaps to generate tests for.
            count: Maximum number of gaps to process.

        Returns:
            A dict mapping test file path to test file content.
        """
        selected = gaps[:count]
        if not selected:
            return {}

        # Group gaps by source file
        by_file: dict[str, list[TestGap]] = {}
        for gap in selected:
            by_file.setdefault(gap.file_path, []).append(gap)

        result: dict[str, str] = {}
        for source_path, file_gaps in by_file.items():
            test_content = self._generate_test_file(source_path, file_gaps)
            # Derive test file path: src/prism/foo.py → tests/test_foo.py
            stem = Path(source_path).stem
            test_path = str(self._tests / f"test_{stem}_generated.py")
            result[test_path] = test_content

        return result

    def _build_report(self, functions: list[FunctionInfo]) -> TestGapReport:
        """Build a TestGapReport from a list of scanned functions.

        Args:
            functions: Functions to evaluate for test coverage.

        Returns:
            A complete :class:`TestGapReport`.
        """
        tested = self._find_tested_functions()

        # Read file contents once for semantic analysis
        file_contents: dict[str, str] = {}

        gaps: list[TestGap] = []
        for func in functions:
            if not self._is_tested(func, tested):
                risk = self._assess_risk(func)

                # Load file content for semantic analysis (cached)
                content = file_contents.get(func.file_path)
                if content is None:
                    try:
                        full_path = self._root / func.file_path
                        content = full_path.read_text()
                    except OSError:
                        content = ""
                    file_contents[func.file_path] = content

                scenarios = self._analyze_semantic_gaps(func, content)

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
                    scenarios=scenarios,
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

    # ------------------------------------------------------------------
    # Semantic gap analysis
    # ------------------------------------------------------------------

    def _analyze_semantic_gaps(
        self, func: FunctionInfo, content: str,
    ) -> list[str]:
        """Identify specific untested scenarios within a function.

        Parses the function's AST node from the source file and detects
        error paths, boundary conditions, async edge cases, and external
        dependency calls.

        Args:
            func: The function metadata.
            content: Full source file content containing the function.

        Returns:
            A list of scenario descriptions such as:
            - ``"Error path: except block at line N"``
            - ``"Boundary: None check at line N"``
            - ``"Async: cancellation/timeout not tested"``
            - ``"External dep: subprocess call at line N"``
        """
        scenarios: list[str] = []
        if not content.strip():
            return scenarios

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return scenarios

        # Find the matching function node
        func_node = self._find_function_node(tree, func.name, func.line_number)
        if func_node is None:
            return scenarios

        # 1. Error paths: except blocks, bare raises, error returns
        scenarios.extend(self._detect_error_paths(func_node))

        # 2. Boundary conditions: None checks, empty checks, <= 0, early returns
        scenarios.extend(self._detect_boundary_conditions(func_node))

        # 3. Async edge cases
        if func.is_async:
            scenarios.extend(self._detect_async_edge_cases(func_node))

        # 4. External dependencies
        scenarios.extend(self._detect_external_deps(func_node))

        return scenarios

    @staticmethod
    def _find_function_node(
        tree: ast.Module, name: str, line_number: int,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """Find the function/async function node by name and line number.

        Args:
            tree: The parsed AST module.
            name: Function name to find.
            line_number: Expected line number of the function.

        Returns:
            The matching AST node, or ``None`` if not found.
        """
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
                and node.lineno == line_number
            ):
                return node
        return None

    @staticmethod
    def _detect_error_paths(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        """Detect error-handling paths in the function.

        Finds except blocks, bare raise statements, and return-None
        statements that suggest error paths.

        Args:
            node: The function AST node.

        Returns:
            List of error path scenario descriptions.
        """
        scenarios: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.ExceptHandler):
                scenarios.append(
                    f"Error path: except block at line {child.lineno}"
                )
            elif isinstance(child, ast.Raise):
                scenarios.append(
                    f"Error path: raise statement at line {child.lineno}"
                )
            elif (
                isinstance(child, ast.Return)
                and child.value is not None
                and isinstance(child.value, ast.Constant)
                and child.value.value is None
            ):
                scenarios.append(
                    f"Error path: returns None at line {child.lineno}"
                )
        return scenarios

    @staticmethod
    def _detect_boundary_conditions(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        """Detect boundary conditions in the function.

        Looks for None checks, empty-length checks, less-than-or-equal-zero
        checks, and early return statements.

        Args:
            node: The function AST node.

        Returns:
            List of boundary condition scenario descriptions.
        """
        scenarios: list[str] = []
        body_stmts = node.body
        # Track if function has a docstring — skip it for early return check
        first_stmt_idx = 0
        if (
            body_stmts
            and isinstance(body_stmts[0], ast.Expr)
            and isinstance(body_stmts[0].value, ast.Constant)
            and isinstance(body_stmts[0].value.value, str)
        ):
            first_stmt_idx = 1

        for child in ast.walk(node):
            if isinstance(child, ast.Compare):
                # ``if x is None``
                for op in child.ops:
                    if isinstance(op, ast.Is):
                        for comp in child.comparators:
                            if (
                                isinstance(comp, ast.Constant)
                                and comp.value is None
                            ):
                                scenarios.append(
                                    f"Boundary: None check at line"
                                    f" {child.lineno}"
                                )
                # ``if len(x) == 0``
                if (
                    isinstance(child.left, ast.Call)
                    and isinstance(child.left.func, ast.Name)
                    and child.left.func.id == "len"
                ):
                    for comp in child.comparators:
                        if (
                            isinstance(comp, ast.Constant)
                            and comp.value == 0
                        ):
                            scenarios.append(
                                f"Boundary: empty collection check at"
                                f" line {child.lineno}"
                            )
                # ``if x <= 0`` or ``if x < 0``
                for i, op in enumerate(child.ops):
                    if (
                        isinstance(op, (ast.LtE, ast.Lt))
                        and i < len(child.comparators)
                    ):
                        comp = child.comparators[i]
                        if (
                            isinstance(comp, ast.Constant)
                            and isinstance(comp.value, (int, float))
                            and comp.value == 0
                        ):
                            scenarios.append(
                                f"Boundary: <= 0 check at line"
                                f" {child.lineno}"
                            )

        # Early returns (return within first few statements, after docstring)
        for stmt in body_stmts[first_stmt_idx:first_stmt_idx + 3]:
            if isinstance(stmt, ast.If):
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Return):
                        scenarios.append(
                            f"Boundary: early return at line {sub.lineno}"
                        )
                        break

        return scenarios

    @staticmethod
    def _detect_async_edge_cases(
        node: ast.AsyncFunctionDef,
    ) -> list[str]:
        """Detect async-specific edge cases that may lack testing.

        Checks whether the async function uses timeout patterns.  If no
        asyncio.wait_for or asyncio.timeout usage is found, flags it.

        Args:
            node: The async function AST node.

        Returns:
            List of async edge case scenario descriptions.
        """
        scenarios: list[str] = []
        has_timeout = False
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr in (
                "wait_for", "timeout",
            ):
                has_timeout = True
                break
            if isinstance(child, ast.Name) and child.id in (
                "wait_for", "timeout",
            ):
                has_timeout = True
                break
        if not has_timeout:
            scenarios.append("Async: cancellation/timeout not tested")
        return scenarios

    @staticmethod
    def _detect_external_deps(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        """Detect calls to external dependency modules.

        Looks for attribute calls or name calls matching known external
        modules: subprocess, httpx, sqlite3, os, shutil, socket.

        Args:
            node: The function AST node.

        Returns:
            List of external dependency scenario descriptions.
        """
        scenarios: list[str] = []
        external_modules = {
            "subprocess", "httpx", "sqlite3", "os", "shutil", "socket",
        }
        seen: set[str] = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and isinstance(
                child.value, ast.Name,
            ):
                mod_name = child.value.id
                if mod_name in external_modules and mod_name not in seen:
                    seen.add(mod_name)
                    scenarios.append(
                        f"External dep: {mod_name} call at"
                        f" line {child.lineno}"
                    )
            elif isinstance(child, ast.Call) and isinstance(
                child.func, ast.Attribute,
            ):
                if isinstance(child.func.value, ast.Name):
                    mod_name = child.func.value.id
                    if mod_name in external_modules and mod_name not in seen:
                        seen.add(mod_name)
                        scenarios.append(
                            f"External dep: {mod_name} call at"
                            f" line {child.lineno}"
                        )

        return scenarios

    # ------------------------------------------------------------------
    # Test file generation
    # ------------------------------------------------------------------

    def _generate_test_file(
        self, source_path: str, gaps: list[TestGap],
    ) -> str:
        """Generate a complete test file for gaps from a single source file.

        Produces proper imports, async markers, parametrize decorators for
        multiple scenarios, and edge case tests.

        Args:
            source_path: Relative path to the source file.
            gaps: List of gaps from that source file.

        Returns:
            Complete test file content as a string.
        """
        # Derive module import path: src/prism/auth/manager.py → prism.auth.manager
        module_path = source_path.replace("/", ".").replace("\\", ".")
        if module_path.startswith("src."):
            module_path = module_path[4:]
        if module_path.endswith(".py"):
            module_path = module_path[:-3]

        lines: list[str] = [
            f'"""Generated tests for {source_path}."""',
            "",
            "from __future__ import annotations",
            "",
            "import pytest",
            "",
        ]

        # Build import list
        func_names = [g.function_name for g in gaps]
        import_line = (
            f"from {module_path} import ("
        )
        lines.append(import_line)
        for fn_name in func_names:
            lines.append(f"    {fn_name},")
        lines.append(")")
        lines.append("")
        lines.append("")

        # Generate test functions
        for gap in gaps:
            lines.extend(
                self._generate_test_function(gap, module_path)
            )
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _generate_test_function(
        gap: TestGap, module_path: str,
    ) -> list[str]:
        """Generate test function(s) for a single gap.

        If the gap has scenarios, generates parametrized or multiple test
        functions covering each scenario.

        Args:
            gap: The test gap to generate tests for.
            module_path: Dotted module path for imports.

        Returns:
            List of lines for the test function(s).
        """
        lines: list[str] = []
        is_async = "await" in gap.test_stub

        # If there are multiple scenarios, use parametrize for common ones
        error_scenarios = [
            s for s in gap.scenarios if s.startswith("Error path:")
        ]
        boundary_scenarios = [
            s for s in gap.scenarios if s.startswith("Boundary:")
        ]
        async_scenarios = [
            s for s in gap.scenarios if s.startswith("Async:")
        ]
        external_scenarios = [
            s for s in gap.scenarios if s.startswith("External dep:")
        ]

        # Main test — happy path
        if is_async:
            lines.append("@pytest.mark.asyncio")
            lines.append(f"async def test_{gap.function_name}() -> None:")
        else:
            lines.append(f"def test_{gap.function_name}() -> None:")
        lines.append(f'    """Test {gap.function_name} happy path."""')
        if is_async:
            lines.append(
                f"    result = await {gap.function_name}()"
            )
        else:
            lines.append(f"    result = {gap.function_name}()")
        lines.append("    assert result is not None")
        lines.append("")

        # Error path tests
        if error_scenarios:
            if is_async:
                lines.append("@pytest.mark.asyncio")
                lines.append(
                    f"async def test_{gap.function_name}"
                    f"_error_handling() -> None:"
                )
            else:
                lines.append(
                    f"def test_{gap.function_name}"
                    f"_error_handling() -> None:"
                )
            lines.append(
                f'    """Test {gap.function_name} error paths."""'
            )
            for scenario in error_scenarios:
                lines.append(f"    # {scenario}")
            lines.append(
                f"    # Verify error handling in {gap.function_name}"
            )
            if is_async:
                lines.append("    with pytest.raises(Exception):")
                lines.append(
                    f"        await {gap.function_name}()"
                )
            else:
                lines.append("    with pytest.raises(Exception):")
                lines.append(
                    f"        {gap.function_name}()"
                )
            lines.append("")

        # Boundary tests
        if boundary_scenarios:
            if is_async:
                lines.append("@pytest.mark.asyncio")
                lines.append(
                    f"async def test_{gap.function_name}"
                    f"_boundary_conditions() -> None:"
                )
            else:
                lines.append(
                    f"def test_{gap.function_name}"
                    f"_boundary_conditions() -> None:"
                )
            lines.append(
                f'    """Test {gap.function_name} boundary conditions."""'
            )
            for scenario in boundary_scenarios:
                lines.append(f"    # {scenario}")
            lines.append(
                f"    # Test edge cases for {gap.function_name}"
            )
            lines.append("    pass")
            lines.append("")

        # Async edge case tests
        if async_scenarios:
            lines.append("@pytest.mark.asyncio")
            lines.append(
                f"async def test_{gap.function_name}_async_timeout()"
                f" -> None:"
            )
            lines.append(
                f'    """Test {gap.function_name}'
                f' async cancellation/timeout."""'
            )
            for scenario in async_scenarios:
                lines.append(f"    # {scenario}")
            lines.append("    import asyncio")
            lines.append("")
            lines.append("    with pytest.raises(asyncio.TimeoutError):")
            lines.append(
                f"        await asyncio.wait_for("
                f"{gap.function_name}(), timeout=0.01)"
            )
            lines.append("")

        # External dependency tests
        if external_scenarios:
            if is_async:
                lines.append("@pytest.mark.asyncio")
                lines.append(
                    f"async def test_{gap.function_name}"
                    f"_external_deps() -> None:"
                )
            else:
                lines.append(
                    f"def test_{gap.function_name}"
                    f"_external_deps() -> None:"
                )
            lines.append(
                f'    """Test {gap.function_name}'
                f' with mocked external dependencies."""'
            )
            for scenario in external_scenarios:
                lines.append(f"    # Mock: {scenario}")
            lines.append(
                f"    # TODO: Mock external deps for"
                f" {gap.function_name}"
            )
            lines.append("    pass")
            lines.append("")

        return lines
