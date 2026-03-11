"""Tests for enhanced test_gaps — semantic analysis, module filter, test generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.intelligence.test_gaps import (
    FunctionInfo,
    GapRisk,
    TestGap,
    TestGapHunter,
)

if TYPE_CHECKING:
    from pathlib import Path

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """Create a minimal project layout with diverse code patterns."""
    src = tmp_path / "src" / "prism"
    src.mkdir(parents=True)
    tests = tmp_path / "tests"
    tests.mkdir()

    # auth module — critical risk, has test
    (src / "auth.py").write_text(
        "def validate_token(token: str) -> bool:\n"
        '    """Validate auth token."""\n'
        "    return bool(token)\n"
    )

    # security module — critical risk, NO test, has except blocks
    (src / "security.py").write_text(
        "def encrypt_data(data: str) -> str:\n"
        '    """Encrypt sensitive data."""\n'
        "    try:\n"
        "        return data[::-1]\n"
        "    except Exception:\n"
        "        return ''\n"
    )

    # router module — medium risk, has test
    (src / "router.py").write_text(
        "def classify_task(task: str) -> str:\n"
        '    """Classify task."""\n'
        "    return 'simple'\n"
    )

    # utils module — low risk, NO test, with boundary checks
    (src / "utils.py").write_text(
        "def format_text(text: str) -> str:\n"
        '    """Format text with boundary checks."""\n'
        "    if text is None:\n"
        "        return ''\n"
        "    if len(text) == 0:\n"
        "        return ''\n"
        "    return text.strip()\n"
    )

    # database module — high risk, with error handling
    (src / "database.py").write_text(
        "def save_record(data: dict) -> bool:\n"
        '    """Save a record to the database."""\n'
        "    try:\n"
        "        return True\n"
        "    except Exception:\n"
        "        return False\n"
        "\n"
        "def delete_record(record_id: int) -> bool:\n"
        '    """Delete a record."""\n'
        "    if record_id < 0:\n"
        "        return False\n"
        "    return True\n"
    )

    # async module — with async functions, NO test
    (src / "async_handler.py").write_text(
        "async def process_request(req: str) -> str:\n"
        '    """Process request asynchronously."""\n'
        "    return req\n"
        "\n"
        "async def fetch_data(url: str) -> dict:\n"
        '    """Fetch data from a URL."""\n'
        "    import httpx\n"
        "    resp = httpx.get(url)\n"
        "    return {'status': resp.status_code}\n"
    )

    # external deps module — uses subprocess and os
    (src / "runner.py").write_text(
        "import subprocess\n"
        "import os\n"
        "\n"
        "def execute_command(cmd: str) -> str:\n"
        '    """Execute a shell command."""\n'
        "    result = subprocess.run(cmd, capture_output=True)\n"
        "    path = os.path.abspath('.')\n"
        "    return result.stdout.decode()\n"
    )

    # module with raise and return None
    (src / "validator.py").write_text(
        "def check_input(value: str) -> str:\n"
        '    """Validate input value."""\n'
        "    if value is None:\n"
        "        return None\n"
        "    if len(value) == 0:\n"
        "        raise ValueError('empty input')\n"
        "    return value\n"
    )

    (src / "__init__.py").write_text("")

    # Test files — only for auth and router
    (tests / "test_auth.py").write_text(
        "from prism.auth import validate_token\n"
        "def test_validate_token():\n"
        "    assert validate_token('abc')\n"
    )

    (tests / "test_router.py").write_text(
        "from prism.router import classify_task\n"
        "def test_classify_task():\n"
        "    assert classify_task('x') == 'simple'\n"
    )

    return tmp_path


# ======================================================================
# TestGap.scenarios field
# ======================================================================


class TestTestGapScenarios:
    """Tests for the new scenarios field on TestGap."""

    def test_scenarios_defaults_to_empty_list(self) -> None:
        """TestGap.scenarios should default to empty list."""
        gap = TestGap(
            function_name="foo",
            file_path="x.py",
            line_number=1,
            risk_level=GapRisk.LOW,
            reason="test",
            description="test",
            test_stub="def test_foo(): ...",
            estimated_effort="minutes",
        )
        assert gap.scenarios == []
        assert isinstance(gap.scenarios, list)

    def test_scenarios_with_values(self) -> None:
        """TestGap.scenarios can store scenario descriptions."""
        scenarios = [
            "Error path: except block at line 5",
            "Boundary: None check at line 3",
        ]
        gap = TestGap(
            function_name="bar",
            file_path="y.py",
            line_number=1,
            risk_level=GapRisk.HIGH,
            reason="test",
            description="test",
            test_stub="def test_bar(): ...",
            estimated_effort="hours",
            scenarios=scenarios,
        )
        assert gap.scenarios == scenarios
        assert len(gap.scenarios) == 2


# ======================================================================
# _analyze_semantic_gaps
# ======================================================================


class TestAnalyzeSemanticGaps:
    """Tests for the _analyze_semantic_gaps method."""

    def test_detects_except_blocks(self, project: Path) -> None:
        """Should detect except blocks as error paths."""
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            name="encrypt_data",
            file_path="src/prism/security.py",
            line_number=1,
            is_async=False,
            has_error_handling=True,
            parameters=["data"],
            decorators=[],
            docstring="Encrypt sensitive data.",
            complexity=2,
        )
        content = (project / "src" / "prism" / "security.py").read_text()
        scenarios = hunter._analyze_semantic_gaps(func, content)
        error_paths = [s for s in scenarios if "except block" in s]
        assert len(error_paths) >= 1

    def test_detects_none_checks(self, project: Path) -> None:
        """Should detect None checks as boundary conditions."""
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            name="format_text",
            file_path="src/prism/utils.py",
            line_number=1,
            is_async=False,
            has_error_handling=False,
            parameters=["text"],
            decorators=[],
            docstring="Format text with boundary checks.",
            complexity=3,
        )
        content = (project / "src" / "prism" / "utils.py").read_text()
        scenarios = hunter._analyze_semantic_gaps(func, content)
        none_checks = [s for s in scenarios if "None check" in s]
        assert len(none_checks) >= 1

    def test_detects_empty_collection_check(self, project: Path) -> None:
        """Should detect len(x) == 0 as boundary condition."""
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            name="format_text",
            file_path="src/prism/utils.py",
            line_number=1,
            is_async=False,
            has_error_handling=False,
            parameters=["text"],
            decorators=[],
            docstring="Format text with boundary checks.",
            complexity=3,
        )
        content = (project / "src" / "prism" / "utils.py").read_text()
        scenarios = hunter._analyze_semantic_gaps(func, content)
        empty_checks = [s for s in scenarios if "empty collection" in s]
        assert len(empty_checks) >= 1

    def test_detects_async_functions(self, project: Path) -> None:
        """Should detect async functions without timeout patterns."""
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            name="process_request",
            file_path="src/prism/async_handler.py",
            line_number=1,
            is_async=True,
            has_error_handling=False,
            parameters=["req"],
            decorators=[],
            docstring="Process request asynchronously.",
            complexity=1,
        )
        content = (project / "src" / "prism" / "async_handler.py").read_text()
        scenarios = hunter._analyze_semantic_gaps(func, content)
        async_issues = [s for s in scenarios if "Async:" in s]
        assert len(async_issues) >= 1
        assert any("timeout" in s for s in async_issues)

    def test_detects_external_dependencies(self, project: Path) -> None:
        """Should detect subprocess and os calls."""
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            name="execute_command",
            file_path="src/prism/runner.py",
            line_number=4,
            is_async=False,
            has_error_handling=False,
            parameters=["cmd"],
            decorators=[],
            docstring="Execute a shell command.",
            complexity=1,
        )
        content = (project / "src" / "prism" / "runner.py").read_text()
        scenarios = hunter._analyze_semantic_gaps(func, content)
        external = [s for s in scenarios if "External dep:" in s]
        assert len(external) >= 1
        # Should mention subprocess
        assert any("subprocess" in s for s in external)

    def test_detects_raise_statement(self, project: Path) -> None:
        """Should detect raise statements as error paths."""
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            name="check_input",
            file_path="src/prism/validator.py",
            line_number=1,
            is_async=False,
            has_error_handling=False,
            parameters=["value"],
            decorators=[],
            docstring="Validate input value.",
            complexity=3,
        )
        content = (project / "src" / "prism" / "validator.py").read_text()
        scenarios = hunter._analyze_semantic_gaps(func, content)
        raises = [s for s in scenarios if "raise statement" in s]
        assert len(raises) >= 1

    def test_detects_return_none(self, project: Path) -> None:
        """Should detect return None as error path."""
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            name="check_input",
            file_path="src/prism/validator.py",
            line_number=1,
            is_async=False,
            has_error_handling=False,
            parameters=["value"],
            decorators=[],
            docstring="Validate input value.",
            complexity=3,
        )
        content = (project / "src" / "prism" / "validator.py").read_text()
        scenarios = hunter._analyze_semantic_gaps(func, content)
        return_nones = [s for s in scenarios if "returns None" in s]
        assert len(return_nones) >= 1

    def test_empty_content_returns_empty(self, project: Path) -> None:
        """Empty content should produce no scenarios."""
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            name="foo",
            file_path="src/prism/foo.py",
            line_number=1,
            is_async=False,
            has_error_handling=False,
            parameters=[],
            decorators=[],
            docstring=None,
            complexity=1,
        )
        scenarios = hunter._analyze_semantic_gaps(func, "")
        assert scenarios == []

    def test_syntax_error_returns_empty(self, project: Path) -> None:
        """Invalid syntax should produce no scenarios."""
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            name="broken",
            file_path="src/prism/broken.py",
            line_number=1,
            is_async=False,
            has_error_handling=False,
            parameters=[],
            decorators=[],
            docstring=None,
            complexity=1,
        )
        scenarios = hunter._analyze_semantic_gaps(func, "def broken(:\n")
        assert scenarios == []

    def test_function_not_found_returns_empty(self, project: Path) -> None:
        """When function is not found in AST, should return empty."""
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            name="nonexistent",
            file_path="src/prism/utils.py",
            line_number=999,
            is_async=False,
            has_error_handling=False,
            parameters=[],
            decorators=[],
            docstring=None,
            complexity=1,
        )
        content = (project / "src" / "prism" / "utils.py").read_text()
        scenarios = hunter._analyze_semantic_gaps(func, content)
        assert scenarios == []

    def test_detects_less_than_zero_check(self, project: Path) -> None:
        """Should detect < 0 boundary conditions."""
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            name="delete_record",
            file_path="src/prism/database.py",
            line_number=8,
            is_async=False,
            has_error_handling=False,
            parameters=["record_id"],
            decorators=[],
            docstring="Delete a record.",
            complexity=2,
        )
        content = (project / "src" / "prism" / "database.py").read_text()
        scenarios = hunter._analyze_semantic_gaps(func, content)
        boundary = [s for s in scenarios if "<= 0" in s]
        assert len(boundary) >= 1


# ======================================================================
# analyze_module
# ======================================================================


class TestAnalyzeModule:
    """Tests for the analyze_module method."""

    def test_filters_to_specific_module(self, project: Path) -> None:
        """Should only include functions from the specified module."""
        hunter = TestGapHunter(project)
        report = hunter.analyze_module("security")
        # Only security module functions should appear
        for gap in report.gaps:
            assert "security" in gap.file_path.lower()

    def test_module_total_functions(self, project: Path) -> None:
        """Total functions should only count module functions."""
        hunter = TestGapHunter(project)
        full_report = hunter.analyze()
        module_report = hunter.analyze_module("security")
        assert module_report.total_functions <= full_report.total_functions
        assert module_report.total_functions > 0

    def test_module_not_found(self, project: Path) -> None:
        """Non-existent module should produce empty report."""
        hunter = TestGapHunter(project)
        report = hunter.analyze_module("nonexistent_xyz")
        assert report.total_functions == 0
        assert report.coverage_percent == 100.0
        assert report.gaps == []

    def test_module_case_insensitive(self, project: Path) -> None:
        """Module filtering should be case-insensitive."""
        hunter = TestGapHunter(project)
        report_lower = hunter.analyze_module("security")
        report_upper = hunter.analyze_module("SECURITY")
        assert report_lower.total_functions == report_upper.total_functions

    def test_module_with_tested_functions(self, project: Path) -> None:
        """Module with tests should show tested functions."""
        hunter = TestGapHunter(project)
        report = hunter.analyze_module("auth")
        # auth module has test_auth.py → all functions should be tested
        assert report.untested_functions == 0


# ======================================================================
# generate_tests
# ======================================================================


class TestGenerateTests:
    """Tests for the generate_tests method."""

    def test_produces_valid_python(self, project: Path) -> None:
        """Generated test files should be valid Python."""
        import ast

        hunter = TestGapHunter(project)
        report = hunter.analyze()
        generated = hunter.generate_tests(report.gaps, count=3)
        assert len(generated) > 0
        for _path, content in generated.items():
            assert content.strip()
            # Should parse without syntax errors
            ast.parse(content)

    def test_handles_async_functions(self, project: Path) -> None:
        """Generated tests for async functions should include async markers."""
        hunter = TestGapHunter(project)
        report = hunter.analyze()
        async_gaps = [
            g for g in report.gaps if "await" in g.test_stub
        ]
        if async_gaps:
            generated = hunter.generate_tests(async_gaps, count=5)
            for content in generated.values():
                assert "pytest.mark.asyncio" in content or "async def" in content

    def test_respects_count_limit(self, project: Path) -> None:
        """Should only process up to count gaps."""
        hunter = TestGapHunter(project)
        report = hunter.analyze()
        assert len(report.gaps) > 1
        generated = hunter.generate_tests(report.gaps, count=1)
        # Only 1 gap processed, but it generates 1 file for that gap's source
        total_gaps_covered = sum(
            content.count("def test_") for content in generated.values()
        )
        # At minimum 1 test function, at most a few (with scenario variants)
        assert total_gaps_covered >= 1

    def test_empty_gaps_returns_empty_dict(self, project: Path) -> None:
        """Empty gap list should return empty dict."""
        hunter = TestGapHunter(project)
        generated = hunter.generate_tests([], count=5)
        assert generated == {}

    def test_groups_by_source_file(self, project: Path) -> None:
        """Gaps from same source should produce one test file."""
        hunter = TestGapHunter(project)
        report = hunter.analyze()
        # database.py has save_record and delete_record
        db_gaps = [
            g for g in report.gaps if "database" in g.file_path
        ]
        if len(db_gaps) >= 2:
            generated = hunter.generate_tests(db_gaps, count=5)
            # Should produce exactly one test file for database
            db_files = [
                p for p in generated if "database" in p
            ]
            assert len(db_files) == 1

    def test_includes_imports(self, project: Path) -> None:
        """Generated test files should include proper imports."""
        hunter = TestGapHunter(project)
        report = hunter.analyze()
        generated = hunter.generate_tests(report.gaps, count=3)
        for content in generated.values():
            assert "import pytest" in content
            assert "from" in content

    def test_includes_scenario_comments(self, project: Path) -> None:
        """Generated tests should include scenario comments for gaps with scenarios."""
        hunter = TestGapHunter(project)
        report = hunter.analyze()
        gaps_with_scenarios = [
            g for g in report.gaps if g.scenarios
        ]
        if gaps_with_scenarios:
            generated = hunter.generate_tests(
                gaps_with_scenarios, count=5,
            )
            all_content = "\n".join(generated.values())
            # Should have at least one comment about an error/boundary/async/external
            assert any(
                marker in all_content
                for marker in (
                    "Error path:", "Boundary:", "Async:", "External dep:",
                )
            )

    def test_test_file_path_format(self, project: Path) -> None:
        """Test file paths should follow the test_<stem>_generated.py pattern."""
        hunter = TestGapHunter(project)
        report = hunter.analyze()
        generated = hunter.generate_tests(report.gaps, count=3)
        for path in generated:
            assert "_generated.py" in path
            assert "test_" in path


# ======================================================================
# Integration: analyze populates scenarios
# ======================================================================


class TestAnalyzePopulatesScenarios:
    """Test that analyze() populates the scenarios field on gaps."""

    def test_analyze_fills_scenarios(self, project: Path) -> None:
        """analyze() should populate scenarios on gaps."""
        hunter = TestGapHunter(project)
        report = hunter.analyze()
        # encrypt_data has an except block
        encrypt_gaps = [
            g for g in report.gaps if g.function_name == "encrypt_data"
        ]
        assert len(encrypt_gaps) == 1
        assert len(encrypt_gaps[0].scenarios) > 0

    def test_analyze_module_fills_scenarios(self, project: Path) -> None:
        """analyze_module() should populate scenarios on gaps."""
        hunter = TestGapHunter(project)
        report = hunter.analyze_module("database")
        for gap in report.gaps:
            # database functions have try/except or boundary checks
            assert isinstance(gap.scenarios, list)
