"""Tests for prism.intelligence.deps — Autonomous Dependency Health Monitor."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from prism.intelligence.deps import (
    DependencyInfo,
    DependencyMonitor,
    DepsReport,
    MigrationComplexity,
    Vulnerability,
    VulnerabilityReport,
    VulnerabilitySeverity,
)

# ======================================================================
# TestMigrationComplexity
# ======================================================================


class TestMigrationComplexity:
    """Tests for the MigrationComplexity enum."""

    def test_trivial_value(self) -> None:
        assert MigrationComplexity.TRIVIAL.value == "trivial"

    def test_simple_value(self) -> None:
        assert MigrationComplexity.SIMPLE.value == "simple"

    def test_moderate_value(self) -> None:
        assert MigrationComplexity.MODERATE.value == "moderate"

    def test_complex_value(self) -> None:
        assert MigrationComplexity.COMPLEX.value == "complex"

    def test_all_values(self) -> None:
        values = {m.value for m in MigrationComplexity}
        assert values == {"trivial", "simple", "moderate", "complex"}

    def test_from_string(self) -> None:
        assert MigrationComplexity("trivial") is MigrationComplexity.TRIVIAL


# ======================================================================
# TestVulnerabilitySeverity
# ======================================================================


class TestVulnerabilitySeverity:
    """Tests for the VulnerabilitySeverity enum."""

    def test_critical_value(self) -> None:
        assert VulnerabilitySeverity.CRITICAL.value == "critical"

    def test_high_value(self) -> None:
        assert VulnerabilitySeverity.HIGH.value == "high"

    def test_medium_value(self) -> None:
        assert VulnerabilitySeverity.MEDIUM.value == "medium"

    def test_low_value(self) -> None:
        assert VulnerabilitySeverity.LOW.value == "low"


# ======================================================================
# TestDependencyInfo
# ======================================================================


class TestDependencyInfo:
    """Tests for the DependencyInfo dataclass."""

    def test_fields(self) -> None:
        dep = DependencyInfo(
            name="requests",
            current_version="2.28.0",
            latest_version="2.31.0",
            is_outdated=True,
            age_days=180,
            ecosystem="python",
            source_file="pyproject.toml",
            usages=5,
        )
        assert dep.name == "requests"
        assert dep.current_version == "2.28.0"
        assert dep.latest_version == "2.31.0"
        assert dep.is_outdated is True
        assert dep.age_days == 180
        assert dep.ecosystem == "python"
        assert dep.source_file == "pyproject.toml"
        assert dep.usages == 5

    def test_default_migration_complexity(self) -> None:
        dep = DependencyInfo(
            name="flask",
            current_version="2.0",
            latest_version="3.0",
            is_outdated=True,
            age_days=365,
            ecosystem="python",
            source_file="requirements.txt",
            usages=10,
        )
        assert dep.migration_complexity is MigrationComplexity.TRIVIAL

    def test_custom_migration_complexity(self) -> None:
        dep = DependencyInfo(
            name="flask",
            current_version="2.0",
            latest_version="3.0",
            is_outdated=True,
            age_days=365,
            ecosystem="python",
            source_file="requirements.txt",
            usages=10,
            migration_complexity=MigrationComplexity.COMPLEX,
        )
        assert dep.migration_complexity is MigrationComplexity.COMPLEX


# ======================================================================
# TestVulnerability
# ======================================================================


class TestVulnerability:
    """Tests for the Vulnerability dataclass."""

    def test_fields(self) -> None:
        vuln = Vulnerability(
            package="django",
            severity=VulnerabilitySeverity.CRITICAL,
            cve_id="CVE-2024-1234",
            description="SQL injection in ORM",
            fixed_version="4.2.8",
            current_version="4.2.5",
        )
        assert vuln.package == "django"
        assert vuln.severity is VulnerabilitySeverity.CRITICAL
        assert vuln.cve_id == "CVE-2024-1234"
        assert vuln.description == "SQL injection in ORM"
        assert vuln.fixed_version == "4.2.8"
        assert vuln.current_version == "4.2.5"


# ======================================================================
# TestDepsReport
# ======================================================================


class TestDepsReport:
    """Tests for the DepsReport dataclass."""

    def test_fields(self) -> None:
        report = DepsReport(
            total_deps=10,
            outdated=3,
            vulnerable=1,
            unused=2,
            dependencies=[],
            vulnerabilities=[],
            unused_deps=["foo", "bar"],
            generated_at="2025-01-01T00:00:00+00:00",
            ecosystem="python",
        )
        assert report.total_deps == 10
        assert report.outdated == 3
        assert report.vulnerable == 1
        assert report.unused == 2
        assert report.dependencies == []
        assert report.vulnerabilities == []
        assert report.unused_deps == ["foo", "bar"]
        assert report.ecosystem == "python"

    def test_empty_report(self) -> None:
        report = DepsReport(
            total_deps=0,
            outdated=0,
            vulnerable=0,
            unused=0,
            dependencies=[],
            vulnerabilities=[],
            unused_deps=[],
            generated_at="2025-01-01T00:00:00+00:00",
            ecosystem="all",
        )
        assert report.total_deps == 0
        assert report.ecosystem == "all"


# ======================================================================
# TestDependencyMonitor
# ======================================================================


class TestDependencyMonitor:
    """Tests for the DependencyMonitor class."""

    # --- Initialization ---

    def test_init_valid_dir(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        assert monitor._root == tmp_path.resolve()

    def test_init_invalid_dir(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nope"
        with pytest.raises(ValueError, match="not a directory"):
            DependencyMonitor(nonexistent)

    def test_init_file_not_dir(self, tmp_path: Path) -> None:
        f = tmp_path / "afile.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="not a directory"):
            DependencyMonitor(f)

    # --- Ecosystem detection ---

    def test_detect_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'foo'\n")
        monitor = DependencyMonitor(tmp_path)
        assert "pyproject.toml" in monitor.detected_ecosystems
        assert monitor.detected_ecosystems["pyproject.toml"] == "python"

    def test_detect_requirements(self, tmp_path: Path) -> None:
        (tmp_path / "requirements.txt").write_text("requests>=2.28\n")
        monitor = DependencyMonitor(tmp_path)
        assert "requirements.txt" in monitor.detected_ecosystems

    def test_detect_package_json(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text('{"name": "test"}')
        monitor = DependencyMonitor(tmp_path)
        assert "package.json" in monitor.detected_ecosystems
        assert monitor.detected_ecosystems["package.json"] == "node"

    def test_detect_cargo_toml(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text("[package]\nname = 'foo'\n")
        monitor = DependencyMonitor(tmp_path)
        assert "Cargo.toml" in monitor.detected_ecosystems
        assert monitor.detected_ecosystems["Cargo.toml"] == "rust"

    def test_detect_go_mod(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text("module example.com/foo\n")
        monitor = DependencyMonitor(tmp_path)
        assert "go.mod" in monitor.detected_ecosystems
        assert monitor.detected_ecosystems["go.mod"] == "go"

    def test_detect_gemfile(self, tmp_path: Path) -> None:
        (tmp_path / "Gemfile").write_text("gem 'rails'\n")
        monitor = DependencyMonitor(tmp_path)
        assert "Gemfile" in monitor.detected_ecosystems
        assert monitor.detected_ecosystems["Gemfile"] == "ruby"

    def test_detect_pom_xml(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text("<project></project>")
        monitor = DependencyMonitor(tmp_path)
        assert "pom.xml" in monitor.detected_ecosystems
        assert monitor.detected_ecosystems["pom.xml"] == "java"

    def test_detect_multiple(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / "package.json").write_text("{}")
        monitor = DependencyMonitor(tmp_path)
        assert len(monitor.detected_ecosystems) == 2

    def test_detect_none(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        assert len(monitor.detected_ecosystems) == 0

    # --- Parse pyproject.toml ---

    def test_parse_pyproject_simple(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "myapp"\n\n'
            "dependencies = [\n"
            '    "typer>=0.9",\n'
            '    "rich>=13.0",\n'
            "]\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pyproject(tmp_path / "pyproject.toml")
        assert len(deps) == 2
        assert deps[0].name == "typer"
        assert deps[0].current_version == "0.9"
        assert deps[0].ecosystem == "python"
        assert deps[1].name == "rich"
        assert deps[1].current_version == "13.0"

    def test_parse_pyproject_with_extras(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            "dependencies = [\n"
            '    "httpx[http2]>=0.24",\n'
            "]\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pyproject(tmp_path / "pyproject.toml")
        assert len(deps) == 1
        assert deps[0].name == "httpx"
        assert deps[0].current_version == "0.24"

    def test_parse_pyproject_missing_file(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pyproject(tmp_path / "nonexistent.toml")
        assert deps == []

    def test_parse_pyproject_no_deps_section(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "nodeps"\n')
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pyproject(tmp_path / "pyproject.toml")
        assert deps == []

    # --- Parse requirements.txt ---

    def test_parse_requirements_simple(self, tmp_path: Path) -> None:
        (tmp_path / "requirements.txt").write_text(
            "requests>=2.28\nflask==2.0.0\nnumpy\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_requirements(tmp_path / "requirements.txt")
        assert len(deps) == 3
        assert deps[0].name == "requests"
        assert deps[0].current_version == "2.28"
        assert deps[1].name == "flask"
        assert deps[1].current_version == "2.0.0"
        assert deps[2].name == "numpy"

    def test_parse_requirements_comments_and_blanks(self, tmp_path: Path) -> None:
        (tmp_path / "requirements.txt").write_text(
            "# This is a comment\n\n"
            "requests>=2.28\n"
            "# Another comment\n"
            "-r other.txt\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_requirements(tmp_path / "requirements.txt")
        assert len(deps) == 1
        assert deps[0].name == "requests"

    def test_parse_requirements_missing_file(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_requirements(tmp_path / "nonexistent.txt")
        assert deps == []

    # --- Parse package.json ---

    def test_parse_package_json_simple(self, tmp_path: Path) -> None:
        data = {
            "name": "myapp",
            "dependencies": {"express": "^4.18.0", "lodash": "~4.17.21"},
            "devDependencies": {"jest": "^29.0.0"},
        }
        (tmp_path / "package.json").write_text(json.dumps(data))
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_package_json(tmp_path / "package.json")
        assert len(deps) == 3
        names = {d.name for d in deps}
        assert names == {"express", "lodash", "jest"}
        # Version prefixes stripped
        express_dep = next(d for d in deps if d.name == "express")
        assert express_dep.current_version == "4.18.0"
        assert express_dep.ecosystem == "node"

    def test_parse_package_json_no_deps(self, tmp_path: Path) -> None:
        data = {"name": "empty-app"}
        (tmp_path / "package.json").write_text(json.dumps(data))
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_package_json(tmp_path / "package.json")
        assert deps == []

    def test_parse_package_json_malformed(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("not valid json {{{")
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_package_json(tmp_path / "package.json")
        assert deps == []

    def test_parse_package_json_missing_file(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_package_json(tmp_path / "nonexistent.json")
        assert deps == []

    # --- Count usages ---

    def test_count_usages_finds_imports(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "app.py").write_text("import requests\nfrom requests import get\n")
        (src / "utils.py").write_text("import os\n")
        monitor = DependencyMonitor(tmp_path)
        count = monitor._count_usages("requests", "python")
        assert count == 1  # only app.py

    def test_count_usages_hyphen_to_underscore(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "app.py").write_text("import prompt_toolkit\n")
        monitor = DependencyMonitor(tmp_path)
        count = monitor._count_usages("prompt-toolkit", "python")
        assert count == 1

    def test_count_usages_no_src_dir(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("import requests\n")
        monitor = DependencyMonitor(tmp_path)
        # Falls back to scanning project root
        count = monitor._count_usages("requests", "python")
        assert count == 1

    def test_count_usages_no_matches(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "app.py").write_text("import os\n")
        monitor = DependencyMonitor(tmp_path)
        count = monitor._count_usages("requests", "python")
        assert count == 0

    def test_count_usages_node(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "index.js").write_text("const express = require('express');\n")
        monitor = DependencyMonitor(tmp_path)
        count = monitor._count_usages("express", "node")
        assert count == 1

    # --- Find unused ---

    def test_find_unused(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = [
            DependencyInfo("used-pkg", "1.0", "", False, 0, "python", "pyproject.toml", usages=5),
            DependencyInfo("unused-pkg", "1.0", "", False, 0, "python", "pyproject.toml", usages=0),
        ]
        unused = monitor._find_unused(deps)
        assert unused == ["unused-pkg"]

    def test_find_unused_implicit_packages_excluded(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = [
            DependencyInfo("pytest", "7.0", "", False, 0, "python", "pyproject.toml", usages=0),
            DependencyInfo("ruff", "0.1", "", False, 0, "python", "pyproject.toml", usages=0),
            DependencyInfo("mypy", "1.0", "", False, 0, "python", "pyproject.toml", usages=0),
            DependencyInfo("bandit", "1.7", "", False, 0, "python", "pyproject.toml", usages=0),
        ]
        unused = monitor._find_unused(deps)
        assert unused == []

    def test_find_unused_all_used(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = [
            DependencyInfo("pkg-a", "1.0", "", False, 0, "python", "pyproject.toml", usages=3),
            DependencyInfo("pkg-b", "2.0", "", False, 0, "python", "pyproject.toml", usages=1),
        ]
        unused = monitor._find_unused(deps)
        assert unused == []

    # --- Assess migration ---

    def test_assess_migration_not_outdated(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        dep = DependencyInfo("pkg", "1.0", "1.0", False, 0, "python", "pyproject.toml", usages=50)
        assert monitor.assess_migration(dep) is MigrationComplexity.TRIVIAL

    def test_assess_migration_trivial(self, tmp_path: Path) -> None:
        """Patch bump → TRIVIAL via version heuristic."""
        monitor = DependencyMonitor(tmp_path)
        dep = DependencyInfo("pkg", "1.0.0", "1.0.1", True, 180, "python", "pyproject.toml", usages=2)
        assert monitor.assess_migration(dep) is MigrationComplexity.TRIVIAL

    def test_assess_migration_simple(self, tmp_path: Path) -> None:
        """Minor bump → SIMPLE via version heuristic."""
        monitor = DependencyMonitor(tmp_path)
        dep = DependencyInfo("pkg", "1.0.0", "1.1.0", True, 180, "python", "pyproject.toml", usages=5)
        assert monitor.assess_migration(dep) is MigrationComplexity.SIMPLE

    def test_assess_migration_moderate(self, tmp_path: Path) -> None:
        """Large minor gap (>5) → MODERATE via version heuristic."""
        monitor = DependencyMonitor(tmp_path)
        dep = DependencyInfo("pkg", "1.0.0", "1.8.0", True, 180, "python", "pyproject.toml", usages=15)
        assert monitor.assess_migration(dep) is MigrationComplexity.MODERATE

    def test_assess_migration_complex(self, tmp_path: Path) -> None:
        """Major version change → COMPLEX via version heuristic."""
        monitor = DependencyMonitor(tmp_path)
        dep = DependencyInfo("pkg", "1.0", "2.0", True, 180, "python", "pyproject.toml", usages=25)
        assert monitor.assess_migration(dep) is MigrationComplexity.COMPLEX

    def test_assess_migration_boundary_simple(self, tmp_path: Path) -> None:
        """Fallback: 4 usages -> SIMPLE (>3 threshold) when no latest_version."""
        monitor = DependencyMonitor(tmp_path)
        dep = DependencyInfo("pkg", "1.0", "", True, 0, "python", "pyproject.toml", usages=4)
        assert monitor.assess_migration(dep) is MigrationComplexity.SIMPLE

    def test_assess_migration_boundary_moderate(self, tmp_path: Path) -> None:
        """Fallback: 11 usages -> MODERATE (>10 threshold) when no latest_version."""
        monitor = DependencyMonitor(tmp_path)
        dep = DependencyInfo("pkg", "1.0", "", True, 0, "python", "pyproject.toml", usages=11)
        assert monitor.assess_migration(dep) is MigrationComplexity.MODERATE

    def test_assess_migration_boundary_complex(self, tmp_path: Path) -> None:
        """Fallback: 21 usages -> COMPLEX (>20 threshold) when no latest_version."""
        monitor = DependencyMonitor(tmp_path)
        dep = DependencyInfo("pkg", "1.0", "", True, 0, "python", "pyproject.toml", usages=21)
        assert monitor.assess_migration(dep) is MigrationComplexity.COMPLEX

    # --- Check vulnerabilities ---

    @patch("prism.intelligence.deps.DependencyMonitor._query_osv")
    def test_check_vulnerabilities_with_results(
        self, mock_query: MagicMock, tmp_path: Path,
    ) -> None:
        mock_query.return_value = [
            VulnerabilityReport(
                package="django",
                cve_id="CVE-2024-9999",
                severity=VulnerabilitySeverity.HIGH,
                description="XSS vulnerability",
                fixed_version="4.2.10",
                url="",
            ),
        ]
        (tmp_path / "requirements.txt").write_text("django>=4.2.5\n")
        monitor = DependencyMonitor(tmp_path)
        vulns = monitor._check_vulnerabilities()
        assert len(vulns) == 1
        assert vulns[0].package == "django"
        assert vulns[0].cve_id == "CVE-2024-9999"
        assert vulns[0].fixed_version == "4.2.10"
        assert vulns[0].severity is VulnerabilitySeverity.HIGH

    @patch("prism.intelligence.deps.DependencyMonitor._query_osv")
    def test_check_vulnerabilities_no_results(
        self, mock_query: MagicMock, tmp_path: Path,
    ) -> None:
        mock_query.return_value = []
        (tmp_path / "requirements.txt").write_text("django>=4.2.5\n")
        monitor = DependencyMonitor(tmp_path)
        vulns = monitor._check_vulnerabilities()
        assert vulns == []

    @patch("prism.intelligence.deps.DependencyMonitor._query_osv")
    def test_check_vulnerabilities_query_error(
        self, mock_query: MagicMock, tmp_path: Path,
    ) -> None:
        mock_query.side_effect = Exception("Network error")
        (tmp_path / "requirements.txt").write_text("django>=4.2.5\n")
        monitor = DependencyMonitor(tmp_path)
        vulns = monitor._check_vulnerabilities()
        assert vulns == []

    def test_check_vulnerabilities_no_deps(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        vulns = monitor._check_vulnerabilities()
        assert vulns == []

    def test_check_vulnerabilities_non_python_ecosystem(self, tmp_path: Path) -> None:
        """With no node deps, filtering by node yields empty."""
        monitor = DependencyMonitor(tmp_path)
        vulns = monitor._check_vulnerabilities(ecosystem="node")
        assert vulns == []

    @patch("prism.intelligence.deps.DependencyMonitor._query_osv")
    def test_check_vulnerabilities_no_fix_versions(
        self, mock_query: MagicMock, tmp_path: Path,
    ) -> None:
        mock_query.return_value = [
            VulnerabilityReport(
                package="old-pkg",
                cve_id="CVE-2024-0001",
                severity=VulnerabilitySeverity.MEDIUM,
                description="Some vuln",
                fixed_version="",
                url="",
            ),
        ]
        (tmp_path / "requirements.txt").write_text("old-pkg>=1.0.0\n")
        monitor = DependencyMonitor(tmp_path)
        vulns = monitor._check_vulnerabilities()
        assert len(vulns) == 1
        assert vulns[0].fixed_version == ""

    # --- get_status (integration) ---

    @patch("prism.intelligence.deps.DependencyMonitor._query_osv")
    def test_get_status_full_report(self, mock_query: MagicMock, tmp_path: Path) -> None:
        mock_query.return_value = []
        # Create pyproject.toml with deps
        src = tmp_path / "src"
        src.mkdir()
        (src / "app.py").write_text("import typer\nfrom rich.console import Console\n")
        (tmp_path / "pyproject.toml").write_text(
            "dependencies = [\n"
            '    "typer>=0.9",\n'
            '    "rich>=13.0",\n'
            '    "some-unused-pkg>=1.0",\n'
            "]\n"
        )

        monitor = DependencyMonitor(tmp_path)
        report = monitor.get_status()

        assert isinstance(report, DepsReport)
        assert report.total_deps == 3
        assert report.ecosystem == "all"
        assert report.generated_at  # non-empty
        assert report.vulnerable == 0
        # some-unused-pkg should be flagged as unused
        assert "some-unused-pkg" in report.unused_deps

    @patch("prism.intelligence.deps.DependencyMonitor._query_osv")
    def test_get_status_ecosystem_filter(
        self, mock_query: MagicMock, tmp_path: Path,
    ) -> None:
        mock_query.return_value = []
        (tmp_path / "pyproject.toml").write_text(
            "dependencies = [\n"
            '    "requests>=2.28",\n'
            "]\n"
        )
        (tmp_path / "package.json").write_text(
            json.dumps({"dependencies": {"express": "^4.0.0"}}),
        )

        monitor = DependencyMonitor(tmp_path)

        py_report = monitor.get_status(ecosystem="python")
        assert py_report.ecosystem == "python"
        py_names = {d.name for d in py_report.dependencies}
        assert "requests" in py_names
        assert "express" not in py_names

        node_report = monitor.get_status(ecosystem="node")
        assert node_report.ecosystem == "node"
        node_names = {d.name for d in node_report.dependencies}
        assert "express" in node_names
        assert "requests" not in node_names

    @patch("prism.intelligence.deps.DependencyMonitor._query_osv")
    def test_get_status_empty_project(
        self, mock_query: MagicMock, tmp_path: Path,
    ) -> None:
        mock_query.return_value = []
        monitor = DependencyMonitor(tmp_path)
        report = monitor.get_status()
        assert report.total_deps == 0
        assert report.outdated == 0
        assert report.vulnerable == 0
        assert report.unused == 0

    @patch("prism.intelligence.deps.DependencyMonitor._query_osv")
    def test_get_status_assigns_migration_complexity(
        self, mock_query: MagicMock, tmp_path: Path,
    ) -> None:
        mock_query.return_value = []
        # Create a project where a dep has many usages
        src = tmp_path / "src"
        src.mkdir()
        for i in range(25):
            (src / f"mod_{i}.py").write_text("import requests\n")
        (tmp_path / "requirements.txt").write_text("requests>=2.28\n")

        monitor = DependencyMonitor(tmp_path)
        report = monitor.get_status()
        assert report.total_deps == 1
        # The dep is not marked as outdated, so complexity stays TRIVIAL
        assert report.dependencies[0].migration_complexity is MigrationComplexity.TRIVIAL

    # --- detected_ecosystems returns copy ---

    def test_detected_ecosystems_returns_copy(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        monitor = DependencyMonitor(tmp_path)
        eco1 = monitor.detected_ecosystems
        eco2 = monitor.detected_ecosystems
        assert eco1 == eco2
        assert eco1 is not eco2  # different dict objects
