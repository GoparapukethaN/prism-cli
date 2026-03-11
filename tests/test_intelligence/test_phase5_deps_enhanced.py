"""Tests for enhanced prism.intelligence.deps — multi-ecosystem parsing,
OSV.dev vulnerability scanning, version-based migration assessment,
and unused detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

from prism.intelligence.deps import (
    DependencyInfo,
    DependencyMonitor,
    DependencyStatusReport,
    MigrationComplexity,
    Vulnerability,
    VulnerabilityReport,
    VulnerabilitySeverity,
    _parse_semver,
    assess_migration_by_version,
)

# ======================================================================
# Test _parse_semver helper
# ======================================================================


class TestParseSemver:
    """Tests for the _parse_semver helper function."""

    def test_full_version(self) -> None:
        assert _parse_semver("2.28.1") == (2, 28, 1)

    def test_major_minor_only(self) -> None:
        assert _parse_semver("2.28") == (2, 28, 0)

    def test_major_only(self) -> None:
        assert _parse_semver("2") == (2, 0, 0)

    def test_v_prefix(self) -> None:
        assert _parse_semver("v1.9.1") == (1, 9, 1)

    def test_prerelease_suffix(self) -> None:
        assert _parse_semver("3.0.0-alpha.1") == (3, 0, 0)

    def test_rc_suffix_no_separator(self) -> None:
        assert _parse_semver("2.0.0rc1") == (2, 0, 0)

    def test_beta_suffix(self) -> None:
        assert _parse_semver("1.5.0beta2") == (1, 5, 0)

    def test_build_metadata(self) -> None:
        assert _parse_semver("1.0.0+build.42") == (1, 0, 0)

    def test_empty_string(self) -> None:
        assert _parse_semver("") == (0, 0, 0)

    def test_whitespace(self) -> None:
        assert _parse_semver("  2.1.0  ") == (2, 1, 0)


# ======================================================================
# Test assess_migration_by_version
# ======================================================================


class TestAssessMigrationByVersion:
    """Tests for version-based migration complexity assessment."""

    def test_trivial_patch_bump(self) -> None:
        result = assess_migration_by_version("2.28.0", "2.28.1")
        assert result is MigrationComplexity.TRIVIAL

    def test_simple_minor_bump(self) -> None:
        result = assess_migration_by_version("2.28.0", "2.29.0")
        assert result is MigrationComplexity.SIMPLE

    def test_simple_small_minor_bump(self) -> None:
        result = assess_migration_by_version("1.0.0", "1.3.0")
        assert result is MigrationComplexity.SIMPLE

    def test_moderate_large_minor_bump(self) -> None:
        result = assess_migration_by_version("1.0.0", "1.10.0")
        assert result is MigrationComplexity.MODERATE

    def test_complex_major_bump(self) -> None:
        result = assess_migration_by_version("1.5.0", "2.0.0")
        assert result is MigrationComplexity.COMPLEX

    def test_complex_large_major_bump(self) -> None:
        result = assess_migration_by_version("1.0.0", "5.0.0")
        assert result is MigrationComplexity.COMPLEX

    def test_same_version(self) -> None:
        result = assess_migration_by_version("1.0.0", "1.0.0")
        assert result is MigrationComplexity.TRIVIAL

    def test_empty_from_version(self) -> None:
        result = assess_migration_by_version("", "2.0.0")
        assert result is MigrationComplexity.TRIVIAL

    def test_empty_to_version(self) -> None:
        result = assess_migration_by_version("1.0.0", "")
        assert result is MigrationComplexity.TRIVIAL

    def test_both_empty(self) -> None:
        result = assess_migration_by_version("", "")
        assert result is MigrationComplexity.TRIVIAL


# ======================================================================
# Test VulnerabilityReport dataclass
# ======================================================================


class TestVulnerabilityReport:
    """Tests for the VulnerabilityReport dataclass."""

    def test_fields(self) -> None:
        report = VulnerabilityReport(
            package="requests",
            cve_id="CVE-2024-1234",
            severity=VulnerabilitySeverity.HIGH,
            description="Test vuln",
            fixed_version="2.32.0",
            url="https://example.com",
        )
        assert report.package == "requests"
        assert report.cve_id == "CVE-2024-1234"
        assert report.severity is VulnerabilitySeverity.HIGH
        assert report.fixed_version == "2.32.0"
        assert report.url == "https://example.com"


# ======================================================================
# Test DependencyStatusReport dataclass
# ======================================================================


class TestDependencyStatusReport:
    """Tests for the DependencyStatusReport dataclass."""

    def test_fields(self) -> None:
        report = DependencyStatusReport(
            dependencies=[],
            total=5,
            outdated_count=2,
            vulnerable_count=1,
        )
        assert report.total == 5
        assert report.outdated_count == 2
        assert report.vulnerable_count == 1
        assert report.dependencies == []
        assert report.vulnerabilities == []


# ======================================================================
# Test Vulnerability dataclass url field
# ======================================================================


class TestVulnerabilityUrlField:
    """Tests for the Vulnerability dataclass url field."""

    def test_url_default(self) -> None:
        vuln = Vulnerability(
            package="pkg",
            severity=VulnerabilitySeverity.LOW,
            cve_id="CVE-2024-0001",
            description="desc",
            fixed_version="1.0",
            current_version="0.9",
        )
        assert vuln.url == ""

    def test_url_explicit(self) -> None:
        vuln = Vulnerability(
            package="pkg",
            severity=VulnerabilitySeverity.LOW,
            cve_id="CVE-2024-0001",
            description="desc",
            fixed_version="1.0",
            current_version="0.9",
            url="https://example.com/cve",
        )
        assert vuln.url == "https://example.com/cve"


# ======================================================================
# Test setup.py parsing
# ======================================================================


class TestParseSetupPy:
    """Tests for parsing setup.py files."""

    def test_parse_simple_setup_py(self, tmp_path: Path) -> None:
        (tmp_path / "setup.py").write_text(
            "from setuptools import setup\n"
            "setup(\n"
            "    name='myapp',\n"
            "    install_requires=[\n"
            "        'requests>=2.28',\n"
            "        'flask==2.0.0',\n"
            "    ],\n"
            ")\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_setup_py(tmp_path / "setup.py")
        assert len(deps) == 2
        assert deps[0].name == "requests"
        assert deps[0].ecosystem == "python"
        assert deps[0].source_file == "setup.py"

    def test_parse_setup_py_no_deps(self, tmp_path: Path) -> None:
        (tmp_path / "setup.py").write_text(
            "from setuptools import setup\n"
            "setup(name='myapp')\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_setup_py(tmp_path / "setup.py")
        assert deps == []

    def test_parse_setup_py_missing_file(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_setup_py(tmp_path / "nonexistent.py")
        assert deps == []

    def test_parse_setup_py_with_extras(self, tmp_path: Path) -> None:
        (tmp_path / "setup.py").write_text(
            "from setuptools import setup\n"
            "setup(\n"
            "    install_requires=[\n"
            "        'httpx[http2]>=0.24',\n"
            "    ],\n"
            ")\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_setup_py(tmp_path / "setup.py")
        assert len(deps) == 1
        assert deps[0].name == "httpx"


# ======================================================================
# Test Pipfile parsing
# ======================================================================


class TestParsePipfile:
    """Tests for parsing Pipfile files."""

    def test_parse_simple_pipfile(self, tmp_path: Path) -> None:
        (tmp_path / "Pipfile").write_text(
            "[packages]\n"
            'requests = ">=2.28"\n'
            'flask = "*"\n'
            "\n"
            "[dev-packages]\n"
            'pytest = ">=7.0"\n'
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pipfile(tmp_path / "Pipfile")
        assert len(deps) == 3
        names = {d.name for d in deps}
        assert names == {"requests", "flask", "pytest"}

    def test_parse_pipfile_wildcard_version(self, tmp_path: Path) -> None:
        (tmp_path / "Pipfile").write_text(
            "[packages]\n"
            'flask = "*"\n'
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pipfile(tmp_path / "Pipfile")
        assert len(deps) == 1
        assert deps[0].current_version == "any"

    def test_parse_pipfile_missing_file(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pipfile(tmp_path / "Pipfile")
        assert deps == []

    def test_parse_pipfile_empty(self, tmp_path: Path) -> None:
        (tmp_path / "Pipfile").write_text("")
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pipfile(tmp_path / "Pipfile")
        assert deps == []

    def test_parse_pipfile_comments(self, tmp_path: Path) -> None:
        (tmp_path / "Pipfile").write_text(
            "[packages]\n"
            "# This is a comment\n"
            'requests = ">=2.28"\n'
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pipfile(tmp_path / "Pipfile")
        assert len(deps) == 1


# ======================================================================
# Test Cargo.toml parsing
# ======================================================================


class TestParseCargoToml:
    """Tests for parsing Cargo.toml files."""

    def test_parse_simple_deps(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text(
            "[package]\n"
            'name = "myapp"\n'
            'version = "0.1.0"\n'
            "\n"
            "[dependencies]\n"
            'serde = "1.0.193"\n'
            'tokio = "1.35.0"\n'
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_cargo_toml(tmp_path / "Cargo.toml")
        assert len(deps) == 2
        assert deps[0].name == "serde"
        assert deps[0].current_version == "1.0.193"
        assert deps[0].ecosystem == "rust"
        assert deps[0].source_file == "Cargo.toml"

    def test_parse_table_version(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text(
            "[dependencies]\n"
            'serde = { version = "1.0", features = ["derive"] }\n'
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_cargo_toml(tmp_path / "Cargo.toml")
        assert len(deps) == 1
        assert deps[0].name == "serde"
        assert deps[0].current_version == "1.0"

    def test_parse_dev_deps(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text(
            "[dev-dependencies]\n"
            'pretty_assertions = "1.4.0"\n'
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_cargo_toml(tmp_path / "Cargo.toml")
        assert len(deps) == 1
        assert deps[0].name == "pretty_assertions"

    def test_parse_no_deps(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text(
            "[package]\nname = \"myapp\"\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_cargo_toml(tmp_path / "Cargo.toml")
        assert deps == []

    def test_parse_missing_file(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_cargo_toml(tmp_path / "nonexistent.toml")
        assert deps == []


# ======================================================================
# Test go.mod parsing
# ======================================================================


class TestParseGoMod:
    """Tests for parsing go.mod files."""

    def test_parse_block_require(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text(
            "module example.com/myapp\n\n"
            "go 1.21\n\n"
            "require (\n"
            "\tgithub.com/gin-gonic/gin v1.9.1\n"
            "\tgithub.com/go-sql-driver/mysql v1.7.1\n"
            ")\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_go_mod(tmp_path / "go.mod")
        assert len(deps) == 2
        assert deps[0].name == "github.com/gin-gonic/gin"
        assert deps[0].current_version == "1.9.1"
        assert deps[0].ecosystem == "go"

    def test_parse_single_require(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text(
            "module example.com/myapp\n\n"
            "require github.com/pkg/errors v0.9.1\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_go_mod(tmp_path / "go.mod")
        assert len(deps) == 1
        assert deps[0].name == "github.com/pkg/errors"
        assert deps[0].current_version == "0.9.1"

    def test_parse_no_requires(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text(
            "module example.com/myapp\n\ngo 1.21\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_go_mod(tmp_path / "go.mod")
        assert deps == []

    def test_parse_missing_file(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_go_mod(tmp_path / "nonexistent")
        assert deps == []


# ======================================================================
# Test Gemfile parsing
# ======================================================================


class TestParseGemfile:
    """Tests for parsing Gemfile files."""

    def test_parse_simple_gemfile(self, tmp_path: Path) -> None:
        (tmp_path / "Gemfile").write_text(
            "source 'https://rubygems.org'\n\n"
            "gem 'rails', '~> 7.0'\n"
            "gem 'puma'\n"
            "gem 'pg', '1.5.4'\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_gemfile(tmp_path / "Gemfile")
        assert len(deps) == 3
        assert deps[0].name == "rails"
        assert deps[0].ecosystem == "ruby"
        assert deps[1].name == "puma"
        assert deps[1].current_version == "any"

    def test_parse_gemfile_double_quotes(self, tmp_path: Path) -> None:
        (tmp_path / "Gemfile").write_text(
            'gem "rails", "~> 7.0"\n'
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_gemfile(tmp_path / "Gemfile")
        assert len(deps) == 1
        assert deps[0].name == "rails"

    def test_parse_gemfile_comments(self, tmp_path: Path) -> None:
        (tmp_path / "Gemfile").write_text(
            "# comment\n"
            "gem 'rails'\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_gemfile(tmp_path / "Gemfile")
        assert len(deps) == 1

    def test_parse_gemfile_empty(self, tmp_path: Path) -> None:
        (tmp_path / "Gemfile").write_text("")
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_gemfile(tmp_path / "Gemfile")
        assert deps == []

    def test_parse_gemfile_missing_file(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_gemfile(tmp_path / "Gemfile")
        assert deps == []


# ======================================================================
# Test build.gradle parsing
# ======================================================================


class TestParseBuildGradle:
    """Tests for parsing build.gradle files."""

    def test_parse_simple_gradle(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text(
            "dependencies {\n"
            "    implementation 'com.google.guava:guava:31.1-jre'\n"
            "    testImplementation 'junit:junit:4.13.2'\n"
            "}\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_build_gradle(tmp_path / "build.gradle")
        assert len(deps) == 2
        assert deps[0].name == "com.google.guava:guava"
        assert deps[0].current_version == "31.1-jre"
        assert deps[0].ecosystem == "java"
        assert deps[0].source_file == "build.gradle"

    def test_parse_gradle_double_quotes(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text(
            "dependencies {\n"
            '    implementation "org.slf4j:slf4j-api:1.7.36"\n'
            "}\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_build_gradle(tmp_path / "build.gradle")
        assert len(deps) == 1
        assert deps[0].name == "org.slf4j:slf4j-api"

    def test_parse_gradle_empty(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text(
            "apply plugin: 'java'\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_build_gradle(tmp_path / "build.gradle")
        assert deps == []

    def test_parse_gradle_missing_file(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_build_gradle(
            tmp_path / "nonexistent.gradle",
        )
        assert deps == []


# ======================================================================
# Test pom.xml parsing
# ======================================================================


class TestParsePomXml:
    """Tests for parsing pom.xml files."""

    def test_parse_simple_pom(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text(
            '<?xml version="1.0"?>\n'
            "<project>\n"
            "  <dependencies>\n"
            "    <dependency>\n"
            "      <groupId>junit</groupId>\n"
            "      <artifactId>junit</artifactId>\n"
            "      <version>4.13.2</version>\n"
            "    </dependency>\n"
            "  </dependencies>\n"
            "</project>\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pom_xml(tmp_path / "pom.xml")
        assert len(deps) == 1
        assert deps[0].name == "junit:junit"
        assert deps[0].current_version == "4.13.2"
        assert deps[0].ecosystem == "java"
        assert deps[0].source_file == "pom.xml"

    def test_parse_pom_no_version(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text(
            '<?xml version="1.0"?>\n'
            "<project>\n"
            "  <dependencies>\n"
            "    <dependency>\n"
            "      <groupId>junit</groupId>\n"
            "      <artifactId>junit</artifactId>\n"
            "    </dependency>\n"
            "  </dependencies>\n"
            "</project>\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pom_xml(tmp_path / "pom.xml")
        assert len(deps) == 1
        assert deps[0].current_version == "any"

    def test_parse_pom_malformed_xml(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text("not valid xml <<<")
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pom_xml(tmp_path / "pom.xml")
        assert deps == []

    def test_parse_pom_missing_file(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pom_xml(tmp_path / "nonexistent.xml")
        assert deps == []

    def test_parse_pom_with_namespace(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text(
            '<?xml version="1.0"?>\n'
            '<project xmlns="http://maven.apache.org/'
            'POM/4.0.0">\n'
            "  <dependencies>\n"
            "    <dependency>\n"
            "      <groupId>org.springframework</groupId>\n"
            "      <artifactId>spring-core</artifactId>\n"
            "      <version>5.3.20</version>\n"
            "    </dependency>\n"
            "  </dependencies>\n"
            "</project>\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pom_xml(tmp_path / "pom.xml")
        assert len(deps) == 1
        assert deps[0].name == "org.springframework:spring-core"
        assert deps[0].current_version == "5.3.20"


# ======================================================================
# Test ecosystem detection for new file types
# ======================================================================


class TestEcosystemDetectionEnhanced:
    """Tests for detecting the new dependency file types."""

    def test_detect_pipfile(self, tmp_path: Path) -> None:
        (tmp_path / "Pipfile").write_text("[packages]\n")
        monitor = DependencyMonitor(tmp_path)
        assert "Pipfile" in monitor.detected_ecosystems
        assert monitor.detected_ecosystems["Pipfile"] == "python"

    def test_detect_setup_py(self, tmp_path: Path) -> None:
        (tmp_path / "setup.py").write_text("setup()\n")
        monitor = DependencyMonitor(tmp_path)
        assert "setup.py" in monitor.detected_ecosystems
        assert monitor.detected_ecosystems["setup.py"] == "python"

    def test_detect_build_gradle(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text(
            "apply plugin: 'java'\n"
        )
        monitor = DependencyMonitor(tmp_path)
        assert "build.gradle" in monitor.detected_ecosystems
        assert monitor.detected_ecosystems["build.gradle"] == "java"


# ======================================================================
# Test OSV.dev vulnerability checking
# ======================================================================


class TestCheckVulnerabilities:
    """Tests for the OSV.dev vulnerability checking."""

    @patch("prism.intelligence.deps.DependencyMonitor._query_osv")
    def test_check_vulnerabilities_with_results(
        self, mock_query: MagicMock, tmp_path: Path,
    ) -> None:
        mock_query.return_value = [
            VulnerabilityReport(
                package="requests",
                cve_id="CVE-2024-1234",
                severity=VulnerabilitySeverity.HIGH,
                description="Test vulnerability",
                fixed_version="2.32.0",
                url="https://example.com",
            ),
        ]
        monitor = DependencyMonitor(tmp_path)
        deps = [
            DependencyInfo(
                name="requests",
                current_version="2.28.0",
                latest_version="",
                is_outdated=False,
                age_days=0,
                ecosystem="python",
                source_file="requirements.txt",
                usages=5,
            ),
        ]
        reports = monitor.check_vulnerabilities(deps)
        assert len(reports) == 1
        assert reports[0].cve_id == "CVE-2024-1234"
        assert reports[0].severity is VulnerabilitySeverity.HIGH

    @patch("prism.intelligence.deps.DependencyMonitor._query_osv")
    def test_check_vulnerabilities_no_results(
        self, mock_query: MagicMock, tmp_path: Path,
    ) -> None:
        mock_query.return_value = []
        monitor = DependencyMonitor(tmp_path)
        deps = [
            DependencyInfo(
                name="requests",
                current_version="2.31.0",
                latest_version="",
                is_outdated=False,
                age_days=0,
                ecosystem="python",
                source_file="requirements.txt",
                usages=5,
            ),
        ]
        reports = monitor.check_vulnerabilities(deps)
        assert reports == []

    def test_check_vulnerabilities_any_version_skipped(
        self, tmp_path: Path,
    ) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = [
            DependencyInfo(
                name="requests",
                current_version="any",
                latest_version="",
                is_outdated=False,
                age_days=0,
                ecosystem="python",
                source_file="requirements.txt",
                usages=5,
            ),
        ]
        reports = monitor.check_vulnerabilities(deps)
        assert reports == []

    def test_check_vulnerabilities_unknown_ecosystem(
        self, tmp_path: Path,
    ) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = [
            DependencyInfo(
                name="unknown-pkg",
                current_version="1.0.0",
                latest_version="",
                is_outdated=False,
                age_days=0,
                ecosystem="unknown",
                source_file="unknown.txt",
                usages=0,
            ),
        ]
        reports = monitor.check_vulnerabilities(deps)
        assert reports == []

    @patch("prism.intelligence.deps.DependencyMonitor._query_osv")
    def test_check_vulnerabilities_query_error(
        self, mock_query: MagicMock, tmp_path: Path,
    ) -> None:
        mock_query.side_effect = Exception("Network error")
        monitor = DependencyMonitor(tmp_path)
        deps = [
            DependencyInfo(
                name="requests",
                current_version="2.28.0",
                latest_version="",
                is_outdated=False,
                age_days=0,
                ecosystem="python",
                source_file="requirements.txt",
                usages=5,
            ),
        ]
        reports = monitor.check_vulnerabilities(deps)
        assert reports == []


# ======================================================================
# Test OSV.dev response parsing
# ======================================================================


class TestParseOsvResponse:
    """Tests for parsing OSV.dev API responses."""

    def test_parse_with_cve_alias(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        data = {
            "vulns": [
                {
                    "id": "GHSA-abc123",
                    "aliases": ["CVE-2024-1234"],
                    "summary": "XSS vulnerability",
                    "affected": [
                        {
                            "ranges": [
                                {
                                    "events": [
                                        {"introduced": "0"},
                                        {"fixed": "2.32.0"},
                                    ],
                                },
                            ],
                        },
                    ],
                    "references": [
                        {"url": "https://github.com/advisory"},
                    ],
                },
            ],
        }
        reports = monitor._parse_osv_response("requests", data)
        assert len(reports) == 1
        assert reports[0].cve_id == "CVE-2024-1234"
        assert reports[0].fixed_version == "2.32.0"
        assert reports[0].url == "https://github.com/advisory"

    def test_parse_no_vulns(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        data: dict[str, list[object]] = {"vulns": []}
        reports = monitor._parse_osv_response("requests", data)
        assert reports == []

    def test_parse_empty_response(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        data: dict[str, list[object]] = {}
        reports = monitor._parse_osv_response("requests", data)
        assert reports == []

    def test_parse_with_severity_score(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        data = {
            "vulns": [
                {
                    "id": "GHSA-xyz",
                    "summary": "Critical vuln",
                    "severity": [{"score": "9.5"}],
                    "affected": [],
                },
            ],
        }
        reports = monitor._parse_osv_response("pkg", data)
        assert len(reports) == 1
        assert reports[0].severity is VulnerabilitySeverity.CRITICAL

    def test_parse_with_database_specific_severity(
        self, tmp_path: Path,
    ) -> None:
        monitor = DependencyMonitor(tmp_path)
        data = {
            "vulns": [
                {
                    "id": "GHSA-xyz",
                    "summary": "Low vuln",
                    "database_specific": {"severity": "LOW"},
                    "affected": [],
                },
            ],
        }
        reports = monitor._parse_osv_response("pkg", data)
        assert len(reports) == 1
        assert reports[0].severity is VulnerabilitySeverity.LOW

    def test_parse_no_fixed_version(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        data = {
            "vulns": [
                {
                    "id": "GHSA-xyz",
                    "summary": "No fix yet",
                    "affected": [
                        {
                            "ranges": [
                                {
                                    "events": [
                                        {"introduced": "0"},
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        }
        reports = monitor._parse_osv_response("pkg", data)
        assert len(reports) == 1
        assert reports[0].fixed_version == ""

    def test_parse_cve_url_fallback(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        data = {
            "vulns": [
                {
                    "id": "GHSA-xyz",
                    "aliases": ["CVE-2024-9999"],
                    "summary": "Test",
                    "affected": [],
                    "references": [],
                },
            ],
        }
        reports = monitor._parse_osv_response("pkg", data)
        assert len(reports) == 1
        assert "CVE-2024-9999" in reports[0].url


# ======================================================================
# Test migration assessment (version-based)
# ======================================================================


class TestAssessMigrationVersionBased:
    """Tests for version-based migration on DependencyMonitor."""

    def test_version_based_trivial(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        dep = DependencyInfo(
            "pkg", "1.0.0", "1.0.1",
            True, 10, "python", "pyproject.toml", 5,
        )
        result = monitor.assess_migration(dep)
        assert result is MigrationComplexity.TRIVIAL

    def test_version_based_simple(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        dep = DependencyInfo(
            "pkg", "1.0.0", "1.2.0",
            True, 30, "python", "pyproject.toml", 5,
        )
        result = monitor.assess_migration(dep)
        assert result is MigrationComplexity.SIMPLE

    def test_version_based_complex(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        dep = DependencyInfo(
            "pkg", "1.0.0", "2.0.0",
            True, 365, "python", "pyproject.toml", 5,
        )
        result = monitor.assess_migration(dep)
        assert result is MigrationComplexity.COMPLEX

    def test_fallback_to_usage_when_no_latest(
        self, tmp_path: Path,
    ) -> None:
        monitor = DependencyMonitor(tmp_path)
        dep = DependencyInfo(
            "pkg", "1.0.0", "",
            True, 30, "python", "pyproject.toml", 25,
        )
        result = monitor.assess_migration(dep)
        assert result is MigrationComplexity.COMPLEX

    def test_not_outdated_is_trivial(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        dep = DependencyInfo(
            "pkg", "2.0.0", "2.0.0",
            False, 0, "python", "pyproject.toml", 50,
        )
        result = monitor.assess_migration(dep)
        assert result is MigrationComplexity.TRIVIAL


# ======================================================================
# Test unused dependency detection (public API)
# ======================================================================


class TestFindUnusedPublicApi:
    """Tests for the public find_unused method."""

    def test_find_unused_returns_names(self, tmp_path: Path) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = [
            DependencyInfo(
                "used-pkg", "1.0", "", False, 0,
                "python", "pyproject.toml", usages=5,
            ),
            DependencyInfo(
                "unused-pkg", "1.0", "", False, 0,
                "python", "pyproject.toml", usages=0,
            ),
        ]
        unused = monitor.find_unused(deps)
        assert unused == ["unused-pkg"]

    def test_find_unused_implicit_excluded(
        self, tmp_path: Path,
    ) -> None:
        monitor = DependencyMonitor(tmp_path)
        deps = [
            DependencyInfo(
                "pytest", "7.0", "", False, 0,
                "python", "pyproject.toml", usages=0,
            ),
        ]
        unused = monitor.find_unused(deps)
        assert unused == []


# ======================================================================
# Test generate_status_report
# ======================================================================


class TestGenerateStatusReport:
    """Tests for generate_status_report method."""

    @patch("prism.intelligence.deps.DependencyMonitor._query_osv")
    def test_basic_report(
        self, mock_query: MagicMock, tmp_path: Path,
    ) -> None:
        mock_query.return_value = []
        monitor = DependencyMonitor(tmp_path)
        deps = [
            DependencyInfo(
                "requests", "2.28.0", "2.31.0",
                True, 180, "python", "requirements.txt", 5,
            ),
            DependencyInfo(
                "flask", "2.0.0", "2.0.0",
                False, 0, "python", "requirements.txt", 3,
            ),
        ]
        report = monitor.generate_status_report(deps)
        assert isinstance(report, DependencyStatusReport)
        assert report.total == 2
        assert report.outdated_count == 1
        assert report.vulnerable_count == 0
        assert len(report.dependencies) == 2

    @patch("prism.intelligence.deps.DependencyMonitor._query_osv")
    def test_report_with_vulns(
        self, mock_query: MagicMock, tmp_path: Path,
    ) -> None:
        mock_query.return_value = [
            VulnerabilityReport(
                package="requests",
                cve_id="CVE-2024-1234",
                severity=VulnerabilitySeverity.HIGH,
                description="vuln",
                fixed_version="2.32.0",
                url="",
            ),
        ]
        monitor = DependencyMonitor(tmp_path)
        deps = [
            DependencyInfo(
                "requests", "2.28.0", "",
                False, 0, "python", "requirements.txt", 5,
            ),
        ]
        report = monitor.generate_status_report(deps)
        assert report.vulnerable_count == 1
        assert len(report.vulnerabilities) == 1


# ======================================================================
# Test _parse_dependencies dispatching
# ======================================================================


class TestParseDependenciesDispatch:
    """Tests for _parse_dependencies routing to correct parsers."""

    def test_dispatches_to_setup_py(self, tmp_path: Path) -> None:
        (tmp_path / "setup.py").write_text(
            "from setuptools import setup\n"
            "setup(install_requires=['requests>=2.28'])\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_dependencies()
        assert len(deps) == 1
        assert deps[0].name == "requests"

    def test_dispatches_to_pipfile(self, tmp_path: Path) -> None:
        (tmp_path / "Pipfile").write_text(
            "[packages]\n"
            'requests = ">=2.28"\n'
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_dependencies()
        assert len(deps) == 1
        assert deps[0].name == "requests"

    def test_dispatches_to_cargo(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text(
            "[dependencies]\n"
            'serde = "1.0"\n'
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_dependencies()
        assert len(deps) == 1
        assert deps[0].name == "serde"

    def test_dispatches_to_go_mod(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text(
            "module example.com/app\n\n"
            "require github.com/pkg/errors v0.9.1\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_dependencies()
        assert len(deps) == 1
        assert deps[0].name == "github.com/pkg/errors"

    def test_dispatches_to_gemfile(self, tmp_path: Path) -> None:
        (tmp_path / "Gemfile").write_text(
            "gem 'rails', '~> 7.0'\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_dependencies()
        assert len(deps) == 1
        assert deps[0].name == "rails"

    def test_dispatches_to_build_gradle(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text(
            "dependencies {\n"
            "    implementation 'com.google:guava:31.1'\n"
            "}\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_dependencies()
        assert len(deps) == 1
        assert deps[0].name == "com.google:guava"

    def test_ecosystem_filter(self, tmp_path: Path) -> None:
        (tmp_path / "requirements.txt").write_text("requests>=2.28\n")
        (tmp_path / "Gemfile").write_text("gem 'rails'\n")
        monitor = DependencyMonitor(tmp_path)

        py_deps = monitor._parse_dependencies(ecosystem="python")
        assert all(d.ecosystem == "python" for d in py_deps)

        rb_deps = monitor._parse_dependencies(ecosystem="ruby")
        assert all(d.ecosystem == "ruby" for d in rb_deps)


# ======================================================================
# Test edge cases
# ======================================================================


class TestEdgeCases:
    """Tests for edge cases across parsers."""

    def test_empty_requirements_file(self, tmp_path: Path) -> None:
        (tmp_path / "requirements.txt").write_text("")
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_requirements(
            tmp_path / "requirements.txt",
        )
        assert deps == []

    def test_malformed_version_in_pyproject(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "pyproject.toml").write_text(
            "dependencies = [\n"
            '    "weirdpkg",\n'
            "]\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pyproject(tmp_path / "pyproject.toml")
        # The dep won't match the version regex
        assert deps == []

    def test_pom_xml_empty_project(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text(
            '<?xml version="1.0"?>\n<project></project>\n'
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_pom_xml(tmp_path / "pom.xml")
        assert deps == []

    def test_go_mod_with_indirect(self, tmp_path: Path) -> None:
        """Indirect deps are still parsed from require blocks."""
        (tmp_path / "go.mod").write_text(
            "module example.com/app\n\n"
            "require (\n"
            "\tgithub.com/foo/bar v1.0.0\n"
            ")\n"
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_go_mod(tmp_path / "go.mod")
        assert len(deps) == 1

    def test_cargo_toml_with_comments(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text(
            "[dependencies]\n"
            "# This is a comment\n"
            'serde = "1.0"\n'
        )
        monitor = DependencyMonitor(tmp_path)
        deps = monitor._parse_cargo_toml(tmp_path / "Cargo.toml")
        assert len(deps) == 1
        assert deps[0].name == "serde"
