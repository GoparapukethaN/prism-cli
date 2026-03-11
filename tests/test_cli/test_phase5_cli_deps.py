"""Tests for the CLI 'prism deps' command — status, audit, unused."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from prism.cli.app import app

runner = CliRunner()


# ======================================================================
# Helpers
# ======================================================================


def _make_deps_report(
    *,
    total: int = 3,
    outdated: int = 1,
    vulnerable: int = 0,
    unused: int = 1,
    deps: list[object] | None = None,
    vulns: list[object] | None = None,
    unused_deps: list[str] | None = None,
) -> MagicMock:
    """Create a mock DepsReport."""
    report = MagicMock()
    report.total_deps = total
    report.outdated = outdated
    report.vulnerable = vulnerable
    report.unused = unused
    report.ecosystem = "all"
    report.generated_at = "2025-01-01T00:00:00+00:00"

    if deps is None:
        dep1 = MagicMock()
        dep1.name = "requests"
        dep1.current_version = "2.28.0"
        dep1.latest_version = "2.31.0"
        dep1.ecosystem = "python"
        dep1.usages = 5
        dep1.migration_complexity = MagicMock()
        dep1.migration_complexity.value = "simple"

        dep2 = MagicMock()
        dep2.name = "flask"
        dep2.current_version = "2.0.0"
        dep2.latest_version = "2.0.0"
        dep2.ecosystem = "python"
        dep2.usages = 3
        dep2.migration_complexity = MagicMock()
        dep2.migration_complexity.value = "trivial"

        dep3 = MagicMock()
        dep3.name = "unused-pkg"
        dep3.current_version = "1.0.0"
        dep3.latest_version = ""
        dep3.ecosystem = "python"
        dep3.usages = 0
        dep3.migration_complexity = MagicMock()
        dep3.migration_complexity.value = "trivial"

        deps = [dep1, dep2, dep3]

    report.dependencies = deps

    if vulns is None:
        vulns = []
    report.vulnerabilities = vulns

    if unused_deps is None:
        unused_deps = ["unused-pkg"]
    report.unused_deps = unused_deps

    return report


def _make_vuln(
    *,
    package: str = "requests",
    cve_id: str = "CVE-2024-1234",
    severity_value: str = "high",
    fixed_version: str = "2.32.0",
    current_version: str = "2.28.0",
) -> MagicMock:
    """Create a mock Vulnerability."""
    vuln = MagicMock()
    vuln.package = package
    vuln.cve_id = cve_id
    vuln.severity = MagicMock()
    vuln.severity.value = severity_value

    # Make severity comparison work for audit exit code
    from prism.intelligence.deps import VulnerabilitySeverity
    if severity_value == "critical":
        vuln.severity = VulnerabilitySeverity.CRITICAL
    elif severity_value == "high":
        vuln.severity = VulnerabilitySeverity.HIGH
    elif severity_value == "medium":
        vuln.severity = VulnerabilitySeverity.MEDIUM
    else:
        vuln.severity = VulnerabilitySeverity.LOW

    vuln.fixed_version = fixed_version
    vuln.current_version = current_version
    return vuln


# ======================================================================
# prism deps status
# ======================================================================


class TestDepsStatus:
    """Tests for 'prism deps status'."""

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_status_shows_summary(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = _make_deps_report()
        mock_monitor_cls.return_value = mock_monitor

        result = runner.invoke(app, ["deps", "status"])
        assert result.exit_code == 0
        assert "Dependency Health" in result.output
        assert "Total: 3" in result.output

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_status_shows_packages(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = _make_deps_report()
        mock_monitor_cls.return_value = mock_monitor

        result = runner.invoke(app, ["deps", "status"])
        assert result.exit_code == 0
        assert "requests" in result.output
        assert "flask" in result.output

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_status_default_action(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        """Running 'prism deps' with no action defaults to 'status'."""
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = _make_deps_report()
        mock_monitor_cls.return_value = mock_monitor

        result = runner.invoke(app, ["deps"])
        assert result.exit_code == 0
        assert "Dependency Health" in result.output

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_status_shows_unused(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = _make_deps_report()
        mock_monitor_cls.return_value = mock_monitor

        result = runner.invoke(app, ["deps", "status"])
        assert result.exit_code == 0
        assert "unused-pkg" in result.output

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_status_empty_project(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = _make_deps_report(
            total=0, outdated=0, vulnerable=0, unused=0,
            deps=[], vulns=[], unused_deps=[],
        )
        mock_monitor_cls.return_value = mock_monitor

        result = runner.invoke(app, ["deps", "status"])
        assert result.exit_code == 0
        assert "Total: 0" in result.output


# ======================================================================
# prism deps audit
# ======================================================================


class TestDepsAudit:
    """Tests for 'prism deps audit'."""

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_audit_no_vulns(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = _make_deps_report(
            vulnerable=0, vulns=[],
        )
        mock_monitor_cls.return_value = mock_monitor

        result = runner.invoke(app, ["deps", "audit"])
        assert result.exit_code == 0
        assert "No vulnerabilities found" in result.output

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_audit_with_high_vulns_exits_nonzero(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        vuln = _make_vuln(severity_value="high")
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = _make_deps_report(
            vulnerable=1, vulns=[vuln],
        )
        mock_monitor_cls.return_value = mock_monitor

        result = runner.invoke(app, ["deps", "audit"])
        assert result.exit_code == 1
        assert "CRITICAL/HIGH" in result.output

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_audit_with_critical_vulns_exits_nonzero(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        vuln = _make_vuln(
            severity_value="critical",
            cve_id="CVE-2024-9999",
        )
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = _make_deps_report(
            vulnerable=1, vulns=[vuln],
        )
        mock_monitor_cls.return_value = mock_monitor

        result = runner.invoke(app, ["deps", "audit"])
        assert result.exit_code == 1
        assert "CVE-2024-9999" in result.output

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_audit_with_medium_vulns_exits_zero(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        vuln = _make_vuln(severity_value="medium")
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = _make_deps_report(
            vulnerable=1, vulns=[vuln],
        )
        mock_monitor_cls.return_value = mock_monitor

        result = runner.invoke(app, ["deps", "audit"])
        assert result.exit_code == 0

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_audit_shows_total_count(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        vuln = _make_vuln(severity_value="medium")
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = _make_deps_report(
            vulnerable=1, vulns=[vuln],
        )
        mock_monitor_cls.return_value = mock_monitor

        result = runner.invoke(app, ["deps", "audit"])
        assert "Total vulnerabilities" in result.output


# ======================================================================
# prism deps unused
# ======================================================================


class TestDepsUnused:
    """Tests for 'prism deps unused'."""

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_unused_shows_packages(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = _make_deps_report(
            unused=2,
            unused_deps=["old-lib", "dead-code"],
        )
        mock_monitor_cls.return_value = mock_monitor

        result = runner.invoke(app, ["deps", "unused"])
        assert result.exit_code == 0
        assert "old-lib" in result.output
        assert "dead-code" in result.output

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_unused_none_found(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = _make_deps_report(
            unused=0, unused_deps=[],
        )
        mock_monitor_cls.return_value = mock_monitor

        result = runner.invoke(app, ["deps", "unused"])
        assert result.exit_code == 0
        assert "No unused dependencies" in result.output

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_unused_shows_note(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = _make_deps_report(
            unused=1, unused_deps=["stale-pkg"],
        )
        mock_monitor_cls.return_value = mock_monitor

        result = runner.invoke(app, ["deps", "unused"])
        assert result.exit_code == 0
        assert "Build tools" in result.output


# ======================================================================
# Error handling
# ======================================================================


class TestDepsErrorHandling:
    """Tests for error handling in deps commands."""

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_invalid_action(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        mock_monitor = MagicMock()
        mock_monitor_cls.return_value = mock_monitor

        result = runner.invoke(app, ["deps", "badaction"])
        assert result.exit_code == 1
        assert "Unknown deps action" in result.output

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_invalid_root_dir(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        mock_monitor_cls.side_effect = ValueError(
            "Project root is not a directory"
        )

        result = runner.invoke(
            app, ["deps", "status", "--root", "/tmp"],
        )
        # ValueError is caught and shows error message
        assert result.exit_code == 1

    @patch("prism.intelligence.deps.DependencyMonitor")
    def test_status_with_vulns_shows_vuln_table(
        self, mock_monitor_cls: MagicMock,
    ) -> None:
        vuln = _make_vuln()
        mock_monitor = MagicMock()
        mock_monitor.get_status.return_value = _make_deps_report(
            vulnerable=1, vulns=[vuln],
        )
        mock_monitor_cls.return_value = mock_monitor

        result = runner.invoke(app, ["deps", "status"])
        assert result.exit_code == 0
        assert "CVE-2024-1234" in result.output
        assert "Vulnerabilities" in result.output
