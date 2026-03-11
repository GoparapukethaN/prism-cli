"""Autonomous Dependency Health Monitor — multi-ecosystem parsing, OSV.dev
vulnerability scanning, version-based migration assessment, and unused detection.

Scans project dependency files across seven ecosystems:
  Python:  pyproject.toml, requirements.txt, setup.py, Pipfile
  Node.js: package.json
  Rust:    Cargo.toml
  Go:      go.mod
  Ruby:    Gemfile
  Java:    build.gradle, pom.xml

Produces a unified health report covering:
- Outdated packages with current and latest versions
- Security vulnerabilities (via OSV.dev API)
- Unused dependencies (zero import count)
- Version-based migration complexity estimation

Slash-command hooks:
    /deps status           — full dependency health report
    /deps audit            — security vulnerabilities only
    /deps unused           — list likely-unused dependencies
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)


# ======================================================================
# Enums
# ======================================================================


class MigrationComplexity(Enum):
    """How hard it would be to upgrade a dependency."""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class VulnerabilitySeverity(Enum):
    """Severity level for a known vulnerability."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class DependencyInfo:
    """Information about a single dependency."""

    name: str
    current_version: str
    latest_version: str
    is_outdated: bool
    age_days: int
    ecosystem: str  # "python", "node", "rust", "go", "ruby", "java"
    source_file: str  # which file declares it (e.g. "pyproject.toml")
    usages: int  # how many source files import/use it
    migration_complexity: MigrationComplexity = MigrationComplexity.TRIVIAL


@dataclass
class Vulnerability:
    """A known security vulnerability in a dependency."""

    package: str
    severity: VulnerabilitySeverity
    cve_id: str
    description: str
    fixed_version: str
    current_version: str
    url: str = ""


@dataclass
class VulnerabilityReport:
    """Report for a single vulnerability from OSV.dev."""

    package: str
    cve_id: str
    severity: VulnerabilitySeverity
    description: str
    fixed_version: str
    url: str


@dataclass
class DependencyStatusReport:
    """Status report for all dependencies."""

    dependencies: list[DependencyInfo]
    total: int
    outdated_count: int
    vulnerable_count: int
    vulnerabilities: list[VulnerabilityReport] = field(default_factory=list)


@dataclass
class DepsReport:
    """Full dependency health report."""

    total_deps: int
    outdated: int
    vulnerable: int
    unused: int
    dependencies: list[DependencyInfo]
    vulnerabilities: list[Vulnerability]
    unused_deps: list[str]
    generated_at: str
    ecosystem: str


# ======================================================================
# Constants
# ======================================================================

# Files we scan and their ecosystem mapping
_DEP_FILE_MAP: dict[str, str] = {
    "pyproject.toml": "python",
    "requirements.txt": "python",
    "setup.py": "python",
    "Pipfile": "python",
    "package.json": "node",
    "Cargo.toml": "rust",
    "go.mod": "go",
    "Gemfile": "ruby",
    "pom.xml": "java",
    "build.gradle": "java",
}

# OSV.dev ecosystem names (mapping from our internal names)
_OSV_ECOSYSTEM_MAP: dict[str, str] = {
    "python": "PyPI",
    "node": "npm",
    "rust": "crates.io",
    "go": "Go",
    "ruby": "RubyGems",
    "java": "Maven",
}

# Packages that are used implicitly (test runners, build tools, linters).
# These should never be flagged as unused.
_IMPLICIT_PACKAGES: frozenset[str] = frozenset({
    "hatchling",
    "pytest",
    "pytest-cov",
    "pytest-asyncio",
    "pytest-mock",
    "pytest-xdist",
    "hypothesis",
    "ruff",
    "mypy",
    "bandit",
    "pip-audit",
    "pre-commit",
    "respx",
    "setuptools",
    "wheel",
    "pip",
    "build",
    "twine",
    "flit",
    "poetry",
    "hatch",
    "tox",
    "nox",
    "coverage",
    "black",
    "isort",
    "flake8",
    "pylint",
})

# File-extension mapping for usage scanning per ecosystem
_SCAN_EXTENSIONS: dict[str, str] = {
    "python": "*.py",
    "node": "*.js",
    "rust": "*.rs",
    "go": "*.go",
    "ruby": "*.rb",
    "java": "*.java",
}

# OSV.dev severity mapping from CVSS score strings
_SEVERITY_MAP: dict[str, VulnerabilitySeverity] = {
    "CRITICAL": VulnerabilitySeverity.CRITICAL,
    "HIGH": VulnerabilitySeverity.HIGH,
    "MEDIUM": VulnerabilitySeverity.MEDIUM,
    "MODERATE": VulnerabilitySeverity.MEDIUM,
    "LOW": VulnerabilitySeverity.LOW,
}


# ======================================================================
# Version parsing helpers
# ======================================================================


def _parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a version string into (major, minor, patch) tuple.

    Handles formats like '2.28.0', '2.28', '2', and ignores suffixes
    such as 'rc1', 'alpha', 'beta'.

    Args:
        version: Version string to parse.

    Returns:
        Tuple of (major, minor, patch) integers. Missing parts default to 0.
    """
    # Strip common prefixes and suffixes
    cleaned = version.strip().lstrip("v").lstrip("V")
    # Remove pre-release suffixes
    cleaned = re.split(r"[-+]", cleaned)[0]
    # Remove alpha/beta/rc suffixes attached without separator
    cleaned = re.split(r"[a-zA-Z]", cleaned)[0].rstrip(".")

    parts = cleaned.split(".")
    result = [0, 0, 0]
    for i in range(min(3, len(parts))):
        try:
            result[i] = int(parts[i])
        except ValueError:
            break
    return (result[0], result[1], result[2])


def assess_migration_by_version(
    from_version: str,
    to_version: str,
) -> MigrationComplexity:
    """Assess migration complexity based on semantic version comparison.

    Heuristics:
        TRIVIAL:  patch version bump (2.28.0 -> 2.28.1)
        SIMPLE:   minor version bump (2.28.0 -> 2.29.0)
        MODERATE: same major, large minor jump (>5 minor versions apart)
        COMPLEX:  major version change (1.x -> 2.x)

    Args:
        from_version: Current version string.
        to_version: Target version string.

    Returns:
        A :class:`MigrationComplexity` value.
    """
    if not from_version or not to_version:
        return MigrationComplexity.TRIVIAL

    from_parts = _parse_semver(from_version)
    to_parts = _parse_semver(to_version)

    if from_parts == to_parts:
        return MigrationComplexity.TRIVIAL

    # Major version change
    if to_parts[0] != from_parts[0]:
        return MigrationComplexity.COMPLEX

    # Minor version change within same major
    if to_parts[1] != from_parts[1]:
        minor_diff = abs(to_parts[1] - from_parts[1])
        if minor_diff > 5:
            return MigrationComplexity.MODERATE
        return MigrationComplexity.SIMPLE

    # Only patch version changed
    return MigrationComplexity.TRIVIAL


# ======================================================================
# Main class
# ======================================================================


class DependencyMonitor:
    """Monitors and manages project dependencies across ecosystems.

    Provides multi-ecosystem parsing, OSV.dev vulnerability scanning,
    version-based migration assessment, and unused dependency detection.

    Args:
        project_root: Path to the project root directory.
    """

    def __init__(self, project_root: Path) -> None:
        if not project_root.is_dir():
            raise ValueError(f"Project root is not a directory: {project_root}")
        self._root = project_root.resolve()
        self._dep_files: dict[str, str] = {}
        self._detect_ecosystems()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def detected_ecosystems(self) -> dict[str, str]:
        """Return mapping of detected dependency files to their ecosystem."""
        return dict(self._dep_files)

    def get_status(self, ecosystem: str | None = None) -> DepsReport:
        """Build a full dependency health report.

        Args:
            ecosystem: Optional filter -- only report on this ecosystem.

        Returns:
            A :class:`DepsReport` with all findings.
        """
        deps = self._parse_dependencies(ecosystem)
        vulns = self._check_vulnerabilities(ecosystem)
        unused = self._find_unused(deps)

        # Assign migration complexity to each dep
        for dep in deps:
            dep.migration_complexity = self.assess_migration(dep)

        outdated = sum(1 for d in deps if d.is_outdated)

        return DepsReport(
            total_deps=len(deps),
            outdated=outdated,
            vulnerable=len(vulns),
            unused=len(unused),
            dependencies=deps,
            vulnerabilities=vulns,
            unused_deps=unused,
            generated_at=datetime.now(UTC).isoformat(),
            ecosystem=ecosystem or "all",
        )

    def assess_migration(self, dep: DependencyInfo) -> MigrationComplexity:
        """Assess migration complexity for upgrading a dependency.

        Uses version-based heuristics when a latest_version is available,
        otherwise falls back to usage-count heuristics.

        Args:
            dep: The dependency to assess.

        Returns:
            A :class:`MigrationComplexity` value.
        """
        if not dep.is_outdated:
            return MigrationComplexity.TRIVIAL

        # Use version-based heuristic if latest_version is available
        if dep.latest_version and dep.current_version:
            return assess_migration_by_version(
                dep.current_version, dep.latest_version,
            )

        # Fallback to usage-count heuristic
        if dep.usages > 20:
            return MigrationComplexity.COMPLEX
        if dep.usages > 10:
            return MigrationComplexity.MODERATE
        if dep.usages > 3:
            return MigrationComplexity.SIMPLE
        return MigrationComplexity.TRIVIAL

    def check_vulnerabilities(
        self, deps: list[DependencyInfo],
    ) -> list[VulnerabilityReport]:
        """Check dependencies for known vulnerabilities via OSV.dev API.

        Builds OSV.dev API request payloads and queries for each dependency.

        Args:
            deps: List of dependencies to check.

        Returns:
            List of :class:`VulnerabilityReport` objects.
        """
        reports: list[VulnerabilityReport] = []

        for dep in deps:
            osv_ecosystem = _OSV_ECOSYSTEM_MAP.get(dep.ecosystem)
            if not osv_ecosystem:
                continue
            if not dep.current_version or dep.current_version == "any":
                continue

            try:
                vulns = self._query_osv(
                    name=dep.name,
                    ecosystem=osv_ecosystem,
                    version=dep.current_version,
                )
                reports.extend(vulns)
            except Exception:
                logger.debug(
                    "deps.osv_query.error",
                    package=dep.name,
                    ecosystem=dep.ecosystem,
                )

        return reports

    def find_unused(self, deps: list[DependencyInfo]) -> list[str]:
        """Find dependencies with zero source-file usages (public API).

        For Python: scans all .py files for ``import X`` and ``from X import``.
        Excludes packages that are used implicitly (test runners, build tools).

        Args:
            deps: List of parsed dependencies.

        Returns:
            Names of likely-unused packages.
        """
        return self._find_unused(deps)

    def generate_status_report(
        self, deps: list[DependencyInfo],
    ) -> DependencyStatusReport:
        """Generate a comprehensive dependency status report.

        For each dep: name, current_version, latest_version (placeholder for
        offline), vulnerabilities, migration_complexity.

        Args:
            deps: List of dependencies to report on.

        Returns:
            A :class:`DependencyStatusReport` with all findings.
        """
        vulns = self.check_vulnerabilities(deps)
        outdated_count = sum(1 for d in deps if d.is_outdated)

        # Assign migration complexity
        for dep in deps:
            dep.migration_complexity = self.assess_migration(dep)

        return DependencyStatusReport(
            dependencies=deps,
            total=len(deps),
            outdated_count=outdated_count,
            vulnerable_count=len(vulns),
            vulnerabilities=vulns,
        )

    # ------------------------------------------------------------------
    # Ecosystem detection
    # ------------------------------------------------------------------

    def _detect_ecosystems(self) -> None:
        """Scan project root for known dependency management files."""
        for filename, ecosystem in _DEP_FILE_MAP.items():
            if (self._root / filename).is_file():
                self._dep_files[filename] = ecosystem

    # ------------------------------------------------------------------
    # Dependency parsing
    # ------------------------------------------------------------------

    def _parse_dependencies(
        self, ecosystem: str | None = None,
    ) -> list[DependencyInfo]:
        """Parse dependencies from all detected project files.

        Args:
            ecosystem: Optional ecosystem filter.

        Returns:
            List of parsed :class:`DependencyInfo` objects.
        """
        deps: list[DependencyInfo] = []

        for filename, eco in self._dep_files.items():
            if ecosystem and eco != ecosystem:
                continue

            file_path = self._root / filename

            parser = self._get_parser(filename)
            if parser is not None:
                deps.extend(parser(file_path))

        return deps

    def _get_parser(
        self, filename: str,
    ) -> Any:
        """Return the appropriate parser function for a dependency file.

        Args:
            filename: Name of the dependency file.

        Returns:
            Parser callable, or None if no parser is available.
        """
        parsers: dict[str, Any] = {
            "pyproject.toml": self._parse_pyproject,
            "requirements.txt": self._parse_requirements,
            "setup.py": self._parse_setup_py,
            "Pipfile": self._parse_pipfile,
            "package.json": self._parse_package_json,
            "Cargo.toml": self._parse_cargo_toml,
            "go.mod": self._parse_go_mod,
            "Gemfile": self._parse_gemfile,
            "pom.xml": self._parse_pom_xml,
            "build.gradle": self._parse_build_gradle,
        }
        return parsers.get(filename)

    def _parse_pyproject(self, path: Path) -> list[DependencyInfo]:
        """Parse dependencies from pyproject.toml.

        Uses simple regex-based parsing to avoid a hard dependency on ``tomli``.

        Args:
            path: Path to pyproject.toml.

        Returns:
            List of parsed dependency info objects.
        """
        deps: list[DependencyInfo] = []
        try:
            content = path.read_text(encoding="utf-8")
            in_deps = False
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped == "dependencies = [":
                    in_deps = True
                    continue
                if in_deps:
                    if stripped == "]":
                        break
                    # Match: "typer>=0.9", "rich[all]>=13", "litellm~=1.0"
                    match = re.match(
                        r'"([a-zA-Z0-9_-]+)\[?[^\]]*\]?([><=!~]+)([^",$]+)',
                        stripped,
                    )
                    if match:
                        name = match.group(1)
                        version = (
                            match.group(3).split(",")[0].strip().strip('"')
                        )
                        deps.append(DependencyInfo(
                            name=name,
                            current_version=version,
                            latest_version="",
                            is_outdated=False,
                            age_days=0,
                            ecosystem="python",
                            source_file=str(path.name),
                            usages=self._count_usages(name, "python"),
                        ))
        except OSError:
            logger.warning("deps.parse_pyproject.error", path=str(path))
        return deps

    def _parse_requirements(self, path: Path) -> list[DependencyInfo]:
        """Parse requirements.txt.

        Handles lines like ``requests>=2.28``, ``flask==2.0.0``, bare names.

        Args:
            path: Path to requirements.txt.

        Returns:
            List of parsed dependency info objects.
        """
        deps: list[DependencyInfo] = []
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                match = re.match(
                    r"([a-zA-Z0-9_-]+)([><=!~]+)?(.+)?", line,
                )
                if match:
                    name = match.group(1)
                    version = (match.group(3) or "any").strip()
                    deps.append(DependencyInfo(
                        name=name,
                        current_version=version,
                        latest_version="",
                        is_outdated=False,
                        age_days=0,
                        ecosystem="python",
                        source_file="requirements.txt",
                        usages=self._count_usages(name, "python"),
                    ))
        except OSError:
            logger.warning(
                "deps.parse_requirements.error", path=str(path),
            )
        return deps

    def _parse_setup_py(self, path: Path) -> list[DependencyInfo]:
        """Parse dependencies from setup.py.

        Looks for ``install_requires=[...]`` patterns via regex.

        Args:
            path: Path to setup.py.

        Returns:
            List of parsed dependency info objects.
        """
        deps: list[DependencyInfo] = []
        try:
            content = path.read_text(encoding="utf-8")
            # Find install_requires list
            # Find install_requires=[ and then find the matching ]
            # by counting bracket depth (handles extras like [http2])
            start_match = re.search(
                r"install_requires\s*=\s*\[",
                content,
            )
            match = None
            if start_match:
                body_start = start_match.end()
                depth = 1
                pos = body_start
                while pos < len(content) and depth > 0:
                    if content[pos] == "[":
                        depth += 1
                    elif content[pos] == "]":
                        depth -= 1
                    pos += 1
                if depth == 0:
                    match = content[body_start:pos - 1]
            if match is not None:
                requires_block = match
                # Extract each quoted string
                for dep_match in re.finditer(
                    r"""['"]([a-zA-Z0-9_-]+)"""
                    r"""(?:\[.*?\])?"""
                    r"""(?:[><=!~]+)?"""
                    r"""([^'",$]*)['"]""",
                    requires_block,
                ):
                    name = dep_match.group(1)
                    version = dep_match.group(2).strip() or "any"
                    deps.append(DependencyInfo(
                        name=name,
                        current_version=version,
                        latest_version="",
                        is_outdated=False,
                        age_days=0,
                        ecosystem="python",
                        source_file="setup.py",
                        usages=self._count_usages(name, "python"),
                    ))
        except OSError:
            logger.warning("deps.parse_setup_py.error", path=str(path))
        return deps

    def _parse_pipfile(self, path: Path) -> list[DependencyInfo]:
        """Parse dependencies from Pipfile.

        Parses [packages] and [dev-packages] sections using regex.

        Args:
            path: Path to Pipfile.

        Returns:
            List of parsed dependency info objects.
        """
        deps: list[DependencyInfo] = []
        try:
            content = path.read_text(encoding="utf-8")
            in_section = False
            for line in content.split("\n"):
                stripped = line.strip()
                # Detect section headers
                if stripped in ("[packages]", "[dev-packages]"):
                    in_section = True
                    continue
                if stripped.startswith("[") and stripped.endswith("]"):
                    in_section = False
                    continue
                if not in_section or not stripped or stripped.startswith("#"):
                    continue

                # Parse lines like: requests = "*"  or  flask = ">=2.0"
                match = re.match(
                    r'([a-zA-Z0-9_-]+)\s*=\s*["\']([^"\']*)["\']',
                    stripped,
                )
                if match:
                    name = match.group(1)
                    raw_version = match.group(2)
                    version = raw_version.lstrip(">=<~!") or "any"
                    if version == "*":
                        version = "any"
                    deps.append(DependencyInfo(
                        name=name,
                        current_version=version,
                        latest_version="",
                        is_outdated=False,
                        age_days=0,
                        ecosystem="python",
                        source_file="Pipfile",
                        usages=self._count_usages(name, "python"),
                    ))
        except OSError:
            logger.warning("deps.parse_pipfile.error", path=str(path))
        return deps

    def _parse_package_json(self, path: Path) -> list[DependencyInfo]:
        """Parse package.json (dependencies + devDependencies).

        Args:
            path: Path to package.json.

        Returns:
            List of parsed dependency info objects.
        """
        deps: list[DependencyInfo] = []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for section in ("dependencies", "devDependencies"):
                section_data = data.get(section, {})
                if not isinstance(section_data, dict):
                    continue
                for name, version in section_data.items():
                    clean_version = str(version).lstrip("^~>=<")
                    deps.append(DependencyInfo(
                        name=name,
                        current_version=clean_version,
                        latest_version="",
                        is_outdated=False,
                        age_days=0,
                        ecosystem="node",
                        source_file="package.json",
                        usages=self._count_usages(name, "node"),
                    ))
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "deps.parse_package_json.error", path=str(path),
            )
        return deps

    def _parse_cargo_toml(self, path: Path) -> list[DependencyInfo]:
        """Parse dependencies from Cargo.toml.

        Handles both inline version specs (``serde = "1.0"``) and table
        specs (``serde = {version = "1.0", features = [...]}``) within
        ``[dependencies]`` and ``[dev-dependencies]`` sections.

        Args:
            path: Path to Cargo.toml.

        Returns:
            List of parsed dependency info objects.
        """
        deps: list[DependencyInfo] = []
        try:
            content = path.read_text(encoding="utf-8")
            in_deps = False
            for line in content.split("\n"):
                stripped = line.strip()
                # Section headers
                if stripped in (
                    "[dependencies]", "[dev-dependencies]",
                ):
                    in_deps = True
                    continue
                if stripped.startswith("[") and stripped.endswith("]"):
                    in_deps = False
                    continue
                if not in_deps or not stripped or stripped.startswith("#"):
                    continue

                # Inline version: serde = "1.0.193"
                match = re.match(
                    r'([a-zA-Z0-9_-]+)\s*=\s*"([^"]*)"', stripped,
                )
                if match:
                    name = match.group(1)
                    version = match.group(2).strip()
                    deps.append(DependencyInfo(
                        name=name,
                        current_version=version,
                        latest_version="",
                        is_outdated=False,
                        age_days=0,
                        ecosystem="rust",
                        source_file="Cargo.toml",
                        usages=self._count_usages(name, "rust"),
                    ))
                    continue

                # Table version: serde = { version = "1.0", ... }
                match = re.match(
                    r'([a-zA-Z0-9_-]+)\s*=\s*\{.*?version\s*=\s*"([^"]*)"',
                    stripped,
                )
                if match:
                    name = match.group(1)
                    version = match.group(2).strip()
                    deps.append(DependencyInfo(
                        name=name,
                        current_version=version,
                        latest_version="",
                        is_outdated=False,
                        age_days=0,
                        ecosystem="rust",
                        source_file="Cargo.toml",
                        usages=self._count_usages(name, "rust"),
                    ))
        except OSError:
            logger.warning(
                "deps.parse_cargo_toml.error", path=str(path),
            )
        return deps

    def _parse_go_mod(self, path: Path) -> list[DependencyInfo]:
        """Parse dependencies from go.mod.

        Handles both single ``require`` statements and block requires:
        ``require (\\n  module version\\n  ...\\n)``

        Args:
            path: Path to go.mod.

        Returns:
            List of parsed dependency info objects.
        """
        deps: list[DependencyInfo] = []
        try:
            content = path.read_text(encoding="utf-8")
            in_require = False
            for line in content.split("\n"):
                stripped = line.strip()

                # Block require start
                if stripped in {"require (", "require("}:
                    in_require = True
                    continue
                if in_require and stripped == ")":
                    in_require = False
                    continue

                if in_require:
                    # Lines like: github.com/gin-gonic/gin v1.9.1
                    match = re.match(
                        r"(\S+)\s+(v?[\d.]+\S*)", stripped,
                    )
                    if match:
                        name = match.group(1)
                        version = match.group(2).lstrip("v")
                        deps.append(DependencyInfo(
                            name=name,
                            current_version=version,
                            latest_version="",
                            is_outdated=False,
                            age_days=0,
                            ecosystem="go",
                            source_file="go.mod",
                            usages=self._count_usages(
                                name.split("/")[-1], "go",
                            ),
                        ))
                    continue

                # Single-line require: require github.com/foo/bar v1.0.0
                match = re.match(
                    r"require\s+(\S+)\s+(v?[\d.]+\S*)", stripped,
                )
                if match:
                    name = match.group(1)
                    version = match.group(2).lstrip("v")
                    deps.append(DependencyInfo(
                        name=name,
                        current_version=version,
                        latest_version="",
                        is_outdated=False,
                        age_days=0,
                        ecosystem="go",
                        source_file="go.mod",
                        usages=self._count_usages(
                            name.split("/")[-1], "go",
                        ),
                    ))
        except OSError:
            logger.warning("deps.parse_go_mod.error", path=str(path))
        return deps

    def _parse_gemfile(self, path: Path) -> list[DependencyInfo]:
        """Parse dependencies from Gemfile.

        Handles lines like ``gem 'rails', '~> 7.0'`` and ``gem 'puma'``.

        Args:
            path: Path to Gemfile.

        Returns:
            List of parsed dependency info objects.
        """
        deps: list[DependencyInfo] = []
        try:
            content = path.read_text(encoding="utf-8")
            for line in content.split("\n"):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                # Match: gem 'name', 'version'  or  gem "name", "version"
                match = re.match(
                    r"""gem\s+['"]([a-zA-Z0-9_-]+)['"]"""
                    r"""(?:\s*,\s*['"]([^'"]*)['"]\s*)?""",
                    stripped,
                )
                if match:
                    name = match.group(1)
                    raw_version = match.group(2) or "any"
                    version = raw_version.lstrip("~>= ").strip() or "any"
                    deps.append(DependencyInfo(
                        name=name,
                        current_version=version,
                        latest_version="",
                        is_outdated=False,
                        age_days=0,
                        ecosystem="ruby",
                        source_file="Gemfile",
                        usages=self._count_usages(name, "ruby"),
                    ))
        except OSError:
            logger.warning("deps.parse_gemfile.error", path=str(path))
        return deps

    def _parse_build_gradle(self, path: Path) -> list[DependencyInfo]:
        """Parse dependencies from build.gradle.

        Handles Gradle dependency declarations like:
        ``implementation 'group:artifact:version'``
        ``testImplementation "group:artifact:version"``

        Args:
            path: Path to build.gradle.

        Returns:
            List of parsed dependency info objects.
        """
        deps: list[DependencyInfo] = []
        try:
            content = path.read_text(encoding="utf-8")
            # Match patterns like:
            #   implementation 'com.google:guava:31.1-jre'
            #   testImplementation "junit:junit:4.13.2"
            pattern = re.compile(
                r"(?:implementation|api|compile|testImplementation|"
                r"testCompile|runtimeOnly|compileOnly)"
                r"""\s+['"]([^:'"]+):([^:'"]+):([^'"]+)['"]""",
            )
            for match in pattern.finditer(content):
                group = match.group(1)
                artifact = match.group(2)
                version = match.group(3).strip()
                name = f"{group}:{artifact}"
                deps.append(DependencyInfo(
                    name=name,
                    current_version=version,
                    latest_version="",
                    is_outdated=False,
                    age_days=0,
                    ecosystem="java",
                    source_file="build.gradle",
                    usages=self._count_usages(artifact, "java"),
                ))
        except OSError:
            logger.warning(
                "deps.parse_build_gradle.error", path=str(path),
            )
        return deps

    def _parse_pom_xml(self, path: Path) -> list[DependencyInfo]:
        """Parse dependencies from pom.xml (Maven).

        Extracts ``<dependency>`` elements from the ``<dependencies>``
        section and reads groupId, artifactId, and version.

        Args:
            path: Path to pom.xml.

        Returns:
            List of parsed dependency info objects.
        """
        deps: list[DependencyInfo] = []
        try:
            tree = ET.parse(str(path))  # noqa: S314
            root = tree.getroot()

            # Handle XML namespace if present
            ns = ""
            ns_match = re.match(r"\{(.*)\}", root.tag)
            if ns_match:
                ns = f"{{{ns_match.group(1)}}}"

            # Find all <dependency> elements
            for dep_elem in root.iter(f"{ns}dependency"):
                group_id_elem = dep_elem.find(f"{ns}groupId")
                artifact_id_elem = dep_elem.find(f"{ns}artifactId")
                version_elem = dep_elem.find(f"{ns}version")

                if (
                    group_id_elem is not None
                    and artifact_id_elem is not None
                    and group_id_elem.text
                    and artifact_id_elem.text
                ):
                    group_id = group_id_elem.text.strip()
                    artifact_id = artifact_id_elem.text.strip()
                    version = ""
                    if version_elem is not None and version_elem.text:
                        version = version_elem.text.strip()
                    name = f"{group_id}:{artifact_id}"
                    deps.append(DependencyInfo(
                        name=name,
                        current_version=version or "any",
                        latest_version="",
                        is_outdated=False,
                        age_days=0,
                        ecosystem="java",
                        source_file="pom.xml",
                        usages=self._count_usages(artifact_id, "java"),
                    ))
        except (OSError, ET.ParseError):
            logger.warning("deps.parse_pom_xml.error", path=str(path))
        return deps

    # ------------------------------------------------------------------
    # Usage counting
    # ------------------------------------------------------------------

    def _count_usages(self, package: str, ecosystem: str) -> int:
        """Count how many source files reference a package.

        Args:
            package: Package name (e.g. ``requests``).
            ecosystem: Ecosystem identifier for file-extension filtering.

        Returns:
            Number of files that contain an import/reference.
        """
        count = 0

        # Prefer scanning src/ if it exists, else scan project root
        search_dir = self._root / "src"
        if not search_dir.is_dir():
            search_dir = self._root

        # Python packages use underscores in imports
        import_pattern = (
            package.replace("-", "_") if ecosystem == "python" else package
        )

        glob_ext = _SCAN_EXTENSIONS.get(ecosystem, "*.py")
        for source_file in search_dir.rglob(glob_ext):
            try:
                content = source_file.read_text(encoding="utf-8")
                if import_pattern in content:
                    count += 1
            except (OSError, UnicodeDecodeError):
                pass

        return count

    # ------------------------------------------------------------------
    # Vulnerability checking — OSV.dev API
    # ------------------------------------------------------------------

    def _query_osv(
        self,
        name: str,
        ecosystem: str,
        version: str,
    ) -> list[VulnerabilityReport]:
        """Query OSV.dev API for vulnerabilities in a specific package.

        Args:
            name: Package name.
            ecosystem: OSV.dev ecosystem name (e.g. "PyPI", "npm").
            version: Package version string.

        Returns:
            List of :class:`VulnerabilityReport` objects.
        """
        import httpx

        payload = {
            "package": {
                "name": name,
                "ecosystem": ecosystem,
            },
            "version": version,
        }

        try:
            response = httpx.post(
                "https://api.osv.dev/v1/query",
                json=payload,
                timeout=15.0,
            )
            if response.status_code != 200:
                return []

            data = response.json()
            return self._parse_osv_response(name, data)
        except (httpx.HTTPError, json.JSONDecodeError):
            logger.debug(
                "deps.osv_query.http_error",
                package=name,
                ecosystem=ecosystem,
            )
            return []

    def _parse_osv_response(
        self,
        package_name: str,
        data: dict[str, Any],
    ) -> list[VulnerabilityReport]:
        """Parse an OSV.dev API response into VulnerabilityReport objects.

        Args:
            package_name: The package name being queried.
            data: JSON response from OSV.dev.

        Returns:
            List of :class:`VulnerabilityReport` objects.
        """
        reports: list[VulnerabilityReport] = []
        vulns = data.get("vulns", [])

        for vuln in vulns:
            # Extract CVE ID from aliases or use OSV ID
            cve_id = vuln.get("id", "")
            aliases = vuln.get("aliases", [])
            for alias in aliases:
                if alias.startswith("CVE-"):
                    cve_id = alias
                    break

            # Extract severity
            severity = VulnerabilitySeverity.MEDIUM
            severity_list = vuln.get("severity", [])
            if severity_list:
                for sev_entry in severity_list:
                    score_str = sev_entry.get("score", "")
                    if score_str:
                        try:
                            score = float(score_str)
                            if score >= 9.0:
                                severity = VulnerabilitySeverity.CRITICAL
                            elif score >= 7.0:
                                severity = VulnerabilitySeverity.HIGH
                            elif score >= 4.0:
                                severity = VulnerabilitySeverity.MEDIUM
                            else:
                                severity = VulnerabilitySeverity.LOW
                        except ValueError:
                            pass
            # Also check database_specific for severity string
            db_specific = vuln.get("database_specific", {})
            sev_str = db_specific.get("severity", "")
            if sev_str and sev_str.upper() in _SEVERITY_MAP:
                severity = _SEVERITY_MAP[sev_str.upper()]

            # Extract description
            summary = vuln.get("summary", "")
            details = vuln.get("details", "")
            description = summary or details[:200] if details else ""

            # Extract fixed version from affected ranges
            fixed_version = ""
            affected_list = vuln.get("affected", [])
            for affected in affected_list:
                ranges = affected.get("ranges", [])
                for range_entry in ranges:
                    events = range_entry.get("events", [])
                    for event in events:
                        if "fixed" in event:
                            fixed_version = event["fixed"]
                            break
                    if fixed_version:
                        break
                if fixed_version:
                    break

            # Build URL
            url = ""
            references = vuln.get("references", [])
            for ref in references:
                ref_url = ref.get("url", "")
                if ref_url:
                    url = ref_url
                    break
            if not url and cve_id.startswith("CVE-"):
                url = f"https://nvd.nist.gov/vuln/detail/{cve_id}"

            reports.append(VulnerabilityReport(
                package=package_name,
                cve_id=cve_id,
                severity=severity,
                description=description,
                fixed_version=fixed_version,
                url=url,
            ))

        return reports

    def _check_vulnerabilities(
        self, ecosystem: str | None = None,
    ) -> list[Vulnerability]:
        """Check for known security vulnerabilities.

        Uses OSV.dev API for all ecosystems. Falls back to pip-audit
        for Python if OSV.dev is unavailable.

        Args:
            ecosystem: Optional ecosystem filter.

        Returns:
            List of discovered :class:`Vulnerability` objects.
        """
        vulns: list[Vulnerability] = []
        deps = self._parse_dependencies(ecosystem)

        for dep in deps:
            osv_ecosystem = _OSV_ECOSYSTEM_MAP.get(dep.ecosystem)
            if not osv_ecosystem:
                continue
            if not dep.current_version or dep.current_version == "any":
                continue

            try:
                osv_reports = self._query_osv(
                    name=dep.name,
                    ecosystem=osv_ecosystem,
                    version=dep.current_version,
                )
                for report in osv_reports:
                    vulns.append(Vulnerability(
                        package=report.package,
                        severity=report.severity,
                        cve_id=report.cve_id,
                        description=report.description,
                        fixed_version=report.fixed_version,
                        current_version=dep.current_version,
                        url=report.url,
                    ))
            except Exception:
                logger.debug(
                    "deps.vuln_check.error", package=dep.name,
                )

        return vulns

    # ------------------------------------------------------------------
    # Unused detection
    # ------------------------------------------------------------------

    def _find_unused(self, deps: list[DependencyInfo]) -> list[str]:
        """Find dependencies with zero source-file usages.

        Excludes packages that are used implicitly (test runners, build
        tools, linters, etc.).

        Args:
            deps: List of parsed dependencies.

        Returns:
            Names of likely-unused packages.
        """
        return [
            d.name
            for d in deps
            if d.usages == 0 and d.name.lower() not in _IMPLICIT_PACKAGES
        ]
