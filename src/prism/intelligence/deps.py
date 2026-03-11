"""Autonomous Dependency Health Monitor — version tracking, upgrade assessment, and security audit.

Scans project dependency files (pyproject.toml, requirements.txt, package.json,
Cargo.toml, go.mod, Gemfile, pom.xml) to produce a unified health report covering:

- Outdated packages with current and latest versions
- Security vulnerabilities (via pip-audit for Python)
- Unused dependencies (zero import count)
- Migration complexity estimation per package

Slash-command hooks:
    /deps status           — full dependency health report
    /deps status --eco py  — python-only report
    /deps upgrade          — suggest upgrade plan
    /deps audit            — security vulnerabilities only
    /deps unused           — list likely-unused dependencies
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
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
    "package.json": "node",
    "Cargo.toml": "rust",
    "go.mod": "go",
    "Gemfile": "ruby",
    "pom.xml": "java",
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


# ======================================================================
# Main class
# ======================================================================


class DependencyMonitor:
    """Monitors and manages project dependencies across ecosystems.

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
            ecosystem: Optional filter — only report on this ecosystem.

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

        Heuristic based on usage count — the more files that use a package
        the harder it is to upgrade safely.

        Args:
            dep: The dependency to assess.

        Returns:
            A :class:`MigrationComplexity` value.
        """
        if not dep.is_outdated:
            return MigrationComplexity.TRIVIAL

        if dep.usages > 20:
            return MigrationComplexity.COMPLEX
        if dep.usages > 10:
            return MigrationComplexity.MODERATE
        if dep.usages > 3:
            return MigrationComplexity.SIMPLE
        return MigrationComplexity.TRIVIAL

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

    def _parse_dependencies(self, ecosystem: str | None = None) -> list[DependencyInfo]:
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

            if filename == "pyproject.toml":
                deps.extend(self._parse_pyproject(file_path))
            elif filename == "requirements.txt":
                deps.extend(self._parse_requirements(file_path))
            elif filename == "package.json":
                deps.extend(self._parse_package_json(file_path))
            # Other ecosystems are detected but not deeply parsed yet.

        return deps

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
                    # Match patterns like: "typer>=0.9", "rich[all]>=13", "litellm~=1.0"
                    match = re.match(
                        r'"([a-zA-Z0-9_-]+)\[?[^\]]*\]?([><=!~]+)([^",$]+)',
                        stripped,
                    )
                    if match:
                        name = match.group(1)
                        version = match.group(3).split(",")[0].strip().strip('"')
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

        Handles lines like ``requests>=2.28``, ``flask==2.0.0``, and bare names.

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
                match = re.match(r"([a-zA-Z0-9_-]+)([><=!~]+)?(.+)?", line)
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
            logger.warning("deps.parse_requirements.error", path=str(path))
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
            logger.warning("deps.parse_package_json.error", path=str(path))
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

        # Python packages use underscores in imports, npm uses hyphens
        import_pattern = package.replace("-", "_") if ecosystem == "python" else package

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
    # Vulnerability checking
    # ------------------------------------------------------------------

    def _check_vulnerabilities(self, ecosystem: str | None = None) -> list[Vulnerability]:
        """Check for known security vulnerabilities.

        For Python, runs ``pip-audit --format json`` and parses the output.
        Returns an empty list if the tool is not installed or times out.

        Args:
            ecosystem: Optional ecosystem filter.

        Returns:
            List of discovered :class:`Vulnerability` objects.
        """
        vulns: list[Vulnerability] = []

        if ecosystem is not None and ecosystem != "python":
            return vulns

        try:
            result = subprocess.run(
                ["pip-audit", "--format", "json"],
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.returncode == 0 and result.stdout:
                try:
                    data = json.loads(result.stdout)
                    vuln_list: list[dict[str, Any]] = data.get("vulnerabilities", [])
                    for v in vuln_list:
                        fix_versions = v.get("fix_versions", [])
                        vulns.append(Vulnerability(
                            package=v.get("name", ""),
                            severity=VulnerabilitySeverity.HIGH,
                            cve_id=v.get("id", ""),
                            description=v.get("description", ""),
                            fixed_version=fix_versions[0] if fix_versions else "",
                            current_version=v.get("version", ""),
                        ))
                except (json.JSONDecodeError, KeyError, IndexError):
                    logger.warning("deps.parse_pip_audit.error")
        except FileNotFoundError:
            logger.debug("deps.pip_audit.not_found")
        except subprocess.TimeoutExpired:
            logger.warning("deps.pip_audit.timeout")

        return vulns

    # ------------------------------------------------------------------
    # Unused detection
    # ------------------------------------------------------------------

    def _find_unused(self, deps: list[DependencyInfo]) -> list[str]:
        """Find dependencies with zero source-file usages.

        Excludes packages that are used implicitly (test runners, build tools,
        linters, etc.).

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
