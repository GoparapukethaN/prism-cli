"""Living Architecture Map — auto-generated codebase architecture with drift detection.

Provides:
- :class:`ArchitectureMapper` — scans a Python codebase and generates an architecture
  document (ARCHITECTURE.md) with module inventory, dependency graph (Mermaid), drift
  detection, and diff reporting.
- Data classes: :class:`ModuleInfo`, :class:`DependencyEdge`, :class:`DriftViolation`,
  :class:`ArchitectureState`.

Usage::

    mapper = ArchitectureMapper(project_root=Path("/path/to/project"))
    state = mapper.generate()
    mapper.save(state)
    violations = mapper.detect_drift(state)
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)


@dataclass
class ModuleInfo:
    """Information about a single Python module in the codebase."""

    name: str
    path: str
    description: str
    responsibilities: list[str]
    public_api: list[str]
    dependencies: list[str]
    line_count: int
    is_package: bool


@dataclass
class DependencyEdge:
    """A directed dependency edge from one module to another."""

    source: str
    target: str
    import_type: str  # "direct", "from", "conditional"
    count: int = 1


@dataclass
class DriftViolation:
    """A detected architectural drift violation."""

    violation_type: str  # "new_dependency", "boundary_crossing", "new_module", "removed_module"
    source: str
    target: str | None
    description: str
    severity: str  # "high", "medium", "low"


@dataclass
class ArchitectureState:
    """Complete snapshot of the codebase architecture at a point in time."""

    modules: list[ModuleInfo]
    dependencies: list[DependencyEdge]
    generated_at: str
    project_root: str
    total_lines: int
    total_modules: int


# ---------------------------------------------------------------------------
# Allowed dependency boundaries: layer → set of layers it may depend on.
# Violations crossing these boundaries are flagged as "high" severity.
# ---------------------------------------------------------------------------

_BOUNDARY_RULES: dict[str, set[str]] = {
    "cli": {"router", "providers", "tools", "context", "auth", "db", "cost", "git",
            "security", "config", "intelligence", "llm", "architect", "network",
            "logging_config", "exceptions", "plugins", "cache", "workspace"},
    "router": {"providers", "config", "cost", "db", "exceptions", "logging_config"},
    "providers": {"config", "auth", "exceptions", "logging_config"},
    "tools": {"security", "config", "exceptions", "logging_config"},
    "context": {"config", "db", "exceptions", "logging_config"},
    "auth": {"config", "exceptions", "logging_config"},
    "db": {"config", "exceptions", "logging_config"},
    "cost": {"db", "config", "exceptions", "logging_config"},
    "git": {"config", "exceptions", "logging_config", "security"},
    "security": {"config", "exceptions", "logging_config"},
    "config": {"exceptions", "logging_config"},
}


class ArchitectureMapper:
    """Auto-generates and maintains a living architecture map of the codebase.

    Args:
        project_root: Absolute path to the project root directory.
        src_dir: Relative name of the source directory (default ``"src"``).
    """

    def __init__(self, project_root: Path, src_dir: str = "src") -> None:
        self._root = project_root.resolve()
        self._src = self._root / src_dir
        self._arch_file = self._root / "ARCHITECTURE.md"
        self._state_file = self._root / ".prism" / "architecture_state.json"
        self._previous_state: ArchitectureState | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> ArchitectureState:
        """Scan the codebase and produce a fresh :class:`ArchitectureState`."""
        modules = self._scan_modules()
        dependencies = self._build_dependency_graph(modules)

        state = ArchitectureState(
            modules=modules,
            dependencies=dependencies,
            generated_at=datetime.now(UTC).isoformat(),
            project_root=str(self._root),
            total_lines=sum(m.line_count for m in modules),
            total_modules=len(modules),
        )

        self._previous_state = self._load_previous_state()
        self._save_state(state)
        logger.info(
            "architecture.generated",
            total_modules=state.total_modules,
            total_lines=state.total_lines,
        )
        return state

    def save(self, state: ArchitectureState) -> Path:
        """Write ``ARCHITECTURE.md`` from *state* and return its path."""
        lines = self._render_markdown(state)
        self._arch_file.write_text("\n".join(lines))
        logger.info("architecture.saved", path=str(self._arch_file))
        return self._arch_file

    def detect_drift(self, current: ArchitectureState) -> list[DriftViolation]:
        """Compare *current* state to the previously saved state and return violations."""
        previous = self._previous_state or self._load_previous_state()
        if previous is None:
            return []

        violations: list[DriftViolation] = []

        prev_modules = {m.name for m in previous.modules}
        curr_modules = {m.name for m in current.modules}

        # New modules ---------------------------------------------------------
        for name in sorted(curr_modules - prev_modules):
            violations.append(
                DriftViolation(
                    violation_type="new_module",
                    source=name,
                    target=None,
                    description=f"New module '{name}' added",
                    severity="low",
                )
            )

        # Removed modules -----------------------------------------------------
        for name in sorted(prev_modules - curr_modules):
            violations.append(
                DriftViolation(
                    violation_type="removed_module",
                    source=name,
                    target=None,
                    description=f"Module '{name}' removed",
                    severity="medium",
                )
            )

        # New dependencies -----------------------------------------------------
        prev_deps = {(d.source, d.target) for d in previous.dependencies}
        curr_deps = {(d.source, d.target) for d in current.dependencies}

        for src, tgt in sorted(curr_deps - prev_deps):
            severity = self._boundary_severity(src, tgt)
            violations.append(
                DriftViolation(
                    violation_type="new_dependency" if severity != "high" else "boundary_crossing",
                    source=src,
                    target=tgt,
                    description=f"New dependency: {src} -> {tgt}",
                    severity=severity,
                )
            )

        return violations

    def get_diff(self) -> str:
        """Return a human-readable diff summary between current and previous state."""
        current = self.generate()
        violations = self.detect_drift(current)

        if not violations:
            return "No architecture changes detected."

        lines = [f"Architecture changes ({len(violations)} items):"]
        for v in violations:
            lines.append(f"  [{v.severity.upper()}] {v.description}")
        return "\n".join(lines)

    def get_dependency_graph(self) -> list[DependencyEdge]:
        """Build and return the import/dependency graph for the codebase."""
        modules = self._scan_modules()
        return self._build_dependency_graph(modules)

    def generate_mermaid(self, state: ArchitectureState) -> str:
        """Generate a Mermaid dependency diagram from *state*."""
        lines = ["graph TD"]

        seen: set[str] = set()
        for m in state.modules:
            node_id = self._mermaid_id(m.name)
            if node_id not in seen:
                lines.append(f"    {node_id}[{m.name}]")
                seen.add(node_id)

        for dep in state.dependencies:
            src = self._mermaid_id(dep.source)
            tgt = self._mermaid_id(dep.target)
            if src in seen and tgt in seen:
                lines.append(f"    {src} --> {tgt}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Scanning helpers
    # ------------------------------------------------------------------

    def _scan_modules(self) -> list[ModuleInfo]:
        """Walk the source tree and extract :class:`ModuleInfo` for every Python file."""
        modules: list[ModuleInfo] = []

        if not self._src.is_dir():
            logger.warning("architecture.src_dir_missing", path=str(self._src))
            return modules

        for py_file in sorted(self._src.rglob("*.py")):
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue

            rel = py_file.relative_to(self._src)
            module_name = str(rel).replace("/", ".").replace(".py", "")
            if module_name.endswith(".__init__"):
                module_name = module_name[:-9]

            try:
                content = py_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                logger.warning("architecture.read_failed", path=str(py_file))
                continue

            line_count = content.count("\n") + 1
            public_api = self._extract_public_api(content)
            deps = self._extract_imports(content)
            desc = self._extract_docstring(content)

            modules.append(
                ModuleInfo(
                    name=module_name,
                    path=str(rel),
                    description=desc,
                    responsibilities=[],
                    public_api=public_api,
                    dependencies=deps,
                    line_count=line_count,
                    is_package=(py_file.name == "__init__.py"),
                )
            )

        return modules

    def _build_dependency_graph(self, modules: list[ModuleInfo]) -> list[DependencyEdge]:
        """Aggregate per-module imports into a list of unique :class:`DependencyEdge`."""
        edges: dict[tuple[str, str], DependencyEdge] = {}

        for module in modules:
            for dep in module.dependencies:
                key = (module.name, dep)
                if key not in edges:
                    edges[key] = DependencyEdge(
                        source=module.name, target=dep, import_type="direct"
                    )
                else:
                    edges[key].count += 1

        return list(edges.values())

    # ------------------------------------------------------------------
    # AST extraction helpers (all static)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_public_api(content: str) -> list[str]:
        """Return public function and class names defined in *content*."""
        api: list[str] = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return api

        for node in ast.iter_child_nodes(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and not node.name.startswith("_")
            ):
                api.append(node.name)
        return api

    @staticmethod
    def _extract_imports(content: str) -> list[str]:
        """Return deduplicated import target names from *content*."""
        imports: list[str] = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        return sorted(set(imports))

    @staticmethod
    def _extract_docstring(content: str) -> str:
        """Return the first line (up to 200 chars) of the module docstring."""
        try:
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)
            if docstring:
                return docstring.split("\n")[0][:200]
        except SyntaxError:
            pass
        return ""

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_markdown(self, state: ArchitectureState) -> list[str]:
        """Render *state* as a list of markdown lines for ``ARCHITECTURE.md``."""
        lines = [
            "# Architecture Map",
            "",
            f"*Auto-generated on {state.generated_at[:10]}*",
            "",
            f"**{state.total_modules} modules** | **{state.total_lines:,} lines**",
            "",
            "## Module Inventory",
            "",
            "| Module | Lines | Public API | Dependencies |",
            "|--------|-------|------------|--------------|",
        ]

        for m in sorted(state.modules, key=lambda x: x.name):
            api_count = len(m.public_api)
            dep_count = len(m.dependencies)
            desc = f" — {m.description}" if m.description else ""
            lines.append(
                f"| {m.name}{desc} | {m.line_count} | {api_count} | {dep_count} |"
            )

        lines.extend(["", "## Dependency Graph", "", "```mermaid"])
        lines.append(self.generate_mermaid(state))
        lines.extend(["```", ""])

        return lines

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self, state: ArchitectureState) -> None:
        """Serialize *state* to ``.prism/architecture_state.json``."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(json.dumps(asdict(state), indent=2))

    def _load_previous_state(self) -> ArchitectureState | None:
        """Deserialize previously saved state, or return ``None``."""
        if not self._state_file.is_file():
            return None
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            return ArchitectureState(
                modules=[ModuleInfo(**m) for m in data["modules"]],
                dependencies=[DependencyEdge(**d) for d in data["dependencies"]],
                generated_at=data["generated_at"],
                project_root=data["project_root"],
                total_lines=data["total_lines"],
                total_modules=data["total_modules"],
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("architecture.load_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Boundary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _boundary_severity(source: str, target: str) -> str:
        """Determine severity of a new dependency based on boundary rules.

        Returns ``"high"`` if the dependency crosses a documented boundary,
        ``"medium"`` otherwise.
        """
        src_layer = source.split(".", maxsplit=1)[0] if "." in source else source
        tgt_layer = target.split(".", maxsplit=1)[0] if "." in target else target

        allowed = _BOUNDARY_RULES.get(src_layer)
        if allowed is not None and tgt_layer not in allowed and tgt_layer != src_layer:
            return "high"
        return "medium"

    @staticmethod
    def _mermaid_id(name: str) -> str:
        """Sanitize *name* into a valid Mermaid node identifier."""
        return name.replace(".", "_").replace("/", "_").replace("-", "_")
