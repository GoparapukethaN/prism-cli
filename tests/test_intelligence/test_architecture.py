"""Tests for prism.intelligence.architecture — Living Architecture Map."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from prism.intelligence.architecture import (
    ArchitectureMapper,
    ArchitectureState,
    DependencyEdge,
    DriftViolation,
    ModuleInfo,
)

if TYPE_CHECKING:
    from pathlib import Path


# =========================================================================
# Data-class unit tests
# =========================================================================


class TestModuleInfo:
    """Tests for the ModuleInfo dataclass."""

    def test_fields(self) -> None:
        m = ModuleInfo(
            name="foo.bar",
            path="foo/bar.py",
            description="A test module",
            responsibilities=["testing"],
            public_api=["do_thing"],
            dependencies=["os", "sys"],
            line_count=42,
            is_package=False,
        )
        assert m.name == "foo.bar"
        assert m.path == "foo/bar.py"
        assert m.description == "A test module"
        assert m.responsibilities == ["testing"]
        assert m.public_api == ["do_thing"]
        assert m.dependencies == ["os", "sys"]
        assert m.line_count == 42
        assert m.is_package is False

    def test_package_flag(self) -> None:
        m = ModuleInfo(
            name="pkg",
            path="pkg/__init__.py",
            description="",
            responsibilities=[],
            public_api=[],
            dependencies=[],
            line_count=1,
            is_package=True,
        )
        assert m.is_package is True

    def test_empty_fields(self) -> None:
        m = ModuleInfo(
            name="x",
            path="x.py",
            description="",
            responsibilities=[],
            public_api=[],
            dependencies=[],
            line_count=0,
            is_package=False,
        )
        assert m.public_api == []
        assert m.dependencies == []
        assert m.line_count == 0


class TestDependencyEdge:
    """Tests for the DependencyEdge dataclass."""

    def test_fields(self) -> None:
        e = DependencyEdge(source="a", target="b", import_type="direct")
        assert e.source == "a"
        assert e.target == "b"
        assert e.import_type == "direct"

    def test_default_count(self) -> None:
        e = DependencyEdge(source="x", target="y", import_type="from")
        assert e.count == 1

    def test_custom_count(self) -> None:
        e = DependencyEdge(source="x", target="y", import_type="from", count=5)
        assert e.count == 5


class TestDriftViolation:
    """Tests for the DriftViolation dataclass."""

    def test_fields(self) -> None:
        v = DriftViolation(
            violation_type="new_module",
            source="foo",
            target=None,
            description="New module 'foo' added",
            severity="low",
        )
        assert v.violation_type == "new_module"
        assert v.source == "foo"
        assert v.target is None
        assert v.description == "New module 'foo' added"
        assert v.severity == "low"

    def test_severity_values(self) -> None:
        for sev in ("high", "medium", "low"):
            v = DriftViolation(
                violation_type="test", source="s", target="t",
                description="d", severity=sev,
            )
            assert v.severity == sev

    def test_violation_types(self) -> None:
        for vtype in ("new_dependency", "boundary_crossing", "new_module", "removed_module"):
            v = DriftViolation(
                violation_type=vtype, source="s", target="t",
                description="d", severity="low",
            )
            assert v.violation_type == vtype


class TestArchitectureState:
    """Tests for the ArchitectureState dataclass."""

    def test_fields(self) -> None:
        s = ArchitectureState(
            modules=[],
            dependencies=[],
            generated_at="2026-01-01T00:00:00+00:00",
            project_root="/tmp/proj",
            total_lines=100,
            total_modules=5,
        )
        assert s.modules == []
        assert s.dependencies == []
        assert s.generated_at.startswith("2026")
        assert s.project_root == "/tmp/proj"
        assert s.total_lines == 100
        assert s.total_modules == 5

    def test_with_modules(self) -> None:
        m = ModuleInfo(
            name="a", path="a.py", description="", responsibilities=[],
            public_api=[], dependencies=[], line_count=10, is_package=False,
        )
        s = ArchitectureState(
            modules=[m], dependencies=[], generated_at="t",
            project_root="/p", total_lines=10, total_modules=1,
        )
        assert len(s.modules) == 1
        assert s.modules[0].name == "a"


# =========================================================================
# ArchitectureMapper tests
# =========================================================================


class TestArchitectureMapper:
    """Tests for the ArchitectureMapper class."""

    @pytest.fixture()
    def sample_project(self, tmp_path: Path) -> Path:
        """Create a minimal project directory with Python files for scanning."""
        src = tmp_path / "src"
        src.mkdir()

        # Package: mylib
        pkg = src / "mylib"
        pkg.mkdir()
        (pkg / "__init__.py").write_text('"""mylib — a test library."""\n\nfrom mylib.utils import helper\n')
        (pkg / "utils.py").write_text(
            '"""Utility helpers."""\n\nimport os\nimport json\n\n\n'
            'def helper() -> str:\n    """Return hello."""\n    return "hello"\n\n\n'
            'def _private() -> None:\n    pass\n'
        )
        (pkg / "core.py").write_text(
            '"""Core logic."""\n\nfrom mylib.utils import helper\nimport sys\n\n\n'
            'class Engine:\n    """Main engine."""\n\n'
            '    def run(self) -> None:\n        helper()\n'
        )

        # A file with syntax error (should be skipped gracefully)
        (pkg / "broken.py").write_text("def bad(:\n")

        # A private module (should be skipped — name starts with _)
        (pkg / "_internal.py").write_text("SECRET = 42\n")

        # Subdirectory
        sub = pkg / "sub"
        sub.mkdir()
        (sub / "__init__.py").write_text("")
        (sub / "leaf.py").write_text(
            '"""Leaf module."""\n\nfrom mylib.core import Engine\n\n\n'
            'class Leaf:\n    pass\n'
        )

        # .prism dir
        (tmp_path / ".prism").mkdir()

        return tmp_path

    # --- init ---

    def test_init_paths(self, tmp_path: Path) -> None:
        mapper = ArchitectureMapper(tmp_path)
        assert mapper._root == tmp_path.resolve()
        assert mapper._src == tmp_path.resolve() / "src"
        assert mapper._arch_file == tmp_path.resolve() / "ARCHITECTURE.md"

    def test_init_custom_src(self, tmp_path: Path) -> None:
        mapper = ArchitectureMapper(tmp_path, src_dir="lib")
        assert mapper._src == tmp_path.resolve() / "lib"

    # --- _extract_public_api ---

    def test_extract_public_api_functions_and_classes(self) -> None:
        code = (
            "def public_func():\n    pass\n\n"
            "def _private_func():\n    pass\n\n"
            "class PublicClass:\n    pass\n\n"
            "class _PrivateClass:\n    pass\n"
        )
        api = ArchitectureMapper._extract_public_api(code)
        assert "public_func" in api
        assert "PublicClass" in api
        assert "_private_func" not in api
        assert "_PrivateClass" not in api

    def test_extract_public_api_async(self) -> None:
        code = "async def fetch_data():\n    pass\n"
        api = ArchitectureMapper._extract_public_api(code)
        assert "fetch_data" in api

    def test_extract_public_api_syntax_error(self) -> None:
        api = ArchitectureMapper._extract_public_api("def bad(:\n")
        assert api == []

    def test_extract_public_api_empty(self) -> None:
        api = ArchitectureMapper._extract_public_api("")
        assert api == []

    # --- _extract_imports ---

    def test_extract_imports_basic(self) -> None:
        code = "import os\nimport json\nfrom pathlib import Path\n"
        imports = ArchitectureMapper._extract_imports(code)
        assert "os" in imports
        assert "json" in imports
        assert "pathlib" in imports

    def test_extract_imports_deduplicated(self) -> None:
        code = "import os\nimport os\n"
        imports = ArchitectureMapper._extract_imports(code)
        assert imports.count("os") == 1

    def test_extract_imports_syntax_error(self) -> None:
        imports = ArchitectureMapper._extract_imports("from (\n")
        assert imports == []

    def test_extract_imports_from_import(self) -> None:
        code = "from mylib.utils import helper\n"
        imports = ArchitectureMapper._extract_imports(code)
        assert "mylib.utils" in imports

    def test_extract_imports_empty(self) -> None:
        imports = ArchitectureMapper._extract_imports("")
        assert imports == []

    # --- _extract_docstring ---

    def test_extract_docstring_present(self) -> None:
        code = '"""Module docstring."""\n\nimport os\n'
        doc = ArchitectureMapper._extract_docstring(code)
        assert doc == "Module docstring."

    def test_extract_docstring_multiline(self) -> None:
        code = '"""First line.\n\nSecond paragraph.\n"""\n'
        doc = ArchitectureMapper._extract_docstring(code)
        assert doc == "First line."

    def test_extract_docstring_absent(self) -> None:
        doc = ArchitectureMapper._extract_docstring("import os\n")
        assert doc == ""

    def test_extract_docstring_syntax_error(self) -> None:
        doc = ArchitectureMapper._extract_docstring("def bad(:\n")
        assert doc == ""

    def test_extract_docstring_truncated(self) -> None:
        long_doc = '"""' + "A" * 300 + '"""\n'
        doc = ArchitectureMapper._extract_docstring(long_doc)
        assert len(doc) <= 200

    # --- _scan_modules ---

    def test_scan_modules_finds_files(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        modules = mapper._scan_modules()
        names = [m.name for m in modules]
        assert "mylib" in names  # __init__.py
        assert "mylib.utils" in names
        assert "mylib.core" in names
        assert "mylib.sub" in names  # sub/__init__.py
        assert "mylib.sub.leaf" in names

    def test_scan_modules_skips_private(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        modules = mapper._scan_modules()
        names = [m.name for m in modules]
        # _internal.py should be skipped
        assert all("_internal" not in n for n in names)

    def test_scan_modules_skips_syntax_errors_gracefully(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        modules = mapper._scan_modules()
        # broken.py has a syntax error — it should still appear (read succeeds)
        # but public_api will be empty
        broken = [m for m in modules if "broken" in m.name]
        assert len(broken) == 1
        assert broken[0].public_api == []

    def test_scan_modules_empty_src(self, tmp_path: Path) -> None:
        # src dir doesn't exist
        mapper = ArchitectureMapper(tmp_path)
        modules = mapper._scan_modules()
        assert modules == []

    def test_scan_modules_line_count(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        modules = mapper._scan_modules()
        for m in modules:
            assert m.line_count > 0

    def test_scan_modules_is_package(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        modules = mapper._scan_modules()
        init_modules = [m for m in modules if m.is_package]
        non_init = [m for m in modules if not m.is_package]
        assert len(init_modules) >= 2  # mylib and mylib.sub
        assert len(non_init) >= 3  # utils, core, broken, leaf

    # --- _build_dependency_graph ---

    def test_build_dependency_graph(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        modules = mapper._scan_modules()
        edges = mapper._build_dependency_graph(modules)
        assert len(edges) > 0
        sources = {e.source for e in edges}
        targets = {e.target for e in edges}
        assert "mylib.utils" in sources  # utils imports os, json
        assert "os" in targets

    def test_build_dependency_graph_dedup(self) -> None:
        modules = [
            ModuleInfo(
                name="a", path="a.py", description="", responsibilities=[],
                public_api=[], dependencies=["x", "x"], line_count=1, is_package=False,
            ),
        ]
        mapper = ArchitectureMapper.__new__(ArchitectureMapper)
        edges = mapper._build_dependency_graph(modules)
        # "x" appears twice in deps, but should produce 1 edge with count=2
        a_to_x = [e for e in edges if e.source == "a" and e.target == "x"]
        assert len(a_to_x) == 1
        assert a_to_x[0].count == 2

    # --- get_dependency_graph ---

    def test_get_dependency_graph(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        edges = mapper.get_dependency_graph()
        assert isinstance(edges, list)
        assert all(isinstance(e, DependencyEdge) for e in edges)

    # --- generate_mermaid ---

    def test_generate_mermaid_basic(self) -> None:
        state = ArchitectureState(
            modules=[
                ModuleInfo(
                    name="a", path="a.py", description="", responsibilities=[],
                    public_api=[], dependencies=[], line_count=1, is_package=False,
                ),
                ModuleInfo(
                    name="b", path="b.py", description="", responsibilities=[],
                    public_api=[], dependencies=[], line_count=1, is_package=False,
                ),
            ],
            dependencies=[DependencyEdge(source="a", target="b", import_type="direct")],
            generated_at="t",
            project_root="/p",
            total_lines=2,
            total_modules=2,
        )
        mapper = ArchitectureMapper.__new__(ArchitectureMapper)
        mermaid = mapper.generate_mermaid(state)
        assert "graph TD" in mermaid
        assert "a[a]" in mermaid
        assert "b[b]" in mermaid
        assert "a --> b" in mermaid

    def test_generate_mermaid_dotted_names(self) -> None:
        state = ArchitectureState(
            modules=[
                ModuleInfo(
                    name="foo.bar", path="foo/bar.py", description="", responsibilities=[],
                    public_api=[], dependencies=[], line_count=1, is_package=False,
                ),
            ],
            dependencies=[],
            generated_at="t",
            project_root="/p",
            total_lines=1,
            total_modules=1,
        )
        mapper = ArchitectureMapper.__new__(ArchitectureMapper)
        mermaid = mapper.generate_mermaid(state)
        assert "foo_bar[foo.bar]" in mermaid

    def test_generate_mermaid_no_edge_for_missing_node(self) -> None:
        state = ArchitectureState(
            modules=[
                ModuleInfo(
                    name="a", path="a.py", description="", responsibilities=[],
                    public_api=[], dependencies=[], line_count=1, is_package=False,
                ),
            ],
            dependencies=[DependencyEdge(source="a", target="missing", import_type="direct")],
            generated_at="t",
            project_root="/p",
            total_lines=1,
            total_modules=1,
        )
        mapper = ArchitectureMapper.__new__(ArchitectureMapper)
        mermaid = mapper.generate_mermaid(state)
        assert "missing" not in mermaid.split("\n")[-1]  # edge should not appear

    # --- _render_markdown ---

    def test_render_markdown_contains_sections(self) -> None:
        state = ArchitectureState(
            modules=[
                ModuleInfo(
                    name="mod", path="mod.py", description="A module",
                    responsibilities=[], public_api=["func"], dependencies=["os"],
                    line_count=20, is_package=False,
                ),
            ],
            dependencies=[],
            generated_at="2026-03-11T00:00:00+00:00",
            project_root="/p",
            total_lines=20,
            total_modules=1,
        )
        mapper = ArchitectureMapper.__new__(ArchitectureMapper)
        lines = mapper._render_markdown(state)
        text = "\n".join(lines)
        assert "# Architecture Map" in text
        assert "## Module Inventory" in text
        assert "## Dependency Graph" in text
        assert "```mermaid" in text
        assert "mod" in text
        assert "2026-03-11" in text

    def test_render_markdown_table_row(self) -> None:
        state = ArchitectureState(
            modules=[
                ModuleInfo(
                    name="alpha", path="alpha.py", description="Alpha desc",
                    responsibilities=[], public_api=["a", "b"], dependencies=["os"],
                    line_count=50, is_package=False,
                ),
            ],
            dependencies=[],
            generated_at="t",
            project_root="/p",
            total_lines=50,
            total_modules=1,
        )
        mapper = ArchitectureMapper.__new__(ArchitectureMapper)
        lines = mapper._render_markdown(state)
        text = "\n".join(lines)
        # Check table row has module name, description, counts
        assert "alpha" in text
        assert "Alpha desc" in text
        assert "50" in text
        assert "2" in text  # 2 public API items

    # --- save ---

    def test_save_creates_file(self, tmp_path: Path) -> None:
        mapper = ArchitectureMapper(tmp_path)
        state = ArchitectureState(
            modules=[], dependencies=[], generated_at="t",
            project_root=str(tmp_path), total_lines=0, total_modules=0,
        )
        path = mapper.save(state)
        assert path.exists()
        content = path.read_text()
        assert "# Architecture Map" in content

    def test_save_overwrites(self, tmp_path: Path) -> None:
        mapper = ArchitectureMapper(tmp_path)
        state1 = ArchitectureState(
            modules=[], dependencies=[], generated_at="first",
            project_root=str(tmp_path), total_lines=0, total_modules=0,
        )
        state2 = ArchitectureState(
            modules=[], dependencies=[], generated_at="second",
            project_root=str(tmp_path), total_lines=0, total_modules=0,
        )
        mapper.save(state1)
        path = mapper.save(state2)
        content = path.read_text()
        assert "secon" in content  # generated_at[:10] of "second" = "second"

    # --- generate (full integration) ---

    def test_generate_full(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        state = mapper.generate()
        assert state.total_modules > 0
        assert state.total_lines > 0
        assert len(state.modules) > 0
        assert state.generated_at != ""
        assert state.project_root == str(sample_project.resolve())

    def test_generate_saves_state_file(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        mapper.generate()
        assert mapper._state_file.exists()
        data = json.loads(mapper._state_file.read_text())
        assert "modules" in data
        assert "dependencies" in data

    # --- detect_drift ---

    def test_detect_drift_no_previous(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        state = mapper.generate()
        # First generation — no previous state to compare
        mapper._previous_state = None
        # Remove state file so _load_previous_state returns None
        if mapper._state_file.exists():
            mapper._state_file.unlink()
        violations = mapper.detect_drift(state)
        assert violations == []

    def test_detect_drift_new_module(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        mapper.generate()

        # Add a new module
        (sample_project / "src" / "mylib" / "newmod.py").write_text(
            '"""New module."""\n\ndef new_func():\n    pass\n'
        )
        state2 = mapper.generate()

        violations = mapper.detect_drift(state2)
        new_module_violations = [v for v in violations if v.violation_type == "new_module"]
        assert len(new_module_violations) >= 1
        names = [v.source for v in new_module_violations]
        assert any("newmod" in n for n in names)

    def test_detect_drift_removed_module(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        mapper.generate()

        # Remove a module
        (sample_project / "src" / "mylib" / "utils.py").unlink()
        state2 = mapper.generate()

        violations = mapper.detect_drift(state2)
        removed = [v for v in violations if v.violation_type == "removed_module"]
        assert len(removed) >= 1
        names = [v.source for v in removed]
        assert any("utils" in n for n in names)

    def test_detect_drift_new_dependency(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        mapper.generate()  # baseline

        # Add a new import to core.py
        core = sample_project / "src" / "mylib" / "core.py"
        content = core.read_text()
        core.write_text("import hashlib\n" + content)

        state2 = mapper.generate()
        violations = mapper.detect_drift(state2)
        dep_violations = [
            v for v in violations
            if v.violation_type in ("new_dependency", "boundary_crossing")
        ]
        assert len(dep_violations) >= 1
        targets = [v.target for v in dep_violations]
        assert "hashlib" in targets

    # --- get_diff ---

    def test_get_diff_no_changes(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        mapper.generate()
        # Second call — no changes
        diff = mapper.get_diff()
        assert diff == "No architecture changes detected."

    def test_get_diff_with_changes(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        mapper.generate()

        # Add new module
        (sample_project / "src" / "mylib" / "extra.py").write_text("x = 1\n")
        diff = mapper.get_diff()
        assert "Architecture changes" in diff
        assert "extra" in diff

    # --- save/load state round-trip ---

    def test_state_round_trip(self, sample_project: Path) -> None:
        mapper = ArchitectureMapper(sample_project)
        state = mapper.generate()

        loaded = mapper._load_previous_state()
        assert loaded is not None
        assert loaded.total_modules == state.total_modules
        assert loaded.total_lines == state.total_lines
        assert len(loaded.modules) == len(state.modules)
        assert len(loaded.dependencies) == len(state.dependencies)

    def test_load_previous_state_missing_file(self, tmp_path: Path) -> None:
        mapper = ArchitectureMapper(tmp_path)
        result = mapper._load_previous_state()
        assert result is None

    def test_load_previous_state_corrupted_json(self, tmp_path: Path) -> None:
        mapper = ArchitectureMapper(tmp_path)
        mapper._state_file.parent.mkdir(parents=True, exist_ok=True)
        mapper._state_file.write_text("{bad json")
        result = mapper._load_previous_state()
        assert result is None

    def test_load_previous_state_missing_keys(self, tmp_path: Path) -> None:
        mapper = ArchitectureMapper(tmp_path)
        mapper._state_file.parent.mkdir(parents=True, exist_ok=True)
        mapper._state_file.write_text('{"modules": []}')
        result = mapper._load_previous_state()
        assert result is None

    # --- _boundary_severity ---

    def test_boundary_severity_allowed(self) -> None:
        # cli -> router is allowed
        sev = ArchitectureMapper._boundary_severity("cli.app", "router.selector")
        assert sev == "medium"

    def test_boundary_severity_crossing(self) -> None:
        # db -> cli is NOT in the allowed set for "db"
        sev = ArchitectureMapper._boundary_severity("db.models", "cli.app")
        assert sev == "high"

    def test_boundary_severity_same_layer(self) -> None:
        # same layer -> always medium (not a crossing)
        sev = ArchitectureMapper._boundary_severity("config.settings", "config.schema")
        assert sev == "medium"

    def test_boundary_severity_unknown_layer(self) -> None:
        # unknown layers not in rules -> medium
        sev = ArchitectureMapper._boundary_severity("unknown.x", "other.y")
        assert sev == "medium"

    # --- _mermaid_id ---

    def test_mermaid_id_dots(self) -> None:
        assert ArchitectureMapper._mermaid_id("foo.bar.baz") == "foo_bar_baz"

    def test_mermaid_id_slashes(self) -> None:
        assert ArchitectureMapper._mermaid_id("foo/bar") == "foo_bar"

    def test_mermaid_id_dashes(self) -> None:
        assert ArchitectureMapper._mermaid_id("my-module") == "my_module"

    def test_mermaid_id_plain(self) -> None:
        assert ArchitectureMapper._mermaid_id("simple") == "simple"
