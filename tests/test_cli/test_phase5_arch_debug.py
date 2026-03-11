"""Tests for Phase 5 /arch, /debug-memory enhancements and /impact alias.

Covers:
- /arch mermaid — Mermaid diagram generation
- /arch check — boundary violation checking
- /arch diff — architecture diff display
- /arch (default) — existing map behavior
- /arch drift — existing drift behavior
- /debug-memory bugs / list — browse all stored fixes
- /debug-memory forget <id> — delete a fix by ID
- /debug-memory export — export fixes to JSON
- /debug-memory import <path> — import fixes from JSON
- /debug-memory stats — existing stats behavior
- /debug-memory search <query> — existing search behavior
- /impact dispatches to _cmd_blast
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

from rich.console import Console

from prism.cli.repl import (
    _dispatch_command,
    _SessionState,
)
from prism.config.schema import PrismConfig
from prism.config.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------


def _make_settings(tmp_path: Path) -> Settings:
    """Create a minimal Settings pointing at *tmp_path*."""
    config = PrismConfig(prism_home=tmp_path / ".prism")
    settings = Settings(
        config=config, project_root=tmp_path,
    )
    settings.ensure_directories()
    return settings


def _make_console(width: int = 300) -> Console:
    """In-memory console for capturing output."""
    buf = io.StringIO()
    return Console(
        file=buf,
        force_terminal=False,
        no_color=True,
        width=width,
    )


def _get_output(console: Console) -> str:
    """Extract text from an in-memory console."""
    assert isinstance(console.file, io.StringIO)
    return console.file.getvalue()


def _make_state() -> _SessionState:
    """Create a default SessionState for tests."""
    state = _SessionState(pinned_model=None)
    state.session_id = "test-session"
    return state


def _cmd(
    command: str,
    tmp_path: Path,
    console: Console | None = None,
    settings: Settings | None = None,
    state: _SessionState | None = None,
) -> tuple[str, str, Console, Settings, _SessionState]:
    """Run a slash command, return (action, output, ...)."""
    con = console or _make_console()
    stg = settings or _make_settings(tmp_path)
    st = state or _make_state()
    action = _dispatch_command(
        command,
        console=con,
        settings=stg,
        state=st,
        dry_run=False,
        offline=False,
    )
    return action, _get_output(con), con, stg, st


# -----------------------------------------------------------
# Mock data classes for Architecture
# -----------------------------------------------------------


@dataclass
class _MockModuleInfo:
    name: str = "prism.cli"
    path: str = "prism/cli/__init__.py"
    description: str = "CLI module"
    responsibilities: list[str] = field(
        default_factory=list,
    )
    public_api: list[str] = field(
        default_factory=lambda: ["run", "main"],
    )
    dependencies: list[str] = field(
        default_factory=lambda: ["prism.router"],
    )
    line_count: int = 500
    is_package: bool = True


@dataclass
class _MockDependencyEdge:
    source: str = "prism.cli"
    target: str = "prism.router"
    import_type: str = "direct"
    count: int = 1


@dataclass
class _MockDriftViolation:
    violation_type: str = "boundary_crossing"
    source: str = "tools"
    target: str = "cli"
    description: str = "tools depends on cli (boundary violation)"
    severity: str = "high"


@dataclass
class _MockArchState:
    modules: list[Any] = field(
        default_factory=lambda: [_MockModuleInfo()],
    )
    dependencies: list[Any] = field(
        default_factory=lambda: [_MockDependencyEdge()],
    )
    generated_at: str = "2026-03-11T00:00:00"
    project_root: str = "/tmp/project"
    total_lines: int = 5000
    total_modules: int = 10


# -----------------------------------------------------------
# Mock data classes for Debug Memory
# -----------------------------------------------------------


@dataclass
class _MockFixRecord:
    id: int = 1
    fingerprint: str = "abc123def456"
    error_type: str = "TypeError"
    stack_pattern: str = "NoneType has no attribute"
    fix_pattern: str = "Add None check before access"
    fix_diff: str = "- x.val\n+ x.val if x else None"
    confidence: float = 0.85
    project: str = "myapp"
    model_used: str = "gpt-4o"
    timestamp: str = "2026-03-10T12:00:00"
    language: str = "python"
    framework: str = "django"
    affected_files_json: str = '["app.py"]'
    affected_functions_json: str = '["process"]'


# ===========================================================
# TestArchMermaid
# ===========================================================


class TestArchMermaid:
    """Tests for /arch mermaid subcommand."""

    def test_arch_mermaid_generates_diagram(
        self, tmp_path: Path,
    ) -> None:
        """/arch mermaid displays a Mermaid diagram panel."""
        mock_mapper = MagicMock()
        state = _MockArchState()
        mock_mapper.generate.return_value = state
        mock_mapper.generate_mermaid.return_value = (
            "graph TD\n    cli[prism.cli]\n"
            "    router[prism.router]\n"
            "    cli --> router"
        )

        with patch(
            "prism.intelligence.architecture"
            ".ArchitectureMapper",
            return_value=mock_mapper,
        ):
            action, out, _, _, _ = _cmd(
                "/arch mermaid", tmp_path,
            )

        assert action == "continue"
        assert "Mermaid Dependency Diagram" in out
        assert "graph TD" in out
        assert "cli --> router" in out
        mock_mapper.generate.assert_called_once()
        mock_mapper.generate_mermaid.assert_called_once_with(
            state,
        )


# ===========================================================
# TestArchCheck
# ===========================================================


class TestArchCheck:
    """Tests for /arch check subcommand."""

    def test_arch_check_shows_boundary_violations(
        self, tmp_path: Path,
    ) -> None:
        """/arch check shows boundary crossing violations."""
        mock_mapper = MagicMock()
        state = _MockArchState()
        mock_mapper.generate.return_value = state

        boundary_v = _MockDriftViolation(
            violation_type="boundary_crossing",
            source="tools",
            target="cli",
            description="tools depends on cli",
            severity="high",
        )
        new_dep_v = _MockDriftViolation(
            violation_type="new_dependency",
            source="router",
            target="cost",
            description="New dependency: router -> cost",
            severity="medium",
        )
        mock_mapper.detect_drift.return_value = [
            boundary_v, new_dep_v,
        ]

        with patch(
            "prism.intelligence.architecture"
            ".ArchitectureMapper",
            return_value=mock_mapper,
        ):
            action, out, _, _, _ = _cmd(
                "/arch check", tmp_path,
            )

        assert action == "continue"
        assert "Boundary Violations" in out
        assert "tools" in out
        assert "cli" in out
        # new_dependency should not appear in boundary table
        assert "router -> cost" not in out

    def test_arch_check_no_violations(
        self, tmp_path: Path,
    ) -> None:
        """/arch check with no boundary violations shows green."""
        mock_mapper = MagicMock()
        state = _MockArchState()
        mock_mapper.generate.return_value = state

        # Only non-boundary violations
        mock_mapper.detect_drift.return_value = [
            _MockDriftViolation(
                violation_type="new_module",
                source="new_pkg",
                target=None,
                description="New module added",
                severity="low",
            ),
        ]

        with patch(
            "prism.intelligence.architecture"
            ".ArchitectureMapper",
            return_value=mock_mapper,
        ):
            action, out, _, _, _ = _cmd(
                "/arch check", tmp_path,
            )

        assert action == "continue"
        assert "No boundary violations detected" in out


# ===========================================================
# TestArchDiff
# ===========================================================


class TestArchDiff:
    """Tests for /arch diff subcommand."""

    def test_arch_diff_shows_changes(
        self, tmp_path: Path,
    ) -> None:
        """/arch diff displays a diff panel."""
        mock_mapper = MagicMock()
        mock_mapper.get_diff.return_value = (
            "Architecture changes (2 items):\n"
            "  [HIGH] New dependency: tools -> cli\n"
            "  [LOW] New module 'utils' added"
        )

        with patch(
            "prism.intelligence.architecture"
            ".ArchitectureMapper",
            return_value=mock_mapper,
        ):
            action, out, _, _, _ = _cmd(
                "/arch diff", tmp_path,
            )

        assert action == "continue"
        assert "Architecture Diff" in out
        assert "Architecture changes (2 items)" in out
        assert "HIGH" in out
        mock_mapper.get_diff.assert_called_once()

    def test_arch_diff_no_changes(
        self, tmp_path: Path,
    ) -> None:
        """/arch diff when no changes shows clean message."""
        mock_mapper = MagicMock()
        mock_mapper.get_diff.return_value = (
            "No architecture changes detected."
        )

        with patch(
            "prism.intelligence.architecture"
            ".ArchitectureMapper",
            return_value=mock_mapper,
        ):
            action, out, _, _, _ = _cmd(
                "/arch diff", tmp_path,
            )

        assert action == "continue"
        assert "No architecture changes detected" in out


# ===========================================================
# TestArchDefault
# ===========================================================


class TestArchDefault:
    """Tests for /arch default (map) behavior."""

    def test_arch_default_shows_map(
        self, tmp_path: Path,
    ) -> None:
        """/arch with no args shows architecture map."""
        mock_mapper = MagicMock()
        state = _MockArchState()
        mock_mapper.generate.return_value = state
        mock_mapper.save.return_value = (
            tmp_path / "ARCHITECTURE.md"
        )

        with patch(
            "prism.intelligence.architecture"
            ".ArchitectureMapper",
            return_value=mock_mapper,
        ):
            action, out, _, _, _ = _cmd(
                "/arch", tmp_path,
            )

        assert action == "continue"
        assert "Architecture Map" in out
        assert "10" in out  # total_modules
        assert "5,000" in out  # total_lines
        mock_mapper.generate.assert_called_once()
        mock_mapper.save.assert_called_once_with(state)

    def test_arch_drift_still_works(
        self, tmp_path: Path,
    ) -> None:
        """/arch drift continues to work as before."""
        mock_mapper = MagicMock()
        state = _MockArchState()
        mock_mapper.generate.return_value = state
        mock_mapper.detect_drift.return_value = [
            _MockDriftViolation(
                violation_type="new_module",
                source="newpkg",
                target=None,
                description="New module 'newpkg' added",
                severity="low",
            ),
        ]

        with patch(
            "prism.intelligence.architecture"
            ".ArchitectureMapper",
            return_value=mock_mapper,
        ):
            action, out, _, _, _ = _cmd(
                "/arch drift", tmp_path,
            )

        assert action == "continue"
        assert "Architecture Drift" in out
        assert "newpkg" in out

    def test_arch_drift_no_violations(
        self, tmp_path: Path,
    ) -> None:
        """/arch drift with no violations shows green message."""
        mock_mapper = MagicMock()
        state = _MockArchState()
        mock_mapper.generate.return_value = state
        mock_mapper.detect_drift.return_value = []

        with patch(
            "prism.intelligence.architecture"
            ".ArchitectureMapper",
            return_value=mock_mapper,
        ):
            action, out, _, _, _ = _cmd(
                "/arch drift", tmp_path,
            )

        assert action == "continue"
        assert "No architecture drift detected" in out


# ===========================================================
# TestDebugMemoryBugs
# ===========================================================


class TestDebugMemoryBugs:
    """Tests for /debug-memory bugs (and list) subcommand."""

    def test_debug_memory_bugs_shows_all_fixes(
        self, tmp_path: Path,
    ) -> None:
        """/debug-memory bugs displays a table of stored fixes."""
        mock_memory = MagicMock()
        mock_memory.browse_fixes.return_value = [
            _MockFixRecord(id=1, error_type="TypeError"),
            _MockFixRecord(
                id=2,
                error_type="ValueError",
                fix_pattern="Validate input",
                project="other",
                timestamp="2026-03-09T10:00:00",
            ),
        ]

        with patch(
            "prism.intelligence.debug_memory.DebugMemory",
            return_value=mock_memory,
        ):
            action, out, _, _, _ = _cmd(
                "/debug-memory bugs", tmp_path,
            )

        assert action == "continue"
        assert "All Stored Fixes" in out
        assert "TypeError" in out
        assert "ValueError" in out
        assert "Validate input" in out
        mock_memory.browse_fixes.assert_called_once_with(
            limit=100,
        )
        mock_memory.close.assert_called_once()

    def test_debug_memory_list_alias(
        self, tmp_path: Path,
    ) -> None:
        """/debug-memory list is an alias for bugs."""
        mock_memory = MagicMock()
        mock_memory.browse_fixes.return_value = [
            _MockFixRecord(),
        ]

        with patch(
            "prism.intelligence.debug_memory.DebugMemory",
            return_value=mock_memory,
        ):
            action, out, _, _, _ = _cmd(
                "/debug-memory list", tmp_path,
            )

        assert action == "continue"
        assert "All Stored Fixes" in out
        mock_memory.browse_fixes.assert_called_once_with(
            limit=100,
        )

    def test_debug_memory_bugs_empty(
        self, tmp_path: Path,
    ) -> None:
        """/debug-memory bugs with no fixes shows dim message."""
        mock_memory = MagicMock()
        mock_memory.browse_fixes.return_value = []

        with patch(
            "prism.intelligence.debug_memory.DebugMemory",
            return_value=mock_memory,
        ):
            action, out, _, _, _ = _cmd(
                "/debug-memory bugs", tmp_path,
            )

        assert action == "continue"
        assert "No fixes stored yet" in out


# ===========================================================
# TestDebugMemoryForget
# ===========================================================


class TestDebugMemoryForget:
    """Tests for /debug-memory forget <id> subcommand."""

    def test_forget_existing_fix(
        self, tmp_path: Path,
    ) -> None:
        """/debug-memory forget 1 deletes fix #1."""
        mock_memory = MagicMock()
        mock_memory.forget.return_value = True

        with patch(
            "prism.intelligence.debug_memory.DebugMemory",
            return_value=mock_memory,
        ):
            action, out, _, _, _ = _cmd(
                "/debug-memory forget 1", tmp_path,
            )

        assert action == "continue"
        assert "Fix #1 deleted" in out
        mock_memory.forget.assert_called_once_with(1)

    def test_forget_nonexistent_fix(
        self, tmp_path: Path,
    ) -> None:
        """/debug-memory forget 999 shows not-found message."""
        mock_memory = MagicMock()
        mock_memory.forget.return_value = False

        with patch(
            "prism.intelligence.debug_memory.DebugMemory",
            return_value=mock_memory,
        ):
            action, out, _, _, _ = _cmd(
                "/debug-memory forget 999", tmp_path,
            )

        assert action == "continue"
        assert "Fix #999 not found" in out

    def test_forget_invalid_id(
        self, tmp_path: Path,
    ) -> None:
        """/debug-memory forget abc shows invalid ID error."""
        mock_memory = MagicMock()

        with patch(
            "prism.intelligence.debug_memory.DebugMemory",
            return_value=mock_memory,
        ):
            action, out, _, _, _ = _cmd(
                "/debug-memory forget abc", tmp_path,
            )

        assert action == "continue"
        assert "Invalid fix ID" in out


# ===========================================================
# TestDebugMemoryExport
# ===========================================================


class TestDebugMemoryExport:
    """Tests for /debug-memory export subcommand."""

    def test_export_creates_json_file(
        self, tmp_path: Path,
    ) -> None:
        """/debug-memory export writes fixes to JSON."""
        mock_memory = MagicMock()
        mock_memory.browse_fixes.return_value = [
            _MockFixRecord(id=1),
            _MockFixRecord(id=2, error_type="ValueError"),
        ]

        with patch(
            "prism.intelligence.debug_memory.DebugMemory",
            return_value=mock_memory,
        ):
            action, out, _, stg, _ = _cmd(
                "/debug-memory export", tmp_path,
            )

        assert action == "continue"
        assert "Exported 2 fixes" in out

        export_path = stg.prism_home / "debug_memory_export.json"
        assert export_path.is_file()

        data = json.loads(
            export_path.read_text(encoding="utf-8"),
        )
        assert len(data) == 2
        assert data[0]["id"] == 1
        assert data[1]["error_type"] == "ValueError"
        mock_memory.browse_fixes.assert_called_once_with(
            limit=10000,
        )

    def test_export_empty_database(
        self, tmp_path: Path,
    ) -> None:
        """/debug-memory export with no fixes creates empty array."""
        mock_memory = MagicMock()
        mock_memory.browse_fixes.return_value = []

        with patch(
            "prism.intelligence.debug_memory.DebugMemory",
            return_value=mock_memory,
        ):
            action, out, _, stg, _ = _cmd(
                "/debug-memory export", tmp_path,
            )

        assert action == "continue"
        assert "Exported 0 fixes" in out

        export_path = stg.prism_home / "debug_memory_export.json"
        data = json.loads(
            export_path.read_text(encoding="utf-8"),
        )
        assert data == []


# ===========================================================
# TestDebugMemoryImport
# ===========================================================


class TestDebugMemoryImport:
    """Tests for /debug-memory import <path> subcommand."""

    def test_import_valid_json(
        self, tmp_path: Path,
    ) -> None:
        """/debug-memory import loads fixes from a JSON file."""
        import_data = [
            {
                "fingerprint": "aaa111",
                "error_type": "KeyError",
                "stack_pattern": "key missing",
                "fix_pattern": "Use .get() default",
                "fix_diff": "- d[k]\n+ d.get(k, None)",
                "confidence": 0.9,
                "project": "webapp",
                "model_used": "claude-3",
                "language": "python",
                "framework": "flask",
                "affected_files_json": '["views.py"]',
                "affected_functions_json": '["index"]',
            },
        ]
        import_file = tmp_path / "import_fixes.json"
        import_file.write_text(
            json.dumps(import_data), encoding="utf-8",
        )

        mock_memory = MagicMock()

        with patch(
            "prism.intelligence.debug_memory.DebugMemory",
            return_value=mock_memory,
        ), patch(
            "prism.intelligence.debug_memory.BugFingerprint",
        ) as mock_fp_cls:
            mock_fp_cls.return_value = MagicMock()
            action, out, _, _, _ = _cmd(
                f"/debug-memory import {import_file}",
                tmp_path,
            )

        assert action == "continue"
        assert "Imported 1 fixes" in out
        mock_memory.store_fix.assert_called_once()

    def test_import_file_not_found(
        self, tmp_path: Path,
    ) -> None:
        """/debug-memory import with missing file shows error."""
        mock_memory = MagicMock()

        with patch(
            "prism.intelligence.debug_memory.DebugMemory",
            return_value=mock_memory,
        ):
            action, out, _, _, _ = _cmd(
                "/debug-memory import /nonexistent/file.json",
                tmp_path,
            )

        assert action == "continue"
        assert "File not found" in out

    def test_import_invalid_json(
        self, tmp_path: Path,
    ) -> None:
        """/debug-memory import with malformed JSON shows error."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{", encoding="utf-8")

        mock_memory = MagicMock()

        with patch(
            "prism.intelligence.debug_memory.DebugMemory",
            return_value=mock_memory,
        ):
            action, out, _, _, _ = _cmd(
                f"/debug-memory import {bad_file}",
                tmp_path,
            )

        assert action == "continue"
        assert "Import error" in out

    def test_import_not_array(
        self, tmp_path: Path,
    ) -> None:
        """/debug-memory import rejects non-array JSON."""
        obj_file = tmp_path / "obj.json"
        obj_file.write_text(
            '{"key": "value"}', encoding="utf-8",
        )

        mock_memory = MagicMock()

        with patch(
            "prism.intelligence.debug_memory.DebugMemory",
            return_value=mock_memory,
        ):
            action, out, _, _, _ = _cmd(
                f"/debug-memory import {obj_file}",
                tmp_path,
            )

        assert action == "continue"
        assert "Invalid format" in out


# ===========================================================
# TestDebugMemoryExisting
# ===========================================================


class TestDebugMemoryExisting:
    """Verify existing /debug-memory subcommands still work."""

    def test_debug_memory_stats_default(
        self, tmp_path: Path,
    ) -> None:
        """/debug-memory with no args shows stats."""
        mock_memory = MagicMock()
        mock_memory.get_stats.return_value = {
            "total_fixes": 42,
            "projects": 3,
            "error_types": 7,
            "avg_confidence": 0.756,
        }

        with patch(
            "prism.intelligence.debug_memory.DebugMemory",
            return_value=mock_memory,
        ):
            action, out, _, _, _ = _cmd(
                "/debug-memory", tmp_path,
            )

        assert action == "continue"
        assert "Debug Memory" in out
        assert "42" in out
        assert "76%" in out
        mock_memory.get_stats.assert_called_once()

    def test_debug_memory_search(
        self, tmp_path: Path,
    ) -> None:
        """/debug-memory search TypeError finds matching fixes."""
        mock_memory = MagicMock()
        mock_memory.search_by_description.return_value = [
            _MockFixRecord(
                id=5, error_type="TypeError",
                fix_pattern="Add type check",
            ),
        ]

        with patch(
            "prism.intelligence.debug_memory.DebugMemory",
            return_value=mock_memory,
        ):
            action, out, _, _, _ = _cmd(
                "/debug-memory search TypeError",
                tmp_path,
            )

        assert action == "continue"
        assert "TypeError" in out
        assert "Add type check" in out
        mock_memory.search_by_description.assert_called_once_with(
            "TypeError",
        )

    def test_debug_memory_search_no_results(
        self, tmp_path: Path,
    ) -> None:
        """/debug-memory search with no matches shows dim text."""
        mock_memory = MagicMock()
        mock_memory.search_by_description.return_value = []

        with patch(
            "prism.intelligence.debug_memory.DebugMemory",
            return_value=mock_memory,
        ):
            action, out, _, _, _ = _cmd(
                "/debug-memory search ZZZ", tmp_path,
            )

        assert action == "continue"
        assert "No fixes found" in out


# ===========================================================
# TestImpactAlias
# ===========================================================


class TestImpactAlias:
    """Tests for /impact as an alias for /blast."""

    def test_impact_dispatches_to_blast(
        self, tmp_path: Path,
    ) -> None:
        """/impact routes to _cmd_blast handler."""
        mock_analyzer = MagicMock()

        @dataclass
        class _MockBlastReport:
            description: str = "change file"
            risk_score: int = 25
            estimated_complexity: str = "low"
            affected_files: list[Any] = field(
                default_factory=list,
            )
            missing_tests: list[str] = field(
                default_factory=list,
            )
            recommended_test_order: list[str] = field(
                default_factory=list,
            )
            execution_order: list[str] = field(
                default_factory=list,
            )
            critical_paths: list[str] = field(
                default_factory=list,
            )
            created_at: str = "2026-03-11"

            @property
            def file_count(self) -> int:
                return len(self.affected_files)

        mock_analyzer.analyze.return_value = _MockBlastReport()

        with patch(
            "prism.intelligence.blast_radius"
            ".BlastRadiusAnalyzer",
            return_value=mock_analyzer,
        ):
            action, out, _, _, _ = _cmd(
                "/impact some_file.py", tmp_path,
            )

        assert action == "continue"
        assert "Blast Radius Report" in out
        mock_analyzer.analyze.assert_called_once()

    def test_impact_no_args_shows_usage(
        self, tmp_path: Path,
    ) -> None:
        """/impact with no args shows usage (same as /blast)."""
        action, out, _, _, _ = _cmd(
            "/impact", tmp_path,
        )

        assert action == "continue"
        assert "Usage" in out
