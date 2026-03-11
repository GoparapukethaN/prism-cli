"""Tests for WorkspaceManager, ProjectInfo, and WorkspaceState."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import pytest

from prism.workspace.manager import ProjectInfo, WorkspaceManager, WorkspaceState

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# ProjectInfo tests
# ---------------------------------------------------------------------------


class TestProjectInfo:
    """Tests for ProjectInfo dataclass."""

    def test_required_fields(self) -> None:
        """ProjectInfo can be created with required fields only."""
        project = ProjectInfo(
            name="my-project",
            path="/tmp/my-project",
            created_at="2026-01-01T00:00:00+00:00",
            last_accessed="2026-01-01T00:00:00+00:00",
        )
        assert project.name == "my-project"
        assert project.path == "/tmp/my-project"
        assert project.description == ""
        assert project.active is False

    def test_all_fields(self) -> None:
        """ProjectInfo can be created with all fields."""
        project = ProjectInfo(
            name="my-project",
            path="/tmp/my-project",
            created_at="2026-01-01T00:00:00+00:00",
            last_accessed="2026-01-02T00:00:00+00:00",
            description="Test project",
            active=True,
        )
        assert project.description == "Test project"
        assert project.active is True
        assert project.last_accessed == "2026-01-02T00:00:00+00:00"

    def test_fields_are_mutable(self) -> None:
        """ProjectInfo fields can be mutated."""
        project = ProjectInfo(
            name="a",
            path="/tmp/a",
            created_at="2026-01-01T00:00:00+00:00",
            last_accessed="2026-01-01T00:00:00+00:00",
        )
        project.active = True
        assert project.active is True
        project.description = "Updated"
        assert project.description == "Updated"


# ---------------------------------------------------------------------------
# WorkspaceState tests
# ---------------------------------------------------------------------------


class TestWorkspaceState:
    """Tests for WorkspaceState dataclass."""

    def test_defaults(self) -> None:
        """WorkspaceState defaults to empty projects and no active project."""
        state = WorkspaceState()
        assert state.projects == {}
        assert state.active_project is None
        assert state.version == 1

    def test_with_projects(self) -> None:
        """WorkspaceState can be created with projects."""
        p = ProjectInfo(
            name="test",
            path="/tmp/test",
            created_at="2026-01-01T00:00:00+00:00",
            last_accessed="2026-01-01T00:00:00+00:00",
        )
        state = WorkspaceState(projects={"test": p}, active_project="test")
        assert "test" in state.projects
        assert state.active_project == "test"


# ---------------------------------------------------------------------------
# WorkspaceManager tests
# ---------------------------------------------------------------------------


class TestWorkspaceManager:
    """Tests for WorkspaceManager."""

    def test_init_creates_empty_state(self, tmp_path: Path) -> None:
        """New manager starts with an empty workspace state."""
        manager = WorkspaceManager(tmp_path / ".prism")
        assert manager.list_projects() == []
        assert manager.get_active_project() is None

    def test_init_creates_prism_home(self, tmp_path: Path) -> None:
        """Manager creates prism home on first save."""
        prism_home = tmp_path / ".prism"
        manager = WorkspaceManager(prism_home)
        # Create a project so the state file gets written
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        manager.register_project("p1", project_dir)
        assert prism_home.is_dir()
        assert (prism_home / "workspace.json").is_file()

    def test_register_project(self, tmp_path: Path) -> None:
        """register_project creates and returns a ProjectInfo."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "my-project"
        project_dir.mkdir()

        project = manager.register_project("my-project", project_dir, description="A test")
        assert project.name == "my-project"
        assert project.path == str(project_dir.resolve())
        assert project.description == "A test"
        assert project.created_at != ""
        assert project.last_accessed != ""

    def test_register_project_creates_prism_dir(self, tmp_path: Path) -> None:
        """register_project creates .prism/ inside the project root."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        manager.register_project("proj", project_dir)
        assert (project_dir / ".prism").is_dir()

    def test_register_duplicate_raises(self, tmp_path: Path) -> None:
        """Registering a project with the same name raises ValueError."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        manager.register_project("proj", project_dir)
        with pytest.raises(ValueError, match="already exists"):
            manager.register_project("proj", project_dir)

    def test_register_duplicate_path_raises(self, tmp_path: Path) -> None:
        """Registering a different project with the same path raises ValueError."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        manager.register_project("proj-a", project_dir)
        with pytest.raises(ValueError, match="already registered"):
            manager.register_project("proj-b", project_dir)

    def test_register_invalid_name_raises(self, tmp_path: Path) -> None:
        """Invalid project names raise ValueError."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        with pytest.raises(ValueError, match="Invalid project name"):
            manager.register_project("has spaces", project_dir)

        with pytest.raises(ValueError, match="Invalid project name"):
            manager.register_project("has/slashes", project_dir)

        with pytest.raises(ValueError, match="Invalid project name"):
            manager.register_project("", project_dir)

    def test_register_name_starting_with_hyphen_raises(self, tmp_path: Path) -> None:
        """Names starting with hyphen are rejected."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        with pytest.raises(ValueError, match="Invalid project name"):
            manager.register_project("-bad-start", project_dir)

    def test_register_missing_path_raises(self, tmp_path: Path) -> None:
        """Registering with a non-existent path raises ValueError."""
        manager = WorkspaceManager(tmp_path / ".prism")

        with pytest.raises(ValueError, match="does not exist"):
            manager.register_project("ghost", tmp_path / "nonexistent")

    def test_register_file_path_raises(self, tmp_path: Path) -> None:
        """Registering with a file path (not dir) raises ValueError."""
        manager = WorkspaceManager(tmp_path / ".prism")
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        with pytest.raises(ValueError, match="does not exist or is not a directory"):
            manager.register_project("file-proj", file_path)

    def test_first_project_auto_active(self, tmp_path: Path) -> None:
        """The first registered project becomes the active project."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "first"
        project_dir.mkdir()

        project = manager.register_project("first", project_dir)
        assert project.active is True
        assert manager.get_active_project() is not None
        assert manager.get_active_project().name == "first"

    def test_second_project_not_auto_active(self, tmp_path: Path) -> None:
        """The second registered project does not become active."""
        manager = WorkspaceManager(tmp_path / ".prism")
        dir1 = tmp_path / "first"
        dir1.mkdir()
        dir2 = tmp_path / "second"
        dir2.mkdir()

        manager.register_project("first", dir1)
        second = manager.register_project("second", dir2)
        assert second.active is False
        assert manager.get_active_project().name == "first"

    def test_remove_project(self, tmp_path: Path) -> None:
        """remove_project removes the project from the workspace."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        manager.register_project("proj", project_dir)
        assert len(manager.list_projects()) == 1

        manager.remove_project("proj")
        assert len(manager.list_projects()) == 0

    def test_remove_nonexistent_raises(self, tmp_path: Path) -> None:
        """Removing a non-existent project raises ValueError."""
        manager = WorkspaceManager(tmp_path / ".prism")
        with pytest.raises(ValueError, match="not found"):
            manager.remove_project("ghost")

    def test_remove_active_switches_to_next(self, tmp_path: Path) -> None:
        """Removing the active project switches to the most recent."""
        manager = WorkspaceManager(tmp_path / ".prism")
        dir1 = tmp_path / "first"
        dir1.mkdir()
        dir2 = tmp_path / "second"
        dir2.mkdir()

        manager.register_project("first", dir1)
        manager.register_project("second", dir2)
        assert manager.get_active_project().name == "first"

        manager.remove_project("first")
        active = manager.get_active_project()
        assert active is not None
        assert active.name == "second"
        assert active.active is True

    def test_remove_last_project_clears_active(self, tmp_path: Path) -> None:
        """Removing the last project clears the active project."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "only"
        project_dir.mkdir()

        manager.register_project("only", project_dir)
        manager.remove_project("only")
        assert manager.get_active_project() is None

    def test_switch_project(self, tmp_path: Path) -> None:
        """switch_project changes the active project."""
        manager = WorkspaceManager(tmp_path / ".prism")
        dir1 = tmp_path / "first"
        dir1.mkdir()
        dir2 = tmp_path / "second"
        dir2.mkdir()

        manager.register_project("first", dir1)
        manager.register_project("second", dir2)
        assert manager.get_active_project().name == "first"

        result = manager.switch_project("second")
        assert result.name == "second"
        assert result.active is True
        assert manager.get_active_project().name == "second"

        # First should no longer be active
        first = manager.get_project("first")
        assert first.active is False

    def test_switch_nonexistent_raises(self, tmp_path: Path) -> None:
        """Switching to a non-existent project raises ValueError."""
        manager = WorkspaceManager(tmp_path / ".prism")
        with pytest.raises(ValueError, match="not found"):
            manager.switch_project("ghost")

    def test_switch_updates_last_accessed(self, tmp_path: Path) -> None:
        """Switching a project updates its last_accessed timestamp."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        project = manager.register_project("proj", project_dir)
        original_accessed = project.last_accessed

        time.sleep(0.01)  # Ensure timestamp differs
        manager.switch_project("proj")
        updated = manager.get_project("proj")
        assert updated.last_accessed >= original_accessed

    def test_list_projects_sorted_by_access_time(self, tmp_path: Path) -> None:
        """list_projects returns projects sorted by last_accessed (newest first)."""
        manager = WorkspaceManager(tmp_path / ".prism")
        dirs = []
        for name in ["alpha", "beta", "gamma"]:
            d = tmp_path / name
            d.mkdir()
            dirs.append(d)
            time.sleep(0.01)
            manager.register_project(name, d)

        projects = manager.list_projects()
        # Most recently registered (gamma) should be first
        assert projects[0].name == "gamma"

        # Now switch to alpha — it should come first
        time.sleep(0.01)
        manager.switch_project("alpha")
        projects = manager.list_projects()
        assert projects[0].name == "alpha"

    def test_get_active_project_none_when_empty(self, tmp_path: Path) -> None:
        """get_active_project returns None for an empty workspace."""
        manager = WorkspaceManager(tmp_path / ".prism")
        assert manager.get_active_project() is None

    def test_get_project(self, tmp_path: Path) -> None:
        """get_project returns the correct ProjectInfo."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        manager.register_project("proj", project_dir, description="test")
        project = manager.get_project("proj")
        assert project.name == "proj"
        assert project.description == "test"

    def test_get_project_nonexistent_raises(self, tmp_path: Path) -> None:
        """get_project raises ValueError for missing projects."""
        manager = WorkspaceManager(tmp_path / ".prism")
        with pytest.raises(ValueError, match="not found"):
            manager.get_project("ghost")

    def test_get_project_config_path(self, tmp_path: Path) -> None:
        """get_project_config_path returns the correct .prism.yaml path."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        manager.register_project("proj", project_dir)
        config_path = manager.get_project_config_path("proj")
        assert config_path == project_dir.resolve() / ".prism.yaml"

    def test_get_project_memory_path(self, tmp_path: Path) -> None:
        """get_project_memory_path returns the correct .prism.md path."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        manager.register_project("proj", project_dir)
        memory_path = manager.get_project_memory_path("proj")
        assert memory_path == project_dir.resolve() / ".prism.md"

    def test_get_project_history_dir(self, tmp_path: Path) -> None:
        """get_project_history_dir returns and creates the history directory."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        manager.register_project("proj", project_dir)
        history_dir = manager.get_project_history_dir("proj")
        assert history_dir.is_dir()
        assert history_dir == project_dir.resolve() / ".prism" / "history"

    def test_get_project_cost_dir(self, tmp_path: Path) -> None:
        """get_project_cost_dir returns and creates the cost directory."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        manager.register_project("proj", project_dir)
        cost_dir = manager.get_project_cost_dir("proj")
        assert cost_dir.is_dir()
        assert cost_dir == project_dir.resolve() / ".prism" / "costs"

    def test_get_project_ignore_path(self, tmp_path: Path) -> None:
        """get_project_ignore_path returns the correct .prismignore path."""
        manager = WorkspaceManager(tmp_path / ".prism")
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        manager.register_project("proj", project_dir)
        ignore_path = manager.get_project_ignore_path("proj")
        assert ignore_path == project_dir.resolve() / ".prismignore"

    def test_get_recent_projects(self, tmp_path: Path) -> None:
        """get_recent_projects returns at most `limit` projects."""
        manager = WorkspaceManager(tmp_path / ".prism")
        for i in range(5):
            d = tmp_path / f"proj{i}"
            d.mkdir()
            time.sleep(0.01)
            manager.register_project(f"proj{i}", d)

        recent = manager.get_recent_projects(limit=3)
        assert len(recent) == 3

    def test_get_recent_projects_limit_zero(self, tmp_path: Path) -> None:
        """get_recent_projects with limit=0 returns nothing."""
        manager = WorkspaceManager(tmp_path / ".prism")
        d = tmp_path / "proj"
        d.mkdir()
        manager.register_project("proj", d)

        recent = manager.get_recent_projects(limit=0)
        assert len(recent) == 0

    def test_get_recent_projects_limit_negative(self, tmp_path: Path) -> None:
        """get_recent_projects with negative limit returns nothing."""
        manager = WorkspaceManager(tmp_path / ".prism")
        d = tmp_path / "proj"
        d.mkdir()
        manager.register_project("proj", d)

        recent = manager.get_recent_projects(limit=-1)
        assert len(recent) == 0

    def test_project_exists(self, tmp_path: Path) -> None:
        """project_exists returns True for registered projects."""
        manager = WorkspaceManager(tmp_path / ".prism")
        d = tmp_path / "proj"
        d.mkdir()
        manager.register_project("proj", d)

        assert manager.project_exists("proj") is True
        assert manager.project_exists("ghost") is False

    def test_update_last_accessed(self, tmp_path: Path) -> None:
        """update_last_accessed changes the timestamp."""
        manager = WorkspaceManager(tmp_path / ".prism")
        d = tmp_path / "proj"
        d.mkdir()
        manager.register_project("proj", d)

        original = manager.get_project("proj").last_accessed
        time.sleep(0.01)
        manager.update_last_accessed("proj")
        updated = manager.get_project("proj").last_accessed
        assert updated >= original

    def test_update_last_accessed_nonexistent_raises(self, tmp_path: Path) -> None:
        """update_last_accessed raises ValueError for missing projects."""
        manager = WorkspaceManager(tmp_path / ".prism")
        with pytest.raises(ValueError, match="not found"):
            manager.update_last_accessed("ghost")

    # ------------------------------------------------------------------
    # Persistence tests
    # ------------------------------------------------------------------

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        """State survives save/load cycle."""
        prism_home = tmp_path / ".prism"

        # Create manager and register a project
        manager1 = WorkspaceManager(prism_home)
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        manager1.register_project("proj", project_dir, description="Persisted")

        # Create a new manager from the same state file
        manager2 = WorkspaceManager(prism_home)
        projects = manager2.list_projects()
        assert len(projects) == 1
        assert projects[0].name == "proj"
        assert projects[0].description == "Persisted"
        assert manager2.get_active_project().name == "proj"

    def test_corrupt_state_file_returns_empty(self, tmp_path: Path) -> None:
        """Corrupt workspace.json results in an empty state."""
        prism_home = tmp_path / ".prism"
        prism_home.mkdir()
        state_file = prism_home / "workspace.json"
        state_file.write_text("not valid json {{{")

        manager = WorkspaceManager(prism_home)
        assert manager.list_projects() == []
        assert manager.get_active_project() is None

    def test_missing_state_file_returns_empty(self, tmp_path: Path) -> None:
        """Missing workspace.json results in an empty state."""
        manager = WorkspaceManager(tmp_path / ".prism")
        assert manager.list_projects() == []

    def test_state_file_format(self, tmp_path: Path) -> None:
        """The workspace.json file has the expected structure."""
        prism_home = tmp_path / ".prism"
        manager = WorkspaceManager(prism_home)
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        manager.register_project("proj", project_dir)

        state_file = prism_home / "workspace.json"
        data = json.loads(state_file.read_text())
        assert "projects" in data
        assert "active_project" in data
        assert "version" in data
        assert data["version"] == 1
        assert "proj" in data["projects"]
        assert data["active_project"] == "proj"

    def test_register_with_valid_special_chars(self, tmp_path: Path) -> None:
        """Names with hyphens and underscores are accepted."""
        manager = WorkspaceManager(tmp_path / ".prism")
        d1 = tmp_path / "proj1"
        d1.mkdir()
        d2 = tmp_path / "proj2"
        d2.mkdir()

        p1 = manager.register_project("my-project", d1)
        p2 = manager.register_project("my_project2", d2)
        assert p1.name == "my-project"
        assert p2.name == "my_project2"

    def test_active_project_stale_reference(self, tmp_path: Path) -> None:
        """get_active_project returns None if active_project name is stale."""
        prism_home = tmp_path / ".prism"
        prism_home.mkdir()
        # Write state with an active_project that doesn't exist in projects
        state_data = {
            "projects": {},
            "active_project": "deleted-project",
            "version": 1,
        }
        (prism_home / "workspace.json").write_text(json.dumps(state_data))

        manager = WorkspaceManager(prism_home)
        assert manager.get_active_project() is None
