"""Multi-project workspace — manage multiple projects with independent configs.

Each registered project maintains its own:
- ``.prism.md`` memory file
- ``.prism.yaml`` config overrides
- Conversation history
- Cost tracking
- ``.prismignore`` patterns

Global configuration (``~/.prism/config.yaml``) is inherited, and
project-level settings in ``.prism.yaml`` override them.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Project name validation pattern: alphanumeric, hyphens, underscores, 1-64 chars
_PROJECT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


@dataclass
class ProjectInfo:
    """Metadata for a registered project.

    Attributes:
        name: Unique human-readable project identifier.
        path: Absolute filesystem path to the project root.
        created_at: ISO-8601 timestamp of registration.
        last_accessed: ISO-8601 timestamp of last switch/access.
        description: Optional short description of the project.
        active: Whether this project is currently the active one.
    """

    name: str
    path: str
    created_at: str
    last_accessed: str
    description: str = ""
    active: bool = False


@dataclass
class WorkspaceState:
    """Global workspace state persisted to ``~/.prism/workspace.json``.

    Attributes:
        projects: Mapping of project name to its metadata.
        active_project: Name of the currently active project, or ``None``.
        version: Schema version for future migrations.
    """

    projects: dict[str, ProjectInfo] = field(default_factory=dict)
    active_project: str | None = None
    version: int = 1


class WorkspaceManager:
    """Manages multiple projects with independent configurations.

    The manager stores workspace state in ``<prism_home>/workspace.json``
    and creates project-specific ``.prism/`` directories inside each
    registered project root.

    Args:
        prism_home: Path to the Prism data directory (typically ``~/.prism``).
    """

    def __init__(self, prism_home: Path) -> None:
        self._home = prism_home
        self._state_file = prism_home / "workspace.json"
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> WorkspaceState:
        """Load workspace state from disk.

        Returns:
            A ``WorkspaceState`` instance.  If the file is missing or
            corrupt, returns a fresh empty state.
        """
        if not self._state_file.is_file():
            return WorkspaceState()

        try:
            raw = self._state_file.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(raw)
            projects: dict[str, ProjectInfo] = {}
            for name, pdata in data.get("projects", {}).items():
                projects[name] = ProjectInfo(**pdata)
            return WorkspaceState(
                projects=projects,
                active_project=data.get("active_project"),
                version=data.get("version", 1),
            )
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("workspace_state_corrupt", error=str(exc))
            return WorkspaceState()

    def _save_state(self) -> None:
        """Persist workspace state to disk.

        Creates the parent directory if it does not exist.
        """
        self._home.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "projects": {n: asdict(p) for n, p in self._state.projects.items()},
            "active_project": self._state.active_project,
            "version": self._state.version,
        }
        self._state_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Project registration
    # ------------------------------------------------------------------

    def register_project(
        self,
        name: str,
        path: str | Path,
        description: str = "",
    ) -> ProjectInfo:
        """Register a new project in the workspace.

        Creates a ``.prism/`` directory inside the project root for
        project-specific state.

        Args:
            name: Unique project name (alphanumeric, hyphens, underscores).
            path: Filesystem path to the project root.
            description: Optional short description.

        Returns:
            The newly created ``ProjectInfo``.

        Raises:
            ValueError: If the name is invalid, already taken, or the
                path does not exist.
        """
        if name in self._state.projects:
            raise ValueError(f"Project '{name}' already exists")

        if not _PROJECT_NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid project name: '{name}'. "
                "Use alphanumeric characters, hyphens, and underscores. "
                "Must start with an alphanumeric character and be 1-64 characters."
            )

        resolved = Path(path).resolve()
        if not resolved.is_dir():
            raise ValueError(f"Path does not exist or is not a directory: {resolved}")

        # Check for duplicate paths
        for existing in self._state.projects.values():
            if Path(existing.path).resolve() == resolved:
                raise ValueError(
                    f"Path '{resolved}' is already registered as project '{existing.name}'"
                )

        now = datetime.now(UTC).isoformat()
        project = ProjectInfo(
            name=name,
            path=str(resolved),
            created_at=now,
            last_accessed=now,
            description=description,
            active=False,
        )

        # Create project-specific .prism directory
        project_prism_dir = resolved / ".prism"
        project_prism_dir.mkdir(exist_ok=True)

        self._state.projects[name] = project

        # If this is the first project, make it active
        if self._state.active_project is None:
            self._state.active_project = name
            project.active = True

        self._save_state()
        logger.info("project_registered", name=name, path=str(resolved))
        return project

    def remove_project(self, name: str) -> None:
        """Unregister a project from the workspace.

        Does **not** delete any files on disk — only removes the
        registration.

        Args:
            name: Name of the project to remove.

        Raises:
            ValueError: If the project is not found.
        """
        if name not in self._state.projects:
            raise ValueError(f"Project '{name}' not found")

        del self._state.projects[name]

        if self._state.active_project == name:
            if self._state.projects:
                most_recent = max(
                    self._state.projects.values(),
                    key=lambda p: p.last_accessed,
                )
                self._state.active_project = most_recent.name
                most_recent.active = True
            else:
                self._state.active_project = None

        self._save_state()
        logger.info("project_removed", name=name)

    # ------------------------------------------------------------------
    # Project switching
    # ------------------------------------------------------------------

    def switch_project(self, name: str) -> ProjectInfo:
        """Switch the active project.

        Args:
            name: Name of the project to switch to.

        Returns:
            The ``ProjectInfo`` of the newly active project.

        Raises:
            ValueError: If the project is not found.
        """
        if name not in self._state.projects:
            raise ValueError(f"Project '{name}' not found")

        # Deactivate current
        if (
            self._state.active_project
            and self._state.active_project in self._state.projects
        ):
            self._state.projects[self._state.active_project].active = False

        # Activate new
        project = self._state.projects[name]
        project.active = True
        project.last_accessed = datetime.now(UTC).isoformat()
        self._state.active_project = name

        self._save_state()
        logger.info("project_switched", name=name)
        return project

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_projects(self) -> list[ProjectInfo]:
        """List all registered projects, most recently accessed first.

        Returns:
            Sorted list of ``ProjectInfo`` instances.
        """
        return sorted(
            self._state.projects.values(),
            key=lambda p: p.last_accessed,
            reverse=True,
        )

    def get_active_project(self) -> ProjectInfo | None:
        """Get the currently active project.

        Returns:
            The active ``ProjectInfo``, or ``None`` if no project is active.
        """
        if (
            self._state.active_project
            and self._state.active_project in self._state.projects
        ):
            return self._state.projects[self._state.active_project]
        return None

    def get_project(self, name: str) -> ProjectInfo:
        """Get a specific project by name.

        Args:
            name: Project name.

        Returns:
            The ``ProjectInfo`` for the project.

        Raises:
            ValueError: If the project is not found.
        """
        if name not in self._state.projects:
            raise ValueError(f"Project '{name}' not found")
        return self._state.projects[name]

    def get_project_config_path(self, name: str) -> Path:
        """Get the ``.prism.yaml`` config path for a project.

        Args:
            name: Project name.

        Returns:
            Path to the project's ``.prism.yaml`` file.

        Raises:
            ValueError: If the project is not found.
        """
        project = self.get_project(name)
        return Path(project.path) / ".prism.yaml"

    def get_project_memory_path(self, name: str) -> Path:
        """Get the ``.prism.md`` memory path for a project.

        Args:
            name: Project name.

        Returns:
            Path to the project's ``.prism.md`` file.

        Raises:
            ValueError: If the project is not found.
        """
        project = self.get_project(name)
        return Path(project.path) / ".prism.md"

    def get_project_history_dir(self, name: str) -> Path:
        """Get the conversation history directory for a project.

        Args:
            name: Project name.

        Returns:
            Path to the project's ``.prism/history/`` directory.

        Raises:
            ValueError: If the project is not found.
        """
        project = self.get_project(name)
        history_dir = Path(project.path) / ".prism" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        return history_dir

    def get_project_cost_dir(self, name: str) -> Path:
        """Get the cost tracking directory for a project.

        Args:
            name: Project name.

        Returns:
            Path to the project's ``.prism/costs/`` directory.

        Raises:
            ValueError: If the project is not found.
        """
        project = self.get_project(name)
        cost_dir = Path(project.path) / ".prism" / "costs"
        cost_dir.mkdir(parents=True, exist_ok=True)
        return cost_dir

    def get_project_ignore_path(self, name: str) -> Path:
        """Get the ``.prismignore`` path for a project.

        Args:
            name: Project name.

        Returns:
            Path to the project's ``.prismignore`` file.

        Raises:
            ValueError: If the project is not found.
        """
        project = self.get_project(name)
        return Path(project.path) / ".prismignore"

    def get_recent_projects(self, limit: int = 5) -> list[ProjectInfo]:
        """Get the most recently accessed projects.

        Args:
            limit: Maximum number of projects to return.

        Returns:
            List of ``ProjectInfo`` instances, most recent first.
        """
        limit = max(limit, 0)
        return self.list_projects()[:limit]

    def update_last_accessed(self, name: str) -> None:
        """Update the last_accessed timestamp for a project.

        Args:
            name: Project name.

        Raises:
            ValueError: If the project is not found.
        """
        if name not in self._state.projects:
            raise ValueError(f"Project '{name}' not found")
        self._state.projects[name].last_accessed = datetime.now(UTC).isoformat()
        self._save_state()

    def project_exists(self, name: str) -> bool:
        """Check whether a project with the given name is registered.

        Args:
            name: Project name to check.

        Returns:
            ``True`` if registered, ``False`` otherwise.
        """
        return name in self._state.projects
