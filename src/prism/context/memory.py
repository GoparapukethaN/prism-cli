"""Project memory — persistent key-value facts stored in PRISM_MEMORY.md.

Each project can have a ``PRISM_MEMORY.md`` file in its root that stores
structured facts the assistant should remember across sessions.
"""

from __future__ import annotations

import re
from pathlib import Path

import structlog

from prism.exceptions import ContextError

logger = structlog.get_logger(__name__)

_MEMORY_FILENAME = "PRISM_MEMORY.md"

# Regex to parse ``**key**: value`` lines
_FACT_RE = re.compile(r"^\*\*(.+?)\*\*:\s*(.+)$", re.MULTILINE)


class ProjectMemory:
    """Read/write a ``PRISM_MEMORY.md`` file for persistent project facts.

    Facts are stored as markdown key-value pairs::

        # Project Memory

        ## Facts

        **stack**: Python 3.12, FastAPI
        **test_framework**: pytest
        **deploy_target**: AWS Lambda

    Args:
        project_root: Directory where ``PRISM_MEMORY.md`` resides.
    """

    def __init__(self, project_root: Path | str) -> None:
        self._project_root = Path(project_root)
        self._memory_path = self._project_root / _MEMORY_FILENAME
        self._facts: dict[str, str] = {}
        self._loaded = False

    @property
    def memory_path(self) -> Path:
        return self._memory_path

    # ------------------------------------------------------------------
    # Loading / Saving
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load facts from disk if not already loaded."""
        if not self._loaded:
            self._load()

    def _load(self) -> None:
        """Parse facts from the PRISM_MEMORY.md file."""
        self._facts.clear()

        if not self._memory_path.is_file():
            self._loaded = True
            return

        try:
            content = self._memory_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("memory_load_failed", error=str(exc))
            self._loaded = True
            return

        for match in _FACT_RE.finditer(content):
            key = match.group(1).strip()
            value = match.group(2).strip()
            self._facts[key] = value

        self._loaded = True
        logger.debug("memory_loaded", facts=len(self._facts))

    def _save(self) -> None:
        """Write facts to the PRISM_MEMORY.md file."""
        content = self._render()
        try:
            self._memory_path.write_text(content, encoding="utf-8")
            logger.debug("memory_saved", facts=len(self._facts))
        except OSError as exc:
            raise ContextError(
                f"Failed to save project memory: {exc}"
            ) from exc

    def _render(self) -> str:
        """Render facts to markdown content."""
        lines = ["# Project Memory", "", "## Facts", ""]
        for key, value in sorted(self._facts.items()):
            lines.append(f"**{key}**: {value}")
        lines.append("")  # trailing newline
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fact(self, key: str, value: str) -> None:
        """Store or update a key-value fact.

        Args:
            key: The fact identifier.
            value: The fact value.

        Raises:
            ContextError: If *key* or *value* is empty.
        """
        if not key or not key.strip():
            raise ContextError("Fact key must not be empty")
        if not value or not value.strip():
            raise ContextError("Fact value must not be empty")

        self._ensure_loaded()
        self._facts[key.strip()] = value.strip()
        self._save()

    def remove_fact(self, key: str) -> bool:
        """Remove a fact by key.

        Returns:
            *True* if the fact existed and was removed.
        """
        self._ensure_loaded()
        if key in self._facts:
            del self._facts[key]
            self._save()
            return True
        return False

    def get_fact(self, key: str) -> str | None:
        """Retrieve a single fact value by key.

        Returns:
            The fact value, or *None* if not found.
        """
        self._ensure_loaded()
        return self._facts.get(key)

    def get_facts(self) -> dict[str, str]:
        """Return all stored facts.

        Returns:
            A dictionary mapping fact keys to values.
        """
        self._ensure_loaded()
        return dict(self._facts)

    def get_context_block(self) -> str:
        """Format all facts as a context block for LLM injection.

        Returns:
            A formatted string suitable for inclusion in system context,
            or an empty string if there are no facts.
        """
        self._ensure_loaded()
        if not self._facts:
            return ""

        lines = ["[Project Memory]"]
        for key, value in sorted(self._facts.items()):
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Remove all facts and delete the memory file."""
        self._facts.clear()
        self._loaded = True
        if self._memory_path.is_file():
            try:
                self._memory_path.unlink()
            except OSError as exc:
                raise ContextError(
                    f"Failed to delete memory file: {exc}"
                ) from exc

    def reload(self) -> None:
        """Force reload facts from disk."""
        self._loaded = False
        self._ensure_loaded()
