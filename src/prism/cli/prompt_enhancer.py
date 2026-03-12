"""Smart prompt enhancement for better LLM responses.

Analyses user prompts, classifies the task type, and automatically injects
relevant project context (file contents, project structure, debugging hints)
so that downstream models produce higher-quality output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists used for strategy detection
# ---------------------------------------------------------------------------

_DEBUG_WORDS: tuple[str, ...] = (
    "error",
    "bug",
    "fix",
    "crash",
    "traceback",
    "exception",
    "fail",
    "broken",
    "debug",
)

_CREATE_WORDS: tuple[str, ...] = (
    "create",
    "make",
    "build",
    "generate",
    "scaffold",
    "new file",
    "new project",
    "init",
)

_EDIT_WORDS: tuple[str, ...] = (
    "edit",
    "modify",
    "change",
    "update",
    "refactor",
    "rename",
    "add to",
    "remove from",
)

_EXPLAIN_WORDS: tuple[str, ...] = (
    "explain",
    "what does",
    "how does",
    "why",
    "understand",
    "describe",
    "tell me about",
)

# File-path regex: matches common source file extensions.
_FILE_REF_PATTERN = re.compile(
    r"[\w./\\-]+\.(?:py|js|ts|jsx|tsx|rs|go|java|cpp|c|h|yaml|yml|json|toml|md|txt|html|css)"
)

# Maximum characters of file content to include in an enhanced prompt.
_MAX_FILE_CONTENT_CHARS = 2000
_MAX_FILE_CONTENT_CHARS_EXPLAIN = 3000

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class EnhancedPrompt:
    """Result of prompt enhancement.

    Attributes:
        original: The raw user prompt.
        enhanced: The enriched prompt with injected context.
        context_added: Human-readable descriptions of what was injected.
        strategy: One of ``code_edit``, ``question``, ``create``,
            ``debug``, ``explain``.
    """

    original: str
    enhanced: str
    context_added: list[str] = field(default_factory=list)
    strategy: str = "question"


# ---------------------------------------------------------------------------
# Enhancer
# ---------------------------------------------------------------------------


class PromptEnhancer:
    """Enhances user prompts with project context for better LLM responses.

    Strategies:
        - **code_edit** — adds current file content, imports, related files.
        - **question** — returns the prompt mostly unchanged.
        - **create** — adds project structure, conventions, dependencies.
        - **debug** — adds error-handling instructions and relevant files.
        - **explain** — adds referenced code and docstrings.

    Args:
        project_root: Path to the root of the current project.
        active_files: Optional list of file paths (relative to
            *project_root*) that the user is currently working with.
    """

    def __init__(
        self,
        project_root: Path,
        active_files: list[str] | None = None,
    ) -> None:
        self._root = project_root
        self._active_files = active_files or []

    # ----- public API -------------------------------------------------------

    def enhance(self, prompt: str) -> EnhancedPrompt:
        """Enhance a user prompt with relevant context.

        Args:
            prompt: The raw user prompt.

        Returns:
            An :class:`EnhancedPrompt` containing the enriched text and
            metadata about what context was added.
        """
        strategy = self._detect_strategy(prompt)
        context_parts: list[str] = []
        enhanced = prompt

        # Attach active-file list when available
        if self._active_files:
            file_list = ", ".join(self._active_files[:5])
            enhanced = f"{prompt}\n\n[Active files: {file_list}]"
            context_parts.append(f"Active files: {file_list}")

        # Strategy-specific enrichment
        if strategy == "code_edit":
            enhanced = self._enhance_code_edit(prompt, context_parts)
        elif strategy == "debug":
            enhanced = self._enhance_debug(prompt, context_parts)
        elif strategy == "create":
            enhanced = self._enhance_create(prompt, context_parts)
        elif strategy == "explain":
            enhanced = self._enhance_explain(prompt, context_parts)
        # "question" gets no extra enrichment beyond active files

        return EnhancedPrompt(
            original=prompt,
            enhanced=enhanced,
            context_added=context_parts,
            strategy=strategy,
        )

    # ----- strategy detection -----------------------------------------------

    def _detect_strategy(self, prompt: str) -> str:
        """Classify the user prompt into one of the known strategy types.

        The detection is intentionally keyword-based and cheap — no LLM call.
        Order of precedence: debug > create > code_edit > explain > question.
        """
        lower = prompt.lower()

        for word in _DEBUG_WORDS:
            if word in lower:
                return "debug"
        for word in _CREATE_WORDS:
            if word in lower:
                return "create"
        for word in _EDIT_WORDS:
            if word in lower:
                return "code_edit"
        for word in _EXPLAIN_WORDS:
            if word in lower:
                return "explain"

        return "question"

    # ----- per-strategy enhancers -------------------------------------------

    def _enhance_code_edit(self, prompt: str, context: list[str]) -> str:
        """Enhance a code-editing prompt by including referenced file content."""
        files = self._extract_file_refs(prompt)
        parts = [prompt]

        for fpath in files[:3]:  # cap at 3 files
            full = self._root / fpath
            if full.exists() and full.is_file():
                try:
                    content = full.read_text(encoding="utf-8")[:_MAX_FILE_CONTENT_CHARS]
                    parts.append(f"\n--- Current content of {fpath} ---\n{content}")
                    context.append(f"Included content of {fpath}")
                except Exception:
                    logger.debug("prompt_enhancer_file_read_failed", path=str(fpath))

        return "\n".join(parts)

    def _enhance_debug(self, prompt: str, context: list[str]) -> str:
        """Enhance a debugging prompt with investigation instructions."""
        parts = [prompt]
        parts.append(
            "\n[Debugging context: Check error messages carefully. "
            "Read relevant files before suggesting fixes. "
            "Run tests after making changes.]"
        )
        context.append("Added debugging instructions")
        return "\n".join(parts)

    def _enhance_create(self, prompt: str, context: list[str]) -> str:
        """Enhance a creation prompt with the project's top-level structure."""
        parts = [prompt]

        try:
            entries = sorted(self._root.iterdir())
            dirs = [
                e.name
                for e in entries
                if e.is_dir() and not e.name.startswith(".")
            ][:10]
            files = [
                e.name
                for e in entries
                if e.is_file() and not e.name.startswith(".")
            ][:10]

            if dirs or files:
                struct = "Project structure:\n"
                for d in dirs:
                    struct += f"  {d}/\n"
                for f in files:
                    struct += f"  {f}\n"
                parts.append(f"\n{struct}")
                context.append("Added project structure")
        except Exception:
            logger.debug("prompt_enhancer_structure_failed")

        return "\n".join(parts)

    def _enhance_explain(self, prompt: str, context: list[str]) -> str:
        """Enhance an explanation prompt by attaching referenced files."""
        files = self._extract_file_refs(prompt)
        parts = [prompt]

        for fpath in files[:2]:  # cap at 2 files for explanations
            full = self._root / fpath
            if full.exists() and full.is_file():
                try:
                    content = full.read_text(encoding="utf-8")[
                        :_MAX_FILE_CONTENT_CHARS_EXPLAIN
                    ]
                    parts.append(f"\n--- {fpath} ---\n{content}")
                    context.append(f"Included {fpath} for reference")
                except Exception:
                    logger.debug("prompt_enhancer_file_read_failed", path=str(fpath))

        return "\n".join(parts)

    # ----- utilities --------------------------------------------------------

    def _extract_file_refs(self, prompt: str) -> list[str]:
        """Extract file-path references from a prompt string.

        Only returns paths that actually exist on disk relative to
        *project_root*.

        Returns:
            A deduplicated list of relative path strings, preserving order.
        """
        files: list[str] = []
        for match in _FILE_REF_PATTERN.finditer(prompt):
            candidate = match.group(0)
            if (self._root / candidate).exists():
                files.append(candidate)

        # Deduplicate while preserving insertion order
        return list(dict.fromkeys(files))
