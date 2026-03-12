"""Hook system for pre/post tool execution.

Allows users to define shell commands that run before and/or after tool
executions.  Hooks are loaded from ``.prism/hooks.yaml`` or
``.prism-hooks.yaml`` in the project root.

Example ``hooks.yaml``::

    hooks:
      - event: pre_tool
        command: "echo running tool"
        tool: write_file
        enabled: true
      - event: post_tool
        command: "ruff check ."
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class HookConfig:
    """Configuration for a single hook.

    Attributes:
        event:       When the hook fires (``pre_tool``, ``post_tool``,
                     ``pre_command``, ``post_command``).
        command:     Shell command to execute.
        tool_filter: If set, only run for a specific tool name (e.g.
                     ``"write_file"``).
        enabled:     Whether the hook is active.
    """

    event: str
    command: str
    tool_filter: str | None = None
    enabled: bool = True


VALID_EVENTS = frozenset({
    "pre_tool",
    "post_tool",
    "pre_command",
    "post_command",
})

_HOOK_TIMEOUT_SECONDS = 30


@dataclass
class HookResult:
    """Result from running a hook.

    Attributes:
        success: Whether the hook command exited with return code 0.
        output:  Combined stdout and stderr from the hook command.
        hook:    The :class:`HookConfig` that was executed.
        blocked: If ``True``, the associated tool execution should be
                 blocked (non-zero exit from a ``pre_tool`` hook).
    """

    success: bool
    output: str
    hook: HookConfig
    blocked: bool = False


class HookManager:
    """Manages and executes hooks around tool calls.

    Hooks are loaded from YAML files in the project root on
    construction and cached for the lifetime of the manager.

    Args:
        project_root: Path to the project directory where hook config
                      files are located.
    """

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root
        self._hooks: list[HookConfig] = []
        self._load_hooks()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hooks(self) -> list[HookConfig]:
        """Return all loaded hooks (read-only copy)."""
        return list(self._hooks)

    @property
    def project_root(self) -> Path:
        """Return the project root this manager was initialised with."""
        return self._project_root

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_hooks(self) -> None:
        """Load hooks from ``.prism/hooks.yaml`` or ``.prism-hooks.yaml``.

        Silently returns an empty list when no hooks file is found or
        when the YAML content is invalid.
        """
        try:
            import yaml
        except ImportError:
            logger.debug("hooks_yaml_unavailable", reason="pyyaml not installed")
            return

        hooks_file = self._resolve_hooks_file()
        if hooks_file is None:
            return

        try:
            raw_text = hooks_file.read_text(encoding="utf-8")
            data = yaml.safe_load(raw_text)
            if not isinstance(data, dict):
                logger.debug("hooks_invalid_format", path=str(hooks_file))
                return

            raw_hooks = data.get("hooks", [])
            if not isinstance(raw_hooks, list):
                logger.debug("hooks_invalid_format", path=str(hooks_file))
                return

            for hook_data in raw_hooks:
                if not isinstance(hook_data, dict):
                    continue
                event = str(hook_data.get("event", ""))
                command = str(hook_data.get("command", ""))
                if not event or not command:
                    continue
                if event not in VALID_EVENTS:
                    logger.debug(
                        "hooks_invalid_event",
                        event=event,
                        valid=sorted(VALID_EVENTS),
                    )
                    continue
                self._hooks.append(
                    HookConfig(
                        event=event,
                        command=command,
                        tool_filter=hook_data.get("tool", None),
                        enabled=hook_data.get("enabled", True),
                    )
                )

            logger.debug("hooks_loaded", count=len(self._hooks), path=str(hooks_file))

        except Exception:
            logger.debug("hooks_load_failed", path=str(hooks_file), exc_info=True)

    def _resolve_hooks_file(self) -> Path | None:
        """Find the first existing hooks config file.

        Search order:
        1. ``<project_root>/.prism/hooks.yaml``
        2. ``<project_root>/.prism-hooks.yaml``

        Returns:
            The resolved :class:`Path` or ``None`` if no file exists.
        """
        candidates = [
            self._project_root / ".prism" / "hooks.yaml",
            self._project_root / ".prism-hooks.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    # ------------------------------------------------------------------
    # Hook retrieval
    # ------------------------------------------------------------------

    def get_hooks(
        self,
        event: str,
        tool_name: str | None = None,
    ) -> list[HookConfig]:
        """Get all hooks matching *event* and optionally *tool_name*.

        Disabled hooks are excluded.  If a hook has a ``tool_filter`` and
        *tool_name* does not match, the hook is excluded.

        Args:
            event:     The event type to filter on.
            tool_name: Optional tool name to filter on.

        Returns:
            A list of matching :class:`HookConfig` instances.
        """
        results: list[HookConfig] = []
        for hook in self._hooks:
            if not hook.enabled:
                continue
            if hook.event != event:
                continue
            if hook.tool_filter and tool_name and hook.tool_filter != tool_name:
                continue
            results.append(hook)
        return results

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_hook(
        self,
        hook: HookConfig,
        context: dict[str, Any] | None = None,
    ) -> HookResult:
        """Execute a single hook command.

        Environment variables prefixed with ``PRISM_HOOK_`` are injected
        from *context* so hooks can reference tool arguments and outputs.

        A non-zero exit code results in ``blocked=True`` (useful for
        pre-tool hooks that should gate execution).

        Args:
            hook:    The hook configuration to execute.
            context: Optional key/value pairs exposed as
                     ``PRISM_HOOK_<KEY>`` env vars.

        Returns:
            A :class:`HookResult` with the execution outcome.
        """
        try:
            env_vars: dict[str, str] = {}
            if context:
                for key, value in context.items():
                    env_key = f"PRISM_HOOK_{key.upper()}"
                    env_vars[env_key] = str(value)

            env = {**os.environ, **env_vars}

            proc = subprocess.run(  # noqa: S602
                hook.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=_HOOK_TIMEOUT_SECONDS,
                cwd=str(self._project_root),
                env=env,
            )

            blocked = proc.returncode != 0
            output = (proc.stdout or "") + (proc.stderr or "")

            logger.debug(
                "hook_executed",
                command=hook.command,
                returncode=proc.returncode,
                blocked=blocked,
            )

            return HookResult(
                success=proc.returncode == 0,
                output=output,
                hook=hook,
                blocked=blocked,
            )

        except subprocess.TimeoutExpired:
            logger.warning(
                "hook_timeout",
                command=hook.command,
                timeout=_HOOK_TIMEOUT_SECONDS,
            )
            return HookResult(
                success=False,
                output=f"Hook timed out after {_HOOK_TIMEOUT_SECONDS}s",
                hook=hook,
                blocked=False,
            )

        except OSError as exc:
            logger.warning("hook_os_error", command=hook.command, error=str(exc))
            return HookResult(
                success=False,
                output=f"OS error: {exc}",
                hook=hook,
                blocked=False,
            )

        except Exception as exc:
            logger.warning("hook_unexpected_error", command=hook.command, error=str(exc))
            return HookResult(
                success=False,
                output=str(exc),
                hook=hook,
                blocked=False,
            )

    def run_pre_hooks(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> list[HookResult]:
        """Run all ``pre_tool`` hooks for the given tool.

        Hook context includes the tool name and (truncated) arguments.
        If any hook result has ``blocked=True``, the caller should skip
        the tool execution.

        Args:
            tool_name: The name of the tool about to execute.
            arguments: The tool arguments dictionary.

        Returns:
            A list of :class:`HookResult` instances.
        """
        hooks = self.get_hooks("pre_tool", tool_name)
        if not hooks:
            return []

        context: dict[str, str] = {"tool": tool_name}
        for key, value in arguments.items():
            context[key] = str(value)[:200]

        return [self.run_hook(h, context) for h in hooks]

    def run_post_hooks(
        self,
        tool_name: str,
        result_output: str,
    ) -> list[HookResult]:
        """Run all ``post_tool`` hooks for the given tool.

        Hook context includes the tool name and (truncated) result output.

        Args:
            tool_name:     The name of the tool that just executed.
            result_output: The tool's output string (truncated to 500 chars).

        Returns:
            A list of :class:`HookResult` instances.
        """
        hooks = self.get_hooks("post_tool", tool_name)
        if not hooks:
            return []

        context: dict[str, str] = {
            "tool": tool_name,
            "output": result_output[:500],
        }

        return [self.run_hook(h, context) for h in hooks]

    def add_hook(self, hook: HookConfig) -> None:
        """Programmatically add a hook at runtime.

        Args:
            hook: The hook configuration to add.

        Raises:
            ValueError: If the event type is invalid.
        """
        if hook.event not in VALID_EVENTS:
            raise ValueError(
                f"Invalid event '{hook.event}'. "
                f"Must be one of: {sorted(VALID_EVENTS)}"
            )
        self._hooks.append(hook)
        logger.debug("hook_added", hook_event=hook.event, command=hook.command)

    def remove_hooks(self, event: str | None = None) -> int:
        """Remove hooks, optionally filtered by event.

        Args:
            event: If specified, only remove hooks for this event.
                   If ``None``, remove all hooks.

        Returns:
            The number of hooks removed.
        """
        if event is None:
            count = len(self._hooks)
            self._hooks.clear()
            return count

        before = len(self._hooks)
        self._hooks = [h for h in self._hooks if h.event != event]
        return before - len(self._hooks)
