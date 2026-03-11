"""Background task queue — run long tasks in separate threads with progress tracking.

Provides a task queue that executes callables in daemon threads, persists results
to ``~/.prism/tasks/<id>/``, and sends desktop notifications on completion.
"""

from __future__ import annotations

import json
import platform
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_MAX_PARALLEL: int = 4
DEFAULT_CLEANUP_HOURS: int = 24


class TaskStatus(Enum):
    """Status of a background task."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a completed background task.

    Attributes:
        output:      The task's return value as a string.
        error:       Error message if the task failed, ``None`` otherwise.
        exit_code:   0 for success, 1 for failure.
        duration_ms: Wall-clock execution time in milliseconds.
    """

    output: str
    error: str | None
    exit_code: int
    duration_ms: float


@dataclass
class BackgroundTask:
    """A task running or queued in the background.

    Attributes:
        id:               Short unique identifier (8 hex chars).
        name:             Human-readable task name.
        description:      Longer description of what the task does.
        status:           Current task status.
        created_at:       ISO 8601 timestamp of task creation.
        started_at:       ISO 8601 timestamp when execution began.
        completed_at:     ISO 8601 timestamp when execution ended.
        progress:         Completion fraction (0.0 to 1.0).
        progress_message: Human-readable progress message.
        result:           Task result, populated on completion or failure.
    """

    id: str
    name: str
    description: str
    status: TaskStatus
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    progress: float = 0.0
    progress_message: str = ""
    result: TaskResult | None = field(default=None, repr=False)

    @property
    def is_terminal(self) -> bool:
        """Whether the task has reached a final state."""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )


class TaskQueue:
    """Background task queue with threading, notifications, and persistence.

    Usage::

        queue = TaskQueue(tasks_dir=Path("~/.prism/tasks").expanduser())
        task = queue.submit("refactor", some_function, description="Refactor module")
        # ... continue working ...
        result = queue.get_results(task.id)

    Tasks run in daemon threads and persist their results to disk.  Desktop
    notifications are sent on completion (macOS ``osascript``, Linux
    ``notify-send``).
    """

    def __init__(
        self,
        tasks_dir: Path,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
    ) -> None:
        """Initialise the task queue.

        Args:
            tasks_dir:    Directory for persisting task results.
            max_parallel: Maximum number of tasks to run concurrently.

        Raises:
            ValueError: If *max_parallel* is less than 1.
        """
        if max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")

        self._tasks_dir = tasks_dir
        self._tasks_dir.mkdir(parents=True, exist_ok=True)
        self._max_parallel = max_parallel
        self._tasks: dict[str, BackgroundTask] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tasks_dir(self) -> Path:
        """Directory where task results are persisted."""
        return self._tasks_dir

    @property
    def max_parallel(self) -> int:
        """Maximum number of concurrent tasks."""
        return self._max_parallel

    @property
    def active_count(self) -> int:
        """Number of currently running tasks."""
        with self._lock:
            return sum(
                1
                for t in self._tasks.values()
                if t.status == TaskStatus.RUNNING
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        name: str,
        callable_fn: Callable[..., Any],
        description: str = "",
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> BackgroundTask:
        """Submit a task to the queue.

        The task begins executing immediately in a daemon thread (up to
        *max_parallel* tasks may run concurrently).

        Args:
            name:         Short human-readable name for the task.
            callable_fn:  The callable to execute.
            description:  Longer description shown in queue listings.
            args:         Positional arguments for *callable_fn*.
            kwargs:       Keyword arguments for *callable_fn*.

        Returns:
            The newly created :class:`BackgroundTask`.

        Raises:
            ValueError: If *name* is empty.
            RuntimeError: If the maximum number of parallel tasks is reached.
        """
        if not name or not name.strip():
            raise ValueError("Task name must not be empty")

        with self._lock:
            running = sum(
                1
                for t in self._tasks.values()
                if t.status == TaskStatus.RUNNING
            )
            if running >= self._max_parallel:
                raise RuntimeError(
                    f"Maximum parallel tasks reached ({self._max_parallel}). "
                    "Wait for a task to complete or cancel one."
                )

        task_id = str(uuid4())[:8]
        now = datetime.now(UTC).isoformat()

        task = BackgroundTask(
            id=task_id,
            name=name.strip(),
            description=description.strip() or name.strip(),
            status=TaskStatus.QUEUED,
            created_at=now,
        )

        cancel_event = threading.Event()

        with self._lock:
            self._tasks[task_id] = task
            self._cancel_events[task_id] = cancel_event

        thread = threading.Thread(
            target=self._run_task,
            args=(task_id, callable_fn, args, kwargs or {}, cancel_event),
            daemon=True,
            name=f"prism-task-{task_id}",
        )

        with self._lock:
            self._threads[task_id] = thread

        thread.start()
        logger.info("task_submitted", task_id=task_id, name=name)
        return task

    def cancel(self, task_id: str) -> bool:
        """Cancel a running or queued task.

        Args:
            task_id: The task identifier.

        Returns:
            True if the task was cancelled, False if it was already in a
            terminal state.

        Raises:
            ValueError: If *task_id* does not refer to a known task.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise ValueError(f"Task not found: {task_id}")

            if task.is_terminal:
                return False

            cancel_event = self._cancel_events.get(task_id)
            if cancel_event:
                cancel_event.set()

            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now(UTC).isoformat()
            self._save_task(task)

        logger.info("task_cancelled", task_id=task_id)
        return True

    def get_task(self, task_id: str) -> BackgroundTask:
        """Get a task by ID.

        Args:
            task_id: The task identifier.

        Returns:
            The :class:`BackgroundTask` instance.

        Raises:
            ValueError: If *task_id* does not refer to a known task.
        """
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")
        return task

    def list_tasks(self, include_completed: bool = True) -> list[BackgroundTask]:
        """List all tasks, newest first.

        Args:
            include_completed: When False, exclude tasks in terminal states.

        Returns:
            A list of :class:`BackgroundTask` instances sorted by creation
            time (most recent first).
        """
        with self._lock:
            tasks = list(self._tasks.values())
        if not include_completed:
            tasks = [t for t in tasks if not t.is_terminal]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def get_results(self, task_id: str) -> TaskResult | None:
        """Get results for a completed task.

        Checks the in-memory task first, then falls back to disk persistence.

        Args:
            task_id: The task identifier.

        Returns:
            A :class:`TaskResult` if the task completed, ``None`` otherwise.

        Raises:
            ValueError: If *task_id* does not refer to a known task.
        """
        task = self.get_task(task_id)
        if task.result is not None:
            return task.result

        # Try loading from disk
        result_path = self._tasks_dir / task_id / "result.json"
        if result_path.is_file():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            return TaskResult(**data)

        return None

    def update_progress(
        self,
        task_id: str,
        progress: float,
        message: str = "",
    ) -> None:
        """Update the progress of a running task.

        Args:
            task_id:  The task identifier.
            progress: Completion fraction (0.0 to 1.0).
            message:  Optional human-readable progress message.

        Raises:
            ValueError: If *task_id* is unknown or *progress* is out of range.
        """
        if not 0.0 <= progress <= 1.0:
            raise ValueError("Progress must be between 0.0 and 1.0")

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise ValueError(f"Task not found: {task_id}")
            task.progress = progress
            task.progress_message = message

    def cleanup_completed(self, max_age_hours: int = DEFAULT_CLEANUP_HOURS) -> int:
        """Remove completed tasks older than *max_age_hours*.

        Args:
            max_age_hours: Tasks older than this many hours are removed.

        Returns:
            The number of tasks removed.
        """
        if max_age_hours < 0:
            raise ValueError("max_age_hours must be non-negative")

        cutoff = datetime.now(UTC).timestamp() - (max_age_hours * 3600)
        removed = 0

        with self._lock:
            to_remove: list[str] = []
            for task_id, task in self._tasks.items():
                if task.is_terminal and task.completed_at:
                    try:
                        completed_ts = datetime.fromisoformat(
                            task.completed_at
                        ).timestamp()
                        if completed_ts < cutoff:
                            to_remove.append(task_id)
                    except ValueError:
                        pass

            for task_id in to_remove:
                del self._tasks[task_id]
                self._cancel_events.pop(task_id, None)
                self._threads.pop(task_id, None)
                removed += 1

        logger.info("tasks_cleaned_up", removed=removed, max_age_hours=max_age_hours)
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_task(
        self,
        task_id: str,
        callable_fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        """Execute a task in a background thread."""
        with self._lock:
            task = self._tasks[task_id]
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now(UTC).isoformat()

        start = time.monotonic()

        try:
            if cancel_event.is_set():
                return

            output = callable_fn(*args, **kwargs)
            elapsed = (time.monotonic() - start) * 1000

            if cancel_event.is_set():
                return

            result = TaskResult(
                output=str(output),
                error=None,
                exit_code=0,
                duration_ms=round(elapsed, 2),
            )

            with self._lock:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now(UTC).isoformat()
                task.progress = 1.0
                task.progress_message = "Done"
                task.result = result

            self._save_task(task)
            self._notify(task)

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            error_msg = f"{type(exc).__name__}: {exc}"
            result = TaskResult(
                output="",
                error=error_msg,
                exit_code=1,
                duration_ms=round(elapsed, 2),
            )

            with self._lock:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now(UTC).isoformat()
                task.progress_message = f"Failed: {error_msg}"
                task.result = result

            self._save_task(task)
            self._notify(task)
            logger.error(
                "task_failed",
                task_id=task_id,
                error=error_msg,
            )

    def _save_task(self, task: BackgroundTask) -> None:
        """Persist task result to disk as JSON."""
        task_dir = self._tasks_dir / task.id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Save task metadata
        meta_path = task_dir / "task.json"
        meta: dict[str, Any] = {
            "id": task.id,
            "name": task.name,
            "description": task.description,
            "status": task.status.value,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "progress": task.progress,
            "progress_message": task.progress_message,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Save result if available
        if task.result:
            result_path = task_dir / "result.json"
            result_path.write_text(
                json.dumps(asdict(task.result), indent=2),
                encoding="utf-8",
            )

    @staticmethod
    def _notify(task: BackgroundTask) -> None:
        """Send a desktop notification about task completion.

        Uses ``osascript`` on macOS and ``notify-send`` on Linux.  Failures
        are silently ignored — notifications are best-effort.
        """
        status_emoji = "completed" if task.status == TaskStatus.COMPLETED else "failed"
        title = f"Prism: {task.name}"
        message = f"Task {status_emoji}: {task.description}"

        # Sanitise for shell safety — remove quotes and special chars
        title = title.replace('"', "'").replace("\\", "")
        message = message.replace('"', "'").replace("\\", "")

        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        f'display notification "{message}" with title "{title}"',
                    ],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            elif system == "Linux":
                subprocess.run(
                    ["notify-send", title, message],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # Notification not available — best-effort
