"""Tests for the background task queue."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from prism.tools.task_queue import (
    DEFAULT_CLEANUP_HOURS,
    DEFAULT_MAX_PARALLEL,
    BackgroundTask,
    TaskQueue,
    TaskResult,
    TaskStatus,
)

if TYPE_CHECKING:
    from pathlib import Path


# =====================================================================
# TaskStatus
# =====================================================================


class TestTaskStatus:
    """Tests for the TaskStatus enum."""

    def test_values(self) -> None:
        assert TaskStatus.QUEUED.value == "queued"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"

    def test_all_members(self) -> None:
        assert set(TaskStatus.__members__) == {
            "QUEUED",
            "RUNNING",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
        }


# =====================================================================
# TaskResult
# =====================================================================


class TestTaskResult:
    """Tests for the TaskResult dataclass."""

    def test_fields(self) -> None:
        result = TaskResult(
            output="done",
            error=None,
            exit_code=0,
            duration_ms=100.0,
        )
        assert result.output == "done"
        assert result.error is None
        assert result.exit_code == 0
        assert result.duration_ms == 100.0

    def test_failed_result(self) -> None:
        result = TaskResult(
            output="",
            error="ValueError: bad input",
            exit_code=1,
            duration_ms=50.0,
        )
        assert result.error is not None
        assert result.exit_code == 1


# =====================================================================
# BackgroundTask
# =====================================================================


class TestBackgroundTask:
    """Tests for the BackgroundTask dataclass."""

    def test_fields(self) -> None:
        task = BackgroundTask(
            id="abc12345",
            name="test-task",
            description="A test task",
            status=TaskStatus.QUEUED,
            created_at="2025-01-01T00:00:00+00:00",
        )
        assert task.id == "abc12345"
        assert task.name == "test-task"
        assert task.description == "A test task"
        assert task.status == TaskStatus.QUEUED
        assert task.started_at is None
        assert task.completed_at is None
        assert task.progress == 0.0
        assert task.progress_message == ""
        assert task.result is None

    def test_is_terminal_queued(self) -> None:
        task = BackgroundTask(
            id="a", name="t", description="d",
            status=TaskStatus.QUEUED, created_at="now",
        )
        assert task.is_terminal is False

    def test_is_terminal_running(self) -> None:
        task = BackgroundTask(
            id="a", name="t", description="d",
            status=TaskStatus.RUNNING, created_at="now",
        )
        assert task.is_terminal is False

    def test_is_terminal_completed(self) -> None:
        task = BackgroundTask(
            id="a", name="t", description="d",
            status=TaskStatus.COMPLETED, created_at="now",
        )
        assert task.is_terminal is True

    def test_is_terminal_failed(self) -> None:
        task = BackgroundTask(
            id="a", name="t", description="d",
            status=TaskStatus.FAILED, created_at="now",
        )
        assert task.is_terminal is True

    def test_is_terminal_cancelled(self) -> None:
        task = BackgroundTask(
            id="a", name="t", description="d",
            status=TaskStatus.CANCELLED, created_at="now",
        )
        assert task.is_terminal is True


# =====================================================================
# TaskQueue — Initialisation
# =====================================================================


class TestTaskQueueInit:
    """Tests for TaskQueue initialisation."""

    def test_creates_tasks_dir(self, tmp_path: Path) -> None:
        tasks_dir = tmp_path / "tasks"
        assert not tasks_dir.exists()
        queue = TaskQueue(tasks_dir=tasks_dir)
        assert tasks_dir.is_dir()
        assert queue.tasks_dir == tasks_dir

    def test_default_max_parallel(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        assert queue.max_parallel == DEFAULT_MAX_PARALLEL

    def test_custom_max_parallel(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path, max_parallel=8)
        assert queue.max_parallel == 8

    def test_max_parallel_too_low(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            TaskQueue(tasks_dir=tmp_path, max_parallel=0)

    def test_active_count_initially_zero(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        assert queue.active_count == 0


# =====================================================================
# TaskQueue — Submit
# =====================================================================


class TestTaskQueueSubmit:
    """Tests for task submission."""

    def test_submit_returns_task(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("test", lambda: "ok")
        assert isinstance(task, BackgroundTask)
        assert task.name == "test"
        assert len(task.id) == 8

    def test_submit_empty_name_raises(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        with pytest.raises(ValueError, match="must not be empty"):
            queue.submit("", lambda: "ok")

    def test_submit_whitespace_name_raises(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        with pytest.raises(ValueError, match="must not be empty"):
            queue.submit("   ", lambda: "ok")

    def test_submit_with_description(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("test", lambda: "ok", description="A detailed desc")
        assert task.description == "A detailed desc"

    def test_submit_uses_name_as_default_description(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("my-task", lambda: "ok")
        assert task.description == "my-task"

    @patch("prism.tools.task_queue.TaskQueue._notify")
    def test_task_completes_successfully(
        self, mock_notify: MagicMock, tmp_path: Path
    ) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("fast-task", lambda: "result-value")

        # Wait for thread to finish
        time.sleep(0.5)

        updated = queue.get_task(task.id)
        assert updated.status == TaskStatus.COMPLETED
        assert updated.progress == 1.0
        assert updated.result is not None
        assert updated.result.output == "result-value"
        assert updated.result.exit_code == 0
        assert updated.result.error is None

    @patch("prism.tools.task_queue.TaskQueue._notify")
    def test_task_fails_with_exception(
        self, mock_notify: MagicMock, tmp_path: Path
    ) -> None:
        def failing_fn() -> str:
            msg = "something went wrong"
            raise ValueError(msg)

        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("failing-task", failing_fn)

        time.sleep(0.5)

        updated = queue.get_task(task.id)
        assert updated.status == TaskStatus.FAILED
        assert updated.result is not None
        assert updated.result.exit_code == 1
        assert "ValueError" in (updated.result.error or "")

    def test_submit_max_parallel_reached(self, tmp_path: Path) -> None:
        barrier = threading.Event()

        def blocking_fn() -> str:
            barrier.wait(timeout=5)
            return "done"

        queue = TaskQueue(tasks_dir=tmp_path, max_parallel=1)
        queue.submit("blocker", blocking_fn)

        # Wait for the first task to start running
        time.sleep(0.3)

        with pytest.raises(RuntimeError, match="Maximum parallel tasks"):
            queue.submit("second", lambda: "ok")

        barrier.set()

    @patch("prism.tools.task_queue.TaskQueue._notify")
    def test_submit_with_args_and_kwargs(
        self, mock_notify: MagicMock, tmp_path: Path
    ) -> None:
        def adder(a: int, b: int, extra: int = 0) -> str:
            return str(a + b + extra)

        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("add", adder, args=(3, 4), kwargs={"extra": 10})

        time.sleep(0.5)

        result = queue.get_results(task.id)
        assert result is not None
        assert result.output == "17"


# =====================================================================
# TaskQueue — Cancel
# =====================================================================


class TestTaskQueueCancel:
    """Tests for task cancellation."""

    def test_cancel_unknown_task(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        with pytest.raises(ValueError, match="Task not found"):
            queue.cancel("nonexistent")

    @patch("prism.tools.task_queue.TaskQueue._notify")
    def test_cancel_completed_task_returns_false(
        self, mock_notify: MagicMock, tmp_path: Path
    ) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("fast", lambda: "done")
        time.sleep(0.5)

        result = queue.cancel(task.id)
        assert result is False

    def test_cancel_running_task(self, tmp_path: Path) -> None:
        barrier = threading.Event()

        def blocking_fn() -> str:
            barrier.wait(timeout=5)
            return "done"

        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("blocker", blocking_fn)
        time.sleep(0.3)

        result = queue.cancel(task.id)
        assert result is True

        updated = queue.get_task(task.id)
        assert updated.status == TaskStatus.CANCELLED
        assert updated.completed_at is not None

        barrier.set()


# =====================================================================
# TaskQueue — Get task / List tasks
# =====================================================================


class TestTaskQueueGetList:
    """Tests for getting and listing tasks."""

    def test_get_unknown_task_raises(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        with pytest.raises(ValueError, match="Task not found"):
            queue.get_task("nonexistent")

    def test_get_task_by_id(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("test", lambda: "ok")
        fetched = queue.get_task(task.id)
        assert fetched.id == task.id
        assert fetched.name == "test"

    @patch("prism.tools.task_queue.TaskQueue._notify")
    def test_list_tasks_all(
        self, mock_notify: MagicMock, tmp_path: Path
    ) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        queue.submit("task1", lambda: "a")
        queue.submit("task2", lambda: "b")
        time.sleep(0.5)

        tasks = queue.list_tasks(include_completed=True)
        assert len(tasks) == 2

    @patch("prism.tools.task_queue.TaskQueue._notify")
    def test_list_tasks_exclude_completed(
        self, mock_notify: MagicMock, tmp_path: Path
    ) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        queue.submit("task1", lambda: "a")
        time.sleep(0.5)

        tasks = queue.list_tasks(include_completed=False)
        assert len(tasks) == 0  # Task already completed

    def test_list_tasks_sorted_newest_first(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        t1 = queue.submit("task1", lambda: "a")
        time.sleep(0.05)
        t2 = queue.submit("task2", lambda: "b")

        tasks = queue.list_tasks()
        assert tasks[0].id == t2.id
        assert tasks[1].id == t1.id

    def test_list_tasks_empty(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        tasks = queue.list_tasks()
        assert tasks == []


# =====================================================================
# TaskQueue — Results
# =====================================================================


class TestTaskQueueResults:
    """Tests for getting task results."""

    @patch("prism.tools.task_queue.TaskQueue._notify")
    def test_get_results_completed(
        self, mock_notify: MagicMock, tmp_path: Path
    ) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("test", lambda: "my-output")
        time.sleep(0.5)

        result = queue.get_results(task.id)
        assert result is not None
        assert result.output == "my-output"
        assert result.exit_code == 0

    def test_get_results_pending_returns_none(self, tmp_path: Path) -> None:
        barrier = threading.Event()

        def blocking_fn() -> str:
            barrier.wait(timeout=5)
            return "done"

        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("blocker", blocking_fn)

        result = queue.get_results(task.id)
        assert result is None

        barrier.set()

    def test_get_results_unknown_task(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        with pytest.raises(ValueError, match="Task not found"):
            queue.get_results("nonexistent")

    @patch("prism.tools.task_queue.TaskQueue._notify")
    def test_results_persisted_to_disk(
        self, mock_notify: MagicMock, tmp_path: Path
    ) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("persist", lambda: "saved-output")
        time.sleep(0.5)

        result_path = tmp_path / task.id / "result.json"
        assert result_path.is_file()

        data = json.loads(result_path.read_text(encoding="utf-8"))
        assert data["output"] == "saved-output"
        assert data["exit_code"] == 0

    @patch("prism.tools.task_queue.TaskQueue._notify")
    def test_task_metadata_persisted(
        self, mock_notify: MagicMock, tmp_path: Path
    ) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("meta-test", lambda: "ok", description="Test desc")
        time.sleep(0.5)

        meta_path = tmp_path / task.id / "task.json"
        assert meta_path.is_file()

        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["name"] == "meta-test"
        assert data["description"] == "Test desc"
        assert data["status"] == "completed"


# =====================================================================
# TaskQueue — Progress
# =====================================================================


class TestTaskQueueProgress:
    """Tests for task progress updates."""

    def test_update_progress(self, tmp_path: Path) -> None:
        barrier = threading.Event()

        def blocking_fn() -> str:
            barrier.wait(timeout=5)
            return "done"

        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("progress-test", blocking_fn)
        time.sleep(0.2)

        queue.update_progress(task.id, 0.5, "Halfway there")

        updated = queue.get_task(task.id)
        assert updated.progress == 0.5
        assert updated.progress_message == "Halfway there"

        barrier.set()

    def test_update_progress_invalid_value(self, tmp_path: Path) -> None:
        barrier = threading.Event()
        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("test", lambda: (barrier.wait(5), "ok")[-1])
        time.sleep(0.2)

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            queue.update_progress(task.id, 1.5)

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            queue.update_progress(task.id, -0.1)

        barrier.set()

    def test_update_progress_unknown_task(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        with pytest.raises(ValueError, match="Task not found"):
            queue.update_progress("nope", 0.5)


# =====================================================================
# TaskQueue — Cleanup
# =====================================================================


class TestTaskQueueCleanup:
    """Tests for cleaning up old tasks."""

    @patch("prism.tools.task_queue.TaskQueue._notify")
    def test_cleanup_removes_old_tasks(
        self, mock_notify: MagicMock, tmp_path: Path
    ) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        task = queue.submit("old-task", lambda: "done")
        time.sleep(0.5)

        # Manually set completed_at to 48 hours ago
        old_time = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        with queue._lock:
            queue._tasks[task.id].completed_at = old_time

        removed = queue.cleanup_completed(max_age_hours=24)
        assert removed == 1

        tasks = queue.list_tasks()
        assert len(tasks) == 0

    @patch("prism.tools.task_queue.TaskQueue._notify")
    def test_cleanup_keeps_recent_tasks(
        self, mock_notify: MagicMock, tmp_path: Path
    ) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        queue.submit("recent", lambda: "ok")
        time.sleep(0.5)

        removed = queue.cleanup_completed(max_age_hours=DEFAULT_CLEANUP_HOURS)
        assert removed == 0

    def test_cleanup_negative_hours_raises(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        with pytest.raises(ValueError, match="non-negative"):
            queue.cleanup_completed(max_age_hours=-1)

    def test_cleanup_empty_queue(self, tmp_path: Path) -> None:
        queue = TaskQueue(tasks_dir=tmp_path)
        removed = queue.cleanup_completed()
        assert removed == 0

    @patch("prism.tools.task_queue.TaskQueue._notify")
    def test_cleanup_ignores_running_tasks(
        self, mock_notify: MagicMock, tmp_path: Path
    ) -> None:
        barrier = threading.Event()

        def blocking_fn() -> str:
            barrier.wait(timeout=5)
            return "done"

        queue = TaskQueue(tasks_dir=tmp_path)
        queue.submit("running-task", blocking_fn)
        time.sleep(0.3)

        removed = queue.cleanup_completed(max_age_hours=0)
        assert removed == 0

        barrier.set()


# =====================================================================
# TaskQueue — Notifications
# =====================================================================


class TestTaskQueueNotifications:
    """Tests for desktop notification sending."""

    @patch("prism.tools.task_queue.platform.system", return_value="Darwin")
    @patch("prism.tools.task_queue.subprocess.run")
    def test_macos_notification(
        self, mock_run: MagicMock, mock_system: MagicMock
    ) -> None:
        task = BackgroundTask(
            id="abc", name="test", description="Test task",
            status=TaskStatus.COMPLETED, created_at="now",
        )
        TaskQueue._notify(task)

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "osascript"

    @patch("prism.tools.task_queue.platform.system", return_value="Linux")
    @patch("prism.tools.task_queue.subprocess.run")
    def test_linux_notification(
        self, mock_run: MagicMock, mock_system: MagicMock
    ) -> None:
        task = BackgroundTask(
            id="abc", name="test", description="Test task",
            status=TaskStatus.COMPLETED, created_at="now",
        )
        TaskQueue._notify(task)

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "notify-send"

    @patch("prism.tools.task_queue.platform.system", return_value="Windows")
    @patch("prism.tools.task_queue.subprocess.run")
    def test_windows_no_notification(
        self, mock_run: MagicMock, mock_system: MagicMock
    ) -> None:
        task = BackgroundTask(
            id="abc", name="test", description="Test task",
            status=TaskStatus.COMPLETED, created_at="now",
        )
        TaskQueue._notify(task)
        mock_run.assert_not_called()

    @patch("prism.tools.task_queue.platform.system", return_value="Darwin")
    @patch("prism.tools.task_queue.subprocess.run", side_effect=FileNotFoundError)
    def test_notification_failure_silenced(
        self, mock_run: MagicMock, mock_system: MagicMock
    ) -> None:
        task = BackgroundTask(
            id="abc", name="test", description="Test task",
            status=TaskStatus.COMPLETED, created_at="now",
        )
        # Should not raise
        TaskQueue._notify(task)

    @patch("prism.tools.task_queue.platform.system", return_value="Darwin")
    @patch("prism.tools.task_queue.subprocess.run")
    def test_notification_shows_failed_status(
        self, mock_run: MagicMock, mock_system: MagicMock
    ) -> None:
        task = BackgroundTask(
            id="abc", name="fail-task", description="Failing task",
            status=TaskStatus.FAILED, created_at="now",
        )
        TaskQueue._notify(task)

        args = mock_run.call_args[0][0]
        osascript_cmd = args[2]
        assert "failed" in osascript_cmd.lower()


# =====================================================================
# TaskQueue — Concurrent tasks
# =====================================================================


class TestTaskQueueConcurrency:
    """Tests for concurrent task execution."""

    @patch("prism.tools.task_queue.TaskQueue._notify")
    def test_multiple_tasks_run_in_parallel(
        self, mock_notify: MagicMock, tmp_path: Path
    ) -> None:
        results: list[str] = []
        lock = threading.Lock()

        def worker(name: str) -> str:
            time.sleep(0.1)
            with lock:
                results.append(name)
            return name

        queue = TaskQueue(tasks_dir=tmp_path, max_parallel=4)
        t1 = queue.submit("w1", worker, args=("a",))
        t2 = queue.submit("w2", worker, args=("b",))
        t3 = queue.submit("w3", worker, args=("c",))

        time.sleep(1.0)

        assert len(results) == 3
        assert set(results) == {"a", "b", "c"}

        for tid in [t1.id, t2.id, t3.id]:
            assert queue.get_task(tid).status == TaskStatus.COMPLETED

    @patch("prism.tools.task_queue.TaskQueue._notify")
    def test_active_count_tracks_running(
        self, mock_notify: MagicMock, tmp_path: Path
    ) -> None:
        barrier = threading.Event()

        def blocking_fn() -> str:
            barrier.wait(timeout=5)
            return "ok"

        queue = TaskQueue(tasks_dir=tmp_path, max_parallel=4)
        queue.submit("b1", blocking_fn)
        queue.submit("b2", blocking_fn)
        time.sleep(0.3)

        assert queue.active_count == 2

        barrier.set()
        time.sleep(0.5)

        assert queue.active_count == 0
