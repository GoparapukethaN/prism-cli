"""Tests for prism.context.session — SessionManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.exceptions import ContextError

if TYPE_CHECKING:
    from prism.context.session import SessionManager

# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_returns_uuid(self, session_mgr: SessionManager) -> None:
        sid = session_mgr.create_session("/projects/my-app")
        assert isinstance(sid, str)
        assert len(sid) == 36  # UUID format

    def test_creates_file(self, session_mgr: SessionManager) -> None:
        sid = session_mgr.create_session("/projects/my-app")
        path = session_mgr.sessions_dir / f"{sid}.json"
        assert path.is_file()

    def test_session_data_structure(self, session_mgr: SessionManager) -> None:
        sid = session_mgr.create_session("/projects/my-app")
        data = session_mgr.load_session(sid)
        assert data["session_id"] == sid
        assert data["project_root"] == "/projects/my-app"
        assert data["messages"] == []
        assert "created_at" in data
        assert "updated_at" in data


# ---------------------------------------------------------------------------
# save_session / load_session
# ---------------------------------------------------------------------------


class TestSaveLoadSession:
    def test_round_trip(self, session_mgr: SessionManager) -> None:
        sid = session_mgr.create_session("/tmp/proj")
        data = session_mgr.load_session(sid)
        data["messages"] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        data["metadata"] = {"model": "test-model"}
        session_mgr.save_session(sid, data)

        loaded = session_mgr.load_session(sid)
        assert len(loaded["messages"]) == 2
        assert loaded["messages"][0]["content"] == "Hello"
        assert loaded["metadata"]["model"] == "test-model"

    def test_save_updates_timestamp(self, session_mgr: SessionManager) -> None:
        sid = session_mgr.create_session("/tmp/proj")
        data1 = session_mgr.load_session(sid)
        original_updated = data1["updated_at"]

        import time
        time.sleep(0.01)

        session_mgr.save_session(sid, data1)
        data2 = session_mgr.load_session(sid)
        assert data2["updated_at"] >= original_updated

    def test_save_empty_session_id_raises(self, session_mgr: SessionManager) -> None:
        with pytest.raises(ContextError, match="empty"):
            session_mgr.save_session("", {"messages": []})

    def test_load_nonexistent_raises(self, session_mgr: SessionManager) -> None:
        with pytest.raises(ContextError, match="not found"):
            session_mgr.load_session("does-not-exist")

    def test_load_corrupt_file_raises(self, session_mgr: SessionManager) -> None:
        sid = "corrupt-session"
        path = session_mgr.sessions_dir / f"{sid}.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        with pytest.raises(ContextError, match="Failed to load"):
            session_mgr.load_session(sid)


# ---------------------------------------------------------------------------
# delete_session
# ---------------------------------------------------------------------------


class TestDeleteSession:
    def test_delete_existing(self, session_mgr: SessionManager) -> None:
        sid = session_mgr.create_session("/tmp/proj")
        assert session_mgr.delete_session(sid) is True
        assert not session_mgr.session_exists(sid)

    def test_delete_nonexistent(self, session_mgr: SessionManager) -> None:
        assert session_mgr.delete_session("nope") is False


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_list_empty(self, session_mgr: SessionManager) -> None:
        result = session_mgr.list_sessions()
        assert result == []

    def test_list_all(self, session_mgr: SessionManager) -> None:
        session_mgr.create_session("/proj/a")
        session_mgr.create_session("/proj/b")
        session_mgr.create_session("/proj/c")
        result = session_mgr.list_sessions()
        assert len(result) == 3

    def test_list_filtered_by_project(self, session_mgr: SessionManager) -> None:
        session_mgr.create_session("/proj/alpha")
        session_mgr.create_session("/proj/alpha")
        session_mgr.create_session("/proj/beta")
        result = session_mgr.list_sessions(project_root="/proj/alpha")
        assert len(result) == 2

    def test_list_sorted_by_updated(self, session_mgr: SessionManager) -> None:
        import time
        session_mgr.create_session("/proj/x")
        time.sleep(0.01)
        sid2 = session_mgr.create_session("/proj/x")
        result = session_mgr.list_sessions()
        # Most recently updated first
        assert result[0]["session_id"] == sid2

    def test_list_includes_message_count(self, session_mgr: SessionManager) -> None:
        sid = session_mgr.create_session("/proj/x")
        data = session_mgr.load_session(sid)
        data["messages"] = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
        ]
        session_mgr.save_session(sid, data)

        result = session_mgr.list_sessions()
        assert result[0]["message_count"] == 2

    def test_list_skips_corrupt_files(self, session_mgr: SessionManager) -> None:
        session_mgr.create_session("/proj/good")
        # Write a corrupt file
        (session_mgr.sessions_dir / "bad.json").write_text("!corrupt!")
        result = session_mgr.list_sessions()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# session_exists
# ---------------------------------------------------------------------------


class TestSessionExists:
    def test_exists(self, session_mgr: SessionManager) -> None:
        sid = session_mgr.create_session("/tmp/proj")
        assert session_mgr.session_exists(sid) is True

    def test_not_exists(self, session_mgr: SessionManager) -> None:
        assert session_mgr.session_exists("nope") is False
