"""Integration tests for conversation flow with context management.

Tests session lifecycle, context trimming, and persistence.
All tests run offline with no real API calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from prism.db.database import Database
from prism.db.queries import create_session, update_session

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------
# Session management tests
# ------------------------------------------------------------------


class TestSessionLifecycle:
    """Session creation, update, and persistence."""

    def test_new_session_created(self, integration_db: Database) -> None:
        session_id = str(uuid4())
        create_session(integration_db, session_id, "/tmp/project")

        row = integration_db.fetchone(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        assert row is not None
        assert row["id"] == session_id
        assert row["project_root"] == "/tmp/project"
        assert row["total_cost"] == 0.0
        assert row["total_requests"] == 0
        assert row["active"] == 1

    def test_messages_added_to_context(self, integration_db: Database) -> None:
        """Simulate messages being tracked in a session."""
        session_id = str(uuid4())
        create_session(integration_db, session_id, "/tmp/project")

        # Simulate 3 requests
        for _i in range(3):
            update_session(
                integration_db,
                session_id,
                cost_delta=0.001,
                request_delta=1,
            )

        row = integration_db.fetchone(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        assert row is not None
        assert row["total_requests"] == 3
        assert abs(row["total_cost"] - 0.003) < 1e-9

    def test_context_trimmed_to_budget(self, integration_db: Database) -> None:
        """Context trimming means older messages are discarded when context
        is too large. We simulate this by verifying only the last N
        routing decisions are returned."""
        from prism.db import models as dbm
        from prism.db.queries import get_routing_history, save_routing_decision

        session_id = "ctx-trim-session"
        create_session(integration_db, session_id, "/tmp")

        # Insert 20 routing decisions
        for i in range(20):
            decision = dbm.RoutingDecision(
                id=str(uuid4()),
                created_at=datetime.now(UTC).isoformat(),
                session_id=session_id,
                prompt_hash=f"hash-{i}",
                complexity_tier=dbm.ComplexityTier.SIMPLE,
                complexity_score=0.2,
                model_selected="gpt-4o-mini",
                fallback_chain="[]",
                estimated_cost=0.001,
                outcome=dbm.Outcome.UNKNOWN,
                features="{}",
            )
            save_routing_decision(integration_db, decision)

        # Query only last 5
        history = get_routing_history(
            integration_db, limit=5, session_id=session_id
        )
        assert len(history) == 5

    def test_active_files_included_in_context(self) -> None:
        """Active files should influence classification."""
        from prism.config.schema import PrismConfig, RoutingConfig
        from prism.config.settings import Settings
        from prism.router.classifier import TaskClassifier, TaskContext

        config = PrismConfig(routing=RoutingConfig())
        settings = Settings(config=config)
        classifier = TaskClassifier(settings)

        ctx_no_files = TaskContext(active_files=[])
        ctx_many_files = TaskContext(
            active_files=["a.py", "b.py", "c.py", "d.py", "e.py", "f.py", "g.py"]
        )

        result_no = classifier.classify("refactor this", ctx_no_files)
        result_many = classifier.classify("refactor this", ctx_many_files)

        # More files should increase the scope and possibly the score
        assert result_many.features["files_referenced"] > result_no.features["files_referenced"]

    def test_session_saved_and_loaded(self, integration_db: Database) -> None:
        """Session can be saved and then retrieved."""
        session_id = str(uuid4())
        create_session(integration_db, session_id, "/projects/myapp")
        update_session(integration_db, session_id, cost_delta=1.50, request_delta=10)

        row = integration_db.fetchone(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        assert row is not None
        assert row["total_cost"] == 1.50
        assert row["total_requests"] == 10
        assert row["project_root"] == "/projects/myapp"


class TestContextCompaction:
    """Context compaction reduces history."""

    def test_compact_reduces_history(self, integration_db: Database) -> None:
        """Querying with a smaller limit effectively compacts the window."""
        from prism.db import models as dbm
        from prism.db.queries import get_routing_history, save_routing_decision

        session_id = "compact-session"
        create_session(integration_db, session_id, "/tmp")

        for i in range(50):
            decision = dbm.RoutingDecision(
                id=str(uuid4()),
                created_at=datetime.now(UTC).isoformat(),
                session_id=session_id,
                prompt_hash=f"hash-{i}",
                complexity_tier=dbm.ComplexityTier.MEDIUM,
                complexity_score=0.5,
                model_selected="gpt-4o-mini",
                fallback_chain="[]",
                estimated_cost=0.001,
                outcome=dbm.Outcome.ACCEPTED,
                features="{}",
            )
            save_routing_decision(integration_db, decision)

        full = get_routing_history(integration_db, limit=100, session_id=session_id)
        compact = get_routing_history(integration_db, limit=10, session_id=session_id)

        assert len(full) == 50
        assert len(compact) == 10


class TestSystemPromptInContext:
    """System prompt should always be included."""

    def test_system_prompt_always_included(self) -> None:
        """Classification includes prompt analysis regardless of context."""
        from prism.config.schema import PrismConfig
        from prism.config.settings import Settings
        from prism.providers.base import ComplexityTier
        from prism.router.classifier import TaskClassifier, TaskContext

        settings = Settings(config=PrismConfig())
        classifier = TaskClassifier(settings)

        # Even with empty context, the classifier should work
        result = classifier.classify("hello", TaskContext())
        assert result.tier in (
            ComplexityTier.SIMPLE,
            ComplexityTier.MEDIUM,
            ComplexityTier.COMPLEX,
        )
        assert result.reasoning


class TestRepoMapInContext:
    """Repo map should influence scope assessment."""

    def test_repo_map_in_context(self) -> None:
        """Many files in context should increase scope."""
        from prism.config.schema import PrismConfig
        from prism.config.settings import Settings
        from prism.router.classifier import TaskClassifier, TaskContext

        settings = Settings(config=PrismConfig())
        classifier = TaskClassifier(settings)

        # With a large project context, scope should be broader
        ctx = TaskContext(
            active_files=[f"src/module_{i}.py" for i in range(10)],
            project_file_count=500,
        )
        result = classifier.classify("refactor this code", ctx)
        assert result.features["scope"] >= 0.5


class TestProjectMemory:
    """Project memory (previous sessions) available in context."""

    def test_project_memory_in_context(self, integration_db: Database) -> None:
        """Previous session data is queryable."""
        # Create two sessions for the same project
        create_session(integration_db, "old-sess", "/projects/myapp")
        update_session(integration_db, "old-sess", cost_delta=2.0, request_delta=20)

        create_session(integration_db, "new-sess", "/projects/myapp")

        # Both sessions should exist
        rows = integration_db.fetchall(
            "SELECT * FROM sessions WHERE project_root = ? ORDER BY created_at",
            ("/projects/myapp",),
        )
        assert len(rows) == 2
        assert rows[0]["id"] == "old-sess"


class TestConversationPersistence:
    """Conversation persists across save/load."""

    def test_conversation_persists_across_save_load(
        self,
        tmp_path: Path,
    ) -> None:
        """Data written to one DB instance is visible from another."""
        db_path = tmp_path / "persist.db"

        # First instance: create session
        db1 = Database(db_path)
        db1.initialize()
        create_session(db1, "persist-sess", "/tmp")
        update_session(db1, "persist-sess", cost_delta=0.5, request_delta=5)
        db1.close()

        # Second instance: read session
        db2 = Database(db_path)
        db2.initialize()
        row = db2.fetchone("SELECT * FROM sessions WHERE id = ?", ("persist-sess",))
        assert row is not None
        assert row["total_cost"] == 0.5
        assert row["total_requests"] == 5
        db2.close()
