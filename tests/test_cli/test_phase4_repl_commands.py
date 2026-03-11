"""Tests for Phase 4 REPL command enhancements.

Covers enhanced /cache, /image, /compare, /undo, /rollback,
/branch, /sandbox, and /tasks commands with comprehensive
mocking of all external dependencies.
"""

from __future__ import annotations

import io
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


def _make_state(
    pinned_model: str | None = None,
    active_files: list[str] | None = None,
    cache_enabled: bool = True,
    sandbox_enabled: bool = True,
    sandbox_type: str | None = None,
) -> _SessionState:
    """Create a SessionState with optional overrides."""
    state = _SessionState(
        pinned_model=pinned_model,
        cache_enabled=cache_enabled,
    )
    if active_files is not None:
        state.active_files = active_files
    state.session_id = "test-session"
    state.sandbox_enabled = sandbox_enabled
    state.sandbox_type = sandbox_type
    return state


def _cmd(
    command: str,
    tmp_path: Path,
    active_files: list[str] | None = None,
    pinned_model: str | None = None,
    console: Console | None = None,
    settings: Settings | None = None,
    state: _SessionState | None = None,
    dry_run: bool = False,
    offline: bool = False,
) -> tuple[str, str, Console, Settings, _SessionState]:
    """Run a slash command, return (action, output, ...)."""
    con = console or _make_console()
    stg = settings or _make_settings(tmp_path)
    st = state or _make_state(
        pinned_model=pinned_model,
        active_files=(
            active_files
            if active_files is not None else []
        ),
    )
    action = _dispatch_command(
        command,
        console=con,
        settings=stg,
        state=st,
        dry_run=dry_run,
        offline=offline,
    )
    return action, _get_output(con), con, stg, st


# -----------------------------------------------------------
# Mock data classes
# -----------------------------------------------------------

@dataclass
class _MockCacheStats:
    total_entries: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    tokens_saved: int = 0
    cost_saved: float = 0.0
    cache_size_bytes: int = 0
    oldest_entry: str | None = None
    newest_entry: str | None = None


@dataclass
class _MockSandboxResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time_ms: float = 10.0
    sandbox_type: str = "subprocess"
    timed_out: bool = False
    memory_exceeded: bool = False


@dataclass
class _MockBranchMeta:
    name: str = "test-branch"
    created_at: str = "2026-01-01T00:00:00"
    parent_branch: str = "main"
    fork_point_index: int = 0
    git_commit_at_fork: str = ""
    message_count: int = 5
    is_active: bool = False
    description: str = ""


@dataclass
class _MockChangeRecord:
    index: int = 1
    commit_hash: str = "abc12345" * 5
    short_hash: str = "abc12345"
    message: str = "Fix bug in router"
    timestamp: str = "2026-01-01T00:00:00"
    files_changed: list[str] = field(
        default_factory=lambda: ["router.py"],
    )
    insertions: int = 10
    deletions: int = 3


@dataclass
class _MockTimeline:
    session_id: str = "sess-1"
    changes: list[Any] = field(default_factory=list)
    start_commit: str = ""


@dataclass
class _MockConversationBranch:
    metadata: _MockBranchMeta = field(
        default_factory=_MockBranchMeta,
    )
    messages: list[dict[str, Any]] = field(
        default_factory=list,
    )
    file_edits: list[str] = field(
        default_factory=list,
    )


# ===========================================================
# TestCacheCommand
# ===========================================================

class TestCacheCommand:
    """Tests for enhanced /cache command."""

    def test_cache_shows_stats(
        self, tmp_path: Path,
    ) -> None:
        """``/cache`` shows stats panel."""
        mock_cache = MagicMock()
        mock_cache.get_stats.return_value = (
            _MockCacheStats(
                total_entries=42,
                total_hits=100,
                total_misses=50,
                hit_rate=0.667,
                tokens_saved=5000,
                cost_saved=0.1234,
                cache_size_bytes=2048,
            )
        )
        mock_cache.close = MagicMock()

        with patch(
            "prism.cache.response_cache.ResponseCache",
            return_value=mock_cache,
        ):
            action, out, _, _, _ = _cmd(
                "/cache", tmp_path,
            )

        assert action == "continue"
        assert "42" in out  # total_entries
        assert "100" in out  # total_hits
        assert "66.7%" in out  # hit rate

    def test_cache_stats_subcommand(
        self, tmp_path: Path,
    ) -> None:
        """``/cache stats`` shows hit rate, tokens, money."""
        mock_cache = MagicMock()
        mock_cache.get_stats.return_value = (
            _MockCacheStats(
                total_entries=10,
                total_hits=80,
                total_misses=20,
                hit_rate=0.80,
                tokens_saved=12000,
                cost_saved=0.5678,
                cache_size_bytes=4096,
            )
        )
        mock_cache.close = MagicMock()

        with patch(
            "prism.cache.response_cache.ResponseCache",
            return_value=mock_cache,
        ):
            action, out, _, _, _ = _cmd(
                "/cache stats", tmp_path,
            )

        assert action == "continue"
        assert "80.0%" in out
        assert "12,000" in out
        assert "$0.5678" in out

    def test_cache_clear_all(
        self, tmp_path: Path,
    ) -> None:
        """``/cache clear`` clears all entries."""
        mock_cache = MagicMock()
        mock_cache.clear.return_value = 15
        mock_cache.close = MagicMock()

        with patch(
            "prism.cache.response_cache.ResponseCache",
            return_value=mock_cache,
        ):
            action, out, _, _, _ = _cmd(
                "/cache clear", tmp_path,
            )

        assert action == "continue"
        assert "15" in out
        assert "cleared" in out.lower()

    def test_cache_clear_older_than(
        self, tmp_path: Path,
    ) -> None:
        """``/cache clear --older-than 24h`` parses dur."""
        mock_cache = MagicMock()
        mock_cache.clear.return_value = 5
        mock_cache.close = MagicMock()

        with patch(
            "prism.cache.response_cache.ResponseCache",
            return_value=mock_cache,
        ):
            action, out, _, _, _ = _cmd(
                "/cache clear --older-than 24h",
                tmp_path,
            )

        assert action == "continue"
        assert "5" in out
        mock_cache.clear.assert_called_once()

    def test_cache_off(self, tmp_path: Path) -> None:
        """``/cache off`` disables caching."""
        state = _make_state(cache_enabled=True)
        action, out, _, _, st = _cmd(
            "/cache off", tmp_path, state=state,
        )
        assert action == "continue"
        assert not st.cache_enabled
        assert "disabled" in out.lower()

    def test_cache_on(self, tmp_path: Path) -> None:
        """``/cache on`` enables caching."""
        state = _make_state(cache_enabled=False)
        action, out, _, _, st = _cmd(
            "/cache on", tmp_path, state=state,
        )
        assert action == "continue"
        assert st.cache_enabled
        assert "enabled" in out.lower()

    def test_cache_stats_zero_entries(
        self, tmp_path: Path,
    ) -> None:
        """Stats with zero entries shows zeros."""
        mock_cache = MagicMock()
        mock_cache.get_stats.return_value = (
            _MockCacheStats()
        )
        mock_cache.close = MagicMock()

        with patch(
            "prism.cache.response_cache.ResponseCache",
            return_value=mock_cache,
        ):
            action, out, _, _, _ = _cmd(
                "/cache", tmp_path,
            )

        assert action == "continue"
        assert "0" in out

    def test_cache_stats_with_entries(
        self, tmp_path: Path,
    ) -> None:
        """Stats with populated cache shows all fields."""
        mock_cache = MagicMock()
        mock_cache.get_stats.return_value = (
            _MockCacheStats(
                total_entries=99,
                total_hits=500,
                total_misses=200,
                hit_rate=0.714,
                tokens_saved=25000,
                cost_saved=1.2345,
                cache_size_bytes=1_048_576,
                oldest_entry="2026-01-01T00:00:00",
                newest_entry="2026-03-10T12:00:00",
            )
        )
        mock_cache.close = MagicMock()

        with patch(
            "prism.cache.response_cache.ResponseCache",
            return_value=mock_cache,
        ):
            action, out, _, _, _ = _cmd(
                "/cache", tmp_path,
            )

        assert action == "continue"
        assert "99" in out
        assert "25,000" in out
        assert "2026-01-01" in out

    def test_cache_invalid_older_than(
        self, tmp_path: Path,
    ) -> None:
        """Invalid duration string shows error."""
        mock_cache = MagicMock()
        mock_cache.close = MagicMock()

        with patch(
            "prism.cache.response_cache.ResponseCache",
            return_value=mock_cache,
        ):
            action, out, _, _, _ = _cmd(
                "/cache clear --older-than xyz",
                tmp_path,
            )

        assert action == "continue"
        assert "invalid" in out.lower()

    def test_cache_returns_continue(
        self, tmp_path: Path,
    ) -> None:
        """All cache subcommands return 'continue'."""
        state = _make_state()
        action, _, _, _, _ = _cmd(
            "/cache off", tmp_path, state=state,
        )
        assert action == "continue"


# ===========================================================
# TestImageCommand
# ===========================================================

class TestImageCommand:
    """Tests for enhanced /image command."""

    def test_image_no_args_shows_usage(
        self, tmp_path: Path,
    ) -> None:
        """``/image`` with no args shows usage."""
        action, out, _, _, _ = _cmd("/image", tmp_path)
        assert action == "continue"
        assert "usage" in out.lower()

    def test_image_nonexistent_file(
        self, tmp_path: Path,
    ) -> None:
        """``/image missing.png`` shows error."""
        action, out, _, _, _ = _cmd(
            "/image missing.png", tmp_path,
        )
        assert action == "continue"
        # Either "not found" or "no valid image"
        lower = out.lower()
        assert (
            "not found" in lower
            or "no valid" in lower
        )

    def test_image_processes_with_prompt(
        self, tmp_path: Path,
    ) -> None:
        """``/image path.png "prompt"`` processes."""
        img_path = tmp_path / "test.png"
        # Create a minimal valid PNG (1x1 pixel)
        png_data = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01"  # width
            b"\x00\x00\x00\x01"  # height
            b"\x08\x02"  # bit depth, color type
            b"\x00\x00\x00"  # compression, filter, interlace
            b"\x90wS\xde"  # CRC
            b"\x00\x00\x00\x0cIDATx"
            b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
            b"\x18\xd8N"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        img_path.write_bytes(png_data)

        mock_att = MagicMock()
        mock_att.size_bytes = 100
        mock_att.width = 1
        mock_att.height = 1
        mock_att.was_compressed = False
        mock_att.to_message_content.return_value = {
            "type": "image",
        }
        mock_att.original_path = img_path
        mock_att.media_type = "image/png"
        mock_att.base64_data = "AAAA"

        mock_result = MagicMock()
        mock_result.content = "A test image."
        mock_result.cost_usd = 0.001
        mock_result.input_tokens = 100
        mock_result.output_tokens = 50

        with (
            patch(
                "prism.tools.vision.process_image",
                return_value=mock_att,
            ),
            patch(
                "prism.tools.vision."
                "detect_terminal_image_support",
                return_value=None,
            ),
            patch(
                "prism.tools.vision."
                "build_multimodal_messages",
                return_value=[
                    {"type": "text", "text": "hi"},
                ],
            ),
            patch(
                "prism.llm.completion.CompletionEngine",
            ) as mock_engine_cls,
        ):
            mock_engine = MagicMock()
            mock_engine_cls.return_value = mock_engine

            import asyncio
            future: asyncio.Future[Any] = (
                asyncio.Future()
            )
            future.set_result(mock_result)
            mock_engine.complete.return_value = future

            action, _out, _, _, _ = _cmd(
                f'/image {img_path} "describe this"',
                tmp_path,
            )

        assert action == "continue"

    def test_image_auto_routes_to_vision(
        self, tmp_path: Path,
    ) -> None:
        """Auto-routes when current model lacks vision."""
        img_path = tmp_path / "photo.jpg"
        img_path.write_bytes(
            b"\xff\xd8\xff\xe0" + b"\x00" * 100,
        )

        state = _make_state(pinned_model="gpt-3.5-turbo")

        mock_att = MagicMock()
        mock_att.size_bytes = 50
        mock_att.width = 0
        mock_att.height = 0
        mock_att.was_compressed = False

        mock_result = MagicMock()
        mock_result.content = "A photo."
        mock_result.cost_usd = 0.0
        mock_result.input_tokens = 0
        mock_result.output_tokens = 0

        with (
            patch(
                "prism.tools.vision.process_image",
                return_value=mock_att,
            ),
            patch(
                "prism.tools.vision."
                "detect_terminal_image_support",
                return_value=None,
            ),
            patch(
                "prism.tools.vision."
                "build_multimodal_messages",
                return_value=[],
            ),
            patch(
                "prism.llm.completion.CompletionEngine",
            ) as mock_eng,
        ):
            mock_e = MagicMock()
            mock_eng.return_value = mock_e
            import asyncio
            fut: asyncio.Future[Any] = asyncio.Future()
            fut.set_result(mock_result)
            mock_e.complete.return_value = fut

            action, out, _, _, _ = _cmd(
                f"/image {img_path}",
                tmp_path,
                state=state,
            )

        assert action == "continue"
        # Should mention the non-vision model being replaced
        lower = out.lower()
        assert (
            "does not support vision" in lower
            or "vision model" in lower
        )

    def test_image_multiple_paths(
        self, tmp_path: Path,
    ) -> None:
        """Multiple image paths are accepted."""
        img1 = tmp_path / "a.png"
        img2 = tmp_path / "b.png"
        png_data = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00"
            b"\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx"
            b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
            b"\x18\xd8N"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        img1.write_bytes(png_data)
        img2.write_bytes(png_data)

        mock_att = MagicMock()
        mock_att.size_bytes = 50
        mock_att.width = 1
        mock_att.height = 1
        mock_att.was_compressed = False

        mock_result = MagicMock()
        mock_result.content = "Two images."
        mock_result.cost_usd = 0.0
        mock_result.input_tokens = 0
        mock_result.output_tokens = 0

        with (
            patch(
                "prism.tools.vision.process_image",
                return_value=mock_att,
            ),
            patch(
                "prism.tools.vision."
                "detect_terminal_image_support",
                return_value=None,
            ),
            patch(
                "prism.tools.vision."
                "build_multimodal_messages",
                return_value=[],
            ),
            patch(
                "prism.llm.completion.CompletionEngine",
            ) as mock_eng,
        ):
            mock_e = MagicMock()
            mock_eng.return_value = mock_e
            import asyncio
            fut: asyncio.Future[Any] = asyncio.Future()
            fut.set_result(mock_result)
            mock_e.complete.return_value = fut

            action, _out, _, _, _ = _cmd(
                f"/image {img1} {img2}",
                tmp_path,
            )

        assert action == "continue"

    def test_image_unsupported_format(
        self, tmp_path: Path,
    ) -> None:
        """Unsupported image format shows error."""
        bad_file = tmp_path / "doc.pdf"
        bad_file.write_text("not an image")

        action, out, _, _, _ = _cmd(
            f"/image {bad_file}", tmp_path,
        )
        assert action == "continue"
        lower = out.lower()
        assert (
            "no valid" in lower
            or "not found" in lower
            or "unsupported" in lower
        )

    def test_image_terminal_preview_detection(
        self, tmp_path: Path,
    ) -> None:
        """Terminal preview is attempted when detected."""
        img_path = tmp_path / "shot.png"
        png_data = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00"
            b"\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx"
            b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
            b"\x18\xd8N"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        img_path.write_bytes(png_data)

        mock_att = MagicMock()
        mock_att.size_bytes = 50
        mock_att.width = 1
        mock_att.height = 1
        mock_att.was_compressed = False

        mock_result = MagicMock()
        mock_result.content = "image"
        mock_result.cost_usd = 0.0
        mock_result.input_tokens = 0
        mock_result.output_tokens = 0

        with (
            patch(
                "prism.tools.vision.process_image",
                return_value=mock_att,
            ),
            patch(
                "prism.tools.vision."
                "detect_terminal_image_support",
                return_value="iterm2",
            ),
            patch(
                "prism.tools.vision."
                "display_image_preview",
                return_value=True,
            ) as mock_preview,
            patch(
                "prism.tools.vision."
                "build_multimodal_messages",
                return_value=[],
            ),
            patch(
                "prism.llm.completion.CompletionEngine",
            ) as mock_eng,
        ):
            mock_e = MagicMock()
            mock_eng.return_value = mock_e
            import asyncio
            fut: asyncio.Future[Any] = asyncio.Future()
            fut.set_result(mock_result)
            mock_e.complete.return_value = fut

            _cmd(f"/image {img_path}", tmp_path)
            mock_preview.assert_called_once()

    def test_image_multimodal_message_building(
        self, tmp_path: Path,
    ) -> None:
        """build_multimodal_messages is called."""
        img_path = tmp_path / "img.png"
        png_data = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00"
            b"\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx"
            b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
            b"\x18\xd8N"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        img_path.write_bytes(png_data)

        mock_att = MagicMock()
        mock_att.size_bytes = 50
        mock_att.width = 1
        mock_att.height = 1
        mock_att.was_compressed = False

        mock_result = MagicMock()
        mock_result.content = "desc"
        mock_result.cost_usd = 0.0
        mock_result.input_tokens = 0
        mock_result.output_tokens = 0

        with (
            patch(
                "prism.tools.vision.process_image",
                return_value=mock_att,
            ),
            patch(
                "prism.tools.vision."
                "detect_terminal_image_support",
                return_value=None,
            ),
            patch(
                "prism.tools.vision."
                "build_multimodal_messages",
                return_value=[
                    {"type": "text", "text": "hi"},
                ],
            ) as mock_build,
            patch(
                "prism.llm.completion.CompletionEngine",
            ) as mock_eng,
        ):
            mock_e = MagicMock()
            mock_eng.return_value = mock_e
            import asyncio
            fut: asyncio.Future[Any] = asyncio.Future()
            fut.set_result(mock_result)
            mock_e.complete.return_value = fut

            _cmd(f"/image {img_path}", tmp_path)
            mock_build.assert_called_once()


# ===========================================================
# TestCompareEnhancements
# ===========================================================

class TestCompareEnhancements:
    """Tests for enhanced /compare command."""

    def test_compare_config_shows_models(
        self, tmp_path: Path,
    ) -> None:
        """``/compare config`` shows current models."""
        mock_comparator = MagicMock()
        mock_comparator.display_config = MagicMock()

        with (
            patch(
                "prism.cli.compare.ModelComparator",
                return_value=mock_comparator,
            ),
            patch(
                "prism.llm.completion.CompletionEngine",
            ),
        ):
            action, _out, _, _, _ = _cmd(
                "/compare config", tmp_path,
            )

        assert action == "continue"
        mock_comparator.display_config.assert_called_once()

    def test_compare_history_shows_past(
        self, tmp_path: Path,
    ) -> None:
        """``/compare history`` shows past comparisons."""
        mock_comparator = MagicMock()
        mock_comparator.history = []

        with (
            patch(
                "prism.cli.compare.ModelComparator",
                return_value=mock_comparator,
            ),
            patch(
                "prism.llm.completion.CompletionEngine",
            ),
        ):
            action, out, _, _, _ = _cmd(
                "/compare history", tmp_path,
            )

        assert action == "continue"
        assert "no comparison history" in out.lower()

    def test_compare_prompt_shows_cost_estimate(
        self, tmp_path: Path,
    ) -> None:
        """``/compare prompt`` shows cost estimate."""
        mock_comparator = MagicMock()
        mock_comparator.models = ["gpt-4o"]

        mock_session = MagicMock()

        import asyncio
        fut: asyncio.Future[Any] = asyncio.Future()
        fut.set_result(mock_session)
        mock_comparator.compare.return_value = fut

        mock_pricing = MagicMock()
        mock_pricing.input_cost_per_token = 0.00001
        mock_pricing.output_cost_per_token = 0.00003

        with (
            patch(
                "prism.cli.compare.ModelComparator",
                return_value=mock_comparator,
            ),
            patch(
                "prism.cli.compare.MODEL_DISPLAY_NAMES",
                {"gpt-4o": "GPT-4o"},
            ),
            patch(
                "prism.llm.completion.CompletionEngine",
            ),
            patch(
                "prism.cost.pricing.get_model_pricing",
                return_value=mock_pricing,
            ),
        ):
            action, out, _, _, _ = _cmd(
                "/compare explain quantum physics",
                tmp_path,
            )

        assert action == "continue"
        lower = out.lower()
        assert (
            "estimated cost" in lower
            or "running comparison" in lower
        )

    def test_compare_empty_prompt_shows_usage(
        self, tmp_path: Path,
    ) -> None:
        """``/compare`` with no prompt shows usage."""
        action, out, _, _, _ = _cmd(
            "/compare", tmp_path,
        )
        assert action == "continue"
        assert "usage" in out.lower()

    def test_compare_with_default_models(
        self, tmp_path: Path,
    ) -> None:
        """Compare uses default model set."""
        mock_comparator = MagicMock()
        mock_comparator.models = [
            "claude-sonnet-4-20250514",
            "gpt-4o",
        ]

        mock_session = MagicMock()
        import asyncio
        fut: asyncio.Future[Any] = asyncio.Future()
        fut.set_result(mock_session)
        mock_comparator.compare.return_value = fut

        with (
            patch(
                "prism.cli.compare.ModelComparator",
                return_value=mock_comparator,
            ),
            patch(
                "prism.cli.compare.MODEL_DISPLAY_NAMES",
                {},
            ),
            patch(
                "prism.llm.completion.CompletionEngine",
            ),
            patch(
                "prism.cost.pricing.get_model_pricing",
                side_effect=ValueError("unknown"),
            ),
        ):
            action, out, _, _, _ = _cmd(
                "/compare test prompt", tmp_path,
            )

        assert action == "continue"
        assert "pricing unavailable" in out.lower()

    def test_compare_config_display(
        self, tmp_path: Path,
    ) -> None:
        """Config subcommand returns continue."""
        mock_comparator = MagicMock()
        mock_comparator.display_config = MagicMock()

        with (
            patch(
                "prism.cli.compare.ModelComparator",
                return_value=mock_comparator,
            ),
            patch(
                "prism.llm.completion.CompletionEngine",
            ),
        ):
            action, _, _, _, _ = _cmd(
                "/compare config", tmp_path,
            )
        assert action == "continue"


# ===========================================================
# TestUndoEnhancements
# ===========================================================

class TestUndoEnhancements:
    """Tests for enhanced /undo command."""

    def test_undo_last_change(
        self, tmp_path: Path,
    ) -> None:
        """``/undo`` undoes the last change."""
        mock_manager = MagicMock()
        mock_manager.start_session = MagicMock()
        mock_manager.change_count = 3
        mock_manager.undo.return_value = ["abc12345"]
        mock_manager.get_timeline.return_value = (
            _MockTimeline(
                changes=[_MockChangeRecord()],
            )
        )
        mock_manager.get_diff.return_value = (
            "- old\n+ new"
        )

        with patch(
            "prism.git.history.RollbackManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/undo", tmp_path,
            )

        assert action == "continue"
        lower = out.lower()
        assert "undone" in lower or "undo" in lower

    def test_undo_n_changes(
        self, tmp_path: Path,
    ) -> None:
        """``/undo 3`` undoes 3 changes."""
        mock_manager = MagicMock()
        mock_manager.start_session = MagicMock()
        mock_manager.change_count = 5
        mock_manager.undo.return_value = [
            "aaa", "bbb", "ccc",
        ]
        mock_manager.get_timeline.return_value = (
            _MockTimeline(
                changes=[
                    _MockChangeRecord(
                        index=i, short_hash=f"h{i}",
                    )
                    for i in range(1, 4)
                ],
            )
        )

        with patch(
            "prism.git.history.RollbackManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/undo 3", tmp_path,
            )

        assert action == "continue"
        assert "3" in out

    def test_undo_all(
        self, tmp_path: Path,
    ) -> None:
        """``/undo all`` undoes every change."""
        mock_manager = MagicMock()
        mock_manager.start_session = MagicMock()
        mock_manager.change_count = 2
        mock_manager.undo.return_value = ["a", "b"]
        mock_manager.get_timeline.return_value = (
            _MockTimeline(
                changes=[
                    _MockChangeRecord(index=1),
                    _MockChangeRecord(index=2),
                ],
            )
        )

        with patch(
            "prism.git.history.RollbackManager",
            return_value=mock_manager,
        ):
            action, _out, _, _, _ = _cmd(
                "/undo all", tmp_path,
            )

        assert action == "continue"
        mock_manager.undo.assert_called_once_with(
            count=2,
        )

    def test_undo_no_changes(
        self, tmp_path: Path,
    ) -> None:
        """``/undo`` with no changes shows message."""
        mock_manager = MagicMock()
        mock_manager.start_session = MagicMock()
        mock_manager.change_count = 0
        mock_manager.undo.return_value = []
        mock_manager.get_timeline.return_value = (
            _MockTimeline()
        )

        with patch(
            "prism.git.history.RollbackManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/undo", tmp_path,
            )

        assert action == "continue"
        assert "nothing" in out.lower()

    def test_undo_invalid_number(
        self, tmp_path: Path,
    ) -> None:
        """``/undo xyz`` shows usage."""
        mock_manager = MagicMock()
        mock_manager.start_session = MagicMock()

        with patch(
            "prism.git.history.RollbackManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/undo xyz", tmp_path,
            )

        assert action == "continue"
        assert "usage" in out.lower()

    def test_undo_shows_diff(
        self, tmp_path: Path,
    ) -> None:
        """Diff is shown after single undo."""
        mock_manager = MagicMock()
        mock_manager.start_session = MagicMock()
        mock_manager.change_count = 1
        mock_manager.undo.return_value = ["abc12345"]
        mock_manager.get_timeline.return_value = (
            _MockTimeline(
                changes=[_MockChangeRecord()],
            )
        )
        mock_manager.get_diff.return_value = (
            "- removed line\n+ added line"
        )

        with patch(
            "prism.git.history.RollbackManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/undo", tmp_path,
            )

        assert action == "continue"
        # Diff content or panel is displayed
        assert (
            "removed line" in out
            or "Diff" in out
            or "undone" in out.lower()
        )

    def test_undo_shows_affected_files(
        self, tmp_path: Path,
    ) -> None:
        """Affected files listed after undo."""
        change = _MockChangeRecord(
            files_changed=["router.py", "cli.py"],
        )
        mock_manager = MagicMock()
        mock_manager.start_session = MagicMock()
        mock_manager.change_count = 1
        mock_manager.undo.return_value = ["abc12345"]
        mock_manager.get_timeline.return_value = (
            _MockTimeline(changes=[change])
        )
        mock_manager.get_diff.return_value = ""

        with patch(
            "prism.git.history.RollbackManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/undo", tmp_path,
            )

        assert action == "continue"
        assert "router.py" in out

    def test_undo_zero_shows_error(
        self, tmp_path: Path,
    ) -> None:
        """``/undo 0`` shows error."""
        mock_manager = MagicMock()
        mock_manager.start_session = MagicMock()

        with patch(
            "prism.git.history.RollbackManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/undo 0", tmp_path,
            )

        assert action == "continue"
        lower = out.lower()
        assert "at least 1" in lower


# ===========================================================
# TestRollbackEnhancements
# ===========================================================

class TestRollbackEnhancements:
    """Tests for enhanced /rollback command."""

    def test_rollback_list_shows_timeline(
        self, tmp_path: Path,
    ) -> None:
        """``/rollback list`` shows timeline table."""
        mock_manager = MagicMock()
        mock_manager.start_session = MagicMock()
        mock_manager.get_timeline.return_value = (
            _MockTimeline(
                changes=[
                    _MockChangeRecord(index=1),
                    _MockChangeRecord(index=2),
                ],
            )
        )

        with patch(
            "prism.git.history.RollbackManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/rollback list", tmp_path,
            )

        assert action == "continue"
        assert "Session Timeline" in out

    def test_rollback_diff_shows_colored_diff(
        self, tmp_path: Path,
    ) -> None:
        """``/rollback diff 1`` shows diff panel."""
        change = _MockChangeRecord(index=1)
        mock_manager = MagicMock()
        mock_manager.start_session = MagicMock()
        mock_manager.get_change.return_value = change
        mock_manager.get_diff.return_value = (
            "- old\n+ new"
        )

        with patch(
            "prism.git.history.RollbackManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/rollback diff 1", tmp_path,
            )

        assert action == "continue"
        assert "Change #1" in out

    def test_rollback_restore_hash(
        self, tmp_path: Path,
    ) -> None:
        """``/rollback restore <hash>`` restores state."""
        mock_manager = MagicMock()
        mock_manager.start_session = MagicMock()
        mock_manager.get_restore_preview.return_value = (
            "files to restore"
        )
        mock_manager.restore.return_value = "new12345"

        with patch(
            "prism.git.history.RollbackManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/rollback restore abc12345",
                tmp_path,
            )

        assert action == "continue"
        assert "Restored" in out

    def test_rollback_empty_timeline(
        self, tmp_path: Path,
    ) -> None:
        """Empty timeline shows message."""
        mock_manager = MagicMock()
        mock_manager.start_session = MagicMock()
        mock_manager.get_timeline.return_value = (
            _MockTimeline()
        )

        with patch(
            "prism.git.history.RollbackManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/rollback list", tmp_path,
            )

        assert action == "continue"
        assert "no changes" in out.lower()

    def test_rollback_invalid_diff_index(
        self, tmp_path: Path,
    ) -> None:
        """Invalid diff index shows usage."""
        mock_manager = MagicMock()
        mock_manager.start_session = MagicMock()

        with patch(
            "prism.git.history.RollbackManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/rollback diff abc", tmp_path,
            )

        assert action == "continue"
        assert "usage" in out.lower()

    def test_rollback_restore_no_hash(
        self, tmp_path: Path,
    ) -> None:
        """Missing hash shows usage."""
        mock_manager = MagicMock()
        mock_manager.start_session = MagicMock()

        with patch(
            "prism.git.history.RollbackManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/rollback restore", tmp_path,
            )

        assert action == "continue"
        assert "usage" in out.lower()


# ===========================================================
# TestBranchEnhancements
# ===========================================================

class TestBranchEnhancements:
    """Tests for enhanced /branch command."""

    def test_branch_create(
        self, tmp_path: Path,
    ) -> None:
        """``/branch test-branch`` creates a branch."""
        mock_manager = MagicMock()
        mock_manager.branch_count = 0
        mock_manager.create_branch.return_value = (
            _MockBranchMeta(name="test-branch")
        )

        with patch(
            "prism.context.branching.BranchManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/branch test-branch", tmp_path,
            )

        assert action == "continue"
        assert "test-branch" in out
        assert "created" in out.lower()

    def test_branch_list(
        self, tmp_path: Path,
    ) -> None:
        """``/branch list`` shows all branches."""
        mock_manager = MagicMock()
        mock_manager.list_branches.return_value = [
            _MockBranchMeta(
                name="main",
                message_count=10,
                description="Main branch",
            ),
            _MockBranchMeta(
                name="feature",
                message_count=5,
                description="Feature work",
            ),
        ]
        mock_manager.active_branch = "main"

        with patch(
            "prism.context.branching.BranchManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/branch list", tmp_path,
            )

        assert action == "continue"
        assert "main" in out
        assert "feature" in out

    def test_branch_switch(
        self, tmp_path: Path,
    ) -> None:
        """``/branch switch test-branch`` switches."""
        mock_manager = MagicMock()
        mock_manager.switch_branch.return_value = [
            {"role": "user", "content": "hi"},
        ]
        branch = _MockConversationBranch()
        mock_manager.get_branch.return_value = branch

        with patch(
            "prism.context.branching.BranchManager",
            return_value=mock_manager,
        ):
            state = _make_state()
            action, out, _, _, _st = _cmd(
                "/branch switch test-branch",
                tmp_path,
                state=state,
            )

        assert action == "continue"
        assert "Switched" in out

    def test_branch_merge(
        self, tmp_path: Path,
    ) -> None:
        """``/branch merge test-branch`` merges."""
        mock_manager = MagicMock()
        source = _MockConversationBranch()
        source.metadata.fork_point_index = 0
        source.messages = [
            {"role": "user", "content": "x"},
        ]
        mock_manager.get_branch.return_value = source
        mock_manager.active_branch = "main"
        mock_manager.merge_branch.return_value = [
            {"role": "user", "content": "merged"},
        ]

        with patch(
            "prism.context.branching.BranchManager",
            return_value=mock_manager,
        ):
            state = _make_state()
            action, out, _, _, _ = _cmd(
                "/branch merge test-branch",
                tmp_path,
                state=state,
            )

        assert action == "continue"
        assert "Merged" in out

    def test_branch_delete(
        self, tmp_path: Path,
    ) -> None:
        """``/branch delete test-branch`` deletes."""
        mock_manager = MagicMock()

        with patch(
            "prism.context.branching.BranchManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/branch delete test-branch",
                tmp_path,
            )

        assert action == "continue"
        assert "deleted" in out.lower()

    def test_branch_save(
        self, tmp_path: Path,
    ) -> None:
        """``/branch save`` marks persistent."""
        mock_manager = MagicMock()
        mock_manager.active_branch = "feature"
        branch = _MockConversationBranch()
        branch.metadata.name = "feature"
        branch.metadata.description = ""
        mock_manager.list_branches.return_value = [
            branch.metadata,
        ]
        mock_manager.get_branch.return_value = branch

        with patch(
            "prism.context.branching.BranchManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/branch save", tmp_path,
            )

        assert action == "continue"
        assert "persistent" in out.lower()

    def test_branch_max_enforced(
        self, tmp_path: Path,
    ) -> None:
        """Max 20 branches enforced."""
        mock_manager = MagicMock()
        mock_manager.branch_count = 20

        with patch(
            "prism.context.branching.BranchManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/branch new-branch", tmp_path,
            )

        assert action == "continue"
        assert "limit" in out.lower()

    def test_branch_duplicate_name(
        self, tmp_path: Path,
    ) -> None:
        """Duplicate branch name shows error."""
        mock_manager = MagicMock()
        mock_manager.branch_count = 1
        mock_manager.create_branch.side_effect = (
            ValueError("Branch 'dup' already exists")
        )

        with patch(
            "prism.context.branching.BranchManager",
            return_value=mock_manager,
        ):
            action, out, _, _, _ = _cmd(
                "/branch dup", tmp_path,
            )

        assert action == "continue"
        assert "already exists" in out


# ===========================================================
# TestSandboxEnhancements
# ===========================================================

class TestSandboxEnhancements:
    """Tests for enhanced /sandbox command."""

    def test_sandbox_execute(
        self, tmp_path: Path,
    ) -> None:
        """``/sandbox print("hello")`` executes code."""
        mock_sandbox = MagicMock()
        mock_sandbox.execute.return_value = (
            _MockSandboxResult(
                stdout="hello\n",
                exit_code=0,
                execution_time_ms=12.5,
                sandbox_type="subprocess",
            )
        )

        state = _make_state(sandbox_enabled=True)

        with patch(
            "prism.tools.code_sandbox.CodeSandbox",
            return_value=mock_sandbox,
        ):
            action, out, _, _, _ = _cmd(
                '/sandbox print("hello")',
                tmp_path,
                state=state,
            )

        assert action == "continue"
        assert "hello" in out

    def test_sandbox_on(
        self, tmp_path: Path,
    ) -> None:
        """``/sandbox on`` enables sandbox."""
        state = _make_state(sandbox_enabled=False)
        action, out, _, _, st = _cmd(
            "/sandbox on", tmp_path, state=state,
        )
        assert action == "continue"
        assert st.sandbox_enabled
        assert "enabled" in out.lower()

    def test_sandbox_off(
        self, tmp_path: Path,
    ) -> None:
        """``/sandbox off`` disables sandbox."""
        state = _make_state(sandbox_enabled=True)
        action, out, _, _, st = _cmd(
            "/sandbox off", tmp_path, state=state,
        )
        assert action == "continue"
        assert not st.sandbox_enabled
        assert "disabled" in out.lower()

    def test_sandbox_docker(
        self, tmp_path: Path,
    ) -> None:
        """``/sandbox docker`` sets type to docker."""
        state = _make_state()
        action, out, _, _, st = _cmd(
            "/sandbox docker", tmp_path, state=state,
        )
        assert action == "continue"
        assert st.sandbox_type == "docker"
        assert "Docker" in out

    def test_sandbox_subprocess(
        self, tmp_path: Path,
    ) -> None:
        """``/sandbox subprocess`` sets type."""
        state = _make_state()
        action, _out, _, _, st = _cmd(
            "/sandbox subprocess", tmp_path, state=state,
        )
        assert action == "continue"
        assert st.sandbox_type == "subprocess"

    def test_sandbox_status(
        self, tmp_path: Path,
    ) -> None:
        """``/sandbox status`` shows configuration."""
        state = _make_state(
            sandbox_enabled=True,
            sandbox_type="docker",
        )

        with patch(
            "prism.tools.code_sandbox.DEFAULT_TIMEOUT",
            30,
        ), patch(
            "prism.tools.code_sandbox.DEFAULT_MEMORY_MB",
            512,
        ):
            action, out, _, _, _ = _cmd(
                "/sandbox status",
                tmp_path,
                state=state,
            )

        assert action == "continue"
        lower = out.lower()
        assert "sandbox" in lower

    def test_sandbox_language_auto_detect(
        self, tmp_path: Path,
    ) -> None:
        """Language auto-detection picks python."""
        mock_sandbox = MagicMock()
        mock_sandbox.execute.return_value = (
            _MockSandboxResult(
                stdout="ok",
                exit_code=0,
                sandbox_type="subprocess",
            )
        )

        state = _make_state(sandbox_enabled=True)

        with patch(
            "prism.tools.code_sandbox.CodeSandbox",
            return_value=mock_sandbox,
        ):
            action, _out, _, _, _ = _cmd(
                "/sandbox import os; print(os.getcwd())",
                tmp_path,
                state=state,
            )

        assert action == "continue"
        # Verify execute was called with python language
        mock_sandbox.execute.assert_called_once()
        call_args = mock_sandbox.execute.call_args
        assert call_args[1]["language"] == "python"

    def test_sandbox_output_displayed(
        self, tmp_path: Path,
    ) -> None:
        """Execution output is displayed in panel."""
        mock_sandbox = MagicMock()
        mock_sandbox.execute.return_value = (
            _MockSandboxResult(
                stdout="result: 42",
                exit_code=0,
                sandbox_type="subprocess",
            )
        )

        state = _make_state(sandbox_enabled=True)

        with patch(
            "prism.tools.code_sandbox.CodeSandbox",
            return_value=mock_sandbox,
        ):
            action, out, _, _, _ = _cmd(
                "/sandbox print(6*7)",
                tmp_path,
                state=state,
            )

        assert action == "continue"
        assert "result: 42" in out

    def test_sandbox_disabled_shows_message(
        self, tmp_path: Path,
    ) -> None:
        """Running code when disabled shows message."""
        state = _make_state(sandbox_enabled=False)
        action, out, _, _, _ = _cmd(
            "/sandbox print('hi')",
            tmp_path,
            state=state,
        )
        assert action == "continue"
        assert "disabled" in out.lower()


# ===========================================================
# TestTasksEnhancements
# ===========================================================

class TestTasksEnhancements:
    """Tests for enhanced /tasks command."""

    def test_tasks_list_shows_table(
        self, tmp_path: Path,
    ) -> None:
        """``/tasks list`` shows task table."""
        from prism.tools.task_queue import (
            BackgroundTask,
            TaskStatus,
        )

        mock_queue = MagicMock()
        task1 = BackgroundTask(
            id="abc12345",
            name="refactor",
            description="Refactor module",
            status=TaskStatus.RUNNING,
            created_at="2026-01-01T00:00:00+00:00",
            started_at="2026-01-01T00:00:01+00:00",
            progress=0.5,
            progress_message="Halfway",
        )
        task2 = BackgroundTask(
            id="def67890",
            name="test",
            description="Run tests",
            status=TaskStatus.COMPLETED,
            created_at="2026-01-01T00:00:00+00:00",
            started_at="2026-01-01T00:00:01+00:00",
            completed_at="2026-01-01T00:00:10+00:00",
            progress=1.0,
            progress_message="Done",
        )
        mock_queue.list_tasks.return_value = [
            task1, task2,
        ]

        with patch(
            "prism.tools.task_queue.TaskQueue",
            return_value=mock_queue,
        ):
            action, out, _, _, _ = _cmd(
                "/tasks list", tmp_path,
            )

        assert action == "continue"
        assert "Background Tasks" in out
        assert "abc12345" in out
        assert "def67890" in out
        assert "1 running" in out
        assert "1 completed" in out

    def test_tasks_cancel(
        self, tmp_path: Path,
    ) -> None:
        """``/tasks cancel <id>`` cancels a task."""
        mock_queue = MagicMock()
        mock_queue.cancel.return_value = True

        with patch(
            "prism.tools.task_queue.TaskQueue",
            return_value=mock_queue,
        ):
            action, out, _, _, _ = _cmd(
                "/tasks cancel abc12345", tmp_path,
            )

        assert action == "continue"
        assert "cancelled" in out.lower()

    def test_tasks_results_shows_output(
        self, tmp_path: Path,
    ) -> None:
        """``/tasks results <id>`` shows output panel."""
        from prism.tools.task_queue import TaskResult

        mock_queue = MagicMock()
        mock_queue.get_results.return_value = TaskResult(
            output="All tests passed!",
            error=None,
            exit_code=0,
            duration_ms=1234.5,
        )

        with patch(
            "prism.tools.task_queue.TaskQueue",
            return_value=mock_queue,
        ):
            action, out, _, _, _ = _cmd(
                "/tasks results abc12345", tmp_path,
            )

        assert action == "continue"
        assert "All tests passed!" in out
        assert "Exit: 0" in out

    def test_tasks_clear_removes_completed(
        self, tmp_path: Path,
    ) -> None:
        """``/tasks clear`` removes completed tasks."""
        mock_queue = MagicMock()
        mock_queue.cleanup_completed.return_value = 3

        with patch(
            "prism.tools.task_queue.TaskQueue",
            return_value=mock_queue,
        ):
            action, out, _, _, _ = _cmd(
                "/tasks clear", tmp_path,
            )

        assert action == "continue"
        assert "3" in out
        assert "cleared" in out.lower()

    def test_tasks_empty_list(
        self, tmp_path: Path,
    ) -> None:
        """Empty task list shows message."""
        mock_queue = MagicMock()
        mock_queue.list_tasks.return_value = []

        with patch(
            "prism.tools.task_queue.TaskQueue",
            return_value=mock_queue,
        ):
            action, out, _, _, _ = _cmd(
                "/tasks", tmp_path,
            )

        assert action == "continue"
        assert "no background tasks" in out.lower()

    def test_tasks_not_found(
        self, tmp_path: Path,
    ) -> None:
        """Task not found shows error."""
        mock_queue = MagicMock()
        mock_queue.cancel.side_effect = ValueError(
            "Task not found: xyz",
        )

        with patch(
            "prism.tools.task_queue.TaskQueue",
            return_value=mock_queue,
        ):
            action, out, _, _, _ = _cmd(
                "/tasks cancel xyz", tmp_path,
            )

        assert action == "continue"
        assert "Task not found" in out

    def test_tasks_results_no_results_yet(
        self, tmp_path: Path,
    ) -> None:
        """Results for running task shows no-results msg."""
        from prism.tools.task_queue import (
            BackgroundTask,
            TaskStatus,
        )

        mock_queue = MagicMock()
        mock_queue.get_results.return_value = None
        mock_queue.get_task.return_value = BackgroundTask(
            id="abc12345",
            name="running-task",
            description="Still running",
            status=TaskStatus.RUNNING,
            created_at="2026-01-01T00:00:00+00:00",
        )

        with patch(
            "prism.tools.task_queue.TaskQueue",
            return_value=mock_queue,
        ):
            action, out, _, _, _ = _cmd(
                "/tasks results abc12345", tmp_path,
            )

        assert action == "continue"
        assert "no results" in out.lower()

    def test_tasks_results_failed_task(
        self, tmp_path: Path,
    ) -> None:
        """Results for failed task shows error output."""
        from prism.tools.task_queue import TaskResult

        mock_queue = MagicMock()
        mock_queue.get_results.return_value = TaskResult(
            output="",
            error="RuntimeError: crashed",
            exit_code=1,
            duration_ms=500.0,
        )

        with patch(
            "prism.tools.task_queue.TaskQueue",
            return_value=mock_queue,
        ):
            action, out, _, _, _ = _cmd(
                "/tasks results fail123", tmp_path,
            )

        assert action == "continue"
        assert "Exit: 1" in out
        assert "crashed" in out

    def test_tasks_cancel_no_id(
        self, tmp_path: Path,
    ) -> None:
        """Cancel without ID shows usage."""
        mock_queue = MagicMock()

        with patch(
            "prism.tools.task_queue.TaskQueue",
            return_value=mock_queue,
        ):
            action, out, _, _, _ = _cmd(
                "/tasks cancel", tmp_path,
            )

        assert action == "continue"
        assert "usage" in out.lower()

    def test_tasks_results_no_id(
        self, tmp_path: Path,
    ) -> None:
        """Results without ID shows usage."""
        mock_queue = MagicMock()

        with patch(
            "prism.tools.task_queue.TaskQueue",
            return_value=mock_queue,
        ):
            action, out, _, _, _ = _cmd(
                "/tasks results", tmp_path,
            )

        assert action == "continue"
        assert "usage" in out.lower()

    def test_tasks_status_group_counts(
        self, tmp_path: Path,
    ) -> None:
        """Status group counts shown above table."""
        from prism.tools.task_queue import (
            BackgroundTask,
            TaskStatus,
        )

        mock_queue = MagicMock()
        tasks = [
            BackgroundTask(
                id=f"t{i}",
                name=f"task-{i}",
                description=f"Task {i}",
                status=status,
                created_at="2026-01-01T00:00:00+00:00",
                started_at="2026-01-01T00:00:01+00:00",
                completed_at=(
                    "2026-01-01T00:00:10+00:00"
                    if status in (
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                    ) else None
                ),
                progress=(
                    1.0 if status == TaskStatus.COMPLETED
                    else 0.5
                ),
            )
            for i, status in enumerate([
                TaskStatus.RUNNING,
                TaskStatus.RUNNING,
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
            ])
        ]
        mock_queue.list_tasks.return_value = tasks

        with patch(
            "prism.tools.task_queue.TaskQueue",
            return_value=mock_queue,
        ):
            action, out, _, _, _ = _cmd(
                "/tasks", tmp_path,
            )

        assert action == "continue"
        assert "2 running" in out
        assert "1 completed" in out
        assert "1 failed" in out

    def test_tasks_eta_displayed(
        self, tmp_path: Path,
    ) -> None:
        """ETA column displayed for running tasks."""
        from prism.tools.task_queue import (
            BackgroundTask,
            TaskStatus,
        )

        mock_queue = MagicMock()
        mock_queue.list_tasks.return_value = [
            BackgroundTask(
                id="eta1",
                name="long-task",
                description="Long running task",
                status=TaskStatus.RUNNING,
                created_at="2025-01-01T00:00:00+00:00",
                started_at="2025-01-01T00:00:00+00:00",
                progress=0.5,
            ),
        ]

        with patch(
            "prism.tools.task_queue.TaskQueue",
            return_value=mock_queue,
        ):
            action, out, _, _, _ = _cmd(
                "/tasks", tmp_path,
            )

        assert action == "continue"
        # Should have ETA column
        assert "ETA" in out

    def test_tasks_progress_bar_running(
        self, tmp_path: Path,
    ) -> None:
        """Running tasks show progress bar."""
        from prism.tools.task_queue import (
            BackgroundTask,
            TaskStatus,
        )

        mock_queue = MagicMock()
        mock_queue.list_tasks.return_value = [
            BackgroundTask(
                id="bar1",
                name="running",
                description="Running task",
                status=TaskStatus.RUNNING,
                created_at="2026-01-01T00:00:00+00:00",
                started_at="2026-01-01T00:00:01+00:00",
                progress=0.3,
            ),
        ]

        with patch(
            "prism.tools.task_queue.TaskQueue",
            return_value=mock_queue,
        ):
            action, out, _, _, _ = _cmd(
                "/tasks", tmp_path,
            )

        assert action == "continue"
        # Progress should contain ### and --- chars
        assert "#" in out
        assert "30%" in out
