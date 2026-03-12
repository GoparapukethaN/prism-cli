"""Tests for prism.cli.keybindings — keyboard shortcuts for the REPL."""

from __future__ import annotations

from unittest.mock import MagicMock

from prompt_toolkit.key_binding import KeyBindings

from prism.cli.keybindings import create_keybindings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_console() -> MagicMock:
    """Create a mock Rich console."""
    console = MagicMock()
    console.clear = MagicMock()
    return console


def _make_mock_state() -> MagicMock:
    """Create a mock session state."""
    return MagicMock()


def _get_bound_keys(bindings: KeyBindings) -> list[str]:
    """Extract all bound key sequences as strings (e.g. 'c-c', 'c-l')."""
    keys: list[str] = []
    for binding in bindings.bindings:
        # Each binding.keys is a tuple of Keys enum values; use .value
        key_parts = []
        for key in binding.keys:
            val = key.value if hasattr(key, "value") else str(key)
            key_parts.append(val)
        keys.append("-".join(key_parts) if len(key_parts) > 1 else key_parts[0])
    return keys


def _find_handler(bindings: KeyBindings, key_name: str):
    """Find and return the handler for a given key name (e.g. 'c-c')."""
    for binding in bindings.bindings:
        val = binding.keys[0].value if hasattr(binding.keys[0], "value") else str(binding.keys[0])
        if val == key_name:
            return binding.handler
    return None


def _make_event(text: str = "", cursor_pos: int | None = None) -> MagicMock:
    """Create a mock prompt_toolkit event with a buffer."""
    event = MagicMock()
    buf = MagicMock()
    buf.text = text
    buf.cursor_position = cursor_pos if cursor_pos is not None else len(text)

    # Set up document mock for text_before_cursor / text_after_cursor
    doc = MagicMock()
    pos = cursor_pos if cursor_pos is not None else len(text)
    doc.text_before_cursor = text[:pos]
    doc.text_after_cursor = text[pos:]
    buf.document = doc

    event.app.current_buffer = buf
    event.app.renderer = MagicMock()
    return event


# ---------------------------------------------------------------------------
# create_keybindings — basic
# ---------------------------------------------------------------------------


class TestCreateKeybindings:
    """Tests for the create_keybindings factory function."""

    def test_returns_key_bindings_instance(self) -> None:
        bindings = create_keybindings(
            state=_make_mock_state(), console=_make_mock_console(),
        )
        assert isinstance(bindings, KeyBindings)

    def test_has_bindings(self) -> None:
        bindings = create_keybindings(
            state=_make_mock_state(), console=_make_mock_console(),
        )
        assert len(bindings.bindings) > 0

    def test_binds_ctrl_c(self) -> None:
        bindings = create_keybindings(
            state=_make_mock_state(), console=_make_mock_console(),
        )
        keys = _get_bound_keys(bindings)
        assert any("c-c" in k for k in keys)

    def test_binds_ctrl_l(self) -> None:
        bindings = create_keybindings(
            state=_make_mock_state(), console=_make_mock_console(),
        )
        keys = _get_bound_keys(bindings)
        assert any("c-l" in k for k in keys)

    def test_binds_ctrl_u(self) -> None:
        bindings = create_keybindings(
            state=_make_mock_state(), console=_make_mock_console(),
        )
        keys = _get_bound_keys(bindings)
        assert any("c-u" in k for k in keys)

    def test_binds_ctrl_k(self) -> None:
        bindings = create_keybindings(
            state=_make_mock_state(), console=_make_mock_console(),
        )
        keys = _get_bound_keys(bindings)
        assert any("c-k" in k for k in keys)

    def test_binds_ctrl_a(self) -> None:
        bindings = create_keybindings(
            state=_make_mock_state(), console=_make_mock_console(),
        )
        keys = _get_bound_keys(bindings)
        assert any("c-a" in k for k in keys)

    def test_binds_ctrl_e(self) -> None:
        bindings = create_keybindings(
            state=_make_mock_state(), console=_make_mock_console(),
        )
        keys = _get_bound_keys(bindings)
        assert any("c-e" in k for k in keys)

    def test_expected_binding_count(self) -> None:
        """We expect exactly 6 bindings: c-c, c-l, c-u, c-k, c-a, c-e."""
        bindings = create_keybindings(
            state=_make_mock_state(), console=_make_mock_console(),
        )
        assert len(bindings.bindings) == 6


# ---------------------------------------------------------------------------
# Handler behavior
# ---------------------------------------------------------------------------


class TestCtrlCHandler:
    """Tests for the Ctrl+C handler."""

    def test_ctrl_c_resets_buffer(self) -> None:
        bindings = create_keybindings(
            state=_make_mock_state(), console=_make_mock_console(),
        )
        event = _make_event(text="some input")
        handler = _find_handler(bindings, "c-c")
        assert handler is not None
        handler(event)
        event.app.current_buffer.reset.assert_called_once()


class TestCtrlLHandler:
    """Tests for the Ctrl+L handler."""

    def test_ctrl_l_clears_console(self) -> None:
        console = _make_mock_console()
        bindings = create_keybindings(
            state=_make_mock_state(), console=console,
        )
        event = _make_event()
        handler = _find_handler(bindings, "c-l")
        assert handler is not None
        handler(event)
        console.clear.assert_called_once()

    def test_ctrl_l_clears_renderer(self) -> None:
        console = _make_mock_console()
        bindings = create_keybindings(
            state=_make_mock_state(), console=console,
        )
        event = _make_event()
        handler = _find_handler(bindings, "c-l")
        assert handler is not None
        handler(event)
        event.app.renderer.clear.assert_called_once()


class TestCtrlUHandler:
    """Tests for the Ctrl+U handler."""

    def test_ctrl_u_deletes_before_cursor(self) -> None:
        bindings = create_keybindings(
            state=_make_mock_state(), console=_make_mock_console(),
        )
        event = _make_event(text="hello world", cursor_pos=5)
        handler = _find_handler(bindings, "c-u")
        assert handler is not None
        handler(event)
        event.app.current_buffer.delete_before_cursor.assert_called_once_with(
            count=5,  # "hello" is 5 chars before cursor
        )


class TestCtrlKHandler:
    """Tests for the Ctrl+K handler."""

    def test_ctrl_k_deletes_after_cursor(self) -> None:
        bindings = create_keybindings(
            state=_make_mock_state(), console=_make_mock_console(),
        )
        event = _make_event(text="hello world", cursor_pos=5)
        handler = _find_handler(bindings, "c-k")
        assert handler is not None
        handler(event)
        event.app.current_buffer.delete.assert_called_once_with(
            count=6,  # " world" is 6 chars after cursor
        )

    def test_ctrl_k_at_end_does_nothing(self) -> None:
        bindings = create_keybindings(
            state=_make_mock_state(), console=_make_mock_console(),
        )
        event = _make_event(text="hello", cursor_pos=5)
        handler = _find_handler(bindings, "c-k")
        assert handler is not None
        handler(event)
        event.app.current_buffer.delete.assert_not_called()


class TestCtrlAHandler:
    """Tests for the Ctrl+A handler."""

    def test_ctrl_a_moves_to_start(self) -> None:
        bindings = create_keybindings(
            state=_make_mock_state(), console=_make_mock_console(),
        )
        event = _make_event(text="hello world", cursor_pos=8)
        handler = _find_handler(bindings, "c-a")
        assert handler is not None
        handler(event)
        assert event.app.current_buffer.cursor_position == 0


class TestCtrlEHandler:
    """Tests for the Ctrl+E handler."""

    def test_ctrl_e_moves_to_end(self) -> None:
        bindings = create_keybindings(
            state=_make_mock_state(), console=_make_mock_console(),
        )
        event = _make_event(text="hello world", cursor_pos=3)
        handler = _find_handler(bindings, "c-e")
        assert handler is not None
        handler(event)
        assert event.app.current_buffer.cursor_position == len("hello world")
