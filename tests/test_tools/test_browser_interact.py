"""Tests for BrowserInteractTool -- all Playwright calls are mocked."""

from __future__ import annotations

import base64
import threading
from unittest.mock import MagicMock, patch

import pytest

from prism.tools.base import PermissionLevel
from prism.tools.browser_interact import (
    _INACTIVITY_TIMEOUT,
    _VALID_ACTIONS,
    BrowserInteractTool,
)

# ------------------------------------------------------------------
# Helper: build a fully mocked Playwright stack
# ------------------------------------------------------------------


def _build_mock_playwright(
    *,
    page_url: str = "https://example.com",
    page_title: str = "Example Page",
    inner_text: str = "Hello World",
    screenshot_bytes: bytes | None = None,
    evaluate_result: object = None,
    scroll_y: int = 600,
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Build a complete mock Playwright stack.

    Returns:
        (sync_playwright_func, mock_page, mock_browser, mock_pw_instance)
    """
    if screenshot_bytes is None:
        screenshot_bytes = b"\x89PNG\r\n\x1a\nfake_image_data"

    mock_element = MagicMock()
    mock_element.inner_text.return_value = inner_text
    mock_element.screenshot.return_value = screenshot_bytes

    mock_page = MagicMock()
    mock_page.url = page_url
    mock_page.title.return_value = page_title
    mock_page.screenshot.return_value = screenshot_bytes
    mock_page.query_selector.return_value = mock_element
    mock_page.goto = MagicMock()
    mock_page.click = MagicMock()
    mock_page.fill = MagicMock()
    mock_page.type = MagicMock()
    mock_page.select_option = MagicMock()
    mock_page.go_back = MagicMock()
    mock_page.wait_for_selector = MagicMock()
    mock_page.close = MagicMock()
    mock_page.set_default_timeout = MagicMock()

    # evaluate returns different values based on argument
    def _evaluate_side_effect(script: str) -> object:
        if "innerText" in script:
            return inner_text
        if "scrollBy" in script:
            return None
        if "scrollY" in script:
            return scroll_y
        return evaluate_result

    mock_page.evaluate = MagicMock(
        side_effect=_evaluate_side_effect
    )

    mock_browser = MagicMock()
    mock_browser.new_page.return_value = mock_page
    mock_browser.close = MagicMock()

    mock_chromium = MagicMock()
    mock_chromium.launch.return_value = mock_browser

    mock_pw_instance = MagicMock()
    mock_pw_instance.chromium = mock_chromium
    mock_pw_instance.stop = MagicMock()

    # sync_playwright() returns an object whose .start() gives pw_instance
    sync_playwright_func = MagicMock()
    sync_playwright_func.return_value.start.return_value = (
        mock_pw_instance
    )

    return (
        sync_playwright_func,
        mock_page,
        mock_browser,
        mock_pw_instance,
    )


@pytest.fixture
def tool() -> BrowserInteractTool:
    """Create a fresh BrowserInteractTool instance."""
    return BrowserInteractTool()


@pytest.fixture
def tool_with_page() -> (
    tuple[BrowserInteractTool, MagicMock, MagicMock, MagicMock]
):
    """Create a tool with an already-initialized mock page.

    Returns:
        (tool, mock_page, mock_browser, mock_pw_instance)
    """
    t = BrowserInteractTool()
    (
        _sync_pw,
        mock_page,
        mock_browser,
        mock_pw_instance,
    ) = _build_mock_playwright()

    t._playwright_ctx = mock_pw_instance
    t._browser = mock_browser
    t._page = mock_page
    t._last_activity = 1000.0
    return t, mock_page, mock_browser, mock_pw_instance


# =====================================================================
# Tool properties
# =====================================================================


class TestToolProperties:
    def test_tool_name(self, tool: BrowserInteractTool) -> None:
        assert tool.name == "browser_interact"

    def test_tool_permission(
        self, tool: BrowserInteractTool
    ) -> None:
        assert tool.permission_level == PermissionLevel.CONFIRM

    def test_tool_description(
        self, tool: BrowserInteractTool
    ) -> None:
        desc = tool.description.lower()
        assert "browser" in desc
        assert "interact" in desc or "interact" in tool.name

    def test_tool_parameters_schema(
        self, tool: BrowserInteractTool
    ) -> None:
        schema = tool.parameters_schema
        assert "action" in schema["properties"]
        assert "action" in schema["required"]
        assert "url" in schema["properties"]
        assert "selector" in schema["properties"]
        assert "text" in schema["properties"]
        assert "value" in schema["properties"]
        assert "script" in schema["properties"]
        assert "direction" in schema["properties"]
        assert "timeout" in schema["properties"]

    def test_timeout_default(
        self, tool: BrowserInteractTool
    ) -> None:
        schema = tool.parameters_schema
        assert schema["properties"]["timeout"]["default"] == 30


# =====================================================================
# URL safety
# =====================================================================


class TestUrlSafety:
    def test_safe_https_url(
        self, tool: BrowserInteractTool
    ) -> None:
        assert tool._is_safe_url("https://example.com") is True

    def test_safe_http_url(
        self, tool: BrowserInteractTool
    ) -> None:
        assert tool._is_safe_url("http://example.com") is True

    def test_unsafe_file_url(
        self, tool: BrowserInteractTool
    ) -> None:
        assert (
            tool._is_safe_url("file:///etc/passwd") is False
        )

    def test_unsafe_localhost(
        self, tool: BrowserInteractTool
    ) -> None:
        assert (
            tool._is_safe_url("http://localhost:8080") is False
        )

    def test_safe_localhost_ollama(
        self, tool: BrowserInteractTool
    ) -> None:
        assert (
            tool._is_safe_url("http://localhost:11434") is True
        )

    def test_unsafe_private_ip(
        self, tool: BrowserInteractTool
    ) -> None:
        assert (
            tool._is_safe_url("http://192.168.1.1") is False
        )

    def test_unsafe_loopback(
        self, tool: BrowserInteractTool
    ) -> None:
        assert (
            tool._is_safe_url("http://127.0.0.1") is False
        )

    def test_unsafe_10_prefix(
        self, tool: BrowserInteractTool
    ) -> None:
        assert tool._is_safe_url("http://10.0.0.1") is False

    def test_unsafe_ftp_scheme(
        self, tool: BrowserInteractTool
    ) -> None:
        assert (
            tool._is_safe_url("ftp://example.com/file") is False
        )

    def test_unsafe_javascript_scheme(
        self, tool: BrowserInteractTool
    ) -> None:
        assert (
            tool._is_safe_url("javascript:alert(1)") is False
        )

    def test_unsafe_data_scheme(
        self, tool: BrowserInteractTool
    ) -> None:
        assert (
            tool._is_safe_url("data:text/html,<h1>x</h1>")
            is False
        )

    def test_unsafe_ipv6_loopback(
        self, tool: BrowserInteractTool
    ) -> None:
        assert tool._is_safe_url("http://[::1]:80") is False

    def test_empty_url(
        self, tool: BrowserInteractTool
    ) -> None:
        assert tool._is_safe_url("") is False

    def test_navigate_rejects_unsafe_url(
        self, tool: BrowserInteractTool
    ) -> None:
        result = tool.execute({
            "action": "navigate",
            "url": "file:///etc/passwd",
        })
        assert result.success is False
        assert "rejected" in (result.error or "").lower()

    def test_navigate_rejects_private_ip(
        self, tool: BrowserInteractTool
    ) -> None:
        result = tool.execute({
            "action": "navigate",
            "url": "http://192.168.1.1/admin",
        })
        assert result.success is False
        assert "rejected" in (result.error or "").lower()


# =====================================================================
# Validation
# =====================================================================


class TestValidation:
    def test_missing_action_raises(
        self, tool: BrowserInteractTool
    ) -> None:
        with pytest.raises(ValueError, match="Missing required"):
            tool.execute({})

    def test_unknown_action_returns_error(
        self, tool: BrowserInteractTool
    ) -> None:
        result = tool.execute({"action": "destroy"})
        assert result.success is False
        assert "Unknown action" in (result.error or "")

    def test_unknown_argument_raises(
        self, tool: BrowserInteractTool
    ) -> None:
        with pytest.raises(ValueError, match="Unknown arguments"):
            tool.execute({
                "action": "navigate",
                "url": "https://example.com",
                "bogus": True,
            })


# =====================================================================
# Playwright not installed
# =====================================================================


class TestPlaywrightNotInstalled:
    def test_navigate_without_playwright(
        self, tool: BrowserInteractTool
    ) -> None:
        with patch(
            "prism.tools.browser_interact._get_sync_playwright",
            side_effect=ImportError(
                "No module named 'playwright'"
            ),
        ):
            result = tool.execute({
                "action": "navigate",
                "url": "https://example.com",
            })
            assert result.success is False
            assert "playwright" in (result.error or "").lower()


# =====================================================================
# Navigate action
# =====================================================================


class TestNavigateAction:
    def test_navigate_success(
        self, tool: BrowserInteractTool
    ) -> None:
        sync_pw, mock_page, _, _ = _build_mock_playwright(
            page_url="https://example.com",
            page_title="Example Domain",
            inner_text="Example Domain\nMore info...",
        )

        with patch(
            "prism.tools.browser_interact._get_sync_playwright",
            return_value=sync_pw,
        ):
            result = tool.execute({
                "action": "navigate",
                "url": "https://example.com",
            })

        assert result.success is True
        assert "example.com" in result.output.lower()
        assert result.metadata is not None
        mock_page.goto.assert_called_once()

    def test_navigate_missing_url(
        self, tool: BrowserInteractTool
    ) -> None:
        result = tool.execute({"action": "navigate"})
        assert result.success is False
        assert "'url' parameter is required" in (
            result.error or ""
        )

    def test_navigate_updates_activity(
        self, tool: BrowserInteractTool
    ) -> None:
        sync_pw, _, _, _ = _build_mock_playwright()

        with patch(
            "prism.tools.browser_interact._get_sync_playwright",
            return_value=sync_pw,
        ):
            tool.execute({
                "action": "navigate",
                "url": "https://example.com",
            })

        assert tool._last_activity > 0


# =====================================================================
# Click action
# =====================================================================


class TestClickAction:
    def test_click_success(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        result = tool.execute({
            "action": "click",
            "selector": "#submit-btn",
        })
        assert result.success is True
        mock_page.click.assert_called_once_with(
            "#submit-btn", timeout=30000
        )
        assert "submit-btn" in result.output

    def test_click_missing_selector(
        self, tool: BrowserInteractTool
    ) -> None:
        result = tool.execute({"action": "click"})
        assert result.success is False
        assert "'selector' parameter is required" in (
            result.error or ""
        )

    def test_click_element_not_found(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        mock_page.click.side_effect = Exception(
            "Element not found"
        )
        result = tool.execute({
            "action": "click",
            "selector": "#nonexistent",
        })
        assert result.success is False
        assert "failed" in (result.error or "").lower()


# =====================================================================
# Fill action
# =====================================================================


class TestFillAction:
    def test_fill_success(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        result = tool.execute({
            "action": "fill",
            "selector": "#email",
            "text": "user@example.com",
        })
        assert result.success is True
        mock_page.fill.assert_called_once_with(
            "#email", "user@example.com", timeout=30000
        )
        assert "16 chars" in result.output

    def test_fill_missing_selector(
        self, tool: BrowserInteractTool
    ) -> None:
        result = tool.execute({
            "action": "fill",
            "text": "hello",
        })
        assert result.success is False
        assert "'selector'" in (result.error or "")

    def test_fill_missing_text(
        self, tool: BrowserInteractTool
    ) -> None:
        result = tool.execute({
            "action": "fill",
            "selector": "#email",
        })
        assert result.success is False
        assert "'text'" in (result.error or "")

    def test_fill_empty_text(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        result = tool.execute({
            "action": "fill",
            "selector": "#email",
            "text": "",
        })
        assert result.success is True
        mock_page.fill.assert_called_once_with(
            "#email", "", timeout=30000
        )


# =====================================================================
# Type action
# =====================================================================


class TestTypeAction:
    def test_type_success(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        result = tool.execute({
            "action": "type",
            "selector": "#search",
            "text": "python",
        })
        assert result.success is True
        mock_page.type.assert_called_once_with(
            "#search", "python", timeout=30000
        )

    def test_type_missing_selector(
        self, tool: BrowserInteractTool
    ) -> None:
        result = tool.execute({
            "action": "type",
            "text": "hello",
        })
        assert result.success is False
        assert "'selector'" in (result.error or "")

    def test_type_missing_text(
        self, tool: BrowserInteractTool
    ) -> None:
        result = tool.execute({
            "action": "type",
            "selector": "#search",
        })
        assert result.success is False
        assert "'text'" in (result.error or "")


# =====================================================================
# Select action
# =====================================================================


class TestSelectAction:
    def test_select_success(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        result = tool.execute({
            "action": "select",
            "selector": "#country",
            "value": "US",
        })
        assert result.success is True
        mock_page.select_option.assert_called_once_with(
            "#country", "US", timeout=30000
        )
        assert "US" in result.output

    def test_select_missing_selector(
        self, tool: BrowserInteractTool
    ) -> None:
        result = tool.execute({
            "action": "select",
            "value": "US",
        })
        assert result.success is False
        assert "'selector'" in (result.error or "")

    def test_select_missing_value(
        self, tool: BrowserInteractTool
    ) -> None:
        result = tool.execute({
            "action": "select",
            "selector": "#country",
        })
        assert result.success is False
        assert "'value'" in (result.error or "")


# =====================================================================
# Screenshot action
# =====================================================================


class TestScreenshotAction:
    def test_screenshot_returns_base64(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        fake_png = b"\x89PNG\r\n\x1a\nscreenshot_data"
        mock_page.screenshot.return_value = fake_png

        result = tool.execute({"action": "screenshot"})

        assert result.success is True
        decoded = base64.b64decode(result.output)
        assert decoded == fake_png
        assert result.metadata is not None
        assert result.metadata["format"] == "png"
        assert result.metadata["size_bytes"] == len(fake_png)


# =====================================================================
# Scroll action
# =====================================================================


class TestScrollAction:
    def test_scroll_down(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        result = tool.execute({
            "action": "scroll",
            "direction": "down",
        })
        assert result.success is True
        assert "down" in result.output.lower()
        # scrollBy call + scrollY call
        assert mock_page.evaluate.call_count >= 2

    def test_scroll_up(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, _mock_page, _, _ = tool_with_page
        result = tool.execute({
            "action": "scroll",
            "direction": "up",
        })
        assert result.success is True
        assert "up" in result.output.lower()

    def test_scroll_default_direction(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, _, _, _ = tool_with_page
        result = tool.execute({"action": "scroll"})
        assert result.success is True
        assert "down" in result.output.lower()

    def test_scroll_invalid_direction(
        self, tool: BrowserInteractTool
    ) -> None:
        result = tool.execute({
            "action": "scroll",
            "direction": "left",
        })
        assert result.success is False
        assert "'direction'" in (result.error or "")


# =====================================================================
# Get text action
# =====================================================================


class TestGetTextAction:
    def test_get_text_full_page(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, _, _, _ = tool_with_page
        result = tool.execute({"action": "get_text"})
        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["selector"] == "body"

    def test_get_text_by_selector(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        mock_element = MagicMock()
        mock_element.inner_text.return_value = "Button text"
        mock_page.query_selector.return_value = mock_element

        result = tool.execute({
            "action": "get_text",
            "selector": "#my-button",
        })
        assert result.success is True
        assert "Button text" in result.output

    def test_get_text_element_not_found(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        mock_page.query_selector.return_value = None

        result = tool.execute({
            "action": "get_text",
            "selector": "#nonexistent",
        })
        assert result.success is False
        assert "not found" in (result.error or "").lower()


# =====================================================================
# Wait action
# =====================================================================


class TestWaitAction:
    def test_wait_success(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        result = tool.execute({
            "action": "wait",
            "selector": ".loaded",
        })
        assert result.success is True
        mock_page.wait_for_selector.assert_called_once_with(
            ".loaded", timeout=30000
        )

    def test_wait_missing_selector(
        self, tool: BrowserInteractTool
    ) -> None:
        result = tool.execute({"action": "wait"})
        assert result.success is False
        assert "'selector'" in (result.error or "")

    def test_wait_timeout(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        mock_page.wait_for_selector.side_effect = Exception(
            "Timeout 10000ms exceeded"
        )
        result = tool.execute({
            "action": "wait",
            "selector": ".never-appears",
            "timeout": 10,
        })
        assert result.success is False
        assert "failed" in (result.error or "").lower()


# =====================================================================
# Evaluate action
# =====================================================================


class TestEvaluateAction:
    def test_evaluate_success(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        mock_page.evaluate = MagicMock(return_value=42)

        result = tool.execute({
            "action": "evaluate",
            "script": "2 + 2",
        })
        assert result.success is True
        assert "42" in result.output

    def test_evaluate_returns_none(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        mock_page.evaluate = MagicMock(return_value=None)

        result = tool.execute({
            "action": "evaluate",
            "script": "void(0)",
        })
        assert result.success is True
        assert "undefined" in result.output

    def test_evaluate_missing_script(
        self, tool: BrowserInteractTool
    ) -> None:
        result = tool.execute({"action": "evaluate"})
        assert result.success is False
        assert "'script'" in (result.error or "")

    def test_evaluate_js_error(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        mock_page.evaluate = MagicMock(
            side_effect=Exception("ReferenceError: foo is not defined")
        )
        result = tool.execute({
            "action": "evaluate",
            "script": "foo.bar()",
        })
        assert result.success is False
        assert "failed" in (result.error or "").lower()


# =====================================================================
# Back action
# =====================================================================


class TestBackAction:
    def test_back_success(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        result = tool.execute({"action": "back"})
        assert result.success is True
        mock_page.go_back.assert_called_once()
        assert "back" in result.output.lower()


# =====================================================================
# Close action
# =====================================================================


class TestCloseAction:
    def test_close_success(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, mock_browser, mock_pw = tool_with_page
        result = tool.execute({"action": "close"})
        assert result.success is True
        assert "closed" in result.output.lower()
        assert tool._page is None
        assert tool._browser is None
        assert tool._playwright_ctx is None
        mock_page.close.assert_called_once()
        mock_browser.close.assert_called_once()
        mock_pw.stop.assert_called_once()

    def test_close_when_no_session(
        self, tool: BrowserInteractTool
    ) -> None:
        result = tool.execute({"action": "close"})
        assert result.success is True
        assert "closed" in result.output.lower()


# =====================================================================
# Browser lifecycle
# =====================================================================


class TestBrowserLifecycle:
    def test_lazy_launch_on_first_action(
        self, tool: BrowserInteractTool
    ) -> None:
        sync_pw, _mock_page, _, _ = _build_mock_playwright()

        with patch(
            "prism.tools.browser_interact._get_sync_playwright",
            return_value=sync_pw,
        ):
            result = tool.execute({
                "action": "navigate",
                "url": "https://example.com",
            })

        assert result.success is True
        assert tool._page is not None

    def test_reuses_existing_session(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, _mock_page, _, _ = tool_with_page
        original_page = tool._page

        result = tool.execute({
            "action": "click",
            "selector": "#btn",
        })
        assert result.success is True
        assert tool._page is original_page

    def test_browser_launch_failure(
        self, tool: BrowserInteractTool
    ) -> None:
        sync_pw = MagicMock()
        sync_pw.return_value.start.side_effect = Exception(
            "No browser installed"
        )

        with patch(
            "prism.tools.browser_interact._get_sync_playwright",
            return_value=sync_pw,
        ):
            result = tool.execute({
                "action": "navigate",
                "url": "https://example.com",
            })

        assert result.success is False
        assert "failed to launch" in (result.error or "").lower()

    def test_cleanup_closes_all_resources(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, mock_browser, mock_pw = tool_with_page
        tool._cleanup_browser()

        mock_page.close.assert_called_once()
        mock_browser.close.assert_called_once()
        mock_pw.stop.assert_called_once()
        assert tool._page is None
        assert tool._browser is None
        assert tool._playwright_ctx is None

    def test_cleanup_handles_exceptions(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, mock_browser, mock_pw = tool_with_page
        mock_page.close.side_effect = Exception("close error")
        mock_browser.close.side_effect = Exception("close error")
        mock_pw.stop.side_effect = Exception("stop error")

        # Should not raise
        tool._cleanup_browser()
        assert tool._page is None
        assert tool._browser is None
        assert tool._playwright_ctx is None


# =====================================================================
# Auto-close (inactivity timeout)
# =====================================================================


class TestAutoClose:
    def test_auto_close_after_inactivity(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _mock_browser, _mock_pw = tool_with_page
        # Simulate that last activity was long ago
        tool._last_activity = 0.0

        with patch(
            "prism.tools.browser_interact._time.monotonic",
            return_value=_INACTIVITY_TIMEOUT + 1,
        ):
            tool._auto_close()

        assert tool._page is None
        mock_page.close.assert_called_once()

    def test_auto_close_reschedules_if_active(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        # Very recent activity
        tool._last_activity = _INACTIVITY_TIMEOUT - 10

        with patch(
            "prism.tools.browser_interact._time.monotonic",
            return_value=_INACTIVITY_TIMEOUT - 5,
        ), patch.object(
            threading.Timer, "start"
        ):
            tool._auto_close()

        # Browser should still be alive
        assert tool._page is mock_page
        # A new timer should have been scheduled
        assert tool._cleanup_timer is not None

    def test_schedule_cleanup_cancels_existing(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, _, _, _ = tool_with_page
        old_timer = MagicMock()
        tool._cleanup_timer = old_timer

        tool._schedule_cleanup()

        old_timer.cancel.assert_called_once()
        assert tool._cleanup_timer is not old_timer

    def test_inactivity_timeout_value(self) -> None:
        assert _INACTIVITY_TIMEOUT == 300  # 5 minutes


# =====================================================================
# Custom timeout
# =====================================================================


class TestCustomTimeout:
    def test_custom_timeout_passed_to_navigate(
        self, tool: BrowserInteractTool
    ) -> None:
        sync_pw, mock_page, _, _ = _build_mock_playwright()

        with patch(
            "prism.tools.browser_interact._get_sync_playwright",
            return_value=sync_pw,
        ):
            tool.execute({
                "action": "navigate",
                "url": "https://example.com",
                "timeout": 60,
            })

        mock_page.goto.assert_called_once_with(
            "https://example.com", timeout=60000
        )

    def test_custom_timeout_passed_to_click(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        tool.execute({
            "action": "click",
            "selector": "#btn",
            "timeout": 10,
        })
        mock_page.click.assert_called_once_with(
            "#btn", timeout=10000
        )


# =====================================================================
# Error handling
# =====================================================================


class TestErrorHandling:
    def test_action_exception_caught(
        self,
        tool_with_page: tuple[
            BrowserInteractTool,
            MagicMock,
            MagicMock,
            MagicMock,
        ],
    ) -> None:
        tool, mock_page, _, _ = tool_with_page
        mock_page.goto.side_effect = Exception(
            "net::ERR_CONNECTION_REFUSED"
        )

        result = tool.execute({
            "action": "navigate",
            "url": "https://example.com",
        })
        assert result.success is False
        assert "failed" in (result.error or "").lower()

    def test_get_page_info_handles_errors(
        self, tool: BrowserInteractTool
    ) -> None:
        """_get_page_info should not raise even if page is broken."""
        tool._page = MagicMock()
        tool._page.url = property(
            lambda self: (_ for _ in ()).throw(Exception("error"))
        )
        # Should not raise
        info = tool._get_page_info()
        assert isinstance(info, dict)

    def test_get_visible_text_returns_empty_on_error(
        self, tool: BrowserInteractTool
    ) -> None:
        """_get_visible_text should return empty string on error."""
        tool._page = MagicMock()
        tool._page.evaluate.side_effect = Exception("eval error")
        result = tool._get_visible_text()
        assert result == ""

    def test_get_visible_text_no_page(
        self, tool: BrowserInteractTool
    ) -> None:
        assert tool._get_visible_text() == ""


# =====================================================================
# Thread safety
# =====================================================================


class TestThreadSafety:
    def test_has_threading_lock(
        self, tool: BrowserInteractTool
    ) -> None:
        assert isinstance(tool._lock, type(threading.Lock()))

    def test_concurrent_ensure_browser_uses_lock(
        self, tool: BrowserInteractTool
    ) -> None:
        """Verify _ensure_browser acquires the lock."""
        sync_pw, _, _, _ = _build_mock_playwright()

        acquired = []

        class TrackingLock:
            """A lock wrapper that records acquire/release calls."""

            def __init__(self) -> None:
                self._real_lock = threading.Lock()

            def acquire(self, *a: object, **kw: object) -> bool:
                acquired.append(True)
                return self._real_lock.acquire(*a, **kw)

            def release(self) -> None:
                self._real_lock.release()

            def __enter__(self) -> TrackingLock:
                self.acquire()
                return self

            def __exit__(self, *a: object) -> None:
                self.release()

        tool._lock = TrackingLock()  # type: ignore[assignment]

        with patch(
            "prism.tools.browser_interact._get_sync_playwright",
            return_value=sync_pw,
        ):
            tool._ensure_browser()

        assert len(acquired) > 0


# =====================================================================
# Valid actions constant
# =====================================================================


class TestValidActions:
    def test_all_actions_have_handlers(
        self, tool: BrowserInteractTool
    ) -> None:
        for action in _VALID_ACTIONS:
            handler = getattr(tool, f"_action_{action}", None)
            assert handler is not None, (
                f"Missing handler for action: {action}"
            )

    def test_valid_actions_count(self) -> None:
        assert len(_VALID_ACTIONS) == 12
