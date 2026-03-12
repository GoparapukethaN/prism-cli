"""Interactive browser automation tool -- Playwright-based web interaction.

Provides the LLM with full browser interaction capabilities: navigation,
clicking, form filling, typing, dropdown selection, scrolling, screenshots,
JavaScript evaluation, and text extraction.

Maintains a persistent browser session across calls so the page state is
preserved between actions (login sessions, multi-step workflows, etc.).
Auto-closes the browser after 5 minutes of inactivity to free resources.

Security:
  Inherits the same URL safety checks as :class:`BrowseWebTool`:
  ``file://`` URLs, localhost (except Ollama port 11434), and
  private/internal IPs are rejected by default.
"""

from __future__ import annotations

import base64
import contextlib
import ipaddress
import threading
import time as _time
from typing import Any
from urllib.parse import urlparse

import structlog

from prism.tools.base import PermissionLevel, Tool, ToolResult

logger = structlog.get_logger(__name__)

# Ports that are allowed on localhost.
_LOCALHOST_ALLOWED_PORTS: frozenset[int] = frozenset({11434})  # Ollama

# Private IP ranges.
_PRIVATE_PREFIXES = (
    "10.", "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
    "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
    "172.30.", "172.31.", "192.168.", "169.254.",
)

# Inactivity timeout before browser is auto-closed (seconds).
_INACTIVITY_TIMEOUT = 300  # 5 minutes

# Valid actions for the tool.
_VALID_ACTIONS = frozenset({
    "navigate", "click", "fill", "type", "select",
    "screenshot", "scroll", "get_text", "wait",
    "evaluate", "back", "close",
})


def _get_sync_playwright() -> Any:
    """Import and return ``sync_playwright`` from Playwright.

    Raises:
        ImportError: If Playwright is not installed.
    """
    from playwright.sync_api import sync_playwright

    return sync_playwright


class BrowserInteractTool(Tool):
    """Interactive browser automation via Playwright.

    Keeps a persistent browser session (page stays open between calls)
    so the LLM can perform multi-step interactions: navigate to a page,
    fill a form, click submit, read the result.

    The browser is lazy-loaded on first use and auto-closed after
    5 minutes of inactivity.  Thread-safe lifecycle management
    ensures safe concurrent access.
    """

    def __init__(self) -> None:
        """Initialize the tool with no active browser session."""
        self._lock = threading.Lock()
        self._playwright_ctx: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._last_activity: float = 0.0
        self._cleanup_timer: threading.Timer | None = None

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "browser_interact"

    @property
    def description(self) -> str:
        return (
            "Interact with web pages using a persistent browser session. "
            "Supports navigation, clicking, form filling, typing, "
            "dropdown selection, scrolling, screenshots, JavaScript "
            "evaluation, and text extraction."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "The browser action to perform. One of: "
                        "navigate, click, fill, type, select, "
                        "screenshot, scroll, get_text, wait, "
                        "evaluate, back, close."
                    ),
                },
                "url": {
                    "type": "string",
                    "description": (
                        "URL to navigate to (for 'navigate' action)."
                    ),
                },
                "selector": {
                    "type": "string",
                    "description": (
                        "CSS selector for the target element "
                        "(for click, fill, type, select, "
                        "get_text, wait)."
                    ),
                },
                "text": {
                    "type": "string",
                    "description": (
                        "Text to enter (for 'fill' and 'type' "
                        "actions)."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": (
                        "Option value to select "
                        "(for 'select' action)."
                    ),
                },
                "script": {
                    "type": "string",
                    "description": (
                        "JavaScript code to evaluate "
                        "(for 'evaluate' action)."
                    ),
                },
                "direction": {
                    "type": "string",
                    "description": (
                        "Scroll direction: 'up' or 'down' "
                        "(for 'scroll' action)."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Timeout in seconds for the action "
                        "(default 30)."
                    ),
                    "default": 30,
                },
            },
            "required": ["action"],
        }

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.CONFIRM

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute a browser interaction action.

        Args:
            arguments: Must contain ``action``; other parameters depend
                on the action type.

        Returns:
            A :class:`ToolResult` describing the outcome.
        """
        validated = self.validate_arguments(arguments)

        action: str = validated["action"]
        timeout: int = validated.get("timeout", 30)

        if action not in _VALID_ACTIONS:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Unknown action: '{action}'. "
                    f"Valid actions: {sorted(_VALID_ACTIONS)}"
                ),
            )

        # Dispatch to action handler
        handler = getattr(self, f"_action_{action}", None)
        if handler is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Action handler not implemented: {action}",
            )

        try:
            return handler(validated, timeout)
        except Exception as exc:
            logger.error(
                "browser_interact_error",
                action=action,
                error=str(exc),
            )
            return ToolResult(
                success=False,
                output="",
                error=f"Browser action '{action}' failed: {exc}",
            )

    # ------------------------------------------------------------------
    # Browser lifecycle
    # ------------------------------------------------------------------

    def _ensure_browser(self, timeout: int = 30) -> ToolResult | None:
        """Ensure a browser session is active, launching if needed.

        Args:
            timeout: Navigation timeout in seconds (used for launch).

        Returns:
            A ToolResult with an error if browser launch fails,
            or None on success.
        """
        with self._lock:
            if self._page is not None:
                self._touch_activity()
                return None

            try:
                sync_playwright = _get_sync_playwright()
            except ImportError:
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        "Playwright is not installed. "
                        "Install with: pip install playwright "
                        "&& playwright install chromium"
                    ),
                )

            try:
                self._playwright_ctx = sync_playwright().start()
                self._browser = self._playwright_ctx.chromium.launch(
                    headless=True,
                )
                self._page = self._browser.new_page()
                self._page.set_default_timeout(timeout * 1000)
                self._touch_activity()
                self._schedule_cleanup()
                logger.debug("browser_session_started")
                return None
            except Exception as exc:
                self._cleanup_browser_unlocked()
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Failed to launch browser: {exc}",
                )

    def _cleanup_browser(self) -> None:
        """Close browser and Playwright, releasing all resources.

        Safe to call even if no session is active.  Acquires the
        instance lock internally.
        """
        with self._lock:
            self._cleanup_browser_unlocked()

    def _cleanup_browser_unlocked(self) -> None:
        """Close browser without acquiring the lock.

        Must only be called while holding ``self._lock``.
        """
        if self._cleanup_timer is not None:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None

        if self._page is not None:
            with contextlib.suppress(Exception):
                self._page.close()
            self._page = None

        if self._browser is not None:
            with contextlib.suppress(Exception):
                self._browser.close()
            self._browser = None

        if self._playwright_ctx is not None:
            with contextlib.suppress(Exception):
                self._playwright_ctx.stop()
            self._playwright_ctx = None

        logger.debug("browser_session_closed")

    def _touch_activity(self) -> None:
        """Record the time of the last browser activity."""
        self._last_activity = _time.monotonic()

    def _schedule_cleanup(self) -> None:
        """Schedule auto-close after inactivity timeout.

        Cancels any previously scheduled timer before creating a new one.
        """
        if self._cleanup_timer is not None:
            self._cleanup_timer.cancel()

        self._cleanup_timer = threading.Timer(
            _INACTIVITY_TIMEOUT,
            self._auto_close,
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def _auto_close(self) -> None:
        """Auto-close browser if inactive for too long.

        Called by the cleanup timer.  Checks elapsed time since last
        activity to avoid premature closure if a request arrived
        just before the timer fired.
        """
        with self._lock:
            elapsed = _time.monotonic() - self._last_activity
            if elapsed >= _INACTIVITY_TIMEOUT:
                logger.info(
                    "browser_auto_close",
                    idle_seconds=round(elapsed, 1),
                )
                self._cleanup_browser_unlocked()
            else:
                # Activity occurred since timer was scheduled; reschedule.
                remaining = _INACTIVITY_TIMEOUT - elapsed
                self._cleanup_timer = threading.Timer(
                    remaining,
                    self._auto_close,
                )
                self._cleanup_timer.daemon = True
                self._cleanup_timer.start()

    # ------------------------------------------------------------------
    # URL safety (mirrors BrowseWebTool._is_safe_url)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_safe_url(url: str) -> bool:
        """Reject file://, localhost (except Ollama), private IPs.

        Args:
            url: The URL to validate.

        Returns:
            True if the URL is safe to navigate to.
        """
        try:
            parsed = urlparse(url)
        except Exception:
            return False

        if parsed.scheme not in ("http", "https"):
            return False

        hostname = parsed.hostname or ""

        # Localhost check
        if hostname in (
            "localhost", "127.0.0.1", "::1", "0.0.0.0",
        ):
            port = parsed.port
            return bool(
                port is not None
                and port in _LOCALHOST_ALLOWED_PORTS
            )

        # Private IP check
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_reserved or ip.is_loopback:
                return False
        except ValueError:
            # Not an IP -- check hostname heuristics
            for prefix in _PRIVATE_PREFIXES:
                if hostname.startswith(prefix):
                    return False

        return True

    # ------------------------------------------------------------------
    # Page info helpers
    # ------------------------------------------------------------------

    def _get_page_info(self) -> dict[str, Any]:
        """Gather basic info about the current page state.

        Returns:
            Dict with url, title, and (optionally) visible_text.
        """
        info: dict[str, Any] = {}
        if self._page is not None:
            try:
                info["url"] = self._page.url
            except Exception:
                info["url"] = "unknown"
            try:
                info["title"] = self._page.title()
            except Exception:
                info["title"] = "unknown"
        return info

    def _get_visible_text(self, max_chars: int = 2000) -> str:
        """Extract visible text from the current page.

        Args:
            max_chars: Maximum characters to return.

        Returns:
            Visible text content, truncated if needed.
        """
        if self._page is None:
            return ""
        try:
            text: str = self._page.evaluate(
                "document.body ? document.body.innerText : ''"
            )
            if len(text) > max_chars:
                return text[:max_chars] + "\n[... truncated]"
            return text
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _action_navigate(
        self, args: dict[str, Any], timeout: int
    ) -> ToolResult:
        """Navigate to a URL.

        Args:
            args: Must contain ``url``.
            timeout: Navigation timeout in seconds.
        """
        url = args.get("url")
        if not url:
            return ToolResult(
                success=False,
                output="",
                error="'url' parameter is required for navigate action.",
            )

        if not self._is_safe_url(url):
            return ToolResult(
                success=False,
                output="",
                error=f"URL rejected by security policy: {url}",
            )

        launch_error = self._ensure_browser(timeout)
        if launch_error is not None:
            return launch_error

        self._page.goto(url, timeout=timeout * 1000)
        self._touch_activity()
        self._schedule_cleanup()

        info = self._get_page_info()
        visible = self._get_visible_text()

        return ToolResult(
            success=True,
            output=(
                f"Navigated to {info.get('url', url)}\n"
                f"Title: {info.get('title', 'N/A')}\n\n"
                f"{visible}"
            ),
            metadata=info,
        )

    def _action_click(
        self, args: dict[str, Any], timeout: int
    ) -> ToolResult:
        """Click an element by CSS selector.

        Args:
            args: Must contain ``selector``.
            timeout: Action timeout in seconds.
        """
        selector = args.get("selector")
        if not selector:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "'selector' parameter is required for "
                    "click action."
                ),
            )

        launch_error = self._ensure_browser(timeout)
        if launch_error is not None:
            return launch_error

        self._page.click(selector, timeout=timeout * 1000)
        self._touch_activity()
        self._schedule_cleanup()

        info = self._get_page_info()
        visible = self._get_visible_text()

        return ToolResult(
            success=True,
            output=(
                f"Clicked element: {selector}\n"
                f"Page: {info.get('url', 'N/A')}\n"
                f"Title: {info.get('title', 'N/A')}\n\n"
                f"{visible}"
            ),
            metadata={**info, "selector": selector},
        )

    def _action_fill(
        self, args: dict[str, Any], timeout: int
    ) -> ToolResult:
        """Fill an input field with text (clears existing content first).

        Args:
            args: Must contain ``selector`` and ``text``.
            timeout: Action timeout in seconds.
        """
        selector = args.get("selector")
        text = args.get("text")
        if not selector:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "'selector' parameter is required for "
                    "fill action."
                ),
            )
        if text is None:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "'text' parameter is required for "
                    "fill action."
                ),
            )

        launch_error = self._ensure_browser(timeout)
        if launch_error is not None:
            return launch_error

        self._page.fill(selector, text, timeout=timeout * 1000)
        self._touch_activity()
        self._schedule_cleanup()

        info = self._get_page_info()

        return ToolResult(
            success=True,
            output=(
                f"Filled '{selector}' with text "
                f"({len(text)} chars)\n"
                f"Page: {info.get('url', 'N/A')}"
            ),
            metadata={
                **info,
                "selector": selector,
                "text_length": len(text),
            },
        )

    def _action_type(
        self, args: dict[str, Any], timeout: int
    ) -> ToolResult:
        """Type text character by character (for autocomplete fields).

        Args:
            args: Must contain ``selector`` and ``text``.
            timeout: Action timeout in seconds.
        """
        selector = args.get("selector")
        text = args.get("text")
        if not selector:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "'selector' parameter is required for "
                    "type action."
                ),
            )
        if text is None:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "'text' parameter is required for "
                    "type action."
                ),
            )

        launch_error = self._ensure_browser(timeout)
        if launch_error is not None:
            return launch_error

        self._page.type(selector, text, timeout=timeout * 1000)
        self._touch_activity()
        self._schedule_cleanup()

        info = self._get_page_info()

        return ToolResult(
            success=True,
            output=(
                f"Typed into '{selector}' "
                f"({len(text)} chars)\n"
                f"Page: {info.get('url', 'N/A')}"
            ),
            metadata={
                **info,
                "selector": selector,
                "text_length": len(text),
            },
        )

    def _action_select(
        self, args: dict[str, Any], timeout: int
    ) -> ToolResult:
        """Select a dropdown option by value.

        Args:
            args: Must contain ``selector`` and ``value``.
            timeout: Action timeout in seconds.
        """
        selector = args.get("selector")
        value = args.get("value")
        if not selector:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "'selector' parameter is required for "
                    "select action."
                ),
            )
        if not value:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "'value' parameter is required for "
                    "select action."
                ),
            )

        launch_error = self._ensure_browser(timeout)
        if launch_error is not None:
            return launch_error

        self._page.select_option(
            selector, value, timeout=timeout * 1000
        )
        self._touch_activity()
        self._schedule_cleanup()

        info = self._get_page_info()

        return ToolResult(
            success=True,
            output=(
                f"Selected '{value}' in '{selector}'\n"
                f"Page: {info.get('url', 'N/A')}"
            ),
            metadata={
                **info,
                "selector": selector,
                "selected_value": value,
            },
        )

    def _action_screenshot(
        self, args: dict[str, Any], timeout: int
    ) -> ToolResult:
        """Take a screenshot of the current page.

        Args:
            args: Optional parameters (unused beyond timeout).
            timeout: Action timeout in seconds.

        Returns:
            ToolResult with base64-encoded PNG in output.
        """
        launch_error = self._ensure_browser(timeout)
        if launch_error is not None:
            return launch_error

        screenshot_bytes: bytes = self._page.screenshot(
            full_page=True
        )
        self._touch_activity()
        self._schedule_cleanup()

        encoded = base64.b64encode(screenshot_bytes).decode("utf-8")
        info = self._get_page_info()

        return ToolResult(
            success=True,
            output=encoded,
            metadata={
                **info,
                "format": "png",
                "size_bytes": len(screenshot_bytes),
            },
        )

    def _action_scroll(
        self, args: dict[str, Any], timeout: int
    ) -> ToolResult:
        """Scroll the page up or down.

        Args:
            args: Must contain ``direction`` ('up' or 'down').
            timeout: Action timeout in seconds.
        """
        direction = args.get("direction", "down")
        if direction not in ("up", "down"):
            return ToolResult(
                success=False,
                output="",
                error=(
                    "'direction' must be 'up' or 'down', "
                    f"got '{direction}'."
                ),
            )

        launch_error = self._ensure_browser(timeout)
        if launch_error is not None:
            return launch_error

        # Scroll by ~600px (roughly one viewport height)
        delta = -600 if direction == "up" else 600
        self._page.evaluate(f"window.scrollBy(0, {delta})")
        self._touch_activity()
        self._schedule_cleanup()

        info = self._get_page_info()
        scroll_y: int = self._page.evaluate("window.scrollY")

        return ToolResult(
            success=True,
            output=(
                f"Scrolled {direction}\n"
                f"Scroll position: {scroll_y}px\n"
                f"Page: {info.get('url', 'N/A')}"
            ),
            metadata={
                **info,
                "direction": direction,
                "scroll_y": scroll_y,
            },
        )

    def _action_get_text(
        self, args: dict[str, Any], timeout: int
    ) -> ToolResult:
        """Get text content of an element or the full page.

        Args:
            args: Optional ``selector``. If omitted, returns full
                page text.
            timeout: Action timeout in seconds.
        """
        launch_error = self._ensure_browser(timeout)
        if launch_error is not None:
            return launch_error

        selector = args.get("selector")
        self._touch_activity()
        self._schedule_cleanup()

        info = self._get_page_info()

        if selector:
            element = self._page.query_selector(selector)
            if element is None:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Element not found: {selector}",
                    metadata=info,
                )
            text: str = element.inner_text()
        else:
            text = self._get_visible_text(max_chars=8000)

        return ToolResult(
            success=True,
            output=text,
            metadata={
                **info,
                "selector": selector or "body",
                "text_length": len(text),
            },
        )

    def _action_wait(
        self, args: dict[str, Any], timeout: int
    ) -> ToolResult:
        """Wait for an element to appear on the page.

        Args:
            args: Must contain ``selector``.
            timeout: Maximum wait time in seconds.
        """
        selector = args.get("selector")
        if not selector:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "'selector' parameter is required for "
                    "wait action."
                ),
            )

        launch_error = self._ensure_browser(timeout)
        if launch_error is not None:
            return launch_error

        self._page.wait_for_selector(
            selector, timeout=timeout * 1000
        )
        self._touch_activity()
        self._schedule_cleanup()

        info = self._get_page_info()

        return ToolResult(
            success=True,
            output=(
                f"Element found: {selector}\n"
                f"Page: {info.get('url', 'N/A')}"
            ),
            metadata={**info, "selector": selector},
        )

    def _action_evaluate(
        self, args: dict[str, Any], timeout: int
    ) -> ToolResult:
        """Evaluate JavaScript on the current page.

        Args:
            args: Must contain ``script``.
            timeout: Action timeout in seconds.
        """
        script = args.get("script")
        if not script:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "'script' parameter is required for "
                    "evaluate action."
                ),
            )

        launch_error = self._ensure_browser(timeout)
        if launch_error is not None:
            return launch_error

        result = self._page.evaluate(script)
        self._touch_activity()
        self._schedule_cleanup()

        info = self._get_page_info()
        output = str(result) if result is not None else "undefined"

        return ToolResult(
            success=True,
            output=output,
            metadata={**info, "script": script},
        )

    def _action_back(
        self, args: dict[str, Any], timeout: int
    ) -> ToolResult:
        """Navigate back in browser history.

        Args:
            args: Unused.
            timeout: Navigation timeout in seconds.
        """
        launch_error = self._ensure_browser(timeout)
        if launch_error is not None:
            return launch_error

        self._page.go_back(timeout=timeout * 1000)
        self._touch_activity()
        self._schedule_cleanup()

        info = self._get_page_info()
        visible = self._get_visible_text()

        return ToolResult(
            success=True,
            output=(
                "Navigated back\n"
                f"Page: {info.get('url', 'N/A')}\n"
                f"Title: {info.get('title', 'N/A')}\n\n"
                f"{visible}"
            ),
            metadata=info,
        )

    def _action_close(
        self, args: dict[str, Any], timeout: int
    ) -> ToolResult:
        """Close the browser session and release resources.

        Args:
            args: Unused.
            timeout: Unused.
        """
        self._cleanup_browser()

        return ToolResult(
            success=True,
            output="Browser session closed.",
            metadata={},
        )
