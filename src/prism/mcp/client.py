"""MCP (Model Context Protocol) client for connecting to external tool servers.

Implements a JSON-RPC 2.0 client that communicates with MCP servers over
stdin/stdout (stdio transport).  Servers are configured in
``.prism/mcp.json`` or ``.prism-mcp.json``.

Usage::

    client = MCPClient(project_root)
    client.load_config()          # reads .prism/mcp.json
    client.connect_all()          # starts all configured servers
    tools = client.list_tools()   # get available tools
    result = client.call_tool("server_name", "tool_name", {"arg": "value"})
    client.disconnect_all()       # cleanup
"""

from __future__ import annotations

import json
import os
import random
import select
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    name: str
    command: str  # e.g. "npx @modelcontextprotocol/server-filesystem /path"
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class MCPTool:
    """A tool exposed by an MCP server."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    server_name: str


@dataclass
class MCPResource:
    """A resource exposed by an MCP server."""

    uri: str
    name: str
    description: str
    mime_type: str | None = None
    server_name: str = ""


# ---------------------------------------------------------------------------
# MCP Client
# ---------------------------------------------------------------------------


class MCPClient:
    """Client that manages connections to MCP servers.

    Supports stdio-based MCP servers.  Communicates via JSON-RPC 2.0 over
    stdin/stdout.

    Args:
        project_root: The project directory that contains MCP configuration
            files and serves as the working directory for spawned servers.
    """

    # Protocol version we advertise during ``initialize``.
    _PROTOCOL_VERSION = "2024-11-05"

    # Maximum seconds to wait for a response from the server.
    _READ_TIMEOUT_SECONDS = 10.0

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root
        self._configs: list[MCPServerConfig] = []
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._tools: dict[str, list[MCPTool]] = {}  # server_name -> tools
        self._resources: dict[str, list[MCPResource]] = {}  # server_name -> resources

    @property
    def servers(self) -> list[MCPServerConfig]:
        """Return the list of configured MCP servers."""
        return list(self._configs)

    # ----- configuration ----------------------------------------------------

    def load_config(self) -> None:
        """Load MCP server configs from ``.prism/mcp.json`` or ``.prism-mcp.json``.

        Supports two top-level keys: ``mcpServers`` (Claude-compatible) and
        ``servers`` (alternative).  The first config file found wins.
        """
        for name in (".prism/mcp.json", ".prism-mcp.json"):
            config_path = self._project_root / name
            if config_path.exists():
                try:
                    data = json.loads(config_path.read_text(encoding="utf-8"))
                    servers = data.get("mcpServers", data.get("servers", {}))
                    for server_name, server_conf in servers.items():
                        self._configs.append(
                            MCPServerConfig(
                                name=server_name,
                                command=server_conf.get("command", ""),
                                args=server_conf.get("args", []),
                                env=server_conf.get("env", {}),
                                enabled=server_conf.get("enabled", True),
                            )
                        )
                    logger.debug("mcp_config_loaded", servers=len(self._configs))
                except Exception as exc:
                    logger.debug("mcp_config_error", error=str(exc))
                break

    # ----- connection management --------------------------------------------

    def connect(self, server_name: str) -> bool:
        """Start and connect to a specific MCP server.

        Sends the ``initialize`` handshake, the ``notifications/initialized``
        notification, and then lists available tools.

        Returns:
            ``True`` if the server was started and the handshake succeeded.
        """
        config = next((c for c in self._configs if c.name == server_name), None)
        if not config or not config.enabled:
            return False

        try:
            env = {**os.environ, **config.env}
            cmd = [config.command, *config.args]

            proc: subprocess.Popen[bytes] = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(self._project_root),
            )
            self._processes[server_name] = proc

            # -- MCP handshake ------------------------------------------------
            init_result = self._send_request(
                server_name,
                "initialize",
                {
                    "protocolVersion": self._PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "prism", "version": "0.2.0"},
                },
            )

            if init_result is None:
                self.disconnect(server_name)
                return False

            # Acknowledge initialisation
            self._send_notification(server_name, "notifications/initialized", {})

            # Discover tools
            tools_result = self._send_request(server_name, "tools/list", {})
            if tools_result and "tools" in tools_result:
                self._tools[server_name] = [
                    MCPTool(
                        name=t["name"],
                        description=t.get("description", ""),
                        parameters_schema=t.get("inputSchema", {}),
                        server_name=server_name,
                    )
                    for t in tools_result["tools"]
                ]

            logger.debug(
                "mcp_connected",
                server=server_name,
                tools=len(self._tools.get(server_name, [])),
            )
            return True

        except FileNotFoundError:
            logger.debug("mcp_server_not_found", command=config.command)
        except Exception as exc:
            logger.debug("mcp_connect_error", server=server_name, error=str(exc))

        return False

    def connect_all(self) -> dict[str, bool]:
        """Connect to all enabled configured servers.

        Returns:
            A mapping of ``{server_name: success_bool}`` for each enabled
            server.
        """
        results: dict[str, bool] = {}
        for config in self._configs:
            if config.enabled:
                results[config.name] = self.connect(config.name)
        return results

    def disconnect(self, server_name: str) -> None:
        """Disconnect from a specific MCP server.

        Terminates the subprocess.  If graceful termination fails the process
        is killed forcefully.
        """
        proc = self._processes.pop(server_name, None)
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        self._tools.pop(server_name, None)
        self._resources.pop(server_name, None)

    def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for name in list(self._processes.keys()):
            self.disconnect(name)

    # ----- tool discovery ---------------------------------------------------

    def list_tools(self) -> list[MCPTool]:
        """List all tools from all connected servers."""
        all_tools: list[MCPTool] = []
        for tools in self._tools.values():
            all_tools.extend(tools)
        return all_tools

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-format tool schemas for all MCP tools.

        Each tool name is prefixed with ``mcp_{server}_{tool}`` so the
        agentic loop can route calls back to :pymeth:`handle_mcp_tool_call`.
        """
        schemas: list[dict[str, Any]] = []
        for tool in self.list_tools():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": f"mcp_{tool.server_name}_{tool.name}",
                        "description": f"[MCP:{tool.server_name}] {tool.description}",
                        "parameters": tool.parameters_schema
                        or {"type": "object", "properties": {}},
                    },
                }
            )
        return schemas

    # ----- tool execution ---------------------------------------------------

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on a specific MCP server.

        Args:
            server_name: The name of the MCP server hosting the tool.
            tool_name: The tool name as advertised by the server.
            arguments: The tool input arguments.

        Returns:
            A string representation of the tool result.
        """
        result = self._send_request(
            server_name,
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )

        if result is None:
            return f"Error: MCP server '{server_name}' not responding"

        if "error" in result:
            return f"Error: {result['error']}"

        if "content" in result:
            parts: list[str] = []
            for item in result["content"]:
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    parts.append("[image data]")
            return "\n".join(parts) if parts else str(result)

        return str(result)

    def handle_mcp_tool_call(
        self, full_tool_name: str, arguments: dict[str, Any]
    ) -> str | None:
        """Handle a tool call that might be an MCP tool.

        MCP tool names are prefixed: ``mcp_{server}_{tool}``.

        Returns:
            The tool result string, or ``None`` if *full_tool_name* is not an
            MCP tool.
        """
        if not full_tool_name.startswith("mcp_"):
            return None

        remainder = full_tool_name[4:]  # strip "mcp_"
        parts = remainder.split("_", 1)
        if len(parts) != 2:
            return f"Error: Invalid MCP tool name '{full_tool_name}'"

        server_name, tool_name = parts
        return self.call_tool(server_name, tool_name, arguments)

    # ----- JSON-RPC transport -----------------------------------------------

    def _send_request(
        self, server_name: str, method: str, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Send a JSON-RPC 2.0 request and return the ``result`` field.

        Returns ``None`` on transport errors or timeout.
        """
        proc = self._processes.get(server_name)
        if not proc or proc.poll() is not None:
            return None

        request_id = random.randint(1, 999_999)  # noqa: S311

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        try:
            request_bytes = (json.dumps(request) + "\n").encode()
            assert proc.stdin is not None  # guaranteed by Popen args
            proc.stdin.write(request_bytes)
            proc.stdin.flush()

            # Read response with timeout
            assert proc.stdout is not None  # guaranteed by Popen args
            ready, _, _ = select.select([proc.stdout], [], [], self._READ_TIMEOUT_SECONDS)
            if not ready:
                return None

            line = proc.stdout.readline()
            if not line:
                return None

            response = json.loads(line.decode())
            if "result" in response:
                return response["result"]
            if "error" in response:
                error = response["error"]
                return {"error": error.get("message", str(error))}
            return response

        except Exception as exc:
            logger.debug("mcp_request_error", method=method, error=str(exc))
            return None

    def _send_notification(
        self, server_name: str, method: str, params: dict[str, Any]
    ) -> None:
        """Send a JSON-RPC 2.0 notification (no response expected)."""
        proc = self._processes.get(server_name)
        if not proc or proc.poll() is not None:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            notification_bytes = (json.dumps(notification) + "\n").encode()
            assert proc.stdin is not None  # guaranteed by Popen args
            proc.stdin.write(notification_bytes)
            proc.stdin.flush()
        except Exception:
            logger.debug("mcp_exit_notification_failed")

    # ----- properties -------------------------------------------------------

    @property
    def connected_servers(self) -> list[str]:
        """List names of currently connected servers."""
        return [name for name, proc in self._processes.items() if proc.poll() is None]

    @property
    def is_active(self) -> bool:
        """``True`` if any MCP servers are connected."""
        return len(self.connected_servers) > 0

    @property
    def configs(self) -> list[MCPServerConfig]:
        """Read-only access to the loaded server configurations."""
        return list(self._configs)
