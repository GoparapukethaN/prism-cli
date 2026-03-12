"""Tests for the MCP client module."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from prism.mcp.client import (
    MCPClient,
    MCPResource,
    MCPServerConfig,
    MCPTool,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create a minimal project directory."""
    return tmp_path


@pytest.fixture
def client(project_root: Path) -> MCPClient:
    """Create an MCPClient against a temp directory."""
    return MCPClient(project_root)


@pytest.fixture
def mcp_config_file(project_root: Path) -> Path:
    """Write a sample .prism/mcp.json config and return the path."""
    prism_dir = project_root / ".prism"
    prism_dir.mkdir()
    config_path = prism_dir / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "filesystem": {
                        "command": "npx",
                        "args": ["@modelcontextprotocol/server-filesystem", "/tmp"],
                        "env": {"NODE_ENV": "production"},
                        "enabled": True,
                    },
                    "disabled_server": {
                        "command": "npx",
                        "args": ["@some/server"],
                        "enabled": False,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def alt_config_file(project_root: Path) -> Path:
    """Write a .prism-mcp.json config (alternative location)."""
    config_path = project_root / ".prism-mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "alt_server": {
                        "command": "python",
                        "args": ["-m", "some_server"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _make_mock_process(
    responses: list[dict[str, Any]] | None = None,
    poll_return: int | None = None,
) -> MagicMock:
    """Build a mock subprocess.Popen with controllable stdout responses."""
    proc = MagicMock()
    proc.poll.return_value = poll_return  # None = still running

    if responses:
        lines = [json.dumps(r).encode() + b"\n" for r in responses]
        proc.stdout.readline.side_effect = lines
        proc.stdout.fileno = MagicMock(return_value=99)
    else:
        proc.stdout.readline.return_value = b""
        proc.stdout.fileno = MagicMock(return_value=99)

    proc.stdin.write = MagicMock()
    proc.stdin.flush = MagicMock()
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = MagicMock()

    return proc


# ---------------------------------------------------------------------------
# MCPServerConfig tests
# ---------------------------------------------------------------------------


class TestMCPServerConfig:
    """Tests for the MCPServerConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = MCPServerConfig(name="test", command="echo")
        assert cfg.name == "test"
        assert cfg.command == "echo"
        assert cfg.args == []
        assert cfg.env == {}
        assert cfg.enabled is True

    def test_custom_values(self) -> None:
        cfg = MCPServerConfig(
            name="fs",
            command="npx",
            args=["--flag"],
            env={"KEY": "VAL"},
            enabled=False,
        )
        assert cfg.args == ["--flag"]
        assert cfg.env == {"KEY": "VAL"}
        assert cfg.enabled is False


# ---------------------------------------------------------------------------
# MCPTool tests
# ---------------------------------------------------------------------------


class TestMCPTool:
    """Tests for the MCPTool dataclass."""

    def test_fields(self) -> None:
        tool = MCPTool(
            name="read_file",
            description="Read a file",
            parameters_schema={"type": "object"},
            server_name="fs",
        )
        assert tool.name == "read_file"
        assert tool.description == "Read a file"
        assert tool.parameters_schema == {"type": "object"}
        assert tool.server_name == "fs"


# ---------------------------------------------------------------------------
# MCPResource tests
# ---------------------------------------------------------------------------


class TestMCPResource:
    """Tests for the MCPResource dataclass."""

    def test_defaults(self) -> None:
        res = MCPResource(uri="file:///tmp", name="tmp", description="Temp")
        assert res.mime_type is None
        assert res.server_name == ""

    def test_custom_values(self) -> None:
        res = MCPResource(
            uri="file:///data.json",
            name="data",
            description="Data file",
            mime_type="application/json",
            server_name="fs",
        )
        assert res.mime_type == "application/json"
        assert res.server_name == "fs"


# ---------------------------------------------------------------------------
# MCPClient — init
# ---------------------------------------------------------------------------


class TestMCPClientInit:
    """Tests for MCPClient construction."""

    def test_init_creates_empty_state(self, project_root: Path) -> None:
        c = MCPClient(project_root)
        assert c.list_tools() == []
        assert c.connected_servers == []
        assert c.is_active is False
        assert c.configs == []

    def test_project_root_stored(self, project_root: Path) -> None:
        c = MCPClient(project_root)
        assert c._project_root == project_root


# ---------------------------------------------------------------------------
# MCPClient — load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for MCPClient.load_config."""

    def test_load_prism_dir_config(
        self, client: MCPClient, mcp_config_file: Path
    ) -> None:
        client.load_config()
        assert len(client.configs) == 2
        names = {c.name for c in client.configs}
        assert "filesystem" in names
        assert "disabled_server" in names

    def test_load_alt_config(
        self, client: MCPClient, alt_config_file: Path
    ) -> None:
        client.load_config()
        assert len(client.configs) == 1
        assert client.configs[0].name == "alt_server"

    def test_load_no_config(self, client: MCPClient) -> None:
        """No config file exists — should be a no-op."""
        client.load_config()
        assert client.configs == []

    def test_load_malformed_json(self, project_root: Path) -> None:
        prism_dir = project_root / ".prism"
        prism_dir.mkdir()
        (prism_dir / "mcp.json").write_text("NOT VALID JSON", encoding="utf-8")

        c = MCPClient(project_root)
        c.load_config()  # should not raise
        assert c.configs == []

    def test_prism_dir_takes_priority_over_alt(
        self, project_root: Path, mcp_config_file: Path, alt_config_file: Path
    ) -> None:
        """When both config files exist, .prism/mcp.json wins."""
        c = MCPClient(project_root)
        c.load_config()
        # .prism/mcp.json has 2 servers; .prism-mcp.json has 1
        assert len(c.configs) == 2

    def test_config_env_and_args_parsed(
        self, client: MCPClient, mcp_config_file: Path
    ) -> None:
        client.load_config()
        fs_config = next(c for c in client.configs if c.name == "filesystem")
        assert fs_config.args == [
            "@modelcontextprotocol/server-filesystem",
            "/tmp",
        ]
        assert fs_config.env == {"NODE_ENV": "production"}

    def test_disabled_server_marked(
        self, client: MCPClient, mcp_config_file: Path
    ) -> None:
        client.load_config()
        disabled = next(c for c in client.configs if c.name == "disabled_server")
        assert disabled.enabled is False


# ---------------------------------------------------------------------------
# MCPClient — connect / disconnect
# ---------------------------------------------------------------------------


class TestConnect:
    """Tests for connect, disconnect, connect_all, disconnect_all."""

    def test_connect_unknown_server_returns_false(self, client: MCPClient) -> None:
        assert client.connect("nonexistent") is False

    def test_connect_disabled_server_returns_false(
        self, client: MCPClient, mcp_config_file: Path
    ) -> None:
        client.load_config()
        assert client.connect("disabled_server") is False

    @patch("prism.mcp.client.subprocess.Popen")
    @patch("prism.mcp.client.select.select")
    def test_connect_success(
        self,
        mock_select: MagicMock,
        mock_popen: MagicMock,
        client: MCPClient,
        mcp_config_file: Path,
    ) -> None:
        client.load_config()

        init_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"protocolVersion": "2024-11-05", "capabilities": {}},
        }
        tools_response = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
                    }
                ]
            },
        }

        proc = _make_mock_process(
            responses=[init_response, tools_response], poll_return=None
        )
        mock_popen.return_value = proc
        mock_select.return_value = ([proc.stdout], [], [])

        result = client.connect("filesystem")

        assert result is True
        assert "filesystem" in client.connected_servers
        assert client.is_active is True
        tools = client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "read_file"
        assert tools[0].server_name == "filesystem"

    @patch("prism.mcp.client.subprocess.Popen")
    @patch("prism.mcp.client.select.select")
    def test_connect_init_fails_disconnects(
        self,
        mock_select: MagicMock,
        mock_popen: MagicMock,
        client: MCPClient,
        mcp_config_file: Path,
    ) -> None:
        client.load_config()

        # Return empty line (no valid response) for init
        proc = _make_mock_process(responses=[], poll_return=None)
        mock_popen.return_value = proc
        mock_select.return_value = ([proc.stdout], [], [])

        result = client.connect("filesystem")

        assert result is False

    @patch("prism.mcp.client.subprocess.Popen")
    def test_connect_file_not_found(
        self,
        mock_popen: MagicMock,
        client: MCPClient,
        mcp_config_file: Path,
    ) -> None:
        client.load_config()
        mock_popen.side_effect = FileNotFoundError("npx not found")

        result = client.connect("filesystem")

        assert result is False

    @patch("prism.mcp.client.subprocess.Popen")
    def test_connect_generic_exception(
        self,
        mock_popen: MagicMock,
        client: MCPClient,
        mcp_config_file: Path,
    ) -> None:
        client.load_config()
        mock_popen.side_effect = OSError("spawn failed")

        result = client.connect("filesystem")

        assert result is False

    def test_disconnect_nonexistent_server(self, client: MCPClient) -> None:
        """Disconnecting a server that was never connected should be a no-op."""
        client.disconnect("nonexistent")  # should not raise

    def test_disconnect_kills_on_timeout(self, client: MCPClient) -> None:
        proc = _make_mock_process(poll_return=None)
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="npx", timeout=5)
        client._processes["srv"] = proc

        client.disconnect("srv")

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        assert "srv" not in client._processes

    def test_disconnect_all(self, client: MCPClient) -> None:
        proc_a = _make_mock_process(poll_return=None)
        proc_b = _make_mock_process(poll_return=None)
        client._processes["a"] = proc_a
        client._processes["b"] = proc_b
        client._tools["a"] = []
        client._tools["b"] = []

        client.disconnect_all()

        assert client._processes == {}
        assert client._tools == {}
        proc_a.terminate.assert_called_once()
        proc_b.terminate.assert_called_once()

    @patch("prism.mcp.client.subprocess.Popen")
    @patch("prism.mcp.client.select.select")
    def test_connect_all_returns_mapping(
        self,
        mock_select: MagicMock,
        mock_popen: MagicMock,
        client: MCPClient,
        mcp_config_file: Path,
    ) -> None:
        client.load_config()

        init_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"protocolVersion": "2024-11-05"},
        }
        tools_response = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"tools": []},
        }

        proc = _make_mock_process(
            responses=[init_response, tools_response], poll_return=None
        )
        mock_popen.return_value = proc
        mock_select.return_value = ([proc.stdout], [], [])

        results = client.connect_all()

        # filesystem is enabled, disabled_server is not
        assert "filesystem" in results
        assert results["filesystem"] is True
        assert "disabled_server" not in results


# ---------------------------------------------------------------------------
# MCPClient — tool schemas
# ---------------------------------------------------------------------------


class TestToolSchemas:
    """Tests for get_tool_schemas and list_tools."""

    def test_get_tool_schemas_empty(self, client: MCPClient) -> None:
        assert client.get_tool_schemas() == []

    def test_get_tool_schemas_format(self, client: MCPClient) -> None:
        client._tools["fs"] = [
            MCPTool(
                name="read_file",
                description="Read a file",
                parameters_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                server_name="fs",
            ),
            MCPTool(
                name="write_file",
                description="Write a file",
                parameters_schema={},
                server_name="fs",
            ),
        ]

        schemas = client.get_tool_schemas()

        assert len(schemas) == 2
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "mcp_fs_read_file"
        assert schemas[0]["function"]["description"] == "[MCP:fs] Read a file"
        assert schemas[0]["function"]["parameters"]["type"] == "object"

        # Empty schema gets a default
        assert schemas[1]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }

    def test_list_tools_aggregates_across_servers(self, client: MCPClient) -> None:
        client._tools["a"] = [
            MCPTool(name="t1", description="", parameters_schema={}, server_name="a"),
        ]
        client._tools["b"] = [
            MCPTool(name="t2", description="", parameters_schema={}, server_name="b"),
            MCPTool(name="t3", description="", parameters_schema={}, server_name="b"),
        ]

        tools = client.list_tools()

        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"t1", "t2", "t3"}


# ---------------------------------------------------------------------------
# MCPClient — call_tool
# ---------------------------------------------------------------------------


class TestCallTool:
    """Tests for call_tool and handle_mcp_tool_call."""

    @patch("prism.mcp.client.select.select")
    def test_call_tool_text_content(
        self, mock_select: MagicMock, client: MCPClient
    ) -> None:
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {"type": "text", "text": "hello world"},
                    {"type": "text", "text": "second line"},
                ]
            },
        }
        proc = _make_mock_process(responses=[response], poll_return=None)
        client._processes["srv"] = proc
        mock_select.return_value = ([proc.stdout], [], [])

        result = client.call_tool("srv", "greet", {"name": "test"})

        assert result == "hello world\nsecond line"

    @patch("prism.mcp.client.select.select")
    def test_call_tool_image_content(
        self, mock_select: MagicMock, client: MCPClient
    ) -> None:
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {"type": "image", "data": "base64stuff"},
                ]
            },
        }
        proc = _make_mock_process(responses=[response], poll_return=None)
        client._processes["srv"] = proc
        mock_select.return_value = ([proc.stdout], [], [])

        result = client.call_tool("srv", "screenshot", {})

        assert result == "[image data]"

    def test_call_tool_server_not_connected(self, client: MCPClient) -> None:
        result = client.call_tool("missing", "tool", {})
        assert "not responding" in result

    @patch("prism.mcp.client.select.select")
    def test_call_tool_error_response(
        self, mock_select: MagicMock, client: MCPClient
    ) -> None:
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32600, "message": "Invalid request"},
        }
        proc = _make_mock_process(responses=[response], poll_return=None)
        client._processes["srv"] = proc
        mock_select.return_value = ([proc.stdout], [], [])

        result = client.call_tool("srv", "bad_tool", {})

        assert "Error" in result
        assert "Invalid request" in result

    @patch("prism.mcp.client.select.select")
    def test_call_tool_empty_content(
        self, mock_select: MagicMock, client: MCPClient
    ) -> None:
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": []},
        }
        proc = _make_mock_process(responses=[response], poll_return=None)
        client._processes["srv"] = proc
        mock_select.return_value = ([proc.stdout], [], [])

        result = client.call_tool("srv", "empty_tool", {})

        # Empty parts list -> falls through to str(result)
        assert isinstance(result, str)

    def test_handle_mcp_tool_call_non_mcp_returns_none(
        self, client: MCPClient
    ) -> None:
        result = client.handle_mcp_tool_call("read_file", {"path": "/foo"})
        assert result is None

    def test_handle_mcp_tool_call_invalid_format(
        self, client: MCPClient
    ) -> None:
        result = client.handle_mcp_tool_call("mcp_noseparator", {})
        assert result is not None
        assert "Invalid MCP tool name" in result

    @patch("prism.mcp.client.select.select")
    def test_handle_mcp_tool_call_routes_correctly(
        self, mock_select: MagicMock, client: MCPClient
    ) -> None:
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": "routed!"}]
            },
        }
        proc = _make_mock_process(responses=[response], poll_return=None)
        client._processes["fs"] = proc
        mock_select.return_value = ([proc.stdout], [], [])

        result = client.handle_mcp_tool_call("mcp_fs_read_file", {"path": "/x"})

        assert result == "routed!"


# ---------------------------------------------------------------------------
# MCPClient — JSON-RPC transport
# ---------------------------------------------------------------------------


class TestTransport:
    """Tests for the internal _send_request and _send_notification."""

    @patch("prism.mcp.client.select.select")
    def test_send_request_timeout(
        self, mock_select: MagicMock, client: MCPClient
    ) -> None:
        """When select returns no ready descriptors, result should be None."""
        proc = _make_mock_process(poll_return=None)
        client._processes["srv"] = proc
        mock_select.return_value = ([], [], [])

        result = client._send_request("srv", "tools/list", {})

        assert result is None

    def test_send_request_dead_process(self, client: MCPClient) -> None:
        proc = _make_mock_process(poll_return=1)  # already exited
        client._processes["srv"] = proc

        result = client._send_request("srv", "tools/list", {})

        assert result is None

    def test_send_request_no_process(self, client: MCPClient) -> None:
        result = client._send_request("missing", "tools/list", {})
        assert result is None

    @patch("prism.mcp.client.select.select")
    def test_send_request_json_decode_error(
        self, mock_select: MagicMock, client: MCPClient
    ) -> None:
        proc = _make_mock_process(poll_return=None)
        proc.stdout.readline.return_value = b"NOT JSON\n"
        client._processes["srv"] = proc
        mock_select.return_value = ([proc.stdout], [], [])

        result = client._send_request("srv", "tools/list", {})

        assert result is None

    def test_send_notification_dead_process(self, client: MCPClient) -> None:
        proc = _make_mock_process(poll_return=1)
        client._processes["srv"] = proc

        # Should return without error
        client._send_notification("srv", "notifications/initialized", {})

        proc.stdin.write.assert_not_called()

    def test_send_notification_no_process(self, client: MCPClient) -> None:
        # Should not raise
        client._send_notification("missing", "notifications/initialized", {})

    def test_send_notification_write_error(self, client: MCPClient) -> None:
        proc = _make_mock_process(poll_return=None)
        proc.stdin.write.side_effect = BrokenPipeError("pipe broken")
        client._processes["srv"] = proc

        # Should swallow the exception
        client._send_notification("srv", "test/method", {})


# ---------------------------------------------------------------------------
# MCPClient — properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Tests for connected_servers and is_active properties."""

    def test_connected_servers_filters_dead(self, client: MCPClient) -> None:
        alive = _make_mock_process(poll_return=None)
        dead = _make_mock_process(poll_return=1)
        client._processes["alive"] = alive
        client._processes["dead"] = dead

        assert client.connected_servers == ["alive"]

    def test_is_active_false_when_all_dead(self, client: MCPClient) -> None:
        dead = _make_mock_process(poll_return=1)
        client._processes["dead"] = dead
        assert client.is_active is False

    def test_is_active_true_when_one_alive(self, client: MCPClient) -> None:
        alive = _make_mock_process(poll_return=None)
        client._processes["alive"] = alive
        assert client.is_active is True

    def test_configs_returns_copy(
        self, client: MCPClient, mcp_config_file: Path
    ) -> None:
        client.load_config()
        configs = client.configs
        configs.clear()
        # Internal state should be untouched
        assert len(client.configs) == 2
