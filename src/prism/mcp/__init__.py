"""MCP (Model Context Protocol) client for connecting to external tool servers.

Provides an MCP client that communicates with stdio-based MCP servers using
JSON-RPC 2.0.  Servers are configured via ``.prism/mcp.json`` or
``.prism-mcp.json`` in the project root.
"""

from __future__ import annotations

from prism.mcp.client import (
    MCPClient,
    MCPResource,
    MCPServerConfig,
    MCPTool,
)

__all__ = [
    "MCPClient",
    "MCPResource",
    "MCPServerConfig",
    "MCPTool",
]
