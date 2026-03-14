"""
MCP Client helper for connecting to the files-mcp server from Python scripts.

Usage:
    from common.mcp_client import mcp_tools

    async def main():
        async with mcp_tools(registry) as mcp_tool_names:
            # registry now has MCP tools, use mcp_tool_names in agent defs
            ...
"""

import os
import json
import asyncio
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from common.tools import ToolRegistry, ToolResult


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_MCP_CONNECT_RETRIES = 3
_MCP_RETRY_DELAY = 1.0


def _server_params() -> StdioServerParameters:
    """Build fresh server params each time (env snapshot at call time)."""
    mcp_dir = os.path.join(PROJECT_ROOT, "mcp", "files-mcp")
    return StdioServerParameters(
        command="node",
        args=[
            os.path.join(mcp_dir, "node_modules", "tsx", "dist", "cli.mjs"),
            os.path.join(mcp_dir, "src", "index.ts"),
        ],
        env={
            **os.environ,
            "LOG_LEVEL": "info",
            "FS_ROOT": os.path.join(PROJECT_ROOT, "workspace"),
        },
    )


@asynccontextmanager
async def get_mcp_client(server_params: StdioServerParameters | None = None):
    """Connect to the MCP server and yield an initialized ClientSession.

    Retries up to _MCP_CONNECT_RETRIES times to work around intermittent
    Windows stdio race conditions with npx.
    """
    params = server_params or _server_params()
    last_err = None
    for attempt in range(1, _MCP_CONNECT_RETRIES + 1):
        try:
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session
                    return
        except Exception as e:
            last_err = e
            if attempt < _MCP_CONNECT_RETRIES:
                print(f"  MCP connect attempt {attempt} failed ({e}), retrying...")
                await asyncio.sleep(_MCP_RETRY_DELAY)
            else:
                raise


async def call_tool(session: ClientSession, tool_name: str, arguments: dict) -> str:
    """Call an MCP tool and return the text result."""
    result = await session.call_tool(tool_name, arguments)
    texts = [block.text for block in result.content if hasattr(block, "text")]
    return "\n".join(texts)


async def list_tools(session: ClientSession) -> list[dict]:
    """List all available MCP tools with their names and descriptions."""
    result = await session.list_tools()
    return [{"name": t.name, "description": t.description} for t in result.tools]


@asynccontextmanager
async def mcp_tools(
    tool_registry: ToolRegistry,
    server_params: StdioServerParameters | None = None,
):
    """Context manager that connects to MCP, registers tools, yields tool names.

    Usage:
        async with mcp_tools(registry) as tool_names:
            # tool_names = ["fs_read", "fs_search", ...]
            # registry now has these tools registered
            await run_repl(...)
    """
    async with get_mcp_client(server_params) as session:
        result = await session.list_tools()
        registered = []

        for tool in result.tools:
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        k: v for k, v in tool.inputSchema.items()
                        if k != "$schema"
                    },
                },
            }

            tool_name = tool.name  # capture for closure

            async def handler(_tool_name=tool_name, **kwargs) -> ToolResult:
                text = await call_tool(session, _tool_name, kwargs)
                return ToolResult(data=text)

            tool_registry.register(schema, handler)
            registered.append(tool.name)
            print(f"  MCP REGISTERED: {tool.name}")

        yield registered


if __name__ == "__main__":
    import asyncio

    async def main():
        print("Connecting to files-mcp server...")
        async with get_mcp_client() as client:
            tools = await list_tools(client)
            print(f"\nAvailable tools ({len(tools)}):")
            for t in tools:
                print(f"  - {t['name']}: {t['description'][:80]}")

            # Quick test: write and read a file
            await call_tool(client, "fs_write", {
                "path": "test_hello.txt",
                "operation": "create",
                "content": "Hello from MCP Python client!",
            })
            result = await call_tool(client, "fs_read", {"path": "test_hello.txt"})
            print(f"\nTest read result:\n{result}")

    asyncio.run(main())
