import asyncio
import json
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.session = None
        self._exit_stack = AsyncExitStack()

    async def connect(self):
        """Spawn MCP server subprocess and open session"""
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script],
            env=None
        )

        # Use AsyncExitStack to manage both contexts in the same task
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport

        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        await self.session.initialize()
        print(f"✅ MCP connected to {self.server_script}")

    async def get_tools(self) -> list[dict]:
        """Fetch available tools from MCP server and convert to Groq format"""
        result = await self.session.list_tools()
        tools = []
        for tool in result.tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })
        return tools

    async def call_tool(self, name: str, args: dict) -> str:
        """Call a tool on the MCP server and return result as string"""
        result = await self.session.call_tool(name, args)
        return result.content[0].text if result.content else ""

    async def disconnect(self):
        """Clean up all contexts in reverse order"""
        await self._exit_stack.aclose()
        print("🔌 MCP disconnected")