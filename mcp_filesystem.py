import os
import shutil
import json
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("filesystem-agent")

# ── Register tool list ───────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_files",
            description="List all files recursively in a directory with their filename lengths",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Path to the directory to scan"
                    }
                },
                "required": ["directory"]
            }
        ),
        Tool(
            name="move_file",
            description="Move a file to a target directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Full path to the source file"
                    },
                    "destination_dir": {
                        "type": "string",
                        "description": "Target directory path"
                    }
                },
                "required": ["source", "destination_dir"]
            }
        )
    ]

# ── Handle tool calls ────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "list_files":
        directory = arguments["directory"]
        results = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                full_path = os.path.join(root, f)
                name_no_ext = os.path.splitext(f)[0]
                results.append({
                    "path": full_path,
                    "filename": f,
                    "name_length": len(name_no_ext)
                })
        return [TextContent(type="text", text=json.dumps(results))]

    elif name == "move_file":
        source = arguments["source"]
        destination_dir = arguments["destination_dir"]
        os.makedirs(destination_dir, exist_ok=True)
        dest = os.path.join(destination_dir, os.path.basename(source))
        shutil.move(source, dest)
        return [TextContent(type="text", text=f"Moved {source} → {dest}")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

# ── Entry point ──────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())