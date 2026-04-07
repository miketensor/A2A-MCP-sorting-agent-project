from mcp.server import Server
from mcp.server.stdio import stdio_server
import os, shutil, asyncio, json

server = Server("filesystem-agent")

@server.tool()
async def list_files(directory: str) -> str:
    """List all files recursively with their filename lengths"""
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
    return json.dumps(results)

@server.tool()
async def move_file(source: str, destination_dir: str) -> str:
    """Move a file to a target directory"""
    os.makedirs(destination_dir, exist_ok=True)
    dest = os.path.join(destination_dir, os.path.basename(source))
    shutil.move(source, dest)
    return f"Moved {source} → {dest}"

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())