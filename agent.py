import os, json, asyncio, httpx
from groq import Groq
from dotenv import load_dotenv
from mcp_client import MCPClient

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

# ── A2A sub-agent call ───────────────────────────────────────────────────────

async def sort_by_length(files: list) -> str:
    """Delegate sorting logic to A2A sub-agent"""
    async with httpx.AsyncClient() as http:
        response = await http.post(
            "http://localhost:8001/sort",
            json={"files": files},
            timeout=30.0
        )
        return json.dumps(response.json())

# ── A2A tool definition (not in MCP, stays manual) ──────────────────────────

A2A_TOOL = {
    "type": "function",
    "function": {
        "name": "sort_by_length",
        "description": "Delegate sorting of files by filename length to the A2A sub-agent",
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "description": "List of file objects with path, filename, name_length",
                    "items": {"type": "object"}
                }
            },
            "required": ["files"]
        }
    }
}

# ── Agentic loop ─────────────────────────────────────────────────────────────

async def run_agent(directory: str):
    # Connect to MCP server
    mcp = MCPClient("mcp_filesystem.py")
    await mcp.connect()

    # Discover tools from MCP + add A2A tool
    mcp_tools = await mcp.get_tools()
    all_tools = mcp_tools + [A2A_TOOL]

    print(f"\n🛠  Tools loaded from MCP: {[t['function']['name'] for t in mcp_tools]}")
    print(f"🛠  A2A tool: sort_by_length\n")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a file organization agent. "
                "When asked to sort files by filename length:\n"
                "1. Call list_files to get all files\n"
                "2. Call sort_by_length to delegate sorting to the sub-agent\n"
                "3. For each file in each bucket, call move_file to move it to "
                "./sorted/short, ./sorted/medium, or ./sorted/long\n"
                "4. Report a final summary."
            )
        },
        {
            "role": "user",
            "content": f"Sort all files in '{directory}' by filename length."
        }
    ]

    print(f"🤖 Agent starting — sorting files in: {directory}\n")

    try:
        # Agentic loop
        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=all_tools,
                tool_choice="auto",
                max_tokens=4096
            )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Append assistant turn to history
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in (message.tool_calls or [])
                ] or None
            })

            # Done
            if finish_reason == "stop" or not message.tool_calls:
                print("\n✅ Agent finished:\n")
                print(message.content)
                break

            # Handle tool calls
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                print(f"🔧 Calling tool: {name}({args})")

                # Route: MCP tools vs A2A tool
                if name == "sort_by_length":
                    result = await sort_by_length(args["files"])
                else:
                    # All filesystem tools go through MCP
                    result = await mcp.call_tool(name, args)

                print(f"   ↳ {result[:120]}{'...' if len(result) > 120 else ''}\n")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

    finally:
        await mcp.disconnect()

# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "./test_dir"
    asyncio.run(run_agent(directory))