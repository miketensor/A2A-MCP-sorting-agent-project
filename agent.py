import os, json, shutil, asyncio, httpx
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"  # or llama-3.1-8b-instant for speed

# ── Tool definitions (what MCP would inject in Claude Code) ──────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files recursively in a directory with their filename lengths",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Absolute or relative path to the directory"
                    }
                },
                "required": ["directory"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Move a file to a destination directory",
            "parameters": {
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
        }
    },
    {
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
]

# ── Tool implementations ─────────────────────────────────────────────────────

def list_files(directory: str) -> str:
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

def move_file(source: str, destination_dir: str) -> str:
    os.makedirs(destination_dir, exist_ok=True)
    dest = os.path.join(destination_dir, os.path.basename(source))
    shutil.move(source, dest)
    return f"Moved {source} → {dest}"

async def sort_by_length(files: list) -> str:
    """Delegate to A2A sorting sub-agent"""
    async with httpx.AsyncClient() as http:
        response = await http.post(
            "http://localhost:8001/sort",
            json={"files": files},
            timeout=30.0
        )
        return json.dumps(response.json())

# ── Tool dispatcher ──────────────────────────────────────────────────────────

async def dispatch_tool(name: str, args: dict) -> str:
    if name == "list_files":
        return list_files(args["directory"])
    elif name == "move_file":
        return move_file(args["source"], args["destination_dir"])
    elif name == "sort_by_length":
        return await sort_by_length(args["files"])
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})

# ── Agentic loop ─────────────────────────────────────────────────────────────

async def run_agent(directory: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a file organization agent. "
                "When asked to sort files by filename length:\n"
                "1. Call list_files to get all files in the directory\n"
                "2. Call sort_by_length to delegate sorting to the sub-agent\n"
                "3. For each file in each bucket, call move_file to move it to "
                "./sorted/short, ./sorted/medium, or ./sorted/long\n"
                "4. Report a final summary of what was moved where."
            )
        },
        {
            "role": "user",
            "content": f"Sort all files in '{directory}' by filename length."
        }
    ]

    print(f"\n🤖 Agent starting — sorting files in: {directory}\n")

    # Agentic loop
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=4096
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Append assistant message to history
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
            result = await dispatch_tool(name, args)
            print(f"   ↳ Result: {result[:120]}{'...' if len(result) > 120 else ''}\n")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "./test_dir"
    asyncio.run(run_agent(directory))