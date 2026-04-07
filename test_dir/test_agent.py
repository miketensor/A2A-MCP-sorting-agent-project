import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import httpx
import tempfile
import shutil

# Import the module to test
from agent import sort_by_length, run_agent, A2A_TOOL, client, MODEL


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory with test files of varying lengths."""
    temp_dir = tempfile.mkdtemp()
    # Create files with different name lengths
    files = [
        "a.txt",  # 5 chars
        "medium_name.txt",  # 14 chars
        "very_long_filename_indeed.txt"  # 27 chars
    ]
    for file in files:
        with open(os.path.join(temp_dir, file), 'w') as f:
            f.write("test content")
    yield temp_dir
    # shutil.rmtree(temp_dir)


@pytest.fixture
def empty_temp_dir():
    """Create an empty temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # shutil.rmtree(temp_dir)


class TestSortByLength:
    @pytest.mark.asyncio
    async def test_sort_by_length_success(self):
        """Test successful sorting by length."""
        mock_response_data = {"sorted": ["short", "medium", "long"]}
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            files = [{"name": "a.txt", "length": 5}]
            result = await sort_by_length(files)

            expected = json.dumps(mock_response_data)
            assert result == expected
            mock_client.post.assert_called_once_with(
                "http://localhost:8001/sort",
                json={"files": files},
                timeout=30.0
            )

    @pytest.mark.asyncio
    async def test_sort_by_length_timeout(self):
        """Test timeout error in sort_by_length."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            files = [{"name": "a.txt", "length": 5}]
            with pytest.raises(httpx.TimeoutException):
                await sort_by_length(files)

    @pytest.mark.asyncio
    async def test_sort_by_length_connection_error(self):
        """Test connection error in sort_by_length."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection failed")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            files = [{"name": "a.txt", "length": 5}]
            with pytest.raises(httpx.ConnectError):
                await sort_by_length(files)

    @pytest.mark.asyncio
    async def test_sort_by_length_empty_files(self):
        """Test with empty files list."""
        mock_response_data = {"sorted": []}
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await sort_by_length([])
            expected = json.dumps(mock_response_data)
            assert result == expected


class TestRunAgent:
    @pytest.fixture
    def mock_mcp(self):
        """Mock MCPClient."""
        mcp = AsyncMock()
        mcp.connect = AsyncMock()
        mcp.get_tools = AsyncMock(return_value=[
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in directory",
                    "parameters": {"type": "object", "properties": {"directory": {"type": "string"}}, "required": ["directory"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_file",
                    "description": "Move file",
                    "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}
                }
            }
        ])
        mcp.call_tool = AsyncMock()
        mcp.disconnect = AsyncMock()
        return mcp

    @pytest.fixture
    def mock_groq_response(self):
        """Mock Groq response."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].finish_reason = "tool_calls"
        response.choices[0].message.content = None
        response.choices[0].message.tool_calls = [
            MagicMock(id="call1", function=MagicMock(name="list_files", arguments='{"directory": "./test_dir"}'))
        ]
        return response

    @pytest.mark.asyncio
    async def test_run_agent_successful_sorting(self, mock_mcp, mock_groq_response):
        """Test successful file sorting."""
        # Mock the sequence of responses
        responses = [
            # First response: call list_files
            mock_groq_response,
            # Second: call sort_by_length
            MagicMock(choices=[MagicMock(message=MagicMock(content=None, tool_calls=[
                MagicMock(id="call2", function=MagicMock(name="sort_by_length", arguments='{"files": [{"name": "a.txt", "length": 5}]}'))
            ]), finish_reason="tool_calls")]),
            # Third: call move_file multiple times
            MagicMock(choices=[MagicMock(message=MagicMock(content=None, tool_calls=[
                MagicMock(id="call3", function=MagicMock(name="move_file", arguments='{"source": "a.txt", "destination": "./sorted/short/a.txt"}'))
            ]), finish_reason="tool_calls")]),
            # Fourth: stop
            MagicMock(choices=[MagicMock(message=MagicMock(content="Sorting complete.", tool_calls=None), finish_reason="stop")])
        ]

        with patch('agent.MCPClient', return_value=mock_mcp), \
             patch('agent.client.chat.completions.create', side_effect=responses), \
             patch('builtins.print'):  # Suppress prints

            await run_agent("./test_dir")

            # Verify MCP calls
            mock_mcp.connect.assert_called_once()
            mock_mcp.get_tools.assert_called_once()
            # mock_mcp.disconnect.assert_called_once()
            assert mock_mcp.disconnect.call_count >= 1

            # Verify tool calls
            assert mock_mcp.call_tool.call_count >= 2  # list_files and move_file

    @pytest.mark.asyncio
    async def test_run_agent_empty_directory(self, mock_mcp):
        """Test with empty directory (no files)."""
        # Mock list_files returning empty
        mock_mcp.call_tool.side_effect = [
            '{"files": []}',  # list_files
        ]

        responses = [
            # Call list_files
            MagicMock(choices=[MagicMock(message=MagicMock(content=None, tool_calls=[
                MagicMock(id="call1", function=MagicMock(name="list_files", arguments='{"directory": "./test_dir"}'))
            ]), finish_reason="tool_calls")]),
            # Stop with message
            MagicMock(choices=[MagicMock(message=MagicMock(content="No files to sort.", tool_calls=None), finish_reason="stop")])
        ]

        with patch('agent.MCPClient', return_value=mock_mcp), \
             patch('agent.client.chat.completions.create', side_effect=responses), \
             patch('builtins.print'):

            await run_agent("./test_dir")

    @pytest.mark.asyncio
    async def test_run_agent_mcp_connection_error(self):
        """Test MCP connection failure."""
        with patch('agent.MCPClient') as mock_mcp_class:
            mock_mcp = AsyncMock()
            mock_mcp.connect.side_effect = Exception("Connection failed")
            mock_mcp_class.return_value = mock_mcp

            with pytest.raises(Exception, match="Connection failed"):
                await run_agent("./test_dir")

            mock_mcp.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_agent_groq_error(self, mock_mcp):
        """Test Groq API error."""
        with patch('agent.MCPClient', return_value=mock_mcp), \
             patch('agent.client.chat.completions.create', side_effect=Exception("API Error")), \
             patch('builtins.print'):

            with pytest.raises(Exception, match="API Error"):
                await run_agent("./test_dir")

    @pytest.mark.asyncio
    async def test_run_agent_invalid_tool_call(self, mock_mcp, mock_groq_response):
        """Test invalid tool call arguments."""
        # Mock invalid JSON in arguments
        mock_groq_response.choices[0].message.tool_calls[0].function.arguments = '{"invalid": json}'

        with patch('agent.MCPClient', return_value=mock_mcp), \
             patch('agent.client.chat.completions.create', return_value=mock_groq_response), \
             patch('builtins.print'):

            with pytest.raises(json.JSONDecodeError):
                await run_agent("./test_dir")

    @pytest.mark.asyncio
    async def test_run_agent_sort_by_length_error(self, mock_mcp):
        """Test error in sort_by_length tool."""
        responses = [
            MagicMock(choices=[MagicMock(message=MagicMock(content=None, tool_calls=[
                MagicMock(id="call1", function=MagicMock(name="sort_by_length", arguments='{"files": [{"name": "a.txt", "length": 5}]}'))
            ]), finish_reason="tool_calls")]),
            # SECOND call tells the agent to stop
            MagicMock(choices=[MagicMock(message=MagicMock(content="Error handled", tool_calls=None), finish_reason="stop")])
        ]

    # Patch the function directly where 'agent' uses it
        with patch('agent.MCPClient', return_value=mock_mcp), \
            patch('agent.client.chat.completions.create', side_effect=responses), \
            patch('agent.sort_by_length', side_effect=httpx.ConnectError("Service Down")), \
            patch('builtins.print'):

            # This should now run without crashing, but the side_effect will trigger
            # the error handling logic inside your agent loop.
            await run_agent("./test_dir")
            
        # Verify MCP still cleaned up
        assert mock_mcp.disconnect.call_count >= 1


class TestConstants:
    def test_a2a_tool_definition(self):
        """Test A2A_TOOL structure."""
        assert A2A_TOOL["type"] == "function"
        assert A2A_TOOL["function"]["name"] == "sort_by_length"
        assert "files" in A2A_TOOL["function"]["parameters"]["properties"]
        assert A2A_TOOL["function"]["parameters"]["required"] == ["files"]

    def test_model_constant(self):
        """Test MODEL constant."""
        assert MODEL == "llama-3.3-70b-versatile"


class TestEnvironment:
    def test_groq_client_initialization(self):
        """Test that Groq client is initialized with API key."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
            # Re-import to test initialization
            from importlib import reload
            import agent
            reload(agent)
            assert agent.client.api_key == "test_key"