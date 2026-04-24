"""Tool registry and built-in file tools.

The registry maps tool names to Tool objects. When the LLM requests
a tool call, the registry looks up the function and executes it.

Built-in tools: read_file, write_file — the minimum needed to let
the LLM interact with the local filesystem.
"""

from __future__ import annotations

from pathlib import Path

from .types import Tool, ToolCall, ToolResult


def _read_file(path: str) -> str:
    """Read a file and return its contents."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return resolved.read_text()


def _write_file(path: str, content: str) -> str:
    """Write content to a file, creating directories if needed."""
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content)
    return f"Wrote {len(content)} bytes to {resolved}"


# Built-in tool definitions
READ_FILE = Tool(
    name="read_file",
    description="Read the contents of a file at the given path.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read",
            },
        },
        "required": ["path"],
    },
    fn=_read_file,
)

WRITE_FILE = Tool(
    name="write_file",
    description="Write content to a file at the given path. Creates parent directories if they don't exist.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["path", "content"],
    },
    fn=_write_file,
)

# All built-in file tools
FILE_TOOLS = [READ_FILE, WRITE_FILE]


class ToolRegistry:
    """Maps tool names to Tool objects and executes calls.

    Usage:
        registry = ToolRegistry(FILE_TOOLS)
        result = registry.execute(tool_call)
    """

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry."""
        self._tools[tool.name] = tool

    @property
    def tools(self) -> list[Tool]:
        """All registered tools."""
        return list(self._tools.values())

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result.

        Catches exceptions and returns them as error results so the
        LLM can see what went wrong and retry.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                content=f"Unknown tool: {call.name}",
                is_error=True,
            )
        try:
            result = tool.fn(**call.arguments)
            return ToolResult(tool_call_id=call.id, content=result)
        except Exception as e:
            return ToolResult(
                tool_call_id=call.id,
                content=f"{type(e).__name__}: {e}",
                is_error=True,
            )
