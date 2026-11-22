from mca.tools.base import Tool, ToolBlocked, ToolContext, ToolError, ToolParams, ToolResult, ToolStatus
from mca.tools.fs_tools import ListDirTool, ReadFileTool, SearchCodeTool
from mca.tools.registry import ToolRegistry


def default_tools() -> list[Tool]:
    return [ListDirTool(), ReadFileTool(), SearchCodeTool()]


__all__ = [
    "Tool",
    "ToolBlocked",
    "ToolContext",
    "ToolError",
    "ToolParams",
    "ToolResult",
    "ToolStatus",
    "ToolRegistry",
    "ListDirTool",
    "ReadFileTool",
    "SearchCodeTool",
    "default_tools",
]
