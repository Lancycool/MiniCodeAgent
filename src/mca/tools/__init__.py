from mca.tools.base import Tool, ToolBlocked, ToolContext, ToolError, ToolParams, ToolResult, ToolStatus
from mca.tools.edit_tools import EditFileTool
from mca.tools.exec_tools import RunCommandTool, RunTestsTool
from mca.tools.fs_tools import ListDirTool, ReadFileTool, SearchCodeTool
from mca.tools.registry import ToolRegistry


def default_tools() -> list[Tool]:
    return [
        ListDirTool(),
        ReadFileTool(),
        SearchCodeTool(),
        EditFileTool(),
        RunCommandTool(),
        RunTestsTool(),
    ]


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
    "EditFileTool",
    "RunCommandTool",
    "RunTestsTool",
    "default_tools",
]
