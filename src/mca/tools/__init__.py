from mca.tools.base import Tool, ToolBlocked, ToolContext, ToolError, ToolParams, ToolResult, ToolStatus
from mca.tools.edit_tools import EditFileTool
from mca.tools.exec_tools import RunCommandTool, RunTestsTool
from mca.tools.fs_tools import ListDirTool, ReadFileTool, SearchCodeTool
from mca.tools.registry import ToolRegistry
from mca.tools.submit_tool import SubmitTool


def default_tools() -> list[Tool]:
    return [
        ListDirTool(),
        ReadFileTool(),
        SearchCodeTool(),
        EditFileTool(),
        RunCommandTool(),
        RunTestsTool(),
    ]


def fix_tools() -> list[Tool]:
    """Fix 模式完整工具集（含 submit）。"""
    return [*default_tools(), SubmitTool()]


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
    "SubmitTool",
    "default_tools",
    "fix_tools",
]
