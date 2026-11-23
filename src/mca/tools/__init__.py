from mca.tools.base import Tool, ToolBlocked, ToolContext, ToolError, ToolParams, ToolResult, ToolStatus
from mca.tools.code_search import FindReferencesTool, SearchSymbolTool, ViewOutlineTool
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


def ast_tools() -> list[Tool]:
    """基于 tree-sitter AST 的结构化代码检索工具集（消融实验 E1 的核心变量）。"""
    return [ViewOutlineTool(), SearchSymbolTool(), FindReferencesTool()]


def fix_tools(*, include_ast: bool = True) -> list[Tool]:
    """Fix 模式完整工具集（含 submit；可通过 include_ast 开关做消融实验）。"""
    tools = [*default_tools(), SubmitTool()]
    if include_ast:
        tools.extend(ast_tools())
    return tools


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
    "ViewOutlineTool",
    "SearchSymbolTool",
    "FindReferencesTool",
    "EditFileTool",
    "RunCommandTool",
    "RunTestsTool",
    "SubmitTool",
    "default_tools",
    "ast_tools",
    "fix_tools",
]
