"""基于 AST 的结构化代码检索工具（Skill 模块）：
- view_outline: 查看文件类/函数骨架与行号（节省 Token）
- search_symbol: 精准查找类/函数/方法定义位置
- find_references: 跨文件查找函数/方法调用点
"""

from __future__ import annotations

from pydantic import Field

from mca.codeindex import CodeIndex
from mca.security import is_sensitive_path
from mca.tools.base import Tool, ToolBlocked, ToolContext, ToolError, ToolParams


def _get_index(ctx: ToolContext) -> CodeIndex:
    idx = getattr(ctx, "code_index", None)
    if idx is None:
        idx = CodeIndex(ctx.workspace)
        idx.build()
    return idx


class ViewOutlineParams(ToolParams):
    path: str = Field(description="Python 文件路径，相对仓库根目录")


class ViewOutlineTool(Tool):
    name = "view_outline"
    description = (
        "通过 AST 提取 Python 文件的类、函数、方法签名及起止行号（不返回函数体）。"
        "阅读大文件前应优先调用此工具定位目标函数行号，再用 read_file 精准读取，可大幅节省 Token。"
    )
    params_model = ViewOutlineParams
    readonly = True

    async def run(self, params: ViewOutlineParams, ctx: ToolContext) -> str:
        p = ctx.resolve_path(params.path)
        if is_sensitive_path(p):
            raise ToolBlocked(f"'{params.path}' 为敏感文件，禁止访问。")
        if not p.exists():
            raise ToolError(f"文件不存在: {params.path}。可先用 list_dir 确认路径。")
        if p.is_dir():
            raise ToolError(f"'{params.path}' 是目录，请使用 list_dir。")
        if p.suffix != ".py":
            raise ToolError(f"view_outline 目前仅支持 .py 文件，'{params.path}' 请用 read_file 查看。")
        return _get_index(ctx).file_outline(ctx.relpath(p))


class SearchSymbolParams(ToolParams):
    name: str = Field(min_length=1, description="要查找的类名、函数名或方法名（支持精确匹配与模糊子串）")


class SearchSymbolTool(Tool):
    name = "search_symbol"
    description = "在仓库 AST 索引中查找类/函数/方法的定义位置（文件路径、签名、起止行号）。比纯文本搜索更精准。"
    params_model = SearchSymbolParams
    readonly = True

    async def run(self, params: SearchSymbolParams, ctx: ToolContext) -> str:
        hits = _get_index(ctx).search_symbol(params.name)
        if not hits:
            return f"未找到名为 '{params.name}' 的符号定义。可尝试用 search_code 做正则搜索。"
        lines = [f"找到 {len(hits)} 处定义:"]
        for s in hits:
            lines.append(f"- {s.file}:{s.start_line}-{s.end_line} ({s.kind}) {s.signature}")
        return "\n".join(lines)


class FindReferencesParams(ToolParams):
    name: str = Field(min_length=1, description="要查找调用点的函数名或方法名")


class FindReferencesTool(Tool):
    name = "find_references"
    description = "通过 AST 查找函数或方法在整个仓库中的调用位置（file:line + 代码行）。用于分析修改影响面。"
    params_model = FindReferencesParams
    readonly = True

    async def run(self, params: FindReferencesParams, ctx: ToolContext) -> str:
        refs = _get_index(ctx).find_references(params.name)
        if not refs:
            return f"未找到对 '{params.name}' 的调用引用。"
        lines = [f"找到 {len(refs)} 处调用引用:"]
        for r in refs:
            lines.append(f"- {r.file}:{r.line}: {r.snippet}")
        return "\n".join(lines)
