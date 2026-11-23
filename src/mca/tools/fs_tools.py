"""文件系统只读工具：list_dir / read_file / search_code。

每个工具的设计都围绕两个目标：
  1. 省上下文：输出有上限、带分页提示，不让一次调用撑爆窗口
  2. 可行动的反馈：出错时告诉模型下一步该怎么做
"""

from __future__ import annotations

import re
from fnmatch import fnmatch
from pathlib import Path

from pydantic import Field

from mca.fs_utils import IGNORED_DIRS, iter_files
from mca.security import is_sensitive_path
from mca.tools.base import Tool, ToolBlocked, ToolContext, ToolError, ToolParams

MAX_FILE_BYTES = 1_000_000  # 超过 1MB 的文件视为非源码（数据/生成文件），搜索时跳过


# ============================== list_dir ==============================

class ListDirParams(ToolParams):
    path: str = Field(".", description="要列出的目录，相对仓库根目录")
    depth: int = Field(2, ge=1, le=4, description="递归深度，1 表示只列当前层")


class ListDirTool(Tool):
    name = "list_dir"
    description = "以树形结构列出目录内容（自动忽略 .git、虚拟环境、缓存等目录）。目录以 / 结尾。"
    params_model = ListDirParams
    MAX_ENTRIES = 300

    async def run(self, params: ListDirParams, ctx: ToolContext) -> str:
        root = ctx.resolve_path(params.path)
        if not root.exists():
            raise ToolError(f"目录不存在: {params.path}")
        if root.is_file():
            raise ToolError(f"'{params.path}' 是文件而不是目录，请使用 read_file 查看。")

        lines: list[str] = []
        overflow = False

        def walk(d: Path, level: int) -> None:
            nonlocal overflow
            try:
                children = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except PermissionError:
                return
            for child in children:
                if child.is_dir() and (child.name in IGNORED_DIRS or child.name.endswith(".egg-info")):
                    continue
                if len(lines) >= self.MAX_ENTRIES:
                    overflow = True
                    return
                indent = "  " * (level - 1)
                lines.append(f"{indent}{child.name}/" if child.is_dir() else f"{indent}{child.name}")
                if child.is_dir() and level < params.depth:
                    walk(child, level + 1)

        walk(root, 1)
        header = f"[{ctx.relpath(root) or '.'}]"
        if not lines:
            return f"{header}\n(空目录)"
        footer = f"\n[条目过多，仅显示前 {self.MAX_ENTRIES} 项，请指定更深的子目录或减小 depth]" if overflow else ""
        return header + "\n" + "\n".join(lines) + footer


# ============================== read_file ==============================

class ReadFileParams(ToolParams):
    path: str = Field(description="文件路径，相对仓库根目录")
    start_line: int = Field(1, ge=1, description="起始行号（从 1 开始，包含）")
    end_line: int | None = Field(None, ge=1, description="结束行号（包含）；不填则读到文件末尾或单次上限")


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "读取文本文件的指定行范围，输出带行号（格式 '行号| 内容'，行号前缀不是文件内容的一部分）。"
        "单次最多返回 200 行，长文件请分段读取。"
    )
    params_model = ReadFileParams
    max_output_chars = 20_000
    MAX_LINES = 200

    async def run(self, params: ReadFileParams, ctx: ToolContext) -> str:
        p = ctx.resolve_path(params.path)
        if not p.exists():
            raise ToolError(f"文件不存在: {params.path}。可先用 list_dir 或 search_code 确认路径。")
        if p.is_dir():
            raise ToolError(f"'{params.path}' 是目录，请使用 list_dir 查看。")
        if is_sensitive_path(p):
            raise ToolBlocked(
                f"'{params.path}' 可能包含密钥等敏感信息，禁止读取。"
                f"如需了解配置项，请阅读对应的模板文件（如 .env.example）或配置类源码。"
            )
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"'{params.path}' 不是 UTF-8 文本文件（可能是二进制文件），无法读取。") from None

        lines = text.splitlines()
        total = len(lines)
        rel = ctx.relpath(p)
        if total == 0:
            return f"[{rel} 是空文件]"
        start = params.start_line
        if start > total:
            raise ToolError(f"start_line={start} 超出文件总行数 {total}。")
        if params.end_line is not None and params.end_line < start:
            raise ToolError(f"end_line={params.end_line} 小于 start_line={start}。")

        end = min(params.end_line or total, total, start + self.MAX_LINES - 1)
        width = len(str(end))
        body = "\n".join(f"{i:>{width}}| {lines[i - 1]}" for i in range(start, end + 1))
        out = f"[{rel} 第 {start}-{end} 行，共 {total} 行]\n{body}"
        if end < total:
            out += f"\n[未显示完。如需继续阅读，请使用 start_line={end + 1}]"
        return out


# ============================== search_code ==============================

class SearchCodeParams(ToolParams):
    pattern: str = Field(description="Python 正则表达式，按行匹配")
    path: str = Field(".", description="搜索的目录或文件，相对仓库根目录")
    file_glob: str | None = Field(None, description="按文件名过滤，如 '*.py'")
    case_sensitive: bool = Field(True, description="是否区分大小写")


class SearchCodeTool(Tool):
    name = "search_code"
    description = "在仓库中按正则搜索代码（类似 grep），返回 '路径:行号: 内容'。用于定位符号、字符串、报错信息。"
    params_model = SearchCodeParams
    MAX_RESULTS = 50

    async def run(self, params: SearchCodeParams, ctx: ToolContext) -> str:
        try:
            rx = re.compile(params.pattern, 0 if params.case_sensitive else re.IGNORECASE)
        except re.error as e:
            raise ToolError(f"正则表达式无效: {e}。若要搜索字面量，请对特殊字符转义，如 \\( \\. \\[") from None

        root = ctx.resolve_path(params.path)
        if not root.exists():
            raise ToolError(f"路径不存在: {params.path}")
        files = [root] if root.is_file() else iter_files(root)

        hits: list[str] = []
        total = 0
        for f in files:
            if params.file_glob and not fnmatch(f.name, params.file_glob):
                continue
            if is_sensitive_path(f):
                continue  # 否则 search_code("API_KEY") 就能绕过 read_file 的拦截
            try:
                if f.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # 二进制 / 无权限文件静默跳过
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    total += 1
                    if len(hits) < self.MAX_RESULTS:
                        hits.append(f"{ctx.relpath(f)}:{i}: {line.strip()[:200]}")

        if total == 0:
            # 空结果不是错误，而是有价值的信息（"这个符号不存在"）—— status 仍然是 ok
            return f"未找到匹配 /{params.pattern}/ 的内容。"
        out = "\n".join(hits)
        if total > len(hits):
            out += f"\n[共匹配 {total} 处，仅显示前 {self.MAX_RESULTS} 处。请缩小 path、使用 file_glob 或更精确的 pattern]"
        return out
