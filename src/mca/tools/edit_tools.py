"""代码编辑工具：Search-Replace + 唯一性校验 + AST 语法预检 + git checkpoint。

为什么用 Search-Replace 而不是整文件重写或行号替换？
- 整文件重写：大文件消耗大量输出 Token，且模型经常"顺手"改坏无关代码；
- 行号替换：连续编辑两次后行号漂移，模型极易算错行号；
- Search-Replace：以代码内容本身为锚点，不依赖行号，最稳健（Aider / Claude Code 同款）。
"""

from __future__ import annotations

import ast
import difflib
import re
from pathlib import Path

from pydantic import Field

from mca.security import is_sensitive_path
from mca.tools.base import Tool, ToolBlocked, ToolContext, ToolError, ToolParams

_LINE_PREFIX_RE = re.compile(r"^\s*\d+\|\s?", re.MULTILINE)


class EditFileParams(ToolParams):
    path: str = Field(description="要修改的文件路径，相对仓库根目录")
    old_str: str = Field(
        min_length=1,
        description="文件中要被替换的原始代码片段，必须与文件内容逐字一致且在文件中恰好出现一次（不要带 read_file 的行号前缀）",
    )
    new_str: str = Field(description="替换后的新代码片段（允许为空字符串，表示删除 old_str）")


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "通过精确文本替换（Search-Replace）修改文件。要求 old_str 在文件中恰好出现一次；"
        "修改 Python 文件后会自动做 AST 语法检查，若存在语法错误会自动撤销修改并返回报错。"
    )
    params_model = EditFileParams
    readonly = False  # 写操作：不可与其他工具并发执行

    async def run(self, params: EditFileParams, ctx: ToolContext) -> str:
        p = ctx.resolve_path(params.path)
        rel = ctx.relpath(p)

        # ① 安全与权限检查
        if is_sensitive_path(p):
            raise ToolBlocked(f"'{rel}' 为敏感文件，禁止修改。")
        if ctx.is_protected(rel):
            raise ToolBlocked(f"'{rel}' 受保护（如测试文件），当前模式禁止修改。请修复业务代码本身。")
        if not p.exists():
            raise ToolError(f"文件不存在: {rel}。可先用 list_dir 确认路径。")
        if p.is_dir():
            raise ToolError(f"'{rel}' 是目录，无法编辑。")
        if params.old_str == params.new_str:
            raise ToolError("old_str 与 new_str 完全相同，未发生任何修改。")

        try:
            raw_bytes = p.read_bytes()
            original_raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"'{rel}' 不是 UTF-8 文本文件，无法编辑。") from None

        # 保留文件原有换行符风格（Windows 下直接 write_text 会把 LF 转为 CRLF，污染 Linux 仓库 git diff）
        line_ending = "\r\n" if "\r\n" in original_raw else "\n"
        original = original_raw.replace("\r\n", "\n")

        # ② 唯一性检查
        count = original.count(params.old_str)
        if count == 0:
            hint = _diagnose_not_found(original, params.old_str)
            raise ToolError(f"未在 '{rel}' 中找到完全匹配的 old_str。{hint}")
        if count > 1:
            line_nos = _find_occurrence_lines(original, params.old_str)
            lines_str = ", ".join(str(n) for n in line_nos[:5])
            raise ToolError(
                f"old_str 在 '{rel}' 中出现了 {count} 次（起始行: {lines_str}），无法确定要替换哪一处。"
                f"请在 old_str 前后多包含几行上下文，使其在文件中唯一。"
            )

        updated = original.replace(params.old_str, params.new_str, 1)

        # ③ 确定性语法预检（Python 文件在写入前直接用 ast.parse 校验，有语法错误直接拒收）
        if p.suffix == ".py":
            try:
                ast.parse(updated, filename=rel)
            except SyntaxError as e:
                line_info = f"第 {e.lineno} 行" if e.lineno else ""
                err_line = (e.text or "").rstrip()
                raise ToolError(
                    f"修改后的 Python 代码存在语法错误，已自动拒绝本次修改（原文件未被改动）。\n"
                    f"SyntaxError ({line_info}): {e.msg}\n"
                    f"问题行: {err_line}\n"
                    f"请检查缩进、括号、冒号后重新调用 edit_file。"
                ) from None

        # ④ 写入文件（严格保留原换行符）+ git checkpoint
        with p.open("w", encoding="utf-8", newline=line_ending) as f:
            f.write(updated)
        sha_note = ""
        if ctx.sandbox is not None:
            sha = await ctx.sandbox.checkpoint(f"edit {rel}")
            sha_note = f"（checkpoint: {sha[:7]}）"

        # ⑤ 返回修改处周围的带行号预览，让模型确认修改落点无误
        preview = _format_edit_preview(updated, params.new_str, original.index(params.old_str))
        return f"已成功修改 {rel}{sha_note}。修改后上下文预览:\n{preview}"


def _find_occurrence_lines(text: str, sub: str) -> list[int]:
    lines: list[int] = []
    start = 0
    while True:
        idx = text.find(sub, start)
        if idx == -1:
            break
        lines.append(text.count("\n", 0, idx) + 1)
        start = idx + 1
    return lines


def _diagnose_not_found(content: str, old_str: str) -> str:
    """当 old_str 找不到时，给出最接近的片段和针对性提示。"""
    hints: list[str] = []
    if _LINE_PREFIX_RE.search(old_str):
        hints.append("检测到 old_str 中似乎包含 '行号| ' 前缀，请去掉 read_file 输出的行号前缀。")

    # 用滑动窗口找最相似的代码块，帮助模型发现缩进/空格差异
    file_lines = content.splitlines()
    target_lines = old_str.splitlines()
    window = max(1, len(target_lines))
    best_ratio = 0.0
    best_start = 0
    for i in range(max(1, len(file_lines) - window + 1)):
        chunk = "\n".join(file_lines[i : i + window])
        ratio = difflib.SequenceMatcher(None, old_str, chunk).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    if best_ratio >= 0.55 and file_lines:
        snippet = "\n".join(
            f"{best_start + idx + 1}| {line}"
            for idx, line in enumerate(file_lines[best_start : best_start + window])
        )
        hints.append(f"最相近的片段（相似度 {best_ratio:.0%}）:\n{snippet}")
    else:
        hints.append("请先用 read_file 确认文件当前内容后再修改。")
    return "\n".join(hints)


def _format_edit_preview(updated: str, new_str: str, offset: int, context_lines: int = 3) -> str:
    lines = updated.splitlines()
    if not lines:
        return "(空文件)"
    start_line = updated.count("\n", 0, offset) + 1
    added_lines = max(1, new_str.count("\n") + 1)
    lo = max(1, start_line - context_lines)
    hi = min(len(lines), start_line + added_lines - 1 + context_lines)
    width = len(str(hi))
    return "\n".join(f"{i:>{width}}| {lines[i - 1]}" for i in range(lo, hi + 1))
