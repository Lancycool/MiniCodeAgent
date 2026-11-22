"""ToolRegistry：注册、启停、分发、校验、超时、异常隔离、输出截断。

核心原则：**任何情况下 execute() 都不抛异常**，所有失败都被转换成一个带 status 的
ToolResult 回传给模型。Agent 循环不会因为一次工具失败而崩溃，模型也能看到错误并自我修正。
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import time
from collections.abc import Collection, Iterable, Sequence
from typing import Any

from pydantic import ValidationError

from mca.llm.types import ToolCall
from mca.security import redact_secrets
from mca.tools.base import Tool, ToolBlocked, ToolContext, ToolError, ToolResult, ToolStatus
from mca.tools.truncate import truncate_middle

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = (), *, enabled: Collection[str] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)
        # enabled=None 表示全部启用；消融实验时传入子集，例如去掉 AST 工具
        self._enabled = set(enabled) if enabled is not None else None

    # ---------------- 注册与查询 ----------------

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def active_tools(self) -> list[Tool]:
        return [t for n, t in self._tools.items() if self._enabled is None or n in self._enabled]

    def get(self, name: str) -> Tool | None:
        tool = self._tools.get(name)
        if tool is None or (self._enabled is not None and name not in self._enabled):
            return None
        return tool

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.active_tools()]

    # ---------------- 执行 ----------------

    async def execute_many(self, calls: Sequence[ToolCall], ctx: ToolContext) -> list[ToolResult]:
        """执行一轮中的所有工具调用，结果顺序与 calls 一致。

        全部是只读工具 → 并发执行（互不影响，省时间）；
        只要有一个写工具 → 严格按顺序执行（模型可能依赖"先改 A 再跑测试"的顺序，
        并发会产生竞态条件）。
        """
        tools = [self.get(c.name) for c in calls]
        if len(calls) > 1 and all(t is not None and t.readonly for t in tools):
            return list(await asyncio.gather(*(self.execute(c, ctx) for c in calls)))
        return [await self.execute(c, ctx) for c in calls]

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        start = time.perf_counter()
        result = await self._execute(call, ctx)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    async def _execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        # ① 工具是否存在
        tool = self.get(call.name)
        if tool is None:
            names = [t.name for t in self.active_tools()]
            close = difflib.get_close_matches(call.name, names, n=3)
            hint = f"你是不是想调用: {', '.join(close)}？" if close else ""
            return _result(call, "unknown_tool",
                           f"错误: 不存在名为 '{call.name}' 的工具。{hint}可用工具: {', '.join(names)}")

        # ② 参数 JSON 解析
        try:
            raw = json.loads(call.arguments) if call.arguments.strip() else {}
        except json.JSONDecodeError as e:
            return _result(call, "invalid_args", _format_json_error(call.arguments, e))
        if not isinstance(raw, dict):
            return _result(call, "invalid_args", f"错误: 参数必须是 JSON 对象，收到的是 {type(raw).__name__}。")

        # ③ 参数 Schema 校验
        try:
            params = tool.params_model.model_validate(raw)
        except ValidationError as e:
            return _result(call, "invalid_args", _format_validation_error(tool.name, e))

        # ④ 执行（超时 + 异常隔离）
        try:
            output = await asyncio.wait_for(tool.run(params, ctx), timeout=tool.timeout_s)
        except TimeoutError:
            return _result(call, "timeout",
                           f"错误: 工具 {tool.name} 执行超过 {tool.timeout_s:.0f}s 被终止。请缩小操作范围后重试。")
        except ToolBlocked as e:
            logger.warning("Tool %s blocked by policy: %s", tool.name, call.arguments)
            return _result(call, "blocked", f"已拦截: {e}")
        except ToolError as e:
            return _result(call, "error", f"错误: {e}")
        except Exception as e:  # noqa: BLE001 —— 工具 bug 不能拖垮 Agent
            logger.exception("Tool %s crashed with args %s", tool.name, call.arguments)
            return _result(call, "internal_error",
                           f"工具内部错误（{type(e).__name__}: {e}）。这不是参数问题，请尝试用其他方式完成任务。")

        # ⑤ 密钥脱敏（第二道防线，先于截断：避免密钥恰好被截在边界上导致正则漏匹配）
        output, redactions = redact_secrets(output)
        if redactions:
            logger.warning("Redacted %d secret(s) from %s output", redactions, tool.name)

        # ⑥ 输出截断
        text, truncated = truncate_middle(output, tool.max_output_chars)
        return ToolResult(call.id, call.name, "ok", text,
                          raw_chars=len(output), truncated=truncated, redactions=redactions)


# ---------------- 错误格式化：把异常翻译成模型能看懂、能改正的反馈 ----------------

def _result(call: ToolCall, status: ToolStatus, content: str) -> ToolResult:
    return ToolResult(call.id, call.name, status, content, raw_chars=len(content))


def _format_json_error(arguments: str, e: json.JSONDecodeError) -> str:
    lo, hi = max(0, e.pos - 30), e.pos + 30
    snippet = arguments[lo:hi].replace("\n", "\\n")
    return (f"错误: 参数不是合法的 JSON（{e.msg}，位置 {e.pos}）。\n"
            f"出错位置附近: ...{snippet}...\n"
            f"请检查引号、逗号、括号是否配对，字符串中的换行和引号是否已转义。")


def _format_validation_error(tool_name: str, e: ValidationError) -> str:
    lines = [f"错误: 工具 {tool_name} 的参数校验失败:"]
    for err in e.errors(include_url=False):
        loc = ".".join(str(x) for x in err["loc"]) or "(参数整体)"
        if err["type"] == "missing":
            lines.append(f"- `{loc}`: 缺少必填参数")
        elif err["type"] == "extra_forbidden":
            lines.append(f"- `{loc}`: 不存在该参数，请删除")
        else:
            lines.append(f"- `{loc}`: {err['msg']}（收到: {err.get('input')!r}）")
    lines.append("请修正参数后重新调用。")
    return "\n".join(lines)
