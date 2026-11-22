"""沙箱执行工具：run_command / run_tests。

设计要点：
1. 所有命令必须通过 ctx.sandbox 执行，绝不直接在宿主机跑。
2. 确定性危险命令黑名单：拦截破坏性操作、篡改 git 历史、试图联网下载等行为。
3. 单独抽象 run_tests：
   - 对模型语义更清晰；
   - 后续在 Trace 和 Loop Guard 里可以精准识别"测试是否通过"、"是否连续报同一个错"。
"""

from __future__ import annotations

import re

from pydantic import Field

from mca.tools.base import Tool, ToolBlocked, ToolContext, ToolError, ToolParams

# 确定性危险命令拦截规则（应用层防线，与 Docker OS 层隔离构成纵深防御）
_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brm\s+-[^\s]*r[^\s]*f[^\s]*\s+(/|\*|\.\.)"), "禁止高危递归删除命令"),
    (re.compile(r"\bgit\s+(push|reset|checkout|rebase|clean)\b"), "禁止直接操作 git 历史/工作区，回滚由系统管理"),
    (re.compile(r"\b(curl|wget|nc|ncat|ssh|scp)\b"), "禁止网络传输命令"),
    (re.compile(r"\b(pip|pip3|apt|apt-get)\s+install\b"), "沙箱禁网，禁止安装外部依赖"),
    (re.compile(r"\b(sudo|su|chmod\s+777|mkfs|dd\s+if=)\b"), "禁止系统级特权命令"),
    (re.compile(r":\(\)\s*\{"), "检测到疑似 fork 炸弹"),
]


def check_command_safety(command: str) -> None:
    for pattern, reason in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            raise ToolBlocked(f"{reason}（命令: {command!r}）。")


# ============================== run_command ==============================

class RunCommandParams(ToolParams):
    command: str = Field(min_length=1, description="要在沙箱工作区根目录执行的 Shell 命令")
    timeout_s: int = Field(30, ge=1, le=120, description="超时时间（秒），超时将强制终止进程")


class RunCommandTool(Tool):
    name = "run_command"
    description = "在隔离沙箱的仓库根目录执行 Shell 命令（如运行脚本、检查输出）。禁止联网和修改 git 历史。"
    params_model = RunCommandParams
    readonly = False
    timeout_s = 130.0  # 比参数上限略大，让沙箱自身的进程树超时机制先触发

    async def run(self, params: RunCommandParams, ctx: ToolContext) -> str:
        if ctx.sandbox is None:
            raise ToolError("当前环境未配置沙箱，无法执行命令。")
        check_command_safety(params.command)
        res = await ctx.sandbox.exec(params.command, timeout_s=float(params.timeout_s))
        if res.timed_out:
            return (
                f"[命令执行超时（>{params.timeout_s}s），进程已被强制终止]\n"
                f"已捕获输出:\n{res.combined_output()}"
            )
        return (
            f"[exit_code={res.exit_code} | 耗时 {res.duration_ms:.0f}ms]\n"
            f"{res.combined_output()}"
        )


# ============================== run_tests ==============================

class RunTestsParams(ToolParams):
    target: str = Field("", description="测试文件或目录路径（相对仓库根目录）；留空则运行全部测试")
    keyword: str | None = Field(None, description="pytest -k 过滤表达式，只跑匹配的测试用例")


class RunTestsTool(Tool):
    name = "run_tests"
    description = "在沙箱中运行 pytest 测试套件，返回通过/失败结果及简明 traceback。修改代码后应调用此工具验证。"
    params_model = RunTestsParams
    readonly = False
    timeout_s = 120.0

    async def run(self, params: RunTestsParams, ctx: ToolContext) -> str:
        if ctx.sandbox is None:
            raise ToolError("当前环境未配置沙箱，无法运行测试。")

        parts = [ctx.sandbox.python_bin, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider"]
        if params.target.strip():
            # 通过 resolve_path 校验 target，防止借 target 注入路径穿越或 shell 拼接
            p = ctx.resolve_path(params.target.strip())
            if not p.exists():
                raise ToolError(f"测试路径不存在: {params.target}")
            parts.append(f'"{ctx.relpath(p)}"')
        if params.keyword:
            if any(ch in params.keyword for ch in (";", "&", "|", "`", "$", "\n")):
                raise ToolBlocked("keyword 中包含非法 Shell 元字符。")
            parts.extend(["-k", f'"{params.keyword}"'])

        cmd = " ".join(parts)
        res = await ctx.sandbox.exec(cmd, timeout_s=60.0)
        if res.timed_out:
            return f"[测试执行超时（>60s），可能存在死循环]\n{res.combined_output()}"

        status = "PASSED" if res.exit_code == 0 else f"FAILED (exit_code={res.exit_code})"
        return f"[pytest {status} | {res.duration_ms:.0f}ms]\n{res.combined_output()}"
