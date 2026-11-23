"""确定性守卫层：预算检查 + LoopGuard 死循环/卡死检测。

为什么不能指望大模型“自己意识到陷入死循环”？
模型在上下文里反复看到相同报错时，注意力反而容易被锚定在错误路径上（“越改越偏”）。
因此卡死检测必须由确定性代码完成：一旦命中规则，立即强制转入 RECOVERING 状态，
回滚工作区并注入结构化反思提示。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass

from mca.agent.state import ActionRecord, AgentState, Budget


@dataclass(frozen=True)
class StuckReason:
    code: str
    message: str


def hash_args(args_json: str) -> str:
    """规范化 JSON 后取短哈希，避免字段顺序或空格不同导致漏检重复调用。"""
    try:
        canonical = json.dumps(json.loads(args_json), sort_keys=True, ensure_ascii=False)
    except Exception:
        canonical = args_json.strip()
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def extract_test_failure_signature(output: str) -> str:
    """从 pytest 输出中提取稳定的失败签名（去除耗时等易变信息）。"""
    if "[pytest PASSED" in output:
        return "PASSED"
    failed_lines = [
        line.strip()
        for line in output.splitlines()
        if line.startswith(("FAILED ", "E   ", "AssertionError", "SyntaxError", "ImportError"))
    ]
    raw = "\n".join(failed_lines[:6]) if failed_lines else re.sub(r"\d+(\.\d+)?\s*m?s", "", output[:300])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def check_budget(state: AgentState, budget: Budget) -> str | None:
    """超预算时返回 exit_reason，否则返回 None。"""
    if state.step_no >= budget.max_steps:
        return "budget_steps"
    if state.total_tokens >= budget.max_tokens:
        return "budget_tokens"
    if state.elapsed_s >= budget.max_wall_time_s:
        return "budget_time"
    return None


class LoopGuard:
    """四条确定性卡死检测规则（可配置开关，便于后续做消融实验）。"""

    def __init__(
        self,
        *,
        enabled: bool = True,
        repeat_window: int = 8,
        max_same_call: int = 3,
        max_same_test_failure: int = 3,
        max_same_file_edits: int = 5,
        max_consecutive_arg_errors: int = 3,
    ) -> None:
        self.enabled = enabled
        self.repeat_window = repeat_window
        self.max_same_call = max_same_call
        self.max_same_test_failure = max_same_test_failure
        self.max_same_file_edits = max_same_file_edits
        self.max_consecutive_arg_errors = max_consecutive_arg_errors

    def check(self, state: AgentState) -> StuckReason | None:
        if not self.enabled:
            return None

        # 规则 1：连续参数校验错误 / 工具名幻觉
        if state.consecutive_arg_errors >= self.max_consecutive_arg_errors:
            return StuckReason(
                code="repeated_arg_errors",
                message=f"连续 {state.consecutive_arg_errors} 次工具调用参数校验失败",
            )

        # 规则 2：最近窗口内完全相同的 (tool_name, args_hash) 重复出现
        recent = state.actions[-self.repeat_window :]
        call_counts = Counter((a.tool_name, a.args_hash) for a in recent)
        for (tool_name, _), count in call_counts.items():
            if count >= self.max_same_call:
                return StuckReason(
                    code="repeated_identical_call",
                    message=f"最近 {len(recent)} 次动作中，使用完全相同的参数调用了 {tool_name} {count} 次",
                )

        # 规则 3：连续多次 run_tests 报完全相同的失败签名
        test_actions = [a for a in state.actions if a.tool_name == "run_tests"]
        if len(test_actions) >= self.max_same_test_failure:
            tail = test_actions[-self.max_same_test_failure :]
            sigs = {a.signature for a in tail}
            if len(sigs) == 1 and next(iter(sigs)) not in ("", "PASSED"):
                return StuckReason(
                    code="repeated_test_failure",
                    message=f"连续 {self.max_same_test_failure} 次运行测试均出现相同的失败结果，当前修改方向无效",
                )

        # 规则 4：自上次测试通过以来，同一文件被反复修改过多次数
        edits_since_pass: Counter[str] = Counter()
        for a in reversed(state.actions):
            if a.tool_name == "run_tests" and a.signature == "PASSED":
                break
            if a.tool_name == "edit_file" and a.status == "ok" and a.signature:
                edits_since_pass[a.signature] += 1
        for path, count in edits_since_pass.items():
            if count >= self.max_same_file_edits:
                return StuckReason(
                    code="excessive_file_edits",
                    message=f"文件 '{path}' 已被连续修改 {count} 次但测试仍未通过",
                )

        return None

    @staticmethod
    def build_recovery_prompt(reason: StuckReason, rollback_sha: str) -> str:
        return (
            f"[系统熔断通知 - LoopGuard 触发: {reason.code}]\n"
            f"原因: {reason.message}。\n"
            f"系统已自动将工作区回滚到安全检查点 ({rollback_sha[:7]})。\n"
            f"请暂停当前思路，先简要分析为什么之前的尝试没有奏效，然后提出一个**不同的假设与修改方案**再继续操作。"
        )
