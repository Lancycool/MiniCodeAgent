"""AgentLoop：基于有限状态机（FSM）的多轮 Agent 运行时核心。

状态流转：
  INIT ──▶ RUNNING ──(submit)──▶ VERIFYING ──(pass)──▶ DONE
                │ ▲                   │
         (stuck)│(rollback+反思)  (fail, 限次)
              ▼ │                     │
           RECOVERING ◀──────────────┘ (或直接回 RUNNING)
                │
       (超限/超预算) ──▶ FAILED
"""

from __future__ import annotations

import json
import time
from typing import Protocol

from mca.agent.context import ContextConfig, ContextManager
from mca.agent.guards import (
    LoopGuard,
    StuckReason,
    check_budget,
    extract_test_failure_signature,
    hash_args,
)
from mca.agent.state import ActionRecord, AgentState, Budget, Phase, RunResult
from mca.codeindex import CodeIndex, extract_focus_terms
from mca.llm.client import LLMClient
from mca.llm.types import Message
from mca.observability.tracer import Tracer
from mca.sandbox.base import Sandbox
from mca.tools.base import ToolContext
from mca.tools.registry import ToolRegistry

DEFAULT_FIX_SYSTEM_PROMPT = """你是一个严谨的 Python 软件工程师，在隔离沙箱中自主定位并修复代码仓库中的问题。
工作流：
1. 先用 list_dir / search_code / read_file 了解代码，用 run_tests 确认当前失败现象。
2. 分析根因后，使用 edit_file 做最小、精确的修改（不要改动测试文件）。
3. 每次修改后用 run_tests 验证。
4. 当 run_tests 全部通过且确认没有遗漏边界情况时，直接调用 submit 工具提交修复摘要，不要反复执行多余的临时命令。"""


class VerifyResult:
    def __init__(self, passed: bool, message: str) -> None:
        self.passed = passed
        self.message = message


class Verifier(Protocol):
    """确定性裁判接口：判断任务是否真正完成。"""

    async def verify(self, sandbox: Sandbox) -> VerifyResult: ...


class PytestVerifier:
    """用沙箱内的 pytest 作为 Fix 模式的确定性裁判。"""

    def __init__(self, target: str = "", timeout_s: float = 60.0) -> None:
        self.target = target
        self.timeout_s = timeout_s

    async def verify(self, sandbox: Sandbox) -> VerifyResult:
        cmd = f"{sandbox.python_bin} -m pytest -q --tb=short -p no:cacheprovider"
        if self.target:
            cmd += f' "{self.target}"'
        res = await sandbox.exec(cmd, timeout_s=self.timeout_s)
        if res.timed_out:
            return VerifyResult(False, f"验证超时（>{self.timeout_s:.0f}s）:\n{res.combined_output()}")
        if res.exit_code == 0:
            return VerifyResult(True, res.combined_output())
        return VerifyResult(False, f"pytest 未通过 (exit_code={res.exit_code}):\n{res.combined_output()}")


class AgentLoop:
    def __init__(
        self,
        *,
        llm: LLMClient,
        registry: ToolRegistry,
        sandbox: Sandbox,
        verifier: Verifier | None = None,
        budget: Budget | None = None,
        loop_guard: LoopGuard | None = None,
        context_manager: ContextManager | None = None,
        inject_repo_map: bool = False,
        repo_map_max_chars: int = 3500,
        tracer: Tracer | None = None,
        system_prompt: str = DEFAULT_FIX_SYSTEM_PROMPT,
        protected_globs: tuple[str, ...] = ("test_*.py", "tests/*"),
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.sandbox = sandbox
        self.verifier = verifier or PytestVerifier()
        self.budget = budget or Budget()
        self.loop_guard = loop_guard or LoopGuard()
        self.context_manager = context_manager or ContextManager(ContextConfig(mask_enabled=True, summary_enabled=True))
        self.inject_repo_map = inject_repo_map
        self.repo_map_max_chars = repo_map_max_chars
        self.tracer = tracer or Tracer()
        self.system_prompt = system_prompt
        self.protected_globs = protected_globs

    async def run(self, issue: str, *, task_id: str = "ad_hoc", mode: str = "fix") -> RunResult:
        state = AgentState(task_id=task_id, mode=mode)
        code_index = CodeIndex(self.sandbox.workspace)
        code_index.build()
        ctx = ToolContext(
            workspace=self.sandbox.workspace,
            sandbox=self.sandbox,
            code_index=code_index,
            protected_globs=self.protected_globs,
        )

        # ---- Phase: INIT ----
        state.checkpoints.append((0, self.sandbox.baseline_sha))
        user_prompt = issue
        if self.inject_repo_map:
            repo_map = code_index.build_repo_map(
                max_chars=self.repo_map_max_chars,
                focus_terms=extract_focus_terms(issue),
            )
            user_prompt = f"{issue}\n\n{repo_map}"
        state.messages = [Message.system(self.system_prompt), Message.user(user_prompt)]
        self.tracer.start_run(state.run_id, task_id, mode)
        state.phase = Phase.RUNNING

        pending_stuck: StuckReason | None = None

        # ---- FSM 主循环 ----
        while state.phase not in (Phase.DONE, Phase.FAILED):
            if state.phase == Phase.RUNNING:
                pending_stuck = await self._step_running(state, ctx)
            elif state.phase == Phase.RECOVERING:
                await self._step_recovering(state, pending_stuck)
                pending_stuck = None
            elif state.phase == Phase.VERIFYING:
                await self._step_verifying(state)

        # ---- 收尾与落库 ----
        state.final_patch = await self.sandbox.diff()
        wall_ms = state.elapsed_s * 1000
        status = "done" if state.phase == Phase.DONE else "failed"
        self.tracer.finish_run(
            state.run_id,
            status=status,
            exit_reason=state.exit_reason,
            total_steps=state.step_no,
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            cached_tokens=state.cached_tokens,
            wall_time_ms=wall_ms,
            recover_count=state.recover_count,
            final_patch=state.final_patch,
        )
        return RunResult(
            run_id=state.run_id,
            status=status,
            exit_reason=state.exit_reason,
            total_steps=state.step_no,
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            cached_tokens=state.cached_tokens,
            wall_time_ms=wall_ms,
            recover_count=state.recover_count,
            final_summary=state.final_summary,
            final_patch=state.final_patch,
        )

    async def _step_running(self, state: AgentState, ctx: ToolContext) -> StuckReason | None:
        # ① 预算检查
        exceeded = check_budget(state, self.budget)
        if exceeded is not None:
            state.phase = Phase.FAILED
            state.exit_reason = exceeded
            return None

        state.step_no += 1
        t0 = time.perf_counter()
        step_id = self.tracer.start_step(state.run_id, state.step_no, state.phase.value)

        # ② 构建压缩后的上下文视图并调用 LLM
        view_messages, compression = await self.context_manager.build(state.messages, llm=self.llm)
        if compression != "none":
            self.tracer.update_step(step_id, compression_applied=compression)

        try:
            resp = await self.llm.chat(view_messages, tools=self.registry.schemas())
        except Exception as e:
            self.tracer.log_llm_call(step_id, messages=view_messages, error=f"{type(e).__name__}: {e}")
            self.tracer.update_step(step_id, duration_ms=(time.perf_counter() - t0) * 1000)
            state.phase = Phase.FAILED
            state.exit_reason = "llm_error"
            return None

        state.prompt_tokens += resp.usage.prompt_tokens
        state.completion_tokens += resp.usage.completion_tokens
        state.cached_tokens += resp.usage.cached_tokens
        self.tracer.log_llm_call(step_id, messages=view_messages, resp=resp)
        self.tracer.update_step(step_id, context_tokens=resp.usage.prompt_tokens)

        msg = resp.message
        state.messages.append(msg)

        # ③ 如果模型只说话不调工具：提醒它调用工具或 submit
        if not msg.tool_calls:
            state.consecutive_no_tool += 1
            self.tracer.update_step(step_id, duration_ms=(time.perf_counter() - t0) * 1000)
            if state.consecutive_no_tool >= 3:
                reason = StuckReason("no_tool_calls", "连续 3 轮未调用任何工具")
                self.tracer.update_step(step_id, guard_event=reason.code)
                state.phase = Phase.RECOVERING
                return reason
            state.messages.append(
                Message.user("请调用工具继续调查或修改代码；如果你认为已经完成修复，请调用 submit 工具提交。")
            )
            return None

        state.consecutive_no_tool = 0

        # ④ 执行本轮所有工具调用
        results = await self.registry.execute_many(msg.tool_calls, ctx)
        submitted = False

        for tc, res in zip(msg.tool_calls, results, strict=True):
            self.tracer.log_tool_call(step_id, tc.arguments, res)
            state.messages.append(res.to_message())

            # 更新连续格式错误计数
            if res.status in ("invalid_args", "unknown_tool"):
                state.consecutive_arg_errors += 1
            else:
                state.consecutive_arg_errors = 0

            # 记录动作指纹供 LoopGuard 分析
            sig = _extract_action_signature(tc.name, tc.arguments, res.content)
            state.actions.append(
                ActionRecord(
                    step_no=state.step_no,
                    tool_name=tc.name,
                    args_hash=hash_args(tc.arguments),
                    status=res.status,
                    signature=sig,
                )
            )

            # 记录 edit_file 创建的 checkpoint，并增量刷新 AST 索引
            if tc.name == "edit_file" and res.ok:
                sha = await self.sandbox.checkpoint(f"step {state.step_no}")
                state.checkpoints.append((state.step_no, sha))
                if ctx.code_index is not None and sig:
                    ctx.code_index.refresh_file(sig)

            if tc.name == "submit" and res.ok:
                submitted = True
                try:
                    state.final_summary = json.loads(tc.arguments).get("summary", "")
                except Exception:
                    state.final_summary = res.content

        # ⑤ 状态转移决策：submit → VERIFYING；否则检查 LoopGuard 与只读探索停滞提醒
        stuck: StuckReason | None = None
        if submitted:
            state.phase = Phase.VERIFYING
        else:
            stuck = self.loop_guard.check(state)
            if stuck is not None:
                self.tracer.update_step(step_id, guard_event=stuck.code)
                state.phase = Phase.RECOVERING
            else:
                # 防止“分析瘫痪”：连续多步只读不写且步数过半前仍未动手，注入确定性推进提醒
                has_edited = any(a.tool_name == "edit_file" and a.status == "ok" for a in state.actions)
                if not has_edited and state.step_no in (8, 13):
                    rem = self.budget.max_steps - state.step_no
                    self.tracer.update_step(step_id, guard_event="nudge_start_editing")
                    state.messages.append(
                        Message.user(
                            f"[进度提醒] 你已经连续只读探索了 {state.step_no} 步，尚未进行任何代码修改（仅剩 {rem} 步）。"
                            f"请停止重复读取文件，立即基于已掌握的信息用 edit_file 做出首次修复，并用目标测试文件验证。"
                        )
                    )

        self.tracer.update_step(step_id, duration_ms=(time.perf_counter() - t0) * 1000)
        return stuck

    async def _step_recovering(self, state: AgentState, reason: StuckReason | None) -> None:
        t0 = time.perf_counter()
        step_id = self.tracer.start_step(state.run_id, state.step_no, Phase.RECOVERING.value)
        reason = reason or StuckReason("unknown_stuck", "检测到执行停滞")

        if state.recover_count >= self.budget.max_recovers:
            self.tracer.update_step(
                step_id, guard_event=f"recover_exhausted:{reason.code}", duration_ms=(time.perf_counter() - t0) * 1000
            )
            state.phase = Phase.FAILED
            state.exit_reason = "stuck"
            return

        state.recover_count += 1
        # 回滚到基线（确保从干净状态重新出发）
        rollback_sha = self.sandbox.baseline_sha
        await self.sandbox.rollback(rollback_sha)

        # 清空易触发重复告警的近期计数，给模型一次重新出发的机会
        state.actions.clear()
        state.consecutive_no_tool = 0
        state.consecutive_arg_errors = 0

        state.messages.append(Message.user(LoopGuard.build_recovery_prompt(reason, rollback_sha)))
        self.tracer.update_step(
            step_id, guard_event=f"recovered:{reason.code}", duration_ms=(time.perf_counter() - t0) * 1000
        )
        state.phase = Phase.RUNNING

    async def _step_verifying(self, state: AgentState) -> None:
        t0 = time.perf_counter()
        step_id = self.tracer.start_step(state.run_id, state.step_no, Phase.VERIFYING.value)
        v = await self.verifier.verify(self.sandbox)

        if v.passed:
            self.tracer.update_step(step_id, guard_event="verify_passed", duration_ms=(time.perf_counter() - t0) * 1000)
            state.phase = Phase.DONE
            state.exit_reason = "verified"
            return

        state.verify_failures += 1
        self.tracer.update_step(
            step_id, guard_event=f"verify_failed_{state.verify_failures}", duration_ms=(time.perf_counter() - t0) * 1000
        )
        if state.verify_failures > self.budget.max_verify_retries:
            state.phase = Phase.FAILED
            state.exit_reason = "verify_failed"
            return

        # 确定性裁决失败 → 把真实测试报错回填给模型，打回 RUNNING 继续修
        state.messages.append(
            Message.user(
                f"[确定性验证未通过（第 {state.verify_failures}/{self.budget.max_verify_retries} 次重试机会）]\n"
                f"你调用了 submit，但系统运行完整测试套件后发现仍有失败：\n{v.message}\n"
                f"请定位问题并继续修复，通过后再调用 submit。"
            )
        )
        state.phase = Phase.RUNNING


def _extract_action_signature(tool_name: str, args_json: str, output: str) -> str:
    if tool_name == "run_tests":
        return extract_test_failure_signature(output)
    if tool_name == "edit_file":
        try:
            return str(json.loads(args_json).get("path", ""))
        except Exception:
            return ""
    return ""
