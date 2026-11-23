"""AgentLoop 状态机与 LoopGuard 离线单测（用 FakeLLMClient 做到 100% 确定、0 成本）。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from mca.agent import AgentLoop, Budget, LoopGuard
from mca.llm.types import LLMResponse, Message, ToolCall, Usage
from mca.observability import TraceStore, Tracer
from mca.sandbox import LocalSandbox
from mca.tools import ToolRegistry, fix_tools


class FakeLLMClient:
    """按脚本依次吐出预设的 LLMResponse，用于精确测试各种状态机分支。"""

    def __init__(self, script: list[Message]) -> None:
        self._script = list(script)
        self.calls: int = 0

    async def chat(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        msg = self._script.pop(0) if self._script else Message(role="assistant", content="done")
        return LLMResponse(
            message=msg,
            finish_reason="tool_calls" if msg.tool_calls else "stop",
            usage=Usage(prompt_tokens=100, completion_tokens=20, cached_tokens=50),
            model="fake-model",
            latency_ms=5.0,
            attempts=1,
        )


def tc_msg(name: str, args: str, cid: str = "c1") -> Message:
    return Message(role="assistant", content="", tool_calls=[ToolCall(id=cid, name=name, arguments=args)])


@pytest.fixture
def buggy_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "math_util.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (repo / "test_math.py").write_text(
        "from math_util import add\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    return repo


async def test_happy_path_and_trace_persisted(buggy_repo: Path):
    """正常路径：修改 → submit → VERIFYING 通过 → DONE，且全链路落库 SQLite。"""
    fake_llm = FakeLLMClient([
        tc_msg("edit_file", '{"path": "math_util.py", "old_str": "return a - b", "new_str": "return a + b"}', "c1"),
        tc_msg("submit", '{"summary": "fixed add"}', "c2"),
    ])
    store = TraceStore(":memory:")
    async with LocalSandbox(buggy_repo) as sb:
        loop = AgentLoop(
            llm=fake_llm,
            registry=ToolRegistry(fix_tools()),
            sandbox=sb,
            tracer=Tracer(store),
        )
        res = await loop.run("修复 add 函数", task_id="t1")

    assert res.ok and res.exit_reason == "verified" and res.total_steps == 2
    assert "+    return a + b" in res.final_patch

    trace = store.get_run_trace(res.run_id)
    assert trace is not None and trace["status"] == "done" and len(trace["steps"]) == 3  # 2 RUNNING + 1 VERIFYING
    assert any(s["guard_event"] == "verify_passed" for s in trace["steps"])
    store.close()


async def test_submit_rejected_by_verifier_then_fixed(buggy_repo: Path):
    """模型虚报“已修复”调用 submit → 确定性 Verifier 打回 → 模型真正修复后通过。"""
    fake_llm = FakeLLMClient([
        tc_msg("submit", '{"summary": "我觉得修好了"}', "c1"),  # 没改就提交！
        tc_msg("edit_file", '{"path": "math_util.py", "old_str": "return a - b", "new_str": "return a + b"}', "c2"),
        tc_msg("submit", '{"summary": "这次真的修好了"}', "c3"),
    ])
    store = TraceStore(":memory:")
    async with LocalSandbox(buggy_repo) as sb:
        loop = AgentLoop(llm=fake_llm, registry=ToolRegistry(fix_tools()), sandbox=sb, tracer=Tracer(store))
        res = await loop.run("修复 add", task_id="t_lie")

    assert res.ok and res.exit_reason == "verified" and res.total_steps == 3
    trace = store.get_run_trace(res.run_id)
    events = [s["guard_event"] for s in trace["steps"] if s["guard_event"]]
    assert "verify_failed_1" in events and "verify_passed" in events
    store.close()


async def test_loop_guard_triggers_rollback_and_recovers(buggy_repo: Path):
    """模型反复做相同无用调用 → LoopGuard 熔断 → 回滚坏改动并注入反思 → 恢复后修复成功。"""
    fake_llm = FakeLLMClient([
        tc_msg("edit_file", '{"path": "math_util.py", "old_str": "return a - b", "new_str": "return a * b"}', "c1"),
        tc_msg("read_file", '{"path": "math_util.py"}', "c2"),
        tc_msg("read_file", '{"path": "math_util.py"}', "c3"),
        tc_msg("read_file", '{"path": "math_util.py"}', "c4"),  # 第 3 次重复 → 触发 LoopGuard，回滚到基线！
        # 回滚后文件内容已恢复为 "return a - b"，因此这里的 old_str 能精确匹配
        tc_msg("edit_file", '{"path": "math_util.py", "old_str": "return a - b", "new_str": "return a + b"}', "c5"),
        tc_msg("submit", '{"summary": "回滚后重新修复"}', "c6"),
    ])
    async with LocalSandbox(buggy_repo) as sb:
        loop = AgentLoop(
            llm=fake_llm,
            registry=ToolRegistry(fix_tools()),
            sandbox=sb,
            loop_guard=LoopGuard(max_same_call=3),
        )
        res = await loop.run("修复 add", task_id="t_loop")

    assert res.ok and res.recover_count == 1 and res.exit_reason == "verified"


async def test_loop_guard_exhausted_fails_as_stuck(buggy_repo: Path):
    """超过 max_recovers 后仍反复卡死 → FAILED(exit_reason='stuck')。"""
    fake_llm = FakeLLMClient([tc_msg("read_file", '{"path": "math_util.py"}', f"c{i}") for i in range(10)])
    async with LocalSandbox(buggy_repo) as sb:
        loop = AgentLoop(
            llm=fake_llm,
            registry=ToolRegistry(fix_tools()),
            sandbox=sb,
            budget=Budget(max_steps=20, max_recovers=1),
            loop_guard=LoopGuard(max_same_call=3),
        )
        res = await loop.run("修复 add", task_id="t_stuck")

    assert not res.ok and res.exit_reason == "stuck" and res.recover_count == 1


async def test_budget_steps_stops_run(buggy_repo: Path):
    fake_llm = FakeLLMClient([tc_msg("list_dir", "{}", f"c{i}") for i in range(5)])
    async with LocalSandbox(buggy_repo) as sb:
        loop = AgentLoop(
            llm=fake_llm,
            registry=ToolRegistry(fix_tools()),
            sandbox=sb,
            budget=Budget(max_steps=2),
            loop_guard=LoopGuard(enabled=False),
        )
        res = await loop.run("修复 add", task_id="t_budget")

    assert not res.ok and res.exit_reason == "budget_steps" and res.total_steps == 2
