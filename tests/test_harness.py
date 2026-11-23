"""评测 Harness 与 CLI 离线单测：
1. 校验 MiniBench 每道题在未修复时确实会触发隐藏测试失败（防“假题”）；
2. 用 FakeLLMClient 跑通 run_benchmark 并发消融矩阵并验证 Markdown 报告输出；
3. 验证 CLI 参数解析。
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mca.cli import build_parser
from mca.harness import PRESET_CONFIGS, builtin_tasks, render_ablation_markdown, run_benchmark
from mca.llm.types import LLMResponse, Message, ToolCall, Usage
from mca.sandbox import LocalSandbox


class FixingFakeLLM:
    """专门修复 t01_cart_boundary 的假模型，用于离线验证 Harness 端到端链路。"""

    def __init__(self) -> None:
        self._step = 0

    async def chat(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self._step += 1
        if self._step == 1:
            tc = ToolCall(
                "c1",
                "edit_file",
                '{"path": "cart.py", "old_str": "if subtotal > 100:\\n        subtotal *= 0.9\\n    return round(subtotal - coupon, 2)", '
                '"new_str": "if subtotal >= 100:\\n        subtotal *= 0.9\\n    return round(max(0.0, subtotal - coupon), 2)"}',
            )
            msg = Message(role="assistant", content="", tool_calls=[tc])
        else:
            msg = Message(role="assistant", content="", tool_calls=[ToolCall("c2", "submit", '{"summary": "fixed"}')])
        return LLMResponse(
            message=msg,
            finish_reason="tool_calls",
            usage=Usage(prompt_tokens=200, completion_tokens=40),
            model="fake",
            latency_ms=2.0,
            attempts=1,
        )


async def test_all_minibench_tasks_fail_before_fix():
    """Sanity check：确认每道题的隐藏测试在初始有 Bug 的代码上全部失败。"""
    tasks = builtin_tasks()
    assert len(tasks) >= 6
    for t in tasks:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            t.materialize(p, include_hidden=True)
            async with LocalSandbox(p) as sb:
                res = await sb.exec(f"{sb.python_bin} -m pytest -q --tb=no -p no:cacheprovider", timeout_s=30.0)
                assert res.exit_code != 0, f"Task {t.id} should fail before fix!"


async def test_run_benchmark_ablation_matrix(tmp_path: Path):
    tasks = [builtin_tasks()[0]]  # t01_cart_boundary
    configs = [PRESET_CONFIGS["full"], PRESET_CONFIGS["no_ast"]]
    results = await run_benchmark(
        tasks,
        configs,
        llm_factory=FixingFakeLLM,
        concurrency=2,
        use_docker=False,
        db_path=tmp_path / "bench.db",
    )
    assert len(results) == 2
    assert all(r.hidden_passed for r in results)

    md = render_ablation_markdown(results)
    assert "`full`" in md and "`no_ast`" in md and "100.0%" in md


def test_cli_parser_subcommands():
    p = build_parser()
    assert p.parse_args(["fix", ".", "-i", "bug"]).command == "fix"
    assert p.parse_args(["review", ".", "--refactor"]).refactor is True
    assert p.parse_args(["trace", "stats"]).trace_cmd == "stats"
    assert p.parse_args(["bench", "--configs", "full,no_ast", "--limit", "2"]).limit == 2
