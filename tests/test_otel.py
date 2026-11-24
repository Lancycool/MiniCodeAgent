"""OpenTelemetry 双写追踪单测：用 InMemorySpanExporter 离线验证完整父子 Span 树与 GenAI 属性。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mca.agent import AgentLoop
from mca.llm.types import LLMResponse, Message, ToolCall, Usage
from mca.observability import OTelSink, TraceStore, Tracer
from mca.sandbox import LocalSandbox
from mca.tools import ToolRegistry, fix_tools


class OneShotFixLLM:
    def __init__(self) -> None:
        self._n = 0

    async def chat(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self._n += 1
        tc = (
            ToolCall("c1", "edit_file", '{"path":"a.py","old_str":"return a - b","new_str":"return a + b"}')
            if self._n == 1
            else ToolCall("c2", "submit", '{"summary":"fixed"}')
        )
        return LLMResponse(
            message=Message(role="assistant", content="", tool_calls=[tc]),
            finish_reason="tool_calls",
            usage=Usage(prompt_tokens=150, completion_tokens=30, cached_tokens=64),
            model="deepseek-v4.1-flash",
            latency_ms=12.0,
            attempts=1,
        )


async def test_otel_dual_write_span_hierarchy(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (repo / "test_a.py").write_text("from a import add\ndef test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8")

    exporter = InMemorySpanExporter()
    otel = OTelSink(exporter, use_batch=False)
    store = TraceStore(":memory:")

    async with LocalSandbox(repo) as sb:
        loop = AgentLoop(
            llm=OneShotFixLLM(),
            registry=ToolRegistry(fix_tools()),
            sandbox=sb,
            tracer=Tracer(store, otel=otel),
        )
        res = await loop.run("fix add", task_id="otel_test")

    spans = exporter.get_finished_spans()
    otel.shutdown()
    store.close()

    assert res.ok
    by_name = {s.name: s for s in spans}
    assert "agent.run" in by_name
    assert "agent.step.1" in by_name
    assert "gen_ai.chat" in by_name
    assert "tool.edit_file" in by_name

    # 所有 Span 共享同一个 trace_id
    run_span = by_name["agent.run"]
    trace_ids = {s.context.trace_id for s in spans}
    assert trace_ids == {run_span.context.trace_id}

    # 父子层级正确：agent.step.1 的父节点是 agent.run；tool.edit_file 的父节点是 agent.step.1
    step1 = by_name["agent.step.1"]
    edit_span = by_name["tool.edit_file"]
    assert step1.parent.span_id == run_span.context.span_id
    assert edit_span.parent.span_id == step1.context.span_id

    # 校验 GenAI 语义约定属性
    assert run_span.attributes["mca.exit_reason"] == "verified"
    assert run_span.attributes["gen_ai.usage.input_tokens"] == 300
    assert edit_span.attributes["mca.tool.status"] == "ok"
