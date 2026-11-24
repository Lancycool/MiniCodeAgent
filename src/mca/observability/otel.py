"""OpenTelemetry 标准分布式追踪导出器（遵循 GenAI Semantic Conventions）。

Span 父子层级结构：
  agent.run (trace_id)
  └── agent.step (phase=RUNNING / RECOVERING / VERIFYING)
      ├── gen_ai.chat  (gen_ai.request.model, gen_ai.usage.input_tokens, ...)
      └── tool.execute (mca.tool.name, mca.tool.status, mca.tool.raw_chars, ...)

支持三种模式（通过 --otel 或环境变量 MCA_OTEL_EXPORTER 控制）：
- none    : 关闭（默认，零开销）
- console : 导出到终端 JSON（本地调试）
- otlp    : 通过 OTLP/HTTP 导出到 Jaeger / Grafana Tempo / Arize Phoenix / Langfuse
            （默认地址 http://localhost:4318/v1/traces，可用 MCA_OTEL_ENDPOINT 覆盖）
"""

from __future__ import annotations

import os
import time
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Status, StatusCode, set_span_in_context

from mca.llm.types import LLMResponse
from mca.tools.base import ToolResult


class OTelSink:
    """将 Agent 运行事件实时映射为标准 OpenTelemetry 父子 Span 树。"""

    def __init__(
        self,
        exporter: SpanExporter,
        *,
        service_name: str = "minicodeagent",
        use_batch: bool = False,
    ) -> None:
        self._provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
        processor = BatchSpanProcessor(exporter) if use_batch else SimpleSpanProcessor(exporter)
        self._provider.add_span_processor(processor)
        self._tracer = self._provider.get_tracer("mca.agent")
        self._run_spans: dict[str, Span] = {}
        self._step_spans: dict[int, Span] = {}

    def start_run(self, run_id: str, task_id: str, mode: str) -> None:
        span = self._tracer.start_span(
            "agent.run",
            attributes={
                "mca.run_id": run_id,
                "mca.task_id": task_id,
                "mca.mode": mode,
            },
        )
        self._run_spans[run_id] = span

    def finish_run(self, run_id: str, **kwargs: Any) -> None:
        # 收尾未关闭的 step span（防御性清理）
        for sid in list(self._step_spans):
            self._step_spans.pop(sid).end()

        span = self._run_spans.pop(run_id, None)
        if span is None:
            return
        status = str(kwargs.get("status", ""))
        exit_reason = str(kwargs.get("exit_reason", ""))
        span.set_attribute("mca.status", status)
        span.set_attribute("mca.exit_reason", exit_reason)
        span.set_attribute("mca.total_steps", int(kwargs.get("total_steps", 0)))
        span.set_attribute("gen_ai.usage.input_tokens", int(kwargs.get("prompt_tokens", 0)))
        span.set_attribute("gen_ai.usage.output_tokens", int(kwargs.get("completion_tokens", 0)))
        span.set_attribute("mca.recover_count", int(kwargs.get("recover_count", 0)))
        if status == "failed":
            span.set_status(Status(StatusCode.ERROR, exit_reason))
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()
        self._provider.force_flush(timeout_millis=3000)

    def start_step(self, run_id: str, step_key: int, step_no: int, phase: str) -> None:
        parent = self._run_spans.get(run_id)
        ctx = set_span_in_context(parent) if parent is not None else None
        span = self._tracer.start_span(
            f"agent.step.{step_no}",
            context=ctx,
            attributes={
                "mca.run_id": run_id,
                "mca.step_no": step_no,
                "mca.phase": phase,
            },
        )
        self._step_spans[step_key] = span

    def update_step(self, step_key: int, **kwargs: Any) -> None:
        span = self._step_spans.get(step_key)
        if span is None:
            return
        if kwargs.get("context_tokens") is not None:
            span.set_attribute("mca.context_tokens", int(kwargs["context_tokens"]))
        if kwargs.get("compression_applied") is not None:
            span.set_attribute("mca.compression", str(kwargs["compression_applied"]))
        if kwargs.get("guard_event") is not None:
            span.set_attribute("mca.guard_event", str(kwargs["guard_event"]))
        # duration_ms 仅在步骤结束时传入，此时结束该 step span
        if kwargs.get("duration_ms") is not None:
            self._step_spans.pop(step_key, None)
            span.end()

    def log_llm_call(self, step_key: int, *, resp: LLMResponse | None = None, error: str | None = None) -> None:
        parent = self._step_spans.get(step_key)
        ctx = set_span_in_context(parent) if parent is not None else None
        end_ns = time.time_ns()
        start_ns = end_ns - int((resp.latency_ms if resp else 0.0) * 1_000_000)
        span = self._tracer.start_span("gen_ai.chat", context=ctx, start_time=start_ns)
        if resp is not None:
            span.set_attribute("gen_ai.request.model", resp.model)
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.usage.completion_tokens)
            span.set_attribute("gen_ai.usage.cached_tokens", resp.usage.cached_tokens)
            span.set_attribute("mca.llm.attempts", resp.attempts)
            if resp.finish_reason:
                span.set_attribute("gen_ai.response.finish_reasons", resp.finish_reason)
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR, error or "llm_error"))
        span.end(end_time=end_ns)

    def log_tool_call(self, step_key: int, result: ToolResult) -> None:
        parent = self._step_spans.get(step_key)
        ctx = set_span_in_context(parent) if parent is not None else None
        end_ns = time.time_ns()
        start_ns = end_ns - int(result.latency_ms * 1_000_000)
        span = self._tracer.start_span(
            f"tool.{result.name}",
            context=ctx,
            start_time=start_ns,
            attributes={
                "mca.tool.name": result.name,
                "mca.tool.call_id": result.call_id,
                "mca.tool.status": result.status,
                "mca.tool.raw_chars": result.raw_chars,
                "mca.tool.truncated": result.truncated,
                "mca.tool.redactions": result.redactions,
            },
        )
        if not result.ok:
            span.set_status(Status(StatusCode.ERROR, result.status))
        else:
            span.set_status(Status(StatusCode.OK))
        span.end(end_time=end_ns)

    def shutdown(self) -> None:
        self._provider.shutdown()


def build_otel_sink(mode: str | None = None, *, endpoint: str | None = None) -> OTelSink | None:
    """按参数或环境变量（MCA_OTEL_EXPORTER / MCA_OTEL_ENDPOINT）创建 OTelSink。"""
    chosen = (mode or os.environ.get("MCA_OTEL_EXPORTER", "none")).strip().lower()
    if chosen in ("", "none", "off", "false"):
        return None
    if chosen == "console":
        return OTelSink(ConsoleSpanExporter(), use_batch=False)
    if chosen == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        ep = endpoint or os.environ.get("MCA_OTEL_ENDPOINT", "http://localhost:4318/v1/traces")
        return OTelSink(OTLPSpanExporter(endpoint=ep), use_batch=True)
    raise ValueError(f"Unsupported OTel exporter mode: {chosen!r} (expected: none | console | otlp)")
