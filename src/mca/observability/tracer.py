"""Tracer：容错的可观测性记录器。

核心工程原则：**可观测性绝不能成为故障源**。
即使 SQLite 写入偶尔报错（如磁盘满、锁冲突），Tracer 只记录 warning，不中断 Agent 主流程。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mca.llm.types import LLMResponse, Message
from mca.observability.store import TraceStore
from mca.tools.base import ToolResult

if TYPE_CHECKING:
    from mca.observability.otel import OTelSink

logger = logging.getLogger(__name__)


class Tracer:
    """支持 SQLite + OpenTelemetry 双写的容错追踪器。"""

    def __init__(self, store: TraceStore | None = None, otel: OTelSink | None = None) -> None:
        self.store = store
        self.otel = otel
        self._synthetic_step_seq = 0

    def start_run(self, run_id: str, task_id: str, mode: str) -> None:
        if self.store is not None:
            try:
                self.store.create_run(run_id, task_id, mode)
            except Exception:
                logger.warning("Tracer.start_run (sqlite) failed", exc_info=True)
        if self.otel is not None:
            try:
                self.otel.start_run(run_id, task_id, mode)
            except Exception:
                logger.warning("Tracer.start_run (otel) failed", exc_info=True)

    def finish_run(self, run_id: str, **kwargs: Any) -> None:
        if self.store is not None:
            try:
                self.store.finish_run(run_id, **kwargs)
            except Exception:
                logger.warning("Tracer.finish_run (sqlite) failed", exc_info=True)
        if self.otel is not None:
            try:
                self.otel.finish_run(run_id, **kwargs)
            except Exception:
                logger.warning("Tracer.finish_run (otel) failed", exc_info=True)

    def start_step(self, run_id: str, step_no: int, phase: str) -> int | None:
        step_id: int | None = None
        if self.store is not None:
            try:
                step_id = self.store.add_step(run_id, step_no, phase)
            except Exception:
                logger.warning("Tracer.start_step (sqlite) failed", exc_info=True)
        if self.otel is not None:
            if step_id is None:
                self._synthetic_step_seq -= 1
                step_id = self._synthetic_step_seq
            try:
                self.otel.start_step(run_id, step_id, step_no, phase)
            except Exception:
                logger.warning("Tracer.start_step (otel) failed", exc_info=True)
        return step_id

    def update_step(self, step_id: int | None, **kwargs: Any) -> None:
        if step_id is None:
            return
        if self.store is not None and step_id > 0:
            try:
                self.store.update_step(step_id, **kwargs)
            except Exception:
                logger.warning("Tracer.update_step (sqlite) failed", exc_info=True)
        if self.otel is not None:
            try:
                self.otel.update_step(step_id, **kwargs)
            except Exception:
                logger.warning("Tracer.update_step (otel) failed", exc_info=True)

    def log_llm_call(
        self,
        step_id: int | None,
        *,
        messages: list[Message],
        resp: LLMResponse | None = None,
        error: str | None = None,
    ) -> None:
        if step_id is None:
            return
        if self.store is not None and step_id > 0:
            try:
                if resp is not None:
                    self.store.add_llm_call(
                        step_id,
                        model=resp.model,
                        prompt_tokens=resp.usage.prompt_tokens,
                        completion_tokens=resp.usage.completion_tokens,
                        cached_tokens=resp.usage.cached_tokens,
                        latency_ms=resp.latency_ms,
                        attempts=resp.attempts,
                        finish_reason=resp.finish_reason,
                        request_obj=[m.to_openai() for m in messages],
                        response_obj=resp.message.to_openai(),
                    )
                else:
                    self.store.add_llm_call(
                        step_id,
                        model="",
                        prompt_tokens=0,
                        completion_tokens=0,
                        cached_tokens=0,
                        latency_ms=0.0,
                        attempts=1,
                        error=error,
                        request_obj=[m.to_openai() for m in messages],
                    )
            except Exception:
                logger.warning("Tracer.log_llm_call (sqlite) failed", exc_info=True)
        if self.otel is not None:
            try:
                self.otel.log_llm_call(step_id, resp=resp, error=error)
            except Exception:
                logger.warning("Tracer.log_llm_call (otel) failed", exc_info=True)

    def log_tool_call(self, step_id: int | None, args_json: str, result: ToolResult) -> None:
        if step_id is None:
            return
        if self.store is not None and step_id > 0:
            try:
                self.store.add_tool_call(
                    step_id,
                    call_id=result.call_id,
                    tool_name=result.name,
                    args_json=args_json,
                    output_truncated=result.content,
                    raw_chars=result.raw_chars,
                    truncated=result.truncated,
                    redactions=result.redactions,
                    status=result.status,
                    latency_ms=result.latency_ms,
                )
            except Exception:
                logger.warning("Tracer.log_tool_call (sqlite) failed", exc_info=True)
        if self.otel is not None:
            try:
                self.otel.log_tool_call(step_id, result)
            except Exception:
                logger.warning("Tracer.log_tool_call (otel) failed", exc_info=True)
