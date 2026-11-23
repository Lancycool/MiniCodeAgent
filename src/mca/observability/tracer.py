"""Tracer：容错的可观测性记录器。

核心工程原则：**可观测性绝不能成为故障源**。
即使 SQLite 写入偶尔报错（如磁盘满、锁冲突），Tracer 只记录 warning，不中断 Agent 主流程。
"""

from __future__ import annotations

import logging
from typing import Any

from mca.llm.types import LLMResponse, Message
from mca.observability.store import TraceStore
from mca.tools.base import ToolResult

logger = logging.getLogger(__name__)


class Tracer:
    def __init__(self, store: TraceStore | None = None) -> None:
        self.store = store

    def start_run(self, run_id: str, task_id: str, mode: str) -> None:
        if self.store is None:
            return
        try:
            self.store.create_run(run_id, task_id, mode)
        except Exception:
            logger.warning("Tracer.start_run failed", exc_info=True)

    def finish_run(self, run_id: str, **kwargs: Any) -> None:
        if self.store is None:
            return
        try:
            self.store.finish_run(run_id, **kwargs)
        except Exception:
            logger.warning("Tracer.finish_run failed", exc_info=True)

    def start_step(self, run_id: str, step_no: int, phase: str) -> int | None:
        if self.store is None:
            return None
        try:
            return self.store.add_step(run_id, step_no, phase)
        except Exception:
            logger.warning("Tracer.start_step failed", exc_info=True)
            return None

    def update_step(self, step_id: int | None, **kwargs: Any) -> None:
        if self.store is None or step_id is None:
            return
        try:
            self.store.update_step(step_id, **kwargs)
        except Exception:
            logger.warning("Tracer.update_step failed", exc_info=True)

    def log_llm_call(
        self,
        step_id: int | None,
        *,
        messages: list[Message],
        resp: LLMResponse | None = None,
        error: str | None = None,
    ) -> None:
        if self.store is None or step_id is None:
            return
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
            logger.warning("Tracer.log_llm_call failed", exc_info=True)

    def log_tool_call(self, step_id: int | None, args_json: str, result: ToolResult) -> None:
        if self.store is None or step_id is None:
            return
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
            logger.warning("Tracer.log_tool_call failed", exc_info=True)
