"""LLM 客户端：协议（接口）+ OpenAI 兼容实现。"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, Protocol

from openai import AsyncOpenAI

from mca.config import LLMSettings
from mca.llm.retry import LLMError, RetryPolicy, call_with_retry
from mca.llm.types import LLMResponse, Message, ToolCall, Usage

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Agent Core 只依赖这个接口。测试时可以用一个返回固定响应的 FakeClient 替代。"""

    async def chat(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse: ...


class OpenAICompatClient:
    """适用于任何 OpenAI 兼容接口（DeepSeek / Qwen / GLM / OpenAI ...）。"""

    def __init__(self, settings: LLMSettings, policy: RetryPolicy | None = None) -> None:
        self._settings = settings
        self._policy = policy or RetryPolicy(
            max_attempts=settings.max_attempts,
            base_delay_s=settings.retry_base_delay_s,
            max_delay_s=settings.retry_max_delay_s,
        )
        # max_retries=0：关闭 SDK 内置重试，统一由我们的 call_with_retry 负责。
        # 否则两层重试叠加，实际请求次数 = 5 x 3，既难以预估成本也难以在 Trace 中观察。
        self._client = AsyncOpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
            timeout=settings.timeout_s,
            max_retries=0,
        )

    async def chat(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        s = self._settings
        payload: dict[str, Any] = {
            "model": s.model,
            "messages": [m.to_openai(include_reasoning=s.send_reasoning_content) for m in messages],
            "temperature": s.temperature,
            "max_tokens": s.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        start = time.perf_counter()
        resp, attempts = await call_with_retry(
            lambda: self._client.chat.completions.create(**payload),
            self._policy,
            on_retry=self._log_retry,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return self._parse(resp, latency_ms=latency_ms, attempts=attempts)

    @staticmethod
    def _log_retry(attempt: int, exc: BaseException, delay: float) -> None:
        logger.warning(
            "LLM call failed (attempt %d): %s: %s; retrying in %.2fs",
            attempt, type(exc).__name__, exc, delay,
        )

    @staticmethod
    def _parse(resp: Any, *, latency_ms: float, attempts: int) -> LLMResponse:
        """把 SDK 响应对象转换成内部类型。只依赖鸭子类型，便于用假对象做单测。"""
        if not getattr(resp, "choices", None):
            raise LLMError(f"LLM returned no choices: {resp!r}")
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments or "{}",
            )
            for tc in (msg.tool_calls or [])
        ]

        usage = Usage()
        if resp.usage is not None:
            usage.prompt_tokens = resp.usage.prompt_tokens or 0
            usage.completion_tokens = resp.usage.completion_tokens or 0
            # DeepSeek 用 prompt_cache_hit_tokens；OpenAI 用 prompt_tokens_details.cached_tokens
            cached = getattr(resp.usage, "prompt_cache_hit_tokens", None)
            if cached is None:
                details = getattr(resp.usage, "prompt_tokens_details", None)
                cached = getattr(details, "cached_tokens", None)
            usage.cached_tokens = cached or 0

        return LLMResponse(
            message=Message(
                role="assistant",
                content=msg.content,
                tool_calls=tool_calls,
                reasoning_content=getattr(msg, "reasoning_content", None),
            ),
            finish_reason=choice.finish_reason,
            usage=usage,
            model=getattr(resp, "model", "") or "",
            latency_ms=latency_ms,
            attempts=attempts,
        )
