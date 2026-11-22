"""网络层重试：错误分类 + 指数退避 + 全抖动（Full Jitter）。

这是"分层重试"中的最底层（见 docs/03 §6.1）：只处理**与模型输出内容无关**的
传输/服务端错误。模型输出不合法、工具执行失败等，由更上层处理。
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import openai

T = TypeVar("T")


class LLMError(Exception):
    """LLM 层的非重试型错误（如返回结构异常）。"""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0

    def backoff(self, attempt: int, rng: Callable[[], float] = random.random) -> float:
        """第 attempt 次失败后的等待时间（attempt 从 1 开始）。

        Full Jitter: sleep = U(0, min(cap, base * 2^(attempt-1)))
        - 指数增长：给过载的服务端恢复时间
        - 随机抖动：避免大量客户端在同一时刻重试造成"惊群"（thundering herd）
        """
        ceiling = min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))
        return rng() * ceiling


_RETRYABLE_STATUS = {408, 409, 429}


def is_retryable(exc: BaseException) -> bool:
    """判断一个异常是否值得重试。

    可重试：超时、连接断开、429 限流、5xx 服务端错误 —— 稍后再试有可能成功。
    不可重试：400 参数错误、401 鉴权失败、404 模型不存在 …… 重试多少次结果都一样，
    只会浪费时间和额度，应立即失败并暴露问题。
    """
    # APITimeoutError 是 APIConnectionError 的子类
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS or exc.status_code >= 500
    return False


def retry_after_seconds(exc: BaseException) -> float | None:
    """读取服务端的 Retry-After 响应头（秒）。服务端明确告诉你等多久时应当尊重它。"""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:  # 也可能是 HTTP 日期格式，这里简单忽略
        return None


OnRetry = Callable[[int, BaseException, float], None]


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    retryable: Callable[[BaseException], bool] = is_retryable,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: OnRetry | None = None,
) -> tuple[T, int]:
    """执行 fn，按策略重试。返回 (结果, 实际尝试次数)。

    sleep / retryable 通过参数注入 —— 单元测试时传入假的 sleep，
    就能在 0 秒内验证"重试了几次、每次等多久"，这就是依赖注入带来的可测试性。
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn(), attempt
        except Exception as exc:  # noqa: BLE001 —— 分类后决定是否重抛
            if attempt >= policy.max_attempts or not retryable(exc):
                raise
            delay = policy.backoff(attempt)
            hinted = retry_after_seconds(exc)
            if hinted is not None:
                delay = min(policy.max_delay_s, max(delay, hinted))
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            await sleep(delay)
