"""重试逻辑单测：不发真实网络请求，用假 sleep 在 0 秒内验证行为。"""

from __future__ import annotations

from types import SimpleNamespace as NS

import openai
import pytest

from mca.llm.retry import RetryPolicy, call_with_retry, is_retryable, retry_after_seconds

# 不直接 import SDK 底层的 HTTP 库（openai 2.x 用 httpx，3.x 换成了 httpx2）。
# SDK 的异常只用到 request / status_code / headers.get，用鸭子类型的假对象即可，
# 测试因此不再依赖 SDK 的内部实现细节。
REQ = NS(method="POST", url="https://example.com/v1/chat/completions")


def status_error(cls: type[openai.APIStatusError], code: int, headers: dict | None = None):
    resp = NS(status_code=code, request=REQ, headers={k.lower(): v for k, v in (headers or {}).items()})
    return cls(f"HTTP {code}", response=resp, body=None)


class FakeSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


# ---------- 错误分类 ----------

@pytest.mark.parametrize(
    "exc, expected",
    [
        (openai.APITimeoutError(request=REQ), True),
        (openai.APIConnectionError(request=REQ), True),
        (status_error(openai.RateLimitError, 429), True),
        (status_error(openai.InternalServerError, 503), True),
        (status_error(openai.BadRequestError, 400), False),
        (status_error(openai.AuthenticationError, 401), False),
        (status_error(openai.NotFoundError, 404), False),
        (ValueError("bug in our code"), False),
    ],
)
def test_is_retryable(exc, expected):
    assert is_retryable(exc) is expected


def test_retry_after_header():
    exc = status_error(openai.RateLimitError, 429, {"retry-after": "7"})
    assert retry_after_seconds(exc) == 7.0
    assert retry_after_seconds(ValueError()) is None


# ---------- 退避计算 ----------

def test_backoff_exponential_and_capped():
    policy = RetryPolicy(max_attempts=10, base_delay_s=1.0, max_delay_s=10.0)
    ceilings = [policy.backoff(i, rng=lambda: 1.0) for i in range(1, 7)]
    assert ceilings == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0]
    assert policy.backoff(3, rng=lambda: 0.5) == 2.0  # 抖动落在 [0, ceiling)


# ---------- 重试流程 ----------

async def test_succeeds_after_transient_failures():
    sleep = FakeSleep()
    errors = [openai.APITimeoutError(request=REQ), status_error(openai.RateLimitError, 429)]

    async def fn():
        if errors:
            raise errors.pop(0)
        return "ok"

    result, attempts = await call_with_retry(fn, RetryPolicy(max_attempts=5), sleep=sleep)
    assert (result, attempts) == ("ok", 3)
    assert len(sleep.calls) == 2


async def test_non_retryable_raises_immediately():
    sleep = FakeSleep()
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise status_error(openai.BadRequestError, 400)

    with pytest.raises(openai.BadRequestError):
        await call_with_retry(fn, RetryPolicy(max_attempts=5), sleep=sleep)
    assert calls == 1 and sleep.calls == []


async def test_gives_up_after_max_attempts():
    sleep = FakeSleep()
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise status_error(openai.InternalServerError, 500)

    with pytest.raises(openai.InternalServerError):
        await call_with_retry(fn, RetryPolicy(max_attempts=3), sleep=sleep)
    assert calls == 3 and len(sleep.calls) == 2


async def test_retry_after_is_respected_but_capped():
    sleep = FakeSleep()
    errors = [status_error(openai.RateLimitError, 429, {"retry-after": "100"})]

    async def fn():
        if errors:
            raise errors.pop(0)
        return "ok"

    await call_with_retry(fn, RetryPolicy(max_attempts=3, max_delay_s=20.0), sleep=sleep)
    assert sleep.calls == [20.0]  # 服务端要求 100s，但被 max_delay 封顶
