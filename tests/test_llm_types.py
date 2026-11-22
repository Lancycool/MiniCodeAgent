"""消息协议转换与响应解析单测（用 SimpleNamespace 伪造 SDK 响应）。"""

from __future__ import annotations

from types import SimpleNamespace as NS

from mca.llm.client import OpenAICompatClient
from mca.llm.types import Message, ToolCall


def test_assistant_with_tool_calls_to_openai():
    m = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="call_1", name="add", arguments='{"a":1,"b":2}')],
        reasoning_content="thinking...",
    )
    d = m.to_openai()
    assert d["tool_calls"][0]["function"]["name"] == "add"
    assert "reasoning_content" not in d
    assert m.to_openai(include_reasoning=True)["reasoning_content"] == "thinking..."


def test_tool_message_has_tool_call_id():
    assert Message.tool("call_1", "3").to_openai() == {
        "role": "tool", "content": "3", "tool_call_id": "call_1",
    }


def test_parse_response_with_tool_calls_and_cache():
    fake = NS(
        model="deepseek-v4.1-flash",
        choices=[NS(
            finish_reason="tool_calls",
            message=NS(
                content=None,
                reasoning_content=None,
                tool_calls=[NS(id="call_1", function=NS(name="add", arguments='{"a":1,"b":2}'))],
            ),
        )],
        usage=NS(prompt_tokens=100, completion_tokens=20, prompt_cache_hit_tokens=64),
    )
    r = OpenAICompatClient._parse(fake, latency_ms=12.3, attempts=1)
    assert r.finish_reason == "tool_calls"
    assert r.message.tool_calls[0].arguments == '{"a":1,"b":2}'
    assert r.usage.total_tokens == 120 and r.usage.cached_tokens == 64
