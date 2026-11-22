"""Lesson 1 冒烟测试：用一个"加法工具"走通完整的 function calling 往返。

运行：  python scripts/smoke_llm.py
观察重点：
  1. 模型不是直接回答，而是返回 tool_calls（finish_reason == "tool_calls"）
  2. 我们执行工具后，用 role="tool" + tool_call_id 把结果回传
  3. 模型拿到结果后继续调用工具或给出最终回答
这个 for 循环，就是一个最小的 Agent Loop。
"""

from __future__ import annotations

import asyncio
import json
import logging

from mca.config import LLMSettings
from mca.llm import Message, OpenAICompatClient

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "计算两个整数之和。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "第一个加数"},
                    "b": {"type": "integer", "description": "第二个加数"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


def run_tool(name: str, raw_args: str) -> str:
    # 注意：这里故意写得很"脆"——没有校验参数。下一课用 Pydantic 把它变成健壮的工具层。
    args = json.loads(raw_args)
    if name == "add":
        return str(args["a"] + args["b"])
    return f"error: unknown tool {name}"


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    client = OpenAICompatClient(LLMSettings())

    messages = [
        Message.system("你是一个计算助手。任何算术都必须调用工具完成，不要心算。"),
        Message.user("请计算 123456 + 654321，然后再把结果加上 1000。"),
    ]

    for turn in range(1, 6):  # 最多 5 轮：任何 Agent 循环都必须有上限
        resp = await client.chat(messages, tools=TOOLS)
        msg = resp.message
        print(f"\n=== Turn {turn} | finish_reason={resp.finish_reason} "
              f"| tokens={resp.usage.prompt_tokens}+{resp.usage.completion_tokens} "
              f"(cached {resp.usage.cached_tokens}) | {resp.latency_ms:.0f}ms | attempts={resp.attempts}")
        if msg.reasoning_content:
            print(f"[reasoning] {msg.reasoning_content[:200]}...")
        if msg.content:
            print(f"[assistant] {msg.content}")

        messages.append(msg)  # assistant 消息（含 tool_calls）必须先加入历史
        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            result = run_tool(tc.name, tc.arguments)
            print(f"[tool_call] {tc.name}({tc.arguments}) -> {result}   (id={tc.id})")
            messages.append(Message.tool(tc.id, result))  # 每个 tool_call 都必须有对应的 tool 消息

    print("\n=== 发送给模型的完整消息历史 ===")
    for m in messages:
        print(json.dumps(m.to_openai(), ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
