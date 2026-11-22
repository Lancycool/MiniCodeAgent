"""Lesson 2 演示：用 ToolRegistry + 只读文件工具，让 Agent 回答一个关于本仓库的多跳问题。

运行：  python scripts/l2_agent_demo.py
第一部分：直接调用 registry，展示各种"错误调用"时模型会收到的反馈
第二部分：真实 Agent 循环，观察模型如何组合 list_dir / search_code / read_file
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from mca.config import LLMSettings
from mca.llm import Message, OpenAICompatClient, ToolCall
from mca.tools import ToolContext, ToolRegistry, default_tools

REPO_ROOT = Path(__file__).resolve().parents[1]

SYSTEM = """你是一个代码阅读助手，通过工具探索仓库来回答问题。
规则：
- 先用 list_dir 了解结构，用 search_code 定位，再用 read_file 精读，不要凭空猜测。
- 回答中引用的每个结论都要给出 文件路径:行号 作为证据。"""

QUESTION = (
    "在这个仓库中：1) LLM 调用失败时默认最多请求几次？这个默认值定义在哪个文件第几行？"
    "2) 如果服务端返回 429 且带 Retry-After: 100，默认配置下实际会等待多少秒？为什么？"
)


async def show_error_feedback(registry: ToolRegistry, ctx: ToolContext) -> None:
    print("=" * 20, "Part 1: 模型会收到怎样的错误反馈", "=" * 20)
    bad_calls = [
        ToolCall("d1", "read_fil", '{"path": "README.md"}'),                     # 工具名拼错
        ToolCall("d2", "read_file", '{"path": "README.md", start_line: 1}'),     # 非法 JSON
        ToolCall("d3", "read_file", '{"start_line": 0, "encoding": "utf8"}'),    # 校验失败
        ToolCall("d4", "read_file", '{"path": "../../secret.txt"}'),             # 路径穿越
        ToolCall("d5", "search_code", '{"pattern": "chat("}'),                    # 非法正则
    ]
    for r in await registry.execute_many(bad_calls, ctx):
        print(f"\n[{r.name} → status={r.status}]\n{r.content}")


async def agent_loop(registry: ToolRegistry, ctx: ToolContext) -> None:
    print("\n" + "=" * 20, "Part 2: Agent 循环", "=" * 20)
    client = OpenAICompatClient(LLMSettings())
    messages = [Message.system(SYSTEM), Message.user(QUESTION)]
    total_tokens = 0

    for turn in range(1, 16):
        resp = await client.chat(messages, tools=registry.schemas())
        total_tokens += resp.usage.total_tokens
        msg = resp.message
        messages.append(msg)
        print(f"\n--- Turn {turn} | {resp.finish_reason} | ctx={resp.usage.prompt_tokens} tok | {resp.latency_ms:.0f}ms")
        if msg.content:
            print(f"[assistant] {msg.content}")
        if not msg.tool_calls:
            break

        results = await registry.execute_many(msg.tool_calls, ctx)
        for tc, r in zip(msg.tool_calls, results, strict=True):
            preview = r.content.replace("\n", " | ")[:150]
            print(f"  ↳ {tc.name}({tc.arguments}) → {r.status} {r.latency_ms:.1f}ms "
                  f"{'[truncated]' if r.truncated else ''}\n     {preview}")
            messages.append(r.to_message())
    else:
        print("\n[达到最大轮数，强制停止]")

    print(f"\n总消耗 tokens: {total_tokens}")


async def main() -> None:
    registry = ToolRegistry(default_tools())
    ctx = ToolContext(workspace=REPO_ROOT)
    await show_error_feedback(registry, ctx)
    await agent_loop(registry, ctx)


if __name__ == "__main__":
    asyncio.run(main())
