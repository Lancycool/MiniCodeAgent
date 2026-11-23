"""多级上下文管理器（ContextManager）单测：验证观察折叠、协议配对完整性和摘要检查点。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mca.agent.context import ContextConfig, ContextManager
from mca.llm.types import LLMResponse, Message, ToolCall, Usage


class StubSummaryLLM:
    def __init__(self, summary: str) -> None:
        self.summary = summary
        self.calls = 0

    async def chat(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            message=Message(role="assistant", content=self.summary),
            finish_reason="stop",
            usage=Usage(prompt_tokens=50, completion_tokens=30),
            model="stub",
            latency_ms=1.0,
            attempts=1,
        )


def make_long_history(num_turns: int, obs_len: int = 800) -> list[Message]:
    msgs = [Message.system("sys prompt"), Message.user("fix the bug")]
    for i in range(1, num_turns + 1):
        cid = f"call_{i}"
        msgs.append(
            Message(
                role="assistant",
                content=f"step {i}",
                tool_calls=[ToolCall(id=cid, name="read_file", arguments=f'{{"path": "f{i}.py"}}')],
            )
        )
        msgs.append(Message.tool(cid, f"FILE_{i}_CONTENT " + "x" * obs_len))
    return msgs


async def test_observation_masking_preserves_recent_and_pairing():
    history = make_long_history(num_turns=6, obs_len=600)
    orig_first_tool_content = history[3].content

    cm = ContextManager(ContextConfig(mask_enabled=True, summary_enabled=False, keep_recent_turns=2, min_mask_chars=200))
    view, comp = await cm.build(history)

    assert comp == "mask"
    # 不得修改原始 state.messages
    assert history[3].content == orig_first_tool_content
    # 前 4 轮的 tool 输出被折叠，后 2 轮（call_5, call_6）保持完整
    tool_msgs = [m for m in view if m.role == "tool"]
    assert len(tool_msgs) == 6  # 所有 tool 消息成对保留，一个不少
    assert all("历史观察已折叠" in m.content for m in tool_msgs[:4])
    assert all("FILE_" in m.content and "历史观察已折叠" not in m.content for m in tool_msgs[4:])


async def test_summary_triggers_above_token_threshold():
    history = make_long_history(num_turns=6, obs_len=1500)
    stub = StubSummaryLLM("1. 已排除 f1.py~f4.py\n2. 根因在 f5.py")

    cm = ContextManager(
        ContextConfig(
            mask_enabled=True,
            summary_enabled=True,
            keep_recent_turns=2,
            summary_threshold_tokens=200,  # 故意设低以触发第 2 级压缩
        )
    )
    view, comp = await cm.build(history, llm=stub)

    assert comp == "summary" and stub.calls == 1
    assert view[0].role == "system" and view[1].role == "user"  # L0+L1 固定前缀不变
    assert "工作记忆摘要" in (view[2].content or "")
    assert "f5.py" in (view[2].content or "")
    # 再次 build 未新增需要摘要的轮次时，复用缓存摘要，不重复调 LLM
    _, comp2 = await cm.build(history, llm=stub)
    assert comp2 == "summary" and stub.calls == 1


async def test_compression_disabled_returns_exact_copy():
    history = make_long_history(num_turns=5, obs_len=600)
    cm = ContextManager(ContextConfig(mask_enabled=False, summary_enabled=False))
    view, comp = await cm.build(history)
    assert comp == "none" and view == history
