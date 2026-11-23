"""多级上下文动态压缩管理器（ContextManager）。

设计要点（直接命中 JD 的“上下文压缩与管理”考点）：
1. **前缀缓存友好**：L0（System Prompt）+ L1（Issue + Repo Map）永远固定在最前面不改动，
   保障 DeepSeek/OpenAI 的 Prefix Caching 持续命中。
2. **第一级压缩 —— 历史观察折叠（Observation Masking，零成本、确定性）**：
   保留最近 keep_recent_turns 轮完整对话；更早的 tool 输出若超过阈值，替换为一行可恢复占位符。
   绝不破坏 (assistant.tool_calls ↔ tool.tool_call_id) 的协议成对约束！
3. **第二级压缩 —— 阶段性摘要检查点（Episodic Summary，按需触发）**：
   折叠后估算 Token 仍超过 summary_threshold_tokens 时，调用一次 LLM 把远期历史浓缩成结构化
   工作记忆（已确认事实 / 已排除路径 / 已修改文件 / 失败尝试），释放上下文空间。
4. **不修改原始 state.messages**：始终返回裁剪后的新列表，保证 Trace 记录的完整历史不受影响。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from mca.llm.client import LLMClient
from mca.llm.types import Message

SUMMARY_PROMPT = """请将下面的历史调试对话浓缩为一份简洁的“工作记忆摘要”（不超过 300 字），必须包含：
1. 已确认的根因或关键事实（带文件路径/行号）
2. 已阅读但确认无关的“已排除文件/路径”（避免后续重复读取）
3. 已修改过的文件及效果
4. 已尝试但失败的假设（避免重蹈覆辙）
只输出摘要正文，不要寒暄。"""


@dataclass(frozen=True)
class ContextConfig:
    mask_enabled: bool = True
    summary_enabled: bool = True
    keep_recent_turns: int = 3           # 最近 N 轮保留完整工具输出
    min_mask_chars: int = 300            # 短于此长度的工具输出不折叠（如 "2 passed"）
    summary_threshold_tokens: int = 12_000  # 折叠后仍超此估算 Token 时触发摘要


def estimate_tokens(text: str) -> int:
    """零依赖、离线可用的快速 Token 估算（ASCII 约 4 字符/token，中文等非 ASCII 约 1.5 字符/token）。"""
    if not text:
        return 0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    non_ascii = len(text) - ascii_chars
    return max(1, int(ascii_chars / 4 + non_ascii / 1.5))


def estimate_messages_tokens(messages: list[Message]) -> int:
    total = 0
    for m in messages:
        total += 4  # 每条消息的协议开销
        if m.content:
            total += estimate_tokens(m.content)
        for tc in m.tool_calls:
            total += estimate_tokens(tc.name) + estimate_tokens(tc.arguments) + 4
    return total


class ContextManager:
    def __init__(self, config: ContextConfig | None = None) -> None:
        self.config = config or ContextConfig()
        self._cached_summary: str = ""
        self._summarized_until_idx: int = 0  # 已摘要到的原始消息下标（不含）

    async def build(
        self,
        messages: list[Message],
        *,
        llm: LLMClient | None = None,
    ) -> tuple[list[Message], str]:
        """根据配置生成发给模型的消息视图，返回 (view, compression_applied)。

        compression_applied 取值："none" | "mask" | "summary"
        """
        if not self.config.mask_enabled and not self.config.summary_enabled:
            return list(messages), "none"

        # 切分固定前缀（L0 system + L1 初始 user issue）与后续动态对话
        prefix_end = _find_prefix_end(messages)
        prefix = messages[:prefix_end]
        tail = messages[prefix_end:]

        # 找到最近 keep_recent_turns 轮 assistant 消息的起始位置
        recent_start_in_tail = _find_recent_turn_start(tail, self.config.keep_recent_turns)

        # ---- 第 1 级：Observation Masking（零成本折叠远期 tool 输出） ----
        compression = "none"
        if self.config.mask_enabled and recent_start_in_tail > 0:
            masked_older, did_mask = _mask_tool_observations(
                tail[:recent_start_in_tail],
                min_chars=self.config.min_mask_chars,
            )
            view = prefix + masked_older + tail[recent_start_in_tail:]
            if did_mask:
                compression = "mask"
        else:
            view = list(messages)

        # ---- 第 2 级：Episodic Summary（超阈值时把远期历史压缩为工作记忆） ----
        if (
            self.config.summary_enabled
            and llm is not None
            and recent_start_in_tail > 0
            and estimate_messages_tokens(view) > self.config.summary_threshold_tokens
        ):
            abs_cut = prefix_end + recent_start_in_tail
            if abs_cut > self._summarized_until_idx:
                older_slice = messages[max(prefix_end, self._summarized_until_idx) : abs_cut]
                self._cached_summary = await self._summarize(llm, self._cached_summary, older_slice)
                self._summarized_until_idx = abs_cut
            if self._cached_summary:
                summary_msg = Message.user(
                    f"[工作记忆摘要（前序多轮调试已压缩）]\n{self._cached_summary}"
                )
                view = prefix + [summary_msg] + messages[self._summarized_until_idx :]
                compression = "summary"

        return view, compression

    async def _summarize(self, llm: LLMClient, prev_summary: str, older: list[Message]) -> str:
        lines: list[str] = []
        if prev_summary:
            lines.append(f"【已有摘要】\n{prev_summary}\n")
        lines.append("【新增历史】")
        for m in older:
            if m.role == "assistant" and m.tool_calls:
                calls = ", ".join(f"{tc.name}({tc.arguments[:80]})" for tc in m.tool_calls)
                lines.append(f"Assistant 调用工具: {calls}")
            elif m.role == "tool":
                lines.append(f"Tool 返回: {(m.content or '')[:200]}")
            elif m.content:
                lines.append(f"{m.role}: {m.content[:200]}")

        try:
            resp = await llm.chat(
                [Message.system(SUMMARY_PROMPT), Message.user("\n".join(lines)[:8000])],
                tools=None,
            )
            return (resp.message.content or prev_summary).strip()
        except Exception:
            return prev_summary


def _find_prefix_end(messages: list[Message]) -> int:
    """保留开头的 system + 第一条 user 消息作为永不压缩的固定前缀。"""
    idx = 0
    if idx < len(messages) and messages[idx].role == "system":
        idx += 1
    if idx < len(messages) and messages[idx].role == "user":
        idx += 1
    return idx


def _find_recent_turn_start(tail: list[Message], keep_recent_turns: int) -> int:
    """找到倒数第 keep_recent_turns 个 assistant 消息的下标（保证不拆散 tool_calls 对）。"""
    if keep_recent_turns <= 0:
        return len(tail)
    assistant_indices = [i for i, m in enumerate(tail) if m.role == "assistant"]
    if len(assistant_indices) <= keep_recent_turns:
        return 0
    return assistant_indices[-keep_recent_turns]


# 高信号工具（代码修改确认 / 测试结论）对后续决策影响极大，折叠阈值应高于普通读文件输出，
# 避免模型因“看不到刚改完的代码和刚通过的测试”而陷入失忆重复检查（见 debug_log.md #8）。
_HIGH_SIGNAL_TOOLS = {"edit_file", "run_tests", "submit"}


def _mask_tool_observations(older: list[Message], *, min_chars: int) -> tuple[list[Message], bool]:
    """把远期超过阈值的 tool 消息折叠为一行占位符，保持消息对完整且不改原对象。"""
    # 先建立 tool_call_id → (tool_name, 简短参数) 映射，让占位符清晰可恢复
    call_meta: dict[str, tuple[str, str]] = {}
    for m in older:
        for tc in m.tool_calls:
            call_meta[tc.id] = (tc.name, _short_args(tc.arguments))

    out: list[Message] = []
    did_mask = False
    for m in older:
        if m.role == "tool" and m.content:
            name, short_args = call_meta.get(m.tool_call_id or "", ("tool", ""))
            threshold = max(min_chars, 1500) if name in _HIGH_SIGNAL_TOOLS else min_chars
            if len(m.content) >= threshold:
                placeholder = (
                    f"[历史观察已折叠: {name}({short_args}) 原输出 {len(m.content)} 字符。"
                    f"如需原文请重新调用该工具]"
                )
                out.append(Message.tool(m.tool_call_id or "", placeholder))
                did_mask = True
                continue
        out.append(m)
    return out, did_mask


def _short_args(args_json: str) -> str:
    try:
        d = json.loads(args_json)
        if isinstance(d, dict):
            return ", ".join(f"{k}={v!r}" for k, v in list(d.items())[:2])[:60]
    except Exception:
        pass
    return args_json[:40]
