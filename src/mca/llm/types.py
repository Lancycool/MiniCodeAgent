"""项目内部的 LLM 数据类型。

为什么不直接用 openai SDK 的类型？
1. 解耦：Agent Core 只依赖这些类型，将来换 Anthropic / 本地模型只需新写一个 Client。
2. 可序列化：dataclass 能直接 asdict() 落库做 Trace，SDK 对象不方便。
3. 可控：我们只保留需要的字段，并能加自己的字段（如 latency_ms、attempts）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """模型发起的一次工具调用。

    arguments 保留模型返回的**原始 JSON 字符串**，不在这里解析：
    模型可能生成非法 JSON，解析与校验属于工具层的职责（下一课的 ToolRegistry），
    这样错误能被转换成"可行动的反馈"回传给模型，而不是在 LLM 层直接崩溃。
    """

    id: str
    name: str
    arguments: str

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class Message:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # 仅 role == "tool" 时使用，指向它回应的那个 ToolCall
    reasoning_content: str | None = None  # 思考型模型的推理过程（只用于记录/可选回传）

    # ---- 便捷构造器 ----
    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Message:
        return cls(role="tool", content=content, tool_call_id=tool_call_id)

    def to_openai(self, *, include_reasoning: bool = False) -> dict[str, Any]:
        """转换为 OpenAI Chat Completions 协议的消息格式。"""
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        if self.role == "tool":
            data["tool_call_id"] = self.tool_call_id
        if include_reasoning and self.role == "assistant" and self.reasoning_content:
            data["reasoning_content"] = self.reasoning_content
        return data


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # 命中服务端前缀缓存的 token，便宜很多，后面做成本统计要用

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    message: Message
    finish_reason: str | None  # "stop" | "tool_calls" | "length" | ...
    usage: Usage
    model: str
    latency_ms: float  # 包含重试等待在内的总耗时
    attempts: int  # 实际请求次数，1 表示一次成功
