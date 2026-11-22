"""工具抽象：Tool / ToolParams / ToolContext / ToolResult / ToolError。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from mca.llm.types import Message
from mca.tools.schema import simplify_schema

ToolStatus = Literal[
    "ok",              # 正常
    "unknown_tool",    # 模型调用了不存在的工具（幻觉）
    "invalid_args",    # 参数 JSON 不合法 / 校验失败
    "error",           # 预期内的业务错误（文件不存在、正则非法……）
    "timeout",         # 执行超时
    "internal_error",  # 工具自身的 bug（非预期异常）
    "blocked",         # 被安全策略拦截（后续课程使用）
]


class ToolError(Exception):
    """预期内的工具错误。

    消息会**原样回传给模型**，因此必须写成"可行动的反馈"：
    说明错在哪 + 下一步可以怎么做。例如：
      ✗ "FileNotFoundError"
      ✓ "文件不存在: src/utl.py。可先用 list_dir 或 search_code 确认路径。"
    """


class ToolParams(BaseModel):
    """所有工具参数模型的基类。

    extra="forbid"：模型经常"发明"不存在的参数（如给 read_file 传 encoding），
    直接拒绝并告诉它，比静默忽略更好 —— 静默忽略会让模型以为参数生效了。
    """

    model_config = ConfigDict(extra="forbid")


@dataclass
class ToolContext:
    """工具执行时的环境。目前只有工作区根目录，下一课会加入 Sandbox。"""

    workspace: Path

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).resolve()

    def resolve_path(self, path: str) -> Path:
        """把模型给的路径解析为工作区内的绝对路径，并做越界检查。

        resolve() 会规范化 '..' 并跟随符号链接，因此 '../../etc/passwd'、
        指向工作区外的软链接都会被识别出来。这是"路径穿越（Path Traversal）"防护。
        """
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.workspace):
            raise ToolError(
                f"路径 '{path}' 位于仓库根目录之外，禁止访问。请使用仓库内的相对路径，如 'src/main.py'。"
            )
        return resolved

    def relpath(self, path: Path) -> str:
        return path.relative_to(self.workspace).as_posix()


class Tool(ABC):
    """一个工具 = 名称 + 描述 + 参数模型 + 执行逻辑 + 运行约束。

    参数模型一份定义，两种用途：
      1. 自动生成 JSON Schema 发给模型（告诉它怎么调用）
      2. 运行时校验模型给的参数（保证调用是合法的）
    """

    name: ClassVar[str]
    description: ClassVar[str]
    params_model: ClassVar[type[ToolParams]]
    readonly: ClassVar[bool] = True       # 只读工具可以并行执行
    timeout_s: ClassVar[float] = 30.0
    max_output_chars: ClassVar[int] = 8000

    @abstractmethod
    async def run(self, params: Any, ctx: ToolContext) -> str:
        """执行工具，返回给模型看的文本。预期内错误请抛 ToolError。"""

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": simplify_schema(self.params_model.model_json_schema()),
            },
        }


@dataclass
class ToolResult:
    call_id: str
    name: str
    status: ToolStatus
    content: str              # 回传给模型的文本（已截断）
    raw_chars: int = 0        # 截断前的长度，用于 Trace 分析"哪个工具输出最占上下文"
    truncated: bool = False
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_message(self) -> Message:
        return Message.tool(self.call_id, self.content)
