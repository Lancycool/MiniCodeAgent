"""Agent 运行时状态与预算模型（内存中的唯一真相源）。"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from mca.llm.types import Message


class Phase(StrEnum):
    """AgentLoop 有限状态机（FSM）的六个状态。"""

    INIT = "INIT"
    RUNNING = "RUNNING"
    RECOVERING = "RECOVERING"
    VERIFYING = "VERIFYING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Budget:
    """确定性预算硬上限，防止 Agent 失控烧钱或无限运行。"""

    max_steps: int = 20
    max_tokens: int = 100_000
    max_wall_time_s: float = 300.0
    max_recovers: int = 2
    max_verify_retries: int = 2


@dataclass(frozen=True)
class ActionRecord:
    """记录一次工具调用的摘要指纹，供 LoopGuard 检测重复与卡死。"""

    step_no: int
    tool_name: str
    args_hash: str
    status: str
    signature: str = ""  # 例如 run_tests 的失败摘要指纹，或 edit_file 的目标路径


@dataclass
class AgentState:
    task_id: str
    mode: str = "fix"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    phase: Phase = Phase.INIT
    messages: list[Message] = field(default_factory=list)
    checkpoints: list[tuple[int, str]] = field(default_factory=list)  # (step_no, git_sha)
    actions: list[ActionRecord] = field(default_factory=list)

    step_no: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    recover_count: int = 0
    verify_failures: int = 0
    consecutive_no_tool: int = 0
    consecutive_arg_errors: int = 0

    exit_reason: str = ""
    final_summary: str = ""
    final_patch: str = ""
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_at


@dataclass
class RunResult:
    run_id: str
    status: str           # "done" | "failed"
    exit_reason: str      # "verified" | "budget_steps" | "budget_tokens" | "stuck" | "verify_failed" | "llm_error"
    total_steps: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    wall_time_ms: float
    recover_count: int
    final_summary: str
    final_patch: str

    @property
    def ok(self) -> bool:
        return self.status == "done"
