"""沙箱抽象层：协议定义、配置、仓库隔离拷贝。

核心职责：
1. 隔离：Agent 的任何修改和命令都在独立副本中执行，绝不污染原始仓库或宿主机。
2. 复原：基于 git 的 checkpoint / rollback / diff，保证可回退、可导出标准补丁。
3. 可替换：LocalSandbox（开发调试）和 DockerSandbox（生产隔离）实现同一套接口。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from mca.security import is_sensitive_path
from mca.tools.fs_tools import IGNORED_DIRS


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def combined_output(self) -> str:
        parts: list[str] = []
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.stderr.strip():
            parts.append(f"[stderr]\n{self.stderr.rstrip()}")
        return "\n".join(parts) if parts else "(无输出)"


@dataclass(frozen=True)
class SandboxConfig:
    """沙箱资源与隔离参数（主要作用于 DockerSandbox，超时同时作用于 LocalSandbox）。"""

    image: str = "mca-python:3.11"
    mem_limit: str = "1g"
    nano_cpus: int = 1_000_000_000  # 1 核 CPU
    pids_limit: int = 256           # 限制进程数，防 fork 炸弹
    network_mode: str = "none"      # 彻底断网，防数据外泄和拉取恶意脚本
    user: str = "1000:1000"         # 非 root 运行
    default_timeout_s: float = 30.0


@runtime_checkable
class Sandbox(Protocol):
    workspace: Path
    baseline_sha: str
    startup_ms: float
    python_bin: str

    async def start(self, repo_path: Path) -> None: ...
    async def exec(self, command: str, *, timeout_s: float | None = None) -> ExecResult: ...
    async def checkpoint(self, message: str = "checkpoint") -> str: ...
    async def rollback(self, sha: str) -> None: ...
    async def diff(self) -> str: ...
    async def stop(self) -> None: ...

    async def __aenter__(self) -> Sandbox: ...
    async def __aexit__(self, exc_type, exc, tb) -> None: ...


# 写入 .git/info/exclude（不污染仓库自身的 .gitignore），避免运行测试产生的缓存污染 git diff
GIT_EXCLUDE_RULES = "__pycache__/\n*.py[cod]\n.pytest_cache/\n.coverage\n*.egg-info/\n"


def copy_repo_isolated(src: Path, dst: Path) -> None:
    """将目标仓库拷贝到沙箱工作区。

    纵深防御的又一环：拷贝时直接跳过 .git、缓存目录以及 .env / 私钥等敏感文件，
    让敏感文件从物理上就不存在于沙箱里——即使 Agent 执行 `cat .env` 也读不到。
    """
    src = Path(src).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Repository directory not found: {src}")

    def _ignore(dirpath: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            p = Path(dirpath) / name
            if p.is_dir() and (name in IGNORED_DIRS or name.endswith(".egg-info")):
                ignored.add(name)
            elif p.is_file() and is_sensitive_path(p):
                ignored.add(name)
        return ignored

    shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=True)
