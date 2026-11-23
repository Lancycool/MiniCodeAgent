"""LocalSandbox：基于临时目录 + 子进程的轻量沙箱（仅用于开发调试）。

提供：
- 仓库副本隔离（不改原目录）
- git 基线 / checkpoint / rollback / diff
- 真正可强制终止的超时（超时后杀掉整个进程树，而不是只杀外壳 shell）

注意：LocalSandbox 与宿主机共享文件系统和网络，不能用于执行不可信代码。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from mca.sandbox.base import GIT_EXCLUDE_RULES, ExecResult, SandboxConfig, copy_repo_isolated


class LocalSandbox:
    def __init__(self, repo_path: Path | None = None, config: SandboxConfig | None = None) -> None:
        self._initial_repo = repo_path
        self.config = config or SandboxConfig()
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.workspace: Path = Path()
        self.baseline_sha: str = ""
        self.startup_ms: float = 0.0
        self.python_bin: str = f'"{sys.executable}"'

    async def __aenter__(self) -> LocalSandbox:
        if self._initial_repo is None:
            raise ValueError("repo_path must be provided when using 'async with LocalSandbox(repo_path)'")
        await self.start(self._initial_repo)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    async def start(self, repo_path: Path) -> None:
        t0 = time.perf_counter()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="mca_ws_")
        self.workspace = Path(self._tmpdir.name).resolve()
        copy_repo_isolated(repo_path, self.workspace)

        # 在副本内初始化全新的 git 仓库作为基线，用于 checkpoint / rollback / diff
        await self._git("init", "-q")
        # 写入 .git/info/exclude，过滤运行测试生成的 __pycache__/*.pyc，保持 diff 干净
        (self.workspace / ".git" / "info" / "exclude").write_text(GIT_EXCLUDE_RULES, encoding="utf-8")
        await self._git("config", "user.email", "agent@minicode.local")
        await self._git("config", "user.name", "MiniCodeAgent")
        await self._git("add", "-A")
        await self._git("commit", "-q", "--allow-empty", "-m", "baseline")
        res = await self._git("rev-parse", "HEAD")
        if not res.ok or not res.stdout.strip():
            raise RuntimeError(f"Failed to initialize git baseline: {res.stderr}")
        self.baseline_sha = res.stdout.strip()
        self.startup_ms = (time.perf_counter() - t0) * 1000

    async def exec(self, command: str, *, timeout_s: float | None = None) -> ExecResult:
        timeout = timeout_s if timeout_s is not None else self.config.default_timeout_s
        t0 = time.perf_counter()

        # Windows 用 CREATE_NEW_PROCESS_GROUP，POSIX 用 start_new_session=True，
        # 这样超时时可以一并杀掉 shell 派生出来的所有孙进程，防止僵尸/孤儿进程残留。
        kwargs: dict = {
            "cwd": str(self.workspace),
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        proc = await asyncio.create_subprocess_shell(command, **kwargs)
        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            await _kill_process_tree(proc)
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            except TimeoutError:
                stdout_b, stderr_b = b"", b""

        duration_ms = (time.perf_counter() - t0) * 1000
        return ExecResult(
            exit_code=-1 if timed_out else (proc.returncode if proc.returncode is not None else -1),
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            duration_ms=duration_ms,
            timed_out=timed_out,
        )

    async def checkpoint(self, message: str = "checkpoint") -> str:
        await self._git("add", "-A")
        status = await self._git("status", "--porcelain")
        if status.stdout.strip():
            await self._git("commit", "-q", "-m", message)
        res = await self._git("rev-parse", "HEAD")
        return res.stdout.strip()

    async def rollback(self, sha: str) -> None:
        res = await self._git("reset", "--hard", sha)
        if not res.ok:
            raise RuntimeError(f"git reset failed: {res.stderr}")
        await self._git("clean", "-fd")

    async def diff(self) -> str:
        """导出相对于基线的所有改动（含未提交变更）。"""
        await self._git("add", "-A")
        res = await self._git("diff", "--cached", self.baseline_sha)
        return res.stdout

    async def stop(self) -> None:
        if self._tmpdir is not None:
            try:
                self._tmpdir.cleanup()
            except OSError:
                # Windows 上偶尔因文件句柄未立即释放导致删除失败，静默降级
                shutil.rmtree(self.workspace, ignore_errors=True)
            self._tmpdir = None

    async def _git(self, *args: str) -> ExecResult:
        t0 = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(self.workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()
        return ExecResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            duration_ms=(time.perf_counter() - t0) * 1000,
        )


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """跨平台杀掉整个进程树。"""
    if proc.returncode is not None or proc.pid is None:
        return
    if sys.platform == "win32":
        # /T = 终止子进程树，/F = 强制终止
        killer = await asyncio.create_subprocess_exec(
            "taskkill", "/F", "/T", "/PID", str(proc.pid),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            await asyncio.sleep(0.2)
            if proc.returncode is None:
                os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
