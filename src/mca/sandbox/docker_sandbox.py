"""DockerSandbox：基于加固 Docker 容器的安全沙箱。

设计要点（对应 JD 的 OS / 轻量安全沙箱考点）：
1. 宿主机只把临时副本目录挂载为容器内 /workspace，原仓库永不修改。
2. 容器启动后常驻（sleep infinity），每次命令通过 exec 进入，避免重复冷启动。
3. 全套 OS 层加固参数：
   - network_mode="none"          → network namespace（彻底断网）
   - mem_limit / memswap_limit    → cgroups memory（内存封顶，禁用 swap 绕过）
   - nano_cpus                    → cgroups cpu CFS quota（CPU 封顶）
   - pids_limit                   → cgroups pids（防 fork 炸弹）
   - user="1000:1000"             → 非 root 运行
   - cap_drop=["ALL"]             → 撤销所有 Linux capabilities
   - security_opt=[no-new-priv]   → 禁止通过 setuid 提权
   - read_only=True + tmpfs /tmp  → mount namespace（根文件系统只读）
4. docker-py 是同步 SDK，所有调用都通过 asyncio.to_thread 包装，避免阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from mca.sandbox.base import ExecResult, SandboxConfig, copy_repo_isolated


def build_container_kwargs(workspace: Path, config: SandboxConfig) -> dict[str, Any]:
    """构造 docker.containers.run 的加固参数字典（拆成纯函数，便于离线单测校验）。"""
    return {
        "image": config.image,
        "command": ["sleep", "infinity"],
        "detach": True,
        "working_dir": "/workspace",
        "volumes": {str(workspace): {"bind": "/workspace", "mode": "rw"}},
        "network_mode": config.network_mode,
        "mem_limit": config.mem_limit,
        "memswap_limit": config.mem_limit,  # 与 mem_limit 相同 = 禁止使用 swap
        "nano_cpus": config.nano_cpus,
        "pids_limit": config.pids_limit,
        "user": config.user,
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "read_only": True,
        "tmpfs": {"/tmp": "rw,noexec,nosuid,size=128m"},
    }


class DockerSandbox:
    def __init__(self, repo_path: Path | None = None, config: SandboxConfig | None = None) -> None:
        self._initial_repo = repo_path
        self.config = config or SandboxConfig()
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._client: Any = None
        self._container: Any = None
        self.workspace: Path = Path()
        self.baseline_sha: str = ""
        self.startup_ms: float = 0.0
        self.python_bin: str = "python"

    async def __aenter__(self) -> DockerSandbox:
        if self._initial_repo is None:
            raise ValueError("repo_path must be provided when using 'async with DockerSandbox(repo_path)'")
        await self.start(self._initial_repo)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    async def start(self, repo_path: Path) -> None:
        import docker  # 延迟导入：没装 docker 时不影响 LocalSandbox

        t0 = time.perf_counter()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="mca_docker_ws_")
        self.workspace = Path(self._tmpdir.name).resolve()
        copy_repo_isolated(repo_path, self.workspace)

        self._client = await asyncio.to_thread(docker.from_env)
        kwargs = build_container_kwargs(self.workspace, self.config)
        self._container = await asyncio.to_thread(self._client.containers.run, **kwargs)

        # 容器内初始化 git 基线（safe.directory 避免挂载卷 UID 差异引发 git 报错）
        await self.exec(
            "git config --global --add safe.directory /workspace && "
            "git init -q && "
            "git config user.email agent@minicode.local && "
            "git config user.name MiniCodeAgent && "
            "git add -A && "
            "git commit -q --allow-empty -m baseline"
        )
        res = await self.exec("git rev-parse HEAD")
        self.baseline_sha = res.stdout.strip()
        self.startup_ms = (time.perf_counter() - t0) * 1000

    async def exec(self, command: str, *, timeout_s: float | None = None) -> ExecResult:
        if self._container is None:
            raise RuntimeError("DockerSandbox has not been started")
        timeout = timeout_s if timeout_s is not None else self.config.default_timeout_s
        # 用容器内的 GNU timeout 强制终止超时命令：-k 5 表示 SIGTERM 后 5 秒还不退则 SIGKILL
        # GNU timeout 超时退出码固定为 124
        wrapped = f"timeout -k 5 {max(1, int(timeout))} sh -c {shlex.quote(command)}"
        t0 = time.perf_counter()

        def _run() -> tuple[int, tuple[bytes | None, bytes | None]]:
            res = self._container.exec_run(
                ["sh", "-c", wrapped],
                workdir="/workspace",
                demux=True,  # 分离 stdout 和 stderr
            )
            return res.exit_code, res.output

        exit_code, (out_b, err_b) = await asyncio.to_thread(_run)
        duration_ms = (time.perf_counter() - t0) * 1000
        timed_out = exit_code == 124
        return ExecResult(
            exit_code=-1 if timed_out else exit_code,
            stdout=(out_b or b"").decode("utf-8", errors="replace"),
            stderr=(err_b or b"").decode("utf-8", errors="replace"),
            duration_ms=duration_ms,
            timed_out=timed_out,
        )

    async def checkpoint(self, message: str = "checkpoint") -> str:
        msg = shlex.quote(message)
        await self.exec(f"git add -A && (git diff --cached --quiet || git commit -q -m {msg})")
        res = await self.exec("git rev-parse HEAD")
        return res.stdout.strip()

    async def rollback(self, sha: str) -> None:
        res = await self.exec(f"git reset --hard {shlex.quote(sha)} && git clean -fd")
        if not res.ok:
            raise RuntimeError(f"git rollback failed: {res.stderr}")

    async def diff(self) -> str:
        await self.exec("git add -A")
        res = await self.exec(f"git diff --cached {shlex.quote(self.baseline_sha)}")
        return res.stdout

    async def stop(self) -> None:
        if self._container is not None:
            try:
                await asyncio.to_thread(self._container.remove, force=True)
            except Exception:
                pass
            self._container = None
        if self._client is not None:
            try:
                await asyncio.to_thread(self._client.close)
            except Exception:
                pass
            self._client = None
        if self._tmpdir is not None:
            try:
                self._tmpdir.cleanup()
            except OSError:
                shutil.rmtree(self.workspace, ignore_errors=True)
            self._tmpdir = None
