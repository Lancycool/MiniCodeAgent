"""沙箱单元测试：验证隔离拷贝、超时杀进程树、git checkpoint/rollback/diff 以及 Docker 加固参数。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mca.sandbox import LocalSandbox, SandboxConfig, build_container_kwargs


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "orig_repo"
    repo.mkdir()
    (repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (repo / ".env").write_text("SECRET_KEY=sk-1234567890abcdefghij\n", encoding="utf-8")
    (repo / ".env.example").write_text("SECRET_KEY=sk-placeholder\n", encoding="utf-8")
    return repo


async def test_local_sandbox_isolates_and_skips_sensitive_files(sample_repo: Path):
    async with LocalSandbox(sample_repo) as sb:
        assert sb.workspace != sample_repo
        assert (sb.workspace / "app.py").exists()
        assert (sb.workspace / ".env.example").exists()
        assert not (sb.workspace / ".env").exists()  # 物理隔离：敏感文件不进沙箱
        assert sb.baseline_sha and sb.startup_ms >= 0

        # 修改沙箱副本，原仓库绝不受影响
        (sb.workspace / "app.py").write_text("changed\n", encoding="utf-8")
        assert "return a - b" in (sample_repo / "app.py").read_text(encoding="utf-8")


async def test_local_sandbox_checkpoint_rollback_and_diff(sample_repo: Path):
    async with LocalSandbox(sample_repo) as sb:
        (sb.workspace / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        sha1 = await sb.checkpoint("fix add")
        assert sha1 != sb.baseline_sha

        diff = await sb.diff()
        assert "-    return a - b" in diff and "+    return a + b" in diff
        # 运行产生的 __pycache__ 不应污染 diff
        (sb.workspace / "__pycache__").mkdir()
        (sb.workspace / "__pycache__" / "app.cpython-313.pyc").write_bytes(b"\x00\x01")
        assert "__pycache__" not in await sb.diff()

        # 再做一次坏修改并添加垃圾文件，然后回滚到 sha1
        (sb.workspace / "app.py").write_text("BROKEN\n", encoding="utf-8")
        (sb.workspace / "junk.txt").write_text("temp", encoding="utf-8")
        await sb.rollback(sha1)
        assert "return a + b" in (sb.workspace / "app.py").read_text(encoding="utf-8")
        assert not (sb.workspace / "junk.txt").exists()


async def test_local_sandbox_exec_and_timeout_kills_process(sample_repo: Path):
    async with LocalSandbox(sample_repo) as sb:
        res = await sb.exec(f'"{sys.executable}" -c "print(123)"')
        assert res.ok and res.stdout.strip() == "123"

        # 睡眠 10 秒的进程必须在 0.5 秒超时后被强制杀死
        slow = await sb.exec(f'"{sys.executable}" -c "import time; time.sleep(10)"', timeout_s=0.5)
        assert slow.timed_out and not slow.ok and slow.duration_ms < 5000


def test_docker_hardening_kwargs(tmp_path: Path):
    """不启动 Docker Daemon 也能验证所有安全加固参数是否正确拼装。"""
    cfg = SandboxConfig(mem_limit="512m", nano_cpus=500_000_000, pids_limit=128)
    kw = build_container_kwargs(tmp_path, cfg)
    assert kw["network_mode"] == "none"
    assert kw["mem_limit"] == "512m" and kw["memswap_limit"] == "512m"
    assert kw["nano_cpus"] == 500_000_000
    assert kw["pids_limit"] == 128
    assert kw["user"] == "1000:1000"
    assert kw["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in kw["security_opt"]
    assert kw["read_only"] is True
    assert "/tmp" in kw["tmpfs"]
    assert kw["environment"]["HOME"] == "/tmp"
