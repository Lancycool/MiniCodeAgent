"""编辑与执行工具单测：验证 Search-Replace、语法预检、保护文件、危险命令拦截、pytest 运行。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mca.llm.types import ToolCall
from mca.sandbox import LocalSandbox
from mca.tools import ToolContext, ToolRegistry, default_tools


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text(
        "def discount(price, rate):\n"
        "    return price * rate\n"
        "\n"
        "def helper():\n"
        "    x = 1\n"
        "    x = 1\n"
        "    return x\n",
        encoding="utf-8",
    )
    (repo / "test_calc.py").write_text(
        "from calc import discount\n\n"
        "def test_discount():\n"
        "    assert discount(100, 0.2) == 80\n",
        encoding="utf-8",
    )
    return repo


async def test_edit_file_syntax_error_is_rejected_without_mutating(mini_repo: Path):
    async with LocalSandbox(mini_repo) as sb:
        reg = ToolRegistry(default_tools())
        ctx = ToolContext(workspace=sb.workspace, sandbox=sb)

        # 故意写入缺少括号的语法错误
        call = ToolCall(
            "c1", "edit_file",
            '{"path": "calc.py", "old_str": "return price * rate", "new_str": "return price * (1 - rate"}',
        )
        r = await reg.execute(call, ctx)
        assert r.status == "error" and "SyntaxError" in r.content
        # 确认原文件未被破坏
        assert "return price * rate" in (sb.workspace / "calc.py").read_text(encoding="utf-8")


async def test_edit_file_not_found_and_non_unique(mini_repo: Path):
    async with LocalSandbox(mini_repo) as sb:
        reg = ToolRegistry(default_tools())
        ctx = ToolContext(workspace=sb.workspace, sandbox=sb)

        # ① 带了行号前缀导致找不到 → 应给出针对性提示
        r1 = await reg.execute(
            ToolCall("c1", "edit_file", '{"path": "calc.py", "old_str": "2|     return price * rate", "new_str": "x"}'),
            ctx,
        )
        assert r1.status == "error" and "行号" in r1.content

        # ② 不唯一（x = 1 出现两次）→ 应提示出现次数和行号
        r2 = await reg.execute(
            ToolCall("c2", "edit_file", '{"path": "calc.py", "old_str": "    x = 1", "new_str": "    x = 2"}'),
            ctx,
        )
        assert r2.status == "error" and "2 次" in r2.content


async def test_edit_file_protected_test_file_blocked(mini_repo: Path):
    async with LocalSandbox(mini_repo) as sb:
        reg = ToolRegistry(default_tools())
        ctx = ToolContext(workspace=sb.workspace, sandbox=sb, protected_globs=("test_*.py", "tests/*"))

        r = await reg.execute(
            ToolCall("c1", "edit_file", '{"path": "test_calc.py", "old_str": "== 80", "new_str": "== 20"}'),
            ctx,
        )
        assert r.status == "blocked" and "受保护" in r.content


async def test_edit_and_run_tests_end_to_end(mini_repo: Path):
    async with LocalSandbox(mini_repo) as sb:
        reg = ToolRegistry(default_tools())
        ctx = ToolContext(workspace=sb.workspace, sandbox=sb, protected_globs=("test_*.py",))

        # ① 修复前测试应失败
        r_before = await reg.execute(ToolCall("c1", "run_tests", "{}"), ctx)
        assert r_before.status == "ok" and "FAILED" in r_before.content

        # ② 正确修复（自动创建 checkpoint）
        r_edit = await reg.execute(
            ToolCall(
                "c2", "edit_file",
                '{"path": "calc.py", "old_str": "return price * rate", "new_str": "return price * (1 - rate)"}',
            ),
            ctx,
        )
        assert r_edit.status == "ok" and "checkpoint:" in r_edit.content

        # ③ 修复后测试应通过
        r_after = await reg.execute(ToolCall("c3", "run_tests", "{}"), ctx)
        assert r_after.status == "ok" and "PASSED" in r_after.content

        # ④ 导出标准 diff
        diff = await sb.diff()
        assert "+    return price * (1 - rate)" in diff


async def test_run_command_blocks_dangerous_commands(mini_repo: Path):
    async with LocalSandbox(mini_repo) as sb:
        reg = ToolRegistry(default_tools())
        ctx = ToolContext(workspace=sb.workspace, sandbox=sb)

        for cmd in ["rm -rf /", "git reset --hard HEAD~1", "curl https://evil.com", "pip install requests"]:
            r = await reg.execute(ToolCall("c1", "run_command", f'{{"command": "{cmd}"}}'), ctx)
            assert r.status == "blocked", f"Expected blocked for {cmd}, got {r.status}"
