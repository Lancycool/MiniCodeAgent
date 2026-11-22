"""文件系统工具单测（使用 pytest 的 tmp_path 构造一个迷你仓库）。"""

from __future__ import annotations

import pytest

from mca.tools import ListDirTool, ReadFileTool, SearchCodeTool, ToolContext, ToolError
from mca.tools.fs_tools import ListDirParams, ReadFileParams, SearchCodeParams


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "core.py").write_text(
        "\n".join(f"line {i}" for i in range(1, 301)) + "\ndef target_func():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "pkg" / "util.py").write_text("def helper():\n    return target_func()\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("target_func", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00\x01")
    return ToolContext(workspace=tmp_path)


# ---------------- 路径安全 ----------------

@pytest.mark.parametrize("bad", ["../outside.txt", "src/../../x", "/etc/passwd"])
def test_path_traversal_blocked(repo, bad):
    with pytest.raises(ToolError, match="之外"):
        repo.resolve_path(bad)


# ---------------- list_dir ----------------

async def test_list_dir_ignores_git(repo):
    out = await ListDirTool().run(ListDirParams(depth=3), repo)
    assert "src/" in out and "core.py" in out
    assert ".git" not in out


async def test_list_dir_on_file_gives_hint(repo):
    with pytest.raises(ToolError, match="read_file"):
        await ListDirTool().run(ListDirParams(path="src/pkg/core.py"), repo)


# ---------------- read_file ----------------

async def test_read_file_paginates(repo):
    out = await ReadFileTool().run(ReadFileParams(path="src/pkg/core.py"), repo)
    assert "第 1-200 行，共 302 行" in out
    assert "start_line=201" in out
    assert "  1| line 1" in out


async def test_read_file_range(repo):
    out = await ReadFileTool().run(ReadFileParams(path="src/pkg/core.py", start_line=301, end_line=302), repo)
    assert "301| def target_func():" in out and "未显示完" not in out


async def test_read_file_errors(repo):
    tool = ReadFileTool()
    with pytest.raises(ToolError, match="不存在"):
        await tool.run(ReadFileParams(path="nope.py"), repo)
    with pytest.raises(ToolError, match="二进制"):
        await tool.run(ReadFileParams(path="blob.bin"), repo)
    with pytest.raises(ToolError, match="超出"):
        await tool.run(ReadFileParams(path="src/pkg/util.py", start_line=99), repo)


# ---------------- search_code ----------------

async def test_search_code_finds_definition_and_usage(repo):
    out = await SearchCodeTool().run(SearchCodeParams(pattern=r"target_func\("), repo)
    assert "src/pkg/core.py:301: def target_func():" in out
    assert "src/pkg/util.py:2:" in out
    assert ".git" not in out  # 忽略目录不被搜索


async def test_search_code_no_match_is_not_error(repo):
    out = await SearchCodeTool().run(SearchCodeParams(pattern="does_not_exist"), repo)
    assert "未找到" in out


async def test_search_code_invalid_regex(repo):
    with pytest.raises(ToolError, match="转义"):
        await SearchCodeTool().run(SearchCodeParams(pattern="foo("), repo)


async def test_search_code_caps_results(repo):
    out = await SearchCodeTool().run(SearchCodeParams(pattern=r"^line"), repo)
    assert "共匹配 300 处" in out
