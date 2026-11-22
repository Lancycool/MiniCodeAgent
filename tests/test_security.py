"""敏感信息防护单测。测试中的"密钥"均为伪造字符串。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mca.llm.types import ToolCall
from mca.security import is_sensitive_path, redact_secrets
from mca.tools import ReadFileTool, SearchCodeTool, Tool, ToolContext, ToolParams, ToolRegistry, default_tools
from mca.tools.fs_tools import ReadFileParams, SearchCodeParams

FAKE_KEY = "sk-" + "a1B2c3D4e5F6g7H8i9J0"  # 拼接构造，避免本文件自身被扫描工具误报


@pytest.mark.parametrize(
    "name, expected",
    [
        (".env", True), (".env.local", True), ("prod.env", True), ("server.pem", True),
        ("id_rsa", True), ("id_ed25519.pub", True), ("credentials.json", True),
        (".env.example", False), ("config.py", False), ("environment.py", False),
    ],
)
def test_is_sensitive_path(name, expected):
    assert is_sensitive_path(Path(name)) is expected


def test_redact_token_and_assignment():
    text = f"MCA_LLM_API_KEY={FAKE_KEY}\nDB_PASSWORD='hunter2hunter2x9'\nname = 'ok'"
    out, n = redact_secrets(text)
    assert FAKE_KEY not in out and "hunter2hunter2x9" not in out
    assert "MCA_LLM_API_KEY=[REDACTED]" in out  # 键名保留
    assert "name = 'ok'" in out
    assert n == 2


@pytest.mark.parametrize(
    "code",
    [
        "password = get_password_from_vault()",       # 函数调用不是密钥
        "self.api_key = settings.api_key",            # 变量引用不是密钥
        "max_tokens: int = Field(4096, gt=0)",        # 含 token 的普通字段
        "risk-assessment-framework-config",           # 含 sk- 但不是密钥
    ],
)
def test_redact_no_false_positive_on_code(code):
    assert redact_secrets(code) == (code, 0)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / ".env").write_text(f"MCA_LLM_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("MCA_LLM_API_KEY=sk-xxxxxxxx\n", encoding="utf-8")
    (tmp_path / "settings.py").write_text(f'API_KEY = "{FAKE_KEY}"  # 硬编码的密钥\n', encoding="utf-8")
    return ToolContext(workspace=tmp_path)


async def test_read_env_is_blocked_with_status(repo):
    reg = ToolRegistry(default_tools())
    r = await reg.execute(ToolCall("c1", "read_file", '{"path": ".env"}'), repo)
    assert r.status == "blocked" and FAKE_KEY not in r.content and ".env.example" in r.content


async def test_env_example_is_readable(repo):
    out = await ReadFileTool().run(ReadFileParams(path=".env.example"), repo)
    assert "MCA_LLM_API_KEY" in out


async def test_search_skips_sensitive_files(repo):
    out = await SearchCodeTool().run(SearchCodeParams(pattern="MCA_LLM_API_KEY"), repo)
    assert ".env.example" in out and ".env:" not in out


async def test_hardcoded_key_is_redacted_by_registry(repo):
    """防线 1 拦不住硬编码在源码里的密钥 → 防线 2 兜底。"""
    reg = ToolRegistry(default_tools())
    r = await reg.execute(ToolCall("c1", "read_file", '{"path": "settings.py"}'), repo)
    assert r.status == "ok" and FAKE_KEY not in r.content
    assert "[REDACTED]" in r.content and r.redactions == 1


async def test_any_tool_output_is_redacted(tmp_path):
    class LeakyTool(Tool):
        name = "leaky"
        description = "模拟一个把环境变量打印出来的命令"
        params_model = ToolParams

        async def run(self, params, ctx):
            return f"OPENAI_API_KEY={FAKE_KEY}"

    r = await ToolRegistry([LeakyTool()]).execute(ToolCall("c1", "leaky", "{}"), ToolContext(tmp_path))
    assert FAKE_KEY not in r.content and r.redactions >= 1
