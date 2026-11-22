"""ToolRegistry 单测：重点验证"任何失败都被转换为可行动的反馈，而不是抛异常"。"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import Field

from mca.llm.types import ToolCall
from mca.tools import Tool, ToolContext, ToolError, ToolParams, ToolRegistry
from mca.tools.schema import simplify_schema
from mca.tools.truncate import truncate_middle


# ---------------- 测试用的假工具 ----------------

class EchoParams(ToolParams):
    text: str = Field(description="要回显的文本")
    times: int = Field(1, ge=1, le=5)
    title: str | None = Field(None, description="一个名字恰好叫 title 的字段")


class EchoTool(Tool):
    name = "echo"
    description = "回显文本"
    params_model = EchoParams
    max_output_chars = 100

    async def run(self, params: EchoParams, ctx: ToolContext) -> str:
        return params.text * params.times


class FailTool(Tool):
    name = "fail"
    description = "总是失败"
    params_model = ToolParams
    mode = "tool_error"

    async def run(self, params, ctx):
        if self.mode == "tool_error":
            raise ToolError("文件不存在，请先用 list_dir 确认路径")
        raise KeyError("bug")


class SlowTool(Tool):
    name = "slow"
    description = "很慢"
    params_model = ToolParams
    timeout_s = 0.05

    async def run(self, params, ctx):
        await asyncio.sleep(1)
        return "never"


class SleepTool(Tool):
    """用于验证只读工具并发执行。"""
    name = "sleep"
    description = "sleep"
    params_model = ToolParams

    async def run(self, params, ctx):
        await asyncio.sleep(0.2)
        return "done"


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(workspace=tmp_path)


@pytest.fixture
def registry():
    return ToolRegistry([EchoTool(), FailTool(), SlowTool(), SleepTool()])


def call(name: str, args: str, cid: str = "c1") -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=args)


# ---------------- Schema ----------------

def test_schema_is_simplified():
    params = EchoTool().schema()["function"]["parameters"]
    assert "title" not in params                       # 顶层 title 被删除
    assert "title" in params["properties"]             # 但名为 title 的字段保留
    assert params["properties"]["title"]["type"] == "string"  # Optional 折叠
    assert "anyOf" not in params["properties"]["title"]
    assert params["required"] == ["text"]


def test_nested_models_rejected():
    with pytest.raises(ValueError):
        simplify_schema({"$defs": {}, "type": "object"})


# ---------------- 执行与错误反馈 ----------------

async def test_ok(registry, ctx):
    r = await registry.execute(call("echo", '{"text": "ab", "times": 2}'), ctx)
    assert r.status == "ok" and r.content == "abab" and r.latency_ms >= 0


async def test_unknown_tool_suggests_close_match(registry, ctx):
    r = await registry.execute(call("ecoh", "{}"), ctx)
    assert r.status == "unknown_tool" and "echo" in r.content


async def test_invalid_json_points_to_position(registry, ctx):
    r = await registry.execute(call("echo", '{"text": "ab",}'), ctx)
    assert r.status == "invalid_args" and "位置" in r.content


async def test_validation_error_is_actionable(registry, ctx):
    r = await registry.execute(call("echo", '{"times": 9, "encoding": "utf8"}'), ctx)
    assert r.status == "invalid_args"
    assert "`text`: 缺少必填参数" in r.content
    assert "`times`" in r.content
    assert "`encoding`: 不存在该参数" in r.content


async def test_non_object_args(registry, ctx):
    r = await registry.execute(call("echo", '["a"]'), ctx)
    assert r.status == "invalid_args"


async def test_tool_error_vs_internal_error(registry, ctx):
    r = await registry.execute(call("fail", "{}"), ctx)
    assert r.status == "error" and "list_dir" in r.content

    FailTool.mode = "bug"
    try:
        r = await registry.execute(call("fail", "{}"), ctx)
    finally:
        FailTool.mode = "tool_error"
    assert r.status == "internal_error" and "KeyError" in r.content


async def test_timeout(registry, ctx):
    r = await registry.execute(call("slow", ""), ctx)
    assert r.status == "timeout"


async def test_output_truncated(registry, ctx):
    r = await registry.execute(call("echo", '{"text": "' + "x" * 60 + '", "times": 3}'), ctx)
    assert r.truncated and r.raw_chars == 180 and "已省略" in r.content


async def test_disabled_tool_is_invisible(ctx):
    reg = ToolRegistry([EchoTool(), FailTool()], enabled={"echo"})
    assert [s["function"]["name"] for s in reg.schemas()] == ["echo"]
    r = await reg.execute(call("fail", "{}"), ctx)
    assert r.status == "unknown_tool"


async def test_readonly_calls_run_concurrently_and_keep_order(registry, ctx):
    calls = [call("sleep", "{}", f"c{i}") for i in range(5)]
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    results = await registry.execute_many(calls, ctx)
    elapsed = loop.time() - t0
    assert [r.call_id for r in results] == [f"c{i}" for i in range(5)]
    assert elapsed < 0.6  # 串行需要 1.0s


# ---------------- 截断 ----------------

def test_truncate_middle_keeps_head_and_tail():
    text = "HEAD" + "m" * 1000 + "TAIL"
    out, truncated = truncate_middle(text, 100)
    assert truncated and out.startswith("HEAD") and out.endswith("TAIL")
    assert truncate_middle("short", 100) == ("short", False)
