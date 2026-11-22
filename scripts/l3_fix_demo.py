"""Lesson 3 实战演示：Agent 在沙箱里自主定位 Bug、修改代码、跑 pytest 验证、输出 git diff。

运行：
  python scripts/l3_fix_demo.py            # 默认用 LocalSandbox（秒起）
  python scripts/l3_fix_demo.py --docker   # 使用加固的 DockerSandbox（需先启动 Docker Desktop 并 build 镜像）
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

from mca.config import LLMSettings
from mca.llm import Message, OpenAICompatClient
from mca.sandbox import DockerSandbox, LocalSandbox
from mca.tools import ToolContext, ToolRegistry, default_tools

SYSTEM = """你是一个严谨的 Python 工程师，负责在隔离沙箱中修复仓库中的 Bug。
工作流：
1. 先用 list_dir / read_file 了解代码，用 run_tests 运行测试确认失败现象。
2. 分析根因后，用 edit_file 做最小、精确的修改（严禁修改测试文件）。
3. 再次调用 run_tests 确认所有测试通过，然后给出简明总结。"""

ISSUE = (
    "用户反馈：购物车结算时有两个问题：\n"
    "1) 满 100 元应该触发 9 折优惠，但刚好买 100 元时没有打折；\n"
    "2) 使用定额优惠券后，总价可能变成负数，正确行为是最低为 0.0。\n"
    "请定位并修复这两个 Bug，确保所有测试通过。"
)


def create_buggy_repo(root: Path) -> None:
    """在临时目录构造一个含两个典型 Bug 的迷你项目。"""
    (root / "cart.py").write_text(
        '"""简易购物车结算模块。"""\n\n'
        "def calculate_total(items: list[tuple[str, float, int]], coupon: float = 0.0) -> float:\n"
        '    """计算购物车总价。\n'
        "    items: [(商品名, 单价, 数量), ...]\n"
        "    规则：\n"
        "      1. 小计满 100 元享 9 折；\n"
        "      2. 再减去定额优惠券 coupon，最终总价不能低于 0.0，保留 2 位小数。\n"
        '    """\n'
        "    subtotal = sum(price * qty for _, price, qty in items)\n"
        "    if subtotal > 100:  # Bug 1: 应该是 >=\n"
        "        subtotal *= 0.9\n"
        "    total = subtotal - coupon  # Bug 2: 未做 max(0.0, total)\n"
        "    return round(total, 2)\n",
        encoding="utf-8",
    )
    (root / "test_cart.py").write_text(
        "from cart import calculate_total\n\n"
        "def test_no_discount():\n"
        '    assert calculate_total([("apple", 10.0, 5)]) == 50.0\n\n'
        "def test_exact_threshold_gets_discount():\n"
        '    assert calculate_total([("book", 50.0, 2)]) == 90.0\n\n'
        "def test_above_threshold_with_coupon():\n"
        '    assert calculate_total([("keyboard", 200.0, 1)], coupon=30.0) == 150.0\n\n'
        "def test_coupon_exceeds_total_floors_at_zero():\n"
        '    assert calculate_total([("pen", 5.0, 2)], coupon=50.0) == 0.0\n',
        encoding="utf-8",
    )


async def run_demo(use_docker: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="mca_l3_src_") as src_dir:
        src_path = Path(src_dir)
        create_buggy_repo(src_path)

        sandbox = DockerSandbox(src_path) if use_docker else LocalSandbox(src_path)
        async with sandbox as sb:
            print(f"[Sandbox] {type(sb).__name__} 启动耗时: {sb.startup_ms:.0f}ms | baseline={sb.baseline_sha[:7]}")

            registry = ToolRegistry(default_tools())
            # 禁止修改测试文件，防止模型作弊
            ctx = ToolContext(workspace=sb.workspace, sandbox=sb, protected_globs=("test_*.py", "tests/*"))
            client = OpenAICompatClient(LLMSettings())
            messages = [Message.system(SYSTEM), Message.user(ISSUE)]
            total_tokens = 0

            for turn in range(1, 12):
                resp = await client.chat(messages, tools=registry.schemas())
                total_tokens += resp.usage.total_tokens
                msg = resp.message
                messages.append(msg)
                print(f"\n--- Turn {turn} | {resp.finish_reason} | ctx={resp.usage.prompt_tokens} tok | {resp.latency_ms:.0f}ms")
                if msg.content:
                    print(f"[assistant] {msg.content}")
                if not msg.tool_calls:
                    break

                results = await registry.execute_many(msg.tool_calls, ctx)
                for tc, r in zip(msg.tool_calls, results, strict=True):
                    preview = r.content.replace("\n", " | ")[:180]
                    print(f"  ↳ {tc.name}({tc.arguments[:120]}) → {r.status} {r.latency_ms:.0f}ms\n     {preview}")
                    messages.append(r.to_message())

            print("\n" + "=" * 20 + " 最终 git diff " + "=" * 20)
            print(await sb.diff() or "(无改动)")
            print(f"总消耗 tokens: {total_tokens}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", action="store_true", help="使用 DockerSandbox 而不是 LocalSandbox")
    args = parser.parse_args()
    asyncio.run(run_demo(use_docker=args.docker))
