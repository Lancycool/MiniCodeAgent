"""Lesson 5 实战演示：多文件仓库下的 AST 检索（Repo Map + 符号跳转）与上下文观察折叠。

运行：
  python scripts/l5_ast_context_demo.py
  python scripts/l5_ast_context_demo.py --docker
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

from mca.agent import AgentLoop, Budget, ContextConfig, ContextManager, LoopGuard
from mca.codeindex import CodeIndex, extract_focus_terms
from mca.config import LLMSettings
from mca.llm import OpenAICompatClient
from mca.observability import TraceStore, Tracer, format_run_replay, format_tool_stats
from mca.sandbox import DockerSandbox, LocalSandbox
from mca.tools import ToolRegistry, fix_tools

ISSUE = (
    "在电商结算子系统中，VIP 用户的订单结算金额不对：\n"
    "1) 业务规则：VIP 折扣（vip_discount）应该只对“商品小计”打折，运费（shipping_fee）不参与打折；\n"
    "2) 但现在运费也被一起打折了，导致 test_checkout.py 失败。\n"
    "请使用 AST 工具（view_outline / search_symbol / find_references）精准定位跨文件调用链并修复，完成后调用 submit。"
)


def create_multi_file_project(root: Path) -> None:
    """构造一个 6 文件、跨模块调用的迷你电商仓库（含干扰文件测试检索效率）。"""
    (root / "models.py").write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class Item:\n"
        "    name: str\n"
        "    unit_price: float\n"
        "    qty: int\n\n"
        "@dataclass\n"
        "class Order:\n"
        "    items: list[Item]\n"
        "    is_vip: bool = False\n"
        "    region: str = 'CN'\n",
        encoding="utf-8",
    )
    (root / "shipping.py").write_text(
        "def calc_shipping_fee(subtotal: float, region: str) -> float:\n"
        "    if subtotal >= 200.0:\n"
        "        return 0.0\n"
        "    return 15.0 if region == 'CN' else 40.0\n",
        encoding="utf-8",
    )
    (root / "discounts.py").write_text(
        "def apply_vip_discount(amount: float, is_vip: bool) -> float:\n"
        "    if not is_vip:\n"
        "        return amount\n"
        "    return round(amount * 0.85, 2)\n",
        encoding="utf-8",
    )
    (root / "checkout.py").write_text(
        "from discounts import apply_vip_discount\n"
        "from models import Order\n"
        "from shipping import calc_shipping_fee\n\n"
        "def compute_order_total(order: Order) -> float:\n"
        "    subtotal = sum(it.unit_price * it.qty for it in order.items)\n"
        "    shipping = calc_shipping_fee(subtotal, order.region)\n"
        "    # Bug: 把运费也加进去一起打 VIP 折了！\n"
        "    total = apply_vip_discount(subtotal + shipping, order.is_vip)\n"
        "    return round(total, 2)\n",
        encoding="utf-8",
    )
    # 干扰模块（模拟真实仓库里的无关大文件）
    filler = "\n\n".join(f"def legacy_report_{i}(x: int) -> int:\n    return x * {i}" for i in range(1, 25))
    (root / "legacy_reports.py").write_text(filler + "\n", encoding="utf-8")

    (root / "test_checkout.py").write_text(
        "from checkout import compute_order_total\n"
        "from models import Item, Order\n\n"
        "def test_regular_user_pays_shipping():\n"
        "    order = Order(items=[Item('book', 50.0, 2)], is_vip=False)\n"
        "    assert compute_order_total(order) == 115.0\n\n"
        "def test_vip_discount_excludes_shipping():\n"
        "    # 商品小计 100 -> VIP 85 折 = 85.0；运费 15.0 不打折 -> 合计 100.0\n"
        "    order = Order(items=[Item('book', 50.0, 2)], is_vip=True)\n"
        "    assert compute_order_total(order) == 100.0\n\n"
        "def test_vip_free_shipping_above_200():\n"
        "    order = Order(items=[Item('keyboard', 200.0, 1)], is_vip=True)\n"
        "    assert compute_order_total(order) == 170.0\n",
        encoding="utf-8",
    )


async def main(use_docker: bool) -> None:
    db_path = Path("runs/traces.db")
    store = TraceStore(db_path)

    with tempfile.TemporaryDirectory(prefix="mca_l5_src_") as src_dir:
        src_path = Path(src_dir)
        create_multi_file_project(src_path)

        # 先打印生成的 Repo Map 供直观观察
        idx = CodeIndex(src_path)
        idx.build()
        print(idx.build_repo_map(max_chars=2500, focus_terms=extract_focus_terms(ISSUE)))
        print()

        sandbox = DockerSandbox(src_path) if use_docker else LocalSandbox(src_path)
        async with sandbox as sb:
            loop = AgentLoop(
                llm=OpenAICompatClient(LLMSettings()),
                registry=ToolRegistry(fix_tools(include_ast=True)),
                sandbox=sb,
                budget=Budget(max_steps=15, max_tokens=60_000),
                loop_guard=LoopGuard(enabled=True),
                # 故意把 keep_recent_turns 设小一点（2），方便在短任务中也能看到 COMP=mask 生效
                context_manager=ContextManager(
                    ContextConfig(mask_enabled=True, summary_enabled=True, keep_recent_turns=2, min_mask_chars=200)
                ),
                inject_repo_map=True,
                tracer=Tracer(store),
            )
            res = await loop.run(ISSUE, task_id="demo_multifile_ast", mode="fix")

    print("\n" + format_run_replay(store, res.run_id))
    print("\n=== 工具统计 SQL（累计） ===")
    print(format_tool_stats(store))
    print("\n=== 最终 git diff ===")
    print(res.final_patch or "(无改动)")
    store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", action="store_true")
    asyncio.run(main(use_docker=parser.parse_args().docker))
