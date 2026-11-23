"""Lesson 6 实战演示：Review 模式（屎山代码诊断 + 可选测试护航安全重构）。

运行方式：
  1) 诊断内置屎山样例：
     python scripts/l6_review_demo.py
  2) 诊断 + 测试护航安全重构（看重构前后 CC / 行数 / 嵌套深度对比！）：
     python scripts/l6_review_demo.py --refactor
  3) 直接诊断你自己的旧 Python 项目（生成 runs/review_report.md）：
     python scripts/l6_review_demo.py --repo "E:\\path\\to\\your_old_project"
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

from mca.config import LLMSettings
from mca.llm import OpenAICompatClient
from mca.observability import TraceStore, Tracer
from mca.review import (
    ReviewReport,
    judge_hotspots,
    rank_hotspots,
    safe_refactor_hotspot,
    save_report,
    scan_repository,
)
from mca.sandbox import DockerSandbox, LocalSandbox


def create_shit_mountain_repo(root: Path) -> None:
    """构造一个带完整 pytest 测试的经典“屎山”订单处理仓库。"""
    (root / "order_processor.py").write_text(
        "ORDER_LOG = []\n\n"
        "def process_order(user, items, region, vip_level, coupon, is_express):\n"
        "    global ORDER_LOG\n"
        "    if user:\n"
        "        if items:\n"
        "            subtotal = 0.0\n"
        "            for it in items:\n"
        "                if it['price'] > 0:\n"
        "                    if it['qty'] > 0:\n"
        "                        subtotal += it['price'] * it['qty']\n"
        "            if vip_level == 'bronze':\n"
        "                discount = 0.95\n"
        "            elif vip_level == 'silver':\n"
        "                discount = 0.90\n"
        "            elif vip_level == 'gold':\n"
        "                discount = 0.85\n"
        "            elif vip_level == 'platinum':\n"
        "                discount = 0.80\n"
        "            elif vip_level == 'diamond':\n"
        "                discount = 0.75\n"
        "            else:\n"
        "                discount = 1.0\n"
        "            after_discount = subtotal * discount\n"
        "            if region == 'CN':\n"
        "                shipping = 0.0 if after_discount >= 200 else (25.0 if is_express else 10.0)\n"
        "            else:\n"
        "                shipping = 50.0 if is_express else 30.0\n"
        "            total = max(0.0, after_discount - coupon) + shipping\n"
        "            try:\n"
        "                ORDER_LOG.append((user, round(total, 2)))\n"
        "            except Exception:\n"
        "                pass\n"
        "            return round(total, 2)\n"
        "        else:\n"
        "            return 0.0\n"
        "    else:\n"
        "        return 0.0\n\n"
        "def format_cn_invoice(amount, rate):\n"
        "    tax = amount * rate\n"
        "    gross = amount + tax\n"
        "    rounded = round(gross, 2)\n"
        "    return f'CN:{rounded}'\n\n"
        "def format_us_invoice(price, tax_rate):\n"
        "    fee = price * tax_rate\n"
        "    total = price + fee\n"
        "    final = round(total, 2)\n"
        "    return f'US:{final}'\n",
        encoding="utf-8",
    )
    (root / "test_order_processor.py").write_text(
        "from order_processor import format_cn_invoice, format_us_invoice, process_order\n\n"
        "def test_empty_user_or_items():\n"
        "    assert process_order('', [{'price': 10, 'qty': 1}], 'CN', 'gold', 0, False) == 0.0\n"
        "    assert process_order('u1', [], 'CN', 'gold', 0, False) == 0.0\n\n"
        "def test_gold_cn_standard():\n"
        "    items = [{'price': 50, 'qty': 2}, {'price': -5, 'qty': 1}]\n"
        "    # subtotal=100, gold 0.85 -> 85, coupon 5 -> 80, shipping 10 -> 90\n"
        "    assert process_order('u1', items, 'CN', 'gold', 5, False) == 90.0\n\n"
        "def test_diamond_free_shipping():\n"
        "    items = [{'price': 400, 'qty': 1}]\n"
        "    # subtotal=400, diamond 0.75 -> 300 >= 200 -> free shipping\n"
        "    assert process_order('u2', items, 'CN', 'diamond', 20, True) == 280.0\n\n"
        "def test_overseas_express():\n"
        "    items = [{'price': 100, 'qty': 1}]\n"
        "    assert process_order('u3', items, 'US', 'none', 0, True) == 150.0\n\n"
        "def test_invoices():\n"
        "    assert format_cn_invoice(100, 0.1) == 'CN:110.0'\n"
        "    assert format_us_invoice(100, 0.08) == 'US:108.0'\n",
        encoding="utf-8",
    )


async def run_review(repo_path: Path, *, top_k: int, do_refactor: bool, use_docker: bool) -> None:
    print(f"[1/3] 正在对 {repo_path} 执行确定性静态扫描（AST 检测器 + radon + ruff）...")
    findings, files_scanned, churn = scan_repository(repo_path)
    hotspots = rank_hotspots(findings, churn, top_k=top_k)
    print(f"      扫描 {files_scanned} 个文件，共发现 {len(findings)} 条确定性问题，提取 Top {len(hotspots)} 热点。")

    llm = OpenAICompatClient(LLMSettings())
    if hotspots:
        print(f"[2/3] 正在并发调用 LLM（JSON Mode + Pydantic 约束）研判 Top {len(hotspots)} 热点...")
        await judge_hotspots(hotspots, repo_path, llm, max_concurrency=3)

    report = ReviewReport(
        repo_name=repo_path.name,
        files_scanned=files_scanned,
        total_findings=len(findings),
        hotspots=hotspots,
    )
    out_dir = Path("runs")
    md_path, json_path = save_report(report, out_dir)
    print(f"[3/3] 诊断报告已生成：\n      - {md_path.resolve()}\n      - {json_path.resolve()}\n")
    print("=" * 20 + " 报告预览 " + "=" * 20)
    print(md_path.read_text(encoding="utf-8")[:2200])

    if do_refactor and hotspots:
        top = hotspots[0]
        print("\n" + "=" * 20 + f" 启动测试护航安全重构: {top.file}::{top.symbol} " + "=" * 20)
        store = TraceStore(out_dir / "traces.db")
        sandbox = DockerSandbox(repo_path) if use_docker else LocalSandbox(repo_path)
        async with sandbox as sb:
            res = await safe_refactor_hotspot(top, llm=llm, sandbox=sb, tracer=Tracer(store))
        store.close()

        print(f"重构状态: {res.run_result.status} ({res.run_result.exit_reason}) | steps={res.run_result.total_steps}")
        if res.before and res.after:
            print(
                f"指标对比 `{top.symbol}`:\n"
                f"  - 圈复杂度 CC : {res.before.cc} → {res.after.cc}\n"
                f"  - 函数行数    : {res.before.lines} → {res.after.lines}\n"
                f"  - 最大嵌套深度: {res.before.nesting} → {res.after.nesting}"
            )
        print("\n=== 重构 git diff ===")
        print(res.run_result.final_patch or "(无改动)")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="", help="指定要诊断的本地 Python 项目路径（留空则用内置屎山样例）")
    parser.add_argument("--top-k", type=int, default=5, help="研判的热点数量")
    parser.add_argument("--refactor", action="store_true", help="对排名第 1 的热点执行测试护航安全重构")
    parser.add_argument("--docker", action="store_true", help="重构时使用 DockerSandbox")
    args = parser.parse_args()

    if args.repo:
        await run_review(Path(args.repo).resolve(), top_k=args.top_k, do_refactor=args.refactor, use_docker=args.docker)
    else:
        with tempfile.TemporaryDirectory(prefix="mca_l6_shit_") as tmp:
            p = Path(tmp) / "legacy_orders"
            p.mkdir()
            create_shit_mountain_repo(p)
            await run_review(p, top_k=args.top_k, do_refactor=args.refactor, use_docker=args.docker)


if __name__ == "__main__":
    asyncio.run(main())
