"""MiniCodeAgent 统一命令行入口（`mca`）。

子命令：
  mca fix <repo> --issue "..." [--docker] [--no-ast] [--no-mask]
  mca review <repo> [--top-k 5] [--refactor] [--docker]
  mca trace list | show <run_id> | stats
  mca bench [--configs full,no_ast] [--concurrency 2] [--limit 6] [--docker]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from mca.agent import AgentLoop, Budget, ContextConfig, ContextManager, LoopGuard
from mca.config import LLMSettings
from mca.harness import (
    PRESET_CONFIGS,
    builtin_tasks,
    fetch_swebench_lite,
    run_benchmark,
    run_swebench,
    save_benchmark_results,
    save_swebench_outputs,
)
from mca.llm import OpenAICompatClient
from mca.observability import TraceStore, Tracer, format_run_replay, format_tool_stats
from mca.review import ReviewReport, judge_hotspots, rank_hotspots, safe_refactor_hotspot, save_report, scan_repository
from mca.sandbox import DockerSandbox, LocalSandbox
from mca.tools import ToolRegistry, fix_tools

DEFAULT_DB = Path("runs/traces.db")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mca", description="MiniCodeAgent — 自研代码诊断与修复 Agent 引擎")
    p.add_argument("--db", type=str, default=str(DEFAULT_DB), help="SQLite Trace 数据库路径")
    sub = p.add_subparsers(dest="command", required=True)

    # ---- fix ----
    p_fix = sub.add_parser("fix", help="在隔离沙箱中自主定位并修复仓库 Bug")
    p_fix.add_argument("repo", type=str, help="目标 Python 仓库路径")
    p_fix.add_argument("--issue", "-i", type=str, required=True, help="Bug 描述")
    p_fix.add_argument("--docker", action="store_true", help="使用加固 DockerSandbox")
    p_fix.add_argument("--max-steps", type=int, default=15)
    p_fix.add_argument("--no-ast", action="store_true", help="禁用 AST 工具与 Repo Map")
    p_fix.add_argument("--no-mask", action="store_true", help="禁用上下文观察折叠")

    # ---- review ----
    p_rev = sub.add_parser("review", help="对“屎山代码”做静态扫描、LLM 研判与可选安全重构")
    p_rev.add_argument("repo", type=str, help="目标 Python 仓库路径")
    p_rev.add_argument("--top-k", type=int, default=5, help="研判热点数量")
    p_rev.add_argument("--refactor", action="store_true", help="对排名第一的热点执行测试护航安全重构")
    p_rev.add_argument("--docker", action="store_true", help="重构时使用 DockerSandbox")
    p_rev.add_argument("--out", type=str, default="runs", help="报告输出目录")

    # ---- trace ----
    p_tr = sub.add_parser("trace", help="查看 SQLite 全链路 Trace 与 SQL 指标统计")
    tr_sub = p_tr.add_subparsers(dest="trace_cmd", required=True)
    tr_sub.add_parser("list", help="列出最近的运行记录")
    p_show = tr_sub.add_parser("show", help="回放某次运行的完整调用链")
    p_show.add_argument("run_id", type=str)
    tr_sub.add_parser("stats", help="打印工具失败率/耗时与退出原因 SQL 统计")

    # ---- bench ----
    p_bench = sub.add_parser("bench", help="运行 MiniBench 评测与并发消融实验")
    p_bench.add_argument(
        "--configs",
        type=str,
        default="full,no_ast",
        help=f"逗号分隔的消融配置（可选: {', '.join(PRESET_CONFIGS)}）",
    )
    p_bench.add_argument("--concurrency", type=int, default=2, help="asyncio 最大并发任务数")
    p_bench.add_argument("--limit", type=int, default=0, help="只跑前 N 道题（0 表示全部）")
    p_bench.add_argument("--docker", action="store_true", help="使用 DockerSandbox")
    p_bench.add_argument("--out", type=str, default="bench/results", help="消融报告输出目录")

    # ---- swebench ----
    p_swe = sub.add_parser("swebench", help="在 Docker 沙箱中运行真实 SWE-bench Lite 子集评测")
    p_swe.add_argument("--repo", type=str, default="pallets/flask", help="目标仓库（默认 pallets/flask）")
    p_swe.add_argument("--ids", type=str, default="", help="逗号分隔的 instance_id（留空按 --limit 自动挑选）")
    p_swe.add_argument("--limit", type=int, default=2, help="抽取题数（默认 2）")
    p_swe.add_argument("--configs", type=str, default="full", help=f"逗号分隔配置（可选: {', '.join(PRESET_CONFIGS)}）")
    p_swe.add_argument("--image", type=str, default="mca-swebench-flask:3.11", help="Docker 评测镜像名")
    p_swe.add_argument("--concurrency", type=int, default=1, help="并发容器数")
    p_swe.add_argument("--out", type=str, default="bench/results", help="报告与 preds.jsonl 输出目录")

    return p


async def _cmd_fix(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    store = TraceStore(args.db)
    sandbox = DockerSandbox(repo) if args.docker else LocalSandbox(repo)
    async with sandbox as sb:
        loop = AgentLoop(
            llm=OpenAICompatClient(LLMSettings()),
            registry=ToolRegistry(fix_tools(include_ast=not args.no_ast)),
            sandbox=sb,
            budget=Budget(max_steps=args.max_steps),
            loop_guard=LoopGuard(enabled=True),
            context_manager=ContextManager(ContextConfig(mask_enabled=not args.no_mask)),
            inject_repo_map=not args.no_ast,
            tracer=Tracer(store),
        )
        res = await loop.run(args.issue, task_id=repo.name, mode="fix")
    print(format_run_replay(store, res.run_id))
    print("\n=== Final git diff ===\n" + (res.final_patch or "(无改动)"))
    store.close()


async def _cmd_review(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    findings, files_scanned, churn = scan_repository(repo)
    hotspots = rank_hotspots(findings, churn, top_k=args.top_k)
    print(f"扫描 {files_scanned} 个文件，发现 {len(findings)} 条静态问题，提取 Top {len(hotspots)} 热点。")

    llm = OpenAICompatClient(LLMSettings())
    if hotspots:
        await judge_hotspots(hotspots, repo, llm, max_concurrency=3)

    report = ReviewReport(repo.name, files_scanned, len(findings), hotspots)
    md_path, json_path = save_report(report, Path(args.out))
    print(f"诊断报告已保存：\n  - {md_path.resolve()}\n  - {json_path.resolve()}")

    if args.refactor and hotspots:
        top = hotspots[0]
        store = TraceStore(args.db)
        sandbox = DockerSandbox(repo) if args.docker else LocalSandbox(repo)
        async with sandbox as sb:
            res = await safe_refactor_hotspot(top, llm=llm, sandbox=sb, tracer=Tracer(store))
        store.close()
        print(f"\n重构状态: {res.run_result.status} ({res.run_result.exit_reason}) | steps={res.run_result.total_steps}")
        if res.before and res.after:
            print(
                f"指标对比 `{top.symbol}`: CC {res.before.cc}→{res.after.cc} | "
                f"行数 {res.before.lines}→{res.after.lines} | 嵌套 {res.before.nesting}→{res.after.nesting}"
            )
        print("\n=== 重构 git diff ===\n" + (res.run_result.final_patch or "(无改动)"))


def _cmd_trace(args: argparse.Namespace) -> None:
    store = TraceStore(args.db)
    if args.trace_cmd == "list":
        rows = store.list_runs(limit=20)
        if not rows:
            print("(暂无运行记录)")
        else:
            print(f"{'Run ID':<14} {'Mode':<7} {'Status':<8} {'Exit Reason':<14} {'Steps':>5} {'Tokens':>8}  Task")
            print("-" * 78)
            for r in rows:
                tok = r["prompt_tokens"] + r["completion_tokens"]
                print(
                    f"{r['id']:<14} {r['mode']:<7} {r['status']:<8} "
                    f"{(r['exit_reason'] or '-'):<14} {r['total_steps']:>5} {tok:>8}  {r['task_id']}"
                )
    elif args.trace_cmd == "show":
        print(format_run_replay(store, args.run_id))
    elif args.trace_cmd == "stats":
        print("=== 工具调用统计 ===")
        print(format_tool_stats(store))
        print("\n=== 退出原因统计 ===")
        for r in store.exit_reason_stats():
            print(
                f"  {r['status']:<7} {str(r['exit_reason']):<15} runs={r['runs']:>3} "
                f"avg_steps={r['avg_steps']} avg_tokens={int(r['avg_tokens'])}"
            )
    store.close()


async def _cmd_bench(args: argparse.Namespace) -> None:
    cfg_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    configs = []
    for name in cfg_names:
        if name not in PRESET_CONFIGS:
            raise SystemExit(f"未知配置 '{name}'，可选: {', '.join(PRESET_CONFIGS)}")
        configs.append(PRESET_CONFIGS[name])

    tasks = builtin_tasks()
    if args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"启动 MiniBench：{len(configs)} 组配置 × {len(tasks)} 道题，并发数={args.concurrency}...")
    settings = LLMSettings()
    results = await run_benchmark(
        tasks,
        configs,
        llm_factory=lambda: OpenAICompatClient(settings),
        concurrency=args.concurrency,
        use_docker=args.docker,
        db_path=args.db,
    )
    md_path, json_path = save_benchmark_results(results, Path(args.out))
    print("\n" + md_path.read_text(encoding="utf-8"))
    print(f"\n结果已保存：\n  - {md_path.resolve()}\n  - {json_path.resolve()}")


async def _cmd_swebench(args: argparse.Namespace) -> None:
    cfg_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    configs = []
    for name in cfg_names:
        if name not in PRESET_CONFIGS:
            raise SystemExit(f"未知配置 '{name}'，可选: {', '.join(PRESET_CONFIGS)}")
        # 真实大仓库适当放宽步数与 Token 预算
        base = PRESET_CONFIGS[name]
        configs.append(
            base.__class__(
                name=base.name,
                include_ast=base.include_ast,
                inject_repo_map=base.inject_repo_map,
                mask_enabled=base.mask_enabled,
                summary_enabled=base.summary_enabled,
                loop_guard_enabled=base.loop_guard_enabled,
                max_steps=20,
                max_tokens=250_000,
            )
        )

    ids = [x.strip() for x in args.ids.split(",") if x.strip()] or None
    instances = fetch_swebench_lite(repo=args.repo, instance_ids=ids, limit=args.limit)
    print(f"加载 SWE-bench Lite 真实任务 {len(instances)} 题: {[i.instance_id for i in instances]}")
    print(f"使用 Docker 镜像 `{args.image}`，配置={[c.name for c in configs]}，并发数={args.concurrency}...")

    settings = LLMSettings()
    results = await run_swebench(
        instances,
        configs,
        llm_factory=lambda: OpenAICompatClient(settings),
        image=args.image,
        concurrency=args.concurrency,
        db_path=args.db,
    )
    preds_path, md_path = save_swebench_outputs(results, Path(args.out))
    print("\n" + md_path.read_text(encoding="utf-8"))
    print(f"\n已保存：\n  - 标准 preds.jsonl: {preds_path.resolve()}\n  - 评测报告: {md_path.resolve()}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fix":
        asyncio.run(_cmd_fix(args))
    elif args.command == "review":
        asyncio.run(_cmd_review(args))
    elif args.command == "trace":
        _cmd_trace(args)
    elif args.command == "bench":
        asyncio.run(_cmd_bench(args))
    elif args.command == "swebench":
        asyncio.run(_cmd_swebench(args))


if __name__ == "__main__":
    main()
