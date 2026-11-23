"""评测 Harness 运行器：asyncio 并发执行 + 隐藏测试裁决 + 消融实验对比表生成。"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from mca.agent import AgentLoop, Budget, ContextConfig, ContextManager, LoopGuard
from mca.harness.tasks import BenchTask
from mca.llm.client import LLMClient
from mca.observability import TraceStore, Tracer
from mca.sandbox import DockerSandbox, LocalSandbox, Sandbox
from mca.tools import ToolRegistry, fix_tools


@dataclass(frozen=True)
class AblationConfig:
    """消融实验配置：通过开关精确控制单一变量。"""

    name: str
    include_ast: bool = True
    inject_repo_map: bool = True
    mask_enabled: bool = True
    summary_enabled: bool = True
    loop_guard_enabled: bool = True
    max_steps: int = 15
    max_tokens: int = 80_000


PRESET_CONFIGS: dict[str, AblationConfig] = {
    "full": AblationConfig(name="full"),
    "no_ast": AblationConfig(name="no_ast", include_ast=False, inject_repo_map=False),
    "no_compression": AblationConfig(name="no_compression", mask_enabled=False, summary_enabled=False),
    "no_guard": AblationConfig(name="no_guard", loop_guard_enabled=False),
}


@dataclass
class TaskEvalResult:
    config_name: str
    task_id: str
    category: str
    run_id: str
    agent_status: str
    exit_reason: str
    hidden_passed: bool  # 最终金标准：隐藏测试是否全部通过！
    steps: int
    prompt_tokens: int
    completion_tokens: int
    wall_time_ms: float
    recover_count: int


async def evaluate_task(
    task: BenchTask,
    cfg: AblationConfig,
    *,
    llm_factory: Callable[[], LLMClient],
    use_docker: bool = False,
    tracer: Tracer | None = None,
) -> TaskEvalResult:
    with tempfile.TemporaryDirectory(prefix=f"mca_bench_{task.id}_") as src_dir:
        src_path = Path(src_dir)
        task.materialize(src_path, include_hidden=False)

        sandbox: Sandbox = DockerSandbox(src_path) if use_docker else LocalSandbox(src_path)
        async with sandbox as sb:
            loop = AgentLoop(
                llm=llm_factory(),
                registry=ToolRegistry(fix_tools(include_ast=cfg.include_ast)),
                sandbox=sb,
                budget=Budget(max_steps=cfg.max_steps, max_tokens=cfg.max_tokens),
                loop_guard=LoopGuard(enabled=cfg.loop_guard_enabled),
                context_manager=ContextManager(
                    ContextConfig(
                        mask_enabled=cfg.mask_enabled,
                        summary_enabled=cfg.summary_enabled,
                        keep_recent_turns=3,
                        min_mask_chars=300,
                    )
                ),
                inject_repo_map=cfg.inject_repo_map,
                tracer=tracer,
            )
            res = await loop.run(task.issue, task_id=f"{cfg.name}:{task.id}", mode="fix")

            # 注入隐藏测试做最终独立裁决（Agent 运行期间完全看不到这些测试！）
            for rel, content in task.hidden_tests.items():
                p = sb.workspace / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")

            hidden_res = await sb.exec(f"{sb.python_bin} -m pytest -q --tb=no -p no:cacheprovider", timeout_s=60.0)
            hidden_passed = hidden_res.exit_code == 0 and not hidden_res.timed_out

    return TaskEvalResult(
        config_name=cfg.name,
        task_id=task.id,
        category=task.category,
        run_id=res.run_id,
        agent_status=res.status,
        exit_reason=res.exit_reason,
        hidden_passed=hidden_passed,
        steps=res.total_steps,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        wall_time_ms=res.wall_time_ms,
        recover_count=res.recover_count,
    )


async def run_benchmark(
    tasks: list[BenchTask],
    configs: list[AblationConfig],
    *,
    llm_factory: Callable[[], LLMClient],
    concurrency: int = 2,
    use_docker: bool = False,
    db_path: Path | str = "runs/traces.db",
) -> list[TaskEvalResult]:
    """用 asyncio.Semaphore 并发运行 (config × task) 矩阵。"""
    store = TraceStore(db_path)
    tracer = Tracer(store)
    sem = asyncio.Semaphore(max(1, concurrency))
    all_results: list[TaskEvalResult] = []

    async def _worker(cfg: AblationConfig, task: BenchTask) -> TaskEvalResult:
        async with sem:
            r = await evaluate_task(task, cfg, llm_factory=llm_factory, use_docker=use_docker, tracer=tracer)
            mark = "PASS" if r.hidden_passed else "FAIL"
            print(
                f"  [{cfg.name:<14}] {task.id:<22} -> {mark} "
                f"(exit={r.exit_reason:<12} steps={r.steps:>2} tok={r.prompt_tokens + r.completion_tokens:>6})"
            )
            return r

    coros = [_worker(cfg, t) for cfg in configs for t in tasks]
    all_results = list(await asyncio.gather(*coros))
    store.close()
    return all_results


def render_ablation_markdown(results: list[TaskEvalResult]) -> str:
    """将评测结果汇总为可直接贴进 README 和简历的消融实验 Markdown 对比表。"""
    by_cfg: dict[str, list[TaskEvalResult]] = {}
    for r in results:
        by_cfg.setdefault(r.config_name, []).append(r)

    lines = [
        "# MiniBench 消融实验报告",
        "",
        f"- **生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **总运行次数**：{len(results)}",
        "",
        "## 一、配置对比汇总（按隐藏测试通过率 Resolve Rate）",
        "",
        "| 配置 | 任务数 | 隐藏测试通过率 | 平均步数 | 平均总 Token | 平均耗时(s) | 触发恢复次数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg_name, items in by_cfg.items():
        n = len(items)
        passed = sum(1 for i in items if i.hidden_passed)
        rate = (passed / n * 100) if n else 0.0
        avg_steps = sum(i.steps for i in items) / n
        avg_tok = sum(i.prompt_tokens + i.completion_tokens for i in items) / n
        avg_s = sum(i.wall_time_ms for i in items) / n / 1000.0
        recovers = sum(i.recover_count for i in items)
        lines.append(
            f"| `{cfg_name}` | {n} | **{passed}/{n} ({rate:.1f}%)** | {avg_steps:.1f} "
            f"| {avg_tok:.0f} | {avg_s:.1f} | {recovers} |"
        )

    lines.extend(
        [
            "",
            "## 二、逐任务明细",
            "",
            "| 配置 | 任务 | 类别 | 隐藏测试 | 退出原因 | 步数 | 总 Token | Run ID |",
            "|---|---|---|:-:|---|---:|---:|---|",
        ]
    )
    for r in results:
        mark = "✅" if r.hidden_passed else "❌"
        lines.append(
            f"| `{r.config_name}` | `{r.task_id}` | {r.category} | {mark} "
            f"| `{r.exit_reason}` | {r.steps} | {r.prompt_tokens + r.completion_tokens} | `{r.run_id}` |"
        )
    return "\n".join(lines)


def save_benchmark_results(results: list[TaskEvalResult], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"ablation_{ts}.md"
    json_path = out_dir / f"ablation_{ts}.json"
    md_path.write_text(render_ablation_markdown(results), encoding="utf-8")
    json_path.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, json_path
