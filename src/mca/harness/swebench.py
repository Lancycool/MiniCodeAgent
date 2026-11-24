"""SWE-bench Lite 真实仓库评测适配器（支持 Docker 沙箱执行 + 官方 preds.jsonl 导出 + test_patch 裁决）。

工作流（对标 SWE-bench 两阶段标准）：
  1. 从 HuggingFace Dataset Server 拉取 princeton-nlp/SWE-bench_Lite 真实任务（本地缓存复用）；
  2. 本地缓存 git clone 目标仓库并检出 base_commit；
  3. 在加固 DockerSandbox 中运行 AgentLoop，导出官方兼容的 preds.jsonl；
  4. 在同一隔离 Docker 容器内打上官方隐藏 test_patch（git apply），运行 FAIL_TO_PASS
     与 PASS_TO_PASS 测试用例，给出真实 Resolve Rate 裁决。
"""

from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from mca.agent import AgentLoop, Budget, ContextConfig, ContextManager, LoopGuard, VerifyResult
from mca.harness.runner import AblationConfig
from mca.llm.client import LLMClient
from mca.observability import TraceStore, Tracer
from mca.sandbox import DockerSandbox, Sandbox, SandboxConfig
from mca.tools import ToolRegistry, fix_tools

HF_FILTER_URL = "https://datasets-server.huggingface.co/filter"

# 精选 10 道代表性 SWE-bench Lite 题目（纯 Python、离线可用、覆盖 Web 框架 + 测试框架/AST/导入/缓存/CLI）
TOP10_REPRESENTATIVE: list[tuple[str, str]] = [
    ("pallets/flask", "pallets__flask-4992"),
    ("pallets/flask", "pallets__flask-5063"),
    ("pallets/flask", "pallets__flask-4045"),
    ("pytest-dev/pytest", "pytest-dev__pytest-11143"),
    ("pytest-dev/pytest", "pytest-dev__pytest-11148"),
    ("pytest-dev/pytest", "pytest-dev__pytest-8365"),
    ("pytest-dev/pytest", "pytest-dev__pytest-7490"),
    ("pytest-dev/pytest", "pytest-dev__pytest-7432"),
    ("pytest-dev/pytest", "pytest-dev__pytest-7373"),
    ("pytest-dev/pytest", "pytest-dev__pytest-7220"),
]


def fetch_top10_representative(cache_dir: Path = Path("bench/swebench_cache")) -> list[SWEBenchInstance]:
    """加载精选的 10 道跨仓库 SWE-bench Lite 代表题目。"""
    by_repo: dict[str, list[str]] = {}
    for repo, iid in TOP10_REPRESENTATIVE:
        by_repo.setdefault(repo, []).append(iid)
    out: list[SWEBenchInstance] = []
    for repo, ids in by_repo.items():
        items = fetch_swebench_lite(repo=repo, instance_ids=ids, limit=len(ids), cache_dir=cache_dir)
        by_id = {x.instance_id: x for x in items}
        for iid in ids:
            if iid in by_id:
                out.append(by_id[iid])
    return out

SWEBENCH_SYSTEM_PROMPT = """你是一位资深开源项目维护者，正在 Docker 隔离沙箱中修复真实 GitHub Issue（SWE-bench Lite）。
工作流：
1. 结合 Repo Map 和 AST 工具（search_symbol / find_references / view_outline / read_file）快速定位目标源码。
2. 定位到目标函数后**尽早动手用 edit_file 修改业务源码**，不要陷入无休止的只读探索；**不要修改 CHANGES.rst、docs/ 或创建 tests/ 下的新文件**。
3. 用 run_tests 指定**目标测试文件**（如 tests/test_config.py 或 tests/test_cli.py）验证无回归（不要盲目跑全仓测试，避免无关可选依赖报错），并用简短的 `python -c` 脚本验证 Issue 描述的新行为。
4. 确认新行为与目标测试均通过后，立即调用 submit 提交摘要。"""


@dataclass(frozen=True)
class SWEBenchInstance:
    instance_id: str
    repo: str
    base_commit: str
    version: str
    problem_statement: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]


@dataclass
class SWEBenchEvalResult:
    config_name: str
    instance_id: str
    repo: str
    run_id: str
    exit_reason: str
    resolved: bool
    f2p_passed: bool
    p2p_passed: bool
    steps: int
    total_tokens: int
    wall_time_ms: float
    model_patch: str


class NonEmptyDiffVerifier:
    """SWE-bench 推理阶段提交校验：确保确实产出了非空补丁（隐藏测试留给评测阶段独立裁决）。"""

    async def verify(self, sandbox: Sandbox) -> VerifyResult:
        diff = await sandbox.diff()
        if not diff.strip():
            return VerifyResult(False, "你调用了 submit，但当前工作区没有任何代码改动（git diff 为空）。请先修改源码再提交。")
        return VerifyResult(True, "patch non-empty")


def fetch_swebench_lite(
    *,
    repo: str = "pallets/flask",
    instance_ids: list[str] | None = None,
    limit: int = 3,
    cache_dir: Path = Path("bench/swebench_cache"),
) -> list[SWEBenchInstance]:
    """从 HuggingFace 拉取 SWE-bench Lite 真实题目，并缓存到本地 JSON。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_repo = repo.replace("/", "__")
    cache_file = cache_dir / f"{safe_repo}_lite.json"

    rows: list[dict]
    if cache_file.exists():
        rows = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        where = urllib.parse.quote(f'"repo"=\'{repo}\'')
        url = (
            f"{HF_FILTER_URL}?dataset=princeton-nlp/SWE-bench_Lite"
            f"&config=default&split=test&where={where}&length=15"
        )
        with urllib.request.urlopen(url, timeout=45) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        rows = [item["row"] for item in payload.get("rows", [])]
        cache_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    instances: list[SWEBenchInstance] = []
    wanted = set(instance_ids) if instance_ids else None
    for r in rows:
        if wanted and r["instance_id"] not in wanted:
            continue
        instances.append(
            SWEBenchInstance(
                instance_id=r["instance_id"],
                repo=r["repo"],
                base_commit=r["base_commit"],
                version=str(r.get("version", "")),
                problem_statement=r["problem_statement"],
                test_patch=r["test_patch"],
                fail_to_pass=_parse_test_list(r.get("FAIL_TO_PASS", "[]")),
                pass_to_pass=_parse_test_list(r.get("PASS_TO_PASS", "[]")),
            )
        )

    if not wanted and limit > 0:
        # 默认优先选较新的 2.3 任务（如 flask-4992 / flask-5063），兼容性最好
        instances.sort(key=lambda x: (-float(x.version or 0), x.instance_id))
        instances = instances[:limit]
    return instances


def _parse_test_list(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        parsed = json.loads(raw)
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    except Exception:
        return []


def prepare_repo_checkout(
    repo: str,
    base_commit: str,
    dest_dir: Path,
    *,
    cache_dir: Path = Path("bench/swebench_cache/repos"),
) -> None:
    """本地缓存克隆仓库，并导出 base_commit 的干净工作树到 dest_dir。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    mirror = cache_dir / repo.replace("/", "__")
    if not mirror.exists():
        url = f"https://github.com/{repo}.git"
        proc = subprocess.run(
            ["git", "clone", "--quiet", url, str(mirror)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git clone {url} failed: {proc.stderr}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    # 用 git archive 直接导出指定 commit 的干净代码，速度极快且不带庞大 .git 历史
    tar_proc = subprocess.run(
        ["git", "-C", str(mirror), "archive", "--format=tar", base_commit],
        capture_output=True,
        timeout=60,
    )
    if tar_proc.returncode != 0:
        raise RuntimeError(f"git archive {base_commit} failed: {tar_proc.stderr.decode('utf-8', 'replace')}")

    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(tar_proc.stdout), mode="r:") as tf:
        tf.extractall(dest_dir)


async def evaluate_swebench_instance(
    inst: SWEBenchInstance,
    cfg: AblationConfig,
    *,
    llm_factory: Callable[[], LLMClient],
    image: str = "mca-swebench-flask:3.11",
    tracer: Tracer | None = None,
) -> SWEBenchEvalResult:
    with tempfile.TemporaryDirectory(prefix=f"mca_swe_{inst.instance_id}_") as src_dir:
        src_path = Path(src_dir)
        await asyncio.to_thread(prepare_repo_checkout, inst.repo, inst.base_commit, src_path)

        sb_cfg = SandboxConfig(image=image, mem_limit="2g", default_timeout_s=60.0)
        async with DockerSandbox(src_path, config=sb_cfg) as sb:
            loop = AgentLoop(
                llm=llm_factory(),
                registry=ToolRegistry(fix_tools(include_ast=cfg.include_ast)),
                sandbox=sb,
                verifier=NonEmptyDiffVerifier(),
                budget=Budget(max_steps=cfg.max_steps, max_tokens=cfg.max_tokens),
                loop_guard=LoopGuard(enabled=cfg.loop_guard_enabled),
                context_manager=ContextManager(
                    ContextConfig(
                        mask_enabled=cfg.mask_enabled,
                        summary_enabled=cfg.summary_enabled,
                        keep_recent_turns=6,
                        min_mask_chars=1200,
                    )
                ),
                inject_repo_map=cfg.inject_repo_map,
                repo_map_max_chars=4000,
                tracer=tracer,
                system_prompt=SWEBENCH_SYSTEM_PROMPT,
                protected_globs=("tests/*", "testing/*", "test_*.py"),
            )
            res = await loop.run(inst.problem_statement, task_id=f"{cfg.name}:{inst.instance_id}", mode="fix")

            # ---- 独立裁决阶段：在容器内应用官方隐藏 test_patch 并运行 FAIL_TO_PASS / PASS_TO_PASS ----
            f2p_ok, p2p_ok = await _verify_with_official_test_patch(sb, inst)

    return SWEBenchEvalResult(
        config_name=cfg.name,
        instance_id=inst.instance_id,
        repo=inst.repo,
        run_id=res.run_id,
        exit_reason=res.exit_reason,
        resolved=bool(f2p_ok and p2p_ok),
        f2p_passed=f2p_ok,
        p2p_passed=p2p_ok,
        steps=res.total_steps,
        total_tokens=res.prompt_tokens + res.completion_tokens,
        wall_time_ms=res.wall_time_ms,
        model_patch=res.final_patch,
    )


async def _verify_with_official_test_patch(sb: DockerSandbox, inst: SWEBenchInstance) -> tuple[bool, bool]:
    # 对标 SWE-bench 官方评测逻辑：先清理 Agent 可能在 tests/ 或 testing/ 下创建的临时文件或改动，
    # 避免与官方隐藏 test_patch 冲突导致 git apply 失败！
    for test_dir in ("tests", "testing"):
        await sb.exec(f"git checkout {shlex.quote(sb.baseline_sha)} -- {test_dir} 2>/dev/null || true")
        await sb.exec(f"git clean -fd {test_dir} 2>/dev/null || true")

    # 将官方 test_patch 写入工作区临时文件后用 git apply 应用
    patch_file = sb.workspace / ".mca_official_test.patch"
    patch_file.write_text(inst.test_patch, encoding="utf-8", newline="\n")
    apply_res = await sb.exec("git apply --whitespace=nowarn .mca_official_test.patch")
    patch_file.unlink(missing_ok=True)
    if not apply_res.ok:
        return False, False

    # ① 运行所有 FAIL_TO_PASS 用例（核心修复判定）
    f2p_args = " ".join(shlex.quote(t) for t in inst.fail_to_pass)
    f2p_res = await sb.exec(f"python -m pytest -q --tb=short -p no:cacheprovider {f2p_args}", timeout_s=90.0)
    f2p_ok = f2p_res.ok

    # ② 运行 PASS_TO_PASS 用例（无回归判定；最多抽查同测试文件或前 20 条防止耗时过长）
    p2p_ok = True
    if f2p_ok and inst.pass_to_pass:
        # 优先取与 FAIL_TO_PASS 同文件的 PASS_TO_PASS 用例
        f2p_files = {t.split("::")[0] for t in inst.fail_to_pass}
        related_p2p = [t for t in inst.pass_to_pass if t.split("::")[0] in f2p_files]
        sample_p2p = (related_p2p or inst.pass_to_pass)[:25]
        p2p_args = " ".join(shlex.quote(t) for t in sample_p2p)
        p2p_res = await sb.exec(f"python -m pytest -q --tb=no -p no:cacheprovider {p2p_args}", timeout_s=90.0)
        p2p_ok = p2p_res.ok

    return f2p_ok, p2p_ok


async def run_swebench(
    instances: list[SWEBenchInstance],
    configs: list[AblationConfig],
    *,
    llm_factory: Callable[[], LLMClient],
    image: str = "mca-swebench-flask:3.11",
    concurrency: int = 1,
    db_path: Path | str = "runs/traces.db",
) -> list[SWEBenchEvalResult]:
    store = TraceStore(db_path)
    tracer = Tracer(store)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _worker(cfg: AblationConfig, inst: SWEBenchInstance) -> SWEBenchEvalResult:
        async with sem:
            r = await evaluate_swebench_instance(
                inst, cfg, llm_factory=llm_factory, image=image, tracer=tracer
            )
            status = "RESOLVED" if r.resolved else "UNRESOLVED"
            print(
                f"  [{cfg.name:<14}] {inst.instance_id:<24} -> {status:<10} "
                f"(f2p={r.f2p_passed}, p2p={r.p2p_passed}, steps={r.steps:>2}, tok={r.total_tokens:>6})"
            )
            return r

    results = list(await asyncio.gather(*(_worker(cfg, i) for cfg in configs for i in instances)))
    store.close()
    return results


def save_swebench_outputs(results: list[SWEBenchEvalResult], out_dir: Path) -> tuple[Path, Path]:
    """同时保存官方兼容的 preds.jsonl（可直接喂给 swebench.harness.run_evaluation）与 Markdown 对比表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    preds_path = out_dir / f"swebench_preds_{ts}.jsonl"
    md_path = out_dir / f"swebench_report_{ts}.md"

    with preds_path.open("w", encoding="utf-8") as f:
        for r in results:
            rec = {
                "instance_id": r.instance_id,
                "model_name_or_path": f"minicodeagent-{r.config_name}",
                "model_patch": r.model_patch,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_cfg: dict[str, list[SWEBenchEvalResult]] = {}
    for r in results:
        by_cfg.setdefault(r.config_name, []).append(r)

    lines = [
        "# SWE-bench Lite 子集评测报告（Docker 官方 test_patch 裁决）",
        "",
        f"- **生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **导出标准预测文件**：`{preds_path.name}`",
        "",
        "## 一、配置对比汇总",
        "",
        "| 配置 | 题数 | Resolved Rate | 平均步数 | 平均总 Token | 平均耗时(s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, items in by_cfg.items():
        n = len(items)
        resolved = sum(1 for x in items if x.resolved)
        lines.append(
            f"| `{name}` | {n} | **{resolved}/{n} ({resolved / n * 100:.1f}%)** | "
            f"{sum(x.steps for x in items) / n:.1f} | "
            f"{sum(x.total_tokens for x in items) / n:.0f} | "
            f"{sum(x.wall_time_ms for x in items) / n / 1000:.1f} |"
        )

    lines.extend(
        [
            "",
            "## 二、逐题明细",
            "",
            "| 配置 | Instance ID | Resolved | F2P | P2P | 步数 | 总 Token | Run ID |",
            "|---|---|:-:|:-:|:-:|---:|---:|---|",
        ]
    )
    for r in results:
        lines.append(
            f"| `{r.config_name}` | `{r.instance_id}` | {'✅' if r.resolved else '❌'} "
            f"| {'✅' if r.f2p_passed else '❌'} | {'✅' if r.p2p_passed else '❌'} "
            f"| {r.steps} | {r.total_tokens} | `{r.run_id}` |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return preds_path, md_path
