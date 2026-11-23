"""Review 模式第 4 阶段：测试护航的安全重构（Safe Refactoring）与前后指标量化对比。"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from radon.complexity import cc_visit

from mca.agent import AgentLoop, Budget, LoopGuard, RunResult
from mca.llm.client import LLMClient
from mca.observability import Tracer
from mca.review.detectors import max_nesting_depth
from mca.review.models import Hotspot
from mca.sandbox.base import Sandbox
from mca.tools import ToolRegistry, fix_tools

REFACTOR_SYSTEM_PROMPT = """你是一位资深重构专家，正在隔离沙箱中对遗留“屎山代码”进行**保持外部行为不变**的安全重构。
铁律：
1. 严禁修改任何测试文件，也不得破坏公开函数/类的函数名、参数签名与返回值契约。
2. 先运行 run_tests 确认基线测试全绿。
3. 小步重构：提取辅助函数、用卫语句消除深嵌套、用字典分派替代冗长 if-elif、消除重复代码块、修复吞异常问题。
4. 每次修改后立即运行 run_tests 验证行为完全一致。
5. 所有测试全绿且代码复杂度明显下降后，调用 submit 提交重构摘要。"""


@dataclass(frozen=True)
class FunctionMetrics:
    symbol: str
    cc: int
    lines: int
    nesting: int


@dataclass
class RefactorResult:
    run_result: RunResult
    before: FunctionMetrics | None
    after: FunctionMetrics | None


def measure_function(file_path: Path, symbol_name: str) -> FunctionMetrics | None:
    """用确定性工具测量单个函数的 (圈复杂度 CC, 行数, 最大嵌套深度)。"""
    if not file_path.exists():
        return None
    source = file_path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    cc_map: dict[str, int] = {}
    for b in cc_visit(source):
        sym = f"{b.classname}.{b.name}" if getattr(b, "classname", None) else b.name
        cc_map[sym] = int(b.complexity)

    target_node: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)) and top.name == symbol_name:
            target_node = top
            break
        if isinstance(top, ast.ClassDef):
            for m in top.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and f"{top.name}.{m.name}" == symbol_name:
                    target_node = m
                    break

    if target_node is None:
        return None
    length = (target_node.end_lineno or target_node.lineno) - target_node.lineno + 1
    return FunctionMetrics(
        symbol=symbol_name,
        cc=cc_map.get(symbol_name, 1),
        lines=length,
        nesting=max_nesting_depth(target_node),
    )


async def safe_refactor_hotspot(
    hotspot: Hotspot,
    *,
    llm: LLMClient,
    sandbox: Sandbox,
    budget: Budget | None = None,
    tracer: Tracer | None = None,
) -> RefactorResult:
    """在测试护航下重构指定热点，并量化对比重构前后的 CC / 行数 / 嵌套深度。"""
    target_file = sandbox.workspace / hotspot.file
    before = measure_function(target_file, hotspot.symbol)

    findings_text = "\n".join(f"- [{f.severity}] {f.rule_id}: {f.message}" for f in hotspot.findings)
    suggestion = hotspot.judgment.suggestion if hotspot.judgment else "提取辅助函数、消除重复和深嵌套"
    prompt = (
        f"请重构 `{hotspot.file}` 中的 `{hotspot.symbol}` (L{hotspot.start_line}-L{hotspot.end_line})。\n\n"
        f"静态扫描发现的问题：\n{findings_text}\n\n"
        f"建议重构方向：{suggestion}\n\n"
        f"要求：保持所有对外接口与行为 100% 不变，确保测试全绿后调用 submit。"
    )

    loop = AgentLoop(
        llm=llm,
        registry=ToolRegistry(fix_tools(include_ast=True)),
        sandbox=sandbox,
        budget=budget or Budget(max_steps=15, max_tokens=60_000),
        loop_guard=LoopGuard(enabled=True),
        tracer=tracer,
        system_prompt=REFACTOR_SYSTEM_PROMPT,
        protected_globs=("test_*.py", "tests/*"),
    )
    run_res = await loop.run(prompt, task_id=f"refactor_{hotspot.symbol}", mode="review")
    after = measure_function(target_file, hotspot.symbol)
    return RefactorResult(run_result=run_res, before=before, after=after)
