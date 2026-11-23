"""Review 模式单测：覆盖 8 类 AST 坏味道检测器、radon 复杂度、热点排序、结构化研判与防幻觉校验。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mca.llm.types import LLMResponse, Message, Usage
from mca.review import (
    ReviewReport,
    detect_smells_in_source,
    judge_hotspots,
    measure_function,
    rank_hotspots,
    render_markdown,
    scan_repository,
)

SHIT_MOUNTAIN_CODE = """
GLOBAL_CACHE = {}

def process_order(user, items, coupon, region, vip_level, extra_flag):
    global GLOBAL_CACHE
    total = 0
    if user:
        if items:
            for item in items:
                if item > 0:
                    if vip_level == 1:
                        total += item * 0.9
                    elif vip_level == 2:
                        total += item * 0.85
                    elif vip_level == 3:
                        total += item * 0.8
                    elif vip_level == 4:
                        total += item * 0.75
                    elif vip_level == 5:
                        total += item * 0.7
                    else:
                        total += item
    try:
        GLOBAL_CACHE[user] = total
    except Exception:
        pass
    return total

def copy_a(x, y):
    a = x + 1
    b = y * 2
    c = a + b
    return c * 10

def copy_b(m, n):
    p = m + 1
    q = n * 2
    r = p + q
    return r * 20
"""


class FakeJudgeLLM:
    """第一轮故意返回非法 JSON 验证自愈重试，第二轮返回含越界行号的合法 JSON 验证防幻觉过滤。"""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        assert response_format == {"type": "json_object"}
        if self.calls == 1:
            content = "not valid json"
        else:
            content = (
                '{"verdict": "true_positive", "root_cause": "嵌套过深且吞异常", '
                '"impact": "难以维护与排查线上故障", "suggestion": "卫语句提前返回 + 字典映射折扣率", '
                '"risk": "mid", "extra_issues": ['
                '{"line": 5, "description": "真实行号问题", "confidence": 0.9},'
                '{"line": 9999, "description": "幻觉越界行号", "confidence": 0.8}'
                "]}"
            )
        return LLMResponse(
            message=Message(role="assistant", content=content),
            finish_reason="stop",
            usage=Usage(prompt_tokens=100, completion_tokens=50),
            model="fake",
            latency_ms=1.0,
            attempts=1,
        )


def test_custom_ast_detectors_catch_all_smells():
    findings, block_hashes = detect_smells_in_source(SHIT_MOUNTAIN_CODE, "legacy.py")
    rules = {f.rule_id for f in findings}
    assert "TooManyParams" in rules
    assert "DeepNesting" in rules
    assert "LongIfElifChain" in rules
    assert "BroadException" in rules
    assert "GlobalMutation" in rules
    # copy_a 和 copy_b 变量名和常量数值不同，但归一化 AST 哈希应完全一致！
    dup_groups = [v for v in block_hashes.values() if len(v) >= 2]
    assert len(dup_groups) == 1


def test_scan_repository_radon_and_ranking(tmp_path: Path):
    (tmp_path / "legacy.py").write_text(SHIT_MOUNTAIN_CODE, encoding="utf-8")
    findings, files_scanned, churn = scan_repository(tmp_path, run_ruff=False)
    assert files_scanned == 1
    rules = {f.rule_id for f in findings}
    assert "HighCyclomaticComplexity" in rules
    assert "DuplicateBlock" in rules

    hotspots = rank_hotspots(findings, churn, top_k=5)
    assert hotspots[0].symbol == "process_order"
    assert hotspots[0].max_cc >= 10

    m = measure_function(tmp_path / "legacy.py", "process_order")
    assert m is not None and m.cc >= 10 and m.nesting >= 4


async def test_judge_hotspots_retries_json_and_filters_hallucinated_lines(tmp_path: Path):
    (tmp_path / "legacy.py").write_text(SHIT_MOUNTAIN_CODE, encoding="utf-8")
    findings, files_scanned, churn = scan_repository(tmp_path, run_ruff=False)
    hotspots = rank_hotspots(findings, churn, top_k=1)

    fake_llm = FakeJudgeLLM()
    await judge_hotspots(hotspots, tmp_path, fake_llm, max_concurrency=1)

    assert fake_llm.calls == 2  # 第 1 次非法 JSON 自动重试
    j = hotspots[0].judgment
    assert j is not None and j.verdict == "true_positive"
    # 行号 9999 的幻觉问题必须被确定性校验器过滤掉
    assert len(j.extra_issues) == 1 and j.extra_issues[0].line == 5

    report = ReviewReport("demo", files_scanned, len(findings), hotspots)
    md = render_markdown(report)
    assert "process_order" in md and "卫语句" in md
