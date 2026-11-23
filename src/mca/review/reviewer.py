"""Review 模式第 3 阶段：Structured Output LLM 并发研判 + 证据防幻觉校验 + 报告生成。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import ValidationError

from mca.llm.client import LLMClient
from mca.llm.types import Message
from mca.review.models import Hotspot, LLMJudgment, ReviewReport

JUDGE_SYSTEM_PROMPT = """你是一位资深代码架构师，正在审阅由静态分析工具扫描出的代码热点。
请结合源码与静态指标，给出客观、可落地的诊断结论，并**严格只输出合法的 JSON 对象**（不要包含 Markdown 代码块之外的任何文字）。
JSON 必须符合以下字段：
{
  "verdict": "true_positive" | "false_positive" | "acceptable",
  "root_cause": "一句话说明根因",
  "impact": "对可读性/可测试性/性能/潜在Bug的具体影响",
  "suggestion": "具体重构手法（如提取函数、卫语句、字典分派、dataclass封装等）",
  "risk": "low" | "mid" | "high",
  "extra_issues": [{"line": 行号整数, "description": "静态扫描未发现的额外问题", "confidence": 0.8}]
}"""


async def judge_hotspots(
    hotspots: list[Hotspot],
    workspace: Path,
    llm: LLMClient,
    *,
    max_concurrency: int = 3,
) -> None:
    """使用 asyncio.Semaphore 并发研判 Top-K 热点，并用确定性代码校验模型输出的 JSON 与行号证据。"""
    sem = asyncio.Semaphore(max_concurrency)

    async def _judge_one(h: Hotspot) -> None:
        async with sem:
            file_path = workspace / h.file
            if not file_path.exists():
                return
            all_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            total_lines = len(all_lines)
            lo = max(1, h.start_line - 2)
            hi = min(total_lines, min(h.end_line + 2, h.start_line + 120))
            code_block = "\n".join(f"{i}| {all_lines[i - 1]}" for i in range(lo, hi + 1))

            findings_text = "\n".join(
                f"- [{f.severity}] {f.rule_id} (L{f.start_line}-L{f.end_line}): {f.message}"
                for f in h.findings
            )
            user_prompt = (
                f"文件: {h.file} | 符号: {h.symbol} | 热点分: {h.score:.1f} | 最高圈复杂度: {h.max_cc}\n\n"
                f"静态扫描发现:\n{findings_text}\n\n"
                f"相关源码（第 {lo}-{hi} 行，共 {total_lines} 行）:\n```python\n{code_block}\n```"
            )

            judgment = await _call_structured_judgment(llm, user_prompt)
            if judgment is not None:
                # 确定性证据校验：过滤掉行号超出文件范围的幻觉 extra_issues
                judgment.extra_issues = [
                    ex for ex in judgment.extra_issues if 1 <= ex.line <= total_lines
                ]
                h.judgment = judgment

    await asyncio.gather(*(_judge_one(h) for h in hotspots))


async def _call_structured_judgment(llm: LLMClient, user_prompt: str) -> LLMJudgment | None:
    """服务端 JSON Mode + 客户端 Pydantic 校验 + 失败自愈重试 1 次。"""
    messages = [Message.system(JUDGE_SYSTEM_PROMPT), Message.user(user_prompt)]
    for attempt in range(2):
        resp = await llm.chat(
            messages,
            tools=None,
            response_format={"type": "json_object"},
        )
        raw = _strip_code_fence(resp.message.content or "")
        try:
            return LLMJudgment.model_validate_json(raw)
        except (ValidationError, ValueError) as e:
            if attempt == 0:
                messages.append(resp.message)
                messages.append(
                    Message.user(f"输出未通过 JSON Schema 校验：{e}\n请严格按要求只返回合法 JSON 对象。")
                )
    return None


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return s


def render_markdown(report: ReviewReport) -> str:
    lines = [
        f"# 代码诊断报告：{report.repo_name}",
        "",
        f"- **扫描 Python 文件数**：{report.files_scanned}",
        f"- **静态扫描发现问题总数**：{report.total_findings}",
        f"- **Top 热点数量**：{len(report.hotspots)}",
        "",
        "## 一、热点总览（按 严重度 × 圈复杂度 × 变更频率 排序）",
        "",
        "| 排名 | 文件 | 符号 | 行范围 | 热点分 | 最高 CC | 研判结论 | 风险 |",
        "|---:|---|---|---|---:|---:|---|---|",
    ]
    for idx, h in enumerate(report.hotspots, 1):
        verdict = h.judgment.verdict if h.judgment else "未研判"
        risk = h.judgment.risk if h.judgment else "-"
        lines.append(
            f"| {idx} | `{h.file}` | `{h.symbol}` | L{h.start_line}-L{h.end_line} "
            f"| {h.score:.1f} | {h.max_cc} | `{verdict}` | `{risk}` |"
        )

    lines.extend(["", "## 二、热点详细诊断与优化建议", ""])
    for idx, h in enumerate(report.hotspots, 1):
        lines.append(f"### {idx}. `{h.file}` — `{h.symbol}` (L{h.start_line}-L{h.end_line})")
        lines.append("")
        lines.append("**确定性静态扫描证据：**")
        for f in h.findings:
            lines.append(f"- `[{f.severity}]` **{f.rule_id}** (`L{f.start_line}-L{f.end_line}`): {f.message}")
        if h.judgment:
            j = h.judgment
            lines.extend(
                [
                    "",
                    "**LLM 架构师研判：**",
                    f"- **判定**：`{j.verdict}`（重构风险：`{j.risk}`）",
                    f"- **根因**：{j.root_cause}",
                    f"- **影响**：{j.impact}",
                    f"- **优化建议**：{j.suggestion}",
                ]
            )
            if j.extra_issues:
                lines.append("- **额外发现（LLM 推理，需人工复核）**：")
                for ex in j.extra_issues:
                    lines.append(f"  - `L{ex.line}`（置信度 {ex.confidence:.0%}）: {ex.description}")
        lines.append("")

    return "\n".join(lines)


def save_report(report: ReviewReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "review_report.md"
    json_path = out_dir / "review_report.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, json_path
