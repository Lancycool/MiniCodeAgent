"""Review 模式数据模型：Finding / LLMJudgment（结构化约束输出）/ Hotspot / ReviewReport。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "minor", "major", "critical"]
SEVERITY_WEIGHT: dict[str, float] = {
    "info": 1.0,
    "minor": 2.0,
    "major": 5.0,
    "critical": 10.0,
}


@dataclass
class Finding:
    rule_id: str
    source: Literal["detector", "radon", "ruff"]
    file: str
    start_line: int
    end_line: int
    symbol: str
    severity: Severity
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence: str = ""


class ExtraIssue(BaseModel):
    """LLM 额外发现的、静态扫描未覆盖的语义/设计问题（必须带行号与置信度）。"""

    line: int = Field(ge=1, description="问题所在行号")
    description: str = Field(min_length=1, description="问题描述")
    confidence: float = Field(0.7, ge=0.0, le=1.0, description="置信度 0~1")


class LLMJudgment(BaseModel):
    """LLM 对单个热点的结构化研判结果（JSON Mode + Pydantic 双重约束）。"""

    verdict: Literal["true_positive", "false_positive", "acceptable"] = Field(
        description="true_positive=确有问题需优化; false_positive=误报; acceptable=可接受暂不改"
    )
    root_cause: str = Field(min_length=1, description="一句话概括设计或实现根因")
    impact: str = Field(min_length=1, description="对可读性/可测试性/性能/潜在Bug的影响")
    suggestion: str = Field(min_length=1, description="具体可落地的重构手法与优化方向")
    risk: Literal["low", "mid", "high"] = Field(description="重构风险等级")
    extra_issues: list[ExtraIssue] = Field(default_factory=list, description="静态扫描未发现的额外问题")


@dataclass
class Hotspot:
    file: str
    symbol: str
    start_line: int
    end_line: int
    score: float
    max_cc: int = 1
    churn: int = 0
    findings: list[Finding] = field(default_factory=list)
    judgment: LLMJudgment | None = None


@dataclass
class ReviewReport:
    repo_name: str
    files_scanned: int
    total_findings: int
    hotspots: list[Hotspot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_name": self.repo_name,
            "files_scanned": self.files_scanned,
            "total_findings": self.total_findings,
            "hotspots": [
                {
                    "file": h.file,
                    "symbol": h.symbol,
                    "start_line": h.start_line,
                    "end_line": h.end_line,
                    "score": round(h.score, 2),
                    "max_cc": h.max_cc,
                    "churn": h.churn,
                    "findings": [asdict(f) for f in h.findings],
                    "judgment": h.judgment.model_dump() if h.judgment else None,
                }
                for h in self.hotspots
            ],
        }
