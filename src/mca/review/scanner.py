"""确定性扫描器与热点排序器：聚合自研 AST 检测器 + radon 圈复杂度 + ruff + git churn。"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from radon.complexity import cc_visit

from mca.fs_utils import iter_files
from mca.review.detectors import DetectorThresholds, detect_smells_in_source
from mca.review.models import SEVERITY_WEIGHT, Finding, Hotspot, Severity
from mca.security import is_sensitive_path


def scan_repository(
    repo_path: Path,
    *,
    thresholds: DetectorThresholds | None = None,
    include_tests: bool = False,
    run_ruff: bool = True,
) -> tuple[list[Finding], int, dict[str, int]]:
    """对整个仓库执行确定性静态扫描。返回 (findings, files_scanned, file_churn)。"""
    root = Path(repo_path).resolve()
    findings: list[Finding] = []
    all_block_hashes: dict[str, list[tuple[str, int, int, str]]] = defaultdict(list)
    files_scanned = 0

    for path in iter_files(root):
        if path.suffix != ".py" or is_sensitive_path(path):
            continue
        rel = path.relative_to(root).as_posix()
        if not include_tests and (path.name.startswith("test_") or "/tests/" in f"/{rel}"):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        files_scanned += 1
        # ① 自研 AST 坏味道检测器
        file_findings, block_hashes = detect_smells_in_source(source, rel, thresholds)
        findings.extend(file_findings)
        for h, occs in block_hashes.items():
            all_block_hashes[h].extend(occs)

        # ② radon 圈复杂度（Cyclomatic Complexity）
        findings.extend(_scan_radon_cc(source, rel))

    # 跨文件/跨函数重复代码块汇总
    for _, occs in all_block_hashes.items():
        if len(occs) >= 2:
            peers = ", ".join(f"{f}:{s}({sym})" for f, s, _, sym in occs[:4])
            for f, start, end, sym in occs:
                findings.append(
                    Finding(
                        rule_id="DuplicateBlock",
                        source="detector",
                        file=f,
                        start_line=start,
                        end_line=end,
                        symbol=sym,
                        severity="major",
                        message=f"函数 {sym} 与其他代码存在高度相似的结构重复（出现 {len(occs)} 处: {peers}），建议提取公共逻辑",
                        metrics={"duplicates": len(occs)},
                    )
                )

    # ③ 可选：调用 ruff 收集标准 lint 规则（不存在或报错时静默降级）
    if run_ruff:
        findings.extend(_scan_ruff(root, include_tests=include_tests))

    churn = _git_file_churn(root)
    return findings, files_scanned, churn


def rank_hotspots(
    findings: list[Finding],
    file_churn: dict[str, int],
    *,
    top_k: int = 8,
) -> list[Hotspot]:
    """按符号聚合 Findings，依据 `严重度 × log(1+CC) × (1 + 0.2*log(1+churn))` 排序，输出 Top-K 热点。"""
    grouped: dict[tuple[str, str], list[Finding]] = defaultdict(list)
    for f in findings:
        grouped[(f.file, f.symbol)].append(f)

    hotspots: list[Hotspot] = []
    for (file, symbol), items in grouped.items():
        sev_sum = sum(SEVERITY_WEIGHT.get(i.severity, 1.0) for i in items)
        max_cc = max((int(i.metrics.get("cc", 1)) for i in items), default=1)
        churn = file_churn.get(file, 0)
        score = sev_sum * (1.0 + math.log1p(max_cc)) * (1.0 + 0.2 * math.log1p(churn))
        start_line = min(i.start_line for i in items)
        end_line = max(i.end_line for i in items)
        hotspots.append(
            Hotspot(
                file=file,
                symbol=symbol,
                start_line=start_line,
                end_line=end_line,
                score=score,
                max_cc=max_cc,
                churn=churn,
                findings=items,
            )
        )

    hotspots.sort(key=lambda h: (-h.score, h.file, h.start_line))
    return hotspots[:top_k]


def _scan_radon_cc(source: str, rel_file: str) -> list[Finding]:
    out: list[Finding] = []
    try:
        blocks = cc_visit(source)
    except Exception:
        return out
    for b in blocks:
        cc = int(b.complexity)
        if cc < 8:
            continue
        sev: Severity = "critical" if cc >= 20 else ("major" if cc >= 12 else "minor")
        sym = f"{b.classname}.{b.name}" if getattr(b, "classname", None) else b.name
        out.append(
            Finding(
                rule_id="HighCyclomaticComplexity",
                source="radon",
                file=rel_file,
                start_line=b.lineno,
                end_line=b.endline or b.lineno,
                symbol=sym,
                severity=sev,
                message=f"圈复杂度 CC={cc}（建议 < 10），分支逻辑过于复杂，测试与维护成本高",
                metrics={"cc": cc},
            )
        )
    return out


def _scan_ruff(root: Path, *, include_tests: bool) -> list[Finding]:
    ruff_bin = shutil.which("ruff")
    if ruff_bin is None:
        return []
    try:
        proc = subprocess.run(
            [ruff_bin, "check", "--output-format=json", "--no-fix", str(root)],
            capture_output=True,
            text=True,
            timeout=20,
            encoding="utf-8",
            errors="replace",
        )
        items = json.loads(proc.stdout) if proc.stdout.strip() else []
    except Exception:
        return []

    out: list[Finding] = []
    for item in items:
        filename = Path(item.get("filename", ""))
        try:
            rel = filename.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        if not include_tests and (filename.name.startswith("test_") or "/tests/" in f"/{rel}"):
            continue
        loc = item.get("location", {})
        line = int(loc.get("row", 1))
        code = item.get("code", "RUFF")
        msg = item.get("message", "")
        out.append(
            Finding(
                rule_id=f"ruff:{code}",
                source="ruff",
                file=rel,
                start_line=line,
                end_line=line,
                symbol="<module>",
                severity="minor",
                message=msg,
            )
        )
    return out


def _git_file_churn(root: Path) -> dict[str, int]:
    """若目标目录是 git 仓库，统计近期各文件修改次数（热点 = 复杂 × 频繁改动）。"""
    if not (root / ".git").exists():
        return {}
    try:
        proc = subprocess.run(
            ["git", "log", "-n", "100", "--name-only", "--pretty=format:"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            return {}
        counts = Counter(line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip())
        return dict(counts)
    except Exception:
        return {}
