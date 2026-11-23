"""SWE-bench Lite 适配器离线单测：验证数据解析、preds.jsonl 格式、报告输出与 CLI 子命令。"""

from __future__ import annotations

import json
from pathlib import Path

from mca.cli import build_parser
from mca.harness.swebench import SWEBenchEvalResult, _parse_test_list, save_swebench_outputs


def test_parse_test_list():
    assert _parse_test_list('["tests/test_a.py::test_1"]') == ["tests/test_a.py::test_1"]
    assert _parse_test_list(["a", "b"]) == ["a", "b"]


def test_save_swebench_outputs_produces_official_preds_jsonl(tmp_path: Path):
    results = [
        SWEBenchEvalResult(
            config_name="full",
            instance_id="pallets__flask-4992",
            repo="pallets/flask",
            run_id="abc123",
            exit_reason="verified",
            resolved=True,
            f2p_passed=True,
            p2p_passed=True,
            steps=7,
            total_tokens=22000,
            wall_time_ms=14500.0,
            model_patch="diff --git a/src/flask/config.py b/src/flask/config.py\n",
        )
    ]
    preds_path, md_path = save_swebench_outputs(results, tmp_path)
    lines = preds_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["instance_id"] == "pallets__flask-4992"
    assert rec["model_name_or_path"] == "minicodeagent-full"
    assert "diff --git" in rec["model_patch"]

    md = md_path.read_text(encoding="utf-8")
    assert "pallets__flask-4992" in md and "100.0%" in md


def test_cli_swebench_args():
    args = build_parser().parse_args(["swebench", "--limit", "3", "--configs", "full,no_ast"])
    assert args.command == "swebench" and args.limit == 3 and args.configs == "full,no_ast"
