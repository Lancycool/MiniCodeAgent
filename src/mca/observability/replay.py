"""Trace 回放与指标格式化输出，用于终端快速定位 Agent 异常根因。"""

from __future__ import annotations

from mca.observability.store import TraceStore


def format_run_replay(store: TraceStore, run_id: str) -> str:
    trace = store.get_run_trace(run_id)
    if trace is None:
        return f"未找到 run_id={run_id}"

    total_tok = trace["prompt_tokens"] + trace["completion_tokens"]
    lines = [
        f"=== Trace Replay: {trace['id']} ===",
        f"task={trace['task_id']} | mode={trace['mode']} | status={trace['status']} "
        f"| exit_reason={trace['exit_reason']}",
        f"steps={trace['total_steps']} | tokens={total_tok} "
        f"(prompt={trace['prompt_tokens']}, comp={trace['completion_tokens']}, cached={trace['cached_tokens']}) "
        f"| recovers={trace['recover_count']} | wall={trace['wall_time_ms']:.0f}ms",
    ]
    for s in trace["steps"]:
        guard = f" | GUARD={s['guard_event']}" if s["guard_event"] else ""
        comp = f" | COMP={s['compression_applied']}" if s["compression_applied"] != "none" else ""
        lines.append(
            f"\n[Step {s['step_no']} | {s['phase']} | ctx={s['context_tokens']} tok | {s['duration_ms']:.0f}ms{comp}{guard}]"
        )
        for lc in s["llm_calls"]:
            if lc["error"]:
                lines.append(f"  LLM ERROR: {lc['error']}")
            else:
                lines.append(
                    f"  LLM ({lc['model']}) finish={lc['finish_reason']} "
                    f"tok={lc['prompt_tokens']}+{lc['completion_tokens']} "
                    f"{lc['latency_ms']:.0f}ms attempts={lc['attempts']}"
                )
        for tc in s["tool_calls"]:
            preview = tc["output_truncated"].replace("\n", " | ")[:140]
            lines.append(
                f"  TOOL {tc['tool_name']}({tc['args_json'][:100]}) → {tc['status']} "
                f"{tc['latency_ms']:.0f}ms (raw={tc['raw_chars']})"
            )
            lines.append(f"       {preview}")
    return "\n".join(lines)


def format_tool_stats(store: TraceStore) -> str:
    rows = store.tool_stats()
    if not rows:
        return "(暂无工具调用记录)"
    header = f"{'Tool':<16} {'Calls':>6} {'Fail%':>7} {'AvgMs':>8} {'AvgChars':>10}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['tool_name']:<16} {r['calls']:>6} {r['fail_pct']:>6.1f}% "
            f"{r['avg_ms']:>8.1f} {int(r['avg_raw_chars']):>10}"
        )
    return "\n".join(lines)
