"""SQLite TraceStore 与统计 SQL 单测。"""

from __future__ import annotations

from mca.observability import TraceStore, format_run_replay, format_tool_stats


def test_trace_store_crud_and_sql_stats():
    store = TraceStore(":memory:")
    store.create_run("r1", "task_a", "fix")
    s1 = store.add_step("r1", 1, "RUNNING")
    store.update_step(s1, context_tokens=500, duration_ms=120.0)
    store.add_llm_call(
        s1, model="deepseek-v4.1-flash", prompt_tokens=500, completion_tokens=80,
        cached_tokens=256, latency_ms=100.0, attempts=1, finish_reason="tool_calls",
    )
    store.add_tool_call(
        s1, call_id="c1", tool_name="read_file", args_json='{"path":"a.py"}',
        output_truncated="ok", raw_chars=420, truncated=False, redactions=0, status="ok", latency_ms=2.0,
    )
    store.add_tool_call(
        s1, call_id="c2", tool_name="edit_file", args_json='{}',
        output_truncated="err", raw_chars=50, truncated=False, redactions=0, status="invalid_args", latency_ms=1.0,
    )
    store.finish_run(
        "r1", status="done", exit_reason="verified", total_steps=1,
        prompt_tokens=500, completion_tokens=80, cached_tokens=256,
        wall_time_ms=150.0, recover_count=0, final_patch="diff...",
    )

    assert len(store.list_runs()) == 1
    stats = {row["tool_name"]: row for row in store.tool_stats()}
    assert stats["read_file"]["fail_pct"] == 0.0
    assert stats["edit_file"]["fail_pct"] == 100.0

    exits = store.exit_reason_stats()
    assert exits[0]["exit_reason"] == "verified" and exits[0]["runs"] == 1

    replay = format_run_replay(store, "r1")
    assert "Trace Replay: r1" in replay and "read_file" in replay
    assert "edit_file" in format_tool_stats(store)
    store.close()
