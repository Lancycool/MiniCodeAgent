"""SQLite 全链路 Trace 存储层（关系型设计，可平滑迁移至 PostgreSQL）。

四张核心表形成完整的调用树：
  runs ──< steps ──< llm_calls
                 └──< tool_calls
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS runs (
    id                TEXT PRIMARY KEY,
    task_id           TEXT NOT NULL,
    mode              TEXT NOT NULL,
    status            TEXT NOT NULL,
    exit_reason       TEXT,
    total_steps       INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens     INTEGER NOT NULL DEFAULT 0,
    wall_time_ms      REAL NOT NULL DEFAULT 0.0,
    recover_count     INTEGER NOT NULL DEFAULT 0,
    final_patch       TEXT,
    started_at        TEXT NOT NULL,
    finished_at       TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step_no             INTEGER NOT NULL,
    phase               TEXT NOT NULL,
    context_tokens      INTEGER NOT NULL DEFAULT 0,
    compression_applied TEXT NOT NULL DEFAULT 'none',
    guard_event         TEXT,
    duration_ms         REAL NOT NULL DEFAULT 0.0,
    started_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    step_id           INTEGER NOT NULL REFERENCES steps(id) ON DELETE CASCADE,
    model             TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens     INTEGER NOT NULL DEFAULT 0,
    latency_ms        REAL NOT NULL DEFAULT 0.0,
    attempts          INTEGER NOT NULL DEFAULT 1,
    finish_reason     TEXT,
    error             TEXT,
    request_json      TEXT,
    response_json     TEXT
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    step_id          INTEGER NOT NULL REFERENCES steps(id) ON DELETE CASCADE,
    call_id          TEXT NOT NULL,
    tool_name        TEXT NOT NULL,
    args_json        TEXT NOT NULL,
    output_truncated TEXT NOT NULL,
    raw_chars        INTEGER NOT NULL DEFAULT 0,
    truncated        INTEGER NOT NULL DEFAULT 0,
    redactions       INTEGER NOT NULL DEFAULT 0,
    status           TEXT NOT NULL,
    latency_ms       REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_steps_run_id ON steps(run_id, step_no);
CREATE INDEX IF NOT EXISTS idx_llm_calls_step_id ON llm_calls(step_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_step_id ON tool_calls(step_id);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TraceStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    # ---------------- 写入 ----------------

    def create_run(self, run_id: str, task_id: str, mode: str) -> None:
        self._conn.execute(
            "INSERT INTO runs (id, task_id, mode, status, started_at) VALUES (?, ?, ?, 'running', ?)",
            (run_id, task_id, mode, _now_iso()),
        )
        self._conn.commit()

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        exit_reason: str,
        total_steps: int,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        wall_time_ms: float,
        recover_count: int,
        final_patch: str,
    ) -> None:
        self._conn.execute(
            """
            UPDATE runs
               SET status = ?, exit_reason = ?, total_steps = ?,
                   prompt_tokens = ?, completion_tokens = ?, cached_tokens = ?,
                   wall_time_ms = ?, recover_count = ?, final_patch = ?, finished_at = ?
             WHERE id = ?
            """,
            (
                status, exit_reason, total_steps,
                prompt_tokens, completion_tokens, cached_tokens,
                wall_time_ms, recover_count, final_patch, _now_iso(), run_id,
            ),
        )
        self._conn.commit()

    def add_step(
        self,
        run_id: str,
        step_no: int,
        phase: str,
        *,
        context_tokens: int = 0,
        compression_applied: str = "none",
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO steps (run_id, step_no, phase, context_tokens, compression_applied, started_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, step_no, phase, context_tokens, compression_applied, _now_iso()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def update_step(
        self,
        step_id: int,
        *,
        context_tokens: int | None = None,
        guard_event: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        fields: list[str] = []
        values: list[Any] = []
        if context_tokens is not None:
            fields.append("context_tokens = ?")
            values.append(context_tokens)
        if guard_event is not None:
            fields.append("guard_event = ?")
            values.append(guard_event)
        if duration_ms is not None:
            fields.append("duration_ms = ?")
            values.append(duration_ms)
        if not fields:
            return
        values.append(step_id)
        self._conn.execute(f"UPDATE steps SET {', '.join(fields)} WHERE id = ?", values)
        self._conn.commit()

    def add_llm_call(
        self,
        step_id: int,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        latency_ms: float,
        attempts: int,
        finish_reason: str | None = None,
        error: str | None = None,
        request_obj: Any = None,
        response_obj: Any = None,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO llm_calls (
                step_id, model, prompt_tokens, completion_tokens, cached_tokens,
                latency_ms, attempts, finish_reason, error, request_json, response_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step_id, model, prompt_tokens, completion_tokens, cached_tokens,
                latency_ms, attempts, finish_reason, error,
                json.dumps(request_obj, ensure_ascii=False) if request_obj is not None else None,
                json.dumps(response_obj, ensure_ascii=False) if response_obj is not None else None,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def add_tool_call(
        self,
        step_id: int,
        *,
        call_id: str,
        tool_name: str,
        args_json: str,
        output_truncated: str,
        raw_chars: int,
        truncated: bool,
        redactions: int,
        status: str,
        latency_ms: float,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO tool_calls (
                step_id, call_id, tool_name, args_json, output_truncated,
                raw_chars, truncated, redactions, status, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step_id, call_id, tool_name, args_json, output_truncated,
                raw_chars, int(truncated), redactions, status, latency_ms,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    # ---------------- 查询与分析 SQL ----------------

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run_trace(self, run_id: str) -> dict[str, Any] | None:
        run_row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run_row is None:
            return None
        steps_rows = self._conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY step_no, id", (run_id,)
        ).fetchall()
        steps: list[dict[str, Any]] = []
        for s in steps_rows:
            sd = dict(s)
            sd["llm_calls"] = [
                dict(r) for r in self._conn.execute("SELECT * FROM llm_calls WHERE step_id = ? ORDER BY id", (s["id"],))
            ]
            sd["tool_calls"] = [
                dict(r) for r in self._conn.execute("SELECT * FROM tool_calls WHERE step_id = ? ORDER BY id", (s["id"],))
            ]
            steps.append(sd)
        result = dict(run_row)
        result["steps"] = steps
        return result

    def tool_stats(self) -> list[dict[str, Any]]:
        """SQL 聚合：统计各工具的调用次数、失败率、平均耗时和平均原始输出长度。"""
        rows = self._conn.execute(
            """
            SELECT tool_name,
                   COUNT(*)                                   AS calls,
                   ROUND(AVG(CASE WHEN status != 'ok' THEN 1.0 ELSE 0.0 END) * 100, 1) AS fail_pct,
                   ROUND(AVG(latency_ms), 1)                  AS avg_ms,
                   ROUND(AVG(raw_chars), 0)                   AS avg_raw_chars
              FROM tool_calls
             GROUP BY tool_name
             ORDER BY calls DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def exit_reason_stats(self) -> list[dict[str, Any]]:
        """SQL 聚合：统计运行结束原因分布。"""
        rows = self._conn.execute(
            """
            SELECT status, exit_reason, COUNT(*) AS runs,
                   ROUND(AVG(total_steps), 1) AS avg_steps,
                   ROUND(AVG(prompt_tokens + completion_tokens), 0) AS avg_tokens
              FROM runs
             GROUP BY status, exit_reason
             ORDER BY runs DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
