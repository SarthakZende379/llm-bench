"""
logger.py — SQLite persistence for benchmark runs.

One row per (run_id, provider): the aggregated ProviderMetrics for that provider
in that benchmark cycle, plus a sample response and the is_demo flag that powers
the dashboard's seeded-vs-live distinction.

The DB lives at data/results.db (gitignored). init_db() is idempotent, so every
entry point can call it freely.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pipeline import ProviderMetrics, RawRunResult

DB_PATH = Path(__file__).parent / "data" / "results.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                 TEXT    NOT NULL,
    timestamp              TEXT    NOT NULL,
    prompt                 TEXT    NOT NULL,
    provider               TEXT    NOT NULL,
    model                  TEXT    NOT NULL,
    runs                   INTEGER NOT NULL,
    ttft_ms_p50            REAL,
    ttft_ms_p95            REAL,
    tbt_ms_p50             REAL,
    tokens_per_sec_p50     REAL,
    total_latency_ms_p50   REAL,
    cost_usd_per_run       REAL,
    cost_usd_per_1k_tokens REAL,
    success_rate           REAL,
    errors                 TEXT,
    sample_response        TEXT,
    is_demo                INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_run_id ON benchmark_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_timestamp ON benchmark_runs(timestamp);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the table and indexes if they don't exist. Idempotent."""
    conn = _connect()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _sample_response(runs: list[RawRunResult]) -> str:
    """First successful response text, for the dashboard's raw-output toggle."""
    for r in runs:
        if r.success and r.response_text:
            return r.response_text
    return ""


def persist_run(
    run_id: str,
    timestamp: str,
    prompt: str,
    metrics: dict[str, ProviderMetrics],
    raw: dict[str, list[RawRunResult]],
    is_demo: bool = False,
) -> None:
    """Write one row per provider for this benchmark cycle."""
    init_db()
    rows = [
        (
            run_id, timestamp, prompt, provider, m.model, m.runs,
            m.ttft_ms_p50, m.ttft_ms_p95, m.tbt_ms_p50,
            m.tokens_per_sec_p50, m.total_latency_ms_p50,
            m.cost_usd_per_run, m.cost_usd_per_1k_tokens,
            m.success_rate, ("; ".join(m.errors) if m.errors else None),
            _sample_response(raw.get(provider, [])),
            1 if is_demo else 0,
        )
        for provider, m in metrics.items()
    ]
    conn = _connect()
    try:
        conn.executemany(
            """INSERT INTO benchmark_runs (
                run_id, timestamp, prompt, provider, model, runs,
                ttft_ms_p50, ttft_ms_p95, tbt_ms_p50,
                tokens_per_sec_p50, total_latency_ms_p50,
                cost_usd_per_run, cost_usd_per_1k_tokens,
                success_rate, errors, sample_response, is_demo
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def fetch_runs(include_demo: bool = True, limit: int | None = None) -> list[dict]:
    """Return rows as dicts, newest first. include_demo=False -> live runs only."""
    init_db()
    query = "SELECT * FROM benchmark_runs"
    if not include_demo:
        query += " WHERE is_demo = 0"
    query += " ORDER BY timestamp DESC, id DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    conn = _connect()
    try:
        return [dict(row) for row in conn.execute(query).fetchall()]
    finally:
        conn.close()


def is_empty() -> bool:
    """True if there are no rows — used to decide whether to seed demo data."""
    init_db()
    conn = _connect()
    try:
        (count,) = conn.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()
    finally:
        conn.close()
    return count == 0


if __name__ == "__main__":
    # End-to-end test: run the WHOLE pipeline, then confirm rows landed in SQLite.
    #   python logger.py
    import asyncio

    from pipeline import run_benchmark

    async def _demo() -> None:
        state = await run_benchmark(
            "Explain TTFT in one sentence.",
            ["groq", "together", "gemini"],
            runs_per_provider=3,
        )
        print(f"run_id = {state['run_id']}   persisted = {state['persisted']}")
        rows = fetch_runs(limit=10)
        print(f"rows in db: {len(rows)}\n")
        for row in rows:
            ttft = row["ttft_ms_p50"]
            tok = row["tokens_per_sec_p50"]
            if ttft is None:
                print(f"  {row['provider']:10s}  (failed: {row['errors']})")
            else:
                print(
                    f"  {row['provider']:10s}  TTFT p50={ttft:.0f}ms  "
                    f"tok/s={tok:.0f}  ok={row['success_rate']:.0%}"
                )

    asyncio.run(_demo())