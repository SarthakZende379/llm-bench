"""
seed_demo_data.py — generate frozen, REAL demo data for the dashboard.

Runs the actual benchmark across all providers and writes the results with
is_demo=1, so the deployed dashboard shows real numbers the instant it loads —
no API keys or waiting required from the viewer.

Workflow:
    1. (once) delete data/results.db to clear any earlier test rows
    2. python seed_demo_data.py        # with your keys in .env
    3. commit data/results.db so the seeded rows ship with the app

Re-running clears previous demo rows first (is_demo=1) and regenerates them,
so it's safe to run repeatedly. Live runs (is_demo=0) are left untouched.

We seed the 'medium' and 'long' prompts only: both produce substantial output,
so decode throughput is measured accurately. The 'short' prompt is great for
TTFT but too brief to measure throughput reliably on buffering providers, so we
keep it out of the showcased data.
"""

from __future__ import annotations

import asyncio
import sqlite3

from pipeline import benchmark_node, make_initial_state, metrics_node
from logger import DB_PATH, fetch_runs, init_db, persist_run
from prompts import PROMPT_SETS

PROVIDERS = ["groq", "together", "gemini"]
RUNS_PER_PROVIDER = 3
SEED_LABELS = ["medium", "long"]   # substantial output -> reliable throughput
PACE_SECONDS = 6                   # gap between cycles to respect Gemini's ~15 rpm


def _clear_demo_rows() -> None:
    """Remove existing is_demo=1 rows so a re-run regenerates a clean demo set."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("DELETE FROM benchmark_runs WHERE is_demo = 1")
        conn.commit()
    finally:
        conn.close()


async def _seed_one(label: str, prompt: str) -> None:
    print(f"\n=== seeding '{label}' ===")
    state = make_initial_state(prompt, PROVIDERS, RUNS_PER_PROVIDER)
    state["raw_responses"] = (await benchmark_node(state))["raw_responses"]
    state["computed_metrics"] = metrics_node(state)["computed_metrics"]

    persist_run(
        run_id=state["run_id"],
        timestamp=state["timestamp"],
        prompt=prompt,
        metrics=state["computed_metrics"],
        raw=state["raw_responses"],
        is_demo=True,
    )

    for provider, m in state["computed_metrics"].items():
        line = f"  {provider:10s}  ok={m.success_rate:.0%}"
        if m.ttft_ms_p50 is not None:
            line += f"  TTFT p50={m.ttft_ms_p50:.0f}ms  tok/s={m.tokens_per_sec_p50:.0f}"
        else:
            line += f"  (no successful runs: {m.errors})"
        print(line)


async def main() -> None:
    print("Clearing previous demo rows...")
    _clear_demo_rows()

    for i, label in enumerate(SEED_LABELS):
        await _seed_one(label, PROMPT_SETS[label])
        if i < len(SEED_LABELS) - 1:
            print(f"\n...pausing {PACE_SECONDS}s to ease off Gemini's rate limit...")
            await asyncio.sleep(PACE_SECONDS)

    demo_rows = [r for r in fetch_runs() if r["is_demo"] == 1]
    print(f"\nDone. {len(demo_rows)} demo rows written to data/results.db.")
    print("Commit data/results.db so the seeded data ships with the app.")


if __name__ == "__main__":
    asyncio.run(main())