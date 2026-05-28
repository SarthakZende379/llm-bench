"""
metrics.py — turn raw streaming timings into dashboard-ready metrics.

Pure functions, no I/O. Given the RawRunResults for one provider (one per
repeat run), compute_provider_metrics aggregates them into a single
ProviderMetrics with p50/p95 latency, decode throughput, cost, and reliability.

Methodology notes
-----------------
* TTFT = first_token_at - request_sent_at. The latency a user feels before
  anything appears. Reported as p50 (typical) and p95 (tail).
* Tokens/sec = output_tokens / decode_window, where decode_window =
  last_token_at - first_token_at (pure decode throughput, excluding TTFT). If a
  provider streams the whole reply in a single chunk (decode_window == 0) we
  fall back to last_token_at - request_sent_at so the number stays defined; such
  responses are short enough that the distinction is negligible.
* TBT (inter-token latency) = mean gap between consecutive streamed chunks. Only
  defined when a run produced >= 2 chunks; single-chunk runs contribute no
  sample (some providers buffer the whole response into one chunk).
* Output tokens are estimated from response text (~4 chars/token). Providers
  don't reliably expose exact counts mid-stream and chunk counts vary by how
  aggressively each buffers, so a text-based estimate is the most consistent
  cross-provider basis for throughput and cost.
* Cost is an estimate from public per-token list prices, not billed usage.
"""

from __future__ import annotations

import math
from statistics import mean

from pipeline import ProviderMetrics, RawRunResult
from providers import get_config, get_pricing


def _estimate_tokens(text: str) -> int:
    """Approximate token count from text (~4 characters per token)."""
    return max(1, round(len(text) / 4))


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile. pct in [0, 100]. None if no values."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = math.ceil((pct / 100) * len(ordered))
    idx = min(max(rank - 1, 0), len(ordered) - 1)
    return ordered[idx]


def _run_latencies(run: RawRunResult) -> dict | None:
    """Derive per-run metrics from one RawRunResult, or None if it failed."""
    if not run.success or run.first_token_at is None or run.last_token_at is None:
        return None

    ttft_ms = (run.first_token_at - run.request_sent_at) * 1000.0
    total_ms = (run.last_token_at - run.request_sent_at) * 1000.0
    out_tokens = _estimate_tokens(run.response_text)

    decode_window = run.last_token_at - run.first_token_at
    if decode_window <= 0:  # single-chunk stream — fall back to end-to-end time
        decode_window = max(run.last_token_at - run.request_sent_at, 1e-6)
    tokens_per_sec = out_tokens / decode_window

    ts = run.token_timestamps
    if len(ts) >= 2:
        gaps = [(ts[i] - ts[i - 1]) * 1000.0 for i in range(1, len(ts))]
        tbt_ms = mean(gaps)
    else:
        tbt_ms = None

    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "tokens_per_sec": tokens_per_sec,
        "tbt_ms": tbt_ms,
        "out_tokens": out_tokens,
        "in_tokens": run.input_tokens,
    }


def compute_provider_metrics(provider: str, runs: list[RawRunResult]) -> ProviderMetrics:
    """Aggregate N runs for one provider into a single ProviderMetrics."""
    cfg = get_config(provider)
    price_in, price_out = get_pricing(provider)

    total_runs = len(runs)
    per_run = [_run_latencies(r) for r in runs]
    ok = [m for m in per_run if m is not None]
    errors = [r.error for r in runs if r.error]
    success_rate = (len(ok) / total_runs) if total_runs else 0.0

    if not ok:  # every run failed — record the failure, leave metrics empty
        return ProviderMetrics(
            provider=provider,
            model=cfg.model,
            runs=total_runs,
            success_rate=success_rate,
            errors=errors,
        )

    ttfts = [m["ttft_ms"] for m in ok]
    totals = [m["total_ms"] for m in ok]
    speeds = [m["tokens_per_sec"] for m in ok]
    tbts = [m["tbt_ms"] for m in ok if m["tbt_ms"] is not None]

    per_run_cost = [
        (m["in_tokens"] / 1_000_000) * price_in
        + (m["out_tokens"] / 1_000_000) * price_out
        for m in ok
    ]
    avg_cost = mean(per_run_cost)
    avg_in_tokens = mean(m["in_tokens"] for m in ok)
    avg_out_tokens = mean(m["out_tokens"] for m in ok)
    total_tokens = avg_in_tokens + avg_out_tokens
    cost_per_1k = (avg_cost / total_tokens * 1000) if total_tokens else None

    return ProviderMetrics(
        provider=provider,
        model=cfg.model,
        runs=total_runs,
        ttft_ms_p50=_percentile(ttfts, 50),
        ttft_ms_p95=_percentile(ttfts, 95),
        tbt_ms_p50=_percentile(tbts, 50),
        tokens_per_sec_p50=_percentile(speeds, 50),
        total_latency_ms_p50=_percentile(totals, 50),
        cost_usd_per_run=avg_cost,
        cost_usd_per_1k_tokens=cost_per_1k,
        success_rate=success_rate,
        errors=errors,
    )


if __name__ == "__main__":
    # Test metrics against LIVE data: run a provider N times, aggregate, print.
    #   python metrics.py [groq|gemini|together]
    import asyncio
    import sys

    from providers import run_provider_benchmark

    def _f(v: float | None, suffix: str = "") -> str:
        return f"{v:.1f}{suffix}" if v is not None else "n/a"

    async def _demo(provider: str, n: int = 3) -> None:
        runs = []
        for _ in range(n):
            runs.append(await run_provider_benchmark(provider, "Explain TTFT in one sentence."))
        m = compute_provider_metrics(provider, runs)
        print(f"{m.provider}  ({m.model})   runs={m.runs}  success={m.success_rate:.0%}")
        print(f"  TTFT   p50={_f(m.ttft_ms_p50, 'ms')}   p95={_f(m.ttft_ms_p95, 'ms')}")
        print(f"  tok/s  p50={_f(m.tokens_per_sec_p50)}")
        print(f"  TBT    p50={_f(m.tbt_ms_p50, 'ms')}")
        print(f"  total  p50={_f(m.total_latency_ms_p50, 'ms')}")
        print(f"  cost/run = ${m.cost_usd_per_run:.6f}" if m.cost_usd_per_run else "  cost/run = n/a")
        if m.errors:
            print(f"  errors  = {m.errors}")

    name = sys.argv[1] if len(sys.argv) > 1 else "groq"
    asyncio.run(_demo(name))