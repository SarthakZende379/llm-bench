"""
pipeline.py — LangGraph benchmark pipeline for llm-bench.

Owns ORCHESTRATION only. The graph wires three nodes in sequence:

    START -> benchmark_node -> metrics_node -> logger_node -> END

The actual work is delegated to sibling modules (imported lazily inside each
node so this file loads standalone and there is no circular-import risk):

    - provider streaming calls -> providers.run_provider_benchmark
    - metric math              -> metrics.compute_provider_metrics
    - persistence              -> logger.persist_run

Design notes
------------
* Concurrency model: runs are fired CONCURRENTLY across providers but
  SEQUENTIALLY within a single provider. Hammering one provider with N parallel
  requests inflates its latency via server-side queuing and would corrupt the
  very TTFT / latency numbers we are trying to measure.
* Failure isolation: a provider that errors out is recorded as a failed
  RawRunResult, never an exception that aborts the cycle. Every other provider
  still completes and still gets logged.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# Shared data structures
# (In a larger codebase these would live in a dedicated schemas.py; kept here
#  so the state definition and its building blocks read as one unit.)
# ---------------------------------------------------------------------------


@dataclass
class RawRunResult:
    """Raw timing + content from ONE streaming run against ONE provider.

    All timestamps are time.perf_counter() values (monotonic seconds).
    metrics_node converts these into TTFT / TBT / tokens-per-sec.
    """

    provider: str
    model: str
    success: bool

    # Raw timing capture
    request_sent_at: float                 # when the request was sent
    first_token_at: float | None           # when the first token arrived
    last_token_at: float | None            # when the last token arrived
    token_timestamps: list[float] = field(default_factory=list)  # per-token arrival

    # Content
    response_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    # Failure detail (None on success)
    error: str | None = None


@dataclass
class ProviderMetrics:
    """Aggregated, dashboard-ready metrics for ONE provider across N runs."""

    provider: str
    model: str
    runs: int

    # Latency (milliseconds), summarised across runs
    ttft_ms_p50: float | None = None
    ttft_ms_p95: float | None = None
    tbt_ms_p50: float | None = None
    tokens_per_sec_p50: float | None = None
    total_latency_ms_p50: float | None = None

    # Economics
    cost_usd_per_run: float | None = None
    cost_usd_per_1k_tokens: float | None = None

    # Reliability
    success_rate: float = 0.0              # successes / runs, range 0.0..1.0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------


class BenchmarkState(TypedDict):
    """Shared state threaded through every node in the graph.

    Each node reads the keys it needs and returns ONLY the keys it writes;
    LangGraph merges those partial returns back into the running state.
    """

    # --- Inputs: set before the graph is invoked ---
    prompt: str
    selected_providers: list[str]
    runs_per_provider: int                 # clamped to 1..5
    run_id: str                            # UUID grouping this cycle's rows
    timestamp: str                         # ISO 8601, UTC

    # --- Written by benchmark_node ---
    raw_responses: dict[str, list[RawRunResult]]   # provider -> N raw runs

    # --- Written by metrics_node ---
    computed_metrics: dict[str, ProviderMetrics]   # provider -> aggregate

    # --- Written by logger_node ---
    persisted: bool


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def benchmark_node(state: BenchmarkState) -> dict:
    """Fire the prompt at every selected provider and capture raw timings.

    Concurrent across providers, sequential within a provider (see module docs).
    """
    from providers import run_provider_benchmark  # built in the next step

    prompt = state["prompt"]
    runs = state["runs_per_provider"]

    async def run_all_for_provider(provider: str) -> list[RawRunResult]:
        results: list[RawRunResult] = []
        for _ in range(runs):
            try:
                results.append(await run_provider_benchmark(provider, prompt))
            except Exception as exc:  # failure isolation — never abort the cycle
                results.append(_failed_run(provider, str(exc)))
        return results

    provider_coros = [run_all_for_provider(p) for p in state["selected_providers"]]
    per_provider_results = await asyncio.gather(*provider_coros)

    raw_responses = dict(zip(state["selected_providers"], per_provider_results))
    return {"raw_responses": raw_responses}


def metrics_node(state: BenchmarkState) -> dict:
    """Turn raw timings into aggregated dashboard metrics. Pure — no I/O."""
    from metrics import compute_provider_metrics  # built in a later step

    computed = {
        provider: compute_provider_metrics(provider, runs)
        for provider, runs in state["raw_responses"].items()
    }
    return {"computed_metrics": computed}


def logger_node(state: BenchmarkState) -> dict:
    """Persist one row per provider to SQLite. The only node that touches disk."""
    from logger import persist_run  # built in a later step

    persist_run(
        run_id=state["run_id"],
        timestamp=state["timestamp"],
        prompt=state["prompt"],
        metrics=state["computed_metrics"],
        raw=state["raw_responses"],
        is_demo=False,
    )
    return {"persisted": True}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_pipeline():
    """Compile the linear three-node benchmark graph."""
    graph = StateGraph(BenchmarkState)

    graph.add_node("benchmark", benchmark_node)
    graph.add_node("metrics", metrics_node)
    graph.add_node("logger", logger_node)

    graph.add_edge(START, "benchmark")
    graph.add_edge("benchmark", "metrics")
    graph.add_edge("metrics", "logger")
    graph.add_edge("logger", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Helpers + convenience runner
# ---------------------------------------------------------------------------


def _failed_run(provider: str, error: str) -> RawRunResult:
    """Build a RawRunResult representing a failed attempt."""
    return RawRunResult(
        provider=provider,
        model="unknown",
        success=False,
        request_sent_at=0.0,
        first_token_at=None,
        last_token_at=None,
        token_timestamps=[],
        response_text="",
        input_tokens=0,
        output_tokens=0,
        error=error,
    )


def make_initial_state(
    prompt: str,
    providers: list[str],
    runs_per_provider: int = 3,
) -> BenchmarkState:
    """Construct a fresh state with inputs filled and output keys empty."""
    runs = max(1, min(runs_per_provider, 5))  # clamp to 1..5
    return BenchmarkState(
        prompt=prompt,
        selected_providers=providers,
        runs_per_provider=runs,
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        raw_responses={},
        computed_metrics={},
        persisted=False,
    )


async def run_benchmark(
    prompt: str,
    providers: list[str],
    runs_per_provider: int = 3,
) -> BenchmarkState:
    """End-to-end: build the graph, run it, return the fully populated state."""
    pipeline = build_pipeline()
    initial = make_initial_state(prompt, providers, runs_per_provider)
    final_state: BenchmarkState = await pipeline.ainvoke(initial)
    return final_state


if __name__ == "__main__":
    # Smoke test — runnable once providers.py / metrics.py / logger.py exist.
    async def _demo() -> None:
        state = await run_benchmark(
            prompt="Explain time-to-first-token in one sentence.",
            providers=["groq", "gemini", "together"],
            runs_per_provider=3,
        )
        for provider, m in state["computed_metrics"].items():
            print(f"{provider:10s}  TTFT p50={m.ttft_ms_p50}  tok/s={m.tokens_per_sec_p50}")

    asyncio.run(_demo())