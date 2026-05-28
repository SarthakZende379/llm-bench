"""
app.py — Streamlit dashboard for llm-bench.

Loads pre-seeded real benchmark data instantly, and lets a visitor run a fresh
live benchmark with one click using the app's OWN API keys — no key entry.

Run locally:   streamlit run app.py
Deployed:      Streamlit Cloud, with API keys set in the Secrets manager.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from logger import fetch_runs
from pipeline import run_benchmark
from prompts import PROMPT_SETS
from providers import available_providers, get_config

st.set_page_config(page_title="llm-bench", page_icon="⚡", layout="wide")


def _load_secrets_into_env() -> None:
    """On Streamlit Cloud, copy secrets into env so providers.py (os.getenv) works.
    Locally this is a no-op — providers.py loads .env itself."""
    try:
        for key in ("GROQ_API_KEY", "GOOGLE_API_KEY", "TOGETHER_API_KEY"):
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass


def _run_async(coro):
    """Run an async coroutine from Streamlit's sync context, isolated in a fresh
    thread + event loop to avoid loop conflicts (notably with gRPC / Gemini)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


_load_secrets_into_env()

PROVIDER_LABELS = {p: get_config(p).label for p in available_providers()}


# ---------------------------------------------------------------------------
# Sidebar — live benchmark controls
# ---------------------------------------------------------------------------
st.sidebar.title("⚡ llm-bench")
st.sidebar.caption("Benchmark free LLM APIs by speed, throughput, and cost.")
st.sidebar.subheader("Run a live benchmark")

prompt_choice = st.sidebar.selectbox(
    "Prompt", options=list(PROMPT_SETS) + ["custom"], index=1
)
if prompt_choice == "custom":
    prompt_text = st.sidebar.text_area("Custom prompt", "Explain attention in transformers.")
else:
    prompt_text = PROMPT_SETS[prompt_choice]
    st.sidebar.caption(f"“{prompt_text[:90]}…”")

selected = st.sidebar.multiselect(
    "Providers",
    options=available_providers(),
    default=available_providers(),
    format_func=lambda p: PROVIDER_LABELS[p],
)
runs = st.sidebar.slider("Runs per provider", 1, 5, 3)
run_clicked = st.sidebar.button("Run benchmark", type="primary", use_container_width=True)

st.sidebar.divider()
view = st.sidebar.radio("Show", ["All results", "Demo only", "Live only"], index=0)

# --- Handle a live run (writes is_demo=0; falls through to render fresh data) ---
if run_clicked:
    if not selected:
        st.sidebar.error("Select at least one provider.")
    else:
        with st.spinner(f"Benchmarking {len(selected)} providers × {runs} runs…"):
            try:
                _run_async(run_benchmark(prompt_text, selected, runs))
                st.toast("Benchmark complete", icon="✅")
            except Exception as exc:
                st.sidebar.error(f"Benchmark failed: {exc}")


# ---------------------------------------------------------------------------
# Main — header
# ---------------------------------------------------------------------------
st.title("⚡ LLM Inference Benchmarker")
st.markdown(
    "**Which free LLM API is actually fastest?** Same prompt, fired at every "
    "provider, measured for latency, throughput, and cost. Data below is real — "
    "run your own from the sidebar."
)

rows = fetch_runs()
if not rows:
    st.info("No benchmark data yet. Run one from the sidebar.")
    st.stop()

df = pd.DataFrame(rows)
if view == "Demo only":
    df = df[df["is_demo"] == 1]
elif view == "Live only":
    df = df[df["is_demo"] == 0]

if df.empty:
    st.warning("No rows for this filter. Try 'All results' or run a live benchmark.")
    st.stop()

df["label"] = df["provider"].map(PROVIDER_LABELS).fillna(df["provider"])


# ---------------------------------------------------------------------------
# Headline metric cards (aggregated per provider over the filtered rows)
# ---------------------------------------------------------------------------
agg = (
    df.groupby("label")
    .agg(
        ttft=("ttft_ms_p50", "mean"),
        tps=("tokens_per_sec_p50", "mean"),
        cost=("cost_usd_per_1k_tokens", "mean"),
        success=("success_rate", "mean"),
    )
    .reset_index()
)

c1, c2, c3 = st.columns(3)
ttft_valid = agg.dropna(subset=["ttft"])
if not ttft_valid.empty:
    best = ttft_valid.loc[ttft_valid["ttft"].idxmin()]
    c1.metric("Fastest first token", best["label"], f"{best['ttft']:.0f} ms")
tps_valid = agg.dropna(subset=["tps"])
if not tps_valid.empty:
    best = tps_valid.loc[tps_valid["tps"].idxmax()]
    c2.metric("Highest throughput", best["label"], f"{best['tps']:.0f} tok/s")
cost_valid = agg.dropna(subset=["cost"])
if not cost_valid.empty:
    best = cost_valid.loc[cost_valid["cost"].idxmin()]
    c3.metric("Cheapest", best["label"], f"${best['cost']:.4f} /1K tok")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Time to first token")
    st.caption("Lower is better — how long before streaming starts.")
    fig = px.bar(
        agg.dropna(subset=["ttft"]), x="label", y="ttft", color="label",
        labels={"ttft": "TTFT (ms)", "label": ""},
    )
    fig.update_layout(showlegend=False, height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Throughput")
    st.caption("Higher is better — tokens generated per second.")
    fig = px.bar(
        agg.dropna(subset=["tps"]), x="label", y="tps", color="label",
        labels={"tps": "tokens / sec", "label": ""},
    )
    fig.update_layout(showlegend=False, height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Cost efficiency")
st.caption("Estimated cost per 1,000 tokens, from public list pricing.")
fig = px.bar(
    agg.dropna(subset=["cost"]), x="label", y="cost", color="label",
    labels={"cost": "USD / 1K tokens", "label": ""},
)
fig.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Full results table
# ---------------------------------------------------------------------------
st.subheader("All runs")
table = pd.DataFrame({
    "Provider": df["label"],
    "Prompt": df["prompt"].str.slice(0, 45) + "…",
    "Runs": df["runs"],
    "TTFT p50 (ms)": df["ttft_ms_p50"].round(0),
    "TTFT p95 (ms)": df["ttft_ms_p95"].round(0),
    "tok/s": df["tokens_per_sec_p50"].round(0),
    "TBT (ms)": df["tbt_ms_p50"].round(1),
    "Cost/1K ($)": df["cost_usd_per_1k_tokens"].round(5),
    "Success": (df["success_rate"] * 100).round(0).astype("Int64").astype(str) + "%",
    "Type": df["is_demo"].map({1: "Demo", 0: "🔴 Live"}),
})
st.dataframe(table, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sample responses + methodology
# ---------------------------------------------------------------------------
with st.expander("Sample responses"):
    for _, r in df.iterrows():
        if r["sample_response"]:
            st.markdown(f"**{r['label']}** — _{r['prompt'][:60]}…_")
            st.text(r["sample_response"][:600])
            st.divider()

with st.expander("How these metrics are measured"):
    st.markdown(
        "- **TTFT** — time from request to the first streamed token; the latency a "
        "user feels. Reported as p50 (typical) and p95 (tail) across repeated runs.\n"
        "- **Throughput (tok/s)** — output tokens over the full generation time. "
        "Measured client-side from the stream; most meaningful on substantial "
        "responses, which is why the benchmark prompts are multi-paragraph.\n"
        "- **TBT** — mean gap between streamed chunks. Some providers buffer their "
        "stream into few large chunks, so TBT isn't always available.\n"
        "- **Cost** — estimated from public per-token list pricing, not billed usage.\n"
        "- **Demo vs 🔴 Live** — demo rows are pre-recorded real runs that load "
        "instantly; live rows are benchmarks you ran just now."
    )