"""
providers.py — LangChain provider factory + streaming benchmark runner.

Maps a provider key ("groq" / "gemini" / "together") to a configured LangChain
chat model, then runs a single streaming generation while stamping
time.perf_counter() at request-send, first-token, and every subsequent chunk.
Returns a populated RawRunResult (defined in pipeline.py).

Extensibility
-------------
Adding a provider = add one entry to PROVIDER_REGISTRY and one branch to
_build_model(). The pipeline never changes.

Keys
----
Read from environment variables (loaded from .env locally, or injected from
Streamlit Secrets in the deployed app). Each LangChain model reads its own
standard variable automatically:
    Groq     -> GROQ_API_KEY
    Gemini   -> GOOGLE_API_KEY
    Together -> TOGETHER_API_KEY

Token-count caveat
------------------
Output tokens are approximated by the number of streamed chunks. For these
providers (which stream roughly one token per chunk) this is a close, honest
proxy and keeps the runner provider-agnostic. A v1.1 improvement could read
exact usage_metadata where the provider returns it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from pipeline import RawRunResult

load_dotenv()  # populate os.environ from .env if present


# ---------------------------------------------------------------------------
# Tunables — fixed output length keeps runs comparable and bounds free-tier use
# ---------------------------------------------------------------------------

MAX_TOKENS = 512  # generous cap; thinking models (Gemini 3.x) need room before visible output
TEMPERATURE = 0.7


# ---------------------------------------------------------------------------
# Provider configuration
# Model identifiers verified current (May 2026). If a provider returns a 400
# "model not found", swap the model string here — that's the only change needed.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderConfig:
    label: str               # display name for the dashboard
    model: str               # model identifier sent to the API
    env_key: str             # environment variable holding the API key
    price_in_per_m: float    # USD per 1M input tokens  (public list price)
    price_out_per_m: float   # USD per 1M output tokens (public list price)


PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "groq": ProviderConfig(
        label="Groq",
        model="llama-3.1-8b-instant",        # alt: "llama-3.3-70b-versatile"
        env_key="GROQ_API_KEY",
        price_in_per_m=0.05,
        price_out_per_m=0.08,
    ),
    "gemini": ProviderConfig(
        label="Gemini",
        model="gemini-3.5-flash",            # current model per ai.google.dev (alt: "gemini-3.5-flash-lite")
        env_key="GOOGLE_API_KEY",
        price_in_per_m=0.075,                # estimate — update from ai.google.dev/pricing once confirmed
        price_out_per_m=0.30,
    ),
    "together": ProviderConfig(
        label="Together AI",
        model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",  # current serverless 8B (per Together docs)
        env_key="TOGETHER_API_KEY",
        price_in_per_m=0.10,
        price_out_per_m=0.10,
    ),
}


def get_config(provider: str) -> ProviderConfig:
    """Look up a provider's config, with a clear error for unknown keys."""
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider '{provider}'. Known: {', '.join(PROVIDER_REGISTRY)}"
        )
    return PROVIDER_REGISTRY[provider]


def get_pricing(provider: str) -> tuple[float, float]:
    """Return (input_price_per_1M, output_price_per_1M). Used by metrics.py."""
    cfg = get_config(provider)
    return cfg.price_in_per_m, cfg.price_out_per_m


def available_providers() -> list[str]:
    """All registered provider keys — handy for the dashboard's checkboxes."""
    return list(PROVIDER_REGISTRY)


# ---------------------------------------------------------------------------
# Model factory (lazy imports keep startup fast — only the package for the
# provider actually being used gets imported)
# ---------------------------------------------------------------------------


def _build_model(provider: str, cfg: ProviderConfig):
    if not os.getenv(cfg.env_key):
        raise RuntimeError(
            f"Missing {cfg.env_key} in environment. "
            f"Add it to .env (local) or Streamlit Secrets (deployed)."
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=cfg.model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # thinking_budget=0 disables reasoning tokens so (a) the answer is visible
        # within the token cap and (b) we measure generation speed, not thinking time
        # — a fair comparison against the non-thinking Llama models. If your installed
        # langchain-google-genai is too old to accept this kwarg, delete that one line.
        return ChatGoogleGenerativeAI(
            model=cfg.model,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
            thinking_budget=0,
        )

    if provider == "together":
        from langchain_together import ChatTogether
        return ChatTogether(model=cfg.model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    raise ValueError(f"No model builder for provider '{provider}'")


def _extract_chunk_text(content) -> str:
    """Pull plain text from a streaming chunk's .content.

    LangChain chunk content can be a plain str OR a list of content blocks
    (strings, or dicts like {"type": "text", "text": "..."}). Newer models —
    Gemini 3.x in particular — use the list form, so a naive isinstance(str)
    check silently drops every token. This handles both shapes.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


# ---------------------------------------------------------------------------
# The benchmark runner — one streaming generation, fully instrumented
# ---------------------------------------------------------------------------


async def run_provider_benchmark(provider: str, prompt: str) -> RawRunResult:
    """Run ONE streaming generation and capture per-token timing.

    Timing model (all via time.perf_counter, monotonic seconds):
        request_sent_at  : immediately before streaming begins
        first_token_at   : arrival of the first non-empty chunk  -> TTFT
        token_timestamps : arrival time of every non-empty chunk -> TBT
        last_token_at    : arrival of the final chunk            -> total latency

    Raises on transport/auth/model errors; the pipeline's benchmark_node catches
    those and records a failed run so other providers still complete.
    """
    cfg = get_config(provider)
    model = _build_model(provider, cfg)

    chunks: list[str] = []
    token_timestamps: list[float] = []
    first_token_at: float | None = None

    request_sent_at = time.perf_counter()
    async for chunk in model.astream(prompt):
        now = time.perf_counter()
        text = _extract_chunk_text(chunk.content)
        if text:
            if first_token_at is None:
                first_token_at = now
            token_timestamps.append(now)
            chunks.append(text)

    last_token_at = token_timestamps[-1] if token_timestamps else None
    response_text = "".join(chunks)

    output_tokens = len(token_timestamps)        # streamed chunks ~= output tokens
    input_tokens = max(1, len(prompt) // 4)       # ~4 chars/token heuristic

    return RawRunResult(
        provider=provider,
        model=cfg.model,
        success=bool(response_text),
        request_sent_at=request_sent_at,
        first_token_at=first_token_at,
        last_token_at=last_token_at,
        token_timestamps=token_timestamps,
        response_text=response_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=None if response_text else "Empty response from provider",
    )


if __name__ == "__main__":
    # Smoke test a single provider:  python providers.py [groq|gemini|together]
    import asyncio
    import sys

    async def _smoke(name: str) -> None:
        result = await run_provider_benchmark(name, "Say hello in one short sentence.")
        print(f"provider = {result.provider}  ({result.model})")
        print(f"success  = {result.success}")
        print(f"chunks   = {result.output_tokens}")
        print(f"text     = {result.response_text[:80]!r}")
        if result.first_token_at is not None:
            ttft_ms = (result.first_token_at - result.request_sent_at) * 1000
            print(f"TTFT     = {ttft_ms:.1f} ms")
        if result.error:
            print(f"error    = {result.error}")

    provider_name = sys.argv[1] if len(sys.argv) > 1 else "groq"
    asyncio.run(_smoke(provider_name))