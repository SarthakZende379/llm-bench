"""
prompts.py — benchmark prompt sets.

The default ("medium") prompt is deliberately substantial: it asks for a
multi-paragraph answer so providers stream enough tokens for throughput
(tokens/sec) to be measured meaningfully. Very short prompts produce only 1-2
streamed chunks on providers that buffer (e.g. Gemini), which makes decode
throughput unreliable — see metrics.py methodology notes.
"""

DEFAULT_PROMPT = (
    "Explain how a large language model generates text, step by step: "
    "tokenization, the forward pass through the transformer layers, how the next "
    "token is chosen, and why generation happens one token at a time. "
    "Write three short paragraphs in plain language."
)

# Different output lengths, for comparing how providers behave under different
# loads. Keyed by a short label shown in the dashboard.
PROMPT_SETS: dict[str, str] = {
    "short": "In one sentence, what is time-to-first-token?",
    "medium": DEFAULT_PROMPT,
    "long": (
        "Write a clear, well-structured explanation (around six paragraphs) of how "
        "modern LLM inference is optimized for speed: continuous batching, KV "
        "caching, quantization, and speculative decoding. For each technique, "
        "explain what it does and the trade-offs involved."
    ),
}


def get_prompt(label: str = "medium") -> str:
    """Return a prompt by label, defaulting to the medium benchmark prompt."""
    return PROMPT_SETS.get(label, DEFAULT_PROMPT)