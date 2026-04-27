"""Per-query cost accounting for Anthropic LLM calls (Task 4.12).

Each agent run hits 5 `messages.create()` endpoints (classify, decompose,
route, reflect, synthesize). To know whether the agent is economical to
run we need to attribute tokens — and therefore $ — back to each node,
including the cache-write/read split that Task 4.11 introduced.

Pricing model
-------------
Anthropic bills per 1M tokens. Prompt caching adds two extra meters:

  * ``cache_creation_input_tokens`` — written on the first call after a
    cold prefix; charged at 1.25× the base input rate.
  * ``cache_read_input_tokens``     — served from cache on warm calls;
    charged at 0.1× the base input rate.
  * ``input_tokens``                — uncached input; full rate.
  * ``output_tokens``               — assistant response; full rate.

Rates are from the public pricing page (cached 2026-04-15). They live
here as a lookup table because we'd rather have a one-line patch when
Anthropic changes prices than scatter constants across nodes.

The ``calculate_cost`` and ``log_cost`` helpers are intentionally pure —
no I/O beyond a structured ``log.info`` line — so unit tests can pin the
maths without mocking the SDK.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# Per-1M-token rates, USD. Cache write = 1.25× input, cache read = 0.1×.
# Source: Anthropic pricing page, cached 2026-04-15. Update here on price changes.
_PRICING: dict[str, tuple[float, float]] = {
    # model_id : (input_per_1m, output_per_1m)
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


@dataclass(frozen=True)
class CostBreakdown:
    """Per-call cost split, in USD.

    All fields are USD (not micro-dollars) — the per-call totals are
    typically in the $0.0001–$0.05 range, well within float precision."""

    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    input_usd: float
    output_usd: float
    cache_read_usd: float
    cache_write_usd: float
    total_usd: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "input_usd": round(self.input_usd, 6),
            "output_usd": round(self.output_usd, 6),
            "cache_read_usd": round(self.cache_read_usd, 6),
            "cache_write_usd": round(self.cache_write_usd, 6),
            "total_usd": round(self.total_usd, 6),
        }


def _usage_field(usage: Any, name: str) -> int:
    """Read a token field off the SDK's ``Usage`` object or a dict.

    The Anthropic Python SDK returns a Pydantic model; tests pass plain
    dicts. ``getattr`` handles the SDK shape and ``.get`` handles the
    test shape — falling back to 0 means a missing field (e.g. older
    response without cache fields) doesn't crash cost accounting."""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(name) or 0)
    return int(getattr(usage, name, 0) or 0)


def calculate_cost(usage: Any, model: str) -> CostBreakdown:
    """Compute the dollar cost of one ``messages.create`` response.

    Unknown models fall through to a zero-rate breakdown rather than
    crashing — we'd rather lose the cost line than break the agent over
    a typo or a model added after the table was last refreshed."""
    rates = _PRICING.get(model)
    if rates is None:
        log.warning("calculate_cost: no pricing for model=%s; recording $0", model)
        in_rate, out_rate = 0.0, 0.0
    else:
        in_rate, out_rate = rates

    input_tokens = _usage_field(usage, "input_tokens")
    output_tokens = _usage_field(usage, "output_tokens")
    cache_read_tokens = _usage_field(usage, "cache_read_input_tokens")
    cache_write_tokens = _usage_field(usage, "cache_creation_input_tokens")

    input_usd = (input_tokens / 1_000_000) * in_rate
    output_usd = (output_tokens / 1_000_000) * out_rate
    cache_read_usd = (cache_read_tokens / 1_000_000) * in_rate * CACHE_READ_MULTIPLIER
    cache_write_usd = (cache_write_tokens / 1_000_000) * in_rate * CACHE_WRITE_MULTIPLIER
    total = input_usd + output_usd + cache_read_usd + cache_write_usd

    return CostBreakdown(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        input_usd=input_usd,
        output_usd=output_usd,
        cache_read_usd=cache_read_usd,
        cache_write_usd=cache_write_usd,
        total_usd=total,
    )


class CostTracker:
    """Accumulator for per-node cost lines across one agent run.

    Use one instance per query so the running total in
    ``summary()['total_usd']`` is the dollar cost of answering that one
    question. The tracker is intentionally simple — it doesn't persist
    or aggregate across runs; that's the job of whatever consumes the
    structured log lines (e.g. an eval harness)."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, CostBreakdown]] = []

    def record(self, node: str, usage: Any, model: str) -> CostBreakdown:
        breakdown = calculate_cost(usage, model)
        self.entries.append((node, breakdown))
        log.info(
            "cost node=%s model=%s in=%d out=%d cache_r=%d cache_w=%d total=$%.6f",
            node,
            model,
            breakdown.input_tokens,
            breakdown.output_tokens,
            breakdown.cache_read_tokens,
            breakdown.cache_write_tokens,
            breakdown.total_usd,
        )
        return breakdown

    def summary(self) -> dict[str, Any]:
        total = sum(b.total_usd for _, b in self.entries)
        in_tok = sum(b.input_tokens for _, b in self.entries)
        out_tok = sum(b.output_tokens for _, b in self.entries)
        cache_r = sum(b.cache_read_tokens for _, b in self.entries)
        cache_w = sum(b.cache_write_tokens for _, b in self.entries)
        return {
            "n_calls": len(self.entries),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cache_read_tokens": cache_r,
            "cache_write_tokens": cache_w,
            "total_usd": round(total, 6),
            "by_node": [
                {"node": node, **b.as_dict()} for node, b in self.entries
            ],
        }


def log_cost(usage: Any, model: str, node: str) -> CostBreakdown:
    """One-shot helper for nodes that don't carry a tracker.

    Equivalent to ``CostTracker().record(node, usage, model)`` but
    discards the entry — useful when we only want the structured log
    line emitted, not an in-memory accumulator."""
    breakdown = calculate_cost(usage, model)
    log.info(
        "cost node=%s model=%s in=%d out=%d cache_r=%d cache_w=%d total=$%.6f",
        node,
        model,
        breakdown.input_tokens,
        breakdown.output_tokens,
        breakdown.cache_read_tokens,
        breakdown.cache_write_tokens,
        breakdown.total_usd,
    )
    return breakdown


__all__ = [
    "CACHE_READ_MULTIPLIER",
    "CACHE_WRITE_MULTIPLIER",
    "CostBreakdown",
    "CostTracker",
    "calculate_cost",
    "log_cost",
]
