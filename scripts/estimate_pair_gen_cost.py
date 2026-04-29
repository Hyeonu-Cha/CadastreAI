"""Estimate the cost of running `generate_pairs.py` without making API calls.

Reads `data/processed/chunks.jsonl`, applies the same sampling rules
as `src.training.generate_pairs` (eval-gold exclusion, min-tokens
filter, idempotency skip, random sample with the same seed), then
projects total token usage and per-provider dollar amounts using
cached pricing.

Run before committing API credit:

    python -m scripts.estimate_pair_gen_cost --sample 2000 -n 2

Defaults match `generate_pairs.py` so the projection matches a real
run on the same arguments.

The numbers are projections, not contracts:
- Token counts use each chunk's pre-computed `token_count` (the
  tiktoken cl100k_base count from chunking time) — that's the input
  size for Anthropic; Gemini's tokenizer is similar in this regime
  (English prose) but not identical, so call it ±10%.
- Output tokens are estimated at ~150 per query × N queries per chunk.
  The model usually produces less; this is a deliberate over-estimate.
- Pricing is cached at the date below — verify on the vendor pricing
  page before relying on it for procurement decisions.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from src.training.generate_pairs import (
    _SYSTEM_PROMPT,
    DEFAULT_MIN_TOKENS,
    DEFAULT_PER_CHUNK,
    DEFAULT_SAMPLE,
    _load_chunks,
    _load_eval_gold_ids,
    _load_existing_chunk_ids,
    _select_chunks,
)

log = logging.getLogger(__name__)

# Heuristic constants used for the per-call overhead. The system
# prompt is sent once per call (no caching across providers in the
# pair-gen path), and the user prompt has fixed boilerplate around the
# variable chunk text.
_SYSTEM_TOKENS = max(1, len(_SYSTEM_PROMPT) // 4)  # ~chars/4 heuristic
_USER_PROMPT_OVERHEAD_TOKENS = 60  # metadata header + closing instruction
_OUTPUT_TOKENS_PER_QUERY = 150  # generous; real output is usually shorter


# ---------------------------------------------------------------------
# Pricing — USD per 1M tokens. Cached 2026-04-29; verify at vendor
# pricing pages before relying on these numbers.
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Pricing:
    name: str
    input_per_m: float
    output_per_m: float


_PRICING: dict[str, Pricing] = {
    "anthropic": Pricing(
        name="Anthropic Claude Haiku 4.5",
        input_per_m=1.00,
        output_per_m=5.00,
    ),
    "gemini": Pricing(
        name="Gemini 2.5 Flash",
        # Conservative: thinking-on output rate. Without thinking the
        # output rate is materially lower; treat this as a ceiling.
        input_per_m=0.30,
        output_per_m=2.50,
    ),
}


@dataclass
class Estimate:
    n_chunks: int
    n_pairs_target: int
    total_input_tokens: int
    total_output_tokens: int
    avg_input_tokens_per_call: float
    avg_chunk_token_count: float

    def cost_for(self, p: Pricing) -> float:
        return (
            self.total_input_tokens / 1_000_000 * p.input_per_m
            + self.total_output_tokens / 1_000_000 * p.output_per_m
        )


def project(
    chunks_path: Path,
    out_path: Path,
    eval_dir: Path,
    *,
    sample: int,
    per_chunk: int,
    min_tokens: int,
    seed: int,
) -> Estimate:
    chunks = _load_chunks(chunks_path)
    eval_gold = _load_eval_gold_ids(eval_dir)
    existing = _load_existing_chunk_ids(out_path)
    excluded = eval_gold | existing
    selected = _select_chunks(chunks, sample, min_tokens, excluded, seed)

    if not selected:
        return Estimate(0, 0, 0, 0, 0.0, 0.0)

    total_input = 0
    total_chunk_tokens = 0
    for c in selected:
        chunk_tokens = int(c.get("token_count") or 0)
        total_chunk_tokens += chunk_tokens
        total_input += _SYSTEM_TOKENS + _USER_PROMPT_OVERHEAD_TOKENS + chunk_tokens

    n_pairs = len(selected) * per_chunk
    total_output = n_pairs * _OUTPUT_TOKENS_PER_QUERY

    return Estimate(
        n_chunks=len(selected),
        n_pairs_target=n_pairs,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        avg_input_tokens_per_call=total_input / len(selected),
        avg_chunk_token_count=total_chunk_tokens / len(selected),
    )


def _format_report(est: Estimate, args: argparse.Namespace) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"Pair-generation cost estimate — {args.chunks}")
    lines.append("=" * 72)
    if est.n_chunks == 0:
        lines.append("No chunks selected (all excluded or below --min-tokens). Nothing to spend.")
        return "\n".join(lines)
    lines.append(f"  Chunks selected:           {est.n_chunks:>10,}")
    lines.append(f"  Target pairs (n × chunks): {est.n_pairs_target:>10,}")
    lines.append(f"  Avg chunk tokens:          {est.avg_chunk_token_count:>10,.1f}")
    lines.append(f"  Avg input tokens / call:   {est.avg_input_tokens_per_call:>10,.1f}")
    lines.append(f"  Total input tokens:        {est.total_input_tokens:>10,}")
    lines.append(f"  Total output tokens:       {est.total_output_tokens:>10,}")
    lines.append("-" * 72)
    lines.append(f"  {'Provider':<32} {'Input':>10}  {'Output':>10}  {'Total':>10}")
    lines.append("-" * 72)
    for key in ("anthropic", "gemini"):
        p = _PRICING[key]
        in_cost = est.total_input_tokens / 1_000_000 * p.input_per_m
        out_cost = est.total_output_tokens / 1_000_000 * p.output_per_m
        total = in_cost + out_cost
        lines.append(
            f"  {p.name:<32} ${in_cost:>9.2f}  ${out_cost:>9.2f}  ${total:>9.2f}"
        )
    lines.append("-" * 72)
    lines.append("Pricing cached 2026-04-29. Verify on vendor pricing pages before")
    lines.append("relying on these numbers for procurement.")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/training/pairs_raw.jsonl"))
    p.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    p.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    p.add_argument("-n", "--per-chunk", type=int, default=DEFAULT_PER_CHUNK)
    p.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    est = project(
        chunks_path=args.chunks,
        out_path=args.out,
        eval_dir=args.eval_dir,
        sample=args.sample,
        per_chunk=args.per_chunk,
        min_tokens=args.min_tokens,
        seed=args.seed,
    )
    print(_format_report(est, args))


if __name__ == "__main__":
    main()
