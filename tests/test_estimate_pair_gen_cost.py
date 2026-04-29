"""Unit tests for `scripts.estimate_pair_gen_cost` (Task X.06).

The script is a thin projection over `generate_pairs.py`'s sampling
helpers — its only original logic is the token + dollar arithmetic.
That arithmetic is what these tests pin down.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts import estimate_pair_gen_cost as ep


def _write_chunks(path: Path, n: int, token_count: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            rec = {
                "chunk_id": f"c{i}",
                "publisher": "rba",
                "title": "x",
                "section_heading": "",
                "text": "blah" * 50,
                "token_count": token_count,
            }
            f.write(json.dumps(rec) + "\n")


def test_project_zero_when_nothing_selected(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    _write_chunks(chunks, n=10, token_count=10)  # below default min_tokens
    out = tmp_path / "out.jsonl"
    eval_dir = tmp_path / "eval"

    est = ep.project(
        chunks_path=chunks,
        out_path=out,
        eval_dir=eval_dir,
        sample=100,
        per_chunk=2,
        min_tokens=100,  # excludes everything
        seed=42,
    )
    assert est.n_chunks == 0
    assert est.n_pairs_target == 0
    assert est.total_input_tokens == 0


def test_project_token_math_matches_pricing(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    _write_chunks(chunks, n=50, token_count=200)
    out = tmp_path / "out.jsonl"
    eval_dir = tmp_path / "eval"

    est = ep.project(
        chunks_path=chunks,
        out_path=out,
        eval_dir=eval_dir,
        sample=10,
        per_chunk=2,
        min_tokens=100,
        seed=42,
    )

    assert est.n_chunks == 10
    assert est.n_pairs_target == 20
    # Input per call = system_tokens + user_overhead + chunk_token_count
    expected_per_call = ep._SYSTEM_TOKENS + ep._USER_PROMPT_OVERHEAD_TOKENS + 200
    assert est.total_input_tokens == 10 * expected_per_call
    # Output = n_pairs × per-query estimate
    assert est.total_output_tokens == 20 * ep._OUTPUT_TOKENS_PER_QUERY

    # Anthropic Haiku 4.5: $1/$5 per 1M
    anthropic_cost = est.cost_for(ep._PRICING["anthropic"])
    expected_anthropic = (
        est.total_input_tokens / 1_000_000 * 1.00
        + est.total_output_tokens / 1_000_000 * 5.00
    )
    assert anthropic_cost == expected_anthropic

    # Gemini 2.5 Flash: $0.30/$2.50 per 1M
    gemini_cost = est.cost_for(ep._PRICING["gemini"])
    expected_gemini = (
        est.total_input_tokens / 1_000_000 * 0.30
        + est.total_output_tokens / 1_000_000 * 2.50
    )
    assert gemini_cost == expected_gemini

    # Gemini should be cheaper than Anthropic on the same workload —
    # if this ever flips, the user is probably looking at a price
    # change that should be reflected in _PRICING.
    assert gemini_cost < anthropic_cost


def test_pricing_table_has_both_providers():
    """Catch a future careless removal of either provider from the table."""
    assert "anthropic" in ep._PRICING
    assert "gemini" in ep._PRICING
    for p in ep._PRICING.values():
        assert p.input_per_m > 0
        assert p.output_per_m > 0


def test_format_report_handles_empty(tmp_path):
    """Empty estimate prints a 'nothing to spend' line, not zeros."""
    import argparse

    args = argparse.Namespace(chunks=Path("nope.jsonl"))
    est = ep.Estimate(0, 0, 0, 0, 0.0, 0.0)
    report = ep._format_report(est, args)
    assert "Nothing to spend" in report
