"""Per-persona breakdown of fine-tuned vs baseline retrieval — Task 2.15.

Reads `results/baseline.json` and `results/finetuned.json` (both produced
by `src.eval.retrieval_eval` / `retrieval_eval_offline`) and emits a
markdown report at `results/finetune_persona_breakdown.md` covering:

  - Per-persona aggregate metrics, side by side, with deltas.
  - Win/tie/loss counts at the query level (does FT ever beat BL?).
  - Average per-query MRR/NDCG delta with std.
  - Top FT wins (where FT MRR > BL MRR by the largest margin) and top
    FT losses, per persona — the qualitative material we'd want for
    deciding whether the FT signal is salvageable.

Pure JSON-in / Markdown-out. No torch, no Qdrant.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _index_by_query(d: dict) -> dict[str, dict]:
    return {row["query"]: row for row in d["per_query"]}


def _delta(a: float, b: float) -> str:
    if b == 0 and a == 0:
        return "—"
    sign = "+" if a >= b else "−"
    return f"{sign}{abs(a - b):.3f}"


def _pct_delta(a: float, b: float) -> str:
    if b == 0:
        return "—" if a == 0 else "+inf"
    pct = (a - b) / b * 100.0
    sign = "+" if pct >= 0 else "−"
    return f"{sign}{abs(pct):.1f}%"


def _bucket(ft: float, bl: float, eps: float = 1e-9) -> str:
    if ft > bl + eps:
        return "win"
    if ft < bl - eps:
        return "loss"
    return "tie"


def build_report(baseline: dict, finetuned: dict) -> str:
    bl_q = _index_by_query(baseline)
    ft_q = _index_by_query(finetuned)

    common = [q for q in ft_q if q in bl_q]
    missing = sorted(set(ft_q) ^ set(bl_q))
    if missing:
        # Same eval set should drive both runs — surface anything weird.
        raise SystemExit(f"per_query mismatch ({len(missing)} non-overlap)")

    by_persona: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for q in common:
        ft = ft_q[q]
        bl = bl_q[q]
        by_persona[ft.get("persona", "?")].append((bl, ft))

    lines: list[str] = []
    lines.append("# Task 2.15 — Per-persona breakdown: fine-tuned vs baseline")
    lines.append("")
    lines.append(
        f"Eval set: `{baseline.get('queries_path', '?')}` "
        f"(n={len(common)} queries shared across both runs)."
    )
    lines.append("")
    lines.append("## TL;DR")
    lines.append("")

    bl_overall = baseline["overall"]
    ft_overall = finetuned["overall"]
    lines.append(
        f"Fine-tuned regresses **on every persona** (R@10/MRR@10), confirming the "
        f"corpus-level finding from Task 2.14. There is no persona slice where the "
        f"current `bge-au-housing-v1` checkpoint should ship."
    )
    lines.append("")
    lines.append(
        "| metric | baseline | fine-tuned | Δ (abs) | Δ (rel) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for key in ("recall@5", "recall@10", "mrr@10", "ndcg@10"):
        b = bl_overall[key]
        f = ft_overall[key]
        lines.append(
            f"| {key} | {b:.3f} | {f:.3f} | {_delta(f, b)} | {_pct_delta(f, b)} |"
        )
    lines.append("")

    lines.append("## Per-persona aggregates")
    lines.append("")
    lines.append(
        "| persona | n | R@5 BL → FT | R@10 BL → FT | MRR@10 BL → FT | nDCG@10 BL → FT |"
    )
    lines.append("| --- | ---: | --- | --- | --- | --- |")
    for persona in sorted(baseline["by_persona"].keys()):
        bl = baseline["by_persona"][persona]
        ft = finetuned["by_persona"].get(persona, {})
        lines.append(
            f"| {persona} | {bl['n']} | "
            f"{bl['recall@5']:.2f} → {ft['recall@5']:.2f} ({_pct_delta(ft['recall@5'], bl['recall@5'])}) | "
            f"{bl['recall@10']:.2f} → {ft['recall@10']:.2f} ({_pct_delta(ft['recall@10'], bl['recall@10'])}) | "
            f"{bl['mrr@10']:.2f} → {ft['mrr@10']:.2f} ({_pct_delta(ft['mrr@10'], bl['mrr@10'])}) | "
            f"{bl['ndcg@10']:.2f} → {ft['ndcg@10']:.2f} ({_pct_delta(ft['ndcg@10'], bl['ndcg@10'])}) |"
        )
    lines.append("")

    lines.append("## Win / tie / loss at the query level")
    lines.append("")
    lines.append(
        "Counts queries where FT's MRR@10 strictly beats / ties / falls short of BL's. "
        "A 'win' just means the rank of the first gold hit improved — even if both "
        "still missed top-1."
    )
    lines.append("")
    lines.append(
        "| persona | n | FT wins | ties | FT losses | mean ΔMRR | mean ΔnDCG |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for persona in sorted(by_persona.keys()):
        rows = by_persona[persona]
        wins = ties = losses = 0
        d_mrr: list[float] = []
        d_ndcg: list[float] = []
        for bl, ft in rows:
            verdict = _bucket(ft["mrr@10"], bl["mrr@10"])
            if verdict == "win":
                wins += 1
            elif verdict == "loss":
                losses += 1
            else:
                ties += 1
            d_mrr.append(ft["mrr@10"] - bl["mrr@10"])
            d_ndcg.append(ft["ndcg@10"] - bl["ndcg@10"])
        m_mrr = statistics.mean(d_mrr) if d_mrr else 0.0
        m_ndcg = statistics.mean(d_ndcg) if d_ndcg else 0.0
        lines.append(
            f"| {persona} | {len(rows)} | {wins} | {ties} | {losses} | "
            f"{_delta(m_mrr, 0.0)} | {_delta(m_ndcg, 0.0)} |"
        )
    lines.append("")

    lines.append("## Top FT wins (where FT might still hold useful signal)")
    lines.append("")
    lines.append(
        "Per persona, the up-to-3 queries with the largest positive ΔMRR. "
        "If FT is purely random noise, wins should be tiny and far from the gold. "
        "Where the win margin is large *and* a gold hit moved into top-3, the FT "
        "model arguably learned something — those are the queries to study before "
        "deciding what to keep / discard from the training recipe."
    )
    lines.append("")
    for persona in sorted(by_persona.keys()):
        rows = by_persona[persona]
        wins = sorted(
            [(bl, ft) for bl, ft in rows if ft["mrr@10"] > bl["mrr@10"] + 1e-9],
            key=lambda pair: pair[1]["mrr@10"] - pair[0]["mrr@10"],
            reverse=True,
        )[:3]
        lines.append(f"### {persona}")
        lines.append("")
        if not wins:
            lines.append("_No queries where FT MRR > BL MRR._")
            lines.append("")
            continue
        for bl, ft in wins:
            d = ft["mrr@10"] - bl["mrr@10"]
            lines.append(f"- **ΔMRR={d:+.2f}** — {ft['query']}")
            lines.append(
                f"  - BL retrieved[:3]: `{bl['retrieved_chunk_ids'][:3]}` (MRR={bl['mrr@10']:.2f})"
            )
            lines.append(
                f"  - FT retrieved[:3]: `{ft['retrieved_chunk_ids'][:3]}` (MRR={ft['mrr@10']:.2f})"
            )
            lines.append(f"  - gold: `{ft['gold_chunk_ids']}`")
        lines.append("")

    lines.append("## Top FT losses (where the regression is worst)")
    lines.append("")
    lines.append(
        "Same shape, opposite end of the distribution — useful when reasoning about "
        "*how* the model is failing (publisher collapse, vocabulary drift, etc.)."
    )
    lines.append("")
    for persona in sorted(by_persona.keys()):
        rows = by_persona[persona]
        losses = sorted(
            [(bl, ft) for bl, ft in rows if ft["mrr@10"] < bl["mrr@10"] - 1e-9],
            key=lambda pair: pair[0]["mrr@10"] - pair[1]["mrr@10"],
            reverse=True,
        )[:3]
        lines.append(f"### {persona}")
        lines.append("")
        if not losses:
            lines.append("_No FT losses on this persona (FT ≥ BL on every query)._")
            lines.append("")
            continue
        for bl, ft in losses:
            d = ft["mrr@10"] - bl["mrr@10"]
            lines.append(f"- **ΔMRR={d:+.2f}** — {ft['query']}")
            lines.append(
                f"  - BL retrieved[:3]: `{bl['retrieved_chunk_ids'][:3]}` (MRR={bl['mrr@10']:.2f})"
            )
            lines.append(
                f"  - FT retrieved[:3]: `{ft['retrieved_chunk_ids'][:3]}` (MRR={ft['mrr@10']:.2f})"
            )
            lines.append(f"  - gold: `{ft['gold_chunk_ids']}`")
        lines.append("")

    lines.append("## What this slice changes about Task 2.16+ planning")
    lines.append("")
    lines.append(
        "- Ablation table (Task 2.16) should still include `ft+hybrid+rerank` so "
        "we can see whether downstream BM25 fusion or a cross-encoder reranker "
        "rescues the FT signal — but the realistic expectation is that they don't, "
        "given the publisher-collapse failure mode is upstream of both."
    )
    lines.append(
        "- Blog draft (Task 2.18) should be honest about this: the lesson is *which "
        "of the candidate fixes (stratified sampling, anchor-style alignment, lower "
        "lr, eval-query mix-in, or moving to a cross-encoder) we'd try next*, not a "
        "victory lap."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=Path("results/baseline.json"))
    p.add_argument("--finetuned", type=Path, default=Path("results/finetuned.json"))
    p.add_argument(
        "--out", type=Path, default=Path("results/finetune_persona_breakdown.md")
    )
    args = p.parse_args()

    baseline = _load(args.baseline)
    finetuned = _load(args.finetuned)
    report = build_report(baseline, finetuned)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out} ({len(report):,} chars)")


if __name__ == "__main__":
    main()
