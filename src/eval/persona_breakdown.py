"""Per-persona delta between two retrieval-eval JSONs.

Compares two `src.eval.retrieval_eval` output files side-by-side per
persona (homebuyer / investor / researcher) and emits a Markdown table
plus a JSON delta. Designed for the Week 2 fine-tune story: feed it
`results/hybrid.json` (or whatever base-BGE result you want) as
`--baseline` and `results/finetuned.json` as `--finetuned`, and it tells
you which user persona benefits most.

The metric set comes from whatever both inputs already carry: R@5,
R@10, MRR@10, nDCG@10. Persona n_count is taken from `by_persona[*].n`,
so unseen personas are silently skipped.

    python -m src.eval.persona_breakdown \
        --baseline results/hybrid.json \
        --finetuned results/finetuned.json \
        --out results/persona_breakdown.md \
        --out-json results/persona_breakdown.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

_METRICS = ("recall@5", "recall@10", "mrr@10", "ndcg@10")
_PERSONAS = ("homebuyer", "investor", "researcher")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(metric: str, base: dict, ft: dict) -> dict:
    b = float(base.get(metric, 0.0))
    f = float(ft.get(metric, 0.0))
    return {
        "metric": metric,
        "base": b,
        "finetuned": f,
        "delta": f - b,
        "delta_pct_pts": 100.0 * (f - b),
    }


def _persona_section(
    persona: str, base_by_persona: dict, ft_by_persona: dict
) -> dict | None:
    base = base_by_persona.get(persona)
    ft = ft_by_persona.get(persona)
    if base is None or ft is None:
        return None
    return {
        "persona": persona,
        "n_base": int(base.get("n", 0)),
        "n_finetuned": int(ft.get("n", 0)),
        "metrics": [_row(m, base, ft) for m in _METRICS],
    }


def _format_md(
    base_label: str,
    ft_label: str,
    overall: dict,
    sections: list[dict],
) -> str:
    lines: list[str] = []
    lines.append(f"# Per-persona breakdown: {base_label} → {ft_label}\n")

    # Overall
    lines.append("## Overall\n")
    lines.append("| Metric | Base | Fine-tuned | Δ pts |")
    lines.append("|--------|-----:|-----------:|------:|")
    for r in overall["metrics"]:
        lines.append(
            f"| {r['metric']} | {r['base']:.3f} | {r['finetuned']:.3f} | "
            f"{r['delta_pct_pts']:+.1f} |"
        )
    lines.append("")

    # Per-persona
    for s in sections:
        lines.append(f"## {s['persona']} (n={s['n_finetuned']})\n")
        lines.append("| Metric | Base | Fine-tuned | Δ pts |")
        lines.append("|--------|-----:|-----------:|------:|")
        for r in s["metrics"]:
            lines.append(
                f"| {r['metric']} | {r['base']:.3f} | {r['finetuned']:.3f} | "
                f"{r['delta_pct_pts']:+.1f} |"
            )
        lines.append("")

    # Summary
    biggest = _biggest_lift(sections)
    if biggest:
        persona, metric, delta = biggest
        lines.append(
            f"**Largest lift:** `{metric}` on **{persona}** "
            f"({delta:+.1f} pts).\n"
        )

    return "\n".join(lines).rstrip() + "\n"


def _biggest_lift(sections: list[dict]) -> tuple[str, str, float] | None:
    best = None
    for s in sections:
        for r in s["metrics"]:
            if best is None or r["delta_pct_pts"] > best[2]:
                best = (s["persona"], r["metric"], r["delta_pct_pts"])
    return best


def run(
    baseline_path: Path,
    finetuned_path: Path,
    out_md: Path | None,
    out_json: Path | None,
) -> dict:
    base_doc = _load(baseline_path)
    ft_doc = _load(finetuned_path)

    base_overall = base_doc.get("overall") or {}
    ft_overall = ft_doc.get("overall") or {}
    overall = {
        "n_base": int(base_overall.get("n", 0)),
        "n_finetuned": int(ft_overall.get("n", 0)),
        "metrics": [_row(m, base_overall, ft_overall) for m in _METRICS],
    }

    base_personas = base_doc.get("by_persona") or {}
    ft_personas = ft_doc.get("by_persona") or {}
    sections: list[dict] = []
    for p in _PERSONAS:
        s = _persona_section(p, base_personas, ft_personas)
        if s is not None:
            sections.append(s)
    if not sections:
        log.warning("No persona overlap between %s and %s", baseline_path, finetuned_path)

    summary = {
        "baseline": str(baseline_path),
        "finetuned": str(finetuned_path),
        "overall": overall,
        "by_persona": sections,
        "biggest_lift": _biggest_lift(sections),
    }

    base_label = base_doc.get("retriever", {}).get("kind") or baseline_path.stem
    ft_label = ft_doc.get("retriever", {}).get("kind") or finetuned_path.stem
    md = _format_md(base_label, ft_label, overall, sections)

    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
        log.info("Wrote Markdown report to %s", out_md)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("Wrote JSON delta to %s", out_json)

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--finetuned", type=Path, required=True)
    p.add_argument("--out", dest="out_md", type=Path, default=None)
    p.add_argument("--out-json", dest="out_json", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    summary = run(
        baseline_path=args.baseline,
        finetuned_path=args.finetuned,
        out_md=args.out_md,
        out_json=args.out_json,
    )
    overall = summary["overall"]
    deltas = {r["metric"]: r["delta_pct_pts"] for r in overall["metrics"]}
    big = summary["biggest_lift"]
    print(
        f"baseline={summary['baseline']} finetuned={summary['finetuned']} "
        f"overall_delta_pts={deltas} "
        f"biggest_lift={big[0]}/{big[1]}={big[2]:+.1f}pts" if big else ""
    )


if __name__ == "__main__":
    main()
