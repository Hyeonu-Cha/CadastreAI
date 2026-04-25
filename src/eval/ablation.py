"""Build the Week 2 ablation: base BGE → hybrid → +rerank → fine-tuned.

Given a list of variant labels and result JSON paths, this script emits
a single `results/ablation.json` summary plus a Markdown table that the
Week 2 blog post and `recall_curves.png` (Task 2.17) consume. Each row
is one variant; columns are `n`, R@5, R@10, MRR@10, nDCG@10, and
latency stats when available.

The default variants (per Task 2.16 spec) are:

    base BGE                  results/baseline.json
    + hybrid                  results/hybrid.json
    + hybrid + rerank         results/reranked.json
    ft + hybrid + rerank      results/finetuned.json   (Task 2.14 output)

Use `--variant LABEL=PATH` to override or add to the list. Variants
whose JSON doesn't exist are skipped with a warning rather than
crashing — useful while the fine-tune is still in flight.

    python -m src.eval.ablation \
        --variant "base BGE=results/baseline.json" \
        --variant "+ hybrid=results/hybrid.json" \
        --variant "+ hybrid + rerank=results/reranked.json" \
        --variant "ft + hybrid + rerank=results/finetuned.json" \
        --out results/ablation.json --out-md results/ablation.md
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

DEFAULT_VARIANTS = (
    ("base BGE", Path("results/baseline.json")),
    ("+ hybrid", Path("results/hybrid.json")),
    ("+ hybrid + rerank", Path("results/reranked.json")),
    ("ft + hybrid + rerank", Path("results/finetuned.json")),
)
_METRIC_KEYS = ("recall@5", "recall@10", "mrr@10", "ndcg@10")


def _parse_variant(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"variant must be 'LABEL=PATH', got {spec!r}"
        )
    label, path = spec.split("=", 1)
    return label.strip(), Path(path.strip())


def _load_variant(label: str, path: Path) -> dict | None:
    if not path.exists():
        log.warning("variant '%s' missing %s — skipping", label, path)
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    overall = doc.get("overall") or {}
    latency = doc.get("latency") or {}
    return {
        "label": label,
        "source": str(path),
        "n": int(overall.get("n", 0)),
        "metrics": {m: float(overall.get(m, 0.0)) for m in _METRIC_KEYS},
        "latency": {
            "median_ms": float(latency.get("median_ms", 0.0)),
            "p95_ms": float(latency.get("p95_ms", 0.0)),
            "mean_ms": float(latency.get("mean_ms", 0.0)),
        },
        "retriever": doc.get("retriever") or {},
    }


def _format_md(variants: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Ablation: retrieval pipeline\n")
    lines.append(
        "All variants share the same query set; n is the count of evaluable "
        "queries. Latency is per-query median / p95 from the eval runs.\n"
    )

    lines.append(
        "| Variant | n | R@5 | R@10 | MRR@10 | nDCG@10 | median ms | p95 ms |"
    )
    lines.append(
        "|---------|--:|----:|-----:|-------:|--------:|----------:|-------:|"
    )
    for v in variants:
        m = v["metrics"]
        lat = v["latency"]
        lines.append(
            f"| {v['label']} | {v['n']} | "
            f"{m['recall@5']:.3f} | {m['recall@10']:.3f} | "
            f"{m['mrr@10']:.3f} | {m['ndcg@10']:.3f} | "
            f"{lat['median_ms']:.0f} | {lat['p95_ms']:.0f} |"
        )
    lines.append("")

    if len(variants) >= 2:
        first = variants[0]
        last = variants[-1]
        m1 = first["metrics"]
        m2 = last["metrics"]
        lines.append(
            f"**End-to-end delta** ({first['label']} → {last['label']}):"
        )
        lines.append("")
        lines.append("| Metric | Δ pts |")
        lines.append("|--------|------:|")
        for m in _METRIC_KEYS:
            lines.append(f"| {m} | {(m2[m] - m1[m]) * 100:+.1f} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run(
    variants: list[tuple[str, Path]],
    out_path: Path | None,
    out_md_path: Path | None,
) -> dict:
    rows: list[dict] = []
    for label, path in variants:
        row = _load_variant(label, path)
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("no variants loaded — check paths")

    summary = {
        "variants": rows,
        "metrics": list(_METRIC_KEYS),
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("Wrote ablation JSON to %s", out_path)

    if out_md_path is not None:
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        out_md_path.write_text(_format_md(rows), encoding="utf-8")
        log.info("Wrote ablation Markdown to %s", out_md_path)

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant",
        action="append",
        type=_parse_variant,
        default=None,
        help="repeatable; format LABEL=PATH",
    )
    p.add_argument("--out", type=Path, default=Path("results/ablation.json"))
    p.add_argument("--out-md", type=Path, default=Path("results/ablation.md"))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    variants = args.variant if args.variant else list(DEFAULT_VARIANTS)
    summary = run(variants=variants, out_path=args.out, out_md_path=args.out_md)
    print(
        f"variants={len(summary['variants'])} "
        f"out={args.out} out_md={args.out_md}"
    )


if __name__ == "__main__":
    main()
