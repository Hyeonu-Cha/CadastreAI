"""Plot Recall@K curves for the Week 2 ablation variants.

Reads per-variant retrieval-eval JSONs (the same files
`src.eval.ablation` consumes) and renders a single PNG with one line
per variant — base BGE → +hybrid → +hybrid+rerank → ft+hybrid+rerank —
showing macro-averaged Recall@K for K = 1..top_k.

Why per-query-aggregated rather than just the precomputed `recall@5` /
`recall@10`: each `per_query` row carries `gold_ranks`, the 1-indexed
position of every gold chunk in the retrieved list (or null if missed).
That gives us recall at *any* K up to `top_k` without re-running
retrieval — perfect for a smooth curve.

    python -m src.eval.plot_recall_curves \
        --variant "base BGE (dense)=results/baseline.json" \
        --variant "BM25=results/bm25.json" \
        --variant "BGE + BM25 (hybrid)=results/hybrid.json" \
        --variant "hybrid + cross-encoder=results/reranked.json" \
        --variant "fine-tuned BGE (dense)=results/finetuned.json" \
        --out docs/figures/recall_curves.png

Variants whose JSON doesn't exist are skipped with a warning rather
than crashing — useful while the fine-tune is still in flight.
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
    ("base BGE (dense)", Path("results/baseline.json")),
    ("BM25", Path("results/bm25.json")),
    ("BGE + BM25 (hybrid)", Path("results/hybrid.json")),
    ("hybrid + cross-encoder", Path("results/reranked.json")),
    ("fine-tuned BGE (dense)", Path("results/finetuned.json")),
)


def _parse_variant(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"variant must be 'LABEL=PATH', got {spec!r}"
        )
    label, path = spec.split("=", 1)
    return label.strip(), Path(path.strip())


def _recall_curve(per_query: list[dict], top_k: int) -> list[float]:
    """Macro-averaged Recall@K for K = 1..top_k.

    Each query contributes (# gold ranks ≤ K) / (# gold chunks). Queries
    with no gold chunks are skipped. We average over the remaining
    queries — same convention `retrieval_eval` uses for the scalar
    `recall@5` / `recall@10` it emits, so curve points at K=5 / K=10
    line up with the tabular summary.
    """
    curve: list[float] = []
    for k in range(1, top_k + 1):
        scores: list[float] = []
        for row in per_query:
            ranks = row.get("gold_ranks") or {}
            if not ranks:
                continue
            hits = sum(1 for r in ranks.values() if r is not None and r <= k)
            scores.append(hits / len(ranks))
        curve.append(sum(scores) / len(scores) if scores else 0.0)
    return curve


def _load_variant(label: str, path: Path) -> dict | None:
    if not path.exists():
        log.warning("variant '%s' missing %s — skipping", label, path)
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    per_query = doc.get("per_query") or []
    top_k = int(doc.get("top_k") or 10)
    if not per_query:
        log.warning("variant '%s' has no per_query rows — skipping", label)
        return None
    return {
        "label": label,
        "source": str(path),
        "top_k": top_k,
        "n": len(per_query),
        "curve": _recall_curve(per_query, top_k),
    }


def _plot(variants: list[dict], out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    for v in variants:
        ks = list(range(1, v["top_k"] + 1))
        ax.plot(ks, v["curve"], marker="o", linewidth=2.0, label=v["label"])
    ax.set_xlabel("K (top results)")
    ax.set_ylabel("Recall@K")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    max_k = max(v["top_k"] for v in variants)
    ax.set_xticks(list(range(1, max_k + 1)))
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def run(
    variants: list[tuple[str, Path]],
    out_path: Path,
    title: str,
    out_json: Path | None,
) -> dict:
    rows: list[dict] = []
    for label, path in variants:
        row = _load_variant(label, path)
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("no variants loaded — check paths")

    _plot(rows, out_path, title)
    log.info("Wrote recall curves to %s", out_path)

    summary = {
        "title": title,
        "out": str(out_path),
        "variants": [
            {"label": v["label"], "source": v["source"], "n": v["n"],
             "top_k": v["top_k"], "curve": v["curve"]}
            for v in rows
        ],
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("Wrote curve data to %s", out_json)
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
    p.add_argument(
        "--out", type=Path, default=Path("docs/figures/recall_curves.png")
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="optional JSON dump of the curve points",
    )
    p.add_argument(
        "--title", default="Recall@K — Week 2 retrieval ablation"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    variants = args.variant if args.variant else list(DEFAULT_VARIANTS)
    summary = run(
        variants=variants,
        out_path=args.out,
        title=args.title,
        out_json=args.out_json,
    )
    print(
        f"variants={len(summary['variants'])} out={args.out}"
    )


if __name__ == "__main__":
    main()
