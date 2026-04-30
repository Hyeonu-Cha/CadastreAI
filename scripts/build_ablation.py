"""Build the Task 2.16 ablation comparison from existing eval JSON files.

Reads `results/{baseline,bm25,hybrid,reranked,finetuned}.json` and emits:

  - `results/ablation.json` — machine-readable aggregate (one row per
    variant with overall metrics + latency stats + retriever descriptor).
  - `results/ablation.md` — human-readable Markdown table + a brief
    commentary on the surprises (BM25 leading, rerank regressing,
    FT regressing).

The four variants Task 2.16 originally listed (`base BGE / +hybrid /
+hybrid+rerank / ft+hybrid+rerank`) are partly unmeasured: we don't
have `ft+hybrid+rerank` because that pipeline requires Qdrant for the
fine-tuned collection, which is unavailable while Docker is broken.
The script flags that row as `not run` rather than silently dropping it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANTS = [
    # (key in output, file path or None, label, notes)
    ("baseline",  Path("results/baseline.json"),  "base BGE (dense)",        ""),
    ("bm25",      Path("results/bm25.json"),      "BM25 only",               ""),
    ("hybrid",    Path("results/hybrid.json"),    "base BGE + BM25 (hybrid)", ""),
    ("reranked",  Path("results/reranked.json"),  "hybrid + cross-encoder",  ""),
    ("finetuned", Path("results/finetuned.json"), "fine-tuned BGE (dense)",  "offline matmul; Qdrant unavailable"),
    ("ft_hybrid_rerank", None, "ft + hybrid + cross-encoder", "not run — needs Qdrant collection cadastre_chunks_ft"),
]


def _safe_load(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v: float | None, fmt: str = "{:.3f}") -> str:
    return fmt.format(v) if v is not None else "—"


def build() -> tuple[dict, str]:
    rows: list[dict] = []
    for key, path, label, notes in VARIANTS:
        data = _safe_load(path)
        if data is None:
            rows.append({
                "key": key, "label": label, "notes": notes,
                "n": None, "recall@5": None, "recall@10": None,
                "mrr@10": None, "ndcg@10": None,
                "median_ms": None, "p95_ms": None,
                "retriever": None,
            })
            continue
        o = data["overall"]
        lat = data.get("latency") or {}
        rows.append({
            "key": key,
            "label": label,
            "notes": notes,
            "n": o["n"],
            "recall@5": o["recall@5"],
            "recall@10": o["recall@10"],
            "mrr@10": o["mrr@10"],
            "ndcg@10": o["ndcg@10"],
            "median_ms": lat.get("median_ms"),
            "p95_ms": lat.get("p95_ms"),
            "retriever": data.get("retriever"),
        })

    md_lines: list[str] = []
    md_lines.append("# Task 2.16 — Retriever ablation (queries_all.jsonl, n=100)")
    md_lines.append("")
    md_lines.append(
        "All variants scored against the same 100-query gold set "
        "(`data/eval/queries_all.jsonl`). Bold indicates the best value per metric "
        "across measured rows."
    )
    md_lines.append("")

    # Find best per metric (across non-null rows).
    metrics = ("recall@5", "recall@10", "mrr@10", "ndcg@10")
    best = {m: max((r[m] for r in rows if r[m] is not None), default=None) for m in metrics}

    md_lines.append("| variant | n | R@5 | R@10 | MRR@10 | nDCG@10 | median lat | notes |")
    md_lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in rows:
        cells = []
        for m in metrics:
            v = r[m]
            if v is None:
                cells.append("—")
            elif best[m] is not None and abs(v - best[m]) < 1e-9:
                cells.append(f"**{v:.3f}**")
            else:
                cells.append(f"{v:.3f}")
        lat = f"{r['median_ms']:.0f} ms" if r["median_ms"] is not None else "—"
        n = str(r["n"]) if r["n"] is not None else "—"
        md_lines.append(
            f"| {r['label']} | {n} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {lat} | {r['notes']} |"
        )
    md_lines.append("")

    md_lines.append("## What jumps out")
    md_lines.append("")
    md_lines.append(
        "- **BM25 alone is the strongest measured variant** — R@10=0.91, MRR@10=0.73 — "
        "well ahead of every dense or hybrid configuration. The eval queries lean "
        "heavily on named entities (\"First Home Owner Grant\", \"Home Guarantee Scheme\", "
        "specific publishers) and BM25's lexical bias matches that distribution."
    )
    md_lines.append(
        "- **Hybrid hurts BM25.** Fusing dense BGE into BM25 via RRF *drops* R@10 from "
        "0.91 → 0.75. The dense retriever is pulling well-ranked BM25 hits down. "
        "Worth revisiting the RRF shortlist + k_rrf parameters, or weighting BM25 "
        "higher in the fusion."
    )
    md_lines.append(
        "- **Cross-encoder rerank also regresses** on top of hybrid (R@10 0.75 → 0.67, "
        "median latency 0.4s → 2.9s). On this eval set, the reranker is reordering "
        "the BM25-favoured passages downward. Revisit the reranker model / shortlist."
    )
    md_lines.append(
        "- **Fine-tuned BGE collapses** (R@10 0.45 → 0.21) — see Task 2.14/2.15 reports "
        "for the publisher-collapse failure mode."
    )
    md_lines.append("")
    md_lines.append("## Implications for Task 2.17 / 2.18")
    md_lines.append("")
    md_lines.append(
        "- The Recall@K curve plot (Task 2.17) should put BM25 on top so the curves "
        "tell the right story; the previous mental model of \"hybrid > BM25 > dense\" "
        "doesn't hold on this eval set."
    )
    md_lines.append(
        "- The Week 2 blog section (Task 2.18) should be honest about the fine-tune "
        "regression and the BM25/hybrid/rerank inversion. The interesting story is "
        "*why* BM25 dominates on this corpus, not a clean monotonic ablation."
    )
    md_lines.append(
        "- The `ft + hybrid + cross-encoder` cell is left unmeasured; once Docker is "
        "back up we can run `reembed_finetuned --skip-embed` to upsert the existing "
        "`.npy` and then re-run `retrieval_eval --retriever hybrid --rerank "
        "--dense-collection cadastre_chunks_ft`. Realistic expectation given the "
        "publisher collapse upstream: it won't rescue the FT model."
    )
    return {"variants": rows}, "\n".join(md_lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-json", type=Path, default=Path("results/ablation.json"))
    p.add_argument("--out-md", type=Path, default=Path("results/ablation.md"))
    args = p.parse_args()

    data, md = build()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
