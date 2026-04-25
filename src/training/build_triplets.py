"""Build (anchor, positive, negatives) training triplets from mined candidates.

Reads `data/training/negative_candidates.jsonl` (output of Task 2.08),
keeps the top-N hardest negatives per pair (highest BM25 score), looks
up the chunk text for the positive and each negative from
`data/processed/chunks.jsonl`, and writes one triplet record per pair:

    {"anchor": "<query>",
     "positive": "<positive chunk text>",
     "negatives": ["<neg 1 text>", ..., "<neg N text>"],
     "anchor_chunk_id": "...",
     "negative_chunk_ids": [...]}

Why top-N by BM25 score (per Task 2.09 spec):
 - The mining stage already filtered same-doc and the positive itself,
   so what's left is "lexically similar to the query but in a different
   document". Higher BM25 = closer lexical match = harder negative for
   the dense model to push apart.
 - 5 negatives per anchor matches the MultipleNegativesRankingLoss
   training setup in Task 2.10 — extra negatives beyond the in-batch
   ones, picked for being deliberately hard.

Output is line-delimited JSON, one triplet per line. Pairs whose mining
yielded fewer than `--min-negatives` candidates (default 1) are dropped
with a warning so we don't ship anchor-positive-only rows that the
contrastive loss can't use.

    python -m src.training.build_triplets \
        --in data/training/negative_candidates.jsonl \
        --chunks data/processed/chunks.jsonl \
        --out data/training/triplets.jsonl \
        -n 5
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

DEFAULT_NEGATIVES = 5
DEFAULT_MIN_NEGATIVES = 1


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_text_lookup(chunks_path: Path) -> dict[str, str]:
    """Stream chunks.jsonl once and build chunk_id -> text map."""
    out: dict[str, str] = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("chunk_id")
            text = rec.get("text") or ""
            if cid:
                out[cid] = text
    return out


def _select_top_negatives(candidates: list[dict], n: int) -> list[dict]:
    """Pick the N highest-BM25 negatives. Mining already produced them
    in score order, but we sort defensively in case the upstream order
    changes."""
    ranked = sorted(
        candidates,
        key=lambda c: float(c.get("bm25_score", 0.0)),
        reverse=True,
    )
    return ranked[:n]


def run(
    in_path: Path,
    chunks_path: Path,
    out_path: Path,
    *,
    n_negatives: int,
    min_negatives: int,
) -> dict:
    log.info("Loading mined pairs from %s", in_path)
    pairs = _load_jsonl(in_path)
    log.info("Loaded %d pairs with mined candidates", len(pairs))
    if not pairs:
        log.warning("No input pairs — writing empty output.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        return {"in": 0, "kept": 0, "drop_too_few": 0, "drop_missing_text": 0}

    log.info("Loading chunk text lookup from %s", chunks_path)
    text_lookup = _build_text_lookup(chunks_path)
    log.info("Loaded %d chunk texts", len(text_lookup))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    drop_too_few = 0
    drop_missing_text = 0
    neg_counts: list[int] = []

    with out_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            anchor = pair.get("query") or ""
            anchor_cid = pair.get("chunk_id") or ""
            candidates = pair.get("negative_candidates") or []
            picked = _select_top_negatives(candidates, n_negatives)

            positive_text = text_lookup.get(anchor_cid)
            if not positive_text:
                drop_missing_text += 1
                log.warning("positive %s missing chunk text — skipping", anchor_cid)
                continue

            neg_texts: list[str] = []
            neg_ids: list[str] = []
            for c in picked:
                cid = c.get("chunk_id")
                txt = text_lookup.get(cid)
                if cid and txt:
                    neg_ids.append(cid)
                    neg_texts.append(txt)

            if len(neg_texts) < min_negatives:
                drop_too_few += 1
                continue

            rec = {
                "anchor": anchor,
                "positive": positive_text,
                "negatives": neg_texts,
                "anchor_chunk_id": anchor_cid,
                "negative_chunk_ids": neg_ids,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            neg_counts.append(len(neg_texts))

    summary = {
        "in": len(pairs),
        "kept": kept,
        "drop_too_few": drop_too_few,
        "drop_missing_text": drop_missing_text,
        "mean_negatives": (sum(neg_counts) / kept) if kept else 0.0,
    }
    log.info(
        "Wrote %d triplets to %s (drop_too_few=%d drop_missing=%d mean_negs=%.2f)",
        kept, out_path, drop_too_few, drop_missing_text, summary["mean_negatives"],
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in", dest="in_path", type=Path,
        default=Path("data/training/negative_candidates.jsonl"),
    )
    p.add_argument(
        "--chunks", type=Path, default=Path("data/processed/chunks.jsonl")
    )
    p.add_argument(
        "--out", type=Path, default=Path("data/training/triplets.jsonl")
    )
    p.add_argument(
        "-n", "--n-negatives", type=int, default=DEFAULT_NEGATIVES,
        help="hardest BM25 negatives kept per anchor",
    )
    p.add_argument(
        "--min-negatives", type=int, default=DEFAULT_MIN_NEGATIVES,
        help="drop rows with fewer than this many negatives after lookup",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    summary = run(
        in_path=args.in_path,
        chunks_path=args.chunks,
        out_path=args.out,
        n_negatives=args.n_negatives,
        min_negatives=args.min_negatives,
    )
    print(
        f"in={summary['in']} kept={summary['kept']} "
        f"drop_too_few={summary['drop_too_few']} "
        f"drop_missing={summary['drop_missing_text']} "
        f"mean_negs={summary['mean_negatives']:.2f} out={args.out}"
    )


if __name__ == "__main__":
    main()
