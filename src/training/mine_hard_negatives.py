"""Mine hard-negative candidates from BM25 for each training pair.

For each (query, positive_chunk) pair in `data/training/pairs.jsonl`,
asks the BM25 index for the top-K lexically-similar chunks. The
positive itself and any chunk that shares the positive's
`(title, section_heading)` are filtered out — these would either
be trivially the right answer or near-duplicate paragraphs of it,
neither of which gives a useful contrastive signal.

Why BM25 and not dense:
 - We're trying to teach the *dense* model what's actually a negative
   despite looking similar lexically. BM25's hits are exactly the
   "looks similar but isn't" set we want — high lexical overlap with
   the query but (filtered to) different sections of different docs.
   Hard negatives mined from dense itself collapse the contrastive
   loss when the model is already good at those.

Filter rules:
 - drop the positive chunk_id itself
 - drop any chunk that comes from the same source document as the
   positive (matched by chunk_id prefix before `__NNNN`). Spec says
   "same section", but same-doc-different-section chunks share enough
   topical context that they're not useful contrastive negatives —
   easier and safer to drop the whole doc.

Each input pair gets an extra `negative_candidates` list with
chunk_id + BM25 score for downstream Task 2.09 (top-5 hardest →
triplets.jsonl).

    python -m src.training.mine_hard_negatives \
        --in data/training/pairs.jsonl \
        --out data/training/negative_candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

DEFAULT_TOP_K = 20
DEFAULT_OVERFETCH = 50

_DOC_PREFIX_RE = re.compile(r"^(.*)__\d+$")


def _doc_prefix(chunk_id: str) -> str:
    """Document prefix — the chunk_id with the trailing `__NNNN` stripped."""
    m = _DOC_PREFIX_RE.match(chunk_id or "")
    return m.group(1) if m else chunk_id


def _load_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_chunk_lookup(chunks: list[dict]) -> dict[str, dict]:
    return {c["chunk_id"]: c for c in chunks}


def _is_same_section(positive: dict, candidate: dict) -> bool:
    """True if the candidate comes from the same source document as the
    positive — treat as a near-duplicate rather than a contrastive negative.

    The spec says "filter out same-section", but in practice same-doc-
    different-section chunks share enough vocabulary and topical context
    with the positive that they leak signal into training. Easier and
    safer to drop the whole document.
    """
    return _doc_prefix(positive.get("chunk_id", "")) == _doc_prefix(
        candidate.get("chunk_id", "")
    )


def mine_for_pair(
    pair: dict,
    bm25,
    chunk_lookup: dict[str, dict],
    *,
    top_k: int,
    overfetch: int,
) -> list[dict]:
    """Return up to `top_k` filtered BM25 hits as negative candidates."""
    positive_id = pair["chunk_id"]
    positive = chunk_lookup.get(positive_id)
    if positive is None:
        log.warning("positive chunk_id %s missing from corpus — skipping", positive_id)
        return []
    hits = bm25.retrieve(pair["query"], k=overfetch)
    negatives: list[dict] = []
    for cand, score in hits:
        cand_id = cand.get("chunk_id")
        if not cand_id or cand_id == positive_id:
            continue
        if _is_same_section(positive, cand):
            continue
        negatives.append({"chunk_id": cand_id, "bm25_score": float(score)})
        if len(negatives) >= top_k:
            break
    return negatives


def run(
    in_path: Path,
    out_path: Path,
    bm25_index_path: Path,
    *,
    top_k: int,
    overfetch: int,
) -> dict:
    from src.index.bm25 import BM25Index

    pairs = _load_pairs(in_path)
    log.info("Loaded %d input pairs from %s", len(pairs), in_path)
    if not pairs:
        log.warning("No input pairs — writing empty output.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        return {"in": 0, "with_negatives": 0, "no_negatives": 0, "mean_negatives": 0.0}

    log.info("Loading BM25 index from %s", bm25_index_path)
    bm25 = BM25Index.load(bm25_index_path)
    chunk_lookup = _build_chunk_lookup(bm25.chunks)
    log.info(
        "BM25 ready (%d chunks); mining top-%d candidates per pair (overfetch=%d)",
        len(chunk_lookup), top_k, overfetch,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with_neg = 0
    no_neg = 0
    total_negs = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, pair in enumerate(pairs, start=1):
            negs = mine_for_pair(
                pair, bm25, chunk_lookup, top_k=top_k, overfetch=overfetch
            )
            if negs:
                with_neg += 1
                total_negs += len(negs)
            else:
                no_neg += 1
            rec = {**pair, "negative_candidates": negs}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 200 == 0 or i == len(pairs):
                log.info("  %d/%d pairs mined", i, len(pairs))

    summary = {
        "in": len(pairs),
        "with_negatives": with_neg,
        "no_negatives": no_neg,
        "mean_negatives": total_negs / max(with_neg, 1),
    }
    log.info(
        "Wrote %s | with_negs=%d no_negs=%d mean_negs=%.1f",
        out_path, with_neg, no_neg, summary["mean_negatives"],
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in", dest="in_path", type=Path, default=Path("data/training/pairs.jsonl")
    )
    p.add_argument(
        "--out", type=Path, default=Path("data/training/negative_candidates.jsonl")
    )
    p.add_argument("--bm25-index", type=Path, default=Path("data/processed/bm25.pkl"))
    p.add_argument(
        "-k", "--top-k", type=int, default=DEFAULT_TOP_K,
        help="negative candidates kept per pair after filtering",
    )
    p.add_argument(
        "--overfetch", type=int, default=DEFAULT_OVERFETCH,
        help="BM25 hits to consider before filtering (must exceed top_k)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    summary = run(
        in_path=args.in_path,
        out_path=args.out,
        bm25_index_path=args.bm25_index,
        top_k=args.top_k,
        overfetch=args.overfetch,
    )
    print(
        f"in={summary['in']} with_negs={summary['with_negatives']} "
        f"no_negs={summary['no_negatives']} "
        f"mean_negs={summary['mean_negatives']:.1f} out={args.out}"
    )


if __name__ == "__main__":
    main()
