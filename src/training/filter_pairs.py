"""Quality-filter generated training pairs by query↔chunk cosine.

Reads the raw pairs that `src.training.generate_pairs` produced
(`data/training/pairs_raw.jsonl`), encodes each query with the same
BGE model used to build the index, looks up the chunk vector from
the pre-computed `data/processed/embeddings.npy`, and keeps only
pairs whose cosine similarity falls inside `[--min-cos, --max-cos]`
(default 0.3 .. 0.95).

Why these bounds (per Task 2.07 spec):
- Below `min_cos` (default 0.30): the generator drifted off-topic.
  Keeping these would teach the embedding model that semantically
  unrelated text should be considered a match — pure noise.
- Above `max_cos` (default 0.95): the generated query is essentially
  a paraphrase or substring of the chunk text. The model already
  trivially scores these high; there's no learning signal in
  "this very-similar query → this chunk" for contrastive training.

We use the existing `embeddings.npy` (41,959 × 768, BGE float32) so
no chunk text needs to be re-encoded — only the queries. Queries are
encoded with the same `Represent this sentence for searching relevant
passages: ` prefix the retriever uses, so the cosine numbers here
match production retrieval scores.

Output is the canonical `data/training/pairs.jsonl` consumed downstream
by hard-negative mining (Task 2.08) and embedding fine-tuning (Task 2.10).
Each kept pair carries a `cosine` field so downstream stages can do a
finer split if useful.

    python -m src.training.filter_pairs \
        --in data/training/pairs_raw.jsonl \
        --out data/training/pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

DEFAULT_BGE_MODEL = "BAAI/bge-base-en-v1.5"
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_MIN_COS = 0.30
DEFAULT_MAX_COS = 0.95
DEFAULT_BATCH = 64


def _load_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_chunk_id_index(path: Path) -> dict[str, int]:
    idx: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            cid = line.strip()
            if cid:
                idx[cid] = i
    return idx


def _encode_queries(model, texts: list[str], batch: int) -> np.ndarray:
    return model.encode(
        [_QUERY_PREFIX + t for t in texts],
        batch_size=batch,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def run(
    in_path: Path,
    out_path: Path,
    *,
    chunks_dir: Path,
    bge_model: str,
    min_cos: float,
    max_cos: float,
    batch: int,
) -> dict:
    pairs = _load_pairs(in_path)
    log.info("Loaded %d raw pairs from %s", len(pairs), in_path)
    if not pairs:
        log.warning("No pairs to filter; writing empty output.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        return {"in": 0, "kept": 0, "drop_low": 0, "drop_high": 0, "drop_missing": 0}

    chunk_ids_path = chunks_dir / "chunk_ids.txt"
    embeddings_path = chunks_dir / "embeddings.npy"
    log.info("Loading chunk_id index from %s", chunk_ids_path)
    cid_to_idx = _load_chunk_id_index(chunk_ids_path)
    log.info("Loading chunk embeddings from %s (mmap)", embeddings_path)
    chunk_vecs = np.load(embeddings_path, mmap_mode="r")

    from sentence_transformers import SentenceTransformer

    log.info("Loading query encoder %s", bge_model)
    model = SentenceTransformer(bge_model)

    drop_missing: list[dict] = []
    encodable: list[dict] = []
    for p in pairs:
        if p.get("chunk_id") in cid_to_idx:
            encodable.append(p)
        else:
            drop_missing.append(p)
    if drop_missing:
        log.warning("Dropping %d pairs with chunk_id not in corpus", len(drop_missing))

    queries = [p["query"] for p in encodable]
    log.info("Encoding %d queries (batch=%d)", len(queries), batch)
    q_vecs = _encode_queries(model, queries, batch=batch)

    kept: list[dict] = []
    drop_low = 0
    drop_high = 0
    persona_kept: Counter[str] = Counter()
    for p, q_vec in zip(encodable, q_vecs, strict=True):
        c_vec = chunk_vecs[cid_to_idx[p["chunk_id"]]]
        cosine = float(np.dot(q_vec, c_vec))
        if cosine < min_cos:
            drop_low += 1
            continue
        if cosine > max_cos:
            drop_high += 1
            continue
        kept.append({**p, "cosine": cosine})
        persona_kept[p.get("persona", "?")] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "in": len(pairs),
        "kept": len(kept),
        "drop_low": drop_low,
        "drop_high": drop_high,
        "drop_missing": len(drop_missing),
        "min_cos": min_cos,
        "max_cos": max_cos,
        "persona_kept": dict(persona_kept),
    }
    log.info(
        "Wrote %d kept pairs to %s (drop_low=%d drop_high=%d drop_missing=%d)",
        len(kept), out_path, drop_low, drop_high, len(drop_missing),
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in", dest="in_path", type=Path, default=Path("data/training/pairs_raw.jsonl")
    )
    p.add_argument("--out", type=Path, default=Path("data/training/pairs.jsonl"))
    p.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("data/processed"),
        help="dir containing embeddings.npy and chunk_ids.txt",
    )
    p.add_argument("--bge-model", default=DEFAULT_BGE_MODEL)
    p.add_argument("--min-cos", type=float, default=DEFAULT_MIN_COS)
    p.add_argument("--max-cos", type=float, default=DEFAULT_MAX_COS)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    summary = run(
        in_path=args.in_path,
        out_path=args.out,
        chunks_dir=args.chunks_dir,
        bge_model=args.bge_model,
        min_cos=args.min_cos,
        max_cos=args.max_cos,
        batch=args.batch,
    )
    print(
        f"in={summary['in']} kept={summary['kept']} "
        f"drop_low={summary['drop_low']} drop_high={summary['drop_high']} "
        f"drop_missing={summary['drop_missing']} out={args.out}"
    )


if __name__ == "__main__":
    main()
