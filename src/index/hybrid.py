"""Hybrid dense + sparse retrieval via Reciprocal Rank Fusion (RRF).

Combines BGE dense top-N with BM25 sparse top-N into a single ranked
list using RRF. For each candidate chunk:

    score(d) = sum over each source s of  1 / (k_RRF + rank_s(d))

where rank starts at 1 and sources that didn't retrieve the chunk
contribute 0 for that source. The standard constant is k_RRF = 60
(Cormack, Clarke & Buettcher, 2009). RRF is rank-only — the underlying
scores have different scales (cosine vs BM25) so rank fusion avoids
the calibration problem entirely.

Returns `(chunk_dict, rrf_score)` like the single-source retrievers,
but also exposes `retrieve_with_ranks` for eval/error-analysis where
the per-source ranks matter.

    python -m src.index.hybrid --query "negative gearing rental supply" -k 5
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from src.index.bm25 import DEFAULT_INDEX as DEFAULT_BM25_INDEX
from src.index.bm25 import BM25Index
from src.retrieval.retriever import Retriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

DEFAULT_SHORTLIST = 50
DEFAULT_K_RRF = 60


@dataclass
class HybridRetriever:
    """Combines a dense and a sparse source with RRF.

    Either source can be swapped — tests can pass stubs; future work can
    plug a reranker into one slot. The dense source must follow the
    `retrieve(query, k) -> list[(dict, float)]` contract; likewise sparse.
    """

    dense: Retriever | None = None
    sparse: BM25Index | None = None
    shortlist: int = DEFAULT_SHORTLIST
    k_rrf: int = DEFAULT_K_RRF

    def __post_init__(self) -> None:
        if self.dense is None:
            self.dense = Retriever()
        if self.sparse is None:
            self.sparse = BM25Index.load(DEFAULT_BM25_INDEX)

    def retrieve_with_ranks(
        self, query: str, k: int = 10
    ) -> list[tuple[dict, float, int | None, int | None]]:
        """Return `(chunk, rrf_score, dense_rank, sparse_rank)` tuples.

        Ranks are 1-indexed within the shortlist, or None if the source
        didn't retrieve the chunk within the shortlist. The dict is taken
        from whichever source retrieved the chunk first (dense preferred).
        """
        dense_hits = self.dense.retrieve(query, k=self.shortlist)
        sparse_hits = self.sparse.retrieve(query, k=self.shortlist)

        # Index by chunk_id. Dense payload wins ties — it carries the live
        # Qdrant fields; sparse payload is from the pickled corpus.
        chunks_by_id: dict[str, dict] = {}
        dense_rank: dict[str, int] = {}
        sparse_rank: dict[str, int] = {}
        for i, (c, _) in enumerate(dense_hits, start=1):
            cid = c.get("chunk_id")
            if cid:
                chunks_by_id.setdefault(cid, c)
                dense_rank[cid] = i
        for i, (c, _) in enumerate(sparse_hits, start=1):
            cid = c.get("chunk_id")
            if cid:
                chunks_by_id.setdefault(cid, c)
                sparse_rank[cid] = i

        fused: list[tuple[str, float]] = []
        for cid in chunks_by_id:
            dr = dense_rank.get(cid)
            sr = sparse_rank.get(cid)
            score = 0.0
            if dr is not None:
                score += 1.0 / (self.k_rrf + dr)
            if sr is not None:
                score += 1.0 / (self.k_rrf + sr)
            fused.append((cid, score))
        fused.sort(key=lambda x: x[1], reverse=True)

        out: list[tuple[dict, float, int | None, int | None]] = []
        for cid, score in fused[:k]:
            out.append((chunks_by_id[cid], score, dense_rank.get(cid), sparse_rank.get(cid)))
        return out

    def retrieve(self, query: str, k: int = 10) -> list[tuple[dict, float]]:
        """Match the dense/sparse Retriever interface — drops the rank detail."""
        return [(c, s) for c, s, _, _ in self.retrieve_with_ranks(query, k=k)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--shortlist", type=int, default=DEFAULT_SHORTLIST)
    parser.add_argument("--k-rrf", type=int, default=DEFAULT_K_RRF)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_BM25_INDEX)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    hybrid = HybridRetriever(
        sparse=BM25Index.load(args.bm25_index),
        shortlist=args.shortlist,
        k_rrf=args.k_rrf,
    )
    hits = hybrid.retrieve_with_ranks(args.query, k=args.k)
    for i, (c, score, dr, sr) in enumerate(hits, start=1):
        title = (c.get("title") or "")[:70]
        section = (c.get("section_heading") or "")[:70]
        dr_s = str(dr) if dr is not None else "-"
        sr_s = str(sr) if sr is not None else "-"
        print(
            f"[{i:2d}] rrf={score:.4f} dense={dr_s} sparse={sr_s}  "
            f"{c.get('publisher')} | {title}"
        )
        if section:
            print(f"     {section}")


if __name__ == "__main__":
    main()
