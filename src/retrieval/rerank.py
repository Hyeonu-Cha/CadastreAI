"""Cross-encoder reranker over a base retriever's shortlist.

The baseline error analysis (see `results/error_analysis.md`) found
that 60% of failures were near-miss within/adjacent to the target
document — the retriever landed on the right topic cluster but picked
the wrong chunk. A cross-encoder reranker scores each `(query, chunk)`
pair directly over the full chunk text, so it can disambiguate at the
paragraph level where a bi-encoder's pooled vector cannot.

Pipeline per Task 2.04 spec:

    base.retrieve(q, k=30)  →  reranker.rerank(q, shortlist)  →  top-5

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22M params, ~90MB).
Standard MS MARCO–trained cross-encoder; English-only, which is fine
for an Australian housing corpus. We tried `BAAI/bge-reranker-base`
(278M) and `BAAI/bge-reranker-v2-m3` (568M) but their safetensors
mmap fails on this 4GB-GPU + small-pagefile Windows host (OSError
1455 / segfault during CrossEncoder init when BGE base is already
on the GPU). MiniLM-L6 is the standard fallback: smaller, faster,
proven on retrieval reranking. Override with `--model` on a beefier
host. Keep one model instance per process — first-call latency
≈ 30 × single-pair inference.

    python -m src.retrieval.rerank --query "negative gearing rental supply" -k 5
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_SHORTLIST = 30
DEFAULT_TOP_K = 5


@dataclass
class Reranker:
    model_name: str = DEFAULT_MODEL
    device: str | None = None

    def __post_init__(self) -> None:
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("Loading reranker %s (device=%s)", self.model_name, self.device or "auto")
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[tuple[dict, float]]:
        if not chunks:
            return []
        model = self._ensure_model()
        pairs = [[query, (c.get("text") or "")] for c in chunks]
        scores = model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda x: float(x[1]), reverse=True)
        return [(c, float(s)) for c, s in ranked[:top_k]]


@dataclass
class RerankedRetriever:
    """Wraps a base retriever with a cross-encoder reranker.

    Any base that exposes `retrieve(query, k) -> list[(chunk, score)]`
    works — dense, BM25, or hybrid. The base returns `shortlist` chunks;
    the reranker resorts them and returns the top-k.
    """

    base: object
    reranker: Reranker | None = None
    shortlist: int = DEFAULT_SHORTLIST

    def __post_init__(self) -> None:
        if self.reranker is None:
            self.reranker = Reranker()

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[tuple[dict, float]]:
        base_hits = self.base.retrieve(query, k=self.shortlist)
        chunks = [payload for payload, _ in base_hits]
        return self.reranker.rerank(query, chunks, top_k=k)


def _build_base(kind: str, bm25_index_path: Path, shortlist: int, k_rrf: int) -> object:
    """Small helper so main() doesn't need to import all three at module scope."""
    if kind == "dense":
        from src.retrieval.retriever import Retriever

        return Retriever()
    if kind == "bm25":
        from src.index.bm25 import BM25Index

        return BM25Index.load(bm25_index_path)
    if kind == "hybrid":
        from src.index.bm25 import BM25Index
        from src.index.hybrid import HybridRetriever

        return HybridRetriever(
            sparse=BM25Index.load(bm25_index_path),
            shortlist=shortlist,
            k_rrf=k_rrf,
        )
    raise ValueError(f"unknown base retriever: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--shortlist", type=int, default=DEFAULT_SHORTLIST)
    parser.add_argument(
        "--base",
        choices=["dense", "bm25", "hybrid"],
        default="hybrid",
        help="base retriever feeding the reranker",
    )
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=Path(os.environ.get("BM25_INDEX", "data/processed/bm25.pkl")),
    )
    parser.add_argument("--k-rrf", type=int, default=60)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="cross-encoder model id")
    parser.add_argument(
        "--device",
        default=None,
        help="torch device for the reranker (e.g. 'cpu', 'cuda'); default auto",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    reranker = Reranker(model_name=args.model, device=args.device)
    base = _build_base(args.base, args.bm25_index, args.shortlist, args.k_rrf)
    retriever = RerankedRetriever(base=base, reranker=reranker, shortlist=args.shortlist)
    hits = retriever.retrieve(args.query, k=args.k)
    for i, (c, score) in enumerate(hits, start=1):
        title = (c.get("title") or "")[:70]
        section = (c.get("section_heading") or "")[:70]
        print(f"[{i:2d}] rerank={score:.4f}  {c.get('publisher')} | {title}")
        if section:
            print(f"     {section}")


if __name__ == "__main__":
    main()
