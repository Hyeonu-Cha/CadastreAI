"""Pull BM25-ranked candidate chunks for every eval query to support annotation.

For each query in the three persona files, scans `chunks.jsonl` and
ranks chunks by a stdlib BM25 score (no model load, no external deps).
Writes one compact record per query to `data/eval/candidates.jsonl`:

    {
      "query": "...",
      "persona": "homebuyer",
      "candidates": [
        {"chunk_id": "...", "score": 12.34, "publisher": "...",
         "title": "...", "section": "...", "snippet": "...160 chars..."}
      ]
    }

Why BM25 rather than the BGE retriever we just built? Two reasons:
  1. Methodological — annotating gold chunks using the same model we
     evaluate produces circular recall@k. BM25 as annotation source
     gives the BGE retriever a fair chance to miss lexically-aware gold.
  2. Operational — loading BGE on this machine currently trips a
     Windows pagefile limit (OS error 1455). BM25 is pure stdlib so
     it just runs.

Task 1.27 uses this as the shortlist for manually picking 2-5 gold
chunks per query; final file is `data/eval/queries.jsonl`.

    python -m scripts.gather_eval_candidates --k 10
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

PERSONA_FILES = [
    ("homebuyer", Path("data/eval/queries_homebuyer.jsonl")),
    ("investor", Path("data/eval/queries_investor.jsonl")),
    ("researcher", Path("data/eval/queries_researcher.jsonl")),
]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have how if in into is it
    its my of on or that the this to was what when where which who why will
    with you your we us our over per up down do does did not no nor than
    then there these those they them their i me he she his her one two
    three between also among many much more most some any all each other
    such same very can could should would may might must about across
    after before during because only just like unlike within without
    across towards toward upon under above below off out onto ought
    am being been having
    home housing house housekeeping property properties
    australian australia au
    """.split()
)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOPWORDS]


def _load_chunks(chunks_path: Path) -> tuple[list[dict], list[list[str]]]:
    chunks: list[dict] = []
    tokens: list[list[str]] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunks.append(rec)
            tokens.append(_tokenize(rec.get("text", "")))
    return chunks, tokens


class BM25:
    """Minimal BM25Okapi implementation (stdlib only)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter] = [Counter(d) for d in docs]
        self.doc_len = [len(d) for d in docs]
        n_docs = len(docs)
        self.avgdl = sum(self.doc_len) / max(n_docs, 1)
        df: Counter = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1
        self.idf: dict[str, float] = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query_tokens: list[str], top_k: int) -> list[tuple[int, float]]:
        scores = [0.0] * len(self.doc_freqs)
        for t in query_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.doc_freqs):
                f = tf.get(t, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        ranked = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


def _snippet(text: str, width: int = 160) -> str:
    return " ".join((text or "").split())[:width]


def run(k: int, chunks_path: Path, out_path: Path) -> None:
    log.info("Loading chunks from %s", chunks_path)
    chunks, tokens = _load_chunks(chunks_path)
    log.info("Building BM25 over %d chunks", len(chunks))
    bm25 = BM25(tokens)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_queries = 0
    with out_path.open("w", encoding="utf-8") as out:
        for persona, src in PERSONA_FILES:
            with src.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    q = json.loads(line)["query"]
                    q_tok = _tokenize(q)
                    ranked = bm25.score(q_tok, top_k=k)
                    cands = []
                    for i, score in ranked:
                        c = chunks[i]
                        cands.append(
                            {
                                "chunk_id": c["chunk_id"],
                                "score": round(score, 3),
                                "publisher": c.get("publisher"),
                                "title": (c.get("title") or "")[:100],
                                "section": (c.get("section_heading") or "")[:100],
                                "snippet": _snippet(c.get("text") or "", width=160),
                            }
                        )
                    out.write(
                        json.dumps(
                            {"query": q, "persona": persona, "candidates": cands},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n_queries += 1
                    if n_queries % 10 == 0:
                        log.info("  annotated %d queries", n_queries)

    log.info("Wrote %d query records with top-%d candidates to %s", n_queries, k, out_path)
    print(f"queries={n_queries} k={k} out={out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--chunks", type=Path, default=Path("data/processed/chunks.jsonl")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/eval/candidates.jsonl")
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    run(k=args.k, chunks_path=args.chunks, out_path=args.out)


if __name__ == "__main__":
    main()
