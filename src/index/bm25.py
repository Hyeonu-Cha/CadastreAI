"""BM25 index over the chunk corpus, parallel to the dense Qdrant index.

Hybrid retrieval needs a sparse lexical signal to complement BGE's dense
signal — BM25 is the standard choice. Rather than run BM25 through
Qdrant's sparse-vector API (which needs a recollection with sparse
config), we keep an in-process index here: it's small (~42k docs), loads
in ~1s, and interops cleanly with the RRF fusion code in Task 2.02.

The class mirrors `src.retrieval.retriever.Retriever`'s interface —
`retrieve(query, k) -> list[(chunk_dict, score)]` — so hybrid fusion can
treat dense and sparse sources as swappable.

Tokenizer notes:
- Lowercased alphanum tokens (re `[A-Za-z0-9]+`), no stemming.
- A small domain-aware stopword list: common English fillers plus the
  words that dominate every housing chunk (home, housing, property,
  Australia, ...). Without that filter, every query matches every chunk
  on "housing" and IDF effectively disappears.

Build once, query many:

    python -m src.index.bm25 build --out data/processed/bm25.pkl
    python -m src.index.bm25 query --query "negative gearing effects" -k 5
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

DEFAULT_CHUNKS = Path("data/processed/chunks.jsonl")
DEFAULT_INDEX = Path("data/processed/bm25.pkl")

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


def tokenize(text: str) -> list[str]:
    """Lowercased alphanum tokens with stopword + domain-filler removal."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOPWORDS]


@dataclass
class BM25Index:
    """BM25 index wrapping rank_bm25.BM25Okapi + the original chunk payloads.

    Chunks are stored in rank order of construction; position i in `chunks`
    corresponds to document i in the rank_bm25 model. `save` pickles both.
    """

    chunks: list[dict] = field(default_factory=list)
    _bm25: object = None  # rank_bm25.BM25Okapi, typed as object to keep import lazy

    @classmethod
    def build(cls, chunks_path: Path) -> BM25Index:
        from rank_bm25 import BM25Okapi

        log.info("Loading chunks from %s", chunks_path)
        chunks: list[dict] = []
        tokens: list[list[str]] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                chunks.append(rec)
                tokens.append(tokenize(rec.get("text", "")))
        log.info("Fitting BM25 over %d documents", len(chunks))
        bm25 = BM25Okapi(tokens)
        return cls(chunks=chunks, _bm25=bm25)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            state = {"chunks": self.chunks, "bm25": self._bm25}
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Wrote BM25 index to %s (%d docs)", path, len(self.chunks))

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with path.open("rb") as f:
            state = pickle.load(f)
        log.info("Loaded BM25 index from %s (%d docs)", path, len(state["chunks"]))
        return cls(chunks=state["chunks"], _bm25=state["bm25"])

    def retrieve(self, query: str, k: int = 10) -> list[tuple[dict, float]]:
        if self._bm25 is None:
            raise RuntimeError("BM25 model not built or loaded.")
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        # Partial sort: take top-k indices by score desc.
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]


_DEFAULT: BM25Index | None = None


def retrieve(query: str, k: int = 10, index_path: Path = DEFAULT_INDEX) -> list[tuple[dict, float]]:
    """Module-level convenience wrapper — caches the loaded index."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = BM25Index.load(index_path)
    return _DEFAULT.retrieve(query, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build BM25 index from chunks.jsonl")
    b.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    b.add_argument("--out", type=Path, default=DEFAULT_INDEX)

    q = sub.add_parser("query", help="Query an existing BM25 index")
    q.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    q.add_argument("--query", required=True)
    q.add_argument("-k", type=int, default=10)

    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if args.cmd == "build":
        idx = BM25Index.build(args.chunks)
        idx.save(args.out)
        print(f"built={len(idx.chunks)} out={args.out}")
    elif args.cmd == "query":
        idx = BM25Index.load(args.index)
        hits = idx.retrieve(args.query, k=args.k)
        for i, (c, s) in enumerate(hits, start=1):
            title = (c.get("title") or "")[:70]
            section = (c.get("section_heading") or "")[:70]
            print(f"[{i:2d}] score={s:.3f}  {c.get('publisher')} | {title}")
            if section:
                print(f"     {section}")


if __name__ == "__main__":
    main()
