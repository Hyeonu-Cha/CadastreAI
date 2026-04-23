"""Sanity-check retrieval on a fixed set of canonical queries.

Runs 5 hand-picked queries across the main personas (homebuyer,
investor, researcher, regulator/macro) and prints the top-5 hits for
each. Intended as a quick-glance sniff test after re-indexing — not an
evaluation. MRR / hit-rate measurement comes in Task 1.24.

Assumes `cadastre_chunks` is populated (Task 1.21) and Qdrant is
reachable via the same env vars `src.retrieval.retriever` reads:
`QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`.

    QDRANT_API_KEY=dev-local python -m scripts.sanity_retrieve
"""
from __future__ import annotations

import argparse
import logging
import sys
import textwrap

from src.retrieval.retriever import Retriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

QUERIES: list[tuple[str, str]] = [
    ("macro",      "impact of interest rate rises on Australian housing prices"),
    ("investor",   "negative gearing effect on rental supply and investor activity"),
    ("homebuyer",  "first home buyer grant eligibility NSW"),
    ("researcher", "affordable housing shortage in Melbourne and Sydney"),
    ("regulator",  "APRA lending standards and mortgage serviceability buffer"),
]


def _fmt_snippet(text: str, width: int = 100) -> str:
    clean = " ".join((text or "").split())
    return textwrap.shorten(clean, width=width, placeholder=" …")


def run(k: int = 5) -> None:
    r = Retriever()
    for persona, q in QUERIES:
        print(f"\n=== [{persona}] {q}")
        hits = r.retrieve(q, k=k)
        for i, (payload, score) in enumerate(hits, start=1):
            title = (payload.get("title") or "")[:80]
            section = (payload.get("section_heading") or "")[:80]
            url = payload.get("url") or ""
            snippet = _fmt_snippet(payload.get("text") or "", width=140)
            print(f"  [{i}] score={score:.4f}  {payload.get('publisher')} | {title}")
            if section:
                print(f"       section: {section}")
            print(f"       {url}")
            print(f"       {snippet}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    run(k=args.k)


if __name__ == "__main__":
    main()
