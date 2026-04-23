"""Turn BM25 candidates into annotated gold queries for eval.

Reads `data/eval/candidates.jsonl` (produced by gather_eval_candidates.py)
and writes `data/eval/queries.jsonl` with one record per query:

    {"query": "...", "persona": "...", "gold_chunk_ids": [...]}

For each query we keep the top-N candidates whose title/section/snippet
passes a noise filter. Dropped: bibliography/references pages, site-nav
boilerplate ("skip to Content"), image-only chunks, glossary entries
(sections like "ECOSA—see Essential Services Commission ..."). These
are indexable content but don't answer questions, so would be noise
in an eval gold set.

Picking deterministically from BM25 top-10 is imperfect but defensible:
(1) BM25 was chosen precisely to avoid circularity with the BGE
retriever we evaluate, and (2) top-3 after filtering gives the retriever
a realistic bar — if it can't surface *any* of the 2-3 lexically-salient
chunks we flagged, that's a real recall miss.

    python -m scripts.build_gold_queries --top 3
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

_NOISE_SECTION_RE = re.compile(
    r"^(references?|bibliography|acronyms|list of (figures|tables|acronyms)|"
    r"appendix \d|skip to (main )?(content|navigation)|"
    r"[A-Z]{2,}—see |pwc—see |ecosa—see )",
    re.IGNORECASE,
)
_NOISE_TITLE_RE = re.compile(
    r"^(skip to (main )?(content|navigation)|download)$",
    re.IGNORECASE,
)


def _is_noise(cand: dict) -> bool:
    section = (cand.get("section") or "").strip()
    title = (cand.get("title") or "").strip()
    snippet = (cand.get("snippet") or "").strip()

    if _NOISE_TITLE_RE.match(title):
        return True
    if _NOISE_SECTION_RE.match(section):
        return True
    # Image-only chunks: snippet dominated by image placeholders.
    if snippet.startswith("**==>") or snippet.startswith("<!-- image -->"):
        return True
    # realestate.com.au navigation pages — snippet is just the site menu.
    if snippet.lstrip("- ").startswith("News - Insights - Guides - Lifestyle"):
        return True
    # Nav-list pattern with no substantive content.
    if snippet.count(" - ") >= 4 and "Insights" in snippet and "Lifestyle" in snippet:
        return True
    return False


def run(src: Path, out: Path, top_n: int) -> None:
    log.info("Reading candidates from %s", src)
    n_queries = 0
    n_gold_total = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            query = rec["query"]
            persona = rec["persona"]
            kept = []
            for c in rec.get("candidates", []):
                if _is_noise(c):
                    continue
                kept.append(c["chunk_id"])
                if len(kept) >= top_n:
                    break
            if not kept:
                log.warning("no gold chunks for query: %s", query)
                # Fallback: take the highest-BM25 candidate even if filter flagged it,
                # so every query has at least one gold (retriever still gets scored).
                if rec.get("candidates"):
                    kept = [rec["candidates"][0]["chunk_id"]]
            g.write(
                json.dumps(
                    {"query": query, "persona": persona, "gold_chunk_ids": kept},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_queries += 1
            n_gold_total += len(kept)

    log.info(
        "Wrote %d queries (%d gold chunks, avg %.2f/query) to %s",
        n_queries,
        n_gold_total,
        n_gold_total / max(n_queries, 1),
        out,
    )
    print(f"queries={n_queries} gold={n_gold_total} avg={n_gold_total / max(n_queries, 1):.2f} out={out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=3, help="max gold chunks per query")
    parser.add_argument("--src", type=Path, default=Path("data/eval/candidates.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/eval/queries.jsonl"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    run(src=args.src, out=args.out, top_n=args.top)


if __name__ == "__main__":
    main()
