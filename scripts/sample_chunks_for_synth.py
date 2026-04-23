"""Sample chunks from the corpus for synthetic-query authoring.

Stratified-random sample of chunks across publishers so synthetic
queries cover our source mix (not just AHURI, which dominates raw
chunk count). Prints a compact table of chunk_id / publisher / title /
section / snippet to stdout for inspection.

For Task 1.28: author one realistic persona query per sampled chunk,
with the chunk itself as the gold answer.

    python -m scripts.sample_chunks_for_synth --n 40 --seed 1
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

PUBLISHER_QUOTAS = {
    "AHURI": 12,
    "RBA": 6,
    "Productivity Commission": 5,
    "PropTrack": 5,
    "Housing Australia (NHFIC)": 4,
    "SQM Research": 3,
    "APRA": 2,
    "CoreLogic / Cotality": 2,
    "Australian Treasury": 1,
    "Grattan Institute": 0,
}
# quotas sum to 40


_NOISE_SECTION_RE = re.compile(
    r"^(references?|bibliography|acronyms|list of (figures|tables|acronyms)|"
    r"appendix \d|table of contents|terms (&|and) conditions|committees and charters|"
    r"[A-Z]{2,}—see )",
    re.IGNORECASE,
)
_OFF_TOPIC_TITLE_RE = re.compile(
    r"^(land management|land degradation|greenhouse|working from home|"
    r"terms (&|and) conditions|committees and charters)$",
    re.IGNORECASE,
)


def _is_substantive(rec: dict) -> bool:
    text = rec.get("text") or ""
    if not text:
        return False
    # Skip chunks that are mostly image placeholders or reference lists.
    if text.count("==> picture") > 3:
        return False
    if "**----- End of picture text -----**" in text and len(text) < 500:
        return False
    # Skip glossary / references / boilerplate sections.
    section = (rec.get("section_heading") or "").strip()
    if _NOISE_SECTION_RE.match(section):
        return False
    # Skip publications that aren't about housing.
    title = (rec.get("title") or "").strip()
    if _OFF_TOPIC_TITLE_RE.match(title):
        return False
    # Require at least some English words.
    words = [w for w in text.split() if w.isalpha()]
    return len(words) >= 60


def run(chunks_path: Path, n: int, seed: int, out_path: Path) -> None:
    log.info("Loading chunks from %s", chunks_path)
    by_pub: dict[str, list[dict]] = defaultdict(list)
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if _is_substantive(rec):
                by_pub[rec.get("publisher", "?")].append(rec)

    rng = random.Random(seed)
    sampled: list[dict] = []
    for pub, quota in PUBLISHER_QUOTAS.items():
        pool = by_pub.get(pub, [])
        if len(pool) < quota:
            log.warning("publisher %s: only %d substantive chunks, quota %d", pub, len(pool), quota)
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, quota))

    # Top-up if total less than n.
    if len(sampled) < n:
        remainder = [r for pool in by_pub.values() for r in pool if r not in sampled]
        sampled.extend(rng.sample(remainder, n - len(sampled)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as g:
        for r in sampled:
            g.write(
                json.dumps(
                    {
                        "chunk_id": r["chunk_id"],
                        "publisher": r.get("publisher"),
                        "title": r.get("title"),
                        "section_heading": r.get("section_heading"),
                        "text": r.get("text"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    log.info("Wrote %d sampled chunks to %s", len(sampled), out_path)
    print(f"sampled={len(sampled)} out={out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("data/eval/synth_source_chunks.jsonl"))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    run(chunks_path=args.chunks, n=args.n, seed=args.seed, out_path=args.out)


if __name__ == "__main__":
    main()
