"""Run every available scraper and merge results into a single sources.jsonl.

Schema — one JSON object per line, fields from `src.ingest.scrapers.common.Source`:

    {
      "title":     str,            # human-readable title
      "publisher": str,            # e.g. "RBA", "AHURI", "Grattan Institute"
      "url":       str,            # landing page or direct PDF URL
      "date":      str | null,     # free-form; typically "YYYY", "YYYY-mon", or a date string
      "category":  str | null,     # e.g. "research-discussion-paper", "final-report"
      "extra":     dict            # scraper-specific extras (usually empty)
    }

Usage:

    python scripts/collect_sources.py --out data/raw/sources.jsonl
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

from src.ingest.scrapers import (
    ahuri,
    corelogic,
    domain_proptrack,
    grattan,
    nhfic_apra,
    rba,
    sqm,
    treasury_pc,
)
from src.ingest.scrapers.common import Source, write_jsonl

log = logging.getLogger(__name__)

SCRAPERS = {
    "rba": rba.scrape_all,
    "ahuri": ahuri.scrape,
    "grattan": grattan.scrape,
    "corelogic": corelogic.scrape,
    "sqm": sqm.scrape,
    "domain_proptrack": domain_proptrack.scrape,
    "treasury_pc": treasury_pc.scrape,
    "nhfic_apra": nhfic_apra.scrape,
}


def collect(selected: list[str]) -> list[Source]:
    all_sources: list[Source] = []
    for name in selected:
        fn = SCRAPERS[name]
        log.info("=== Running scraper: %s ===", name)
        try:
            sources = fn()
        except Exception as e:  # noqa: BLE001 — scrapers can fail network-side
            log.exception("Scraper %s failed: %s", name, e)
            continue
        log.info("  +%d sources from %s", len(sources), name)
        all_sources.extend(sources)

    seen: set[str] = set()
    unique: list[Source] = []
    for s in all_sources:
        if s.url in seen:
            continue
        seen.add(s.url)
        unique.append(s)
    log.info("Total unique sources: %d", len(unique))
    return unique


def summarize(sources: list[Source]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in sources:
        counts[s.publisher] = counts.get(s.publisher, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/sources.jsonl"),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(SCRAPERS.keys()),
        help="Only run the given scrapers (default: all)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    selected = args.only or list(SCRAPERS.keys())
    sources = collect(selected)
    write_jsonl(sources, args.out)

    counts = summarize(sources)
    log.info("Wrote %d sources → %s", len(sources), args.out)
    for pub, n in sorted(counts.items(), key=lambda x: -x[1]):
        log.info("  %-20s %d", pub, n)

    # Also emit a small stats file for CI/README generation later.
    stats_path = args.out.with_suffix(".stats.json")
    import json

    stats_path.write_text(
        json.dumps(
            {"total": len(sources), "by_publisher": counts},
            indent=2,
        ),
        encoding="utf-8",
    )

    # Sanity print for CLI users
    print(f"total={len(sources)} by_publisher={counts}")
    # Use asdict to make sure Source is JSON-serializable in future callers.
    _ = [asdict(s) for s in sources[:1]]


if __name__ == "__main__":
    main()
