"""Scraper for RBA Research Discussion Papers (RDPs).

Discovers PDF links from https://www.rba.gov.au/publications/rdp/ and
filters them down to housing-relevant titles using HOUSING_KEYWORDS.

Each RDP has a landing page linking to a PDF. This scraper follows the
index listing (one entry per paper) rather than every landing page, which
is enough to get title + URL + year.

Run directly to emit a JSONL of housing-filtered RDPs:

    python -m src.ingest.scrapers.rba --out data/raw/rba_rdp.jsonl
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from src.ingest.scrapers.common import (
    USER_AGENT,
    Source,
    matches_housing,
    write_jsonl,
)

log = logging.getLogger(__name__)

RDP_INDEX_URL = "https://www.rba.gov.au/publications/rdp/"
PUBLISHER = "RBA"


def fetch(url: str, *, client: httpx.Client | None = None) -> str:
    owns_client = client is None
    if client is None:
        client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=30.0,
            follow_redirects=True,
        )
    try:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text
    finally:
        if owns_client:
            client.close()


def parse_rdp_index(html: str, *, base_url: str = RDP_INDEX_URL) -> list[Source]:
    """Extract (title, year, url) tuples from the RDP listing page.

    The RDP index groups papers by year via <h2> year headings. Each paper is
    a list item whose first <a> points at the landing page (we resolve PDFs
    lazily later if needed — landing-page URLs are stable and suffice for
    metadata).
    """
    soup = BeautifulSoup(html, "lxml")
    sources: list[Source] = []
    current_year: str | None = None

    for node in soup.find_all(["h2", "h3", "li", "a"]):
        if node.name in {"h2", "h3"}:
            heading = node.get_text(strip=True)
            if heading.isdigit() and 1990 <= int(heading) <= 2100:
                current_year = heading
            continue

        if node.name == "li":
            link = node.find("a", href=True)
            if not link:
                continue
            title = link.get_text(strip=True)
            href = link["href"]
            if not title or not href:
                continue
            url = urljoin(base_url, href)
            if "/rdp/" not in url:
                continue
            sources.append(
                Source(
                    title=title,
                    publisher=PUBLISHER,
                    url=url,
                    date=current_year,
                    category="research-discussion-paper",
                )
            )

    seen: set[str] = set()
    unique: list[Source] = []
    for s in sources:
        if s.url in seen:
            continue
        seen.add(s.url)
        unique.append(s)
    return unique


def filter_housing(sources: list[Source]) -> list[Source]:
    return [s for s in sources if matches_housing(s.title)]


def scrape(*, index_url: str = RDP_INDEX_URL) -> list[Source]:
    log.info("Fetching RDP index: %s", index_url)
    html = fetch(index_url)
    all_sources = parse_rdp_index(html, base_url=index_url)
    log.info("Parsed %d RDPs; filtering by housing keywords", len(all_sources))
    housing = filter_housing(all_sources)
    log.info("Kept %d housing-relevant RDPs", len(housing))
    return housing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/rba_rdp.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    sources = scrape()
    write_jsonl(sources, args.out)
    log.info("Wrote %d sources → %s", len(sources), args.out)


if __name__ == "__main__":
    main()
