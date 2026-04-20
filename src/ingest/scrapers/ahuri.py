"""Scraper for AHURI Final Reports.

AHURI (Australian Housing and Urban Research Institute) publishes all
final reports under its research library at
https://www.ahuri.edu.au/research/research-library with a paginated
listing UI (Drupal-style `?page=N`, 0-indexed: bare URL = page 0,
`?page=1` = page 2, …). Each list item links to a report detail page
(`/research/final-reports/<N>`) which in turn links to the PDF. We
capture the detail-page URL plus title/year from the list; PDF
resolution happens in the downloader.

The entire AHURI corpus is housing-relevant, so we do not apply the
`matches_housing` keyword filter (unlike RBA's broader catalogue).

    python -m src.ingest.scrapers.ahuri --out data/raw/ahuri.jsonl
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from src.ingest.scrapers.common import USER_AGENT, Source, write_jsonl

log = logging.getLogger(__name__)

BASE = "https://www.ahuri.edu.au"
LISTING_URL = f"{BASE}/research/research-library"
PUBLISHER = "AHURI"
CATEGORY = "final-report"

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_REPORT_PATH_RE = re.compile(r"^/research/final-reports/\d+/?$")


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
    )


def fetch(url: str, *, client: httpx.Client) -> str:
    resp = client.get(url)
    resp.raise_for_status()
    return resp.text


def parse_listing_page(html: str, *, base_url: str = LISTING_URL) -> list[Source]:
    """Pull each report card from a single AHURI listing page."""
    soup = BeautifulSoup(html, "lxml")
    sources: list[Source] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        url = urljoin(base_url, href)
        path = urlparse(url).path
        if not _REPORT_PATH_RE.match(path):
            continue
        title = a.get_text(strip=True)
        if not title or len(title) < 8:
            continue  # skip "Read more" / icon links

        parent_text = ""
        parent = a.find_parent(["article", "div", "li"])
        if parent is not None:
            parent_text = parent.get_text(" ", strip=True)
        year_match = _YEAR_RE.search(parent_text)
        year = year_match.group(0) if year_match else None

        sources.append(
            Source(
                title=title,
                publisher=PUBLISHER,
                url=url,
                date=year,
                category=CATEGORY,
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


def scrape(
    *,
    listing_url: str = LISTING_URL,
    max_pages: int = 60,
    delay_s: float = 0.8,
) -> list[Source]:
    """Walk paginated listing pages until one returns zero reports."""
    all_sources: list[Source] = []
    with _client() as client:
        for page in range(max_pages):
            url = listing_url if page == 0 else f"{listing_url}?page={page}"
            log.info("Fetching %s", url)
            try:
                html = fetch(url, client=client)
            except httpx.HTTPError as e:
                log.warning("Stopping at page %d: %s", page, e)
                break
            page_sources = parse_listing_page(html, base_url=url)
            log.info("  parsed %d reports", len(page_sources))
            if not page_sources:
                break
            all_sources.extend(page_sources)
            time.sleep(delay_s)

    seen: set[str] = set()
    unique: list[Source] = []
    for s in all_sources:
        if s.url in seen:
            continue
        seen.add(s.url)
        unique.append(s)
    log.info("Total unique AHURI reports: %d", len(unique))
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/ahuri.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--max-pages", type=int, default=60)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    sources = scrape(max_pages=args.max_pages)
    write_jsonl(sources, args.out)
    log.info("Wrote %d sources → %s", len(sources), args.out)


if __name__ == "__main__":
    main()
