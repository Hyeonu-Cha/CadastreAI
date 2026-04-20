"""Scraper for CoreLogic / Cotality AU news & research articles.

CoreLogic AU now publishes under the Cotality brand at
https://www.cotality.com/au/news-research (formerly
https://www.corelogic.com.au/news-research). We walk the listing pagination
and collect article URLs + titles. The listing page for news-research is
wholly housing-relevant so no keyword filter is applied; the `Home Value
Index`, `Pain & Gain`, and `Monthly Indices` methodology posts are what we
really want but it's easier to keep everything and dedupe later.

    python -m src.ingest.scrapers.corelogic --out data/raw/corelogic.jsonl
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

BASE_CANDIDATES = (
    "https://www.cotality.com",
    "https://www.corelogic.com.au",
)
LISTING_PATHS = ("/au/news-research", "/news-research")
PUBLISHER = "CoreLogic / Cotality"
CATEGORY = "market-research"

_DATE_RE = re.compile(
    r"(\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},\s*\d{4}|\b(19|20)\d{2}\b)"
)


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


def discover_base(client: httpx.Client) -> tuple[str, str]:
    """Return the first (base, listing_url) pair that responds 200."""
    for base in BASE_CANDIDATES:
        for path in LISTING_PATHS:
            url = f"{base}{path}"
            try:
                resp = client.get(url)
            except httpx.HTTPError:
                continue
            if resp.status_code == 200:
                return base, url
    raise RuntimeError(
        f"No reachable CoreLogic/Cotality listing among candidates: {BASE_CANDIDATES}"
    )


def _is_article_path(path: str, listing_path: str) -> bool:
    if not path.startswith("/"):
        return False
    if path == listing_path or path == listing_path + "/":
        return False
    # Article paths live under the news-research tree or root with slug.
    if path.startswith(listing_path.rstrip("/") + "/"):
        return True
    # Accept /au/<slug>/ style too (Cotality moves articles around).
    segments = [s for s in path.split("/") if s]
    if len(segments) >= 2 and segments[0] in {"au"} and segments[1] not in {
        "news-research",
        "products",
        "contact",
        "about",
    }:
        return True
    return False


def parse_listing(html: str, *, base_url: str, listing_path: str) -> list[Source]:
    soup = BeautifulSoup(html, "lxml")
    sources: list[Source] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        url = urljoin(base_url, href)
        parsed = urlparse(url)
        if parsed.netloc and parsed.netloc not in urlparse(base_url).netloc:
            continue
        if not _is_article_path(parsed.path, listing_path):
            continue
        title = a.get_text(strip=True)
        if not title or len(title) < 8:
            continue

        block = a.find_parent(["article", "div", "li"]) or a
        block_text = block.get_text(" ", strip=True)
        date_match = _DATE_RE.search(block_text)
        date = date_match.group(0) if date_match else None

        sources.append(
            Source(
                title=title,
                publisher=PUBLISHER,
                url=url,
                date=date,
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


def scrape(*, max_pages: int = 25, delay_s: float = 0.8) -> list[Source]:
    all_sources: list[Source] = []
    with _client() as client:
        base, listing_url = discover_base(client)
        log.info("CoreLogic/Cotality base resolved to %s (listing %s)", base, listing_url)
        listing_path = urlparse(listing_url).path

        for page in range(1, max_pages + 1):
            url = listing_url if page == 1 else f"{listing_url.rstrip('/')}/page/{page}/"
            log.info("Fetching %s", url)
            try:
                html = fetch(url, client=client)
            except httpx.HTTPError as e:
                log.info("Stopping at page %d: %s", page, e)
                break
            page_sources = parse_listing(html, base_url=url, listing_path=listing_path)
            log.info("  parsed %d articles", len(page_sources))
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
    log.info("Total unique CoreLogic/Cotality articles: %d", len(unique))
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/corelogic.jsonl"))
    parser.add_argument("--max-pages", type=int, default=25)
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
