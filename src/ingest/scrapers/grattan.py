"""Scraper for Grattan Institute housing publications.

Grattan tags housing-related publications under
https://grattan.edu.au/topics/housing/. The topic page lists publications
(reports, working papers, submissions) with title + date. Each entry links
to a publication detail page; the detail page links to the final report
PDF. We record detail-page URLs; the downloader resolves PDFs later.

The topic page is wholly housing-relevant so no keyword filter is applied.

    python -m src.ingest.scrapers.grattan --out data/raw/grattan.jsonl
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

BASE = "https://grattan.edu.au"
TOPIC_URL = f"{BASE}/topics/housing/"
PUBLISHER = "Grattan Institute"
CATEGORY = "grattan-housing"

_DATE_RE = re.compile(
    r"(\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{4}|\b(19|20)\d{2}\b)"
)
_SKIP_PATH_PREFIXES = (
    "/topics/",
    "/people/",
    "/about/",
    "/category/",
    "/tag/",
    "/news/",
    "/events/",
    "/contact/",
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


def _looks_like_publication(path: str) -> bool:
    if not path.startswith("/"):
        return False
    if path in {"/", ""}:
        return False
    for prefix in _SKIP_PATH_PREFIXES:
        if path.startswith(prefix):
            return False
    segments = [s for s in path.split("/") if s]
    return len(segments) >= 1 and segments[0].lower() not in {
        "wp-content",
        "wp-admin",
        "feed",
    }


def parse_topic_page(html: str, *, base_url: str = TOPIC_URL) -> list[Source]:
    soup = BeautifulSoup(html, "lxml")
    sources: list[Source] = []

    for article in soup.find_all(["article", "div", "li"]):
        link = article.find("a", href=True)
        if not link:
            continue
        href: str = link["href"]
        url = urljoin(base_url, href)
        parsed = urlparse(url)
        if parsed.netloc and "grattan.edu.au" not in parsed.netloc:
            continue
        if not _looks_like_publication(parsed.path):
            continue
        title = link.get_text(strip=True)
        if not title or len(title) < 8:
            continue

        block_text = article.get_text(" ", strip=True)
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


def scrape(
    *,
    topic_url: str = TOPIC_URL,
    max_pages: int = 30,
    delay_s: float = 0.8,
) -> list[Source]:
    all_sources: list[Source] = []
    with _client() as client:
        for page in range(1, max_pages + 1):
            url = topic_url if page == 1 else f"{topic_url}page/{page}/"
            log.info("Fetching %s", url)
            try:
                html = fetch(url, client=client)
            except httpx.HTTPError as e:
                log.info("Stopping at page %d: %s", page, e)
                break
            page_sources = parse_topic_page(html, base_url=url)
            log.info("  parsed %d publications", len(page_sources))
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
    log.info("Total unique Grattan publications: %d", len(unique))
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/grattan.jsonl"),
    )
    parser.add_argument("--max-pages", type=int, default=30)
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
