"""Scraper for SQM Research press releases and reports.

SQM Research publishes monthly housing indices (vacancy rates, asking
rents/prices, stock on market, distressed listings) through press releases
on https://sqmresearch.com.au/. The index lives at
`/press-releases.php`; each release links to an individual press-release
page (usually `press-release-YYYY-MM-DD-<slug>.php`) and/or a direct PDF.

SQM's output is wholly housing-focused so no keyword filter is applied.

    python -m src.ingest.scrapers.sqm --out data/raw/sqm.jsonl
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

BASE = "https://sqmresearch.com.au"
LISTING_CANDIDATES = (
    "/press-releases.php",
    "/news.php",
    "/media-releases.php",
)
PUBLISHER = "SQM Research"
CATEGORY = "market-research"

_DATE_RE = re.compile(
    r"(\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},\s*\d{4}|\b(19|20)\d{2}\b)"
)
_ISO_DATE_RE = re.compile(r"(19|20)\d{2}-\d{2}-\d{2}")

_SKIP_EXTENSIONS = {".jpg", ".png", ".gif", ".svg", ".css", ".js"}
_SKIP_PATH_PREFIXES = (
    "/general/",
    "/contact",
    "/about",
    "/subscribe",
    "/login",
    "/register",
    "/cart",
    "/account",
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


def discover_listing(client: httpx.Client) -> str:
    """Return the first listing URL on sqmresearch.com.au that responds 200."""
    for path in LISTING_CANDIDATES:
        url = f"{BASE}{path}"
        try:
            resp = client.get(url)
        except httpx.HTTPError:
            continue
        if resp.status_code == 200 and len(resp.text) > 500:
            return url
    raise RuntimeError(
        f"No reachable SQM listing among candidates: {LISTING_CANDIDATES}"
    )


def _looks_like_report(url: str, listing_path: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc and "sqmresearch.com.au" not in parsed.netloc:
        return False
    path = parsed.path
    if not path or path == "/":
        return False
    if path == listing_path:
        return False
    lower = path.lower()
    for prefix in _SKIP_PATH_PREFIXES:
        if lower.startswith(prefix):
            return False
    for ext in _SKIP_EXTENSIONS:
        if lower.endswith(ext):
            return False
    # Accept direct PDFs or press-release/news detail pages.
    if lower.endswith(".pdf"):
        return True
    if "press-release" in lower or "media-release" in lower or "news" in lower:
        return True
    # Some SQM detail pages are dated slugs like /report-2024-06-01.php
    if _ISO_DATE_RE.search(lower):
        return True
    return False


def parse_listing(html: str, *, base_url: str, listing_path: str) -> list[Source]:
    soup = BeautifulSoup(html, "lxml")
    sources: list[Source] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        url = urljoin(base_url, href)
        if not _looks_like_report(url, listing_path):
            continue
        title = a.get_text(strip=True)
        if not title or len(title) < 8:
            # Fallback: use filename stem so we still capture pure-PDF links.
            stem = Path(urlparse(url).path).stem.replace("-", " ").replace("_", " ").strip()
            if len(stem) < 8:
                continue
            title = stem

        block = a.find_parent(["article", "div", "li", "tr", "p"]) or a
        block_text = block.get_text(" ", strip=True)
        date_match = _ISO_DATE_RE.search(block_text) or _DATE_RE.search(block_text)
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


def scrape(*, max_pages: int = 20, delay_s: float = 0.8) -> list[Source]:
    """Walk the SQM press-releases index.

    SQM's press-releases page is typically a single long list rather than
    numbered pagination, but some templates expose `?page=N`. We try both.
    """
    all_sources: list[Source] = []
    with _client() as client:
        listing_url = discover_listing(client)
        log.info("SQM listing resolved to %s", listing_url)
        listing_path = urlparse(listing_url).path

        for page in range(1, max_pages + 1):
            url = listing_url if page == 1 else f"{listing_url}?page={page}"
            log.info("Fetching %s", url)
            try:
                html = fetch(url, client=client)
            except httpx.HTTPError as e:
                log.info("Stopping at page %d: %s", page, e)
                break
            page_sources = parse_listing(html, base_url=url, listing_path=listing_path)
            log.info("  parsed %d reports", len(page_sources))
            if not page_sources:
                break
            # If page 2+ yields the same URLs as page 1, the site has no real
            # pagination — stop so we don't loop pulling identical content.
            if page > 1:
                prev_urls = {s.url for s in all_sources}
                new_urls = {s.url for s in page_sources} - prev_urls
                if not new_urls:
                    log.info("  no new URLs on page %d; assuming no pagination", page)
                    all_sources.extend(page_sources)
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
    log.info("Total unique SQM reports: %d", len(unique))
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/sqm.jsonl"))
    parser.add_argument("--max-pages", type=int, default=20)
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
