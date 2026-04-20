"""Scrapers for Domain Research and PropTrack Insights.

Both publishers release quarterly housing reports (house price reports,
rental reports, market reports) that are core references for AU housing
analysis.

- Domain: https://www.domain.com.au/research/
- PropTrack: https://www.proptrack.com.au/insights/ (also mirrored at
  https://www.realestate.com.au/insights/proptrack/ for some reports)

Both listings are wholly housing-relevant so no keyword filter is applied.

    python -m src.ingest.scrapers.domain_proptrack --out data/raw/domain_proptrack.jsonl
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

DOMAIN_BASE = "https://www.domain.com.au"
DOMAIN_LISTING_CANDIDATES = (
    "/research/",
    "/research",
)
DOMAIN_PUBLISHER = "Domain"
DOMAIN_CATEGORY = "market-research"

PROPTRACK_BASE_CANDIDATES = (
    "https://www.proptrack.com.au",
    "https://www.realestate.com.au",
)
PROPTRACK_LISTING_PATHS = (
    "/insights",
    "/insights/",
    "/insights/proptrack",
    "/insights/proptrack/",
)
PROPTRACK_PUBLISHER = "PropTrack"
PROPTRACK_CATEGORY = "market-research"

_DATE_RE = re.compile(
    r"(\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},\s*\d{4}|Q[1-4]\s*\d{4}|\b(19|20)\d{2}\b)"
)

_COMMON_SKIP_PREFIXES = (
    "/contact",
    "/about",
    "/careers",
    "/terms",
    "/privacy",
    "/cookie",
    "/help",
    "/login",
    "/register",
    "/newsletter",
    "/subscribe",
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


def _discover(
    client: httpx.Client,
    bases: tuple[str, ...],
    paths: tuple[str, ...],
) -> tuple[str, str]:
    """Return (base, listing_url) for the first 200-response combination."""
    for base in bases:
        for path in paths:
            url = f"{base}{path}"
            try:
                resp = client.get(url)
            except httpx.HTTPError:
                continue
            if resp.status_code == 200 and len(resp.text) > 500:
                return base, url
    raise RuntimeError(f"No reachable listing among bases={bases} paths={paths}")


def _looks_like_article(
    url: str,
    *,
    listing_path: str,
    allowed_netlocs: tuple[str, ...],
    required_prefixes: tuple[str, ...],
) -> bool:
    parsed = urlparse(url)
    if parsed.netloc and not any(n in parsed.netloc for n in allowed_netlocs):
        return False
    path = parsed.path.lower()
    if not path or path == "/":
        return False
    if path == listing_path or path == listing_path.rstrip("/"):
        return False
    for skip in _COMMON_SKIP_PREFIXES:
        if path.startswith(skip):
            return False
    # Article path must live under one of the required prefixes (research/insights).
    return any(path.startswith(p) for p in required_prefixes)


def _parse_listing(
    html: str,
    *,
    base_url: str,
    listing_path: str,
    publisher: str,
    category: str,
    allowed_netlocs: tuple[str, ...],
    required_prefixes: tuple[str, ...],
) -> list[Source]:
    soup = BeautifulSoup(html, "lxml")
    sources: list[Source] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        url = urljoin(base_url, href)
        if not _looks_like_article(
            url,
            listing_path=listing_path,
            allowed_netlocs=allowed_netlocs,
            required_prefixes=required_prefixes,
        ):
            continue
        title = a.get_text(strip=True)
        if not title or len(title) < 8:
            continue

        block = a.find_parent(["article", "div", "li", "section"]) or a
        block_text = block.get_text(" ", strip=True)
        date_match = _DATE_RE.search(block_text)
        date = date_match.group(0) if date_match else None

        sources.append(
            Source(
                title=title,
                publisher=publisher,
                url=url,
                date=date,
                category=category,
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


def _walk_pages(
    *,
    listing_url: str,
    listing_path: str,
    publisher: str,
    category: str,
    allowed_netlocs: tuple[str, ...],
    required_prefixes: tuple[str, ...],
    client: httpx.Client,
    max_pages: int,
    delay_s: float,
) -> list[Source]:
    all_sources: list[Source] = []
    for page in range(1, max_pages + 1):
        url = listing_url if page == 1 else f"{listing_url.rstrip('/')}/page/{page}/"
        log.info("Fetching %s", url)
        try:
            html = fetch(url, client=client)
        except httpx.HTTPError as e:
            log.info("Stopping at page %d: %s", page, e)
            break
        page_sources = _parse_listing(
            html,
            base_url=url,
            listing_path=listing_path,
            publisher=publisher,
            category=category,
            allowed_netlocs=allowed_netlocs,
            required_prefixes=required_prefixes,
        )
        log.info("  parsed %d items", len(page_sources))
        if not page_sources:
            break
        if page > 1:
            prev_urls = {s.url for s in all_sources}
            new_urls = {s.url for s in page_sources} - prev_urls
            if not new_urls:
                log.info("  no new URLs on page %d; stopping", page)
                break
        all_sources.extend(page_sources)
        time.sleep(delay_s)
    return all_sources


def scrape_domain(*, max_pages: int = 20, delay_s: float = 0.8) -> list[Source]:
    with _client() as client:
        _, listing_url = _discover(
            client,
            bases=(DOMAIN_BASE,),
            paths=DOMAIN_LISTING_CANDIDATES,
        )
        log.info("Domain research listing: %s", listing_url)
        listing_path = urlparse(listing_url).path.rstrip("/") or "/"
        sources = _walk_pages(
            listing_url=listing_url,
            listing_path=listing_path,
            publisher=DOMAIN_PUBLISHER,
            category=DOMAIN_CATEGORY,
            allowed_netlocs=("domain.com.au",),
            required_prefixes=("/research/",),
            client=client,
            max_pages=max_pages,
            delay_s=delay_s,
        )
    seen: set[str] = set()
    unique = [s for s in sources if not (s.url in seen or seen.add(s.url))]
    log.info("Total unique Domain research items: %d", len(unique))
    return unique


def scrape_proptrack(*, max_pages: int = 20, delay_s: float = 0.8) -> list[Source]:
    with _client() as client:
        base, listing_url = _discover(
            client,
            bases=PROPTRACK_BASE_CANDIDATES,
            paths=PROPTRACK_LISTING_PATHS,
        )
        log.info("PropTrack insights listing: %s (base %s)", listing_url, base)
        listing_path = urlparse(listing_url).path.rstrip("/") or "/"
        # realestate.com.au mirrors PropTrack under /insights/proptrack, so
        # we must pin required_prefixes to the actual listing path rather
        # than a blanket "/insights".
        required_prefix = listing_path + "/" if not listing_path.endswith("/") else listing_path
        sources = _walk_pages(
            listing_url=listing_url,
            listing_path=listing_path,
            publisher=PROPTRACK_PUBLISHER,
            category=PROPTRACK_CATEGORY,
            allowed_netlocs=("proptrack.com.au", "realestate.com.au"),
            required_prefixes=(required_prefix,),
            client=client,
            max_pages=max_pages,
            delay_s=delay_s,
        )
    seen: set[str] = set()
    unique = [s for s in sources if not (s.url in seen or seen.add(s.url))]
    log.info("Total unique PropTrack items: %d", len(unique))
    return unique


def scrape(*, max_pages: int = 20, delay_s: float = 0.8) -> list[Source]:
    """Run both Domain and PropTrack scrapers; dedupe by URL across both."""
    all_sources: list[Source] = []
    for name, fn in (("domain", scrape_domain), ("proptrack", scrape_proptrack)):
        try:
            all_sources.extend(fn(max_pages=max_pages, delay_s=delay_s))
        except Exception as e:  # noqa: BLE001 — each sub-scraper can fail independently
            log.exception("Sub-scraper %s failed: %s", name, e)
    seen: set[str] = set()
    unique: list[Source] = []
    for s in all_sources:
        if s.url in seen:
            continue
        seen.add(s.url)
        unique.append(s)
    log.info("Total unique Domain+PropTrack items: %d", len(unique))
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "which",
        nargs="?",
        choices=("domain", "proptrack", "all"),
        default="all",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/domain_proptrack.jsonl"),
    )
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if args.which == "domain":
        sources = scrape_domain(max_pages=args.max_pages)
    elif args.which == "proptrack":
        sources = scrape_proptrack(max_pages=args.max_pages)
    else:
        sources = scrape(max_pages=args.max_pages)
    write_jsonl(sources, args.out)
    log.info("Wrote %d sources → %s", len(sources), args.out)


if __name__ == "__main__":
    main()
