"""Scrapers for Treasury + Productivity Commission housing pages.

Both publishers cover far more than housing, so unlike the specialist
scrapers (AHURI, Grattan, CoreLogic, SQM, Domain/PropTrack), we DO apply
the `matches_housing` keyword filter against titles and surrounding card
text.

- Treasury: https://treasury.gov.au/ — housing-relevant material lives
  under /policy-topics/housing/ listing pages and publication detail pages
  at /publication/<slug>. The site also has a Policy Topics hub with
  direct PDF links to housing papers (e.g. Intergenerational Report,
  Housing Accord commitments).
- Productivity Commission: https://www.pc.gov.au/ — housing-related
  reports appear under /inquiries/completed/<slug>/, /research/*, and
  /ongoing/*. We walk the topic tag page at /topics/housing first, then
  fall back to the inquiries listing.

    python -m src.ingest.scrapers.treasury_pc --out data/raw/treasury_pc.jsonl
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

from src.ingest.scrapers.common import (
    USER_AGENT,
    Source,
    matches_housing,
    write_jsonl,
)

log = logging.getLogger(__name__)

TREASURY_BASE = "https://treasury.gov.au"
TREASURY_LISTING_CANDIDATES = (
    "/policy-topics/housing",
    "/policy-topics/housing/",
    "/topics/housing",
)
TREASURY_PUBLISHER = "Australian Treasury"
TREASURY_CATEGORY = "government"

PC_BASE = "https://www.pc.gov.au"
PC_LISTING_CANDIDATES = (
    "/topics/housing",
    "/topics/housing/",
    "/ongoing/housing",
    "/inquiries/completed",
)
PC_PUBLISHER = "Productivity Commission"
PC_CATEGORY = "government"

_DATE_RE = re.compile(
    r"(\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},\s*\d{4}|\b(19|20)\d{2}\b)"
)

_SKIP_PATH_TOKENS = (
    "/contact",
    "/about",
    "/careers",
    "/copyright",
    "/privacy",
    "/disclaimer",
    "/accessibility",
    "/sitemap",
    "/news-and-media",
    "/media-release",
    "/search",
)

_ALLOWED_EXT = {".pdf", ".html", ".htm", ""}


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
    base: str,
    paths: tuple[str, ...],
) -> str:
    for path in paths:
        url = f"{base}{path}"
        try:
            resp = client.get(url)
        except httpx.HTTPError:
            continue
        if resp.status_code == 200 and len(resp.text) > 500:
            return url
    raise RuntimeError(f"No reachable listing on {base} among {paths}")


def _looks_like_content(url: str, *, allowed_netloc: str, listing_path: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc and allowed_netloc not in parsed.netloc:
        return False
    path = parsed.path.lower()
    if not path or path == "/" or path == listing_path:
        return False
    for tok in _SKIP_PATH_TOKENS:
        if path.startswith(tok):
            return False
    ext = Path(path).suffix
    if ext not in _ALLOWED_EXT:
        return False
    # Require at least two path segments to avoid navigation roots like /publications.
    segments = [s for s in path.split("/") if s]
    return len(segments) >= 2


def _parse_listing(
    html: str,
    *,
    base_url: str,
    listing_path: str,
    allowed_netloc: str,
    publisher: str,
    category: str,
    apply_keyword_filter: bool,
) -> list[Source]:
    soup = BeautifulSoup(html, "lxml")
    sources: list[Source] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        url = urljoin(base_url, href)
        if not _looks_like_content(
            url, allowed_netloc=allowed_netloc, listing_path=listing_path
        ):
            continue
        title = a.get_text(strip=True)
        if not title or len(title) < 8:
            stem = Path(urlparse(url).path).stem.replace("-", " ").replace("_", " ").strip()
            if len(stem) < 8:
                continue
            title = stem

        # Narrow the parent search: `section` / outer `div` can wrap the
        # entire page on gov templates, which would let a single "housing"
        # mention in a header tag the whole block and leak unrelated links
        # past matches_housing. Card-sized containers only.
        block = a.find_parent(["article", "li", "tr"]) or a
        block_text = block.get_text(" ", strip=True)

        if apply_keyword_filter and not matches_housing(f"{title} {block_text}"):
            continue

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
    allowed_netloc: str,
    publisher: str,
    category: str,
    apply_keyword_filter: bool,
    client: httpx.Client,
    max_pages: int,
    delay_s: float,
) -> list[Source]:
    listing_path = urlparse(listing_url).path.rstrip("/") or "/"
    all_sources: list[Source] = []
    # Treasury and PC both run on Drupal, whose pager is 0-indexed: the
    # landing URL IS page=0, the second page is ?page=1, etc. So our
    # loop variable maps to page 1→landing, page 2→?page=1, page 3→?page=2.
    for page in range(1, max_pages + 1):
        if page == 1:
            url = listing_url
        else:
            url = f"{listing_url}?page={page - 1}"
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
            allowed_netloc=allowed_netloc,
            publisher=publisher,
            category=category,
            apply_keyword_filter=apply_keyword_filter,
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


def scrape_treasury(*, max_pages: int = 15, delay_s: float = 0.8) -> list[Source]:
    with _client() as client:
        listing_url = _discover(client, TREASURY_BASE, TREASURY_LISTING_CANDIDATES)
        log.info("Treasury housing listing: %s", listing_url)
        # Treasury /policy-topics/housing is already topic-scoped; keep the
        # keyword filter on anyway because the page links out to adjacent
        # non-housing material (tax, welfare).
        sources = _walk_pages(
            listing_url=listing_url,
            allowed_netloc="treasury.gov.au",
            publisher=TREASURY_PUBLISHER,
            category=TREASURY_CATEGORY,
            apply_keyword_filter=True,
            client=client,
            max_pages=max_pages,
            delay_s=delay_s,
        )
    seen: set[str] = set()
    unique: list[Source] = []
    for s in sources:
        if s.url in seen:
            continue
        seen.add(s.url)
        unique.append(s)
    log.info("Total unique Treasury items: %d", len(unique))
    return unique


def scrape_pc(*, max_pages: int = 15, delay_s: float = 0.8) -> list[Source]:
    with _client() as client:
        listing_url = _discover(client, PC_BASE, PC_LISTING_CANDIDATES)
        log.info("PC housing listing: %s", listing_url)
        # If we fell back to the generic /inquiries/completed page, we MUST
        # keyword-filter. If we're on /topics/housing the filter is still
        # cheap insurance.
        sources = _walk_pages(
            listing_url=listing_url,
            allowed_netloc="pc.gov.au",
            publisher=PC_PUBLISHER,
            category=PC_CATEGORY,
            apply_keyword_filter=True,
            client=client,
            max_pages=max_pages,
            delay_s=delay_s,
        )
    seen: set[str] = set()
    unique: list[Source] = []
    for s in sources:
        if s.url in seen:
            continue
        seen.add(s.url)
        unique.append(s)
    log.info("Total unique PC items: %d", len(unique))
    return unique


def scrape(*, max_pages: int = 15, delay_s: float = 0.8) -> list[Source]:
    all_sources: list[Source] = []
    for name, fn in (("treasury", scrape_treasury), ("pc", scrape_pc)):
        try:
            all_sources.extend(fn(max_pages=max_pages, delay_s=delay_s))
        except Exception as e:  # noqa: BLE001 — sub-scrapers fail independently
            log.exception("Sub-scraper %s failed: %s", name, e)
    seen: set[str] = set()
    unique: list[Source] = []
    for s in all_sources:
        if s.url in seen:
            continue
        seen.add(s.url)
        unique.append(s)
    log.info("Total unique Treasury+PC items: %d", len(unique))
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "which",
        nargs="?",
        choices=("treasury", "pc", "all"),
        default="all",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/treasury_pc.jsonl"),
    )
    parser.add_argument("--max-pages", type=int, default=15)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if args.which == "treasury":
        sources = scrape_treasury(max_pages=args.max_pages)
    elif args.which == "pc":
        sources = scrape_pc(max_pages=args.max_pages)
    else:
        sources = scrape(max_pages=args.max_pages)
    write_jsonl(sources, args.out)
    log.info("Wrote %d sources → %s", len(sources), args.out)


if __name__ == "__main__":
    main()
