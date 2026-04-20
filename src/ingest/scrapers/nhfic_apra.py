"""Scrapers for NHFIC / Housing Australia and APRA housing-relevant pages.

Two P2 publishers that round out the corpus with mortgage-market and
guarantee-scheme perspectives:

- Housing Australia (formerly NHFIC, renamed in 2023):
  https://www.housingaustralia.gov.au/research and
  https://www.housingaustralia.gov.au/publications — publishes the annual
  "State of the Nation's Housing" report plus Home Guarantee Scheme stats.
  The agency is wholly housing-focused so no keyword filter is applied.
- APRA: https://www.apra.gov.au/publications — publishes banking stats
  including residential mortgage lending. APRA covers a lot of non-
  housing material (insurance, superannuation) so the `matches_housing`
  keyword filter IS applied here.

    python -m src.ingest.scrapers.nhfic_apra --out data/raw/nhfic_apra.jsonl
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

HA_BASE = "https://www.housingaustralia.gov.au"
HA_LISTING_CANDIDATES = (
    "/research",
    "/research/",
    "/publications",
    "/publications/",
    "/resources",
)
HA_PUBLISHER = "Housing Australia (NHFIC)"
HA_CATEGORY = "government"

APRA_BASE = "https://www.apra.gov.au"
# APRA's public /publications page redirects to /news-and-publications,
# which is a category hub rendered server-side with only 8 top-level
# category tiles — walking the listing yields zero individual items.
# The live sitemap at /sitemap.xml is a sitemapindex that points at two
# paginated sub-sitemaps; those list every public page on the site, and
# we keyword-filter the slugs for housing/mortgage/lending relevance.
APRA_SITEMAP_URLS = (
    f"{APRA_BASE}/sitemap.xml?page=1",
    f"{APRA_BASE}/sitemap.xml?page=2",
)
APRA_PUBLISHER = "APRA"
APRA_CATEGORY = "regulatory-statistics"

_DATE_RE = re.compile(
    r"(\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},\s*\d{4}|Q[1-4]\s*\d{4}|\b(19|20)\d{2}\b)"
)

_SKIP_PATH_TOKENS = (
    "/contact",
    "/about",
    "/careers",
    "/privacy",
    "/disclaimer",
    "/copyright",
    "/accessibility",
    "/sitemap",
    "/search",
    "/news",
    "/media",
    "/subscribe",
    "/newsletter",
    "/login",
)

_ALLOWED_EXT = {".pdf", ".html", ".htm", ".xlsx", ""}


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

        # Narrow card containers only; avoid div/section which can wrap
        # large page regions and let a stray "housing" mention taint
        # unrelated links through the keyword filter.
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
    # Both APRA and Housing Australia appear to run on Drupal-family CMSes
    # with 0-indexed pagers. Page 1 → landing URL, page N≥2 → ?page=N-1.
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
            # Require at least 2 new URLs to continue: a persistent "Recent
            # Publications" sidebar could rotate one or two items per page
            # even after the main listing is exhausted, which would otherwise
            # drag us all the way to max_pages.
            if len(new_urls) < 2:
                log.info(
                    "  only %d new URL(s) on page %d; assuming pager exhausted",
                    len(new_urls),
                    page,
                )
                if new_urls:
                    page_sources = [s for s in page_sources if s.url in new_urls]
                    all_sources.extend(page_sources)
                break
        all_sources.extend(page_sources)
        time.sleep(delay_s)
    return all_sources


def scrape_housing_australia(*, max_pages: int = 15, delay_s: float = 0.8) -> list[Source]:
    with _client() as client:
        listing_url = _discover(client, HA_BASE, HA_LISTING_CANDIDATES)
        log.info("Housing Australia listing: %s", listing_url)
        sources = _walk_pages(
            listing_url=listing_url,
            allowed_netloc="housingaustralia.gov.au",
            publisher=HA_PUBLISHER,
            category=HA_CATEGORY,
            apply_keyword_filter=False,  # wholly housing agency
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
    log.info("Total unique Housing Australia items: %d", len(unique))
    return unique


_SITEMAP_LOC_RE = re.compile(r"<loc>([^<]+)</loc>")


def _slug_to_title(path: str) -> str:
    tail = path.rstrip("/").rsplit("/", 1)[-1]
    return tail.replace("-", " ").replace("_", " ").strip().title()


def scrape_apra(*, max_pages: int = 15, delay_s: float = 0.8) -> list[Source]:
    """Discover APRA housing-adjacent pages via the internal sitemap.

    max_pages / delay_s are accepted for signature compatibility; this
    function only issues two HTTP requests (one per sitemap chunk).
    """
    sources: list[Source] = []
    seen: set[str] = set()
    with _client() as client:
        for sitemap_url in APRA_SITEMAP_URLS:
            log.info("Fetching APRA sitemap %s", sitemap_url)
            try:
                resp = client.get(sitemap_url)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                log.warning("  failed: %s", e)
                continue
            urls = _SITEMAP_LOC_RE.findall(resp.text)
            log.info("  %d URLs", len(urls))
            for url in urls:
                # Sitemap entries sometimes return the internal Drupal
                # preview host (prod.apra.shared.skpr.live) instead of the
                # public domain; pin the netloc to the public host so
                # downloader URL validation doesn't choke.
                parsed = urlparse(url)
                if parsed.netloc and parsed.netloc != "www.apra.gov.au":
                    parsed = parsed._replace(netloc="www.apra.gov.au")
                public = parsed.geturl()
                if "apra.gov.au" not in parsed.netloc:
                    continue
                path = parsed.path
                if not path or path == "/":
                    continue
                title = _slug_to_title(path)
                if len(title) < 6:
                    continue
                if not matches_housing(title):
                    continue
                if public in seen:
                    continue
                seen.add(public)
                sources.append(
                    Source(
                        title=title,
                        publisher=APRA_PUBLISHER,
                        url=public,
                        date=None,
                        category=APRA_CATEGORY,
                    )
                )
    log.info("Total unique APRA items: %d", len(sources))
    return sources


def scrape(*, max_pages: int = 15, delay_s: float = 0.8) -> list[Source]:
    all_sources: list[Source] = []
    for name, fn in (("housing_australia", scrape_housing_australia), ("apra", scrape_apra)):
        try:
            all_sources.extend(fn(max_pages=max_pages, delay_s=delay_s))
        except Exception as e:  # noqa: BLE001
            log.exception("Sub-scraper %s failed: %s", name, e)
    seen: set[str] = set()
    unique: list[Source] = []
    for s in all_sources:
        if s.url in seen:
            continue
        seen.add(s.url)
        unique.append(s)
    log.info("Total unique NHFIC/APRA items: %d", len(unique))
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "which",
        nargs="?",
        choices=("housing_australia", "apra", "all"),
        default="all",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/nhfic_apra.jsonl"),
    )
    parser.add_argument("--max-pages", type=int, default=15)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if args.which == "housing_australia":
        sources = scrape_housing_australia(max_pages=args.max_pages)
    elif args.which == "apra":
        sources = scrape_apra(max_pages=args.max_pages)
    else:
        sources = scrape(max_pages=args.max_pages)
    write_jsonl(sources, args.out)
    log.info("Wrote %d sources → %s", len(sources), args.out)


if __name__ == "__main__":
    main()
