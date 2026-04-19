"""Scrapers for RBA research outputs: RDPs, Bulletin, and FSR.

All three share the same RBA web host (`rba.gov.au`) and are filtered by
housing keywords on either the article title or the enclosing section name.

Entrypoints:

    python -m src.ingest.scrapers.rba rdp       --out data/raw/rba_rdp.jsonl
    python -m src.ingest.scrapers.rba bulletin  --out data/raw/rba_bulletin.jsonl
    python -m src.ingest.scrapers.rba fsr       --out data/raw/rba_fsr.jsonl
    python -m src.ingest.scrapers.rba all       --out data/raw/rba.jsonl
"""
from __future__ import annotations

import argparse
import logging
import re
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

RBA_BASE = "https://www.rba.gov.au"
RDP_INDEX_URL = f"{RBA_BASE}/publications/rdp/"
BULLETIN_INDEX_URL = f"{RBA_BASE}/publications/bulletin/"
FSR_INDEX_URL = f"{RBA_BASE}/publications/fsr/"
PUBLISHER = "RBA"


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
    )


def fetch(url: str, *, client: httpx.Client | None = None) -> str:
    owns_client = client is None
    if client is None:
        client = _client()
    try:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text
    finally:
        if owns_client:
            client.close()


def _dedupe(sources: list[Source]) -> list[Source]:
    seen: set[str] = set()
    out: list[Source] = []
    for s in sources:
        if s.url in seen:
            continue
        seen.add(s.url)
        out.append(s)
    return out


# ----------------------------- RDP --------------------------------------

def parse_rdp_index(html: str, *, base_url: str = RDP_INDEX_URL) -> list[Source]:
    """Parse the RDP index, which groups papers by year heading."""
    soup = BeautifulSoup(html, "lxml")
    sources: list[Source] = []
    current_year: str | None = None

    for node in soup.find_all(["h2", "h3", "li"]):
        if node.name in {"h2", "h3"}:
            heading = node.get_text(strip=True)
            if heading.isdigit() and 1990 <= int(heading) <= 2100:
                current_year = heading
            continue

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
    return _dedupe(sources)


def scrape_rdp(*, index_url: str = RDP_INDEX_URL) -> list[Source]:
    log.info("Fetching RDP index: %s", index_url)
    html = fetch(index_url)
    parsed = parse_rdp_index(html, base_url=index_url)
    log.info("Parsed %d RDPs; filtering by housing keywords", len(parsed))
    housing = [s for s in parsed if matches_housing(s.title)]
    log.info("Kept %d housing-relevant RDPs", len(housing))
    return housing


# --------------------------- Bulletin ------------------------------------

# e.g. /publications/bulletin/2024/sep/some-article.html
_BULLETIN_ARTICLE_RE = re.compile(r"/publications/bulletin/\d{4}/[a-z]{3}/[^/]+\.html$")


def parse_bulletin_index(
    html: str, *, base_url: str = BULLETIN_INDEX_URL
) -> list[tuple[str, str | None]]:
    """Return (issue_url, label) pairs for Bulletin issues on the index page."""
    soup = BeautifulSoup(html, "lxml")
    issues: list[tuple[str, str | None]] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        url = urljoin(base_url, href)
        path = urlparse(url).path
        if re.fullmatch(r"/publications/bulletin/\d{4}/[a-z]{3}/?", path):
            issues.append((url.rstrip("/") + "/", a.get_text(strip=True) or None))
    seen: set[str] = set()
    out: list[tuple[str, str | None]] = []
    for url, label in issues:
        if url in seen:
            continue
        seen.add(url)
        out.append((url, label))
    return out


def parse_bulletin_issue(html: str, issue_url: str) -> list[Source]:
    """Extract article links from a Bulletin issue page."""
    soup = BeautifulSoup(html, "lxml")
    date = _date_from_bulletin_path(issue_url)
    sources: list[Source] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        url = urljoin(issue_url, href)
        path = urlparse(url).path
        if not _BULLETIN_ARTICLE_RE.search(path):
            continue
        title = a.get_text(strip=True)
        if not title:
            continue
        sources.append(
            Source(
                title=title,
                publisher=PUBLISHER,
                url=url,
                date=date,
                category="bulletin-article",
            )
        )
    return _dedupe(sources)


def _date_from_bulletin_path(url: str) -> str | None:
    # /publications/bulletin/{YYYY}/{mon}/...
    m = re.search(r"/bulletin/(\d{4})/([a-z]{3})", url)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def scrape_bulletin(*, index_url: str = BULLETIN_INDEX_URL) -> list[Source]:
    log.info("Fetching Bulletin index: %s", index_url)
    with _client() as client:
        index_html = fetch(index_url, client=client)
        issues = parse_bulletin_index(index_html, base_url=index_url)
        log.info("Discovered %d Bulletin issues", len(issues))

        all_articles: list[Source] = []
        for issue_url, _label in issues:
            try:
                html = fetch(issue_url, client=client)
            except httpx.HTTPError as e:
                log.warning("Skipping %s: %s", issue_url, e)
                continue
            all_articles.extend(parse_bulletin_issue(html, issue_url=issue_url))

    deduped = _dedupe(all_articles)
    log.info("Parsed %d Bulletin articles; filtering", len(deduped))
    housing = [s for s in deduped if matches_housing(s.title)]
    log.info("Kept %d housing-relevant Bulletin articles", len(housing))
    return housing


# ----------------------------- FSR --------------------------------------

# FSR is biannual (Apr/Oct). Issue pages live at /publications/fsr/{YYYY}/{mon}/
_FSR_ISSUE_RE = re.compile(r"/publications/fsr/\d{4}/[a-z]{3}/?$")


def parse_fsr_index(
    html: str, *, base_url: str = FSR_INDEX_URL
) -> list[tuple[str, str | None]]:
    soup = BeautifulSoup(html, "lxml")
    issues: list[tuple[str, str | None]] = []
    for a in soup.find_all("a", href=True):
        url = urljoin(base_url, a["href"])
        path = urlparse(url).path
        if _FSR_ISSUE_RE.search(path):
            issues.append((url.rstrip("/") + "/", a.get_text(strip=True) or None))
    seen: set[str] = set()
    out: list[tuple[str, str | None]] = []
    for url, label in issues:
        if url in seen:
            continue
        seen.add(url)
        out.append((url, label))
    return out


def parse_fsr_issue(html: str, issue_url: str) -> list[Source]:
    """Grab PDF links (full review + chapters) from an FSR issue page."""
    soup = BeautifulSoup(html, "lxml")
    date = _date_from_fsr_path(issue_url)
    sources: list[Source] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        url = urljoin(issue_url, href)
        path = urlparse(url).path
        if not path.lower().endswith(".pdf"):
            continue
        if "/fsr/" not in path:
            continue
        title = a.get_text(strip=True) or path.rsplit("/", 1)[-1]
        sources.append(
            Source(
                title=title,
                publisher=PUBLISHER,
                url=url,
                date=date,
                category="financial-stability-review",
            )
        )
    return _dedupe(sources)


def _date_from_fsr_path(url: str) -> str | None:
    m = re.search(r"/fsr/(\d{4})/([a-z]{3})", url)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def scrape_fsr(*, index_url: str = FSR_INDEX_URL) -> list[Source]:
    log.info("Fetching FSR index: %s", index_url)
    with _client() as client:
        index_html = fetch(index_url, client=client)
        issues = parse_fsr_index(index_html, base_url=index_url)
        log.info("Discovered %d FSR issues", len(issues))

        all_pdfs: list[Source] = []
        for issue_url, _label in issues:
            try:
                html = fetch(issue_url, client=client)
            except httpx.HTTPError as e:
                log.warning("Skipping %s: %s", issue_url, e)
                continue
            all_pdfs.extend(parse_fsr_issue(html, issue_url=issue_url))

    # FSR is wholly housing-relevant for our corpus — keep the whole review
    # when titled ambiguously, but also promote chapter-level matches.
    return _dedupe(all_pdfs)


# --------------------------- CLI ----------------------------------------

_SCRAPERS = {
    "rdp": scrape_rdp,
    "bulletin": scrape_bulletin,
    "fsr": scrape_fsr,
}


def scrape_all() -> list[Source]:
    out: list[Source] = []
    for name, fn in _SCRAPERS.items():
        log.info("=== %s ===", name)
        try:
            out.extend(fn())
        except httpx.HTTPError as e:
            log.error("Scraper %s failed: %s", name, e)
    return _dedupe(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "which",
        choices=[*_SCRAPERS.keys(), "all"],
        help="Which RBA scraper to run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/rba.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if args.which == "all":
        sources = scrape_all()
    else:
        sources = _SCRAPERS[args.which]()
    write_jsonl(sources, args.out)
    log.info("Wrote %d sources → %s", len(sources), args.out)


if __name__ == "__main__":
    main()
