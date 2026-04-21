"""Polite concurrent downloader for entries in `data/raw/sources.jsonl`.

Reads a sources.jsonl file and downloads each entry's document to
`data/raw/{publisher_slug}/{yyyy-mm}_{slug}.{pdf,html}`. Three resolution
paths:

1. Direct PDF URL (or response content-type is PDF) → save as `.pdf`.
2. HTML landing page with a PDF link in-body → follow link, save PDF.
3. HTML page with no PDF link → save the HTML body as `.html`. Many
   PropTrack / CoreLogic / Domain articles are pure HTML (no companion
   PDF); dropping them as "no pdf link found" was shrinking the
   investor/homebuyer corpus by ~20%. docling parses HTML too, so
   downstream chunking handles both.

Politeness features:

- **Global concurrency limit:** at most `--concurrency` downloads in
  flight across all hosts (default 4).
- **Per-host serialization:** each host gets its own `asyncio.Lock` plus
  a minimum inter-request delay (`--per-host-delay-s`, default 1.0s) so
  we never slam a single origin.
- **Retries:** HTTP errors retry with exponential backoff up to
  `--max-retries` (default 3). 404s don't retry.
- **Resume on failure:** if the target path already exists and is
  non-empty, the file is skipped. Partial downloads go to `.part` files
  and are atomically renamed on success.
- **Report:** emits `sources.download.json` next to the input jsonl with
  {ok, skipped, failed} tallies and the list of failing URLs.

    python -m src.ingest.download --in data/raw/sources.jsonl --out-dir data/raw
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from src.ingest.scrapers.common import USER_AGENT, slugify

log = logging.getLogger(__name__)


@dataclass
class DownloadTask:
    title: str
    publisher: str
    url: str
    date: str | None
    category: str | None


@dataclass
class DownloadResult:
    task: DownloadTask
    status: str  # "ok" | "skipped" | "failed"
    path: Path | None = None
    error: str | None = None


def _read_sources(path: Path) -> list[DownloadTask]:
    tasks: list[DownloadTask] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                log.warning("sources.jsonl line %d: %s", line_no, e)
                continue
            tasks.append(
                DownloadTask(
                    title=obj["title"],
                    publisher=obj["publisher"],
                    url=obj["url"],
                    date=obj.get("date"),
                    category=obj.get("category"),
                )
            )
    return tasks


def _date_prefix(date: str | None) -> str:
    """Best-effort YYYY-MM extraction for filename prefixes.

    Handles:
      - `YYYY` → `YYYY-00`
      - `YYYY-MM` / `YYYY-MM-DD` → `YYYY-MM`
      - `Mon YYYY`, `Month YYYY`, `DD Month YYYY` → `YYYY-MM`
      - Anything unparsable → `unknown`
    """
    if not date:
        return "unknown"
    import re

    date = date.strip()
    m = re.match(r"^(\d{4})-(\d{1,2})", date)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    m = re.match(r"^(\d{4})$", date)
    if m:
        return f"{m.group(1)}-00"
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    m = re.search(r"([A-Za-z]{3,})\s+(\d{4})", date)
    if m:
        mon = months.get(m.group(1)[:3].lower())
        year = m.group(2)
        if mon:
            return f"{year}-{mon:02d}"
    m = re.search(r"\b(19|20)(\d{2})\b", date)
    if m:
        return f"{m.group(1)}{m.group(2)}-00"
    return "unknown"


def _target_stem(task: DownloadTask, out_dir: Path) -> Path:
    """Return the target path without an extension.

    The extension is chosen at download time: `.pdf` when a PDF is
    retrieved, `.html` when we fall back to saving the landing page
    body. Callers should `.with_suffix(".pdf" | ".html")` before touching
    the filesystem.
    """
    pub_slug = slugify(task.publisher, max_len=40) or "unknown"
    prefix = _date_prefix(task.date)
    slug = slugify(task.title, max_len=80) or "untitled"
    return out_dir / pub_slug / f"{prefix}_{slug}"


class _HostGate:
    """Per-host serialization + min-delay enforcement.

    Used as an async context manager so every gated request is
    guaranteed to release the host lock even on exceptions:

        async with gate.at(url):
            ...request...
    """

    def __init__(self, min_delay_s: float) -> None:
        self.min_delay_s = min_delay_s
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._last_hit: dict[str, float] = {}

    @asynccontextmanager
    async def at(self, url: str) -> AsyncIterator[None]:
        host = urlparse(url).netloc
        lock = self._locks[host]
        async with lock:
            now = time.monotonic()
            last = self._last_hit.get(host, 0.0)
            sleep_for = self.min_delay_s - (now - last)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            try:
                yield
            finally:
                self._last_hit[host] = time.monotonic()


async def _stream_response_to_file(
    resp: httpx.Response, *, target: Path
) -> None:
    """Pipe an already-opened streaming response body to `target`.

    File writes go through `asyncio.to_thread` so they don't block the
    event loop. Uses a `.part` sidecar renamed atomically on success.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    fh = await asyncio.to_thread(part.open, "wb")
    try:
        async for chunk in resp.aiter_bytes(chunk_size=65536):
            if chunk:
                await asyncio.to_thread(fh.write, chunk)
    finally:
        await asyncio.to_thread(fh.close)
    await asyncio.to_thread(part.replace, target)


_MIN_HTML_BYTES = 1024


async def _fetch_and_save(
    task: DownloadTask,
    *,
    stem: Path,
    client: httpx.AsyncClient,
    gate: _HostGate,
) -> tuple[Path | None, str | None]:
    """Resolve and download the document for `task`.

    Returns (saved_path, error). On success, `saved_path` is the actual
    file written (suffix `.pdf` or `.html`) and `error` is None. On
    failure, `saved_path` is None and `error` is a short string.

    Each HTTP request — including the second request when we follow a
    landing page to its PDF — passes through the host gate, so the
    per-host delay is always respected. The initial request is opened
    as a stream so a direct-PDF URL is piped to disk without buffering
    the whole body in memory.
    """
    url = task.url
    async with gate.at(url):
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").lower()
            if "pdf" in content_type or url.lower().endswith(".pdf"):
                target = stem.with_suffix(".pdf")
                await _stream_response_to_file(resp, target=target)
                return target, None
            # HTML landing page — read body now (small) so we can close
            # this response before making the follow-up request.
            html = (await resp.aread()).decode(resp.encoding or "utf-8", errors="replace")

    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if href.lower().endswith(".pdf"):
            pdf_url = urljoin(url, href)
            async with gate.at(pdf_url):
                async with client.stream("GET", pdf_url) as pdf_resp:
                    pdf_resp.raise_for_status()
                    target = stem.with_suffix(".pdf")
                    await _stream_response_to_file(pdf_resp, target=target)
            return target, None

    # No PDF link in the page — save the HTML body so docling can parse
    # the article directly. Guard against tiny "empty shell" pages
    # (login redirects, JS-only containers) that would produce zero
    # useful markdown downstream.
    if len(html.encode("utf-8", errors="replace")) < _MIN_HTML_BYTES:
        return None, "no pdf link found; html body too small to save"
    target = stem.with_suffix(".html")
    target.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(target.write_text, html, "utf-8")
    return target, None


async def _download_one(
    task: DownloadTask,
    *,
    out_dir: Path,
    client: httpx.AsyncClient,
    gate: _HostGate,
    sem: asyncio.Semaphore,
    max_retries: int,
    backoff_s: float,
) -> DownloadResult:
    stem = _target_stem(task, out_dir)
    for suffix in (".pdf", ".html"):
        existing = stem.with_suffix(suffix)
        if existing.exists() and existing.stat().st_size > 0:
            return DownloadResult(task=task, status="skipped", path=existing)

    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        async with sem:
            saved: Path | None = None
            try:
                saved, err = await _fetch_and_save(
                    task, stem=stem, client=client, gate=gate
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return DownloadResult(task=task, status="failed", error="404")
                last_error = f"http {e.response.status_code}"
                err = last_error
            except httpx.HTTPError as e:
                last_error = repr(e)
                err = last_error
            except (ValueError, OSError) as e:
                # Malformed URL (e.g. a scraped href that turned out to
                # be a local filesystem path), bad unicode, path-too-long
                # on Windows, etc. — bail on this task without retry so a
                # single poisoned entry can't crash the whole batch.
                return DownloadResult(task=task, status="failed", error=repr(e))

        if err is None and saved is not None:
            return DownloadResult(task=task, status="ok", path=saved)

        last_error = err or last_error
        if attempt < max_retries:
            await asyncio.sleep(
                backoff_s * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
            )

    return DownloadResult(
        task=task, status="failed", error=last_error or "unknown"
    )


async def _run(
    tasks: list[DownloadTask],
    *,
    out_dir: Path,
    concurrency: int,
    per_host_delay_s: float,
    max_retries: int,
    backoff_s: float,
    timeout_s: float,
) -> list[DownloadResult]:
    sem = asyncio.Semaphore(concurrency)
    gate = _HostGate(min_delay_s=per_host_delay_s)
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(
        headers=headers,
        timeout=timeout_s,
        follow_redirects=True,
    ) as client:
        coros = [
            _download_one(
                t,
                out_dir=out_dir,
                client=client,
                gate=gate,
                sem=sem,
                max_retries=max_retries,
                backoff_s=backoff_s,
            )
            for t in tasks
        ]
        results: list[DownloadResult] = []
        for idx, fut in enumerate(asyncio.as_completed(coros), start=1):
            res = await fut
            results.append(res)
            if idx % 10 == 0 or idx == len(coros):
                ok = sum(r.status == "ok" for r in results)
                skipped = sum(r.status == "skipped" for r in results)
                failed = sum(r.status == "failed" for r in results)
                log.info(
                    "progress %d/%d  ok=%d skipped=%d failed=%d",
                    idx, len(coros), ok, skipped, failed,
                )
        return results


def _write_report(results: list[DownloadResult], path: Path) -> None:
    ok = [r for r in results if r.status == "ok"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]
    report = {
        "total": len(results),
        "ok": len(ok),
        "skipped": len(skipped),
        "failed": len(failed),
        "failures": [
            {"url": r.task.url, "title": r.task.title, "error": r.error}
            for r in failed
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, default=Path("data/raw/sources.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--per-host-delay-s", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--backoff-s", type=float, default=1.5)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--only-publisher",
        help="Filter to one publisher (matches Source.publisher exactly)",
    )
    parser.add_argument("--limit", type=int, help="Only download N files (for smoke tests)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    tasks = _read_sources(args.in_path)
    if args.only_publisher:
        tasks = [t for t in tasks if t.publisher == args.only_publisher]
    if args.limit:
        tasks = tasks[: args.limit]
    log.info("Loaded %d tasks from %s", len(tasks), args.in_path)

    results = asyncio.run(
        _run(
            tasks,
            out_dir=args.out_dir,
            concurrency=args.concurrency,
            per_host_delay_s=args.per_host_delay_s,
            max_retries=args.max_retries,
            backoff_s=args.backoff_s,
            timeout_s=args.timeout_s,
        )
    )

    report_path = args.in_path.with_suffix(".download.json")
    _write_report(results, report_path)
    ok = sum(r.status == "ok" for r in results)
    skipped = sum(r.status == "skipped" for r in results)
    failed = sum(r.status == "failed" for r in results)
    log.info(
        "Done. ok=%d skipped=%d failed=%d. Report: %s",
        ok, skipped, failed, report_path,
    )
    print(f"ok={ok} skipped={skipped} failed={failed} report={report_path}")


if __name__ == "__main__":
    main()
