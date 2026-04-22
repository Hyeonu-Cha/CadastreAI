"""Bulk-parse raw docs to markdown using `pymupdf4llm` + BeautifulSoup.

This is a lighter-weight sibling to `src.ingest.parse` (docling). Use it
when docling is unavailable or unstable: docling leaks memory on PDFs
where its OCR preprocessing hits `std::bad_alloc` on many pages, so on
machines where that happens it can't complete a full corpus parse. The
pymupdf4llm path doesn't touch torch/onnx, so no leak, no OOM.

Trade-off: pymupdf4llm produces lower-fidelity markdown than docling —
tables often render as plain paragraphs, figure captions may be
dropped. But all useful text is recovered, which is what a RAG corpus
needs.

HTML handling uses BeautifulSoup to strip nav/header/footer/script/
style elements, then emits markdown from the remaining structural
elements (h1-h6, p, li, blockquote, pre, table) with light formatting.
Not as good as docling's HTML path, but adequate for the article-style
pages we scrape.

    python -m src.ingest.parse_bulk --raw-dir data/raw --out-dir data/processed
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_SKIP_DIRS = {"processed", "eval", "training"}


@dataclass
class ParseFailure:
    pdf_path: str
    publisher: str
    error: str


def _discover_docs(raw_dir: Path) -> list[Path]:
    docs: list[Path] = []
    for pub_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        if pub_dir.name in _SKIP_DIRS:
            continue
        for pattern in ("*.pdf", "*.html"):
            docs.extend(sorted(pub_dir.glob(pattern)))
    return docs


def _target_md(doc: Path, raw_dir: Path, out_dir: Path) -> Path:
    rel = doc.relative_to(raw_dir)
    return (out_dir / rel).with_suffix(".md")


def convert_pdf(pdf: Path) -> str:
    import pymupdf4llm

    return pymupdf4llm.to_markdown(str(pdf))


_BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre")


def convert_html(html_path: Path) -> str:
    """Minimal HTML→markdown: strip chrome, keep structural blocks."""
    from bs4 import BeautifulSoup

    raw = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "form", "aside", "noscript"]):
        tag.decompose()

    body = soup.body or soup

    lines: list[str] = []
    title = soup.find("title")
    if title and title.text.strip():
        lines.append(f"# {title.text.strip()}")
        lines.append("")

    for el in body.find_all(_BLOCK_TAGS):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        tag = el.name
        if tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
            depth = int(tag[1])
            lines.append("#" * depth + " " + text)
        elif tag == "li":
            lines.append("- " + text)
        elif tag == "blockquote":
            lines.append("> " + text)
        elif tag == "pre":
            lines.append("```")
            lines.append(text)
            lines.append("```")
        else:  # p
            lines.append(text)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def parse_all(
    *,
    raw_dir: Path,
    out_dir: Path,
    limit: int | None = None,
    force: bool = False,
) -> tuple[int, int, list[ParseFailure]]:
    docs = _discover_docs(raw_dir)
    if limit:
        docs = docs[:limit]
    log.info("Discovered %d documents under %s", len(docs), raw_dir)

    ok = 0
    skipped = 0
    failures: list[ParseFailure] = []
    for idx, doc in enumerate(docs, start=1):
        target = _target_md(doc, raw_dir, out_dir)
        if target.exists() and target.stat().st_size > 0 and not force:
            skipped += 1
            if idx % 50 == 0 or idx == len(docs):
                log.info(
                    "progress %d/%d  ok=%d skipped=%d failed=%d",
                    idx, len(docs), ok, skipped, len(failures),
                )
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        try:
            if doc.suffix.lower() == ".pdf":
                md = convert_pdf(doc)
            else:
                md = convert_html(doc)
        except Exception as e:  # noqa: BLE001
            failures.append(
                ParseFailure(
                    pdf_path=str(doc),
                    publisher=doc.parent.name,
                    error=f"{type(e).__name__}: {e}",
                )
            )
            log.warning("FAIL %s — %s", doc.name, e)
        else:
            if not md or not md.strip():
                failures.append(
                    ParseFailure(
                        pdf_path=str(doc),
                        publisher=doc.parent.name,
                        error="empty markdown",
                    )
                )
                log.warning("EMPTY %s", doc.name)
            else:
                target.write_text(md, encoding="utf-8")
                ok += 1
                elapsed = time.monotonic() - start
                log.info(
                    "[%d/%d] %s — %d chars in %.1fs",
                    idx, len(docs), doc.name, len(md), elapsed,
                )

        if idx % 50 == 0 or idx == len(docs):
            log.info(
                "progress %d/%d  ok=%d skipped=%d failed=%d",
                idx, len(docs), ok, skipped, len(failures),
            )

    return ok, skipped, failures


def _write_failures(failures: list[ParseFailure], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for fail in failures:
            f.write(json.dumps(asdict(fail), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    ok, skipped, failures = parse_all(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        limit=args.limit,
        force=args.force,
    )
    failure_path = args.out_dir / "parse.bulk.failures.jsonl"
    _write_failures(failures, failure_path)
    log.info(
        "Done. ok=%d skipped=%d failed=%d. Failures: %s",
        ok, skipped, len(failures), failure_path,
    )
    print(f"ok={ok} skipped={skipped} failed={len(failures)} failures={failure_path}")


if __name__ == "__main__":
    main()
