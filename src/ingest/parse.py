"""Convert downloaded PDFs and HTML articles to markdown using docling.

Walks `data/raw/{publisher}/*.{pdf,html}` and writes markdown to
`data/processed/{publisher}/{same-stem}.md`, preserving document
structure — headings (`#` / `##` / `###`), paragraph flow, and tables
(rendered as GitHub-flavored markdown tables by docling's
`export_to_markdown`). docling handles both formats natively via a
single `DocumentConverter` instance.

Failures (scan-only PDFs with no text, encrypted files, docling
exceptions) are logged to `data/processed/parse.failures.jsonl` so the
Task 1.17 `pymupdf4llm` fallback can target them.

    python -m src.ingest.parse --raw-dir data/raw --out-dir data/processed

Tip: `docling` is heavy (pulls torch + several vision models). First
run downloads ~1–2 GB of model weights to your HuggingFace cache;
subsequent runs hit the cache.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Publisher folders created by the downloader. We iterate anything that
# looks like a publisher dir rather than hardcoding names.
_SKIP_DIRS = {"processed", "eval", "training"}


@dataclass
class ParseFailure:
    pdf_path: str
    publisher: str
    error: str


def _discover_docs(raw_dir: Path) -> list[Path]:
    """Return all .pdf and .html files under publisher subdirectories."""
    docs: list[Path] = []
    for pub_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        if pub_dir.name in _SKIP_DIRS:
            continue
        for pattern in ("*.pdf", "*.html"):
            docs.extend(sorted(pub_dir.glob(pattern)))
    return docs


def _target_md(doc: Path, raw_dir: Path, out_dir: Path) -> Path:
    # data/raw/{publisher}/{stem}.{pdf|html} → data/processed/{publisher}/{stem}.md
    rel = doc.relative_to(raw_dir)
    return (out_dir / rel).with_suffix(".md")


def convert_doc(doc: Path, *, converter) -> str:
    """Run docling on a single PDF or HTML file and return markdown."""
    result = converter.convert(str(doc))
    return result.document.export_to_markdown()


def parse_all(
    *,
    raw_dir: Path,
    out_dir: Path,
    limit: int | None = None,
    force: bool = False,
) -> tuple[int, int, list[ParseFailure]]:
    """Return (ok_count, skipped_count, failures)."""
    # Import docling lazily — the import itself takes several seconds
    # the first time and pulls in torch. Keeping it out of module scope
    # lets --help / argument errors run instantly.
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()

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
            if idx % 25 == 0 or idx == len(docs):
                log.info(
                    "progress %d/%d  ok=%d skipped=%d failed=%d",
                    idx, len(docs), ok, skipped, len(failures),
                )
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        try:
            md = convert_doc(doc, converter=converter)
        except Exception as e:  # noqa: BLE001 — docling raises many different errors
            failures.append(
                ParseFailure(
                    pdf_path=str(doc),
                    publisher=doc.parent.name,
                    error=f"{type(e).__name__}: {e}",
                )
            )
            log.warning("FAIL %s — %s", doc.name, e)
        else:
            target.write_text(md, encoding="utf-8")
            ok += 1
            elapsed = time.monotonic() - start
            log.info(
                "[%d/%d] %s — %d chars in %.1fs",
                idx, len(docs), doc.name, len(md), elapsed,
            )

        if idx % 25 == 0 or idx == len(docs):
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
    parser.add_argument("--limit", type=int, help="Process only the first N PDFs (smoke test)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reparse files whose .md already exists",
    )
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
    failure_path = args.out_dir / "parse.failures.jsonl"
    _write_failures(failures, failure_path)
    log.info(
        "Done. ok=%d skipped=%d failed=%d. Failures: %s",
        ok, skipped, len(failures), failure_path,
    )
    print(f"ok={ok} skipped={skipped} failed={len(failures)} failures={failure_path}")


if __name__ == "__main__":
    main()
