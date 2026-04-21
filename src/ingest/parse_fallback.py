"""Fallback PDF→markdown conversion using `pymupdf4llm`.

`src.ingest.parse` runs `docling` first (high-fidelity layout / tables /
vision models) but it fails on a non-trivial minority of PDFs — scan-only
documents, encrypted files, malformed streams, or files large enough to
trip docling's std::bad_alloc on Windows. Those failures get logged to
`data/processed/parse.failures.jsonl`.

This module is the second pass: it reads that failures file and retries
each entry with `pymupdf4llm`, which wraps PyMuPDF (MuPDF bindings) with
a GitHub-flavored-markdown exporter. It's dramatically lighter (no
torch, no vision models, no GPU) and tends to succeed on the same docs
that broke docling. The markdown is lower-fidelity (tables often render
as plain paragraphs, figure captions may be dropped), but useful text
content is recoverable.

Failures that persist after pymupdf4llm (truly unreadable files) get
logged to `data/processed/parse.failures.final.jsonl` for manual review.

    python -m src.ingest.parse_fallback \
        --failures data/processed/parse.failures.jsonl \
        --raw-dir data/raw \
        --out-dir data/processed
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ParseFailure:
    pdf_path: str
    publisher: str
    error: str


def _read_failures(path: Path) -> list[ParseFailure]:
    failures: list[ParseFailure] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                log.warning("failures line %d: %s", line_no, e)
                continue
            pdf_path = obj.get("pdf_path")
            if not pdf_path:
                log.warning("failures line %d: missing pdf_path, skipping", line_no)
                continue
            failures.append(
                ParseFailure(
                    pdf_path=pdf_path,
                    publisher=obj.get("publisher", ""),
                    error=obj.get("error", ""),
                )
            )
    return failures


def _target_md(pdf: Path, raw_dir: Path, out_dir: Path) -> Path:
    # data/raw/{publisher}/{stem}.pdf → data/processed/{publisher}/{stem}.md
    # If the PDF isn't under raw_dir (shouldn't happen in practice), fall
    # back to out_dir/<publisher>/<stem>.md.
    try:
        rel = pdf.relative_to(raw_dir)
        return (out_dir / rel).with_suffix(".md")
    except ValueError:
        return out_dir / pdf.parent.name / (pdf.stem + ".md")


def convert_pdf(pdf: Path) -> str:
    """Run pymupdf4llm on a single PDF and return the markdown body."""
    # Lazy import for the same reason as parse.py: pymupdf4llm pulls in
    # PyMuPDF which takes a moment to load.
    import pymupdf4llm

    return pymupdf4llm.to_markdown(str(pdf))


def fallback_all(
    *,
    failures_path: Path,
    raw_dir: Path,
    out_dir: Path,
    limit: int | None = None,
    force: bool = False,
) -> tuple[int, int, list[ParseFailure]]:
    """Return (recovered_count, skipped_count, still_failing)."""
    failures = _read_failures(failures_path)
    if limit:
        failures = failures[:limit]
    log.info("Read %d failures from %s", len(failures), failures_path)

    recovered = 0
    skipped = 0
    still_failing: list[ParseFailure] = []

    for idx, fail in enumerate(failures, start=1):
        pdf = Path(fail.pdf_path)
        if pdf.suffix.lower() != ".pdf":
            # pymupdf4llm is PDF-only. HTML docling failures have no
            # retry path here — pass them through to the final failures
            # file unchanged so they surface in manual review.
            still_failing.append(fail)
            log.info("SKIP %s (not a pdf)", pdf.name)
            continue
        if not pdf.exists():
            still_failing.append(
                ParseFailure(
                    pdf_path=fail.pdf_path,
                    publisher=fail.publisher,
                    error=f"source pdf missing: {pdf}",
                )
            )
            log.warning("MISSING %s", pdf)
            continue

        target = _target_md(pdf, raw_dir, out_dir)
        if target.exists() and target.stat().st_size > 0 and not force:
            skipped += 1
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        try:
            md = convert_pdf(pdf)
        except Exception as e:  # noqa: BLE001 — pymupdf4llm can raise several
            still_failing.append(
                ParseFailure(
                    pdf_path=fail.pdf_path,
                    publisher=fail.publisher,
                    error=f"{type(e).__name__}: {e}",
                )
            )
            log.warning("FAIL %s — %s", pdf.name, e)
            continue

        if not md or not md.strip():
            still_failing.append(
                ParseFailure(
                    pdf_path=fail.pdf_path,
                    publisher=fail.publisher,
                    error="pymupdf4llm returned empty markdown",
                )
            )
            log.warning("EMPTY %s", pdf.name)
            continue

        target.write_text(md, encoding="utf-8")
        recovered += 1
        elapsed = time.monotonic() - start
        log.info(
            "[%d/%d] %s — %d chars in %.1fs",
            idx, len(failures), pdf.name, len(md), elapsed,
        )

    return recovered, skipped, still_failing


def _write_failures(failures: list[ParseFailure], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for fail in failures:
            f.write(json.dumps(asdict(fail), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--failures",
        type=Path,
        default=Path("data/processed/parse.failures.jsonl"),
        help="JSONL of failures produced by src.ingest.parse",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit", type=int, help="Only retry the first N failures")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .md outputs",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if not args.failures.exists():
        log.info("No failures file at %s — nothing to do.", args.failures)
        print(f"recovered=0 skipped=0 still_failing=0 failures={args.failures}")
        return

    recovered, skipped, still_failing = fallback_all(
        failures_path=args.failures,
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        limit=args.limit,
        force=args.force,
    )
    final_path = args.out_dir / "parse.failures.final.jsonl"
    _write_failures(still_failing, final_path)
    log.info(
        "Done. recovered=%d skipped=%d still_failing=%d. Final failures: %s",
        recovered, skipped, len(still_failing), final_path,
    )
    print(
        f"recovered={recovered} skipped={skipped} "
        f"still_failing={len(still_failing)} failures={final_path}"
    )


if __name__ == "__main__":
    main()
