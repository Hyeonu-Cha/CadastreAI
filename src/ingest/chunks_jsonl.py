"""Emit the canonical `data/processed/chunks.jsonl` for indexing.

Joins the chunker's output (`src.ingest.chunk.chunk_file`) with source
metadata from `data/raw/sources.jsonl`. Reuses the downloader's
`_target_stem` + `_compute_unique_stems` to map each `.md` file back
to the originating `DownloadTask` — that's the only reliable mapping
because the collision guard mangles some stems with an 8-char MD5 of
the URL.

Output schema (per line):
    {
      "chunk_id": "{publisher}/{stem}__{idx:04d}",
      "publisher": "...",
      "title": "...",
      "date": "2024-oct" | null,
      "url": "https://...",
      "source": "ahuri/2012-00_foo.md",
      "section_heading": "Chapter 3 > 3.2 Methodology" | "",
      "page": null,
      "text": "...",
      "token_count": 612
    }

`page` is always null in this pass — pymupdf4llm's `to_markdown` was
run without `page_chunks=True`, so page boundaries weren't recorded.
Re-parsing with page markers is a follow-up if citation UX needs it.

    python -m src.ingest.chunks_jsonl \\
        --processed-dir data/processed --raw-dir data/raw \\
        --sources data/raw/sources.jsonl --out data/processed/chunks.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from src.ingest.chunk import Chunk, chunk_file
from src.ingest.download import DownloadTask, _compute_unique_stems, _read_sources

log = logging.getLogger(__name__)

_SKIP_DIRS = {"eval", "training"}


@dataclass
class ChunkRecord:
    chunk_id: str
    publisher: str
    title: str
    date: str | None
    url: str
    source: str
    section_heading: str
    page: int | None
    text: str
    token_count: int


def _build_stem_index(
    tasks: list[DownloadTask], raw_dir: Path
) -> dict[str, DownloadTask]:
    """Map `{publisher_slug}/{stem_name}` → DownloadTask.

    The key uses forward slashes and no extension so it aligns with the
    chunker's `source` field (which is `str(relative_path)` with
    backslashes normalized).
    """
    stems = _compute_unique_stems(tasks, raw_dir)
    index: dict[str, DownloadTask] = {}
    for task, stem in zip(tasks, stems, strict=True):
        rel = stem.relative_to(raw_dir)
        key = str(rel).replace("\\", "/")
        index[key] = task
    return index


def _source_to_key(source: str) -> str:
    """Strip the .md extension from a chunker `source` so it matches stem keys."""
    if source.endswith(".md"):
        return source[:-3]
    return source


def _discover_md(processed_dir: Path) -> list[Path]:
    return [
        p
        for p in sorted(processed_dir.rglob("*.md"))
        if not any(part in _SKIP_DIRS for part in p.relative_to(processed_dir).parts)
    ]


def emit(
    *,
    processed_dir: Path,
    raw_dir: Path,
    sources_path: Path,
    out_path: Path,
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap: int = 100,
) -> tuple[int, int, int]:
    tasks = _read_sources(sources_path)
    stem_index = _build_stem_index(tasks, raw_dir)
    log.info("Loaded %d sources → %d unique stems", len(tasks), len(stem_index))

    md_files = _discover_md(processed_dir)
    log.info("Discovered %d markdown files under %s", len(md_files), processed_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    n_unmatched = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx_file, path in enumerate(md_files, start=1):
            source = str(path.relative_to(processed_dir)).replace("\\", "/")
            key = _source_to_key(source)
            task = stem_index.get(key)
            if task is None:
                n_unmatched += 1
                log.warning("no source record for %s — skipping", source)
                continue

            chunks: list[Chunk] = chunk_file(
                path,
                processed_dir,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                overlap=overlap,
            )
            for chunk_idx, c in enumerate(chunks):
                rec = ChunkRecord(
                    chunk_id=f"{key}__{chunk_idx:04d}",
                    publisher=task.publisher,
                    title=task.title,
                    date=task.date,
                    url=task.url,
                    source=source,
                    section_heading=" > ".join(c.heading_path),
                    page=None,
                    text=c.text,
                    token_count=c.token_count,
                )
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            n_chunks += len(chunks)
            if idx_file % 100 == 0 or idx_file == len(md_files):
                log.info(
                    "progress %d/%d  chunks=%d unmatched=%d",
                    idx_file, len(md_files), n_chunks, n_unmatched,
                )

    return len(md_files), n_chunks, n_unmatched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--sources", type=Path, default=Path("data/raw/sources.jsonl"))
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/chunks.jsonl")
    )
    parser.add_argument("--min-tokens", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    n_files, n_chunks, n_unmatched = emit(
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        sources_path=args.sources,
        out_path=args.out,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
    )
    log.info(
        "Done. files=%d chunks=%d unmatched=%d → %s",
        n_files, n_chunks, n_unmatched, args.out,
    )
    print(
        f"files={n_files} chunks={n_chunks} unmatched={n_unmatched} out={args.out}"
    )


if __name__ == "__main__":
    main()
