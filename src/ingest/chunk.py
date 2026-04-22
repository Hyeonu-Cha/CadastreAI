"""Heading-aware chunker for parsed markdown.

Reads `.md` files produced by `src.ingest.parse_bulk` (or `parse`) and
splits each into 500–800 token chunks with 100-token overlap. The
packing strategy is:

1. Stream paragraphs in document order, tracking the current heading
   stack (`#..######`) as we go. Oversize paragraphs are token-sliced
   to <= `max_tokens`.
2. Pack paragraphs greedily into a buffer. Emit a chunk when:
   - the buffer would exceed `max_tokens` (hard break), OR
   - the heading path changes AND the buffer is >= `min_tokens`
     (soft break — respects document structure where possible).
3. After each emit, prepend the tail of the previous chunk (`overlap`
   tokens) so retrievers see context across boundaries.

Each chunk's `heading_path` is the path of the FIRST paragraph
packed into it. Downstream (Task 1.19) concatenates this with
publisher / date / url metadata before indexing.

Tokenization uses tiktoken `cl100k_base`. BGE embed-time truncation
at 512 tokens is accepted — most chunks fit, longer chunks still
capture their opening content.

    python -m src.ingest.chunk --in data/processed --out /tmp/chunks.jsonl --limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_EMPH_RE = re.compile(r"[*_`]+")
_SKIP_DIRS = {"eval", "training"}
_ENC = None


def _enc():
    global _ENC
    if _ENC is None:
        import tiktoken

        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def _clean_heading(text: str) -> str:
    return _EMPH_RE.sub("", text).strip()


@dataclass
class Chunk:
    text: str
    heading_path: list[str]
    token_count: int
    source: str


def _stream_paragraphs(md: str) -> list[tuple[tuple[str, ...], str]]:
    """Yield (heading_path, paragraph_text) across the whole doc in order."""
    events: list[tuple[tuple[str, ...], str]] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        path = tuple(h[1] for h in heading_stack)
        for para in re.split(r"\n\s*\n", body):
            para = para.strip()
            if para:
                events.append((path, para))

    for line in md.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            text = _clean_heading(m.group(2))
            if not text:
                continue
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
        else:
            buffer.append(line)
    flush()
    return events


def chunk_markdown(
    md: str,
    source: str,
    *,
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    enc = _enc()

    sized: list[tuple[tuple[str, ...], str, int]] = []
    for path, para in _stream_paragraphs(md):
        toks = enc.encode(para)
        if len(toks) > max_tokens:
            for i in range(0, len(toks), max_tokens):
                piece = enc.decode(toks[i : i + max_tokens])
                sized.append((path, piece, len(enc.encode(piece))))
        else:
            sized.append((path, para, len(toks)))

    chunks: list[Chunk] = []
    buf_texts: list[str] = []
    buf_tok = 0
    buf_path: tuple[str, ...] | None = None

    def emit() -> str | None:
        nonlocal buf_texts, buf_tok, buf_path
        if not buf_texts:
            return None
        text = "\n\n".join(buf_texts)
        chunks.append(
            Chunk(
                text=text,
                heading_path=list(buf_path or ()),
                token_count=len(enc.encode(text)),
                source=source,
            )
        )
        prev = text
        buf_texts = []
        buf_tok = 0
        buf_path = None
        return prev

    for path, text, ntok in sized:
        heading_break = (
            buf_path is not None and path != buf_path and buf_tok >= min_tokens
        )
        over_max = bool(buf_texts) and (buf_tok + ntok > max_tokens)
        if heading_break or over_max:
            prev = emit()
            if prev and overlap > 0:
                prev_toks = enc.encode(prev)
                tail_size = min(len(prev_toks), overlap)
                if tail_size + ntok <= max_tokens:
                    tail = enc.decode(prev_toks[-overlap:])
                    buf_texts.append(tail)
                    buf_tok = tail_size
        if buf_path is None:
            buf_path = path
        buf_texts.append(text)
        buf_tok += ntok
    emit()

    return chunks


def chunk_file(
    path: Path,
    root: Path,
    *,
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    rel = path.relative_to(root)
    return chunk_markdown(
        path.read_text(encoding="utf-8"),
        source=str(rel).replace("\\", "/"),
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap=overlap,
    )


def _discover_md(root: Path) -> list[Path]:
    return [
        p
        for p in sorted(root.rglob("*.md"))
        if not any(part in _SKIP_DIRS for part in p.relative_to(root).parts)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/chunks.preview.jsonl")
    )
    parser.add_argument("--min-tokens", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    md_files = _discover_md(args.in_dir)
    if args.limit:
        md_files = md_files[: args.limit]
    log.info("Discovered %d markdown files under %s", len(md_files), args.in_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    with args.out.open("w", encoding="utf-8") as f:
        for idx, path in enumerate(md_files, start=1):
            chunks = chunk_file(
                path,
                args.in_dir,
                min_tokens=args.min_tokens,
                max_tokens=args.max_tokens,
                overlap=args.overlap,
            )
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            n_chunks += len(chunks)
            if idx % 50 == 0 or idx == len(md_files):
                log.info("progress %d/%d  chunks=%d", idx, len(md_files), n_chunks)

    log.info("Done. files=%d chunks=%d → %s", len(md_files), n_chunks, args.out)
    print(f"files={len(md_files)} chunks={n_chunks} out={args.out}")


if __name__ == "__main__":
    main()
