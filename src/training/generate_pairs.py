"""Generate (query, chunk_id) training pairs by prompting Claude per chunk.

For each sampled chunk, asks Claude Haiku to author N persona-tagged
questions whose answer is in the chunk. The chunk is the positive for
each generated query. Hard negatives are mined separately in Task 2.08.

Sampling rules (per Task 2.06 spec, with sane defaults):
- Exclude chunk_ids already used as gold in any `queries_*.jsonl`
  eval file under `--eval-dir` (avoids train/eval contamination).
- Drop short boilerplate chunks (`token_count < --min-tokens`, default 100).
- Random sample `--sample` chunks (default 2000) to hit the ~4000-pair
  target at 2 queries per chunk.

Output is line-delimited JSON, one record per (query, chunk_id) pair:

    {"chunk_id": "...", "query": "...", "persona": "homebuyer|investor|researcher",
     "model": "claude-haiku-...", "raw": "<model JSON>"}

The script is idempotent: re-running picks up where it left off by
reading existing `--out` and skipping chunk_ids that already have
records. Run it twice and the second pass should be a no-op.

    python -m src.training.generate_pairs --sample 2000 -n 2 \
        --out data/training/pairs_raw.jsonl

The raw output is filtered by `src.training.filter_pairs` (Task 2.07) to
produce the canonical `data/training/pairs.jsonl` consumed downstream.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_SAMPLE = 2000
DEFAULT_PER_CHUNK = 2
DEFAULT_MIN_TOKENS = 100
DEFAULT_CONCURRENCY = 8
DEFAULT_MAX_TOKENS = 600
PERSONAS = ("homebuyer", "investor", "researcher")

_SYSTEM_PROMPT = (
    "You generate training data for an Australian housing-market retrieval "
    "system. Given one chunk of source text, produce questions whose answer "
    "is clearly contained in that chunk. Each question must be specific "
    "enough that a retriever pointing at this chunk would unambiguously be "
    "correct — not generic ('what is housing affordability?') and not "
    "answerable from the chunk's title alone. Keep questions natural; avoid "
    "lifting verbatim phrases from the chunk.\n\n"
    "Each question is tagged with one persona:\n"
    "- homebuyer: someone looking to buy or having recently bought a home\n"
    "- investor: someone owning or considering rental/investment property\n"
    "- researcher: an academic, policy analyst, or regulator reading the "
    "evidence\n\n"
    "Respond with strict JSON only, no preamble, in this exact shape:\n"
    '{"queries": [{"query": "...", "persona": "..."}, ...]}'
)


def _user_prompt(chunk: dict, n: int) -> str:
    title = (chunk.get("title") or "").strip()
    publisher = (chunk.get("publisher") or "").strip()
    section = (chunk.get("section_heading") or "").strip()
    text = (chunk.get("text") or "").strip()
    header = f"publisher={publisher} | title={title}"
    if section:
        header += f" | section={section}"
    return (
        f"Source chunk metadata: {header}\n\n"
        f"Source chunk text:\n{text}\n\n"
        f"Produce exactly {n} questions for this chunk."
    )


def _load_chunks(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_eval_gold_ids(eval_dir: Path) -> set[str]:
    """Collect every chunk_id that appears as gold in any eval queries file."""
    if not eval_dir.exists():
        return set()
    out: set[str] = set()
    for p in sorted(eval_dir.glob("queries*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for cid in rec.get("gold_chunk_ids") or []:
                    out.add(cid)
    return out


def _load_existing_chunk_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    out: set[str] = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("chunk_id")
            if cid:
                out.add(cid)
    return out


def _select_chunks(
    chunks: list[dict],
    sample: int,
    min_tokens: int,
    excluded: set[str],
    seed: int,
) -> list[dict]:
    pool = [
        c for c in chunks
        if c.get("chunk_id") not in excluded
        and (c.get("token_count") or 0) >= min_tokens
    ]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:sample]


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _parse_response(text: str) -> list[dict]:
    """Extract `queries` list from the model's JSON-only response.

    Tolerates the model wrapping JSON in code fences or trailing text — we
    grab the first balanced `{...}` and parse that.
    """
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"no JSON object found in response: {text[:120]!r}")
    payload = json.loads(m.group(0))
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise ValueError(f"missing 'queries' list in response: {payload!r}")
    valid: list[dict] = []
    for q in queries:
        query_text = (q.get("query") or "").strip() if isinstance(q, dict) else ""
        persona = (q.get("persona") or "").strip().lower() if isinstance(q, dict) else ""
        if not query_text or persona not in PERSONAS:
            continue
        valid.append({"query": query_text, "persona": persona})
    if not valid:
        raise ValueError(f"no valid (query, persona) pairs in response: {payload!r}")
    return valid


async def _generate_one(
    client: Any,
    chunk: dict,
    n: int,
    model: str,
    max_tokens: int,
    sem: asyncio.Semaphore,
) -> tuple[dict, list[dict] | None, str | None]:
    async with sem:
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _user_prompt(chunk, n)}],
            )
        except Exception as e:  # noqa: BLE001 — log + skip on any API failure
            return chunk, None, f"api_error: {type(e).__name__}: {e}"
    text = "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    )
    try:
        queries = _parse_response(text)
    except ValueError as e:
        return chunk, None, f"parse_error: {e}"
    return chunk, queries, text


async def _run_async(
    chunks: list[dict],
    out_path: Path,
    *,
    n: int,
    model: str,
    max_tokens: int,
    concurrency: int,
) -> dict:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot generate.")
    client = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pairs = 0
    n_failed = 0
    n_done = 0
    write_lock = asyncio.Lock()

    async def process(chunk: dict) -> None:
        nonlocal n_pairs, n_failed, n_done
        chunk, queries, raw = await _generate_one(client, chunk, n, model, max_tokens, sem)
        n_done += 1
        if queries is None:
            n_failed += 1
            log.warning("skip %s — %s", chunk.get("chunk_id"), raw)
            return
        records = [
            {
                "chunk_id": chunk["chunk_id"],
                "query": q["query"],
                "persona": q["persona"],
                "model": model,
            }
            for q in queries
        ]
        async with write_lock:
            with out_path.open("a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_pairs += len(records)
        if n_done % 50 == 0 or n_done == len(chunks):
            log.info(
                "  %d/%d chunks done — %d pairs written, %d failed",
                n_done, len(chunks), n_pairs, n_failed,
            )

    await asyncio.gather(*(process(c) for c in chunks))
    return {"n_chunks": n_done, "n_pairs": n_pairs, "n_failed": n_failed}


def run(
    chunks_path: Path,
    out_path: Path,
    *,
    sample: int,
    per_chunk: int,
    min_tokens: int,
    eval_dir: Path,
    model: str,
    max_tokens: int,
    concurrency: int,
    seed: int,
) -> dict:
    log.info("Loading chunks from %s", chunks_path)
    chunks = _load_chunks(chunks_path)

    eval_gold = _load_eval_gold_ids(eval_dir)
    existing = _load_existing_chunk_ids(out_path)
    excluded = eval_gold | existing
    log.info(
        "Excluding %d chunk_ids (%d eval gold, %d already in %s)",
        len(excluded), len(eval_gold), len(existing), out_path.name,
    )

    selected = _select_chunks(chunks, sample, min_tokens, excluded, seed)
    log.info(
        "Selected %d chunks (target sample=%d, min_tokens=%d) for generation",
        len(selected), sample, min_tokens,
    )
    if not selected:
        log.info("Nothing to do.")
        return {"n_chunks": 0, "n_pairs": 0, "n_failed": 0}

    return asyncio.run(
        _run_async(
            selected,
            out_path,
            n=per_chunk,
            model=model,
            max_tokens=max_tokens,
            concurrency=concurrency,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/training/pairs_raw.jsonl"))
    p.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    p.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    p.add_argument("-n", "--per-chunk", type=int, default=DEFAULT_PER_CHUNK)
    p.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    p.add_argument(
        "--model", default=os.environ.get("CLAUDE_GEN_MODEL", DEFAULT_MODEL)
    )
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    summary = run(
        chunks_path=args.chunks,
        out_path=args.out,
        sample=args.sample,
        per_chunk=args.per_chunk,
        min_tokens=args.min_tokens,
        eval_dir=args.eval_dir,
        model=args.model,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        seed=args.seed,
    )
    print(
        f"chunks={summary['n_chunks']} pairs={summary['n_pairs']} "
        f"failed={summary['n_failed']} out={args.out}"
    )


if __name__ == "__main__":
    main()
