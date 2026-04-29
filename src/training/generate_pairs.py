"""Generate (query, chunk_id) training pairs by prompting an LLM per chunk.

For each sampled chunk, asks the LLM to author N persona-tagged
questions whose answer is in the chunk. The chunk is the positive for
each generated query. Hard negatives are mined separately in Task 2.08.

Three providers are supported: Anthropic (Claude Haiku, the original
2.06 implementation), Google (Gemini Flash, added in X.05), and
OpenAI (GPT-4o-mini, added in X.07). Pair generation is the only LLM
step in the fine-tune pipeline that doesn't bind to the production
agent's choice of model — anything that can write a specific question
grounded in a chunk works. Pick the one your credit budget points at,
or the one that isn't currently throttling.

Sampling rules (per Task 2.06 spec, with sane defaults):
- Exclude chunk_ids already used as gold in any `queries_*.jsonl`
  eval file under `--eval-dir` (avoids train/eval contamination).
- Drop short boilerplate chunks (`token_count < --min-tokens`, default 100).
- Random sample `--sample` chunks (default 2000) to hit the ~4000-pair
  target at 2 queries per chunk.

Output is line-delimited JSON, one record per (query, chunk_id) pair:

    {"chunk_id": "...", "query": "...", "persona": "homebuyer|investor|researcher",
     "model": "<provider-model-id>"}

The script is idempotent: re-running picks up where it left off by
reading existing `--out` and skipping chunk_ids that already have
records. Run it twice and the second pass should be a no-op.

    # Default — Anthropic Claude Haiku (uses ANTHROPIC_API_KEY)
    python -m src.training.generate_pairs --sample 2000 -n 2 \
        --out data/training/pairs_raw.jsonl

    # Gemini Flash (uses GEMINI_API_KEY)
    python -m src.training.generate_pairs --provider gemini --sample 2000 -n 2 \
        --out data/training/pairs_raw.jsonl

    # OpenAI GPT-4o-mini (uses OPENAI_API_KEY)
    python -m src.training.generate_pairs --provider openai --sample 2000 -n 2 \
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

PROVIDERS = ("anthropic", "gemini", "openai")
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL_ANTHROPIC = "claude-haiku-4-5-20251001"
DEFAULT_MODEL_GEMINI = "gemini-2.5-flash"
DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
# Back-compat alias: the original public name used by callers.
DEFAULT_MODEL = DEFAULT_MODEL_ANTHROPIC
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


async def _generate_one_anthropic(
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


async def _generate_one_gemini(
    client: Any,
    chunk: dict,
    n: int,
    model: str,
    max_tokens: int,
    sem: asyncio.Semaphore,
) -> tuple[dict, list[dict] | None, str | None]:
    # Lazy import — only the gemini path needs the SDK.
    from google.genai import types  # type: ignore[import-not-found]

    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        max_output_tokens=max_tokens,
        # Force structured JSON output so the parser doesn't have to
        # strip code fences. The same `_parse_response` regex still
        # works as a defence-in-depth fallback.
        response_mime_type="application/json",
    )
    async with sem:
        try:
            resp = await client.aio.models.generate_content(
                model=model,
                contents=_user_prompt(chunk, n),
                config=config,
            )
        except Exception as e:  # noqa: BLE001 — log + skip on any API failure
            return chunk, None, f"api_error: {type(e).__name__}: {e}"
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        return chunk, None, "parse_error: empty response"
    try:
        queries = _parse_response(text)
    except ValueError as e:
        return chunk, None, f"parse_error: {e}"
    return chunk, queries, text


async def _generate_one_openai(
    client: Any,
    chunk: dict,
    n: int,
    model: str,
    max_tokens: int,
    sem: asyncio.Semaphore,
) -> tuple[dict, list[dict] | None, str | None]:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                # JSON mode — caller's system prompt already says "respond
                # with strict JSON only", which is what unlocks this in
                # the OpenAI API.
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _user_prompt(chunk, n)},
                ],
            )
        except Exception as e:  # noqa: BLE001 — log + skip on any API failure
            return chunk, None, f"api_error: {type(e).__name__}: {e}"
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        return chunk, None, "parse_error: empty response"
    try:
        queries = _parse_response(text)
    except ValueError as e:
        return chunk, None, f"parse_error: {e}"
    return chunk, queries, text


async def _run_async(
    chunks: list[dict],
    out_path: Path,
    *,
    provider: str,
    n: int,
    model: str,
    max_tokens: int,
    concurrency: int,
) -> dict:
    if provider == "anthropic":
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot generate.")
        client: Any = anthropic.AsyncAnthropic(api_key=api_key)
        generate_one = _generate_one_anthropic
    elif provider == "gemini":
        from google import genai  # type: ignore[import-not-found]

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set; cannot generate."
            )
        client = genai.Client(api_key=api_key)
        generate_one = _generate_one_gemini
    elif provider == "openai":
        from openai import AsyncOpenAI  # type: ignore[import-not-found]

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot generate.")
        client = AsyncOpenAI(api_key=api_key)
        generate_one = _generate_one_openai
    else:
        raise ValueError(f"unknown provider: {provider!r} (expected one of {PROVIDERS})")

    sem = asyncio.Semaphore(concurrency)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pairs = 0
    n_failed = 0
    n_done = 0
    write_lock = asyncio.Lock()

    async def process(chunk: dict) -> None:
        nonlocal n_pairs, n_failed, n_done
        chunk, queries, raw = await generate_one(client, chunk, n, model, max_tokens, sem)
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
    provider: str,
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
    log.info("Using provider=%s model=%s", provider, model)
    if not selected:
        log.info("Nothing to do.")
        return {"n_chunks": 0, "n_pairs": 0, "n_failed": 0}

    return asyncio.run(
        _run_async(
            selected,
            out_path,
            provider=provider,
            n=per_chunk,
            model=model,
            max_tokens=max_tokens,
            concurrency=concurrency,
        )
    )


def _default_model_for(provider: str) -> str:
    if provider == "anthropic":
        return DEFAULT_MODEL_ANTHROPIC
    if provider == "gemini":
        return DEFAULT_MODEL_GEMINI
    if provider == "openai":
        return DEFAULT_MODEL_OPENAI
    raise ValueError(f"unknown provider: {provider!r} (expected one of {PROVIDERS})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/training/pairs_raw.jsonl"))
    p.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    p.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    p.add_argument("-n", "--per-chunk", type=int, default=DEFAULT_PER_CHUNK)
    p.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    p.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=os.environ.get("CADASTRE_GEN_PROVIDER", DEFAULT_PROVIDER),
        help="LLM provider for query generation (default: anthropic).",
    )
    # `--model` defaults are provider-specific; resolved post-parse.
    p.add_argument(
        "--model",
        default=None,
        help=(
            f"Model id. Defaults to {DEFAULT_MODEL_ANTHROPIC!r} for "
            f"--provider anthropic, {DEFAULT_MODEL_GEMINI!r} for --provider gemini, "
            f"{DEFAULT_MODEL_OPENAI!r} for --provider openai. "
            "Override with CLAUDE_GEN_MODEL, GEMINI_GEN_MODEL, or OPENAI_GEN_MODEL env vars."
        ),
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

    if args.model is None:
        env_override = {
            "anthropic": os.environ.get("CLAUDE_GEN_MODEL"),
            "gemini": os.environ.get("GEMINI_GEN_MODEL"),
            "openai": os.environ.get("OPENAI_GEN_MODEL"),
        }.get(args.provider)
        args.model = env_override or _default_model_for(args.provider)

    summary = run(
        chunks_path=args.chunks,
        out_path=args.out,
        sample=args.sample,
        per_chunk=args.per_chunk,
        min_tokens=args.min_tokens,
        eval_dir=args.eval_dir,
        provider=args.provider,
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
