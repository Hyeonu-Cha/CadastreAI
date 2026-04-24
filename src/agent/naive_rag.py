"""Naive RAG pipeline: BGE top-k retrieval → Claude Sonnet 4 with citations.

This is the baseline pipeline we measure against for Week 1 (Task 1.31).
"Naive" means: no reranking, no query rewriting, no multi-hop, no filters
— just dense retrieval of top-k chunks stuffed into a single prompt, with
the model instructed to cite inline as [1] [2] etc.

The prompt format mirrors what Anthropic recommends for RAG: each chunk
is wrapped in a tagged <source id="..."> block that carries chunk
metadata (publisher, title, section, url). That lets the model attribute
claims to specific chunks without having to re-read raw payload, and
makes the citation markers [1]..[k] unambiguous.

Returns a `RagAnswer` dict with the answer text, the retrieved chunks
(in rank order — so caller can resolve [1] → chunk_id), and the raw
token usage. Callers who only need retrieval results can skip generation
with `generate=False`.

Env:
    ANTHROPIC_API_KEY  required unless generate=False
    ANTHROPIC_MODEL    optional override, default claude-sonnet-4-6

    python -m src.agent.naive_rag --query "what is the first home buyer grant in NSW?" -k 5
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

from src.retrieval.retriever import Retriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_K = 5
DEFAULT_MAX_TOKENS = 1024

_SYSTEM_PROMPT = (
    "You are a research assistant for the Australian housing market. "
    "Answer the user's question using only the <sources> provided below. "
    "Cite every factual claim inline with bracketed numbers matching the "
    "source id, e.g. [1] or [2][3]. If the sources do not contain the "
    "answer, say so explicitly — do not invent facts. Keep the answer "
    "concise (under 200 words) and neutral in tone."
)


@dataclass
class RagAnswer:
    query: str
    answer: str
    chunks: list[dict]  # in retrieval rank order, 1-indexed to match [N] citations
    scores: list[float]
    model: str
    usage: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "model": self.model,
            "usage": self.usage,
            "chunks": [
                {
                    "rank": i + 1,
                    "chunk_id": c.get("chunk_id"),
                    "publisher": c.get("publisher"),
                    "title": c.get("title"),
                    "section_heading": c.get("section_heading"),
                    "url": c.get("url"),
                    "score": s,
                }
                for i, (c, s) in enumerate(zip(self.chunks, self.scores, strict=True))
            ],
        }


def _format_sources(chunks: list[dict]) -> str:
    """Render retrieved chunks as <source id="N"> blocks for the prompt."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        header_bits = [
            f"publisher={c.get('publisher') or '?'}",
            f"title={(c.get('title') or '').strip()}",
        ]
        if section := (c.get("section_heading") or "").strip():
            header_bits.append(f"section={section}")
        if url := (c.get("url") or "").strip():
            header_bits.append(f"url={url}")
        header = " | ".join(header_bits)
        text = (c.get("text") or "").strip()
        parts.append(f'<source id="{i}" {header}>\n{text}\n</source>')
    return "\n\n".join(parts)


def answer(
    query: str,
    k: int = DEFAULT_K,
    *,
    retriever: Retriever | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    generate: bool = True,
) -> RagAnswer:
    """Run the naive pipeline end-to-end.

    If `generate=False`, returns a RagAnswer with the retrieved chunks
    but an empty answer — useful for retrieval-only eval (Task 1.30).
    """
    if retriever is None:
        retriever = Retriever()
    hits = retriever.retrieve(query, k=k)
    chunks = [payload for payload, _ in hits]
    scores = [score for _, score in hits]

    if not generate:
        return RagAnswer(
            query=query, answer="", chunks=chunks, scores=scores, model=model, usage={}
        )

    import anthropic  # local import so retrieval-only callers don't pay it

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot generate.")

    client = anthropic.Anthropic(api_key=api_key)
    user_content = (
        f"<sources>\n{_format_sources(chunks)}\n</sources>\n\n"
        f"Question: {query}"
    )
    log.debug("Calling %s with %d sources", model, len(chunks))
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    text = "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    )
    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
    return RagAnswer(
        query=query,
        answer=text,
        chunks=chunks,
        scores=scores,
        model=model,
        usage=usage,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--no-generate", action="store_true", help="retrieval only")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    out = answer(
        args.query,
        k=args.k,
        model=args.model,
        max_tokens=args.max_tokens,
        generate=not args.no_generate,
    )

    print(f"\n=== retrieved {len(out.chunks)} chunks ===")
    for i, (c, s) in enumerate(zip(out.chunks, out.scores, strict=True), start=1):
        title = (c.get("title") or "")[:70]
        section = (c.get("section_heading") or "")[:70]
        print(f"[{i}] score={s:.4f}  {c.get('publisher')} | {title}")
        if section:
            print(f"    {section}")

    if out.answer:
        print(f"\n=== answer ({out.model}) ===")
        print(out.answer)
        print(f"\nusage: input={out.usage['input_tokens']} output={out.usage['output_tokens']}")


if __name__ == "__main__":
    main()
