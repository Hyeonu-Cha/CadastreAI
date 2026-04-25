"""Re-embed the corpus with a fine-tuned model and upsert to a new Qdrant collection.

Thin orchestrator over the existing index pipeline. After Task 2.12 lands
the fine-tuned checkpoint at `models/bge-au-housing-v1/`, we want to
benchmark it against the base BGE without overwriting the live
`cadastre_chunks` collection. This script drives:

    1. `src.index.embed` with `--model models/bge-au-housing-v1`,
       writing parallel `embeddings.npy` and `chunk_ids.txt` into
       `data/processed/finetuned/` (so we keep both arrays around for
       offline analysis without clobbering the base BGE artifacts).

    2. `src.index.upsert` against the same Qdrant cluster but a new
       collection name (default `cadastre_chunks_ft`). The retriever
       can then point at either collection via its existing
       `--collection` flag, giving a clean A/B harness.

Both underlying scripts are idempotent — re-running this orchestrator
overwrites the .npy/.txt and recreates the collection by default.

    python -m src.training.reembed_finetuned \
        --model models/bge-au-housing-v1 \
        --collection cadastre_chunks_ft

Set `QDRANT_URL` / `QDRANT_API_KEY` in env (the upsert step reads them
the same way `src.index.upsert` does directly).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("models/bge-au-housing-v1")
DEFAULT_CHUNKS = Path("data/processed/chunks.jsonl")
DEFAULT_OUT_DIR = Path("data/processed/finetuned")
DEFAULT_COLLECTION = "cadastre_chunks_ft"


def run(
    *,
    model_dir: Path,
    chunks_path: Path,
    out_dir: Path,
    collection: str,
    batch_size: int,
    upsert_batch_size: int,
    device: str | None,
    skip_embed: bool,
    skip_upsert: bool,
    qdrant_url: str | None,
    qdrant_api_key: str | None,
) -> dict:
    from src.index.embed import embed
    from src.index.upsert import upsert

    emb_path = out_dir / "embeddings.npy"
    ids_path = out_dir / "chunk_ids.txt"

    if not skip_embed:
        log.info("Embedding step → %s with model=%s", out_dir, model_dir)
        n, emb_path, ids_path = embed(
            chunks_path=chunks_path,
            out_dir=out_dir,
            model_name=str(model_dir),
            batch_size=batch_size,
            device=device,
        )
        log.info("Embedded %d chunks → %s, %s", n, emb_path, ids_path)
    else:
        log.info("Skipping embed (using existing %s)", emb_path)

    upserted = 0
    if not skip_upsert:
        log.info("Upsert step → collection=%s", collection)
        upserted, _ = upsert(
            chunks_path=chunks_path,
            embeddings_path=emb_path,
            ids_path=ids_path,
            collection=collection,
            batch_size=upsert_batch_size,
            recreate=True,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
        )
    else:
        log.info("Skipping upsert.")

    return {
        "model_dir": str(model_dir),
        "out_dir": str(out_dir),
        "collection": collection,
        "embeddings": str(emb_path),
        "ids": str(ids_path),
        "upserted": upserted,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", dest="model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--upsert-batch-size", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embed step and reuse existing --out-dir artifacts.",
    )
    p.add_argument(
        "--skip-upsert",
        action="store_true",
        help="Embed only; useful for offline ablations without Qdrant.",
    )
    p.add_argument("--qdrant-url", default=None)
    p.add_argument("--qdrant-api-key", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    summary = run(
        model_dir=args.model_dir,
        chunks_path=args.chunks,
        out_dir=args.out_dir,
        collection=args.collection,
        batch_size=args.batch_size,
        upsert_batch_size=args.upsert_batch_size,
        device=args.device,
        skip_embed=args.skip_embed,
        skip_upsert=args.skip_upsert,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
    )
    print(
        f"model={summary['model_dir']} collection={summary['collection']} "
        f"upserted={summary['upserted']} out_dir={summary['out_dir']}"
    )


if __name__ == "__main__":
    main()
