"""Embed chunks with `BAAI/bge-base-en-v1.5` via sentence-transformers.

Reads `data/processed/chunks.jsonl` (from Task 1.19) and writes two
parallel artifacts to the output dir:

    embeddings.npy    float32, shape (N, 768), L2-normalized
    chunk_ids.txt     one chunk_id per line, same order as rows above

Task 1.21 (Qdrant upsert) zips these together with the payload read back
from chunks.jsonl. Keeping ids in a sidecar file (rather than baking them
into the .npy) lets us reuse this output for non-Qdrant consumers and
makes the vector array directly mmap-friendly.

BGE specifics:
  - The v1.5 series expects NO prefix on passage (document) encoding;
    only query encoding uses "Represent this sentence for searching
    relevant passages: ". Task 1.22 (retriever) will handle that side.
  - `normalize_embeddings=True` gives unit vectors so Qdrant can use
    cosine distance directly on dot product.
  - Max sequence length is 512 WordPiece tokens; longer chunks get
    model-side truncation (accepted — see Task 1.18 docstring).

Device selection is automatic via sentence-transformers (CUDA if
available, else CPU). Pass `--device` to override.

    python -m src.index.embed --chunks data/processed/chunks.jsonl \\
        --out-dir data/processed --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_DIM = 768


def _load_chunks(path: Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids.append(rec["chunk_id"])
            texts.append(rec["text"])
    return ids, texts


def embed(
    *,
    chunks_path: Path,
    out_dir: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[int, Path, Path]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    ids, texts = _load_chunks(chunks_path)
    log.info("Loaded %d chunks from %s", len(ids), chunks_path)

    log.info("Loading model %s (device=%s)", model_name, device or "auto")
    t0 = time.time()
    model = SentenceTransformer(model_name, device=device)
    log.info("Model loaded in %.1fs (device=%s)", time.time() - t0, model.device)

    t0 = time.time()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    dt = time.time() - t0
    log.info(
        "Encoded %d chunks in %.1fs (%.1f chunks/s), shape=%s dtype=%s",
        len(texts), dt, len(texts) / max(dt, 1e-6), vectors.shape, vectors.dtype,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.npy"
    ids_path = out_dir / "chunk_ids.txt"
    np.save(emb_path, vectors)
    ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    log.info("Wrote %s (%.1f MB) and %s", emb_path, emb_path.stat().st_size / 1e6, ids_path)

    return len(ids), emb_path, ids_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks", type=Path, default=Path("data/processed/chunks.jsonl")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", default=None, help="torch device string (e.g. cuda, cpu); auto if unset"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    n, emb_path, ids_path = embed(
        chunks_path=args.chunks,
        out_dir=args.out_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"n={n} embeddings={emb_path} ids={ids_path}")


if __name__ == "__main__":
    main()
