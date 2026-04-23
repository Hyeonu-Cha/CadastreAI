"""Upsert embeddings + chunk metadata into Qdrant collection `cadastre_chunks`.

Reads three parallel artifacts produced by Task 1.19 / 1.20:

    data/processed/embeddings.npy    float32 (N, 768), L2-normalized
    data/processed/chunk_ids.txt     one chunk_id per line, same order
    data/processed/chunks.jsonl      full chunk records, joined by chunk_id

Each Qdrant point uses a deterministic UUID5 derived from `chunk_id`
(string ids aren't natively supported; sequential ints would lose
traceability across re-embeds). The original `chunk_id` is duplicated
into the payload so the retriever can recover it.

Collection geometry: 768-d vectors, Cosine distance. Cosine on unit
vectors is equivalent to dot product, but asking Qdrant for Cosine is
clearer and resilient if a future re-embed forgets to normalize.

Qdrant target is read from env:

    QDRANT_URL      default "http://localhost:6333"  (docker-compose)
    QDRANT_API_KEY  optional

By default the collection is **recreated** (dropped + rebuilt) so
re-runs are idempotent. Pass `--no-recreate` to keep the existing
collection and append/overwrite points.

    python -m src.index.upsert \\
        --chunks data/processed/chunks.jsonl \\
        --embeddings data/processed/embeddings.npy \\
        --ids data/processed/chunk_ids.txt \\
        --collection cadastre_chunks --batch-size 256
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_COLLECTION = "cadastre_chunks"
DEFAULT_DIM = 768
_PAYLOAD_FIELDS = (
    "chunk_id",
    "publisher",
    "title",
    "date",
    "url",
    "source",
    "section_heading",
    "page",
    "text",
    "token_count",
)


def _point_id(chunk_id: str) -> str:
    """Deterministic UUID5 so re-runs overwrite the same point."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _load_payloads(chunks_path: Path) -> dict[str, dict]:
    """Index chunks.jsonl by chunk_id, keeping only payload fields."""
    out: dict[str, dict] = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["chunk_id"]] = {k: rec.get(k) for k in _PAYLOAD_FIELDS}
    return out


def upsert(
    *,
    chunks_path: Path,
    embeddings_path: Path,
    ids_path: Path,
    collection: str = DEFAULT_COLLECTION,
    batch_size: int = 256,
    recreate: bool = True,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
) -> tuple[int, str]:
    import numpy as np
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = qdrant_api_key or os.environ.get("QDRANT_API_KEY") or None

    vectors = np.load(embeddings_path)
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    ids = ids_path.read_text(encoding="utf-8").splitlines()
    if len(ids) != vectors.shape[0]:
        raise ValueError(
            f"id/vector count mismatch: {len(ids)} ids vs {vectors.shape[0]} vectors"
        )
    dim = vectors.shape[1]
    log.info(
        "Loaded %d vectors (dim=%d) and %d ids", vectors.shape[0], dim, len(ids)
    )

    payloads = _load_payloads(chunks_path)
    log.info("Loaded %d payload records from %s", len(payloads), chunks_path)

    missing = [cid for cid in ids if cid not in payloads]
    if missing:
        raise ValueError(
            f"{len(missing)} chunk_ids from {ids_path} missing in {chunks_path}; "
            f"first: {missing[0]}"
        )

    log.info("Connecting to Qdrant %s (api_key=%s)", url, "set" if api_key else "none")
    client = QdrantClient(url=url, api_key=api_key)

    if recreate:
        log.info("Recreating collection %s (dim=%d, Cosine)", collection, dim)
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    else:
        existing = [c.name for c in client.get_collections().collections]
        if collection not in existing:
            log.info("Creating collection %s (dim=%d, Cosine)", collection, dim)
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    n = len(ids)
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = [
            PointStruct(
                id=_point_id(ids[i]),
                vector=vectors[i].tolist(),
                payload=payloads[ids[i]],
            )
            for i in range(start, end)
        ]
        client.upsert(collection_name=collection, points=batch, wait=False)
        if (end // batch_size) % 20 == 0 or end == n:
            log.info("upsert %d/%d (%.1f pts/s)", end, n, end / max(time.time() - t0, 1e-6))

    count = client.count(collection_name=collection, exact=True).count
    log.info("Upsert complete. collection=%s count=%d in %.1fs", collection, count, time.time() - t0)
    return count, collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks", type=Path, default=Path("data/processed/chunks.jsonl")
    )
    parser.add_argument(
        "--embeddings", type=Path, default=Path("data/processed/embeddings.npy")
    )
    parser.add_argument(
        "--ids", type=Path, default=Path("data/processed/chunk_ids.txt")
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--no-recreate",
        dest="recreate",
        action="store_false",
        help="Keep existing collection instead of dropping+rebuilding",
    )
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-api-key", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    count, collection = upsert(
        chunks_path=args.chunks,
        embeddings_path=args.embeddings,
        ids_path=args.ids,
        collection=args.collection,
        batch_size=args.batch_size,
        recreate=args.recreate,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
    )
    print(f"collection={collection} count={count}")


if __name__ == "__main__":
    main()
