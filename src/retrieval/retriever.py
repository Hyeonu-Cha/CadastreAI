"""Dense retriever against the Qdrant `cadastre_chunks` collection.

Encodes a query with `BAAI/bge-base-en-v1.5` (the same model used by
`src.index.embed`) and runs a top-k cosine search. The returned objects
are `(chunk_dict, score)` tuples where `chunk_dict` is the full chunk
payload stored at index time (chunk_id, publisher, title, date, url,
source, section_heading, page, text, token_count).

BGE v1.5 passage encoding uses no prefix, but **query** encoding uses:

    "Represent this sentence for searching relevant passages: "

That prefix is what makes the retrieval symmetric with the indexed
vectors — omitting it degrades recall noticeably, so `_QUERY_PREFIX`
is applied unconditionally here.

The model and Qdrant client are loaded lazily and cached on a singleton
`Retriever` so repeat calls amortize the ~1s model load and avoid
reconnecting. Connection target is env-driven:

    QDRANT_URL         default "http://localhost:6333"
    QDRANT_API_KEY     optional (see MEMORY re: v1.12 empty-string 401)
    QDRANT_COLLECTION  default "cadastre_chunks"

    python -m src.retrieval.retriever --query "first home buyer grant NSW" -k 5
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_COLLECTION = "cadastre_chunks"
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Retriever:
    model_name: str = DEFAULT_MODEL
    collection: str = DEFAULT_COLLECTION
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    use_embedding_cache: bool = True

    def __post_init__(self) -> None:
        self._model = None
        self._client = None
        self._embed_cache = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _ensure_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            url = self.qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
            api_key = self.qdrant_api_key or os.environ.get("QDRANT_API_KEY") or None
            log.info("Connecting to Qdrant %s", url)
            self._client = QdrantClient(url=url, api_key=api_key)
        return self._client

    def _ensure_embed_cache(self):
        if self._embed_cache is None:
            from src.retrieval.embedding_cache import EmbeddingCache

            self._embed_cache = EmbeddingCache(enabled=self.use_embedding_cache)
        return self._embed_cache

    def retrieve(self, query: str, k: int = 10) -> list[tuple[dict, float]]:
        model = self._ensure_model()
        client = self._ensure_client()
        cache = self._ensure_embed_cache()
        vec = cache.embed(
            model,
            query,
            prefix=_QUERY_PREFIX,
            model_name=self.model_name,
        ).tolist()
        res = client.query_points(
            collection_name=self.collection,
            query=vec,
            limit=k,
            with_payload=True,
        ).points
        return [(p.payload, p.score) for p in res]


_DEFAULT: Retriever | None = None


def retrieve(query: str, k: int = 10) -> list[tuple[dict, float]]:
    """Module-level convenience — uses a process-wide cached Retriever."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Retriever(
            collection=os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION),
        )
    return _DEFAULT.retrieve(query, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-api-key", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    r = Retriever(
        model_name=args.model,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
    )
    hits = r.retrieve(args.query, k=args.k)
    for i, (payload, score) in enumerate(hits, start=1):
        title = (payload.get("title") or "")[:70]
        section = (payload.get("section_heading") or "")[:70]
        url = payload.get("url") or ""
        print(f"[{i:2d}] score={score:.4f}  {payload.get('publisher')} | {title}")
        if section:
            print(f"     {section}")
        print(f"     {url}")


if __name__ == "__main__":
    main()
