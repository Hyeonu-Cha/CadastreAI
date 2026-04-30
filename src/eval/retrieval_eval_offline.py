"""Offline retrieval eval — brute-force cosine over a `.npy` corpus.

Same metrics, JSON shape, and scoring as `src.eval.retrieval_eval`, but
loads pre-computed chunk embeddings from disk and runs `matmul` against
each query vector instead of querying Qdrant. Useful when Qdrant is
unavailable (e.g. Docker is down) or for quick A/B of a fine-tuned
checkpoint without standing up a parallel collection.

Inputs (defaults match Task 2.13's outputs):
    --model        SentenceTransformer dir or HF id (default fine-tuned)
    --embeddings   (N, D) float32 array, L2-normalized
    --chunk-ids    one chunk_id per line, same order as the array

    python -m src.eval.retrieval_eval_offline \\
        --queries data/eval/queries_all.jsonl \\
        --out results/finetuned.json -k 10
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.eval.retrieval_eval import run

log = logging.getLogger(__name__)

DEFAULT_MODEL = "models/bge-au-housing-v1"
DEFAULT_EMBEDDINGS = Path("data/processed/finetuned/embeddings.npy")
DEFAULT_CHUNK_IDS = Path("data/processed/finetuned/chunk_ids.txt")
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class NpyRetriever:
    """Drop-in replacement for `Retriever` that searches a NumPy matrix.

    Exposes `.retrieve(query, k)` returning `[(payload, score), ...]`
    matching the dense retriever contract — `payload` only carries the
    `chunk_id` since that's all `retrieval_eval.run` reads.
    """

    def __init__(
        self,
        model_name: str,
        embeddings_path: Path,
        chunk_ids_path: Path,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        log.info("Loading model %s", model_name)
        self.model_name = model_name
        self.collection = f"npy:{embeddings_path}"
        self._model = SentenceTransformer(model_name, device=device)

        log.info("Loading embeddings %s", embeddings_path)
        self._embeds = np.load(embeddings_path).astype(np.float32, copy=False)
        with open(chunk_ids_path, "r", encoding="utf-8") as f:
            self._chunk_ids = [line.strip() for line in f if line.strip()]
        if len(self._chunk_ids) != self._embeds.shape[0]:
            raise ValueError(
                f"chunk_ids ({len(self._chunk_ids)}) != embeddings rows "
                f"({self._embeds.shape[0]})"
            )
        log.info(
            "Ready: %d chunks, dim=%d", self._embeds.shape[0], self._embeds.shape[1]
        )

    def retrieve(self, query: str, k: int = 10) -> list[tuple[dict, float]]:
        vec = self._model.encode(
            [_QUERY_PREFIX + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32, copy=False)
        sims = self._embeds @ vec
        if k >= sims.shape[0]:
            order = np.argsort(-sims)
        else:
            top = np.argpartition(-sims, k)[:k]
            order = top[np.argsort(-sims[top])]
        return [
            ({"chunk_id": self._chunk_ids[int(i)]}, float(sims[int(i)]))
            for i in order[:k]
        ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", type=Path, default=Path("data/eval/queries_all.jsonl"))
    p.add_argument("--out", type=Path, default=Path("results/finetuned.json"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    p.add_argument("--chunk-ids", type=Path, default=DEFAULT_CHUNK_IDS)
    p.add_argument("-k", "--top-k", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None, help="torch device (e.g. 'cuda', 'cpu')")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    retriever = NpyRetriever(
        model_name=args.model,
        embeddings_path=args.embeddings,
        chunk_ids_path=args.chunk_ids,
        device=args.device,
    )
    run(
        queries_path=args.queries,
        out_path=args.out,
        top_k=args.top_k,
        retriever=retriever,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
