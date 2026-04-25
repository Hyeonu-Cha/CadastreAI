"""Fine-tune the BGE base encoder on (anchor, positive, negatives) triplets.

Trains `BAAI/bge-base-en-v1.5` (or any sentence-transformers model) with
`MultipleNegativesRankingLoss` over the triplets `build_triplets.py`
produced. The loss treats every other anchor's positive in the same
batch as an in-batch negative *plus* uses the explicit hard negatives
from each row, which is the standard recipe for retrieval contrastive
fine-tuning.

Why MultipleNegativesRankingLoss:
 - Symmetric InfoNCE: maximises cos(anchor, positive) relative to all
   other positives in the batch (cheap implicit negatives) and the
   explicit hard negatives we mined from BM25 (Task 2.08).
 - No need to compute similarity targets; it just needs (anchor, pos,
   neg, neg, ...) tuples.

Inputs:
 - data/training/triplets.jsonl with rows
   {"anchor": "...", "positive": "...", "negatives": [...]}
   produced by `src.training.build_triplets`.
 - The triplets are split into train / dev (default 90/10, seeded). The
   dev split feeds a `RerankingEvaluator` that computes MAP and MRR@10
   at the end of every epoch — gives us an early-stopping signal and
   training curves without needing the full retrieval-eval pipeline.

Eval output:
 - <out_dir>/eval/RerankingEvaluator_dev_results.csv (per-epoch MAP/MRR)
 - stdout logs include training-loss-per-step from `fit()`.

Output: a trained sentence-transformers model directory under
`--out-dir` (default `models/bge-base-cadastre/`). The directory is
drop-in compatible with `SentenceTransformer.load(...)` and the existing
retriever — just point `embed.py` / `retriever.py` at it.

    python -m src.training.train_embeddings \
        --triplets data/training/triplets.jsonl \
        --out-dir models/bge-base-cadastre \
        --batch 64 --epochs 3 --lr 2e-5
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_OUT_DIR = Path("models/bge-base-cadastre")
DEFAULT_BATCH = 64
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_DEV_FRAC = 0.10
DEFAULT_SEED = 42
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _load_triplets(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("anchor") or not rec.get("positive"):
                continue
            negs = rec.get("negatives") or []
            if not negs:
                continue
            rows.append(
                {
                    "anchor": rec["anchor"],
                    "positive": rec["positive"],
                    "negatives": negs,
                }
            )
    return rows


def _split_train_dev(
    rows: list[dict], dev_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_dev = max(1, int(len(shuffled) * dev_frac)) if dev_frac > 0 else 0
    return shuffled[n_dev:], shuffled[:n_dev]


def _build_dev_eval_samples(rows: list[dict]) -> list[dict]:
    """Shape dev rows for `RerankingEvaluator`.

    The evaluator takes `[{"query", "positive": [...], "negative": [...]}]`
    and computes MAP and MRR@10 by ranking each query against the union
    of its positives + negatives. We use the *raw* anchor without the
    BGE query prefix here — `RerankingEvaluator` calls `model.encode()`
    directly on whatever string we hand it, and adding the prefix here
    would still match because the model is trained with prefixed anchors.
    Keeping it raw matches the eval/inference path our retriever already
    uses (which prefixes inside `embed.py`).
    """
    return [
        {
            "query": _QUERY_PREFIX + r["anchor"],
            "positive": [r["positive"]],
            "negative": list(r["negatives"]),
        }
        for r in rows
        if r.get("anchor") and r.get("positive") and r.get("negatives")
    ]


def _to_input_examples(rows: list[dict], n_negatives: int):
    """Shape rows into `InputExample(texts=[anchor, positive, neg1, ..., negN])`
    — the format `MultipleNegativesRankingLoss` expects when used with
    `model.fit()`. The loss treats the first item as the anchor, the
    second as the positive, and any remaining items as explicit hard
    negatives (in addition to in-batch negatives).

    Pad short negative lists by repeating the last element so every
    example has the same length — `DataLoader` with the default
    collate function requires that.
    """
    from sentence_transformers import InputExample

    out = []
    for r in rows:
        negs = r["negatives"][:n_negatives]
        if not negs:
            continue
        if len(negs) < n_negatives:
            negs = negs + [negs[-1]] * (n_negatives - len(negs))
        out.append(
            InputExample(
                texts=[_QUERY_PREFIX + r["anchor"], r["positive"], *negs]
            )
        )
    return out


def run(
    triplets_path: Path,
    out_dir: Path,
    *,
    base_model: str,
    batch: int,
    epochs: int,
    lr: float,
    warmup_ratio: float,
    dev_frac: float,
    seed: int,
    n_negatives: int,
) -> dict:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import RerankingEvaluator
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from torch.utils.data import DataLoader

    log.info("Loading triplets from %s", triplets_path)
    rows = _load_triplets(triplets_path)
    log.info("Loaded %d valid triplets", len(rows))
    if not rows:
        raise RuntimeError("no triplets to train on")

    train_rows, dev_rows = _split_train_dev(rows, dev_frac, seed)
    log.info(
        "Split %d train / %d dev (dev_frac=%.2f)",
        len(train_rows), len(dev_rows), dev_frac,
    )

    train_examples = _to_input_examples(train_rows, n_negatives)
    log.info("Built %d train InputExamples (anchor+pos+%d negs)", len(train_examples), n_negatives)

    evaluator = None
    if dev_rows:
        dev_samples = _build_dev_eval_samples(dev_rows)
        evaluator = RerankingEvaluator(
            samples=dev_samples,
            name="dev",
            mrr_at_k=10,
            batch_size=batch,
            show_progress_bar=False,
        )
        log.info("Built RerankingEvaluator over %d dev samples", len(dev_samples))

    log.info("Loading base model %s", base_model)
    model = SentenceTransformer(base_model)
    loss = MultipleNegativesRankingLoss(model)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(
        train_examples, shuffle=True, batch_size=batch, collate_fn=model.smart_batching_collate,
    )
    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = int(steps_per_epoch * epochs * warmup_ratio)
    log.info(
        "fit() epochs=%d batch=%d lr=%g warmup_steps=%d eval_per_epoch=%s",
        epochs, batch, lr, warmup_steps, evaluator is not None,
    )
    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=str(out_dir),
        evaluator=evaluator,
        evaluation_steps=steps_per_epoch if evaluator is not None else 0,
        save_best_model=False,
        show_progress_bar=True,
    )

    log.info("Saving final model to %s", out_dir)
    model.save(str(out_dir))

    return {
        "n_train": len(train_rows),
        "n_dev": len(dev_rows),
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "out_dir": str(out_dir),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--triplets", type=Path, default=Path("data/training/triplets.jsonl")
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--dev-frac", type=float, default=DEFAULT_DEV_FRAC)
    p.add_argument("--n-negatives", type=int, default=5)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    summary = run(
        triplets_path=args.triplets,
        out_dir=args.out_dir,
        base_model=args.base_model,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        dev_frac=args.dev_frac,
        seed=args.seed,
        n_negatives=args.n_negatives,
    )
    print(
        f"n_train={summary['n_train']} n_dev={summary['n_dev']} "
        f"epochs={summary['epochs']} batch={summary['batch']} "
        f"lr={summary['lr']} out_dir={summary['out_dir']}"
    )


if __name__ == "__main__":
    main()
