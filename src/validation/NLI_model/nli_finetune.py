#!/usr/bin/env python3
"""Fine-tune a cross-encoder NLI model on FFLO triplet training pairs.

Reads ``nli_train_pairs.csv`` (premise/hypothesis/label), splits by chunk
to prevent leakage, and fine-tunes ``cross-encoder/nli-deberta-v3-small``
(or any base model) using ``sentence_transformers.CrossEncoder.fit()``.

GPU support:
    Single-GPU:   --device cuda:0
    Multi-GPU:    --device-ids 0,1,2,3,4,5,6   (DataParallel)
    Distributed:  torchrun --nproc_per_node=7 nli_finetune.py --distributed

Usage::

    # Single GPU
    food_lab/bin/python src/validation/NLI_model/nli_finetune.py \\
        --train-csv    src/validation/Golden_val/nli_train_pairs.csv \\
        --base-model   cross-encoder/nli-deberta-v3-small \\
        --output-dir   model_checkpoints/run4_deberta_v3_small \\
        --epochs       5 --batch-size 16 --device cuda:0

    # 7-GPU DataParallel
    food_lab/bin/python src/validation/NLI_model/nli_finetune.py \\
        --train-csv    src/validation/Golden_val/nli_train_pairs.csv \\
        --base-model   cross-encoder/nli-deberta-v3-small \\
        --output-dir   model_checkpoints/run4_deberta_v3_small \\
        --epochs       5 --batch-size 16 --device-ids 0,1,2,3,4,5,6

    # 7-GPU Distributed (torchrun)
    torchrun --nproc_per_node=7 src/validation/NLI_model/nli_finetune.py \\
        --train-csv    src/validation/Golden_val/nli_train_pairs.csv \\
        --base-model   cross-encoder/nli-deberta-v3-small \\
        --output-dir   model_checkpoints/run4_deberta_v3_small \\
        --epochs       5 --batch-size 16 --distributed
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_TO_ID = {"contradiction": 0, "entailment": 1, "neutral": 2}
ID_TO_LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}
DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"
DEFAULT_VAL_SPLIT = 0.2
SEED = 42


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_training_data(
    csv_path: Path,
) -> tuple[
    list[dict[str, str]], dict[str, list[dict[str, str]]]
]:
    """Return (all_rows, grouped_by_snippet_id)."""
    rows: list[dict[str, str]] = []
    by_snippet: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
            by_snippet[row["snippet_id"]].append(dict(row))
    return rows, dict(by_snippet)


def stratified_chunk_split(
    by_snippet: dict[str, list[dict[str, str]]],
    val_ratio: float = DEFAULT_VAL_SPLIT,
    seed: int = SEED,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split by unique snippet_id, stratified by predicate presence.

    Ensures every predicate appears in both train and validation if
    possible, and the total chunk ratio is close to val_ratio.
    """
    chunk_preds: dict[str, set[str]] = {}
    chunk_sizes: dict[str, int] = {}
    for sid, rows in by_snippet.items():
        chunk_preds[sid] = {r["predicate"] for r in rows}
        chunk_sizes[sid] = len(rows)

    pred_to_chunks: dict[str, list[str]] = defaultdict(list)
    for sid, preds in chunk_preds.items():
        for p in preds:
            pred_to_chunks[p].append(sid)

    rng = random.Random(seed)
    val_chunks: set[str] = set()

    for p, sids in sorted(pred_to_chunks.items()):
        rng.shuffle(sids)
        # For predicates with >=2 chunks, allocate one to val
        if len(sids) >= 2:
            val_chunks.add(sids[0])
        elif len(sids) == 1 and sum(chunk_sizes[s] for s in val_chunks) < val_ratio * sum(chunk_sizes.values()):
            # Only child — allocate to val if still under budget
            val_chunks.add(sids[0])

    # Rebalance: if val is too large, move excess chunks back to train
    all_sids = list(by_snippet.keys())
    target_val = max(1, int(len(all_sids) * val_ratio))
    val_list = list(val_chunks)
    rng.shuffle(val_list)
    while len(val_chunks) > target_val:
        val_chunks.discard(val_list.pop())

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for sid, rows in by_snippet.items():
        if sid in val_chunks:
            val_rows.extend(rows)
        else:
            train_rows.extend(rows)

    # ---- Filter cross-premise leaks ----
    # Same boilerplate text can appear in different chunks (e.g.
    # "The testing in laboratory shall be ensured…" appears in two
    # different FSSAI documents).  If one chunk landed in train and
    # the other in val, the model sees identical premise text in both
    # splits.  Remove those rows from val.
    train_premises = {r["premise"] for r in train_rows}
    leaked = 0
    clean_val: list[dict[str, str]] = []
    for r in val_rows:
        if r["premise"] in train_premises:
            leaked += 1
        else:
            clean_val.append(r)
    if leaked:
        print(f"  (removed {leaked} cross-premise leak rows from validation)")

    return train_rows, clean_val


def predicate_disjoint_split(
    by_snippet: dict[str, list[dict[str, str]]],
    val_ratio: float = DEFAULT_VAL_SPLIT,
    seed: int = SEED,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split by predicate — zero predicate overlap between train and val.

    Each predicate is assigned entirely to train OR val.  The model is
    tested on relations it has never seen during training, providing a
    cross-relation generalisation test.

    Single-chunk predicates always go to train (not enough data to test).
    """
    chunk_preds: dict[str, set[str]] = {}
    chunk_sizes: dict[str, int] = {}
    for sid, rows in by_snippet.items():
        chunk_preds[sid] = {r["predicate"] for r in rows}
        chunk_sizes[sid] = len(rows)

    # Predicate → total row count (sum of chunk sizes)
    pred_total_rows: dict[str, int] = defaultdict(int)
    pred_to_chunks: dict[str, list[str]] = defaultdict(list)
    for sid, preds in chunk_preds.items():
        for p in preds:
            pred_to_chunks[p].append(sid)
            pred_total_rows[p] += chunk_sizes[sid]

    rng = random.Random(seed)

    # Sort predicates by total rows, pick val predicates to fill the budget
    preds_sorted = sorted(
        pred_total_rows.items(), key=lambda x: -x[1]
    )
    target_val_rows = int(val_ratio * sum(chunk_sizes.values()))

    val_predicates: set[str] = set()
    val_row_budget = 0
    val_eligible = [(p, rows) for p, rows in preds_sorted
                    if len(pred_to_chunks[p]) >= 2]  # need ≥2 chunks to test
    rng.shuffle(val_eligible)

    for p, p_rows in val_eligible:
        if val_row_budget + p_rows <= target_val_rows * 1.3:
            val_predicates.add(p)
            val_row_budget += p_rows

    # Assign chunks to val if ANY of their predicates are in val_predicates
    val_chunks: set[str] = set()
    for sid, preds in chunk_preds.items():
        if preds & val_predicates:
            val_chunks.add(sid)

    # ---- Fix predicate leakage from multi-predicate chunks ----
    # If a chunk assigned to val contains a predicate that was NOT
    # in val_predicates, that predicate leaks from train.  Iteratively
    # remove the leakiest predicates from val_eligible until overlap=0.
    for _ in range(20):
        train_chunks_temp = set(by_snippet.keys()) - val_chunks
        train_preds_temp: set[str] = set()
        for sid in train_chunks_temp:
            train_preds_temp.update(chunk_preds[sid])
        val_preds_temp: set[str] = set()
        for sid in val_chunks:
            val_preds_temp.update(chunk_preds[sid])
        leaking = val_preds_temp & train_preds_temp
        if not leaking:
            break
        # A chunk in val that also contains a train-side predicate
        # leaks that predicate.  Move those chunks to train.
        val_chunks = {sid for sid in val_chunks
                      if not (chunk_preds[sid] & leaking)}

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for sid, rows in by_snippet.items():
        if sid in val_chunks:
            val_rows.extend(rows)
        else:
            train_rows.extend(rows)

    # ---- Verify zero predicate overlap ----
    train_preds = {r["predicate"] for r in train_rows}
    val_preds = {r["predicate"] for r in val_rows}
    overlap = train_preds & val_preds
    if overlap:
        print(
            f"  WARNING: {len(overlap)} predicates leaked between splits"
            f" (chunk sharing): {sorted(list(overlap))[:5]}..."
        )

    # ---- Filter cross-premise leaks (unchanged) ----
    train_premises = {r["premise"] for r in train_rows}
    leaked = 0
    clean_val: list[dict[str, str]] = []
    for r in val_rows:
        if r["premise"] in train_premises:
            leaked += 1
        else:
            clean_val.append(r)
    if leaked:
        print(f"  (removed {leaked} cross-premise leak rows from validation)")

    return train_rows, clean_val


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    rows: list[dict[str, str]],
    predictions: list[int],
) -> dict[str, Any]:
    """Compute per-predicate accuracy, label confusion, and
    per-predicate confusion matrices."""
    pred_to_gold: dict[str, list[tuple[int, int]]] = defaultdict(list)
    correct = 0
    total = 0
    label_confusion: dict[str, Counter] = defaultdict(Counter)
    pred_confusion: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    for i, row in enumerate(rows):
        gold_id = LABEL_TO_ID.get(row["label"], -1)
        pred_id = predictions[i]
        p = row["predicate"]
        pred_to_gold[p].append((gold_id, pred_id))
        label_confusion[ID_TO_LABEL.get(gold_id, "?")][
            ID_TO_LABEL.get(pred_id, "?")
        ] += 1
        # Per-predicate confusion
        pred_confusion[p][ID_TO_LABEL.get(gold_id, "?")][
            ID_TO_LABEL.get(pred_id, "?")
        ] += 1
        if gold_id == pred_id:
            correct += 1
        total += 1

    per_pred: dict[str, dict[str, Any]] = {}
    for p, pairs in sorted(pred_to_gold.items()):
        acc = sum(1 for g, pr in pairs if g == pr) / max(len(pairs), 1)
        per_pred[p] = {
            "accuracy": round(acc, 4),
            "count": len(pairs),
            "confusion": {
                gold: dict(preds)
                for gold, preds in sorted(pred_confusion[p].items())
            },
        }

    confusion: dict[str, dict[str, int]] = {}
    for gold, preds in label_confusion.items():
        confusion[gold] = dict(preds)

    return {
        "accuracy": round(correct / max(total, 1), 4),
        "correct": correct,
        "total": total,
        "per_predicate": per_pred,
        "confusion_matrix": confusion,
    }


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

# All 67 FFLO relation canonical names (synced with schema v2).
# Hardcoded so the script is self-contained — no dependency on extract_triplets.
_ALL_FFLO_RELATIONS = frozenset([
    "affectsPopulation", "amends", "appliesTo", "appliesToCategory",
    "appliesToFood", "associatedWith", "belongsToCategory", "carriedBy",
    "causesEffect", "citesFinding", "collectedAt", "commonIn",
    "comparedAgainst", "compliesWith", "confirmedBy", "constitutes",
    "created_by", "definedIn", "defines", "detectedBy", "forProperty",
    "foundAt", "foundAtStep", "foundIn", "hasAdulterant",
    "hasAdulterationMethod", "hasDefinition", "hasEffectType",
    "hasFraudType", "hasFunction", "hasIngredient", "hasIntentionality",
    "hasLocation", "hasNumericValue", "hasPermissibleLimit",
    "hasSampleType", "hasSubcategory", "hasValue", "identifies",
    "inFood", "inRegion", "inputTo", "isMeasuredIn", "isPerformedAs",
    "isPerformedAt", "isPerformedOn", "isResultOf", "isSubstituteFor",
    "issuedBy", "occursAt", "outputOf", "producedBy",
    "producesIndicator", "propagatesTo", "relatesToProperty",
    "requiresKit", "specifiedIn", "supportedBy", "tableLabel",
    "targets", "triggeredAction", "used", "violates",
    "wasAssociatedWith", "wasAttributedTo", "wasGeneratedBy",
    "wasInfluencedBy",
])


def log_coverage(rows: list[dict[str, str]]) -> None:
    """Log how many FFLO relations have training data."""
    covered = {r["predicate"] for r in rows if r["predicate"]}

    # Strip prefixes for canonical comparison (training data uses mixed)
    def _c(name: str) -> str:
        return name.replace("fflo:", "").replace("fkg:", "").replace(
            "fso:", ""
        ).replace("prov:", "").replace("lkif:", "")

    covered_c = {_c(p) for p in covered}
    total = len(_ALL_FFLO_RELATIONS)
    matched = len(covered_c & _ALL_FFLO_RELATIONS)
    missing = sorted(_ALL_FFLO_RELATIONS - covered_c)

    print(f"\nRelation coverage: {matched}/{total}")
    if missing:
        print(f"  Missing ({len(missing)}):")
        for m in missing:
            print(f"    - {m}")
        print(
            "  (These will not be evaluated during validation.  "
            "The model can still predict them at inference time "
            "using its pretrained knowledge.)"
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _get_torch():
    import torch
    return torch


def _get_crossencoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder


def _get_inputexample():
    from sentence_transformers import InputExample
    return InputExample


def build_dataloader(
    rows: list[dict[str, str]],
    batch_size: int,
    shuffle: bool = True,
) -> Any:
    """Build a DataLoader of InputExample objects."""
    torch = _get_torch()
    InputExample = _get_inputexample()

    examples = [
        InputExample(
            texts=[r["premise"], r["hypothesis"]],
            label=LABEL_TO_ID[r["label"]],
        )
        for r in rows
    ]
    return torch.utils.data.DataLoader(
        examples, batch_size=batch_size, shuffle=shuffle,
    )


def run_finetune(args: argparse.Namespace) -> int:
    torch = _get_torch()
    CrossEncoder = _get_crossencoder()

    # ---- Distributed setup ----
    if args.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        world_size = torch.distributed.get_world_size()
        print(f"[Rank {local_rank}/{world_size}] Distributed training active")
    else:
        local_rank = 0
        world_size = 1

    # ---- Resolve device ----
    if args.device_ids:
        device_ids = [int(x.strip()) for x in args.device_ids.split(",")]
        device = torch.device(f"cuda:{device_ids[0]}")
        print(f"Using DataParallel on GPUs: {device_ids}")
    elif args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device_ids = None
    else:
        device = torch.device(args.device)
        device_ids = None

    # ---- Load data ----
    train_csv = Path(args.train_csv).resolve()
    if not train_csv.exists():
        print(f"error: training CSV not found: {train_csv}", file=sys.stderr)
        return 1
    print(f"Loading {train_csv} ...")
    rows, by_snippet = load_training_data(train_csv)
    print(f"  {len(rows)} rows, {len(by_snippet)} unique chunks")

    # ---- Coverage ----
    if not args.distributed or local_rank == 0:
        log_coverage(rows)

    # ---- Split ----
    if args.split_mode == "predicate_disjoint":
        train_rows, val_rows = predicate_disjoint_split(
            by_snippet, val_ratio=args.val_split,
        )
    else:
        train_rows, val_rows = stratified_chunk_split(
            by_snippet, val_ratio=args.val_split,
        )

    if not args.distributed or local_rank == 0:
        train_chunks = len(set(r["snippet_id"] for r in train_rows))
        val_chunks = len(set(r["snippet_id"] for r in val_rows))
        train_preds = {r["predicate"] for r in train_rows}
        val_preds = {r["predicate"] for r in val_rows}
        print(f"Train: {len(train_rows)} rows ({train_chunks} chunks, "
              f"{len(train_preds)} predicates)")
        print(f"Val:   {len(val_rows)} rows ({val_chunks} chunks, "
              f"{len(val_preds)} predicates, "
              f"overlap={len(train_preds & val_preds)})")
        if train_chunks and val_chunks:
            assert not (set(r["snippet_id"] for r in train_rows) &
                        set(r["snippet_id"] for r in val_rows)), \
                "BUG: chunk leakage detected between train and val"

    # ---- Build dataloaders ----
    effective_bs = args.batch_size * max(1, world_size)
    train_loader = build_dataloader(train_rows, effective_bs, shuffle=True)

    # ---- Load model ----
    print(f"\nLoading base model: {args.base_model}")
    model = CrossEncoder(
        args.base_model,
        num_labels=3,
        device=device,
        max_length=args.max_length,
    )

    # ---- Multi-GPU ----
    if device_ids and len(device_ids) > 1:
        print(f"Wrapping with DataParallel (GPUs: {device_ids})")
        model.model = torch.nn.DataParallel(
            model.model, device_ids=device_ids,
        )
        # DataParallel moves the model; update device reference
        model.model.to(device_ids[0])

    # ---- Build evaluator ----
    # NOTE: EmbeddingSimilarityEvaluator is for bi-encoder (SentenceTransformer)
    # cosine-similarity *regression* — it expects a continuous `score` per pair
    # and isn't valid for CrossEncoder.fit(). For 3-way NLI classification use
    # CEClassificationEvaluator, which takes parallel lists of sentence pairs
    # and integer class labels and reports accuracy/F1 per class.
    def _get_evaluator():
        from sentence_transformers.cross_encoder.evaluation import (
            CrossEncoderClassificationEvaluator,
        )
        return CrossEncoderClassificationEvaluator

    CrossEncoderClassificationEvaluator = _get_evaluator()
    val_sentence_pairs = [[r["premise"], r["hypothesis"]] for r in val_rows]
    val_labels = [LABEL_TO_ID[r["label"]] for r in val_rows]
    evaluator = CrossEncoderClassificationEvaluator(
        sentence_pairs=val_sentence_pairs,
        labels=val_labels,
        name="validation",
    )

    # ---- Output dir ----
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {output_dir}")

    # ---- Train ----
    warmup = int(len(train_loader) * args.warmup_ratio) if args.warmup_ratio else args.warmup_steps
    print(f"\nTraining: epochs={args.epochs} batch={effective_bs} lr={args.learning_rate} warmup={warmup}")
    print(f"  train steps/epoch={len(train_loader)} eval steps={args.eval_steps}")

    # CrossEncoder.fit() is a deprecated shim: internally it hardcodes the HF
    # Trainer's own working directory to a *relative* "checkpoints/model"
    # path, independent of the `output_path` we pass below (which only
    # controls where the final best-model snapshot is saved via callback).
    # Run it from a throwaway temp dir (auto-deleted afterward) rather than
    # from output_dir itself, so that internal bookkeeping never lands
    # inside your model's folder — keeps output_dir containing only the
    # actual model files, which matters if you're training multiple models
    # into sibling output_dir paths.
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(dir=str(output_dir.parent)) as tmp_dir:
        os.chdir(tmp_dir)
        try:
            model.fit(
                train_dataloader=train_loader,
                evaluator=evaluator,
                epochs=args.epochs,
                evaluation_steps=args.eval_steps,
                warmup_steps=warmup,
                optimizer_params={"lr": args.learning_rate},
                output_path=str(output_dir),
                save_best_model=True,
                show_progress_bar=(not args.distributed or local_rank == 0),
            )
        finally:
            os.chdir(original_cwd)

    # ---- Final evaluation ----
    if not args.distributed or local_rank == 0:
        # Load best checkpoint
        best_ckpt = sorted(output_dir.glob("checkpoint-*"))[-1] if list(output_dir.glob("checkpoint-*")) else output_dir
        print(f"\nFinal evaluation using: {best_ckpt}")
        best_model = CrossEncoder(
            str(best_ckpt),
            device=device,
            max_length=args.max_length,
        )

        val_premises = [r["premise"] for r in val_rows]
        val_hypotheses = [r["hypothesis"] for r in val_rows]
        scores = best_model.predict(
            list(zip(val_premises, val_hypotheses)),
            batch_size=effective_bs,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        preds = scores.argmax(dim=1).cpu().tolist()

        metrics = compute_metrics(val_rows, preds)
        print(f"\nVal accuracy: {metrics['accuracy']:.4f}")
        print(f"  Correct: {metrics['correct']}/{metrics['total']}")

        # Chunk-level held-out report
        val_chunk_ids = {r["snippet_id"] for r in val_rows}
        train_chunk_ids = {r["snippet_id"] for r in train_rows}
        overlap = val_chunk_ids & train_chunk_ids
        unique_val_premises = len({r["premise"] for r in val_rows})
        print(
            f"  Held-out chunks: {len(val_chunk_ids)} "
            f"(0 overlap with train, {unique_val_premises} unique premises)"
        )

        # Per-predicate breakdown
        per_pred = metrics.get("per_predicate", {})
        if per_pred:
            print("\nPer-predicate accuracy (validation):")
            for p, m in sorted(per_pred.items(), key=lambda x: -x[1]["accuracy"]):
                bar = "█" * int(m["accuracy"] * 20)
                print(f"  {p:45s} {m['accuracy']:.3f}  {bar}")

        # Save metrics
        metrics_path = output_dir / "eval_results.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nMetrics saved to {metrics_path}")

        # Save training config
        config = {
            "base_model": args.base_model,
            "train_csv": str(train_csv),
            "epochs": args.epochs,
            "batch_size": effective_bs,
            "learning_rate": args.learning_rate,
            "warmup_steps": warmup,
            "max_length": args.max_length,
            "val_split": args.val_split,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
        }
        (output_dir / "training_args.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ---- Cleanup distributed ----
    if args.distributed:
        torch.distributed.destroy_process_group()

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune cross-encoder NLI model on FFLO training pairs"
    )
    p.add_argument("--train-csv", type=str, required=True,
                   help="Path to nli_train_pairs.csv")
    p.add_argument("--base-model", type=str, default=DEFAULT_MODEL,
                   help=f"Base HF model (default: {DEFAULT_MODEL})")
    p.add_argument("--output-dir", type=str,
                   default="model_checkpoints/run4_deberta_v3_small",
                   help="Output directory for checkpoints")
    p.add_argument("--epochs", type=int, default=5,
                   help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Per-GPU batch size")
    p.add_argument("--learning-rate", type=float, default=2e-5,
                   help="Learning rate")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Linear warmup steps (overrides --warmup-ratio)")
    p.add_argument("--warmup-ratio", type=float, default=0.1,
                   help="Linear warmup as fraction of total steps "
                        "(used when --warmup-steps=0)")
    p.add_argument("--max-length", type=int, default=256,
                   help="Max token length for premise+hypothesis")
    p.add_argument("--eval-steps", type=int, default=100,
                   help="Evaluate every N steps")
    p.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT,
                   help="Fraction of chunks for validation")
    p.add_argument("--split-mode", type=str, default="stratified",
                   choices=["stratified", "predicate_disjoint"],
                   help="How to split: 'stratified' ensures every predicate "
                        "appears in both splits; 'predicate_disjoint' ensures "
                        "zero predicate overlap between train and val "
                        "(cross-relation generalisation test)")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto, cpu, cuda:0, ...")
    p.add_argument("--device-ids", type=str, default=None,
                   help="Comma-separated GPU IDs for DataParallel "
                        "(e.g. 0,1,2,3,4,5,6)")
    p.add_argument("--distributed", action="store_true",
                   help="Use torch.distributed (torchrun launcher required)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return run_finetune(args)


if __name__ == "__main__":
    sys.exit(main())