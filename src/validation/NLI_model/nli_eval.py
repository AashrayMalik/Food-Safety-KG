#!/usr/bin/env python3
"""Evaluate one or more NLI models on a labelled test CSV.

Compares pretrained, fine-tuned, and fine-tuned-vs-fine-tuned models
side-by-side.  For every model, saves per-example predictions (so you
know exactly which rows went wrong) plus aggregate metrics.

Usage::

    # Single model
    python nli_eval.py \\
        --models cross-encoder/nli-deberta-v3-small \\
        --eval-csv nli_train_pairs.csv \\
        --device cuda:0

    # Pretrained vs fine-tuned
    python nli_eval.py \\
        --models pretrained,model_checkpoints/run4_deberta_v3_small \\
        --labels pretrained,finetuned \\
        --eval-csv nli_train_pairs.csv \\
        --device cuda:0

    # Three models
    python nli_eval.py \\
        --models pretrained,model_checkpoints/run4_deberta_v3_small,path/to/cp1000 \\
        --labels pretrained,finetuned-v1,finetuned-v2 \\
        --eval-csv nli_train_pairs.csv \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Co-located with nli_finetune.py on HPC — imports LABEL_TO_ID, ID_TO_LABEL,
# compute_metrics.  See nli_finetune.py for their definitions.
from nli_finetune import (
    LABEL_TO_ID,
    ID_TO_LABEL,
    compute_metrics,
)

PRETRAINED_SHORTCUT = "pretrained"
PRETRAINED_MODEL = "cross-encoder/nli-deberta-v3-small"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_data(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: str, max_length: int = 256):
    from sentence_transformers import CrossEncoder

    path = (
        PRETRAINED_MODEL if model_path == PRETRAINED_SHORTCUT
        else model_path
    )
    return CrossEncoder(path, device=device, max_length=max_length)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _predict_safe(model: Any, pairs: list[tuple[str, str]],
                  batch_size: int) -> np.ndarray:
    """Return (n, 3) numpy array of scores, handling v5.x API variance."""
    scores = model.predict(pairs, batch_size=batch_size,
                           show_progress_bar=True)
    # v5.x may return torch.Tensor, numpy, or list of lists
    if hasattr(scores, "cpu"):              # torch.Tensor
        return scores.cpu().numpy()
    if isinstance(scores, np.ndarray):       # numpy
        return scores
    return np.array(scores)                  # list of lists


def _slug(path: str) -> str:
    """Filesystem-safe short name."""
    name = str(path).replace("/", "__").replace("\\", "__")
    return name[:60]


# ---------------------------------------------------------------------------
# Per-example CSV
# ---------------------------------------------------------------------------

def save_examples_csv(
    rows: list[dict[str, str]],
    predictions: list[int],
    confidences: list[float],
    model_label: str,
    output_dir: Path,
) -> None:
    path = output_dir / f"{model_label}_examples.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "premise", "hypothesis", "gold_label", "predicted_label",
            "confidence", "correct",
            "predicate", "snippet_id",
        ])
        for i, row in enumerate(rows):
            gold_id = LABEL_TO_ID.get(row["label"], -1)
            correct = 1 if gold_id == predictions[i] else 0
            writer.writerow([
                row["premise"], row["hypothesis"], row["label"],
                ID_TO_LABEL.get(predictions[i], "?"),
                round(confidences[i], 4), correct,
                row.get("predicate", ""), row.get("snippet_id", ""),
            ])
    print(f"  Examples CSV → {path}")


# ---------------------------------------------------------------------------
# Disagreements CSV (multi-model only)
# ---------------------------------------------------------------------------

def save_disagreements(
    rows: list[dict[str, str]],
    all_preds: dict[str, list[int]],
    all_confs: dict[str, list[float]],
    output_dir: Path,
) -> None:
    path = output_dir / "disagreements.csv"
    model_names = list(all_preds.keys())
    if len(model_names) < 2:
        return

    cols = ["premise", "hypothesis", "gold_label", "predicate", "snippet_id"]
    for m in model_names:
        cols += [f"{m}_pred", f"{m}_conf", f"{m}_correct"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)

        written = 0
        for i, row in enumerate(rows):
            gold_id = LABEL_TO_ID.get(row["label"], -1)
            corrects = {
                m: (1 if gold_id == all_preds[m][i] else 0)
                for m in model_names
            }
            # Row is interesting if any model was wrong or models disagree
            unique = len(set(all_preds[m][i] for m in model_names))
            if unique > 1 or any(c == 0 for c in corrects.values()):
                out = [
                    row["premise"], row["hypothesis"], row["label"],
                    row.get("predicate", ""), row.get("snippet_id", ""),
                ]
                for m in model_names:
                    out += [
                        ID_TO_LABEL.get(all_preds[m][i], "?"),
                        round(all_confs[m][i], 4),
                        corrects[m],
                    ]
                writer.writerow(out)
                written += 1

    print(f"  Disagreements CSV → {path} ({written} rows)")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(
    results: dict[str, dict[str, Any]],
    model_labels: list[str],
) -> None:
    print()
    print("═" * 82)
    hdr = f"{'Model':25s} {'Acc':>7s}  {'entail':>6s}  {'neutral':>6s}  {'contra':>6s}  {'#':>5s}"
    print(hdr)
    print("─" * 82)

    for label in model_labels:
        m = results[label]["aggregate"]
        cf = results[label].get("confusion_matrix", {})
        ent = _f1(cf, "entailment")
        neu = _f1(cf, "neutral")
        con = _f1(cf, "contradiction")
        print(
            f"{label:25s} {m['accuracy']:7.3f}  "
            f"{ent:6.3f}  {neu:6.3f}  {con:6.3f}  {m['total']:>5d}"
        )
    print("═" * 82)

    # Per-predicate delta (if ≥2 models)
    if len(model_labels) >= 2:
        ref = model_labels[0]
        cmp = model_labels[-1]
        ref_pp = results[ref]["aggregate"].get("per_predicate", {})
        cmp_pp = results[cmp]["aggregate"].get("per_predicate", {})
        all_preds = sorted(set(list(ref_pp.keys()) + list(cmp_pp.keys())))
        deltas = []
        for p in all_preds:
            ra = ref_pp.get(p, {}).get("accuracy", 0.0)
            ca = cmp_pp.get(p, {}).get("accuracy", 0.0)
            delta = ca - ra
            if abs(delta) > 0.001:
                deltas.append((delta, p, ra, ca))

        if deltas:
            print(f"\nPer-predicate delta ({cmp} − {ref}):")
            for delta, p, ra, ca in sorted(deltas, reverse=True):
                bar = "█" * min(20, int(abs(delta) * 80))
                sign = "+" if delta >= 0 else ""
                print(
                    f"  {p:45s} {sign}{delta:+.3f}  "
                    f"({ra:.3f} → {ca:.3f})  {bar}"
                )

    # Disagreement summary
    if len(model_labels) >= 2:
        preds = {m: results[m].get("predictions", []) for m in model_labels}
        disagree = 0
        for i in range(len(preds[model_labels[0]])):
            if len(set(preds[m][i] for m in model_labels)) > 1:
                disagree += 1
        print(f"\nModels disagree on {disagree}/{len(preds[model_labels[0]])} "
              f"rows ({disagree/max(len(preds[model_labels[0]]),1)*100:.1f}%)")


def _f1(confusion: dict[str, dict[str, int]], label: str) -> float:
    if label not in confusion:
        return 0.0
    tp = confusion[label].get(label, 0)
    total_pred = sum(confusion[label].values())
    total_gold = sum(confusion.get(g, {}).get(label, 0)
                     for g in confusion)
    prec = tp / max(total_pred, 1)
    rec = tp / max(total_gold, 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    model_paths = [m.strip() for m in args.models.split(",")]
    if args.labels:
        model_labels = [l.strip() for l in args.labels.split(",")]
        if len(model_labels) != len(model_paths):
            print("error: --labels count must match --models count",
                  file=sys.stderr)
            return 1
    else:
        model_labels = [_slug(p) for p in model_paths]

    # Load data
    eval_csv = Path(args.eval_csv)
    if not eval_csv.exists():
        print(f"error: eval CSV not found: {eval_csv}", file=sys.stderr)
        return 1
    print(f"Loading {eval_csv}")
    rows = load_eval_data(eval_csv)
    premises = [r["premise"] for r in rows]
    hypotheses = [r["hypothesis"] for r in rows]
    pairs = list(zip(premises, hypotheses))
    print(f"  {len(rows)} rows\n")

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results per model
    results: dict[str, Any] = {}
    all_preds: dict[str, list[int]] = {}
    all_confs: dict[str, list[float]] = {}

    for path, label in zip(model_paths, model_labels):
        print(f"Model: {label} ({path})")
        model = load_model(path, args.device, args.max_length)

        scores_np = _predict_safe(model, pairs, args.batch_size)
        predictions = scores_np.argmax(axis=1).tolist()
        confidences = scores_np.max(axis=1).tolist()

        all_preds[label] = predictions
        all_confs[label] = confidences

        metrics = compute_metrics(rows, predictions)
        results[label] = {
            "model_path": path,
            "aggregate": metrics,
            "predictions": predictions,
            "confidences": confidences,
        }

        # Per-example CSV
        save_examples_csv(rows, predictions, confidences, label, output_dir)

        # Aggregate JSON
        json_path = output_dir / f"{label}.json"
        json_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Metrics JSON → {json_path}")
        print(f"  → acc={metrics['accuracy']:.3f}  "
              f"correct={metrics['correct']}/{metrics['total']}\n")

    # Comparison CSV
    cmp_path = output_dir / "comparison.csv"
    with cmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "entail_f1", "neutral_f1",
                         "contradiction_f1", "total"])
        for label in model_labels:
            m = results[label]["aggregate"]
            cf = m.get("confusion_matrix", {})
            writer.writerow([
                label, m["accuracy"],
                _f1(cf, "entailment"), _f1(cf, "neutral"),
                _f1(cf, "contradiction"), m["total"],
            ])
    print(f"Comparison CSV → {cmp_path}")

    # Disagreements (multi-model only)
    save_disagreements(rows, all_preds, all_confs, output_dir)

    # Print table
    print_comparison(results, model_labels)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate NLI models on a labelled test CSV"
    )
    p.add_argument("--models", type=str, required=True,
                   help="Comma-separated model paths.  Use 'pretrained' "
                        f"as a shortcut for {PRETRAINED_MODEL}")
    p.add_argument("--labels", type=str, default=None,
                   help="Optional comma-separated display names "
                        "(must match --models count)")
    p.add_argument("--eval-csv", type=str, required=True,
                   help="Path to test CSV (premise,hypothesis,label,...)")
    p.add_argument("--output-dir", type=str,
                   default="/tmp/nli_eval_results",
                   help="Output directory for results")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for inference")
    p.add_argument("--max-length", type=int, default=256,
                   help="Max token length for premise+hypothesis")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Device: cuda:0, cpu, ...")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(_parse_args()))
