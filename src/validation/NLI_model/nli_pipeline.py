#!/usr/bin/env python3
"""NLI entailment pipeline for FFLO knowledge-graph triplets.

Reads extracted triplets, assigns synthetic IDs to structural nodes,
verbalises them into natural-language hypotheses, and runs a cross-encoder
NLI model (pretrained or fine-tuned) against the source chunk text.

Modes:
    convert-gold    Convert Excel training pairs to CSV (one-time).
    pretrained      Run inference with a pretrained HF cross-encoder.
    finetuned       Run inference with a fine-tuned checkpoint.

Usage::

    # Convert Excel → CSV
    food_lab/bin/python src/validation/NLI_model/nli_pipeline.py \\
        --convert-gold src/validation/Golden_val/FFLO_HIL_correction_v6_NLI.xlsx \\
        --output-csv   src/validation/Golden_val/nli_train_pairs.csv

    # Pretrained inference (single GPU)
    food_lab/bin/python src/validation/NLI_model/nli_pipeline.py \\
        --mode pretrained \\
        --nli-model cross-encoder/nli-deberta-v3-small \\
        --triplets-path src/outputs/triplets/triplets.csv \\
        --chunks-csv src/data/FSSAI_docs/processed/chunks.csv \\
        --output-dir src/outputs/nli_validation \\
        --batch-size 64 --device cuda:0

    # Fine-tuned inference
    food_lab/bin/python src/validation/NLI_model/nli_pipeline.py \\
        --mode finetuned \\
        --nli-model model_checkpoints/run4_deberta_v3_small \\
        --triplets-path src/outputs/triplets/triplets.csv \\
        --chunks-csv src/data/FSSAI_docs/processed/chunks.csv \\
        --output-dir src/outputs/nli_validation \\
        --batch-size 128 --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from verbalise import verbalise
from synthetic_ids import assign_ids, resolve_label
from nli_model import NLIModel


# ---------------------------------------------------------------------------
# Convert Excel → CSV
# ---------------------------------------------------------------------------

def convert_excel_to_csv(excel_path: Path, output_csv: Path) -> int:
    """Read NLI_train_pairs sheet and write CSV."""
    try:
        import openpyxl
    except ImportError:
        print("error: openpyxl required.  pip install openpyxl", file=sys.stderr)
        return 1

    wb = openpyxl.load_workbook(str(excel_path))
    if "NLI_train_pairs" not in wb.sheetnames:
        print(f"error: sheet 'NLI_train_pairs' not found in {excel_path}",
              file=sys.stderr)
        return 1

    ws = wb["NLI_train_pairs"]
    headers = [str(ws.cell(1, c).value) for c in range(1, ws.max_column + 1)]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in range(2, ws.max_row + 1):
            row = [str(ws.cell(r, c).value or "")
                   for c in range(1, ws.max_column + 1)]
            writer.writerow(row)

    print(f"Wrote {ws.max_row - 1} rows to {output_csv}")
    return 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_chunks(csv_path: Path) -> dict[str, str]:
    """Return snippet_id → evidence_text."""
    lookup: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lookup[row["snippet_id"]] = row.get("evidence_text", "")
    return lookup


def _load_triplets(csv_path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            items.append(dict(row))
    return items


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    triplets: list[dict[str, Any]],
    evidence_texts: dict[str, str],
    model: NLIModel,
    batch_size: int = 32,
    premise_mode: str = "span",
) -> list[dict[str, Any]]:
    """Verbalise each triplet, then batch-infer with the NLI model.

    *premise_mode*: ``"span"`` uses ``evidence_span`` from the triplet
    row; ``"chunk"`` uses the full ``evidence_text`` from chunks.csv.
    """
    id_map, label_map = assign_ids(triplets, evidence_texts)

    premises: list[str] = []
    hypotheses: list[str] = []
    meta: list[dict[str, Any]] = []

    for t in triplets:
        sid = t.get("snippet_id", "")
        subj = t.get("subject", "")
        stype = t.get("subject_type", "")
        pred = t.get("predicate", "")
        obj = t.get("object", "")
        otype = t.get("object_type", "")

        subj_label = resolve_label(subj, sid, id_map, label_map)
        obj_label = resolve_label(obj, sid, id_map, label_map)
        entity_labels = {subj: subj_label, obj: obj_label}

        hypothesis = verbalise(subj, stype, pred, obj, otype, entity_labels)

        if premise_mode == "span":
            premise = t.get("evidence_span", "")
        else:
            premise = evidence_texts.get(sid, "")

        premises.append(premise)
        hypotheses.append(hypothesis)
        meta.append({
            "snippet_id": sid,
            "subject": subj, "subject_type": stype,
            "predicate": pred,
            "object": obj, "object_type": otype,
            "evidence_span": t.get("evidence_span", ""),
            "verbalisation": hypothesis,
        })

    # 3. Run NLI
    print(f"\nRunning NLI on {len(premises)} pairs (batch_size={batch_size})...")
    t0 = time.time()
    results = model.predict(premises, hypotheses, batch_size=batch_size)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(premises)/elapsed:.0f} pairs/s)")

    # 4. Attach metadata
    for i, r in enumerate(results):
        r.update(meta[i])
        r["nli_verdict"] = model.map_to_verdict(r["label"])

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _save_verdicts(
    results: list[dict[str, Any]], output_jsonl: Path,
) -> None:
    """Write per-triplet verdict dicts as one JSON object per line."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} verdicts to {output_jsonl}")


def _save_summary(
    results: list[dict[str, Any]], output_path: Path,
) -> None:
    """Write aggregate verdict counts (overall and per predicate) to JSON."""
    verdict_counter = Counter(r["nli_verdict"] for r in results)
    pred_counter: dict[str, Counter] = {}
    for r in results:
        p = r.get("predicate", "?")
        pred_counter.setdefault(p, Counter())[r["nli_verdict"]] += 1

    summary = {
        "total": len(results),
        "verdicts": dict(verdict_counter),
        "per_predicate": {
            p: dict(vc) for p, vc in sorted(pred_counter.items())
        },
    }
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"Summary written to {output_path}")
    print(f"Verdicts: {dict(verdict_counter)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NLI entailment pipeline for FFLO triplets"
    )

    # Mode selection
    p.add_argument("--convert-gold", type=Path, default=None,
                   help="Excel file to convert to CSV (skips inference)")
    p.add_argument("--output-csv", type=Path, default=None,
                   help="CSV output path for --convert-gold")

    p.add_argument("--mode", type=str, default="pretrained",
                   choices=["pretrained", "finetuned"],
                   help="Inference mode (default: pretrained)")

    # Model
    p.add_argument("--nli-model", type=str,
                   default="cross-encoder/nli-deberta-v3-small",
                   help="HF model ID or local checkpoint path")

    # Data
    p.add_argument("--triplets-path", type=Path,
                   help="Path to triplets.csv from extraction")
    p.add_argument("--chunks-csv", type=Path,
                   help="Path to chunks.csv for evidence_text lookup")

    # Output
    p.add_argument("--output-dir", type=Path, default=Path("src/outputs/nli_validation"),
                   help="Root output directory")

    # Inference
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for NLI inference")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only first N triplets (0 = all, for testing)")
    p.add_argument("--premise", type=str, default="span",
                   choices=["span", "chunk"],
                   help="What to use as NLI premise: evidence_span (default) "
                        "or full chunk evidence_text")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto, cpu, cuda:0, cuda:1, ...")

    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # --- convert mode ---
    if args.convert_gold:
        if not args.output_csv:
            print("error: --output-csv required with --convert-gold",
                  file=sys.stderr)
            return 1
        return convert_excel_to_csv(args.convert_gold, args.output_csv)

    # --- inference mode ---
    if not args.triplets_path or (
        args.premise == "chunk" and not args.chunks_csv
    ):
        print("error: --triplets-path and --chunks-csv required for inference"
              " (--chunks-csv only needed with --premise chunk)",
              file=sys.stderr)
        return 1

    if not args.triplets_path.exists():
        print(f"error: triplets CSV not found: {args.triplets_path}",
              file=sys.stderr)
        return 1
    if args.premise == "chunk":
        if not args.chunks_csv or not args.chunks_csv.exists():
            print(f"error: --chunks-csv required for --premise chunk",
                  file=sys.stderr)
            return 1

    # Load data
    print(f"Loading triplets from {args.triplets_path}")
    triplets = _load_triplets(args.triplets_path)
    if args.limit and args.limit < len(triplets):
        triplets = triplets[:args.limit]
        print(f"  Limited to {args.limit} triplets")
    print(f"  {len(triplets)} triplets")

    evidence: dict[str, str] = {}
    if args.chunks_csv and args.chunks_csv.exists():
        print(f"Loading chunks from {args.chunks_csv}")
        evidence = _load_chunks(args.chunks_csv)
        print(f"  {len(evidence)} chunks")
    print(f"  {len(evidence)} chunks")

    # Load model
    model_slug = args.nli_model.replace("/", "__").replace("\\", "__")
    print(f"\nLoading NLI model: {args.nli_model}")
    model = NLIModel(args.nli_model, device=args.device, max_length=512)

    # Run with evidence-span premise (matches training data format)
    results = run_inference(triplets, evidence, model,
                            batch_size=args.batch_size,
                            premise_mode=args.premise)

    # Save
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / model_slug / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    _save_verdicts(results, run_dir / "verdicts.jsonl")
    _save_summary(results, run_dir / "summary.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
