#!/usr/bin/env python3
"""Apply actionable LLM judge verdicts to fix triplets.

Reads judgment JSONL files from a judge run directory, then for each
chunk's triplets applies one of three actions per judged triplet:

  KEEP    — verdict is fine as-is (entailed, partially_entailed*)
  FIX     — verdict has a concrete, safe correction (predicate/type/
            direction swap) and the required correction fields are
            present
  DROP    — verdict says the triplet is not supported by the text
            (not_entailed) or the SCHEMA_MISMATCH triplet is
            unsupported/malformed
  REVIEW  — verdict indicates a problem but there's no safe automatic
            fix (uncertain, judge_error, a "wrong_relation"/
            "direction_reversed" verdict missing its correction
            fields, needs_schema_extension, or an index-identity
            mismatch between the judgment and the triplet it points at)

REVIEW items are never merged into the fixed output and never silently
dropped — they're written to a separate file so nothing gets lost.

Before applying any index-based fix, the triplet at judgment.triplet_index
is checked against the judgment's own recorded subject/predicate/object.
If they don't match, the judgment and the triplet have drifted out of
alignment (e.g. triplet_index computed over a branch-filtered list
instead of the full per-chunk list) — that judgment is routed to REVIEW
instead of silently mutating the wrong triplet.

The original files are never modified — always writes new outputs.

Usage::

    food_lab/bin/python src/extraction/apply_judgments.py \\
        --run-dir  outputs/validation/fssai_docs/deepseek-v4-pro/run_20260630_214815/ \\
        --triplets-jsonl outputs/triplets/triplets.jsonl \\
        --triplets-csv   outputs/triplets/triplets.csv \\
        --output-jsonl   outputs/triplets/triplets_fixed.jsonl \\
        --output-csv     outputs/triplets/triplets_fixed.csv \\
        --review-csv     outputs/triplets/needs_review.csv \\
        --schema-extension-csv outputs/triplets/schema_extension_candidates.csv \\
        --chunks-csv     data/FSSAI_docs/processed/chunks.csv \\
        --confidence-threshold 0.7 \\
        [--drop-partial] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


BRANCH_FILES = {
    "zero_triplet": "zero_triplet_judgments.jsonl",
    "schema_mismatch": "schema_mismatch_judgments.jsonl",
    "schema_invalid": "schema_invalid_judgments.jsonl",
    "schema_valid": "schema_valid_judgments.jsonl",
}

# Verdicts meaning "the triplet as extracted is not supported" -> DROP.
DROP_VERDICTS = frozenset({"not_entailed", "unsupported", "malformed"})

# Verdicts meaning "fine, keep it" with no edit needed.
KEEP_VERDICTS = frozenset({"entailed", "partially_entailed"})

# Verdicts with a concrete correction to apply.
FIX_VERDICTS = frozenset({
    "maps_to_existing",
    "entailed_wrong_types",
    "entailed_wrong_relation",
    "entailed_direction_reversed",
})

CSV_FIELDS = [
    "snippet_id", "source_id", "source_file", "source_type",
    "chunk_index", "subject", "subject_type", "subject_id",
    "predicate", "object", "object_type", "object_id",
    "confidence", "evidence_span",
]


def load_judgments(run_dir: Path) -> list[dict[str, Any]]:
    judgments: list[dict[str, Any]] = []
    for branch, fname in BRANCH_FILES.items():
        path = run_dir / fname
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec["_branch"] = branch
                judgments.append(rec)
    return judgments


def load_triplets(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    if not jsonl_path.exists():
        return entries
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("snippet_id", "")
            if sid:
                entries[sid] = obj
    return entries


def load_source_meta(chunks_csv: Path | None) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    if not chunks_csv or not chunks_csv.exists():
        return lookup
    with chunks_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            lookup[row["snippet_id"]] = row
    return lookup


def _identity_matches(t: dict[str, Any], j: dict[str, Any]) -> bool:
    """Guard against index misalignment: confirm the triplet at
    triplet_index is actually the one the judgment describes."""
    for field in ("subject", "predicate", "object"):
        jval = j.get(field)
        if jval is not None and t.get(field) != jval:
            return False
    return True


def apply_fixes(
    judgments: list[dict[str, Any]],
    triplets: dict[str, dict[str, Any]],
    confidence_threshold: float,
    keep_partial: bool,
) -> tuple[Counter, list[dict[str, Any]]]:
    """Mutates `triplets` in place (fixes) and marks drops. Returns
    (action counts, review records)."""
    counts = Counter()
    review: list[dict[str, Any]] = []

    # (snippet_id, triplet_index) -> "drop", collected first so we can
    # filter tlist in one pass at the end without shifting indices for
    # judgments still to be applied.
    to_drop: set[tuple[str, int]] = set()

    for j in judgments:
        verdict = j.get("judge_verdict", "")
        branch = j.get("_branch")
        sid = j.get("snippet_id", "")
        tindex = j.get("triplet_index")
        conf = j.get("judge_confidence")

        # --- zero_triplet branch: only adds triplets, never touches existing ---
        if verdict == "missed_relation":
            if conf is None or conf < confidence_threshold:
                counts["skipped_confidence"] += 1
                continue
            missed = j.get("missed_triplets")
            if not isinstance(missed, list) or not missed:
                counts["skipped_missing_triplet"] += 1
                continue
            entry = triplets.setdefault(sid, {"snippet_id": sid, "triplets": []})
            added = 0
            for mt in missed:
                if not isinstance(mt, dict):
                    continue
                mt.setdefault("confidence", 0.5)
                mt.setdefault("evidence_span", "")
                entry["triplets"].append(mt)
                added += 1
            if added:
                counts["missed_relation"] += added
            continue

        if verdict not in DROP_VERDICTS | KEEP_VERDICTS | FIX_VERDICTS:
            # uncertain, judge_error, needs_schema_extension, out_of_schema_only, etc.
            review.append({**j, "review_reason": f"verdict={verdict}, no automatic action"})
            counts["review"] += 1
            continue

        if tindex is None:
            review.append({**j, "review_reason": "no triplet_index on record"})
            counts["review"] += 1
            continue

        entry = triplets.get(sid)
        if not entry:
            counts["skipped_missing_triplet"] += 1
            continue
        tlist = entry.get("triplets", [])
        if tindex < 0 or tindex >= len(tlist):
            review.append({**j, "review_reason": "triplet_index out of range"})
            counts["review"] += 1
            continue

        t = tlist[tindex]

        if not _identity_matches(t, j):
            review.append({**j, "review_reason": "index/identity mismatch — "
                            "triplet at this index does not match what the "
                            "judgment describes"})
            counts["index_mismatch"] += 1
            continue

        if verdict in DROP_VERDICTS:
            to_drop.add((sid, tindex))
            counts[f"drop:{verdict}"] += 1
            continue

        if verdict in KEEP_VERDICTS:
            if verdict == "partially_entailed" and not keep_partial:
                to_drop.add((sid, tindex))
                counts["drop:partially_entailed"] += 1
            else:
                counts[f"keep:{verdict}"] += 1
            continue

        # verdict in FIX_VERDICTS
        if conf is not None and conf < confidence_threshold:
            counts["skipped_confidence"] += 1
            continue

        if verdict == "maps_to_existing":
            rel = j.get("existing_relation")
            if not rel:
                review.append({**j, "review_reason": "maps_to_existing with no existing_relation"})
                counts["review"] += 1
                continue
            t["predicate"] = rel
            t.pop("mismatch_note", None)
            counts["fix:maps_to_existing"] += 1

        elif verdict == "entailed_wrong_types":
            fixed = False
            for field, jf in (("subject_type", "correct_subject_type"),
                               ("object_type", "correct_object_type")):
                val = j.get(jf)
                if val:
                    t[field] = val
                    fixed = True
            if fixed:
                counts["fix:entailed_wrong_types"] += 1
            else:
                review.append({**j, "review_reason": "entailed_wrong_types with no correction fields"})
                counts["review"] += 1

        elif verdict == "entailed_wrong_relation":
            rel = j.get("correct_relation")
            if not rel:
                review.append({**j, "review_reason": "entailed_wrong_relation with no correct_relation"})
                counts["review"] += 1
            else:
                t["predicate"] = rel
                for field, jf in (("subject_type", "correct_subject_type"),
                                   ("object_type", "correct_object_type")):
                    val = j.get(jf)
                    if val:
                        t[field] = val
                counts["fix:entailed_wrong_relation"] += 1

        elif verdict == "entailed_direction_reversed":
            subj, obj = t.get("subject"), t.get("object")
            stype, otype = t.get("subject_type"), t.get("object_type")
            t["subject"], t["object"] = obj, subj
            t["subject_type"], t["object_type"] = otype, stype
            rel = j.get("correct_relation")
            if rel:
                t["predicate"] = rel
            for field, jf in (("subject_type", "correct_subject_type"),
                               ("object_type", "correct_object_type")):
                val = j.get(jf)
                if val:
                    t[field] = val
            counts["fix:entailed_direction_reversed"] += 1

        tlist[tindex] = t
        entry["triplets"] = tlist
        triplets[sid] = entry

    # Second pass: actually remove dropped triplets.
    for sid, tindex in to_drop:
        entry = triplets.get(sid)
        if not entry:
            continue
        entry["triplets"] = [
            t for i, t in enumerate(entry.get("triplets", [])) if i != tindex
        ]

    return counts, review


def write_triplets(
    triplets: dict[str, dict[str, Any]],
    source_meta: dict[str, dict[str, str]],
    output_jsonl: Path,
    output_csv: Path,
    dry_run: bool,
) -> int:
    triplets_list = [t for t in triplets.values() if t.get("triplets")]

    if dry_run:
        total = sum(len(t["triplets"]) for t in triplets_list)
        print(f"[dry-run] Would write {len(triplets_list)} chunk entries, "
              f"{total} triplets to {output_jsonl} / {output_csv}", flush=True)
        return total

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as fh:
        for entry in triplets_list:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    row_count = 0
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for entry in triplets_list:
            sid = entry.get("snippet_id", "")
            src = source_meta.get(sid, {})
            for t in entry.get("triplets", []):
                writer.writerow({
                    "snippet_id": sid,
                    "source_id": src.get("source_id", ""),
                    "source_file": src.get("source_file", ""),
                    "source_type": src.get("source_type", ""),
                    "chunk_index": src.get("chunk_index", ""),
                    "subject": t.get("subject", ""),
                    "subject_type": t.get("subject_type", ""),
                    "subject_id": t.get("subject_id", ""),
                    "predicate": t.get("predicate", ""),
                    "object": t.get("object", ""),
                    "object_type": t.get("object_type", ""),
                    "object_id": t.get("object_id", ""),
                    "confidence": t.get("confidence", ""),
                    "evidence_span": t.get("evidence_span", ""),
                })
                row_count += 1
    return row_count


def write_review(review: list[dict[str, Any]], review_csv: Path, dry_run: bool) -> int:
    if dry_run:
        print(f"[dry-run] Would write {len(review)} rows to {review_csv}", flush=True)
        return len(review)
    if not review:
        return 0
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in review for k in r.keys()})
    with review_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in review:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                              for k, v in r.items()})
    return len(review)


def run(
    run_dir: Path,
    triplets_jsonl: Path,
    output_jsonl: Path,
    output_csv: Path,
    review_csv: Path,
    *,
    chunks_csv: Path | None = None,
    confidence_threshold: float = 0.7,
    keep_partial: bool = True,
    dry_run: bool = False,
) -> dict[str, int]:
    print(f"Loading judgments from {run_dir} ...", flush=True)
    judgments = load_judgments(run_dir)
    print(f"  {len(judgments)} total judgment records loaded", flush=True)

    print(f"\nLoading triplets from {triplets_jsonl} ...", flush=True)
    triplets = load_triplets(triplets_jsonl)
    orig_total = sum(len(t.get("triplets", [])) for t in triplets.values())
    print(f"  {len(triplets)} chunk entries, {orig_total} total triplets", flush=True)

    source_meta = load_source_meta(chunks_csv)
    if source_meta:
        print(f"  {len(source_meta)} source-meta entries from {chunks_csv}", flush=True)

    print(f"\nApplying keep/fix/drop/review (confidence >= {confidence_threshold}, "
          f"keep_partial={keep_partial}) ...", flush=True)
    counts, review = apply_fixes(judgments, triplets, confidence_threshold, keep_partial)

    print("\nActions:", flush=True)
    for k in sorted(counts.keys()):
        print(f"  {k}: {counts[k]}", flush=True)

    row_count = write_triplets(triplets, source_meta, output_jsonl, output_csv, dry_run)
    review_count = write_review(review, review_csv, dry_run)

    if not dry_run:
        final_total = sum(len(t.get("triplets", [])) for t in triplets.values())
        print(f"\nWritten:", flush=True)
        print(f"  {output_jsonl}", flush=True)
        print(f"  {output_csv} ({row_count} rows)", flush=True)
        print(f"  {review_csv} ({review_count} rows needing manual review)", flush=True)
        print(f"\n  {orig_total} original -> {final_total} final "
              f"({orig_total - final_total} dropped, {review_count} held for review)",
              flush=True)

    return {"rows": row_count, "review": review_count}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply LLM judge verdicts to fix triplets")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--triplets-jsonl", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--review-csv", type=Path, required=True,
                   help="Where to write judgments that need manual review "
                        "(uncertain, judge_error, missing corrections, "
                        "index/identity mismatches)")
    p.add_argument("--chunks-csv", type=Path, default=None)
    p.add_argument("--confidence-threshold", type=float, default=0.7)
    p.add_argument("--drop-partial", action="store_true",
                   help="Drop partially_entailed triplets instead of keeping them")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.run_dir.is_dir():
        print(f"error: run directory not found: {args.run_dir}", file=sys.stderr)
        raise SystemExit(2)
    if not args.triplets_jsonl.exists():
        print(f"error: triplets JSONL not found: {args.triplets_jsonl}", file=sys.stderr)
        raise SystemExit(2)

    run(
        run_dir=args.run_dir,
        triplets_jsonl=args.triplets_jsonl,
        output_jsonl=args.output_jsonl,
        output_csv=args.output_csv,
        review_csv=args.review_csv,
        chunks_csv=args.chunks_csv,
        confidence_threshold=args.confidence_threshold,
        keep_partial=not args.drop_partial,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()