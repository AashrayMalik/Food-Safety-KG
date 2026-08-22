#!/usr/bin/env python3
"""Standalone schema re-validator for previously-extracted knowledge triplets.

Reads ``schema_violations.jsonl``, re-validates each embedded triplet
against an updated schema (loaded from a JSON config file), and appends
newly-compliant triplets to ``triplets.jsonl`` and ``triplets.csv``.

No LLM re-extraction is required — only the schema file changes.

Typical workflow::

  1. Edit ``schema_config.json`` (add entity types, relations, or
     modify domain/range constraints).
  2. Run the violator::

       food_lab/bin/python src/extraction/schema_violator.py \\
           --violations    src/outputs/triplets/schema_violations.jsonl \\
           --schema        src/extraction/schema_config.json \\
           --triplets-jsonl src/outputs/triplets/triplets.jsonl \\
           --triplets-csv  src/outputs/triplets/triplets.csv

  3. Newly-compliant triplets are appended to both output files.
     The original ``schema_violations.jsonl`` is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from schema_validator import SchemaValidator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_existing_snippet_ids(jsonl_path: Path) -> set[str]:
    """Return set of snippet_ids already present in a JSONL file."""
    ids: set[str] = set()
    if not jsonl_path.exists():
        return ids
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("snippet_id", "")
                if sid:
                    ids.add(sid)
            except json.JSONDecodeError:
                pass
    return ids


def _load_existing_csv_keys(csv_path: Path) -> set[tuple[str, str, str, str]]:
    """Return set of (snippet_id, subject, predicate, object) already in CSV."""
    keys: set[tuple[str, str, str, str]] = set()
    if not csv_path.exists():
        return keys
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            keys.add((
                row.get("snippet_id", ""),
                row.get("subject", ""),
                row.get("predicate", ""),
                row.get("object", ""),
            ))
    return keys


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def run(
    violations_path: Path,
    schema_path: Path,
    triplets_jsonl: Path,
    triplets_csv: Path,
    *,
    chunks_csv: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Re-validate violations and append newly-compliant triplets.

    Returns a stats dict with keys: ``total``, ``passed``, ``still_invalid``,
    ``jsonl_appended``, ``csv_appended``, ``duplicate_snippet_ids``.
    """
    # Load schema
    validator = SchemaValidator(schema_path)
    print(f"Loaded schema from {schema_path}")
    print(f"  Entity types: {len(validator.entity_types)}")
    print(f"  Relations:    {len(validator.relations)}")
    print(f"  Domain/range: {len(validator.domain_range)} relations")

    # Load violations
    if not violations_path.exists():
        print(f"error: violations file not found: {violations_path}", file=sys.stderr)
        raise SystemExit(2)

    violations: list[dict[str, Any]] = []
    with violations_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                violations.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  Warning: skipping unparseable line in violations file",
                      file=sys.stderr)

    print(f"\nRead {len(violations)} violation entries from {violations_path}")

    # Load source metadata from chunks CSV if provided
    source_lookup: dict[str, dict[str, str]] = {}
    if chunks_csv and chunks_csv.exists():
        with chunks_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                source_lookup[row["snippet_id"]] = row
        print(f"  Loaded {len(source_lookup)} source-meta entries from {chunks_csv}")

    # Deduplicate triplets from violations (same triplet can have multiple
    # violation types — domain_mismatch + range_mismatch for example).
    # Use (snippet_id, subject, predicate, object) as key.
    seen_triplets: set[tuple[str, str, str, str]] = set()
    unique_triplets: list[dict[str, Any]] = []
    for v in violations:
        t = v.get("triplet", {})
        key = (
            v.get("snippet_id", ""),
            t.get("subject", ""),
            t.get("predicate", ""),
            t.get("object", ""),
        )
        if key not in seen_triplets:
            seen_triplets.add(key)
            unique_triplets.append(v)

    dup_count = len(violations) - len(unique_triplets)
    if dup_count:
        print(f"  Deduplicated: {dup_count} duplicate violation entries "
              f"→ {len(unique_triplets)} unique triplets to re-validate")

    # Re-validate each triplet against the new schema
    passing: list[dict[str, Any]] = []  # list of violation entries that now pass
    still_invalid = 0

    for v in unique_triplets:
        t = v.get("triplet", {})
        snippet_id = v.get("snippet_id", "")
        source_id = v.get("source_id", "")
        source_file = v.get("source_file", "")
        chunk_index = str(v.get("chunk_index", ""))
        triplet_index = v.get("triplet_index", 0)

        vios = validator.validate_triplet(
            t, triplet_index, snippet_id,
            source_id, source_file, chunk_index,
        )
        if not vios:
            passing.append(v)
        else:
            still_invalid += 1

    print(f"\nRe-validation results:")
    print(f"  Total unique triplets re-validated: {len(unique_triplets)}")
    print(f"  Now compliant (passing):            {len(passing)}")
    print(f"  Still invalid (unchanged):          {still_invalid}")
    if passing:
        fmt_vtypes: dict[str, int] = {}
        for v in unique_triplets:
            fmt_vtypes[v.get("violation_type", "?")] = fmt_vtypes.get(
                v.get("violation_type", "?"), 0
            ) + 1
        print(f"\n  Breakdown of now-passing by original violation type:")
        for vtype in sorted(set(v.get("violation_type", "?") for v in passing)):
            count = sum(
                1 for v in passing if v.get("violation_type") == vtype
            )
            print(f"    {vtype}: {count}")

    if not passing:
        print("\nNo triplets became compliant. Nothing to append.")
        return {
            "total": len(unique_triplets),
            "passed": 0,
            "still_invalid": still_invalid,
            "jsonl_appended": 0,
            "csv_appended": 0,
            "duplicate_snippet_ids": 0,
        }

    if dry_run:
        print(f"\n[dry-run] Would append {len(passing)} triplets to:")
        print(f"  JSONL → {triplets_jsonl}")
        print(f"  CSV   → {triplets_csv}")
        return {
            "total": len(unique_triplets),
            "passed": len(passing),
            "still_invalid": still_invalid,
            "jsonl_appended": 0,
            "csv_appended": 0,
            "duplicate_snippet_ids": 0,
        }

    # Group passing triplets by snippet_id for JSONL output
    # JSONL: one entry per snippet_id with all its newly-passing triplets
    passing_by_sid: dict[str, list[dict[str, Any]]] = {}
    for v in passing:
        sid = v.get("snippet_id", "")
        passing_by_sid.setdefault(sid, []).append(v.get("triplet", {}))

    # Check existing snippet_ids in triplets.jsonl for dedup
    existing_sids = _load_existing_snippet_ids(triplets_jsonl)
    existing_csv_keys = _load_existing_csv_keys(triplets_csv)

    jsonl_appended = 0
    csv_appended = 0
    dup_sid_count = 0

    csv_fields = [
        "snippet_id", "source_id", "source_file", "source_type",
        "chunk_index", "subject", "subject_type", "subject_id",
        "predicate", "object", "object_type", "object_id",
        "confidence", "evidence_span",
    ]

    # Append to JSONL
    for sid, trip_list in passing_by_sid.items():
        if sid in existing_sids:
            dup_sid_count += 1
            # Still append to CSV since individual triplets may be new
        else:
            entry = {
                "snippet_id": sid,
                "triplets": trip_list,
                "notes": f"schema_violator: {len(trip_list)} triplets now compliant after schema update",
            }
            if not dry_run:
                _append_jsonl(triplets_jsonl, entry)
            jsonl_appended += 1

    # Append to CSV
    with triplets_csv.open("a" if triplets_csv.exists() else "w",
                           encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        if not triplets_csv.exists() or triplets_csv.stat().st_size == 0:
            writer.writeheader()

        for v in passing:
            t = v.get("triplet", {})
            sid = v.get("snippet_id", "")
            src = source_lookup.get(sid, {})
            key = (
                sid,
                t.get("subject", ""),
                t.get("predicate", ""),
                t.get("object", ""),
            )
            if key in existing_csv_keys:
                continue  # already in CSV
            writer.writerow({
                "snippet_id": sid,
                "source_id": src.get("source_id", v.get("source_id", "")),
                "source_file": src.get("source_file", v.get("source_file", "")),
                "source_type": src.get("source_type", ""),
                "chunk_index": src.get("chunk_index", str(v.get("chunk_index", ""))),
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
            csv_appended += 1

    print(f"\nOutput:")
    if jsonl_appended:
        print(f"  {jsonl_appended} new chunk entries appended to {triplets_jsonl}")
    if dup_sid_count:
        print(f"  {dup_sid_count} snippet_ids already in JSONL (skipped)")
    csv_skipped = (
        sum(1 for v in passing
            if (v["snippet_id"],
                v.get("triplet", {}).get("subject", ""),
                v.get("triplet", {}).get("predicate", ""),
                v.get("triplet", {}).get("object", ""))
            in existing_csv_keys)
    )
    if csv_appended:
        print(f"  {csv_appended} new rows appended to {triplets_csv}")
    if csv_skipped:
        print(f"  {csv_skipped} triplets already in CSV (skipped)")

    return {
        "total": len(unique_triplets),
        "passed": len(passing),
        "still_invalid": still_invalid,
        "jsonl_appended": jsonl_appended,
        "csv_appended": csv_appended,
        "duplicate_snippet_ids": dup_sid_count,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-validate schema_violations.jsonl against an updated schema"
    )
    p.add_argument(
        "--violations", type=Path, required=True,
        help="Path to schema_violations.jsonl",
    )
    p.add_argument(
        "--schema", type=Path, required=True,
        help="Path to schema JSON config file (e.g. schema_config.json)",
    )
    p.add_argument(
        "--triplets-jsonl", type=Path, required=True,
        help="Path to triplets.jsonl (newly-passing entries appended here)",
    )
    p.add_argument(
        "--triplets-csv", type=Path, required=True,
        help="Path to triplets.csv (newly-passing rows appended here)",
    )
    p.add_argument(
        "--chunks-csv", type=Path, default=None,
        help="Optional: path to chunks CSV for source_type metadata "
             "(source_id/source_file from violations are used as fallback)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate only; do not write to output files",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.violations.exists():
        print(f"error: violations file not found: {args.violations}",
              file=sys.stderr)
        raise SystemExit(2)

    if not args.schema.exists():
        print(f"error: schema file not found: {args.schema}",
              file=sys.stderr)
        raise SystemExit(2)

    stats = run(
        violations_path=args.violations,
        schema_path=args.schema,
        triplets_jsonl=args.triplets_jsonl,
        triplets_csv=args.triplets_csv,
        chunks_csv=args.chunks_csv,
        dry_run=args.dry_run,
    )

    if stats["passed"] == 0:
        print("\nTip: to make more triplets compliant, add entity types, "
              "relations, or domain/range pairs to your schema JSON file.")


if __name__ == "__main__":
    main()
