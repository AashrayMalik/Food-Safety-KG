#!/usr/bin/env python3
"""Apply a canonicalisation map to a CSV of triplets.

Reads a ``canonical_map.json`` and resolves ``predicate``,
``subject_type``, and ``object_type`` columns in any triplet CSV.

Usage::

    python src/validation/unconstrained/apply_canonical.py \\
        --input  nli_llm_judge_entailed.csv \\
        --map    canonical_map.json \\
        --output entailed_canonicalised.csv
"""

import argparse
import csv
import sys
from pathlib import Path

# Import from sibling extraction module
_here = Path(__file__).resolve().parent
_ext = _here.parent.parent / "extraction"
if str(_ext) not in sys.path:
    sys.path.insert(0, str(_ext))
from normalize import load_canonical_map, apply_canonical_map


def main() -> int:
    args = _parse_args()

    merges = load_canonical_map(Path(args.map))
    if not merges:
        print(f"Warning: empty or missing map file: {args.map}")
        return 1

    print(f"Loaded {len(merges)} canonical mappings")

    with open(args.input, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)

    # Collect columns to resolve
    predicates = [r.get("predicate", "") for r in rows]
    stypes = [r.get("subject_type", "") for r in rows]
    otypes = [r.get("object_type", "") for r in rows]

    pred_canon = apply_canonical_map(predicates, merges)
    st_canon = apply_canonical_map(stypes, merges)
    ot_canon = apply_canonical_map(otypes, merges)

    changed = 0
    with open(args.output, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            if row["predicate"] != pred_canon[i]:
                changed += 1
                row["predicate"] = pred_canon[i]
            if row["subject_type"] != st_canon[i]:
                changed += 1
                row["subject_type"] = st_canon[i]
            if row["object_type"] != ot_canon[i]:
                changed += 1
                row["object_type"] = ot_canon[i]
            writer.writerow(row)

    print(f"Resolved {changed} field changes across {len(rows)} rows")
    print(f"→ {args.output}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply canonicalisation map to a triplet CSV"
    )
    p.add_argument("--input", required=True, help="Input CSV with triplets")
    p.add_argument("--map", required=True, help="canonical_map.json")
    p.add_argument("--output", required=True, help="Output CSV path")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())
