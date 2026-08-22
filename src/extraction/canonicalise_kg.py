#!/usr/bin/env python3
"""Multi-phase canonicalisation of extracted knowledge-graph triplets.

Phase 1 — merge surface-form duplicate entities to canonical IDs
          (fuzzy + graph neighbour check + chemical-type guard)
Phase 2 — resolve genuine type conflicts vs. mistyped noise
Phase 3 — dedup near-identical triples from chunk overlap

Outputs: canonical CSV, entity map, type conflicts, report.

Usage::

    food_lab/bin/python src/extraction/canonicalise_kg.py \\
        --input  src/outputs/triplets/qwen/triplets_verified.csv \\
        --output-dir src/outputs/triplets/qwen/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
from rapidfuzz.distance import Levenshtein

# ---------------------------------------------------------------------------
# Type hierarchy — loaded from schema_config.json's ``subclass_of`` map so
# canonicalisation, extraction, and RDF output share a single source of
# truth.  Keys/values are unprefixed (e.g. "SyntheticAdulterant" → "Adulterant").
# ---------------------------------------------------------------------------

_PREFIX_RE = re.compile(r"^(fflo|fkg|fso|prov|lkif|ssn|xsd):")


def _strip_prefix(name: str) -> str:
    return _PREFIX_RE.sub("", name.strip())


_SCHEMA_CONFIG_PATH = Path(__file__).resolve().parent / "schema_config.json"


def _load_type_hierarchy(path: Path) -> dict[str, set[str]]:
    """Return child → direct parents (unprefixed) from schema_config.json."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    hierarchy: dict[str, set[str]] = {}
    for child, parents in raw.get("subclass_of", {}).items():
        c = _strip_prefix(child)
        if isinstance(parents, (list, tuple, set)):
            hierarchy[c] = {_strip_prefix(p) for p in parents}
        else:
            hierarchy[c] = {_strip_prefix(parents)}
    return hierarchy


_TYPE_HIERARCHY: dict[str, set[str]] = _load_type_hierarchy(_SCHEMA_CONFIG_PATH)


def _ancestors_of(child: str) -> set[str]:
    """Transitive ancestors of ``child`` (direct parents + grandparents + …)."""
    seen: set[str] = set()
    stack = list(_TYPE_HIERARCHY.get(child, set()))
    while stack:
        parent = stack.pop()
        if parent in seen:
            continue
        seen.add(parent)
        stack.extend(_TYPE_HIERARCHY.get(parent, set()))
    return seen

_CHEMICAL_TYPES = frozenset({
    "ChemicalIngredient", "FoodAdditive", "Microorganism",
    "Adulterant", "SyntheticAdulterant", "NaturalAdulterant",
    "ContaminantAdulterant", "Preservative", "Antioxidant",
    "Emulsifier", "Stabilizer", "AcidityRegulator",
    "FlourTreatmentAgent", "Sequestrant", "HumectantAdditive",
    "Colourant", "SweeteningAgent", "ProcessingAid",
})

def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip common punctuation variants."""
    t = text.strip().lower()
    t = re.sub(r"[,\-—\s]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _related_types(t1: str, t2: str) -> bool:
    """True if t1 and t2 are the same or in an ancestor relationship."""
    if t1 == t2:
        return True
    t1 = _strip_prefix(t1)
    t2 = _strip_prefix(t2)
    if t1 == t2:
        return True
    return t2 in _ancestors_of(t1) or t1 in _ancestors_of(t2)


# ---------------------------------------------------------------------------
# Phase 1 — entity canonicalisation
# ---------------------------------------------------------------------------

def _build_graph(rows: list[dict[str, str]]) -> nx.DiGraph:
    g = nx.DiGraph()
    for r in rows:
        s = r["_norm_subject"]
        o = r["_norm_object"]
        if s and o:
            g.add_edge(s, o)
    return g


def _is_chemical_entity(
    norm_name: str, entity_types: dict[str, str]
) -> bool:
    """Use the assigned type(s) to decide if an entity is chemical-type."""
    t = entity_types.get(norm_name, "")
    t = _strip_prefix(t)
    if t in _CHEMICAL_TYPES:
        return True
    return any(parent in _CHEMICAL_TYPES for parent in _ancestors_of(t))


def _lev_threshold(name: str) -> float:
    """Scale Levenshtein threshold to string length."""
    n = len(name)
    if n <= 3:
        return 0.0
    elif n <= 5:
        return 0.05
    elif n <= 7:
        return 0.10
    return 0.15


def _block_key(name: str) -> tuple[int, str]:
    """Block entities by length bucket + first char for efficient matching."""
    n = len(name)
    bucket = n // 3
    first = name[0] if name else ""
    return (bucket, first)


def canonicalise_entities(
    rows: list[dict[str, str]],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Phase 1: merge surface-form duplicates to canonical IDs.

    Returns (norm_name → canonical_id map, stats dict).
    """
    t0 = time.time()
    print("Phase 1: canonicalising entity surface forms ...", flush=True)

    # Collect normalized entities and their types
    entity_types: dict[str, str] = {}
    entity_raws: dict[str, Counter] = defaultdict(Counter)

    for r in rows:
        ns = r["_norm_subject"]
        no = r["_norm_object"]
        st = _strip_prefix(r.get("subject_type", ""))
        ot = _strip_prefix(r.get("object_type", ""))
        rs = r.get("subject", "").strip()
        ro = r.get("object", "").strip()

        if ns:
            entity_types[ns] = entity_types.get(ns) or st or ""
            entity_raws[ns][rs] += 1
        if no:
            entity_types[no] = entity_types.get(no) or ot or ""
            entity_raws[no][ro] += 1

    names = sorted(entity_types.keys())
    print(f"  {len(names)} unique normalised entity labels", flush=True)

    # Build graph for neighbor check
    g = _build_graph(rows)
    degrees = dict(g.degree())

    # Block entities
    blocks: dict[tuple[int, str], list[str]] = defaultdict(list)
    for name in names:
        blocks[_block_key(name)].append(name)

    # Merge map: norm_name → canonical_id
    merge: dict[str, str] = {}
    stats = {"exact_matches": 0, "fuzzy_matches": 0, "singletons_merged": 0,
             "chemical_blocked": 0, "neighbor_blocked": 0, "total_merged": 0}

    for block in blocks.values():
        for i, a in enumerate(block):
            if a in merge:
                continue
            for b in block[i + 1:]:
                if b in merge:
                    continue
                na = Levenshtein.normalized_distance(a, b)
                threshold = min(_lev_threshold(a), _lev_threshold(b))
                if na > threshold:
                    continue
                if na == 0.0:
                    merge[b] = a
                    stats["exact_matches"] += 1
                    continue

                # Chemical-type guard
                chemo_a = _is_chemical_entity(a, entity_types)
                chemo_b = _is_chemical_entity(b, entity_types)
                if chemo_a or chemo_b:
                    stats["chemical_blocked"] += 1
                    continue

                # Singleton fallback: if one entity has degree 0, merge
                deg_a = degrees.get(a, 0)
                deg_b = degrees.get(b, 0)
                if deg_a == 0 or deg_b == 0:
                    merge[b] = a
                    stats["singletons_merged"] += 1
                    continue

                # Neighbor check
                neighbors_a = set(g.predecessors(a)) | set(g.successors(a))
                neighbors_b = set(g.predecessors(b)) | set(g.successors(b))
                if neighbors_a & neighbors_b:
                    merge[b] = a
                    stats["fuzzy_matches"] += 1
                else:
                    stats["neighbor_blocked"] += 1

    # Resolve merge chains
    for name in names:
        if name in merge:
            root = merge[name]
            while root in merge:
                root = merge[root]
            merge[name] = root

    # Build canonical_id → best display form
    canonical_id_map: dict[str, str] = {}
    for name in names:
        cid = merge.get(name, name)
        if cid not in canonical_id_map:
            raw_counter = entity_raws[name]
            best_raw = raw_counter.most_common(1)[0][0] if raw_counter else name
            canonical_id_map[cid] = best_raw

    # Final map: norm_name → canonical_id (display form)
    final_map: dict[str, str] = {}
    for name in names:
        cid = merge.get(name, name)
        final_map[name] = canonical_id_map.get(cid, name)

    stats["total_merged"] = sum(
        1 for n in names if merge.get(n, n) != n
    )
    stats["elapsed_s"] = round(time.time() - t0, 1)
    print(f"  {stats['total_merged']} entities merged "
          f"({stats['exact_matches']} exact, {stats['fuzzy_matches']} fuzzy, "
          f"{stats['singletons_merged']} singleton) "
          f"({stats['elapsed_s']}s)", flush=True)
    print(f"  Blocked: {stats['chemical_blocked']} chemical, "
          f"{stats['neighbor_blocked']} neighbor", flush=True)

    return final_map, stats


# ---------------------------------------------------------------------------
# Phase 2 — type conflict resolution
# ---------------------------------------------------------------------------

def resolve_types(
    rows: list[dict[str, str]],
    entity_map: dict[str, str],
    confidence_threshold: float = 0.8,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Phase 2: resolve type conflicts per canonical entity.

    Returns (canonical_id → verbatim_prefixed_resolved_type, conflicts).
    The type_map values are verbatim prefixed strings (e.g. 'fkg:Food'),
    never stripped or constructed keys.
    """
    t0 = time.time()
    print("\nPhase 2: resolving type conflicts ...", flush=True)

    cid_stripped_types: dict[str, Counter] = defaultdict(Counter)
    cid_verbatim_types: dict[str, Counter] = defaultdict(Counter)
    cid_counts: dict[str, int] = defaultdict(int)

    for r in rows:
        ns = r["_norm_subject"]
        no = r["_norm_object"]
        st_v = r.get("subject_type", "").strip()
        ot_v = r.get("object_type", "").strip()
        st = _strip_prefix(st_v)
        ot = _strip_prefix(ot_v)

        scid = entity_map.get(ns, ns)
        ocid = entity_map.get(no, no)

        if st:
            cid_stripped_types[scid][st] += 1
            cid_verbatim_types[scid][st_v] += 1
            cid_counts[scid] += 1
        if ot:
            cid_stripped_types[ocid][ot] += 1
            cid_verbatim_types[ocid][ot_v] += 1
            cid_counts[ocid] += 1

    resolved: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []

    for cid, stripped_counter in sorted(cid_stripped_types.items()):
        unique_types = list(stripped_counter.keys())
        if len(unique_types) == 1:
            resolved[cid] = cid_verbatim_types[cid].most_common(1)[0][0]
            continue

        total = cid_counts[cid]
        most_common = stripped_counter.most_common()

        # Check if all types are related (hierarchy)
        all_related = True
        for t1 in unique_types:
            for t2 in unique_types:
                if t1 != t2 and not _related_types(t1, t2):
                    all_related = False
                    break
            if not all_related:
                break

        if all_related:
            best = unique_types[0]
            for t in unique_types[1:]:
                if best in _ancestors_of(t):
                    best = t
            # Find verbatim form of the winning stripped type
            verbatim_counter = cid_verbatim_types[cid]
            for vtype in verbatim_counter:
                if _strip_prefix(vtype) == best:
                    resolved[cid] = vtype
                    break
            else:
                resolved[cid] = best  # fallback
            continue

        # Dominant type ≥ threshold
        dominant_type, dominant_count = most_common[0]
        if dominant_count / total >= confidence_threshold:
            verbatim_counter = cid_verbatim_types[cid]
            for vtype in verbatim_counter:
                if _strip_prefix(vtype) == dominant_type:
                    resolved[cid] = vtype
                    break
            else:
                resolved[cid] = dominant_type
            continue

        # True conflict
        verbatim_types_for_conflict: dict[str, str] = {}
        for t, _ in most_common:
            for vtype in cid_verbatim_types[cid]:
                if _strip_prefix(vtype) == t and t not in verbatim_types_for_conflict:
                    verbatim_types_for_conflict[t] = vtype
                    break
        conflicts.append({
            "entity": cid,
            "types": {verbatim_types_for_conflict.get(t, t): c
                       for t, c in most_common},
            "total_occurrences": total,
        })

    stats = {"resolved": len(resolved), "conflicts": len(conflicts),
             "elapsed_s": round(time.time() - t0, 1)}
    print(f"  {stats['resolved']} types resolved, "
          f"{stats['conflicts']} conflicts flagged "
          f"({stats['elapsed_s']}s)", flush=True)

    return resolved, conflicts, stats


# ---------------------------------------------------------------------------
# Phase 3 — dedup near-identical triples
# ---------------------------------------------------------------------------

def _jaccard_tokens(s1: str, s2: str) -> float:
    t1 = set(re.findall(r"\w+", s1.lower()))
    t2 = set(re.findall(r"\w+", s2.lower()))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def dedup_triples(
    rows: list[dict[str, str]],
    entity_map: dict[str, str],
    type_map: dict[str, str],
) -> list[dict[str, str]]:
    """Phase 3: dedup near-identical triples from chunk overlap.

    Type_map values are verbatim prefixed type strings — apply them
    directly without constructing new strings.
    """
    t0 = time.time()
    print("\nPhase 3: deduplicating triples ...", flush=True)

    # --- Step 3a: apply canonical entity IDs + resolved types ---
    conflict_entities = set()  # entities with unresolved type conflicts
    for cid, vtype in type_map.items():
        if not vtype:  # empty string means conflict (never resolved)
            conflict_entities.add(cid)

    for r in rows:
        ns = r["_norm_subject"]
        no = r["_norm_object"]
        scid = entity_map.get(ns, ns)
        ocid = entity_map.get(no, no)
        r["subject"] = scid
        r["object"] = ocid

        old_st = r.get("subject_type", "").strip()
        old_ot = r.get("object_type", "").strip()

        if scid in type_map and type_map[scid]:
            new_st = type_map[scid]
            if old_st and _strip_prefix(old_st) != _strip_prefix(new_st):
                r["_type_fix"] = f"auto:{old_st}→{new_st}"
            r["subject_type"] = new_st
        elif scid in conflict_entities:
            r["_type_fix"] = f"conflict:{scid}"

        if ocid in type_map and type_map[ocid]:
            new_ot = type_map[ocid]
            if old_ot and _strip_prefix(old_ot) != _strip_prefix(new_ot):
                existing = r.get("_type_fix", "")
                r["_type_fix"] = "; ".join(
                    p for p in [existing, f"auto:{old_ot}→{new_ot}"] if p
                )
            r["object_type"] = new_ot
        elif ocid in conflict_entities:
            existing = r.get("_type_fix", "")
            r["_type_fix"] = "; ".join(
                p for p in [existing, f"conflict:{ocid}"] if p
            )

    # --- Step 3b: exact (s,p,o) dedup ---
    exact_groups: dict[tuple[str, str, str], list[dict[str, str]]] = (
        defaultdict(list)
    )
    for r in rows:
        key = (r["subject"], r.get("predicate", "").strip(), r["object"])
        exact_groups[key].append(r)

    deduped_exact: list[dict[str, str]] = []
    for key, grp in exact_groups.items():
        if len(grp) == 1:
            deduped_exact.append(grp[0])
        else:
            best = max(grp, key=lambda r: float(r.get("confidence", 0) or 0))
            spans = []
            sids = []
            for r in grp:
                es = r.get("evidence_span", "").strip()
                sid = r.get("snippet_id", "").strip()
                if es and es not in spans:
                    spans.append(es)
                if sid and sid not in sids:
                    sids.append(sid)
            best["evidence_span"] = " | ".join(spans[:5])
            if sids:
                best["snippet_id"] = sids[0] if len(sids) == 1 else f"{sids[0]} (+{len(sids)-1})"
            deduped_exact.append(best)

    dropped_exact = len(rows) - len(deduped_exact)
    print(f"  Step 3a (exact s,p,o): {len(rows)} → {len(deduped_exact)} "
          f"({dropped_exact} dropped)", flush=True)

    # --- Step 3c: Jaccard-based evidence-span dedup ---
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = (
        defaultdict(list)
    )
    for r in deduped_exact:
        key = (r["subject"], r.get("subject_type", "").strip(),
               r.get("predicate", "").strip(), r["object"],
               r.get("object_type", "").strip())
        groups[key].append(r)

    deduped: list[dict[str, str]] = []
    for key, grp in groups.items():
        if len(grp) == 1:
            deduped.append(grp[0])
            continue

        best = max(grp, key=lambda r: float(r.get("confidence", 0) or 0))
        spans: list[str] = []
        sids: list[str] = []
        for r in grp:
            es = r.get("evidence_span", "").strip()
            sid = r.get("snippet_id", "").strip()
            if es and es not in spans:
                spans.append(es)
            if sid and sid not in sids:
                sids.append(sid)

        spans_deduped: list[str] = []
        for s in spans:
            is_dup = False
            for existing in spans_deduped:
                if _jaccard_tokens(s, existing) > 0.5:
                    is_dup = True
                    break
            if not is_dup:
                spans_deduped.append(s)

        best["evidence_span"] = " | ".join(spans_deduped[:5])
        if sids:
            best["snippet_id"] = sids[0] if len(sids) == 1 else f"{sids[0]} (+{len(sids)-1})"

        deduped.append(best)

    dropped = len(rows) - len(deduped)
    stats = {"before": len(rows), "after": len(deduped),
             "dropped_exact": dropped_exact, "dropped_total": dropped,
             "elapsed_s": round(time.time() - t0, 1)}
    print(f"  Step 3b (Jaccard): {len(deduped_exact)} → {len(deduped)} "
          f"({len(deduped_exact) - len(deduped)} dropped)", flush=True)
    print(f"  Total: {stats['before']} → {stats['after']} "
          f"({stats['dropped_total']} dropped) "
          f"({stats['elapsed_s']}s)", flush=True)

    return deduped, stats


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    input_csv: Path,
    output_dir: Path,
    confidence_threshold: float = 0.8,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print(f"Loading {input_csv} ...", flush=True)
    rows: list[dict[str, str]] = []
    with input_csv.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            r["_norm_subject"] = _normalize(r.get("subject", ""))
            r["_norm_object"] = _normalize(r.get("object", ""))
            if not r["_norm_subject"] and not r["_norm_object"]:
                continue
            rows.append(r)
    print(f"  {len(rows)} rows loaded", flush=True)

    # Phase 1
    entity_map, p1_stats = canonicalise_entities(rows)

    # Phase 2
    type_map, type_conflicts, p2_stats = resolve_types(
        rows, entity_map, confidence_threshold,
    )

    # Phase 3
    canonical_rows, p3_stats = dedup_triples(rows, entity_map, type_map)

    # Write canonical CSV
    csv_path = output_dir / "triplets_verified_canonical.csv"
    fieldnames = [
        "snippet_id", "source_id", "source_file", "source_type",
        "chunk_index", "subject", "subject_type", "subject_id",
        "predicate", "object", "object_type", "object_id",
        "confidence", "evidence_span", "_type_fix",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for r in canonical_rows:
            writer.writerow(r)
    print(f"\n  ✓ Saved {csv_path} ({len(canonical_rows)} rows)", flush=True)

    # Write entity map
    map_path = output_dir / "entity_map.json"
    map_path.write_text(json.dumps(
        {"entities": {k: v for k, v in sorted(entity_map.items())},
         "total": len(entity_map), "merged": p1_stats["total_merged"]},
        indent=2, ensure_ascii=False,
    ), encoding="utf-8")
    print(f"  ✓ Saved {map_path}", flush=True)

    # Write type conflicts
    if type_conflicts:
        conflicts_path = output_dir / "type_conflicts.json"
        conflicts_path.write_text(json.dumps(
            type_conflicts, indent=2, ensure_ascii=False,
        ), encoding="utf-8")
        print(f"  ✓ Saved {conflicts_path}", flush=True)

    # Write report
    report = {
        "input_rows": len(rows),
        "output_rows": len(canonical_rows),
        "phase1": p1_stats,
        "phase2": p2_stats,
        "phase3": p3_stats,
    }
    report_path = output_dir / "canonicalisation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"  ✓ Saved {report_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-phase canonicalisation of extracted KG triplets"
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Path to input triplets CSV")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory for output files")
    p.add_argument("--confidence-threshold", type=float, default=0.8,
                   help="Type-conflict auto-resolve threshold (default: 0.8)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        raise SystemExit(2)
    run(args.input, args.output_dir, args.confidence_threshold)


if __name__ == "__main__":
    main()
