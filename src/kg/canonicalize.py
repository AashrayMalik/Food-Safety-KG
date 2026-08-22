import csv, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SRC = Path(__file__).resolve().parents[1]

IN_PATH = SRC / "outputs" / "triplets" / "qwen" / "triplets_verified.csv"
OUT_CSV = SRC / "outputs" / "triplets" / "qwen" / "triplets_verified_canonicalized.csv"
AUDIT_CSV = SRC / "outputs" / "triplets" / "qwen" / "canonicalization_audit.csv"
BORDERLINE_CSV = SRC / "outputs" / "triplets" / "qwen" / "canonicalization_borderline_review.csv"
CROSS_TYPE_AUDIT_CSV = SRC / "outputs" / "triplets" / "qwen" / "canonicalization_cross_type_merges.csv"

SIM_THRESHOLD_AUTO = 1.01   # effectively disabled: nothing merges on similarity alone anymore.
                             # Manual review found false merges the similarity signal could not
                             # reliably distinguish (numeric values, negated phrases, single
                             # differing content words mid-string e.g. "ethyl" vs "methyl",
                             # "fish" vs "meat" vs "milk" substitutes). Only exact string matches
                             # (after normalization) merge automatically now; every similarity
                             # match short of exact goes to borderline for judgment instead.
SIM_THRESHOLD_LOW = 0.75    # below this, not even flagged as borderline

def normalize(s):
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)  # replace punctuation with a space (not delete!) --
                                     # deleting outright collapsed "Table-10B" -> "table10b"
                                     # while "Table 10B" -> "table 10b", so hyphenated and
                                     # space-separated variants of the same string normalized
                                     # to two different keys and never merged. Replacing with
                                     # a space (then collapsing) makes both -> "table 10b".
    s = re.sub(r"\s+", " ", s)      # collapse whitespace
    return s.strip()

def is_event_id(raw):
    # Synthetic event-reification IDs (e.g. ev_limit_egg_0001) are already unique
    # per-event identifiers, not natural-language mentions -- never cluster these,
    # or distinct events collapse into one node.
    return bool(re.match(r"^ev_[a-z0-9_]+_\d+$", raw.strip().lower()))

# Types where char n-gram similarity is known to falsely merge distinct values
# (e.g. "1.0% by weight" vs "5% by weight", or long near-identical descriptive
# strings that differ only in a negation/quantifier like "less than" vs "more
# than"). Found via manual random-sample review -- see canonicalization_audit
# review notes. These types get exact-normalized-match only, no similarity
# clustering, so distinct values never collapse just because they share text.
NO_SIMILARITY_CLUSTERING_TYPES = {"fso:Measurement", "xsd:string"}

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

def numeric_tokens(s):
    return set(NUMBER_RE.findall(s))

def has_conflicting_numbers(a, b):
    """True if both strings contain numbers and those number sets differ.
    Guards against merging e.g. '1.0% by weight' with '5% by weight'."""
    na, nb = numeric_tokens(a), numeric_tokens(b)
    if na and nb and na != nb:
        return True
    return False

def type_prefix(t):
    # fkg:Ingredient -> ING, fflo:RegulatoryBody -> REG, etc. First 3 letters after colon, uppercased.
    local = t.split(":")[-1]
    letters = re.sub(r"[^A-Za-z]", "", local).upper()
    return letters[:3] if letters else "GEN"

def main():
    with open(IN_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Decide where to write the KG canonical ids. If the input already carries
    # NLI-assigned ids in subject_id / object_id (synthetic n-ary node ids),
    # do not overwrite them — write the KG canonical ids to separate columns.
    has_existing_ids = any(
        (row.get("subject_id") or "").strip() or (row.get("object_id") or "").strip()
        for row in rows
    )
    if has_existing_ids:
        subject_target = "subject_canonical_id"
        object_target = "object_canonical_id"
    else:
        subject_target = "subject_id"
        object_target = "object_id"
    if subject_target not in fieldnames:
        fieldnames = fieldnames + [subject_target, object_target]
    for row in rows:
        row.setdefault(subject_target, "")
        row.setdefault(object_target, "")

    # Build entity records keyed by (type, normalized_string) with all original variants and row refs.
    # role: 'subject' or 'object' — same string+type used as both should get the same ID, so we
    # unify subject and object pools together, blocked by type.
    entity_index = defaultdict(lambda: {"variants": defaultdict(list)})
    # entity_index[type] = {"variants": {normalized_str: [(row_idx, role, original_str), ...]}}

    event_id_rows = defaultdict(list)  # (typ, raw) -> [(row_idx, role, raw), ...] for existing ev_* ids

    for i, row in enumerate(rows):
        for role, val_col, type_col in [("subject", "subject", "subject_type"), ("object", "object", "object_type")]:
            raw = row[val_col]
            typ = row[type_col]
            if not raw or not raw.strip():
                continue
            if is_event_id(raw):
                event_id_rows[(typ, raw.strip())].append((i, role, raw))
                continue
            norm = normalize(raw)
            entity_index[typ]["variants"][norm].append((i, role, raw))

    id_counter = defaultdict(int)   # per type_prefix counter
    canon_id_of = {}                # (type, normalized_str) -> assigned ID
    cluster_records = []            # for audit: cluster_id, type, members (normalized strings), row_count
    borderline_records = []         # pairs flagged for manual/LLM review, not merged

    for typ, data in entity_index.items():
        norm_strings = list(data["variants"].keys())
        n = len(norm_strings)
        prefix = type_prefix(typ)

        if n == 0:
            continue

        if n == 1:
            clusters = [[0]]
        elif typ in NO_SIMILARITY_CLUSTERING_TYPES:
            # Exact-normalized-match only for types prone to false similarity merges.
            # normalize() already collapsed exact duplicates into one key in norm_strings,
            # so at this point every remaining string is already distinct -- no further
            # clustering, each is its own singleton.
            clusters = [[i] for i in range(n)]
        else:
            # TF-IDF over character n-grams (3-5 chars) captures partial/abbreviation overlap
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
            try:
                X = vectorizer.fit_transform(norm_strings)
                sim = cosine_similarity(X)
            except ValueError:
                # e.g. all-empty after normalization; treat as singleton clusters
                sim = np.eye(n)

            # Union-Find for auto-merge clusters (sim >= AUTO threshold)
            parent = list(range(n))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            def differs_by_leading_qualifier(a, b):
                """True if one string is the other with an extra leading word
                (e.g. 'sodium calcium polyphosphate' vs 'calcium polyphosphate').
                Often signals a distinct compound/entity, not a spelling variant --
                route to borderline for human/LLM review instead of auto-merging."""
                wa, wb = a.split(), b.split()
                shorter, longer = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
                if len(longer) - len(shorter) == 0:
                    return False
                return longer[-len(shorter):] == shorter and len(shorter) > 0

            for a in range(n):
                for b in range(a + 1, n):
                    s = sim[a, b]
                    if has_conflicting_numbers(norm_strings[a], norm_strings[b]):
                        # Never merge strings whose numeric values differ, even at high
                        # textual similarity (e.g. "1.0% by weight" vs "5% by weight").
                        continue
                    if s >= SIM_THRESHOLD_AUTO and differs_by_leading_qualifier(norm_strings[a], norm_strings[b]):
                        # High similarity but one string has an extra leading qualifier word
                        # (e.g. "sodium calcium polyphosphate" vs "calcium polyphosphate") --
                        # often a distinct compound, flag for review rather than auto-merge.
                        borderline_records.append({
                            "type": typ,
                            "string_a": norm_strings[a],
                            "string_b": norm_strings[b],
                            "similarity": round(float(s), 3),
                        })
                    elif s >= SIM_THRESHOLD_AUTO:
                        union(a, b)
                    elif SIM_THRESHOLD_LOW <= s < SIM_THRESHOLD_AUTO:
                        borderline_records.append({
                            "type": typ,
                            "string_a": norm_strings[a],
                            "string_b": norm_strings[b],
                            "similarity": round(float(s), 3),
                        })

            clusters_map = defaultdict(list)
            for idx in range(n):
                clusters_map[find(idx)].append(idx)
            clusters = list(clusters_map.values())

        # Assign one canonical ID per cluster
        for cluster in clusters:
            id_counter[prefix] += 1
            cid = f"fkg:{prefix}_{id_counter[prefix]:05d}"
            member_strings = [norm_strings[idx] for idx in cluster]
            total_rows = sum(len(data["variants"][ms]) for ms in member_strings)
            cluster_records.append({
                "canonical_id": cid,
                "type": typ,
                "member_count": len(member_strings),
                "row_count": total_rows,
                "members": " | ".join(sorted(set(
                    orig for ms in member_strings for (_, _, orig) in data["variants"][ms]
                ))),
            })
            for ms in member_strings:
                canon_id_of[(typ, ms)] = cid

    # Event-reification IDs pass through untouched -- each is already a unique event instance,
    # never merged with siblings that share a prefix.
    event_canon_id_of = {}
    for (typ, raw), refs in event_id_rows.items():
        cid = f"fkg:{raw}"
        event_canon_id_of[(typ, raw)] = cid
        cluster_records.append({
            "canonical_id": cid,
            "type": typ,
            "member_count": 1,
            "row_count": len(refs),
            "members": raw + "  [existing event ID, passed through unmerged]",
        })

    # --- Cross-type exact-match pass ---
    # The same exact (case+spelling identical) string can appear with different
    # subject_type/object_type labels across rows -- this is extraction-time type
    # labeling inconsistency, not two different entities that coincidentally share
    # identical text. Type-blocking above is the right default for FUZZY similarity
    # (prevents unrelated entities merging), but for EXACT string matches it's too
    # strict: it was splitting the same real-world entity into separate IDs whenever
    # extraction tagged it inconsistently.
    #
    # Fix: merge canonical IDs across types when the normalized string is 100%
    # identical. Every such cross-type merge is logged to a separate audit file for
    # review, since a small number of these may be genuinely ambiguous/generic
    # strings (e.g. a bare word like "food") rather than labeling noise.
    exact_string_to_ids = defaultdict(set)
    for (typ, norm_str), cid in canon_id_of.items():
        exact_string_to_ids[norm_str].add((typ, cid))

    ct_parent = {}
    def ct_find(x):
        ct_parent.setdefault(x, x)
        while ct_parent[x] != x:
            ct_parent[x] = ct_parent[ct_parent[x]]
            x = ct_parent[x]
        return x
    def ct_union(a, b):
        ra, rb = ct_find(a), ct_find(b)
        if ra != rb:
            ct_parent[ra] = rb

    cross_type_audit = []
    for norm_str, type_id_pairs in exact_string_to_ids.items():
        if len(type_id_pairs) <= 1:
            continue
        ids = [cid for (_, cid) in type_id_pairs]
        for other in ids[1:]:
            ct_union(ids[0], other)
        cross_type_audit.append({
            "normalized_string": norm_str,
            "types_and_ids": " | ".join(f"{t}:{c}" for t, c in sorted(type_id_pairs)),
        })

    ct_final_id_of = {}
    all_canon_ids = set(canon_id_of.values())
    ct_groups = defaultdict(list)
    for cid in all_canon_ids:
        ct_groups[ct_find(cid)].append(cid)
    for root, members in ct_groups.items():
        chosen = sorted(members)[0]
        for m in members:
            ct_final_id_of[m] = chosen

    # Remap canon_id_of through the cross-type merge
    for k in list(canon_id_of.keys()):
        canon_id_of[k] = ct_final_id_of.get(canon_id_of[k], canon_id_of[k])

    # Write IDs back into rows
    for i, row in enumerate(rows):
        for role, val_col, type_col, id_col in [
            ("subject", "subject", "subject_type", subject_target),
            ("object", "object", "object_type", object_target),
        ]:
            raw = row[val_col]
            typ = row[type_col]
            if not raw or not raw.strip():
                row[id_col] = ""  # e.g. ev_* rows with no object value -- key must still
                                   # exist or the summary print below raises KeyError
                continue
            if is_event_id(raw):
                row[id_col] = event_canon_id_of.get((typ, raw.strip()), "")
            else:
                norm = normalize(raw)
                row[id_col] = canon_id_of.get((typ, norm), "")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(AUDIT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["canonical_id", "type", "member_count", "row_count", "members"])
        writer.writeheader()
        # sort by member_count desc so biggest merges are easy to spot-check
        writer.writerows(sorted(cluster_records, key=lambda r: -r["member_count"]))

    with open(BORDERLINE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "string_a", "string_b", "similarity"])
        writer.writeheader()
        writer.writerows(sorted(borderline_records, key=lambda r: -r["similarity"]))

    with open(CROSS_TYPE_AUDIT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["normalized_string", "types_and_ids"])
        writer.writeheader()
        writer.writerows(cross_type_audit)

    total_entities = sum(len(d["variants"]) for d in entity_index.values())
    total_clusters = len(cluster_records)
    multi_member = sum(1 for r in cluster_records if r["member_count"] > 1)

    print(f"Total unique (type, normalized-string) entities: {total_entities}")
    print(f"Total canonical IDs assigned (pre cross-type merge): {total_clusters}")
    print(f"Clusters with >1 surface-form merged (within-type): {multi_member}")
    print(f"Cross-type exact-match merges applied: {len(cross_type_audit)}")
    print(f"Final unique IDs after cross-type merge: {len(set(ct_final_id_of.values()))}")
    print(f"Borderline pairs flagged for manual/LLM review: {len(borderline_records)}")
    print(f"Rows with {subject_target} filled: {sum(1 for r in rows if r[subject_target])}")
    print(f"Rows with {object_target} filled: {sum(1 for r in rows if r[object_target])}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Populate subject_id / object_id with canonical entity IDs."
    )
    parser.add_argument("--input", type=Path, default=IN_PATH)
    parser.add_argument("--output", type=Path, default=OUT_CSV)
    parser.add_argument("--audit", type=Path, default=AUDIT_CSV)
    parser.add_argument("--borderline", type=Path, default=BORDERLINE_CSV)
    parser.add_argument("--cross-type-audit", type=Path, default=CROSS_TYPE_AUDIT_CSV)
    args = parser.parse_args()

    IN_PATH = args.input
    OUT_CSV = args.output
    AUDIT_CSV = args.audit
    BORDERLINE_CSV = args.borderline
    CROSS_TYPE_AUDIT_CSV = args.cross_type_audit

    main()
