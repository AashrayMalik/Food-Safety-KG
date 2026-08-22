"""Synthetic ID assignment and label resolution for NLI verbalisation.

Determines which entities are structural/synthetic nodes (pipeline-
generated names not appearing verbatim in the source text) and assigns
deterministic hash-based IDs so they can be resolved across multiple
triplets within the same chunk.

Resolution order (first non-empty wins):
1. **CSV columns**: ``subject_id`` / ``object_id`` from the extraction CSV
   (new extraction runs will include these).
2. **Gold file**: ``subject_id`` / ``object_id`` from the golden validation
   Excel (when ``--gold-file`` is provided).
3. **Heuristic fallback**: entity name does NOT appear verbatim in
   ``evidence_text`` → synthetic node, needs label resolution.
"""

import hashlib

from verbalise import type_phrase


def _hash_id(snippet_id: str, name: str, etype: str) -> str:
    raw = f"{snippet_id}|{name}|{etype}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def assign_ids(
    triplets: list[dict],
    evidence_texts: dict[str, str],
    csv_id_cols: tuple[str, str] = ("subject_id", "object_id"),
    gold_ids: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Assign synthetic IDs and build an entity-label lookup.

    Returns ``(id_map, label_map)`` where:

    - *id_map* maps entity names (within a snippet) to their synthetic ID
      (or None for named entities).
    - *label_map* maps synthetic IDs to descriptive labels for
      verbalisation.

    Parameters
    ----------
    triplets:
        List of triplet dicts with keys ``snippet_id``, ``subject``,
        ``subject_type``, ``object``, ``object_type``, plus optional
        ``subject_id`` / ``object_id`` columns.
    evidence_texts:
        ``snippet_id → evidence_text`` lookup (from chunks CSV).
    csv_id_cols:
        Names of the subject-id and object-id columns in *triplets*.
    gold_ids:
        ``snippet_id → {"subject_id": ..., "object_id": ...}`` from the
        golden validation Excel (overrides heuristic).
    """
    # ---- collect all (snippet_id, entity_name, entity_type) triples ----
    entities: dict[tuple[str, str, str], str | None] = {}
    # (snippet, name, type) → synthetic_id or None

    for t in triplets:
        sid = t["snippet_id"]
        subj = t.get("subject", "")
        stype = t.get("subject_type", "")
        obj = t.get("object", "")
        otype = t.get("object_type", "")

        # --- check CSV columns first ---
        csv_sid = t.get(csv_id_cols[0], "") or ""
        csv_oid = t.get(csv_id_cols[1], "") or ""

        if csv_sid:
            entities[(sid, subj, stype)] = csv_sid
        elif (sid, subj, stype) not in entities:
            entities[(sid, subj, stype)] = None  # undecided

        if csv_oid:
            entities[(sid, obj, otype)] = csv_oid
        elif (sid, obj, otype) not in entities:
            entities[(sid, obj, otype)] = None

    # ---- gold file overrides ---
    if gold_ids:
        for t in triplets:
            sid = t["snippet_id"]
            g = gold_ids.get(sid, {})
            g_sid = g.get("subject_id", "")
            g_oid = g.get("object_id", "")
            subj = t.get("subject", "")
            stype = t.get("subject_type", "")
            obj = t.get("object", "")
            otype = t.get("object_type", "")
            if g_sid:
                entities[(sid, subj, stype)] = g_sid
            if g_oid:
                entities[(sid, obj, otype)] = g_oid

    # ---- heuristic: name not in evidence_text → synthetic ----
    for (sid, name, etype), current_id in list(entities.items()):
        if current_id is not None:
            continue  # already decided by CSV or gold
        ev = evidence_texts.get(sid, "")
        if name and name not in ev:
            entities[(sid, name, etype)] = _hash_id(sid, name, etype)
        else:
            entities[(sid, name, etype)] = None  # named entity, no ID

    # ---- build label map for entities WITH an ID ----
    label_map: dict[str, str] = {}
    id_map: dict[str, dict[str, str]] = {}  # snippet → {name → id}

    for (sid, name, etype), eid in entities.items():
        if eid is not None:
            phrase = type_phrase(etype)
            if phrase:
                label_map[eid] = f"{phrase} {name}"
            else:
                label_map[eid] = name
        id_map.setdefault(sid, {})[name] = eid or ""

    return id_map, label_map


def resolve_label(
    entity_name: str,
    snippet_id: str,
    id_map: dict[str, dict[str, str]],
    label_map: dict[str, str],
) -> str:
    """Return the descriptive label for an entity, or its bare name."""
    sid_map = id_map.get(snippet_id, {})
    eid = sid_map.get(entity_name)
    if eid and eid in label_map:
        return label_map[eid]
    return entity_name
