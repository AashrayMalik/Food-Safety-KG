"""String-based entity-type and relation-name normalisation.

No embeddings, no clustering — purely lexical heuristics to minimise
name variation from unconstrained extraction.

Produces a canonical form for every name, plus groups of near-duplicates
that should be manually reviewed.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Stopwords — common English words that don't carry semantic weight
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "the", "a", "an", "of", "for", "in", "at", "to", "is", "be",
    "and", "or", "with", "from", "by", "on", "as", "that", "this",
    "it", "its", "are", "was", "were", "been", "has", "have", "had",
})


# ---------------------------------------------------------------------------
# De-pluralisation
# ---------------------------------------------------------------------------

def _depluralize(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ses") and len(word) > 4 and word[-4] not in "aeiou":
        return word[:-1]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


# ---------------------------------------------------------------------------
# Canonicalise
# ---------------------------------------------------------------------------

def canonicalise(name: str) -> str:
    """Return a canonical, comparison-safe form of *name*."""
    # Strip namespace prefixes (fflo:, fkg:, etc.)
    name = re.sub(r"^(fflo|fkg|fso|prov|lkif|ssn|rdf|rdfs|xsd):", "", name)
    # Split camelCase BEFORE lowering: "PermissibleLimit" → "Permissible Limit"
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    # Lowercase
    name = name.lower().strip()
    # Remove punctuation except hyphens (keep "non-alcoholic")
    name = re.sub(r"[^a-z0-9\s\-]", " ", name)
    # Normalise whitespace
    name = re.sub(r"\s+", " ", name).strip()
    # Tokenise
    tokens = name.split()
    # Remove stopwords
    tokens = [t for t in tokens if t not in _STOPWORDS]
    # De-pluralise each token
    tokens = [_depluralize(t) for t in tokens]
    # Sort tokens (catch "food safety" ↔ "safety food")
    tokens = sorted(set(tokens))
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost,
            )
        prev, curr = curr, prev
    return prev[m]


# ---------------------------------------------------------------------------
# Normalise a list of names
# ---------------------------------------------------------------------------

def normalise_names(
    raw_names: list[str],
    max_edit_distance: int = 2,
) -> dict[str, Any]:
    """Canonicalise and group entity-type or relation names.

    Returns::

        {
            "canonical_map": {raw → canonical},   # every name → canonical
            "frequency":    {canonical → count},   # how many raws per canonical
            "near_duplicates": [
                [canon_A, canon_B, ...],           # groups of canonicals
            ],                                     # within edit distance
            "singletons": [canon_X, ...],          # canonicals with no near dupes
        }
    """
    # ---- canonicalise ----
    raw_to_canon: dict[str, str] = {}
    canon_to_raws: dict[str, list[str]] = defaultdict(list)
    for r in raw_names:
        c = canonicalise(r)
        raw_to_canon[r] = c
        canon_to_raws[c].append(r)

    freq = Counter(raw_to_canon.values())

    # ---- near-duplicate grouping ----
    canons = sorted(canon_to_raws.keys())
    visited: set[str] = set()
    groups: list[list[str]] = []
    singletons: list[str] = []

    for i, ci in enumerate(canons):
        if ci in visited:
            continue
        group = [ci]
        for j, cj in enumerate(canons):
            if i == j or cj in visited:
                continue
            if len(ci) >= 3 and len(cj) >= 3 and abs(len(ci) - len(cj)) <= max_edit_distance:
                if levenshtein(ci, cj) <= max_edit_distance:
                    group.append(cj)
                    visited.add(cj)
        visited.add(ci)
        if len(group) > 1:
            groups.append(sorted(group))
        else:
            singletons.append(ci)

    return {
        "canonical_map": raw_to_canon,
        "frequency": dict(freq),
        "near_duplicates": groups,
        "singletons": singletons,
    }


# ---------------------------------------------------------------------------
# Canonicalisation map persistence
# ---------------------------------------------------------------------------

def save_canonical_map(
    merges: dict[str, str],
    path: Path,
    reviewed_by: str = "",
) -> None:
    """Save human-reviewed merge decisions as JSON."""
    import json
    from datetime import datetime, timezone

    data = {
        "merges": merges,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_by": reviewed_by,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def load_canonical_map(path: Path) -> dict[str, str]:
    """Load merge decisions from JSON.  Returns {raw_name → canonical_name}."""
    import json

    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("merges", {})


def apply_canonical_map(
    raw_names: list[str],
    merge_map: dict[str, str],
) -> list[str]:
    """Pre-apply user-defined merges to a list of raw names.

    Returns *resolved* names — each raw name mapped to its canonical
    target (recursively, for chained merges).
    """
    resolved: list[str] = []
    for name in raw_names:
        current = name
        seen: set[str] = set()
        while current in merge_map and current not in seen:
            seen.add(current)
            current = merge_map[current]
        resolved.append(current)
    return resolved


# ---------------------------------------------------------------------------
# Embedding-based clustering (semantic similarity, not just lexical)
# ---------------------------------------------------------------------------

def cluster_by_embedding(
    canonical_names: list[tuple[str, int]],
    evidence_samples: dict[str, list[str]],
    type_pairs: dict[str, set[tuple[str, str]]],
    threshold: float = 0.82,
    min_type_overlap: float = 0.3,
) -> tuple[list[list[str]], list[list[str]]]:
    """Cluster canonical relation names by cosine similarity.

    Each name is embedded as the name plus up to 3 sample evidence
    spans.  Type-compatibility guards prevent merging relations with
    incompatible (subject_type, object_type) signatures.

    Returns ``(suggested_merges, flagged_for_review)``.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    names = [n for n, _ in canonical_names]
    if len(names) < 2:
        return [], []

    # Build embedding texts
    texts: list[str] = []
    for name in names:
        samples = evidence_samples.get(name, [])
        ctx = " ; ".join(samples[:3]) if samples else name
        texts.append(f"{name} : {ctx}")

    # Embed
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sim = embeddings @ embeddings.T / (norms @ norms.T)

    # Agglomerative clustering
    clusters: list[list[int]] = [[i] for i in range(len(names))]
    while True:
        best = (-1.0, -1, -1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Max-linkage: max sim between any members
                max_sim = max(
                    sim[a][b] for a in clusters[i] for b in clusters[j]
                )
                if max_sim > best[0]:
                    best = (max_sim, i, j)
        if best[0] < threshold:
            break
        clusters[best[1]].extend(clusters[best[2]])
        clusters.pop(best[2])

    # Split into suggested vs flagged by type compatibility
    suggested: list[list[str]] = []
    flagged: list[list[str]] = []
    for cl in clusters:
        if len(cl) < 2:
            continue
        name_group = [names[i] for i in cl]
        if _check_type_compatible(cl, names, type_pairs, min_type_overlap):
            suggested.append(name_group)
        else:
            flagged.append(name_group)

    return suggested, flagged


def _check_type_compatible(
    cluster_indices: list[int],
    names: list[str],
    type_pairs: dict[str, set[tuple[str, str]]],
    min_overlap: float = 0.3,
) -> bool:
    """Check whether all relations in a cluster share similar type signatures."""
    sigs: list[set[tuple[str, str]]] = []
    for idx in cluster_indices:
        name = names[idx]
        sigs.append(type_pairs.get(name, set()))

    if not any(sigs):
        return True  # no data to judge

    # Union of all type pairs in cluster
    total_pairs: set[tuple[str, str]] = set()
    for s in sigs:
        total_pairs.update(s)

    if not total_pairs:
        return True

    # Check: for each signature, does it overlap at least min_overlap
    # with the cluster's union?
    for s in sigs:
        if not s:
            continue
        if len(s & total_pairs) / max(len(s), 1) < min_overlap:
            return False
    return True
