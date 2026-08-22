#!/usr/bin/env python3
"""LLM-driven canonicalisation of free-form relation names.

Reads cluster groups from embedding/Levenshtein clustering, pulls real
grounding examples (subject_type/object_type/evidence_span) for each
member from the triplets data, and asks Qwen to VERIFY each cluster
before naming it — splitting or rejecting where the grounding doesn't
support a merge, rather than forcing one.

Output: ``canonical_map.json`` (auto-applied merges, confidence >=
threshold, not flagged) plus ``canonical_map_needs_review.json`` and
``canonical_map_out_of_scope.json`` for the rest.

Usage::

    python src/validation/llm_canonicalise.py \\
        --report-json  /home/aashray_malik/src/outputs/unconstrained/unconstrained_report.json \\
        --triplets-csv /home/aashray_malik/src/outputs/unconstrained/nli_llm_judge_entailed.csv \\
        --base-url     http://localhost:8030/v1 \\
        --model        Qwen/Qwen3.5-27B-FP8 \\
        --api-key      aashray-fflo-local \\
        --output       canonical_map.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


# ---- LLM prompt ----

CANONICALISE_SYSTEM = """\
You are a knowledge-graph schema curator for food safety regulations.

You will receive GROUPS of relation names that an automated similarity
step (string + embedding matching) suggests might be the SAME concept.
This suggestion is NOT reliable on its own — similarity in spelling or
phrasing does not guarantee the same meaning. Your job is to verify
each group before naming it, using the example triplets provided as
grounding, not the relation-name strings alone.

For each group:

1. **Check whether the members are genuinely the same relation.** Look
   at the subject_type/object_type pairs and evidence spans for each
   member, not just the name. Two relations with similar names but
   different subject/object type patterns are usually NOT the same
   concept (e.g. a name like "act" used with (Regulation, none) vs
   "actor" used with (Person, Organisation) are different relations
   despite string similarity).

2. **Never merge two relations that are lexical opposites or negations
   of each other**, even if their usage patterns look similar — e.g.
   "include" vs "exclude", "allow" vs "forbid", "must" vs "may". These
   must always stay as separate, related-but-distinct relations, no
   matter how similar their subject/object typing looks. Flag such
   pairs explicitly in the rationale if you see them in a group.

3. **If the group is genuinely one concept**, choose ONE canonical
   umbrella name for all of it.

4. **If the group mixes distinct concepts**, split it: return separate
   canonical names, each listing only the members that truly belong
   together. Do not force a merge to avoid returning multiple names.

5. **If you are not confident either way** (grounding is ambiguous or
   insufficient), set needs_review true and explain why, rather than
   guessing.

6. **If a cluster looks administrative/bureaucratic rather than
   food-safety-relevant** (e.g. document printing, form numbering,
   internal filing references), set out_of_scope true instead of
   naming it — these should not enter the food safety ontology.

Naming rules (when you do assign a canonical name):
- Choose the shortest name that is still descriptive.
- Use camelCase (e.g. "hasMaximumLevel", not "has_maximum_level").
- Prefer names that match the style of existing FFLO relations
  (e.g. "belongsToCategory", "hasValue", "appliesTo").
- If a group has only ONE member, return that name unchanged, still
  with a confidence and out_of_scope/needs_review judgement.

Output ONLY a JSON object, no markdown, no commentary:

{
  "clusters": [
    {
      "input_group": "cluster_0",
      "canonical_name": "hasMaximumLevel",
      "members": ["hasMaxLimit", "hasMaximumLimit", "hasPermissibleLimit"],
      "confidence": 0.9,
      "needs_review": false,
      "out_of_scope": false,
      "rationale": "All members share (Food, QuantityValue) typing and describe an upper bound on a measured quantity."
    },
    {
      "input_group": "cluster_0",
      "canonical_name": "hasMinimumLevel",
      "members": ["hasMinLimit", "hasMinimumLimit"],
      "confidence": 0.9,
      "needs_review": false,
      "out_of_scope": false,
      "rationale": "Split from cluster_0 — these describe a lower bound, a distinct concept from the upper-bound members."
    }
  ]
}

Note: multiple output entries may share the same input_group when you
split it. Every member of every input group must appear in exactly
one output entry.
"""


def _build_user_message(
    cluster_id: str,
    names: list[str],
    grounding: dict[str, list[dict[str, Any]]],
    suggestion_source: str = "embedding",
    threshold: float = 0.82,
) -> str:
    """Build the grounded user prompt for ONE cluster group.

    grounding: {relation_name: [{"subject_type":..., "object_type":...,
                                  "evidence_span":..., "count": N}, ...]}
    """
    parts = [
        f"=== CANDIDATE GROUP: {cluster_id} ===",
        f"Suggested by: {suggestion_source} (threshold {threshold})",
        "",
        "Members and grounding (up to 3 example triplets per member):",
        "",
    ]
    for name in names:
        examples = grounding.get(name, [])
        count = examples[0]["count"] if examples else 0
        parts.append(f'  "{name}"  (used {count} times)')
        for ex in examples[:3]:
            parts.append(
                f"    - ({ex['subject_type']}) --{name}--> "
                f"({ex['object_type']})"
            )
            parts.append(f"      evidence: \"{ex['evidence_span']}\"")
        if not examples:
            parts.append("      (no grounding examples found — treat with low confidence)")
        parts.append("")

    parts.append(
        "Verify this group per the rules above and return the JSON result."
    )
    return "\n".join(parts)


def _build_grounding_lookup(
    triplets_df: pd.DataFrame, max_examples: int = 3
) -> dict[str, list[dict[str, Any]]]:
    """Pre-compute example triplets + counts per raw predicate name."""
    grounding: dict[str, list[dict[str, Any]]] = {}
    counts = triplets_df["predicate"].value_counts()
    for name, group in triplets_df.groupby("predicate"):
        rows = group.head(max_examples)
        grounding[name] = [
            {
                "subject_type": r["subject_type"],
                "object_type": r["object_type"],
                "evidence_span": str(r["evidence_span"])[:120],
                "count": int(counts[name]),
            }
            for _, r in rows.iterrows()
        ]
    return grounding


# ---- LLM call (async, ONE cluster per call — grounding makes batching multiple clusters per prompt too token-heavy to keep reliable) ----

async def _llm_canonicalise(
    base_url: str,
    api_key: str,
    model: str,
    clusters: dict[str, list[str]],
    grounding: dict[str, list[dict[str, Any]]],
    concurrency: int = 8,
) -> list[dict[str, Any]]:
    """Send each cluster to Qwen individually (grounded), merge results."""
    import httpx

    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    async def _one(cid: str, names: list[str], client: "httpx.AsyncClient"):
        user_msg = _build_user_message(cid, names, grounding)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": CANONICALISE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        async with sem:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(120),
            )
        data = resp.json()
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
        ).strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(content)
            clusters_out = parsed.get("clusters", [])
            if not clusters_out:
                print(f"  {cid}: model returned no 'clusters' key — check output format")
            return clusters_out
        except json.JSONDecodeError:
            print(f"  {cid}: JSON parse failed, flagging whole group for manual review")
            return [{
                "input_group": cid,
                "canonical_name": names[0],
                "members": names,
                "confidence": 0.0,
                "needs_review": True,
                "out_of_scope": False,
                "rationale": "LLM output failed to parse — needs manual resolution.",
            }]

    async with httpx.AsyncClient() as client:
        tasks = [_one(cid, names, client) for cid, names in clusters.items()]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            batch_result = await coro
            results.extend(batch_result)
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                print(f"  {i+1}/{len(tasks)} clusters processed")

    return results


# ---- Antonym guard (belt-and-suspenders on top of the prompt rule) ----

_ANTONYM_PAIRS = [
    ("include", "exclude"), ("allow", "forbid"), ("allow", "prohibit"),
    ("must", "may"), ("permit", "forbid"), ("approve", "reject"),
    ("add", "remove"), ("max", "min"), ("maximum", "minimum"),
    ("required", "optional"), ("positive", "negative"),
]


def _contains_antonym_pair(members: list[str]) -> str | None:
    lowered = [m.lower() for m in members]
    for a, b in _ANTONYM_PAIRS:
        has_a = any(a in m for m in lowered)
        has_b = any(b in m for m in lowered)
        if has_a and has_b:
            return f"{a}/{b}"
    return None


# ---- Cluster extraction from report ----

def _looks_like_group_list(value: Any) -> bool:
    """A 'group list' is a list whose elements are themselves lists of
    strings (i.e. [['a','b'], ['c','d']]) — not a count, not a single
    flat list of names, not a dict."""
    if not isinstance(value, list) or not value:
        return False
    return all(
        isinstance(item, list) and all(isinstance(x, str) for x in item)
        for item in value
    )


def _find_group_lists(node: Any, path: str = "normalisation") -> dict[str, list[list[str]]]:
    """Recursively walk the report JSON looking for anything that looks
    like a list of clustered name-groups, wherever it actually lives.
    Prints what it finds/skips so you can see the real structure."""
    found: dict[str, list[list[str]]] = {}

    if isinstance(node, dict):
        for key, value in node.items():
            sub_path = f"{path}.{key}"
            if _looks_like_group_list(value):
                found[sub_path] = value
                print(f"  found group list at {sub_path}: {len(value)} groups")
            elif isinstance(value, (dict, list)):
                found.update(_find_group_lists(value, sub_path))
            else:
                # scalar (count, string, etc.) — expected, not an error
                pass
    elif isinstance(node, list):
        for i, item in enumerate(node):
            if isinstance(item, (dict, list)):
                found.update(_find_group_lists(item, f"{path}[{i}]"))

    return found


def _extract_clusters_from_report(report_path: Path) -> dict[str, list[str]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    normalisation = report.get("normalisation", {})

    print("Scanning report for cluster group lists...")
    group_lists = _find_group_lists(normalisation)

    if not group_lists:
        print(
            "  WARNING: no group-list structures found under 'normalisation'. "
            "Top-level keys present: "
            f"{list(normalisation.keys()) if isinstance(normalisation, dict) else type(normalisation)}"
        )

    clusters: dict[str, list[str]] = {}
    for section_path, groups in group_lists.items():
        section_name = section_path.replace(".", "_").replace("[", "_").replace("]", "")
        for i, group in enumerate(groups):
            clusters[f"{section_name}_{i}"] = group
    return clusters


def _load_clusters_json(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("clusters", data)


# ---- Save results, split by disposition ----

def _save_results(
    cluster_results: list[dict[str, Any]],
    output_path: Path,
    confidence_threshold: float = 0.75,
) -> None:
    auto_map: dict[str, str] = {}
    needs_review: list[dict[str, Any]] = []
    out_of_scope: list[dict[str, Any]] = []
    antonym_flagged: list[dict[str, Any]] = []

    for entry in cluster_results:
        members = entry.get("members", [])
        canonical = entry.get("canonical_name", "")
        confidence = entry.get("confidence", 0.0)

        antonym = _contains_antonym_pair(members)
        if antonym:
            entry["_antonym_guard_triggered"] = antonym
            antonym_flagged.append(entry)
            continue

        if entry.get("out_of_scope"):
            out_of_scope.append(entry)
            continue

        if entry.get("needs_review") or confidence < confidence_threshold:
            needs_review.append(entry)
            continue

        for m in members:
            auto_map[m] = canonical
        auto_map[canonical] = canonical

    output_path.write_text(
        json.dumps(
            {"merges": auto_map, "generated_by": "llm_canonicalise.py (fixed)",
             "auto_applied_count": len(auto_map)},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    review_path = output_path.with_name(
        output_path.stem + "_needs_review.json"
    )
    review_path.write_text(
        json.dumps(needs_review + antonym_flagged, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    scope_path = output_path.with_name(
        output_path.stem + "_out_of_scope.json"
    )
    scope_path.write_text(
        json.dumps(out_of_scope, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nAuto-applied merges  → {output_path} ({len(auto_map)} raw→canonical mappings)")
    print(f"Needs human review   → {review_path} ({len(needs_review)} clusters, "
          f"{len(antonym_flagged)} caught by antonym guard)")
    print(f"Out of scope         → {scope_path} ({len(out_of_scope)} clusters)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    if args.report_json:
        clusters = _extract_clusters_from_report(Path(args.report_json))
    elif args.clusters_json:
        clusters = _load_clusters_json(Path(args.clusters_json))
    else:
        print("error: need --report-json or --clusters-json", file=sys.stderr)
        return 1

    if not clusters:
        print("No clusters found in input.")
        return 0

    if not args.triplets_csv:
        print(
            "error: --triplets-csv is required to build grounding examples — "
            "canonicalising from bare names alone reproduces the original bug.",
            file=sys.stderr,
        )
        return 1

    triplets_df = pd.read_csv(args.triplets_csv)
    grounding = _build_grounding_lookup(triplets_df)

    print(f"Loaded {len(clusters)} clusters "
          f"({sum(len(v) for v in clusters.values())} names), "
          f"grounding for {len(grounding)} distinct predicates")

    t0 = time.time()
    cluster_results = asyncio.run(
        _llm_canonicalise(
            args.base_url, args.api_key, args.model, clusters, grounding,
            concurrency=args.concurrency,
        )
    )
    print(f"Done in {time.time()-t0:.1f}s")

    _save_results(cluster_results, Path(args.output),
                  confidence_threshold=args.confidence_threshold)

    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-driven canonicalisation of relation names (grounded)"
    )
    p.add_argument("--report-json", type=str,
                   help="Flag-mode report JSON (has cluster groups)")
    p.add_argument("--clusters-json", type=str,
                   help="Hand-curated clusters JSON file")
    p.add_argument("--triplets-csv", type=str, required=False,
                   help="Entailed/validated triplets CSV — REQUIRED, supplies "
                        "grounding examples for each cluster member")
    p.add_argument("--base-url", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--api-key", type=str, default="")
    p.add_argument("--output", type=str, default="canonical_map.json")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Concurrent LLM calls (one call per cluster)")
    p.add_argument("--confidence-threshold", type=float, default=0.75,
                   help="Below this, cluster goes to needs_review instead of auto-applying")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())