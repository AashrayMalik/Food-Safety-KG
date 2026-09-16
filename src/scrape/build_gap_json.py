#!/usr/bin/env python3
"""
Build the website JSON for the adulterant gap analysis from the curated
`adulterant_gap_analysis.csv` (authoritative GAP/COVERED + category labels),
enriched with context from the news triplets (object type, found-in foods,
evidence, near-miss check against lab materials).

Output: ../food-safety-observatory-with-changes/data/adulterant-comparison.json
"""
import csv
import json
from collections import Counter, defaultdict

from compare_adulterants import (
    NEWS_PATH,
    SCOPE_FILES,
    LABS_XLSX,
    LABS_SHEET,
    normalize,
    dedupe,
    load_news_terms,
    load_scope,
    load_labs,
)

GAP_CSV = "adulterant_gap_analysis.csv"
RESULTS = "results.csv"
JSON_OUT = "../food-safety-observatory-with-changes/data/adulterant-comparison.json"


def find_ctx(term, news):
    """Best-effort context lookup: exact dedupe match, else substring both ways."""
    dk = dedupe(normalize(term))
    if dk in news:
        return news[dk]
    merged = {"object_type": "", "foods": set(), "evidence": set(), "articles": 0}
    for key, info in news.items():
        if dk in key or key in dk:
            if not merged["object_type"]:
                merged["object_type"] = info["object_type"]
            merged["foods"] |= info["foods"]
            merged["evidence"] |= info["evidence"]
            merged["articles"] += info["articles"]
    return merged if merged["articles"] else None


def main():
    # their curated CSV
    gaps_specific = []
    gaps_other = defaultdict(list)
    covered = []
    for r in csv.DictReader(open(GAP_CSV)):
        term = r["adulterant_term_from_news"].strip()
        cnt = int(r["news_mention_count"])
        cat = (r["category"] or "").strip()
        status = (r["lab_scope_status"] or "").strip()
        if status == "COVERED":
            covered.append({"term": term, "mention_count": cnt})
            continue
        if cat == "specific-substance":
            gaps_specific.append({"term": term, "mention_count": cnt})
        elif "product" in cat:
            gaps_other["product"].append({"term": term, "mention_count": cnt})
        elif "vague" in cat or "generic" in cat:
            gaps_other["vague"].append({"term": term, "mention_count": cnt})
        elif "pest" in cat or "physical" in cat:
            gaps_other["pest"].append({"term": term, "mention_count": cnt})
        else:  # brand / proper-noun extraction noise
            gaps_other["noise"].append({"term": term, "mention_count": cnt})

    print(f"gap specific={len(gaps_specific)} other={ {k: len(v) for k, v in gaps_other.items()} } covered={len(covered)}")

    news = load_news_terms(NEWS_PATH)
    _, total_labs, materials = load_scope(SCOPE_FILES)
    materials_blob = "\n".join(sorted(materials))
    labs = load_labs(RESULTS, LABS_XLSX, LABS_SHEET)

    for g in gaps_specific:
        ctx = find_ctx(g["term"], news)
        if ctx:
            foods = sorted(ctx["foods"])
            food_match = [f for f in foods if normalize(f) in materials_blob]
            g["object_type"] = (ctx["object_type"] or "").replace("fflo:", "")
            g["foods"] = foods[:8]
            g["evidence"] = next(iter(ctx["evidence"]), "") if ctx["evidence"] else ""
            g["in_scope"] = "yes" if food_match else "no"
            g["food_match"] = ", ".join(food_match[:3])
        else:
            g["object_type"] = ""
            g["foods"] = []
            g["evidence"] = ""
            g["in_scope"] = "unknown"
            g["food_match"] = ""

    gaps_specific.sort(key=lambda g: -g["mention_count"])
    for k in gaps_other:
        gaps_other[k].sort(key=lambda g: -g["mention_count"])
    covered.sort(key=lambda g: -g["mention_count"])

    payload = {
        "summary": {
            "labs": len(labs),
            "integrated": sum(1 for l in labs if l["doc_type"] == "integrated"),
            "scope": sum(1 for l in labs if l["doc_type"] == "scope"),
            "gap_specific": len(gaps_specific),
            "gap_product": len(gaps_other["product"]),
            "gap_vague": len(gaps_other["vague"]),
            "gap_pest": len(gaps_other["pest"]),
            "gap_noise": len(gaps_other["noise"]),
            "gap_total": len(gaps_specific) + sum(len(v) for v in gaps_other.values()),
            "covered": len(covered),
        },
        "labs": labs,
        "gap": gaps_specific,
        "gap_other": dict(gaps_other),
        "covered": covered,
    }
    with open(JSON_OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {JSON_OUT}")


if __name__ == "__main__":
    main()
