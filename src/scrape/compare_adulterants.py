#!/usr/bin/env python3
"""
Compare NABL accredited test scopes against known adulterant/contaminant terms
from two sources, kept SEPARATE (news and FSSAI/Qwen).

The headline finding is "what the news surfaces but no accredited lab tests
for": news adulterants with NO lab coverage, enriched with context (object
type, associated foods, evidence, near-miss check against lab materials).

Writes:
  adulterant_coverage.csv  - each known term -> tested by any lab? (yes/no)
  new_emerging.csv         - scope params in adulterant/contaminant categories
                             not in news/qwen (secondary "labs test extra" list)
  --json-out               - summary JSON for the website
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict

NEWS_PATH = "../outputs/triplets/news/News_Triplets_Oil_Ghee_Milk_20260820.csv"
QWEN_PATH = "../outputs/triplets/qwen/triplets_canonicalized.csv"
SCOPE_FILES = [
    "scope_entries_chemical.csv",
    "scope_entries_biological.csv",
    "scope_entries_other.csv",
]
LABS_XLSX = "FSSAI_NABL_Certificate_Numbers.xlsx"
LABS_SHEET = "Annexure1_Notified"

# Generic / noise terms that are not specific adulterants — excluded from the
# headline "uncovered" list but reported separately.
GENERIC_TERMS = {
    "unspecified adulterant", "unknown adulterant", "unknown contaminant",
    "adulterant", "adulterants", "contaminant", "contaminants", "chemicals",
    "harmful chemicals", "chemical substances", "substandard ingredients",
    "adulterated substance", "essence", "artificial essence", "synthetic essence",
    "ghee essence", "dust", "poison", "poisoning", "pesticide residues",
    "heavy metals", "impurities", "impurity", "toxin", "toxins", "spurious",
    "substandard", "adulterated ghee", "spurious ghee", "duplicate ghee",
    "fake ghee", "adulterated milk", "cheap oils", "insects", "insect",
    "debris", "foreign matter", "extraneous matter", "fungus", "food", "milk",
    "ghee", "oil", "salt", "sugar", "water",
}


def normalize(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def dedupe(s):
    return re.sub(r"(.)\1+", r"\1", s)


GENERIC_KEYS = {dedupe(normalize(t)) for t in GENERIC_TERMS}


# --- categorization lexicons -------------------------------------------------
METALS = [
    "lead", "arsenic", "mercury", "cadmium", "chromium", "nickel", "tin",
    "antimony", "aluminium", "aluminum", "barium", "beryllium", "thallium",
    "cobalt", "silver", "vanadium", "titanium", "lithium", "bismuth",
]
MYCOTOXINS = [
    "aflatoxin", "ochratoxin", "patulin", "citrinin", "fumonisin",
    "zearalenone", "deoxynivalenol", "sterigmatocystin", "ergot",
    "t-2 toxin", "ht-2", "mycotoxin",
]
ADULTERANT_SUBS = [
    "adulterat", "detection of", "presence of", "test for", "test of",
    "neutrali", "foreign fat", "animal fat", "animal body fat", "vegetable fat in",
    "mineral oil", "vanaspati", "argemone", "metanil yellow",
    "synthetic milk", "synthetic colour", "synthetic color", "synthetic dye",
    "added colour", "added color", "added colouring", "non-permitted",
    "reconstituted", "extraneous water", "added water",
    "reichert", "polenske", "butyro", "b.r.", "b r ", "br value",
    "iodine value", "saponification value", "refractive index", "kerosene",
    "urea", "detergent", "formalin", "melamine", "maltodextrin",
    "hydrogen peroxide", "caustic", "ethylene glycol", "starch",
    "artificial colour", "artificial color", "artificial sweetener",
]
MICRO_SUBS = [
    "coli", "salmonella", "aureus", "bacillus", "clostridium", "listeria",
    "yeast", "mould", "mold", "coliform", "enterobacter", "staphylococcus",
    "vibrio", "campylobacter", "shigella", "plate count", "aerobic",
    "anaerobic", "fungi", "fungus", "pseudomonas", "candida", "aspergillus",
]

METAL_RE = re.compile(r"\b(?:" + "|".join(re.escape(m) for m in METALS) + r")\b")
MYCO_RE = re.compile(r"\b(?:" + "|".join(re.escape(m) for m in MYCOTOXINS) + r")\b")
MICRO_SUBS_D = [dedupe(s) for s in MICRO_SUBS]
ADULTERANT_SUBS_D = [dedupe(s) for s in ADULTERANT_SUBS]


def categorize(param_dedup, dg_dedup):
    """Classify a scope parameter into a coarse contaminant category.

    Checks (in order) for metal, mycotoxin, residue/contaminant,
    microorganism, and adulterant keyword matches, defaulting to "other".
    """
    if METAL_RE.search(param_dedup):
        return "metal"
    if MYCO_RE.search(param_dedup):
        return "mycotoxin"
    if "RESIDUE" in dg_dedup.upper() or "CONTAMINANT" in dg_dedup.upper():
        return "contaminant"
    if any(m in param_dedup for m in MICRO_SUBS_D):
        return "microorganism"
    if any(m in param_dedup for m in ADULTERANT_SUBS_D):
        return "adulterant"
    return "other"


# --- vocab loading -----------------------------------------------------------
def _key(obj):
    return dedupe(normalize(obj))


def load_news_terms(path):
    """Load news adulterant terms (predicate ``fflo:hasAdulterant``).

    Returns a dedup-keyed dict aggregating associated foods, evidence
    spans, and article counts per term.
    """
    terms = {}
    for r in _rows(path):
        if r.get("predicate") != "fflo:hasAdulterant":
            continue
        obj = r.get("object", "").strip()
        if not obj:
            continue
        ot = r.get("object_type", "") or ""
        cat = "contaminant" if "Contaminant" in ot else "adulterant"
        key = _key(obj)
        if key not in terms:
            terms[key] = {"term": obj, "norm": normalize(obj), "dedup": key,
                          "category": cat, "object_type": ot,
                          "foods": set(), "evidence": set(), "articles": 0}
        t = terms[key]
        subj = r.get("subject", "").strip()
        if subj:
            t["foods"].add(subj)
        ev = r.get("evidence_span", "").strip()
        if ev:
            t["evidence"].add(ev[:200])
        t["articles"] += 1
    return terms


def load_qwen_terms(path):
    """Load FSSAI/Qwen adulterant terms into the same dedup-keyed shape."""
    terms = {}
    for r in _rows(path):
        ot = r.get("object_type", "") or ""
        pred = r.get("predicate", "") or ""
        if pred != "fflo:hasAdulterant" and "Adulterant" not in ot:
            continue
        obj = r.get("object", "").strip()
        if not obj:
            continue
        cat = "contaminant" if "Contaminant" in ot else "adulterant"
        key = _key(obj)
        if key and key not in terms:
            terms[key] = {"term": obj, "norm": normalize(obj), "dedup": key,
                          "category": cat, "object_type": ot,
                          "foods": set(), "evidence": set(), "articles": 0}
    return terms


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --- scope loading -----------------------------------------------------------
def load_scope(files):
    """Load lab scope parameters from CSVs into ``(params, n_labs, materials)``.

    Each parameter is dedup-keyed and enriched with discipline group,
    category, and the set of TC numbers that test it.
    """
    params = {}
    all_tcs = set()
    materials = set()
    for fn in files:
        with open(fn, newline="") as f:
            for r in csv.DictReader(f):
                p = r.get("parameter_tested", "").strip()
                if not p:
                    continue
                key = dedupe(normalize(p))
                tc = r.get("tc_number", "").strip()
                if tc:
                    all_tcs.add(tc)
                mp = r.get("materials_products", "").strip()
                if mp:
                    materials.add(dedupe(normalize(mp)))
                if key not in params:
                    dg = " ".join((r.get("discipline_group", "") or "").split())
                    params[key] = {
                        "param": p, "norm": normalize(p), "dedup": key,
                        "dg": dg, "dg_dedup": dedupe(normalize(dg)),
                        "category": categorize(key, dedupe(normalize(dg))),
                        "tcs": set(),
                    }
                params[key]["tcs"].add(tc)
    return list(params.values()), len(all_tcs), materials


def load_labs(results_path, xlsx_path, sheet):
    """Load labs from results.csv, joining state from the certificate XLSX."""
    tc_to_state = {}
    try:
        import openpyxl
        wb = openpyxl.load_workbook(xlsx_path, data_only=True)
        ws = wb[sheet]
        headers = [c.value for c in ws[1]]
        tc_col = headers.index("NABL Certificate No (TC)")
        state_col = headers.index("State/Region")
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[tc_col]:
                tc_to_state[str(row[tc_col]).strip()] = str(row[state_col] or "").strip()
    except Exception:
        pass

    rows = []
    with open(results_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            tc = r["tc_number"].strip()
            rows.append({"tc": tc, "name": r["lab_name"],
                         "state": tc_to_state.get(tc, ""),
                         "doc_type": r.get("doc_type", "")})
    return rows


def main():
    """Compare lab scopes vs. news/Qwen adulterants and emit coverage CSVs/JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default=NEWS_PATH)
    ap.add_argument("--qwen", default=QWEN_PATH)
    ap.add_argument("--scope", nargs="*", default=SCOPE_FILES)
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--labs-xlsx", default=LABS_XLSX)
    ap.add_argument("--labs-sheet", default=LABS_SHEET)
    ap.add_argument("--coverage-out", default="adulterant_coverage.csv")
    ap.add_argument("--new-out", default="new_emerging.csv")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    print("loading vocabularies...")
    news = load_news_terms(args.news)
    qwen = load_qwen_terms(args.qwen)
    print(f"  news terms: {len(news)} | qwen terms: {len(qwen)}")

    print("loading scope...")
    scope, total_labs, materials = load_scope(args.scope)
    print(f"  scope parameters: {len(scope)} | labs: {total_labs} | materials: {len(materials)}")
    materials_blob = "\n".join(sorted(materials))

    labs = load_labs(args.results, args.labs_xlsx, args.labs_sheet)
    print(f"  labs loaded: {len(labs)}")

    # --- coverage ------------------------------------------------------------
    print("matching coverage...")
    coverage_rows = []
    for source, terms in (("news", news), ("qwen", qwen)):
        for info in terms.values():
            tn, td = info["norm"], info["dedup"]
            matched = 0
            example = None
            for p in scope:
                pn, pd = p["norm"], p["dedup"]
                if tn in pn or td in pd:
                    matched += 1
                    if example is None:
                        example = p["param"]
            coverage_rows.append({
                "term": info["term"], "source": source, "category": info["category"],
                "tested": "yes" if matched else "no",
                "n_labs": matched, "example_param": example or "",
            })

    with open(args.coverage_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["term", "source", "category", "tested", "n_labs", "example_param"])
        w.writeheader()
        w.writerows(coverage_rows)
    print(f"  wrote {len(coverage_rows)} rows -> {args.coverage_out}")

    # --- new/emerging (scope extras, secondary) ------------------------------
    relevant_cats = {"adulterant", "metal", "mycotoxin", "contaminant"}
    news_list = list(news.values())
    qwen_list = list(qwen.values())
    new_rows = []
    for p in scope:
        if p["category"] not in relevant_cats:
            continue
        pn, pd = p["norm"], p["dedup"]
        if any(t["norm"] in pn or t["dedup"] in pd for t in news_list):
            continue
        if any(t["norm"] in pn or t["dedup"] in pd for t in qwen_list):
            continue
        new_rows.append({"parameter": p["param"], "category": p["category"],
                         "discipline_group": p["dg"], "n_labs": len(p["tcs"]),
                         "example_tcs": ",".join(sorted(p["tcs"])[:5])})
    new_rows.sort(key=lambda r: -r["n_labs"])
    with open(args.new_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["parameter", "category", "discipline_group",
                                          "in_news", "in_qwen", "n_labs", "example_tcs"])
        w.writeheader()
        for r in new_rows:
            w.writerow({**r, "in_news": "no", "in_qwen": "no"})
    print(f"  wrote {len(new_rows)} rows -> {args.new_out}")

    # --- uncovered (headline): news terms with no lab coverage ---------------
    tested_news = {r["term"].lower() for r in coverage_rows if r["source"] == "news" and r["tested"] == "yes"}
    uncovered_specific = []
    uncovered_generic = []
    for key, info in news.items():
        if info["term"].lower() in tested_news:
            continue  # already tested
        if key in GENERIC_KEYS:
            uncovered_generic.append(info["term"])
            continue
        foods = sorted(info["foods"])
        food_match = [f for f in foods if normalize(f) in materials_blob]
        in_scope = "yes" if food_match else "no"
        uncovered_specific.append({
            "term": info["term"],
            "object_type": info["object_type"].replace("fflo:", ""),
            "category": info["category"],
            "foods": foods[:8],
            "evidence": next(iter(info["evidence"]), "") if info["evidence"] else "",
            "articles": info["articles"],
            "in_scope": in_scope,
            "food_match": ", ".join(food_match[:3]),
        })
    uncovered_specific.sort(key=lambda r: (-r["articles"], r["term"].lower()))
    uncovered_generic = sorted(set(uncovered_generic))
    print(f"  uncovered specific: {len(uncovered_specific)} | generic: {len(uncovered_generic)}")

    # --- summary + JSON ------------------------------------------------------
    tested = sum(1 for r in coverage_rows if r["tested"] == "yes")
    untested = len(coverage_rows) - tested
    print(f"\ncoverage: {tested} tested / {untested} untested "
          f"| uncovered specific {len(uncovered_specific)} | scope extras {len(new_rows)}")

    if args.json_out:
        tested_rows = sorted(
            [r for r in coverage_rows if r["tested"] == "yes"], key=lambda r: -r["n_labs"])
        payload = {
            "summary": {
                "labs": len(labs),
                "integrated": sum(1 for l in labs if l["doc_type"] == "integrated"),
                "scope": sum(1 for l in labs if l["doc_type"] == "scope"),
                "news_terms": len(news),
                "qwen_terms": len(qwen),
                "coverage_tested": tested,
                "coverage_untested": untested,
                "uncovered_specific": len(uncovered_specific),
                "uncovered_generic": len(uncovered_generic),
                "scope_extras": len(new_rows),
            },
            "labs": labs,
            "uncovered": uncovered_specific,
            "uncovered_generic": uncovered_generic,
            "tested": [{"term": r["term"], "source": r["source"], "n_labs": r["n_labs"]}
                       for r in tested_rows[:30]],
            "scope_extras": [{"parameter": r["parameter"], "category": r["category"],
                              "n_labs": r["n_labs"]} for r in new_rows[:40]],
            "scope_extras_by_category": dict(Counter(r["category"] for r in new_rows)),
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote website JSON -> {args.json_out}")


if __name__ == "__main__":
    main()
