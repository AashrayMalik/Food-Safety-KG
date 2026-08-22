#!/usr/bin/env python3
"""Unconstrained extraction + NLI flagging for schema validation.

Runs triplet extraction WITHOUT the FFLO schema — the model freely
names entity types and relations based on observed text.  Then:

1. Normalises free-form names via string heuristics (no embeddings).
2. Runs the cross-encoder NLI model on every triplet.
3. Flags triplets with questionable entailment for human review.
4. Compares extracted types/relations against the FFLO schema to
   identify covered concepts vs. potential gaps.

Usage::

    # Step 1 — Extract unconstrained triplets
    food_lab/bin/python src/validation/unconstrained_pipeline.py \\
        --mode extract \\
        --chunks-csv  src/data/FSSAI_docs/processed/chunks.csv \\
        --output-dir  src/outputs/unconstrained \\
        --api-key     \$DEEPSEEK_API_KEY \\
        --concurrency 10

    # Step 2 — Normalise + NLI-flag + schema-compare
    food_lab/bin/python src/validation/unconstrained_pipeline.py \\
        --mode flag \\
        --triplets-dir  src/outputs/unconstrained \\
        --chunks-csv    src/data/FSSAI_docs/processed/chunks.csv \\
        --nli-model     cross-encoder/nli-deberta-v3-small \\
        --output-dir    src/outputs/unconstrained \\
        --device        cpu
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Allow importing from sibling directories
_here = Path(__file__).resolve().parent
_val = _here.parent / "validation"
if str(_val) not in sys.path:
    sys.path.insert(0, str(_val))

from normalize import normalise_names, canonicalise
from normalize import save_canonical_map as _save_canonical_map
from normalize import load_canonical_map as _load_canonical_map
from normalize import apply_canonical_map
from normalize import cluster_by_embedding
from verbalise import verbalise as _verbalise_triplet

# ---- Constants ----
UNCONSTRAINED_PROMPT = Path(__file__).resolve().parent.parent.parent / \
    "prompts" / "unconstrained_extraction.txt"

# ---- Negation markers ----
_NEGATION_WORDS = frozenset({
    "not", "no", "never", "shall not", "free from", "exempt",
    "without", "excluded", "exclude", "absence", "absent",
})

def _is_negated(span: str) -> bool:
    lower = span.lower()
    return any(w in lower for w in _NEGATION_WORDS)


# ---- FFLO domain/range rules (lazy-loaded from extraction) ----
_FFLO_DOMAIN_RANGE: dict[str, list[tuple[frozenset[str], frozenset[str]]]] = {}


def _load_fflo_domain_range():
    global _FFLO_DOMAIN_RANGE
    if _FFLO_DOMAIN_RANGE:
        return _FFLO_DOMAIN_RANGE

    import sys as _sys
    _ext = Path(__file__).resolve().parent
    if str(_ext) not in _sys.path:
        _sys.path.insert(0, str(_ext))
    from extract_triplets import RELATION_DOMAIN_RANGE as _dr

    for rel_canon, pairs in _dr.items():
        canon_pairs: list[tuple[frozenset[str], frozenset[str]]] = []
        for dom, rng in pairs:
            canon_pairs.append((
                frozenset(canonicalise(d) for d in dom),
                frozenset(canonicalise(r) for r in rng),
            ))
        _FFLO_DOMAIN_RANGE[rel_canon] = canon_pairs
    return _FFLO_DOMAIN_RANGE

_FFLO_ENTITY_TYPES_CANON = {
    canonicalise(t) for t in [
        "fflo:FoodBusinessOperator", "fflo:AdulterationAct",
        "fflo:Adulterant", "fflo:SyntheticAdulterant",
        "fflo:NaturalAdulterant", "fflo:ContaminantAdulterant",
        "fflo:AdulterationMethod", "fflo:FraudType",
        "fflo:IntentionalityLevel", "fflo:FoodAdditive",
        "fflo:AdditiveFunction", "fflo:Preservative", "fflo:Antioxidant",
        "fflo:Emulsifier", "fflo:Stabilizer", "fflo:AcidityRegulator",
        "fflo:FlourTreatmentAgent", "fflo:Sequestrant",
        "fflo:HumectantAdditive", "fflo:Colourant",
        "fflo:SweeteningAgent", "fkg:Food", "fkg:Ingredient",
        "fkg:ChemicalIngredient", "fflo:Microorganism",
        "fflo:ProcessingAid", "fflo:ObligationType",
        "fflo:LabelDeclaration", "fflo:MarkAffixation",
        "fflo:SignatureRequirement", "fflo:PackagingRequirement",
        "fflo:FormFiling", "fflo:TestingRequirement",
        "fflo:ConformanceRequirement", "fflo:ProductionRequirement",
        "fflo:SupplyChainStep",
        "fflo:Production", "fflo:Processing", "fflo:Storage",
        "fflo:Distribution", "fflo:Retail", "fflo:Import",
        "fflo:SpreadEvent", "fflo:TransformationEvent",
        "fflo:GeographicRegion", "fso:Sample", "fso:SampleType",
        "fso:Analysis", "fso:AnalysisResult", "ssn:Property",
        "fso:Measurement", "fso:UnitOfMeasure", "fso:Laboratory",
        "fso:Location", "fflo:IncidentFinding", "fflo:LabConfirmed",
        "fflo:FieldDetected", "fflo:SurveyAggregated",
        "fflo:RecallTriggered", "fflo:EvidenceSource",
        "fflo:DetectionMethod", "fflo:LaboratoryTest",
        "fflo:FieldTest", "fflo:SensoryTest",
        "fflo:DetectionIndicator", "fflo:DetectionKit",
        "fflo:SamplingPlan",
        "fflo:FoodStandard", "fflo:RegulatoryDocument",
        "fflo:RegulatoryBody", "fflo:RegulatoryAction",
        "fflo:PermissibleLimit", "fflo:Violation", "fflo:FoodCategory",
        "fflo:TableReference",
        "fflo:RegulatoryOfficer", "fflo:RegulatoryActionType",
        "fflo:Seizure", "fflo:Removal", "fflo:PenaltyReduction",
        "fflo:Authorization", "fflo:Sealing",
        "fflo:HealthEffect",
        "fflo:CausalHealthEffect", "fflo:AcuteEffect",
        "fflo:ChronicEffect", "fflo:VulnerablePopulation",
    ]
}

_FFLO_RELATIONS_CANON = {
    canonicalise(r) for r in [
        "prov:wasAssociatedWith", "prov:used", "prov:wasGeneratedBy",
        "prov:wasAttributedTo", "fflo:hasAdulterant",
        "fflo:hasAdulterationMethod", "fflo:hasFraudType",
        "fflo:hasIntentionality", "fflo:isSubstituteFor",
        "fflo:commonIn", "fflo:foundAt", "fflo:hasFunction",
        "fkg:hasIngredient", "prov:wasInfluencedBy",
        "fflo:propagatesTo", "fflo:carriedBy", "fflo:occursAt",
        "fflo:inputTo", "fflo:outputOf", "fso:isPerformedOn",
        "fso:isPerformedAt", "fso:isResultOf", "fso:relatesToProperty",
        "fso:isMeasuredIn", "fso:hasSampleType", "fso:hasLocation",
        "fflo:hasNumericValue", "fflo:identifies", "fflo:foundIn",
        "fflo:producedBy", "fflo:confirmedBy", "fflo:supportedBy",
        "fflo:foundAtStep", "fflo:inRegion", "fflo:triggeredAction",
        "fflo:collectedAt", "fflo:detectedBy",
        "fflo:producesIndicator", "fflo:requiresKit",
        "fflo:isPerformedAs", "lkif:created_by",
        "fflo:belongsToCategory", "fflo:appliesToCategory",
        "fflo:hasValue", "fflo:forProperty", "fflo:comparedAgainst",
        "fflo:definedIn", "fflo:defines", "fflo:appliesTo",
        "fflo:inFood", "fflo:hasPermissibleLimit",
        "fflo:specifiedIn", "fflo:appliesToFood",
        "fflo:compliesWith", "fflo:amends", "fflo:hasSubcategory",
        "fflo:hasDefinition", "fflo:tableLabel", "fflo:issuedBy",
        "fflo:targets", "fflo:citesFinding", "fflo:constitutes",
        "fflo:violates", "fflo:causesEffect",
        "fflo:affectsPopulation", "fflo:hasEffectType",
        "fflo:associatedWith",
        "fflo:hasObligation", "fflo:hasScientificName",
        "fflo:governsSampling", "fflo:isEmpoweredTo",
        "fflo:actsOnBehalfOf",
    ]
}


# ---------------------------------------------------------------------------
# Step 1 — Run unconstrained extraction
# ---------------------------------------------------------------------------

def run_extraction(args: argparse.Namespace) -> int:
    """Run the standard extraction pipeline with the unconstrained prompt."""
    import subprocess

    cmd = [
        sys.executable,
        "src/extraction/extract_triplets.py",
        "--chunks-csv", str(args.chunks_csv),
        "--prompt-file", str(UNCONSTRAINED_PROMPT),
        "--output-dir", str(args.output_dir),
        "--model", args.model,
        "--concurrency", str(args.concurrency),
        "--backend", args.backend,
    ]
    if args.api_key:
        cmd += ["--api-key", args.api_key]
    if args.base_url:
        cmd += ["--base-url", args.base_url]
    if args.resume:
        cmd.append("--resume")
    if args.backend == "hf_transformers":
        if args.hf_device:
            cmd += ["--hf-device", args.hf_device]
        if args.hf_dtype:
            cmd += ["--hf-dtype", args.hf_dtype]
        if args.hf_quantize:
            cmd += ["--hf-quantize", args.hf_quantize]
        if args.hf_trust_remote_code:
            cmd.append("--hf-trust-remote-code")

    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


# ---------------------------------------------------------------------------
# Step 2 — Normalise + NLI-flag + schema-compare
# ---------------------------------------------------------------------------

def _load_triplets_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return items


def run_flagging(args: argparse.Namespace) -> int:
    import numpy as np
    from nli_model import NLIModel

    triplets_dir = Path(args.triplets_dir)
    chunks_csv = Path(args.chunks_csv)
    output_dir = Path(args.output_dir)

    # ---- Load chunk text ----
    chunk_text: dict[str, str] = {}
    if chunks_csv.exists():
        with chunks_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                chunk_text[row["snippet_id"]] = row.get("evidence_text", "")

    # ---- Load triplets ----
    trips_path = triplets_dir / "triplets.jsonl"
    if not trips_path.exists():
        subdirs = sorted(triplets_dir.glob("**/triplets.jsonl"))
        if subdirs:
            trips_path = subdirs[0]
            print(f"Auto-discovered: {trips_path}")
    if not trips_path.exists():
        print(f"error: {trips_path} not found — run --mode extract first",
              file=sys.stderr)
        return 1
    entries = _load_triplets_jsonl(trips_path)
    all_triplets: list[dict[str, Any]] = []
    for e in entries:
        sid = e.get("snippet_id", "")
        for t in e.get("triplets", []):
            t["snippet_id"] = sid
            t.setdefault("subject_id", "")
            t.setdefault("object_id", "")
            all_triplets.append(t)
    print(f"Loaded {len(entries)} chunks, {len(all_triplets)} triplets")

    # ---- Collect raw types & relations ----
    raw_etypes = []
    raw_rels = []
    for t in all_triplets:
        raw_etypes.append(t.get("subject_type", ""))
        raw_etypes.append(t.get("object_type", ""))
        raw_rels.append(t.get("predicate", ""))

    # ---- Pre-apply canonical map (human-verified merges) ----
    pre_merges: dict[str, str] = {}
    if args.canonical_map and args.canonical_map.exists():
        pre_merges = _load_canonical_map(args.canonical_map)
        if pre_merges:
            raw_etypes = apply_canonical_map(raw_etypes, pre_merges)
            raw_rels = apply_canonical_map(raw_rels, pre_merges)
            print(f"\nApplied {len(pre_merges)} pre-merges from {args.canonical_map}")

    # ---- Normalise ----
    print("\nNormalising entity types...")
    etype_norm = normalise_names(raw_etypes)
    print(f"  Raw types: {len(etype_norm['frequency'])} canonical forms")
    print(f"  Near-duplicate groups: {len(etype_norm['near_duplicates'])}")
    if etype_norm["near_duplicates"]:
        print("  Examples:")
        for g in etype_norm["near_duplicates"][:5]:
            print(f"    {g}")

    print("\nNormalising relations...")
    rel_norm = normalise_names(raw_rels)
    print(f"  Raw relations: {len(rel_norm['frequency'])} canonical forms")
    print(f"  Near-duplicate groups: {len(rel_norm['near_duplicates'])}")
    if rel_norm["near_duplicates"]:
        print("  Examples:")
        for g in rel_norm["near_duplicates"][:5]:
            print(f"    {g}")

    # ---- Embedding-based clustering (semantic similarity) ----
    # Collect evidence samples + type pairs per relation name
    ev_samples: dict[str, list[str]] = defaultdict(list)
    type_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for t in all_triplets:
        pc = rel_norm["canonical_map"].get(t.get("predicate", ""), "")
        if not pc:
            continue
        span = t.get("evidence_span", "")
        if span:
            ev_samples[pc].append(span)
        sc = etype_norm["canonical_map"].get(t.get("subject_type", ""), "")
        oc = etype_norm["canonical_map"].get(t.get("object_type", ""), "")
        if sc and oc:
            type_pairs[pc].add((sc, oc))

    rel_names = sorted(rel_norm["frequency"].items(), key=lambda x: -x[1])
    embedding_threshold = getattr(args, "embedding_threshold", 0.82)
    sug, flg = [], []
    print(f"\nEmbedding clustering (threshold={embedding_threshold})...")
    try:
        sug, flg = cluster_by_embedding(
            rel_names, ev_samples, dict(type_pairs),
            threshold=embedding_threshold,
        )
        if sug:
            print(f"  Suggested merges: {len(sug)} groups")
            for g in sug[:5]:
                print(f"    {g}")
        if flg:
            print(f"  Flagged (type-incompatible): {len(flg)} groups")
            for g in flg[:5]:
                print(f"    {g}")
    except ImportError:
        print("  (sentence-transformers not available, skipping)")
    except Exception as exc:
        print(f"  (embedding clustering failed: {exc})")

    # ---- Schema comparison ----
    etypes_can = set(etype_norm["frequency"].keys())
    rels_can = set(rel_norm["frequency"].keys())

    covered_types = etypes_can & _FFLO_ENTITY_TYPES_CANON
    new_types = etypes_can - _FFLO_ENTITY_TYPES_CANON
    covered_rels = rels_can & _FFLO_RELATIONS_CANON
    new_rels = rels_can - _FFLO_RELATIONS_CANON

    print(f"\nSchema comparison:")
    print(f"  Entity types covered by FFLO: {len(covered_types)}/{len(etypes_can)}")
    print(f"  Potential new entity types:   {len(new_types)}")
    print(f"  Relations covered by FFLO:    {len(covered_rels)}/{len(rels_can)}")
    print(f"  Potential new relations:      {len(new_rels)}")

    # ---- Save canonical map (auto-gen from near-dupes, editable by human) ----
    cmap_path = args.canonical_map or (output_dir / "canonical_map.json")
    if not pre_merges:
        _save_canonical_map({}, cmap_path)
        print(f"  Canonical map (empty template) → {cmap_path}")

    # ---- NLI flagging ----
    print(f"\nLoading NLI model: {args.nli_model}")
    nli = NLIModel(args.nli_model, device=args.device, max_length=512)

    t0 = time.time()
    premises = []
    hypotheses = []
    t_meta = []
    for t in all_triplets:
        sid = t.get("snippet_id", "")
        subj = t.get("subject", "")
        stype = t.get("subject_type", "")
        pred = t.get("predicate", "")
        obj = t.get("object", "")
        otype = t.get("object_type", "")

        hyp = _verbalise_triplet(subj, stype, pred, obj, otype)
        t_neg = _is_negated(t.get("evidence_span", ""))
        if t_neg:
            hyp = f"It is NOT the case that {hyp}"
        premise = chunk_text.get(sid, "")
        premises.append(premise)
        hypotheses.append(hyp)
        t_meta.append({
            "snippet_id": sid, "subject": subj, "predicate": pred,
            "object": obj, "verbalisation": hyp,
        })

    print(f"Running NLI on {len(premises)} pairs...")
    nli_results = nli.predict(premises, hypotheses,
                              batch_size=args.batch_size)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(premises)/elapsed:.0f} pairs/s)")

    # ---- Attach normalisation + NLI metadata to each triplet ----
    for i, t in enumerate(all_triplets):
        t["nli_label"] = nli_results[i]["label"]
        t["nli_confidence"] = nli_results[i]["confidence"]
        t["nli_scores"] = nli_results[i]["scores"]
        t["nli_verdict"] = NLIModel.map_to_verdict(nli_results[i]["label"])
        t["verbalisation"] = t_meta[i]["verbalisation"]
        t["subject_type_canon"] = etype_norm["canonical_map"].get(
            t.get("subject_type", ""), "")
        t["object_type_canon"] = etype_norm["canonical_map"].get(
            t.get("object_type", ""), "")
        t["predicate_canon"] = rel_norm["canonical_map"].get(
            t.get("predicate", ""), "")
        # --- schema status (3-way: covered / domain_range_violation / new) ---
        dr = _load_fflo_domain_range()
        pred_c = t.get("predicate_canon", "")
        st_c = t.get("subject_type_canon", "")
        ot_c = t.get("object_type_canon", "")

        name_in_fflo = (
            st_c in _FFLO_ENTITY_TYPES_CANON
            and ot_c in _FFLO_ENTITY_TYPES_CANON
            and pred_c in _FFLO_RELATIONS_CANON
        )
        if not name_in_fflo:
            t["schema_status"] = "new"
        elif pred_c in dr and dr[pred_c]:
            pairs = dr[pred_c]
            dom_ok = any(
                (not d or st_c in d)
                for d, _r in pairs
            )
            rng_ok = any(
                (not r or ot_c in r)
                for _d, r in pairs
            )
            if dom_ok and rng_ok:
                t["schema_status"] = "covered"
            else:
                t["schema_status"] = "domain_range_violation"
        else:
            t["schema_status"] = "covered"

        t["in_fflo_schema"] = t["schema_status"] == "covered"  # backward compat

    # ---- Multi-reason flagging (UNMAPPED is unconditional) ----
    for t in all_triplets:
        reasons: list[str] = []
        pred = t.get("predicate", "")

        if pred == "UNMAPPED":
            reasons.append("unmapped")
        if t["nli_label"] in ("neutral", "contradiction"):
            reasons.append("nli_not_entailed")
        if t["nli_confidence"] < 0.7:
            reasons.append("nli_low_confidence")
        if not t["in_fflo_schema"]:
            if t.get("schema_status") == "domain_range_violation":
                reasons.append("domain_range_violation")
            else:
                reasons.append("not_in_fflo")
        t["flag_reasons"] = reasons
        t["flagged"] = len(reasons) > 0

    flagged = [t for t in all_triplets if t["flagged"]]
    multi = sum(1 for t in flagged if len(t["flag_reasons"]) >= 2)
    unmapped = sum(1 for t in flagged if "unmapped" in t["flag_reasons"])
    drv = sum(1 for t in all_triplets
              if t.get("schema_status") == "domain_range_violation")
    print(f"\nFlagged for review: {len(flagged)}/{len(all_triplets)} "
          f"({multi} multi-reason, {unmapped} UNMAPPED, "
          f"{drv} domain/range violations)")

    # ---- Save flagged CSV ----
    flagged_csv = output_dir / "unconstrained_flagged.csv"
    fieldnames = [
        "snippet_id", "subject", "subject_type", "subject_type_canon",
        "predicate", "predicate_canon",
        "object", "object_type", "object_type_canon",
        "confidence", "evidence_span",
        "nli_label", "nli_confidence", "nli_verdict",
        "verbalisation", "in_fflo_schema", "schema_status",
        "flag_reasons",
    ]
    with flagged_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for t in flagged:
            row = dict(t)
            row["flag_reasons"] = ";".join(t["flag_reasons"])
            writer.writerow(row)
    print(f"  → {flagged_csv}")

    # ---- Save full normalised triplets JSONL ----
    full_jsonl = output_dir / "unconstrained_nli_triplets.jsonl"
    with full_jsonl.open("w", encoding="utf-8") as f:
        for t in all_triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"  → {full_jsonl}")

    # ---- Summary report ----
    verdict_dist = Counter(t["nli_verdict"] for t in all_triplets)
    label_dist = Counter(t["nli_label"] for t in all_triplets)
    pred_can_dist = Counter(t["predicate_canon"] for t in all_triplets)

    report = {
        "total_triplets": len(all_triplets),
        "nli_verdicts": dict(verdict_dist),
        "nli_labels": dict(label_dist),
        "flagged_count": len(flagged),
        "multi_reason_count": multi,
        "unmapped_count": unmapped,
        "by_reason": {
            reason: sum(
                1 for t in all_triplets
                if reason in t.get("flag_reasons", [])
            )
            for reason in ("unmapped", "nli_not_entailed",
                           "nli_low_confidence", "not_in_fflo",
                           "domain_range_violation")
        },
        "domain_range_violations": drv,
        "schema_coverage": {
            "covered": sum(1 for t in all_triplets
                           if t.get("schema_status") == "covered"),
            "domain_range_violation": drv,
            "new": sum(1 for t in all_triplets
                       if t.get("schema_status") == "new"),
        },
        "schema_comparison": {
            "types_covered": len(covered_types),
            "types_new": len(new_types),
            "new_type_examples": sorted(list(new_types))[:20],
            "relations_covered": len(covered_rels),
            "relations_new": len(new_rels),
            "new_relation_examples": sorted(list(new_rels))[:20],
        },
        "normalisation": {
            "entity_type_near_dup_groups": len(
                etype_norm["near_duplicates"]
            ),
            "relation_near_dup_groups": len(
                rel_norm["near_duplicates"]
            ),
            "embedding_suggested_merges": len(sug),
            "embedding_flagged_type_incompatible": len(flg),
            "embedding_threshold": embedding_threshold,
        },
        "top_canonical_relations": pred_can_dist.most_common(20),
    }

    report_path = output_dir / "unconstrained_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  → {report_path}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unconstrained extraction + NLI flagging for schema validation"
    )
    p.add_argument("--mode", type=str, required=True,
                   choices=["extract", "flag"],
                   help="Step to run: extract triplets, or flag with NLI")
    # --- shared ---
    p.add_argument("--chunks-csv", type=Path, required=True,
                   help="Path to chunks.csv")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory")
    # --- extract ---
    p.add_argument("--model", type=str, default="deepseek-v4-flash",
                   help="Model for extraction")
    p.add_argument("--api-key", type=str, default="",
                   help="API key for extraction")
    p.add_argument("--concurrency", type=int, default=10,
                   help="Concurrency for extraction")
    p.add_argument("--backend", type=str, default="openai_compat",
                   choices=["openai_compat", "hf_transformers"],
                   help="Extraction backend")
    p.add_argument("--base-url", type=str, default="",
                   help="API base URL (for vLLM/TGI/local server)")
    p.add_argument("--hf-device", type=str, default="auto",
                   help="Device for HF backend")
    p.add_argument("--hf-dtype", type=str, default="auto",
                   help="Dtype for HF backend")
    p.add_argument("--hf-quantize", type=str, default="none",
                   help="Quantisation for HF backend")
    p.add_argument("--hf-trust-remote-code", action="store_true",
                   help="Pass trust_remote_code to HF")
    p.add_argument("--resume", action="store_true",
                   help="Resume extraction")
    # --- flag ---
    p.add_argument("--triplets-dir", type=str, default="",
                   help="Directory with unconstrained triplets.jsonl")
    p.add_argument("--nli-model", type=str,
                   default="cross-encoder/nli-deberta-v3-small",
                   help="NLI model path")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for NLI inference")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for NLI model")
    p.add_argument("--canonical-map", type=Path, default=None,
                   help="JSON file with human-reviewed merge decisions "
                        "(auto-generated on first run)")
    p.add_argument("--embedding-threshold", type=float, default=0.82,
                   help="Cosine similarity threshold for embedding-based "
                        "relation clustering (default: 0.82)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.mode == "extract":
        return run_extraction(args)
    return run_flagging(args)


if __name__ == "__main__":
    sys.exit(main())
