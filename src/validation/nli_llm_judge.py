#!/usr/bin/env python3
"""Two-stage entailment validation: NLI cross-encoder + LLM judge (Qwen).

For every triplet in the unconstrained extraction output:

1. Run chunk-level NLI (premise = full evidence_text)
2. Run span-level NLI (premise = evidence_span)
3. Call local Qwen as LLM judge with both NLI scores
4. Agreed → accepted; disagreed → contested for human review

LLM judge takes precedence in the final verdict callout.

Usage::

    python src/validation/nli_llm_judge.py \\
        --triplets-dir /home/aashray_malik/src/outputs/unconstrained \\
        --chunks-csv   /home/aashray_malik/src/data/FSSAI_docs/processed/chunks.csv \\
        --nli-model    /scratch/aashray_malik/checkpoints \\
        --base-url     http://localhost:8030/v1 \\
        --model        Qwen/Qwen3.5-27B-FP8 \\
        --api-key      aashray-fflo-local \\
        --output-dir   /home/aashray_malik/src/outputs/unconstrained \\
        --device       cuda:0
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

# ---- Lazy imports (heavy deps only loaded when needed) ----


def _get_torch():
    import torch
    return torch


def _get_nli_model():
    import sys as _sys
    _val = Path(__file__).resolve().parent
    if str(_val) not in _sys.path:
        _sys.path.insert(0, str(_val))
    from nli_model import NLIModel
    return NLIModel


def _get_verbalise():
    import sys as _sys
    _val = Path(__file__).resolve().parent
    if str(_val) not in _sys.path:
        _sys.path.insert(0, str(_val))
    from verbalise import verbalise
    return verbalise


# ---- Lazy imports for normalisation ----

def _get_normalise():
    import sys as _sys
    _ext = Path(__file__).resolve().parent.parent / "extraction"
    if str(_ext) not in _sys.path:
        _sys.path.insert(0, str(_ext))
    from normalize import normalise_names, save_canonical_map
    return normalise_names, save_canonical_map


def _get_normalise_tools():
    import sys as _sys
    _ext = Path(__file__).resolve().parent.parent / "extraction"
    if str(_ext) not in _sys.path:
        _sys.path.insert(0, str(_ext))
    from normalize import load_canonical_map, apply_canonical_map
    return load_canonical_map, apply_canonical_map


# ---- LLM canonicalisation prompt (used when --canonicalise is set) ----

CANONICALISE_SYSTEM = """\
You are a knowledge-graph schema curator for food safety regulations.

You will receive GROUPS of similar relation or entity type names from
an extraction pipeline.  Each group contains spelling variants and
near-synonyms for the SAME concept.

For each group, choose ONE canonical umbrella name that best represents
the concept.

Rules:
1. Choose the shortest name that is still descriptive.
2. Use camelCase (e.g. "hasMaximumLevel", not "has_maximum_level").
3. Prefer names matching the style of FFLO relations
   (e.g. "belongsToCategory", "hasValue", "appliesTo").
4. If a group has only ONE name, return that name unchanged.
5. Output ONLY a JSON object — no markdown, no commentary.

Format:
{
  "merges": {
    "cluster_0": "chosenUmbrellaName",
    "cluster_1": "chosenUmbrellaName"
  }
}
"""


# ---- LLM judge prompt (synthetic, user can edit) ----

LLM_JUDGE_SYSTEM = """\
You are an expert entailment validator for food safety knowledge graphs.

A triplet was extracted from a food safety regulation chunk.
Two automated NLI models have scored it — one using the full chunk as
context and one using only the cited evidence span as context.
Your job: read the evidence and give the FINAL human-quality verdict.

**Judging criteria:**
- "entailed" — the text explicitly states or strongly implies the triplet
- "contradicted" — the text states the opposite of this triplet
- "unsupported" — the triplet may be true, but this text does not
  establish it (not mentioned, or requires an inference the text does
  not make)

**How to use the chunk and the span together — they answer different
questions, do not just pick one to "trust":**
- The full chunk resolves REFERENCE and SCOPE — e.g. what a subject
  named only in a heading refers to, or what "it" / "the product"
  refers to earlier in the passage. Use the chunk to confirm the
  subject/object identity is correct.
- The evidence span must still ground the CORE FACTUAL CLAIM — the
  relation itself. If the span doesn't support the claim but the
  chunk elsewhere does, use the chunk. If neither supports the claim,
  verdict is "unsupported" — the span alone lacking full context is
  NOT by itself grounds for "unsupported" or "contradicted".

**Negation, hedging, and disjunction:**
- If the triplet represents a negated claim (the text explicitly
  denies it, e.g. "does not contain X"), judge whether the DENIAL is
  entailed — not whether the positive claim is. A correctly-extracted
  negation should be judged "entailed" if the text does deny it.
- If the triplet is marked hedged (conditional, modal, or one branch
  of a disjunction — "may contain", "either X or Y"), judge whether
  the text's hedge/conditional is faithfully represented. A hedged
  triplet judged against unhedged certainty should not be penalized
  for being uncertain — that uncertainty is the correct extraction.

**Event-linked triplets:**
- If subject or object is a synthetic event ID (e.g. "ev_ghee_adult_0001"
  rather than a name appearing in the text), treat it as referring to
  the real-world event/claim instance described in the chunk. Judge
  whether the specific attribute (detection method, regulation,
  location, etc.) is entailed for that event — do not penalize the ID
  itself for not appearing verbatim in the text.

Return ONLY a JSON object:
{
  "verdict": "entailed",
  "confidence": 0.92,
  "reason_category": "explicit_statement",
  "rationale": "one sentence explaining your reasoning"
}

reason_category must be one of:
"explicit_statement" | "strong_implication" | "reference_resolved_via_chunk"
| "requires_inference" | "not_mentioned" | "explicit_denial_in_text"
| "hedge_faithfully_represented" | "text_states_opposite"
"""

LLM_JUDGE_USER = """\
=== FULL CHUNK TEXT ===
{chunk_text}

=== EVIDENCE SPAN (cited substring) ===
{evidence_span}

=== EXTRACTED TRIPLET ===
Subject:      {subject}  ({subject_type})
Predicate:    {predicate}
Object:       {object}  ({object_type})
Polarity:     {polarity}
Event ID:     {event_id}

=== VERBALISED ===
{verbalisation}

=== AUTOMATED NLI SCORES ===
Chunk-level NLI:  {chunk_nli_label}  (confidence: {chunk_nli_conf:.3f})
Span-level NLI:   {span_nli_label}   (confidence: {span_nli_conf:.3f})

Based on the evidence above, is this triplet entailed, contradicted, or unsupported?"""

# ---------------------------------------------------------------------
# FEW-SHOT EXAMPLES
# Insert these as prior turns (or inline in the system prompt, if the
# judge call is single-turn) before the real triplet to be judged.
# ---------------------------------------------------------------------

JUDGE_FEWSHOT = [
    # Example 1 — clear entailment, straightforward
    {
        "chunk_text": "14.1 Non-alcoholic beverages: 14.1.1 includes waters "
                       "and carbonated waters, 14.1.4 includes water-based "
                       "flavoured carbonated and non-carbonated drinks.",
        "evidence_span": "includes waters and carbonated waters (14.1.1)",
        "subject": "Non-alcoholic beverages", "subject_type": "FoodCategory",
        "predicate": "includesSubcategory",
        "object": "waters and carbonated waters", "object_type": "FoodCategory",
        "polarity": "affirmed", "event_id": "",
        "verbalisation": "Non-alcoholic beverages includesSubcategory waters and carbonated waters",
        "answer": {
            "verdict": "entailed",
            "confidence": 0.95,
            "reason_category": "reference_resolved_via_chunk",
            "rationale": "The section heading in the chunk establishes 'Non-alcoholic beverages' as the parent category for 14.1.1, and the span confirms the subcategory relation."
        }
    },
    # Example 2 — n-ary event, judging one linked attribute
    {
        "chunk_text": "In a surveillance sample from Uttar Pradesh, ghee was "
                       "found adulterated with vanaspati, detected via GC-MS "
                       "analysis, in violation of FSSAI clause 7.2.3.",
        "evidence_span": "detected via GC-MS analysis",
        "subject": "ev_ghee_adult_0001", "subject_type": "AdulterationEvent",
        "predicate": "detectedBy",
        "object": "GC-MS analysis", "object_type": "DetectionMethod",
        "polarity": "affirmed", "event_id": "ev_ghee_adult_0001",
        "verbalisation": "The adulteration event described in the chunk was detectedBy GC-MS analysis",
        "answer": {
            "verdict": "entailed",
            "confidence": 0.93,
            "reason_category": "explicit_statement",
            "rationale": "The event ID refers to the ghee/vanaspati adulteration event in the chunk, and the span directly states the detection method."
        }
    },
    # Example 3 — negation, correctly judged as entailed denial
    {
        "chunk_text": "The tested sample did not contain any traces of Sudan dye.",
        "evidence_span": "did not contain any traces of Sudan dye",
        "subject": "tested sample", "subject_type": "Food",
        "predicate": "hasAdulterant",
        "object": "Sudan dye", "object_type": "Contaminant",
        "polarity": "negated", "event_id": "",
        "verbalisation": "tested sample does NOT hasAdulterant Sudan dye",
        "answer": {
            "verdict": "entailed",
            "confidence": 0.96,
            "reason_category": "explicit_denial_in_text",
            "rationale": "The text explicitly denies the presence of Sudan dye, and the triplet is correctly extracted as a negated claim."
        }
    },
    # Example 4 — hedge, correctly judged as entailed hedge (not penalized for uncertainty)
    {
        "chunk_text": "The product may contain traces of peanut due to shared processing equipment.",
        "evidence_span": "may contain traces of peanut",
        "subject": "product", "subject_type": "Food",
        "predicate": "hasIngredient",
        "object": "peanut", "object_type": "Ingredient",
        "polarity": "hedged", "event_id": "",
        "verbalisation": "product may hasIngredient peanut",
        "answer": {
            "verdict": "entailed",
            "confidence": 0.88,
            "reason_category": "hedge_faithfully_represented",
            "rationale": "The text's conditional language is faithfully preserved as a hedge in the triplet, not overstated as certain."
        }
    },
    # Example 5 — genuinely unsupported (not mentioned), not a contradiction
    {
        "chunk_text": "Ghee shall contain milk fat as its primary ingredient. "
                       "It shall be free from added colouring matter.",
        "evidence_span": "free from added colouring matter",
        "subject": "Ghee", "subject_type": "Food",
        "predicate": "regulatedUnder",
        "object": "FSSAI clause 7.2.3", "object_type": "Regulation",
        "polarity": "affirmed", "event_id": "",
        "verbalisation": "Ghee regulatedUnder FSSAI clause 7.2.3",
        "answer": {
            "verdict": "unsupported",
            "confidence": 0.9,
            "reason_category": "not_mentioned",
            "rationale": "The chunk never cites a specific clause number, so this triplet cannot be grounded here even though it may be true elsewhere."
        }
    },
    # Example 6 — genuine contradiction
    {
        "chunk_text": "Analysis confirmed the sample was free of vanaspati "
                       "and met all compositional standards for pure ghee.",
        "evidence_span": "confirmed the sample was free of vanaspati",
        "subject": "sample", "subject_type": "Food",
        "predicate": "hasAdulterant",
        "object": "Vanaspati", "object_type": "Contaminant",
        "polarity": "affirmed", "event_id": "",
        "verbalisation": "sample hasAdulterant Vanaspati",
        "answer": {
            "verdict": "contradicted",
            "confidence": 0.97,
            "reason_category": "text_states_opposite",
            "rationale": "The text explicitly states the sample was free of vanaspati, the opposite of what this affirmed triplet claims."
        }
    },
]


# ---- Data loading ----

def _load_triplets_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return items


def _load_chunks(csv_path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not csv_path.exists():
        return lookup
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lookup[row["snippet_id"]] = row.get("evidence_text", "")
    return lookup


# ---- LLM judge call (async, batched) ----

async def _llm_judge_batch(
    base_url: str,
    api_key: str,
    model: str,
    items: list[dict[str, Any]],
    concurrency: int = 8,
    checkpoint_path: str = "",
    skip_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Call Qwen for a batch of triplets. Returns list of verdict dicts.

    *checkpoint_path* enables incremental saves every ~200 completions.
    *skip_indices* is a set of already-judged positions to skip (resume).
    """
    import asyncio
    import httpx

    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any] | None] = [None] * len(items)
    skip = skip_indices or set()
    done_count = 0
    lock = asyncio.Lock()
    verdict_counts: dict[str, int] = {}

    # Pre-fill results from skip_indices (loaded from checkpoint)
    if skip and checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            for line in ckpt.open():
                try:
                    obj = json.loads(line)
                    idx = obj.get("idx")
                    if idx is not None and idx < len(results):
                        results[idx] = obj.get("result")
                except json.JSONDecodeError:
                    pass
        done_count = sum(1 for r in results if r is not None)

    async def _one(idx: int) -> None:
        nonlocal done_count
        if idx in skip:
            return  # already done, result pre-loaded
        item = items[idx]
        user_msg = LLM_JUDGE_USER.format(
            chunk_text=item.get("chunk_text", "")[:3000],
            evidence_span=item.get("evidence_span", ""),
            subject=item.get("subject", ""),
            subject_type=item.get("subject_type", ""),
            predicate=item.get("predicate", ""),
            object=item.get("object", ""),
            object_type=item.get("object_type", ""),
            verbalisation=item.get("verbalisation", ""),
            chunk_nli_label=item.get("chunk_nli_label", "?"),
            chunk_nli_conf=item.get("chunk_nli_conf", 0.0),
            span_nli_label=item.get("span_nli_label", "?"),
            span_nli_conf=item.get("span_nli_conf", 0.0),
            polarity=item.get("polarity", "affirmed"),
            event_id=item.get("event_id", "") or "(none)",
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": LLM_JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with sem:
            try:
                resp = await client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=httpx.Timeout(120),
                )
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                content = content.strip()
                if content.startswith("```"):
                    content = content.replace("```json", "").replace(
                        "```", ""
                    ).strip()
                verdict_obj = json.loads(content)
                results[idx] = {
                    "llm_verdict": verdict_obj.get("verdict", "uncertain"),
                    "llm_confidence": verdict_obj.get("confidence", 0.0),
                    "llm_rationale": verdict_obj.get("rationale", ""),
                }
            except Exception as exc:
                results[idx] = {
                    "llm_verdict": "llm_error",
                    "llm_confidence": 0.0,
                    "llm_rationale": str(exc)[:200],
                }

            # Progress bar
            async with lock:
                done_count += 1
                v = results[idx]["llm_verdict"]
                verdict_counts[v] = verdict_counts.get(v, 0) + 1
                if done_count % 50 == 0 or done_count == len(items):
                    print(
                        f"  [{done_count:>5}/{len(items)}] "
                        f"{', '.join(f'{k}:{c}' for k,c in sorted(verdict_counts.items()))}"
                    )
                    # Periodic checkpoint flush
                    if checkpoint_path and results[idx] is not None:
                        _append_checkpoint(checkpoint_path, idx, results[idx])

    async with httpx.AsyncClient() as client:
        tasks = [_one(i) for i in range(len(items))]
        await asyncio.gather(*tasks)

    return [r or {"llm_verdict": "llm_error", "llm_confidence": 0.0,
                  "llm_rationale": "unknown"} for r in results]


# ---- LLM canonicalisation batch ----

async def _llm_canonicalise_batch(
    base_url: str,
    api_key: str,
    model: str,
    clusters: dict[str, list[str]],
    batch_size: int = 15,
) -> dict[str, str]:
    """Send clusters to Qwen in batches, merge results into a dict."""
    import httpx

    merges: dict[str, str] = {}
    cluster_ids = list(clusters.keys())
    all_names = sorted(set(
        n for names in clusters.values() for n in names
    ))

    for batch_start in range(0, len(cluster_ids), batch_size):
        batch_ids = cluster_ids[batch_start: batch_start + batch_size]
        batch_clusters = {cid: clusters[cid] for cid in batch_ids}
        parts = ["=== NAME CLUSTERS ===\n"]
        for cid in batch_ids:
            parts.append(
                f"Group {cid}: [{', '.join(clusters[cid])}]"
            )
        parts.append(
            "\nFor each group, choose ONE umbrella name."
        )
        user_msg = "\n".join(parts)

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

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(120),
            )
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            content = content.strip()
            if content.startswith("```"):
                content = content.replace("```json", "").replace(
                    "```", ""
                ).strip()

            try:
                result = json.loads(content)
                batch_merges = result.get("merges", {})
                for cid, chosen in batch_merges.items():
                    if chosen and cid in clusters:
                        merges[cid] = chosen
            except json.JSONDecodeError:
                for cid in batch_ids:
                    merges[cid] = clusters[cid][0]

        print(
            f"  Batch {batch_start}-{batch_start+len(batch_ids)-1} "
            f"({len(batch_ids)} clusters) done"
        )

    return merges


# ---- Agreement logic ----

def _resolve_agreement(
    chunk_nli_label: str,
    span_nli_label: str,
    llm_verdict: str,
) -> tuple[str, str]:
    """Determine agreement status and final verdict.

    LLM judge takes precedence.  Returns (agreement_status, final_verdict).
    """
    # Map NLI labels to verdicts
    def _nli_to_verdict(label: str) -> str:
        return {
            "entailment": "entailed",
            "neutral": "uncertain",
            "contradiction": "not_entailed",
        }.get(label, "uncertain")

    chunk_v = _nli_to_verdict(chunk_nli_label)
    span_v = _nli_to_verdict(span_nli_label)
    llm_v = llm_verdict if llm_verdict in (
        "entailed", "partially_entailed", "not_entailed"
    ) else "uncertain"

    # Check agreement between chunk-NLI and span-NLI
    nli_agree = chunk_v == span_v
    # Check if either NLI agrees with LLM
    nli_chunk_llm = chunk_v == llm_v
    nli_span_llm = span_v == llm_v

    if nli_agree and nli_chunk_llm:
        status = "agreed_all"
    elif nli_chunk_llm or nli_span_llm:
        status = "agreed_partial"
    else:
        status = "contested"

    return status, llm_v  # LLM takes precedence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    triplets_dir = Path(args.triplets_dir)
    chunks_csv = Path(args.chunks_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load triplets ----
    trips_path = triplets_dir / "triplets.jsonl"
    if not trips_path.exists():
        subdirs = sorted(triplets_dir.glob("**/triplets.jsonl"))
        if subdirs:
            trips_path = subdirs[0]
            print(f"Auto-discovered: {trips_path}")
    if not trips_path.exists():
        print(f"error: {trips_path} not found", file=sys.stderr)
        return 1

    entries = _load_triplets_jsonl(trips_path)
    all_triplets: list[dict[str, Any]] = []
    for e in entries:
        sid = e.get("snippet_id", "")
        for t in e.get("triplets", []):
            t["snippet_id"] = sid
            all_triplets.append(t)
    print(f"Loaded {len(entries)} chunks, {len(all_triplets)} triplets")

    if args.limit and args.limit < len(all_triplets):
        all_triplets = all_triplets[:args.limit]
        print(f"  Limited to {args.limit} triplets")

    # ---- Pre-apply canonical map (if provided) ----
    if args.canonical_map and args.canonical_map.exists():
        load_cmap, apply_cmap = _get_normalise_tools()
        merges = load_cmap(args.canonical_map)
        if merges:
            preds = apply_cmap(
                [t.get("predicate", "") for t in all_triplets], merges,
            )
            stypes = apply_cmap(
                [t.get("subject_type", "") for t in all_triplets], merges,
            )
            otypes = apply_cmap(
                [t.get("object_type", "") for t in all_triplets], merges,
            )
            for i, t in enumerate(all_triplets):
                t["predicate"] = preds[i]
                t["subject_type"] = stypes[i]
                t["object_type"] = otypes[i]
            print(f"  Applied {len(merges)} canonical mappings to triplets")

    # ---- Load chunks ----
    chunk_text = _load_chunks(chunks_csv)
    print(f"  {len(chunk_text)} chunks loaded")

    # ---- Load models ----
    NLIModel = _get_nli_model()
    verbalise_fn = _get_verbalise()

    print(f"\nLoading NLI model: {args.nli_model}")
    nli = NLIModel(args.nli_model, device=args.device, max_length=512)

    # ---- Build hypotheses + premises ----
    premises_chunk: list[str] = []
    premises_span: list[str] = []
    hypotheses: list[str] = []
    t_meta: list[dict[str, Any]] = []

    for t in all_triplets:
        sid = t.get("snippet_id", "")
        subj = t.get("subject", "")
        stype = t.get("subject_type", "")
        pred = t.get("predicate", "")
        obj = t.get("object", "")
        otype = t.get("object_type", "")
        span = t.get("evidence_span", "")

        hyp = verbalise_fn(subj, stype, pred, obj, otype)
        t_neg = _is_negated(span)
        if t_neg:
            hyp = f"It is NOT the case that {hyp}"

        premises_chunk.append(chunk_text.get(sid, ""))
        premises_span.append(span)
        hypotheses.append(hyp)
        t_meta.append({
            "snippet_id": sid, "subject": subj, "predicate": pred,
            "object": obj, "verbalisation": hyp,
            "evidence_span": span,
            "chunk_text": chunk_text.get(sid, ""),
            "subject_type": stype, "object_type": otype,
        })

    # ---- Chunk-level NLI ----
    print(f"\nChunk-level NLI on {len(premises_chunk)} pairs...")
    t0 = time.time()
    chunk_results = nli.predict(
        premises_chunk, hypotheses, batch_size=args.batch_size,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- Span-level NLI ----
    print(f"\nSpan-level NLI on {len(premises_span)} pairs...")
    t0 = time.time()
    span_results = nli.predict(
        premises_span, hypotheses, batch_size=args.batch_size,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- Build LLM items ----
    llm_items: list[dict[str, Any]] = []
    for i, t in enumerate(all_triplets):
        llm_items.append({
            **t_meta[i],
            "chunk_nli_label": chunk_results[i]["label"],
            "chunk_nli_conf": chunk_results[i]["confidence"],
            "span_nli_label": span_results[i]["label"],
            "span_nli_conf": span_results[i]["confidence"],
        })

    # ---- LLM judge (async batch with checkpointing) ----
    ckpt_path = str(output_dir / "llm_judge_checkpoint.jsonl")
    skip_idx: set[int] = set()
    if args.resume:
        skip_idx = _load_checkpoint_indices(ckpt_path)
        if skip_idx:
            print(f"  Resuming: skipping {len(skip_idx)} already-judged triplets")

    print(f"\nLLM judge on {len(llm_items)} triplets "
          f"(concurrency={args.concurrency})...")
    t0 = time.time()
    llm_results = asyncio.run(
        _llm_judge_batch(
            args.base_url, args.api_key, args.model,
            llm_items, concurrency=args.concurrency,
            checkpoint_path=ckpt_path,
            skip_indices=skip_idx,
        )
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- Attach all results + agreement ----
    for i, t in enumerate(all_triplets):
        t["chunk_nli_label"] = chunk_results[i]["label"]
        t["chunk_nli_conf"] = chunk_results[i]["confidence"]
        t["chunk_nli_scores"] = chunk_results[i]["scores"]
        t["span_nli_label"] = span_results[i]["label"]
        t["span_nli_conf"] = span_results[i]["confidence"]
        t["span_nli_scores"] = span_results[i]["scores"]
        t["verbalisation"] = t_meta[i]["verbalisation"]
        t.update(llm_results[i])

        status, final = _resolve_agreement(
            t["chunk_nli_label"], t["span_nli_label"],
            t["llm_verdict"],
        )
        t["agreement_status"] = status
        t["final_verdict"] = final

    # ---- Summary ----
    status_counts = Counter(t["agreement_status"] for t in all_triplets)
    verdict_counts = Counter(t["final_verdict"] for t in all_triplets)
    contested = sum(1 for t in all_triplets
                    if t["agreement_status"] == "contested")

    print(f"\n{'='*50}")
    print(f"Agreement: {dict(status_counts)}")
    print(f"Contested (need review): {contested}/{len(all_triplets)}")
    print(f"Final verdicts: {dict(verdict_counts)}")

    # ---- Save results ----
    # Main CSV
    out_csv = output_dir / "nli_llm_judge_results.csv"
    fieldnames = [
        "snippet_id", "subject", "subject_type", "predicate",
        "object", "object_type", "confidence", "evidence_span",
        "verbalisation",
        "chunk_nli_label", "chunk_nli_conf",
        "span_nli_label", "span_nli_conf",
        "llm_verdict", "llm_confidence", "llm_rationale",
        "agreement_status", "final_verdict",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_triplets)
    print(f"\nFull results → {out_csv}")

    # Contested-only CSV (for human review)
    contested_csv = output_dir / "nli_llm_judge_contested.csv"
    contested_rows = [t for t in all_triplets
                      if t["agreement_status"] == "contested"]
    if contested_rows:
        with contested_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(contested_rows)
        print(f"Contested only → {contested_csv} ({len(contested_rows)} rows)")

    # ---- Entailed-only CSV ----
    entailed_rows = [t for t in all_triplets
                     if t["final_verdict"] in ("entailed", "partially_entailed")
                     and t["agreement_status"] != "contested"]
    if entailed_rows:
        entailed_csv = output_dir / "nli_llm_judge_entailed.csv"
        with entailed_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(entailed_rows)
        print(f"Entailed only → {entailed_csv} ({len(entailed_rows)} rows)")

    # ---- Canonicalisation (optional) ----
    if args.canonicalise:
        print(f"\n{'='*50}")
        print("Canonicalisation mode active")
        normalise_fn, save_canonical = _get_normalise()

        # 1. Collect predicates and entity types from entailed set
        canonical_source = entailed_rows if entailed_rows else all_triplets
        predicates = [t.get("predicate", "") for t in canonical_source]
        entity_types = [t.get("subject_type", "") for t in canonical_source] + \
                       [t.get("object_type", "") for t in canonical_source]

        # 2. Normalise
        rel_norm = normalise_fn(predicates)
        type_norm = normalise_fn(entity_types)
        print(f"Relations: {len(rel_norm['frequency'])} canonical forms, "
              f"{len(rel_norm['near_duplicates'])} near-duplicate groups")
        print(f"Entity types: {len(type_norm['frequency'])} canonical forms, "
              f"{len(type_norm['near_duplicates'])} near-duplicate groups")

        # 3. Build clusters from Levenshtein near-dupes
        clusters: dict[str, list[str]] = {}
        for i, g in enumerate(rel_norm["near_duplicates"]):
            clusters[f"rel_{i}"] = sorted(g)
        for i, g in enumerate(type_norm["near_duplicates"]):
            clusters[f"type_{i}"] = sorted(g)

        if not clusters:
            print("No clusters to resolve — all names are distinct.")
        else:
            print(f"\nResolving {len(clusters)} clusters via LLM...")
            t0 = time.time()
            merges = asyncio.run(
                _llm_canonicalise_batch(
                    args.base_url, args.api_key, args.model,
                    clusters, batch_size=15,
                )
            )
            print(f"  Done in {time.time()-t0:.1f}s")

            # 4. Build canonical_map.json
            canonical_map: dict[str, str] = {}
            for cid, chosen in merges.items():
                variants = clusters.get(cid, [])
                for v in variants:
                    if v != chosen:
                        canonical_map[v] = chosen
                canonical_map[chosen] = chosen

            cmap_path = output_dir / "canonical_map.json"
            cmap_data = {
                "merges": canonical_map,
                "generated_by": "nli_llm_judge.py --canonicalise",
                "cluster_count": len(clusters),
                "resolved_count": len(merges),
                "source": "entailed + agreed triplets only",
            }
            cmap_path.write_text(
                json.dumps(cmap_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Canonical map → {cmap_path} "
                  f"({len(canonical_map)} total mappings)")

    # Summary JSON
    summary = {
        "total": len(all_triplets),
        "agreement": dict(status_counts),
        "verdicts": dict(verdict_counts),
        "contested_count": contested,
    }
    summary_path = output_dir / "nli_llm_judge_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Summary → {summary_path}")

    return 0


# ---- Helpers ----

_NEGATION_WORDS = frozenset({
    "not", "no", "never", "shall not", "free from", "exempt",
    "without", "excluded", "absence", "absent",
})


def _is_negated(span: str) -> bool:
    lower = span.lower()
    return any(w in lower for w in _NEGATION_WORDS)


def _append_checkpoint(path: str, idx: int, result: dict[str, Any]) -> None:
    """Append a single result to the checkpoint JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"idx": idx, "result": result},
                           ensure_ascii=False) + "\n")


def _load_checkpoint_indices(path: str) -> set[int]:
    """Return set of already-processed indices from a checkpoint file."""
    ckpt = Path(path)
    if not ckpt.exists():
        return set()
    indices: set[int] = set()
    for line in ckpt.open():
        try:
            indices.add(json.loads(line).get("idx", -1))
        except json.JSONDecodeError:
            pass
    return indices


# ---- CLI ----

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NLI + LLM-judge entailment validation"
    )
    p.add_argument("--triplets-dir", type=Path, required=True,
                   help="Directory with extracted triplets.jsonl")
    p.add_argument("--chunks-csv", type=Path, required=True)
    p.add_argument("--nli-model", type=str, required=True,
                   help="Path to fine-tuned NLI checkpoint")
    p.add_argument("--base-url", type=str, required=True,
                   help="vLLM endpoint for Qwen judge")
    p.add_argument("--model", type=str, required=True,
                   help="LLM model name for judge")
    p.add_argument("--api-key", type=str, default="",
                   help="API key for LLM endpoint")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for NLI inference")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Concurrent LLM judge calls")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only first N triplets (0=all)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint (skip already-judged triplets)")
    p.add_argument("--canonicalise", action="store_true",
                   help="Run LLM canonicalisation on entailed+agreed triplets "
                        "to produce canonical_map.json")
    p.add_argument("--canonical-map", type=Path, default=None,
                   help="Pre-apply canonical_map.json to resolve predicate "
                        "and entity type names before judging")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())
