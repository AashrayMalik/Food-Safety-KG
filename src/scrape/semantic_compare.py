#!/usr/bin/env python3
"""
Semantic (embedding) comparison of NABL accredited test scopes against
news-reported adulterants, replacing the substring matcher in
compare_adulterants.py.

For each news term, embed it and every scope parameter with a SentenceTransformer
model, take cosine similarity, and mark the term COVERED if any lab's parameter
clears a similarity threshold. Every call is auditable via the top-k matched
parameters and their scores.

The curated `adulterant_gap_analysis.csv` stays AUTHORITATIVE: this script only
validates against it and emits proposals for review, it never overwrites it.

Writes:
  semantic_embeddings/                 cached embedding matrices + string lists
  semantic_adulterant_coverage.csv     term -> top-k params + scores + verdict
  semantic_validation.csv              agreement vs curated ground truth (sweep)
  semantic_gap_proposals.csv           proposed GAP<->COVERED flips for review
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

from compare_adulterants import (
    NEWS_PATH,
    QWEN_PATH,
    SCOPE_FILES,
    normalize,
    dedupe,
    load_news_terms,
    load_qwen_terms,
    load_scope,
)

MODEL_NAME = "all-MiniLM-L6-v2"
GAP_CSV = "adulterant_gap_analysis.csv"
EMBED_DIR = "semantic_embeddings"
BATCH = 256
TOP_K = 3
THRESHOLDS = [round(x, 2) for x in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)]


def load_model():
    return SentenceTransformer(MODEL_NAME)


def embed(texts, model, name, force=False):
    """Embed `texts`; cache to EMBED_DIR/{name}.npy + .json. Returns np.ndarray."""
    os.makedirs(EMBED_DIR, exist_ok=True)
    npy = os.path.join(EMBED_DIR, f"{name}.npy")
    txt = os.path.join(EMBED_DIR, f"{name}.json")
    if not force and os.path.exists(npy) and os.path.exists(txt):
        with open(txt) as f:
            cached = json.load(f)
        if cached == texts:
            return np.load(npy)
    print(f"  embedding {len(texts)} texts ({name}) ...")
    embs = model.encode(
        texts,
        batch_size=BATCH,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(npy, embs)
    with open(txt, "w") as f:
        json.dump(texts, f, ensure_ascii=False)
    return embs


def top_k_indices(sim_row, k):
    """Indices of the k highest values in a 1-D similarity row."""
    k = min(k, sim_row.size)
    if k == 0:
        return np.array([], dtype=int)
    return np.argpartition(sim_row, -k)[-k:]


def resolve_to_news(term, news_by_key):
    """Map a curated term to a news info dict (dedupe key, then substring)."""
    dk = dedupe(normalize(term))
    if dk in news_by_key:
        return news_by_key[dk]
    for key, info in news_by_key.items():
        if dk in key or key in dk:
            return info
    return None


def main():
    """Embed news/Qwen terms and scope params, then audit coverage by similarity."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default=NEWS_PATH)
    ap.add_argument("--qwen", default=QWEN_PATH)
    ap.add_argument("--scope", nargs="*", default=SCOPE_FILES)
    ap.add_argument("--gap-csv", default=GAP_CSV)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the auto-tuned threshold")
    ap.add_argument("--force", action="store_true", help="re-embed (ignore cache)")
    ap.add_argument("--coverage-out", default="semantic_adulterant_coverage.csv")
    ap.add_argument("--validation-out", default="semantic_validation.csv")
    ap.add_argument("--proposals-out", default="semantic_gap_proposals.csv")
    ap.add_argument("--gap-report", default="semantic_gap_validation.csv",
                    help="full semantic audit of every specific-substance GAP term")
    args = ap.parse_args()

    print("loading vocabularies...")
    news = load_news_terms(args.news)
    qwen = load_qwen_terms(args.qwen)
    print(f"  news terms: {len(news)} | qwen terms: {len(qwen)}")

    print("loading scope...")
    scope, total_labs, materials = load_scope(args.scope)
    print(f"  scope parameters: {len(scope)} | labs: {total_labs}")

    model = load_model()

    # --- order-preserving text lists ---------------------------------------
    news_items = list(news.values())
    news_key_to_idx = {info["dedup"]: i for i, info in enumerate(news_items)}
    news_texts = [info["norm"] for info in news_items]

    qwen_items = list(qwen.values())
    qwen_texts = [info["norm"] for info in qwen_items]

    param_texts = [p["norm"] for p in scope]

    print("embedding...")
    news_emb = embed(news_texts, model, "news_terms", force=args.force)
    qwen_emb = embed(qwen_texts, model, "qwen_terms", force=args.force)
    param_emb = embed(param_texts, model, "scope_params", force=args.force)

    print("computing similarity...")
    news_sim = news_emb @ param_emb.T            # (N_news, N_params)
    qwen_sim = qwen_emb @ param_emb.T if qwen_emb.size else None

    # --- validation: tune threshold vs curated ground truth ------------------
    curated = []
    for r in csv.DictReader(open(args.gap_csv)):
        if (r.get("category") or "").strip() != "specific-substance":
            continue
        term = r["adulterant_term_from_news"].strip()
        status = (r.get("lab_scope_status") or "").strip()
        curated.append({"term": term, "status": status,
                        "mentions": int(r["news_mention_count"])})

    # map curated -> news row idx (for best-score lookup)
    rows = []  # dict: term, status, mentions, news_idx, matched_param info
    unmatched = []
    for c in curated:
        info = resolve_to_news(c["term"], news)
        if info is None:
            unmatched.append(c["term"])
            continue
        rows.append({**c, "news_idx": news_key_to_idx[info["dedup"]],
                     "news_key": info["dedup"]})

    best_news = news_sim.max(axis=1)             # per news row, top-1 score

    print(f"  curated specific-substance: {len(curated)} "
          f"(unmatched to news: {len(unmatched)})")

    sweep = []
    best_f1, best_thr = -1.0, None
    for t in THRESHOLDS:
        tp = fp = tn = fn = 0
        for r in rows:
            sem = "COVERED" if best_news[r["news_idx"]] >= t else "GAP"
            if r["status"] == "COVERED" and sem == "COVERED":
                tp += 1
            elif r["status"] == "GAP" and sem == "COVERED":
                fp += 1
            elif r["status"] == "COVERED" and sem == "GAP":
                fn += 1
            else:
                tn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        agr = (tp + tn) / len(rows) if rows else 0.0
        sweep.append({"threshold": t, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                      "precision": round(prec, 3), "recall": round(rec, 3),
                      "f1": round(f1, 3), "agreement": round(agr, 3)})
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    final_thr = args.threshold if args.threshold is not None else best_thr
    print("\nthreshold sweep (curated specific-substance):")
    print("  thr    prec   rec    f1     agr    tp  fp  tn  fn")
    for s in sweep:
        mark = " <=" if s["threshold"] == final_thr else "   "
        print(f"  {s['threshold']:.2f}{mark}  {s['precision']:.3f}  {s['recall']:.3f}  "
              f"{s['f1']:.3f}  {s['agreement']:.3f}  {s['tp']:3d} {s['fp']:3d} "
              f"{s['tn']:3d} {s['fn']:3d}")
    print(f"\nchosen threshold: {final_thr} (f1={best_f1:.3f})")

    # --- coverage output -----------------------------------------------------
    def coverage_rows(items, sim):
        out = []
        for i, info in enumerate(items):
            srow = sim[i]
            idxs = top_k_indices(srow, TOP_K)
            idxs = sorted(idxs, key=lambda j: -srow[j])
            top = [(scope[j]["param"], round(float(srow[j]), 4)) for j in idxs]
            above = [j for j in range(sim.shape[1]) if srow[j] >= final_thr]
            labs = set()
            for j in above:
                labs |= scope[j]["tcs"]
            best = top[0][1] if top else 0.0
            verdict = "COVERED" if best >= final_thr else "GAP"
            row = {
                "term": info["term"], "category": info["category"],
                "object_type": info["object_type"],
                "top1_param": top[0][0] if len(top) > 0 else "",
                "top1_score": top[0][1] if len(top) > 0 else "",
                "top2_param": top[1][0] if len(top) > 1 else "",
                "top2_score": top[1][1] if len(top) > 1 else "",
                "top3_param": top[2][0] if len(top) > 2 else "",
                "top3_score": top[2][1] if len(top) > 2 else "",
                "n_params_above_threshold": len(above),
                "n_labs": len(labs),
                "verdict": verdict,
            }
            out.append(row)
        return out

    cov_fields = ["term", "source", "category", "object_type",
                  "top1_param", "top1_score", "top2_param", "top2_score",
                  "top3_param", "top3_score",
                  "n_params_above_threshold", "n_labs", "verdict"]
    with open(args.coverage_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cov_fields)
        w.writeheader()
        for r in coverage_rows(news_items, news_sim):
            w.writerow({**r, "source": "news"})
        if qwen_sim is not None and qwen_sim.size:
            for r in coverage_rows(qwen_items, qwen_sim):
                w.writerow({**r, "source": "qwen"})
    print(f"wrote {args.coverage_out}")

    # --- validation output ---------------------------------------------------
    with open(args.validation_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "tp", "fp", "tn", "fn",
                    "precision", "recall", "f1", "agreement"])
        for s in sweep:
            w.writerow([s["threshold"], s["tp"], s["fp"], s["tn"], s["fn"],
                        s["precision"], s["recall"], s["f1"], s["agreement"]])
    print(f"wrote {args.validation_out}")

    # --- full gap audit (every specific-substance GAP term) ------------------
    gap_fields = ["term", "curated_status", "mentions",
                  "top1_param", "top1_score", "top2_param", "top2_score",
                  "top3_param", "top3_score", "n_labs", "semantic_verdict"]
    with open(args.gap_report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gap_fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: -x["mentions"]):
            if r["status"] != "GAP":
                continue
            srow = news_sim[r["news_idx"]]
            idxs = sorted(top_k_indices(srow, TOP_K), key=lambda j: -srow[j])
            top = [(scope[j]["param"], round(float(srow[j]), 4)) for j in idxs]
            above = [j for j in range(srow.size) if srow[j] >= final_thr]
            labs = set()
            for j in above:
                labs |= scope[j]["tcs"]
            best = top[0][1] if top else 0.0
            w.writerow({
                "term": r["term"], "curated_status": r["status"],
                "mentions": r["mentions"],
                "top1_param": top[0][0] if top else "",
                "top1_score": top[0][1] if top else "",
                "top2_param": top[1][0] if len(top) > 1 else "",
                "top2_score": top[1][1] if len(top) > 1 else "",
                "top3_param": top[2][0] if len(top) > 2 else "",
                "top3_score": top[2][1] if len(top) > 2 else "",
                "n_labs": len(labs),
                "semantic_verdict": "PROPOSED-COVERED" if best >= final_thr else "AGREE-GAP",
            })
    print(f"wrote {args.gap_report}")

    # --- proposals (disagreements at final threshold) ------------------------
    disagreements = []
    for r in rows:
        sem = "COVERED" if best_news[r["news_idx"]] >= final_thr else "GAP"
        if sem == r["status"]:
            continue
        srow = news_sim[r["news_idx"]]
        idxs = sorted(top_k_indices(srow, TOP_K), key=lambda j: -srow[j])
        top = [(scope[j]["param"], round(float(srow[j]), 4)) for j in idxs]
        disagreements.append({
            "term": r["term"],
            "curated_status": r["status"],
            "semantic_status": sem,
            "mentions": r["mentions"],
            "top1_param": top[0][0] if top else "",
            "top1_score": top[0][1] if top else "",
            "top2_param": top[1][0] if len(top) > 1 else "",
            "top2_score": top[1][1] if len(top) > 1 else "",
            "top3_param": top[2][0] if len(top) > 2 else "",
            "top3_score": top[2][1] if len(top) > 2 else "",
        })
    disagreements.sort(key=lambda d: (d["curated_status"], -d["mentions"]))
    prop_fields = ["term", "curated_status", "semantic_status", "mentions",
                   "top1_param", "top1_score", "top2_param", "top2_score",
                   "top3_param", "top3_score"]
    with open(args.proposals_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=prop_fields)
        w.writeheader()
        w.writerows(disagreements)
    print(f"wrote {args.proposals_out} ({len(disagreements)} disagreements)")

    # summary print of disagreements
    flip = defaultdict(list)
    for d in disagreements:
        flip[d["curated_status"]].append(d)
    print(f"\ndisagreements: GAP->COVERED {len(flip['GAP'])} | "
          f"COVERED->GAP {len(flip['COVERED'])}")
    for d in disagreements[:40]:
        print(f"  [{d['curated_status']}->{d['semantic_status']}] "
              f"{d['term']!r:40s} score={d['top1_score']} -> {d['top1_param']!r}")


if __name__ == "__main__":
    main()
