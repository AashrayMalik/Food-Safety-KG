#!/usr/bin/env python3
"""Enforcement-action enrichment for the news triplets ("action taken on the FBO").

The news extraction emits ``fflo:RegulatoryAction`` nodes as free-text labels
("fine of ₹25 lakh", "suspension of FSSAI licence", "sealing of dairy unit")
linked to foods via ``fflo:targets``, but with no action type, almost no edge to
the Food Business Operator, and the penalty amount buried in the label. This
script adds those three facts as new triplets, using the v9 relations in
``schema_config.json``:

    RegulatoryAction --fflo:hasActionType-->    fflo:RegulatoryActionType subtype
    RegulatoryAction --fflo:takenAgainst-->     fflo:FoodBusinessOperator
    RegulatoryAction --fflo:hasPenaltyAmount--> xsd:decimal (INR)

The source news CSV is read-only; output goes to a separate enrichment CSV in
the same 14-column layout (plus provenance columns), which ``merge_sources.py``
picks up via ``--extra``.

Stages
------
``build`` (deterministic, runs anywhere)
    1. Type each action label with ordered regex rules (vocabulary grounded in
       the FSSAI constrained run, the unconstrained run and the news labels).
    2. Link each action to FBOs in the same snippet through the existing graph:
       direct edges, IncidentFinding --wasAttributedTo--> FBO,
       AdulterationAct --wasAssociatedWith--> FBO, FBO name inside the label,
       or (weakest) the snippet's only FBO.
    3. Parse penalty amounts (₹ / Rs, lakh / crore) for MonetaryPenalty actions.

``type-llm`` (needs the Qwen endpoint)
    Labels no rule matched are classified by Qwen into the closed vocabulary.

``judge`` (needs the Qwen endpoint)
    Every enrichment row goes through the LLM judge (schema_valid branch of
    ``validation/LLM_judge/judge_prompts.py``) — judge only, no NLI. Rows
    judged ``not_entailed`` are dropped from the ``*_validated.csv`` output.

Resume
------
Both LLM stages append every result to a checkpoint next to the CSV
(``news_action_enrichment.<stage>.ckpt.jsonl``) as it arrives, and rewrite the
CSV atomically every 100 calls. After a dropped connection, a killed job or a
SLURM time limit, re-run the same command with ``--resume``: finished calls
are reused and only the rest are sent. Calls that failed (endpoint down,
unparseable reply) are never stored as verdicts; the row stays pending and is
retried on resume. Without ``--resume`` an existing checkpoint stops the run;
``--fresh`` discards it.

Usage::

    python src/extraction/enrich_actions.py build
    python src/extraction/enrich_actions.py type-llm --base-url http://localhost:8030/v1 \\
        --model Qwen/Qwen3.5-27B-FP8 --api-key aashray-fflo-local [--resume]
    python src/extraction/enrich_actions.py judge --base-url http://localhost:8030/v1 \\
        --model Qwen/Qwen3.5-27B-FP8 --api-key aashray-fflo-local [--resume]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEWS_CSV = ROOT / "data" / "FSSAI_docs" / "processed" / "News_Triplets_Oil_Ghee_Milk_20260820.csv"
OUT_CSV = ROOT / "data" / "FSSAI_docs" / "processed" / "news_action_enrichment.csv"
SCHEMA = ROOT / "extraction" / "schema_config.json"
# Article metadata (url, publication date) from the news collection pipeline.
ARTICLES_CSV = ROOT / "food-safety-observatory-with-changes" / "data" / "articles.csv"

ACTION = "fflo:RegulatoryAction"
FBO = "fflo:FoodBusinessOperator"
FINDING_TYPES = {"fflo:IncidentFinding", "fflo:LabConfirmed", "fflo:FieldDetected",
                 "fflo:SurveyAggregated", "fflo:RecallTriggered"}
ACT = "fflo:AdulterationAct"
ISSUER_PREDICATES = {"fflo:issuedBy", "lkif:created_by"}
# Event nodes that get dates (fflo:actionDate / fflo:reportedDate).
EVENT_TYPES = {ACTION, ACT} | FINDING_TYPES

BASE_COLS = ["snippet_id", "source_id", "source_file", "source_type", "chunk_index",
             "subject", "subject_type", "subject_id", "predicate", "object",
             "object_type", "object_id", "confidence", "evidence_span"]
EXTRA_COLS = ["status", "rule", "judge_verdict", "judge_confidence", "judge_rationale"]

# ---------------------------------------------------------------------------
# 1. Action-type rules. Order matters only for readability: a label may get
#    several types ("seizure and sealing of unit"). Grounding per type is
#    documented in documentation/kg_setup.md §1.7.
# ---------------------------------------------------------------------------
TYPE_RULES: list[tuple[str, str]] = [
    ("fflo:LicenceCancellation", r"licen[cs]e\w*\b.{0,40}\b(cancel|revok|revoc)|(cancel|revok|revoc)\w*\b.{0,40}\blicen[cs]|registration\b.{0,20}\bcancel"),
    ("fflo:LicenceSuspension", r"licen[cs]e\w*\b.{0,40}\bsuspen|suspen\w*\b.{0,40}\b(licen[cs]|food registration)|suspension of [\w\s]{0,30}\b(mill|unit|dairy|establishment|firm|plant|factory)s?\b"),
    ("fflo:MonetaryPenalty", r"penalt|\bfine[ds]?\b|\bfining\b|challan|levy|levying|adjudicat|(₹|\brs\.?|\binr)\s?[\d,.]+"),
    ("fflo:Prosecution", r"prosecut|\bfirs?\b|\bfir_|case (was )?(registered|filed|booked|lodged)|registration of (a )?case|case registration|\bbook(ed|ing) (of|against)?|charge[- ]?sheet|\bcourt\b|convict|sentenc|criminal|legal (action|proceeding|steps|measures)|offence registration|charges under|penal action|complaint (filed|lodged)|complaints?\b|punish|offen[cs]e|\bcases?\b|imprison|\bbail\b|proceedings against"),
    ("fflo:Arrest", r"\barrest|\bjail|\bremand|\bnab(bed)?\b|judicial custody|police custody|handover to .{0,20}police"),
    ("fflo:Seizure", r"seiz|confiscat|recover(y|ed) of|detention of|\bdetain|(asset|account)s?\s+(freez|attach)|freez\w* of (assets|accounts|bank)|attachment of"),
    ("fflo:Sealing", r"\bseal(ed|ing|s)?\b"),
    ("fflo:Closure", r"\bclos(ures?|ed|ing)\b|demolition|suspension of (six |\d+ )?(eateries|outlets|shops|units)|suspension of manufactur|supply suspension|shut ?down|\bshut\b|stop[- ]?(production|work|operations)|suspension of (business|operations|establishments)|temporary (halt|hold|suspension)|procurement stoppage|halt of supplies"),
    ("fflo:ProhibitionOrder", r"\bban(s|ned|ning)?\b|prohibit|suspension of sale|stop[- ]?sale|sale stoppage|restrict\w* (on )?sale|stoppage of sale|sale suspension|work restriction|suspend or restrict sale"),
    ("fflo:Destruction", r"destr(oy|uction|oyed)|discard|dispos(al|ed)|dump(ed|ing)|drained"),
    ("fflo:ProductRecall", r"recall|withdraw(al|n)? (of|from)|market withdrawal"),
    ("fflo:Blacklisting", r"blacklist|debar|contract cancell?ation"),
    ("fflo:Removal", r"suspension of (a |the )?\w*\s?(officer|engineer|official|employee|registrar|commissioner)|dismissal of|removal of (a |the )?\w*\s?(officer|official)|transfer of|reprimand"),
    ("fflo:Notice", r"\bnotices?\b|show[- ]cause|request for justification|warning (to|issued to|against)|regulatory warning"),
    ("fflo:Investigation", r"investigat|inquir|enquir|\bprobe|\bsit\b|commission\b|\bcbi\b|\bacb\b|pmla|money laundering"),
    ("fflo:Inspection", r"\braids?\b|inspect|\bsearch(es)?\b|\bdrives?\b|crackdown|surveil|\bchecks?\b|checking|campaign|decoy|\boperation \w+|\bbust(ed)?\b|enforcement operation|sampl|\btesting\b|monitoring|vigil"),
    ("fflo:Advisory", r"advis|directive|\bdirect(s|ion|ions)\b|guideline|circular|instruction|awareness|initiative|\bappeal to|public appeal|\balert\b|\bwarnings?\b|\beat right\b"),
]
TYPE_RE = [(t, re.compile(p, re.I)) for t, p in TYPE_RULES]

# Types that are sanctions or enforcement steps against an operator (as opposed
# to policy measures). Only these get the weak "only FBO in snippet" link.
PUNITIVE = {"fflo:LicenceCancellation", "fflo:LicenceSuspension", "fflo:MonetaryPenalty",
            "fflo:Prosecution", "fflo:Arrest", "fflo:Seizure", "fflo:Sealing", "fflo:Closure",
            "fflo:ProhibitionOrder", "fflo:Destruction", "fflo:ProductRecall",
            "fflo:Blacklisting", "fflo:Notice"}

GENERIC_FBO = re.compile(
    r"^(the )?(food business operators?|fbos?|traders?|vendors?|street vendors?|shopkeepers?|sellers?|"
    r"manufacturers?|dealers?|suppliers?|retailers?|wholesalers?|businesses|accused|owners?|"
    r"dairy owners?|milk vendors?|eateries|restaurants?|hotels?|units?|firms?|companies)$", re.I)

AMOUNT_RE = re.compile(
    r"(?:(₹|\brs\.?|\binr)\s?)?(\d[\d,]*(?:\.\d+)?)\s?(lakhs?|lacs?|crores?|cr\b|thousand|k\b)?", re.I)
STATUTORY_RE = re.compile(r"\bup ?to\b|\bmaximum\b|\bmax\.?\b|\bextend(s|ing)? to\b|\bas much as\b", re.I)
MULT = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7,
        "cr": 1e7, "thousand": 1e3, "k": 1e3}


# A bare amount is not a penalty when the label is about payouts or valuation.
NOT_PENALTY = re.compile(r"compensat|ex.?gratia|relief|subsid|\bworth\b|valued|\bcost of\b|waiver|reward", re.I)
PENALTY_WORD = re.compile(r"penalt|\bfine|challan|levy|levying|sanction", re.I)


def _label_text(label: str) -> str:
    """'RegulatoryAction_LicenseCancellation' -> 'Regulatory Action License Cancellation'."""
    t = re.sub(r"[_]+", " ", label)
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t)


def classify_label(label: str) -> list[str]:
    label = _label_text(label)
    types = [t for t, rx in TYPE_RE if rx.search(label)]
    if "fflo:MonetaryPenalty" in types and NOT_PENALTY.search(label) and not PENALTY_WORD.search(label):
        types.remove("fflo:MonetaryPenalty")
    # FSS Act adjudication is the civil penalty route; its "cases" are not prosecutions.
    if "fflo:Prosecution" in types and re.search(r"adjudicat", label, re.I) \
            and not re.search(r"prosecut|\bfir\b|court|criminal|charge", label, re.I):
        types.remove("fflo:Prosecution")
    return types


def parse_amount(text: str) -> float | None:
    """First currency-qualified (or lakh/crore-qualified) amount in *text*, in INR."""
    for m in AMOUNT_RE.finditer(text):
        cur, num, unit = m.group(1), m.group(2), (m.group(3) or "").lower()
        if not cur and not unit:
            continue
        try:
            val = float(num.replace(",", ""))
        except ValueError:
            continue
        val *= MULT.get(unit, 1)
        if val >= 100:  # filters stray section numbers like "Rs. 5" OCR noise
            return round(val, 2)
    return None


# ---------------------------------------------------------------------------
# Dates. reportedDate = publication date of the article (metadata, always
# available when the article is known); actionDate = a date the text states
# for the event, resolved against the publication date. Repeat analysis uses
# actionDate when present, else reportedDate.
# ---------------------------------------------------------------------------
MONTHS = {m: i for i, m in enumerate(
    "jan feb mar apr may jun jul aug sep oct nov dec".split(), start=1)}
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_PATTERNS = [
    # 5 March 2024 / 5th March, 2024 / 5 March
    ("dmy", re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MON},?\s*(\d{{4}})?\b", re.I)),
    # March 5, 2024 / March 5
    ("mdy", re.compile(rf"\b{_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b,?\s*(\d{{4}})?", re.I)),
    # 05.03.2024 / 05-03-2024 / 05/03/24 (Indian day-first)
    ("num", re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})\b")),
]
WEEKDAY_RE = re.compile(r"\b(?:on|last|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
RELATIVE_RE = re.compile(r"\b(yesterday|today|on the same day)\b", re.I)
WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday".split()


def _url_key(url: str) -> str:
    return url.split("?")[0].split("#")[0].rstrip("/").replace("://www.", "://").lower()


URL_DATE_PATTERNS = [
    re.compile(rf"/(20\d\d)/{_MON}/(\d{{1,2}})/", re.I),        # newindianexpress /2024/Oct/05/
    re.compile(r"/(20\d\d)/(\d{1,2})/(\d{1,2})/"),             # /2024/10/05/
    re.compile(r"(20\d\d)-(\d{2})-(\d{2})(?:\D|$)"),           # -2022-05-20
]


def _date_from_url(url: str):
    import datetime as dt
    for i, rx in enumerate(URL_DATE_PATTERNS):
        m = rx.search(url)
        if not m:
            continue
        try:
            y = int(m.group(1))
            mo = MONTHS[m.group(2)[:3].lower()] if i == 0 else int(m.group(2))
            return dt.date(y, mo, int(m.group(3)))
        except (ValueError, KeyError):
            continue
    return None


def load_report_dates(news: pd.DataFrame, articles_csv: Path) -> dict[str, tuple[str, str]]:
    """source_id -> (ISO publication date, basis) from article metadata or the URL."""
    import datetime as dt
    by_url: dict[str, str] = {}
    if articles_csv.exists():
        art = pd.read_csv(articles_csv, dtype=str, usecols=["url", "date"]).fillna("")
        by_url = {_url_key(u): d for u, d in zip(art.url, art.date) if d}
    out: dict[str, tuple[str, str]] = {}
    for sid, url in news[["source_id", "source_file"]].drop_duplicates("source_id").itertuples(index=False):
        d = by_url.get(_url_key(url))
        if d:
            try:
                out[sid] = (dt.date.fromisoformat(d[:10]).isoformat(), "article_metadata")
                continue
            except ValueError:
                pass
        u = _date_from_url(url)
        if u:
            out[sid] = (u.isoformat(), "url")
    return out


def parse_event_date(text: str, reported: str | None):
    """First absolute or resolvable date stated in *text*, as (date, kind).

    Day-month without a year takes the publication year (minus one if that
    would fall after publication). Weekdays and "yesterday/today" resolve to
    the latest such day on or before publication. Dates after publication or
    before 2000 are rejected.
    """
    import datetime as dt
    ref = dt.date.fromisoformat(reported) if reported else None
    for kind, rx in DATE_PATTERNS:
        for m in rx.finditer(text):
            try:
                if kind == "dmy":
                    d, mo, y = int(m.group(1)), MONTHS[m.group(2)[:3].lower()], m.group(3)
                elif kind == "mdy":
                    mo, d, y = MONTHS[m.group(1)[:3].lower()], int(m.group(2)), m.group(3)
                else:
                    d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
                    y = ("20" + y) if y and len(y) == 2 else y
                if y:
                    cand = dt.date(int(y), mo, d)
                elif ref:
                    cand = dt.date(ref.year, mo, d)
                    if cand > ref + dt.timedelta(days=1):
                        cand = dt.date(ref.year - 1, mo, d)
                else:
                    continue
            except (ValueError, KeyError):
                continue
            if cand.year >= 2000 and (ref is None or cand <= ref + dt.timedelta(days=1)):
                return cand.isoformat(), "stated" if y else "stated_no_year"
    if ref:
        m = WEEKDAY_RE.search(text)
        if m:
            back = (ref.weekday() - WEEKDAYS.index(m.group(1).lower())) % 7
            return (ref - dt.timedelta(days=back)).isoformat(), "weekday"
        m = RELATIVE_RE.search(text)
        if m:
            back = 1 if m.group(1).lower() == "yesterday" else 0
            return (ref - dt.timedelta(days=back)).isoformat(), "relative"
    return None


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


_SUFFIX = re.compile(r"\b(pvt\.?|private|ltd\.?|limited|llp|inc\.?|co\.?|& co|company|firm|"
                     r"m/s\.?|the|shri|sri)\b", re.I)


def _name_key(name: str) -> str:
    """Lower-cased operator name without legal suffixes/honorifics, for label matching."""
    return re.sub(r"\s+", " ", _SUFFIX.sub(" ", re.sub(r"[^\w&/ ]", " ", name.lower()))).strip()


def _named_in(fbo: str, label: str) -> bool:
    key = _name_key(fbo)
    return len(key) >= 4 and re.search(rf"\b{re.escape(key)}\b", _name_key(label)) is not None


def build(news_csv: Path, out_csv: Path, articles_csv: Path = ARTICLES_CSV) -> pd.DataFrame:
    df = pd.read_csv(news_csv, low_memory=False, dtype=str).fillna("")
    for c in ("subject", "object"):
        df[c] = df[c].map(_norm)
    report_dates = load_report_dates(df, articles_csv)
    dated: set[tuple[str, str]] = set()  # (source_id, event node) already dated

    by_snip = {sid: g for sid, g in df.groupby("snippet_id", sort=False)}
    is_fbo_row = (df.subject_type == FBO) | (df.object_type == FBO)
    article_fbos: dict[str, set[str]] = defaultdict(set)
    for r in df[is_fbo_row].itertuples():
        for name, typ in ((r.subject, r.subject_type), (r.object, r.object_type)):
            if typ == FBO and not GENERIC_FBO.match(name):
                article_fbos[r.source_id].add(name)
    out: list[dict] = []
    stats: Counter = Counter()

    def row(meta: pd.Series, subj: str, pred: str, obj: str, obj_type: str,
            spans: list[str], rule: str, conf: float, status: str,
            subj_type: str = ACTION) -> dict:
        return {
            "snippet_id": meta.snippet_id, "source_id": meta.source_id,
            "source_file": meta.source_file, "source_type": meta.source_type,
            "chunk_index": meta.chunk_index,
            "subject": subj, "subject_type": subj_type, "subject_id": "",
            "predicate": pred, "object": obj, "object_type": obj_type, "object_id": "",
            "confidence": f"{conf:.2f}",
            "evidence_span": " | ".join(dict.fromkeys(s for s in spans if s))[:600],
            "status": status, "rule": rule,
            "judge_verdict": "", "judge_confidence": "", "judge_rationale": "",
        }

    for sid, g in by_snip.items():
        meta = g.iloc[0]

        # -- 0. dates for every event node (action, finding, act) in the chunk
        reported = report_dates.get(meta.source_id)
        events = {(r.subject, r.subject_type) for r in g.itertuples() if r.subject_type in EVENT_TYPES}
        events |= {(r.object, r.object_type) for r in g.itertuples() if r.object_type in EVENT_TYPES}
        for node, ntype in sorted(events):
            key = (meta.source_id, node)
            e_spans = list(g.loc[(g.subject == node) | (g.object == node), "evidence_span"])
            if key not in dated and reported:
                stats[f"reported:{reported[1]}"] += 1
                out.append(row(meta, node, "fflo:reportedDate", reported[0], "xsd:date",
                               [meta.source_file], reported[1], 1.0, "metadata", ntype))
            stated = parse_event_date(" | ".join([node, *e_spans]), reported[0] if reported else None)
            if stated and (key, "stated") not in dated:
                dated.add((key, "stated"))
                stats[f"action_date:{stated[1]}"] += 1
                out.append(row(meta, node, "fflo:actionDate", stated[0], "xsd:date",
                               e_spans, f"date_{stated[1]}", 0.9 if stated[1] == "stated" else 0.7,
                               "pending_judge", ntype))
            dated.add(key)

        actions = sorted(set(g.loc[g.subject_type == ACTION, "subject"]) |
                         set(g.loc[g.object_type == ACTION, "object"]))
        if not actions:
            continue
        fbos = set(g.loc[g.subject_type == FBO, "subject"]) | set(g.loc[g.object_type == FBO, "object"])
        specific_fbos = {f for f in fbos if not GENERIC_FBO.match(f)}

        # finding / act -> FBO, and finding / act <-> action, within the snippet
        attributed = defaultdict(set)  # finding-or-act -> FBOs
        for r in g.itertuples():
            if r.predicate == "prov:wasAttributedTo" and r.object_type == FBO:
                attributed[r.subject].add(r.object)
            if r.predicate == "prov:wasAssociatedWith" and r.object_type == FBO:
                attributed[r.subject].add(r.object)

        for a in actions:
            stats["actions"] += 1
            a_rows = g[(g.subject == a) | (g.object == a)]
            spans = list(a_rows.evidence_span)

            # -- 1. action type
            types = classify_label(a)
            if not types:
                stats["untyped"] += 1
                out.append(row(meta, a, "fflo:hasActionType", "", "fflo:RegulatoryActionType",
                               spans, "untyped", 0.0, "pending_type_llm"))
            for t in types:
                stats[t] += 1
                out.append(row(meta, a, "fflo:hasActionType", t.split(":")[1], t,
                               spans, "label_regex", 0.95, "pending_judge"))

            # -- 2. FBO link
            links: dict[str, tuple[str, float, list[str]]] = {}
            for r in a_rows.itertuples():
                other, otype = (r.object, r.object_type) if r.subject == a else (r.subject, r.subject_type)
                if otype == FBO:
                    # an FBO that *issued* the action (e.g. a brand's own advisory)
                    # is the actor, not the target
                    if r.predicate in ISSUER_PREDICATES or types == ["fflo:Advisory"]:
                        continue
                    links.setdefault(other, ("direct_edge", 0.9, [r.evidence_span]))
                elif otype in FINDING_TYPES or otype == ACT:
                    for f in sorted(attributed.get(other, ())):
                        f_spans = list(g.loc[(g.subject == other) & (g.object == f), "evidence_span"])
                        rule = "via_finding" if otype in FINDING_TYPES else "via_act"
                        links.setdefault(f, (rule, 0.8, [r.evidence_span, *f_spans]))
            for f in sorted(specific_fbos | article_fbos[meta.source_id]):
                if _named_in(f, a):
                    links.setdefault(f, ("fbo_in_label", 0.85, spans))
            # Weak fallbacks for sanctions (or not-yet-typed actions): the only
            # named operator in the chunk, else in the article. Judge decides.
            weak_ok = not types or PUNITIVE.intersection(types)
            if not links and weak_ok and len(specific_fbos) == 1:
                f = next(iter(specific_fbos))
                f_spans = list(g.loc[(g.subject == f) | (g.object == f), "evidence_span"])[:2]
                links[f] = ("sole_fbo_in_snippet", 0.6, [*spans[:2], *f_spans])
            elif not links and weak_ok and not fbos and len(article_fbos[meta.source_id]) == 1:
                f = next(iter(article_fbos[meta.source_id]))
                f_spans = list(df.loc[(df.source_id == meta.source_id) &
                                      ((df.subject == f) | (df.object == f)), "evidence_span"])[:2]
                links[f] = ("sole_fbo_in_article", 0.5, [*spans[:2], *f_spans])
            for f, (rule, conf, sp) in links.items():
                stats[f"link:{rule}"] += 1
                if GENERIC_FBO.match(f):
                    stats["link:generic_fbo"] += 1
                out.append(row(meta, a, "fflo:takenAgainst", f, FBO, sp, rule, conf, "pending_judge"))
            if links:
                stats["actions_linked"] += 1

            # -- 3. penalty amount
            if "fflo:MonetaryPenalty" in types:
                src_text = next((t for t in [a, *spans] if parse_amount(t) is not None), None)
                if src_text is not None and STATUTORY_RE.search(src_text):
                    stats["amount_statutory_skipped"] += 1
                elif src_text is not None:
                    amt = parse_amount(src_text)
                    stats["amount"] += 1
                    out.append(row(meta, a, "fflo:hasPenaltyAmount", f"{amt:.0f}", "xsd:decimal",
                                   [src_text] if src_text != a else spans, "amount_regex", 0.9,
                                   "pending_judge"))

    res = pd.DataFrame(out, columns=BASE_COLS + EXTRA_COLS)
    # The same action is often mentioned in several chunks of one article; keep
    # one row per (article, subject, predicate, object) so the judge sees it once
    # (the canonicaliser would collapse these after article scoping anyway).
    before = len(res)
    res = res.drop_duplicates(["source_id", "subject", "predicate", "object"], keep="first")
    stats["dupes_within_article"] = before - len(res)
    _write_csv_atomic(res, out_csv)
    print(f"wrote {len(res)} rows → {out_csv}")
    _report(res, stats)
    return res


def _report(res: pd.DataFrame, stats: Counter) -> None:
    print(f"\naction nodes (snippet-level): {stats['actions']}  "
          f"linked to an FBO: {stats['actions_linked']}  untyped: {stats['untyped']}")
    print("\nhasActionType by type:")
    at = res[res.predicate == "fflo:hasActionType"]
    print(at.object_type.value_counts().to_string())
    print("\ntakenAgainst by rule:")
    print(res[res.predicate == "fflo:takenAgainst"].rule.value_counts().to_string())
    print(f"  of which generic FBO labels: {stats['link:generic_fbo']}")
    print(f"\nhasPenaltyAmount rows: {stats['amount']}  "
          f"(statutory maxima skipped: {stats['amount_statutory_skipped']})")
    ev = res[res.predicate == "fflo:reportedDate"]
    print(f"\nreportedDate rows: {len(ev)} (by basis: {ev.rule.value_counts().to_dict()}; "
          f"by node type: {ev.subject_type.value_counts().to_dict()})")
    ad = res[res.predicate == "fflo:actionDate"]
    print(f"actionDate rows: {len(ad)} (by kind: {ad.rule.value_counts().to_dict()})")
    act_dates = res[(res.subject_type == ACTION) & res.predicate.isin(["fflo:reportedDate", "fflo:actionDate"])]
    n_act = res.loc[res.subject_type == ACTION, ["source_id", "subject"]].drop_duplicates().shape[0]
    print(f"actions (article-level) with a date: "
          f"{act_dates[['source_id', 'subject']].drop_duplicates().shape[0]} / {n_act}")
    _report_repeats(res)
    print(f"dropped {stats['dupes_within_article']} duplicate rows (same fact in several chunks of one article)")
    print("\nstatus:", res.status.value_counts().to_dict())


def _report_repeats(res: pd.DataFrame, window_days: int = 30) -> None:
    """Operators with more than one dated action instance (pre-judge sanity view).

    Instances against the same operator within *window_days* of each other are
    counted once: they are usually the same event reported by several outlets.
    """
    link = res[res.predicate == "fflo:takenAgainst"][["source_id", "subject", "object"]]
    dates = res[res.predicate.isin(["fflo:actionDate", "fflo:reportedDate"])]
    # actionDate wins over reportedDate for the same (article, action)
    dates = dates.assign(_p=(dates.predicate == "fflo:reportedDate").astype(int)) \
                 .sort_values("_p").drop_duplicates(["source_id", "subject"])
    j = link.merge(dates[["source_id", "subject", "object"]].rename(columns={"object": "date"}),
                   on=["source_id", "subject"])
    if j.empty:
        return
    j = j[~j["object"].map(lambda f: bool(GENERIC_FBO.match(f)))]
    j["date"] = pd.to_datetime(j["date"])
    # Coarse operator identity: "A R Dairy Food Private Limited" == "AR Dairy Foods".
    j["fbo_key"] = j["object"].map(lambda f: re.sub(r"s$", "", _name_key(f).replace(" ", "")))
    names = j.groupby("fbo_key")["object"].agg(lambda x: sorted(set(x), key=len)[-1])
    repeats = {}
    for key, g in j.groupby("fbo_key"):
        fbo = names[key]
        days = sorted(g["date"])
        episodes = 1 + sum((b - a).days > window_days for a, b in zip(days, days[1:]))
        if episodes > 1:
            repeats[fbo] = episodes
    print(f"FBOs with dated actions: {j.fbo_key.nunique()}; with >1 episode "
          f"(>{window_days} days apart): {len(repeats)} "
          f"{dict(sorted(repeats.items(), key=lambda x: -x[1])[:8])}")


# ---------------------------------------------------------------------------
# LLM helpers (type-llm, judge) — OpenAI-compatible endpoint (vLLM / Qwen)
# ---------------------------------------------------------------------------

async def _chat(client, base_url: str, api_key: str, model: str, system: str, user: str) -> dict:
    import httpx  # noqa: F401  (imported lazily: only the LLM stages need it)
    for attempt in range(4):
        try:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                json={"model": model, "temperature": 0.0, "max_tokens": 1024,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}],
                      "response_format": {"type": "json_object"},
                      "chat_template_kwargs": {"enable_thinking": False}},
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
            return json.loads(content)
        except Exception as exc:  # retry transient errors, give up after 4
            if attempt == 3:
                return {"error": str(exc)[:200]}
            await asyncio.sleep(2 ** attempt)
    return {}


def _vocab() -> list[str]:
    cfg = json.loads(SCHEMA.read_text(encoding="utf-8"))
    return [t.split(":")[1] for t, p in cfg["subclass_of"].items() if p == "fflo:RegulatoryActionType"]


# ---------------------------------------------------------------------------
# Checkpointing — same idea as validation/LLM_judge/judge.py (append-only JSONL
# + --resume), keyed on row content so a re-run of `build` keeps keys stable.
# ---------------------------------------------------------------------------

def _row_key(r) -> str:
    raw = f"{r['source_id']}|{r['subject']}|{r['predicate']}|{r['object']}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _ckpt_path(out_csv: Path, stage: str) -> Path:
    return out_csv.with_name(f"{out_csv.stem}.{stage.replace('-', '_')}.ckpt.jsonl")


def _load_ckpt(path: Path) -> dict[str, dict]:
    """key -> last successful result. Failed attempts are not 'done'."""
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # a line cut off by a crash
                continue
            if rec.get("ok"):
                done[rec["key"]] = rec["result"]
    return done


def _append_ckpt(fh, key: str, result: dict, ok: bool) -> None:
    fh.write(json.dumps({"key": key, "ok": ok, "result": result,
                         "ts": round(time.time(), 1)}, ensure_ascii=False) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def _write_csv_atomic(res: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    res.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _open_ckpt(args, stage: str) -> Path:
    path = _ckpt_path(args.out, stage)
    if path.exists() and args.fresh:
        path.unlink()
    elif path.exists() and not args.resume and path.stat().st_size > 0:
        n = sum(1 for _ in path.open("r", encoding="utf-8"))
        sys.exit(f"{path.name} already has {n} results from an earlier run.\n"
                 f"Re-run with --resume to continue it, or --fresh to discard it.")
    return path


async def _run_llm(res: pd.DataFrame, todo: pd.DataFrame, make_prompt, is_ok, apply,
                   mark_error, args, ckpt: Path, stage: str) -> tuple[int, int, bool]:
    """Call the LLM for every row of *todo*, checkpointing each result.

    Returns (done, errors, interrupted). SIGINT/SIGTERM cancel cleanly: results
    already received are on disk and the caller writes a final snapshot.
    """
    import httpx
    sem = asyncio.Semaphore(args.concurrency)
    total, t0 = len(todo), time.time()
    counts = {"done": 0, "errors": 0}
    main_task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, main_task.cancel)
        except (NotImplementedError, RuntimeError):
            pass

    with ckpt.open("a", encoding="utf-8") as fh:
        async with httpx.AsyncClient() as client:
            async def one(idx):
                async with sem:
                    system, user = make_prompt(res.loc[idx])
                    out = await _chat(client, args.base_url, args.api_key, args.model, system, user)
                ok = is_ok(out)
                _append_ckpt(fh, _row_key(res.loc[idx]), out, ok)
                if ok:
                    apply(idx, out)
                else:
                    counts["errors"] += 1
                    mark_error(idx, out)
                counts["done"] += 1
                d = counts["done"]
                if d % 50 == 0 or d == total:
                    print(f"[{stage}] {d}/{total} done, {counts['errors']} errors, "
                          f"{time.time() - t0:.0f}s", flush=True)
                if d % 100 == 0:
                    _write_csv_atomic(res, args.out)

            try:
                await asyncio.gather(*(one(i) for i in todo.index))
                interrupted = False
            except asyncio.CancelledError:
                interrupted = True
    return counts["done"], counts["errors"], interrupted


def _finish(res: pd.DataFrame, args, stage: str, done: int, errors: int,
            interrupted: bool, pending_status: str) -> bool:
    """Write the final snapshot; say how to resume if anything is left."""
    _write_csv_atomic(res, args.out)
    left = int((res.status == pending_status).sum())
    if interrupted:
        print(f"\n[{stage}] interrupted after {done} calls; results so far are saved.", flush=True)
    if left:
        print(f"[{stage}] {left} rows still {pending_status} ({errors} failed calls this run). "
              f"Re-run the same command with --resume to finish them.", flush=True)
    return left == 0


def _vocab() -> list[str]:
    cfg = json.loads(SCHEMA.read_text(encoding="utf-8"))
    return [t.split(":")[1] for t, p in cfg["subclass_of"].items() if p == "fflo:RegulatoryActionType"]


def type_llm(args) -> None:
    ckpt = _open_ckpt(args, "type-llm")
    res = pd.read_csv(args.out, dtype=str).fillna("")
    vocab = _vocab()
    glosses = "\n".join(f"- {t}: {TYPE_GLOSSES[t]}" for t in vocab if t in TYPE_GLOSSES)
    system = (
        "You classify an action reported in Indian food-safety news into ONE type from a closed "
        "list. Read the label together with the evidence.\n\nTypes:\n"
        f"{glosses}\n\n"
        "Answer 'none' when the text is not a concrete action by an authority, police or court "
        "against a business, product or official. Examples of 'none': compensation or ex-gratia "
        "payments, free treatment, tax / duty / cess changes, price caps, stock limits, loan "
        "waivers, licence relaxations, certificates or NOCs granted, and vague phrases with no "
        "concrete step ('strict action', 'necessary action', 'regulatory action').\n"
        'Return JSON: {"type": "<one type from the list, or none>", "rationale": "<=15 words"}.'
    )

    def prompt(r):
        return system, f"ACTION LABEL: {r.subject}\nEVIDENCE: {r.evidence_span}"

    def is_ok(out):
        return "error" not in out and bool(str(out.get("type", "")).strip())

    def apply(idx, out):
        t = str(out.get("type", "")).strip()
        if t in vocab:
            res.loc[idx, ["object", "object_type", "rule", "status", "confidence"]] = \
                [t, f"fflo:{t}", "label_llm", "pending_judge", "0.80"]
        else:
            res.loc[idx, ["rule", "status"]] = ["label_llm_none", "dropped_no_type"]

    def mark_error(idx, out):  # stays pending_type_llm -> retried on --resume
        res.loc[idx, "rule"] = "type_llm_error"

    if args.fresh:  # start the stage over: undo earlier LLM typing as well
        redo = res.rule.isin(["label_llm", "label_llm_none", "type_llm_error"])
        res.loc[redo, ["object", "object_type", "rule", "status", "confidence",
                       "judge_verdict", "judge_confidence", "judge_rationale"]] = \
            ["", "fflo:RegulatoryActionType", "untyped", "pending_type_llm", "0.00", "", "", ""]
        print(f"[type-llm] --fresh: reset {int(redo.sum())} previously LLM-typed rows", flush=True)
    todo = res[res.status == "pending_type_llm"]
    cached = _load_ckpt(ckpt)
    reused = [i for i in todo.index if _row_key(res.loc[i]) in cached]
    for i in reused:
        apply(i, cached[_row_key(res.loc[i])])
    todo = res[res.status == "pending_type_llm"]
    print(f"[type-llm] {len(reused)} results reused from checkpoint, {len(todo)} to send", flush=True)

    done, errors, interrupted = asyncio.run(
        _run_llm(res, todo, prompt, is_ok, apply, mark_error, args, ckpt, "type-llm"))
    _finish(res, args, "type-llm", done, errors, interrupted, "pending_type_llm")
    print("[type-llm] status:", res.status.value_counts().to_dict(), flush=True)
    if interrupted:
        sys.exit(130)


# One-line definitions shown to the LLM in type-llm (same meanings as kg_setup.md §1.7).
TYPE_GLOSSES = {
    "Inspection": "raid, search, inspection drive, sampling campaign, enforcement operation, bust",
    "Seizure": "goods, stock, documents or assets seized, detained or frozen",
    "Sealing": "premises, unit or stock sealed",
    "Notice": "improvement notice, show-cause notice, warning issued to a named party",
    "ProhibitionOrder": "ban or stop-sale on a product; sale or manufacture prohibited",
    "Closure": "business, unit or supply shut down, suspended or demolished",
    "LicenceSuspension": "FSSAI licence or registration suspended",
    "LicenceCancellation": "FSSAI licence or registration cancelled or revoked",
    "MonetaryPenalty": "fine, penalty, challan, adjudication penalty",
    "Prosecution": "FIR, police complaint, case filed, charge-sheet, court proceedings, conviction, bail denied",
    "Arrest": "persons arrested, detained by police, judicial custody",
    "Destruction": "goods destroyed, discarded or dumped",
    "ProductRecall": "product recalled or withdrawn from the market",
    "Blacklisting": "supplier blacklisted, debarred or contract cancelled",
    "Investigation": "SIT, inquiry, probe, committee to investigate",
    "Advisory": "directive, advisory, guideline, public appeal or awareness measure (not a sanction)",
    "Removal": "official suspended, transferred, dismissed or reprimanded",
    "PenaltyReduction": "penalty reduced or waived",
    "Authorization": "authority granted a permission or approval",
}

JUDGE_VERDICTS = {"entailed", "partially_entailed", "not_entailed", "uncertain"}  # schema_valid labels


def judge(args) -> None:
    sys.path.insert(0, str(ROOT / "validation" / "LLM_judge"))
    from judge_prompts import build_schema_valid_user, schema_valid_system

    ckpt = _open_ckpt(args, "judge")
    res = pd.read_csv(args.out, dtype=str).fillna("")
    news = pd.read_csv(args.news, dtype=str, low_memory=False).fillna("")
    chunk_text: dict[str, str] = {}
    if args.chunks_csv and args.chunks_csv.exists():
        ch = pd.read_csv(args.chunks_csv, dtype=str).fillna("")
        text_col = "text" if "text" in ch.columns else ch.columns[-1]
        chunk_text = dict(zip(ch.snippet_id, ch[text_col]))
    # Fallback premise: every evidence span extracted from that news chunk.
    spans_by_snip = news.groupby("snippet_id").evidence_span.apply(
        lambda s: "\n".join(dict.fromkeys(x for x in s if x))).to_dict()
    system = schema_valid_system(str(SCHEMA))

    def prompt(r):
        chunk = chunk_text.get(r.snippet_id) or spans_by_snip.get(r.snippet_id, r.evidence_span)
        return system, build_schema_valid_user(chunk, r.to_dict())

    def is_ok(out):
        return "error" not in out and str(out.get("verdict", "")).strip() in JUDGE_VERDICTS

    def apply(idx, out):
        verdict = str(out.get("verdict", "")).strip()
        res.loc[idx, ["judge_verdict", "judge_confidence", "judge_rationale", "status"]] = [
            verdict, str(out.get("confidence", "")), str(out.get("rationale", ""))[:200],
            "accepted" if verdict in ("entailed", "partially_entailed") else "rejected",
        ]

    def mark_error(idx, out):  # stays pending_judge -> retried on --resume
        res.loc[idx, ["judge_verdict", "judge_rationale"]] = [
            "judge_error", str(out.get("error", "no verdict in reply"))[:200]]

    # Rows the pre-resume script marked "rejected" because the *call* failed
    # carry an error message instead of a verdict; they were never judged.
    stale = (res.status == "rejected") & ~res.judge_verdict.isin(JUDGE_VERDICTS)
    if stale.any():
        res.loc[stale, ["status", "judge_verdict", "judge_confidence", "judge_rationale"]] = \
            ["pending_judge", "", "", ""]
        print(f"[judge] {int(stale.sum())} rows were 'rejected' by a failed call, not a verdict; "
              f"re-queued", flush=True)
    if args.fresh:  # start the stage over: earlier verdicts are discarded too
        redo = res.status.isin(["accepted", "rejected"]) | (res.judge_verdict == "judge_error")
        res.loc[redo, ["status", "judge_verdict", "judge_confidence", "judge_rationale"]] = \
            ["pending_judge", "", "", ""]
        print(f"[judge] --fresh: reset {int(redo.sum())} previously judged rows", flush=True)
    todo = res[res.status == "pending_judge"]
    cached = _load_ckpt(ckpt)
    reused = [i for i in todo.index if _row_key(res.loc[i]) in cached]
    for i in reused:
        apply(i, cached[_row_key(res.loc[i])])
    todo = res[res.status == "pending_judge"]
    print(f"[judge] {len(reused)} verdicts reused from checkpoint, {len(todo)} to send", flush=True)

    done, errors, interrupted = asyncio.run(
        _run_llm(res, todo, prompt, is_ok, apply, mark_error, args, ckpt, "judge"))
    complete = _finish(res, args, "judge", done, errors, interrupted, "pending_judge")

    judged = res[res.status.isin(["accepted", "rejected"])]
    print("[judge] verdicts so far:", judged.judge_verdict.value_counts().to_dict(), flush=True)
    validated = args.out.with_name(args.out.stem + "_validated.csv")
    if complete:
        ok = res[res.status.isin(["accepted", "metadata"])]  # metadata = reportedDate
        _write_csv_atomic(ok, validated)
        print(f"accepted {int((res.status == 'accepted').sum())} / {len(judged)} judged → {validated}")
        print(ok.groupby("predicate").size().to_string())
    else:
        print(f"[judge] {validated.name} not written until every row is judged.", flush=True)
    if (res.status == "pending_type_llm").any():
        print(f"[judge] note: {(res.status == 'pending_type_llm').sum()} rows still await type-llm "
              f"and are not in the validated file.", flush=True)
    if interrupted:
        sys.exit(130)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    b = sub.add_parser("build")
    b.add_argument("--news", type=Path, default=NEWS_CSV)
    b.add_argument("--out", type=Path, default=OUT_CSV)
    b.add_argument("--articles-csv", type=Path, default=ARTICLES_CSV,
                   help="article metadata (url, date) used for fflo:reportedDate")
    for name in ("type-llm", "judge"):
        p = sub.add_parser(name)
        p.add_argument("--news", type=Path, default=NEWS_CSV)
        p.add_argument("--out", type=Path, default=OUT_CSV)
        p.add_argument("--base-url", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--api-key", default="")
        p.add_argument("--concurrency", type=int, default=8)
        p.add_argument("--chunks-csv", type=Path, default=None,
                       help="optional news chunk CSV (snippet_id,text) used as the judge premise")
        g = p.add_mutually_exclusive_group()
        g.add_argument("--resume", action="store_true",
                       help="continue from the checkpoint: reuse finished calls, retry failed ones")
        g.add_argument("--fresh", action="store_true",
                       help="start this stage over: discard the checkpoint and reset rows the "
                            "stage already decided (verdicts / LLM types) to pending")
    args = ap.parse_args()
    if args.stage == "build":
        build(args.news, args.out, args.articles_csv)
    elif args.stage == "type-llm":
        type_llm(args)
    else:
        judge(args)


if __name__ == "__main__":
    main()
