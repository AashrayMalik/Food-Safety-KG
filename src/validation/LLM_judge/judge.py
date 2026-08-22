#!/usr/bin/env python3
"""LLM-as-Judge entailment validation for knowledge-graph triplet extraction.

Reads parsed triplets, joins them with source chunks, routes each item into
one of four evaluation branches, and uses a second LLM call to judge
entailment / schema-correctness per branch.  Results are written as JSONL
and aggregated into a JSON report.

Directory layout::

    outputs/validation/{csv_id}/{judge_model}/run_{timestamp}/
        zero_triplet_judgments.jsonl
        schema_mismatch_judgments.jsonl
        schema_invalid_judgments.jsonl
        schema_valid_judgments.jsonl
        aggregate_report.json

Resume: pass ``--resume`` to find the most recent run directory for the
same (csv_id, judge_model) combination and continue where it left off.

Usage::

    food_lab/bin/python src/validation/LLM_judge/judge.py \\
        --chunks-csv  src/data/FSSAI_docs/processed/chunks.csv \\
        --triplets-dir outputs/triplets \\
        --output-dir  outputs/validation \\
        --judge-model deepseek-chat \\
        --concurrency 8 \\
        [--resume]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# Allow importing judge_prompts from the same directory
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from judge_prompts import (
    zero_triplet_system,
    schema_mismatch_system,
    schema_invalid_system,
    schema_valid_system,
    build_zero_triplet_user,
    build_schema_mismatch_user,
    build_schema_invalid_user,
    build_schema_valid_user,
    get_schema_text,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-Pro"
DEFAULT_CONCURRENCY = 8
DEFAULT_TIMEOUT = 180
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_CHUNK_CHARS = 3000  # truncate long chunks in prompts

# Branch file names
BRANCH_FILES = {
    "zero_triplet": "zero_triplet_judgments.jsonl",
    "schema_mismatch": "schema_mismatch_judgments.jsonl",
    "schema_invalid_triplet": "schema_invalid_judgments.jsonl",
    "schema_valid_triplet": "schema_valid_judgments.jsonl",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_chunks(csv_path: Path) -> dict[str, dict[str, str]]:
    """Return snippet_id → row dict (evidence_text, source_id, …)."""
    lookup: dict[str, dict[str, str]] = {}
    if not csv_path.exists():
        print(f"  [WARN]  chunks CSV not found: {csv_path}", file=sys.stderr, flush=True)
        return lookup
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            lookup[row["snippet_id"]] = row
    return lookup


def load_triplets(jsonl_path: Path) -> list[dict[str, Any]]:
    """Return list of per-snippet parsed outputs."""
    items: list[dict[str, Any]] = []
    if not jsonl_path.exists():
        print(f"  [WARN]  triplets JSONL not found: {jsonl_path}", file=sys.stderr, flush=True)
        return items
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return items


def load_failures(jsonl_path: Path) -> list[dict[str, Any]]:
    """Return list of hard-failure records."""
    items: list[dict[str, Any]] = []
    if not jsonl_path.exists():
        return items
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return items


def load_violations(jsonl_path: Path) -> dict[tuple[str, int], list[str]]:
    """Return (snippet_id, triplet_index) → list of violation_type strings."""
    lookup: dict[tuple[str, int], list[str]] = defaultdict(list)
    if not jsonl_path.exists():
        return lookup
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (obj.get("snippet_id", ""), obj.get("triplet_index", -1))
            lookup[key].append(obj.get("violation_type", ""))
    return lookup


# ---------------------------------------------------------------------------
# Routing — classify every triplet-level record into a branch
# ---------------------------------------------------------------------------

class JudgeItem:
    __slots__ = (
        "key", "branch", "snippet_id", "triplet_index",
        "chunk_text", "model_notes", "triplet", "violations",
    )

    def __init__(
        self,
        key: str,
        branch: str,
        snippet_id: str,
        chunk_text: str,
        *,
        triplet_index: int | None = None,
        model_notes: str = "",
        triplet: dict[str, Any] | None = None,
        violations: list[str] | None = None,
    ) -> None:
        self.key = key
        self.branch = branch
        self.snippet_id = snippet_id
        self.triplet_index = triplet_index
        self.chunk_text = chunk_text
        self.model_notes = model_notes
        self.triplet = triplet or {}
        self.violations = violations or []


def classify_items(
    chunks_lookup: dict[str, dict[str, str]],
    triplet_outputs: list[dict[str, Any]],
    violations: dict[tuple[str, int], list[str]],
) -> dict[str, list[JudgeItem]]:
    """Route every parsed output into zero_triplet / schema_mismatch /
    schema_invalid_triplet / schema_valid_triplet branches.

    Hard failures (failures.jsonl) are NOT routed here — they are
    deterministic API/parse errors with no triplet to judge.
    """
    branches: dict[str, list[JudgeItem]] = {
        "zero_triplet": [],
        "schema_mismatch": [],
        "schema_invalid_triplet": [],
        "schema_valid_triplet": [],
    }

    for obj in triplet_outputs:
        sid = obj.get("snippet_id", "")
        notes = obj.get("notes", "") or ""
        chunk = chunks_lookup.get(sid, {})
        chunk_text = chunk.get("evidence_text", "") or ""
        triplets = obj.get("triplets", []) or []

        if not triplets:
            # Zero-triplet branch
            key = sid
            branches["zero_triplet"].append(JudgeItem(
                key=key, branch="zero_triplet",
                snippet_id=sid, chunk_text=chunk_text,
                model_notes=notes,
            ))
            continue

        for ti, t in enumerate(triplets):
            pred = t.get("predicate", "")
            key = f"{sid}::{ti}"

            if pred == "SCHEMA_MISMATCH":
                branches["schema_mismatch"].append(JudgeItem(
                    key=key, branch="schema_mismatch",
                    snippet_id=sid, chunk_text=chunk_text,
                    triplet_index=ti, triplet=t,
                ))
            elif (sid, ti) in violations:
                branches["schema_invalid_triplet"].append(JudgeItem(
                    key=key, branch="schema_invalid_triplet",
                    snippet_id=sid, chunk_text=chunk_text,
                    triplet_index=ti, triplet=t,
                    violations=violations[(sid, ti)],
                ))
            else:
                branches["schema_valid_triplet"].append(JudgeItem(
                    key=key, branch="schema_valid_triplet",
                    snippet_id=sid, chunk_text=chunk_text,
                    triplet_index=ti, triplet=t,
                ))

    return branches


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(item: JudgeItem, schema_path: str | None = None) -> tuple[str, str]:
    """Return (system_prompt, user_message) for a JudgeItem."""
    if item.branch == "zero_triplet":
        return (
            zero_triplet_system(schema_path),
            build_zero_triplet_user(item.chunk_text, item.model_notes),
        )
    elif item.branch == "schema_mismatch":
        return (
            schema_mismatch_system(schema_path),
            build_schema_mismatch_user(item.chunk_text, item.triplet),
        )
    elif item.branch == "schema_invalid_triplet":
        return (
            schema_invalid_system(schema_path),
            build_schema_invalid_user(item.chunk_text, item.triplet, item.violations),
        )
    elif item.branch == "schema_valid_triplet":
        return (
            schema_valid_system(schema_path),
            build_schema_valid_user(item.chunk_text, item.triplet),
        )
    raise ValueError(f"Unknown branch: {item.branch}")


# ---------------------------------------------------------------------------
# LLM calling
# ---------------------------------------------------------------------------

class JudgeError(Exception):
    """Non-retryable judge error."""


class RetryableError(Exception):
    """Transient failure worth retrying."""


async def call_judge(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    timeout: float,
    use_logprobs: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    """Send one chat-completion request and return (parsed JSON verdict,
    logprobs_content list or None).

    When *use_logprobs* is True the request payload includes
    ``"logprobs": True`` and the second element of the return tuple
    is the ``choice["logprobs"]["content"]`` array.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }
    if use_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 1

    response = await client.post(
        url, json=payload, headers=headers, timeout=httpx.Timeout(timeout),
    )

    if response.status_code == 401:
        body = response.text[:500]
        raise JudgeError(f"Authentication failed. Check API key. {body}")
    if response.status_code == 400:
        body = response.text[:500]
        raise JudgeError(f"Bad request (400). {body}")
    if response.status_code in RETRYABLE_STATUSES:
        raise RetryableError(f"HTTP {response.status_code}: {response.text[:300]}")
    if response.status_code != 200:
        raise JudgeError(f"Unexpected HTTP {response.status_code}: {response.text[:500]}")

    data = response.json()
    choice = data.get("choices", [{}])[0]
    content = (choice.get("message", {}).get("content", "") or "").strip()

    if not content:
        finish = choice.get("finish_reason", "unknown")
        raise JudgeError(f"Empty judge response (finish_reason={finish})")

    # Strip markdown fences
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"Unparseable judge JSON: {exc}\nRaw: {content[:500]}")

    if not isinstance(parsed, dict):
        raise JudgeError(f"Expected JSON object, got {type(parsed).__name__}")

    # Stash the raw (de-fenced) content for logprob field extraction
    parsed["_raw_content"] = content

    logprobs_content: list[dict[str, Any]] | None = None
    if use_logprobs:
        logprobs = choice.get("logprobs") or {}
        logprobs_content = logprobs.get("content") or None

    return parsed, logprobs_content


# ---------------------------------------------------------------------------
# Logprob scoring — map token-level probabilities to JSON fields
# ---------------------------------------------------------------------------

def _extract_verdict_logprobs(
    content_raw: str,
    logprobs_content: list[dict[str, Any]],
) -> dict[str, Any]:
    """Map token-level logprobs to the ``"verdict"`` JSON field.

    Returns a dict with:
    - ``total_logprob``: sum of all output token logprobs
    - ``avg_logprob``: total_logprob / token count
    - ``output_perplexity``: exp(-avg_logprob) — lower = more certain
    - ``verdict_logprob``: logprob sum for tokens covering the verdict value
    - ``verdict_token_count``: number of tokens in the verdict value
    """
    import math

    # ---- cumulative byte-offset → token map ----
    byte_pos = 0
    token_map: dict[int, tuple[int, float, int]] = {}
    for i, lp in enumerate(logprobs_content):
        b = lp.get("bytes") or []
        b_len = len(b)
        if b_len:
            token_map[byte_pos] = (i, lp.get("logprob", 0.0), b_len)
        byte_pos += b_len

    total_logprob = sum(
        lp.get("logprob", 0.0) for lp in logprobs_content
    )
    num_tokens = len(logprobs_content)
    avg_logprob = total_logprob / max(num_tokens, 1)
    output_perplexity = math.exp(-avg_logprob) if avg_logprob else 0.0

    # ---- find the "verdict" value byte range ----
    verdict_logprob: float | None = None
    verdict_count: int = 0

    try:
        parsed = json.loads(content_raw)
    except json.JSONDecodeError:
        parsed = {}

    verdict_value = parsed.get("verdict", "")
    if isinstance(verdict_value, str) and verdict_value:
        escaped = re.escape(verdict_value)
        m = re.search(rf'"verdict"\s*:\s*"({escaped})"', content_raw)
        if not m:
            m = re.search(
                rf'"verdict"\s*:\s*({re.escape(str(verdict_value))})',
                content_raw,
            )
        if m:
            val_start_bytes = len(content_raw[: m.start(1)].encode("utf-8"))
            val_bytes = m.group(1).encode("utf-8")
            val_byte_end = val_start_bytes + len(val_bytes)

            token_indices: set[int] = set()
            for bp, (ti, _lp, blen) in token_map.items():
                if bp + blen > val_start_bytes and bp < val_byte_end:
                    token_indices.add(ti)
            verdict_logprob = sum(
                logprobs_content[i].get("logprob", 0.0)
                for i in token_indices
            )
            verdict_count = len(token_indices)

    return {
        "total_logprob": total_logprob,
        "avg_logprob": avg_logprob,
        "output_perplexity": round(output_perplexity, 3),
        "verdict_logprob": verdict_logprob,
        "verdict_token_count": verdict_count,
    }


# ---------------------------------------------------------------------------
# Branch processing
# ---------------------------------------------------------------------------

async def _judge_one(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    item: JudgeItem,
    timeout: float,
    output_path: Path,
    total: int,
    *,
    use_logprobs: bool = False,
    schema_path: str | None = None,
) -> tuple[str, str]:
    """Process a single JudgeItem and append result to output_path."""
    async with sem:
        short_key = item.key[:60]
        system_msg, user_msg = build_prompt(item, schema_path)
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                verdict_obj, logprobs_raw = await call_judge(
                    client, base_url, api_key, model,
                    system_msg, user_msg, timeout,
                    use_logprobs=use_logprobs,
                )

                # Enrich with metadata
                record: dict[str, Any] = {
                    "snippet_id": item.snippet_id,
                    "branch": item.branch,
                    "triplet_index": item.triplet_index,
                    "subject": item.triplet.get("subject", "") or None,
                    "predicate": item.triplet.get("predicate", "") or None,
                    "object": item.triplet.get("object", "") or None,
                    "evidence_span": item.triplet.get("evidence_span", "") or None,
                    "violations": item.violations or None,
                    "model_notes": item.model_notes or None,
                    "judge_verdict": verdict_obj.get("verdict", ""),
                    "judge_confidence": verdict_obj.get("confidence"),
                    "judge_rationale": verdict_obj.get("rationale", ""),
                }

                # Branch-specific optional fields
                for _f in ("evidence_span_exact", "entity_types_correct",
                           "direction_correct", "confidence_reasonable",
                           "existing_relation", "proposed_relation",
                           "proposed_domain", "proposed_range",
                           "correct_relation", "correct_subject_type",
                           "correct_object_type", "missed_triplets"):
                    if _f in verdict_obj:
                        record[_f] = verdict_obj[_f]

                # Attach logprob scores when available
                if use_logprobs and logprobs_raw:
                    raw_content = verdict_obj.pop("_raw_content", "")
                    lp_scores = _extract_verdict_logprobs(
                        raw_content, logprobs_raw,
                    )
                    record["total_logprob"] = lp_scores["total_logprob"]
                    record["avg_logprob"] = lp_scores["avg_logprob"]
                    record["output_perplexity"] = lp_scores["output_perplexity"]
                    if lp_scores["verdict_logprob"] is not None:
                        record["verdict_logprob"] = lp_scores["verdict_logprob"]
                        record["verdict_token_count"] = lp_scores["verdict_token_count"]
                else:
                    # Pop the helper key so it never leaks into output
                    verdict_obj.pop("_raw_content", None)

                record["timestamp"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                )

                _append_jsonl(output_path, record)
                done = _count_jsonl(output_path)
                print(f"  [{done:>5d}/{total}] OK  {short_key}", flush=True)
                return item.key, "ok"

            except RetryableError as exc:
                last_error = str(exc)
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    print(
                        f"  [·] RETRY {attempt}/{MAX_RETRIES} "
                        f"{short_key} (wait {delay:.0f}s)  {exc!r}",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                else:
                    print(f"  [✗] FAIL after retries {short_key}: {exc!r}",
                          flush=True)

            except JudgeError as exc:
                last_error = str(exc)
                print(f"  [✗] FAIL {short_key}: {exc!r}", flush=True)
                break

        # Write failure record
        fail_record: dict[str, Any] = {
            "snippet_id": item.snippet_id,
            "branch": item.branch,
            "triplet_index": item.triplet_index,
            "error": last_error,
            "judge_verdict": "judge_error",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _append_jsonl(output_path, fail_record)
        return item.key, "failed"


async def process_branch(
    branch_name: str,
    items: list[JudgeItem],
    output_path: Path,
    *,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_TIMEOUT,
    use_logprobs: bool = False,
    schema_path: str | None = None,
) -> tuple[int, int]:
    """Judge all items in a branch concurrently.  Returns (ok, failed)."""
    if not items:
        return 0, 0

    total = len(items)
    print(f"\n{'='*60}", flush=True)
    print(f"Branch: {branch_name}  ({total} items)", flush=True)
    print(f"{'='*60}", flush=True)

    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(
        max_keepalive_connections=concurrency + 5,
        max_connections=concurrency + 10,
    )
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [
            _judge_one(sem, client, base_url, api_key, model,
                       item, timeout, output_path, total,
                       use_logprobs=use_logprobs,
                       schema_path=schema_path)
            for item in items
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    ok = sum(1 for r in results if isinstance(r, tuple) and r[1] == "ok")
    failed = sum(1 for r in results if isinstance(r, tuple) and r[1] == "failed")
    exc = sum(1 for r in results if isinstance(r, BaseException))
    print(f"  → OK: {ok}  Failed: {failed}  Exceptions: {exc}", flush=True)
    return ok, failed


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _loaded_keys(path: Path) -> set[str]:
    """Return set of keys already present in a JSONL output file."""
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("snippet_id", "")
            ti = obj.get("triplet_index")
            if ti is not None:
                keys.add(f"{sid}::{ti}")
            else:
                keys.add(sid)
    return keys


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _load_verdicts(path: Path) -> list[dict[str, Any]]:
    """Load all verdict records from a JSONL file (skip errors)."""
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def aggregate(run_dir: Path, branches: dict[str, list[JudgeItem]],
              failures_count: int) -> dict[str, Any]:
    """Produce an aggregate report from judgment files."""
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "hard_failures": failures_count,
        "branches": {},
        "global_summary": {},
    }

    total_ok = 0
    total_failed = 0
    global_verdicts = Counter()

    for branch_name, fname in BRANCH_FILES.items():
        path = run_dir / fname
        recs = _load_verdicts(path)
        verdict_counter = Counter()
        ok = 0
        failed = 0
        extra: dict[str, Any] = {}

        for r in recs:
            v = r.get("judge_verdict", "unknown")
            verdict_counter[v] += 1
            if v == "judge_error":
                failed += 1
            else:
                ok += 1

        # Per-branch extras
        if branch_name == "zero_triplet":
            extra["total_items"] = len(branches.get(branch_name, []))
            extra["judged"] = len(recs)
        elif branch_name == "schema_mismatch":
            # Collect proposed extensions
            proposed: Counter = Counter()
            for r in recs:
                pr = r.get("proposed_relation", "")
                if pr:
                    proposed[pr] += 1
            extra["top_proposed_relations"] = proposed.most_common(15)
            extra["total_items"] = len(branches.get(branch_name, []))
            extra["judged"] = len(recs)
        elif branch_name == "schema_invalid_triplet":
            extra["total_items"] = len(branches.get(branch_name, []))
            extra["judged"] = len(recs)
        elif branch_name == "schema_valid_triplet":
            extra["total_items"] = len(branches.get(branch_name, []))
            extra["judged"] = len(recs)
            # evidence_span_exact distribution
            span_counter = Counter()
            for r in recs:
                span_counter[str(r.get("evidence_span_exact", ""))] += 1
            extra["evidence_span_exact_dist"] = dict(span_counter)
            conf_counter = Counter()
            for r in recs:
                conf_counter[str(r.get("confidence_reasonable", ""))] += 1
            extra["confidence_reasonable_dist"] = dict(conf_counter)

        # --- logprob stats (populated when --use-logprobs was enabled) ---
        if any(r.get("verdict_logprob") is not None for r in recs):
            import statistics as _st
            lp_by_verdict: dict[str, list[float]] = {}
            overconfident = 0
            for r in recs:
                vlp = r.get("verdict_logprob")
                if vlp is None:
                    continue
                v = r.get("judge_verdict", "unknown")
                lp_by_verdict.setdefault(v, []).append(vlp)
                jc = r.get("judge_confidence")
                if jc is not None and jc > 0.7 and vlp < -3.0:
                    overconfident += 1
            v_breakdown: dict[str, dict[str, float]] = {}
            for v, vals in lp_by_verdict.items():
                vs = sorted(vals)
                n = len(vs)
                v_breakdown[v] = {
                    "count": n,
                    "mean": round(_st.mean(vs), 3),
                    "median": round(_st.median(vs), 3),
                    "p10": round(vs[max(0, n // 10)], 3),
                    "p90": round(vs[min(n - 1, 9 * n // 10)], 3),
                }
            extra["logprob_stats"] = {
                "verdict_logprob_by_type": v_breakdown,
                "overconfident_count": overconfident,
            }

        report["branches"][branch_name] = {
            "file": fname,
            "verdicts": dict(verdict_counter),
            "ok": ok,
            "failed": failed,
            **extra,
        }
        total_ok += ok
        total_failed += failed
        for k, v in verdict_counter.items():
            global_verdicts[k] += v

    report["global_summary"] = {
        "total_judged_ok": total_ok,
        "total_failed": total_failed,
        "verdict_distribution": dict(global_verdicts),
    }

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r") as fh:
        for _ in fh:
            count += 1
    return count


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", model).strip("-").lower()


def _derive_csv_id(chunks_csv: Path) -> str:
    """Derive dataset id from the CSV path."""
    skip = frozenset({"processed", "chunks", "data", "output", "outputs"})
    parts = list(chunks_csv.resolve().parts)
    for i in range(len(parts) - 1, -1, -1):
        part = parts[i]
        if part.lower() in skip:
            continue
        if part.endswith(".csv"):
            continue
        candidate = re.sub(r"[^a-zA-Z0-9_-]+", "_", part).strip("_").lower()
        if candidate:
            return candidate
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", chunks_csv.stem).strip("_").lower() or "corpus"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_judge_pipeline(
    chunks_csv: Path,
    triplets_dir: Path,
    output_dir: Path,
    *,
    csv_id: str | None = None,
    api_key: str = "",
    base_url: str = DEFAULT_BASE_URL,
    judge_model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_TIMEOUT,
    resume: bool = False,
    use_logprobs: bool = False,
    retry_failures: bool = False,
    schema_file: str | None = None,
) -> int:
    """Main async judge pipeline."""

    # Derive identifiers
    if csv_id is None:
        csv_id = _derive_csv_id(chunks_csv)
    model_slug = _model_slug(judge_model)
    base_run_parent = output_dir / csv_id / model_slug

    # --- resolve run directory ---
    if resume:
        existing = sorted(
            [d for d in base_run_parent.glob("run_*") if d.is_dir()],
            key=lambda p: p.name,
        )
        if existing:
            run_dir = existing[-1]
            print(f"Resuming from existing run: {run_dir}", flush=True)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = base_run_parent / f"run_{ts}"
            print(f"No prior run found; creating new: {run_dir}", flush=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_run_parent / f"run_{ts}"

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}", flush=True)

    # --- retry-failures: strip judge_error entries so resume picks them up ---
    if retry_failures:
        resume = True
        stripped_total = 0
        for fname in BRANCH_FILES.values():
            path = run_dir / fname
            if not path.exists():
                continue
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            kept = []
            stripped = 0
            for line in lines:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    kept.append(line)
                    continue
                if obj.get("judge_verdict") == "judge_error":
                    stripped += 1
                else:
                    kept.append(line)
            path.write_text("\n".join(kept) + "\n" if kept else "",
                            encoding="utf-8")
            stripped_total += stripped
        print(f"  Stripped {stripped_total} judge_error entries; will retry\n", flush=True)

    # --- load data ---
    print("\nLoading data...", flush=True)
    chunks_lookup = load_chunks(chunks_csv)
    print(f"  chunks: {len(chunks_lookup)}", flush=True)

    triplets_path = triplets_dir / "triplets.jsonl"
    failures_path = triplets_dir / "failures.jsonl"
    violations_path = triplets_dir / "schema_violations.jsonl"

    triplet_outputs = load_triplets(triplets_path)
    failures = load_failures(failures_path)
    violations = load_violations(violations_path)
    print(f"  triplets: {len(triplet_outputs)} parsed outputs", flush=True)
    print(f"  failures: {len(failures)} hard failures", flush=True)
    print(f"  violations: {len(violations)} violation records", flush=True)

    # --- classify ---
    branches = classify_items(chunks_lookup, triplet_outputs, violations)
    branch_totals = {k: len(v) for k, v in branches.items()}
    print(f"\nBranch routing:", flush=True)
    for bname, cnt in branch_totals.items():
        print(f"  {bname}: {cnt} items", flush=True)

    # --- resume filtering ---
    total_skipped = 0
    for bname, fname in BRANCH_FILES.items():
        path = run_dir / fname
        if not resume or not path.exists():
            continue
        existing_keys = _loaded_keys(path)
        if existing_keys:
            items = branches.get(bname, [])
            new_items = [it for it in items if it.key not in existing_keys]
            skipped = len(items) - len(new_items)
            branches[bname] = new_items
            print(f"  [{bname}] skipping {skipped} already judged items", flush=True)
            total_skipped += skipped

    if total_skipped:
        print(f"  Total skipped (resume): {total_skipped}", flush=True)

    # --- judge each branch sequentially ---
    total_ok = 0
    total_failed = 0

    for bname, fname in BRANCH_FILES.items():
        items = branches.get(bname, [])
        if not items:
            continue
        output_path = run_dir / fname
        ok, failed = await process_branch(
            bname, items, output_path,
            base_url=base_url, api_key=api_key, model=judge_model,
            concurrency=concurrency, timeout=timeout,
            use_logprobs=use_logprobs,
            schema_path=schema_file,
        )
        total_ok += ok
        total_failed += failed

    # --- aggregate ---
    print(f"\n{'='*60}", flush=True)
    print("Aggregating results...", flush=True)
    report = aggregate(run_dir, branches, len(failures))
    report_path = run_dir / "aggregate_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Report written to {report_path}", flush=True)

    # --- summary ---
    print(f"\n{'='*60}", flush=True)
    print("DONE", flush=True)
    print(f"  Judged OK:  {total_ok}", flush=True)
    print(f"  Failed:     {total_failed}", flush=True)
    for bname, info in report["branches"].items():
        vd = info.get("verdicts", {})
        vd_str = ", ".join(f"{k}:{v}" for k, v in sorted(vd.items()))
        print(f"  [{bname}] {vd_str}", flush=True)

    return 0 if total_failed == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-as-Judge entailment validation for KG triplets"
    )
    p.add_argument("--chunks-csv", type=Path, required=True,
                   help="Path to chunk CSV from build_corpus.py")
    p.add_argument("--triplets-dir", type=Path, required=True,
                   help="Directory containing triplets.jsonl, failures.jsonl, "
                        "schema_violations.jsonl")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Root directory for validation outputs "
                        "(results go to output-dir/csv-id/judge-model/run_ts/)")
    p.add_argument("--csv-id", type=str, default=None,
                   help="Dataset identifier (default: derived from CSV path)")
    p.add_argument("--judge-model", type=str, default=DEFAULT_MODEL,
                   help=f"LLM model to use as judge (default: {DEFAULT_MODEL})")
    p.add_argument("--api-key", type=str,
                   default=os.environ.get("DEEPSEEK_API_KEY", "")
                           or os.environ.get("OPENAI_API_KEY", ""),
                   help="API key (or set DEEPSEEK_API_KEY / OPENAI_API_KEY env var)")
    p.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                   help=f"API base URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help=f"Max concurrent LLM calls (default: {DEFAULT_CONCURRENCY})")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT})")
    p.add_argument("--use-logprobs", action="store_true",
                   help="Request token log-probabilities from the judge model "
                        "to compute objective confidence scores per judgment")
    p.add_argument("--schema-file", type=str, default=None,
                   help="Path to schema JSON config file "
                        "(default: auto-detected schema_config.json)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the most recent run directory for the same "
                        "dataset+model pair")
    p.add_argument("--retry-failures", action="store_true",
                   help="Strip judge_error entries from output files and "
                        "re-judge only those items (use with --resume)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.api_key:
        print("ERROR: No API key provided. Set DEEPSEEK_API_KEY or OPENAI_API_KEY, "
              "or pass --api-key.", file=sys.stderr, flush=True)
        return 1

    return asyncio.run(
        run_judge_pipeline(
            chunks_csv=args.chunks_csv,
            triplets_dir=args.triplets_dir,
            output_dir=args.output_dir,
            csv_id=args.csv_id,
            api_key=args.api_key,
            base_url=args.base_url,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
            timeout=args.timeout,
            resume=args.resume,
            use_logprobs=args.use_logprobs,
            retry_failures=args.retry_failures,
            schema_file=args.schema_file,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
