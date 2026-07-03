# Entailment Validation Pipeline (LLM-as-Judge)

Grades the quality of extracted knowledge-graph triplets by using a second LLM
to judge whether each triplet is entailed by its source text.  Routes every item
into one of four evaluation branches, each with its own judging criteria and
few-shot prompt, and produces per-branch verdicts plus an aggregate report.

## Architecture

```
triplets.jsonl ──→  classify_items()  ──→  4 branches
failures.jsonl              │
violations.jsonl            │
                            ▼
  ┌──────────┬───────────────┬──────────────────┬────────────────┐
  │          │               │                  │                │
  ▼          ▼               ▼                  ▼                ▼
zero_triplet  schema_mismatch  schema_invalid    schema_valid     hard_failures
   │            │                │                 │               
   ▼            ▼                ▼                 ▼
LLM judge    LLM judge        LLM judge         LLM judge
   │            │                │                 │
   └────────────┴────────────────┴─────────────────┘
                            │
                            ▼
                  4 × judgments.jsonl  +  aggregate_report.json
```

Each branch has its own system prompt with:
- Condensed FFLO schema (entity types + relations with domain/range)
- Branch-specific judging criteria
- 2-3 few-shot examples
- Structured JSON output format enforced via `response_format: json_object`

## The four branches

### 1. Zero-triplet

Judges chunks where the extraction model produced zero triplets.

| Verdict | Meaning |
|---|---|
| `correct_zero` | The chunk truly contains no entity-entity relations |
| `missed_relation` | The extraction model SHOULD have produced triplets but did not |
| `out_of_schema_only` | A meaningful relation exists but no current schema relation can express it |
| `uncertain` | Cannot decide |

**Why this matters**: Tells you whether 598 zero-triplet chunks are acceptable
(84% were — mostly category-code lists and version metadata) or whether the
extraction model is missing content (59 chunks had missed relations).

### 2. Schema-mismatch

Judges triplets where the model deliberately output `predicate: "SCHEMA_MISMATCH"`
with a `mismatch_note`.

| Verdict | Meaning |
|---|---|
| `needs_schema_extension` | The text genuinely supports a relation not in the schema — proposes a new relation with domain/range |
| `maps_to_existing` | The relation actually maps to an existing schema relation — the model made an error |
| `unsupported` | The evidence text does not support the claimed relation |
| `malformed` | The extraction is garbled |

**Why this matters**: Produces a ranked list of candidate schema additions.
565 genuine new relations were identified across the FSSAI dataset, including
`fflo:hasFunction`, `fflo:hasSubCategory`, `fflo:amends`, `fflo:defines`,
and `fflo:compliesWith`.

### 3. Schema-invalid

Judges ordinary triplets that failed schema validation (wrong domain/range,
unknown relation, or unknown entity type).

| Verdict | Meaning |
|---|---|
| `entailed_direction_reversed` | Text supports the relation but subject/object should be swapped |
| `entailed_wrong_types` | Text supports the relation but entity types are wrong |
| `entailed_wrong_relation` | Text supports the relation but the predicate is wrong |
| `not_entailed` | Text does not support the relation at all |
| `uncertain` | Cannot decide |

**Why this matters**: Separates fixable schema errors from genuine fabrications.
Of 738 schema-invalid triplets, 589 are semantically correct but structurally
wrong (reversible via prompt improvement).  145 are genuinely not supported.

### 4. Schema-valid

Judges triplets that passed ALL schema checks — the main entailment branch.

| Verdict | Meaning |
|---|---|
| `entailed` | All parts of the triplet are explicitly stated in the text |
| `partially_entailed` | Plausible but requires one inferential step beyond what is stated |
| `not_entailed` | Fabricated or over-extrapolated from the text |
| `uncertain` | Cannot decide |

**Auxiliary checks** (reported for every entailed/partially-entailed triplet):

| Check | Values |
|---|---|
| `evidence_span_exact` | `true` / `false` / `partial` — is the cited span verbatim in the text? |
| `entity_types_correct` | `true` / `false` — are subject_type and object_type appropriate? |
| `direction_correct` | `true` / `false` — does the relation arrow point the right way? |
| `confidence_reasonable` | `yes` / `no` / `overconfident` / `underconfident` |

## Logprob confidence scoring

When run with `--use-logprobs`, the judge requests token-level log-probabilities
from the DeepSeek API.  These provide an **objective** confidence signal that
complements the model's self-reported `judge_confidence`:

| Field | Meaning |
|---|---|
| `verdict_logprob` | Cumulative log-probability of the tokens forming the verdict value (e.g., -0.45 for "entailed") — closer to 0 = more certain |
| `total_logprob` | Sum of all output token logprobs |
| `avg_logprob` | Per-token average |
| `output_perplexity` | `exp(-avg_logprob)` — lower = more certain |

The aggregate report flags **overconfident judgments**: cases where the model
self-reports high confidence but the token logprobs show it was actually
uncertain.  These are high-priority for manual audit.

## Chunk CSV Requirements

The judge re-reads the same chunk CSV used during extraction.  It only accesses
two columns:

| Column | Why the judge needs it |
|---|---|
| `snippet_id` | Joins judgment records back to specific chunks.  Every verdict references this ID so you can trace a finding back to the exact paragraph. |
| `evidence_text` | The full text shown to the judge for entailment verification.  The judge sees the entire chunk (not just the extracted `evidence_span`) to determine whether the triplet is genuinely supported. |

The remaining columns (`source_id`, `chunk_index`, `source_file`, `source_type`)
are ignored by the judge — it only needs to retrieve text by `snippet_id`.

**Important**: The chunk CSV must be the same one used during extraction.

## Schema file

The judge reads entity types and relations from an extraction prompt file to
provide schema context in its system prompts.  By default it uses
`prompts/triplet_extraction.txt`.  If you've moved or renamed the schema file,
pass the new path explicitly:

```bash
food_lab/bin/python src/validation/judge.py \
    --chunks-csv    src/data/FSSAI_docs/processed/chunks.csv \
    --triplets-dir  src/outputs/triplets \
    --output-dir    src/outputs/validation \
    --schema-file   schema_iterations/v2.txt
```

This should match the `--prompt-file` you used during extraction.  The schema is
cached per path, so the file is only read once even when thousands of judgments
are produced.
Using a different CSV (or one with renamed columns) will produce empty
`evidence_text` for every judgment, rendering all verdicts meaningless — the
judge would have no text to verify entailment against.

The judge is dataset-agnostic: point `--triplets-dir` and `--chunks-csv` at any
food item's extraction results and it works without changes.  It auto-derives a
`csv_id` from the chunks CSV path to namespace its output directory.

## Quick Start

```bash
food_lab/bin/python src/validation/judge.py \
    --chunks-csv    src/data/FSSAI_docs/processed/chunks.csv \
    --triplets-dir  src/outputs/triplets \
    --output-dir    src/outputs/validation \
    --judge-model   deepseek-v4-pro \
    --api-key       $DEEPSEEK_API_KEY \
    --use-logprobs \
    --concurrency   8
```

### Resume after interruption

```bash
food_lab/bin/python src/validation/judge.py \
    --chunks-csv    src/data/FSSAI_docs/processed/chunks.csv \
    --triplets-dir  src/outputs/triplets \
    --output-dir    src/outputs/validation \
    --judge-model   deepseek-v4-pro \
    --api-key       $DEEPSEEK_API_KEY \
    --use-logprobs \
    --resume
```

### Retry failed judgments only

```bash
food_lab/bin/python src/validation/judge.py \
    --chunks-csv    src/data/FSSAI_docs/processed/chunks.csv \
    --triplets-dir  src/outputs/triplets \
    --output-dir    src/outputs/validation \
    --judge-model   deepseek-v4-pro \
    --api-key       $DEEPSEEK_API_KEY \
    --retry-failures
```

`--retry-failures` strips all `judge_error` entries from the most recent run
directory, then resumes — only the previously-failed items get re-judged.

## Output

```
src/outputs/validation/fssai_docs/deepseek-v4-pro/run_YYYYMMDD_HHMMSS/
├── zero_triplet_judgments.jsonl        # One line per zero-triplet chunk
├── schema_mismatch_judgments.jsonl     # One line per SCHEMA_MISMATCH triplet
├── schema_invalid_judgments.jsonl      # One line per schema-invalid triplet
├── schema_valid_judgments.jsonl        # One line per schema-valid triplet
└── aggregate_report.json               # Summary with verdict distributions,
                                        # top proposed schema extensions,
                                        # evidence_span stats,
                                        # logprob calibration
```

### Judgment JSON format (common fields)

```json
{
  "snippet_id": "src_pdf_005_chunk_0015",
  "branch": "schema_valid_triplet",
  "triplet_index": 0,
  "subject": "Paneer",
  "predicate": "fflo:belongsToCategory",
  "object": "UnripenedCheese",
  "evidence_span": "paneer (milk protein coagulated by the addition of citric acid",
  "judge_verdict": "entailed",
  "judge_confidence": 0.95,
  "judge_rationale": "Text lists paneer under unripened cheese",
  "evidence_span_exact": true,
  "entity_types_correct": true,
  "direction_correct": true,
  "confidence_reasonable": "yes",
  "verdict_logprob": -0.12,
  "total_logprob": -18.4,
  "avg_logprob": -0.09,
  "output_perplexity": 1.09,
  "timestamp": "2026-06-30T21:48:15Z"
}
```

### Schema-mismatch example (with proposed extension)

```json
{
  "branch": "schema_mismatch",
  "subject": "Amendment Rules, 2017",
  "predicate": "SCHEMA_MISMATCH",
  "object": "Food Safety Rules, 2011",
  "judge_verdict": "needs_schema_extension",
  "proposed_relation": "fflo:amends",
  "proposed_domain": "fflo:RegulatoryDocument",
  "proposed_range": "fflo:RegulatoryDocument"
}
```

### Aggregate report structure

```json
{
  "hard_failures": 0,
  "branches": {
    "zero_triplet": {
      "verdicts": {"correct_zero": 504, "missed_relation": 59, ...},
      "ok": 597, "failed": 1
    },
    "schema_mismatch": {
      "verdicts": {"needs_schema_extension": 565, "unsupported": 46, ...},
      "top_proposed_relations": [
        ["fflo:hasFunction", 58],
        ["fflo:hasSubCategory", 42],
        ["fflo:amends", 34],
        ...
      ]
    },
    "schema_valid_triplet": {
      "verdicts": {"entailed": 1354, "not_entailed": 399, "partially_entailed": 488},
      "evidence_span_exact_dist": {"True": 1500, "False": 200, "partial": 546},
      "confidence_reasonable_dist": {"yes": 1600, "overconfident": 400, "underconfident": 246}
    }
  },
  "global_summary": {
    "total_judged_ok": 4256,
    "total_failed": 11,
    "verdict_distribution": {...}
  }
}
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--chunks-csv` | *(required)* | Chunk CSV used for extraction |
| `--triplets-dir` | *(required)* | Directory with `triplets.jsonl`, `failures.jsonl`, `schema_violations.jsonl` |
| `--output-dir` | *(required)* | Root directory for validation outputs |
| `--judge-model` | `deepseek-chat` | LLM model to use as judge |
| `--api-key` | `$DEEPSEEK_API_KEY` | API key |
| `--base-url` | `https://api.deepseek.com` | API base URL |
| `--concurrency` | `8` | Max concurrent LLM calls |
| `--timeout` | `180` | Per-request timeout in seconds |
| `--resume` | `False` | Resume from the most recent run directory |
| `--retry-failures` | `False` | Strip `judge_error` entries and re-judge only those |
| `--use-logprobs` | `False` | Request token log-probabilities for objective confidence scoring |
| `--schema-file` | `prompts/triplet_extraction.txt` | Path to the extraction prompt file containing entity types and relations |
| `--csv-id` | *(auto)* | Dataset identifier (auto-derived from CSV path) |

## FSSAI Results (4,267 judgments)

| Branch | Items | Key findings |
|---|---|---|
| zero_triplet | 598 | 504 correct (84%), 59 missed |
| schema_mismatch | 685 | 565 needs extension, 53 maps to existing, 46 unsupported |
| schema_invalid | 738 | 280 wrong relation, 187 wrong types, 122 direction reversed, 145 not entailed |
| schema_valid | 2,246 | 1,354 entailed (60%), 488 partially (22%), 399 not entailed (18%) |

**Failure rate**: 11 API errors out of 4,267 judgments (0.3%).

## Adding your own few-shot prompts

The judge ships with few-shot examples tuned for FSSAI food safety regulations.
When running on a **different food item** or domain, you should replace the
few-shot examples in `src/validation/judge_prompts.py` with examples from your
own data.  Each branch has its own set of examples:

| Branch | Prompt variable | Location in `judge_prompts.py` |
|---|---|---|
| zero_triplet | `ZERO_TRIPLET_SYSTEM` | Look for `=== FEW-SHOT EXAMPLES ===` block |
| schema_mismatch | `_SCHEMA_MISMATCH_SYSTEM` | Look for `=== FEW-SHOT EXAMPLES ===` block |
| schema_invalid | `_SCHEMA_INVALID_SYSTEM` | Look for `=== FEW-SHOT EXAMPLES ===` block |
| schema_valid | `_SCHEMA_VALID_SYSTEM` | Look for `=== FEW-SHOT EXAMPLES ===` block |

### Why this matters

Few-shot examples teach the judge what your domain looks like.  An example
showing "Paneer → belongsToCategory → UnripenedCheese" helps the judge
understand FSSAI taxonomy.  If you switch to a different food item (e.g., milk
adulteration, spice contamination), your examples should reflect the entity
types, relations, and text patterns specific to that domain.

Without domain-specific examples, the judge may:
- Mislabel domain-specific relations as unsupported
- Fail to recognize legitimate domain entities
- Produce less calibrated confidence scores

### How to add examples

Each few-shot block follows this format:

```
--- EXAMPLE N ---
CHUNK: <short excerpt from your actual data>
TRIPLET: <an actual triplet from your extraction>
VERDICT: <what the correct verdict should be>
CONFIDENCE: <how certain you are>
RATIONALE: <why this verdict is correct>
```

Pick 2-3 real examples per branch from a manual audit of your data.  Each
example should be concise (short chunk text, 1-2 triplet fields) and represent
the verdict you want to teach.

After updating the prompts, re-run the judge — the new examples will be used
for every judgment in that run.

## Requirements

```bash
pip install httpx
```
