# Entailment Validation (`src/validation/`)

Grades the quality of extracted triplets by checking whether each one is
entailed by its source text. Three complementary approaches live here:

1. **LLM-as-judge** (`LLM_judge/judge.py`) — a second LLM routes every item
   into one of four branches and assigns a structured verdict.
2. **NLI cross-encoder** (`NLI_model/`) — a fine-tuned
   `cross-encoder/nli-deberta-v3-small` scores entailment/contradiction/neutral
   and can be fine-tuned and evaluated on held-out data.
3. **Two-stage NLI + LLM judge** (`nli_llm_judge.py`) — runs chunk-level and
   span-level NLI, then a Qwen judge resolves the final verdict; disagreements
   are flagged for human review.

The `unconstrained/` subfolder adds LLM-driven canonicalisation of the
open-domain extraction output.

---

# 1. LLM-as-Judge (`LLM_judge/judge.py`)

Uses a second LLM to judge each triplet against its source chunk. Routes every
item into one of four evaluation branches, each with its own criteria and
few-shot prompt, and writes per-branch verdicts plus an aggregate report.

## Architecture

```
triplets.jsonl ──→  classify_items()  ──→  4 branches
failures.jsonl              │
violations.jsonl            │
                            ▼
  ┌──────────┬───────────────┬──────────────────────┬────────────────────┐
  ▼          ▼               ▼                      ▼                    ▼
zero_triplet  schema_mismatch  schema_invalid_triplet  schema_valid_triplet  hard_failures
   │            │                │                      │
   └────────────┴────────────────┴──────────────────────┘
                            │
                            ▼
                  4 × judgments.jsonl  +  aggregate_report.json
```

## The four branches

### 1. Zero-triplet

Judges chunks where the extraction model produced zero triplets.

| Verdict | Meaning |
|---|---|
| `correct_zero` | The chunk truly contains no entity-entity relations |
| `missed_relation` | The model SHOULD have produced triplets but did not |
| `out_of_schema_only` | A meaningful relation exists but no current schema relation can express it |
| `uncertain` | Cannot decide |

### 2. Schema-mismatch

Judges triplets where the model deliberately output `predicate: "SCHEMA_MISMATCH"`.

| Verdict | Meaning |
|---|---|
| `needs_schema_extension` | Text supports a relation not in the schema — proposes a new relation with domain/range |
| `maps_to_existing` | The relation maps to an existing schema relation — the model made an error |
| `unsupported` | The evidence text does not support the claimed relation |
| `malformed` | The extraction is garbled |

This branch produces a ranked list of candidate schema additions (e.g.
`fflo:hasFunction`, `fflo:hasSubcategory`, `fflo:amends`, `fflo:defines`,
`fflo:compliesWith`).

### 3. Schema-invalid

Judges ordinary triplets that failed schema validation.

| Verdict | Meaning |
|---|---|
| `entailed_direction_reversed` | Text supports the relation but subject/object should be swapped |
| `entailed_wrong_types` | Text supports the relation but entity types are wrong |
| `entailed_wrong_relation` | Text supports the relation but the predicate is wrong |
| `not_entailed` | Text does not support the relation at all |
| `uncertain` | Cannot decide |

### 4. Schema-valid

Judges triplets that passed all schema checks — the main entailment branch.

| Verdict | Meaning |
|---|---|
| `entailed` | All parts of the triplet are explicitly stated in the text |
| `partially_entailed` | Plausible but requires one inferential step |
| `not_entailed` | Fabricated or over-extrapolated |
| `uncertain` | Cannot decide |

**Auxiliary checks** (reported for entailed/partially-entailed triplets):

| Check | Values |
|---|---|
| `evidence_span_exact` | `true` / `false` / `partial` |
| `entity_types_correct` | `true` / `false` |
| `direction_correct` | `true` / `false` |
| `confidence_reasonable` | `yes` / `no` / `overconfident` / `underconfident` |

## Logprob confidence scoring

With `--use-logprobs`, the judge requests token-level log-probabilities to
compute an **objective** confidence signal alongside the model's self-reported
`judge_confidence`:

| Field | Meaning |
|---|---|
| `verdict_logprob` | Cumulative log-prob of the tokens forming the verdict value (closer to 0 = more certain) |
| `total_logprob` | Sum of all output token logprobs |
| `avg_logprob` | Per-token average |
| `output_perplexity` | `exp(-avg_logprob)` — lower = more certain |

The report flags **overconfident judgments** (high self-reported confidence but
low token logprobs) for manual audit.

## Schema file

The judge reads entity types, relations, and domain/range directly from the
schema JSON config (`src/extraction/schema_config.json`), so the judge and the
schema checker share one source of truth. Override with `--schema-file` (a JSON
config, not a prompt file).

## Chunk CSV Requirements

The judge re-reads the same chunk CSV used during extraction and only accesses
two columns: `snippet_id` (joins verdicts back to chunks) and `evidence_text`
(the full text shown for entailment verification). It must be the **same** CSV
used during extraction.

## Quick Start

```bash
food_lab/bin/python src/validation/LLM_judge/judge.py \
    --chunks-csv    src/data/FSSAI_docs/processed/chunks.csv \
    --triplets-dir  src/outputs/triplets \
    --output-dir    src/outputs/validation \
    --judge-model   Qwen/Qwen3.5-27B-FP8 \
    --base-url      http://localhost:8030/v1 \
    --api-key       $DEEPSEEK_API_KEY \
    --use-logprobs \
    --concurrency   8
```

### Resume / retry

```bash
# resume from the most recent run directory
... --resume

# strip judge_error entries and re-judge only those
... --retry-failures
```

## Output

```
src/outputs/validation/{csv_id}/{judge_model}/run_YYYYMMDD_HHMMSS/
├── zero_triplet_judgments.jsonl        # one line per zero-triplet chunk
├── schema_mismatch_judgments.jsonl     # one line per SCHEMA_MISMATCH triplet
├── schema_invalid_judgments.jsonl      # one line per schema-invalid triplet
├── schema_valid_judgments.jsonl        # one line per schema-valid triplet
└── aggregate_report.json               # verdict distributions, proposed extensions,
                                        # evidence_span stats, logprob calibration
```

### Judgment JSON (common fields)

```json
{
  "snippet_id": "src_pdf_005_chunk_0015",
  "branch": "schema_valid_triplet",
  "triplet_index": 0,
  "subject": "Paneer",
  "predicate": "fflo:belongsToCategory",
  "object": "UnripenedCheese",
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

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--chunks-csv` | *(required)* | Chunk CSV used for extraction |
| `--triplets-dir` | *(required)* | Directory with `triplets.jsonl`, `failures.jsonl`, `schema_violations.jsonl` |
| `--output-dir` | *(required)* | Root directory for validation outputs |
| `--judge-model` | `deepseek-v4-Pro` | LLM model used as judge |
| `--api-key` | `$DEEPSEEK_API_KEY` | API key |
| `--base-url` | `https://api.deepseek.com` | API base URL |
| `--concurrency` | `8` | Max concurrent LLM calls |
| `--timeout` | `180` | Per-request timeout in seconds |
| `--resume` | `False` | Resume from the most recent run directory |
| `--retry-failures` | `False` | Strip `judge_error` entries and re-judge only those |
| `--use-logprobs` | `False` | Request token log-probabilities for objective confidence scoring |
| `--schema-file` | *(auto)* | Path to schema JSON config (default: auto-detected `schema_config.json`) |
| `--csv-id` | *(auto)* | Dataset identifier (auto-derived from CSV path) |

## Adding your own few-shot prompts

The judge ships with few-shot examples tuned for FSSAI food-safety regulations.
For a different food item or domain, replace the examples in
`src/validation/LLM_judge/judge_prompts.py`. Each branch has a dedicated
prompt-builder function:

| Branch | Function | Prompt location |
|---|---|---|
| zero_triplet | `zero_triplet_system()` | `=== FEW-SHOT EXAMPLES ===` block |
| schema_mismatch | `schema_mismatch_system()` | `=== FEW-SHOT EXAMPLES ===` block |
| schema_invalid | `schema_invalid_system()` | `=== FEW-SHOT EXAMPLES ===` block |
| schema_valid | `schema_valid_system()` | `=== FEW-SHOT EXAMPLES ===` block |

Few-shot examples teach the judge what your domain looks like (e.g. "Paneer →
belongsToCategory → UnripenedCheese"). Without domain-specific examples the
judge may mislabel domain relations as unsupported and produce less calibrated
confidence scores. Pick 2–3 real examples per branch from a manual audit of
your data.

---

# 2. NLI Cross-Encoder (`NLI_model/`)

A `sentence-transformers` cross-encoder classifies each triplet's verbalisation
against its source text as `entailment` / `contradiction` / `neutral`. The
fine-tuned checkpoint lives at `model_checkpoints/run4_deberta_v3_small`.

| Module | Purpose |
|---|---|
| `nli_model.py` | `NLIModel` wrapper (single/multi-GPU, CPU) with a label→verdict map |
| `verbalise.py` | Relation/entity-type → natural-language hypothesis templates |
| `synthetic_ids.py` | Assigns IDs + descriptive labels to structural/synthetic nodes |
| `nli_pipeline.py` | End-to-end inference: convert Excel, or run pretrained/fine-tuned |
| `nli_finetune.py` | Fine-tune the cross-encoder (single/multi-GPU/distributed) |
| `nli_eval.py` | Evaluate one or more models on a labelled CSV |

## Fine-tuning

```bash
food_lab/bin/python src/validation/NLI_model/nli_finetune.py \
    --train-csv    src/validation/Golden_val/nli_train_pairs.csv \
    --base-model   cross-encoder/nli-deberta-v3-small \
    --output-dir   model_checkpoints/run4_deberta_v3_small \
    --epochs       5 --batch-size 16 --device cuda:0
```

Splits by unique `snippet_id` (stratified by predicate, or
`--split-mode predicate_disjoint` for a cross-relation generalisation test) and
filters cross-premise leakage.

## Inference

```bash
# pretrained
food_lab/bin/python src/validation/NLI_model/nli_pipeline.py \
    --mode pretrained \
    --nli-model cross-encoder/nli-deberta-v3-small \
    --triplets-path src/outputs/triplets/triplets.csv \
    --chunks-csv src/data/FSSAI_docs/processed/chunks.csv \
    --output-dir src/outputs/nli_validation \
    --device cuda:0

# fine-tuned
food_lab/bin/python src/validation/NLI_model/nli_pipeline.py \
    --mode finetuned \
    --nli-model model_checkpoints/run4_deberta_v3_small \
    --triplets-path src/outputs/triplets/triplets.csv \
    --chunks-csv src/data/FSSAI_docs/processed/chunks.csv \
    --output-dir src/outputs/nli_validation \
    --device cuda:0
```

## Evaluation

```bash
python3 src/validation/NLI_model/nli_eval.py \
    --models cross-encoder/nli-deberta-v3-small,model_checkpoints/run4_deberta_v3_small \
    --labels pretrained,finetuned \
    --eval-csv src/validation/Golden_val/nli_train_pairs.csv \
    --device cuda:0
```

Writes per-example predictions, per-model metrics, a comparison table, and
(≥2 models) a `disagreements.csv`.

---

# 3. Two-Stage NLI + LLM Judge (`nli_llm_judge.py`)

For every triplet in the **unconstrained** extraction output:

1. Chunk-level NLI (premise = full `evidence_text`)
2. Span-level NLI (premise = `evidence_span`)
3. Local Qwen LLM judge with both NLI scores
4. Agreement status: `agreed_all` / `agreed_partial` / `contested` — contested
   rows are flagged for human review, with the LLM judge taking precedence in
   the final verdict.

```bash
python3 src/validation/nli_llm_judge.py \
    --triplets-dir src/outputs/unconstrained \
    --chunks-csv   src/data/FSSAI_docs/processed/chunks.csv \
    --nli-model    model_checkpoints/run4_deberta_v3_small \
    --base-url     http://localhost:8030/v1 \
    --model        Qwen/Qwen3.5-27B-FP8 \
    --api-key      aashray-fflo-local \
    --output-dir   src/outputs/unconstrained \
    --device       cuda:0
```

Outputs: `nli_llm_judge_results.csv` (full), `nli_llm_judge_contested.csv`
(human review), `nli_llm_judge_entailed.csv` (accepted), and a summary JSON.
Optional `--canonicalise` also runs LLM canonicalisation on the entailed+agreed
triplets.

---

# 4. Unconstrained Canonicalisation (`unconstrained/`)

Supports the open-domain extraction workflow:

| Module | Purpose |
|---|---|
| `llm_canonicalise.py` | Grounded, LLM-driven canonicalisation of free-form relation names (verify-then-name, with antonym guard) |
| `apply_canonical.py` | Apply a `canonical_map.json` to resolve predicate/subject_type/object_type in a triplet CSV |

```bash
python3 src/validation/unconstrained/llm_canonicalise.py \
    --report-json  src/outputs/unconstrained/unconstrained_report.json \
    --triplets-csv src/outputs/unconstrained/nli_llm_judge_entailed.csv \
    --base-url     http://localhost:8030/v1 \
    --model        Qwen/Qwen3.5-27B-FP8 \
    --output       canonical_map.json

python3 src/validation/unconstrained/apply_canonical.py \
    --input  nli_llm_judge_entailed.csv \
    --map    canonical_map.json \
    --output entailed_canonicalised.csv
```

## Requirements

```bash
pip install httpx
# NLI components additionally:
pip install sentence-transformers
# unconstrained/llm_canonicalise.py additionally:
pip install pandas
```
