# Triplet Extraction Pipeline

Extracts structured knowledge-graph triplets from FSSAI food safety regulation
chunks using a schema-guided LLM.  Supports DeepSeek (default) and local
HuggingFace models including SciPhi/Triplex.

## Architecture

```
chunks.csv  ──→  build_context()  ──→  LLM call  ──→  validate_triplet()
  (1586 rows)     (neighbor chunks)    (API / local)   (schema checks)
                                              │
                    ┌─────────────────────────┘
                    ▼
          triplets.jsonl     failures.jsonl     schema_violations.jsonl
          (per-chunk JSON)   (API/parse errors)  (domain/range/type errors)
                    │
                    ▼
              triplets.csv
              (flattened, one row per individual triplet)
```

## Schema

The extraction model is given 92 entity types and 67 relations drawn from the
FFLO (Food Fraud Lifecycle Ontology) v7, spanning five lifecycle stages:

| Stage | Example types | Example relations | v7 additions |
|---|---|---|---|
| ACT | Adulterant, FoodAdditive, AdditiveFunction, FraudType | hasAdulterant, hasFunction, hasIngredient, hasFraudType | Microorganism, ProcessingAid, ObligationType (8 subtypes), hasObligation, hasScientificName |
| SPREAD | SupplyChainStep, SpreadEvent | propagatesTo, occursAt | — |
| DETECTION | IncidentFinding, DetectionMethod, Sample | identifies, hasNumericValue, foundIn | SamplingPlan, governsSampling |
| REGULATION | PermissibleLimit, FoodStandard, TableReference | hasValue, appliesTo, defines, compliesWith, amends, hasSubcategory, hasDefinition | RegulatoryOfficer, RegulatoryActionType (5 subtypes), isEmpoweredTo, actsOnBehalfOf |
| HEALTH | HealthEffect, VulnerablePopulation | causesEffect, affectsPopulation | — |

The schema is defined in `src/schema_iterations/schema_v3.md` (canonical spec).
The runtime validation rules are loaded from **`src/extraction/schema_config.json`**
(92 entity types, 67 relations, full domain/range constraints).  This JSON
config is read by the `schema_validator` module at startup — `extract_triplets.py`
no longer contains hardcoded entity/relation lists.

The system prompt is what the LLM sees.  The v7 prompt is at
`prompts/triplet_extraction_v2.txt`.  Pass it via `--prompt-file`:

```bash
food_lab/bin/python src/extraction/extract_triplets.py \
    --prompt-file prompts/triplet_extraction_v2.txt
    --schema-file src/extraction/schema_config.json
```

If you iterate on the schema, point at a custom config:

```bash
food_lab/bin/python src/extraction/extract_triplets.py \
    --prompt-file prompts/triplet_extraction_v3.txt
    --schema-file schema_iterations/schema_v4.json
```

If you update the schema, edit exactly three files:
1. `src/schema_iterations/schema_v3.md` (canonical — add entity types, relations, domain/range)
2. `src/extraction/schema_config.json` (runtime config — add entries to `entity_types`, `relations`, and `domain_range`)
3. `prompts/triplet_extraction_v2.txt` (system prompt — list every type and relation the LLM can output)

The `schema_validator` module handles the rest automatically — Triplex
prefixed-name lists, canonical→prefixed mappings, and startup assertions are all
derived from `schema_config.json`.  No more hardcoded lists to keep in sync.

### SCHEMA_MISMATCH

When the text contains a meaningful relation not expressible with the current
schema, the model deliberately outputs `predicate: "SCHEMA_MISMATCH"` with a
`mismatch_note`.  This is intentional — it flags gaps in the ontology for
later review.

## Chunk CSV Format

The pipeline does not process raw documents — it expects **pre-chunked text** in
a CSV.  This is because:

1. **Context windows are finite** — sending a 50-page PDF to an LLM in one
   request would overflow its token budget and produce worse results.  Chunking
   breaks each document into ~200-word paragraphs so the model sees focused,
   manageable text.

2. **Neighbor context helps** — the pipeline reads the chunk BEFORE and AFTER
   the target chunk (same `source_id`) so the model can resolve pronouns,
   cross-sentence references, and hanging clauses.  `chunk_index` tells the
   pipeline which order chunks appear in, and `source_id` groups chunks from
   the same document.

3. **Triplets must be traceable** — `snippet_id` uniquely identifies each
   chunk, so every extracted triplet can be traced back to its exact source
   paragraph for downstream validation.

### Required columns

| Column | Purpose |
|---|---|
| `snippet_id` | Unique ID per chunk.  Used as the primary key throughout the pipeline — triplets, failures, violations, and judgments all reference this ID. |
| `evidence_text` | The text content the model actually reads.  This is the only column sent to the LLM. |
| `source_id` | Groups chunks from the same document.  The pipeline looks backward/forward within the same `source_id` to build context. |
| `chunk_index` | Ordinal position within a source.  Chunks with `chunk_index=4`, `5`, `6` in the same `source_id` are neighbors. |

### Optional metadata columns

| Column | Purpose |
|---|---|
| `source_file` | Human-readable path.  Written to output files so you know which regulation each chunk came from. |
| `source_type` | Type tag (`pdf`, `markdown`).  Informational only in the flattened CSV. |

### Producing the chunk CSV

The pipeline expects a CSV matching this schema.  If you're starting from raw
documents (PDFs, markdown files, crawled text), run the upstream chunking
pipeline first:

```bash
food_lab/bin/python src/preprocessing_cleaning/build_corpus.py \
    --input-dir   src/data/FSSAI_docs/processed/markdown \
    --output-csv  src/data/FSSAI_docs/processed/chunks.csv
```

`build_corpus.py` handles:
- Reading raw text files from a directory
- Splitting by paragraphs
- Chunking paragraphs into ~200-word units
- Deduplication and metadata noise removal
- Assigning `snippet_id`, `source_id`, `chunk_index`, `source_file`, `source_type`

### If your CSV has different column names

Extraction will fail with a `KeyError`.  Either:
- Rename your columns to match this schema (easiest), or
- Update `load_chunks()`, `build_context()`, and `_flatten_to_csv()` in
  `src/extraction/extract_triplets.py` to reference your column names (cleaner
  if you have a large existing corpus you don't want to reformat).

## Quick Start

### Remote API (DeepSeek)

```bash
food_lab/bin/python src/extraction/extract_triplets.py \
    --chunks-csv  src/data/FSSAI_docs/processed/chunks.csv \
    --prompt-file prompts/triplet_extraction.txt \
    --output-dir  src/outputs/triplets \
    --model       deepseek-v4-flash \
    --api-key     $DEEPSEEK_API_KEY \
    --concurrency 10 \
    --resume
```

### Local HuggingFace model

```bash
# Standard prompt format (system + user messages via chat template)
food_lab/bin/python src/extraction/extract_triplets.py \
    --chunks-csv     src/data/FSSAI_docs/processed/chunks.csv \
    --prompt-file    prompts/triplet_extraction.txt \
    --output-dir     src/outputs/triplets \
    --backend        hf_transformers \
    --model          meta-llama/Meta-Llama-3.1-8B-Instruct \
    --hf-device      mps \
    --hf-dtype       bfloat16 \
    --hf-quantize    4bit \
    --resume

# Triplex native format (entity types + predicates as JSON arrays)
food_lab/bin/python src/extraction/extract_triplets.py \
    --chunks-csv           src/data/FSSAI_docs/processed/chunks.csv \
    --prompt-file          prompts/triplet_extraction.txt \
    --output-dir           src/outputs/triplets \
    --backend              hf_transformers \
    --hf-prompt-format     triplex \
    --hf-trust-remote-code \
    --model                SciPhi/Triplex \
    --hf-device            cpu \
    --hf-dtype             bfloat16 \
    --hf-quantize          4bit \
    --resume
```

## Output

All output is written to `{output_dir}/{csv_id}/{model_slug}/` (or prefix
`hf__{model_slug}/` for the HuggingFace backend).  Using a flat output
directory (recommended for single-dataset workflows) writes directly to
`{output_dir}/`.

| File | Description |
|---|---|
| `triplets.jsonl` | One JSON line per chunk.  Even chunks with zero triplets get an entry (`triplets: []`) |
| `failures.jsonl` | Chunks where the LLM call failed (API errors, parse errors, length truncation) |
| `schema_violations.jsonl` | Individual triplets that failed schema checks (unknown type, wrong domain/range, bad confidence) |
| `triplets.csv` | Flattened view — one row per individual triplet, with all snippet metadata |

### Triplet JSON schema

```json
{
  "snippet_id": "src_pdf_005_chunk_0015",
  "triplets": [
    {
      "subject": "Paneer",
      "subject_type": "fkg:Food",
      "subject_id": "",
      "predicate": "fflo:belongsToCategory",
      "object": "UnripenedCheese",
      "object_type": "fflo:FoodCategory",
      "object_id": "",
      "confidence": 0.9,
      "evidence_span": "paneer (milk protein coagulated by the addition of citric acid",
      "event_id": "",
      "event_class": "",
      "polarity": "affirmed",
      "mismatch_note": ""
    }
  ],
  "notes": ""
}
```

| v7 field | Type | Description |
|---|---|---|
| `subject_id` / `object_id` | string | Synthetic ID for structural nodes not appearing verbatim in the text. Omit (`""`) for named entities that do appear. |
| `event_id` | string | Shared across all triplets from the same real-world event (format `ev_<topic>_<NNNN>`). Omit for standalone triplets. |
| `event_class` | string | The FFLO class this `event_id` instantiates. Omit if `event_id` is omitted. |
| `polarity` | string | `"affirmed"`, `"negated"`, or `"hedged"` (conditional/modal/disjunctive). |
| `mismatch_note` | string | Filled only when `predicate` is `"UNMAPPED"` — describes the natural-language relation that couldn't be expressed. |

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--chunks-csv` | *(required)* | Chunk CSV from `build_corpus.py` |
| `--prompt-file` | *(required)* | System prompt template (contains the schema) |
| `--schema-file` | `src/extraction/schema_config.json` | Path to schema JSON config (entity types, relations, domain/range constraints) |
| `--output-dir` | *(required)* | Root output directory |
| `--model` | `deepseek-v4-flash` | Model name, slugified for directory naming |
| `--api-key` | `$DEEPSEEK_API_KEY` | API key (not needed for `--backend hf_transformers`) |
| `--base-url` | `https://api.deepseek.com` | API base URL |
| `--concurrency` | `10` | Max parallel requests (forced to `1` for HF backend) |
| `--timeout` | `120` | Per-request timeout in seconds |
| `--resume` | `False` | Skip snippet_ids already in triplets.jsonl |
| `--retry-failures` | `False` | Only process chunks listed in failures.jsonl |
| `--skip-until` | *(none)* | Jump forward to a specific snippet_id |

### HuggingFace-specific flags

| Flag | Default | Description |
|---|---|---|
| `--backend` | `openai_compat` | `openai_compat` or `hf_transformers` |
| `--hf-device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--hf-dtype` | `auto` | `auto`, `float16`, `bfloat16`, or `float32` |
| `--hf-quantize` | `none` | `none`, `4bit`, or `8bit` (requires `bitsandbytes`) |
| `--hf-prompt-format` | `standard` | `standard` (chat template) or `triplex` (Triplex native format) |
| `--hf-trust-remote-code` | `False` | Required for Triplex and other custom-code models |
| `--hf-cache-dir` | *(none)* | HuggingFace model cache directory |
| `--hf-max-new-tokens` | `4096` | Max tokens for HF generation |
| `--hf-temperature` | `0.0` | Generation temperature |

## Retry workflow

When chunks fail due to length truncation or transient API errors:

```bash
# 1. Increase the token budget in the prompt and the API call
#    (edit max_tokens in extract_triplets.py and token budget rules in the prompt)

# 2. Retry only the failed chunks
food_lab/bin/python src/extraction/extract_triplets.py \
    --chunks-csv  src/data/FSSAI_docs/processed/chunks.csv \
    --prompt-file prompts/triplet_extraction.txt \
    --output-dir  src/outputs/triplets \
    --model       deepseek-v4-flash \
    --retry-failures \
    --concurrency 10

# 3. Merge retried results into the flat output directory
food_lab/bin/python src/validation/merge_retry.py \
    --flat-dir   src/outputs/triplets \
    --sub-dir    src/outputs/triplets/fssai_docs/deepseek-v4-flash \
    --chunks-csv src/data/FSSAI_docs/processed/chunks.csv

# 4. Delete the subdir and repeat if needed
rm -rf src/outputs/triplets/fssai_docs
```

## FSSAI Results (1,586 chunks — v6 schema)

| Metric | Count |
|---|---|
| Total chunks | 1,586 |
| Productive chunks (≥1 triplet) | 988 |
| Zero-triplet chunks | 598 |
| Total individual triplets | 7,306 |
| Schema violations | 2,100 |
| Extraction failures (after retries) | 0 |

*These results used the v6 schema (`schema_v2.md` + `triplet_extraction.txt`).
v7 results with the new schema config are pending.*

## Requirements

```bash
pip install httpx
# Optional — for HuggingFace backend:
pip install torch transformers
# Optional — for quantization:
pip install bitsandbytes
```
