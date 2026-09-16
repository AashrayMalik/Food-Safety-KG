# Triplet Extraction Pipeline

Extracts structured knowledge-graph triplets from FSSAI food-safety regulation
chunks using a schema-guided LLM. Two extraction strategies are available:

- **Constrained** (`extract_triplets.py`) — the model must map every relation
  to the FFLO schema (or emit `SCHEMA_MISMATCH`).
- **Unconstrained** (`unconstrained_pipeline.py`) — the model freely names
  entity types and relations, for schema gap discovery.

Extraction runs against any OpenAI-compatible `/chat/completions` endpoint
(DeepSeek cloud, or a local vLLM/TGI server — the project currently serves
**Qwen3.5-27B-FP8** via vLLM), or a local HuggingFace model (including
SciPhi/Triplex).

## Architecture

```
chunks.csv  ──→  build_context()  ──→  LLM call  ──→  validate_triplet()
  (per-document)    (neighbor chunks)   (API / local)   (schema checks)
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

The extraction model is given the FFLO (Food Fraud Lifecycle Ontology) v7
schema — 87 entity types (plus `xsd:string` / `xsd:decimal` datatypes) and 72
relations across five lifecycle stages:

| Stage | Example types | Example relations |
|---|---|---|
| ACT | Adulterant, FoodAdditive, AdditiveFunction, FraudType, ObligationType | hasAdulterant, hasFunction, hasIngredient, hasFraudType, hasObligation |
| SPREAD | SupplyChainStep, SpreadEvent, TransformationEvent | propagatesTo, occursAt, carriedBy |
| DETECTION | IncidentFinding, DetectionMethod, Sample, SamplingPlan | identifies, hasNumericValue, foundIn, governsSampling |
| REGULATION | PermissibleLimit, FoodStandard, TableReference, RegulatoryActionType | hasValue, appliesTo, defines, compliesWith, amends, hasSubcategory, hasDefinition, isEmpoweredTo |
| HEALTH | HealthEffect, VulnerablePopulation | causesEffect, affectsPopulation |

The runtime source of truth is **`src/extraction/schema_config.json`**
(entity types, relations, domain/range, and `subclass_of` hierarchy). The
`schema_validator` module loads this at startup and derives the canonical
names, Triplex prefixed lists, and startup assertions — there are no hardcoded
entity/relation lists in `extract_triplets.py`.

The human-readable spec lives in `src/schema_iterations/schema_v3.md`, and the
system prompt the LLM sees is `prompts/triplet_extraction_v2.txt` (pass it via
`--prompt-file`).

```bash
food_lab/bin/python src/extraction/extract_triplets.py \
    --prompt-file prompts/triplet_extraction_v2.txt \
    --schema-file src/extraction/schema_config.json
```

If you update the schema, edit exactly three files:

1. `src/schema_iterations/schema_v3.md` (canonical spec)
2. `src/extraction/schema_config.json` (runtime config — `entity_types`,
   `relations`, `domain_range`, `subclass_of`)
3. `prompts/triplet_extraction_v2.txt` (what the LLM sees)

`schema_validator` handles the rest automatically.

### SCHEMA_MISMATCH

When the text contains a meaningful relation not expressible with the current
schema, the model deliberately outputs `predicate: "SCHEMA_MISMATCH"` with a
`mismatch_note`. This is intentional — it flags ontology gaps for later review
(the judge's schema-mismatch branch then proposes extensions).

## Chunk CSV Format

The pipeline expects **pre-chunked text** in a CSV (not raw documents):

1. **Context windows are finite** — chunking breaks documents into ~200-word
   paragraphs the model can read in one request.
2. **Neighbor context helps** — the pipeline reads the chunk before and after
   the target (same `source_id`) to resolve pronouns and cross-sentence
   references (`chunk_index` orders them).
3. **Triplets must be traceable** — `snippet_id` uniquely identifies each
   chunk, so every triplet maps back to its exact source paragraph.

### Required columns

| Column | Purpose |
|---|---|
| `snippet_id` | Unique ID per chunk — the primary key referenced by triplets, failures, violations, and judgments. |
| `evidence_text` | The text the model reads (the only column sent to the LLM). |
| `source_id` | Groups chunks from the same document (used to build neighbor context). |
| `chunk_index` | Ordinal position within a source. |

### Optional metadata columns

| Column | Purpose |
|---|---|
| `source_file` | Human-readable path, written to outputs for traceability. |
| `source_type` | Type tag (`pdf`, `markdown`); informational only in the flattened CSV. |

### Producing the chunk CSV

`build_corpus.py` (in `src/preprocessing_cleaning/`) converts PDFs and scraped
text into the chunk CSV:

```bash
food_lab/bin/python src/preprocessing_cleaning/build_corpus.py \
    --pdf-dir    src/data/FSSAI_docs/pdfs \
    --output-dir src/data/FSSAI_docs/processed
```

This writes `{output_dir}/chunks.csv` (plus `aggregate.md` and per-stage
manifests), handling PDF→markdown, cleaning, paragraph splitting, chunking, and
`source_id`/`chunk_index` assignment.

### If your CSV has different column names

Extraction fails with a `KeyError`. Either rename your columns to match, or
update `load_chunks()`, `build_context()`, and `_flatten_to_csv()` in
`extract_triplets.py` to reference your names.

## Quick Start

### Remote / local OpenAI-compatible API (default backend)

The default `openai_compat` backend works with DeepSeek cloud or any
OpenAI-compatible endpoint. The project currently runs **Qwen3.5-27B-FP8**
locally via vLLM:

```bash
food_lab/bin/python src/extraction/extract_triplets.py \
    --chunks-csv  src/data/FSSAI_docs/processed/chunks.csv \
    --prompt-file prompts/triplet_extraction_v2.txt \
    --schema-file src/extraction/schema_config.json \
    --output-dir  src/outputs/triplets \
    --base-url    http://localhost:8030/v1 \
    --model       Qwen/Qwen3.5-27B-FP8 \
    --api-key     $DEEPSEEK_API_KEY \
    --concurrency 10 \
    --resume
```

(For the DeepSeek cloud API, drop `--base-url` and use
`--model deepseek-v4-flash` with `--api-key $DEEPSEEK_API_KEY`.)

### Local HuggingFace model

```bash
# Standard prompt format (system + user messages via chat template)
food_lab/bin/python src/extraction/extract_triplets.py \
    --chunks-csv     src/data/FSSAI_docs/processed/chunks.csv \
    --prompt-file    prompts/triplet_extraction_v2.txt \
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
    --prompt-file          prompts/triplet_extraction_v2.txt \
    --output-dir           src/outputs/triplets \
    --backend              hf_transformers \
    --hf-prompt-format     triplex \
    --hf-trust-remote-code \
    --model                SciPhi/Triplex \
    --hf-device            cpu \
    --hf-quantize          4bit \
    --resume
```

## Output

All output is written to `{output_dir}/{csv_id}/{model_slug}/` (or
`hf__{model_slug}/` for the HuggingFace backend). `csv_id` is auto-derived from
the CSV path (override with `--csv-id`).

| File | Description |
|---|---|
| `triplets.jsonl` | One JSON line per chunk (even zero-triplet chunks get `triplets: []`) |
| `failures.jsonl` | Chunks where the LLM call failed (API/parse/length errors) |
| `schema_violations.jsonl` | Triplets that failed schema checks (unknown type, domain/range mismatch, bad confidence) |
| `triplets.csv` | Flattened view — one row per triplet with snippet metadata |

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

| Field | Type | Description |
|---|---|---|
| `subject_id` / `object_id` | string | Synthetic ID for structural nodes not appearing verbatim in the text. Omit (`""`) for named entities. |
| `event_id` | string | Shared across triplets from the same real-world event (`ev_<topic>_<NNNN>`). Omit for standalone triplets. |
| `event_class` | string | The FFLO class this `event_id` instantiates. Omit if `event_id` is omitted. |
| `polarity` | string | `"affirmed"`, `"negated"`, or `"hedged"`. |
| `mismatch_note` | string | Filled only when `predicate` is `"SCHEMA_MISMATCH"`. |

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--chunks-csv` | *(required)* | Chunk CSV from `build_corpus.py` |
| `--prompt-file` | *(required)* | System prompt template |
| `--schema-file` | `src/extraction/schema_config.json` | Schema JSON config (entity types, relations, domain/range) |
| `--output-dir` | *(required)* | Root output directory |
| `--model` | `deepseek-v4-flash` | Model name, slugified for directory naming |
| `--api-key` | `$DEEPSEEK_API_KEY` | API key (not needed for `--backend hf_transformers`) |
| `--base-url` | `https://api.deepseek.com` | API base URL |
| `--concurrency` | `10` | Max parallel requests (forced to `1` for HF backend) |
| `--timeout` | `120` | Per-request timeout in seconds |
| `--api-max-tokens` | `768` | Max completion tokens for the API backend |
| `--api-temperature` | `0.0` | Sampling temperature |
| `--api-top-p` / `--api-top-k` | *(none)* | Optional sampling controls (vLLM) |
| `--disable-thinking` | `False` | Pass `enable_thinking=false` to vLLM/Qwen chat templates |
| `--resume` | `False` | Skip snippet_ids already in `triplets.jsonl` |
| `--retry-failures` | `False` | Only re-process chunks listed in `failures.jsonl` |
| `--skip-until` | *(none)* | Jump forward to a specific snippet_id |

### HuggingFace-specific flags

| Flag | Default | Description |
|---|---|---|
| `--backend` | `openai_compat` | `openai_compat` or `hf_transformers` |
| `--hf-device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--hf-dtype` | `auto` | `auto`, `float16`, `bfloat16`, or `float32` |
| `--hf-quantize` | `none` | `none`, `4bit`, or `8bit` (requires `bitsandbytes`) |
| `--hf-prompt-format` | `standard` | `standard` (chat template) or `triplex` |
| `--hf-trust-remote-code` | `False` | Required for Triplex and other custom-code models |
| `--hf-cache-dir` | *(none)* | HuggingFace model cache directory |
| `--hf-max-new-tokens` | `2048` | Max tokens for HF generation |
| `--hf-temperature` | `0.0` | Generation temperature |

## Sibling modules

The extraction stage contains several supporting scripts beyond the core
`extract_triplets.py`:

| Module | Purpose |
|---|---|
| `schema_validator.py` | Shared schema loading + triplet validation (canonicalisation, domain/range, subclass hierarchy). |
| `schema_violator.py` | Re-validate `schema_violations.jsonl` against an updated schema and append newly-compliant triplets — no re-extraction needed. |
| `unconstrained_pipeline.py` | Open-domain extraction (`--mode extract`) + normalisation/NLI-flagging/schema-comparison (`--mode flag`) for gap discovery. |
| `normalize.py` | String normalisation, Levenshtein grouping, and embedding-based clustering of free-form relation/type names. |
| `canonicalise_kg.py` | Multi-phase canonicalisation: entity surface-form merge, type-conflict resolution, triple dedup. |
| `apply_judgments.py` | Apply LLM-judge verdicts (keep/fix/drop/review) to fix triplets after validation. |

## Requirements

```bash
pip install httpx
# Optional — for the HuggingFace backend:
pip install torch transformers
# Optional — for quantization:
pip install bitsandbytes
```
