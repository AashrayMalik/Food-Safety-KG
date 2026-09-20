# Building a Food-Safety Knowledge Graph from Regulatory Text

**Author:** Aashray Malik

**Abstract.** This report documents the design, implementation, and evaluation of an
end-to-end pipeline that constructs a reified RDF/OWL knowledge graph (KG) from
food-safety regulation text. The primary corpus is FSSAI (Food Safety and Standards
Authority of India) regulations, though the pipeline is dataset-agnostic and generalises
to any food item. The system extracts typed `<subject, predicate, object>` triplets with a
schema-guided large language model (LLM), verifies every triplet against its source text
using an LLM-as-judge and a fine-tuned natural-language-inference (NLI) cross-encoder,
canonicalises the resulting entities, and serialises the graph with full provenance. The
schema is the Food Fraud Lifecycle Ontology (FFLO) v7 — 89 entity types and 72 relations
organised into five lifecycle stages. The fine-tuned NLI verifier reaches a macro-F1 of
**0.843** on the held-out validation set, and the final graph contains **130,708**
statements covering 90 classes.

---

## Table of contents

1. [Setting up the KG](#1-setting-up-the-kg)
2. [Entity and relation type design](#2-entity-and-relation-type-design)
3. [Entity and relation extraction](#3-entity-and-relation-extraction)
4. [NLI verification](#4-nli-verification)
5. [KG construction](#5-kg-construction)
6. [Results summary](#6-results-summary)
7. [Reproducibility](#7-reproducibility)

---

## 1. Setting up the KG

### 1.1 Objective

The pipeline builds an **FFLO-based (Food Fraud Lifecycle Ontology) RDF/OWL knowledge
graph** from food-safety regulation text. It turns unstructured regulatory prose into a
structured, queryable graph in which every statement carries provenance back to its exact
source paragraph. The workflow is a four-stage cascade:

```
Collection + cleaning  →  LLM extraction  →  NLI + judge validation  →  KG construction
```

| Stage | Implementation |
|---|---|
| S1 — Collection + cleaning | `src/preprocessing_cleaning/build_corpus.py` |
| S2 — Extraction | `src/extraction/extract_triplets.py` (constrained) + `src/extraction/unconstrained_pipeline.py` (open-domain) |
| S3 — Validation | `src/validation/` (NLI cross-encoder + LLM-as-judge) |
| S4 — KG construction | `src/extraction/canonicalise_kg.py` + `src/kg/build_rdf_kg.py` |

### 1.2 Environment

```bash
# Python 3.11 recommended
python3.11 -m venv food_lab
source food_lab/bin/activate

pip install -r src/requirements.txt
```

Dependencies are grouped by pipeline stage in `src/requirements.txt`:

- **Core pipeline** — `httpx`, `networkx`, `rapidfuzz`, `rdflib`, `requests`.
- **Corpus building / scraping** — `markitdown`, `selenium`, `beautifulsoup4`.
- **Canonicalisation / NLI evaluation** — `numpy`, `pandas`, `openpyxl`,
  `scikit-learn`, `scikit-multilearn`.
- **NLI validation** — `sentence-transformers`.
- **Local HuggingFace extraction backend** — `transformers`, `torch`.

### 1.3 Inference backend

Extraction and judging run against **Qwen3.5-27B-FP8** served locally via
[vLLM](https://github.com/vllm-project/vllm) (OpenAI-compatible API). The system can
alternatively target the DeepSeek cloud API or a local HuggingFace model
(`meta-llama/Meta-Llama-3.1-8B-Instruct`, `SciPhi/Triplex`).

```text
base URL: http://localhost:8030/v1
model:    Qwen/Qwen3.5-27B-FP8
```

The fine-tuned NLI cross-encoder checkpoint lives at
`model_checkpoints/run4_deberta_v3_small`.

### 1.4 Repository layout

```
.
├── src/
│   ├── preprocessing_cleaning/   # PDF→markdown, URL scraping, cleaning, chunking
│   ├── extraction/               # triplet extraction, schema validation, canonicalisation
│   ├── validation/               # LLM-as-judge, NLI model, unconstrained validation
│   ├── kg/                       # canonical entity IDs + RDF/OWL KG builder + unit tests
│   ├── scrape/                   # NABL lab-scope scraping + adulterant gap analysis
│   ├── schema_iterations/        # canonical ontology specs (v2/v6, v3/v7)
│   ├── documentation/            # per-stage pipeline docs
│   ├── outputs/                  # triplets + KG output (kg.ttl, ontology.ttl, …)
│   └── requirements.txt
├── prompts/                      # extraction + unconstrained system prompts
├── model_checkpoints/            # fine-tuned NLI cross-encoder (deberta-v3-small)
└── README.md
```

Input data (`src/data/` — FSSAI PDFs and processed chunks) is not checked into the repo;
the pipeline expects the user to supply it.

---

## 2. Entity and relation type design

### 2.1 The FFLO ontology (v7)

The KG is typed against the **Food Fraud Lifecycle Ontology (FFLO) v7**, which organises
food-fraud knowledge into five lifecycle stages:

| Stage | Scope | Example relations |
|---|---|---|
| **ACT** | Adulterants, additives, fraud methods | `hasAdulterant`, `hasFunction`, `hasFraudType` |
| **SPREAD** | Supply-chain propagation | `propagatesTo`, `occursAt`, `carriedBy` |
| **DETECTION** | Sampling, analysis, findings | `identifies`, `foundIn`, `detectedBy` |
| **REGULATION** | Standards, limits, actions | `hasValue`, `appliesTo`, `compliesWith` |
| **HEALTH** | Health effects & populations | `causesEffect`, `affectsPopulation` |

The v7 schema contains **89 entity types** (87 ontology classes plus the `xsd:string`
and `xsd:decimal` datatypes), **72 relations**, and **45 `subclass_of` axioms**. It
reuses standard vocabularies where they fit — PROV-O (`prov:`) for provenance, SSN/SOSA
(`ssn:`/`sosa:`) for observations, the Food Safety Ontology (`fso:`), and LKIF
(`lkif:`) — alongside the FFLO (`fflo:`) and Food Knowledge Graph (`fkg:`) namespaces.

### 2.2 Entity types by stage

**ACT stage** distinguishes the four adulterant subtypes
(`SyntheticAdulterant`, `NaturalAdulterant`, `ContaminantAdulterant`, `Microorganism`),
permitted `FoodAdditive`s (with their `AdditiveFunction` functional classes), and the
`AdulterationMethod` / `FraudType` / `IntentionalityLevel` apparatus for modelling *how*
fraud happens. `fkg:Food`, `fkg:Ingredient`, and the `fkg:ChemicalIngredient` superclass
ground the food side of the taxonomy.

**SPREAD stage** models the supply chain as a `SupplyChainStep` hierarchy
(`Production` → `Processing` → `Storage` → `Distribution` → `Retail` → `Import`), with
`SpreadEvent` and `TransformationEvent` reifying contamination movement.

**DETECTION stage** reuses the Food Safety Ontology (`fso:Sample`, `fso:Analysis`,
`fso:AnalysisResult`, `fso:Measurement`, `ssn:Property`) and defines the core
`IncidentFinding` entity with four subtypes (`LabConfirmed`, `FieldDetected`,
`SurveyAggregated`, `RecallTriggered`), plus the `DetectionMethod` hierarchy
(`LaboratoryTest`, `FieldTest`, `SensoryTest`).

**REGULATION stage** models `FoodStandard`, `RegulatoryDocument`, `RegulatoryBody`,
`RegulatoryOfficer`, `RegulatoryAction`, and the quantity-limit apparatus
(`PermissibleLimit` grounded to `ssn:Property` + `fso:Measurement`).

**HEALTH stage** defines `HealthEffect` with its `CausalHealthEffect` /
`AcuteEffect` / `ChronicEffect` refinement and `VulnerablePopulation`.

### 2.3 Schema evolution (v6 → v7) with quantified justification

Each v7 addition is justified by the fraction of validated triplets that could not be
expressed in v6. The denominator is **5,882 validated triplets**:

| Addition | Share of triplets | Justification |
|---|---|---|
| `ObligationType` (+ `hasObligation`) | 426 / 5,882 (7.2%) | `must*` predicates (`mustComplyWith`, `mustBeStoredAt`, …) had no v6 relation — the largest single gap by volume |
| `RegulatoryOfficer` (+ `actsOnBehalfOf`) | 375 / 5,882 (6.4%) | Officer/analyst/inspector roles (`hasDuty`, `requiresQualification`, …) were typed inconsistently with no formal class |
| `Microorganism` (subtype of `ContaminantAdulterant`) | 175 / 5,882 (3.0%) | Microbial content ("Salmonella detectedBy IS 5887 Part 3") had no subtype to map to |
| `SamplingPlan` (+ `governsSampling`) | 45 / 5,882 (0.8%) | The m/c/M acceptance-criteria sub-pattern is self-contained but recurring |
| `hasScientificName` | 24 / 5,882 (0.4%) | Clean, unambiguous taxonomic-naming pattern |
| `RegulatoryActionType` (+ `isEmpoweredTo`) | 8 / 5,882 (0.1%) | `mayAdd`/`maySeal`/`canSeize` discretionary powers — thin but zero mapping options |

A further 378 / 5,882 (6.4%) of triplets used bare limit predicates
(`hasAcidityLimit`, `hasFatLimit`, `hasResidueLimit`, …). These are **not** schema
additions: they all decompose into the existing
`PermissibleLimit --forProperty--> ssn:Property / --hasValue--> fso:Measurement` pattern,
so they are a canonicalisation/mapping concern rather than an ontology gap.

### 2.4 Single source of truth

The schema is defined in exactly three places, kept in sync by the `schema_validator`
module:

1. `src/schema_iterations/schema_v3.md` — the canonical human-readable spec.
2. `src/extraction/schema_config.json` — the runtime source of truth (`entity_types`,
   `relations`, `domain_range`, `subclass_of`). Extraction, validation, and RDF building
   all load this at startup; there are no hardcoded lists in the code.
3. `prompts/triplet_extraction_v2.txt` — the system prompt the LLM sees.

Updating the ontology means editing only these three files; `schema_validator` derives
canonical names, Triplex prefixed lists, and startup assertions automatically.

---

## 3. Entity and relation extraction

### 3.1 Two extraction strategies

| Strategy | Script | Behaviour |
|---|---|---|
| **Constrained** | `extract_triplets.py` | The model must map every relation to the FFLO schema, or emit `UNMAPPED` |
| **Unconstrained** | `unconstrained_pipeline.py` | The model freely names entity types and relations, for schema-gap discovery |

### 3.2 Input: chunked corpus

Extraction consumes a **chunk CSV** produced by `build_corpus.py`, not raw documents.
Chunking matters for three reasons:

1. **Finite context windows** — documents are split into ~200-word paragraphs the model
   can read in one request.
2. **Neighbour context** — the pipeline reads the chunk before and after the target
   (same `source_id`, ordered by `chunk_index`) to resolve pronouns and cross-sentence
   references.
3. **Traceability** — `snippet_id` uniquely identifies each chunk, so every triplet maps
   back to its exact source paragraph.

Required columns: `snippet_id`, `evidence_text`, `source_id`, `chunk_index`.

### 3.3 Schema-guided prompting

The system prompt (`prompts/triplet_extraction_v2.txt`) instructs the model to:

- Use **only** the v7 entity types and relations, never inventing new names.
- Decompose n-ary events into multiple triplets sharing one synthetic `event_id`,
  reified as the correct `event_class` (e.g. an `IncidentFinding` with `identifies` /
  `foundIn` / `producedBy` / `inRegion` edges).
- Decompose every quantity limit into the mandatory three-relation pattern
  (`PermissibleLimit --forProperty--> ssn:Property`, `--hasValue--> fso:Measurement`,
  `--inFood--> fkg:Food`) rather than a bare `hasLimit` relation.
- Mark polarity (`affirmed` / `negated` / `hedged`) and attach a verbatim
  `evidence_span` (≤ 15 words) to every triplet.
- Emit `predicate: "UNMAPPED"` with a `mismatch_note` when no schema relation fits —
  this is a deliberate signal that flags ontology gaps for later review, rather than
  forcing a fit.

### 3.4 Triplet JSON schema

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

`subject_id`/`object_id` carry synthetic IDs for structural nodes that do not appear
verbatim in the text; `event_id` is shared across all triplets describing one real-world
event.

### 3.5 Outputs

All output is written to `{output_dir}/{csv_id}/{model_slug}/`:

| File | Description |
|---|---|
| `triplets.jsonl` | One JSON line per chunk (zero-triplet chunks get `triplets: []`) |
| `failures.jsonl` | Chunks where the LLM call failed |
| `schema_violations.jsonl` | Triplets that failed schema checks (unknown type, domain/range mismatch) |
| `triplets.csv` | Flattened view — one row per triplet with snippet metadata |

Supporting modules: `schema_validator.py` (shared schema loading + validation),
`schema_violator.py` (re-validate violations against an updated schema without
re-extraction), `normalize.py` (string normalisation + clustering of free-form names),
and `apply_judgments.py` (apply judge verdicts to fix triplets post-validation).

---

## 4. NLI verification

Every extracted triplet is graded for whether it is **entailed** by its source text.
Three complementary approaches are used:

### 4.1 LLM-as-judge (`LLM_judge/judge.py`)

A second LLM routes each item into one of four evaluation branches, each with its own
criteria and few-shot prompt:

| Branch | Verdicts |
|---|---|
| **Zero-triplet** | `correct_zero`, `missed_relation`, `out_of_schema_only`, `uncertain` |
| **Schema-mismatch** | `needs_schema_extension`, `maps_to_existing`, `unsupported`, `malformed` |
| **Schema-invalid** | `entailed_direction_reversed`, `entailed_wrong_types`, `entailed_wrong_relation`, `not_entailed`, `uncertain` |
| **Schema-valid** | `entailed`, `partially_entailed`, `not_entailed`, `uncertain` |

The schema-valid branch is the main entailment check and also reports auxiliary signals:
`evidence_span_exact`, `entity_types_correct`, `direction_correct`, and
`confidence_reasonable`. With `--use-logprobs`, the judge requests token-level
log-probabilities to compute an **objective** confidence signal (cumulative `verdict_logprob`,
`output_perplexity`) alongside the model's self-reported confidence — this is used to
flag overconfident judgments for manual audit.

### 4.2 NLI cross-encoder (`NLI_model/`)

A `sentence-transformers` cross-encoder classifies each triplet's **verbalisation**
against its source text as `entailment` / `contradiction` / `neutral`. Each relation and
entity type has a natural-language hypothesis template (`verbalise.py`); structural
nodes are given descriptive labels by `synthetic_ids.py`.

The fine-tuned checkpoint is based on `cross-encoder/nli-deberta-v3-small`:

| Property | Value |
|---|---|
| Base model | `cross-encoder/nli-deberta-v3-small` |
| Training samples | 4,471 (split by `snippet_id`, stratified by predicate) |
| Epochs / batch / LR | 5 / 16 / 5e-5 |
| **Validation F1-macro** | **0.843** (micro 0.844, weighted 0.844) |
| Max sequence length | 256 tokens |

Fine-tuning splits by unique `snippet_id` (optionally `--split-mode predicate_disjoint`
for a cross-relation generalisation test) and filters cross-premise leakage. Evaluation
(`nli_eval.py`) emits per-example predictions, per-model metrics, a comparison table, and
a disagreement matrix across models.

### 4.3 Two-stage NLI + LLM judge (`nli_llm_judge.py`)

For every triplet in the **unconstrained** extraction output:

1. Chunk-level NLI (premise = full `evidence_text`),
2. Span-level NLI (premise = `evidence_span`),
3. A local Qwen LLM judge consuming both NLI scores,
4. An agreement status: `agreed_all` / `agreed_partial` / `contested` — contested rows
   are flagged for human review, with the LLM judge taking precedence in the final
   verdict.

### 4.4 Unconstrained canonicalisation (`unconstrained/`)

The open-domain extraction output is canonicalised back onto the ontology with an
LLM-driven "verify-then-name" procedure (with an antonym guard) in
`llm_canonicalise.py`, then applied to the triplet CSV by `apply_canonical.py`.

---

## 5. KG construction

Construction has two stages: entity canonicalisation, then reified RDF serialisation.

```
triplets.csv ──→ canonicalize.py ──→ triplets_canonicalized.csv ──→ build_rdf_kg.py ──→ kg.ttl / kg.nt / ontology.ttl
```

### 5.1 Entity canonicalisation (`kg/canonicalize.py`)

Each entity is assigned a stable, type-prefixed canonical ID so surface-form variants of
the same real-world entity collapse to one node (`fkg:ING_00001` — `fkg:` + first three
letters of the type uppercased + counter). The procedure:

- **Normalises** — lowercase, punctuation → space (never deleted, so hyphenated and
  space-separated variants collapse to the same key), whitespace collapse.
- **Blocks by type** — entities cluster only within the same type, so unrelated entities
  never merge.
- **Clusters** — TF-IDF over character n-grams (3–5) → cosine similarity → union-find.
  Only exact normalised matches auto-merge; near-duplicates go to a borderline file for
  human/LLM review.
- **Guards against false merges** — conflicting numeric values never merge; a leading
  qualifier difference is flagged; `Measurement`/`xsd:string` nodes use exact-match only.

Outputs include `triplets_canonicalized.csv` plus `canonicalization_audit.csv`,
`canonicalization_borderline_review.csv`, and `canonicalization_cross_type_merges.csv`.

### 5.2 RDF construction with reification and provenance (`kg/build_rdf_kg.py`)

Every statement is **reified** as an `fflo:Assertion` node linked to its subject,
predicate, and object (`assertsSubject`/`assertsPredicate`/`assertsObject`), carrying
full provenance: `snippetId`, `sourceId`, `sourceFile`, `sourceType`, `chunkIndex`,
`evidenceSpan`, and `confidence`.

The builder reads the live `schema_config.json` at runtime and quarantines **schema
drift** rather than silently emitting invalid triples. Each row is checked for unmapped
predicates, unknown types, and **value-kind consistency** — a predicate's declared OWL
property kind (Datatype/Object/Annotation) must match whether the object is a literal or
an entity, preventing OWL punning violations that would otherwise break Protégé/OWL-API
tools. Failing rows are written to `schema_drift.csv` with reasons and suggested fixes,
forming a feedback loop for ontology iteration.

### 5.3 Outputs

| File | Description |
|---|---|
| `kg.ttl` | ABox (data) graph with an `owl:imports` of the ontology |
| `kg.nt` | Same graph in N-Triples |
| `ontology.ttl` | TBox: classes, subclass axioms, property kinds, domain/range |
| `kg_full.ttl` | Self-contained schema + data (no external imports) |
| `catalog-v001.xml` | Protégé/OWL-API catalog |
| `schema_drift.csv` | Quarantined rows with reasons + suggestions |
| `build_report.json` | Counts, drift reasons, output paths |

Optional `--load-endpoint` uploads `kg.ttl` to a GraphDB/SPARQL endpoint after
generation.

---

## 6. Results summary

| Metric | Value |
|---|---|
| Ontology entity types | 89 (87 classes + `xsd:string`, `xsd:decimal`) |
| Ontology relations | 72 |
| Subclass axioms | 45 |
| Extracted triplets (raw) | 15,501 |
| Triplets after canonicalisation | 11,542 |
| Validated triplets (schema-gap denominator) | 5,882 |
| NLI cross-encoder validation F1-macro | **0.843** (micro 0.844) |
| NLI training samples | 4,471 |
| Final graph statements (`kg.ttl`) | **130,708** |
| Ontology classes (`ontology.ttl`) | 90 |
| Ontology property declarations | 85 |

---

## 7. Reproducibility

```bash
# 1. Build the chunk corpus from PDFs
food_lab/bin/python src/preprocessing_cleaning/build_corpus.py \
    --pdf-dir   src/data/FSSAI_docs/pdfs \
    --output-dir src/data/FSSAI_docs/processed

# 2. Extract schema-guided triplets (Qwen via local vLLM)
food_lab/bin/python src/extraction/extract_triplets.py \
    --chunks-csv  src/data/FSSAI_docs/processed/chunks.csv \
    --prompt-file prompts/triplet_extraction_v2.txt \
    --schema-file src/extraction/schema_config.json \
    --output-dir  src/outputs/triplets \
    --base-url    http://localhost:8030/v1 \
    --model       Qwen/Qwen3.5-27B-FP8 \
    --resume

# 3. Validate triplets for entailment (LLM-as-judge)
food_lab/bin/python src/validation/LLM_judge/judge.py \
    --chunks-csv    src/data/FSSAI_docs/processed/chunks.csv \
    --triplets-dir  src/outputs/triplets \
    --output-dir    src/outputs/validation \
    --judge-model   Qwen/Qwen3.5-27B-FP8 \
    --base-url      http://localhost:8030/v1 \
    --use-logprobs

# 4. Canonicalise entities and deduplicate triples
food_lab/bin/python src/extraction/canonicalise_kg.py \
    --input      src/outputs/triplets/triplets.csv \
    --output-dir src/outputs/triplets/

# 5. Build the RDF/OWL knowledge graph
food_lab/bin/python src/kg/build_rdf_kg.py \
    --triplets-csv  src/outputs/triplets/triplets_canonicalized.csv \
    --schema-config src/extraction/schema_config.json \
    --output-dir    src/outputs/kg_output
```

**NLI fine-tuning / inference / evaluation:**

```bash
# fine-tune
food_lab/bin/python src/validation/NLI_model/nli_finetune.py \
    --train-csv  src/validation/Golden_val/nli_train_pairs.csv \
    --base-model cross-encoder/nli-deberta-v3-small \
    --output-dir model_checkpoints/run4_deberta_v3_small \
    --epochs 5 --batch-size 16 --device cuda:0

# evaluate
python3 src/validation/NLI_model/nli_eval.py \
    --models cross-encoder/nli-deberta-v3-small,model_checkpoints/run4_deberta_v3_small \
    --labels pretrained,finetuned \
    --eval-csv src/validation/Golden_val/nli_train_pairs.csv \
    --device cuda:0
```

**Tests:**

```bash
python3 -m unittest src/kg/test_build_rdf_kg.py
```
