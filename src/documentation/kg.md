# Knowledge-Graph Construction (`src/kg/`)

Turns validated, canonicalised triplets into a reified **RDF/OWL knowledge
graph** with full provenance. Two stages:

1. `canonicalize.py` — assign stable canonical entity IDs to each subject/object.
2. `build_rdf_kg.py` — validate against the live schema and serialise to
   Turtle / N-Triples, quarantining any schema drift.

A third file, `test_build_rdf_kg.py`, is a `unittest` suite for the RDF builder.

```
triplets.csv ──→ canonicalize.py ──→ triplets_canonicalized.csv ──→ build_rdf_kg.py ──→ kg.ttl / kg.nt
                                          (+ audit CSVs)                    (+ schema drift CSV)     ontology.ttl / kg_full.ttl
```

## Stage 1 — `canonicalize.py`

Assigns each entity a stable, type-prefixed canonical ID so surface-form
variants of the same real-world entity collapse to one node.

```bash
python3 src/kg/canonicalize.py \
    --input  src/outputs/triplets/triplets.csv \
    --output src/outputs/triplets/triplets_canonicalized.csv
```

### How it works

- **Normalise** — lowercase, punctuation→space (never delete, so hyphenated and
  space-separated variants collapse to the same key), collapse whitespace.
- **Block by type** — entities are clustered only within the same
  `subject_type`/`object_type`, so unrelated entities never merge.
- **Cluster** — TF-IDF over character n-grams (3–5) → cosine similarity, then
  union-find. Similarity is effectively disabled for auto-merging; only exact
  normalised matches merge automatically, everything short of exact goes to a
  borderline file for human/LLM review.
- **Guards against false merges**:
  - conflicting numeric values (`"1.0%"` vs `"5%"`) never merge;
  - an extra leading qualifier (`"sodium calcium polyphosphate"` vs
    `"calcium polyphosphate"`) is flagged, not merged;
  - `NO_SIMILARITY_CLUSTERING_TYPES = {fso:Measurement, xsd:string}` get
    exact-normalised-match only.
- **Event IDs** — synthetic `ev_*` reification IDs pass through untouched (each
  is already a unique event instance).
- **Cross-type exact merge** — the same exact string with inconsistent
  extraction-time type labels is merged across types (logged to a separate
  audit file).

IDs look like `fkg:ING_00001` (`fkg:` + first 3 letters of the type uppercased
+ counter). If the input already carries NLI-assigned `subject_id`/`object_id`
(synthetic n-ary node IDs), those are left untouched and canonical IDs are
written to `subject_canonical_id`/`object_canonical_id` instead.

### Outputs

| File | Description |
|---|---|
| `triplets_canonicalized.csv` | input with `subject_id`/`object_id` populated |
| `canonicalization_audit.csv` | every cluster: canonical ID, type, members, row count |
| `canonicalization_borderline_review.csv` | near-duplicate pairs flagged for manual/LLM review |
| `canonicalization_cross_type_merges.csv` | cross-type exact-match merges for review |

> Note: `src/extraction/canonicalise_kg.py` is a separate, more elaborate
> multi-phase canonicaliser (fuzzy + graph-neighbour + type-conflict resolution,
> triple dedup). `kg/canonicalize.py` is the lightweight ID-assignment step the
> RDF builder expects as input.

## Stage 2 — `build_rdf_kg.py`

Reads the active `schema_config.json` at runtime, validates every triplet,
writes schema-conformant rows to RDF, and quarantines drift.

```bash
python3 src/kg/build_rdf_kg.py \
    --triplets-csv  src/outputs/triplets/triplets_canonicalized.csv \
    --schema-config src/extraction/schema_config.json \
    --output-dir    src/outputs/kg_output
```

### Validation & drift quarantine

Each row is checked against the live schema:

- `UNMAPPED` predicate, unknown predicate, unknown subject/object type.
- **Value-kind consistency** — a predicate's declared OWL property kind
  (`DatatypeProperty` for `xsd:` ranges, `ObjectProperty` for entity ranges,
  `AnnotationProperty` when both) must match whether the object is a literal or
  an entity. This prevents OWL punning violations that otherwise break
  `ontology.ttl` + `kg.ttl` in Protégé/OWL-API tools.

Rows that fail are written to `schema_drift.csv` (with reasons and suggested
fixes) rather than the graph — a feedback loop for ontology iteration.

### Reification & provenance

Every statement is reified as an `fflo:Assertion` node linked to its subject,
predicate, and object (`assertsSubject`/`assertsPredicate`/`assertsObject`),
plus provenance metadata: `snippetId`, `sourceId`, `sourceFile`, `sourceType`,
`chunkIndex`, `evidenceSpan`, and `confidence`.

### Outputs

| File | Description |
|---|---|
| `kg.ttl` | ABox (data) graph with an `owl:imports` of the ontology |
| `kg.nt` | same graph in N-Triples |
| `ontology.ttl` | TBox: classes, subclass axioms, property kinds, domain/range |
| `kg_full.ttl` | self-contained schema + data (no external imports) |
| `catalog-v001.xml` | Protégé/OWL-API catalog mapping the ontology IRI to `ontology.ttl` |
| `schema_drift.csv` | quarantined rows with reasons + suggestions |
| `build_report.json` | counts, drift reasons, output paths |

### Optional graph-store upload

`--load-endpoint http://host:7200/repositories/kg/statements` POSTs `kg.ttl`
(after generation) to any endpoint that accepts Turtle.

### Namespaces

`fflo`, `fkg`, `fso`, `ssn`, `sosa`, `prov`, `lkif`, plus standard
`rdf`/`rdfs`/`owl`/`xsd`. Entity IRIs live under `urn:fflo:resource:` (or the
configured `--base-iri`); assertions under `urn:fflo:assertion:`.

## Tests

```bash
# from repo root
python3 -m unittest src/kg/test_build_rdf_kg.py
```

Covers IRI slugging/joining, schema-drift quarantining, ontology subclass
axioms, and the provenance vocabulary emitted into `kg.ttl` / `ontology.ttl` /
`kg_full.ttl`.

## Requirements

```bash
pip install rdflib
# canonicalize.py additionally:
pip install numpy scikit-learn
```
