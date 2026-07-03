"""Prompt builders for the LLM-as-Judge entailment validation pipeline.

Each branch gets its own system prompt with instructions, schema context,
few-shot examples, and a strict JSON output format.

The ``build_*_prompt`` functions return ``(system_prompt: str, user_message: str)``
matching the chat-completions call convention used across the codebase.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Schema summary — cached, configurable path
# ---------------------------------------------------------------------------

_DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "prompts" / "triplet_extraction.txt"
)
_schema_cache: dict[str, str] = {}


def _trim_schema(raw: str) -> str:
    """Extract entity-types + relations table from a raw prompt file."""
    start_marker = "## ENTITY TYPES"
    end_marker = "## EXTRACTION RULES"
    start = raw.find(start_marker)
    end = raw.find(end_marker, start) if start != -1 else -1
    if start != -1 and end != -1:
        section = raw[start:end].strip()
    else:
        section = raw

    lines = section.split("\n")
    trimmed: list[str] = []
    for line in lines:
        if (line.strip().startswith("|")
                or "fflo:" in line or "fkg:" in line
                or "fso:" in line or "ssn:" in line
                or "prov:" in line or "lkif:" in line
                or "rdf:" in line):
            trimmed.append(line)
        elif line.strip().startswith("###") or line.strip().startswith("##"):
            trimmed.append(line)
        elif line.strip().startswith("-"):
            trimmed.append(line)
    return "\n".join(trimmed).strip()


def get_schema_text(schema_path: str | Path | None = None) -> str:
    """Return the condensed entity-types + relations table.

    The result is cached per *schema_path*, so the file is only read once
    per unique path.
    """
    path = str(schema_path or _DEFAULT_SCHEMA_PATH)
    if path not in _schema_cache:
        raw = Path(path).read_text(encoding="utf-8")
        _schema_cache[path] = _trim_schema(raw)
    return _schema_cache[path]


# ---------------------------------------------------------------------------
# Output JSON shapes (shared across branches)
# ---------------------------------------------------------------------------

_OUTPUT_BLOCK = """
Return ONLY a single JSON object (no markdown fences, no commentary) with these fields:

Required fields: "verdict", "confidence", "rationale"
- "verdict": one of the allowed labels listed above
- "confidence": float 0.0–1.0 indicating how certain you are
- "rationale": short string explaining your reasoning (≤ 60 chars, one sentence)

IMPORTANT: Keep your total output under 2000 tokens.  If you are running out
of room, skip optional fields rather than truncation.

Branch-specific optional fields are listed in each section below.
"""

# ---------------------------------------------------------------------------
# 1. ZERO-TRIPLET branch
# ---------------------------------------------------------------------------

def zero_triplet_system(schema_path: str | None = None) -> str:
    schema = get_schema_text(schema_path)
    return f"""\
You are an expert validator for a knowledge-graph extraction pipeline.
Your task: decide whether a text chunk TRULY contains ZERO extractable
semantic relations given the FFLO ontology schema, or whether the
extraction model missed a relation it should have captured.

=== SCHEMA (condensed) ===

{schema}

=== JUDGING CRITERIA ===

Read the FULL TARGET CHUNK and the model's NOTES, then assign ONE verdict:

- "correct_zero" — The chunk truly has no entity-entity relations
  (e.g. purely administrative metadata, page numbers, category codes
  without explicit linking, version info).
- "missed_relation" — The chunk DOES contain at least one entity-entity
  relation that maps to the schema above, and the extraction model
  failed to capture it.
- "out_of_schema_only" — The chunk contains a meaningful relation, but
  none of the current schema relations can express it.  The model was
  right to produce zero triplets.
- "uncertain" — You cannot decide confidently.

=== FEW-SHOT EXAMPLES ===

--- EXAMPLE 1 ---
CHUNK: "Dairy products and analogues 01.0: — | Milk and dairy-based
drinks: 1.1 | Fermented and renneted milk products (plain): 1.2"
NOTES: "Target chunk contains only food category codes and names. No
information about adulteration, detection, regulation, or health
effects."
VERDICT: correct_zero
CONFIDENCE: 0.95
RATIONALE: "Purely a list of category codes. No entity-entity relation
is stated."

--- EXAMPLE 2 ---
CHUNK: "35 | Versi on 3 (07.05.2025)"
NOTES: "Chunk contains no extractable information; only a page number
and date."
VERDICT: correct_zero
CONFIDENCE: 1.0
RATIONALE: "Only page/version metadata. No substantive content."

--- EXAMPLE 3 ---
CHUNK: "Paneer is a soft cheese made by coagulating milk with citric
acid. It is classified under unripened cheeses along with mozzarella
and scamorza."
NOTES: "The chunk describes cheese types. No food fraud information."
VERDICT: missed_relation
CONFIDENCE: 0.9
RATIONALE: "States paneer is a cheese. Should have extracted
fflo:belongsToCategory(Paneer, UnripenedCheese)."

{_OUTPUT_BLOCK}

For this branch, also return these optional fields:
- "missed_triplets" (null or array): if verdict is "missed_relation",
  list the triplets that should have been extracted
  [{{"subject":"...","subject_type":"...","predicate":"...","object":"...","object_type":"..."}}]
"""

# ---------------------------------------------------------------------------
# 2. SCHEMA-MISMATCH branch
# ---------------------------------------------------------------------------

def schema_mismatch_system(schema_path: str | None = None) -> str:
    schema = get_schema_text(schema_path)
    return f"""\
You are an expert validator for a knowledge-graph extraction pipeline.
The extraction model was given the FFLO ontology schema.  When it
encountered a relation that did NOT map to any schema relation, it
output predicate="SCHEMA_MISMATCH" with a `mismatch_note` describing
the natural-language relation.

Your task for each SCHEMA_MISMATCH triplet:
1. Is the claimed relation actually SUPPORTED by the evidence text?
2. Is the relation GENUINELY outside the current schema?
3. If yes, what schema extension would capture it?

=== SCHEMA (condensed) ===

{schema}

=== JUDGING CRITERIA ===

- "needs_schema_extension" — The text supports the relation AND it is
  genuinely not expressible with current schema relations.  Propose a
  new relation with domain/range.
- "maps_to_existing" — The text supports the relation BUT it actually
  maps to an existing schema relation (the model made an error).
  Specify which existing relation.
- "unsupported" — The evidence text does NOT actually state the claimed
  relation (fabrication or over-interpretation).
- "malformed" — The extraction is garbled / nonsensical even by LLM
  standards.
- "uncertain" — Cannot decide confidently.

=== FEW-SHOT EXAMPLES ===

--- EXAMPLE 1 ---
CHUNK: "further to amend the Food Safety and Standards Rules, 2011"
TRIPLET: subject="Amendment Rules 2017"
subject_type="fflo:RegulatoryDocument" predicate="SCHEMA_MISMATCH"
object="Food Safety Rules 2011"
object_type="fflo:RegulatoryDocument"
MISMATCH_NOTE: "amends"
VERDICT: needs_schema_extension
CONFIDENCE: 0.95
RATIONALE: "'amends' is a genuine legal relation between two regulatory
documents, not in current schema."
PROPOSED_RELATION: "fflo:amends"
PROPOSED_DOMAIN: "fflo:RegulatoryDocument"
PROPOSED_RANGE: "fflo:RegulatoryDocument"

--- EXAMPLE 2 ---
CHUNK: "de-oiled meal means the residual material left over when oil
is extracted from oil-bearing material by a process of solvent or
mechanical extraction"
TRIPLET: subject="FSSAI Regulations"
subject_type="fflo:RegulatoryDocument" predicate="SCHEMA_MISMATCH"
object="De-oiled meal" object_type="fkg:Ingredient"
MISMATCH_NOTE: "defines term"
VERDICT: needs_schema_extension
CONFIDENCE: 0.9
RATIONALE: "Text defines a regulated term. No 'defines' relation
exists in schema."
PROPOSED_RELATION: "fflo:defines"
PROPOSED_DOMAIN: "fflo:RegulatoryDocument"
PROPOSED_RANGE: "rdfs:Resource"

--- EXAMPLE 3 ---
CHUNK: "The oil shall be clear and free from rancidity..."
TRIPLET: subject="BENZOATES" subject_type="fflo:Adulterant"
predicate="SCHEMA_MISMATCH"
object="PermissibleLimit_BENZOATES_250"
object_type="fflo:PermissibleLimit"
MISMATCH_NOTE: "has permissible limit"
VERDICT: unsupported
CONFIDENCE: 0.85
RATIONALE: "Text does not mention BENZOATES at all; hallucinated
from table column header pattern."

{_OUTPUT_BLOCK}

For this branch, also return:
- "existing_relation" (null or string): if verdict is "maps_to_existing"
- "proposed_relation" (null or string): if verdict is "needs_schema_extension"
- "proposed_domain" (null or string)
- "proposed_range" (null or string)
"""

# ---------------------------------------------------------------------------
# 3. SCHEMA-INVALID ordinary-triplet branch
# ---------------------------------------------------------------------------

def schema_invalid_system(schema_path: str | None = None) -> str:
    schema = get_schema_text(schema_path)
    return f"""\
You are an expert validator for a knowledge-graph extraction pipeline.
The extraction model produced a triplet that FAILED schema validation
(wrong domain/range, unknown relation, or unknown entity type).

Your task: determine whether the triplet is SEMANTICALLY CORRECT
despite the schema violation — i.e., whether the evidence text actually
supports the claimed relation, regardless of whether the type/relation
annotations are formally correct.

=== SCHEMA (condensed) ===

{schema}

=== JUDGING CRITERIA ===

- "entailed_direction_reversed" — The text supports the relation but
  subject and object roles should be SWAPPED (the relation arrow
  points the wrong way).
- "entailed_wrong_types" — The text supports the relation but one or
  both entity types are incorrect (e.g. RegulatoryDocument instead of
  FoodStandard, or FoodCategory instead of Food).
- "entailed_wrong_relation" — The text supports the relation but the
  predicate is wrong (unknown or misnamed relation).  Suggest the
  correct one if it exists.
- "not_entailed" — The evidence text does NOT actually support the
  claimed relation (fabrication).
- "uncertain" — Cannot decide confidently.

=== FEW-SHOT EXAMPLES ===

--- EXAMPLE 1 ---
CHUNK: "In exercise of the powers conferred by Section 91 of the Food
Safety and Standards Act, 2006, the Central Government hereby makes
the following rules further to amend the Food Safety and Standards
Rules, 2011"
TRIPLET: subject="Amendment Rules 2017"
subject_type="fflo:RegulatoryDocument"
predicate="lkif:created_by"
object="Ministry of Health and Family Welfare"
object_type="fflo:RegulatoryBody"
VIOLATIONS: domain_mismatch (lkif:created_by expects FoodStandard)
VERDICT: entailed_wrong_types
CONFIDENCE: 0.9
RATIONALE: "Text says government created rules. Semantically correct
but subject_type should be FoodStandard, not RegulatoryDocument.
Or the schema should broaden lkif:created_by domain."

--- EXAMPLE 2 ---
CHUNK: "POLYSORBATES | 1,000 mg/kg"
TRIPLET: subject="Polysorbates"
subject_type="fkg:ChemicalIngredient"
predicate="fflo:appliesTo"
object="PermissibleLimit_Polysorbates"
object_type="fflo:PermissibleLimit"
VIOLATIONS: domain_mismatch, range_mismatch
VERDICT: entailed_direction_reversed
CONFIDENCE: 0.85
RATIONALE: "Relation direction is reversed. Schema says
PermissibleLimit→appliesTo→Adulterant, but model put
Adulterant→appliesTo→PermissibleLimit."

--- EXAMPLE 3 ---
CHUNK: "GMP: GMP | Sodium alginate: Sodium carbonate | 401: 500(i)"
TRIPLET: subject="Sodium alginate"
subject_type="fflo:Adulterant"
predicate="fflo:hasPermissibleLimit"
object="PermissibleLimit_Sodium_alginate"
object_type="fflo:PermissibleLimit"
VIOLATIONS: unknown_relation
VERDICT: not_entailed
CONFIDENCE: 0.85
RATIONALE: "fflo:hasPermissibleLimit is not a real relation. The
model invented it. The garbled table does show a limit exists for
sodium alginate, so the idea is plausible, but the relation name is
wrong and the extraction is unreliable."

{_OUTPUT_BLOCK}

For this branch, also return:
- "correct_relation" (null or string): the correct relation if you can
  identify one, or null if none fits
- "correct_subject_type" (null or string)
- "correct_object_type" (null or string)
"""

# ---------------------------------------------------------------------------
# 4. SCHEMA-VALID entailment branch
# ---------------------------------------------------------------------------

def schema_valid_system(schema_path: str | None = None) -> str:
    schema = get_schema_text(schema_path)
    return f"""\
You are an expert validator for a knowledge-graph extraction pipeline.
A triplet PASSED schema validation — entity types and relation are
formally correct.

Your task: verify whether the triplet is ENTAILED by the evidence text.
That is, does the text actually state (or strongly imply) the subject-
predicate-object relationship claimed by the triplet?

Key principle: the extraction model was instructed to extract ONLY from
the TARGET CHUNK and NOT to infer from world knowledge.  Apply the
same standard.

=== SCHEMA (condensed) ===

{schema}

=== JUDGING CRITERIA ===

- "entailed" — The triplet's subject, predicate, and object are all
  explicitly stated in the text.  A reasonable reader would agree.
- "partially_entailed" — The triplet is plausible/highly inferable
  from the text, but requires one inferential step beyond what is
  explicitly stated.  The `evidence_span` may be an approximation.
- "not_entailed" — The triplet is fabricated, over-extrapolated, or
  inferred from world knowledge not present in the text.  The
  evidence_span does not genuinely support the claim.
- "uncertain" — You cannot decide confidently.

=== AUXILIARY CHECKS ===

For every triplet also report (regardless of verdict):

- "evidence_span_exact": true, false, or "partial"
  (true = verbatim substring of chunk text,
   partial = close but rephrased/substring,
   false = fabricated or from different part of text)
- "entity_types_correct": true or false
  (are the subject_type and object_type appropriate for these entities?)
- "direction_correct": true or false
  (does the relation arrow point the right way?)
- "confidence_reasonable": "yes", "no", "overconfident", or
  "underconfident"
  (is the model's self-reported confidence appropriate given the text?)

Include ALL four auxiliary fields for "entailed" and "partially_entailed"
verdicts only.  For "not_entailed" and "uncertain" you may omit them to
save tokens.

=== FEW-SHOT EXAMPLES ===

--- EXAMPLE 1 ---
CHUNK: "paneer (milk protein coagulated by the addition of citric
acid)... mozzarella and scamorza cheeses"
TRIPLET: subject="Paneer" subject_type="fkg:Food"
predicate="fflo:belongsToCategory" object="UnripenedCheese"
object_type="fflo:FoodCategory"
EVIDENCE_SPAN: "paneer (milk protein coagulated by the addition of
citric acid"
CONFIDENCE: 0.9
VERDICT: entailed
CONFIDENCE(judge): 0.95
RATIONALE: "Text lists paneer as an unripened cheese. Evidence_span is
a verbatim substring. types, direction, confidence all correct."
EVIDENCE_SPAN_EXACT: true
ENTITY_TYPES_CORRECT: true
DIRECTION_CORRECT: true
CONFIDENCE_REASONABLE: "yes"

--- EXAMPLE 2 ---
CHUNK: "Aspartame: Aspartame- | 951: 962 | 600 mg/kg: 350 mg/kg"
TRIPLET: subject="Aspartame" subject_type="fkg:ChemicalIngredient"
predicate="SCHEMA_MISMATCH" object="600 mg/kg"
object_type="fso:Measurement"
EVIDENCE_SPAN: "600 mg/kg"
CONFIDENCE: 0.7
VERDICT: partially_entailed
CONFIDENCE(judge): 0.8
RATIONALE: "Table associates Aspartame with 600 mg/kg, but garbled
format makes relation semantics ambiguous. Numeric value is present
but meaning is unclear."
EVIDENCE_SPAN_EXACT: "partial"
ENTITY_TYPES_CORRECT: true
DIRECTION_CORRECT: false
CONFIDENCE_REASONABLE: "underconfident"

--- EXAMPLE 3 ---
CHUNK: "FSSAI food categories: 1.0 Dairy, 2.0 Fats and oils, 3.0
Fruits and vegetables..."
TRIPLET: subject="Milk" subject_type="fkg:Food"
predicate="fflo:hasAdulterant" object="Water"
object_type="fflo:NaturalAdulterant"
EVIDENCE_SPAN: "1.0 Dairy"
CONFIDENCE: 0.5
VERDICT: not_entailed
CONFIDENCE(judge): 0.98
RATIONALE: "Text only lists category codes. No mention of milk, water,
or adulteration anywhere. Completely fabricated."
EVIDENCE_SPAN_EXACT: false
ENTITY_TYPES_CORRECT: false
DIRECTION_CORRECT: false
CONFIDENCE_REASONABLE: "overconfident"

{_OUTPUT_BLOCK}

For this branch, also return:
- "evidence_span_exact": true, false, or "partial"
- "entity_types_correct": true or false
- "direction_correct": true or false
- "confidence_reasonable": "yes", "no", "overconfident", or
  "underconfident"
"""

# ---------------------------------------------------------------------------
# Public API — build user messages for each branch
# ---------------------------------------------------------------------------

def build_zero_triplet_user(chunk_text: str, model_notes: str) -> str:
    chunk = chunk_text[:3000]  # truncate very long chunks
    return (
        f"=== TARGET CHUNK ===\n{chunk}\n=== END CHUNK ===\n\n"
        f"=== MODEL NOTES ===\n{model_notes or '(no notes)'}\n"
    )


def build_schema_mismatch_user(
    chunk_text: str,
    triplet: dict,
) -> str:
    chunk = chunk_text[:3000]
    note = triplet.get("mismatch_note", "") or ""
    return (
        f"=== TARGET CHUNK ===\n{chunk}\n=== END CHUNK ===\n\n"
        f"=== TRIPLET TO JUDGE ===\n"
        f"  subject:      {triplet.get('subject','')}\n"
        f"  subject_type: {triplet.get('subject_type','')}\n"
        f"  predicate:    SCHEMA_MISMATCH\n"
        f"  object:       {triplet.get('object','')}\n"
        f"  object_type:  {triplet.get('object_type','')}\n"
        f"  evidence_span: {triplet.get('evidence_span','')}\n"
        f"  mismatch_note: {note}\n"
    )


def build_schema_invalid_user(
    chunk_text: str,
    triplet: dict,
    violations: list[str],
) -> str:
    chunk = chunk_text[:3000]
    return (
        f"=== TARGET CHUNK ===\n{chunk}\n=== END CHUNK ===\n\n"
        f"=== TRIPLET TO JUDGE ===\n"
        f"  subject:      {triplet.get('subject','')}\n"
        f"  subject_type: {triplet.get('subject_type','')}\n"
        f"  predicate:    {triplet.get('predicate','')}\n"
        f"  object:       {triplet.get('object','')}\n"
        f"  object_type:  {triplet.get('object_type','')}\n"
        f"  evidence_span: {triplet.get('evidence_span','')}\n"
        f"  confidence:   {triplet.get('confidence','')}\n"
        f"  VIOLATIONS:   {', '.join(violations)}\n"
    )


def build_schema_valid_user(
    chunk_text: str,
    triplet: dict,
) -> str:
    chunk = chunk_text[:3000]
    return (
        f"=== TARGET CHUNK ===\n{chunk}\n=== END CHUNK ===\n\n"
        f"=== TRIPLET TO JUDGE ===\n"
        f"  subject:      {triplet.get('subject','')}\n"
        f"  subject_type: {triplet.get('subject_type','')}\n"
        f"  predicate:    {triplet.get('predicate','')}\n"
        f"  object:       {triplet.get('object','')}\n"
        f"  object_type:  {triplet.get('object_type','')}\n"
        f"  evidence_span: {triplet.get('evidence_span','')}\n"
        f"  confidence:   {triplet.get('confidence','')}\n"
    )
