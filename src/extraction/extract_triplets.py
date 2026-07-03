#!/usr/bin/env python3
"""Async parallel triplet extraction from text chunks via DeepSeek API or HuggingFace.

Reads a chunk CSV produced by build_corpus.py, sends each chunk to the
LLM backend with surrounding context for continuity, and writes structured
triplets to JSONL (per-chunk) and a flattened CSV.

Backends:
    openai_compat  – any OpenAI-compatible /chat/completions endpoint
                     (DeepSeek, vLLM, TGI, …).  Default.
    hf_transformers – local HuggingFace model via torch + transformers.

The ``--model`` name is slugified and used as the output sub-directory
(prefix ``hf__`` is added for the hf_transformers backend), so running
multiple models / backends never overwrites previous results.

Usage (remote API)::

    food_lab/bin/python src/extract_triplets.py \\
        --chunks-csv  src/data/FSSAI_docs/processed/chunks.csv \\
        --prompt-file prompts/triplet_extraction.txt \\
        --output-dir  outputs/triplets \\
        --api-key     $DEEPSEEK_API_KEY \\
        --concurrency 10 \\
        --resume

Usage (local HF model)::

    food_lab/bin/python src/extract_triplets.py \\
        --chunks-csv  src/data/FSSAI_docs/processed/chunks.csv \\
        --prompt-file prompts/triplet_extraction.txt \\
        --output-dir  outputs/triplets \\
        --backend     hf_transformers \\
        --model       meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --hf-device   mps \\
        --resume
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
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_CONCURRENCY = 10
DEFAULT_TIMEOUT = 120  # seconds per request
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry
MAX_CONTEXT_CHUNKS_PREV = 2  # previous chunks for continuity
MAX_CONTEXT_CHUNKS_NEXT = 1  # next chunk for continuity

# HTTP statuses that warrant a retry
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

# Detect if torch + transformers are available for the hf_transformers
# backend.  They are *not* required for the default openai_compat backend.
_HF_AVAILABLE = False
try:
    import torch  # noqa: F401
    from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: F401
    _HF_AVAILABLE = True
except ImportError:
    pass

DEFAULT_BACKEND = "openai_compat"
DEFAULT_HF_MAX_NEW_TOKENS = 4096
DEFAULT_HF_TEMPERATURE = 0.0
DEFAULT_HF_QUANTIZE = "none"
DEFAULT_HF_PROMPT_FORMAT = "standard"
DEFAULT_HF_TRUST_REMOTE_CODE = False

# ---------------------------------------------------------------------------
# Schema — used to validate every triplet the model returns.
# Synced with src/schema_v5.md.  Names are canonicalised to their unprefixed
# form (e.g. "fflo:Adulterant" → "Adulterant") so the validator tolerates
# both prefixed and unprefixed model output.
# ---------------------------------------------------------------------------

# Map namespace prefixes → nothing for canonicalisation
_PREFIX_RE = re.compile(r"^(fflo|fkg|fso|prov|lkif|ssn):")

def _canonical(name: str) -> str:
    """Strip known namespace prefixes for fuzzy matching."""
    return _PREFIX_RE.sub("", name.strip())


VALID_ENTITY_TYPES = frozenset(_canonical(t) for t in [
    # ACT stage
    "fflo:FoodBusinessOperator", "fflo:AdulterationAct",
    "fflo:Adulterant", "fflo:SyntheticAdulterant", "fflo:NaturalAdulterant",
    "fflo:ContaminantAdulterant", "fflo:AdulterationMethod",
    "fflo:FraudType", "fflo:IntentionalityLevel",
    "fkg:Food", "fkg:Ingredient", "fkg:ChemicalIngredient",
    "fflo:FoodAdditive",
    # ACT stage — AdditiveFunction vocabulary
    "fflo:AdditiveFunction",
    "fflo:Preservative", "fflo:Antioxidant", "fflo:Emulsifier",
    "fflo:Stabilizer", "fflo:AcidityRegulator",
    "fflo:FlourTreatmentAgent", "fflo:Sequestrant",
    "fflo:HumectantAdditive", "fflo:Colourant", "fflo:SweeteningAgent",
    # SPREAD stage
    "fflo:SupplyChainStep",
    "fflo:Production", "fflo:Processing", "fflo:Storage",
    "fflo:Distribution", "fflo:Retail", "fflo:Import",
    "fflo:SpreadEvent", "fflo:TransformationEvent",
    "fflo:GeographicRegion",
    # DETECTION stage — FSO/SOSA
    "fso:Sample", "fso:SampleType", "fso:Analysis", "fso:AnalysisResult",
    "ssn:Property",
    "fso:Measurement", "fso:UnitOfMeasure", "fso:Laboratory", "fso:Location",
    # DETECTION stage — FFLO Incident
    "fflo:IncidentFinding", "fflo:LabConfirmed", "fflo:FieldDetected",
    "fflo:SurveyAggregated", "fflo:RecallTriggered",
    "fflo:EvidenceSource",
    # DETECTION stage — FFLO Methods
    "fflo:DetectionMethod", "fflo:LaboratoryTest",
    "fflo:FieldTest", "fflo:SensoryTest",
    "fflo:DetectionIndicator", "fflo:DetectionKit",
    # REGULATION stage — LKIF
    "fflo:FoodStandard", "fflo:RegulatoryDocument",
    "fflo:RegulatoryBody", "fflo:RegulatoryAction",
    "fflo:PermissibleLimit", "fflo:Violation",
    # REGULATION stage — FFLO
    "fflo:FoodCategory", "fflo:TableReference",
    # HEALTH stage — OAE
    "fflo:HealthEffect", "fflo:CausalHealthEffect",
    "fflo:AcuteEffect", "fflo:ChronicEffect",
    "fflo:VulnerablePopulation",
])


VALID_RELATIONS = frozenset(_canonical(r) for r in [
    # ACT stage — PROV-O
    "prov:wasAssociatedWith", "prov:used",
    "prov:wasGeneratedBy", "prov:wasAttributedTo",
    # ACT stage — FFLO
    "fflo:hasAdulterant", "fflo:hasAdulterationMethod",
    "fflo:hasFraudType", "fflo:hasIntentionality",
    "fflo:isSubstituteFor", "fflo:commonIn", "fflo:foundAt",
    # ACT stage — FFLO v2 additions
    "fflo:hasFunction", "fkg:hasIngredient",
    # SPREAD stage — PROV-O
    "prov:wasInfluencedBy",
    # SPREAD stage — FFLO
    "fflo:propagatesTo", "fflo:carriedBy", "fflo:occursAt",
    "fflo:inputTo", "fflo:outputOf",
    # DETECTION stage — FSO
    "fso:isPerformedOn", "fso:isPerformedAt", "fso:isResultOf",
    "fso:relatesToProperty", "fso:isMeasuredIn",
    "fso:hasSampleType", "fso:hasLocation",
    # DETECTION stage — FFLO v2 additions
    "fflo:hasNumericValue",
    # DETECTION stage — FFLO Incident
    "fflo:identifies", "fflo:foundIn", "fflo:producedBy",
    "fflo:confirmedBy", "fflo:supportedBy",
    "fflo:foundAtStep", "fflo:inRegion", "fflo:triggeredAction",
    "fflo:collectedAt",
    # DETECTION stage — FFLO Methods
    "fflo:detectedBy", "fflo:producesIndicator",
    "fflo:requiresKit", "fflo:isPerformedAs",
    # REGULATION stage — LKIF
    "lkif:created_by",
    # REGULATION stage — FFLO bridges
    "fflo:belongsToCategory", "fflo:appliesToCategory",
    "fflo:hasValue", "fflo:forProperty", "fflo:comparedAgainst",
    # REGULATION stage — FFLO structure
    "fflo:definedIn", "fflo:appliesTo", "fflo:inFood",
    "fflo:specifiedIn", "fflo:issuedBy", "fflo:targets",
    "fflo:citesFinding", "fflo:constitutes", "fflo:violates",
    # REGULATION stage — FFLO v2 additions
    "fflo:defines", "fflo:hasPermissibleLimit",
    "fflo:appliesToFood", "fflo:compliesWith",
    "fflo:amends", "fflo:hasSubcategory",
    "fflo:hasDefinition", "fflo:tableLabel",
    # HEALTH stage — FFLO
    "fflo:causesEffect", "fflo:affectsPopulation",
    "fflo:hasEffectType", "fflo:associatedWith",
])


# relation canonical-name → list of (valid_domain, valid_range) pairs.
# Multiple pairs per relation allowed (e.g. prov:wasGeneratedBy has two).
# Empty frozenset = skip check (open-ended).
_D = RELATION_DOMAIN_RANGE = {}

def _dr(rel: str, *pairs: tuple[frozenset[str], frozenset[str]]) -> None:
    _D[_canonical(rel)] = [(frozenset(_canonical(e) for e in d),
                            frozenset(_canonical(e) for e in r))
                           for d, r in pairs]

# -- ACT stage --
_dr("prov:wasAssociatedWith",
    ({"fflo:AdulterationAct"}, {"fflo:FoodBusinessOperator"}))
_dr("prov:used",
    ({"fflo:AdulterationAct"}, {"fkg:Food"}))
_dr("prov:wasGeneratedBy",
    ({"fflo:Adulterant"}, {"fflo:AdulterationAct"}),
    ({"fflo:SpreadEvent"}, {"fflo:TransformationEvent"}))
_dr("prov:wasAttributedTo",
    ({"fflo:IncidentFinding"}, {"fflo:FoodBusinessOperator"}))
_dr("fflo:hasAdulterant",
    ({"fkg:Food"}, {"fflo:Adulterant", "fflo:SyntheticAdulterant",
                     "fflo:NaturalAdulterant", "fflo:ContaminantAdulterant"}))
_dr("fflo:hasAdulterationMethod",
    ({"fkg:Food"}, {"fflo:AdulterationMethod"}))
_dr("fflo:hasFraudType",
    ({"fflo:AdulterationMethod"}, {"fflo:FraudType"}))
_dr("fflo:hasIntentionality",
    ({"fflo:AdulterationMethod"}, {"fflo:IntentionalityLevel"}))
_dr("fflo:isSubstituteFor",
    ({"fflo:Adulterant", "fflo:SyntheticAdulterant",
      "fflo:NaturalAdulterant", "fflo:ContaminantAdulterant"}, {"fkg:Food"}))
_dr("fflo:commonIn",
    ({"fflo:Adulterant", "fflo:SyntheticAdulterant",
      "fflo:NaturalAdulterant", "fflo:ContaminantAdulterant"},
     {"fflo:GeographicRegion"}))
_dr("fflo:foundAt",
    ({"fflo:Adulterant", "fflo:SyntheticAdulterant",
      "fflo:NaturalAdulterant", "fflo:ContaminantAdulterant"},
     {"fflo:SupplyChainStep", "fflo:Production", "fflo:Processing",
      "fflo:Storage", "fflo:Distribution", "fflo:Retail", "fflo:Import"}))

# -- ACT stage v2 additions --
_dr("fflo:hasFunction",
    ({"fflo:FoodAdditive"},
     {"fflo:AdditiveFunction", "fflo:Preservative", "fflo:Antioxidant",
      "fflo:Emulsifier", "fflo:Stabilizer", "fflo:AcidityRegulator",
      "fflo:FlourTreatmentAgent", "fflo:Sequestrant",
      "fflo:HumectantAdditive", "fflo:Colourant", "fflo:SweeteningAgent"}))
_dr("fkg:hasIngredient",
    ({"fkg:Food"}, {"fkg:Ingredient"}))

# -- SPREAD stage --
_dr("prov:wasInfluencedBy",
    ({"fflo:SpreadEvent"}, {"fflo:AdulterationAct"}))
_dr("fflo:propagatesTo",
    ({"fflo:SupplyChainStep", "fflo:Production", "fflo:Processing",
      "fflo:Storage", "fflo:Distribution", "fflo:Retail", "fflo:Import"},
     {"fflo:SupplyChainStep", "fflo:Production", "fflo:Processing",
      "fflo:Storage", "fflo:Distribution", "fflo:Retail", "fflo:Import"}))
_dr("fflo:carriedBy",
    ({"fflo:SpreadEvent"}, {"fkg:Food"}))
_dr("fflo:occursAt",
    ({"fflo:SpreadEvent"}, {"fflo:SupplyChainStep", "fflo:Production",
      "fflo:Processing", "fflo:Storage", "fflo:Distribution",
      "fflo:Retail", "fflo:Import"}))
_dr("fflo:inputTo",
    ({"fkg:Food"}, {"fflo:TransformationEvent"}))
_dr("fflo:outputOf",
    ({"fkg:Food"}, {"fflo:TransformationEvent"}))

# -- DETECTION stage --
_dr("fso:isPerformedOn",
    ({"fso:Analysis"}, {"fso:Sample"}))
_dr("fso:isPerformedAt",
    ({"fso:Analysis"}, {"fso:Laboratory"}))
_dr("fso:isResultOf",
    ({"fso:AnalysisResult"}, {"fso:Analysis"}))
_dr("fso:relatesToProperty",
    ({"fso:AnalysisResult"}, {"ssn:Property"}))
_dr("fso:isMeasuredIn",
    ({"fso:Measurement"}, {"fso:UnitOfMeasure"}))
_dr("fso:hasSampleType",
    ({"fso:Sample"}, {"fso:SampleType"}))
_dr("fso:hasLocation",
    ({"fso:Sample"}, {"fso:Location"}))

# -- DETECTION v2 additions --
_dr("fflo:hasNumericValue",
    ({"fso:Measurement"}, frozenset()))  # xsd:decimal, open-ended

_dr("fflo:identifies",
    ({"fflo:IncidentFinding", "fflo:LabConfirmed", "fflo:FieldDetected",
      "fflo:SurveyAggregated", "fflo:RecallTriggered"},
     {"fflo:Adulterant", "fflo:SyntheticAdulterant",
      "fflo:NaturalAdulterant", "fflo:ContaminantAdulterant"}))
_dr("fflo:foundIn",
    ({"fflo:IncidentFinding", "fflo:LabConfirmed", "fflo:FieldDetected",
      "fflo:SurveyAggregated", "fflo:RecallTriggered"}, {"fkg:Food"}))
_dr("fflo:producedBy",
    ({"fflo:IncidentFinding", "fflo:LabConfirmed", "fflo:FieldDetected",
      "fflo:SurveyAggregated", "fflo:RecallTriggered"},
     {"fflo:DetectionMethod", "fflo:LaboratoryTest",
      "fflo:FieldTest", "fflo:SensoryTest"}))
_dr("fflo:confirmedBy",
    ({"fflo:IncidentFinding", "fflo:LabConfirmed"}, {"fso:AnalysisResult"}))
_dr("fflo:supportedBy",
    ({"fflo:IncidentFinding", "fflo:LabConfirmed", "fflo:FieldDetected",
      "fflo:SurveyAggregated", "fflo:RecallTriggered"},
     {"fflo:EvidenceSource"}))
_dr("fflo:foundAtStep",
    ({"fflo:IncidentFinding", "fflo:LabConfirmed", "fflo:FieldDetected",
      "fflo:SurveyAggregated", "fflo:RecallTriggered"},
     {"fflo:SupplyChainStep", "fflo:Production", "fflo:Processing",
      "fflo:Storage", "fflo:Distribution", "fflo:Retail", "fflo:Import"}))
_dr("fflo:inRegion",
    ({"fflo:IncidentFinding", "fflo:LabConfirmed", "fflo:FieldDetected",
      "fflo:SurveyAggregated", "fflo:RecallTriggered"},
     {"fflo:GeographicRegion"}))
_dr("fflo:triggeredAction",
    ({"fflo:IncidentFinding", "fflo:LabConfirmed", "fflo:FieldDetected",
      "fflo:SurveyAggregated", "fflo:RecallTriggered"},
     {"fflo:RegulatoryAction"}))
_dr("fflo:collectedAt",
    ({"fso:Sample"}, {"fflo:SupplyChainStep", "fflo:Production",
      "fflo:Processing", "fflo:Storage", "fflo:Distribution",
      "fflo:Retail", "fflo:Import"}))
_dr("fflo:detectedBy",
    ({"fflo:Adulterant", "fflo:SyntheticAdulterant",
      "fflo:NaturalAdulterant", "fflo:ContaminantAdulterant"},
     {"fflo:DetectionMethod", "fflo:LaboratoryTest",
      "fflo:FieldTest", "fflo:SensoryTest"}))
_dr("fflo:producesIndicator",
    ({"fflo:DetectionMethod", "fflo:LaboratoryTest",
      "fflo:FieldTest", "fflo:SensoryTest"}, {"fflo:DetectionIndicator"}))
_dr("fflo:requiresKit",
    ({"fflo:DetectionMethod", "fflo:LaboratoryTest",
      "fflo:FieldTest", "fflo:SensoryTest"}, {"fflo:DetectionKit"}))
_dr("fflo:isPerformedAs",
    ({"fflo:LaboratoryTest"}, {"fso:Analysis"}))

# -- REGULATION stage --
_dr("lkif:created_by",
    ({"fflo:FoodStandard"}, {"fflo:RegulatoryBody"}))
_dr("fflo:belongsToCategory",
    ({"fkg:Food"}, {"fflo:FoodCategory"}))
_dr("fflo:appliesToCategory",
    ({"fflo:FoodStandard"}, {"fflo:FoodCategory"}))
_dr("fflo:hasValue",
    ({"fflo:PermissibleLimit"}, {"fso:Measurement"}))
_dr("fflo:forProperty",
    ({"fflo:PermissibleLimit"}, {"ssn:Property"}))
_dr("fflo:comparedAgainst",
    ({"fso:AnalysisResult"}, {"fflo:PermissibleLimit"}))
_dr("fflo:definedIn",
    ({"fflo:FoodStandard"}, {"fflo:RegulatoryDocument"}))
_dr("fflo:appliesTo",
    ({"fflo:PermissibleLimit"}, {"fflo:Adulterant",
      "fflo:SyntheticAdulterant", "fflo:NaturalAdulterant",
      "fflo:ContaminantAdulterant"}))
_dr("fflo:inFood",
    ({"fflo:PermissibleLimit"}, {"fkg:Food"}))
_dr("fflo:specifiedIn",
    ({"fflo:PermissibleLimit"}, {"fflo:FoodStandard", "fflo:TableReference"}))

# -- REGULATION v2 additions --
_dr("fflo:defines",
    ({"fflo:RegulatoryDocument"}, {"fflo:FoodCategory", "fflo:FoodStandard"}))
_dr("fflo:hasPermissibleLimit",
    ({"fkg:Food", "fkg:Ingredient"}, {"fflo:PermissibleLimit"}))
_dr("fflo:appliesToFood",
    ({"fflo:FoodStandard", "fflo:RegulatoryDocument"}, {"fkg:Food"}))
_dr("fflo:compliesWith",
    ({"fflo:FoodBusinessOperator", "fkg:Food"},
     {"fflo:FoodStandard", "fflo:RegulatoryDocument"}))
_dr("fflo:amends",
    ({"fflo:RegulatoryDocument"}, {"fflo:RegulatoryDocument"}))
_dr("fflo:hasSubcategory",
    ({"fflo:FoodCategory"}, {"fflo:FoodCategory"}))
_dr("fflo:hasDefinition",
    ({"fflo:FoodCategory", "fflo:Adulterant", "fflo:SyntheticAdulterant",
      "fflo:NaturalAdulterant", "fflo:ContaminantAdulterant"}, frozenset()))
_dr("fflo:tableLabel",
    ({"fflo:TableReference"}, frozenset()))  # xsd:string, open-ended
_dr("fflo:issuedBy",
    ({"fflo:RegulatoryAction"}, {"fflo:RegulatoryBody"}))
_dr("fflo:targets",
    ({"fflo:RegulatoryAction"}, {"fkg:Food"}))
_dr("fflo:citesFinding",
    ({"fflo:RegulatoryAction"}, {"fflo:IncidentFinding",
      "fflo:LabConfirmed", "fflo:FieldDetected",
      "fflo:SurveyAggregated", "fflo:RecallTriggered"}))
_dr("fflo:constitutes",
    ({"fflo:IncidentFinding", "fflo:LabConfirmed", "fflo:FieldDetected",
      "fflo:SurveyAggregated", "fflo:RecallTriggered"},
     {"fflo:Violation"}))
_dr("fflo:violates",
    ({"fflo:Violation"}, {"fflo:PermissibleLimit"}))

# -- HEALTH stage --
_dr("fflo:causesEffect",
    ({"fflo:Adulterant", "fflo:SyntheticAdulterant",
      "fflo:NaturalAdulterant", "fflo:ContaminantAdulterant"},
     {"fflo:HealthEffect", "fflo:CausalHealthEffect",
      "fflo:AcuteEffect", "fflo:ChronicEffect"}))
_dr("fflo:affectsPopulation",
    ({"fflo:HealthEffect", "fflo:CausalHealthEffect",
      "fflo:AcuteEffect", "fflo:ChronicEffect"},
     {"fflo:VulnerablePopulation"}))
_dr("fflo:hasEffectType",
    ({"fflo:HealthEffect", "fflo:CausalHealthEffect"},
     {"fflo:AcuteEffect", "fflo:ChronicEffect"}))
_dr("fflo:associatedWith",
    ({"fflo:HealthEffect", "fflo:CausalHealthEffect",
      "fflo:AcuteEffect", "fflo:ChronicEffect"},
     {"fflo:IncidentFinding", "fflo:LabConfirmed", "fflo:FieldDetected",
      "fflo:SurveyAggregated", "fflo:RecallTriggered"}))

# Clean up helper
del _dr

REQUIRED_TRIPLET_FIELDS = (
    "subject", "subject_type", "predicate", "object", "object_type",
    "confidence", "evidence_span",
)

# ---------------------------------------------------------------------------
# Triplex prompt helpers — entity types & relations for the Triplex
# knowledge-graph model.  We send *canonical* (unprefixed) names because
# Triplex outputs ``E0, TYPE: value`` and colons inside names would break
# parsing.  ``_CANONICAL_TO_PREFIXED_*`` maps canonical names back to the
# full ontology term after parsing.
# ---------------------------------------------------------------------------

_TRIPLEX_ENTITY_TYPES_PREFIXED: list[str] = [
    # ACT stage
    "fflo:FoodBusinessOperator", "fflo:AdulterationAct",
    "fflo:Adulterant", "fflo:SyntheticAdulterant", "fflo:NaturalAdulterant",
    "fflo:ContaminantAdulterant", "fflo:AdulterationMethod",
    "fflo:FraudType", "fflo:IntentionalityLevel",
    "fkg:Food", "fkg:Ingredient", "fkg:ChemicalIngredient",
    "fflo:FoodAdditive",
    "fflo:AdditiveFunction",
    "fflo:Preservative", "fflo:Antioxidant", "fflo:Emulsifier",
    "fflo:Stabilizer", "fflo:AcidityRegulator",
    "fflo:FlourTreatmentAgent", "fflo:Sequestrant",
    "fflo:HumectantAdditive", "fflo:Colourant", "fflo:SweeteningAgent",
    # SPREAD stage
    "fflo:SupplyChainStep",
    "fflo:Production", "fflo:Processing", "fflo:Storage",
    "fflo:Distribution", "fflo:Retail", "fflo:Import",
    "fflo:SpreadEvent", "fflo:TransformationEvent",
    "fflo:GeographicRegion",
    # DETECTION stage — FSO/SOSA
    "fso:Sample", "fso:SampleType", "fso:Analysis", "fso:AnalysisResult",
    "ssn:Property",
    "fso:Measurement", "fso:UnitOfMeasure", "fso:Laboratory", "fso:Location",
    # DETECTION stage — FFLO Incident
    "fflo:IncidentFinding", "fflo:LabConfirmed", "fflo:FieldDetected",
    "fflo:SurveyAggregated", "fflo:RecallTriggered",
    "fflo:EvidenceSource",
    # DETECTION stage — FFLO Methods
    "fflo:DetectionMethod", "fflo:LaboratoryTest",
    "fflo:FieldTest", "fflo:SensoryTest",
    "fflo:DetectionIndicator", "fflo:DetectionKit",
    # REGULATION stage — LKIF
    "fflo:FoodStandard", "fflo:RegulatoryDocument",
    "fflo:RegulatoryBody", "fflo:RegulatoryAction",
    "fflo:PermissibleLimit", "fflo:Violation",
    # REGULATION stage — FFLO
    "fflo:FoodCategory", "fflo:TableReference",
    # HEALTH stage — OAE
    "fflo:HealthEffect", "fflo:CausalHealthEffect",
    "fflo:AcuteEffect", "fflo:ChronicEffect",
    "fflo:VulnerablePopulation",
]

_TRIPLEX_RELATIONS_PREFIXED: list[str] = [
    # ACT stage — PROV-O
    "prov:wasAssociatedWith", "prov:used",
    "prov:wasGeneratedBy", "prov:wasAttributedTo",
    # ACT stage — FFLO
    "fflo:hasAdulterant", "fflo:hasAdulterationMethod",
    "fflo:hasFraudType", "fflo:hasIntentionality",
    "fflo:isSubstituteFor", "fflo:commonIn", "fflo:foundAt",
    # ACT stage — FFLO v2 additions
    "fflo:hasFunction", "fkg:hasIngredient",
    # SPREAD stage — PROV-O
    "prov:wasInfluencedBy",
    # SPREAD stage — FFLO
    "fflo:propagatesTo", "fflo:carriedBy", "fflo:occursAt",
    "fflo:inputTo", "fflo:outputOf",
    # DETECTION stage — FSO
    "fso:isPerformedOn", "fso:isPerformedAt", "fso:isResultOf",
    "fso:relatesToProperty", "fso:isMeasuredIn",
    "fso:hasSampleType", "fso:hasLocation",
    # DETECTION stage — FFLO v2 additions
    "fflo:hasNumericValue",
    # DETECTION stage — FFLO Incident
    "fflo:identifies", "fflo:foundIn", "fflo:producedBy",
    "fflo:confirmedBy", "fflo:supportedBy",
    "fflo:foundAtStep", "fflo:inRegion", "fflo:triggeredAction",
    "fflo:collectedAt",
    # DETECTION stage — FFLO Methods
    "fflo:detectedBy", "fflo:producesIndicator",
    "fflo:requiresKit", "fflo:isPerformedAs",
    # REGULATION stage — LKIF
    "lkif:created_by",
    # REGULATION stage — FFLO bridges
    "fflo:belongsToCategory", "fflo:appliesToCategory",
    "fflo:hasValue", "fflo:forProperty", "fflo:comparedAgainst",
    # REGULATION stage — FFLO structure
    "fflo:definedIn", "fflo:appliesTo", "fflo:inFood",
    "fflo:specifiedIn", "fflo:issuedBy", "fflo:targets",
    "fflo:citesFinding", "fflo:constitutes", "fflo:violates",
    # REGULATION stage — FFLO v2 additions
    "fflo:defines", "fflo:hasPermissibleLimit",
    "fflo:appliesToFood", "fflo:compliesWith",
    "fflo:amends", "fflo:hasSubcategory",
    "fflo:hasDefinition", "fflo:tableLabel",
    # HEALTH stage — FFLO
    "fflo:causesEffect", "fflo:affectsPopulation",
    "fflo:hasEffectType", "fflo:associatedWith",
]

_TRIPLEX_ENTITY_NAMES = [_canonical(n) for n in _TRIPLEX_ENTITY_TYPES_PREFIXED]
_TRIPLEX_RELATION_NAMES = [_canonical(r)
                          for r in _TRIPLEX_RELATIONS_PREFIXED]

_CANONICAL_TO_PREFIXED_ENTITY: dict[str, str] = dict(
    zip(_TRIPLEX_ENTITY_NAMES, _TRIPLEX_ENTITY_TYPES_PREFIXED)
)
_CANONICAL_TO_PREFIXED_RELATION: dict[str, str] = dict(
    zip(_TRIPLEX_RELATION_NAMES, _TRIPLEX_RELATIONS_PREFIXED)
)

# Keep in sync guard (fires at import time if lists diverge)
assert set(_TRIPLEX_ENTITY_NAMES) == set(VALID_ENTITY_TYPES), \
    "Triplex entity types out of sync with VALID_ENTITY_TYPES"
assert set(_TRIPLEX_RELATION_NAMES) == set(VALID_RELATIONS), \
    "Triplex relations out of sync with VALID_RELATIONS"

del _TRIPLEX_ENTITY_NAMES, _TRIPLEX_RELATION_NAMES  # only used for the assert


def validate_triplet(
    triplet: dict[str, Any],
    triplet_index: int,
    snippet_id: str,
    source_id: str,
    source_file: str,
    chunk_index: str,
) -> list[dict[str, Any]]:
    """Validate one triplet against the schema.  Returns a list of violations
    (empty list = fully valid).

    Entity type and relation names are canonicalised (prefixes stripped) before
    comparison, so the model may return either "fflo:Adulterant" or "Adulterant".
    """
    violations: list[dict[str, Any]] = []

    def _violation(vtype: str, field: str, actual: object, allowed: object | None = None) -> dict[str, Any]:
        return {
            "snippet_id": snippet_id,
            "source_id": source_id,
            "source_file": source_file,
            "chunk_index": chunk_index,
            "triplet_index": triplet_index,
            "violation_type": vtype,
            "field_name": field,
            "actual_value": actual,
            "allowed_values": sorted(allowed) if isinstance(allowed, (set, frozenset)) else allowed,
            "triplet": triplet,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # 1. Required fields present and non-empty
    for field in REQUIRED_TRIPLET_FIELDS:
        val = triplet.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            violations.append(_violation("missing_field", field, val))
            return violations  # can't validate further without core fields

    # 2. Entity types (canonicalised)
    for role in ("subject_type", "object_type"):
        etype = _canonical(triplet.get(role, ""))
        if etype and etype not in VALID_ENTITY_TYPES:
            violations.append(_violation(
                "unknown_entity_type", role, triplet.get(role, ""),
                allowed=sorted(VALID_ENTITY_TYPES),
            ))

    # 3. Relation (canonicalised)
    pred_raw = triplet.get("predicate", "")
    pred = _canonical(pred_raw)
    if pred and pred not in VALID_RELATIONS:
        violations.append(_violation(
            "unknown_relation", "predicate", pred_raw,
            allowed=sorted(VALID_RELATIONS),
        ))
        return violations  # can't check domain/range without a known relation

    # 4. Domain / range (supports multiple valid pairs per relation)
    pairs = RELATION_DOMAIN_RANGE.get(pred)
    if pairs:
        stype = _canonical(triplet.get("subject_type", ""))
        otype = _canonical(triplet.get("object_type", ""))

        domain_ok = any(
            (not valid_domain or stype in valid_domain)
            for valid_domain, _valid_range in pairs
        )
        range_ok = any(
            (not valid_range or otype in valid_range)
            for _valid_domain, valid_range in pairs
        )

        if not domain_ok:
            all_domains = sorted({e for d, _ in pairs for e in d})
            violations.append(_violation(
                "domain_mismatch", "subject_type", triplet.get("subject_type", ""),
                allowed=all_domains,
            ))
        if not range_ok:
            all_ranges = sorted({e for _, r in pairs for e in r})
            violations.append(_violation(
                "range_mismatch", "object_type", triplet.get("object_type", ""),
                allowed=all_ranges,
            ))

    # 5. Confidence sanity
    conf = triplet.get("confidence")
    if conf is not None:
        try:
            cf = float(conf)
            if cf < 0.0 or cf > 1.0:
                violations.append(_violation("invalid_confidence", "confidence", conf))
        except (TypeError, ValueError):
            violations.append(_violation("invalid_confidence", "confidence", conf))

    return violations


# ---------------------------------------------------------------------------
# Build system prompt (schema embedded for resilience)
# ---------------------------------------------------------------------------

def build_system_prompt(prompt_file: Path) -> str:
    """Read user's prompt template and append schema hints."""
    base = prompt_file.read_text(encoding="utf-8").strip()
    return base


def _derive_csv_id(chunks_csv: Path) -> str:
    """Derive a short dataset identifier from the CSV path.

    Walks path components from right to left, skipping generic directory
    names ("processed", "chunks", "data") and uses the first meaningful
    directory it finds.  Falls back to the CSV stem.
    """
    skip = frozenset({"processed", "chunks", "data", "output", "outputs"})
    parts = list(chunks_csv.resolve().parts)
    # Walk backwards from the CSV file, skipping skip-words and the filename
    for i in range(len(parts) - 1, -1, -1):
        part = parts[i]
        if part.lower() in skip:
            continue
        if part.endswith(".csv"):
            continue
        candidate = re.sub(r"[^a-zA-Z0-9_-]+", "_", part).strip("_").lower()
        if candidate:
            return candidate
    # Last resort
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", chunks_csv.stem).strip("_").lower() or "corpus"


def _model_slug(model: str) -> str:
    """Filesystem-safe model name."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", model).strip("-").lower()


# ---------------------------------------------------------------------------
# Chunk context building
# ---------------------------------------------------------------------------

def load_chunks(csv_path: Path) -> list[dict[str, str]]:
    """Load chunk CSV, return list of rows keyed by column name."""
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def build_context(
    rows: list[dict[str, str]],
    current_idx: int,
) -> tuple[str, str]:
    """Build a user message with surrounding chunks for continuity.

    Returns (user_message, target_snippet_id).
    """
    row = rows[current_idx]
    source_id = row["source_id"]
    snippet_id = row["snippet_id"]
    target_text = row["evidence_text"]

    # Collect previous chunks from same source
    prev_texts: list[str] = []
    idx = current_idx - 1
    while idx >= 0 and len(prev_texts) < MAX_CONTEXT_CHUNKS_PREV:
        if rows[idx]["source_id"] == source_id:
            prev_texts.insert(
                0,
                f"[CHUNK {rows[idx]['chunk_index']} — {rows[idx]['snippet_id']}]\n"
                f"{rows[idx]['evidence_text']}",
            )
        idx -= 1

    # Collect next chunks from same source
    next_texts: list[str] = []
    idx = current_idx + 1
    while idx < len(rows) and len(next_texts) < MAX_CONTEXT_CHUNKS_NEXT:
        if rows[idx]["source_id"] == source_id:
            next_texts.append(
                f"[CHUNK {rows[idx]['chunk_index']} — {rows[idx]['snippet_id']}]\n"
                f"{rows[idx]['evidence_text']}",
            )
        idx += 1

    # Assemble
    parts: list[str] = []

    if prev_texts:
        parts.append("=== CONTEXT (previous chunks — for continuity, do NOT extract from these) ===")
        parts.extend(prev_texts)

    parts.append("=== TARGET CHUNK (extract triplets ONLY from this) ===")
    parts.append(
        f"snippet_id: {snippet_id}\n"
        f"source_file: {row['source_file']}\n"
        f"source_type: {row['source_type']}\n"
        f"chunk_index: {row['chunk_index']}\n\n"
        f"{target_text}"
    )
    parts.append("=== END TARGET CHUNK ===")

    if next_texts:
        parts.append("=== CONTEXT (next chunk — for continuity, do NOT extract from this) ===")
        parts.extend(next_texts)

    return "\n\n".join(parts), snippet_id


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

class ExtractionError(Exception):
    """Non-retryable extraction failure (bad request, parse error, etc.)."""


class RetryableError(Exception):
    """Transient failure worth retrying (rate limit, server error, timeout)."""


async def call_deepseek(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    snippet_id: str,
    timeout: float,
) -> dict[str, Any]:
    """Send one chat-completion request and return parsed JSON."""
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
        "max_tokens": 8192,
        "response_format": {"type": "json_object"},
    }

    response = await client.post(
        url, json=payload, headers=headers, timeout=httpx.Timeout(timeout)
    )

    if response.status_code == 401:
        body = response.text[:500]
        raise ExtractionError(f"Authentication failed. Check DEEPSEEK_API_KEY. {body}")

    if response.status_code == 400:
        body = response.text[:500]
        raise ExtractionError(f"Bad request (400). {body}")

    if response.status_code in RETRYABLE_STATUSES:
        raise RetryableError(
            f"HTTP {response.status_code}: {response.text[:300]}"
        )

    if response.status_code != 200:
        raise ExtractionError(
            f"Unexpected HTTP {response.status_code}: {response.text[:500]}"
        )

    data = response.json()
    choice = data.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")

    if not content:
        finish = choice.get("finish_reason", "unknown")
        raise ExtractionError(
            f"Empty model response (finish_reason={finish}). "
            f"Tokens: prompt={data.get('usage', {}).get('prompt_tokens')}, "
            f"completion={data.get('usage', {}).get('completion_tokens')}"
        )

    # Try to parse as JSON; strip markdown fences if present
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"Model returned unparseable JSON: {exc}\n"
            f"Raw (first 800 chars): {content[:800]}"
        )

    if not isinstance(parsed, dict):
        raise ExtractionError(
            f"Expected JSON object, got {type(parsed).__name__}: "
            f"{json.dumps(parsed)[:500]}"
        )

    parsed.setdefault("snippet_id", snippet_id)
    parsed.setdefault("triplets", [])
    parsed.setdefault("notes", "")

    return parsed


# ---------------------------------------------------------------------------
# HuggingFace local-inference backend
# ---------------------------------------------------------------------------

def _resolve_hf_device(requested: str) -> str:
    """Map ``"auto"`` → first available device; passthrough explicit values."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_hf_model(
    model_name: str,
    device: str,
    dtype: str,
    cache_dir: str | None,
    quantize: str = "none",
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Download (if needed) and load a HuggingFace CausalLM model once.

    ``quantize`` may be ``"none"``, ``"4bit"``, or ``"8bit"``.
    4-bit / 8-bit require ``bitsandbytes`` to be installed.

    ``trust_remote_code`` is required for models with custom config /
    modeling code (e.g. Triplex / Phi-3 based variants).
    """
    import torch
    from transformers import AutoTokenizer as _AT, AutoModelForCausalLM as _AM

    dtype_map: dict[str, Any] = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, "auto")

    quantize_cfg = None
    if quantize in ("4bit", "8bit"):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "Quantization requires bitsandbytes."
                "  Install with: pip install bitsandbytes"
            )
        if quantize == "4bit":
            quantize_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantize_cfg = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = _AT.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
    }
    if quantize_cfg is not None:
        model_kwargs["quantization_config"] = quantize_cfg
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch_dtype
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = None

    model = _AM.from_pretrained(model_name, **model_kwargs)

    if quantize_cfg is None and device not in ("auto", None):
        model = model.to(device)
    model.eval()
    return model, tokenizer


def _call_hf_model(
    hf_model: Any,
    hf_tokenizer: Any,
    hf_device: str,
    system_prompt: str,
    user_message: str,
    snippet_id: str,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Generate structured triplets with a locally loaded HF model.

    Mirrors the JSON-parsing and error-handling behaviour of
    ``call_deepseek`` so the rest of the pipeline stays identical.
    """
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    # Prefer the tokenizer's chat template; fall back to plain concatenation
    try:
        prompt: str = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        prompt = f"{system_prompt}\n\n{user_message}"

    inputs = hf_tokenizer(prompt, return_tensors="pt")
    if hf_device != "auto":
        inputs = {k: v.to(hf_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            pad_token_id=hf_tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    text = hf_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # --- Same JSON-extraction logic as call_deepseek ---
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"Model returned unparseable JSON: {exc}\n"
            f"Raw (first 800 chars): {text[:800]}"
        )

    if not isinstance(parsed, dict):
        raise ExtractionError(
            f"Expected JSON object, got {type(parsed).__name__}: "
            f"{json.dumps(parsed)[:500]}"
        )

    parsed.setdefault("snippet_id", snippet_id)
    parsed.setdefault("triplets", [])
    parsed.setdefault("notes", "")
    return parsed


# ---------------------------------------------------------------------------
# HuggingFace Triplex-specific backend
# ---------------------------------------------------------------------------

def _build_triplex_prompt(chunk_text: str) -> str:
    """Build a Triplex-native prompt with FFLO entity types and relations."""
    entity_json = json.dumps(
        {"entity_types": _TRIPLEX_ENTITY_TYPES_PREFIXED},
        ensure_ascii=False,
    )
    relations_json = json.dumps(
        {"predicates": _TRIPLEX_RELATIONS_PREFIXED},
        ensure_ascii=False,
    )
    return (
        "Perform Named Entity Recognition (NER) and extract knowledge "
        "graph triplets from the text. NER identifies named entities of "
        "given entity types, and triple extraction identifies relationships "
        "between entities using specified predicates. Return the result as "
        'a JSON object with an "entities_and_triples" key containing an '
        "array of entities and triples.\n\n"
        "**Entity Types:**\n"
        f"{entity_json}\n\n"
        "**Predicates:**\n"
        f"{relations_json}\n\n"
        "**Text:**\n"
        f"{chunk_text}"
    )


def _parse_triplex_output(
    raw_output: str, snippet_id: str, chunk_text: str
) -> dict[str, Any]:
    """Parse Triplex's ``entities_and_triples`` into standard triplet format.

    Triplex output looks like::

        {"entities_and_triples": [
            "E0, LOCATION: San Francisco",
            "E1, NUMBER: 808437",
            "E0, POPULATION, E1",
        ]}

    Entity-definition strings are ``E<id>, TYPE: value``.
    Triplet strings are ``E<id>, PREDICATE, E<id>``.

    Because Triplex uses a colon to separate type from value, we send
    *prefixed* names and parse the FULL type name including the prefix.
    """
    text = raw_output.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    data: dict[str, Any] | None = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```",
                      raw_output, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    if data is None:
        raise ExtractionError(
            f"Triplex returned unparseable JSON:\n{text[:800]}"
        )

    items = data.get("entities_and_triples", [])
    if not isinstance(items, list):
        raise ExtractionError(
            "Triplex output missing 'entities_and_triples' array"
        )

    entities: dict[str, tuple[str, str]] = {}  # E0 → (type, value)
    triplets: list[dict[str, Any]] = []

    for item in items:
        if not isinstance(item, str):
            continue
        item = item.strip()

        # Split on ", " — entities have 2 comma-separated parts
        # (E0, TYPE: value), triplets have 3 (E0, PREDICATE, E1).
        # TYPE may contain colons (fkg:Food) so we use rfind(": ")
        # on the remainder rather than a naive regex.
        parts = item.split(", ")
        if len(parts) == 3 and re.match(r"^E\d+$", parts[2].strip()):
            # Triplet: E0, PREDICATE, E1
            subj_id = parts[0].strip()
            pred = parts[1].strip()
            obj_id = parts[2].strip()
            subj = entities.get(subj_id, ("", subj_id))
            obj = entities.get(obj_id, ("", obj_id))
            triplets.append({
                "subject": subj[1],
                "subject_type": subj[0],
                "predicate": pred,
                "object": obj[1],
                "object_type": obj[0],
                "confidence": 0.8,
                "evidence_span": "",
            })
        elif len(parts) >= 2:
            # Entity: E0, TYPE: value
            # TYPE may contain colons — use last ": " as separator
            eid = parts[0].strip()
            rest = ", ".join(parts[1:])
            last_colon = rest.rfind(": ")
            if last_colon > 0:
                etype_raw = rest[:last_colon].strip()
                evalue = rest[last_colon + 2:].strip()
                entities[eid] = (etype_raw, evalue)

    return {
        "snippet_id": snippet_id,
        "triplets": triplets,
        "notes": (
            f"Triplex: {len(triplets)} triplets, "
            f"{len(entities)} entities"
        ),
    }


def _call_hf_triplex(
    hf_model: Any,
    hf_tokenizer: Any,
    hf_device: str,
    chunk_text: str,
    snippet_id: str,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Extract triplets with Triplex's native prompt format.

    Builds a Triplex-specific prompt (entity types + predicates as JSON
    arrays), generates, and parses ``entities_and_triples`` back into the
    standard ``{snippet_id, triplets, notes}`` shape.
    """
    import torch

    prompt = _build_triplex_prompt(chunk_text)
    messages = [{"role": "user", "content": prompt}]

    try:
        formatted: str = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        formatted = prompt

    inputs = hf_tokenizer(formatted, return_tensors="pt")
    if hf_device != "auto":
        inputs = {k: v.to(hf_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            pad_token_id=hf_tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    text = hf_tokenizer.decode(
        generated_ids, skip_special_tokens=True,
    ).strip()

    return _parse_triplex_output(text, snippet_id, chunk_text)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def process_one(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient | None,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    rows: list[dict[str, str]],
    idx: int,
    timeout: float,
    output_jsonl: Path,
    failures_jsonl: Path,
    violations_jsonl: Path,
    total: int,
    *,
    backend: str = "openai_compat",
    hf_model: Any = None,
    hf_tokenizer: Any = None,
    hf_device: str = "cpu",
    hf_max_new_tokens: int = 4096,
    hf_temperature: float = 0.0,
    hf_prompt_format: str = "standard",
) -> tuple[int, str]:
    """Process a single chunk row with retries + schema validation.

    Dispatches to ``call_deepseek`` (openai_compat), ``_call_hf_model``
    (hf_transformers with standard prompt), or ``_call_hf_triplex``
    (hf_transformers with triplex prompt) depending on the *backend* and
    *hf_prompt_format* parameters.
    """
    async with sem:
        user_msg, snippet_id = build_context(rows, idx)
        row = rows[idx]
        source_id = row["source_id"]
        source_file = row["source_file"]
        chunk_index = row["chunk_index"]
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if backend == "hf_transformers":
                    if hf_prompt_format == "triplex":
                        result = await asyncio.to_thread(
                            _call_hf_triplex,
                            hf_model, hf_tokenizer, hf_device,
                            row["evidence_text"], snippet_id,
                            hf_max_new_tokens, hf_temperature,
                        )
                    else:
                        result = await asyncio.to_thread(
                            _call_hf_model,
                            hf_model, hf_tokenizer, hf_device,
                            system_prompt, user_msg, snippet_id,
                            hf_max_new_tokens, hf_temperature,
                        )
                else:
                    result = await call_deepseek(
                        client, base_url, api_key, model,
                        system_prompt, user_msg, snippet_id, timeout,
                    )

                # --- Schema validation ---
                triplet_list = result.get("triplets", [])
                violation_count = 0
                for ti, triplet in enumerate(triplet_list):
                    vios = validate_triplet(
                        triplet, ti, snippet_id,
                        source_id, source_file, chunk_index,
                    )
                    for v in vios:
                        _append_jsonl(violations_jsonl, v)
                        violation_count += 1

                # Write result immediately
                _append_jsonl(output_jsonl, result)
                done = _count_jsonl(output_jsonl)
                parts = [f"  [{done:>5d}/{total}] OK  {snippet_id}"]
                parts.append(f"({len(triplet_list)} triplets)")
                if violation_count:
                    parts.append(f"⚠ {violation_count} schema violations")
                print("  ".join(parts))
                return idx, "ok"

            except RetryableError as exc:
                last_error = str(exc)
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    print(
                        f"  [  ·  /{total}] RETRY {attempt}/{MAX_RETRIES} "
                        f"{snippet_id}  (wait {delay:.0f}s)  {exc!r}"
                    )
                    await asyncio.sleep(delay)
                else:
                    print(
                        f"  [  ✗  /{total}] FAIL {snippet_id}  "
                        f"after {MAX_RETRIES} retries: {exc!r}"
                    )

            except ExtractionError as exc:
                last_error = str(exc)
                print(f"  [  ✗  /{total}] FAIL {snippet_id}  {exc!r}")
                break  # don't retry on bad request / parse error

        # If we get here, all attempts failed
        failure = {
            "snippet_id": snippet_id,
            "source_id": source_id,
            "source_file": source_file,
            "chunk_index": chunk_index,
            "error": last_error,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _append_jsonl(failures_jsonl, failure)
        return idx, "failed"


async def run_pipeline(
    chunks_csv: Path,
    prompt_file: Path,
    output_dir: Path,
    api_key: str,
    *,
    csv_id: str | None = None,
    backend: str = DEFAULT_BACKEND,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_TIMEOUT,
    resume: bool = False,
    retry_failures: bool = False,
    skip_until: str | None = None,
    hf_device: str = "auto",
    hf_dtype: str = "auto",
    hf_cache_dir: str | None = None,
    hf_quantize: str = DEFAULT_HF_QUANTIZE,
    hf_prompt_format: str = DEFAULT_HF_PROMPT_FORMAT,
    hf_trust_remote_code: bool = DEFAULT_HF_TRUST_REMOTE_CODE,
    hf_max_new_tokens: int = DEFAULT_HF_MAX_NEW_TOKENS,
    hf_temperature: float = DEFAULT_HF_TEMPERATURE,
) -> int:
    """Main async pipeline.

    Outputs are written to ``output_dir / {csv_id} / {model_slug} /`` (or
    ``output_dir / {csv_id} / hf__{model_slug} /`` for the HuggingFace
    backend) so multiple datasets, LLMs, and backends can share the same
    root without overwriting.
    """
    if csv_id is None:
        csv_id = _derive_csv_id(chunks_csv)
    model_slug = _model_slug(model)
    if backend == "hf_transformers":
        run_dir = output_dir / csv_id / f"hf__{model_slug}"
    else:
        run_dir = output_dir / csv_id / model_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = run_dir / "triplets.jsonl"
    failures_jsonl = run_dir / "failures.jsonl"
    violations_jsonl = run_dir / "schema_violations.jsonl"
    triplets_csv = run_dir / "triplets.csv"

    # Clear outputs unless resuming
    if not resume:
        for p in (output_jsonl, failures_jsonl, violations_jsonl, triplets_csv):
            if p.exists():
                p.unlink()

    # Load data
    print(f"Loading chunks from {chunks_csv}")
    rows = load_chunks(chunks_csv)
    print(f"  {len(rows)} chunks loaded  →  dataset={csv_id}  model={model}  dir={run_dir}")

    system_prompt = build_system_prompt(prompt_file)
    print(f"  System prompt: {len(system_prompt)} chars from {prompt_file}")

    # Determine which indices to process
    existing_ids: set[str] = set()
    if resume and output_jsonl.exists():
        with output_jsonl.open("r") as fh:
            for line in fh:
                try:
                    existing_ids.add(json.loads(line).get("snippet_id", ""))
                except json.JSONDecodeError:
                    pass

    pending: list[int] = []
    skip_found = skip_until is None
    for idx, row in enumerate(rows):
        sid = row["snippet_id"]
        if skip_until and not skip_found:
            if sid == skip_until:
                skip_found = True
            else:
                continue
        if resume and sid in existing_ids:
            continue
        pending.append(idx)

    if not pending:
        print("All chunks already processed. Nothing to do.")
        # Still produce CSV from existing JSONL
        _flatten_to_csv(output_jsonl, rows, triplets_csv)
        return 0

    # --- retry-failures filter ---
    if retry_failures:
        # Read failures from the flat output_dir first (backward compat
        # with runs that saved directly to output_dir without csv/model
        # subdirs), then fall back to the run_dir copy.
        flat_failures = output_dir / "failures.jsonl"
        fail_source = flat_failures if flat_failures.exists() else failures_jsonl
        failed_ids: set[str] = set()
        if fail_source.exists():
            with fail_source.open("r") as fh:
                for line in fh:
                    try:
                        failed_ids.add(json.loads(line).get("snippet_id", ""))
                    except json.JSONDecodeError:
                        pass
        if failed_ids:
            prev = len(pending)
            pending = [i for i in pending
                       if rows[i]["snippet_id"] in failed_ids]
            print(f"  Retry-failures: {len(failed_ids)} failures from "
                  f"{fail_source.name} → "
                  f"{len(pending)} chunks to retry (from {prev})")
        else:
            print("  Retry-failures: no failed snippet_ids found")

    print(f"  Processing {len(pending)} pending chunks "
          f"(resume={resume}, concurrency={concurrency})")

    # --- Backend-specific setup ---
    resolved_hf_device = "cpu"

    if backend == "hf_transformers":
        if not _HF_AVAILABLE:
            print("error: --backend hf_transformers requires torch + "
                  "transformers.  Install with: pip install torch",
                  file=sys.stderr)
            return 1
        if concurrency > 1:
            print("  Note: forcing concurrency=1 for hf_transformers backend")
            concurrency = 1
        resolved_hf_device = _resolve_hf_device(hf_device)
        print(f"\nLoading {model} on {resolved_hf_device} ...")
        try:
            hf_model, hf_tokenizer = _load_hf_model(
                model, resolved_hf_device, hf_dtype, hf_cache_dir,
                hf_quantize, hf_trust_remote_code,
            )
        except Exception as exc:
            print(f"error: failed to load HF model: {exc}", file=sys.stderr)
            return 1
        print("  Model loaded. Starting inference.\n")

    # Concurrent processing
    sem = asyncio.Semaphore(concurrency)

    if backend == "hf_transformers":
        # Local model — run via asyncio.to_thread (concurrency defaults to 1)
        tasks = [
            process_one(
                sem, None, base_url, api_key, model,
                system_prompt, rows, idx, timeout,
                output_jsonl, failures_jsonl, violations_jsonl, len(pending),
                backend=backend,
                hf_model=hf_model, hf_tokenizer=hf_tokenizer,
                hf_device=resolved_hf_device,
                hf_max_new_tokens=hf_max_new_tokens,
                hf_temperature=hf_temperature,
                hf_prompt_format=hf_prompt_format,
            )
            for idx in pending
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # Remote API — concurrent HTTP calls
        limits = httpx.Limits(max_keepalive_connections=concurrency + 5,
                               max_connections=concurrency + 10)
        async with httpx.AsyncClient(limits=limits) as client:
            tasks = [
                process_one(
                    sem, client, base_url, api_key, model,
                    system_prompt, rows, idx, timeout,
                    output_jsonl, failures_jsonl, violations_jsonl, len(pending),
                )
                for idx in pending
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

    # Summarise
    ok_count = sum(1 for r in results if isinstance(r, tuple) and r[1] == "ok")
    fail_count = sum(1 for r in results if isinstance(r, tuple) and r[1] == "failed")
    exc_count = sum(1 for r in results if isinstance(r, BaseException))
    violation_count = _count_jsonl(violations_jsonl)

    print(f"\n  OK: {ok_count}  Failed: {fail_count}  Exceptions: {exc_count}  Schema-violations: {violation_count}")

    if exc_count:
        for r in results:
            if isinstance(r, BaseException):
                print(f"  Unhandled exception: {r}", file=sys.stderr)

    # Flatten to CSV
    print(f"\nFlattening triplets to {triplets_csv}")
    _flatten_to_csv(output_jsonl, rows, triplets_csv)

    print(f"\nOutputs ({run_dir}/):")
    print(f"  triplets.jsonl")
    print(f"  failures.jsonl")
    print(f"  schema_violations.jsonl")
    print(f"  triplets.csv")

    return 0 if fail_count == 0 and exc_count == 0 else 1


# ---------------------------------------------------------------------------
# Flatten JSONL → CSV
# ---------------------------------------------------------------------------

def _flatten_to_csv(
    jsonl_path: Path,
    rows: list[dict[str, str]],
    out_csv: Path,
) -> None:
    """Convert per-chunk JSONL into a flat triplet CSV."""
    # Build source_id lookup
    source_lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        source_lookup[row["snippet_id"]] = row

    fieldnames = [
        "snippet_id", "source_id", "source_file", "source_type",
        "chunk_index", "subject", "subject_type", "predicate",
        "object", "object_type", "confidence", "evidence_span",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        if jsonl_path.exists():
            with jsonl_path.open("r") as jf:
                for line in jf:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    sid = obj.get("snippet_id", "")
                    src = source_lookup.get(sid, {})
                    for t in obj.get("triplets", []):
                        writer.writerow({
                            "snippet_id": sid,
                            "source_id": src.get("source_id", ""),
                            "source_file": src.get("source_file", ""),
                            "source_type": src.get("source_type", ""),
                            "chunk_index": src.get("chunk_index", ""),
                            "subject": t.get("subject", ""),
                            "subject_type": t.get("subject_type", ""),
                            "predicate": t.get("predicate", ""),
                            "object": t.get("object", ""),
                            "object_type": t.get("object_type", ""),
                            "confidence": t.get("confidence", ""),
                            "evidence_span": t.get("evidence_span", ""),
                        })

    triplet_count = _count_csv_triplets(out_csv)
    print(f"  {triplet_count} triplets written to {out_csv}")


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


def _count_csv_triplets(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r") as fh:
        reader = csv.DictReader(fh)
        for _ in reader:
            count += 1
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Async parallel triplet extraction via LLM API"
    )
    p.add_argument("--chunks-csv", type=Path, required=True,
                   help="Path to chunk CSV from build_corpus.py")
    p.add_argument("--prompt-file", type=Path, required=True,
                   help="Text file containing the system prompt template")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Root output directory (results go to output-dir/csv-id/model/)")
    p.add_argument("--csv-id", type=str, default=None,
                   help="Dataset identifier for subdirectory naming "
                        "(default: derived from CSV path, e.g. 'fssai_docs')")
    p.add_argument("--api-key", type=str,
                   default=os.environ.get("DEEPSEEK_API_KEY", "")
                           or os.environ.get("OPENAI_API_KEY", ""),
                   help="API key (or set DEEPSEEK_API_KEY / OPENAI_API_KEY env var)")
    p.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                   help=f"API base URL (default {DEFAULT_BASE_URL})")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help=f"Model name, slugified for directory naming (default {DEFAULT_MODEL})")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help=f"Max parallel requests (default {DEFAULT_CONCURRENCY})")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help="Per-request timeout in seconds")
    p.add_argument("--resume", action="store_true",
                   help="Skip snippet_ids already present in triplets.jsonl")
    p.add_argument("--retry-failures", action="store_true",
                   help="Only re-process chunks listed in failures.jsonl")
    p.add_argument("--skip-until", type=str, default=None,
                   help="Skip forward to this snippet_id before starting")
    p.add_argument("--backend", type=str, default=DEFAULT_BACKEND,
                   choices=["openai_compat", "hf_transformers"],
                   help=f"Inference backend (default: {DEFAULT_BACKEND})")
    p.add_argument("--hf-device", type=str, default="auto",
                   help="Device for --backend hf_transformers: auto|cpu|cuda|mps "
                        "(default: auto)")
    p.add_argument("--hf-dtype", type=str, default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"],
                   help="Torch dtype for HF model (default: auto)")
    p.add_argument("--hf-quantize", type=str, default=DEFAULT_HF_QUANTIZE,
                   choices=["none", "4bit", "8bit"],
                   help="Quantize HF model to reduce memory "
                        f"(default: {DEFAULT_HF_QUANTIZE}; "
                        "requires: pip install bitsandbytes)")
    p.add_argument("--hf-prompt-format", type=str,
                   default=DEFAULT_HF_PROMPT_FORMAT,
                   choices=["standard", "triplex"],
                   help="Prompt format for hf_transformers backend "
                        f"(default: {DEFAULT_HF_PROMPT_FORMAT}). "
                        "'triplex' is for the SciPhi/Triplex model")
    p.add_argument("--hf-trust-remote-code", action="store_true",
                   default=DEFAULT_HF_TRUST_REMOTE_CODE,
                   help="Pass trust_remote_code=True to HF from_pretrained "
                        "(required for Triplex and other custom-code models)")
    p.add_argument("--hf-cache-dir", type=str, default=None,
                   help="HuggingFace model cache directory")
    p.add_argument("--hf-max-new-tokens", type=int,
                   default=DEFAULT_HF_MAX_NEW_TOKENS,
                   help=f"Max new tokens for HF generation "
                        f"(default: {DEFAULT_HF_MAX_NEW_TOKENS})")
    p.add_argument("--hf-temperature", type=float,
                   default=DEFAULT_HF_TEMPERATURE,
                   help=f"Temperature for HF generation "
                        f"(default: {DEFAULT_HF_TEMPERATURE})")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.api_key and args.backend != "hf_transformers":
        print("error: no API key — set DEEPSEEK_API_KEY or OPENAI_API_KEY, "
              "or pass --api-key (not needed for --backend hf_transformers)",
              file=sys.stderr)
        raise SystemExit(2)

    if not args.chunks_csv.exists():
        print(f"error: chunks CSV not found: {args.chunks_csv}", file=sys.stderr)
        raise SystemExit(2)

    if not args.prompt_file.exists():
        print(f"error: prompt file not found: {args.prompt_file}", file=sys.stderr)
        raise SystemExit(2)

    raise SystemExit(asyncio.run(run_pipeline(
        chunks_csv=args.chunks_csv,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        api_key=args.api_key,
        csv_id=args.csv_id,
        backend=args.backend,
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
        timeout=args.timeout,
        resume=args.resume,
        retry_failures=args.retry_failures,
        skip_until=args.skip_until,
        hf_device=args.hf_device,
        hf_dtype=args.hf_dtype,
        hf_cache_dir=args.hf_cache_dir,
        hf_quantize=args.hf_quantize,
        hf_prompt_format=args.hf_prompt_format,
        hf_trust_remote_code=args.hf_trust_remote_code,
        hf_max_new_tokens=args.hf_max_new_tokens,
        hf_temperature=args.hf_temperature,
    )))


if __name__ == "__main__":
    main()
