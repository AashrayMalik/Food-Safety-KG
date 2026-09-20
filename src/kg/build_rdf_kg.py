#!/usr/bin/env python3
"""Build RDF knowledge-graph files from verified triplet CSV exports.

The pipeline reads the active ``schema_config.json`` at runtime, validates
triplet predicates/types against it, writes schema-conformant rows to RDF,
and quarantines schema drift for ontology iteration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

try:
    from rdflib import BNode, Graph, Literal, Namespace, URIRef
    from rdflib.namespace import OWL, RDF, RDFS, XSD
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise SystemExit(
        "ERROR: rdflib is required. Install dependencies with "
        "`food_lab/bin/pip install -r src/requirements.txt`."
    ) from exc


DEFAULT_TRIPLETS_CSV = Path("src/outputs/triplets/qwen/triplets_canonicalized.csv")
DEFAULT_SCHEMA_CONFIG = Path("src/extraction/schema_config.json")
DEFAULT_OUTPUT_DIR = Path("src/outputs/kg/qwen")

DEFAULT_BASE_IRI = "urn:fflo:resource:"
DEFAULT_ASSERTION_BASE_IRI = "urn:fflo:assertion:"

NAMESPACE_IRIS = {
    "fflo": "https://w3id.org/fflo/ontology#",
    "fkg": "https://w3id.org/fkg/ontology#",
    "fso": "https://w3id.org/fso/ontology#",
    "ssn": "http://www.w3.org/ns/ssn/",
    "sosa": "http://www.w3.org/ns/sosa/",
    "prov": "http://www.w3.org/ns/prov#",
    "lkif": "http://www.estrellaproject.org/lkif-core/lkif-core.owl#",
    "rdf": str(RDF),
    "rdfs": str(RDFS),
    "owl": str(OWL),
    "xsd": str(XSD),
}

REQUIRED_COLUMNS = (
    "snippet_id",
    "source_id",
    "source_file",
    "source_type",
    "chunk_index",
    "subject",
    "subject_type",
    "subject_id",
    "predicate",
    "object",
    "object_type",
    "object_id",
    "confidence",
    "evidence_span",
)

PROJECT_TERMS = {
    "Assertion": URIRef(NAMESPACE_IRIS["fflo"] + "Assertion"),
    "assertsSubject": URIRef(NAMESPACE_IRIS["fflo"] + "assertsSubject"),
    "assertsPredicate": URIRef(NAMESPACE_IRIS["fflo"] + "assertsPredicate"),
    "assertsObject": URIRef(NAMESPACE_IRIS["fflo"] + "assertsObject"),
    "snippetId": URIRef(NAMESPACE_IRIS["fflo"] + "snippetId"),
    "sourceId": URIRef(NAMESPACE_IRIS["fflo"] + "sourceId"),
    "sourceFile": URIRef(NAMESPACE_IRIS["fflo"] + "sourceFile"),
    "sourceType": URIRef(NAMESPACE_IRIS["fflo"] + "sourceType"),
    "chunkIndex": URIRef(NAMESPACE_IRIS["fflo"] + "chunkIndex"),
    "confidence": URIRef(NAMESPACE_IRIS["fflo"] + "confidence"),
    "evidenceSpan": URIRef(NAMESPACE_IRIS["fflo"] + "evidenceSpan"),
    "hasSignature": URIRef(NAMESPACE_IRIS["fflo"] + "hasSignature"),
    "signatureDomain": URIRef(NAMESPACE_IRIS["fflo"] + "signatureDomain"),
    "signatureRange": URIRef(NAMESPACE_IRIS["fflo"] + "signatureRange"),
}

PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*):(.+)$")
SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Schema:
    entity_types: frozenset[str]
    relations: frozenset[str]
    domain_range: dict[str, list[dict[str, list[str]]]]
    canonical_entities: dict[str, str]
    canonical_relations: dict[str, str]
    subclass_of: dict[str, list[str]]


@dataclass
class DriftRow:
    row: dict[str, str]
    reasons: list[str]
    suggestions: list[str] = field(default_factory=list)


@dataclass
class BuildResult:
    accepted_rows: int
    drift_rows: list[DriftRow]
    duplicate_statement_triples: int
    entity_count: int
    statement_triple_count: int


def canonical_qname(value: str) -> str:
    return value.strip().lower()


def local_name(qname: str) -> str:
    match = PREFIX_RE.match(qname.strip())
    return match.group(2) if match else qname.strip()


def split_qname(qname: str) -> tuple[str, str]:
    match = PREFIX_RE.match(qname.strip())
    if not match:
        raise ValueError(f"Expected prefixed name, got {qname!r}")
    return match.group(1), match.group(2)


def slugify(value: str, default: str = "unnamed") -> str:
    value = value.strip()
    value = re.sub(r"([a-z])([A-Z])", r"\1-\2", value)
    value = value.lower()
    value = SLUG_RE.sub("-", value).strip("-")
    return value or default


def short_hash(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _normalise_subclass_of(raw: dict) -> dict[str, list[str]]:
    """Tolerate both string and list values for subclass parents."""
    out: dict[str, list[str]] = {}
    for child, parents in raw.items():
        if isinstance(parents, (list, tuple, set)):
            out[child] = [str(p) for p in parents]
        else:
            out[child] = [str(parents)]
    return out


def load_schema(schema_config: Path) -> Schema:
    raw = json.loads(schema_config.read_text(encoding="utf-8"))
    entity_types = frozenset(raw.get("entity_types", []))
    relations = frozenset(raw.get("relations", []))
    domain_range = raw.get("domain_range", {})
    return Schema(
        entity_types=entity_types,
        relations=relations,
        domain_range=domain_range,
        canonical_entities={canonical_qname(t): t for t in entity_types},
        canonical_relations={canonical_qname(r): r for r in relations},
        subclass_of=_normalise_subclass_of(raw.get("subclass_of", {})),
    )


def bind_namespaces(graph: Graph) -> None:
    for prefix, iri in NAMESPACE_IRIS.items():
        graph.bind(prefix, Namespace(iri))


def term_uri(qname: str) -> URIRef:
    prefix, name = split_qname(qname)
    if prefix not in NAMESPACE_IRIS:
        raise ValueError(f"Unknown namespace prefix {prefix!r} in {qname!r}")
    return URIRef(NAMESPACE_IRIS[prefix] + name)


def is_absolute_iri(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value.strip()))


def join_iri(base_iri: str, *parts: str) -> URIRef:
    clean_parts = [slugify(part) for part in parts if part.strip()]
    if base_iri.startswith("urn:"):
        return URIRef(base_iri.rstrip(":") + ":" + ":".join(clean_parts))
    return URIRef(base_iri.rstrip("/") + "/" + "/".join(clean_parts))


def make_entity_uri(
    label: str,
    qname_type: str,
    provided_id: str,
    base_iri: str,
) -> URIRef:
    if provided_id.strip():
        raw_id = provided_id.strip()
        # Canonical ids are emitted as "fkg:FOO_00001" (a food-KG instance id,
        # distinct from the "fkg" ontology namespace). Strip the "fkg:" prefix
        # and join into the resource base IRI so entities resolve to instances.
        local_id = raw_id[4:] if raw_id.startswith("fkg:") else raw_id
        if is_absolute_iri(local_id):
            return URIRef(local_id)
        return join_iri(base_iri, local_id)

    prefix, type_name = split_qname(qname_type)
    identity = f"{qname_type}\0{label}"
    label_id = f"{slugify(label)}-{short_hash(identity, 8)}"
    return join_iri(base_iri, prefix, type_name, label_id)


def literal_from_value(value: str, object_type: str) -> Literal:
    datatype = term_uri(object_type)
    if object_type == "xsd:decimal":
        try:
            return Literal(Decimal(value.strip()), datatype=datatype)
        except InvalidOperation:
            return Literal(value, datatype=datatype)
    return Literal(value, datatype=datatype)


def classify_property(
    predicate: str, schema: Schema, observed_kinds: dict[str, set[str]]
) -> URIRef:
    """OWL property kind for a relation.

    Primary source is the schema's declared range value-kinds - this is what
    lets validate_row() catch genuine data errors (e.g. a hasDefinition row
    whose object is an entity instead of definition text) against a documented
    design, not just against whatever happened to be extracted this run.
    domain_range entries document usage patterns, not an exhaustive range
    allow-list (many predicates like appliesTo/hasFunction are legitimately
    polymorphic across many entity subtypes that aren't all individually
    enumerated), so this only ever looks at literal-vs-entity value-kind, never
    at which specific entity subtype is declared.

    Falls back to observed_kinds (actual object value-kinds seen in this CSV
    run) only when the schema declares no range at all for the predicate -
    otherwise an undocumented predicate silently defaults to ObjectProperty
    and punning-violates the moment it's used with a literal, regardless of
    what the data actually shows.

      - only xsd: kinds seen        -> owl:DatatypeProperty (literal-only)
      - only entity kinds seen      -> owl:ObjectProperty (entity-only)
      - both kinds seen              -> owl:AnnotationProperty, the only OWL
        construct that can legally take either a literal or an IRI - a
        Data/Object property cannot be punned to accept both.
      - neither (predicate declared but never used, schema silent)
                                      -> owl:ObjectProperty, a safe default
        since nothing contradicts it.
    """
    specs = schema.domain_range.get(predicate, [])
    all_ranges = [r for spec in specs for r in spec.get("range", [])]
    has_literal = any(r.startswith("xsd:") for r in all_ranges)
    has_object = any(not r.startswith("xsd:") for r in all_ranges)

    if not has_literal and not has_object:
        kinds = observed_kinds.get(predicate, set())
        has_literal = "literal" in kinds
        has_object = "entity" in kinds

    if has_literal and has_object:
        return OWL.AnnotationProperty
    if has_literal:
        return OWL.DatatypeProperty
    return OWL.ObjectProperty


def validate_row(
    row: dict[str, str], schema: Schema, observed_kinds: dict[str, set[str]]
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    suggestions: list[str] = []

    predicate = row.get("predicate", "").strip()
    subject_type = row.get("subject_type", "").strip()
    object_type = row.get("object_type", "").strip()

    if predicate == "UNMAPPED":
        reasons.append("unmapped_predicate")
    elif predicate not in schema.relations:
        reasons.append("unknown_predicate")
        by_local = {local_name(rel).lower(): rel for rel in schema.relations}.get(
            local_name(predicate).lower()
        )
        if by_local:
            suggestions.append(f"predicate_same_local_name:{by_local}")

    for field_name, value in (
        ("subject_type", subject_type),
        ("object_type", object_type),
    ):
        if value not in schema.entity_types:
            reasons.append(f"unknown_{field_name}")
            by_local = {local_name(t).lower(): t for t in schema.entity_types}.get(
                local_name(value).lower()
            )
            if by_local:
                suggestions.append(f"{field_name}_same_local_name:{by_local}")

    # Value-kind consistency check: catches rows whose object is a literal vs.
    # an entity in a way that contradicts the predicate's declared OWL property
    # kind. Left unchecked, these produce OWL punning violations once
    # ontology.ttl (which declares each predicate as exactly one of
    # ObjectProperty / DatatypeProperty / AnnotationProperty) and kg.ttl (the
    # actual triples) are loaded together - a property cannot legally take a
    # literal AND an individual as its value in the same OWL model unless it's
    # an AnnotationProperty. This is why kg_full.ttl (or kg.ttl + ontology.ttl
    # via owl:imports) can fail to open in Protégé/OWL-API tools even though
    # kg.ttl parses fine on its own as plain RDF.
    #
    # This deliberately checks value-kind only (literal vs. entity), not which
    # specific entity subtype is used - domain_range entries document usage
    # patterns, not an exhaustive range allow-list, so many predicates are
    # legitimately polymorphic across entity subtypes never individually
    # enumerated there.
    if not reasons and predicate in schema.relations:
        property_kind = classify_property(predicate, schema, observed_kinds)
        is_literal = object_type.startswith("xsd:")
        if property_kind == OWL.DatatypeProperty and not is_literal:
            reasons.append("object_value_kind_mismatch")
            suggestions.append("predicate_expects_literal_object")
        elif property_kind == OWL.ObjectProperty and is_literal:
            reasons.append("object_value_kind_mismatch")
            suggestions.append("predicate_expects_entity_object")

    return reasons, suggestions


def add_ontology_graph(schema: Schema, observed_kinds: dict[str, set[str]]) -> Graph:
    graph = Graph()
    bind_namespaces(graph)
    ontology = URIRef(NAMESPACE_IRIS["fflo"])
    graph.add((ontology, RDF.type, OWL.Ontology))
    graph.add(
        (ontology, RDFS.label, Literal("Food Fraud Lifecycle Ontology derived schema"))
    )

    # Provenance / reification vocabulary used by the ABox (kg.ttl / kg.nt):
    # declared here so the terms are defined rather than implicitly referenced.
    graph.add((PROJECT_TERMS["Assertion"], RDF.type, OWL.Class))
    # assertsObject can point at either an entity (URIRef) or a literal value
    # (whenever the reified triple's own object is a datatype-valued fact, e.g.
    # hasScientificName, hasNumericValue) - so, same reasoning as
    # classify_property(), it must be an AnnotationProperty, not ObjectProperty,
    # or it punning-violates on every literal-valued assertion it reifies.
    graph.add((PROJECT_TERMS["assertsObject"], RDF.type, OWL.AnnotationProperty))
    for term in (
        "assertsSubject",
        "assertsPredicate",
        "hasSignature",
        "signatureDomain",
        "signatureRange",
    ):
        graph.add((PROJECT_TERMS[term], RDF.type, OWL.ObjectProperty))
    for term in (
        "snippetId",
        "sourceId",
        "sourceFile",
        "sourceType",
        "chunkIndex",
        "evidenceSpan",
        "confidence",
    ):
        graph.add((PROJECT_TERMS[term], RDF.type, OWL.AnnotationProperty))

    for entity_type in sorted(schema.entity_types):
        graph.add((term_uri(entity_type), RDF.type, OWL.Class))

    for child, parents in sorted(schema.subclass_of.items()):
        child_uri = term_uri(child)
        for parent in parents:
            graph.add((child_uri, RDFS.subClassOf, term_uri(parent)))

    for relation in sorted(schema.relations):
        rel_uri = term_uri(relation)
        specs = schema.domain_range.get(relation, [])

        # See classify_property() for why this is value-kind only (schema range
        # primary, observed-data fallback), not a full range allow-list check.
        graph.add((rel_uri, RDF.type, classify_property(relation, schema, observed_kinds)))

        if len(specs) == 1:
            domains = specs[0].get("domain", [])
            ranges = specs[0].get("range", [])
            if len(domains) == 1:
                graph.add((rel_uri, RDFS.domain, term_uri(domains[0])))
            if len(ranges) == 1:
                graph.add((rel_uri, RDFS.range, term_uri(ranges[0])))
        elif specs:
            for spec in specs:
                sig = BNode()
                graph.add((rel_uri, PROJECT_TERMS["hasSignature"], sig))
                for domain in spec.get("domain", []):
                    graph.add((sig, PROJECT_TERMS["signatureDomain"], term_uri(domain)))
                for range_type in spec.get("range", []):
                    graph.add(
                        (sig, PROJECT_TERMS["signatureRange"], term_uri(range_type))
                    )

    return graph


def _add_entity_type(
    graph: Graph,
    node: URIRef,
    type_uri: URIRef,
    node_types: dict[URIRef, URIRef],
    type_conflicts: Counter,
) -> None:
    """Assert at most one rdf:type per node.

    First-seen type wins; any later conflicting type is recorded rather than
    emitted, so a node never carries two rdf:type statements (which happens
    when canonicalisation leaves a type conflict unresolved or a cross-type
    exact-match merge joins two differently-labelled entities).
    """
    existing = node_types.get(node)
    if existing is None:
        node_types[node] = type_uri
        graph.add((node, RDF.type, type_uri))
    elif existing != type_uri:
        type_conflicts[(existing, type_uri)] += 1


def add_row_to_graph(
    graph: Graph,
    row: dict[str, str],
    row_number: int,
    base_iri: str,
    assertion_base_iri: str,
    node_types: dict[URIRef, URIRef],
    type_conflicts: Counter,
) -> tuple[URIRef, URIRef, URIRef | Literal]:
    subject = make_entity_uri(
        row["subject"], row["subject_type"], row.get("subject_id", ""), base_iri
    )
    predicate = term_uri(row["predicate"])
    _add_entity_type(graph, subject, term_uri(row["subject_type"]), node_types, type_conflicts)
    graph.add((subject, RDFS.label, Literal(row["subject"])))

    object_type = row["object_type"].strip()
    if object_type.startswith("xsd:"):
        obj: URIRef | Literal = literal_from_value(row["object"], object_type)
    else:
        obj = make_entity_uri(
            row["object"], object_type, row.get("object_id", ""), base_iri
        )
        _add_entity_type(graph, obj, term_uri(object_type), node_types, type_conflicts)
        graph.add((obj, RDFS.label, Literal(row["object"])))

    graph.add((subject, predicate, obj))

    assertion_id = "|".join(
        [
            str(row_number),
            row.get("snippet_id", ""),
            row.get("subject", ""),
            row.get("predicate", ""),
            row.get("object", ""),
            row.get("evidence_span", ""),
        ]
    )
    assertion = join_iri(assertion_base_iri, short_hash(assertion_id, 16))
    graph.add((assertion, RDF.type, PROJECT_TERMS["Assertion"]))
    graph.add((assertion, PROJECT_TERMS["assertsSubject"], subject))
    graph.add((assertion, PROJECT_TERMS["assertsPredicate"], predicate))
    graph.add((assertion, PROJECT_TERMS["assertsObject"], obj))

    metadata_fields = {
        "snippetId": "snippet_id",
        "sourceId": "source_id",
        "sourceFile": "source_file",
        "sourceType": "source_type",
        "chunkIndex": "chunk_index",
        "evidenceSpan": "evidence_span",
    }
    for term_name, field_name in metadata_fields.items():
        value = row.get(field_name, "").strip()
        if value:
            graph.add((assertion, PROJECT_TERMS[term_name], Literal(value)))

    confidence = row.get("confidence", "").strip()
    if confidence:
        graph.add(
            (
                assertion,
                PROJECT_TERMS["confidence"],
                literal_from_value(confidence, "xsd:decimal"),
            )
        )

    return subject, predicate, obj


def write_drift_csv(path: Path, drift_rows: list[DriftRow]) -> None:
    fieldnames = list(REQUIRED_COLUMNS) + ["drift_reasons", "suggestions"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for drift in drift_rows:
            out = {name: drift.row.get(name, "") for name in REQUIRED_COLUMNS}
            out["drift_reasons"] = ";".join(drift.reasons)
            out["suggestions"] = ";".join(drift.suggestions)
            writer.writerow(out)


def serialize_graph(graph: Graph, path: Path, fmt: str) -> None:
    path.write_text(graph.serialize(format=fmt), encoding="utf-8")


def write_owl_catalog(output_dir: Path, ontology_iri: str, ontology_file: str) -> Path:
    """Write a Protégé/OWL-API catalog mapping the ontology IRI to a local file."""
    content = (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<catalog prefer="public" '
        'xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">\n'
        f'    <uri name="{ontology_iri}" uri="{ontology_file}"/>\n'
        '</catalog>\n'
    )
    path = output_dir / "catalog-v001.xml"
    path.write_text(content, encoding="utf-8")
    return path


def collect_observed_kinds(
    rows: list[dict[str, str]], schema: Schema
) -> dict[str, set[str]]:
    """Per-predicate object value-kinds ("literal"/"entity") actually seen.

    Used as classify_property()'s fallback for predicates the schema declares
    no range for at all. Only counts rows whose predicate/object_type are
    individually recognized - doesn't require the row to be otherwise valid,
    since that determination depends on this very classification.
    """
    observed: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        predicate = row.get("predicate", "").strip()
        object_type = row.get("object_type", "").strip()
        if predicate == "UNMAPPED" or predicate not in schema.relations:
            continue
        if object_type.startswith("xsd:"):
            observed[predicate].add("literal")
        elif object_type in schema.entity_types:
            observed[predicate].add("entity")
    return observed


def build_rdf_kg(
    triplets_csv: Path,
    schema_config: Path,
    output_dir: Path,
    base_iri: str = DEFAULT_BASE_IRI,
    assertion_base_iri: str = DEFAULT_ASSERTION_BASE_IRI,
) -> BuildResult:
    schema = load_schema(schema_config)
    graph = Graph()
    bind_namespaces(graph)

    # Ontology header + import so the ABox references (and can resolve to)
    # its TBox (ontology.ttl) when a tool follows owl:imports.
    kg_ontology = URIRef("urn:fflo:kg")
    graph.add((kg_ontology, RDF.type, OWL.Ontology))
    graph.add(
        (kg_ontology, RDFS.label, Literal("Food Fraud Lifecycle Knowledge Graph"))
    )
    graph.add((kg_ontology, OWL.imports, URIRef(NAMESPACE_IRIS["fflo"])))

    accepted_rows = 0
    drift_rows: list[DriftRow] = []
    statement_counts: Counter[tuple[Any, Any, Any]] = Counter()
    entities: set[URIRef] = set()
    node_types: dict[URIRef, URIRef] = {}
    type_conflicts: Counter = Counter()

    with triplets_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Triplet CSV is missing required columns: {missing}")
        rows = list(reader)

    # Needed before validation: classify_property() (used inside validate_row)
    # falls back to observed usage only for predicates the schema declares no
    # range for at all, so that fallback data has to exist up front.
    observed_kinds = collect_observed_kinds(rows, schema)

    for row_number, row in enumerate(rows, start=1):
        reasons, suggestions = validate_row(row, schema, observed_kinds)
        if reasons:
            drift_rows.append(
                DriftRow(row=row, reasons=reasons, suggestions=suggestions)
            )
            continue

        subject, predicate, obj = add_row_to_graph(
            graph, row, row_number, base_iri, assertion_base_iri,
            node_types, type_conflicts,
        )
        statement_counts[(subject, predicate, obj)] += 1
        entities.add(subject)
        if isinstance(obj, URIRef):
            entities.add(obj)
        accepted_rows += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    ontology_graph = add_ontology_graph(schema, observed_kinds)
    serialize_graph(graph, output_dir / "kg.ttl", "turtle")
    serialize_graph(graph, output_dir / "kg.nt", "nt")
    serialize_graph(ontology_graph, output_dir / "ontology.ttl", "turtle")

    # kg_full.ttl is self-contained (schema + data): strip the ABox ontology
    # header/import so Protégé doesn't try to resolve the unpublished ontology.
    full_graph = graph + ontology_graph
    full_graph.remove((kg_ontology, None, None))
    serialize_graph(full_graph, output_dir / "kg_full.ttl", "turtle")

    write_drift_csv(output_dir / "schema_drift.csv", drift_rows)
    write_owl_catalog(output_dir, NAMESPACE_IRIS["fflo"], "ontology.ttl")

    duplicate_statement_triples = sum(
        count - 1 for count in statement_counts.values() if count > 1
    )
    result = BuildResult(
        accepted_rows=accepted_rows,
        drift_rows=drift_rows,
        duplicate_statement_triples=duplicate_statement_triples,
        entity_count=len(entities),
        statement_triple_count=len(statement_counts),
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "triplets_csv": str(triplets_csv),
        "schema_config": str(schema_config),
        "output_dir": str(output_dir),
        "base_iri": base_iri,
        "assertion_base_iri": assertion_base_iri,
        "accepted_rows": result.accepted_rows,
        "quarantined_rows": len(result.drift_rows),
        "entity_count": result.entity_count,
        "statement_triple_count": result.statement_triple_count,
        "duplicate_statement_triples": result.duplicate_statement_triples,
        "node_type_conflicts": sum(type_conflicts.values()),
        "rdf_triples_total": len(graph),
        "drift_reasons": dict(
            Counter(reason for drift in result.drift_rows for reason in drift.reasons)
        ),
        "suggestions": dict(
            Counter(s for drift in result.drift_rows for s in drift.suggestions)
        ),
        "outputs": {
            "kg_ttl": str(output_dir / "kg.ttl"),
            "kg_nt": str(output_dir / "kg.nt"),
            "ontology_ttl": str(output_dir / "ontology.ttl"),
            "kg_full_ttl": str(output_dir / "kg_full.ttl"),
            "owl_catalog": str(output_dir / "catalog-v001.xml"),
            "schema_drift_csv": str(output_dir / "schema_drift.csv"),
        },
    }
    (output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def load_to_graph_store(endpoint: str, turtle_path: Path) -> None:
    import requests

    response = requests.post(
        endpoint,
        data=turtle_path.read_bytes(),
        headers={"Content-Type": "text/turtle"},
        timeout=120,
    )
    response.raise_for_status()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RDF KG files from verified triplet CSV output."
    )
    parser.add_argument("--triplets-csv", type=Path, default=DEFAULT_TRIPLETS_CSV)
    parser.add_argument("--schema-config", type=Path, default=DEFAULT_SCHEMA_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-iri", default=DEFAULT_BASE_IRI)
    parser.add_argument("--assertion-base-iri", default=DEFAULT_ASSERTION_BASE_IRI)
    parser.add_argument(
        "--load-endpoint",
        default="",
        help="Optional endpoint that accepts Turtle; kg.ttl is POSTed after generation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_rdf_kg(
        triplets_csv=args.triplets_csv,
        schema_config=args.schema_config,
        output_dir=args.output_dir,
        base_iri=args.base_iri,
        assertion_base_iri=args.assertion_base_iri,
    )

    if args.load_endpoint:
        load_to_graph_store(args.load_endpoint, args.output_dir / "kg.ttl")

    print(f"Accepted rows:     {result.accepted_rows}")
    print(f"Quarantined rows:  {len(result.drift_rows)}")
    print(f"Entities:          {result.entity_count}")
    print(f"Statement triples: {result.statement_triple_count}")
    print(f"Output directory:  {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())