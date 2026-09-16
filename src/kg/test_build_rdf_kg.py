"""Unit tests for the RDF knowledge-graph builder.

Covers IRI slugging/joining, schema-drift quarantining, ontology
subclass axioms, and the provenance vocabulary emitted into kg.ttl /
ontology.ttl / kg_full.ttl.
"""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_rdf_kg import build_rdf_kg, join_iri, make_entity_uri, slugify


class RdfKgBuilderTests(unittest.TestCase):
    def test_slugify_stabilizes_labels(self) -> None:
        self.assertEqual(slugify("Food Safety & Standards"), "food-safety-standards")

    def test_default_iris_use_urns(self) -> None:
        self.assertEqual(
            str(make_entity_uri("milk", "fkg:Food", "", "urn:fflo:resource:")),
            "urn:fflo:resource:fkg:food:milk-e505d634",
        )
        self.assertEqual(
            str(join_iri("urn:fflo:assertion:", "abc123")),
            "urn:fflo:assertion:abc123",
        )

    def test_http_iri_bases_still_work_when_configured(self) -> None:
        self.assertEqual(
            str(join_iri("https://example.org/fflo/resource/", "fkg", "food", "milk")),
            "https://example.org/fflo/resource/fkg/food/milk",
        )

    def test_build_quarantines_schema_drift_and_writes_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_path = root / "schema.json"
            schema_path.write_text(
                json.dumps(
                    {
                        "entity_types": ["fkg:Food", "fkg:Ingredient", "xsd:string"],
                        "relations": ["fkg:hasIngredient", "fflo:hasDefinition"],
                        "domain_range": {
                            "fkg:hasIngredient": [
                                {"domain": ["fkg:Food"], "range": ["fkg:Ingredient"]}
                            ],
                            "fflo:hasDefinition": [
                                {"domain": ["fkg:Food"], "range": ["xsd:string"]}
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            csv_path = root / "triplets.csv"
            fieldnames = [
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
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "snippet_id": "s1",
                        "source_id": "src1",
                        "source_file": "doc.md",
                        "source_type": "markdown",
                        "chunk_index": "1",
                        "subject": "salt substitute",
                        "subject_type": "fkg:Food",
                        "subject_id": "",
                        "predicate": "fkg:hasIngredient",
                        "object": "potassium salt",
                        "object_type": "fkg:Ingredient",
                        "object_id": "",
                        "confidence": "0.95",
                        "evidence_span": "composed of potassium salt",
                    }
                )
                writer.writerow(
                    {
                        "snippet_id": "s2",
                        "source_id": "src1",
                        "source_file": "doc.md",
                        "source_type": "markdown",
                        "chunk_index": "2",
                        "subject": "food",
                        "subject_type": "fkg:Food",
                        "subject_id": "",
                        "predicate": "UNMAPPED",
                        "object": "unknown",
                        "object_type": "any",
                        "object_id": "",
                        "confidence": "0.5",
                        "evidence_span": "unknown relation",
                    }
                )

            out = root / "out"
            result = build_rdf_kg(csv_path, schema_path, out)

            self.assertEqual(result.accepted_rows, 1)
            self.assertEqual(len(result.drift_rows), 1)
            self.assertTrue((out / "kg.ttl").exists())
            self.assertTrue((out / "kg.nt").exists())
            self.assertTrue((out / "ontology.ttl").exists())
            drift_text = (out / "schema_drift.csv").read_text(encoding="utf-8")
            self.assertIn("unmapped_predicate", drift_text)

    def test_ontology_emits_subclass_axioms(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_path = root / "schema.json"
            schema_path.write_text(
                json.dumps(
                    {
                        "entity_types": [
                            "fkg:Food",
                            "fkg:Ingredient",
                            "fkg:ChemicalIngredient",
                            "xsd:string",
                        ],
                        "relations": ["fkg:hasIngredient"],
                        "domain_range": {
                            "fkg:hasIngredient": [
                                {"domain": ["fkg:Food"], "range": ["fkg:Ingredient"]}
                            ]
                        },
                        "subclass_of": {
                            "fkg:ChemicalIngredient": "fkg:Ingredient",
                        },
                    }
                ),
                encoding="utf-8",
            )
            csv_path = root / "triplets.csv"
            fieldnames = [
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
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "snippet_id": "s1",
                        "source_id": "src1",
                        "source_file": "doc.md",
                        "source_type": "markdown",
                        "chunk_index": "1",
                        "subject": "food",
                        "subject_type": "fkg:Food",
                        "subject_id": "",
                        "predicate": "fkg:hasIngredient",
                        "object": "salt",
                        "object_type": "fkg:ChemicalIngredient",
                        "object_id": "",
                        "confidence": "0.9",
                        "evidence_span": "contains salt",
                    }
                )

            out = root / "out"
            build_rdf_kg(csv_path, schema_path, out)

            ontology_text = (out / "ontology.ttl").read_text(encoding="utf-8")
            self.assertIn("subClassOf", ontology_text)

    def test_kg_imports_ontology_and_declares_provenance_vocab(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_path = root / "schema.json"
            schema_path.write_text(
                json.dumps(
                    {
                        "entity_types": ["fkg:Food", "fkg:Ingredient"],
                        "relations": ["fkg:hasIngredient"],
                        "domain_range": {
                            "fkg:hasIngredient": [
                                {"domain": ["fkg:Food"], "range": ["fkg:Ingredient"]}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            csv_path = root / "triplets.csv"
            fieldnames = [
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
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "snippet_id": "s1",
                        "source_id": "src1",
                        "source_file": "doc.md",
                        "source_type": "markdown",
                        "chunk_index": "1",
                        "subject": "food",
                        "subject_type": "fkg:Food",
                        "subject_id": "",
                        "predicate": "fkg:hasIngredient",
                        "object": "salt",
                        "object_type": "fkg:Ingredient",
                        "object_id": "",
                        "confidence": "0.9",
                        "evidence_span": "contains salt",
                    }
                )

            out = root / "out"
            build_rdf_kg(csv_path, schema_path, out)

            kg_text = (out / "kg.ttl").read_text(encoding="utf-8")
            ontology_text = (out / "ontology.ttl").read_text(encoding="utf-8")
            full_text = (out / "kg_full.ttl").read_text(encoding="utf-8")
            catalog_text = (out / "catalog-v001.xml").read_text(encoding="utf-8")

            self.assertIn("owl:imports", kg_text)
            self.assertIn("fflo:Assertion", ontology_text)
            self.assertIn("owl:Class", ontology_text)
            self.assertIn("fflo:confidence", ontology_text)

            self.assertTrue((out / "kg_full.ttl").exists())
            self.assertIn("fkg:Food a owl:Class", full_text)
            self.assertIn("fkg:hasIngredient", full_text)
            self.assertNotIn("owl:imports", full_text)

            self.assertIn("https://w3id.org/fflo/ontology#", catalog_text)
            self.assertIn('uri="ontology.ttl"', catalog_text)


if __name__ == "__main__":
    unittest.main()
