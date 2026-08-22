#!/usr/bin/env python3
"""Shared schema-validation module used by extract_triplets and schema_violator.

Loads entity types, relations, and domain/range constraints from a
JSON config file, canonicalises them (strips known namespace prefixes
like fflo:/fkg:/fso:), and exposes:

* :class:`SchemaValidator` — re-usable validator with a given schema
* module-level ``VALID_ENTITY_TYPES``, ``VALID_RELATIONS``,
  ``RELATION_DOMAIN_RANGE``, ``REQUIRED_TRIPLET_FIELDS`` —
  loaded from the default ``schema_config.json`` (backward-compatible
  with ``extract_triplets.py``)
* Triplex-specific prefixed lists and canonical→prefixed maps

Usage as a library::

    from schema_validator import SchemaValidator

    sv = SchemaValidator("path/to/schema_config.json")
    violations = sv.validate_triplet(triplet, idx, sid, src_id, file, chunk)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------

_PREFIX_RE = re.compile(r"^(fflo|fkg|fso|prov|lkif|ssn):")


def _canonical(name: str) -> str:
    """Strip known namespace prefixes for fuzzy matching."""
    return _PREFIX_RE.sub("", name.strip())


# ---------------------------------------------------------------------------
# SchemaLoader
# ---------------------------------------------------------------------------


class SchemaValidator:
    """Validates triplets against a schema loaded from a JSON config file."""

    def __init__(self, schema_path: str | Path) -> None:
        schema_path = Path(schema_path)
        self._schema_path = schema_path
        self._raw: dict[str, Any] = json.loads(schema_path.read_text(encoding="utf-8"))

        # Canonicalised entity types
        self.entity_types: frozenset[str] = frozenset(
            _canonical(t) for t in self._raw.get("entity_types", [])
        )

        # Canonicalised relations
        self.relations: frozenset[str] = frozenset(
            _canonical(r) for r in self._raw.get("relations", [])
        )

        # Domain/range lookup: canonical relation → list of (domain_frozenset, range_frozenset)
        self.domain_range: dict[str, list[tuple[frozenset[str], frozenset[str]]]] = {}
        for rel, pairs in self._raw.get("domain_range", {}).items():
            canon_rel = _canonical(rel)
            entry: list[tuple[frozenset[str], frozenset[str]]] = []
            for pair in pairs:
                d = frozenset(_canonical(e) for e in pair.get("domain", []))
                r = frozenset(_canonical(e) for e in pair.get("range", []))
                entry.append((d, r))
            self.domain_range[canon_rel] = entry

        # Subclass hierarchy: canonical child → direct parents (tolerates a
        # single string or a list), expanded into a transitive ancestor map so
        # a subclass instance satisfies a parent's declared domain/range.
        self.subclass_of: dict[str, frozenset[str]] = {}
        for child, parents in self._raw.get("subclass_of", {}).items():
            if isinstance(parents, (list, tuple, set)):
                parent_set = frozenset(_canonical(p) for p in parents)
            else:
                parent_set = frozenset({_canonical(parents)})
            self.subclass_of[_canonical(child)] = parent_set

        self.ancestors: dict[str, frozenset[str]] = {}
        for child in self.subclass_of:
            seen: set[str] = set()
            stack = list(self.subclass_of[child])
            while stack:
                parent = stack.pop()
                if parent in seen:
                    continue
                seen.add(parent)
                stack.extend(self.subclass_of.get(parent, ()))
            self.ancestors[child] = frozenset(seen)

        # Prefixed entity / relation lists (for Triplex prompt format)
        self.entity_types_prefixed: list[str] = list(
            self._raw.get("entity_types", [])
        )
        self.relations_prefixed: list[str] = list(
            self._raw.get("relations", [])
        )

        # canonical → prefixed maps
        self.entity_canonical_to_prefixed: dict[str, str] = dict(
            zip([_canonical(n) for n in self.entity_types_prefixed],
                self.entity_types_prefixed)
        )
        self.relation_canonical_to_prefixed: dict[str, str] = dict(
            zip([_canonical(r) for r in self.relations_prefixed],
                self.relations_prefixed)
        )

        self.required_triplet_fields: tuple[str, ...] = (
            "subject", "subject_type", "predicate", "object", "object_type",
            "confidence", "evidence_span",
        )

    # -- Public validate method ------------------------------------------------

    def _is_subtype(self, actual: str, allowed: frozenset[str] | set[str]) -> bool:
        """True if ``actual`` is an allowed type or a subclass of one."""
        if actual in allowed:
            return True
        return any(a in allowed for a in self.ancestors.get(actual, ()))

    def validate_triplet(
        self,
        triplet: dict[str, Any],
        triplet_index: int,
        snippet_id: str,
        source_id: str,
        source_file: str,
        chunk_index: str,
    ) -> list[dict[str, Any]]:
        """Validate one triplet against the schema.  Returns a list of violations
        (empty list = fully valid).

        Entity type and relation names are canonicalised (prefixes stripped)
        before comparison, so the model may return either
        ``"fflo:Adulterant"`` or ``"Adulterant"``.
        """
        violations: list[dict[str, Any]] = []

        def _violation(
            vtype: str,
            field: str,
            actual: object,
            allowed: object | None = None,
        ) -> dict[str, Any]:
            return {
                "snippet_id": snippet_id,
                "source_id": source_id,
                "source_file": source_file,
                "chunk_index": chunk_index,
                "triplet_index": triplet_index,
                "violation_type": vtype,
                "field_name": field,
                "actual_value": actual,
                "allowed_values": (
                    sorted(allowed)
                    if isinstance(allowed, (set, frozenset))
                    else allowed
                ),
                "triplet": triplet,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

        # 1. Required fields present and non-empty
        for field in self.required_triplet_fields:
            val = triplet.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                violations.append(_violation("missing_field", field, val))
                return violations  # can't validate further without core fields

        # 2. Entity types (canonicalised)
        for role in ("subject_type", "object_type"):
            etype = _canonical(triplet.get(role, ""))
            if etype and etype not in self.entity_types:
                violations.append(
                    _violation(
                        "unknown_entity_type",
                        role,
                        triplet.get(role, ""),
                        allowed=sorted(self.entity_types),
                    )
                )

        # 3. Relation (canonicalised)
        pred_raw = triplet.get("predicate", "")
        pred = _canonical(pred_raw)
        if pred and pred not in self.relations:
            violations.append(
                _violation(
                    "unknown_relation",
                    "predicate",
                    pred_raw,
                    allowed=sorted(self.relations),
                )
            )
            return violations  # can't check domain/range without a known relation

        # 4. Domain / range (supports multiple valid pairs per relation)
        pairs = self.domain_range.get(pred)
        if pairs:
            stype = _canonical(triplet.get("subject_type", ""))
            otype = _canonical(triplet.get("object_type", ""))

            domain_ok = any(
                (not valid_domain or self._is_subtype(stype, valid_domain))
                for valid_domain, _valid_range in pairs
            )
            range_ok = any(
                (not valid_range or self._is_subtype(otype, valid_range))
                for _valid_domain, valid_range in pairs
            )

            if not domain_ok:
                all_domains = sorted({e for d, _ in pairs for e in d})
                violations.append(
                    _violation(
                        "domain_mismatch",
                        "subject_type",
                        triplet.get("subject_type", ""),
                        allowed=all_domains,
                    )
                )
            if not range_ok:
                all_ranges = sorted({e for _, r in pairs for e in r})
                violations.append(
                    _violation(
                        "range_mismatch",
                        "object_type",
                        triplet.get("object_type", ""),
                        allowed=all_ranges,
                    )
                )

        # 5. Confidence sanity
        conf = triplet.get("confidence")
        if conf is not None:
            try:
                cf = float(conf)
                if cf < 0.0 or cf > 1.0:
                    violations.append(
                        _violation("invalid_confidence", "confidence", conf)
                    )
            except (TypeError, ValueError):
                violations.append(
                    _violation("invalid_confidence", "confidence", conf)
                )

        return violations


# ---------------------------------------------------------------------------
# Module-level default schema (backward-compatible import surface for
# extract_triplets.py)
# ---------------------------------------------------------------------------

_DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "schema_config.json"

_default_sv = SchemaValidator(_DEFAULT_SCHEMA_PATH)

VALID_ENTITY_TYPES: frozenset[str] = _default_sv.entity_types
VALID_RELATIONS: frozenset[str] = _default_sv.relations
RELATION_DOMAIN_RANGE: dict[str, list[tuple[frozenset[str], frozenset[str]]]] = (
    _default_sv.domain_range
)
REQUIRED_TRIPLET_FIELDS: tuple[str, ...] = _default_sv.required_triplet_fields

# Triplex-specific lists (prefixed names preserved)
_TRIPLEX_ENTITY_TYPES_PREFIXED: list[str] = _default_sv.entity_types_prefixed
_TRIPLEX_RELATIONS_PREFIXED: list[str] = _default_sv.relations_prefixed
_CANONICAL_TO_PREFIXED_ENTITY: dict[str, str] = (
    _default_sv.entity_canonical_to_prefixed
)
_CANONICAL_TO_PREFIXED_RELATION: dict[str, str] = (
    _default_sv.relation_canonical_to_prefixed
)

# Re-export the validate function bound to the default schema
validate_triplet = _default_sv.validate_triplet


# Keep-in-sync assertion (fires at import time)
assert set(_CANONICAL_TO_PREFIXED_ENTITY.keys()) == set(VALID_ENTITY_TYPES), (
    "Triplex entity types out of sync with VALID_ENTITY_TYPES"
)
assert set(_CANONICAL_TO_PREFIXED_RELATION.keys()) == set(VALID_RELATIONS), (
    "Triplex relations out of sync with VALID_RELATIONS"
)
