#!/usr/bin/env python3
"""Aggregate FoodItems source text into annotation-friendly CSV chunks.

Markdown-aware: tables are flattened to pipe-separated inline text before
chunking so no table ever spans two rows. Section headings are tracked and
prepended to the first chunk that begins a new table block.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Lifecycle keyword sets (unchanged from original)
# ---------------------------------------------------------------------------

LIFECYCLE_KEYWORDS = {
    "fraud_act": [
        "adulterant", "spurious", "fake", "counterfeit", "synthetic",
        "dilute", "substitute", "misbrand", "substandard",
    ],
    "fraud_driver": [
        "profit", "cost", "cheap", "demand", "shortage", "festival",
        "economic", "incentive",
    ],
    "enabling_condition": [
        "loophole", "weak monitoring", "limited", "lack of", "unlicensed",
        "unauthorised", "unauthorized", "informal",
    ],
    "spread_pathway": [
        "supply", "distribution", "retail", "transport", "tanker", "market",
        "restaurant", "hotel", "collection centre", "chilling centre",
        "point of sale",
    ],
    "detection": [
        "detect", "test", "analysis", "laborator", "sample", "kit",
        "lactometer", "chromatograph", "lc-ms", "hplc", "gc-ms",
    ],
    "regulatory_response": [
        "fssai", "fssr", "regulat", "inspection", "seized", "license",
        "advisory", "recall", "compliance", "unsafe", "food safety",
        "standard", "permissible", "limit",
    ],
    "health_consequence": [
        "health", "risk", "toxic", "poison", "kidney", "renal", "liver",
        "organ", "cancer", "vomiting", "hospital", "death", "children",
        "elderly",
    ],
}

MILK_DAIRY_TERMS = [
    "milk", "dairy", "paneer", "butter", "ghee", "khoya", "curd",
    "cheese", "cream", "whey",
]


# ---------------------------------------------------------------------------
# Markdown table flattening
# ---------------------------------------------------------------------------

def _is_table_row(line: str) -> bool:
    return "|" in line

def _is_separator_row(line: str) -> bool:
    """Matches lines like |---|:---:|---| with only dashes, colons, pipes, spaces."""
    stripped = line.strip()
    return bool(re.fullmatch(r"[|\-: ]+", stripped)) and "|" in stripped

def _parse_cells(line: str) -> list[str]:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return cells

def flatten_markdown_tables(text: str) -> str:
    """
    Replace markdown tables with compact pipe-separated prose lines.

    Each data row becomes:  [Section heading if present]  Col1: val | Col2: val
    The header row is consumed as column names; separator rows are dropped.
    The nearest preceding heading is embedded only in the first flattened row
    of each table (as a bracketed prefix).

    Non-table lines pass through unchanged.
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    current_heading = ""

    while i < len(lines):
        line = lines[i]

        # Track the most recent heading for context
        heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading_match:
            current_heading = heading_match.group(2).strip()
            out.append(line)
            i += 1
            continue

        # Detect start of a table: current line has pipes, next is separator
        if (
            _is_table_row(line)
            and i + 1 < len(lines)
            and _is_separator_row(lines[i + 1])
        ):
            headers = _parse_cells(line)
            i += 2  # skip header + separator

            flattened: list[str] = []
            while i < len(lines) and _is_table_row(lines[i]) and not _is_separator_row(lines[i]):
                cells = _parse_cells(lines[i])
                # Zip against headers; if row is shorter, use empty string
                pairs = []
                for h, v in zip(headers, cells):
                    if h and v:
                        pairs.append(f"{h}: {v}")
                    elif h:
                        pairs.append(f"{h}: —")
                if pairs:
                    flattened.append(" | ".join(pairs))
                i += 1

            if flattened:
                # Prepend heading context only to the first flattened row
                if current_heading:
                    flattened[0] = f"[{current_heading}] {flattened[0]}"
                out.extend(flattened)
            continue

        # Plain line — pass through but strip lone separator lines
        if not _is_separator_row(line):
            out.append(line)
        i += 1

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Text normalization and splitting
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        if len(block) <= 3 and block.isdigit():
            continue
        paragraphs.append(block)
    return paragraphs


def chunk_paragraphs(
    paragraphs: list[str],
    target_words: int,
    max_words: int,
    min_words: int,
) -> list[str]:
    """Greedily pack paragraphs into word-bounded chunks, splitting oversized ones."""
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        word_count = len(words)

        if word_count > max_words:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_words = 0
            for start in range(0, word_count, target_words):
                part = " ".join(words[start: start + target_words]).strip()
                if part:
                    chunks.append(part)
            continue

        would_exceed = current_words + word_count > max_words
        has_enough = current_words >= min_words
        if current and would_exceed and has_enough:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_words = word_count
        else:
            current.append(paragraph)
            current_words += word_count

        if current_words >= target_words:
            chunks.append("\n\n".join(current))
            current = []
            current_words = 0

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def source_type_for(path: Path) -> str:
    """Classify a source file into a coarse source-type label by name/suffix."""
    name = path.name.lower()
    if "report" in name or "survey" in name or "fssai" in name or "nmqs" in name:
        return "regulatory_report"
    if path.suffix.lower() == ".txt" and re.match(r"\d{3}-", name):
        return "news_article"
    if path.suffix.lower() == ".md":
        return "converted_pdf"
    return "text_source"


def matched_stages(text: str) -> list[str]:
    """Return lifecycle stages whose keywords appear in the text."""
    lower = text.lower()
    stages = []
    for stage, keywords in LIFECYCLE_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            stages.append(stage)
    return stages


def matched_food_terms(text: str) -> list[str]:
    """Return milk/dairy terms present in the text (word-boundary match)."""
    lower = text.lower()
    return [term for term in MILK_DAIRY_TERMS if re.search(rf"\b{re.escape(term)}\b", lower)]


def csv_safe_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def source_id(index: int, path: Path) -> str:
    """Build a stable source identifier like ``milk_pdfmd_001`` from index/path."""
    prefix = "pdfmd" if path.suffix.lower() == ".md" else "txt"
    return f"milk_{prefix}_{index:03d}"


def iter_sources(food_dir: Path) -> list[Path]:
    """Discover source files (markdown + text) under a food-item directory."""
    markdown_dir = food_dir / "processed" / "markdown"
    sources = sorted(markdown_dir.glob("*.md")) if markdown_dir.exists() else []
    sources.extend(
        path
        for path in sorted(food_dir.glob("*.txt"))
        if path.name.lower() != "links.txt"
    )
    existing_md = [
        path for path in sorted(food_dir.glob("*.md"))
        if path.name != "schema.md"
    ]
    for path in existing_md:
        if path not in sources:
            sources.append(path)
    return sources


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Chunk all sources under --food-dir and write chunk CSV + aggregate Markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--food-dir", type=Path,
        default=Path("src/data/FoodItems/Milk"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("src/data/FoodItems/Milk/processed/chunks/milk_dairy_chunks.csv"),
    )
    parser.add_argument(
        "--aggregate-output", type=Path,
        default=Path("src/data/FoodItems/Milk/processed/chunks/milk_dairy_aggregate.md"),
    )
    parser.add_argument("--target-words", type=int, default=180)
    parser.add_argument("--max-words", type=int, default=320)
    parser.add_argument("--min-words", type=int, default=60)
    args = parser.parse_args()

    food_dir = args.food_dir
    output = args.output
    aggregate_output = args.aggregate_output
    output.parent.mkdir(parents=True, exist_ok=True)
    aggregate_output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    aggregate_sections = ["# Milk/Dairy Aggregated Source Text\n"]
    sources = iter_sources(food_dir)

    for index, path in enumerate(sources, start=1):
        raw_text = path.read_text(encoding="utf-8", errors="replace")

        # Flatten markdown tables before any other processing
        if path.suffix.lower() == ".md":
            raw_text = flatten_markdown_tables(raw_text)

        text = normalize_text(raw_text)
        paragraphs = split_paragraphs(text)
        chunks = chunk_paragraphs(
            paragraphs,
            target_words=args.target_words,
            max_words=args.max_words,
            min_words=args.min_words,
        )
        sid = source_id(index, path)
        stype = source_type_for(path)
        relative_path = path.relative_to(food_dir)

        aggregate_sections.append(
            f"\n\n## Source: {sid}\n\n"
            f"- File: `{relative_path}`\n"
            f"- Source type: `{stype}`\n\n"
            f"{text}\n"
        )

        for chunk_index, chunk_text in enumerate(chunks, start=1):
            stages = matched_stages(chunk_text)
            food_terms = matched_food_terms(chunk_text)
            word_count = len(chunk_text.split())
            char_count = len(chunk_text)
            rows.append(
                {
                    "snippet_id": f"{sid}_chunk_{chunk_index:04d}",
                    "source_id": sid,
                    "source_file": str(relative_path),
                    "source_type": stype,
                    "food_item": "Milk/Dairy",
                    "detected_food_terms": ";".join(food_terms),
                    "chunk_index": chunk_index,
                    "word_count": word_count,
                    "char_count": char_count,
                    "lifecycle_stage_hint": ";".join(stages),
                    "include_in_gold": "",
                    "ontology_requirement_id": "",
                    "annotator_notes": "",
                    "evidence_text": csv_safe_text(chunk_text),
                }
            )

    fieldnames = [
        "snippet_id", "source_id", "source_file", "source_type",
        "food_item", "detected_food_terms", "chunk_index",
        "word_count", "char_count", "lifecycle_stage_hint",
        "include_in_gold", "ontology_requirement_id",
        "annotator_notes", "evidence_text",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_output.write_text("".join(aggregate_sections), encoding="utf-8")
    print(f"Wrote {len(rows)} chunks from {len(sources)} sources to {output}")
    print(f"Wrote aggregate Markdown to {aggregate_output}")


if __name__ == "__main__":
    main()
