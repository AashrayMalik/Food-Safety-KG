#!/usr/bin/env python3
"""End-to-end corpus building pipeline.

Converts PDFs to markdown via markitdown, scrapes URLs to plain-text via
Selenium, cleans both, aggregates into a single markdown file, and produces
neutral CSV chunks suitable for annotation / downstream NLP.

Usage:
    food_lab/bin/python src/build_corpus.py \
        --pdf-dir   src/data/FoodItems/Milk/pdfs \
        --links-file src/data/FoodItems/Milk/links.txt \
        --output-dir output/corpus

    # Skip scraping if only PDFs are needed:
    food_lab/bin/python src/build_corpus.py \
        --pdf-dir my_pdfs --output-dir out --skip-scrape
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import types
from pathlib import Path

# -- PDF conversion -------------------------------------------------------
from markitdown import MarkItDown

# -- Helpers from existing sibling scripts --------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from scrape_links import (  # noqa: E402
    urls_from_file,
    safe_filename,
    build_driver,
    extract_article_text,
    wait_for_captcha,
)
from preprocessing_cleaning.chunk_fooditem_sources import (  # noqa: E402
    flatten_markdown_tables,
    normalize_text,
    split_paragraphs,
    chunk_paragraphs,
    csv_safe_text,
)


# ===========================================================================
# Stage 1 — PDF → Markdown
# ===========================================================================

def _pdf_stem(pdf_path: Path) -> str:
    stem = pdf_path.stem
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return stem or "document"


def convert_pdfs(pdf_dir: Path, output_dir: Path) -> list[dict[str, str]]:
    """Convert every *.pdf in *pdf_dir* to markdown via markitdown."""
    md_converter = MarkItDown()
    md_dir = output_dir / "markdown"
    md_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"  No PDFs found in {pdf_dir}", file=sys.stderr)
        return []

    manifest: list[dict[str, str]] = []
    for pdf_path in pdfs:
        stem = _pdf_stem(pdf_path)
        out_path = md_dir / f"{stem}.md"

        if out_path.exists():
            print(f"  [skip] {out_path.name}")
            manifest.append(
                {"pdf": pdf_path.name, "md": out_path.name, "status": "skipped", "error": ""}
            )
            continue

        try:
            result = md_converter.convert(pdf_path)
            text = result.text_content
            if not text.strip():
                raise ValueError("markitdown returned empty content")
            out_path.write_text(text, encoding="utf-8")
            print(f"  [ok]   {pdf_path.name}  →  {out_path.name}")
            manifest.append(
                {"pdf": pdf_path.name, "md": out_path.name, "status": "ok", "error": ""}
            )
        except Exception as exc:
            print(f"  [fail] {pdf_path.name}: {exc}", file=sys.stderr)
            manifest.append(
                {"pdf": pdf_path.name, "md": out_path.name, "status": "failed", "error": str(exc)}
            )

    return manifest


# ===========================================================================
# Stage 2 — URL → Text (Selenium)
# ===========================================================================

def scrape_urls(
    links_file: Path,
    output_dir: Path,
    *,
    headless: bool = False,
    timeout: int = 30,
    delay: float = 1.0,
    page_wait: float = 4.0,
    captcha_wait: int = 300,
    no_captcha_pause: bool = False,
    overwrite: bool = False,
    profile_dir: Path | None = None,
) -> list[dict[str, str]]:
    """Scrape article text from every URL found in *links_file*."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    text_dir = output_dir / "text_files"
    text_dir.mkdir(parents=True, exist_ok=True)

    urls = urls_from_file(links_file)
    if not urls:
        print(f"  No URLs found in {links_file}", file=sys.stderr)
        return []

    if profile_dir is None:
        profile_dir = Path(".chrome-scraper-profile")
    profile_dir = profile_dir.expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)

    driver = build_driver(profile_dir, headless)

    captcha_args = types.SimpleNamespace(
        no_captcha_pause=no_captcha_pause,
        headless=headless,
        captcha_wait=captcha_wait,
    )

    manifest: list[dict[str, str]] = []
    try:
        wait = WebDriverWait(driver, timeout)
        for index, url in enumerate(urls, start=1):
            out_path = text_dir / safe_filename(url, index)

            if out_path.exists() and not overwrite:
                print(f"  [skip] {out_path.name}")
                manifest.append(
                    {"url": url, "file": out_path.name, "status": "skipped", "error": ""}
                )
                continue

            try:
                driver.get(url)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                wait_for_captcha(driver, url, captcha_args)
                time.sleep(page_wait)
                article_text = extract_article_text(driver)
                if not article_text:
                    raise ValueError("no article body text found")

                out_path.write_text(article_text.rstrip() + "\n", encoding="utf-8")
                print(f"  [ok]   {url[:70]}…  →  {out_path.name}")
                manifest.append(
                    {"url": url, "file": out_path.name, "status": "ok", "error": ""}
                )
            except Exception as exc:
                print(f"  [fail] {url[:70]}… : {exc}", file=sys.stderr)
                manifest.append(
                    {"url": url, "file": out_path.name, "status": "failed", "error": str(exc)}
                )

            if delay and index < len(urls):
                time.sleep(delay)
    finally:
        driver.quit()

    return manifest


# ===========================================================================
# Stage 3 — Cleaning
# ===========================================================================

_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"this article was downloaded by", re.IGNORECASE),
    re.compile(r"please scroll down for article", re.IGNORECASE),
    re.compile(r"^\d*\s*(?:accepted\s+)?manuscript\s*$", re.IGNORECASE),
    re.compile(r"to cite this article:", re.IGNORECASE),
    re.compile(r"to link to this article:", re.IGNORECASE),
    re.compile(r"taylor\s*[&]\s*francis", re.IGNORECASE),
    re.compile(r"informa ltd registered", re.IGNORECASE),
    re.compile(r"publication details, including instructions", re.IGNORECASE),
    re.compile(r"click for updates", re.IGNORECASE),
    re.compile(r"accepted author version posted online", re.IGNORECASE),
    re.compile(r"^disclaimer:", re.IGNORECASE),
    re.compile(r"terms\s*[&]\s*conditions of access", re.IGNORECASE),
    re.compile(r"terms and conditions of use", re.IGNORECASE),
    re.compile(r"^\|?\s*\d+\s*\|\s*P\s*age\s*\|?\s*$"),
    re.compile(r"^\[\s*\d+\s*\]\s*$"),
    re.compile(r"^on:\s+\d{1,2}\s+\w+\s+\d{4}", re.IGNORECASE),
    re.compile(r"^publisher:", re.IGNORECASE),
    re.compile(r"registered office:", re.IGNORECASE),
    re.compile(r"registered in england and wales", re.IGNORECASE),
    re.compile(r"^https?://", re.IGNORECASE),
    re.compile(r"^www\.", re.IGNORECASE),
    re.compile(r"^doi\s*:?\s*10\.\d{4}/", re.IGNORECASE),
    re.compile(r"^\d+\.\d{4,}/", re.IGNORECASE),
    re.compile(r"critical reviews in food science and nutrition", re.IGNORECASE),
    re.compile(r"^\d+\s*\|\s*\|\s*\w+", re.IGNORECASE),
    re.compile(r"reproduction, redistribution, reselling", re.IGNORECASE),
    re.compile(r"subscription information", re.IGNORECASE),
    re.compile(r"mortimer street", re.IGNORECASE),
    re.compile(r"to authors and researchers", re.IGNORECASE),
    re.compile(r"independently verified with primary sources", re.IGNORECASE),
    re.compile(r"expressly forbidden", re.IGNORECASE),
    re.compile(r"representations or warranties whatsoever", re.IGNORECASE),
    re.compile(r"all legal disclaimers that apply", re.IGNORECASE),
    re.compile(r"this article may be used for research", re.IGNORECASE),
    re.compile(r"any substantial or systematic", re.IGNORECASE),
    re.compile(r"any losses, actions, claims, proceedings", re.IGNORECASE),
    re.compile(r"arising directly or indirectly", re.IGNORECASE),
    re.compile(r"during production and pre-press", re.IGNORECASE),
    re.compile(r"version of record", re.IGNORECASE),
    re.compile(r"unedited manuscript", re.IGNORECASE),
    re.compile(r"copyediting, typesetting", re.IGNORECASE),
    re.compile(r"final publication of the Version of Record", re.IGNORECASE),
    re.compile(r"PLEASE SCROLL DOWN", re.IGNORECASE),
]

_PAGE_NUM_RE = re.compile(r"^\d{1,4}\s*$")
_PAGE_FRAG_RE = re.compile(r"^\|\s*\d{1,4}\s*\|?\s*$")

_OCR_GARBAGE_RE = re.compile(
    r"(?:\d{2,4}\s+)?[a-z]{2,3}\s+\d{1,2}\s+\d{2}:\d{2}\s+(?:ta\s+)?\]",
    re.IGNORECASE,
)

_INLINE_NOISE_RE = re.compile(
    r"\d{1,2}\s+ACCEPTED\s+MANUSCRIPT", re.IGNORECASE,
)

_METADATA_DENSITY_RE = re.compile(
    r"\b(?:registered|registration|publisher|doi|http|www\.|download|mortimer|taylor|francis|informa)\b",
    re.IGNORECASE,
)


def _is_boilerplate(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(stripped):
            return True
    return _OCR_GARBAGE_RE.search(stripped) is not None


def _is_page_artifact(line: str) -> bool:
    stripped = line.strip()
    if _PAGE_NUM_RE.fullmatch(stripped) or _PAGE_FRAG_RE.fullmatch(stripped):
        return True
    return False


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().casefold()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def clean_markdown_text(text: str, *, is_pdf: bool = True) -> str:
    """Clean markdown / plain-text by removing boilerplate and normalizing.

    For *is_pdf=True* sources, applies extra PDF-specific noise filters
    (journal headers, page-number artifacts, watermark OCR garbage).
    For web articles (*is_pdf=False*) the text is already cleaned by the
    scraper, so only normalisation and deduplication are applied.
    """
    lines = text.splitlines()
    cleaned: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned.append("")
            continue

        if not is_pdf:
            cleaned.append(line)
            continue

        if _is_boilerplate(stripped):
            continue
        if _is_page_artifact(stripped):
            continue

        if len(stripped) < 20 and not stripped.startswith("#") and "|" not in stripped:
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)
    text = normalize_text(text)

    if is_pdf:
        text = _INLINE_NOISE_RE.sub("", text)
        text = normalize_text(text)

    paragraphs = split_paragraphs(text)

    if is_pdf:
        paragraphs = _filter_metadata_paragraphs(paragraphs)

    unique = _dedup_keep_order(paragraphs)

    return "\n\n".join(unique)


def _filter_metadata_paragraphs(paragraphs: list[str]) -> list[str]:
    """Remove paragraphs that are dominated by publisher metadata noise.

    A paragraph is dropped if it contains 4+ boilerplate/metadata keywords
    and is under 400 characters (real content paragraphs are typically longer
    and have lower metadata keyword density).
    """
    result: list[str] = []
    for para in paragraphs:
        if len(para) > 400:
            result.append(para)
            continue
        hits = len(_METADATA_DENSITY_RE.findall(para))
        words = len(para.split())
        if hits >= 4 and words < 80:
            continue
        result.append(para)
    return result


# ===========================================================================
# Stage 4 — Aggregate & Chunk
# ===========================================================================

def _iter_sources(output_dir: Path) -> list[tuple[Path, str]]:
    sources: list[tuple[Path, str]] = []

    md_dir = output_dir / "markdown"
    if md_dir.exists():
        sources.extend((p, "pdf") for p in sorted(md_dir.glob("*.md")))

    txt_dir = output_dir / "text_files"
    if txt_dir.exists():
        sources.extend((p, "web_article") for p in sorted(txt_dir.glob("*.txt")))

    return sources


def _source_id(index: int, stype: str) -> str:
    prefix = "pdf" if stype == "pdf" else "web"
    return f"src_{prefix}_{index:03d}"


def _write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# Pipeline entry-point
# ===========================================================================

def build_corpus(
    pdf_dir: Path | None,
    links_file: Path | None,
    output_dir: Path,
    *,
    target_words: int = 180,
    max_words: int = 320,
    min_words: int = 60,
    headless: bool = False,
    timeout: int = 30,
    delay: float = 1.0,
    page_wait: float = 4.0,
    captcha_wait: int = 300,
    no_captcha_pause: bool = False,
    overwrite: bool = False,
    profile_dir: Path | None = None,
) -> int:
    """Run the full corpus building pipeline.

    Returns 0 on success, non-zero on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: PDF → Markdown ----
    print("\n=== Stage 1: Converting PDFs to Markdown ===")
    pdf_manifest: list[dict[str, str]] = []
    if pdf_dir and pdf_dir.is_dir():
        pdf_manifest = convert_pdfs(pdf_dir, output_dir)
        _write_manifest(output_dir / "pdf_manifest.csv", pdf_manifest,
                        ["pdf", "md", "status", "error"])
        print(f"  PDF manifest → {output_dir / 'pdf_manifest.csv'}")
    else:
        print("  No PDF directory; skipped.")

    # ---- Stage 2: Scrape URLs ----
    print("\n=== Stage 2: Scraping URLs ===")
    scrape_manifest: list[dict[str, str]] = []
    if links_file and links_file.is_file():
        scrape_manifest = scrape_urls(
            links_file, output_dir,
            headless=headless, timeout=timeout, delay=delay,
            page_wait=page_wait, captcha_wait=captcha_wait,
            no_captcha_pause=no_captcha_pause, overwrite=overwrite,
            profile_dir=profile_dir,
        )
        _write_manifest(output_dir / "scrape_manifest.csv", scrape_manifest,
                        ["url", "file", "status", "error"])
        print(f"  Scrape manifest → {output_dir / 'scrape_manifest.csv'}")
    else:
        print("  No links file; skipped.")

    # ---- Stage 3: Clean & Aggregate & Chunk ----
    print("\n=== Stage 3: Cleaning, Aggregating, Chunking ===")
    sources = _iter_sources(output_dir)
    if not sources:
        print("  No sources found to process.", file=sys.stderr)
        return 1

    aggregate_parts: list[str] = ["# Corpus Aggregated Source Text\n"]
    chunk_rows: list[dict[str, object]] = []

    for index, (path, stype) in enumerate(sources, start=1):
        raw = path.read_text(encoding="utf-8", errors="replace")
        is_pdf = stype == "pdf"

        if is_pdf and path.suffix.lower() == ".md":
            raw = flatten_markdown_tables(raw)

        text = clean_markdown_text(raw, is_pdf=is_pdf)
        sid = _source_id(index, stype)
        rel = path.relative_to(output_dir)

        aggregate_parts.append(
            f"\n\n## Source: {sid}\n\n"
            f"- File: `{rel}`\n"
            f"- Source type: `{stype}`\n\n"
            f"{text}\n"
        )

        paragraphs = split_paragraphs(text)
        chunks = chunk_paragraphs(paragraphs, target_words, max_words, min_words)

        for ci, chunk_text in enumerate(chunks, start=1):
            chunk_rows.append({
                "snippet_id": f"{sid}_chunk_{ci:04d}",
                "source_id": sid,
                "source_file": str(rel),
                "source_type": stype,
                "chunk_index": ci,
                "word_count": len(chunk_text.split()),
                "char_count": len(chunk_text),
                "evidence_text": csv_safe_text(chunk_text),
            })

        print(f"  [{sid}] {path.name}  →  {len(chunks)} chunks")

    # ---- Write outputs ----
    print("\n=== Stage 4: Writing Outputs ===")

    agg_path = output_dir / "aggregate.md"
    agg_path.write_text("".join(aggregate_parts), encoding="utf-8")
    print(f"  Aggregate Markdown → {agg_path}")

    csv_path = output_dir / "chunks.csv"
    fieldnames = [
        "snippet_id", "source_id", "source_file", "source_type",
        "chunk_index", "word_count", "char_count", "evidence_text",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chunk_rows)
    print(f"  Chunk CSV         → {csv_path}")

    ok_pdfs = sum(1 for r in pdf_manifest if r["status"] == "ok")
    ok_urls = sum(1 for r in scrape_manifest if r["status"] == "ok")
    print(f"\n=== Done ===")
    print(f"  PDFs converted : {ok_pdfs}")
    print(f"  URLs scraped   : {ok_urls}")
    print(f"  Total sources  : {len(sources)}")
    print(f"  Total chunks   : {len(chunk_rows)}")

    return 0


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a corpus from PDFs and URLs: convert, scrape, clean, aggregate, chunk."
    )
    p.add_argument("--pdf-dir", type=Path, help="Directory containing PDF files")
    p.add_argument("--links-file", type=Path, help="Text file with URLs (any format)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory for all generated artifacts")

    g = p.add_argument_group("Chunking")
    g.add_argument("--target-words", type=int, default=180)
    g.add_argument("--max-words", type=int, default=320)
    g.add_argument("--min-words", type=int, default=60)

    g = p.add_argument_group("Scraping (Selenium)")
    g.add_argument("--headless", action="store_true")
    g.add_argument("--timeout", type=int, default=30)
    g.add_argument("--delay", type=float, default=1.0,
                   help="Seconds between URL requests")
    g.add_argument("--page-wait", type=float, default=4.0,
                   help="Extra seconds to wait after page load")
    g.add_argument("--captcha-wait", type=int, default=300,
                   help="Seconds to wait for manual CAPTCHA solve")
    g.add_argument("--no-captcha-pause", action="store_true")
    g.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing converted/scraped files")
    g.add_argument("--profile-dir", type=Path, default=Path(".chrome-scraper-profile"))

    g = p.add_argument_group("Stage control")
    g.add_argument("--skip-pdf", action="store_true")
    g.add_argument("--skip-scrape", action="store_true")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    pdf_dir = args.pdf_dir
    links_file = args.links_file

    if args.skip_pdf:
        pdf_dir = None
    if args.skip_scrape:
        links_file = None

    if not pdf_dir and not links_file:
        print("error: at least one of --pdf-dir or --links-file is required "
              "(unless --skip-pdf/--skip-scrape targets existing output)",
              file=sys.stderr)
        raise SystemExit(2)

    raise SystemExit(build_corpus(
        pdf_dir=pdf_dir,
        links_file=links_file,
        output_dir=args.output_dir,
        target_words=args.target_words,
        max_words=args.max_words,
        min_words=args.min_words,
        headless=args.headless,
        timeout=args.timeout,
        delay=args.delay,
        page_wait=args.page_wait,
        captcha_wait=args.captcha_wait,
        no_captcha_pause=args.no_captcha_pause,
        overwrite=args.overwrite,
        profile_dir=args.profile_dir,
    ))


if __name__ == "__main__":
    main()
