# Adulterant Gap Analysis (`src/scrape/`)

An auxiliary analysis that answers one question: **what do news reports surface
as food adulterants that no NABL-accredited lab is currently testing for?**

It is separate from the main extraction → validation → KG pipeline. The scrape
folder:

1. Scrapes the NABL portal for each lab's "Integrated Certificate" PDF URL,
2. Compares each lab's accredited *test scope* against adulterant terms
   extracted from news and FSSAI/Qwen triplets,
3. Flags the coverage gap ("uncovered" adulterants) and, optionally, re-runs the
   comparison with embedding similarity instead of substring matching,
4. Emits the JSON that drives the adulterant-comparison page of the observatory
   website.

```
NABL portal ──→ nabl_scraper.py ──→ results.csv ──────────────┐
                                                              │
news/Qwen triplets ──→ compare_adulterants.py ──→ coverage CSVs│
scope CSVs ───────────┘   (substring match)                   │
      │                                                       │
      └──→ semantic_compare.py (embedding match, optional) ───┤
                                                              │
curated adulterant_gap_analysis.csv ──→ build_gap_json.py ──→ website JSON
```

## Inputs (local, not checked into the repo)

| File | Produced by | Purpose |
|---|---|---|
| `FSSAI_NABL_Certificate_Numbers.xlsx` (sheet `Annexure1_Notified`) | supplied | TC certificate numbers + lab names + state |
| `results.csv` | `nabl_scraper.py` | resolved Integrated Certificate PDF URL per lab |
| `scope_entries_chemical.csv`, `scope_entries_biological.csv`, `scope_entries_other.csv` | supplied | lab scope parameters (parameter, discipline group, materials) |
| `adulterant_gap_analysis.csv` | curated | **authoritative** GAP/COVERED + category labels per news term |
| `../outputs/triplets/news/*.csv`, `../outputs/triplets/qwen/*.csv` | extraction pipeline | news / Qwen adulterant triplets |

## Scripts

### 1. `nabl_scraper.py` — resolve Integrated Certificate URLs

Looks up each `TC-xxxxx` certificate number on
`https://nablwp.qci.org.in/laboratorysearchone` and extracts the FSSAI
"Integrated Certificate" PDF URL (an S3 link). No PDF is downloaded — only the
URL is recorded.

Two equivalent flows:

- **Selenium (default)** — drives a real Chrome window. It fills the certificate
  number, clicks "View Integrated Certificate" (an ASP.NET LinkButton
  postback), then reads the FSSAI row's "View" link from the re-rendered page.
- **`--requests`** — replays the two ASP.NET postbacks with `requests` (no
  browser), capturing viewstate/cookies from the form.

```bash
# Selenium (default)
python3 src/scrape/nabl_scraper.py \
    --input FSSAI_NABL_Certificate_Numbers.xlsx \
    --sheet Annexure1_Notified \
    --log results.csv

# requests (no browser)
python3 src/scrape/nabl_scraper.py --input ... --sheet ... --log results.csv --requests

# resume / limit
python3 src/scrape/nabl_scraper.py --input ... --sheet ... --resume --limit 50
```

Writes `results.csv` with columns `tc_number, lab_name, status, pdf_url, error`.
S3 buckets are public-read, so the presigned query string is dropped to store a
permanent, non-expiring URL.

### 2. `compare_adulterants.py` — substring-based coverage

Compares lab scope parameters against news and FSSAI/Qwen adulterant terms
(kept separate). The headline finding is news adulterants with **no** lab
coverage.

```bash
python3 src/scrape/compare_adulterants.py \
    --news  ../outputs/triplets/news/News_Triplets_Oil_Ghee_Milk_20260820.csv \
    --qwen  ../outputs/triplets/qwen/triplets_canonicalized.csv \
    --scope scope_entries_chemical.csv scope_entries_biological.csv scope_entries_other.csv \
    --results results.csv \
    --labs-xlsx FSSAI_NABL_Certificate_Numbers.xlsx \
    --json-out adulterant_comparison.json
```

Terms are normalised/deduplicated and categorised (metal, mycotoxin,
contaminant, microorganism, adulterant, other) via keyword lexicons. Outputs:

| File | Description |
|---|---|
| `adulterant_coverage.csv` | each known term → tested by any lab? (yes/no) |
| `new_emerging.csv` | scope params in adulterant/contaminant categories not in news/Qwen |
| `--json-out` (optional) | summary JSON for the website |

### 3. `semantic_compare.py` — embedding-based coverage

Replaces the substring matcher with cosine similarity over
`sentence-transformers` (`all-MiniLM-L6-v2`) embeddings. News terms, Qwen terms,
and scope parameters are embedded (cached in `semantic_embeddings/` as
`.npy`/`.json`), then each news term is marked COVERED if any lab's parameter
clears a similarity threshold.

The threshold is auto-tuned by sweeping against the curated
`adulterant_gap_analysis.csv` ground truth (or overridden with `--threshold`).
The curated CSV stays **authoritative**: this script only *proposes* flips, it
never overwrites it.

```bash
python3 src/scrape/semantic_compare.py \
    --gap-csv adulterant_gap_analysis.csv \
    --threshold 0.60 \
    --force
```

| Output | Description |
|---|---|
| `semantic_adulterant_coverage.csv` | term → top-k matched params + scores + verdict |
| `semantic_validation.csv` | threshold sweep (precision/recall/F1/agreement vs curated) |
| `semantic_gap_proposals.csv` | proposed GAP↔COVERED flips for review |
| `semantic_gap_validation.csv` (via `--gap-report`) | full semantic audit of every specific-substance GAP term |
| `semantic_embeddings/` | cached embedding matrices + string lists |

### 4. `build_gap_json.py` — website JSON

Builds the final `adulterant-comparison.json` consumed by the observatory
website from the curated `adulterant_gap_analysis.csv`, enriched with news
context (object type, associated foods, evidence, near-miss check against lab
materials).

```bash
python3 src/scrape/build_gap_json.py
```

Writes `../food-safety-observatory-with-changes/data/adulterant-comparison.json`.

## Requirements

```bash
pip install selenium requests openpyxl beautifulsoup4
# semantic_compare.py additionally:
pip install sentence-transformers numpy
```
