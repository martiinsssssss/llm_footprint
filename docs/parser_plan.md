# Parser work plan — first working version (`feature/parser`)

> Hand-off document for Claude Code. Read it completely before writing any code.

## Goal of this version

A **first working parser** that turns the ISMIR PDFs into a clean, sectioned JSONL the group can start testing analyses with. Keep it simple and readable; improvements come later (Section 9). When in doubt, choose the simpler option and tell me.

## 1. Context

Group project for the Research Methods course (SMC Master, UPF): a meta-research paper on how LLMs have affected academic writing in the ISMIR community.

Research questions:

1. How many papers published since 2023 explicitly disclose AI usage?
2. Did the vocabulary and sentence structure of academic writing shift more sharply after the release of LLMs?
3. Is the share of papers with an LLM-like writing style larger than the share that disclose AI usage?

Group pipeline:

```
scrapers/ (Emiliano, branch initial-scrapers)  →  parsing/ (THIS TASK)  →  analysis (Ines, Ari)
  data/pdfs/ + data/metadata_*.csv                  data/processed/*.jsonl
```

The analyses (Liang et al., Kobak et al., EditLens) work on word frequencies and raw prose. Any extraction artifact that changes from one year to another can look like a change in writing style. So the one thing this parser must get right is: **clean text, processed exactly the same way for every year.**

## 2. Working rules

- Work only on branch `feature/parser`. Never commit to or push `main`.
- **One phase at a time.** At the end of each phase: summarize what you did in plain language, explain any design decision, tell me exactly how to test it, and **wait for my OK** before the next phase.
- After I approve a phase, commit with a clear message. Don't push unless I ask.
- Only create or change files in `parsing/`, `tests/`, the parser section of `README.md`, and the one `.gitignore` line described in Phase 0. Ask before touching anything else.
- **Inputs are read-only:** never modify, move or rename anything in `data/pdfs/` or the `data/metadata_*.csv` files.
- Python 3.12 (the `.venv/` already exists). Code, comments and docstrings in English. Type hints. Small functions.
- Do **not** use LLM- or VLM-based extraction tools (Nougat, olmOCR, Marker's LLM mode…): we measure AI writing style, and a generative model in the extraction step could alter the text we measure.
- If something doesn't match this plan, stop and ask instead of guessing.

## 3. The data already in the repo

### PDFs

- Path: `data/pdfs/{year}/{paper_id}.pdf`, with `paper_id = {year}_{6 digits}`, e.g. `data/pdfs/2024/2024_000001.pdf`.
- 655 PDFs: 2020: 115 · 2021: 104 · 2022: 113 · 2023: 101 · 2024: 123 · 2025: 99.
- All have a text layer (no OCR needed) and an `ABSTRACT` heading. Median 8 pages, ~6,000 words.
- 2020 PDFs are the authors' originals (pdfTeX). 2021–2025 PDFs were re-stamped by ISMIR with pypdf (proceedings header and footer on each page).

### Metadata CSVs (from the scraper)

- `data/metadata_{year}.csv`, one row per PDF. All 655 rows match a PDF by `paper_id`.
- Columns: `year, paper_id, title, authors, pages, doi, pdf_url, pdf_path, abstract, discussion, conclusion`.
- **Use:** `title`, `authors`, `pages`, `doi`, `pdf_url`. The title is clean (from the ISMIR website); in the PDFs titles are all caps.
- **Don't use:**
  - `pdf_path`: it is a path inside the scraper's Docker container (`/data/pdfs/...`) and doesn't exist on the host. Build paths from `year` and `paper_id`.
  - `abstract`, `discussion`, `conclusion`: extracted with pypdf and full of artifacts whose frequency changes by year (% of abstracts affected):

    | Year | Split words ("W e present") | Ligatures ("ﬁ") | Line-break hyphens ("clas- siﬁes") |
    |---|---|---|---|
    | 2020 | 0% | 78% | 99% |
    | 2021 | 6% | 78% | 99% |
    | 2022 | 74% | 13% | 100% |
    | 2023 | 81% | 87% | 98% |
    | 2024 | 77% | 90% | 100% |
    | 2025 | 85% | 93% | 98% |

    They also have 9 empty abstracts, 23 papers where discussion and conclusion are the same duplicated text, broken characters ("stagesÐfeature") and table captions mixed into the text. The PDF text layer itself is fine: with a different extractor (pdftotext) "W e" appears in only 1 of 655 papers.
- Some CSV files contain NUL bytes (2020: 1, 2025: 13). Strip `\0` before parsing them with the `csv` module.

### Structure facts that affect the parser

- **~980 distinct top-level headings.** The frequent ones: INTRODUCTION, RELATED WORK(S), BACKGROUND, METHOD(S)/METHODOLOGY, EXPERIMENTS, RESULTS, DISCUSSION, CONCLUSION(S), REFERENCES.
- **Combined headings are common:** "EXPERIMENTS AND RESULTS" (39 papers), "RESULTS AND DISCUSSION" (38), "CONCLUSION(S) AND FUTURE WORK" (85), "DISCUSSION AND CONCLUSION(S)" (21), "INTRODUCTION AND RELATED WORK" (12), "BACKGROUND AND RELATED WORK" (10).
- **Acknowledgments** appear in ~60% of papers every year, spelled four ways (ACKNOWLEDGMENTS, ACKNOWLEDGEMENTS, ACKNOWLEDGMENT, ACKNOWLEDGEMENT).
- **Ethics statements** appear in 0–2 papers per year in 2020–2023, but in 37 papers in 2024 and 28 in 2025. They are a new type of text appearing right after ChatGPT, so they **must be labeled separately** so that style analyses can exclude them.
- A few papers have **appendices after the references** (2024_000075, 2025_000029, 2025_000062), and some are long (2024_000084: 38 pages).

## 4. Environment

- macOS on Apple Silicon. Docker Desktop with at least 4 GB of memory.
- GROBID, lightweight CRF image:

  ```bash
  docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.1-crf
  # if a platform error appears, add: --platform linux/amd64
  curl localhost:8070/api/isalive   # should return true
  ```

- Dependencies go in **`parsing/requirements.txt`** (requests, lxml, pandas, pytest), **not** in a root `requirements.txt`: the scraper branch already has one, and we'll merge the two later.

## 5. Architecture (first version)

```
parsing/
├── __init__.py
├── requirements.txt
├── sample_ids.txt     # test set (Phase 0)
├── models.py          # Section, Paper + JSONL read/write
├── metadata.py        # load the metadata CSVs → {paper_id: row}
├── grobid_parser.py   # GrobidParser (PDF → TEI, cached) + TeiReader (TEI → Paper)
├── labeler.py         # SectionLabeler (heading → labels)
├── cleaner.py         # TextCleaner (shared text normalization)
├── quality.py         # per-paper flags and artifact counters + per-year report
├── pipeline.py        # Pipeline (orchestrates everything)
└── run.py             # CLI entry point
tests/
├── test_cleaner.py
└── test_labeler.py
```

Flow for each PDF:

```
GrobidParser (TEI cached) → TeiReader → Paper + CSV metadata → SectionLabeler → TextCleaner → QualityChecker → JSONL
```

- No fallback parser in this version. Keep `GrobidParser.parse(pdf_path) -> Paper` as a clean, self-contained method so a common interface and a second parser can be added later.
- The `Pipeline` receives its components in the constructor, so each can be replaced or tested separately.
- **Structural** noise (citations, formulas, figures, footnotes) is removed in `TeiReader`. **Text** normalization is done by `TextCleaner`, identically for every paper.

### Output contract — shared with the rest of the group

One JSON object per line, sorted by `paper_id`, in `data/processed/ismir_2020_2025_v1.jsonl`:

| Field | Type | Source / description |
|---|---|---|
| `paper_id` | str | e.g. `2024_000001` (same as the CSV) |
| `year` | int | |
| `title` | str | CSV |
| `authors` | str | CSV |
| `doi` | str | CSV (empty for 2020–2021) |
| `abstract` | str | GROBID, cleaned |
| `sections` | list | each `{heading, number, location, labels, text}` |
| `full_text` | str | abstract + all sections (references excluded); for keyword searches such as AI disclosures |
| `n_words` | int | words in `full_text` |
| `warnings` | list[str] | quality flags (Phase 4) |
| `artifacts` | dict | artifact counters (Phase 4) |
| `parser` | str | `"grobid"` |
| `pipeline_version` | str | `"v1"` |

`Section` fields:

- `heading`: original title, e.g. `"PROPOSED METHOD"`
- `number`: `"3"`, `"3.1"` or `null`
- `location`: `"body"` or `"back"` (where GROBID put it)
- `labels`: list of labels (see below), never empty
- `text`: cleaned paragraph text

### Section labels

| Label | Heading contains (case-insensitive) |
|---|---|
| `introduction` | INTRODUCTION |
| `related_work` | RELATED WORK, BACKGROUND, PRIOR WORK, PREVIOUS WORK, LITERATURE |
| `discussion` | DISCUSSION |
| `conclusion` | CONCLUSION, CONCLUDING, FUTURE WORK |
| `acknowledgments` | ACKNOWLEDG |
| `ethics` | ETHIC |
| `appendix` | APPENDIX, SUPPLEMENTARY — or a `back` section of GROBID type `annex` that isn't acknowledgments or ethics |
| `body` | none of the above (method, data, experiments, results…) |

- A heading can get **several labels**: "DISCUSSION AND CONCLUSION" → `[discussion, conclusion]`; "INTRODUCTION AND RELATED WORK" → `[introduction, related_work]`; "RESULTS AND DISCUSSION" → `[discussion]`; "CONCLUSION AND FUTURE WORK" → `[conclusion]`. This avoids arbitrary precedence rules and duplicated text.
- **Subsections** (number like `6.1`) that only get `body` inherit the labels of their parent number (e.g. `6.1` under `6. CONCLUSION` → `[conclusion]`).
- `body` is only used when nothing else matches.

## 6. Phases

### Phase 0 — Setup

- Create the structure above (empty modules, one-line docstring each) and `parsing/requirements.txt`.
- In `.gitignore`, add `!data/processed/` right after `data/*`, so the output JSONL and report can be shared through git. Everything else in `data/` stays ignored.
- Create `parsing/sample_ids.txt` with this test set (one id per line):
  - Normal papers, two per year: `2020_000005`, `2020_000006`, `2021_000001`, `2021_000002`, `2022_000001`, `2022_000002`, `2023_000001`, `2023_000002`, `2024_000001`, `2024_000002`, `2025_000001`, `2025_000002`
  - Headings hard to detect with plain text extraction: `2023_000089`, `2024_000018`, `2024_000108`, `2025_000026`
  - Long papers / appendices: `2024_000084`, `2025_000029`, `2024_000075`, `2025_000062`
  - Abstract missing in the CSV: `2020_000023`, `2021_000037`
- **Done when:** `pip install -r parsing/requirements.txt` works, `python -c "import parsing"` works, `pytest` runs, and GROBID answers `true` on `isalive`.

### Phase 1 — Data model and GROBID parser

- `models.py`: `Section` and `Paper` dataclasses matching the contract; `full_text` and `n_words` computed; `to_dict()`, `write_jsonl()`, `read_jsonl()`.
- `metadata.py`: load all `data/metadata_*.csv` (stripping NUL bytes) into `{paper_id: row}`, keeping only the columns we use.
- `GrobidParser`:
  - POST the PDF to `http://localhost:8070/api/processFulltextDocument` with `consolidateHeader=0` and `consolidateCitations=0` (no external lookups: faster and reproducible). Generous timeout; clear error message if GROBID isn't running.
  - Cache the TEI at `data/interim/tei/{year}/{paper_id}.tei.xml`. If it exists, don't call GROBID again (`force=True` overrides).
- `TeiReader` (TEI namespace `http://www.tei-c.org/ns/1.0`):
  - Abstract: `teiHeader/profileDesc/abstract`.
  - Sections: every `div` under `text/body` (`location="body"`) and under `text/back` (`location="back"`), in document order, **except the bibliography** (`div[@type="references"]` / `listBibl`). Heading from `head`, number from `head/@n`. If a `back` div has no `head`, use its `@type` (e.g. `acknowledgement`) as heading.
  - Before extracting text, remove `ref[@type="bibr"]` (citation markers), `formula`, `figure` (includes tables) and `note` (footnotes). Keep the text of other `ref`s ("Figure 2", "Section 3").
  - Keep subsections as separate sections; the labeler handles inheritance.
- **Done when:** a small debug command prints, for each paper in `sample_ids.txt`, the abstract length and the list of `(number, heading, location)`. I'll compare with the PDFs, especially the edge cases.

### Phase 2 — Cleaner and labeler

- `TextCleaner.clean(text) -> str`, as small private steps in this order:
  1. Unicode NFKC (ligatures: `ﬁ` → `fi`)
  2. Remove NUL, control characters and `�`
  3. Join line-break hyphens: `word- word` with a lowercase letter after the space (`clas- sifies` → `classifies`). Real compounds have no space after the hyphen (`state-of-the-art`) and must stay intact.
  4. Remove leftover citation brackets: `[12]`, `[3, 4]`, `[5–7]`
  5. Remove URLs and e-mail addresses
  6. Normalize whitespace
- `SectionLabeler` implementing the label table and rules in Section 5.
- Tests (`tests/test_cleaner.py`, `tests/test_labeler.py`) with concrete examples: each cleaning step, compounds kept intact, and the real headings listed in Section 3 (including the four spellings of acknowledgments, combined headings and subsection inheritance).
- **Done when:** tests pass, and you show me every heading in the test set with its labels.

### Phase 3 — Pipeline and CLI

- `Pipeline(parser, labeler, cleaner, checker, metadata)` with `process_one(pdf_path) -> Paper` and `run(...)`.
- Metadata merge by `paper_id`. If a PDF has no CSV row, add the warning `no_metadata` and continue.
- One failing PDF must not stop the run: write its record with the `parse_error` warning and continue.
- Thread pool, 4 workers by default. Progress output (`n/total`) and a final summary.
- CLI:

  ```bash
  python -m parsing.run \
    --pdf-dir data/pdfs --metadata-dir data \
    --out data/processed/ismir_2020_2025_v1.jsonl \
    [--ids-file parsing/sample_ids.txt] [--years 2024 2025] [--workers 4] [--force]
  ```

- Output sorted by `paper_id`, so re-running gives an identical file.
- **Done when:** the test set produces a JSONL with 22 lines that `read_jsonl` loads, and then the full corpus produces 655 lines. GROBID on the full corpus can take a while; thanks to the TEI cache, later runs only redo the cheap steps.

### Phase 4 — Quality report and README

- Per-paper `warnings`: `parse_error`, `no_metadata`, `no_abstract`, `few_sections` (fewer than 3 sections labeled `introduction`, `related_work`, `body`, `discussion` or `conclusion`), `no_introduction`, `no_conclusion`.
- Per-paper `artifacts`, counted on the **cleaned** text:
  - `split_words`: matches of `\b(?:W e|W ith|W hen|W hile|W hat|W here|T o|T his|T he|T hese|T hus)\b`
  - `hyphen_breaks`: matches of `\b[a-z]+- [a-z]+\b`
  - `ligatures`: characters in `ﬀ`–`ﬆ`
  - `bad_chars`: NUL, control characters, `�`
- Per-year report saved to `data/processed/quality_report_v1.csv` and printed: number of papers, parse errors, % of papers with each warning, artifacts per 1,000 words, % with acknowledgments, % with ethics, median `n_words`.
- README, parser section: how to start GROBID, how to run the parser, the output schema, the notes in Section 7, and the versioning rule (any change to cleaning or labeling → new `v{N}` output file).
- **Done when:** the report exists for the full corpus. Artifact rates should be close to 0 and similar across years; if they aren't, show me which papers are responsible.

## 7. Notes for the analysis (also copy into the README)

- **Style analyses** (vocabulary, Liang, Kobak, EditLens): use `abstract` plus the sections whose `labels` include none of `acknowledgments`, `ethics` or `appendix`. Ethics statements only became common in 2024; including them would bias any before/after comparison.
- **AI-disclosure search:** use `full_text`, which includes acknowledgments and ethics statements. Many ISMIR papers are *about* LLMs or AI tools, so a keyword hit is not a disclosure.
- Don't use the `abstract`/`discussion`/`conclusion` columns of the metadata CSVs for analysis (see the artifact table in Section 3).

## 8. Out of scope for this version

- Downloading PDFs or changing the scraper.
- Detecting AI disclosures or AI-written text (analysis tasks).
- Deciding which sections the analyses use (the parser labels everything; the group decides).

## 9. Later improvements (not now)

- A `BaseParser` interface and a PyMuPDF fallback for papers GROBID handles badly.
- Finer labels for the middle of the paper (method, data, experiments, results).
- A tool to sample ~20 papers per year for manual review, and a documented manual check.
- Cross-checking GROBID abstracts against the CSV abstracts.
- Unit tests with small TEI fixtures.
- Merging `parsing/requirements.txt` with the scraper's `requirements.txt`.
- Better handling of constructions like "pre- and post-processing".

## 10. Definition of done (v1)

- `python -m parsing.run ...` processes all 655 PDFs with one command and gives an identical file when re-run.
- The same labeling and cleaning for every paper, whatever its year.
- Per-year quality report next to the dataset, with artifact rates close to 0 in every year.
- Tests pass, README updated, and the JSONL committed in `data/processed/` on `feature/parser`.
