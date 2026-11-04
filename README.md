# LLM Footprint

## Scrapers

Each scraper lives in `scrapers/` and runs in one shared Docker image.

### ISMIR

Downloads ISMIR papers (2020–2025) and saves metadata plus the abstract, discussion and conclusion of each paper to `<out>/metadata_<year>.csv` (one file per year; re-running a year replaces only that file). PDFs go to `<out>/pdfs/<year>/`. Change `--out data` to pick the output folder (a path inside the repository).

The `doi` column is empty for 2020–2021: those conference pages link straight to the PDF instead of a DOI.

```bash
docker compose build                                                                  # build the image
docker compose run --rm --no-deps app scrapers/ismir.py --out data                      # all years
docker compose run --rm --no-deps app scrapers/ismir.py --out data --years 2024 2025    # some years
docker compose run --rm --no-deps app scrapers/ismir.py --out data --limit 3            # quick test
```

## Parser (`parsing/`)

Turns the ISMIR 2020–2025 PDFs (`data/pdfs/`) into one clean, sectioned JSON record per paper, processed in exactly the same way for every year. It uses [GROBID](https://github.com/kermitt2/grobid) (no LLM- or VLM-based extraction, since we measure AI writing style). The full design is in [docs/parser_plan.md](docs/parser_plan.md).

### Getting the dataset (no setup needed)

Nothing under `data/` is committed to git. The outputs live in the group's **shared Google Drive folder** `<shared Drive folder>`:

| File | What it is |
|---|---|
| `ismir_2020_2025_v{N}.jsonl` | the dataset: one paper per line |
| `quality_report_v{N}.csv` | per-year quality report for that dataset |
| `tei_cache_v{N}.zip` | GROBID's TEI output for every PDF (lets you regenerate the dataset without GROBID) |

To use them, download the JSONL and the CSV into `data/processed/`:

```python
from pathlib import Path
from parsing.models import read_jsonl

papers = read_jsonl(Path("data/processed/ismir_2020_2025_v1.jsonl"))
```

Every upload keeps its version in the file name, and nothing is overwritten. This table records which git commit produced each version:

| Version | Git commit | Date | Changes |
|---|---|---|---|
| v1 | `939fa4a` | 2026-10-20 | first version: 655 papers |

**Versioning rule:** any change to cleaning or labeling produces a new `v{N}` (new file names, new row in the table above, new `pipeline_version` in the records). Don't mix versions in one analysis.

### Output schema

One JSON object per line, sorted by `paper_id`:

| Field | Type | Description |
|---|---|---|
| `paper_id` | str | e.g. `2024_000001`, same as in `data/metadata_*.csv` |
| `year` | int | |
| `title`, `authors`, `doi` | str | from the metadata CSVs (`doi` is empty for 2020–2021) |
| `abstract` | str | from GROBID, cleaned |
| `sections` | list | each `{heading, number, location, labels, text}` (see below) |
| `full_text` | str | abstract + all section texts, references excluded |
| `n_words` | int | words in `full_text` |
| `warnings` | list[str] | quality flags, see [Quality report](#quality-report) |
| `artifacts` | dict | extraction artifact counts, see [Quality report](#quality-report) |
| `parser` | str | `"grobid"` |
| `pipeline_version` | str | `"v1"` |

Section fields: `heading` (as printed), `number` (`"3"`, `"3.1"` or `null`), `location` (`"body"` or `"back"`, where GROBID put it), `labels` (never empty), `text` (cleaned; paragraphs separated by a blank line).

Section labels: `introduction`, `related_work`, `discussion`, `conclusion`, `acknowledgments`, `ethics`, `appendix`, and `body` for everything else (method, data, experiments, results…). A section can have several labels ("DISCUSSION AND CONCLUSION" → `[discussion, conclusion]`). The labeling rules (subsections, GROBID's false headings, appendices, ethics statements) are in Section 5 of the [plan](docs/parser_plan.md).

Text cleaning, identical for every paper: Unicode NFKC (ligatures), removal of control characters, joining of line-break hyphens (`clas- sifies` → `classifies`, compounds like `state-of-the-art` kept), removal of citation brackets, URLs and e-mails, of the empty brackets and stray spaces they leave, whitespace normalization, and removal of letter-by-letter noise (runs of 30 or more single-character tokens, e.g. LaTeX image data that some PDFs carry as invisible text: `A A A C 1 n i c j V H L…`). Citation markers, formulas, figures, tables, footnotes and the bibliography are removed from the GROBID output before that.

### Notes for the analysis

- **Style analyses** (vocabulary, Liang, Kobak, EditLens): use `abstract` plus the sections whose `labels` include none of `acknowledgments`, `ethics` or `appendix`. Ethics statements only became common in 2024; including them would bias any before/after comparison.
- **AI-disclosure search:** use `full_text`, which includes acknowledgments and ethics statements. Many ISMIR papers are *about* LLMs or AI tools, so a keyword hit is not a disclosure.
- Don't use the `abstract`/`discussion`/`conclusion` columns of the metadata CSVs for analysis: they were extracted with pypdf and have year-dependent artifacts (split words, ligatures, line-break hyphens).
- Wrongly *including* LLM-generated text or ethics statements in the style analyses is worse than wrongly *excluding* a human paragraph, because the first error correlates with year (2024–2025). The labeling rules are tuned with that in mind.

```python
EXCLUDED = {"acknowledgments", "ethics", "appendix"}

def style_text(paper) -> str:
    parts = [paper.abstract] + [s.text for s in paper.sections if not EXCLUDED & set(s.labels)]
    return "\n\n".join(p for p in parts if p)
```

### Quality report

`quality_report_v{N}.csv` has one row per year and a `total` row:

- `n_papers`, `parse_errors`
- `pct_<warning>`: % of papers with each warning: `parse_error` (GROBID failed; the record has metadata only), `no_metadata`, `no_abstract`, `few_sections` (fewer than 3 sections labeled `introduction`, `related_work`, `body`, `discussion` or `conclusion`), `no_introduction`, `no_conclusion`
- `<artifact>_per_1000_words`, counted on the cleaned `full_text`: `split_words` ("W e", "T he"…), `hyphen_breaks` ("clas- sifies"; 0 by construction, since the cleaner joins this pattern), `ligatures` ("ﬁ"), `bad_chars` (control characters, "�")
- `char_run_tokens_removed_per_1000_words`: tokens removed as letter-by-letter noise per 1,000 remaining words (per paper: `artifacts["char_run_tokens_removed"]`)
- `pct_with_acknowledgments`, `pct_with_ethics`, `median_n_words`

v1 summary: no parse errors, no missing metadata or abstracts. `no_conclusion` is 7.2% overall (11.3% in 2020, 3.5–10.1% in other years): mostly papers that end with a discussion. All artifact rates are ≤ 0.006 per 1,000 words in every year; the few `split_words` hits are false positives (math notation such as `W e ∈ R`). Letter-by-letter noise removed: 29.0 tokens per 1,000 words in 2020, 11.5 in 2021, 2.3 in 2022 and below 0.3 from 2023 on, 20,681 tokens in total from 18 papers, mostly 2020_000099 (9,968), 2021_000067 (4,929) and 2020_000188 (4,281). Without this step those three papers had about twice their real length.

To recompute the report from a downloaded JSONL: `docker compose run --rm app -m parsing.quality data/processed/ismir_2020_2025_v1.jsonl`.

### Running the parser

Everything runs in Docker, in the same image as the scrapers (Docker Desktop with at least 4 GB of memory). `compose.yaml` starts GROBID next to it and waits until it is ready; the repository is mounted into the container, so code edits need no rebuild.

```bash
docker compose build                       # once, or after changing requirements.txt
```

**Run** (from the repository root):

```bash
docker compose run --rm app -m parsing.run --out data/processed/ismir_2020_2025_v1.jsonl \
  [--ids-file parsing/sample_ids.txt] [--years 2024 2025] [--workers 4] [--force]
docker compose down                        # stop GROBID when done
```

This writes the JSONL and `quality_report_v1.csv` next to it, and prints progress and a summary. `parsing/sample_ids.txt` is a 22-paper test set with normal papers and known edge cases. Re-running gives an identical file.

Processing all 655 PDFs takes about 5 minutes with 4 workers. TEI files are cached in `data/interim/tei/`, so later runs take seconds and only redo the cheap steps (`--force` ignores the cache). To skip GROBID's work entirely, download `tei_cache_v{N}.zip` from the Drive and unzip it so the files end up in `data/interim/tei/{year}/{paper_id}.tei.xml`: `unzip tei_cache_v1.zip -d data/interim/`. The run then gives exactly the same JSONL; use this to regenerate the dataset after changing the cleaning or labeling.

**Uploading a new version to the Drive:**

1. Bump `pipeline_version` in `parsing/models.py` and the version in the `--out` file name, commit, and add a row to the version table above.
2. Run the parser, then zip the TEI cache (`*.zip` is git-ignored):
   ```bash
   (cd data/interim && zip -qr ../../tei_cache_v1.zip tei)
   ```
3. Upload `data/processed/ismir_2020_2025_v{N}.jsonl`, `data/processed/quality_report_v{N}.csv` and `tei_cache_v{N}.zip` to the Drive folder. The TEI cache only needs a new version if it was regenerated with GROBID (`--force` or a new GROBID version).

**Other commands:**

```bash
docker compose run --rm app -m pytest                     # unit tests (cleaner, labeler)
docker compose run --rm app -m parsing.labeler            # debug: every section of the test set with labels and word counts
docker compose run --rm app -m parsing.review data/processed/ismir_2020_2025_v1.jsonl  # label review reports
```

`parsing.review` prints every `ethics` section, every section labeled `appendix` by the backfill rule, where each appendix region starts (flagging those that start before the conclusion), and the back sections with a numeric number (body text that GROBID misplaced).

### Known limitations (v1)

These come mostly from GROBID; none of them puts LLM-generated text or ethics statements into the style text.

- **Body text that GROBID moved to the back matter is labeled `appendix`**, so it is excluded from the style text. It is concentrated in 2020 (sections with a numeric number in the back matter, data-availability and author-contribution statements not counted):

  | Year | Sections | Papers | Words lost |
  |---|---|---|---|
  | 2020 | 9 | 4 | 1,328 |
  | 2021 | 1 | 1 | 166 |
  | 2022 | 3 | 2 | 422 |
  | 2023 | 2 | 1 | 552 |
  | 2024 | 3 | 2 | 353 |
  | 2025 | 0 | 0 | 0 |

  That is at most 0.3% of a year's style text. It includes the conclusions of 2020_000101 and 2020_000165, which therefore get `no_conclusion`.
- **2024_000042:** a dataset field named "Ethical Considerations" and the 6 false headings after it (about 400 words of dataset description) are labeled `ethics`.
- **2024_000097:** the heading "Supplementary Material / Discussion" starts the appendix, so its last 74 words of discussion are excluded.
- **Headings lost by GROBID are not recovered.** Their text is merged into the previous section, and their subsections get `body` (e.g. 2023_000089's conclusion is inside "3.7 Examples"; 2024_000018 section 4; 2025_000026 section 2). The text stays in the style analyses; only its label is wrong.
- **Tables GROBID doesn't detect leak into the text**, as cell values or headers (e.g. 2024_000002, where a table's "Content … Saraga Saraga AV" became a section). Sections of style text with ≥ 30% numeric tokens: 5–14 per year, 0.8k–4.8k words per year.
- **Letter-by-letter noise shorter than 30 tokens is kept** (the cleaner only removes longer runs, so that sequences like `x y z` or chord names survive). Removed runs are counted in `char_run_tokens_removed`; their year dependence (mostly 2020–2021) is visible in the report.
- **Stray footnote numbers:** footnote markers that GROBID doesn't tag stay in the text as lone digits (`…for weakly aligned data. 1 This procedure…`), about 190 in the corpus.
- **Dashes dropped by GROBID** are not restored: `betweenborrowing` in 2024_000001 is already joined in GROBID's output.
- Citation leftovers after "e.g."/"i.e." can leave punctuation such as `e.g.,.` (punctuation only), and "pre- and post-processing" becomes "preand post-processing".

## Analysis (`analysis/`)

Works on the parser's JSONL. `classify.py` (human vs. LLM) is not implemented yet.

**Synthetic abstracts.** Ask Claude Code to use the `synthetic-abstract` agent (`.claude/agents/synthetic-abstract.md`), for example: *"use the synthetic-abstract agent for 2024_000001 and 2024_000002"*. For each paper it reads the paper text without the abstract (`python -m analysis.generate export <id>`), writes an abstract from the proposal's prompt, and saves it to `data/interim/synthetic/<id>.json` with the model and the prompt. The prompt asks for the original abstract's word count, so length does not give the source away; use `--words 150` to get the proposal's fixed length. Existing files are never replaced without `--force`.

Run it from the repo root, with Docker running, either inside a `claude` session (type the request) or from the terminal:

```bash
claude -p "use the synthetic-abstract agent for 2024_000003 2024_000004"
```

`.claude/settings.json` allows the agent's two commands (`analysis.generate export` and `save`) to run without approval prompts, which `claude -p` cannot answer. Any other command still needs approval, so the agent cannot read the original abstracts.

```bash
# Features of every abstract -> one CSV row per abstract (source = human)
docker compose run --rm --no-deps app -m analysis.features data/processed/ismir_2020_2025_v1.jsonl --out data/processed/features_v1.csv
# Every sentence that mentions an AI tool -> CSV with an empty `label` column to fill by hand
docker compose run --rm --no-deps app -m analysis.disclosure data/processed/ismir_2020_2025_v1.jsonl --out data/processed/disclosure_hits_v1.csv
```

Features (after Fröhling & Zubiaga 2021): characters per word, words per sentence, Flesch reading ease, MTLD, stop-word and hapax shares, word 1/2-gram overlap between consecutive sentences, shares of words in the top 1,000/10,000 English words (`wordfreq`), the share of each POS tag except `INTJ`, `SYM` and `X`, and `ORG` entities per 100 words (spaCy `en_core_web_sm`). A word is any token that is not punctuation or whitespace. Dropped from v1 because they are 0 in most abstracts (66–99.8%): rare POS tags, the other entity labels and 3-gram overlap. Dropped because they repeat another feature (Spearman ρ ≥ 0.84 in v1): character and syllable counts and syllables per word (`n_words`, `chars_per_word`, Flesch), top-100 share (`stopword_share`) and `CARDINAL` entities (`pos_NUM`). Dropped because they measure length: sentence count and type-token ratio (MTLD measures lexical diversity without the length effect). `n_words` stays in the CSV as a control column (`CONTROL_COLUMNS`): the generation prompt fixes the length, so it is not a style feature. Use it to check that the synthetic abstracts match the length, and leave it out of the PCA and the classifier.

`analysis.disclosure` does not overwrite an existing output (it may hold manual labels) unless you pass `--force`.

To label the hits, open the QA page and label each paper (keys `1`–`4`, arrows to move). Every click saves to the CSV:

```bash
docker compose up -d qa      # then open http://localhost:8000
docker compose stop qa       # when done
```

The page lives in `qa/` (`server.py` and `index.html`).

Labels: `used_ai` (the authors disclose AI use), `no_ai_statement` (they state that they used no AI), `not_disclosure` (the paper studies or only mentions AI), `unsure`. The label and note go to every row of the paper.

## Notebooks (`notebook/`)

`feature_pca.ipynb`: PCA of the abstract features, by year (no ground truth yet, so the year is the proxy). It is saved with its outputs.

```bash
docker compose up -d notebook    # then open http://localhost:8888 (no token: the port only listens on this computer)
docker compose stop notebook
# Re-run from the command line after the features change:
docker compose run --rm --no-deps -w /app/notebook app -m jupyter nbconvert --to notebook --execute --inplace feature_pca.ipynb
```
