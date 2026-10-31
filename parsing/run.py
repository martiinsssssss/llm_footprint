"""Command-line entry point: PDFs -> parse, label, clean, check -> JSONL + quality report."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from parsing.cleaner import TextCleaner
from parsing.grobid_parser import (
    GROBID_URL,
    PAPER_ID_PATTERN,
    GrobidParser,
    paper_id_and_year,
    pdf_path_for,
    read_ids,
)
from parsing.labeler import SectionLabeler
from parsing.metadata import load_metadata
from parsing.models import Paper, write_jsonl
from parsing.quality import QualityChecker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse ISMIR PDFs into a JSONL file.")
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, help="only process these paper ids")
    parser.add_argument("--years", type=int, nargs="+", help="only process these years")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="ignore the TEI cache")
    return parser.parse_args()


def find_pdfs(pdf_dir: Path, ids_file: Path | None, years: list[int] | None) -> list[Path]:
    """PDF paths to process: from the ids file, or every `{year}/{paper_id}.pdf`."""
    if ids_file is not None:
        paths = [pdf_path_for(pdf_dir, paper_id) for paper_id in read_ids(ids_file)]
    else:
        paths = [p for p in pdf_dir.glob("*/*.pdf") if PAPER_ID_PATTERN.match(p.stem)]
    if years:
        paths = [p for p in paths if paper_id_and_year(p)[1] in years]
    return sorted(paths)


def grobid_is_alive() -> bool:
    try:
        response = requests.get(f"{GROBID_URL}/api/isalive", timeout=5)
    except requests.RequestException:
        return False
    return response.ok and response.text.strip() == "true"


def process_one(
    pdf_path: Path, grobid: GrobidParser, metadata: dict[str, dict[str, str]]
) -> tuple[Paper, str | None]:
    """Process one PDF. A failure gives a metadata-only record with the `parse_error` warning."""
    paper_id, year = paper_id_and_year(pdf_path)
    error = None
    try:
        paper = grobid.parse(pdf_path)
        SectionLabeler().label_paper(paper)
        TextCleaner().clean_paper(paper)
    except Exception as exc:  # one bad PDF must not stop the run
        error = f"{type(exc).__name__}: {exc}"
        paper = Paper(paper_id=paper_id, year=year, warnings=["parse_error"])
    row = metadata.get(paper_id)
    if row is None:
        paper.warnings.append("no_metadata")
    else:
        paper.title, paper.authors, paper.doi = row["title"], row["authors"], row["doi"]
    QualityChecker().check(paper)
    return paper, error


def main() -> None:
    args = parse_args()
    if not grobid_is_alive():
        print(f"Warning: GROBID is not reachable at {GROBID_URL}; "
              "only PDFs with a cached TEI can be parsed.")
    metadata = load_metadata(args.metadata_dir)
    pdf_paths = find_pdfs(args.pdf_dir, args.ids_file, args.years)
    print(f"Processing {len(pdf_paths)} PDFs with {args.workers} workers")

    start = time.monotonic()
    grobid = GrobidParser(force=args.force)
    papers: list[Paper] = []
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one, path, grobid, metadata) for path in pdf_paths]
        for done, future in enumerate(as_completed(futures), start=1):
            paper, error = future.result()
            papers.append(paper)
            if error:
                errors[paper.paper_id] = error
            status = f"parse_error ({error})" if error else f"{len(paper.sections)} sections"
            print(f"[{done}/{len(pdf_paths)}] {paper.paper_id}: {status}", file=sys.stderr, flush=True)
    write_jsonl(papers, args.out)

    per_year = Counter(paper.year for paper in papers)
    warnings = Counter(warning for paper in papers for warning in paper.warnings)
    print(f"\nDone in {time.monotonic() - start:.0f}s: {len(papers)} papers -> {args.out}")
    print("Per year: " + ", ".join(f"{year}: {n}" for year, n in sorted(per_year.items())))
    print("Warnings: " + (", ".join(f"{w}: {n}" for w, n in sorted(warnings.items())) or "none"))
    for paper_id, error in sorted(errors.items()):
        print(f"  parse_error {paper_id}: {error}")
    QualityChecker().save_report(papers, args.out)

    if args.ids_file is None and not args.years:
        missing = sorted(set(metadata) - {path.stem for path in pdf_paths})
        print(f"Metadata rows without a PDF: {len(missing)}" + (f" {missing}" if missing else ""))


if __name__ == "__main__":
    main()
