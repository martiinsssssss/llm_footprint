"""Command-line entry point for running the parser pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

from parsing.cleaner import TextCleaner
from parsing.grobid_parser import (
    GROBID_URL,
    PAPER_ID_PATTERN,
    GrobidParser,
    pdf_path_for,
    read_ids,
)
from parsing.labeler import SectionLabeler
from parsing.metadata import load_metadata
from parsing.pipeline import DEFAULT_WORKERS, Pipeline
from parsing.quality import QualityChecker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse ISMIR PDFs into a JSONL file.")
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, help="only process these paper ids")
    parser.add_argument("--years", type=int, nargs="+", help="only process these years")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--force", action="store_true", help="ignore the TEI cache")
    return parser.parse_args()


def find_pdfs(pdf_dir: Path, ids_file: Path | None, years: list[int] | None) -> list[Path]:
    """PDF paths to process: from the ids file, or every `{year}/{paper_id}.pdf`."""
    if ids_file is not None:
        paths = [pdf_path_for(pdf_dir, paper_id) for paper_id in read_ids(ids_file)]
    else:
        paths = [p for p in pdf_dir.glob("*/*.pdf") if PAPER_ID_PATTERN.match(p.stem)]
    if years:
        paths = [p for p in paths if int(p.stem.split("_")[0]) in years]
    return sorted(paths)


def grobid_is_alive(server_url: str = GROBID_URL) -> bool:
    try:
        response = requests.get(f"{server_url}/api/isalive", timeout=5)
    except requests.RequestException:
        return False
    return response.ok and response.text.strip() == "true"


def main() -> None:
    args = parse_args()
    if not grobid_is_alive():
        print(f"Warning: GROBID is not reachable at {GROBID_URL}; "
              "only PDFs with a cached TEI can be parsed.")
    metadata = load_metadata(args.metadata_dir)
    pdf_paths = find_pdfs(args.pdf_dir, args.ids_file, args.years)
    print(f"Processing {len(pdf_paths)} PDFs with {args.workers} workers")

    checker = QualityChecker()
    pipeline = Pipeline(
        parser=GrobidParser(force=args.force),
        labeler=SectionLabeler(),
        cleaner=TextCleaner(),
        checker=checker,
        metadata=metadata,
        workers=args.workers,
    )
    papers = pipeline.run(pdf_paths, args.out)
    checker.save_report(papers, args.out)

    if args.ids_file is None and not args.years:
        processed = {path.stem for path in pdf_paths}
        missing = sorted(set(metadata) - processed)
        print(f"Metadata rows without a PDF: {len(missing)}" + (f" {missing}" if missing else ""))


if __name__ == "__main__":
    main()
