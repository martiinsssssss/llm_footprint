"""Orchestrates parsing, labeling, cleaning and quality checks (Pipeline)."""

from __future__ import annotations

import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol

from parsing.cleaner import TextCleaner
from parsing.grobid_parser import paper_id_and_year
from parsing.labeler import SectionLabeler
from parsing.models import Paper, write_jsonl

DEFAULT_WORKERS = 4


class Parser(Protocol):
    def parse(self, pdf_path: Path) -> Paper: ...


class QualityChecker(Protocol):
    def check(self, paper: Paper) -> None: ...


class Pipeline:
    """Turns PDFs into labeled, cleaned Papers and writes them to JSONL.

    Every component is passed in, so each one can be replaced or tested alone.
    `checker` may be None to skip the quality checks.
    """

    def __init__(
        self,
        parser: Parser,
        labeler: SectionLabeler,
        cleaner: TextCleaner,
        checker: QualityChecker | None,
        metadata: dict[str, dict[str, str]],
        workers: int = DEFAULT_WORKERS,
    ) -> None:
        self.parser = parser
        self.labeler = labeler
        self.cleaner = cleaner
        self.checker = checker
        self.metadata = metadata
        self.workers = workers
        self.errors: dict[str, str] = {}

    def process_one(self, pdf_path: Path) -> Paper:
        """Process one PDF. A failure gives a record with the `parse_error` warning."""
        paper_id, year = paper_id_and_year(pdf_path)
        try:
            paper = self.parser.parse(pdf_path)
            self.labeler.label_paper(paper)
            self.cleaner.clean_paper(paper)
        except Exception as error:  # one bad PDF must not stop the run
            self.errors[paper_id] = f"{type(error).__name__}: {error}"
            paper = Paper(paper_id=paper_id, year=year, warnings=["parse_error"])
        self._add_metadata(paper)
        if self.checker is not None:
            self.checker.check(paper)
        return paper

    def run(self, pdf_paths: list[Path], out_path: Path) -> list[Paper]:
        """Process all PDFs in parallel, write them sorted by paper_id, print a summary."""
        start = time.monotonic()
        papers: list[Paper] = []
        total = len(pdf_paths)
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(self.process_one, path) for path in pdf_paths]
            for done, future in enumerate(as_completed(futures), start=1):
                paper = future.result()
                papers.append(paper)
                _print_progress(done, total, paper, self.errors.get(paper.paper_id))
        write_jsonl(papers, out_path)
        self._print_summary(papers, out_path, time.monotonic() - start)
        return papers

    def _add_metadata(self, paper: Paper) -> None:
        row = self.metadata.get(paper.paper_id)
        if row is None:
            paper.warnings.append("no_metadata")
            return
        paper.title = row["title"]
        paper.authors = row["authors"]
        paper.doi = row["doi"]

    def _print_summary(self, papers: list[Paper], out_path: Path, seconds: float) -> None:
        per_year = Counter(paper.year for paper in papers)
        warnings = Counter(warning for paper in papers for warning in paper.warnings)
        print(f"\nDone in {seconds:.0f}s: {len(papers)} papers -> {out_path}")
        print("Per year: " + ", ".join(f"{year}: {n}" for year, n in sorted(per_year.items())))
        print("Warnings: " + (", ".join(f"{w}: {n}" for w, n in sorted(warnings.items())) or "none"))
        for paper_id, error in sorted(self.errors.items()):
            print(f"  parse_error {paper_id}: {error}")


def _print_progress(done: int, total: int, paper: Paper, error: str | None) -> None:
    status = f"parse_error ({error})" if error else f"{len(paper.sections)} sections"
    print(f"[{done}/{total}] {paper.paper_id}: {status}", file=sys.stderr, flush=True)
