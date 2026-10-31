"""Loads the scraper metadata CSVs into a {paper_id: row} mapping."""

from __future__ import annotations

import csv
import io
from pathlib import Path

METADATA_GLOB = "metadata_*.csv"

# Columns we use. The CSV `pdf_path`, `abstract`, `discussion` and `conclusion`
# columns are deliberately dropped (see docs/parser_plan.md, Section 3).
METADATA_COLUMNS = ("year", "paper_id", "title", "authors", "pages", "doi", "pdf_url")


def load_metadata(metadata_dir: Path) -> dict[str, dict[str, str]]:
    """Load every metadata CSV in `metadata_dir` into {paper_id: row}."""
    csv_paths = sorted(metadata_dir.glob(METADATA_GLOB))
    if not csv_paths:
        raise FileNotFoundError(f"No {METADATA_GLOB} files found in {metadata_dir}")

    metadata: dict[str, dict[str, str]] = {}
    for csv_path in csv_paths:
        for row in _read_csv(csv_path):
            paper_id = row["paper_id"]
            if paper_id in metadata:
                raise ValueError(f"Duplicate paper_id {paper_id} in {csv_path}")
            metadata[paper_id] = {column: row[column] for column in METADATA_COLUMNS}
    return metadata


def _read_csv(csv_path: Path) -> list[dict[str, str]]:
    """Read one CSV, stripping NUL bytes that would break the csv module."""
    raw = csv_path.read_text(encoding="utf-8").replace("\0", "")
    return list(csv.DictReader(io.StringIO(raw, newline="")))
