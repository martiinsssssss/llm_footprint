"""Per-paper quality flags and artifact counters, plus the per-year quality report."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from parsing.models import Paper, read_jsonl

# A paper needs at least this many sections with a main-text label.
MIN_MAIN_SECTIONS = 3
MAIN_LABELS = {"introduction", "related_work", "body", "discussion", "conclusion"}

# Warnings, in report order. `parse_error` and `no_metadata` are set by parsing.run.
WARNINGS = (
    "parse_error",
    "no_metadata",
    "no_abstract",
    "few_sections",
    "no_introduction",
    "no_conclusion",
)

# Extraction artifacts, counted on the cleaned `full_text`.
ARTIFACT_PATTERNS: dict[str, re.Pattern[str]] = {
    "split_words": re.compile(
        r"\b(?:W e|W ith|W hen|W hile|W hat|W here|T o|T his|T he|T hese|T hus)\b"
    ),
    "hyphen_breaks": re.compile(r"\b[a-z]+- [a-z]+\b"),
    "ligatures": re.compile(r"[ﬀ-ﬆ]"),
    "bad_chars": re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f�]"),
}

# Set by TextCleaner: tokens removed as letter-by-letter noise (cleaning step 8).
CHAR_RUN_ARTIFACT = "char_run_tokens_removed"
REPORT_ARTIFACTS = (*ARTIFACT_PATTERNS, CHAR_RUN_ARTIFACT)

REPORT_FILE_TEMPLATE = "quality_report_{version}.csv"


class QualityChecker:
    """Adds quality warnings and artifact counts to papers and builds the per-year report."""

    def check(self, paper: Paper) -> None:
        """Fill `paper.warnings` and `paper.artifacts` in place."""
        if "parse_error" not in paper.warnings:
            for warning in self._content_warnings(paper):
                if warning not in paper.warnings:
                    paper.warnings.append(warning)
        paper.warnings.sort(key=WARNINGS.index)
        paper.artifacts = {
            **count_artifacts(paper.full_text),
            CHAR_RUN_ARTIFACT: paper.artifacts.get(CHAR_RUN_ARTIFACT, 0),
        }

    def _content_warnings(self, paper: Paper) -> list[str]:
        labels = [set(section.labels) for section in paper.sections]
        warnings = []
        if not paper.abstract.strip():
            warnings.append("no_abstract")
        if sum(bool(section_labels & MAIN_LABELS) for section_labels in labels) < MIN_MAIN_SECTIONS:
            warnings.append("few_sections")
        if not any("introduction" in section_labels for section_labels in labels):
            warnings.append("no_introduction")
        if not any("conclusion" in section_labels for section_labels in labels):
            warnings.append("no_conclusion")
        return warnings

    def report(self, papers: list[Paper]) -> pd.DataFrame:
        """One row per year plus a total row, with the columns described in the README."""
        rows = [_paper_row(paper) for paper in papers]
        table = pd.DataFrame(rows)
        per_year = [_summarize(group, str(year)) for year, group in table.groupby("year")]
        return pd.DataFrame(per_year + [_summarize(table, "total")]).set_index("year")

    def save_report(self, papers: list[Paper], jsonl_path: Path) -> Path:
        """Write the report as CSV next to the JSONL and print it."""
        version = papers[0].pipeline_version if papers else "v1"
        report_path = jsonl_path.parent / REPORT_FILE_TEMPLATE.format(version=version)
        report = self.report(papers)
        report.to_csv(report_path)
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(f"\nQuality report -> {report_path}\n{report.astype(object).T.to_string()}")
        return report_path


def count_artifacts(text: str) -> dict[str, int]:
    return {name: len(pattern.findall(text)) for name, pattern in ARTIFACT_PATTERNS.items()}


def _paper_row(paper: Paper) -> dict[str, object]:
    labels = {label for section in paper.sections for label in section.labels}
    row: dict[str, object] = {"year": paper.year, "n_words": paper.n_words}
    row.update({warning: warning in paper.warnings for warning in WARNINGS})
    row.update({name: paper.artifacts.get(name, 0) for name in REPORT_ARTIFACTS})
    row["has_acknowledgments"] = "acknowledgments" in labels
    row["has_ethics"] = "ethics" in labels
    return row


def _summarize(group: pd.DataFrame, year: str) -> dict[str, object]:
    total_words = group["n_words"].sum()
    summary: dict[str, object] = {
        "year": year,
        "n_papers": len(group),
        "parse_errors": int(group["parse_error"].sum()),
    }
    for warning in WARNINGS:
        summary[f"pct_{warning}"] = _pct(group[warning].mean())
    for name in REPORT_ARTIFACTS:
        per_1000 = group[name].sum() / total_words * 1000 if total_words else 0.0
        summary[f"{name}_per_1000_words"] = round(per_1000, 3)
    summary["pct_with_acknowledgments"] = _pct(group["has_acknowledgments"].mean())
    summary["pct_with_ethics"] = _pct(group["has_ethics"].mean())
    summary["median_n_words"] = int(group["n_words"].median())
    return summary


def _pct(fraction: float) -> float:
    return round(100 * fraction, 1)


def main() -> None:
    """Print and save the quality report of an existing JSONL (no PDFs needed)."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()
    papers = read_jsonl(args.jsonl)
    QualityChecker().save_report(papers, args.jsonl)


if __name__ == "__main__":
    main()
