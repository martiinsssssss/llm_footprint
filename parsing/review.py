"""Reports for reviewing the labeling rules on a parsed JSONL file.

Usage: python -m parsing.review data/processed/ismir_2020_2025_v1.jsonl

The labeler only uses headings, numbers and locations, which are all in the
JSONL, so it is re-run here to recover the backfill and appendix-start details.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from parsing.labeler import LabelResult, SectionLabeler
from parsing.models import Paper, read_jsonl

NUMERIC_NUMBER = re.compile(r"^\d+(\.\d+)*$")
STANDALONE_LABELS = {"acknowledgments", "ethics"}


def label_results(papers: list[Paper]) -> dict[str, LabelResult]:
    """Re-run the labeler and check it agrees with the stored labels."""
    labeler = SectionLabeler()
    results = {}
    for paper in papers:
        result = labeler.label_details(paper.sections)
        stored = [section.labels for section in paper.sections]
        if result.labels != stored:
            print(f"WARNING: {paper.paper_id} labels differ from the JSONL; re-run the parser")
        results[paper.paper_id] = result
    return results


def report_ethics(papers: list[Paper]) -> None:
    rows = [
        (paper.paper_id, section)
        for paper in papers
        for section in paper.sections
        if "ethics" in section.labels
    ]
    _title(f"1. Sections labeled ethics ({len(rows)} in "
           f"{len({paper_id for paper_id, _ in rows})} papers)")
    for paper_id, section in rows:
        print(f"  {paper_id}  {section.number or '-':>5}  {section.location:<4}  {section.heading}")
    _per_year_counts({paper_id for paper_id, _ in rows}, papers, "papers with ethics")


def report_backfill(papers: list[Paper], results: dict[str, LabelResult]) -> None:
    rows = [
        (paper.paper_id, index, paper.sections[index])
        for paper in papers
        for index in results[paper.paper_id].backfilled
    ]
    _title(f"2. Sections labeled appendix by backfill ({len(rows)})")
    for paper_id, index, section in rows:
        print(f"  {paper_id}  #{index + 1:<3} {section.heading}")


def report_appendix_starts(papers: list[Paper], results: dict[str, LabelResult]) -> None:
    body_rows, back_rows, flagged = [], [], 0
    for paper in papers:
        start, source = _appendix_start(paper, results[paper.paper_id])
        if start is None:
            continue
        has_conclusion_before = any(
            "conclusion" in section.labels for section in paper.sections[:start]
        )
        flag = "" if has_conclusion_before else "  <-- STARTS BEFORE THE CONCLUSION"
        flagged += not has_conclusion_before
        row = (f"  {paper.paper_id}  {start + 1:>3}/{len(paper.sections):<3} "
               f"{paper.sections[start].heading[:70]}{flag}")
        (body_rows if source == "body" else back_rows).append(row)

    _title(f"3. Where each appendix region starts ({len(body_rows) + len(back_rows)} papers, "
           f"{flagged} flagged)")
    print(f"  In the body (rule 2, including backfill): {len(body_rows)}")
    print("\n".join(body_rows))
    print(f"\n  Only in back matter: {len(back_rows)}")
    print("\n".join(back_rows))


def report_numeric_back_sections(papers: list[Paper]) -> None:
    rows = [
        (paper, section)
        for paper in papers
        for section in paper.sections
        if section.location == "back"
        and section.number
        and NUMERIC_NUMBER.match(section.number)
        and not STANDALONE_LABELS & set(section.labels)
    ]
    _title(f"4. Back sections with a numeric number (misplaced body text): {len(rows)}")
    for paper, section in rows:
        print(f"  {paper.paper_id}  {section.number:>7}  {section.heading[:60]}  "
              f"({len(section.text.split())} words)")
    words = Counter()
    for paper, section in rows:
        words[paper.year] += len(section.text.split())
    sections = Counter(paper.year for paper, _ in rows)
    affected = {paper.paper_id for paper, _ in rows}
    print("\n  Year  sections  papers  words")
    for year in sorted({paper.year for paper in papers}):
        n_papers = len({p.paper_id for p, _ in rows if p.year == year})
        print(f"  {year}  {sections[year]:>8}  {n_papers:>6}  {words[year]:>5}")
    print(f"  Total {len(rows):>8}  {len(affected):>6}  {sum(words.values()):>5}")


def _appendix_start(paper: Paper, result: LabelResult) -> tuple[int | None, str]:
    """Effective start of the appendix: body region (with backfill) or first back appendix."""
    if result.appendix_start is not None:
        return min([result.appendix_start, *result.backfilled]), "body"
    for index, section in enumerate(paper.sections):
        if section.location == "back" and "appendix" in section.labels:
            return index, "back"
    return None, ""


def _per_year_counts(paper_ids: set[str], papers: list[Paper], what: str) -> None:
    totals = Counter(paper.year for paper in papers)
    hits = Counter(paper.year for paper in papers if paper.paper_id in paper_ids)
    print(f"\n  {what} per year: " + ", ".join(
        f"{year}: {hits[year]}/{totals[year]}" for year in sorted(totals)))


def _title(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the label review reports.")
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()

    papers = read_jsonl(args.jsonl)
    results = label_results(papers)
    report_ethics(papers)
    report_backfill(papers, results)
    report_appendix_starts(papers, results)
    report_numeric_back_sections(papers)


if __name__ == "__main__":
    main()
