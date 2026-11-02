"""Tests for qa.server (labels are written to every row of a paper; other rows stay)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from analysis.disclosure import COLUMNS
from qa.server import load, papers, save_label


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({column: "" for column in COLUMNS} | row for row in rows)


def test_save_label_sets_every_row_of_the_paper(tmp_path: Path) -> None:
    path = tmp_path / "hits.csv"
    write_rows(path, [
        {"paper_id": "2024_000001", "sentence": "We used ChatGPT."},
        {"paper_id": "2024_000001", "sentence": "LLMs, again."},
        {"paper_id": "2024_000002", "sentence": "Other paper.", "label": "unsure", "note": "keep me"},
    ])
    save_label(path, "2024_000001", "used_ai", "in the acknowledgments")
    rows = load(path)
    assert [(r["label"], r["note"]) for r in rows] == [
        ("used_ai", "in the acknowledgments"), ("used_ai", "in the acknowledgments"), ("unsure", "keep me"),
    ]
    assert not (tmp_path / "hits.csv.tmp").exists()


@pytest.mark.parametrize("paper_id, label", [("2024_000001", "maybe"), ("2099_000001", "unsure")])
def test_save_label_rejects_bad_input(tmp_path: Path, paper_id: str, label: str) -> None:
    path = tmp_path / "hits.csv"
    write_rows(path, [{"paper_id": "2024_000001"}])
    with pytest.raises(ValueError):
        save_label(path, paper_id, label, "")
    assert load(path)[0]["label"] == ""


def test_papers_groups_rows_and_finds_spans() -> None:
    rows = [{column: "" for column in COLUMNS} | {"paper_id": "p", "sentence": "We used ChatGPT."}] * 2
    (paper,) = papers(rows)
    assert len(paper["hits"]) == 2
    assert paper["hits"][0]["spans"] == [(8, 15)]
