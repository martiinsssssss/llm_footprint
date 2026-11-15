"""Tests for analysis.generate (export never shows the abstract; save never overwrites silently)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.generate import paper_text, prompt_for, save
from parsing.models import Paper, Section

PAPER = Paper(paper_id="2024_000001", year=2024, title="A Title", abstract="Secret original abstract words here.",
              sections=[Section("1 INTRODUCTION", "1", "body", ["introduction"], "Body text.")])


def test_export_has_no_abstract() -> None:
    text = paper_text(PAPER)
    assert "Secret" not in text and "A Title" in text and "Body text." in text


def test_prompt_uses_original_length_or_override() -> None:
    assert "about 5 words" in prompt_for(PAPER)
    assert "about 150 words" in prompt_for(PAPER, 150)


def test_save_writes_record_and_refuses_overwrite(tmp_path: Path) -> None:
    path = save(PAPER, "  Synthetic text.\n", "test-model", out_dir=tmp_path)
    record = json.loads(path.read_text())
    assert (record["abstract"], record["model"], record["year"]) == ("Synthetic text.", "test-model", 2024)
    with pytest.raises(SystemExit):
        save(PAPER, "Other text.", "test-model", out_dir=tmp_path)
    save(PAPER, "Other text.", "test-model", out_dir=tmp_path, force=True)
    assert json.loads(path.read_text())["abstract"] == "Other text."
