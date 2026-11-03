"""Tests for analysis.generate (export never shows the abstract; save never overwrites silently)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import analysis.generate as generate_module
from analysis.generate import generate, message_for, paper_text, prompt_for, sample, save, strip_label
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


def test_message_is_prompt_then_paper_without_abstract() -> None:
    message = message_for(PAPER)
    assert message.startswith(prompt_for(PAPER) + "\n\n---\n\n") and "Secret" not in message


def test_sample_is_fixed_and_extends() -> None:
    papers = [Paper(paper_id=f"{year}_{i:06d}", year=year) for year in (2020, 2021) for i in range(10)]
    two, five = sample(papers, 2), sample(papers, 5)
    assert two == sample(list(reversed(papers)), 2)
    assert two == five[:2] + five[5:7] and len(five) == 10


def fake_claude(tmp_path: Path, thinking_tokens: int) -> str:
    result = {"is_error": False, "result": "Synthetic.", "duration_ms": 5, "modelUsage": {"claude-sonnet-5-5": {}},
              "usage": {"input_tokens": 1, "cache_creation_input_tokens": 2, "cache_read_input_tokens": 3,
                        "output_tokens": 4, "output_tokens_details": {"thinking_tokens": thinking_tokens}}}
    script = tmp_path / "claude"
    script.write_text(f"#!/bin/sh\ncat > /dev/null\necho '{json.dumps(result)}'\n")
    script.chmod(0o755)
    return str(script)


def test_generate_reads_cli_output_and_rejects_thinking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate_module, "WORKDIR", tmp_path / "work")
    abstract, meta = generate(PAPER, fake_claude(tmp_path, 0))
    assert (abstract, meta["input_tokens"], meta["output_tokens"]) == ("Synthetic.", 6, 4)
    with pytest.raises(RuntimeError, match="thinking"):
        generate(PAPER, fake_claude(tmp_path, 7))


@pytest.mark.parametrize("raw", ["**Abstract**\n\nText here.", "**Abstract.** Text here.", "## Abstract\nText here.",
                                 "Abstract: Text here.", "Text here.", "**A Title: Sub**\n\nText here.",
                                 "**Title**\n\n**Abstract**\n\nText here."])
def test_strip_label(raw: str) -> None:
    assert strip_label(raw) == "Text here."


def test_strip_label_keeps_words_starting_with_abstract() -> None:
    assert strip_label("Abstraction is key.") == "Abstraction is key."


def test_strip_label_keeps_inline_bold() -> None:
    assert strip_label("**Bold start** of a sentence.") == "**Bold start** of a sentence."
