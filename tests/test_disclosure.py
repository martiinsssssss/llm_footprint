"""Tests for analysis.disclosure."""

from __future__ import annotations

import pytest

from analysis.disclosure import find_hits
from parsing.models import Paper, Section


def terms(text: str) -> list[str]:
    paper = Paper(paper_id="2024_000001", year=2024, sections=[Section("ACKNOWLEDGMENTS", None, "back", ["acknowledgments"], text)])
    return [row["terms"] for row in find_hits(paper)]


@pytest.mark.parametrize("text, expected", [
    ("We used ChatGPT to improve the language of this paper.", ["chatgpt"]),
    ("The text was checked with Grammarly.", ["grammarly"]),
    ("We used an AI tool to polish the writing.", ["ai_writing"]),
    ("We prompt GPT-4 and Claude with the lyrics.", ["gpt|claude"]),
    ("Large language models are popular. We thank our funders.", ["llm"]),
    ("We used generative AI for editing.", ["generative_ai|ai_writing"]),
])
def test_finds_terms(text: str, expected: list[str]) -> None:
    assert terms(text) == expected


@pytest.mark.parametrize("text", [
    "We analyze piano works by Claude Debussy.",
    "Following Claude Shannon, we measure entropy.",
    "We thank Claude Latour for supporting this research.",
    "The bard sang under the gemini constellation.",
    "We thank the reviewers for their comments.",
])
def test_ignores_false_positives(text: str) -> None:
    assert terms(text) == []


def test_row_has_context() -> None:
    paper = Paper(paper_id="2024_000001", year=2024, abstract="We study music. LLMs can write lyrics.")
    (row,) = find_hits(paper)
    assert (row["section"], row["sentence"], row["label"]) == ("abstract", "LLMs can write lyrics.", "")
