"""Tests for TextCleaner."""

from __future__ import annotations

import pytest

from parsing.cleaner import TextCleaner


def clean(text: str) -> str:
    return TextCleaner().clean(text)


# 1. Unicode NFKC


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("classiﬁes", "classifies"),
        ("ﬂow", "flow"),
        ("eﬀect", "effect"),
        ("oﬃcial", "official"),
        ("full width", "full width"),  # non-breaking space
    ],
)
def test_nfkc_normalization(raw: str, expected: str) -> None:
    assert clean(raw) == expected


# 2. NUL, control characters, replacement character


def test_removes_nul_control_and_replacement_characters() -> None:
    assert clean("da\x00ta se\x07t mu�sic") == "data set music"


# 3. Line-break hyphens


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("clas- sifies", "classifies"),
        ("pro- posed method", "proposed method"),
        ("Re- current", "Recurrent"),
    ],
)
def test_joins_line_break_hyphens(raw: str, expected: str) -> None:
    assert clean(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "state-of-the-art",
        "data-driven",
        "pitch-shifting",
        "text-to-music",
        "Text-Only LMs",
        "a 2- or 3-voice texture",  # digit before the hyphen
        "Section 2 - Results",  # spaced dash
        "MIR- Based",  # uppercase after the space
    ],
)
def test_keeps_compounds_and_dashes(text: str) -> None:
    assert clean(text) == text


# 4. Citation brackets


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("as shown in [12].", "as shown in."),
        ("previous work [3, 4] has", "previous work has"),
        ("models [5-7] and", "models and"),
        ("models [5–7] and", "models and"),
        ("see [1, 3–5; 9]", "see"),
    ],
)
def test_removes_citation_brackets(raw: str, expected: str) -> None:
    assert clean(raw) == expected


def test_keeps_non_citation_brackets() -> None:
    assert clean("the interval [0, 1) and [x] values") == "the interval [0, 1) and [x] values"


# 5. URLs and e-mail addresses


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Code at https://github.com/user/repo.", "Code at."),
        ("see http://example.org/a?b=1 for", "see for"),
        ("visit www.ismir.net today", "visit today"),
        # URLs split by GROBID after a dot or a slash
        ("scripts at https://github. com/yongyizang/AreYouReallyListening and",
         "scripts at and"),
        ("RUL-MuChoMusic at https://huggingface.co/ datasets/yongyizang/RUListening.",
         "RUL-MuChoMusic at."),
        # A sentence-final URL must not swallow the next word
        ("available at https://example.org/. we also", "available at. we also"),
        ("mail john.doe@upf.edu for details", "mail for details"),
    ],
)
def test_removes_urls_and_emails(raw: str, expected: str) -> None:
    assert clean(raw) == expected


# 6. Empty brackets and spaces before punctuation


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("in music , as well", "in music, as well"),
        ("in music [1], [2] , as well", "in music, as well"),
        ("the method ( ) works", "the method works"),
        ("as reported ([4], [5]).", "as reported."),
        ("results [ ] here", "results here"),
        ("Is it true ? Yes !", "Is it true? Yes!"),
        ("in the MuChoMusic paper ([7], see Fig. 6b)", "in the MuChoMusic paper (see Fig. 6b)"),
        ("such as [3], [4]; or", "such as; or"),
        # decimals and ellipses stay intact
        ("with α=0.5, .154 with α=1", "with α=0.5, .154 with α=1"),
        ("for n ∈ {1, ..., K}", "for n ∈ {1,..., K}"),
    ],
)
def test_removes_empty_brackets_and_space_before_punctuation(raw: str, expected: str) -> None:
    assert clean(raw) == expected


# 7. Whitespace


def test_normalizes_whitespace_but_keeps_paragraphs() -> None:
    raw = "  First   paragraph\nstill first.\n\n\n\tSecond\tparagraph.  "
    assert clean(raw) == "First paragraph still first.\n\nSecond paragraph."


def test_all_steps_together() -> None:
    raw = "We pro- pose a data-driven classiﬁer [3, 4] , see https://github. com/a/b ."
    assert clean(raw) == "We propose a data-driven classifier, see."
