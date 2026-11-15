"""Tests for TextCleaner."""

from __future__ import annotations

import pytest

from parsing.cleaner import TextCleaner
from parsing.models import Paper, Section


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


# 5b. ISMIR copyright footnote

FOOTNOTE = ('Licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0). '
            'Attribution: E. Quinton, "Equivariant self-supervision for musical tempo estimation", '
            'in Proc. of the 23rd Int. Society for Music Information Retrieval Conf., Bengaluru, India, 2022.')


@pytest.mark.parametrize(
    "raw, expected",
    [
        (f"downstream tasks © E. Quinton. {FOOTNOTE}\n\nthat are", "downstream tasks\n\nthat are"),
        # GROBID turned "©" into "c": the author names stay, the rest goes
        (f"the problem c K. Tanaka. {FOOTNOTE} We use", "the problem c K. Tanaka. We use"),
        (FOOTNOTE.replace("Conf., Bengaluru, India", "Conf., Online"), ""),
    ],
)
def test_removes_license_footnote(raw: str, expected: str) -> None:
    assert clean(raw) == expected


def test_keeps_dataset_license_mentions() -> None:
    text = "released under a Creative Commons Attribution 4.0 International License for further research."
    assert clean(text) == text


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


# 8. Letter-by-letter noise

# Real run from 2020_000099 ("Music Interpolation"): LaTeX image data in the PDF.
REAL_CHAR_RUN = (
    '1 _ b a s e 6 4 = " I 3 E b f 2 S z I Q t T L R O c p h 3 h N W W H c l U = " > '
    "A A A C 1 n i c j V H L S s N A F D 2 N r 1 p f q S 7 d B I v g o p R E B F 0 W"
)


def test_removes_real_char_run_and_counts_its_tokens() -> None:
    raw = f"The interpolated melody is shown below. {REAL_CHAR_RUN} We then evaluate it."
    text, removed = TextCleaner().clean_with_stats(raw)
    assert text == "The interpolated melody is shown below. We then evaluate it."
    assert removed == len(REAL_CHAR_RUN.split())


@pytest.mark.parametrize(
    "text",
    [
        "the axes x y z of the space",
        "with variables a, b and c where a b c d e f g h are notes",
        "the chord C E G and the scale C D E F G A B C",
        # 29 single-character tokens: one below the threshold
        " ".join("abcdefghijklmnopqrstuvwxyzabc"),
    ],
)
def test_keeps_short_single_character_sequences(text: str) -> None:
    cleaned, removed = TextCleaner().clean_with_stats(text)
    assert cleaned == text
    assert removed == 0


def test_char_run_removal_keeps_paragraphs() -> None:
    raw = f"First paragraph.\n\n{REAL_CHAR_RUN}\n\nSecond paragraph."
    assert clean(raw) == "First paragraph.\n\nSecond paragraph."


def test_clean_paper_records_removed_tokens() -> None:
    paper = Paper(
        paper_id="2020_000099",
        year=2020,
        abstract="An abstract.",
        sections=[Section(heading="Music Interpolation", number="4", location="body",
                          text=f"Text. {REAL_CHAR_RUN} More text.")],
    )
    TextCleaner().clean_paper(paper)
    assert paper.sections[0].text == "Text. More text."
    assert paper.artifacts["char_run_tokens_removed"] == len(REAL_CHAR_RUN.split())


def test_all_steps_together() -> None:
    raw = "We pro- pose a data-driven classiﬁer [3, 4] , see https://github. com/a/b ."
    assert clean(raw) == "We propose a data-driven classifier, see."
