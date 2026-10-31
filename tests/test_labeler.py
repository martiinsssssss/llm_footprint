"""Tests for SectionLabeler.

The real cases are section sequences produced by GROBID for papers in
parsing/sample_ids.txt, written as (number, heading, location, grobid_type)
so the tests don't need GROBID.
"""

from __future__ import annotations

import pytest

from parsing.labeler import SectionLabeler
from parsing.models import Section

SectionTuple = tuple[str | None, str, str, str | None]

INTRO = ["introduction"]
RELATED = ["related_work"]
BODY = ["body"]
DISC = ["discussion"]
CONC = ["conclusion"]
ACK = ["acknowledgments"]
ETHICS = ["ethics"]
APPX = ["appendix"]


def labels_of(rows: list[SectionTuple]) -> list[list[str]]:
    sections = [
        Section(heading=heading, number=number, location=location, grobid_type=grobid_type)
        for number, heading, location, grobid_type in rows
    ]
    return SectionLabeler().label(sections)


def check(expected: list[tuple[SectionTuple, list[str]]]) -> None:
    rows = [row for row, _ in expected]
    for (row, want), got in zip(expected, labels_of(rows)):
        assert got == want, f"{row}: expected {want}, got {got}"


# --- Keyword table ----------------------------------------------------------


@pytest.mark.parametrize(
    "heading, expected",
    [
        ("INTRODUCTION", INTRO),
        ("Related works", RELATED),
        ("BACKGROUND", RELATED),
        ("PRIOR WORK AND PREREQUISITES", RELATED),
        ("DISCUSSION", DISC),
        ("CONCLUSIONS", CONC),
        ("Concluding remarks", CONC),
        ("ACKNOWLEDGMENTS", ACK),
        ("ACKNOWLEDGEMENTS", ACK),
        ("ACKNOWLEDGMENT", ACK),
        ("Acknowledgement", ACK),
        ("ETHICS STATEMENT", ETHICS),
        ("APPENDIX", APPX),
        # Combined headings get several labels, in table order.
        ("DISCUSSION AND CONCLUSION", ["discussion", "conclusion"]),
        ("DISCUSSION AND CONCLUSIONS", ["discussion", "conclusion"]),
        ("INTRODUCTION AND RELATED WORK", ["introduction", "related_work"]),
        ("BACKGROUND AND RELATED WORK", RELATED),
        ("RESULTS AND DISCUSSION", DISC),
        ("CONCLUSION AND FUTURE WORK", CONC),
        ("CONCLUSIONS AND FUTURE WORK", CONC),
        ("EXPERIMENTS AND RESULTS", []),
        ("PROPOSED METHOD", []),
    ],
)
def test_keyword_labels(heading: str, expected: list[str]) -> None:
    assert SectionLabeler().keyword_labels(heading) == expected


@pytest.mark.parametrize(
    "heading",
    # Real misspellings: 2021_000046, 2022_000005, 2022_000052, 2022_000085, 2024_000072
    ["ACKNOLWEDGEMENTS", "ACKNOWLEGMENT", "AKNOWLEDGEMENTS", "ACKNOLEDGEMENT", "ACKNOWLEGDEMENTS"],
)
def test_misspelled_acknowledgments(heading: str) -> None:
    assert SectionLabeler().keyword_labels(heading) == ACK
    check([(("5", "CONCLUSION", "body", None), CONC),
           (("6", heading, "body", None), ACK),
           (("7", heading, "back", None), ACK)])


@pytest.mark.parametrize("heading", ["ALIGNMENT", "ACCOMPANIMENT", "KNOWLEDGE GRAPHS"])
def test_similar_words_are_not_acknowledgments(heading: str) -> None:
    assert SectionLabeler().keyword_labels(heading) == []


def test_body_only_when_nothing_matches() -> None:
    check([(("1", "INTRODUCTION", "body", None), INTRO),
           (("2", "EXPERIMENTS AND RESULTS", "body", None), BODY)])


# --- Subsections and unnumbered sections -------------------------------------


def test_subsection_inherits_parent_labels() -> None:
    check([
        (("5", "CONCLUSION", "body", None), CONC),
        (("6", "DISCUSSION", "body", None), DISC),
        (("6.1", "Limitations of the study", "body", None), DISC),
        (("6.2", "Future work", "body", None), CONC),  # own keyword wins
    ])


def test_subsection_with_unknown_parent_is_body() -> None:
    check([(("1", "Introduction", "body", None), INTRO),
           (("2.1", "Symbolic Music Datasets", "body", None), BODY)])


def test_unnumbered_section_inherits_previous_labels() -> None:
    check([
        (("4", "DISCUSSION", "body", None), DISC),
        ((None, "Cadential Resolution:", "body", None), DISC),
        ((None, "Another paragraph heading", "body", None), DISC),
    ])


def test_false_top_level_number_inherits_previous_labels() -> None:
    check([
        (("4", "DISCUSSION", "body", None), DISC),
        (("2", "Token Selection.", "body", None), DISC),  # 2 after 4: false heading
        (("4.1", "Limitations", "body", None), DISC),
        (("5", "CONCLUSION", "body", None), CONC),
    ])


# --- Appendices and back matter -------------------------------------------------


def test_letter_heading_starts_appendix_region() -> None:
    check([
        (("5", "CONCLUSION", "body", None), CONC),
        ((None, "A.1 Prompt templates", "body", None), APPX),
        ((None, "Further discussion", "body", None), APPX),
        (("6", "ETHICS STATEMENT", "body", None), ETHICS),
        ((None, "B. More results", "body", None), APPX),
    ])


def test_letter_heading_before_any_numbered_section_is_not_appendix() -> None:
    check([((None, "A. Smith's approach", "body", None), BODY),
           (("1", "INTRODUCTION", "body", None), INTRO)])


def test_backfill_extends_appendix_over_unnumbered_sections_before_it() -> None:
    rows: list[SectionTuple] = [
        ("6", "DISCUSSION", "body", None),
        ("7", "CONCLUSION", "body", None),
        (None, "Model output one", "body", None),
        (None, "Model output two", "body", None),
        (None, "B. MORE EXAMPLES", "body", None),
    ]
    result = SectionLabeler().label_details(
        [Section(heading=h, number=n, location=loc, grobid_type=t) for n, h, loc, t in rows]
    )
    assert result.labels == [DISC, CONC, APPX, APPX, APPX]
    assert result.backfilled == [2, 3]


def test_backfill_stops_at_keyword_section() -> None:
    check([
        (("7", "CONCLUSION", "body", None), CONC),
        ((None, "Limitations.", "body", None), CONC),
        ((None, "Future work", "body", None), CONC),  # keyword: stops the backfill
        ((None, "Model output", "body", None), APPX),
        ((None, "A. EXAMPLES", "body", None), APPX),
    ])


def test_no_backfill_without_appendix_region() -> None:
    check([
        (("7", "CONCLUSION", "body", None), CONC),
        ((None, "Limitations.", "body", None), CONC),
        (("8", "ACKNOWLEDGMENTS", "back", "acknowledgement"), ACK),
    ])


@pytest.mark.parametrize(
    "heading",
    ["ETHICS STATEMENT", "Ethical Considerations", "ETHICS", "7. Ethics"],
)
def test_ethics_statement_headings_are_ethics_anywhere(heading: str) -> None:
    check([(("1", "INTRODUCTION", "body", None), INTRO),
           (("5", "DISCUSSION", "body", None), DISC),
           (("6", heading, "body", None), ETHICS)])


def test_other_ethic_headings_need_to_follow_conclusion() -> None:
    check([
        (("2", "THE ETHICS OF MUSIC AI", "body", None), BODY),
        (("2.1", "Fairness", "body", None), BODY),
        (("3", "CONCLUSION", "body", None), CONC),
        (("4", "Ethics and Broader Impact", "body", None), ETHICS),
    ])


def test_ethic_heading_in_back_matter_is_ethics() -> None:
    check([(("5", "DISCUSSION", "body", None), DISC),
           ((None, "Ethics and broader impact", "back", "annex"), ETHICS)])


def test_back_sections_are_appendix_unless_acknowledgments_or_ethics() -> None:
    check([
        (("7", "ACKNOWLEDGMENTS", "back", "acknowledgement"), ACK),
        ((None, "acknowledgement", "back", "acknowledgement"), ACK),
        (("8", "ETHICS STATEMENT", "back", "availability"), ETHICS),
        ((None, "contribution", "back", "contribution"), APPX),
        (("3.3.5", "Viterbi Training", "back", "annex"), APPX),
    ])


# --- Real cases from the test set -------------------------------------------------


def test_2025_000062_false_numbered_headings_and_annex() -> None:
    check([
        (("1", "INTRODUCTION", "body", None), INTRO),
        (("2", "RELATED WORK", "body", None), RELATED),
        (("3", "BACKGROUND", "body", None), RELATED),
        (("3.1", "MAGNeT's inference", "body", None), RELATED),
        (("3.2", "Rescoring", "body", None), RELATED),
        (("3.3", "Hybrid MAGNeT", "body", None), RELATED),
        (("4", "METHOD", "body", None), BODY),
        (("4.1", "Iterative overview", "body", None), BODY),
        (("4.2", "Inference algorithm", "body", None), BODY),
        # Algorithm steps that GROBID turned into numbered headings.
        (("2", "MAGNeT Inference.", "body", None), BODY),
        (("3", "Token Selection.", "body", None), BODY),
        (("4", "Repeat until completion.", "body", None), BODY),
        (("5", "Extract the final loop.", "body", None), BODY),
        (("4.3", "Hybrid variant", "body", None), BODY),
        (("4.4", "Signature-aware length control", "body", None), BODY),
        (("5", "EVALUATION METRICS", "body", None), BODY),
        ((None, "Cross-Entropy", "body", None), BODY),
        ((None, "MusicGen Naive", "body", None), BODY),
        (("5.1", "Seam perplexity", "body", None), BODY),
        (("6", "EXPERIMENTS", "body", None), BODY),
        (("6.3", "Final results", "body", None), BODY),
        ((None, "2512.39", "body", None), BODY),
        ((None, "56.17", "body", None), BODY),
        (("6.4", "Human evaluation", "body", None), BODY),
        (("7", "CONCLUSIONS", "body", None), CONC),
        (("8", "ACKNOWLEDGMENTS", "back", "acknowledgement"), ACK),
        ((None, "A. SEAM PERPLEXITY'S ERROR MARGINS", "back", "annex"), APPX),
        ((None, "B. 10 SECONDS EXPERIMENTS", "back", "annex"), APPX),
        ((None, "Mean Model", "back", "annex"), APPX),
        ((None, "310.21", "back", "annex"), APPX),
        ((None, "C. BEAT ALIGNMENT ALGORITHM VISUALIZATION", "back", "annex"), APPX),
        ((None, "D. MAGNET'S INFERENCE", "back", "annex"), APPX),
    ])


def test_2025_000029_appendix_in_body_and_annex() -> None:
    check([
        (("1", "INTRODUCTION", "body", None), INTRO),
        (("2", "RELATED WORK", "body", None), RELATED),
        (("2.1", "LALMs", "body", None), RELATED),
        (("3", "REASONING IS ENOUGH TO SOLVE CURRENT MUSIC QA BENCHMARK", "body", None), BODY),
        ((None, "It starts by eliminating impossible answers:", "body", None), BODY),
        (("4", "CONSTRUCTING RULISTENING", "body", None), BODY),
        (("4.1", "Methodology", "body", None), BODY),
        (("5", "RESULTS", "body", None), BODY),
        (("6", "DISCUSSION", "body", None), DISC),
        (("7", "CONCLUSION", "body", None), CONC),
        # Appendix A lost its heading in GROBID; these LLM responses get
        # appendix through the backfill rule.
        ((None, "Mixtral 8x22B (correct):", "body", None), APPX),
        ((None, "My guess is: (D) Llama 3.1 70B (correct):", "body", None), APPX),
        ((None, "B. RESPONSE EXAMPLES FOR LALMS", "body", None), APPX),
        ((None, "Audio", "body", None), APPX),
        (("8", "ACKNOWLEDGMENTS", "back", "acknowledgement"), ACK),
        ((None, "C.2.2 Sample 2: Question ID 1185", "back", "annex"), APPX),
        ((None, "C.3 Distractor Details", "back", "annex"), APPX),
    ])


def test_2024_000075_appendix_in_body_and_unnumbered_headings() -> None:
    check([
        (("1", "INTRODUCTION", "body", None), INTRO),
        (("2", "METHOD", "body", None), BODY),
        (("2.1", "User Prompt Set Curation", "body", None), BODY),
        ((None, "Prompt", "body", None), BODY),
        (("2.3.1", "Agent Role Assignment", "body", None), BODY),
        ((None, "Melody Agent:", "body", None), BODY),
        (("3", "EXPERIMENTS", "body", None), BODY),
        (("3.3", "Results", "body", None), BODY),
        (("4", "DISCUSSION", "body", None), DISC),
        ((None, "Cadential Resolution:", "body", None), DISC),
        (("5", "CONCLUSION", "body", None), CONC),
        ((None, "A.1 Single agent role-play Role-play Prompting with Additional Music Knowledge",
          "body", None), APPX),
        ((None, "Table 4.", "body", None), APPX),
        ((None, "Single-agent In-context learning prompting method", "body", None), APPX),
        ((None, "A.2 Melody Agent Prompt", "body", None), APPX),
        ((None, "Melody Agent Prompt", "back", "annex"), APPX),
    ])


def test_2022_000002_contribution_statement_is_appendix() -> None:
    check([
        (("1", "INTRODUCTION", "body", None), INTRO),
        (("2", "OVERVIEW OF THE STATE-OF-THE-ART", "body", None), BODY),
        (("2.1", "Feature extraction", "body", None), BODY),
        (("4", "EVALUATION", "body", None), BODY),
        (("4.3", "Result", "body", None), BODY),
        (("5", "CONCLUSION", "body", None), CONC),
        ((None, "contribution", "back", "contribution"), APPX),
    ])


def test_2021_000037_paper_about_ethics_is_not_an_ethics_statement() -> None:
    check([
        (("1", "INTRODUCTION", "body", None), INTRO),
        (("2", "A REVIEW", "body", None), BODY),
        (("3", "NON-WESTERN APPROACHES TO TECHNOLOGICAL ETHICS", "body", None), BODY),
        (("4", "TOWARD AN ETHICALLY ALIGNED DESIGN FOR AI IN EAST ASIA", "body", None), BODY),
        (("4.1", '"Human Rights"', "body", None), BODY),
        (("4.3", '"Awareness of Misuse"', "body", None), BODY),
        (("6", "CONCLUSION", "body", None), CONC),
        (("7", "ACKNOWLEDGEMENTS", "back", "acknowledgement"), ACK),
    ])


def test_2024_000108_empty_first_heading_is_introduction() -> None:
    check([
        ((None, "", "body", None), INTRO),
        (("2.1", "Neural Semi-CRF for Piano Transcription", "body", None), BODY),
        (("3", "Revisiting Interval Scoring for Semi-CRFs", "body", None), BODY),
        (("3.1", "Interval Scoring in [2]", "body", None), BODY),
        (("5.4", "Results", "body", None), BODY),
        ((None, "Scoring Methods Comparison.", "body", None), BODY),
        (("6", "Conclusion", "body", None), CONC),
        (("7", "Acknowledgement", "back", "acknowledgement"), ACK),
    ])
