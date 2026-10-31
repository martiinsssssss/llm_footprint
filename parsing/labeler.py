"""Maps section headings to section labels (SectionLabeler)."""

from __future__ import annotations

import argparse
import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from parsing.models import Paper, Section

# Label -> keywords searched (case-insensitively) anywhere in the heading.
# The order of this table is the order of labels in the output.
KEYWORDS: dict[str, tuple[str, ...]] = {
    "introduction": ("INTRODUCTION",),
    "related_work": ("RELATED WORK", "BACKGROUND", "PRIOR WORK", "PREVIOUS WORK", "LITERATURE"),
    "discussion": ("DISCUSSION",),
    "conclusion": ("CONCLUSION", "CONCLUDING", "FUTURE WORK"),
    "acknowledgments": ("ACKNOWLEDG",),
    "ethics": ("ETHIC",),
    "appendix": ("APPENDIX", "SUPPLEMENTARY"),
}
BODY = "body"
APPENDIX = "appendix"

# Misspelled acknowledgments ("ACKNOLWEDGEMENTS", "AKNOWLEDGEMENTS"): a heading
# that is a single word this similar to ACKNOWLEDGMENTS also gets the label.
# On the full corpus the misspellings score 0.90-0.93 and the next closest
# single-word heading ("ALIGNMENT") 0.58.
ACKNOWLEDGMENTS_WORD = "ACKNOWLEDGMENTS"
ACKNOWLEDGMENTS_MIN_SIMILARITY = 0.8

# Labels that keep their own identity inside back matter and appendices.
STANDALONE_LABELS = ("acknowledgments", "ethics")

# Appendix headings or numbers: a single capital letter followed by a dot,
# optionally with sub-numbers: "A.", "B. RESULTS", "A.1 Prompt", "C.2.3 ...".
APPENDIX_HEADING_PATTERN = re.compile(r"^[A-Z]\.(\d+(\.\d+)*\.?)?(\s|$)")
APPENDIX_NUMBER_PATTERN = re.compile(r"^[A-Z](\.\d+)*$")

# An ETHIC heading is an ethics statement wherever it appears if it also
# contains one of these words, or if it is just "ETHICS".
ETHICS_STATEMENT_WORDS = ("STATEMENT", "CONSIDERATION")


@dataclass
class LabelResult:
    """Labels of each section, plus where the body appendix region starts.

    `appendix_start` is the index of the section that started the body appendix
    region (rule 2), or None; `backfilled` are the indices that got `appendix`
    by backfill, all just before `appendix_start`.
    """

    labels: list[list[str]]
    backfilled: list[int]
    appendix_start: int | None = None


@dataclass
class _State:
    """What the labeler remembers while walking through a paper's sections."""

    previous_labels: list[str] = field(default_factory=lambda: [BODY])
    top_level_labels: dict[str, list[str]] = field(default_factory=dict)
    last_top_level: int | None = None
    in_appendix: bool = False
    seen_body: bool = False
    seen_conclusion: bool = False


class SectionLabeler:
    """Assigns labels to the sections of a paper, in document order.

    Rules, by precedence (see docs/parser_plan.md, Section 5):
    1. `back` sections: acknowledgments/ethics by keyword, everything else `appendix`.
    2. Appendix region: from the first body section whose number or heading starts
       with a single capital letter ("A.", "B.1"), or whose heading contains an
       appendix keyword, every body section is `appendix` unless it matches
       acknowledgments or ethics. Only applies after a numbered top-level section,
       so a paper numbered "I., II., ..." is not mistaken for an appendix.
    3. First body section with an empty heading: `introduction`.
    4. Headings that match keywords get those labels (several are possible).
    5. Valid numbered sections without keywords: top-level -> `body`; subsection ->
       the labels of its top-level parent, or `body` if the parent is unknown.
    6. Unnumbered sections without keywords, and false top-level numbers (not greater
       than the previous top-level number), inherit the previous section's labels.

    Two refinements, because wrongly including LLM-generated text or ethics
    statements in the style analyses is worse than wrongly excluding a paragraph:
    - Ethics: an ETHIC heading gets `ethics` only if it looks like a statement
      ("ETHICS STATEMENT", "ETHICAL CONSIDERATIONS", "ETHICS") or if it comes after
      a conclusion or in back matter.
    - Appendix backfill: when a body appendix region starts, the unnumbered,
      keyword-less body sections just before it (back to the last numbered or
      keyword section) also get `appendix`.
    """

    def label_paper(self, paper: Paper) -> LabelResult:
        """Set the labels of every section of `paper` in place."""
        result = self.label_details(paper.sections)
        for section, labels in zip(paper.sections, result.labels):
            section.labels = labels
        return result

    def label(self, sections: Sequence[Section]) -> list[list[str]]:
        """Return the labels of each section, in the same order."""
        return self.label_details(sections).labels

    def label_details(self, sections: Sequence[Section]) -> LabelResult:
        """Label the sections and report which ones were backfilled as appendix."""
        state = _State()
        all_labels = []
        appendix_start: int | None = None
        for index, section in enumerate(sections):
            labels = self._label_one(section, state)
            if state.in_appendix and appendix_start is None:
                appendix_start = index
            state.previous_labels = labels
            if section.location == "body":
                state.seen_body = True
                state.seen_conclusion |= "conclusion" in labels
            all_labels.append(labels)
        backfilled = self._backfill_appendix(sections, appendix_start)
        for index in backfilled:
            all_labels[index] = [APPENDIX]
        return LabelResult(
            labels=all_labels, backfilled=backfilled, appendix_start=appendix_start
        )

    def keyword_labels(self, heading: str) -> list[str]:
        """Labels whose keywords appear in the heading, in table order."""
        upper = heading.upper()
        return [
            label
            for label, keywords in KEYWORDS.items()
            if any(keyword in upper for keyword in keywords)
            or (label == "acknowledgments" and _is_misspelled_acknowledgments(upper))
        ]

    def _label_one(self, section: Section, state: _State) -> list[str]:
        keywords = self.keyword_labels(section.heading)
        if "ethics" in keywords and not self._is_ethics_statement(section, state):
            keywords.remove("ethics")
        if section.location == "back":
            return _standalone_or_appendix(keywords)

        if not state.in_appendix and self._starts_appendix(section, keywords, state):
            state.in_appendix = True
        if state.in_appendix:
            return _standalone_or_appendix(keywords)

        if not state.seen_body and not section.heading.strip():
            return ["introduction"]

        if _is_subsection(section.number):
            return keywords or self._parent_labels(section.number, state)
        if self._is_valid_top_level(section.number, state):
            labels = keywords or [BODY]
            state.last_top_level = int(section.number)
            state.top_level_labels[section.number] = labels
            return labels
        # Unnumbered section or false top-level number.
        return keywords or list(state.previous_labels)

    def _starts_appendix(
        self, section: Section, keywords: list[str], state: _State
    ) -> bool:
        if state.last_top_level is None:
            return False
        number = section.number or ""
        return (
            APPENDIX in keywords
            or bool(APPENDIX_NUMBER_PATTERN.match(number))
            or bool(APPENDIX_HEADING_PATTERN.match(section.heading))
        )

    def _is_ethics_statement(self, section: Section, state: _State) -> bool:
        """An ETHIC heading that is a statement, not a paper section about ethics."""
        upper = section.heading.upper()
        words = re.sub(r"[^A-Z ]", " ", upper).split()
        return (
            any(word in upper for word in ETHICS_STATEMENT_WORDS)
            or words == ["ETHICS"]
            or section.location == "back"
            or state.seen_conclusion
        )

    def _backfill_appendix(
        self, sections: Sequence[Section], appendix_start: int | None
    ) -> list[int]:
        """Indices of the unnumbered, keyword-less body sections right before the appendix.

        The run stops at the first section (going backwards) that has a number,
        matches a keyword or is in back matter, and is only backfilled if such a
        section exists, i.e. the run sits after a numbered or keyword section.
        """
        if appendix_start is None:
            return []
        run: list[int] = []
        for index in range(appendix_start - 1, -1, -1):
            section = sections[index]
            if (
                section.location != "body"
                or section.number is not None
                or self.keyword_labels(section.heading)
            ):
                return sorted(run)
            run.append(index)
        return []

    def _parent_labels(self, number: str, state: _State) -> list[str]:
        parent = number.split(".")[0]
        return list(state.top_level_labels.get(parent, [BODY]))

    def _is_valid_top_level(self, number: str | None, state: _State) -> bool:
        if number is None or not number.isdigit():
            return False
        return state.last_top_level is None or int(number) > state.last_top_level


def _is_misspelled_acknowledgments(upper_heading: str) -> bool:
    words = re.sub(r"[^A-Z ]", " ", upper_heading).split()
    if len(words) != 1:
        return False
    similarity = difflib.SequenceMatcher(None, words[0], ACKNOWLEDGMENTS_WORD).ratio()
    return similarity >= ACKNOWLEDGMENTS_MIN_SIMILARITY


def _is_subsection(number: str | None) -> bool:
    """True for numbers like "3.1" or "3.1.2"."""
    return number is not None and bool(re.fullmatch(r"\d+(\.\d+)+", number))


def _standalone_or_appendix(keywords: list[str]) -> list[str]:
    standalone = [label for label in keywords if label in STANDALONE_LABELS]
    return standalone or [APPENDIX]


def main() -> None:
    """Debug command: print every section of the sample papers with its labels."""
    from parsing.grobid_parser import GrobidParser, pdf_path_for, read_ids

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--ids-file", type=Path, default=Path("parsing/sample_ids.txt"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    args = parser.parse_args()

    grobid = GrobidParser()
    labeler = SectionLabeler()
    for paper_id in read_ids(args.ids_file):
        paper = grobid.parse(pdf_path_for(args.pdf_dir, paper_id))
        labeler.label_paper(paper)
        print(f"\n=== {paper.paper_id}")
        for section in paper.sections:
            number = section.number or "-"
            labels = ", ".join(section.labels)
            grobid_type = f"  [{section.grobid_type}]" if section.grobid_type else ""
            print(f"  {number:>6}  {section.location:<4}  {labels:<26}  {section.heading[:70]}"
                  f"{grobid_type}  ({len(section.text.split())} words)")


if __name__ == "__main__":
    main()
