"""Text normalization shared by all parsers (TextCleaner)."""

from __future__ import annotations

import re
import unicodedata

from parsing.models import Paper

REPLACEMENT_CHARACTER = "�"

# "clas- sifies" -> "classifies": a hyphen followed by a space and a lowercase
# letter. Real compounds ("state-of-the-art") have no space after the hyphen.
LINE_BREAK_HYPHEN = re.compile(r"(?<=[A-Za-z])- +(?=[a-z])")

# Numeric citation brackets: [12], [3, 4], [5-7], [5–7], [1, 3-5].
CITATION_BRACKET = re.compile(r"\[\s*\d+(?:\s*[,;\-–]\s*\d+)*\s*\]")

# A URL, plus continuation pieces when GROBID split it after a "." or "/"
# ("https://github. com/user/repo"). A continuation must contain a "/", so a
# normal word after a sentence-final URL is not swallowed.
URL = re.compile(r"(?:https?://|www\.)\S+(?:(?<=[./])\s+[a-z0-9][\w.\-~%]*/\S*)*")
URL_TRAILING_PUNCTUATION = ".,;:)]"
EMAIL = re.compile(r"\b[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+\b")

# Brackets left empty (or with only separators) after removing their content.
EMPTY_BRACKETS = re.compile(r"\([\s,;]*\)|\[[\s,;]*\]")
# A "." followed by a digit is a decimal (".154"), not punctuation.
SPACE_BEFORE_PUNCTUATION = re.compile(r" +(?=[,;:!?)\]]|\.(?!\d))")
# Separators doubled by removed citations: "a [1], [2], b" -> "a,, b" -> "a, b".
DOUBLED_SEPARATOR = re.compile(r"[,;]\s*(?=[,;)\]])")
SEPARATOR_AFTER_OPENING = re.compile(r"(?<=[(\[])\s*[,;]\s*")

# Letter-by-letter noise: invisible text in some PDFs (e.g. LaTeX image data)
# that GROBID extracts as "A A A C 1 n i c j V H L ...". Runs this long never
# occur in prose; short sequences such as "x y z" are kept.
MIN_CHAR_RUN = 30
# Applied after whitespace normalization, so tokens are separated by single
# spaces and a run never spans a paragraph break.
CHAR_RUN = re.compile(rf"(?<!\S)(?:\S ){{{MIN_CHAR_RUN - 1},}}\S(?!\S)")


class TextCleaner:
    """Normalizes text identically for every paper, whatever its year or source.

    A pure function of the text: no parser- or year-specific logic.
    Paragraph breaks ("\\n\\n") are kept; all other whitespace becomes one space.
    """

    def clean(self, text: str) -> str:
        return self.clean_with_stats(text)[0]

    def clean_with_stats(self, text: str) -> tuple[str, int]:
        """Clean the text; also return how many tokens the char-run step removed."""
        text = unicodedata.normalize("NFKC", text)  # ligatures -> letters ("ﬁ" -> "fi")
        text = self._remove_bad_characters(text)
        text = LINE_BREAK_HYPHEN.sub("", text)
        text = CITATION_BRACKET.sub("", text)
        text = EMAIL.sub("", URL.sub(_keep_trailing_punctuation, text))
        text = self._remove_empty_brackets_and_stray_spaces(text)
        return self._remove_char_runs(self._normalize_whitespace(text))

    def clean_paper(self, paper: Paper) -> None:
        """Clean the abstract and every section text of `paper` in place.

        Records the tokens removed as letter-by-letter noise in
        `paper.artifacts["char_run_tokens_removed"]`.
        """
        paper.abstract, removed = self.clean_with_stats(paper.abstract)
        for section in paper.sections:
            section.text, section_removed = self.clean_with_stats(section.text)
            removed += section_removed
        paper.artifacts["char_run_tokens_removed"] = removed

    def _remove_bad_characters(self, text: str) -> str:
        """Drop NUL, control characters (except newlines and tabs) and "�"."""
        return "".join(
            char
            for char in text
            if char in "\n\t"
            or (char != REPLACEMENT_CHARACTER and unicodedata.category(char) != "Cc")
        )

    def _remove_empty_brackets_and_stray_spaces(self, text: str) -> str:
        """Fix leftovers of removed citations/URLs: "music [1], as" -> "music , as" -> "music, as"."""
        text = EMPTY_BRACKETS.sub("", text)
        text = SPACE_BEFORE_PUNCTUATION.sub("", text)
        text = DOUBLED_SEPARATOR.sub("", text)
        return SEPARATOR_AFTER_OPENING.sub("", text)

    def _remove_char_runs(self, text: str) -> tuple[str, int]:
        """Remove runs of MIN_CHAR_RUN or more single-character tokens; count their tokens."""
        removed = sum(len(match.group(0).split()) for match in CHAR_RUN.finditer(text))
        if removed == 0:
            return text, 0
        return self._normalize_whitespace(CHAR_RUN.sub("", text)), removed

    def _normalize_whitespace(self, text: str) -> str:
        paragraphs = re.split(r"\n\s*\n", text)
        collapsed = (" ".join(paragraph.split()) for paragraph in paragraphs)
        return "\n\n".join(paragraph for paragraph in collapsed if paragraph)


def _keep_trailing_punctuation(match: re.Match[str]) -> str:
    """Remove a URL but keep the punctuation that ends its sentence."""
    url = match.group(0)
    return url[len(url.rstrip(URL_TRAILING_PUNCTUATION)):]
