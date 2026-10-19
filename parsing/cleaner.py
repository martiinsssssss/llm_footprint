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


class TextCleaner:
    """Normalizes text identically for every paper, whatever its year or source.

    A pure function of the text: no parser- or year-specific logic.
    Paragraph breaks ("\\n\\n") are kept; all other whitespace becomes one space.
    """

    def clean(self, text: str) -> str:
        text = self._normalize_unicode(text)
        text = self._remove_bad_characters(text)
        text = self._join_line_break_hyphens(text)
        text = self._remove_citation_brackets(text)
        text = self._remove_urls_and_emails(text)
        text = self._remove_empty_brackets_and_stray_spaces(text)
        return self._normalize_whitespace(text)

    def clean_paper(self, paper: Paper) -> None:
        """Clean the abstract and every section text of `paper` in place."""
        paper.abstract = self.clean(paper.abstract)
        for section in paper.sections:
            section.text = self.clean(section.text)

    def _normalize_unicode(self, text: str) -> str:
        """NFKC turns ligatures into letters ("ﬁ" -> "fi") and unifies look-alikes."""
        return unicodedata.normalize("NFKC", text)

    def _remove_bad_characters(self, text: str) -> str:
        """Drop NUL, control characters (except newlines and tabs) and "�"."""
        return "".join(
            char
            for char in text
            if char in "\n\t"
            or (char != REPLACEMENT_CHARACTER and unicodedata.category(char) != "Cc")
        )

    def _join_line_break_hyphens(self, text: str) -> str:
        return LINE_BREAK_HYPHEN.sub("", text)

    def _remove_citation_brackets(self, text: str) -> str:
        return CITATION_BRACKET.sub("", text)

    def _remove_urls_and_emails(self, text: str) -> str:
        text = URL.sub(_keep_trailing_punctuation, text)
        return EMAIL.sub("", text)

    def _remove_empty_brackets_and_stray_spaces(self, text: str) -> str:
        """Fix leftovers of removed citations/URLs: "music [1], as" -> "music , as" -> "music, as"."""
        text = EMPTY_BRACKETS.sub("", text)
        text = SPACE_BEFORE_PUNCTUATION.sub("", text)
        text = DOUBLED_SEPARATOR.sub("", text)
        return SEPARATOR_AFTER_OPENING.sub("", text)

    def _normalize_whitespace(self, text: str) -> str:
        paragraphs = re.split(r"\n\s*\n", text)
        collapsed = (" ".join(paragraph.split()) for paragraph in paragraphs)
        return "\n\n".join(paragraph for paragraph in collapsed if paragraph)


def _keep_trailing_punctuation(match: re.Match[str]) -> str:
    """Remove a URL but keep the punctuation that ends its sentence."""
    url = match.group(0)
    return url[len(url.rstrip(URL_TRAILING_PUNCTUATION)):]
