"""Clean the ISMIR abstracts before running AI-text detectors on them.

Removes GROBID leftovers that are not the authors' prose: footnote markers and
footnote text, broken URL fragments, and paragraph breaks. Drops papers whose
abstract field is not the real abstract.

Usage (from the repository root):
    python -m ai_detection.cleaning \
        --in ../data/ismir_2020_2025_v1.jsonl \
        --out ../data/abstracts_clean_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Abstract field holds something else: a figure caption (2023_000014) or the
# abstract plus part of the introduction (2024_000108).
DROP = {"2023_000014", "2024_000108"}

# Footnotes that GROBID puts into the abstract (affiliation / internship notes).
FOOTNOTE = re.compile(
    r"(\d+\s+)?[*†]?\s*(Equal contribution[.;]?( corresponding authors?\.)?"
    r"|(This )?[Ww]ork (was )?(done|completed) (while|during|as)\b[^.]*\."
    r"|Considered synonymous to[^.]*\."
    r"|[A-Z][\w ]+ was an intern at [^.]*\."
    r"|\s*The first two authors (have )?equal contribution\.)"
)
URL = re.compile(r"https?:\s*//\s*\S+|\S*github\.(?:com|io)\S*")
# Footnote marker (digits, *, †) right after a sentence end. In the second half of the
# abstract it starts the footnotes GROBID appends ("...for MIR. 1 MIR has a broad
# definition..."), so the rest is cut. In the first half it is a marker on a word
# early in the abstract and the text after it is still the abstract, so only the
# marker is removed.
FOOTNOTE_BLOCK = re.compile(r"(?<=[.!?:])\s+(?:(?:\d{1,2}|\*+|†)\s+)+(?=[A-Z])")
MARKERS_BEFORE_PUNCT = re.compile(r"(\s+[\d*†]){1,3}(?=[.,;:]?$|[.,;:]\s)")  # "online 1." / "1 2." / " * *"
TRAILING_MARKERS = re.compile(r"(\s+[\d*†]){1,3}\s*$")  # " 1", " 1 2" after the last sentence


def clean_abstract(text: str) -> str:
    text = " ".join(text.split("\n\n"))
    text = FOOTNOTE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    half = len(text) / 2
    for match in FOOTNOTE_BLOCK.finditer(text):
        if match.start() > half:
            text = text[: match.start()]
            break
    text = FOOTNOTE_BLOCK.sub(" ", text)  # markers left in the first half
    text = URL.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = MARKERS_BEFORE_PUNCT.sub("", text)
    text = TRAILING_MARKERS.sub("", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)  # space left before punctuation
    return text


def load_abstracts(path: Path) -> list[dict]:
    """Return one {paper_id, year, title, abstract} dict per kept paper."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            if paper["paper_id"] in DROP:
                continue
            rows.append({
                "paper_id": paper["paper_id"],
                "year": paper["year"],
                "title": paper["title"],
                "abstract": clean_abstract(paper["abstract"]),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="inp", type=Path, default=Path("../data/ismir_2020_2025_v1.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("../data/abstracts_clean_v1.jsonl"))
    args = parser.parse_args()

    rows = load_abstracts(args.inp)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} abstracts -> {args.out} ({len(DROP)} dropped)")


if __name__ == "__main__":
    main()
