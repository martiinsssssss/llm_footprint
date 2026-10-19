"""Data model for parsed papers (Section, Paper) and JSONL read/write."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Section:
    """One section of a paper, as found by the parser.

    `grobid_type` is the GROBID `div/@type` (e.g. "annex", "acknowledgement").
    It is only used for labeling and is not written to the JSONL output.
    """

    heading: str
    number: str | None
    location: str  # "body" or "back"
    labels: list[str] = field(default_factory=list)
    text: str = ""
    grobid_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "number": self.number,
            "location": self.location,
            "labels": list(self.labels),
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Section:
        return cls(
            heading=data["heading"],
            number=data["number"],
            location=data["location"],
            labels=list(data["labels"]),
            text=data["text"],
        )


@dataclass
class Paper:
    """One parsed paper. `full_text` and `n_words` are computed from the content."""

    paper_id: str
    year: int
    title: str = ""
    authors: str = ""
    doi: str = ""
    abstract: str = ""
    sections: list[Section] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, int] = field(default_factory=dict)
    parser: str = "grobid"
    pipeline_version: str = "v1"

    @property
    def full_text(self) -> str:
        """Abstract followed by the text of every section, separated by blank lines."""
        parts = [self.abstract] + [section.text for section in self.sections]
        return "\n\n".join(part for part in parts if part)

    @property
    def n_words(self) -> int:
        return len(self.full_text.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "year": self.year,
            "title": self.title,
            "authors": self.authors,
            "doi": self.doi,
            "abstract": self.abstract,
            "sections": [section.to_dict() for section in self.sections],
            "full_text": self.full_text,
            "n_words": self.n_words,
            "warnings": list(self.warnings),
            "artifacts": dict(self.artifacts),
            "parser": self.parser,
            "pipeline_version": self.pipeline_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Paper:
        """Build a Paper from `to_dict()` output; `full_text` and `n_words` are recomputed."""
        return cls(
            paper_id=data["paper_id"],
            year=data["year"],
            title=data["title"],
            authors=data["authors"],
            doi=data["doi"],
            abstract=data["abstract"],
            sections=[Section.from_dict(section) for section in data["sections"]],
            warnings=list(data["warnings"]),
            artifacts=dict(data["artifacts"]),
            parser=data["parser"],
            pipeline_version=data["pipeline_version"],
        )


def write_jsonl(papers: Iterable[Paper], path: Path) -> None:
    """Write papers to `path`, one JSON object per line, sorted by `paper_id`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for paper in sorted(papers, key=lambda paper: paper.paper_id):
            file.write(json.dumps(paper.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Paper]:
    """Read papers written by `write_jsonl`."""
    with path.open(encoding="utf-8") as file:
        return [Paper.from_dict(json.loads(line)) for line in file if line.strip()]
