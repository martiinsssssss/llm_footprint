"""Data model for parsed papers (Section, Paper) and JSONL read/write."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Section:
    """One section of a paper. `grobid_type` (GROBID `div/@type`) is only used for labeling, not written out."""

    heading: str
    number: str | None
    location: str  # "body" or "back"
    labels: list[str] = field(default_factory=list)
    text: str = ""
    grobid_type: str | None = None


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
        data = asdict(self)
        for section in data["sections"]:
            del section["grobid_type"]
        # Key order of the JSONL: computed fields go right after `sections`.
        tail = {key: data.pop(key) for key in ("warnings", "artifacts", "parser", "pipeline_version")}
        return {**data, "full_text": self.full_text, "n_words": self.n_words, **tail}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Paper:
        """Build a Paper from `to_dict()` output; `full_text` and `n_words` are recomputed."""
        fields = {key: value for key, value in data.items() if key not in ("full_text", "n_words")}
        fields["sections"] = [Section(**section) for section in data["sections"]]
        return cls(**fields)


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
