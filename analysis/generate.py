"""Inputs and outputs for synthetic abstracts (proposal, Section 4.1).

The LLM itself is the `synthetic-abstract` Claude agent (.claude/agents/synthetic-abstract.md).
This module only gives it the paper without its abstract and saves its answer.

  python -m analysis.generate export 2024_000001            # prompt + paper text, no abstract
  python -m analysis.generate save 2024_000001 --model M < abstract.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from parsing.models import Paper, read_jsonl

JSONL = Path("data/processed/ismir_2020_2025_v1.jsonl")
OUT_DIR = Path("data/interim/synthetic")
# The proposal's prompt. The word count is the original abstract's length, so length is not a giveaway.
PROMPT = "Write an ISMIR-style abstract of about {words} words for the attached research paper."


def find_paper(paper_id: str, jsonl: Path = JSONL) -> Paper:
    for paper in read_jsonl(jsonl):
        if paper.paper_id == paper_id:
            return paper
    raise SystemExit(f"Unknown paper_id {paper_id} in {jsonl}")


def prompt_for(paper: Paper, words: int | None = None) -> str:
    return PROMPT.format(words=words or len(paper.abstract.split()))


def paper_text(paper: Paper) -> str:
    """Title and every section, without the abstract."""
    parts = [paper.title] + [f"{s.heading}\n{s.text}" for s in paper.sections]
    return "\n\n".join(part for part in parts if part.strip())


def save(paper: Paper, abstract: str, model: str, words: int | None = None,
         out_dir: Path = OUT_DIR, force: bool = False) -> Path:
    abstract = abstract.strip()
    if not abstract:
        raise SystemExit("Empty abstract: nothing saved.")
    path = out_dir / f"{paper.paper_id}.json"
    if path.exists() and not force:
        raise SystemExit(f"{path} exists. Use --force to replace it.")
    record = {"paper_id": paper.paper_id, "year": paper.year, "model": model,
              "prompt": prompt_for(paper, words), "abstract": abstract}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["export", "save"])
    parser.add_argument("paper_id")
    parser.add_argument("--words", type=int, help="target length (default: the original abstract's length)")
    parser.add_argument("--model", help="model that wrote the abstract (save only)")
    parser.add_argument("--force", action="store_true", help="replace an existing synthetic abstract")
    args = parser.parse_args()

    paper = find_paper(args.paper_id)
    if args.action == "export":
        print(f"{prompt_for(paper, args.words)}\n\n---\n\n{paper_text(paper)}")
    else:
        if not args.model:
            raise SystemExit("save needs --model")
        print(save(paper, sys.stdin.read(), args.model, args.words, force=args.force))


if __name__ == "__main__":
    main()
