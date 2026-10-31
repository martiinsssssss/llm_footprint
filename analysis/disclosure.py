"""Find possible AI-use disclosures (RQ1): every sentence that mentions an AI tool.

Each hit still needs a manual check, because many ISMIR papers study LLMs
or only mention them. Fill the empty `label` column of the output by hand.

Usage: python -m analysis.disclosure data/processed/ismir_2020_2025_v1.jsonl --out data/processed/disclosure_hits_v1.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from parsing.models import Paper, read_jsonl

# Term name -> pattern. Product names are case-sensitive, so "bard" or "gemini" in prose don't match.
TERMS = {
    "chatgpt": r"\bChat-?GPT\b",
    "gpt": r"\bGPT-?\d",
    "claude": r"\bClaude\b(?! (?:Debussy|Shannon|Latour))",  # people in MIR papers (Latour: a funder)
    "gemini": r"\bGemini\b",
    "bard": r"\bBard\b",
    "copilot": r"\bCopilot\b",
    "grammarly": r"\bGrammarly\b",
    "deepl": r"\bDeepL\b",
    "quillbot": r"\bQuillBot\b",
    "llm": r"(?i:\blarge language models?\b)|\bLLMs?\b",
    "generative_ai": r"(?i:\bgenerative AI\b)",
    # AI near a writing verb, in one sentence: "used an AI tool to polish the text".
    "ai_writing": r"\b(?:AI|(?i:artificial intelligence))\b[^.]{0,60}"
                  r"(?i:\b(?:writ|edit|proofread|polish|grammar|rephras|paraphras))",
}
PATTERNS = {name: re.compile(pattern) for name, pattern in TERMS.items()}
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
COLUMNS = ["paper_id", "year", "section", "section_labels", "terms", "sentence", "label"]


def find_hits(paper: Paper) -> list[dict[str, object]]:
    """One row per sentence (in the abstract or any section) that matches at least one term."""
    parts = [("abstract", "abstract", paper.abstract)]
    parts += [(s.heading, "|".join(s.labels), s.text) for s in paper.sections]
    rows = []
    for heading, labels, text in parts:
        for sentence in SENTENCE_END.split(text):
            terms = [name for name, pattern in PATTERNS.items() if pattern.search(sentence)]
            if terms:
                rows.append({
                    "paper_id": paper.paper_id, "year": paper.year, "section": heading,
                    "section_labels": labels, "terms": "|".join(terms),
                    "sentence": sentence.strip(), "label": "",
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Write every sentence that mentions an AI tool to a CSV.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="overwrite the output (deletes manual labels)")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists and may hold manual labels. Use --force to overwrite it.")

    papers = read_jsonl(args.jsonl)
    hits = pd.DataFrame([row for paper in papers for row in find_hits(paper)], columns=COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hits.to_csv(args.out, index=False)
    per_year = hits.groupby("year")["paper_id"].nunique().reindex(sorted({p.year for p in papers}), fill_value=0)
    print(f"{len(hits)} sentences in {hits['paper_id'].nunique()} papers -> {args.out}")
    print("Papers with a hit per year: " + ", ".join(f"{y}: {n}" for y, n in per_year.items()))


if __name__ == "__main__":
    main()
