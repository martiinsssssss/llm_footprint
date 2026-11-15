"""Convert the synthetic (LLM-written) abstracts into the same JSONL format as
abstracts_clean_v1.jsonl, so every detector script runs on them unchanged.

Input: the zip of synthetic/<paper_id>.json files, one per ISMIR paper, each
written by an LLM from the full paper ("Write an ISMIR-style abstract of about N
words ..."). The `abstract` field is used (the generator already removed the
"**Abstract**" header that is in `raw_output`). This script also removes:
  - word-count notes the model appended: "(≈139 words)", "*(Word count: 98)*"
  - drafts before a revised version: when the output contains several versions
    (e.g. 2020_000080: a short draft, then "here is a version expanded to
    match: **Abstract** ..."), only the last one is kept
  - markdown emphasis and paragraph breaks
and drops the same papers as cleaning.py, so real and synthetic stay paired.

Usage (from the repository root):
    python -m ai_detection.synthetic \
        --zip ../data/synthetic-20261103T185220Z-1-001.zip \
        --out ../data/synthetic_abstracts_v1.jsonl
Then run any detector with --in ../data/synthetic_abstracts_v1.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

from ai_detection.cleaning import DROP

WORD_COUNT_NOTE = re.compile(r"^\W*(?:\(?\s*(?:word count:?\s*)?[~≈]?\s*\d+\s*(?:words)?\s*\)?)\W*$", re.I)
VERSION_HEADER = re.compile(r"^\s*\*\*\s*Abstract\s*\*\*\s*$", re.I | re.M)


def clean_synthetic(text: str) -> str:
    headers = list(VERSION_HEADER.finditer(text))
    if headers:  # several versions: keep the text after the last "**Abstract**" header
        text = text[headers[-1].end():]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text)]
    paragraphs = [p for p in paragraphs if p and p != "---" and not WORD_COUNT_NOTE.match(p)]
    text = " ".join(paragraphs)
    text = re.sub(r"(?<!\*)\*\*(?!\s*(?:kern|harm)\b)([^*]+?)\*\*", r"\1", text)  # **bold**, but keep Humdrum **kern
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, default=Path("../data/synthetic-20261103T185220Z-1-001.zip"))
    parser.add_argument("--titles", type=Path, default=Path("../data/ismir_2020_2025_v1.jsonl"),
                        help="dataset JSONL to take the paper titles from")
    parser.add_argument("--out", type=Path, default=Path("../data/synthetic_abstracts_v1.jsonl"))
    args = parser.parse_args()

    titles = {}
    with args.titles.open(encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            titles[paper["paper_id"]] = paper["title"]

    rows = []
    with zipfile.ZipFile(args.zip) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".json")):
            item = json.loads(z.read(name))
            if item["paper_id"] in DROP:
                continue
            rows.append({
                "paper_id": item["paper_id"],
                "year": item["year"],
                "title": titles.get(item["paper_id"], ""),
                "abstract": clean_synthetic(item["abstract"]),
                "generator_model": item["model"],
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} synthetic abstracts -> {args.out} ({len(DROP)} dropped)")


if __name__ == "__main__":
    main()
