"""Synthetic abstracts (proposal, Section 4.1).

Claude writes them through the Claude Code CLI (`claude -p`, logged in with a Claude subscription, no API key).
Every paper is a new session that sees only the prompt and that paper's text, without its abstract.
`run` needs the CLI, so it runs on the host, not in Docker.

  python -m analysis.generate sample --per-year 2 > analysis/synthetic_ids.txt   # which papers (fixed seed)
  python -m analysis.generate run --ids-file analysis/synthetic_ids.txt          # generate and save each one
  python -m analysis.generate export 2024_000001            # prompt + paper text, no abstract
  python -m analysis.generate save 2024_000001 --model M < abstract.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from parsing.models import Paper, read_jsonl

JSONL = Path("data/processed/ismir_2020_2025_v1.jsonl")
OUT_DIR = Path("data/interim/synthetic")
# The proposal's prompt. The word count is the original abstract's length, so length is not a giveaway.
PROMPT = "Write an ISMIR-style abstract of about 150 words for the attached research paper."

# Generation settings, saved in every record. Changing one changes the dataset.
MODEL = "claude-sonnet-5-5"
EFFORT = "high"  # between_tools needs high or lower; Claude Code's default is higher
# Claude Code has no flag to turn thinking off on Sonnet 5.5, so the request body is extended. Its
# thinking-clearing context edit is rejected without thinking, so it is removed as well.
EXTRA_BODY = {"thinking": {"type": "between_tools"}, "context_management": {"edits": []}}
# Claude Code always adds a short system prompt that names the working directory: run from an empty, neutral one.
WORKDIR = Path("/tmp/abstract")
SEED = 0
# The model puts headings before the text: a "**Abstract**" line, a bold title, or an inline "**Abstract.**".
# They are removed from `abstract`; the record keeps the raw output too.
HEADING = re.compile(r"(?:#+ [^\n]*|\*\*[^\n]*\*\*)[ \t]*\n")
LABEL = re.compile(r"^(?:#+\s*)?\**\s*abstract\b\s*[.:]?\s*\**\s*[.:]?\s*", re.IGNORECASE)


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


def message_for(paper: Paper, words: int | None = None) -> str:
    """The whole user message: the prompt, then the attached paper."""
    return f"{prompt_for(paper, words)}\n\n---\n\n{paper_text(paper)}"


def sample(papers: list[Paper], per_year: int, seed: int = SEED) -> list[str]:
    """The first `per_year` papers of each year in a fixed shuffled order, so a larger `per_year` extends a smaller one."""
    by_year: dict[int, list[str]] = defaultdict(list)
    for paper in papers:
        by_year[paper.year].append(paper.paper_id)
    ids = []
    for year in sorted(by_year):
        order = sorted(by_year[year])
        random.Random(f"{seed}-{year}").shuffle(order)
        ids += order[:per_year]
    return ids


def strip_label(text: str) -> str:
    text = text.strip()
    while heading := HEADING.match(text):
        text = text[heading.end():].lstrip()
    return LABEL.sub("", text, count=1)


def claude_command(claude: str) -> list[str]:
    """`claude -p` with no tools, no system prompt of ours, and no user or project settings."""
    return [claude, "-p", "--model", MODEL, "--effort", EFFORT, "--system-prompt", "", "--tools", "",
            "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
            "--no-session-persistence", "--output-format", "json"]


def generate(paper: Paper, claude: str, words: int | None = None) -> tuple[str, dict]:
    """One new Claude Code session for one paper. Returns the abstract and the generation metadata."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CLAUDE_CODE_EXTRA_BODY": json.dumps(EXTRA_BODY)}
    done = subprocess.run(claude_command(claude), input=message_for(paper, words), capture_output=True,
                          text=True, cwd=WORKDIR, env=env, timeout=600)
    try:
        result = json.loads(done.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(done.stderr.strip() or done.stdout.strip() or f"exit code {done.returncode}")
    if result.get("is_error"):
        raise RuntimeError(result.get("result"))
    if set(result.get("modelUsage", {})) != {MODEL}:
        raise RuntimeError(f"answered by {sorted(result.get('modelUsage', {}))}, not {MODEL}")
    usage = result["usage"]
    thinking = usage.get("output_tokens_details", {}).get("thinking_tokens", 0)
    if thinking:
        raise RuntimeError(f"{thinking} thinking tokens: thinking was not turned off")
    meta = {"effort": EFFORT, "request_overrides": EXTRA_BODY,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input_tokens": usage["input_tokens"] + usage["cache_creation_input_tokens"]
            + usage["cache_read_input_tokens"],
            "output_tokens": usage["output_tokens"], "duration_ms": result.get("duration_ms")}
    return result["result"], meta


def save(paper: Paper, abstract: str, model: str, words: int | None = None,
         out_dir: Path = OUT_DIR, force: bool = False, meta: dict | None = None) -> Path:
    abstract = abstract.strip()
    if not abstract:
        raise SystemExit("Empty abstract: nothing saved.")
    path = out_dir / f"{paper.paper_id}.json"
    if path.exists() and not force:
        raise SystemExit(f"{path} exists. Use --force to replace it.")
    record = {"paper_id": paper.paper_id, "year": paper.year, "model": model,
              "prompt": prompt_for(paper, words), "abstract": abstract, **(meta or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def run(ids: list[str], claude: str, words: int | None = None, workers: int = 4, force: bool = False) -> None:
    """Generate and save every paper in `ids`, `workers` sessions at a time. Prints one line per paper."""
    papers = {paper.paper_id: paper for paper in read_jsonl(JSONL)}
    if unknown := [pid for pid in ids if pid not in papers]:
        raise SystemExit(f"Unknown paper_id(s) in {JSONL}: {', '.join(unknown)}")
    version = subprocess.run([claude, "--version"], capture_output=True, text=True, check=True).stdout.split()[0]

    def one(paper_id: str) -> str:
        if (OUT_DIR / f"{paper_id}.json").exists() and not force:
            return "skipped (exists)"
        try:
            abstract, meta = generate(papers[paper_id], claude, words)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            return f"failed ({error})"
        meta = {"generator": "claude-code", "claude_code_version": version, **meta, "raw_output": abstract}
        return f"saved {save(papers[paper_id], strip_label(abstract), MODEL, words, force=force, meta=meta)}"

    with ThreadPoolExecutor(workers) as pool:
        for paper_id, outcome in zip(ids, pool.map(one, ids)):
            print(f"{paper_id}: {outcome}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["export", "save", "sample", "run"])
    parser.add_argument("paper_ids", nargs="*", help="one for export/save; any number for run")
    parser.add_argument("--ids-file", type=Path, help="run: file with one paper_id per line")
    parser.add_argument("--per-year", type=int, default=2, help="sample: papers per year")
    parser.add_argument("--words", type=int, help="target length (default: the original abstract's length)")
    parser.add_argument("--model", help="model that wrote the abstract (save only)")
    parser.add_argument("--claude", default=os.environ.get("CLAUDE_BIN", "claude"),
                        help="Claude Code CLI (run only; default: $CLAUDE_BIN or claude)")
    parser.add_argument("--workers", type=int, default=4, help="run: parallel sessions")
    parser.add_argument("--force", action="store_true", help="replace an existing synthetic abstract")
    args = parser.parse_args()

    if args.action == "sample":
        print("\n".join(sample(read_jsonl(JSONL), args.per_year)))
        return
    if args.action == "run":
        ids = args.paper_ids + (args.ids_file.read_text().split() if args.ids_file else [])
        run(ids, args.claude, args.words, args.workers, args.force)
        return
    if len(args.paper_ids) != 1:
        raise SystemExit(f"{args.action} needs exactly one paper_id")
    paper = find_paper(args.paper_ids[0])
    if args.action == "export":
        print(message_for(paper, args.words))
    else:
        if not args.model:
            raise SystemExit("save needs --model")
        print(save(paper, sys.stdin.read(), args.model, args.words, force=args.force))


if __name__ == "__main__":
    main()
