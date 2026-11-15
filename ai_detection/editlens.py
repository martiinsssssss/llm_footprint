"""Score the cleaned abstracts with EditLens (Pangram Labs, ICLR 2026).

EditLens predicts how much of a text was written or edited by AI. The model
classifies a text into ordered buckets (0 = human ... n-1 = fully AI); the score
is the expected bucket divided by n-1, so it lies in [0, 1]. This matches
`score_pred` in the official scripts/inference.py.

Model: https://huggingface.co/pangram/editlens_roberta-large (gated, CC BY-NC-SA 4.0).
Accept the terms on that page, then log in once with `hf auth login`
(or set HF_TOKEN).

Usage (from the repository root):
    python -m ai_detection.editlens \
        --in ../data/abstracts_clean_v1.jsonl \
        --out ../data/editlens_abstracts_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "pangram/editlens_roberta-large"
TOKENIZER = "FacebookAI/roberta-large"  # base model of the checkpoint
MAX_LENGTH = 512  # same as the official inference example


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.inference_mode()
def score(texts: list[str], model, tokenizer, batch_size: int, device: str) -> list[dict]:
    n_buckets = model.config.num_labels
    buckets = torch.arange(n_buckets, dtype=torch.float32, device=device)
    results = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(device)
        probs = model(**enc).logits.float().softmax(dim=-1)
        scores = (probs * buckets).sum(dim=-1) / (n_buckets - 1)
        n_tokens = enc["attention_mask"].sum(dim=-1)
        for p, s, n in zip(probs.tolist(), scores.tolist(), n_tokens.tolist()):
            results.append({
                "editlens_score": round(s, 6),
                "editlens_bucket": max(range(n_buckets), key=p.__getitem__),
                "editlens_probs": [round(x, 6) for x in p],
                "n_tokens": n,
                "truncated": n >= MAX_LENGTH,
            })
        print(f"{min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="inp", type=Path, default=Path("../data/abstracts_clean_v1.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("../data/editlens_abstracts_v1.jsonl"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only score the first N abstracts (quick test)")
    args = parser.parse_args()

    rows = read_jsonl(args.inp)[: args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    print(f"{MODEL}: {model.config.num_labels} buckets, device={device}, {len(rows)} abstracts")

    results = score([r["abstract"] for r in rows], model, tokenizer, args.batch_size, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row, res in zip(rows, results):
            out = {k: row[k] for k in ("paper_id", "year", "title")} | res
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    by_year: dict[int, list[float]] = {}
    for row, res in zip(rows, results):
        by_year.setdefault(row["year"], []).append(res["editlens_score"])
    print(f"-> {args.out}  ({sum(r['truncated'] for r in results)} truncated)")
    for year, s in sorted(by_year.items()):
        print(f"{year}: n={len(s)}  mean score={sum(s) / len(s):.3f}")


if __name__ == "__main__":
    main()
