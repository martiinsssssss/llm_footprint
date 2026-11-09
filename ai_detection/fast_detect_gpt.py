"""Score the cleaned abstracts with Fast-DetectGPT (Bao et al., ICLR 2024).

Zero-shot detector: needs no training on AI text. For each token t, with the
scoring model's log-probabilities s_t and the sampling model's probabilities q_t:

    l_t   = s_t[x_t]                       log-likelihood of the observed token
    mu_t  = sum_v q_t(v) s_t(v)            expected log-likelihood under sampling
    var_t = sum_v q_t(v) s_t(v)^2 - mu_t^2

    criterion = (sum_t l_t - sum_t mu_t) / sqrt(sum_t var_t)

Higher = more machine-like. This is the analytic criterion of the official
implementation (github.com/baoguangsheng/fast-detect-gpt, MIT license,
get_sampling_discrepancy_analytic). The raw criterion is reported, not the
official demo's probability, which comes from reference distributions fitted
on other data.

Default models: GPT-Neo-2.7B as both sampling and scoring model (the official
local demo default; ~5.4 GB in bfloat16, runs on CPU). The README reports better
results with larger pairs (Llama-3-8B / Llama-3-8B-Instruct, falcon-7b /
falcon-7b-instruct) on a GPU: pass them with --sampling-model / --scoring-model.

Usage (from the repository root):
    python -m ai_detection.fast_detect_gpt --limit 5        # quick test
    python -m ai_detection.fast_detect_gpt                  # all abstracts
Re-running skips paper_ids already in --out, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_TOKENS = 512


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_model(name: str, dtype: torch.dtype, device: str):
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    return model


@torch.inference_mode()
def criterion(text: str, tokenizer, scoring_model, sampling_model, device: str) -> tuple[float, int]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(device)
    labels = enc["input_ids"][:, 1:]
    logits_score = scoring_model(**enc).logits[:, :-1].float()
    if sampling_model is scoring_model:
        logits_ref = logits_score
    else:
        logits_ref = sampling_model(**enc).logits[:, :-1].float()
    vocab = min(logits_score.size(-1), logits_ref.size(-1))  # models may pad the vocabulary differently
    logits_score, logits_ref = logits_score[..., :vocab], logits_ref[..., :vocab]

    log_p = logits_score.log_softmax(dim=-1)
    q = logits_ref.softmax(dim=-1)
    ll = log_p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mu = (q * log_p).sum(dim=-1)
    var = (q * log_p.square()).sum(dim=-1) - mu.square()
    crit = (ll.sum() - mu.sum()) / var.sum().sqrt()
    return crit.item(), labels.size(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="inp", type=Path, default=Path("../data/abstracts_clean_v1.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("../data/fastdetectgpt_abstracts_v1.jsonl"))
    parser.add_argument("--sampling-model", default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument("--scoring-model", default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--limit", type=int, help="only score the first N abstracts (quick test)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    rows = read_jsonl(args.inp)[: args.limit]
    done = {r["paper_id"] for r in read_jsonl(args.out)}
    todo = [r for r in rows if r["paper_id"] not in done]
    print(f"{len(rows)} abstracts, {len(done)} already scored, {len(todo)} to do, device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.scoring_model)
    if AutoTokenizer.from_pretrained(args.sampling_model).get_vocab() != tokenizer.get_vocab():
        raise SystemExit("sampling and scoring models must share a tokenizer")
    scoring_model = load_model(args.scoring_model, dtype, device)
    sampling_model = (
        scoring_model if args.sampling_model == args.scoring_model else load_model(args.sampling_model, dtype, device)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with args.out.open("a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            crit, n_tokens = criterion(row["abstract"], tokenizer, scoring_model, sampling_model, device)
            f.write(json.dumps({
                "paper_id": row["paper_id"],
                "year": row["year"],
                "fdg_criterion": round(crit, 6),
                "n_tokens": n_tokens,
                "truncated": n_tokens >= MAX_TOKENS - 1,
                "sampling_model": args.sampling_model,
                "scoring_model": args.scoring_model,
            }) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                eta = (time.time() - start) / i * (len(todo) - i)
                print(f"{i}/{len(todo)}  eta {eta / 60:.1f} min", flush=True)

    by_year: dict[int, list[float]] = {}
    for r in read_jsonl(args.out):
        by_year.setdefault(r["year"], []).append(r["fdg_criterion"])
    print(f"-> {args.out}")
    for year, s in sorted(by_year.items()):
        print(f"{year}: n={len(s)}  mean criterion={sum(s) / len(s):.3f}")


if __name__ == "__main__":
    main()
