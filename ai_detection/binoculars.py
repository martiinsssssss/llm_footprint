"""Score the cleaned abstracts with Binoculars (Hans et al., ICML 2024).

Zero-shot detector built from two related LLMs, an observer and a performer:

    log PPL   = mean_t  -log performer(x_t | x_<t)                    (perplexity of the text)
    log X-PPL = mean_t  -sum_v observer_t(v) log performer_t(v)       (cross-perplexity)

    score = log PPL / log X-PPL

LOWER = more machine-like. This reproduces the official implementation
(github.com/ahans30/Binoculars, BSD-3-Clause): binoculars/detector.py and
binoculars/metrics.py, including its exact token masking. The official thresholds
apply only to the default Falcon pair in bfloat16:
    low-fpr  0.8536  (default mode, ~0.01% false-positive rate)
    accuracy 0.9015  (tuned for F1)
Scores below a threshold mean "most likely AI-generated".

Hardware: the default pair (falcon-7b + falcon-7b-instruct) needs ~28 GB in
bfloat16, i.e. an A100 40GB/L4 24GB+ or 2 GPUs. On a free Colab T4 (15 GB) use
--quantize 4bit (needs bitsandbytes); quantization shifts the scores slightly, so
the official thresholds are then only approximate, but year comparisons stay valid
as long as every abstract runs with the same setting.

Colab:
    !git clone <this repo> && cd llm_footprint
    !pip install -q -r ai_detection/requirements.txt bitsandbytes
    # upload abstracts_clean_v1.jsonl, then:
    !python -m ai_detection.binoculars --in abstracts_clean_v1.jsonl --out binoculars_abstracts_v1.jsonl --limit 5
    !python -m ai_detection.binoculars --in abstracts_clean_v1.jsonl --out binoculars_abstracts_v1.jsonl
Re-running skips paper_ids already in --out, so an interrupted run resumes
(useful when Colab disconnects: write --out to Google Drive).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OBSERVER = "tiiuae/falcon-7b"
PERFORMER = "tiiuae/falcon-7b-instruct"
THRESHOLD_LOW_FPR = 0.8536432310785527
THRESHOLD_ACCURACY = 0.9015310749276843
MAX_TOKENS = 512


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def use_plain_linear_in_falcon() -> None:
    """Let bitsandbytes quantize Falcon.

    Falcon builds its layers with FalconLinear, a subclass of nn.Linear with the
    same math (x @ W.T + b). transformers' bitsandbytes integration only replaces
    modules whose type is exactly nn.Linear, so it silently skips every Falcon
    layer and the model loads unquantized (~13 GB in float16, which does not fit
    twice on a 15 GB T4). Building plain nn.Linear layers instead fixes that.
    """
    from transformers.models.falcon import modeling_falcon

    modeling_falcon.FalconLinear = torch.nn.Linear


def load_model(name: str, dtype: torch.dtype, quantize: str, device: str):
    kwargs: dict = {"dtype": dtype}
    if quantize != "none":
        from transformers import BitsAndBytesConfig

        use_plain_linear_in_falcon()

        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            if quantize == "4bit"
            else BitsAndBytesConfig(load_in_8bit=True)
        )
        kwargs["device_map"] = {"": device}
        return AutoModelForCausalLM.from_pretrained(name, **kwargs).eval()
    return AutoModelForCausalLM.from_pretrained(name, **kwargs).to(device).eval()


def log_ppl(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean next-token NLL per text (metrics.perplexity in the official code)."""
    nll = F.cross_entropy(logits[:, :-1].transpose(1, 2), input_ids[:, 1:], reduction="none")
    mask = attention_mask[:, 1:].float()
    return (nll * mask).sum(1) / mask.sum(1)


def log_x_ppl(p_logits: torch.Tensor, q_logits: torch.Tensor, input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Mean cross-entropy H(p, q) per position (metrics.entropy in the official code).

    As in the official code: no shift (every position, including the last, counts)
    and padding is masked by comparing input_ids with the pad token.
    """
    ce = -(p_logits.softmax(-1) * q_logits.log_softmax(-1)).sum(-1)
    mask = (input_ids != pad_id).float()
    return (ce * mask).sum(1) / mask.sum(1)


@torch.inference_mode()
def score_batch(texts: list[str], tokenizer, observer, performer) -> list[dict]:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="longest" if len(texts) > 1 else False,
        truncation=True,
        max_length=MAX_TOKENS,
        return_token_type_ids=False,
    )
    obs_logits = observer(**enc.to(observer.device)).logits
    perf_logits = performer(**enc.to(performer.device)).logits.to(obs_logits.device)
    vocab = min(obs_logits.size(-1), perf_logits.size(-1))  # models may pad the vocabulary differently
    obs_logits, perf_logits = obs_logits[..., :vocab].float(), perf_logits[..., :vocab].float()
    enc = enc.to(obs_logits.device)

    # Official compute_score: ppl from the performer, x-ppl = H(observer, performer).
    ppl = log_ppl(perf_logits, enc["input_ids"], enc["attention_mask"])
    x_ppl = log_x_ppl(obs_logits, perf_logits, enc["input_ids"], tokenizer.pad_token_id)
    n_tokens = enc["attention_mask"].sum(1)
    return [
        {"binoculars_score": (p / x).item(), "log_ppl": p.item(), "log_x_ppl": x.item(), "n_tokens": int(n)}
        for p, x, n in zip(ppl, x_ppl, n_tokens)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="inp", type=Path, default=Path("../data/abstracts_clean_v1.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("../data/binoculars_abstracts_v1.jsonl"))
    parser.add_argument("--observer", default=OBSERVER)
    parser.add_argument("--performer", default=PERFORMER)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--quantize", choices=["none", "8bit", "4bit"], default="none", help="needs a CUDA GPU")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, help="only score the first N abstracts (quick test)")
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    if args.quantize != "none" and not n_gpus:
        raise SystemExit("--quantize needs a CUDA GPU (bitsandbytes)")
    obs_device = "cuda:0" if n_gpus else "cpu"
    perf_device = "cuda:1" if n_gpus > 1 else obs_device  # as in the official code
    dtype = getattr(torch, args.dtype)
    official = (args.observer, args.performer, args.dtype, args.quantize) == (OBSERVER, PERFORMER, "bfloat16", "none")

    rows = read_jsonl(args.inp)[: args.limit]
    done = {r["paper_id"] for r in read_jsonl(args.out)}
    todo = [r for r in rows if r["paper_id"] not in done]
    print(f"{len(rows)} abstracts, {len(done)} already scored, {len(todo)} to do, "
          f"observer on {obs_device}, performer on {perf_device}")

    tokenizer = AutoTokenizer.from_pretrained(args.observer)
    if AutoTokenizer.from_pretrained(args.performer).get_vocab() != tokenizer.get_vocab():
        raise SystemExit("observer and performer must share a tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    observer = load_model(args.observer, dtype, args.quantize, obs_device)
    performer = load_model(args.performer, dtype, args.quantize, perf_device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with args.out.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i : i + args.batch_size]
            for row, res in zip(batch, score_batch([r["abstract"] for r in batch], tokenizer, observer, performer)):
                s = res["binoculars_score"]
                f.write(json.dumps({
                    "paper_id": row["paper_id"],
                    "year": row["year"],
                    "binoculars_score": round(s, 6),
                    "log_ppl": round(res["log_ppl"], 6),
                    "log_x_ppl": round(res["log_x_ppl"], 6),
                    "n_tokens": res["n_tokens"],
                    "truncated": res["n_tokens"] >= MAX_TOKENS,
                    # official thresholds are only valid for the default, unquantized pair
                    "ai_low_fpr": s < THRESHOLD_LOW_FPR if official else None,
                    "ai_accuracy": s < THRESHOLD_ACCURACY if official else None,
                    "observer": args.observer,
                    "performer": args.performer,
                    "dtype": args.dtype,
                    "quantize": args.quantize,
                }) + "\n")
            f.flush()
            n = min(i + args.batch_size, len(todo))
            print(f"{n}/{len(todo)}  eta {(time.time() - start) / n * (len(todo) - n) / 60:.1f} min", flush=True)

    by_year: dict[int, list[float]] = {}
    for r in read_jsonl(args.out):
        by_year.setdefault(r["year"], []).append(r["binoculars_score"])
    print(f"-> {args.out}")
    for year, s in sorted(by_year.items()):
        print(f"{year}: n={len(s)}  mean score={sum(s) / len(s):.4f}")


if __name__ == "__main__":
    main()
