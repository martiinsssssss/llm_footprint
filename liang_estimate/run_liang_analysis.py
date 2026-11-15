"""Estimate the yearly fraction of LLM-modified sentences in ISMIR abstracts (Liang et al., 2024).

Corpora:
  P (human reference)  real abstracts 2020-2022, fold A
  Q (AI reference)     synthetic abstracts of the same fold-A papers
  targets              real abstracts per year; fold B for 2020-2022, all papers for 2023-2025
The 2020-2022 papers are split 50/50 by paper, stratified by year, and cross-fitted:
every estimate is the mean of (fit on A, evaluate on B) and (fit on B, evaluate on A).

Usage (from the repo root):
    python liang_estimate/run_liang_analysis.py --stage validation   # check recovery first (~11 min)
    python liang_estimate/run_liang_analysis.py --stage estimates    # then the real estimates
    python liang_estimate/run_liang_analysis.py                      # or everything at once
Outputs go to liang_estimate/results/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from liang_estimator import (VARIANTS, alpha_weighted, bootstrap_ci, fit_pq, mle_alpha,  # noqa: E402
                             paper_resample_weights, responsibilities, tokenize_abstracts)

# ---------------------------------------------------------------- parameters
SEED = 0
HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
REAL = DATA / "abstracts_clean_v1.jsonl"
SYNTH = DATA / "synthetic_abstracts_v1.jsonl"
EDITLENS = DATA / "editlens_abstracts_v1.jsonl"
OUT = HERE / "results"
SPACY_MODEL = "en_core_web_sm"

REFERENCE_YEARS = (2020, 2021, 2022)  # human reference period (before ChatGPT)
YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
MIN_COUNT_P = 5  # vocabulary: word in >= this many P sentences ...
MIN_COUNT_Q = 5  # ... and >= this many Q sentences
SMOOTH_K = 1.0  # add-k smoothing
N_BOOT = 1000

VAL_ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
VAL_SIZE = 100  # abstracts per validation mixture (about one ISMIR year)
VAL_REPS = 30  # random mixtures per alpha and fold direction
MAE_LIMIT = 0.05

N_TOP_WORDS = 25
COLORS = dict(zip(VARIANTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKERS = dict(zip(VARIANTS, ["o", "s", "^", "D", "v"]))
LABELS = {"adj": "adjectives", "adv": "adverbs", "verb": "verbs",
          "all": "all words (no stopwords)", "allstop": "all words (with stopwords)"}
DIRECTIONS = (("A", "B"), ("B", "A"))  # (fit fold, evaluation fold)


# ---------------------------------------------------------------- data
def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_folds(paper_years: dict[str, int], years, rng) -> dict[str, str]:
    """50/50 split by paper, stratified by year. Returns paper_id -> 'A' | 'B'."""
    folds = {}
    for y in years:
        ids = sorted(p for p, yy in paper_years.items() if yy == y)
        rng.shuffle(ids)
        half = len(ids) // 2
        folds.update({p: "A" for p in ids[:half]} | {p: "B" for p in ids[half:]})
    return folds


def sents_of(paper_ids, tokens):
    return [s for p in paper_ids for s in tokens[p]]


def estimate(parts, n_boot, rng):
    """Cross-fitted alpha and paper-bootstrap CI.

    parts: list of (model, sentences). alpha = mean over parts of the MLE of each.
    When two parts share the same sentences (2023-2025: one target set, two fits) they
    share the bootstrap resamples; disjoint parts (2020-2022) are resampled independently.
    """
    alphas, boots, cache = [], [], {}
    for model, sents in parts:
        lp, lq = model.loglik(sents)
        alphas.append(mle_alpha(lp, lq))
        key = id(sents)
        if key not in cache:
            cache[key] = paper_resample_weights([s.paper_id for s in sents], n_boot, rng)
        boots.append(alpha_weighted(lp, lq, cache[key]))
    b = np.mean(boots, axis=0)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(np.mean(alphas)), float(lo), float(hi), alphas


# ---------------------------------------------------------------- stages
def run_validation(ctx) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for variant in VARIANTS:
        for fit_fold, eval_fold in DIRECTIONS:
            model = ctx["fits"][variant][fit_fold]
            pool = sorted(p for p in ctx["ref_ids"] if ctx["folds"][p] == eval_fold)
            for alpha in VAL_ALPHAS:
                for rep in range(VAL_REPS):
                    chosen = rng.choice(pool, size=VAL_SIZE, replace=False)
                    n_ai = round(alpha * VAL_SIZE)
                    sents = sents_of(chosen[n_ai:], ctx["real"]) + sents_of(chosen[:n_ai], ctx["synth"])
                    n_ai_sents = sum(len(ctx["synth"][p]) for p in chosen[:n_ai])
                    lp, lq = model.loglik(sents)
                    a = mle_alpha(lp, lq)
                    lo, hi = bootstrap_ci(lp, lq, [s.paper_id for s in sents], N_BOOT, rng=rng)
                    true = n_ai_sents / len(sents)
                    rows.append(dict(variant=variant, fit_fold=fit_fold, alpha_nominal=alpha, rep=rep,
                                     alpha_true=true, alpha_hat=a, ci_low=lo, ci_high=hi,
                                     covered=lo <= true <= hi, n_sentences=len(sents),
                                     vocab_size=len(model.vocab)))
        print(f"  validated {variant}", flush=True)
    df = pd.DataFrame(rows)
    df["abs_error"] = (df.alpha_hat - df.alpha_true).abs()
    df["ci_width"] = df.ci_high - df.ci_low
    df.to_csv(OUT / "validation.csv", index=False)

    summary = (df.groupby(["variant", "alpha_nominal"])
                 .agg(vocab_size=("vocab_size", "mean"), alpha_true=("alpha_true", "mean"),
                      alpha_hat_mean=("alpha_hat", "mean"), alpha_hat_sd=("alpha_hat", "std"),
                      mae=("abs_error", "mean"), ci_width=("ci_width", "mean"),
                      coverage=("covered", "mean"))
                 .reset_index())
    summary.to_csv(OUT / "validation_summary.csv", index=False)
    plot_validation(df)
    return df


def run_estimates(ctx, passed: list[str]) -> None:
    rng = np.random.default_rng(SEED + 2)
    real, folds, fits = ctx["real"], ctx["folds"], ctx["fits"]
    year_ids = ctx["year_ids"]

    # per-year estimates (main result)
    rows, resp_rows = [], []
    for variant in VARIANTS:
        for y in YEARS:
            if y in REFERENCE_YEARS:
                parts = [(fits[variant][f], sents_of([p for p in year_ids[y] if folds[p] == e], real))
                         for f, e in DIRECTIONS]
            else:
                target = sents_of(year_ids[y], real)
                parts = [(fits[variant][f], target) for f, _ in DIRECTIONS]
            a, lo, hi, per_fit = estimate(parts, N_BOOT, rng)
            n_sents = sum(len(s) for _, s in parts) // (1 if y in REFERENCE_YEARS else 2)
            rows.append(dict(year=y, variant=variant, alpha_hat=a, ci_low=lo, ci_high=hi,
                             alpha_fit_A=per_fit[0], alpha_fit_B=per_fit[1],
                             n_sentences=n_sents, n_papers=len(year_ids[y]),
                             vocab_size_A=len(fits[variant]["A"].vocab),
                             vocab_size_B=len(fits[variant]["B"].vocab),
                             passed_validation=variant in passed))
            # exploratory per-paper responsibilities, averaged over the two fits where both apply
            for (model, sents), alpha in zip(parts, per_fit):
                lp, lq = model.loglik(sents)
                r = responsibilities(alpha, lp, lq)
                resp_rows += [dict(variant=variant, year=y, paper_id=s.paper_id, r=v) for s, v in zip(sents, r)]
    by_year = pd.DataFrame(rows)
    by_year.to_csv(OUT / "alpha_by_year.csv", index=False)
    print(by_year.pivot(index="year", columns="variant", values="alpha_hat").round(3))

    # sensitivity 1: Q from that year's synthetic abstracts (P unchanged), cross-fitted within the year
    year_folds = make_folds({p: y for y in YEARS if y not in REFERENCE_YEARS for p in year_ids[y]},
                            [y for y in YEARS if y not in REFERENCE_YEARS], np.random.default_rng(SEED + 3))
    year_folds |= folds
    sens = []
    for variant in VARIANTS:
        for y in YEARS:
            parts = []
            for f, e in DIRECTIONS:
                p_sents = sents_of([p for p in ctx["ref_ids"] if folds[p] == f], real)
                q_sents = sents_of([p for p in year_ids[y] if year_folds[p] == f], ctx["synth"])
                model = fit_pq(p_sents, q_sents, variant, MIN_COUNT_P, MIN_COUNT_Q, SMOOTH_K)
                parts.append((model, sents_of([p for p in year_ids[y] if year_folds[p] == e], real)))
            a, lo, hi, _ = estimate(parts, N_BOOT, rng)
            main = by_year[(by_year.year == y) & (by_year.variant == variant)].iloc[0]
            sens.append(dict(check="same_year_Q", variant=variant, year=y, alpha_hat=a, ci_low=lo, ci_high=hi,
                             alpha_main=main.alpha_hat, main_ci_low=main.ci_low, main_ci_high=main.ci_high,
                             vocab_size_A=len(parts[0][0].vocab), vocab_size_B=len(parts[1][0].vocab)))
    pd.DataFrame(sens).to_csv(OUT / "sensitivity.csv", index=False)
    print(pd.DataFrame(sens).pivot(index="year", columns="variant", values="alpha_hat").round(3))

    # top discriminative words, fitted on all 2020-2022 papers
    top = []
    for variant in VARIANTS:
        m = fit_pq(sents_of(ctx["ref_ids"], real), sents_of(ctx["ref_ids"], ctx["synth"]),
                   variant, MIN_COUNT_P, MIN_COUNT_Q, SMOOTH_K)
        lr = np.log(m.q / m.p)
        for side, order in (("AI", np.argsort(-lr)), ("human", np.argsort(lr))):
            for rank, j in enumerate(order[:N_TOP_WORDS], 1):
                top.append(dict(variant=variant, more_frequent_in=side, rank=rank, word=m.vocab[j],
                                log_q_over_p=lr[j], p=m.p[j], q=m.q[j],
                                sentences_P=int(m.count_p[j]), sentences_Q=int(m.count_q[j]),
                                n_P=m.n_p, n_Q=m.n_q))
    pd.DataFrame(top).to_csv(OUT / "top_words.csv", index=False)

    # exploratory: per-paper mean responsibility vs EditLens
    resp = pd.DataFrame(resp_rows).groupby(["variant", "year", "paper_id"]).r.mean().reset_index()
    resp = resp.merge(ctx["editlens"], on="paper_id")
    corr = []
    for variant, g in resp.groupby("variant"):
        for name, sub in (("all years", g), ("2023-2025", g[g.year > max(REFERENCE_YEARS)])):
            rho, pval = spearmanr(sub.r, sub.editlens_score)
            corr.append(dict(variant=variant, papers=name, n=len(sub), spearman_rho=rho, p_value=pval))
    resp.to_csv(OUT / "responsibility_per_paper.csv", index=False)
    pd.DataFrame(corr).to_csv(OUT / "responsibility_vs_editlens.csv", index=False)
    print(pd.DataFrame(corr).round(3))

    plot_by_year(by_year, ctx["editlens_by_year"])


# ---------------------------------------------------------------- plots
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_validation(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(3.1 * len(VARIANTS), 3.4), sharex=True, sharey=True)
    for ax, variant in zip(axes, VARIANTS):
        d = df[df.variant == variant]
        ax.plot([0, 0.6], [0, 0.6], color="#8a8980", linewidth=1, linestyle="--", label="identity")
        ax.scatter(d.alpha_true, d.alpha_hat, s=8, alpha=0.25, color=COLORS[variant], linewidths=0)
        g = d.groupby("alpha_nominal")[["alpha_true", "alpha_hat"]].agg(["mean", "std"])
        ax.errorbar(g["alpha_true"]["mean"], g["alpha_hat"]["mean"], yerr=g["alpha_hat"]["std"],
                    color=COLORS[variant], marker=MARKERS[variant], markersize=6, linewidth=2, capsize=3)
        mae = d.abs_error.mean()
        ax.set_title(f"{LABELS[variant]}\nMAE {mae:.3f}, coverage {d.covered.mean():.0%}", fontsize=9)
        ax.set_xlabel("true α (share of sentences)")
        style(ax)
    axes[0].set_ylabel("estimated α")
    fig.suptitle(f"Recovery of known α in mixtures of {VAL_SIZE} abstracts (2020–2022, held-out fold; "
                 f"mean ± sd of {2 * VAL_REPS} mixtures)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "validation.png", dpi=150)
    plt.close(fig)


def plot_by_year(by_year: pd.DataFrame, editlens: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    offsets = np.linspace(-0.2, 0.2, len(VARIANTS))
    for off, variant in zip(offsets, VARIANTS):
        d = by_year[by_year.variant == variant]
        ax1.errorbar(d.year + off, d.alpha_hat, yerr=[d.alpha_hat - d.ci_low, d.ci_high - d.alpha_hat],
                     color=COLORS[variant], marker=MARKERS[variant], markersize=6, linewidth=2, capsize=2,
                     label=LABELS[variant] + ("" if d.passed_validation.iloc[0] else " (failed validation)"))
    ax1.axvline(2022.5, color="#8a8980", linewidth=1, linestyle=":")
    ax1.text(2022.55, ax1.get_ylim()[1] * 0.95, "ChatGPT\n(Nov 2022)", fontsize=8, color="#5f5e57", va="top")
    ax1.set_ylabel("α̂: share of LLM-modified sentences\n(95% CI, paper bootstrap)")
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    style(ax1)
    ax2.plot(editlens.year, editlens.real_mean, color="#2a78d6", marker="o", linewidth=2, label="real abstracts")
    ax2.plot(editlens.year, editlens.synth_mean, color="#8a8980", marker="s", linewidth=1.5, linestyle="--",
             label="synthetic abstracts (reference)")
    ax2.set_ylabel("EditLens mean score")
    ax2.set_xlabel("ISMIR year")
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    style(ax2)
    fig.suptitle("Distributional estimate (Liang et al.) vs EditLens, per year", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "alpha_by_year.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["validation", "estimates", "all"], default="all",
                        help="estimates: reuse results/validation.csv from an earlier run")
    parser.add_argument("--force", action="store_true",
                        help="compute real estimates even if no variant passes validation")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)

    real_rows, synth_rows = read_jsonl(REAL), read_jsonl(SYNTH)
    assert {r["paper_id"] for r in real_rows} == {r["paper_id"] for r in synth_rows}
    print(f"generators in the synthetic set: {sorted({r['generator_model'] for r in synth_rows})}")
    nlp = spacy.load(SPACY_MODEL, disable=["ner"])
    print("tokenizing ...", flush=True)
    real, synth = tokenize_abstracts(real_rows, nlp), tokenize_abstracts(synth_rows, nlp)
    years = {r["paper_id"]: r["year"] for r in real_rows}
    year_ids = {y: sorted(p for p, yy in years.items() if yy == y) for y in YEARS}
    ref_ids = sorted(p for p, y in years.items() if y in REFERENCE_YEARS)
    folds = make_folds({p: years[p] for p in ref_ids}, REFERENCE_YEARS, np.random.default_rng(SEED))

    fits = {}
    for variant in VARIANTS:
        fits[variant] = {f: fit_pq(sents_of([p for p in ref_ids if folds[p] == f], real),
                                   sents_of([p for p in ref_ids if folds[p] == f], synth),
                                   variant, MIN_COUNT_P, MIN_COUNT_Q, SMOOTH_K) for f in "AB"}
        a, b = fits[variant]["A"], fits[variant]["B"]
        print(f"{variant:8s} vocab A={len(a.vocab):4d} B={len(b.vocab):4d}   "
              f"P sentences {a.n_p}/{b.n_p}, Q sentences {a.n_q}/{b.n_q}")

    el = pd.DataFrame(read_jsonl(EDITLENS))[["paper_id", "year", "editlens_score"]]
    el_synth = pd.DataFrame(read_jsonl(EDITLENS.with_name("editlens_synthetic_v1.jsonl")))
    editlens_by_year = pd.DataFrame({
        "year": YEARS,
        "real_mean": [el[el.year == y].editlens_score.mean() for y in YEARS],
        "synth_mean": [el_synth[el_synth.year == y].editlens_score.mean() for y in YEARS],
    })
    ctx = dict(real=real, synth=synth, folds=folds, fits=fits, ref_ids=ref_ids, year_ids=year_ids,
               editlens=el[["paper_id", "editlens_score"]], editlens_by_year=editlens_by_year)

    if args.stage == "estimates":
        val = pd.read_csv(OUT / "validation.csv")
    else:
        print("validation ...", flush=True)
        val = run_validation(ctx)
    mae = val.groupby("variant").abs_error.mean()
    cov = val.groupby("variant").covered.mean()
    print(pd.DataFrame({"MAE": mae, "coverage": cov}).round(3))
    passed = [v for v in VARIANTS if mae[v] <= MAE_LIMIT]
    print(f"variants with MAE <= {MAE_LIMIT}: {passed or 'none'}")
    if args.stage == "validation":
        return
    if not passed and not args.force:
        raise SystemExit("no variant recovers alpha acceptably; stopping before real estimates (use --force)")
    print("real estimates ...", flush=True)
    run_estimates(ctx, passed)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
