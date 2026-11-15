"""Excess-vocabulary lower bound on LLM-processed ISMIR abstracts, 2023-2025 (Kobak et al., 2025).

Two word groups, both measured on the real abstracts against a counterfactual
extrapolated from 2020-2022:
  A. kobak_lexicon: the style words Kobak et al. found in excess in PubMed 2024,
     grouped by their PubMed frequency. Fixed in advance (no selection on our data).
  B. data_driven: style words in excess in our own abstracts, selected on one half of
     the papers and measured on the other half (cross-fitted), dropping words that also
     rise in the synthetic abstracts of the same papers (topic words).
Checks: placebo year 2022 (anchors 2020-2021), the synthetic abstracts as a control
corpus, and injecting known shares of synthetic abstracts into 2023-2025.

Usage (from the repo root):  python excess_vocabulary/run_excess_analysis.py
Outputs go to excess_vocabulary/results/.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from excess_vocab import (excess_table, group_bounds, presence_matrix, select_excess_words,  # noqa: E402
                          stratified_weights, tokenize)

# ---------------------------------------------------------------- parameters
SEED = 0
HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
REAL = DATA / "abstracts_clean_v1.jsonl"
SYNTH = DATA / "synthetic_abstracts_v1.jsonl"
EDITLENS = DATA / "editlens_abstracts_v1.jsonl"
OUT = HERE / "results"
KOBAK_LIST = HERE / "lexicon" / "kobak_style_words_2024.csv"
KOBAK_COMMIT = "53db991afc251782106cd817a1c3fa47a4d41781"  # github.com/berenslab/llm-excess-vocab
SPACY_MODEL = "en_core_web_sm"

ANCHORS = (2020, 2021, 2022)  # before ChatGPT (Nov 2022)
TARGETS = (2023, 2024, 2025)
PLACEBO_ANCHORS, PLACEBO_TARGET = (2020, 2021), 2022
METHODS = ("flat", "linear")  # counterfactuals reported side by side (see excess_vocab.counterfactual)
LEXICON_EXTRA_METHODS = ("kobak",)  # the reference's rule, lexicon arm only (too noisy here)

KOBAK_CUTOFFS = (0.01, 0.02, 0.05, 0.1)  # word groups: PubMed 2024 frequency below the cutoff
MAIN_CUTOFF = 0.01  # smallest cutoff: keeps the group's q near 0.45 (0.02 gives ~0.75)

CONTENT_POS = {"NOUN", "PROPN", "NUM", "SYM", "X"}  # data-driven arm keeps every other POS
GAP_MIN, RATIO_MIN, MIN_DOCS, TOPIC_SHARE = 0.03, 1.5, 5, 0.5
BASELINE_MAX, MAX_WORDS = 0.1, 15  # data-driven group: rare-ish words only, top 15 by gap

N_BOOT = 1000
INJECT_SHARES = (0.0, 0.1, 0.2, 0.3, 0.5)
INJECT_REPS = 20

COLORS = {"kobak_lexicon": "#2a78d6", "data_driven": "#eb6834"}
LABELS = {"kobak_lexicon": f"A. Kobak et al. style words (PubMed freq < {MAIN_CUTOFF})",
          "data_driven": "B. data-driven excess words (cross-fitted)"}


# ---------------------------------------------------------------- data
def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def kobak_style_words() -> pd.DataFrame:
    """Style words in excess in PubMed 2024 (their criterion and annotations), cached in lexicon/."""
    if KOBAK_LIST.exists():
        return pd.read_csv(KOBAK_LIST)
    base = f"https://raw.githubusercontent.com/berenslab/llm-excess-vocab/{KOBAK_COMMIT}/results/"
    print("downloading the Kobak et al. counts and annotations ...", flush=True)
    with urllib.request.urlopen(base + "yearly-counts.csv.gz") as r:
        yc = pd.read_csv(io.BytesIO(r.read()), compression="gzip", keep_default_na=False)
    with urllib.request.urlopen(base + "excess_words.csv") as r:
        ann = pd.read_csv(io.BytesIO(r.read()))
    totals = yc.iloc[-1, 1:].astype(float).values  # last row: number of abstracts per year
    yc = yc.iloc[:-1]
    years = np.array([int(c) for c in yc.columns[1:]])
    f = (yc.iloc[:, 1:].astype(float).values + 1) / (totals + 1)
    # their 03-figures.ipynb: compute_excess(2024) and the excess criterion
    p, a, b = f[:, years == 2024].ravel(), f[:, years == 2022].ravel(), f[:, years == 2021].ravel()
    q = a + np.maximum(a - b, 0) * 2
    excess = (np.log10(p / q) > np.log10(2) - (np.log10(p) + 4) * (np.log10(2) / 4)) | (p - q > 0.01)
    style = set(ann.word[ann.type == "style"])
    df = pd.DataFrame({"word": yc.word.values, "pubmed_freq_2024": p, "pubmed_gap_2024": p - q,
                       "pubmed_ratio_2024": p / q})[excess & yc.word.isin(style).values]
    df = df.sort_values("pubmed_gap_2024", ascending=False)
    KOBAK_LIST.parent.mkdir(exist_ok=True)
    df.to_csv(KOBAK_LIST, index=False)
    return df


def make_folds(years: np.ndarray, rng) -> np.ndarray:
    """50/50 split of papers, stratified by year: array of 'A' / 'B'."""
    folds = np.empty(len(years), dtype="<U1")
    for y in np.unique(years):
        idx = np.where(years == y)[0]
        rng.shuffle(idx)
        folds[idx[: len(idx) // 2]], folds[idx[len(idx) // 2:]] = "A", "B"
    return folds


def word_pos(texts: list[str], nlp) -> dict[str, str]:
    """Most frequent coarse POS tag of each lowercased word in the corpus."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for doc in nlp.pipe(texts, batch_size=64):
        for t in doc:
            counts[t.text.lower()][t.pos_] += 1
    return {w: c.most_common(1)[0][0] for w, c in counts.items()}


# ---------------------------------------------------------------- analyses
def ci(values) -> tuple[float, float]:
    """95% percentile interval; NaN if more than 5% of resamples have no bound (q >= 0.9)."""
    v = np.asarray(values, dtype=float)
    if np.isnan(v).mean() > 0.05:
        return np.nan, np.nan
    return tuple(np.nanpercentile(v, [2.5, 97.5]))


def bound_rows(group, years, targets, anchors, method, rng, **info) -> list[dict]:
    """Point bounds and percentile bootstrap CIs (papers resampled within year)."""
    point = group_bounds(group, years, targets, anchors, method)
    boot = group_bounds(group, years, targets, anchors, method, stratified_weights(years, N_BOOT, rng))
    rows = []
    for key, v in point.items():
        lo, hi = ci(boot[key]["bound"])
        n = int(np.isin(years, targets).sum()) if key == "pooled" else int((years == key).sum())
        rows.append(dict(info, method=method, year=str(key), p=v["p"], q=v["q"], delta=v["delta"],
                         bound=v["bound"], ci_low=lo, ci_high=hi, n_papers=n))
    return rows


def cross_fitted(ctx, x_real, x_synth, targets, anchors, method, rng, n_boot=N_BOOT):
    """Arm B: select on one fold, measure on the other, average the two directions.

    Returns {corpus: {year: values incl. a bootstrap array of bounds}} and the selected words.
    """
    years, folds = ctx["years"], ctx["folds"]
    point, boots, selected = defaultdict(list), defaultdict(list), {}
    for sel, ev in (("A", "B"), ("B", "A")):
        s, e = folds == sel, folds == ev
        idx = select_excess_words(x_real[s], x_synth[s], years[s], ctx["style_mask"], targets, anchors, method,
                                  GAP_MIN, RATIO_MIN, MIN_DOCS, TOPIC_SHARE, BASELINE_MAX, MAX_WORDS)
        selected[sel] = idx
        w = stratified_weights(years[e], n_boot, rng) if n_boot else None
        for corpus, x in (("real", x_real), ("synthetic", x_synth)):
            g = x[e][:, idx].any(1)
            pt = group_bounds(g, years[e], targets, anchors, method)
            bt = group_bounds(g, years[e], targets, anchors, method, w) if n_boot else None
            for key in pt:
                point[(corpus, key)].append(pt[key])
                if n_boot:
                    boots[(corpus, key)].append(bt[key]["bound"])
    out = {}
    for (corpus, key), vals in point.items():
        mean = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        mean["boot"] = np.mean(boots[(corpus, key)], axis=0) if n_boot else None
        out.setdefault(corpus, {})[key] = mean
    return out, selected


def run(ctx) -> None:
    rng = np.random.default_rng(SEED + 1)
    years, xr, xs, vocab = ctx["years"], ctx["x_real"], ctx["x_synth"], ctx["vocab"]
    index = {w: j for j, w in enumerate(vocab)}
    rows = []

    # A. Kobak lexicon (fixed word groups)
    kob = ctx["kobak"]
    lexicon_groups = {}
    for cutoff in KOBAK_CUTOFFS:
        words = kob.word[kob.pubmed_freq_2024 < cutoff].tolist()
        idx = [index[w] for w in words if w in index]
        lexicon_groups[cutoff] = idx
        info = dict(arm="kobak_lexicon", cutoff=cutoff, n_words=len(words),
                    n_words_in_ismir=int(xr[:, idx].any(0).sum()))
        p_mask = years <= PLACEBO_TARGET
        for corpus, x in (("real", xr), ("synthetic", xs)):
            g = x[:, idx].any(1)
            for method in METHODS + LEXICON_EXTRA_METHODS:
                rows += bound_rows(g, years, TARGETS, ANCHORS, method, rng, corpus=corpus, placebo=False, **info)
            for method in METHODS:
                rows += bound_rows(g[p_mask], years[p_mask], (PLACEBO_TARGET,), PLACEBO_ANCHORS, method, rng,
                                   corpus=corpus, placebo=True, **info)

    # B. data-driven, cross-fitted
    p_mask = years <= PLACEBO_TARGET
    sub = dict(ctx, years=years[p_mask], folds=ctx["folds"][p_mask])
    sel_rows = []
    for method in METHODS:
        main_b, selected = cross_fitted(ctx, xr, xs, TARGETS, ANCHORS, method, rng)
        plac_b, plac_sel = cross_fitted(sub, xr[p_mask], xs[p_mask], (PLACEBO_TARGET,), PLACEBO_ANCHORS,
                                        method, rng)
        for res, placebo, sel in ((main_b, False, selected), (plac_b, True, plac_sel)):
            for corpus, by_key in res.items():
                for key, v in by_key.items():
                    lo, hi = ci(v["boot"])
                    n = int((years == key).sum()) if key != "pooled" else int(np.isin(years, TARGETS).sum())
                    rows.append(dict(arm="data_driven", cutoff=None, n_words=f"{len(sel['A'])}/{len(sel['B'])}",
                                     n_words_in_ismir=None, corpus=corpus, placebo=placebo, method=method,
                                     year=str(key), p=v["p"], q=v["q"], delta=v["delta"], bound=v["bound"],
                                     ci_low=lo, ci_high=hi, n_papers=n))
            sel_rows += [dict(method=method, selection="placebo 2022" if placebo else "2023-2025",
                              selected_on_fold=f, rank=r, word=vocab[j], pos=ctx["pos"].get(vocab[j]),
                              in_kobak_lexicon=vocab[j] in set(kob.word))
                         for f, idx in sel.items() for r, j in enumerate(idx, 1)]
    bounds = pd.DataFrame(rows)
    bounds.to_csv(OUT / "lower_bound_by_year.csv", index=False)
    selected = pd.DataFrame(sel_rows)
    selected.to_csv(OUT / "selected_words.csv", index=False)

    # per-word excess on all papers
    docs_post = xr[np.isin(years, TARGETS)].sum(0)
    cols = {}
    for method in METHODS:
        er = excess_table(xr, years, TARGETS, ANCHORS, method)
        es = excess_table(xs, years, TARGETS, ANCHORS, method)
        cols |= {f"q_{method}": er["q"], f"gap_{method}": er["gap"], f"ratio_{method}": er["ratio"],
                 f"synthetic_gap_{method}": es["gap"]}
    words = pd.DataFrame({
        "word": vocab, "pos": [ctx["pos"].get(w) for w in vocab], "style": ctx["style_mask"],
        "in_kobak_lexicon": np.isin(vocab, kob.word), "abstracts_2023_25": docs_post,
        "p_2023_25": excess_table(xr, years, TARGETS, ANCHORS)["p"], **cols,
        **{f"share_{y}": xr[years == y].mean(0) for y in sorted(set(years))},
        **{f"synthetic_share_{y}": xs[years == y].mean(0) for y in sorted(set(years))},
    })
    words = words[docs_post >= MIN_DOCS].sort_values("gap_flat", ascending=False)
    words.to_csv(OUT / "excess_words.csv", index=False)

    # injection control: replace a known share of 2023-2025 real abstracts by their synthetic version
    inj = []
    post = np.where(np.isin(years, TARGETS))[0]
    lex = lexicon_groups[MAIN_CUTOFF]
    for share in INJECT_SHARES:
        for rep in range(INJECT_REPS if share > 0 else 1):
            chosen = rng.choice(post, size=round(share * len(post)), replace=False)
            x_inj = xr.copy()
            x_inj[chosen] = xs[chosen]
            row = dict(injected_share=share, rep=rep)
            for method in METHODS:
                b, _ = cross_fitted(ctx, x_inj, xs, TARGETS, ANCHORS, method, rng, n_boot=0)
                a = group_bounds(x_inj[:, lex].any(1), years, TARGETS, ANCHORS, method)
                row |= {f"data_driven_{method}": b["real"]["pooled"]["bound"],
                        f"kobak_lexicon_{method}": a["pooled"]["bound"]}
            inj.append(row)
    inj = pd.DataFrame(inj)
    inj.to_csv(OUT / "injection.csv", index=False)

    # exploratory: per-abstract count of lexicon words vs EditLens
    el = ctx["editlens"]
    n_lex = xr[:, lex].sum(1)
    corr = []
    for name, m in (("all years", np.ones(len(years), bool)), ("2023-2025", np.isin(years, TARGETS))):
        rho, pval = spearmanr(n_lex[m], el[m])
        corr.append(dict(papers=name, n=int(m.sum()), spearman_rho=rho, p_value=pval))
    pd.DataFrame(corr).to_csv(OUT / "lexicon_vs_editlens.csv", index=False)

    report(bounds, inj, words, selected, pd.DataFrame(corr))
    plot_bounds(bounds, ctx["editlens_by_year"])
    plot_words(words)


# ---------------------------------------------------------------- output
def report(bounds, inj, words, selected, corr) -> None:
    pd.set_option("display.width", 250)
    b = bounds.assign(
        txt=lambda d: d.apply(lambda r: f"{r.bound:+.3f} [{r.ci_low:+.3f},{r.ci_high:+.3f}] q={r.q:.2f}", axis=1),
        cutoff=lambda d: d.cutoff.fillna("-"))
    print(b.pivot_table(index=["arm", "cutoff", "method", "corpus", "placebo"], columns="year", values="txt",
                        aggfunc="first").to_string())
    cols = [c for c in inj.columns if c not in ("injected_share", "rep")]
    print(inj.groupby("injected_share")[cols].agg(["mean", "std"]).round(3).to_string())
    for (m, s, f), g in selected.groupby(["method", "selection", "selected_on_fold"]):
        print(f"{m:6s} {s:12s} fold {f}: {', '.join(g.word)}")
    print(words[words["style"]].head(30)[["word", "pos", "in_kobak_lexicon", "abstracts_2023_25", "q_flat",
                                         "p_2023_25", "gap_flat", "gap_linear", "synthetic_gap_flat"]]
          .round(3).to_string())
    print(corr.round(3))


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_bounds(bounds: pd.DataFrame, editlens: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(METHODS) + 1, 1, figsize=(8, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [3] * len(METHODS) + [2]})
    d = bounds[(bounds.year != "pooled") & ((bounds.arm == "data_driven") | (bounds.cutoff == MAIN_CUTOFF))]
    d = d[d.method.isin(METHODS)].assign(year=lambda x: x.year.astype(int))
    titles = {"flat": "counterfactual = mean of the reference years",
              "linear": "counterfactual = linear trend of the reference years (slope ≥ 0)"}
    for ax, method in zip(axes, METHODS):
        for off, arm in ((-0.1, "kobak_lexicon"), (0.06, "data_driven")):
            for corpus, ls, mfc in (("real", "-", None), ("synthetic", ":", "white")):
                s = d[(d.arm == arm) & (d.corpus == corpus) & (d.method == method)].sort_values("year")
                for placebo in (False, True):
                    t = s[s.placebo == placebo]
                    err = np.clip([t.bound - t.ci_low, t.ci_high - t.bound], 0, None)
                    ax.errorbar(t.year + off + (0.05 if corpus == "synthetic" else 0), t.bound, yerr=err,
                                color=COLORS[arm], linestyle=ls if not placebo else "none",
                                marker="o" if not placebo else "D", markersize=6,
                                markerfacecolor=mfc or COLORS[arm], linewidth=2 if corpus == "real" else 1.2,
                                capsize=2, alpha=1 if corpus == "real" else 0.75,
                                label=(f"{LABELS[arm]}: {'real' if corpus == 'real' else 'synthetic control'}"
                                       if not placebo else None))
        ax.axhline(0, color="#8a8980", linewidth=1)
        ax.axvline(2022.5, color="#8a8980", linewidth=1, linestyle=":")
        ax.set_title(titles[method], fontsize=9, loc="left")
        ax.set_ylabel("lower bound on share of\nLLM-processed abstracts")
        style(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7.5, frameon=False, loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 0.965))
    for ax in axes[:-1]:
        ax.text(2022.0, ax.get_ylim()[1], "placebo: 2022\nvs 2020–21", fontsize=8,
                color="#5f5e57", va="top", ha="center")
    ax2 = axes[-1]
    ax2.plot(editlens.year, editlens.real_mean, color="#2a78d6", marker="o", linewidth=2)
    ax2.set_ylabel("EditLens mean score\n(real abstracts)")
    ax2.set_xlabel("ISMIR year (2020–2021: reference only)")
    ax2.set_xticks(range(2020, 2026))
    style(ax2)
    fig.suptitle("Excess-vocabulary lower bound (Kobak et al.), 95% CI, vs EditLens", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "lower_bound_by_year.png", dpi=150)
    plt.close(fig)


def plot_words(words: pd.DataFrame, n: int = 25) -> None:
    top = words[words["style"]].head(n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.28 * n + 1.2))
    y = np.arange(len(top))
    ax.barh(y + 0.2, top.gap_flat, height=0.38, color="#2a78d6", label="real abstracts")
    ax.barh(y - 0.2, top.synthetic_gap_flat, height=0.38, color="#8a8980", label="synthetic abstracts (same papers)")
    ax.set_yticks(y, [f"{w}{' *' if k else ''}" for w, k in zip(top.word, top.in_kobak_lexicon)], fontsize=8)
    ax.axvline(0, color="#5f5e57", linewidth=0.8)
    ax.set_xlabel("excess share of abstracts in 2023–2025 (observed − mean of 2020–2022)")
    ax.set_title(f"Top {n} style words in excess, 2023–2025  (* = in the Kobak et al. list)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "top_excess_words.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(exist_ok=True)
    real, synth = read_jsonl(REAL), read_jsonl(SYNTH)
    synth_by_id = {r["paper_id"]: r for r in synth}
    synth = [synth_by_id[r["paper_id"]] for r in real]  # same paper order
    years = np.array([r["year"] for r in real])
    docs_r, docs_s = [tokenize(r["abstract"]) for r in real], [tokenize(r["abstract"]) for r in synth]
    vocab = sorted(set().union(*docs_r, *docs_s))
    print(f"{len(real)} papers, {len(vocab)} word types", flush=True)

    nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser", "lemmatizer"])
    pos = word_pos([r["abstract"] for r in real + synth], nlp)
    el = {r["paper_id"]: r for r in read_jsonl(EDITLENS)}
    ctx = dict(
        years=years, vocab=vocab, pos=pos,
        x_real=presence_matrix(docs_r, vocab), x_synth=presence_matrix(docs_s, vocab),
        style_mask=np.array([pos.get(w) is not None and pos[w] not in CONTENT_POS for w in vocab]),
        folds=make_folds(years, np.random.default_rng(SEED)), kobak=kobak_style_words(),
        editlens=np.array([el[r["paper_id"]]["editlens_score"] for r in real]),
        editlens_by_year=pd.DataFrame({"year": sorted(set(years)), "real_mean": [
            np.mean([el[r["paper_id"]]["editlens_score"] for r in real if r["year"] == y])
            for y in sorted(set(years))]}),
    )
    print(f"Kobak et al. style words: {len(ctx['kobak'])}", flush=True)
    run(ctx)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
