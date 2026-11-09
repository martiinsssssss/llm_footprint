"""Tables, significance tests and figures for the three AI detectors (reproduces observations.md).

Reads the per-abstract score files written by editlens.py, fast_detect_gpt.py and
binoculars.py, for both the real and the synthetic abstracts, and writes:

  results/by_year.csv            per detector and year: real and synthetic mean/median (+ 95% CI)
  results/positive_control.csv   per detector: AUC synthetic vs real 2020-2022 (+ 95% CI), paired share
  results/tests.csv              2023-2025 vs 2020-2022, trend over years, per-year tests (real and synthetic)
  results/agreement.csv          Spearman correlations between detectors
  results/editlens_shares.csv    EditLens share above 0.5 / 0.8 and predicted buckets, per year
  results/figures/*.png|pdf      EditLens figures and the positive-control ROC curves (PDF for LaTeX)

Every detector is oriented so that HIGHER = MORE AI before AUCs, pair shares and
correlations are computed (Binoculars is negated); tables of raw scores keep the original
scale.

    python -m ai_detection.analyze                       # defaults: ../data -> ai_detection/results
    python -m ai_detection.analyze --data /path/to/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, spearmanr

SEED = 0
N_BOOT = 2000
PRE = (2020, 2021, 2022)  # before ChatGPT (Nov 2022)
POST = (2023, 2024, 2025)

DETECTORS = {  # name: (file stem, score field, sign that makes higher = more AI, label)
    "editlens": ("editlens", "editlens_score", 1, "EditLens"),
    "fastdetectgpt": ("fastdetectgpt", "fdg_criterion", 1, "Fast-DetectGPT"),
    "binoculars": ("binoculars", "binoculars_score", -1, "Binoculars"),
}

# reference palette (dataviz skill): fixed categorical order, recessive grays
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, GRID = "#8a8980", "#3d3c37", "#e4e3dd"
DET_COLORS = {"editlens": BLUE, "fastdetectgpt": ORANGE, "binoculars": AQUA}
BUCKET_COLORS = ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95"]  # one-hue ordinal ramp, bucket 0 -> 3
BUCKET_LABELS = ["0 · human", "1 · lightly AI-edited", "2 · heavily AI-edited", "3 · fully AI"]


# ---------------------------------------------------------------- data
def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two frames (real, synthetic), one row per paper, one column per detector (raw score)."""
    frames = {}
    for corpus, suffix in (("real", "abstracts"), ("synthetic", "synthetic")):
        df = None
        for name, (stem, field, _, _) in DETECTORS.items():
            rows = read_jsonl(data / f"{stem}_{suffix}_v1.jsonl")
            keep = {"paper_id": [r["paper_id"] for r in rows], name: [r[field] for r in rows]}
            if df is None:
                keep["year"] = [r["year"] for r in rows]
            if name == "editlens":
                keep["editlens_bucket"] = [r["editlens_bucket"] for r in rows]
            part = pd.DataFrame(keep)
            df = part if df is None else df.merge(part, on="paper_id", how="outer", validate="1:1")
        if df[list(DETECTORS)].isna().any().any():
            raise SystemExit(f"{corpus}: some papers lack a score for some detector")
        frames[corpus] = df.sort_values("paper_id").reset_index(drop=True)
    real, synth = frames["real"], frames["synthetic"]
    if not real.paper_id.equals(synth.paper_id):
        raise SystemExit("real and synthetic files do not cover the same papers")
    return real, synth


# ---------------------------------------------------------------- statistics
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random positive > random negative), ties count half (= Mann-Whitney U / n1 n2)."""
    return mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))


def boot_ci(stat, arrays, rng, n_boot=N_BOOT) -> tuple[float, float]:
    """Percentile 95% CI of stat(*arrays), resampling each array independently."""
    vals = [stat(*[a[rng.integers(0, len(a), len(a))] for a in arrays]) for _ in range(n_boot)]
    return tuple(np.percentile(vals, [2.5, 97.5]))


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def roc(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False- and true-positive rates over all thresholds (higher score = positive)."""
    thr = np.unique(np.concatenate([pos, neg]))[::-1]
    tpr = np.array([0.0] + [(pos >= t).mean() for t in thr])
    fpr = np.array([0.0] + [(neg >= t).mean() for t in thr])
    return fpr, tpr


# ---------------------------------------------------------------- tables
def table_by_year(real, synth, rng) -> pd.DataFrame:
    rows = []
    for name, (_, _, sign, label) in DETECTORS.items():
        for y in sorted(real.year.unique()):
            r, s = real[real.year == y][name].values, synth[synth.year == y][name].values
            lo, hi = boot_ci(np.mean, [r], rng)
            slo, shi = boot_ci(np.mean, [s], rng)
            rows.append(dict(detector=label, year=y, n=len(r),
                             real_mean=r.mean(), real_mean_ci_low=lo, real_mean_ci_high=hi, real_median=np.median(r),
                             synthetic_mean=s.mean(), synthetic_mean_ci_low=slo, synthetic_mean_ci_high=shi,
                             synthetic_median=np.median(s),
                             synthetic_more_ai_than_pair=((s - r) * sign > 0).mean()))
    return pd.DataFrame(rows)


def table_positive_control(real, synth, rng) -> pd.DataFrame:
    rows = []
    pre = real.year.isin(PRE).values
    for name, (_, _, sign, label) in DETECTORS.items():
        s, r = synth[name].values * sign, real[name].values * sign
        a = auc(s, r[pre])
        lo, hi = boot_ci(auc, [s, r[pre]], rng)
        rows.append(dict(detector=label, auc_synthetic_vs_real_2020_22=a, ci_low=lo, ci_high=hi,
                         synthetic_more_ai_than_pair=(s > r).mean(), n_synthetic=len(s), n_real_2020_22=pre.sum()))
    return pd.DataFrame(rows)


def table_tests(real, synth) -> pd.DataFrame:
    """Year effects in the real abstracts and, as a topic control, in the synthetic ones.

    Direction is reported with every test (positive = more AI later). The synthetic
    abstracts come from one model and one prompt, so a year effect there means the
    detector reacts to content, not authorship.
    """
    rows = []
    for corpus, df in (("real", real), ("synthetic", synth)):
        pre = df.year.isin(PRE)
        for name, (_, _, sign, label) in DETECTORS.items():
            x = df[name] * sign
            info = dict(corpus=corpus, detector=label)
            u = mannwhitneyu(x[~pre], x[pre])
            rows.append(info | dict(test="Mann-Whitney U, 2023-2025 vs 2020-2022 (two-sided)",
                                    statistic=u.statistic, effect=auc(x[~pre].values, x[pre].values),
                                    effect_name="AUC (P[2023-25 more AI than 2020-22])", p_value=u.pvalue,
                                    p_holm=np.nan))
            tau = kendalltau(df.year, x)
            rows.append(info | dict(test="Kendall tau, year vs score", statistic=tau.statistic,
                                    effect=tau.statistic, effect_name="tau (>0: more AI over time)",
                                    p_value=tau.pvalue, p_holm=np.nan))
            per_year = []
            for y in sorted(df.year.unique()):
                if y == min(PRE):
                    continue
                ref = x[pre & (df.year != y)]
                u = mannwhitneyu(x[df.year == y], ref)
                per_year.append(info | dict(test=f"Mann-Whitney U, {y} vs other 2020-2022 years",
                                            statistic=u.statistic, effect=auc(x[df.year == y].values, ref.values),
                                            effect_name="AUC", p_value=u.pvalue))
            for row, p in zip(per_year, holm([r["p_value"] for r in per_year])):
                rows.append(row | {"p_holm": p})
    return pd.DataFrame(rows)


def table_agreement(real, synth) -> pd.DataFrame:
    rows = []
    names = list(DETECTORS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for corpus, df in (("real", real), ("synthetic", synth)):
                rho, p = spearmanr(df[a] * DETECTORS[a][2], df[b] * DETECTORS[b][2])
                rows.append(dict(pair=f"{DETECTORS[a][3]} vs {DETECTORS[b][3]}", corpus=corpus, spearman_rho=rho,
                                 p_value=p))
    return pd.DataFrame(rows)


def table_editlens_shares(real, synth) -> pd.DataFrame:
    rows = []
    groups = [(f"real {y}", real[real.year == y]) for y in sorted(real.year.unique())]
    groups += [("real 2020-2022", real[real.year.isin(PRE)]), ("real 2023-2025", real[real.year.isin(POST)]),
               ("synthetic (all years)", synth)]
    for label, df in groups:
        row = dict(group=label, n=len(df), share_above_0_5=(df.editlens > 0.5).mean(),
                   share_above_0_8=(df.editlens > 0.8).mean())
        row |= {f"bucket_{b}": int((df.editlens_bucket == b).sum()) for b in range(4)}
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)


def save(fig, path: Path) -> None:
    """PNG for viewing, PDF (vector) for the paper. Titles go in the LaTeX captions, not the images."""
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def chatgpt_line(ax, label=True):
    ax.axvline(2022.5, color=GRAY, linewidth=1, linestyle=":")
    if label:
        ax.text(2022.55, 0.98, "ChatGPT\n(Nov 2022)", transform=ax.get_xaxis_transform(), fontsize=8,
                color=GRAY, va="top")


def _panel_mean(ax, d, real, synth) -> str:
    years = d.year.values
    for col, color, label, ls, mk in (("real", BLUE, "real abstracts", "-", "o"),
                                      ("synthetic", GRAY, "synthetic abstracts (Claude)", "--", "s")):
        m = d[f"{col}_mean"].values
        err = [m - d[f"{col}_mean_ci_low"].values, d[f"{col}_mean_ci_high"].values - m]
        ax.errorbar(years, m, yerr=err, color=color, linestyle=ls, marker=mk, markersize=6, linewidth=2,
                    capsize=3, label=label)
    ax.annotate(f"{d.real_mean.iloc[-1]:.2f}", (years[-1], d.real_mean.iloc[-1]), xytext=(6, 0),
                textcoords="offset points", fontsize=8, color=INK, va="center")
    ax.set_ylabel("mean EditLens score (95% CI)", fontsize=9)
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    return "Mean score"


def _panel_share(ax, d, real, synth) -> str:
    for df, color, label, ls, mk in ((real, BLUE, "real", "-", "o"), (synth, GRAY, "synthetic", "--", "s")):
        share = df.groupby("year").editlens.apply(lambda s: (s > 0.5).mean())
        ax.plot(share.index, share.values * 100, color=color, linestyle=ls, marker=mk, markersize=6, linewidth=2,
                label=label)
    ax.set_ylabel("abstracts scoring above 0.5 (%)", fontsize=9)
    ax.set_ylim(0, 50)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    return "Share above 0.5"


def _panel_paired(ax, d, real, synth) -> str:
    years = d.year.values
    pair = d.synthetic_more_ai_than_pair.values * 100
    ax.bar(years, pair, width=0.6, color=[GRAY if y in PRE else BLUE for y in years])
    for y, v in zip(years, pair):
        ax.text(y, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
    ax.axhline(50, color=INK, linewidth=0.8, linestyle="--")
    ax.text(2025.55, 50, "no\ndifference", fontsize=7.5, color=INK, ha="left", va="center", clip_on=False)
    ax.set_ylabel("papers whose synthetic abstract\nscores higher than the real one (%)", fontsize=9)
    ax.set_ylim(0, 100)
    return "Paired comparison"


SYNTH_X = 2026.3  # x position of the synthetic bar in the levels panel


def _panel_levels(ax, d, real, synth) -> str:
    """Stacked shares of the predicted levels per year, plus one bar for all synthetic abstracts."""
    years = list(d.year.values)
    groups = [real[real.year == y] for y in years] + [synth]
    x = np.array(years + [SYNTH_X], dtype=float)
    shares = np.array([[(g.editlens_bucket == b).mean() * 100 for b in range(4)] for g in groups])
    bottom = np.zeros(len(groups))
    for b in range(4):
        ax.bar(x, shares[:, b], bottom=bottom, width=0.8, color=BUCKET_COLORS[b], edgecolor="white", linewidth=0.8,
               label=BUCKET_LABELS[b])
        bottom += shares[:, b]
    ax.set_xticks(x, [str(y) for y in years] + ["synth."])
    ax.set_xlim(years[0] - 0.6, SYNTH_X + 0.6)
    ax.set_ylabel("abstracts (%)", fontsize=9)
    ax.set_ylim(0, 100)
    return "Predicted level"  # its legend is drawn by the figure (one row below all panels)


PANELS = {"mean": _panel_mean, "share": _panel_share, "paired": _panel_paired, "levels": _panel_levels}


def fig_editlens_by_year(by_year, real, synth, out: Path, panels=("mean", "share", "paired"),
                         name="editlens_by_year", compact=False) -> None:
    """Per-year EditLens panels side by side, lettered a, b, ... in the order given.

    compact: drawn close to print size (about 7.8 in wide for three panels), with shorter
    labels, so the text stays legible when the figure is scaled to the page width.
    """
    d = by_year[by_year.detector == "EditLens"]
    w, h = (2.6, 2.75) if compact else (4.35, 3.9)
    legend_w = (0.95 if compact else 1.5) if "levels" in panels else 0  # room for the levels legend
    fig, axes = plt.subplots(1, len(panels), figsize=(w * len(panels) + legend_w, h))
    for i, (ax, panel) in enumerate(zip(np.atleast_1d(axes), panels)):
        title = PANELS[panel](ax, d, real, synth)
        ax.set_title(f"{'abc'[i]}. {title}", fontsize=10, loc="left")
        chatgpt_line(ax, label=(i == 0 and not compact))  # compact: the caption explains the line
        if panel != "levels":  # the levels panel sets its own ticks (synthetic bar)
            ax.set_xticks(d.year.values)
        ax.set_xlabel("ISMIR year", fontsize=9)
        style(ax)
        if compact:
            ax.set_ylabel({"mean": "mean score (95% CI)", "share": "abstracts above 0.5 (%)",
                           "paired": "synthetic scores higher (%)", "levels": "abstracts (%)"}[panel], fontsize=8)
            ax.set_xlabel("")
            ax.tick_params(labelsize=7)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.title.set_fontsize(9)
            for t in ax.texts:  # annotations (bar values, "no difference") scaled to print size
                t.set_fontsize(min(t.get_fontsize(), 5.5 if panel == "levels" else 6.5))
            if ax.get_legend():
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, [lab.replace(" abstracts", "") for lab in labels], fontsize=7, frameon=False,
                          loc="upper left", handlelength=1.6)
    if "levels" in panels:  # legend right of the levels panel, so it clearly belongs to that panel only
        ax = np.atleast_1d(axes)[list(panels).index("levels")]
        handles, labels = ax.get_legend_handles_labels()
        if compact:
            labels = [lab.split(" · ")[1].replace("AI-edited", "edited") for lab in labels]
        fs = 7 if compact else 8
        ax.legend(handles[::-1], labels[::-1], fontsize=fs, title="predicted\nlevel", title_fontsize=fs,
                  frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.2,
                  alignment="left")
    fig.tight_layout()
    save(fig, out / name)


def fig_editlens_distribution(real, synth, out: Path) -> None:
    """Full score distribution per year: box plots with every abstract as a point."""
    rng = np.random.default_rng(SEED)
    years = sorted(real.year.unique())
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for off, df, color, label in ((-0.19, real, BLUE, "real abstracts"), (0.19, synth, GRAY, "synthetic abstracts")):
        data = [df[df.year == y].editlens.values for y in years]
        pos = np.array(years) + off
        ax.boxplot(data, positions=pos, widths=0.3, showfliers=False, patch_artist=True,
                   medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=INK, linewidth=0.8),
                   capprops=dict(color=INK, linewidth=0.8),
                   boxprops=dict(facecolor="white", edgecolor=color, linewidth=1.5))
        for p, v in zip(pos, data):
            ax.scatter(p + rng.uniform(-0.11, 0.11, len(v)), v, s=7, color=color, alpha=0.45, linewidths=0,
                       zorder=3)
        ax.scatter([], [], s=25, color=color, label=label)
    chatgpt_line(ax)
    ax.set_xticks(years, [str(y) for y in years])
    ax.set_xlabel("ISMIR year", fontsize=9)
    ax.set_ylabel("EditLens score (0 = human, 1 = fully AI)", fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    save(fig, out / "editlens_distribution_by_year")


def fig_editlens_buckets(real, synth, out: Path) -> None:
    """Share of abstracts in each predicted EditLens bucket, per year (real) and for the synthetic set."""
    years = sorted(real.year.unique())
    groups = [(str(y), real[real.year == y]) for y in years] + [("synthetic\n(all years)", synth)]
    shares = np.array([[(df.editlens_bucket == b).mean() * 100 for b in range(4)] for _, df in groups])
    x = np.arange(len(groups)) + np.array([0] * len(years) + [0.6])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bottom = np.zeros(len(groups))
    for b in range(4):
        ax.bar(x, shares[:, b], bottom=bottom, width=0.62, color=BUCKET_COLORS[b], edgecolor="white", linewidth=1,
               label=BUCKET_LABELS[b])
        bottom += shares[:, b]
    ax.axvline((x[len(years) - 4] + x[len(years) - 3]) / 2, color=GRAY, linewidth=1, linestyle=":")
    ax.set_xticks(x, [g for g, _ in groups])
    ax.set_ylabel("abstracts (%)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="predicted bucket",
              title_fontsize=8, reverse=True)
    style(ax)
    fig.tight_layout()
    save(fig, out / "editlens_buckets_by_year")


def fig_positive_control(real, synth, pc: pd.DataFrame, out: Path) -> None:
    """ROC curves: synthetic abstracts vs real 2020-2022 abstracts, all three detectors."""
    pre = real.year.isin(PRE).values
    fig, ax = plt.subplots(figsize=(5.2, 5.8))
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1, linestyle="--")
    ax.text(0.62, 0.55, "chance", fontsize=8, color=GRAY, rotation=45)
    for name, (_, _, sign, label) in DETECTORS.items():
        fpr, tpr = roc(synth[name].values * sign, real[name].values[pre] * sign)
        row = pc[pc.detector == label].iloc[0]
        ax.plot(fpr, tpr, color=DET_COLORS[name], linewidth=2,
                label=f"{label}: AUC {row.auc_synthetic_vs_real_2020_22:.2f} "
                      f"[{row.ci_low:.2f}, {row.ci_high:.2f}]")
    ax.set_xlabel("false-positive rate (real 2020–2022 abstracts flagged)", fontsize=9)
    ax.set_ylabel("true-positive rate (synthetic abstracts flagged)", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    style(ax, grid="both")
    fig.tight_layout()
    save(fig, out / "positive_control_roc")


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("../data"), help="folder with the score files")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    figs = args.out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    real, synth = load(args.data)
    print(f"{len(real)} papers with real and synthetic scores from all {len(DETECTORS)} detectors")

    by_year = table_by_year(real, synth, rng)
    pc = table_positive_control(real, synth, rng)
    tests = table_tests(real, synth)
    agreement = table_agreement(real, synth)
    shares = table_editlens_shares(real, synth)
    for name, df in (("by_year", by_year), ("positive_control", pc), ("tests", tests), ("agreement", agreement),
                     ("editlens_shares", shares)):
        df.to_csv(args.out / f"{name}.csv", index=False)

    fig_editlens_by_year(by_year, real, synth, figs)
    fig_editlens_by_year(by_year, real, synth, figs, ("mean", "paired"), "editlens_by_year_ac")
    fig_editlens_by_year(by_year, real, synth, figs, ("mean", "paired", "levels"), "editlens_summary",
                         compact=True)
    fig_editlens_distribution(real, synth, figs)
    fig_editlens_buckets(real, synth, figs)
    fig_positive_control(real, synth, pc, figs)

    pd.set_option("display.width", 220)
    print(by_year.round(3).to_string(index=False))
    print(pc.round(3).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    print(agreement.round(3).to_string(index=False))
    print(shares.round(3).to_string(index=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
