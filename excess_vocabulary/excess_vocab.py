"""Excess-vocabulary lower bound on LLM use (Kobak et al., Science Advances 2025).

Kobak, González-Márquez, Horvát & Lause, "Delving into LLM-assisted writing in
biomedical publications through excess vocabulary", Sci. Adv. 11(27), eadt3813
(arXiv:2406.07016). Reference code: github.com/berenslab/llm-excess-vocab (MIT).

Idea: no AI reference corpus is needed. For a word (or a group of words) let p be
the share of abstracts in a target year that contain it, and q the counterfactual
share expected without LLMs, extrapolated from the years before ChatGPT. The gap
delta = p - q is the "excess". If a fraction L of abstracts was LLM-processed and the
others follow q, then p = (1 - L) q + L r with r <= 1, so

    L >= (p - q) / (1 - q)

which is a lower bound on L for any group of words that LLMs favor (it is tight only
if every LLM-processed abstract contains the group). Groups of words that are rare
without LLMs (small q) give the most informative bounds.

All frequencies here are shares of ABSTRACTS containing a word at least once.
"""

from __future__ import annotations

import re

import numpy as np

_TOKEN = re.compile(r"\b\w\w+\b")  # sklearn CountVectorizer default, as in the reference


def tokenize(text: str) -> set[str]:
    """Lowercased word types of an abstract (tokens with digits or underscores dropped)."""
    return {t for t in _TOKEN.findall(text.lower()) if t.isalpha()}


def presence_matrix(docs: list[set[str]], vocab: list[str]) -> np.ndarray:
    """Boolean (n_docs, n_words) matrix: does document i contain word j."""
    index = {w: j for j, w in enumerate(vocab)}
    x = np.zeros((len(docs), len(vocab)), dtype=bool)
    for i, d in enumerate(docs):
        for w in d:
            j = index.get(w)
            if j is not None:
                x[i, j] = True
    return x


def yearly_freq(x: np.ndarray, years: np.ndarray, year_list, smooth: float = 1.0) -> np.ndarray:
    """Share of each year's documents containing each column: (count + smooth) / (n + smooth).

    The reference smooths single-word frequencies with +1 so that ratios are finite.
    """
    return np.array([(x[years == y].sum(0) + smooth) / ((years == y).sum() + smooth) for y in year_list])


def counterfactual(freq_by_year: dict, target: int, anchors: tuple[int, ...], method: str = "linear"):
    """Expected frequency in `target` without LLMs, from the anchor (pre-LLM) years.

    flat:   mean of the anchor years
    kobak:  last anchor + max(0, slope between the last two anchors) x years ahead (the reference)
    linear: least-squares line through all anchors, slope clipped at >= 0
    Upward trends are extrapolated, downward ones are not, which can only lower the bound.
    Values may be scalars or arrays (one per word or per bootstrap sample).
    """
    a = np.array(anchors, dtype=float)
    f = np.stack([np.asarray(freq_by_year[y], dtype=float) for y in anchors])
    if method == "flat":
        return f.mean(0)
    if method == "kobak":
        slope = np.maximum(f[-1] - f[-2], 0) / (a[-1] - a[-2])
        return f[-1] + slope * (target - a[-1])
    if method == "linear":
        ac = (a - a.mean()).reshape((-1,) + (1,) * (f.ndim - 1))
        slope = np.maximum((ac * (f - f.mean(0))).sum(0) / (ac ** 2).sum(), 0)
        return f.mean(0) + slope * (target - a.mean())
    raise ValueError(method)


def lower_bound(p, q, q_max: float = 0.9):
    """(p - q) / (1 - q): lower bound on the share of LLM-processed documents.

    NaN when q >= q_max: a group that almost every abstract contains anyway carries no
    information, and dividing by 1 - q would only amplify noise.
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(q < q_max, (p - q) / (1 - q), np.nan)


def group_bounds(group: np.ndarray, years: np.ndarray, targets, anchors, method: str = "linear",
                 weights: np.ndarray | None = None) -> dict:
    """p, q, delta and the lower bound of a word group, per target year and pooled ("pooled").

    group: boolean per document (contains at least one word of the group). Group shares
    are not smoothed. weights: optional (B, n_docs) bootstrap weights; every value is
    then an array of B. The pooled q is the document-weighted mean of the yearly q.
    """
    g = np.asarray(group, dtype=float)
    w = np.ones((1, len(g))) if weights is None else weights

    def share(y):
        m = years == y
        return (w[:, m] * g[m]).sum(1) / w[:, m].sum(1)

    freq = {y: share(y) for y in set(anchors) | set(targets)}
    out, p_sum, q_sum, n_sum = {}, 0.0, 0.0, 0.0
    for t in targets:
        q = counterfactual(freq, t, anchors, method)
        out[t] = dict(p=freq[t], q=q, delta=freq[t] - q, bound=lower_bound(freq[t], q))
        n_t = w[:, years == t].sum(1)
        p_sum, q_sum, n_sum = p_sum + freq[t] * n_t, q_sum + q * n_t, n_sum + n_t
    p, q = p_sum / n_sum, q_sum / n_sum
    out["pooled"] = dict(p=p, q=q, delta=p - q, bound=lower_bound(p, q))
    if weights is None:
        out = {k: {kk: float(np.ravel(vv)[0]) for kk, vv in v.items()} for k, v in out.items()}
    return out


def stratified_weights(years: np.ndarray, n_boot: int, rng: np.random.Generator,
                       mask: np.ndarray | None = None) -> np.ndarray:
    """(n_boot, n_docs) bootstrap weights, resampling documents within each year.

    Documents outside `mask` get weight 0.
    """
    mask = np.ones(len(years), dtype=bool) if mask is None else mask
    w = np.zeros((n_boot, len(years)))
    for y in np.unique(years[mask]):
        idx = np.where(mask & (years == y))[0]
        w[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot)
    return w


def excess_table(x: np.ndarray, years: np.ndarray, targets, anchors, method: str = "linear",
                 smooth: float = 1.0) -> dict[str, np.ndarray]:
    """Per-word excess pooled over the target years: p, q, gap = p - q, ratio = p / q."""
    yl = sorted(set(anchors) | set(targets))
    f = dict(zip(yl, yearly_freq(x, years, yl, smooth)))
    n = np.array([(years == t).sum() for t in targets], dtype=float)
    p = sum(f[t] * k for t, k in zip(targets, n)) / n.sum()
    q = sum(counterfactual(f, t, anchors, method) * k for t, k in zip(targets, n)) / n.sum()
    return dict(p=p, q=q, gap=p - q, ratio=p / q)


def select_excess_words(x_real: np.ndarray, x_synth: np.ndarray, years: np.ndarray, style_mask: np.ndarray,
                        targets, anchors, method: str = "linear", gap_min: float = 0.03,
                        ratio_min: float = 1.5, min_docs: int = 5, topic_share: float = 0.5,
                        baseline_max: float = 0.1, max_words: int | None = 15) -> np.ndarray:
    """Indices of style words in excess in the real abstracts but not in the synthetic ones.

    Selected if, pooled over the target years: gap >= gap_min, ratio >= ratio_min,
    present in >= min_docs target-year abstracts, counterfactual share q <= baseline_max
    (rare words keep the group's q, and so the bound, informative), style_mask is True,
    and the gap of the same word in the synthetic abstracts of the same papers is below
    topic_share x the real gap. The synthetic abstracts come from one model with a
    constant style, so a word that also rises there most likely rises because of topics.
    Of the words that pass, the `max_words` with the largest gap are kept.
    """
    real = excess_table(x_real, years, targets, anchors, method)
    synth = excess_table(x_synth, years, targets, anchors, method)
    docs = x_real[np.isin(years, targets)].sum(0)
    keep = ((real["gap"] >= gap_min) & (real["ratio"] >= ratio_min) & (docs >= min_docs)
            & (real["q"] <= baseline_max) & style_mask & (synth["gap"] < topic_share * real["gap"]))
    idx = np.where(keep)[0]
    idx = idx[np.argsort(-real["gap"][idx], kind="stable")]
    return idx[:max_words] if max_words else idx
