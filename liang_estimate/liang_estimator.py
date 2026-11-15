"""Distributional estimate of the fraction of LLM-modified sentences (Liang et al., 2024).

Method from Liang et al., "Monitoring AI-Modified Content at Scale" (ICML 2024,
arXiv:2403.07183) and "Mapping the Increasing Use of LLMs in Scientific Papers"
(COLM 2024, arXiv:2404.01268); reference code:
github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers (MIT).

Each sentence is a binary bag of words x over a vocabulary V. Human text P and AI
text Q are modelled as independent Bernoullis, one per word:

    log P(x) = sum_{w in V} [ x_w log p_w + (1 - x_w) log(1 - p_w) ]     (same for Q)

and a target corpus as the mixture (1 - alpha) P + alpha Q. alpha is the
maximum-likelihood fraction of sentences drawn from Q:

    alpha_hat = argmax_alpha  sum_s log( (1 - alpha) P(s) + alpha Q(s) )

This is the reference repo's formulation (src/MLE.py: presence AND absence terms).
Differences from the repo, all deliberate:
  - add-k smoothing of p_w and q_w (repo: none; k=0 reproduces it);
  - the point estimate is the MLE on the full target set (repo: the midpoint of
    the bootstrap CI);
  - the bootstrap resamples papers, not sentences (sentences of one abstract are
    correlated).

Typical use:
    sents = tokenize_abstracts(rows, nlp)             # one Sentence per spaCy sentence
    model = fit_pq(p_sents, q_sents, variant="adj")   # vocabulary + p_w, q_w
    lp, lq = model.loglik(target_sents)
    alpha = mle_alpha(lp, lq)
    lo, hi = bootstrap_ci(lp, lq, paper_ids, rng=rng)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

VARIANTS = ("adj", "adv", "verb", "all", "allstop")
"""Feature sets. adj/adv/verb: one part of speech (spaCy coarse tags ADJ, ADV, VERB);
all: every word except stopwords; allstop: every word, stopwords included (what the
reference repo's tokenizer keeps)."""

_WORD = re.compile(r"[a-z][a-z'-]*")


@dataclass
class Sentence:
    """One tokenized sentence. `words` maps each variant to its set of words."""

    paper_id: str
    words: dict[str, frozenset[str]] = field(default_factory=dict)


def tokenize_abstract(paper_id: str, text: str, nlp) -> list[Sentence]:
    """Split an abstract into sentences and build each variant's binary bag of words.

    Words are lowercased token texts (no lemmatization, as in the reference) that
    start with a letter, so numbers and punctuation are dropped. Sentences with
    fewer than 2 words (all words, stopwords included) are dropped, as in the
    reference repo.
    """
    return _sentences(paper_id, nlp(text))


def _sentences(paper_id: str, doc) -> list[Sentence]:
    out = []
    for sent in doc.sents:
        toks = [(t.text.lower(), t.pos_, t.is_stop) for t in sent if _WORD.fullmatch(t.text.lower())]
        if len(toks) < 2:
            continue
        out.append(Sentence(paper_id, {
            "adj": frozenset(w for w, pos, _ in toks if pos == "ADJ"),
            "adv": frozenset(w for w, pos, _ in toks if pos == "ADV"),
            "verb": frozenset(w for w, pos, _ in toks if pos == "VERB"),
            "all": frozenset(w for w, _, stop in toks if not stop),
            "allstop": frozenset(w for w, _, _ in toks),
        }))
    return out


def tokenize_abstracts(rows: list[dict], nlp) -> dict[str, list[Sentence]]:
    """Tokenize {paper_id, abstract} rows; returns paper_id -> sentences."""
    docs = nlp.pipe((r["abstract"] for r in rows), batch_size=64)
    return {r["paper_id"]: _sentences(r["paper_id"], doc) for r, doc in zip(rows, docs)}


@dataclass
class PQModel:
    """Fitted word distributions for one vocabulary variant."""

    variant: str
    vocab: list[str]
    p: np.ndarray  # P(word in sentence | human), smoothed
    q: np.ndarray  # P(word in sentence | AI), smoothed
    count_p: np.ndarray  # number of P sentences containing the word
    count_q: np.ndarray
    n_p: int  # number of P sentences
    n_q: int

    def __post_init__(self):
        self.index = {w: i for i, w in enumerate(self.vocab)}
        lp1, lq1 = np.log1p(-self.p), np.log1p(-self.q)
        # log P(x) = sum_w log(1-p_w) + sum_{w in x} [log p_w - log(1-p_w)]
        self._base_p, self._base_q = lp1.sum(), lq1.sum()
        self._delta_p = np.log(self.p) - lp1
        self._delta_q = np.log(self.q) - lq1

    def indicator(self, sents: list[Sentence]) -> np.ndarray:
        """Binary sentence x vocabulary matrix (words outside the vocabulary are ignored)."""
        x = np.zeros((len(sents), len(self.vocab)), dtype=np.float64)
        for i, s in enumerate(sents):
            for w in s.words[self.variant]:
                j = self.index.get(w)
                if j is not None:
                    x[i, j] = 1.0
        return x

    def loglik(self, sents: list[Sentence]) -> tuple[np.ndarray, np.ndarray]:
        """Per-sentence log P(x) and log Q(x) (full Bernoulli, presence and absence terms)."""
        x = self.indicator(sents)
        return self._base_p + x @ self._delta_p, self._base_q + x @ self._delta_q


def fit_pq(p_sents: list[Sentence], q_sents: list[Sentence], variant: str,
           min_count_p: int = 5, min_count_q: int = 5, k: float = 1.0) -> PQModel:
    """Estimate p_w and q_w from human (P) and AI (Q) reference sentences.

    Vocabulary: words in at least `min_count_p` P sentences AND at least
    `min_count_q` Q sentences (reference repo: 5 and 3). Probabilities use add-k
    smoothing on the binary occurrence: p_w = (count_w + k) / (n + 2k).
    """
    cp = Counter(w for s in p_sents for w in s.words[variant])
    cq = Counter(w for s in q_sents for w in s.words[variant])
    vocab = sorted(w for w in cp.keys() & cq.keys() if cp[w] >= min_count_p and cq[w] >= min_count_q)
    count_p = np.array([cp[w] for w in vocab], dtype=float)
    count_q = np.array([cq[w] for w in vocab], dtype=float)
    n_p, n_q = len(p_sents), len(q_sents)
    return PQModel(variant, vocab,
                   (count_p + k) / (n_p + 2 * k), (count_q + k) / (n_q + 2 * k),
                   count_p, count_q, n_p, n_q)


def mixture_loglik(alpha: float, lp: np.ndarray, lq: np.ndarray) -> float:
    """sum_s log((1 - alpha) P(s) + alpha Q(s)), computed with logsumexp."""
    a = np.stack([lp, lq])
    b = np.array([[1.0 - alpha], [alpha]])
    with np.errstate(divide="ignore"):
        return float(logsumexp(a, axis=0, b=b).sum())


def mle_alpha(lp: np.ndarray, lq: np.ndarray, check_grid: bool = True) -> float:
    """Maximum-likelihood alpha in [0, 1] (bounded Brent).

    The log-likelihood is concave in alpha, so the optimum is unique. With
    `check_grid`, a 1001-point grid search must agree to within 0.002, as a sanity check.
    """
    res = minimize_scalar(lambda a: -mixture_loglik(a, lp, lq), bounds=(0.0, 1.0),
                          method="bounded", options={"xatol": 1e-6})
    alpha = float(res.x)
    # Brent never evaluates the bounds exactly; snap when the optimum is on the boundary.
    for edge in (0.0, 1.0):
        if mixture_loglik(edge, lp, lq) >= mixture_loglik(alpha, lp, lq):
            alpha = edge
    if check_grid:
        grid = np.linspace(0, 1, 1001)
        g = grid[np.argmax([mixture_loglik(a, lp, lq) for a in grid])]
        if abs(g - alpha) > 0.002:
            raise RuntimeError(f"optimizer ({alpha:.4f}) and grid search ({g:.4f}) disagree")
    return alpha


def paper_resample_weights(paper_ids, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_sentences) weights: how often each sentence's paper is drawn in each resample."""
    papers, inv = np.unique(np.asarray(paper_ids), return_inverse=True)
    counts = rng.multinomial(len(papers), np.full(len(papers), 1 / len(papers)), size=n_boot)
    return counts[:, inv].astype(float)


def alpha_weighted(lp: np.ndarray, lq: np.ndarray, weights: np.ndarray, iters: int = 60) -> np.ndarray:
    """MLE alpha for many weighted copies of one sentence set at once (vectorized bisection).

    weights is (B, n_sentences); returns B alphas. With d = log Q(s) - log P(s), solves the
    first-order condition sum_s w_s (e^d_s - 1) / (1 + alpha (e^d_s - 1)) = 0,
    whose left side decreases in alpha (concave likelihood).
    """
    d = lq - lp
    # numerically stable form of (e^d - 1) / (1 + a (e^d - 1)) for both signs of d
    pos = d > 0
    e = np.exp(-np.abs(d))
    num = np.where(pos, 1.0 - e, e - 1.0)  # (e^d - 1) scaled by e^-d when d > 0
    den0 = np.where(pos, e, 1.0)  # 1 scaled by e^-d when d > 0

    def grad(a):  # a: (B,)
        return (weights * (num / (den0 + a[:, None] * num))).sum(1)

    B = weights.shape[0]
    lo, hi = np.zeros(B), np.ones(B)
    g0, g1 = grad(lo), grad(hi - 1e-12)
    for _ in range(iters):
        mid = (lo + hi) / 2
        up = grad(mid) > 0
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    alpha = (lo + hi) / 2
    alpha[g0 <= 0] = 0.0
    alpha[g1 >= 0] = 1.0
    return alpha


def bootstrap_alphas(lp: np.ndarray, lq: np.ndarray, paper_ids: list[str] | np.ndarray,
                     n_boot: int = 1000, rng: np.random.Generator | None = None) -> np.ndarray:
    """alpha_hat on `n_boot` bootstrap resamples of PAPERS (all sentences of a drawn paper go in)."""
    rng = rng or np.random.default_rng()
    return alpha_weighted(lp, lq, paper_resample_weights(paper_ids, n_boot, rng))


def bootstrap_ci(lp, lq, paper_ids, n_boot: int = 1000, level: float = 0.95,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Percentile bootstrap CI for alpha, resampling papers."""
    a = bootstrap_alphas(lp, lq, paper_ids, n_boot, rng)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(a, [tail, 100 - tail])
    return float(lo), float(hi)


def responsibilities(alpha: float, lp: np.ndarray, lq: np.ndarray) -> np.ndarray:
    """Posterior P(sentence from Q) = alpha Q / ((1 - alpha) P + alpha Q). Exploratory only."""
    with np.errstate(divide="ignore"):
        a, b = np.log(alpha), np.log1p(-alpha)
    num = a + lq
    return np.exp(num - np.logaddexp(b + lp, num))
