"""Distance between human and LLM abstracts in style-feature space.

Usage (from the repo root):
  python -m analysis.distance

Three analyses, each in two spaces (`space="z"` or `space="whitened"`):
  1. Drift: do the human abstracts move over the years towards the style of LLM abstracts?
  2. Centroids of every (year, source) group, and the distances between them.
  3. Paired similarity: how close is each paper's LLM abstract to its human abstract?

Everything is scaled by the reference set REF (human abstracts of 2020-2022, before ChatGPT). The scaling is fitted
once and stays fixed: it is not fitted again per year, per source or per bootstrap replicate.

Outputs (analysis/results/distance/):
  drift.csv, direction_agreement.csv, centroid_distances.csv, group_feature_means.csv,
  paired_similarity.csv, paired_similarity_by_year.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ABSTRACTS_CSV = Path("data/processed/abstracts_v1.csv")
PAPERS_CSV = Path("data/processed/papers_v1.csv")
RESULTS_DIR = Path("analysis/results/distance")

HUMAN, CLAUDE, GPT = "abstract_human", "abstract_synthetic_claude", "abstract_synthetic_gpt"  # as in analysis.build_frame
SOURCES = (HUMAN, CLAUDE, GPT)
REF_YEARS = (2020, 2021, 2022)
SPACES = ("z", "whitened")
N_BOOT = 2_000
SEED = 0
SMALL_N = 10  # groups below this size get small_n = True in the outputs (the ChatGPT groups have n = 5)

META = ("paper_id", "year", "title", "authors", "n_authors", "doi", "source", "model", "text")
# Every column except META and analysis.features.CONTROL_COLUMNS. Written out because importing analysis.features
# loads spaCy; load_abstracts() checks this list against the table, so it can't go stale unnoticed.
FEATURES = (
    "chars_per_word", "words_per_sentence", "flesch_reading_ease", "mtld", "stopword_share", "hapax_share",
    "overlap_1gram", "overlap_2gram", "top1000_share", "top10000_share",
    "pos_ADJ", "pos_ADP", "pos_ADV", "pos_AUX", "pos_CCONJ", "pos_DET", "pos_NOUN", "pos_NUM", "pos_PART", "pos_PRON",
    "pos_PROPN", "pos_PUNCT", "pos_SCONJ", "pos_VERB", "ent_ORG_per_100_words",
)


def load_abstracts(path: Path = ABSTRACTS_CSV) -> pd.DataFrame:
    """The abstracts table; built with analysis.build_frame if the CSV is missing."""
    from analysis.features import CONTROL_COLUMNS  # imports spaCy

    if path.exists():
        abstracts = pd.read_csv(path)
    else:
        from analysis.build_frame import build

        abstracts, papers = build()
        path.parent.mkdir(parents=True, exist_ok=True)
        abstracts.to_csv(path, index=False)
        papers.to_csv(path.with_name(PAPERS_CSV.name), index=False)
    expected = set(abstracts.columns) - set(META) - set(CONTROL_COLUMNS)
    if expected != set(FEATURES):
        raise SystemExit(f"distance.FEATURES is out of date; differs from the table in {expected ^ set(FEATURES)}")
    return abstracts


@dataclass(frozen=True)
class Prepared:
    """Every abstract in both spaces; rows line up with `meta`."""

    meta: pd.DataFrame  # paper_id, year, source
    z: np.ndarray
    whitened: np.ndarray
    features: tuple[str, ...]
    nan_filled: pd.DataFrame  # NaNs filled: one row per source, one column per feature
    eigenvalues: np.ndarray  # of the PCA on REF's z-scores, largest first

    def vectors(self, space: str) -> np.ndarray:
        if space not in SPACES:
            raise ValueError(f"space must be one of {SPACES}, not {space!r}")
        return self.z if space == "z" else self.whitened


def preprocess(abstracts: pd.DataFrame, features: tuple[str, ...] = FEATURES) -> Prepared:
    """Fill NaNs with REF's medians, z-score with REF's mean and SD, and whiten with a PCA fitted on REF's z-scores."""
    meta = abstracts[["paper_id", "year", "source"]].astype({"source": str}).reset_index(drop=True)
    if meta.duplicated(["paper_id", "source"]).any():
        raise ValueError("more than one abstract for a (paper_id, source)")
    values = abstracts[list(features)].reset_index(drop=True).astype(float)
    is_ref = (meta.source == HUMAN) & meta.year.isin(REF_YEARS)

    nan_filled = values.isna().groupby(meta.source).sum()
    values = values.fillna(values[is_ref].median())
    for source, count in nan_filled.sum(axis=1).items():
        print(f"NaNs filled with the REF median, {source}: {count}")

    ref = values[is_ref]
    if (ref.std() == 0).any():
        raise ValueError(f"no variation in REF: {list(ref.columns[ref.std() == 0])}")
    z = ((values - ref.mean()) / ref.std()).to_numpy()
    pca = PCA(whiten=True, svd_solver="full").fit(z[is_ref])  # all components, each divided by sqrt(eigenvalue)
    return Prepared(meta, z, pca.transform(z), tuple(features), nan_filled, pca.explained_variance_)


# ---------------------------------------------------------------------------------------------------------------------
# 1. Drift towards the LLM direction
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Paired:
    """One row per paper that has a human abstract; NaN where a paper has no abstract from that source."""

    paper_id: np.ndarray
    year: np.ndarray
    vectors: dict[str, np.ndarray]  # source -> (papers, dimensions)


def paired(prep: Prepared, space: str = "z") -> Paired:
    x = prep.vectors(space)
    human = prep.meta[prep.meta.source == HUMAN].sort_values("paper_id")
    position = pd.Index(human.paper_id)
    vectors = {}
    for source in SOURCES:
        rows = prep.meta.index[prep.meta.source == source]
        at = position.get_indexer(prep.meta.paper_id[rows])
        block = np.full((len(human), x.shape[1]), np.nan)
        block[at[at >= 0]] = x[rows[at >= 0]]
        vectors[source] = block
    return Paired(human.paper_id.to_numpy(), human.year.to_numpy(), vectors)


def direction_papers(pairs: Paired) -> dict[str, tuple[str, np.ndarray]]:
    """Direction name -> (LLM source, indices of the pre-ChatGPT papers that define it)."""
    pre = np.isin(pairs.year, REF_YEARS)
    has = {source: ~np.isnan(block[:, 0]) for source, block in pairs.vectors.items()}
    subset = np.flatnonzero(pre & has[GPT] & has[CLAUDE])
    return {
        "claude_full": (CLAUDE, np.flatnonzero(pre & has[CLAUDE])),
        "gpt_15": (GPT, subset),
        "claude_15": (CLAUDE, subset),
    }


def _pooled_drift(means: np.ndarray, counts: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Year means (..., years, dims) -> drift, the same shape. A REF year's reference leaves that year out."""
    is_ref = np.isin(years, REF_YEARS)
    if is_ref.sum() < 2:
        raise ValueError("leave-one-year-out needs at least two REF years")
    weighted = means * counts[:, None]
    total = weighted[..., is_ref, :].sum(axis=-2, keepdims=True)
    n_total = counts[is_ref].sum()
    leave_out = np.where(is_ref[:, None], weighted, 0.0)
    n_rest = n_total - np.where(is_ref, counts, 0)
    return means - (total - leave_out) / n_rest[:, None]


def yearly_drift(prep: Prepared, space: str = "z") -> pd.DataFrame:
    """drift_y = mean(human, y) - mean(human, REF), without y in REF. One row per year."""
    pairs = paired(prep, space)
    human = pairs.vectors[HUMAN]
    years = np.unique(pairs.year)
    means = np.stack([human[pairs.year == y].mean(axis=0) for y in years])
    counts = np.array([(pairs.year == y).sum() for y in years], dtype=float)
    return pd.DataFrame(_pooled_drift(means, counts, years), index=pd.Index(years, name="year"))


def _cos_proj(drift: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dot = (drift * direction).sum(axis=-1)
    norm_drift, norm_d = np.linalg.norm(drift, axis=-1), np.linalg.norm(direction, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dot / (norm_drift * norm_d), dot / norm_d**2


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def _ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return tuple(np.percentile(values, [2.5, 97.5], axis=0))


def sign_flip_directions(differences: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Null directions: (replicates, papers) signs x (papers, dims) differences, averaged over the papers."""
    return (signs[:, :, None] * differences).mean(axis=1)


def drift_analysis(prep: Prepared, space: str = "z", n_boot: int = N_BOOT, seed: int = SEED,
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The tables of drift.csv and direction_agreement.csv.

    Bootstrap: papers are resampled within each year (5 per year for the 15-paper directions). A set of papers is drawn
    only once per replicate, so a paper counts the same number of times wherever it appears: claude_full uses the same
    draw as the REF part of the drift, and claude_15 and gpt_15 use the same 15-paper draw.

    Null: every paper's difference gets a random sign, with the drift fixed at its observed value. A paper keeps its
    sign in all directions, so the pairs in direction_agreement.csv keep what they share by construction.
    """
    rng = np.random.default_rng(seed)
    pairs = paired(prep, space)
    human = pairs.vectors[HUMAN]
    years = np.unique(pairs.year)
    counts = np.array([(pairs.year == y).sum() for y in years], dtype=float)
    directions = direction_papers(pairs)
    differences = {name: pairs.vectors[source][papers] - human[papers] for name, (source, papers) in directions.items()}

    draws: dict[bytes, np.ndarray] = {}

    def draw(ids: np.ndarray) -> np.ndarray:
        """A resample (replicates, len(ids)) of these papers; the same one if they were already drawn."""
        key = ids.tobytes()
        if key not in draws:
            draws[key] = ids[rng.integers(len(ids), size=(n_boot, len(ids)))]
        return draws[key]

    year_ids = {y: np.flatnonzero(pairs.year == y) for y in years}
    point_drift = _pooled_drift(np.stack([human[year_ids[y]].mean(axis=0) for y in years]), counts, years)
    boot_drift = _pooled_drift(np.stack([human[draw(year_ids[y])].mean(axis=1) for y in years], axis=1), counts, years)

    point_d, boot_d, null_d = {}, {}, {}
    signs = rng.choice((-1.0, 1.0), size=(n_boot, len(pairs.year)))
    for name, (source, papers) in directions.items():
        vectors = pairs.vectors[source] - human
        point_d[name] = vectors[papers].mean(axis=0)
        by_year = [papers[pairs.year[papers] == y] for y in REF_YEARS]
        chosen = np.concatenate([draw(ids) for ids in by_year if len(ids)], axis=1)
        boot_d[name] = vectors[chosen].mean(axis=1)
        null_d[name] = sign_flip_directions(differences[name], signs[:, papers])

    rows = []
    for name, (_, papers) in directions.items():
        cos, proj = _cos_proj(point_drift, point_d[name])
        boot_cos, boot_proj = _cos_proj(boot_drift, boot_d[name][:, None, :])
        null_cos, null_proj = _cos_proj(point_drift, null_d[name][:, None, :])
        cos_lo, cos_hi = _ci(boot_cos)
        proj_lo, proj_hi = _ci(boot_proj)
        null_cos_lo, null_cos_hi = _ci(null_cos)
        null_proj_lo, null_proj_hi = _ci(null_proj)
        for i, year in enumerate(years):
            rows.append({
                "space": space, "direction": name, "n_papers_direction": len(papers), "year": year,
                "cos": cos[i], "cos_lo": cos_lo[i], "cos_hi": cos_hi[i],
                "proj": proj[i], "proj_lo": proj_lo[i], "proj_hi": proj_hi[i],
                "null_cos_lo": null_cos_lo[i], "null_cos_hi": null_cos_hi[i],
                "null_proj_lo": null_proj_lo[i], "null_proj_hi": null_proj_hi[i],
            })

    agreement = []
    for a, b in combinations(directions, 2):
        lo, hi = _ci(_cos(boot_d[a], boot_d[b]))
        null_lo, null_hi = _ci(_cos(null_d[a], null_d[b]))
        agreement.append({
            "space": space, "direction_a": a, "direction_b": b,
            "n_papers_a": len(directions[a][1]), "n_papers_b": len(directions[b][1]),
            "n_papers_shared": len(np.intersect1d(directions[a][1], directions[b][1])),
            "cos": _cos(point_d[a], point_d[b]), "cos_lo": lo, "cos_hi": hi, "null_cos_lo": null_lo, "null_cos_hi": null_hi,
        })
    return pd.DataFrame(rows), pd.DataFrame(agreement)


# ---------------------------------------------------------------------------------------------------------------------
# 2. Group centroids
# ---------------------------------------------------------------------------------------------------------------------


def group_means(prep: Prepared, space: str = "z") -> pd.DataFrame:
    """Centroid of each (year, source) group: n, small_n and one column per dimension.

    In the z space the columns are the features, and each value is the group's shift in REF SDs. In the whitened space
    they are the components (PC1 has the largest eigenvalue), since a whitened dimension is not a feature.
    """
    columns = list(prep.features) if space == "z" else [f"PC{i + 1}" for i in range(prep.whitened.shape[1])]
    table = pd.DataFrame(prep.vectors(space), columns=columns).assign(year=prep.meta.year, source=prep.meta.source)
    groups = table.groupby(["year", "source"])
    means = groups[columns].mean()
    means.insert(0, "n", groups.size())
    means.insert(1, "small_n", means.n < SMALL_N)
    return means.reset_index()


def centroid_distances(prep: Prepared, space: str = "z") -> pd.DataFrame:
    """Euclidean distance between all pairs of centroids (kind = "pair") and from each to REF's (kind = "to_ref").

    The distance to REF does not leave a year out: this table describes the groups, it does not measure drift.
    """
    means = group_means(prep, space)
    x = prep.vectors(space)
    is_ref = ((prep.meta.source == HUMAN) & prep.meta.year.isin(REF_YEARS)).to_numpy()
    centroid = means.drop(columns=["year", "source", "n", "small_n"]).to_numpy()
    ref_centroid = x[is_ref].mean(axis=0)

    def row(kind, i, year_b, source_b, n_b, distance):
        return {"space": space, "kind": kind, "year_a": means.year[i], "source_a": means.source[i], "n_a": means.n[i],
                "year_b": year_b, "source_b": source_b, "n_b": n_b, "distance": distance,
                "small_n": bool(means.n[i] < SMALL_N or n_b < SMALL_N)}

    rows = [row("to_ref", i, pd.NA, "REF_human", int(is_ref.sum()), np.linalg.norm(centroid[i] - ref_centroid))
            for i in range(len(means))]
    rows += [row("pair", i, means.year[j], means.source[j], means.n[j], np.linalg.norm(centroid[i] - centroid[j]))
             for i, j in combinations(range(len(means)), 2)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# 3. Paired similarity
# ---------------------------------------------------------------------------------------------------------------------

PAIRS = {"human_claude": (HUMAN, CLAUDE), "human_gpt": (HUMAN, GPT), "claude_gpt": (CLAUDE, GPT)}


def paired_similarity(prep: Prepared, space: str = "z") -> pd.DataFrame:
    """Per paper and pair of versions: cosine similarity and Euclidean distance. in_gpt_30: it has all three versions."""
    pairs = paired(prep, space)
    has_all = ~np.isnan(pairs.vectors[CLAUDE][:, 0]) & ~np.isnan(pairs.vectors[GPT][:, 0])
    frames = []
    for name, (a, b) in PAIRS.items():
        keep = ~np.isnan(pairs.vectors[a][:, 0]) & ~np.isnan(pairs.vectors[b][:, 0])
        va, vb = pairs.vectors[a][keep], pairs.vectors[b][keep]
        frames.append(pd.DataFrame({
            "space": space, "paper_id": pairs.paper_id[keep], "year": pairs.year[keep], "pair": name,
            "in_gpt_30": has_all[keep], "cosine": _cos(va, vb), "euclidean": np.linalg.norm(va - vb, axis=1),
        }))
    return pd.concat(frames, ignore_index=True)


def paired_similarity_by_year(prep: Prepared, space: str = "z", n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Yearly mean of each comparison, with a bootstrap 95% CI (papers resampled within year).

    human_claude_30 is human_claude on the 30 papers with a ChatGPT version, the like-for-like match of human_gpt.
    """
    per_paper = paired_similarity(prep, space)
    comparisons = {
        "human_claude": per_paper[per_paper.pair == "human_claude"],
        "human_claude_30": per_paper[(per_paper.pair == "human_claude") & per_paper.in_gpt_30],
        "human_gpt": per_paper[per_paper.pair == "human_gpt"],
        "claude_gpt": per_paper[per_paper.pair == "claude_gpt"],
    }
    rng = np.random.default_rng(seed)
    rows = []
    for name, table in comparisons.items():
        for year, group in table.groupby("year"):
            row = {"space": space, "comparison": name, "year": year, "n": len(group), "small_n": len(group) < SMALL_N}
            picks = rng.integers(len(group), size=(n_boot, len(group)))
            for metric in ("cosine", "euclidean"):
                values = group[metric].to_numpy()
                lo, hi = _ci(values[picks].mean(axis=1))
                row |= {metric: values.mean(), f"{metric}_lo": lo, f"{metric}_hi": hi}
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------


def run_all(prep: Prepared, out_dir: Path = RESULTS_DIR, n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Every analysis in both spaces; writes one CSV per table and returns the tables by name."""
    drift, agreement = zip(*(drift_analysis(prep, space, n_boot, seed) for space in SPACES))
    tables = {
        "drift": pd.concat(drift, ignore_index=True),
        "direction_agreement": pd.concat(agreement, ignore_index=True),
        "centroid_distances": pd.concat([centroid_distances(prep, s) for s in SPACES], ignore_index=True),
        "group_feature_means": group_means(prep, "z"),
        "paired_similarity": pd.concat([paired_similarity(prep, s) for s in SPACES], ignore_index=True),
        "paired_similarity_by_year": pd.concat([paired_similarity_by_year(prep, s, n_boot, seed) for s in SPACES],
                                               ignore_index=True),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--abstracts", type=Path, default=ABSTRACTS_CSV)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    run_all(preprocess(load_abstracts(args.abstracts)), args.out_dir)
    print(f"Tables -> {args.out_dir}")


if __name__ == "__main__":
    main()
