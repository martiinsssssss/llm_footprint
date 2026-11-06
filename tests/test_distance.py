"""Tests for analysis.distance, on toy data (no spaCy)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.distance import (
    CLAUDE,
    GPT,
    HUMAN,
    centroid_distances,
    drift_analysis,
    paired_similarity_by_year,
    preprocess,
    sign_flip_directions,
    yearly_drift,
)

FEATURES = ("a", "b", "c", "d")
YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
SHIFT = np.array([2.0, -1.0, 1.0, 0.5])  # what the toy LLM adds to a human abstract


def toy(n_per_year=400, post_copies_llm=False, seed=0, noise=1.0):
    """Human abstracts ~ N(0, 1) in every year; LLM = human + SHIFT + small noise; ChatGPT for 5 papers per year."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in YEARS:
        for i in range(n_per_year):
            human = rng.normal(0, noise, len(FEATURES))
            claude = human + SHIFT + rng.normal(0, 0.05, len(FEATURES))
            gpt = human + SHIFT + rng.normal(0, 0.05, len(FEATURES))
            written = claude if post_copies_llm and year >= 2023 else human
            paper = f"{year}_{i:06d}"
            rows.append((paper, year, HUMAN, *written))
            rows.append((paper, year, CLAUDE, *claude))
            if i < 5:
                rows.append((paper, year, GPT, *gpt))
    return pd.DataFrame(rows, columns=["paper_id", "year", "source", *FEATURES])


def table(frame, space="z", n_boot=200):
    return drift_analysis(preprocess(frame, FEATURES), space, n_boot=n_boot)[0]


@pytest.mark.parametrize("space", ["z", "whitened"])
def test_post_period_copies_of_the_llm_give_proj_and_cos_near_1(space) -> None:
    result = table(toy(post_copies_llm=True), space)
    post = result[(result.year >= 2023) & (result.direction == "claude_full")]
    assert post.proj.to_numpy() == pytest.approx(1, abs=0.1)
    assert post.cos.to_numpy() == pytest.approx(1, abs=0.05)
    pre = result[(result.year < 2023) & (result.direction == "claude_full")]
    assert np.abs(pre.proj).max() < 0.1


@pytest.mark.parametrize("space", ["z", "whitened"])
def test_unchanging_human_abstracts_give_proj_near_0(space) -> None:
    result = table(toy(), space)
    assert np.abs(result[result.direction == "claude_full"].proj).max() < 0.05


def test_leave_one_year_out_reference() -> None:
    frame = toy(n_per_year=50)
    frame.loc[(frame.source == HUMAN) & (frame.year == 2021), "a"] += 5.0
    prep = preprocess(frame, FEATURES)
    human = prep.meta.source == HUMAN
    z = pd.DataFrame(prep.z[human], index=prep.meta.year[human])
    drift = yearly_drift(prep)
    # By hand: without 2021 in its reference, and pooled over the papers of the other two REF years.
    assert drift.loc[2021].to_numpy() == pytest.approx((z.loc[2021].mean() - z.loc[[2020, 2022]].mean()).to_numpy())
    assert drift.loc[2020].to_numpy() == pytest.approx((z.loc[2020].mean() - z.loc[[2021, 2022]].mean()).to_numpy())
    # A post-ChatGPT year uses the whole of REF.
    assert drift.loc[2024].to_numpy() == pytest.approx((z.loc[2024].mean() - z.loc[[2020, 2021, 2022]].mean()).to_numpy())


def test_a_year_is_not_shrunk_towards_its_own_reference() -> None:
    frame = toy(n_per_year=50)
    prep = preprocess(frame, FEATURES)
    human = (prep.meta.source == HUMAN).to_numpy()
    before = yearly_drift(prep)
    z = prep.z.copy()
    z[human & (prep.meta.year == 2021).to_numpy(), 0] += 3.0  # move only 2021, in z units
    after = yearly_drift(type(prep)(prep.meta, z, prep.whitened, prep.features, prep.nan_filled, prep.eigenvalues))
    assert after.loc[2021, 0] - before.loc[2021, 0] == pytest.approx(3.0)  # all of it, not (1 - 1/3) of it


def test_sign_flip_null_is_centred_on_0() -> None:
    rng = np.random.default_rng(0)
    differences = rng.normal(0, 1, (15, 4)) + SHIFT  # a real direction, which the flips must remove
    drift = np.array([1.0, -0.5, 0.5, 0.25])
    signs = rng.choice((-1.0, 1.0), size=(20_000, 15))
    null = sign_flip_directions(differences, signs)
    assert np.abs(null.mean(axis=0)).max() < 0.05
    cos = (null @ drift) / (np.linalg.norm(null, axis=1) * np.linalg.norm(drift))
    assert cos.mean() == pytest.approx(0, abs=0.02)
    assert np.percentile(cos, 2.5) == pytest.approx(-np.percentile(cos, 97.5), abs=0.03)


def test_null_interval_covers_0_and_directions_have_the_right_papers() -> None:
    result = table(toy(n_per_year=100), n_boot=500)
    assert (result.null_cos_lo < 0).all() and (result.null_cos_hi > 0).all()
    assert result.groupby("direction").n_papers_direction.first().to_dict() == {
        "claude_15": 15, "claude_full": 300, "gpt_15": 15}


def test_nan_filling_uses_the_ref_median() -> None:
    frame = toy(n_per_year=20)
    frame.loc[frame.year >= 2023, "a"] += 100.0  # the median of all data would be far from REF's
    is_ref = (frame.source == HUMAN) & (frame.year <= 2022)
    ref_median = frame.loc[is_ref, "a"].median()
    missing = frame.index[(frame.source == CLAUDE) & (frame.year == 2024)][:3]
    frame.loc[missing, "a"] = np.nan

    prep = preprocess(frame, FEATURES)
    ref = frame.loc[is_ref, "a"]
    expected = (ref_median - ref.mean()) / ref.std()
    assert prep.z[missing, 0] == pytest.approx(expected)
    assert prep.nan_filled.loc[CLAUDE, "a"] == 3
    assert prep.nan_filled.loc[HUMAN, "a"] == 0


def test_nan_in_ref_is_filled_before_the_scaling_is_fitted() -> None:
    frame = toy(n_per_year=20)
    is_ref = (frame.source == HUMAN) & (frame.year <= 2022)
    frame.loc[frame.index[is_ref][0], "b"] = np.nan
    prep = preprocess(frame, FEATURES)
    assert not np.isnan(prep.z).any() and not np.isnan(prep.whitened).any()
    assert prep.z[is_ref.to_numpy()].mean(axis=0) == pytest.approx(np.zeros(4), abs=1e-9)


def test_scaling_is_fixed_by_ref_and_whitening_gives_identity_on_ref() -> None:
    frame = toy(n_per_year=60)
    frame[list(FEATURES)] = frame[list(FEATURES)].to_numpy() @ np.array(
        [[1, 0.8, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0, 1]])  # correlated features
    prep = preprocess(frame, FEATURES)
    is_ref = ((prep.meta.source == HUMAN) & prep.meta.year.isin((2020, 2021, 2022))).to_numpy()
    assert prep.z[is_ref].std(axis=0, ddof=1) == pytest.approx(np.ones(4))
    assert np.cov(prep.whitened[is_ref].T) == pytest.approx(np.eye(4), abs=1e-9)
    assert prep.whitened[is_ref].mean(axis=0) == pytest.approx(np.zeros(4), abs=1e-9)


def test_unknown_space_is_rejected() -> None:
    with pytest.raises(ValueError):
        drift_analysis(preprocess(toy(n_per_year=10), FEATURES), "raw", n_boot=5)


def test_bootstrap_is_reproducible_and_ci_brackets_the_estimate() -> None:
    prep = preprocess(toy(post_copies_llm=True, n_per_year=100), FEATURES)
    a, b = (drift_analysis(prep, "z", n_boot=200)[0] for _ in range(2))
    pd.testing.assert_frame_equal(a, b)
    full = a[a.direction == "claude_full"]
    assert ((full.proj_lo <= full.proj) & (full.proj <= full.proj_hi)).all()


def test_direction_agreement_rows() -> None:
    agreement = drift_analysis(preprocess(toy(n_per_year=100), FEATURES), "z", n_boot=100)[1]
    assert len(agreement) == 3
    row = agreement[(agreement.direction_a == "gpt_15") & (agreement.direction_b == "claude_15")].iloc[0]
    assert row.n_papers_shared == 15
    assert row.cos > 0.9  # both toy LLMs add the same SHIFT


def test_small_groups_are_flagged_and_ref_distance_is_present() -> None:
    distances = centroid_distances(preprocess(toy(n_per_year=30), FEATURES))
    to_ref = distances[distances.kind == "to_ref"]
    assert len(to_ref) == 18  # 6 years x 3 sources
    assert to_ref[to_ref.source_a == GPT].small_n.all()
    assert not to_ref[to_ref.source_a == HUMAN].small_n.any()


def test_paired_similarity_rows() -> None:
    by_year = paired_similarity_by_year(preprocess(toy(n_per_year=30), FEATURES), n_boot=50)
    assert set(by_year.comparison) == {"human_claude", "human_claude_30", "human_gpt", "claude_gpt"}
    assert by_year.groupby("comparison").n.sum().to_dict() == {
        "human_claude": 180, "human_claude_30": 30, "human_gpt": 30, "claude_gpt": 30}
    assert by_year[by_year.comparison == "human_gpt"].small_n.all()
