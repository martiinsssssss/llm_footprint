"""Tests for analysis.features."""

from __future__ import annotations

import math

import pytest

from analysis.features import ENT_LABELS, features, mtld, nlp, sentence_overlap


def test_entity_labels_match_the_model() -> None:
    assert set(ENT_LABELS) == set(nlp().get_pipe("ner").labels)


def test_mtld_small_case() -> None:
    # "a b a": TTR drops to 2/3 <= 0.72, one factor; "b" is a partial factor of 0.
    assert mtld(["a", "b", "a", "b"]) == 4


def test_mtld_no_repeats_is_nan() -> None:
    assert math.isnan(mtld(["a", "b", "c"]))


def test_sentence_overlap() -> None:
    assert sentence_overlap([["a", "b"], ["a", "c"]], 1) == pytest.approx(1 / 3)
    assert sentence_overlap([["a", "b"], ["a", "b"]], 2) == 1
    assert math.isnan(sentence_overlap([["a", "b"]], 1))


def test_features_on_short_text() -> None:
    f = features("The cat sat. The cat ran.")
    assert (f["n_words"], f["n_sentences"]) == (6, 2)
    assert f["words_per_sentence"] == 3
    assert f["type_token_ratio"] == pytest.approx(4 / 6)
    assert f["overlap_1gram"] == pytest.approx(2 / 4)  # {the, cat} / {the, cat, sat, ran}
    assert f["pos_PUNCT"] == pytest.approx(2 / 8)
    assert 0 < f["top100_share"] <= f["top1000_share"] <= f["top10000_share"] <= 1


def test_same_columns_for_every_text() -> None:
    assert features("We propose a model.").keys() == features("In 2021, Sony released a dataset in Tokyo.").keys()


def test_named_entities_are_counted() -> None:
    f = features("In 2021, Sony released a dataset in Tokyo.")
    assert f["ent_ORG_per_100_words"] > 0 and f["ent_GPE_per_100_words"] > 0


def test_empty_text_raises() -> None:
    with pytest.raises(ValueError):
        features(" . ")
