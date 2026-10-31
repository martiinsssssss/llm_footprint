"""Linguistic and stylistic features of a text (after Fröhling & Zubiaga 2021).

Usage: python -m analysis.features data/processed/ismir_2020_2025_v1.jsonl --out data/processed/features_v1.csv
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from functools import cache
from pathlib import Path

import pandas as pd
import spacy
import textstat
from wordfreq import top_n_list

from parsing.models import read_jsonl

# Fixed tag and label lists, so every row has the same columns.
POS_TAGS = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
            "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X")
ENT_LABELS = ("CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY",
              "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME",
              "WORK_OF_ART")  # the labels of en_core_web_sm (checked in the tests)
TOP_N = (100, 1_000, 10_000)
MAX_NGRAM = 3
MTLD_THRESHOLD = 0.72  # standard value (McCarthy & Jarvis 2010)

_TOP_WORDS = top_n_list("en", max(TOP_N))
TOP_SETS = {n: set(_TOP_WORDS[:n]) for n in TOP_N}


@cache
def nlp() -> spacy.language.Language:
    return spacy.load("en_core_web_sm")


def features(text: str) -> dict[str, float]:
    """All features of one text. A word is any token that is not punctuation or whitespace."""
    doc = nlp()(text)
    tokens = [t for t in doc if not t.is_space]
    words = [t for t in tokens if not t.is_punct]
    if not words:
        raise ValueError("text has no words")
    lower = [t.lower_ for t in words]
    sentences = [[t.lower_ for t in s if not (t.is_punct or t.is_space)] for s in doc.sents]
    sentences = [s for s in sentences if s]
    n_words, n_syllables = len(words), textstat.syllable_count(text)
    counts = Counter(lower)
    pos = Counter(t.pos_ for t in tokens)
    ents = Counter(e.label_ for e in doc.ents)
    return {
        # Basic counts
        "n_chars": len(text),
        "n_syllables": n_syllables,
        "n_words": n_words,
        "n_sentences": len(sentences),
        "chars_per_word": sum(len(t.text) for t in words) / n_words,
        "syllables_per_word": n_syllables / n_words,
        "words_per_sentence": n_words / len(sentences),
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        # Lexical diversity
        "type_token_ratio": len(counts) / n_words,
        "mtld": mtld(lower),
        # Repetitiveness
        "stopword_share": sum(t.is_stop for t in words) / n_words,
        "hapax_share": sum(c == 1 for c in counts.values()) / len(counts),
        **{f"overlap_{n}gram": sentence_overlap(sentences, n) for n in range(1, MAX_NGRAM + 1)},
        # Common vocabulary
        **{f"top{n}_share": sum(w in TOP_SETS[n] for w in lower) / n_words for n in TOP_N},
        # Syntax and named entities
        **{f"pos_{tag}": pos[tag] / len(tokens) for tag in POS_TAGS},
        **{f"ent_{label}_per_100_words": 100 * ents[label] / n_words for label in ENT_LABELS},
    }


def mtld(words: list[str]) -> float:
    """Measure of Textual Lexical Diversity: mean of a forward and a backward pass."""
    return (_mtld_pass(words) + _mtld_pass(words[::-1])) / 2


def _mtld_pass(words: list[str]) -> float:
    factors, types, count = 0.0, set(), 0
    for word in words:
        types.add(word)
        count += 1
        if len(types) / count <= MTLD_THRESHOLD:
            factors, types, count = factors + 1, set(), 0
    if count:  # partial factor for the rest
        factors += (1 - len(types) / count) / (1 - MTLD_THRESHOLD)
    return len(words) / factors if factors else math.nan  # nan: no word ever repeats


def sentence_overlap(sentences: list[list[str]], n: int) -> float:
    """Mean Jaccard overlap of the word n-grams of consecutive sentences (nan if < 2 sentences)."""
    grams = [set(zip(*(s[i:] for i in range(n)))) for s in sentences]
    scores = [len(a & b) / len(a | b) for a, b in zip(grams, grams[1:]) if a | b]
    return sum(scores) / len(scores) if scores else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the features of every abstract to a CSV.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    papers = [paper for paper in read_jsonl(args.jsonl) if paper.abstract]
    rows = [
        {"paper_id": paper.paper_id, "year": paper.year, "source": "human", **features(paper.abstract)}
        for paper in papers
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{len(rows)} abstracts -> {args.out}")


if __name__ == "__main__":
    main()
