# Excess-vocabulary lower bound on LLM use in ISMIR abstracts (Kobak et al., 2025)

**Summary.** Kobak et al.'s ChatGPT-era style words (*leverages, intricate, seamlessly, nuances, struggle, introduces*…) become clearly more frequent in ISMIR abstracts after 2022. Under the simplest counterfactual (2023–2025 would look like the 2020–2022 average), this implies that **at least about 30% of 2023–2025 abstracts [95% CI 20–41%] were LLM-processed**: 10% in 2023, 31% in 2024 and 49% in 2025.

Two checks show that part of this is not LLM use:

- **Placebo year 2022 (pre-ChatGPT):** +12%.
- **Claude-written synthetic abstracts of the same papers:** +12% (topic effects).

Taking both into account, a cautious reading is **≥ ~15–20% in 2024 and ≥ ~35% in 2025**. The trend is consistent with EditLens, and per abstract the lexicon count correlates with EditLens (ρ = 0.39 in 2023–2025). With about 110 abstracts per year, any trend-based counterfactual is too noisy to use, so these numbers rest on the "no pre-trend" assumption.

Reproduce from the repo root (about 20 seconds):

```bash
pip install -r excess_vocabulary/requirements.txt && python -m spacy download en_core_web_sm
python excess_vocabulary/run_excess_analysis.py
```

All parameters are at the top of `run_excess_analysis.py`. Seed: 0.

## 1. Method

For a group of words, let p be the share of a year's abstracts that contain at least one of them, and q the counterfactual share expected without LLMs. If a share L of abstracts was LLM-processed and the rest follow q, then p = (1 − L)·q + L·r with r ≤ 1, so

**L ≥ (p − q) / (1 − q).**

This is a lower bound for any group of words that LLMs favor. It equals L only if every LLM-processed abstract contains the group. The bound is most informative for groups of words that are rare without LLMs (small q). It is not reported when q ≥ 0.9.

- **Data:** the 653 cleaned real abstracts (`abstracts_clean_v1.jsonl`), about 100–120 per year, and the 653 synthetic ones (`synthetic_abstracts_v1.jsonl`, from `claude-sonnet-5-5`).
- **Tokens:** lowercased alphabetic tokens (the reference's `\b\w\w+\b`, without digits). One count per abstract per word.
- **Counterfactual q:**
  - `flat`: the mean of 2020–2022.
  - `linear`: a least-squares trend through 2020–2022, with a slope of at least 0.
  - The reference's rule (`kobak`: 2022 + max(0, 2022 − 2021) × years ahead) is also computed for the lexicon arm.
- **CIs:** 1,000 bootstrap resamples of abstracts within each year (target and reference years).

**Arm A: Kobak lexicon (fixed in advance).**
- I recomputed the 379 style words in excess in PubMed 2024 from the reference repo, pinned at commit `53db991`, using their criterion and their manual style/content annotations. The list is cached in `lexicon/kobak_style_words_2024.csv`.
- Words are grouped by their PubMed 2024 frequency (< 0.01, 0.02, 0.05, 0.1).
- **Main group: PubMed frequency < 0.01.** It has 244 words, 127 of which occur in ISMIR abstracts, and q ≈ 0.38. Broader groups contain common words (*these, while, into*), which pushes q to 0.56–0.90 and makes the bound unstable.
- No word is selected using ISMIR data, so this arm has no selection bias.

**Arm B: data-driven (selection on our own data).**
- Papers are split 50/50 within each year. Words are selected on one half and the bound is measured on the other half; then the halves swap and the results are averaged.
- A word is selected if:
  - it is not a noun, proper noun, number or symbol, by spaCy POS;
  - it appears in at least 5 abstracts from 2023–2025;
  - its excess is at least 0.03 and its ratio p/q at least 1.5;
  - its baseline q is at most 0.1;
  - **its excess in the synthetic abstracts of the same papers is less than half its real excess.** This is the topic control: Claude's style does not change with the year, so a word that also rises in the synthetic abstracts rises because of topics.
- The 15 words with the largest gap are kept.

**Checks:**
- **Placebo:** the same analysis with 2022 as the target and 2020–2021 as reference. 2022 predates ChatGPT, so the bound should be about 0.
- **Synthetic control:** the same word groups measured on the synthetic abstracts. Their style is fixed by one model, so any excess there comes from topics.
- **Injection:** a known share of 2023–2025 real abstracts is replaced by their synthetic versions (20 repetitions per share).

## 2. Results (`results/lower_bound_by_year.csv`, `lower_bound_by_year.png`)

Lower bound [95% CI], with the `flat` counterfactual:

| | Placebo 2022 | 2023 | 2024 | 2025 | 2023–2025 pooled |
|---|---|---|---|---|---|
| **A. Lexicon, real** | +0.12 [−0.06, 0.28] | +0.10 [−0.07, 0.26] | **+0.31 [0.16, 0.46]** | **+0.49 [0.34, 0.65]** | **+0.30 [0.20, 0.41]** |
| A. Lexicon, synthetic control | +0.10 [−0.09, 0.27] | +0.05 [−0.13, 0.23] | +0.15 [−0.02, 0.33] | +0.15 [−0.05, 0.32] | +0.12 [−0.01, 0.24] |
| **B. Data-driven, real** | +0.09 [−0.27, 0.33] | +0.01 [−0.23, 0.22] | +0.10 [−0.11, 0.29] | **+0.34 [0.14, 0.53]** | +0.15 [0.00, 0.28] |
| B. Data-driven, synthetic control | −0.10 [−0.41, 0.13] | −0.18 [−0.43, 0.03] | +0.05 [−0.17, 0.24] | +0.12 [−0.13, 0.31] | −0.00 [−0.17, 0.14] |

Lexicon arm with the `linear` counterfactual: 2023 +0.03 [−0.31, 0.23], 2024 +0.23 [−0.21, 0.43], 2025 +0.41 [−0.19, 0.60]. The placebo is +0.12, and the synthetic control is −0.06 to −0.01.

- **Both arms rise in 2024–2025 and agree on the order of magnitude for 2025** (lexicon 0.49, data-driven 0.34).
- **2023 is not distinguishable from zero.** This fits the timing: ISMIR 2023 papers were written in early 2023, only months after ChatGPT came out.
- **The placebo is not zero** (+0.12 for the lexicon, CI including 0). Either the lexicon words were already rising slowly before ChatGPT (the earlier marker-word count showed 18% → 19% → 27% across 2020–2022), or it is noise. In either case, the `flat` bound probably overstates the LLM share by about that much.
- **The synthetic control rises by about 0.12–0.15 in 2024–2025 for the lexicon.** Claude writes lexicon words more often for newer papers, so part of the real rise is topic-driven, even though the lexicon contains only style words. The data-driven arm, which removes topic words explicitly, has a synthetic control near zero.
- **The linear counterfactual has the right shape but CIs of ±0.4–0.8.** With about 110 abstracts per year, a slope estimated from three points and extrapolated one to three years ahead is mostly noise. The reference's two-point rule is worse: q goes above 1 for some groups. Both work for PubMed with about a million abstracts per year, but not here.
- **Broader lexicon groups** (cutoff 0.02, 0.05) give larger bounds (2025: 0.65, 0.95), but they **fail the placebo** (cutoff 0.02: +0.26 [0.03, 0.45]), because q is high and the division by (1 − q) amplifies noise. Only the < 0.01 group is reliable.

### Which words drive it (`results/excess_words.csv`, `top_excess_words.png`)

- **Lexicon words** (PubMed frequency < 0.01) that are rare in ISMIR 2020–2022 and rise after it:

  | Word | 2020–2022 | 2023–2025 |
  |---|---|---|
  | *introduces* | 2.7% | 7.4% |
  | *struggle* | 0.9% | 5.2% |
  | *integrates* | 1.2% | 4.0% |
  | *nuances* | 1.5% | 3.7% |
  | *leverages* | 2.4% | 4.6% |
  | *leveraging* | 4.8% | 6.8% |
  | *seamlessly*, *intricate*, *fostering*, *refine* | ≈ 1% | 2.5–2.8% |

  None of them rise in the synthetic abstracts.
- **The largest style excesses overall:** *introduce* (+15.5 points), *our*, *across*, *within*, *diverse*, *novel*, *additionally*, *critical*, *comprehensive*. Some also rise in the synthetic abstracts, so they are partly topical (*introduce*, *existing*, *musical*). *within*, *diverse*, *novel* and *additionally* do not rise there.
- **Selected words (data-driven arm, `selected_words.csv`):**
  - fold A: *additionally, within, diverse, struggle, critical, current, introduces, reveal, enables*…
  - fold B: *despite, furthermore, diverse, comprehensive, critical, jointly, extensive, offer*…
  - The placebo selections are different and look generic (*due, known, moreover, promising, typically*).

## 3. Injection check (`results/injection.csv`)

Replacing a known share of 2023–2025 abstracts by their Claude-written versions (pooled bound, `flat`, mean of 20 repetitions):

| Injected share | 0 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| A. Lexicon | 0.30 | 0.30 | 0.28 | 0.28 | 0.25 |
| B. Data-driven | 0.15 | 0.17 | 0.16 | 0.19 | 0.40 |

- **The lexicon does not detect Claude-written text: the bound *drops*.** Swapping in Claude abstracts removes GPT-era words. This is the same mismatch found with the Liang estimator (Q from Claude lacks these words), seen from the other side. **The lexicon bound measures ChatGPT-style editing specifically.** It says nothing about Claude- or Gemini-style use, which makes it more conservative.
- **The data-driven arm responds only weakly up to 30%,** and clearly at 50%. The real GPT-era words still outrank Claude's words in the top 15 until Claude text dominates. Both behaviors are consistent with a *lower* bound, but they show that it can be far from tight.

## 4. Comparison with EditLens

- **Per year:** EditLens real means are 0.19, 0.19, 0.19, 0.26, 0.35 and 0.48. The bounds follow the same shape: flat until 2022, small in 2023, rising in 2024 and steeper in 2025.
- **Per abstract (exploratory, `lexicon_vs_editlens.csv`):** the number of lexicon words in an abstract correlates with its EditLens score, with Spearman ρ = 0.29 over all years and **0.39 for 2023–2025** (p < 1e-12). The two methods share no inputs (word counts vs a RoBERTa classifier), so this is independent supporting evidence. It is still no basis for judging individual papers.

## 5. Caveats

- **Sample size.** About 110 abstracts per year, against about 1M per year in PubMed. Only the flat counterfactual is stable. It assumes no pre-trend, and the placebo suggests a pre-trend or noise of about +0.1.
- **What is bounded.** The share of abstracts with LLM-*influenced* wording, not authorship, and not any individual paper. Light editing that changes no lexicon word is invisible to the method.
- **The lexicon comes from biomedicine (PubMed) and ChatGPT-era models.** It misses ISMIR-specific LLM habits and the styles of other models (see the injection check).
- **Topic effects.** Both the lexicon synthetic control (+0.12–0.15) and the placebo suggest discounting about 0.1. The data-driven arm controls topics explicitly but is noisier.
- **One synthetic generator.** The topic control depends on Claude's style being constant over the years, which holds by construction (one model and one prompt for every paper).

## 6. Differences from the reference (Kobak et al.)

| Reference | Here | Why |
|---|---|---|
| Counterfactual from the two years before the target, slope × 2 | `flat` (main) and `linear`; the reference rule is reported for the lexicon arm only | With about 110 abstracts per year, the two-point slope is noise (q > 1 for some groups) |
| Style/content annotation by hand | Lexicon arm: their annotations. Data-driven arm: POS filter plus the synthetic-abstract topic control | No manual annotation of our words |
| Bound reported as the gap Δ = p − q | (p − q)/(1 − q), with Δ also in the CSV | The (1 − q) correction matters when q ≈ 0.4 |
| Bootstrap: none, given the corpus size | Abstracts resampled within each year | Our sample is small |
| Words selected and measured on the same data | Data-driven arm cross-fitted (select on half, measure on the other half) | Avoids selection bias on small samples |
