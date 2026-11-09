# Distributional LLM-usage estimate (Liang et al., 2024) on ISMIR abstracts

**Summary.** The estimator works on held-out mixtures: one vocabulary (verbs) meets the MAE ≤ 0.05 target, and two more come close. On the real abstracts, every variant except adjectives estimates a higher share of LLM-modified sentences after 2022. But the size of the rise depends heavily on the vocabulary, and some 2020–2022 estimates are not near zero. **The main limitation is Q**, the AI reference. It comes from a single recent model (`claude-sonnet-5-5`) that barely uses the ChatGPT-era words whose frequency doubles in real ISMIR abstracts after 2022 (§6). These numbers should be treated as a rough lower bound with wide uncertainty, not as a measurement.

Reproduce from the repo root (about 12 minutes on a laptop CPU):

```bash
pip install -r liang_estimate/requirements.txt && python -m spacy download en_core_web_sm
python liang_estimate/run_liang_analysis.py --stage validation   # ~11 min
python liang_estimate/run_liang_analysis.py --stage estimates    # ~1 min
```

All parameters are at the top of `run_liang_analysis.py`. Seed: 0.

## 1. Setup

| | Definition | Size |
|---|---|---|
| Human reference P | real abstracts, 2020–2022, one fold | 166 papers, ~1,190 sentences |
| AI reference Q | synthetic abstracts of the **same** papers | 166 papers, ~1,200 sentences |
| Targets | real abstracts per year: the other fold for 2020–2022, all papers for 2023–2025 | 99–122 papers, 717–890 sentences per year |

- **Folds:** the 2020–2022 papers are split 50/50 by paper, stratified by year, and **cross-fitted**. Each estimate is the mean of (fit on A, evaluate on B) and (fit on B, evaluate on A), so for 2020–2022 every paper is used, always out of fold.
- **Tokenization:** spaCy `en_core_web_sm` splits sentences and assigns POS tags. Words are lowercased token texts that start with a letter, without lemmatization. Sentences with fewer than 2 words are dropped.
- **Vocabulary:** words in at least 5 P sentences **and** at least 5 Q sentences.
- **Probabilities:** add-1 smoothing on binary occurrence, so p_w = (count + 1) / (n + 2).
- **Likelihood:** full Bernoulli, with presence and absence terms (§7).
- **α̂:** bounded Brent optimization of the mixture log-likelihood in log space, checked against a 1,001-point grid search.
- **95% CI:** percentile bootstrap with 1,000 resamples of **papers**. For cross-fitted estimates, each resample averages the two fits.

| Variant | Words | Vocabulary size (fold A / B) |
|---|---|---|
| `adj` | adjectives (as in the peer-review paper) | 136 / 141 |
| `adv` | adverbs | 30 / 35 |
| `verb` | verbs | 94 / 100 |
| `all` | all words except stopwords | 571 / 580 |
| `allstop` | all words, stopwords included (what the reference repo's tokenizer keeps) | 674 / 693 |

## 2. Validation (`results/validation.csv`, `validation_summary.csv`, `validation.png`)

**Mixtures:** 100 abstracts drawn from the held-out fold. A known share is replaced by the papers' synthetic versions, at α ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5}. There are 30 random mixtures per α in each fold direction, so 60 per α.

**True α:** the share of *sentences* that are synthetic, which is what the estimator targets. It is within 0.006 of the nominal share of papers.

| Variant | MAE | Mean α̂ at true α = 0 | Bias at α = 0.5 | 95% CI coverage | Mean CI width |
|---|---|---|---|---|---|
| `verb` | **0.043** ✅ | 0.033 | −0.004 | 100% | 0.26 |
| `all` | 0.051 | 0.051 | −0.048 | 81% | 0.16 |
| `allstop` | 0.052 | 0.071 | −0.004 | 63% | 0.14 |
| `adj` | 0.065 | 0.062 | −0.042 | 87% | 0.24 |
| `adv` | 0.078 | 0.072 | +0.003 | 98% | 0.39 |

- **Only `verb` meets the MAE ≤ 0.05 target**, and `all` and `allstop` miss it narrowly. Following the plan, I tried other vocabulary settings before running the real estimates: a frequency threshold of 5, 10, 20 or 40 sentences, and smoothing k of 0, 1 or 5 (20 mixtures per α, point estimates only).
  - A higher threshold made every variant worse. For example, `all` goes from MAE 0.049 at threshold 5 to 0.092 at 20. Fewer words add more noise than they remove bias.
  - k = 5 removes most of the bias at α = 0, but adds a similar negative bias at α = 0.5.
  - **The defaults (threshold 5, k = 1) were kept.** The sweep used the same validation folds, so it slightly flatters these settings.
- **There is a consistent upward bias at α = 0 (+0.03 to +0.07).** Held-out *human* abstracts look partly AI to the model. My interpretation is that held-out papers bring topics and words that neither P nor Q saw. A sentence that fits P poorly is partly absorbed by Q. New papers in 2023–2025 bring the same kind of novelty, and more of it (newer topics), so **expect a similar or larger baseline in the real estimates**.
- **Coverage is too low for `allstop` and `all` at small α** (7% and 58% at α = 0). The bootstrap captures sampling noise but not this bias.

## 3. Real estimates (`results/alpha_by_year.csv`, `alpha_by_year.png`)

α̂ [95% CI]. Only `verb` passed validation; the other variants are reported for completeness.

| Year | `verb` ✅ | `all` | `allstop` | `adj` | `adv` |
|---|---|---|---|---|---|
| 2020 | 0.08 [0.00, 0.22] | 0.02 [0.00, 0.07] | 0.04 [0.01, 0.07] | 0.01 [0.00, 0.09] | 0.00 [0.00, 0.19] |
| 2021 | 0.00 [0.00, 0.11] | 0.11 [0.06, 0.17] | 0.12 [0.08, 0.16] | 0.20 [0.08, 0.32] | 0.01 [0.00, 0.23] |
| 2022 | 0.01 [0.00, 0.12] | 0.03 [0.00, 0.08] | 0.07 [0.03, 0.10] | 0.05 [0.00, 0.13] | 0.21 [0.06, 0.42] |
| 2023 | 0.09 [0.00, 0.21] | 0.10 [0.05, 0.16] | 0.11 [0.07, 0.15] | 0.02 [0.00, 0.10] | 0.05 [0.00, 0.24] |
| 2024 | **0.31 [0.19, 0.42]** | 0.09 [0.05, 0.14] | 0.12 [0.09, 0.16] | 0.04 [0.00, 0.14] | 0.18 [0.01, 0.38] |
| 2025 | **0.27 [0.16, 0.40]** | 0.09 [0.05, 0.14] | 0.17 [0.12, 0.22] | 0.10 [0.00, 0.21] | 0.12 [0.01, 0.31] |
| **Mean 2020–22 → 2023–25** | 0.03 → 0.22 | 0.05 → 0.10 | 0.07 → 0.13 | 0.09 → 0.06 | 0.07 → 0.12 |

- **`verb`, the only variant that passes validation, shows the clearest rise.** It sits near zero in 2021–2022, then estimates about 0.3 in 2024–2025, with CIs that exclude the 2021–2022 values. This matches the timing of the EditLens rise, which starts in 2023 and is strongest in 2024–2025.
- **`all` and `allstop` rise only about 0.05**, comparable to the validation bias at α = 0. Taken alone, they would not be convincing.
- **The near-zero check for 2020–2022 fails in places.** Examples are `adj` 2021 (0.20), `all` 2021 (0.11), `adv` 2022 (0.21) and `allstop` in every year (0.04–0.12, CIs excluding 0). ChatGPT did not exist then, so these values are **bias**. Most of it comes from one fold direction (`all` 2021: 0.02 with fit A, 0.20 with fit B). With ~1,200 reference sentences, the fitted distributions depend on which papers end up in the fold.
- **`adj`, the vocabulary of the original peer-review study, shows no rise** (0.09 → 0.06). In this corpus, the adjective differences between P and Q are mostly not the ones that changed after 2022 (§5, §6).

## 4. Sensitivity (`results/sensitivity.csv`)

**Q per year (topic drift).** For each year, P stays as in the main analysis, while Q is built from the synthetic abstracts of *that year's* papers. It is cross-fitted within the year, so Q never contains the target papers.

| Variant | Main analysis: 2020–22 → 2023–25 | Same-year Q: 2020–22 → 2023–25 | Same-year Q, 2025 [CI] |
|---|---|---|---|
| `verb` | 0.03 → 0.22 | 0.01 → 0.09 | 0.18 [0.06, 0.33] |
| `all` | 0.05 → 0.10 | 0.01 → 0.09 | 0.12 [0.06, 0.18] |
| `allstop` | 0.07 → 0.13 | 0.02 → 0.10 | 0.15 [0.10, 0.21] |
| `adj` | 0.09 → 0.06 | 0.05 → 0.09 | 0.13 [0.03, 0.25] |
| `adv` | 0.07 → 0.06 | 0.07 → 0.06 | 0.06 [0.00, 0.18] |

- **The 2020–2022 baseline gets closer to zero with a same-year Q** (`all` 0.01, `allstop` 0.02, `verb` 0.01). This supports the idea that part of the bias is topic novelty.
- **A rise in 2025 remains for `verb`, `all`, `allstop` and `adj`**, at roughly 0.12–0.18 with CIs above zero.
- **The `verb` estimate for 2024 drops from 0.31 to 0.04.** Part of the main `verb` signal is therefore topic-related, or just noise: with Q fitted on about 50 papers, the verb vocabulary shrinks to about 31 words.
- **This check is weaker than the main analysis:** Q comes from about 50 papers per fold, so vocabularies are 2–4 times smaller.

**Generator dependence.** This check could not be run: every synthetic abstract comes from `claude-sonnet-5-5`.

## 5. Top discriminative words (`results/top_words.csv`)

Fitted on all 2020–2022 papers. In parentheses: sentences containing the word in P / Q (2,386 / 2,414 sentences).

- **More frequent in Q (Claude):**
  - verbs: *built* (6/42), *evaluated* (9/41), *combines* (5/23), *improves* (12/49), *released* (5/19), *relies* (5/17), *outperforms* (28/69), *remains* (25/64)
  - adverbs: *largely* (8/39), *rather* (13/44), *jointly*, *typically*, *particularly*
  - adjectives: *random*, *wise* (as in "frame-wise"), *best*, *compact*, *prior*, *noisy*, *limited*
- **More frequent in P (human 2020–2022):**
  - verbs: *explore* (36/5), *demonstrate* (55/11), *proposed* (102/24), *achieve*, *provide*, *focus*
  - adverbs: *however* (92/16), *thus*, *finally*, *therefore*
  - adjectives: *different* (126/19), *important* (40/7), *previous*, *recent*, *novel* (67/19), *challenging*, *promising*
- **"delve" does not appear.** Q's style is terse and factual ("X combines A and B; it outperforms C"). The words that separate P from Q are mostly the human paper-writing template ("in this paper we propose a novel…", "we demonstrate…"). The estimator therefore partly measures *movement away from that template*, rather than movement *toward* a particular LLM style.

## 6. Why Q is the main limitation: ChatGPT-era words

These words are known to increase in scientific writing after ChatGPT. The table shows the share of abstracts containing at least one of: *delve, intricate, pivotal, showcase, leverage, comprehensive, notably, crucial, seamless(ly), realm, underscore, meticulous, nuanced, landscape, enhance, robust* (and inflections). This was computed separately, not by the scripts.

| | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| Real abstracts | 18% | 19% | 27% | 30% | 40% | 41% |
| Synthetic abstracts (Claude) | 10% | 9% | 17% | 12% | 8% | 14% |

The real abstracts more than double their use of these words (most often *leverage*, *comprehensive*, *enhance*, *robust*, *crucial*, *underscore*). The synthetic abstracts barely use them. **An estimator whose Q comes from this generator cannot attribute those words to AI**, so the real α̂ is probably an underestimate of ChatGPT-style editing. It also explains why the adjective variant stays flat: *comprehensive*, *crucial* and *robust* are in the vocabulary, but Q does not favor them. EditLens observation 4 (real 2025 scores above the synthetic ones) points the same way.

## 7. Comparison with EditLens

- **Per year** (`alpha_by_year.png`, lower panel): EditLens real means are 0.19, 0.19, 0.19, 0.26, 0.35, 0.48. Only `verb` follows a similar shape (flat, then up from 2024). The other variants are flatter.
- **Per paper (exploratory, `responsibility_vs_editlens.csv`):** each paper's mean posterior responsibility r(s), compared with its EditLens score. The Spearman correlation is weakly positive for most variants:

  | Variant | ρ, all years | ρ, 2023–2025 |
  |---|---|---|
  | `verb` | 0.35 | 0.22 |
  | `adj` | 0.15 | 0.31 |
  | `allstop` | 0.21 | 0.18 |
  | `adv` | 0.18 | 0.16 |
  | `all` | 0.14 | −0.04 |

  The two methods partly agree on *which* abstracts look AI-like. The method is not meant for per-document claims, so these numbers should not be used to rank or flag papers.

## 8. Differences from the plan and from the reference code

- **The likelihood is the same as the reference repo's** (`src/MLE.py`): presence and absence terms over the vocabulary. The repo's other choices, and what this implementation does instead:

  | Reference repo | This implementation | Why |
  |---|---|---|
  | No smoothing | Add-k, k = 1 (k = 0 reproduces the repo) | As specified in the plan |
  | Vocabulary threshold: ≥ 5 human and ≥ 3 AI sentences | 5 and 5, both configurable | As specified in the plan |
  | Reports the **midpoint of the bootstrap CI** | Reports the MLE on the full sample | The MLE is the estimate itself; the CI midpoint shifts with skew |
  | Resamples sentences | Resamples papers | Sentences of one abstract are correlated |
  | Regex tokenizer on all words, no POS filter | spaCy POS variants | The `allstop` variant matches the repo |

- **Cross-fitting was used throughout**, as the plan suggested.
- **The validation gate passed with one variant.** Before running the real estimates, I tried other vocabulary settings (§2), but none passed with more variants. All variants are reported, as requested; each row in `alpha_by_year.csv` carries a `passed_validation` flag.
- **Not in the plan:** the ChatGPT-era word count in §6, and the extra `allstop` variant.
- **Corpus definitions were not changed.**

## 9. Caveats

- **Small samples.** About 700–900 sentences per year, and reference distributions from about 1,200 sentences per fold. CIs are wide (0.14–0.39 at the 100-abstract scale), and the estimates for one year depend on how the folds are split.
- **α̂ depends on Q.** One generator and one prompt ("write an abstract from the full paper"). §6 shows that Q's style differs from the style that grew in ISMIR abstracts after 2022.
- **α is the share of sentences that were *substantially LLM-modified*.** It says nothing about who wrote a paper, or about any individual paper.
- **Style drift unrelated to LLMs biases α̂ upward.** New topics, new authors and changing conventions all count. Validation shows a +0.03–0.07 baseline even within 2020–2022, and the same-year-Q check shows that part of the main signal is topic-related.

## 10. Suggested next step

The most useful improvement is a **second Q** built from a ChatGPT-era model (for example GPT-4o or GPT-4o-mini) with a "polish this abstract" prompt applied to the 2020–2022 real abstracts. This mirrors how Liang et al. built their AI corpus, and is closer to what ISMIR authors most likely did. The scripts need no changes: point `SYNTH` at the new file. This adds a new corpus, so it is a decision for the group.
