# AI-detection observations on ISMIR abstracts (2020–2025)

**Reproducing.** Every table and test below, plus the figures in `results/figures/`, comes from one script. It needs the six score files (`{editlens,fastdetectgpt,binoculars}_{abstracts,synthetic}_v1.jsonl`, in the shared Drive) in `../data` and takes about 10 seconds:

```bash
python -m ai_detection.analyze            # writes ai_detection/results/*.csv and results/figures/*.png
```

Figures:
- `editlens_by_year.png`: mean score (95% CI), share above 0.5, and the paired comparison, per year.
- `editlens_distribution_by_year.png`: every abstract's score, by year.
- `editlens_buckets_by_year.png`: EditLens's predicted bucket, by year.
- `positive_control_roc.png`: ROC curves of all three detectors, synthetic vs real 2020–2022 abstracts.

## Data

- **Real abstracts:** `abstracts_clean_v1.jsonl`. These are 653 ISMIR abstracts from 2020–2025, taken from the parser output (dataset v1) and cleaned with `ai_detection/cleaning.py`, which removes footnotes and broken URLs. Two papers are dropped because their abstract field holds something else: `2023_000014` (a figure caption) and `2024_000108` (the abstract plus part of the introduction).
- **Synthetic abstracts:** `synthetic_abstracts_v1.jsonl`, made with `ai_detection/synthetic.py`. There is one per real paper, written by `claude-sonnet-5-5` from the full paper with the prompt "Write an ISMIR-style abstract of about N words…". N is the length of the real abstract. The same two papers are dropped, so both sets have the same 653 paper IDs and every real abstract has a synthetic pair.
- **How the synthetic set was generated** (branch `feature/analysis`: `analysis/generate.py` and `.claude/agents/synthetic-abstract.md`):
  - A Claude Sonnet agent received the prompt and the paper's title and section texts, **with the original abstract removed**.
  - It was told to write only from that text: no web search, nothing remembered about the paper, and no other abstracts.
  - N is the word count of the original GROBID abstract, so length does not give the synthetic abstract away.
  - In 8 of the 653 outputs the agent added a markdown `**Abstract**` header or a word-count note. `synthetic.py` removes these; no other text is changed.
- **Role of the synthetic set:** it is a positive control. Every synthetic abstract is fully AI-written, on the same paper and at about the same length as its real pair.

## Detectors

| Detector | Type | Score | AI direction | Status |
|---|---|---|---|---|
| EditLens (`pangram/editlens_roberta-large`) | Trained classifier, 4 buckets | 0–1, the expected bucket divided by 3 | higher = more AI | done |
| Fast-DetectGPT (GPT-Neo-2.7B as sampling and scoring model, float32, Colab T4) | Zero-shot | analytic criterion | higher = more AI | done |
| Binoculars (falcon-7b / falcon-7b-instruct, 4-bit, float16, Colab T4) | Zero-shot | log PPL / log X-PPL | **lower** = more AI | done |

## EditLens

Mean scores per year. The last column is the share of paper pairs in which the synthetic abstract scores higher than the real one.

| Year | n | Real mean | Real median | Synthetic mean | Synthetic median | Synthetic > real |
|---|---|---|---|---|---|---|
| 2020 | 115 | 0.187 | 0.147 | 0.426 | 0.378 | 83% |
| 2021 | 104 | 0.188 | 0.139 | 0.407 | 0.373 | 87% |
| 2022 | 113 | 0.193 | 0.148 | 0.375 | 0.329 | 82% |
| 2023 | 100 | 0.259 | 0.242 | 0.418 | 0.384 | 75% |
| 2024 | 122 | 0.349 | 0.319 | 0.416 | 0.371 | 62% |
| 2025 | 99 | 0.476 | 0.447 | 0.411 | 0.381 | 41% |

| Share of abstracts | Real 2020–2022 | Real 2024–2025 | Synthetic (all years) |
|---|---|---|---|
| score > 0.5 | 5.7% | 29.4% | 23.0% |
| score > 0.8 | 0.6% | 11.3% | 5.5% |

| Predicted bucket | 0 (human) | 1 | 2 | 3 (fully AI) |
|---|---|---|---|---|
| Real | 353 | 240 | 1 | 59 |
| Synthetic | 95 | 471 | 0 | 87 |

AUC for separating synthetic abstracts (2023 papers left out) from real 2020–2022 abstracts: **0.84**.

### Observations

1. **Real abstracts rise steadily after ChatGPT.** Scores are flat at about 0.19 for 2020–2022, then rise to 0.26 (2023), 0.35 (2024) and 0.48 (2025). The share above 0.5 grows from 5.7% in 2020–2022 to 29.4% in 2024–2025.
2. **Changes in paper topics do not explain the rise.** The synthetic abstracts stay flat at about 0.41 in every year, although each is written from a paper of that year. If newer topics (for example more papers about LLMs) raised the scores by themselves, the synthetic scores would rise too.
3. **EditLens underestimates fully AI-written abstracts from this generator.** Synthetic abstracts have a median of only 0.38, and 72% fall in bucket 1 rather than bucket 3. A likely reason is that they summarize the real paper and reuse its content and terminology, which EditLens may read as human text with light AI editing. EditLens scores therefore work for comparing years, but should not be read as an "X% AI" measure.
4. **Real 2025 abstracts score higher than the synthetic ones** (0.476 vs 0.411; the synthetic abstract is higher in only 41% of 2025 pairs). This does not mean they are more AI-written than fully AI-written text. One hypothesis, not tested: the models 2024–2025 authors used, such as the GPT-4o generation, are closer to EditLens's training data than `claude-sonnet-5-5`, the newer model behind the synthetic set. The synthetic set is a valid positive control, but its generator is newer than the models 2024–2025 authors would have used.
5. **Bucket 2 is almost never predicted** (1 of 1,306 abstracts). In practice EditLens separates these abstracts into "human", "lightly edited" and "fully AI".

## Binoculars

Lower = more AI. The official thresholds are 0.9015 (accuracy) and 0.8536 (low false-positive rate). Under 4-bit quantization they are only approximate.

| Year | n | Real mean | Real median | Real < 0.90 | Synthetic mean | Synthetic median | Synthetic < 0.90 | Synthetic < real |
|---|---|---|---|---|---|---|---|---|
| 2020 | 115 | 0.989 | 0.990 | 4.3% | 1.041 | 1.041 | 0% | 18% |
| 2021 | 104 | 0.995 | 0.994 | 2.9% | 1.039 | 1.039 | 0% | 21% |
| 2022 | 113 | 1.004 | 1.006 | 3.5% | 1.041 | 1.040 | 0% | 24% |
| 2023 | 100 | 0.995 | 0.998 | 4.0% | 1.049 | 1.048 | 0% | 22% |
| 2024 | 122 | 0.996 | 0.995 | 1.6% | 1.051 | 1.047 | 0% | 14% |
| 2025 | 99 | 1.001 | 1.005 | 3.0% | 1.054 | 1.051 | 0% | 19% |

AUC for separating synthetic from real 2020–2022 abstracts: **0.23**. Below 0.5 means the detector ranks them the wrong way round.

### Observations

1. **No trend in the real abstracts.** Means stay between 0.989 and 1.004 in every year, and the share below 0.90 stays between 1.6% and 4.3%. The most AI-like tail (10th percentile) does not move either.
2. **The positive control fails, and fails inverted.** Not one synthetic abstract falls below 0.90. In about 80% of pairs, Binoculars rates the AI-written abstract as *more human* than the real one. The perplexity (log PPL) of the synthetic abstracts is higher than that of the real ones (3.42 vs 3.00): the generator's abstracts are less predictable to Falcon-7B than the human ones.
3. **The setup is not to blame.** On 60 paper pairs, an unquantized float32 pair from another model family (Qwen2.5-0.5B base + Instruct, run locally) shows the same inversion: the synthetic abstract scores lower in only 10% of pairs, against 18% with Falcon in 4-bit on the same 60. The same Qwen pair correctly flags a generic AI-written paragraph (0.74). The implementation also matches the official code exactly in a test with small models.
4. **Interpretation.** Binoculars detects text that is *more predictable* than an LLM would expect. The synthetic abstracts are dense summaries of a specific paper, full of its method names, datasets and numbers, written by a recent frontier model. Text like this is not predictable to a 2023 7B model. With this generator and prompt, Binoculars cannot detect AI-written abstracts. Its flat curve for the real abstracts is therefore **not evidence of no AI use**.

## Fast-DetectGPT

Higher = more AI. The raw criterion is reported, with no probability calibration. All scores were computed in one Colab run (the script does not record the dtype in its output; the run used float32).

| Year | n | Real mean | Real median | Synthetic mean | Synthetic median | Synthetic > real |
|---|---|---|---|---|---|---|
| 2020 | 115 | −0.044 | −0.060 | −1.202 | −1.178 | 17% |
| 2021 | 104 | −0.248 | −0.305 | −1.303 | −1.131 | 13% |
| 2022 | 113 | −0.406 | −0.497 | −1.439 | −1.483 | 20% |
| 2023 | 100 | −0.343 | −0.354 | −1.624 | −1.614 | 14% |
| 2024 | 122 | −0.478 | −0.442 | −1.678 | −1.630 | 12% |
| 2025 | 99 | −0.516 | −0.504 | −1.882 | −1.718 | 13% |

AUC for separating synthetic from real 2020–2022 abstracts: **0.19** (inverted, as with Binoculars).

### Observations

1. **The positive control fails, inverted.** Synthetic abstracts score about 1.2 points *lower* (more human) than real ones in every year, and are lower than their own real pair in 83–88% of pairs. Only 0.3% of synthetic abstracts exceed the 90th percentile of the real 2020–2022 scores.
2. **Real scores drift down over time (toward "more human").** The mean falls from −0.04 (2020) to −0.52 (2025). The synthetic abstracts drift down by about the same amount, from −1.20 to −1.88, so the drift is not about who wrote the text. The most likely cause is the age of the scoring model. GPT-Neo-2.7B was trained on the Pile (data up to 2020), so later papers use terms and methods it has never seen (diffusion models, LLMs, newer datasets), which makes them less predictable to it. **This is a confound for any year comparison with an old scoring model.**
3. **No sign of an AI-like tail after 2023.** About 10% of real abstracts exceed the 2020–2022 90th percentile in every year (5–10% from 2023 on).

## Agreement between detectors

Spearman rank correlation of the scores per paper. Binoculars is negated, so all three point the same way (higher = more AI):

| | Real abstracts | Synthetic abstracts |
|---|---|---|
| Fast-DetectGPT vs Binoculars | 0.81 | 0.73 |
| EditLens vs Fast-DetectGPT | 0.01 | 0.04 |
| EditLens vs Binoculars | 0.06 | 0.02 |

Per-year means of the real abstracts:

| Year | EditLens ↑ | Fast-DetectGPT ↑ | Binoculars ↓ |
|---|---|---|---|
| 2020 | 0.187 | −0.044 | 0.989 |
| 2021 | 0.188 | −0.248 | 0.995 |
| 2022 | 0.193 | −0.406 | 1.004 |
| 2023 | 0.259 | −0.343 | 0.995 |
| 2024 | 0.349 | −0.478 | 0.996 |
| 2025 | 0.476 | −0.516 | 1.001 |

### Observations

1. **The two zero-shot detectors measure the same thing; EditLens measures something else.** Fast-DetectGPT and Binoculars agree strongly (ρ = 0.81), since both score how predictable the text is to an LLM. EditLens is uncorrelated with both (ρ ≈ 0.01–0.06). The 59 real abstracts in EditLens's "fully AI" bucket (27 from 2025, 17 from 2024) look no different from the rest to the zero-shot detectors.
2. **Only EditLens passes the positive control** (AUC 0.84 vs 0.23 and 0.19), and only EditLens shows a rise after 2022. With this generator and prompt, the zero-shot detectors do not detect AI-written abstracts, so their flat or falling curves say nothing about AI use in ISMIR abstracts.
3. **Caveats for the synthetic control.**
   - It comes from a single recent model (`claude-sonnet-5-5`) and a single task: writing an abstract *from the full paper*. Authors in 2023–2025 more likely used ChatGPT-era models (GPT-3.5, GPT-4, GPT-4o), mostly to *polish their own draft*, not to write from scratch.
   - Zero-shot detectors are reported to work much better on older or smaller generators, so they may still catch that kind of text.
   - A fairer control would add synthetic sets from (a) an older or open model (e.g. GPT-4o-mini, Llama-3-8B-Instruct) and (b) a "polish this human abstract" prompt, which mimics editing.
4. **The year comparison needs a recent scoring model.** For zero-shot detectors, the downward drift of the synthetic Fast-DetectGPT scores shows that an old scoring model penalizes newer topics. A scoring model trained on data after 2025 (or the Llama-3 pair the Fast-DetectGPT README recommends) would reduce this confound, but would not remove it.

## Significance tests (`results/tests.csv`, `results/positive_control.csv`)

**Year groups (shared with the rest of the project):**
- Before ChatGPT: 2020–2022. After: 2024–2025.
- 2023 is a transition year (papers submitted a few months after ChatGPT's release) and is left out of every two-group comparison, including the positive control.
- Per-year results still show 2023.

**Tests:**
- Each detector is oriented so that higher = more AI.
- Mann-Whitney U tests are two-sided. AUC = probability that a random abstract from the first group scores as more AI than one from the second (0.5 = no difference). CIs come from 2,000 bootstrap resamples.
- The per-year tests compare every year with the 2020–2022 reference years (a reference year with the other two) and are Holm-corrected over the six years within each detector.

| | EditLens | Fast-DetectGPT | Binoculars |
|---|---|---|---|
| Positive control: AUC synthetic vs real 2020–2022 [95% CI] | **0.84 [0.81, 0.87]** | 0.19 [0.16, 0.22] | 0.23 [0.20, 0.27] |
| Real 2024–2025 vs 2020–2022: AUC, p | **0.75, p = 6·10⁻²⁴** | 0.43, p = 0.007 (toward *more human*) | 0.49, p = 0.65 |
| Synthetic 2024–2025 vs 2020–2022 (topic control): AUC, p | 0.54, p = 0.10 | 0.37, p = 2·10⁻⁷ | 0.42, p = 0.003 |
| Real 2020 / 2021 / 2022 vs the other reference years (Holm p) | 1.00 / 1.00 / 1.00 | **0.03** / 0.79 / 0.06 | 0.15 / 1.00 / 0.08 |
| Real 2023 / 2024 / 2025 vs 2020–2022: AUC (Holm p) | **0.61 (0.004) / 0.70 (3·10⁻¹⁰) / 0.82 (3·10⁻²¹)** | 0.47 / 0.43 / 0.43 (all ≥ 0.12) | 0.50 / 0.51 / 0.47 (all 1.00) |

The trend over all six years is also in `tests.csv` (Kendall τ: EditLens +0.29, p = 8·10⁻²⁵).

- **EditLens:** the rise is significant from 2023 on and grows every year. The three reference years do not differ from each other (Holm p = 1.00), so the baseline is stable. The synthetic abstracts do not change: no synthetic year differs from the reference years (Holm p ≥ 0.23), so new topics do not raise the scores.
- **Fast-DetectGPT:** the only significant changes point toward *more human*. Both the real and the synthetic abstracts drift that way, and 2020 already differs from 2021–2022, which matches the scoring-model drift described above. It is not evidence about AI use.
- **Binoculars:** no change in the real abstracts. The synthetic abstracts drift slightly toward *more human*, again a topic effect.

## Ready-made detectors also checked, and not used

Two more off-the-shelf detectors were checked to see whether a trained classifier other than EditLens would pass the positive control.

**Setup:**
- A paired sample of 200 papers (seed 0): 100 from 2020–2022 and 100 from 2023–2025, with each paper's real and synthetic abstract (400 texts).
- EditLens on the same 200 papers for comparison.
- The scripts were one-off and not kept. The numbers are recorded here so the result can be cited.
- This check predates the year-group rule above. Its "2023–2025" sample includes 2023 papers, and it was not rerun.

**Detectors:**
- **RADAR** (`TrustSafeAI/RADAR-Vicuna-7B`): a RoBERTa-large classifier trained adversarially against a Vicuna-7B paraphraser (2023). Non-commercial license.
- **adaptive-classifier** (`adaptive-classifier/ai-detector`): a small classification head on frozen RADAR embeddings, trained on 2,000 EditLens examples.

| | RADAR | adaptive-classifier | EditLens (same 200 papers) |
|---|---|---|---|
| A generic AI-written paragraph, P(AI) | 0.94 | 0.26 (called human) | — |
| Mean P(AI): real 2020–22 / real 2023–25 / synthetic | 0.06 / 0.03 / 0.006 | 0.150 / 0.149 / 0.157 | 0.20 / 0.35 / 0.38 |
| AUC synthetic vs real 2020–22 | **0.03** (inverted) | 0.65 | **0.82** |
| Synthetic abstract scores higher than its real pair | 2% | 62% | 70% |
| Real abstracts with P(AI) > 0.5: 2020–22 / 2023–25 | 4% / 0% | 0% / 0% | 7% / 26% |

**RADAR** recognizes old-style AI text, but it rates the synthetic abstracts as *more human* than the real ones. This is the same inversion as Fast-DetectGPT and Binoculars, only stronger. **adaptive-classifier** gives almost every text about 0.15, so it is uninformative.

**Conclusion:** of the five ready-to-use detectors tried, only EditLens detects this generator's text. It is also the only one trained on AI-*edited* text, the most likely way ISMIR authors use LLMs.
