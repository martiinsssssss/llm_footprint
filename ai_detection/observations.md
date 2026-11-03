# AI-detection observations on ISMIR abstracts (2020–2025)

## Data

- **Real abstracts:** `abstracts_clean_v1.jsonl`. These are 653 ISMIR abstracts from 2020–2025, taken from the parser output (dataset v1) and cleaned with `ai_detection/cleaning.py`, which removes footnotes and broken URLs. Two papers are dropped because their abstract field holds something else: `2023_000014` (a figure caption) and `2024_000108` (the abstract plus part of the introduction).
- **Synthetic abstracts:** `synthetic_abstracts_v1.jsonl`, made with `ai_detection/synthetic.py`. There is one per real paper, written by `claude-sonnet-5-5` from the full paper with the prompt "Write an ISMIR-style abstract of about N words…". N is the length of the real abstract. The same two papers are dropped, so both sets have the same 653 paper IDs and every real abstract has a synthetic pair.
- **Role of the synthetic set:** it is a positive control. Every synthetic abstract is fully AI-written, on the same paper and at about the same length as its real pair.

## Detectors

| Detector | Type | Score | AI direction | Status |
|---|---|---|---|---|
| EditLens (`pangram/editlens_roberta-large`) | Trained classifier, 4 buckets | 0–1, the expected bucket divided by 3 | higher = more AI | done |
| Fast-DetectGPT (GPT-Neo-2.7B as sampling and scoring model) | Zero-shot | analytic criterion | higher = more AI | pending |
| Binoculars (falcon-7b / falcon-7b-instruct, 4-bit, float16, Colab T4) | Zero-shot | log PPL / log X-PPL | **lower** = more AI | pending |

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

| Share of abstracts | Real 2020–2022 | Real 2023–2025 | Synthetic (all years) |
|---|---|---|---|
| score > 0.5 | 5.7% | 23.7% | 23.0% |
| score > 0.8 | 0.6% | 8.4% | 5.5% |

| Predicted bucket | 0 (human) | 1 | 2 | 3 (fully AI) |
|---|---|---|---|---|
| Real | 353 | 240 | 1 | 59 |
| Synthetic | 95 | 471 | 0 | 87 |

AUC for separating synthetic abstracts from real 2020–2022 abstracts: **0.84**.

### Observations

1. **Real abstracts rise steadily after ChatGPT.** Scores are flat at about 0.19 for 2020–2022, then rise to 0.26 (2023), 0.35 (2024) and 0.48 (2025). The share above 0.5 grows from 5.7% before 2023 to 23.7% from 2023 on.
2. **Changes in paper topics do not explain the rise.** The synthetic abstracts stay flat at about 0.41 in every year, although each is written from a paper of that year. If newer topics (for example more papers about LLMs) raised the scores by themselves, the synthetic scores would rise too.
3. **EditLens underestimates fully AI-written abstracts from this generator.** Synthetic abstracts have a median of only 0.38, and 72% fall in bucket 1 rather than bucket 3. A likely reason is that they summarize the real paper and reuse its content and terminology, which EditLens may read as human text with light AI editing. EditLens scores therefore work for comparing years, but should not be read as an "X% AI" measure.
4. **Real 2025 abstracts score higher than the synthetic ones** (0.476 vs 0.411; the synthetic abstract is higher in only 41% of 2025 pairs). This does not mean they are more AI-written than fully AI-written text. One hypothesis, not tested: the models 2024–2025 authors used, such as the GPT-4o generation, are closer to EditLens's training data than `claude-sonnet-5-5`, the newer model behind the synthetic set. The synthetic set is a valid positive control, but its generator is newer than the models 2024–2025 authors would have used.
5. **Bucket 2 is almost never predicted** (1 of 1,306 abstracts). In practice EditLens separates these abstracts into "human", "lightly edited" and "fully AI".

Note: these are descriptive statistics only. No significance tests have been run yet.

## Fast-DetectGPT

_Pending._

## Binoculars

_Pending._

## Agreement between detectors

_Pending. Compare the per-year trends and the rank correlations of the scores per paper._
