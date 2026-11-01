# Experiment report: main

Generated 2026-11-01T16:36:41+00:00 · git commit `9201e718bdac23b76c6366323ecfda9cb9bf590f`

> All numbers below are measured values computed from `per_question.jsonl`. Evaluation is **automatic** (LLM judge + NLI model). Only the retrieval gold labels are human-annotated (MuSiQue). No claims in this project have been labelled by hand. Cells show mean [95% bootstrap CI].

## Setup

- Dataset: musique_eval (seed 13); test questions: 36 (simple: 12, moderate: 12, complex: 12); corpus 954 paragraphs.

- Pipeline models: chat `gpt-4o-mini`, embedding `text-embedding-3-small`, verifier NLI `cross-encoder/nli-roberta-base`. Evaluation: judge `gpt-4.1`, evaluator NLI `cross-encoder/nli-MiniLM2-L6-H768` (independent of the pipeline).

- Questions per variant in k/retry sweeps: 18.

- Errors: 0 question runs failed and are excluded from their variant's means.

## Ablation: answer quality and retrieval (Experiments 1 and 2)

| Variant | Correct (judge) | Token F1 | Recall@k (own k) | Precision@k (own k) | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| **B** Baseline (fixed k=5) | 0.583 [0.417, 0.750] | 0.139 [0.099, 0.181] | 0.731 [0.632, 0.829] | 0.272 [0.233, 0.317] | 0.830 [0.729, 0.918] | 0.679 [0.594, 0.763] |
| **A** Adaptive retrieval only | 0.444 [0.278, 0.611] | 0.127 [0.087, 0.169] | 0.662 [0.549, 0.771] | 0.409 [0.294, 0.529] | 0.772 [0.645, 0.884] | 0.623 [0.517, 0.721] |
| **A-** Adaptive depth, no decomposition | 0.444 [0.278, 0.611] | 0.118 [0.078, 0.161] | 0.606 [0.484, 0.722] | 0.393 [0.274, 0.517] | 0.778 [0.639, 0.903] | 0.609 [0.493, 0.718] |
| **V** Verification only (fixed k=5) | 0.500 [0.333, 0.667] | 0.109 [0.075, 0.146] | 0.745 [0.644, 0.843] | 0.264 [0.224, 0.308] | 0.830 [0.729, 0.918] | 0.687 [0.602, 0.770] |
| **A+V** Adaptive + verification | 0.444 [0.278, 0.611] | 0.102 [0.069, 0.138] | 0.699 [0.588, 0.801] | 0.405 [0.298, 0.518] | 0.786 [0.666, 0.889] | 0.639 [0.542, 0.731] |

## Evidence reliability (independent NLI evaluator)

Claims = sentences of the final answer. Support = entailed by retrieved context at probability ≥ 0.5. Faithfulness is reported as the claim support rate. Answers with no claims (abstentions) are excluded from these means.

| Variant | Claim support rate | Unsupported claim rate | Citation precision | Citation recall | Abstained |
|---|---|---|---|---|---|
| **B** Baseline (fixed k=5) | 0.286 [0.143, 0.443] | 0.714 [0.557, 0.857] | 0.327 [0.154, 0.500] | 0.181 [0.071, 0.310] | 0.22 |
| **A** Adaptive retrieval only | 0.290 [0.152, 0.443] | 0.710 [0.557, 0.848] | 0.386 [0.205, 0.591] | 0.186 [0.076, 0.314] | 0.33 |
| **A-** Adaptive depth, no decomposition | 0.250 [0.125, 0.389] | 0.750 [0.611, 0.875] | 0.405 [0.214, 0.619] | 0.204 [0.088, 0.333] | 0.31 |
| **V** Verification only (fixed k=5) | 0.328 [0.181, 0.480] | 0.672 [0.520, 0.819] | 0.330 [0.167, 0.506] | 0.206 [0.088, 0.333] | 0.28 |
| **A+V** Adaptive + verification | 0.348 [0.197, 0.515] | 0.652 [0.485, 0.803] | 0.450 [0.250, 0.650] | 0.237 [0.106, 0.379] | 0.36 |

## Efficiency

Latencies were measured with several queries in flight on a CPU machine; compare variants with each other, not in absolute terms. Cost is the estimated pipeline API cost (chat + embeddings) in USD per 1000 queries, using the configured price table.

| Variant | Chunks retrieved | Retrieval latency (s) | Total latency (s) | Tokens / query | USD / 1000 queries |
|---|---|---|---|---|---|
| **B** Baseline (fixed k=5) | 5.00 [5.00, 5.00] | 0.44 [0.34, 0.56] | 1.6 [1.5, 1.8] | 837 [772, 904] | 0.143 [0.132, 0.155] |
| **A** Adaptive retrieval only | 5.08 [4.00, 6.17] | 0.39 [0.37, 0.42] | 3.0 [2.8, 3.1] | 1357 [1181, 1531] | 0.261 [0.230, 0.291] |
| **A-** Adaptive depth, no decomposition | 4.83 [3.78, 5.89] | 0.31 [0.29, 0.33] | 2.8 [2.7, 2.9] | 1278 [1118, 1437] | 0.250 [0.222, 0.278] |
| **V** Verification only (fixed k=5) | 5.39 [5.00, 5.86] | 0.36 [0.31, 0.43] | 10.4 [7.4, 13.8] | 2853 [2172, 3626] | 0.412 [0.300, 0.544] |
| **A+V** Adaptive + verification | 5.14 [4.08, 6.17] | 0.44 [0.40, 0.48] | 21.9 [12.9, 32.8] | 2999 [2301, 3757] | 0.462 [0.370, 0.557] |

## By complexity group (correct / recall@k / chunks retrieved)

| Variant | simple | moderate | complex |
|---|---|---|---|
| **B** | 0.83 / 1.00 / 5.0 | 0.42 / 0.62 / 5.0 | 0.50 / 0.57 / 5.0 |
| **A** | 0.58 / 0.67 / 1.9 | 0.42 / 0.67 / 5.1 | 0.33 / 0.65 / 8.2 |
| **A-** | 0.58 / 0.67 / 1.9 | 0.33 / 0.50 / 4.7 | 0.42 / 0.65 / 7.9 |
| **V** | 0.75 / 1.00 / 5.5 | 0.33 / 0.62 / 5.0 | 0.42 / 0.61 / 5.7 |
| **A+V** | 0.50 / 0.75 / 2.1 | 0.33 / 0.67 / 5.2 | 0.50 / 0.68 / 8.2 |

## Paired comparisons against the baseline (B)

Same questions, difference = variant − baseline, paired bootstrap (10,000 resamples). McNemar/Wilcoxon p-values are in `summary.json`.

- **A vs B, answer correct (judge)**: mean difference -0.139 (95% CI -0.250 to -0.028, n=36); the 95% CI excludes 0.
- **A vs B, recall@k**: mean difference -0.069 (95% CI -0.208 to +0.056, n=36); the 95% CI includes 0, so no reliable difference at this sample size.
- **A vs B, unsupported claim rate**: mean difference -0.005 (95% CI -0.167 to +0.157, n=35); the 95% CI includes 0, so no reliable difference at this sample size.
- **A vs B, citation precision**: mean difference +0.119 (95% CI -0.071 to +0.310, n=21); the 95% CI includes 0, so no reliable difference at this sample size.
- **A vs B, chunks retrieved**: mean difference +0.083 (95% CI -1.028 to +1.194, n=36); the 95% CI includes 0, so no reliable difference at this sample size.
- **A vs B, total latency**: mean difference +1.350 s (95% CI +1.156 to +1.522, n=36); the 95% CI excludes 0.
- **A vs B, cost per query**: mean difference +0.118 USD/1000 (95% CI +0.091 to +0.144, n=36); the 95% CI excludes 0.
- **A- vs B, answer correct (judge)**: mean difference -0.139 (95% CI -0.250 to -0.028, n=36); the 95% CI excludes 0.
- **A- vs B, recall@k**: mean difference -0.125 (95% CI -0.264 to +0.000, n=36); the 95% CI includes 0, so no reliable difference at this sample size.
- **A- vs B, unsupported claim rate**: mean difference +0.029 (95% CI -0.100 to +0.157, n=35); the 95% CI includes 0, so no reliable difference at this sample size.
- **A- vs B, citation precision**: mean difference +0.048 (95% CI -0.143 to +0.238, n=21); the 95% CI includes 0, so no reliable difference at this sample size.
- **A- vs B, chunks retrieved**: mean difference -0.167 (95% CI -1.250 to +0.917, n=36); the 95% CI includes 0, so no reliable difference at this sample size.
- **A- vs B, total latency**: mean difference +1.199 s (95% CI +1.009 to +1.363, n=36); the 95% CI excludes 0.
- **A- vs B, cost per query**: mean difference +0.107 USD/1000 (95% CI +0.083 to +0.130, n=36); the 95% CI excludes 0.
- **V vs B, answer correct (judge)**: mean difference -0.083 (95% CI -0.194 to +0.000, n=36); the 95% CI includes 0, so no reliable difference at this sample size.
- **V vs B, recall@k**: mean difference +0.014 (95% CI +0.000 to +0.042, n=36); the 95% CI includes 0, so no reliable difference at this sample size.
- **V vs B, unsupported claim rate**: mean difference -0.034 (95% CI -0.118 to +0.034, n=34); the 95% CI includes 0, so no reliable difference at this sample size.
- **V vs B, citation precision**: mean difference +0.003 (95% CI -0.153 to +0.167, n=24); the 95% CI includes 0, so no reliable difference at this sample size.
- **V vs B, chunks retrieved**: mean difference +0.389 (95% CI +0.000 to +0.889, n=36); the 95% CI includes 0, so no reliable difference at this sample size.
- **V vs B, total latency**: mean difference +8.744 s (95% CI +5.752 to +12.141, n=36); the 95% CI excludes 0.
- **V vs B, cost per query**: mean difference +0.269 USD/1000 (95% CI +0.164 to +0.397, n=36); the 95% CI excludes 0.
- **A+V vs B, answer correct (judge)**: mean difference -0.139 (95% CI -0.250 to -0.028, n=36); the 95% CI excludes 0.
- **A+V vs B, recall@k**: mean difference -0.032 (95% CI -0.157 to +0.083, n=36); the 95% CI includes 0, so no reliable difference at this sample size.
- **A+V vs B, unsupported claim rate**: mean difference -0.045 (95% CI -0.227 to +0.136, n=33); the 95% CI includes 0, so no reliable difference at this sample size.
- **A+V vs B, citation precision**: mean difference +0.132 (95% CI -0.079 to +0.368, n=19); the 95% CI includes 0, so no reliable difference at this sample size.
- **A+V vs B, chunks retrieved**: mean difference +0.139 (95% CI -0.917 to +1.194, n=36); the 95% CI includes 0, so no reliable difference at this sample size.
- **A+V vs B, total latency**: mean difference +20.272 s (95% CI +11.400 to +30.880, n=36); the 95% CI excludes 0.
- **A+V vs B, cost per query**: mean difference +0.318 USD/1000 (95% CI +0.233 to +0.411, n=36); the 95% CI excludes 0.

## Retrieval depth by complexity (Experiment 3: fixed-k baseline sweep)

Cells: correct / recall@k / precision@k. Shows which fixed depth works best for each group; compare with the depths the adaptive variants choose.

| Fixed depth | simple | moderate | complex | Overall (correct / recall@k) |
|---|---|---|---|---|
| Baseline k=1 | 0.50 / 0.58 / 0.58 | 0.25 / 0.38 / 0.75 | 0.08 / 0.26 / 0.83 | 0.28 / 0.41 |
| Baseline k=3 | 0.75 / 0.92 / 0.31 | 0.42 / 0.54 / 0.36 | 0.25 / 0.47 / 0.50 | 0.47 / 0.64 |
| Baseline k=5 | 0.83 / 1.00 / 0.20 | 0.42 / 0.62 / 0.25 | 0.50 / 0.57 / 0.37 | 0.58 / 0.73 |
| Baseline k=8 | 0.83 / 1.00 / 0.12 | 0.42 / 0.67 / 0.17 | 0.42 / 0.61 / 0.25 | 0.56 / 0.76 |
| Baseline k=10 | 0.83 / 1.00 / 0.10 | 0.42 / 0.83 / 0.17 | 0.33 / 0.71 / 0.23 | 0.53 / 0.85 |

## Retries (Experiment 4)

| Variant | Correct | Claim support | Unsupported rate | Attempts | Latency (s) | USD / 1000 |
|---|---|---|---|---|---|---|
| A+V, max_retries=0 | 0.444 [0.222, 0.667] | 0.235 [0.059, 0.471] | 0.765 [0.529, 0.941] | 1.00 | 18.6 [10.9, 27.9] | 0.363 [0.296, 0.438] |
| A+V, max_retries=1 | 0.389 [0.167, 0.611] | 0.176 [0.000, 0.353] | 0.824 [0.647, 1.000] | 1.17 | 9.1 [6.4, 12.4] | 0.447 [0.316, 0.604] |
| A+V, max_retries=2 | 0.444 [0.222, 0.667] | 0.294 [0.118, 0.529] | 0.706 [0.471, 0.882] | 1.22 | 13.4 [7.7, 20.5] | 0.446 [0.316, 0.603] |

## Reliability versus cost (Experiment 5)

| Variant | Unsupported rate | Correct | Latency (s) | USD / 1000 queries |
|---|---|---|---|---|
| **B** Baseline (fixed k=5) | 0.714 | 0.583 | 1.6 | 0.143 |
| **A** Adaptive retrieval only | 0.710 | 0.444 | 3.0 | 0.261 |
| **A-** Adaptive depth, no decomposition | 0.750 | 0.444 | 2.8 | 0.250 |
| **V** Verification only (fixed k=5) | 0.672 | 0.500 | 10.4 | 0.412 |
| **A+V** Adaptive + verification | 0.652 | 0.444 | 21.9 | 0.462 |
| **R0** A+V, max_retries=0 | 0.765 | 0.444 | 18.6 | 0.363 |
| **R1** A+V, max_retries=1 | 0.824 | 0.389 | 9.1 | 0.447 |
| **R2** A+V, max_retries=2 | 0.706 | 0.444 | 13.4 | 0.446 |

## Complexity classifier

Agreement of predicted label with hop-count group: 0.639 over 36 queries. Accuracy against hop-count groups, a proxy for complexity, not a human complexity label.

| True group \ predicted | simple | moderate | complex |
|---|---|---|---|
| simple | 10 | 2 | 0 |
| moderate | 3 | 8 | 1 |
| complex | 0 | 7 | 5 |

| Group | n | Mean score | Mean k | Mean chunks |
|---|---|---|---|---|
| simple | 12 | 0.099 | 1.83 | 2.08 |
| moderate | 12 | 0.466 | 5.17 | 5.17 |
| complex | 12 | 0.672 | 7.67 | 8.17 |

## Costs of running this suite

- Pipeline cost summed over all recorded query runs (including answers served from the cache, priced when first generated): USD 0.1048.

- Judge calls newly made in this run: USD 0.0000.

## Limitations and sources of bias

- Small sample (test questions per group are listed above): confidence intervals are wide; treat differences whose CI includes 0 as unresolved, not as ties.
- Correctness is judged by an LLM against the MuSiQue gold answer and evidence support by an NLI model. Both are automatic and imperfect; no claims were labelled by humans, so judge/NLI error rates are unmeasured here.
- The pipeline's chat model and the judge are both OpenAI models and may share blind spots; the evaluator NLI differs from the verifier NLI (different checkpoint, size and threshold) but both are SNLI/MNLI-trained cross-encoders and share known weaknesses with numbers, dates and multi-hop entailment.
- The verifier used in these experiments is cross-encoder/nli-roberta-base, not the application default (DeBERTa-v3 base, which measured 1-3 s per pair on this CPU and made the suite impractical). Results describe this configuration; a stronger verifier may behave differently.
- The evaluator NLI (MiniLM2, threshold 0.5) looks strict: spot checks of the per-question records show answers the LLM judge marks correct and that cite the right passage flagged as unsupported (long sentences with extra detail, or facts spread over passages). Absolute claim-support and citation values are therefore probably underestimates and were not validated against human labels; use them only for relative comparison between variants, and treat them as unverified until a human-labelled sample exists.
- Claims for evidence metrics are sentences of the answer, not atomic facts, so a sentence containing one unsupported fact among supported ones counts as supported if the NLI accepts it.
- 'Simple' questions are single hops derived from MuSiQue decompositions with earlier answers substituted; 'moderate' and 'complex' are the original 2-hop and 3-4-hop questions. Complexity is therefore proxied by hop count, and the corpus (about 20 paragraphs per source question) is small, so retrieval is easier than on a large open corpus.
- MuSiQue gold evidence may be incomplete: a distractor paragraph can also contain the answer, which lowers measured precision.
- Latencies were measured while several queries ran concurrently on one CPU machine and cached responses keep the latency of their first run; use them for relative comparison only.
- Cost uses the configured price table (USD per million tokens); embedding tokens are estimated with tiktoken. Verify prices before quoting costs.
- Unanswerable questions are not part of this run, so the abstention behaviour of the verified pipeline is not evaluated here.
