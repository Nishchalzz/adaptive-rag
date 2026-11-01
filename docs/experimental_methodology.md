# Experimental methodology

This document describes how the experiments are built and measured. It contains **no result values**: results
exist only as files written by the runner (`experiments/results/<suite>/summary.json`, `report.md`,
`figures/`, `per_question.jsonl`). Whatever those files show is what the project reports, including null or
negative findings.

## 1. Reproducing

```bash
cd backend
python -m scripts.prepare_dataset                       # builds experiments/datasets/musique_eval*.{json,jsonl}
python -m scripts.run_experiments --suite pilot --per-group 1   # cost pilot (3 questions)
python -m scripts.run_experiments --suite main                  # full suite
```

Fixed: sampling seed 13, temperature 0, LLM seed 42, pinned dependencies (`requirements.lock.txt`). Pipeline
answers and judge verdicts are cached under `experiments/cache/` keyed by the complete configuration, so a rerun
reproduces the same responses without new API calls. Each suite stores its configuration snapshot, models, git
commit (if the repository has one), dataset statistics and per-question records.

## 2. Dataset

Source: MuSiQue v1.0 dev set (Trivedi et al., TACL 2022), answerable questions. MuSiQue provides, for each
question, 20 paragraphs (supporting + distractors), a human-written decomposition into hops, and
human-annotated supporting paragraphs.

| Group | Construction | Gold evidence |
|---|---|---|
| simple | one hop of a decomposition that is not a relation template (`X >> relation`); `#k` references replaced by the earlier hops' answers, so the question stands alone | that hop's supporting paragraph |
| moderate | original 2-hop questions | the 2 supporting paragraphs |
| complex | original 3- and 4-hop questions | the 3-4 supporting paragraphs |

Per group: 17 questions, split 5 dev / 12 test (stratified, seeded). **All reported numbers use test questions
only**; dev questions are available for calibration (score floors, thresholds) and were not used to tune
anything in this run. Source questions for different groups are disjoint. The corpus is every paragraph of
every sampled source question (supporting and distractors), indexed as one chunk per paragraph into a
dedicated Chroma collection (`musique_eval`) with the same embedding model as the application.

The sample is deliberately small to keep API cost low; see limitations.

## 3. Systems and ablation

All variants share the vector store, embedding model, chat model, generation prompt, and answer-token limit.

| Name | Pipeline | Retrieval | Decomposition | Verification and retry |
|---|---|---|---|---|
| B | baseline class | fixed top-k (k=5) | no | no |
| A | proposed graph | adaptive k from complexity + confidence expansion | yes | no |
| A- | proposed graph | adaptive k | no | no |
| V | proposed graph | fixed k=5 | no | yes (max 2 retries) |
| A+V | proposed graph | adaptive | yes | yes (max 2 retries) |

Sweeps: fixed-k baseline with k in {1, 3, 5, 8, 10} (Experiment 3); A+V with `max_retries` in {0, 1, 2}
(Experiment 4, on a subset of questions per group; `R2` equals A+V). Experiment 5 is the combination of
reliability and cost columns across all variants.

## 4. Metrics

| Family | Metric | Computed from | Nature |
|---|---|---|---|
| Retrieval | Precision@k, Recall@k at the system's own depth; P/R at fixed cutoffs 1, 3, 5, 10; MRR; nDCG@10 (binary) | MuSiQue gold paragraphs vs retrieved chunk ids (including chunks added by refinement) | gold labels are human-annotated; metric is automatic |
| Answer | exact match, token F1, containment vs gold answer and aliases | normalised final answer text (citations removed) | automatic |
| Answer | correctness (correct / incorrect / abstained), answer relevance (addresses the question) | independent judge LLM (`JUDGE_MODEL`) comparing answer with the gold answer | automatic (LLM judge) |
| Evidence | claim support rate (= faithfulness), unsupported claim rate | evaluator NLI: claim entailed by some retrieved chunk (or the cited chunks together) at probability >= 0.5 | automatic |
| Evidence | citation recall: claim entailed by its cited chunks together; citation precision: share of citations that are relevant under the ALCE rule (alone entails, or needed for the set to entail) | evaluator NLI | automatic |
| Efficiency | chunks retrieved, retrieval latency, total latency, input/output tokens, estimated cost | `QueryResponse` | measured |

Precision@c divides by c even if fewer chunks were retrieved. Claims for the evidence metrics are the
answer's sentences with abstention sentences removed; answers with no claims are excluded from those means.
RAGAS was not used: its faithfulness and correctness metrics are LLM-judge based, and the equivalent
measures here are implemented directly (judge for correctness/relevance, NLI for faithfulness) so that all
evaluators are explicit, cached and independent of the evaluated pipeline.

### Evaluator independence

The proposed pipeline verifies its own answers with an NLI cross-encoder (threshold 0.75), an LLM claim
extractor and a quote-checked LLM judge. Evidence metrics are computed with a **different checkpoint**
(threshold 0.5) over rule-split sentences, so the pipeline is not graded by its own verifier.

Checkpoints used in the experiments (recorded in each `summary.json`):

| Role | Checkpoint | Why |
|---|---|---|
| Pipeline verifier | `cross-encoder/nli-roberta-base` | The application default is `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, but with the installed transformers/torch on CPU it measured about 1-3 s per sentence pair, which made a full suite take hours. The experiments therefore override it (`--verifier-nli`). Results describe this configuration. |
| Evaluator | `cross-encoder/nli-MiniLM2-L6-H768` | Different architecture and size from the verifier. Both were trained on SNLI/MNLI, so they are not fully independent (shared weaknesses with numbers and dates). |
| Answer judge | `JUDGE_MODEL` (default `gpt-4.1`) | Larger than the pipeline's chat model; sees only question, gold answer, final answer. |

### Complexity classifier

Predicted label vs hop group (agreement and confusion matrix) and mean selected k per group are reported from
the A+V runs. The hop group is a proxy; it is not a human complexity annotation.

## 5. Statistics

Means carry 95% percentile bootstrap CIs (10,000 resamples over questions). Comparisons against B are paired
on questions: mean difference with a paired-bootstrap CI, Wilcoxon signed-rank p-value, and an exact McNemar
test for binary metrics. With about 36 test questions, CIs are wide; a CI that includes 0 means the
experiment could not resolve a difference, not that there is none. No multiple-comparison correction is
applied; read the paired tables as exploratory.

## 6. Human verification

None. Retrieval gold labels are human-annotated by the MuSiQue authors; every other label in the results is
produced by a model. A human-labelled sample of claims (two annotators, Cohen's kappa against the judge and
NLI) is the recommended next step and is not part of these results. Automatic and human-verified figures
must be reported separately.

## 7. Limitations and sources of bias

- Small sample and a small corpus (about 20 paragraphs per question), so retrieval is easier than in open-domain use.
- Hop count is a proxy for complexity; "simple" questions are re-written from decompositions.
- LLM judge and NLI model errors are unmeasured; both families share known weaknesses (numbers, dates, multi-hop entailment).
- Pipeline and judge are both OpenAI models and may share blind spots.
- Sentence-level claims are coarser than atomic facts.
- MuSiQue gold evidence can be incomplete (a distractor may also contain the answer), which lowers measured precision.
- Latency is measured under concurrency on CPU and cached responses keep their first-run latency: use for relative comparison.
- Cost depends on the configured price table; embedding tokens are estimated.
- Unanswerable questions are not included; abstention behaviour is not evaluated.
- Verification adds LLM calls and NLI compute by design; the cost/latency trade-off is reported, not hidden.
