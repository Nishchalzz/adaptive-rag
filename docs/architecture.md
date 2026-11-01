# Architecture

This document records the approved Phase 1 design. Sections marked *(planned)* describe components
that later phases implement; everything else exists in the codebase.

## 1. Research goal

Conventional RAG retrieves a fixed number of chunks (top-k) for every question and trusts the
generated answer. This project tests two additions:

1. **Query-adaptive retrieval:** estimate how complex a question is and set the retrieval depth
   from that estimate.
2. **Evidence verification:** split the answer into atomic claims, check each claim against the
   passage it cites, and when claims are unsupported, retrieve more evidence and regenerate (bounded
   retries). If evidence remains insufficient, uncertainty is stated explicitly.

### Related work and positioning

Both ideas have close prior work, which the report must cite:
- Adaptive-RAG (Jeong et al., NAACL 2024): routes queries by a simple/moderate/complex classifier.
- Self-RAG (Asai et al., 2023) and CRAG (Yan et al., 2024): critique and correct retrieval.
- FActScore (Min et al., 2023): atomic-fact decomposition.
- ALCE (Gao et al., 2023): NLI-based citation precision and recall.
- RARR (Gao et al., 2023): post-hoc revision of unsupported output.

The contribution evaluated here is their combination: a continuous complexity score that sets
retrieval depth, plus claim-level citation verification that drives bounded re-retrieval, measured
in a controlled 2×2 ablation.

## 2. System overview

```
React + Vite + Tailwind (planned)
  Chat · Documents · Query Inspector · Research Dashboard
        │ REST (+ SSE for live pipeline steps, planned)
FastAPI
  api/: documents · query · system (· experiments, planned)
   ├─ BaselineRAGPipeline         retrieve(k fixed) → generate
   └─ Proposed pipeline (LangGraph, planned; flags: adaptive, decompose, verify)
  shared by all pipelines: DenseRetriever · AnswerGenerator (one prompt) · LLM/embedding factories · UsageTracker
   │
   ├─ ChromaDB     chunks + embeddings, one collection per corpus
   ├─ SQL DB       documents registry, query traces (SQLite default, PostgreSQL via DATABASE_URL)
   └─ experiments/results/<run_id>/  config snapshot, per-question results, summaries (planned)
```

### Fairness by construction

The baseline and the proposed pipeline share:
- the vector store and its corpus
- the embedding model (`DenseRetriever` is the only component that embeds queries)
- the chat model
- the generation prompt (`app/generation/prompts.py`)

They differ only in retrieval policy and in whether verification runs. Every response records the
model IDs and a full snapshot of the research configuration that produced it.

## 3. Configuration

- `Settings` (environment / `.env`): API key, model IDs, temperature/seed, storage paths.
- `PipelineConfig` (code defaults, optional YAML via `PIPELINE_CONFIG_PATH`, per-request
  `overrides`): chunking, baseline top-k, generation budget, price table. Unknown keys are
  rejected, so a typo in an experiment config fails loudly instead of silently using a default.

## 4. Ingestion (implemented)

1. **Load** (`ingestion/loaders.py`): PDF via pypdf, keeping 1-based page numbers; DOCX via
   docx2txt; TXT/Markdown with UTF-8 → CP1252 fallback. Files with no extractable text are rejected
   (scanned PDFs would need OCR, which is out of scope).
2. **Clean** (`ingestion/cleaning.py`): NFKC normalisation (expands ligatures), control-character
   removal, Markdown syntax stripping, and for PDFs, de-hyphenation and re-joining of hard-wrapped
   lines.
3. **Chunk** (`ingestion/chunking.py`): recursive character splitting that prefers paragraph,
   line, then sentence boundaries. Size and overlap are configurable (default 1000/150 characters).
   Pages are chunked separately so each chunk maps to exactly one page.
4. **Metadata:** every chunk carries `chunk_id`, `doc_id`, `filename`, `page`, `chunk_index` and
   `char_start`, so citations resolve to an exact source location.
5. **Embed and store:** embeddings are computed outside the vector store and upserted into the
   corpus collection (cosine space).

The document registry records file hash (duplicate uploads rejected per corpus), status
(`processing` / `ready` / `failed` with error message), page and chunk counts, and the chunking
settings used, so corpora with mixed chunking can be detected.

## 5. Baseline pipeline (implemented)

`question → embed → top-k (default 5) → numbered context [1..k] → generate → parse citations`

The generation prompt requires a citation on every factual sentence and an explicit statement when
the passages are insufficient. Citation parsing (`generation/citations.py`) splits the answer into
sentences, attaches bracketed citations to their sentence, maps numbers to chunk IDs, and reports
numbers that match no retrieved passage as `invalid_citations`.

Each query is logged with: question, pipeline, k, number retrieved, retrieval latency (query
embedding + vector search), total latency, tokens per stage, and estimated cost. The full response
is stored as a trace.

## 6. Query-adaptive retrieval *(planned, Phase 2b)*

1. **Query analysis** (one structured LLM call, temperature 0): atomic information needs
   (sub-questions), reasoning type (lookup, aggregation, comparison, temporal/numeric, causal,
   bridge), whether multiple sources are required, the LLM's own label and a rationale.
2. **Lexical features** (deterministic): comparison markers, enumerations, nested relational
   chains, entity/date/number counts. Query length is deliberately not a feature.
3. **Score** `S = 0.35·needs + 0.25·reasoning + 0.15·multi_source + 0.15·lexical + 0.10·llm_label`,
   where `needs = min(n−1, 3)/3`. All weights are configurable and checked on the dev split.
4. **Label and depth:** simple (S < 0.33) → k ∈ [1,3]; moderate (< 0.66) → k ∈ [4,7]; complex →
   k ∈ [8,12]. k is interpolated by where S falls within its label's range.
5. **Retrieval:** when there are ≥2 information needs and decomposition is enabled, retrieve for
   the question and each sub-question, fuse the lists with Reciprocal Rank Fusion, and keep the
   top k.
6. **Confidence** `C = 0.5·top1 + 0.2·mean_topk + 0.3·coverage`, using dev-calibrated score
   normalisation. Coverage is the fraction of information needs with a chunk above a relevance
   threshold.
7. **Expansion:** if C is below threshold, raise k (capped) and re-retrieve for uncovered needs.
   The number of expansions is bounded.

## 7. Evidence verification *(planned, Phase 2c)*

1. **Claim extraction** (structured LLM call): atomic, self-contained claims, each inheriting its
   sentence's citations.
2. **Evidence matching:** rank windows of about 3 sentences from the cited chunks by embedding
   similarity. Similarity only selects which text to test; it never decides support.
3. **Entailment:** a local NLI cross-encoder (default
   `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) scores each window and the top-2 windows joined.
4. **Decision:**
   - entailment ≥ 0.75 → Supported
   - contradiction ≥ 0.75 → Unsupported (contradicted)
   - otherwise an LLM judge decides Supported / Partially Supported / Unsupported, and must quote
     evidence that exists in the chunk
5. **Citation check:** a claim supported only by a non-cited retrieved chunk is `misattributed`,
   and the citation is repaired and logged. A claim without a citation is `missing`.
6. **Scores:** support = (supported + 0.5·partial) / claims. The answer passes when support ≥ 0.9
   and no claim is Unsupported.
7. **Refinement:**
   - no evidence → targeted retrieval using the claim as the query
   - contradiction → feedback only

   Then regenerate and re-verify, up to `max_retries` (default 2). If retries run out, return the
   best attempt with an uncertainty notice and every unsupported claim flagged.

## 8. LangGraph workflow *(planned, Phase 2d)*

```
START → analyze_query → select_depth → retrieve → assess_confidence ⇄ expand (bounded)
assess_confidence → generate → extract_claims → verify
verify ─ pass ─────────────────────────► finalize
verify ─ fail & attempt < max ─► refine ─► generate
verify ─ fail & attempt = max ─────────► finalize_with_uncertainty
```

The flags `adaptive`, `decompose` and `verify` produce the ablation variants from one graph. The
baseline remains a separate minimal class, and a test asserts that it retrieves exactly what the
graph retrieves with all flags off.

## 9. Evaluation *(planned, Phase 3)*

See `docs/experimental_methodology.md` (written in Phase 3). In summary:
- **Data:** MuSiQue, which has human-annotated supporting paragraphs. Simple questions are
  single-hop decomposition steps, moderate are 2-hop, complex are 3–4-hop; optional unanswerable
  questions. Split 30% dev / 70% test.
- **Configurations:** baseline, adaptive, adaptive without decomposition, verification only, full
  system, a k-sweep and a retry-sweep.
- **Metrics:** retrieval P/R@k, MRR and nDCG; answer EM/F1 and RAGAS; claim support,
  unsupported-claim rate and citation precision/recall; latency, tokens and cost.
- **Integrity:** evaluators are independent of the verifier (a different NLI model, a different
  judge LLM, and a human-labelled claim sample with Cohen's κ). Paired bootstrap confidence
  intervals are reported. Results are measured, never hand-entered.
