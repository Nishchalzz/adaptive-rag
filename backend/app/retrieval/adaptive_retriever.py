"""Query-adaptive retrieval (research contribution 1, steps 2-4).

* Depth k comes from the complexity assessment.
* With decomposition, the question and each information need are searched separately and the ranked
  lists are fused with Reciprocal Rank Fusion; each query's best chunk is guaranteed a place so every
  information need is represented.
* Retrieval confidence combines the normalised best score, the normalised mean score and the share of
  queries covered by a sufficiently similar chunk. Low confidence triggers a bounded expansion that
  deepens the search for uncovered queries.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field

from app.core.config import AdaptiveConfig
from app.retrieval.dense import DenseRetriever
from app.schemas.pipeline import ComplexityAssessment, RetrievalConfidence
from app.vectorstore.store import SearchHit


@dataclass
class FusedHit:
    hit: SearchHit
    query_index: int  # query whose list ranked this chunk best (0 = original question)
    rrf_score: float


@dataclass
class AdaptiveRetrievalState:
    corpus: str
    queries: list[str]
    vectors: list[list[float]]
    per_query: list[list[SearchHit]]
    k_initial: int
    k: int
    selected: list[FusedHit] = field(default_factory=list)
    expansions: int = 0
    embedding_latency_ms: float = 0.0
    search_latency_ms: float = 0.0

    @property
    def latency_ms(self) -> float:
        return self.embedding_latency_ms + self.search_latency_ms


def build_queries(question: str, assessment: ComplexityAssessment | None, config: AdaptiveConfig) -> list[str]:
    queries = [question]
    if assessment is not None and config.decompose and len(assessment.information_needs) >= 2:
        for need in assessment.information_needs[: config.max_subqueries]:
            if need.strip().lower() not in {q.strip().lower() for q in queries}:
                queries.append(need)
    return queries


def fuse(per_query: list[list[SearchHit]], k: int, rrf_k: int) -> list[FusedHit]:
    if len(per_query) == 1:  # no fusion: identical to plain top-k
        return [FusedHit(hit=h, query_index=0, rrf_score=1 / (rrf_k + r)) for r, h in enumerate(per_query[0][:k], 1)]

    rrf: dict[str, float] = defaultdict(float)
    best: dict[str, SearchHit] = {}
    source: dict[str, tuple[int, int]] = {}  # chunk -> (best rank, query index)
    for query_index, hits in enumerate(per_query):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            rrf[chunk_id] += 1 / (rrf_k + rank)
            if chunk_id not in best or hit.score > best[chunk_id].score:
                best[chunk_id] = hit
            if chunk_id not in source or rank < source[chunk_id][0]:
                source[chunk_id] = (rank, query_index)
    order = sorted(rrf, key=lambda c: (-rrf[c], -best[c].score))

    chosen: list[str] = []
    for hits in per_query:  # coverage guarantee: each query's best chunk first
        if hits and len(chosen) < k and hits[0].chunk.chunk_id not in chosen:
            chosen.append(hits[0].chunk.chunk_id)
    for chunk_id in order:
        if len(chosen) >= k:
            break
        if chunk_id not in chosen:
            chosen.append(chunk_id)
    chosen.sort(key=order.index)
    return [FusedHit(hit=best[c], query_index=source[c][1], rrf_score=rrf[c]) for c in chosen]


def _normalise(score: float, config: AdaptiveConfig) -> float:
    return min(max((score - config.score_floor) / (config.score_ceiling - config.score_floor), 0.0), 1.0)


def assess_confidence(state: AdaptiveRetrievalState, config: AdaptiveConfig) -> RetrievalConfidence:
    scores = [f.hit.score for f in state.selected]
    if not scores:
        return RetrievalConfidence(score=0, top1=0, mean=0, coverage=0, uncovered_queries=list(state.queries), sufficient=False)
    top1 = _normalise(max(scores), config)
    mean = sum(_normalise(s, config) for s in scores) / len(scores)
    uncovered = [
        q for q, hits in zip(state.queries, state.per_query, strict=True)
        if not hits or hits[0].score < config.relevance_threshold
    ]
    coverage = 1 - len(uncovered) / len(state.queries)
    weights = (config.confidence_weight_top1, config.confidence_weight_mean, config.confidence_weight_coverage)
    total = sum(weights) or 1.0
    score = (weights[0] * top1 + weights[1] * mean + weights[2] * coverage) / total
    return RetrievalConfidence(
        score=round(score, 4),
        top1=round(top1, 4),
        mean=round(mean, 4),
        coverage=round(coverage, 4),
        uncovered_queries=uncovered,
        sufficient=score >= config.confidence_threshold,
    )


class AdaptiveRetriever:
    strategy = "adaptive"

    def __init__(self, dense: DenseRetriever) -> None:
        self._dense = dense

    async def _search_all(self, corpus: str, vectors: list[list[float]], depth: int) -> tuple[list[list[SearchHit]], float]:
        results = await asyncio.gather(*(self._dense.search(corpus, v, depth) for v in vectors))
        # searches run concurrently, so wall time is the slowest one
        return [hits for hits, _ in results], max(ms for _, ms in results)

    async def retrieve(
        self, corpus: str, question: str, assessment: ComplexityAssessment | None, k: int, config: AdaptiveConfig
    ) -> AdaptiveRetrievalState:
        queries = build_queries(question, assessment, config)
        vectors, embed_ms = await self._dense.embed_queries(queries)
        per_query, search_ms = await self._search_all(corpus, vectors, k)
        state = AdaptiveRetrievalState(
            corpus=corpus,
            queries=queries,
            vectors=vectors,
            per_query=per_query,
            k_initial=k,
            k=k,
            embedding_latency_ms=embed_ms,
            search_latency_ms=search_ms,
        )
        state.selected = fuse(per_query, k, config.rrf_k)
        return state

    async def expand(self, state: AdaptiveRetrievalState, confidence: RetrievalConfidence, config: AdaptiveConfig) -> None:
        """Deepen the search (k += step, capped); uncovered queries are re-searched, or all if none is uncovered."""
        state.k = min(state.k + config.expansion_step, config.k_cap)
        targets = [i for i, q in enumerate(state.queries) if q in confidence.uncovered_queries] or list(range(len(state.queries)))
        depth = state.k
        hits, search_ms = await self._search_all(state.corpus, [state.vectors[i] for i in targets], depth)
        for index, new_hits in zip(targets, hits, strict=True):
            state.per_query[index] = new_hits
        state.search_latency_ms += search_ms
        state.expansions += 1
        state.selected = fuse(state.per_query, state.k, config.rrf_k)
