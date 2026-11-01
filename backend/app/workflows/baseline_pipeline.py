"""Conventional RAG baseline: fixed top-k retrieval -> cited answer generation.

No query analysis, no adaptive depth, no evidence verification. It shares the dense retriever,
embedding model, LLM and generation prompt with the proposed pipeline, so a comparison between the
two isolates the effect of the proposed components.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from app.core.config import PipelineConfig
from app.core.errors import EmptyCorpusError
from app.core.usage import UsageTracker
from app.generation.citations import parse_citations
from app.generation.generator import AnswerGenerator
from app.retrieval.baseline_retriever import FixedKRetriever
from app.retrieval.dense import DenseRetriever
from app.schemas.query import PipelineName, QueryResponse, RetrievalTrace
from app.workflows.common import build_citations, build_sentences, to_retrieved_chunks

logger = logging.getLogger(__name__)


class BaselineRAGPipeline:
    name = PipelineName.BASELINE

    def __init__(self, dense: DenseRetriever, generator: AnswerGenerator, *, embedding_model: str) -> None:
        self._dense = dense
        self._generator = generator
        self._embedding_model = embedding_model

    async def run(self, question: str, corpus: str, config: PipelineConfig) -> QueryResponse:
        started = perf_counter()
        if self._dense.corpus_size(corpus) == 0:
            raise EmptyCorpusError(f"Corpus '{corpus}' has no indexed documents")
        usage = UsageTracker(config.pricing)
        retriever = FixedKRetriever(self._dense, config.baseline.top_k)

        retrieval = await retriever.retrieve(corpus, question)
        usage.record_embedding("query_embedding", self._embedding_model, [question])
        chunks = to_retrieved_chunks(retrieval.hits)

        generation_start = perf_counter()
        generation = await self._generator.generate(
            question, [h.chunk for h in retrieval.hits], max_tokens=config.generation.max_answer_tokens
        )
        generation_ms = (perf_counter() - generation_start) * 1000
        usage.record_chat("generation", self._generator.model_name, generation.message, generation.prompt_text)

        parsed = parse_citations(generation.answer, len(chunks))
        total_ms = (perf_counter() - started) * 1000
        return QueryResponse(
            query_id=uuid4().hex,
            created_at=datetime.now(timezone.utc),
            pipeline=self.name,
            corpus=corpus,
            question=question,
            answer=generation.answer,
            citations=build_citations(parsed, chunks),
            sentences=build_sentences(parsed),
            invalid_citations=parsed.invalid,
            retrieval=RetrievalTrace(
                strategy=retriever.strategy,
                k_requested=retrieval.k_requested,
                k_final=retrieval.k_requested,
                num_retrieved=len(chunks),
                queries=[question],
                latency_ms=retrieval.latency_ms,
                embedding_latency_ms=retrieval.embedding_latency_ms,
                search_latency_ms=retrieval.search_latency_ms,
                chunks=chunks,
            ),
            usage=usage.summary(),
            timings_ms={"retrieval": retrieval.latency_ms, "generation": generation_ms, "total": total_ms},
            total_latency_ms=total_ms,
            models={"chat": self._generator.model_name, "embedding": self._embedding_model},
            config=config.model_dump(mode="json"),
        )
