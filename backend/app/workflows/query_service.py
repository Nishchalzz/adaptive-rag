"""Dispatches a query to the requested pipeline and records the trace."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import PipelineConfig
from app.core.errors import LLMNotConfiguredError, QueryNotFoundError
from app.core.llm import MISSING_KEY_MESSAGE
from app.database.models import QueryLogRow
from app.database.repositories import QueryLogRepository
from app.schemas.query import PipelineName, QueryLogSummary, QueryResponse

logger = logging.getLogger(__name__)


class Pipeline(Protocol):
    name: PipelineName

    async def run(self, question: str, corpus: str, config: PipelineConfig) -> QueryResponse: ...


class QueryService:
    def __init__(
        self, pipelines: Mapping[PipelineName, Pipeline], sessionmaker: async_sessionmaker[AsyncSession]
    ) -> None:
        self._pipelines = pipelines
        self._sessionmaker = sessionmaker

    @property
    def available(self) -> bool:
        return bool(self._pipelines)

    def _pipeline(self, name: PipelineName) -> Pipeline:
        if name not in self._pipelines:
            raise LLMNotConfiguredError(MISSING_KEY_MESSAGE)
        return self._pipelines[name]

    async def answer(self, question: str, pipeline: PipelineName, corpus: str, config: PipelineConfig) -> QueryResponse:
        response = await self._pipeline(pipeline).run(question, corpus, config)
        await self.record(response)
        return response

    async def stream(
        self, question: str, pipeline: PipelineName, corpus: str, config: PipelineConfig
    ) -> AsyncIterator[dict[str, Any]]:
        """Progress events for pipelines that support them; the final event carries the response."""
        selected = self._pipeline(pipeline)
        if hasattr(selected, "stream"):
            async for event in selected.stream(question, corpus, config):
                if event["type"] == "result":
                    await self.record(event["response"])
                yield event
            return
        response = await selected.run(question, corpus, config)
        await self.record(response)
        yield {"type": "result", "response": response}

    async def record(self, response: QueryResponse) -> None:
        retrieval = response.retrieval
        complexity = response.complexity
        verification = response.verification
        logger.info(
            "query=%s pipeline=%s variant=%s complexity=%s score=%s k=%d retrieved=%d retrieval_ms=%.1f "
            "support=%s total_ms=%.1f tokens=%d/%d question=%r",
            response.query_id,
            response.pipeline.value,
            response.variant,
            complexity.label if complexity else "-",
            f"{complexity.score:.3f}" if complexity else "-",
            retrieval.k_requested,
            retrieval.num_retrieved,
            retrieval.latency_ms,
            f"{verification.support_score:.2f}" if verification else "-",
            response.total_latency_ms,
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.question,
        )
        row = QueryLogRow(
            id=response.query_id,
            created_at=response.created_at,
            pipeline=response.pipeline.value,
            corpus=response.corpus,
            question=response.question,
            complexity_label=complexity.label if complexity else None,
            complexity_score=complexity.score if complexity else None,
            k_selected=retrieval.k_requested,
            num_retrieved=retrieval.num_retrieved,
            retrieval_latency_ms=retrieval.latency_ms,
            total_latency_ms=response.total_latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=response.usage.cost_usd,
            trace=response.model_dump(mode="json"),
        )
        async with self._sessionmaker() as session:
            await QueryLogRepository(session).add(row)
            await session.commit()

    async def list(self, *, limit: int, offset: int, pipeline: PipelineName | None) -> list[QueryLogSummary]:
        async with self._sessionmaker() as session:
            rows = await QueryLogRepository(session).list(
                limit=limit, offset=offset, pipeline=pipeline.value if pipeline else None
            )
        return [QueryLogSummary.model_validate(r, from_attributes=True) for r in rows]

    async def get(self, query_id: str) -> QueryResponse:
        async with self._sessionmaker() as session:
            row = await QueryLogRepository(session).get(query_id)
        if row is None:
            raise QueryNotFoundError(f"Query {query_id} not found")
        return QueryResponse.model_validate(row.trace)
