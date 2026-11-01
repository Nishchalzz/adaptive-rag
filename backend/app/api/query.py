import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from app.api.deps import get_container
from app.core.container import ServiceContainer
from app.core.errors import AppError
from app.schemas.query import PipelineName, QueryLogSummary, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])

Container = Annotated[ServiceContainer, Depends(get_container)]


def _resolve(request: QueryRequest, container: ServiceContainer) -> tuple[Any, str]:
    config = container.pipeline_config.with_overrides(request.overrides)
    return config, request.corpus or container.settings.default_corpus


@router.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest, container: Container) -> QueryResponse:
    config, corpus = _resolve(request, container)
    return await container.queries.answer(request.question, request.pipeline, corpus, config)


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(jsonable_encoder(event))}\n\n"


@router.post("/query/stream")
async def stream_query(request: QueryRequest, container: Container) -> StreamingResponse:
    """Server-sent events: one ``step`` event per pipeline node, then ``result`` (or ``error``)."""
    config, corpus = _resolve(request, container)
    events = container.queries.stream(request.question, request.pipeline, corpus, config)
    # Pull the first event before responding, so configuration errors still become normal HTTP errors.
    first = await anext(events)

    async def body() -> AsyncIterator[str]:
        yield _sse(first)
        try:
            async for event in events:
                yield _sse(event)
        except AppError as exc:
            yield _sse({"type": "error", "detail": exc.message})
        except Exception:  # noqa: BLE001 - the client must learn that the stream failed
            logger.exception("Streaming query failed")
            yield _sse({"type": "error", "detail": "Internal error while answering the question"})

    return StreamingResponse(
        body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.get("/queries", response_model=list[QueryLogSummary])
async def list_queries(
    container: Container,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    pipeline: PipelineName | None = None,
) -> list[QueryLogSummary]:
    return await container.queries.list(limit=limit, offset=offset, pipeline=pipeline)


@router.get("/queries/{query_id}", response_model=QueryResponse)
async def get_query(query_id: str, container: Container) -> QueryResponse:
    return await container.queries.get(query_id)
