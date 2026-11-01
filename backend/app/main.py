import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.api import documents, experiments, query, system
from app.core.config import Settings, get_settings
from app.core.container import build_container
from app.core.errors import AppError
from app.core.logging import configure_logging
from app.seed import seed_demo_corpus
from app.vectorstore.store import VectorStore
from app.verification.nli import NLIRegistry

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    chat_model: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    store: VectorStore | None = None,
    nli: NLIRegistry | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = await build_container(settings, chat_model=chat_model, embeddings=embeddings, store=store, nli=nli)
        app.state.container = container
        try:
            await seed_demo_corpus(container)
        except Exception:  # noqa: BLE001 - a failed demo seed must not prevent the API from starting
            logger.exception("Seeding the demo corpus failed")
        try:
            yield
        finally:
            await container.aclose()

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    app.include_router(system.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(experiments.router)
    return app


app = create_app()
