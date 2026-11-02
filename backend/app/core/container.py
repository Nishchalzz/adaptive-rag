"""Builds the long-lived services once at startup and wires them together."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import PipelineConfig, Settings, load_pipeline_config
from app.core.llm import MISSING_KEY_MESSAGE, create_chat_model
from app.database.session import create_engine, create_sessionmaker, init_db
from app.embeddings.factory import create_embeddings
from app.generation.generator import AnswerGenerator
from app.ingestion.service import DocumentService
from app.retrieval.adaptive_retriever import AdaptiveRetriever
from app.retrieval.complexity_classifier import ComplexityClassifier
from app.retrieval.dense import DenseRetriever
from app.schemas.query import PipelineName
from app.vectorstore.store import ChromaVectorStore, VectorStore
from app.verification.claim_extractor import ClaimExtractor
from app.verification.evidence_matcher import EvidenceMatcher
from app.verification.nli import NLIRegistry
from app.verification.verifier import EvidenceVerifier
from app.workflows.baseline_pipeline import BaselineRAGPipeline
from app.workflows.demo_cache import DemoCache
from app.workflows.query_service import Pipeline, QueryService
from app.workflows.rag_graph import ProposedComponents, ProposedRAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class ServiceContainer:
    settings: Settings
    pipeline_config: PipelineConfig
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]
    store: VectorStore
    nli: NLIRegistry
    documents: DocumentService
    queries: QueryService
    pipelines: dict[PipelineName, Pipeline]

    @property
    def llm_configured(self) -> bool:
        return self.queries.available

    async def aclose(self) -> None:
        await self.engine.dispose()


def build_pipelines(
    *,
    settings: Settings,
    store: VectorStore,
    chat_model: BaseChatModel,
    embeddings: Embeddings,
    nli: NLIRegistry,
) -> dict[PipelineName, Pipeline]:
    """Both pipelines share the dense retriever, chat model and generator (same prompt)."""
    dense = DenseRetriever(store, embeddings)
    generator = AnswerGenerator(chat_model, settings.chat_model)
    proposed = ProposedRAGPipeline(
        ProposedComponents(
            dense=dense,
            adaptive=AdaptiveRetriever(dense),
            classifier=ComplexityClassifier(chat_model, settings.chat_model),
            generator=generator,
            extractor=ClaimExtractor(chat_model, settings.chat_model),
            verifier=EvidenceVerifier(
                nli=nli,
                matcher=EvidenceMatcher(embeddings, settings.embedding_model),
                judge_model=chat_model,
                judge_model_name=settings.chat_model,
            ),
            embedding_model=settings.embedding_model,
        )
    )
    return {
        PipelineName.BASELINE: BaselineRAGPipeline(dense, generator, embedding_model=settings.embedding_model),
        PipelineName.PROPOSED: proposed,
    }


async def build_container(
    settings: Settings,
    *,
    chat_model: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    store: VectorStore | None = None,
    nli: NLIRegistry | None = None,
) -> ServiceContainer:
    """Injected components (tests) take precedence over those created from settings."""
    pipeline_config = load_pipeline_config(settings.pipeline_config_path)
    assert settings.database_url and settings.chroma_dir and settings.upload_dir and settings.demo_cache_dir  # derived
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(settings.database_url)
    await init_db(engine)
    sessionmaker = create_sessionmaker(engine)
    store = store or ChromaVectorStore(settings.chroma_dir)
    nli = nli or NLIRegistry(device=settings.nli_device)

    if settings.llm_configured:
        chat_model = chat_model or create_chat_model(settings)
        embeddings = embeddings or create_embeddings(settings)
    if chat_model is None or embeddings is None:
        logger.warning("LLM provider not configured: ingestion and querying are disabled. %s", MISSING_KEY_MESSAGE)

    documents = DocumentService(
        sessionmaker=sessionmaker,
        store=store,
        embeddings=embeddings,
        chunking=pipeline_config.chunking,
        upload_dir=settings.upload_dir,
        max_upload_bytes=settings.max_upload_mb * 1024 * 1024,
    )

    pipelines: dict[PipelineName, Pipeline] = {}
    if chat_model is not None and embeddings is not None:
        pipelines = build_pipelines(settings=settings, store=store, chat_model=chat_model, embeddings=embeddings, nli=nli)
        if settings.preload_nli:
            nli.preload(pipeline_config.verification.nli_model)

    return ServiceContainer(
        settings=settings,
        pipeline_config=pipeline_config,
        engine=engine,
        sessionmaker=sessionmaker,
        store=store,
        nli=nli,
        documents=documents,
        queries=QueryService(
            pipelines,
            sessionmaker,
            demo_cache=DemoCache(
                settings.demo_cache_dir,
                sessionmaker,
                chat_model=settings.chat_model,
                embedding_model=settings.embedding_model,
                enabled=settings.demo_cache_enabled,
            ),
        ),
        pipelines=pipelines,
    )
