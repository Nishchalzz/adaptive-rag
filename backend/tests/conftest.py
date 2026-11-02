from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, VerificationConfig
from app.main import create_app
from app.verification.nli import NLIRegistry
from tests.fakes import HashingEmbeddings, OverlapNLI, ScriptedChatModel, research_responder


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        log_level="WARNING",
        data_dir=tmp_path / "data",
        experiments_dir=tmp_path / "experiments",
        openai_api_key=None,
        database_url=None,
        preload_nli=False,
        seed_demo_corpus=False,
        demo_cache_dir=tmp_path / "demo_cache",
    )


@pytest.fixture
def embeddings() -> HashingEmbeddings:
    return HashingEmbeddings()


@pytest.fixture
def chat_model() -> ScriptedChatModel:
    return ScriptedChatModel(responder=research_responder())


@pytest.fixture
def nli() -> OverlapNLI:
    # Registered under the default checkpoint name so tests exercise the production configuration.
    return OverlapNLI(name=VerificationConfig().nli_model)


@pytest.fixture
def nli_registry(nli: OverlapNLI) -> NLIRegistry:
    registry = NLIRegistry()
    registry.register(nli)
    return registry


@pytest.fixture
def client(
    settings: Settings, chat_model: ScriptedChatModel, embeddings: HashingEmbeddings, nli_registry: NLIRegistry
) -> Iterator[TestClient]:
    app = create_app(settings, chat_model=chat_model, embeddings=embeddings, nli=nli_registry)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_llm(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
