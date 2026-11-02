"""Replay cache end to end: entries recorded by scripts.build_demo_cache from live (faked-LLM) runs."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import PipelineConfig, Settings
from app.core.container import ServiceContainer
from app.main import create_app
from app.workflows.demo_cache import DemoCache
from scripts.build_demo_cache import AUDIT_OVERRIDES, CacheBuildError, build, print_table
from tests.fakes import HashingEmbeddings, ScriptedChatModel, research_responder

LAB = b"The Retrieval and Reasoning Lab is led by Lena Park. It was founded in 2016."
EXTRA = b"Aster Hall was completed in 2003 and has six floors."
QUESTION = "Who is the Retrieval and Reasoning Lab led by?"
QUESTIONS = [{"category": "simple", "question": QUESTION}]


def _upload(client: TestClient, name: str, data: bytes) -> str:
    response = client.post("/api/documents?wait=true", files={"file": (name, data)})
    assert response.status_code == 201, response.text
    return response.json()["id"]


@pytest.fixture
def cached_client(client: TestClient) -> Iterator[TestClient]:
    _upload(client, "labs.md", LAB)
    yield client


@pytest.fixture
def cache_dir(cached_client: TestClient) -> Path:
    return cached_client.app.state.container.settings.demo_cache_dir


@pytest.fixture
def recorded(cached_client: TestClient, cache_dir: Path) -> list[dict]:
    return build(cached_client, QUESTIONS, cache_dir, workers=1)


def _post(client: TestClient, **body) -> dict:
    response = client.post("/api/query", json={"question": QUESTION, **body})
    assert response.status_code == 200, response.text
    return response.json()


def test_build_records_live_runs_for_the_three_ui_request_shapes(recorded, cache_dir: Path) -> None:
    assert [r["shape"] for r in recorded] == ["baseline", "proposed", "audit"]
    assert len(list(cache_dir.glob("*.json"))) == 3
    entry = json.loads(next(p for p in cache_dir.glob("*.json") if json.loads(p.read_text())["overrides"]).read_text())
    assert entry["overrides"] == AUDIT_OVERRIDES and entry["response"]["cached"] is False
    print_table(recorded)  # smoke test of the summary output


def test_replay_returns_the_recorded_run_without_calling_the_llm(
    cached_client: TestClient, chat_model: ScriptedChatModel, recorded
) -> None:
    live = _post(cached_client, pipeline="proposed", use_cache=False)  # not the recorded run, but same shape
    assert live["cached"] is False
    calls_before = len(chat_model.calls)

    replay = _post(cached_client, pipeline="proposed")
    assert len(chat_model.calls) == calls_before  # served without any model call
    assert replay["cached"] is True and replay["served_ms"] is not None and replay["served_ms"] < 2000

    stored_entry = next(
        json.loads(p.read_text())
        for p in cached_client.app.state.container.settings.demo_cache_dir.glob("*.json")
        if json.loads(p.read_text())["pipeline"] == "proposed" and not json.loads(p.read_text())["overrides"]
    )["response"]
    assert replay["cached_from"].replace("Z", "+00:00") == stored_entry["created_at"].replace("Z", "+00:00")
    assert replay["query_id"] != stored_entry["query_id"]
    for field in ("answer", "total_latency_ms", "timings_ms", "usage", "attempts", "verification", "retrieval", "complexity"):
        assert replay[field] == stored_entry[field], field  # original latency/tokens/cost are kept


def test_baseline_and_audit_shapes_replay_too(cached_client: TestClient, recorded) -> None:
    assert _post(cached_client, pipeline="baseline")["cached"] is True
    assert _post(cached_client, pipeline="proposed", overrides=AUDIT_OVERRIDES)["cached"] is True


def test_different_overrides_pipeline_or_question_are_misses(
    cached_client: TestClient, chat_model: ScriptedChatModel, recorded
) -> None:
    assert _post(cached_client, pipeline="baseline", overrides={"baseline": {"top_k": 2}})["cached"] is False
    assert _post(cached_client, pipeline="proposed", overrides={"verification": {"max_retries": 1}})["cached"] is False
    other = cached_client.post("/api/query", json={"question": "Who leads Aster Hall?", "pipeline": "baseline"})
    assert other.json()["cached"] is False


def test_use_cache_false_bypasses_the_cache(cached_client: TestClient, chat_model: ScriptedChatModel, recorded) -> None:
    before = len(chat_model.calls)
    response = _post(cached_client, pipeline="baseline", use_cache=False)
    assert response["cached"] is False and len(chat_model.calls) > before


def test_adding_or_deleting_a_document_invalidates_the_cache(cached_client: TestClient, recorded) -> None:
    assert _post(cached_client, pipeline="baseline")["cached"] is True
    extra_id = _upload(cached_client, "campus.md", EXTRA)
    assert _post(cached_client, pipeline="baseline")["cached"] is False  # corpus changed -> answered live
    assert cached_client.delete(f"/api/documents/{extra_id}").status_code == 204
    assert _post(cached_client, pipeline="baseline")["cached"] is True  # same documents again -> valid again

    lab_id = cached_client.get("/api/documents").json()["documents"][0]["id"]
    cached_client.delete(f"/api/documents/{lab_id}")
    _upload(cached_client, "labs-changed.md", LAB + b" It has 14 members.")
    assert _post(cached_client, pipeline="baseline")["cached"] is False  # different document content


def test_changed_models_or_config_invalidate_entries(cached_client: TestClient, recorded) -> None:
    container: ServiceContainer = cached_client.app.state.container
    settings: Settings = container.settings

    def cache(**changes) -> DemoCache:
        args = {"chat_model": settings.chat_model, "embedding_model": settings.embedding_model, "enabled": True, **changes}
        return DemoCache(settings.demo_cache_dir, container.sessionmaker, **args)

    config = container.pipeline_config
    corpus = settings.default_corpus

    async def lookup(c: DemoCache, cfg: PipelineConfig):
        return await c.lookup(QUESTION, "baseline", corpus, None, cfg)

    import asyncio

    assert asyncio.run(lookup(cache(), config)) is not None
    assert asyncio.run(lookup(cache(chat_model="another-model"), config)) is None
    assert asyncio.run(lookup(cache(embedding_model="another-embedding"), config)) is None
    assert asyncio.run(lookup(cache(enabled=False), config)) is None
    assert asyncio.run(lookup(cache(), config.with_overrides({"baseline": {"top_k": 7}}))) is None


def test_corrupt_entry_falls_back_to_a_live_answer(cached_client: TestClient, cache_dir: Path, recorded) -> None:
    for path in cache_dir.glob("*.json"):
        path.write_text("{ not json", encoding="utf-8")
    assert _post(cached_client, pipeline="baseline")["cached"] is False


def test_replays_are_logged_like_normal_queries(cached_client: TestClient, recorded) -> None:
    before = len(cached_client.get("/api/queries").json())
    replay = _post(cached_client, pipeline="proposed")
    logs = cached_client.get("/api/queries").json()
    assert len(logs) == before + 1 and logs[0]["id"] == replay["query_id"]
    assert logs[0]["complexity_label"] == "simple" and logs[0]["k_selected"] == replay["retrieval"]["k_requested"]
    stored = cached_client.get(f"/api/queries/{replay['query_id']}").json()
    assert stored["cached"] is True and stored["cached_from"] is not None


def test_streaming_a_replay_emits_steps_then_the_cached_result(
    cached_client: TestClient, chat_model: ScriptedChatModel, recorded
) -> None:
    calls_before = len(chat_model.calls)
    with cached_client.stream("POST", "/api/query/stream", json={"question": QUESTION}) as response:
        assert response.status_code == 200
        events = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]
    nodes = [e["node"] for e in events if e["type"] == "step"]
    assert nodes[:3] == ["analyze_query", "select_depth", "retrieve"] and nodes[-1] == "finalize"
    assert {"generate", "extract_claims", "verify"} <= set(nodes)
    assert events[-1]["type"] == "result" and events[-1]["response"]["cached"] is True
    assert len(chat_model.calls) == calls_before


def test_baseline_stream_replay_is_just_the_result(cached_client: TestClient, recorded) -> None:
    with cached_client.stream("POST", "/api/query/stream", json={"question": QUESTION, "pipeline": "baseline"}) as response:
        events = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["result"] and events[0]["response"]["cached"] is True


def test_build_refuses_an_empty_corpus(client: TestClient) -> None:
    with pytest.raises(CacheBuildError, match="no documents"):
        build(client, QUESTIONS, client.app.state.container.settings.demo_cache_dir, workers=1)


def test_build_never_records_a_replay(cached_client: TestClient, cache_dir: Path, recorded) -> None:
    # Even though valid entries now exist, the builder asks for live runs (use_cache=false).
    rows = build(cached_client, QUESTIONS, cache_dir, workers=1)
    assert len(rows) == 3
    for path in cache_dir.glob("*.json"):
        assert json.loads(path.read_text())["response"]["cached"] is False


def test_cache_dir_is_isolated_per_test(settings: Settings) -> None:
    assert "demo_cache" in str(settings.demo_cache_dir) and "seed" not in str(settings.demo_cache_dir)
