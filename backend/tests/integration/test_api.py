"""End-to-end API tests: real FastAPI app, SQLite, ChromaDB and loaders; LLM provider and NLI faked."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.fakes import HashingEmbeddings, ScriptedChatModel, cite_first_passage, research_responder
from tests.helpers import make_docx, make_pdf

ARTICLE = (
    "Adaptive retrieval selects how many passages to retrieve for each question. "
    "Simple questions need few passages, while multi-hop questions need evidence from several documents."
)


def _upload(client: TestClient, name: str, data: bytes, corpus: str | None = None, wait: bool = True):
    form = {"corpus": corpus} if corpus else {}
    return client.post(f"/api/documents?wait={str(wait).lower()}", files={"file": (name, data)}, data=form)


def test_health_and_config(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["llm_configured"] is True
    config = client.get("/api/config").json()
    assert config["pipeline"]["baseline"]["top_k"] == 5
    assert set(config["pipelines"]) == {"baseline", "proposed"}
    assert "openai_api_key" not in str(config)


def test_upload_formats_list_and_delete(client: TestClient) -> None:
    pdf = _upload(client, "paper.pdf", make_pdf([ARTICLE, "Second page about citations."]))
    assert pdf.status_code == 201, pdf.text
    assert pdf.json()["status"] == "ready" and pdf.json()["num_pages"] == 2

    for name, data in [
        ("notes.docx", make_docx(["Evidence verification checks claims.", "It uses entailment."])),
        ("readme.md", b"# Title\nMarkdown **content** about vector databases."),
        ("plain.txt", b"Plain text about chunk overlap."),
    ]:
        response = _upload(client, name, data)
        assert response.status_code == 201, response.text
        assert response.json()["num_chunks"] >= 1 and response.json()["num_pages"] is None

    listing = client.get("/api/documents").json()
    assert listing["total"] == 4
    assert {d["chunk_size"] for d in listing["documents"]} == {1000}

    doc_id = pdf.json()["id"]
    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.get(f"/api/documents/{doc_id}").status_code == 404
    assert client.get("/api/documents").json()["total"] == 3


def test_background_ingestion_completes(client: TestClient) -> None:
    response = _upload(client, "bg.txt", ARTICLE.encode(), wait=False)
    assert response.status_code == 202 and response.json()["status"] == "processing"
    # TestClient runs background tasks before returning control
    assert client.get(f"/api/documents/{response.json()['id']}").json()["status"] == "ready"


def test_upload_validation(client: TestClient) -> None:
    assert _upload(client, "slides.pptx", b"data").status_code == 415
    assert _upload(client, "empty.txt", b"").status_code == 422
    assert _upload(client, "a.txt", b"hello", corpus="bad corpus!").status_code == 422
    assert _upload(client, "a.txt", ARTICLE.encode()).status_code == 201
    assert _upload(client, "copy.txt", ARTICLE.encode()).status_code == 409
    assert _upload(client, "copy.txt", ARTICLE.encode(), corpus="other").status_code == 201


def test_failed_ingestion_is_recorded(client: TestClient) -> None:
    response = _upload(client, "broken.pdf", b"%PDF-1.4 this is not really a pdf")
    assert response.status_code == 422
    assert response.json()["status"] == "failed" and response.json()["error"]


def test_baseline_query_end_to_end(tmp_path: Path, settings: Settings, embeddings: HashingEmbeddings, nli_registry) -> None:
    chat = ScriptedChatModel(responder=cite_first_passage)
    with TestClient(create_app(settings, chat_model=chat, embeddings=embeddings, nli=nli_registry)) as client:
        _upload(client, "paper.txt", ARTICLE.encode())
        _upload(client, "other.txt", b"Vector databases store embeddings for similarity search.")

        response = client.post("/api/query", json={"question": "How does adaptive retrieval choose passages?", "pipeline": "baseline"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["pipeline"] == "baseline"
        assert body["retrieval"]["k_requested"] == 5
        assert body["retrieval"]["num_retrieved"] == 2  # corpus holds only two chunks
        assert body["retrieval"]["chunks"][0]["filename"] == "paper.txt"
        assert body["citations"][0]["filename"] == "paper.txt"
        assert body["invalid_citations"] == [99]

        logs = client.get("/api/queries").json()
        assert len(logs) == 1 and logs[0]["k_selected"] == 5 and logs[0]["complexity_label"] is None
        assert client.get(f"/api/queries/{body['query_id']}").json()["answer"] == body["answer"]


def test_proposed_query_end_to_end(client: TestClient) -> None:
    _upload(client, "labs.md", b"The Retrieval and Reasoning Lab is led by Dr. Lena Park. It was founded in 2016.")
    _upload(client, "campus.md", b"Aster Hall was completed in 2003 and has six floors.")
    response = client.post("/api/query", json={"question": "Who leads the Retrieval and Reasoning Lab?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pipeline"] == "proposed"
    assert body["complexity"]["label"] == "simple"
    assert body["verification"]["passed"] is True
    assert len(body["attempts"]) == 2  # the scripted generator hallucinates once, then is corrected
    log = client.get("/api/queries").json()[0]
    assert log["complexity_label"] == "simple" and log["complexity_score"] == body["complexity"]["score"]


def test_streaming_endpoint_emits_steps_then_result(client: TestClient) -> None:
    _upload(client, "labs.md", b"The Retrieval and Reasoning Lab is led by Dr. Lena Park. It was founded in 2016.")
    with client.stream("POST", "/api/query/stream", json={"question": "Who leads the Retrieval and Reasoning Lab?"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]
    assert events[0] == {"type": "step", "node": "analyze_query", "summary": events[0]["summary"]}
    assert events[-1]["type"] == "result" and events[-1]["response"]["verification"]["passed"]
    assert client.get("/api/queries").json()[0]["pipeline"] == "proposed"


def test_streaming_errors_before_first_event_are_http_errors(client: TestClient) -> None:
    response = client.post("/api/query/stream", json={"question": "Anything at all?"})
    assert response.status_code == 409  # empty corpus


def test_query_overrides_and_errors(client: TestClient) -> None:
    assert client.post("/api/query", json={"question": "Anything here?"}).status_code == 409  # empty corpus
    _upload(client, "paper.txt", ARTICLE.encode())
    ok = client.post("/api/query", json={"question": "What is adaptive retrieval?", "pipeline": "baseline", "overrides": {"baseline": {"top_k": 1}}})
    assert ok.json()["retrieval"]["k_requested"] == 1
    bad = client.post("/api/query", json={"question": "What is adaptive retrieval?", "overrides": {"baseline": {"k": 1}}})
    assert bad.status_code == 422
    assert client.get("/api/queries/doesnotexist").status_code == 404


def test_demo_corpus_is_seeded_once(settings: Settings, embeddings: HashingEmbeddings, nli_registry) -> None:
    seeded = settings.model_copy(update={"seed_demo_corpus": True})
    chat = ScriptedChatModel(responder=research_responder())
    with TestClient(create_app(seeded, chat_model=chat, embeddings=embeddings, nli=nli_registry)) as client:
        documents = client.get("/api/documents").json()
        assert documents["total"] == 3
        assert {d["status"] for d in documents["documents"]} == {"ready"}
        pdf = next(d for d in documents["documents"] if d["file_type"] == "pdf")
        assert pdf["num_pages"] == 3
        assert len(client.get("/api/demo/questions").json()["questions"]) >= 5
        client.delete(f"/api/documents/{pdf['id']}")
    with TestClient(create_app(seeded, chat_model=chat, embeddings=embeddings, nli=nli_registry)) as client:
        assert client.get("/api/documents").json()["total"] == 2  # deleted demo documents are not re-added


def test_experiments_api_reads_result_files(client: TestClient, settings: Settings) -> None:
    assert client.get("/api/experiments").json() == []
    suite = settings.experiments_dir / "results" / "demo_suite"
    (suite / "figures").mkdir(parents=True)
    (suite / "summary.json").write_text(json.dumps({"created_at": "2026-11-01", "variants": {"B": {}}, "dataset": {"num_test_questions": 3}}))
    (suite / "figures" / "plot.png").write_bytes(b"\x89PNG\r\n")
    listing = client.get("/api/experiments").json()
    assert listing == [{"name": "demo_suite", "created_at": "2026-11-01", "description": None, "num_questions": 3, "variants": ["B"]}]
    assert client.get("/api/experiments/demo_suite").json()["variants"] == {"B": {}}
    assert client.get("/api/experiments/demo_suite/figures/plot.png").status_code == 200
    assert client.get("/api/experiments/demo_suite/figures/..%2Fsummary.json").status_code == 404
    assert client.get("/api/experiments/missing").status_code == 404


def test_without_api_key_reads_work_and_llm_features_return_503(client_without_llm: TestClient) -> None:
    assert client_without_llm.get("/api/health").json()["llm_configured"] is False
    assert client_without_llm.get("/api/documents").json()["total"] == 0
    assert client_without_llm.get("/api/queries").json() == []
    upload = _upload(client_without_llm, "a.txt", b"text")
    assert upload.status_code == 503 and "OPENAI_API_KEY" in upload.json()["detail"]
    assert client_without_llm.post("/api/query", json={"question": "Hello there?"}).status_code == 503
