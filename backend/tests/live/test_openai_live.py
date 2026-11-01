"""Smoke test against the real OpenAI API. Deselected by default; run with:  pytest -m live

Uses the models from .env, costs a fraction of a cent, and writes only to a temporary directory.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

pytestmark = pytest.mark.live

DOCUMENT = (
    "The Kessler Library was founded in 1952 by the engineer Hanna Kessler. "
    "It holds 40,000 volumes on the history of computing and is located in Graz."
)


@pytest.fixture
def live_client(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data", database_url=None, log_level="WARNING")
    if not settings.llm_configured:
        pytest.skip("OPENAI_API_KEY not configured")
    with TestClient(create_app(settings)) as client:
        yield client


def test_baseline_answers_with_real_models(live_client: TestClient) -> None:
    upload = live_client.post("/api/documents?wait=true", files={"file": ("library.txt", DOCUMENT.encode())})
    assert upload.status_code == 201, upload.text

    response = live_client.post("/api/query", json={"question": "Who founded the Kessler Library and when?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "1952" in body["answer"]
    assert body["citations"], "expected the answer to cite the retrieved passage"
    generation = next(s for s in body["usage"]["stages"] if s["stage"] == "generation")
    assert generation["estimated"] is False and generation["input_tokens"] > 0
