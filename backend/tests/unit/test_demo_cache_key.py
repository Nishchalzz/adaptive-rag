import json
from pathlib import Path

from app.schemas.query import PipelineName
from app.workflows.demo_cache import cache_key, normalise_question, save_entry


def test_question_normalisation_ignores_case_and_whitespace() -> None:
    assert normalise_question("  Who  LEADS\nthe lab? ") == "who leads the lab?"
    assert cache_key("Who leads the lab?", "proposed", "default", None) == cache_key(
        "  who  leads the LAB? ", PipelineName.PROPOSED, "default", {}
    )


def test_key_depends_on_pipeline_corpus_and_overrides_but_not_their_order() -> None:
    base = cache_key("q?", "proposed", "default", None)
    assert cache_key("q?", "baseline", "default", None) != base
    assert cache_key("q?", "proposed", "other", None) != base
    assert cache_key("q?", "proposed", "default", {"baseline": {"top_k": 3}}) != base
    a = {"proposed": {"adaptive": False, "verify": True}, "verification": {"max_retries": 0}}
    b = {"verification": {"max_retries": 0}, "proposed": {"verify": True, "adaptive": False}}
    assert cache_key("q?", "proposed", "default", a) == cache_key("q?", "proposed", "default", b)


def test_save_entry_stores_the_recorded_response_unchanged(tmp_path: Path) -> None:
    response = {"created_at": "2026-11-01T10:00:00Z", "answer": "A", "models": {}, "config": {}}
    path = save_entry(
        tmp_path, question="Q?", pipeline="baseline", corpus="default", overrides=None,
        corpus_fingerprint=["b", "a"], response=response,
    )
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == f"{cache_key('Q?', 'baseline', 'default', None)}.json"
    assert entry["response"] == response and entry["corpus_fingerprint"] == ["a", "b"]
    assert entry["recorded_at"] == "2026-11-01T10:00:00Z"
