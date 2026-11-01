import json

import pytest

from app.core.config import AdaptiveConfig, ComplexityConfig
from app.core.usage import UsageTracker
from app.retrieval.complexity_classifier import ComplexityClassifier, label_for_score, lexical_features, select_k
from tests.fakes import ScriptedChatModel

COMPLEXITY = ComplexityConfig()
ADAPTIVE = AdaptiveConfig()


def _classifier(reply: str) -> tuple[ComplexityClassifier, ScriptedChatModel]:
    model = ScriptedChatModel(responder=lambda _: reply)
    return ComplexityClassifier(model, "fake"), model


def test_lexical_features_detect_comparison_and_nesting() -> None:
    simple = lexical_features("When was the institute founded?")
    nested = lexical_features("Who is the spouse of the director of the film that won the award in 1999?")
    comparison = lexical_features("Which is older, Aster Hall or Birch Pavilion?")
    assert simple["score"] < nested["score"]
    assert comparison["comparison_markers"] >= 1 and comparison["score"] > simple["score"]
    assert nested["nesting_markers"] >= 3


def test_length_alone_does_not_make_a_query_complex() -> None:
    long_simple = lexical_features("Could you please tell me in which particular calendar year the library was opened")
    assert long_simple["score"] < 0.33


@pytest.mark.parametrize(
    "score,label,expected_k",
    [(0.0, "simple", 1), (0.32, "simple", 3), (0.33, "moderate", 4), (0.65, "moderate", 7), (0.66, "complex", 8), (1.0, "complex", 12)],
)
def test_k_is_interpolated_within_the_label_band(score: float, label: str, expected_k: int) -> None:
    assert label_for_score(score, COMPLEXITY) == label
    assert select_k(score, label, COMPLEXITY, ADAPTIVE) == expected_k


async def test_hybrid_score_combines_llm_analysis_and_features() -> None:
    reply = json.dumps({
        "information_needs": ["Which lab runs Project Veritas?", "Where is that lab located?", "Who leads it?"],
        "reasoning_type": "bridge",
        "requires_multiple_sources": True,
        "complexity": "complex",
        "rationale": "multi-hop",
    })
    classifier, _ = _classifier(reply)
    usage = UsageTracker({})
    result = await classifier.assess("Where is the lab that runs Project Veritas, and who leads it?", COMPLEXITY, ADAPTIVE, usage)
    assert result.components["information_needs"] == pytest.approx(2 / 3, abs=1e-3)
    assert result.components["reasoning"] == 1.0 and result.components["multi_source"] == 1.0
    assert result.label == "complex" and 8 <= result.k <= 12
    assert result.fallback is False
    assert [s.stage for s in usage.records] == ["complexity_analysis"]


async def test_simple_lookup_gets_small_k() -> None:
    reply = json.dumps({
        "information_needs": ["When was the institute founded?"],
        "reasoning_type": "lookup",
        "requires_multiple_sources": False,
        "complexity": "simple",
        "rationale": "single fact",
    })
    classifier, _ = _classifier(reply)
    result = await classifier.assess("When was the institute founded?", COMPLEXITY, ADAPTIVE, UsageTracker({}))
    assert result.label == "simple" and 1 <= result.k <= 3


async def test_invalid_llm_output_falls_back_to_lexical_features() -> None:
    classifier, model = _classifier("this is not json")
    result = await classifier.assess("Who leads the lab?", COMPLEXITY, ADAPTIVE, UsageTracker({}))
    assert result.fallback is True
    assert result.score == result.lexical["score"]
    assert len(model.calls) == 2  # one repair attempt before giving up
