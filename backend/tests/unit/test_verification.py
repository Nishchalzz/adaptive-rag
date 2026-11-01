import json

import pytest

from app.core.config import VerificationConfig
from app.core.usage import UsageTracker
from app.schemas.pipeline import CitationStatus, SupportStatus, VerificationReport
from app.schemas.query import AttemptSummary, CitedSentence, RetrievedChunk
from app.verification.citation_verifier import citation_status, quote_in_passage
from app.verification.claim_extractor import Claim, ClaimExtractor, is_abstention
from app.verification.evidence_matcher import EvidenceMatcher, sentence_windows
from app.verification.finalize import NO_VERIFIED_ANSWER, annotate_sentences, build_uncertainty, select_best_attempt, strip_unsupported
from app.verification.nli import NLIRegistry
from app.verification.verifier import EvidenceVerifier
from tests.fakes import HashingEmbeddings, OverlapNLI, ScriptedChatModel, research_responder

CONFIG = VerificationConfig(nli_model="fake")


def _chunk(number: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{number}", doc_id=f"d{number}", filename=f"doc{number}.md", page=None, chunk_index=0,
        text=text, score=0.5, rank=number, citation_number=number,
    )


CHUNKS = [
    _chunk(1, "The Retrieval and Reasoning Lab was founded in 2016. It is led by Dr. Lena Park."),
    _chunk(2, "The Grid Storage Lab commissioned a pilot storage plant in 2023. The plant stores 4 megawatt-hours."),
    _chunk(3, "Aster Hall was completed in 2003 and has six floors."),
]


def _claim(text: str, citations: list[int], index: int = 0) -> Claim:
    return Claim(claim_id=f"c{index}", text=text, sentence_index=index, citations=citations)


def _verifier(responder=None) -> tuple[EvidenceVerifier, ScriptedChatModel]:
    registry = NLIRegistry()
    registry.register(OverlapNLI(name="fake"))
    judge = ScriptedChatModel(responder=responder or research_responder())
    verifier = EvidenceVerifier(
        nli=registry, matcher=EvidenceMatcher(HashingEmbeddings(), "fake-emb"), judge_model=judge, judge_model_name="fake"
    )
    return verifier, judge


async def _verify(claims: list[Claim], config: VerificationConfig = CONFIG, responder=None) -> tuple[VerificationReport, ScriptedChatModel]:
    verifier, judge = _verifier(responder)
    return await verifier.verify(claims, CHUNKS, config, UsageTracker({}), {}), judge


async def test_claim_supported_by_cited_passage() -> None:
    report, judge = await _verify([_claim("The Retrieval and Reasoning Lab was founded in 2016", [1])])
    claim = report.claims[0]
    assert claim.status is SupportStatus.SUPPORTED and claim.citation_status is CitationStatus.CORRECT
    assert claim.method == "nli" and claim.evidence.citation_number == 1
    assert report.passed and report.support_score == 1.0 and judge.calls == []


async def test_supported_by_other_passage_is_misattributed() -> None:
    report, _ = await _verify([_claim("Aster Hall was completed in 2003", [1])])
    claim = report.claims[0]
    assert claim.status is SupportStatus.SUPPORTED
    assert claim.citation_status is CitationStatus.MISATTRIBUTED and claim.suggested_citation == 3
    assert report.citation_accuracy == 0.0


async def test_uncited_claim_has_missing_citation() -> None:
    report, _ = await _verify([_claim("Aster Hall was completed in 2003", [])])
    assert report.claims[0].citation_status is CitationStatus.MISSING
    assert report.citation_accuracy is None


async def test_contradicted_claim_is_unsupported() -> None:
    report, _ = await _verify([_claim("The Retrieval and Reasoning Lab was founded in 1999", [1])])
    claim = report.claims[0]
    assert claim.status is SupportStatus.UNSUPPORTED and claim.failure == "contradicted"
    assert claim.citation_status is CitationStatus.INCORRECT
    assert not report.passed


async def test_unrelated_claim_goes_to_judge_and_is_unsupported() -> None:
    report, judge = await _verify([_claim("The institute operates an observatory on the Moon", [1])])
    claim = report.claims[0]
    assert claim.status is SupportStatus.UNSUPPORTED and claim.method == "llm_judge" and claim.failure == "no_evidence"
    assert len(judge.calls) == 1


async def test_judge_quote_that_is_not_in_the_passage_voids_support() -> None:
    def lying_judge(_messages):
        return json.dumps({"verdict": "supported", "passage": 1, "evidence_quote": "an invented sentence that is nowhere", "rationale": "x"})

    report, _ = await _verify([_claim("Lena Park also founded a second lab in Cedar Centre", [1])], responder=lying_judge)
    claim = report.claims[0]
    assert claim.method == "llm_judge" and claim.judge_quote_verified is False
    assert claim.status is SupportStatus.UNSUPPORTED


async def test_without_judge_ambiguous_claims_use_thresholds() -> None:
    config = VerificationConfig(nli_model="fake", use_llm_judge=False)
    # 4 of 5 content words overlap -> fake NLI returns entailment 0.5 (ambiguous)
    report, judge = await _verify([_claim("Lena Park leads the Retrieval Lab", [1])], config=config)
    assert report.claims[0].method == "nli_threshold" and report.claims[0].status is SupportStatus.PARTIAL
    assert judge.calls == []


async def test_no_claims_means_abstention() -> None:
    report, _ = await _verify([])
    assert report.abstained and report.passed and report.claims == []


def test_sentence_windows() -> None:
    text = "One. Two. Three. Four."
    assert sentence_windows(text, 3) == ["One. Two. Three.", "Two. Three. Four."]
    assert sentence_windows("Only one sentence", 3) == ["Only one sentence"]


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("The passages do not mention the founding date.", True),
        ("There is not enough information to answer.", True),
        ("He was not elected to the board [2].", False),
        ("The lab was not founded until 2016 [1].", False),
    ],
)
def test_abstention_detection(sentence: str, expected: bool) -> None:
    assert is_abstention(sentence) is expected


def test_quote_validation_tolerates_small_edits_only() -> None:
    passage = "The Grid Storage Lab commissioned a pilot storage plant in 2023."
    assert quote_in_passage("commissioned a pilot storage plant in 2023", passage)
    assert quote_in_passage("commissioned a pilot storage plant in 2023!", passage)
    assert not quote_in_passage("commissioned a nuclear reactor in 2023", passage)
    assert not quote_in_passage("2023", passage)  # too short to be evidence


def test_citation_status_rules() -> None:
    assert citation_status(SupportStatus.SUPPORTED, {1}, 1) is CitationStatus.CORRECT
    assert citation_status(SupportStatus.PARTIAL, {1}, 2) is CitationStatus.MISATTRIBUTED
    assert citation_status(SupportStatus.UNSUPPORTED, {1}, None) is CitationStatus.INCORRECT
    assert citation_status(SupportStatus.SUPPORTED, set(), 2) is CitationStatus.MISSING


async def test_claim_extractor_inherits_citations_from_sentences() -> None:
    model = ScriptedChatModel(responder=research_responder())
    sentences = [
        CitedSentence(index=0, text="The lab was founded in 2016 [1].", citation_numbers=[1]),
        CitedSentence(index=1, text="The passages do not contain the budget.", citation_numbers=[]),
    ]
    claims, fallback = await ClaimExtractor(model, "fake").extract(sentences, max_claims=5, max_tokens=300, usage=UsageTracker({}), attempt=0)
    assert not fallback
    assert [(c.text, c.citations) for c in claims] == [("The lab was founded in 2016", [1])]


# ----------------------------------------------------------------------------- finalize


def _report(statuses: list[SupportStatus], passed: bool, score: float) -> VerificationReport:
    from app.schemas.pipeline import ClaimVerification

    claims = [
        ClaimVerification(claim_id=f"c{i}", text=f"claim {i}", sentence_index=i, citations=[1], status=s,
                          citation_status=CitationStatus.CORRECT, method="nli")
        for i, s in enumerate(statuses)
    ]
    unsupported = sum(s is SupportStatus.UNSUPPORTED for s in statuses)
    return VerificationReport(
        claims=claims, support_score=score, citation_accuracy=1.0, num_supported=len(statuses) - unsupported,
        num_partial=0, num_unsupported=unsupported, passed=passed, abstained=False, extraction_fallback=False,
        nli_model="fake", latency_ms=0,
    )


def _attempt(index: int, report: VerificationReport | None) -> AttemptSummary:
    return AttemptSummary(
        attempt=index, answer="", sentences=[], invalid_citations=[], num_context_chunks=1, verification=report,
        latency_ms=0, cumulative_latency_ms=0, cumulative_input_tokens=0, cumulative_output_tokens=0, cumulative_cost_usd=None,
    )


def test_best_attempt_prefers_passed_then_support() -> None:
    attempts = [
        _attempt(0, _report([SupportStatus.UNSUPPORTED], False, 0.5)),
        _attempt(1, _report([SupportStatus.SUPPORTED], True, 1.0)),
        _attempt(2, _report([SupportStatus.UNSUPPORTED], False, 0.7)),
    ]
    assert select_best_attempt(attempts) == 1
    assert select_best_attempt([attempts[0], attempts[2]]) == 1


def test_unsupported_sentences_are_removed_and_reported() -> None:
    answer = "The lab was founded in 2016 [1]. It is on the Moon [1].\n- Extra point [2]"
    sentences = [
        CitedSentence(index=0, text="The lab was founded in 2016 [1].", citation_numbers=[1]),
        CitedSentence(index=1, text="It is on the Moon [1].", citation_numbers=[1]),
        CitedSentence(index=2, text="- Extra point [2]", citation_numbers=[2]),
    ]
    report = _report([SupportStatus.SUPPORTED, SupportStatus.UNSUPPORTED, SupportStatus.UNSUPPORTED], False, 0.33)
    annotated = annotate_sentences(sentences, report)
    text, removed = strip_unsupported(answer, annotated)
    assert text == "The lab was founded in 2016 [1]."
    assert removed == ["It is on the Moon [1].", "- Extra point [2]"]
    uncertainty = build_uncertainty(report, removed)
    assert uncertainty.flagged and "removed" in uncertainty.notice and len(uncertainty.unsupported_claims) == 2


def test_answer_with_no_supported_sentence_is_replaced() -> None:
    sentences = [CitedSentence(index=0, text="It is on the Moon [1].", citation_numbers=[1], status=SupportStatus.UNSUPPORTED)]
    text, removed = strip_unsupported("It is on the Moon [1].", sentences)
    assert text == NO_VERIFIED_ANSWER and removed == ["It is on the Moon [1]."]


def test_sentence_with_mixed_claims_is_partial_not_removed() -> None:
    from app.schemas.pipeline import ClaimVerification

    report = _report([SupportStatus.SUPPORTED], True, 1.0)
    report.claims.append(
        ClaimVerification(claim_id="x", text="x", sentence_index=0, citations=[1], status=SupportStatus.UNSUPPORTED,
                          citation_status=CitationStatus.INCORRECT, method="nli")
    )
    annotated = annotate_sentences([CitedSentence(index=0, text="A and B [1].", citation_numbers=[1])], report)
    assert annotated[0].status is SupportStatus.PARTIAL
