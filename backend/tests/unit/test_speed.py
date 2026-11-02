"""Latency optimisations must not change results: ordering, tiering and caching behaviour."""

import contextlib
from collections.abc import Sequence

from app.core.config import VerificationConfig
from app.core.usage import UsageTracker
from app.schemas.query import RetrievedChunk
from app.verification.claim_extractor import Claim
from app.verification.evidence_matcher import EvidenceMatcher
from app.verification.nli import HFEntailmentModel, NLIRegistry, NLIResult
from app.verification.verifier import EvidenceVerifier
from tests.fakes import HashingEmbeddings, OverlapNLI, ScriptedChatModel, research_responder


class _Probs:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def softmax(self, dim: int) -> "_Probs":
        return self

    def tolist(self) -> list[list[float]]:
        return self._rows


class _Logits:
    def __init__(self, rows: list[list[float]]) -> None:
        self.logits = _Probs(rows)


class _Encoded(dict):
    def to(self, _device: str) -> "_Encoded":
        return self


class _Torch:
    inference_mode = staticmethod(contextlib.nullcontext)


def test_nli_batches_by_length_but_returns_results_in_input_order() -> None:
    """Fake tokenizer/model: the 'probabilities' encode the premise length, so a mix-up would show."""
    batches: list[list[int]] = []

    def tokenizer(premises: list[str], hypotheses: list[str], **_: object) -> _Encoded:
        batches.append([len(p) for p in premises])
        return _Encoded(lengths=[len(p) for p in premises])

    def model(lengths: list[int]) -> _Logits:
        return _Logits([[float(n), 0.0, 0.0] for n in lengths])

    nli = HFEntailmentModel("stub", batch_size=2)
    nli._loaded, nli._torch, nli._tokenizer, nli._model = True, _Torch, tokenizer, model
    nli._index = {"entailment": 0, "neutral": 1, "contradiction": 2}

    premises = ["x" * 50, "x" * 5, "x" * 500, "x" * 20, "x" * 100]
    results = nli.score([(p, "claim") for p in premises])

    assert [r.entailment for r in results] == [50.0, 5.0, 500.0, 20.0, 100.0]  # input order restored
    assert batches == [[5, 20], [50, 100], [500]]  # but processed shortest first, so little padding


def _chunk(number: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{number}", doc_id=f"d{number}", filename=f"doc{number}.md", page=None, chunk_index=0,
        text=text, score=0.5, rank=number, citation_number=number,
    )


CHUNKS = [
    _chunk(1, "The Retrieval and Reasoning Lab was founded in 2016. It is led by Dr. Lena Park. It is in Aster Hall. "
              "It has fourteen members."),
    _chunk(2, "Aster Hall was completed in 2003 and has six floors."),
]


class CountingNLI(OverlapNLI):
    def __init__(self) -> None:
        super().__init__(name="counting")
        self.batches: list[list[tuple[str, str]]] = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIResult]:
        self.batches.append(list(pairs))
        return super().score(pairs)


def _verifier(nli: OverlapNLI) -> EvidenceVerifier:
    registry = NLIRegistry()
    registry.register(nli)
    return EvidenceVerifier(
        nli=registry,
        matcher=EvidenceMatcher(HashingEmbeddings(), "fake-emb"),
        judge_model=ScriptedChatModel(responder=research_responder()),
        judge_model_name="fake",
    )


async def test_long_pairs_are_only_scored_for_claims_no_window_supports() -> None:
    config = VerificationConfig(nli_model="counting")
    nli = CountingNLI()
    claims = [
        Claim(claim_id="c0", text="The Retrieval and Reasoning Lab was founded in 2016", sentence_index=0, citations=[1]),
        Claim(claim_id="c1", text="Aster Hall hosts a research cluster of fifty machines", sentence_index=1, citations=[2]),
    ]
    report = await _verifier(nli).verify(claims, CHUNKS, config, UsageTracker({}), {})

    first_stage, second_stage = nli.batches[0], nli.batches[1]
    assert len(first_stage) >= 2
    # the supported claim contributes only window pairs; the unsupported one adds whole-passage pairs
    assert all(h != claims[0].text for _, h in second_stage)
    assert any(h == claims[1].text for _, h in second_stage)
    assert report.claims[0].status.value == "supported"
    assert report.timings_ms["nli_pairs"] == sum(len(b) for b in nli.batches)
    assert {"embedding", "nli", "judge"} <= set(report.timings_ms)


async def test_window_embeddings_are_reused_across_queries() -> None:
    embeddings = HashingEmbeddings()
    matcher = EvidenceMatcher(embeddings, "fake-emb")
    first_usage, second_usage = UsageTracker({}), UsageTracker({})

    await matcher.build_index(CHUNKS, ["claim one"], window_sentences=3, cache={}, usage=first_usage)
    calls_after_first = embeddings.document_calls
    await matcher.build_index(CHUNKS, ["claim two"], window_sentences=3, cache={}, usage=second_usage)

    assert embeddings.document_calls == calls_after_first + 1  # only the new claim text is embedded
    assert second_usage.records[0].input_tokens < first_usage.records[0].input_tokens


async def test_prefetch_makes_the_later_index_embed_only_claims() -> None:
    embeddings = HashingEmbeddings()
    matcher = EvidenceMatcher(embeddings, "fake-emb")
    cache: dict[str, list[float]] = {}
    usage = UsageTracker({})

    await matcher.prefetch(CHUNKS, window_sentences=3, cache=cache, usage=usage)
    prefetched = len(cache)
    index = await matcher.build_index(CHUNKS, ["The lab was founded in 2016"], window_sentences=3, cache=cache, usage=usage)

    assert prefetched > 0 and len(cache) == prefetched + 1
    assert index.candidates(0, None, 1)[0].chunk.citation_number == 1
