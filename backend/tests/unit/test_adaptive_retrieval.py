from pathlib import Path

import pytest

from app.core.config import AdaptiveConfig
from app.retrieval.adaptive_retriever import AdaptiveRetriever, assess_confidence, build_queries, fuse
from app.retrieval.dense import DenseRetriever
from app.schemas.documents import Chunk
from app.schemas.pipeline import ComplexityAssessment
from app.vectorstore.store import ChromaVectorStore, SearchHit
from tests.fakes import HashingEmbeddings


def _hit(chunk_id: str, score: float) -> SearchHit:
    return SearchHit(Chunk(chunk_id=chunk_id, doc_id=chunk_id, corpus="c", filename="f", chunk_index=0, text=chunk_id), score)


def _assessment(needs: list[str]) -> ComplexityAssessment:
    return ComplexityAssessment(
        label="moderate", score=0.5, k=4, information_needs=needs, reasoning_type="bridge",
        requires_multiple_sources=True, llm_label="moderate", rationale="", components={}, lexical={}, latency_ms=0,
    )


def test_single_query_fusion_equals_plain_top_k() -> None:
    hits = [_hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7)]
    assert [f.hit.chunk.chunk_id for f in fuse([hits], 2, 60)] == ["a", "b"]


def test_fusion_guarantees_each_query_its_best_chunk() -> None:
    original = [_hit("a", 0.9), _hit("b", 0.85), _hit("c", 0.8)]
    sub1 = [_hit("b", 0.7), _hit("a", 0.6), _hit("c", 0.5)]
    sub2 = [_hit("z", 0.4), _hit("a", 0.3)]  # its best chunk would lose on RRF alone
    fused = fuse([original, sub1, sub2], 3, 60)
    ids = [f.hit.chunk.chunk_id for f in fused]
    assert "z" in ids and len(ids) == 3
    assert ids[0] == "a"  # highest fused rank first
    z = next(f for f in fused if f.hit.chunk.chunk_id == "z")
    assert z.query_index == 2


def test_build_queries_respects_decomposition_flag() -> None:
    assessment = _assessment(["Which lab runs Veritas?", "Where is that lab?"])
    assert build_queries("Q?", assessment, AdaptiveConfig()) == ["Q?", "Which lab runs Veritas?", "Where is that lab?"]
    assert build_queries("Q?", assessment, AdaptiveConfig(decompose=False)) == ["Q?"]
    assert build_queries("Q?", _assessment(["Q?"]), AdaptiveConfig()) == ["Q?"]


@pytest.fixture
def adaptive(tmp_path: Path) -> AdaptiveRetriever:
    embeddings = HashingEmbeddings()
    store = ChromaVectorStore(tmp_path / "chroma")
    texts = [
        "Project Veritas is run by the Retrieval and Reasoning Lab.",
        "The Retrieval and Reasoning Lab is located in Aster Hall.",
        "The Grid Storage Lab builds sodium-ion batteries.",
        "Birch Pavilion was completed in 1991.",
        "The Fenwick Library holds 310,000 volumes.",
    ]
    chunks = [Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", corpus="kb", filename="kb.md", chunk_index=i, text=t) for i, t in enumerate(texts)]
    store.add_chunks(chunks, embeddings.embed_documents(texts))
    return AdaptiveRetriever(DenseRetriever(store, embeddings))


async def test_decomposed_retrieval_covers_both_hops(adaptive: AdaptiveRetriever) -> None:
    assessment = _assessment(["Which lab runs Project Veritas?", "Where is the Retrieval and Reasoning Lab located?"])
    state = await adaptive.retrieve("kb", "Where is the lab that runs Project Veritas?", assessment, 2, AdaptiveConfig())
    assert len(state.queries) == 3
    assert {f.hit.chunk.chunk_id for f in state.selected} == {"c0", "c1"}
    confidence = assess_confidence(state, AdaptiveConfig(score_floor=0.0, score_ceiling=1.0, relevance_threshold=0.3))
    assert confidence.coverage == 1.0 and 0 < confidence.score <= 1


async def test_low_confidence_expansion_deepens_search(adaptive: AdaptiveRetriever) -> None:
    config = AdaptiveConfig(relevance_threshold=0.99, expansion_step=2)  # nothing counts as covered
    state = await adaptive.retrieve("kb", "Tell me about batteries", None, 1, config)
    confidence = assess_confidence(state, config)
    assert not confidence.sufficient and confidence.uncovered_queries == ["Tell me about batteries"]
    await adaptive.expand(state, confidence, config)
    assert state.k == 3 and state.expansions == 1 and len(state.selected) == 3
