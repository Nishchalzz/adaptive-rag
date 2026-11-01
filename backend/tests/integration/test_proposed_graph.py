"""The proposed LangGraph pipeline end to end with real vector store and orchestration (LLM and NLI faked)."""

from pathlib import Path

import pytest

from app.core.config import PipelineConfig, Settings, VerificationConfig
from app.core.container import build_pipelines
from app.generation.prompts import GENERATION_SYSTEM_PROMPT
from app.schemas.documents import Chunk
from app.schemas.query import PipelineName
from app.vectorstore.store import ChromaVectorStore
from app.verification.nli import NLIRegistry
from tests.fakes import HALLUCINATION, HashingEmbeddings, OverlapNLI, ScriptedChatModel, research_responder

FACTS = [
    "The Retrieval and Reasoning Lab is led by Lena Park. It was founded in 2016.",
    "Project Veritas is run by the Retrieval and Reasoning Lab and started in 2022.",
    "The Retrieval and Reasoning Lab is located on the third floor of Aster Hall.",
    "The Grid Storage Lab commissioned a pilot storage plant with a capacity of 4 megawatt-hours in 2023.",
    "Prof. Anika Bergstrom has been dean of the School of Energy Systems since 2016.",
    "The Fenwick Library opened in 1994 and holds about 310,000 volumes.",
    "Aster Hall was completed in 2003 and has six floors.",
    "The Neural Interfaces Lab was founded in 2018 and is led by Dr. Kofi Mensah.",
]


def _pipelines(tmp_path: Path, responder):
    embeddings = HashingEmbeddings()
    store = ChromaVectorStore(tmp_path / "chroma")
    chunks = [Chunk(chunk_id=f"d{i}:00000", doc_id=f"d{i}", corpus="kb", filename=f"fact{i}.md", chunk_index=0, text=t) for i, t in enumerate(FACTS)]
    store.add_chunks(chunks, embeddings.embed_documents(FACTS))
    registry = NLIRegistry()
    registry.register(OverlapNLI(name=VerificationConfig().nli_model))
    chat = ScriptedChatModel(responder=responder)
    settings = Settings(_env_file=None, chat_model="fake-chat", embedding_model="fake-emb", openai_api_key=None)
    pipelines = build_pipelines(settings=settings, store=store, chat_model=chat, embeddings=embeddings, nli=registry)
    return pipelines, chat


@pytest.fixture
def setup(tmp_path: Path):
    return _pipelines(tmp_path, research_responder())


async def test_unsupported_claim_triggers_refinement_and_is_fixed(setup) -> None:
    pipelines, chat = setup
    response = await pipelines[PipelineName.PROPOSED].run("Who is the Retrieval and Reasoning Lab led by?", "kb", PipelineConfig())

    assert response.complexity is not None and response.complexity.label == "simple"
    assert response.retrieval.strategy == "adaptive"
    assert response.retrieval.k_requested == response.complexity.k

    first, second = response.attempts
    assert first.verification.num_unsupported == 1 and not first.verification.passed
    assert any(c.text.startswith("The institute also operates") for c in first.verification.claims if c.status.value == "unsupported")
    assert second.verification.passed and response.selected_attempt == 1
    assert len(response.refinements) == 1 and response.refinements[0].targeted_claims == [HALLUCINATION]
    assert "Verification feedback" in chat.calls_for(GENERATION_SYSTEM_PROMPT)[1][-1].content

    assert HALLUCINATION not in response.answer and "Lena Park" in response.answer
    assert response.uncertainty is not None and not response.uncertainty.flagged
    assert response.citations and response.citations[0].filename == "fact0.md"
    assert all(s.status.value == "supported" for s in response.sentences)
    assert {"analyze_query", "retrieval", "generation", "claim_extraction", "verification", "refinement", "total"} <= set(response.timings_ms)
    assert response.attempts[1].cumulative_latency_ms >= response.attempts[0].cumulative_latency_ms
    assert response.variant == {"adaptive": True, "decompose": True, "verify": True}


async def test_persistent_hallucination_is_removed_and_flagged(tmp_path: Path) -> None:
    pipelines, _ = _pipelines(tmp_path, research_responder(always_hallucinate=True))
    config = PipelineConfig().with_overrides({"verification": {"max_retries": 1}})
    response = await pipelines[PipelineName.PROPOSED].run("Who is the Retrieval and Reasoning Lab led by?", "kb", config)

    assert len(response.attempts) == 2 and not any(a.verification.passed for a in response.attempts)
    assert HALLUCINATION not in response.answer
    assert response.uncertainty.flagged and response.uncertainty.removed_sentences
    assert HALLUCINATION in response.uncertainty.unsupported_claims[0]


async def test_all_flags_off_matches_the_baseline_exactly(setup) -> None:
    pipelines, chat = setup
    question = "When was Aster Hall completed?"
    baseline = await pipelines[PipelineName.BASELINE].run(question, "kb", PipelineConfig())
    off = PipelineConfig().with_overrides({"proposed": {"adaptive": False, "verify": False}})
    graph = await pipelines[PipelineName.PROPOSED].run(question, "kb", off)

    assert [c.chunk_id for c in graph.retrieval.chunks] == [c.chunk_id for c in baseline.retrieval.chunks]
    assert [c.score for c in graph.retrieval.chunks] == pytest.approx([c.score for c in baseline.retrieval.chunks])
    prompts = chat.calls_for(GENERATION_SYSTEM_PROMPT)
    assert [m.content for m in prompts[0]] == [m.content for m in prompts[1]]  # identical generation prompt
    assert graph.answer == baseline.answer
    assert graph.verification is None and graph.complexity is None and len(graph.attempts) == 1


async def test_verification_only_variant_uses_fixed_k(setup) -> None:
    pipelines, _ = setup
    config = PipelineConfig().with_overrides({"proposed": {"adaptive": False}})
    response = await pipelines[PipelineName.PROPOSED].run("Who is the Retrieval and Reasoning Lab led by?", "kb", config)
    assert response.retrieval.strategy == "fixed_top_k" and response.retrieval.k_requested == 5
    assert response.complexity is None and response.verification is not None


async def test_adaptive_only_variant_skips_verification(setup) -> None:
    pipelines, chat = setup
    config = PipelineConfig().with_overrides({"proposed": {"verify": False}})
    response = await pipelines[PipelineName.PROPOSED].run(
        "Where is the lab that runs Project Veritas and who leads it?", "kb", config
    )
    assert response.complexity.label in {"moderate", "complex"}
    assert len(response.retrieval.queries) > 1
    assert response.verification is None and response.uncertainty is None
    assert HALLUCINATION in response.answer  # nothing removes it without verification


async def test_zero_retries_stops_after_first_attempt(setup) -> None:
    pipelines, _ = setup
    config = PipelineConfig().with_overrides({"verification": {"max_retries": 0}})
    response = await pipelines[PipelineName.PROPOSED].run("Who is the Retrieval and Reasoning Lab led by?", "kb", config)
    assert len(response.attempts) == 1 and response.refinements == []
    assert response.uncertainty.flagged


async def test_stream_reports_nodes_in_order(setup) -> None:
    pipelines, _ = setup
    nodes = []
    async for event in pipelines[PipelineName.PROPOSED].stream("Who is the Retrieval and Reasoning Lab led by?", "kb", PipelineConfig()):
        nodes.append(event["node"] if event["type"] == "step" else "RESULT")
    assert nodes[:4] == ["analyze_query", "select_depth", "retrieve", "assess_confidence"]
    assert nodes[-2:] == ["finalize", "RESULT"]
    assert nodes.count("generate") == 2 and "refine" in nodes
