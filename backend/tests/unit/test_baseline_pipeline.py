from pathlib import Path

import pytest

from app.core.config import PipelineConfig
from app.core.errors import EmptyCorpusError
from app.generation.generator import AnswerGenerator
from app.generation.prompts import GENERATION_SYSTEM_PROMPT
from app.retrieval.dense import DenseRetriever
from app.schemas.documents import Chunk
from app.vectorstore.store import ChromaVectorStore
from app.workflows.baseline_pipeline import BaselineRAGPipeline
from tests.fakes import HashingEmbeddings, ScriptedChatModel, cite_first_passage

FACTS = [
    "The Eiffel Tower is located in Paris. It was completed in 1889.",
    "The Colosseum is an amphitheatre in Rome.",
    "Mount Everest is the highest mountain above sea level.",
    "The Amazon is the largest rainforest on Earth.",
    "The Great Wall of China is thousands of kilometres long.",
    "Tokyo is the capital city of Japan.",
    "The Nile is one of the longest rivers in the world.",
]


@pytest.fixture
def pipeline_parts(tmp_path: Path) -> tuple[BaselineRAGPipeline, ScriptedChatModel]:
    embeddings = HashingEmbeddings()
    store = ChromaVectorStore(tmp_path / "chroma")
    chunks = [
        Chunk(chunk_id=f"d{i}:00000", doc_id=f"d{i}", corpus="kb", filename=f"fact{i}.txt", chunk_index=0, text=t)
        for i, t in enumerate(FACTS)
    ]
    store.add_chunks(chunks, embeddings.embed_documents(FACTS))
    chat = ScriptedChatModel(responder=cite_first_passage)
    pipeline = BaselineRAGPipeline(
        DenseRetriever(store, embeddings), AnswerGenerator(chat, "fake-chat"), embedding_model="fake-embedding"
    )
    return pipeline, chat


async def test_baseline_retrieves_fixed_k_and_maps_citations(pipeline_parts) -> None:
    pipeline, chat = pipeline_parts
    response = await pipeline.run("When was the Eiffel Tower in Paris completed?", "kb", PipelineConfig())

    assert response.retrieval.strategy == "fixed_top_k"
    assert response.retrieval.k_requested == 5 and response.retrieval.num_retrieved == 5
    assert response.retrieval.chunks[0].doc_id == "d0"
    assert [c.citation_number for c in response.retrieval.chunks] == [1, 2, 3, 4, 5]

    assert response.answer.startswith("The Eiffel Tower is located in Paris [1].")
    assert [c.chunk_id for c in response.citations] == ["d0:00000"]
    assert response.invalid_citations == [99]
    assert [s.citation_numbers for s in response.sentences] == [[1], [99]]

    # one generation call, using the shared prompt and the configured answer budget
    assert len(chat.calls) == 1
    assert chat.calls[0][0].content == GENERATION_SYSTEM_PROMPT
    assert chat.call_kwargs[0]["max_tokens"] == 700
    assert response.usage.input_tokens > 0
    assert {s.stage for s in response.usage.stages} == {"query_embedding", "generation"}
    assert response.config["baseline"]["top_k"] == 5


async def test_baseline_k_is_configurable(pipeline_parts) -> None:
    pipeline, _ = pipeline_parts
    config = PipelineConfig().with_overrides({"baseline": {"top_k": 2}})
    response = await pipeline.run("Which city is the capital of Japan?", "kb", config)
    assert response.retrieval.num_retrieved == 2
    assert response.retrieval.chunks[0].doc_id == "d5"


async def test_empty_corpus_is_rejected_before_any_llm_call(pipeline_parts) -> None:
    pipeline, chat = pipeline_parts
    with pytest.raises(EmptyCorpusError):
        await pipeline.run("Anything?", "empty", PipelineConfig())
    assert chat.calls == []
