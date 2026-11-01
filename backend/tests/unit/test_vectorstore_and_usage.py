from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.core.config import ModelPrice
from app.core.usage import UsageTracker
from app.schemas.documents import Chunk
from app.vectorstore.store import ChromaVectorStore
from tests.fakes import HashingEmbeddings


def _chunk(chunk_id: str, doc_id: str, text: str, corpus: str = "c1", page: int | None = None) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, corpus=corpus, filename=f"{doc_id}.txt", page=page, chunk_index=0, text=text)


class TestChromaVectorStore:
    @pytest.fixture(autouse=True)
    def _store(self, tmp_path: Path) -> None:
        self.embeddings = HashingEmbeddings()
        self.store = ChromaVectorStore(persist_dir=tmp_path / "chroma")
        chunks = [
            _chunk("a", "doc1", "photosynthesis converts light into chemical energy", page=2),
            _chunk("b", "doc1", "mitochondria produce cellular energy"),
            _chunk("c", "doc2", "the french revolution began in 1789"),
        ]
        self.store.add_chunks(chunks, self.embeddings.embed_documents([c.text for c in chunks]))

    def test_search_ranks_by_similarity_and_restores_metadata(self) -> None:
        hits = self.store.search("c1", self.embeddings.embed_query("how does photosynthesis use light"), k=2)
        assert hits[0].chunk.chunk_id == "a"
        assert hits[0].chunk.page == 2 and hits[1].chunk.page is None
        assert hits[0].score > hits[1].score
        assert -1.0 <= hits[1].score <= 1.0

    def test_k_larger_than_collection(self) -> None:
        assert len(self.store.search("c1", self.embeddings.embed_query("energy"), k=50)) == 3

    def test_delete_document_removes_only_its_chunks(self) -> None:
        assert self.store.delete_document("c1", "doc1") == 2
        assert self.store.count("c1") == 1
        assert self.store.delete_document("c1", "doc1") == 0

    def test_corpora_are_isolated(self) -> None:
        assert self.store.count("other") == 0
        assert self.store.search("other", self.embeddings.embed_query("energy"), k=3) == []


class TestUsageTracker:
    PRICING = {"chat": ModelPrice(input_per_million=1.0, output_per_million=2.0), "emb": ModelPrice(input_per_million=0.5)}

    def test_reported_usage_and_cost(self) -> None:
        tracker = UsageTracker(self.PRICING)
        message = AIMessage(content="hi", usage_metadata={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500})
        tracker.record_chat("generation", "chat", message, prompt_text="ignored")
        summary = tracker.summary()
        assert summary.input_tokens == 1000 and summary.output_tokens == 500
        assert summary.stages[0].estimated is False
        assert abs(summary.cost_usd - (1000 * 1.0 + 500 * 2.0) / 1e6) < 1e-12

    def test_missing_usage_is_estimated_and_flagged(self) -> None:
        tracker = UsageTracker(self.PRICING)
        tracker.record_chat("generation", "chat", AIMessage(content="four words of output"), prompt_text="a prompt")
        tracker.record_embedding("query_embedding", "emb", ["some query text"])
        summary = tracker.summary()
        assert all(s.estimated for s in summary.stages)
        assert summary.input_tokens > 0 and summary.cost_usd is not None

    def test_unknown_model_makes_total_cost_unknown(self) -> None:
        tracker = UsageTracker(self.PRICING)
        tracker.record_embedding("query_embedding", "unpriced-model", ["text"])
        assert tracker.summary().cost_usd is None
