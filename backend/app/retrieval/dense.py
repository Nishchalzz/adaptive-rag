"""Dense retrieval primitives shared by the baseline and the adaptive retriever.

Both pipelines go through this class, so they use the same embedding model, the same vector store
and the same similarity scores; only the retrieval *policy* (how many chunks, for which queries)
differs between them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from langchain_core.embeddings import Embeddings

from app.vectorstore.store import SearchHit, VectorStore


@dataclass
class RetrievalResult:
    hits: list[SearchHit]
    k_requested: int
    embedding_latency_ms: float
    search_latency_ms: float

    @property
    def latency_ms(self) -> float:
        return self.embedding_latency_ms + self.search_latency_ms


class DenseRetriever:
    def __init__(self, store: VectorStore, embeddings: Embeddings) -> None:
        self._store = store
        self._embeddings = embeddings

    def corpus_size(self, corpus: str) -> int:
        return self._store.count(corpus)

    async def embed_query(self, text: str) -> tuple[list[float], float]:
        start = perf_counter()
        vector = await self._embeddings.aembed_query(text)
        return vector, (perf_counter() - start) * 1000

    async def embed_queries(self, texts: Sequence[str]) -> tuple[list[list[float]], float]:
        """Embed several queries in one request (a single query goes through ``embed_query``)."""
        if len(texts) == 1:
            vector, ms = await self.embed_query(texts[0])
            return [vector], ms
        start = perf_counter()
        vectors = await self._embeddings.aembed_documents(list(texts))
        return vectors, (perf_counter() - start) * 1000

    async def search(self, corpus: str, query_embedding: Sequence[float], k: int) -> tuple[list[SearchHit], float]:
        start = perf_counter()
        hits = await asyncio.to_thread(self._store.search, corpus, query_embedding, k)
        return hits, (perf_counter() - start) * 1000

    async def retrieve(self, corpus: str, query: str, k: int) -> RetrievalResult:
        vector, embed_ms = await self.embed_query(query)
        hits, search_ms = await self.search(corpus, vector, k)
        return RetrievalResult(hits=hits, k_requested=k, embedding_latency_ms=embed_ms, search_latency_ms=search_ms)
