"""Vector store interface and its ChromaDB implementation.

Pipelines depend only on the ``VectorStore`` protocol, so FAISS, pgvector or another backend can
be added by implementing these four methods. Embeddings are computed outside the store, which keeps
the embedding model identical across the baseline and the proposed pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.schemas.documents import Chunk

_COLLECTION_PREFIX = "rag_"
_ADD_BATCH_SIZE = 500
_TIE_MARGIN = 5


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float  # cosine similarity, higher is more similar


class VectorStore(Protocol):
    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...

    def search(
        self, corpus: str, query_embedding: Sequence[float], k: int, where: dict[str, Any] | None = None
    ) -> list[SearchHit]: ...

    def delete_document(self, corpus: str, doc_id: str) -> int: ...

    def count(self, corpus: str) -> int: ...


class ChromaVectorStore:
    def __init__(self, persist_dir: Path | None) -> None:
        settings = ChromaSettings(anonymized_telemetry=False)
        if persist_dir is None:
            self._client = chromadb.EphemeralClient(settings=settings)
        else:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)

    def _collection(self, corpus: str):  # type: ignore[no-untyped-def]
        return self._client.get_or_create_collection(
            name=f"{_COLLECTION_PREFIX}{corpus}",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return
        corpora = {c.corpus for c in chunks}
        if len(corpora) != 1:
            raise ValueError(f"add_chunks expects chunks from a single corpus, got {sorted(corpora)}")
        collection = self._collection(chunks[0].corpus)
        for start in range(0, len(chunks), _ADD_BATCH_SIZE):
            batch = chunks[start : start + _ADD_BATCH_SIZE]
            collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=[list(e) for e in embeddings[start : start + _ADD_BATCH_SIZE]],
                documents=[c.text for c in batch],
                metadatas=[c.vector_metadata() for c in batch],
            )

    def search(
        self, corpus: str, query_embedding: Sequence[float], k: int, where: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        collection = self._collection(corpus)
        available = collection.count()
        if available == 0 or k <= 0:
            return []
        # A few extra candidates let equal scores be ordered by chunk id, so results are reproducible
        # even when the index returns ties in arbitrary order.
        result = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(k + _TIE_MARGIN, available),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        ):
            # Chroma's cosine space returns distance = 1 - cosine similarity.
            hits.append(SearchHit(chunk=Chunk.from_vector_record(chunk_id, text, dict(meta)), score=1.0 - distance))
        hits.sort(key=lambda h: (-round(h.score, 6), h.chunk.chunk_id))
        return hits[:k]

    def delete_document(self, corpus: str, doc_id: str) -> int:
        collection = self._collection(corpus)
        ids = collection.get(where={"doc_id": doc_id}, include=[])["ids"]
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    def count(self, corpus: str) -> int:
        return self._collection(corpus).count()
