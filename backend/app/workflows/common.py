"""Response assembly shared by all pipelines, so every pipeline reports results in the same shape."""

from collections.abc import Sequence

from app.generation.citations import ParsedCitations
from app.schemas.query import CitedSentence, Citation, RetrievedChunk
from app.vectorstore.store import SearchHit


def to_retrieved_chunks(hits: Sequence[SearchHit], retrieved_by: str = "original_query") -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=h.chunk.chunk_id,
            doc_id=h.chunk.doc_id,
            filename=h.chunk.filename,
            page=h.chunk.page,
            chunk_index=h.chunk.chunk_index,
            text=h.chunk.text,
            score=h.score,
            rank=rank,
            citation_number=rank,
            retrieved_by=retrieved_by,
        )
        for rank, h in enumerate(hits, start=1)
    ]


def build_citations(parsed: ParsedCitations, chunks: Sequence[RetrievedChunk]) -> list[Citation]:
    by_number = {c.citation_number: c for c in chunks}
    return [
        Citation(
            number=n,
            chunk_id=by_number[n].chunk_id,
            doc_id=by_number[n].doc_id,
            filename=by_number[n].filename,
            page=by_number[n].page,
        )
        for n in parsed.cited
    ]


def build_sentences(parsed: ParsedCitations) -> list[CitedSentence]:
    return [CitedSentence(index=s.index, text=s.text, citation_numbers=s.citation_numbers) for s in parsed.sentences]
