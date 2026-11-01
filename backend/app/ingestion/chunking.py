"""Splits cleaned page text into overlapping chunks that keep their source location."""

from collections.abc import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import ChunkingConfig
from app.ingestion.loaders import PageText
from app.schemas.documents import Chunk

# Prefer paragraph, then line, then sentence boundaries before falling back to words.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


class Chunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=_SEPARATORS,
            keep_separator="end",
            add_start_index=True,
            strip_whitespace=True,
        )

    def split(self, *, doc_id: str, corpus: str, filename: str, pages: Sequence[PageText]) -> list[Chunk]:
        """Chunk each page separately so every chunk maps to exactly one page."""
        chunks: list[Chunk] = []
        for page in pages:
            if not page.text.strip():
                continue
            for piece in self._splitter.create_documents([page.text]):
                text = piece.page_content.strip()
                if not text:
                    continue
                index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}:{index:05d}",
                        doc_id=doc_id,
                        corpus=corpus,
                        filename=filename,
                        page=page.page,
                        chunk_index=index,
                        char_start=piece.metadata.get("start_index"),
                        text=text,
                    )
                )
        return chunks
