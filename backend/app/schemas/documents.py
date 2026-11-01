from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, StringConstraints

# Corpus names double as vector-store collection suffixes, so keep them to a safe character set.
CorpusName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")]


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    corpus: str
    filename: str
    file_type: str
    file_size: int
    sha256: str
    status: DocumentStatus
    error: str | None = None
    num_pages: int | None = None
    num_chunks: int = 0
    chunk_size: int
    chunk_overlap: int
    created_at: datetime
    updated_at: datetime


class DocumentList(BaseModel):
    documents: list[DocumentRecord]
    total: int


class Chunk(BaseModel):
    """A retrievable unit of text. Carries everything needed to cite its origin."""

    chunk_id: str
    doc_id: str
    corpus: str
    filename: str
    page: int | None = None
    chunk_index: int
    char_start: int | None = None
    text: str

    def vector_metadata(self) -> dict[str, Any]:
        # Chroma metadata values must be str/int/float/bool, so optional fields are omitted when None.
        meta: dict[str, Any] = {
            "doc_id": self.doc_id,
            "corpus": self.corpus,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
        }
        if self.page is not None:
            meta["page"] = self.page
        if self.char_start is not None:
            meta["char_start"] = self.char_start
        return meta

    @classmethod
    def from_vector_record(cls, chunk_id: str, text: str, meta: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=chunk_id,
            doc_id=str(meta["doc_id"]),
            corpus=str(meta["corpus"]),
            filename=str(meta["filename"]),
            page=meta.get("page"),
            chunk_index=int(meta["chunk_index"]),
            char_start=meta.get("char_start"),
            text=text,
        )
