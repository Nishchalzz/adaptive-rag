from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("corpus", "sha256", name="uq_documents_corpus_sha256"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    corpus: Mapped[str] = mapped_column(String(64), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    file_type: Mapped[str] = mapped_column(String(16))
    file_size: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    storage_path: Mapped[str] = mapped_column(String(1024))
    status: Mapped[str] = mapped_column(String(16), index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    num_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    # Recorded per document so a corpus mixing chunking settings can be detected when analysing results.
    chunk_size: Mapped[int] = mapped_column(Integer)
    chunk_overlap: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class QueryLogRow(Base):
    """One row per answered query. The scalar columns are the fields required for analysis;
    ``trace`` holds the complete QueryResponse for the UI and for later inspection."""

    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    pipeline: Mapped[str] = mapped_column(String(32), index=True)
    corpus: Mapped[str] = mapped_column(String(64))
    question: Mapped[str] = mapped_column(Text)
    complexity_label: Mapped[str | None] = mapped_column(String(16), nullable=True)
    complexity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    k_selected: Mapped[int] = mapped_column(Integer)
    num_retrieved: Mapped[int] = mapped_column(Integer)
    retrieval_latency_ms: Mapped[float] = mapped_column(Float)
    total_latency_ms: Mapped[float] = mapped_column(Float)
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    trace: Mapped[dict[str, Any]] = mapped_column(JSON)
