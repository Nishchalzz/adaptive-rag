"""Document lifecycle: upload -> load -> clean -> chunk -> embed -> store, plus listing and deletion."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from uuid import uuid4

from langchain_core.embeddings import Embeddings
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import ChunkingConfig
from app.core.errors import (
    AppError,
    DocumentBusyError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    FileTooLargeError,
    IngestionError,
    LLMNotConfiguredError,
)
from app.core.llm import MISSING_KEY_MESSAGE
from app.database.models import DocumentRow
from app.database.repositories import DocumentRepository
from app.ingestion.chunking import Chunker
from app.ingestion.cleaning import clean_text
from app.ingestion.loaders import PageText, detect_file_type, load_document
from app.schemas.documents import DocumentList, DocumentRecord, DocumentStatus
from app.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        store: VectorStore,
        embeddings: Embeddings | None,
        chunking: ChunkingConfig,
        upload_dir: Path,
        max_upload_bytes: int,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._store = store
        self._embeddings = embeddings
        self._chunking = chunking
        self._chunker = Chunker(chunking)
        self._upload_dir = upload_dir
        self._max_upload_bytes = max_upload_bytes

    async def register_upload(self, filename: str, data: bytes, corpus: str) -> DocumentRecord:
        """Validate and store the uploaded file, creating a ``processing`` record. Call ``ingest`` next."""
        if self._embeddings is None:
            raise LLMNotConfiguredError(MISSING_KEY_MESSAGE)
        safe_name = Path(filename).name
        file_type = detect_file_type(safe_name)
        if not data:
            raise IngestionError(f"{safe_name} is empty")
        if len(data) > self._max_upload_bytes:
            raise FileTooLargeError(f"{safe_name} exceeds the {self._max_upload_bytes // (1024 * 1024)} MB upload limit")
        sha256 = hashlib.sha256(data).hexdigest()

        async with self._sessionmaker() as session:
            repo = DocumentRepository(session)
            existing = await repo.get_by_hash(corpus, sha256)
            if existing is not None:
                if existing.status != DocumentStatus.FAILED.value:
                    raise DuplicateDocumentError(
                        f"This file is already in corpus '{corpus}' as '{existing.filename}' (id {existing.id})"
                    )
                await self._remove(session, existing)  # a failed earlier attempt is replaced by the new upload

            doc_id = uuid4().hex
            path = self._upload_dir / corpus / f"{doc_id}{Path(safe_name).suffix.lower()}"
            await asyncio.to_thread(_write_file, path, data)
            row = DocumentRow(
                id=doc_id,
                corpus=corpus,
                filename=safe_name,
                file_type=file_type,
                file_size=len(data),
                sha256=sha256,
                storage_path=str(path),
                status=DocumentStatus.PROCESSING.value,
                chunk_size=self._chunking.chunk_size,
                chunk_overlap=self._chunking.chunk_overlap,
            )
            try:
                await repo.add(row)
                await session.commit()
            except IntegrityError:
                path.unlink(missing_ok=True)
                raise DuplicateDocumentError(f"This file is already being added to corpus '{corpus}'") from None
        return DocumentRecord.model_validate(row)

    async def ingest(self, doc_id: str) -> DocumentRecord:
        """Extract, chunk, embed and index a registered document. Failures are recorded on the
        document instead of raised, because this usually runs as a background task."""
        if self._embeddings is None:
            raise LLMNotConfiguredError(MISSING_KEY_MESSAGE)
        row = await self._get_row(doc_id)
        try:
            raw_pages = await asyncio.to_thread(load_document, Path(row.storage_path), row.file_type)
            pages = [PageText(page=p.page, text=clean_text(p.text, row.file_type)) for p in raw_pages]
            chunks = self._chunker.split(doc_id=row.id, corpus=row.corpus, filename=row.filename, pages=pages)
            if not chunks:
                raise IngestionError(f"{row.filename} produced no text chunks after cleaning")
            vectors = await self._embeddings.aembed_documents([c.text for c in chunks])
            await asyncio.to_thread(self._store.add_chunks, chunks, vectors)
        except Exception as exc:  # noqa: BLE001 - any failure must end in a FAILED record, not a stuck one
            logger.exception("Ingestion failed for document %s (%s)", row.id, row.filename)
            await self._discard_vectors(row)
            message = exc.message if isinstance(exc, AppError) else f"{type(exc).__name__}: {exc}"
            return await self._update(doc_id, status=DocumentStatus.FAILED.value, error=message)

        paged = [p for p in pages if p.page is not None]
        logger.info("Ingested %s (%s): %d chunks", row.filename, row.id, len(chunks))
        return await self._update(
            doc_id,
            status=DocumentStatus.READY.value,
            error=None,
            num_pages=len(paged) or None,
            num_chunks=len(chunks),
        )

    async def list(self, corpus: str | None = None) -> DocumentList:
        async with self._sessionmaker() as session:
            rows = await DocumentRepository(session).list(corpus)
        return DocumentList(documents=[DocumentRecord.model_validate(r) for r in rows], total=len(rows))

    async def get(self, doc_id: str) -> DocumentRecord:
        return DocumentRecord.model_validate(await self._get_row(doc_id))

    async def delete(self, doc_id: str) -> None:
        async with self._sessionmaker() as session:
            row = await DocumentRepository(session).get(doc_id)
            if row is None:
                raise DocumentNotFoundError(f"Document {doc_id} not found")
            if row.status == DocumentStatus.PROCESSING.value:
                raise DocumentBusyError(f"Document {doc_id} is still being ingested; try again when it finishes")
            await self._remove(session, row)
            await session.commit()

    async def _remove(self, session: AsyncSession, row: DocumentRow) -> None:
        removed = await asyncio.to_thread(self._store.delete_document, row.corpus, row.id)
        Path(row.storage_path).unlink(missing_ok=True)
        await DocumentRepository(session).delete(row.id)
        logger.info("Deleted document %s (%s) and %d chunks", row.filename, row.id, removed)

    async def _discard_vectors(self, row: DocumentRow) -> None:
        try:
            await asyncio.to_thread(self._store.delete_document, row.corpus, row.id)
        except Exception:  # noqa: BLE001 - best-effort cleanup after a failure that is already being reported
            logger.exception("Could not remove partial vectors for document %s", row.id)

    async def _get_row(self, doc_id: str) -> DocumentRow:
        async with self._sessionmaker() as session:
            row = await DocumentRepository(session).get(doc_id)
        if row is None:
            raise DocumentNotFoundError(f"Document {doc_id} not found")
        return row

    async def _update(self, doc_id: str, **fields: object) -> DocumentRecord:
        async with self._sessionmaker() as session:
            row = await DocumentRepository(session).update(doc_id, **fields)
            await session.commit()
        if row is None:
            raise DocumentNotFoundError(f"Document {doc_id} was deleted during ingestion")
        return DocumentRecord.model_validate(row)


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
