from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import DocumentRow, QueryLogRow


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, row: DocumentRow) -> DocumentRow:
        self._session.add(row)
        await self._session.flush()
        return row

    async def get(self, doc_id: str) -> DocumentRow | None:
        return await self._session.get(DocumentRow, doc_id)

    async def get_by_hash(self, corpus: str, sha256: str) -> DocumentRow | None:
        result = await self._session.execute(
            select(DocumentRow).where(DocumentRow.corpus == corpus, DocumentRow.sha256 == sha256)
        )
        return result.scalar_one_or_none()

    async def list(self, corpus: str | None = None) -> Sequence[DocumentRow]:
        stmt = select(DocumentRow).order_by(DocumentRow.created_at.desc())
        if corpus is not None:
            stmt = stmt.where(DocumentRow.corpus == corpus)
        return (await self._session.execute(stmt)).scalars().all()

    async def update(self, doc_id: str, **fields: Any) -> DocumentRow | None:
        row = await self.get(doc_id)
        if row is None:
            return None
        for key, value in fields.items():
            setattr(row, key, value)
        await self._session.flush()
        return row

    async def delete(self, doc_id: str) -> None:
        await self._session.execute(delete(DocumentRow).where(DocumentRow.id == doc_id))


class QueryLogRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, row: QueryLogRow) -> None:
        self._session.add(row)
        await self._session.flush()

    async def get(self, query_id: str) -> QueryLogRow | None:
        return await self._session.get(QueryLogRow, query_id)

    async def list(self, *, limit: int = 50, offset: int = 0, pipeline: str | None = None) -> Sequence[QueryLogRow]:
        stmt = select(QueryLogRow).order_by(QueryLogRow.created_at.desc()).limit(limit).offset(offset)
        if pipeline is not None:
            stmt = stmt.where(QueryLogRow.pipeline == pipeline)
        return (await self._session.execute(stmt)).scalars().all()
