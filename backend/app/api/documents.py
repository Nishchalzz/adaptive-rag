from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, Response, UploadFile, status
from pydantic import TypeAdapter, ValidationError

from app.api.deps import get_container
from app.core.container import ServiceContainer
from app.core.errors import FileTooLargeError, InvalidConfigError
from app.schemas.documents import CorpusName, DocumentList, DocumentRecord, DocumentStatus

router = APIRouter(prefix="/api/documents", tags=["documents"])

Container = Annotated[ServiceContainer, Depends(get_container)]
_corpus_adapter = TypeAdapter(CorpusName)


def _validate_corpus(corpus: str | None, default: str) -> str:
    try:
        return _corpus_adapter.validate_python(corpus or default)
    except ValidationError:
        raise InvalidConfigError(
            "Corpus names may contain letters, digits, '_' and '-' (max 48 characters)"
        ) from None


@router.post("", response_model=DocumentRecord, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    container: Container,
    background_tasks: BackgroundTasks,
    response: Response,
    file: Annotated[UploadFile, File(description="PDF, DOCX, TXT or Markdown file")],
    corpus: Annotated[str | None, Form()] = None,
    wait: Annotated[bool, Query(description="Ingest before responding instead of in the background")] = False,
) -> DocumentRecord:
    corpus_name = _validate_corpus(corpus, container.settings.default_corpus)
    limit = container.settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise FileTooLargeError(f"{file.filename} exceeds the {container.settings.max_upload_mb} MB upload limit")

    record = await container.documents.register_upload(file.filename or "upload", data, corpus_name)
    if not wait:
        background_tasks.add_task(container.documents.ingest, record.id)
        return record

    record = await container.documents.ingest(record.id)
    if record.status == DocumentStatus.FAILED:
        response.status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    else:
        response.status_code = status.HTTP_201_CREATED
    return record


@router.get("", response_model=DocumentList)
async def list_documents(container: Container, corpus: str | None = None) -> DocumentList:
    return await container.documents.list(corpus)


@router.get("/{doc_id}", response_model=DocumentRecord)
async def get_document(doc_id: str, container: Container) -> DocumentRecord:
    return await container.documents.get(doc_id)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str, container: Container) -> Response:
    await container.documents.delete(doc_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
