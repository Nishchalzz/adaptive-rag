"""Text extraction for the supported upload formats.

PDFs keep their page structure so citations can point to a page; DOCX/TXT/Markdown have no
reliable page concept and are returned as a single unpaged unit.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import docx2txt
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.errors import IngestionError, UnsupportedFileTypeError

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "markdown",
    ".markdown": "markdown",
}


@dataclass(frozen=True)
class PageText:
    page: int | None  # 1-based page number, None for unpaged formats
    text: str


def detect_file_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    try:
        return SUPPORTED_EXTENSIONS[suffix]
    except KeyError:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(f"Unsupported file type '{suffix or filename}'. Allowed: {allowed}") from None


def load_document(path: Path, file_type: str) -> list[PageText]:
    loaders = {"pdf": _load_pdf, "docx": _load_docx, "txt": _load_text, "markdown": _load_text}
    if file_type not in loaders:
        raise UnsupportedFileTypeError(f"No loader for file type '{file_type}'")
    pages = loaders[file_type](path)
    if not any(p.text.strip() for p in pages):
        hint = " (scanned PDFs need OCR, which is not supported)" if file_type == "pdf" else ""
        raise IngestionError(f"No extractable text found in {path.name}{hint}")
    return pages


def _load_pdf(path: Path) -> list[PageText]:
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise IngestionError(f"{path.name} is password-protected")
        return [PageText(page=i + 1, text=page.extract_text() or "") for i, page in enumerate(reader.pages)]
    except PdfReadError as exc:
        raise IngestionError(f"Could not read PDF {path.name}: {exc}") from exc


def _load_docx(path: Path) -> list[PageText]:
    try:
        return [PageText(page=None, text=docx2txt.process(str(path)) or "")]
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, OSError) as exc:
        raise IngestionError(f"Could not read DOCX {path.name}: {exc}") from exc


def _load_text(path: Path) -> list[PageText]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return [PageText(page=None, text=raw.decode(encoding))]
        except UnicodeDecodeError:
            continue
    return [PageText(page=None, text=raw.decode("latin-1"))]
