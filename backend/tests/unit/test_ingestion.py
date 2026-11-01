from pathlib import Path

import pytest

from app.core.config import ChunkingConfig
from app.core.errors import IngestionError, UnsupportedFileTypeError
from app.ingestion.chunking import Chunker
from app.ingestion.cleaning import clean_text
from app.ingestion.loaders import PageText, detect_file_type, load_document
from tests.helpers import make_docx, make_pdf


class TestLoaders:
    def test_pdf_keeps_page_numbers(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        path.write_bytes(make_pdf(["Alpha page one.", "Beta page two."]))
        pages = load_document(path, "pdf")
        assert [p.page for p in pages] == [1, 2]
        assert "Alpha" in pages[0].text and "Beta" in pages[1].text

    def test_docx(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.docx"
        path.write_bytes(make_docx(["First paragraph.", "Second paragraph."]))
        pages = load_document(path, "docx")
        assert pages[0].page is None
        assert "First paragraph." in pages[0].text and "Second paragraph." in pages[0].text

    def test_text_falls_back_to_cp1252(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_bytes("Caf\xe9 menu".encode("cp1252"))
        assert load_document(path, "txt")[0].text == "Café menu"

    def test_corrupt_docx_raises_ingestion_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.docx"
        path.write_bytes(b"not a zip file")
        with pytest.raises(IngestionError):
            load_document(path, "docx")

    def test_pdf_without_text_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "blank.pdf"
        path.write_bytes(make_pdf([""]))
        with pytest.raises(IngestionError, match="No extractable text"):
            load_document(path, "pdf")

    @pytest.mark.parametrize("name,expected", [("a.PDF", "pdf"), ("b.md", "markdown"), ("c.docx", "docx"), ("d.txt", "txt")])
    def test_detect_file_type(self, name: str, expected: str) -> None:
        assert detect_file_type(name) == expected

    def test_unsupported_type(self) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            detect_file_type("slides.pptx")


class TestCleaning:
    def test_normalises_ligatures_and_whitespace(self) -> None:
        assert clean_text("The ﬁnal\t\t result  \n\n\n\nNext", "txt") == "The final result\n\nNext"

    def test_pdf_joins_wrapped_lines_and_hyphenation(self) -> None:
        text = "Retrieval aug-\nmented generation\ncombines search.\n\nNew paragraph."
        assert clean_text(text, "pdf") == "Retrieval augmented generation combines search.\n\nNew paragraph."

    def test_markdown_syntax_is_stripped(self) -> None:
        text = "# Title\nSee [the docs](http://x.y) and **bold** ![img](a.png)<br>"
        assert clean_text(text, "markdown") == "Title\nSee the docs and bold"

    def test_control_characters_removed(self) -> None:
        assert clean_text("a\x00b\x07c", "txt") == "abc"


class TestChunker:
    def _sentences(self, n: int) -> str:
        return " ".join(f"Sentence number {i} talks about topic {i}." for i in range(n))

    def test_chunks_respect_size_and_overlap(self) -> None:
        config = ChunkingConfig(chunk_size=200, chunk_overlap=50)
        page_text = self._sentences(30)
        chunks = Chunker(config).split(doc_id="d1", corpus="c", filename="f.txt", pages=[PageText(None, page_text)])
        assert len(chunks) > 1
        assert all(len(c.text) <= 200 for c in chunks)
        # consecutive chunks share text because of the overlap
        assert any(chunks[i].text[-20:] in chunks[i + 1].text or chunks[i + 1].text[:20] in chunks[i].text for i in range(len(chunks) - 1))
        # start offsets point back into the source text
        for chunk in chunks:
            assert chunk.char_start is not None
            assert page_text[chunk.char_start : chunk.char_start + len(chunk.text)] == chunk.text

    def test_chunks_keep_page_and_unique_ids(self) -> None:
        pages = [PageText(1, self._sentences(10)), PageText(2, ""), PageText(3, self._sentences(10))]
        chunks = Chunker(ChunkingConfig(chunk_size=300, chunk_overlap=30)).split(
            doc_id="doc", corpus="c", filename="paper.pdf", pages=pages
        )
        assert {c.page for c in chunks} == {1, 3}
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert all(c.filename == "paper.pdf" and c.doc_id == "doc" for c in chunks)
