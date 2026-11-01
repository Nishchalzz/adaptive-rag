from app.generation.citations import citation_numbers, parse_citations, split_sentences
from app.generation.prompts import build_generation_messages, format_context
from app.schemas.documents import Chunk


def test_split_sentences_keeps_citations_with_their_sentence() -> None:
    answer = "Paris is the capital [1]. It has 2.1 million residents. [2] The Seine flows through it [1][3]."
    assert split_sentences(answer) == [
        "Paris is the capital [1].",
        "It has 2.1 million residents. [2]",
        "The Seine flows through it [1][3].",
    ]


def test_abbreviations_and_initials_do_not_end_sentences() -> None:
    answer = "The lab is led by Dr. Lena Park [1]. Prof. A. Bergström is dean, e.g. since 2016 [2]."
    assert split_sentences(answer) == ["The lab is led by Dr. Lena Park [1].", "Prof. A. Bergström is dean, e.g. since 2016 [2]."]


def test_split_sentences_handles_lists_and_unterminated_lines() -> None:
    answer = "Key points:\n- First point [1]\n- Second point [2]"
    assert split_sentences(answer) == ["Key points:", "- First point [1]", "- Second point [2]"]


def test_citation_numbers_supports_grouped_forms() -> None:
    assert citation_numbers("A [1, 3] and B [2;4] and C [3]") == [1, 3, 2, 4]


def test_parse_citations_separates_invalid_numbers() -> None:
    parsed = parse_citations("Claim one [2]. Claim two [5][1].", num_passages=3)
    assert parsed.cited == [2, 1]
    assert parsed.invalid == [5]
    assert [s.citation_numbers for s in parsed.sentences] == [[2], [5, 1]]


def test_answer_without_citations() -> None:
    parsed = parse_citations("The documents do not say.", num_passages=3)
    assert parsed.cited == [] and parsed.invalid == []
    assert parsed.sentences[0].citation_numbers == []


def test_context_is_numbered_with_sources() -> None:
    chunks = [
        Chunk(chunk_id="a", doc_id="d", corpus="c", filename="paper.pdf", page=4, chunk_index=0, text="Alpha."),
        Chunk(chunk_id="b", doc_id="d", corpus="c", filename="notes.md", page=None, chunk_index=1, text="Beta."),
    ]
    assert format_context(chunks) == "[1] (source: paper.pdf, page 4)\nAlpha.\n\n[2] (source: notes.md)\nBeta."
    messages = build_generation_messages("What?", chunks)
    assert "Question: What?" in messages[-1].content
