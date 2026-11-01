"""Parsing of bracketed citations ("[2]", "[1, 3]", "[1][3]") in generated answers."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CITATION_GROUP = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
# A sentence ends at . ! or ? (optionally followed by citation groups) when whitespace or the end of
# the line follows. "3.5" and "[1]." do not end a sentence early.
_SENTENCE_END = re.compile(r"[.!?](?:\s*\[\d+(?:\s*[,;]\s*\d+)*\])*(?=\s|$)")
# A full stop after these words (or after a single letter, as in initials) does not end a sentence.
_ABBREVIATIONS = frozenset(
    "dr prof mr mrs ms st jr sr vs etc e.g i.e fig al inc ltd corp dept univ approx mt ft gen col lt rev hon "
    "u.s u.k jan feb mar apr jun jul aug sep sept oct nov dec".split()
)
_WORD_BEFORE = re.compile(r"([A-Za-z][A-Za-z.]*)$")


def _is_abbreviation(line: str, period_at: int) -> bool:
    if line[period_at] != ".":
        return False
    match = _WORD_BEFORE.search(line[:period_at])
    if not match:
        return False
    word = match.group(1).lower().rstrip(".")
    return word in _ABBREVIATIONS or len(word) == 1


@dataclass(frozen=True)
class ParsedSentence:
    index: int
    text: str
    citation_numbers: list[int]


@dataclass(frozen=True)
class ParsedCitations:
    sentences: list[ParsedSentence]
    cited: list[int]  # valid citation numbers in order of first appearance
    invalid: list[int]  # numbers that do not correspond to a provided passage


def strip_citations(text: str) -> str:
    """Sentence text without bracketed citation markers."""
    text = _CITATION_GROUP.sub("", text)
    return re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()


def citation_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for group in _CITATION_GROUP.findall(text):
        for part in re.split(r"[,;]", group):
            number = int(part.strip())
            if number not in numbers:
                numbers.append(number)
    return numbers


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        start = 0
        for match in _SENTENCE_END.finditer(line):
            if _is_abbreviation(line, match.start()) and match.end() == match.start() + 1:
                continue  # "Dr. Park" - unless citations follow the full stop, which marks a real ending
            sentence = line[start : match.end()].strip()
            if sentence:
                sentences.append(sentence)
            start = match.end()
        tail = line[start:].strip()
        if tail:
            if _CITATION_GROUP.fullmatch(tail) and sentences:
                sentences[-1] = f"{sentences[-1]} {tail}"  # stray citation after the final full stop
            else:
                sentences.append(tail)
    return sentences


def parse_citations(answer: str, num_passages: int) -> ParsedCitations:
    sentences = [
        ParsedSentence(index=i, text=s, citation_numbers=citation_numbers(s)) for i, s in enumerate(split_sentences(answer))
    ]
    cited: list[int] = []
    invalid: list[int] = []
    for sentence in sentences:
        for number in sentence.citation_numbers:
            bucket = cited if 1 <= number <= num_passages else invalid
            if number not in bucket:
                bucket.append(number)
    return ParsedCitations(sentences=sentences, cited=cited, invalid=invalid)
