"""Text normalisation applied before chunking."""

import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_EMPHASIS = re.compile(r"(\*\*|__)(.+?)\1")
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")


def clean_text(text: str, file_type: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # also expands ligatures such as "ﬁ"
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)

    if file_type == "markdown":
        text = _strip_markdown(text)
    if file_type == "pdf":
        # Undo line-break hyphenation, then join hard-wrapped lines inside a paragraph.
        text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_markdown(text: str) -> str:
    text = _MD_IMAGE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    text = _MD_HEADING.sub("", text)
    return _MD_EMPHASIS.sub(r"\2", text)
