"""Citation correctness and judge-quote validation."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from app.schemas.pipeline import CitationStatus, SupportStatus


def citation_status(status: SupportStatus, cited: set[int], supporting_passage: int | None) -> CitationStatus:
    """Classify a claim's citation given which passage (if any) supports it.

    correct        a cited passage supports (or partially supports) the claim
    misattributed  the claim cites passages, but only a different retrieved passage supports it
    missing        the claim carries no valid citation
    incorrect      the claim cites passages and nothing retrieved supports it
    """
    if not cited:
        return CitationStatus.MISSING
    if status is SupportStatus.UNSUPPORTED or supporting_passage is None:
        return CitationStatus.INCORRECT
    return CitationStatus.CORRECT if supporting_passage in cited else CitationStatus.MISATTRIBUTED


def _normalise(text: str) -> str:
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_in_passage(quote: str, passage: str, min_ratio: float = 0.85) -> bool:
    """True if the judge's evidence quote really occurs in the passage (tolerating minor edits).

    Guards against a judge that "supports" a claim with evidence it invented.
    """
    q, p = _normalise(quote), _normalise(passage)
    if len(q) < 8:
        return False
    if q in p:
        return True
    match = SequenceMatcher(None, p, q, autojunk=False).find_longest_match(0, len(p), 0, len(q))
    return match.size / len(q) >= min_ratio
