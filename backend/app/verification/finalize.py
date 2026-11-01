"""Turning verified attempts into the final answer (research contribution 2, step 7).

Unsupported claims are never presented as fact: sentences whose claims are all unsupported are
removed from the final answer and listed in an explicit uncertainty notice; partially supported
sentences stay but are flagged.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.schemas.pipeline import SupportStatus, Uncertainty, VerificationReport
from app.schemas.query import AttemptSummary, CitedSentence

NO_VERIFIED_ANSWER = (
    "The retrieved documents do not provide enough verified evidence to answer this question."
)

_SEVERITY = {SupportStatus.SUPPORTED: 0, SupportStatus.PARTIAL: 1, SupportStatus.UNSUPPORTED: 2}


def select_best_attempt(attempts: Sequence[AttemptSummary]) -> int:
    """Index of the attempt to return: passed first, then highest support score, then fewest
    unsupported claims, then the latest attempt."""

    def key(item: tuple[int, AttemptSummary]) -> tuple[bool, float, int, int]:
        index, attempt = item
        report = attempt.verification
        if report is None:
            return (False, 0.0, 0, index)
        return (report.passed, report.support_score, -report.num_unsupported, index)

    return max(enumerate(attempts), key=key)[0]


def annotate_sentences(sentences: Sequence[CitedSentence], report: VerificationReport | None) -> list[CitedSentence]:
    if report is None:
        return list(sentences)
    worst: dict[int, SupportStatus] = {}
    for claim in report.claims:
        current = worst.get(claim.sentence_index)
        if current is None or _SEVERITY[claim.status] > _SEVERITY[current]:
            worst[claim.sentence_index] = claim.status
    # A sentence is only "unsupported" if none of its claims has support.
    any_support = {c.sentence_index for c in report.claims if c.status is not SupportStatus.UNSUPPORTED}
    annotated = []
    for sentence in sentences:
        status = worst.get(sentence.index)
        if status is SupportStatus.UNSUPPORTED and sentence.index in any_support:
            status = SupportStatus.PARTIAL
        annotated.append(sentence.model_copy(update={"status": status}))
    return annotated


def strip_unsupported(answer: str, sentences: Sequence[CitedSentence]) -> tuple[str, list[str]]:
    """Remove fully unsupported sentences from the answer text, keeping its line structure."""
    removed = [s.text for s in sentences if s.status is SupportStatus.UNSUPPORTED]
    if not removed:
        return answer, []
    text = answer
    for sentence in removed:
        text = text.replace(sentence, "", 1)
    lines = [re.sub(r"\s{2,}", " ", line).rstrip() for line in text.splitlines()]
    lines = [line for line in lines if line.strip() not in {"", "-", "*", "•"} or line == ""]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if not any(s.status in (SupportStatus.SUPPORTED, SupportStatus.PARTIAL) for s in sentences):
        return NO_VERIFIED_ANSWER, removed
    return cleaned, removed


def build_uncertainty(report: VerificationReport | None, removed: list[str]) -> Uncertainty | None:
    if report is None:
        return None
    unsupported = [c.text for c in report.claims if c.status is SupportStatus.UNSUPPORTED]
    partial = [c.text for c in report.claims if c.status is SupportStatus.PARTIAL]
    if report.passed and not partial:
        return Uncertainty(flagged=False, notice=None, unsupported_claims=[], removed_sentences=[])
    parts = []
    if unsupported:
        parts.append(
            f"{len(unsupported)} statement(s) could not be verified against the retrieved documents"
            + (" and were removed from the answer." if removed else ".")
        )
    if partial:
        parts.append(f"{len(partial)} statement(s) are only partially supported by the documents.")
    return Uncertainty(flagged=True, notice=" ".join(parts), unsupported_claims=unsupported, removed_sentences=removed)
