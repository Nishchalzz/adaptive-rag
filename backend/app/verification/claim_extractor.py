"""Atomic claim extraction from a generated answer (research contribution 2, step 2).

The LLM only splits sentences into self-contained claims. Citations are not taken from the LLM:
each claim inherits the citation markers actually present in its source sentence, so traceability
is grounded in the answer text.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.core.structured import StructuredOutputError, structured_call
from app.core.usage import UsageTracker
from app.generation.citations import strip_citations
from app.schemas.query import CitedSentence

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You split an answer into atomic factual claims for fact-checking.

Rules:
- Each claim states exactly one fact and is understandable on its own: replace pronouns and vague references with the names they refer to.
- Keep the meaning of the answer. Do not add, correct or judge anything.
- Ignore sentences that only say that information is missing, unknown or not in the documents.
- Do not copy the bracketed citation markers into the claim text.
- For each claim give "sentence_index": the number of the sentence it comes from.

Return JSON: {"claims": [{"text": "...", "sentence_index": 0}]}. Return {"claims": []} if the answer makes no factual claims."""

# Sentences that only say the answer is unavailable are not factual claims. A negation alone is not
# enough ("X was not elected" is a claim), so the weak patterns also require a reference to the sources.
_STRONG_ABSTENTION = re.compile(
    r"\b(?:insufficient|not enough)\s+(?:information|evidence|details?)\b|"
    r"\bcannot be (?:determined|answered)\b|\bunable to (?:find|determine|answer)\b",
    re.IGNORECASE,
)
_WEAK_ABSTENTION = re.compile(
    r"\b(?:do|does|did)\s+not\s+(?:mention|provide|contain|specify|state|say|include|indicate|give|describe|discuss|identify)\b|"
    r"\bno\s+(?:information|mention|details?|data|indication)\b|"
    r"\bnot\s+(?:mentioned|provided|specified|stated|described|addressed)\b",
    re.IGNORECASE,
)
_SOURCE_REFERENCE = re.compile(r"\b(?:passages?|documents?|context|sources?|provided (?:text|information))\b", re.IGNORECASE)


class _ExtractedClaim(BaseModel):
    text: str = Field(min_length=3)
    sentence_index: int


class _ClaimList(BaseModel):
    claims: list[_ExtractedClaim]


class Claim(BaseModel):
    claim_id: str
    text: str
    sentence_index: int
    citations: list[int]


def is_abstention(sentence: str) -> bool:
    if _STRONG_ABSTENTION.search(sentence):
        return True
    return bool(_WEAK_ABSTENTION.search(sentence) and _SOURCE_REFERENCE.search(sentence))


class ClaimExtractor:
    def __init__(self, chat_model: BaseChatModel, model_name: str) -> None:
        self._model = chat_model
        self.model_name = model_name

    async def extract(
        self, sentences: Sequence[CitedSentence], *, max_claims: int, max_tokens: int, usage: UsageTracker, attempt: int
    ) -> tuple[list[Claim], bool]:
        """Return (claims, used_fallback)."""
        factual = [s for s in sentences if strip_citations(s.text) and not is_abstention(s.text)]
        if not factual:
            return [], False
        listing = "\n".join(f"[s{s.index}] {s.text}" for s in sentences)
        by_index = {s.index: s for s in sentences}
        try:
            result = await structured_call(
                self._model,
                [SystemMessage(content=EXTRACTION_PROMPT), HumanMessage(content=f"Answer sentences:\n{listing}")],
                _ClaimList,
                stage="claim_extraction",
                model_name=self.model_name,
                usage=usage,
                max_tokens=max_tokens,
            )
            extracted = [(c.text.strip(), c.sentence_index) for c in result.claims if c.sentence_index in by_index]
            fallback = False
        except StructuredOutputError:
            logger.warning("Claim extraction failed; using one claim per sentence", exc_info=True)
            extracted = [(strip_citations(s.text), s.index) for s in factual]
            fallback = True

        claims = []
        for text, index in extracted[:max_claims]:
            claims.append(
                Claim(
                    claim_id=f"a{attempt}c{len(claims)}",
                    text=strip_citations(text),
                    sentence_index=index,
                    citations=list(by_index[index].citation_numbers),
                )
            )
        return claims, fallback
