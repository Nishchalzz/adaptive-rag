"""Claim-level evidence verification (research contribution 2, steps 3-6).

For every claim:
  1. Test the best windows of its *cited* passages with NLI.
     entailment >= support_threshold                         -> supported (citation correct)
  2. Otherwise test the best windows of the *other* retrieved passages.
     entailment >= support_threshold                         -> supported, citation misattributed/missing
  3. Otherwise, contradiction >= contradiction_threshold in a cited passage
     (and weak entailment)                                    -> unsupported (contradicted)
  4. Otherwise the NLI result is ambiguous: an LLM judge returns supported / partial / unsupported and
     must quote its evidence; a quote that cannot be found in the passage voids the verdict.
     Without the judge: entailment >= partial_threshold -> partial, else unsupported.

Similarity never decides support; it only selects which windows are tested.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.config import VerificationConfig
from app.core.structured import StructuredOutputError, structured_call
from app.core.usage import UsageTracker
from app.schemas.pipeline import ClaimVerification, EvidenceSpan, SupportStatus, VerificationReport
from app.schemas.query import RetrievedChunk
from app.verification.citation_verifier import citation_status, quote_in_passage
from app.verification.claim_extractor import Claim
from app.verification.evidence_matcher import EvidenceIndex, EvidenceMatcher, EvidenceWindow
from app.verification.nli import EntailmentModel, NLIRegistry, NLIResult

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You check whether a claim is supported by numbered evidence passages.

Verdicts:
- "supported": the passages state or directly imply everything in the claim, either in one passage or by combining facts from several passages.
- "partial": the passages support part of the claim, but some detail (for example a date, number, name or qualifier) is missing or different.
- "unsupported": the passages do not support the claim, or contradict it.

Use only the passages, not your own knowledge.
Return JSON: {"verdict": "supported" | "partial" | "unsupported", "passage": <number of the passage you rely on, or null>, "evidence_quote": "<text copied exactly from that passage, or empty>", "rationale": "<one sentence>"}"""

MAX_JUDGE_PASSAGES = 4


class _JudgeVerdict(BaseModel):
    verdict: Literal["supported", "partial", "unsupported"]
    passage: int | None = None
    evidence_quote: str = ""
    rationale: str = ""


@dataclass
class _Scored:
    window: EvidenceWindow
    nli: NLIResult


@dataclass
class _Decision:
    kind: Literal["supported", "contradicted", "ambiguous", "none"]
    best: _Scored | None  # highest entailment
    most_contradicting: _Scored | None


def _decide(scored: list[_Scored], config: VerificationConfig) -> _Decision:
    if not scored:
        return _Decision("none", None, None)
    best = max(scored, key=lambda s: s.nli.entailment)
    contra = max(scored, key=lambda s: s.nli.contradiction)
    if best.nli.entailment >= config.support_threshold:
        return _Decision("supported", best, contra)
    if contra.nli.contradiction >= config.contradiction_threshold and best.nli.entailment < config.partial_threshold:
        return _Decision("contradicted", best, contra)
    return _Decision("ambiguous", best, contra)


def _span(scored: _Scored) -> EvidenceSpan:
    chunk = scored.window.chunk
    return EvidenceSpan(
        citation_number=chunk.citation_number,
        chunk_id=chunk.chunk_id,
        filename=chunk.filename,
        page=chunk.page,
        text=scored.window.text,
        similarity=round(scored.window.similarity, 4),
        entailment=round(scored.nli.entailment, 4),
        neutral=round(scored.nli.neutral, 4),
        contradiction=round(scored.nli.contradiction, 4),
    )


class EvidenceVerifier:
    def __init__(
        self, *, nli: NLIRegistry, matcher: EvidenceMatcher, judge_model: BaseChatModel, judge_model_name: str
    ) -> None:
        self._nli = nli
        self._matcher = matcher
        self._judge = judge_model
        self._judge_name = judge_model_name

    async def verify(
        self,
        claims: list[Claim],
        chunks: list[RetrievedChunk],
        config: VerificationConfig,
        usage: UsageTracker,
        embedding_cache: dict[str, list[float]],
    ) -> VerificationReport:
        started = perf_counter()
        model = self._nli.get(config.nli_model)
        if not claims:
            return VerificationReport(
                claims=[], support_score=1.0, citation_accuracy=None, num_supported=0, num_partial=0,
                num_unsupported=0, passed=True, abstained=True, extraction_fallback=False,
                nli_model=model.name, latency_ms=(perf_counter() - started) * 1000,
            )

        index = await self._matcher.build_index(
            chunks, [c.text for c in claims], window_sentences=config.window_sentences, cache=embedding_cache, usage=usage
        )
        valid = {c.citation_number for c in chunks}
        cited = [set(c.citations) & valid for c in claims]
        top_m = config.candidate_windows

        cited_candidates = [index.candidates(i, cited[i], top_m) if cited[i] else [] for i in range(len(claims))]
        cited_decisions = [
            _decide(s, config) for s in await self._score(model, claims, cited_candidates)
        ]

        need_other = [i for i, d in enumerate(cited_decisions) if d.kind != "supported"]
        other_candidates = [index.candidates(i, valid - cited[i], top_m) for i in need_other]
        other_scored = await self._score(model, [claims[i] for i in need_other], other_candidates)
        other_decisions = {i: _decide(s, config) for i, s in zip(need_other, other_scored, strict=True)}

        tasks = [
            self._resolve(claims[i], cited[i], cited_decisions[i], other_decisions.get(i), index, i, chunks, config, usage)
            for i in range(len(claims))
        ]
        results = list(await asyncio.gather(*tasks))

        supported = sum(r.status is SupportStatus.SUPPORTED for r in results)
        partial = sum(r.status is SupportStatus.PARTIAL for r in results)
        unsupported = len(results) - supported - partial
        with_citations = [r for r, c in zip(results, cited, strict=True) if c]
        support_score = (supported + 0.5 * partial) / len(results)
        return VerificationReport(
            claims=results,
            support_score=round(support_score, 4),
            citation_accuracy=(
                round(sum(r.citation_status.value == "correct" for r in with_citations) / len(with_citations), 4)
                if with_citations
                else None
            ),
            num_supported=supported,
            num_partial=partial,
            num_unsupported=unsupported,
            passed=support_score >= config.pass_support_score and unsupported == 0,
            abstained=False,
            extraction_fallback=False,
            nli_model=model.name,
            latency_ms=(perf_counter() - started) * 1000,
        )

    async def _score(
        self, model: EntailmentModel, claims: list[Claim], candidates: list[list[EvidenceWindow]]
    ) -> list[list[_Scored]]:
        """NLI-score each claim against its candidate windows, the whole passages those windows come
        from (windows can lose the entity a passage is about, e.g. "the lab"), and the two best
        passages combined (claims that join facts from two passages)."""
        pairs: list[tuple[str, str]] = []
        owners: list[tuple[int, EvidenceWindow]] = []
        for i, (claim, windows) in enumerate(zip(claims, candidates, strict=True)):
            tested = list(windows)
            passages = list({w.chunk.chunk_id: w for w in windows}.values())  # best window per passage
            for best in passages:
                if best.chunk.text != best.text:
                    tested.append(EvidenceWindow(best.chunk, best.chunk.text, best.similarity))
            if len(passages) >= 2:
                first, second = passages[0], passages[1]
                tested.append(EvidenceWindow(first.chunk, f"{first.chunk.text}\n{second.chunk.text}", first.similarity))
            elif len(windows) >= 2 and windows[0].text != windows[1].text:
                first, second = windows[0], windows[1]
                tested.append(EvidenceWindow(first.chunk, f"{first.text} {second.text}", first.similarity))
            for window in tested:
                pairs.append((window.text, claim.text))
                owners.append((i, window))
        scores = await asyncio.to_thread(model.score, pairs)
        grouped: list[list[_Scored]] = [[] for _ in claims]
        for (i, window), result in zip(owners, scores, strict=True):
            grouped[i].append(_Scored(window, result))
        return grouped

    async def _resolve(
        self,
        claim: Claim,
        cited: set[int],
        cited_decision: _Decision,
        other_decision: _Decision | None,
        index: EvidenceIndex,
        claim_index: int,
        chunks: list[RetrievedChunk],
        config: VerificationConfig,
        usage: UsageTracker,
    ) -> ClaimVerification:
        base = {"claim_id": claim.claim_id, "text": claim.text, "sentence_index": claim.sentence_index, "citations": claim.citations}

        if cited_decision.kind == "supported":
            assert cited_decision.best is not None
            return ClaimVerification(
                **base, status=SupportStatus.SUPPORTED, method="nli", evidence=_span(cited_decision.best),
                citation_status=citation_status(SupportStatus.SUPPORTED, cited, cited_decision.best.window.chunk.citation_number),
            )
        if other_decision is not None and other_decision.kind == "supported":
            assert other_decision.best is not None
            number = other_decision.best.window.chunk.citation_number
            return ClaimVerification(
                **base, status=SupportStatus.SUPPORTED, method="nli", evidence=_span(other_decision.best),
                citation_status=citation_status(SupportStatus.SUPPORTED, cited, number), suggested_citation=number,
            )

        all_scored = [s for d in (cited_decision, other_decision) if d for s in (d.best, d.most_contradicting) if s]
        contradicted = any(s.nli.contradiction >= config.contradiction_threshold for s in all_scored)
        if cited_decision.kind == "contradicted":
            return ClaimVerification(
                **base, status=SupportStatus.UNSUPPORTED, failure="contradicted", method="nli",
                evidence=_span(cited_decision.most_contradicting) if cited_decision.most_contradicting else None,
                citation_status=citation_status(SupportStatus.UNSUPPORTED, cited, None),
            )

        best = max(all_scored, key=lambda s: s.nli.entailment, default=None)
        if not config.use_llm_judge:
            status = (
                SupportStatus.PARTIAL
                if best is not None and best.nli.entailment >= config.partial_threshold
                else SupportStatus.UNSUPPORTED
            )
            number = best.window.chunk.citation_number if best and status is SupportStatus.PARTIAL else None
            return ClaimVerification(
                **base, status=status, method="nli_threshold", evidence=_span(best) if best else None,
                failure=None if status is SupportStatus.PARTIAL else ("contradicted" if contradicted else "no_evidence"),
                citation_status=citation_status(status, cited, number),
                suggested_citation=number if number is not None and number not in cited else None,
            )
        return await self._judge_claim(base, claim, cited, index, claim_index, chunks, all_scored, contradicted, usage)

    async def _judge_claim(
        self,
        base: dict,
        claim: Claim,
        cited: set[int],
        index: EvidenceIndex,
        claim_index: int,
        chunks: list[RetrievedChunk],
        scored: list[_Scored],
        contradicted: bool,
        usage: UsageTracker,
    ) -> ClaimVerification:
        by_number = {c.citation_number: c for c in chunks}
        ranked = [w.chunk.citation_number for w in index.candidates(claim_index, None, 3 * MAX_JUDGE_PASSAGES)]
        numbers = list(dict.fromkeys([n for n in sorted(cited) if n in by_number] + ranked))[:MAX_JUDGE_PASSAGES]
        passages = "\n\n".join(f"Passage [{n}] (source: {by_number[n].filename})\n{by_number[n].text}" for n in numbers)
        try:
            verdict = await structured_call(
                self._judge,
                [SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=f"{passages}\n\nClaim: {claim.text}")],
                _JudgeVerdict,
                stage="verification_judge",
                model_name=self._judge_name,
                usage=usage,
                max_tokens=300,
            )
        except StructuredOutputError:
            logger.warning("Verification judge failed for claim %s; marking unsupported", claim.claim_id, exc_info=True)
            verdict = _JudgeVerdict(verdict="unsupported", rationale="Judge output could not be parsed.")

        status = SupportStatus(verdict.verdict)
        number = verdict.passage if verdict.passage in numbers else None
        quote_ok: bool | None = None
        if status is not SupportStatus.UNSUPPORTED:
            quote_ok = number is not None and quote_in_passage(verdict.evidence_quote, by_number[number].text)
            if not quote_ok:
                status = SupportStatus.UNSUPPORTED
        evidence = None
        if number is not None:
            matching = [s for s in scored if s.window.chunk.citation_number == number]
            evidence = _span(max(matching, key=lambda s: s.nli.entailment)) if matching else None
        supporting = number if status is not SupportStatus.UNSUPPORTED else None
        return ClaimVerification(
            **base,
            status=status,
            method="llm_judge",
            evidence=evidence,
            failure=None if supporting is not None else ("contradicted" if contradicted else "no_evidence"),
            citation_status=citation_status(status, cited, supporting),
            suggested_citation=supporting if supporting is not None and supporting not in cited else None,
            judge_rationale=verdict.rationale,
            judge_quote=verdict.evidence_quote or None,
            judge_quote_verified=quote_ok,
        )
