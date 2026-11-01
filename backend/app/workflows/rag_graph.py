"""Proposed pipeline: query-adaptive retrieval + evidence verification as a LangGraph state machine.

    START ─┬─ adaptive ─► analyze_query ─► select_depth ─► retrieve ─► assess_confidence ⇄ expand_retrieval
           └─ fixed ──────────────────────────────────────► retrieve ─┐            │
                                                                       ▼            ▼
                                                                    generate ◄──────┘ ◄── refine ◄─┐
                                                                       │                           │
                                     verify off ◄──────────────────────┤                           │
                                         │                       extract_claims ─► verify ─ fail & retries left
                                         ▼                                           │
                                      finalize ◄──────── pass, or retries exhausted ─┘

Feature flags (``PipelineConfig.proposed``) turn the adaptive and verification branches on and off,
which yields every ablation variant from this one graph.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.core.config import PipelineConfig
from app.core.errors import EmptyCorpusError
from app.core.usage import UsageTracker
from app.generation.citations import parse_citations
from app.generation.generator import AnswerGenerator
from app.generation.prompts import Revision
from app.retrieval.adaptive_retriever import AdaptiveRetrievalState, AdaptiveRetriever, FusedHit, assess_confidence
from app.retrieval.complexity_classifier import ComplexityClassifier
from app.retrieval.dense import DenseRetriever, RetrievalResult
from app.schemas.pipeline import ComplexityAssessment, RefinementStep, RetrievalConfidence, SupportStatus
from app.schemas.query import AttemptSummary, PipelineName, QueryResponse, RetrievalTrace, RetrievedChunk
from app.verification.claim_extractor import Claim, ClaimExtractor
from app.verification.finalize import (
    NO_VERIFIED_ANSWER,
    annotate_sentences,
    build_uncertainty,
    select_best_attempt,
    strip_unsupported,
)
from app.verification.verifier import EvidenceVerifier
from app.workflows.common import build_citations, build_sentences, to_retrieved_chunks

RECURSION_LIMIT = 60
MAX_REFINEMENT_QUERIES = 5


@dataclass
class ProposedComponents:
    dense: DenseRetriever
    adaptive: AdaptiveRetriever
    classifier: ComplexityClassifier
    generator: AnswerGenerator
    extractor: ClaimExtractor
    verifier: EvidenceVerifier
    embedding_model: str


class GraphState(TypedDict, total=False):
    question: str
    corpus: str
    config: PipelineConfig
    usage: UsageTracker
    started: float
    timings: dict[str, float]
    assessment: ComplexityAssessment | None
    k: int
    retrieval: AdaptiveRetrievalState | None
    fixed_retrieval: RetrievalResult | None
    confidence: RetrievalConfidence | None
    chunks: list[RetrievedChunk]
    embedding_cache: dict[str, list[float]]
    revision: Revision | None
    attempt_started: float
    attempts: list[AttemptSummary]
    claims: list[Claim]
    claims_fallback: bool
    refinements: list[RefinementStep]
    response: QueryResponse


def _with_timing(state: GraphState, name: str, started: float) -> dict[str, float]:
    timings = dict(state.get("timings", {}))
    timings[name] = timings.get(name, 0.0) + (perf_counter() - started) * 1000
    return timings


def _fused_to_chunks(selected: list[FusedHit]) -> list[RetrievedChunk]:
    chunks = []
    for rank, fused in enumerate(selected, start=1):
        hit = fused.hit
        chunks.append(
            RetrievedChunk(
                chunk_id=hit.chunk.chunk_id,
                doc_id=hit.chunk.doc_id,
                filename=hit.chunk.filename,
                page=hit.chunk.page,
                chunk_index=hit.chunk.chunk_index,
                text=hit.chunk.text,
                score=hit.score,
                rank=rank,
                citation_number=rank,
                retrieved_by="original_query" if fused.query_index == 0 else f"sub_query:{fused.query_index}",
            )
        )
    return chunks


class ProposedRAGPipeline:
    name = PipelineName.PROPOSED

    def __init__(self, components: ProposedComponents) -> None:
        self.c = components
        self._graph = self._build_graph()

    # ------------------------------------------------------------------ graph
    def _build_graph(self):  # type: ignore[no-untyped-def]
        graph = StateGraph(GraphState)
        graph.add_node("analyze_query", self._analyze_query)
        graph.add_node("select_depth", self._select_depth)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("assess_confidence", self._assess_confidence)
        graph.add_node("expand_retrieval", self._expand_retrieval)
        graph.add_node("generate", self._generate)
        graph.add_node("extract_claims", self._extract_claims)
        graph.add_node("verify", self._verify)
        graph.add_node("refine", self._refine)
        graph.add_node("finalize", self._finalize)

        graph.add_conditional_edges(
            START, lambda s: "analyze_query" if s["config"].proposed.adaptive else "retrieve", ["analyze_query", "retrieve"]
        )
        graph.add_edge("analyze_query", "select_depth")
        graph.add_edge("select_depth", "retrieve")
        graph.add_conditional_edges(
            "retrieve", lambda s: "assess_confidence" if s["config"].proposed.adaptive else "generate",
            ["assess_confidence", "generate"],
        )
        graph.add_conditional_edges("assess_confidence", self._route_confidence, ["expand_retrieval", "generate"])
        graph.add_edge("expand_retrieval", "assess_confidence")
        graph.add_conditional_edges(
            "generate", lambda s: "extract_claims" if s["config"].proposed.verify else "finalize",
            ["extract_claims", "finalize"],
        )
        graph.add_edge("extract_claims", "verify")
        graph.add_conditional_edges("verify", self._route_verification, ["refine", "finalize"])
        graph.add_edge("refine", "generate")
        graph.add_edge("finalize", END)
        return graph.compile()

    # ------------------------------------------------------------------ public API
    async def stream(self, question: str, corpus: str, config: PipelineConfig) -> AsyncIterator[dict[str, Any]]:
        """Yield one event per executed node, then the final response."""
        if self.c.dense.corpus_size(corpus) == 0:
            raise EmptyCorpusError(f"Corpus '{corpus}' has no indexed documents")
        state: dict[str, Any] = {
            "question": question,
            "corpus": corpus,
            "config": config,
            "usage": UsageTracker(config.pricing),
            "started": perf_counter(),
            "timings": {},
            "assessment": None,
            "retrieval": None,
            "fixed_retrieval": None,
            "confidence": None,
            "chunks": [],
            "embedding_cache": {},
            "revision": None,
            "attempts": [],
            "claims": [],
            "claims_fallback": False,
            "refinements": [],
        }
        async for update in self._graph.astream(dict(state), stream_mode="updates", config={"recursion_limit": RECURSION_LIMIT}):
            for node, delta in update.items():
                if delta:
                    state.update(delta)
                yield {"type": "step", "node": node, "summary": _summarise(node, state)}
        yield {"type": "result", "response": state["response"]}

    async def run(self, question: str, corpus: str, config: PipelineConfig) -> QueryResponse:
        response: QueryResponse | None = None
        async for event in self.stream(question, corpus, config):
            if event["type"] == "result":
                response = event["response"]
        assert response is not None
        return response

    # ------------------------------------------------------------------ nodes
    async def _analyze_query(self, state: GraphState) -> dict[str, Any]:
        started = perf_counter()
        config = state["config"]
        assessment = await self.c.classifier.assess(state["question"], config.complexity, config.adaptive, state["usage"])
        return {"assessment": assessment, "timings": _with_timing(state, "analyze_query", started)}

    async def _select_depth(self, state: GraphState) -> dict[str, Any]:
        assessment = state["assessment"]
        assert assessment is not None
        return {"k": assessment.k}

    async def _retrieve(self, state: GraphState) -> dict[str, Any]:
        started = perf_counter()
        config, usage = state["config"], state["usage"]
        if config.proposed.adaptive:
            retrieval = await self.c.adaptive.retrieve(
                state["corpus"], state["question"], state["assessment"], state["k"], config.adaptive
            )
            usage.record_embedding("query_embedding", self.c.embedding_model, retrieval.queries)
            return {
                "retrieval": retrieval,
                "chunks": _fused_to_chunks(retrieval.selected),
                "timings": _with_timing(state, "retrieval", started),
            }
        fixed = await self.c.dense.retrieve(state["corpus"], state["question"], config.baseline.top_k)
        usage.record_embedding("query_embedding", self.c.embedding_model, [state["question"]])
        return {
            "fixed_retrieval": fixed,
            "k": config.baseline.top_k,
            "chunks": to_retrieved_chunks(fixed.hits),
            "timings": _with_timing(state, "retrieval", started),
        }

    async def _assess_confidence(self, state: GraphState) -> dict[str, Any]:
        retrieval = state["retrieval"]
        assert retrieval is not None
        return {"confidence": assess_confidence(retrieval, state["config"].adaptive)}

    def _route_confidence(self, state: GraphState) -> str:
        confidence, retrieval, config = state["confidence"], state["retrieval"], state["config"].adaptive
        assert confidence is not None and retrieval is not None
        can_expand = retrieval.expansions < config.max_expansions and retrieval.k < config.k_cap
        return "expand_retrieval" if not confidence.sufficient and can_expand else "generate"

    async def _expand_retrieval(self, state: GraphState) -> dict[str, Any]:
        started = perf_counter()
        retrieval, confidence = state["retrieval"], state["confidence"]
        assert retrieval is not None and confidence is not None
        await self.c.adaptive.expand(retrieval, confidence, state["config"].adaptive)
        return {
            "retrieval": retrieval,
            "chunks": _fused_to_chunks(retrieval.selected),
            "timings": _with_timing(state, "retrieval", started),
        }

    async def _generate(self, state: GraphState) -> dict[str, Any]:
        started = perf_counter()
        config, usage, chunks = state["config"], state["usage"], state["chunks"]
        generation = await self.c.generator.generate(
            state["question"], chunks, max_tokens=config.generation.max_answer_tokens, revision=state.get("revision")
        )
        usage.record_chat("generation", self.c.generator.model_name, generation.message, generation.prompt_text)
        parsed = parse_citations(generation.answer, len(chunks))
        attempt = AttemptSummary(
            attempt=len(state["attempts"]),
            answer=generation.answer,
            sentences=build_sentences(parsed),
            invalid_citations=parsed.invalid,
            num_context_chunks=len(chunks),
            verification=None,
            latency_ms=0.0,
            cumulative_latency_ms=0.0,
            cumulative_input_tokens=0,
            cumulative_output_tokens=0,
            cumulative_cost_usd=None,
        )
        update: dict[str, Any] = {"attempt_started": started, "timings": _with_timing(state, "generation", started)}
        if not config.proposed.verify:
            attempt = self._close_attempt(state, attempt, started)
        update["attempts"] = [*state["attempts"], attempt]
        return update

    async def _extract_claims(self, state: GraphState) -> dict[str, Any]:
        started = perf_counter()
        config, attempt = state["config"].verification, state["attempts"][-1]
        claims, fallback = await self.c.extractor.extract(
            attempt.sentences, max_claims=config.max_claims, max_tokens=config.max_tokens,
            usage=state["usage"], attempt=attempt.attempt,
        )
        return {"claims": claims, "claims_fallback": fallback, "timings": _with_timing(state, "claim_extraction", started)}

    async def _verify(self, state: GraphState) -> dict[str, Any]:
        started = perf_counter()
        attempt = state["attempts"][-1]
        report = await self.c.verifier.verify(
            state["claims"], state["chunks"], state["config"].verification, state["usage"], state["embedding_cache"]
        )
        report = report.model_copy(update={"extraction_fallback": state["claims_fallback"]})
        attempt = attempt.model_copy(
            update={"verification": report, "sentences": annotate_sentences(attempt.sentences, report)}
        )
        attempt = self._close_attempt(state, attempt, state["attempt_started"])
        return {"attempts": [*state["attempts"][:-1], attempt], "timings": _with_timing(state, "verification", started)}

    def _route_verification(self, state: GraphState) -> str:
        report = state["attempts"][-1].verification
        assert report is not None
        retries_used = len(state["attempts"]) - 1
        if report.passed or retries_used >= state["config"].verification.max_retries:
            return "finalize"
        return "refine"

    async def _refine(self, state: GraphState) -> dict[str, Any]:
        """Targeted retrieval for claims lacking evidence, plus revision notes for regeneration."""
        started = perf_counter()
        config, usage, attempt = state["config"], state["usage"], state["attempts"][-1]
        report = attempt.verification
        assert report is not None
        unsupported = [c for c in report.claims if c.status is SupportStatus.UNSUPPORTED]
        contradicted = [c for c in unsupported if c.failure == "contradicted"]
        lacking = [c for c in unsupported if c.failure != "contradicted"]
        partial = [c for c in report.claims if c.status is SupportStatus.PARTIAL]

        chunks = list(state["chunks"])
        seen = {c.chunk_id for c in chunks}
        added: list[RetrievedChunk] = []
        targets = [c.text for c in lacking + partial][:MAX_REFINEMENT_QUERIES]
        if targets:
            vectors, _ = await self.c.dense.embed_queries(targets)
            usage.record_embedding("refinement_embedding", self.c.embedding_model, targets)
            depth = config.verification.refine_k + len(seen)
            for vector in vectors:
                hits, _ = await self.c.dense.search(state["corpus"], vector, depth)
                fresh = [h for h in hits if h.chunk.chunk_id not in seen][: config.verification.refine_k]
                for hit in fresh:
                    seen.add(hit.chunk.chunk_id)
                    number = len(chunks) + len(added) + 1
                    added.append(
                        RetrievedChunk(
                            chunk_id=hit.chunk.chunk_id, doc_id=hit.chunk.doc_id, filename=hit.chunk.filename,
                            page=hit.chunk.page, chunk_index=hit.chunk.chunk_index, text=hit.chunk.text,
                            score=hit.score, rank=number, citation_number=number,
                            retrieved_by=f"refinement:{attempt.attempt + 1}",
                        )
                    )

        notes = []
        for claim in lacking:
            notes.append(f'- "{claim.text}" is not supported by the passages. Remove it unless a passage states it.')
        for claim in contradicted:
            where = f" passage [{claim.evidence.citation_number}]" if claim.evidence else " the passages"
            notes.append(f'- "{claim.text}" conflicts with{where}. Correct it to match the passage or remove it.')
        for claim in partial:
            notes.append(f'- "{claim.text}" is only partially supported. State only what the passages support.')
        for claim in report.claims:
            if claim.suggested_citation is not None and claim.status is not SupportStatus.UNSUPPORTED:
                notes.append(f'- "{claim.text}" should cite passage [{claim.suggested_citation}].')
        if added:
            notes.append(f"- New passages [{added[0].citation_number}]-[{added[-1].citation_number}] were added to the context.")

        step = RefinementStep(
            attempt=attempt.attempt + 1,
            targeted_claims=targets,
            contradicted_claims=[c.text for c in contradicted],
            added_chunk_ids=[c.chunk_id for c in added],
            latency_ms=(perf_counter() - started) * 1000,
        )
        return {
            "chunks": chunks + added,
            "revision": Revision(previous_answer=attempt.answer, notes="\n".join(notes) or "- Improve support for every claim."),
            "refinements": [*state["refinements"], step],
            "timings": _with_timing(state, "refinement", started),
        }

    async def _finalize(self, state: GraphState) -> dict[str, Any]:
        config, attempts, chunks = state["config"], state["attempts"], state["chunks"]
        verify = config.proposed.verify
        index = select_best_attempt(attempts) if verify else len(attempts) - 1
        chosen = attempts[index]
        report = chosen.verification

        if verify:
            answer, removed = strip_unsupported(chosen.answer, chosen.sentences)
            uncertainty = build_uncertainty(report, removed)
            sentences = [s for s in chosen.sentences if s.status is not SupportStatus.UNSUPPORTED]
            if answer == NO_VERIFIED_ANSWER:
                sentences = []
        else:
            answer, uncertainty, sentences = chosen.answer, None, chosen.sentences
        parsed = parse_citations(answer, len(chunks))

        timings = dict(state.get("timings", {}))
        timings["total"] = (perf_counter() - state["started"]) * 1000
        usage = state["usage"].summary()
        if config.proposed.adaptive:
            retrieval_state = state["retrieval"]
            assert retrieval_state is not None
            trace = RetrievalTrace(
                strategy=AdaptiveRetriever.strategy,
                k_requested=retrieval_state.k_initial,
                k_final=retrieval_state.k,
                num_retrieved=len(chunks),
                latency_ms=retrieval_state.latency_ms,
                embedding_latency_ms=retrieval_state.embedding_latency_ms,
                search_latency_ms=retrieval_state.search_latency_ms,
                queries=retrieval_state.queries,
                confidence=state["confidence"],
                expansions=retrieval_state.expansions,
                chunks=chunks,
            )
        else:
            fixed = state["fixed_retrieval"]
            assert fixed is not None
            trace = RetrievalTrace(
                strategy="fixed_top_k",
                k_requested=fixed.k_requested,
                k_final=fixed.k_requested,
                num_retrieved=len(chunks),
                latency_ms=fixed.latency_ms,
                embedding_latency_ms=fixed.embedding_latency_ms,
                search_latency_ms=fixed.search_latency_ms,
                queries=[state["question"]],
                chunks=chunks,
            )

        response = QueryResponse(
            query_id=uuid4().hex,
            created_at=datetime.now(timezone.utc),
            pipeline=self.name,
            variant={
                "adaptive": config.proposed.adaptive,
                "decompose": config.proposed.adaptive and config.adaptive.decompose,
                "verify": verify,
            },
            corpus=state["corpus"],
            question=state["question"],
            answer=answer,
            citations=build_citations(parsed, chunks),
            sentences=sentences,
            invalid_citations=chosen.invalid_citations,
            retrieval=trace,
            complexity=state.get("assessment"),
            verification=report,
            attempts=attempts,
            selected_attempt=index,
            refinements=state["refinements"],
            uncertainty=uncertainty,
            usage=usage,
            timings_ms=timings,
            total_latency_ms=timings["total"],
            models={
                "chat": self.c.generator.model_name,
                "embedding": self.c.embedding_model,
                **({"nli": config.verification.nli_model} if verify else {}),
            },
            config=config.model_dump(mode="json"),
        )
        return {"response": response}

    def _close_attempt(self, state: GraphState, attempt: AttemptSummary, attempt_started: float) -> AttemptSummary:
        summary = state["usage"].summary()
        return attempt.model_copy(
            update={
                "latency_ms": (perf_counter() - attempt_started) * 1000,
                "cumulative_latency_ms": (perf_counter() - state["started"]) * 1000,
                "cumulative_input_tokens": summary.input_tokens,
                "cumulative_output_tokens": summary.output_tokens,
                "cumulative_cost_usd": summary.cost_usd,
            }
        )


def _summarise(node: str, state: dict[str, Any]) -> dict[str, Any]:
    """Small, JSON-friendly progress payload for streaming clients."""
    if node == "analyze_query" and state.get("assessment"):
        a = state["assessment"]
        return {"label": a.label, "score": a.score, "reasoning_type": a.reasoning_type, "information_needs": a.information_needs}
    if node == "select_depth":
        return {"k": state.get("k")}
    if node in {"retrieve", "expand_retrieval"}:
        retrieval = state.get("retrieval") or state.get("fixed_retrieval")
        return {"num_chunks": len(state.get("chunks", [])), "k": getattr(retrieval, "k", state.get("k"))}
    if node == "assess_confidence" and state.get("confidence"):
        c = state["confidence"]
        return {"confidence": c.score, "sufficient": c.sufficient, "coverage": c.coverage}
    if node == "generate":
        attempt = state["attempts"][-1]
        return {"attempt": attempt.attempt, "preview": attempt.answer[:240]}
    if node == "extract_claims":
        return {"num_claims": len(state.get("claims", []))}
    if node == "verify":
        report = state["attempts"][-1].verification
        return {"support_score": report.support_score, "unsupported": report.num_unsupported, "passed": report.passed}
    if node == "refine":
        step = state["refinements"][-1]
        return {"added_chunks": len(step.added_chunk_ids), "targeted_claims": len(step.targeted_claims)}
    return {}
