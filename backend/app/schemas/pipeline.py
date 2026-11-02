"""Schemas for the proposed pipeline: complexity assessment, retrieval confidence, verification, attempts."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

ComplexityLabel = Literal["simple", "moderate", "complex"]
ReasoningType = Literal["lookup", "aggregation", "comparison", "temporal", "causal", "bridge"]


class ComplexityAssessment(BaseModel):
    label: ComplexityLabel
    score: float = Field(ge=0, le=1)
    k: int = Field(description="Retrieval depth selected from label and score.")
    information_needs: list[str]
    reasoning_type: ReasoningType
    requires_multiple_sources: bool
    llm_label: ComplexityLabel | None
    rationale: str
    components: dict[str, float] = Field(description="Normalised score components before weighting.")
    lexical: dict[str, float]
    fallback: bool = Field(False, description="True when the LLM analysis failed and only lexical features were used.")
    latency_ms: float


class RetrievalConfidence(BaseModel):
    score: float
    top1: float
    mean: float
    coverage: float
    uncovered_queries: list[str]
    sufficient: bool


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class CitationStatus(str, Enum):
    CORRECT = "correct"  # a cited passage supports the claim
    MISATTRIBUTED = "misattributed"  # cited passages do not, but another retrieved passage does
    MISSING = "missing"  # the claim carries no valid citation
    INCORRECT = "incorrect"  # cited, but no retrieved passage supports it


class EvidenceSpan(BaseModel):
    citation_number: int
    chunk_id: str
    filename: str
    page: int | None
    text: str
    similarity: float
    entailment: float
    neutral: float
    contradiction: float


class ClaimVerification(BaseModel):
    claim_id: str
    text: str
    sentence_index: int
    citations: list[int]
    status: SupportStatus
    citation_status: CitationStatus
    failure: Literal["no_evidence", "contradicted"] | None = None
    method: Literal["nli", "llm_judge", "nli_threshold"]
    evidence: EvidenceSpan | None = None
    suggested_citation: int | None = Field(None, description="Passage that supports a misattributed or uncited claim.")
    judge_rationale: str | None = None
    judge_quote: str | None = None
    judge_quote_verified: bool | None = None


class VerificationReport(BaseModel):
    claims: list[ClaimVerification]
    support_score: float
    citation_accuracy: float | None
    num_supported: int
    num_partial: int
    num_unsupported: int
    passed: bool
    abstained: bool = Field(description="The answer made no factual claims (e.g. it said the documents lack the answer).")
    extraction_fallback: bool
    nli_model: str
    latency_ms: float
    timings_ms: dict[str, float] = Field(
        default_factory=dict, description="Verification sub-stages: embedding, nli, nli_pairs (count), judge."
    )


class RefinementStep(BaseModel):
    attempt: int
    targeted_claims: list[str]
    contradicted_claims: list[str]
    added_chunk_ids: list[str]
    latency_ms: float


class Uncertainty(BaseModel):
    flagged: bool
    notice: str | None
    unsupported_claims: list[str]
    removed_sentences: list[str]
