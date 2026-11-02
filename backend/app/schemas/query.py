from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.documents import CorpusName
from app.schemas.pipeline import (
    ComplexityAssessment,
    RefinementStep,
    RetrievalConfidence,
    SupportStatus,
    Uncertainty,
    VerificationReport,
)


class PipelineName(str, Enum):
    BASELINE = "baseline"
    PROPOSED = "proposed"


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    pipeline: PipelineName = PipelineName.PROPOSED
    corpus: CorpusName | None = None
    overrides: dict[str, Any] | None = Field(
        default=None,
        description="Per-request PipelineConfig overrides, e.g. {'baseline': {'top_k': 8}} or "
        "{'proposed': {'verify': false}} for an ablation variant.",
    )
    use_cache: bool = Field(
        True,
        description="Allow replaying a saved real run of a bundled demo question (see demo_cache). "
        "Set false to always run the pipeline live.",
    )


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page: int | None
    chunk_index: int
    text: str
    score: float = Field(description="Cosine similarity between the retrieving query and the chunk.")
    rank: int
    citation_number: int = Field(description="Number the generator saw this chunk under, e.g. [3].")
    retrieved_by: str = Field("original_query", description="original_query, sub_query:<i> or refinement:<attempt>")


class RetrievalTrace(BaseModel):
    strategy: str
    k_requested: int = Field(description="Depth selected before expansion (fixed k for the baseline).")
    k_final: int | None = None
    num_retrieved: int = Field(description="Chunks given to the generator, including refinement retrieval.")
    latency_ms: float
    embedding_latency_ms: float
    search_latency_ms: float
    queries: list[str] = Field(default_factory=list)
    confidence: RetrievalConfidence | None = None
    expansions: int = 0
    chunks: list[RetrievedChunk]


class Citation(BaseModel):
    number: int
    chunk_id: str
    doc_id: str
    filename: str
    page: int | None


class CitedSentence(BaseModel):
    index: int
    text: str
    citation_numbers: list[int]
    status: SupportStatus | None = Field(None, description="Weakest verification status among the sentence's claims.")


class StageUsage(BaseModel):
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated: bool
    cost_usd: float | None


class UsageSummary(BaseModel):
    stages: list[StageUsage]
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = Field(description="None when any stage used a model missing from the price table.")


class AttemptSummary(BaseModel):
    attempt: int
    answer: str
    sentences: list[CitedSentence]
    invalid_citations: list[int]
    num_context_chunks: int
    verification: VerificationReport | None
    latency_ms: float
    cumulative_latency_ms: float = Field(description="Pipeline time from the start of the query to the end of this attempt.")
    cumulative_input_tokens: int
    cumulative_output_tokens: int
    cumulative_cost_usd: float | None


class QueryResponse(BaseModel):
    query_id: str
    created_at: datetime
    pipeline: PipelineName
    variant: dict[str, bool] = Field(default_factory=dict)
    corpus: str
    question: str
    answer: str
    citations: list[Citation]
    sentences: list[CitedSentence]
    invalid_citations: list[int] = Field(description="Citation numbers in the answer that match no retrieved chunk.")
    retrieval: RetrievalTrace
    complexity: ComplexityAssessment | None = None
    verification: VerificationReport | None = None
    attempts: list[AttemptSummary] = Field(default_factory=list)
    selected_attempt: int | None = None
    refinements: list[RefinementStep] = Field(default_factory=list)
    uncertainty: Uncertainty | None = None
    usage: UsageSummary
    timings_ms: dict[str, float]
    total_latency_ms: float
    models: dict[str, str]
    config: dict[str, Any]
    cached: bool = Field(False, description="True when this response is a replay of a saved real run.")
    cached_from: datetime | None = Field(None, description="When the replayed run was originally recorded.")
    served_ms: float | None = Field(None, description="Time taken to serve the replay; the pipeline timings stay those of the original run.")


class QueryLogSummary(BaseModel):
    id: str
    created_at: datetime
    pipeline: str
    corpus: str
    question: str
    complexity_label: str | None
    complexity_score: float | None
    k_selected: int
    num_retrieved: int
    retrieval_latency_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
