export type Status = 'supported' | 'partial' | 'unsupported';
export type PipelineName = 'baseline' | 'proposed';

export interface Sentence {
  index: number;
  text: string;
  citation_numbers: number[];
  status: Status | null;
}

export interface RetrievedChunk {
  chunk_id: string;
  doc_id: string;
  filename: string;
  page: number | null;
  chunk_index: number;
  text: string;
  score: number;
  rank: number;
  citation_number: number;
  retrieved_by: string;
}

export interface RetrievalConfidence {
  score: number;
  top1: number;
  mean: number;
  coverage: number;
  uncovered_queries: string[];
  sufficient: boolean;
}

export interface RetrievalTrace {
  strategy: string;
  k_requested: number;
  k_final: number | null;
  num_retrieved: number;
  latency_ms: number;
  embedding_latency_ms: number;
  search_latency_ms: number;
  queries: string[];
  confidence: RetrievalConfidence | null;
  expansions: number;
  chunks: RetrievedChunk[];
}

export interface Complexity {
  label: 'simple' | 'moderate' | 'complex';
  score: number;
  k: number;
  information_needs: string[];
  reasoning_type: string;
  requires_multiple_sources: boolean;
  llm_label: string | null;
  rationale: string;
  components: Record<string, number>;
  lexical: Record<string, number>;
  fallback: boolean;
  latency_ms: number;
}

export interface EvidenceSpan {
  citation_number: number;
  chunk_id: string;
  filename: string;
  page: number | null;
  text: string;
  similarity: number;
  entailment: number;
  neutral: number;
  contradiction: number;
}

export interface ClaimVerification {
  claim_id: string;
  text: string;
  sentence_index: number;
  citations: number[];
  status: Status;
  citation_status: 'correct' | 'misattributed' | 'missing' | 'incorrect';
  failure: 'no_evidence' | 'contradicted' | null;
  method: 'nli' | 'llm_judge' | 'nli_threshold';
  evidence: EvidenceSpan | null;
  suggested_citation: number | null;
  judge_rationale: string | null;
  judge_quote: string | null;
  judge_quote_verified: boolean | null;
}

export interface VerificationReport {
  claims: ClaimVerification[];
  support_score: number;
  citation_accuracy: number | null;
  num_supported: number;
  num_partial: number;
  num_unsupported: number;
  passed: boolean;
  abstained: boolean;
  extraction_fallback: boolean;
  nli_model: string;
  latency_ms: number;
}

export interface Attempt {
  attempt: number;
  answer: string;
  sentences: Sentence[];
  invalid_citations: number[];
  num_context_chunks: number;
  verification: VerificationReport | null;
  latency_ms: number;
  cumulative_latency_ms: number;
  cumulative_input_tokens: number;
  cumulative_output_tokens: number;
  cumulative_cost_usd: number | null;
}

export interface Refinement {
  attempt: number;
  targeted_claims: string[];
  contradicted_claims: string[];
  added_chunk_ids: string[];
  latency_ms: number;
}

export interface Uncertainty {
  flagged: boolean;
  notice: string | null;
  unsupported_claims: string[];
  removed_sentences: string[];
}

export interface StageUsage {
  stage: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  estimated: boolean;
  cost_usd: number | null;
}

export interface Usage {
  stages: StageUsage[];
  input_tokens: number;
  output_tokens: number;
  cost_usd: number | null;
}

export interface Citation {
  number: number;
  chunk_id: string;
  doc_id: string;
  filename: string;
  page: number | null;
}

export interface QueryResponse {
  query_id: string;
  created_at: string;
  pipeline: PipelineName;
  variant: Record<string, boolean>;
  corpus: string;
  question: string;
  answer: string;
  citations: Citation[];
  sentences: Sentence[];
  invalid_citations: number[];
  retrieval: RetrievalTrace;
  complexity: Complexity | null;
  verification: VerificationReport | null;
  attempts: Attempt[];
  selected_attempt: number | null;
  refinements: Refinement[];
  uncertainty: Uncertainty | null;
  usage: Usage;
  timings_ms: Record<string, number>;
  total_latency_ms: number;
  models: Record<string, string>;
  config: Record<string, unknown>;
  /** True when this is a replay of a saved real run; latency, tokens and cost are those of the original run. */
  cached?: boolean;
  cached_from?: string | null;
  served_ms?: number | null;
}

export interface QueryLog {
  id: string;
  created_at: string;
  pipeline: string;
  corpus: string;
  question: string;
  complexity_label: string | null;
  complexity_score: number | null;
  k_selected: number;
  num_retrieved: number;
  retrieval_latency_ms: number;
  total_latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number | null;
}

export interface DocumentRecord {
  id: string;
  corpus: string;
  filename: string;
  file_type: string;
  file_size: number;
  status: 'processing' | 'ready' | 'failed';
  error: string | null;
  num_pages: number | null;
  num_chunks: number;
  chunk_size: number;
  chunk_overlap: number;
  created_at: string;
}

export interface StepEvent {
  type: 'step';
  node: string;
  summary: Record<string, unknown>;
}
export type StreamEvent =
  | StepEvent
  | { type: 'result'; response: QueryResponse }
  | { type: 'error'; detail: string };

export interface ExperimentListItem {
  name: string;
  created_at: string | null;
  description: string | null;
  num_questions: number | null;
  variants: string[];
}
