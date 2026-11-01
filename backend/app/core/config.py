"""Configuration.

Two layers are kept deliberately separate:

* ``Settings``: environment-specific values (API keys, model IDs, storage locations), read from
  environment variables or ``.env``. Never committed.
* ``PipelineConfig``: research parameters (chunking, retrieval depth, prices, ...). Defaults live
  here; an optional YAML file, per-request overrides and experiment configs can change them.
  Every query trace and experiment run stores a snapshot of the ``PipelineConfig`` it used.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import InvalidConfigError

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Adaptive RAG Research Prototype"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.0
    llm_seed: int | None = 42
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3

    data_dir: Path = BACKEND_DIR / "data"
    database_url: str | None = None
    chroma_dir: Path | None = None
    upload_dir: Path | None = None
    pipeline_config_path: Path | None = None

    default_corpus: str = "default"
    max_upload_mb: int = 25
    cors_origins: list[str] = ["http://localhost:5173"]

    # Evaluation-only models; never used inside the pipelines being evaluated.
    judge_model: str = "gpt-4.1"
    eval_nli_model: str = "cross-encoder/nli-roberta-base"
    nli_device: str = "cpu"
    preload_nli: bool = True
    seed_demo_corpus: bool = True
    experiments_dir: Path = PROJECT_ROOT / "experiments"

    @model_validator(mode="after")
    def _derive_paths(self) -> Settings:
        if not self.openai_base_url:
            self.openai_base_url = None
        if not self.database_url:
            self.database_url = f"sqlite+aiosqlite:///{(self.data_dir / 'app.db').as_posix()}"
        if self.chroma_dir is None:
            self.chroma_dir = self.data_dir / "chroma"
        if self.upload_dir is None:
            self.upload_dir = self.data_dir / "uploads"
        return self

    @property
    def llm_configured(self) -> bool:
        return self.openai_api_key is not None and bool(self.openai_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()


class _StrictModel(BaseModel):
    # Unknown keys are rejected so a typo in an experiment config cannot silently fall back to a default.
    model_config = ConfigDict(extra="forbid")


class ChunkingConfig(_StrictModel):
    chunk_size: int = Field(1000, ge=100, description="Maximum characters per chunk.")
    chunk_overlap: int = Field(150, ge=0, description="Characters shared by consecutive chunks.")

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class BaselineConfig(_StrictModel):
    top_k: int = Field(5, ge=1, le=50, description="Fixed retrieval depth of the conventional RAG baseline.")


class GenerationConfig(_StrictModel):
    max_answer_tokens: int = Field(700, ge=50, le=4000)


class ComplexityWeights(_StrictModel):
    """Weights of the complexity score components (normalised by their sum)."""

    information_needs: float = Field(0.35, ge=0)
    reasoning: float = Field(0.25, ge=0)
    multi_source: float = Field(0.15, ge=0)
    lexical: float = Field(0.15, ge=0)
    llm_label: float = Field(0.10, ge=0)

    @model_validator(mode="after")
    def _positive_total(self) -> ComplexityWeights:
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("at least one complexity weight must be positive")
        return self


def _default_reasoning_scores() -> dict[str, float]:
    return {"lookup": 0.0, "aggregation": 0.5, "temporal": 0.5, "comparison": 0.6, "causal": 0.7, "bridge": 1.0}


class ComplexityConfig(_StrictModel):
    weights: ComplexityWeights = Field(default_factory=ComplexityWeights)
    reasoning_scores: dict[str, float] = Field(default_factory=_default_reasoning_scores)
    simple_max: float = Field(0.33, gt=0, lt=1, description="Scores below this are 'simple'.")
    moderate_max: float = Field(0.66, gt=0, lt=1, description="Scores below this (and >= simple_max) are 'moderate'.")
    max_tokens: int = Field(500, ge=100)

    @model_validator(mode="after")
    def _ordered_thresholds(self) -> ComplexityConfig:
        if not self.simple_max < self.moderate_max:
            raise ValueError("simple_max must be smaller than moderate_max")
        missing = {"lookup", "aggregation", "temporal", "comparison", "causal", "bridge"} - set(self.reasoning_scores)
        if missing:
            raise ValueError(f"reasoning_scores is missing {sorted(missing)}")
        return self


class KBand(_StrictModel):
    min: int = Field(ge=1)
    max: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> KBand:
        if self.min > self.max:
            raise ValueError("k band min must not exceed max")
        return self


class AdaptiveConfig(_StrictModel):
    k_simple: KBand = Field(default_factory=lambda: KBand(min=1, max=3))
    k_moderate: KBand = Field(default_factory=lambda: KBand(min=4, max=7))
    k_complex: KBand = Field(default_factory=lambda: KBand(min=8, max=12))
    k_cap: int = Field(15, ge=1, description="Upper bound on k, including expansions.")
    decompose: bool = Field(True, description="Retrieve separately for each information need and fuse (RRF).")
    max_subqueries: int = Field(4, ge=1)
    rrf_k: int = Field(60, ge=1)
    # Cosine-similarity normalisation for the confidence estimate; calibrated on the dev split.
    score_floor: float = 0.15
    score_ceiling: float = 0.65
    relevance_threshold: float = Field(0.40, description="A query counts as covered if a chunk scores at least this.")
    confidence_weight_top1: float = Field(0.5, ge=0)
    confidence_weight_mean: float = Field(0.2, ge=0)
    confidence_weight_coverage: float = Field(0.3, ge=0)
    confidence_threshold: float = Field(0.5, ge=0, le=1)
    max_expansions: int = Field(1, ge=0, le=5)
    expansion_step: int = Field(3, ge=1)

    @model_validator(mode="after")
    def _check(self) -> AdaptiveConfig:
        if self.score_ceiling <= self.score_floor:
            raise ValueError("score_ceiling must be greater than score_floor")
        if self.k_complex.max > self.k_cap:
            raise ValueError("k_cap must be at least k_complex.max")
        return self

    def band(self, label: str) -> KBand:
        return {"simple": self.k_simple, "moderate": self.k_moderate, "complex": self.k_complex}[label]


class VerificationConfig(_StrictModel):
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    window_sentences: int = Field(3, ge=1, description="Sentences per evidence window tested with NLI.")
    candidate_windows: int = Field(3, ge=1, description="Windows per claim selected by embedding similarity.")
    support_threshold: float = Field(0.75, gt=0, le=1)
    contradiction_threshold: float = Field(0.75, gt=0, le=1)
    partial_threshold: float = Field(0.40, gt=0, le=1)
    use_llm_judge: bool = Field(True, description="Resolve NLI-ambiguous claims with an LLM judge.")
    pass_support_score: float = Field(0.9, ge=0, le=1)
    max_claims: int = Field(12, ge=1)
    max_retries: int = Field(2, ge=0, le=5)
    refine_k: int = Field(3, ge=1, description="Chunks retrieved per unsupported claim during refinement.")
    max_tokens: int = Field(900, ge=100)

    @model_validator(mode="after")
    def _ordered(self) -> VerificationConfig:
        if self.partial_threshold > self.support_threshold:
            raise ValueError("partial_threshold must not exceed support_threshold")
        return self


class ProposedConfig(_StrictModel):
    """Feature flags of the proposed pipeline; switching them off yields the ablation variants."""

    adaptive: bool = True
    verify: bool = True


class ModelPrice(_StrictModel):
    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(0.0, ge=0)


def _default_pricing() -> dict[str, ModelPrice]:
    # USD per 1M tokens as published by OpenAI at the time of writing. Verify against current
    # pricing before reporting costs; models missing from this table are reported with cost=None.
    return {
        "gpt-4o-mini": ModelPrice(input_per_million=0.15, output_per_million=0.60),
        "gpt-4o": ModelPrice(input_per_million=2.50, output_per_million=10.00),
        "gpt-4.1": ModelPrice(input_per_million=2.00, output_per_million=8.00),
        "gpt-4.1-mini": ModelPrice(input_per_million=0.40, output_per_million=1.60),
        "text-embedding-3-small": ModelPrice(input_per_million=0.02),
        "text-embedding-3-large": ModelPrice(input_per_million=0.13),
    }


class PipelineConfig(_StrictModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    proposed: ProposedConfig = Field(default_factory=ProposedConfig)
    complexity: ComplexityConfig = Field(default_factory=ComplexityConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    pricing: dict[str, ModelPrice] = Field(default_factory=_default_pricing)

    def with_overrides(self, overrides: dict[str, Any] | None) -> PipelineConfig:
        """Return a copy with ``overrides`` deep-merged in, re-validated as a whole."""
        if not overrides:
            return self
        merged = _deep_merge(self.model_dump(), overrides)
        try:
            return PipelineConfig.model_validate(merged)
        except ValidationError as exc:
            raise InvalidConfigError(f"Invalid pipeline overrides: {exc}") from exc


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_pipeline_config(path: Path | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return PipelineConfig.model_validate(data)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise InvalidConfigError(f"Could not load pipeline config from {path}: {exc}") from exc
