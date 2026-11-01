"""Query complexity assessment (research contribution 1, step 1).

Hybrid of an LLM analysis and deterministic lexical features:

  S = sum(w_i * f_i) / sum(w_i)   with components f_i in [0, 1]
    information_needs = min(n - 1, 3) / 3        n = atomic sub-questions identified by the LLM
    reasoning         = score of the reasoning type (lookup 0 ... bridge 1)
    multi_source      = 1 if the facts likely sit in different documents
    lexical           = comparison / enumeration / nesting / entity cues (see lexical_features)
    llm_label         = the LLM's own label mapped to 0 / 0.5 / 1

The label comes from thresholds on S, and the retrieval depth k is interpolated inside the label's
configured band. Query length is deliberately not a feature.
"""

from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from app.core.config import AdaptiveConfig, ComplexityConfig
from app.core.structured import StructuredOutputError, structured_call
from app.core.usage import UsageTracker
from app.schemas.pipeline import ComplexityAssessment, ComplexityLabel, ReasoningType

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You analyse questions for a document retrieval system. Do not answer the question.

Return a JSON object with these keys:
- "information_needs": the distinct facts that must be looked up to answer the question, each written as a standalone search question. A question that needs one fact has exactly one entry. For multi-step questions list every step in order; when a step depends on an earlier answer, describe the dependency in words (for example "Who directed the film Inception?", then "Who is the spouse of the director of Inception?"). When the question refers to something by a description instead of its name (for example "the lab that runs Project X" or "the city where Y was born"), identifying it is a separate information need.
- "reasoning_type": one of "lookup" (retrieve a single fact), "aggregation" (combine or list several facts), "comparison" (compare two or more entities), "temporal" (dates, durations or ordering in time), "causal" (explain why or how), "bridge" (one fact must be found first to know what to look up next, including any question that refers to an entity by description).
- "requires_multiple_sources": true if the facts are likely to be found in different documents or passages.
- "complexity": "simple" (one fact, direct lookup), "moderate" (a few related facts or a simple comparison), or "complex" (multi-step chains, several conditions, or synthesis across many sources).
- "rationale": one short sentence explaining the assessment."""

_LABEL_VALUE: dict[str, float] = {"simple": 0.0, "moderate": 0.5, "complex": 1.0}

_COMPARISON = re.compile(
    r"\b(compare[ds]?|comparison|versus|vs\.?|differ(?:s|ence|ences|ent)?|similar(?:ity|ities)?|both|"
    r"either|neither|than)\b|\bwhich\b[^?]*\bor\b",
    re.IGNORECASE,
)
_ENUMERATION = re.compile(r",|;|\band\b|\bas well as\b|\brespectively\b", re.IGNORECASE)
_NESTING = re.compile(r"\b(?:of|for|by|from|in|at|with) the\b|'s\b|\b(?:who|whom|whose|which|that|where)\b", re.IGNORECASE)
_ENTITY = re.compile(r"(?<!^)(?<![.?!]\s)\b[A-Z][a-zA-Z0-9'’-]+|\b\d{3,4}\b")
_LEADING_QUESTION_WORD = re.compile(r"^\s*(who|whom|whose|which|what|where|when|why|how)\b", re.IGNORECASE)


class QueryAnalysis(BaseModel):
    information_needs: list[str] = Field(min_length=1)
    reasoning_type: ReasoningType
    requires_multiple_sources: bool
    complexity: ComplexityLabel
    rationale: str = ""

    @field_validator("information_needs")
    @classmethod
    def _clean_needs(cls, needs: list[str]) -> list[str]:
        cleaned = [n.strip() for n in needs if n and n.strip()]
        if not cleaned:
            raise ValueError("information_needs must contain at least one non-empty question")
        return cleaned


def lexical_features(question: str) -> dict[str, float]:
    """Deterministic surface cues of reasoning demand, each also returned as a raw count."""
    body = _LEADING_QUESTION_WORD.sub("", question)  # the leading wh-word is not a relative clause
    comparison = len(_COMPARISON.findall(question))
    enumeration = len(_ENUMERATION.findall(question))
    nesting = len(_NESTING.findall(body))
    entities = len(_ENTITY.findall(question))
    score = (
        0.35 * min(comparison, 1)
        + 0.25 * min(enumeration / 2, 1)
        + 0.25 * min(nesting / 3, 1)
        + 0.15 * min(max(entities - 1, 0) / 3, 1)
    )
    return {
        "comparison_markers": float(comparison),
        "enumeration_markers": float(enumeration),
        "nesting_markers": float(nesting),
        "entity_mentions": float(entities),
        "score": round(score, 4),
    }


def label_for_score(score: float, config: ComplexityConfig) -> Literal["simple", "moderate", "complex"]:
    if score < config.simple_max:
        return "simple"
    if score < config.moderate_max:
        return "moderate"
    return "complex"


def select_k(score: float, label: str, complexity: ComplexityConfig, adaptive: AdaptiveConfig) -> int:
    """Interpolate k inside the label's band by where the score falls inside the label's score range."""
    low, high = {
        "simple": (0.0, complexity.simple_max),
        "moderate": (complexity.simple_max, complexity.moderate_max),
        "complex": (complexity.moderate_max, 1.0),
    }[label]
    band = adaptive.band(label)
    position = min(max((score - low) / (high - low), 0.0), 1.0)
    return band.min + round(position * (band.max - band.min))


class ComplexityClassifier:
    def __init__(self, chat_model: BaseChatModel, model_name: str) -> None:
        self._model = chat_model
        self.model_name = model_name

    async def assess(
        self, question: str, complexity: ComplexityConfig, adaptive: AdaptiveConfig, usage: UsageTracker
    ) -> ComplexityAssessment:
        started = perf_counter()
        lexical = lexical_features(question)
        try:
            analysis: QueryAnalysis | None = await structured_call(
                self._model,
                [SystemMessage(content=ANALYSIS_PROMPT), HumanMessage(content=f"Question: {question}")],
                QueryAnalysis,
                stage="complexity_analysis",
                model_name=self.model_name,
                usage=usage,
                max_tokens=complexity.max_tokens,
            )
        except StructuredOutputError:
            logger.warning("Query analysis failed; falling back to lexical features only", exc_info=True)
            analysis = None

        if analysis is None:
            components = {"lexical": lexical["score"]}
            score = lexical["score"]
            needs, reasoning, multi, llm_label, rationale = [question], "lookup", False, None, "LLM analysis unavailable"
        else:
            needs = analysis.information_needs
            reasoning, multi, llm_label, rationale = (
                analysis.reasoning_type,
                analysis.requires_multiple_sources,
                analysis.complexity,
                analysis.rationale,
            )
            components = {
                "information_needs": min(len(needs) - 1, 3) / 3,
                "reasoning": complexity.reasoning_scores[reasoning],
                "multi_source": 1.0 if multi else 0.0,
                "lexical": lexical["score"],
                "llm_label": _LABEL_VALUE[llm_label],
            }
            weights = complexity.weights.model_dump()
            score = sum(weights[name] * value for name, value in components.items()) / sum(weights.values())

        score = round(min(max(score, 0.0), 1.0), 4)
        label = label_for_score(score, complexity)
        return ComplexityAssessment(
            label=label,
            score=score,
            k=select_k(score, label, complexity, adaptive),
            information_needs=needs,
            reasoning_type=reasoning,
            requires_multiple_sources=multi,
            llm_label=llm_label,
            rationale=rationale,
            components={name: round(value, 4) for name, value in components.items()},
            lexical=lexical,
            fallback=analysis is None,
            latency_ms=(perf_counter() - started) * 1000,
        )
