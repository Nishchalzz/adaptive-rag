"""Token and cost accounting per pipeline stage.

Chat token counts come from the provider's usage metadata. Embedding requests do not report
usage through LangChain, so their token counts are estimated with tiktoken and flagged as such.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.messages import BaseMessage

from app.core.config import ModelPrice
from app.core.llm import message_text
from app.schemas.query import StageUsage, UsageSummary

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _encoding():  # type: ignore[no-untyped-def]
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 - tokenizer download can fail offline; fall back to a heuristic
        logger.warning("tiktoken encoding unavailable; estimating tokens as characters/4")
        return None


def estimate_tokens(text: str) -> int:
    encoding = _encoding()
    if encoding is None:
        return max(1, len(text) // 4)
    return len(encoding.encode(text, disallowed_special=()))


@dataclass
class UsageRecord:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated: bool


@dataclass
class UsageTracker:
    pricing: Mapping[str, ModelPrice]
    records: list[UsageRecord] = field(default_factory=list)

    def record_chat(self, stage: str, model: str, message: BaseMessage, prompt_text: str) -> None:
        usage = getattr(message, "usage_metadata", None)
        if usage:
            self.records.append(
                UsageRecord(stage, model, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)), False)
            )
        else:
            self.records.append(
                UsageRecord(stage, model, estimate_tokens(prompt_text), estimate_tokens(message_text(message)), True)
            )

    def record_embedding(self, stage: str, model: str, texts: Sequence[str]) -> None:
        tokens = sum(estimate_tokens(t) for t in texts)
        self.records.append(UsageRecord(stage, model, tokens, 0, True))

    def _cost(self, record: UsageRecord) -> float | None:
        price = self.pricing.get(record.model)
        if price is None:
            return None
        return (record.input_tokens * price.input_per_million + record.output_tokens * price.output_per_million) / 1e6

    def summary(self) -> UsageSummary:
        stages = [
            StageUsage(
                stage=r.stage,
                model=r.model,
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                estimated=r.estimated,
                cost_usd=self._cost(r),
            )
            for r in self.records
        ]
        costs = [s.cost_usd for s in stages]
        return UsageSummary(
            stages=stages,
            input_tokens=sum(s.input_tokens for s in stages),
            output_tokens=sum(s.output_tokens for s in stages),
            cost_usd=None if any(c is None for c in costs) else sum(c for c in costs if c is not None),
        )
