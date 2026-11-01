"""JSON-structured LLM calls validated against Pydantic schemas.

Uses the provider's JSON mode plus schema validation, with one repair round when the reply does not
validate. This works with any chat model (including offline test doubles), unlike tool-calling
based structured output.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from app.core.errors import AppError
from app.core.llm import message_text
from app.core.usage import UsageTracker

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class StructuredOutputError(AppError):
    status_code = 502


def _extract_json(text: str) -> str:
    text = _FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


async def structured_call(
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    schema: type[T],
    *,
    stage: str,
    model_name: str,
    usage: UsageTracker | None = None,
    max_tokens: int = 800,
    attempts: int = 2,
) -> T:
    bound = model.bind(response_format={"type": "json_object"}, max_tokens=max_tokens)
    conversation = list(messages)
    error: ValidationError | None = None
    for _ in range(attempts):
        response = await bound.ainvoke(conversation)
        if usage is not None:
            usage.record_chat(stage, model_name, response, "\n\n".join(message_text(m) for m in conversation))
        try:
            return schema.model_validate_json(_extract_json(message_text(response)))
        except ValidationError as exc:
            error = exc
            conversation = [
                *conversation,
                response,
                HumanMessage(
                    content=f"That reply did not match the required JSON format ({exc.errors()[:3]}). "
                    "Reply again with only the corrected JSON object."
                ),
            ]
    raise StructuredOutputError(f"{stage}: no valid JSON after {attempts} attempts: {error}")
