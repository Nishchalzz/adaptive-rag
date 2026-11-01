"""Independent LLM judge for answer correctness and relevance (gold-answer based).

The judge model (``Settings.judge_model``, stronger than the pipeline's chat model) never sees the retrieved
passages or the pipeline's verification output; it compares the final answer to the MuSiQue gold answer.
Verdicts are cached on disk, so reruns do not pay twice.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.structured import structured_call
from app.core.usage import UsageTracker
from app.generation.citations import strip_citations

JUDGE_PROMPT = """You grade a system's answer to a question against the gold answer.

Verdict:
- "correct": the answer states the gold answer or an equivalent (different formatting, a longer or shorter form of the same name, or a number or date written differently). Extra correct detail is fine.
- "incorrect": the answer gives a different answer, contradicts the gold answer, or only gives part of a multi-part answer.
- "abstained": the answer declines to answer or says the information is not available.

Also report "addresses_question": true if the answer is on topic and attempts to answer what was asked (even if wrong); false if it is off topic or an abstention.
Ignore bracketed citation markers like [1]. Do not use outside knowledge to override the gold answer.
Return JSON: {"verdict": "correct" | "incorrect" | "abstained", "addresses_question": true | false, "rationale": "<one sentence>"}"""


class Verdict(BaseModel):
    verdict: Literal["correct", "incorrect", "abstained"]
    addresses_question: bool = True
    rationale: str = ""


class AnswerJudge:
    def __init__(self, model: BaseChatModel, model_name: str, cache_dir: Path) -> None:
        self._model = model
        self.model_name = model_name
        self._cache = cache_dir
        self._cache.mkdir(parents=True, exist_ok=True)

    async def judge(self, question: str, gold: str, aliases: list[str], answer: str, usage: UsageTracker) -> Verdict:
        answer = strip_citations(answer)
        key = hashlib.sha256(json.dumps([self.model_name, JUDGE_PROMPT, question, gold, aliases, answer]).encode()).hexdigest()
        path = self._cache / f"{key}.json"
        if path.is_file():
            return Verdict.model_validate_json(path.read_text(encoding="utf-8"))
        gold_text = gold + (f" (also acceptable: {'; '.join(aliases)})" if aliases else "")
        verdict = await structured_call(
            self._model,
            [SystemMessage(content=JUDGE_PROMPT),
             HumanMessage(content=f"Question: {question}\nGold answer: {gold_text}\nSystem answer: {answer}")],
            Verdict,
            stage="answer_judge",
            model_name=self.model_name,
            usage=usage,
            max_tokens=200,
        )
        path.write_text(verdict.model_dump_json(), encoding="utf-8")
        return verdict
