"""Offline test doubles for the LLM provider and the NLI model. Used only by tests."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.generation.prompts import GENERATION_SYSTEM_PROMPT
from app.retrieval.complexity_classifier import ANALYSIS_PROMPT
from app.verification.claim_extractor import EXTRACTION_PROMPT
from app.verification.nli import NLIResult
from app.verification.verifier import JUDGE_PROMPT

_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "is", "was", "and", "or", "by", "for", "with", "it", "its",
    "that", "this", "which", "who", "what", "when", "where", "how", "are", "were", "be", "as", "from", "has", "have",
}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words vectors: texts sharing words are close, so retrieval tests are meaningful."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.document_calls = 0
        self.query_calls = 0

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._embed(text)


class OverlapNLI:
    """Word-overlap stand-in for an NLI model.

    entailment when nearly all content words of the hypothesis occur in the premise; contradiction when
    most words overlap but the hypothesis contains a number absent from the premise; neutral otherwise.
    """

    status = "ready"

    def __init__(self, name: str = "fake-overlap-nli") -> None:
        self.name = name
        self.calls = 0

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIResult]:
        self.calls += 1
        results = []
        for premise, hypothesis in pairs:
            hyp, prem = content_words(hypothesis), content_words(premise)
            overlap = len(hyp & prem) / max(len(hyp), 1)
            numbers = set(re.findall(r"\d+", hypothesis)) - set(re.findall(r"\d+", premise))
            if numbers and overlap >= 0.5:
                results.append(NLIResult(0.03, 0.07, 0.90))
            elif overlap >= 0.85:
                results.append(NLIResult(0.92, 0.06, 0.02))
            elif overlap >= 0.6:
                results.append(NLIResult(0.50, 0.45, 0.05))  # ambiguous zone -> judge
            else:
                results.append(NLIResult(0.02, 0.95, 0.03))
        return results


class ScriptedChatModel(BaseChatModel):
    """Chat model whose reply is computed from the prompt by ``responder``; records every call."""

    responder: Callable[[list[BaseMessage]], str]
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    call_kwargs: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.calls.append(messages)
        self.call_kwargs.append(kwargs)
        message = AIMessage(
            content=self.responder(messages),
            usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def calls_for(self, system_prompt: str) -> list[list[BaseMessage]]:
        return [c for c in self.calls if c and c[0].content == system_prompt]


def cite_first_passage(messages: list[BaseMessage]) -> str:
    """Answers with the first sentence of passage [1], cited, plus one citation to a missing passage."""
    prompt = str(messages[-1].content)
    match = re.search(r"\[1\] \(source: [^)]*\)\n([^\n]+)", prompt)
    first = match.group(1).split(". ")[0].rstrip(".") if match else "No passage"
    return f"{first} [1]. This sentence cites a passage that does not exist [99]."


HALLUCINATION = "The institute also operates an observatory on the Moon"


def _passages(prompt: str) -> list[tuple[int, str]]:
    return [(int(n), text.strip()) for n, text in re.findall(r"\[(\d+)\] \(source: [^)]*\)\n(.*?)(?=\n\n\[\d+\] |\n\nQuestion:)", prompt, re.S)]


def research_responder(hallucinate_until_revision: bool = True, always_hallucinate: bool = False) -> Callable[[list[BaseMessage]], str]:
    """Plays every LLM role of the proposed pipeline from the prompt contents.

    The generator answers with the best-matching passage sentence and, before any revision feedback,
    adds an unsupported sentence, so the verification -> refinement loop is exercised.
    """

    def respond(messages: list[BaseMessage]) -> str:
        system, user = str(messages[0].content), str(messages[-1].content)
        if system == ANALYSIS_PROMPT:
            question = user.removeprefix("Question: ")
            pieces = re.split(r",?\s+and\s+(?=(?:who|what|when|where|which|how|since)\b)", question)
            parts = [p.strip(" ?") + "?" for p in pieces if p.strip(" ?")]
            multi = len(parts) > 1
            return json.dumps({
                "information_needs": parts,
                "reasoning_type": "bridge" if multi else "lookup",
                "requires_multiple_sources": multi,
                "complexity": "complex" if len(parts) > 2 else ("moderate" if multi else "simple"),
                "rationale": "test double",
            })
        if system == EXTRACTION_PROMPT:
            claims = []
            for index, text in re.findall(r"\[s(\d+)\] (.+)", user):
                clean = re.sub(r"\s*\[\d+(?:[,;]\s*\d+)*\]", "", text).strip()
                if "not contain" not in clean:
                    claims.append({"text": clean.rstrip("."), "sentence_index": int(index)})
            return json.dumps({"claims": claims})
        if system == JUDGE_PROMPT:
            claim = user.rsplit("Claim: ", 1)[-1]
            best = None
            for number, text in re.findall(r"Passage \[(\d+)\] \(source: [^)]*\)\n(.+?)(?=\n\nPassage \[|\n\nClaim:)", user, re.S):
                for sentence in re.split(r"(?<=[.!?])\s+", text):
                    overlap = len(content_words(claim) & content_words(sentence)) / max(len(content_words(claim)), 1)
                    if best is None or overlap > best[0]:
                        best = (overlap, int(number), sentence)
            if best and best[0] >= 0.6:
                return json.dumps({"verdict": "supported", "passage": best[1], "evidence_quote": best[2], "rationale": "overlap"})
            return json.dumps({"verdict": "unsupported", "passage": None, "evidence_quote": "", "rationale": "no overlap"})
        if system == GENERATION_SYSTEM_PROMPT:
            question = user.rsplit("Question: ", 1)[-1].split("\n\n")[0]
            passages = _passages(user)
            if not passages:
                return "The passages do not contain the answer."
            scored = []
            for number, text in passages:
                first = re.split(r"(?<=[.!?])\s+", text)[0].rstrip(".")
                scored.append((len(content_words(question) & content_words(first)), number, first))
            scored.sort(key=lambda s: -s[0])
            answer = f"{scored[0][2]} [{scored[0][1]}]."
            revised = "Verification feedback" in user
            if always_hallucinate or (hallucinate_until_revision and not revised):
                answer += f" {HALLUCINATION} [1]."
            return answer
        raise AssertionError(f"unexpected prompt: {system[:80]}")

    return respond
