"""The answer-generation prompt.

This single prompt is used by every pipeline (baseline, ablations, proposed system) so differences
in results cannot come from prompt wording. Changing it changes all pipelines at once.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


class ContextPassage(Protocol):
    """Anything with a source and text: stored ``Chunk``s and ``RetrievedChunk``s both qualify."""

    @property
    def filename(self) -> str: ...

    @property
    def page(self) -> int | None: ...

    @property
    def text(self) -> str: ...

GENERATION_SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's question using only the numbered context passages provided.

Rules:
1. Use only information stated in the passages. Do not add facts from prior knowledge.
2. Support every factual sentence with citations to the passages it relies on, written as bracketed passage numbers at the end of the sentence, for example [2] or [1][3].
3. If the passages do not contain enough information to answer fully, say so explicitly and state what is missing. Do not guess.
4. Be concise and direct."""


def format_context(chunks: Sequence[ContextPassage]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        source = chunk.filename if chunk.page is None else f"{chunk.filename}, page {chunk.page}"
        blocks.append(f"[{number}] (source: {source})\n{chunk.text}")
    return "\n\n".join(blocks)


@dataclass(frozen=True)
class Revision:
    """Feedback from evidence verification for a regeneration attempt (proposed pipeline only)."""

    previous_answer: str
    notes: str


def build_generation_messages(
    question: str, chunks: Sequence[ContextPassage], revision: Revision | None = None
) -> list[BaseMessage]:
    user = f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {question}"
    if revision is not None:
        user += (
            f"\n\nYour previous answer:\n{revision.previous_answer}\n\n"
            f"Verification feedback on the previous answer:\n{revision.notes}\n\n"
            "Write a corrected, complete answer that follows all the rules. Keep only statements that the "
            "passages support, and say explicitly if the passages do not contain the information."
        )
    return [SystemMessage(content=GENERATION_SYSTEM_PROMPT), HumanMessage(content=user)]
