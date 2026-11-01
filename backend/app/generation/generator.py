from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from app.core.llm import message_text
from app.generation.prompts import ContextPassage, Revision, build_generation_messages


@dataclass
class GenerationResult:
    answer: str
    message: AIMessage
    prompt_text: str


class AnswerGenerator:
    def __init__(self, chat_model: BaseChatModel, model_name: str) -> None:
        self._model = chat_model
        self.model_name = model_name

    async def generate(
        self, question: str, chunks: Sequence[ContextPassage], *, max_tokens: int, revision: Revision | None = None
    ) -> GenerationResult:
        messages = build_generation_messages(question, chunks, revision)
        response = await self._model.bind(max_tokens=max_tokens).ainvoke(messages)
        prompt_text = "\n\n".join(message_text(m) for m in messages)
        return GenerationResult(answer=message_text(response).strip(), message=response, prompt_text=prompt_text)
