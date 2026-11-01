"""Chat model factory. All pipelines get their LLM from here so model settings stay in one place."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.core.errors import LLMNotConfiguredError

MISSING_KEY_MESSAGE = (
    "OPENAI_API_KEY is not configured. Copy .env.example to .env, set the key and restart the backend."
)


def create_chat_model(settings: Settings, model: str | None = None) -> BaseChatModel:
    if not settings.llm_configured:
        raise LLMNotConfiguredError(MISSING_KEY_MESSAGE)
    return ChatOpenAI(
        model=model or settings.chat_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


def message_text(message: BaseMessage) -> str:
    """Plain text of a chat message whose content may be a string or a list of content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)
