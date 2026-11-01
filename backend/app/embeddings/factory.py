from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings
from app.core.errors import LLMNotConfiguredError
from app.core.llm import MISSING_KEY_MESSAGE


def create_embeddings(settings: Settings) -> Embeddings:
    if not settings.llm_configured:
        raise LLMNotConfiguredError(MISSING_KEY_MESSAGE)
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )
