from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.api.deps import get_container
from app.core.container import ServiceContainer
from app.seed import demo_questions

router = APIRouter(prefix="/api", tags=["system"])

Container = Annotated[ServiceContainer, Depends(get_container)]


@router.get("/health")
async def health(container: Container) -> dict[str, Any]:
    return {"status": "ok", "llm_configured": container.llm_configured, "nli_models": container.nli.status()}


@router.get("/config")
async def config(container: Container) -> dict[str, Any]:
    """Active non-secret configuration, so the UI and experiment logs can show what produced a result."""
    settings = container.settings
    return {
        "models": {
            "chat": settings.chat_model,
            "embedding": settings.embedding_model,
            "nli": container.pipeline_config.verification.nli_model,
        },
        "llm": {"temperature": settings.llm_temperature, "seed": settings.llm_seed},
        "default_corpus": settings.default_corpus,
        "llm_configured": container.llm_configured,
        "pipelines": [p.value for p in container.pipelines],
        "pipeline": container.pipeline_config.model_dump(mode="json"),
    }


@router.get("/demo/questions")
async def get_demo_questions() -> dict[str, Any]:
    return demo_questions()
