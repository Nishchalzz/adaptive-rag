"""Read-only access to experiment results written by the evaluation CLI."""

import json
import re
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, PlainTextResponse

from app.api.deps import get_container
from app.core.container import ServiceContainer
from app.core.errors import AppError

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

Container = Annotated[ServiceContainer, Depends(get_container)]
_SAFE_NAME = re.compile(r"^[\w.-]+$")


class ExperimentNotFoundError(AppError):
    status_code = 404


def _suite_dir(container: ServiceContainer, suite: str) -> Path:
    if not _SAFE_NAME.match(suite):
        raise ExperimentNotFoundError(f"Experiment suite '{suite}' not found")
    path = container.settings.experiments_dir / "results" / suite
    if not (path / "summary.json").is_file():
        raise ExperimentNotFoundError(f"Experiment suite '{suite}' has no results yet")
    return path


@router.get("")
async def list_experiments(container: Container) -> list[dict[str, Any]]:
    root = container.settings.experiments_dir / "results"
    suites = []
    for summary_path in sorted(root.glob("*/summary.json")) if root.is_dir() else []:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        suites.append(
            {
                "name": summary_path.parent.name,
                "created_at": summary.get("created_at"),
                "description": summary.get("description"),
                "num_questions": summary.get("dataset", {}).get("num_test_questions"),
                "variants": list(summary.get("variants", {})),
            }
        )
    return suites


@router.get("/{suite}")
async def get_experiment(suite: str, container: Container) -> dict[str, Any]:
    return json.loads((_suite_dir(container, suite) / "summary.json").read_text(encoding="utf-8"))


@router.get("/{suite}/report", response_class=PlainTextResponse)
async def get_report(suite: str, container: Container) -> str:
    path = _suite_dir(container, suite) / "report.md"
    if not path.is_file():
        raise ExperimentNotFoundError("Report not generated yet")
    return path.read_text(encoding="utf-8")


@router.get("/{suite}/figures/{name}")
async def get_figure(suite: str, name: str, container: Container) -> FileResponse:
    if not _SAFE_NAME.match(name) or not name.endswith(".png"):
        raise ExperimentNotFoundError("Figure not found")
    path = _suite_dir(container, suite) / "figures" / name
    if not path.is_file():
        raise ExperimentNotFoundError("Figure not found")
    return FileResponse(path, media_type="image/png")
