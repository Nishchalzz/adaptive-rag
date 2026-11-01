"""Indexes the bundled demo documents into the default corpus on first start.

Seeding runs once per data directory (a marker file records it), so documents the user deletes are not
re-added on the next start. Without an API key, seeding is skipped and retried on a later start.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.core.config import BACKEND_DIR
from app.core.container import ServiceContainer
from app.core.errors import AppError
from app.schemas.documents import DocumentStatus

logger = logging.getLogger(__name__)

SEED_DIR = BACKEND_DIR / "seed"
SEED_DOCUMENTS = ("institute_overview.md", "research_labs.md", "graduate_regulations.pdf")
MARKER = ".demo_corpus_seeded"


async def seed_demo_corpus(container: ServiceContainer) -> int:
    """Return the number of documents added."""
    settings = container.settings
    marker = settings.data_dir / MARKER
    if not settings.seed_demo_corpus or marker.exists():
        return 0
    if not container.llm_configured:
        logger.info("Demo corpus not seeded: no LLM provider configured")
        return 0
    corpus = settings.default_corpus
    if (await container.documents.list(corpus)).total > 0:
        marker.write_text("existing documents found; demo corpus not added\n", encoding="utf-8")
        return 0

    added = 0
    for name in SEED_DOCUMENTS:
        path = SEED_DIR / name
        try:
            record = await container.documents.register_upload(name, path.read_bytes(), corpus)
            record = await container.documents.ingest(record.id)
        except (AppError, OSError):
            logger.exception("Could not seed demo document %s", name)
            return added
        if record.status != DocumentStatus.READY:
            logger.error("Demo document %s failed to ingest: %s", name, record.error)
            return added
        added += 1
    marker.write_text(f"seeded {added} documents into corpus '{corpus}'\n", encoding="utf-8")
    logger.info("Seeded %d demo documents into corpus '%s'", added, corpus)
    return added


def demo_questions() -> dict[str, Any]:
    path: Path = SEED_DIR / "demo_questions.json"
    return json.loads(path.read_text(encoding="utf-8"))
