"""Replay cache for the bundled demo questions.

Every entry is a real, previously recorded ``QueryResponse`` (written by ``scripts/build_demo_cache.py``
from live pipeline runs); nothing is hand-written. A replay returns that recorded response unchanged,
including its original latency, tokens and cost, and only adds ``cached=True``, ``cached_from`` and
``served_ms`` so clients can state plainly that it is a replay.

An entry is served only while it is still valid for the current system:
  * same normalised question, pipeline, corpus and request overrides (the cache key),
  * the indexed documents of the corpus are exactly the ones present when it was recorded,
  * the same chat/embedding models and the same resolved pipeline configuration.
Any change (a document uploaded or deleted, a model or parameter changed) makes the entry a miss, so
the question is answered live instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import PipelineConfig
from app.database.repositories import DocumentRepository
from app.schemas.documents import DocumentStatus
from app.schemas.query import QueryResponse

logger = logging.getLogger(__name__)

CACHE_VERSION = 1


def normalise_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip()).lower()


def cache_key(question: str, pipeline: Any, corpus: str, overrides: dict[str, Any] | None) -> str:
    canonical = json.dumps(
        {
            "question": normalise_question(question),
            "pipeline": getattr(pipeline, "value", pipeline),
            "corpus": corpus,
            "overrides": overrides or {},
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_entry(
    directory: Path,
    *,
    question: str,
    pipeline: Any,
    corpus: str,
    overrides: dict[str, Any] | None,
    corpus_fingerprint: list[str],
    response: dict[str, Any],
) -> Path:
    """Store one recorded response (the JSON a live API call returned) under its cache key."""
    key = cache_key(question, pipeline, corpus, overrides)
    directory.mkdir(parents=True, exist_ok=True)
    entry = {
        "version": CACHE_VERSION,
        "key": key,
        "question": question,
        "pipeline": getattr(pipeline, "value", pipeline),
        "corpus": corpus,
        "overrides": overrides or {},
        "corpus_fingerprint": sorted(corpus_fingerprint),
        "recorded_at": response.get("created_at"),
        "response": response,
    }
    path = directory / f"{key}.json"
    path.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


class DemoCache:
    def __init__(
        self,
        directory: Path,
        sessionmaker: async_sessionmaker[AsyncSession],
        *,
        chat_model: str,
        embedding_model: str,
        enabled: bool = True,
    ) -> None:
        self._directory = directory
        self._sessionmaker = sessionmaker
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._enabled = enabled

    async def fingerprint(self, corpus: str) -> list[str]:
        """Sorted content hashes of the ready documents currently indexed in ``corpus``."""
        async with self._sessionmaker() as session:
            rows = await DocumentRepository(session).list(corpus)
        return sorted(r.sha256 for r in rows if r.status == DocumentStatus.READY.value)

    async def lookup(
        self, question: str, pipeline: Any, corpus: str, overrides: dict[str, Any] | None, config: PipelineConfig
    ) -> QueryResponse | None:
        started = perf_counter()
        if not self._enabled:
            return None
        path = self._directory / f"{cache_key(question, pipeline, corpus, overrides)}.json"
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            recorded = entry["response"]
            valid = (
                entry.get("version") == CACHE_VERSION
                and recorded.get("config") == config.model_dump(mode="json")
                and recorded.get("models", {}).get("chat") == self._chat_model
                and recorded.get("models", {}).get("embedding") == self._embedding_model
                and entry.get("corpus_fingerprint") == await self.fingerprint(corpus)
            )
            if not valid:
                logger.info("Demo cache entry for %r is stale; answering live", question)
                return None
            original = QueryResponse.model_validate(recorded)
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("Unreadable demo cache entry %s; answering live", path.name, exc_info=True)
            return None
        return original.model_copy(
            update={
                "query_id": uuid4().hex,
                "created_at": datetime.now(timezone.utc),
                "cached": True,
                "cached_from": original.created_at,
                "served_ms": (perf_counter() - started) * 1000,
            }
        )


def replay_events(response: QueryResponse) -> list[dict[str, Any]]:
    """Progress events rebuilt from a recorded trace, matching what the live graph streams."""
    events: list[dict[str, Any]] = []
    if response.pipeline.value != "proposed":
        return events
    if response.complexity is not None:
        c = response.complexity
        events.append(
            {
                "type": "step",
                "node": "analyze_query",
                "summary": {
                    "label": c.label,
                    "score": c.score,
                    "reasoning_type": c.reasoning_type,
                    "information_needs": c.information_needs,
                },
            }
        )
        events.append({"type": "step", "node": "select_depth", "summary": {"k": response.retrieval.k_requested}})
    events.append(
        {
            "type": "step",
            "node": "retrieve",
            "summary": {"num_chunks": response.retrieval.num_retrieved, "k": response.retrieval.k_requested},
        }
    )
    confidence = response.retrieval.confidence
    if confidence is not None:
        events.append(
            {
                "type": "step",
                "node": "assess_confidence",
                "summary": {"confidence": confidence.score, "sufficient": confidence.sufficient, "coverage": confidence.coverage},
            }
        )
    refinements = {r.attempt: r for r in response.refinements}
    for attempt in response.attempts:
        events.append(
            {"type": "step", "node": "generate", "summary": {"attempt": attempt.attempt, "preview": attempt.answer[:240]}}
        )
        report = attempt.verification
        if report is not None:
            events.append({"type": "step", "node": "extract_claims", "summary": {"num_claims": len(report.claims)}})
            events.append(
                {
                    "type": "step",
                    "node": "verify",
                    "summary": {"support_score": report.support_score, "unsupported": report.num_unsupported, "passed": report.passed},
                }
            )
        step = refinements.get(attempt.attempt + 1)
        if step is not None:
            events.append(
                {
                    "type": "step",
                    "node": "refine",
                    "summary": {"added_chunks": len(step.added_chunk_ids), "targeted_claims": len(step.targeted_claims)},
                }
            )
    events.append({"type": "step", "node": "finalize", "summary": {}})
    return events
