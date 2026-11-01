"""Runs experiment variants over the benchmark and aggregates measured results.

Nothing here contains result values: every number in ``summary.json`` / ``report.md`` is computed from
per-question records produced by running the pipelines and the evaluators.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import PipelineConfig, Settings
from app.core.usage import UsageTracker
from app.evaluation.dataset import GROUPS
from app.evaluation.judges import AnswerJudge
from app.evaluation.metrics import EvidenceScorer, answer_text_metrics, efficiency_metrics, retrieval_metrics
from app.evaluation.stats import bootstrap_ci, paired_comparison
from app.schemas.query import PipelineName, QueryResponse
from app.workflows.query_service import Pipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str
    pipeline: PipelineName
    overrides: dict[str, Any] = field(default_factory=dict)
    sweep: str = "main"  # main | k | retries


def _ov(**kw: Any) -> dict[str, Any]:
    return kw


MAIN_VARIANTS = [
    VariantSpec("B", "Baseline (fixed k=5)", PipelineName.BASELINE),
    VariantSpec("A", "Adaptive retrieval only", PipelineName.PROPOSED, _ov(proposed={"verify": False})),
    VariantSpec("A-", "Adaptive depth, no decomposition", PipelineName.PROPOSED, _ov(proposed={"verify": False}, adaptive={"decompose": False})),
    VariantSpec("V", "Verification only (fixed k=5)", PipelineName.PROPOSED, _ov(proposed={"adaptive": False})),
    VariantSpec("A+V", "Adaptive + verification", PipelineName.PROPOSED),
]
K_SWEEP = [VariantSpec(f"K{k}", f"Baseline k={k}", PipelineName.BASELINE, _ov(baseline={"top_k": k}), "k") for k in (1, 3, 5, 8, 10)]
RETRY_SWEEP = [
    VariantSpec(f"R{r}", f"A+V, max_retries={r}", PipelineName.PROPOSED, _ov(verification={"max_retries": r}), "retries") for r in (0, 1, 2)
]

METRICS = [
    "p_at_k", "r_at_k", *[f"{p}_at_{c}" for c in (1, 3, 5, 10) for p in "pr"], "mrr", "ndcg_at_10",
    "em", "f1", "contains", "correct", "abstained", "addresses_question",
    "num_claims", "claim_support_rate", "unsupported_rate", "citation_precision", "citation_recall",
    "num_chunks", "k_selected", "retrieval_latency_ms", "total_latency_ms", "input_tokens", "output_tokens",
    "total_tokens", "cost_usd", "num_attempts", "verifier_support_score", "verifier_passed", "uncertainty_flagged",
]
BINARY = {"correct", "em", "contains"}


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


class Benchmark:
    def __init__(
        self,
        *,
        settings: Settings,
        config: PipelineConfig,
        pipelines: dict[PipelineName, Pipeline],
        judge: AnswerJudge,
        scorer: EvidenceScorer,
        corpus: str,
        cache_dir: Path,
        concurrency: int = 3,
    ) -> None:
        self._settings, self._config = settings, config
        self._pipelines, self._judge, self._scorer = pipelines, judge, scorer
        self._corpus = corpus
        self._cache = cache_dir / "responses"
        self._cache.mkdir(parents=True, exist_ok=True)
        self._sem = asyncio.Semaphore(concurrency)
        self.judge_usage = UsageTracker(config.pricing)

    # ------------------------------------------------------------------ running
    def _key(self, spec: VariantSpec, question: str) -> tuple[PipelineConfig, Path]:
        config = self._config.with_overrides(spec.overrides)
        payload = json.dumps(
            [spec.pipeline.value, question, self._corpus, config.model_dump(mode="json"),
             self._settings.chat_model, self._settings.embedding_model, self._settings.llm_temperature, self._settings.llm_seed],
            sort_keys=True,
        )
        return config, self._cache / f"{hashlib.sha256(payload.encode()).hexdigest()}.json"

    async def _answer(self, spec: VariantSpec, item: dict[str, Any]) -> tuple[QueryResponse | None, str | None, bool]:
        config, path = self._key(spec, item["question"])
        if path.is_file():
            return QueryResponse.model_validate_json(path.read_text(encoding="utf-8")), None, True
        async with self._sem:
            try:
                response = await self._pipelines[spec.pipeline].run(item["question"], self._corpus, config)
            except Exception as exc:  # noqa: BLE001 - one failed question must not abort the whole suite
                logger.exception("%s failed on %s", spec.name, item["qid"])
                return None, f"{type(exc).__name__}: {exc}", False
        path.write_text(response.model_dump_json(), encoding="utf-8")
        return response, None, False

    async def _evaluate(self, spec: VariantSpec, item: dict[str, Any]) -> dict[str, Any]:
        response, error, cached = await self._answer(spec, item)
        record: dict[str, Any] = {"variant": spec.name, "qid": item["qid"], "group": item["group"], "question": item["question"],
                                  "gold": item["answer"], "error": error, "cached": cached}
        if response is None:
            return record
        retrieved = [c.chunk_id for c in response.retrieval.chunks]
        record["retrieved_chunk_ids"] = retrieved
        record["gold_chunk_ids"] = item["gold_chunk_ids"]
        record["answer"] = response.answer
        record.update(retrieval_metrics(retrieved, set(item["gold_chunk_ids"])))
        record.update(answer_text_metrics(response.answer, item["answer"], item["aliases"]))
        record.update(efficiency_metrics(response))
        try:
            async with self._sem:
                verdict = await self._judge.judge(item["question"], item["answer"], item["aliases"], response.answer, self.judge_usage)
            record.update(correct=float(verdict.verdict == "correct"), abstained=float(verdict.verdict == "abstained"),
                          addresses_question=float(verdict.addresses_question), judge_rationale=verdict.rationale)
        except Exception as exc:  # noqa: BLE001
            logger.exception("judge failed for %s/%s", spec.name, item["qid"])
            record["error"] = f"judge: {type(exc).__name__}: {exc}"
        v = response.verification
        record["verifier_support_score"] = v.support_score if v else None
        record["verifier_passed"] = float(v.passed) if v else None
        record["uncertainty_flagged"] = float(response.uncertainty.flagged) if response.uncertainty else None
        if response.complexity:
            record["complexity_label"], record["complexity_score"] = response.complexity.label, response.complexity.score
        return record

    async def run(self, specs: list[VariantSpec], items: list[dict[str, Any]], out_dir: Path) -> dict[str, list[dict[str, Any]]]:
        """Answer + judge concurrently, then score evidence with the (CPU, single-threaded) NLI model."""
        out_dir.mkdir(parents=True, exist_ok=True)
        records: dict[str, list[dict[str, Any]]] = {}
        for spec in specs:
            rows = await asyncio.gather(*(self._evaluate(spec, it) for it in items))
            responses = {}
            for row, it in zip(rows, items, strict=True):
                if row.get("answer") is None:
                    continue
                _, path = self._key(spec, it["question"])
                response = QueryResponse.model_validate_json(path.read_text(encoding="utf-8"))
                row.update(await asyncio.to_thread(self._scorer.score, response))
                responses[it["qid"]] = response
            records[spec.name] = list(rows)
            with (out_dir / "per_question.jsonl").open("a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, default=str) + "\n")
            failed = sum(1 for r in rows if r["error"])
            logger.info("variant %s done: %d questions, %d with errors", spec.name, len(rows), failed)
        return records


# ---------------------------------------------------------------------- aggregation
def aggregate(rows: list[dict[str, Any]], metrics: list[str] = METRICS) -> dict[str, dict[str, float]]:
    out = {}
    for metric in metrics:
        values = [float(r[metric]) for r in rows if r.get(metric) is not None]
        if values:
            out[metric] = bootstrap_ci(values)
    return out


def summarise_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r["error"]]
    return {
        "n": len(rows),
        "n_errors": len(rows) - len(ok),
        "overall": aggregate(ok),
        "by_group": {g: aggregate([r for r in ok if r["group"] == g]) for g in GROUPS},
    }


def per_question(rows: list[dict[str, Any]], metric: str) -> dict[str, float | None]:
    return {r["qid"]: r.get(metric) for r in rows if not r["error"]}


PAIRED_METRICS = ["correct", "f1", "r_at_k", "p_at_k", "mrr", "claim_support_rate", "unsupported_rate", "citation_precision",
                  "citation_recall", "num_chunks", "total_latency_ms", "total_tokens", "cost_usd"]


def paired_against(records: dict[str, list[dict[str, Any]]], reference: str, names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in names:
        if name == reference or name not in records:
            continue
        out[name] = {
            m: paired_comparison(per_question(records[name], m), per_question(records[reference], m), binary=m in BINARY)
            for m in PAIRED_METRICS
        }
    return out


def classifier_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labelled = [r for r in rows if r.get("complexity_label")]
    if not labelled:
        return {}
    confusion = {g: {p: sum(1 for r in labelled if r["group"] == g and r["complexity_label"] == p) for p in GROUPS} for g in GROUPS}
    by_group: dict[str, Any] = defaultdict(dict)
    for g in GROUPS:
        members = [r for r in labelled if r["group"] == g]
        if members:
            by_group[g] = {"n": len(members), "mean_score": sum(r["complexity_score"] for r in members) / len(members),
                           "mean_k": sum(r["k_selected"] for r in members) / len(members),
                           "mean_chunks": sum(r["num_chunks"] for r in members) / len(members)}
    return {"accuracy": sum(1 for r in labelled if r["group"] == r["complexity_label"]) / len(labelled), "n": len(labelled),
            "confusion_true_by_predicted": confusion, "by_group": dict(by_group),
            "note": "Accuracy against hop-count groups, a proxy for complexity, not a human complexity label."}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
