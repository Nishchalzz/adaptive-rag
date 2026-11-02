"""Records the demo cache from real pipeline runs against a LIVE backend.

    python -m scripts.build_demo_cache --base http://127.0.0.1:8000

For every question in seed/demo_questions.json it runs the three request shapes the UI sends (baseline,
proposed, and the baseline-audit used by side-by-side mode) with ``use_cache=false`` and stores each real
response in seed/demo_cache/. The entries are tied to the exact documents indexed at recording time; they
stop being served if any document is added or removed, or if a model or parameter changes.

Re-run it after changing the demo questions, the demo documents, the models or the pipeline config.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from app.core.config import BACKEND_DIR
from app.workflows.demo_cache import save_entry

AUDIT_OVERRIDES: dict[str, Any] = {"proposed": {"adaptive": False, "verify": True}, "verification": {"max_retries": 0}}
SHAPES: list[tuple[str, str, dict[str, Any] | None]] = [
    ("baseline", "baseline", None),
    ("proposed", "proposed", None),
    ("audit", "proposed", AUDIT_OVERRIDES),
]


class CacheBuildError(RuntimeError):
    pass


def corpus_fingerprint(client: httpx.Client, corpus: str) -> list[str]:
    response = client.get("/api/documents", params={"corpus": corpus})
    response.raise_for_status()
    documents = response.json()["documents"]
    not_ready = [d["filename"] for d in documents if d["status"] != "ready"]
    if not_ready:
        raise CacheBuildError(f"Documents not ready: {', '.join(not_ready)}")
    if not documents:
        raise CacheBuildError(f"Corpus '{corpus}' has no documents; nothing to record against")
    return sorted(d["sha256"] for d in documents)


def build(
    client: httpx.Client,
    questions: list[dict[str, str]],
    directory: Path,
    corpus: str = "default",
    workers: int = 2,
) -> list[dict[str, Any]]:
    fingerprint = corpus_fingerprint(client, corpus)

    def record(job: tuple[dict[str, str], tuple[str, str, dict[str, Any] | None]]) -> dict[str, Any]:
        item, (shape, pipeline, overrides) = job
        body: dict[str, Any] = {"question": item["question"], "pipeline": pipeline, "corpus": corpus, "use_cache": False}
        if overrides:
            body["overrides"] = overrides
        response = client.post("/api/query", json=body)
        response.raise_for_status()
        data = response.json()
        if data.get("cached"):
            raise CacheBuildError("Server returned a replay while recording; entries must come from live runs")
        # The cache key uses the request as sent by the UI (no corpus override -> default corpus, no use_cache).
        save_entry(
            directory,
            question=item["question"],
            pipeline=pipeline,
            corpus=data["corpus"],
            overrides=overrides,
            corpus_fingerprint=fingerprint,
            response=data,
        )
        verification = data.get("verification")
        return {
            "category": item.get("category", ""),
            "shape": shape,
            "question": item["question"],
            "k": data["retrieval"]["k_requested"],
            "chunks": data["retrieval"]["num_retrieved"],
            "latency_s": data["total_latency_ms"] / 1000,
            "support": verification["support_score"] if verification else None,
            "unsupported": verification["num_unsupported"] if verification else None,
        }

    jobs = [(q, shape) for q in questions for shape in SHAPES]
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(record, jobs))

    if corpus_fingerprint(client, corpus) != fingerprint:
        raise CacheBuildError("The indexed documents changed while recording; entries are not valid. Re-run.")
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    print(f"{'category':<13}{'shape':<10}{'k':>3}{'chunks':>8}{'latency':>9}{'support':>9}{'unsup':>6}  question")
    for r in rows:
        support = "-" if r["support"] is None else f"{r['support']:.2f}"
        unsupported = "-" if r["unsupported"] is None else str(r["unsupported"])
        print(
            f"{r['category']:<13}{r['shape']:<10}{r['k']:>3}{r['chunks']:>8}{r['latency_s']:>8.1f}s{support:>9}{unsupported:>6}  "
            f"{r['question'][:70]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="URL of the running backend")
    parser.add_argument("--questions", type=Path, default=BACKEND_DIR / "seed" / "demo_questions.json")
    parser.add_argument("--out", type=Path, default=BACKEND_DIR / "seed" / "demo_cache")
    parser.add_argument("--corpus", default="default")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))["questions"]
    with httpx.Client(base_url=args.base, timeout=900) as client:
        try:
            rows = build(client, questions, args.out, args.corpus, args.workers)
        except (CacheBuildError, httpx.HTTPError) as exc:
            sys.exit(f"Demo cache build failed: {exc}")
    print_table(rows)
    print(f"\nWrote {len(rows)} entries to {args.out}")


if __name__ == "__main__":
    main()
