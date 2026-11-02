"""Run the benchmark and write results to experiments/results/<suite>/.

  python -m scripts.run_experiments --suite pilot --per-group 1      # cost pilot (3 questions)
  python -m scripts.run_experiments --suite main                     # full run (cached answers are reused)

Requires OPENAI_API_KEY. LLM answers and judge verdicts are cached under experiments/cache/, keyed by the full
configuration, so reruns only pay for what changed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
from collections import Counter

from app.core.config import PROJECT_ROOT, PipelineConfig, Settings
from app.core.container import build_pipelines
from app.core.llm import create_chat_model
from app.core.logging import configure_logging
from app.embeddings.factory import create_embeddings
from app.evaluation.benchmark import (
    K_SWEEP, MAIN_VARIANTS, RETRY_SWEEP, Benchmark, classifier_report, git_commit, now_iso, paired_against, summarise_variant,
)
from app.evaluation.dataset import CORPUS_NAME, GROUPS, index_corpus, load_dataset
from app.evaluation.judges import AnswerJudge
from app.evaluation.metrics import EvidenceScorer
from app.evaluation.report import write_report
from app.vectorstore.store import ChromaVectorStore
from app.verification.nli import HFEntailmentModel, NLIRegistry
from scripts.prepare_dataset import DATASETS, prepare

LIMITATIONS = [
    "Small sample (test questions per group are listed above): confidence intervals are wide; treat differences whose CI includes 0 as unresolved, not as ties.",
    "Correctness is judged by an LLM against the MuSiQue gold answer and evidence support by an NLI model. Both are automatic and imperfect; no claims were labelled by humans, so judge/NLI error rates are unmeasured here.",
    "The pipeline's chat model and the judge are both OpenAI models and may share blind spots; the evaluator NLI differs from the verifier NLI (different checkpoint, size and threshold) but both are SNLI/MNLI-trained cross-encoders and share known weaknesses with numbers, dates and multi-hop entailment.",
    "The verifier used in these experiments is cross-encoder/nli-roberta-base, not the application default (DeBERTa-v3 base, which measured 1-3 s per pair on this CPU and made the suite impractical). Results describe this configuration; a stronger verifier may behave differently.",
    "The evaluator NLI (MiniLM2, threshold 0.5) looks strict: spot checks of the per-question records show answers the LLM judge marks correct and that cite the right passage flagged as unsupported (long sentences with extra detail, or facts spread over passages). Absolute claim-support and citation values are therefore probably underestimates and were not validated against human labels; use them only for relative comparison between variants, and treat them as unverified until a human-labelled sample exists.",
    "Claims for evidence metrics are sentences of the answer, not atomic facts, so a sentence containing one unsupported fact among supported ones counts as supported if the NLI accepts it.",
    "'Simple' questions are single hops derived from MuSiQue decompositions with earlier answers substituted; 'moderate' and 'complex' are the original 2-hop and 3-4-hop questions. Complexity is therefore proxied by hop count, and the corpus (about 20 paragraphs per source question) is small, so retrieval is easier than on a large open corpus.",
    "MuSiQue gold evidence may be incomplete: a distractor paragraph can also contain the answer, which lowers measured precision.",
    "Latencies were measured while several queries ran concurrently on one CPU machine and cached responses keep the latency of their first run; use them for relative comparison only.",
    "Cost uses the configured price table (USD per million tokens); embedding tokens are estimated with tiktoken. Verify prices before quoting costs.",
    "Unanswerable questions are not part of this run, so the abstention behaviour of the verified pipeline is not evaluated here.",
]


async def main(args: argparse.Namespace) -> int:
    configure_logging("INFO")
    settings = Settings()
    if not settings.llm_configured:
        print("OPENAI_API_KEY is not configured (see .env.example)", file=sys.stderr)
        return 2
    dataset_path = DATASETS / "musique_eval.json"
    if not dataset_path.is_file():
        prepare()
    dataset = load_dataset(dataset_path)
    cache_dir = settings.experiments_dir / "cache"
    out_dir = settings.experiments_dir / "results" / args.suite
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    store = ChromaVectorStore(cache_dir / "chroma")
    embeddings = create_embeddings(settings)
    added = await index_corpus(DATASETS / "musique_eval_corpus.jsonl", store, embeddings)
    print(f"corpus '{CORPUS_NAME}': {store.count(CORPUS_NAME)} paragraphs indexed ({added} newly)")

    # The application default verifier (DeBERTa-v3) measured 1-3 s per pair on this CPU with the installed
    # transformers/torch, which made the suite impractical, so experiments use a faster checkpoint for the
    # verifier. The evaluator uses a different checkpoint again (see docs/experimental_methodology.md).
    config = PipelineConfig().with_overrides({"verification": {"nli_model": args.verifier_nli}})
    nli = NLIRegistry(device=settings.nli_device)
    nli.get(config.verification.nli_model).load()
    evaluator_nli = HFEntailmentModel(args.evaluator_nli, device=settings.nli_device)
    evaluator_nli.load()
    chat = create_chat_model(settings)
    pipelines = build_pipelines(settings=settings, store=store, chat_model=chat, embeddings=embeddings, nli=nli)
    judge = AnswerJudge(create_chat_model(settings, settings.judge_model), settings.judge_model, cache_dir / "judge")
    bench = Benchmark(settings=settings, config=config, pipelines=pipelines, judge=judge, scorer=EvidenceScorer(evaluator_nli),
                      corpus=CORPUS_NAME, cache_dir=cache_dir, concurrency=args.concurrency)

    test = [q for q in dataset["questions"] if q["split"] == "test"]
    items = [q for g in GROUPS for q in [x for x in test if x["group"] == g][: args.per_group]]
    sweep_items = [q for g in GROUPS for q in [x for x in items if x["group"] == g][: args.sweep_per_group]]

    records = await bench.run(MAIN_VARIANTS, items, out_dir)
    specs = list(MAIN_VARIANTS)
    if not args.no_sweeps:
        records.update(await bench.run(K_SWEEP, items, out_dir))
        records.update(await bench.run(RETRY_SWEEP, sweep_items, out_dir))
        specs += K_SWEEP + RETRY_SWEEP

    variants = {s.name: {"label": s.label, "pipeline": s.pipeline.value, "overrides": s.overrides, "sweep": s.sweep,
                         **summarise_variant(records[s.name])} for s in specs if s.name in records}
    pipeline_cost = sum(r["cost_usd"] or 0 for rows in records.values() for r in rows if r.get("cost_usd") is not None)
    summary = {
        "suite": args.suite,
        "created_at": now_iso(),
        "description": "Baseline vs adaptive retrieval vs evidence verification on MuSiQue (automatic evaluation, no human labels).",
        "git_commit": git_commit(PROJECT_ROOT),
        "dataset": {
            "name": dataset["name"], "seed": dataset["seed"], "source": dataset["source"], "corpus_size": dataset["corpus_size"],
            "num_test_questions": len(items), "test_counts": dict(Counter(q["group"] for q in items)),
            "split_counts": dataset["counts"], "hops": dataset["hops"],
        },
        "sweep_questions": len(sweep_items),
        "models": {"chat": settings.chat_model, "embedding": settings.embedding_model, "verifier_nli": config.verification.nli_model,
                   "judge": settings.judge_model, "evaluator_nli": args.evaluator_nli},
        "config": config.model_dump(mode="json"),
        "llm": {"temperature": settings.llm_temperature, "seed": settings.llm_seed},
        "variants": variants,
        "paired": paired_against(records, "B", [s.name for s in specs]),
        "paired_k_baseline": paired_against(records, "K5", [s.name for s in K_SWEEP]) if not args.no_sweeps else {},
        "classifier": classifier_report(records["A+V"] if "A+V" in records else []),
        "classifier_adaptive_only": classifier_report(records["A"]),
        "total_errors": sum(v["n_errors"] for v in variants.values()),
        "costs": {"pipeline_usd": pipeline_cost, "judge_usd_this_run": bench.judge_usage.summary().cost_usd or 0.0},
        "limitations": LIMITATIONS,
        "human_verified": False,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_report(summary, out_dir)
    print(f"wrote {out_dir}  pipeline_cost=${pipeline_cost:.4f} judge_cost_this_run=${summary['costs']['judge_usd_this_run']:.4f} errors={summary['total_errors']}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="main")
    parser.add_argument("--per-group", type=int, default=12, help="test questions per complexity group")
    parser.add_argument("--sweep-per-group", type=int, default=6, help="questions per group for the retry sweep")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--no-sweeps", action="store_true")
    parser.add_argument("--verifier-nli", default="cross-encoder/nli-roberta-base", help="NLI checkpoint used by the pipeline's verifier")
    parser.add_argument("--evaluator-nli", default="cross-encoder/nli-MiniLM2-L6-H768", help="independent NLI checkpoint used for evaluation")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
