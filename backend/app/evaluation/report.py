"""Markdown report and figures generated from ``summary.json`` (values are read, never typed in)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app.evaluation.dataset import GROUPS  # noqa: E402

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def _m(summary: dict[str, Any], variant: str, metric: str, group: str | None = None) -> dict[str, float] | None:
    v = summary["variants"].get(variant)
    if not v:
        return None
    block = v["overall"] if group is None else v["by_group"].get(group, {})
    return block.get(metric)


def _fmt(stat: dict[str, float] | None, scale: float = 1.0, digits: int = 3, ci: bool = True) -> str:
    if not stat or math.isnan(stat["mean"]):
        return "n/a"
    text = f"{stat['mean'] * scale:.{digits}f}"
    return f"{text} [{stat['lo'] * scale:.{digits}f}, {stat['hi'] * scale:.{digits}f}]" if ci else text


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _variants(summary: dict[str, Any], sweep: str) -> list[str]:
    return [n for n, v in summary["variants"].items() if v["sweep"] == sweep]


def _paired_sentence(name: str, metric: str, comp: dict[str, Any], label: str, scale: float = 1.0, unit: str = "") -> str | None:
    if not comp or comp.get("n", 0) == 0:
        return None
    d = comp["diff"]
    verdict = "the 95% CI excludes 0" if d["lo"] > 0 or d["hi"] < 0 else "the 95% CI includes 0, so no reliable difference at this sample size"
    return (f"- **{name} vs B, {label}**: mean difference {d['mean'] * scale:+.3f}{unit} "
            f"(95% CI {d['lo'] * scale:+.3f} to {d['hi'] * scale:+.3f}, n={comp['n']}); {verdict}.")


def write_report(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    main, ks, rs = _variants(summary, "main"), _variants(summary, "k"), _variants(summary, "retries")
    labels = {n: v["label"] for n, v in summary["variants"].items()}
    ds = summary["dataset"]
    test_counts = ", ".join(f"{g}: {ds['test_counts'].get(g, 0)}" for g in GROUPS)
    parts: list[str] = [
        f"# Experiment report: {summary['suite']}",
        f"Generated {summary['created_at']} · git commit `{summary.get('git_commit') or 'none (repository has no commits)'}`",
        "> All numbers below are measured values computed from `per_question.jsonl`. Evaluation is **automatic** "
        "(LLM judge + NLI model). Only the retrieval gold labels are human-annotated (MuSiQue). No claims in this "
        "project have been labelled by hand. Cells show mean [95% bootstrap CI].",
        "## Setup",
        f"- Dataset: {ds['name']} (seed {ds['seed']}); test questions: {ds['num_test_questions']} "
        f"({test_counts}); corpus {ds['corpus_size']} paragraphs.",
        f"- Pipeline models: chat `{summary['models']['chat']}`, embedding `{summary['models']['embedding']}`, "
        f"verifier NLI `{summary['models']['verifier_nli']}`. Evaluation: judge `{summary['models']['judge']}`, "
        f"evaluator NLI `{summary['models']['evaluator_nli']}` (independent of the pipeline).",
        f"- Questions per variant in k/retry sweeps: {summary['sweep_questions']}.",
        f"- Errors: {summary['total_errors']} question runs failed and are excluded from their variant's means.",
    ]

    rows = [[f"**{n}** {labels[n]}", _fmt(_m(summary, n, "correct")), _fmt(_m(summary, n, "f1")), _fmt(_m(summary, n, "r_at_k")),
             _fmt(_m(summary, n, "p_at_k")), _fmt(_m(summary, n, "mrr")), _fmt(_m(summary, n, "ndcg_at_10"))] for n in main]
    parts += ["## Ablation: answer quality and retrieval (Experiments 1 and 2)",
              _table(["Variant", "Correct (judge)", "Token F1", "Recall@k (own k)", "Precision@k (own k)", "MRR", "nDCG@10"], rows)]

    rows = [[f"**{n}** {labels[n]}", _fmt(_m(summary, n, "claim_support_rate")), _fmt(_m(summary, n, "unsupported_rate")),
             _fmt(_m(summary, n, "citation_precision")), _fmt(_m(summary, n, "citation_recall")),
             _fmt(_m(summary, n, "abstained"), digits=2, ci=False)] for n in main]
    parts += ["## Evidence reliability (independent NLI evaluator)",
              "Claims = sentences of the final answer. Support = entailed by retrieved context at probability ≥ 0.5. Faithfulness "
              "is reported as the claim support rate. Answers with no claims (abstentions) are excluded from these means.",
              _table(["Variant", "Claim support rate", "Unsupported claim rate", "Citation precision", "Citation recall", "Abstained"], rows)]

    rows = [[f"**{n}** {labels[n]}", _fmt(_m(summary, n, "num_chunks"), digits=2), _fmt(_m(summary, n, "retrieval_latency_ms"), 1e-3, 2),
             _fmt(_m(summary, n, "total_latency_ms"), 1e-3, 1), _fmt(_m(summary, n, "total_tokens"), digits=0),
             _fmt(_m(summary, n, "cost_usd"), 1000, 3)] for n in main]
    parts += ["## Efficiency",
              "Latencies were measured with several queries in flight on a CPU machine; compare variants with each other, not in absolute terms. "
              "Cost is the estimated pipeline API cost (chat + embeddings) in USD per 1000 queries, using the configured price table.",
              _table(["Variant", "Chunks retrieved", "Retrieval latency (s)", "Total latency (s)", "Tokens / query", "USD / 1000 queries"], rows)]

    rows = []
    for n in main:
        rows.append([f"**{n}**"] + [f"{_fmt(_m(summary, n, 'correct', g), digits=2, ci=False)} / {_fmt(_m(summary, n, 'r_at_k', g), digits=2, ci=False)} / "
                                    f"{_fmt(_m(summary, n, 'num_chunks', g), digits=1, ci=False)}" for g in GROUPS])
    parts += ["## By complexity group (correct / recall@k / chunks retrieved)", _table(["Variant", *GROUPS], rows)]

    paired = summary.get("paired", {})
    sentences = []
    for n in main:
        if n == "B":
            continue
        for metric, label, scale, unit in [("correct", "answer correct (judge)", 1, ""), ("r_at_k", "recall@k", 1, ""),
                                           ("unsupported_rate", "unsupported claim rate", 1, ""), ("citation_precision", "citation precision", 1, ""),
                                           ("num_chunks", "chunks retrieved", 1, ""), ("total_latency_ms", "total latency", 1e-3, " s"),
                                           ("cost_usd", "cost per query", 1000, " USD/1000")]:
            s = _paired_sentence(n, metric, paired.get(n, {}).get(metric, {}), label, scale, unit)
            if s:
                sentences.append(s)
    parts += ["## Paired comparisons against the baseline (B)",
              "Same questions, difference = variant − baseline, paired bootstrap (10,000 resamples). McNemar/Wilcoxon p-values are in `summary.json`.",
              "\n".join(sentences) or "_No paired data._"]

    if ks:
        rows = [[labels[k]] + [f"{_fmt(_m(summary, k, 'correct', g), digits=2, ci=False)} / {_fmt(_m(summary, k, 'r_at_k', g), digits=2, ci=False)} / "
                               f"{_fmt(_m(summary, k, 'p_at_k', g), digits=2, ci=False)}" for g in GROUPS] +
                [f"{_fmt(_m(summary, k, 'correct'), digits=2, ci=False)} / {_fmt(_m(summary, k, 'r_at_k'), digits=2, ci=False)}"] for k in ks]
        parts += ["## Retrieval depth by complexity (Experiment 3: fixed-k baseline sweep)",
                  "Cells: correct / recall@k / precision@k. Shows which fixed depth works best for each group; compare with the depths the adaptive variants choose.",
                  _table(["Fixed depth", *GROUPS, "Overall (correct / recall@k)"], rows)]

    if rs:
        rows = [[labels[r], _fmt(_m(summary, r, "correct")), _fmt(_m(summary, r, "claim_support_rate")), _fmt(_m(summary, r, "unsupported_rate")),
                 _fmt(_m(summary, r, "num_attempts"), digits=2, ci=False), _fmt(_m(summary, r, "total_latency_ms"), 1e-3, 1),
                 _fmt(_m(summary, r, "cost_usd"), 1000, 3)] for r in rs]
        parts += ["## Retries (Experiment 4)",
                  _table(["Variant", "Correct", "Claim support", "Unsupported rate", "Attempts", "Latency (s)", "USD / 1000"], rows)]

    rows = [[f"**{n}** {labels[n]}", _fmt(_m(summary, n, "unsupported_rate"), ci=False), _fmt(_m(summary, n, "correct"), ci=False),
             _fmt(_m(summary, n, "total_latency_ms"), 1e-3, 1, ci=False), _fmt(_m(summary, n, "cost_usd"), 1000, 3, ci=False)] for n in main + rs]
    parts += ["## Reliability versus cost (Experiment 5)", _table(["Variant", "Unsupported rate", "Correct", "Latency (s)", "USD / 1000 queries"], rows)]

    cls = summary.get("classifier", {})
    if cls:
        parts += ["## Complexity classifier",
                  f"Agreement of predicted label with hop-count group: {cls['accuracy']:.3f} over {cls['n']} queries. {cls['note']}",
                  _table(["True group \\ predicted", *GROUPS], [[g] + [str(cls["confusion_true_by_predicted"][g][p]) for p in GROUPS] for g in GROUPS]),
                  _table(["Group", "n", "Mean score", "Mean k", "Mean chunks"],
                         [[g, str(c["n"]), f"{c['mean_score']:.3f}", f"{c['mean_k']:.2f}", f"{c['mean_chunks']:.2f}"] for g, c in cls["by_group"].items()])]

    parts += ["## Costs of running this suite",
              f"- Pipeline cost summed over all recorded query runs (including answers served from the cache, priced when first generated): "
              f"USD {summary['costs']['pipeline_usd']:.4f}.",
              f"- Judge calls newly made in this run: USD {summary['costs']['judge_usd_this_run']:.4f}.",
              "## Limitations and sources of bias",
              "\n".join(f"- {x}" for x in summary["limitations"])]
    (out_dir / "report.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    _figures(summary, out_dir / "figures", main, ks, rs, labels)


def _bar(ax: Any, summary: dict[str, Any], names: list[str], metric: str, title: str, scale: float = 1.0) -> None:
    stats = [_m(summary, n, metric) for n in names]
    means = [s["mean"] * scale if s else 0 for s in stats]
    err = [[(s["mean"] - s["lo"]) * scale if s else 0 for s in stats], [(s["hi"] - s["mean"]) * scale if s else 0 for s in stats]]
    ax.bar(names, means, yerr=err, color=PALETTE[: len(names)], capsize=3)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def _figures(summary: dict[str, Any], fig_dir: Path, main: list[str], ks: list[str], rs: list[str], labels: dict[str, str]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (metric, title) in zip(axes.flat, [("correct", "Answer correct (judge)"), ("r_at_k", "Recall@k (own k)"), ("p_at_k", "Precision@k (own k)"),
                                                ("claim_support_rate", "Claim support rate (NLI)"), ("citation_precision", "Citation precision"),
                                                ("citation_recall", "Citation recall")], strict=True):
        _bar(ax, summary, main, metric, title)
        ax.set_ylim(0, 1.05)
    fig.suptitle("Quality, retrieval and evidence reliability (mean, 95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(fig_dir / "quality.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (metric, title, scale) in zip(axes, [("num_chunks", "Chunks retrieved", 1), ("total_latency_ms", "Total latency (s)", 1e-3),
                                                  ("total_tokens", "Tokens per query", 1), ("cost_usd", "USD per 1000 queries", 1000)], strict=True):
        _bar(ax, summary, main, metric, title, scale)
    fig.tight_layout()
    fig.savefig(fig_dir / "efficiency.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(len(main), 1)
    for i, n in enumerate(main):
        vals = [(_m(summary, n, "num_chunks", g) or {"mean": 0})["mean"] for g in GROUPS]
        ax.bar([x + i * width for x in range(len(GROUPS))], vals, width, label=n, color=PALETTE[i])
    ax.set_xticks([x + 0.4 - width / 2 for x in range(len(GROUPS))], GROUPS)
    ax.set_ylabel("Chunks retrieved")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "chunks_by_group.png", dpi=140)
    plt.close(fig)

    if ks:
        depth = [int(k[1:]) for k in ks]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric, title in zip(axes, ["correct", "r_at_k"], ["Answer correct (judge)", "Recall@k"], strict=True):
            for i, g in enumerate(GROUPS):
                ax.plot(depth, [(_m(summary, k, metric, g) or {"mean": float("nan")})["mean"] for k in ks], marker="o", label=g, color=PALETTE[i])
            ax.set_xlabel("fixed top-k")
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / "k_sweep.png", dpi=140)
        plt.close(fig)

    if rs:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax, (metric, title, scale) in zip(axes, [("claim_support_rate", "Claim support rate", 1), ("correct", "Answer correct", 1),
                                                      ("total_latency_ms", "Latency (s)", 1e-3)], strict=True):
            _bar(ax, summary, rs, metric, title, scale)
        fig.suptitle("Effect of verification retries (A+V)")
        fig.tight_layout()
        fig.savefig(fig_dir / "retries.png", dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, n in enumerate(main + rs):
        u, c = _m(summary, n, "unsupported_rate"), _m(summary, n, "cost_usd")
        if u and c:
            ax.scatter(c["mean"] * 1000, u["mean"], s=60, color=PALETTE[i % len(PALETTE)])
            ax.annotate(n, (c["mean"] * 1000, u["mean"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("USD per 1000 queries")
    ax.set_ylabel("Unsupported claim rate")
    ax.set_title("Reliability versus cost", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "tradeoff.png", dpi=140)
    plt.close(fig)
