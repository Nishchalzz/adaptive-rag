"""Build the MuSiQue benchmark files.  Run from backend/:  python -m scripts.prepare_dataset"""

import argparse
import json
from pathlib import Path

from app.core.config import PROJECT_ROOT
from app.evaluation.dataset import build_dataset

DATASETS = PROJECT_ROOT / "experiments" / "datasets"


def prepare(seed: int = 13, n_test: int = 12, n_dev: int = 5) -> Path:
    dataset, corpus = build_dataset(DATASETS / "raw" / "musique_full_v1.0_dev.jsonl", seed=seed, n_test=n_test, n_dev=n_dev)
    (DATASETS / "musique_eval.json").write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    (DATASETS / "musique_eval_corpus.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in corpus) + "\n", encoding="utf-8")
    print(f"questions={len(dataset['questions'])} counts={dataset['counts']} corpus={len(corpus)} paragraphs")
    return DATASETS / "musique_eval.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-test", type=int, default=12, help="test questions per complexity group")
    parser.add_argument("--n-dev", type=int, default=5, help="dev questions per complexity group")
    args = parser.parse_args()
    prepare(args.seed, args.n_test, args.n_dev)
