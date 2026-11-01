"""Paired statistics for comparing two systems on the same questions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

N_BOOT = 10_000


def bootstrap_ci(values: Sequence[float], *, seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(N_BOOT, arr.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi), "n": int(arr.size)}


def paired_values(a: Mapping[str, float | None], b: Mapping[str, float | None]) -> tuple[list[str], np.ndarray, np.ndarray]:
    keys = sorted(k for k in a if k in b and a[k] is not None and b[k] is not None)
    return keys, np.array([a[k] for k in keys], dtype=float), np.array([b[k] for k in keys], dtype=float)


def paired_comparison(a: Mapping[str, float | None], b: Mapping[str, float | None], *, binary: bool = False, seed: int = 0) -> dict[str, Any]:
    """Compare system ``a`` against reference ``b`` (difference = a - b) on shared questions.

    Reports mean difference with a paired-bootstrap 95% CI and a Wilcoxon signed-rank p-value; for binary
    metrics also an exact McNemar test.
    """
    keys, x, y = paired_values(a, b)
    if not keys:
        return {"n": 0}
    diff = x - y
    result: dict[str, Any] = {"n": len(keys), "mean_a": float(x.mean()), "mean_b": float(y.mean())}
    result["diff"] = bootstrap_ci(diff, seed=seed)
    if np.any(diff != 0):
        try:
            result["wilcoxon_p"] = float(stats.wilcoxon(x, y).pvalue)
        except ValueError:
            result["wilcoxon_p"] = None
    else:
        result["wilcoxon_p"] = 1.0
    if binary:
        only_a, only_b = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
        result["only_a"], result["only_b"] = only_a, only_b
        result["mcnemar_p"] = float(stats.binomtest(only_a, only_a + only_b, 0.5).pvalue) if only_a + only_b else 1.0
    return result
