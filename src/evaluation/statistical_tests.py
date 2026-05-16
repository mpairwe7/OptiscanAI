"""
Statistical significance tests for model comparison.
McNemar's test, paired bootstrap, Wilcoxon signed-rank.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def mcnemar_test(y_pred_a: np.ndarray, y_pred_b: np.ndarray, y_true: np.ndarray) -> dict:
    """McNemar's test: are two models' errors significantly different?"""
    correct_a = (y_pred_a == y_true).all(axis=1)
    correct_b = (y_pred_b == y_true).all(axis=1)

    # Contingency: A correct & B wrong, A wrong & B correct
    b_only = (~correct_a & correct_b).sum()
    a_only = (correct_a & ~correct_b).sum()

    if b_only + a_only == 0:
        return {"statistic": 0, "p_value": 1.0, "significant": False}

    # McNemar with continuity correction
    stat = (abs(b_only - a_only) - 1) ** 2 / (b_only + a_only)
    p_value = 1 - scipy_stats.chi2.cdf(stat, df=1)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "a_better_count": int(a_only),
        "b_better_count": int(b_only),
    }


def paired_bootstrap_test(
    metric_fn,
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    n_bootstrap: int = 1000,
    threshold: float = 0.5,
    seed: int = 42,
) -> dict:
    """Bootstrap test: is model A significantly better than model B?"""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            score_a = metric_fn(y_true[idx], (y_prob_a[idx] > threshold).astype(float))
            score_b = metric_fn(y_true[idx], (y_prob_b[idx] > threshold).astype(float))
            diffs.append(score_a - score_b)
        except (ValueError, IndexError):
            continue

    if not diffs:
        return {"mean_diff": 0, "p_value": 1.0, "significant": False}

    diffs = np.array(diffs)
    p_value = (diffs <= 0).mean() * 2  # Two-sided

    return {
        "mean_diff": float(diffs.mean()),
        "std_diff": float(diffs.std()),
        "p_value": float(min(p_value, 1.0)),
        "significant": p_value < 0.05,
        "ci_95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
    }


def wilcoxon_test(scores_a: list[float], scores_b: list[float]) -> dict:
    """Wilcoxon signed-rank test for k-fold score comparison."""
    if len(scores_a) < 5:
        return {"statistic": 0, "p_value": 1.0, "significant": False, "note": "need >=5 folds"}
    stat, p_value = scipy_stats.wilcoxon(scores_a, scores_b)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }
