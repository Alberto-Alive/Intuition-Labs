"""Statistical significance tests for comparing systems."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap test for the difference in means.

    Returns:
        mean_diff, ci_lower, ci_upper, p_value.
    """
    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    observed_diff = float(np.mean(a) - np.mean(b))

    n = len(a)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diffs.append(np.mean(a[idx]) - np.mean(b[idx]))

    diffs = np.array(diffs)
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))

    # Two-sided p-value: fraction of bootstrap diffs on the other side of 0
    if observed_diff >= 0:
        p_value = float(2 * np.mean(diffs <= 0))
    else:
        p_value = float(2 * np.mean(diffs >= 0))
    p_value = min(p_value, 1.0)

    return {
        "mean_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
    }


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, float]:
    """Paired t-test over seeds."""
    a = np.array(scores_a)
    b = np.array(scores_b)

    if len(a) < 2:
        return {"t_statistic": 0.0, "p_value": 1.0, "df": 0}

    t_stat, p_val = stats.ttest_rel(a, b)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "df": len(a) - 1,
    }


def wilcoxon_test(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, float]:
    """Wilcoxon signed-rank test (nonparametric)."""
    a = np.array(scores_a)
    b = np.array(scores_b)

    diff = a - b
    if np.all(diff == 0) or len(a) < 5:
        return {"statistic": 0.0, "p_value": 1.0}

    try:
        stat, p_val = stats.wilcoxon(a, b)
        return {"statistic": float(stat), "p_value": float(p_val)}
    except ValueError:
        return {"statistic": 0.0, "p_value": 1.0}


def run_all_significance_tests(
    digit_scores: List[float],
    baseline_scores: List[float],
    metric_name: str = "",
) -> Dict[str, Dict]:
    """Run all three significance tests."""
    return {
        "bootstrap": paired_bootstrap_test(digit_scores, baseline_scores),
        "paired_t": paired_t_test(digit_scores, baseline_scores),
        "wilcoxon": wilcoxon_test(digit_scores, baseline_scores),
        "metric": metric_name,
        "digit_mean": float(np.mean(digit_scores)),
        "digit_std": float(np.std(digit_scores)),
        "baseline_mean": float(np.mean(baseline_scores)),
        "baseline_std": float(np.std(baseline_scores)),
    }


def apply_bonferroni(
    test_results: List[Dict],
    alpha: float = 0.05,
) -> List[Dict]:
    """Apply Bonferroni correction to a list of test results."""
    n_tests = len(test_results)
    corrected_alpha = alpha / max(n_tests, 1)

    for r in test_results:
        for test_name in ["bootstrap", "paired_t", "wilcoxon"]:
            if test_name in r:
                p = r[test_name]["p_value"]
                r[test_name]["corrected_p_value"] = min(p * n_tests, 1.0)
                r[test_name]["significant_corrected"] = p < corrected_alpha
                r[test_name]["significant_uncorrected"] = p < alpha

    return test_results
