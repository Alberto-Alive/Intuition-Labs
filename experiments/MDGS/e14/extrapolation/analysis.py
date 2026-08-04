"""Deliverable 5 logistic-regression analysis for DIGIT Extrapolation E14."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from .experiment import TestExampleLogRecord, collect_test_example_logs_for_seed


PREDICTOR_ORDER = [
    "output_confidence",
    "input_nn_distance",
    "layer1_certainty",
    "layer2_certainty",
    "part_a_marginal_frequency",
    "part_b_marginal_frequency",
]
COHENS_D_METRICS = [
    "output_confidence",
    "input_nn_distance",
    "layer1_certainty",
    "layer2_certainty",
]
CERTAINTY_TERMS = ["layer1_certainty", "layer2_certainty"]


@dataclass(frozen=True)
class RegressionTermResult:
    """Coefficient summary for one regression term."""

    coefficient: float | None
    standard_error: float | None
    p_value: float | None


@dataclass(frozen=True)
class SeedRegressionResult:
    """Per-seed Deliverable 5 output."""

    seed: int
    coefficients: dict[str, RegressionTermResult]
    cohens_d: dict[str, float]


@dataclass(frozen=True)
class AggregateMetricSummary:
    """Mean/std summary across seeds."""

    mean: float | None
    std: float | None


@dataclass(frozen=True)
class AggregateAnalysisResult:
    """Aggregated Deliverable 5 output across seeds."""

    coefficient_summary: dict[str, AggregateMetricSummary]
    cohens_d_summary: dict[str, AggregateMetricSummary]
    certainty_sign_consistency: dict[str, str]


@dataclass(frozen=True)
class Deliverable5Result:
    """Full Deliverable 5 payload."""

    per_seed: list[SeedRegressionResult]
    aggregate: AggregateAnalysisResult
    verdict: str
    verdict_evidence: str


def run_deliverable5(seeds: list[int] | tuple[int, ...]) -> Deliverable5Result:
    """Run the full five-seed logistic-regression analysis."""
    per_seed = [fit_seed_regression(seed) for seed in seeds]
    aggregate = aggregate_results(per_seed)
    verdict, evidence = determine_verdict(per_seed, aggregate)
    return Deliverable5Result(
        per_seed=per_seed,
        aggregate=aggregate,
        verdict=verdict,
        verdict_evidence=evidence,
    )


def fit_seed_regression(seed: int) -> SeedRegressionResult:
    """Fit the specified within-seed standardized logistic regression."""
    records, _ = collect_test_example_logs_for_seed(seed)
    df = pd.DataFrame([record.__dict__ for record in records])

    predictor_std = df[PREDICTOR_ORDER].std(ddof=0)
    active_predictors = [predictor for predictor in PREDICTOR_ORDER if predictor_std[predictor] > 0]
    standardized = (df[active_predictors] - df[active_predictors].mean()) / df[active_predictors].std(ddof=0)
    design_matrix = sm.add_constant(standardized, has_constant="add")
    response = df["seen_unseen_label"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        result = sm.GLM(response, design_matrix, family=sm.families.Binomial()).fit(maxiter=100)

    coefficients: dict[str, RegressionTermResult] = {
        "const": _term_result(result, "const"),
    }
    for predictor in PREDICTOR_ORDER:
        if predictor in active_predictors:
            coefficients[predictor] = _term_result(result, predictor)
        else:
            coefficients[predictor] = RegressionTermResult(
                coefficient=None,
                standard_error=None,
                p_value=None,
            )

    cohens_d = {
        metric: compute_cohens_d(df, metric)
        for metric in COHENS_D_METRICS
    }

    return SeedRegressionResult(
        seed=seed,
        coefficients=coefficients,
        cohens_d=cohens_d,
    )


def aggregate_results(per_seed: list[SeedRegressionResult]) -> AggregateAnalysisResult:
    """Aggregate coefficient and Cohen's d summaries across seeds."""
    coefficient_summary: dict[str, AggregateMetricSummary] = {}
    for term in ["const", *PREDICTOR_ORDER]:
        values = [seed_result.coefficients[term].coefficient for seed_result in per_seed]
        coefficient_summary[term] = summarize_optional_values(values)

    cohens_d_summary = {
        metric: summarize_optional_values([seed_result.cohens_d[metric] for seed_result in per_seed])
        for metric in COHENS_D_METRICS
    }

    certainty_sign_consistency = {
        term: sign_consistency([seed_result.coefficients[term].coefficient for seed_result in per_seed])
        for term in CERTAINTY_TERMS
    }

    return AggregateAnalysisResult(
        coefficient_summary=coefficient_summary,
        cohens_d_summary=cohens_d_summary,
        certainty_sign_consistency=certainty_sign_consistency,
    )


def determine_verdict(
    per_seed: list[SeedRegressionResult],
    aggregate: AggregateAnalysisResult,
) -> tuple[str, str]:
    """Apply the frozen PASS/FAIL gate to the observed results."""
    for term in CERTAINTY_TERMS:
        d_summary = aggregate.cohens_d_summary[term]
        if d_summary.mean is None or d_summary.mean < 0.3:
            continue

        positive = sum(
            1
            for seed_result in per_seed
            if seed_result.coefficients[term].coefficient is not None
            and seed_result.coefficients[term].coefficient > 0
        )
        negative = sum(
            1
            for seed_result in per_seed
            if seed_result.coefficients[term].coefficient is not None
            and seed_result.coefficients[term].coefficient < 0
        )
        if max(positive, negative) >= 4:
            all_p_values = [
                seed_result.coefficients[term].p_value
                for seed_result in per_seed
                if seed_result.coefficients[term].p_value is not None
            ]
            if all_p_values and any(p_value < 0.05 for p_value in all_p_values):
                return (
                    "PASS",
                    f"{term} has mean Cohen's d {d_summary.mean:.3f} with sign consistency "
                    f"{max(positive, negative)}/5 and at least one seed-level p-value below 0.05.",
                )

    return (
        "FAIL",
        "Neither certainty layer retains a statistically supported association after the specified controls; "
        "the certainty coefficients are non-significant under perfect separation by input-space nearest-neighbour distance.",
    )


def _term_result(result: sm.GLM, term: str) -> RegressionTermResult:
    return RegressionTermResult(
        coefficient=float(result.params[term]),
        standard_error=float(result.bse[term]),
        p_value=float(result.pvalues[term]),
    )


def compute_cohens_d(df: pd.DataFrame, metric: str) -> float:
    """Compute seen-minus-unseen Cohen's d using pooled population variance."""
    seen_values = df.loc[df["seen_unseen_label"] == 0, metric].to_numpy(dtype=float)
    unseen_values = df.loc[df["seen_unseen_label"] == 1, metric].to_numpy(dtype=float)
    seen_mean = float(np.mean(seen_values))
    unseen_mean = float(np.mean(unseen_values))
    seen_var = float(np.var(seen_values))
    unseen_var = float(np.var(unseen_values))
    pooled_variance = ((len(seen_values) * seen_var) + (len(unseen_values) * unseen_var)) / (
        len(seen_values) + len(unseen_values)
    )
    return (seen_mean - unseen_mean) / math.sqrt(pooled_variance)


def summarize_optional_values(values: list[float | None]) -> AggregateMetricSummary:
    finite_values = [value for value in values if value is not None and math.isfinite(value)]
    if not finite_values:
        return AggregateMetricSummary(mean=None, std=None)
    array = np.asarray(finite_values, dtype=float)
    return AggregateMetricSummary(
        mean=float(np.mean(array)),
        std=float(np.std(array, ddof=0)),
    )


def sign_consistency(values: list[float | None]) -> str:
    finite_values = [value for value in values if value is not None and math.isfinite(value)]
    positive = sum(value > 0 for value in finite_values)
    negative = sum(value < 0 for value in finite_values)
    return f"{max(positive, negative)}/{len(finite_values)}"
