"""Post-hoc regression analysis for E12."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


@dataclass(frozen=True)
class RegressionResult:
    feature_columns: list[str]
    coefficients: dict[str, float]
    p_values: dict[str, float]
    standardized_coefficients: dict[str, float]
    r2_full: float
    r2_reduced: float
    partial_r2_seen_unseen: float
    pass_condition: bool


def _standardize(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std == 0.0:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def _fit_ols(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> tuple[dict[str, float], dict[str, float], float]:
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    predictions = X @ beta
    residuals = y - predictions
    rss = float(residuals.T @ residuals)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - (rss / tss if tss > 0.0 else 0.0)

    dof = max(len(y) - X.shape[1], 1)
    sigma2 = rss / dof
    covariance = sigma2 * np.linalg.pinv(X.T @ X)
    std_err = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    t_values = beta / std_err
    p_values = 2.0 * (1.0 - student_t.cdf(np.abs(t_values), df=dof))

    coefficients = {name: float(value) for name, value in zip(feature_names, beta, strict=True)}
    p_value_map = {name: float(value) for name, value in zip(feature_names, p_values, strict=True)}
    return coefficients, p_value_map, r2


def _fit_support_regression(
    records: list[dict[str, float | int]],
    feature_columns: list[str],
    reduced_columns: list[str],
) -> RegressionResult:
    frame = pd.DataFrame.from_records(records)
    target = frame["final_support"].to_numpy(dtype=np.float64)

    standardized = {column: _standardize(frame[column].to_numpy(dtype=np.float64)) for column in feature_columns}
    standardized_target = _standardize(target)

    X_full = np.column_stack(
        [np.ones(len(frame), dtype=np.float64)]
        + [standardized[column] for column in feature_columns]
    )
    full_feature_names = ["intercept"] + feature_columns
    full_coefficients, full_p_values, r2_full = _fit_ols(X_full, standardized_target, full_feature_names)

    X_reduced = np.column_stack(
        [np.ones(len(frame), dtype=np.float64)]
        + [standardized[column] for column in reduced_columns]
    )
    _, _, r2_reduced = _fit_ols(X_reduced, standardized_target, ["intercept"] + reduced_columns)

    partial_r2 = max(r2_full - r2_reduced, 0.0)
    standardized_coefficients = full_coefficients
    pass_condition = (
        standardized_coefficients["seen_unseen_label"] < 0.0
        and full_p_values["seen_unseen_label"] < 0.01
        and abs(standardized_coefficients["seen_unseen_label"])
        > max(
            abs(standardized_coefficients["marginal_frequency"]),
            abs(standardized_coefficients["confidence"]),
            abs(standardized_coefficients["loss"]),
        )
    )

    return RegressionResult(
        feature_columns=feature_columns,
        coefficients=full_coefficients,
        p_values=full_p_values,
        standardized_coefficients=standardized_coefficients,
        r2_full=r2_full,
        r2_reduced=r2_reduced,
        partial_r2_seen_unseen=partial_r2,
        pass_condition=pass_condition,
    )


def fit_support_regression(records: list[dict[str, float | int]]) -> RegressionResult:
    return _fit_support_regression(
        records=records,
        feature_columns=[
            "marginal_frequency",
            "confidence",
            "loss",
            "local_support",
            "joint_support",
            "seen_unseen_label",
        ],
        reduced_columns=[
            "marginal_frequency",
            "confidence",
            "loss",
            "local_support",
            "joint_support",
        ],
    )


def fit_primary_pass_regression(records: list[dict[str, float | int]]) -> RegressionResult:
    return _fit_support_regression(
        records=records,
        feature_columns=[
            "marginal_frequency",
            "confidence",
            "loss",
            "seen_unseen_label",
        ],
        reduced_columns=[
            "marginal_frequency",
            "confidence",
            "loss",
        ],
    )
