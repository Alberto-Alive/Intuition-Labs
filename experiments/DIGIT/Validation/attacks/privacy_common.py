"""Shared helpers for comparative privacy attacks."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right")
    idx = min(idx, len(tpr) - 1)
    return float(tpr[idx])


def attack_advantage(auroc: float) -> float:
    return 2.0 * (float(auroc) - 0.5)


def binary_metrics_from_probs(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    auroc = float(roc_auc_score(y_true, y_score))
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "auroc": auroc,
        "advantage": attack_advantage(auroc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "tpr_at_1pct_fpr": tpr_at_fpr(y_true, y_score, 0.01),
        "tpr_at_5pct_fpr": tpr_at_fpr(y_true, y_score, 0.05),
    }


def chance_metrics(binary: bool = True) -> Dict[str, float]:
    if binary:
        return {
            "auroc": 0.5,
            "advantage": 0.0,
            "accuracy": 0.5,
            "balanced_accuracy": 0.5,
            "macro_f1": 0.5,
            "tpr_at_1pct_fpr": 0.0,
            "tpr_at_5pct_fpr": 0.0,
        }
    return {
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "macro_f1": 0.0,
    }


def best_binary_classifier(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    seed: int,
) -> Tuple[Optional[Dict[str, float]], Optional[np.ndarray], str]:
    best_metrics: Optional[Dict[str, float]] = None
    best_probs: Optional[np.ndarray] = None
    best_name = "none"

    for AttackCls, kw in [
        (LogisticRegression, dict(max_iter=1000, random_state=seed, C=0.1)),
        (
            MLPClassifier,
            dict(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.1,
            ),
        ),
    ]:
        try:
            clf = AttackCls(**kw)
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)[:, 1]
            metrics = binary_metrics_from_probs(y_te, probs)
            if best_metrics is None or metrics["auroc"] > best_metrics["auroc"]:
                best_metrics = metrics
                best_probs = probs
                best_name = AttackCls.__name__
        except Exception:
            continue
    return best_metrics, best_probs, best_name


def run_binary_attack(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, str]:
    if X.size == 0 or len(np.unique(y)) < 2:
        return chance_metrics(), np.array([], dtype=float), np.array([], dtype=float), "none"

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=seed,
        stratify=y,
    )
    metrics, probs, name = best_binary_classifier(X_tr, y_tr, X_te, y_te, seed)
    if metrics is None or probs is None:
        return chance_metrics(), y_te, np.zeros_like(y_te, dtype=float), "none"
    return metrics, y_te, probs, name
