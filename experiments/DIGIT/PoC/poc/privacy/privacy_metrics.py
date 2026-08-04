"""Privacy metric aggregation across seeds."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def aggregate_privacy_results(
    mia_results: List[Dict[str, float]],
    aia_results: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate privacy attack results across seeds.

    Returns mean +/- std for each metric.
    """
    result = {}

    # MIA metrics
    for key in ["lr_auc", "mlp_auc", "best_auc", "lr_accuracy",
                "mlp_accuracy", "best_accuracy", "lr_advantage", "mlp_advantage"]:
        values = [r[key] for r in mia_results if key in r]
        if values:
            result[f"mia_{key}_mean"] = float(np.mean(values))
            result[f"mia_{key}_std"] = float(np.std(values))

    # AIA metrics
    for key in ["baseline_auc", "informed_auc", "auc_advantage",
                "baseline_accuracy", "informed_accuracy", "accuracy_advantage",
                "mlp_baseline_auc", "mlp_informed_auc", "mlp_advantage"]:
        values = [r[key] for r in aia_results if key in r]
        if values:
            result[f"aia_{key}_mean"] = float(np.mean(values))
            result[f"aia_{key}_std"] = float(np.std(values))

    return result
