"""Attribute Inference Attack against the DIGIT bottleneck.

Threat model: adversary knows a target individual's non-sensitive attributes
but NOT their income. The adversary also has access to DIGIT's output for
queries matching the target's demographic subgroup.

Question: does the DIGIT output help the adversary predict income better
than knowing just the demographics?

If DIGIT provides no extra signal, the advantage should be near 0.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

from ..data.private_store import PrivateAdultDataset
from ..models.digit import DIGITModel

logger = logging.getLogger(__name__)


def run_attribute_inference_attack(
    model: DIGITModel,
    private_data: PrivateAdultDataset,
    target_features: torch.Tensor,
    target_labels: torch.Tensor,
    device: torch.device,
    num_samples: int = 2000,
) -> Dict[str, float]:
    """Run attribute inference attack.

    Compares two attackers:
    1. Baseline: predicts income from demographics only
    2. Informed: predicts income from demographics + DIGIT output

    The difference in AUC is the "attribute inference advantage".
    """
    model.eval()
    rng = np.random.RandomState(42)

    n = min(num_samples, len(target_features))
    indices = rng.choice(len(target_features), size=n, replace=False)

    X_demographics = []
    X_with_digit = []
    y_true = []

    for idx in indices:
        record = target_features[idx]
        label = target_labels[idx].item()

        # Build a query from the record's demographics
        query = torch.zeros(13, dtype=torch.long)
        for col in range(12):
            query[col] = record[col].item()
        query[12] = 12  # max specificity

        query_input = query.unsqueeze(0).to(device)

        with torch.no_grad():
            result = model.generate(query_input, private_data)
            prims = model.bottleneck.get_primitive_indices(result["primitives"])

        record_np = record.cpu().numpy().astype(float)
        prim_np = np.array([
            prims["answer"][0].item(),
            prims["support"][0].item(),
            prims["confidence"][0].item(),
            prims["risk"][0].item(),
        ], dtype=float)

        X_demographics.append(record_np)
        X_with_digit.append(np.concatenate([record_np, prim_np]))
        y_true.append(label)

    X_demo = np.array(X_demographics)
    X_informed = np.array(X_with_digit)
    y = np.array(y_true)

    # Consistent split using index shuffle
    all_idx = np.arange(len(y))
    rng2 = np.random.RandomState(42)
    rng2.shuffle(all_idx)
    split_point = int(len(all_idx) * 0.6)
    train_mask = all_idx[:split_point]
    test_mask = all_idx[split_point:]

    X_demo_train, X_demo_test = X_demo[train_mask], X_demo[test_mask]
    X_inf_train, X_inf_test = X_informed[train_mask], X_informed[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    results = {}

    # Logistic Regression
    lr_baseline = LogisticRegression(max_iter=1000, random_state=42)
    lr_baseline.fit(X_demo_train, y_train)
    lr_base_probs = lr_baseline.predict_proba(X_demo_test)[:, 1]
    lr_base_preds = lr_baseline.predict(X_demo_test)

    lr_informed = LogisticRegression(max_iter=1000, random_state=42)
    lr_informed.fit(X_inf_train, y_train)
    lr_inf_probs = lr_informed.predict_proba(X_inf_test)[:, 1]
    lr_inf_preds = lr_informed.predict(X_inf_test)

    results["baseline_auc"] = float(roc_auc_score(y_test, lr_base_probs))
    results["informed_auc"] = float(roc_auc_score(y_test, lr_inf_probs))
    results["auc_advantage"] = results["informed_auc"] - results["baseline_auc"]
    results["baseline_accuracy"] = float(accuracy_score(y_test, lr_base_preds))
    results["informed_accuracy"] = float(accuracy_score(y_test, lr_inf_preds))
    results["accuracy_advantage"] = results["informed_accuracy"] - results["baseline_accuracy"]

    # MLP attack (stronger)
    mlp_baseline = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp_baseline.fit(X_demo_train, y_train)
    mlp_base_probs = mlp_baseline.predict_proba(X_demo_test)[:, 1]

    mlp_informed = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp_informed.fit(X_inf_train, y_train)
    mlp_inf_probs = mlp_informed.predict_proba(X_inf_test)[:, 1]

    results["mlp_baseline_auc"] = float(roc_auc_score(y_test, mlp_base_probs))
    results["mlp_informed_auc"] = float(roc_auc_score(y_test, mlp_inf_probs))
    results["mlp_advantage"] = results["mlp_informed_auc"] - results["mlp_baseline_auc"]

    logger.info(
        f"AIA Results — LR advantage: {results['auc_advantage']:.4f}, "
        f"MLP advantage: {results['mlp_advantage']:.4f}"
    )

    return results
