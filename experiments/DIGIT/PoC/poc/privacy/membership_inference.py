"""Membership Inference Attack against the DIGIT bottleneck.

Threat model: adversary has black-box access to DIGIT (can submit queries,
observe output primitives). Adversary wants to determine if a specific
record was in the private dataset.

Approach: shadow model attack (Shokri et al., 2017 adapted for DIGIT).
- Train shadow DIGIT models on random subsets of the private data
- For each record, observe DIGIT's output on a query targeting its group
- Train an attack classifier: (record features, DIGIT output) -> member/non-member
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split

from ..data.private_store import PrivateAdultDataset
from ..models.digit import DIGITModel

logger = logging.getLogger(__name__)


def build_attack_features(
    model: DIGITModel,
    private_data: PrivateAdultDataset,
    member_indices: np.ndarray,
    nonmember_indices: np.ndarray,
    all_features: torch.Tensor,
    all_labels: torch.Tensor,
    device: torch.device,
    num_samples: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X_attack, y_attack) for the membership inference classifier.

    For each target record:
    1. Construct a query matching the record's demographic group
    2. Run DIGIT to get output primitives
    3. Concatenate (record features, primitive outputs) as attack features
    4. Label: 1 if member, 0 if non-member

    Returns:
        X_attack: (num_samples, feature_dim) attack classifier features
        y_attack: (num_samples,) binary labels
    """
    model.eval()
    rng = np.random.RandomState(42)

    # Sample balanced member/non-member sets
    n_each = min(num_samples // 2, len(member_indices), len(nonmember_indices))
    sampled_members = rng.choice(member_indices, size=n_each, replace=False)
    sampled_nonmembers = rng.choice(nonmember_indices, size=n_each, replace=False)

    attack_X = []
    attack_y = []

    for idx, is_member in [(sampled_members, 1), (sampled_nonmembers, 0)]:
        for record_idx in idx:
            record_features = all_features[record_idx]  # (12,)

            # Construct query from record's attributes
            query = torch.zeros(13, dtype=torch.long)
            # Use the record's own attributes as the query filter
            for col in range(12):
                query[col] = record_features[col].item()
            query[12] = 12  # max specificity

            query = query.unsqueeze(0).to(device)

            with torch.no_grad():
                result = model.generate(query, private_data)
                prims = model.bottleneck.get_primitive_indices(result["primitives"])

            # Attack features: record attributes + DIGIT output
            record_np = record_features.cpu().numpy().astype(float)
            prim_np = np.array([
                prims["answer"][0].item(),
                prims["support"][0].item(),
                prims["confidence"][0].item(),
                prims["risk"][0].item(),
            ], dtype=float)

            attack_feat = np.concatenate([record_np, prim_np])
            attack_X.append(attack_feat)
            attack_y.append(is_member)

    return np.array(attack_X), np.array(attack_y)


def run_membership_inference_attack(
    model: DIGITModel,
    private_data: PrivateAdultDataset,
    holdout_features: torch.Tensor,
    holdout_labels: torch.Tensor,
    device: torch.device,
    num_samples: int = 2000,
) -> Dict[str, float]:
    """Run membership inference attack and return metrics.

    Args:
        model: Trained DIGIT model.
        private_data: The private dataset used during training.
        holdout_features: Features of records NOT in private_data.
        holdout_labels: Labels of holdout records.
        device: Torch device.
        num_samples: Total attack samples (half member, half non-member).

    Returns:
        Dict with attack_auc, attack_accuracy, attack_advantage.
    """
    logger.info("Running membership inference attack...")

    # Member indices: records in private_data
    member_indices = np.arange(private_data.num_records)
    # Non-member indices: records in holdout
    nonmember_indices = np.arange(len(holdout_features))

    # Build attack features
    # For members: use private_data.features
    # For non-members: use holdout_features
    all_features = torch.cat([private_data.features, holdout_features], dim=0)
    all_labels = torch.cat([private_data.labels, holdout_labels], dim=0)

    # Non-member indices offset by private_data size
    nonmember_indices_offset = nonmember_indices + private_data.num_records

    X_attack, y_attack = build_attack_features(
        model=model,
        private_data=private_data,
        member_indices=member_indices,
        nonmember_indices=nonmember_indices_offset,
        all_features=all_features,
        all_labels=all_labels,
        device=device,
        num_samples=num_samples,
    )

    # Split attack data for training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_attack, y_attack, test_size=0.4, random_state=42, stratify=y_attack,
    )

    results = {}

    # Logistic Regression attack
    lr_attack = LogisticRegression(max_iter=1000, random_state=42)
    lr_attack.fit(X_train, y_train)
    lr_probs = lr_attack.predict_proba(X_test)[:, 1]
    lr_preds = lr_attack.predict(X_test)

    results["lr_auc"] = float(roc_auc_score(y_test, lr_probs))
    results["lr_accuracy"] = float(accuracy_score(y_test, lr_preds))
    # advantage = 2*(AUROC - 0.5)  [harness v1.0]
    results["lr_advantage"] = 2.0 * (results["lr_auc"] - 0.5)

    def _tpr_at_fpr(probs, targets, target_fpr):
        fpr, tpr, _ = roc_curve(targets, probs)
        idx = min(np.searchsorted(fpr, target_fpr, side="right"), len(tpr) - 1)
        return float(tpr[idx])

    results["lr_tpr_at_1pct_fpr"] = _tpr_at_fpr(lr_probs, y_test, 0.01)
    results["lr_tpr_at_5pct_fpr"] = _tpr_at_fpr(lr_probs, y_test, 0.05)

    # MLP attack (stronger)
    mlp_attack = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
    )
    mlp_attack.fit(X_train, y_train)
    mlp_probs = mlp_attack.predict_proba(X_test)[:, 1]
    mlp_preds = mlp_attack.predict(X_test)

    results["mlp_auc"] = float(roc_auc_score(y_test, mlp_probs))
    results["mlp_accuracy"] = float(accuracy_score(y_test, mlp_preds))
    results["mlp_advantage"] = 2.0 * (results["mlp_auc"] - 0.5)
    results["mlp_tpr_at_1pct_fpr"] = _tpr_at_fpr(mlp_probs, y_test, 0.01)
    results["mlp_tpr_at_5pct_fpr"] = _tpr_at_fpr(mlp_probs, y_test, 0.05)

    # Best attack (stronger of LR / MLP, measured by AUROC)
    best = "mlp" if results["mlp_auc"] >= results["lr_auc"] else "lr"
    results["best_auc"]           = results[f"{best}_auc"]
    results["best_accuracy"]      = results[f"{best}_accuracy"]
    results["best_advantage"]     = results[f"{best}_advantage"]
    results["best_tpr_at_1pct_fpr"] = results[f"{best}_tpr_at_1pct_fpr"]
    results["best_tpr_at_5pct_fpr"] = results[f"{best}_tpr_at_5pct_fpr"]

    logger.info(f"MIA Results — LR AUC: {results['lr_auc']:.3f}, "
                f"MLP AUC: {results['mlp_auc']:.3f}, "
                f"best_advantage: {results['best_advantage']:.3f}")

    return results
