"""Unit smoke test for the E4 monotone certainty and ordinal heads."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

VARIANT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VARIANT_DIR))

from extrapolation.models.bottleneck import MonotoneCertHead, OrderedThresholdHead


def main() -> None:
    torch.manual_seed(0)

    cert_head = MonotoneCertHead(input_dim=6, hidden_dim=8, scale_init=4.0)
    low_badness = torch.zeros(4, 6)
    mid_badness = torch.full((4, 6), 0.5)
    high_badness = torch.ones(4, 6)

    cert_low = cert_head(low_badness)
    cert_mid = cert_head(mid_badness)
    cert_high = cert_head(high_badness)

    if not torch.all(cert_low >= cert_mid):
        raise AssertionError("MonotoneCertHead increased certainty for higher badness")
    if not torch.all(cert_mid >= cert_high):
        raise AssertionError("MonotoneCertHead increased certainty for highest badness")

    outcome_head = OrderedThresholdHead(
        threshold_init=0.9,
        threshold_gap_init=0.6,
        scale_init=4.0,
        reorder_indices=(0, 1, 2),
    )
    confidence_head = OrderedThresholdHead(
        threshold_init=0.7,
        threshold_gap_init=0.45,
        scale_init=4.0,
        reorder_indices=(2, 1, 0),
    )

    risks = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    outcome_probs = torch.softmax(outcome_head(risks), dim=-1)
    confidence_probs = torch.softmax(confidence_head(risks), dim=-1)

    success_probs = outcome_probs[:, 0]
    failure_probs = outcome_probs[:, 2]
    high_confidence_probs = confidence_probs[:, 2]
    low_confidence_probs = confidence_probs[:, 0]

    if not torch.all(success_probs[:-1] >= success_probs[1:]):
        raise AssertionError("Outcome success probability increased with higher risk")
    if not torch.all(failure_probs[:-1] <= failure_probs[1:]):
        raise AssertionError("Outcome failure probability decreased with higher risk")
    if not torch.all(high_confidence_probs[:-1] >= high_confidence_probs[1:]):
        raise AssertionError("Confidence HIGH probability increased with higher risk")
    if not torch.all(low_confidence_probs[:-1] <= low_confidence_probs[1:]):
        raise AssertionError("Confidence LOW probability decreased with higher risk")

    thresholds = outcome_head.thresholds()
    if not bool(thresholds[1] > thresholds[0]):
        raise AssertionError("OrderedThresholdHead thresholds are not ordered")

    print("E4 monotone heads: OK")


if __name__ == "__main__":
    main()
