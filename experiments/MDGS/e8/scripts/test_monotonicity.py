"""Unit monotonicity test for the E8 certainty/fragility scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

VARIANT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VARIANT_DIR))

from extrapolation.models.bottleneck import (
    DiffusionFragilityHead,
    DiffusionPrototypeBank,
    DiffusionTrustHead,
    OrderedThresholdHead,
)


def main() -> None:
    torch.manual_seed(0)

    confidence_head = OrderedThresholdHead(
        threshold_init=0.35,
        threshold_gap_init=0.30,
        scale_init=4.0,
    )

    signals = torch.tensor([0.1, 0.3, 0.6, 0.9])
    confidence_log_probs = confidence_head(signals)
    confidence_probs = torch.softmax(confidence_log_probs, dim=-1)

    high_confidence_probs = confidence_probs[:, 2]
    low_confidence_probs = confidence_probs[:, 0]

    if not torch.all(high_confidence_probs[:-1] <= high_confidence_probs[1:]):
        raise AssertionError("Confidence HIGH probability should increase with higher certainty signal")
    if not torch.all(low_confidence_probs[:-1] >= low_confidence_probs[1:]):
        raise AssertionError("Confidence LOW probability should decrease with higher certainty signal")

    thresholds = confidence_head.thresholds()
    if not bool(thresholds[1] > thresholds[0]):
        raise AssertionError("OrderedThresholdHead thresholds are not ordered")

    print("OrderedThresholdHead monotonicity: OK")

    trust_head = DiffusionTrustHead(hidden_dim=32)

    good_stats = torch.tensor([[
        0.05,  # latent_recovery_var_norm   low (good)
        0.90,  # answer_agreement           high (good)
        0.10,  # answer_entropy_norm        low (good)
        0.05,  # denoise_energy_norm        low (good)
        0.90,  # basin_stability            high (good)
    ]])
    bad_stats = torch.tensor([[
        0.80,  # latent_recovery_var_norm   high (bad)
        0.20,  # answer_agreement           low (bad)
        0.85,  # answer_entropy_norm        high (bad)
        0.80,  # denoise_energy_norm        high (bad)
        0.15,  # basin_stability            low (bad)
    ]])

    certainty_good = trust_head(good_stats)
    certainty_bad = trust_head(bad_stats)

    if not bool(certainty_good > certainty_bad):
        raise AssertionError(
            f"DiffusionTrustHead: certainty should be higher for good stats "
            f"(got good={certainty_good.item():.4f}, bad={certainty_bad.item():.4f})"
        )

    assert certainty_good.shape == (1,), f"unexpected certainty shape: {certainty_good.shape}"
    assert certainty_bad.shape == (1,), f"unexpected certainty shape: {certainty_bad.shape}"
    assert torch.isfinite(certainty_good).all(), "certainty contains non-finite values"
    assert torch.isfinite(certainty_bad).all(), "certainty contains non-finite values"
    print("DiffusionTrustHead certainty ordering: OK")

    prototype_bank = DiffusionPrototypeBank(state_dim=8, prototypes_per_family=4)
    path_states = torch.randn(6, 5, 8)
    prototype_payload = prototype_bank(path_states)
    assert prototype_payload["family_support"].shape == (5, 3)
    assert prototype_payload["anchor_assignment_probs"].shape == (5, 3, 4)
    assert torch.isfinite(prototype_payload["family_conflict"]).all()
    assert torch.isfinite(prototype_payload["overlap"]).all()
    assert torch.isfinite(prototype_payload["family_switch_rate"]).all()
    assert torch.isfinite(prototype_payload["anchor_switch_rate"]).all()
    print("DiffusionPrototypeBank shapes: OK")

    structured_bank = DiffusionPrototypeBank(state_dim=2, prototypes_per_family=4)
    with torch.no_grad():
        structured_bank.centers.zero_()
        structured_bank.centers[0, 0] = torch.tensor([0.0, 0.0])
        structured_bank.centers[0, 1] = torch.tensor([2.0, 0.0])
        structured_bank.centers[0, 2] = torch.tensor([4.0, 0.0])
        structured_bank.centers[0, 3] = torch.tensor([6.0, 0.0])
        structured_bank.centers[1].copy_(torch.tensor([[0.0, 10.0], [2.0, 10.0], [4.0, 10.0], [6.0, 10.0]]))
        structured_bank.centers[2].copy_(torch.tensor([[0.0, -10.0], [2.0, -10.0], [4.0, -10.0], [6.0, -10.0]]))
        structured_bank.raw_scales.zero_()

    structured_states = torch.tensor(
        [
            [[0.0, 0.0]],
            [[2.0, 0.0]],
            [[0.0, 0.0]],
            [[2.0, 0.0]],
        ]
    )
    structured_payload = structured_bank(structured_states)
    if not bool(structured_payload["family_switch_rate"][0] < 1e-6):
        raise AssertionError("family_switch_rate should stay near zero when paths remain in one family")
    if not bool(structured_payload["anchor_switch_rate"][0] > 0.05):
        raise AssertionError("anchor_switch_rate should rise when paths switch anchors inside one family")
    if not bool(structured_payload["overlap"][0] < 0.2):
        raise AssertionError("prototype overlap should stay low when one family clearly dominates")
    print("Prototype family/anchor switch separation: OK")

    fragility_head = DiffusionFragilityHead(input_dim=19, hidden_dim=16)
    fragility_stats = torch.randn(4, 19)
    fragility_risk = fragility_head(fragility_stats)
    assert fragility_risk.shape == (4,), f"unexpected fragility shape: {fragility_risk.shape}"
    assert torch.isfinite(fragility_risk).all(), "fragility risk contains non-finite values"
    print("DiffusionFragilityHead forward: OK")

    certainty_score = torch.tensor([0.60, 0.60, 0.60])
    ambiguity_cert = torch.tensor([0.10, 0.50, 0.90])
    decisiveness_score = (1.0 - ambiguity_cert).clamp(0.0, 1.0)
    fragility_score = torch.tensor([0.90, 0.90, 0.90])
    effective_certainty = certainty_score * fragility_score
    commitment_depth = effective_certainty * decisiveness_score

    if not torch.all(commitment_depth[:-1] >= commitment_depth[1:]):
        raise AssertionError(
            "commitment_depth should decrease as ambiguity_cert increases at fixed certainty_score"
        )
    fragility_score = torch.tensor([0.90, 0.50, 0.10])
    effective_certainty = certainty_score * fragility_score
    commitment_depth = effective_certainty * torch.full_like(certainty_score, 0.80)
    if not torch.all(commitment_depth[:-1] >= commitment_depth[1:]):
        raise AssertionError(
            "commitment_depth should decrease as fragility increases at fixed certainty/decisiveness"
        )
    print("Effective certainty / commitment monotonicity: OK")

    print("\nAll E8 monotonicity tests PASSED")


if __name__ == "__main__":
    main()
