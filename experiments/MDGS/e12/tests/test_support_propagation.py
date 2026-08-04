from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extrapolation.models import SupportPropagation


def test_support_propagation_is_logit_additive_and_explicit() -> None:
    propagation = SupportPropagation(eps=1e-6)
    previous_context_logit = torch.tensor([0.2, -0.4])
    local_support = torch.tensor(
        [
            [0.8, 0.6, 0.4],
            [0.9, 0.2, 0.5],
        ]
    )
    joint_support = torch.tensor([0.75, 0.4])

    state = propagation(
        previous_context_logit=previous_context_logit,
        local_support=local_support,
        joint_support=joint_support,
    )

    expected_per_neuron = torch.sigmoid(
        previous_context_logit.unsqueeze(-1)
        + torch.logit(local_support)
        + torch.logit(joint_support).unsqueeze(-1)
    )
    expected_layer_evidence = 0.5 * (local_support.mean(dim=-1) + joint_support)
    expected_next_context_logit = previous_context_logit + torch.logit(expected_layer_evidence)

    assert torch.allclose(state.per_neuron_support, expected_per_neuron, atol=1e-6)
    assert torch.allclose(state.layer_evidence, expected_layer_evidence, atol=1e-6)
    assert torch.allclose(state.next_context_logit, expected_next_context_logit, atol=1e-6)
    assert torch.allclose(state.next_context_support, torch.sigmoid(expected_next_context_logit), atol=1e-6)


def test_support_propagation_depends_on_previous_layer_context() -> None:
    propagation = SupportPropagation(eps=1e-6)
    local_support = torch.tensor([[0.7, 0.3]])
    joint_support = torch.tensor([0.8])

    low_context = propagation(
        previous_context_logit=torch.tensor([-1.0]),
        local_support=local_support,
        joint_support=joint_support,
    )
    high_context = propagation(
        previous_context_logit=torch.tensor([1.0]),
        local_support=local_support,
        joint_support=joint_support,
    )

    assert not torch.allclose(low_context.per_neuron_support, high_context.per_neuron_support)
    assert not torch.allclose(low_context.next_context_support, high_context.next_context_support)
