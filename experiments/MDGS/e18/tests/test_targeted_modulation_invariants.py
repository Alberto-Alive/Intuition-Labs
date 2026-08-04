"""Pre-implementation invariant tests for DIGIT Extrapolation E18."""

from __future__ import annotations

import torch


THRESHOLD = 0.95
ALPHA = 10.0
BETA = 0.1


def build_targeted_row_weights(
    losses: torch.Tensor,
    certainty: torch.Tensor,
    *,
    threshold: float = THRESHOLD,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> torch.Tensor:
    """Return the per-example row weights implied by the frozen E18 rule."""
    detached_losses = losses.detach()
    detached_certainty = certainty.detach()

    row_weights = torch.ones_like(detached_losses)
    low_certainty_mask = (~torch.isnan(detached_certainty)) & (detached_certainty < threshold)

    if int(low_certainty_mask.sum().item()) == 0:
        return row_weights

    low_certainty = detached_certainty[low_certainty_mask]
    raw_scale = alpha * low_certainty + beta * (1.0 - low_certainty)
    normalized_raw_scale = raw_scale / raw_scale.mean()
    row_weights[low_certainty_mask] = detached_losses[low_certainty_mask] * normalized_raw_scale
    return row_weights


def apply_targeted_hook(
    activation: torch.Tensor,
    losses: torch.Tensor,
    certainty: torch.Tensor,
) -> torch.Tensor:
    """Attach the frozen E18 hook to an activation tensor and return row weights."""
    row_weights = build_targeted_row_weights(losses, certainty)
    activation.register_hook(lambda grad: grad * row_weights.unsqueeze(-1))
    return row_weights


def test_high_certainty_examples_above_0_95_are_left_unmodified() -> None:
    """High-certainty rows must pass through the hook unchanged."""
    activation = torch.ones((3, 4), dtype=torch.float32, requires_grad=True)
    losses = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
    certainty = torch.tensor([0.96, 0.98, 0.999], dtype=torch.float32)

    row_weights = apply_targeted_hook(activation, losses, certainty)
    activation.sum().backward()

    assert torch.allclose(row_weights, torch.ones_like(losses))
    assert torch.allclose(activation.grad, torch.ones_like(activation))


def test_low_certainty_examples_receive_batch_normalized_asymmetric_scaling() -> None:
    """Low-certainty rows must receive the exact frozen E18 scaling."""
    activation = torch.ones((3, 2), dtype=torch.float64, requires_grad=True)
    losses = torch.tensor([2.0, 4.0, 3.0], dtype=torch.float64)
    certainty = torch.tensor([0.80, 0.90, 0.99], dtype=torch.float64)

    row_weights = apply_targeted_hook(activation, losses, certainty)
    activation.sum().backward()

    raw_low = torch.tensor(
        [
            ALPHA * 0.80 + BETA * (1.0 - 0.80),
            ALPHA * 0.90 + BETA * (1.0 - 0.90),
        ],
        dtype=torch.float64,
    )
    normalized_low = raw_low / raw_low.mean()
    expected_row_weights = torch.tensor(
        [
            2.0 * normalized_low[0].item(),
            4.0 * normalized_low[1].item(),
            1.0,
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(row_weights, expected_row_weights)
    expected_grad = expected_row_weights.unsqueeze(-1).expand_as(activation)
    assert torch.allclose(activation.grad, expected_grad)


def test_normalization_uses_only_low_certainty_subset() -> None:
    """High-certainty rows must not contribute to the low-certainty normalization mean."""
    losses = torch.ones(3, dtype=torch.float64)
    certainty = torch.tensor([0.80, 0.90, 0.99], dtype=torch.float64)

    row_weights = build_targeted_row_weights(losses, certainty)

    raw_low = torch.tensor(
        [
            ALPHA * 0.80 + BETA * (1.0 - 0.80),
            ALPHA * 0.90 + BETA * (1.0 - 0.90),
        ],
        dtype=torch.float64,
    )
    expected_low_subset_only = raw_low / raw_low.mean()

    raw_all = torch.tensor(
        [
            ALPHA * 0.80 + BETA * (1.0 - 0.80),
            ALPHA * 0.90 + BETA * (1.0 - 0.90),
            ALPHA * 0.99 + BETA * (1.0 - 0.99),
        ],
        dtype=torch.float64,
    )
    all_batch_normalized = raw_all / raw_all.mean()

    assert torch.allclose(row_weights[:2], expected_low_subset_only)
    assert not torch.allclose(row_weights[:2], all_batch_normalized[:2])
    assert torch.allclose(row_weights[2:], torch.ones(1, dtype=torch.float64))


def test_batches_with_no_low_certainty_examples_apply_no_modulation() -> None:
    """If no row is below threshold, the hook must be a no-op for the full batch."""
    activation = torch.ones((2, 3), dtype=torch.float32, requires_grad=True)
    losses = torch.tensor([1.5, 2.5], dtype=torch.float32)
    certainty = torch.tensor([0.96, 0.999], dtype=torch.float32)

    row_weights = apply_targeted_hook(activation, losses, certainty)
    activation.sum().backward()

    assert torch.allclose(row_weights, torch.ones_like(losses))
    assert torch.allclose(activation.grad, torch.ones_like(activation))


def test_initial_fill_period_behaves_exactly_like_e16() -> None:
    """Invalid certainty must apply no modulation, matching the E16 backward path."""
    baseline_activation = torch.ones((2, 2), dtype=torch.float32, requires_grad=True)
    targeted_activation = torch.ones((2, 2), dtype=torch.float32, requires_grad=True)
    losses = torch.tensor([2.0, 4.0], dtype=torch.float32)
    invalid_certainty = torch.tensor([float("nan"), float("nan")], dtype=torch.float32)

    baseline_activation.sum().backward()
    row_weights = apply_targeted_hook(targeted_activation, losses, invalid_certainty)
    targeted_activation.sum().backward()

    assert torch.allclose(row_weights, torch.ones_like(losses))
    assert torch.allclose(targeted_activation.grad, baseline_activation.grad)
