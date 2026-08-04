"""Pre-training invariant tests for DIGIT Extrapolation E25.

These four tests must all pass before any training or analysis code is written.
They are deliberately self-contained: they do not import any e25 training
module. Each test constructs its own minimal substrate and verifies one frozen
spec invariant.

Test 1 — E_jk values are bounded between 0 and 1 for all weights.
Test 2 — M_j binary activity indicator is exactly correct against known inputs.
Test 3 — 50% global magnitude pruning is computed across both weight matrices
          jointly, not per layer, and produces the correct global mask.
Test 4 — Rewound weights exactly match step-0 initialisation values.
"""

from __future__ import annotations

import os
import tempfile
from math import sqrt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

EPS = 1e-8  # spec-frozen numerical stability constant


def _compute_familiarity_exposure(
    F_scores: torch.Tensor,   # (N,)   per-example familiarity in [0,1]
    a_magnitudes: torch.Tensor,  # (N, d_in)  |a_k(x_i)| for each source neuron k
    M_indicators: torch.Tensor,  # (N, d_out) binary: destination neuron j active
) -> torch.Tensor:
    """Compute E_jk for a (d_out x d_in) weight matrix.

    E_jk = sum_i [ F(x_i) * |a_k(x_i)| * M_j(x_i) ]
           / ( sum_i [ |a_k(x_i)| * M_j(x_i) ] + epsilon )

    Returns a tensor of shape (d_out, d_in).
    """
    N, d_in = a_magnitudes.shape
    N2, d_out = M_indicators.shape
    assert N == N2, "N mismatch between a_magnitudes and M_indicators"

    # numerator:   sum_i F_i * M_j(x_i) * |a_k(x_i)|
    # shape: (N, d_out, d_in)  ->  (d_out, d_in)
    # Efficient broadcast form:
    #   weighted_a : (N, d_in)   = F_i * |a_k(x_i)|
    #   numerator  : (d_out, d_in) = M^T @ weighted_a
    weighted_a = F_scores.unsqueeze(1) * a_magnitudes        # (N, d_in)
    numerator = M_indicators.t().float() @ weighted_a.float()  # (d_out, d_in)

    # denominator: sum_i M_j(x_i) * |a_k(x_i)| + eps
    denominator = M_indicators.t().float() @ a_magnitudes.float() + EPS  # (d_out, d_in)

    return numerator / denominator  # (d_out, d_in)


def _compute_activity_indicator(post_relu: torch.Tensor) -> torch.Tensor:
    """Return binary M_j: 1 where post_relu > 0, else 0."""
    return (post_relu > 0).to(torch.long)


def _global_magnitude_prune(
    w1: torch.Tensor,
    w2: torch.Tensor,
    sparsity: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """One-shot global magnitude pruning across w1 and w2 jointly.

    Returns masks (same shapes as w1 and w2) and the threshold used.
    mask[i] = 1 if weight survives, 0 if pruned.
    """
    combined = torch.cat([w1.flatten().abs(), w2.flatten().abs()])
    n_total = combined.numel()
    n_prune = int(round(sparsity * n_total))
    # kth-value threshold: the n_prune-th smallest magnitude
    sorted_mags, _ = combined.sort()
    threshold = float(sorted_mags[n_prune - 1].item())

    mask1 = (w1.abs() > threshold).to(torch.long)
    mask2 = (w2.abs() > threshold).to(torch.long)
    return mask1, mask2, threshold


# ---------------------------------------------------------------------------
# Minimal model for Test 4 (weight rewinding)
# ---------------------------------------------------------------------------

class TinySemanticModel(nn.Module):
    """Two-layer MLP whose semantic weight matrices are tracked for LTH tests."""

    def __init__(self) -> None:
        super().__init__()
        self.w_semantic_1 = nn.Linear(32, 32, bias=False)
        self.w_semantic_2 = nn.Linear(32, 32, bias=False)
        self.head = nn.Linear(32, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.w_semantic_1(x))
        h2 = torch.relu(self.w_semantic_2(h1))
        return self.head(h2)


# ===========================================================================
# TEST 1 — E_jk values are bounded between 0 and 1 for all weights
# ===========================================================================

def test_familiarity_exposure_scores_are_bounded_between_zero_and_one() -> None:
    """E_jk must lie in [0, 1] inclusive for all weight positions.

    Three sub-cases are verified:
    a) Random realistic inputs: all values in [0, 1].
    b) Degenerate case where M_j is always zero: E_jk effectively zero (< 1e-6).
    c) Perfect familiarity case (F=1, M always active, a always positive): E_jk = 1.
    """
    torch.manual_seed(0)

    N = 100
    d_in = 32
    d_out = 32

    # --- sub-case a: random realistic inputs ---
    F_scores = torch.rand(N)                     # in [0, 1]
    a_mags = torch.rand(N, d_in) * 5.0          # in [0, 5]
    M_rand = torch.randint(0, 2, (N, d_out))    # binary

    E = _compute_familiarity_exposure(F_scores, a_mags, M_rand)

    assert E.shape == (d_out, d_in), f"Expected shape ({d_out}, {d_in}), got {tuple(E.shape)}"
    assert float(E.min().item()) >= 0.0, f"E_jk below 0: min={E.min().item()}"
    assert float(E.max().item()) <= 1.0 + 1e-6, f"E_jk above 1: max={E.max().item()}"

    # --- sub-case b: M_j always zero — denominator = eps, numerator = 0 ---
    M_zero = torch.zeros(N, d_out, dtype=torch.long)
    E_zero = _compute_familiarity_exposure(F_scores, a_mags, M_zero)

    # With M=0, numerator=0, denominator=eps → E = 0/eps = 0
    assert float(E_zero.max().item()) < 1e-6, (
        f"E_jk should be ~0 when M_j always zero, got max={E_zero.max().item()}"
    )

    # --- sub-case c: F=1, M always 1, a uniform positive → E_jk = 1 ---
    F_ones = torch.ones(N)
    a_uniform = torch.ones(N, d_in) * 2.0
    M_ones = torch.ones(N, d_out, dtype=torch.long)

    E_ones = _compute_familiarity_exposure(F_ones, a_uniform, M_ones)

    assert torch.allclose(E_ones, torch.ones(d_out, d_in), atol=1e-6), (
        f"E_jk should be 1.0 when F=1, M=1 always, got max deviation "
        f"{(E_ones - 1.0).abs().max().item()}"
    )

    print("PASS  Test 1: E_jk bounded in [0, 1] for all sub-cases.")
    print(f"       Random case  — min={E.min().item():.6f}, max={E.max().item():.6f}")
    print(f"       Zero-M case  — max={E_zero.max().item():.2e}  (expected ~0)")
    print(f"       Perfect case — max deviation from 1.0: {(E_ones - 1.0).abs().max().item():.2e}")


# ===========================================================================
# TEST 2 — M_j binary activity indicator is computed correctly
# ===========================================================================

def test_activity_indicator_is_exactly_correct_for_controlled_post_relu_inputs() -> None:
    """M_j[i,j] must be 1 iff post_relu[i,j] > 0.

    A controlled post-ReLU tensor is constructed with known sign patterns.
    The test verifies:
    a) M is binary (contains only 0 and 1).
    b) Every positive activation maps to M=1.
    c) Every zero activation maps to M=0.
    d) No non-zero positive value maps to M=0.
    e) No zero value maps to M=1.

    The ReLU property guarantees no negative post-ReLU values, so the sign
    boundary is exactly at zero.
    """
    # Construct a (5, 6) post-ReLU tensor with determistic sign pattern
    post_relu = torch.tensor(
        [
            [0.0,  0.5,  1.2,  0.0,  3.7,  0.0],   # row 0: some zeros, some positive
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0],   # row 1: all zeros
            [0.1,  0.2,  0.3,  0.4,  0.5,  0.6],   # row 2: all positive
            [0.0,  0.0,  1.0,  0.0,  0.0,  2.5],   # row 3: sparse
            [9.9,  0.0,  0.0,  0.0,  0.0,  0.001], # row 4: first and last active
        ],
        dtype=torch.float32,
    )

    M = _compute_activity_indicator(post_relu)

    # --- a: binary ---
    unique_vals = M.unique().tolist()
    assert set(unique_vals).issubset({0, 1}), (
        f"M_j is not binary — found values: {unique_vals}"
    )

    # --- b: every positive activation → M=1 ---
    positive_mask = post_relu > 0
    assert (M[positive_mask] == 1).all(), (
        "Some positive post-ReLU activations have M_j=0"
    )

    # --- c: every zero activation → M=0 ---
    zero_mask = post_relu == 0
    assert (M[zero_mask] == 0).all(), (
        "Some zero post-ReLU activations have M_j=1"
    )

    # --- d: no non-zero positive maps to M=0 (same as b, explicit cross-check) ---
    false_negatives = ((post_relu > 0) & (M == 0)).sum().item()
    assert false_negatives == 0, f"False negatives in M_j: {false_negatives}"

    # --- e: no zero maps to M=1 ---
    false_positives = ((post_relu == 0) & (M == 1)).sum().item()
    assert false_positives == 0, f"False positives in M_j: {false_positives}"

    # Verify exact expected pattern row by row
    expected = torch.tensor(
        [
            [0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
        ],
        dtype=torch.long,
    )
    assert torch.equal(M, expected), (
        f"M_j does not match expected pattern.\nGot:\n{M}\nExpected:\n{expected}"
    )

    print("PASS  Test 2: M_j binary activity indicator is exactly correct.")
    print(f"       Tensor shape: {tuple(post_relu.shape)}")
    print(f"       Active cells: {M.sum().item()} / {M.numel()}")
    print(f"       False positives: 0, False negatives: 0")


# ===========================================================================
# TEST 3 — Global magnitude pruning is computed across both matrices jointly
# ===========================================================================

def test_global_magnitude_pruning_is_across_both_matrices_jointly_not_per_layer() -> None:
    """The 50% pruning threshold must be computed over the combined magnitude
    distribution of W_semantic_1 and W_semantic_2.

    The adversarial configuration used here is:
    - W1: all 1024 weights = 0.1 (uniformly small)
    - W2: 512 weights = 0.3, 512 weights = 0.7 (split)

    Sorted magnitudes across 2048 weights:
      [0.1 × 1024, 0.3 × 512, 0.7 × 512]
    The 1024th smallest magnitude is 0.1 — the global threshold.
    Global mask uses |w| > 0.1:
      - W1: all 1024 weights equal 0.1, so |w| > 0.1 is False → all pruned
      - W2: all 1024 weights are 0.3 or 0.7, both > 0.1 → all survive

    Per-layer 50% pruning of W2 would prune the 512 weights at 0.3 (threshold=0.3)
    and keep the 512 weights at 0.7 → 512 W2 weights pruned.
    Global pruning of W2 prunes zero W2 weights.
    The two procedures produce genuinely different masks for W2.

    A second case verifies exact 50% global sparsity on random inputs.
    """
    d = 32

    # W1: all weights = 0.1
    W1 = torch.full((d, d), 0.1)

    # W2: first half of rows = 0.3, second half = 0.7
    W2 = torch.zeros(d, d)
    W2[:d // 2, :] = 0.3
    W2[d // 2:, :] = 0.7

    # Global pruning: sorted order is [0.1×1024, 0.3×512, 0.7×512]
    # The 1024th smallest (index 1023) = 0.1 → threshold = 0.1
    # Mask uses |w| > 0.1
    mask1, mask2, threshold = _global_magnitude_prune(W1, W2, sparsity=0.5)

    assert abs(threshold - 0.1) < 1e-6, (
        f"Global threshold should be 0.1, got {threshold:.6f}"
    )

    # Global: all 1024 W1 weights are pruned (all equal 0.1, not strictly > 0.1)
    assert (mask1 == 0).all(), (
        f"Adversarial case: W1 should be entirely pruned globally, but mask1 has "
        f"{(mask1 == 1).sum().item()} survivors"
    )
    # Global: all 1024 W2 weights survive (both 0.3 and 0.7 strictly > 0.1)
    assert (mask2 == 1).all(), (
        f"Adversarial case: W2 should be entirely surviving globally, but mask2 has "
        f"{(mask2 == 0).sum().item()} pruned weights"
    )

    # Per-layer 50% pruning of W2: threshold = 0.3 (median of {0.3×512, 0.7×512})
    # Per-layer mask2 prunes the 0.3 tier (512 weights) and keeps the 0.7 tier.
    per_layer_threshold_w2 = float(torch.quantile(W2.abs().flatten(), 0.5).item())
    per_layer_mask2 = (W2.abs() > per_layer_threshold_w2).long()
    per_layer_survivors_w2 = per_layer_mask2.sum().item()

    # Global says 1024 survivors in W2; per-layer says ~512 survivors in W2
    assert per_layer_survivors_w2 < mask2.sum().item(), (
        f"Per-layer W2 mask should have fewer survivors than global "
        f"(per-layer: {per_layer_survivors_w2}, global: {mask2.sum().item()})"
    )
    assert not torch.equal(mask2, per_layer_mask2), (
        f"Global mask2 and per-layer mask2 are identical — "
        f"the adversarial case does not distinguish the two procedures. "
        f"Global survivors: {mask2.sum().item()}, per-layer survivors: {per_layer_survivors_w2}"
    )

    # --- standard case: verify exactly 50% global sparsity on random inputs ---
    torch.manual_seed(42)
    W1_rand = torch.randn(d, d)
    W2_rand = torch.randn(d, d)
    mask1_rand, mask2_rand, threshold_rand = _global_magnitude_prune(W1_rand, W2_rand, sparsity=0.5)

    n_survived_rand = mask1_rand.sum().item() + mask2_rand.sum().item()
    n_pruned_rand = (d * d * 2) - n_survived_rand

    # Allow off-by-one due to ties at the threshold boundary
    assert abs(n_pruned_rand - 1024) <= 1, (
        f"Expected ~1024 weights pruned globally, got {n_pruned_rand}"
    )

    # Every pruned weight must have |w| <= threshold
    for w, mask in [(W1_rand, mask1_rand), (W2_rand, mask2_rand)]:
        pruned_mags = w[mask == 0].abs()
        if pruned_mags.numel() > 0:
            assert float(pruned_mags.max().item()) <= threshold_rand + 1e-6, (
                f"A pruned weight has magnitude above threshold: "
                f"{pruned_mags.max().item():.6f} > {threshold_rand:.6f}"
            )

    # Every surviving weight must have |w| >= threshold
    for w, mask in [(W1_rand, mask1_rand), (W2_rand, mask2_rand)]:
        survived_mags = w[mask == 1].abs()
        if survived_mags.numel() > 0:
            assert float(survived_mags.min().item()) >= threshold_rand - 1e-6, (
                f"A surviving weight has magnitude below threshold: "
                f"{survived_mags.min().item():.6f} < {threshold_rand:.6f}"
            )

    print("PASS  Test 3: Global magnitude pruning is correctly across both matrices jointly.")
    print(f"       Adversarial case  — W1 pruned: {(mask1==0).sum().item()}/1024 (100%), "
          f"W2 pruned: {(mask2==0).sum().item()}/1024 (0%)")
    print(f"       Per-layer contrast — W2 per-layer survivors: {per_layer_survivors_w2}/1024, "
          f"global survivors: {mask2.sum().item()}/1024 — different masks confirmed")
    print(f"       Random case       — global pruned: {n_pruned_rand}/2048 "
          f"(target 1024, off by {abs(n_pruned_rand - 1024)})")
    print(f"       Global threshold  — {threshold_rand:.6f}")


# ===========================================================================
# TEST 4 — Rewound weights exactly match step-0 initialisation checkpoint
# ===========================================================================

def test_rewound_weights_exactly_match_step_zero_initialisation_checkpoint() -> None:
    """After several training steps, rewinding must restore surviving weights
    to their exact step-0 values, and pruned positions must be exactly 0.

    Procedure:
    1. Instantiate TinySemanticModel with a fixed seed.
    2. Save W_semantic_1 and W_semantic_2 to init_weights.pt.
    3. Run 5 gradient steps — weights change.
    4. Apply one-shot global 50% pruning mask based on post-training magnitudes.
    5. Rewind: surviving positions ← init_weights.pt; pruned positions ← 0.
    6. Apply mask after each subsequent step to enforce sparsity.
    7. Assert: for every surviving position, rewound value == init_weights.pt value
       (absolute tolerance 1e-7).
    8. Assert: for every pruned position, value == 0 exactly.
    """
    torch.manual_seed(7)
    model = TinySemanticModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Step 2: save initialisation checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        init_path = Path(tmpdir) / "init_weights.pt"
        init_checkpoint = {
            "w_semantic_1": model.w_semantic_1.weight.detach().clone(),
            "w_semantic_2": model.w_semantic_2.weight.detach().clone(),
        }
        torch.save(init_checkpoint, init_path)

        # Verify the file was written
        assert init_path.exists(), "init_weights.pt was not created"

        # Step 3: run 5 gradient steps to move weights away from initialisation
        for step in range(5):
            x = torch.randn(64, 32)
            labels = torch.randint(0, 4, (64,))
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        # Confirm weights changed from initialisation
        loaded_init = torch.load(init_path, weights_only=True)
        assert not torch.equal(
            model.w_semantic_1.weight.detach(),
            loaded_init["w_semantic_1"],
        ), "W_semantic_1 did not change after 5 training steps"
        assert not torch.equal(
            model.w_semantic_2.weight.detach(),
            loaded_init["w_semantic_2"],
        ), "W_semantic_2 did not change after 5 training steps"

        # Step 4: compute global pruning mask from post-training weights
        mask1, mask2, threshold = _global_magnitude_prune(
            model.w_semantic_1.weight.detach(),
            model.w_semantic_2.weight.detach(),
            sparsity=0.5,
        )

        # Step 5: rewind surviving weights to initialisation values
        with torch.no_grad():
            # Layer 1
            w1 = model.w_semantic_1.weight
            w1.copy_(loaded_init["w_semantic_1"] * mask1.float())
            # Layer 2
            w2 = model.w_semantic_2.weight
            w2.copy_(loaded_init["w_semantic_2"] * mask2.float())

        # Step 6 (verification pre-retraining): apply mask to zero out pruned positions
        with torch.no_grad():
            model.w_semantic_1.weight.mul_(mask1.float())
            model.w_semantic_2.weight.mul_(mask2.float())

        # Step 7: assert surviving positions match init_weights.pt exactly
        rewound_w1 = model.w_semantic_1.weight.detach()
        rewound_w2 = model.w_semantic_2.weight.detach()

        surviving_w1 = mask1 == 1
        surviving_w2 = mask2 == 1

        # Surviving positions must match init_weights.pt to within 1e-7
        max_dev_w1 = (
            (rewound_w1[surviving_w1] - loaded_init["w_semantic_1"][surviving_w1]).abs().max().item()
            if surviving_w1.any() else 0.0
        )
        max_dev_w2 = (
            (rewound_w2[surviving_w2] - loaded_init["w_semantic_2"][surviving_w2]).abs().max().item()
            if surviving_w2.any() else 0.0
        )

        assert max_dev_w1 < 1e-7, (
            f"W_semantic_1 surviving weights deviate from init_weights.pt: "
            f"max deviation = {max_dev_w1:.2e}"
        )
        assert max_dev_w2 < 1e-7, (
            f"W_semantic_2 surviving weights deviate from init_weights.pt: "
            f"max deviation = {max_dev_w2:.2e}"
        )

        # Step 8: assert pruned positions are exactly 0
        pruned_w1 = mask1 == 0
        pruned_w2 = mask2 == 0

        if pruned_w1.any():
            assert (rewound_w1[pruned_w1] == 0.0).all(), (
                f"W_semantic_1 pruned positions are not exactly 0: "
                f"max={rewound_w1[pruned_w1].abs().max().item():.2e}"
            )
        if pruned_w2.any():
            assert (rewound_w2[pruned_w2] == 0.0).all(), (
                f"W_semantic_2 pruned positions are not exactly 0: "
                f"max={rewound_w2[pruned_w2].abs().max().item():.2e}"
            )

        # Count survivors and pruned for reporting
        n_survived_w1 = surviving_w1.sum().item()
        n_pruned_w1 = pruned_w1.sum().item()
        n_survived_w2 = surviving_w2.sum().item()
        n_pruned_w2 = pruned_w2.sum().item()

    print("PASS  Test 4: Rewound weights exactly match step-0 initialisation checkpoint.")
    print(f"       W_semantic_1 — survivors: {n_survived_w1}/1024, pruned: {n_pruned_w1}/1024")
    print(f"       W_semantic_2 — survivors: {n_survived_w2}/1024, pruned: {n_pruned_w2}/1024")
    print(f"       Max deviation from init_weights.pt (W1 survivors): {max_dev_w1:.2e}")
    print(f"       Max deviation from init_weights.pt (W2 survivors): {max_dev_w2:.2e}")
    print(f"       Pruned positions are exactly 0 for both matrices.")


# ===========================================================================
# Runner (also invocable via pytest)
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("E25 Pre-Training Invariant Tests")
    print("=" * 70)
    print()

    tests = [
        test_familiarity_exposure_scores_are_bounded_between_zero_and_one,
        test_activity_indicator_is_exactly_correct_for_controlled_post_relu_inputs,
        test_global_magnitude_pruning_is_across_both_matrices_jointly_not_per_layer,
        test_rewound_weights_exactly_match_step_zero_initialisation_checkpoint,
    ]

    passed = 0
    failed = 0
    for fn in tests:
        print(f"Running: {fn.__name__}")
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
            failed += 1
        print()

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("ALL INVARIANT TESTS PASS — training code may proceed.")
    else:
        print("INVARIANT TESTS FAILED — do not proceed to training code.")
    print("=" * 70)
