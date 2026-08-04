# E30 Architecture Note

`e30` is a targeted witness-agreement correction experiment.

The goal is to preserve raw witness diversity, avoid reader collapse, and make the commitment gate behave like an actual gate without overproducing `FAILURE_LIKELY`.

## Decision Path

- The encoder emits one anchor latent `a` per example.
- A conditional latent diffusion model samples `K` raw witness latents `z_k` around `a`.
- Pre-reader witness diagnostics are computed directly on the raw samples:
  - pairwise cosine agreement
  - pairwise cosine agreement standard deviation
  - mean squared distance to anchor
  - witness latent variance
  - witness latent norm mean and standard deviation
  - witness latent spread
  - principal singular value ratio
- The witness latents then pass through a configurable point reader.
- Post-reader witness diagnostics are computed on the point-reader outputs:
  - pairwise cosine agreement
  - local witness margin mean and standard deviation
  - coherence score mean and standard deviation
  - witness weight entropy
  - witness weight max
  - witness weight KL divergence vs uniform
  - witness margin range
  - weighted minus unweighted margin

## Point Reader Family

`point_reader_mode` selects one of three explicit readers:

- `identity`: no learned point reader. Raw witness latents are passed through unchanged.
- `linear_scalar_margin`: one affine projection per witness into a scalar margin.
- `linear_shared_proj`: one affine projection per witness into the latent space, with no residual stack.

Default:

- `point_reader_mode = identity`

This keeps the main run maximally conservative about reader-induced collapse. If the downstream tensor path needs scalar margins immediately, `linear_scalar_margin` is the narrow fallback.

## Readout

The public geometry uses only witness-derived quantities:

- local witness margins `m_k`
- weighted global margin `M = sum_k omega_k * m_k`
- vote disagreement `D = sum_k omega_k * (m_k - M)^2`
- geometric disagreement `G = 1 - mean pairwise cosine agreement`
- uncertainty `U = softplus(alpha * D + beta * G - gamma * abs(M) + b_u)`
- commitment score `C = abs(M) - lambda_commit * U`

The commitment gate is controlled by `gate_mode`.

### Soft Commitment Gate

`gate_mode = soft_commitment_gate`

The differentiable gate uses a shared commitment term:

```text
K = commitment_scale * (C - tau_uncertain)
uncertain = softplus(-K)
success = K + M - softplus(-K)
failure = K - M - softplus(-K)
```

This makes `UNCERTAIN` dominate more cleanly when commitment is weak.

### Legacy Logits

`gate_mode = legacy_logits`

```text
success = M - U
uncertain = U
failure = -M - U
```

### Hard Evaluation Gate

`use_hard_eval_gate = true` applies a discrete evaluation-time ablation:

```text
if C < tau_uncertain then UNCERTAIN else sign(M)
```

Training remains differentiable. Hard gating is only for reporting and selection.

## Selection Metric

`primary_metric` can be `macro_f1`, `failure_recall`, `balanced_commitment_primary`, or `balanced_margin_primary`.

`balanced_margin_primary` is:

```text
balanced_margin_primary = outcome_macro_f1
                          + 0.25 * failure_recall
                          + 0.15 * success_recall
                          - max(0.0, 0.08 - success_share)
                          - max(0.0, 0.08 - failure_share)
                          - max(0.0, uncertain_share - 0.70)
                          - 0.5 * max(0.0, 0.15 - uncertain_share)
                          - 0.5 * max(0.0, dominant_class_share - 0.75)
                          - 0.5 * max(0.0, 0.10 - margin_positive_fraction)
```

This rewards both failure recall and success recall, while penalizing zero-success regimes, dominant-class collapse, extreme uncertain collapse, and all-negative-margin behavior.

`balanced_commitment_primary` is retained for compatibility, but the main experiment now uses `balanced_margin_primary`.

## Report Sections

The experiment writes the following high-value sections into `metrics.json` and `summary.json`:

- `pre_reader_witness_geometry`
- `post_reader_witness_geometry`
- `diversity_retention`
- `decision_geometry`
- `margin_channel_diagnostics`
- `failure_channel_diagnostics`
- `success_channel_diagnostics`
- `weighting_diagnostics`
- `report_highlights`

How to read them:

- `pre_reader_witness_geometry` answers whether diffusion already collapsed the witnesses.
- `post_reader_witness_geometry` answers whether the point reader is collapsing them further.
- `diversity_retention` compares pre vs post directly with reader agreement gain and spread ratio.
- `decision_geometry` exposes `M`, `D`, `G`, `U`, commitment score, and the gate terms.
- `margin_channel_diagnostics` exposes signed-margin support, witness sign counts, and whether weighting flips the margin sign.
- `failure_channel_diagnostics` and `success_channel_diagnostics` expose true-class slices so you can tell whether failures are dying in the margin path or later in gating / class pressure.
- `weighting_diagnostics` shows whether coherence scores have any leverage over witness weights.
- `report_highlights` provides a compact readout of the main answers for quick inspection.

Interpretation rule for the failure pass:

- If true failure examples mostly have `M > 0`, the problem is the margin path.
- If true failure examples mostly have `M < 0` or mixed-sign witnesses but still do not predict `FAILURE_LIKELY`, the problem is downstream in gating, thresholding, or class pressure.
- If `linear_scalar_margin` improves the true-failure negative-margin rate without reintroducing reader collapse, it is the better reader mode.

## Constraints

- No semantic evidence heads in the core path.
- No mixed-feature rescue MLPs.
- No executor-feature shortcuts.
- No hidden compression tower in the default reader path.
