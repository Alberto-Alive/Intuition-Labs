# E30 Diagnosis Note

This follow-up is meant to answer four questions directly:

1. Did removing the reader preserve witness diversity?
2. Did witness weighting become meaningfully selective?
3. Did the signed margin channel recover positive values?
4. Is uncertainty still aligned with errors?
5. Do true failure examples die in the margin path or later in gating/class pressure?

## Where to look

- `pre_reader_witness_geometry.pairwise_cosine_agreement` and `pre_reader_witness_geometry.witness_latent_variance`
  - raw diffusion diversity before the point reader
- `post_reader_witness_geometry.pairwise_cosine_agreement`
  - diversity after the point reader
- `diversity_retention.reader_agreement_gain`
  - direct collapse signal from reader compression
- `margin_channel_diagnostics.margin_positive_fraction`
  - whether the global margin channel actually produces positive values
- `margin_channel_diagnostics.witness_positive_count` and `margin_channel_diagnostics.witness_negative_count`
  - whether witnesses themselves contain mixed-sign evidence
- `margin_channel_diagnostics.witness_sign_disagreement_rate`
  - whether the witness set itself contains sign disagreement, not just a one-sided margin
- `failure_channel_diagnostics.true_failure_margin_negative_fraction`
  - whether true failures already have negative margin before gating
- `failure_channel_diagnostics.true_failure_predicted_failure_fraction`
  - whether true failures survive into the FAILURE_LIKELY bucket
- `failure_channel_diagnostics.true_failure_predicted_uncertain_fraction`
  - whether failures are being diverted into UNCERTAIN instead of FAILURE_LIKELY
- `margin_channel_diagnostics.weighted_margin_sign_flip_rate`
  - whether weighting changes the sign of the global margin
- `post_reader_witness_geometry.witness_weight_entropy`
  - near `1.0` means weights are close to uniform
- `post_reader_witness_geometry.witness_weight_kl_uniform`
  - larger values mean more selective weighting
- `weighting_diagnostics.coherence_score_spread_before_softmax`
  - whether coherence scores actually separate witnesses before softmax
- `weighting_diagnostics.coherence_margin_corr`
  - whether coherence and raw witness margin move together
- `weighting_diagnostics.coherence_abs_margin_corr`
  - whether coherence relates to magnitude even when the sign is ignored
- `weighting_diagnostics.weighted_minus_unweighted_margin`
  - whether weighting changes the global margin at all
- `decision_geometry.commitment_score`
  - commitment by true class, predicted class, and correctness
- `test_metrics.uncertainty_error_corr`
  - whether uncertainty rises on errors or not

## Interpreting the run

- If `post_reader_witness_geometry.pairwise_cosine_agreement` is much larger than `pre_reader_witness_geometry.pairwise_cosine_agreement`, the reader is still collapsing witnesses.
- If `witness_weight_entropy` stays near `1.0` and `witness_weight_kl_uniform` stays near `0.0`, weighting is still effectively uniform.
- If `failure_recall` improves while `class_shares["FAILURE_LIKELY"]` does not dominate, the gate is behaving more sensibly.
- If `margin_channel_diagnostics.margin_positive_fraction` is no longer near zero, the signed margin channel has recovered.
- If `weighted_minus_unweighted_margin` and `weighted_margin_sign_flip_rate` remain near zero, coherence weighting still has little leverage.
- If `uncertainty_error_corr` is positive and `decision_geometry.uncertainty_U` is higher on incorrect examples, uncertainty is aligned with errors.
- If true failures mostly have `M > 0`, the problem is the margin path.
- If true failures mostly have `M < 0` or mixed-sign witnesses but still do not predict `FAILURE_LIKELY`, the problem is downstream in gating, thresholds, or class pressure.
- If `linear_scalar_margin` raises the true-failure negative-margin rate without restoring collapse, it is the better reader mode.

## Main Experiment

- `point_reader_mode = identity`
- fallback if needed: `point_reader_mode = linear_scalar_margin`
- `gate_mode = soft_commitment_gate`
- `use_hard_eval_gate = true`
- `primary_metric = balanced_margin_primary`

## Example Commands

```bash
python3 experiments/DIGIT/Extrapolation/e30/scripts/run_experiment.py \
  --seed 123 \
  --point-reader-mode identity \
  --primary-metric balanced_margin_primary
```

```bash
python3 experiments/DIGIT/Extrapolation/e30/scripts/run_experiment.py \
  --seed 123 \
  --point-reader-mode linear_scalar_margin \
  --primary-metric balanced_margin_primary
```
