# E31 Diagnosis Note

This follow-up keeps the `e30` witness diagnostics and adds v3-style latent path
stability diagnostics. It is meant to answer these questions directly:

1. Did removing the reader preserve witness diversity?
2. Did witness weighting become meaningfully selective?
3. Did the signed margin channel recover positive values?
4. Is uncertainty still aligned with errors?
5. Do true failure examples die in the margin path or later in gating/class pressure?
6. Are independent diffusion paths stable for the same query?
7. Does path instability catch errors missed by endpoint witness geometry?

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
- `path_uncertainty_diagnostics.path_disagreement`
  - whether two independent diffusion trajectories preserve the same compact path meaning
- `path_uncertainty_diagnostics.method_conflict`
  - whether endpoint confidence and path confidence disagree
- `test_metrics.path_disagreement_error_corr`
  - whether path instability rises on errors
- `test_metrics.endpoint_path_conflict_error_corr`
  - whether endpoint/path conflict rises on errors

## Interpreting the run

- If `post_reader_witness_geometry.pairwise_cosine_agreement` is much larger than `pre_reader_witness_geometry.pairwise_cosine_agreement`, the reader is still collapsing witnesses.
- If `witness_weight_entropy` stays near `1.0` and `witness_weight_kl_uniform` stays near `0.0`, weighting is still effectively uniform.
- If `failure_recall` improves while `class_shares["FAILURE_LIKELY"]` does not dominate, the gate is behaving more sensibly.
- If `margin_channel_diagnostics.margin_positive_fraction` is no longer near zero, the signed margin channel has recovered.
- If `weighted_minus_unweighted_margin` and `weighted_margin_sign_flip_rate` remain near zero, coherence weighting still has little leverage.
- If `uncertainty_error_corr` is positive and `decision_geometry.uncertainty_U` is higher on incorrect examples, uncertainty is aligned with errors.
- If `path_disagreement_error_corr` is more positive than `uncertainty_error_corr`, the path channel is adding useful uncertainty signal.
- If `method_conflict` is high on incorrect confident examples, the cooperative path/witness check is catching hidden brittleness.
- If true failures mostly have `M > 0`, the problem is the margin path.
- If true failures mostly have `M < 0` or mixed-sign witnesses but still do not predict `FAILURE_LIKELY`, the problem is downstream in gating, thresholds, or class pressure.
- If `linear_scalar_margin` raises the true-failure negative-margin rate without restoring collapse, it is the better reader mode.

## Main Experiment

- `point_reader_mode = identity`
- fallback if needed: `point_reader_mode = linear_scalar_margin`
- `gate_mode = soft_commitment_gate`
- `use_hard_eval_gate = true`
- `primary_metric = balanced_margin_primary`
- `enable_path_uncertainty = true`
- first pass: leave cooperative path flags off to collect diagnostics
- active cooperative ablation: add `--enable-cooperative-path-uncertainty`
- full coupling ablation: add `--enable-path-aware-witness-weights`

## Example Commands

```bash
python3 experiments/DIGIT/Extrapolation/e31/scripts/run_experiment.py \
  --seed 123 \
  --device cuda \
  --point-reader-mode identity \
  --primary-metric balanced_margin_primary
```

```bash
python3 experiments/DIGIT/Extrapolation/e31/scripts/run_experiment.py \
  --seed 123 \
  --device cuda \
  --point-reader-mode identity \
  --primary-metric balanced_margin_primary \
  --enable-cooperative-path-uncertainty
```

```bash
python3 experiments/DIGIT/Extrapolation/e31/scripts/run_experiment.py \
  --seed 123 \
  --device cuda \
  --point-reader-mode linear_scalar_margin \
  --primary-metric balanced_margin_primary \
  --enable-cooperative-path-uncertainty \
  --enable-path-aware-witness-weights
```
