# E10 Audit Note

Source artifacts:
- `experiments/DIGIT/Extrapolation/results/e10/fresh_seed_123/summary.json`
- `experiments/DIGIT/Extrapolation/results/e10/fresh_seed_123/metrics.json`
- `experiments/DIGIT/Extrapolation/results/e10/fresh_seed_123/evidence_interventions/summary.json`
- `experiments/DIGIT/Extrapolation/results/e10/fresh_seed_123/head_ablation/summary.json`
- `experiments/DIGIT/Extrapolation/results/e10/fresh_seed_123/path_count_ablation/summary.json`

## 1. Executive summary

`e10` now has a real causal evidence core, but it does not yet validate a clean uncertainty-native architecture. The audited stage1 checkpoint is non-collapsed, performs reasonably well on test, and responds to direct evidence interventions in the expected direction. The strongest result is that changing `boundary`, `support`, and `order` changes the model's public outputs systematically, not just its latent scores.

What it has not demonstrated is full faithfulness. The simplest evidence-only readouts fail badly, the best flexible readout is still an MLP, and one key chain is leaky: `order` strongly moves outcome but barely moves `commitment_depth`. So the evidence state is real, but the final architecture is still partly propped up by flexible heads and imperfect monotonicity. This audit is also effectively stage1-only (`trace_force_stage1_only=true`; stage2 was not active).

## 2. What is clearly established

The audited run generalizes decently without collapsing. Best validation was `outcome_acc=0.9182`, `outcome_macro_f1=0.8893`, `failure_recall=1.0`; test was `outcome_acc=0.8774`, `outcome_macro_f1=0.8202`, `failure_recall=0.8`. Safety is strong in the narrow sense: `unsafe_success_rate_correctness=0.0` and `success_precision_vs_is_correct=1.0`. The collapse audit is clean: `collapsed_model=false`, `prediction_collapsed=false`, `structure_collapsed=false`, and all three prototype families are active. The model is uncertainty-heavy rather than reckless: on test it predicts `UNCERTAIN` 72.0 percent of the time, `SUCCESS_LIKELY` 20.8 percent, and `FAILURE_LIKELY` 7.2 percent.

The evidence-intervention audit is the main causal result. `boundary_up` has sign consistency `0.789`, raises `ambiguity_cert` by `0.195`, lowers `success_cert` by `0.118`, lowers `commitment_depth` by `0.102`, and flips the predicted class on `35.2 percent` of examples. `support_down` is almost perfectly consistent (`0.9996`) and lowers `support_cert` by `0.027`, raises `ambiguity_cert` by `0.150`, lowers `success_cert` by `0.118`, and lowers `commitment_depth` by `0.088`. `order_failure` is also highly consistent (`0.993`) and moves `success_base` down by `0.123`, `failure_cert` up by `0.123`, and `success_cert` down by `0.089`. So `order`, `support`, and `boundary` are causal in the model, not decorative.

The path ablation shows that diffusion multiplicity helps, but only modestly. Multi-path beats single-path stochastic on `outcome_acc` (`0.8868` vs `0.8648`), `outcome_macro_f1` (`0.8381` vs `0.8119`), threshold swing (`0.0566` vs `0.0786`), and seed variance (`mean_success_prob_std=0.0216` vs `0.0439`). Deterministic single-path removes the variance entirely, but it does not recover multi-path performance. The clean reading is that multiplicity is a stabilizer, not the main engine.

The head ablation shows that simple evidence-only heads are not sufficient. The flexible MLP is the only viable mode: `outcome_acc=0.8711`, `outcome_macro_f1=0.8157`, `failure_recall=0.8`, `unsafe_success_rate_correctness=0.0`, `success_precision_vs_is_correct=1.0`, and the threshold audit stays stable (`max_outcome_share_swing=0.0346`, `label_revision_trigger=false`). But `linear_evidence_only` collapses to `outcome_acc=0.6981`, `outcome_macro_f1=0.2741`, `failure_recall=0.0`, and `ordered_threshold` collapses further to `outcome_acc=0.2201`, `outcome_macro_f1=0.1685`, `failure_recall=0.0`, with unsafe success rate `0.1144` and huge threshold swing `0.7767`. That is strong evidence that the shared evidence state alone is not enough unless the readout remains flexible.

## 3. What is still weak or unresolved

The main weakness is faithfulness, not raw accuracy. The audited run still has `commitment_monotonicity_violation_rate=0.0447`, and its run-level label audit still shows `max_outcome_share_swing=0.2083` with `label_revision_trigger=true`, so threshold robustness is not fully settled. More importantly, the head ablation shows that even the best flexible readout does not give a clean monotone system: after head-only finetuning, the MLP mode still has `monotonicity_violation_rate=0.1693`, `support_cert_violation_rate=0.3560`, `success_guard_violation_rate=0.2263`, and `success_prob_monotonicity_violation_rate=0.1308`. That is too leaky for a fully faithful uncertainty architecture.

The other unresolved issue is that `order` affects outcome much more than commitment. In the intervention audit, `order_failure` moved outcome strongly (`success_base -0.123`, `failure_cert +0.123`, `success_cert -0.089`), but `commitment_depth` barely moved (`-0.0018`). That means the model's public outcome path is causally real, but the claimed unified evidence-to-commitment chain is still incomplete. In plain terms: the evidence state controls the answer, but not all of the downstream confidence behavior equally well.

## 4. Best interpretation of the architecture now

`e10` is not architecture theater. The evidence interventions show a real causal core: the internal `order`, `boundary`, and `support` variables actually move the public outputs in the expected direction. But it is also not a fully validated uncertainty-native architecture. The final readout still needs a flexible MLP to work, monotonicity is imperfect, and the `order -> commitment` link is too weak. The best interpretation is: real causal evidence layer, leaky and partly flexible readout layer. That is materially better than theater, but still short of the architecture's strongest story.

## 5. Layman explanation

The model really does react when you change its internal confidence signals, so those signals are not fake. But the last step that turns those signals into a final answer is still too flexible, so the system is promising but not yet fully trustworthy as a clean confidence machine.

## 6. Implication for next version

`e29` should tighten the readout, not add more capacity. The next version should force outcome, confidence, and commitment to come from constrained monotone mappings of the shared evidence state, then re-run the same intervention tests until `order`, `boundary`, and `support` move both outcome and commitment together. Keep multi-path diffusion if it helps robustness, but treat it as a stabilizer, not the main claim. The main job is to close the `order -> commitment` leak and remove the need for flexible shortcut readouts.

## 7. One-sentence takeaway

`e10` has a real causal evidence core, but it still depends on flexible readouts and only partially realizes the clean uncertainty-native architecture it claims.
