# Activation-Aware Coordination Experiments

This repository is a reproducible harness for testing whether internal activation
telemetry from cloned agents improves multi-agent coordination over visible
outputs alone.

The first implemented benchmark is deliberately simple and inspectable:

- deterministic synthetic arithmetic labels
- clean train/dev/test split
- cloned simulated agents that produce candidate answers, confidence, rationale
  text, and simulated hidden states
- learned and non-learned coordinators
- activation anti-cheat controls
- multi-seed reporting

The design goal is empirical discipline, not a demo. Coordinators never receive
ground truth, evaluator results, or test labels during prediction. Learned
coordinators train on train labels, use dev for early stopping/model selection,
and the runner evaluates the final test split only after all methods for a seed
are fit.

## Repository Layout

```text
/README.md
/reports/
/configs/
/src/datasets/
/src/agents/
/src/telemetry/
/src/coordinators/
/src/evaluation/
/src/experiments/
/tests/
/results/
```

## Quick Start

Run tests:

```powershell
python -m pytest
```

Run the synthetic experiment on CUDA when available:

```powershell
python -m src.experiments.run_synthetic --config configs/synthetic_cuda.json
```

Outputs:

- `results/synthetic_cuda_results.json`
- `reports/REPORT.md`

## Implemented Methods

Baselines:

- `single_agent`: first cloned agent answer.
- `independent_swarm`: chooses the candidate with highest visible confidence.
- `text_only_coordinator`: MLP over visible answers, confidence, votes, and
  task id.
- `capacity_matched_text_only`: text-only coordinator with a larger effective
  feature expansion and comparable capacity to activation MLPs.

Activation-aware variants:

- `activation_pool_mlp`: visible features plus pooled hidden states.
- `activation_pca_mlp`: visible features plus train-fitted PCA compression of
  hidden states.
- `activation_cluster_router`: clusters train activations and routes to the
  candidate whose activation cluster historically predicts correctness.

Diagnostics:

- `telemetry_only_label_probe`: predicts final labels from activations only.
- `output_only_oracle_probe`: predicts final labels from all visible outputs,
  confidence fields, and parseable traces with a parameter budget matched to
  `activation_pca_mlp`.
- `agent_correctness_probe`: predicts each agent's correctness from text-only,
  activation-only, and combined features; reports accuracy and AUC.
- `redundancy_probe`: clusters activations and reports purity by visible answer
  label and by agent correctness.
- `stage0_label_reduced`: reruns Stage 0 after removing task/label identity
  channels from simulated hidden states while preserving uncertainty, agent
  quality, reasoning-mode, and stuckness signals.
- `strict_coordination`: agents receive private partial evidence; visible
  outputs do not encode the private evidence, and hidden states reveal evidence
  state rather than the final label.

Controls:

- shuffled activations
- random activations
- wrong-task activations
- wrong-agent activations
- randomized-label leakage test

## CUDA / 16GB Budget

The default config requests `cuda` and sets a per-process CUDA memory fraction
based on a 16GB budget. If CUDA is not available, the runner falls back to CPU
and records that in the result metadata.

## Open-Weight Transformer Support

`src/agents/transformer.py` includes optional support for capturing hidden
states from a local Hugging Face causal language model through forward hooks.
The synthetic experiment does not require network access or model downloads.
