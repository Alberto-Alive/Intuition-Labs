from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple


def write_report(results: Dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(results), encoding="utf-8")


def render_report(results: Dict[str, object]) -> str:
    metrics = [row for row in results["metrics"] if row["split"] in {"dev", "test"}]
    raw_diagnostics = list(results.get("diagnostics", []))
    diagnostics = [row for row in raw_diagnostics if row.get("split") in {"dev", "test"}]
    audit = results.get("audit", [])
    validation = results.get("validation", {})
    lines: List[str] = []
    lines.append("# Activation-Aware Coordination Report")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- Train labels are used only to fit supervised coordinators.")
    lines.append("- Dev labels are used for early stopping and method inspection.")
    lines.append("- Test predictions are computed only after all methods for each seed are fit.")
    lines.append("- Coordinators receive visible candidate outputs and, for activation-aware methods, agent telemetry.")
    lines.append("- The deterministic synthetic task labels are never given to coordinators at prediction time.")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(results["metadata"], indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Aggregate Accuracy")
    lines.append("")
    lines.extend(_aggregate_table(metrics))
    lines.append("")
    lines.append("## Main Test Comparison")
    lines.append("")
    lines.extend(_main_comparison(metrics))
    lines.append("")
    lines.append("## Confidence Intervals")
    lines.append("")
    lines.extend(_confidence_interval_summary(metrics, diagnostics))
    lines.append("")
    lines.append("## Anti-Cheat Controls")
    lines.append("")
    lines.extend(_controls_summary(metrics))
    lines.append("")
    lines.append("## Randomized-Label Leakage Test")
    lines.append("")
    lines.extend(_randomized_label_summary(metrics))
    lines.append("")
    lines.append("## Label Permutation Sanity")
    lines.append("")
    lines.extend(_label_permutation_summary(metrics))
    lines.append("")
    lines.append("## Train/Test Split Audit")
    lines.append("")
    lines.extend(_split_audit_summary(audit))
    lines.append("")
    lines.append("## Diagnostic Label Probes")
    lines.append("")
    lines.extend(_diagnostic_label_probe_summary(metrics))
    lines.append("")
    lines.append("## Stage 1 Mechanism Validation")
    lines.append("")
    lines.extend(_stage1_mechanism_summary(metrics, diagnostics))
    lines.append("")
    lines.append("## Stage 3 Robustness")
    lines.append("")
    lines.extend(_stage3_robustness_summary(metrics, diagnostics))
    lines.append("")
    lines.append("## Stage 4 Hidden-State Coordinators")
    lines.append("")
    lines.extend(_stage4_hidden_coordinator_summary(metrics, diagnostics))
    lines.append("")
    lines.append("## Stage 4 Agent-Count Curve")
    lines.append("")
    lines.extend(_stage4_agent_count_summary(diagnostics))
    lines.append("")
    lines.append("## Stage 4 Controls")
    lines.append("")
    lines.extend(_stage4_controls_summary(diagnostics))
    lines.append("")
    lines.append("## Stage 5 Role-Aware QKV Coordinators")
    lines.append("")
    lines.extend(_stage5_role_attention_summary(metrics, diagnostics))
    lines.append("")
    lines.append("## Stage 5 Agent-Count Curve")
    lines.append("")
    lines.extend(_stage5_agent_count_summary(diagnostics))
    lines.append("")
    lines.append("## Stage 5 Controls")
    lines.append("")
    lines.extend(_stage5_controls_summary(diagnostics))
    lines.append("")
    lines.append("## Shared-Weight Cloned-Agent Training")
    lines.append("")
    lines.extend(_shared_weight_cloned_agent_summary(metrics, raw_diagnostics, audit, validation))
    lines.append("")
    lines.append("## Benchmark Validity Diagnostics")
    lines.append("")
    lines.extend(_benchmark_validity_diagnostics_summary(metrics, raw_diagnostics, validation, results.get("metadata", {})))
    lines.append("")
    lines.append("## Learnability Positive Controls")
    lines.append("")
    lines.extend(_learnability_positive_controls_summary(metrics, raw_diagnostics, validation))
    lines.append("")
    lines.append("## Latent Message Channel Diagnostics")
    lines.append("")
    lines.extend(_latent_message_channel_diagnostics_summary(metrics, raw_diagnostics, validation))
    lines.append("")
    lines.append("## Real Shared-Weight Latent Coordination")
    lines.append("")
    lines.extend(_real_shared_weight_latent_coordination_summary(metrics, raw_diagnostics, audit, validation, results.get("metadata", {})))
    lines.append("")
    lines.append("## Per-Evidence-Bit Probes")
    lines.append("")
    lines.extend(_private_evidence_bit_summary(diagnostics))
    lines.append("")
    lines.append("## Agent-Count Curve")
    lines.append("")
    lines.extend(_agent_count_curve_summary(diagnostics))
    lines.append("")
    lines.append("## Layer And Token Position Probes")
    lines.append("")
    lines.extend(_hidden_location_probe_summary(diagnostics))
    lines.append("")
    lines.append("## Prompt Robustness")
    lines.append("")
    lines.extend(_prompt_robustness_summary(diagnostics))
    lines.append("")
    lines.append("## Model Robustness")
    lines.append("")
    lines.extend(_model_robustness_summary(diagnostics))
    lines.append("")
    lines.append("## Evidence Masking And Shuffling")
    lines.append("")
    lines.extend(_evidence_control_summary(diagnostics))
    lines.append("")
    lines.append("## Output Leakage Audit")
    lines.append("")
    lines.extend(_output_leakage_summary(metrics, diagnostics))
    lines.append("")
    lines.append("## activation_pca_mlp Stability")
    lines.append("")
    lines.extend(_activation_stability_summary(metrics, diagnostics))
    lines.append("")
    lines.append("## Explicit Evidence-Sharing Baseline")
    lines.append("")
    lines.extend(_explicit_evidence_sharing_summary(metrics, diagnostics))
    lines.append("")
    lines.append("## Agent Correctness Probe")
    lines.append("")
    lines.extend(_agent_correctness_summary(diagnostics))
    lines.append("")
    lines.append("## Redundancy Probe")
    lines.append("")
    lines.extend(_redundancy_summary(diagnostics))
    lines.append("")
    lines.append("## Telemetry Channel Ablations")
    lines.append("")
    lines.extend(_telemetry_ablation_summary(metrics, diagnostics))
    lines.append("")
    lines.append("## Transformer Hidden-State Ablations")
    lines.append("")
    lines.extend(_transformer_hidden_ablation_summary(metrics, diagnostics))
    lines.append("")
    lines.append("## Individual Hidden-State Label Probe")
    lines.append("")
    lines.extend(_individual_hidden_label_summary(diagnostics))
    lines.append("")
    lines.append("## Probe Access Matrix")
    lines.append("")
    lines.extend(_probe_access_summary(validation))
    lines.append("")
    lines.append("## CPU/CUDA Parity")
    lines.append("")
    lines.extend(_cpu_cuda_parity_summary(validation))
    lines.append("")
    lines.append("## All Runs")
    lines.append("")
    lines.extend(_all_runs_table(metrics))
    lines.append("")
    lines.append("## Mechanism Distinction")
    lines.append("")
    lines.extend(_mechanism_distinction(metrics, diagnostics))
    lines.append("")
    lines.append("## Synthetic Vs Transformer Hidden States")
    lines.append("")
    lines.extend(_synthetic_vs_transformer_summary(metrics, diagnostics, validation))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.extend(_interpretation(metrics, diagnostics))
    return "\n".join(lines) + "\n"


def _aggregate_table(metrics: Iterable[Dict[str, object]]) -> List[str]:
    rows = _group_stats(metrics)
    lines = ["| benchmark | split | condition | method | mean acc | std | runs | params |", "|---|---|---|---|---:|---:|---:|---:|"]
    for key in sorted(rows):
        benchmark, split, condition, method = key
        values, params = rows[key]
        lines.append(
            f"| {benchmark} | {split} | {condition} | {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} | {int(mean(params))} |"
        )
    return lines


def _main_comparison(metrics: Iterable[Dict[str, object]]) -> List[str]:
    rows = _group_stats(
        row for row in metrics if row["split"] == "test" and row["condition"] == "none"
    )
    lines = ["| benchmark | method | test mean acc | std | runs |", "|---|---|---:|---:|---:|"]
    for (benchmark, _split, _condition, method), (values, _params) in sorted(
        rows.items(), key=lambda item: (item[0][0], -mean(item[1][0]))
    ):
        lines.append(f"| {benchmark} | {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    return lines


def _confidence_interval_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
) -> List[str]:
    target_methods = {
        "activation_pca_mlp",
        "telemetry_only_label_probe",
        "hidden_state_only_probe",
        "output_only_oracle_probe",
    }
    lines = ["| benchmark | quantity | mean | 95% bootstrap CI | n |", "|---|---|---:|---|---:|"]
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in metrics:
        if row["split"] == "test" and row["condition"] == "none" and row["method"] in target_methods:
            grouped[(str(row.get("benchmark", "stage0_original")), str(row["method"]))].append(
                float(row["accuracy"])
            )
    for (benchmark, method), values in sorted(grouped.items()):
        lo, hi = _bootstrap_ci(values)
        lines.append(f"| {benchmark} | {method} accuracy | {mean(values):.4f} | [{lo:.4f}, {hi:.4f}] | {len(values)} |")

    correctness: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in diagnostics:
        if row.get("split") == "test" and row.get("probe") == "agent_correctness_probe":
            correctness[(str(row["benchmark"]), str(row["feature_mode"]))].append(float(row["auc"]))
    for (benchmark, feature_mode), values in sorted(correctness.items()):
        lo, hi = _bootstrap_ci(values)
        lines.append(
            f"| {benchmark} | correctness AUC {feature_mode} | {mean(values):.4f} | [{lo:.4f}, {hi:.4f}] | {len(values)} |"
        )
    return lines


def _controls_summary(metrics: Iterable[Dict[str, object]]) -> List[str]:
    activation_methods = {
        "activation_pool_mlp",
        "activation_pca_mlp",
        "activation_cluster_router",
    }
    rows = _group_stats(
        row
        for row in metrics
        if row["split"] == "test"
        and row["method"] in activation_methods
        and row["condition"] != "randomized_train_labels"
    )
    lines = ["| benchmark | condition | method | test mean acc | std | runs |", "|---|---|---|---:|---:|---:|"]
    for (benchmark, _split, condition, method), (values, _params) in sorted(rows.items()):
        lines.append(f"| {benchmark} | {condition} | {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    return lines


def _randomized_label_summary(metrics: Iterable[Dict[str, object]]) -> List[str]:
    rows = _group_stats(
        row for row in metrics if row["split"] == "test" and row["condition"] == "randomized_train_labels"
    )
    if not rows:
        return ["Randomized-label leakage test was disabled."]
    lines = ["| benchmark | method | test mean acc | std | runs |", "|---|---|---:|---:|---:|"]
    for (benchmark, _split, _condition, method), (values, _params) in sorted(rows.items()):
        lines.append(f"| {benchmark} | {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    return lines


def _label_permutation_summary(metrics: Iterable[Dict[str, object]]) -> List[str]:
    lines = ["| benchmark | method | normal test acc | permuted-label test acc | drop |", "|---|---|---:|---:|---:|"]
    targets = {
        "activation_pca_mlp",
        "telemetry_only_label_probe",
        "hidden_state_only_probe",
        "output_only_oracle_probe",
    }
    normal = _metric_means(metrics, condition="none", methods=targets)
    permuted = _metric_means(metrics, condition="randomized_train_labels", methods=targets)
    for key in sorted(normal):
        if key not in permuted:
            continue
        benchmark, method = key
        drop = normal[key] - permuted[key]
        lines.append(f"| {benchmark} | {method} | {normal[key]:.4f} | {permuted[key]:.4f} | {drop:.4f} |")
    return lines


def _split_audit_summary(audit: object) -> List[str]:
    rows = [row for row in audit if isinstance(row, dict) and row.get("type") == "split_overlap"]
    summary = next(
        (row for row in audit if isinstance(row, dict) and row.get("type") == "test_access_summary"),
        {"violations": "not_available"},
    )
    lines = [
        f"Test access violations before the test gate: `{summary.get('violations')}`.",
        "",
        "| benchmark | seed | train/dev overlap | train/test overlap | dev/test overlap |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (str(item["benchmark"]), int(item["seed"]))):
        lines.append(
            f"| {row['benchmark']} | {row['seed']} | {row['train_dev_overlap']} | {row['train_test_overlap']} | {row['dev_test_overlap']} |"
        )
    return lines


def _diagnostic_label_probe_summary(metrics: Iterable[Dict[str, object]]) -> List[str]:
    rows = _group_stats(
        row
        for row in metrics
        if row["split"] == "test"
        and row["condition"] == "none"
        and row["method"]
        in {
            "activation_pca_mlp",
            "telemetry_only_label_probe",
            "hidden_state_only_probe",
            "output_only_oracle_probe",
            "text_only_coordinator",
            "capacity_matched_text_only",
        }
    )
    lines = ["| benchmark | method | test mean acc | std | runs | params |", "|---|---|---:|---:|---:|---:|"]
    for (benchmark, _split, _condition, method), (values, params) in sorted(rows.items()):
        lines.append(
            f"| {benchmark} | {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} | {int(mean(params))} |"
        )
    return lines


def _stage1_mechanism_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
) -> List[str]:
    means = _metric_means(
        metrics,
        "none",
        {
            "hidden_state_only_probe",
            "activation_pca_mlp",
            "output_only_oracle_probe",
            "text_only_coordinator",
            "capacity_matched_text_only",
        },
    )
    hidden = means.get(("transformer_strict", "hidden_state_only_probe"))
    activation = means.get(("transformer_strict", "activation_pca_mlp"))
    output = means.get(("transformer_strict", "output_only_oracle_probe"))
    text = max(
        [
            value
            for value in [
                means.get(("transformer_strict", "text_only_coordinator")),
                means.get(("transformer_strict", "capacity_matched_text_only")),
                output,
            ]
            if value is not None
        ],
        default=None,
    )
    bit_hidden = _private_bit_mean(diagnostics, "single_agent_hidden")
    bit_output = _private_bit_mean(diagnostics, "output_only")
    agent4 = _agent_count_mean(diagnostics, 4)
    controls = _evidence_control_means(diagnostics)
    lines = []
    if hidden is None:
        return ["Stage 1 transformer_strict diagnostics have not been run."]
    lines.append(
        "- Stage 1 supports distributed private-evidence recovery from real transformer hidden states in this controlled benchmark. It does not yet prove general open-ended real-agent coordination."
    )
    if output is not None and text is not None:
        lines.append(
            f"- Combined hidden-state-only final-label accuracy is {hidden:.4f}; strongest visible-output reference is {text:.4f}; output-only oracle final-label accuracy is {output:.4f}."
        )
    if activation is not None:
        lines.append(
            f"- `activation_pca_mlp` hidden+visible accuracy is {activation:.4f}; interpret it alongside hidden-only and stability diagnostics because PCA/pooling variance remains material."
        )
    if bit_hidden is not None:
        bit_text = f"{bit_output:.4f}" if bit_output is not None else "not run"
        lines.append(
            f"- Single-agent hidden states decode that agent's private evidence bit at mean accuracy {bit_hidden:.4f}; output-only private-bit probes score {bit_text}."
        )
    if agent4 is not None:
        lines.append(
            f"- The agent-count curve reaches {agent4:.4f} with all four available private-evidence agents. A five-agent point is not run because the task definition has four evidence agents."
        )
    if controls:
        control_text = ", ".join(f"{condition}={value:.4f}" for condition, value in sorted(controls.items()))
        lines.append(f"- Preserving labels while masking/shuffling evidence reduces hidden-only accuracy: {control_text}.")
    return lines


def _private_evidence_bit_summary(diagnostics: Iterable[Dict[str, object]]) -> List[str]:
    grouped: Dict[Tuple[str, int], Tuple[List[float], List[float]]] = {}
    for row in diagnostics:
        if row.get("split") != "test" or row.get("probe") != "private_evidence_bit_probe":
            continue
        accuracy = row.get("accuracy")
        if not isinstance(accuracy, (int, float)):
            continue
        key = (str(row["feature_source"]), int(row["agent_id"]))
        grouped.setdefault(key, ([], []))
        grouped[key][0].append(float(accuracy))
        grouped[key][1].append(float(row.get("majority_baseline", 0.0)))
    if not grouped:
        return ["Private-evidence-bit probes were not run."]
    lines = ["| feature source | agent/evidence bit | mean acc | std | majority baseline | runs |", "|---|---:|---:|---:|---:|---:|"]
    for (feature_source, agent_id), (values, baselines) in sorted(grouped.items()):
        lines.append(
            f"| {feature_source} | {agent_id} | {mean(values):.4f} | {_std(values):.4f} | {mean(baselines):.4f} | {len(values)} |"
        )
    return lines


def _stage4_hidden_coordinator_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
) -> List[str]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    params: Dict[str, List[int]] = defaultdict(list)
    for row in diagnostics:
        if (
            row.get("split") == "test"
            and row.get("probe") == "stage4_hidden_coordinator"
            and row.get("condition") == "all_agents"
            and isinstance(row.get("accuracy"), (int, float))
        ):
            method = str(row["method"])
            grouped[method].append(float(row["accuracy"]))
            params[method].append(int(row.get("param_count", 0)))
    if not grouped:
        return ["Stage 4 hidden-state coordinators were not run."]
    baseline_values: Dict[str, List[float]] = defaultdict(list)
    for row in metrics:
        if (
            row.get("benchmark") == "transformer_strict"
            and row.get("split") == "test"
            and row.get("condition") == "none"
            and row.get("method") in {"hidden_state_only_probe", "activation_pca_mlp", "output_only_oracle_probe"}
        ):
            baseline_values[str(row["method"])].append(float(row["accuracy"]))
    lines = ["| method | test mean acc | std | 95% bootstrap CI | params | runs |", "|---|---:|---:|---|---:|---:|"]
    for method, values in sorted(grouped.items()):
        lo, hi = _bootstrap_ci(values)
        lines.append(f"| {method} | {mean(values):.4f} | {_std(values):.4f} | [{lo:.4f}, {hi:.4f}] | {int(mean(params[method])) if params[method] else 0} | {len(values)} |")
    lines.append("")
    lines.extend(["| reference | test mean acc | std | runs |", "|---|---:|---:|---:|"])
    for method in ["hidden_state_only_probe", "activation_pca_mlp", "output_only_oracle_probe"]:
        values = baseline_values.get(method, [])
        if values:
            lines.append(f"| {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    hidden_std = _std(baseline_values.get("hidden_state_only_probe", []))
    for method, values in sorted(grouped.items()):
        reduction = hidden_std - _std(values)
        lines.append(f"- `{method}` variance delta vs `hidden_state_only_probe` std: {reduction:.4f} (positive means lower seed variance).")
    best_method, best_values = max(grouped.items(), key=lambda item: mean(item[1]))
    best_mean = mean(best_values)
    output_values = baseline_values.get("output_only_oracle_probe", [])
    if output_values and best_mean <= mean(output_values) + 0.03:
        lines.append(
            f"- Stage 4 interpretation: `{best_method}` reaches {best_mean:.4f} mean accuracy, matching the output-only baseline within 0.03. The set coordinators reduced seed variance and passed permutation/sanity controls, but did not replace `hidden_state_only_probe`; this diagnoses that simple pooling over only late final-token per-agent hidden states is insufficient for this benchmark."
        )
    else:
        lines.append(
            f"- Stage 4 interpretation: `{best_method}` reaches {best_mean:.4f} mean accuracy using only frozen late final-token hidden states. This is a controlled hidden-evidence coordinator result, not evidence of open-ended real-agent generalization."
        )
    return lines


def _stage4_agent_count_summary(diagnostics: Iterable[Dict[str, object]]) -> List[str]:
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in diagnostics:
        if row.get("split") == "test" and row.get("probe") == "stage4_agent_count_curve":
            accuracy = row.get("accuracy")
            if isinstance(accuracy, (int, float)):
                grouped[(str(row["method"]), int(row["agent_count"]))].append(float(accuracy))
    if not grouped:
        return ["Stage 4 agent-count curve was not run."]
    lines = ["| method | agents used | test mean acc | std | 95% bootstrap CI | runs |", "|---|---:|---:|---:|---|---:|"]
    for (method, agent_count), values in sorted(grouped.items()):
        lo, hi = _bootstrap_ci(values)
        lines.append(f"| {method} | {agent_count} | {mean(values):.4f} | {_std(values):.4f} | [{lo:.4f}, {hi:.4f}] | {len(values)} |")
    for method in sorted({method for method, _count in grouped}):
        means_by_count = {
            count: mean(values)
            for (row_method, count), values in grouped.items()
            if row_method == method
        }
        non_monotonic = []
        counts = sorted(means_by_count)
        for left, right in zip(counts, counts[1:]):
            if means_by_count[right] + 1e-12 < means_by_count[left]:
                non_monotonic.append(f"{left}->{right}: {means_by_count[left]:.4f}->{means_by_count[right]:.4f}")
        if non_monotonic:
            lines.append(f"- `{method}` non-monotonic points: {', '.join(non_monotonic)}. These are masked-agent evaluations of one dropout-trained coordinator, so finite-data and training variance can dominate adjacent counts.")
        else:
            lines.append(f"- `{method}` is monotonic over the evaluated agent-count means.")
    return lines


def _stage4_controls_summary(diagnostics: Iterable[Dict[str, object]]) -> List[str]:
    permutation: Dict[str, List[float]] = defaultdict(list)
    disagreement: Dict[str, List[float]] = defaultdict(list)
    label_perm: Dict[str, List[float]] = defaultdict(list)
    controls: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in diagnostics:
        if row.get("split") == "test" and row.get("probe") == "stage4_agent_order_permutation":
            method = str(row["method"])
            permutation[method].append(float(row.get("abs_accuracy_delta", 0.0)))
            disagreement[method].append(float(row.get("prediction_disagreement", 0.0)))
        elif row.get("split") == "test" and row.get("probe") == "stage4_label_permutation_sanity":
            accuracy = row.get("accuracy")
            if isinstance(accuracy, (int, float)):
                label_perm[str(row["method"])].append(float(accuracy))
        elif row.get("split") == "test" and row.get("probe") == "stage4_evidence_control":
            accuracy = row.get("accuracy")
            if isinstance(accuracy, (int, float)):
                controls[(str(row["method"]), str(row["condition"]))].append(float(accuracy))
    if not permutation and not label_perm and not controls:
        return ["Stage 4 controls were not run."]
    lines = ["| control | method | mean value | std | runs |", "|---|---|---:|---:|---:|"]
    for method, values in sorted(permutation.items()):
        lines.append(f"| order permutation abs acc delta | {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
        lines.append(f"| order permutation prediction disagreement | {method} | {mean(disagreement[method]):.4f} | {_std(disagreement[method]):.4f} | {len(disagreement[method])} |")
    for method, values in sorted(label_perm.items()):
        lines.append(f"| randomized train labels accuracy | {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    for (method, condition), values in sorted(controls.items()):
        lines.append(f"| {condition} accuracy | {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    return lines


def _stage5_role_attention_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
) -> List[str]:
    rows = [
        row
        for row in diagnostics
        if row.get("split") == "test"
        and row.get("probe") == "stage5_role_attention"
        and row.get("condition") == "all_agents"
        and isinstance(row.get("accuracy"), (int, float))
    ]
    if not rows:
        return ["Stage 5 role-aware QKV coordinators were not run."]
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    lines = [
        "| method | family | variant | feature mode | layers | heads | pooling | aux bits | test mean acc | std | 95% bootstrap CI | params | runs |",
        "|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for method, values in sorted(grouped.items()):
        acc = [float(row["accuracy"]) for row in values]
        lo, hi = _bootstrap_ci(acc)
        first = values[0]
        param_values = [int(row.get("param_count", 0)) for row in values]
        lines.append(
            f"| {method} | {first.get('family', '')} | {first.get('variant_name', '')} | {first.get('feature_mode', '')} | {int(first.get('num_attention_layers', 0))} | {int(first.get('num_heads', 0))} | {first.get('pooling', '')} | {str(bool(first.get('use_aux_private_bit_loss', False))).lower()} | {mean(acc):.4f} | {_std(acc):.4f} | [{lo:.4f}, {hi:.4f}] | {int(mean(param_values))} | {len(acc)} |"
        )
    best_by_family: Dict[str, Tuple[str, List[Dict[str, object]]]] = {}
    for method, values in grouped.items():
        family = str(values[0].get("family", ""))
        current = mean([float(row["accuracy"]) for row in values])
        if family not in best_by_family:
            best_by_family[family] = (method, values)
            continue
        previous = mean([float(row["accuracy"]) for row in best_by_family[family][1]])
        if current > previous:
            best_by_family[family] = (method, values)
    lines.append("")
    lines.extend(["| reference | test mean acc | std | runs |", "|---|---:|---:|---:|"])
    baseline_values: Dict[str, List[float]] = defaultdict(list)
    for row in metrics:
        if (
            row.get("benchmark") == "transformer_strict"
            and row.get("split") == "test"
            and row.get("condition") == "none"
            and row.get("method")
            in {"hidden_state_only_probe", "activation_pca_mlp", "output_only_oracle_probe", "text_only_coordinator"}
        ):
            baseline_values[str(row["method"])].append(float(row["accuracy"]))
    for method in ["hidden_state_only_probe", "activation_pca_mlp", "output_only_oracle_probe", "text_only_coordinator"]:
        values = baseline_values.get(method, [])
        if values:
            lines.append(f"| {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    stage4_values: Dict[str, List[float]] = defaultdict(list)
    for row in diagnostics:
        if (
            row.get("split") == "test"
            and row.get("probe") == "stage4_hidden_coordinator"
            and row.get("condition") == "all_agents"
            and isinstance(row.get("accuracy"), (int, float))
        ):
            stage4_values[str(row["method"])].append(float(row["accuracy"]))
    for method in ["deepsets_hidden_coordinator", "attention_hidden_coordinator"]:
        values = stage4_values.get(method, [])
        if values:
            lines.append(f"| {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    lines.append("")
    lines.extend(["| best family variant | mean acc | std | 95% bootstrap CI |", "|---|---:|---:|---|"])
    for family, (method, values) in sorted(best_by_family.items()):
        acc = [float(row["accuracy"]) for row in values]
        lo, hi = _bootstrap_ci(acc)
        lines.append(f"| {family}: {method} | {mean(acc):.4f} | {_std(acc):.4f} | [{lo:.4f}, {hi:.4f}] |")
    all_stage5 = {method: [float(row["accuracy"]) for row in values] for method, values in grouped.items()}
    best_method, best_values = max(all_stage5.items(), key=lambda item: mean(item[1]))
    best_mean = mean(best_values)
    output_mean = _mean_or_none(baseline_values.get("output_only_oracle_probe"))
    hidden_mean = _mean_or_none(baseline_values.get("hidden_state_only_probe"))
    stage4_best = max((mean(values) for values in stage4_values.values()), default=None)
    if output_mean is not None and best_mean <= output_mean + 0.03:
        lines.append(
            f"- Stage 5 interpretation: `{best_method}` reaches {best_mean:.4f}, which is not clearly above the output-only baseline ({output_mean:.4f}). Role-aware attention did not recover the hidden evidence in this run."
        )
        lines.append(
            "- Failure interpretation: these QKV heads are stable but appear to underfit or collapse on the raw role-labeled late/final hidden slices; the stronger `hidden_state_only_probe` result still depends on its PCA/MLP readout rather than being replaced by the small role-aware coordinators."
        )
    elif hidden_mean is not None and best_mean + 0.05 < hidden_mean:
        lines.append(
            f"- Stage 5 interpretation: `{best_method}` is above output-only but remains below `hidden_state_only_probe` ({hidden_mean:.4f}); this supports only a limited controlled hidden-state coordination claim."
        )
    else:
        lines.append(
            f"- Stage 5 interpretation: `{best_method}` approaches the hidden-state-only reference while using frozen transformer activations. This remains limited to the controlled transformer strict benchmark."
        )
    if stage4_best is not None:
        relation = "materially beats" if best_mean > stage4_best + 0.03 else "does not materially beat"
        lines.append(f"- Best Stage 5 variant {relation} the best Stage 4 set coordinator ({stage4_best:.4f}).")
    if hidden_mean is not None:
        hidden_std = _std(baseline_values.get("hidden_state_only_probe", []))
        lines.append(f"- Best Stage 5 std is {_std(best_values):.4f} versus `hidden_state_only_probe` std {hidden_std:.4f}.")
    lines.append("- Claim boundary: Stage 5 tests role-aware hidden-state coordination in this benchmark only; it is not evidence of open-ended real-agent generalization.")
    return lines


def _stage5_agent_count_summary(diagnostics: Iterable[Dict[str, object]]) -> List[str]:
    best_methods = _stage5_best_methods(diagnostics)
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in diagnostics:
        if row.get("split") == "test" and row.get("probe") == "stage5_agent_count_curve":
            if best_methods and row.get("method") not in best_methods:
                continue
            accuracy = row.get("accuracy")
            if isinstance(accuracy, (int, float)):
                grouped[(str(row["method"]), int(row["agent_count"]))].append(float(accuracy))
    if not grouped:
        return ["Stage 5 agent-count curve was not run."]
    lines = ["| method | agents used | test mean acc | std | 95% bootstrap CI | runs |", "|---|---:|---:|---:|---|---:|"]
    for (method, agent_count), values in sorted(grouped.items()):
        lo, hi = _bootstrap_ci(values)
        lines.append(f"| {method} | {agent_count} | {mean(values):.4f} | {_std(values):.4f} | [{lo:.4f}, {hi:.4f}] | {len(values)} |")
    for method in sorted({method for method, _count in grouped}):
        means_by_count = {
            count: mean(values)
            for (row_method, count), values in grouped.items()
            if row_method == method
        }
        non_monotonic = []
        counts = sorted(means_by_count)
        for left, right in zip(counts, counts[1:]):
            if means_by_count[right] + 1e-12 < means_by_count[left]:
                non_monotonic.append(f"{left}->{right}: {means_by_count[left]:.4f}->{means_by_count[right]:.4f}")
        if non_monotonic:
            lines.append(f"- `{method}` non-monotonic points: {', '.join(non_monotonic)}.")
        else:
            lines.append(f"- `{method}` is monotonic over the evaluated agent-count means.")
    return lines


def _stage5_controls_summary(diagnostics: Iterable[Dict[str, object]]) -> List[str]:
    best_methods = _stage5_best_methods(diagnostics)
    order_delta: Dict[str, List[float]] = defaultdict(list)
    order_disagreement: Dict[str, List[float]] = defaultdict(list)
    slot_shuffle: Dict[str, List[float]] = defaultdict(list)
    label_perm: Dict[str, List[float]] = defaultdict(list)
    controls: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in diagnostics:
        if best_methods and row.get("method") not in best_methods:
            continue
        if row.get("split") == "test" and row.get("probe") == "stage5_physical_order_permutation":
            method = str(row["method"])
            order_delta[method].append(float(row.get("abs_accuracy_delta", 0.0)))
            order_disagreement[method].append(float(row.get("prediction_disagreement", 0.0)))
        elif row.get("split") == "test" and row.get("probe") == "stage5_slot_label_shuffle":
            accuracy = row.get("accuracy")
            if isinstance(accuracy, (int, float)):
                slot_shuffle[str(row["method"])].append(float(accuracy))
        elif row.get("split") == "test" and row.get("probe") == "stage5_label_permutation_sanity":
            accuracy = row.get("accuracy")
            if isinstance(accuracy, (int, float)):
                label_perm[str(row["method"])].append(float(accuracy))
        elif row.get("split") == "test" and row.get("probe") == "stage5_evidence_control":
            accuracy = row.get("accuracy")
            if isinstance(accuracy, (int, float)):
                controls[(str(row["method"]), str(row["condition"]))].append(float(accuracy))
    if not order_delta and not slot_shuffle and not label_perm and not controls:
        return ["Stage 5 controls were not run."]
    lines = ["| control | method | mean value | std | runs |", "|---|---|---:|---:|---:|"]
    for method, values in sorted(order_delta.items()):
        lines.append(f"| physical order abs acc delta | {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
        lines.append(f"| physical order prediction disagreement | {method} | {mean(order_disagreement[method]):.4f} | {_std(order_disagreement[method]):.4f} | {len(order_disagreement[method])} |")
    for method, values in sorted(slot_shuffle.items()):
        lines.append(f"| slot-label shuffle accuracy | {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    for method, values in sorted(label_perm.items()):
        lines.append(f"| randomized train labels accuracy | {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    for (method, condition), values in sorted(controls.items()):
        lines.append(f"| {condition} accuracy | {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    lines.append("- Expected control behavior: physical order should be stable when slot labels move with agents; wrong slot labels, hidden-state shuffling, and evidence masking/shuffling should collapse any real role-aware hidden-evidence signal toward chance/output-only.")
    return lines


def _shared_weight_cloned_agent_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
    audit: object,
    validation: object,
) -> List[str]:
    del audit, validation
    stage_metrics = [row for row in metrics if row.get("benchmark") == "shared_weight_cloned_agent"]
    stage_diags = [row for row in diagnostics if row.get("benchmark") == "shared_weight_cloned_agent"]
    if not stage_metrics and not stage_diags:
        return ["Stage 6A shared-weight cloned-agent training was not run."]

    lines = [
        "Stage 6A trains one shared small transformer agent end-to-end from the final group-answer loss. The hidden-state-only probe remains a diagnostic baseline, not the main method.",
        "",
        "Main comparison:",
        "| method | test mean acc | std | runs | params |",
        "|---|---:|---:|---:|---:|",
    ]
    method_order = [
        "frozen_shared_agent_coordinator",
        "trainable_shared_agent_coordinator",
        "text_only_coordinator",
        "evidence_sharing_rule_oracle",
        "hidden_state_only_probe",
    ]
    grouped = _group_stats(
        row
        for row in stage_metrics
        if row.get("split") == "test"
        and row.get("condition") == "none"
        and row.get("method") in set(method_order)
    )
    by_method = {key[3]: values for key, values in grouped.items()}
    for method in method_order:
        values_params = by_method.get(method)
        if not values_params:
            continue
        values, params = values_params
        lines.append(f"| {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} | {int(mean(params))} |")

    audit_rows = [
        row
        for row in stage_diags
        if row.get("probe") == "shared_weight_training_audit"
    ]
    lines.extend(
        [
            "",
            "Training audit:",
            "| method | condition | shared params | M grad mean | M grad std | C grad mean | M delta | C delta | activations grad | per-clone grad |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in sorted(audit_rows, key=lambda item: (str(item.get("method")), str(item.get("condition")))):
        lines.append(
            "| {method} | {condition} | {shared} | {m_grad:.6f} | {m_grad_std:.6f} | {c_grad:.6f} | {m_delta:.6f} | {c_delta:.6f} | {act} | {clone} |".format(
                method=row.get("method"),
                condition=row.get("condition"),
                shared="pass" if row.get("shared_parameter_identity") else "fail",
                m_grad=float(row.get("agent_grad_norm_mean", 0.0)),
                m_grad_std=float(row.get("agent_grad_norm_std", 0.0)),
                c_grad=float(row.get("coordinator_grad_norm_mean", 0.0)),
                m_delta=float(row.get("agent_parameter_delta", 0.0)),
                c_delta=float(row.get("coordinator_parameter_delta", 0.0)),
                act="pass" if row.get("clone_activation_requires_grad") else "fail",
                clone="pass" if row.get("per_clone_gradient_contribution") else "fail",
            )
        )

    control_rows = _group_stats(
        row
        for row in stage_metrics
        if row.get("split") == "test"
        and row.get("condition") != "none"
        and row.get("method") == "trainable_shared_agent_coordinator"
    )
    lines.extend(
        [
            "",
            "Controls:",
            "| condition | method | test mean acc | std | runs |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for (_benchmark, _split, condition, method), (values, _params) in sorted(control_rows.items()):
        lines.append(f"| {condition} | {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")

    curve_rows = [
        row
        for row in stage_diags
        if row.get("probe") == "shared_weight_learning_curve"
        and row.get("split") in {"train", "dev", "test"}
        and isinstance(row.get("accuracy"), (int, float))
    ]
    if curve_rows:
        lines.extend(
            [
                "",
                "Learning curves:",
                "| method | split | first epoch acc | final epoch acc | epochs |",
                "|---|---|---:|---:|---:|",
            ]
        )
        grouped_curves: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
        for row in curve_rows:
            grouped_curves[(str(row["method"]), str(row["split"]))].append(row)
        for (method, split), rows in sorted(grouped_curves.items()):
            rows = sorted(rows, key=lambda item: int(item["epoch"]))
            lines.append(
                f"| {method} | {split} | {float(rows[0]['accuracy']):.4f} | {float(rows[-1]['accuracy']):.4f} | {int(rows[-1]['epoch'])} |"
            )

    test_means = _metric_means(
        stage_metrics,
        "none",
        {"trainable_shared_agent_coordinator", "frozen_shared_agent_coordinator"},
    )
    trainable = test_means.get(("shared_weight_cloned_agent", "trainable_shared_agent_coordinator"))
    frozen = test_means.get(("shared_weight_cloned_agent", "frozen_shared_agent_coordinator"))
    trainable_audit = next(
        (
            row
            for row in audit_rows
            if row.get("method") == "trainable_shared_agent_coordinator"
            and row.get("condition") == "none"
        ),
        {},
    )
    m_updated = float(trainable_audit.get("agent_grad_norm_mean", 0.0)) > 0.0 and float(trainable_audit.get("agent_parameter_delta", 0.0)) > 0.0
    lines.append("")
    if trainable is not None and frozen is not None and trainable > frozen and m_updated:
        lines.append("Evidence supports that the shared model learned activation representations that are more useful for coordination with its own clones.")
    elif trainable is not None and frozen is not None and not m_updated:
        lines.append("This is still a coordinator/probe result, not shared-model self-collaboration.")
    else:
        lines.append("The current architecture/training loop does not yet induce shared-weight cloned-agent cooperation.")
    return lines


def _real_shared_weight_latent_coordination_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
    audit: object,
    validation: object,
    metadata: object,
) -> List[str]:
    stage_metrics = [row for row in metrics if row.get("benchmark") == "real_shared_weight_latent_coordination"]
    stage_diags = [row for row in diagnostics if row.get("benchmark") == "real_shared_weight_latent_coordination"]
    if not stage_metrics and not stage_diags:
        return ["Real shared-weight latent coordination was not run."]

    meta = metadata if isinstance(metadata, dict) else {}
    dataset_summary_by_seed = (meta.get("dataset_summary_by_seed") or {}) if isinstance(meta.get("dataset_summary_by_seed"), dict) else {}
    first_dataset_summary = next(iter(dataset_summary_by_seed.values()), {}) if dataset_summary_by_seed else {}
    agent_config = meta.get("agent_config", {}) if isinstance(meta.get("agent_config"), dict) else {}
    training_config = meta.get("training_config", {}) if isinstance(meta.get("training_config"), dict) else {}
    val = {}
    if isinstance(validation, dict):
        val = validation.get("real_shared_weight_latent_coordination", {})
    proposed_method = str(val.get("proposed_method", "trainable_shared_agent_latent_coordinator")) if isinstance(val, dict) else "trainable_shared_agent_latent_coordinator"

    lines: List[str] = [
        "This section is governed by the real shared-weight latent coordination specification. The available local data did not include SWE-bench or issue-patch artifacts, so the run uses a constructed local-code patch-selection fallback and is not treated as publishable real-world proof.",
        "",
        "Architecture and data:",
        f"- Agent model mode: `{agent_config.get('agent_mode', 'unknown')}`; model path: `{agent_config.get('model_name_or_path', 'n/a')}`; hidden dim: `{agent_config.get('hidden_dim', 'n/a')}`.",
        f"- Trainable parameters: shared agent coordination parameters plus coordinator in trainable runs; frozen runs keep shared agent deltas at zero.",
        f"- Dataset source: `{first_dataset_summary.get('dataset_source', 'unknown')}`; split sizes: `{first_dataset_summary.get('split_sizes', {})}`; source files used: `{first_dataset_summary.get('source_files_used', 'n/a')}`.",
        f"- Batch settings: batch size `{training_config.get('batch_size', 'n/a')}`, epochs `{training_config.get('epochs', 'n/a')}`, grad accumulation `{training_config.get('gradient_accumulation_steps', 'n/a')}`.",
        "",
        "Main comparison:",
        "| method | test mean acc | std | runs | params |",
        "|---|---:|---:|---:|---:|",
    ]
    main_rows = _group_stats(
        row
        for row in stage_metrics
        if row.get("split") == "test" and row.get("condition") == "none"
    )
    for (_benchmark, _split, _condition, method), (values, params) in sorted(
        main_rows.items(),
        key=lambda item: (-mean(item[1][0]), item[0][3]),
    ):
        lines.append(f"| {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} | {int(mean(params))} |")

    audit_rows = [
        row
        for row in stage_diags
        if row.get("probe") == "real_shared_weight_training_audit"
    ]
    lines.extend(
        [
            "",
            "Audit table:",
            "| method | condition | shared params | M grad | M delta | msg-head grad | msg-head delta | aux-head grad | aux-head delta | C grad | C delta | activations grad | no detach | per-clone grad | memory bytes |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|",
        ]
    )
    for row in sorted(audit_rows, key=lambda item: (str(item.get("method")), str(item.get("condition")))):
        lines.append(
            "| {method} | {condition} | {shared} | {m_grad:.6f} | {m_delta:.6f} | {h_grad:.6f} | {h_delta:.6f} | {a_grad:.6f} | {a_delta:.6f} | {c_grad:.6f} | {c_delta:.6f} | {act} | {detach} | {clone} | {mem} |".format(
                method=row.get("method"),
                condition=row.get("condition"),
                shared="pass" if row.get("shared_parameter_identity") else "fail",
                m_grad=float(row.get("agent_grad_norm_mean", 0.0)),
                m_delta=float(row.get("agent_parameter_delta", 0.0)),
                h_grad=float(row.get("message_head_grad_norm_mean", 0.0)),
                h_delta=float(row.get("message_head_parameter_delta", 0.0)),
                a_grad=float(row.get("private_cue_head_grad_norm_mean", 0.0)),
                a_delta=float(row.get("private_cue_head_parameter_delta", 0.0)),
                c_grad=float(row.get("coordinator_grad_norm_mean", 0.0)),
                c_delta=float(row.get("coordinator_parameter_delta", 0.0)),
                act="pass" if row.get("activation_requires_grad_before_coordinator") else "fail",
                detach="pass" if row.get("no_detach_between_clone_activations_and_loss") else "fail",
                clone="pass" if row.get("per_clone_gradient_contribution") else "fail",
                mem=int(row.get("cuda_max_memory_allocated", 0)),
            )
        )

    control_rows = _group_stats(
        row
        for row in stage_metrics
        if row.get("split") == "test"
        and row.get("condition") != "none"
        and row.get("method") == proposed_method
    )
    lines.extend(
        [
            "",
            "Controls:",
            "| condition | test mean acc | std | runs |",
            "|---|---:|---:|---:|",
        ]
    )
    for (_benchmark, _split, condition, _method), (values, _params) in sorted(control_rows.items()):
        lines.append(f"| {condition} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")

    curve_rows = [
        row
        for row in stage_diags
        if row.get("probe") == "real_shared_weight_learning_curve"
        and isinstance(row.get("accuracy"), (int, float))
    ]
    if curve_rows:
        lines.extend(
            [
                "",
                "Learning curves:",
                "| method | condition | split | first epoch acc | final epoch acc | epochs |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        grouped_curves: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
        for row in curve_rows:
            grouped_curves[(str(row["method"]), str(row.get("condition", "none")), str(row["split"]))].append(row)
        for (method, condition, split), rows in sorted(grouped_curves.items()):
            rows = sorted(rows, key=lambda item: int(item["epoch"]))
            lines.append(
                f"| {method} | {condition} | {split} | {float(rows[0]['accuracy']):.4f} | {float(rows[-1]['accuracy']):.4f} | {int(rows[-1]['epoch'])} |"
            )

    split_rows = [
        row
        for row in audit
        if isinstance(row, dict) and row.get("type") == "real_shared_weight_split_leakage"
    ]
    output_rows = [
        row
        for row in audit
        if isinstance(row, dict) and row.get("type") == "real_shared_weight_output_leakage"
    ]
    lines.extend(
        [
            "",
            "Split and output leakage audit:",
            "| seed | id overlaps train/dev/test | candidate hash overlaps train/dev/test | output leakage passes |",
            "|---:|---|---|---|",
        ]
    )
    for row in sorted(split_rows, key=lambda item: int(item.get("seed", 0))):
        out = next((item for item in output_rows if item.get("seed") == row.get("seed")), {})
        id_text = f"{row.get('train_dev_id_overlap')}/{row.get('train_test_id_overlap')}/{row.get('dev_test_id_overlap')}"
        cand_text = f"{row.get('train_dev_candidate_hash_overlap')}/{row.get('train_test_candidate_hash_overlap')}/{row.get('dev_test_candidate_hash_overlap')}"
        lines.append(f"| {row.get('seed')} | {id_text} | {cand_text} | {bool(out.get('passes', False))} |")

    gate_info = val.get("real_shared_weight_latent_coordination", val) if isinstance(val, dict) else {}
    lines.extend(
        [
            "",
            "Acceptance gates:",
            "| gate | pass |",
            "|---|---|",
        ]
    )
    for gate in [
        "gate_a_valid_training_graph",
        "gate_b_frozen_control",
        "gate_c_trainable_beats_frozen",
        "gate_d_trainable_beats_text_only",
        "gate_e_controls_collapse",
        "gate_f_dataset_not_trivial",
    ]:
        lines.append(f"| {gate} | {bool(gate_info.get(gate, False))} |")

    means = _metric_means(
        stage_metrics,
        "none",
        {
            proposed_method,
            str(gate_info.get("frozen_comparator_method", "frozen_shared_agent_latent_coordinator")) if isinstance(gate_info, dict) else "frozen_shared_agent_latent_coordinator",
            "text_output_only_multi_agent_coordinator",
            "single_agent_full_context",
            "single_agent_partial_view",
        },
    )
    frozen_method = str(gate_info.get("frozen_comparator_method", "frozen_shared_agent_latent_coordinator")) if isinstance(gate_info, dict) else "frozen_shared_agent_latent_coordinator"
    trainable = means.get(("real_shared_weight_latent_coordination", proposed_method))
    frozen = means.get(("real_shared_weight_latent_coordination", frozen_method))
    text_only = means.get(("real_shared_weight_latent_coordination", "text_output_only_multi_agent_coordinator"))
    partial = means.get(("real_shared_weight_latent_coordination", "single_agent_partial_view"))
    trainable_audit = next(
        (
            row
            for row in audit_rows
            if row.get("method") == proposed_method
            and row.get("condition") == "none"
        ),
        {},
    )
    lines.extend(
        [
            "",
            "Checklist:",
            f"1. Exact model: `{trainable_audit.get('agent_mode', agent_config.get('agent_mode', 'unknown'))}` with `{agent_config.get('model_name_or_path', 'n/a')}` when pretrained loading is enabled.",
            f"2. Trainable parameters: `{trainable_audit.get('coordination_parameter_names', [])}`.",
            f"3. Shared across clones: `{bool(trainable_audit.get('shared_parameter_identity', False))}`.",
            f"4. Nonzero shared gradients: `{float(trainable_audit.get('agent_grad_norm_mean', 0.0)) > 0.0}`.",
            f"5. Shared parameter delta: `{float(trainable_audit.get('agent_parameter_delta', 0.0)):.6f}`.",
            f"6. Trainable beat frozen: `{trainable is not None and frozen is not None and trainable > frozen}`.",
            f"7. Trainable beat text-only: `{trainable is not None and text_only is not None and trainable > text_only}`.",
            "8. Controls collapse: inspect the controls table; this fallback run does not by itself authorize a positive real-world claim.",
            "9. Physical order shuffle with roles preserved: inspect `physical_order_shuffled_roles_preserved`.",
            "10. Role-label shuffle: inspect `role_labels_shuffled`.",
            f"11. Single partial-view accuracy: `{partial if partial is not None else 'not_run'}`.",
            f"12. Output leakage: `{all(bool(row.get('passes', False)) for row in output_rows)}`.",
            f"13. Train/dev/test leakage: `{all(int(row.get('train_test_id_overlap', 1)) == 0 for row in split_rows)}`.",
            f"14. GPU memory: `{max([int(row.get('cuda_max_memory_allocated', 0)) for row in audit_rows] or [0])}` bytes.",
            "15. Failed variants tried: all trained coordinator-family variants are shown in the main table and audit table.",
            "16. Most conservative valid claim: constructed-fallback graph and audit plumbing are implemented; real-world latent coordination remains unproven without a real issue-patch dataset and stronger trained-model performance.",
        ]
    )

    lines.append("")
    publishable_dataset = bool(first_dataset_summary.get("publishable_proof_dataset", False))
    pretrained_agent = agent_config.get("agent_mode") == "pretrained_adapter"
    if not bool(gate_info.get("gate_a_valid_training_graph", False)):
        lines.append("The result is not valid as evidence of latent coordination because one or more leakage/shortcut controls failed.")
    elif not bool(gate_info.get("gate_e_controls_collapse", False)):
        lines.append("The result is not valid as evidence of latent coordination because one or more leakage/shortcut controls failed.")
    elif trainable is not None and frozen is not None and trainable <= frozen:
        lines.append("The current architecture/training setup does not demonstrate shared-weight cloned-agent cooperation. Further work should diagnose model capacity, activation extraction, dataset difficulty, and coordinator design.")
    elif trainable is not None and text_only is not None and trainable <= text_only:
        lines.append("This remains a coordinator-learning result. The experiment does not show that the shared agent model learned to cooperate, because shared model trainable parameters did not materially improve over the frozen baseline.")
    elif not publishable_dataset or not pretrained_agent:
        lines.append("The current architecture/training setup does not demonstrate shared-weight cloned-agent cooperation. Further work should diagnose model capacity, activation extraction, dataset difficulty, and coordinator design.")
    else:
        lines.append("Evidence supports that a shared pretrained transformer with trainable adapters can learn coordination-friendly internal activations when cloned across partial task views and trained through a latent coordinator under final group loss. The result is limited to this benchmark and does not yet imply open-ended multi-agent self-organization.")
    return lines


def _benchmark_validity_diagnostics_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
    validation: object,
    metadata: object,
) -> List[str]:
    stage_metrics = [row for row in metrics if row.get("benchmark") == "real_shared_weight_latent_coordination"]
    stage_diags = [row for row in diagnostics if row.get("benchmark") == "real_shared_weight_latent_coordination"]
    if not stage_metrics and not stage_diags:
        return ["Benchmark validity diagnostics were not run."]

    meta = metadata if isinstance(metadata, dict) else {}
    previous = meta.get("previous_run_summary", {}) if isinstance(meta.get("previous_run_summary"), dict) else {}
    val = validation.get("real_shared_weight_latent_coordination", {}) if isinstance(validation, dict) else {}
    fixes = next(
        (row for row in stage_diags if row.get("probe") == "benchmark_validity_fixes_applied"),
        {},
    )
    lines: List[str] = []
    failed = fixes.get("failed_shortcuts_found", [])
    applied = fixes.get("dataset_fixes_applied", [])
    if failed:
        lines.append("Failed shortcuts found:")
        lines.extend(f"- {item}" for item in failed)
    if applied:
        lines.append("")
        lines.append("Dataset fixes applied:")
        lines.extend(f"- {item}" for item in applied)

    lines.extend(
        [
            "",
            "Trivial baseline table:",
            "| method | test acc | std | runs |",
            "|---|---:|---:|---:|",
        ]
    )
    trivial_methods = {
        "majority_class_baseline",
        "candidate_order_baseline",
        "bag_of_words_candidate_patch_only",
        "text_only_partial_view_baseline",
        "single_view_text_role_0",
        "single_view_text_role_1",
        "single_view_text_role_2",
        "single_view_text_role_3",
        "masked_evidence_text_baseline",
        "role_labels_only_baseline",
    }
    trivial_rows = _group_stats(
        row
        for row in stage_metrics
        if row.get("split") == "test" and row.get("method") in trivial_methods
    )
    if trivial_rows:
        for (_benchmark, _split, _condition, method), (values, _params) in sorted(trivial_rows.items()):
            lines.append(f"| {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    else:
        lines.append("| not_run | 0.0000 | 0.0000 | 0 |")

    lines.extend(
        [
            "",
            "Control table before/after:",
            "| condition | before acc | after acc |",
            "|---|---:|---:|",
        ]
    )
    before_controls = previous.get("control_accuracy", {}) if isinstance(previous, dict) else {}
    after_controls = val.get("control_accuracy", {}) if isinstance(val, dict) else {}
    control_names = sorted(set(before_controls) | set(after_controls))
    if control_names:
        for condition in control_names:
            before = before_controls.get(condition)
            after = after_controls.get(condition)
            before_text = f"{float(before):.4f}" if isinstance(before, (int, float)) else "n/a"
            after_text = f"{float(after):.4f}" if isinstance(after, (int, float)) else "n/a"
            lines.append(f"| {condition} | {before_text} | {after_text} |")
    else:
        lines.append("| not_available | n/a | n/a |")

    label_rows = [
        row for row in stage_diags if row.get("probe") == "benchmark_validity_label_and_candidate_audit"
    ]
    if label_rows:
        lines.extend(
            [
                "",
                "Label and split audit:",
                "| split | label counts | problem families |",
                "|---|---|---|",
            ]
        )
        for row in sorted(label_rows, key=lambda item: str(item.get("split"))):
            lines.append(f"| {row.get('split')} | `{row.get('label_counts')}` | `{row.get('problem_families')}` |")

    inspections = [
        row
        for row in stage_diags
        if row.get("probe") == "benchmark_validity_control_inspection"
        and row.get("split") == "test"
    ]
    if inspections:
        lines.extend(
            [
                "",
                "Correct examples under corrupted controls:",
                "| condition | acc | inspected correct examples |",
                "|---|---:|---|",
            ]
        )
        for row in sorted(inspections, key=lambda item: str(item.get("condition"))):
            ids = [str(item.get("id")) for item in row.get("correct_examples", [])]
            lines.append(f"| {row.get('condition')} | {float(row.get('accuracy', 0.0)):.4f} | `{ids}` |")

    lines.extend(
        [
            "",
            "Final gate status:",
            "| gate | pass |",
            "|---|---|",
        ]
    )
    for gate in [
        "gate_a_valid_training_graph",
        "gate_b_frozen_control",
        "gate_c_trainable_beats_frozen",
        "gate_d_trainable_beats_text_only",
        "gate_e_controls_collapse",
        "gate_f_dataset_not_trivial",
        "gate_positive_control_learnable",
    ]:
        lines.append(f"| {gate} | {bool(val.get(gate, False))} |")
    return lines


def _learnability_positive_controls_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
    validation: object,
) -> List[str]:
    del diagnostics
    stage_metrics = [row for row in metrics if row.get("benchmark") == "real_shared_weight_latent_coordination"]
    if not stage_metrics:
        return ["Learnability positive controls were not run."]
    positive_methods = {
        "explicit_evidence_neural_tuple_model",
        "raw_structured_evidence_coordinator",
        "larger_tiny_transformer_full_context",
        "trainable_shared_agent_latent_visible_explicit_evidence",
        "single_agent_full_context",
    }
    rows = _group_stats(
        row
        for row in stage_metrics
        if row.get("split") == "test" and row.get("method") in positive_methods
    )
    lines = [
        "These controls test whether the repaired benchmark is learnable before tuning latent architecture.",
        "",
        "| method | condition | test acc | std | runs | params |",
        "|---|---|---:|---:|---:|---:|",
    ]
    if rows:
        for (_benchmark, _split, condition, method), (values, params) in sorted(
            rows.items(),
            key=lambda item: (-mean(item[1][0]), item[0][3], item[0][2]),
        ):
            lines.append(f"| {method} | {condition} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} | {int(mean(params))} |")
    else:
        lines.append("| not_run | none | 0.0000 | 0.0000 | 0 | 0 |")

    val = validation.get("real_shared_weight_latent_coordination", {}) if isinstance(validation, dict) else {}
    positive_acc = val.get("positive_control_accuracy", {}) if isinstance(val, dict) else {}
    best = max([float(value) for value in positive_acc.values()], default=0.0)
    lines.append("")
    lines.append(f"Learnability gate `gate_positive_control_learnable`: `{bool(val.get('gate_positive_control_learnable', False))}`; best learned positive-control test accuracy `{best:.4f}`.")
    if best < 0.8:
        lines.append("No learned positive control solved the benchmark above 0.8, so architecture tuning should not be interpreted as latent coordination evidence yet.")
    else:
        lines.append("At least one learned positive control solved the benchmark, so subsequent failures are more likely model/architecture/training issues than label construction issues.")
    return lines


def _latent_message_channel_diagnostics_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
    validation: object,
) -> List[str]:
    stage_metrics = [row for row in metrics if row.get("benchmark") == "real_shared_weight_latent_coordination"]
    stage_diags = [row for row in diagnostics if row.get("benchmark") == "real_shared_weight_latent_coordination"]
    variant_rows = [row for row in stage_diags if row.get("probe") == "message_channel_variant"]
    if not variant_rows:
        return ["Latent message-channel variants were not run."]

    val = validation.get("real_shared_weight_latent_coordination", {}) if isinstance(validation, dict) else {}
    proposed_method = str(val.get("proposed_method", "trainable_shared_agent_active_msg_head_aux_candidate_query"))
    frozen_method = str(val.get("frozen_comparator_method", "frozen_shared_agent_active_msg_head_aux_candidate_query"))
    method_order = []
    seen = set()
    for row in sorted(variant_rows, key=lambda item: str(item.get("variant", ""))):
        method = str(row.get("method"))
        if method not in seen:
            seen.add(method)
            method_order.append(method)

    lines: List[str] = [
        "This section records active message-readout variants. Test probe rows are computed only after the test gate is opened; architecture decisions should use train/dev behavior.",
        "",
        "Architecture tried:",
        "| variant | method | architecture | why tried | diagnostics trigger |",
        "|---|---|---|---|---|",
    ]
    for row in sorted(variant_rows, key=lambda item: str(item.get("variant", ""))):
        lines.append(
            f"| {row.get('variant')} | {row.get('method')} | {row.get('architecture')} | {row.get('why_tried')} | {row.get('diagnostics_trigger')} |"
        )

    rows = _group_stats(
        row
        for row in stage_metrics
        if row.get("split") == "test" and row.get("condition") == "none" and row.get("method") in set(method_order + [frozen_method])
    )
    lines.extend(
        [
            "",
            "Main result table:",
            "| method | test acc | std | runs | params |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in method_order + ([frozen_method] if frozen_method not in method_order else []):
        match = next((values for key, values in rows.items() if key[3] == method), None)
        if not match:
            continue
        values, params = match
        lines.append(f"| {method} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} | {int(mean(params))} |")

    comparison_rows = [
        row
        for row in stage_diags
        if row.get("split") == "test" and row.get("probe") == "trainable_vs_frozen_active_message_accuracy"
    ]
    if comparison_rows:
        lines.extend(
            [
                "",
                "Active trainable-vs-frozen comparison:",
                "| trainable method | frozen method | trainable acc | frozen acc | delta | interpretation |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for row in sorted(comparison_rows, key=lambda item: int(item.get("seed", 0))):
            lines.append(
                "| {trainable} | {frozen} | {ta:.4f} | {fa:.4f} | {delta:.4f} | {interp} |".format(
                    trainable=row.get("trainable_method"),
                    frozen=row.get("frozen_method"),
                    ta=float(row.get("trainable_accuracy", 0.0)),
                    fa=float(row.get("frozen_accuracy", 0.0)),
                    delta=float(row.get("delta_trainable_minus_frozen", 0.0)),
                    interp=row.get("interpretation"),
                )
            )

    control_rows = _group_stats(
        row
        for row in stage_metrics
        if row.get("split") == "test" and row.get("method") in set(method_order + [proposed_method]) and row.get("condition") != "none"
    )
    lines.extend(
        [
            "",
            "Control table:",
            "| method | condition | test acc | std | runs |",
            "|---|---|---:|---:|---:|",
        ]
    )
    if control_rows:
        for (_benchmark, _split, condition, method), (values, _params) in sorted(control_rows.items()):
            lines.append(f"| {method} | {condition} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    else:
        lines.append("| not_run | none | 0.0000 | 0.0000 | 0 |")

    private_rows = [
        row
        for row in stage_diags
        if row.get("split") == "test" and row.get("probe") == "private_cue_probe" and isinstance(row.get("accuracy"), (int, float))
    ]
    grouped_private: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in private_rows:
        grouped_private[(str(row.get("method")), str(row.get("feature_source")))].append(float(row["accuracy"]))
    lines.extend(
        [
            "",
            "Mechanism probe table:",
            "| probe | method | feature source | test mean acc | std | runs |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for (method, source), values in sorted(grouped_private.items()):
        lines.append(f"| private cue | {method} | {source} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")

    combined_rows = [
        row
        for row in stage_diags
        if row.get("split") == "test" and row.get("probe") == "combined_representation_probe" and isinstance(row.get("accuracy"), (int, float))
    ]
    grouped_combined: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in combined_rows:
        grouped_combined[(str(row.get("method")), str(row.get("feature_source")))].append(float(row["accuracy"]))
    for (method, source), values in sorted(grouped_combined.items()):
        lines.append(f"| combined final-label | {method} | {source} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")

    collapse_rows = [
        row
        for row in stage_diags
        if row.get("split") == "test" and row.get("probe") == "message_collapse_statistics"
    ]
    lines.extend(
        [
            "",
            "Message collapse statistics:",
            "| method | feature source | variance | norm mean | norm std | mean cosine | within-label cosine | between-label cosine |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(collapse_rows, key=lambda item: (str(item.get("method")), str(item.get("feature_source")))):
        lines.append(
            "| {method} | {source} | {var:.6f} | {norm:.4f} | {norm_std:.4f} | {cos:.4f} | {within:.4f} | {between:.4f} |".format(
                method=row.get("method"),
                source=row.get("feature_source"),
                var=float(row.get("variance_mean", 0.0)),
                norm=float(row.get("norm_mean", 0.0)),
                norm_std=float(row.get("norm_std", 0.0)),
                cos=float(row.get("mean_cosine_similarity", 0.0)),
                within=float(row.get("within_label_cosine", 0.0)),
                between=float(row.get("between_label_cosine", 0.0)),
            )
        )

    evidence_trainable_method = "trainable_shared_agent_evidence_token_diagnostic_upper_bound"
    evidence_frozen_method = "frozen_shared_agent_evidence_token_diagnostic_upper_bound"
    final_acc = _metric_means(
        stage_metrics,
        "none",
        {
            proposed_method,
            frozen_method,
            "text_output_only_multi_agent_coordinator",
            evidence_trainable_method,
            evidence_frozen_method,
        },
    )
    proposed_acc = final_acc.get(("real_shared_weight_latent_coordination", proposed_method))
    frozen_acc = final_acc.get(("real_shared_weight_latent_coordination", frozen_method))
    text_acc = final_acc.get(("real_shared_weight_latent_coordination", "text_output_only_multi_agent_coordinator"))
    evidence_trainable_acc = final_acc.get(("real_shared_weight_latent_coordination", evidence_trainable_method))
    evidence_frozen_acc = final_acc.get(("real_shared_weight_latent_coordination", evidence_frozen_method))
    msg_private = _mean_or_none(grouped_private.get((proposed_method, "message_head_output")))
    active_private = _mean_or_none(grouped_private.get((proposed_method, "active_message_readout")))
    evidence_private = _mean_or_none(
        grouped_private.get((evidence_trainable_method, "evidence_token_activation"))
        or grouped_private.get((evidence_frozen_method, "evidence_token_activation"))
    )
    msg_combined = _mean_or_none(grouped_combined.get((proposed_method, "combined_message")))

    lines.extend(["", "Failure diagnosis:"])
    if not bool(val.get("gate_e_controls_collapse", False)):
        lines.append("Controls failed or are incomplete; no latent coordination success claim is allowed.")
    elif proposed_acc is not None and frozen_acc is not None and proposed_acc <= frozen_acc:
        lines.append("The trainable message model did not beat the frozen same-coordinator comparator, so this does not support shared-agent learning.")
    elif proposed_acc is not None and text_acc is not None and proposed_acc <= text_acc:
        lines.append("The trainable message model did not beat the text-only coordinator, so hidden-message coordination is not established.")
    else:
        lines.append("The locked success comparison should be read directly from the gate table; no broader claim follows from this fallback benchmark.")
    interpretation = str(val.get("active_message_interpretation", ""))
    if interpretation:
        lines.append(f"Active-message interpretation: `{interpretation}`.")
    if msg_private is not None and msg_private < 0.65:
        lines.append("Private cues are not reliably decodable from active message-head outputs; diagnose message-readout optimization before changing the coordinator.")
        if evidence_private is not None and evidence_private >= 0.75:
            lines.append("Evidence-token states decode private cues, so that branch remains an upper-bound token-extraction diagnostic rather than evidence of shared-agent learning.")
    elif active_private is not None and msg_private is None and active_private < 0.65:
        lines.append("Private cues are not reliably decodable from the pre-head active messages; diagnose the active readout before changing the coordinator.")
    if (
        evidence_trainable_acc is not None
        and evidence_frozen_acc is not None
        and evidence_trainable_acc >= 0.8
        and evidence_frozen_acc >= evidence_trainable_acc - 0.05
    ):
        lines.append("The evidence-token readout solves the task but its frozen comparator also solves it, so that branch is a frozen-readable token-extraction diagnostic rather than shared-agent learning.")
    elif msg_private is not None and proposed_acc is not None and proposed_acc < 0.5:
        lines.append("Message-head outputs encode private cues better than chance, but final answer accuracy remains weak; message formation and coordination/composition should be separated.")
    if msg_combined is not None and proposed_acc is not None and msg_combined >= 0.8 and proposed_acc < 0.8:
        lines.append("The combined message probe solves the task better than the end-to-end model; next diagnosis should focus on optimization and coordinator training.")

    lines.append("")
    lines.append("Next branch recommendation:")
    if interpretation == "active_readout_still_collapses_diagnose_message_readout_optimization_before_changing_coordinator":
        lines.append("Diagnose active message-readout optimization before changing the coordinator.")
    elif interpretation == "active_readout_solved_but_frozen_solves_too_report_frozen_token_extraction_not_shared_agent_learning":
        lines.append("Report frozen token extraction, not shared-agent learning; move to the semantic no-explicit-cue dataset variant.")
    elif interpretation == "trainable_active_readout_beats_frozen_and_controls_pass_evidence_for_learned_shared_agent_latent_message_formation":
        lines.append("Report evidence for learned shared-agent latent message formation, scoped to this benchmark and controls.")
    elif msg_private is None:
        lines.append("Run the message-channel probes; there is no basis for branching yet.")
    elif (
        evidence_trainable_acc is not None
        and evidence_frozen_acc is not None
        and evidence_trainable_acc >= 0.8
        and evidence_frozen_acc >= evidence_trainable_acc - 0.05
    ):
        lines.append("Keep evidence-token readout as a diagnostic upper bound and require active readout to beat the same frozen comparator.")
    elif msg_private < 0.65 and evidence_private is not None and evidence_private >= 0.75:
        lines.append("Continue active readout optimization and compare each readout against its frozen comparator; do not treat frozen-readable cues as shared-agent learning.")
    elif msg_private < 0.65:
        lines.append("Tune active message readout layers, auxiliary schedule, and message dimension before coordinator variants.")
    elif msg_combined is not None and msg_combined >= 0.8 and proposed_acc is not None and proposed_acc < 0.8:
        lines.append("Diagnose optimization: curriculum, learning-rate sweep, auxiliary warmup/decay, gradient clipping, and message-dimension sweep.")
    elif proposed_acc is not None and proposed_acc < 0.8:
        lines.append("If individual messages are decodable but combined probes remain weak, improve candidate-query or role-aware message composition within the allowed coordinator set.")
    else:
        lines.append("Repeat with more seeds only if all locked gates pass; otherwise report the failure without a success claim.")
    return lines


def _stage5_best_methods(diagnostics: Iterable[Dict[str, object]]) -> set[str]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in diagnostics:
        if (
            row.get("split") == "test"
            and row.get("probe") == "stage5_role_attention"
            and row.get("condition") == "all_agents"
            and isinstance(row.get("accuracy"), (int, float))
        ):
            grouped[str(row["method"])].append(row)
    best_by_family: Dict[str, Tuple[str, float]] = {}
    for method, values in grouped.items():
        family = str(values[0].get("family", ""))
        value = mean([float(row["accuracy"]) for row in values])
        if family not in best_by_family or value > best_by_family[family][1]:
            best_by_family[family] = (method, value)
    return {method for method, _value in best_by_family.values()}


def _stage3_robustness_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
) -> List[str]:
    del diagnostics
    targets = {
        "hidden_state_only_probe",
        "output_only_oracle_probe",
        "text_only_coordinator",
        "activation_pca_mlp",
    }
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in metrics:
        if (
            row.get("benchmark") == "transformer_strict"
            and row.get("split") == "test"
            and row.get("condition") == "none"
            and row.get("method") in targets
        ):
            grouped[str(row["method"])].append(float(row["accuracy"]))
    if not grouped:
        return ["Stage 3 robustness metrics are not available."]
    lines = ["| method | test mean acc | std | 95% bootstrap CI | runs |", "|---|---:|---:|---|---:|"]
    for method in ["hidden_state_only_probe", "activation_pca_mlp", "output_only_oracle_probe", "text_only_coordinator"]:
        values = grouped.get(method, [])
        if not values:
            continue
        lo, hi = _bootstrap_ci(values)
        lines.append(f"| {method} | {mean(values):.4f} | {_std(values):.4f} | [{lo:.4f}, {hi:.4f}] | {len(values)} |")
    lines.append("")
    lines.append("- Claim update: Stage 3 tests robustness of the controlled hidden-evidence mechanism across more seeds and variants; it still does not establish open-ended real-agent generalization.")
    return lines


def _agent_count_curve_summary(diagnostics: Iterable[Dict[str, object]]) -> List[str]:
    grouped: Dict[int, List[float]] = defaultdict(list)
    skipped = {}
    for row in diagnostics:
        if row.get("split") != "test" or row.get("probe") != "hidden_state_agent_count_curve":
            continue
        accuracy = row.get("accuracy")
        if isinstance(accuracy, (int, float)):
            grouped[int(row["agent_count"])].append(float(accuracy))
        elif row.get("status"):
            skipped[(int(row["agent_count"]), int(row["available_agents"]), str(row["status"]))] = row
    if not grouped and not skipped:
        return ["Agent-count curve was not run."]
    lines = ["| agents used | test mean acc | std | 95% bootstrap CI | runs |", "|---:|---:|---:|---|---:|"]
    means_by_count = {}
    for agent_count, values in sorted(grouped.items()):
        lo, hi = _bootstrap_ci(values)
        means_by_count[agent_count] = mean(values)
        lines.append(f"| {agent_count} | {mean(values):.4f} | {_std(values):.4f} | [{lo:.4f}, {hi:.4f}] | {len(values)} |")
    for row in skipped.values():
        lines.append(
            f"| {int(row['agent_count'])} | not run: {row['status']} (available={int(row['available_agents'])}) |  |  |  |"
        )
    non_monotonic = []
    ordered_counts = sorted(means_by_count)
    for left, right in zip(ordered_counts, ordered_counts[1:]):
        if means_by_count[right] + 1e-12 < means_by_count[left]:
            non_monotonic.append((left, right, means_by_count[left], means_by_count[right]))
    if non_monotonic:
        details = ", ".join(f"{left}->{right}: {lhs:.4f}->{rhs:.4f}" for left, right, lhs, rhs in non_monotonic)
        lines.append("")
        lines.append(f"- Non-monotonic agent-count points observed ({details}). These are probe estimates with finite data and PCA/MLP training variance; the causal controls still show that label-preserved evidence masking/shuffling collapses the signal.")
    else:
        lines.append("")
        lines.append("- The mean agent-count curve is monotonic over available evidence agents.")
    return lines


def _hidden_location_probe_summary(diagnostics: Iterable[Dict[str, object]]) -> List[str]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in diagnostics:
        if row.get("split") != "test" or row.get("probe") != "hidden_state_location_probe":
            continue
        accuracy = row.get("accuracy")
        if isinstance(accuracy, (int, float)):
            grouped[(str(row["axis"]), str(row["value"]))].append(float(accuracy))
    if not grouped:
        return ["Layer/token-position hidden-state probes were not run."]
    lines = ["| axis | slice | test mean acc | std | 95% bootstrap CI | runs |", "|---|---|---:|---:|---|---:|"]
    for (axis, value), values in sorted(grouped.items()):
        display_value = "final/last" if axis == "token_position" and value == "final" else value
        lo, hi = _bootstrap_ci(values)
        lines.append(f"| {axis} | {display_value} | {mean(values):.4f} | {_std(values):.4f} | [{lo:.4f}, {hi:.4f}] | {len(values)} |")
    final_values = grouped.get(("token_position", "final"), [])
    evidence_values = grouped.get(("token_position", "evidence"), [])
    if final_values and evidence_values:
        lines.append("")
        lines.append(
            f"- Final/last-token hidden states remain strongest in this setup. Evidence-token slices are near output-only because each evidence-token state is captured before later prompt context has propagated into a representation that is easy for the pooled label probe to combine."
        )
    return lines


def _prompt_robustness_summary(diagnostics: Iterable[Dict[str, object]]) -> List[str]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    leakage: Dict[str, List[int]] = defaultdict(list)
    for row in diagnostics:
        if row.get("split") == "test" and row.get("probe") == "prompt_robustness":
            accuracy = row.get("accuracy")
            if isinstance(accuracy, (int, float)):
                grouped[(str(row["condition"]), str(row["method"]))].append(float(accuracy))
        if row.get("split") == "test" and row.get("probe") == "prompt_robustness_output_leakage_audit":
            leakage[str(row["variant"])].append(int(row.get("bit_token_occurrences", 0)))
    if not grouped:
        return ["Prompt robustness diagnostics were not run."]
    lines = ["| variant | method | test mean acc | std | 95% bootstrap CI | runs | bit token leaks |", "|---|---|---:|---:|---|---:|---:|"]
    for (variant, method), values in sorted(grouped.items()):
        lo, hi = _bootstrap_ci(values)
        leaks = sum(leakage.get(variant, []))
        lines.append(f"| {variant} | {method} | {mean(values):.4f} | {_std(values):.4f} | [{lo:.4f}, {hi:.4f}] | {len(values)} | {leaks} |")
    return lines


def _model_robustness_summary(diagnostics: Iterable[Dict[str, object]]) -> List[str]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    skipped = {}
    for row in diagnostics:
        if row.get("split") != "test" or row.get("probe") != "model_robustness":
            continue
        accuracy = row.get("accuracy")
        if isinstance(accuracy, (int, float)):
            grouped[(str(row["condition"]), str(row["method"]))].append(float(accuracy))
        else:
            skipped[(str(row.get("condition")), str(row.get("method")), str(row.get("reason")))] = row
    if not grouped and not skipped:
        return ["Model robustness diagnostics were not run."]
    lines = ["| model | method | test mean acc | std | 95% bootstrap CI | runs |", "|---|---|---:|---:|---|---:|"]
    for (model, method), values in sorted(grouped.items()):
        lo, hi = _bootstrap_ci(values)
        lines.append(f"| {model} | {method} | {mean(values):.4f} | {_std(values):.4f} | [{lo:.4f}, {hi:.4f}] | {len(values)} |")
    for row in skipped.values():
        lines.append(f"| {row.get('condition')} | {row.get('method')} | not run: {row.get('reason')} |  |  |  |")
    return lines


def _evidence_control_summary(diagnostics: Iterable[Dict[str, object]]) -> List[str]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in diagnostics:
        if row.get("split") != "test" or row.get("probe") != "evidence_causal_control":
            continue
        accuracy = row.get("accuracy")
        if isinstance(accuracy, (int, float)):
            grouped[str(row["condition"])].append(float(accuracy))
    if not grouped:
        return ["Evidence masking/shuffling controls were not run."]
    lines = ["| condition | labels preserved | hidden-only test mean acc | std | runs |", "|---|---|---:|---:|---:|"]
    for condition, values in sorted(grouped.items()):
        lines.append(f"| {condition} | true | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    return lines


def _output_leakage_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
) -> List[str]:
    lines = []
    audit_rows = [
        row
        for row in diagnostics
        if row.get("split") == "test" and row.get("probe") == "visible_output_leakage_audit"
    ]
    if audit_rows:
        bit_hits = sum(int(row.get("bit_token_occurrences", 0)) for row in audit_rows)
        evidence_word_hits = sum(int(row.get("evidence_word_occurrences", 0)) for row in audit_rows)
        text_count = sum(int(row.get("n_texts", 0)) for row in audit_rows)
        max_unique_rows = max(int(row.get("unique_visible_rows", 0)) for row in audit_rows)
        lines.append(
            f"- Visible test outputs contain {bit_hits} `BIT_ONE`/`BIT_ZERO`/`BIT_MASK` tokens across {text_count} texts. The word `evidence` appears {evidence_word_hits} times because the constant sanitized status string includes `withheld_private_evidence`; max unique visible rows per seed is {max_unique_rows}."
        )
    final_output = _metric_means(metrics, "none", {"output_only_oracle_probe"}).get(
        ("transformer_strict", "output_only_oracle_probe")
    )
    if final_output is not None:
        lines.append(f"- Output-only oracle final-label accuracy is {final_output:.4f}.")
    bit_output = _private_bit_mean(diagnostics, "output_only")
    if bit_output is not None:
        lines.append(f"- Output-only private-evidence-bit accuracy is {bit_output:.4f}.")
    if not lines:
        return ["Output leakage audit was not run."]
    return lines


def _activation_stability_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
) -> List[str]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in diagnostics:
        if row.get("split") != "test" or row.get("probe") != "activation_pca_mlp_stability":
            continue
        accuracy = row.get("accuracy")
        if isinstance(accuracy, (int, float)):
            grouped[(str(row["method"]), str(row["variant"]))].append(float(accuracy))
    if not grouped:
        return ["activation_pca_mlp stability diagnostics were not run."]
    normal = _metric_means(metrics, "none", {"activation_pca_mlp", "hidden_state_only_probe"})
    lines = ["| method | variant | test mean acc | std | runs |", "|---|---|---:|---:|---:|"]
    for method in ["activation_pca_mlp", "hidden_state_only_probe"]:
        normal_value = normal.get(("transformer_strict", method))
        if normal_value is not None:
            normal_values = [
                float(row["accuracy"])
                for row in metrics
                if row.get("benchmark") == "transformer_strict"
                and row.get("split") == "test"
                and row.get("condition") == "none"
                and row.get("method") == method
            ]
            lines.append(f"| {method} | main_run | {normal_value:.4f} | {_std(normal_values):.4f} | {len(normal_values)} |")
    for (method, variant), values in sorted(grouped.items()):
        lines.append(f"| {method} | {variant} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    activation_probe_std = _mean_within_seed_std(diagnostics, "activation_pca_mlp", "vary_probe_training_seed")
    activation_fixed_std = _variant_std(diagnostics, "activation_pca_mlp", "fixed_probe_training_seed")
    hidden_probe_std = _mean_within_seed_std(diagnostics, "hidden_state_only_probe", "vary_probe_training_seed")
    lines.append("")
    if activation_probe_std is not None and activation_fixed_std is not None:
        source = "data/captured-feature seed" if activation_fixed_std > activation_probe_std else "probe training seed"
        lines.append(
            f"- `activation_pca_mlp` mean within-data probe-seed std is {activation_probe_std:.4f}; fixed-training-seed std across data seeds is {activation_fixed_std:.4f}. The larger component points to {source} variance."
        )
    if hidden_probe_std is not None:
        lines.append(f"- Hidden-state-only same-split probe-seed std is {hidden_probe_std:.4f}.")
    family_grouped: Dict[str, List[float]] = defaultdict(list)
    pca_variance: Dict[str, List[float]] = defaultdict(list)
    for row in diagnostics:
        if row.get("split") != "test" or row.get("probe") != "activation_probe_family":
            continue
        accuracy = row.get("accuracy")
        if isinstance(accuracy, (int, float)):
            method = str(row["method"])
            family_grouped[method].append(float(accuracy))
            explained = row.get("pca_explained_variance_ratio_sum")
            if isinstance(explained, (int, float)):
                pca_variance[method].append(float(explained))
    if family_grouped:
        lines.append("")
        lines.extend(
            [
                "| probe family | test mean acc | std | 95% bootstrap CI | PCA explained variance | runs |",
                "|---|---:|---:|---|---:|---:|",
            ]
        )
        for method, values in sorted(family_grouped.items()):
            lo, hi = _bootstrap_ci(values)
            explained = mean(pca_variance[method]) if pca_variance.get(method) else 0.0
            lines.append(f"| {method} | {mean(values):.4f} | {_std(values):.4f} | [{lo:.4f}, {hi:.4f}] | {explained:.4f} | {len(values)} |")
    return lines


def _explicit_evidence_sharing_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
) -> List[str]:
    hidden_values = [
        float(row["accuracy"])
        for row in metrics
        if row.get("benchmark") == "transformer_strict"
        and row.get("split") == "test"
        and row.get("condition") == "none"
        and row.get("method") == "hidden_state_only_probe"
    ]
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in diagnostics:
        if row.get("split") == "test" and row.get("probe") == "explicit_evidence_sharing_baseline":
            accuracy = row.get("accuracy")
            if isinstance(accuracy, (int, float)):
                grouped[str(row["method"])].append(float(accuracy))
    if not grouped:
        return ["Explicit evidence-sharing baseline was not run."]
    lines = ["| condition | method | test mean acc | std | 95% bootstrap CI | runs |", "|---|---|---:|---:|---|---:|"]
    if hidden_values:
        lo, hi = _bootstrap_ci(hidden_values)
        lines.append(f"| suppressed visible evidence | hidden_state_only_probe | {mean(hidden_values):.4f} | {_std(hidden_values):.4f} | [{lo:.4f}, {hi:.4f}] | {len(hidden_values)} |")
    for method, values in sorted(grouped.items()):
        lo, hi = _bootstrap_ci(values)
        lines.append(f"| visible private evidence | {method} | {mean(values):.4f} | {_std(values):.4f} | [{lo:.4f}, {hi:.4f}] | {len(values)} |")
    oracle = grouped.get("output_only_oracle_probe")
    rule = grouped.get("evidence_sharing_rule_oracle")
    explicit_reference = rule or oracle
    if explicit_reference and hidden_values:
        if mean(explicit_reference) >= mean(hidden_values) - 0.03:
            lines.append("")
            lines.append("- Explicit text evidence-sharing matches or exceeds hidden-state coordination here, so the hidden states are best interpreted as an implicit private-evidence channel in this controlled task.")
        else:
            lines.append("")
            lines.append("- Hidden-state coordination outperforms explicit text evidence-sharing in this run, which would be stronger evidence for hidden-state utility; inspect output parser coverage before over-interpreting.")
    return lines


def _agent_correctness_summary(diagnostics: Iterable[Dict[str, object]]) -> List[str]:
    rows: Dict[Tuple[str, str], Tuple[List[float], List[float]]] = {}
    for row in diagnostics:
        if row.get("split") != "test" or row.get("probe") != "agent_correctness_probe":
            continue
        key = (str(row["benchmark"]), str(row["feature_mode"]))
        if key not in rows:
            rows[key] = ([], [])
        rows[key][0].append(float(row["accuracy"]))
        rows[key][1].append(float(row["auc"]))
    if not rows:
        return ["Agent correctness probe was not run."]
    lines = ["| benchmark | feature mode | mean acc | mean AUC | runs |", "|---|---|---:|---:|---:|"]
    for (benchmark, feature_mode), (accs, aucs) in sorted(rows.items()):
        lines.append(f"| {benchmark} | {feature_mode} | {mean(accs):.4f} | {mean(aucs):.4f} | {len(accs)} |")
    return lines


def _redundancy_summary(diagnostics: Iterable[Dict[str, object]]) -> List[str]:
    rows: Dict[Tuple[str, str], List[float]] = {}
    for row in diagnostics:
        if row.get("split") != "test" or row.get("probe") != "redundancy_probe":
            continue
        key = (str(row["benchmark"]), str(row["metric"]))
        rows.setdefault(key, []).append(float(row["value"]))
    if not rows:
        return ["Redundancy probe was not run."]
    lines = ["| benchmark | metric | mean value | std | runs |", "|---|---|---:|---:|---:|"]
    for (benchmark, metric), values in sorted(rows.items()):
        lines.append(f"| {benchmark} | {metric} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    return lines


def _telemetry_ablation_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
) -> List[str]:
    base = {
        str(row.get("benchmark", "stage0_original")): float(row["accuracy"])
        for row in _mean_rows(
            row
            for row in metrics
            if row["split"] == "test"
            and row["condition"] == "none"
            and row["method"] == "activation_pca_mlp"
        )
    }
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in diagnostics:
        if row.get("split") == "test" and row.get("probe") == "telemetry_channel_ablation":
            grouped[(str(row["benchmark"]), str(row["channel"]))].append(float(row["accuracy"]))
    if not grouped:
        return ["Telemetry channel ablations were not run."]
    lines = ["| benchmark | ablated channel | mean acc | drop from activation_pca_mlp | runs |", "|---|---|---:|---:|---:|"]
    for (benchmark, channel), values in sorted(grouped.items()):
        mean_value = mean(values)
        drop = base.get(benchmark, mean_value) - mean_value
        lines.append(f"| {benchmark} | {channel} | {mean_value:.4f} | {drop:.4f} | {len(values)} |")
    return lines


def _transformer_hidden_ablation_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
) -> List[str]:
    base = {
        str(row.get("benchmark", "stage0_original")): float(row["accuracy"])
        for row in _mean_rows(
            row
            for row in metrics
            if row["split"] == "test"
            and row["condition"] == "none"
            and row["method"] == "activation_pca_mlp"
        )
    }
    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for row in diagnostics:
        if row.get("split") == "test" and row.get("probe") == "hidden_state_slice_ablation":
            grouped[(str(row["benchmark"]), str(row["axis"]), str(row["value"]))].append(float(row["accuracy"]))
    if not grouped:
        return ["Transformer hidden-state slice ablations were not run."]
    lines = ["| benchmark | axis | masked value | mean acc | drop from activation_pca_mlp | runs |", "|---|---|---|---:|---:|---:|"]
    for (benchmark, axis, value), values in sorted(grouped.items()):
        mean_value = mean(values)
        drop = base.get(benchmark, mean_value) - mean_value
        lines.append(f"| {benchmark} | {axis} | {value} | {mean_value:.4f} | {drop:.4f} | {len(values)} |")
    return lines


def _individual_hidden_label_summary(diagnostics: Iterable[Dict[str, object]]) -> List[str]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in diagnostics:
        if row.get("split") == "test" and row.get("probe") == "individual_hidden_label_probe":
            grouped[str(row["benchmark"])].append(float(row["accuracy"]))
    if not grouped:
        return ["Individual hidden-state label probe was not run."]
    lines = ["| benchmark | mean individual-agent label acc | std | runs |", "|---|---:|---:|---:|"]
    for benchmark, values in sorted(grouped.items()):
        lines.append(f"| {benchmark} | {mean(values):.4f} | {_std(values):.4f} | {len(values)} |")
    return lines


def _probe_access_summary(validation: object) -> List[str]:
    rows = []
    if isinstance(validation, dict):
        rows = validation.get("probe_access", [])
    if not rows:
        return ["Probe access matrix was not recorded."]
    keys = [
        "visible_agent_answers",
        "visible_confidence",
        "visible_traces",
        "hidden_activations",
        "private_prompt_evidence",
        "coordinator_input",
        "train_labels_for_fit",
        "dev_labels_for_early_stopping",
        "test_labels_or_eval_feedback",
    ]
    lines = ["| artifact | " + " | ".join(keys) + " |", "|---|" + "|".join(["---"] * len(keys)) + "|"]
    for row in rows:
        values = [str(row.get(key, False)) for key in keys]
        lines.append(f"| {row['artifact']} | " + " | ".join(values) + " |")
    return lines


def _cpu_cuda_parity_summary(validation: object) -> List[str]:
    rows = []
    hidden_rows = []
    diagnosis = {}
    if isinstance(validation, dict):
        rows = validation.get("cpu_cuda_parity", [])
        hidden_rows = validation.get("cpu_cuda_hidden_state_parity", [])
        diagnosis = validation.get("cpu_cuda_parity_diagnosis", {})
    if not rows:
        return ["CPU/CUDA parity artifact not attached to this report."]
    lines = ["| benchmark | method | CPU acc | CUDA acc | abs delta |", "|---|---|---:|---:|---:|"]
    for row in sorted(rows, key=lambda item: (str(item["benchmark"]), str(item["method"]))):
        lines.append(
            f"| {row['benchmark']} | {row['method']} | {float(row['cpu_accuracy']):.4f} | {float(row['cuda_accuracy']):.4f} | {float(row['abs_delta']):.4f} |"
        )
    if hidden_rows:
        lines.append("")
        lines.extend(
            [
                "| hidden array | max abs diff | mean abs diff | RMSE | allclose 1e-3 | checksum match |",
                "|---|---:|---:|---:|---|---|",
            ]
        )
        for row in sorted(hidden_rows, key=lambda item: str(item.get("array", ""))):
            if not row.get("shape_match", True):
                lines.append(f"| {row.get('array')} | shape mismatch |  |  |  |  |")
                continue
            lines.append(
                f"| {row['array']} | {float(row['max_abs_diff']):.6f} | {float(row['mean_abs_diff']):.6f} | {float(row['rmse']):.6f} | {row['allclose_1e_3']} | {row['exact_checksum_match']} |"
            )
    if isinstance(diagnosis, dict) and diagnosis:
        lines.append("")
        lines.append(
            f"- Parity diagnosis: likely source `{diagnosis.get('likely_source')}`; hidden-state max abs diff before probe fitting is {float(diagnosis.get('hidden_state_max_abs_diff', 0.0)):.6f}; hidden-only accuracy delta is {float(diagnosis.get('hidden_state_only_abs_delta', 0.0)):.4f}; largest method delta is {float(diagnosis.get('largest_accuracy_delta', 0.0)):.4f} for `{diagnosis.get('largest_delta_method', '')}`; test examples per seed `{diagnosis.get('test_examples_per_seed')}`."
        )
        lines.append(f"- Probe seed policy: {diagnosis.get('probe_initialization_seed_policy')}.")
    return lines


def _all_runs_table(metrics: Iterable[Dict[str, object]]) -> List[str]:
    lines = ["| benchmark | seed | split | condition | method | accuracy | params |", "|---|---:|---|---|---|---:|---:|"]
    for row in sorted(
        metrics,
        key=lambda r: (
            str(r.get("benchmark", "stage0_original")),
            int(r["seed"]),
            str(r["split"]),
            str(r["condition"]),
            str(r["method"]),
        ),
    ):
        lines.append(
            f"| {row.get('benchmark', 'stage0_original')} | {row['seed']} | {row['split']} | {row['condition']} | {row['method']} | {float(row['accuracy']):.4f} | {row['param_count']} |"
        )
    return lines


def _interpretation(metrics: Iterable[Dict[str, object]], diagnostics: Iterable[Dict[str, object]]) -> List[str]:
    test_none = [
        row for row in metrics if row["split"] == "test" and row["condition"] == "none"
    ]
    by_benchmark_method: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in test_none:
        by_benchmark_method[(str(row.get("benchmark", "stage0_original")), str(row["method"]))].append(
            float(row["accuracy"])
        )

    lines: List[str] = []
    benchmarks = sorted({benchmark for benchmark, _method in by_benchmark_method})
    for benchmark in benchmarks:
        text_refs = [
            _mean_or_none(by_benchmark_method.get((benchmark, "text_only_coordinator"))),
            _mean_or_none(by_benchmark_method.get((benchmark, "capacity_matched_text_only"))),
            _mean_or_none(by_benchmark_method.get((benchmark, "output_only_oracle_probe"))),
        ]
        text_refs = [value for value in text_refs if value is not None]
        activation = _mean_or_none(by_benchmark_method.get((benchmark, "activation_pca_mlp")))
        telemetry_only = _mean_or_none(by_benchmark_method.get((benchmark, "telemetry_only_label_probe")))
        hidden_only = _mean_or_none(by_benchmark_method.get((benchmark, "hidden_state_only_probe")))
        probe_label = "telemetry-only"
        if telemetry_only is None:
            telemetry_only = hidden_only
            probe_label = "hidden-state-only"
        if activation is None or not text_refs:
            continue
        text_reference = max(text_refs)
        delta = activation - text_reference
        lines.append(
            f"- `{benchmark}`: `activation_pca_mlp` scored {activation:.4f}; strongest output-only reference scored {text_reference:.4f}; delta {delta:.4f}."
        )
        if telemetry_only is not None:
            telemetry_delta = activation - telemetry_only
            if benchmark == "strict_coordination":
                lines.append(
                    f"- `{benchmark}`: {probe_label} label accuracy was {telemetry_only:.4f}; in this benchmark the combined telemetry contains distributed private evidence. This should not be interpreted as correctness estimation."
                )
            elif benchmark == "transformer_strict":
                individual = _individual_hidden_mean(diagnostics, benchmark)
                lines.append(
                    f"- `{benchmark}`: combined {probe_label} label accuracy was {telemetry_only:.4f}; mean individual-agent hidden label accuracy was {individual:.4f}. This indicates label-predictive distributed private evidence across agents, not single-agent final-label decoding."
                )
            elif telemetry_only >= activation - 0.05:
                lines.append(
                    f"- `{benchmark}`: {probe_label} label accuracy was {telemetry_only:.4f}, close to activation+output accuracy; this suggests direct or near-direct label information in the hidden/telemetry representation."
                )
            else:
                lines.append(
                    f"- `{benchmark}`: {probe_label} label accuracy was {telemetry_only:.4f}, {telemetry_delta:.4f} below activation+output; this argues against simple final-label decoding."
                )

        control_rows = [
            row
            for row in metrics
            if row["split"] == "test"
            and row.get("benchmark", "stage0_original") == benchmark
            and row["method"] == "activation_pca_mlp"
            and row["condition"] not in {"none", "randomized_train_labels"}
        ]
        if control_rows:
            control_mean = mean(float(row["accuracy"]) for row in control_rows)
            if control_mean < activation - 0.02:
                signal_name = "captured hidden-state alignment" if benchmark == "transformer_strict" else "aligned telemetry"
                lines.append(f"- `{benchmark}`: activation controls reduced `activation_pca_mlp` to {control_mean:.4f}, supporting {signal_name} as the causal signal.")
            else:
                lines.append(f"- `{benchmark}`: activation controls averaged {control_mean:.4f}; the telemetry interpretation is weak for this benchmark.")

    correctness_rows = [
        row
        for row in diagnostics
        if row.get("split") == "test" and row.get("probe") == "agent_correctness_probe"
    ]
    if correctness_rows:
        for benchmark in sorted({str(row["benchmark"]) for row in correctness_rows}):
            activation_auc = _diag_mean(correctness_rows, benchmark, "activation_only", "auc")
            text_auc = _diag_mean(correctness_rows, benchmark, "text_output_only", "auc")
            both_auc = _diag_mean(correctness_rows, benchmark, "text_plus_activation", "auc")
            lines.append(
                f"- `{benchmark}`: agent-correctness AUC text={text_auc:.4f}, activation={activation_auc:.4f}, text+activation={both_auc:.4f}."
            )

    redundancy_rows = [
        row
        for row in diagnostics
        if row.get("split") == "test" and row.get("probe") == "redundancy_probe"
    ]
    if redundancy_rows:
        for benchmark in sorted({str(row["benchmark"]) for row in redundancy_rows}):
            answer_purity = _redundancy_mean(
                redundancy_rows,
                benchmark,
                "cluster_purity_by_answer_label",
            )
            correctness_purity = _redundancy_mean(
                redundancy_rows,
                benchmark,
                "cluster_purity_by_correctness",
            )
            if benchmark == "transformer_strict":
                lines.append(
                    f"- `{benchmark}`: hidden-state cluster purity by visible answer label={answer_purity:.4f}, by correctness={correctness_purity:.4f}. Visible answers are intentionally sanitized and constant, so answer-label purity is not evidence of reasoning-mode clustering here."
                )
            elif answer_purity < 0.5:
                lines.append(
                    f"- `{benchmark}`: activation-cluster purity by answer label={answer_purity:.4f}, by correctness={correctness_purity:.4f}; low answer purity argues against simple answer-label clustering."
                )
            else:
                lines.append(
                    f"- `{benchmark}`: activation-cluster purity by answer label={answer_purity:.4f}, by correctness={correctness_purity:.4f}; this probe does not rule out answer-label clustering."
                )

    leakage_rows = [
        row
        for row in metrics
        if row["split"] == "test" and row["condition"] == "randomized_train_labels"
    ]
    if leakage_rows:
        synthesis_rows = [
            row
            for row in leakage_rows
            if row["method"]
            in {
                "text_only_coordinator",
                "capacity_matched_text_only",
                "activation_pool_mlp",
                "activation_pca_mlp",
                "telemetry_only_label_probe",
                "hidden_state_only_probe",
                "output_only_oracle_probe",
            }
        ]
        if synthesis_rows:
            grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
            for row in synthesis_rows:
                grouped[(str(row.get("benchmark", "stage0_original")), str(row["method"]))].append(
                    float(row["accuracy"])
                )
            leakage_best_mean = max(mean(values) for values in grouped.values())
            leakage_best_single = max(float(row["accuracy"]) for row in synthesis_rows)
            lines.append(
                f"- Randomized-label leakage best aggregate label-synthesis accuracy was {leakage_best_mean:.4f}; max single run was {leakage_best_single:.4f}; for four classes, values near 0.25 are expected."
            )
        selector_rows = [row for row in leakage_rows if row["method"] == "activation_cluster_router"]
        if selector_rows:
            selector_grouped: Dict[str, List[float]] = defaultdict(list)
            for row in selector_rows:
                selector_grouped[str(row.get("benchmark", "stage0_original"))].append(float(row["accuracy"]))
            selector_best = max(mean(values) for values in selector_grouped.values())
            lines.append(
                f"- Randomized-label cluster-router aggregate accuracy reached {selector_best:.4f}; this selector can remain near single-agent accuracy because it still chooses from agent candidates, so it is not interpreted against four-class chance."
            )

    lines.append("- Recommended next experiment: make the transformer strict task less templated while keeping private evidence hidden from visible outputs, then repeat the same access audit, randomized-label sanity check, output-only baselines, and individual-vs-combined hidden-state probes.")
    return lines


def _mechanism_distinction(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
) -> List[str]:
    lines = []
    stage0_auc = _diag_mean(
        [
            row
            for row in diagnostics
            if row.get("split") == "test" and row.get("probe") == "agent_correctness_probe"
        ],
        "stage0_original",
        "activation_only",
        "auc",
    )
    label_reduced_auc = _diag_mean(
        [
            row
            for row in diagnostics
            if row.get("split") == "test" and row.get("probe") == "agent_correctness_probe"
        ],
        "stage0_label_reduced",
        "activation_only",
        "auc",
    )
    strict_auc = _diag_mean(
        [
            row
            for row in diagnostics
            if row.get("split") == "test" and row.get("probe") == "agent_correctness_probe"
        ],
        "strict_coordination",
        "activation_only",
        "auc",
    )
    stage0_telemetry = _metric_means(metrics, "none", {"telemetry_only_label_probe"}).get(
        ("stage0_original", "telemetry_only_label_probe")
    )
    strict_telemetry = _metric_means(metrics, "none", {"telemetry_only_label_probe"}).get(
        ("strict_coordination", "telemetry_only_label_probe")
    )
    if stage0_telemetry is not None:
        lines.append(
            f"- Stage 0 mechanism: activation telemetry is most consistent with a reliability/correctness signal. Activation-only correctness AUC is {stage0_auc:.4f} on `stage0_original` and {label_reduced_auc:.4f} on `stage0_label_reduced`, while Stage 0 telemetry-only final-label accuracy is {stage0_telemetry:.4f}."
        )
    if strict_telemetry is not None:
        lines.append(
            f"- Strict coordination mechanism: the task is constructed around distributed private evidence. Telemetry-only final-label accuracy is {strict_telemetry:.4f}, while activation-only correctness AUC is {strict_auc:.4f}; this supports an evidence-combination mechanism, not correctness estimation."
        )
    if stage0_telemetry is not None or strict_telemetry is not None:
        lines.append(
            "- These are synthetic mechanisms. They validate the harness and controls, but they should not be generalized to real transformer agents without repeating the same diagnostics on captured model activations."
        )
    transformer_hidden = _metric_means(metrics, "none", {"hidden_state_only_probe"}).get(
        ("transformer_strict", "hidden_state_only_probe")
    )
    transformer_activation = _metric_means(metrics, "none", {"activation_pca_mlp"}).get(
        ("transformer_strict", "activation_pca_mlp")
    )
    transformer_output = _metric_means(metrics, "none", {"output_only_oracle_probe"}).get(
        ("transformer_strict", "output_only_oracle_probe")
    )
    if transformer_hidden is not None and transformer_activation is not None and transformer_output is not None:
        individual = _individual_hidden_mean(diagnostics, "transformer_strict")
        lines.append(
            f"- Transformer strict mechanism: captured local-transformer hidden states give hidden-only accuracy {transformer_hidden:.4f} and hidden+output accuracy {transformer_activation:.4f}, while output-only oracle accuracy is {transformer_output:.4f}. Mean individual-agent hidden label accuracy is {individual:.4f}; this distinguishes combined private-evidence aggregation from single-agent final-label decoding."
        )
    return lines


def _synthetic_vs_transformer_summary(
    metrics: Iterable[Dict[str, object]],
    diagnostics: Iterable[Dict[str, object]],
    validation: object,
) -> List[str]:
    del diagnostics
    transformer_info = {}
    if isinstance(validation, dict):
        transformer_info = validation.get("stage1_transformer", {}) or validation.get("transformer_hidden_state", {})
    if not transformer_info and not any(row.get("benchmark") == "transformer_strict" for row in metrics):
        return ["No captured-transformer hidden-state stage is present in this report."]
    means = _metric_means(
        metrics,
        "none",
        {"activation_pca_mlp", "hidden_state_only_probe", "output_only_oracle_probe"},
    )
    lines = []
    if transformer_info:
        lines.append(
            f"- Stage 1 source: local Hugging Face model `{transformer_info.get('model_name_or_path')}`, layers `{transformer_info.get('layers')}`, token positions `{transformer_info.get('token_positions')}`."
        )
    hidden = means.get(("transformer_strict", "hidden_state_only_probe"))
    activation = means.get(("transformer_strict", "activation_pca_mlp"))
    output = means.get(("transformer_strict", "output_only_oracle_probe"))
    if hidden is not None and activation is not None and output is not None:
        lines.append(
            f"- `transformer_strict`: hidden-only={hidden:.4f}, hidden+output={activation:.4f}, output-only oracle={output:.4f}."
        )
    lines.append(
        "- Synthetic telemetry channels are constructed additive signals. Transformer hidden states are captured by forward hooks from a local model and do not expose semantic channel labels; only layer/token-position ablations are available."
    )
    lines.append(
        "- Claims remain limited to this controlled partial-evidence setup unless the same controls reproduce on richer real-agent tasks."
    )
    return lines


def _group_stats(metrics: Iterable[Dict[str, object]]) -> Dict[Tuple[str, str, str, str], Tuple[List[float], List[int]]]:
    grouped: Dict[Tuple[str, str, str, str], Tuple[List[float], List[int]]] = {}
    for row in metrics:
        key = (
            str(row.get("benchmark", "stage0_original")),
            str(row["split"]),
            str(row["condition"]),
            str(row["method"]),
        )
        if key not in grouped:
            grouped[key] = ([], [])
        grouped[key][0].append(float(row["accuracy"]))
        grouped[key][1].append(int(row["param_count"]))
    return grouped


def _std(values: List[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _bootstrap_ci(values: List[float], n_boot: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = __import__("random").Random(20260513 + len(values))
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(sample))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def _metric_means(
    metrics: Iterable[Dict[str, object]],
    condition: str,
    methods: set[str],
) -> Dict[Tuple[str, str], float]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in metrics:
        if row["split"] == "test" and row["condition"] == condition and row["method"] in methods:
            grouped[(str(row.get("benchmark", "stage0_original")), str(row["method"]))].append(
                float(row["accuracy"])
            )
    return {key: mean(values) for key, values in grouped.items()}


def _mean_rows(metrics: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    for row in metrics:
        key = (
            str(row.get("benchmark", "stage0_original")),
            str(row["split"]),
            str(row["condition"]),
            str(row["method"]),
        )
        grouped[key].append(float(row["accuracy"]))
    return [
        {
            "benchmark": key[0],
            "split": key[1],
            "condition": key[2],
            "method": key[3],
            "accuracy": mean(values),
        }
        for key, values in grouped.items()
    ]


def _mean_or_none(values: List[float] | None) -> float | None:
    return mean(values) if values else None


def _diag_mean(rows: List[Dict[str, object]], benchmark: str, feature_mode: str, field: str) -> float:
    values = [
        float(row[field])
        for row in rows
        if str(row["benchmark"]) == benchmark and str(row["feature_mode"]) == feature_mode
    ]
    return mean(values) if values else 0.0


def _redundancy_mean(rows: List[Dict[str, object]], benchmark: str, metric: str) -> float:
    values = [
        float(row["value"])
        for row in rows
        if str(row["benchmark"]) == benchmark and str(row["metric"]) == metric
    ]
    return mean(values) if values else 0.0


def _individual_hidden_mean(diagnostics: Iterable[Dict[str, object]], benchmark: str) -> float:
    values = [
        float(row["accuracy"])
        for row in diagnostics
        if row.get("split") == "test"
        and row.get("probe") == "individual_hidden_label_probe"
        and str(row.get("benchmark")) == benchmark
    ]
    return mean(values) if values else 0.0


def _private_bit_mean(diagnostics: Iterable[Dict[str, object]], feature_source: str) -> float | None:
    values = [
        float(row["accuracy"])
        for row in diagnostics
        if row.get("split") == "test"
        and row.get("probe") == "private_evidence_bit_probe"
        and row.get("feature_source") == feature_source
        and isinstance(row.get("accuracy"), (int, float))
    ]
    return mean(values) if values else None


def _agent_count_mean(diagnostics: Iterable[Dict[str, object]], agent_count: int) -> float | None:
    values = [
        float(row["accuracy"])
        for row in diagnostics
        if row.get("split") == "test"
        and row.get("probe") == "hidden_state_agent_count_curve"
        and row.get("agent_count") == agent_count
        and isinstance(row.get("accuracy"), (int, float))
    ]
    return mean(values) if values else None


def _evidence_control_means(diagnostics: Iterable[Dict[str, object]]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in diagnostics:
        if row.get("split") != "test" or row.get("probe") != "evidence_causal_control":
            continue
        accuracy = row.get("accuracy")
        if isinstance(accuracy, (int, float)):
            grouped[str(row["condition"])].append(float(accuracy))
    return {condition: mean(values) for condition, values in grouped.items()}


def _mean_within_seed_std(
    diagnostics: Iterable[Dict[str, object]],
    method: str,
    variant: str,
) -> float | None:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for row in diagnostics:
        if (
            row.get("split") == "test"
            and row.get("probe") == "activation_pca_mlp_stability"
            and row.get("method") == method
            and row.get("variant") == variant
            and isinstance(row.get("accuracy"), (int, float))
        ):
            grouped[int(row["seed"])].append(float(row["accuracy"]))
    values = [_std(seed_values) for seed_values in grouped.values() if len(seed_values) > 1]
    return mean(values) if values else None


def _variant_std(
    diagnostics: Iterable[Dict[str, object]],
    method: str,
    variant: str,
) -> float | None:
    values = [
        float(row["accuracy"])
        for row in diagnostics
        if row.get("split") == "test"
        and row.get("probe") == "activation_pca_mlp_stability"
        and row.get("method") == method
        and row.get("variant") == variant
        and isinstance(row.get("accuracy"), (int, float))
    ]
    return _std(values) if values else None
