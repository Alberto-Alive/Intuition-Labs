"""Path-count ablation for the E10 diffusion evidence stack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
VARIANT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(VARIANT_DIR))

from eval_common import (  # noqa: E402
    COMMON_MONOTONICITY_CHECKS,
    apply_trace_perturbation,
    compute_monotonicity_violation_rate,
    compute_success_safety_metrics,
    summarize_scalar_series,
)
from experiments.DIGIT.Extrapolation.e10.e10_falsified.extrapolation.config import Config  # noqa: E402
from experiments.DIGIT.Extrapolation.e10.e10_falsified.extrapolation.falsification import (  # noqa: E402
    SimpleVocabulary,
    build_loader,
    build_model,
    compute_model_threshold_sensitivity,
    compute_multiclass_metrics,
    load_trace_metadata,
    load_trace_split,
    set_global_seed,
    slice_records,
    write_json,
)


MODE_SPECS = {
    "multi_path": {
        "description": "Current multi-path diffusion with the checkpoint's native path count.",
        "config_overrides": {},
        "deterministic_cache": False,
    },
    "single_path_stochastic": {
        "description": "Single-path diffusion with fresh noise each run.",
        "config_overrides": {"num_random_paths": 1},
        "deterministic_cache": False,
    },
    "single_path_deterministic": {
        "description": "Single-path diffusion with cached t/eps replayed across repeats.",
        "config_overrides": {"num_random_paths": 1},
        "deterministic_cache": True,
    },
}


def _detach_diffusion_state(state: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
    if state is None:
        return None
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in state.items()}


def _stack_series(series: list[torch.Tensor]) -> torch.Tensor:
    if not series:
        return torch.empty(0)
    return torch.cat(series, dim=0)


def _run_repeat(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    records,
    *,
    diffusion_cache: list[dict[str, torch.Tensor] | None] | None = None,
    capture_cache: bool = False,
) -> dict[str, object]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    outcome_logits: list[torch.Tensor] = []
    outcome_probs: list[torch.Tensor] = []
    primitive_series: dict[str, list[torch.Tensor]] = {
        "stability_cert": [],
        "support_cert": [],
        "success_guard": [],
        "success_cert": [],
        "ambiguity_cert": [],
        "commitment_depth": [],
        "success_base": [],
        "failure_cert": [],
    }
    cached_states: list[dict[str, torch.Tensor] | None] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            queries, trace_inputs, evidence_targets, prim_targets, target_ids = batch
            queries = queries.to(device)
            trace_inputs = trace_inputs.to(device)
            prim_targets = prim_targets.to(device)
            target_ids = target_ids.to(device)

            diffusion_override = None
            if diffusion_cache is not None and batch_idx < len(diffusion_cache):
                diffusion_override = diffusion_cache[batch_idx]

            out = model(
                queries,
                trace_inputs,
                target_ids,
                bottleneck_mode="hard",
                tau=1.0,
                skip_decoder=True,
                diffusion_override=diffusion_override,
            )
            primitives = out["primitives"]
            probs = torch.softmax(primitives.outcome_logits, dim=-1)

            predictions.append(
                torch.stack(
                    [
                        primitives.trajectory_shape_logits.argmax(dim=-1),
                        primitives.attention_pattern_logits.argmax(dim=-1),
                        primitives.confidence_logits.argmax(dim=-1),
                        primitives.outcome_logits.argmax(dim=-1),
                    ],
                    dim=1,
                ).cpu()
            )
            targets.append(prim_targets.cpu())
            outcome_logits.append(primitives.outcome_logits.detach().cpu())
            outcome_probs.append(probs.detach().cpu())
            primitive_series["stability_cert"].append(primitives.stability_cert.detach().cpu())
            primitive_series["support_cert"].append(primitives.support_cert.detach().cpu())
            primitive_series["success_guard"].append(primitives.success_guard.detach().cpu())
            primitive_series["success_cert"].append(primitives.success_cert.detach().cpu())
            primitive_series["ambiguity_cert"].append(primitives.ambiguity_cert.detach().cpu())
            primitive_series["commitment_depth"].append(primitives.commitment_depth.detach().cpu())
            primitive_series["success_base"].append(primitives.success_base.detach().cpu())
            primitive_series["failure_cert"].append(primitives.failure_cert.detach().cpu())

            if capture_cache:
                cached_states.append(_detach_diffusion_state(out.get("diffusion_state")))

    predictions_tensor = torch.cat(predictions, dim=0)
    targets_tensor = torch.cat(targets, dim=0)
    outcome_probs_tensor = torch.cat(outcome_probs, dim=0)
    outcome_logits_tensor = torch.cat(outcome_logits, dim=0)
    metrics = compute_multiclass_metrics(predictions_tensor, targets_tensor)
    is_correct = np.array([int(bool(record.get("is_correct", False))) for record in records], dtype=np.int64)
    metrics.update(
        compute_success_safety_metrics(
            outcome_probabilities=outcome_probs_tensor.numpy(),
            outcome_predictions=predictions_tensor[:, 3].numpy(),
            outcome_targets=targets_tensor[:, 3].numpy(),
            is_correct=is_correct,
        )
    )
    threshold = compute_model_threshold_sensitivity(outcome_probs_tensor)
    metrics["max_outcome_share_swing"] = float(threshold["max_outcome_share_swing"])
    metrics["threshold_sensitivity"] = threshold

    primitive_means = {
        key: float(_stack_series(series).mean().item()) if series else float("nan")
        for key, series in primitive_series.items()
    }

    bundle = {
        "predictions": predictions_tensor,
        "targets": targets_tensor,
        "outcome_logits": outcome_logits_tensor,
        "outcome_probs": outcome_probs_tensor,
        "primitive_series": {key: _stack_series(series) for key, series in primitive_series.items()},
    }
    if capture_cache:
        bundle["diffusion_cache"] = cached_states
    return {
        "metrics": metrics,
        "primitive_means": primitive_means,
        "bundle": bundle,
    }


def _audit_monotonicity(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    diffusion_cache: list[dict[str, torch.Tensor] | None] | None = None,
) -> dict[str, object]:
    perturbation_kinds = list(COMMON_MONOTONICITY_CHECKS)
    first_batch = next(iter(loader))
    trace_input_dim = first_batch[1].shape[-1]
    if trace_input_dim > 11:
        perturbation_kinds.append("higher_variation_ratio")

    success_violation_values: list[float] = []
    commitment_violation_values: list[float] = []
    support_violation_values: list[float] = []
    guard_violation_values: list[float] = []
    success_cert_violation_values: list[float] = []
    ambiguity_increase_values: list[float] = []
    delta_sums: dict[str, dict[str, list[float]]] = {
        kind: {
            "success_prob": [],
            "commitment_depth": [],
            "support_cert": [],
            "success_guard": [],
            "success_cert": [],
            "ambiguity_cert": [],
        }
        for kind in perturbation_kinds
    }

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            queries, trace_inputs, _, _, target_ids = batch
            queries = queries.to(device)
            trace_inputs = trace_inputs.to(device)
            target_ids = target_ids.to(device)

            diffusion_override = None
            if diffusion_cache is not None and batch_idx < len(diffusion_cache):
                diffusion_override = diffusion_cache[batch_idx]

            base_out = model(
                queries,
                trace_inputs,
                target_ids,
                bottleneck_mode="hard",
                tau=1.0,
                skip_decoder=True,
                diffusion_override=diffusion_override,
            )
            base_primitives = base_out["primitives"]
            base_probs = torch.softmax(base_primitives.outcome_logits, dim=-1)
            base_success = base_probs[:, 0]
            base_commitment = base_primitives.commitment_depth
            base_support = base_primitives.support_cert
            base_guard = base_primitives.success_guard
            base_success_cert = base_primitives.success_cert
            base_ambiguity = base_primitives.ambiguity_cert
            base_diffusion = base_out.get("diffusion_state")

            for kind in perturbation_kinds:
                perturbed_inputs = apply_trace_perturbation(trace_inputs, kind)
                perturbed_out = model(
                    queries,
                    perturbed_inputs,
                    target_ids,
                    bottleneck_mode="hard",
                    tau=1.0,
                    skip_decoder=True,
                    diffusion_override=base_diffusion,
                )
                perturbed_primitives = perturbed_out["primitives"]
                perturbed_probs = torch.softmax(perturbed_primitives.outcome_logits, dim=-1)
                perturbed_success = perturbed_probs[:, 0]
                perturbed_commitment = perturbed_primitives.commitment_depth
                perturbed_support = perturbed_primitives.support_cert
                perturbed_guard = perturbed_primitives.success_guard
                perturbed_success_cert = perturbed_primitives.success_cert
                perturbed_ambiguity = perturbed_primitives.ambiguity_cert

                success_violation_values.append(
                    compute_monotonicity_violation_rate(base_success, perturbed_success)
                )
                commitment_violation_values.append(
                    compute_monotonicity_violation_rate(base_commitment, perturbed_commitment)
                )
                support_violation_values.append(
                    compute_monotonicity_violation_rate(base_support, perturbed_support)
                )
                guard_violation_values.append(
                    compute_monotonicity_violation_rate(base_guard, perturbed_guard)
                )
                success_cert_violation_values.append(
                    compute_monotonicity_violation_rate(base_success_cert, perturbed_success_cert)
                )
                ambiguity_increase_values.append(
                    compute_monotonicity_violation_rate(base_ambiguity, perturbed_ambiguity)
                )

                delta_sums[kind]["success_prob"].append(float((perturbed_success - base_success).mean().item()))
                delta_sums[kind]["commitment_depth"].append(
                    float((perturbed_commitment - base_commitment).mean().item())
                )
                delta_sums[kind]["support_cert"].append(float((perturbed_support - base_support).mean().item()))
                delta_sums[kind]["success_guard"].append(float((perturbed_guard - base_guard).mean().item()))
                delta_sums[kind]["success_cert"].append(
                    float((perturbed_success_cert - base_success_cert).mean().item())
                )
                delta_sums[kind]["ambiguity_cert"].append(
                    float((perturbed_ambiguity - base_ambiguity).mean().item())
                )

    by_perturbation = {
        kind: {
            f"{metric}_mean_delta": summarize_scalar_series(values)["mean"]
            for metric, values in metric_values.items()
            if values
        }
        for kind, metric_values in delta_sums.items()
    }

    return {
        "num_checks": int(len(success_violation_values)),
        "supported_checks": perturbation_kinds,
        "success_prob_monotonicity_violation_rate": float(np.mean(success_violation_values))
        if success_violation_values
        else float("nan"),
        "commitment_monotonicity_violation_rate": float(np.mean(commitment_violation_values))
        if commitment_violation_values
        else float("nan"),
        "support_cert_violation_rate": float(np.mean(support_violation_values)) if support_violation_values else float("nan"),
        "success_guard_violation_rate": float(np.mean(guard_violation_values)) if guard_violation_values else float("nan"),
        "success_cert_violation_rate": float(np.mean(success_cert_violation_values))
        if success_cert_violation_values
        else float("nan"),
        "ambiguity_increase_rate": float(np.mean(ambiguity_increase_values)) if ambiguity_increase_values else float("nan"),
        "overall_monotonicity_violation_rate": float(
            np.mean([np.mean(success_violation_values), np.mean(commitment_violation_values)])
        )
        if success_violation_values and commitment_violation_values
        else float("nan"),
        "by_perturbation": by_perturbation,
    }


def _prediction_disagreement_rate(repeat_predictions: list[torch.Tensor]) -> float:
    if len(repeat_predictions) <= 1:
        return 0.0
    stacked = torch.stack(repeat_predictions, dim=0).cpu().numpy()
    outcome_preds = stacked[:, :, 3]
    mode_counts = []
    for column in outcome_preds.T:
        counts = np.bincount(column.astype(np.int64), minlength=3)
        mode_counts.append(float(counts.max()))
    disagreement = 1.0 - np.asarray(mode_counts, dtype=np.float64) / float(outcome_preds.shape[0])
    return float(np.mean(disagreement))


def _seed_variance_summary(repeat_results: list[dict[str, object]]) -> dict[str, object]:
    if not repeat_results:
        return {}

    repeat_metrics = [result["metrics"] for result in repeat_results]
    repeat_bundles = [result["bundle"] for result in repeat_results]
    repeat_predictions = [bundle["predictions"] for bundle in repeat_bundles]
    repeat_outcome_probs = [bundle["outcome_probs"] for bundle in repeat_bundles]
    repeat_commitment = [bundle["primitive_series"]["commitment_depth"] for bundle in repeat_bundles]
    repeat_success_cert = [bundle["primitive_series"]["success_cert"] for bundle in repeat_bundles]
    repeat_success_guard = [bundle["primitive_series"]["success_guard"] for bundle in repeat_bundles]

    success_prob_std = float(torch.stack([probs[:, 0] for probs in repeat_outcome_probs], dim=0).std(dim=0).mean().item())
    commitment_std = float(torch.stack(repeat_commitment, dim=0).std(dim=0).mean().item())
    success_cert_std = float(torch.stack(repeat_success_cert, dim=0).std(dim=0).mean().item())
    success_guard_std = float(torch.stack(repeat_success_guard, dim=0).std(dim=0).mean().item())

    return {
        "outcome_macro_f1": summarize_scalar_series([float(metrics["outcome_macro_f1"]) for metrics in repeat_metrics]),
        "failure_recall": summarize_scalar_series([float(metrics["failure_recall"]) for metrics in repeat_metrics]),
        "unsafe_success_rate_correctness": summarize_scalar_series(
            [float(metrics["unsafe_success_rate_correctness"]) for metrics in repeat_metrics]
        ),
        "max_outcome_share_swing": summarize_scalar_series(
            [float(metrics["max_outcome_share_swing"]) for metrics in repeat_metrics]
        ),
        "seed_disagreement_rate": _prediction_disagreement_rate(repeat_predictions),
        "mean_success_prob_std": success_prob_std,
        "mean_commitment_depth_std": commitment_std,
        "mean_success_cert_std": success_cert_std,
        "mean_success_guard_std": success_guard_std,
    }


def _summarize_mode(
    mode: str,
    repeat_results: list[dict[str, object]],
    monotonicity: dict[str, object],
) -> dict[str, object]:
    repeat_metrics = [result["metrics"] for result in repeat_results]
    repeat_primitives = [result["primitive_means"] for result in repeat_results]
    repeat_summaries = [
        {
            "seed": int(result.get("seed", -1)),
            "metrics": result["metrics"],
            "primitive_means": result["primitive_means"],
        }
        for result in repeat_results
    ]
    summary = {
        "mode": mode,
        "description": MODE_SPECS[mode]["description"],
        "repeats": repeat_summaries,
        "aggregate": {
            "outcome_acc": summarize_scalar_series([float(metrics["outcome_acc"]) for metrics in repeat_metrics]),
            "outcome_macro_f1": summarize_scalar_series([float(metrics["outcome_macro_f1"]) for metrics in repeat_metrics]),
            "failure_recall": summarize_scalar_series([float(metrics["failure_recall"]) for metrics in repeat_metrics]),
            "unsafe_success_rate_correctness": summarize_scalar_series(
                [float(metrics["unsafe_success_rate_correctness"]) for metrics in repeat_metrics]
            ),
            "success_precision_vs_is_correct": summarize_scalar_series(
                [float(metrics["success_precision_vs_is_correct"]) for metrics in repeat_metrics]
            ),
            "max_outcome_share_swing": summarize_scalar_series(
                [float(metrics["max_outcome_share_swing"]) for metrics in repeat_metrics]
            ),
            "stability_cert": summarize_scalar_series(
                [float(primitives["stability_cert"]) for primitives in repeat_primitives]
            ),
            "support_cert": summarize_scalar_series(
                [float(primitives["support_cert"]) for primitives in repeat_primitives]
            ),
            "success_guard": summarize_scalar_series(
                [float(primitives["success_guard"]) for primitives in repeat_primitives]
            ),
            "success_cert": summarize_scalar_series(
                [float(primitives["success_cert"]) for primitives in repeat_primitives]
            ),
            "ambiguity_cert": summarize_scalar_series(
                [float(primitives["ambiguity_cert"]) for primitives in repeat_primitives]
            ),
            "commitment_depth": summarize_scalar_series(
                [float(primitives["commitment_depth"]) for primitives in repeat_primitives]
            ),
        },
        "seed_variance": _seed_variance_summary(repeat_results),
        "monotonicity": monotonicity,
        "threshold_sensitivity": {
            "max_outcome_share_swing": summarize_scalar_series(
                [float(result["metrics"]["max_outcome_share_swing"]) for result in repeat_results]
            ),
            "label_revision_trigger": bool(
                any(bool(result["metrics"]["threshold_sensitivity"]["label_revision_trigger"]) for result in repeat_results)
            ),
        },
    }
    return summary


def _conclusion(mode_summaries: dict[str, dict[str, object]]) -> str:
    multi = mode_summaries["multi_path"]
    stochastic = mode_summaries["single_path_stochastic"]
    deterministic = mode_summaries["single_path_deterministic"]

    multi_failure = float(multi["aggregate"]["failure_recall"]["mean"])
    stochastic_failure = float(stochastic["aggregate"]["failure_recall"]["mean"])
    deterministic_failure = float(deterministic["aggregate"]["failure_recall"]["mean"])

    multi_commit_v = float(multi["monotonicity"]["commitment_monotonicity_violation_rate"])
    stochastic_commit_v = float(stochastic["monotonicity"]["commitment_monotonicity_violation_rate"])
    deterministic_commit_v = float(deterministic["monotonicity"]["commitment_monotonicity_violation_rate"])

    multi_seed_std = float(multi["seed_variance"]["mean_success_prob_std"])
    stochastic_seed_std = float(stochastic["seed_variance"]["mean_success_prob_std"])
    deterministic_seed_std = float(deterministic["seed_variance"]["mean_success_prob_std"])

    if (
        multi_failure >= max(stochastic_failure, deterministic_failure) - 0.02
        and multi_commit_v <= min(stochastic_commit_v, deterministic_commit_v) + 0.02
        and multi_seed_std <= min(stochastic_seed_std, deterministic_seed_std) * 0.9
    ):
        return "path multiplicity looks causal"
    return "path multiplicity mostly descriptive"


def _format_markdown(mode_summaries: dict[str, dict[str, object]], conclusion: str) -> str:
    headers = [
        "mode",
        "outcome_acc",
        "outcome_f1",
        "failure_recall",
        "unsafe_success",
        "mono_v",
        "commit_v",
        "swing",
        "seed_std_p(success)",
        "seed_std_commit",
    ]
    lines = [
        "# E10 Path Count Ablation",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for mode, summary in mode_summaries.items():
        aggregate = summary["aggregate"]
        seed_variance = summary["seed_variance"]
        monotonicity = summary["monotonicity"]
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    f"{float(aggregate['outcome_acc']['mean']):.3f}",
                    f"{float(aggregate['outcome_macro_f1']['mean']):.3f}",
                    f"{float(aggregate['failure_recall']['mean']):.3f}",
                    f"{float(aggregate['unsafe_success_rate_correctness']['mean']):.3f}",
                    f"{float(monotonicity['success_prob_monotonicity_violation_rate']):.3f}",
                    f"{float(monotonicity['commitment_monotonicity_violation_rate']):.3f}",
                    f"{float(aggregate['max_outcome_share_swing']['mean']):.3f}",
                    f"{float(seed_variance['mean_success_prob_std']):.4f}",
                    f"{float(seed_variance['mean_commitment_depth_std']):.4f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            conclusion,
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--head-mode", type=str, default="mlp")
    args = parser.parse_args()

    config = Config()
    trace_dir = Path(args.trace_dir).expanduser().resolve() if args.trace_dir else (VARIANT_DIR / config.trace_dir).resolve()
    device = torch.device(
        args.device if args.device is not None else (config.device if torch.cuda.is_available() else "cpu")
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path(__file__).resolve().parents[2] / "results" / "e10" / "path_count_ablation").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    metadata = load_trace_metadata(trace_dir)
    vocab = SimpleVocabulary()
    records = slice_records(load_trace_split(trace_dir, args.split), limit=args.limit, offset=args.offset)
    dataset, loader = build_loader(
        records,
        vocab,
        max_output_len=config.max_output_len,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    del dataset

    mode_summaries: dict[str, dict[str, object]] = {}
    for mode, spec in MODE_SPECS.items():
        set_global_seed(args.seed)
        strict = args.head_mode.strip().lower() == "mlp"
        model = build_model(
            config,
            vocab,
            checkpoint_path=args.checkpoint,
            head_mode=args.head_mode,
            config_overrides=spec["config_overrides"],
            trace_metadata=metadata,
            device=device,
            strict=strict,
        )
        repeat_results: list[dict[str, object]] = []
        cached_states: list[dict[str, torch.Tensor] | None] | None = None

        for repeat_idx in range(max(int(args.repeats), 1)):
            repeat_seed = args.seed + repeat_idx
            set_global_seed(repeat_seed)
            if spec["deterministic_cache"] and cached_states is not None:
                repeat_result = _run_repeat(
                    model,
                    loader,
                    device,
                    records,
                    diffusion_cache=cached_states,
                    capture_cache=False,
                )
            else:
                repeat_result = _run_repeat(
                    model,
                    loader,
                    device,
                    records,
                    diffusion_cache=None,
                    capture_cache=(repeat_idx == 0),
                )
                if repeat_idx == 0 and spec["deterministic_cache"]:
                    cached_states = repeat_result["bundle"].get("diffusion_cache")  # type: ignore[assignment]
                elif repeat_idx == 0:
                    cached_states = repeat_result["bundle"].get("diffusion_cache")  # type: ignore[assignment]
            repeat_result["seed"] = repeat_seed
            repeat_results.append(repeat_result)

        monotonicity = _audit_monotonicity(
            model,
            loader,
            device,
            diffusion_cache=cached_states,
        )
        mode_summary = _summarize_mode(mode, repeat_results, monotonicity)
        mode_summary["config_overrides"] = spec["config_overrides"]
        mode_summary["deterministic_cache"] = bool(spec["deterministic_cache"])
        mode_summary["repeats_used"] = int(args.repeats)
        mode_summaries[mode] = mode_summary

    conclusion = _conclusion(mode_summaries)
    markdown = _format_markdown(mode_summaries, conclusion)
    (output_dir / "summary.md").write_text(markdown)

    payload = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "trace_dir": str(trace_dir),
        "split": args.split,
        "limit": args.limit,
        "offset": args.offset,
        "seed": args.seed,
        "repeats": int(args.repeats),
        "head_mode": args.head_mode,
        "conclusion": conclusion,
        "modes": mode_summaries,
    }
    write_json(output_dir / "summary.json", payload)
    print(markdown.rstrip())
    print(conclusion)


if __name__ == "__main__":
    main()
