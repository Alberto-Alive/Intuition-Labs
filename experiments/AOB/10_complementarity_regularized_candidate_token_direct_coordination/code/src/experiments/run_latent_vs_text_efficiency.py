from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from statistics import mean, pstdev
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch

from src.coordinators.latent_coordination import LatentCoordinatorConfig, make_candidate_query_coordinator, make_latent_coordinator
from src.coordinators.mlp import MLPClassifier, MLPTrainingConfig
from src.datasets.multiview_code_patch_selection import (
    MultiViewTaskExample,
    attempt_batch_from_examples,
    build_multiview_code_patch_splits,
    format_candidate_block,
    format_clone_prompt,
)
from src.experiments.architecture_search import (
    StageConfig,
    _dataset_config,
)
from src.experiments.real_shared_weight_latent_coordination import (
    BENCHMARK,
    MessageChannelConfig,
    SharedClonedAgentSystem,
    SharedTransformerAgent,
    SharedTransformerAgentConfig,
    TextOutputOnlyCoordinator,
    predict_latent_system,
    _text_tokens,
)


DEFAULT_STAGE3_CONFIG = "configs/stage3_gpu_hard_validation.json"
DEFAULT_STAGE3_RESULTS = "results/stage3_gpu_hard_validation_results.json"
DEFAULT_OUTPUT = "results/latent_vs_text_efficiency_results.json"
DEFAULT_REPORT = "reports/LATENT_VS_TEXT_EFFICIENCY.md"
ARCHITECTURE = "topk_attention_no_head"

METHODS = (
    "trainable_topk_attention_no_head",
    "frozen_topk_attention_no_head",
    "text_only_multi_agent_baseline",
    "raw_latent_baseline",
    "explicit_evidence_oracle",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 latent-vs-text efficiency for the locked top-k no-head architecture.")
    parser.add_argument("--stage3-config", default=DEFAULT_STAGE3_CONFIG)
    parser.add_argument("--stage3-results", default=DEFAULT_STAGE3_RESULTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()
    run_efficiency(
        stage3_config_path=Path(args.stage3_config),
        stage3_results_path=Path(args.stage3_results),
        output_path=Path(args.output),
        report_path=Path(args.report),
    )


def run_efficiency(
    stage3_config_path: Path,
    stage3_results_path: Path,
    output_path: Path,
    report_path: Path,
) -> Dict[str, object]:
    stage3_config = json.loads(stage3_config_path.read_text(encoding="utf-8"))
    stage3_results = json.loads(stage3_results_path.read_text(encoding="utf-8"))
    device, hardware = _configure_cuda(stage3_config)
    result = _load_result(output_path)
    result.setdefault("metadata", {})
    result["metadata"].update(
        {
            "benchmark": BENCHMARK,
            "task": "latent_vs_text_efficiency",
            "architecture": ARCHITECTURE,
            "architecture_changes": "forbidden",
            "stage3_config_path": str(stage3_config_path),
            "stage3_results_path": str(stage3_results_path),
            "device": device,
            "hardware": hardware,
            "updated_at_utc": _now(),
            "token_accounting": {
                "latent_generated_text_tokens": "zero unless actual text is produced",
                "latent_communication": "reported separately as dense latent floats, not counted as text tokens",
                "text_only_baseline": "agents communicate textual summary/answer messages only; coordinator consumes those messages plus candidate text",
            },
        }
    )
    result["stage3_summary"] = stage3_results.get("summary", {})
    result["stage3_corrected_controls_pass"] = bool(stage3_results.get("summary", {}).get("passed_stage3_locked_success_criteria", False))
    result.setdefault("per_seed", [])

    completed = {
        int(row["seed"])
        for row in result["per_seed"]
        if row.get("status") == "completed" and row.get("measurement_source") == "exact_stage3_checkpoint"
    }
    stage = _stage_from_config(stage3_config)
    stage_rows = {int(row["seed"]): row for row in stage3_results.get("stage3_rows", []) if row.get("status") == "completed"}
    seeds = [int(value) for value in stage3_config["stage"]["seeds"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        if seed in completed:
            print(f"efficiency seed={seed}: existing completed row found; skipping")
            continue
        if seed not in stage_rows:
            raise RuntimeError(f"Stage 3 seed {seed} is missing from {stage3_results_path}")
        result["per_seed"] = [row for row in result["per_seed"] if int(row.get("seed", -1)) != seed]
        print(f"efficiency seed={seed}: building splits")
        splits = build_multiview_code_patch_splits(_dataset_config(stage), seed=seed, repo_root=Path("."))
        test_examples = list(splits["test"])
        labels = np.asarray([example.label for example in test_examples], dtype=np.int64)
        token_counts = _token_counts(test_examples, hidden_dim=stage.hidden_dim)

        checkpoints = stage_rows[seed].get("checkpoint_paths", {})
        _require_checkpoint_paths(seed, checkpoints)
        print(f"efficiency seed={seed}: loading exact Stage 3 checkpoints")
        trainable = _load_latent_checkpoint(Path(str(checkpoints["trainable"])), device=device)
        frozen = _load_latent_checkpoint(Path(str(checkpoints["frozen"])), device=device)
        raw = _load_latent_checkpoint(Path(str(checkpoints["raw_latent"])), device=device)
        text = _load_text_checkpoint(Path(str(checkpoints["text_only"])), device=device)

        method_rows = []
        method_rows.append(
            _measure_method(
                method="trainable_topk_attention_no_head",
                examples=test_examples,
                labels=labels,
                token_counts=token_counts["latent"],
                device=device,
                predict_fn=lambda examples, result=trainable: predict_latent_system(result, examples, "none", seed),
                stage3_accuracy=float(stage_rows[seed]["test_accuracy"]["trainable"]),
                checkpoint_path=str(checkpoints["trainable"]),
            )
        )
        method_rows.append(
            _measure_method(
                method="frozen_topk_attention_no_head",
                examples=test_examples,
                labels=labels,
                token_counts=token_counts["latent"],
                device=device,
                predict_fn=lambda examples, result=frozen: predict_latent_system(result, examples, "none", seed),
                stage3_accuracy=float(stage_rows[seed]["test_accuracy"]["frozen"]),
                checkpoint_path=str(checkpoints["frozen"]),
            )
        )
        method_rows.append(
            _measure_method(
                method="text_only_multi_agent_baseline",
                examples=test_examples,
                labels=labels,
                token_counts=token_counts["text_only"],
                device=device,
                predict_fn=lambda examples, text=text: text.predict(examples),
                stage3_accuracy=float(stage_rows[seed]["test_accuracy"]["text_only"]),
                checkpoint_path=str(checkpoints["text_only"]),
            )
        )
        method_rows.append(
            _measure_method(
                method="raw_latent_baseline",
                examples=test_examples,
                labels=labels,
                token_counts=token_counts["latent"],
                device=device,
                predict_fn=lambda examples, raw=raw: predict_latent_system(raw, examples, "none", seed),
                stage3_accuracy=float(stage_rows[seed]["test_accuracy"]["raw_latent"]),
                checkpoint_path=str(checkpoints["raw_latent"]),
            )
        )
        method_rows.append(
            _measure_method(
                method="explicit_evidence_oracle",
                examples=test_examples,
                labels=labels,
                token_counts=token_counts["oracle"],
                device=device,
                predict_fn=lambda examples: np.asarray([example.label for example in examples], dtype=np.int64),
                stage3_accuracy=float(stage_rows[seed]["test_accuracy"]["explicit_evidence_oracle"]),
                checkpoint_path="not_applicable",
            )
        )
        seed_row = {
            "seed": seed,
            "status": "completed",
            "n_examples": len(test_examples),
            "measurement_source": "exact_stage3_checkpoint",
            "methods": method_rows,
            "stage3_control_gate_pass": bool(stage3_results.get("summary", {}).get("passed_stage3_locked_success_criteria", False)),
            "completed_at_utc": _now(),
        }
        result["per_seed"].append(seed_row)
        result["aggregate"] = _aggregate(result["per_seed"], result["stage3_summary"])
        _write_outputs(result, output_path, report_path)
        print(f"efficiency seed={seed}: completed")

        del trainable, frozen, raw, text
        _clear_cuda()

    result["aggregate"] = _aggregate(result["per_seed"], result["stage3_summary"])
    _write_outputs(result, output_path, report_path)
    return result


def _require_checkpoint_paths(seed: int, checkpoints: object) -> None:
    if not isinstance(checkpoints, dict):
        raise RuntimeError(f"Stage 3 seed {seed} has no checkpoint_paths mapping")
    required = ("trainable", "frozen", "text_only", "raw_latent")
    missing = [name for name in required if not checkpoints.get(name) or not Path(str(checkpoints[name])).exists()]
    if missing:
        raise RuntimeError(f"Stage 3 seed {seed} is missing checkpoint files for: {missing}")


def _load_latent_checkpoint(path: Path, device: str):
    payload = _torch_load(path)
    if payload.get("checkpoint_type") != "latent_system":
        raise RuntimeError(f"{path} is not a latent system checkpoint")
    agent_config = _make_dataclass(SharedTransformerAgentConfig, payload.get("agent_config", {}))
    message_config = _make_dataclass(MessageChannelConfig, payload.get("message_config", {}))
    coordinator_config = _make_dataclass(LatentCoordinatorConfig, payload.get("coordinator_config", {}))
    agent = SharedTransformerAgent(agent_config).to(device)
    agent.configure_trainable(bool(payload.get("trainable_agent", False)))
    coordinator_input_dim = int(message_config.message_dim) if message_config.use_message_head else int(agent.output_dim)
    effective_coordinator_config = replace(coordinator_config, input_dim=coordinator_input_dim)
    candidate_query_families = {
        "candidate_query_cross_attention",
        "candidate_token_cross_attention",
        "bilinear_candidate",
        "contrastive_candidate",
        "global_then_candidate",
        "two_round_message_passing",
    }
    if message_config.coordinator_family in candidate_query_families:
        family = "cross_attention" if message_config.coordinator_family == "candidate_query_cross_attention" else message_config.coordinator_family
        coordinator = make_candidate_query_coordinator(
            n_roles=int(payload.get("n_roles", 4)),
            candidate_feature_dim=4,
            config=replace(effective_coordinator_config, family=family),
        ).to(device)
    elif message_config.coordinator_family == "latent":
        coordinator = make_latent_coordinator(
            n_roles=int(payload.get("n_roles", 4)),
            num_classes=int(payload.get("num_classes", 8)),
            config=effective_coordinator_config,
        ).to(device)
    else:
        raise RuntimeError(f"unknown message coordinator family in {path}: {message_config.coordinator_family}")
    system = SharedClonedAgentSystem(
        agent,
        coordinator,
        n_roles=int(payload.get("n_roles", 4)),
        visible_explicit_evidence=bool(payload.get("visible_explicit_evidence", False)),
        message_config=message_config,
    ).to(device)
    system.load_state_dict(payload["system_state_dict"])
    system.to(device)
    system.eval()
    return SimpleNamespace(system=system, method=payload.get("method"), checkpoint_path=str(path))


def _load_text_checkpoint(path: Path, device: str) -> TextOutputOnlyCoordinator:
    payload = _torch_load(path)
    if payload.get("checkpoint_type") != "text_only_multi_agent_baseline":
        raise RuntimeError(f"{path} is not a text-only checkpoint")
    training = _make_dataclass(MLPTrainingConfig, payload.get("training", {}))
    text = TextOutputOnlyCoordinator(
        num_classes=int(payload.get("num_classes", 8)),
        training=training,
        seed=int(payload.get("seed", 0)),
        device=device,
        feature_dim=int(payload.get("feature_dim", 128)),
    )
    text.model = MLPClassifier(text.feature_dim, training.hidden_dims, text.num_classes).to(device)
    text.model.load_state_dict(payload["model_state_dict"])
    text.model.to(device)
    text.model.eval()
    text.param_count = int(payload.get("param_count", sum(parameter.numel() for parameter in text.model.parameters())))
    text.history = list(payload.get("history", []))
    return text


def _torch_load(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _make_dataclass(cls, data: object):
    values = dict(data or {})
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def _measure_method(
    method: str,
    examples: Sequence[MultiViewTaskExample],
    labels: np.ndarray,
    token_counts: Dict[str, float],
    device: str,
    predict_fn: Callable[[Sequence[MultiViewTaskExample]], np.ndarray],
    stage3_accuracy: float,
    checkpoint_path: str,
) -> Dict[str, object]:
    _clear_cuda()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    predictions = predict_fn(examples)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gpu_peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") and torch.cuda.is_available() else 0
    checkpoint_accuracy = float(np.mean(predictions.astype(np.int64) == labels.astype(np.int64))) if len(labels) else 0.0
    per_example = elapsed / max(1, len(examples))
    total_tokens = float(token_counts["total_text_tokens_per_example"]) * len(examples)
    return {
        "method": method,
        "accuracy": checkpoint_accuracy,
        "checkpoint_accuracy": checkpoint_accuracy,
        "stage3_recorded_accuracy": stage3_accuracy,
        "accuracy_abs_delta_from_stage3": abs(checkpoint_accuracy - stage3_accuracy),
        "checkpoint_path": checkpoint_path,
        "measurement_source": "exact_stage3_checkpoint" if checkpoint_path != "not_applicable" else "deterministic_oracle",
        "refit_drift": "not_applicable",
        "input_tokens_per_example": float(token_counts["input_tokens_per_example"]),
        "output_communication_tokens_per_example": float(token_counts["output_communication_tokens_per_example"]),
        "generated_text_tokens_per_example": float(token_counts["generated_text_tokens_per_example"]),
        "latent_floats_per_example": float(token_counts["latent_floats_per_example"]),
        "latent_vectors_per_example": float(token_counts["latent_vectors_per_example"]),
        "total_text_tokens_per_example": float(token_counts["total_text_tokens_per_example"]),
        "total_text_tokens": total_tokens,
        "total_latent_floats": float(token_counts["latent_floats_per_example"]) * len(examples),
        "wall_clock_seconds": elapsed,
        "wall_clock_latency_per_example": per_example,
        "gpu_peak_memory_bytes": gpu_peak,
        "gpu_peak_memory_gb": gpu_peak / float(1024**3),
        "accuracy_per_1k_text_tokens": checkpoint_accuracy / max(float(token_counts["total_text_tokens_per_example"]) / 1000.0, 1e-12),
        "accuracy_per_second": checkpoint_accuracy / max(per_example, 1e-12),
    }


def _token_counts(examples: Sequence[MultiViewTaskExample], hidden_dim: int) -> Dict[str, Dict[str, float]]:
    latent_inputs = []
    text_agent_inputs = []
    text_coordinator_inputs = []
    text_outputs = []
    oracle_inputs = []
    for example in examples:
        n_roles = len(example.views)
        clone_prompt_tokens = sum(_count_tokens(format_clone_prompt(example, role_index)) for role_index in range(n_roles))
        latent_inputs.append(clone_prompt_tokens)
        text_agent_inputs.append(clone_prompt_tokens)
        batch = attempt_batch_from_examples([example], split="efficiency")
        summary_answer_messages = []
        for role_index, visible in enumerate(batch.visible_texts[0]):
            answer = int(batch.answers[0, role_index])
            confidence = float(batch.confidences[0, role_index])
            summary_answer_messages.append(f"{visible} answer={answer} confidence={confidence:.3f}")
        summary_tokens = sum(_count_tokens(text) for text in summary_answer_messages)
        candidate_tokens = _count_tokens(format_candidate_block(example))
        text_outputs.append(summary_tokens)
        text_coordinator_inputs.append(summary_tokens + candidate_tokens)
        oracle_inputs.append(_count_tokens(_explicit_oracle_text(example)))

    n_roles = len(examples[0].views) if examples else 4
    latent_floats = float(n_roles * hidden_dim)
    latent_vectors = float(n_roles)
    latent_input = _mean(latent_inputs)
    text_input = _mean(text_agent_inputs) + _mean(text_coordinator_inputs)
    text_output = _mean(text_outputs)
    oracle_input = _mean(oracle_inputs)
    return {
        "latent": {
            "input_tokens_per_example": latent_input,
            "output_communication_tokens_per_example": 0.0,
            "generated_text_tokens_per_example": 0.0,
            "latent_floats_per_example": latent_floats,
            "latent_vectors_per_example": latent_vectors,
            "total_text_tokens_per_example": latent_input,
        },
        "text_only": {
            "input_tokens_per_example": text_input,
            "output_communication_tokens_per_example": text_output,
            "generated_text_tokens_per_example": text_output,
            "latent_floats_per_example": 0.0,
            "latent_vectors_per_example": 0.0,
            "total_text_tokens_per_example": text_input + text_output,
        },
        "oracle": {
            "input_tokens_per_example": oracle_input,
            "output_communication_tokens_per_example": 0.0,
            "generated_text_tokens_per_example": 0.0,
            "latent_floats_per_example": 0.0,
            "latent_vectors_per_example": 0.0,
            "total_text_tokens_per_example": oracle_input,
        },
    }


def _aggregate(per_seed: Sequence[Dict[str, object]], stage3_summary: Dict[str, object]) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, object]]] = {method: [] for method in METHODS}
    for seed_row in per_seed:
        if seed_row.get("status") != "completed":
            continue
        for row in seed_row.get("methods", []):
            grouped[str(row["method"])].append(row)
    methods = {}
    for method, rows in grouped.items():
        if not rows:
            continue
        total_examples = sum(int(seed_row.get("n_examples", 0)) for seed_row in per_seed if seed_row.get("status") == "completed")
        accuracy_values = [float(row["accuracy"]) for row in rows]
        latency_values = [float(row["wall_clock_latency_per_example"]) for row in rows]
        total_text_tokens = sum(float(row["total_text_tokens"]) for row in rows)
        total_latent_floats = sum(float(row["total_latent_floats"]) for row in rows)
        total_seconds = sum(float(row["wall_clock_seconds"]) for row in rows)
        mean_total_tokens_per_example = _mean([float(row["total_text_tokens_per_example"]) for row in rows])
        mean_accuracy = _mean(accuracy_values)
        methods[method] = {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": pstdev(accuracy_values) if len(accuracy_values) > 1 else 0.0,
            "input_tokens_per_example": _mean([float(row["input_tokens_per_example"]) for row in rows]),
            "output_communication_tokens_per_example": _mean([float(row["output_communication_tokens_per_example"]) for row in rows]),
            "generated_text_tokens_per_example": _mean([float(row["generated_text_tokens_per_example"]) for row in rows]),
            "latent_floats_per_example": _mean([float(row["latent_floats_per_example"]) for row in rows]),
            "latent_vectors_per_example": _mean([float(row["latent_vectors_per_example"]) for row in rows]),
            "total_text_tokens_per_example": mean_total_tokens_per_example,
            "total_text_tokens": total_text_tokens,
            "total_latent_floats": total_latent_floats,
            "wall_clock_latency_per_example": _mean(latency_values),
            "wall_clock_seconds_total": total_seconds,
            "gpu_peak_memory_gb_mean": _mean([float(row["gpu_peak_memory_gb"]) for row in rows]),
            "gpu_peak_memory_gb_max": max(float(row["gpu_peak_memory_gb"]) for row in rows),
            "accuracy_per_1k_text_tokens": mean_accuracy / max(mean_total_tokens_per_example / 1000.0, 1e-12),
            "accuracy_per_second": mean_accuracy / max(_mean(latency_values), 1e-12),
            "accuracy_abs_delta_from_stage3_max": max(float(row["accuracy_abs_delta_from_stage3"]) for row in rows),
            "n_seeds": len(rows),
            "n_examples_total": total_examples,
        }
    trainable = methods.get("trainable_topk_attention_no_head", {})
    text = methods.get("text_only_multi_agent_baseline", {})
    controls_pass = bool(stage3_summary.get("passed_stage3_locked_success_criteria", False))
    claim_allowed = bool(
        trainable
        and text
        and float(trainable.get("mean_accuracy", 0.0)) >= 0.80
        and float(text.get("mean_accuracy", 1.0)) < float(trainable.get("mean_accuracy", 0.0)) - 0.10
        and float(trainable.get("output_communication_tokens_per_example", 1e9)) < float(text.get("output_communication_tokens_per_example", 0.0))
        and controls_pass
    )
    return {
        "methods": methods,
        "max_refit_drift": "not_applicable",
        "main_claim_allowed": claim_allowed,
        "claim_gate": {
            "trainable_latent_accuracy_high": float(trainable.get("mean_accuracy", 0.0)) >= 0.80 if trainable else False,
            "text_only_near_chance_or_lower": float(text.get("mean_accuracy", 1.0)) < float(trainable.get("mean_accuracy", 0.0)) - 0.10 if trainable and text else False,
            "latent_uses_fewer_output_communication_tokens": float(trainable.get("output_communication_tokens_per_example", 1e9)) < float(text.get("output_communication_tokens_per_example", 0.0)) if trainable and text else False,
            "stage3_controls_still_pass": controls_pass,
        },
    }


def _write_outputs(result: Dict[str, object], output_path: Path, report_path: Path) -> None:
    _normalize_checkpoint_accuracy_fields(result)
    result["aggregate"] = _aggregate(result.get("per_seed", []), result.get("stage3_summary", {}))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")


def _normalize_checkpoint_accuracy_fields(result: Dict[str, object]) -> None:
    for seed_row in result.get("per_seed", []):
        for row in seed_row.get("methods", []):
            if "stage3_recorded_accuracy" not in row:
                continue
            if row.get("measurement_source") not in {"exact_stage3_checkpoint", "deterministic_oracle"}:
                continue
            row["accuracy"] = float(row.get("checkpoint_accuracy", row.get("accuracy", 0.0)))
            row["accuracy_abs_delta_from_stage3"] = abs(float(row["accuracy"]) - float(row["stage3_recorded_accuracy"]))
            row["refit_drift"] = "not_applicable"
            row["accuracy_per_1k_text_tokens"] = float(row["accuracy"]) / max(float(row["total_text_tokens_per_example"]) / 1000.0, 1e-12)
            row["accuracy_per_second"] = float(row["accuracy"]) / max(float(row["wall_clock_latency_per_example"]), 1e-12)


def _render_report(result: Dict[str, object]) -> str:
    aggregate = result.get("aggregate", {})
    methods = aggregate.get("methods", {})
    lines = [
        "# Latent vs Text Efficiency",
        "",
        "## Scope",
        "",
        f"- Benchmark: `{BENCHMARK}` Stage 3 semantic no-literal-cue test split.",
        f"- Architecture: `{ARCHITECTURE}`; no architecture changes were made.",
        "- Latent methods count generated text tokens as zero. Dense latent communication is reported separately as floats/vectors, not text tokens.",
        "- Text-only baseline uses textual summary/answer messages from agents plus candidate text at the coordinator.",
        "- Accuracy, latency, memory, and token counts are measured from the exact saved Stage 3 checkpoints for trainable, frozen, text-only, and raw-latent methods.",
        "- No refit-based latency or memory measurements are used in the main proof; refit drift is not applicable.",
        "- Latent coordination is not claimed to be faster than text-only coordination unless the measured latency table shows it.",
        f"- CUDA device: `{result.get('metadata', {}).get('hardware', {}).get('cuda_device_name', 'n/a')}`",
        f"- Stage 3 controls still pass: `{bool(result.get('stage3_corrected_controls_pass', False))}`",
        "",
        "## Efficiency Table",
        "",
        "| method | checkpoint accuracy | input tok/ex | output comm tok/ex | latent floats/ex | total text tokens | total latent floats | latency ms/ex | GPU GB peak | acc/1k tok | acc/sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = methods.get(method)
        if not row:
            continue
        lines.append(
            "| {method} | {acc:.4f} | {inp:.1f} | {out:.1f} | {latent:.1f} | {total:.0f} | {total_latent:.0f} | {lat:.3f} | {gpu:.4f} | {apt:.3f} | {aps:.2f} |".format(
                method=method,
                acc=float(row["mean_accuracy"]),
                inp=float(row["input_tokens_per_example"]),
                out=float(row["output_communication_tokens_per_example"]),
                latent=float(row["latent_floats_per_example"]),
                total=float(row["total_text_tokens"]),
                total_latent=float(row["total_latent_floats"]),
                lat=float(row["wall_clock_latency_per_example"]) * 1000.0,
                gpu=float(row["gpu_peak_memory_gb_max"]),
                apt=float(row["accuracy_per_1k_text_tokens"]),
                aps=float(row["accuracy_per_second"]),
            )
        )
    lines.extend(["", "## Stage 3 Control Gate", ""])
    lines.append("| criterion | pass | value |")
    lines.append("|---|---|---|")
    for item in result.get("stage3_summary", {}).get("success_criteria", []):
        criterion = str(item.get("criterion"))
        if criterion in {
            "corruption controls and non-latent baselines remain near chance",
            "invariance controls preserve accuracy",
            "role-label shuffle does not explain the result",
            "leakage audits pass",
        }:
            lines.append(f"| {criterion} | {bool(item.get('pass'))} | `{item.get('value')}` |")
    lines.extend(
        [
            "",
            "## Claim Gate",
            "",
            "| condition | pass |",
            "|---|---|",
        ]
    )
    for key, value in aggregate.get("claim_gate", {}).items():
        lines.append(f"| {key} | {bool(value)} |")
    lines.extend(["", "## Per-Seed Accuracy Check", ""])
    lines.append("| seed | trainable | frozen | text-only | raw-latent | oracle |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for seed_row in result.get("per_seed", []):
        values = {row["method"]: row for row in seed_row.get("methods", [])}
        lines.append(
            "| {seed} | {train:.4f} | {frozen:.4f} | {text:.4f} | {raw:.4f} | {oracle:.4f} |".format(
                seed=seed_row["seed"],
                train=float(values.get("trainable_topk_attention_no_head", {}).get("accuracy", 0.0)),
                frozen=float(values.get("frozen_topk_attention_no_head", {}).get("accuracy", 0.0)),
                text=float(values.get("text_only_multi_agent_baseline", {}).get("accuracy", 0.0)),
                raw=float(values.get("raw_latent_baseline", {}).get("accuracy", 0.0)),
                oracle=float(values.get("explicit_evidence_oracle", {}).get("accuracy", 0.0)),
            )
        )
    lines.extend(["", "## Interpretation", ""])
    if bool(aggregate.get("main_claim_allowed", False)):
        trainable = methods.get("trainable_topk_attention_no_head", {})
        text = methods.get("text_only_multi_agent_baseline", {})
        if trainable and text and float(trainable.get("wall_clock_latency_per_example", 0.0)) > float(text.get("wall_clock_latency_per_example", 0.0)):
            lines.append("The measured latent path is not faster than the text-only baseline in this run, so no speed advantage is claimed.")
        lines.append(
            "latent coordination is much more accurate than text-only coordination while using zero generated communication tokens on this controlled benchmark. This does not establish real-world coding success."
        )
    else:
        lines.append(
            "The efficiency claim gate did not pass. Do not claim a latent communication advantage from this run."
        )
    return "\n".join(lines) + "\n"


def _stage_from_config(config: Dict[str, object]) -> StageConfig:
    stage = dict(config["stage"])
    return StageConfig(
        name="stage3",
        n_train=int(stage["n_train"]),
        n_dev=int(stage["n_dev"]),
        n_test=int(stage["n_test"]),
        seeds=tuple(int(value) for value in stage["seeds"]),
        epochs=int(stage["epochs"]),
        patience=int(stage["patience"]),
        hidden_dim=int(stage["hidden_dim"]),
        tiny_layers=int(stage["tiny_layers"]),
        tiny_ff_dim=int(stage["tiny_ff_dim"]),
        batch_size=int(stage.get("batch_size", 32)),
        lr=float(stage.get("lr", 0.002)),
        max_candidates=1,
        gradient_accumulation_steps=int(stage.get("gradient_accumulation_steps", 1)),
        mixed_precision=str(stage.get("mixed_precision", "none")),
    )


def _configure_cuda(config: Dict[str, object]) -> tuple[str, Dict[str, object]]:
    requested = str(config.get("device", "cuda"))
    require_cuda = bool(config.get("require_cuda", True))
    hardware: Dict[str, object] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": requested,
    }
    if requested.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        budget_gb = float(config.get("cuda_memory_budget_gb", 16.0))
        fraction = min(1.0, (budget_gb * (1024**3)) / float(props.total_memory))
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        hardware.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_total_memory": int(props.total_memory),
                "cuda_memory_budget_gb": budget_gb,
                "cuda_memory_fraction": fraction,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
        return "cuda", hardware
    if require_cuda:
        raise RuntimeError("CUDA was requested but is not available")
    return "cpu", hardware


def _load_result(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _explicit_oracle_text(example: MultiViewTaskExample) -> str:
    evidence = ",".join(str(int(value)) for value in example.metadata.get("evidence_bits", [0, 0, 0, 0]))
    candidates = "; ".join(f"{index}:{','.join(candidate.attributes)}" for index, candidate in enumerate(example.candidates))
    return f"explicit evidence tuple {evidence}; candidate attribute tuples {candidates}"


def _count_tokens(text: str) -> int:
    return len(_text_tokens(text))


def _mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    main()
