from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from src.agents.simulated import SimulatedAgentConfig, SimulatedAgentSwarm
from src.agents.strict_coordination import StrictCoordinationAgentConfig, StrictCoordinationSwarm
from src.agents.types import AttemptBatch
from src.coordinators.activation import (
    ActivationClusterRouter,
    ActivationPCAMLPCoordinator,
    ActivationPoolingMLPCoordinator,
    CapacityMatchedTextOnlyCoordinator,
    TextOnlyMLPCoordinator,
)
from src.coordinators.baselines import IndependentSwarmCoordinator, SingleAgentCoordinator
from src.coordinators.mlp import (
    MLPTrainingConfig,
    count_mlp_params,
    match_hidden_dims_to_budget,
)
from src.coordinators.probes import HiddenStateOnlyLabelProbe, OutputOnlyOracleProbe, TelemetryOnlyLabelProbe
from src.datasets.strict_coordination import (
    StrictCoordinationConfig,
    build_strict_coordination_splits,
)
from src.datasets.synthetic import SyntheticDatasetConfig, build_synthetic_splits
from src.evaluation.audit import audit_event, split_overlap_audit, summarize_test_access
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.report import write_report
from src.experiments.diagnostics import fit_agent_correctness_probes, fit_redundancy_probe
from src.telemetry.ablations import ablate_telemetry_channel
from src.telemetry.controls import apply_activation_control, randomized_train_labels
from src.telemetry.features import activation_features, output_oracle_features, visible_features


FittedRun = Tuple[str, object, Dict[str, AttemptBatch]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic activation-aware coordination experiment.")
    parser.add_argument("--config", default="configs/synthetic_cuda.json", help="Path to JSON config.")
    args = parser.parse_args()

    config = _load_config(args.config)
    deterministic = bool(config.get("deterministic", False))
    if deterministic:
        torch.use_deterministic_algorithms(True)
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
    device, hardware = _resolve_device(
        requested=str(config.get("device", "cpu")),
        cuda_memory_budget_gb=float(config.get("cuda_memory_budget_gb", 16.0)),
        deterministic=deterministic,
    )
    dataset_config = _make_dataclass(SyntheticDatasetConfig, config["dataset"])
    agent_config = _make_dataclass(SimulatedAgentConfig, config["agents"])
    label_reduced_agent_config = _make_dataclass(
        SimulatedAgentConfig,
        {**config["agents"], **config.get("label_reduced_agents", {})},
    )
    strict_dataset_config = _make_dataclass(
        StrictCoordinationConfig,
        config.get("strict_dataset", config["dataset"]),
    )
    strict_agent_config = _make_dataclass(
        StrictCoordinationAgentConfig,
        {**config["agents"], **config.get("strict_agents", {})},
    )
    training_config = MLPTrainingConfig(
        epochs=int(config["training"]["epochs"]),
        batch_size=int(config["training"]["batch_size"]),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        patience=int(config["training"]["patience"]),
        hidden_dims=tuple(int(x) for x in config["training"]["hidden_dims"]),
    )

    all_metrics: List[Dict[str, object]] = []
    all_diagnostics: List[Dict[str, object]] = []
    all_audits: List[Dict[str, object]] = []
    for seed in [int(x) for x in config["seeds"]]:
        print(f"seed={seed}: generating Stage 0 data and simulated agent attempts")
        splits = build_synthetic_splits(dataset_config, seed=seed)
        swarm = SimulatedAgentSwarm(
            config=agent_config,
            num_tasks=dataset_config.num_tasks,
            num_classes=dataset_config.num_classes,
            seed=seed,
        )
        _run_benchmark(
            benchmark="stage0_original",
            seed=seed,
            base_batches=swarm.run(splits),
            dataset_config=dataset_config,
            training_config=training_config,
            config=config,
            device=device,
            all_metrics=all_metrics,
            all_diagnostics=all_diagnostics,
            all_audits=all_audits,
        )

        if bool(config.get("run_label_signal_ablation", True)):
            print(f"seed={seed}: generating label-signal-reduced Stage 0 attempts")
            reduced_swarm = SimulatedAgentSwarm(
                config=label_reduced_agent_config,
                num_tasks=dataset_config.num_tasks,
                num_classes=dataset_config.num_classes,
                seed=seed + 50_000,
            )
            _run_benchmark(
                benchmark="stage0_label_reduced",
                seed=seed,
                base_batches=reduced_swarm.run(splits),
                dataset_config=dataset_config,
                training_config=training_config,
                config=config,
                device=device,
                all_metrics=all_metrics,
                all_diagnostics=all_diagnostics,
                all_audits=all_audits,
            )

        if bool(config.get("run_strict_coordination", True)):
            print(f"seed={seed}: generating strict coordination benchmark")
            strict_splits = build_strict_coordination_splits(strict_dataset_config, seed=seed)
            strict_swarm = StrictCoordinationSwarm(
                config=strict_agent_config,
                n_evidence_bits=strict_dataset_config.n_evidence_bits,
                num_classes=strict_dataset_config.num_classes,
                seed=seed,
            )
            _run_benchmark(
                benchmark="strict_coordination",
                seed=seed,
                base_batches=strict_swarm.run(strict_splits),
                dataset_config=strict_dataset_config,
                training_config=training_config,
                config=config,
                device=device,
                all_metrics=all_metrics,
                all_diagnostics=all_diagnostics,
                all_audits=all_audits,
            )

    all_audits.append(summarize_test_access(all_audits))

    results = {
        "metadata": {
            "config_path": str(Path(args.config)),
            "config": config,
            "dataset_config": asdict(dataset_config),
            "agent_config": asdict(agent_config),
            "label_reduced_agent_config": asdict(label_reduced_agent_config),
            "strict_dataset_config": asdict(strict_dataset_config),
            "strict_agent_config": asdict(strict_agent_config),
            "training_config": {
                **asdict(training_config),
                "hidden_dims": list(training_config.hidden_dims),
            },
            "hardware": hardware,
            "device_used": device,
        },
        "metrics": all_metrics,
        "diagnostics": all_diagnostics,
        "audit": all_audits,
        "validation": {
            "probe_access": _probe_access_matrix(),
        },
    }

    output_path = Path(str(config["output_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    write_report(results, config["report_path"])
    print(f"wrote {output_path}")
    print(f"wrote {config['report_path']}")


def _load_config(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _make_dataclass(cls, data: object):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in dict(data).items() if key in allowed})


def _without_labels(batch: AttemptBatch) -> AttemptBatch:
    return batch.copy_with_labels(np.full_like(batch.labels, -1))


def _run_benchmark(
    benchmark: str,
    seed: int,
    base_batches: Dict[str, AttemptBatch],
    dataset_config: object,
    training_config: MLPTrainingConfig,
    config: Dict[str, object],
    device: str,
    all_metrics: List[Dict[str, object]],
    all_diagnostics: List[Dict[str, object]],
    all_audits: List[Dict[str, object]],
    extra_pretest_fitters: List[Callable[[int], Tuple[int, List[Callable[[int], int]]]]] | None = None,
) -> None:
    fitted_runs: List[FittedRun] = []
    fitted_agent_probes = []
    fitted_redundancy_probe = None
    fitted_channel_ablations = []
    extra_test_callbacks: List[Callable[[int], int]] = []
    event_index = 0
    num_classes = int(getattr(dataset_config, "num_classes"))
    num_tasks = int(getattr(dataset_config, "num_tasks"))
    all_audits.append(split_overlap_audit(benchmark, seed, base_batches))

    for control in config["controls"]:
        controlled_batches = apply_activation_control(base_batches, control=control, seed=seed)
        methods = _build_methods(
            control=control,
            train_batch=controlled_batches["train"],
            num_classes=num_classes,
            num_tasks=num_tasks,
            training_config=training_config,
            pca_components=int(config["training"]["pca_components"]),
            clusters=int(config["training"]["clusters"]),
            seed=seed,
            device=device,
            include_diagnostic_probes=control == "none",
            hidden_state_probe_name=str(config.get("hidden_state_probe_name", "telemetry_only_label_probe")),
        )
        for method in methods:
            print(f"seed={seed} benchmark={benchmark} condition={control}: fitting {method.name}")
            all_audits.append(
                audit_event(
                    benchmark,
                    seed,
                    event_index,
                    "fit_start",
                    method=method.name,
                    condition=control,
                    split_inputs=("train", "dev"),
                )
            )
            event_index += 1
            method.fit(controlled_batches["train"], controlled_batches["dev"])
            all_audits.append(
                audit_event(
                    benchmark,
                    seed,
                    event_index,
                    "dev_predict",
                    method=method.name,
                    condition=control,
                    split_inputs=("dev",),
                )
            )
            event_index += 1
            dev_predictions = method.predict(_without_labels(controlled_batches["dev"]))
            all_metrics.append(
                evaluate_predictions(
                    method=method.name,
                    condition=control,
                    predictions=dev_predictions,
                    batch=controlled_batches["dev"],
                    seed=seed,
                    param_count=int(getattr(method, "param_count", 0)),
                    benchmark=benchmark,
                    metadata=method.metadata() if hasattr(method, "metadata") else {},
                )
            )
            fitted_runs.append((control, method, controlled_batches))

    if bool(config.get("run_diagnostics", True)):
        print(f"seed={seed} benchmark={benchmark}: fitting agent correctness and redundancy probes")
        all_audits.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "agent_correctness_probe_fit",
                method="agent_correctness_probe",
                condition="none",
                split_inputs=("train", "dev"),
            )
        )
        event_index += 1
        dev_probe_metrics, fitted_agent_probes = fit_agent_correctness_probes(
            benchmark=benchmark,
            seed=seed,
            train_batch=base_batches["train"],
            dev_batch=base_batches["dev"],
            num_classes=num_classes,
            num_tasks=num_tasks,
            training=training_config,
            device=device,
        )
        all_diagnostics.extend(dev_probe_metrics)
        all_audits.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "redundancy_probe_fit",
                method="redundancy_probe",
                condition="none",
                split_inputs=("train", "dev"),
            )
        )
        event_index += 1
        dev_redundancy_metrics, fitted_redundancy_probe = fit_redundancy_probe(
            benchmark=benchmark,
            seed=seed,
            train_batch=base_batches["train"],
            dev_batch=base_batches["dev"],
            clusters=int(config["training"]["clusters"]),
        )
        all_diagnostics.extend(dev_redundancy_metrics)

    if bool(config.get("run_telemetry_channel_ablations", True)) and base_batches["train"].telemetry_channels:
        for channel_name in sorted(base_batches["train"].telemetry_channels):
            print(f"seed={seed} benchmark={benchmark}: fitting channel ablation {channel_name}")
            channel_batches = {
                "train": ablate_telemetry_channel(base_batches["train"], channel_name),
                "dev": ablate_telemetry_channel(base_batches["dev"], channel_name),
            }
            method = ActivationPCAMLPCoordinator(
                num_classes=num_classes,
                num_tasks=num_tasks,
                components=int(config["training"]["pca_components"]),
                training=training_config,
                seed=seed + 120_000 + len(channel_name),
                device=device,
            )
            all_audits.append(
                audit_event(
                    benchmark,
                    seed,
                    event_index,
                    "telemetry_channel_ablation_fit",
                    method="activation_pca_mlp",
                    condition=f"ablate_{channel_name}",
                    split_inputs=("train", "dev"),
                )
            )
            event_index += 1
            method.fit(channel_batches["train"], channel_batches["dev"])
            all_audits.append(
                audit_event(
                    benchmark,
                    seed,
                    event_index,
                    "telemetry_channel_ablation_dev_predict",
                    method="activation_pca_mlp",
                    condition=f"ablate_{channel_name}",
                    split_inputs=("dev",),
                )
            )
            event_index += 1
            dev_predictions = method.predict(_without_labels(channel_batches["dev"]))
            all_diagnostics.append(
                _channel_ablation_result(
                    benchmark=benchmark,
                    seed=seed,
                    split="dev",
                    channel_name=channel_name,
                    predictions=dev_predictions,
                    batch=channel_batches["dev"],
                    param_count=int(getattr(method, "param_count", 0)),
                )
            )
            fitted_channel_ablations.append((channel_name, method))

    if bool(config.get("run_randomized_label_leakage", True)):
        leakage_batches = apply_activation_control(base_batches, control="none", seed=seed)
        randomized_train = randomized_train_labels(
            leakage_batches["train"],
            seed=seed,
            num_classes=num_classes,
        )
        leakage_fit_batches = {
            "train": randomized_train,
            "dev": leakage_batches["dev"],
            "test": leakage_batches["test"],
        }
        leakage_methods = _build_methods(
            control="none",
            train_batch=leakage_batches["train"],
            num_classes=num_classes,
            num_tasks=num_tasks,
            training_config=training_config,
            pca_components=int(config["training"]["pca_components"]),
            clusters=int(config["training"]["clusters"]),
            seed=seed + 90_000,
            device=device,
            include_diagnostic_probes=True,
            hidden_state_probe_name=str(config.get("hidden_state_probe_name", "telemetry_only_label_probe")),
        )
        for method in leakage_methods:
            if method.name in {"single_agent", "independent_swarm"}:
                continue
            print(f"seed={seed} benchmark={benchmark} condition=randomized_train_labels: fitting {method.name}")
            all_audits.append(
                audit_event(
                    benchmark,
                    seed,
                    event_index,
                    "randomized_label_fit",
                    method=method.name,
                    condition="randomized_train_labels",
                    split_inputs=("train",),
                )
            )
            event_index += 1
            method.fit(leakage_fit_batches["train"], None)
            all_audits.append(
                audit_event(
                    benchmark,
                    seed,
                    event_index,
                    "randomized_label_dev_predict",
                    method=method.name,
                    condition="randomized_train_labels",
                    split_inputs=("dev",),
                )
            )
            event_index += 1
            dev_predictions = method.predict(_without_labels(leakage_fit_batches["dev"]))
            all_metrics.append(
                evaluate_predictions(
                    method=method.name,
                    condition="randomized_train_labels",
                    predictions=dev_predictions,
                    batch=leakage_fit_batches["dev"],
                    seed=seed,
                    param_count=int(getattr(method, "param_count", 0)),
                    benchmark=benchmark,
                    metadata=method.metadata() if hasattr(method, "metadata") else {},
                )
            )
            fitted_runs.append(("randomized_train_labels", method, leakage_fit_batches))

    if extra_pretest_fitters:
        for extra_fitter in extra_pretest_fitters:
            event_index, callbacks = extra_fitter(event_index)
            extra_test_callbacks.extend(callbacks)

    print(f"seed={seed} benchmark={benchmark}: all methods fit; evaluating held-out test split")
    all_audits.append(
        audit_event(
            benchmark,
            seed,
            event_index,
            "test_gate_opened",
            split_inputs=(),
        )
    )
    event_index += 1
    for condition, method, batches in fitted_runs:
        all_audits.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "test_predict",
                method=method.name,
                condition=condition,
                split_inputs=("test",),
            )
        )
        event_index += 1
        test_predictions = method.predict(_without_labels(batches["test"]))
        all_metrics.append(
            evaluate_predictions(
                method=method.name,
                condition=condition,
                predictions=test_predictions,
                batch=batches["test"],
                seed=seed,
                param_count=int(getattr(method, "param_count", 0)),
                benchmark=benchmark,
                metadata=method.metadata() if hasattr(method, "metadata") else {},
            )
        )
    for probe in fitted_agent_probes:
        all_audits.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "agent_correctness_probe_test",
                method="agent_correctness_probe",
                condition="none",
                split_inputs=("test",),
            )
        )
        event_index += 1
        all_diagnostics.append(probe.evaluate(base_batches["test"]))
    if fitted_redundancy_probe is not None:
        all_audits.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "redundancy_probe_test",
                method="redundancy_probe",
                condition="none",
                split_inputs=("test",),
            )
        )
        event_index += 1
        all_diagnostics.extend(fitted_redundancy_probe.evaluate(base_batches["test"]))
    for channel_name, method in fitted_channel_ablations:
        test_batch = ablate_telemetry_channel(base_batches["test"], channel_name)
        all_audits.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "telemetry_channel_ablation_test_predict",
                method="activation_pca_mlp",
                condition=f"ablate_{channel_name}",
                split_inputs=("test",),
            )
        )
        event_index += 1
        test_predictions = method.predict(_without_labels(test_batch))
        all_diagnostics.append(
            _channel_ablation_result(
                benchmark=benchmark,
                seed=seed,
                split="test",
                channel_name=channel_name,
                predictions=test_predictions,
                batch=test_batch,
                param_count=int(getattr(method, "param_count", 0)),
            )
        )
    for callback in extra_test_callbacks:
        event_index = callback(event_index)


def _resolve_device(
    requested: str,
    cuda_memory_budget_gb: float,
    deterministic: bool = False,
) -> tuple[str, Dict[str, object]]:
    requested = requested.lower()
    hardware: Dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "deterministic": deterministic,
    }
    if requested == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(0)
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        total_gb = total_bytes / (1024**3)
        fraction = min(1.0, cuda_memory_budget_gb / total_gb)
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
        if not deterministic:
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        hardware.update(
            {
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_total_gb": total_gb,
                "cuda_memory_budget_gb": cuda_memory_budget_gb,
                "cuda_memory_fraction": fraction,
            }
        )
        return "cuda", hardware
    hardware["fallback_reason"] = "cuda unavailable or not requested"
    return "cpu", hardware


def _build_methods(
    control: str,
    train_batch: AttemptBatch,
    num_classes: int,
    num_tasks: int,
    training_config: MLPTrainingConfig,
    pca_components: int,
    clusters: int,
    seed: int,
    device: str,
    include_diagnostic_probes: bool = False,
    hidden_state_probe_name: str = "telemetry_only_label_probe",
) -> List[object]:
    visible_dim = visible_features(train_batch, num_classes, num_tasks).shape[1]
    target_dim = (
        visible_dim
        + activation_features(train_batch).shape[1]
    )
    activation_methods: List[object] = [
        ActivationPoolingMLPCoordinator(
            num_classes=num_classes,
            num_tasks=num_tasks,
            training=training_config,
            seed=seed + 1_001,
            device=device,
        ),
        ActivationPCAMLPCoordinator(
            num_classes=num_classes,
            num_tasks=num_tasks,
            components=pca_components,
            training=training_config,
            seed=seed + 2_001,
            device=device,
        ),
        ActivationClusterRouter(
            clusters=clusters,
            seed=seed + 3_001,
        ),
    ]
    if control != "none":
        return activation_methods
    methods: List[object] = [
        SingleAgentCoordinator(agent_index=0),
        IndependentSwarmCoordinator(),
        TextOnlyMLPCoordinator(
            num_classes=num_classes,
            num_tasks=num_tasks,
            training=training_config,
            seed=seed + 4_001,
            device=device,
        ),
        CapacityMatchedTextOnlyCoordinator(
            num_classes=num_classes,
            num_tasks=num_tasks,
            target_dim=target_dim,
            training=training_config,
            seed=seed + 5_001,
            device=device,
        ),
        *activation_methods,
    ]
    if include_diagnostic_probes:
        activation_pca_params = count_mlp_params(
            input_dim=visible_dim + pca_components,
            hidden_dims=training_config.hidden_dims,
            num_classes=num_classes,
        )
        output_dim = output_oracle_features(train_batch, num_classes, num_tasks).shape[1]
        oracle_hidden_dims = match_hidden_dims_to_budget(
            input_dim=output_dim,
            num_classes=num_classes,
            target_params=activation_pca_params,
        )
        oracle_training = MLPTrainingConfig(
            epochs=training_config.epochs,
            batch_size=training_config.batch_size,
            lr=training_config.lr,
            weight_decay=training_config.weight_decay,
            patience=training_config.patience,
            hidden_dims=oracle_hidden_dims,
        )
        methods.extend(
            [
                _hidden_state_probe(
                    name=hidden_state_probe_name,
                    num_classes=num_classes,
                    components=pca_components,
                    training=training_config,
                    seed=seed + 6_001,
                    device=device,
                ),
                OutputOnlyOracleProbe(
                    num_classes=num_classes,
                    num_tasks=num_tasks,
                    target_param_count=activation_pca_params,
                    training=oracle_training,
                    seed=seed + 7_001,
                    device=device,
                ),
            ]
        )
    return methods


def _hidden_state_probe(
    name: str,
    num_classes: int,
    components: int,
    training: MLPTrainingConfig,
    seed: int,
    device: str,
):
    if name == "hidden_state_only_probe":
        return HiddenStateOnlyLabelProbe(
            num_classes=num_classes,
            components=components,
            training=training,
            seed=seed,
            device=device,
        )
    return TelemetryOnlyLabelProbe(
        num_classes=num_classes,
        components=components,
        training=training,
        seed=seed,
        device=device,
    )


def _channel_ablation_result(
    benchmark: str,
    seed: int,
    split: str,
    channel_name: str,
    predictions: np.ndarray,
    batch: AttemptBatch,
    param_count: int,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "telemetry_channel_ablation",
        "method": "activation_pca_mlp",
        "channel": channel_name,
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": param_count,
    }


def _probe_access_matrix() -> List[Dict[str, object]]:
    return [
        {
            "artifact": "single_agent",
            "visible_agent_answers": True,
            "visible_confidence": False,
            "visible_traces": False,
            "hidden_activations": False,
            "train_labels_for_fit": False,
            "dev_labels_for_early_stopping": False,
            "test_labels_or_eval_feedback": False,
        },
        {
            "artifact": "independent_swarm",
            "visible_agent_answers": True,
            "visible_confidence": True,
            "visible_traces": False,
            "hidden_activations": False,
            "train_labels_for_fit": False,
            "dev_labels_for_early_stopping": False,
            "test_labels_or_eval_feedback": False,
        },
        {
            "artifact": "text_only_coordinator",
            "visible_agent_answers": True,
            "visible_confidence": True,
            "visible_traces": False,
            "hidden_activations": False,
            "train_labels_for_fit": True,
            "dev_labels_for_early_stopping": True,
            "test_labels_or_eval_feedback": False,
        },
        {
            "artifact": "output_only_oracle_probe",
            "visible_agent_answers": True,
            "visible_confidence": True,
            "visible_traces": True,
            "hidden_activations": False,
            "train_labels_for_fit": True,
            "dev_labels_for_early_stopping": True,
            "test_labels_or_eval_feedback": False,
        },
        {
            "artifact": "telemetry_only_label_probe",
            "visible_agent_answers": False,
            "visible_confidence": False,
            "visible_traces": False,
            "hidden_activations": True,
            "train_labels_for_fit": True,
            "dev_labels_for_early_stopping": True,
            "test_labels_or_eval_feedback": False,
        },
        {
            "artifact": "hidden_state_only_probe",
            "visible_agent_answers": False,
            "visible_confidence": False,
            "visible_traces": False,
            "hidden_activations": True,
            "train_labels_for_fit": True,
            "dev_labels_for_early_stopping": True,
            "test_labels_or_eval_feedback": False,
        },
        {
            "artifact": "activation_pca_mlp",
            "visible_agent_answers": True,
            "visible_confidence": True,
            "visible_traces": False,
            "hidden_activations": True,
            "train_labels_for_fit": True,
            "dev_labels_for_early_stopping": True,
            "test_labels_or_eval_feedback": False,
        },
        {
            "artifact": "agent_correctness_probe",
            "visible_agent_answers": "varies_by_feature_mode",
            "visible_confidence": "varies_by_feature_mode",
            "visible_traces": "varies_by_feature_mode",
            "hidden_activations": "varies_by_feature_mode",
            "train_labels_for_fit": True,
            "dev_labels_for_early_stopping": True,
            "test_labels_or_eval_feedback": False,
        },
    ]


if __name__ == "__main__":
    main()
