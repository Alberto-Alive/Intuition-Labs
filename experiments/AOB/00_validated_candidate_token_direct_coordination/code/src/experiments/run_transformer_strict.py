from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, fields
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from src.agents.transformer_strict import (
    TransformerStrictConfig,
    TransformerStrictPartialEvidenceSwarm,
)
from src.agents.types import AttemptBatch
from src.coordinators.activation import ActivationPCAMLPCoordinator, TextOnlyMLPCoordinator
from src.coordinators.mlp import FeatureMLPCoordinator, MLPTrainingConfig
from src.coordinators.probes import HiddenStateOnlyLabelProbe, OutputOnlyOracleProbe
from src.coordinators.role_attention import (
    CoordinatorTokenCrossAttentionCoordinator,
    RoleAttentionConfig,
    RoleAwareAttentionCoordinator,
    RoleAwareSelfAttentionCoordinator,
)
from src.coordinators.set_hidden import (
    AttentionHiddenCoordinator,
    DeepSetsHiddenCoordinator,
    HiddenSetConfig,
    HiddenSetCoordinator,
)
from src.datasets.strict_coordination import StrictCoordinationConfig, build_strict_coordination_splits
from src.evaluation.audit import audit_event, summarize_test_access
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.report import write_report
from src.experiments.run_synthetic import (
    _make_dataclass,
    _probe_access_matrix,
    _resolve_device,
    _run_benchmark,
    _without_labels,
)
from src.telemetry.controls import apply_activation_control, randomized_train_labels
from src.telemetry.features import activation_features, output_oracle_features, visible_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 local-transformer strict benchmark.")
    parser.add_argument("--config", default="configs/transformer_strict_cuda.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
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
    dataset_config = _make_dataclass(StrictCoordinationConfig, config["strict_dataset"])
    transformer_config = _make_dataclass(
        TransformerStrictConfig,
        {**config["transformer"], "device": device},
    )
    training_config = MLPTrainingConfig(
        epochs=int(config["training"]["epochs"]),
        batch_size=int(config["training"]["batch_size"]),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        patience=int(config["training"]["patience"]),
        hidden_dims=tuple(int(x) for x in config["training"]["hidden_dims"]),
    )

    metrics: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    audit: List[Dict[str, object]] = []
    all_batches: Dict[str, Dict[str, AttemptBatch]] = {}
    slot_labels: List[str] = []
    benchmark = "transformer_strict"

    for seed in [int(x) for x in config["seeds"]]:
        print(f"seed={seed}: capturing local transformer hidden states")
        splits = build_strict_coordination_splits(dataset_config, seed=seed)
        swarm = TransformerStrictPartialEvidenceSwarm(
            config=transformer_config,
            n_evidence_bits=dataset_config.n_evidence_bits,
            num_classes=dataset_config.num_classes,
            seed=seed,
        )
        slot_labels = swarm.slot_labels
        base_batches = swarm.run(splits)
        all_batches[str(seed)] = base_batches
        extra_pretest_fitters: List[Callable[[int], Tuple[int, List[Callable[[int], int]]]]] = []
        masked_prompt_batches = None
        shuffled_prompt_batches = None
        if bool(config.get("run_stage1_mechanism_diagnostics", True)):
            extra_pretest_fitters.extend(
                [
                    lambda event_index, seed=seed, base_batches=base_batches: _run_private_evidence_bit_probes(
                        benchmark=benchmark,
                        seed=seed,
                        base_batches=base_batches,
                        pca_components=int(config["training"]["pca_components"]),
                        training_config=training_config,
                        device=device,
                        diagnostics=diagnostics,
                        audit=audit,
                        event_index=event_index,
                    ),
                    lambda event_index, seed=seed, base_batches=base_batches: _run_hidden_agent_count_curve(
                        benchmark=benchmark,
                        seed=seed,
                        base_batches=base_batches,
                        num_classes=dataset_config.num_classes,
                        training_config=training_config,
                        pca_components=int(config["training"]["pca_components"]),
                        max_agents=int(config.get("agent_count_curve_max_agents", 5)),
                        device=device,
                        diagnostics=diagnostics,
                        audit=audit,
                        event_index=event_index,
                    ),
                    lambda event_index, seed=seed, base_batches=base_batches, slot_labels=slot_labels: _run_hidden_location_probes(
                        benchmark=benchmark,
                        seed=seed,
                        base_batches=base_batches,
                        slot_labels=slot_labels,
                        num_classes=dataset_config.num_classes,
                        training_config=training_config,
                        pca_components=int(config["training"]["pca_components"]),
                        device=device,
                        diagnostics=diagnostics,
                        audit=audit,
                        event_index=event_index,
                    ),
                    lambda event_index, seed=seed, base_batches=base_batches: _run_output_leakage_audit(
                        benchmark=benchmark,
                        seed=seed,
                        base_batches=base_batches,
                        num_classes=dataset_config.num_classes,
                        num_tasks=dataset_config.num_tasks,
                        training_config=training_config,
                        device=device,
                        diagnostics=diagnostics,
                        audit=audit,
                        event_index=event_index,
                    ),
                    lambda event_index, seed=seed, base_batches=base_batches: _run_activation_stability_diagnostics(
                        benchmark=benchmark,
                        seed=seed,
                        base_batches=base_batches,
                        num_classes=dataset_config.num_classes,
                        num_tasks=dataset_config.num_tasks,
                        training_config=training_config,
                        pca_components=int(config["training"]["pca_components"]),
                        probe_seed_offsets=[
                            int(value)
                            for value in config.get("activation_stability_probe_seed_offsets", [0, 1, 2, 3, 4])
                        ],
                        fixed_training_seed=int(config.get("activation_stability_fixed_training_seed", 777_001)),
                        device=device,
                        diagnostics=diagnostics,
                        audit=audit,
                        event_index=event_index,
                    ),
                ]
            )
            if bool(config.get("run_prompt_evidence_controls", True)):
                print(f"seed={seed}: capturing masked and shuffled private-evidence prompts")
                masked_prompt_batches = swarm.run_masked_evidence(splits)
                shuffled_prompt_batches = swarm.run_shuffled_evidence(splits, seed=seed + 770_000)
                extra_pretest_fitters.append(
                    lambda event_index, seed=seed, base_batches=base_batches, masked_prompt_batches=masked_prompt_batches, shuffled_prompt_batches=shuffled_prompt_batches: _run_evidence_causal_controls(
                        benchmark=benchmark,
                        seed=seed,
                        base_batches=base_batches,
                        masked_prompt_batches=masked_prompt_batches,
                        shuffled_prompt_batches=shuffled_prompt_batches,
                        num_classes=dataset_config.num_classes,
                        training_config=training_config,
                        pca_components=int(config["training"]["pca_components"]),
                        device=device,
                        diagnostics=diagnostics,
                        audit=audit,
                        event_index=event_index,
                    )
                )
        if bool(config.get("run_stage4_hidden_coordinators", False)):
            extra_pretest_fitters.append(
                lambda event_index, seed=seed, base_batches=base_batches, slot_labels=slot_labels, masked_prompt_batches=masked_prompt_batches, shuffled_prompt_batches=shuffled_prompt_batches: _run_stage4_hidden_coordinators(
                    benchmark=benchmark,
                    seed=seed,
                    base_batches=base_batches,
                    slot_labels=slot_labels,
                    masked_prompt_batches=masked_prompt_batches,
                    shuffled_prompt_batches=shuffled_prompt_batches,
                    num_classes=dataset_config.num_classes,
                    training_config=training_config,
                    device=device,
                    diagnostics=diagnostics,
                    audit=audit,
                    event_index=event_index,
                    agent_dropout=float(config.get("stage4_agent_dropout", 0.25)),
                )
            )
        if bool(config.get("run_stage5_role_attention_coordinators", False)):
            extra_pretest_fitters.append(
                lambda event_index, seed=seed, base_batches=base_batches, slot_labels=slot_labels, masked_prompt_batches=masked_prompt_batches, shuffled_prompt_batches=shuffled_prompt_batches: _run_stage5_role_attention_coordinators(
                    benchmark=benchmark,
                    seed=seed,
                    base_batches=base_batches,
                    slot_labels=slot_labels,
                    masked_prompt_batches=masked_prompt_batches,
                    shuffled_prompt_batches=shuffled_prompt_batches,
                    num_classes=dataset_config.num_classes,
                    training_config=training_config,
                    device=device,
                    diagnostics=diagnostics,
                    audit=audit,
                    event_index=event_index,
                    variants=list(config.get("stage5_variants", _default_stage5_variants())),
                    agent_dropout=float(config.get("stage5_agent_dropout", 0.25)),
                    aux_loss_weight=float(config.get("stage5_aux_loss_weight", 0.2)),
                )
            )
        if bool(config.get("run_stage3_robustness_diagnostics", False)):
            if _seed_enabled(seed, config.get("prompt_robustness_seeds")):
                extra_pretest_fitters.append(
                    lambda event_index, seed=seed, splits=splits, transformer_config=transformer_config: _run_prompt_robustness_diagnostics(
                        benchmark=benchmark,
                        seed=seed,
                        splits=splits,
                        transformer_config=transformer_config,
                        prompt_variants=list(config.get("prompt_robustness_variants", [])),
                        n_evidence_bits=dataset_config.n_evidence_bits,
                        num_classes=dataset_config.num_classes,
                        num_tasks=dataset_config.num_tasks,
                        training_config=training_config,
                        pca_components=int(config["training"]["pca_components"]),
                        device=device,
                        diagnostics=diagnostics,
                        audit=audit,
                        event_index=event_index,
                        include_stage4=bool(config.get("run_stage4_hidden_coordinators", False)),
                        stage4_agent_dropout=float(config.get("stage4_agent_dropout", 0.25)),
                        include_stage5=bool(config.get("run_stage5_role_attention_coordinators", False)),
                        stage5_variants=list(config.get("stage5_prompt_variants", _default_stage5_prompt_variants())),
                        stage5_agent_dropout=float(config.get("stage5_agent_dropout", 0.25)),
                        stage5_aux_loss_weight=float(config.get("stage5_aux_loss_weight", 0.2)),
                    )
                )
            if _seed_enabled(seed, config.get("model_robustness_seeds")):
                extra_pretest_fitters.append(
                    lambda event_index, seed=seed, splits=splits, transformer_config=transformer_config: _run_model_robustness_diagnostics(
                        benchmark=benchmark,
                        seed=seed,
                        splits=splits,
                        base_transformer_config=transformer_config,
                        model_variants=list(config.get("model_robustness_variants", [])),
                        n_evidence_bits=dataset_config.n_evidence_bits,
                        num_classes=dataset_config.num_classes,
                        num_tasks=dataset_config.num_tasks,
                        training_config=training_config,
                        pca_components=int(config["training"]["pca_components"]),
                        device=device,
                        diagnostics=diagnostics,
                        audit=audit,
                        event_index=event_index,
                    )
                )
            if _seed_enabled(seed, config.get("evidence_sharing_seeds")):
                sharing_swarm = TransformerStrictPartialEvidenceSwarm(
                    config=_variant_transformer_config(
                        transformer_config,
                        {
                            "answer_format": "default",
                            "visible_evidence": True,
                        },
                    ),
                    n_evidence_bits=dataset_config.n_evidence_bits,
                    num_classes=dataset_config.num_classes,
                    seed=seed,
                )
                sharing_batches = sharing_swarm.run(splits)
                extra_pretest_fitters.append(
                    lambda event_index, seed=seed, sharing_batches=sharing_batches: _run_explicit_evidence_sharing_baseline(
                        benchmark=benchmark,
                        seed=seed,
                        sharing_batches=sharing_batches,
                        num_classes=dataset_config.num_classes,
                        num_tasks=dataset_config.num_tasks,
                        training_config=training_config,
                        pca_components=int(config["training"]["pca_components"]),
                        device=device,
                        diagnostics=diagnostics,
                        audit=audit,
                        event_index=event_index,
                    )
                )
            extra_pretest_fitters.append(
                lambda event_index, seed=seed, base_batches=base_batches: _run_stage3_probe_family_diagnostics(
                    benchmark=benchmark,
                    seed=seed,
                    base_batches=base_batches,
                    num_classes=dataset_config.num_classes,
                    num_tasks=dataset_config.num_tasks,
                    training_config=training_config,
                    pca_components=int(config["training"]["pca_components"]),
                    device=device,
                    diagnostics=diagnostics,
                    audit=audit,
                    event_index=event_index,
                )
            )
        if bool(config.get("run_layer_position_ablations", True)):
            extra_pretest_fitters.append(
                lambda event_index, seed=seed, base_batches=base_batches, slot_labels=slot_labels: _run_hidden_slice_ablations(
                    benchmark=benchmark,
                    seed=seed,
                    base_batches=base_batches,
                    slot_labels=slot_labels,
                    num_classes=dataset_config.num_classes,
                    num_tasks=dataset_config.num_tasks,
                    training_config=training_config,
                    pca_components=int(config["training"]["pca_components"]),
                    device=device,
                    diagnostics=diagnostics,
                    audit=audit,
                    event_index=event_index,
                )
            )
        if bool(config.get("run_individual_hidden_label_probe", True)):
            extra_pretest_fitters.append(
                lambda event_index, seed=seed, base_batches=base_batches: _run_individual_hidden_label_probe(
                    benchmark=benchmark,
                    seed=seed,
                    base_batches=base_batches,
                    num_classes=dataset_config.num_classes,
                    training_config=training_config,
                    device=device,
                    diagnostics=diagnostics,
                    audit=audit,
                    event_index=event_index,
                )
            )
        _run_benchmark(
            benchmark=benchmark,
            seed=seed,
            base_batches=base_batches,
            dataset_config=dataset_config,
            training_config=training_config,
            config=config,
            device=device,
            all_metrics=metrics,
            all_diagnostics=diagnostics,
            all_audits=audit,
            extra_pretest_fitters=extra_pretest_fitters,
        )

    audit.append(summarize_test_access(audit))
    validation = {
        "probe_access": _transformer_probe_access_matrix(),
        "transformer_hidden_state": {
            "source": "local_huggingface_model",
            "model_name_or_path": transformer_config.model_name_or_path,
            "layers": list(transformer_config.layers),
            "token_positions": list(transformer_config.token_positions),
            "slot_labels": slot_labels,
            "agent_visible_outputs_sanitized": True,
            "final_label_in_prompt": False,
        },
    }
    results = {
        "metadata": {
            "stage": "stage1_transformer_strict",
            "config_path": str(Path(args.config)),
            "config": config,
            "strict_dataset_config": asdict(dataset_config),
            "transformer_config": asdict(transformer_config),
            "training_config": {
                **asdict(training_config),
                "hidden_dims": list(training_config.hidden_dims),
            },
            "hardware": hardware,
            "device_used": device,
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
        "audit": audit,
        "validation": validation,
    }

    output_path = Path(str(config["output_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    _write_access_artifacts(config, all_batches, slot_labels)
    merged = _merge_with_existing(results, config)
    write_report(merged, config["report_path"])
    print(f"wrote {output_path}")
    print(f"wrote {config['report_path']}")


def _run_private_evidence_bit_probes(
    benchmark: str,
    seed: int,
    base_batches: Dict[str, AttemptBatch],
    pca_components: int,
    training_config: MLPTrainingConfig,
    device: str,
    diagnostics: List[Dict[str, object]],
    audit: List[Dict[str, object]],
    event_index: int,
) -> Tuple[int, List[Callable[[int], int]]]:
    fitted = []
    for agent_id in range(base_batches["train"].n_agents):
        train = _select_single_agent(base_batches["train"], agent_id).copy_with_labels(
            _private_bit_labels(base_batches["train"], agent_id)
        )
        dev = _select_single_agent(base_batches["dev"], agent_id).copy_with_labels(
            _private_bit_labels(base_batches["dev"], agent_id)
        )
        method = HiddenStateOnlyLabelProbe(
            num_classes=2,
            components=pca_components,
            training=training_config,
            seed=seed + 410_000 + agent_id,
            device=device,
        )
        audit.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "private_evidence_bit_probe_fit",
                method=method.name,
                condition=f"agent_{agent_id}",
                split_inputs=("train", "dev"),
            )
        )
        event_index += 1
        method.fit(train, dev)
        preds = method.predict(_without_labels(dev))
        diagnostics.append(
            _private_bit_probe_result(
                benchmark,
                seed,
                "dev",
                agent_id,
                "single_agent_hidden",
                preds,
                dev.labels,
                int(getattr(method, "param_count", 0)),
            )
        )
        fitted.append((agent_id, method))

    def evaluate_test(current_event_index: int) -> int:
        for agent_id, method in fitted:
            test = _select_single_agent(base_batches["test"], agent_id).copy_with_labels(
                _private_bit_labels(base_batches["test"], agent_id)
            )
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "private_evidence_bit_probe_test_predict",
                    method=method.name,
                    condition=f"agent_{agent_id}",
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(test))
            diagnostics.append(
                _private_bit_probe_result(
                    benchmark,
                    seed,
                    "test",
                    agent_id,
                    "single_agent_hidden",
                    preds,
                    test.labels,
                    int(getattr(method, "param_count", 0)),
                )
            )
        return current_event_index

    return event_index, [evaluate_test]


def _run_hidden_agent_count_curve(
    benchmark: str,
    seed: int,
    base_batches: Dict[str, AttemptBatch],
    num_classes: int,
    training_config: MLPTrainingConfig,
    pca_components: int,
    max_agents: int,
    device: str,
    diagnostics: List[Dict[str, object]],
    audit: List[Dict[str, object]],
    event_index: int,
) -> Tuple[int, List[Callable[[int], int]]]:
    fitted = []
    available_agents = base_batches["train"].hidden_states.shape[1]
    for agent_count in range(1, min(max_agents, available_agents) + 1):
        train = _select_agents(base_batches["train"], agent_count)
        dev = _select_agents(base_batches["dev"], agent_count)
        method = HiddenStateOnlyLabelProbe(
            num_classes=num_classes,
            components=pca_components,
            training=training_config,
            seed=seed + 420_000 + agent_count,
            device=device,
        )
        audit.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "hidden_agent_count_probe_fit",
                method=method.name,
                condition=f"{agent_count}_agents",
                split_inputs=("train", "dev"),
            )
        )
        event_index += 1
        method.fit(train, dev)
        preds = method.predict(_without_labels(dev))
        diagnostics.append(
            _agent_count_result(
                benchmark,
                seed,
                "dev",
                agent_count,
                available_agents,
                preds,
                dev,
                int(getattr(method, "param_count", 0)),
            )
        )
        fitted.append((agent_count, method))
    if max_agents > available_agents:
        diagnostics.append(
            {
                "benchmark": benchmark,
                "seed": seed,
                "split": "test",
                "probe": "hidden_state_agent_count_curve",
                "agent_count": max_agents,
                "available_agents": available_agents,
                "status": "not_run_exceeds_task_agent_count",
                "accuracy": None,
                "n_examples": base_batches["test"].n_examples,
                "param_count": 0,
            }
        )

    def evaluate_test(current_event_index: int) -> int:
        for agent_count, method in fitted:
            test = _select_agents(base_batches["test"], agent_count)
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "hidden_agent_count_probe_test_predict",
                    method=method.name,
                    condition=f"{agent_count}_agents",
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(test))
            diagnostics.append(
                _agent_count_result(
                    benchmark,
                    seed,
                    "test",
                    agent_count,
                    available_agents,
                    preds,
                    test,
                    int(getattr(method, "param_count", 0)),
                )
            )
        return current_event_index

    return event_index, [evaluate_test]


def _run_hidden_location_probes(
    benchmark: str,
    seed: int,
    base_batches: Dict[str, AttemptBatch],
    slot_labels: List[str],
    num_classes: int,
    training_config: MLPTrainingConfig,
    pca_components: int,
    device: str,
    diagnostics: List[Dict[str, object]],
    audit: List[Dict[str, object]],
    event_index: int,
) -> Tuple[int, List[Callable[[int], int]]]:
    selections = _hidden_location_selections(slot_labels)
    fitted = []
    for selection_id, (axis, value, indices) in enumerate(selections):
        if not indices:
            continue
        train = _select_slots(base_batches["train"], indices)
        dev = _select_slots(base_batches["dev"], indices)
        method = HiddenStateOnlyLabelProbe(
            num_classes=num_classes,
            components=pca_components,
            training=training_config,
            seed=seed + 430_000 + selection_id,
            device=device,
        )
        audit.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "hidden_location_probe_fit",
                method=method.name,
                condition=f"{axis}_{value}",
                split_inputs=("train", "dev"),
            )
        )
        event_index += 1
        method.fit(train, dev)
        preds = method.predict(_without_labels(dev))
        diagnostics.append(
            _hidden_location_result(
                benchmark,
                seed,
                "dev",
                axis,
                value,
                indices,
                preds,
                dev,
                int(getattr(method, "param_count", 0)),
            )
        )
        fitted.append((axis, value, indices, method))

    def evaluate_test(current_event_index: int) -> int:
        for axis, value, indices, method in fitted:
            test = _select_slots(base_batches["test"], indices)
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "hidden_location_probe_test_predict",
                    method=method.name,
                    condition=f"{axis}_{value}",
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(test))
            diagnostics.append(
                _hidden_location_result(
                    benchmark,
                    seed,
                    "test",
                    axis,
                    value,
                    indices,
                    preds,
                    test,
                    int(getattr(method, "param_count", 0)),
                )
            )
        return current_event_index

    return event_index, [evaluate_test]


def _run_output_leakage_audit(
    benchmark: str,
    seed: int,
    base_batches: Dict[str, AttemptBatch],
    num_classes: int,
    num_tasks: int,
    training_config: MLPTrainingConfig,
    device: str,
    diagnostics: List[Dict[str, object]],
    audit: List[Dict[str, object]],
    event_index: int,
) -> Tuple[int, List[Callable[[int], int]]]:
    for split, batch in base_batches.items():
        diagnostics.append(_visible_output_audit_result(benchmark, seed, split, batch))

    fitted = []
    for agent_id in range(base_batches["train"].n_agents):
        train = base_batches["train"].copy_with_labels(_private_bit_labels(base_batches["train"], agent_id))
        dev = base_batches["dev"].copy_with_labels(_private_bit_labels(base_batches["dev"], agent_id))
        method = FeatureMLPCoordinator(
            name="output_only_private_evidence_probe",
            feature_builder=lambda batch: output_oracle_features(batch, num_classes, num_tasks),
            num_classes=2,
            training=training_config,
            seed=seed + 440_000 + agent_id,
            device=device,
        )
        audit.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "output_only_private_evidence_probe_fit",
                method=method.name,
                condition=f"agent_{agent_id}",
                split_inputs=("train", "dev"),
            )
        )
        event_index += 1
        method.fit(train, dev)
        preds = method.predict(_without_labels(dev))
        diagnostics.append(
            _private_bit_probe_result(
                benchmark,
                seed,
                "dev",
                agent_id,
                "output_only",
                preds,
                dev.labels,
                int(getattr(method, "param_count", 0)),
            )
        )
        fitted.append((agent_id, method))

    def evaluate_test(current_event_index: int) -> int:
        for agent_id, method in fitted:
            test = base_batches["test"].copy_with_labels(_private_bit_labels(base_batches["test"], agent_id))
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "output_only_private_evidence_probe_test_predict",
                    method=method.name,
                    condition=f"agent_{agent_id}",
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(test))
            diagnostics.append(
                _private_bit_probe_result(
                    benchmark,
                    seed,
                    "test",
                    agent_id,
                    "output_only",
                    preds,
                    test.labels,
                    int(getattr(method, "param_count", 0)),
                )
            )
        return current_event_index

    return event_index, [evaluate_test]


def _run_evidence_causal_controls(
    benchmark: str,
    seed: int,
    base_batches: Dict[str, AttemptBatch],
    masked_prompt_batches: Dict[str, AttemptBatch],
    shuffled_prompt_batches: Dict[str, AttemptBatch],
    num_classes: int,
    training_config: MLPTrainingConfig,
    pca_components: int,
    device: str,
    diagnostics: List[Dict[str, object]],
    audit: List[Dict[str, object]],
    event_index: int,
) -> Tuple[int, List[Callable[[int], int]]]:
    hidden_shuffled_batches = apply_activation_control(
        base_batches,
        control="shuffled_activations",
        seed=seed + 450_000,
    )
    controls = [
        ("mask_private_evidence_prompt", masked_prompt_batches),
        ("shuffle_private_evidence_prompt", shuffled_prompt_batches),
        ("shuffle_agent_hidden_states", hidden_shuffled_batches),
    ]
    fitted = []
    for control_id, (condition, batches) in enumerate(controls):
        method = HiddenStateOnlyLabelProbe(
            num_classes=num_classes,
            components=pca_components,
            training=training_config,
            seed=seed + 450_000 + control_id,
            device=device,
        )
        audit.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "evidence_causal_control_fit",
                method=method.name,
                condition=condition,
                split_inputs=("train", "dev"),
            )
        )
        event_index += 1
        method.fit(batches["train"], batches["dev"])
        preds = method.predict(_without_labels(batches["dev"]))
        diagnostics.append(
            _evidence_control_result(
                benchmark,
                seed,
                "dev",
                condition,
                preds,
                batches["dev"],
                int(getattr(method, "param_count", 0)),
            )
        )
        fitted.append((condition, batches, method))

    def evaluate_test(current_event_index: int) -> int:
        for condition, batches, method in fitted:
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "evidence_causal_control_test_predict",
                    method=method.name,
                    condition=condition,
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(batches["test"]))
            diagnostics.append(
                _evidence_control_result(
                    benchmark,
                    seed,
                    "test",
                    condition,
                    preds,
                    batches["test"],
                    int(getattr(method, "param_count", 0)),
                )
            )
        return current_event_index

    return event_index, [evaluate_test]


def _run_activation_stability_diagnostics(
    benchmark: str,
    seed: int,
    base_batches: Dict[str, AttemptBatch],
    num_classes: int,
    num_tasks: int,
    training_config: MLPTrainingConfig,
    pca_components: int,
    probe_seed_offsets: List[int],
    fixed_training_seed: int,
    device: str,
    diagnostics: List[Dict[str, object]],
    audit: List[Dict[str, object]],
    event_index: int,
) -> Tuple[int, List[Callable[[int], int]]]:
    fitted = []
    specs: List[Tuple[str, str, int, object]] = []
    for offset in probe_seed_offsets:
        specs.extend(
            [
                (
                    "activation_pca_mlp",
                    "vary_probe_training_seed",
                    seed + 460_000 + offset,
                    ActivationPCAMLPCoordinator(
                        num_classes=num_classes,
                        num_tasks=num_tasks,
                        components=pca_components,
                        training=training_config,
                        seed=seed + 460_000 + offset,
                        device=device,
                    ),
                ),
                (
                    "hidden_state_only_probe",
                    "vary_probe_training_seed",
                    seed + 470_000 + offset,
                    HiddenStateOnlyLabelProbe(
                        num_classes=num_classes,
                        components=pca_components,
                        training=training_config,
                        seed=seed + 470_000 + offset,
                        device=device,
                    ),
                ),
            ]
        )
    specs.extend(
        [
            (
                "activation_pca_mlp",
                "fixed_probe_training_seed",
                fixed_training_seed,
                ActivationPCAMLPCoordinator(
                    num_classes=num_classes,
                    num_tasks=num_tasks,
                    components=pca_components,
                    training=training_config,
                    seed=fixed_training_seed,
                    device=device,
                ),
            ),
            (
                "hidden_state_only_probe",
                "fixed_probe_training_seed",
                fixed_training_seed,
                HiddenStateOnlyLabelProbe(
                    num_classes=num_classes,
                    components=pca_components,
                    training=training_config,
                    seed=fixed_training_seed,
                    device=device,
                ),
            ),
        ]
    )
    for method_name, variant, training_seed, method in specs:
        audit.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "activation_stability_fit",
                method=method_name,
                condition=variant,
                split_inputs=("train", "dev"),
            )
        )
        event_index += 1
        method.fit(base_batches["train"], base_batches["dev"])
        preds = method.predict(_without_labels(base_batches["dev"]))
        diagnostics.append(
            _activation_stability_result(
                benchmark,
                seed,
                "dev",
                method_name,
                variant,
                training_seed,
                preds,
                base_batches["dev"],
                int(getattr(method, "param_count", 0)),
            )
        )
        fitted.append((method_name, variant, training_seed, method))

    def evaluate_test(current_event_index: int) -> int:
        for method_name, variant, training_seed, method in fitted:
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "activation_stability_test_predict",
                    method=method_name,
                    condition=variant,
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(base_batches["test"]))
            diagnostics.append(
                _activation_stability_result(
                    benchmark,
                    seed,
                    "test",
                    method_name,
                    variant,
                    training_seed,
                    preds,
                    base_batches["test"],
                    int(getattr(method, "param_count", 0)),
                )
            )
        return current_event_index

    return event_index, [evaluate_test]


def _run_stage4_hidden_coordinators(
    benchmark: str,
    seed: int,
    base_batches: Dict[str, AttemptBatch],
    slot_labels: List[str],
    masked_prompt_batches: Dict[str, AttemptBatch] | None,
    shuffled_prompt_batches: Dict[str, AttemptBatch] | None,
    num_classes: int,
    training_config: MLPTrainingConfig,
    device: str,
    diagnostics: List[Dict[str, object]],
    audit: List[Dict[str, object]],
    event_index: int,
    agent_dropout: float,
) -> Tuple[int, List[Callable[[int], int]]]:
    set_config = HiddenSetConfig(
        slot_index=_late_final_slot_index(slot_labels),
        agent_dropout=agent_dropout,
    )
    method_specs = [
        (
            "deepsets_hidden_coordinator",
            lambda method_seed: DeepSetsHiddenCoordinator(
                num_classes=num_classes,
                training=training_config,
                seed=method_seed,
                device=device,
                config=set_config,
            ),
        ),
        (
            "attention_hidden_coordinator",
            lambda method_seed: AttentionHiddenCoordinator(
                num_classes=num_classes,
                training=training_config,
                seed=method_seed,
                device=device,
                config=set_config,
            ),
        ),
    ]
    fitted: List[HiddenSetCoordinator] = []
    for method_index, (_method_name, factory) in enumerate(method_specs):
        method = factory(seed + 600_000 + method_index)
        audit.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "stage4_hidden_coordinator_fit",
                method=method.name,
                condition="all_agents",
                split_inputs=("train", "dev"),
            )
        )
        event_index += 1
        method.fit(base_batches["train"], base_batches["dev"])
        dev_preds = method.predict(_without_labels(base_batches["dev"]))
        diagnostics.append(
            _stage4_accuracy_result(
                benchmark,
                seed,
                "dev",
                method,
                "all_agents",
                dev_preds,
                base_batches["dev"],
                set_config,
            )
        )
        for agent_count in range(1, base_batches["dev"].n_agents + 1):
            mask = _agent_count_mask(base_batches["dev"], agent_count)
            preds = method.predict_with_mask(_without_labels(base_batches["dev"]), mask)
            diagnostics.append(
                _stage4_agent_count_result(
                    benchmark,
                    seed,
                    "dev",
                    method,
                    agent_count,
                    preds,
                    base_batches["dev"],
                    set_config,
                )
            )
        fitted.append(method)

    randomized_fitted: List[HiddenSetCoordinator] = []
    randomized_train = randomized_train_labels(base_batches["train"], seed=seed + 600_100, num_classes=num_classes)
    for method_index, (_method_name, factory) in enumerate(method_specs):
        method = factory(seed + 601_000 + method_index)
        audit.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "stage4_label_permutation_fit",
                method=method.name,
                condition="randomized_train_labels",
                split_inputs=("train",),
            )
        )
        event_index += 1
        method.fit(randomized_train, None)
        randomized_fitted.append(method)

    control_fitted: List[Tuple[str, HiddenSetCoordinator, Dict[str, AttemptBatch]]] = []
    controls: List[Tuple[str, Dict[str, AttemptBatch]]] = [
        (
            "shuffle_agent_hidden_states",
            apply_activation_control(base_batches, control="shuffled_activations", seed=seed + 602_000),
        )
    ]
    if masked_prompt_batches is not None:
        controls.append(("mask_private_evidence_prompt", masked_prompt_batches))
    if shuffled_prompt_batches is not None:
        controls.append(("shuffle_private_evidence_prompt", shuffled_prompt_batches))
    for control_index, (condition, batches) in enumerate(controls):
        for method_index, (_method_name, factory) in enumerate(method_specs):
            method = factory(seed + 603_000 + control_index * 37 + method_index)
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    event_index,
                    "stage4_evidence_control_fit",
                    method=method.name,
                    condition=condition,
                    split_inputs=("train", "dev"),
                )
            )
            event_index += 1
            method.fit(batches["train"], batches["dev"])
            dev_preds = method.predict(_without_labels(batches["dev"]))
            diagnostics.append(
                _stage4_control_result(
                    benchmark,
                    seed,
                    "dev",
                    method,
                    condition,
                    dev_preds,
                    batches["dev"],
                    set_config,
                )
            )
            control_fitted.append((condition, method, batches))

    def evaluate_test(current_event_index: int) -> int:
        base_predictions: Dict[str, np.ndarray] = {}
        for method in fitted:
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "stage4_hidden_coordinator_test_predict",
                    method=method.name,
                    condition="all_agents",
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(base_batches["test"]))
            base_predictions[method.name] = preds
            diagnostics.append(
                _stage4_accuracy_result(
                    benchmark,
                    seed,
                    "test",
                    method,
                    "all_agents",
                    preds,
                    base_batches["test"],
                    set_config,
                )
            )
            for agent_count in range(1, base_batches["test"].n_agents + 1):
                mask = _agent_count_mask(base_batches["test"], agent_count)
                count_preds = method.predict_with_mask(_without_labels(base_batches["test"]), mask)
                diagnostics.append(
                    _stage4_agent_count_result(
                        benchmark,
                        seed,
                        "test",
                        method,
                        agent_count,
                        count_preds,
                        base_batches["test"],
                        set_config,
                    )
                )
            perm_preds = method.predict_with_permutation(
                _without_labels(base_batches["test"]),
                seed=seed + 604_000,
            )
            diagnostics.append(
                _stage4_permutation_result(
                    benchmark,
                    seed,
                    method,
                    preds,
                    perm_preds,
                    base_batches["test"],
                    set_config,
                )
            )
        for method in randomized_fitted:
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "stage4_label_permutation_test_predict",
                    method=method.name,
                    condition="randomized_train_labels",
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(base_batches["test"]))
            diagnostics.append(
                _stage4_label_permutation_result(
                    benchmark,
                    seed,
                    method,
                    preds,
                    base_batches["test"],
                    set_config,
                )
            )
        for condition, method, batches in control_fitted:
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "stage4_evidence_control_test_predict",
                    method=method.name,
                    condition=condition,
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(batches["test"]))
            diagnostics.append(
                _stage4_control_result(
                    benchmark,
                    seed,
                    "test",
                    method,
                    condition,
                    preds,
                    batches["test"],
                    set_config,
                )
            )
        return current_event_index

    return event_index, [evaluate_test]


def _run_stage5_role_attention_coordinators(
    benchmark: str,
    seed: int,
    base_batches: Dict[str, AttemptBatch],
    slot_labels: List[str],
    masked_prompt_batches: Dict[str, AttemptBatch] | None,
    shuffled_prompt_batches: Dict[str, AttemptBatch] | None,
    num_classes: int,
    training_config: MLPTrainingConfig,
    device: str,
    diagnostics: List[Dict[str, object]],
    audit: List[Dict[str, object]],
    event_index: int,
    variants: List[object],
    agent_dropout: float,
    aux_loss_weight: float,
) -> Tuple[int, List[Callable[[int], int]]]:
    variant_specs = _stage5_variant_specs(
        variants=variants,
        slot_labels=slot_labels,
        agent_dropout=agent_dropout,
        aux_loss_weight=aux_loss_weight,
    )
    fitted: List[RoleAwareAttentionCoordinator] = []
    for variant_index, role_config in enumerate(variant_specs):
        method = _make_stage5_method(
            role_config=role_config,
            num_classes=num_classes,
            training_config=training_config,
            seed=seed + 700_000 + variant_index,
            device=device,
        )
        audit.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "stage5_role_attention_fit",
                method=method.name,
                condition="all_agents",
                split_inputs=("train", "dev"),
            )
        )
        event_index += 1
        method.fit(base_batches["train"], base_batches["dev"])
        dev_preds = method.predict(_without_labels(base_batches["dev"]))
        diagnostics.append(
            _stage5_accuracy_result(
                benchmark,
                seed,
                "dev",
                method,
                "all_agents",
                dev_preds,
                base_batches["dev"],
            )
        )
        for agent_count in range(1, base_batches["dev"].n_agents + 1):
            mask = _agent_count_mask(base_batches["dev"], agent_count)
            preds = method.predict_with_mask(_without_labels(base_batches["dev"]), mask)
            diagnostics.append(
                _stage5_agent_count_result(
                    benchmark,
                    seed,
                    "dev",
                    method,
                    agent_count,
                    preds,
                    base_batches["dev"],
                )
            )
        fitted.append(method)

    randomized_fitted: List[RoleAwareAttentionCoordinator] = []
    randomized_train = randomized_train_labels(base_batches["train"], seed=seed + 710_100, num_classes=num_classes)
    for variant_index, role_config in enumerate(variant_specs):
        method = _make_stage5_method(
            role_config=role_config,
            num_classes=num_classes,
            training_config=training_config,
            seed=seed + 711_000 + variant_index,
            device=device,
        )
        audit.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "stage5_label_permutation_fit",
                method=method.name,
                condition="randomized_train_labels",
                split_inputs=("train",),
            )
        )
        event_index += 1
        method.fit(randomized_train, None)
        randomized_fitted.append(method)

    control_fitted: List[Tuple[str, RoleAwareAttentionCoordinator, Dict[str, AttemptBatch]]] = []
    controls: List[Tuple[str, Dict[str, AttemptBatch]]] = [
        (
            "shuffle_agent_hidden_states",
            apply_activation_control(base_batches, control="shuffled_activations", seed=seed + 712_000),
        )
    ]
    if masked_prompt_batches is not None:
        controls.append(("mask_private_evidence_prompt", masked_prompt_batches))
    if shuffled_prompt_batches is not None:
        controls.append(("shuffle_private_evidence_prompt", shuffled_prompt_batches))
    for control_index, (condition, batches) in enumerate(controls):
        for variant_index, role_config in enumerate(variant_specs):
            method = _make_stage5_method(
                role_config=role_config,
                num_classes=num_classes,
                training_config=training_config,
                seed=seed + 713_000 + control_index * 101 + variant_index,
                device=device,
            )
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    event_index,
                    "stage5_evidence_control_fit",
                    method=method.name,
                    condition=condition,
                    split_inputs=("train", "dev"),
                )
            )
            event_index += 1
            method.fit(batches["train"], batches["dev"])
            dev_preds = method.predict(_without_labels(batches["dev"]))
            diagnostics.append(
                _stage5_control_result(
                    benchmark,
                    seed,
                    "dev",
                    method,
                    condition,
                    dev_preds,
                    batches["dev"],
                )
            )
            control_fitted.append((condition, method, batches))

    def evaluate_test(current_event_index: int) -> int:
        base_predictions: Dict[str, np.ndarray] = {}
        for method in fitted:
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "stage5_role_attention_test_predict",
                    method=method.name,
                    condition="all_agents",
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(base_batches["test"]))
            base_predictions[method.name] = preds
            diagnostics.append(
                _stage5_accuracy_result(
                    benchmark,
                    seed,
                    "test",
                    method,
                    "all_agents",
                    preds,
                    base_batches["test"],
                )
            )
            for agent_count in range(1, base_batches["test"].n_agents + 1):
                mask = _agent_count_mask(base_batches["test"], agent_count)
                count_preds = method.predict_with_mask(_without_labels(base_batches["test"]), mask)
                diagnostics.append(
                    _stage5_agent_count_result(
                        benchmark,
                        seed,
                        "test",
                        method,
                        agent_count,
                        count_preds,
                        base_batches["test"],
                    )
                )
            perm_preds = method.predict_with_permutation(
                _without_labels(base_batches["test"]),
                seed=seed + 714_000,
            )
            diagnostics.append(
                _stage5_permutation_result(
                    benchmark,
                    seed,
                    method,
                    "physical_order_permutation",
                    preds,
                    perm_preds,
                    base_batches["test"],
                )
            )
            slot_shuffle_preds = method.predict_with_slot_label_shuffle(
                _without_labels(base_batches["test"]),
                seed=seed + 715_000,
            )
            diagnostics.append(
                _stage5_permutation_result(
                    benchmark,
                    seed,
                    method,
                    "slot_label_shuffle",
                    preds,
                    slot_shuffle_preds,
                    base_batches["test"],
                )
            )
        for method in randomized_fitted:
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "stage5_label_permutation_test_predict",
                    method=method.name,
                    condition="randomized_train_labels",
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(base_batches["test"]))
            diagnostics.append(
                _stage5_label_permutation_result(
                    benchmark,
                    seed,
                    method,
                    preds,
                    base_batches["test"],
                )
            )
        for condition, method, batches in control_fitted:
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "stage5_evidence_control_test_predict",
                    method=method.name,
                    condition=condition,
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(batches["test"]))
            diagnostics.append(
                _stage5_control_result(
                    benchmark,
                    seed,
                    "test",
                    method,
                    condition,
                    preds,
                    batches["test"],
                )
            )
        return current_event_index

    return event_index, [evaluate_test]


def _default_stage5_variants() -> List[Dict[str, object]]:
    return [
        {"family": "self_attention", "layers": 1, "heads": 1, "pooling": "cls", "feature_mode": "late_final", "aux": False},
        {"family": "self_attention", "layers": 2, "heads": 1, "pooling": "cls", "feature_mode": "late_final", "aux": False},
        {"family": "self_attention", "layers": 1, "heads": 4, "pooling": "cls", "feature_mode": "late_final", "aux": False},
        {"family": "self_attention", "layers": 1, "heads": 1, "pooling": "mean", "feature_mode": "late_final", "aux": False},
        {"family": "self_attention", "layers": 1, "heads": 1, "pooling": "cls", "feature_mode": "final_bundle", "aux": False},
        {"family": "self_attention", "layers": 1, "heads": 1, "pooling": "cls", "feature_mode": "late_final", "aux": True},
        {"family": "cross_attention", "layers": 1, "heads": 1, "pooling": "coordinator", "feature_mode": "late_final", "aux": False},
        {"family": "cross_attention", "layers": 2, "heads": 1, "pooling": "coordinator", "feature_mode": "late_final", "aux": False},
        {"family": "cross_attention", "layers": 1, "heads": 4, "pooling": "coordinator", "feature_mode": "late_final", "aux": False},
        {"family": "cross_attention", "layers": 1, "heads": 1, "pooling": "coordinator", "feature_mode": "final_bundle", "aux": False},
        {"family": "cross_attention", "layers": 1, "heads": 1, "pooling": "coordinator", "feature_mode": "late_final", "aux": True},
    ]


def _default_stage5_prompt_variants() -> List[Dict[str, object]]:
    return [
        {"family": "self_attention", "layers": 1, "heads": 1, "pooling": "cls", "feature_mode": "late_final", "aux": False},
        {"family": "cross_attention", "layers": 1, "heads": 1, "pooling": "coordinator", "feature_mode": "late_final", "aux": False},
    ]


def _stage5_variant_specs(
    variants: List[object],
    slot_labels: List[str],
    agent_dropout: float,
    aux_loss_weight: float,
) -> List[RoleAttentionConfig]:
    specs: List[RoleAttentionConfig] = []
    raw_variants = variants or _default_stage5_variants()
    for index, raw in enumerate(raw_variants):
        variant = dict(raw) if isinstance(raw, dict) else {}
        family = str(variant.get("family", "self_attention"))
        if family in {"self", "role_self_attention"}:
            family = "self_attention"
        if family in {"cross", "coordinator_cross_attention"}:
            family = "cross_attention"
        layers = int(variant.get("layers", variant.get("num_attention_layers", 1)))
        heads = int(variant.get("heads", variant.get("num_heads", 1)))
        pooling = str(variant.get("pooling", "cls" if family == "self_attention" else "coordinator"))
        feature_mode = str(variant.get("feature_mode", "late_final"))
        aux = bool(variant.get("aux", variant.get("use_aux_private_bit_loss", False)))
        slot_indices = _stage5_slot_indices(slot_labels, feature_mode)
        selected_labels = tuple(slot_labels[slot_id] for slot_id in slot_indices)
        name = str(
            variant.get(
                "name",
                f"{family}_l{layers}_h{heads}_{pooling}_{feature_mode}_{'aux' if aux else 'noaux'}",
            )
        )
        specs.append(
            RoleAttentionConfig(
                slot_indices=slot_indices,
                slot_labels=selected_labels,
                family=family,
                variant_name=name,
                num_attention_layers=max(1, layers),
                num_heads=max(1, heads),
                pooling=pooling,
                feature_mode=feature_mode,
                use_aux_private_bit_loss=aux,
                aux_loss_weight=aux_loss_weight,
                agent_dropout=agent_dropout,
            )
        )
    return specs


def _stage5_slot_indices(slot_labels: List[str], feature_mode: str) -> tuple[int, ...]:
    if feature_mode in {"final_bundle", "late_final_bundle"}:
        final_slots = tuple(index for index, label in enumerate(slot_labels) if label.startswith("final:"))
        if final_slots:
            return final_slots
    if feature_mode in {"all_slots", "layer_token_bundle"}:
        return tuple(range(len(slot_labels)))
    return (_late_final_slot_index(slot_labels),)


def _make_stage5_method(
    role_config: RoleAttentionConfig,
    num_classes: int,
    training_config: MLPTrainingConfig,
    seed: int,
    device: str,
) -> RoleAwareAttentionCoordinator:
    if role_config.family == "cross_attention":
        return CoordinatorTokenCrossAttentionCoordinator(
            num_classes=num_classes,
            training=training_config,
            seed=seed,
            device=device,
            config=role_config,
        )
    return RoleAwareSelfAttentionCoordinator(
        num_classes=num_classes,
        training=training_config,
        seed=seed,
        device=device,
        config=role_config,
    )


def _run_prompt_robustness_diagnostics(
    benchmark: str,
    seed: int,
    splits: Dict[str, list],
    transformer_config: TransformerStrictConfig,
    prompt_variants: List[object],
    n_evidence_bits: int,
    num_classes: int,
    num_tasks: int,
    training_config: MLPTrainingConfig,
    pca_components: int,
    device: str,
    diagnostics: List[Dict[str, object]],
    audit: List[Dict[str, object]],
    event_index: int,
    include_stage4: bool = False,
    stage4_agent_dropout: float = 0.25,
    include_stage5: bool = False,
    stage5_variants: List[object] | None = None,
    stage5_agent_dropout: float = 0.25,
    stage5_aux_loss_weight: float = 0.2,
) -> Tuple[int, List[Callable[[int], int]]]:
    fitted = []
    for variant_index, raw_variant in enumerate(prompt_variants):
        variant = dict(raw_variant)
        variant_name = str(variant.get("name", f"variant_{variant_index}"))
        config = _variant_transformer_config(transformer_config, variant)
        swarm = TransformerStrictPartialEvidenceSwarm(
            config=config,
            n_evidence_bits=n_evidence_bits,
            num_classes=num_classes,
            seed=seed,
        )
        batches = swarm.run(splits)
        for split, batch in batches.items():
            diagnostics.append(
                {
                    **_visible_output_audit_result(benchmark, seed, split, batch),
                    "probe": "prompt_robustness_output_leakage_audit",
                    "variant": variant_name,
                    "prompt_template": config.prompt_template,
                    "answer_format": config.answer_format,
                    "evidence_tokens": list(config.evidence_tokens),
                }
            )
        methods = [
            HiddenStateOnlyLabelProbe(
                num_classes=num_classes,
                components=pca_components,
                training=training_config,
                seed=seed + 500_000 + variant_index,
                device=device,
            ),
            TextOnlyMLPCoordinator(
                num_classes=num_classes,
                num_tasks=num_tasks,
                training=training_config,
                seed=seed + 501_000 + variant_index,
                device=device,
            ),
            OutputOnlyOracleProbe(
                num_classes=num_classes,
                num_tasks=num_tasks,
                target_param_count=0,
                training=training_config,
                seed=seed + 502_000 + variant_index,
                device=device,
            ),
        ]
        if include_stage4:
            set_config = HiddenSetConfig(
                slot_index=_late_final_slot_index(swarm.slot_labels),
                agent_dropout=stage4_agent_dropout,
            )
            methods.extend(
                [
                    DeepSetsHiddenCoordinator(
                        num_classes=num_classes,
                        training=training_config,
                        seed=seed + 503_000 + variant_index,
                        device=device,
                        config=set_config,
                    ),
                    AttentionHiddenCoordinator(
                        num_classes=num_classes,
                        training=training_config,
                        seed=seed + 504_000 + variant_index,
                        device=device,
                        config=set_config,
                    ),
                ]
            )
        if include_stage5:
            for role_config in _stage5_variant_specs(
                variants=list(stage5_variants or _default_stage5_prompt_variants()),
                slot_labels=swarm.slot_labels,
                agent_dropout=stage5_agent_dropout,
                aux_loss_weight=stage5_aux_loss_weight,
            ):
                methods.append(
                    _make_stage5_method(
                        role_config=role_config,
                        num_classes=num_classes,
                        training_config=training_config,
                        seed=seed + 505_000 + variant_index * 101 + len(methods),
                        device=device,
                    )
                )
        for method in methods:
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    event_index,
                    "prompt_robustness_fit",
                    method=method.name,
                    condition=variant_name,
                    split_inputs=("train", "dev"),
                )
            )
            event_index += 1
            method.fit(batches["train"], batches["dev"])
            preds = method.predict(_without_labels(batches["dev"]))
            diagnostics.append(
                _prompt_or_model_result(
                    benchmark=benchmark,
                    seed=seed,
                    split="dev",
                    probe="prompt_robustness",
                    condition=variant_name,
                    method=method.name,
                    predictions=preds,
                    batch=batches["dev"],
                    param_count=int(getattr(method, "param_count", 0)),
                    extra={
                        "prompt_template": config.prompt_template,
                        "answer_format": config.answer_format,
                        "evidence_tokens": list(config.evidence_tokens),
                    },
                )
            )
            fitted.append((variant_name, config, batches, method))

    def evaluate_test(current_event_index: int) -> int:
        for variant_name, config, batches, method in fitted:
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "prompt_robustness_test_predict",
                    method=method.name,
                    condition=variant_name,
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(batches["test"]))
            diagnostics.append(
                _prompt_or_model_result(
                    benchmark=benchmark,
                    seed=seed,
                    split="test",
                    probe="prompt_robustness",
                    condition=variant_name,
                    method=method.name,
                    predictions=preds,
                    batch=batches["test"],
                    param_count=int(getattr(method, "param_count", 0)),
                    extra={
                        "prompt_template": config.prompt_template,
                        "answer_format": config.answer_format,
                        "evidence_tokens": list(config.evidence_tokens),
                    },
                )
            )
        return current_event_index

    return event_index, [evaluate_test]


def _run_model_robustness_diagnostics(
    benchmark: str,
    seed: int,
    splits: Dict[str, list],
    base_transformer_config: TransformerStrictConfig,
    model_variants: List[object],
    n_evidence_bits: int,
    num_classes: int,
    num_tasks: int,
    training_config: MLPTrainingConfig,
    pca_components: int,
    device: str,
    diagnostics: List[Dict[str, object]],
    audit: List[Dict[str, object]],
    event_index: int,
) -> Tuple[int, List[Callable[[int], int]]]:
    fitted = []
    for variant_index, raw_variant in enumerate(model_variants):
        variant = dict(raw_variant)
        model_name = str(variant.get("model_name_or_path", ""))
        if not model_name:
            continue
        config = _variant_transformer_config(base_transformer_config, variant)
        try:
            swarm = TransformerStrictPartialEvidenceSwarm(
                config=config,
                n_evidence_bits=n_evidence_bits,
                num_classes=num_classes,
                seed=seed,
            )
            batches = swarm.run(splits)
        except Exception as exc:
            diagnostics.append(
                {
                    "benchmark": benchmark,
                    "seed": seed,
                    "split": "test",
                    "probe": "model_robustness",
                    "condition": model_name,
                    "method": "hidden_state_only_probe",
                    "status": "not_run",
                    "reason": str(exc),
                    "accuracy": None,
                    "n_examples": 0,
                    "param_count": 0,
                }
            )
            continue
        methods = [
            HiddenStateOnlyLabelProbe(
                num_classes=num_classes,
                components=pca_components,
                training=training_config,
                seed=seed + 510_000 + variant_index,
                device=device,
            ),
            OutputOnlyOracleProbe(
                num_classes=num_classes,
                num_tasks=num_tasks,
                target_param_count=0,
                training=training_config,
                seed=seed + 511_000 + variant_index,
                device=device,
            ),
        ]
        for method in methods:
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    event_index,
                    "model_robustness_fit",
                    method=method.name,
                    condition=model_name,
                    split_inputs=("train", "dev"),
                )
            )
            event_index += 1
            method.fit(batches["train"], batches["dev"])
            preds = method.predict(_without_labels(batches["dev"]))
            diagnostics.append(
                _prompt_or_model_result(
                    benchmark=benchmark,
                    seed=seed,
                    split="dev",
                    probe="model_robustness",
                    condition=model_name,
                    method=method.name,
                    predictions=preds,
                    batch=batches["dev"],
                    param_count=int(getattr(method, "param_count", 0)),
                    extra={
                        "model_name_or_path": model_name,
                        "layers": list(config.layers),
                        "token_positions": list(config.token_positions),
                    },
                )
            )
            fitted.append((model_name, config, batches, method))

    def evaluate_test(current_event_index: int) -> int:
        for model_name, config, batches, method in fitted:
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "model_robustness_test_predict",
                    method=method.name,
                    condition=model_name,
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(batches["test"]))
            diagnostics.append(
                _prompt_or_model_result(
                    benchmark=benchmark,
                    seed=seed,
                    split="test",
                    probe="model_robustness",
                    condition=model_name,
                    method=method.name,
                    predictions=preds,
                    batch=batches["test"],
                    param_count=int(getattr(method, "param_count", 0)),
                    extra={
                        "model_name_or_path": model_name,
                        "layers": list(config.layers),
                        "token_positions": list(config.token_positions),
                    },
                )
            )
        return current_event_index

    return event_index, [evaluate_test]


def _run_explicit_evidence_sharing_baseline(
    benchmark: str,
    seed: int,
    sharing_batches: Dict[str, AttemptBatch],
    num_classes: int,
    num_tasks: int,
    training_config: MLPTrainingConfig,
    pca_components: int,
    device: str,
    diagnostics: List[Dict[str, object]],
    audit: List[Dict[str, object]],
    event_index: int,
) -> Tuple[int, List[Callable[[int], int]]]:
    del pca_components
    methods = [
        EvidenceSharingRuleCoordinator(),
        TextOnlyMLPCoordinator(
            num_classes=num_classes,
            num_tasks=num_tasks,
            training=training_config,
            seed=seed + 520_000,
            device=device,
        ),
        OutputOnlyOracleProbe(
            num_classes=num_classes,
            num_tasks=num_tasks,
            target_param_count=0,
            training=training_config,
            seed=seed + 521_000,
            device=device,
        ),
    ]
    fitted = []
    for method in methods:
        audit.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "explicit_evidence_sharing_fit",
                method=method.name,
                condition="visible_private_evidence",
                split_inputs=("train", "dev"),
            )
        )
        event_index += 1
        method.fit(sharing_batches["train"], sharing_batches["dev"])
        preds = method.predict(_without_labels(sharing_batches["dev"]))
        diagnostics.append(
            _prompt_or_model_result(
                benchmark=benchmark,
                seed=seed,
                split="dev",
                probe="explicit_evidence_sharing_baseline",
                condition="visible_private_evidence",
                method=method.name,
                predictions=preds,
                batch=sharing_batches["dev"],
                param_count=int(getattr(method, "param_count", 0)),
                extra={"visible_private_evidence": True},
            )
        )
        fitted.append(method)

    def evaluate_test(current_event_index: int) -> int:
        for method in fitted:
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "explicit_evidence_sharing_test_predict",
                    method=method.name,
                    condition="visible_private_evidence",
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(sharing_batches["test"]))
            diagnostics.append(
                _prompt_or_model_result(
                    benchmark=benchmark,
                    seed=seed,
                    split="test",
                    probe="explicit_evidence_sharing_baseline",
                    condition="visible_private_evidence",
                    method=method.name,
                    predictions=preds,
                    batch=sharing_batches["test"],
                    param_count=int(getattr(method, "param_count", 0)),
                    extra={"visible_private_evidence": True},
                )
            )
        return current_event_index

    return event_index, [evaluate_test]


def _run_stage3_probe_family_diagnostics(
    benchmark: str,
    seed: int,
    base_batches: Dict[str, AttemptBatch],
    num_classes: int,
    num_tasks: int,
    training_config: MLPTrainingConfig,
    pca_components: int,
    device: str,
    diagnostics: List[Dict[str, object]],
    audit: List[Dict[str, object]],
    event_index: int,
) -> Tuple[int, List[Callable[[int], int]]]:
    linear_training = MLPTrainingConfig(
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
        patience=training_config.patience,
        hidden_dims=(),
    )
    methods: List[object] = [
        ActivationPCAMLPCoordinator(
            num_classes=num_classes,
            num_tasks=num_tasks,
            components=pca_components,
            training=training_config,
            seed=seed + 530_000,
            device=device,
            name="activation_pca_mlp_family",
        ),
        ActivationPCAMLPCoordinator(
            num_classes=num_classes,
            num_tasks=num_tasks,
            components=pca_components,
            training=linear_training,
            seed=seed + 531_000,
            device=device,
            name="activation_pca_linear_logistic",
        ),
        RidgePCACoordinator(
            num_classes=num_classes,
            num_tasks=num_tasks,
            components=pca_components,
            alpha=1.0,
        ),
        FeatureMLPCoordinator(
            name="set_cross_agent_mlp",
            feature_builder=lambda batch: _set_cross_agent_features(batch, num_classes, num_tasks),
            num_classes=num_classes,
            training=training_config,
            seed=seed + 532_000,
            device=device,
        ),
    ]
    fitted = []
    for method in methods:
        audit.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "stage3_probe_family_fit",
                method=method.name,
                condition="none",
                split_inputs=("train", "dev"),
            )
        )
        event_index += 1
        method.fit(base_batches["train"], base_batches["dev"])
        preds = method.predict(_without_labels(base_batches["dev"]))
        diagnostics.append(
            _probe_family_result(
                benchmark,
                seed,
                "dev",
                method.name,
                preds,
                base_batches["dev"],
                int(getattr(method, "param_count", 0)),
                method.metadata() if hasattr(method, "metadata") else {},
            )
        )
        fitted.append(method)

    def evaluate_test(current_event_index: int) -> int:
        for method in fitted:
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "stage3_probe_family_test_predict",
                    method=method.name,
                    condition="none",
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(base_batches["test"]))
            diagnostics.append(
                _probe_family_result(
                    benchmark,
                    seed,
                    "test",
                    method.name,
                    preds,
                    base_batches["test"],
                    int(getattr(method, "param_count", 0)),
                    method.metadata() if hasattr(method, "metadata") else {},
                )
            )
        return current_event_index

    return event_index, [evaluate_test]


def _run_hidden_slice_ablations(
    benchmark: str,
    seed: int,
    base_batches: Dict[str, AttemptBatch],
    slot_labels: List[str],
    num_classes: int,
    num_tasks: int,
    training_config: MLPTrainingConfig,
    pca_components: int,
    device: str,
    diagnostics: List[Dict[str, object]],
    audit: List[Dict[str, object]],
    event_index: int,
) -> Tuple[int, List[Callable[[int], int]]]:
    ablations = []
    positions = sorted({label.split(":")[0] for label in slot_labels})
    layers = sorted({label.split(":")[1] for label in slot_labels})
    for position in positions:
        ablations.append(("position", position, [i for i, label in enumerate(slot_labels) if label.startswith(f"{position}:")]))
    for layer in layers:
        ablations.append(("layer", layer, [i for i, label in enumerate(slot_labels) if label.endswith(f":{layer}")]))

    fitted = []
    for axis, value, indices in ablations:
        train = _mask_slots(base_batches["train"], indices)
        dev = _mask_slots(base_batches["dev"], indices)
        method = ActivationPCAMLPCoordinator(
            num_classes=num_classes,
            num_tasks=num_tasks,
            components=pca_components,
            training=training_config,
            seed=seed + 220_000 + len(value),
            device=device,
        )
        audit.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "hidden_slice_ablation_fit",
                method="activation_pca_mlp",
                condition=f"mask_{axis}_{value}",
                split_inputs=("train", "dev"),
            )
        )
        event_index += 1
        method.fit(train, dev)
        preds = method.predict(_without_labels(dev))
        diagnostics.append(
            _slice_ablation_result(
                benchmark, seed, "dev", axis, value, preds, dev, int(getattr(method, "param_count", 0))
            )
        )
        fitted.append((axis, value, indices, method))

    def evaluate_test(current_event_index: int) -> int:
        for axis, value, indices, method in fitted:
            test = _mask_slots(base_batches["test"], indices)
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "hidden_slice_ablation_test_predict",
                    method="activation_pca_mlp",
                    condition=f"mask_{axis}_{value}",
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(test))
            diagnostics.append(
                _slice_ablation_result(
                    benchmark, seed, "test", axis, value, preds, test, int(getattr(method, "param_count", 0))
                )
            )
        return current_event_index

    return event_index, [evaluate_test]


def _run_individual_hidden_label_probe(
    benchmark: str,
    seed: int,
    base_batches: Dict[str, AttemptBatch],
    num_classes: int,
    training_config: MLPTrainingConfig,
    device: str,
    diagnostics: List[Dict[str, object]],
    audit: List[Dict[str, object]],
    event_index: int,
) -> Tuple[int, List[Callable[[int], int]]]:
    fitted = []
    for agent_id in range(base_batches["train"].n_agents):
        method = FeatureMLPCoordinator(
            name="individual_hidden_label_probe",
            feature_builder=lambda batch, a=agent_id: batch.hidden_states[:, a, :, :].reshape(batch.n_examples, -1),
            num_classes=num_classes,
            training=training_config,
            seed=seed + 330_000 + agent_id,
            device=device,
        )
        audit.append(
            audit_event(
                benchmark,
                seed,
                event_index,
                "individual_hidden_label_probe_fit",
                method="individual_hidden_label_probe",
                condition=f"agent_{agent_id}",
                split_inputs=("train", "dev"),
            )
        )
        event_index += 1
        method.fit(base_batches["train"], base_batches["dev"])
        preds = method.predict(_without_labels(base_batches["dev"]))
        diagnostics.append(
            _individual_label_result(
                benchmark, seed, "dev", agent_id, preds, base_batches["dev"], int(getattr(method, "param_count", 0))
            )
        )
        fitted.append((agent_id, method))

    def evaluate_test(current_event_index: int) -> int:
        for agent_id, method in fitted:
            audit.append(
                audit_event(
                    benchmark,
                    seed,
                    current_event_index,
                    "individual_hidden_label_probe_test_predict",
                    method="individual_hidden_label_probe",
                    condition=f"agent_{agent_id}",
                    split_inputs=("test",),
                )
            )
            current_event_index += 1
            preds = method.predict(_without_labels(base_batches["test"]))
            diagnostics.append(
                _individual_label_result(
                    benchmark,
                    seed,
                    "test",
                    agent_id,
                    preds,
                    base_batches["test"],
                    int(getattr(method, "param_count", 0)),
                )
            )
        return current_event_index

    return event_index, [evaluate_test]


def _mask_slots(batch: AttemptBatch, slot_indices: List[int]) -> AttemptBatch:
    hidden = batch.hidden_states.copy()
    hidden[:, :, slot_indices, :] = 0.0
    return batch.copy_with_hidden(hidden)


def _slice_ablation_result(
    benchmark: str,
    seed: int,
    split: str,
    axis: str,
    value: str,
    predictions: np.ndarray,
    batch: AttemptBatch,
    param_count: int,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "hidden_state_slice_ablation",
        "axis": axis,
        "value": value,
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": param_count,
    }


def _individual_label_result(
    benchmark: str,
    seed: int,
    split: str,
    agent_id: int,
    predictions: np.ndarray,
    batch: AttemptBatch,
    param_count: int,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "individual_hidden_label_probe",
        "agent_id": agent_id,
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": param_count,
    }


def _private_bit_probe_result(
    benchmark: str,
    seed: int,
    split: str,
    agent_id: int,
    feature_source: str,
    predictions: np.ndarray,
    labels: np.ndarray,
    param_count: int,
) -> Dict[str, object]:
    labels = labels.astype(np.int64)
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "private_evidence_bit_probe",
        "agent_id": agent_id,
        "feature_source": feature_source,
        "accuracy": float(np.mean(predictions.astype(np.int64) == labels)),
        "majority_baseline": _majority_baseline(labels),
        "positive_rate": float(np.mean(labels == 1)),
        "n_examples": int(labels.shape[0]),
        "param_count": param_count,
    }


def _agent_count_result(
    benchmark: str,
    seed: int,
    split: str,
    agent_count: int,
    available_agents: int,
    predictions: np.ndarray,
    batch: AttemptBatch,
    param_count: int,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "hidden_state_agent_count_curve",
        "agent_count": agent_count,
        "available_agents": available_agents,
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": param_count,
    }


def _hidden_location_result(
    benchmark: str,
    seed: int,
    split: str,
    axis: str,
    value: str,
    slot_indices: List[int],
    predictions: np.ndarray,
    batch: AttemptBatch,
    param_count: int,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "hidden_state_location_probe",
        "axis": axis,
        "value": value,
        "slot_indices": [int(index) for index in slot_indices],
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": param_count,
    }


def _visible_output_audit_result(
    benchmark: str,
    seed: int,
    split: str,
    batch: AttemptBatch,
) -> Dict[str, object]:
    texts = [text for row in batch.visible_texts for text in row]
    bit_token_hits = sum(("BIT_ONE" in text) or ("BIT_ZERO" in text) or ("BIT_MASK" in text) for text in texts)
    evidence_word_hits = sum("evidence" in text.lower() for text in texts)
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "visible_output_leakage_audit",
        "bit_token_occurrences": int(bit_token_hits),
        "evidence_word_occurrences": int(evidence_word_hits),
        "unique_visible_texts": len(set(texts)),
        "unique_visible_rows": len({tuple(row) for row in batch.visible_texts}),
        "unique_answer_values": [int(value) for value in sorted(np.unique(batch.answers).tolist())],
        "unique_confidence_values": [float(value) for value in sorted(np.unique(batch.confidences).tolist())],
        "n_texts": len(texts),
        "n_examples": batch.n_examples,
        "param_count": 0,
    }


def _evidence_control_result(
    benchmark: str,
    seed: int,
    split: str,
    condition: str,
    predictions: np.ndarray,
    batch: AttemptBatch,
    param_count: int,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "evidence_causal_control",
        "condition": condition,
        "labels_preserved": True,
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": param_count,
    }


def _activation_stability_result(
    benchmark: str,
    seed: int,
    split: str,
    method: str,
    variant: str,
    training_seed: int,
    predictions: np.ndarray,
    batch: AttemptBatch,
    param_count: int,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "activation_pca_mlp_stability",
        "method": method,
        "variant": variant,
        "training_seed": training_seed,
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": param_count,
    }


def _stage4_accuracy_result(
    benchmark: str,
    seed: int,
    split: str,
    method: HiddenSetCoordinator,
    condition: str,
    predictions: np.ndarray,
    batch: AttemptBatch,
    set_config: HiddenSetConfig,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "stage4_hidden_coordinator",
        "method": method.name,
        "condition": condition,
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": int(getattr(method, "param_count", 0)),
        "slot_index": set_config.slot_index,
        "agent_dropout": set_config.agent_dropout,
    }


def _stage4_agent_count_result(
    benchmark: str,
    seed: int,
    split: str,
    method: HiddenSetCoordinator,
    agent_count: int,
    predictions: np.ndarray,
    batch: AttemptBatch,
    set_config: HiddenSetConfig,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "stage4_agent_count_curve",
        "method": method.name,
        "agent_count": agent_count,
        "available_agents": batch.n_agents,
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": int(getattr(method, "param_count", 0)),
        "slot_index": set_config.slot_index,
        "agent_dropout": set_config.agent_dropout,
    }


def _stage4_permutation_result(
    benchmark: str,
    seed: int,
    method: HiddenSetCoordinator,
    base_predictions: np.ndarray,
    permuted_predictions: np.ndarray,
    batch: AttemptBatch,
    set_config: HiddenSetConfig,
) -> Dict[str, object]:
    base_acc = float(np.mean(base_predictions.astype(np.int64) == batch.labels.astype(np.int64)))
    perm_acc = float(np.mean(permuted_predictions.astype(np.int64) == batch.labels.astype(np.int64)))
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": "test",
        "probe": "stage4_agent_order_permutation",
        "method": method.name,
        "accuracy": perm_acc,
        "base_accuracy": base_acc,
        "abs_accuracy_delta": abs(base_acc - perm_acc),
        "prediction_disagreement": float(
            np.mean(base_predictions.astype(np.int64) != permuted_predictions.astype(np.int64))
        ),
        "n_examples": batch.n_examples,
        "param_count": int(getattr(method, "param_count", 0)),
        "slot_index": set_config.slot_index,
    }


def _stage4_label_permutation_result(
    benchmark: str,
    seed: int,
    method: HiddenSetCoordinator,
    predictions: np.ndarray,
    batch: AttemptBatch,
    set_config: HiddenSetConfig,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": "test",
        "probe": "stage4_label_permutation_sanity",
        "method": method.name,
        "condition": "randomized_train_labels",
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": int(getattr(method, "param_count", 0)),
        "slot_index": set_config.slot_index,
    }


def _stage4_control_result(
    benchmark: str,
    seed: int,
    split: str,
    method: HiddenSetCoordinator,
    condition: str,
    predictions: np.ndarray,
    batch: AttemptBatch,
    set_config: HiddenSetConfig,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "stage4_evidence_control",
        "method": method.name,
        "condition": condition,
        "labels_preserved": True,
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": int(getattr(method, "param_count", 0)),
        "slot_index": set_config.slot_index,
        "agent_dropout": set_config.agent_dropout,
    }


def _stage5_variant_fields(method: RoleAwareAttentionCoordinator) -> Dict[str, object]:
    config = method.config
    return {
        "family": config.family,
        "variant_name": config.variant_name,
        "num_attention_layers": config.num_attention_layers,
        "num_heads": config.num_heads,
        "pooling": config.pooling,
        "feature_mode": config.feature_mode,
        "slot_indices": list(config.slot_indices),
        "slot_labels": list(config.slot_labels),
        "agent_dropout": config.agent_dropout,
        "use_aux_private_bit_loss": config.use_aux_private_bit_loss,
        "aux_loss_weight": config.aux_loss_weight,
        "aux_targets_seen": method.aux_targets_seen,
    }


def _stage5_accuracy_result(
    benchmark: str,
    seed: int,
    split: str,
    method: RoleAwareAttentionCoordinator,
    condition: str,
    predictions: np.ndarray,
    batch: AttemptBatch,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "stage5_role_attention",
        "method": method.name,
        "condition": condition,
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": int(getattr(method, "param_count", 0)),
        **_stage5_variant_fields(method),
    }


def _stage5_agent_count_result(
    benchmark: str,
    seed: int,
    split: str,
    method: RoleAwareAttentionCoordinator,
    agent_count: int,
    predictions: np.ndarray,
    batch: AttemptBatch,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "stage5_agent_count_curve",
        "method": method.name,
        "agent_count": agent_count,
        "available_agents": batch.n_agents,
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": int(getattr(method, "param_count", 0)),
        **_stage5_variant_fields(method),
    }


def _stage5_permutation_result(
    benchmark: str,
    seed: int,
    method: RoleAwareAttentionCoordinator,
    condition: str,
    base_predictions: np.ndarray,
    controlled_predictions: np.ndarray,
    batch: AttemptBatch,
) -> Dict[str, object]:
    base_acc = float(np.mean(base_predictions.astype(np.int64) == batch.labels.astype(np.int64)))
    controlled_acc = float(np.mean(controlled_predictions.astype(np.int64) == batch.labels.astype(np.int64)))
    probe = (
        "stage5_slot_label_shuffle"
        if condition == "slot_label_shuffle"
        else "stage5_physical_order_permutation"
    )
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": "test",
        "probe": probe,
        "method": method.name,
        "condition": condition,
        "accuracy": controlled_acc,
        "base_accuracy": base_acc,
        "abs_accuracy_delta": abs(base_acc - controlled_acc),
        "prediction_disagreement": float(
            np.mean(base_predictions.astype(np.int64) != controlled_predictions.astype(np.int64))
        ),
        "n_examples": batch.n_examples,
        "param_count": int(getattr(method, "param_count", 0)),
        **_stage5_variant_fields(method),
    }


def _stage5_label_permutation_result(
    benchmark: str,
    seed: int,
    method: RoleAwareAttentionCoordinator,
    predictions: np.ndarray,
    batch: AttemptBatch,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": "test",
        "probe": "stage5_label_permutation_sanity",
        "method": method.name,
        "condition": "randomized_train_labels",
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": int(getattr(method, "param_count", 0)),
        **_stage5_variant_fields(method),
    }


def _stage5_control_result(
    benchmark: str,
    seed: int,
    split: str,
    method: RoleAwareAttentionCoordinator,
    condition: str,
    predictions: np.ndarray,
    batch: AttemptBatch,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "stage5_evidence_control",
        "method": method.name,
        "condition": condition,
        "labels_preserved": True,
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": int(getattr(method, "param_count", 0)),
        **_stage5_variant_fields(method),
    }


def _prompt_or_model_result(
    benchmark: str,
    seed: int,
    split: str,
    probe: str,
    condition: str,
    method: str,
    predictions: np.ndarray,
    batch: AttemptBatch,
    param_count: int,
    extra: Dict[str, object],
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": probe,
        "condition": condition,
        "method": method,
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": param_count,
        **extra,
    }


def _probe_family_result(
    benchmark: str,
    seed: int,
    split: str,
    method: str,
    predictions: np.ndarray,
    batch: AttemptBatch,
    param_count: int,
    metadata: Dict[str, object],
) -> Dict[str, object]:
    return {
        "benchmark": benchmark,
        "seed": seed,
        "split": split,
        "probe": "activation_probe_family",
        "method": method,
        "accuracy": float(np.mean(predictions.astype(np.int64) == batch.labels.astype(np.int64))),
        "n_examples": batch.n_examples,
        "param_count": param_count,
        "pca_explained_variance_ratio_sum": metadata.get("pca_explained_variance_ratio_sum"),
        "pca_first_component_variance_ratio": metadata.get("pca_first_component_variance_ratio"),
    }


class RidgePCACoordinator:
    name = "activation_pca_ridge_probe"

    def __init__(
        self,
        num_classes: int,
        num_tasks: int,
        components: int,
        alpha: float,
    ) -> None:
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.components = components
        self.alpha = alpha
        self._mean: np.ndarray | None = None
        self._basis: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self.param_count = 0
        self._explained_variance_ratio: np.ndarray | None = None

    def fit(self, train_batch: AttemptBatch, dev_batch: AttemptBatch | None = None) -> None:
        del dev_batch
        activation = activation_features(train_batch)
        self._fit_pca(activation)
        x = self._features(train_batch)
        y = np.zeros((train_batch.n_examples, self.num_classes), dtype=np.float32)
        y[np.arange(train_batch.n_examples), train_batch.labels.astype(np.int64)] = 1.0
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        gram = x_aug.T @ x_aug
        gram += self.alpha * np.eye(gram.shape[0], dtype=np.float32)
        try:
            self._weights = np.linalg.solve(gram, x_aug.T @ y).astype(np.float32)
        except np.linalg.LinAlgError:
            self._weights = (np.linalg.pinv(gram) @ x_aug.T @ y).astype(np.float32)
        self.param_count = int(self._weights.size)

    def predict(self, batch: AttemptBatch) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("activation_pca_ridge_probe has not been fit")
        x = self._features(batch)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        return np.argmax(x_aug @ self._weights, axis=1).astype(np.int64)

    def metadata(self) -> Dict[str, object]:
        meta: Dict[str, object] = {
            "alpha": self.alpha,
            "pca_components": self.components,
            "param_count": self.param_count,
        }
        if self._explained_variance_ratio is not None:
            meta["pca_explained_variance_ratio_sum"] = float(self._explained_variance_ratio.sum())
            meta["pca_first_component_variance_ratio"] = float(self._explained_variance_ratio[0]) if len(self._explained_variance_ratio) else 0.0
        return meta

    def _fit_pca(self, x: np.ndarray) -> None:
        self._mean = x.mean(axis=0, keepdims=True)
        centered = x - self._mean
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        keep = min(self.components, vt.shape[0])
        self._basis = vt[:keep].astype(np.float32)
        variances = (singular_values.astype(np.float64) ** 2) / max(1, x.shape[0] - 1)
        total = float(variances.sum())
        self._explained_variance_ratio = (
            (variances[:keep] / total).astype(np.float32)
            if total > 0.0
            else np.zeros(keep, dtype=np.float32)
        )

    def _features(self, batch: AttemptBatch) -> np.ndarray:
        raw = activation_features(batch)
        if self._mean is None or self._basis is None:
            compressed = raw[:, : self.components]
        else:
            compressed = (raw - self._mean) @ self._basis.T
        return np.concatenate(
            [
                visible_features(batch, self.num_classes, self.num_tasks),
                compressed.astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)


class EvidenceSharingRuleCoordinator:
    name = "evidence_sharing_rule_oracle"
    param_count = 0

    def fit(self, train_batch: AttemptBatch, dev_batch: AttemptBatch | None = None) -> None:
        del train_batch, dev_batch

    def predict(self, batch: AttemptBatch) -> np.ndarray:
        labels = []
        for row in batch.visible_texts:
            bits = [_extract_visible_private_bit(text) for text in row[:4]]
            if any(bit is None for bit in bits) or len(bits) < 4:
                labels.append(0)
                continue
            high = int(bits[0]) ^ int(bits[1])
            low = int(bits[2]) ^ int(bits[3])
            labels.append((high << 1) | low)
        return np.asarray(labels, dtype=np.int64)

    def metadata(self) -> Dict[str, object]:
        return {
            "hidden_activations": False,
            "uses_visible_traces": True,
            "rule": "parse private_evidence_bit fields and apply strict_label xor",
            "param_count": self.param_count,
        }


def _extract_visible_private_bit(text: str) -> int | None:
    match = re.search(r"private_evidence_bit=(-?\d+)", text)
    if match:
        return int(match.group(1))
    match = re.search(r'"private_evidence_bit"\s*:\s*(-?\d+)', text)
    if match:
        return int(match.group(1))
    return None


def _single_agent_hidden_features(batch: AttemptBatch, agent_id: int) -> np.ndarray:
    return batch.hidden_states[:, agent_id, :, :].reshape(batch.hidden_states.shape[0], -1).astype(np.float32)


def _private_bit_labels(batch: AttemptBatch, agent_id: int) -> np.ndarray:
    labels = []
    for row in batch.private_agent_views:
        text = row[agent_id]
        if "BIT_ONE" in text:
            labels.append(1)
        elif "BIT_ZERO" in text:
            labels.append(0)
        else:
            raise ValueError(f"private evidence bit is not present for agent {agent_id}")
    return np.asarray(labels, dtype=np.int64)


def _select_agents(batch: AttemptBatch, agent_count: int) -> AttemptBatch:
    return AttemptBatch(
        split=batch.split,
        example_ids=batch.example_ids.copy(),
        task_ids=batch.task_ids.copy(),
        labels=batch.labels.copy(),
        answers=batch.answers[:, :agent_count].copy(),
        confidences=batch.confidences[:, :agent_count].copy(),
        hidden_states=batch.hidden_states[:, :agent_count, :, :].astype(np.float32, copy=True),
        visible_texts=[row[:agent_count] for row in batch.visible_texts],
        telemetry_channels={key: value.astype(np.float32, copy=True) for key, value in batch.telemetry_channels.items()},
        private_agent_views=[row[:agent_count] for row in batch.private_agent_views],
    )


def _select_single_agent(batch: AttemptBatch, agent_id: int) -> AttemptBatch:
    return AttemptBatch(
        split=batch.split,
        example_ids=batch.example_ids.copy(),
        task_ids=batch.task_ids.copy(),
        labels=batch.labels.copy(),
        answers=batch.answers[:, agent_id : agent_id + 1].copy(),
        confidences=batch.confidences[:, agent_id : agent_id + 1].copy(),
        hidden_states=batch.hidden_states[:, agent_id : agent_id + 1, :, :].astype(np.float32, copy=True),
        visible_texts=[[row[agent_id]] for row in batch.visible_texts],
        telemetry_channels={key: value.astype(np.float32, copy=True) for key, value in batch.telemetry_channels.items()},
        private_agent_views=[[row[agent_id]] for row in batch.private_agent_views],
    )


def _set_cross_agent_features(batch: AttemptBatch, num_classes: int, num_tasks: int) -> np.ndarray:
    pooled = batch.hidden_states.mean(axis=2).astype(np.float32)
    mean_hidden = pooled.mean(axis=1)
    std_hidden = pooled.std(axis=1)
    max_hidden = pooled.max(axis=1)
    min_hidden = pooled.min(axis=1)
    if batch.n_agents > 1:
        diffs = []
        for left in range(batch.n_agents):
            for right in range(left + 1, batch.n_agents):
                diffs.append(np.abs(pooled[:, left, :] - pooled[:, right, :]))
        pairwise = np.stack(diffs, axis=1).mean(axis=1)
    else:
        pairwise = np.zeros_like(mean_hidden)
    return np.concatenate(
        [
            visible_features(batch, num_classes, num_tasks),
            mean_hidden,
            std_hidden,
            max_hidden,
            min_hidden,
            pairwise,
        ],
        axis=1,
    ).astype(np.float32)


def _select_slots(batch: AttemptBatch, slot_indices: List[int]) -> AttemptBatch:
    return batch.copy_with_hidden(batch.hidden_states[:, :, slot_indices, :])


def _hidden_location_selections(slot_labels: List[str]) -> List[Tuple[str, str, List[int]]]:
    positions = list(dict.fromkeys(label.split(":")[0] for label in slot_labels))
    layers = list(dict.fromkeys(label.split(":")[1] for label in slot_labels))
    selections: List[Tuple[str, str, List[int]]] = []
    for position in positions:
        selections.append(
            (
                "token_position",
                position,
                [index for index, label in enumerate(slot_labels) if label.startswith(f"{position}:")],
            )
        )
    for index, layer in enumerate(layers):
        selections.append(
            (
                "layer_stage",
                f"{_layer_stage_name(index, len(layers))}:{layer}",
                [slot_id for slot_id, label in enumerate(slot_labels) if label.endswith(f":{layer}")],
            )
        )
    return selections


def _late_final_slot_index(slot_labels: List[str]) -> int:
    candidates = []
    for index, label in enumerate(slot_labels):
        if not label.startswith("final:"):
            continue
        layer_text = label.split("layer_")[-1]
        try:
            layer_id = int(layer_text)
        except ValueError:
            layer_id = index
        candidates.append((layer_id, index))
    if not candidates:
        return len(slot_labels) - 1
    return max(candidates)[1]


def _agent_count_mask(batch: AttemptBatch, agent_count: int) -> np.ndarray:
    mask = np.zeros((batch.n_examples, batch.n_agents), dtype=bool)
    mask[:, : max(1, min(agent_count, batch.n_agents))] = True
    return mask


def _layer_stage_name(index: int, n_layers: int) -> str:
    if n_layers <= 1:
        return "single"
    if index == 0:
        return "early"
    if index == n_layers - 1:
        return "late"
    return "mid"


def _majority_baseline(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    values, counts = np.unique(labels.astype(np.int64), return_counts=True)
    del values
    return float(counts.max() / labels.size)


def _seed_enabled(seed: int, seed_list: object) -> bool:
    if seed_list is None:
        return True
    return seed in {int(value) for value in list(seed_list)}


def _variant_transformer_config(
    base_config: TransformerStrictConfig,
    overrides: Dict[str, object],
) -> TransformerStrictConfig:
    data = asdict(base_config)
    for key, value in overrides.items():
        if key == "name":
            continue
        if key in data:
            data[key] = value
    if "layers" in data:
        data["layers"] = tuple(int(value) for value in data["layers"])
    if "token_positions" in data:
        data["token_positions"] = tuple(str(value) for value in data["token_positions"])
    if "evidence_tokens" in data:
        data["evidence_tokens"] = tuple(str(value) for value in data["evidence_tokens"])
    return TransformerStrictConfig(**data)


def _write_access_artifacts(
    config: Dict[str, object],
    all_batches: Dict[str, Dict[str, AttemptBatch]],
    slot_labels: List[str],
) -> None:
    private_path = Path(str(config["agent_view_log_path"]))
    visible_path = Path(str(config["coordinator_visible_log_path"]))
    hidden_path = Path(str(config["hidden_state_npz_path"]))
    private_path.parent.mkdir(parents=True, exist_ok=True)
    visible_path.parent.mkdir(parents=True, exist_ok=True)
    hidden_path.parent.mkdir(parents=True, exist_ok=True)
    with private_path.open("w", encoding="utf-8") as private_file, visible_path.open("w", encoding="utf-8") as visible_file:
        arrays: Dict[str, np.ndarray] = {}
        for seed, batches in all_batches.items():
            for split, batch in batches.items():
                arrays[f"seed_{seed}_{split}_hidden_states"] = batch.hidden_states.astype(np.float32)
                arrays[f"seed_{seed}_{split}_answers"] = batch.answers.astype(np.int64)
                arrays[f"seed_{seed}_{split}_confidences"] = batch.confidences.astype(np.float32)
                for example_index, example_id in enumerate(batch.example_ids.tolist()):
                    visible_file.write(
                        json.dumps(
                            {
                                "seed": int(seed),
                                "split": split,
                                "example_id": str(example_id),
                                "answers": batch.answers[example_index].astype(int).tolist(),
                                "confidences": batch.confidences[example_index].astype(float).tolist(),
                                "visible_texts": batch.visible_texts[example_index],
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    for agent_id, prompt in enumerate(batch.private_agent_views[example_index]):
                        private_file.write(
                            json.dumps(
                                {
                                    "seed": int(seed),
                                    "split": split,
                                    "example_id": str(example_id),
                                    "agent_id": agent_id,
                                    "private_prompt": prompt,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
        arrays["slot_labels"] = np.asarray(slot_labels, dtype=object)
        np.savez_compressed(hidden_path, **arrays)


def _merge_with_existing(stage_results: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    merge_path = config.get("merge_with_path")
    if not merge_path:
        return stage_results
    path = Path(str(merge_path))
    if not path.exists():
        return stage_results
    merged = json.loads(path.read_text(encoding="utf-8"))
    benchmark = "transformer_strict"
    merged["metrics"] = [
        row for row in merged.get("metrics", []) if not (isinstance(row, dict) and row.get("benchmark") == benchmark)
    ]
    merged["metrics"].extend(stage_results["metrics"])
    merged["diagnostics"] = [
        row
        for row in merged.get("diagnostics", [])
        if not (isinstance(row, dict) and row.get("benchmark") == benchmark)
    ]
    merged["diagnostics"].extend(stage_results["diagnostics"])
    merged["audit"] = [
        row
        for row in merged.get("audit", [])
        if not (
            isinstance(row, dict)
            and (row.get("benchmark") == benchmark or row.get("type") == "test_access_summary")
        )
    ]
    merged["audit"].extend(
        row
        for row in stage_results["audit"]
        if not (isinstance(row, dict) and row.get("type") == "test_access_summary")
    )
    merged["audit"].append(summarize_test_access(merged["audit"]))
    merged.setdefault("validation", {})
    merged["validation"]["stage1_transformer"] = stage_results["validation"]["transformer_hidden_state"]
    merged["validation"]["stage1_probe_access"] = stage_results["validation"]["probe_access"]
    merged["validation"]["probe_access"] = stage_results["validation"]["probe_access"]
    merged.setdefault("metadata", {})
    merged["metadata"]["stage1_transformer_strict"] = stage_results["metadata"]
    path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    return merged


def _transformer_probe_access_matrix() -> List[Dict[str, object]]:
    rows = _probe_access_matrix()
    rows.append(
        {
            "artifact": "transformer_private_agent_view_log",
            "visible_agent_answers": False,
            "visible_confidence": False,
            "visible_traces": False,
            "hidden_activations": False,
            "private_prompt_evidence": True,
            "coordinator_input": False,
            "test_labels_or_eval_feedback": False,
        }
    )
    rows.append(
        {
            "artifact": "transformer_coordinator_visible_input_log",
            "visible_agent_answers": True,
            "visible_confidence": True,
            "visible_traces": True,
            "hidden_activations": False,
            "private_prompt_evidence": False,
            "coordinator_input": True,
            "train_labels_for_fit": False,
            "dev_labels_for_early_stopping": False,
            "test_labels_or_eval_feedback": False,
        }
    )
    rows.extend(
        [
            {
                "artifact": "deepsets_hidden_coordinator",
                "visible_agent_answers": False,
                "visible_confidence": False,
                "visible_traces": False,
                "hidden_activations": True,
                "private_prompt_evidence": False,
                "coordinator_input": True,
                "train_labels_for_fit": True,
                "dev_labels_for_early_stopping": True,
                "test_labels_or_eval_feedback": False,
            },
            {
                "artifact": "attention_hidden_coordinator",
                "visible_agent_answers": False,
                "visible_confidence": False,
                "visible_traces": False,
                "hidden_activations": True,
                "private_prompt_evidence": False,
                "coordinator_input": True,
                "train_labels_for_fit": True,
                "dev_labels_for_early_stopping": True,
                "test_labels_or_eval_feedback": False,
            },
            {
                "artifact": "role_self_attention",
                "visible_agent_answers": False,
                "visible_confidence": False,
                "visible_traces": False,
                "hidden_activations": True,
                "private_prompt_evidence": "train_aux_targets_only_when_aux_loss_enabled",
                "coordinator_input": True,
                "train_labels_for_fit": True,
                "dev_labels_for_early_stopping": True,
                "test_labels_or_eval_feedback": False,
            },
            {
                "artifact": "coordinator_cross_attention",
                "visible_agent_answers": False,
                "visible_confidence": False,
                "visible_traces": False,
                "hidden_activations": True,
                "private_prompt_evidence": "train_aux_targets_only_when_aux_loss_enabled",
                "coordinator_input": True,
                "train_labels_for_fit": True,
                "dev_labels_for_early_stopping": True,
                "test_labels_or_eval_feedback": False,
            },
        ]
    )
    return rows


def _make_dataclass(cls, data: object):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in dict(data).items() if key in allowed})


if __name__ == "__main__":
    main()
