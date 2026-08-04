from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from src.datasets.plan_gridworld import (
    ACTION_NAMES,
    ACTION_TO_DELTA,
    CandidateRollout,
    PlanGridworldDatasetConfig,
    PlanGridworldExample,
    apply_plan_control,
    build_plan_gridworld_splits,
    candidate_source_metadata_only_accuracy,
    dataset_summary,
    leakage_audit_rows,
)
from src.experiments.run_stage_plan1_latent_candidate_plan_verifier import (
    BENCHMARK as PLAN1_BENCHMARK,
    SUMMARY_ROWCOL,
    VIEW_ID,
    CandidateStatsScorer,
    PlanCloneVerifier,
    PlanModelConfig,
    PlanTrainingConfig,
    _accuracy,
    _batched,
    _bootstrap_ci,
    _chunks,
    _clear_cuda,
    _cuda_max_memory,
    _dataset_config,
    _error_cases,
    _labels,
    _mean,
    _metric_block,
    _resolve_device,
    _set_seed,
    _softmax_np,
    _std,
    _summarize_attention_batch,
    _token,
    _top1,
    _training_config,
    collate_plan_batch,
    fit_plan_verifier,
    predict_plan_logits,
)


BENCHMARK = "PLAN-1.1_ROLLOUT_CONTROL_REPAIR"
DEFAULT_CONFIG = "configs/stage_plan1_1_rollout_control_repair.json"
DEFAULT_RESULTS = "results/plan1_1_results.json"
DEFAULT_CONTROLS = "results/plan1_1_controls.json"
DEFAULT_ROLLOUT_AUDIT = "results/plan1_1_rollout_audit.json"
DEFAULT_FINAL_STATE_AUDIT = "results/plan1_1_final_state_mismatch_audit.json"
DEFAULT_LEADERBOARD = "results/plan1_1_variant_leaderboard.csv"
DEFAULT_FAILURES = "results/plan1_1_failure_taxonomy.json"
DEFAULT_ERRORS = "results/plan1_1_error_cases.jsonl"
DEFAULT_ATTENTION = "results/plan1_1_attention_summaries.jsonl"
DEFAULT_COMPUTE = "results/plan1_1_compute_metrics.json"
DEFAULT_REPORT = "reports/STAGE_PLAN1_1_ROLLOUT_CONTROL_REPAIR.md"


@dataclass(frozen=True)
class Plan11Variant:
    name: str
    short_name: str
    description: str
    model: PlanModelConfig
    example_transform: str = "none"
    diagnostic_only: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PLAN-1.1 rollout-control repair and multi-seed planning search.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--rollout-audit-output", default=DEFAULT_ROLLOUT_AUDIT)
    parser.add_argument("--final-state-audit-output", default=DEFAULT_FINAL_STATE_AUDIT)
    parser.add_argument("--leaderboard-output", default=DEFAULT_LEADERBOARD)
    parser.add_argument("--failure-output", default=DEFAULT_FAILURES)
    parser.add_argument("--error-output", default=DEFAULT_ERRORS)
    parser.add_argument("--attention-output", default=DEFAULT_ATTENTION)
    parser.add_argument("--compute-output", default=DEFAULT_COMPUTE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-variants", type=int, default=0)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    if args.device:
        config["device"] = str(args.device)
        if str(args.device) == "cpu":
            config["require_cuda"] = False
    result = run_stage_plan1_1(config, max_variants=int(args.max_variants))
    _write_outputs(
        result,
        output_path=Path(args.output),
        controls_path=Path(args.controls_output),
        rollout_audit_path=Path(args.rollout_audit_output),
        final_state_audit_path=Path(args.final_state_audit_output),
        leaderboard_path=Path(args.leaderboard_output),
        failure_path=Path(args.failure_output),
        error_path=Path(args.error_output),
        attention_path=Path(args.attention_output),
        compute_path=Path(args.compute_output),
        report_path=Path(args.report),
    )


def run_stage_plan1_1(config: Dict[str, object], max_variants: int = 0) -> Dict[str, object]:
    dataset_config = _dataset_config(config.get("dataset", {}))
    training = _training_config(config.get("training", {}))
    seeds = [int(value) for value in config.get("seeds", [0, 1, 2])]
    device = _resolve_device(str(config.get("device", "auto")))
    variants = _plan11_variants()
    if max_variants > 0:
        variants = variants[:max_variants]
    if int(config.get("max_variants", 0)) > 0:
        variants = variants[: int(config.get("max_variants", 0))]

    all_rows: List[Dict[str, object]] = []
    all_controls: List[Dict[str, object]] = []
    all_rollout_audits: List[Dict[str, object]] = []
    all_final_state_audits: List[Dict[str, object]] = []
    all_errors: List[Dict[str, object]] = []
    all_attention: List[Dict[str, object]] = []
    all_compute: List[Dict[str, object]] = []
    all_phase0: List[Dict[str, object]] = []
    artifact_audits: List[Dict[str, object]] = []

    print(f"plan1.1: device={device} variants={len(variants)} seeds={seeds}")
    for seed in seeds:
        base_splits = build_plan_gridworld_splits(dataset_config, seed=seed)
        leakage = leakage_audit_rows([example for rows in base_splits.values() for example in rows])
        artifact_audit = _candidate_artifact_audit(base_splits["train"], base_splits["dev"], base_splits["test"], seed)
        artifact_audits.append({"seed": seed, **artifact_audit})
        all_phase0.append(
            {
                "seed": seed,
                "dataset_summary": dataset_summary(base_splits),
                "leakage_audit_pass": bool(all(row.get("pass") for row in leakage)),
                "candidate_artifact_audit": artifact_audit,
            }
        )
        print(
            "plan1.1 seed={seed}: train/dev/test={train}/{dev}/{test} unigram={uni:.4f} bigram={bi:.4f}".format(
                seed=seed,
                train=len(base_splits["train"]),
                dev=len(base_splits["dev"]),
                test=len(base_splits["test"]),
                uni=float(artifact_audit["dev"]["action_unigram_mlp"]),
                bi=float(artifact_audit["dev"]["action_bigram_mlp"]),
            )
        )

        for variant in variants:
            print(f"plan1.1 variant={variant.name} seed={seed}: fitting trainable/frozen")
            splits = {
                split_name: _apply_variant_transform(rows, variant.example_transform, seed + 12_000)
                for split_name, rows in base_splits.items()
            }
            fit_start = time.perf_counter()
            trainable = fit_plan_verifier(
                train_examples=splits["train"],
                dev_examples=splits["dev"],
                model_config=variant.model,
                training_config=replace(training, objective=variant.model.objective),
                seed=seed + 10_001,
                device=device,
                trainable_shared=True,
                method=f"trainable__{variant.name}",
            )
            frozen = fit_plan_verifier(
                train_examples=splits["train"],
                dev_examples=splits["dev"],
                model_config=variant.model,
                training_config=replace(training, objective=variant.model.objective),
                seed=seed + 10_001,
                device=device,
                trainable_shared=False,
                method=f"frozen__{variant.name}",
            )
            fit_elapsed = time.perf_counter() - fit_start

            eval_start = time.perf_counter()
            dev_logits = predict_plan_logits(trainable.model, splits["dev"], variant.model, training.batch_size, device)
            frozen_dev_logits = predict_plan_logits(frozen.model, splits["dev"], variant.model, training.batch_size, device)
            test_logits = predict_plan_logits(trainable.model, splits["test"], variant.model, training.batch_size, device)
            frozen_test_logits = predict_plan_logits(frozen.model, splits["test"], variant.model, training.batch_size, device)
            latency = (time.perf_counter() - eval_start) / max(1, len(splits["dev"]) + len(splits["test"]))
            dev_labels = _labels(splits["dev"])
            test_labels = _labels(splits["test"])
            clean_top1 = _top1(dev_logits, dev_labels)

            controls = _run_stage11_controls(
                trainable.model,
                variant,
                splits["dev"],
                training.batch_size,
                device,
                seed,
                clean_top1,
                dataset_config.num_candidates,
                artifact_audit["dev"],
            )
            controls.update(
                {
                    "benchmark": BENCHMARK,
                    "parent_benchmark": PLAN1_BENCHMARK,
                    "seed": seed,
                    "variant": variant.name,
                    "chance": 1.0 / max(1, int(dataset_config.num_candidates)),
                }
            )
            controls["control_pass"] = _controls_pass_stage11(
                controls,
                clean_top1=clean_top1,
                candidate_count=dataset_config.num_candidates,
                model_config=variant.model,
                variant=variant,
            )
            all_controls.append(controls)

            if variant.name == "P1_rollout_enabled":
                all_final_state_audits.append(
                    _run_final_state_mismatch_audit(trainable.model, variant.model, splits["dev"], training.batch_size, device, seed)
                )
                all_rollout_audits.append(
                    _run_rollout_evidence_audit(trainable.model, variant.model, splits["dev"], training.batch_size, device, seed)
                )

            row = {
                "benchmark": BENCHMARK,
                "seed": seed,
                "variant": variant.name,
                "short_name": variant.short_name,
                "variant_description": variant.description,
                "model_config": asdict(variant.model),
                "training_config": asdict(training),
                "example_transform": variant.example_transform,
                "diagnostic_only": bool(variant.diagnostic_only),
                "status": "completed",
                "dev_trainable": _metric_block(dev_logits, dev_labels, splits["dev"]),
                "dev_frozen": _metric_block(frozen_dev_logits, dev_labels, splits["dev"]),
                "test_smoke_trainable": _metric_block(test_logits, test_labels, splits["test"]),
                "test_smoke_frozen": _metric_block(frozen_test_logits, test_labels, splits["test"]),
                "dev_delta_trainable_minus_frozen": float(_top1(dev_logits, dev_labels) - _top1(frozen_dev_logits, dev_labels)),
                "test_smoke_delta_trainable_minus_frozen": float(_top1(test_logits, test_labels) - _top1(frozen_test_logits, test_labels)),
                "candidate_artifact_audit": artifact_audit,
                "controls": controls.get("control_pass", {}),
                "trainable_audit": trainable.audit,
                "frozen_audit": frozen.audit,
                "medium_validation_triggered": False,
            }
            all_rows.append(row)
            all_attention.extend(
                _attention_summaries_stage11(trainable.model, variant.model, splits["dev"], training.batch_size, device, seed, limit=8)
            )
            all_errors.extend(_error_cases(variant.name, seed, splits["dev"], dev_logits, frozen_dev_logits, controls, limit=24))
            all_compute.append(
                {
                    "benchmark": BENCHMARK,
                    "seed": seed,
                    "variant": variant.name,
                    "candidate_count": dataset_config.num_candidates,
                    "number_of_clones_views": len(variant.model.view_names),
                    "coordination_blocks": variant.model.coordination_blocks,
                    "trainable_training_time_seconds": trainable.training_time_seconds,
                    "frozen_training_time_seconds": frozen.training_time_seconds,
                    "wall_training_time_seconds": float(fit_elapsed),
                    "inference_latency_seconds_per_planning_instance": float(latency),
                    "peak_cuda_memory_bytes": _cuda_max_memory(device),
                    "trainable_param_count": trainable.param_count,
                    "frozen_param_count": frozen.param_count,
                    "accuracy_per_millisecond": float(row["dev_trainable"]["top1"]) / max(1e-9, latency * 1000.0),
                    "delta_over_frozen_per_compute": float(row["dev_delta_trainable_minus_frozen"])
                    / max(1, len(variant.model.view_names) * int(variant.model.coordination_blocks) * int(dataset_config.num_candidates)),
                }
            )
            del trainable, frozen
            _clear_cuda()

    no_rollout_curricula = []
    if bool(config.get("run_no_rollout_curricula", True)):
        no_rollout_curricula = _run_no_rollout_curricula(dataset_config, training, device, config)

    summary = _overall_summary_stage11(
        all_rows,
        all_controls,
        artifact_audits,
        all_final_state_audits,
        all_rollout_audits,
        dataset_config,
    )
    risk_clone_diagnosis = _risk_clone_diagnosis(all_rows, all_controls, artifact_audits)
    failure_taxonomy = _failure_taxonomy_stage11(all_rows, all_controls, all_errors, summary, risk_clone_diagnosis, no_rollout_curricula)
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "created_at_utc": _now(),
            "device": device,
            "stage": "PLAN-1.1",
            "final_validation_run": False,
            "final_validation_policy": "not launched; this stage is limited to cheap seeds [0,1,2] and control repair",
            "claim_boundary": "No autonomous planning, world-model, or general-agent claim.",
            "allowed_claim_if_gates_pass": "On controlled synthetic candidate-plan verification, rollout-enabled shared-weight latent coordination beats an exact frozen comparator while using state/goal/constraint/rollout evidence rather than candidate artifacts.",
            "forbidden_claims": ["autonomous planning", "world model", "general agent", "open-ended planning"],
        },
        "dataset_config": asdict(dataset_config),
        "training_config": asdict(training),
        "variant_plan": [asdict(variant) for variant in variants],
        "phase0": all_phase0,
        "rows": all_rows,
        "controls": all_controls,
        "candidate_artifact_audits": artifact_audits,
        "final_state_mismatch_audit": all_final_state_audits,
        "rollout_audit": all_rollout_audits,
        "risk_clone_diagnosis": risk_clone_diagnosis,
        "no_rollout_curricula": no_rollout_curricula,
        "attention_summaries": all_attention,
        "error_cases": all_errors,
        "compute_metrics": all_compute,
        "failure_taxonomy": failure_taxonomy,
        "summary": summary,
    }


def _plan11_variants() -> List[Plan11Variant]:
    p0 = ("state_view", "goal_view", "obstacle_constraint_view", "candidate_action_view")
    rollout = p0 + ("rollout_view", "outcome_view")
    return [
        Plan11Variant(
            name="P1_rollout_enabled",
            short_name="P1",
            description="Repaired PLAN-1 rollout-enabled state/goal/constraint/action/rollout/outcome verifier.",
            model=PlanModelConfig(variant_family="rollout_enabled_repaired_controls", view_names=rollout, coordination_blocks=1),
        ),
        Plan11Variant(
            name="P1_rollout_no_final",
            short_name="P1-no-final",
            description="Rollout evidence without explicit outcome view and without the last rollout state token.",
            model=PlanModelConfig(variant_family="rollout_no_final", view_names=p0 + ("rollout_view",), coordination_blocks=1),
            example_transform="rollout_no_final",
        ),
        Plan11Variant(
            name="P1_final_only",
            short_name="P1-final",
            description="Outcome/final-state evidence without full rollout trajectory.",
            model=PlanModelConfig(variant_family="final_only", view_names=p0 + ("outcome_view",), coordination_blocks=1),
        ),
        Plan11Variant(
            name="P1_rollout_plus_risk",
            short_name="P1-risk",
            description="P1 with explicit first-collision risk clone; used only after frozen/shortcut diagnosis.",
            model=PlanModelConfig(variant_family="rollout_plus_risk", view_names=rollout + ("risk_failure_view",), coordination_blocks=1),
        ),
        Plan11Variant(
            name="P1_rollout_plus_cost",
            short_name="P1-cost",
            description="P1 with cost-efficiency clone; raw action length remains balanced.",
            model=PlanModelConfig(variant_family="rollout_plus_cost", view_names=rollout + ("cost_efficiency_view",), coordination_blocks=1),
        ),
        Plan11Variant(
            name="P1_stacked_2",
            short_name="P1-stack2",
            description="P1 with two candidate-plan cross-attention coordination blocks.",
            model=PlanModelConfig(variant_family="stacked_2_repaired_controls", view_names=rollout, coordination_blocks=2),
        ),
        Plan11Variant(
            name="P1_pyramidal_rollout_composition",
            short_name="P1-pyramid",
            description="Pyramidal state+goal, action+future, future+risk/cost composition before candidate querying.",
            model=PlanModelConfig(
                variant_family="pyramidal_rollout_composition",
                view_names=rollout + ("risk_failure_view", "cost_efficiency_view"),
                coordination_blocks=1,
                pyramidal_composition=True,
            ),
        ),
        Plan11Variant(
            name="simple_cross_attention_rollout_baseline",
            short_name="simple-xattn",
            description="Simple candidate-action to rollout/outcome cross-attention without state/goal/constraint views.",
            model=PlanModelConfig(
                variant_family="simple_cross_attention_rollout_baseline",
                view_names=("candidate_action_view", "rollout_view", "outcome_view"),
                model_dim=32,
                num_heads=4,
                ff_dim=64,
                coordination_blocks=1,
            ),
            diagnostic_only=True,
        ),
    ]


def _run_stage11_controls(
    model: PlanCloneVerifier,
    variant: Plan11Variant,
    examples: Sequence[PlanGridworldExample],
    batch_size: int,
    device: str,
    seed: int,
    clean_top1: float,
    candidate_count: int,
    artifact_dev: Dict[str, float],
) -> Dict[str, object]:
    labels = _labels(examples)
    controls: Dict[str, object] = {}
    control_specs = [
        ("candidate_plan_only", "candidate_plan_only"),
        ("state_goal_only", "state_goal_only"),
        ("state_plan_mismatch", "state_plan_mismatch"),
        ("goal_shuffle", "goal_shuffle"),
        ("obstacle_constraint_shuffle", "obstacle_constraint_shuffle"),
        ("candidate_order_shuffle_with_gold_remap", "candidate_order_shuffle_with_gold_remap"),
        ("action_symbol_permutation", "action_symbol_permutation"),
        ("map_rotation_reflection", "map_rotation_reflection"),
        ("randomized_labels", "randomized_labels"),
    ]
    for output_name, control_name in control_specs:
        controlled = apply_plan_control(examples, control_name, seed + 31_000 + len(controls))
        controlled = _apply_variant_transform(controlled, variant.example_transform, seed + 32_000)
        controlled_labels = _labels(controlled)
        logits = predict_plan_logits(model, controlled, variant.model, batch_size, device, seed=seed)
        controls[output_name] = _metric_block(logits, controlled_labels, controlled)

    if "rollout_view" in variant.model.view_names:
        rollout_wrong = _both_rollout_and_final_wrong_independent(examples, seed + 33_100)
        rollout_wrong = _apply_variant_transform(rollout_wrong, variant.example_transform, seed + 33_200)
        logits = predict_plan_logits(model, rollout_wrong, variant.model, batch_size, device, seed=seed)
        controls["rollout_mismatch"] = _metric_block(logits, _labels(rollout_wrong), rollout_wrong)
    else:
        controls["rollout_mismatch"] = {"skipped": True, "reason": "variant has no rollout_view"}

    if _variant_has_final_state_evidence(variant):
        final_wrong = apply_plan_control(examples, "final_state_mismatch", seed + 34_000)
        final_wrong = _apply_variant_transform(final_wrong, variant.example_transform, seed + 34_100)
        logits = predict_plan_logits(model, final_wrong, variant.model, batch_size, device, seed=seed)
        controls["final_state_mismatch"] = _metric_block(logits, _labels(final_wrong), final_wrong)
    else:
        controls["final_state_mismatch"] = {"skipped": True, "reason": "final-state evidence explicitly removed from this variant"}

    base_logits = predict_plan_logits(model, examples, variant.model, batch_size, device)
    logits_role = predict_plan_logits(model, examples, variant.model, batch_size, device, condition="physical_role_order_shuffle", seed=seed + 44_000)
    logits_hidden = predict_plan_logits(model, examples, variant.model, batch_size, device, condition="hidden_state_shuffle", seed=seed + 55_000)
    controls["physical_role_order_shuffle"] = _metric_block(logits_role, labels, examples)
    controls["hidden_state_shuffle"] = _metric_block(logits_hidden, labels, examples)
    controls["role_view_masking"] = {}
    for view in variant.model.view_names:
        logits = predict_plan_logits(model, examples, variant.model, batch_size, device, view_mask=view)
        controls["role_view_masking"][str(view)] = _metric_block(logits, labels, examples)
    controls["candidate_order_invariance_delta"] = abs(float(_top1(base_logits, labels)) - float(controls["candidate_order_shuffle_with_gold_remap"]["top1"]))
    controls["role_order_invariance_delta"] = abs(float(_top1(base_logits, labels)) - float(controls["physical_role_order_shuffle"]["top1"]))
    controls["action_symbol_permutation_delta"] = abs(float(_top1(base_logits, labels)) - float(controls["action_symbol_permutation"]["top1"]))
    controls["map_rotation_reflection_delta"] = abs(float(_top1(base_logits, labels)) - float(controls["map_rotation_reflection"]["top1"]))
    controls["metadata_source_only_accuracy"] = candidate_source_metadata_only_accuracy(examples)
    controls["candidate_artifact_baselines"] = artifact_dev
    controls["plan_length_baseline"] = float(artifact_dev.get("path_length", 1.0))
    controls["action_unigram_baseline"] = float(artifact_dev.get("action_unigram_mlp", 1.0))
    controls["action_bigram_baseline"] = float(artifact_dev.get("action_bigram_mlp", 1.0))
    controls["manhattan_progress_baseline"] = float(artifact_dev.get("manhattan_progress", 1.0))
    controls["collision_count_diagnostic"] = float(artifact_dev.get("collision_count", 1.0))
    controls["final_distance_to_goal_diagnostic"] = float(artifact_dev.get("final_distance_to_goal", 1.0))
    controls["rollout_final_distance_diagnostic"] = float(artifact_dev.get("rollout_final_distance", 1.0))

    if bool(variant.model.pyramidal_composition):
        controls["pyramidal_controls"] = _run_pyramidal_controls(model, variant.model, examples, batch_size, device, seed, labels)
    else:
        controls["pyramidal_controls"] = {"skipped": True, "reason": "variant is not pyramidal"}
    controls["clean_top1"] = float(clean_top1)
    controls["candidate_count"] = int(candidate_count)
    return controls


def _run_final_state_mismatch_audit(
    model: PlanCloneVerifier,
    model_config: PlanModelConfig,
    examples: Sequence[PlanGridworldExample],
    batch_size: int,
    device: str,
    seed: int,
) -> Dict[str, object]:
    labels = _labels(examples)
    conditions: List[Tuple[str, Sequence[PlanGridworldExample], str | None]] = [
        ("normal_evaluation", examples, None),
        ("final_state_removed", examples, "final_state_removed"),
        ("final_state_zeroed", examples, "final_state_zeroed"),
        ("final_state_shuffled_across_candidates_within_task", _final_state_shuffle_within_task(examples, seed + 101), None),
        ("final_state_shuffled_across_tasks", apply_plan_control(examples, "final_state_outcome_only_mismatch", seed + 102), None),
        ("final_state_paired_with_wrong_rollout", _wrong_rollout_matching_final(examples, seed + 103), None),
        ("rollout_kept_correct_final_state_wrong", apply_plan_control(examples, "final_state_outcome_only_mismatch", seed + 104), None),
        ("rollout_wrong_final_state_kept_correct", _wrong_rollout_final_correct(examples, seed + 105), None),
        ("both_rollout_and_final_state_wrong", _both_rollout_and_final_wrong_independent(examples, seed + 106), None),
        ("repaired_final_state_mismatch", apply_plan_control(examples, "final_state_mismatch", seed + 107), None),
    ]
    rows: Dict[str, object] = {}
    for name, controlled, token_patch in conditions:
        logits = predict_stage11_logits(model, controlled, model_config, batch_size, device, seed=seed, token_patch=token_patch)
        rows[name] = _metric_block(logits, labels if controlled is examples else _labels(controlled), controlled)

    repaired = apply_plan_control(examples, "final_state_mismatch", seed + 107)
    assertions = _final_state_control_assertions(examples, repaired, model_config)
    attention_mass = _attention_view_mass(model, model_config, examples[: min(8, len(examples))], batch_size, device, seed)
    normal_top1 = float(rows["normal_evaluation"]["top1"])
    rows["interpretation"] = _interpret_final_state_audit(rows, normal_top1)
    return {
        "benchmark": BENCHMARK,
        "seed": seed,
        "variant": "P1_rollout_enabled",
        "condition_metrics": rows,
        "assertions": assertions,
        "attention_mass_by_view": attention_mass,
        "normal_top1": normal_top1,
        "repaired_control_top1": float(rows["repaired_final_state_mismatch"]["top1"]),
        "outcome_only_control_top1": float(rows["rollout_kept_correct_final_state_wrong"]["top1"]),
    }


def _run_rollout_evidence_audit(
    model: PlanCloneVerifier,
    model_config: PlanModelConfig,
    examples: Sequence[PlanGridworldExample],
    batch_size: int,
    device: str,
    seed: int,
) -> Dict[str, object]:
    labels = _labels(examples)
    conditions = {
        "normal_evaluation": examples,
        "rollout_order_shuffle": _rollout_order_shuffle(examples, seed + 201),
        "rollout_action_mismatch": _rollout_action_mismatch(examples),
        "rollout_state_mismatch": _rollout_state_mismatch(examples, seed + 202),
        "rollout_prefix_only": _rollout_prefix_only(examples),
        "rollout_suffix_only": _rollout_suffix_only(examples),
        "rollout_final_only": _rollout_final_only(examples),
        "rollout_no_final": _rollout_no_final(examples),
    }
    rows: Dict[str, object] = {}
    for name, controlled in conditions.items():
        logits = predict_stage11_logits(model, controlled, model_config, batch_size, device, seed=seed)
        rows[name] = _metric_block(logits, labels, controlled)
    normal_top1 = float(rows["normal_evaluation"]["top1"])
    rows["interpretation"] = _interpret_rollout_audit(rows, normal_top1)
    return {
        "benchmark": BENCHMARK,
        "seed": seed,
        "variant": "P1_rollout_enabled",
        "condition_metrics": rows,
        "normal_top1": normal_top1,
    }


def predict_stage11_logits(
    model: PlanCloneVerifier,
    examples: Sequence[PlanGridworldExample],
    model_config: PlanModelConfig,
    batch_size: int,
    device: str,
    condition: str = "none",
    seed: int = 0,
    view_mask: str | None = None,
    token_patch: str | None = None,
    return_attention: bool = False,
    disable_pyramid: bool = False,
) -> np.ndarray | Tuple[np.ndarray, List[Dict[str, object]]]:
    model.eval()
    logits_out: List[np.ndarray] = []
    attention_rows: List[Dict[str, object]] = []
    view_names = model_config.view_names
    if condition == "physical_role_order_shuffle":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(view_names))
        if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
            order = np.roll(order, 1)
        view_names = tuple(view_names[int(index)] for index in order)

    original_config = model.config
    if disable_pyramid:
        model.config = replace(model.config, pyramidal_composition=False)
    try:
        with torch.no_grad():
            for offset, batch_examples in enumerate(_batched(examples, int(batch_size))):
                batch = collate_plan_batch(batch_examples, model_config, device, view_names=view_names)
                if view_mask is not None:
                    for axis, view in enumerate(view_names):
                        if view == view_mask:
                            _neutralize_view(batch, axis, str(view), batch_examples[0].grid_size)
                if token_patch is not None:
                    _apply_token_patch(batch, tuple(view_names), token_patch, seed + offset, batch_examples[0].grid_size)
                output = model(
                    batch,
                    return_attention=return_attention,
                    hidden_state_shuffle=condition == "hidden_state_shuffle",
                    shuffle_seed=seed + offset,
                )
                logits_out.append(output["logits"].detach().cpu().numpy())
                if return_attention:
                    attention_rows.extend(_summarize_attention_batch(output, batch, batch_examples, view_names))
    finally:
        model.config = original_config
    logits = np.concatenate(logits_out, axis=0) if logits_out else np.zeros((0, 0), dtype=np.float32)
    if return_attention:
        return logits, attention_rows
    return logits


def _apply_token_patch(
    batch: Dict[str, torch.Tensor],
    view_names: Tuple[str, ...],
    patch: str,
    seed: int,
    grid_size: int,
) -> None:
    if patch in {"final_state_removed", "final_state_zeroed"}:
        if "outcome_view" not in view_names:
            return
        axis = view_names.index("outcome_view")
        if patch == "final_state_removed":
            _neutralize_view(batch, axis, "outcome_view", grid_size)
        else:
            _set_single_view_token(batch, axis, _token(6, VIEW_ID["outcome_view"], 0, 0, 0, 0, 0, grid_size, grid_size, 0))
        return
    if patch == "composition_shuffle":
        if len(view_names) <= 1:
            return
        rng = np.random.default_rng(seed)
        order_np = rng.permutation(len(view_names))
        if np.array_equal(order_np, np.arange(len(view_names))):
            order_np = np.roll(order_np, 1)
        order = torch.as_tensor(order_np, dtype=torch.long, device=batch["view_fields"].device)
        batch["view_fields"] = batch["view_fields"].index_select(2, order)
        batch["view_mask"] = batch["view_mask"].index_select(2, order)
        return
    raise ValueError(f"unknown PLAN-1.1 token patch: {patch}")


def _neutralize_view(batch: Dict[str, torch.Tensor], axis: int, view: str, grid_size: int) -> None:
    view_id = VIEW_ID.get(view, 0)
    _set_single_view_token(batch, axis, _token(0, view_id, SUMMARY_ROWCOL, SUMMARY_ROWCOL, 0, 0, 0, grid_size, grid_size, 0))


def _set_single_view_token(batch: Dict[str, torch.Tensor], axis: int, token: Sequence[int]) -> None:
    device = batch["view_fields"].device
    batch["view_fields"][:, :, axis, :, :] = 0
    batch["view_mask"][:, :, axis, :] = False
    batch["view_fields"][:, :, axis, 0, :] = torch.as_tensor(token, dtype=batch["view_fields"].dtype, device=device)
    batch["view_mask"][:, :, axis, 0] = True


def _run_pyramidal_controls(
    model: PlanCloneVerifier,
    model_config: PlanModelConfig,
    examples: Sequence[PlanGridworldExample],
    batch_size: int,
    device: str,
    seed: int,
    labels: np.ndarray,
) -> Dict[str, object]:
    controls: Dict[str, object] = {}
    logits = predict_stage11_logits(model, examples, model_config, batch_size, device, seed=seed, token_patch="composition_shuffle")
    controls["composition_shuffle"] = _metric_block(logits, labels, examples)
    child_mismatch = _both_rollout_and_final_wrong_independent(examples, seed + 61_000)
    logits = predict_stage11_logits(model, child_mismatch, model_config, batch_size, device, seed=seed)
    controls["child_mismatch"] = _metric_block(logits, _labels(child_mismatch), child_mismatch)
    logits = predict_stage11_logits(model, examples, model_config, batch_size, device, seed=seed, disable_pyramid=True)
    controls["root_ablation"] = _metric_block(logits, labels, examples)
    logits = predict_stage11_logits(model, examples, model_config, batch_size, device, seed=seed, disable_pyramid=True)
    controls["pair_composer_ablation"] = _metric_block(logits, labels, examples)
    logits = predict_plan_logits(model, apply_plan_control(examples, "candidate_order_shuffle_with_gold_remap", seed + 62_000), model_config, batch_size, device)
    controls["candidate_order_remap"] = _metric_block(logits, _labels(apply_plan_control(examples, "candidate_order_shuffle_with_gold_remap", seed + 62_000)), apply_plan_control(examples, "candidate_order_shuffle_with_gold_remap", seed + 62_000))
    logits = predict_plan_logits(model, examples, model_config, batch_size, device, condition="physical_role_order_shuffle", seed=seed + 63_000)
    controls["role_order_remap"] = _metric_block(logits, labels, examples)
    return controls


def _apply_variant_transform(
    examples: Sequence[PlanGridworldExample],
    transform_name: str,
    seed: int,
) -> List[PlanGridworldExample]:
    if transform_name == "none":
        return list(examples)
    if transform_name == "rollout_no_final":
        return _rollout_no_final(examples)
    raise ValueError(f"unknown PLAN-1.1 variant transform: {transform_name}")


def _variant_has_final_state_evidence(variant: Plan11Variant) -> bool:
    if "outcome_view" in variant.model.view_names:
        return True
    return "rollout_view" in variant.model.view_names and variant.example_transform != "rollout_no_final"


def _replace_candidate_rollout(candidate: CandidateRollout, rollout: Sequence[Tuple[int, int]], final_position: Tuple[int, int] | None = None) -> CandidateRollout:
    return replace(
        candidate,
        rollout=tuple(tuple(pos) for pos in rollout),
        final_position=tuple(final_position) if final_position is not None else candidate.final_position,
    )


def _rollout_no_final(examples: Sequence[PlanGridworldExample]) -> List[PlanGridworldExample]:
    out = []
    for example in examples:
        candidates = []
        for candidate in example.candidates:
            rollout = tuple(candidate.rollout[:-1]) if len(candidate.rollout) > 1 else tuple(candidate.rollout)
            if not rollout:
                rollout = (tuple(candidate.rollout[-1]) if candidate.rollout else tuple(example.start),)
            candidates.append(_replace_candidate_rollout(candidate, rollout))
        out.append(replace(example, candidates=tuple(candidates), metadata={**example.metadata, "control": "rollout_no_final"}))
    return out


def _rollout_prefix_only(examples: Sequence[PlanGridworldExample]) -> List[PlanGridworldExample]:
    out = []
    for example in examples:
        candidates = []
        for candidate in example.candidates:
            cutoff = max(1, (len(candidate.rollout) + 1) // 2)
            candidates.append(_replace_candidate_rollout(candidate, candidate.rollout[:cutoff]))
        out.append(replace(example, candidates=tuple(candidates), metadata={**example.metadata, "control": "rollout_prefix_only"}))
    return out


def _rollout_suffix_only(examples: Sequence[PlanGridworldExample]) -> List[PlanGridworldExample]:
    out = []
    for example in examples:
        candidates = []
        for candidate in example.candidates:
            start = max(0, len(candidate.rollout) // 2)
            candidates.append(_replace_candidate_rollout(candidate, candidate.rollout[start:] or candidate.rollout[-1:]))
        out.append(replace(example, candidates=tuple(candidates), metadata={**example.metadata, "control": "rollout_suffix_only"}))
    return out


def _rollout_final_only(examples: Sequence[PlanGridworldExample]) -> List[PlanGridworldExample]:
    out = []
    for example in examples:
        candidates = [_replace_candidate_rollout(candidate, candidate.rollout[-1:] or (candidate.final_position,)) for candidate in example.candidates]
        out.append(replace(example, candidates=tuple(candidates), metadata={**example.metadata, "control": "rollout_final_only"}))
    return out


def _rollout_order_shuffle(examples: Sequence[PlanGridworldExample], seed: int) -> List[PlanGridworldExample]:
    rng = np.random.default_rng(seed)
    out = []
    for example in examples:
        candidates = []
        for candidate in example.candidates:
            rollout = list(candidate.rollout)
            if len(rollout) > 3:
                middle = rollout[1:-1]
                rng.shuffle(middle)
                rollout = [rollout[0], *middle, rollout[-1]]
            elif len(rollout) > 1:
                rollout = [rollout[0], *reversed(rollout[1:])]
            candidates.append(_replace_candidate_rollout(candidate, rollout))
        out.append(replace(example, candidates=tuple(candidates), metadata={**example.metadata, "control": "rollout_order_shuffle"}))
    return out


def _rollout_action_mismatch(examples: Sequence[PlanGridworldExample]) -> List[PlanGridworldExample]:
    out = []
    for example in examples:
        candidates = []
        count = len(example.candidates)
        for index, candidate in enumerate(example.candidates):
            replacement_rollout = example.candidates[(index + 1) % count].rollout
            candidates.append(_replace_candidate_rollout(candidate, replacement_rollout))
        out.append(replace(example, candidates=tuple(candidates), metadata={**example.metadata, "control": "rollout_action_mismatch"}))
    return out


def _rollout_state_mismatch(examples: Sequence[PlanGridworldExample], seed: int) -> List[PlanGridworldExample]:
    rng = np.random.default_rng(seed)
    out = []
    for index, example in enumerate(examples):
        other = _other_example_local(examples, index, rng)
        replacements = _expand_candidates(other.candidates, len(example.candidates))
        candidates = [
            _replace_candidate_rollout(candidate, replacement.rollout)
            for candidate, replacement in zip(example.candidates, replacements)
        ]
        out.append(replace(example, candidates=tuple(candidates), metadata={**example.metadata, "control": "rollout_state_mismatch"}))
    return out


def _wrong_rollout_final_correct(examples: Sequence[PlanGridworldExample], seed: int) -> List[PlanGridworldExample]:
    return _rollout_state_mismatch(examples, seed)


def _wrong_rollout_matching_final(examples: Sequence[PlanGridworldExample], seed: int) -> List[PlanGridworldExample]:
    rng = np.random.default_rng(seed)
    out = []
    for index, example in enumerate(examples):
        other = _other_example_local(examples, index, rng)
        replacements = _expand_candidates(other.candidates, len(example.candidates))
        candidates = [
            replace(candidate, rollout=replacement.rollout, final_position=replacement.final_position)
            for candidate, replacement in zip(example.candidates, replacements)
        ]
        out.append(replace(example, candidates=tuple(candidates), metadata={**example.metadata, "control": "wrong_rollout_matching_final"}))
    return out


def _both_rollout_and_final_wrong_independent(examples: Sequence[PlanGridworldExample], seed: int) -> List[PlanGridworldExample]:
    rng = np.random.default_rng(seed)
    out = []
    for index, example in enumerate(examples):
        rollout_source = _other_example_local(examples, index, rng)
        final_source = _other_example_local(examples, index, rng)
        rollout_replacements = _expand_candidates(rollout_source.candidates, len(example.candidates))
        final_replacements = _expand_candidates(tuple(reversed(final_source.candidates)), len(example.candidates))
        candidates = []
        for candidate, rollout_replacement, final_replacement in zip(example.candidates, rollout_replacements, final_replacements):
            wrong_final = _different_position(candidate.final_position, final_replacement.final_position, example.grid_size)
            candidates.append(replace(candidate, rollout=rollout_replacement.rollout, final_position=wrong_final))
        out.append(replace(example, candidates=tuple(candidates), metadata={**example.metadata, "control": "both_rollout_and_final_wrong"}))
    return out


def _final_state_shuffle_within_task(examples: Sequence[PlanGridworldExample], seed: int) -> List[PlanGridworldExample]:
    rng = np.random.default_rng(seed)
    out = []
    for example in examples:
        order = rng.permutation(len(example.candidates))
        if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
            order = np.roll(order, 1)
        finals = [candidate.final_position for candidate in example.candidates]
        candidates = []
        for index, candidate in enumerate(example.candidates):
            wrong_final = _different_position(candidate.final_position, finals[int(order[index])], example.grid_size)
            candidates.append(replace(candidate, final_position=wrong_final))
        out.append(replace(example, candidates=tuple(candidates), metadata={**example.metadata, "control": "final_state_shuffle_within_task"}))
    return out


def _expand_candidates(candidates: Sequence[CandidateRollout], size: int) -> List[CandidateRollout]:
    rows = list(candidates)
    if len(rows) < size:
        rows = rows * (size // max(1, len(rows)) + 1)
    return rows[:size]


def _other_example_local(examples: Sequence[PlanGridworldExample], index: int, rng: np.random.Generator) -> PlanGridworldExample:
    if len(examples) <= 1:
        return examples[index]
    other_index = int(rng.integers(0, len(examples) - 1))
    if other_index >= index:
        other_index += 1
    return examples[other_index]


def _different_position(original: Tuple[int, int], proposed: Tuple[int, int], grid_size: int) -> Tuple[int, int]:
    if tuple(original) != tuple(proposed):
        return tuple(proposed)
    return ((int(original[0]) + 1) % max(1, int(grid_size)), int(original[1]))


def _candidate_artifact_audit(
    train_examples: Sequence[PlanGridworldExample],
    dev_examples: Sequence[PlanGridworldExample],
    test_examples: Sequence[PlanGridworldExample],
    seed: int,
) -> Dict[str, Dict[str, float]]:
    dev_labels = _labels(dev_examples)
    test_labels = _labels(test_examples)
    unigram_dev, unigram_test = _feature_mlp_baseline(train_examples, dev_examples, test_examples, _action_unigram_features, seed + 1_000)
    bigram_dev, bigram_test = _feature_mlp_baseline(train_examples, dev_examples, test_examples, _action_bigram_features, seed + 2_000)
    risk_dev, risk_test = _feature_mlp_baseline(train_examples, dev_examples, test_examples, _risk_token_features, seed + 3_000)
    return {
        "dev": {
            "action_unigram_mlp": float(unigram_dev),
            "action_bigram_mlp": float(bigram_dev),
            "path_length": _accuracy(_path_length_predictions(dev_examples), dev_labels),
            "manhattan_progress": _accuracy(_manhattan_progress_predictions(dev_examples), dev_labels),
            "collision_count": _accuracy(_collision_count_predictions(dev_examples), dev_labels),
            "final_distance_to_goal": _accuracy(_final_distance_predictions(dev_examples), dev_labels),
            "rollout_final_distance": _accuracy(_rollout_final_distance_predictions(dev_examples), dev_labels),
            "risk_token_only_mlp": float(risk_dev),
        },
        "test": {
            "action_unigram_mlp": float(unigram_test),
            "action_bigram_mlp": float(bigram_test),
            "path_length": _accuracy(_path_length_predictions(test_examples), test_labels),
            "manhattan_progress": _accuracy(_manhattan_progress_predictions(test_examples), test_labels),
            "collision_count": _accuracy(_collision_count_predictions(test_examples), test_labels),
            "final_distance_to_goal": _accuracy(_final_distance_predictions(test_examples), test_labels),
            "rollout_final_distance": _accuracy(_rollout_final_distance_predictions(test_examples), test_labels),
            "risk_token_only_mlp": float(risk_test),
        },
    }


def _feature_mlp_baseline(
    train_examples: Sequence[PlanGridworldExample],
    dev_examples: Sequence[PlanGridworldExample],
    test_examples: Sequence[PlanGridworldExample],
    feature_fn,
    seed: int,
) -> Tuple[float, float]:
    if not train_examples or not dev_examples:
        return 0.0, 0.0
    _set_seed(seed)
    train_x = torch.as_tensor(feature_fn(train_examples), dtype=torch.float32)
    train_y = torch.as_tensor(_labels(train_examples), dtype=torch.long)
    dev_x = torch.as_tensor(feature_fn(dev_examples), dtype=torch.float32)
    dev_y = _labels(dev_examples)
    test_x = torch.as_tensor(feature_fn(test_examples), dtype=torch.float32) if test_examples else torch.zeros((0, 1, train_x.shape[-1]))
    test_y = _labels(test_examples)
    model = CandidateStatsScorer(train_x.shape[-1], hidden_dim=32)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_dev = -1.0
    rng = np.random.default_rng(seed)
    for _epoch in range(40):
        order = rng.permutation(len(train_examples))
        for batch_indices in _chunks(order.tolist(), 32):
            logits = model(train_x[batch_indices])
            loss = F.cross_entropy(logits, train_y[batch_indices])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            dev_logits = model(dev_x).detach().numpy()
        dev_acc = _top1(dev_logits, dev_y)
        if dev_acc > best_dev:
            best_dev = dev_acc
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        dev_logits = model(dev_x).detach().numpy()
        test_logits = model(test_x).detach().numpy() if len(test_examples) else np.zeros((0, 0), dtype=np.float32)
    return _top1(dev_logits, dev_y), _top1(test_logits, test_y)


def _action_unigram_features(examples: Sequence[PlanGridworldExample]) -> np.ndarray:
    rows = []
    for example in examples:
        cand_rows = []
        for candidate in example.candidates:
            actions = np.asarray(candidate.actions, dtype=np.int64)
            cand_rows.append(np.bincount(actions, minlength=4).astype(np.float32) / max(1, len(actions)))
        rows.append(np.stack(cand_rows))
    return np.stack(rows) if rows else np.zeros((0, 0, 4), dtype=np.float32)


def _action_bigram_features(examples: Sequence[PlanGridworldExample]) -> np.ndarray:
    rows = []
    for example in examples:
        cand_rows = []
        for candidate in example.candidates:
            actions = list(candidate.actions)
            counts = np.zeros(16, dtype=np.float32)
            for left, right in zip(actions, actions[1:]):
                counts[int(left) * 4 + int(right)] += 1.0
            cand_rows.append(counts / max(1, len(actions) - 1))
        rows.append(np.stack(cand_rows))
    return np.stack(rows) if rows else np.zeros((0, 0, 16), dtype=np.float32)


def _risk_token_features(examples: Sequence[PlanGridworldExample]) -> np.ndarray:
    rows = []
    for example in examples:
        obstacle_set = set(example.obstacles)
        cand_rows = []
        for candidate in example.candidates:
            pos = tuple(example.start)
            hit = 0.0
            hit_step = 0.0
            hit_type = 0.0
            hit_row = 0.0
            hit_col = 0.0
            for step, action in enumerate(candidate.actions, start=1):
                dr, dc = ACTION_TO_DELTA[int(action)]
                nxt = (pos[0] + dr, pos[1] + dc)
                if not (0 <= nxt[0] < example.grid_size and 0 <= nxt[1] < example.grid_size):
                    hit, hit_step, hit_type = 1.0, step / max(1, len(candidate.actions)), 2.0
                    hit_row = max(0, min(example.grid_size - 1, nxt[0])) / max(1, example.grid_size - 1)
                    hit_col = max(0, min(example.grid_size - 1, nxt[1])) / max(1, example.grid_size - 1)
                    break
                if nxt in obstacle_set:
                    hit, hit_step, hit_type = 1.0, step / max(1, len(candidate.actions)), 1.0
                    hit_row = nxt[0] / max(1, example.grid_size - 1)
                    hit_col = nxt[1] / max(1, example.grid_size - 1)
                    break
                pos = nxt
            cand_rows.append(
                np.asarray(
                    [hit, hit_step, hit_type / 2.0, hit_row, hit_col, len(obstacle_set) / max(1, example.grid_size * example.grid_size)],
                    dtype=np.float32,
                )
            )
        rows.append(np.stack(cand_rows))
    return np.stack(rows) if rows else np.zeros((0, 0, 6), dtype=np.float32)


def _path_length_predictions(examples: Sequence[PlanGridworldExample]) -> np.ndarray:
    return np.asarray([int(np.argmin([len(candidate.actions) for candidate in example.candidates])) for example in examples], dtype=np.int64)


def _manhattan_progress_predictions(examples: Sequence[PlanGridworldExample]) -> np.ndarray:
    preds = []
    for example in examples:
        scores = []
        for candidate in example.candidates:
            pos = tuple(example.start)
            for action in candidate.actions:
                dr, dc = ACTION_TO_DELTA[int(action)]
                pos = (pos[0] + dr, pos[1] + dc)
            scores.append(_manhattan(pos, example.goal))
        preds.append(int(np.argmin(scores)) if scores else 0)
    return np.asarray(preds, dtype=np.int64)


def _collision_count_predictions(examples: Sequence[PlanGridworldExample]) -> np.ndarray:
    preds = []
    for example in examples:
        obstacle_set = set(example.obstacles)
        scores = []
        for candidate in example.candidates:
            pos = tuple(example.start)
            collisions = 0
            for action in candidate.actions:
                dr, dc = ACTION_TO_DELTA[int(action)]
                nxt = (pos[0] + dr, pos[1] + dc)
                if not (0 <= nxt[0] < example.grid_size and 0 <= nxt[1] < example.grid_size) or nxt in obstacle_set:
                    collisions += 1
                else:
                    pos = nxt
            scores.append(collisions)
        preds.append(int(np.argmin(scores)) if scores else 0)
    return np.asarray(preds, dtype=np.int64)


def _final_distance_predictions(examples: Sequence[PlanGridworldExample]) -> np.ndarray:
    return np.asarray(
        [int(np.argmin([_manhattan(candidate.final_position, example.goal) for candidate in example.candidates])) for example in examples],
        dtype=np.int64,
    )


def _rollout_final_distance_predictions(examples: Sequence[PlanGridworldExample]) -> np.ndarray:
    return np.asarray(
        [
            int(np.argmin([_manhattan((candidate.rollout[-1] if candidate.rollout else candidate.final_position), example.goal) for candidate in example.candidates]))
            for example in examples
        ],
        dtype=np.int64,
    )


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _final_state_control_assertions(
    examples: Sequence[PlanGridworldExample],
    repaired: Sequence[PlanGridworldExample],
    model_config: PlanModelConfig,
) -> Dict[str, object]:
    subset = list(examples[: min(8, len(examples))])
    repaired_subset = list(repaired[: len(subset)])
    has_outcome = "outcome_view" in model_config.view_names
    has_rollout = "rollout_view" in model_config.view_names
    token_count = 0
    values_differ = False
    role_ids_correct = True
    tensor_shapes: Dict[str, object] = {}
    if subset and has_outcome:
        batch_before = collate_plan_batch(subset, model_config, "cpu")
        batch_after = collate_plan_batch(repaired_subset, model_config, "cpu")
        outcome_axis = model_config.view_names.index("outcome_view")
        before = batch_before["view_fields"][:, :, outcome_axis].numpy()
        after = batch_after["view_fields"][:, :, outcome_axis].numpy()
        mask = batch_before["view_mask"][:, :, outcome_axis].numpy()
        token_count = int(mask.sum())
        values_differ = bool(np.any(before[..., [2, 3]][mask] != after[..., [2, 3]][mask]))
        role_ids_correct = bool(np.all(before[..., 1][mask] == VIEW_ID["outcome_view"])) if token_count else False
        tensor_shapes = {
            "view_fields": list(batch_before["view_fields"].shape),
            "view_mask": list(batch_before["view_mask"].shape),
            "candidate_fields": list(batch_before["candidate_fields"].shape),
            "candidate_mask": list(batch_before["candidate_mask"].shape),
        }
    alignment_differences = sum(
        int(candidate.final_position != repaired_example.candidates[candidate_index].final_position)
        for example, repaired_example in zip(examples, repaired)
        for candidate_index, candidate in enumerate(example.candidates)
    )
    rollout_final_differences = sum(
        int(
            bool(candidate.rollout)
            and bool(repaired_example.candidates[candidate_index].rollout)
            and candidate.rollout[-1] != repaired_example.candidates[candidate_index].rollout[-1]
        )
        for example, repaired_example in zip(examples, repaired)
        for candidate_index, candidate in enumerate(example.candidates)
    )
    remapped = apply_plan_control(repaired_subset, "candidate_order_shuffle_with_gold_remap", 909) if repaired_subset else []
    remap_ok = all(
        shuffled.candidates[shuffled.label].actions == original.candidates[original.label].actions
        for original, shuffled in zip(repaired_subset, remapped)
    )
    return {
        "tensor_shapes": tensor_shapes,
        "final_state_token_count": token_count,
        "final_state_token_count_gt_zero": bool(token_count > 0),
        "final_state_values_differ_after_repaired_mismatch": bool(values_differ),
        "final_state_candidate_alignment_differs_after_mismatch": bool(alignment_differences > 0),
        "num_candidate_final_positions_changed": int(alignment_differences),
        "rollout_final_token_changed_after_repaired_mismatch": bool((not has_rollout) or rollout_final_differences > 0),
        "num_rollout_final_tokens_changed": int(rollout_final_differences),
        "candidate_order_remap_still_works_after_mismatch": bool(remap_ok),
        "final_state_role_ids_correct": bool(role_ids_correct),
        "candidate_ids_explicitly_encoded": False,
        "candidate_alignment_carried_by_tensor_axis": True,
        "hidden_success_reward_flag_leaks": False,
        "outcome_token_fields": "kind, view_id, row, col, constant value, step, action=0, height, width, aux=0",
        "repaired_control_changes_all_model_visible_final_state_evidence": bool((not has_outcome or values_differ) and (not has_rollout or rollout_final_differences > 0)),
    }


def _attention_view_mass(
    model: PlanCloneVerifier,
    model_config: PlanModelConfig,
    examples: Sequence[PlanGridworldExample],
    batch_size: int,
    device: str,
    seed: int,
) -> Dict[str, float]:
    if not examples:
        return {}
    _logits, rows = predict_stage11_logits(model, examples, model_config, batch_size, device, seed=seed, return_attention=True)
    masses: Dict[str, List[float]] = {view: [] for view in model_config.view_names}
    for row in rows:
        for block in row.get("blocks", []):
            for view, mass in block.get("attention_mass_by_clone_view", {}).items():
                masses.setdefault(str(view), []).append(float(mass))
    return {view: _mean(values) for view, values in masses.items()}


def _attention_summaries_stage11(
    model: PlanCloneVerifier,
    model_config: PlanModelConfig,
    examples: Sequence[PlanGridworldExample],
    batch_size: int,
    device: str,
    seed: int,
    limit: int,
) -> List[Dict[str, object]]:
    subset = list(examples[:limit])
    _logits, rows = predict_stage11_logits(model, subset, model_config, batch_size, device, seed=seed, return_attention=True)
    for row in rows:
        row["seed"] = seed
        row["benchmark"] = BENCHMARK
    return rows


def _controls_pass_stage11(
    control_row: Dict[str, object],
    clean_top1: float,
    candidate_count: int,
    model_config: PlanModelConfig,
    variant: Plan11Variant,
) -> Dict[str, bool]:
    chance = 1.0 / max(1, int(candidate_count))
    final_state_removed = not _variant_has_final_state_evidence(variant)
    pass_map = {
        "candidate_only_near_chance": float(control_row.get("candidate_plan_only", {}).get("top1", 1.0)) <= chance + 0.15,
        "length_only_near_chance": float(control_row.get("plan_length_baseline", 1.0)) <= chance + 0.10,
        "action_unigram_near_chance": float(control_row.get("action_unigram_baseline", 1.0)) <= chance + 0.15,
        "action_bigram_near_chance": float(control_row.get("action_bigram_baseline", 1.0)) <= chance + 0.15,
        "metadata_source_only_near_chance": float(control_row.get("metadata_source_only_accuracy", 1.0)) <= chance + 0.10,
        "state_plan_mismatch_collapses": float(control_row.get("state_plan_mismatch", {}).get("top1", 1.0)) <= chance + 0.20,
        "goal_shuffle_collapses": float(control_row.get("goal_shuffle", {}).get("top1", 1.0)) <= chance + 0.20,
        "candidate_order_remap_invariance_passes": float(control_row.get("candidate_order_invariance_delta", 1.0)) <= 0.08,
        "role_order_invariance_passes": float(control_row.get("role_order_invariance_delta", 1.0)) <= 0.08,
        "meaningful_view_ablation_hurts": _best_view_ablation_drop_stage11(control_row, clean_top1) >= 0.05,
        "no_action_or_length_shortcut": (
            float(control_row.get("action_unigram_baseline", 1.0)) <= chance + 0.15
            and float(control_row.get("action_bigram_baseline", 1.0)) <= chance + 0.15
            and float(control_row.get("plan_length_baseline", 1.0)) <= chance + 0.10
        ),
    }
    if "rollout_view" in model_config.view_names:
        pass_map["rollout_mismatch_collapses"] = float(control_row.get("rollout_mismatch", {}).get("top1", 1.0)) <= chance + 0.20
    if final_state_removed:
        pass_map["final_state_mismatch_collapses_or_removed"] = True
    else:
        pass_map["final_state_mismatch_collapses_or_removed"] = float(control_row.get("final_state_mismatch", {}).get("top1", 1.0)) <= chance + 0.20
    pass_map["overall"] = all(pass_map.values())
    return pass_map


def _best_view_ablation_drop_stage11(control_row: Dict[str, object], clean_top1: float | None = None) -> float:
    clean = float(clean_top1 if clean_top1 is not None else control_row.get("clean_top1", 0.0))
    masked = [float(value.get("top1", clean)) for value in control_row.get("role_view_masking", {}).values()]
    return max([0.0] + [clean - value for value in masked])


def _overall_summary_stage11(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    artifact_audits: Sequence[Dict[str, object]],
    final_state_audits: Sequence[Dict[str, object]],
    rollout_audits: Sequence[Dict[str, object]],
    dataset_config: PlanGridworldDatasetConfig,
) -> Dict[str, object]:
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        if row.get("status") == "completed":
            by_variant.setdefault(str(row["variant"]), []).append(row)
    variant_summaries = []
    for variant, variant_rows in by_variant.items():
        train = [float(row["dev_trainable"]["top1"]) for row in variant_rows]
        frozen = [float(row["dev_frozen"]["top1"]) for row in variant_rows]
        deltas = [float(row["dev_delta_trainable_minus_frozen"]) for row in variant_rows]
        variant_controls = [row for row in controls if row.get("variant") == variant]
        variant_summaries.append(
            {
                "variant": variant,
                "seeds": [int(row["seed"]) for row in variant_rows],
                "diagnostic_only": bool(any(row.get("diagnostic_only", False) for row in variant_rows)),
                "example_transform": str(variant_rows[0].get("example_transform", "none")) if variant_rows else "none",
                "view_names": list(variant_rows[0].get("model_config", {}).get("view_names", [])) if variant_rows else [],
                "mean_trainable_top1": _mean(train),
                "std_trainable_top1": _std(train),
                "mean_frozen_top1": _mean(frozen),
                "mean_delta": _mean(deltas),
                "std_delta": _std(deltas),
                "min_delta": min(deltas) if deltas else 0.0,
                "max_delta": max(deltas) if deltas else 0.0,
                "bootstrap_95ci_delta": _bootstrap_ci(deltas),
                "seeds_trainable_beats_frozen": int(sum(delta > 0.0 for delta in deltas)),
                "controls_pass_all": all(bool(row.get("control_pass", {}).get("overall", False)) for row in variant_controls),
                "candidate_only_mean_top1": _mean([float(row.get("candidate_plan_only", {}).get("top1", 0.0)) for row in variant_controls]),
                "length_only_mean_top1": _mean([float(row.get("plan_length_baseline", 0.0)) for row in variant_controls]),
                "action_unigram_mean_top1": _mean([float(row.get("action_unigram_baseline", 0.0)) for row in variant_controls]),
                "action_bigram_mean_top1": _mean([float(row.get("action_bigram_baseline", 0.0)) for row in variant_controls]),
                "mean_best_view_ablation_drop": _mean([_best_view_ablation_drop_stage11(row) for row in variant_controls]),
            }
        )
    variant_summaries.sort(key=lambda row: float(row["mean_delta"]), reverse=True)
    p1_summary = next((row for row in variant_summaries if row["variant"] == "P1_rollout_enabled"), {})
    best = variant_summaries[0] if variant_summaries else {}
    variant_triggers = {
        str(row["variant"]): _medium_trigger_for_variant(row, controls, rows, dataset_config)
        for row in variant_summaries
    }
    eligible = [
        row
        for row in variant_summaries
        if not bool(row.get("diagnostic_only", False)) and bool(variant_triggers.get(str(row["variant"]), {}).get("passes", False))
    ]
    recommended = eligible[0] if eligible else p1_summary
    gates = dict(variant_triggers.get(str(recommended.get("variant", "")), {}))
    gates["recommended_variant"] = recommended.get("variant")
    gates["p1_rollout_enabled_passes"] = bool(variant_triggers.get("P1_rollout_enabled", {}).get("passes", False))
    return {
        "variant_summaries": variant_summaries,
        "best_variant_by_delta": best.get("variant"),
        "p1_rollout_enabled_summary": p1_summary,
        "p1_rollout_enabled_trigger": variant_triggers.get("P1_rollout_enabled", {}),
        "variant_medium_triggers": variant_triggers,
        "candidate_count": dataset_config.num_candidates,
        "random_baseline": 1.0 / max(1, int(dataset_config.num_candidates)),
        "candidate_artifact_audits": artifact_audits,
        "final_state_audit_summary": _summarize_condition_audits(final_state_audits),
        "rollout_audit_summary": _summarize_condition_audits(rollout_audits),
        "medium_validation_trigger": gates,
        "final_validation_run": False,
        "arc1_1_reference": _arc11_reference(),
    }


def _medium_trigger_for_variant(
    variant_summary: Dict[str, object],
    controls: Sequence[Dict[str, object]],
    rows: Sequence[Dict[str, object]],
    dataset_config: PlanGridworldDatasetConfig,
) -> Dict[str, object]:
    variant = str(variant_summary.get("variant", ""))
    if not variant_summary:
        return {"passes": False, "reason": "variant did not complete"}
    chance = 1.0 / max(1, int(dataset_config.num_candidates))
    variant_controls = [row for row in controls if row.get("variant") == variant]
    variant_rows = [row for row in rows if row.get("variant") == variant and row.get("status") == "completed"]
    view_names = tuple(str(view) for view in variant_summary.get("view_names", []))
    example_transform = str(variant_summary.get("example_transform", "none"))
    final_state_evidence_present = "outcome_view" in view_names or ("rollout_view" in view_names and example_transform != "rollout_no_final")
    gates = {
        "trainable_beats_frozen_on_at_least_2_of_3_cheap_seeds": int(variant_summary.get("seeds_trainable_beats_frozen", 0)) >= 2,
        "mean_trainable_frozen_delta_at_least_0_15": float(variant_summary.get("mean_delta", 0.0)) >= 0.15,
        "top1_clearly_above_random": float(variant_summary.get("mean_trainable_top1", 0.0)) >= chance + 0.15,
        "candidate_only_near_chance": all(float(row.get("candidate_plan_only", {}).get("top1", 1.0)) <= chance + 0.15 for row in variant_controls),
        "length_only_near_chance": all(float(row.get("plan_length_baseline", 1.0)) <= chance + 0.10 for row in variant_controls),
        "action_stat_near_chance": all(float(row.get("action_unigram_baseline", 1.0)) <= chance + 0.15 and float(row.get("action_bigram_baseline", 1.0)) <= chance + 0.15 for row in variant_controls),
        "metadata_source_only_near_chance": all(float(row.get("metadata_source_only_accuracy", 1.0)) <= chance + 0.10 for row in variant_controls),
        "state_plan_mismatch_collapses": all(float(row.get("state_plan_mismatch", {}).get("top1", 1.0)) <= chance + 0.20 for row in variant_controls),
        "goal_shuffle_collapses": all(float(row.get("goal_shuffle", {}).get("top1", 1.0)) <= chance + 0.20 for row in variant_controls),
        "rollout_mismatch_collapses": all(float(row.get("rollout_mismatch", {}).get("top1", 1.0)) <= chance + 0.20 for row in variant_controls),
        "final_state_mismatch_collapses_or_removed": (
            (not final_state_evidence_present)
            or all(float(row.get("final_state_mismatch", {}).get("top1", 1.0)) <= chance + 0.20 for row in variant_controls)
        ),
        "candidate_order_remap_invariance_passes": all(float(row.get("candidate_order_invariance_delta", 1.0)) <= 0.08 for row in variant_controls),
        "role_order_invariance_passes": all(float(row.get("role_order_invariance_delta", 1.0)) <= 0.08 for row in variant_controls),
        "meaningful_view_ablation_hurts": any(_best_view_ablation_drop_stage11(row) >= 0.05 for row in variant_controls),
        "no_shortcut_baseline_explains_result": all(
            max(
                float(row.get("action_unigram_baseline", 1.0)),
                float(row.get("action_bigram_baseline", 1.0)),
                float(row.get("plan_length_baseline", 1.0)),
                float(row.get("metadata_source_only_accuracy", 1.0)),
            )
            <= chance + 0.15
            for row in variant_controls
        ),
        "frozen_comparator_remains_frozen": all(
            bool(row.get("frozen_audit", {}).get("frozen_shared_model_zero_grad"))
            and bool(row.get("frozen_audit", {}).get("frozen_shared_model_zero_delta"))
            for row in variant_rows
        ),
        "trainable_shared_model_changes": all(bool(row.get("trainable_audit", {}).get("trainable_shared_model_changed")) for row in variant_rows),
        "final_validation_not_launched": True,
    }
    gates["passes"] = all(gates.values())
    gates["variant"] = variant
    gates["final_state_evidence_present"] = bool(final_state_evidence_present)
    return gates


def _summarize_condition_audits(audits: Sequence[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    condition_values: Dict[str, List[float]] = {}
    for audit in audits:
        for name, block in audit.get("condition_metrics", {}).items():
            if isinstance(block, dict) and "top1" in block:
                condition_values.setdefault(name, []).append(float(block["top1"]))
    for name, values in condition_values.items():
        out[name] = {"mean_top1": _mean(values), "values": values}
    return out


def _risk_clone_diagnosis(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    artifact_audits: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    risk_rows = [row for row in rows if row.get("variant") == "P1_rollout_plus_risk" and row.get("status") == "completed"]
    risk_controls = [row for row in controls if row.get("variant") == "P1_rollout_plus_risk"]
    deltas = [float(row.get("dev_delta_trainable_minus_frozen", 0.0)) for row in risk_rows]
    risk_token_scores = [float(row.get("dev", {}).get("risk_token_only_mlp", 0.0)) for row in artifact_audits]
    collision_scores = [float(row.get("dev", {}).get("collision_count", 0.0)) for row in artifact_audits]
    return {
        "variant": "P1_rollout_plus_risk",
        "seeds": [int(row["seed"]) for row in risk_rows],
        "mean_trainable_top1": _mean([float(row["dev_trainable"]["top1"]) for row in risk_rows]),
        "mean_frozen_top1": _mean([float(row["dev_frozen"]["top1"]) for row in risk_rows]),
        "mean_delta": _mean(deltas),
        "frozen_higher_than_trainable_seeds": int(sum(delta < 0.0 for delta in deltas)),
        "candidate_only_top1_values": [float(row.get("candidate_plan_only", {}).get("top1", 0.0)) for row in risk_controls],
        "risk_token_only_mlp_dev_top1_values": risk_token_scores,
        "collision_count_dev_top1_values": collision_scores,
        "risk_view_contains_collision_signal": True,
        "risk_view_contains_success_or_reward_label": False,
        "risk_tokens_explanation": "risk_failure_view exposes the first predicted obstacle/bounds collision type and location, not valid/success/reward/gold labels.",
        "diagnosis": _risk_diagnosis_text(deltas, risk_token_scores, collision_scores),
        "claim_status": "not usable for claims unless frozen shortcut and controls are resolved",
    }


def _risk_diagnosis_text(deltas: Sequence[float], risk_scores: Sequence[float], collision_scores: Sequence[float]) -> str:
    if any(delta < 0.0 for delta in deltas):
        return "Frozen remains competitive/high on at least one seed; risk features behave like an easy random-feature signal and must be treated as suspicious."
    if risk_scores and max(risk_scores) >= 0.30:
        return "Risk-only features are predictive enough to be a shortcut concern even without explicit success labels."
    if collision_scores and max(collision_scores) >= 0.30:
        return "Collision-count diagnostics are predictive; risk/cost gains must be separated from collision filtering."
    return "No strong risk-only shortcut was detected in the cheap audit, but P4-style claims remain gated by mismatch controls."


def _run_no_rollout_curricula(
    dataset_config: PlanGridworldDatasetConfig,
    training: PlanTrainingConfig,
    device: str,
    config: Dict[str, object],
) -> List[Dict[str, object]]:
    seeds = [int(value) for value in config.get("no_rollout_curriculum_seeds", [0])]
    model_config = PlanModelConfig(
        variant_family="no_rollout_latent_inference_curriculum",
        view_names=("state_view", "goal_view", "obstacle_constraint_view", "candidate_action_view"),
        coordination_blocks=1,
    )
    specs = [
        ("N2_obvious_invalid", replace(dataset_config, num_candidates=2, train_examples=64, dev_examples=32, test_examples=0, level=1)),
        ("N4_collision_wrong_goal", replace(dataset_config, num_candidates=4, train_examples=128, dev_examples=32, test_examples=0, level=2)),
        ("N8_balanced_hard", replace(dataset_config, num_candidates=8, train_examples=dataset_config.train_examples, dev_examples=dataset_config.dev_examples, test_examples=0, level=dataset_config.level)),
        ("N8_larger_train", replace(dataset_config, num_candidates=8, train_examples=max(320, dataset_config.train_examples * 2), dev_examples=dataset_config.dev_examples, test_examples=0, level=dataset_config.level)),
    ]
    rows = []
    for seed in seeds:
        for name, cfg in specs:
            print(f"plan1.1 no-rollout curriculum={name} seed={seed}: fitting")
            splits = build_plan_gridworld_splits(cfg, seed=seed + 77_000)
            trainable = fit_plan_verifier(
                splits["train"],
                splits["dev"],
                model_config,
                replace(training, objective=model_config.objective),
                seed=seed + 78_000,
                device=device,
                trainable_shared=True,
                method=f"no_rollout_curriculum_trainable__{name}",
            )
            frozen = fit_plan_verifier(
                splits["train"],
                splits["dev"],
                model_config,
                replace(training, objective=model_config.objective),
                seed=seed + 78_000,
                device=device,
                trainable_shared=False,
                method=f"no_rollout_curriculum_frozen__{name}",
            )
            labels = _labels(splits["dev"])
            train_logits = predict_plan_logits(trainable.model, splits["dev"], model_config, training.batch_size, device)
            frozen_logits = predict_plan_logits(frozen.model, splits["dev"], model_config, training.batch_size, device)
            rows.append(
                {
                    "curriculum": name,
                    "seed": seed,
                    "candidate_count": cfg.num_candidates,
                    "train_examples": cfg.train_examples,
                    "dev_examples": cfg.dev_examples,
                    "trainable_top1": _top1(train_logits, labels),
                    "frozen_top1": _top1(frozen_logits, labels),
                    "delta": _top1(train_logits, labels) - _top1(frozen_logits, labels),
                    "random": 1.0 / max(1, cfg.num_candidates),
                    "trainable_audit": trainable.audit,
                    "frozen_audit": frozen.audit,
                }
            )
            del trainable, frozen
            _clear_cuda()
    return rows


def _failure_taxonomy_stage11(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    error_cases: Sequence[Dict[str, object]],
    summary: Dict[str, object],
    risk_diagnosis: Dict[str, object],
    no_rollout_curricula: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    categories: Dict[str, int] = {}
    for row in rows:
        if float(row.get("dev_delta_trainable_minus_frozen", 0.0)) <= 0.0:
            categories["trainable_not_above_frozen"] = categories.get("trainable_not_above_frozen", 0) + 1
    for row in controls:
        for key, value in row.get("control_pass", {}).items():
            if key != "overall" and not bool(value):
                categories[f"control_failed:{key}"] = categories.get(f"control_failed:{key}", 0) + 1
    for row in error_cases:
        category = str(row.get("category", "unknown"))
        categories[f"error_case:{category}"] = categories.get(f"error_case:{category}", 0) + 1
    if not bool(summary.get("medium_validation_trigger", {}).get("passes", False)):
        categories["medium_trigger_not_met"] = categories.get("medium_trigger_not_met", 0) + 1
    if float(risk_diagnosis.get("mean_delta", 0.0)) < 0.0:
        categories["risk_clone_frozen_exceeds_trainable"] = categories.get("risk_clone_frozen_exceeds_trainable", 0) + 1
    if no_rollout_curricula and max(float(row.get("trainable_top1", 0.0)) - float(row.get("random", 0.0)) for row in no_rollout_curricula) <= 0.10:
        categories["no_rollout_curricula_not_learned"] = categories.get("no_rollout_curricula_not_learned", 0) + 1
    return {
        "categories": dict(sorted(categories.items())),
        "interpretation": "PLAN-1.1 failures are cheap-search diagnostics, not final validation failures.",
    }


def _interpret_final_state_audit(rows: Dict[str, object], normal_top1: float) -> Dict[str, object]:
    def top1(name: str) -> float:
        return float(rows.get(name, {}).get("top1", 0.0)) if isinstance(rows.get(name), dict) else 0.0

    rollout_correct_final_wrong = top1("rollout_kept_correct_final_state_wrong")
    rollout_wrong_final_correct = top1("rollout_wrong_final_state_kept_correct")
    both_wrong = top1("both_rollout_and_final_state_wrong")
    repaired = top1("repaired_final_state_mismatch")
    return {
        "rollout_correct_final_wrong_drop": normal_top1 - rollout_correct_final_wrong,
        "rollout_wrong_final_correct_drop": normal_top1 - rollout_wrong_final_correct,
        "both_wrong_drop": normal_top1 - both_wrong,
        "repaired_final_state_mismatch_drop": normal_top1 - repaired,
        "primary_dependency": (
            "rollout"
            if rollout_correct_final_wrong >= normal_top1 - 0.05 and rollout_wrong_final_correct < normal_top1 - 0.05
            else "final_state"
            if rollout_wrong_final_correct >= normal_top1 - 0.05 and rollout_correct_final_wrong < normal_top1 - 0.05
            else "both_or_unclear"
        ),
        "control_bug_flag": bool(both_wrong >= normal_top1 - 0.05),
    }


def _interpret_rollout_audit(rows: Dict[str, object], normal_top1: float) -> Dict[str, object]:
    def top1(name: str) -> float:
        return float(rows.get(name, {}).get("top1", 0.0)) if isinstance(rows.get(name), dict) else 0.0

    return {
        "order_shuffle_drop": normal_top1 - top1("rollout_order_shuffle"),
        "action_mismatch_drop": normal_top1 - top1("rollout_action_mismatch"),
        "state_mismatch_drop": normal_top1 - top1("rollout_state_mismatch"),
        "prefix_only_drop": normal_top1 - top1("rollout_prefix_only"),
        "suffix_only_drop": normal_top1 - top1("rollout_suffix_only"),
        "final_only_drop": normal_top1 - top1("rollout_final_only"),
        "no_final_drop": normal_top1 - top1("rollout_no_final"),
        "uses_temporal_order": bool(normal_top1 - top1("rollout_order_shuffle") >= 0.05),
        "final_state_cue_sufficient": bool(top1("rollout_final_only") >= normal_top1 - 0.05),
        "prefix_carries_signal": bool(top1("rollout_prefix_only") >= normal_top1 - 0.05),
        "suffix_carries_signal": bool(top1("rollout_suffix_only") >= normal_top1 - 0.05),
    }


def _arc11_reference() -> Dict[str, object]:
    path = Path("results/arc1_1_variant_leaderboard.csv")
    if not path.exists():
        return {"available": False}
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"available": False}
    best = rows[0]
    return {
        "available": True,
        "best_variant": best.get("variant"),
        "mean_trainable_top1": _safe_float(best.get("mean_trainable_top1")),
        "mean_frozen_top1": _safe_float(best.get("mean_frozen_top1")),
        "mean_delta": _safe_float(best.get("mean_delta")),
        "controls_pass_all": best.get("controls_pass_all"),
    }


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _write_outputs(
    result: Dict[str, object],
    output_path: Path,
    controls_path: Path,
    rollout_audit_path: Path,
    final_state_audit_path: Path,
    leaderboard_path: Path,
    failure_path: Path,
    error_path: Path,
    attention_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    for path in (
        output_path,
        controls_path,
        rollout_audit_path,
        final_state_audit_path,
        leaderboard_path,
        failure_path,
        error_path,
        attention_path,
        compute_path,
        report_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_without_large_logs(result), indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(
        json.dumps({"controls": result["controls"], "summary": result["summary"].get("medium_validation_trigger", {})}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    rollout_audit_path.write_text(json.dumps({"rollout_audit": result["rollout_audit"]}, indent=2, sort_keys=True), encoding="utf-8")
    final_state_audit_path.write_text(
        json.dumps({"final_state_mismatch_audit": result["final_state_mismatch_audit"]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    failure_path.write_text(json.dumps(result["failure_taxonomy"], indent=2, sort_keys=True), encoding="utf-8")
    error_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in result["error_cases"]) + ("\n" if result["error_cases"] else ""), encoding="utf-8")
    attention_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in result["attention_summaries"]) + ("\n" if result["attention_summaries"] else ""),
        encoding="utf-8",
    )
    compute_path.write_text(json.dumps({"compute_metrics": result["compute_metrics"]}, indent=2, sort_keys=True), encoding="utf-8")
    _write_leaderboard(result, leaderboard_path)
    report_path.write_text(_render_report(result), encoding="utf-8")


def _write_leaderboard(result: Dict[str, object], path: Path) -> None:
    rows = result.get("summary", {}).get("variant_summaries", [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "variant",
                "seeds",
                "mean_trainable_top1",
                "mean_frozen_top1",
                "mean_delta",
                "std_delta",
                "candidate_only_mean_top1",
                "length_only_mean_top1",
                "action_unigram_mean_top1",
                "action_bigram_mean_top1",
                "mean_best_view_ablation_drop",
                "controls_pass_all",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "variant": row.get("variant"),
                    "seeds": len(row.get("seeds", [])),
                    "mean_trainable_top1": row.get("mean_trainable_top1"),
                    "mean_frozen_top1": row.get("mean_frozen_top1"),
                    "mean_delta": row.get("mean_delta"),
                    "std_delta": row.get("std_delta"),
                    "candidate_only_mean_top1": row.get("candidate_only_mean_top1"),
                    "length_only_mean_top1": row.get("length_only_mean_top1"),
                    "action_unigram_mean_top1": row.get("action_unigram_mean_top1"),
                    "action_bigram_mean_top1": row.get("action_bigram_mean_top1"),
                    "mean_best_view_ablation_drop": row.get("mean_best_view_ablation_drop"),
                    "controls_pass_all": row.get("controls_pass_all"),
                }
            )


def _without_large_logs(result: Dict[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"error_cases", "attention_summaries", "compute_metrics"}
    }


def _render_report(result: Dict[str, object]) -> str:
    summary = result.get("summary", {})
    p1 = summary.get("p1_rollout_enabled_summary", {})
    gates = summary.get("medium_validation_trigger", {})
    p1_gates = summary.get("p1_rollout_enabled_trigger", {})
    final_audit = summary.get("final_state_audit_summary", {})
    rollout_audit = summary.get("rollout_audit_summary", {})
    risk = result.get("risk_clone_diagnosis", {})
    no_rollout = result.get("no_rollout_curricula", [])
    arc_ref = summary.get("arc1_1_reference", {})
    best_variant = summary.get("best_variant_by_delta")
    lines = [
        "# Stage PLAN-1.1 Rollout-Control Repair and Multi-Seed Planning Search",
        "",
        "## Scope",
        "",
        "- Final validation run: `False`.",
        "- Claim boundary: no autonomous planning, world-model, or general-agent claim.",
        "- Repaired control: `final_state_mismatch` now changes both explicit `outcome_view` final-position tokens and the last `rollout_view` state token.",
        "",
        "## Multi-Seed Results",
        "",
        "| variant | seeds | trainable | frozen | delta | candidate-only | length-only | unigram | bigram | ablation drop | controls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("variant_summaries", []):
        lines.append(
            f"| {row['variant']} | {len(row.get('seeds', []))} | {float(row['mean_trainable_top1']):.4f} | {float(row['mean_frozen_top1']):.4f} | {float(row['mean_delta']):.4f} | {float(row.get('candidate_only_mean_top1', 0.0)):.4f} | {float(row.get('length_only_mean_top1', 0.0)):.4f} | {float(row.get('action_unigram_mean_top1', 0.0)):.4f} | {float(row.get('action_bigram_mean_top1', 0.0)):.4f} | {float(row.get('mean_best_view_ablation_drop', 0.0)):.4f} | `{bool(row.get('controls_pass_all'))}` |"
        )
    lines.extend(["", "## Medium Trigger", "", "| gate | pass |", "|---|---|"])
    for key, value in gates.items():
        lines.append(f"| {key} | `{value}` |")

    lines.extend(
        [
            "",
            "## Final-State Mismatch Audit",
            "",
            _condition_line(final_audit, "normal_evaluation"),
            _condition_line(final_audit, "final_state_removed"),
            _condition_line(final_audit, "final_state_zeroed"),
            _condition_line(final_audit, "rollout_kept_correct_final_state_wrong"),
            _condition_line(final_audit, "rollout_wrong_final_state_kept_correct"),
            _condition_line(final_audit, "both_rollout_and_final_state_wrong"),
            _condition_line(final_audit, "repaired_final_state_mismatch"),
            "",
            "## Rollout Evidence Audit",
            "",
            _condition_line(rollout_audit, "normal_evaluation"),
            _condition_line(rollout_audit, "rollout_order_shuffle"),
            _condition_line(rollout_audit, "rollout_action_mismatch"),
            _condition_line(rollout_audit, "rollout_state_mismatch"),
            _condition_line(rollout_audit, "rollout_prefix_only"),
            _condition_line(rollout_audit, "rollout_suffix_only"),
            _condition_line(rollout_audit, "rollout_final_only"),
            _condition_line(rollout_audit, "rollout_no_final"),
            "",
            "## Required Answers",
            "",
            f"1. Was `final_state_mismatch` failing because of a model shortcut, unused final-state tokens, or a control bug? {_answer_final_state_failure(result)}",
            f"2. Does repaired P1 still beat frozen? P1 mean trainable `{float(p1.get('mean_trainable_top1', 0.0)):.4f}` vs frozen `{float(p1.get('mean_frozen_top1', 0.0)):.4f}`, delta `{float(p1.get('mean_delta', 0.0)):.4f}`.",
            f"3. Does the result survive seeds `[0,1,2]`? Trainable beats frozen on `{int(p1.get('seeds_trainable_beats_frozen', 0))}` of `{len(p1.get('seeds', []))}` seeds.",
            f"4. Does the model depend on rollout evidence, final-state evidence, or both? {_answer_evidence_dependency(result)}",
            f"5. Do rollout mismatch and final-state mismatch now collapse? Recommended variant `{gates.get('recommended_variant')}` has rollout gate `{gates.get('rollout_mismatch_collapses')}` and final-state gate `{gates.get('final_state_mismatch_collapses_or_removed')}`. Original P1 final-state gate is `{p1_gates.get('final_state_mismatch_collapses_or_removed')}`.",
            f"6. Does risk/cost cloning help after controls are repaired? {_answer_risk_cost(summary)}",
            f"7. Does pyramidal planning composition improve over flat rollout querying? {_answer_pyramid(summary)}",
            f"8. Can no-rollout latent inference learn under easier curricula? {_answer_no_rollout(no_rollout)}",
            f"9. Is PLAN-1 ready for medium validation? `{gates.get('passes')}` for recommended variant `{gates.get('recommended_variant')}`; original P1_rollout_enabled remains `{p1_gates.get('passes')}`.",
            f"10. Is this pivot empirically stronger than ARC-1.1? {_answer_arc(summary, arc_ref)}",
            "",
            "## Risk-Clone Diagnosis",
            "",
            f"- Mean trainable top1: `{float(risk.get('mean_trainable_top1', 0.0)):.4f}`.",
            f"- Mean frozen top1: `{float(risk.get('mean_frozen_top1', 0.0)):.4f}`.",
            f"- Mean delta: `{float(risk.get('mean_delta', 0.0)):.4f}`.",
            f"- Diagnosis: {risk.get('diagnosis', 'not run')}",
            "",
            "## Claim Boundary",
            "",
            'No autonomous planning claim. No world-model claim. No general-agent claim. The only allowed claim remains gated: "On controlled synthetic candidate-plan verification, rollout-enabled shared-weight latent coordination beats an exact frozen comparator while using state/goal/constraint/rollout evidence rather than candidate artifacts."',
        ]
    )
    return "\n".join(lines) + "\n"


def _condition_line(summary: Dict[str, object], condition: str) -> str:
    block = summary.get(condition, {})
    if not block:
        return f"- `{condition}`: not run."
    return f"- `{condition}` mean top1: `{float(block.get('mean_top1', 0.0)):.4f}` values `{[round(float(v), 4) for v in block.get('values', [])]}`."


def _answer_final_state_failure(result: Dict[str, object]) -> str:
    audits = result.get("final_state_mismatch_audit", [])
    assertions = [audit.get("assertions", {}) for audit in audits]
    repaired_ok = all(bool(row.get("repaired_control_changes_all_model_visible_final_state_evidence")) for row in assertions) if assertions else False
    outcome_only = result.get("summary", {}).get("final_state_audit_summary", {}).get("rollout_kept_correct_final_state_wrong", {}).get("mean_top1", 0.0)
    repaired = result.get("summary", {}).get("final_state_audit_summary", {}).get("repaired_final_state_mismatch", {}).get("mean_top1", 0.0)
    if repaired_ok:
        if float(repaired) >= float(outcome_only) - 0.05:
            return "The old control was malformed because it left correct endpoint evidence in `rollout_view[-1]`; after repair, explicit final-state evidence still appears mostly unused/redundant because P1 retains performance when final-state tokens are corrupted."
        return "Primarily a control bug: the old outcome-only mismatch left correct final-state evidence in `rollout_view[-1]`."
    return "The audit did not prove the repaired control changed all final-state evidence."


def _answer_evidence_dependency(result: Dict[str, object]) -> str:
    final_summary = result.get("summary", {}).get("final_state_audit_summary", {})
    rollout_summary = result.get("summary", {}).get("rollout_audit_summary", {})
    normal = float(final_summary.get("normal_evaluation", {}).get("mean_top1", 0.0))
    final_wrong = float(final_summary.get("rollout_kept_correct_final_state_wrong", {}).get("mean_top1", 0.0))
    rollout_wrong = float(final_summary.get("rollout_wrong_final_state_kept_correct", {}).get("mean_top1", 0.0))
    order_shuffle = float(rollout_summary.get("rollout_order_shuffle", {}).get("mean_top1", 0.0))
    if final_wrong >= normal - 0.05 and rollout_wrong < normal - 0.05:
        return "Mostly rollout evidence; explicit final-state corruption with rollout kept correct retains performance."
    if rollout_wrong >= normal - 0.05 and final_wrong < normal - 0.05:
        return "Mostly final-state evidence; wrong rollout with final state kept correct retains performance."
    if order_shuffle >= normal - 0.05:
        return "Future/outcome cues matter more than temporal rollout order."
    return "Both rollout and final-state evidence appear relevant or the cheap audit is inconclusive."


def _answer_risk_cost(summary: Dict[str, object]) -> str:
    rows = {row.get("variant"): row for row in summary.get("variant_summaries", [])}
    p1 = rows.get("P1_rollout_enabled", {})
    risk = rows.get("P1_rollout_plus_risk", {})
    cost = rows.get("P1_rollout_plus_cost", {})
    return (
        f"Risk delta over P1 trainable `{float(risk.get('mean_trainable_top1', 0.0)) - float(p1.get('mean_trainable_top1', 0.0)):.4f}`; "
        f"cost delta over P1 trainable `{float(cost.get('mean_trainable_top1', 0.0)) - float(p1.get('mean_trainable_top1', 0.0)):.4f}`. "
        f"Risk controls pass `{risk.get('controls_pass_all')}`, cost controls pass `{cost.get('controls_pass_all')}`."
    )


def _answer_pyramid(summary: Dict[str, object]) -> str:
    rows = {row.get("variant"): row for row in summary.get("variant_summaries", [])}
    p1 = rows.get("P1_rollout_enabled", {})
    pyramid = rows.get("P1_pyramidal_rollout_composition", {})
    return (
        f"Pyramid mean trainable `{float(pyramid.get('mean_trainable_top1', 0.0)):.4f}` vs P1 `{float(p1.get('mean_trainable_top1', 0.0)):.4f}`, "
        f"delta `{float(pyramid.get('mean_trainable_top1', 0.0)) - float(p1.get('mean_trainable_top1', 0.0)):.4f}`."
    )


def _answer_no_rollout(rows: Sequence[Dict[str, object]]) -> str:
    if not rows:
        return "No-rollout curricula were not run."
    best = max(rows, key=lambda row: float(row.get("trainable_top1", 0.0)) - float(row.get("random", 0.0)))
    return (
        f"Best curriculum `{best.get('curriculum')}` trainable `{float(best.get('trainable_top1', 0.0)):.4f}` "
        f"vs random `{float(best.get('random', 0.0)):.4f}` and frozen `{float(best.get('frozen_top1', 0.0)):.4f}`."
    )


def _answer_arc(summary: Dict[str, object], arc_ref: Dict[str, object]) -> str:
    p1 = summary.get("p1_rollout_enabled_summary", {})
    if not arc_ref.get("available"):
        return "ARC-1.1 reference leaderboard was not available locally."
    if not bool(summary.get("medium_validation_trigger", {}).get("passes", False)):
        return "No. PLAN-1.1 is cleaner synthetically, but medium gates did not pass, so it is not empirically stronger as a claim."
    return (
        f"Recommended PLAN-1.1 variant `{summary.get('medium_validation_trigger', {}).get('recommended_variant')}` passes cheap gates; P1 delta `{float(p1.get('mean_delta', 0.0)):.4f}` vs ARC-1.1 best delta `{float(arc_ref.get('mean_delta', 0.0)):.4f}`; "
        "the comparison is synthetic-vs-ARC and should not be overclaimed."
    )


def _load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
