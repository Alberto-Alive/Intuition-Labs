from __future__ import annotations

import argparse
import json
import math
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import pstdev
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from src.coordinators.mlp import MLPTrainingConfig
from src.datasets.multiview_code_patch_selection import (
    BALANCED_34B_CANDIDATE_REPRESENTATION,
    BALANCED_34B_DATASET_SOURCE,
    MultiViewTaskExample,
    apply_example_control,
    build_multiview_code_patch_splits,
    dataset_summary,
    output_leakage_audit,
    randomized_labels_for_examples,
    split_leakage_audit,
)
from src.experiments.architecture_search import (
    StageConfig,
    _agent_config,
    _audit_subset,
    _bootstrap_ci,
    _candidate_config,
    _candidate_specs,
    _coordinator_config,
    _fit_baselines,
    _leakage_passes,
    _per_role_ablation,
    _training_config,
)
from src.experiments.real_shared_weight_latent_coordination import (
    BENCHMARK,
    fit_latent_system,
    predict_context_baseline,
    predict_latent_system,
)
from src.experiments.run_stage3_gpu_hard_validation import (
    _audit_float,
    _clear_cuda,
    _compact_leakage,
    _configure_cuda,
    _failure_row,
    _is_oom,
    _load_result,
    _save_bow_checkpoint,
    _save_context_checkpoint,
    _save_fixed_checkpoint,
    _save_latent_checkpoint,
    _save_oracle_checkpoint,
    _save_text_checkpoint,
)
from src.experiments.run_stage34_candidate_sanitization_diagnostics import _accuracy, _structured_oracle_predictions
from src.experiments.run_stage35_model_facing_learnability import CompatibilityFeatureScorer
from src.experiments.run_stage38_full_hard_validation import (
    _dataset_config_for_stage,
    _labels,
    _positive_training,
    _pre_run_checks,
    _save_candidate_pair_checkpoint,
    _stage_from_config,
)
from src.experiments.run_stage38_schema_aware_controls import (
    _candidate_evidence_mismatch,
    _candidate_only,
    _cross_example_view_bundle_shuffle,
    _evidence_only_no_candidates,
    _null_evidence_values,
    _schema_only,
    _schema_preserved_role_value_shuffle,
    _value_shuffle_within_schema,
)
from src.experiments.run_stage4_final_candidate_token_direct_validation import (
    INVARIANCE_CONTROLS,
    PRIMARY_CONTROLS,
    _per_family_compare,
    _readout_stats,
)


DEFAULT_BASE_CONFIG = "configs/stage4_final_candidate_token_direct_validation.json"
RESULTS_PATH = Path("results/stage6_complementarity_search_results.json")
AUDIT_PATH = Path("results/stage6_complementarity_search_audit.jsonl")
REPORT_PATH = Path("reports/STAGE6_COMPLEMENTARITY_SEARCH.md")
CHECKPOINT_DIR = Path("results/stage6_complementarity_search_checkpoints")
BASELINE_4X1 = {
    "name": "candidate_token_direct_lr3e4_clip1",
    "trainable_mean": 0.7892,
    "frozen_mean": 0.1518,
    "delta": 0.6374,
    "min_seed_accuracy": 0.4277,
    "weak_seeds_below_060": 4,
    "std_accuracy": 0.2447,
}
SEARCH_CONTROL_NAMES = tuple(name for name in PRIMARY_CONTROLS if name != "candidate_metadata_only")


@dataclass(frozen=True)
class Stage6Variant:
    name: str
    description: str
    family: str
    num_avenues: int
    avenue_prompt_mode: str = "single"
    avenue_topk: int = 0
    avenue_dropout: float = 0.0
    bottleneck_dim: int = 0
    lambda_adv: float = 0.0
    redundancy_type: str = "none"
    lambda_red: float = 0.0
    coordinator_style: str = "standard_attention"
    anti_dominance: str = "off"
    anti_dominance_weight: float = 0.0
    lr: float = 0.0003
    dropout: float = 0.0
    weight_decay: float = 0.0001
    epochs: int = 50
    patience: int = 6
    gradient_clip_norm: float = 1.0
    include_in_search: bool = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 6 complementarity-regularized multi-avenue search.")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--max-phase", choices=("cheap", "medium", "final"), default="cheap")
    parser.add_argument("--device", default=None)
    parser.add_argument("--recompute-preflight", action="store_true")
    args = parser.parse_args()
    result = run_stage6(
        base_config_path=Path(args.base_config),
        max_phase=str(args.max_phase),
        device_override=args.device,
        recompute_preflight=bool(args.recompute_preflight),
    )
    print(
        "stage6: wrote {results}, {audit}, {report}; final_passed={passed}".format(
            results=RESULTS_PATH,
            audit=AUDIT_PATH,
            report=REPORT_PATH,
            passed=bool(result.get("summary", {}).get("passed_stage6_final_gates", False)),
        )
    )


def run_stage6(
    base_config_path: Path,
    max_phase: str = "cheap",
    device_override: str | None = None,
    recompute_preflight: bool = False,
) -> Dict[str, object]:
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    config = _stage6_config(base_config)
    if device_override is not None:
        config["device"] = str(device_override)
        config["require_cuda"] = str(device_override).startswith("cuda")
    device, hardware = _configure_cuda(config)
    result = _load_result(RESULTS_PATH)
    result["metadata"] = _metadata(config, base_config_path, hardware, device)
    result["config"] = config
    result["stage4_baseline"] = BASELINE_4X1
    result.setdefault("phases", {})
    for phase, rows in list(result["phases"].items()):
        result["phases"][phase] = [row for row in rows if row.get("status") != "completed" or _valid_completed_row(row)]
    result.setdefault("failed_or_interrupted_runs", [])
    result.setdefault("oom_retries", [])
    result["search_space"] = [asdict(variant) for variant in _variants()]

    if recompute_preflight or "pre_run" not in result:
        result["pre_run"] = _load_or_compute_pre_run(config, base_config_path, device, recompute_preflight)
        _write_outputs(result)
        if bool(config.get("stop_on_preflight_failure", True)) and not bool(result["pre_run"]["summary"]["passes"]):
            print("stage6: pre-run shortcut diagnostics failed; search not started")
            return result

    phase_order = ("cheap", "medium", "final")
    max_index = phase_order.index(max_phase)
    selected = [variant for variant in _variants() if variant.include_in_search]
    for phase in phase_order[: max_index + 1]:
        stage = _stage_for_phase(config, phase)
        if phase == "cheap":
            phase_variants = selected
        elif phase == "medium":
            phase_variants = _selected_variants(result, "cheap", top_k=3)
            if not phase_variants:
                print("stage6 medium: no cheap-stage variant passed selection gates")
                break
        else:
            phase_variants = _selected_variants(result, "medium", top_k=1, require_baseline_improvement=True)
            if not phase_variants:
                print("stage6 final: no medium-stage variant passed selection and baseline-improvement gates")
                break
        result["phases"].setdefault(phase, [])
        completed = {
            (int(row["seed"]), str(row["candidate"]))
            for row in result["phases"][phase]
            if _valid_completed_row(row)
        }
        def _persist_row(new_row: Dict[str, object], phase_name: str = phase) -> None:
            existing = [
                row
                for row in result["phases"].setdefault(phase_name, [])
                if (int(row.get("seed", -1)), str(row.get("candidate", ""))) != (int(new_row.get("seed", -1)), str(new_row.get("candidate", "")))
            ]
            existing.append(new_row)
            result["phases"][phase_name] = existing
            _write_outputs(result)

        rows = _run_phase(config, stage, phase, phase_variants, completed, device, hardware, on_row=_persist_row)
        result["phases"][phase] = [
            row
            for row in result["phases"][phase]
            if (int(row.get("seed", -1)), str(row.get("candidate", ""))) not in {
                (int(new_row.get("seed", -1)), str(new_row.get("candidate", ""))) for new_row in rows
            }
        ] + rows
        _write_outputs(result)
    return result


def _run_phase(
    config: Dict[str, object],
    stage: StageConfig,
    phase: str,
    variants: Sequence[Stage6Variant],
    completed: set[tuple[int, str]],
    device: str,
    hardware: Dict[str, object],
    on_row=None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    fallbacks = [int(value) for value in config.get("gpu_safety", {}).get("batch_size_fallbacks", [stage.batch_size, 16, 8])]
    effective_batch = int(config.get("gpu_safety", {}).get("effective_batch_size", stage.batch_size))
    for seed in stage.seeds:
        if all((int(seed), variant.name) in completed for variant in variants):
            print(f"stage6 {phase} seed={seed}: all variant rows already completed; skipping")
            continue
        print(f"stage6 {phase} seed={seed}: building fixed schema-aware splits")
        dataset_config = _dataset_config_for_stage(config, stage)
        splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=Path("code"))
        print(f"stage6 {phase} seed={seed}: fitting required baselines")
        baselines = _fit_baselines(stage, splits, seed=seed, device=device)
        positive = CompatibilityFeatureScorer(training=_positive_training(config), seed=seed + 14_000)
        positive.fit(splits["train"], splits["dev"])
        split_leakage = split_leakage_audit(BENCHMARK, seed, splits)
        output_leakage = output_leakage_audit(BENCHMARK, seed, splits["train"] + splits["dev"] + splits["test"])
        for variant in variants:
            if (int(seed), variant.name) in completed:
                print(f"stage6 {phase} seed={seed}: existing completed row for {variant.name}; skipping")
                continue
            print(f"stage6 {phase} seed={seed}: fitting {variant.name}")
            row_completed = False
            for batch_size in fallbacks:
                accumulation = max(1, int(math.ceil(effective_batch / float(batch_size))))
                active_stage = replace(stage, seeds=(seed,), batch_size=batch_size, gradient_accumulation_steps=accumulation)
                try:
                    row = _run_variant_seed(
                        config=config,
                        stage=active_stage,
                        phase=phase,
                        variant=variant,
                        splits=splits,
                        baselines=baselines,
                        positive=positive,
                        seed=int(seed),
                        device=device,
                        hardware=hardware,
                        split_leakage=split_leakage,
                        output_leakage=output_leakage,
                    )
                    rows.append(row)
                    if on_row is not None:
                        on_row(row)
                    row_completed = True
                    print(
                        "stage6 {phase} seed={seed} {name}: trainable={trainable:.4f} frozen={frozen:.4f} delta={delta:.4f} comp={comp:.4f}".format(
                            phase=phase,
                            seed=seed,
                            name=variant.name,
                            trainable=float(row["dev_accuracy"]["trainable"]),
                            frozen=float(row["dev_accuracy"]["frozen"]),
                            delta=float(row["dev_delta"]),
                            comp=float(row.get("dev_complementarity_score", 0.0)),
                        )
                    )
                    break
                except RuntimeError as exc:
                    if _is_oom(exc) and batch_size != fallbacks[-1]:
                        _clear_cuda()
                        retry = {
                            **_failure_row(seed, batch_size, accumulation, "oom_retry", exc),
                            "phase": phase,
                            "candidate": variant.name,
                        }
                        rows.append(retry)
                        if on_row is not None:
                            on_row(retry)
                        continue
                    _clear_cuda()
                    failure = {
                        **_failure_row(seed, batch_size, accumulation, "failed", exc),
                        "phase": phase,
                        "candidate": variant.name,
                    }
                    rows.append(failure)
                    if on_row is not None:
                        on_row(failure)
                    break
                except Exception as exc:
                    _clear_cuda()
                    failure = {
                        "phase": phase,
                        "seed": int(seed),
                        "candidate": variant.name,
                        "batch_size": int(batch_size),
                        "gradient_accumulation_steps": int(accumulation),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "time_utc": _now(),
                    }
                    rows.append(failure)
                    if on_row is not None:
                        on_row(failure)
                    break
                finally:
                    _clear_cuda()
            if not row_completed and bool(config.get("stop_on_failed_seed", False)):
                return rows
    return rows


def _valid_completed_row(row: Dict[str, object]) -> bool:
    if row.get("status") != "completed":
        return False
    summary = row.get("dataset_summary", {})
    if isinstance(summary, dict) and bool(summary.get("constructed_fallback", False)):
        return False
    return True


def _run_variant_seed(
    config: Dict[str, object],
    stage: StageConfig,
    phase: str,
    variant: Stage6Variant,
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    baselines: Dict[str, object],
    positive: CompatibilityFeatureScorer,
    seed: int,
    device: str,
    hardware: Dict[str, object],
    split_leakage: Dict[str, object],
    output_leakage: Dict[str, object],
) -> Dict[str, object]:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    candidate = _variant_candidate(stage, variant)
    fit = _fit_variant_methods(candidate, variant, stage, splits, seed, device, config, include_randomized=phase == "final")
    fit["candidate_pair_compatibility_mlp"] = positive
    metrics = _metrics(fit, baselines, splits, seed, variant)
    checkpoints = _save_stage6_checkpoints(
        config=config,
        phase=phase,
        seed=seed,
        stage=stage,
        candidate=candidate,
        fit=fit,
        baselines=baselines,
        save_full=phase == "final",
    )
    cuda_peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") and torch.cuda.is_available() else 0
    dev_avenue_audit = _avenue_audit(fit["trainable"], splits["dev"], seed + 31_000, variant.num_avenues)
    test_avenue_audit = _avenue_audit(fit["trainable"], splits["test"], seed, variant.num_avenues)
    subset_curve = _avenue_subset_curve(fit["trainable"], splits["test"], seed, variant.num_avenues)
    removal_audit = _avenue_removal_audit(fit["trainable"], splits["test"], seed, variant.num_avenues)
    row = {
        "stage": "stage6_complementarity_search",
        "phase": phase,
        "seed": int(seed),
        "candidate": variant.name,
        "description": variant.description,
        "status": "completed",
        "completed_at_utc": _now(),
        "variant_config": asdict(variant),
        "architecture_config": _candidate_config(candidate),
        "dev_accuracy": metrics["dev"],
        "test_accuracy": metrics["test"],
        "dev_delta": float(metrics["dev"]["trainable"] - metrics["dev"]["frozen"]),
        "test_delta": float(metrics["test"]["trainable"] - metrics["test"]["frozen"]),
        "dev_complementarity_score": float(dev_avenue_audit["base_accuracy"] - dev_avenue_audit["max_single_avenue_accuracy"]),
        "test_complementarity_score": float(test_avenue_audit["base_accuracy"] - test_avenue_audit["max_single_avenue_accuracy"]),
        "stage_config": asdict(stage),
        "dataset_config": asdict(_dataset_config_for_stage(config, stage)),
        "dataset_summary": dataset_summary(splits),
        "schema_policy": "role-specific schema fields preserved as legitimate program-analysis evidence",
        "dataset_or_control_changes": "none",
        "role_embedding_shuffle_primary_gate": False,
        "selection_uses_test": False,
        "device": device,
        "cuda_device_name": hardware.get("cuda_device_name"),
        "cuda_total_memory": hardware.get("cuda_total_memory"),
        "cuda_max_memory_allocated": cuda_peak,
        "cuda_max_memory_allocated_gb": cuda_peak / float(1024**3),
        "checkpoint_paths": checkpoints,
        "split_leakage_audit": _compact_leakage(split_leakage),
        "split_leakage_audit_passes": _leakage_passes(split_leakage),
        "output_leakage_audit": output_leakage,
        "output_leakage_audit_passes": bool(output_leakage.get("passes", False)),
        "trainable_audit": _audit_subset(fit["trainable"].audit),
        "frozen_audit": _audit_subset(fit["frozen"].audit),
        "randomized_audit": _audit_subset(fit["randomized"].audit) if fit.get("randomized") is not None else None,
        "readout_collapse": {
            "trainable": _readout_stats(fit["trainable"], splits["test"]),
            "frozen": _readout_stats(fit["frozen"], splits["test"]),
        },
        "per_family_accuracy": _per_family_compare(fit["trainable"], fit["frozen"], splits["test"], seed),
        "per_role_ablation": _per_role_ablation(fit["trainable"], splits["test"], seed),
        "dev_avenue_audit": dev_avenue_audit,
        "avenue_audit": test_avenue_audit,
        "avenue_subset_curve": subset_curve,
        "avenue_removal_audit": removal_audit,
    }
    row["success_criteria"] = _row_criteria(row, config)
    return row


def _fit_variant_methods(candidate, variant: Stage6Variant, stage: StageConfig, splits, seed: int, device: str, config: Dict[str, object], include_randomized: bool) -> Dict[str, object]:
    agent_config = _agent_config(stage)
    training = replace(
        _training_config(stage),
        epochs=int(variant.epochs),
        patience=int(variant.patience),
        lr=float(variant.lr),
        weight_decay=float(variant.weight_decay),
        gradient_clip_norm=float(variant.gradient_clip_norm),
    )
    trainable = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=candidate.coordinator_config,
        training_config=training,
        num_classes=8,
        seed=seed + 10_101,
        device=device,
        trainable_agent=True,
        method=f"stage6_{candidate.name}__trainable",
        message_config=candidate.message_config,
    )
    frozen = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=candidate.coordinator_config,
        training_config=training,
        num_classes=8,
        seed=seed + 20_201,
        device=device,
        trainable_agent=False,
        method=f"stage6_{candidate.name}__frozen",
        message_config=candidate.message_config,
    )
    randomized = None
    if include_randomized:
        randomized = fit_latent_system(
            train_examples=splits["train"],
            dev_examples=splits["dev"],
            agent_config=agent_config,
            coordinator_config=candidate.coordinator_config,
            training_config=training,
            num_classes=8,
            seed=seed + 30_301,
            device=device,
            trainable_agent=True,
            method=f"stage6_{candidate.name}__randomized_labels",
            condition="randomized_labels",
            train_labels=randomized_labels_for_examples(splits["train"], seed + 40_401, 8),
            dev_labels=randomized_labels_for_examples(splits["dev"], seed + 50_501, 8),
            message_config=candidate.message_config,
        )
    return {"trainable": trainable, "frozen": frozen, "randomized": randomized}


def _metrics(
    fit: Dict[str, object],
    baselines: Dict[str, object],
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    seed: int,
    variant: Stage6Variant,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for split_name in ("dev", "test"):
        examples = list(splits[split_name])
        split_seed = seed + (1_000 if split_name == "dev" else 2_000)
        y = _labels(examples)
        values = {
            "trainable": _accuracy(predict_latent_system(fit["trainable"], examples, "none", split_seed), y),
            "frozen": _accuracy(predict_latent_system(fit["frozen"], examples, "none", split_seed), y),
            "text_only": _accuracy(baselines["text"].predict(examples), y),
            "raw_latent": _accuracy(predict_latent_system(baselines["raw"], examples, "none", split_seed), y),
            "majority_baseline": float(np.mean(np.full(len(y), int(baselines["majority_prediction"]), dtype=np.int64) == y)),
            "candidate_order_baseline": float(np.mean(np.full(len(y), int(baselines["candidate_order_prediction"]), dtype=np.int64) == y)),
            "single_agent_full_context": _accuracy(predict_context_baseline(baselines["full_context"], examples), y),
            "candidate_pair_compatibility_mlp": _accuracy(fit["candidate_pair_compatibility_mlp"].predict(examples), y),
            "all_role_oracle": _accuracy(_structured_oracle_predictions(examples, [0, 1, 2, 3]), y),
            "explicit_evidence_oracle": _accuracy(_structured_oracle_predictions(examples, [0, 1, 2, 3]), y),
            "hidden_states_shuffled_across_examples": _accuracy(
                predict_latent_system(fit["trainable"], examples, "hidden_states_shuffled_across_examples", split_seed), y
            ),
            "physical_order_shuffled_roles_preserved": _accuracy(
                predict_latent_system(fit["trainable"], examples, "physical_order_shuffled_roles_preserved", split_seed), y
            ),
            "role_embedding_shuffle_diagnostic": _accuracy(
                predict_latent_system(fit["trainable"], examples, "role_labels_shuffled", split_seed), y
            ),
        }
        if fit.get("randomized") is not None:
            values["randomized_labels"] = _accuracy(predict_latent_system(fit["randomized"], examples, "none", split_seed), y)
        for method, baseline in dict(baselines.get("single_view", {})).items():
            values[str(method)] = _accuracy(baseline.predict(examples), y)
        for name, rows in _control_examples(examples, split_seed).items():
            cy = _labels(rows)
            values[name] = _predict_control_accuracy(fit["trainable"], rows, cy, name, split_seed)
        shuffled = apply_example_control(examples, "candidate_order_shuffled", seed=split_seed + 77_000)
        values["candidate_order_shuffled_with_gold_remap"] = _accuracy(
            predict_latent_system(fit["trainable"], shuffled, "none", split_seed), _labels(shuffled)
        )
        values["candidate_order_shuffled"] = values["candidate_order_shuffled_with_gold_remap"]
        values["view_masked"] = values["view_masked_candidates_visible"]
        values["role_labels_shuffled"] = values["role_embedding_shuffle_diagnostic"]
        if int(variant.num_avenues) > 1:
            for avenue_index in range(int(variant.num_avenues)):
                values[f"avenue_only_{avenue_index}"] = _accuracy(
                    predict_latent_system(fit["trainable"], examples, f"avenue_only_{avenue_index}", split_seed), y
                )
        out[split_name] = values
    return out


def _control_examples(examples: Sequence[MultiViewTaskExample], seed: int) -> Dict[str, Sequence[MultiViewTaskExample]]:
    return {
        "candidate_only": _candidate_only(examples),
        "view_masked_candidates_visible": apply_example_control(examples, "view_masked", seed=seed + 11_000),
        "schema_only": _schema_only(examples),
        "null_evidence_values": _null_evidence_values(examples),
        "evidence_only_no_candidates": _evidence_only_no_candidates(examples),
        "value_shuffle_within_schema": _value_shuffle_within_schema(examples, seed + 12_000),
        "cross_example_view_bundle_shuffle": _cross_example_view_bundle_shuffle(examples, seed + 13_000),
        "candidate_evidence_mismatch": _candidate_evidence_mismatch(examples, seed + 14_000),
        "schema_preserved_role_value_shuffle": _schema_preserved_role_value_shuffle(examples, seed + 15_000),
    }


def _predict_control_accuracy(result, examples: Sequence[MultiViewTaskExample], labels: np.ndarray, control: str, seed: int) -> float:
    if control == "candidate_only":
        with _zero_role_and_avenue_embeddings(result.system):
            return _accuracy(predict_latent_system(result, examples, "none", seed), labels)
    return _accuracy(predict_latent_system(result, examples, "none", seed), labels)


@contextmanager
def _zero_role_and_avenue_embeddings(system):
    saved = []
    for module in system.modules():
        for attr in ("role_embedding", "avenue_embedding"):
            embedding = getattr(module, attr, None)
            if isinstance(embedding, torch.nn.Embedding):
                saved.append((embedding, embedding.weight.detach().clone()))
                embedding.weight.data.zero_()
    try:
        yield
    finally:
        for embedding, weight in saved:
            embedding.weight.data.copy_(weight.to(device=embedding.weight.device, dtype=embedding.weight.dtype))


def _variant_candidate(stage: StageConfig, variant: Stage6Variant):
    locked = next(candidate for candidate in _candidate_specs() if candidate.name == "topk_attention_no_head")
    coordinator_family = _coordinator_family(variant.coordinator_style)
    anti_weight = float(variant.anti_dominance_weight)
    if anti_weight <= 0.0 and str(variant.anti_dominance) == "weak":
        anti_weight = 0.003
    if anti_weight <= 0.0 and str(variant.anti_dominance) == "medium":
        anti_weight = 0.01
    msg = replace(
        locked.message_config,
        readout_source="pooled",
        use_message_head=False,
        use_private_cue_aux=False,
        aux_loss_weight=0.0,
        coordinator_family=coordinator_family,
        num_avenues=int(variant.num_avenues),
        avenue_prompt_mode=str(variant.avenue_prompt_mode),
        avenue_dropout=float(variant.avenue_dropout),
        canonicalize_role_order=False,
        canonicalize_avenue_order=False,
        avenue_bottleneck_dim=int(variant.bottleneck_dim),
        single_avenue_adversarial_weight=float(variant.lambda_adv),
        single_avenue_adversarial_num_classes=8,
        redundancy_loss_type=str(variant.redundancy_type),
        redundancy_loss_weight=float(variant.lambda_red),
        anti_dominance_mode=str(variant.anti_dominance),
        anti_dominance_weight=anti_weight,
    )
    coord = replace(
        _coordinator_config(stage, family=coordinator_family, num_layers=1, dropout=float(variant.dropout)),
        family=coordinator_family,
        num_avenues=int(variant.num_avenues),
        avenue_topk=int(variant.avenue_topk),
    )
    return SimpleNamespace(name=variant.name, description=variant.description, message_config=msg, coordinator_config=coord)


def _coordinator_family(style: str) -> str:
    normalized = str(style)
    if normalized == "gated_attention":
        return "candidate_token_gated_attention"
    if normalized == "product_of_experts":
        return "candidate_token_product_of_experts"
    return "candidate_token_cross_attention"


def _variants() -> List[Stage6Variant]:
    return [
        Stage6Variant(
            name="stage5_reference_dropout05",
            description="Stage 5 selected four-avenue dropout architecture without complementarity regularization.",
            family="reference",
            num_avenues=4,
            avenue_prompt_mode="goal",
            avenue_dropout=0.05,
        ),
        *[
            Stage6Variant(
                name=f"bottleneck_d{dim}",
                description=f"Message bottleneck sweep: per-avenue token bottleneck dim {dim}.",
                family="message_bottleneck_sweep",
                num_avenues=4,
                avenue_prompt_mode="goal",
                avenue_dropout=0.05,
                bottleneck_dim=dim,
            )
            for dim in (8, 16, 32, 64)
        ],
        *[
            Stage6Variant(
                name=f"adv_lambda_{str(value).replace('.', 'p')}",
                description=f"Single-avenue adversarial heads with lambda {value}.",
                family="single_avenue_adversarial_heads",
                num_avenues=4,
                avenue_prompt_mode="goal",
                avenue_dropout=0.05,
                bottleneck_dim=32,
                lambda_adv=value,
            )
            for value in (0.01, 0.03, 0.1, 0.3)
        ],
        Stage6Variant(
            name="red_cosine_l001_standard",
            description="Cosine redundancy penalty with standard candidate-token attention.",
            family="redundancy_dominance_penalty",
            num_avenues=4,
            avenue_prompt_mode="goal",
            avenue_dropout=0.10,
            bottleneck_dim=32,
            redundancy_type="cosine",
            lambda_red=0.001,
        ),
        Stage6Variant(
            name="red_cosine_l003_standard_dominance_weak",
            description="Cosine redundancy plus weak coordinator anti-dominance.",
            family="redundancy_dominance_penalty",
            num_avenues=4,
            avenue_prompt_mode="goal",
            avenue_dropout=0.10,
            bottleneck_dim=32,
            redundancy_type="cosine",
            lambda_red=0.003,
            anti_dominance="weak",
        ),
        Stage6Variant(
            name="red_orthogonal_l01_gated",
            description="Orthogonality penalty with gated attention coordinator.",
            family="redundancy_dominance_penalty",
            num_avenues=4,
            avenue_prompt_mode="goal",
            avenue_dropout=0.10,
            bottleneck_dim=32,
            redundancy_type="orthogonality",
            lambda_red=0.01,
            coordinator_style="gated_attention",
            anti_dominance="weak",
        ),
        Stage6Variant(
            name="red_covariance_l003_poe",
            description="Covariance-style redundancy penalty with product-of-experts coordinator.",
            family="redundancy_dominance_penalty",
            num_avenues=4,
            avenue_prompt_mode="goal",
            avenue_dropout=0.10,
            bottleneck_dim=32,
            redundancy_type="covariance",
            lambda_red=0.003,
            coordinator_style="product_of_experts",
            anti_dominance="medium",
        ),
        Stage6Variant(
            name="combo_d16_adv003_red003_weakdom",
            description="Top-pick combination: bottleneck dim 16, adversarial heads, weak cosine redundancy, weak anti-dominance.",
            family="combined_top_pick",
            num_avenues=4,
            avenue_prompt_mode="goal",
            avenue_dropout=0.10,
            bottleneck_dim=16,
            lambda_adv=0.03,
            redundancy_type="cosine",
            lambda_red=0.003,
            anti_dominance="weak",
        ),
    ]


def _selected_variants(result: Dict[str, object], phase: str, top_k: int, require_baseline_improvement: bool = False) -> List[Stage6Variant]:
    summaries = _phase_variant_summaries(result.get("phases", {}).get(phase, []))
    valid = [row for row in summaries if row.get("selection_valid") and str(row["candidate"]) != "stage5_reference_dropout05"]
    if require_baseline_improvement:
        baseline = next((row for row in summaries if str(row["candidate"]).startswith("baseline_4x1")), None)
        if baseline is None:
            baseline = {
                "mean_dev_trainable": float(BASELINE_4X1["trainable_mean"]),
                "min_dev_trainable": float(BASELINE_4X1["min_seed_accuracy"]),
                "weak_dev_seeds_below_060": int(BASELINE_4X1["weak_seeds_below_060"]),
                "mean_dev_delta": float(BASELINE_4X1["delta"]),
                "std_dev_trainable": float(BASELINE_4X1["std_accuracy"]),
            }
        valid = [row for row in valid if _improves_baseline(row, baseline)]
    valid.sort(
        key=lambda row: (
            float(row.get("mean_dev_complementarity_score", -999.0)),
            float(row.get("mean_dev_trainable", -999.0)),
            -float(row.get("mean_dev_best_single_avenue", 999.0)),
            float(row.get("mean_dev_delta", -999.0)),
            float(row.get("min_dev_trainable", -999.0)),
            -float(row.get("std_dev_trainable", 999.0)),
        ),
        reverse=True,
    )
    names = {str(row["candidate"]) for row in valid[:top_k]}
    return [variant for variant in _variants() if variant.name in names]


def _improves_baseline(row: Dict[str, object], baseline: Dict[str, object] | None) -> bool:
    if baseline is None:
        return True
    if float(row.get("mean_dev_trainable", 0.0)) < 0.75:
        return False
    if float(row.get("mean_dev_complementarity_score", 0.0)) < 0.10:
        return False
    return bool(
        float(row.get("mean_dev_trainable", 0.0)) > float(baseline.get("mean_dev_trainable", 0.0))
        or float(row.get("min_dev_trainable", 0.0)) > float(baseline.get("min_dev_trainable", 0.0))
        or int(row.get("weak_dev_seeds_below_060", 99)) < int(baseline.get("weak_dev_seeds_below_060", 99))
        or float(row.get("mean_dev_delta", 0.0)) > float(baseline.get("mean_dev_delta", 0.0))
        or float(row.get("std_dev_trainable", 99.0)) < float(baseline.get("std_dev_trainable", 99.0))
    )


def _phase_variant_summaries(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        if row.get("status") == "completed":
            grouped.setdefault(str(row["candidate"]), []).append(row)
    out = []
    for candidate, items in grouped.items():
        dev_train = [float(row["dev_accuracy"]["trainable"]) for row in items]
        dev_frozen = [float(row["dev_accuracy"]["frozen"]) for row in items]
        dev_delta = [float(row["dev_delta"]) for row in items]
        best_single = [
            float(row.get("dev_avenue_audit", row.get("avenue_audit", {})).get("max_single_avenue_accuracy", 0.0))
            for row in items
        ]
        complementarity = [float(row.get("dev_complementarity_score", dev_train[index] - best_single[index])) for index, row in enumerate(items)]
        control_means = {
            name: _mean([float(row["dev_accuracy"].get(name, 0.0)) for row in items])
            for name in SEARCH_CONTROL_NAMES
            if name in items[0].get("dev_accuracy", {})
        }
        controls_ok = all(value <= 0.18 for value in control_means.values())
        invariance_ok = all(_invariance_preserved(row["dev_accuracy"]) for row in items)
        wins = sum(float(delta) > 0.0 for delta in dev_delta)
        std_dev = pstdev(dev_train) if len(dev_train) > 1 else 0.0
        weak_seeds = sum(value < 0.60 for value in dev_train)
        summary = {
            "candidate": candidate,
            "n_completed": len(items),
            "seeds": [int(row["seed"]) for row in items],
            "mean_dev_trainable": _mean(dev_train),
            "mean_dev_frozen": _mean(dev_frozen),
            "mean_dev_delta": _mean(dev_delta),
            "mean_dev_best_single_avenue": _mean(best_single),
            "mean_dev_complementarity_score": _mean(complementarity),
            "min_dev_trainable": min(dev_train) if dev_train else 0.0,
            "std_dev_trainable": std_dev,
            "weak_dev_seeds_below_060": weak_seeds,
            "trainable_beats_frozen_seeds": wins,
            "controls_near_chance": controls_ok,
            "mean_dev_controls": control_means,
            "invariance_preserved": invariance_ok,
            "selection_valid": bool(
                len(items) >= min(3, max(1, len(items)))
                and wins >= max(1, math.ceil(0.67 * len(items)))
                and _mean(dev_delta) >= 0.20
                and _mean(dev_train) >= 0.60
                and _mean(complementarity) >= 0.02
                and controls_ok
                and invariance_ok
                and std_dev <= 0.25
                and weak_seeds <= 1
            ),
        }
        out.append(summary)
    return out


def _controls_near(metrics: Dict[str, float]) -> bool:
    return all(float(metrics.get(name, 0.0)) <= 0.18 for name in SEARCH_CONTROL_NAMES if name in metrics)


def _invariance_preserved(metrics: Dict[str, float]) -> bool:
    trainable = float(metrics.get("trainable", 0.0))
    return all(abs(float(metrics.get(name, 0.0)) - trainable) <= 0.08 for name in INVARIANCE_CONTROLS if name in metrics)


def _avenue_audit(result, examples: Sequence[MultiViewTaskExample], seed: int, num_avenues: int) -> Dict[str, object]:
    y = _labels(examples)
    base = _accuracy(predict_latent_system(result, examples, "none", seed + 91_000), y)
    only = {}
    for avenue_index in range(max(1, int(num_avenues))):
        only[str(avenue_index)] = _accuracy(
            predict_latent_system(result, examples, f"avenue_only_{avenue_index}", seed + 92_000 + avenue_index),
            y,
        )
    max_single = max(only.values()) if only else base
    return {
        "num_avenues": int(num_avenues),
        "base_accuracy": base,
        "single_avenue_accuracy": only,
        "max_single_avenue_accuracy": max_single,
        "no_single_avenue_explains_result": bool(int(num_avenues) <= 1 or max_single < base - 0.02),
        "complementarity_score": float(base - max_single),
    }


def _avenue_subset_curve(result, examples: Sequence[MultiViewTaskExample], seed: int, num_avenues: int) -> Dict[str, object]:
    y = _labels(examples)
    avenues = list(range(max(1, int(num_avenues))))
    out: Dict[str, object] = {}
    for subset_size in range(1, len(avenues) + 1):
        rows = {}
        for subset in combinations(avenues, subset_size):
            condition = "avenue_subset_" + "-".join(str(index) for index in subset)
            rows[",".join(str(index) for index in subset)] = _accuracy(
                predict_latent_system(result, examples, condition, seed + 93_000 + subset_size * 100 + sum(subset)),
                y,
            )
        values = list(rows.values())
        out[str(subset_size)] = {
            "mean": _mean(values),
            "max": max(values) if values else 0.0,
            "min": min(values) if values else 0.0,
            "subsets": rows,
        }
    return out


def _avenue_removal_audit(result, examples: Sequence[MultiViewTaskExample], seed: int, num_avenues: int) -> Dict[str, object]:
    y = _labels(examples)
    base = _accuracy(predict_latent_system(result, examples, "none", seed + 94_000), y)
    masked = {}
    drops = {}
    for avenue_index in range(max(1, int(num_avenues))):
        acc = _accuracy(
            predict_latent_system(result, examples, f"avenue_masked_{avenue_index}", seed + 95_000 + avenue_index),
            y,
        )
        masked[str(avenue_index)] = acc
        drops[str(avenue_index)] = base - acc
    return {
        "base_accuracy": base,
        "accuracy_when_removed": masked,
        "drop_from_removing": drops,
        "min_drop": min(drops.values()) if drops else 0.0,
        "max_drop": max(drops.values()) if drops else 0.0,
    }


def _row_criteria(row: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    test = row["test_accuracy"]
    near = float(config.get("near_chance_max", 0.18))
    inv_tol = float(config.get("invariance_tolerance", 0.08))
    trainable = float(test.get("trainable", 0.0))
    return {
        "trainable_beats_frozen": trainable > float(test.get("frozen", 0.0)),
        "combined_minus_best_single_avenue_at_least_020": float(row.get("test_complementarity_score", 0.0)) >= 0.20,
        "combined_high_and_single_low": bool(
            trainable >= 0.80
            and float(row.get("avenue_audit", {}).get("max_single_avenue_accuracy", 1.0)) <= 0.55
        ),
        "primary_controls_near_chance": {name: float(test.get(name, 0.0)) <= near for name in SEARCH_CONTROL_NAMES if name in test},
        "invariance_controls_preserve_accuracy": {
            name: abs(float(test.get(name, 0.0)) - trainable) <= inv_tol for name in INVARIANCE_CONTROLS
        },
        "role_embedding_shuffle_primary_gate": False,
        "no_single_avenue_explains_result": bool(row.get("avenue_audit", {}).get("no_single_avenue_explains_result", True)),
        "trainable_agent_changed": float(row.get("trainable_audit", {}).get("agent_parameter_delta") or 0.0) > 0.0,
        "frozen_agent_unchanged": _audit_float(row.get("frozen_audit", {}), "agent_parameter_delta", 1.0) == 0.0,
    }


def _save_stage6_checkpoints(
    config: Dict[str, object],
    phase: str,
    seed: int,
    stage: StageConfig,
    candidate,
    fit: Dict[str, object],
    baselines: Dict[str, object],
    save_full: bool,
) -> Dict[str, str]:
    if not bool(config.get("save_checkpoints", True)):
        return {}
    root = CHECKPOINT_DIR / phase / candidate.name / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "trainable": root / f"trainable_{candidate.name}.pt",
        "frozen": root / f"frozen_{candidate.name}.pt",
    }
    _save_latent_checkpoint(paths["trainable"], seed, stage, candidate.name, _candidate_config(candidate), fit["trainable"])
    _save_latent_checkpoint(paths["frozen"], seed, stage, candidate.name, _candidate_config(candidate), fit["frozen"])
    if fit.get("randomized") is not None:
        paths["randomized_labels"] = root / f"randomized_labels_{candidate.name}.pt"
        _save_latent_checkpoint(paths["randomized_labels"], seed, stage, f"randomized_labels__{candidate.name}", _candidate_config(candidate), fit["randomized"])
    if save_full:
        full_paths = {
            "raw_latent": root / "raw_latent_baseline.pt",
            "text_only": root / "text_only_multi_agent_baseline.pt",
            "majority_baseline": root / "majority_baseline.pt",
            "candidate_order_baseline": root / "candidate_order_baseline.pt",
            "single_agent_full_context": root / "single_agent_full_context.pt",
            "candidate_pair_compatibility_mlp": root / "candidate_pair_compatibility_mlp.pt",
            "explicit_evidence_oracle": root / "explicit_evidence_oracle.pt",
        }
        paths.update(full_paths)
        _save_latent_checkpoint(
            full_paths["raw_latent"],
            seed,
            stage,
            "raw_latent_baseline",
            {
                "description": "Raw pooled latent baseline.",
                "message_config": asdict(baselines["raw"].system.message_config),
                "coordinator_config": asdict(_coordinator_config(stage, family="cross_attention", num_layers=1, dropout=0.0)),
            },
            baselines["raw"],
        )
        _save_text_checkpoint(full_paths["text_only"], seed, stage, baselines["text"])
        _save_fixed_checkpoint(full_paths["majority_baseline"], seed, stage, "majority_baseline", int(baselines["majority_prediction"]))
        _save_fixed_checkpoint(full_paths["candidate_order_baseline"], seed, stage, "candidate_order_baseline", int(baselines["candidate_order_prediction"]))
        _save_context_checkpoint(full_paths["single_agent_full_context"], seed, stage, baselines["full_context"])
        for method, baseline in dict(baselines.get("single_view", {})).items():
            path = root / f"{method}.pt"
            paths[str(method)] = path
            _save_bow_checkpoint(path, seed, stage, baseline)
        _save_candidate_pair_checkpoint(full_paths["candidate_pair_compatibility_mlp"], seed, stage, fit["candidate_pair_compatibility_mlp"])
        _save_oracle_checkpoint(full_paths["explicit_evidence_oracle"], seed, stage)
    return {name: str(path) for name, path in paths.items()}


def _stage6_config(base: Dict[str, object]) -> Dict[str, object]:
    out = json.loads(json.dumps(base))
    out["output_path"] = str(RESULTS_PATH)
    out["audit_log_path"] = str(AUDIT_PATH)
    out["report_path"] = str(REPORT_PATH)
    out["checkpoint_dir"] = str(CHECKPOINT_DIR)
    out["architecture"] = "stage6_complementarity_regularized_candidate_token_direct"
    out["architecture_changes"] = "message_bottlenecks_single_avenue_adversaries_redundancy_and_anti_dominance"
    out["dataset"] = BALANCED_34B_DATASET_SOURCE
    out["dataset_config"] = {
        **dict(out.get("dataset_config", {})),
        "dataset_source": BALANCED_34B_DATASET_SOURCE,
        "generator_version": "real_import_restore_candidate_balanced_34b_stage6_complementarity",
        "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
    }
    out["stage6"] = {
        "cheap": {
            "name": "stage6a_cheap_complementarity_search",
            "n_train": 512,
            "n_dev": 256,
            "n_test": 256,
            "seeds": [0, 1, 2],
            "epochs": 30,
            "patience": 4,
            "hidden_dim": 64,
            "tiny_layers": 3,
            "tiny_ff_dim": 128,
            "batch_size": 32,
            "lr": 0.0003,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "bf16",
        },
        "medium": {
            "name": "stage6b_medium_complementarity_validation",
            "n_train": 1024,
            "n_dev": 512,
            "n_test": 512,
            "seeds": [0, 1, 2, 3, 4],
            "epochs": 40,
            "patience": 5,
            "hidden_dim": 64,
            "tiny_layers": 3,
            "tiny_ff_dim": 128,
            "batch_size": 32,
            "lr": 0.0003,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "bf16",
        },
        "final": {
            "name": "stage6c_final_complementarity_validation",
            "n_train": 2048,
            "n_dev": 512,
            "n_test": 1024,
            "seeds": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            "epochs": 50,
            "patience": 6,
            "hidden_dim": 64,
            "tiny_layers": 3,
            "tiny_ff_dim": 128,
            "batch_size": 32,
            "lr": 0.0003,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "bf16",
        },
    }
    out["stage"] = out["stage6"]["cheap"]
    return out


def _stage_for_phase(config: Dict[str, object], phase: str) -> StageConfig:
    return _stage_from_config({**config, "stage": dict(config["stage6"][phase])})


def _load_or_compute_pre_run(config: Dict[str, object], base_config_path: Path, device: str, recompute: bool) -> Dict[str, object]:
    if not recompute:
        stage4_path = Path("results/stage4_final_candidate_token_direct_validation_results.json")
        if stage4_path.exists():
            stage4 = json.loads(stage4_path.read_text(encoding="utf-8"))
            pre_run = dict(stage4.get("pre_run", {}))
            if pre_run:
                pre_run["source"] = "reused_from_stage4_final_because_dataset_and_controls_are_fixed"
                return pre_run
    pre_config = {**config, "stage": dict(config["stage6"]["final"])}
    return _pre_run_checks(pre_config, config_path=base_config_path, device=device)


def _metadata(config: Dict[str, object], base_config_path: Path, hardware: Dict[str, object], device: str) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "stage": "stage6_complementarity_search",
        "created_at_utc": _now(),
        "base_config_path": str(base_config_path),
        "output_path": str(RESULTS_PATH),
        "audit_log_path": str(AUDIT_PATH),
        "report_path": str(REPORT_PATH),
        "device": device,
        "hardware": hardware,
        "dataset_policy": "Stage 3.8 schema-aware balanced_categories_v3; role-specific schemas preserved.",
        "selection_rule": "Variant selection uses dev complementarity score from cheap and medium phases only; final test seeds are not used for tuning.",
        "frozen_comparator_rule": "Exact same architecture and training budget; only shared agent/model weights are frozen.",
    }


def _write_outputs(result: Dict[str, object]) -> None:
    _attach_pre_run_candidate_metadata(result)
    result["summary"] = _summary(result)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(RESULTS_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(RESULTS_PATH)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_rows = _audit_rows(result)
    AUDIT_PATH.write_text("\n".join(json.dumps(row, sort_keys=True) for row in audit_rows) + ("\n" if audit_rows else ""), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_report(result), encoding="utf-8")


def _attach_pre_run_candidate_metadata(result: Dict[str, object]) -> None:
    seed_values = {}
    for row in result.get("pre_run", {}).get("shortcut_diagnostics", {}).get("seed_rows", []):
        baselines = row.get("baselines", {})
        seed_values[int(row.get("seed", -1))] = float(baselines.get("candidate_metadata_only_accuracy", 1.0))
    fallback = float(
        result.get("pre_run", {})
        .get("shortcut_diagnostics", {})
        .get("summary", {})
        .get("mean_candidate_metadata_only_accuracy", 1.0)
    )
    for rows in result.get("phases", {}).values():
        for row in rows:
            if row.get("status") != "completed":
                continue
            value = seed_values.get(int(row.get("seed", -1)), fallback)
            row.setdefault("test_accuracy", {})["candidate_metadata_only"] = value
            row.setdefault("dev_accuracy", {})["candidate_metadata_only"] = value


def _summary(result: Dict[str, object]) -> Dict[str, object]:
    phase_summaries = {
        phase: _phase_variant_summaries(rows) for phase, rows in dict(result.get("phases", {})).items()
    }
    final_rows = [row for row in result.get("phases", {}).get("final", []) if row.get("status") == "completed"]
    criteria = _final_criteria(final_rows, result.get("config", {}))
    return {
        "phase_summaries": phase_summaries,
        "selected_after_cheap": [variant.name for variant in _selected_variants(result, "cheap", top_k=3)],
        "selected_after_medium": [variant.name for variant in _selected_variants(result, "medium", top_k=1, require_baseline_improvement=True)],
        "final_completed_seeds": [int(row["seed"]) for row in final_rows],
        "n_final_completed": len(final_rows),
        "stage4_baseline": BASELINE_4X1,
        "final_success_criteria": criteria,
        "passed_stage6_final_gates": bool(criteria and all(bool(item["pass"]) for item in criteria)),
    }


def _final_criteria(rows: Sequence[Dict[str, object]], config: Dict[str, object]) -> List[Dict[str, object]]:
    if not rows:
        return [{"criterion": "completed seeds >= 10", "pass": False, "value": 0}]
    near = float(config.get("near_chance_max", 0.18))
    inv_tol = float(config.get("invariance_tolerance", 0.08))
    deltas = [float(row["test_delta"]) for row in rows]
    trainable = [float(row["test_accuracy"]["trainable"]) for row in rows]
    frozen = [float(row["test_accuracy"]["frozen"]) for row in rows]
    best_single = [float(row.get("avenue_audit", {}).get("max_single_avenue_accuracy", 1.0)) for row in rows]
    complementarity_scores = [float(row.get("test_complementarity_score", trainable[index] - best_single[index])) for index, row in enumerate(rows)]
    train_mean = _mean(trainable)
    best_single_mean = _mean(best_single)
    complementarity_mean = _mean(complementarity_scores)
    ci_low, ci_high = _bootstrap_ci(deltas)
    comp_ci_low, comp_ci_high = _bootstrap_ci(complementarity_scores)
    baseline_names = ("text_only", "raw_latent", "majority_baseline", "candidate_order_baseline", "single_agent_full_context")
    baseline_means = {name: _mean([float(row["test_accuracy"].get(name, 0.0)) for row in rows]) for name in baseline_names}
    single_view_means = {
        f"single_view_text_role_{index}": _mean([float(row["test_accuracy"].get(f"single_view_text_role_{index}", 0.0)) for row in rows])
        for index in range(4)
    }
    control_means = {name: _mean([float(row["test_accuracy"].get(name, 1.0)) for row in rows]) for name in PRIMARY_CONTROLS}
    invariance = {name: _mean([float(row["test_accuracy"].get(name, 0.0)) for row in rows]) for name in INVARIANCE_CONTROLS}
    return [
        {"criterion": "completed seeds >= 10", "pass": len(rows) >= 10, "value": len(rows)},
        {"criterion": "trainable beats frozen on at least 8/10 seeds", "pass": sum(delta > 0.0 for delta in deltas) >= 8, "value": sum(delta > 0.0 for delta in deltas)},
        {"criterion": "mean trainable-frozen delta >= +0.20", "pass": _mean(deltas) >= 0.20, "value": _mean(deltas)},
        {"criterion": "bootstrap 95% CI lower bound > +0.05", "pass": ci_low > 0.05, "value": [ci_low, ci_high]},
        {"criterion": "mean full model accuracy >= 0.75", "pass": train_mean >= 0.75, "value": train_mean},
        {"criterion": "combined accuracy - best single avenue accuracy >= 0.20", "pass": complementarity_mean >= 0.20, "value": complementarity_mean},
        {"criterion": "bootstrap 95% CI lower bound for full-minus-best-single > +0.05", "pass": comp_ci_low > 0.05, "value": [comp_ci_low, comp_ci_high]},
        {"criterion": "combined >= 0.80 and best single <= 0.55", "pass": train_mean >= 0.80 and best_single_mean <= 0.55, "value": {"combined": train_mean, "best_single": best_single_mean}},
        {"criterion": "trainable beats text/raw/majority/candidate-order/full-context by mean accuracy", "pass": all(train_mean > value for value in baseline_means.values()), "value": {"trainable": train_mean, **baseline_means}},
        {"criterion": "trainable beats every single-view baseline", "pass": all(train_mean > value for value in single_view_means.values()), "value": single_view_means},
        {"criterion": "all Stage 4 controls pass", "pass": all(value <= near for value in control_means.values()), "value": control_means},
        {"criterion": "physical and candidate order invariance pass", "pass": all(abs(value - train_mean) <= inv_tol for value in invariance.values()), "value": {name: value - train_mean for name, value in invariance.items()}},
        {"criterion": "every final seed has positive full-minus-best-single", "pass": all(value > 0.0 for value in complementarity_scores), "value": complementarity_scores},
        {"criterion": "multi-avenue improves over Stage 4 baseline mean or robustness", "pass": _final_improves_stage4(rows), "value": {"stage4_baseline": BASELINE_4X1, "stage6_mean": train_mean, "stage6_min": min(trainable), "stage6_delta": _mean(deltas), "stage6_std": pstdev(trainable) if len(trainable) > 1 else 0.0}},
        {"criterion": "leakage audits pass", "pass": all(bool(row.get("split_leakage_audit_passes")) and bool(row.get("output_leakage_audit_passes")) for row in rows), "value": "split_and_output"},
        {"criterion": "trainable shared model receives gradients and changes", "pass": all(float(row.get("trainable_audit", {}).get("agent_grad_norm_mean") or 0.0) > 0.0 and float(row.get("trainable_audit", {}).get("agent_parameter_delta") or 0.0) > 0.0 for row in rows), "value": "all_final_rows"},
        {"criterion": "frozen comparator remains frozen", "pass": all(_audit_float(row.get("frozen_audit", {}), "agent_grad_norm_mean", 1.0) == 0.0 and _audit_float(row.get("frozen_audit", {}), "agent_parameter_delta", 1.0) == 0.0 for row in rows), "value": "all_final_rows"},
        {
            "criterion": "checkpoints saved",
            "pass": all(
                all(row.get("checkpoint_paths", {}).get(name) and Path(str(row.get("checkpoint_paths", {}).get(name))).exists() for name in _required_checkpoint_names())
                for row in rows
            ),
            "value": list(_required_checkpoint_names()),
        },
    ]


def _required_checkpoint_names() -> tuple[str, ...]:
    return (
        "trainable",
        "frozen",
        "randomized_labels",
        "raw_latent",
        "text_only",
        "majority_baseline",
        "candidate_order_baseline",
        "single_agent_full_context",
        "single_view_text_role_0",
        "single_view_text_role_1",
        "single_view_text_role_2",
        "single_view_text_role_3",
        "candidate_pair_compatibility_mlp",
        "explicit_evidence_oracle",
    )


def _final_improves_stage4(rows: Sequence[Dict[str, object]]) -> bool:
    trainable = [float(row["test_accuracy"]["trainable"]) for row in rows]
    deltas = [float(row["test_delta"]) for row in rows]
    if not trainable:
        return False
    return bool(
        _mean(trainable) > float(BASELINE_4X1["trainable_mean"])
        or min(trainable) > float(BASELINE_4X1["min_seed_accuracy"])
        or sum(value < 0.60 for value in trainable) < int(BASELINE_4X1["weak_seeds_below_060"])
        or _mean(deltas) > float(BASELINE_4X1["delta"])
        or (len(trainable) > 1 and pstdev(trainable) < float(BASELINE_4X1["std_accuracy"]))
    )


def _audit_rows(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = [{"event": "pre_run_summary", "time_utc": _now(), **dict(result.get("pre_run", {}).get("summary", {}))}]
    for phase, phase_rows in dict(result.get("phases", {})).items():
        for row in phase_rows:
            rows.append(
                {
                    "event": "stage6_row",
                    "phase": phase,
                    "seed": row.get("seed"),
                    "candidate": row.get("candidate"),
                    "status": row.get("status"),
                    "dev_accuracy": row.get("dev_accuracy"),
                    "test_accuracy": row.get("test_accuracy"),
                    "dev_delta": row.get("dev_delta"),
                    "test_delta": row.get("test_delta"),
                    "cuda_max_memory_allocated_gb": row.get("cuda_max_memory_allocated_gb"),
                    "success_criteria": row.get("success_criteria"),
                }
            )
    rows.append({"event": "summary", "time_utc": _now(), **dict(result.get("summary", {}))})
    return rows


def _render_report(result: Dict[str, object]) -> str:
    summary = result.get("summary", {})
    lines = [
        "# Stage 6 Complementarity Search",
        "",
        "## Scope",
        "",
        "- Fixed benchmark: Stage 3.8 schema-aware balanced_categories_v3.",
        "- Dataset labels, oracle metadata separation, leakage controls, schema-aware corruption controls, and frozen comparator policy are unchanged.",
        "- Baseline reference: Stage 4 `candidate_token_direct_lr3e4_clip1`, trainable `0.7892`, frozen `0.1518`, delta `+0.6374`.",
        "- Selection uses `full_model_accuracy - best_single_avenue_accuracy` on dev metrics from cheap and medium phases only.",
        "- Final success is not claimed unless the selected 10-seed run passes the complementarity gates and all Stage 4 controls.",
        "",
        "## Search Space",
        "",
        "| variant | family | d | adv | red | red lambda | coord | dropout | dominance |",
        "|---|---|---:|---:|---|---:|---|---:|---|",
    ]
    for variant in result.get("search_space", []):
        lines.append(
            "| {name} | {family} | {dim} | {adv:.3g} | {red} | {lam:.3g} | {coord} | {dropout:.2f} | {dom} |".format(
                name=variant["name"],
                family=variant["family"],
                dim=int(variant["bottleneck_dim"]),
                adv=float(variant["lambda_adv"]),
                red=variant["redundancy_type"],
                lam=float(variant["lambda_red"]),
                coord=variant["coordinator_style"],
                dropout=float(variant["avenue_dropout"]),
                dom=variant["anti_dominance"],
            )
        )
    lines.extend(["", "## Pre-Run Controls", ""])
    pre = result.get("pre_run", {})
    lines.append(f"- Source: `{pre.get('source', 'computed_for_stage6')}`")
    gates = pre.get("summary", {}).get("gates", {})
    lines.append(f"- Shortcut diagnostics pass: `{bool(pre.get('summary', {}).get('passes', False))}`")
    if gates:
        lines.extend(["", "| gate | pass |", "|---|---|"])
        for name, passed in gates.items():
            lines.append(f"| {name} | `{bool(passed)}` |")
    lines.extend(["", "## Phase Summaries", ""])
    for phase, rows in summary.get("phase_summaries", {}).items():
        lines.extend([f"### {phase.title()}", "", "| variant | n | dev train | best single | comp score | dev frozen | dev delta | controls | invariance | selected-valid |", "|---|---:|---:|---:|---:|---:|---:|---|---|---|"])
        for row in rows:
            lines.append(
                "| {candidate} | {n} | {train:.4f} | {single:.4f} | {comp:.4f} | {frozen:.4f} | {delta:.4f} | `{controls}` | `{inv}` | `{valid}` |".format(
                    candidate=row["candidate"],
                    n=int(row["n_completed"]),
                    train=float(row["mean_dev_trainable"]),
                    single=float(row.get("mean_dev_best_single_avenue", 0.0)),
                    comp=float(row.get("mean_dev_complementarity_score", 0.0)),
                    frozen=float(row["mean_dev_frozen"]),
                    delta=float(row["mean_dev_delta"]),
                    controls=bool(row["controls_near_chance"]),
                    inv=bool(row["invariance_preserved"]),
                    valid=bool(row["selection_valid"]),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Selected Variants",
            "",
            f"- Selected after cheap: `{summary.get('selected_after_cheap', [])}`",
            f"- Selected after medium: `{summary.get('selected_after_medium', [])}`",
            "",
            "## Final Validation",
            "",
        ]
    )
    final_rows = [row for row in result.get("phases", {}).get("final", []) if row.get("status") == "completed"]
    if final_rows:
        lines.extend(["| seed | variant | full | best single | comp score | frozen | delta | physical-order | candidate-order |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"])
        for row in final_rows:
            test = row["test_accuracy"]
            avenue = row.get("avenue_audit", {})
            lines.append(
                f"| {row['seed']} | {row['candidate']} | {float(test['trainable']):.4f} | {float(avenue.get('max_single_avenue_accuracy', 0.0)):.4f} | {float(row.get('test_complementarity_score', 0.0)):.4f} | {float(test['frozen']):.4f} | {float(row['test_delta']):.4f} | {float(test.get('physical_order_shuffled_roles_preserved', 0.0)):.4f} | {float(test.get('candidate_order_shuffled_with_gold_remap', 0.0)):.4f} |"
            )
        lines.extend(["", "## Final Avenue Subset Curve", ""])
        for row in final_rows:
            lines.append(f"### Seed {row['seed']}")
            lines.append("| subset size | mean | max | min |")
            lines.append("|---:|---:|---:|---:|")
            for size, stats in row.get("avenue_subset_curve", {}).items():
                lines.append(f"| {size} | {float(stats.get('mean', 0.0)):.4f} | {float(stats.get('max', 0.0)):.4f} | {float(stats.get('min', 0.0)):.4f} |")
            removal = row.get("avenue_removal_audit", {})
            lines.append(f"- Drop from removing each avenue: `{json.dumps(removal.get('drop_from_removing', {}), sort_keys=True)}`")
    else:
        lines.append("- Final validation has not run; this report does not claim Stage 6 success.")
    lines.extend(["", "## Final Gates", "", "| criterion | pass | value |", "|---|---|---|"])
    for item in summary.get("final_success_criteria", []):
        lines.append(f"| {item['criterion']} | `{bool(item['pass'])}` | `{json.dumps(item['value'], sort_keys=True)}` |")
    failed = [item for item in summary.get("final_success_criteria", []) if not bool(item.get("pass"))]
    lines.extend(["", "## Failure Summary", ""])
    if failed:
        lines.append("- Final Stage 6 success is not claimed.")
        lines.append("- Failed final gates: `" + ", ".join(str(item["criterion"]) for item in failed) + "`.")
        if any("single" in str(item.get("criterion")) or "full-minus-best-single" in str(item.get("criterion")) for item in failed):
            lines.append("- The selected variant did not cleanly prove joint multi-avenue fusion under the requested complementarity gate.")
    else:
        lines.append("- No final gates failed.")
    lines.extend(
        [
            "",
            "## Conservative Interpretation",
            "",
            f"- Stage 6 final gates passed: `{bool(summary.get('passed_stage6_final_gates', False))}`.",
            "- Do not claim Stage 6 success unless the selected final variant passes all complementarity, baseline, control, invariance, and leakage gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
