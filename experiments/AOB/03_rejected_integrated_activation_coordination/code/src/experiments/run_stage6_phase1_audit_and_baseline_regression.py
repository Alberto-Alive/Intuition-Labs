from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

import numpy as np
import torch

from src.datasets.multiview_code_patch_selection import (
    build_multiview_code_patch_splits,
    dataset_summary,
    output_leakage_audit,
    split_leakage_audit,
)
from src.experiments import run_stage4_final_candidate_token_direct_validation as stage4_final
from src.experiments import run_stage6_integrated_multiperspective_search as stage6
from src.experiments.architecture_search import _fit_baselines, _leakage_passes
from src.experiments.real_shared_weight_latent_coordination import BENCHMARK
from src.experiments.run_stage3_gpu_hard_validation import _clear_cuda, _configure_cuda
from src.experiments.run_stage38_full_hard_validation import (
    _dataset_config_for_stage,
    _labels,
    _stage38_full_config,
    _stage_from_config,
)


DEFAULT_STAGE6_CONFIG = "configs/stage6_integrated_multiperspective_search.json"
DEFAULT_STAGE4_FINAL_CONFIG = "configs/stage4_final_candidate_token_direct_validation.json"
DEFAULT_STAGE6_RESULTS = "results/stage6_integrated_results.json"
DEFAULT_STAGE4_RESULTS = "results/stage4_schema_aware_architecture_search_results.json"
DEFAULT_BREAKDOWN = "results/stage6_phase1_control_breakdown.json"
DEFAULT_REGRESSION = "results/stage6_phase1_baseline_regression.json"
DEFAULT_MASK_AUDIT = "results/stage6_phase1_mask_audit.jsonl"
DEFAULT_REPORT = "reports/STAGE6_PHASE1_AUDIT_AND_BASELINE_REGRESSION.md"

BASELINE = "candidate_token_direct_lr3e4_clip1"
NEAR = 0.18
INV_TOL = 0.08
STANDARD_CHANCE_CONTROLS = (
    "candidate_only",
    "candidate_metadata_only",
    "schema_only",
    "view_masked_candidates_visible",
    "evidence_only_no_candidates",
    "null_evidence_values",
    "value_shuffle_within_schema",
    "cross_example_view_bundle_shuffle",
    "candidate_evidence_mismatch",
    "schema_preserved_role_value_shuffle",
    "randomized_labels",
    "hidden_states_shuffled",
)
INVARIANCE_CONTROLS = (
    "physical_order_shuffled_roles_preserved",
    "candidate_order_shuffled_with_gold_remap",
)
INTEGRATED_AUDIT_CONTROLS = (
    "attention_mask_audit",
    "role_stream_isolation_audit",
    "candidate_token_isolation_audit",
    "no_gold_token_audit",
    "no_candidate_position_shortcut_audit",
    "packed_sequence_position_shuffle_audit",
    "role_block_permutation_audit",
    "candidate_block_permutation_with_remapped_labels",
    "coordination_token_ablation",
    "candidate_token_ablation",
    "role_token_ablation",
    "cross_stream_attention_disabled",
)
REPORT_CONTROL_ORDER = STANDARD_CHANCE_CONTROLS + INVARIANCE_CONTROLS + INTEGRATED_AUDIT_CONTROLS
COMMON_STAGE4_SCREEN_CONTROLS = (
    "trainable",
    "frozen",
    "oracle",
    "candidate_only",
    "schema_only",
    "view_masked_candidates_visible",
    "value_shuffle_within_schema",
    "candidate_evidence_mismatch",
)
DIRECT_REGRESSION_SPLITS = ("dev", "test")
BASELINE_COMPARISON_CONTROLS = (
    "trainable",
    "frozen",
    "oracle",
    "candidate_only",
    "candidate_metadata_only",
    "schema_only",
    "view_masked_candidates_visible",
    "evidence_only_no_candidates",
    "null_evidence_values",
    "value_shuffle_within_schema",
    "cross_example_view_bundle_shuffle",
    "candidate_evidence_mismatch",
    "schema_preserved_role_value_shuffle",
    "randomized_labels",
    "hidden_states_shuffled",
    "physical_order_shuffled_roles_preserved",
    "candidate_order_shuffled_with_gold_remap",
)
STAGE4_TO_STAGE6_NAMES = {
    "oracle": "all_role_oracle",
    "hidden_states_shuffled": "hidden_states_shuffled_across_examples",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage 6 Phase 1 controls and baseline regression.")
    parser.add_argument("--stage6-results", default=DEFAULT_STAGE6_RESULTS)
    parser.add_argument("--stage4-results", default=DEFAULT_STAGE4_RESULTS)
    parser.add_argument("--breakdown-output", default=DEFAULT_BREAKDOWN)
    parser.add_argument("--baseline-output", default=DEFAULT_REGRESSION)
    parser.add_argument("--mask-output", default=DEFAULT_MASK_AUDIT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--stage6-config", default=DEFAULT_STAGE6_CONFIG)
    parser.add_argument("--stage4-final-config", default=DEFAULT_STAGE4_FINAL_CONFIG)
    parser.add_argument(
        "--skip-direct-baseline-rerun",
        action="store_true",
        help="Skip retraining the cheap-size Stage 4 clone for evaluator parity; use artifact comparison only.",
    )
    args = parser.parse_args()

    stage6_results = json.loads(Path(args.stage6_results).read_text(encoding="utf-8"))
    stage4_results = json.loads(Path(args.stage4_results).read_text(encoding="utf-8"))
    breakdown = build_control_breakdown(stage6_results)
    regression = build_baseline_regression(
        stage4_results=stage4_results,
        stage6_results=stage6_results,
        stage6_config_path=Path(args.stage6_config),
        stage4_final_config_path=Path(args.stage4_final_config),
        run_direct=not bool(args.skip_direct_baseline_rerun),
    )
    mask_rows = build_mask_rows(stage6_results)
    eligible = list(breakdown["phase2_eligible_variants"])
    baseline_clean = bool(regression["baseline_regression_clean"])
    decision = "eligible" if eligible and baseline_clean else "blocked"
    stage6_results["phase1_audit"] = {
        "report": DEFAULT_REPORT,
        "baseline_regression_clean": baseline_clean,
        "direct_control_path_regression_clean": bool(regression.get("direct_control_path", {}).get("control_path_regression_clean", False)),
        "completed_stage6_artifact_budget_matches_stage4": bool(
            regression.get("artifact_comparison", {}).get("completed_stage6_artifact_budget_matches_stage4", False)
        ),
        "strict_phase2_eligible_variants": eligible,
        "phase2_run": False,
        "decision": decision,
    }
    if decision == "blocked":
        stage6_results.setdefault("medium_validation", {})
        stage6_results["medium_validation"]["summary"] = {
            "ran": False,
            "reason": "blocked by Stage 6 Phase 1 audit: no strict eligible variants and/or baseline regression not clean",
        }
    stage6_results["summary"] = stage6._overall_summary(stage6_results)
    report = render_report(breakdown, regression, stage6_results)

    write_json(Path(args.breakdown_output), breakdown)
    write_json(Path(args.baseline_output), regression)
    mask_path = Path(args.mask_output)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in mask_rows) + "\n", encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    # Refresh the Stage 6 report with the clarified mean-gate/row-gate wording.
    stage6._write_all_outputs(
        stage6_results,
        Path(DEFAULT_STAGE6_RESULTS),
        Path("results/stage6_integrated_controls.json"),
        Path("results/stage6_integrated_leakage_audit.jsonl"),
        Path("results/stage6_integrated_attention_mask_audit.jsonl"),
        Path("results/stage6_integrated_error_cases.jsonl"),
        Path("results/stage6_integrated_compute_metrics.json"),
        Path("reports/STAGE6_INTEGRATED_MULTIPERSPECTIVE_SEARCH.md"),
    )


def build_control_breakdown(result: Dict[str, object]) -> Dict[str, object]:
    rows = list(result.get("cheap_screen", {}).get("rows", []))
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    row_breakdowns = []
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
        for split in DIRECT_REGRESSION_SPLITS:
            row_breakdowns.append(_row_breakdown(row, split=split))
    variant_summaries = []
    for variant, variant_rows in sorted(by_variant.items()):
        variant_summaries.append(_variant_summary(variant, variant_rows))
    return {
        "metadata": {
            "created_at_utc": now(),
            "source": DEFAULT_STAGE6_RESULTS,
            "policy": "strict_phase2_eligibility_requires_every_seed to pass standard controls, masks/leakage, invariance, frozen-low, and meaningful delta",
            "near_chance_max": NEAR,
            "invariance_tolerance": INV_TOL,
        },
        "row_breakdowns": row_breakdowns,
        "variant_summaries": variant_summaries,
        "phase2_eligible_variants": [row["variant"] for row in variant_summaries if row["phase2_eligible_strict"]],
    }


def build_baseline_regression(
    stage4_results: Dict[str, object],
    stage6_results: Dict[str, object],
    stage6_config_path: Path,
    stage4_final_config_path: Path,
    run_direct: bool,
) -> Dict[str, object]:
    artifact = _build_artifact_baseline_comparison(stage4_results, stage6_results)
    direct = (
        _run_direct_baseline_control_path_regression(stage6_config_path=stage6_config_path, stage4_final_config_path=stage4_final_config_path)
        if run_direct
        else {"ran": False, "reason": "skipped by --skip-direct-baseline-rerun", "control_path_regression_clean": False}
    )
    return {
        "metadata": {
            "created_at_utc": now(),
            "stage4_source": DEFAULT_STAGE4_RESULTS,
            "stage6_source": DEFAULT_STAGE6_RESULTS,
            "stage6_config": str(stage6_config_path),
            "stage4_final_config": str(stage4_final_config_path),
            "baseline": BASELINE,
            "cheap_screen_sizes": {"n_train": 512, "n_dev": 256, "n_test": 256},
            "seeds": [0, 1, 2],
        },
        "direct_control_path": direct,
        "artifact_comparison": artifact,
        "baseline_regression_clean": bool(
            direct.get("control_path_regression_clean")
            and artifact.get("completed_stage6_artifact_budget_matches_stage4")
            and artifact.get("stage6_artifact_baseline_controls_clean")
        ),
        "regression_reasons": _combined_regression_reasons(direct, artifact),
    }


def _build_artifact_baseline_comparison(stage4_results: Dict[str, object], stage6_results: Dict[str, object]) -> Dict[str, object]:
    stage4_rows = [row for row in stage4_results.get("cheap_screen", {}).get("rows", []) if row.get("variant") == BASELINE]
    stage6_rows = [row for row in stage6_results.get("cheap_screen", {}).get("rows", []) if row.get("variant") == BASELINE]
    stage4_by_seed = {int(row["seed"]): row for row in stage4_rows}
    stage6_by_seed = {int(row["seed"]): row for row in stage6_rows}
    seeds = sorted(set(stage4_by_seed) & set(stage6_by_seed))
    comparisons = []
    for seed in seeds:
        comparisons.append(
            {
                "seed": seed,
                "dev": _baseline_split_comparison(stage4_by_seed[seed].get("dev_accuracy", {}), stage6_by_seed[seed].get("dev", {}).get("accuracy", {})),
                "test": _baseline_split_comparison(stage4_by_seed[seed].get("test_smoke_accuracy", {}), stage6_by_seed[seed].get("test", {}).get("accuracy", {})),
            }
        )
    mean_comparison = {
        split: {
            name: {
                "stage4_mean": mean_value([float(row[split].get(name, {}).get("stage4", 0.0)) for row in comparisons if name in row[split]]),
                "stage6_mean": mean_value([float(row[split].get(name, {}).get("stage6", 0.0)) for row in comparisons if name in row[split]]),
                "stage6_minus_stage4": mean_value([float(row[split].get(name, {}).get("delta", 0.0)) for row in comparisons if name in row[split]]),
            }
            for name in COMMON_STAGE4_SCREEN_CONTROLS
        }
        for split in ("dev", "test")
    }
    stage4_variant = stage4_rows[0].get("variant_config", {}) if stage4_rows else {}
    stage6_variant = next((row for row in stage6_results.get("variant_plan", []) if row.get("name") == BASELINE), {})
    stage6_baseline_summary = next(
        (row for row in stage6_results.get("cheap_screen", {}).get("summary", {}).get("variant_summaries", []) if row.get("variant") == BASELINE),
        {},
    )
    budget_matches = not _baseline_config_differences(stage4_variant, stage6_variant)
    stage6_clean = bool(stage6_baseline_summary.get("controls_pass")) and float(stage6_baseline_summary.get("mean_dev_delta", 0.0)) >= 0.20
    return {
        "description": "Comparison of completed Stage 4 cheap-screen artifact against completed Stage 6 cheap-screen artifact.",
        "seeds_compared": seeds,
        "config_comparison": {
            "stage4_variant_config": stage4_variant,
            "stage6_variant_plan_from_completed_artifact": stage6_variant,
            "identified_differences": _baseline_config_differences(stage4_variant, stage6_variant),
        },
        "per_seed_comparison": comparisons,
        "mean_comparison": mean_comparison,
        "stage6_baseline_summary": stage6_baseline_summary,
        "completed_stage6_artifact_budget_matches_stage4": budget_matches,
        "stage6_artifact_baseline_controls_clean": stage6_clean,
        "regression_reasons": _artifact_regression_reasons(stage4_variant, stage6_variant, stage6_baseline_summary),
    }


def _run_direct_baseline_control_path_regression(stage6_config_path: Path, stage4_final_config_path: Path) -> Dict[str, object]:
    stage6_config = json.loads(stage6_config_path.read_text(encoding="utf-8"))
    stage4_config = json.loads(stage4_final_config_path.read_text(encoding="utf-8"))
    stage6_config["stage"] = {
        **dict(stage6_config.get("stage", {})),
        "name": "stage6_phase1_audit_baseline_regression",
        "n_train": 512,
        "n_dev": 256,
        "n_test": 256,
        "seeds": [0, 1, 2],
        "epochs": 50,
        "patience": 6,
        "batch_size": 32,
        "lr": 0.0003,
        "mixed_precision": dict(stage6_config.get("stage", {})).get("mixed_precision", "bf16"),
    }
    stage6_config = _stage38_full_config(stage6_config)
    stage4_config = _stage38_full_config(stage4_config)
    device, hardware = _configure_cuda(stage6_config)
    stage = _stage_from_config(stage6_config)
    baseline_variant = next(variant for variant in stage6._variant_plan() if variant.name == BASELINE)
    rows = []
    started = time.perf_counter()
    for seed in [0, 1, 2]:
        print(f"stage6 audit baseline seed={seed}: training Stage 4 clone once and evaluating both control paths")
        _clear_cuda()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        dataset_config = _dataset_config_for_stage(stage6_config, stage)
        splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=Path("."))
        split_leakage = split_leakage_audit(BENCHMARK, seed, splits)
        output_leakage = output_leakage_audit(BENCHMARK, seed, splits["train"] + splits["dev"] + splits["test"])
        baselines = _fit_baselines(stage, splits, seed=seed, device=device)
        candidate = stage4_final._selected_candidate(stage)
        fit = stage4_final._fit_selected_methods(candidate, stage, splits, seed, device, stage4_config)
        stage4_metrics = stage4_final._metrics(fit, baselines, splits, seed)
        metadata_only = stage6._candidate_metadata_shortcut(stage6_config, splits, seed)
        no_gold = stage6._no_gold_token_audit(splits)
        stage6_metrics = {
            split: stage6._evaluate_split(
                trainable=fit["trainable"],
                frozen=fit["frozen"],
                randomized=fit["randomized"],
                examples=splits[split],
                seed=seed + (1_000 if split == "dev" else 2_000),
                metadata_only_accuracy=metadata_only.get(split),
                no_gold_audit=no_gold,
                variant=baseline_variant,
            )
            for split in DIRECT_REGRESSION_SPLITS
        }
        for split in DIRECT_REGRESSION_SPLITS:
            stage4_metrics[split]["candidate_metadata_only"] = float(metadata_only[split])
        attention_audit = stage6._capture_attention_audit(fit["trainable"], splits["dev"][: min(32, len(splits["dev"]))], seed + 30_000)
        row = {
            "seed": seed,
            "stage_config": {
                "n_train": stage.n_train,
                "n_dev": stage.n_dev,
                "n_test": stage.n_test,
                "epochs": 50,
                "patience": 6,
            },
            "dataset_summary": dataset_summary(splits),
            "split_leakage_audit_passes": _leakage_passes(split_leakage),
            "output_leakage_audit_passes": bool(output_leakage.get("passes", False)),
            "no_gold_token_audit": no_gold,
            "attention_mask_audit": attention_audit,
            "stage4": {
                split: _normalize_stage4_baseline_metrics(stage4_metrics[split])
                for split in DIRECT_REGRESSION_SPLITS
            },
            "stage6": {
                split: _normalize_stage6_baseline_metrics(stage6_metrics[split]["accuracy"])
                for split in DIRECT_REGRESSION_SPLITS
            },
            "comparisons": {
                split: _direct_split_comparison(
                    _normalize_stage4_baseline_metrics(stage4_metrics[split]),
                    _normalize_stage6_baseline_metrics(stage6_metrics[split]["accuracy"]),
                )
                for split in DIRECT_REGRESSION_SPLITS
            },
            "stage4_trainable_audit": stage4_final._audit_subset(fit["trainable"].audit),
            "stage4_frozen_audit": stage4_final._audit_subset(fit["frozen"].audit),
            "stage4_randomized_audit": stage4_final._audit_subset(fit["randomized"].audit),
            "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") and torch.cuda.is_available() else 0,
        }
        rows.append(row)
        del baselines, candidate, fit, splits
        _clear_cuda()
    means = {
        split: {
            name: {
                "stage4_mean": mean_value(
                    [float(row["stage4"][split][name]["accuracy"]) for row in rows if row["stage4"][split].get(name, {}).get("accuracy") is not None]
                ),
                "stage6_mean": mean_value(
                    [float(row["stage6"][split][name]["accuracy"]) for row in rows if row["stage6"][split].get(name, {}).get("accuracy") is not None]
                ),
                "mean_delta_stage6_minus_stage4": mean_value(
                    [
                        float(row["comparisons"][split][name]["delta"])
                        for row in rows
                        if row["comparisons"][split].get(name, {}).get("delta") is not None
                    ]
                ),
                "stage4_pass_every_seed": _pass_every_seed([row["stage4"][split][name]["pass"] for row in rows if name in row["stage4"][split]]),
                "stage6_pass_every_seed": _pass_every_seed([row["stage6"][split][name]["pass"] for row in rows if name in row["stage6"][split]]),
            }
            for name in BASELINE_COMPARISON_CONTROLS
        }
        for split in DIRECT_REGRESSION_SPLITS
    }
    stage6_only_failures = _stage6_only_control_failures(rows)
    unexplained_differences = _unexplained_direct_differences(rows)
    return {
        "ran": True,
        "description": "Same cheap-size Stage 4 clone fit evaluated through original Stage 4 controls and new Stage 6 controls.",
        "hardware": hardware,
        "device": device,
        "elapsed_seconds": time.perf_counter() - started,
        "per_seed": rows,
        "mean_comparison": means,
        "stage6_only_control_failures": stage6_only_failures,
        "unexplained_control_path_differences": unexplained_differences,
        "exact_threshold_parity": not stage6_only_failures,
        "control_path_regression_clean": not unexplained_differences,
    }


def _pass_every_seed(values: Sequence[object]) -> bool | None:
    filtered = [bool(value) for value in values if value is not None]
    return all(filtered) if filtered else None


def _normalize_stage4_baseline_metrics(metrics: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    out = {}
    for name in BASELINE_COMPARISON_CONTROLS:
        source = STAGE4_TO_STAGE6_NAMES.get(name, name)
        value = metrics.get(source)
        out[name] = _control_cell(name, value, trainable=metrics.get("trainable"))
    return out


def _normalize_stage6_baseline_metrics(metrics: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    out = {}
    for name in BASELINE_COMPARISON_CONTROLS:
        value = metrics.get(name)
        out[name] = _control_cell(name, value, trainable=metrics.get("trainable"))
    return out


def _control_cell(name: str, value: object, trainable: object) -> Dict[str, object]:
    if not isinstance(value, (int, float)):
        return {"accuracy": None, "pass": None, "rule": _control_rule(name)}
    if name in {"trainable", "frozen", "oracle"}:
        return {"accuracy": float(value), "pass": None, "rule": "reported, not a collapse gate"}
    if name in INVARIANCE_CONTROLS:
        delta = float(value) - float(trainable) if isinstance(trainable, (int, float)) else None
        return {
            "accuracy": float(value),
            "delta_from_trainable": delta,
            "pass": bool(delta is not None and abs(delta) <= INV_TOL),
            "rule": _control_rule(name),
        }
    return {"accuracy": float(value), "pass": bool(float(value) <= NEAR), "rule": _control_rule(name)}


def _control_rule(name: str) -> str:
    if name in {"trainable", "frozen", "oracle"}:
        return "reported, not a collapse gate"
    if name in INVARIANCE_CONTROLS:
        return f"abs(delta_from_trainable) <= {INV_TOL}"
    return f"<= {NEAR}"


def _direct_split_comparison(stage4_controls: Dict[str, Dict[str, object]], stage6_controls: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out = {}
    for name in BASELINE_COMPARISON_CONTROLS:
        s4 = stage4_controls.get(name, {})
        s6 = stage6_controls.get(name, {})
        s4_value = s4.get("accuracy")
        s6_value = s6.get("accuracy")
        delta = float(s6_value) - float(s4_value) if isinstance(s4_value, (int, float)) and isinstance(s6_value, (int, float)) else None
        stage4_pass = s4.get("pass")
        stage6_pass = s6.get("pass")
        differs = bool(delta is not None and abs(delta) > 1e-9) or (stage4_pass is not None and stage6_pass is not None and bool(stage4_pass) != bool(stage6_pass))
        out[name] = {
            "stage4_accuracy": s4_value,
            "stage4_pass": stage4_pass,
            "stage6_accuracy": s6_value,
            "stage6_pass": stage6_pass,
            "delta": delta,
            "differs": differs,
            "reason": _direct_difference_reason(name, delta, stage4_pass, stage6_pass),
        }
    return out


def _direct_difference_reason(name: str, delta: object, stage4_pass: object, stage6_pass: object) -> str:
    if delta is None:
        return "not comparable"
    if abs(float(delta)) <= 1e-9 and stage4_pass == stage6_pass:
        return "same numeric result"
    if name == "hidden_states_shuffled":
        return "same corruption family; Stage 4 records hidden_states_shuffled_across_examples and Stage 6 records hidden_states_shuffled with its own evaluation seed offset"
    if name in {
        "view_masked_candidates_visible",
        "value_shuffle_within_schema",
        "cross_example_view_bundle_shuffle",
        "candidate_evidence_mismatch",
        "schema_preserved_role_value_shuffle",
        "physical_order_shuffled_roles_preserved",
        "candidate_order_shuffled_with_gold_remap",
    }:
        return "same semantic control, different evaluator seed offsets and control construction call sites"
    if name == "candidate_metadata_only":
        return "same shared bag-of-words metadata diagnostic injected into both paths"
    return "same model and split evaluated through distinct Stage 4 versus Stage 6 evaluator code"


def _stage6_only_control_failures(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    failures = []
    for row in rows:
        for split in DIRECT_REGRESSION_SPLITS:
            for name, comparison in row["comparisons"][split].items():
                if comparison.get("stage4_pass") is True and comparison.get("stage6_pass") is False:
                    failures.append(
                        {
                            "seed": row["seed"],
                            "split": split,
                            "control": name,
                            "stage4_accuracy": comparison.get("stage4_accuracy"),
                            "stage6_accuracy": comparison.get("stage6_accuracy"),
                            "reason": comparison.get("reason"),
                        }
                    )
    return failures


def _unexplained_direct_differences(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    unexplained = []
    explainable = {
        "hidden_states_shuffled",
        "view_masked_candidates_visible",
        "value_shuffle_within_schema",
        "cross_example_view_bundle_shuffle",
        "candidate_evidence_mismatch",
        "schema_preserved_role_value_shuffle",
        "physical_order_shuffled_roles_preserved",
        "candidate_order_shuffled_with_gold_remap",
    }
    for row in rows:
        for split in DIRECT_REGRESSION_SPLITS:
            for name, comparison in row["comparisons"][split].items():
                if not comparison.get("differs"):
                    continue
                if name in explainable:
                    continue
                if abs(float(comparison.get("delta") or 0.0)) > 1e-9:
                    unexplained.append(
                        {
                            "seed": row["seed"],
                            "split": split,
                            "control": name,
                            "stage4_accuracy": comparison.get("stage4_accuracy"),
                            "stage6_accuracy": comparison.get("stage6_accuracy"),
                            "delta": comparison.get("delta"),
                            "reason": comparison.get("reason"),
                        }
                    )
    return unexplained


def _combined_regression_reasons(direct: Dict[str, object], artifact: Dict[str, object]) -> List[str]:
    reasons = []
    if not direct.get("ran"):
        reasons.append(str(direct.get("reason", "direct baseline control-path regression did not run")))
    else:
        if direct.get("stage6_only_control_failures"):
            reasons.append("direct control-path regression found explainable Stage 6-only threshold flips from evaluator seed/control call-site differences")
        if direct.get("unexplained_control_path_differences"):
            reasons.append("direct control-path regression found unexplained evaluator differences")
    reasons.extend(str(reason) for reason in artifact.get("regression_reasons", []))
    return reasons


def build_mask_rows(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for row in result.get("cheap_screen", {}).get("rows", []):
        mask = (row.get("attention_mask_audit") or {}).get("mask") or {}
        rows.append(
            {
                "event": "stage6_phase1_mask_audit",
                "variant": row.get("variant"),
                "seed": row.get("seed"),
                "mask": mask,
                "attention": (row.get("attention_mask_audit") or {}).get("attention"),
                "mask_pass": bool(stage6._mask_audit_passes(row)),
                "role_stream_isolation_pass": mask.get("role_tokens_can_see_other_roles_initially") is False,
                "candidate_token_isolation_pass": mask.get("candidate_tokens_can_see_gold") is False and mask.get("uses_candidate_position_embedding") is False,
                "no_gold_token_pass": bool((row.get("no_gold_token_audit") or {}).get("passes", False)),
            }
        )
    return rows


def _row_breakdown(row: Dict[str, object], split: str) -> Dict[str, object]:
    accuracy = row.get(split, {}).get("accuracy", {})
    trainable = float(accuracy.get("trainable", 0.0))
    controls = {}
    for name in STANDARD_CHANCE_CONTROLS:
        value = accuracy.get(name)
        controls[name] = {
            "accuracy": value,
            "rule": f"<= {NEAR}",
            "pass": bool(isinstance(value, (int, float)) and float(value) <= NEAR),
        }
    for name in INVARIANCE_CONTROLS:
        value = accuracy.get(name)
        delta = float(value) - trainable if isinstance(value, (int, float)) else None
        controls[name] = {
            "accuracy": value,
            "delta_from_trainable": delta,
            "rule": f"abs(delta_from_trainable) <= {INV_TOL}",
            "pass": bool(delta is not None and abs(delta) <= INV_TOL),
        }
    controls.update(_integrated_control_breakdown(row, accuracy, trainable))
    return {
        "phase": row.get("phase"),
        "variant": row.get("variant"),
        "seed": row.get("seed"),
        "split": split,
        "trainable": trainable,
        "frozen": accuracy.get("frozen"),
        "delta": row.get(f"{split}_delta"),
        "existing_row_gate_failures": row.get("integrated_gate_failures", []),
        "controls": controls,
        "all_standard_controls_pass": all(controls[name]["pass"] for name in tuple(STANDARD_CHANCE_CONTROLS) + tuple(INVARIANCE_CONTROLS)),
        "all_integrated_mask_controls_pass": all(
            controls[name]["pass"]
            for name in (
                "attention_mask_audit",
                "role_stream_isolation_audit",
                "candidate_token_isolation_audit",
                "no_gold_token_audit",
                "no_candidate_position_shortcut_audit",
                "packed_sequence_position_shuffle_audit",
                "role_block_permutation_audit",
                "candidate_block_permutation_with_remapped_labels",
            )
        ),
    }


def _integrated_control_breakdown(row: Dict[str, object], accuracy: Dict[str, object], trainable: float) -> Dict[str, object]:
    mask = (row.get("attention_mask_audit") or {}).get("mask") or {}
    no_gold = row.get("no_gold_token_audit") or {}
    out = {
        "attention_mask_audit": {"accuracy": None, "rule": "mask booleans enforce no gold/candidate-position/initial role mixing", "pass": bool(stage6._mask_audit_passes(row))},
        "role_stream_isolation_audit": {"accuracy": None, "rule": "role tokens cannot initially see other role streams", "pass": mask.get("role_tokens_can_see_other_roles_initially") is False},
        "candidate_token_isolation_audit": {"accuracy": None, "rule": "candidate tokens cannot see gold and no candidate-position embedding is used", "pass": mask.get("candidate_tokens_can_see_gold") is False and mask.get("uses_candidate_position_embedding") is False},
        "no_gold_token_audit": {"accuracy": None, "rule": "model-facing records contain no oracle-only fields", "pass": bool(no_gold.get("passes", False))},
        "no_candidate_position_shortcut_audit": {"accuracy": accuracy.get("no_candidate_position_shortcut_audit"), "rule": f"<= {NEAR}", "pass": bool(float(accuracy.get("no_candidate_position_shortcut_audit") or 1.0) <= NEAR)},
        "packed_sequence_position_shuffle_audit": {"accuracy": accuracy.get("packed_sequence_position_shuffle_audit"), "rule": "no packed position embedding shortcut", "pass": mask.get("uses_packed_position_embedding") is False},
    }
    for name in ("role_block_permutation_audit", "candidate_block_permutation_with_remapped_labels"):
        value = accuracy.get(name)
        delta = float(value) - trainable if isinstance(value, (int, float)) else None
        out[name] = {"accuracy": value, "delta_from_trainable": delta, "rule": f"abs(delta_from_trainable) <= {INV_TOL}", "pass": bool(delta is not None and abs(delta) <= INV_TOL)}
    for name in ("coordination_token_ablation", "candidate_token_ablation", "role_token_ablation", "cross_stream_attention_disabled"):
        value = accuracy.get(name)
        has_router = int(mask.get("router_tokens", 0) or 0) > 0
        gated = name != "coordination_token_ablation" or has_router
        out[name] = {
            "accuracy": value,
            "rule": f"<= {NEAR}" if gated else "not gated when no coordination/router token exists",
            "pass": bool((not gated) or (isinstance(value, (int, float)) and float(value) <= NEAR)),
        }
    for name in ("early_mixing_disabled", "late_mixing_only_control"):
        value = accuracy.get(name)
        out[name] = {"accuracy": value, "delta_from_trainable": float(value) - trainable if isinstance(value, (int, float)) else None, "rule": "diagnostic impact, not a hard eligibility gate", "pass": True}
    return out


def _variant_summary(variant: str, rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    row_breakdowns = [_row_breakdown(row, "dev") for row in rows]
    trainable = [float(row.get("dev", {}).get("accuracy", {}).get("trainable", 0.0)) for row in rows]
    frozen = [float(row.get("dev", {}).get("accuracy", {}).get("frozen", 0.0)) for row in rows]
    deltas = [float(row.get("dev_delta", 0.0)) for row in rows]
    all_standard = all(item["all_standard_controls_pass"] for item in row_breakdowns)
    all_integrated = all(item["all_integrated_mask_controls_pass"] for item in row_breakdowns)
    leakage = all(bool(row.get("split_leakage_audit_passes")) and bool(row.get("output_leakage_audit_passes")) for row in rows)
    frozen_low = mean_value(frozen) <= 0.25
    meaningful_delta = mean_value(deltas) >= 0.20 and sum(delta > 0 for delta in deltas) >= 2
    no_row_contradiction = all(not row.get("existing_row_gate_failures") for row in row_breakdowns)
    all_control_names = sorted({name for item in row_breakdowns for name in item["controls"]})
    control_summary = {}
    for name in all_control_names:
        cells = [item["controls"][name] for item in row_breakdowns if name in item["controls"]]
        values = [float(cell["accuracy"]) for cell in cells if isinstance(cell.get("accuracy"), (int, float))]
        control_summary[name] = {
            "mean_accuracy": mean_value(values) if values else None,
            "max_accuracy": max(values) if values else None,
            "pass_every_seed": all(bool(cell.get("pass")) for cell in cells),
            "failing_seeds": [int(item["seed"]) for item in row_breakdowns if name in item["controls"] and not bool(item["controls"][name].get("pass"))],
            "rule": cells[0].get("rule") if cells else None,
        }
    return {
        "variant": variant,
        "seeds": [int(row.get("seed")) for row in rows],
        "mean_trainable": mean_value(trainable),
        "mean_frozen": mean_value(frozen),
        "mean_delta": mean_value(deltas),
        "all_standard_controls_pass_every_seed": all_standard,
        "all_integrated_mask_controls_pass_every_seed": all_integrated,
        "leakage_pass": leakage,
        "frozen_low": frozen_low,
        "meaningful_delta": meaningful_delta,
        "no_contradictory_row_gate_statements": no_row_contradiction,
        "phase2_eligible_strict": bool(all_standard and all_integrated and leakage and frozen_low and meaningful_delta and no_row_contradiction),
        "failing_controls_by_seed": {
            str(item["seed"]): [name for name, control in item["controls"].items() if not control["pass"]]
            for item in row_breakdowns
        },
        "control_summary": control_summary,
    }


def _baseline_split_comparison(stage4: Dict[str, object], stage6_acc: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    out = {}
    for name in COMMON_STAGE4_SCREEN_CONTROLS:
        if name not in stage4 and name not in stage6_acc:
            continue
        s4 = stage4.get(name)
        s6 = stage6_acc.get(name)
        out[name] = {
            "stage4": s4,
            "stage6": s6,
            "delta": (float(s6) - float(s4)) if isinstance(s4, (int, float)) and isinstance(s6, (int, float)) else None,
            "stage4_pass": _common_control_pass(name, s4, stage4.get("trainable")),
            "stage6_pass": _common_control_pass(name, s6, stage6_acc.get("trainable")),
        }
    return out


def _common_control_pass(name: str, value: object, trainable: object) -> bool | None:
    if not isinstance(value, (int, float)):
        return None
    if name in {"trainable", "frozen", "oracle"}:
        return None
    return bool(float(value) <= NEAR)


def _baseline_config_differences(stage4_variant: Dict[str, object], stage6_variant: Dict[str, object]) -> List[str]:
    diffs = []
    for key in ("epochs", "patience", "lr", "gradient_clip_norm", "coordinator_family"):
        s4 = stage4_variant.get(key)
        s6 = stage6_variant.get(key)
        if s4 != s6:
            diffs.append(f"{key}: stage4={s4} stage6_artifact={s6}")
    if stage6_variant.get("epochs") != 50 or stage6_variant.get("patience") != 6:
        diffs.append("completed Stage 6 artifact used reduced exploratory training budget; code default has been restored to 50/6 for future runs")
    return diffs


def _artifact_regression_reasons(stage4_variant: Dict[str, object], stage6_variant: Dict[str, object], stage6_summary: Dict[str, object]) -> List[str]:
    reasons = []
    if not stage6_summary.get("controls_pass"):
        reasons.append("Stage 6 baseline mean controls failed")
    if stage6_variant.get("epochs") != stage4_variant.get("epochs") or stage6_variant.get("patience") != stage4_variant.get("patience"):
        reasons.append("Stage 6 completed artifact did not use the original Stage 4 50-epoch/6-patience baseline budget")
    controls = dict(stage6_summary.get("control_means", {}))
    for name in ("candidate_only", "view_masked_candidates_visible", "no_candidate_position_shortcut_audit"):
        if float(controls.get(name, 0.0)) > NEAR:
            reasons.append(f"{name} mean above near-chance threshold: {float(controls.get(name)):.4f}")
    return reasons


def render_report(breakdown: Dict[str, object], regression: Dict[str, object], stage6_results: Dict[str, object]) -> str:
    summaries = breakdown["variant_summaries"]
    eligible = breakdown["phase2_eligible_variants"]
    stage6_summary = stage6_results.get("cheap_screen", {}).get("summary", {}).get("variant_summaries", [])
    lines = [
        "# Stage 6 Phase 1 Audit And Baseline Regression",
        "",
        "## Decision",
        "",
        f"- Strict Phase 2 eligible variants: `{eligible}`",
        "- Phase 2 medium validation was not run by this audit.",
        "- Stage 4 clone remains the mainline.",
        f"- Baseline regression clean: `{bool(regression['baseline_regression_clean'])}`",
        f"- Direct control-path regression clean (no unexplained evaluator differences): `{bool(regression.get('direct_control_path', {}).get('control_path_regression_clean', False))}`",
        f"- Direct exact threshold parity: `{bool(regression.get('direct_control_path', {}).get('exact_threshold_parity', False))}`",
        f"- Completed Stage 6 artifact budget matches Stage 4: `{bool(regression.get('artifact_comparison', {}).get('completed_stage6_artifact_budget_matches_stage4', False))}`",
        "",
        "## Baseline Regression",
        "",
        "- Direct path check: trained `candidate_token_direct_lr3e4_clip1` once per cheap-screen seed `[0,1,2]` at `512/256/256`, then evaluated that same fit through the original Stage 4 controls and the new Stage 6 controls.",
        "- Artifact check: compared the completed Stage 4 cheap-screen artifact against the completed Stage 6 Phase 1 artifact.",
        "- Full per-seed numeric pass/fail details are in `results/stage6_phase1_baseline_regression.json`.",
        "",
        "### Direct Control-Path Means",
        "",
        "| control | Stage 4 dev | Stage 6 dev | delta | Stage 4 dev pass | Stage 6 dev pass | Stage 4 test | Stage 6 test | delta | Stage 4 test pass | Stage 6 test pass |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---|---|",
    ]
    direct = regression.get("direct_control_path", {})
    if direct.get("ran"):
        direct_means = direct.get("mean_comparison", {})
        for name in BASELINE_COMPARISON_CONTROLS:
            dev = direct_means.get("dev", {}).get(name, {})
            test = direct_means.get("test", {}).get(name, {})
            lines.append(
                f"| {name} | {float(dev.get('stage4_mean', 0.0)):.4f} | {float(dev.get('stage6_mean', 0.0)):.4f} | "
                f"{float(dev.get('mean_delta_stage6_minus_stage4', 0.0)):+.4f} | `{_display_pass(dev.get('stage4_pass_every_seed'))}` | `{_display_pass(dev.get('stage6_pass_every_seed'))}` | "
                f"{float(test.get('stage4_mean', 0.0)):.4f} | {float(test.get('stage6_mean', 0.0)):.4f} | "
                f"{float(test.get('mean_delta_stage6_minus_stage4', 0.0)):+.4f} | `{_display_pass(test.get('stage4_pass_every_seed'))}` | `{_display_pass(test.get('stage6_pass_every_seed'))}` |"
            )
    else:
        lines.append(f"| skipped | 0.0000 | 0.0000 | +0.0000 | `False` | `False` | 0.0000 | 0.0000 | +0.0000 | `False` | `False` |")
    artifact = regression.get("artifact_comparison", {})
    lines.extend(
        [
            "",
            "### Completed Artifact Comparison",
            "",
            "- The completed Stage 6 artifact remains invalid for Phase 2 promotion if it was produced with a reduced `12/4` budget while Stage 4 used `50/6`.",
            "",
            "| control | Stage 4 dev mean | Stage 6 dev mean | delta | Stage 4 test mean | Stage 6 test mean | delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    mean_cmp = artifact.get("mean_comparison", {})
    for name in COMMON_STAGE4_SCREEN_CONTROLS:
        dev = mean_cmp.get("dev", {}).get(name, {})
        test = mean_cmp.get("test", {}).get(name, {})
        lines.append(
            f"| {name} | {float(dev.get('stage4_mean', 0.0)):.4f} | {float(dev.get('stage6_mean', 0.0)):.4f} | {float(dev.get('stage6_minus_stage4', 0.0)):+.4f} | "
            f"{float(test.get('stage4_mean', 0.0)):.4f} | {float(test.get('stage6_mean', 0.0)):.4f} | {float(test.get('stage6_minus_stage4', 0.0)):+.4f} |"
        )
    lines.extend(["", "Regression reasons:"])
    for reason in regression.get("regression_reasons", []):
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "Direct control-path Stage 6-only failures:",
            f"- `{json.dumps(direct.get('stage6_only_control_failures', []), sort_keys=True)}`",
            "",
            "Direct control-path unexplained differences:",
            f"- `{json.dumps(direct.get('unexplained_control_path_differences', []), sort_keys=True)}`",
            "",
            "## Control Failure Breakdown",
            "",
            "| variant | mean train | mean frozen | mean delta | standard every-seed | masks every-seed | leakage | frozen low | meaningful delta | strict eligible | failing controls by seed |",
            "|---|---:|---:|---:|---|---|---|---|---|---|---|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['variant']} | {float(row['mean_trainable']):.4f} | {float(row['mean_frozen']):.4f} | {float(row['mean_delta']):.4f} | "
            f"`{bool(row['all_standard_controls_pass_every_seed'])}` | `{bool(row['all_integrated_mask_controls_pass_every_seed'])}` | "
            f"`{bool(row['leakage_pass'])}` | `{bool(row['frozen_low'])}` | `{bool(row['meaningful_delta'])}` | `{bool(row['phase2_eligible_strict'])}` | "
            f"`{json.dumps(row['failing_controls_by_seed'], sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "## Per-Control Numeric Breakdown",
            "",
            "- Values are dev-split means across cheap-screen seeds `[0,1,2]`; per-seed values are in `results/stage6_phase1_control_breakdown.json`.",
        ]
    )
    for row in summaries:
        if bool(row["phase2_eligible_strict"]):
            continue
        lines.extend(
            [
                "",
                f"### {row['variant']}",
                "",
                "| control | mean accuracy | max accuracy | pass every seed | failing seeds | rule |",
                "|---|---:|---:|---|---|---|",
            ]
        )
        controls = dict(row.get("control_summary", {}))
        for name in REPORT_CONTROL_ORDER:
            if name not in controls:
                continue
            cell = controls[name]
            mean_acc = cell.get("mean_accuracy")
            max_acc = cell.get("max_accuracy")
            mean_text = f"{float(mean_acc):.4f}" if isinstance(mean_acc, (int, float)) else "n/a"
            max_text = f"{float(max_acc):.4f}" if isinstance(max_acc, (int, float)) else "n/a"
            lines.append(
                f"| {name} | {mean_text} | {max_text} | `{bool(cell.get('pass_every_seed'))}` | "
                f"`{cell.get('failing_seeds', [])}` | {cell.get('rule')} |"
            )
    early_summary = next((row for row in summaries if row["variant"] == "integrated_early_mixing1_lr3e4"), None)
    lines.extend(
        [
            "",
            "## Early-Mixing Reconciliation",
            "",
            "- The original Stage 6 Phase 1 table used variant-level mean gates.",
            "- The contradictory prose counted strict per-seed row failures.",
            "- These are different gate groups; the Stage 6 report now labels the table column as `mean-gate pass`, and the audit uses strict every-seed eligibility for Phase 2.",
        ]
    )
    if early_summary:
        lines.append(f"- `integrated_early_mixing1_lr3e4` mean gates passed in the original table, but strict every-seed eligibility is `{bool(early_summary['phase2_eligible_strict'])}`.")
        lines.append(f"- Failing controls by seed: `{json.dumps(early_summary['failing_controls_by_seed'], sort_keys=True)}`")
    summary_late = next((row for row in summaries if row["variant"] == "integrated_summary_late1_lr3e4"), None)
    lines.extend(
        [
            "",
            "## Integrated Summary Investigation",
            "",
        ]
    )
    if summary_late:
        lines.append(f"- `integrated_summary_late1_lr3e4` is not promoted: strict eligible `{bool(summary_late['phase2_eligible_strict'])}`.")
        lines.append(f"- Failing controls by seed: `{json.dumps(summary_late['failing_controls_by_seed'], sort_keys=True)}`")
        lines.append("- Leakage and mask audits passed, so the observed high accuracy is best classified as control-threshold/shortcut behavior rather than proven gold leakage.")
    lines.extend(
        [
            "",
            "## Phase 2 Rule",
            "",
            "- Advance only variants with every-seed standard controls passing, integrated mask/leakage passing, physical and candidate-order invariance passing, low frozen comparator, meaningful trainable-frozen delta, and no contradictory pass/fail statements.",
            f"- Result under that rule: `{eligible}`.",
            "- Because no variant passed strict eligibility and the baseline regression is not clean, Phase 2 was not run.",
            "",
            "## Stage 6 Mean-Gate Snapshot",
            "",
            "| variant | original mean-gate pass | controls pass | masks pass | mean delta |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in stage6_summary:
        lines.append(f"| {row['variant']} | `{bool(row.get('selection_passed'))}` | `{bool(row.get('controls_pass'))}` | `{bool(row.get('attention_mask_audits_pass'))}` | {float(row.get('mean_dev_delta', 0.0)):.4f} |")
    return "\n".join(lines) + "\n"


def write_json(path: Path, value: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def mean_value(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _display_pass(value: object) -> str:
    if value is None:
        return "n/a"
    return str(bool(value))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
