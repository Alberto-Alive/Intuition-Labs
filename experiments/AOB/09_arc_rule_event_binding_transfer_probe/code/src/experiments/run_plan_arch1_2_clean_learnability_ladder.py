from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

from src.experiments.run_plan_arch1_empirical_architecture_discovery import (
    LOCKED_BASELINE,
    FeatureTrainingConfig,
    PlanDatasetConfig,
    PlanTrainingConfig,
    PlanVariant,
    _best_shortcut_baseline,
    _clear_cuda,
    _feature_training_config,
    _fit_and_evaluate_variant,
    _load_config,
    _micro_gates,
    _resolve_device,
    _run_overfit_gates,
    _training_config,
    _variant_plan,
    build_plan_splits,
)


BENCHMARK = "plan_arch1_2_clean_learnability_ladder"
DEFAULT_CONFIG = "configs/plan_arch1_2_clean_learnability_ladder.json"
DEFAULT_RESULTS = "results/plan_arch1_2_ladder_results.json"
DEFAULT_CONTROLS = "results/plan_arch1_2_controls.json"
DEFAULT_AUDIT = "results/plan_arch1_2_transition_representation_audit.json"
DEFAULT_LEADERBOARD = "results/plan_arch1_2_variant_leaderboard.csv"
DEFAULT_FAILURES = "results/plan_arch1_2_failure_taxonomy.json"
DEFAULT_ERRORS = "results/plan_arch1_2_error_cases.jsonl"
DEFAULT_COMPUTE = "results/plan_arch1_2_compute_metrics.json"
DEFAULT_REPORT = "reports/PLAN_ARCH1_2_CLEAN_LEARNABILITY_LADDER.md"

LEVELS = {
    0: "oracle_evaluator_sanity",
    1: "obvious_clean_success_failure",
    2: "goal_proximity_clean",
    3: "near_miss_clean",
    4: "final_transition_clean",
    5: "no_final_inference",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PLAN-ARCH-1.2 clean learnability ladder.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--audit-output", default=DEFAULT_AUDIT)
    parser.add_argument("--leaderboard-output", default=DEFAULT_LEADERBOARD)
    parser.add_argument("--failure-output", default=DEFAULT_FAILURES)
    parser.add_argument("--error-output", default=DEFAULT_ERRORS)
    parser.add_argument("--compute-output", default=DEFAULT_COMPUTE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = _default_config(_load_config(Path(args.config)))
    if args.device:
        config["device"] = str(args.device)
    result = run_plan_arch1_2_ladder(config)
    _write_plan_arch1_2_outputs(
        result,
        Path(args.output),
        Path(args.controls_output),
        Path(args.audit_output),
        Path(args.leaderboard_output),
        Path(args.failure_output),
        Path(args.error_output),
        Path(args.compute_output),
        Path(args.report),
    )


def run_plan_arch1_2_ladder(config: Dict[str, object]) -> Dict[str, object]:
    seeds = [int(seed) for seed in config.get("seeds", [0, 1, 2])]
    device = _resolve_device(str(config.get("device", "cpu")))
    base_dataset = _dataset_config(config.get("dataset", {}))
    training = _training_config(config.get("training", {}))
    feature_training = _feature_training_config(config.get("feature_training", {}))
    micro_training = _training_config({**dict(config.get("training", {})), **dict(config.get("micro_overfit_training", {}))})
    levels = [int(level) for level in config.get("levels", [0, 1, 2, 3, 4, 5])]
    variants_by_name = {variant.name: variant for variant in _variant_plan(None)}

    rows: List[Dict[str, object]] = []
    controls: List[Dict[str, object]] = []
    ablations: List[Dict[str, object]] = []
    attention: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    compute: List[Dict[str, object]] = []
    transition_audit: List[Dict[str, object]] = []
    micro_rows: List[Dict[str, object]] = []

    for level in levels:
        level_name = LEVELS.get(level, f"level_{level}")
        level_variants = _variants_for_level(level, variants_by_name)
        level_dataset = replace(base_dataset, learnability_level=level)
        split_cache: Dict[int, Dict[str, object]] = {}
        baseline_cache: Dict[int, float] = {}
        print(f"plan-arch1.2 level={level} {level_name} variants={len(level_variants)} seeds={seeds}")
        for seed in seeds:
            split_cache[seed] = build_plan_splits(level_dataset, seed=seed)
            transition_audit.append(_transition_audit_row(level, level_name, seed, split_cache[seed], level_variants))
        for variant in level_variants:
            for seed in seeds:
                splits = split_cache[seed]
                level_training = replace(training, epochs=max(int(training.epochs), 8), patience=max(int(training.patience), 8)) if int(level) == 0 else training
                row, control_row, ablation_rows, attention_rows, error_rows, compute_row = _fit_and_evaluate_variant(
                    phase=f"level_{level}",
                    variant=variant,
                    splits=splits,
                    training=replace(level_training, objective=variant.objective),
                    feature_training=feature_training,
                    seed=seed,
                    device=device,
                    baseline_cache=baseline_cache,
                )
                _annotate_row(row, level, level_name, variant)
                _annotate_control(control_row, level, level_name, variant)
                for item in ablation_rows:
                    item.update({"level": level, "level_name": level_name, "evidence_branch": _branch(variant)})
                for item in attention_rows:
                    item.update({"level": level, "level_name": level_name, "evidence_branch": _branch(variant)})
                for item in error_rows:
                    item.update({"level": level, "level_name": level_name, "evidence_branch": _branch(variant)})
                compute_row.update({"level": level, "level_name": level_name, "evidence_branch": _branch(variant)})
                rows.append(row)
                controls.append(control_row)
                ablations.extend(ablation_rows)
                attention.extend(attention_rows)
                errors.extend(error_rows)
                compute.append(compute_row)
                _clear_cuda()

    leaderboard = _leaderboard(rows, controls, ablations)
    level_summaries = _level_summaries(rows, controls, ablations)
    first_learned = _first_level(level_summaries, learned=True)
    first_failed = _first_failed_after_learning(level_summaries)
    if first_learned is None:
        micro_rows = _run_ladder_micro_overfit(base_dataset, micro_training, seeds[0] if seeds else 0, device, variants_by_name)
    failure_taxonomy = _failure_taxonomy(rows, controls, ablations, micro_rows)
    summary = {
        "first_learnable_level": first_learned,
        "first_failed_level_after_learning": first_failed,
        "medium_ready": _medium_ready(leaderboard),
        "final_validation_launched": False,
        "medium_validation_launched": False,
        "level_summaries": level_summaries,
        "if_no_level_learns_micro_overfit": micro_rows,
    }
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "scope": "clean learnability ladder for candidate-plan verification; no medium validation launched",
            "device": device,
            "claim_boundary": "No planning claim and no architecture improvement claim.",
            "medium_validation_launched": False,
        },
        "dataset_config": asdict(base_dataset),
        "training_config": asdict(training),
        "feature_training_config": asdict(feature_training),
        "rows": rows,
        "controls": controls,
        "ablations": ablations,
        "attention_summaries": attention,
        "error_cases": errors,
        "compute_metrics": compute,
        "transition_representation_audit": transition_audit,
        "leaderboard": leaderboard,
        "failure_taxonomy": failure_taxonomy,
        "summary": summary,
    }


def _variants_for_level(level: int, variants_by_name: Dict[str, PlanVariant]) -> List[PlanVariant]:
    if int(level) == 0:
        return [replace(variants_by_name["oracle_success_flag_sanity"], model_dim=32, num_heads=2, ff_dim=64)]
    include_final = int(level) in {1, 2, 3, 4}
    q_self = replace(variants_by_name["Q_candidate_self_attention"], include_final_state=include_final, model_dim=32, num_heads=2, ff_dim=64)
    q_bidir = replace(variants_by_name["Q_bidirectional_candidate_evidence"], include_final_state=include_final, model_dim=32, num_heads=2, ff_dim=64)
    return [
        replace(variants_by_name[LOCKED_BASELINE], model_dim=32, num_heads=2, ff_dim=64),
        replace(variants_by_name["P1_rollout_with_final_state"], model_dim=32, num_heads=2, ff_dim=64),
        replace(variants_by_name["transition_tuple_verifier"], include_final_state=include_final, model_dim=32, num_heads=2, ff_dim=64),
        q_self,
        q_bidir,
        replace(variants_by_name["candidate_token_direct_transition_tokens"], include_final_state=include_final, model_dim=32, num_heads=2, ff_dim=64),
        replace(variants_by_name["simple_cross_attention_verifier_baseline"], include_final_state=include_final, model_dim=32, num_heads=2, ff_dim=64),
    ]


def _annotate_row(row: Dict[str, object], level: int, level_name: str, variant: PlanVariant) -> None:
    baselines = row.get("baselines", {})
    row.update(
        {
            "benchmark": BENCHMARK,
            "level": int(level),
            "level_name": level_name,
            "evidence_branch": _branch(variant),
            "clean_shortcut_best": _clean_shortcut_best(baselines if isinstance(baselines, dict) else {}),
            "medium_validation_launched": False,
        }
    )


def _annotate_control(control: Dict[str, object], level: int, level_name: str, variant: PlanVariant) -> None:
    control.update({"benchmark": BENCHMARK, "level": int(level), "level_name": level_name, "evidence_branch": _branch(variant)})


def _clean_shortcut_best(baselines: Dict[str, object]) -> float:
    keys = ("candidate_only_index0", "length_only", "unigram_action_stat", "bigram_action_stat", "candidate_source_metadata_only", "stats_mlp")
    values = [float(baselines.get(key, 0.0)) for key in keys if isinstance(baselines.get(key, 0.0), (int, float))]
    return max(values) if values else 0.0


def _branch(variant: PlanVariant) -> str:
    return "rollout_with_final_state" if bool(variant.include_final_state) or bool(variant.include_outcome_tokens) else "rollout_no_final_state"


def _leaderboard(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]], ablations: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    control_by_key = {(row["level"], row["variant"], row["seed"]): row for row in controls}
    ablation_by_key: Dict[Tuple[int, str, int], bool] = {}
    for row in ablations:
        key = (int(row["level"]), str(row["variant"]), int(row["seed"]))
        ablation_by_key[key] = bool(ablation_by_key.get(key, False) or row.get("meaningful_drop", False))
    out = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        key = (int(row["level"]), str(row["variant"]), int(row["seed"]))
        control = control_by_key.get(key, {})
        chance = float(row.get("baselines", {}).get("random", 0.125))
        trainable = float(row.get("trainable", {}).get("top1", 0.0))
        frozen = float(row.get("frozen", {}).get("top1", 0.0))
        delta = float(row.get("delta_trainable_minus_frozen", trainable - frozen))
        controls_pass = bool(control.get("control_pass", {}).get("overall", False))
        shortcut_best = float(row.get("clean_shortcut_best", 1.0))
        meaningful_ablation = bool(ablation_by_key.get(key, False))
        gate_pass = bool(delta > 0 and trainable > chance and controls_pass and shortcut_best <= chance + 0.10 and meaningful_ablation)
        out.append(
            {
                "level": int(row["level"]),
                "level_name": row["level_name"],
                "variant": row["variant"],
                "seed": int(row["seed"]),
                "evidence_branch": row["evidence_branch"],
                "trainable_top1": trainable,
                "frozen_top1": frozen,
                "delta": delta,
                "chance": chance,
                "clean_shortcut_best": shortcut_best,
                "controls_pass": controls_pass,
                "meaningful_ablation": meaningful_ablation,
                "gate_pass_seed": gate_pass,
                "score": trainable + delta + (0.2 if controls_pass else -1.0) - max(0.0, shortcut_best - chance),
                "status": row["status"],
            }
        )
    return sorted(out, key=lambda r: (r["level"], -float(r["score"]), str(r["variant"]), int(r["seed"])))


def _level_summaries(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]], ablations: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    leaderboard = _leaderboard(rows, controls, ablations)
    by_key: Dict[Tuple[int, str], List[Dict[str, object]]] = {}
    for row in leaderboard:
        by_key.setdefault((int(row["level"]), str(row["variant"])), []).append(row)
    summaries = []
    for (level, variant), group in sorted(by_key.items()):
        if int(level) == 0:
            learned = bool(mean([float(row["trainable_top1"]) for row in group]) >= 0.90)
        else:
            learned = _variant_gate(group)
        summaries.append(
            {
                "level": level,
                "level_name": LEVELS.get(level, f"level_{level}"),
                "variant": variant,
                "mean_trainable_top1": mean([float(row["trainable_top1"]) for row in group]),
                "mean_frozen_top1": mean([float(row["frozen_top1"]) for row in group]),
                "mean_delta": mean([float(row["delta"]) for row in group]),
                "seeds_trainable_beats_frozen": sum(float(row["delta"]) > 0.0 for row in group),
                "controls_pass_all": all(bool(row["controls_pass"]) for row in group),
                "shortcuts_near_chance_all": all(float(row["clean_shortcut_best"]) <= float(row["chance"]) + 0.10 for row in group),
                "meaningful_ablation_any": any(bool(row["meaningful_ablation"]) for row in group),
                "learned_cleanly": learned,
            }
        )
    return summaries


def _variant_gate(group: Sequence[Dict[str, object]]) -> bool:
    if not group:
        return False
    return bool(
        sum(float(row["delta"]) > 0.0 for row in group) >= 2
        and mean([float(row["delta"]) for row in group]) >= 0.10
        and mean([float(row["trainable_top1"]) for row in group]) > mean([float(row["chance"]) for row in group])
        and all(bool(row["controls_pass"]) for row in group)
        and all(float(row["clean_shortcut_best"]) <= float(row["chance"]) + 0.10 for row in group)
        and any(bool(row["meaningful_ablation"]) for row in group)
    )


def _first_level(level_summaries: Sequence[Dict[str, object]], learned: bool) -> Dict[str, object] | None:
    candidates = [row for row in level_summaries if int(row["level"]) > 0 and bool(row["learned_cleanly"]) is bool(learned)]
    return min(candidates, key=lambda row: int(row["level"])) if candidates else None


def _first_failed_after_learning(level_summaries: Sequence[Dict[str, object]]) -> Dict[str, object] | None:
    learned = _first_level(level_summaries, learned=True)
    if not learned:
        return _first_level(level_summaries, learned=False)
    for level in range(int(learned["level"]) + 1, 6):
        rows = [row for row in level_summaries if int(row["level"]) == level]
        if rows and not any(bool(row["learned_cleanly"]) for row in rows):
            return {"level": level, "level_name": LEVELS.get(level, f"level_{level}")}
    return None


def _medium_ready(leaderboard: Sequence[Dict[str, object]]) -> Dict[str, object] | None:
    by_key: Dict[Tuple[int, str], List[Dict[str, object]]] = {}
    for row in leaderboard:
        by_key.setdefault((int(row["level"]), str(row["variant"])), []).append(row)
    for (level, variant), group in sorted(by_key.items()):
        if int(level) > 0 and _variant_gate(group):
            return {"level": level, "level_name": LEVELS.get(level, f"level_{level}"), "variant": variant}
    return None


def _run_ladder_micro_overfit(
    dataset: PlanDatasetConfig,
    training: PlanTrainingConfig,
    seed: int,
    device: str,
    variants_by_name: Dict[str, PlanVariant],
) -> List[Dict[str, object]]:
    variant = replace(variants_by_name["P1_rollout_with_final_state"], model_dim=32, num_heads=2, ff_dim=64)
    rows = _run_overfit_gates(
        variant,
        replace(dataset, learnability_level=1),
        training,
        seed,
        device,
        _micro_gates(None),
    )
    for row in rows:
        row.update({"benchmark": BENCHMARK, "level": 1, "level_name": LEVELS[1], "variant": variant.name})
    return rows


def _failure_taxonomy(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    ablations: Sequence[Dict[str, object]],
    micro_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    counts: Dict[str, int] = {}
    for row in _leaderboard(rows, controls, ablations):
        if not bool(row["controls_pass"]):
            counts["control_failure"] = counts.get("control_failure", 0) + 1
        if float(row["clean_shortcut_best"]) > float(row["chance"]) + 0.10:
            counts["shortcut_baseline_above_chance"] = counts.get("shortcut_baseline_above_chance", 0) + 1
        if float(row["delta"]) <= 0.0:
            counts["did_not_beat_frozen"] = counts.get("did_not_beat_frozen", 0) + 1
        if not bool(row["meaningful_ablation"]):
            counts["no_meaningful_ablation_drop"] = counts.get("no_meaningful_ablation_drop", 0) + 1
    if micro_rows:
        if all(bool(row.get("pass")) for row in micro_rows):
            counts["heldout_generalization_bottleneck"] = counts.get("heldout_generalization_bottleneck", 0) + 1
        else:
            counts["micro_overfit_failure"] = counts.get("micro_overfit_failure", 0) + 1
    return {
        "counts": counts,
        "failure_reasons": {
            "control_failure": "one or more repaired controls failed",
            "shortcut_baseline_above_chance": "candidate-only, length, action-stat, metadata, or candidate-stat shortcut exceeded chance window",
            "did_not_beat_frozen": "trainable shared model did not improve over exact frozen same architecture",
            "no_meaningful_ablation_drop": "view ablation did not remove enough accuracy",
            "heldout_generalization_bottleneck": "micro-overfit passed but held-out ladder gates did not",
            "micro_overfit_failure": "scoring, labels, masks, gradients, or tokenization remain suspect",
        },
    }


def _transition_audit_row(level: int, level_name: str, seed: int, splits: Dict[str, Sequence[object]], variants: Sequence[PlanVariant]) -> Dict[str, object]:
    examples = list(splits["test"])
    if not examples:
        return {"level": level, "level_name": level_name, "seed": seed, "empty": True}
    first = examples[0]
    return {
        "level": level,
        "level_name": level_name,
        "seed": seed,
        "candidate_count": len(first.candidates),
        "label_histogram": {str(i): sum(1 for ex in examples if int(ex.label) == i) for i in range(len(first.candidates))},
        "exactly_one_success": all(sum(bool(value) for value in ex.goal_reached) == 1 for ex in examples),
        "all_same_length": all(len(set(len(cand) for cand in ex.candidates)) == 1 for ex in examples),
        "candidate_sources_hidden": all(not bool(ex.metadata.get("model_visible_candidate_sources", True)) for ex in examples),
        "success_flag_hidden": all(not bool(ex.metadata.get("model_visible_simulator_success_flag", True)) for ex in examples),
        "transition_variants": [variant.name for variant in variants if any("transition" in view for view in variant.view_names)],
        "branches": sorted({_branch(variant) for variant in variants}),
    }


def _write_plan_arch1_2_outputs(
    result: Dict[str, object],
    results_path: Path,
    controls_path: Path,
    audit_path: Path,
    leaderboard_path: Path,
    failure_path: Path,
    error_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    for path in (results_path, controls_path, audit_path, leaderboard_path, failure_path, error_path, compute_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(_without_large_logs(result), indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(json.dumps({"controls": result["controls"], "summary": result["summary"]}, indent=2, sort_keys=True), encoding="utf-8")
    audit_path.write_text(json.dumps({"transition_representation_audit": result["transition_representation_audit"]}, indent=2, sort_keys=True), encoding="utf-8")
    failure_path.write_text(json.dumps(result["failure_taxonomy"], indent=2, sort_keys=True), encoding="utf-8")
    compute_path.write_text(json.dumps({"compute_metrics": result["compute_metrics"]}, indent=2, sort_keys=True), encoding="utf-8")
    with leaderboard_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "level",
            "level_name",
            "variant",
            "seed",
            "evidence_branch",
            "trainable_top1",
            "frozen_top1",
            "delta",
            "chance",
            "clean_shortcut_best",
            "controls_pass",
            "meaningful_ablation",
            "gate_pass_seed",
            "score",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["leaderboard"]:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    with error_path.open("w", encoding="utf-8") as handle:
        for row in result["error_cases"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report_path.write_text(_render_report(result), encoding="utf-8")


def _without_large_logs(result: Dict[str, object]) -> Dict[str, object]:
    trimmed = dict(result)
    trimmed["attention_summaries"] = result.get("attention_summaries", [])[:40]
    return trimmed


def _render_report(result: Dict[str, object]) -> str:
    summary = result["summary"]
    medium_ready = summary.get("medium_ready")
    first = summary.get("first_learnable_level")
    failed = summary.get("first_failed_level_after_learning")
    leaderboard = result["leaderboard"][:12]
    best_transition = _best_by_variant_prefix(result["leaderboard"], ("transition_tuple_verifier", "candidate_token_direct_transition_tokens", "simple_cross_attention_verifier_baseline"))
    best_rollout = _best_by_variant_prefix(result["leaderboard"], (LOCKED_BASELINE, "P1_rollout_with_final_state"))
    q_rows = [row for row in result["leaderboard"] if row["variant"] == "Q_candidate_self_attention"]
    q_help = any(row.get("gate_pass_seed") for row in q_rows)
    lines = [
        "# PLAN-ARCH-1.2 Clean Learnability Ladder",
        "",
        "## Scope",
        "- No planning claim.",
        "- No architecture improvement claim.",
        "- Medium validation was not launched.",
        "",
        "## Answers",
        f"1. Which clean difficulty level first becomes learnable? `{_level_answer(first)}`.",
        f"2. Does final-state evidence help when controls are repaired? `{_final_state_answer(result['leaderboard'])}`.",
        f"3. Can the model infer the final transition without seeing final state? `{_no_final_answer(result['leaderboard'])}`.",
        f"4. Which representation works best: rollout tokens, transition tokens, or hybrid? `{_representation_answer(best_rollout, best_transition)}`.",
        f"5. Does candidate self-attention still help on any clean level? `{q_help}`.",
        f"6. Does trainable beat frozen on clean held-out splits? `{any(row.get('gate_pass_seed') for row in result['leaderboard'])}`.",
        f"7. Are failures due to over-hard candidate pools or architecture weakness? `{_failure_answer(result)}`.",
        f"8. Is there a medium-ready benchmark/variant? `{medium_ready or 'none'}`.",
        "",
        "## First Failure",
        f"- `{_level_answer(failed)}`",
        "",
        "## Leaderboard",
        "| level | variant | seed | trainable | frozen | delta | shortcut | controls |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in leaderboard:
        lines.append(
            f"| {row['level']} | {row['variant']} | {row['seed']} | {float(row['trainable_top1']):.4f} | "
            f"{float(row['frozen_top1']):.4f} | {float(row['delta']):.4f} | {float(row['clean_shortcut_best']):.4f} | `{bool(row['controls_pass'])}` |"
        )
    return "\n".join(lines) + "\n"


def _best_by_variant_prefix(leaderboard: Sequence[Dict[str, object]], names: Sequence[str]) -> Dict[str, object] | None:
    rows = [row for row in leaderboard if row["variant"] in names]
    return max(rows, key=lambda row: float(row["trainable_top1"]), default=None)


def _level_answer(row: object) -> str:
    if not isinstance(row, dict):
        return "none"
    return f"level {row.get('level')} {row.get('level_name')} variant={row.get('variant', 'n/a')}"


def _final_state_answer(leaderboard: Sequence[Dict[str, object]]) -> str:
    with_final = [row for row in leaderboard if row["evidence_branch"] == "rollout_with_final_state"]
    no_final = [row for row in leaderboard if row["evidence_branch"] == "rollout_no_final_state"]
    best_with = max([float(row["trainable_top1"]) for row in with_final], default=0.0)
    best_without = max([float(row["trainable_top1"]) for row in no_final], default=0.0)
    return f"best_with_final={best_with:.4f}, best_no_final={best_without:.4f}, helps={best_with > best_without}"


def _no_final_answer(leaderboard: Sequence[Dict[str, object]]) -> str:
    rows = [row for row in leaderboard if row["level"] == 5 and row["evidence_branch"] == "rollout_no_final_state"]
    return f"clean_gate={any(row.get('gate_pass_seed') for row in rows)}, best_top1={max([float(row['trainable_top1']) for row in rows], default=0.0):.4f}"


def _representation_answer(rollout: Dict[str, object] | None, transition: Dict[str, object] | None) -> str:
    if not rollout and not transition:
        return "none"
    rollout_score = float(rollout["trainable_top1"]) if rollout else 0.0
    transition_score = float(transition["trainable_top1"]) if transition else 0.0
    if transition_score > rollout_score:
        return f"transition/hybrid ({transition['variant']} top1={transition_score:.4f})"
    return f"rollout ({rollout['variant']} top1={rollout_score:.4f})"


def _failure_answer(result: Dict[str, object]) -> str:
    micro = result["summary"].get("if_no_level_learns_micro_overfit", [])
    if micro:
        return "representation/generalization bottleneck" if all(row.get("pass") for row in micro) else "scoring/tokenization/mask/gradient issue"
    return "over-hard levels after first learned level or frozen comparator parity; see failure taxonomy"


def _default_config(config: Dict[str, object]) -> Dict[str, object]:
    base = {
        "device": "cpu",
        "seeds": [0, 1, 2],
        "levels": [0, 1, 2, 3, 4, 5],
        "dataset": {
            "grid_size": 8,
            "num_candidates": 8,
            "plan_length": 12,
            "obstacle_count": 0,
            "train_examples": 24,
            "dev_examples": 12,
            "test_examples": 32,
            "max_generation_attempts": 800,
            "shortcut_pool_attempts": 500,
            "require_exactly_one_success": True,
        },
        "training": {
            "epochs": 2,
            "batch_size": 12,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "patience": 2,
            "gradient_clip_norm": 1.0,
        },
        "micro_overfit_training": {
            "epochs": 25,
            "patience": 25,
            "batch_size": 16,
            "lr": 0.001,
        },
        "feature_training": {
            "epochs": 4,
            "batch_size": 16,
            "lr": 0.003,
            "weight_decay": 0.0001,
            "patience": 2,
            "hidden_dim": 24,
        },
    }
    return _deep_update(base, config)


def _dataset_config(data: object) -> PlanDatasetConfig:
    values = dict(data or {})
    allowed = set(PlanDatasetConfig.__dataclass_fields__.keys())
    return PlanDatasetConfig(**{key: value for key, value in values.items() if key in allowed})


def _deep_update(base: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(dict(out[key]), value)
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    main()
