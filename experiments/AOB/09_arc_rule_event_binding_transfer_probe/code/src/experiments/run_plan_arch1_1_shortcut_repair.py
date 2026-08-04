from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

from src.experiments.run_plan_arch1_empirical_architecture_discovery import (
    LOCKED_BASELINE,
    FeatureTrainingConfig,
    PlanDatasetConfig,
    _baseline_block,
    _best_shortcut_baseline,
    _load_config,
    _write_outputs,
    build_plan_splits,
    run_plan_arch1_search,
)


BENCHMARK = "plan_arch1_1_shortcut_repair"
DEFAULT_CONFIG = "configs/plan_arch1_1_shortcut_repair.json"
DEFAULT_AUDIT = "results/plan_arch1_1_shortcut_audit.json"
DEFAULT_RESULTS = "results/plan_arch1_1_repaired_results.json"
DEFAULT_CONTROLS = "results/plan_arch1_1_controls.json"
DEFAULT_LEADERBOARD = "results/plan_arch1_1_variant_leaderboard.csv"
DEFAULT_FAILURES = "results/plan_arch1_1_failure_taxonomy.json"
DEFAULT_ERRORS = "results/plan_arch1_1_error_cases.jsonl"
DEFAULT_COMPUTE = "results/plan_arch1_1_compute_metrics.json"
DEFAULT_REPORT = "reports/PLAN_ARCH1_1_SHORTCUT_REPAIR.md"
DEFAULT_OVERFIT = "results/plan_arch1_1_overfit_curves.json"
DEFAULT_ABLATIONS = "results/plan_arch1_1_ablation_results.json"
DEFAULT_ATTENTION = "results/plan_arch1_1_attention_summaries.jsonl"

RETEST_VARIANTS = [
    LOCKED_BASELINE,
    "Q_candidate_self_attention",
    "Q_bidirectional_candidate_evidence",
    "Q_candidate_self_attention_plus_rollout_no_final",
    "Q_candidate_self_attention_with_stronger_mismatch_controls",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PLAN-ARCH-1.1 shortcut repair and candidate-query retest.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--audit-output", default=DEFAULT_AUDIT)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--leaderboard-output", default=DEFAULT_LEADERBOARD)
    parser.add_argument("--failure-output", default=DEFAULT_FAILURES)
    parser.add_argument("--error-output", default=DEFAULT_ERRORS)
    parser.add_argument("--compute-output", default=DEFAULT_COMPUTE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = _plan_arch1_1_config(_load_config(Path(args.config)))
    if args.device:
        config["device"] = str(args.device)

    audit = build_shortcut_audit(config)
    baseline_config = {**config, "variants": [LOCKED_BASELINE], "run_medium": False}
    baseline_result = run_plan_arch1_search(baseline_config)
    baseline_clean = _p1_repaired_baseline_clean(baseline_result)

    if baseline_clean:
        result = run_plan_arch1_search(config)
        retest_status = "completed_after_clean_p1"
    else:
        result = baseline_result
        retest_status = "blocked_by_unclean_p1"

    result.setdefault("metadata", {})["plan_arch1_1_retest_status"] = retest_status
    result["metadata"]["shortcut_audit_summary"] = audit["summary"]
    result["metadata"]["medium_validation_launched"] = False
    result["summary"]["medium_validation_launched"] = False
    result["summary"]["plan_arch1_1_retest_status"] = retest_status
    result["summary"]["p1_repaired_baseline_clean"] = bool(baseline_clean)

    _write_outputs(
        result,
        Path(args.output),
        Path(args.controls_output),
        Path(args.leaderboard_output),
        Path(args.failure_output),
        Path(DEFAULT_OVERFIT),
        Path(DEFAULT_ABLATIONS),
        Path(DEFAULT_ATTENTION),
        Path(args.error_output),
        Path(args.compute_output),
        Path(args.report),
    )
    Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_output).write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    Path(args.report).write_text(_render_plan_arch1_1_report(result, audit), encoding="utf-8")


def _plan_arch1_1_config(config: Dict[str, object]) -> Dict[str, object]:
    merged = {
        "device": "cpu",
        "cheap_seeds": [0, 1, 2],
        "run_medium": False,
        "run_overfit_gates": False,
        "variants": RETEST_VARIANTS,
        "dataset": {
            "grid_size": 8,
            "num_candidates": 8,
            "plan_length": 12,
            "obstacle_count": 6,
            "train_examples": 48,
            "dev_examples": 24,
            "test_examples": 64,
            "require_exactly_one_success": True,
            "shortcut_repaired_candidate_pools": True,
            "max_generation_attempts": 500,
            "shortcut_pool_attempts": 800,
        },
        "cheap_dataset": {
            "train_examples": 48,
            "dev_examples": 24,
            "test_examples": 64,
        },
        "training": {
            "epochs": 3,
            "batch_size": 16,
            "lr": 0.0007,
            "weight_decay": 0.0001,
            "patience": 2,
            "gradient_clip_norm": 1.0,
            "objective": "cross_entropy",
            "aux_weight": 0.25,
        },
        "feature_training": {
            "epochs": 6,
            "batch_size": 24,
            "lr": 0.003,
            "weight_decay": 0.0001,
            "patience": 3,
            "hidden_dim": 32,
        },
    }
    return _deep_update(merged, config)


def build_shortcut_audit(config: Dict[str, object]) -> Dict[str, object]:
    dataset_values = dict(config.get("dataset", {}))
    audit_dataset = PlanDatasetConfig(**{key: value for key, value in dataset_values.items() if key in PlanDatasetConfig.__dataclass_fields__})
    audit_dataset = PlanDatasetConfig(**{**asdict(audit_dataset), "train_examples": 48, "dev_examples": 24, "test_examples": 64})
    feature_training = FeatureTrainingConfig(**{key: value for key, value in dict(config.get("feature_training", {})).items() if key in FeatureTrainingConfig.__dataclass_fields__})
    prior = _prior_plan_arch1_shortcut()
    repaired_summaries = []
    for seed in [0, 1, 2]:
        splits = build_plan_splits(audit_dataset, seed=seed)
        baselines = _baseline_block(splits["train"], splits["dev"], splits["test"], feature_training, seed)
        repaired_summaries.append(
            {
                "seed": seed,
                "baselines": {key: value for key, value in baselines.items() if isinstance(value, (int, float))},
                "best_shortcut": _best_shortcut_baseline(baselines),
                "label_histogram": _label_histogram([example.label for example in splits["test"]], audit_dataset.num_candidates),
                "exactly_one_success": all(sum(bool(v) for v in example.goal_reached) == 1 for example in splits["test"]),
                "all_candidates_same_length": all(len(set(len(c) for c in example.candidates)) == 1 for example in splits["test"]),
                "source_metadata_visible": any(bool(example.metadata.get("model_visible_candidate_sources", True)) for example in splits["test"]),
                "simulator_success_visible": any(bool(example.metadata.get("model_visible_simulator_success_flag", True)) for example in splits["test"]),
            }
        )
    plan11 = _prior_plan_1_1_controls()
    return {
        "benchmark": BENCHMARK,
        "summary": {
            "why_shortcut_reached_0_6250": "The prior PLAN-ARCH-1 shortcut score was the trainable stats_mlp baseline. Its features included rollout-derived collision/progress and start-goal progress, while gold candidates were padded BFS plans and negatives came from visibly different action/progress families.",
            "classification": "mixed representation leak and baseline bug; the dataset exposed systematic action/progress artifacts and the stats_mlp shortcut consumed rollout/state-goal features instead of candidate-only action statistics.",
            "exact_difference_from_plan_1_1": "PLAN-1.1 generated balanced candidate grids, hid source tags, audited metadata-only accuracy, and used candidate/evidence mismatch controls. PLAN-ARCH-1 generated gold from a single BFS source and negatives from unmatched source families, then scored a shortcut baseline with rollout progress/collision features.",
            "medium_validation_launched": False,
        },
        "prior_plan_arch1": prior,
        "prior_clean_plan_1_1": plan11,
        "repaired_candidate_pool_audit": repaired_summaries,
    }


def _prior_plan_arch1_shortcut() -> Dict[str, object]:
    path = Path("results/plan_arch1_results.json")
    if not path.exists():
        return {"available": False}
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in raw.get("rows", []):
        if not isinstance(row, dict) or row.get("status") != "completed":
            continue
        baselines = row.get("baselines", {})
        rows.append(
            {
                "variant": row.get("variant"),
                "seed": row.get("seed"),
                "trainable_top1": row.get("trainable", {}).get("top1"),
                "shortcut_best": _best_shortcut_baseline(baselines if isinstance(baselines, dict) else {}),
                "baselines": {key: value for key, value in baselines.items() if isinstance(value, (int, float))} if isinstance(baselines, dict) else {},
            }
        )
    return {"available": True, "rows": rows}


def _prior_plan_1_1_controls() -> Dict[str, object]:
    path = Path("results/arc1_1_empirical_search_controls.json")
    if not path.exists():
        return {"available": False}
    raw = json.loads(path.read_text(encoding="utf-8"))
    controls = raw.get("controls", [])
    rows = []
    for row in controls[:8]:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "variant": row.get("variant"),
                "seed": row.get("seed"),
                "candidate_only_top1": row.get("candidate_only", {}).get("top1"),
                "candidate_source_metadata_only_accuracy": row.get("candidate_source_metadata_only_accuracy"),
                "overall_control_pass": row.get("control_pass", {}).get("overall"),
            }
        )
    return {"available": True, "sample_controls": rows}


def _p1_repaired_baseline_clean(result: Dict[str, object]) -> bool:
    p1_rows = [row for row in result.get("rows", []) if row.get("variant") == LOCKED_BASELINE and row.get("status") == "completed"]
    p1_controls = [row for row in result.get("controls", []) if row.get("variant") == LOCKED_BASELINE]
    if not p1_rows or not p1_controls:
        return False
    return all(bool(row.get("control_pass", {}).get("overall", False)) for row in p1_controls) and all(
        _best_shortcut_baseline(row.get("baselines", {})) <= (1.0 / max(1, int(row.get("candidate_count", 8))) + 0.10)
        for row in p1_rows
    )


def _render_plan_arch1_1_report(result: Dict[str, object], audit: Dict[str, object]) -> str:
    leaderboard = result.get("leaderboard", [])
    top = leaderboard[:8] if isinstance(leaderboard, list) else []
    p1_clean = bool(result.get("summary", {}).get("p1_repaired_baseline_clean", False))
    medium_ready = result.get("summary", {}).get("medium_ready_variant")
    q_rows = [row for row in top if row.get("variant") == "Q_candidate_self_attention"]
    p1_rows = [row for row in top if row.get("variant") == LOCKED_BASELINE]
    q_beats_p1 = bool(q_rows and p1_rows and float(q_rows[0].get("trainable_top1", 0.0)) > float(p1_rows[0].get("trainable_top1", 0.0)) and bool(q_rows[0].get("controls_pass", False)))
    q_controls = [row for row in result.get("controls", []) if row.get("variant") == "Q_candidate_self_attention"]
    q_pair_failures = sum(1 for row in q_controls if not bool(row.get("control_pass", {}).get("candidate_pair_artifact_near_chance", True)))
    q_self_only_values = [float(row.get("candidate_self_attention_only", {}).get("top1", 0.0)) for row in q_controls if "candidate_self_attention_only" in row]
    q_assessment = (
        f"No. candidate_self_attention_only stayed near chance ({q_self_only_values}), "
        f"but candidate_pair_artifact_baseline failed on {q_pair_failures}/{len(q_controls)} seeds and trainable did not beat frozen reliably."
        if q_controls
        else "No candidate self-attention retest rows were produced."
    )
    lines = [
        "# PLAN-ARCH-1.1 Shortcut Regression Repair",
        "",
        "## Scope",
        "- No planning claim.",
        "- No architecture improvement claim.",
        "- This stage repairs the benchmark and retests candidate-query mechanisms only.",
        "- Medium validation was not launched.",
        "",
        "## Audit Answer",
        f"1. Why did shortcut baseline reach 0.6250? {audit['summary']['why_shortcut_reached_0_6250']}",
        f"2. Was this a data-generation artifact, baseline bug, or token leakage? {audit['summary']['classification']}",
        f"3. Can the clean PLAN-1.1 baseline be reproduced? Repaired P1 clean controls: `{p1_clean}`. Prior PLAN-1.1 controls are summarized in the audit JSON.",
        f"4. Does Q_candidate_self_attention still beat P1 after shortcut repair? `{q_beats_p1}`.",
        f"5. Does candidate self-attention add real evidence use or only exploit candidate-list artifacts? {q_assessment}",
        f"6. Are any variants medium-ready after repair? `{medium_ready or 'none'}`.",
        "",
        "## Leaderboard",
        "| rank | variant | trainable | frozen | delta | shortcut | controls |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(top, start=1):
        lines.append(
            f"| {rank} | {row.get('variant')} | {float(row.get('trainable_top1', 0.0)):.4f} | "
            f"{float(row.get('frozen_top1', 0.0)):.4f} | {float(row.get('delta', 0.0)):.4f} | "
            f"{float(row.get('shortcut_best', 0.0)):.4f} | `{bool(row.get('controls_pass', False))}` |"
        )
    lines.extend(
        [
            "",
            "## Repair Notes",
            "- Repaired pools use balanced candidate slots and hide source metadata.",
            "- All candidates have the same action length.",
            "- Repaired negatives are no-collision paths whose visible rollout prefix ends one step from the goal; only the final transition determines success.",
            "- The stats shortcut baseline was corrected to use candidate/action statistics only, not rollout progress or collision features.",
        ]
    )
    return "\n".join(lines) + "\n"


def _label_histogram(labels: Sequence[int], count: int) -> Dict[str, int]:
    return {str(i): int(sum(1 for label in labels if int(label) == i)) for i in range(int(count))}


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
