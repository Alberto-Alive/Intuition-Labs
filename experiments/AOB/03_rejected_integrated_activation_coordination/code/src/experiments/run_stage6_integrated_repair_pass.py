from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Dict, List, Sequence

import torch

from src.datasets.multiview_code_patch_selection import (
    build_multiview_code_patch_splits,
    dataset_summary,
    output_leakage_audit,
    randomized_labels_for_examples,
    split_leakage_audit,
)
from src.experiments.architecture_search import (
    _agent_config,
    _candidate_config,
    _coordinator_config,
    _leakage_passes,
    _training_config,
)
from src.experiments.real_shared_weight_latent_coordination import (
    BENCHMARK,
    MessageChannelConfig,
    fit_latent_system,
)
from src.experiments.run_stage3_gpu_hard_validation import _clear_cuda, _configure_cuda
from src.experiments.run_stage38_full_hard_validation import _dataset_config_for_stage, _stage38_full_config, _stage_from_config
from src.experiments import run_stage6_integrated_multiperspective_search as stage6


DEFAULT_CONFIG = "configs/stage6_integrated_multiperspective_search.json"
DEFAULT_RESULTS = "results/stage6_integrated_repair_results.json"
DEFAULT_CONTROLS = "results/stage6_integrated_repair_controls.json"
DEFAULT_MASK_AUDIT = "results/stage6_integrated_repair_mask_audit.jsonl"
DEFAULT_REPORT = "reports/STAGE6_INTEGRATED_REPAIR_PASS.md"
DEFAULT_CHECKPOINT_DIR = "results/stage6_integrated_repair_checkpoints"

NEAR = 0.18
INV_TOL = 0.08
FROZEN_MAX = 0.25
DELTA_MIN = 0.20
BASELINE = "candidate_token_direct_lr3e4_clip1"
STANDARD_CONTROLS = (
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
INTEGRATED_CONTROLS = (
    "no_candidate_position_shortcut_audit",
    "role_token_ablation",
    "cross_stream_attention_disabled",
)


@dataclass(frozen=True)
class RepairVariant:
    name: str
    family: str
    coordinator_family: str
    description: str
    coordinator_layers: int = 1
    lr: float = 0.0003
    epochs: int = 50
    patience: int = 6
    dropout: float = 0.0
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0
    repair_uniform_conditions: tuple[str, ...] = ()
    repair_uniform_weight: float = 0.0
    repair_uniform_controls_per_batch: int = 0
    repair_candidate_order_shuffle_weight: float = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded Stage 6 integrated repair/salvage pass.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--mask-audit", default=DEFAULT_MASK_AUDIT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.device is not None:
        config["device"] = str(args.device)
        if str(args.device) == "cpu":
            config["require_cuda"] = False
    result = run_repair_pass(config=config, config_path=config_path, checkpoint_dir=Path(args.checkpoint_dir))
    write_outputs(
        result=result,
        output_path=Path(args.output),
        controls_path=Path(args.controls_output),
        mask_path=Path(args.mask_audit),
        report_path=Path(args.report),
    )


def run_repair_pass(config: Dict[str, object], config_path: Path, checkpoint_dir: Path) -> Dict[str, object]:
    config = _stage38_full_config(config)
    stage_cfg = dict(config["stage"])
    config["stage"] = {
        **stage_cfg,
        "name": "stage6_integrated_repair_pass",
        "n_train": 512,
        "n_dev": 256,
        "n_test": 256,
        "seeds": [0, 1, 2],
        "epochs": 50,
        "patience": 6,
        "batch_size": 32,
        "lr": 0.0003,
        "mixed_precision": stage_cfg.get("mixed_precision", "bf16"),
    }
    device, hardware = _configure_cuda(config)
    variants = _repair_variants()
    rows = _run_rows(config, variants, device, checkpoint_dir)
    summary = _summary(rows)
    return {
        "metadata": {
            "created_at_utc": now(),
            "config_path": str(config_path),
            "stage": "stage6_integrated_repair_pass",
            "scope": "bounded cheap-screen repair pass only; no Phase 2 or final validation",
            "mainline": BASELINE,
            "device": device,
            "hardware": hardware,
            "thresholds": {"near_chance_max": NEAR, "invariance_tolerance": INV_TOL, "frozen_max": FROZEN_MAX, "delta_min": DELTA_MIN},
        },
        "variant_plan": [asdict(variant) for variant in variants],
        "cheap_screen": {"rows": rows, "summary": summary},
        "medium_validation": {"rows": [], "summary": {"ran": False, "reason": "forbidden for repair pass"}},
        "final_validation": {"rows": [], "summary": {"ran": False, "reason": "forbidden for repair pass"}},
        "decision": _decision(summary),
    }


def _run_rows(config: Dict[str, object], variants: Sequence[RepairVariant], device: str, checkpoint_dir: Path) -> List[Dict[str, object]]:
    stage = _stage_from_config(config)
    rows: List[Dict[str, object]] = []
    for seed in stage.seeds:
        print(f"stage6 repair seed={seed}: building fixed cheap-screen splits")
        splits = build_multiview_code_patch_splits(_dataset_config_for_stage(config, stage), seed=int(seed), repo_root=Path("."))
        split_leakage = split_leakage_audit(BENCHMARK, int(seed), splits)
        output_leakage = output_leakage_audit(BENCHMARK, int(seed), splits["train"] + splits["dev"] + splits["test"])
        metadata_only = stage6._candidate_metadata_shortcut(config, splits, int(seed))
        no_gold = stage6._no_gold_token_audit(splits)
        for index, variant in enumerate(variants):
            print(f"stage6 repair: variant={variant.name} seed={seed}")
            _clear_cuda()
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            candidate = _candidate_for_variant(stage, variant)
            training = replace(
                _training_config(stage),
                epochs=int(variant.epochs),
                patience=int(variant.patience),
                lr=float(variant.lr),
                weight_decay=float(variant.weight_decay),
                gradient_clip_norm=float(variant.gradient_clip_norm),
            )
            fit_seed = int(seed) + index * 100_000 + 760_001
            repair_kwargs = _repair_kwargs(variant)
            t0 = time.perf_counter()
            trainable = fit_latent_system(
                train_examples=splits["train"],
                dev_examples=splits["dev"],
                agent_config=_agent_config(stage),
                coordinator_config=candidate.coordinator_config,
                training_config=training,
                num_classes=8,
                seed=fit_seed,
                device=device,
                trainable_agent=True,
                method=f"stage6_repair_trainable__{variant.name}",
                message_config=candidate.message_config,
                **repair_kwargs,
            )
            trainable_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            frozen = fit_latent_system(
                train_examples=splits["train"],
                dev_examples=splits["dev"],
                agent_config=_agent_config(stage),
                coordinator_config=candidate.coordinator_config,
                training_config=training,
                num_classes=8,
                seed=fit_seed,
                device=device,
                trainable_agent=False,
                method=f"stage6_repair_frozen__{variant.name}",
                message_config=candidate.message_config,
                **repair_kwargs,
            )
            frozen_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            randomized = fit_latent_system(
                train_examples=splits["train"],
                dev_examples=splits["dev"],
                agent_config=_agent_config(stage),
                coordinator_config=candidate.coordinator_config,
                training_config=training,
                num_classes=8,
                seed=fit_seed,
                device=device,
                trainable_agent=True,
                method=f"stage6_repair_randomized__{variant.name}",
                condition="randomized_labels",
                train_labels=randomized_labels_for_examples(splits["train"], fit_seed + 77, 8),
                dev_labels=randomized_labels_for_examples(splits["dev"], fit_seed + 88, 8),
                message_config=candidate.message_config,
                **repair_kwargs,
            )
            randomized_seconds = time.perf_counter() - t0
            evals = {
                split: stage6._evaluate_split(
                    trainable=trainable,
                    frozen=frozen,
                    randomized=randomized,
                    examples=splits[split],
                    seed=int(seed) + (10_000 if split == "dev" else 20_000),
                    metadata_only_accuracy=metadata_only.get(split),
                    no_gold_audit=no_gold,
                    variant=variant,
                )
                for split in ("dev", "test")
            }
            attention = stage6._capture_attention_audit(trainable, splits["dev"][: min(32, len(splits["dev"]))], int(seed) + 30_000)
            checkpoint_paths = stage6._save_stage6_checkpoints(checkpoint_dir, variant, int(seed), stage, candidate, trainable, frozen, randomized)
            row = {
                "phase": "repair_cheap",
                "seed": int(seed),
                "variant": variant.name,
                "variant_family": variant.family,
                "variant_description": variant.description,
                "repair_config": _repair_kwargs(variant),
                "architecture_config": _candidate_config(candidate),
                "stage_config": asdict(stage),
                "dataset_summary": dataset_summary(splits),
                "split_leakage_audit": stage6._compact_leakage(split_leakage),
                "split_leakage_audit_passes": _leakage_passes(split_leakage),
                "output_leakage_audit": output_leakage,
                "output_leakage_audit_passes": bool(output_leakage.get("passes", False)),
                "no_gold_token_audit": no_gold,
                "dev": evals["dev"],
                "test": evals["test"],
                "dev_delta": float(evals["dev"]["accuracy"]["trainable"] - evals["dev"]["accuracy"]["frozen"]),
                "test_delta": float(evals["test"]["accuracy"]["trainable"] - evals["test"]["accuracy"]["frozen"]),
                "attention_mask_audit": attention,
                "compute": {
                    "trainable_train_seconds": trainable_seconds,
                    "frozen_train_seconds": frozen_seconds,
                    "randomized_train_seconds": randomized_seconds,
                    "total_fit_seconds": trainable_seconds + frozen_seconds + randomized_seconds,
                    "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") and torch.cuda.is_available() else 0,
                    "trainable_param_count": int(trainable.param_count),
                    "frozen_param_count": int(frozen.param_count),
                },
                "trainable_audit": stage6._compact_training_audit(trainable.audit),
                "frozen_audit": stage6._compact_training_audit(frozen.audit),
                "randomized_audit": stage6._compact_training_audit(randomized.audit),
                "checkpoint_paths": checkpoint_paths,
            }
            row["strict_gate_failures"] = _strict_gate_failures(row, split="dev")
            row["integrated_gate_failures"] = list(row["strict_gate_failures"])
            rows.append(row)
            del trainable, frozen, randomized
            _clear_cuda()
    stage6._attach_clone_transitions(rows)
    return rows


def _repair_variants() -> List[RepairVariant]:
    uniform = (
        "candidate_only",
        "role_token_ablation",
        "cross_stream_attention_disabled",
        "hidden_states_shuffled_across_examples",
        "candidate_token_ablation",
    )
    return [
        RepairVariant(
            name=BASELINE,
            family="stage4_clone_anchor",
            coordinator_family="candidate_token_cross_attention",
            description="Unmodified Stage 4 clone anchor under the same fair 50/6 cheap-screen budget.",
        ),
        RepairVariant(
            name="integrated_candidate_guided2_repair_uniform_lr3e4",
            family="repair_candidate_guided",
            coordinator_family="integrated_candidate_guided",
            coordinator_layers=2,
            dropout=0.10,
            repair_uniform_conditions=uniform,
            repair_uniform_weight=0.20,
            repair_uniform_controls_per_batch=2,
            repair_candidate_order_shuffle_weight=0.25,
            description="Original candidate-guided family with train-only uniform penalties on ablations and shuffled candidate-order training.",
        ),
        RepairVariant(
            name="integrated_candidate_guided_late2_repair_lr3e4",
            family="repair_candidate_guided_late_only",
            coordinator_family="integrated_candidate_guided_late",
            coordinator_layers=2,
            dropout=0.10,
            repair_uniform_conditions=uniform,
            repair_uniform_weight=0.20,
            repair_uniform_controls_per_batch=2,
            repair_candidate_order_shuffle_weight=0.25,
            description="Delayed candidate-guided variant: candidates no longer write into role summaries before evidence access.",
        ),
        RepairVariant(
            name="integrated_early_mixing1_isolated_repair_lr3e4",
            family="repair_early_mixing_isolated",
            coordinator_family="integrated_early_mixing_isolated",
            coordinator_layers=2,
            dropout=0.10,
            repair_uniform_conditions=uniform,
            repair_uniform_weight=0.20,
            repair_uniform_controls_per_batch=2,
            repair_candidate_order_shuffle_weight=0.25,
            description="Early-mixing family with an extra integrated layer and train-only ablation/candidate-order repair losses.",
        ),
    ]


def _candidate_for_variant(stage, variant: RepairVariant):
    msg = MessageChannelConfig(
        use_msg_token=False,
        readout_source="pooled",
        use_message_head=False,
        coordinator_family=variant.coordinator_family,
        use_private_cue_aux=False,
        aux_loss_weight=0.0,
        active_message_layers=(-1,),
    )
    coord = _coordinator_config(
        stage,
        family=variant.coordinator_family,
        num_layers=int(variant.coordinator_layers),
        dropout=float(variant.dropout),
    )
    return SimpleNamespace(name=variant.name, description=variant.description, message_config=msg, coordinator_config=coord)


def _repair_kwargs(variant: RepairVariant) -> Dict[str, object]:
    return {
        "repair_uniform_conditions": tuple(variant.repair_uniform_conditions),
        "repair_uniform_weight": float(variant.repair_uniform_weight),
        "repair_uniform_controls_per_batch": int(variant.repair_uniform_controls_per_batch),
        "repair_candidate_order_shuffle_weight": float(variant.repair_candidate_order_shuffle_weight),
    }


def _strict_gate_failures(row: Dict[str, object], split: str) -> List[str]:
    acc = row[split]["accuracy"]
    failures = []
    trainable = float(acc.get("trainable", 0.0))
    frozen = float(acc.get("frozen", 1.0))
    if frozen > FROZEN_MAX:
        failures.append("frozen_high")
    if trainable - frozen < DELTA_MIN:
        failures.append("delta_low")
    for name in STANDARD_CONTROLS:
        value = acc.get(name)
        if isinstance(value, (int, float)) and float(value) > NEAR:
            failures.append(name)
    for name in INVARIANCE_CONTROLS:
        value = acc.get(name)
        if not isinstance(value, (int, float)) or abs(float(value) - trainable) > INV_TOL:
            failures.append(name)
    for name in INTEGRATED_CONTROLS:
        value = acc.get(name)
        if isinstance(value, (int, float)) and float(value) > NEAR:
            failures.append(name)
    mask = (row.get("attention_mask_audit") or {}).get("mask") or {}
    if int(mask.get("router_tokens", 0) or 0) > 0:
        value = acc.get("coordination_token_ablation")
        if isinstance(value, (int, float)) and float(value) > NEAR:
            failures.append("coordination_token_ablation")
    if not stage6._mask_audit_passes(row):
        failures.append("attention_mask_audit")
    if not bool(row.get("split_leakage_audit_passes")) or not bool(row.get("output_leakage_audit_passes")):
        failures.append("leakage_audit")
    if not stage6._trainable_update_passes(row):
        failures.append("trainable_update_audit")
    if not stage6._frozen_update_passes(row):
        failures.append("frozen_comparator_audit")
    return sorted(set(failures))


def _summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    summaries = []
    for variant, variant_rows in sorted(by_variant.items()):
        train = [float(row["dev"]["accuracy"]["trainable"]) for row in variant_rows]
        frozen = [float(row["dev"]["accuracy"]["frozen"]) for row in variant_rows]
        deltas = [float(row["dev_delta"]) for row in variant_rows]
        failures = {str(row["seed"]): list(row.get("strict_gate_failures", [])) for row in variant_rows}
        repaired = bool(len(variant_rows) == 3 and all(not row.get("strict_gate_failures") for row in variant_rows) and variant != BASELINE)
        summaries.append(
            {
                "variant": variant,
                "n_seeds": len(variant_rows),
                "mean_dev_trainable": _mean(train),
                "mean_dev_frozen": _mean(frozen),
                "mean_dev_delta": _mean(deltas),
                "strict_every_seed_pass": repaired,
                "failing_controls_by_seed": failures,
                "control_means": _control_means(variant_rows, split="dev"),
            }
        )
    summaries.sort(key=lambda row: (bool(row["strict_every_seed_pass"]), float(row["mean_dev_delta"])), reverse=True)
    repaired = [row["variant"] for row in summaries if row["strict_every_seed_pass"]]
    return {
        "variant_summaries": summaries,
        "repaired_variants": repaired,
        "repaired_variant_found": bool(repaired),
        "selected_later_phase2_candidate": repaired[0] if repaired else None,
        "selection_rule": "strict every-seed dev controls/audits under fair 50/6 cheap-screen budget",
    }


def _control_means(rows: Sequence[Dict[str, object]], split: str) -> Dict[str, float]:
    names = list(STANDARD_CONTROLS) + list(INVARIANCE_CONTROLS) + list(INTEGRATED_CONTROLS) + ["coordination_token_ablation"]
    out = {}
    for name in names:
        values = [float(row[split]["accuracy"][name]) for row in rows if isinstance(row[split]["accuracy"].get(name), (int, float))]
        if values:
            out[name] = _mean(values)
    return out


def _decision(summary: Dict[str, object]) -> Dict[str, object]:
    found = bool(summary.get("repaired_variant_found", False))
    selected = summary.get("selected_later_phase2_candidate")
    return {
        "repaired_variant_found": found,
        "later_phase2_candidate": selected if found else None,
        "final_decision": (
            f"repaired variant found: {selected}; keep Stage 4 clone mainline until a later Phase 2 run"
            if found
            else "repaired variant found: no; freeze Stage 6 and keep clone/stacking/multi-avenue as mainline"
        ),
        "phase2_ran": False,
        "final_validation_ran": False,
        "integrated_success_claim": False,
    }


def write_outputs(result: Dict[str, object], output_path: Path, controls_path: Path, mask_path: Path, report_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    controls_path.parent.mkdir(parents=True, exist_ok=True)
    controls_path.write_text(json.dumps(stage6._controls_artifact(result), indent=2, sort_keys=True), encoding="utf-8")
    mask_rows = stage6._mask_rows(result)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in mask_rows) + ("\n" if mask_rows else ""), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")


def _render_report(result: Dict[str, object]) -> str:
    summary = result["cheap_screen"]["summary"]
    decision = result["decision"]
    lines = [
        "# Stage 6 Integrated Repair Pass",
        "",
        "## Scope",
        "",
        "- Bounded cheap-screen repair/salvage pass only.",
        "- Phase 2 was not run.",
        "- Final validation was not run.",
        "- Stage 4 `candidate_token_direct_lr3e4_clip1` remains the mainline.",
        "- No controls were weakened or removed.",
        "",
        "## Decision",
        "",
        f"- Repaired variant found: `{bool(decision['repaired_variant_found'])}`",
        f"- Later Phase 2 candidate: `{decision['later_phase2_candidate']}`",
        f"- Final decision: {decision['final_decision']}",
        "",
        "## Variants",
        "",
        "| variant | family | coordinator | layers | repair |",
        "|---|---|---|---:|---|",
    ]
    for variant in result["variant_plan"]:
        repair = {
            "uniform_conditions": variant.get("repair_uniform_conditions", []),
            "uniform_weight": variant.get("repair_uniform_weight", 0.0),
            "candidate_order_shuffle_weight": variant.get("repair_candidate_order_shuffle_weight", 0.0),
            "dropout": variant.get("dropout", 0.0),
        }
        lines.append(f"| {variant['name']} | {variant['family']} | {variant['coordinator_family']} | {variant['coordinator_layers']} | `{json.dumps(repair, sort_keys=True)}` |")
    lines.extend(
        [
            "",
            "## Strict Eligibility",
            "",
            "| variant | seeds | mean train | mean frozen | mean delta | strict every-seed pass | failing controls by seed |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in summary["variant_summaries"]:
        lines.append(
            f"| {row['variant']} | {row['n_seeds']} | {float(row['mean_dev_trainable']):.4f} | {float(row['mean_dev_frozen']):.4f} | "
            f"{float(row['mean_dev_delta']):.4f} | `{bool(row['strict_every_seed_pass'])}` | `{json.dumps(row['failing_controls_by_seed'], sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "## Control Means",
            "",
            "| variant | candidate-only | metadata-only | schema-only | hidden-shuffled | randomized | no-candidate-position | role-ablation | cross-stream-disabled | physical delta ok | candidate-order delta ok |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in summary["variant_summaries"]:
        controls = dict(row.get("control_means", {}))
        train = float(row["mean_dev_trainable"])
        phys = abs(float(controls.get("physical_order_shuffled_roles_preserved", 0.0)) - train) <= INV_TOL
        cand = abs(float(controls.get("candidate_order_shuffled_with_gold_remap", 0.0)) - train) <= INV_TOL
        lines.append(
            f"| {row['variant']} | {float(controls.get('candidate_only', 0.0)):.4f} | {float(controls.get('candidate_metadata_only', 0.0)):.4f} | "
            f"{float(controls.get('schema_only', 0.0)):.4f} | {float(controls.get('hidden_states_shuffled', 0.0)):.4f} | "
            f"{float(controls.get('randomized_labels', 0.0)):.4f} | {float(controls.get('no_candidate_position_shortcut_audit', 0.0)):.4f} | "
            f"{float(controls.get('role_token_ablation', 0.0)):.4f} | {float(controls.get('cross_stream_attention_disabled', 0.0)):.4f} | `{phys}` | `{cand}` |"
        )
    lines.extend(
        [
            "",
            "## Router Status",
            "",
            "- `integrated_router_plus_tokens_late1_lr3e4` was not included. The previous run showed coordination-token ablation failures, and this bounded pass did not add a separate fixed router path.",
            "",
            "## Interpretation",
            "",
            "- A repaired variant requires every seed to pass Stage 4 controls, integrated mask/leakage audits, ablation controls, physical-order invariance, candidate-order remap invariance, low frozen comparator, and meaningful delta.",
            "- Integrated success is not claimed by this repair pass.",
        ]
    )
    return "\n".join(lines) + "\n"


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
