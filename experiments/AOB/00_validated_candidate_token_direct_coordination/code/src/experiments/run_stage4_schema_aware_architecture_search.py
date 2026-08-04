from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch

from src.datasets.multiview_code_patch_selection import (
    apply_example_control,
    build_multiview_code_patch_splits,
)
from src.experiments.architecture_search import (
    _agent_config,
    _candidate_config,
    _candidate_specs,
    _coordinator_config,
    _training_config,
)
from src.experiments.real_shared_weight_latent_coordination import fit_latent_system, predict_latent_system
from src.experiments.run_stage3_gpu_hard_validation import (
    ARCHITECTURE,
    _clear_cuda,
    _configure_cuda,
    _save_latent_checkpoint,
)
from src.experiments.run_stage34_candidate_sanitization_diagnostics import _accuracy, _structured_oracle_predictions
from src.experiments.run_stage38_full_hard_validation import (
    _dataset_config_for_stage,
    _labels,
    _stage38_full_config,
    _stage_from_config,
)
from src.experiments.run_stage38_schema_aware_controls import (
    _candidate_evidence_mismatch,
    _candidate_only,
    _schema_only,
    _value_shuffle_within_schema,
    _zero_role_embeddings,
)


DEFAULT_CONFIG = "configs/stage38_full_hard_validation.json"
DEFAULT_STAGE38_RESULTS = "results/stage38_full_hard_validation_results.json"
DEFAULT_OUTPUT = "results/stage4_schema_aware_architecture_search_results.json"
DEFAULT_AUDIT = "results/stage4_schema_aware_architecture_search_audit.jsonl"
DEFAULT_REPORT = "reports/STAGE4_SCHEMA_AWARE_ARCHITECTURE_SEARCH.md"
DEFAULT_CHECKPOINT_DIR = "results/stage4_schema_aware_architecture_search_checkpoints"
NEAR_CHANCE = 0.18


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    description: str
    message_updates: Dict[str, object]
    coordinator_family: str = "cross_attention"
    coordinator_layers: int = 1
    lr: float = 0.001
    epochs: int = 50
    patience: int = 6
    dropout: float = 0.0
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded Stage 4 architecture search on the fixed Stage 3.8 benchmark.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage38-results", default=DEFAULT_STAGE38_RESULTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", default=DEFAULT_AUDIT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-variants", type=int, default=0, help="Optional prefix limit for faster local probes. 0 means all.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.device is not None:
        config["device"] = str(args.device)
        if str(args.device) == "cpu":
            config["require_cuda"] = False
    result = run_stage4_search(
        config=config,
        config_path=config_path,
        stage38_results_path=Path(args.stage38_results),
        checkpoint_dir=Path(args.checkpoint_dir),
        max_variants=int(args.max_variants),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in _audit_rows(result)) + "\n", encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")


def run_stage4_search(
    config: Dict[str, object],
    config_path: Path,
    stage38_results_path: Path,
    checkpoint_dir: Path,
    max_variants: int = 0,
) -> Dict[str, object]:
    config = _stage38_full_config(config)
    device, hardware = _configure_cuda(config)
    stage38 = json.loads(stage38_results_path.read_text(encoding="utf-8"))
    variants = _variant_plan()
    if max_variants > 0:
        variants = variants[:max_variants]
    print(f"stage4: cheap screen starting for {len(variants)} variants")
    cheap_rows = _run_phase(
        phase="cheap",
        config=config,
        stage_updates={"name": "stage4_cheap_screen", "n_train": 512, "n_dev": 256, "n_test": 256, "seeds": [0, 1, 2]},
        variants=variants,
        device=device,
        checkpoint_dir=checkpoint_dir / "cheap",
    )
    cheap_summary = _phase_summary(cheap_rows, min_seed_wins=2, min_delta=0.20)
    medium_rows: List[Dict[str, object]] = []
    medium_summary = {"ran": False, "reason": "no cheap-screen variant passed"}
    top = [row for row in cheap_summary["variant_summaries"] if row["cheap_screen_passed"]]
    top.sort(key=lambda row: float(row["mean_dev_delta"]), reverse=True)
    if top:
        top_names = {str(row["variant"]) for row in top[:3]}
        medium_variants = [variant for variant in variants if variant.name in top_names]
        print(f"stage4: medium validation starting for top variants {sorted(top_names)}")
        medium_rows = _run_phase(
            phase="medium",
            config=config,
            stage_updates={"name": "stage4_medium_validation", "n_train": 1024, "n_dev": 512, "n_test": 512, "seeds": [0, 1, 2, 3, 4]},
            variants=medium_variants,
            device=device,
            checkpoint_dir=checkpoint_dir / "medium",
        )
        medium_summary = _phase_summary(medium_rows, min_seed_wins=4, min_delta=0.20)
        medium_summary["ran"] = True
    return {
        "metadata": {
            "stage": "stage4_schema_aware_architecture_search",
            "created_at_utc": _now(),
            "config_path": str(config_path),
            "stage38_results_path": str(stage38_results_path),
            "device": device,
            "hardware": hardware,
            "fixed_benchmark": "Stage 3.8 schema-aware balanced_categories_v3",
            "dataset_or_control_changes": "none",
            "selection_policy": "architecture selection uses dev metrics and dev controls only; test is logged as smoke sanity",
            "final_validation_run": False,
        },
        "stage38_reference": {
            "summary": stage38.get("summary", {}),
            "pre_run": stage38.get("pre_run", {}).get("summary", {}),
        },
        "variant_plan": [asdict(variant) for variant in variants],
        "cheap_screen": {
            "rows": cheap_rows,
            "summary": cheap_summary,
        },
        "medium_validation": {
            "rows": medium_rows,
            "summary": medium_summary,
        },
        "final_validation": {
            "ran": False,
            "reason": "final validation is only allowed after a medium variant passes; this runner does not launch final validation",
            "reserved_seed_policy": "use seeds [10..19] for final validation if a medium candidate passes",
        },
        "summary": _overall_summary(cheap_summary, medium_summary),
    }


def _run_phase(
    phase: str,
    config: Dict[str, object],
    stage_updates: Dict[str, object],
    variants: Sequence[Variant],
    device: str,
    checkpoint_dir: Path,
) -> List[Dict[str, object]]:
    rows = []
    stage_config = _phase_config(config, stage_updates)
    stage = _stage_from_config(stage_config)
    for variant in variants:
        for seed in stage.seeds:
            print(f"stage4 {phase}: variant={variant.name} seed={seed}")
            _clear_cuda()
            splits = build_multiview_code_patch_splits(_dataset_config_for_stage(stage_config, stage), seed=int(seed), repo_root=Path("."))
            candidate = _candidate_for_variant(stage, variant)
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
                agent_config=_agent_config(stage),
                coordinator_config=candidate.coordinator_config,
                training_config=training,
                num_classes=8,
                seed=int(seed) + 10_101,
                device=device,
                trainable_agent=True,
                method=f"stage4_{phase}_trainable__{variant.name}",
                message_config=candidate.message_config,
            )
            frozen = fit_latent_system(
                train_examples=splits["train"],
                dev_examples=splits["dev"],
                agent_config=_agent_config(stage),
                coordinator_config=candidate.coordinator_config,
                training_config=training,
                num_classes=8,
                seed=int(seed) + 20_201,
                device=device,
                trainable_agent=False,
                method=f"stage4_{phase}_frozen__{variant.name}",
                message_config=candidate.message_config,
            )
            metrics = {
                "dev": _screen_metrics(trainable, frozen, splits["dev"], seed=int(seed)),
                "test_smoke": _screen_metrics(trainable, frozen, splits["test"], seed=int(seed) + 1000),
            }
            paths = _save_phase_checkpoints(checkpoint_dir, phase, variant, int(seed), stage, candidate, trainable, frozen)
            rows.append(
                {
                    "phase": phase,
                    "variant": variant.name,
                    "variant_family": variant.family,
                    "variant_description": variant.description,
                    "seed": int(seed),
                    "stage_config": asdict(stage),
                    "variant_config": asdict(variant),
                    "architecture_config": _candidate_config(candidate),
                    "dev_accuracy": metrics["dev"],
                    "test_smoke_accuracy": metrics["test_smoke"],
                    "dev_delta": float(metrics["dev"]["trainable"] - metrics["dev"]["frozen"]),
                    "test_smoke_delta": float(metrics["test_smoke"]["trainable"] - metrics["test_smoke"]["frozen"]),
                    "trainable_audit": _compact_audit(trainable.audit),
                    "frozen_audit": _compact_audit(frozen.audit),
                    "checkpoint_paths": paths,
                }
            )
            del trainable, frozen
            _clear_cuda()
    return rows


def _screen_metrics(trainable, frozen, examples, seed: int) -> Dict[str, float]:
    y = _labels(examples)
    values = {
        "trainable": _accuracy(predict_latent_system(trainable, examples, "none", seed), y),
        "frozen": _accuracy(predict_latent_system(frozen, examples, "none", seed), y),
        "oracle": _accuracy(_structured_oracle_predictions(examples, [0, 1, 2, 3]), y),
    }
    controls = {
        "candidate_only": _candidate_only(examples),
        "schema_only": _schema_only(examples),
        "view_masked_candidates_visible": apply_example_control(examples, "view_masked", seed=seed + 33_000),
        "value_shuffle_within_schema": _value_shuffle_within_schema(examples, seed + 44_000),
        "candidate_evidence_mismatch": _candidate_evidence_mismatch(examples, seed + 55_000),
    }
    for name, rows in controls.items():
        cy = _labels(rows)
        if name == "candidate_only":
            with _zero_role_embeddings(trainable.system):
                pred = predict_latent_system(trainable, rows, "none", seed)
        else:
            pred = predict_latent_system(trainable, rows, "none", seed)
        values[name] = _accuracy(pred, cy)
    return values


def _candidate_for_variant(stage, variant: Variant):
    locked = next(candidate for candidate in _candidate_specs() if candidate.name == ARCHITECTURE)
    msg = replace(locked.message_config, **variant.message_updates)
    coord = replace(
        _coordinator_config(stage, family=variant.coordinator_family, num_layers=int(variant.coordinator_layers), dropout=float(variant.dropout)),
        family=variant.coordinator_family,
    )
    if msg.coordinator_family != "candidate_query_cross_attention":
        coord = replace(coord, family=msg.coordinator_family)
    return SimpleNamespace(name=variant.name, description=variant.description, message_config=msg, coordinator_config=coord)


def _variant_plan() -> List[Variant]:
    base = {
        "use_message_head": False,
        "use_private_cue_aux": False,
        "aux_loss_weight": 0.0,
        "coordinator_family": "candidate_query_cross_attention",
    }
    return [
        Variant(
            name="candidate_token_direct_lr1e3_clip1",
            family="candidate_conditioned_readout",
            description="Candidate queries attend directly over role token states; pooled clone path only feeds compatibility with existing system.",
            message_updates={**base, "readout_source": "pooled", "coordinator_family": "candidate_token_cross_attention"},
            coordinator_family="candidate_token_cross_attention",
            lr=0.001,
            gradient_clip_norm=1.0,
        ),
        Variant(
            name="candidate_token_direct_lr3e4_clip1",
            family="candidate_conditioned_readout_training",
            description="Candidate-token direct readout with lower LR for stability.",
            message_updates={**base, "readout_source": "pooled", "coordinator_family": "candidate_token_cross_attention"},
            coordinator_family="candidate_token_cross_attention",
            lr=0.0003,
            gradient_clip_norm=1.0,
        ),
        Variant(
            name="topk_k4_last2_lr1e3_clip1",
            family="layer_token_readout",
            description="Top-k no-head readout with k=4 over the last two layers.",
            message_updates={**base, "active_message_readout_type": "topk_attention", "active_message_topk": 4, "active_message_layers": (-2, -1)},
            lr=0.001,
            gradient_clip_norm=1.0,
        ),
        Variant(
            name="multi_query_topk4_lr1e3_clip1",
            family="multi_query_topk_readout",
            description="Four learned queries, per-query top-k=4, no message head.",
            message_updates={**base, "active_message_readout_type": "multi_query_topk", "active_message_num_queries": 4, "active_message_topk": 4},
            lr=0.001,
            gradient_clip_norm=1.0,
        ),
        Variant(
            name="attention_pool_lr1e3_clip1",
            family="layer_token_readout",
            description="Soft token attention over all tokens instead of hard top-k.",
            message_updates={**base, "active_message_readout_type": "attention_pool"},
            lr=0.001,
            gradient_clip_norm=1.0,
        ),
        Variant(
            name="bilinear_topk_lr1e3_clip1",
            family="coordinator_variant",
            description="Bilinear candidate-message scoring with top-k no-head readout.",
            message_updates={**base, "active_message_readout_type": "topk_attention", "coordinator_family": "bilinear_candidate"},
            coordinator_family="bilinear_candidate",
            lr=0.001,
            gradient_clip_norm=1.0,
        ),
        Variant(
            name="locked_topk_lr3e4_clip1",
            family="training_variant",
            description="Locked top-k no-head architecture with lower LR and gradient clipping.",
            message_updates={**base, "active_message_readout_type": "topk_attention", "active_message_topk": 8},
            lr=0.0003,
            gradient_clip_norm=1.0,
        ),
        Variant(
            name="locked_topk_lr3e3_clip1",
            family="training_variant",
            description="Locked top-k no-head architecture with higher LR and gradient clipping.",
            message_updates={**base, "active_message_readout_type": "topk_attention", "active_message_topk": 8},
            lr=0.003,
            gradient_clip_norm=1.0,
        ),
    ]


def _phase_summary(rows: Sequence[Dict[str, object]], min_seed_wins: int, min_delta: float) -> Dict[str, object]:
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    summaries = []
    for variant, variant_rows in sorted(by_variant.items()):
        dev_deltas = [float(row["dev_delta"]) for row in variant_rows]
        dev_train = [float(row["dev_accuracy"]["trainable"]) for row in variant_rows]
        dev_frozen = [float(row["dev_accuracy"]["frozen"]) for row in variant_rows]
        wins = sum(delta > 0.0 for delta in dev_deltas)
        control_means = {
            name: _mean([float(row["dev_accuracy"].get(name, 1.0)) for row in variant_rows])
            for name in ("candidate_only", "schema_only", "view_masked_candidates_visible", "value_shuffle_within_schema", "candidate_evidence_mismatch")
        }
        controls_pass = all(value <= NEAR_CHANCE for value in control_means.values())
        oracle_high = _mean([float(row["dev_accuracy"].get("oracle", 0.0)) for row in variant_rows]) >= 0.90
        variance_ok = pstdev(dev_train) <= 0.35 if len(dev_train) > 1 else True
        passed = bool(len(variant_rows) >= min_seed_wins and wins >= min_seed_wins and _mean(dev_deltas) >= min_delta and controls_pass and oracle_high and variance_ok)
        summaries.append(
            {
                "variant": variant,
                "n_seeds": len(variant_rows),
                "dev_wins_trainable_over_frozen": wins,
                "mean_dev_trainable": _mean(dev_train),
                "mean_dev_frozen": _mean(dev_frozen),
                "mean_dev_delta": _mean(dev_deltas),
                "std_dev_trainable": pstdev(dev_train) if len(dev_train) > 1 else 0.0,
                "control_means": control_means,
                "controls_pass": controls_pass,
                "oracle_high": oracle_high,
                "variance_ok": variance_ok,
                "cheap_screen_passed": passed,
                "medium_passed": passed,
            }
        )
    summaries.sort(key=lambda row: float(row["mean_dev_delta"]), reverse=True)
    return {
        "variant_summaries": summaries,
        "n_variants": len(summaries),
        "n_passed": sum(bool(row["cheap_screen_passed"]) for row in summaries),
        "selection_uses": "dev only",
        "min_seed_wins": int(min_seed_wins),
        "min_mean_delta": float(min_delta),
    }


def _overall_summary(cheap_summary: Dict[str, object], medium_summary: Dict[str, object]) -> Dict[str, object]:
    medium_ran = bool(medium_summary.get("ran", False))
    medium_passed = bool(medium_ran and any(bool(row.get("medium_passed")) for row in medium_summary.get("variant_summaries", [])))
    return {
        "cheap_screen_passed_variants": [
            row["variant"] for row in cheap_summary.get("variant_summaries", []) if row.get("cheap_screen_passed")
        ],
        "medium_validation_ran": medium_ran,
        "medium_passed_variants": [
            row["variant"] for row in medium_summary.get("variant_summaries", []) if row.get("medium_passed")
        ]
        if medium_ran
        else [],
        "final_validation_justified": medium_passed,
        "final_validation_run": False,
    }


def _phase_config(config: Dict[str, object], updates: Dict[str, object]) -> Dict[str, object]:
    out = json.loads(json.dumps(config))
    out["stage"] = {**dict(out["stage"]), **updates, "batch_size": 32, "mixed_precision": "bf16"}
    return _stage38_full_config(out)


def _save_phase_checkpoints(checkpoint_dir: Path, phase: str, variant: Variant, seed: int, stage, candidate, trainable, frozen) -> Dict[str, str]:
    root = checkpoint_dir / variant.name / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "trainable.pt"
    frozen_path = root / "frozen.pt"
    _save_latent_checkpoint(train_path, seed, stage, candidate.name, _candidate_config(candidate), trainable)
    _save_latent_checkpoint(frozen_path, seed, stage, candidate.name, _candidate_config(candidate), frozen)
    return {"trainable": str(train_path), "frozen": str(frozen_path), "phase": phase}


def _compact_audit(audit: Dict[str, object]) -> Dict[str, object]:
    keys = (
        "agent_grad_norm_mean",
        "agent_parameter_delta",
        "coordinator_grad_norm_mean",
        "coordinator_parameter_delta",
        "active_message_readout_grad_norm_mean",
        "active_message_readout_parameter_delta",
        "shared_parameter_identity",
        "uses_message_head",
        "uses_private_cue_aux",
        "active_message_readout_type",
        "active_message_topk",
        "message_coordinator_family",
        "gradient_clip_norm",
    )
    return {key: audit.get(key) for key in keys}


def _audit_rows(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = [{"event": "search_summary", "time_utc": _now(), **result.get("summary", {})}]
    for phase in ("cheap_screen", "medium_validation"):
        for row in result.get(phase, {}).get("rows", []):
            rows.append(
                {
                    "event": "variant_seed_completed",
                    "phase": row["phase"],
                    "variant": row["variant"],
                    "seed": row["seed"],
                    "dev_accuracy": row["dev_accuracy"],
                    "test_smoke_accuracy": row["test_smoke_accuracy"],
                    "dev_delta": row["dev_delta"],
                    "checkpoint_paths": row["checkpoint_paths"],
                    "time_utc": _now(),
                }
            )
    return rows


def _render_report(result: Dict[str, object]) -> str:
    lines = [
        "# Stage 4 Schema-Aware Architecture Search",
        "",
        "## Scope",
        "",
        "- Fixed benchmark: Stage 3.8 schema-aware `balanced_categories_v3`.",
        "- Dataset/control changes: none.",
        "- Role-specific schemas preserved as legitimate evidence.",
        "- `role_embedding_shuffle` remains diagnostic-only and non-gating.",
        "- Selection uses dev metrics and dev controls only; test is smoke sanity only.",
        "- Full 10-seed final validation: not run.",
        "",
        "## Variant Plan",
        "",
        "| variant | family | lr | epochs | clip | description |",
        "|---|---|---:|---:|---:|---|",
    ]
    for variant in result.get("variant_plan", []):
        lines.append(
            f"| {variant['name']} | {variant['family']} | {float(variant['lr']):.4g} | {int(variant['epochs'])} | {float(variant['gradient_clip_norm']):.2f} | {variant['description']} |"
        )
    lines.extend(["", "## Cheap Screen", "", "| rank | variant | seeds | wins | trainable | frozen | delta | controls | pass |", "|---:|---|---:|---:|---:|---:|---:|---|---|"])
    for index, row in enumerate(result.get("cheap_screen", {}).get("summary", {}).get("variant_summaries", []), start=1):
        lines.append(
            f"| {index} | {row['variant']} | {int(row['n_seeds'])} | {int(row['dev_wins_trainable_over_frozen'])} | {float(row['mean_dev_trainable']):.4f} | {float(row['mean_dev_frozen']):.4f} | {float(row['mean_dev_delta']):.4f} | `{bool(row['controls_pass'])}` | `{bool(row['cheap_screen_passed'])}` |"
        )
    lines.extend(["", "## Cheap Per-Seed Rows", "", "| variant | seed | dev train | dev frozen | dev delta | dev cand-only | dev schema | dev masked | dev value-shuffle | dev mismatch | test train | test frozen |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in result.get("cheap_screen", {}).get("rows", []):
        dev = row["dev_accuracy"]
        test = row["test_smoke_accuracy"]
        lines.append(
            f"| {row['variant']} | {row['seed']} | {float(dev['trainable']):.4f} | {float(dev['frozen']):.4f} | {float(row['dev_delta']):.4f} | {float(dev['candidate_only']):.4f} | {float(dev['schema_only']):.4f} | {float(dev['view_masked_candidates_visible']):.4f} | {float(dev['value_shuffle_within_schema']):.4f} | {float(dev['candidate_evidence_mismatch']):.4f} | {float(test['trainable']):.4f} | {float(test['frozen']):.4f} |"
        )
    medium = result.get("medium_validation", {})
    lines.extend(["", "## Medium Validation", ""])
    if not medium.get("summary", {}).get("ran", False):
        lines.append(f"- Not run: `{medium.get('summary', {}).get('reason', 'no medium candidates')}`")
    else:
        lines.append("| variant | seeds | wins | trainable | frozen | delta | controls | pass |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---|")
        for row in medium.get("summary", {}).get("variant_summaries", []):
            lines.append(
                f"| {row['variant']} | {int(row['n_seeds'])} | {int(row['dev_wins_trainable_over_frozen'])} | {float(row['mean_dev_trainable']):.4f} | {float(row['mean_dev_frozen']):.4f} | {float(row['mean_dev_delta']):.4f} | `{bool(row['controls_pass'])}` | `{bool(row['medium_passed'])}` |"
            )
    summary = result.get("summary", {})
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Cheap-screen passed variants: `{summary.get('cheap_screen_passed_variants', [])}`",
            f"- Medium validation ran: `{bool(summary.get('medium_validation_ran', False))}`",
            f"- Medium passed variants: `{summary.get('medium_passed_variants', [])}`",
            f"- Final validation justified: `{bool(summary.get('final_validation_justified', False))}`",
            "- No success claim is made from this search screen.",
        ]
    )
    return "\n".join(lines) + "\n"


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
