from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.experiments.run_plan_arch1_empirical_architecture_discovery import (
    ACTION_DELTAS,
    FIELD_VOCABS,
    ROLE_IDS,
    FeatureTrainingConfig,
    PlanDatasetConfig,
    PlanExample,
    PlanFitResult,
    PlanLatentVerifier,
    PlanTokenEncoder,
    PlanTrainingConfig,
    PlanVariant,
    _action_counts,
    _accuracy,
    _bigram_predictions,
    _candidate_loss,
    _candidate_pair_artifact_accuracy,
    _candidate_source_metadata_only_accuracy,
    _candidate_tokens,
    _canonical_view,
    _clear_cuda,
    _collision_heuristic_predictions,
    _evidence_tokens,
    _expanded_views,
    _feature_training_config,
    _grad_norm,
    _labels,
    _length_only_predictions,
    _load_config,
    _manhattan,
    _masked_mean,
    _metric_block,
    _micro_gates,
    _pack_tokens,
    _progress_heuristic_predictions,
    _resolve_device,
    _run_overfit_gates,
    _set_seed,
    _softmax_np,
    _top1,
    _training_config,
    _unigram_predictions,
    _variant_plan,
    apply_plan_control,
    build_plan_splits,
    collate_plan_batch,
    fit_plan_verifier,
    predict_plan_logits,
)


BENCHMARK = "plan_arch1_3_transition_token_repair"
DEFAULT_CONFIG = "configs/plan_arch1_3_transition_token_repair.json"
DEFAULT_RESULTS = "results/plan_arch1_3_transition_results.json"
DEFAULT_CONTROLS = "results/plan_arch1_3_controls.json"
DEFAULT_TOKEN_AUDIT = "results/plan_arch1_3_token_visibility_audit.jsonl"
DEFAULT_GRADIENT_AUDIT = "results/plan_arch1_3_gradient_logit_audit.json"
DEFAULT_MICRO = "results/plan_arch1_3_micro_overfit_curves.json"
DEFAULT_HEURISTIC = "results/plan_arch1_3_heuristic_sanity.json"
DEFAULT_ERRORS = "results/plan_arch1_3_error_cases.jsonl"
DEFAULT_COMPUTE = "results/plan_arch1_3_compute_metrics.json"
DEFAULT_REPORT = "reports/PLAN_ARCH1_3_TRANSITION_TOKEN_REPAIR.md"

LEVELS = {
    0: "oracle_success_flag_diagnostic",
    1: "obvious_success_failure_final_visible",
    2: "no_collision_goal_distance_final_visible",
    3: "near_miss_final_visible",
    4: "final_transition_decides_final_visible",
    5: "no_final_state_transition_inference",
}

CLONE_VARIANTS = (
    "transition_tuple_verifier",
    "candidate_token_direct_transition_tokens",
    "simple_cross_attention_verifier_baseline",
)
PRIMARY_HELDOUT_MODELS = ("transition_tuple_verifier", "minimal_transition_cross_attention")
DIAGNOSTIC_MODEL = "engineered_feature_diagnostic_mlp"
HEURISTICS = (
    "final_position_equals_goal",
    "final_distance_to_goal",
    "any_collision_from_visible_rollout",
    "last_action_reaches_goal",
    "rollout_endpoint_validity",
    "goal_distance_improvement",
    "transition_consistency",
)


@dataclass
class MinimalFitResult:
    method: str
    model: nn.Module
    variant: PlanVariant
    audit: Dict[str, object]
    history: List[Dict[str, float]]
    training_time_seconds: float
    param_count: int


class MinimalTransitionCrossAttention(nn.Module):
    """A tiny non-clone verifier over the same token fields used by transition variants."""

    def __init__(self, model_dim: int = 32, num_heads: int = 2, ff_dim: int = 64) -> None:
        super().__init__()
        self.token_encoder = PlanTokenEncoder(model_dim)
        self.query = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim))
        self.attn = nn.MultiheadAttention(model_dim, num_heads, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(nn.Linear(model_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, model_dim))
        self.score_head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, 1))
        self.aux_collision = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
        self.aux_success = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))

    def forward(self, batch: Dict[str, torch.Tensor], hidden_state_shuffle: bool = False, shuffle_seed: int = 0) -> Dict[str, torch.Tensor]:
        evidence = batch["evidence_fields"]
        evidence_mask = batch["evidence_mask"].bool()
        candidate = batch["candidate_fields"]
        candidate_mask = batch["candidate_mask"].bool()
        bsz, cand_count, view_count, evid_tokens, fields = evidence.shape
        cand_tokens = candidate.shape[2]

        evid_emb = self.token_encoder(evidence.reshape(bsz * cand_count * view_count, evid_tokens, fields))
        evid_emb = evid_emb.reshape(bsz, cand_count, view_count * evid_tokens, -1)
        evid_mask = evidence_mask.reshape(bsz, cand_count, view_count * evid_tokens)
        if hidden_state_shuffle and bsz > 1:
            generator = torch.Generator(device=evid_emb.device)
            generator.manual_seed(int(shuffle_seed))
            order = torch.randperm(bsz, generator=generator, device=evid_emb.device)
            evid_emb = evid_emb.index_select(0, order)
            evid_mask = evid_mask.index_select(0, order)

        cand_emb = self.token_encoder(candidate.reshape(bsz * cand_count, cand_tokens, fields))
        cand_mask = candidate_mask.reshape(bsz * cand_count, cand_tokens)
        pooled_candidate = _masked_mean(cand_emb, cand_mask)
        query = self.query(pooled_candidate).unsqueeze(1)
        flat_evid = evid_emb.reshape(bsz * cand_count, view_count * evid_tokens, -1)
        flat_mask = evid_mask.reshape(bsz * cand_count, view_count * evid_tokens)
        attended, _weights = self.attn(query, flat_evid, flat_evid, key_padding_mask=~flat_mask, need_weights=False)
        fused = pooled_candidate + attended.squeeze(1)
        fused = fused + self.ff(self.norm(fused))
        pooled = fused.reshape(bsz, cand_count, -1)
        return {
            "logits": self.score_head(pooled).squeeze(-1),
            "aux_collision": self.aux_collision(pooled).squeeze(-1),
            "aux_success": self.aux_success(pooled).squeeze(-1),
        }


class EngineeredFeatureDiagnosticMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        bsz, cand_count, dim = features.shape
        return self.net(features.reshape(bsz * cand_count, dim)).reshape(bsz, cand_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PLAN-ARCH-1.3 transition-token verifier repair.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--token-audit-output", default=DEFAULT_TOKEN_AUDIT)
    parser.add_argument("--gradient-audit-output", default=DEFAULT_GRADIENT_AUDIT)
    parser.add_argument("--micro-output", default=DEFAULT_MICRO)
    parser.add_argument("--heuristic-output", default=DEFAULT_HEURISTIC)
    parser.add_argument("--error-output", default=DEFAULT_ERRORS)
    parser.add_argument("--compute-output", default=DEFAULT_COMPUTE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = _default_config(_load_config(Path(args.config)))
    if args.device:
        config["device"] = str(args.device)
    result = run_plan_arch1_3_transition_repair(config)
    _write_outputs(
        result,
        Path(args.output),
        Path(args.controls_output),
        Path(args.token_audit_output),
        Path(args.gradient_audit_output),
        Path(args.micro_output),
        Path(args.heuristic_output),
        Path(args.error_output),
        Path(args.compute_output),
        Path(args.report),
    )


def run_plan_arch1_3_transition_repair(config: Dict[str, object]) -> Dict[str, object]:
    started = time.perf_counter()
    seeds = [int(seed) for seed in config.get("seeds", [0])]
    device = _resolve_device(str(config.get("device", "cpu")))
    base_dataset = _dataset_config(config.get("dataset", {}))
    training = _training_config(config.get("training", {}))
    micro_training = _training_config({**dict(config.get("training", {})), **dict(config.get("micro_overfit_training", {}))})
    minimal_training = _training_config({**dict(config.get("training", {})), **dict(config.get("minimal_training", {}))})
    diagnostic_training = _training_config({**dict(config.get("training", {})), **dict(config.get("diagnostic_training", {}))})
    feature_training = _feature_training_config(config.get("feature_training", {}))
    audit_examples_per_level = int(config.get("audit_examples_per_level", 3))
    levels_requested = [int(level) for level in config.get("levels", [1, 2, 3, 4, 5])]
    variants_by_name = {variant.name: variant for variant in _variant_plan(None)}
    transition_variant = _transition_variant_for_level(variants_by_name["transition_tuple_verifier"], 1)

    print(f"plan-arch1.3: device={device} seeds={seeds} levels={levels_requested}")
    micro_rows = _run_transition_micro_overfit(base_dataset, micro_training, seeds[0] if seeds else 0, device, transition_variant)
    micro_pass = bool(micro_rows and all(bool(row.get("pass")) for row in micro_rows))
    gradient_audit = _gradient_logit_audit(base_dataset, micro_training, seeds[0] if seeds else 0, device, transition_variant)
    gradient_pass = bool(gradient_audit.get("gate_pass", False))

    levels_to_build = sorted(set(level for level in levels_requested if level in {1, 2, 3, 4}))
    if 5 in levels_requested:
        levels_to_build.append(5)
    heuristic_rows: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []
    controls: List[Dict[str, object]] = []
    ablations: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    compute: List[Dict[str, object]] = []
    token_audit_rows: List[Dict[str, object]] = []
    final_state_cases: List[Dict[str, object]] = []
    split_cache: Dict[Tuple[int, int], Dict[str, List[PlanExample]]] = {}

    if micro_pass and gradient_pass:
        for level in levels_to_build:
            if level == 5 and not _easier_levels_passed(rows):
                compute.append(
                    {
                        "benchmark": BENCHMARK,
                        "level": 5,
                        "level_name": LEVELS[5],
                        "status": "skipped",
                        "reason": "Levels 1-4 did not pass transition-only held-out gates.",
                    }
                )
                continue
            for seed in seeds:
                level_dataset = replace(base_dataset, learnability_level=level)
                splits = build_plan_splits(level_dataset, seed=seed)
                split_cache[(level, seed)] = splits
                final_visible = _level_has_final_state(level)
                heuristic_rows.append(_heuristic_sanity_row(level, LEVELS[level], seed, splits["test"], final_visible))
                variant = _transition_variant_for_level(variants_by_name["transition_tuple_verifier"], level)
                level_rows = _run_level_transition_models(
                    level,
                    LEVELS[level],
                    seed,
                    splits,
                    variant,
                    training,
                    minimal_training,
                    diagnostic_training,
                    feature_training,
                    device,
                    bool(config.get("run_auxiliary_level1_variants", True)),
                    variants_by_name,
                )
                rows.extend(level_rows["rows"])
                controls.extend(level_rows["controls"])
                ablations.extend(level_rows["ablations"])
                errors.extend(level_rows["errors"])
                compute.extend(level_rows["compute"])
                transition_row = next((row for row in level_rows["rows"] if row["variant"] == "transition_tuple_verifier"), None)
                transition_logits = transition_row.get("trainable_logits") if transition_row else None
                token_audit_rows.extend(
                    _token_visibility_audit_rows(
                        level,
                        LEVELS[level],
                        seed,
                        splits["test"][:audit_examples_per_level],
                        variant,
                        transition_logits,
                    )
                )
                _clear_cuda()
    else:
        heuristic_rows.append({"status": "skipped", "reason": "micro-overfit or gradient/logit gate failed"})
        compute.append(
            {
                "benchmark": BENCHMARK,
                "status": "stopped_before_heldout",
                "micro_overfit_pass": micro_pass,
                "gradient_logit_gate_pass": gradient_pass,
            }
        )

    if bool(config.get("run_oracle_final_state_diagnostic", True)):
        final_state_cases.extend(_run_final_state_evidence_cases(base_dataset, training, seeds[:1], device, variants_by_name))

    rows_for_output = [_drop_large_row_fields(row) for row in rows]
    controls_summary = _controls_summary(controls, ablations)
    summary = _summary(rows_for_output, controls, ablations, micro_rows, gradient_audit, heuristic_rows, final_state_cases)
    compute.append(
        {
            "benchmark": BENCHMARK,
            "status": "completed",
            "total_runtime_seconds": float(time.perf_counter() - started),
            "device": device,
            "medium_validation_launched": False,
            "row_count": len(rows_for_output),
        }
    )
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "scope": "Transition-token verifier representation repair only; no broad architecture search and no medium validation.",
            "device": device,
            "claim_boundary": "No planning claim and no architecture improvement claim.",
            "medium_validation_launched": False,
            "tested_clone_variants": list(CLONE_VARIANTS),
            "tested_primary_models": list(PRIMARY_HELDOUT_MODELS),
            "diagnostic_models": [DIAGNOSTIC_MODEL, "oracle_success_flag_sanity"],
        },
        "dataset_config": asdict(base_dataset),
        "training_config": asdict(training),
        "micro_training_config": asdict(micro_training),
        "minimal_training_config": asdict(minimal_training),
        "feature_training_config": asdict(feature_training),
        "micro_overfit": micro_rows,
        "gradient_logit_audit": gradient_audit,
        "heuristic_sanity": heuristic_rows,
        "token_visibility_audit": token_audit_rows,
        "rows": rows_for_output,
        "controls": controls,
        "control_summary": controls_summary,
        "ablations": ablations,
        "error_cases": errors,
        "compute_metrics": compute,
        "final_state_evidence_cases": final_state_cases,
        "summary": summary,
    }


def _run_level_transition_models(
    level: int,
    level_name: str,
    seed: int,
    splits: Dict[str, List[PlanExample]],
    transition_variant: PlanVariant,
    training: PlanTrainingConfig,
    minimal_training: PlanTrainingConfig,
    diagnostic_training: PlanTrainingConfig,
    feature_training: FeatureTrainingConfig,
    device: str,
    run_auxiliary_level1_variants: bool,
    variants_by_name: Dict[str, PlanVariant],
) -> Dict[str, List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    controls: List[Dict[str, object]] = []
    ablations: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    compute: List[Dict[str, object]] = []
    final_visible = _level_has_final_state(level)

    clone_names = ["transition_tuple_verifier"]
    if int(level) == 1 and run_auxiliary_level1_variants:
        clone_names.extend(["candidate_token_direct_transition_tokens", "simple_cross_attention_verifier_baseline"])

    baseline_cache: Dict[int, float] = {}
    for name in clone_names:
        clone_variant = _transition_variant_for_level(variants_by_name[name], level)
        start = time.perf_counter()
        print(f"plan-arch1.3 level={level} variant={name} seed={seed}: clone trainable/frozen")
        trainable = fit_plan_verifier(
            splits["train"],
            splits["dev"],
            clone_variant,
            replace(training, objective=clone_variant.objective),
            seed + 10_001,
            device,
            True,
            f"plan_arch1_3_trainable__level_{level}__{clone_variant.name}",
        )
        frozen = fit_plan_verifier(
            splits["train"],
            splits["dev"],
            clone_variant,
            replace(training, objective=clone_variant.objective),
            seed + 10_001,
            device,
            False,
            f"plan_arch1_3_frozen__level_{level}__{clone_variant.name}",
        )
        train_logits = predict_plan_logits(trainable.model, splits["test"], clone_variant, training.batch_size, device)
        frozen_logits = predict_plan_logits(frozen.model, splits["test"], clone_variant, training.batch_size, device)
        labels = _labels(splits["test"])
        train_metric = _metric_block(train_logits, labels, splits["test"])
        frozen_metric = _metric_block(frozen_logits, labels, splits["test"])
        shortcut_baselines = _shortcut_baselines(splits["test"])
        control_row = _run_clone_controls(trainable.model, clone_variant, splits["test"], train_logits, labels, training.batch_size, device, seed)
        control_row.update({"level": level, "level_name": level_name, "model_family": "clone"})
        ablation_rows = _run_generic_ablations(
            clone_variant.name,
            "clone",
            level,
            level_name,
            seed,
            splits["test"],
            train_logits,
            labels,
            clone_variant,
            lambda controlled, variant, condition, view_mask: predict_plan_logits(
                trainable.model,
                controlled,
                variant,
                training.batch_size,
                device,
                condition=condition,
                seed=seed + 61_000,
                view_mask=view_mask,
            ),
        )
        elapsed = time.perf_counter() - start
        row = {
            "benchmark": BENCHMARK,
            "level": level,
            "level_name": level_name,
            "variant": clone_variant.name,
            "model_family": "clone",
            "seed": seed,
            "status": "completed",
            "include_final_state": bool(clone_variant.include_final_state),
            "trainable": train_metric,
            "frozen": frozen_metric,
            "delta_trainable_minus_frozen": float(train_metric["top1"] - frozen_metric["top1"]),
            "shortcut_baselines": shortcut_baselines,
            "clean_shortcut_best": _clean_shortcut_best(shortcut_baselines),
            "control_pass": control_row["control_pass"],
            "trainable_audit": trainable.audit,
            "frozen_audit": frozen.audit,
            "param_count": int(trainable.param_count),
            "training_time_seconds": float(elapsed),
            "trainable_logits": train_logits.tolist(),
            "medium_validation_launched": False,
        }
        rows.append(row)
        controls.append(control_row)
        ablations.extend(ablation_rows)
        errors.extend(_error_rows(level, level_name, clone_variant.name, seed, splits["test"], train_logits, frozen_logits, shortcut_baselines, limit=20))
        compute.append(
            {
                "benchmark": BENCHMARK,
                "level": level,
                "level_name": level_name,
                "variant": clone_variant.name,
                "model_family": "clone",
                "seed": seed,
                "train_examples": len(splits["train"]),
                "dev_examples": len(splits["dev"]),
                "test_examples": len(splits["test"]),
                "candidate_count": len(splits["test"][0].candidates) if splits["test"] else 0,
                "param_count": int(trainable.param_count),
                "training_time_seconds": float(elapsed),
            }
        )
        if name == "transition_tuple_verifier":
            baseline_cache[seed] = float(train_metric["top1"])

    minimal = fit_minimal_transition_model(
        splits["train"],
        splits["dev"],
        transition_variant,
        minimal_training,
        seed + 30_001,
        device,
        f"minimal_transition_cross_attention__level_{level}",
    )
    min_logits = predict_minimal_logits(minimal.model, splits["test"], transition_variant, minimal_training.batch_size, device)
    labels = _labels(splits["test"])
    min_metric = _metric_block(min_logits, labels, splits["test"])
    min_controls = _run_minimal_controls(minimal.model, transition_variant, splits["test"], min_logits, labels, minimal_training.batch_size, device, seed)
    min_controls.update({"level": level, "level_name": level_name, "model_family": "minimal_non_clone"})
    min_ablations = _run_generic_ablations(
        "minimal_transition_cross_attention",
        "minimal_non_clone",
        level,
        level_name,
        seed,
        splits["test"],
        min_logits,
        labels,
        transition_variant,
        lambda controlled, variant, condition, view_mask: predict_minimal_logits(
            minimal.model,
            controlled,
            variant,
            minimal_training.batch_size,
            device,
            condition=condition,
            seed=seed + 62_000,
            view_mask=view_mask,
        ),
    )
    rows.append(
        {
            "benchmark": BENCHMARK,
            "level": level,
            "level_name": level_name,
            "variant": "minimal_transition_cross_attention",
            "model_family": "minimal_non_clone",
            "seed": seed,
            "status": "completed",
            "include_final_state": final_visible,
            "trainable": min_metric,
            "frozen": None,
            "delta_trainable_minus_frozen": None,
            "shortcut_baselines": _shortcut_baselines(splits["test"]),
            "clean_shortcut_best": _clean_shortcut_best(_shortcut_baselines(splits["test"])),
            "control_pass": min_controls["control_pass"],
            "trainable_audit": minimal.audit,
            "param_count": int(minimal.param_count),
            "training_time_seconds": float(minimal.training_time_seconds),
            "medium_validation_launched": False,
        }
    )
    controls.append(min_controls)
    ablations.extend(min_ablations)
    errors.extend(_error_rows(level, level_name, "minimal_transition_cross_attention", seed, splits["test"], min_logits, None, _shortcut_baselines(splits["test"]), limit=20))
    compute.append(
        {
            "benchmark": BENCHMARK,
            "level": level,
            "level_name": level_name,
            "variant": "minimal_transition_cross_attention",
            "model_family": "minimal_non_clone",
            "seed": seed,
            "train_examples": len(splits["train"]),
            "dev_examples": len(splits["dev"]),
            "test_examples": len(splits["test"]),
            "candidate_count": len(splits["test"][0].candidates) if splits["test"] else 0,
            "param_count": int(minimal.param_count),
            "training_time_seconds": float(minimal.training_time_seconds),
        }
    )

    diag = fit_engineered_diagnostic_mlp(splits["train"], splits["dev"], final_visible, diagnostic_training, seed + 40_001)
    diag_logits = predict_engineered_diagnostic_logits(diag.model, splits["test"], final_visible)
    rows.append(
        {
            "benchmark": BENCHMARK,
            "level": level,
            "level_name": level_name,
            "variant": DIAGNOSTIC_MODEL,
            "model_family": "diagnostic_engineered_features",
            "seed": seed,
            "status": "completed",
            "diagnostic_only": True,
            "include_final_state": final_visible,
            "trainable": _metric_block(diag_logits, labels, splits["test"]),
            "frozen": None,
            "delta_trainable_minus_frozen": None,
            "shortcut_baselines": _shortcut_baselines(splits["test"]),
            "clean_shortcut_best": _clean_shortcut_best(_shortcut_baselines(splits["test"])),
            "control_pass": {"diagnostic_only": True, "overall": False},
            "trainable_audit": diag.audit,
            "param_count": int(diag.param_count),
            "training_time_seconds": float(diag.training_time_seconds),
            "medium_validation_launched": False,
        }
    )
    compute.append(
        {
            "benchmark": BENCHMARK,
            "level": level,
            "level_name": level_name,
            "variant": DIAGNOSTIC_MODEL,
            "model_family": "diagnostic_engineered_features",
            "seed": seed,
            "train_examples": len(splits["train"]),
            "dev_examples": len(splits["dev"]),
            "test_examples": len(splits["test"]),
            "candidate_count": len(splits["test"][0].candidates) if splits["test"] else 0,
            "param_count": int(diag.param_count),
            "training_time_seconds": float(diag.training_time_seconds),
        }
    )
    return {"rows": rows, "controls": controls, "ablations": ablations, "errors": errors, "compute": compute}


def fit_minimal_transition_model(
    train_examples: Sequence[PlanExample],
    dev_examples: Sequence[PlanExample],
    variant: PlanVariant,
    training: PlanTrainingConfig,
    seed: int,
    device: str,
    method: str,
) -> MinimalFitResult:
    _set_seed(seed)
    model = MinimalTransitionCrossAttention(model_dim=int(variant.model_dim), num_heads=int(variant.num_heads), ff_dim=int(variant.ff_dim)).to(device)
    initial = _flat_model_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.lr), weight_decay=float(training.weight_decay))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_dev = -1.0
    bad = 0
    history: List[Dict[str, float]] = []
    grad_norms: List[float] = []
    start = time.perf_counter()
    for epoch in range(int(training.epochs)):
        model.train()
        losses = []
        order = np.random.default_rng(seed + epoch).permutation(len(train_examples)).tolist()
        for batch_ids in _chunks(order, int(training.batch_size)):
            batch_examples = [train_examples[i] for i in batch_ids]
            batch = collate_plan_batch(batch_examples, variant, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = _candidate_loss(out, batch, training)
            loss.backward()
            grad_norms.append(_grad_norm(list(model.parameters())))
            if float(training.gradient_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.gradient_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_logits = predict_minimal_logits(model, dev_examples, variant, training.batch_size, device)
        dev_acc = _top1(dev_logits, _labels(dev_examples))
        history.append({"epoch": float(epoch), "loss": _mean(losses), "dev_top1": float(dev_acc)})
        if dev_acc > best_dev:
            best_dev = float(dev_acc)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= int(training.patience):
            break
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    final = _flat_model_params(model)
    return MinimalFitResult(
        method=method,
        model=model,
        variant=variant,
        audit={
            "method": method,
            "initialization_seed": int(seed),
            "non_clone_baseline": True,
            "parameter_delta": _l2_delta(initial, final),
            "grad_norm_mean": _mean(grad_norms),
            "trainable_parameters_changed": bool(_l2_delta(initial, final) > 0.0),
            "received_gradients": bool(_mean(grad_norms) > 0.0),
        },
        history=history,
        training_time_seconds=float(time.perf_counter() - start),
        param_count=sum(p.numel() for p in model.parameters()),
    )


def predict_minimal_logits(
    model: nn.Module,
    examples: Sequence[PlanExample],
    variant: PlanVariant,
    batch_size: int,
    device: str,
    condition: str = "none",
    seed: int = 0,
    view_mask: str | None = None,
) -> np.ndarray:
    model.eval()
    eval_variant = variant
    if condition == "role_order_shuffle":
        views = list(_expanded_views(variant))
        rng = np.random.default_rng(seed)
        rng.shuffle(views)
        eval_variant = replace(variant, view_names=tuple(views), avenues_per_view=1)
    logits = []
    with torch.no_grad():
        for offset, batch_examples in enumerate(_batched(examples, int(batch_size))):
            batch = collate_plan_batch(batch_examples, eval_variant, device, view_mask=view_mask)
            out = model(batch, hidden_state_shuffle=condition == "hidden_state_shuffle", shuffle_seed=seed + offset)
            logits.append(out["logits"].detach().cpu().numpy())
    return np.concatenate(logits, axis=0) if logits else np.zeros((0, 0), dtype=np.float32)


def fit_engineered_diagnostic_mlp(
    train_examples: Sequence[PlanExample],
    dev_examples: Sequence[PlanExample],
    final_state_visible: bool,
    training: PlanTrainingConfig,
    seed: int,
) -> MinimalFitResult:
    _set_seed(seed)
    train_features, train_labels = _engineered_feature_batch(train_examples, final_state_visible)
    dev_features, dev_labels = _engineered_feature_batch(dev_examples, final_state_visible)
    model = EngineeredFeatureDiagnosticMLP(train_features.shape[-1], hidden_dim=32)
    initial = _flat_model_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.lr), weight_decay=float(training.weight_decay))
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_dev = -1.0
    bad = 0
    history: List[Dict[str, float]] = []
    grad_norms: List[float] = []
    start = time.perf_counter()
    for epoch in range(int(training.epochs)):
        losses = []
        order = np.random.default_rng(seed + epoch).permutation(len(train_examples)).tolist()
        for batch_ids in _chunks(order, int(training.batch_size)):
            features = train_features[batch_ids]
            labels = train_labels[batch_ids]
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            grad_norms.append(_grad_norm(list(model.parameters())))
            if float(training.gradient_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.gradient_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            dev_logits = model(dev_features).detach().cpu().numpy()
        dev_acc = _top1(dev_logits, dev_labels.detach().cpu().numpy())
        history.append({"epoch": float(epoch), "loss": _mean(losses), "dev_top1": float(dev_acc)})
        if dev_acc > best_dev:
            best_dev = float(dev_acc)
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= int(training.patience):
            break
    model.load_state_dict(best_state)
    final = _flat_model_params(model)
    return MinimalFitResult(
        method=DIAGNOSTIC_MODEL,
        model=model,
        variant=PlanVariant(name=DIAGNOSTIC_MODEL, family="diagnostic", description="engineered visible heuristic feature diagnostic", view_names=()),
        audit={
            "method": DIAGNOSTIC_MODEL,
            "diagnostic_only": True,
            "heuristic_features_exposed": True,
            "initialization_seed": int(seed),
            "parameter_delta": _l2_delta(initial, final),
            "grad_norm_mean": _mean(grad_norms),
        },
        history=history,
        training_time_seconds=float(time.perf_counter() - start),
        param_count=sum(p.numel() for p in model.parameters()),
    )


def predict_engineered_diagnostic_logits(model: nn.Module, examples: Sequence[PlanExample], final_state_visible: bool) -> np.ndarray:
    features, _labels_t = _engineered_feature_batch(examples, final_state_visible)
    model.eval()
    with torch.no_grad():
        return model(features).detach().cpu().numpy()


def _run_transition_micro_overfit(
    base_dataset: PlanDatasetConfig,
    training: PlanTrainingConfig,
    seed: int,
    device: str,
    variant: PlanVariant,
) -> List[Dict[str, object]]:
    gates = [("4_examples_N2", 2, 4, 0.95), ("16_examples_N4", 4, 16, 0.90), ("64_examples_N8", 8, 64, 0.85)]
    rows = _run_overfit_gates(
        variant,
        replace(base_dataset, learnability_level=1),
        training,
        seed,
        device,
        gates,
    )
    for row in rows:
        row.update(
            {
                "benchmark": BENCHMARK,
                "stage": "micro_overfit_before_heldout",
                "level": 1,
                "level_name": LEVELS[1],
                "variant": "transition_tuple_verifier",
                "target_interpretation": "Stop before held-out if this gate fails.",
            }
        )
    return rows


def _gradient_logit_audit(
    base_dataset: PlanDatasetConfig,
    training: PlanTrainingConfig,
    seed: int,
    device: str,
    variant: PlanVariant,
) -> Dict[str, object]:
    dataset = replace(base_dataset, learnability_level=1, num_candidates=4, train_examples=32, dev_examples=8, test_examples=8)
    splits = build_plan_splits(dataset, seed=seed + 800)
    sample = splits["train"][: min(16, len(splits["train"]))]
    _set_seed(seed + 50_001)
    probe = PlanLatentVerifier(variant).to(device)
    probe.configure_shared_trainable(True)
    before_logits = predict_plan_logits(probe, sample, variant, training.batch_size, device)
    batch = collate_plan_batch(sample[: min(8, len(sample))], variant, device)
    probe.train()
    for parameter in probe.parameters():
        parameter.grad = None
    out = probe(batch)
    loss = _candidate_loss(out, batch, training)
    loss.backward()
    role_grad = probe.token_encoder.embeddings[0].weight.grad
    role_grad_norms = {
        "transition_token_encoder": _role_embedding_grad_norm(role_grad, "transition"),
        "goal_token_encoder": _role_embedding_grad_norm(role_grad, "goal"),
        "obstacle_token_encoder": _role_embedding_grad_norm(role_grad, "obstacle"),
        "candidate_token_encoder": _role_embedding_grad_norm(role_grad, "action") + _role_embedding_grad_norm(role_grad, "summary"),
    }
    module_grad_norms = {
        "candidate_attention": _named_grad_norm(probe, ("blocks.", ".attn")),
        "scoring_head": _named_grad_norm(probe, ("score_head.",)),
        "shared_model": _grad_norm([parameter for _name, parameter in probe.shared_parameter_items()]),
    }

    fit = fit_plan_verifier(
        sample,
        sample,
        variant,
        replace(training, batch_size=min(int(training.batch_size), len(sample)), patience=int(training.epochs)),
        seed + 50_001,
        device,
        True,
        "plan_arch1_3_gradient_logit_audit_train",
    )
    after_logits = predict_plan_logits(fit.model, sample, variant, training.batch_size, device)
    labels = _labels(sample)
    stats_before = _logit_stats(before_logits, labels)
    stats_after = _logit_stats(after_logits, labels)
    changed_l2 = float(np.linalg.norm(after_logits - before_logits)) if before_logits.shape == after_logits.shape else 0.0
    fail_reasons = []
    if float(stats_after["candidate_logit_std"]) <= 1e-5:
        fail_reasons.append("logits_almost_constant")
    if changed_l2 <= 1e-5:
        fail_reasons.append("candidate_logits_unchanged_after_training")
    if float(role_grad_norms["transition_token_encoder"]) <= 0.0:
        fail_reasons.append("transition_encoder_no_gradients")
    if float(module_grad_norms["candidate_attention"]) <= 0.0:
        fail_reasons.append("candidate_attention_no_gradients")
    if len(stats_after["prediction_distribution"]) <= 1 and len(sample[0].candidates) > 1:
        fail_reasons.append("always_predicts_same_candidate_slot")
    return {
        "benchmark": BENCHMARK,
        "variant": variant.name,
        "seed": int(seed),
        "level": 1,
        "sample_count": len(sample),
        "logits_before_training": _matrix_preview(before_logits),
        "logits_after_training": _matrix_preview(after_logits),
        "logit_change_l2": changed_l2,
        "before_training_stats": stats_before,
        "after_training_stats": stats_after,
        "gradient_norms": {**role_grad_norms, **module_grad_norms},
        "fit_history": fit.history,
        "fit_audit": fit.audit,
        "fail_reasons": fail_reasons,
        "gate_pass": not fail_reasons,
    }


def _heuristic_sanity_row(level: int, level_name: str, seed: int, examples: Sequence[PlanExample], final_state_visible: bool) -> Dict[str, object]:
    labels = _labels(examples)
    heuristic_metrics: Dict[str, object] = {}
    score_snapshots = []
    for name in HEURISTICS:
        scores = np.asarray([_heuristic_scores(example, name, final_state_visible) for example in examples], dtype=np.float32)
        preds = np.argmax(scores, axis=1) if scores.size else np.zeros(0, dtype=np.int64)
        heuristic_metrics[name] = {
            "top1": _accuracy(preds, labels),
            "prediction_histogram": _histogram(preds.tolist()),
        }
    for example in examples[: min(5, len(examples))]:
        score_snapshots.append(
            {
                "example_id": example.id,
                "label": int(example.label),
                "scores": {name: _heuristic_scores(example, name, final_state_visible) for name in HEURISTICS},
            }
        )
    best_name = max(HEURISTICS, key=lambda item: float(heuristic_metrics[item]["top1"])) if examples else "none"
    return {
        "benchmark": BENCHMARK,
        "level": level,
        "level_name": level_name,
        "seed": seed,
        "example_count": len(examples),
        "candidate_count": len(examples[0].candidates) if examples else 0,
        "final_state_visible_to_model": final_state_visible,
        "heuristics": heuristic_metrics,
        "best_heuristic": best_name,
        "best_top1": float(heuristic_metrics[best_name]["top1"]) if examples else 0.0,
        "snapshots": score_snapshots,
        "diagnostic_only_not_model_input": True,
    }


def _token_visibility_audit_rows(
    level: int,
    level_name: str,
    seed: int,
    examples: Sequence[PlanExample],
    variant: PlanVariant,
    logits: object | None,
) -> List[Dict[str, object]]:
    logits_np = np.asarray(logits, dtype=np.float32) if logits is not None else np.zeros((len(examples), 0), dtype=np.float32)
    rows = []
    for ex_index, example in enumerate(examples):
        scores = logits_np[ex_index].tolist() if logits_np.size and ex_index < logits_np.shape[0] else []
        pred = int(np.argmax(scores)) if scores else None
        candidates = []
        transition_signatures = []
        all_candidate_assertions = []
        for cand_index, actions in enumerate(example.candidates):
            transition_info = _candidate_transition_token_info(example, cand_index, variant)
            candidate_token_info = _candidate_token_info(example, cand_index, variant)
            transition_tokens = transition_info["transition_tokens"]
            transition_signatures.append(json.dumps(transition_tokens, sort_keys=True))
            alignment = _candidate_evidence_alignment(example, cand_index, transition_tokens)
            assertions = {
                "nonzero_transition_tokens": len(transition_tokens) > 0,
                "goal_tokens_visible": len(transition_info["goal_tokens"]) > 0,
                "obstacle_tokens_visible": len(transition_info["obstacle_tokens"]) > 0 or len(example.obstacles) == 0,
                "candidate_token_can_attend_to_transition_tokens": candidate_token_info["nonzero_candidate_tokens"] and len(transition_tokens) > 0 and transition_info["transition_mask_sum"] > 0,
                "candidate_token_can_attend_to_goal_tokens": candidate_token_info["nonzero_candidate_tokens"] and len(transition_info["goal_tokens"]) > 0,
                "candidate_token_can_attend_to_obstacle_tokens": candidate_token_info["nonzero_candidate_tokens"] and (len(transition_info["obstacle_tokens"]) > 0 or len(example.obstacles) == 0),
                "padding_masks_keep_decisive_tokens": transition_info["decisive_tokens_unmasked"],
                "candidate_evidence_alignment": alignment["aligned"],
            }
            all_candidate_assertions.append(assertions)
            states = list(example.rollouts[cand_index])
            candidates.append(
                {
                    "candidate_id": int(cand_index),
                    "action_sequence": [int(action) for action in actions],
                    "pre_final_position": list(states[-2]) if len(states) > 1 else list(states[-1]),
                    "final_action": int(actions[-1]) if actions else None,
                    "final_position_if_visible": list(states[-1]) if bool(variant.include_final_state) and states else None,
                    "goal_position": list(example.goal),
                    "obstacle_positions": [list(cell) for cell in example.obstacles],
                    "rollout_positions": [list(pos) for pos in states],
                    "transition_tokens": transition_tokens,
                    "token_masks": transition_info["token_masks"],
                    "candidate_tokens": candidate_token_info["candidate_tokens"],
                    "candidate_token_mask_sum": candidate_token_info["candidate_mask_sum"],
                    "candidate_evidence_alignment": alignment,
                    "assertions": assertions,
                    "score": float(scores[cand_index]) if scores and cand_index < len(scores) else None,
                }
            )
        remap = _candidate_order_remap_assertion(example, seed + 2000)
        role_remap = _role_order_remap_assertion(example, variant, seed + 3000)
        assertions = _merge_candidate_assertions(all_candidate_assertions)
        assertions.update(
            {
                "transition_tokens_differ_across_candidates": len(set(transition_signatures)) > 1,
                "candidate_order_remap_changes_labels_correctly": remap["pass"],
                "role_order_remap_permuted_consistently": role_remap["pass"],
            }
        )
        rows.append(
            {
                "type": "plan_arch1_3_token_visibility_audit",
                "benchmark": BENCHMARK,
                "level": level,
                "level_name": level_name,
                "seed": seed,
                "example_id": example.id,
                "gold_index": int(example.label),
                "predicted_index": pred,
                "scores": scores,
                "view_names": list(_expanded_views(variant)),
                "include_final_state": bool(variant.include_final_state),
                "candidates": candidates,
                "candidate_order_remap": remap,
                "role_order_remap": role_remap,
                "assertions": assertions,
                "pass": all(bool(value) for value in assertions.values()),
            }
        )
    return rows


def _run_final_state_evidence_cases(
    base_dataset: PlanDatasetConfig,
    training: PlanTrainingConfig,
    seeds: Sequence[int],
    device: str,
    variants_by_name: Dict[str, PlanVariant],
) -> List[Dict[str, object]]:
    rows = []
    for seed in seeds:
        oracle_variant = replace(variants_by_name["oracle_success_flag_sanity"], model_dim=32, num_heads=2, ff_dim=64)
        oracle_dataset = replace(base_dataset, learnability_level=0, train_examples=24, dev_examples=12, test_examples=24)
        splits = build_plan_splits(oracle_dataset, seed=seed + 900)
        fit = fit_plan_verifier(splits["train"], splits["dev"], oracle_variant, replace(training, epochs=max(4, int(training.epochs)), patience=max(4, int(training.patience))), seed + 90_001, device, True, "oracle_success_flag_diagnostic")
        logits = predict_plan_logits(fit.model, splits["test"], oracle_variant, training.batch_size, device)
        labels = _labels(splits["test"])
        controls = _run_clone_controls(fit.model, oracle_variant, splits["test"], logits, labels, training.batch_size, device, seed + 900)
        rows.append(
            {
                "case": "A_oracle_success_flag",
                "diagnostic_only": True,
                "seed": int(seed),
                "top1": float(_top1(logits, labels)),
                "controls_overall": bool(controls["control_pass"]["overall"]),
                "interpretation": "Should solve, but is not claimable because the success flag is an oracle diagnostic.",
            }
        )
        final_variant = _transition_variant_for_level(variants_by_name["transition_tuple_verifier"], 1)
        no_final_variant = _transition_variant_for_level(variants_by_name["transition_tuple_verifier"], 5)
        rows.append(
            {
                "case": "B_final_state_visible",
                "diagnostic_only": False,
                "seed": int(seed),
                "variant": final_variant.name,
                "include_final_state": bool(final_variant.include_final_state),
                "interpretation": "Measured in held-out transition rows; no oracle success flag included.",
            }
        )
        rows.append(
            {
                "case": "C_no_final_state",
                "diagnostic_only": False,
                "seed": int(seed),
                "variant": no_final_variant.name,
                "include_final_state": bool(no_final_variant.include_final_state),
                "interpretation": "Run only after Levels 1-4 pass; otherwise skipped by the 1.3 gate.",
            }
        )
    return rows


def _run_clone_controls(
    model: PlanLatentVerifier,
    variant: PlanVariant,
    examples: Sequence[PlanExample],
    clean_logits: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: str,
    seed: int,
) -> Dict[str, object]:
    return _run_model_controls(
        variant.name,
        "clone",
        examples,
        clean_logits,
        labels,
        variant,
        lambda controlled, eval_variant, condition, view_mask: predict_plan_logits(
            model,
            controlled,
            eval_variant,
            batch_size,
            device,
            condition=condition,
            seed=seed + 44_000,
            view_mask=view_mask,
        ),
        seed,
    )


def _run_minimal_controls(
    model: nn.Module,
    variant: PlanVariant,
    examples: Sequence[PlanExample],
    clean_logits: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: str,
    seed: int,
) -> Dict[str, object]:
    return _run_model_controls(
        "minimal_transition_cross_attention",
        "minimal_non_clone",
        examples,
        clean_logits,
        labels,
        variant,
        lambda controlled, eval_variant, condition, view_mask: predict_minimal_logits(
            model,
            controlled,
            eval_variant,
            batch_size,
            device,
            condition=condition,
            seed=seed + 45_000,
            view_mask=view_mask,
        ),
        seed,
    )


def _run_model_controls(
    variant_name: str,
    model_family: str,
    examples: Sequence[PlanExample],
    clean_logits: np.ndarray,
    labels: np.ndarray,
    variant: PlanVariant,
    predict_fn: Callable[[Sequence[PlanExample], PlanVariant, str, str | None], np.ndarray],
    seed: int,
) -> Dict[str, object]:
    chance = 1.0 / max(1, len(examples[0].candidates) if examples else 1)
    controls: Dict[str, object] = {
        "benchmark": BENCHMARK,
        "variant": variant_name,
        "model_family": model_family,
        "seed": int(seed),
        "chance": float(chance),
        "clean": _metric_block(clean_logits, labels, examples),
        "candidate_source_metadata_only_accuracy": _candidate_source_metadata_only_accuracy(examples),
        "candidate_pair_artifact_baseline": _candidate_pair_artifact_accuracy(examples),
        "length_only": _accuracy(_length_only_predictions(examples), labels),
        "unigram_action_stat": _accuracy(_unigram_predictions(examples), labels),
        "bigram_action_stat": _accuracy(_bigram_predictions(examples), labels),
    }
    for name in (
        "candidate_only",
        "state_plan_mismatch",
        "goal_shuffle",
        "rollout_mismatch",
        "candidate_order_shuffle_with_gold_remap",
        "candidate_identity_shuffle",
        "candidate_list_composition_control",
        "randomized_labels",
        "final_state_mismatch",
    ):
        controlled = apply_plan_control(examples, name, seed + 17_000 + len(controls))
        logits = predict_fn(controlled, variant, "none", None)
        controls[name] = _metric_block(logits, _labels(controlled), controlled)
    role_variant = replace(variant, view_names=_shuffled_views(variant, seed + 44_000), avenues_per_view=1)
    role_logits = predict_fn(examples, role_variant, "none", None)
    hidden_logits = predict_fn(examples, variant, "hidden_state_shuffle", None)
    controls["role_order_shuffle"] = _metric_block(role_logits, labels, examples)
    controls["hidden_state_shuffle"] = _metric_block(hidden_logits, labels, examples)
    clean_top1 = float(_top1(clean_logits, labels))
    shortcut_values = [
        float(controls["candidate_only"]["top1"]),
        float(controls["length_only"]),
        float(controls["unigram_action_stat"]),
        float(controls["bigram_action_stat"]),
        float(controls["candidate_source_metadata_only_accuracy"]),
        float(controls["candidate_pair_artifact_baseline"]),
    ]
    controls["candidate_order_invariance_delta"] = abs(clean_top1 - float(controls["candidate_order_shuffle_with_gold_remap"]["top1"]))
    controls["candidate_identity_invariance_delta"] = abs(clean_top1 - float(controls["candidate_identity_shuffle"]["top1"]))
    controls["role_order_invariance_delta"] = abs(clean_top1 - float(controls["role_order_shuffle"]["top1"]))
    controls["control_pass"] = {
        "candidate_only_near_chance": float(controls["candidate_only"]["top1"]) <= chance + 0.10,
        "length_only_near_chance": float(controls["length_only"]) <= chance + 0.10,
        "action_unigram_near_chance": float(controls["unigram_action_stat"]) <= chance + 0.10,
        "action_bigram_near_chance": float(controls["bigram_action_stat"]) <= chance + 0.10,
        "candidate_source_metadata_near_chance": float(controls["candidate_source_metadata_only_accuracy"]) <= chance + 0.10,
        "candidate_pair_artifact_near_chance": float(controls["candidate_pair_artifact_baseline"]) <= chance + 0.10,
        "shortcut_baselines_near_chance": max(shortcut_values) <= chance + 0.10,
        "state_plan_mismatch_collapses": float(controls["state_plan_mismatch"]["top1"]) <= chance + 0.15,
        "goal_shuffle_collapses": float(controls["goal_shuffle"]["top1"]) <= chance + 0.15,
        "rollout_mismatch_collapses": float(controls["rollout_mismatch"]["top1"]) <= chance + 0.15,
        "candidate_list_composition_collapses": float(controls["candidate_list_composition_control"]["top1"]) <= chance + 0.15,
        "final_state_mismatch_collapses": float(controls["final_state_mismatch"]["top1"]) <= chance + 0.15,
        "candidate_order_remap_invariance": float(controls["candidate_order_invariance_delta"]) <= 0.08,
        "candidate_identity_shuffle_invariance": float(controls["candidate_identity_invariance_delta"]) <= 0.08,
        "role_order_invariance": float(controls["role_order_invariance_delta"]) <= 0.08,
        "hidden_state_shuffle_collapses": float(controls["hidden_state_shuffle"]["top1"]) <= max(chance + 0.20, clean_top1 - 0.05),
        "randomized_labels_collapses": float(controls["randomized_labels"]["top1"]) <= chance + 0.15,
    }
    controls["control_pass"]["overall"] = all(bool(value) for value in controls["control_pass"].values())
    return controls


def _run_generic_ablations(
    variant_name: str,
    model_family: str,
    level: int,
    level_name: str,
    seed: int,
    examples: Sequence[PlanExample],
    clean_logits: np.ndarray,
    labels: np.ndarray,
    variant: PlanVariant,
    predict_fn: Callable[[Sequence[PlanExample], PlanVariant, str, str | None], np.ndarray],
) -> List[Dict[str, object]]:
    clean_top1 = _top1(clean_logits, labels)
    rows = []
    for view in sorted(set(_canonical_view(view) for view in _expanded_views(variant))):
        logits = predict_fn(examples, variant, "none", view)
        top1 = _top1(logits, labels)
        rows.append(
            {
                "benchmark": BENCHMARK,
                "level": level,
                "level_name": level_name,
                "variant": variant_name,
                "model_family": model_family,
                "seed": int(seed),
                "view": view,
                "clean_top1": float(clean_top1),
                "ablated_top1": float(top1),
                "drop": float(clean_top1 - top1),
                "meaningful_drop": bool(clean_top1 - top1 >= 0.05),
            }
        )
    return rows


def _heuristic_scores(example: PlanExample, name: str, final_state_visible: bool) -> List[float]:
    return [_heuristic_score_for_candidate(example, cand_index, name, final_state_visible) for cand_index in range(len(example.candidates))]


def _heuristic_score_for_candidate(example: PlanExample, cand_index: int, name: str, final_state_visible: bool) -> float:
    info = _visible_endpoint_info(example, cand_index, final_state_visible)
    start_distance = float(_manhattan(example.start, example.goal))
    if name == "final_position_equals_goal":
        return 1.0 if info["endpoint"] == example.goal and not info["final_collision"] else 0.0
    if name == "final_distance_to_goal":
        return -float(_manhattan(info["endpoint"], example.goal))
    if name == "any_collision_from_visible_rollout":
        return -float(info["visible_collision_count"])
    if name == "last_action_reaches_goal":
        return 1.0 if info["inferred_endpoint"] == example.goal and not info["final_collision"] else 0.0
    if name == "rollout_endpoint_validity":
        return 1.0 if _in_bounds(info["endpoint"], example.grid_size) and info["endpoint"] not in set(example.obstacles) and not info["final_collision"] else 0.0
    if name == "goal_distance_improvement":
        return start_distance - float(_manhattan(info["endpoint"], example.goal))
    if name == "transition_consistency":
        return float(_transition_consistency_score(example, cand_index, final_state_visible))
    raise ValueError(f"unknown heuristic: {name}")


def _visible_endpoint_info(example: PlanExample, cand_index: int, final_state_visible: bool) -> Dict[str, object]:
    states = list(example.rollouts[cand_index])
    actions = list(example.candidates[cand_index])
    collisions = list(example.collisions[cand_index])
    pre_final = states[-2] if len(states) > 1 else states[-1]
    final_action = int(actions[-1]) if actions else 0
    inferred, final_collision = _apply_action(example.grid_size, pre_final, final_action, set(example.obstacles))
    endpoint = states[-1] if final_state_visible and states else inferred
    visible_collision_count = int(sum(collisions if final_state_visible else collisions[:-1] + [int(final_collision)]))
    return {
        "endpoint": endpoint,
        "inferred_endpoint": inferred,
        "final_collision": bool(final_collision),
        "visible_collision_count": visible_collision_count,
    }


def _transition_consistency_score(example: PlanExample, cand_index: int, final_state_visible: bool) -> float:
    states = list(example.rollouts[cand_index])
    actions = list(example.candidates[cand_index])
    collisions = list(example.collisions[cand_index])
    max_step = min(len(actions), max(0, len(states) - 1))
    if not final_state_visible:
        max_step = max(0, max_step - 1)
    if max_step == 0:
        return 0.0
    ok = 0
    obstacle_set = set(example.obstacles)
    for step in range(max_step):
        expected, collide = _apply_action(example.grid_size, states[step], int(actions[step]), obstacle_set)
        if expected == states[step + 1] and int(collide) == int(collisions[step]):
            ok += 1
    return ok / max_step


def _engineered_feature_batch(examples: Sequence[PlanExample], final_state_visible: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = [[_engineered_features(example, cand_index, final_state_visible) for cand_index in range(len(example.candidates))] for example in examples]
    return torch.as_tensor(rows, dtype=torch.float32), torch.as_tensor([example.label for example in examples], dtype=torch.long)


def _engineered_features(example: PlanExample, cand_index: int, final_state_visible: bool) -> List[float]:
    grid_scale = float(max(1, example.grid_size - 1))
    scores = {name: _heuristic_score_for_candidate(example, cand_index, name, final_state_visible) for name in HEURISTICS}
    actions = list(example.candidates[cand_index])
    action_counts = np.asarray(_action_counts(actions), dtype=np.float32) / max(1, len(actions))
    info = _visible_endpoint_info(example, cand_index, final_state_visible)
    endpoint = info["endpoint"]
    pre_final = example.rollouts[cand_index][-2] if len(example.rollouts[cand_index]) > 1 else example.rollouts[cand_index][-1]
    return [
        float(scores["final_position_equals_goal"]),
        -float(scores["final_distance_to_goal"]) / grid_scale,
        -float(scores["any_collision_from_visible_rollout"]) / max(1, len(actions)),
        float(scores["last_action_reaches_goal"]),
        float(scores["rollout_endpoint_validity"]),
        float(scores["goal_distance_improvement"]) / grid_scale,
        float(scores["transition_consistency"]),
        float(_manhattan(pre_final, example.goal)) / grid_scale,
        float(_manhattan(endpoint, example.goal)) / grid_scale,
        float(actions[-1] if actions else 0) / 3.0,
        *action_counts.tolist(),
    ]


def _shortcut_baselines(examples: Sequence[PlanExample]) -> Dict[str, float]:
    labels = _labels(examples)
    return {
        "random": 1.0 / max(1, len(examples[0].candidates) if examples else 1),
        "candidate_only_index0": _accuracy(np.zeros(len(examples), dtype=np.int64), labels),
        "length_only": _accuracy(_length_only_predictions(examples), labels),
        "unigram_action_stat": _accuracy(_unigram_predictions(examples), labels),
        "bigram_action_stat": _accuracy(_bigram_predictions(examples), labels),
        "candidate_source_metadata_only": _candidate_source_metadata_only_accuracy(examples),
        "candidate_pair_artifact_baseline": _candidate_pair_artifact_accuracy(examples),
        "rollout_collision_heuristic": _accuracy(_collision_heuristic_predictions(examples), labels),
        "progress_heuristic": _accuracy(_progress_heuristic_predictions(examples), labels),
    }


def _clean_shortcut_best(baselines: Dict[str, object]) -> float:
    keys = (
        "candidate_only_index0",
        "length_only",
        "unigram_action_stat",
        "bigram_action_stat",
        "candidate_source_metadata_only",
        "candidate_pair_artifact_baseline",
    )
    values = [float(baselines.get(key, 0.0)) for key in keys if isinstance(baselines.get(key, 0.0), (int, float))]
    return max(values) if values else 0.0


def _candidate_transition_token_info(example: PlanExample, cand_index: int, variant: PlanVariant) -> Dict[str, object]:
    views = _expanded_views(variant)
    transition_tokens: List[List[int]] = []
    goal_tokens: List[List[int]] = []
    obstacle_tokens: List[List[int]] = []
    token_masks: Dict[str, object] = {}
    decisive_unmasked = True
    for view_id, view in enumerate(views):
        tokens = _evidence_tokens(example, cand_index, view, view_id, variant)
        fields, mask = _pack_tokens(tokens, 72)
        canonical = _canonical_view(view)
        masked_tokens = fields[mask].tolist()
        token_masks[canonical] = {"mask_sum": int(mask.sum()), "limit": int(mask.shape[0])}
        if canonical == "transition_tuple":
            transition_tokens.extend(masked_tokens)
            decisive_unmasked = decisive_unmasked and int(mask.sum()) >= min(len(tokens), 72)
        elif canonical == "goal":
            goal_tokens.extend(masked_tokens)
        elif canonical == "obstacle":
            obstacle_tokens.extend(masked_tokens)
    return {
        "transition_tokens": transition_tokens,
        "goal_tokens": goal_tokens,
        "obstacle_tokens": obstacle_tokens,
        "token_masks": token_masks,
        "transition_mask_sum": int(token_masks.get("transition_tuple", {}).get("mask_sum", 0)),
        "decisive_tokens_unmasked": bool(decisive_unmasked),
    }


def _candidate_token_info(example: PlanExample, cand_index: int, variant: PlanVariant) -> Dict[str, object]:
    tokens = _candidate_tokens(example, cand_index, example.candidates[cand_index], variant)
    fields, mask = _pack_tokens(tokens, 48)
    return {"candidate_tokens": fields[mask].tolist(), "candidate_mask_sum": int(mask.sum()), "nonzero_candidate_tokens": bool(mask.sum() > 0)}


def _candidate_evidence_alignment(example: PlanExample, cand_index: int, transition_tokens: Sequence[Sequence[int]]) -> Dict[str, object]:
    actions = list(example.candidates[cand_index])
    action_by_step = {int(token[4]): int(token[3]) for token in transition_tokens[0::2] if len(token) >= 5}
    checks = []
    for step, action in enumerate(actions[: len(action_by_step)]):
        checks.append(bool(action_by_step.get(step) == int(action)))
    return {
        "checked_steps": len(checks),
        "aligned_steps": int(sum(checks)),
        "aligned": bool(checks and all(checks)),
    }


def _candidate_order_remap_assertion(example: PlanExample, seed: int) -> Dict[str, object]:
    remapped = apply_plan_control([example], "candidate_order_shuffle_with_gold_remap", seed)[0]
    changed = int(remapped.label) != int(example.label) or any(tuple(a) != tuple(b) for a, b in zip(remapped.candidates, example.candidates))
    same_set = sorted(map(tuple, remapped.candidates)) == sorted(map(tuple, example.candidates))
    gold_ok = bool(remapped.goal_reached[remapped.label])
    return {"pass": bool(changed and same_set and gold_ok), "old_label": int(example.label), "new_label": int(remapped.label), "changed": bool(changed), "same_candidate_set": bool(same_set), "gold_success_at_new_label": gold_ok}


def _role_order_remap_assertion(example: PlanExample, variant: PlanVariant, seed: int) -> Dict[str, object]:
    shuffled = replace(variant, view_names=_shuffled_views(variant, seed), avenues_per_view=1)
    original = _role_counts_for_example(example, variant)
    remapped = _role_counts_for_example(example, shuffled)
    return {"pass": original == remapped, "original_role_counts": original, "remapped_role_counts": remapped, "shuffled_view_names": list(shuffled.view_names)}


def _role_counts_for_example(example: PlanExample, variant: PlanVariant) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for cand_index in range(len(example.candidates)):
        for view_id, view in enumerate(_expanded_views(variant)):
            for token in _evidence_tokens(example, cand_index, view, view_id, variant):
                role = _role_name(int(token[0]))
                counts[role] = counts.get(role, 0) + 1
    return dict(sorted(counts.items()))


def _merge_candidate_assertions(rows: Sequence[Dict[str, bool]]) -> Dict[str, bool]:
    keys = sorted({key for row in rows for key in row})
    return {key: all(bool(row.get(key, False)) for row in rows) for key in keys}


def _error_rows(
    level: int,
    level_name: str,
    variant_name: str,
    seed: int,
    examples: Sequence[PlanExample],
    train_logits: np.ndarray,
    frozen_logits: np.ndarray | None,
    baselines: Dict[str, object],
    limit: int,
) -> List[Dict[str, object]]:
    labels = _labels(examples)
    train_pred = np.argmax(train_logits, axis=1)
    frozen_pred = np.argmax(frozen_logits, axis=1) if frozen_logits is not None and frozen_logits.size else np.full_like(train_pred, -1)
    rows = []
    for index, example in enumerate(examples):
        if len(rows) >= int(limit):
            break
        category = None
        if train_pred[index] == labels[index] and frozen_pred[index] >= 0 and frozen_pred[index] != labels[index]:
            category = "trainable_right_frozen_wrong"
        elif train_pred[index] != labels[index] and frozen_pred[index] == labels[index]:
            category = "frozen_right_trainable_wrong"
        elif train_pred[index] != labels[index]:
            category = "model_wrong"
        if category is None:
            continue
        rows.append(
            {
                "type": "plan_arch1_3_error_case",
                "level": int(level),
                "level_name": level_name,
                "category": category,
                "variant": variant_name,
                "seed": int(seed),
                "example_id": example.id,
                "start": list(example.start),
                "goal": list(example.goal),
                "obstacles": [list(cell) for cell in example.obstacles],
                "gold_candidate_index": int(example.label),
                "predicted_candidate_index": int(train_pred[index]),
                "frozen_candidate_index": int(frozen_pred[index]) if frozen_pred[index] >= 0 else None,
                "candidate_probabilities": _softmax_np(train_logits[index]).tolist(),
                "selected_plan_success": bool(example.goal_reached[int(train_pred[index])]),
                "gold_plan_success": bool(example.goal_reached[int(example.label)]),
                "baseline_snapshot": {key: value for key, value in baselines.items() if isinstance(value, (int, float, str))},
            }
        )
    return rows


def _summary(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    ablations: Sequence[Dict[str, object]],
    micro_rows: Sequence[Dict[str, object]],
    gradient_audit: Dict[str, object],
    heuristic_rows: Sequence[Dict[str, object]],
    final_state_cases: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    transition_rows = [row for row in rows if row.get("variant") == "transition_tuple_verifier"]
    minimal_rows = [row for row in rows if row.get("variant") == "minimal_transition_cross_attention"]
    control_by_key = {(row.get("level"), row.get("variant"), row.get("seed")): row for row in controls}
    ablation_by_key: Dict[Tuple[object, object, object], bool] = {}
    for row in ablations:
        key = (row.get("level"), row.get("variant"), row.get("seed"))
        ablation_by_key[key] = bool(ablation_by_key.get(key, False) or row.get("meaningful_drop", False))
    level_pass = {}
    for level in sorted({int(row["level"]) for row in rows if isinstance(row.get("level"), int)}):
        t_rows = [row for row in transition_rows if int(row["level"]) == level]
        m_rows = [row for row in minimal_rows if int(row["level"]) == level]
        level_pass[str(level)] = {
            "transition_tuple_top1_mean": _mean([float(row["trainable"]["top1"]) for row in t_rows]),
            "minimal_top1_mean": _mean([float(row["trainable"]["top1"]) for row in m_rows]),
            "transition_controls_all": all(bool(control_by_key.get((row.get("level"), row.get("variant"), row.get("seed")), {}).get("control_pass", {}).get("overall", False)) for row in t_rows) if t_rows else False,
            "minimal_controls_all": all(bool(control_by_key.get((row.get("level"), row.get("variant"), row.get("seed")), {}).get("control_pass", {}).get("overall", False)) for row in m_rows) if m_rows else False,
        }
    medium_trigger = _medium_trigger(rows, controls, ablations)
    best_heuristics = {
        str(row.get("level")): {"best_heuristic": row.get("best_heuristic"), "best_top1": row.get("best_top1")}
        for row in heuristic_rows
        if isinstance(row, dict) and "level" in row
    }
    return {
        "micro_overfit_pass": bool(micro_rows and all(bool(row.get("pass")) for row in micro_rows)),
        "gradient_logit_gate_pass": bool(gradient_audit.get("gate_pass", False)),
        "gradient_logit_fail_reasons": gradient_audit.get("fail_reasons", []),
        "heuristic_best_by_level": best_heuristics,
        "heldout_level_summary": level_pass,
        "minimal_model_learns": any(float(row.get("trainable", {}).get("top1", 0.0)) >= 0.80 for row in minimal_rows),
        "clone_transition_learns": any(float(row.get("trainable", {}).get("top1", 0.0)) >= 0.80 for row in transition_rows),
        "final_state_cases": final_state_cases,
        "medium_trigger": medium_trigger,
        "medium_validation_launched": False,
        "ready_for_medium_validation": bool(medium_trigger.get("overall", False)),
        "problem_diagnosis": _problem_diagnosis(rows, controls, ablations, micro_rows, gradient_audit),
    }


def _medium_trigger(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]], ablations: Sequence[Dict[str, object]]) -> Dict[str, object]:
    transition_rows = [row for row in rows if row.get("variant") == "transition_tuple_verifier" and int(row.get("level", -1)) in {1, 2}]
    by_level: Dict[int, List[Dict[str, object]]] = {}
    for row in transition_rows:
        by_level.setdefault(int(row["level"]), []).append(row)
    control_by_key = {(row.get("level"), row.get("variant"), row.get("seed")): row for row in controls}
    transition_ablation_hurts = any(row.get("variant") == "transition_tuple_verifier" and row.get("view") == "transition_tuple" and bool(row.get("meaningful_drop")) for row in ablations)
    level_1_pass = _level_clean_gate(by_level.get(1, []), control_by_key)
    level_2_pass = _level_clean_gate(by_level.get(2, []), control_by_key)
    deltas = [float(row.get("delta_trainable_minus_frozen", 0.0)) for row in transition_rows if row.get("delta_trainable_minus_frozen") is not None]
    seeds = sorted({int(row["seed"]) for row in transition_rows})
    seed_delta_pass = len(seeds) >= 3 and sum(delta > 0.0 for delta in deltas) >= 2
    mean_delta = _mean(deltas)
    controls_all = all(bool(control_by_key.get((row.get("level"), row.get("variant"), row.get("seed")), {}).get("control_pass", {}).get("overall", False)) for row in transition_rows) if transition_rows else False
    shortcut_near = all(float(row.get("clean_shortcut_best", 1.0)) <= float(row.get("shortcut_baselines", {}).get("random", 0.125)) + 0.10 for row in transition_rows) if transition_rows else False
    return {
        "transition_tuple_micro_overfits": True,
        "level_1_clean_heldout_passes": level_1_pass,
        "level_2_clean_heldout_passes": level_2_pass,
        "three_seed_requirement_evaluable": len(seeds) >= 3,
        "trainable_beats_frozen_on_at_least_2_of_3_seeds": seed_delta_pass,
        "mean_delta": mean_delta,
        "mean_delta_at_least_0_10": mean_delta >= 0.10,
        "shortcut_baselines_near_chance": shortcut_near,
        "controls_pass": controls_all,
        "meaningful_transition_token_ablation_hurts": transition_ablation_hurts,
        "candidate_order_remap_passes": all(bool(control_by_key.get((row.get("level"), row.get("variant"), row.get("seed")), {}).get("control_pass", {}).get("candidate_order_remap_invariance", False)) for row in transition_rows) if transition_rows else False,
        "role_order_remap_passes": all(bool(control_by_key.get((row.get("level"), row.get("variant"), row.get("seed")), {}).get("control_pass", {}).get("role_order_invariance", False)) for row in transition_rows) if transition_rows else False,
        "hidden_state_shuffle_collapses": all(bool(control_by_key.get((row.get("level"), row.get("variant"), row.get("seed")), {}).get("control_pass", {}).get("hidden_state_shuffle_collapses", False)) for row in transition_rows) if transition_rows else False,
        "overall": bool(level_1_pass and level_2_pass and seed_delta_pass and mean_delta >= 0.10 and shortcut_near and controls_all and transition_ablation_hurts),
    }


def _level_clean_gate(rows: Sequence[Dict[str, object]], control_by_key: Dict[Tuple[object, object, object], Dict[str, object]]) -> bool:
    if not rows:
        return False
    return all(
        float(row.get("trainable", {}).get("top1", 0.0)) >= 0.80
        and bool(control_by_key.get((row.get("level"), row.get("variant"), row.get("seed")), {}).get("control_pass", {}).get("overall", False))
        for row in rows
    )


def _problem_diagnosis(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    ablations: Sequence[Dict[str, object]],
    micro_rows: Sequence[Dict[str, object]],
    gradient_audit: Dict[str, object],
) -> str:
    if not micro_rows or not all(bool(row.get("pass")) for row in micro_rows):
        return "scoring, masks, gradients, or tokenization still block micro-overfit"
    if not bool(gradient_audit.get("gate_pass", False)):
        return "gradient/logit behavior remains suspect"
    transition_learns = any(row.get("variant") == "transition_tuple_verifier" and float(row.get("trainable", {}).get("top1", 0.0)) >= 0.80 for row in rows)
    minimal_learns = any(row.get("variant") == "minimal_transition_cross_attention" and float(row.get("trainable", {}).get("top1", 0.0)) >= 0.80 for row in rows)
    controls_fail = any(row.get("variant") == "transition_tuple_verifier" and not bool(row.get("control_pass", {}).get("overall", False)) for row in controls)
    if minimal_learns and not transition_learns:
        return "clone/candidate-query architecture is weaker than the minimal transition baseline"
    if not minimal_learns and not transition_learns:
        return "tokenization/dataset/objective difficulty remains the likely issue"
    if controls_fail:
        return "representation is learnable, but controls/shortcut/mismatch gates are not clean"
    if transition_learns and minimal_learns:
        return "representation is learnable in this repair scope; medium readiness still depends on trigger gates"
    return "inconclusive"


def _controls_summary(controls: Sequence[Dict[str, object]], ablations: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_variant: Dict[str, Dict[str, object]] = {}
    for row in controls:
        variant = str(row.get("variant"))
        by_variant.setdefault(variant, {"rows": 0, "overall_pass": 0})
        by_variant[variant]["rows"] = int(by_variant[variant]["rows"]) + 1
        by_variant[variant]["overall_pass"] = int(by_variant[variant]["overall_pass"]) + int(bool(row.get("control_pass", {}).get("overall", False)))
    return {
        "by_variant": by_variant,
        "transition_token_ablation_rows": [row for row in ablations if row.get("view") == "transition_tuple"],
    }


def _easier_levels_passed(rows: Sequence[Dict[str, object]]) -> bool:
    required = {1, 2, 3, 4}
    passed = set()
    for level in required:
        level_rows = [row for row in rows if row.get("variant") in PRIMARY_HELDOUT_MODELS and int(row.get("level", -1)) == level]
        if level_rows and any(float(row.get("trainable", {}).get("top1", 0.0)) >= 0.80 and bool(row.get("control_pass", {}).get("overall", False)) for row in level_rows):
            passed.add(level)
    return passed == required


def _transition_variant_for_level(variant: PlanVariant, level: int) -> PlanVariant:
    return replace(variant, include_final_state=_level_has_final_state(level), include_outcome_tokens=False, model_dim=32, num_heads=2, ff_dim=64)


def _level_has_final_state(level: int) -> bool:
    return int(level) in {1, 2, 3, 4}


def _dataset_config(data: object) -> PlanDatasetConfig:
    values = dict(data or {})
    allowed = set(PlanDatasetConfig.__dataclass_fields__.keys())
    return PlanDatasetConfig(**{key: value for key, value in values.items() if key in allowed})


def _default_config(config: Dict[str, object]) -> Dict[str, object]:
    base: Dict[str, object] = {
        "device": "cpu",
        "seeds": [0],
        "levels": [1, 2, 3, 4, 5],
        "audit_examples_per_level": 3,
        "run_auxiliary_level1_variants": True,
        "run_oracle_final_state_diagnostic": True,
        "dataset": {
            "grid_size": 8,
            "num_candidates": 8,
            "plan_length": 12,
            "obstacle_count": 1,
            "train_examples": 48,
            "dev_examples": 16,
            "test_examples": 32,
            "max_generation_attempts": 1200,
            "shortcut_pool_attempts": 900,
            "require_exactly_one_success": True,
        },
        "training": {
            "epochs": 8,
            "batch_size": 16,
            "lr": 0.003,
            "weight_decay": 0.0,
            "patience": 8,
            "gradient_clip_norm": 1.0,
        },
        "micro_overfit_training": {
            "epochs": 80,
            "patience": 80,
            "batch_size": 16,
            "lr": 0.003,
            "weight_decay": 0.0,
        },
        "minimal_training": {
            "epochs": 16,
            "patience": 8,
            "batch_size": 16,
            "lr": 0.003,
            "weight_decay": 0.0,
        },
        "diagnostic_training": {
            "epochs": 16,
            "patience": 8,
            "batch_size": 16,
            "lr": 0.003,
            "weight_decay": 0.0,
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


def _deep_update(base: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _write_outputs(
    result: Dict[str, object],
    results_path: Path,
    controls_path: Path,
    token_audit_path: Path,
    gradient_audit_path: Path,
    micro_path: Path,
    heuristic_path: Path,
    error_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    for path in (results_path, controls_path, token_audit_path, gradient_audit_path, micro_path, heuristic_path, error_path, compute_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(_trim_result(result), indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(json.dumps({"controls": result["controls"], "control_summary": result["control_summary"], "summary": result["summary"]}, indent=2, sort_keys=True), encoding="utf-8")
    with token_audit_path.open("w", encoding="utf-8") as handle:
        for row in result["token_visibility_audit"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    gradient_audit_path.write_text(json.dumps(result["gradient_logit_audit"], indent=2, sort_keys=True), encoding="utf-8")
    micro_path.write_text(json.dumps({"micro_overfit": result["micro_overfit"]}, indent=2, sort_keys=True), encoding="utf-8")
    heuristic_path.write_text(json.dumps({"heuristic_sanity": result["heuristic_sanity"]}, indent=2, sort_keys=True), encoding="utf-8")
    with error_path.open("w", encoding="utf-8") as handle:
        for row in result["error_cases"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    compute_path.write_text(json.dumps({"compute_metrics": result["compute_metrics"]}, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(_render_report(result), encoding="utf-8")


def _trim_result(result: Dict[str, object]) -> Dict[str, object]:
    trimmed = dict(result)
    trimmed["token_visibility_audit"] = result.get("token_visibility_audit", [])[:10]
    return trimmed


def _render_report(result: Dict[str, object]) -> str:
    summary = result["summary"]
    rows = result["rows"]
    transition_rows = [row for row in rows if row.get("variant") == "transition_tuple_verifier"]
    minimal_rows = [row for row in rows if row.get("variant") == "minimal_transition_cross_attention"]
    lines = [
        "# PLAN-ARCH-1.3 Transition-Token Verifier Repair",
        "",
        "## Scope",
        "- No planning claim.",
        "- No architecture improvement claim.",
        "- Medium validation was not launched.",
        "- Candidate self-attention, pyramids, stacking, and multi-avenue variants were not tested.",
        "",
        "## Answers",
        f"1. Can deterministic visible heuristics solve each clean ladder level? `{_heuristic_answer(result['heuristic_sanity'])}`.",
        f"2. Are decisive transition/goal/obstacle tokens visible to the model? `{_token_audit_answer(result['token_visibility_audit'])}`.",
        f"3. Can transition_tuple_verifier micro-overfit? `{summary['micro_overfit_pass']}`.",
        f"4. Do candidate logits and gradients behave correctly? `{summary['gradient_logit_gate_pass']}`, failures={summary.get('gradient_logit_fail_reasons', [])}.",
        f"5. Does a minimal transition model learn? `{summary['minimal_model_learns']}`.",
        f"6. Does the clone architecture learn once transition representation is fixed? `{summary['clone_transition_learns']}`.",
        f"7. Does final-state-visible help without oracle flags? `{_final_state_answer(rows, result['final_state_evidence_cases'])}`.",
        f"8. Can no-final transition inference work after easier levels pass? `{_no_final_answer(rows, result['compute_metrics'])}`.",
        f"9. Is the problem representation, architecture, objective, or data difficulty? `{summary['problem_diagnosis']}`.",
        f"10. Is any transition verifier ready for medium validation? `{summary['ready_for_medium_validation']}`.",
        "",
        "## Micro-Overfit",
        "| gate | N | examples | train acc | target | pass |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["micro_overfit"]:
        lines.append(f"| {row['gate']} | {row['candidate_count']} | {row['examples']} | {float(row['train_accuracy']):.4f} | {float(row['threshold']):.4f} | `{bool(row['pass'])}` |")
    lines.extend(
        [
            "",
            "## Held-Out Rows",
            "| level | variant | seed | top1 | frozen | delta | shortcut | controls |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(rows, key=lambda item: (int(item.get("level", 99)), str(item.get("variant")), int(item.get("seed", 0))))[:30]:
        frozen = row.get("frozen")
        frozen_top1 = float(frozen["top1"]) if isinstance(frozen, dict) else 0.0
        delta = row.get("delta_trainable_minus_frozen")
        lines.append(
            f"| {row.get('level')} | {row.get('variant')} | {row.get('seed')} | {float(row.get('trainable', {}).get('top1', 0.0)):.4f} | "
            f"{frozen_top1:.4f} | {float(delta) if isinstance(delta, (int, float)) else 0.0:.4f} | {float(row.get('clean_shortcut_best', 0.0)):.4f} | `{bool(row.get('control_pass', {}).get('overall', False))}` |"
        )
    lines.extend(
        [
            "",
            "## Medium Trigger",
            "```json",
            json.dumps(summary["medium_trigger"], indent=2, sort_keys=True),
            "```",
            "",
            "## Claim Boundary",
            "This is an implementation and representation repair stage only. The outputs do not claim planning ability or an architecture improvement.",
        ]
    )
    return "\n".join(lines) + "\n"


def _heuristic_answer(rows: Sequence[Dict[str, object]]) -> str:
    items = []
    for row in rows:
        if not isinstance(row, dict) or "level" not in row:
            continue
        items.append(f"L{row['level']} {row.get('best_heuristic')}={float(row.get('best_top1', 0.0)):.3f}")
    return "; ".join(items) if items else "not run"


def _token_audit_answer(rows: Sequence[Dict[str, object]]) -> str:
    if not rows:
        return "not run"
    return f"pass={all(bool(row.get('pass')) for row in rows)}, audited_examples={len(rows)}"


def _final_state_answer(rows: Sequence[Dict[str, object]], cases: Sequence[Dict[str, object]]) -> str:
    oracle = next((row for row in cases if row.get("case") == "A_oracle_success_flag"), {})
    visible = [row for row in rows if row.get("variant") == "transition_tuple_verifier" and bool(row.get("include_final_state"))]
    no_final = [row for row in rows if row.get("variant") == "transition_tuple_verifier" and not bool(row.get("include_final_state"))]
    visible_best = max([float(row.get("trainable", {}).get("top1", 0.0)) for row in visible], default=0.0)
    no_final_best = max([float(row.get("trainable", {}).get("top1", 0.0)) for row in no_final], default=0.0)
    return f"oracle_top1={oracle.get('top1', 'not_run')} diagnostic_only=True; final_visible_best={visible_best:.3f}; no_final_best={no_final_best:.3f}"


def _no_final_answer(rows: Sequence[Dict[str, object]], compute: Sequence[Dict[str, object]]) -> str:
    no_final = [row for row in rows if int(row.get("level", -1)) == 5]
    if no_final:
        return f"ran=True best_top1={max(float(row.get('trainable', {}).get('top1', 0.0)) for row in no_final):.3f}"
    skipped = next((row for row in compute if row.get("level") == 5 and row.get("status") == "skipped"), None)
    return f"ran=False reason={skipped.get('reason') if skipped else 'not requested'}"


def _drop_large_row_fields(row: Dict[str, object]) -> Dict[str, object]:
    out = dict(row)
    out.pop("trainable_logits", None)
    return out


def _logit_stats(logits: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    if logits.size == 0:
        return {
            "candidate_logit_mean": 0.0,
            "candidate_logit_std": 0.0,
            "candidate_logit_min": 0.0,
            "candidate_logit_max": 0.0,
            "per_example_candidate_logit_variance_mean": 0.0,
            "gold_vs_best_negative_margin_mean": 0.0,
            "prediction_distribution": {},
            "argmax_accuracy": 0.0,
            "argmin_accuracy": 0.0,
        }
    preds = np.argmax(logits, axis=1)
    argmins = np.argmin(logits, axis=1)
    margins = []
    for idx, label in enumerate(labels):
        wrong = np.delete(logits[idx], int(label))
        margins.append(float(logits[idx, int(label)] - (np.max(wrong) if wrong.size else logits[idx, int(label)])))
    return {
        "candidate_logit_mean": float(np.mean(logits)),
        "candidate_logit_std": float(np.std(logits)),
        "candidate_logit_min": float(np.min(logits)),
        "candidate_logit_max": float(np.max(logits)),
        "per_example_candidate_logit_variance_mean": float(np.mean(np.var(logits, axis=1))),
        "gold_vs_best_negative_margin_mean": float(np.mean(margins)) if margins else 0.0,
        "gold_vs_best_negative_margin_min": float(np.min(margins)) if margins else 0.0,
        "prediction_distribution": _histogram(preds.tolist()),
        "argmax_accuracy": _accuracy(preds, labels),
        "argmin_accuracy": _accuracy(argmins, labels),
    }


def _matrix_preview(values: np.ndarray, rows: int = 8) -> List[List[float]]:
    return [[float(v) for v in row] for row in values[:rows]]


def _role_embedding_grad_norm(role_grad: torch.Tensor | None, role: str) -> float:
    if role_grad is None:
        return 0.0
    idx = int(ROLE_IDS.get(role, -1))
    if idx < 0 or idx >= role_grad.shape[0]:
        return 0.0
    return float(torch.linalg.vector_norm(role_grad[idx].detach().float()).cpu())


def _named_grad_norm(model: nn.Module, parts: Tuple[str, ...]) -> float:
    params = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if all(part in name for part in parts):
            params.append(parameter)
    return _grad_norm(params)


def _flat_model_params(model: nn.Module) -> torch.Tensor:
    tensors = [parameter.detach().float().cpu().reshape(-1) for parameter in model.parameters()]
    return torch.cat(tensors) if tensors else torch.zeros(0)


def _l2_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0 or a.numel() != b.numel():
        return 0.0
    return float(torch.linalg.vector_norm(a - b).item())


def _histogram(values: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _chunks(values: Sequence[int], size: int) -> Iterable[List[int]]:
    for index in range(0, len(values), max(1, int(size))):
        yield list(values[index : index + max(1, int(size))])


def _batched(values: Sequence[PlanExample], size: int) -> Iterable[List[PlanExample]]:
    for index in range(0, len(values), max(1, int(size))):
        yield list(values[index : index + max(1, int(size))])


def _apply_action(grid: int, pos: Tuple[int, int], action: int, obstacles: set[Tuple[int, int]]) -> Tuple[Tuple[int, int], bool]:
    dr, dc = ACTION_DELTAS[int(action)]
    nxt = (pos[0] + dr, pos[1] + dc)
    collide = (not _in_bounds(nxt, grid)) or nxt in obstacles
    return (pos if collide else nxt), bool(collide)


def _in_bounds(cell: Tuple[int, int], grid: int) -> bool:
    return 0 <= int(cell[0]) < int(grid) and 0 <= int(cell[1]) < int(grid)


def _role_name(role_id: int) -> str:
    for name, value in ROLE_IDS.items():
        if int(value) == int(role_id):
            return str(name)
    return f"role_{role_id}"


def _shuffled_views(variant: PlanVariant, seed: int) -> Tuple[str, ...]:
    views = list(_expanded_views(variant))
    rng = np.random.default_rng(seed)
    rng.shuffle(views)
    return tuple(views)


if __name__ == "__main__":
    main()
