from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
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
    selected_plan_metrics,
)


BENCHMARK = "PLAN-1_LATENT_CANDIDATE_PLAN_VERIFIER"
DEFAULT_CONFIG = "configs/stage_plan1_latent_candidate_plan_verifier_smoke.json"
DEFAULT_RESULTS = "results/plan1_results.json"
DEFAULT_CONTROLS = "results/plan1_controls.json"
DEFAULT_LEADERBOARD = "results/plan1_variant_leaderboard.csv"
DEFAULT_FAILURES = "results/plan1_failure_taxonomy.json"
DEFAULT_OVERFIT = "results/plan1_overfit_curves.json"
DEFAULT_ERRORS = "results/plan1_error_cases.jsonl"
DEFAULT_ATTENTION = "results/plan1_attention_summaries.jsonl"
DEFAULT_COMPUTE = "results/plan1_compute_metrics.json"
DEFAULT_REPORT = "reports/STAGE_PLAN1_LATENT_CANDIDATE_PLAN_VERIFIER.md"

PLAN_FIELD_VOCABS = (16, 16, 16, 16, 16, 32, 8, 16, 16, 32)
SUMMARY_ROWCOL = 15
VIEW_NAMES = (
    "state_view",
    "goal_view",
    "obstacle_constraint_view",
    "candidate_action_view",
    "rollout_view",
    "outcome_view",
    "cost_efficiency_view",
    "risk_failure_view",
)
VIEW_ID = {name: index for index, name in enumerate(VIEW_NAMES)}


@dataclass(frozen=True)
class PlanModelConfig:
    variant_family: str = "candidate_plan_direct"
    view_names: Tuple[str, ...] = (
        "state_view",
        "goal_view",
        "obstacle_constraint_view",
        "candidate_action_view",
    )
    interaction_style: str = "late_candidate_query"
    objective: str = "cross_entropy"
    model_dim: int = 48
    num_heads: int = 4
    view_layers: int = 1
    candidate_layers: int = 1
    coordination_blocks: int = 1
    ff_dim: int = 96
    dropout: float = 0.0
    max_view_tokens: int = 40
    max_candidate_tokens: int = 20
    pyramidal_composition: bool = False


@dataclass(frozen=True)
class PlanTrainingConfig:
    epochs: int = 10
    batch_size: int = 16
    lr: float = 0.0008
    weight_decay: float = 0.0001
    patience: int = 4
    gradient_clip_norm: float = 1.0
    objective: str = "cross_entropy"
    margin: float = 0.25


@dataclass(frozen=True)
class PlanVariant:
    name: str
    short_name: str
    description: str
    model: PlanModelConfig
    diagnostic_only: bool = False


@dataclass
class PlanFitResult:
    method: str
    model: "PlanCloneVerifier"
    config: PlanModelConfig
    trainable_shared: bool
    audit: Dict[str, object]
    history: List[Dict[str, float]]
    training_time_seconds: float
    param_count: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage PLAN-1 latent candidate-plan verifier/ranker.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--leaderboard-output", default=DEFAULT_LEADERBOARD)
    parser.add_argument("--failure-output", default=DEFAULT_FAILURES)
    parser.add_argument("--overfit-output", default=DEFAULT_OVERFIT)
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
    result = run_stage_plan1(config, max_variants=int(args.max_variants))
    _write_outputs(
        result,
        output_path=Path(args.output),
        controls_path=Path(args.controls_output),
        leaderboard_path=Path(args.leaderboard_output),
        failure_path=Path(args.failure_output),
        overfit_path=Path(args.overfit_output),
        error_path=Path(args.error_output),
        attention_path=Path(args.attention_output),
        compute_path=Path(args.compute_output),
        report_path=Path(args.report),
    )


def run_stage_plan1(config: Dict[str, object], max_variants: int = 0) -> Dict[str, object]:
    dataset_config = _dataset_config(config.get("dataset", {}))
    training = _training_config(config.get("training", {}))
    overfit_training = _training_config({**dict(config.get("training", {})), **dict(config.get("overfit_training", {}))})
    seeds = [int(value) for value in config.get("seeds", [0])]
    device = _resolve_device(str(config.get("device", "auto")))
    variants = _variant_plan()
    if max_variants > 0:
        variants = variants[:max_variants]
    if int(config.get("max_variants", 0)) > 0:
        variants = variants[: int(config.get("max_variants", 0))]
    enforce_overfit = bool(config.get("enforce_overfit_gates", True))

    all_rows: List[Dict[str, object]] = []
    all_controls: List[Dict[str, object]] = []
    all_overfit: List[Dict[str, object]] = []
    all_errors: List[Dict[str, object]] = []
    all_attention: List[Dict[str, object]] = []
    all_compute: List[Dict[str, object]] = []
    all_leakage: List[Dict[str, object]] = []
    phase0_rows: List[Dict[str, object]] = []

    print(f"plan1: device={device} variants={len(variants)} seeds={seeds}")
    for seed in seeds:
        base_splits = build_plan_gridworld_splits(dataset_config, seed=seed)
        leakage = leakage_audit_rows([example for split_rows in base_splits.values() for example in split_rows])
        all_leakage.extend(leakage)
        phase0 = _phase0_summary(base_splits, leakage, seed)
        phase0_rows.append(phase0)
        print(
            "plan1 seed={seed}: examples train/dev/test={train}/{dev}/{test} candidate-only-mlp={cand:.4f}".format(
                seed=seed,
                train=len(base_splits["train"]),
                dev=len(base_splits["dev"]),
                test=len(base_splits["test"]),
                cand=float(phase0["candidate_action_stats_mlp_dev_top1"]),
            )
        )
        for variant in variants:
            print(f"plan1 variant={variant.name} seed={seed}: micro-overfit gates")
            overfit_rows = _run_overfit_gates(variant, dataset_config, overfit_training, seed, device)
            all_overfit.extend(overfit_rows)
            gates_pass = all(bool(row["pass"]) for row in overfit_rows if bool(row.get("gate_required", True)))
            row_base = {
                "benchmark": BENCHMARK,
                "seed": seed,
                "variant": variant.name,
                "short_name": variant.short_name,
                "variant_description": variant.description,
                "model_config": asdict(variant.model),
                "training_config": asdict(training),
                "diagnostic_only": bool(variant.diagnostic_only),
                "overfit_gates_pass": bool(gates_pass),
            }
            if enforce_overfit and not gates_pass:
                all_rows.append({**row_base, "status": "failed_overfit_gate", "failure_reason": _first_failed_gate(overfit_rows)})
                print(f"plan1 variant={variant.name} seed={seed}: skipped held-out fit after overfit gate failure")
                continue

            split_config = replace(dataset_config)
            splits = base_splits if split_config == dataset_config else build_plan_gridworld_splits(split_config, seed=seed)
            print(f"plan1 variant={variant.name} seed={seed}: fitting trainable/frozen")
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
            controls = _run_controls(trainable.model, variant.model, splits["dev"], training.batch_size, device, seed)
            controls.update(
                {
                    "benchmark": BENCHMARK,
                    "seed": seed,
                    "variant": variant.name,
                    "chance": 1.0 / max(1, dataset_config.num_candidates),
                    "control_pass": _controls_pass(controls, _top1(dev_logits, dev_labels), dataset_config.num_candidates, variant.model),
                }
            )
            all_controls.append(controls)
            baselines = _baseline_block(splits["train"], splits["dev"], splits["test"], seed)
            row = {
                **row_base,
                "status": "completed",
                "leakage_audit_pass": bool(all(row.get("pass") for row in leakage)),
                "dev_trainable": _metric_block(dev_logits, dev_labels, splits["dev"]),
                "dev_frozen": _metric_block(frozen_dev_logits, dev_labels, splits["dev"]),
                "test_smoke_trainable": _metric_block(test_logits, test_labels, splits["test"]),
                "test_smoke_frozen": _metric_block(frozen_test_logits, test_labels, splits["test"]),
                "dev_delta_trainable_minus_frozen": float(_top1(dev_logits, dev_labels) - _top1(frozen_dev_logits, dev_labels)),
                "test_smoke_delta_trainable_minus_frozen": float(_top1(test_logits, test_labels) - _top1(frozen_test_logits, test_labels)),
                "baselines": baselines,
                "controls": controls.get("control_pass", {}),
                "trainable_audit": trainable.audit,
                "frozen_audit": frozen.audit,
                "medium_validation_triggered": False,
            }
            all_rows.append(row)
            all_attention.extend(_attention_summaries(trainable.model, variant.model, splits["dev"], training.batch_size, device, seed, limit=8))
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

    summary = _overall_summary(all_rows, all_controls, phase0_rows, dataset_config)
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "created_at_utc": _now(),
            "device": device,
            "claim_boundary": "candidate-plan verification/ranking only; no autonomous plan generation and no open-ended world modeling",
            "architecture_anchor": "candidate-plan-token direct attention over shared-weight latent planning-view clones",
            "final_validation_run": False,
            "final_validation_policy": "not launched by this bounded search; reserved seeds [70,71,72,73,74,75,76,77,78,79]",
            "allowed_claim_if_final_gates_pass": "On controlled candidate-plan verification tasks, shared-weight latent clone coordination learns to rank candidate plans using state, goal, constraint, and rollout evidence, outperforming an exact frozen same-architecture comparator and shortcut baselines while passing mismatch, shuffle, leakage, and invariance controls.",
            "forbidden_claims": [
                "autonomous planning",
                "general agentic AI",
                "LeCun-style world-model intelligence",
                "open-ended world modeling",
            ],
        },
        "dataset_config": asdict(dataset_config),
        "training_config": asdict(training),
        "variant_plan": [asdict(variant) for variant in variants],
        "phase0": phase0_rows,
        "rows": all_rows,
        "controls": all_controls,
        "leakage_audit": all_leakage,
        "overfit_curves": all_overfit,
        "error_cases": all_errors,
        "attention_summaries": all_attention,
        "compute_metrics": all_compute,
        "failure_taxonomy": _failure_taxonomy(all_rows, all_controls, all_errors),
        "summary": summary,
    }


def fit_plan_verifier(
    train_examples: Sequence[PlanGridworldExample],
    dev_examples: Sequence[PlanGridworldExample],
    model_config: PlanModelConfig,
    training_config: PlanTrainingConfig,
    seed: int,
    device: str,
    trainable_shared: bool,
    method: str,
) -> PlanFitResult:
    _set_seed(seed)
    model = PlanCloneVerifier(model_config).to(device)
    model.configure_shared_trainable(trainable_shared)
    initial_shared = _flat_params(model.shared_parameter_items())
    initial_coord = _flat_params(model.coordinator_parameter_items())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_config.lr),
        weight_decay=float(training_config.weight_decay),
    )
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_dev = -1.0
    bad_epochs = 0
    history: List[Dict[str, float]] = []
    shared_grad_norms: List[float] = []
    coord_grad_norms: List[float] = []
    start = time.perf_counter()
    for epoch in range(int(training_config.epochs)):
        model.train()
        losses: List[float] = []
        order = np.random.default_rng(seed + epoch).permutation(len(train_examples))
        for batch_indices in _chunks(order.tolist(), int(training_config.batch_size)):
            batch_examples = [train_examples[int(index)] for index in batch_indices]
            batch = collate_plan_batch(batch_examples, model_config, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)["logits"]
            loss = _candidate_objective_loss(logits, batch["labels"], training_config)
            loss.backward()
            shared_grad_norms.append(_grad_norm([parameter for _name, parameter in model.shared_parameter_items()]))
            coord_grad_norms.append(_grad_norm([parameter for _name, parameter in model.coordinator_parameter_items()]))
            if float(training_config.gradient_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(training_config.gradient_clip_norm),
                )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_logits = predict_plan_logits(model, dev_examples, model_config, int(training_config.batch_size), device)
        dev_labels = _labels(dev_examples)
        dev_acc = _top1(dev_logits, dev_labels)
        history.append({"epoch": float(epoch), "loss": _mean(losses), "dev_top1": float(dev_acc)})
        if dev_acc > best_dev:
            best_dev = float(dev_acc)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(training_config.patience):
            break
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    elapsed = time.perf_counter() - start
    final_shared = _flat_params(model.shared_parameter_items())
    final_coord = _flat_params(model.coordinator_parameter_items())
    audit = {
        "method": method,
        "initialization_seed": seed,
        "shared_model_trainable": bool(trainable_shared),
        "shared_parameter_names": [name for name, _parameter in model.shared_parameter_items()],
        "coordinator_parameter_names": [name for name, _parameter in model.coordinator_parameter_items()],
        "shared_grad_norm_mean": _mean(shared_grad_norms),
        "coordinator_grad_norm_mean": _mean(coord_grad_norms),
        "shared_parameter_delta": _l2_delta(initial_shared, final_shared),
        "coordinator_parameter_delta": _l2_delta(initial_coord, final_coord),
        "frozen_shared_model_zero_grad": bool((not trainable_shared) and _mean(shared_grad_norms) == 0.0),
        "frozen_shared_model_zero_delta": bool((not trainable_shared) and _l2_delta(initial_shared, final_shared) == 0.0),
        "trainable_shared_model_changed": bool(trainable_shared and _l2_delta(initial_shared, final_shared) > 0.0),
        "gradient_clip_norm": float(training_config.gradient_clip_norm),
        "same_architecture_comparator": True,
    }
    return PlanFitResult(
        method=method,
        model=model,
        config=model_config,
        trainable_shared=trainable_shared,
        audit=audit,
        history=history,
        training_time_seconds=float(elapsed),
        param_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def predict_plan_logits(
    model: "PlanCloneVerifier",
    examples: Sequence[PlanGridworldExample],
    model_config: PlanModelConfig,
    batch_size: int,
    device: str,
    condition: str = "none",
    seed: int = 0,
    view_mask: str | None = None,
    return_attention: bool = False,
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
    with torch.no_grad():
        for offset, batch_examples in enumerate(_batched(examples, int(batch_size))):
            batch = collate_plan_batch(batch_examples, model_config, device, view_names=view_names)
            if view_mask is not None:
                for axis, view in enumerate(view_names):
                    if view == view_mask:
                        batch["view_mask"][:, :, axis, :] = False
            output = model(
                batch,
                return_attention=return_attention,
                hidden_state_shuffle=condition == "hidden_state_shuffle",
                shuffle_seed=seed + offset,
            )
            logits_out.append(output["logits"].detach().cpu().numpy())
            if return_attention:
                attention_rows.extend(_summarize_attention_batch(output, batch, batch_examples, view_names))
    logits = np.concatenate(logits_out, axis=0) if logits_out else np.zeros((0, 0), dtype=np.float32)
    if return_attention:
        return logits, attention_rows
    return logits


def _candidate_objective_loss(logits: torch.Tensor, labels: torch.Tensor, training_config: PlanTrainingConfig) -> torch.Tensor:
    objective = str(training_config.objective)
    if objective == "cross_entropy":
        return F.cross_entropy(logits, labels)
    gold = logits.gather(1, labels.view(-1, 1))
    wrong_mask = torch.ones_like(logits, dtype=torch.bool)
    wrong_mask.scatter_(1, labels.view(-1, 1), False)
    wrong = logits.masked_select(wrong_mask).view(logits.shape[0], -1)
    if wrong.numel() == 0:
        return F.cross_entropy(logits, labels)
    if objective == "pairwise_logistic":
        return F.softplus(wrong - gold).mean()
    if objective == "margin_ranking":
        return F.relu(float(training_config.margin) - gold + wrong).mean()
    if objective == "contrastive_candidate_scoring":
        centered = logits - logits.mean(dim=1, keepdim=True)
        return F.cross_entropy(centered, labels)
    if objective == "ce_plus_pairwise":
        return F.cross_entropy(logits, labels) + 0.5 * F.softplus(wrong - gold).mean()
    if objective == "ce_plus_margin":
        return F.cross_entropy(logits, labels) + 0.5 * F.relu(float(training_config.margin) - gold + wrong).mean()
    raise ValueError(f"unknown PLAN-1 candidate objective: {objective}")


class PlanCloneVerifier(nn.Module):
    def __init__(self, config: PlanModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = _compatible_dim(config.model_dim, config.num_heads)
        self.token_encoder = PlanFieldTokenEncoder(dim)
        view_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=int(config.num_heads),
            dim_feedforward=int(config.ff_dim),
            dropout=float(config.dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        candidate_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=int(config.num_heads),
            dim_feedforward=int(config.ff_dim),
            dropout=float(config.dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.view_encoder = nn.TransformerEncoder(view_layer, num_layers=int(config.view_layers))
        self.candidate_encoder = nn.TransformerEncoder(candidate_layer, num_layers=int(config.candidate_layers))
        self.cross_blocks = nn.ModuleList(
            [CandidatePlanCrossAttentionBlock(dim, int(config.num_heads), int(config.ff_dim), float(config.dropout)) for _ in range(int(config.coordination_blocks))]
        )
        self.candidate_to_view_projection = nn.Linear(dim, dim)
        self.composer = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.score_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def configure_shared_trainable(self, trainable: bool) -> None:
        shared_ids = {id(parameter) for _name, parameter in self.shared_parameter_items()}
        for parameter in self.parameters():
            parameter.requires_grad = True
        if not trainable:
            for parameter in self.parameters():
                if id(parameter) in shared_ids:
                    parameter.requires_grad = False

    def shared_parameter_items(self) -> List[Tuple[str, nn.Parameter]]:
        modules = ("token_encoder", "view_encoder", "candidate_encoder", "composer")
        return [(name, parameter) for name, parameter in self.named_parameters() if name.split(".", 1)[0] in modules]

    def coordinator_parameter_items(self) -> List[Tuple[str, nn.Parameter]]:
        modules = ("cross_blocks", "candidate_to_view_projection", "score_head")
        return [(name, parameter) for name, parameter in self.named_parameters() if name.split(".", 1)[0] in modules]

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_attention: bool = False,
        hidden_state_shuffle: bool = False,
        shuffle_seed: int = 0,
    ) -> Dict[str, object]:
        view_fields = batch["view_fields"]
        view_mask = batch["view_mask"].bool()
        candidate_fields = batch["candidate_fields"]
        candidate_mask = batch["candidate_mask"].bool()
        batch_size, candidates, views, view_tokens, _fields = view_fields.shape
        candidate_tokens = candidate_fields.shape[2]

        flat_view_fields = view_fields.reshape(batch_size * candidates * views, view_tokens, -1)
        flat_view_mask = view_mask.reshape(batch_size * candidates * views, view_tokens)
        view_emb = self.token_encoder(flat_view_fields)
        view_encoded = self.view_encoder(view_emb, src_key_padding_mask=~flat_view_mask)
        view_encoded = view_encoded.reshape(batch_size, candidates, views, view_tokens, -1)
        if hidden_state_shuffle and batch_size > 1:
            generator = torch.Generator(device=view_encoded.device)
            generator.manual_seed(int(shuffle_seed))
            order = torch.randperm(batch_size, generator=generator, device=view_encoded.device)
            view_encoded = view_encoded.index_select(0, order)

        cand_emb = self.token_encoder(candidate_fields.reshape(batch_size * candidates, candidate_tokens, -1))
        cand_encoded = self.candidate_encoder(
            cand_emb,
            src_key_padding_mask=~candidate_mask.reshape(batch_size * candidates, candidate_tokens),
        )

        flat_view = view_encoded.reshape(batch_size * candidates, views * view_tokens, -1)
        flat_mask = view_mask.reshape(batch_size * candidates, views * view_tokens)
        if bool(self.config.pyramidal_composition):
            composed = self._composed_tokens(view_encoded, view_mask, batch["view_ids"])
            flat_view = torch.cat([flat_view, composed], dim=1)
            flat_mask = torch.cat([flat_mask, torch.ones(flat_view.shape[0], composed.shape[1], dtype=torch.bool, device=flat_view.device)], dim=1)

        attention_weights: List[torch.Tensor] = []
        for block in self.cross_blocks:
            view_for_candidate = flat_view
            if str(self.config.interaction_style) == "candidate_guided_world_interrogation":
                pooled_candidate = _masked_mean(cand_encoded, candidate_mask.reshape(batch_size * candidates, candidate_tokens))
                view_for_candidate = view_for_candidate + self.candidate_to_view_projection(pooled_candidate).unsqueeze(1)
            cand_encoded, weights = block(cand_encoded, view_for_candidate, flat_mask, return_attention=return_attention)
            if weights is not None:
                attention_weights.append(weights.reshape(batch_size, candidates, *weights.shape[1:]))
        pooled = _masked_mean(cand_encoded, candidate_mask.reshape(batch_size * candidates, candidate_tokens))
        scores = self.score_head(pooled).reshape(batch_size, candidates)
        return {"logits": scores, "attention_weights": attention_weights}

    def _composed_tokens(self, view_encoded: torch.Tensor, view_mask: torch.Tensor, view_ids: torch.Tensor) -> torch.Tensor:
        batch_size, candidates, _views, _tokens, dim = view_encoded.shape
        pooled = _masked_mean(view_encoded.reshape(batch_size * candidates, view_encoded.shape[2], view_encoded.shape[3], dim), view_mask.reshape(batch_size * candidates, view_mask.shape[2], view_mask.shape[3]))
        pooled = pooled.reshape(batch_size * candidates, view_encoded.shape[2], dim)
        view_id_list = [int(v) for v in view_ids.detach().cpu().tolist()]

        def get(name: str) -> torch.Tensor:
            if VIEW_ID[name] in view_id_list:
                return pooled[:, view_id_list.index(VIEW_ID[name])]
            return torch.zeros((batch_size * candidates, dim), dtype=pooled.dtype, device=pooled.device)

        state_goal = get("state_view") + get("goal_view")
        future = get("candidate_action_view") + get("rollout_view") + get("outcome_view")
        risk_adjusted = future + get("obstacle_constraint_view") + get("risk_failure_view") + get("cost_efficiency_view")
        composed = torch.stack([state_goal, future, risk_adjusted], dim=1)
        return self.composer(composed)


class PlanFieldTokenEncoder(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, model_dim) for size in PLAN_FIELD_VOCABS])
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        fields = fields.long()
        out = torch.zeros((*fields.shape[:2], self.embeddings[0].embedding_dim), dtype=torch.float32, device=fields.device)
        for index, embedding in enumerate(self.embeddings):
            values = fields[..., index].clamp(min=0, max=embedding.num_embeddings - 1)
            out = out + embedding(values)
        return self.norm(out)


class CandidatePlanCrossAttentionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(model_dim)
        self.norm_ff = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(nn.Linear(model_dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, model_dim))

    def forward(
        self,
        query: torch.Tensor,
        view_tokens: torch.Tensor,
        view_mask: torch.Tensor,
        return_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.attn(
            query=self.norm_q(query),
            key=view_tokens,
            value=view_tokens,
            key_padding_mask=~view_mask.bool(),
            need_weights=return_attention,
            average_attn_weights=False,
        )
        query = query + attended
        query = query + self.ff(self.norm_ff(query))
        return query, weights if return_attention else None


def collate_plan_batch(
    examples: Sequence[PlanGridworldExample],
    model_config: PlanModelConfig,
    device: str,
    view_names: Sequence[str] | None = None,
) -> Dict[str, torch.Tensor]:
    views = tuple(view_names or model_config.view_names)
    view_rows = []
    view_masks = []
    candidate_rows = []
    candidate_masks = []
    labels = []
    for example in examples:
        fields, mask = _view_token_fields(example, views, model_config)
        candidate_fields, candidate_mask = _candidate_token_fields(example, model_config)
        view_rows.append(fields)
        view_masks.append(mask)
        candidate_rows.append(candidate_fields)
        candidate_masks.append(candidate_mask)
        labels.append(int(example.label))
    return {
        "view_fields": torch.as_tensor(np.stack(view_rows), dtype=torch.long, device=device),
        "view_mask": torch.as_tensor(np.stack(view_masks), dtype=torch.bool, device=device),
        "candidate_fields": torch.as_tensor(np.stack(candidate_rows), dtype=torch.long, device=device),
        "candidate_mask": torch.as_tensor(np.stack(candidate_masks), dtype=torch.bool, device=device),
        "labels": torch.as_tensor(labels, dtype=torch.long, device=device),
        "view_ids": torch.as_tensor([VIEW_ID[str(view)] for view in views], dtype=torch.long, device=device),
    }


def _view_token_fields(
    example: PlanGridworldExample,
    view_names: Sequence[str],
    model_config: PlanModelConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    candidates = len(example.candidates)
    fields = np.zeros((candidates, len(view_names), int(model_config.max_view_tokens), len(PLAN_FIELD_VOCABS)), dtype=np.int64)
    mask = np.zeros((candidates, len(view_names), int(model_config.max_view_tokens)), dtype=bool)
    for candidate_index, candidate in enumerate(example.candidates):
        for view_axis, view in enumerate(view_names):
            view_id = VIEW_ID.get(str(view), view_axis)
            tokens = _tokens_for_view(example, candidate, str(view), view_id)
            if not tokens:
                tokens = [_token(0, view_id, SUMMARY_ROWCOL, SUMMARY_ROWCOL, 0, 0, 0, example.grid_size, example.grid_size, 0)]
            tokens = tokens[: int(model_config.max_view_tokens)]
            count = min(len(tokens), int(model_config.max_view_tokens))
            fields[candidate_index, view_axis, :count] = np.asarray(tokens[:count], dtype=np.int64)
            mask[candidate_index, view_axis, :count] = True
    return fields, mask


def _candidate_token_fields(
    example: PlanGridworldExample,
    model_config: PlanModelConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    fields = np.zeros((len(example.candidates), int(model_config.max_candidate_tokens), len(PLAN_FIELD_VOCABS)), dtype=np.int64)
    mask = np.zeros((len(example.candidates), int(model_config.max_candidate_tokens)), dtype=bool)
    for candidate_index, candidate in enumerate(example.candidates):
        tokens = _candidate_action_tokens(candidate, VIEW_ID["candidate_action_view"], example.grid_size)
        tokens = tokens[: int(model_config.max_candidate_tokens)]
        count = min(len(tokens), int(model_config.max_candidate_tokens))
        fields[candidate_index, :count] = np.asarray(tokens[:count], dtype=np.int64)
        mask[candidate_index, :count] = True
    return fields, mask


def _tokens_for_view(
    example: PlanGridworldExample,
    candidate: CandidateRollout,
    view: str,
    view_id: int,
) -> List[List[int]]:
    size = int(example.grid_size)
    if view == "state_view":
        return [
            _token(1, view_id, example.start[0], example.start[1], 1, 0, 0, size, size, len(example.obstacles)),
            _token(1, view_id, SUMMARY_ROWCOL, SUMMARY_ROWCOL, 2, 0, 0, size, size, size * size),
        ]
    if view == "goal_view":
        return [_token(2, view_id, example.goal[0], example.goal[1], 1, 0, 0, size, size, 0)]
    if view == "obstacle_constraint_view":
        tokens = [_token(3, view_id, row, col, 1, 0, 0, size, size, len(example.obstacles)) for row, col in example.obstacles]
        tokens.insert(0, _token(3, view_id, SUMMARY_ROWCOL, SUMMARY_ROWCOL, 0, 0, 0, size, size, len(example.obstacles)))
        return tokens
    if view == "candidate_action_view":
        return _candidate_action_tokens(candidate, view_id, size)
    if view == "rollout_view":
        return [_token(5, view_id, pos[0], pos[1], 1, step, 0, size, size, 0) for step, pos in enumerate(candidate.rollout)]
    if view == "outcome_view":
        return [_token(6, view_id, candidate.final_position[0], candidate.final_position[1], 1, len(candidate.actions), 0, size, size, 0)]
    if view == "cost_efficiency_view":
        return _cost_tokens(candidate, view_id, size)
    if view == "risk_failure_view":
        return _risk_tokens(example, candidate, view_id)
    return [_token(0, view_id, SUMMARY_ROWCOL, SUMMARY_ROWCOL, 0, 0, 0, size, size, 0)]


def _candidate_action_tokens(candidate: CandidateRollout, view_id: int, grid_size: int) -> List[List[int]]:
    tokens = [_token(4, view_id, SUMMARY_ROWCOL, SUMMARY_ROWCOL, 0, 0, 0, grid_size, grid_size, len(candidate.actions))]
    for step, action in enumerate(candidate.actions, start=1):
        dr, dc = ACTION_TO_DELTA[int(action)]
        aux = (dr + 1) * 3 + (dc + 1)
        tokens.append(_token(4, view_id, SUMMARY_ROWCOL, SUMMARY_ROWCOL, int(action) + 1, step, int(action), grid_size, grid_size, aux))
    return tokens


def _cost_tokens(candidate: CandidateRollout, view_id: int, grid_size: int) -> List[List[int]]:
    tokens = [_token(7, view_id, SUMMARY_ROWCOL, SUMMARY_ROWCOL, 0, 0, 0, grid_size, grid_size, len(candidate.actions))]
    turns = 0
    reversals = 0
    prev_action = None
    for step, action in enumerate(candidate.actions, start=1):
        if prev_action is not None:
            turns += int(action != prev_action)
            reversals += int(_is_reverse(prev_action, int(action)))
        prev_action = int(action)
        if step in {1, len(candidate.actions) // 2, len(candidate.actions)}:
            tokens.append(_token(7, view_id, SUMMARY_ROWCOL, SUMMARY_ROWCOL, min(15, turns), step, int(action), grid_size, grid_size, min(31, reversals)))
    return tokens


def _risk_tokens(example: PlanGridworldExample, candidate: CandidateRollout, view_id: int) -> List[List[int]]:
    size = int(example.grid_size)
    obstacle_set = set(example.obstacles)
    pos = tuple(example.start)
    tokens = [_token(8, view_id, SUMMARY_ROWCOL, SUMMARY_ROWCOL, 0, 0, 0, size, size, len(obstacle_set))]
    for step, action in enumerate(candidate.actions, start=1):
        if pos == tuple(example.goal):
            break
        dr, dc = ACTION_TO_DELTA[int(action)]
        nxt = (pos[0] + dr, pos[1] + dc)
        risk = 0
        if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
            risk = 2
        elif nxt in obstacle_set:
            risk = 1
        if risk:
            row = max(0, min(size - 1, nxt[0]))
            col = max(0, min(size - 1, nxt[1]))
            tokens.append(_token(8, view_id, row, col, risk, step, int(action), size, size, risk))
            break
        pos = nxt
    return tokens


def _token(
    kind: int,
    view_id: int,
    row: int,
    col: int,
    value: int,
    step: int,
    action: int,
    height: int,
    width: int,
    aux: int,
) -> List[int]:
    values = [
        int(kind),
        int(view_id),
        int(row),
        int(col),
        int(value),
        int(step),
        int(action),
        int(height),
        int(width),
        int(aux),
    ]
    return [max(0, min(PLAN_FIELD_VOCABS[index] - 1, value)) for index, value in enumerate(values)]


def _run_overfit_gates(
    variant: PlanVariant,
    dataset_config: PlanGridworldDatasetConfig,
    training: PlanTrainingConfig,
    seed: int,
    device: str,
) -> List[Dict[str, object]]:
    specs = [
        {"name": "4_examples_N2", "train_examples": 4, "candidate_count": 2, "level": 1, "target": 0.95},
        {"name": "16_examples_N4", "train_examples": 16, "candidate_count": 4, "level": 2, "target": 0.90},
        {"name": "64_examples_N8", "train_examples": 64, "candidate_count": 8, "level": max(3, int(dataset_config.level)), "target": 0.85},
    ]
    rows: List[Dict[str, object]] = []
    for spec in specs:
        if int(spec["candidate_count"]) > int(dataset_config.num_candidates) and int(dataset_config.num_candidates) < 8:
            continue
        cfg = replace(
            dataset_config,
            num_candidates=int(spec["candidate_count"]),
            train_examples=int(spec["train_examples"]),
            dev_examples=max(4, min(16, int(spec["train_examples"]))),
            test_examples=0,
            level=int(spec["level"]),
        )
        splits = build_plan_gridworld_splits(cfg, seed=seed + 70_000 + int(spec["candidate_count"]))
        fit = fit_plan_verifier(
            train_examples=splits["train"],
            dev_examples=splits["train"],
            model_config=variant.model,
            training_config=replace(training, objective=variant.model.objective, patience=int(training.epochs)),
            seed=seed + 80_000 + int(spec["candidate_count"]),
            device=device,
            trainable_shared=True,
            method=f"overfit__{variant.name}__{spec['name']}",
        )
        logits = predict_plan_logits(fit.model, splits["train"], variant.model, training.batch_size, device)
        labels = _labels(splits["train"])
        train_acc = _top1(logits, labels)
        rows.append(
            {
                "benchmark": BENCHMARK,
                "variant": variant.name,
                "seed": seed,
                "gate": spec["name"],
                "examples": int(spec["train_examples"]),
                "candidate_count": int(spec["candidate_count"]),
                "target_train_top1": float(spec["target"]),
                "train_top1": float(train_acc),
                "pass": bool(train_acc >= float(spec["target"])),
                "gate_required": True,
                "history": fit.history,
                "audit": fit.audit,
            }
        )
        del fit
        _clear_cuda()
    return rows


def _run_controls(
    model: PlanCloneVerifier,
    model_config: PlanModelConfig,
    examples: Sequence[PlanGridworldExample],
    batch_size: int,
    device: str,
    seed: int,
) -> Dict[str, object]:
    labels = _labels(examples)
    controls: Dict[str, object] = {}
    for name in (
        "candidate_plan_only",
        "state_goal_only",
        "state_plan_mismatch",
        "goal_shuffle",
        "obstacle_constraint_shuffle",
        "rollout_mismatch",
        "final_state_mismatch",
        "candidate_order_shuffle_with_gold_remap",
        "action_symbol_permutation",
        "map_rotation_reflection",
        "randomized_labels",
    ):
        if name == "rollout_mismatch" and "rollout_view" not in model_config.view_names:
            controls[name] = {"skipped": True, "reason": "variant has no rollout_view"}
            continue
        if name == "final_state_mismatch" and "outcome_view" not in model_config.view_names:
            controls[name] = {"skipped": True, "reason": "variant has no outcome_view"}
            continue
        controlled = apply_plan_control(examples, name, seed + 31_000 + len(controls))
        controlled_labels = _labels(controlled)
        logits = predict_plan_logits(model, controlled, model_config, batch_size, device, seed=seed)
        controls[name] = _metric_block(logits, controlled_labels, controlled)
    logits_role = predict_plan_logits(model, examples, model_config, batch_size, device, condition="physical_role_order_shuffle", seed=seed + 44_000)
    logits_hidden = predict_plan_logits(model, examples, model_config, batch_size, device, condition="hidden_state_shuffle", seed=seed + 55_000)
    base_logits = predict_plan_logits(model, examples, model_config, batch_size, device)
    controls["physical_role_order_shuffle"] = _metric_block(logits_role, labels, examples)
    controls["hidden_state_shuffle"] = _metric_block(logits_hidden, labels, examples)
    controls["role_view_masking"] = {}
    for view in model_config.view_names:
        logits = predict_plan_logits(model, examples, model_config, batch_size, device, view_mask=view)
        controls["role_view_masking"][str(view)] = _metric_block(logits, labels, examples)
    controls["candidate_order_invariance_delta"] = abs(
        float(_top1(base_logits, labels)) - float(controls["candidate_order_shuffle_with_gold_remap"]["top1"])
    )
    controls["role_order_invariance_delta"] = abs(float(_top1(base_logits, labels)) - float(controls["physical_role_order_shuffle"]["top1"]))
    controls["action_symbol_permutation_delta"] = abs(float(_top1(base_logits, labels)) - float(controls["action_symbol_permutation"]["top1"]))
    controls["map_rotation_reflection_delta"] = abs(float(_top1(base_logits, labels)) - float(controls["map_rotation_reflection"]["top1"]))
    controls["candidate_source_metadata_only_accuracy"] = candidate_source_metadata_only_accuracy(examples)
    controls["plan_length_baseline"] = _accuracy(_length_only_predictions(examples), labels)
    controls["action_distribution_baseline"] = _accuracy(_action_stat_predictions(examples), labels)
    return controls


def _phase0_summary(
    splits: Dict[str, Sequence[PlanGridworldExample]],
    leakage_rows: Sequence[Dict[str, object]],
    seed: int,
) -> Dict[str, object]:
    baselines = _baseline_block(splits["train"], splits["dev"], splits["test"], seed)
    return {
        "seed": seed,
        "dataset_summary": dataset_summary(splits),
        "leakage_audit_pass": bool(all(row.get("pass") for row in leakage_rows)),
        "candidate_metadata_only_dev_accuracy": candidate_source_metadata_only_accuracy(splits["dev"]),
        "candidate_action_stats_mlp_dev_top1": float(baselines["candidate_action_stats_mlp_dev"]),
        "plan_length_only_dev_top1": float(baselines["plan_length_only_dev"]),
        "action_distribution_only_dev_top1": float(baselines["action_distribution_only_dev"]),
        "source_metadata_hidden": bool(all(not row.get("candidate_source_ids_visible_to_model") for row in leakage_rows)),
    }


def _baseline_block(
    train_examples: Sequence[PlanGridworldExample],
    dev_examples: Sequence[PlanGridworldExample],
    test_examples: Sequence[PlanGridworldExample],
    seed: int,
) -> Dict[str, object]:
    dev_labels = _labels(dev_examples)
    test_labels = _labels(test_examples)
    candidate_mlp_dev, candidate_mlp_test = _candidate_stats_mlp_baseline(train_examples, dev_examples, test_examples, seed)
    return {
        "random": 1.0 / max(1, len(dev_examples[0].candidates) if dev_examples else 1),
        "candidate_order_index0_dev": _accuracy(np.zeros(len(dev_examples), dtype=np.int64), dev_labels),
        "candidate_order_index0_test": _accuracy(np.zeros(len(test_examples), dtype=np.int64), test_labels),
        "candidate_source_metadata_only_dev": candidate_source_metadata_only_accuracy(dev_examples),
        "candidate_source_metadata_only_test": candidate_source_metadata_only_accuracy(test_examples),
        "plan_length_only_dev": _accuracy(_length_only_predictions(dev_examples), dev_labels),
        "plan_length_only_test": _accuracy(_length_only_predictions(test_examples), test_labels),
        "action_distribution_only_dev": _accuracy(_action_stat_predictions(dev_examples), dev_labels),
        "action_distribution_only_test": _accuracy(_action_stat_predictions(test_examples), test_labels),
        "candidate_action_stats_mlp_dev": candidate_mlp_dev,
        "candidate_action_stats_mlp_test": candidate_mlp_test,
        "simulator_oracle_present": 1.0 if all(_exactly_one_best_by_hidden_metrics(example) for example in list(dev_examples) + list(test_examples)) else 0.0,
    }


def _length_only_predictions(examples: Sequence[PlanGridworldExample]) -> np.ndarray:
    preds = []
    for example in examples:
        lengths = [len(candidate.actions) for candidate in example.candidates]
        preds.append(int(np.argmin(lengths)) if lengths else 0)
    return np.asarray(preds, dtype=np.int64)


def _action_stat_predictions(examples: Sequence[PlanGridworldExample]) -> np.ndarray:
    preds = []
    for example in examples:
        scores = []
        for candidate in example.candidates:
            actions = list(candidate.actions)
            turns = sum(int(actions[i] != actions[i - 1]) for i in range(1, len(actions)))
            reversals = sum(int(_is_reverse(actions[i - 1], actions[i])) for i in range(1, len(actions)))
            counts = np.bincount(np.asarray(actions, dtype=np.int64), minlength=4).astype(np.float32)
            balance = float(np.std(counts / max(1, len(actions))))
            scores.append(turns + 0.5 * reversals + balance)
        preds.append(int(np.argmin(scores)) if scores else 0)
    return np.asarray(preds, dtype=np.int64)


class CandidateStatsScorer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _candidate_stats_mlp_baseline(
    train_examples: Sequence[PlanGridworldExample],
    dev_examples: Sequence[PlanGridworldExample],
    test_examples: Sequence[PlanGridworldExample],
    seed: int,
) -> Tuple[float, float]:
    if not train_examples or not dev_examples:
        return 0.0, 0.0
    _set_seed(seed + 91_000)
    train_x = torch.as_tensor(_candidate_stat_features(train_examples), dtype=torch.float32)
    train_y = torch.as_tensor(_labels(train_examples), dtype=torch.long)
    dev_x = torch.as_tensor(_candidate_stat_features(dev_examples), dtype=torch.float32)
    dev_y = _labels(dev_examples)
    test_x = torch.as_tensor(_candidate_stat_features(test_examples), dtype=torch.float32) if test_examples else torch.zeros((0, 1, train_x.shape[-1]))
    test_y = _labels(test_examples)
    model = CandidateStatsScorer(train_x.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_dev = -1.0
    for epoch in range(50):
        order = np.random.default_rng(seed + epoch).permutation(len(train_examples))
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


def _candidate_stat_features(examples: Sequence[PlanGridworldExample]) -> np.ndarray:
    rows = []
    for example in examples:
        cand_rows = []
        for candidate in example.candidates:
            actions = np.asarray(candidate.actions, dtype=np.int64)
            counts = np.bincount(actions, minlength=4).astype(np.float32) / max(1, len(actions))
            turns = sum(int(actions[i] != actions[i - 1]) for i in range(1, len(actions))) / max(1, len(actions))
            reversals = sum(int(_is_reverse(int(actions[i - 1]), int(actions[i]))) for i in range(1, len(actions))) / max(1, len(actions))
            net_row = float(np.sum(actions == 1) - np.sum(actions == 0)) / max(1, len(actions))
            net_col = float(np.sum(actions == 3) - np.sum(actions == 2)) / max(1, len(actions))
            first = np.bincount(actions[: max(1, len(actions) // 2)], minlength=4).astype(np.float32) / max(1, len(actions) // 2)
            cand_rows.append(np.concatenate([counts, first, np.asarray([turns, reversals, net_row, net_col, len(actions) / 32.0], dtype=np.float32)]))
        rows.append(np.stack(cand_rows))
    return np.stack(rows) if rows else np.zeros((0, 0, 13), dtype=np.float32)


def _metric_block(logits: np.ndarray, labels: np.ndarray, examples: Sequence[PlanGridworldExample]) -> Dict[str, object]:
    if logits.size == 0:
        return {"top1": 0.0, "top2": 0.0, "top3": 0.0, "mrr": 0.0, "mean_gold_rank": 0.0, "gold_ranks": []}
    order = np.argsort(-logits, axis=1)
    ranks = [int(np.where(row == int(label))[0][0]) + 1 for row, label in zip(order, labels)]
    predictions = order[:, 0]
    margins = []
    for row_index, label in enumerate(labels):
        gold_logit = float(logits[row_index, int(label)])
        wrong = np.delete(logits[row_index], int(label))
        best_wrong = float(np.max(wrong)) if wrong.size else gold_logit
        margins.append(gold_logit - best_wrong)
    block = {
        "top1": _accuracy(predictions, labels),
        "top2": float(np.mean([rank <= 2 for rank in ranks])) if ranks else 0.0,
        "top3": float(np.mean([rank <= 3 for rank in ranks])) if ranks else 0.0,
        "mrr": float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0,
        "mean_gold_rank": float(np.mean(ranks)) if ranks else 0.0,
        "gold_ranks": ranks,
        "mean_gold_vs_best_wrong_logit_margin": float(np.mean(margins)) if margins else 0.0,
        "min_gold_vs_best_wrong_logit_margin": float(np.min(margins)) if margins else 0.0,
        "by_plan_length": _accuracy_by_key(predictions, labels, [str(len(example.candidates[0].actions)) for example in examples]),
        "by_obstacle_density": _accuracy_by_key(predictions, labels, [str(round(len(example.obstacles) / (example.grid_size * example.grid_size), 2)) for example in examples]),
        "by_map_size": _accuracy_by_key(predictions, labels, [str(example.grid_size) for example in examples]),
        "by_goal_distance": _accuracy_by_key(predictions, labels, [str(example.metadata.get("goal_distance", "unknown")) for example in examples]),
        "by_number_of_constraints": _accuracy_by_key(predictions, labels, [str(len(example.obstacles)) for example in examples]),
        "selected_vs_gold_family_confusion": _family_confusion(predictions, examples),
    }
    block.update(selected_plan_metrics(examples, predictions))
    return block


def _attention_summaries(
    model: PlanCloneVerifier,
    model_config: PlanModelConfig,
    examples: Sequence[PlanGridworldExample],
    batch_size: int,
    device: str,
    seed: int,
    limit: int,
) -> List[Dict[str, object]]:
    subset = list(examples[:limit])
    _logits, rows = predict_plan_logits(model, subset, model_config, batch_size, device, seed=seed, return_attention=True)
    for row in rows:
        row["seed"] = seed
        row["benchmark"] = BENCHMARK
    return rows


def _summarize_attention_batch(
    output: Dict[str, object],
    batch: Dict[str, torch.Tensor],
    examples: Sequence[PlanGridworldExample],
    view_names: Sequence[str],
) -> List[Dict[str, object]]:
    weights_list = output.get("attention_weights", [])
    if not weights_list:
        return []
    logits = output["logits"].detach().cpu().numpy()
    preds = np.argmax(logits, axis=1)
    rows = []
    view_count = len(view_names)
    view_tokens = batch["view_fields"].shape[3]
    view_fields = batch["view_fields"].detach().cpu().numpy()
    for example_index, example in enumerate(examples):
        block_rows = []
        for block_index, weights in enumerate(weights_list):
            w = weights[example_index].detach().float().cpu().numpy()
            non_composed_width = view_count * view_tokens
            visible = w[..., :non_composed_width]
            by_view = visible.reshape(w.shape[0], w.shape[1], w.shape[2], view_count, view_tokens).mean(axis=(0, 1, 2, 4))
            candidate_mass = w.sum(axis=(1, 2, 3)) / max(1, w.shape[1] * w.shape[2])
            flat = visible.mean(axis=(0, 1, 2)).reshape(view_count, view_tokens)
            top = []
            for flat_index in np.argsort(-flat.reshape(-1))[:5]:
                view_axis = int(flat_index // view_tokens)
                token_axis = int(flat_index % view_tokens)
                fields = view_fields[example_index, 0, view_axis, token_axis].tolist()
                top.append(
                    {
                        "view": str(view_names[view_axis]),
                        "kind": int(fields[0]),
                        "row": int(fields[2]),
                        "col": int(fields[3]),
                        "step": int(fields[5]),
                        "action": int(fields[6]),
                        "mass": float(flat[view_axis, token_axis]),
                    }
                )
            probs = np.clip(flat.reshape(-1), 1e-9, 1.0)
            entropy = float(-np.sum(probs * np.log(probs)))
            block_rows.append(
                {
                    "block": int(block_index),
                    "attention_mass_by_clone_view": {str(view_names[i]): float(by_view[i]) for i in range(view_count)},
                    "attention_mass_by_candidate": {str(i): float(candidate_mass[i]) for i in range(w.shape[0])},
                    "attention_entropy": entropy,
                    "top_attended_tokens": top,
                }
            )
        rows.append(
            {
                "type": "plan1_attention_summary",
                "example_id": example.id,
                "predicted_candidate": int(preds[example_index]),
                "gold_candidate": int(example.label),
                "candidate_probabilities": _softmax_np(logits[example_index]).tolist(),
                "blocks": block_rows,
            }
        )
    return rows


def _error_cases(
    variant: str,
    seed: int,
    examples: Sequence[PlanGridworldExample],
    train_logits: np.ndarray,
    frozen_logits: np.ndarray,
    controls: Dict[str, object],
    limit: int,
) -> List[Dict[str, object]]:
    labels = _labels(examples)
    train_pred = np.argmax(train_logits, axis=1) if train_logits.size else np.zeros(len(examples), dtype=np.int64)
    frozen_pred = np.argmax(frozen_logits, axis=1) if frozen_logits.size else np.zeros(len(examples), dtype=np.int64)
    action_pred = _action_stat_predictions(examples)
    buckets = {
        "trainable_right_frozen_wrong": [],
        "frozen_right_trainable_wrong": [],
        "candidate_only_right_trainable_wrong": [],
        "selected_collision": [],
        "selected_near_miss": [],
        "selected_valid_but_suboptimal": [],
        "all_models_wrong": [],
    }
    for index, example in enumerate(examples):
        tr = int(train_pred[index])
        fr = int(frozen_pred[index])
        ap = int(action_pred[index])
        label = int(labels[index])
        selected = example.candidates[tr]
        if tr == label and fr != label:
            buckets["trainable_right_frozen_wrong"].append(index)
        if fr == label and tr != label:
            buckets["frozen_right_trainable_wrong"].append(index)
        if ap == label and tr != label:
            buckets["candidate_only_right_trainable_wrong"].append(index)
        if selected.collision:
            buckets["selected_collision"].append(index)
        if selected.near_miss:
            buckets["selected_near_miss"].append(index)
        if selected.valid and tr != label:
            buckets["selected_valid_but_suboptimal"].append(index)
        if tr != label and fr != label and ap != label:
            buckets["all_models_wrong"].append(index)
    rows = []
    per_bucket = max(1, limit // max(1, len(buckets)))
    for category, indices in buckets.items():
        for index in indices[:per_bucket]:
            example = examples[index]
            tr = int(train_pred[index])
            probs = _softmax_np(train_logits[index]).tolist()
            rank = int(np.where(np.argsort(-train_logits[index]) == int(example.label))[0][0]) + 1
            rows.append(
                {
                    "type": "plan1_error_case",
                    "category": category,
                    "variant": variant,
                    "seed": seed,
                    "example_id": example.id,
                    "grid_size": example.grid_size,
                    "start": list(example.start),
                    "goal": list(example.goal),
                    "obstacles": [list(pos) for pos in example.obstacles],
                    "predicted_candidate_index": tr,
                    "gold_candidate_index": int(example.label),
                    "predicted_actions": [ACTION_NAMES[int(action)] for action in example.candidates[tr].actions],
                    "gold_actions": [ACTION_NAMES[int(action)] for action in example.candidates[int(example.label)].actions],
                    "candidate_probabilities": probs,
                    "gold_rank": rank,
                    "candidate_families_hidden_for_model": [candidate.family for candidate in example.candidates],
                    "selected_candidate_metrics": _candidate_public_debug(example.candidates[tr]),
                    "gold_candidate_metrics": _candidate_public_debug(example.candidates[int(example.label)]),
                    "ablation_sensitivity": controls.get("role_view_masking", {}),
                    "control_status": controls.get("control_pass", {}),
                }
            )
    return rows


def _overall_summary(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    phase0_rows: Sequence[Dict[str, object]],
    dataset_config: PlanGridworldDatasetConfig,
) -> Dict[str, object]:
    completed = [row for row in rows if row.get("status") == "completed"]
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in completed:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    variant_summaries = []
    for variant, variant_rows in by_variant.items():
        train = [float(row["dev_trainable"]["top1"]) for row in variant_rows]
        frozen = [float(row["dev_frozen"]["top1"]) for row in variant_rows]
        deltas = [float(row["dev_delta_trainable_minus_frozen"]) for row in variant_rows]
        variant_controls = [row for row in controls if row.get("variant") == variant]
        ablation_drops = [_best_ablation_drop(row) for row in variant_controls]
        variant_summaries.append(
            {
                "variant": variant,
                "seeds": [int(row["seed"]) for row in variant_rows],
                "mean_trainable_top1": _mean(train),
                "std_trainable_top1": _std(train),
                "min_trainable_top1": min(train) if train else 0.0,
                "max_trainable_top1": max(train) if train else 0.0,
                "mean_frozen_top1": _mean(frozen),
                "mean_delta": _mean(deltas),
                "std_delta": _std(deltas),
                "min_delta": min(deltas) if deltas else 0.0,
                "max_delta": max(deltas) if deltas else 0.0,
                "bootstrap_95ci_delta": _bootstrap_ci(deltas),
                "seeds_trainable_beats_frozen": int(sum(delta > 0.0 for delta in deltas)),
                "controls_pass_all": all(bool(row.get("control_pass", {}).get("overall", False)) for row in variant_controls),
                "mean_best_view_ablation_drop": _mean(ablation_drops),
                "candidate_only_mean_top1": _mean([float(row.get("candidate_plan_only", {}).get("top1", 0.0)) for row in variant_controls]),
                "length_only_mean_top1": _mean([float(row.get("plan_length_baseline", 0.0)) for row in variant_controls]),
                "action_distribution_mean_top1": _mean([float(row.get("action_distribution_baseline", 0.0)) for row in variant_controls]),
            }
        )
    variant_summaries.sort(key=lambda row: float(row["mean_delta"]), reverse=True)
    best = variant_summaries[0] if variant_summaries else {}
    return {
        "variant_summaries": variant_summaries,
        "best_variant_by_delta": best.get("variant"),
        "candidate_count": dataset_config.num_candidates,
        "random_baseline": 1.0 / max(1, dataset_config.num_candidates),
        "phase0_pass": all(
            bool(row.get("leakage_audit_pass"))
            and float(row.get("candidate_action_stats_mlp_dev_top1", 1.0)) <= 1.0 / max(1, dataset_config.num_candidates) + 0.15
            for row in phase0_rows
        ),
        "medium_validation_trigger": _medium_validation_trigger(best, controls, rows, dataset_config),
        "final_validation_run": False,
        "claim_boundary": "No autonomous-planning claim. This is a controlled candidate-plan verifier/ranker search.",
    }


def _medium_validation_trigger(
    best: Dict[str, object],
    controls: Sequence[Dict[str, object]],
    rows: Sequence[Dict[str, object]],
    dataset_config: PlanGridworldDatasetConfig,
) -> Dict[str, object]:
    best_variant = best.get("variant")
    if not best_variant:
        return {"passes": False, "reason": "no completed variant"}
    chance = 1.0 / max(1, int(dataset_config.num_candidates))
    best_controls = [row for row in controls if row.get("variant") == best_variant]
    best_rows = [row for row in rows if row.get("variant") == best_variant and row.get("status") == "completed"]
    best_views = tuple(best_rows[0].get("model_config", {}).get("view_names", [])) if best_rows else tuple()
    seeds_run = len(best_rows)
    frozen_ok = all(
        bool(row.get("frozen_audit", {}).get("frozen_shared_model_zero_grad"))
        and bool(row.get("frozen_audit", {}).get("frozen_shared_model_zero_delta"))
        for row in best_rows
    )
    trainable_ok = all(bool(row.get("trainable_audit", {}).get("trainable_shared_model_changed")) for row in best_rows)
    gates = {
        "trainable_beats_frozen_on_at_least_2_of_3_cheap_seeds": bool(seeds_run >= 3 and int(best.get("seeds_trainable_beats_frozen", 0)) >= 2),
        "mean_trainable_frozen_delta_at_least_0_15": float(best.get("mean_delta", 0.0)) >= 0.15,
        "top1_clearly_above_random": float(best.get("mean_trainable_top1", 0.0)) >= chance + 0.15,
        "candidate_only_near_chance": all(float(row.get("candidate_plan_only", {}).get("top1", 1.0)) <= chance + 0.15 for row in best_controls),
        "metadata_source_only_near_chance": all(float(row.get("candidate_source_metadata_only_accuracy", 1.0)) <= chance + 0.10 for row in best_controls),
        "state_plan_mismatch_collapses": all(float(row.get("state_plan_mismatch", {}).get("top1", 1.0)) <= chance + 0.20 for row in best_controls),
        "goal_shuffle_collapses": all(float(row.get("goal_shuffle", {}).get("top1", 1.0)) <= chance + 0.20 for row in best_controls),
        "candidate_order_remap_invariance_passes": all(float(row.get("candidate_order_invariance_delta", 1.0)) <= 0.08 for row in best_controls),
        "role_order_invariance_passes": all(float(row.get("role_order_invariance_delta", 1.0)) <= 0.08 for row in best_controls),
        "meaningful_view_ablation_hurts": any(_best_ablation_drop(row) >= 0.05 for row in best_controls),
        "no_length_source_action_shortcut": all(
            float(row.get("plan_length_baseline", 1.0)) <= chance + 0.10
            and float(row.get("action_distribution_baseline", 1.0)) <= chance + 0.15
            for row in best_controls
        ),
        "frozen_comparator_remains_frozen": frozen_ok,
        "trainable_shared_model_changes": trainable_ok,
        "final_validation_not_launched": True,
    }
    if "rollout_view" in best_views:
        gates["rollout_mismatch_collapses"] = all(float(row.get("rollout_mismatch", {}).get("top1", 1.0)) <= chance + 0.20 for row in best_controls)
    if "outcome_view" in best_views:
        gates["final_state_mismatch_collapses"] = all(float(row.get("final_state_mismatch", {}).get("top1", 1.0)) <= chance + 0.20 for row in best_controls)
    gates["passes"] = all(gates.values())
    return gates


def _controls_pass(
    control_row: Dict[str, object],
    clean_top1: float,
    candidate_count: int,
    model_config: PlanModelConfig,
) -> Dict[str, bool]:
    chance = 1.0 / max(1, int(candidate_count))
    pass_map = {
        "candidate_plan_only": float(control_row.get("candidate_plan_only", {}).get("top1", 1.0)) <= chance + 0.15,
        "state_goal_only": float(control_row.get("state_goal_only", {}).get("top1", 1.0)) <= chance + 0.15,
        "state_plan_mismatch": float(control_row.get("state_plan_mismatch", {}).get("top1", 1.0)) <= chance + 0.20,
        "goal_shuffle": float(control_row.get("goal_shuffle", {}).get("top1", 1.0)) <= chance + 0.20,
        "obstacle_constraint_shuffle": float(control_row.get("obstacle_constraint_shuffle", {}).get("top1", 1.0)) <= chance + 0.20,
        "candidate_order_shuffle_with_gold_remap": float(control_row.get("candidate_order_invariance_delta", 1.0)) <= 0.08,
        "physical_role_order_shuffle": float(control_row.get("role_order_invariance_delta", 1.0)) <= 0.08,
        "action_symbol_permutation": float(control_row.get("action_symbol_permutation_delta", 1.0)) <= 0.12,
        "map_rotation_reflection": float(control_row.get("map_rotation_reflection_delta", 1.0)) <= 0.12,
        "hidden_state_shuffle": float(control_row.get("hidden_state_shuffle", {}).get("top1", 1.0)) <= max(chance + 0.20, clean_top1 - 0.05),
        "randomized_labels": float(control_row.get("randomized_labels", {}).get("top1", 1.0)) <= chance + 0.20,
        "candidate_source_audit": float(control_row.get("candidate_source_metadata_only_accuracy", 1.0)) <= chance + 0.10,
        "plan_length_baseline": float(control_row.get("plan_length_baseline", 1.0)) <= chance + 0.10,
        "action_distribution_baseline": float(control_row.get("action_distribution_baseline", 1.0)) <= chance + 0.15,
    }
    if "rollout_view" in model_config.view_names:
        pass_map["rollout_mismatch"] = float(control_row.get("rollout_mismatch", {}).get("top1", 1.0)) <= chance + 0.20
    if "outcome_view" in model_config.view_names:
        pass_map["final_state_mismatch"] = float(control_row.get("final_state_mismatch", {}).get("top1", 1.0)) <= chance + 0.20
    pass_map["overall"] = all(pass_map.values())
    return pass_map


def _failure_taxonomy(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    error_cases: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    categories: Dict[str, int] = {}
    for row in rows:
        if row.get("status") == "failed_overfit_gate":
            categories["failed_micro_overfit"] = categories.get("failed_micro_overfit", 0) + 1
        elif row.get("status") == "completed" and float(row.get("dev_delta_trainable_minus_frozen", 0.0)) <= 0.0:
            categories["trainable_not_above_frozen"] = categories.get("trainable_not_above_frozen", 0) + 1
    for row in controls:
        pass_map = row.get("control_pass", {})
        for key, value in pass_map.items():
            if key != "overall" and not bool(value):
                categories[f"control_failed:{key}"] = categories.get(f"control_failed:{key}", 0) + 1
    for row in error_cases:
        category = str(row.get("category", "unknown"))
        categories[f"error_case:{category}"] = categories.get(f"error_case:{category}", 0) + 1
    return {
        "categories": dict(sorted(categories.items())),
        "interpretation": "Counts are diagnostic for the bounded search and are not final-validation failure rates.",
    }


def _write_outputs(
    result: Dict[str, object],
    output_path: Path,
    controls_path: Path,
    leaderboard_path: Path,
    failure_path: Path,
    overfit_path: Path,
    error_path: Path,
    attention_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    for path in (output_path, controls_path, leaderboard_path, failure_path, overfit_path, error_path, attention_path, compute_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_without_large_logs(result), indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(json.dumps({"controls": result["controls"], "summary": result["summary"].get("medium_validation_trigger", {})}, indent=2, sort_keys=True), encoding="utf-8")
    failure_path.write_text(json.dumps(result["failure_taxonomy"], indent=2, sort_keys=True), encoding="utf-8")
    overfit_path.write_text(json.dumps({"overfit_curves": result["overfit_curves"]}, indent=2, sort_keys=True), encoding="utf-8")
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
                "action_distribution_mean_top1",
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
                    "action_distribution_mean_top1": row.get("action_distribution_mean_top1"),
                    "mean_best_view_ablation_drop": row.get("mean_best_view_ablation_drop"),
                    "controls_pass_all": row.get("controls_pass_all"),
                }
            )


def _without_large_logs(result: Dict[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"leakage_audit", "overfit_curves", "error_cases", "attention_summaries", "compute_metrics"}
    }


def _render_report(result: Dict[str, object]) -> str:
    summary = result.get("summary", {})
    best_variant = summary.get("best_variant_by_delta")
    best_rows = [row for row in result.get("rows", []) if row.get("variant") == best_variant and row.get("status") == "completed"]
    best_controls = [row for row in result.get("controls", []) if row.get("variant") == best_variant]
    best_trainable = _mean([float(row["dev_trainable"]["top1"]) for row in best_rows])
    best_frozen = _mean([float(row["dev_frozen"]["top1"]) for row in best_rows])
    best_delta = _mean([float(row["dev_delta_trainable_minus_frozen"]) for row in best_rows])
    random_baseline = float(summary.get("random_baseline", 0.0))
    gates = summary.get("medium_validation_trigger", {})
    rollout_rows = [row for row in result.get("rows", []) if row.get("status") == "completed" and "rollout_view" in row.get("model_config", {}).get("view_names", [])]
    no_rollout_rows = [row for row in result.get("rows", []) if row.get("status") == "completed" and "rollout_view" not in row.get("model_config", {}).get("view_names", [])]
    risk_rows = [row for row in result.get("rows", []) if row.get("status") == "completed" and "risk_failure_view" in row.get("model_config", {}).get("view_names", [])]
    stacked_rows = [row for row in result.get("rows", []) if row.get("status") == "completed" and int(row.get("model_config", {}).get("coordination_blocks", 1)) > 1]
    pyramidal_rows = [row for row in result.get("rows", []) if row.get("status") == "completed" and bool(row.get("model_config", {}).get("pyramidal_composition", False))]
    guided_rows = [row for row in result.get("rows", []) if row.get("status") == "completed" and row.get("model_config", {}).get("interaction_style") == "candidate_guided_world_interrogation"]
    lines = [
        "# Stage PLAN-1 Latent Candidate-Plan Verifier / Ranker",
        "",
        "## Scope",
        "",
        "- Stage name: `PLAN-1_LATENT_CANDIDATE_PLAN_VERIFIER`.",
        "- Task: select the best candidate action sequence for a synthetic gridworld state, goal, and obstacle map.",
        "- This is not autonomous planning: candidate plans are generated by a simulator and the model only ranks them.",
        "- No natural-language chain-of-thought channel is used; inputs are structured latent tokens.",
        f"- Final validation run: `{bool(summary.get('final_validation_run', False))}`.",
        "",
        "## Phase 0 Data Smoke",
        "",
    ]
    for row in result.get("phase0", []):
        ds = row.get("dataset_summary", {})
        lines.extend(
            [
                f"- Seed `{row['seed']}` examples: `{ds.get('num_examples', {})}`.",
                f"- Length balanced: `{ds.get('all_length_balanced')}`; exactly one best candidate: `{ds.get('exactly_one_best_candidate')}`.",
                f"- Leakage audit pass: `{row.get('leakage_audit_pass')}`; source metadata hidden: `{row.get('source_metadata_hidden')}`.",
                f"- Candidate-action stats MLP dev top1: `{float(row.get('candidate_action_stats_mlp_dev_top1', 0.0)):.4f}`; random baseline `{random_baseline:.4f}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Bounded Search Results",
            "",
            "| variant | seeds | trainable | frozen | delta | candidate-only | length-only | action-stat | ablation drop | controls |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary.get("variant_summaries", []):
        lines.append(
            f"| {row['variant']} | {len(row.get('seeds', []))} | {float(row['mean_trainable_top1']):.4f} | {float(row['mean_frozen_top1']):.4f} | {float(row['mean_delta']):.4f} | {float(row.get('candidate_only_mean_top1', 0.0)):.4f} | {float(row.get('length_only_mean_top1', 0.0)):.4f} | {float(row.get('action_distribution_mean_top1', 0.0)):.4f} | {float(row.get('mean_best_view_ablation_drop', 0.0)):.4f} | `{bool(row.get('controls_pass_all'))}` |"
        )
    lines.extend(["", "## Medium Validation Trigger", "", "| gate | pass |", "|---|---|"])
    for key, value in gates.items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Required Questions",
            "",
            "1. Can the architecture evaluate candidate plans better than frozen?",
            f"   - Bounded-search answer: best variant `{best_variant}` scored trainable `{best_trainable:.4f}` vs frozen `{best_frozen:.4f}` on dev, delta `{best_delta:.4f}`. This is not final validation.",
            "2. Does it use state/goal/constraints rather than candidate artifacts?",
            f"   - Candidate-only, length-only, action-stat, mismatch, shuffle, and metadata controls are logged. The medium trigger status is `{gates.get('passes')}`.",
            "3. Which views matter most?",
            f"   - View ablation drops are saved in controls. Best observed drop for the best variant is `{_mean([_best_ablation_drop(row) for row in best_controls]):.4f}`.",
            "4. Does rollout evidence help?",
            f"   - Rollout-enabled mean trainable top1 across completed rows: `{_mean([float(row['dev_trainable']['top1']) for row in rollout_rows]):.4f}`.",
            "5. Can it work without rollout evidence?",
            f"   - No-rollout mean trainable top1 across completed rows: `{_mean([float(row['dev_trainable']['top1']) for row in no_rollout_rows]):.4f}`.",
            "6. Does risk/cost cloning help?",
            f"   - Risk/cost-enabled completed-row mean top1: `{_mean([float(row['dev_trainable']['top1']) for row in risk_rows]):.4f}`.",
            "7. Does stacking help?",
            f"   - Stacked completed-row mean top1: `{_mean([float(row['dev_trainable']['top1']) for row in stacked_rows]):.4f}`.",
            "8. Does pyramidal view composition help in planning?",
            f"   - Pyramidal completed-row mean top1: `{_mean([float(row['dev_trainable']['top1']) for row in pyramidal_rows]):.4f}`.",
            "9. Does candidate-guided querying help or leak?",
            f"   - Candidate-guided completed-row mean top1: `{_mean([float(row['dev_trainable']['top1']) for row in guided_rows]):.4f}`. Leakage controls decide whether it remains usable.",
            "10. Is this easier/more promising than ARC candidate verification?",
            "   - This synthetic gridworld is easier to audit than ARC because simulator labels, mismatch controls, and selected-plan outcome metrics are exact. It should be treated as a cleaner PLAN-1 probe, not as a stronger real-world claim.",
            "11. Is it worth moving toward learned world-model rollouts later?",
            "   - Only if medium/final gates pass with rollout and no-rollout variants. This bounded run writes the trigger decision but does not launch final validation.",
            "",
            "## Claim Boundary",
            "",
            "No claim is made about autonomous planning, general agentic AI, or open-ended world modeling. Final validation remains locked behind the medium trigger and reserved fresh seeds.",
        ]
    )
    return "\n".join(lines) + "\n"


def _variant_plan() -> List[PlanVariant]:
    p0_views = ("state_view", "goal_view", "obstacle_constraint_view", "candidate_action_view")
    rollout_views = p0_views + ("rollout_view", "outcome_view")
    risk_views = rollout_views + ("risk_failure_view",)
    cost_risk_views = rollout_views + ("cost_efficiency_view", "risk_failure_view")
    return [
        PlanVariant(
            name="P0_flat_candidate_plan_direct",
            short_name="P0",
            description="State/goal/constraint/action views; candidate-plan direct attention over shared clone states; one coordination block; CE objective.",
            model=PlanModelConfig(variant_family="flat_candidate_plan_direct", view_names=p0_views, coordination_blocks=1, objective="cross_entropy"),
        ),
        PlanVariant(
            name="P1_rollout_enabled",
            short_name="P1",
            description="Adds simulator rollout and final outcome views as evidence.",
            model=PlanModelConfig(variant_family="rollout_enabled", view_names=rollout_views, coordination_blocks=1, objective="cross_entropy"),
        ),
        PlanVariant(
            name="P2_no_rollout_latent_inference",
            short_name="P2",
            description="No rollout or final state; model must infer plan quality from state/goal/obstacle/action tokens.",
            model=PlanModelConfig(variant_family="no_rollout_latent_inference", view_names=p0_views, coordination_blocks=1, objective="cross_entropy"),
        ),
        PlanVariant(
            name="P3_outcome_only",
            short_name="P3",
            description="Includes final outcome state but omits full rollout evidence.",
            model=PlanModelConfig(variant_family="outcome_only", view_names=p0_views + ("outcome_view",), coordination_blocks=1, objective="cross_entropy"),
        ),
        PlanVariant(
            name="P4_risk_clone",
            short_name="P4",
            description="Adds explicit collision/constraint-relevant risk view without exposing gold/reward labels.",
            model=PlanModelConfig(variant_family="risk_clone", view_names=risk_views, coordination_blocks=1, objective="cross_entropy"),
        ),
        PlanVariant(
            name="P5_cost_value_clone",
            short_name="P5",
            description="Adds a cost-efficiency view over plan dynamics while all raw plan lengths remain balanced.",
            model=PlanModelConfig(variant_family="cost_value_clone", view_names=cost_risk_views, coordination_blocks=1, objective="cross_entropy"),
        ),
        PlanVariant(
            name="P6_pairwise_plan_comparison",
            short_name="P6",
            description="Same rollout views as P1, trained with CE plus pairwise ranking pressure.",
            model=PlanModelConfig(variant_family="pairwise_plan_comparison", view_names=rollout_views, coordination_blocks=1, objective="ce_plus_pairwise"),
        ),
        PlanVariant(
            name="P7_stacked_coordination_2block",
            short_name="P7",
            description="Two candidate-plan coordination blocks to measure wrong-to-right corrections.",
            model=PlanModelConfig(variant_family="stacked_coordination", view_names=rollout_views, coordination_blocks=2, objective="cross_entropy"),
        ),
        PlanVariant(
            name="P8_pyramidal_planning_views",
            short_name="P8",
            description="Composes state+goal, action+future, and future+constraints into additional latent planning states.",
            model=PlanModelConfig(
                variant_family="pyramidal_planning_views",
                view_names=cost_risk_views,
                coordination_blocks=1,
                objective="cross_entropy",
                pyramidal_composition=True,
            ),
        ),
        PlanVariant(
            name="P9_candidate_guided_world_interrogation",
            short_name="P9",
            description="Candidate representation participates while querying each planning view; leakage is checked by controls.",
            model=PlanModelConfig(
                variant_family="candidate_guided_world_interrogation",
                view_names=rollout_views,
                coordination_blocks=1,
                objective="cross_entropy",
                interaction_style="candidate_guided_world_interrogation",
            ),
        ),
        PlanVariant(
            name="P10_multi_avenue_planning",
            short_name="P10",
            description="Multiple planning avenues via outcome, constraint safety, cost, and risk views.",
            model=PlanModelConfig(variant_family="multi_avenue_planning", view_names=cost_risk_views, coordination_blocks=2, objective="ce_plus_margin"),
        ),
    ]


def _dataset_config(raw: object) -> PlanGridworldDatasetConfig:
    values = dict(raw or {})
    valid = {field.name for field in PlanGridworldDatasetConfig.__dataclass_fields__.values()}
    return PlanGridworldDatasetConfig(**{key: value for key, value in values.items() if key in valid})


def _training_config(raw: object) -> PlanTrainingConfig:
    values = dict(raw or {})
    valid = {field.name for field in PlanTrainingConfig.__dataclass_fields__.values()}
    return PlanTrainingConfig(**{key: value for key, value in values.items() if key in valid})


def _load_config(path: Path) -> Dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return str(requested)


def _clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cuda_max_memory(device: str) -> int:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return 0


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _compatible_dim(dim: int, heads: int) -> int:
    heads = max(1, int(heads))
    return int(math.ceil(int(dim) / heads) * heads)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.dim() == 4:
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=2).clamp(min=1.0)
        return (values * mask_f).sum(dim=2) / denom
    mask_f = mask.float().unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp(min=1.0)
    return (values * mask_f).sum(dim=1) / denom


def _flat_params(items: Sequence[Tuple[str, nn.Parameter]]) -> torch.Tensor:
    tensors = [parameter.detach().cpu().reshape(-1) for _name, parameter in items]
    return torch.cat(tensors) if tensors else torch.zeros(0)


def _l2_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    return float(torch.norm(a.float() - b.float()).item())


def _grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().norm().item())
        total += value * value
    return math.sqrt(total)


def _chunks(values: Sequence[int], size: int) -> Iterable[List[int]]:
    for index in range(0, len(values), max(1, int(size))):
        yield list(values[index : index + max(1, int(size))])


def _batched(values: Sequence[PlanGridworldExample], size: int) -> Iterable[Sequence[PlanGridworldExample]]:
    for index in range(0, len(values), max(1, int(size))):
        yield values[index : index + max(1, int(size))]


def _labels(examples: Sequence[PlanGridworldExample]) -> np.ndarray:
    return np.asarray([int(example.label) for example in examples], dtype=np.int64)


def _accuracy(predictions: Sequence[int] | np.ndarray, labels: Sequence[int] | np.ndarray) -> float:
    preds = np.asarray(predictions, dtype=np.int64)
    labs = np.asarray(labels, dtype=np.int64)
    if labs.size == 0:
        return 0.0
    return float(np.mean(preds == labs))


def _top1(logits: np.ndarray, labels: np.ndarray) -> float:
    if logits.size == 0 or len(labels) == 0:
        return 0.0
    return _accuracy(np.argmax(logits, axis=1), labels)


def _accuracy_by_key(predictions: np.ndarray, labels: np.ndarray, keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in sorted(set(keys)):
        indices = [idx for idx, value in enumerate(keys) if value == key]
        if not indices:
            continue
        out[str(key)] = {
            "n": int(len(indices)),
            "top1": _accuracy(predictions[indices], labels[indices]),
        }
    return out


def _family_confusion(predictions: np.ndarray, examples: Sequence[PlanGridworldExample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for pred, example in zip(predictions, examples):
        family = example.candidates[int(pred)].family
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def _candidate_public_debug(candidate: CandidateRollout) -> Dict[str, object]:
    return {
        "family_hidden_from_model": candidate.family,
        "reaches_goal": candidate.reaches_goal,
        "collision": candidate.collision,
        "near_miss": candidate.near_miss,
        "wrong_goal": candidate.wrong_goal,
        "valid": candidate.valid,
        "first_goal_step": int(candidate.first_goal_step),
        "final_position": list(candidate.final_position),
    }


def _exactly_one_best_by_hidden_metrics(example: PlanGridworldExample) -> bool:
    scores = []
    for candidate in example.candidates:
        if candidate.valid:
            scores.append((1, -candidate.first_goal_step))
        else:
            scores.append((0, -10_000 + int(candidate.near_miss) - int(candidate.collision)))
    return bool(scores and scores.count(max(scores)) == 1)


def _is_reverse(a: int, b: int) -> bool:
    return (int(a), int(b)) in {(0, 1), (1, 0), (2, 3), (3, 2)}


def _softmax_np(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _bootstrap_ci(values: Sequence[float], iterations: int = 1000, seed: int = 17) -> List[float]:
    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size == 0:
        return [0.0, 0.0]
    if vals.size == 1:
        return [float(vals[0]), float(vals[0])]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(iterations):
        sample = vals[rng.integers(0, vals.size, size=vals.size)]
        means.append(float(np.mean(sample)))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _best_ablation_drop(control_row: Dict[str, object]) -> float:
    clean = float(control_row.get("physical_role_order_shuffle", {}).get("top1", 0.0))
    drops = []
    for value in control_row.get("role_view_masking", {}).values():
        drops.append(max(0.0, clean - float(value.get("top1", clean))))
    return max(drops) if drops else 0.0


def _first_failed_gate(rows: Sequence[Dict[str, object]]) -> str:
    for row in rows:
        if not bool(row.get("pass", False)):
            return str(row.get("gate", "unknown"))
    return "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
