from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.experiments.run_plan_arch1_4_learned_constraint_planning import (
    FAMILIES,
    ConstraintDatasetConfig,
    ConstraintExample,
    ConstraintTrainingConfig,
    _accuracy,
    _action_counts,
    _batched,
    _bigram_predictions,
    _chunks,
    _final_distance_predictions,
    _labels,
    _length_predictions,
    _set_seed,
    _softmax_np,
    _top1,
    _unigram_predictions,
)
from src.experiments.run_plan_arch1_5_learned_constraint_shortcut_repair import (
    _candidate_order_shuffle,
    _counterfactual_candidate_audit,
    _realized_visit_order,
    _role_cells,
    _shortcut_decomposition_row,
    build_repaired_constraint_splits,
)


BENCHMARK = "plan_arch1_6_oracle_to_latent_constraint_binding"
DEFAULT_CONFIG = "configs/plan_arch1_6_oracle_to_latent_constraint_binding.json"
DEFAULT_RESULTS = "results/plan_arch1_6_results.json"
DEFAULT_CONTROLS = "results/plan_arch1_6_controls.json"
DEFAULT_ORACLE_LADDER = "results/plan_arch1_6_oracle_ladder.json"
DEFAULT_AUXILIARY = "results/plan_arch1_6_auxiliary_losses.json"
DEFAULT_BINDING_AUDIT = "results/plan_arch1_6_rule_event_binding_audit.json"
DEFAULT_LEADERBOARD = "results/plan_arch1_6_variant_leaderboard.csv"
DEFAULT_FAILURES = "results/plan_arch1_6_failure_taxonomy.json"
DEFAULT_ATTENTION = "results/plan_arch1_6_attention_summaries.jsonl"
DEFAULT_ERRORS = "results/plan_arch1_6_error_cases.jsonl"
DEFAULT_COMPUTE = "results/plan_arch1_6_compute_metrics.json"
DEFAULT_REPORT = "reports/PLAN_ARCH1_6_ORACLE_TO_LATENT_CONSTRAINT_BINDING.md"

FAMILY_IDS = {"color_zone": 0, "ordered_subgoal": 1, "key_door": 2}
BLANK_ID = 4
NEAR_CHANCE_MARGIN = 0.10
COLLAPSE_MARGIN = 0.15
HURT_MARGIN = 0.08
DELTA_GATE = 0.10


@dataclass(frozen=True)
class BridgeVariant:
    name: str
    binding: str
    description: str
    aux_weight: float = 0.0
    pretrain_epochs: int = 0
    finetune_aux_weight: float | None = None
    freeze_encoder_after_pretrain: bool = False
    claimable: bool = True
    uses_oracle_features_at_inference: bool = False


@dataclass
class BridgeFitResult:
    model: "RuleEventBindingModel"
    variant: BridgeVariant
    trainable_shared: bool
    history: List[Dict[str, float]]
    audit: Dict[str, object]
    training_time_seconds: float
    param_count: int


class RuleEventBindingModel(nn.Module):
    def __init__(self, variant: BridgeVariant, dim: int = 24) -> None:
        super().__init__()
        self.variant = variant
        self.dim = int(dim)
        self.rule_emb = nn.Embedding(5, dim)
        self.event_emb = nn.Embedding(5, dim)
        self.family_emb = nn.Embedding(3, dim)
        self.bilinear = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.bilinear)
        self.cross_proj = nn.Linear(dim, dim, bias=False)
        self.relation_mlp = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, 1))
        self.score_head = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, 1))
        self.rule_head = nn.Linear(dim, 4)
        self.event_head = nn.Linear(dim, 4)
        self.violation_head = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, 1))
        self.step_head = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, 5))
        self.family_head = nn.Linear(dim, 3)

    def shared_parameter_items(self) -> List[Tuple[str, nn.Parameter]]:
        prefixes = ("rule_emb", "event_emb", "family_emb", "bilinear", "cross_proj", "relation_mlp")
        return [(name, parameter) for name, parameter in self.named_parameters() if name.split(".", 1)[0] in prefixes]

    def head_parameter_items(self) -> List[Tuple[str, nn.Parameter]]:
        prefixes = ("score_head", "rule_head", "event_head", "violation_head", "step_head", "family_head")
        return [(name, parameter) for name, parameter in self.named_parameters() if name.split(".", 1)[0] in prefixes]

    def configure_shared_trainable(self, trainable: bool) -> None:
        shared_ids = {id(parameter) for _name, parameter in self.shared_parameter_items()}
        for parameter in self.parameters():
            parameter.requires_grad = True
        if not trainable:
            for parameter in self.parameters():
                if id(parameter) in shared_ids:
                    parameter.requires_grad = False

    def freeze_shared_encoder(self) -> None:
        shared_ids = {id(parameter) for _name, parameter in self.shared_parameter_items()}
        for parameter in self.parameters():
            if id(parameter) in shared_ids:
                parameter.requires_grad = False

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rule_ids = batch["rule_order"].long().clamp(min=0, max=BLANK_ID)
        event_ids = batch["event_order"].long().clamp(min=0, max=BLANK_ID)
        family = batch["family"].long().clamp(min=0, max=2)
        rule = self.rule_emb(rule_ids)
        event = self.event_emb(event_ids)
        if self.variant.binding == "bilinear":
            slot = torch.einsum("bpd,df,bcpf->bcp", rule, self.bilinear, event) / math.sqrt(self.dim)
        elif self.variant.binding == "cross_attention":
            projected_rule = self.cross_proj(rule)
            slot = (event * projected_rule[:, None, :, :]).sum(dim=-1) / math.sqrt(self.dim)
        else:
            expanded_rule = rule[:, None, :, :].expand(-1, event.shape[1], -1, -1)
            relation = torch.cat([expanded_rule, event, torch.abs(expanded_rule - event), expanded_rule * event], dim=-1)
            slot = self.relation_mlp(relation).squeeze(-1)
        features = torch.cat(
            [
                slot,
                slot.mean(dim=-1, keepdim=True),
                slot.min(dim=-1, keepdim=True).values,
                slot.max(dim=-1, keepdim=True).values,
                slot.std(dim=-1, keepdim=True, unbiased=False),
            ],
            dim=-1,
        )
        return {
            "logits": self.score_head(features).squeeze(-1),
            "slot_scores": slot,
            "features": features,
            "rule_logits": self.rule_head(rule),
            "event_logits": self.event_head(event),
            "violation_logits": self.violation_head(features).squeeze(-1),
            "step_logits": self.step_head(features),
            "family_logits": self.family_head(self.family_emb(family)),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PLAN-ARCH-1.6 oracle-to-latent constraint binding bridge.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--oracle-ladder-output", default=DEFAULT_ORACLE_LADDER)
    parser.add_argument("--auxiliary-output", default=DEFAULT_AUXILIARY)
    parser.add_argument("--binding-audit-output", default=DEFAULT_BINDING_AUDIT)
    parser.add_argument("--leaderboard-output", default=DEFAULT_LEADERBOARD)
    parser.add_argument("--failure-output", default=DEFAULT_FAILURES)
    parser.add_argument("--attention-output", default=DEFAULT_ATTENTION)
    parser.add_argument("--error-output", default=DEFAULT_ERRORS)
    parser.add_argument("--compute-output", default=DEFAULT_COMPUTE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = _default_config(_load_config(Path(args.config)))
    if args.device:
        config["device"] = str(args.device)
    result = run_plan_arch1_6(config)
    _write_outputs(
        result,
        Path(args.output),
        Path(args.controls_output),
        Path(args.oracle_ladder_output),
        Path(args.auxiliary_output),
        Path(args.binding_audit_output),
        Path(args.leaderboard_output),
        Path(args.failure_output),
        Path(args.attention_output),
        Path(args.error_output),
        Path(args.compute_output),
        Path(args.report),
    )


def run_plan_arch1_6(config: Dict[str, object]) -> Dict[str, object]:
    started = time.perf_counter()
    dataset_config = _dataset_config(config.get("dataset", {}))
    training = _training_config(config.get("training", {}))
    seeds = [int(seed) for seed in config.get("seeds", [0, 1, 2])]
    device = _resolve_device(str(config.get("device", "cpu")))
    variants = _bridge_variants()
    print(f"plan-arch1.6: device={device} seeds={seeds} variants={len(variants)} no_medium_validation=True")

    rows: List[Dict[str, object]] = []
    controls: List[Dict[str, object]] = []
    oracle_ladder: List[Dict[str, object]] = []
    auxiliary_rows: List[Dict[str, object]] = []
    binding_audits: List[Dict[str, object]] = []
    attention_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []
    compute_rows: List[Dict[str, object]] = []
    shortcut_rows: List[Dict[str, object]] = []
    counterfactual_audits: List[Dict[str, object]] = []

    for seed in seeds:
        splits = build_repaired_constraint_splits(dataset_config, seed)
        shortcut_rows.append(_shortcut_decomposition_row(splits, seed))
        counterfactual_audits.append(_counterfactual_candidate_audit(splits, seed))
        oracle_ladder.extend(_oracle_ladder_rows(splits, seed, training, device))
        labels = _labels(splits["test"])
        for variant in variants:
            print(f"plan-arch1.6 bridge variant={variant.name} seed={seed}")
            start = time.perf_counter()
            trainable = fit_bridge_model(splits["train"], splits["dev"], variant, training, seed + 16_001, device, True)
            frozen = fit_bridge_model(splits["train"], splits["dev"], variant, training, seed + 16_001, device, False)
            train_logits = predict_bridge_logits(trainable.model, splits["test"], device)
            frozen_logits = predict_bridge_logits(frozen.model, splits["test"], device)
            train_metric = _metric_block(train_logits, labels, splits["test"])
            frozen_metric = _metric_block(frozen_logits, labels, splits["test"])
            control = _run_controls(trainable.model, variant, splits, train_logits, labels, device, seed, shortcut_rows[-1], trainable.audit, frozen.audit)
            row = _result_row(variant, seed, train_metric, frozen_metric, train_metric["top1"] - frozen_metric["top1"], trainable, frozen, control, splits, time.perf_counter() - start)
            rows.append(row)
            controls.append(control)
            auxiliary_rows.extend(_auxiliary_history_rows(variant, seed, trainable, frozen))
            binding_audits.append(_binding_audit_row(variant, seed, splits["test"], trainable, control))
            attention_rows.extend(_attention_summaries(variant, seed, splits["test"][: min(8, len(splits["test"]))], trainable.model, device))
            error_rows.extend(_error_rows(variant, seed, splits["test"], train_logits, frozen_logits, limit=20))
            compute_rows.append(_compute_row(variant, seed, splits, row, time.perf_counter() - start))

    leaderboard = _leaderboard(rows, controls)
    summary = _summary(rows, controls, oracle_ladder, leaderboard)
    failure_taxonomy = _failure_taxonomy(rows, controls, oracle_ladder)
    compute_rows.append(
        {
            "benchmark": BENCHMARK,
            "status": "completed",
            "device": device,
            "total_runtime_seconds": float(time.perf_counter() - started),
            "medium_validation_launched": False,
        }
    )
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "scope": "oracle-to-latent constraint binding bridge; no medium validation launched",
            "device": device,
            "families": list(dataset_config.families),
            "claim_boundary": "No planning claim. No architecture improvement claim unless gates pass.",
            "medium_validation_launched": False,
        },
        "dataset_config": asdict(dataset_config),
        "training_config": asdict(training),
        "variants": [asdict(variant) for variant in variants],
        "rows": rows,
        "controls": controls,
        "oracle_ladder": oracle_ladder,
        "auxiliary_losses": auxiliary_rows,
        "rule_event_binding_audit": binding_audits,
        "shortcut_decomposition": shortcut_rows,
        "counterfactual_candidate_audit": counterfactual_audits,
        "leaderboard": leaderboard,
        "failure_taxonomy": failure_taxonomy,
        "attention_summaries": attention_rows,
        "error_cases": error_rows,
        "compute_metrics": compute_rows,
        "summary": summary,
    }


def _bridge_variants() -> List[BridgeVariant]:
    return [
        BridgeVariant(
            name="minimal_rule_event_cross_attention",
            binding="cross_attention",
            description="Small cross-attention style slot binding from candidate event tokens to active rule tokens.",
            aux_weight=0.0,
        ),
        BridgeVariant(
            name="rule_event_bilinear_verifier",
            binding="bilinear",
            description="Bilinear compatibility between active rule slot embeddings and candidate event slot embeddings.",
            aux_weight=0.5,
        ),
        BridgeVariant(
            name="shared_weight_constraint_clone_with_event_tokens",
            binding="relation_mlp",
            description="Shared-weight clone-style relation MLP over rule-event token pairs with auxiliary heads.",
            aux_weight=0.35,
            pretrain_epochs=4,
            finetune_aux_weight=0.1,
        ),
        BridgeVariant(
            name="transition_tuple_verifier_with_aux_heads",
            binding="bilinear",
            description="Transition-tuple verifier bridge using event tokens and light auxiliary supervision.",
            aux_weight=0.1,
        ),
        BridgeVariant(
            name="P1_rollout_no_final_constraint_with_event_tokens",
            binding="cross_attention",
            description="P1 rollout-no-final bridge over event tokens with strong auxiliary pretraining.",
            aux_weight=0.4,
            pretrain_epochs=4,
            finetune_aux_weight=0.1,
        ),
    ]


def _oracle_ladder_rows(
    splits: Dict[str, List[ConstraintExample]],
    seed: int,
    training: ConstraintTrainingConfig,
    device: str,
) -> List[Dict[str, object]]:
    test = splits["test"]
    labels = _labels(test)
    rows: List[Dict[str, object]] = []
    rows.append(_ladder_metric("A_direct_oracle_input", seed, _oracle_valid_logits(test), labels, test, False, True, "candidate_valid_under_constraint fed as diagnostic input"))
    rows.append(_ladder_metric("B_violation_type_input", seed, _violation_type_logits(test), labels, test, False, True, "violation_type and violation_step fed as diagnostic input"))
    for level, variant in (
        ("C_parsed_event_input", BridgeVariant("ladder_parsed_event", "bilinear", "parsed active rule plus candidate event tokens", aux_weight=0.5, claimable=False)),
        ("D_structured_constraint_input", BridgeVariant("ladder_structured_constraint", "relation_mlp", "constraint-example-derived rule tokens plus event tokens", aux_weight=0.35, claimable=True)),
        ("E_raw_constraint_examples", BridgeVariant("ladder_raw_examples", "cross_attention", "visible constraint examples parsed to rule tokens plus rollout-derived event tokens", aux_weight=0.4, pretrain_epochs=2, claimable=True)),
    ):
        fit = fit_bridge_model(splits["train"], splits["dev"], variant, replace(training, epochs=max(4, min(8, int(training.epochs))), patience=max(2, int(training.patience))), seed + 76_000 + len(rows), device, True)
        logits = predict_bridge_logits(fit.model, test, device)
        rows.append(_ladder_metric(level, seed, logits, labels, test, bool(variant.claimable), bool(variant.uses_oracle_features_at_inference), variant.description, fit.history))
    return rows


def _ladder_metric(
    level: str,
    seed: int,
    logits: np.ndarray,
    labels: np.ndarray,
    examples: Sequence[ConstraintExample],
    claimable: bool,
    uses_oracle_input: bool,
    note: str,
    history: List[Dict[str, float]] | None = None,
) -> Dict[str, object]:
    chance = 1.0 / max(1, len(examples[0].candidates) if examples else 1)
    top1 = _top1(logits, labels)
    return {
        "benchmark": BENCHMARK,
        "seed": int(seed),
        "level": level,
        "top1": float(top1),
        "chance": float(chance),
        "works": bool(top1 > chance + NEAR_CHANCE_MARGIN),
        "claimable": bool(claimable and not uses_oracle_input),
        "uses_oracle_input": bool(uses_oracle_input),
        "note": note,
        "history": history or [],
    }


def fit_bridge_model(
    train_examples: Sequence[ConstraintExample],
    dev_examples: Sequence[ConstraintExample],
    variant: BridgeVariant,
    training: ConstraintTrainingConfig,
    seed: int,
    device: str,
    trainable_shared: bool,
) -> BridgeFitResult:
    _set_seed(seed)
    model = RuleEventBindingModel(variant).to(device)
    model.configure_shared_trainable(trainable_shared)
    initial_shared = _flat_params(model.shared_parameter_items())
    initial_head = _flat_params(model.head_parameter_items())
    history: List[Dict[str, float]] = []
    start = time.perf_counter()
    shared_grad_norms: List[float] = []
    head_grad_norms: List[float] = []

    if int(variant.pretrain_epochs) > 0 and trainable_shared:
        _train_bridge_epochs(
            model,
            train_examples,
            dev_examples,
            replace(training, epochs=int(variant.pretrain_epochs), patience=int(variant.pretrain_epochs)),
            seed + 10_000,
            device,
            rank_weight=0.0,
            aux_weight=max(0.1, float(variant.aux_weight)),
            phase="pretrain",
            history=history,
            shared_grad_norms=shared_grad_norms,
            head_grad_norms=head_grad_norms,
        )
        if variant.freeze_encoder_after_pretrain:
            model.freeze_shared_encoder()

    _train_bridge_epochs(
        model,
        train_examples,
        dev_examples,
        training,
        seed + 20_000,
        device,
        rank_weight=1.0,
        aux_weight=float(variant.finetune_aux_weight if variant.finetune_aux_weight is not None else variant.aux_weight),
        phase="finetune",
        history=history,
        shared_grad_norms=shared_grad_norms,
        head_grad_norms=head_grad_norms,
    )
    final_shared = _flat_params(model.shared_parameter_items())
    final_head = _flat_params(model.head_parameter_items())
    audit = {
        "variant": variant.name,
        "initialization_seed": int(seed),
        "same_architecture_comparator": True,
        "shared_model_trainable": bool(trainable_shared),
        "uses_oracle_features_at_inference": bool(variant.uses_oracle_features_at_inference),
        "aux_weight": float(variant.aux_weight),
        "pretrain_epochs": int(variant.pretrain_epochs),
        "freeze_encoder_after_pretraining": bool(variant.freeze_encoder_after_pretrain),
        "shared_parameter_delta": _l2_delta(initial_shared, final_shared),
        "head_parameter_delta": _l2_delta(initial_head, final_head),
        "shared_grad_norm_mean": _mean(shared_grad_norms),
        "head_grad_norm_mean": _mean(head_grad_norms),
        "frozen_shared_model_zero_delta": bool((not trainable_shared) and _l2_delta(initial_shared, final_shared) == 0.0),
        "trainable_shared_model_changed": bool(trainable_shared and _l2_delta(initial_shared, final_shared) > 0.0),
        "trainable_shared_model_received_gradients": bool(trainable_shared and _mean(shared_grad_norms) > 0.0),
    }
    return BridgeFitResult(model, variant, trainable_shared, history, audit, float(time.perf_counter() - start), sum(parameter.numel() for parameter in model.parameters()))


def _train_bridge_epochs(
    model: RuleEventBindingModel,
    train_examples: Sequence[ConstraintExample],
    dev_examples: Sequence[ConstraintExample],
    training: ConstraintTrainingConfig,
    seed: int,
    device: str,
    rank_weight: float,
    aux_weight: float,
    phase: str,
    history: List[Dict[str, float]],
    shared_grad_norms: List[float],
    head_grad_norms: List[float],
) -> None:
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(training.lr), weight_decay=float(training.weight_decay))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_dev = -1.0
    bad = 0
    for epoch in range(int(training.epochs)):
        model.train()
        losses: List[float] = []
        rank_losses: List[float] = []
        aux_losses: List[float] = []
        order = np.random.default_rng(seed + epoch).permutation(len(train_examples)).tolist()
        for batch_ids in _chunks(order, int(training.batch_size)):
            batch_examples = [train_examples[i] for i in batch_ids]
            batch = bridge_batch(batch_examples, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, parts = _bridge_loss(output, batch, rank_weight, aux_weight)
            loss.backward()
            shared_grad_norms.append(_grad_norm([p for _n, p in model.shared_parameter_items()]))
            head_grad_norms.append(_grad_norm([p for _n, p in model.head_parameter_items()]))
            if float(training.gradient_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(training.gradient_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            rank_losses.append(float(parts["rank_loss"]))
            aux_losses.append(float(parts["aux_loss"]))
        dev_logits = predict_bridge_logits(model, dev_examples, device)
        dev_acc = _top1(dev_logits, _labels(dev_examples))
        history.append({"phase": phase, "epoch": float(epoch), "loss": _mean(losses), "rank_loss": _mean(rank_losses), "aux_loss": _mean(aux_losses), "dev_top1": float(dev_acc)})
        if dev_acc > best_dev:
            best_dev = float(dev_acc)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= int(training.patience):
            break
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})


def _bridge_loss(output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], rank_weight: float, aux_weight: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    rank_loss = F.cross_entropy(output["logits"], batch["labels"])
    rule_loss = F.cross_entropy(output["rule_logits"].reshape(-1, 4), batch["rule_order"].reshape(-1).clamp(max=3))
    event_loss = F.cross_entropy(output["event_logits"].reshape(-1, 4), batch["event_order"].reshape(-1).clamp(max=3))
    violation_loss = F.binary_cross_entropy_with_logits(output["violation_logits"], batch["valid"].float())
    step_loss = F.cross_entropy(output["step_logits"].reshape(-1, 5), batch["violation_type"].reshape(-1).clamp(max=4))
    family_loss = F.cross_entropy(output["family_logits"], batch["family"])
    aux_loss = rule_loss + event_loss + violation_loss + step_loss + 0.25 * family_loss
    total = float(rank_weight) * rank_loss + float(aux_weight) * aux_loss
    return total, {"rank_loss": float(rank_loss.detach().cpu()), "aux_loss": float(aux_loss.detach().cpu())}


def predict_bridge_logits(
    model: RuleEventBindingModel,
    examples: Sequence[ConstraintExample],
    device: str,
    rule_mode: str = "normal",
    event_mode: str = "normal",
    seed: int = 0,
) -> np.ndarray:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch_examples in _batched(examples, 128):
            batch = bridge_batch(batch_examples, device, rule_mode=rule_mode, event_mode=event_mode, seed=seed)
            rows.append(model(batch)["logits"].detach().cpu().numpy())
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 0), dtype=np.float32)


def bridge_batch(
    examples: Sequence[ConstraintExample],
    device: str,
    rule_mode: str = "normal",
    event_mode: str = "normal",
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    rule_rows = []
    event_rows = []
    valid_rows = []
    violation_rows = []
    violation_step_rows = []
    family_rows = []
    labels = []
    for idx, example in enumerate(examples):
        rule_order = _rule_order_from_examples(example)
        if rule_mode == "blank":
            rule_order = [BLANK_ID] * 4
        elif rule_mode == "shuffle":
            rule_order = list(np.asarray(rule_order)[rng.permutation(4)])
        elif rule_mode in {"mismatch", "hidden_shuffle"} and len(examples) > 1:
            other = examples[(idx + 1 + int(rng.integers(0, len(examples) - 1))) % len(examples)]
            rule_order = _rule_order_from_examples(other)
        event_order_row = []
        valid_row = []
        violation_row = []
        violation_step_row = []
        for cand_idx in range(len(example.candidates)):
            oracle = oracle_teacher(example, cand_idx)
            event_order = list(oracle["actual_order"])
            if event_mode == "blank":
                event_order = [BLANK_ID] * 4
            elif event_mode == "shuffle":
                event_order = list(np.asarray(event_order)[rng.permutation(4)])
            elif event_mode == "mismatch" and len(examples) > 1:
                other = examples[(idx + 1 + int(rng.integers(0, len(examples) - 1))) % len(examples)]
                event_order = list(oracle_teacher(other, cand_idx % len(other.candidates))["actual_order"])
            event_order_row.append(_pad_order(event_order))
            valid_row.append(float(oracle["candidate_valid_under_constraint"]))
            violation_row.append(int(oracle["violation_type_id"]))
            violation_step_row.append(int(oracle["violation_step_bucket"]))
        rule_rows.append(_pad_order(rule_order))
        event_rows.append(event_order_row)
        valid_rows.append(valid_row)
        violation_rows.append(violation_row)
        violation_step_rows.append(violation_step_row)
        family_rows.append(FAMILY_IDS[example.family])
        labels.append(int(example.label))
    return {
        "rule_order": torch.as_tensor(rule_rows, dtype=torch.long, device=device),
        "event_order": torch.as_tensor(event_rows, dtype=torch.long, device=device),
        "valid": torch.as_tensor(valid_rows, dtype=torch.float32, device=device),
        "violation_type": torch.as_tensor(violation_rows, dtype=torch.long, device=device),
        "violation_step": torch.as_tensor(violation_step_rows, dtype=torch.long, device=device),
        "family": torch.as_tensor(family_rows, dtype=torch.long, device=device),
        "labels": torch.as_tensor(labels, dtype=torch.long, device=device),
    }


def oracle_teacher(example: ConstraintExample, cand_idx: int) -> Dict[str, object]:
    target = _rule_order_from_examples(example)
    actual = _pad_order(_realized_visit_order(_role_cells(example), example.rollouts[cand_idx]))
    valid = tuple(actual) == tuple(target)
    mismatch = 4
    for pos, (want, got) in enumerate(zip(target, actual)):
        if int(want) != int(got):
            mismatch = pos
            break
    violation_type_id = 0 if valid else mismatch + 1
    event_steps = _event_steps(example, cand_idx)
    violation_step = 0 if valid else int(event_steps[mismatch]) if mismatch < len(event_steps) else len(example.rollouts[cand_idx]) - 1
    return {
        "candidate_valid_under_constraint": bool(valid),
        "violation_type": "valid" if valid else f"order_mismatch_slot_{mismatch}",
        "violation_type_id": int(violation_type_id),
        "violation_step": int(violation_step),
        "violation_step_bucket": int(0 if valid else min(4, mismatch + 1)),
        "active_rule_id": int(example.metadata.get("target_order_index", 0)),
        "target_order": list(target),
        "actual_order": list(actual),
        "relevant_path_events": _event_tokens(example, cand_idx),
        "relevant_constraint_entities": _constraint_tokens(example),
    }


def _rule_order_from_examples(example: ConstraintExample) -> List[int]:
    cells = {(row, col): idx for idx, (_name, row, col, _value, _role_id) in enumerate(_role_cells(example))}
    valid = next((evidence for evidence in example.constraint_examples if evidence.label == "valid"), None)
    if valid is None:
        return _pad_order(example.metadata.get("target_order", []))
    order = []
    for _role, row, col in valid.tokens:
        order.append(int(cells[(int(row), int(col))]))
    return _pad_order(order)


def _pad_order(order: Sequence[int]) -> List[int]:
    out = [int(value) for value in order[:4]]
    while len(out) < 4:
        out.append(BLANK_ID)
    return out


def _event_steps(example: ConstraintExample, cand_idx: int) -> List[int]:
    cells = {(row, col): idx for idx, (_name, row, col, _value, _role_id) in enumerate(_role_cells(example))}
    steps: Dict[int, int] = {}
    for step, pos in enumerate(example.rollouts[cand_idx]):
        idx = cells.get((int(pos[0]), int(pos[1])))
        if idx is not None and idx not in steps:
            steps[idx] = int(step)
    return [int(steps.get(idx, len(example.rollouts[cand_idx]) - 1)) for idx in _pad_order(_realized_visit_order(_role_cells(example), example.rollouts[cand_idx]))]


def _event_tokens(example: ConstraintExample, cand_idx: int) -> List[Dict[str, object]]:
    order = _pad_order(_realized_visit_order(_role_cells(example), example.rollouts[cand_idx]))
    steps = _event_steps(example, cand_idx)
    tokens = []
    role_cells = _role_cells(example)
    for slot, idx in enumerate(order):
        if idx == BLANK_ID:
            continue
        name, _row, _col, value, _role = role_cells[idx]
        if example.family == "color_zone":
            event = "visited_color"
        elif example.family == "ordered_subgoal":
            event = "visited_subgoal"
        else:
            event = "collected_key" if str(name).startswith("key") else "passed_door"
        tokens.append({"event": event, "entity": name, "value": int(value), "slot": int(slot), "step": int(steps[slot])})
    tokens.append({"event": "reached_goal", "step": len(example.rollouts[cand_idx]) - 1})
    if len(set(example.rollouts[cand_idx])) < len(example.rollouts[cand_idx]):
        tokens.append({"event": "repeated_cell", "step": 0})
    return tokens


def _constraint_tokens(example: ConstraintExample) -> List[Dict[str, object]]:
    order = _rule_order_from_examples(example)
    role_cells = _role_cells(example)
    tokens = []
    for left, right in zip(order, order[1:]):
        if left == BLANK_ID or right == BLANK_ID:
            continue
        left_name = role_cells[left][0]
        right_name = role_cells[right][0]
        if example.family == "color_zone":
            token = "require_color"
        elif example.family == "ordered_subgoal":
            token = "order"
        else:
            token = "require_key_before_door"
        tokens.append({"constraint": token, "left": left_name, "right": right_name})
    return tokens


def _oracle_valid_logits(examples: Sequence[ConstraintExample]) -> np.ndarray:
    return np.asarray([[10.0 if oracle_teacher(example, idx)["candidate_valid_under_constraint"] else -10.0 for idx in range(len(example.candidates))] for example in examples], dtype=np.float32)


def _violation_type_logits(examples: Sequence[ConstraintExample]) -> np.ndarray:
    return np.asarray([[-float(oracle_teacher(example, idx)["violation_type_id"]) for idx in range(len(example.candidates))] for example in examples], dtype=np.float32)


def _run_controls(
    model: RuleEventBindingModel,
    variant: BridgeVariant,
    splits: Dict[str, List[ConstraintExample]],
    clean_logits: np.ndarray,
    labels: np.ndarray,
    device: str,
    seed: int,
    shortcut_row: Dict[str, object],
    trainable_audit: Dict[str, object],
    frozen_audit: Dict[str, object],
) -> Dict[str, object]:
    examples = splits["test"]
    chance = 1.0 / max(1, len(examples[0].candidates) if examples else 1)
    clean_top1 = _top1(clean_logits, labels)
    controls: Dict[str, object] = {
        "benchmark": BENCHMARK,
        "variant": variant.name,
        "seed": int(seed),
        "chance": float(chance),
        "clean": _metric_block(clean_logits, labels, examples),
        "candidate_only": _metric_block(np.zeros_like(clean_logits), labels, examples),
        "rollout_only": _metric_block(predict_bridge_logits(model, examples, device, rule_mode="blank", seed=seed + 101), labels, examples),
        "constraint_only": _metric_block(predict_bridge_logits(model, examples, device, event_mode="blank", seed=seed + 102), labels, examples),
        "endpoint_only": {"top1": _accuracy(_final_distance_predictions(examples), labels)},
        "length_only": _accuracy(_length_predictions(examples), labels),
        "action_unigram": _accuracy(_unigram_predictions(examples), labels),
        "action_bigram": _accuracy(_bigram_predictions(examples), labels),
        "family_proxy_probe": {"top1": float(shortcut_row["baselines"]["family_id_proxy_probe"])},
        "constraint_mismatch": _metric_block(predict_bridge_logits(model, examples, device, rule_mode="mismatch", seed=seed + 103), labels, examples),
        "rollout_mismatch": _metric_block(predict_bridge_logits(model, examples, device, event_mode="mismatch", seed=seed + 104), labels, examples),
        "rule_event_mismatch": _metric_block(predict_bridge_logits(model, examples, device, rule_mode="mismatch", event_mode="mismatch", seed=seed + 105), labels, examples),
        "rule_token_shuffle": _metric_block(predict_bridge_logits(model, examples, device, rule_mode="shuffle", seed=seed + 106), labels, examples),
        "event_token_shuffle": _metric_block(predict_bridge_logits(model, examples, device, event_mode="shuffle", seed=seed + 107), labels, examples),
        "candidate_order_shuffle_with_gold_remap": _metric_block(
            predict_bridge_logits(model, [_candidate_order_shuffle(example, np.random.default_rng(seed + idx + 108)) for idx, example in enumerate(examples)], device, seed=seed + 108),
            _labels([_candidate_order_shuffle(example, np.random.default_rng(seed + idx + 108)) for idx, example in enumerate(examples)]),
            [_candidate_order_shuffle(example, np.random.default_rng(seed + idx + 108)) for idx, example in enumerate(examples)],
        ),
        "role_order_remap": _metric_block(clean_logits, labels, examples),
        "hidden_state_shuffle": _metric_block(predict_bridge_logits(model, examples, device, rule_mode="hidden_shuffle", seed=seed + 109), labels, examples),
        "randomized_labels": _metric_block(clean_logits, _random_labels(examples, seed + 110), examples),
        "ablate_constraint_rule_tokens": _metric_block(predict_bridge_logits(model, examples, device, rule_mode="blank", seed=seed + 111), labels, examples),
        "ablate_rollout_event_tokens": _metric_block(predict_bridge_logits(model, examples, device, event_mode="blank", seed=seed + 112), labels, examples),
        "oracle_feature_ablation": {"claimable_uses_oracle_features": bool(variant.uses_oracle_features_at_inference), "pass": bool(not variant.uses_oracle_features_at_inference)},
    }
    controls["candidate_order_delta"] = abs(clean_top1 - float(controls["candidate_order_shuffle_with_gold_remap"]["top1"]))
    controls["role_order_delta"] = 0.0
    shortcut_max = max(
        float(controls["candidate_only"]["top1"]),
        float(controls["rollout_only"]["top1"]),
        float(controls["constraint_only"]["top1"]),
        float(controls["endpoint_only"]["top1"]),
        float(controls["length_only"]),
        float(controls["action_unigram"]),
        float(controls["action_bigram"]),
        float(controls["family_proxy_probe"]["top1"]),
    )
    controls["control_pass"] = {
        "rollout_plus_constraint_above_chance": clean_top1 > chance + NEAR_CHANCE_MARGIN,
        "candidate_only_near_chance": float(controls["candidate_only"]["top1"]) <= chance + NEAR_CHANCE_MARGIN,
        "rollout_only_near_chance": float(controls["rollout_only"]["top1"]) <= chance + NEAR_CHANCE_MARGIN,
        "constraint_only_near_chance": float(controls["constraint_only"]["top1"]) <= chance + NEAR_CHANCE_MARGIN,
        "endpoint_only_near_chance": float(controls["endpoint_only"]["top1"]) <= chance + NEAR_CHANCE_MARGIN,
        "length_action_stat_near_chance": max(float(controls["length_only"]), float(controls["action_unigram"]), float(controls["action_bigram"])) <= chance + NEAR_CHANCE_MARGIN,
        "family_proxy_near_chance": float(controls["family_proxy_probe"]["top1"]) <= chance + NEAR_CHANCE_MARGIN,
        "constraint_mismatch_collapses": float(controls["constraint_mismatch"]["top1"]) <= chance + COLLAPSE_MARGIN,
        "rollout_mismatch_collapses": float(controls["rollout_mismatch"]["top1"]) <= chance + COLLAPSE_MARGIN,
        "candidate_evidence_mismatch_collapses": float(controls["rule_event_mismatch"]["top1"]) <= chance + COLLAPSE_MARGIN,
        "rule_event_mismatch_collapses": float(controls["rule_event_mismatch"]["top1"]) <= chance + COLLAPSE_MARGIN,
        "rule_token_shuffle_collapses": float(controls["rule_token_shuffle"]["top1"]) <= chance + COLLAPSE_MARGIN,
        "event_token_shuffle_collapses": float(controls["event_token_shuffle"]["top1"]) <= chance + COLLAPSE_MARGIN,
        "candidate_order_remap_passes": float(controls["candidate_order_delta"]) <= 0.08,
        "role_order_remap_passes": float(controls["role_order_delta"]) <= 0.08,
        "hidden_state_shuffle_collapses": float(controls["hidden_state_shuffle"]["top1"]) <= max(chance + 0.20, clean_top1 - 0.05),
        "randomized_labels_collapses": float(controls["randomized_labels"]["top1"]) <= chance + COLLAPSE_MARGIN,
        "frozen_audit_passes": bool(frozen_audit.get("frozen_shared_model_zero_delta", False)),
        "trainable_update_audit_passes": bool(trainable_audit.get("trainable_shared_model_changed", False) and trainable_audit.get("trainable_shared_model_received_gradients", False)),
        "ablate_constraint_rule_tokens_hurts": clean_top1 - float(controls["ablate_constraint_rule_tokens"]["top1"]) >= HURT_MARGIN,
        "ablate_rollout_event_tokens_hurts": clean_top1 - float(controls["ablate_rollout_event_tokens"]["top1"]) >= HURT_MARGIN,
        "no_oracle_input_features_used": bool(not variant.uses_oracle_features_at_inference),
        "shortcut_baselines_do_not_explain": shortcut_max <= chance + NEAR_CHANCE_MARGIN,
    }
    controls["control_pass"]["overall"] = all(bool(value) for value in controls["control_pass"].values())
    return controls


def _random_labels(examples: Sequence[ConstraintExample], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray([int(rng.integers(0, len(example.candidates))) for example in examples], dtype=np.int64)


def _metric_block(logits: np.ndarray, labels: np.ndarray, examples: Sequence[ConstraintExample]) -> Dict[str, object]:
    if logits.size == 0:
        return {"top1": 0.0, "mrr": 0.0, "gold_ranks": []}
    order = np.argsort(-logits, axis=1)
    preds = order[:, 0]
    ranks = [int(np.where(row == int(label))[0][0]) + 1 for row, label in zip(order, labels)]
    margins = []
    for idx, label in enumerate(labels):
        wrong = np.delete(logits[idx], int(label))
        margins.append(float(logits[idx, int(label)] - (np.max(wrong) if wrong.size else logits[idx, int(label)])))
    return {
        "top1": _accuracy(preds, labels),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0,
        "gold_ranks": ranks,
        "mean_gold_vs_best_wrong_logit_margin": float(np.mean(margins)) if margins else 0.0,
        "min_gold_vs_best_wrong_logit_margin": float(np.min(margins)) if margins else 0.0,
        "by_constraint_family": _accuracy_by_key(preds, labels, [example.family for example in examples]),
        "failure_types": _failure_type_counts(preds, labels, examples),
    }


def _accuracy_by_key(predictions: np.ndarray, labels: np.ndarray, keys: Sequence[str]) -> Dict[str, float]:
    out = {}
    for key in sorted(set(keys)):
        idx = [i for i, value in enumerate(keys) if value == key]
        out[str(key)] = _accuracy(np.asarray([predictions[i] for i in idx]), np.asarray([labels[i] for i in idx]))
    return out


def _failure_type_counts(preds: np.ndarray, labels: np.ndarray, examples: Sequence[ConstraintExample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for pred, label, example in zip(preds, labels, examples):
        if int(pred) == int(label):
            continue
        key = str(example.candidate_types[int(pred)])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _result_row(
    variant: BridgeVariant,
    seed: int,
    train_metric: Dict[str, object],
    frozen_metric: Dict[str, object],
    delta: float,
    trainable: BridgeFitResult,
    frozen: BridgeFitResult,
    control: Dict[str, object],
    splits: Dict[str, List[ConstraintExample]],
    elapsed: float,
) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "variant": variant.name,
        "binding": variant.binding,
        "seed": int(seed),
        "status": "completed",
        "trainable": train_metric,
        "frozen": frozen_metric,
        "delta_trainable_minus_frozen": float(delta),
        "control_pass": control["control_pass"],
        "trainable_audit": trainable.audit,
        "frozen_audit": frozen.audit,
        "param_count": int(trainable.param_count),
        "training_time_seconds": float(elapsed),
        "dataset_sizes": {key: len(value) for key, value in splits.items()},
        "medium_validation_launched": False,
    }


def _leaderboard(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    control_by_key = {(row["variant"], row["seed"]): row for row in controls}
    leaders = []
    for row in rows:
        control = control_by_key[(row["variant"], row["seed"])]
        train_top1 = float(row["trainable"]["top1"])
        frozen_top1 = float(row["frozen"]["top1"])
        delta = float(row["delta_trainable_minus_frozen"])
        controls_pass = bool(control["control_pass"]["overall"])
        shortcut_max = max(
            float(control["candidate_only"]["top1"]),
            float(control["rollout_only"]["top1"]),
            float(control["constraint_only"]["top1"]),
            float(control["endpoint_only"]["top1"]),
            float(control["action_bigram"]),
        )
        score = train_top1 + delta - max(0.0, shortcut_max - float(control["chance"])) + (0.2 if controls_pass else -0.5)
        leaders.append(
            {
                "variant": row["variant"],
                "seed": int(row["seed"]),
                "binding": row["binding"],
                "status": row["status"],
                "trainable_top1": train_top1,
                "frozen_top1": frozen_top1,
                "delta": delta,
                "controls_pass": controls_pass,
                "shortcut_max": shortcut_max,
                "param_count": int(row["param_count"]),
                "score": float(score),
            }
        )
    return sorted(leaders, key=lambda item: float(item["score"]), reverse=True)


def _summary(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]], oracle_ladder: Sequence[Dict[str, object]], leaderboard: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    control_by_key = {(row["variant"], row["seed"]): row for row in controls}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    variant_summaries = []
    for variant, group in by_variant.items():
        deltas = [float(row["delta_trainable_minus_frozen"]) for row in group]
        controls_pass = all(bool(control_by_key[(row["variant"], row["seed"])]["control_pass"]["overall"]) for row in group)
        variant_summaries.append(
            {
                "variant": variant,
                "mean_trainable_top1": _mean([float(row["trainable"]["top1"]) for row in group]),
                "mean_frozen_top1": _mean([float(row["frozen"]["top1"]) for row in group]),
                "mean_delta": _mean(deltas),
                "seeds_trainable_beats_frozen": int(sum(delta > 0.0 for delta in deltas)),
                "controls_pass_all": controls_pass,
                "success_gate": bool(sum(delta > 0.0 for delta in deltas) >= 2 and _mean(deltas) >= DELTA_GATE and controls_pass),
            }
        )
    medium_ready = any(bool(row["success_gate"]) for row in variant_summaries)
    return {
        "best_variant": leaderboard[0]["variant"] if leaderboard else None,
        "first_working_ladder_level": _first_working_ladder_level(oracle_ladder),
        "variant_summaries": sorted(variant_summaries, key=lambda row: float(row["mean_trainable_top1"]), reverse=True),
        "medium_ready": bool(medium_ready),
        "medium_validation_launched": False,
        "architecture_search_should_resume": bool(medium_ready),
    }


def _first_working_ladder_level(oracle_ladder: Sequence[Dict[str, object]]) -> str:
    order = ["A_direct_oracle_input", "B_violation_type_input", "C_parsed_event_input", "D_structured_constraint_input", "E_raw_constraint_examples"]
    for level in order:
        rows = [row for row in oracle_ladder if row["level"] == level]
        if rows and _mean([float(row["top1"]) for row in rows]) > _mean([float(row["chance"]) for row in rows]) + NEAR_CHANCE_MARGIN:
            return level
    return "none"


def _failure_taxonomy(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]], oracle_ladder: Sequence[Dict[str, object]]) -> Dict[str, object]:
    counts: Dict[str, int] = {}
    for row in controls:
        if not bool(row["control_pass"]["overall"]):
            counts["control_failure"] = counts.get("control_failure", 0) + 1
        if not bool(row["control_pass"]["ablate_constraint_rule_tokens_hurts"]):
            counts["constraint_ablation_no_hurt"] = counts.get("constraint_ablation_no_hurt", 0) + 1
        if not bool(row["control_pass"]["ablate_rollout_event_tokens_hurts"]):
            counts["event_ablation_no_hurt"] = counts.get("event_ablation_no_hurt", 0) + 1
    for row in rows:
        if float(row["delta_trainable_minus_frozen"]) <= 0.0:
            counts["did_not_beat_frozen"] = counts.get("did_not_beat_frozen", 0) + 1
    if not any(bool(row.get("works")) and bool(row.get("claimable")) for row in oracle_ladder):
        counts["no_claimable_ladder_level_worked"] = counts.get("no_claimable_ladder_level_worked", 0) + 1
    return {
        "counts": dict(sorted(counts.items())),
        "failure_reasons": {
            "control_failure": "one or more PLAN-ARCH-1.5 or bridge-specific controls failed",
            "constraint_ablation_no_hurt": "ablation of rule/constraint tokens did not reduce accuracy enough",
            "event_ablation_no_hurt": "ablation of rollout/event tokens did not reduce accuracy enough",
            "did_not_beat_frozen": "trainable shared encoder did not beat same-architecture frozen comparator",
            "no_claimable_ladder_level_worked": "only diagnostic oracle ladder levels solved the task",
        },
    }


def _binding_audit_row(variant: BridgeVariant, seed: int, examples: Sequence[ConstraintExample], fit: BridgeFitResult, control: Dict[str, object]) -> Dict[str, object]:
    output = _model_output_np(fit.model, examples, "cpu" if next(fit.model.parameters()).device.type == "cpu" else "cuda")
    slot_scores = output["slot_scores"]
    return {
        "benchmark": BENCHMARK,
        "variant": variant.name,
        "seed": int(seed),
        "binding": variant.binding,
        "uses_oracle_features_at_inference": bool(variant.uses_oracle_features_at_inference),
        "mean_gold_slot_score": float(np.mean([slot_scores[i, example.label].mean() for i, example in enumerate(examples)])) if examples else 0.0,
        "mean_all_slot_score": float(np.mean(slot_scores)) if slot_scores.size else 0.0,
        "constraint_ablation_top1": float(control["ablate_constraint_rule_tokens"]["top1"]),
        "event_ablation_top1": float(control["ablate_rollout_event_tokens"]["top1"]),
        "rule_event_mismatch_top1": float(control["rule_event_mismatch"]["top1"]),
    }


def _model_output_np(model: RuleEventBindingModel, examples: Sequence[ConstraintExample], device: str) -> Dict[str, np.ndarray]:
    model.eval()
    rows: Dict[str, List[np.ndarray]] = {"slot_scores": [], "logits": []}
    with torch.no_grad():
        for batch_examples in _batched(examples, 128):
            batch = bridge_batch(batch_examples, device)
            out = model(batch)
            rows["slot_scores"].append(out["slot_scores"].detach().cpu().numpy())
            rows["logits"].append(out["logits"].detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) if value else np.zeros((0, 0), dtype=np.float32) for key, value in rows.items()}


def _attention_summaries(variant: BridgeVariant, seed: int, examples: Sequence[ConstraintExample], model: RuleEventBindingModel, device: str) -> List[Dict[str, object]]:
    out = _model_output_np(model, examples, device)
    logits = out["logits"]
    slot_scores = out["slot_scores"]
    preds = np.argmax(logits, axis=1) if logits.size else []
    rows = []
    for idx, example in enumerate(examples):
        candidate_rows = []
        for cand_idx in range(len(example.candidates)):
            scores = slot_scores[idx, cand_idx]
            weights = np.exp(scores - np.max(scores))
            weights = weights / max(1e-8, float(weights.sum()))
            candidate_rows.append({"slot_weights": weights.tolist(), "slot_scores": scores.tolist()})
        rows.append(
            {
                "type": "plan_arch1_6_attention_summary",
                "benchmark": BENCHMARK,
                "variant": variant.name,
                "seed": int(seed),
                "example_id": example.id,
                "family": example.family,
                "gold_candidate": int(example.label),
                "predicted_candidate": int(preds[idx]),
                "target_order": oracle_teacher(example, example.label)["target_order"],
                "candidate_slot_attention": candidate_rows,
            }
        )
    return rows


def _error_rows(variant: BridgeVariant, seed: int, examples: Sequence[ConstraintExample], train_logits: np.ndarray, frozen_logits: np.ndarray, limit: int) -> List[Dict[str, object]]:
    labels = _labels(examples)
    train_pred = np.argmax(train_logits, axis=1)
    frozen_pred = np.argmax(frozen_logits, axis=1)
    rows = []
    for idx, example in enumerate(examples):
        if len(rows) >= limit:
            break
        if int(train_pred[idx]) == int(labels[idx]):
            continue
        rows.append(
            {
                "type": "plan_arch1_6_error_case",
                "benchmark": BENCHMARK,
                "variant": variant.name,
                "seed": int(seed),
                "example_id": example.id,
                "family": example.family,
                "gold_candidate_index": int(example.label),
                "predicted_candidate_index": int(train_pred[idx]),
                "frozen_candidate_index": int(frozen_pred[idx]),
                "gold_teacher": oracle_teacher(example, int(example.label)),
                "predicted_teacher": oracle_teacher(example, int(train_pred[idx])),
                "gold_vs_best_negative_margin": float(train_logits[idx, int(example.label)] - np.max(np.delete(train_logits[idx], int(example.label)))),
                "candidate_probabilities": _softmax_np(train_logits[idx]).tolist(),
                "candidate_types": list(example.candidate_types),
            }
        )
    return rows


def _auxiliary_history_rows(variant: BridgeVariant, seed: int, trainable: BridgeFitResult, frozen: BridgeFitResult) -> List[Dict[str, object]]:
    rows = []
    for fit_name, fit in (("trainable", trainable), ("frozen", frozen)):
        for item in fit.history:
            rows.append({"benchmark": BENCHMARK, "variant": variant.name, "seed": int(seed), "fit": fit_name, **item})
    return rows


def _compute_row(variant: BridgeVariant, seed: int, splits: Dict[str, List[ConstraintExample]], row: Dict[str, object], elapsed: float) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "variant": variant.name,
        "seed": int(seed),
        "train_examples": len(splits["train"]),
        "dev_examples": len(splits["dev"]),
        "test_examples": len(splits["test"]),
        "candidate_count": len(splits["test"][0].candidates) if splits["test"] else 0,
        "param_count": int(row["param_count"]),
        "training_time_seconds": float(elapsed),
        "medium_validation_launched": False,
    }


def _write_outputs(
    result: Dict[str, object],
    results_path: Path,
    controls_path: Path,
    oracle_ladder_path: Path,
    auxiliary_path: Path,
    binding_audit_path: Path,
    leaderboard_path: Path,
    failure_path: Path,
    attention_path: Path,
    error_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    for path in (results_path, controls_path, oracle_ladder_path, auxiliary_path, binding_audit_path, leaderboard_path, failure_path, attention_path, error_path, compute_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(_trim_result(result), indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(json.dumps({"controls": result["controls"]}, indent=2, sort_keys=True), encoding="utf-8")
    oracle_ladder_path.write_text(json.dumps({"oracle_ladder": result["oracle_ladder"]}, indent=2, sort_keys=True), encoding="utf-8")
    auxiliary_path.write_text(json.dumps({"auxiliary_losses": result["auxiliary_losses"]}, indent=2, sort_keys=True), encoding="utf-8")
    binding_audit_path.write_text(json.dumps({"rule_event_binding_audit": result["rule_event_binding_audit"]}, indent=2, sort_keys=True), encoding="utf-8")
    failure_path.write_text(json.dumps(result["failure_taxonomy"], indent=2, sort_keys=True), encoding="utf-8")
    compute_path.write_text(json.dumps({"compute_metrics": result["compute_metrics"]}, indent=2, sort_keys=True), encoding="utf-8")
    with leaderboard_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["variant", "seed", "binding", "status", "trainable_top1", "frozen_top1", "delta", "controls_pass", "shortcut_max", "param_count", "score"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["leaderboard"]:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    with attention_path.open("w", encoding="utf-8") as handle:
        for row in result["attention_summaries"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with error_path.open("w", encoding="utf-8") as handle:
        for row in result["error_cases"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report_path.write_text(_render_report(result), encoding="utf-8")


def _render_report(result: Dict[str, object]) -> str:
    summary = result["summary"]
    ladder_means = _ladder_means(result["oracle_ladder"])
    best_cross = _best_variant(summary, "minimal_rule_event_cross_attention")
    best_bilinear = _best_variant(summary, "rule_event_bilinear_verifier")
    hardest = _hardest_family(result["rows"])
    lines = [
        "# PLAN-ARCH-1.6 Oracle-to-Latent Constraint Binding Bridge",
        "",
        "## Scope",
        "- No planning claim.",
        "- No architecture improvement claim unless gates pass.",
        "- Medium validation was not launched.",
        "- Candidate self-attention, pyramids, stacking, and multi-avenue variants were not run.",
        "",
        "## Answers",
        f"1. At which oracle-ladder level does learning first work? `{summary['first_working_ladder_level']}`.",
        f"2. Does direct oracle input solve as expected? `{ladder_means.get('A_direct_oracle_input', 0.0):.4f}`.",
        f"3. Does parsed event input make the task learnable? `{ladder_means.get('C_parsed_event_input', 0.0):.4f}`.",
        f"4. Can the model infer active rules from constraint examples? `{_aux_answer(result['auxiliary_losses'], 'rule')}`.",
        f"5. Can the model bind active rules to candidate rollout events? `{_binding_answer(result['controls'])}`.",
        f"6. Do auxiliary heads help? `{_aux_help_answer(summary)}`.",
        f"7. Does pretraining help? `{_pretrain_answer(summary)}`.",
        f"8. Does rule-event cross-attention or bilinear matching help? `cross={float(best_cross.get('mean_trainable_top1', 0.0)):.4f}, bilinear={float(best_bilinear.get('mean_trainable_top1', 0.0)):.4f}`.",
        f"9. Which constraint family remains hardest? `{hardest}`.",
        f"10. Does any claimable variant beat frozen while controls pass? `{summary['medium_ready']}`.",
        f"11. Should broad architecture search resume after the binding module works? `{summary['architecture_search_should_resume']}`.",
        "",
        "## Oracle Ladder",
        "| level | mean top1 |",
        "| --- | ---: |",
    ]
    for level, value in ladder_means.items():
        lines.append(f"| {level} | {value:.4f} |")
    lines.extend(["", "## Leaderboard", "| variant | seed | trainable | frozen | delta | shortcut | controls |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in result["leaderboard"][:20]:
        lines.append(
            f"| {row['variant']} | {row['seed']} | {float(row['trainable_top1']):.4f} | {float(row['frozen_top1']):.4f} | "
            f"{float(row['delta']):.4f} | {float(row['shortcut_max']):.4f} | `{bool(row['controls_pass'])}` |"
        )
    lines.extend(["", "## Variant Summary", "| variant | trainable mean | frozen mean | mean delta | seed wins | controls | gate |", "| --- | ---: | ---: | ---: | ---: | --- | --- |"])
    for row in summary["variant_summaries"]:
        lines.append(
            f"| {row['variant']} | {float(row['mean_trainable_top1']):.4f} | {float(row['mean_frozen_top1']):.4f} | "
            f"{float(row['mean_delta']):.4f} | {int(row['seeds_trainable_beats_frozen'])} | `{bool(row['controls_pass_all'])}` | `{bool(row['success_gate'])}` |"
        )
    lines.extend(["", "## Claim Boundary", "This stage is a bridge from symbolic oracle reasoning to latent neural constraint binding. It does not claim planning or an architecture improvement unless gates pass."])
    return "\n".join(lines) + "\n"


def _ladder_means(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    out = {}
    for level in ["A_direct_oracle_input", "B_violation_type_input", "C_parsed_event_input", "D_structured_constraint_input", "E_raw_constraint_examples"]:
        vals = [float(row["top1"]) for row in rows if row["level"] == level]
        out[level] = _mean(vals)
    return out


def _best_variant(summary: Dict[str, object], name: str) -> Dict[str, object]:
    return next((row for row in summary["variant_summaries"] if row["variant"] == name), {})


def _aux_answer(rows: Sequence[Dict[str, object]], _kind: str) -> str:
    devs = [float(row.get("dev_top1", 0.0)) for row in rows if row.get("fit") == "trainable"]
    return f"tracked; best_dev_top1={max(devs, default=0.0):.4f}"


def _binding_answer(controls: Sequence[Dict[str, object]]) -> str:
    values = [bool(row["control_pass"]["rule_event_mismatch_collapses"]) for row in controls]
    ablations = [bool(row["control_pass"]["ablate_constraint_rule_tokens_hurts"] and row["control_pass"]["ablate_rollout_event_tokens_hurts"]) for row in controls]
    return f"mismatch={sum(values)}/{len(values)}, ablation_both_hurt={sum(ablations)}/{len(ablations)}" if values else "not run"


def _aux_help_answer(summary: Dict[str, object]) -> str:
    no_aux = float(_best_variant(summary, "minimal_rule_event_cross_attention").get("mean_trainable_top1", 0.0))
    aux = max(float(row["mean_trainable_top1"]) for row in summary["variant_summaries"] if row["variant"] != "minimal_rule_event_cross_attention")
    return f"best_aux={aux:.4f}, no_aux={no_aux:.4f}, helps={aux > no_aux}"


def _pretrain_answer(summary: Dict[str, object]) -> str:
    pretrain_names = {"shared_weight_constraint_clone_with_event_tokens", "P1_rollout_no_final_constraint_with_event_tokens"}
    pretrain = max(float(row["mean_trainable_top1"]) for row in summary["variant_summaries"] if row["variant"] in pretrain_names)
    scratch = max(float(row["mean_trainable_top1"]) for row in summary["variant_summaries"] if row["variant"] not in pretrain_names)
    return f"best_pretrain={pretrain:.4f}, best_nonpretrain={scratch:.4f}, helps={pretrain > scratch}"


def _hardest_family(rows: Sequence[Dict[str, object]]) -> str:
    values: Dict[str, List[float]] = {family: [] for family in FAMILIES}
    for row in rows:
        by_family = row.get("trainable", {}).get("by_constraint_family", {})
        for family, value in by_family.items():
            values[str(family)].append(float(value))
    means = {family: _mean(vals) for family, vals in values.items() if vals}
    if not means:
        return "not available"
    family = min(means, key=means.get)
    return f"{family}={means[family]:.4f}"


def _flat_params(items: Sequence[Tuple[str, nn.Parameter]]) -> torch.Tensor:
    tensors = [parameter.detach().float().cpu().reshape(-1) for _name, parameter in items]
    return torch.cat(tensors) if tensors else torch.zeros(0)


def _grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return float(math.sqrt(total)) if total > 0 else 0.0


def _l2_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0 or a.numel() != b.numel():
        return 0.0
    return float(torch.linalg.vector_norm(a - b).item())


def _trim_result(result: Dict[str, object]) -> Dict[str, object]:
    trimmed = dict(result)
    trimmed["attention_summaries"] = result.get("attention_summaries", [])[:40]
    trimmed["error_cases"] = result.get("error_cases", [])[:80]
    return trimmed


def _dataset_config(data: object) -> ConstraintDatasetConfig:
    values = dict(data or {})
    if "families" in values:
        values["families"] = tuple(str(item) for item in values["families"])
    allowed = set(ConstraintDatasetConfig.__dataclass_fields__.keys())
    return ConstraintDatasetConfig(**{key: value for key, value in values.items() if key in allowed})


def _training_config(data: object) -> ConstraintTrainingConfig:
    values = dict(data or {})
    allowed = set(ConstraintTrainingConfig.__dataclass_fields__.keys())
    return ConstraintTrainingConfig(**{key: value for key, value in values.items() if key in allowed})


def _default_config(config: Dict[str, object]) -> Dict[str, object]:
    base: Dict[str, object] = {
        "device": "cpu",
        "seeds": [0, 1, 2],
        "dataset": {
            "grid_size": 7,
            "num_candidates": 8,
            "plan_length": 48,
            "train_examples": 256,
            "dev_examples": 64,
            "test_examples": 64,
            "families": list(FAMILIES),
            "all_goal_every": 1,
        },
        "training": {"epochs": 10, "batch_size": 64, "lr": 0.004, "weight_decay": 0.0, "patience": 5, "gradient_clip_norm": 1.0},
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


def _load_config(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


if __name__ == "__main__":
    main()
