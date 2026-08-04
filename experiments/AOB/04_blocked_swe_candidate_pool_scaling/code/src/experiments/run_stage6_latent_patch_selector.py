from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from src.coordinators.latent_coordination import LatentCoordinatorConfig, make_candidate_query_coordinator, make_latent_coordinator
from src.coordinators.mlp import MLPTrainingConfig
from src.datasets.swe_patch_selection_dataset import (
    STAGE6_CONTROL_NAMES,
    STAGE6_NUM_CANDIDATES,
    PatchSelectionExample,
    apply_stage6_control,
    duplicate_candidate_patch_hash_audit,
    edited_files_from_diff,
    label_matrix,
    load_patch_selection_jsonl,
    patch_selection_to_multiview_many,
    randomized_label_matrix,
    stage6_dataset_summary,
    stage6_output_leakage_audit,
    stage6_split_leakage_audit,
    validate_patch_selection_examples,
)
from src.experiments.real_shared_weight_latent_coordination import (
    CandidatewiseScoringModel,
    FitResult,
    MessageChannelConfig,
    RealSharedWeightTrainingConfig,
    SharedClonedAgentSystem,
    SharedTransformerAgent,
    SharedTransformerAgentConfig,
    _autocast_context,
    _auxiliary_loss_weight,
    _batches,
    _cuda_max_memory,
    _example_batches,
    _flat_module_parameters,
    _flat_named_parameters,
    _grad_norm,
    _load_system_checkpoint,
    _message_regularization_loss,
    _parameter_delta,
    _private_cue_targets,
    _replace_dataclass,
    _system_checkpoint,
    _text_tokens,
    shared_parameter_identity_check,
)
from src.experiments.stage6_candidate_pool_builder import build_synthetic_stage6_candidate_pool, load_or_build_candidate_pool


BENCHMARK = "stage6_latent_patch_selector"
DEFAULT_CANDIDATE_POOL_PATH = Path("results/stage6_candidate_pools.jsonl")
DEFAULT_RESULTS_PATH = Path("results/stage6_latent_patch_selector_results.json")
DEFAULT_AUDIT_PATH = Path("results/stage6_latent_patch_selector_audit.jsonl")
DEFAULT_SPLITS_PATH = Path("results/stage6_splits.json")
DEFAULT_REPORT_PATH = Path("reports/STAGE6_LATENT_PATCH_SELECTOR.md")
DEFAULT_CHECKPOINT_DIR = Path("results/stage6_latent_patch_selector_checkpoints")


@dataclass(frozen=True)
class Stage6Architecture:
    name: str
    description: str
    num_avenues: int
    avenue_prompt_mode: str = "stage6_patch_selection"
    avenue_topk: int = 0
    coordinator_family: str = "candidate_token_cross_attention"
    lr: float = 0.0003
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class Stage6RunConfig:
    phase: str = "6A"
    device: str = "cpu"
    synthetic_if_missing: bool = False
    synthetic_tasks: int = 48
    seed_for_pool: int = 0
    hidden_dim: int = 32
    tiny_layers: int = 1
    tiny_heads: int = 1
    tiny_ff_dim: int = 64
    max_length: int = 160
    epochs: int = 3
    patience: int = 2
    batch_size: int = 8
    mixed_precision: str = "none"
    run_controls: bool = True
    save_checkpoints: bool = True
    allow_gold_diagnostic: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 6 latent patch selector on fixed SWE-style candidate pools.")
    parser.add_argument("--phase", choices=("6A", "6B", "6B-real", "6C", "smoke"), default="6A")
    parser.add_argument("--candidate-pool", default=str(DEFAULT_CANDIDATE_POOL_PATH))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--synthetic-if-missing", action="store_true")
    parser.add_argument("--synthetic-tasks", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--no-controls", action="store_true")
    args = parser.parse_args()

    config = _phase_config(args.phase)
    if args.epochs is not None:
        config = replace(config, epochs=int(args.epochs))
    if args.batch_size is not None:
        config = replace(config, batch_size=int(args.batch_size))
    if args.hidden_dim is not None:
        config = replace(config, hidden_dim=int(args.hidden_dim))
    config = replace(
        config,
        phase=args.phase,
        device=str(args.device),
        synthetic_if_missing=bool(args.synthetic_if_missing),
        synthetic_tasks=int(args.synthetic_tasks),
        run_controls=not bool(args.no_controls),
    )
    result = run_stage6(
        candidate_pool_path=Path(args.candidate_pool),
        results_path=Path(args.results),
        audit_path=Path(args.audit),
        splits_path=Path(args.splits),
        report_path=Path(args.report),
        checkpoint_dir=Path(args.checkpoint_dir),
        config=config,
    )
    print(
        "stage6: wrote {results}, {audit}, {report}; completed_rows={rows}".format(
            results=args.results,
            audit=args.audit,
            report=args.report,
            rows=len(result.get("rows", [])),
        )
    )


def run_stage6(
    candidate_pool_path: Path = DEFAULT_CANDIDATE_POOL_PATH,
    results_path: Path = DEFAULT_RESULTS_PATH,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    splits_path: Path = DEFAULT_SPLITS_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    config: Stage6RunConfig | None = None,
) -> Dict[str, object]:
    config = config or Stage6RunConfig()
    examples = load_or_build_candidate_pool(
        candidate_pool_path=Path(candidate_pool_path),
        synthetic_if_missing=config.synthetic_if_missing,
        n_tasks=config.synthetic_tasks,
        seed=config.seed_for_pool,
    )
    if not examples:
        examples = build_synthetic_stage6_candidate_pool(n_tasks=config.synthetic_tasks, seed=config.seed_for_pool)
    validation = validate_patch_selection_examples(examples, allow_gold_diagnostic=bool(config.allow_gold_diagnostic))
    splits = _split_examples(examples)
    _write_splits(splits_path, splits)
    seeds = _phase_seeds(config.phase)
    architectures = _phase_architectures(config.phase)
    rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    selected_architecture = architectures[-1].name
    for seed in seeds:
        split_leakage = stage6_split_leakage_audit(BENCHMARK, seed, splits)
        output_leakage = stage6_output_leakage_audit(BENCHMARK, seed, [example for rows_in in splits.values() for example in rows_in])
        duplicate_audit = duplicate_candidate_patch_hash_audit([example for rows_in in splits.values() for example in rows_in])
        audit_rows.extend([split_leakage, output_leakage, {"benchmark": BENCHMARK, "seed": seed, "type": "duplicate_candidate_patch_hash", **duplicate_audit}])
        for architecture in architectures:
            row = _run_seed(
                config=config,
                architecture=architecture,
                splits=splits,
                seed=seed,
                checkpoint_dir=checkpoint_dir,
                split_leakage=split_leakage,
                output_leakage=output_leakage,
                duplicate_audit=duplicate_audit,
            )
            rows.append(row)
            audit_rows.extend(_audit_rows_from_seed_row(row))
    architecture_selection = _architecture_selection(rows) if config.phase == "6B" else {}
    if architecture_selection:
        selected_architecture = str(architecture_selection.get("selected_architecture", selected_architecture))
    result = {
        "metadata": {
            "benchmark": BENCHMARK,
            "created_at_utc": _now(),
            "candidate_pool_path": str(candidate_pool_path),
            "selector_scope": "selector only; no patch generation claim",
            "primary_claim_template": (
                "Given the same 8 candidate patches for a real issue-resolution task, a trained shared-weight "
                "latent cloned-agent selector chooses passing patches more reliably than specified baselines."
            ),
            "phase": config.phase,
            "selected_architecture": selected_architecture,
            "architecture_selection": architecture_selection,
            "config": asdict(config),
            "dataset_validation": validation,
            "dataset_summary": stage6_dataset_summary(splits),
        },
        "rows": rows,
        "summary": _overall_summary(rows, config.phase),
    }
    _write_outputs(result, audit_rows, results_path, audit_path, report_path)
    return result


def fit_stage6_latent_selector(
    train_examples: Sequence[PatchSelectionExample],
    dev_examples: Sequence[PatchSelectionExample],
    agent_config: SharedTransformerAgentConfig,
    coordinator_config: LatentCoordinatorConfig,
    training_config: RealSharedWeightTrainingConfig,
    seed: int,
    device: str,
    trainable_agent: bool,
    method: str,
    message_config: MessageChannelConfig,
    train_label_masks: np.ndarray | None = None,
    dev_label_masks: np.ndarray | None = None,
) -> FitResult:
    train_examples = _canonical_candidate_order(train_examples)
    dev_examples = _canonical_candidate_order(dev_examples)
    train_mv = patch_selection_to_multiview_many(train_examples)
    dev_mv = patch_selection_to_multiview_many(dev_examples)
    train_masks = label_matrix(train_examples) if train_label_masks is None else np.asarray(train_label_masks, dtype=np.float32)
    dev_masks = label_matrix(dev_examples) if dev_label_masks is None else np.asarray(dev_label_masks, dtype=np.float32)
    if not np.any(train_masks.sum(axis=1) > 0):
        raise ValueError("Stage 6 latent training requires at least one oracle-positive train example")

    torch.manual_seed(seed + 619_007)
    np_rng = np.random.default_rng(seed + 629_001)
    agent = SharedTransformerAgent(agent_config).to(device)
    agent.configure_trainable(trainable_agent)
    coordinator_input_dim = int(message_config.message_dim) if message_config.use_message_head else int(agent.output_dim)
    effective_coordinator_config = _replace_dataclass(coordinator_config, input_dim=coordinator_input_dim)
    candidate_query_families = {
        "candidate_query_cross_attention",
        "candidate_token_cross_attention",
        "bilinear_candidate",
        "contrastive_candidate",
        "global_then_candidate",
        "two_round_message_passing",
    }
    if message_config.coordinator_family in candidate_query_families:
        family = "cross_attention" if message_config.coordinator_family == "candidate_query_cross_attention" else message_config.coordinator_family
        coordinator = make_candidate_query_coordinator(
            n_roles=len(train_mv[0].views),
            candidate_feature_dim=4,
            config=_replace_dataclass(effective_coordinator_config, family=family),
        ).to(device)
    elif message_config.coordinator_family == "latent":
        coordinator = make_latent_coordinator(
            n_roles=len(train_mv[0].views),
            num_classes=STAGE6_NUM_CANDIDATES,
            config=effective_coordinator_config,
        ).to(device)
    else:
        raise ValueError(f"unknown Stage 6 coordinator family: {message_config.coordinator_family}")
    system = SharedClonedAgentSystem(
        agent,
        coordinator,
        n_roles=len(train_mv[0].views),
        visible_explicit_evidence=False,
        message_config=message_config,
    ).to(device)
    params = [parameter for parameter in system.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("no trainable parameters available for Stage 6 latent selector")
    optimizer = torch.optim.AdamW(params, lr=training_config.lr, weight_decay=training_config.weight_decay)
    initial_agent = _flat_named_parameters(agent.coordination_parameter_items())
    initial_coordinator = _flat_module_parameters(coordinator)
    initial_active_message_readout = _flat_module_parameters(system.active_message_readout) if system.active_message_readout is not None else torch.zeros(0)
    initial_message_head = _flat_module_parameters(system.message_head) if system.message_head is not None else torch.zeros(0)
    initial_private_cue_head = _flat_module_parameters(system.private_cue_head) if system.private_cue_head is not None else torch.zeros(0)

    history: List[Dict[str, float]] = []
    agent_grad_norms: List[float] = []
    coordinator_grad_norms: List[float] = []
    active_message_readout_grad_norms: List[float] = []
    message_head_grad_norms: List[float] = []
    private_cue_head_grad_norms: List[float] = []
    best_state = None
    best_dev = -1.0
    stale = 0
    first_backward: Dict[str, object] | None = None
    step = 0
    accumulation = max(1, int(training_config.gradient_accumulation_steps))

    for epoch in range(training_config.epochs):
        system.train()
        order = np_rng.permutation(len(train_mv))
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for batch_ids in _batches(order, training_config.batch_size):
            batch_examples = [train_mv[int(index)] for index in batch_ids]
            batch_masks_np = train_masks[np.asarray(batch_ids, dtype=np.int64)]
            positive_rows = batch_masks_np.sum(axis=1) > 0
            if not np.any(positive_rows):
                continue
            labels = torch.as_tensor(batch_masks_np, dtype=torch.float32, device=device)
            valid = torch.as_tensor(positive_rows, dtype=torch.bool, device=device)
            with _autocast_context(device, training_config.mixed_precision):
                logits, forward_audit, clone_activations = system(
                    batch_examples,
                    condition="none",
                    seed=seed + epoch,
                    retain_activation_grad=first_backward is None,
                )
                main_loss = multi_positive_selector_loss(logits.index_select(0, valid.nonzero(as_tuple=False).flatten()), labels.index_select(0, valid.nonzero(as_tuple=False).flatten()))
                aux_weight = _auxiliary_loss_weight(epoch, message_config)
                aux_loss = torch.zeros((), dtype=main_loss.dtype, device=main_loss.device)
                if message_config.use_private_cue_aux:
                    if system.last_aux_logits is None:
                        raise RuntimeError("private-cue auxiliary loss requested but no auxiliary logits were produced")
                    cue_targets = _private_cue_targets(batch_examples, logits.device)
                    aux_loss = F.cross_entropy(system.last_aux_logits.reshape(-1, 2), cue_targets.reshape(-1))
                total_loss = main_loss + aux_weight * aux_loss + _message_regularization_loss(clone_activations, message_config)
                loss = total_loss / accumulation
            epoch_loss_sum += float(total_loss.detach().float().cpu().item()) * int(valid.sum().detach().cpu().item())
            epoch_loss_count += int(valid.sum().detach().cpu().item())
            loss.backward()
            agent_grad_norm = _grad_norm(parameter for _name, parameter in agent.coordination_parameter_items())
            coordinator_grad_norm = _grad_norm(coordinator.parameters())
            active_message_readout_grad_norm = _grad_norm(system.active_message_readout.parameters()) if system.active_message_readout is not None else 0.0
            message_head_grad_norm = _grad_norm(system.message_head.parameters()) if system.message_head is not None else 0.0
            private_cue_head_grad_norm = _grad_norm(system.private_cue_head.parameters()) if system.private_cue_head is not None else 0.0
            agent_grad_norms.append(agent_grad_norm)
            coordinator_grad_norms.append(coordinator_grad_norm)
            active_message_readout_grad_norms.append(active_message_readout_grad_norm)
            message_head_grad_norms.append(message_head_grad_norm)
            private_cue_head_grad_norms.append(private_cue_head_grad_norm)
            if first_backward is None:
                per_clone = []
                if bool(getattr(clone_activations, "retains_grad", False)) and clone_activations.grad is not None:
                    per_clone = [
                        float(clone_activations.grad[:, clone_id, :].detach().float().norm().cpu().item())
                        for clone_id in range(clone_activations.shape[1])
                    ]
                first_backward = {
                    **forward_audit,
                    "loss_backward_reaches_shared_agent": bool(agent_grad_norm > 0.0),
                    "loss_backward_reaches_active_message_readout": bool(active_message_readout_grad_norm > 0.0),
                    "loss_backward_reaches_message_head": bool(message_head_grad_norm > 0.0),
                    "loss_backward_reaches_private_cue_head": bool(private_cue_head_grad_norm > 0.0),
                    "per_clone_activation_grad_norms": per_clone,
                    "per_clone_gradient_contribution": bool(per_clone and all(value > 0.0 for value in per_clone)),
                }
            step += 1
            if step % accumulation == 0:
                if float(training_config.gradient_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(params, float(training_config.gradient_clip_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if step % accumulation != 0:
            if float(training_config.gradient_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(training_config.gradient_clip_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        train_scores = predict_stage6_latent_logits_from_system(system, train_mv, condition="none", seed=seed)
        dev_scores = predict_stage6_latent_logits_from_system(system, dev_mv, condition="none", seed=seed)
        train_metrics = evaluate_scores("trainable", train_scores, train_examples, split="train", unavailable=False)
        dev_metrics = evaluate_scores("trainable", dev_scores, dev_examples, split="dev", unavailable=False)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(epoch_loss_sum / max(1, epoch_loss_count)),
                "train_pass_at_1": float(train_metrics["pass_at_1"]),
                "dev_pass_at_1": float(dev_metrics["pass_at_1"]),
                "dev_conditional_accuracy": float(dev_metrics["conditional_selector_accuracy"]),
                "aux_weight": _auxiliary_loss_weight(epoch, message_config),
            }
        )
        dev_selection = float(dev_metrics["conditional_selector_accuracy"])
        if dev_selection > best_dev:
            best_dev = dev_selection
            best_state = _system_checkpoint(system)
            stale = 0
        else:
            stale += 1
            if stale >= training_config.patience:
                break
    if best_state is not None:
        _load_system_checkpoint(system, best_state, device)
    agent_delta = _parameter_delta(agent.coordination_parameter_items(), initial_agent)
    coordinator_delta = _parameter_delta(list(coordinator.named_parameters()), initial_coordinator)
    active_message_readout_delta = (
        _parameter_delta(list(system.active_message_readout.named_parameters()), initial_active_message_readout)
        if system.active_message_readout is not None
        else 0.0
    )
    message_head_delta = _parameter_delta(list(system.message_head.named_parameters()), initial_message_head) if system.message_head is not None else 0.0
    private_cue_head_delta = (
        _parameter_delta(list(system.private_cue_head.named_parameters()), initial_private_cue_head)
        if system.private_cue_head is not None
        else 0.0
    )
    first_backward = first_backward or {}
    param_count = int(sum(parameter.numel() for parameter in params))
    audit = {
        "benchmark": BENCHMARK,
        "seed": int(seed),
        "split": "dev",
        "probe": "stage6_multi_positive_training_audit",
        "method": method,
        "agent_trainable": bool(trainable_agent),
        "multi_positive_loss": "negative_log_sum_softmax_probability_assigned_to_passing_candidates",
        "message_config": asdict(message_config),
        "coordinator_config": asdict(coordinator_config),
        "shared_parameter_identity": shared_parameter_identity_check(agent, len(train_mv[0].views))["pass"],
        "shared_parameter_count": shared_parameter_identity_check(agent, len(train_mv[0].views))["parameter_count"],
        "agent_grad_norm_mean": _mean(agent_grad_norms),
        "agent_grad_norm_std": _std(agent_grad_norms),
        "coordinator_grad_norm_mean": _mean(coordinator_grad_norms),
        "coordinator_grad_norm_std": _std(coordinator_grad_norms),
        "active_message_readout_grad_norm_mean": _mean(active_message_readout_grad_norms),
        "active_message_readout_parameter_delta": active_message_readout_delta,
        "message_head_grad_norm_mean": _mean(message_head_grad_norms),
        "message_head_parameter_delta": message_head_delta,
        "private_cue_head_grad_norm_mean": _mean(private_cue_head_grad_norms),
        "private_cue_head_parameter_delta": private_cue_head_delta,
        "agent_parameter_delta": agent_delta,
        "coordinator_parameter_delta": coordinator_delta,
        "activation_requires_grad_before_coordinator": bool(first_backward.get("activation_requires_grad_before_coordinator", False)),
        "clone_activation_requires_grad": bool(first_backward.get("activation_requires_grad_before_coordinator", False)),
        "no_detach_between_clone_activations_and_loss": bool(first_backward.get("no_detach_between_clone_activations_and_loss", False)),
        "loss_backward_reaches_shared_agent": bool(first_backward.get("loss_backward_reaches_shared_agent", False)),
        "per_clone_gradient_contribution": bool(first_backward.get("per_clone_gradient_contribution", False)),
        "per_clone_activation_grad_norms": first_backward.get("per_clone_activation_grad_norms", []),
        "num_avenues": int(message_config.num_avenues),
        "avenue_prompt_mode": str(message_config.avenue_prompt_mode),
        "epochs_run": len(history),
        "best_dev_conditional_selector_accuracy": best_dev,
        "batch_size": int(training_config.batch_size),
        "gradient_accumulation_steps": int(training_config.gradient_accumulation_steps),
        "gradient_clip_norm": float(training_config.gradient_clip_norm),
        "mixed_precision": str(training_config.mixed_precision),
        "device": str(device),
        "cuda_max_memory_allocated": _cuda_max_memory(device),
        "param_count": param_count,
    }
    return FitResult(method, "none", agent, coordinator, system, trainable_agent, param_count, audit, history)


def multi_positive_selector_loss(logits: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != label_mask.shape:
        raise ValueError(f"logits and label mask shape mismatch: {tuple(logits.shape)} vs {tuple(label_mask.shape)}")
    positives = label_mask > 0
    if not bool(positives.any(dim=1).all()):
        raise ValueError("multi-positive selector loss received an oracle-empty row")
    log_probs = F.log_softmax(logits, dim=1)
    masked = log_probs.masked_fill(~positives, torch.finfo(log_probs.dtype).min)
    return -torch.logsumexp(masked, dim=1).mean()


def predict_stage6_latent_logits(result: FitResult, examples: Sequence[PatchSelectionExample], condition: str, seed: int) -> np.ndarray:
    canonical = _canonical_candidate_order(examples)
    mv = patch_selection_to_multiview_many(canonical)
    canonical_scores = predict_stage6_latent_logits_from_system(result.system, mv, condition=condition, seed=seed)
    return _restore_candidate_order_scores(canonical_scores, canonical, examples)


def predict_stage6_latent_logits_from_system(system: SharedClonedAgentSystem, examples, condition: str, seed: int) -> np.ndarray:
    system.eval()
    rows = []
    with torch.no_grad():
        for batch in _example_batches(examples, 128):
            logits, _audit, _acts = system(batch, condition=condition, seed=seed)
            rows.append(logits.detach().float().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32) if rows else np.zeros((0, STAGE6_NUM_CANDIDATES), dtype=np.float32)


def evaluate_scores(
    method: str,
    scores: np.ndarray,
    examples: Sequence[PatchSelectionExample],
    split: str,
    unavailable: bool = False,
    metadata: Dict[str, object] | None = None,
) -> Dict[str, object]:
    labels = label_matrix(examples)
    if unavailable:
        scores = np.zeros((len(examples), STAGE6_NUM_CANDIDATES), dtype=np.float32)
    if scores.shape != labels.shape:
        raise ValueError(f"score shape mismatch for {method}: {scores.shape} vs {labels.shape}")
    ranking = _rank_candidate_scores(scores, examples)
    pred = ranking[:, 0] if len(ranking) else np.zeros(0, dtype=np.int64)
    positive = labels.sum(axis=1) > 0
    correct = np.asarray([bool(labels[row, int(choice)] > 0) for row, choice in enumerate(pred)], dtype=bool)
    top2 = np.asarray([bool(labels[row, ranking[row, :2]].sum() > 0) for row in range(len(examples))], dtype=bool)
    reciprocal = []
    for row in range(len(examples)):
        rr = 0.0
        for rank_index, candidate_index in enumerate(ranking[row], start=1):
            if labels[row, int(candidate_index)] > 0:
                rr = 1.0 / float(rank_index)
                break
        reciprocal.append(rr)
    oracle = float(np.mean(positive)) if len(examples) else 0.0
    pass_at_1 = float(np.mean(correct)) if len(examples) else 0.0
    conditional = float(np.mean(correct[positive])) if np.any(positive) else 0.0
    return {
        "benchmark": BENCHMARK,
        "method": method,
        "split": split,
        "unavailable": bool(unavailable),
        "pass_at_1": pass_at_1,
        "conditional_selector_accuracy": conditional,
        "oracle_pass_at_8": oracle,
        "selection_efficiency": pass_at_1 / oracle if oracle > 0.0 else 0.0,
        "mrr": float(np.mean(reciprocal)) if reciprocal else 0.0,
        "top_2_accuracy": float(np.mean(top2)) if len(top2) else 0.0,
        "n_examples": len(examples),
        "n_oracle_positive_examples": int(np.sum(positive)),
        "predictions": [int(value) for value in pred.tolist()],
        "per_repository_accuracy": _group_accuracy(examples, correct, lambda example: example.repo),
        "per_generator_accuracy": _selected_generator_accuracy(examples, pred, correct),
        "per_task_family_accuracy": _group_accuracy(examples, correct, lambda example: str(example.metadata.get("task_family", "unknown"))),
        "cost_tokens_latency": metadata or {},
    }


class Stage6CandidatewiseReranker:
    def __init__(
        self,
        method: str,
        feature_builder: Callable[[Sequence[PatchSelectionExample]], np.ndarray],
        training: MLPTrainingConfig,
        seed: int,
        device: str,
        hidden_dims: Iterable[int] = (64,),
    ) -> None:
        self.method = method
        self.feature_builder = feature_builder
        self.training = training
        self.seed = int(seed)
        self.device = torch.device(device)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.model: CandidatewiseScoringModel | None = None
        self.param_count = 0
        self.history: List[Dict[str, float]] = []

    def fit(self, train_examples: Sequence[PatchSelectionExample], dev_examples: Sequence[PatchSelectionExample]) -> None:
        torch.manual_seed(self.seed + 71_006)
        x_train = self.feature_builder(train_examples)
        y_train = label_matrix(train_examples)
        x_dev = self.feature_builder(dev_examples)
        self.model = CandidatewiseScoringModel(input_dim=x_train.shape[-1], hidden_dims=self.hidden_dims).to(self.device)
        self.param_count = int(sum(parameter.numel() for parameter in self.model.parameters()))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training.lr, weight_decay=self.training.weight_decay)
        tx = torch.as_tensor(x_train, dtype=torch.float32, device=self.device)
        ty = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)
        best_state = None
        best_dev = -1.0
        stale = 0
        rng = np.random.default_rng(self.seed + 71_007)
        for epoch in range(self.training.epochs):
            self.model.train()
            for batch_idx in _batches(rng.permutation(len(train_examples)), self.training.batch_size):
                mask = y_train[np.asarray(batch_idx, dtype=np.int64)].sum(axis=1) > 0
                if not np.any(mask):
                    continue
                idx = torch.as_tensor(np.asarray(batch_idx, dtype=np.int64)[mask], dtype=torch.long, device=self.device)
                loss = multi_positive_selector_loss(self.model(tx.index_select(0, idx)), ty.index_select(0, idx))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            dev_metrics = evaluate_scores(self.method, self.predict_scores(dev_examples), dev_examples, split="dev")
            self.history.append({"epoch": float(epoch + 1), "dev_conditional_selector_accuracy": float(dev_metrics["conditional_selector_accuracy"])})
            if float(dev_metrics["conditional_selector_accuracy"]) > best_dev:
                best_dev = float(dev_metrics["conditional_selector_accuracy"])
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.training.patience:
                    break
        if best_state is not None and self.model is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def predict_scores(self, examples: Sequence[PatchSelectionExample]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.method} is not fit")
        features = torch.as_tensor(self.feature_builder(examples), dtype=torch.float32, device=self.device)
        rows = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, features.shape[0], 2048):
                rows.append(self.model(features[start : start + 2048]).detach().float().cpu().numpy())
        return np.concatenate(rows, axis=0).astype(np.float32) if rows else np.zeros((0, STAGE6_NUM_CANDIDATES), dtype=np.float32)


class BestGeneratorOnDevBaseline:
    def __init__(self) -> None:
        self.method = "best_generator_on_dev_baseline"
        self.generator_scores: Dict[str, float] = {}

    def fit(self, dev_examples: Sequence[PatchSelectionExample]) -> None:
        wins: Dict[str, List[int]] = {}
        for example in dev_examples:
            for candidate, label in zip(example.candidates, example.labels_pass_fail):
                wins.setdefault(candidate.candidate_source_agent, []).append(int(label))
        self.generator_scores = {name: float(np.mean(values)) if values else 0.0 for name, values in wins.items()}

    def predict_scores(self, examples: Sequence[PatchSelectionExample]) -> np.ndarray:
        rows = []
        for example in examples:
            rows.append([self.generator_scores.get(candidate.candidate_source_agent, 0.0) for candidate in example.candidates])
        return np.asarray(rows, dtype=np.float32)


def _run_seed(
    config: Stage6RunConfig,
    architecture: Stage6Architecture,
    splits: Dict[str, Sequence[PatchSelectionExample]],
    seed: int,
    checkpoint_dir: Path,
    split_leakage: Dict[str, object],
    output_leakage: Dict[str, object],
    duplicate_audit: Dict[str, object],
) -> Dict[str, object]:
    start = time.perf_counter()
    device = _resolve_device(config.device)
    agent_config = _agent_config(config)
    training = _training_config(config, architecture)
    message_config = _message_config(architecture)
    coordinator_config = _coordinator_config(config, architecture)
    models = _fit_models(splits, seed, device, agent_config, training, message_config, coordinator_config, architecture)
    metrics = {
        split: _evaluate_methods(models, list(examples), split, seed, config)
        for split, examples in splits.items()
        if examples
    }
    controls = _evaluate_controls(models["trainable"], list(splits.get("test") or splits.get("dev") or []), seed, config) if config.run_controls else {}
    invariance = _invariance_audit(models["trainable"], list(splits.get("test") or splits.get("dev") or []), seed)
    paired_tests = _paired_tests_for_split(
        metrics.get("test") or metrics.get("dev", {}),
        list(splits.get("test") or splits.get("dev") or []),
        seed,
    )
    checkpoint_paths = _save_checkpoints(checkpoint_dir, seed, architecture, models, config) if config.save_checkpoints else {}
    row = {
        "stage": BENCHMARK,
        "phase": config.phase,
        "seed": int(seed),
        "architecture": architecture.name,
        "architecture_config": asdict(architecture),
        "status": "completed",
        "completed_at_utc": _now(),
        "device": device,
        "metrics": metrics,
        "controls": controls,
        "invariance_audit": invariance,
        "paired_tests_vs_best_baseline": paired_tests,
        "checkpoint_paths": checkpoint_paths,
        "elapsed_seconds": float(time.perf_counter() - start),
        "split_leakage_audit": split_leakage,
        "split_leakage_audit_passes": bool(split_leakage.get("passes", False)),
        "output_leakage_audit": output_leakage,
        "output_leakage_audit_passes": bool(output_leakage.get("passes", False)),
        "duplicate_candidate_patch_hash_audit": duplicate_audit,
        "trainable_audit": _audit_subset(models["trainable"].audit),
        "frozen_audit": _audit_subset(models["frozen"].audit),
        "raw_latent_audit": _audit_subset(models["raw_latent"].audit),
        "randomized_labels_audit": _audit_subset(models["randomized_labels"].audit),
    }
    row["success_gates"] = _seed_success_gates(row)
    return row


def _fit_models(
    splits: Dict[str, Sequence[PatchSelectionExample]],
    seed: int,
    device: str,
    agent_config: SharedTransformerAgentConfig,
    training: RealSharedWeightTrainingConfig,
    message_config: MessageChannelConfig,
    coordinator_config: LatentCoordinatorConfig,
    architecture: Stage6Architecture,
) -> Dict[str, object]:
    train = list(splits["train"])
    dev = list(splits["dev"])
    raw_message = MessageChannelConfig(coordinator_family="latent")
    raw_coord = LatentCoordinatorConfig(
        family="cross_attention",
        input_dim=agent_config.hidden_dim,
        model_dim=agent_config.hidden_dim,
        num_heads=1,
        num_layers=1,
        ff_dim=max(agent_config.hidden_dim * 2, 16),
        dropout=0.0,
    )
    raw = fit_stage6_latent_selector(
        train,
        dev,
        agent_config=agent_config,
        coordinator_config=raw_coord,
        training_config=training,
        seed=seed + 6_100,
        device=device,
        trainable_agent=True,
        method="raw_latent_selector",
        message_config=raw_message,
    )
    frozen = fit_stage6_latent_selector(
        train,
        dev,
        agent_config=agent_config,
        coordinator_config=coordinator_config,
        training_config=training,
        seed=seed + 6_200,
        device=device,
        trainable_agent=False,
        method="frozen_same_architecture_latent_selector",
        message_config=message_config,
    )
    trainable = fit_stage6_latent_selector(
        train,
        dev,
        agent_config=agent_config,
        coordinator_config=coordinator_config,
        training_config=training,
        seed=seed + 6_300,
        device=device,
        trainable_agent=True,
        method="trainable_shared_weight_latent_selector",
        message_config=message_config,
    )
    randomized = fit_stage6_latent_selector(
        train,
        dev,
        agent_config=agent_config,
        coordinator_config=coordinator_config,
        training_config=training,
        seed=seed + 6_400,
        device=device,
        trainable_agent=True,
        method="randomized_labels_trainable_latent_selector",
        message_config=message_config,
        train_label_masks=randomized_label_matrix(train, seed + 6_401),
        dev_label_masks=randomized_label_matrix(dev, seed + 6_402),
    )
    baseline_training = MLPTrainingConfig(
        epochs=max(2, training.epochs),
        batch_size=training.batch_size,
        lr=0.002,
        weight_decay=0.0001,
        patience=max(1, training.patience),
        hidden_dims=(64,),
    )
    static = Stage6CandidatewiseReranker(
        "static_patch_feature_reranker",
        _static_patch_features,
        baseline_training,
        seed + 6_500,
        device,
        hidden_dims=(32,),
    )
    static.fit(train, dev)
    embedding = Stage6CandidatewiseReranker(
        "embedding_reranker",
        lambda rows: _hashed_candidate_text_features(rows, feature_dim=256, mode="full"),
        baseline_training,
        seed + 6_600,
        device,
        hidden_dims=(64,),
    )
    embedding.fit(train, dev)
    single = Stage6CandidatewiseReranker(
        "single_agent_full_context_reviewer",
        lambda rows: _hashed_candidate_text_features(rows, feature_dim=256, mode="single_full_context"),
        baseline_training,
        seed + 6_700,
        device,
        hidden_dims=(64,),
    )
    single.fit(train, dev)
    text_multi = Stage6CandidatewiseReranker(
        "text_only_multi_agent_reviewer",
        lambda rows: _hashed_candidate_text_features(rows, feature_dim=256, mode="multi_role_text"),
        baseline_training,
        seed + 6_800,
        device,
        hidden_dims=(64,),
    )
    text_multi.fit(train, dev)
    best_generator = BestGeneratorOnDevBaseline()
    best_generator.fit(dev)
    return {
        "architecture": architecture,
        "raw_latent": raw,
        "frozen": frozen,
        "trainable": trainable,
        "randomized_labels": randomized,
        "static_patch_feature_reranker": static,
        "embedding_reranker": embedding,
        "single_agent_full_context_reviewer": single,
        "text_only_multi_agent_reviewer": text_multi,
        "best_generator_on_dev_baseline": best_generator,
    }


def _evaluate_methods(
    models: Dict[str, object],
    examples: Sequence[PatchSelectionExample],
    split: str,
    seed: int,
    config: Stage6RunConfig,
) -> Dict[str, Dict[str, object]]:
    del config
    score_rows: Dict[str, Tuple[np.ndarray, bool, Dict[str, object]]] = {
        "random_candidate": (_random_scores(examples, seed), False, _cost_metadata(examples, "random_candidate", 0.0)),
        "first_candidate_order_baseline": (_first_candidate_scores(examples), False, _cost_metadata(examples, "first_candidate", 0.0)),
        "visible_test_heuristic": (_visible_test_scores(examples), False, _cost_metadata(examples, "visible_test_heuristic", 0.0)),
        "oracle_pass_at_8": (label_matrix(examples), False, {"oracle": True}),
        "llm_judge_baseline": (np.zeros((len(examples), STAGE6_NUM_CANDIDATES), dtype=np.float32), not _llm_judge_available(), {"available": _llm_judge_available()}),
    }
    timed = [
        ("best_generator_on_dev_baseline", lambda: models["best_generator_on_dev_baseline"].predict_scores(examples)),
        ("static_patch_feature_reranker", lambda: models["static_patch_feature_reranker"].predict_scores(examples)),
        ("embedding_reranker", lambda: models["embedding_reranker"].predict_scores(examples)),
        ("single_agent_full_context_reviewer", lambda: models["single_agent_full_context_reviewer"].predict_scores(examples)),
        ("text_only_multi_agent_reviewer", lambda: models["text_only_multi_agent_reviewer"].predict_scores(examples)),
        ("raw_latent_selector", lambda: predict_stage6_latent_logits(models["raw_latent"], examples, "none", seed)),
        ("frozen_same_architecture_latent_selector", lambda: predict_stage6_latent_logits(models["frozen"], examples, "none", seed)),
        ("trainable_shared_weight_latent_selector", lambda: predict_stage6_latent_logits(models["trainable"], examples, "none", seed)),
        ("randomized_labels_trainable_latent_selector", lambda: predict_stage6_latent_logits(models["randomized_labels"], examples, "none", seed)),
    ]
    for name, fn in timed:
        start = time.perf_counter()
        scores = fn()
        latency = time.perf_counter() - start
        score_rows[name] = (scores, False, _cost_metadata(examples, name, latency))
    return {
        name: evaluate_scores(name, scores, examples, split=split, unavailable=unavailable, metadata=metadata)
        for name, (scores, unavailable, metadata) in score_rows.items()
    }


def _evaluate_controls(
    trainable: FitResult,
    examples: Sequence[PatchSelectionExample],
    seed: int,
    config: Stage6RunConfig,
) -> Dict[str, object]:
    del config
    out = {}
    control_names = (
        "candidate_only",
        "issue_only",
        "patch_only",
        "context_only",
        "view_masked_candidates_visible",
        "evidence_only_no_candidates",
        "candidate_evidence_mismatch",
        "cross_task_view_bundle_shuffle",
        "schema_template_only",
        "hidden_states_shuffled_across_examples",
        "physical_order_shuffled_roles_avenues_preserved",
        "candidate_order_shuffled_with_label_remap",
    )
    for control in control_names:
        if control == "hidden_states_shuffled_across_examples":
            controlled = list(examples)
            condition = "hidden_states_shuffled_across_examples"
        elif control == "physical_order_shuffled_roles_avenues_preserved":
            controlled = list(examples)
            condition = "physical_order_shuffled_roles_preserved"
        elif control == "candidate_order_shuffled_with_label_remap":
            controlled = apply_stage6_control(examples, control, seed + 7_700)
            condition = "none"
        else:
            controlled = apply_stage6_control(examples, control, seed + 7_700)
            condition = "none"
        scores = predict_stage6_latent_logits(trainable, controlled, condition, seed + 7_701)
        out[control] = evaluate_scores(control, scores, controlled, split="control")
    return out


def _invariance_audit(trainable: FitResult, examples: Sequence[PatchSelectionExample], seed: int) -> Dict[str, object]:
    if not examples:
        return {"passes": False, "reason": "no examples"}
    base_scores = predict_stage6_latent_logits(trainable, examples, "none", seed + 8_000)
    base_pred = _rank_candidate_scores(base_scores, examples)[:, 0]
    physical_scores = predict_stage6_latent_logits(trainable, examples, "physical_order_shuffled_roles_preserved", seed + 8_001)
    physical_pred = _rank_candidate_scores(physical_scores, examples)[:, 0]
    shuffled = apply_stage6_control(examples, "candidate_order_shuffled_with_label_remap", seed + 8_002)
    shuffled_scores = predict_stage6_latent_logits(trainable, shuffled, "none", seed + 8_003)
    shuffled_pred = _rank_candidate_scores(shuffled_scores, shuffled)[:, 0]
    base_candidate_ids = [examples[row].candidates[int(choice)].candidate_id for row, choice in enumerate(base_pred)]
    shuffled_candidate_ids = [shuffled[row].candidates[int(choice)].candidate_id for row, choice in enumerate(shuffled_pred)]
    physical_match = float(np.mean(base_pred == physical_pred)) if len(base_pred) else 0.0
    candidate_match = float(np.mean([left == right for left, right in zip(base_candidate_ids, shuffled_candidate_ids)])) if base_candidate_ids else 0.0
    base_metrics = evaluate_scores("base", base_scores, examples, split="invariance")
    physical_metrics = evaluate_scores("physical", physical_scores, examples, split="invariance")
    shuffled_metrics = evaluate_scores("candidate_order", shuffled_scores, shuffled, split="invariance")
    return {
        "physical_role_avenue_order_prediction_match_rate": physical_match,
        "candidate_order_selected_candidate_id_match_rate": candidate_match,
        "base_pass_at_1": base_metrics["pass_at_1"],
        "physical_order_pass_at_1": physical_metrics["pass_at_1"],
        "candidate_order_pass_at_1": shuffled_metrics["pass_at_1"],
        "physical_order_invariance_passes": physical_match >= 0.95 or abs(float(base_metrics["pass_at_1"]) - float(physical_metrics["pass_at_1"])) <= 0.02,
        "candidate_order_invariance_passes": candidate_match >= 0.95 or abs(float(base_metrics["pass_at_1"]) - float(shuffled_metrics["pass_at_1"])) <= 0.02,
        "passes": bool(
            (physical_match >= 0.95 or abs(float(base_metrics["pass_at_1"]) - float(physical_metrics["pass_at_1"])) <= 0.02)
            and (candidate_match >= 0.95 or abs(float(base_metrics["pass_at_1"]) - float(shuffled_metrics["pass_at_1"])) <= 0.02)
        ),
    }


def _rank_candidate_scores(scores: np.ndarray, examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    if len(examples) == 0:
        return np.zeros((0, STAGE6_NUM_CANDIDATES), dtype=np.int64)
    rows: List[List[int]] = []
    for row, example in enumerate(examples):
        rows.append(
            sorted(
                range(len(example.candidates)),
                key=lambda index: (-float(scores[row, index]), example.candidates[index].candidate_id),
            )
        )
    return np.asarray(rows, dtype=np.int64)


def _static_patch_features(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    rows = []
    agents = sorted({candidate.candidate_source_agent for example in examples for candidate in example.candidates})
    agent_to_index = {name: index for index, name in enumerate(agents[:8])}
    for example in examples:
        per_candidate = []
        for candidate in example.candidates:
            diff = candidate.candidate_diff
            files = edited_files_from_diff(diff)
            added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
            deleted = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
            visible = _visible_score(candidate.candidate_visible_test_result)
            tests = float(any("test" in path.lower() for path in files) or "assert" in diff.lower())
            security = float(bool(__import__("re").search(r"\b(auth|token|password|sql|xss|csrf|permission)\b", diff, __import__("re").IGNORECASE)))
            agent_vec = np.zeros(8, dtype=np.float32)
            if candidate.candidate_source_agent in agent_to_index:
                agent_vec[agent_to_index[candidate.candidate_source_agent]] = 1.0
            numeric = np.asarray(
                [
                    math.log1p(max(0, added)),
                    math.log1p(max(0, deleted)),
                    math.log1p(len(files)),
                    math.log1p(len(diff)),
                    visible,
                    tests,
                    security,
                    float(len(files) > 1),
                ],
                dtype=np.float32,
            )
            per_candidate.append(np.concatenate([numeric, agent_vec]))
        rows.append(per_candidate)
    return np.asarray(rows, dtype=np.float32)


def _hashed_candidate_text_features(examples: Sequence[PatchSelectionExample], feature_dim: int, mode: str) -> np.ndarray:
    rows = []
    for example in examples:
        per_candidate = []
        for candidate in example.candidates:
            if mode == "single_full_context":
                text = _full_context_text(example) + "\n" + candidate.candidate_diff
            elif mode == "multi_role_text":
                text = "\n".join(
                    [
                        "role0:" + example.issue_text + "\n" + example.failing_test_summary,
                        "role1:" + candidate.candidate_diff,
                        "role2:" + "\n".join(example.retrieved_contexts),
                        "role3:" + str(example.metadata.get("dependency_callgraph_related_file_evidence", "")),
                    ]
                )
            elif mode == "full":
                text = _full_context_text(example) + "\n" + candidate.candidate_source_agent + "\n" + candidate.candidate_diff
            else:
                raise ValueError(f"unknown Stage 6 text feature mode: {mode}")
            per_candidate.append(_hashed_vector(text, feature_dim))
        rows.append(per_candidate)
    return np.asarray(rows, dtype=np.float32)


def _full_context_text(example: PatchSelectionExample) -> str:
    return "\n".join(
        [
            example.repo,
            example.issue_text,
            example.failing_test_summary,
            "\n".join(example.retrieved_contexts),
            str(example.metadata.get("dependency_callgraph_related_file_evidence", "")),
        ]
    )


def _hashed_vector(text: str, feature_dim: int) -> np.ndarray:
    values = np.zeros(feature_dim, dtype=np.float32)
    for token in _text_tokens(text):
        raw = __import__("hashlib").sha256(token.encode("utf-8")).hexdigest()
        index = int(raw[:8], 16) % feature_dim
        sign = 1.0 if int(raw[8:10], 16) % 2 == 0 else -1.0
        values[index] += sign
    norm = float(np.linalg.norm(values))
    return values / max(1.0, norm)


def _random_scores(examples: Sequence[PatchSelectionExample], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 9_100)
    return rng.normal(size=(len(examples), STAGE6_NUM_CANDIDATES)).astype(np.float32)


def _first_candidate_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    rows = np.zeros((len(examples), STAGE6_NUM_CANDIDATES), dtype=np.float32)
    if len(examples):
        rows[:, 0] = 1.0
    return rows


def _visible_test_scores(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    return np.asarray(
        [[_visible_score(candidate.candidate_visible_test_result) for candidate in example.candidates] for example in examples],
        dtype=np.float32,
    )


def _visible_score(value: str | None) -> float:
    text = str(value or "").lower()
    if "pass" in text and "fail" not in text:
        return 1.0
    if "fail" in text or "error" in text:
        return -1.0
    return 0.0


def _group_accuracy(examples: Sequence[PatchSelectionExample], correct: np.ndarray, key_fn: Callable[[PatchSelectionExample], str]) -> Dict[str, float]:
    grouped: Dict[str, List[bool]] = {}
    for example, value in zip(examples, correct):
        grouped.setdefault(str(key_fn(example)), []).append(bool(value))
    return {key: float(np.mean(values)) for key, values in sorted(grouped.items())}


def _selected_generator_accuracy(examples: Sequence[PatchSelectionExample], predictions: np.ndarray, correct: np.ndarray) -> Dict[str, float]:
    grouped: Dict[str, List[bool]] = {}
    for row, example in enumerate(examples):
        generator = example.candidates[int(predictions[row])].candidate_source_agent
        grouped.setdefault(generator, []).append(bool(correct[row]))
    return {key: float(np.mean(values)) for key, values in sorted(grouped.items())}


def _phase_config(phase: str) -> Stage6RunConfig:
    if phase == "smoke":
        return Stage6RunConfig(phase=phase, synthetic_if_missing=True, synthetic_tasks=24, hidden_dim=16, max_length=96, epochs=1, patience=1, batch_size=8)
    if phase == "6A":
        return Stage6RunConfig(phase=phase, synthetic_if_missing=False, synthetic_tasks=48, hidden_dim=24, max_length=128, epochs=2, patience=1, batch_size=8)
    if phase in {"6B", "6B-real"}:
        return Stage6RunConfig(phase=phase, synthetic_if_missing=False, synthetic_tasks=160, hidden_dim=32, max_length=160, epochs=3, patience=2, batch_size=8)
    return Stage6RunConfig(phase=phase, synthetic_if_missing=False, synthetic_tasks=240, hidden_dim=32, max_length=160, epochs=3, patience=2, batch_size=8)


def _phase_seeds(phase: str) -> Tuple[int, ...]:
    if phase in {"smoke", "6A"}:
        return (0,)
    if phase in {"6B", "6B-real"}:
        return (0, 1, 2)
    return (10, 11, 12, 13, 14, 15, 16, 17, 18, 19)


def _phase_architectures(phase: str) -> Tuple[Stage6Architecture, ...]:
    baseline = Stage6Architecture(
        name="stage4_4x1_candidate_token_direct",
        description="Stage 4 candidate-token-direct selector with four roles and one avenue per role.",
        num_avenues=1,
        avenue_prompt_mode="single",
    )
    avenue = Stage6Architecture(
        name="stage5_4x4_avenue_candidate_token_direct",
        description="Stage 5 candidate-token-direct selector with four roles and four Stage 6 avenue prompts per role.",
        num_avenues=4,
        avenue_prompt_mode="stage6_patch_selection",
    )
    if phase == "6B":
        return (baseline, avenue)
    if phase == "6B-real":
        return (avenue,)
    return (avenue,)


def _agent_config(config: Stage6RunConfig) -> SharedTransformerAgentConfig:
    return SharedTransformerAgentConfig(
        agent_mode="tiny_transformer",
        max_length=int(config.max_length),
        hidden_dim=int(config.hidden_dim),
        tiny_vocab_size=4096,
        tiny_layers=int(config.tiny_layers),
        tiny_heads=int(config.tiny_heads),
        tiny_ff_dim=int(config.tiny_ff_dim),
        adapter_hidden_dim=int(config.hidden_dim),
    )


def _training_config(config: Stage6RunConfig, architecture: Stage6Architecture) -> RealSharedWeightTrainingConfig:
    return RealSharedWeightTrainingConfig(
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        lr=float(architecture.lr),
        weight_decay=float(architecture.weight_decay),
        patience=int(config.patience),
        gradient_accumulation_steps=1,
        mixed_precision=str(config.mixed_precision),
        gradient_clip_norm=float(architecture.gradient_clip_norm),
    )


def _message_config(architecture: Stage6Architecture) -> MessageChannelConfig:
    return MessageChannelConfig(
        use_msg_token=False,
        readout_source="pooled",
        use_message_head=False,
        coordinator_family=str(architecture.coordinator_family),
        use_private_cue_aux=False,
        num_avenues=int(architecture.num_avenues),
        avenue_prompt_mode=str(architecture.avenue_prompt_mode),
        avenue_dropout=0.0,
        canonicalize_role_order=True,
        canonicalize_avenue_order=True,
    )


def _coordinator_config(config: Stage6RunConfig, architecture: Stage6Architecture) -> LatentCoordinatorConfig:
    return LatentCoordinatorConfig(
        family=str(architecture.coordinator_family),
        input_dim=int(config.hidden_dim),
        model_dim=int(config.hidden_dim),
        num_heads=1,
        num_layers=1,
        ff_dim=max(16, int(config.hidden_dim) * 2),
        dropout=0.0,
        num_avenues=int(architecture.num_avenues),
        avenue_topk=int(architecture.avenue_topk),
    )


def _split_examples(examples: Sequence[PatchSelectionExample]) -> Dict[str, List[PatchSelectionExample]]:
    splits = {"train": [], "dev": [], "test": []}
    for example in examples:
        if example.split in splits:
            splits[example.split].append(example)
    if all(splits.values()):
        return splits
    ordered = list(examples)
    n = len(ordered)
    train_cut = max(1, int(round(n * 0.6)))
    dev_cut = max(train_cut + 1, int(round(n * 0.8)))
    return {"train": ordered[:train_cut], "dev": ordered[train_cut:dev_cut], "test": ordered[dev_cut:]}


def _canonical_candidate_order(examples: Sequence[PatchSelectionExample]) -> List[PatchSelectionExample]:
    out = []
    for example in examples:
        order = sorted(
            range(len(example.candidates)),
            key=lambda index: (example.candidates[index].candidate_id, example.candidates[index].patch_hash),
        )
        if order == list(range(len(example.candidates))):
            out.append(example)
            continue
        out.append(
            replace(
                example,
                candidates=tuple(example.candidates[index] for index in order),
                labels_pass_fail=tuple(int(example.labels_pass_fail[index]) for index in order),
            )
        )
    return out


def _restore_candidate_order_scores(
    canonical_scores: np.ndarray,
    canonical_examples: Sequence[PatchSelectionExample],
    original_examples: Sequence[PatchSelectionExample],
) -> np.ndarray:
    if len(canonical_examples) != len(original_examples):
        raise ValueError("canonical and original example counts differ")
    restored = np.zeros_like(canonical_scores)
    for row, (canonical, original) in enumerate(zip(canonical_examples, original_examples)):
        index_by_id = {candidate.candidate_id: index for index, candidate in enumerate(canonical.candidates)}
        for original_index, candidate in enumerate(original.candidates):
            restored[row, original_index] = canonical_scores[row, index_by_id[candidate.candidate_id]]
    return restored


def _resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(device)


def _cost_metadata(examples: Sequence[PatchSelectionExample], method: str, latency_seconds: float) -> Dict[str, object]:
    token_count = 0
    for example in examples:
        token_count += len(_text_tokens(_full_context_text(example)))
        token_count += sum(len(_text_tokens(candidate.candidate_diff)) for candidate in example.candidates)
    return {
        "method": method,
        "estimated_tokens_total": int(token_count),
        "estimated_tokens_per_decision": float(token_count / max(1, len(examples))),
        "latency_seconds_total": float(latency_seconds),
        "latency_seconds_per_decision": float(latency_seconds / max(1, len(examples))),
    }


def _llm_judge_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("STAGE6_LLM_JUDGE_MODEL"))


def _save_checkpoints(
    checkpoint_dir: Path,
    seed: int,
    architecture: Stage6Architecture,
    models: Dict[str, object],
    config: Stage6RunConfig,
) -> Dict[str, str]:
    root = checkpoint_dir / architecture.name / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {
        "trainable": root / "trainable_shared_weight_latent_selector.pt",
        "frozen": root / "frozen_same_architecture_latent_selector.pt",
        "raw_latent": root / "raw_latent_selector.pt",
        "randomized_labels": root / "randomized_labels_trainable_latent_selector.pt",
        "static_patch_feature_reranker": root / "static_patch_feature_reranker.pt",
        "embedding_reranker": root / "embedding_reranker.pt",
        "single_agent_full_context_reviewer": root / "single_agent_full_context_reviewer.pt",
        "text_only_multi_agent_reviewer": root / "text_only_multi_agent_reviewer.pt",
    }
    for name in ("trainable", "frozen", "raw_latent", "randomized_labels"):
        fit = models[name]
        torch.save(
            {
                "benchmark": BENCHMARK,
                "seed": int(seed),
                "architecture": asdict(architecture),
                "config": asdict(config),
                "method": fit.method,
                "system_state_dict": fit.system.state_dict(),
                "audit": fit.audit,
                "history": fit.history,
            },
            paths[name],
        )
    for name in ("static_patch_feature_reranker", "embedding_reranker", "single_agent_full_context_reviewer", "text_only_multi_agent_reviewer"):
        model = models[name]
        torch.save(
            {
                "benchmark": BENCHMARK,
                "seed": int(seed),
                "method": model.method,
                "state_dict": model.model.state_dict() if model.model is not None else None,
                "param_count": model.param_count,
                "history": model.history,
            },
            paths[name],
        )
    return {name: str(path) for name, path in paths.items()}


def _audit_subset(audit: Dict[str, object]) -> Dict[str, object]:
    keys = (
        "agent_grad_norm_mean",
        "agent_parameter_delta",
        "coordinator_grad_norm_mean",
        "coordinator_parameter_delta",
        "shared_parameter_identity",
        "activation_requires_grad_before_coordinator",
        "clone_activation_requires_grad",
        "no_detach_between_clone_activations_and_loss",
        "per_clone_gradient_contribution",
        "agent_trainable",
        "multi_positive_loss",
        "num_avenues",
        "avenue_prompt_mode",
        "device",
        "batch_size",
        "gradient_clip_norm",
        "param_count",
    )
    return {key: audit.get(key) for key in keys}


def _audit_rows_from_seed_row(row: Dict[str, object]) -> List[Dict[str, object]]:
    return [
        {"benchmark": BENCHMARK, "seed": row["seed"], "type": "trainable_training_audit", **row.get("trainable_audit", {})},
        {"benchmark": BENCHMARK, "seed": row["seed"], "type": "frozen_training_audit", **row.get("frozen_audit", {})},
        {"benchmark": BENCHMARK, "seed": row["seed"], "type": "invariance_audit", **row.get("invariance_audit", {})},
    ]


def _seed_success_gates(row: Dict[str, object]) -> Dict[str, object]:
    metrics = row.get("metrics", {}).get("test") or row.get("metrics", {}).get("dev", {})
    train = float(metrics.get("trainable_shared_weight_latent_selector", {}).get("pass_at_1", 0.0))
    frozen = float(metrics.get("frozen_same_architecture_latent_selector", {}).get("pass_at_1", 0.0))
    baseline_names = [name for name in metrics if name not in {"trainable_shared_weight_latent_selector", "oracle_pass_at_8"}]
    best_baseline = max([float(metrics[name].get("pass_at_1", 0.0)) for name in baseline_names], default=0.0)
    collapse_threshold = max(0.20, train - 0.05)
    controls = row.get("controls", {})
    shortcut_names = (
        "candidate_only",
        "issue_only",
        "patch_only",
        "context_only",
        "view_masked_candidates_visible",
        "evidence_only_no_candidates",
        "candidate_evidence_mismatch",
        "cross_task_view_bundle_shuffle",
        "schema_template_only",
    )
    shortcut_values = {name: float(controls.get(name, {}).get("pass_at_1", 1.0)) for name in shortcut_names if name in controls}
    randomized_value = float(metrics.get("randomized_labels_trainable_latent_selector", {}).get("pass_at_1", 1.0))
    hidden_shuffle_value = float(controls.get("hidden_states_shuffled_across_examples", {}).get("pass_at_1", 1.0))
    return {
        "trainable_beats_frozen": train > frozen,
        "trainable_beats_best_non_oracle_baseline_by_5_points": train >= best_baseline + 0.05,
        "selection_efficiency_improves_over_best_baseline": float(metrics.get("trainable_shared_weight_latent_selector", {}).get("selection_efficiency", 0.0))
        > max([float(metrics[name].get("selection_efficiency", 0.0)) for name in baseline_names], default=0.0),
        "gradient_audits_pass": bool(
            float(row.get("trainable_audit", {}).get("agent_grad_norm_mean") or 0.0) > 0.0
            and float(row.get("trainable_audit", {}).get("agent_parameter_delta") or 0.0) > 0.0
            and float(row.get("frozen_audit", {}).get("agent_grad_norm_mean") or 0.0) == 0.0
            and float(row.get("frozen_audit", {}).get("agent_parameter_delta") or 0.0) == 0.0
        ),
        "leakage_audits_pass": bool(row.get("split_leakage_audit_passes") and row.get("output_leakage_audit_passes") and row.get("duplicate_candidate_patch_hash_audit", {}).get("passes")),
        "order_invariance_passes": bool(row.get("invariance_audit", {}).get("passes", False)),
        "shortcut_controls_collapse": bool(shortcut_values and all(value <= collapse_threshold for value in shortcut_values.values())),
        "shortcut_control_pass_at_1": shortcut_values,
        "randomized_labels_collapse": randomized_value <= collapse_threshold,
        "hidden_state_shuffle_collapse": hidden_shuffle_value <= collapse_threshold,
        "checkpoints_saved": all(Path(str(path)).exists() for path in row.get("checkpoint_paths", {}).values()),
    }


def _overall_summary(rows: Sequence[Dict[str, object]], phase: str) -> Dict[str, object]:
    completed = [row for row in rows if row.get("status") == "completed"]
    if not completed:
        return {"completed_seeds": 0, "success_gates_pass": False}
    final_rows = completed
    train_values = [_metric_value(row, "trainable_shared_weight_latent_selector", "pass_at_1") for row in final_rows]
    frozen_values = [_metric_value(row, "frozen_same_architecture_latent_selector", "pass_at_1") for row in final_rows]
    baseline_by_row = [_best_non_oracle_baseline(row) for row in final_rows]
    diffs_best = [train - best for train, best in zip(train_values, baseline_by_row)]
    ci_low, ci_high = paired_bootstrap_ci(np.asarray(diffs_best, dtype=np.float64), seed=73)
    wins_frozen = sum(train > frozen for train, frozen in zip(train_values, frozen_values))
    completed_seeds = len({int(row["seed"]) for row in final_rows})
    final_gate_applicable = phase == "6C"
    return {
        "completed_rows": len(completed),
        "completed_seeds": completed_seeds,
        "mean_trainable_pass_at_1": _mean(train_values),
        "mean_frozen_pass_at_1": _mean(frozen_values),
        "mean_best_non_oracle_baseline_pass_at_1": _mean(baseline_by_row),
        "mean_delta_vs_best_non_oracle_baseline": _mean(diffs_best),
        "paired_bootstrap_95ci_delta_vs_best_baseline": [ci_low, ci_high],
        "trainable_beats_frozen_seed_count": wins_frozen,
        "final_gate_applicable": final_gate_applicable,
        "success_gates_pass": bool(
            final_gate_applicable
            and completed_seeds >= 10
            and wins_frozen >= 8
            and _mean(diffs_best) >= 0.05
            and ci_low > 0.02
            and all(bool(row.get("success_gates", {}).get("leakage_audits_pass", False)) for row in final_rows)
            and all(bool(row.get("success_gates", {}).get("order_invariance_passes", False)) for row in final_rows)
            and all(bool(row.get("success_gates", {}).get("shortcut_controls_collapse", False)) for row in final_rows)
            and all(bool(row.get("success_gates", {}).get("randomized_labels_collapse", False)) for row in final_rows)
            and all(bool(row.get("success_gates", {}).get("hidden_state_shuffle_collapse", False)) for row in final_rows)
            and all(bool(row.get("success_gates", {}).get("checkpoints_saved", False)) for row in final_rows)
        ),
        "claim_status": (
            "success_template_available"
            if final_gate_applicable and completed_seeds >= 10 and wins_frozen >= 8 and _mean(diffs_best) >= 0.05 and ci_low > 0.02
            else "no_final_claim"
        ),
    }


def _metric_value(row: Dict[str, object], method: str, key: str) -> float:
    split_metrics = row.get("metrics", {}).get("test") or row.get("metrics", {}).get("dev", {})
    return float(split_metrics.get(method, {}).get(key, 0.0))


def _best_non_oracle_baseline(row: Dict[str, object]) -> float:
    split_metrics = row.get("metrics", {}).get("test") or row.get("metrics", {}).get("dev", {})
    values = [
        float(metrics.get("pass_at_1", 0.0))
        for name, metrics in split_metrics.items()
        if name not in {"trainable_shared_weight_latent_selector", "oracle_pass_at_8"}
    ]
    return max(values) if values else 0.0


def _architecture_selection(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        if row.get("status") == "completed":
            grouped.setdefault(str(row.get("architecture")), []).append(row)
    summaries = []
    for architecture, items in grouped.items():
        dev_train = [
            float(row.get("metrics", {}).get("dev", {}).get("trainable_shared_weight_latent_selector", {}).get("pass_at_1", 0.0))
            for row in items
        ]
        dev_frozen = [
            float(row.get("metrics", {}).get("dev", {}).get("frozen_same_architecture_latent_selector", {}).get("pass_at_1", 0.0))
            for row in items
        ]
        summaries.append(
            {
                "architecture": architecture,
                "n_completed": len(items),
                "seeds": [int(row["seed"]) for row in items],
                "mean_dev_trainable_pass_at_1": _mean(dev_train),
                "mean_dev_frozen_pass_at_1": _mean(dev_frozen),
                "mean_dev_delta_vs_frozen": _mean([left - right for left, right in zip(dev_train, dev_frozen)]),
                "selection_uses_test": False,
            }
        )
    summaries.sort(
        key=lambda row: (
            float(row.get("mean_dev_trainable_pass_at_1", 0.0)),
            float(row.get("mean_dev_delta_vs_frozen", 0.0)),
        ),
        reverse=True,
    )
    return {
        "selection_rule": "highest mean dev pass@1, tie-broken by mean dev delta vs frozen; test metrics ignored",
        "selected_architecture": summaries[0]["architecture"] if summaries else None,
        "summaries": summaries,
    }


def _paired_tests_for_split(metrics: Dict[str, Dict[str, object]], examples: Sequence[PatchSelectionExample], seed: int) -> Dict[str, object]:
    if not metrics or "trainable_shared_weight_latent_selector" not in metrics or not examples:
        return {}
    labels = label_matrix(examples)
    train_pred = np.asarray(metrics["trainable_shared_weight_latent_selector"].get("predictions", []), dtype=np.int64)
    if train_pred.size != labels.shape[0]:
        return {}
    train_correct = np.asarray([labels[row, int(choice)] > 0 for row, choice in enumerate(train_pred)], dtype=bool)
    candidates = []
    for name, row in metrics.items():
        if name in {"trainable_shared_weight_latent_selector", "oracle_pass_at_8"} or bool(row.get("unavailable", False)):
            continue
        pred = np.asarray(row.get("predictions", []), dtype=np.int64)
        if pred.size != labels.shape[0]:
            continue
        correct = np.asarray([labels[item, int(choice)] > 0 for item, choice in enumerate(pred)], dtype=bool)
        candidates.append((name, correct, float(np.mean(correct)) if correct.size else 0.0))
    if not candidates:
        return {}
    best_name, best_correct, best_pass = max(candidates, key=lambda row: row[2])
    train_values = train_correct.astype(np.float64)
    best_values = best_correct.astype(np.float64)
    diff = train_values - best_values
    ci_low, ci_high = paired_bootstrap_ci(diff, seed=seed + 991)
    return {
        "best_baseline": best_name,
        "trainable_pass_at_1": float(np.mean(train_values)),
        "best_baseline_pass_at_1": best_pass,
        "mean_delta": float(np.mean(diff)),
        "paired_bootstrap_95ci": [ci_low, ci_high],
        "paired_permutation_pvalue": paired_permutation_pvalue(train_values, best_values, seed=seed + 992),
        "mcnemar_pvalue": mcnemar_pvalue(train_correct, best_correct),
    }


def paired_bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 2000) -> Tuple[float, float]:
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed + 906_001)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, values.size, size=values.size)
        means.append(float(np.mean(values[idx])))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_permutation_pvalue(left: np.ndarray, right: np.ndarray, seed: int, n_permutations: int = 5000) -> float:
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if diff.size == 0:
        return 1.0
    observed = abs(float(np.mean(diff)))
    rng = np.random.default_rng(seed + 906_101)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=diff.size)
        if abs(float(np.mean(diff * signs))) >= observed:
            count += 1
    return float((count + 1) / (n_permutations + 1))


def mcnemar_pvalue(left_correct: np.ndarray, right_correct: np.ndarray) -> float:
    left = np.asarray(left_correct, dtype=bool)
    right = np.asarray(right_correct, dtype=bool)
    b = int(np.sum(left & ~right))
    c = int(np.sum(~left & right))
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return float(min(1.0, 2.0 * tail))


def _write_outputs(result: Dict[str, object], audit_rows: Sequence[Dict[str, object]], results_path: Path, audit_path: Path, report_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    audit_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in audit_rows), encoding="utf-8")
    report_path.write_text(render_stage6_report(result), encoding="utf-8")


def _write_splits(path: Path, splits: Dict[str, Sequence[PatchSelectionExample]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({split: [example.id for example in rows] for split, rows in splits.items()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def render_stage6_report(result: Dict[str, object]) -> str:
    metadata = result.get("metadata", {})
    summary = result.get("summary", {})
    rows = [row for row in result.get("rows", []) if row.get("status") == "completed"]
    lines = [
        "# Stage 6: Latent Patch Selector",
        "",
        "## Scope",
        "",
        "- Selector only: fixed candidate patches are supplied; this experiment does not claim patch generation.",
        "- Pool size: K=8 candidate patches per task.",
        "- Primary metric: unconditional pass@1, with conditional selector accuracy reported only where oracle pass@8 > 0.",
        f"- Phase: `{metadata.get('phase')}`.",
        f"- Candidate pool: `{metadata.get('candidate_pool_path')}`.",
        "",
        "## Dataset",
        "",
        f"- Summary: `{json.dumps(metadata.get('dataset_summary', {}), sort_keys=True)}`",
        f"- Validation passes: `{metadata.get('dataset_validation', {}).get('passes')}`",
        "",
        "## Results",
        "",
        "| seed | architecture | trainable pass@1 | frozen pass@1 | best non-oracle baseline | oracle pass@8 | order invariant | leakage |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        metrics = row.get("metrics", {}).get("test") or row.get("metrics", {}).get("dev", {})
        train = float(metrics.get("trainable_shared_weight_latent_selector", {}).get("pass_at_1", 0.0))
        frozen = float(metrics.get("frozen_same_architecture_latent_selector", {}).get("pass_at_1", 0.0))
        oracle = float(metrics.get("oracle_pass_at_8", {}).get("oracle_pass_at_8", 0.0))
        lines.append(
            "| {seed} | `{arch}` | {train:.4f} | {frozen:.4f} | {best:.4f} | {oracle:.4f} | `{inv}` | `{leak}` |".format(
                seed=row.get("seed"),
                arch=row.get("architecture"),
                train=train,
                frozen=frozen,
                best=_best_non_oracle_baseline(row),
                oracle=oracle,
                inv=row.get("invariance_audit", {}).get("passes"),
                leak=row.get("success_gates", {}).get("leakage_audits_pass"),
            )
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Completed seeds: `{summary.get('completed_seeds', 0)}`",
            f"- Mean trainable pass@1: `{summary.get('mean_trainable_pass_at_1', 0.0):.4f}`",
            f"- Mean best non-oracle baseline pass@1: `{summary.get('mean_best_non_oracle_baseline_pass_at_1', 0.0):.4f}`",
            f"- Mean delta vs best baseline: `{summary.get('mean_delta_vs_best_non_oracle_baseline', 0.0):.4f}`",
            f"- Bootstrap 95% CI: `{summary.get('paired_bootstrap_95ci_delta_vs_best_baseline', [0.0, 0.0])}`",
            f"- Final gates pass: `{summary.get('success_gates_pass', False)}`",
            "",
            "## Conclusion",
            "",
        ]
    )
    if bool(summary.get("success_gates_pass", False)):
        lines.append(
            "On fixed candidate pools for SWE-bench-style issue-resolution tasks, the shared-weight latent cloned-agent selector "
            "improved patch selection over frozen latent, text-only, LLM judge, heuristic, and single-context baselines under "
            "strict leakage, shortcut, and invariance controls. This supports latent coordination as a decision/reranking "
            "architecture, not end-to-end patch generation superiority."
        )
    else:
        lines.append(
            "No final Stage 6 claim is made from this run. The report is a pipeline or validation artifact unless the 6C gates pass on held-out tasks."
        )
    return "\n".join(lines) + "\n"


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
