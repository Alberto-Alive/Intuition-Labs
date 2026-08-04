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
from src.datasets.taubench_trajectory_selection_dataset import (
    STAGE7_CONTROL_NAMES,
    STAGE7_NUM_CANDIDATES,
    TauBenchTrajectorySelectionExample,
    apply_stage7_control,
    build_taubench_selection_examples,
    duplicate_trajectory_hash_audit,
    load_taubench_candidates_jsonl,
    near_duplicate_trajectory_audit,
    randomized_label_matrix,
    stage7_dataset_summary,
    stage7_output_leakage_audit,
    stage7_split_leakage_audit,
    taubench_label_matrix,
    taubench_to_multiview_many,
    trajectory_candidate_text,
    trajectory_static_feature_vector,
    validate_taubench_selection_examples,
    write_taubench_candidates_jsonl,
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
    _replace_dataclass,
    _system_checkpoint,
    _text_tokens,
    shared_parameter_identity_check,
)
from src.experiments.stage7_taubench_candidate_generator import build_synthetic_stage7_candidate_records


BENCHMARK = "stage7_taubench_trajectory_selector"
DEFAULT_STAGE7A_CANDIDATES = Path("results/stage7a_taubench_smoke_candidates.jsonl")
DEFAULT_STAGE7B_CANDIDATES = Path("results/stage7b_taubench_dev_candidates.jsonl")
DEFAULT_STAGE7C_CANDIDATES = Path("results/stage7c_taubench_final_candidates.jsonl")
DEFAULT_SPLITS_PATH = Path("results/stage7_taubench_splits.json")
DEFAULT_RESULTS_PATH = Path("results/stage7_taubench_selector_results.json")
DEFAULT_AUDIT_PATH = Path("results/stage7_taubench_selector_audit.jsonl")
DEFAULT_REPORT_PATH = Path("reports/STAGE7_TAUBENCH_TRAJECTORY_SELECTOR.md")
DEFAULT_CHECKPOINT_DIR = Path("results/stage7_taubench_selector_checkpoints")


@dataclass(frozen=True)
class Stage7Architecture:
    name: str
    description: str
    num_avenues: int
    avenue_prompt_mode: str = "stage7_taubench_trajectory_selection"
    avenue_topk: int = 0
    coordinator_family: str = "candidate_token_cross_attention"
    lr: float = 0.0003
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class Stage7RunConfig:
    phase: str = "7A"
    domain: str = "retail"
    device: str = "cpu"
    synthetic_if_missing: bool = False
    synthetic_tasks: int = 20
    seed_for_pool: int = 707_000
    hidden_dim: int = 32
    tiny_layers: int = 1
    tiny_heads: int = 1
    tiny_ff_dim: int = 64
    max_length: int = 192
    epochs: int = 3
    patience: int = 2
    batch_size: int = 8
    mixed_precision: str = "none"
    run_controls: bool = True
    save_checkpoints: bool = True
    run_latent: bool = True
    bootstrap_samples: int = 2000


class Stage7CandidatewiseReranker:
    def __init__(
        self,
        method: str,
        feature_builder: Callable[[Sequence[TauBenchTrajectorySelectionExample]], np.ndarray],
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

    def fit(
        self,
        train_examples: Sequence[TauBenchTrajectorySelectionExample],
        dev_examples: Sequence[TauBenchTrajectorySelectionExample],
    ) -> None:
        torch.manual_seed(self.seed + 71_706)
        x_train = self.feature_builder(train_examples)
        y_train = taubench_label_matrix(train_examples)
        if not np.any(y_train.sum(axis=1) > 0):
            raise ValueError(f"{self.method} requires at least one oracle-positive train example")
        self.model = CandidatewiseScoringModel(input_dim=x_train.shape[-1], hidden_dims=self.hidden_dims).to(self.device)
        self.param_count = int(sum(parameter.numel() for parameter in self.model.parameters()))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training.lr, weight_decay=self.training.weight_decay)
        tx = torch.as_tensor(x_train, dtype=torch.float32, device=self.device)
        ty = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)
        best_state = None
        best_dev = -1.0
        stale = 0
        rng = np.random.default_rng(self.seed + 71_707)
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

    def predict_scores(self, examples: Sequence[TauBenchTrajectorySelectionExample]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.method} is not fit")
        features = torch.as_tensor(self.feature_builder(examples), dtype=torch.float32, device=self.device)
        rows = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, features.shape[0], 2048):
                rows.append(self.model(features[start : start + 2048]).detach().float().cpu().numpy())
        return np.concatenate(rows, axis=0).astype(np.float32) if rows else np.zeros((0, STAGE7_NUM_CANDIDATES), dtype=np.float32)


class BestGeneratorOnDevBaseline:
    def __init__(self) -> None:
        self.method = "best_generator_on_dev_baseline"
        self.generator_scores: Dict[str, float] = {}

    def fit(self, dev_examples: Sequence[TauBenchTrajectorySelectionExample]) -> None:
        wins: Dict[str, List[int]] = {}
        for example in dev_examples:
            for candidate, label in zip(example.candidates, example.labels_pass_fail):
                wins.setdefault(candidate.generator_name, []).append(int(label))
        self.generator_scores = {name: float(np.mean(values)) if values else 0.0 for name, values in wins.items()}

    def predict_scores(self, examples: Sequence[TauBenchTrajectorySelectionExample]) -> np.ndarray:
        rows = []
        for example in examples:
            rows.append([self.generator_scores.get(candidate.generator_name, 0.0) for candidate in example.candidates])
        return np.asarray(rows, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 7 tau-bench latent trajectory selector.")
    parser.add_argument("--phase", choices=("7A", "7B", "7C", "7D", "smoke"), default="7A")
    parser.add_argument("--domain", choices=("retail", "airline"), default=None)
    parser.add_argument("--candidate-pool", default=None)
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--synthetic-if-missing", action="store_true")
    parser.add_argument("--synthetic-tasks", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--no-latent", action="store_true")
    args = parser.parse_args()

    config = _phase_config(args.phase)
    if args.domain is not None:
        config = replace(config, domain=str(args.domain))
    if args.synthetic_tasks is not None:
        config = replace(config, synthetic_tasks=int(args.synthetic_tasks))
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
        run_controls=not bool(args.no_controls),
        run_latent=not bool(args.no_latent),
    )
    candidate_pool = Path(args.candidate_pool) if args.candidate_pool else _default_candidate_pool(args.phase)
    result = run_stage7(
        candidate_pool_path=candidate_pool,
        results_path=Path(args.results),
        audit_path=Path(args.audit),
        splits_path=Path(args.splits),
        report_path=Path(args.report),
        checkpoint_dir=Path(args.checkpoint_dir),
        config=config,
    )
    print(
        "stage7: wrote {results}, {audit}, {report}; rows={rows}; final_claim_allowed={claim}".format(
            results=args.results,
            audit=args.audit,
            report=args.report,
            rows=len(result.get("rows", [])),
            claim=result.get("summary", {}).get("stage7c_final_claim_allowed", False),
        )
    )


def run_stage7(
    candidate_pool_path: Path,
    results_path: Path = DEFAULT_RESULTS_PATH,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    splits_path: Path = DEFAULT_SPLITS_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    config: Stage7RunConfig | None = None,
) -> Dict[str, object]:
    config = config or Stage7RunConfig()
    candidates = load_taubench_candidates_jsonl(candidate_pool_path)
    if not candidates and config.synthetic_if_missing:
        task_ids = [str(index) for index in range(int(config.synthetic_tasks))]
        candidates = build_synthetic_stage7_candidate_records(
            domain=config.domain,
            task_ids=task_ids,
            raw_root=Path("results/stage7_synthetic_raw_trajectories"),
            eval_root=Path("results/stage7_synthetic_eval_metadata"),
            seed=config.seed_for_pool,
        )
        write_taubench_candidates_jsonl(candidate_pool_path, candidates)
    if not candidates:
        raise FileNotFoundError(
            f"No Stage 7 candidate pool found at {candidate_pool_path}. Generate/evaluate tau-bench candidates first."
        )
    examples = build_taubench_selection_examples(candidates)
    validation = validate_taubench_selection_examples(examples, require_labels=True)
    splits = _split_examples(examples, phase=config.phase, splits_path=splits_path, seed=config.seed_for_pool)
    _write_splits(splits_path, splits, config)
    seeds = _phase_seeds(config.phase)
    architectures = _phase_architectures(config.phase, splits_path)
    rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    selected_architecture = architectures[-1].name
    for seed in seeds:
        all_split_examples = [example for rows_in in splits.values() for example in rows_in]
        split_leakage = stage7_split_leakage_audit(BENCHMARK, seed, splits)
        output_leakage = stage7_output_leakage_audit(BENCHMARK, seed, all_split_examples)
        duplicate_audit = duplicate_trajectory_hash_audit(all_split_examples)
        near_duplicate_audit = near_duplicate_trajectory_audit(all_split_examples)
        audit_rows.extend(
            [
                split_leakage,
                output_leakage,
                {"benchmark": BENCHMARK, "seed": seed, "type": "duplicate_trajectory_hash", **duplicate_audit},
                {"benchmark": BENCHMARK, "seed": seed, "type": "near_duplicate_trajectory", **near_duplicate_audit},
            ]
        )
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
                near_duplicate_audit=near_duplicate_audit,
            )
            rows.append(row)
            audit_rows.extend(_audit_rows_from_seed_row(row))
    architecture_selection = _architecture_selection(rows) if config.phase == "7B" else {}
    if architecture_selection:
        selected_architecture = str(architecture_selection.get("selected_architecture", selected_architecture))
        _write_selected_architecture(splits_path, selected_architecture)
    result = {
        "metadata": {
            "benchmark": BENCHMARK,
            "created_at_utc": _now(),
            "candidate_pool_path": str(candidate_pool_path),
            "selector_scope": "selector only; no trajectory generation or end-to-end tau-bench SOTA claim",
            "primary_claim_template": (
                "Given a fixed pool of K=8 tau-bench trajectories for the same task, a trained shared-weight "
                "latent cloned-agent selector chooses successful trajectories more reliably than specified baselines."
            ),
            "phase": config.phase,
            "domain": config.domain,
            "selected_architecture": selected_architecture,
            "architecture_selection": architecture_selection,
            "config": asdict(config),
            "dataset_validation": validation,
            "dataset_summary": stage7_dataset_summary(splits),
        },
        "rows": rows,
        "summary": _overall_summary(rows, config.phase),
    }
    _write_outputs(result, audit_rows, results_path, audit_path, report_path)
    return result


def fit_stage7_latent_selector(
    train_examples: Sequence[TauBenchTrajectorySelectionExample],
    dev_examples: Sequence[TauBenchTrajectorySelectionExample],
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
    train_examples, train_order = _canonical_candidate_order(train_examples)
    dev_examples, dev_order = _canonical_candidate_order(dev_examples)
    train_mv = taubench_to_multiview_many(train_examples)
    dev_mv = taubench_to_multiview_many(dev_examples)
    train_masks = _reorder_label_masks(train_label_masks, train_order) if train_label_masks is not None else taubench_label_matrix(train_examples)
    dev_masks = _reorder_label_masks(dev_label_masks, dev_order) if dev_label_masks is not None else taubench_label_matrix(dev_examples)
    if not np.any(train_masks.sum(axis=1) > 0):
        raise ValueError("Stage 7 latent training requires at least one oracle-positive train example")

    torch.manual_seed(seed + 719_007)
    np_rng = np.random.default_rng(seed + 729_001)
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
            num_classes=STAGE7_NUM_CANDIDATES,
            config=effective_coordinator_config,
        ).to(device)
    else:
        raise ValueError(f"unknown Stage 7 coordinator family: {message_config.coordinator_family}")
    system = SharedClonedAgentSystem(
        agent,
        coordinator,
        n_roles=len(train_mv[0].views),
        visible_explicit_evidence=False,
        message_config=message_config,
    ).to(device)
    params = [parameter for parameter in system.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("no trainable parameters available for Stage 7 latent selector")
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
                valid_idx = valid.nonzero(as_tuple=False).flatten()
                main_loss = multi_positive_selector_loss(logits.index_select(0, valid_idx), labels.index_select(0, valid_idx))
                aux_weight = _auxiliary_loss_weight(epoch, message_config)
                aux_loss = torch.zeros((), dtype=main_loss.dtype, device=main_loss.device)
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
        train_scores = predict_stage7_latent_logits_from_system(system, train_mv, condition="none", seed=seed)
        dev_scores = predict_stage7_latent_logits_from_system(system, dev_mv, condition="none", seed=seed)
        train_metrics = evaluate_scores("trainable", train_scores, train_examples, split="train", unavailable=False)
        dev_metrics = evaluate_scores("trainable", dev_scores, dev_examples, split="dev", unavailable=False)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(epoch_loss_sum / max(1, epoch_loss_count)),
                "train_pass_at_1": float(train_metrics["pass_at_1"]),
                "dev_pass_at_1": float(dev_metrics["pass_at_1"]),
                "dev_conditional_selector_accuracy": float(dev_metrics["conditional_selector_accuracy"]),
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
    identity = shared_parameter_identity_check(agent, len(train_mv[0].views))
    audit = {
        "benchmark": BENCHMARK,
        "seed": int(seed),
        "split": "dev",
        "probe": "stage7_multi_positive_training_audit",
        "method": method,
        "agent_trainable": bool(trainable_agent),
        "multi_positive_loss": "negative_log_sum_softmax_probability_assigned_to_successful_candidates",
        "message_config": asdict(message_config),
        "coordinator_config": asdict(coordinator_config),
        "shared_parameter_identity": identity["pass"],
        "shared_parameter_count": identity["parameter_count"],
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


def predict_stage7_latent_logits(
    result: FitResult,
    examples: Sequence[TauBenchTrajectorySelectionExample],
    condition: str,
    seed: int,
) -> np.ndarray:
    canonical, _orders = _canonical_candidate_order(examples)
    mv = taubench_to_multiview_many(canonical)
    canonical_scores = predict_stage7_latent_logits_from_system(result.system, mv, condition=condition, seed=seed)
    return _restore_candidate_order_scores(canonical_scores, canonical, examples)


def predict_stage7_latent_logits_from_system(system: SharedClonedAgentSystem, examples, condition: str, seed: int) -> np.ndarray:
    system.eval()
    rows = []
    with torch.no_grad():
        for batch in _example_batches(examples, 128):
            logits, _audit, _acts = system(batch, condition=condition, seed=seed)
            rows.append(logits.detach().float().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32) if rows else np.zeros((0, STAGE7_NUM_CANDIDATES), dtype=np.float32)


def evaluate_scores(
    method: str,
    scores: np.ndarray,
    examples: Sequence[TauBenchTrajectorySelectionExample],
    split: str,
    unavailable: bool = False,
    metadata: Dict[str, object] | None = None,
) -> Dict[str, object]:
    labels = taubench_label_matrix(examples)
    if unavailable:
        scores = np.zeros((len(examples), STAGE7_NUM_CANDIDATES), dtype=np.float32)
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
        "selected_candidate_ids": [examples[row].candidates[int(choice)].candidate_id for row, choice in enumerate(pred)] if len(examples) else [],
        "correct_by_example": [bool(value) for value in correct.tolist()],
        "oracle_positive_by_example": [bool(value) for value in positive.tolist()],
        "per_domain_accuracy": _group_accuracy(examples, correct, lambda example: example.domain),
        "per_task_type_accuracy": _group_accuracy(examples, correct, lambda example: str(example.metadata.get("task_type", "unknown"))),
        "per_selected_generator_accuracy_audit_only": _selected_generator_accuracy(examples, pred, correct),
        "cost_tokens_latency": metadata or {},
    }


def _run_seed(
    config: Stage7RunConfig,
    architecture: Stage7Architecture,
    splits: Dict[str, Sequence[TauBenchTrajectorySelectionExample]],
    seed: int,
    checkpoint_dir: Path,
    split_leakage: Dict[str, object],
    output_leakage: Dict[str, object],
    duplicate_audit: Dict[str, object],
    near_duplicate_audit: Dict[str, object],
) -> Dict[str, object]:
    start = time.perf_counter()
    device = _resolve_device(config.device)
    agent_config = _agent_config(config)
    training = _training_config(config, architecture)
    message_config = _message_config(architecture)
    coordinator_config = _coordinator_config(config, architecture)
    models = _fit_models(splits, seed, device, agent_config, training, message_config, coordinator_config, architecture, config)
    metrics = {
        split: _evaluate_methods(models, list(examples), split, seed, config)
        for split, examples in splits.items()
        if examples
    }
    eval_examples = list(splits.get("test") or splits.get("dev") or [])
    controls = _evaluate_controls(models, eval_examples, seed, config) if config.run_controls else {}
    invariance = _invariance_audit(models.get("trainable"), eval_examples, seed) if models.get("trainable") is not None else {"passes": False, "reason": "latent disabled"}
    paired_tests = _paired_tests_for_split(
        metrics.get("test") or metrics.get("dev", {}),
        eval_examples,
        seed,
        bootstrap_samples=int(config.bootstrap_samples),
    )
    checkpoint_paths = _save_checkpoints(checkpoint_dir, seed, architecture, models, config) if config.save_checkpoints else {}
    row = {
        "stage": BENCHMARK,
        "phase": config.phase,
        "domain": config.domain,
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
        "duplicate_trajectory_hash_audit": duplicate_audit,
        "near_duplicate_trajectory_audit": near_duplicate_audit,
        "trainable_audit": _audit_subset(models["trainable"].audit) if models.get("trainable") is not None else {},
        "frozen_audit": _audit_subset(models["frozen"].audit) if models.get("frozen") is not None else {},
        "raw_latent_audit": _audit_subset(models["raw_latent"].audit) if models.get("raw_latent") is not None else {},
        "randomized_labels_audit": _audit_subset(models["randomized_labels"].audit) if models.get("randomized_labels") is not None else {},
        "generator_identity_only_diagnostic": _generator_identity_only_diagnostic(splits.get("dev", []), eval_examples),
    }
    row["success_gates"] = _seed_success_gates(row)
    return row


def _fit_models(
    splits: Dict[str, Sequence[TauBenchTrajectorySelectionExample]],
    seed: int,
    device: str,
    agent_config: SharedTransformerAgentConfig,
    training: RealSharedWeightTrainingConfig,
    message_config: MessageChannelConfig,
    coordinator_config: LatentCoordinatorConfig,
    architecture: Stage7Architecture,
    config: Stage7RunConfig,
) -> Dict[str, object]:
    train = list(splits["train"])
    dev = list(splits["dev"] or splits["test"])
    models: Dict[str, object] = {"architecture": architecture}
    if config.run_latent:
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
        models["raw_latent"] = fit_stage7_latent_selector(
            train,
            dev,
            agent_config=agent_config,
            coordinator_config=raw_coord,
            training_config=training,
            seed=seed + 7_100,
            device=device,
            trainable_agent=True,
            method="raw_latent_selector",
            message_config=raw_message,
        )
        models["frozen"] = fit_stage7_latent_selector(
            train,
            dev,
            agent_config=agent_config,
            coordinator_config=coordinator_config,
            training_config=training,
            seed=seed + 7_200,
            device=device,
            trainable_agent=False,
            method="frozen_same_architecture_latent_selector",
            message_config=message_config,
        )
        models["trainable"] = fit_stage7_latent_selector(
            train,
            dev,
            agent_config=agent_config,
            coordinator_config=coordinator_config,
            training_config=training,
            seed=seed + 7_300,
            device=device,
            trainable_agent=True,
            method="trainable_shared_weight_latent_selector",
            message_config=message_config,
        )
        models["randomized_labels"] = fit_stage7_latent_selector(
            train,
            dev,
            agent_config=agent_config,
            coordinator_config=coordinator_config,
            training_config=training,
            seed=seed + 7_400,
            device=device,
            trainable_agent=True,
            method="randomized_labels_trainable_latent_selector",
            message_config=message_config,
            train_label_masks=randomized_label_matrix(train, seed + 7_401),
            dev_label_masks=randomized_label_matrix(dev, seed + 7_402),
        )
    baseline_training = MLPTrainingConfig(
        epochs=max(2, training.epochs),
        batch_size=training.batch_size,
        lr=0.002,
        weight_decay=0.0001,
        patience=max(1, training.patience),
        hidden_dims=(64,),
    )
    static = Stage7CandidatewiseReranker(
        "static_trajectory_feature_reranker",
        _static_trajectory_features,
        baseline_training,
        seed + 7_500,
        device,
        hidden_dims=(32,),
    )
    static.fit(train, dev)
    single = Stage7CandidatewiseReranker(
        "text_only_single_reviewer",
        lambda rows: _hashed_candidate_text_features(rows, feature_dim=256, mode="single_full_context"),
        baseline_training,
        seed + 7_600,
        device,
        hidden_dims=(64,),
    )
    single.fit(train, dev)
    text_multi = Stage7CandidatewiseReranker(
        "text_only_multi_agent_reviewer",
        lambda rows: _hashed_candidate_text_features(rows, feature_dim=256, mode="multi_role_text"),
        baseline_training,
        seed + 7_700,
        device,
        hidden_dims=(64,),
    )
    text_multi.fit(train, dev)
    best_generator = BestGeneratorOnDevBaseline()
    best_generator.fit(dev)
    models.update(
        {
            "static_trajectory_feature_reranker": static,
            "text_only_single_reviewer": single,
            "text_only_multi_agent_reviewer": text_multi,
            "best_generator_on_dev_baseline": best_generator,
        }
    )
    return models


def _evaluate_methods(
    models: Dict[str, object],
    examples: Sequence[TauBenchTrajectorySelectionExample],
    split: str,
    seed: int,
    config: Stage7RunConfig,
) -> Dict[str, Dict[str, object]]:
    del config
    llm_scores, llm_unavailable, llm_meta = _llm_judge_scores(examples)
    score_rows: Dict[str, Tuple[np.ndarray, bool, Dict[str, object]]] = {
        "random_trajectory": (_random_scores(examples, seed), False, _cost_metadata(examples, "random_trajectory", 0.0)),
        "first_trajectory_order_baseline": (_first_candidate_scores(examples), False, _cost_metadata(examples, "first_trajectory", 0.0)),
        "embedding_reranker": (_embedding_reranker_scores(examples), False, _cost_metadata(examples, "embedding_reranker", 0.0)),
        "oracle_pass_at_8": (taubench_label_matrix(examples), False, {"oracle": True}),
        "llm_judge_baseline": (llm_scores, llm_unavailable, llm_meta),
    }
    timed = [
        ("best_generator_on_dev_baseline", lambda: models["best_generator_on_dev_baseline"].predict_scores(examples)),
        ("static_trajectory_feature_reranker", lambda: models["static_trajectory_feature_reranker"].predict_scores(examples)),
        ("text_only_single_reviewer", lambda: models["text_only_single_reviewer"].predict_scores(examples)),
        ("text_only_multi_agent_reviewer", lambda: models["text_only_multi_agent_reviewer"].predict_scores(examples)),
    ]
    if models.get("raw_latent") is not None:
        timed.extend(
            [
                ("raw_latent_selector", lambda: predict_stage7_latent_logits(models["raw_latent"], examples, "none", seed)),
                ("frozen_same_architecture_latent_selector", lambda: predict_stage7_latent_logits(models["frozen"], examples, "none", seed)),
                ("trainable_shared_weight_latent_selector", lambda: predict_stage7_latent_logits(models["trainable"], examples, "none", seed)),
                ("randomized_labels_trainable_latent_selector", lambda: predict_stage7_latent_logits(models["randomized_labels"], examples, "none", seed)),
            ]
        )
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
    models: Dict[str, object],
    examples: Sequence[TauBenchTrajectorySelectionExample],
    seed: int,
    config: Stage7RunConfig,
) -> Dict[str, object]:
    del config
    trainable = models.get("trainable")
    if trainable is None or not examples:
        return {}
    out = {}
    control_names = (
        "randomized_labels",
        "candidate_order_shuffled_with_label_remap",
        "physical_order_shuffled_roles_avenues_preserved",
        "candidate_only",
        "user_goal_only",
        "policy_only",
        "trajectory_only",
        "tool_observations_only",
        "evidence_only_no_candidate",
        "candidate_evidence_mismatch",
        "cross_task_user_goal_shuffle",
        "cross_task_policy_shuffle",
        "cross_task_tool_observation_shuffle",
        "hidden_states_shuffled_across_examples",
        "schema_template_only",
    )
    for control in control_names:
        if control == "hidden_states_shuffled_across_examples":
            controlled = list(examples)
            condition = "hidden_states_shuffled_across_examples"
        elif control == "physical_order_shuffled_roles_avenues_preserved":
            controlled = list(examples)
            condition = "physical_order_shuffled_roles_preserved"
        elif control == "candidate_order_shuffled_with_label_remap":
            controlled = apply_stage7_control(examples, control, seed + 7_800)
            condition = "none"
        else:
            controlled = apply_stage7_control(examples, control, seed + 7_800)
            condition = "none"
        scores = predict_stage7_latent_logits(trainable, controlled, condition, seed + 7_801)
        out[control] = evaluate_scores(control, scores, controlled, split="control")
    return out


def _invariance_audit(trainable: FitResult | None, examples: Sequence[TauBenchTrajectorySelectionExample], seed: int) -> Dict[str, object]:
    if trainable is None:
        return {"passes": False, "reason": "latent disabled"}
    if not examples:
        return {"passes": False, "reason": "no examples"}
    base_scores = predict_stage7_latent_logits(trainable, examples, "none", seed + 8_000)
    base_pred = _rank_candidate_scores(base_scores, examples)[:, 0]
    physical_scores = predict_stage7_latent_logits(trainable, examples, "physical_order_shuffled_roles_preserved", seed + 8_001)
    physical_pred = _rank_candidate_scores(physical_scores, examples)[:, 0]
    shuffled = apply_stage7_control(examples, "candidate_order_shuffled_with_label_remap", seed + 8_002)
    shuffled_scores = predict_stage7_latent_logits(trainable, shuffled, "none", seed + 8_003)
    shuffled_pred = _rank_candidate_scores(shuffled_scores, shuffled)[:, 0]
    base_candidate_ids = [examples[row].candidates[int(choice)].candidate_id for row, choice in enumerate(base_pred)]
    shuffled_candidate_ids = [shuffled[row].candidates[int(choice)].candidate_id for row, choice in enumerate(shuffled_pred)]
    physical_match = float(np.mean(base_pred == physical_pred)) if len(base_pred) else 0.0
    candidate_match = float(np.mean([left == right for left, right in zip(base_candidate_ids, shuffled_candidate_ids)])) if base_candidate_ids else 0.0
    base_metrics = evaluate_scores("base", base_scores, examples, split="invariance")
    physical_metrics = evaluate_scores("physical", physical_scores, examples, split="invariance")
    shuffled_metrics = evaluate_scores("candidate_order", shuffled_scores, shuffled, split="invariance")
    physical_pass = physical_match >= 0.95 or abs(float(base_metrics["pass_at_1"]) - float(physical_metrics["pass_at_1"])) <= 0.02
    candidate_pass = candidate_match >= 0.95 or abs(float(base_metrics["pass_at_1"]) - float(shuffled_metrics["pass_at_1"])) <= 0.02
    return {
        "physical_role_avenue_order_prediction_match_rate": physical_match,
        "candidate_order_selected_candidate_id_match_rate": candidate_match,
        "base_pass_at_1": base_metrics["pass_at_1"],
        "physical_order_pass_at_1": physical_metrics["pass_at_1"],
        "candidate_order_pass_at_1": shuffled_metrics["pass_at_1"],
        "physical_order_invariance_passes": physical_pass,
        "candidate_order_invariance_passes": candidate_pass,
        "passes": bool(physical_pass and candidate_pass),
    }


def _rank_candidate_scores(scores: np.ndarray, examples: Sequence[TauBenchTrajectorySelectionExample]) -> np.ndarray:
    if len(examples) == 0:
        return np.zeros((0, STAGE7_NUM_CANDIDATES), dtype=np.int64)
    rows: List[List[int]] = []
    for row, example in enumerate(examples):
        rows.append(
            sorted(
                range(len(example.candidates)),
                key=lambda index: (-float(scores[row, index]), example.candidates[index].candidate_id),
            )
        )
    return np.asarray(rows, dtype=np.int64)


def _static_trajectory_features(examples: Sequence[TauBenchTrajectorySelectionExample]) -> np.ndarray:
    rows = []
    for example in examples:
        rows.append([trajectory_static_feature_vector(example, candidate) for candidate in example.candidates])
    return np.asarray(rows, dtype=np.float32)


def _hashed_candidate_text_features(
    examples: Sequence[TauBenchTrajectorySelectionExample],
    feature_dim: int,
    mode: str,
) -> np.ndarray:
    rows = []
    for example in examples:
        role_texts = [view.text for view in taubench_to_multiview_many([example])[0].views]
        full_context = "\n".join(role_texts)
        per_candidate = []
        for candidate in example.candidates:
            candidate_text = trajectory_candidate_text(candidate)
            if mode == "single_full_context":
                text = full_context + "\n" + candidate_text
            elif mode == "multi_role_text":
                text = "\n".join(f"role{index}:{role}" for index, role in enumerate(role_texts)) + "\n" + candidate_text
            elif mode == "candidate_only":
                text = candidate_text
            else:
                raise ValueError(f"unknown Stage 7 text feature mode: {mode}")
            per_candidate.append(_hashed_vector(text, feature_dim))
        rows.append(per_candidate)
    return np.asarray(rows, dtype=np.float32)


def _embedding_reranker_scores(examples: Sequence[TauBenchTrajectorySelectionExample]) -> np.ndarray:
    rows = []
    for example in examples:
        role_texts = [view.text for view in taubench_to_multiview_many([example])[0].views]
        query = _hashed_vector("\n".join(role_texts[:2] + role_texts[3:]), 384)
        values = []
        for candidate in example.candidates:
            vector = _hashed_vector(trajectory_candidate_text(candidate), 384)
            values.append(float(np.dot(query, vector)))
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def _random_scores(examples: Sequence[TauBenchTrajectorySelectionExample], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 9_700)
    return rng.normal(size=(len(examples), STAGE7_NUM_CANDIDATES)).astype(np.float32)


def _first_candidate_scores(examples: Sequence[TauBenchTrajectorySelectionExample]) -> np.ndarray:
    rows = np.zeros((len(examples), STAGE7_NUM_CANDIDATES), dtype=np.float32)
    if len(examples):
        rows[:, 0] = 1.0
    return rows


def _llm_judge_scores(examples: Sequence[TauBenchTrajectorySelectionExample]) -> Tuple[np.ndarray, bool, Dict[str, object]]:
    if not _llm_judge_available():
        return (
            np.zeros((len(examples), STAGE7_NUM_CANDIDATES), dtype=np.float32),
            True,
            {"available": False, "reason": "OPENAI_API_KEY and STAGE7_LLM_JUDGE_MODEL are required"},
        )
    return (
        np.zeros((len(examples), STAGE7_NUM_CANDIDATES), dtype=np.float32),
        True,
        {"available": True, "not_executed_by_default": True, "reason": "LLM judge API hook is configured but intentionally not called inside unit runner"},
    )


def _llm_judge_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("STAGE7_LLM_JUDGE_MODEL"))


def _hashed_vector(text: str, feature_dim: int) -> np.ndarray:
    values = np.zeros(feature_dim, dtype=np.float32)
    for token in _text_tokens(text):
        raw = __import__("hashlib").sha256(token.encode("utf-8")).hexdigest()
        index = int(raw[:8], 16) % feature_dim
        sign = 1.0 if int(raw[8:10], 16) % 2 == 0 else -1.0
        values[index] += sign
    norm = float(np.linalg.norm(values))
    return values / max(1.0, norm)


def _group_accuracy(
    examples: Sequence[TauBenchTrajectorySelectionExample],
    correct: np.ndarray,
    key_fn: Callable[[TauBenchTrajectorySelectionExample], str],
) -> Dict[str, float]:
    grouped: Dict[str, List[bool]] = {}
    for example, value in zip(examples, correct):
        grouped.setdefault(str(key_fn(example)), []).append(bool(value))
    return {key: float(np.mean(values)) for key, values in sorted(grouped.items())}


def _selected_generator_accuracy(
    examples: Sequence[TauBenchTrajectorySelectionExample],
    predictions: np.ndarray,
    correct: np.ndarray,
) -> Dict[str, float]:
    grouped: Dict[str, List[bool]] = {}
    for row, example in enumerate(examples):
        generator = example.candidates[int(predictions[row])].generator_name
        grouped.setdefault(generator, []).append(bool(correct[row]))
    return {key: float(np.mean(values)) for key, values in sorted(grouped.items())}


def _phase_config(phase: str) -> Stage7RunConfig:
    if phase == "smoke":
        return Stage7RunConfig(phase=phase, synthetic_if_missing=True, synthetic_tasks=24, hidden_dim=16, max_length=128, epochs=1, patience=1, batch_size=8, bootstrap_samples=200)
    if phase == "7A":
        return Stage7RunConfig(phase=phase, domain="retail", synthetic_if_missing=False, synthetic_tasks=20, hidden_dim=24, max_length=160, epochs=2, patience=1, batch_size=8)
    if phase == "7B":
        return Stage7RunConfig(phase=phase, domain="retail", synthetic_if_missing=False, synthetic_tasks=100, hidden_dim=32, max_length=192, epochs=3, patience=2, batch_size=8)
    if phase == "7D":
        return Stage7RunConfig(phase=phase, domain="airline", synthetic_if_missing=False, synthetic_tasks=100, hidden_dim=32, max_length=192, epochs=3, patience=2, batch_size=8)
    return Stage7RunConfig(phase=phase, domain="retail", synthetic_if_missing=False, synthetic_tasks=100, hidden_dim=32, max_length=192, epochs=3, patience=2, batch_size=8)


def _phase_seeds(phase: str) -> Tuple[int, ...]:
    if phase in {"smoke", "7A"}:
        return (0,)
    if phase == "7B":
        return (0, 1, 2)
    return (10, 11, 12, 13, 14, 15, 16, 17, 18, 19)


def _phase_architectures(phase: str, splits_path: Path) -> Tuple[Stage7Architecture, ...]:
    baseline = Stage7Architecture(
        name="stage4_4x1_candidate_token_direct",
        description="Candidate-token-direct selector with four roles and one avenue per role.",
        num_avenues=1,
        avenue_prompt_mode="single",
    )
    avenue = Stage7Architecture(
        name="stage5_stage7_4x4_candidate_token_direct",
        description="Stage 5/6 candidate-token-direct selector with four tau-bench roles and four Stage 7 avenues per role.",
        num_avenues=4,
        avenue_prompt_mode="stage7_taubench_trajectory_selection",
    )
    if phase == "7B":
        return (baseline, avenue)
    if phase in {"7C", "7D"}:
        selected = _read_selected_architecture(splits_path)
        if selected == baseline.name:
            return (baseline,)
    return (avenue,)


def _agent_config(config: Stage7RunConfig) -> SharedTransformerAgentConfig:
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


def _training_config(config: Stage7RunConfig, architecture: Stage7Architecture) -> RealSharedWeightTrainingConfig:
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


def _message_config(architecture: Stage7Architecture) -> MessageChannelConfig:
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


def _coordinator_config(config: Stage7RunConfig, architecture: Stage7Architecture) -> LatentCoordinatorConfig:
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


def _split_examples(
    examples: Sequence[TauBenchTrajectorySelectionExample],
    phase: str,
    splits_path: Path,
    seed: int,
) -> Dict[str, List[TauBenchTrajectorySelectionExample]]:
    saved = _read_saved_split_map(splits_path)
    if saved:
        assigned = {"train": [], "dev": [], "test": []}
        for example in examples:
            split = saved.get(f"{example.domain}:{example.task_id}", example.split)
            if split in assigned:
                assigned[split].append(replace(example, split=split))
        if assigned["train"] and (assigned["dev"] or assigned["test"]):
            return assigned
    split_field = {"train": [], "dev": [], "test": []}
    for example in examples:
        if example.split in split_field:
            split_field[example.split].append(example)
    if split_field["train"] and split_field["dev"] and (phase in {"7B"} or split_field["test"]):
        return split_field
    ordered = sorted(examples, key=lambda example: _stable_split_key(example, seed))
    n = len(ordered)
    if phase == "7B":
        train_cut = max(1, int(round(n * 0.8)))
        return {"train": ordered[:train_cut], "dev": ordered[train_cut:], "test": []}
    train_cut = max(1, int(round(n * 0.6)))
    dev_cut = max(train_cut + 1, int(round(n * 0.8)))
    return {"train": ordered[:train_cut], "dev": ordered[train_cut:dev_cut], "test": ordered[dev_cut:]}


def _canonical_candidate_order(
    examples: Sequence[TauBenchTrajectorySelectionExample],
) -> Tuple[List[TauBenchTrajectorySelectionExample], List[List[int]]]:
    out = []
    orders = []
    for example in examples:
        order = sorted(
            range(len(example.candidates)),
            key=lambda index: (example.candidates[index].candidate_id, example.candidates[index].trajectory_hash),
        )
        orders.append(order)
        if order == list(range(len(example.candidates))):
            out.append(example)
            continue
        out.append(replace(example, candidates=tuple(example.candidates[index] for index in order)))
    return out, orders


def _reorder_label_masks(label_masks: np.ndarray | None, orders: Sequence[Sequence[int]]) -> np.ndarray:
    if label_masks is None:
        raise ValueError("label_masks is required")
    masks = np.asarray(label_masks, dtype=np.float32)
    out = np.zeros_like(masks)
    for row, order in enumerate(orders):
        out[row] = masks[row, np.asarray(order, dtype=np.int64)]
    return out


def _restore_candidate_order_scores(
    canonical_scores: np.ndarray,
    canonical_examples: Sequence[TauBenchTrajectorySelectionExample],
    original_examples: Sequence[TauBenchTrajectorySelectionExample],
) -> np.ndarray:
    if len(canonical_examples) != len(original_examples):
        raise ValueError("canonical and original example counts differ")
    restored = np.zeros_like(canonical_scores)
    for row, (canonical, original) in enumerate(zip(canonical_examples, original_examples)):
        index_by_id = {candidate.candidate_id: index for index, candidate in enumerate(canonical.candidates)}
        for original_index, candidate in enumerate(original.candidates):
            restored[row, original_index] = canonical_scores[row, index_by_id[candidate.candidate_id]]
    return restored


def _paired_tests_for_split(
    metrics: Dict[str, Dict[str, object]],
    examples: Sequence[TauBenchTrajectorySelectionExample],
    seed: int,
    bootstrap_samples: int,
) -> Dict[str, object]:
    del examples
    trainable = metrics.get("trainable_shared_weight_latent_selector")
    if not trainable:
        return {"available": False, "reason": "trainable metrics missing"}
    baseline_names = [
        name
        for name in metrics
        if name
        not in {
            "oracle_pass_at_8",
            "trainable_shared_weight_latent_selector",
            "randomized_labels_trainable_latent_selector",
        }
        and not bool(metrics[name].get("unavailable", False))
    ]
    if not baseline_names:
        return {"available": False, "reason": "no non-oracle baseline metrics"}
    best_name = max(baseline_names, key=lambda name: float(metrics[name].get("pass_at_1", 0.0)))
    trainable_correct = np.asarray(trainable.get("correct_by_example", []), dtype=np.float32)
    baseline_correct = np.asarray(metrics[best_name].get("correct_by_example", []), dtype=np.float32)
    diff = trainable_correct - baseline_correct
    ci = _paired_bootstrap_ci(diff, seed=seed, samples=bootstrap_samples)
    sign = _paired_sign_test(diff)
    return {
        "available": True,
        "best_baseline_method": best_name,
        "trainable_pass_at_1": float(trainable.get("pass_at_1", 0.0)),
        "best_baseline_pass_at_1": float(metrics[best_name].get("pass_at_1", 0.0)),
        "trainable_selection_efficiency": float(trainable.get("selection_efficiency", 0.0)),
        "best_baseline_selection_efficiency": float(metrics[best_name].get("selection_efficiency", 0.0)),
        "absolute_pass_at_1_delta": float(np.mean(diff)) if len(diff) else 0.0,
        "paired_bootstrap_95_ci": ci,
        "paired_sign_test": sign,
    }


def _paired_bootstrap_ci(diff: np.ndarray, seed: int, samples: int = 2000) -> Dict[str, float]:
    if len(diff) == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng = np.random.default_rng(seed + 97_001)
    means = []
    for _ in range(max(1, int(samples))):
        idx = rng.integers(0, len(diff), size=len(diff))
        means.append(float(np.mean(diff[idx])))
    return {
        "mean": float(np.mean(diff)),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
    }


def _paired_sign_test(diff: np.ndarray) -> Dict[str, object]:
    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    ties = int(np.sum(diff == 0))
    n = wins + losses
    if n == 0:
        p = 1.0
    elif n <= 200:
        tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / float(2**n)
        p = min(1.0, 2.0 * tail)
    else:
        z = (abs(wins - losses) - 1.0) / math.sqrt(max(1.0, n))
        p = math.erfc(z / math.sqrt(2.0))
    return {"wins": wins, "losses": losses, "ties": ties, "two_sided_p_value": float(p)}


def _generator_identity_only_diagnostic(
    dev_examples: Sequence[TauBenchTrajectorySelectionExample],
    examples: Sequence[TauBenchTrajectorySelectionExample],
) -> Dict[str, object]:
    if not dev_examples or not examples:
        return {"available": False}
    baseline = BestGeneratorOnDevBaseline()
    baseline.fit(dev_examples)
    scores = baseline.predict_scores(examples)
    metrics = evaluate_scores("generator_identity_only_diagnostic_audit_only", scores, examples, split="audit")
    return {
        "available": True,
        "audit_only_not_selector_visible": True,
        "metrics": {key: value for key, value in metrics.items() if key not in {"predictions", "correct_by_example", "selected_candidate_ids"}},
    }


def _seed_success_gates(row: Dict[str, object]) -> Dict[str, bool]:
    target = row.get("metrics", {}).get("test") or row.get("metrics", {}).get("dev", {})
    trainable = target.get("trainable_shared_weight_latent_selector", {}) if isinstance(target, dict) else {}
    frozen = target.get("frozen_same_architecture_latent_selector", {}) if isinstance(target, dict) else {}
    paired = row.get("paired_tests_vs_best_baseline", {})
    controls = row.get("controls", {})
    randomized = target.get("randomized_labels_trainable_latent_selector", {}) if isinstance(target, dict) else {}
    mismatch = controls.get("candidate_evidence_mismatch", {}) if isinstance(controls, dict) else {}
    cross_goal = controls.get("cross_task_user_goal_shuffle", {}) if isinstance(controls, dict) else {}
    base_pass = float(trainable.get("pass_at_1", 0.0))
    return {
        "oracle_pass_at_8_between_0_25_and_0_85": 0.25 <= float(trainable.get("oracle_pass_at_8", 0.0)) <= 0.85,
        "trainable_beats_frozen": base_pass > float(frozen.get("pass_at_1", 0.0)),
        "trainable_beats_best_non_oracle_by_5_points": float(paired.get("absolute_pass_at_1_delta", 0.0)) >= 0.05,
        "bootstrap_ci_lower_vs_best_baseline_gt_0": float(paired.get("paired_bootstrap_95_ci", {}).get("lower", 0.0)) > 0.0,
        "selection_efficiency_improves": float(trainable.get("selection_efficiency", 0.0)) > float(paired.get("best_baseline_selection_efficiency", -1.0)),
        "randomized_labels_collapse": float(randomized.get("pass_at_1", 0.0)) <= max(0.0, base_pass - 0.05),
        "mismatch_or_cross_task_shuffle_degrades": (
            base_pass - min(float(mismatch.get("pass_at_1", base_pass)), float(cross_goal.get("pass_at_1", base_pass))) >= 0.05
        ),
        "candidate_order_invariance_passes": bool(row.get("invariance_audit", {}).get("candidate_order_invariance_passes", False)),
        "physical_role_avenue_order_invariance_passes": bool(row.get("invariance_audit", {}).get("physical_order_invariance_passes", False)),
        "leakage_audits_pass": bool(row.get("output_leakage_audit_passes", False)) and bool(row.get("split_leakage_audit_passes", False)),
        "checkpoints_saved": bool(row.get("checkpoint_paths", {})),
    }


def _overall_summary(rows: Sequence[Dict[str, object]], phase: str) -> Dict[str, object]:
    target_rows = [row for row in rows if row.get("status") == "completed"]
    target_metrics = [((row.get("metrics", {}).get("test") or row.get("metrics", {}).get("dev") or {}), row) for row in target_rows]
    trainable_pass = [
        float(metrics.get("trainable_shared_weight_latent_selector", {}).get("pass_at_1", 0.0))
        for metrics, _row in target_metrics
        if "trainable_shared_weight_latent_selector" in metrics
    ]
    frozen_pass = [
        float(metrics.get("frozen_same_architecture_latent_selector", {}).get("pass_at_1", 0.0))
        for metrics, _row in target_metrics
        if "frozen_same_architecture_latent_selector" in metrics
    ]
    oracle = [
        float(metrics.get("trainable_shared_weight_latent_selector", {}).get("oracle_pass_at_8", 0.0))
        for metrics, _row in target_metrics
        if "trainable_shared_weight_latent_selector" in metrics
    ]
    paired_deltas = [float(row.get("paired_tests_vs_best_baseline", {}).get("absolute_pass_at_1_delta", 0.0)) for _metrics, row in target_metrics]
    gates = {
        "completed_seeds_gte_10": len({int(row.get("seed", -1)) for row in target_rows}) >= 10,
        "oracle_pass_at_8_between_0_25_and_0_85": bool(oracle and 0.25 <= float(np.mean(oracle)) <= 0.85),
        "trainable_beats_frozen_on_8_of_10_seeds": sum(t > f for t, f in zip(trainable_pass, frozen_pass)) >= 8,
        "trainable_beats_best_non_oracle_by_5_points": bool(paired_deltas and float(np.mean(paired_deltas)) >= 0.05),
        "paired_bootstrap_ci_lower_gt_0_all_rows": bool(
            target_rows
            and all(float(row.get("paired_tests_vs_best_baseline", {}).get("paired_bootstrap_95_ci", {}).get("lower", 0.0)) > 0.0 for row in target_rows)
        ),
        "invariance_and_leakage_audits_pass": bool(target_rows and all(bool(row.get("invariance_audit", {}).get("passes", False)) and bool(row.get("output_leakage_audit_passes", False)) for row in target_rows)),
        "checkpoints_saved": bool(target_rows and all(bool(row.get("checkpoint_paths", {})) for row in target_rows)),
    }
    final_claim = bool(phase == "7C" and all(gates.values()))
    return {
        "phase": phase,
        "completed_rows": len(target_rows),
        "completed_seeds": sorted({int(row.get("seed", -1)) for row in target_rows}),
        "trainable_pass_at_1_mean": float(np.mean(trainable_pass)) if trainable_pass else 0.0,
        "trainable_pass_at_1_std": float(np.std(trainable_pass)) if trainable_pass else 0.0,
        "frozen_pass_at_1_mean": float(np.mean(frozen_pass)) if frozen_pass else 0.0,
        "oracle_pass_at_8_mean": float(np.mean(oracle)) if oracle else 0.0,
        "stage7c_success_gates": gates,
        "stage7c_final_claim_allowed": final_claim,
        "conclusion_policy": "No final claim is allowed until Stage 7C gates pass.",
    }


def _architecture_selection(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    dev_scores: Dict[str, List[float]] = {}
    for row in rows:
        if row.get("phase") != "7B":
            continue
        architecture = str(row.get("architecture"))
        metrics = row.get("metrics", {}).get("dev", {})
        value = float(metrics.get("trainable_shared_weight_latent_selector", {}).get("conditional_selector_accuracy", 0.0))
        dev_scores.setdefault(architecture, []).append(value)
    if not dev_scores:
        return {}
    selected = max(sorted(dev_scores), key=lambda name: float(np.mean(dev_scores[name])))
    return {
        "selected_architecture": selected,
        "selection_rule": "highest mean dev conditional selector accuracy across Stage 7B seeds",
        "dev_conditional_accuracy_by_architecture": {name: float(np.mean(values)) for name, values in sorted(dev_scores.items())},
        "no_final_test_metrics_used": True,
    }


def _save_checkpoints(
    checkpoint_dir: Path,
    seed: int,
    architecture: Stage7Architecture,
    models: Dict[str, object],
    config: Stage7RunConfig,
) -> Dict[str, str]:
    root = checkpoint_dir / architecture.name / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for name in ("trainable", "frozen", "raw_latent", "randomized_labels"):
        fit = models.get(name)
        if fit is None:
            continue
        path = root / f"{fit.method}.pt"
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
            path,
        )
        paths[name] = str(path)
    for name in ("static_trajectory_feature_reranker", "text_only_single_reviewer", "text_only_multi_agent_reviewer"):
        model = models.get(name)
        if model is None or getattr(model, "model", None) is None:
            continue
        path = root / f"{name}.pt"
        torch.save(
            {
                "benchmark": BENCHMARK,
                "seed": int(seed),
                "architecture": asdict(architecture),
                "config": asdict(config),
                "method": model.method,
                "model_state_dict": model.model.state_dict(),
                "history": model.history,
                "param_count": model.param_count,
            },
            path,
        )
        paths[name] = str(path)
    return paths


def render_stage7_report(result: Dict[str, object]) -> str:
    meta = result.get("metadata", {})
    summary = result.get("summary", {})
    rows = result.get("rows", [])
    latest = rows[-1] if rows else {}
    target = latest.get("metrics", {}).get("test") or latest.get("metrics", {}).get("dev", {})
    trainable = target.get("trainable_shared_weight_latent_selector", {}) if isinstance(target, dict) else {}
    best = latest.get("paired_tests_vs_best_baseline", {})
    claim_allowed = bool(summary.get("stage7c_final_claim_allowed", False))
    conclusion = (
        "On fixed K=8 tau-bench trajectory pools, the shared-weight latent cloned-agent selector chooses successful "
        "tool-use trajectories more reliably than frozen latent, text-only, embedding, heuristic, and candidate-order "
        "baselines under strict leakage, corruption, and invariance controls. This supports latent coordination as a "
        "trajectory-selection architecture, not end-to-end trajectory generation superiority."
        if claim_allowed
        else "No final benchmark claim is made. Stage 7C success gates have not all passed or have not been run."
    )
    lines = [
        "# Stage 7 tau-bench Trajectory Selector",
        "",
        "## 1. Claim and non-claims",
        "- Claim under test: selector-only choice among fixed K=8 candidate trajectories.",
        "- Non-claims: no trajectory generation superiority and no end-to-end tau-bench SOTA claim.",
        f"- Final claim allowed: `{claim_allowed}`.",
        "",
        "## 2. Benchmark setup",
        f"- Phase: `{meta.get('phase')}`.",
        f"- Domain: `{meta.get('domain')}`.",
        "- Labels are official tau-bench task success/failure from the final evaluator when real tau2 evaluation is used.",
        "",
        "## 3. Candidate trajectory pool construction",
        f"- Candidate pool: `{meta.get('candidate_pool_path')}`.",
        "- K is fixed at 8. Candidate order is randomized and saved. Generator identity is audit metadata only.",
        "",
        "## 4. Selector architecture",
        f"- Selected architecture: `{meta.get('selected_architecture')}`.",
        "- Four roles and four avenues use candidate-token-direct scoring with shared cloned weights.",
        "",
        "## 5. Baselines",
        "- Random, first trajectory, best generator on dev, embedding reranker, static feature reranker, text-only reviewers, LLM judge hook, raw latent, frozen same-architecture latent, and oracle pass@8.",
        "",
        "## 6. Controls and leakage audits",
        f"- Output leakage audit passed: `{latest.get('output_leakage_audit_passes', False)}`.",
        f"- Split leakage audit passed: `{latest.get('split_leakage_audit_passes', False)}`.",
        f"- Invariance audit passed: `{latest.get('invariance_audit', {}).get('passes', False)}`.",
        "",
        "## 7. Main results",
        f"- Trainable pass@1: `{float(trainable.get('pass_at_1', 0.0)):.4f}`.",
        f"- Oracle pass@8: `{float(trainable.get('oracle_pass_at_8', 0.0)):.4f}`.",
        f"- Best baseline: `{best.get('best_baseline_method', 'n/a')}`.",
        f"- Delta vs best baseline: `{float(best.get('absolute_pass_at_1_delta', 0.0)):.4f}`.",
        "",
        "## 8. Ablations",
        "- Controls include randomized labels, candidate/evidence mismatch, cross-task evidence shuffles, schema-only, and single-evidence views.",
        "",
        "## 9. Failure cases",
        "- If trainable does not beat the best baseline, diagnose oracle pass@8, pool difficulty, reviewer saturation, evidence-shuffle degradation, generator identity leakage, pool size, and architecture mismatch.",
        "",
        "## 10. Cost/token/latency",
        "- Per-method cost/tokens/latency metadata is stored in `cost_tokens_latency` for each metric row.",
        "",
        "## 11. Conservative conclusion",
        conclusion,
        "",
    ]
    return "\n".join(lines)


def _write_outputs(
    result: Dict[str, object],
    audit_rows: Sequence[Dict[str, object]],
    results_path: Path,
    audit_path: Path,
    report_path: Path,
) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    audit_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in audit_rows), encoding="utf-8")
    report_path.write_text(render_stage7_report(result), encoding="utf-8")


def _write_splits(splits_path: Path, splits: Dict[str, Sequence[TauBenchTrajectorySelectionExample]], config: Stage7RunConfig) -> None:
    selected = _read_selected_architecture(splits_path)
    row = {
        "benchmark": BENCHMARK,
        "created_at_utc": _now(),
        "phase": config.phase,
        "domain": config.domain,
        "seed": int(config.seed_for_pool),
        "splits": {name: [f"{example.domain}:{example.task_id}" for example in rows] for name, rows in splits.items()},
        "selected_architecture": selected,
        "deterministic_saved_splits": True,
    }
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")


def _read_saved_split_map(splits_path: Path) -> Dict[str, str]:
    if not splits_path.exists():
        return {}
    try:
        row = json.loads(splits_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, str] = {}
    for split, ids in dict(row.get("splits", {})).items():
        for value in ids:
            out[str(value)] = str(split)
    return out


def _write_selected_architecture(splits_path: Path, selected: str) -> None:
    row = {}
    if splits_path.exists():
        try:
            row = json.loads(splits_path.read_text(encoding="utf-8"))
        except Exception:
            row = {}
    row["selected_architecture"] = selected
    row["selected_architecture_source"] = "Stage 7B dev-only selection"
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")


def _read_selected_architecture(splits_path: Path) -> str:
    if not splits_path.exists():
        return ""
    try:
        row = json.loads(splits_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(row.get("selected_architecture", ""))


def _audit_rows_from_seed_row(row: Dict[str, object]) -> List[Dict[str, object]]:
    out = []
    for key in ("trainable_audit", "frozen_audit", "raw_latent_audit", "randomized_labels_audit"):
        audit = row.get(key)
        if isinstance(audit, dict) and audit:
            out.append({"benchmark": BENCHMARK, "seed": row.get("seed"), "type": key, **audit})
    out.append({"benchmark": BENCHMARK, "seed": row.get("seed"), "type": "invariance_audit", **dict(row.get("invariance_audit", {}))})
    out.append({"benchmark": BENCHMARK, "seed": row.get("seed"), "type": "success_gates", **dict(row.get("success_gates", {}))})
    return out


def _audit_subset(audit: Dict[str, object]) -> Dict[str, object]:
    keys = (
        "benchmark",
        "seed",
        "method",
        "agent_trainable",
        "multi_positive_loss",
        "shared_parameter_identity",
        "shared_parameter_count",
        "agent_grad_norm_mean",
        "agent_parameter_delta",
        "coordinator_grad_norm_mean",
        "coordinator_parameter_delta",
        "activation_requires_grad_before_coordinator",
        "no_detach_between_clone_activations_and_loss",
        "loss_backward_reaches_shared_agent",
        "per_clone_gradient_contribution",
        "num_avenues",
        "avenue_prompt_mode",
        "epochs_run",
        "best_dev_conditional_selector_accuracy",
        "param_count",
    )
    return {key: audit.get(key) for key in keys if key in audit}


def _cost_metadata(examples: Sequence[TauBenchTrajectorySelectionExample], method: str, latency_seconds: float) -> Dict[str, object]:
    token_count = 0
    for example in examples:
        token_count += sum(len(_text_tokens(view.text)) for view in taubench_to_multiview_many([example])[0].views)
        token_count += sum(len(_text_tokens(trajectory_candidate_text(candidate))) for candidate in example.candidates)
    return {
        "method": method,
        "estimated_tokens_total": int(token_count),
        "estimated_tokens_per_selector_decision": float(token_count / max(1, len(examples))),
        "latency_seconds_total": float(latency_seconds),
        "latency_seconds_per_selector_decision": float(latency_seconds / max(1, len(examples))),
    }


def _stable_split_key(example: TauBenchTrajectorySelectionExample, seed: int) -> str:
    raw = __import__("hashlib").sha256(f"{seed}:{example.domain}:{example.task_id}".encode("utf-8")).hexdigest()
    return raw


def _resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(device)


def _default_candidate_pool(phase: str) -> Path:
    if phase == "7B":
        return DEFAULT_STAGE7B_CANDIDATES
    if phase in {"7C", "7D"}:
        return DEFAULT_STAGE7C_CANDIDATES
    return DEFAULT_STAGE7A_CANDIDATES


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
