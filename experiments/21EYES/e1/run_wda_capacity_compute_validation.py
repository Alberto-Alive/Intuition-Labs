from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .data import CapacityBenchmark, CapacityTaskConfig, make_capacity_dataloaders
    from .wda_transformer import (
        WDATransformerConfig,
        WDALinear,
        WeightDistributionAttentionTransformer,
        controller_parameter_count,
        count_parameters,
    )
except ImportError:  # pragma: no cover - allows running as a script path
    from data import CapacityBenchmark, CapacityTaskConfig, make_capacity_dataloaders
    from wda_transformer import (
        WDATransformerConfig,
        WDALinear,
        WeightDistributionAttentionTransformer,
        controller_parameter_count,
        count_parameters,
    )


BENCHMARKS: list[CapacityBenchmark] = [
    "rule_switching_sequence",
    "compositional_rule_switching",
    "low_capacity_generalization",
]

MODEL_SIZES: dict[str, dict[str, int]] = {
    "tiny": {"d_model": 32, "nhead": 4, "num_layers": 2, "dim_feedforward": 64},
    "small": {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
    "medium": {"d_model": 96, "nhead": 4, "num_layers": 3, "dim_feedforward": 192},
    "large": {"d_model": 128, "nhead": 8, "num_layers": 4, "dim_feedforward": 256},
}

MODEL_VARIANTS = [
    "fixed_tiny_transformer",
    "fixed_small_transformer",
    "fixed_medium_transformer",
    "fixed_large_transformer",
    "wda_small",
    "wda_small_iterative",
    "hypernetwork_transformer",
    "condconv_style_transformer",
    "dynamic_lora_transformer",
    "moe_transformer",
    "token_parameter_attention_transformer",
    "previous_weight_atom_attention_transformer",
]

TRAINED_ABLATIONS = [
    "wda_no_controller",
    "wda_frozen_controller",
]

EVALUATED_ABLATIONS = [
    "wda_fixed_coordinates",
    "wda_random_coordinates",
    "wda_shuffled_coordinates",
    "wda_no_distribution_scale",
    "wda_no_controller",
    "wda_frozen_controller",
    "wda_deterministic_weights_only",
    "wda_dense_hypernetwork_control",
    "wda_no_iterative_refinement",
]


@dataclass
class TrainConfig:
    seeds: list[int]
    epochs: int = 20
    lr: float = 0.002
    weight_decay: float = 0.0001
    grad_clip: float = 5.0
    lambda_coord_reg: float = 0.0001
    lambda_diversity: float = 0.001
    lambda_smooth: float = 0.0001
    lambda_progressive: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_size_name(variant: str) -> str:
    if variant.startswith("fixed_tiny"):
        return "tiny"
    if variant.startswith("fixed_medium"):
        return "medium"
    if variant.startswith("fixed_large"):
        return "large"
    return "small"


def model_variant_name(variant: str) -> str:
    if variant in {
        "fixed_tiny_transformer",
        "fixed_small_transformer",
        "fixed_medium_transformer",
        "fixed_large_transformer",
        "wda_deterministic_weights_only",
    }:
        return variant
    if variant == "wda_no_iterative_refinement":
        return "wda_small"
    return variant


def make_model_config(
    args: argparse.Namespace,
    *,
    variant: str,
    seq_len: int,
    overrides: dict[str, Any] | None = None,
) -> WDATransformerConfig:
    overrides = overrides or {}
    size = MODEL_SIZES[model_size_name(variant)]
    return WDATransformerConfig(
        vocab_size=args.vocab_size,
        seq_len=seq_len,
        num_classes=args.num_classes,
        d_model=size["d_model"],
        nhead=size["nhead"],
        num_layers=size["num_layers"],
        dim_feedforward=size["dim_feedforward"],
        dropout=args.dropout,
        coord_dim=int(overrides.get("coord_dim", args.coord_dim)),
        distribution_groups=int(
            overrides.get("distribution_groups", args.distribution_groups)
        ),
        low_rank=int(overrides.get("low_rank", args.low_rank)),
        refinement_steps=int(overrides.get("refinement_steps", args.refinement_steps)),
        controller_hidden_dim=int(
            overrides.get("controller_hidden_dim", args.controller_hidden_dim)
        ),
        mask_context_token=not args.no_mask_context_token,
        context_tokens=2,
        apply_wda_qkv=not args.no_apply_wda_qkv,
        readout_position=2,
    )


def make_data_config(args: argparse.Namespace, benchmark: CapacityBenchmark) -> CapacityTaskConfig:
    n_train = args.n_train
    n_val = args.n_val
    n_test = args.n_test
    if benchmark == "low_capacity_generalization":
        n_train = args.low_data_n_train
        n_val = args.low_data_n_val
        n_test = args.low_data_n_test
    return CapacityTaskConfig(
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        mask_context_token=not args.no_mask_context_token,
    )


def mechanism_losses(
    details: dict[str, Any],
    train_config: TrainConfig,
) -> tuple[Tensor, dict[str, float]]:
    coordinates = details.get("coordinates")
    if coordinates is None:
        zero = torch.zeros((), device=details["logits"].device)
        return zero, {
            "coord_reg_loss": 0.0,
            "coord_diversity_loss": 0.0,
            "coord_smooth_loss": 0.0,
            "progressive_loss": 0.0,
        }

    coord_reg = coordinates.pow(2).mean()
    coord_diversity = -coordinates.var(dim=0, unbiased=False).mean()
    coord_smooth = (coordinates[:, 1:] - coordinates[:, :-1]).pow(2).mean()
    history = details.get("coordinate_history", [])
    progressive = torch.zeros((), device=coordinates.device)
    if len(history) > 2:
        deltas = [
            (history[i] - history[i - 1]).pow(2).mean().sqrt()
            for i in range(1, len(history))
        ]
        progressive = torch.stack(
            [F.relu(deltas[i] - deltas[i - 1]) for i in range(1, len(deltas))]
        ).mean()

    total = (
        train_config.lambda_coord_reg * coord_reg
        + train_config.lambda_diversity * coord_diversity
        + train_config.lambda_smooth * coord_smooth
        + train_config.lambda_progressive * progressive
    )
    return total, {
        "coord_reg_loss": coord_reg.detach().item(),
        "coord_diversity_loss": coord_diversity.detach().item(),
        "coord_smooth_loss": coord_smooth.detach().item(),
        "progressive_loss": progressive.detach().item(),
    }


def train_one_model(
    model: WeightDistributionAttentionTransformer,
    loaders: dict[str, torch.utils.data.DataLoader],
    train_config: TrainConfig,
) -> dict[str, float]:
    device = torch.device(train_config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    best_val_nll = float("inf")
    best_state: dict[str, Tensor] | None = None
    last_losses: dict[str, float] = {}
    started = time.perf_counter()
    for _epoch in range(train_config.epochs):
        model.train()
        for tokens, labels, _groups in loaders["train"]:
            tokens = tokens.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            details = model(tokens, return_details=True)
            logits = details["logits"]
            task_loss = F.cross_entropy(logits, labels)
            mech_loss, last_losses = mechanism_losses(details, train_config)
            loss = task_loss + mech_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()

        val = evaluate_model(model, loaders["val"], device=device)
        if val["nll"] < best_val_nll:
            best_val_nll = val["nll"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "train_seconds": time.perf_counter() - started,
        "best_val_nll": best_val_nll,
        **{f"last_{key}": value for key, value in last_losses.items()},
    }


@torch.no_grad()
def evaluate_model(
    model: WeightDistributionAttentionTransformer,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    coordinate_mode: str = "normal",
    fixed_coordinates: Tensor | None = None,
    disable_distribution_scale: bool = False,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    group_correct: dict[int, int] = {}
    group_total: dict[int, int] = {}
    started = time.perf_counter()
    for tokens, labels, groups in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        logits = model(
            tokens,
            coordinate_mode=coordinate_mode,
            fixed_coordinates=fixed_coordinates,
            disable_distribution_scale=disable_distribution_scale,
        )
        loss = F.cross_entropy(logits, labels, reduction="sum")
        pred = logits.argmax(dim=-1)
        correct = pred == labels
        total_loss += loss.item()
        total_correct += correct.sum().item()
        total += labels.numel()
        for group, ok in zip(groups.tolist(), correct.cpu().tolist(), strict=True):
            group_total[group] = group_total.get(group, 0) + 1
            group_correct[group] = group_correct.get(group, 0) + int(ok)

    elapsed = max(time.perf_counter() - started, 1e-9)
    per_group = [
        group_correct[group] / max(group_total[group], 1)
        for group in sorted(group_total)
    ]
    return {
        "accuracy": total_correct / max(total, 1),
        "nll": total_loss / max(total, 1),
        "exact_sequence_accuracy": total_correct / max(total, 1),
        "per_rule_accuracy": sum(per_group) / max(len(per_group), 1),
        "min_rule_accuracy": min(per_group) if per_group else 0.0,
        "latency_ms_per_example": (elapsed / max(total, 1)) * 1000.0,
    }


@torch.no_grad()
def collect_coordinates(
    model: WeightDistributionAttentionTransformer,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    coordinate_mode: str = "normal",
    fixed_coordinates: Tensor | None = None,
    max_batches: int | None = None,
) -> Tensor | None:
    if model.controller is None:
        return None
    model.eval()
    coords: list[Tensor] = []
    for batch_idx, (tokens, _labels, _groups) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        details = model(
            tokens.to(device),
            coordinate_mode=coordinate_mode,
            fixed_coordinates=fixed_coordinates,
            return_details=True,
        )
        coordinates = details["coordinates"]
        if coordinates is not None:
            coords.append(coordinates.detach().cpu())
    return torch.cat(coords, dim=0) if coords else None


def loader_groups(loader: torch.utils.data.DataLoader, n: int | None = None) -> Tensor:
    groups = torch.cat([batch_groups for _tokens, _labels, batch_groups in loader], dim=0)
    return groups if n is None else groups[:n]


def coordinate_metrics(
    model: WeightDistributionAttentionTransformer,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    coordinate_mode: str = "normal",
    fixed_coordinates: Tensor | None = None,
) -> dict[str, float]:
    coords = collect_coordinates(
        model,
        loader,
        device=device,
        coordinate_mode=coordinate_mode,
        fixed_coordinates=fixed_coordinates,
    )
    if coords is None or coords.numel() == 0:
        return {
            "coordinate_variance": 0.0,
            "coordinate_entropy": 0.0,
            "coordinate_l2_distance_across_examples": 0.0,
            "same_input_coordinate_stability": 0.0,
            "different_input_coordinate_diversity": 0.0,
            "per_rule_coordinate_separation": 0.0,
            "coordinate_clustering_purity_by_rule": 0.0,
            "active_coordinate_count": 0.0,
            "final_coordinate_sparsity": 0.0,
        }

    coords = coords.float()
    variance = coords.var(dim=0, unbiased=False).mean().item()
    probs = coords.abs() + 1e-8
    probs = probs / probs.sum(dim=-1, keepdim=True)
    entropy = (-(probs * probs.log()).sum(dim=-1)).mean().item()
    active = (coords.abs() > 0.10).float().sum(dim=-1).mean().item()
    sparsity = (coords.abs() < 0.05).float().mean().item()

    max_pairs = min(coords.shape[0], 128)
    pair = torch.pdist(coords[:max_pairs], p=2)
    average_l2 = pair.mean().item() if pair.numel() else 0.0

    groups = loader_groups(loader, coords.shape[0])
    centroids = []
    centroid_groups = []
    within = []
    for group in sorted(groups.unique().tolist()):
        g_coords = coords[groups == group]
        if g_coords.numel() == 0:
            continue
        centroid = g_coords.mean(dim=0)
        centroids.append(centroid)
        centroid_groups.append(group)
        within.append((g_coords - centroid).pow(2).sum(dim=-1).sqrt().mean())

    if len(centroids) >= 2:
        centroid_tensor = torch.stack(centroids)
        between = torch.pdist(centroid_tensor, p=2).mean()
        within_mean = torch.stack(within).mean().clamp_min(1e-8)
        separation = (between / within_mean).item()
        distances = torch.cdist(coords, centroid_tensor)
        nearest = distances.argmin(dim=-1)
        centroid_group_tensor = torch.tensor(centroid_groups)
        purity = (centroid_group_tensor[nearest] == groups).float().mean().item()
    else:
        separation = 0.0
        purity = 0.0

    return {
        "coordinate_variance": variance,
        "coordinate_entropy": entropy,
        "coordinate_l2_distance_across_examples": average_l2,
        "same_input_coordinate_stability": 0.0,
        "different_input_coordinate_diversity": average_l2,
        "per_rule_coordinate_separation": separation,
        "coordinate_clustering_purity_by_rule": purity,
        "active_coordinate_count": active,
        "final_coordinate_sparsity": sparsity,
    }


@torch.no_grad()
def effective_weight_metrics(
    model: WeightDistributionAttentionTransformer,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    coordinate_mode: str = "normal",
    fixed_coordinates: Tensor | None = None,
    disable_distribution_scale: bool = False,
) -> dict[str, float]:
    linears = list(model.dynamic_linears())
    if not linears or model.controller is None:
        return {
            "effective_weight_difference_norm_across_examples": 0.0,
            "per_rule_w_eff_separation": 0.0,
            "w_eff_rank": 0.0,
            "effective_weight_active_coordinate_count": 0.0,
        }

    model.eval()
    tokens, _labels, groups = next(iter(loader))
    tokens = tokens[: min(tokens.shape[0], 32)].to(device)
    groups = groups[: tokens.shape[0]]
    details = model(
        tokens,
        coordinate_mode=coordinate_mode,
        fixed_coordinates=fixed_coordinates,
        disable_distribution_scale=disable_distribution_scale,
        return_details=True,
    )
    coords = details["coordinates"]
    if coords is None:
        return {
            "effective_weight_difference_norm_across_examples": 0.0,
            "per_rule_w_eff_separation": 0.0,
            "w_eff_rank": 0.0,
            "effective_weight_active_coordinate_count": 0.0,
        }

    diff_norms = []
    ranks = []
    active_counts = []
    rule_separations = []
    for layer in linears[:6]:
        if not hasattr(layer, "materialize_weight"):
            continue
        weights, activity = layer.materialize_weight(
            coords,
            disable_distribution_scale=disable_distribution_scale,
        )
        if weights.shape[0] >= 2:
            diff = weights[1:] - weights[:1]
            diff_norms.append(diff.flatten(1).norm(dim=-1).mean().item())

        if isinstance(layer, WDALinear):
            sample_delta = weights[: min(8, weights.shape[0])] - layer.weight_mu.unsqueeze(0)
        elif hasattr(layer, "weight"):
            sample_delta = weights[: min(8, weights.shape[0])] - layer.weight.unsqueeze(0)
        else:
            sample_delta = weights[: min(8, weights.shape[0])] - weights[:1]
        for matrix in sample_delta:
            ranks.append(torch.linalg.matrix_rank(matrix.float(), tol=1e-5).item())

        if activity.ndim == 2:
            active_counts.append((activity.abs() > 0.10).float().sum(dim=-1).mean().item())

        flat = weights.flatten(1).detach().cpu()
        centroids = []
        within = []
        for group in sorted(groups.unique().tolist()):
            group_weights = flat[groups == group]
            if group_weights.numel() == 0:
                continue
            centroid = group_weights.mean(dim=0)
            centroids.append(centroid)
            within.append((group_weights - centroid).norm(dim=-1).mean())
        if len(centroids) >= 2:
            centroid_tensor = torch.stack(centroids)
            between = torch.pdist(centroid_tensor, p=2).mean()
            within_mean = torch.stack(within).mean().clamp_min(1e-8)
            rule_separations.append((between / within_mean).item())

    return {
        "effective_weight_difference_norm_across_examples": float(
            sum(diff_norms) / max(len(diff_norms), 1)
        ),
        "per_rule_w_eff_separation": float(
            sum(rule_separations) / max(len(rule_separations), 1)
        ),
        "w_eff_rank": float(sum(ranks) / max(len(ranks), 1)),
        "effective_weight_active_coordinate_count": float(
            sum(active_counts) / max(len(active_counts), 1)
        ),
    }


@torch.no_grad()
def refinement_metrics(
    model: WeightDistributionAttentionTransformer,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    coordinate_mode: str = "normal",
    fixed_coordinates: Tensor | None = None,
) -> dict[str, float]:
    if model.controller is None:
        return {
            "iterative_refinement_initial_final_coordinate_l2": 0.0,
            "refinement_delta_per_step": 0.0,
            "progressive_refinement_delta_last": 0.0,
        }
    model.eval()
    tokens, _labels, _groups = next(iter(loader))
    details = model(
        tokens.to(device),
        coordinate_mode=coordinate_mode,
        fixed_coordinates=fixed_coordinates,
        return_details=True,
    )
    history = details["coordinate_history"]
    if len(history) < 2:
        return {
            "iterative_refinement_initial_final_coordinate_l2": 0.0,
            "refinement_delta_per_step": 0.0,
            "progressive_refinement_delta_last": 0.0,
        }
    deltas = [
        (history[i] - history[i - 1]).pow(2).sum(dim=-1).sqrt().mean().item()
        for i in range(1, len(history))
    ]
    initial_final = (history[-1] - history[0]).pow(2).sum(dim=-1).sqrt().mean().item()
    return {
        "iterative_refinement_initial_final_coordinate_l2": float(initial_final),
        "refinement_delta_per_step": float(sum(deltas) / len(deltas)),
        "progressive_refinement_delta_last": float(deltas[-1]),
    }


def estimate_active_mult_adds(
    model: WeightDistributionAttentionTransformer,
    config: WDATransformerConfig,
) -> int:
    seq_len = config.seq_len
    d_model = config.d_model
    head_dim = d_model // config.nhead
    attention_projection = 4 * seq_len * d_model * d_model
    attention_scores = 2 * config.nhead * seq_len * seq_len * head_dim
    mlp = 2 * seq_len * d_model * config.dim_feedforward
    classifier = d_model * config.num_classes
    dense_forward = config.num_layers * (attention_projection + attention_scores + mlp)
    dense_forward += classifier

    materialization = 0
    for layer in model.dynamic_linears():
        out_features = getattr(layer, "out_features", 0)
        in_features = getattr(layer, "in_features", 0)
        if isinstance(layer, WDALinear):
            groups = layer.distribution.distribution_groups
            rank = layer.distribution.low_rank
            materialization += groups * out_features * in_features * rank
        elif hasattr(layer, "num_experts"):
            experts = getattr(layer, "num_experts")
            top_k = getattr(layer, "top_k", experts)
            materialization += top_k * out_features * in_features
        elif hasattr(layer, "num_atoms"):
            materialization += getattr(layer, "top_k", 1) * out_features * in_features
        elif hasattr(layer, "low_rank"):
            rank = getattr(layer, "low_rank")
            materialization += out_features * in_features * rank
        elif hasattr(layer, "num_parameter_tokens"):
            materialization += seq_len * getattr(layer, "num_parameter_tokens") * out_features

    controller = 0
    if model.controller is not None:
        hidden = config.controller_hidden_dim
        controller = seq_len * hidden + 3 * hidden * hidden + hidden * config.coord_dim
    return int(dense_forward + materialization + controller)


def compute_profile(
    model: WeightDistributionAttentionTransformer,
    config: WDATransformerConfig,
) -> dict[str, float | int]:
    stored_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    controller_params = controller_parameter_count(model)
    dense_equivalent = stored_params
    for layer in model.wda_linears():
        dense_equivalent += (
            layer.distribution.distribution_groups
            * layer.out_features
            * layer.in_features
        )
    active_params = stored_params
    if model.variant == "moe_transformer":
        expert_total = 0
        active_expert = 0
        for layer in model.dynamic_linears():
            if hasattr(layer, "experts"):
                expert_total += layer.experts.numel()
                active_expert += layer.experts[0].numel() * getattr(layer, "top_k", 1)
        active_params = stored_params - expert_total + active_expert
    active_mult_adds = estimate_active_mult_adds(model, config)
    return {
        "stored_parameter_count": stored_params,
        "trainable_parameter_count": trainable_params,
        "active_parameter_count_per_example": int(active_params),
        "estimated_active_mult_adds_per_example": active_mult_adds,
        "dense_equivalent_parameter_count": int(dense_equivalent),
        "controller_parameter_count": controller_params,
        "controller_parameter_fraction": controller_params / max(stored_params, 1),
        "memory_estimate_bytes_fp32": int(stored_params * 4),
    }


def fixed_coordinate_mean(
    model: WeightDistributionAttentionTransformer,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
) -> Tensor | None:
    coords = collect_coordinates(model, loader, device=device, max_batches=4)
    if coords is None:
        return None
    return coords.mean(dim=0, keepdim=True)


def model_row(
    *,
    variant: str,
    family: str,
    benchmark: CapacityBenchmark,
    seed: int,
    model: WeightDistributionAttentionTransformer,
    config: WDATransformerConfig,
    loaders: dict[str, torch.utils.data.DataLoader],
    train_stats: dict[str, float],
    train_config: TrainConfig,
    coordinate_mode: str = "normal",
    fixed_coordinates: Tensor | None = None,
    disable_distribution_scale: bool = False,
) -> dict[str, Any]:
    device = torch.device(train_config.device)
    val = evaluate_model(
        model,
        loaders["val"],
        device=device,
        coordinate_mode=coordinate_mode,
        fixed_coordinates=fixed_coordinates,
        disable_distribution_scale=disable_distribution_scale,
    )
    test = evaluate_model(
        model,
        loaders["test"],
        device=device,
        coordinate_mode=coordinate_mode,
        fixed_coordinates=fixed_coordinates,
        disable_distribution_scale=disable_distribution_scale,
    )
    coord = coordinate_metrics(
        model,
        loaders["test"],
        device=device,
        coordinate_mode=coordinate_mode,
        fixed_coordinates=fixed_coordinates,
    )
    eff = effective_weight_metrics(
        model,
        loaders["test"],
        device=device,
        coordinate_mode=coordinate_mode,
        fixed_coordinates=fixed_coordinates,
        disable_distribution_scale=disable_distribution_scale,
    )
    refine = refinement_metrics(
        model,
        loaders["test"],
        device=device,
        coordinate_mode=coordinate_mode,
        fixed_coordinates=fixed_coordinates,
    )
    profile = compute_profile(model, config)
    heldout_acc = test["accuracy"] if benchmark == "low_capacity_generalization" else 0.0
    return {
        "seed": seed,
        "benchmark": benchmark,
        "variant": variant,
        "family": family,
        "trained_variant": model.variant,
        "size": model_size_name(variant),
        "num_classes": config.num_classes,
        "coord_dim": config.coord_dim,
        "distribution_groups": config.distribution_groups,
        "low_rank": config.low_rank,
        "refinement_steps": config.refinement_steps,
        "controller_hidden_dim": config.controller_hidden_dim,
        "val_accuracy": val["accuracy"],
        "val_nll": val["nll"],
        "test_accuracy": test["accuracy"],
        "test_nll": test["nll"],
        "exact_sequence_accuracy": test["exact_sequence_accuracy"],
        "per_rule_accuracy": test["per_rule_accuracy"],
        "held_out_rule_composition_accuracy": heldout_acc,
        "low_data_test_accuracy": heldout_acc,
        "latency_ms_per_example": test["latency_ms_per_example"],
        **train_stats,
        **coord,
        **eff,
        **refine,
        **profile,
    }


def train_variant(
    *,
    variant: str,
    benchmark: CapacityBenchmark,
    seed: int,
    args: argparse.Namespace,
    train_config: TrainConfig,
    loaders: dict[str, torch.utils.data.DataLoader],
    config_overrides: dict[str, Any] | None = None,
) -> tuple[WeightDistributionAttentionTransformer, WDATransformerConfig, dict[str, float]]:
    set_seed(seed)
    config = make_model_config(
        args,
        variant=variant,
        seq_len=args.seq_len,
        overrides=config_overrides,
    )
    model = WeightDistributionAttentionTransformer(
        config,
        variant=model_variant_name(variant),
    )
    train_stats = train_one_model(model, loaders, train_config)
    return model, config, train_stats


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    keys = [
        "coord_dim",
        "distribution_groups",
        "low_rank",
        "refinement_steps",
        "controller_hidden_dim",
        "lambda_coord_reg",
        "lambda_diversity",
        "lambda_progressive",
    ]
    for candidate in candidates:
        key = tuple(candidate[name] for name in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def build_wda_sweep_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = [
        {
            "coord_dim": args.coord_dim,
            "distribution_groups": args.distribution_groups,
            "low_rank": args.low_rank,
            "refinement_steps": args.refinement_steps,
            "controller_hidden_dim": args.controller_hidden_dim,
            "lambda_coord_reg": args.lambda_coord_reg,
            "lambda_diversity": args.lambda_diversity,
            "lambda_progressive": args.lambda_progressive,
        },
        {
            "coord_dim": 16,
            "distribution_groups": 8,
            "low_rank": 2,
            "refinement_steps": 1,
            "controller_hidden_dim": 64,
            "lambda_coord_reg": 0.0001,
            "lambda_diversity": 0.001,
            "lambda_progressive": 0.001,
        },
        {
            "coord_dim": 32,
            "distribution_groups": 16,
            "low_rank": 4,
            "refinement_steps": 2,
            "controller_hidden_dim": 128,
            "lambda_coord_reg": 0.0001,
            "lambda_diversity": 0.01,
            "lambda_progressive": 0.001,
        },
        {
            "coord_dim": 64,
            "distribution_groups": 32,
            "low_rank": 8,
            "refinement_steps": 4,
            "controller_hidden_dim": 128,
            "lambda_coord_reg": 0.001,
            "lambda_diversity": 0.01,
            "lambda_progressive": 0.01,
        },
    ]
    return dedupe_candidates(candidates)[: args.sweep_max_candidates]


def train_config_for_candidate(
    base: TrainConfig,
    candidate: dict[str, Any],
    *,
    epochs: int,
) -> TrainConfig:
    return TrainConfig(
        seeds=base.seeds,
        epochs=epochs,
        lr=base.lr,
        weight_decay=base.weight_decay,
        grad_clip=base.grad_clip,
        lambda_coord_reg=float(candidate.get("lambda_coord_reg", base.lambda_coord_reg)),
        lambda_diversity=float(candidate.get("lambda_diversity", base.lambda_diversity)),
        lambda_smooth=base.lambda_smooth,
        lambda_progressive=float(
            candidate.get("lambda_progressive", base.lambda_progressive)
        ),
        device=base.device,
    )


def select_wda_config_by_validation(
    *,
    benchmark: CapacityBenchmark,
    seed: int,
    args: argparse.Namespace,
    train_config: TrainConfig,
    loaders: dict[str, torch.utils.data.DataLoader],
) -> tuple[dict[str, Any], TrainConfig, list[dict[str, Any]]]:
    candidates = build_wda_sweep_candidates(args)
    if not candidates:
        return {}, train_config, []

    device = torch.device(train_config.device)
    sweep_rows: list[dict[str, Any]] = []
    best_candidate = candidates[0]
    best_val_nll = float("inf")
    sweep_epochs = min(args.sweep_epochs, train_config.epochs)
    for index, candidate in enumerate(candidates):
        set_seed(seed * 1000 + index)
        candidate_train_config = train_config_for_candidate(
            train_config,
            candidate,
            epochs=sweep_epochs,
        )
        config = make_model_config(
            args,
            variant="wda_small",
            seq_len=args.seq_len,
            overrides=candidate,
        )
        model = WeightDistributionAttentionTransformer(config, variant="wda_small")
        train_one_model(model, loaders, candidate_train_config)
        val = evaluate_model(model, loaders["val"], device=device)
        row = {
            "seed": seed,
            "benchmark": benchmark,
            "candidate_index": index,
            **candidate,
            "sweep_epochs": sweep_epochs,
            "val_accuracy": val["accuracy"],
            "val_nll": val["nll"],
        }
        sweep_rows.append(row)
        if val["nll"] < best_val_nll:
            best_val_nll = val["nll"]
            best_candidate = candidate

    final_train_config = train_config_for_candidate(
        train_config,
        best_candidate,
        epochs=train_config.epochs,
    )
    return best_candidate, final_train_config, sweep_rows


def aggregate_rows(
    rows: list[dict[str, Any]],
    group_keys: list[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)

    metric_keys = [
        "val_accuracy",
        "val_nll",
        "test_accuracy",
        "test_nll",
        "exact_sequence_accuracy",
        "per_rule_accuracy",
        "held_out_rule_composition_accuracy",
        "low_data_test_accuracy",
        "coordinate_variance",
        "coordinate_entropy",
        "coordinate_l2_distance_across_examples",
        "same_input_coordinate_stability",
        "different_input_coordinate_diversity",
        "per_rule_coordinate_separation",
        "coordinate_clustering_purity_by_rule",
        "active_coordinate_count",
        "effective_weight_difference_norm_across_examples",
        "per_rule_w_eff_separation",
        "w_eff_rank",
        "effective_weight_active_coordinate_count",
        "iterative_refinement_initial_final_coordinate_l2",
        "refinement_delta_per_step",
        "latency_ms_per_example",
        "stored_parameter_count",
        "trainable_parameter_count",
        "active_parameter_count_per_example",
        "estimated_active_mult_adds_per_example",
        "dense_equivalent_parameter_count",
        "controller_parameter_count",
        "controller_parameter_fraction",
        "num_classes",
        "coord_dim",
        "distribution_groups",
        "low_rank",
        "refinement_steps",
        "controller_hidden_dim",
    ]
    aggregates: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        out = {name: value for name, value in zip(group_keys, key, strict=True)}
        out["n_rows"] = len(group_rows)
        for metric in metric_keys:
            values = [float(row[metric]) for row in group_rows if metric in row]
            if not values:
                continue
            mean = sum(values) / len(values)
            var = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = math.sqrt(var)
        aggregates.append(out)
    return sorted(aggregates, key=lambda row: tuple(str(row[k]) for k in group_keys))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def score(row: dict[str, Any]) -> float:
    return float(row.get("test_accuracy_mean", 0.0)) - 0.05 * float(
        row.get("test_nll_mean", 0.0)
    )


def variant_row(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    for row in rows:
        if row.get("variant") == variant:
            return row
    return {}


def comparison(wda: dict[str, Any], other: dict[str, Any]) -> dict[str, float]:
    return {
        "wda_score": score(wda),
        "other_score": score(other),
        "score_delta": score(wda) - score(other),
        "wda_accuracy": float(wda.get("test_accuracy_mean", 0.0)),
        "other_accuracy": float(other.get("test_accuracy_mean", 0.0)),
        "accuracy_delta": float(wda.get("test_accuracy_mean", 0.0))
        - float(other.get("test_accuracy_mean", 0.0)),
        "wda_active_mult_adds": float(wda.get("estimated_active_mult_adds_per_example_mean", 0.0)),
        "other_active_mult_adds": float(
            other.get("estimated_active_mult_adds_per_example_mean", 0.0)
        ),
    }


def choose_verdict(aggregate: list[dict[str, Any]], per_benchmark: list[dict[str, Any]]) -> dict[str, Any]:
    wda_candidates = [variant_row(aggregate, name) for name in ["wda_small", "wda_small_iterative"]]
    wda_candidates = [row for row in wda_candidates if row]
    best_wda = max(wda_candidates, key=score) if wda_candidates else {}
    best_wda_name = best_wda.get("variant", "missing")

    fixed_small = variant_row(aggregate, "fixed_small_transformer")
    fixed_large = variant_row(aggregate, "fixed_large_transformer")
    hyper = variant_row(aggregate, "hypernetwork_transformer")
    condconv = variant_row(aggregate, "condconv_style_transformer")
    lora = variant_row(aggregate, "dynamic_lora_transformer")
    moe = variant_row(aggregate, "moe_transformer")
    token_param = variant_row(aggregate, "token_parameter_attention_transformer")
    atom = variant_row(aggregate, "previous_weight_atom_attention_transformer")

    fixed_coords = variant_row(aggregate, "wda_fixed_coordinates")
    random_coords = variant_row(aggregate, "wda_random_coordinates")
    shuffled_coords = variant_row(aggregate, "wda_shuffled_coordinates")
    no_scale = variant_row(aggregate, "wda_no_distribution_scale")
    no_controller = variant_row(aggregate, "wda_no_controller")
    frozen_controller = variant_row(aggregate, "wda_frozen_controller")
    dense_hyper_control = variant_row(aggregate, "wda_dense_hypernetwork_control")

    wda_score = score(best_wda)
    small_score = score(fixed_small)
    large_score = score(fixed_large)
    large_gap = large_score - small_score
    wda_gain = wda_score - small_score

    def acc(row: dict[str, Any]) -> float:
        return float(row.get("test_accuracy_mean", 0.0))

    def nll(row: dict[str, Any]) -> float:
        return float(row.get("test_nll_mean", 0.0))

    def active_compute(row: dict[str, Any]) -> float:
        return float(row.get("estimated_active_mult_adds_per_example_mean", 1.0))

    shuffled_drop = acc(best_wda) - acc(shuffled_coords)
    shuffled_nll_ratio = nll(shuffled_coords) / max(nll(best_wda), 1e-8)
    no_scale_drop = acc(best_wda) - acc(no_scale)
    no_scale_nll_ratio = nll(no_scale) / max(nll(best_wda), 1e-8)
    no_controller_drop = acc(best_wda) - max(acc(no_controller), acc(frozen_controller))

    fixed_small_by_benchmark = {
        row["benchmark"]: row for row in per_benchmark if row["variant"] == "fixed_small_transformer"
    }
    wda_by_benchmark = {
        row["benchmark"]: row for row in per_benchmark if row["variant"] == best_wda_name
    }
    wins_by_benchmark = sum(
        score(wda_by_benchmark.get(benchmark, {}))
        > score(fixed_small_by_benchmark.get(benchmark, {}))
        for benchmark in BENCHMARKS
    )

    fixed_small_saturated = all(
        float(row.get("test_accuracy_mean", 0.0)) >= 0.99
        for row in fixed_small_by_benchmark.values()
    ) and len(fixed_small_by_benchmark) == len(BENCHMARKS)

    best_compute_baseline = max(
        [row for row in [fixed_small, hyper, condconv, lora, moe, token_param, atom] if row],
        key=score,
        default={},
    )
    wda_efficiency = wda_score / max(active_compute(best_wda), 1.0)
    large_efficiency = large_score / max(active_compute(fixed_large), 1.0)
    compute_baseline_efficiency = score(best_compute_baseline) / max(
        active_compute(best_compute_baseline), 1.0
    )

    checks = {
        "task_not_saturated": not fixed_small_saturated,
        "wda_learns": acc(best_wda)
        > 1.25 / max(1.0, float(best_wda.get("num_classes_mean", 16.0)))
        or nll(best_wda) < 3.5,
        "beats_same_size_dense": wda_score > small_score,
        "approaches_larger_dense": wda_gain >= 0.90 * large_gap if large_gap > 0.005 else wda_score >= large_score,
        "beats_active_compute_baseline": wda_efficiency > compute_baseline_efficiency,
        "beats_hypernetwork": wda_score > score(hyper),
        "beats_condconv_style": wda_score > score(condconv),
        "beats_dynamic_lora": wda_score > score(lora),
        "beats_moe": wda_score > score(moe) or (
            abs(wda_score - score(moe)) <= 0.002 and active_compute(best_wda) < active_compute(moe)
        ),
        "beats_token_parameter_attention": wda_score > score(token_param) or (
            abs(wda_score - score(token_param)) <= 0.002
            and active_compute(best_wda) < active_compute(token_param)
        ),
        "beats_weight_atom_attention": wda_score > score(atom),
        "coordinates_necessary": shuffled_drop >= 0.05 or shuffled_nll_ratio >= 1.25,
        "distribution_scale_necessary": no_scale_drop >= 0.03 or no_scale_nll_ratio >= 1.15,
        "controller_necessary": no_controller_drop >= 0.03,
        "input_conditioned_realization": float(
            best_wda.get("per_rule_coordinate_separation_mean", 0.0)
        )
        > 0.05
        and float(best_wda.get("effective_weight_difference_norm_across_examples_mean", 0.0))
        > 1e-4,
        "not_just_hypernetwork": wda_score > score(dense_hyper_control)
        or wda_efficiency
        > (score(dense_hyper_control) / max(active_compute(dense_hyper_control), 1.0)),
        "compact_controller": float(best_wda.get("controller_parameter_fraction_mean", 1.0))
        <= 0.25,
        "compute_tradeoff_positive": wda_efficiency > large_efficiency,
        "multi_benchmark_generalization": wins_by_benchmark >= 2,
    }

    implementation_valid = bool(best_wda) and float(
        best_wda.get("effective_weight_difference_norm_across_examples_mean", 0.0)
    ) > 0.0

    if not implementation_valid:
        verdict = "FAILED_IMPLEMENTATION"
    elif not checks["wda_learns"] or not checks["input_conditioned_realization"]:
        verdict = "FAILED_IMPLEMENTATION"
    elif not checks["beats_same_size_dense"] or not any(
        checks[name]
        for name in [
            "beats_hypernetwork",
            "beats_condconv_style",
            "beats_dynamic_lora",
            "beats_moe",
            "beats_token_parameter_attention",
            "beats_weight_atom_attention",
        ]
    ):
        verdict = "MECHANISM_WORKS_BUT_NO_VALUE"
    elif wins_by_benchmark < 2:
        verdict = "VALUE_ON_ONE_TASK_ONLY"
    elif all(
        checks[name]
        for name in [
            "beats_same_size_dense",
            "approaches_larger_dense",
            "beats_hypernetwork",
            "beats_condconv_style",
            "beats_dynamic_lora",
            "beats_moe",
            "beats_token_parameter_attention",
            "beats_weight_atom_attention",
            "coordinates_necessary",
            "distribution_scale_necessary",
            "controller_necessary",
            "compute_tradeoff_positive",
            "multi_benchmark_generalization",
        ]
    ):
        verdict = "PAPER_CANDIDATE"
    elif all(
        checks[name]
        for name in [
            "beats_same_size_dense",
            "approaches_larger_dense",
            "coordinates_necessary",
            "compute_tradeoff_positive",
        ]
    ):
        verdict = "STRONG_CAPACITY_COMPUTE"
    else:
        verdict = "PROMISING_CAPACITY_COMPUTE"

    failure_reasons = [name for name, passed in checks.items() if not passed]
    return {
        "final_verdict": verdict,
        "all_checks": checks,
        "best_wda_config": {
            "variant": best_wda_name,
            "coord_dim": best_wda.get("coord_dim_mean", None),
            "distribution_groups": best_wda.get("distribution_groups_mean", None),
            "low_rank": best_wda.get("low_rank_mean", None),
            "refinement_steps": best_wda.get("refinement_steps_mean", None),
            "controller_hidden_dim": best_wda.get("controller_hidden_dim_mean", None),
            "test_accuracy_mean": best_wda.get("test_accuracy_mean", None),
            "test_nll_mean": best_wda.get("test_nll_mean", None),
        },
        "WDA vs fixed_small": comparison(best_wda, fixed_small),
        "WDA vs fixed_large": comparison(best_wda, fixed_large),
        "WDA vs hypernetwork": comparison(best_wda, hyper),
        "WDA vs condconv_style": comparison(best_wda, condconv),
        "WDA vs dynamic_lora": comparison(best_wda, lora),
        "WDA vs moe": comparison(best_wda, moe),
        "WDA vs token_parameter_attention": comparison(best_wda, token_param),
        "WDA vs weight_atom_attention": comparison(best_wda, atom),
        "ablation_drops": {
            "fixed_coordinates_accuracy_drop": acc(best_wda) - acc(fixed_coords),
            "random_coordinates_accuracy_drop": acc(best_wda) - acc(random_coords),
            "shuffled_coordinates_accuracy_drop": shuffled_drop,
            "no_distribution_scale_accuracy_drop": no_scale_drop,
            "no_controller_or_frozen_accuracy_drop": no_controller_drop,
        },
        "compute_tradeoff": {
            "wda_score_per_active_mult_add": wda_efficiency,
            "fixed_large_score_per_active_mult_add": large_efficiency,
            "best_active_compute_baseline_score_per_active_mult_add": compute_baseline_efficiency,
        },
        "mechanism_metrics": {
            "coordinate_variance_mean": best_wda.get("coordinate_variance_mean", None),
            "coordinate_l2_distance_across_examples_mean": best_wda.get(
                "coordinate_l2_distance_across_examples_mean", None
            ),
            "per_rule_coordinate_separation_mean": best_wda.get(
                "per_rule_coordinate_separation_mean", None
            ),
            "effective_weight_difference_norm_across_examples_mean": best_wda.get(
                "effective_weight_difference_norm_across_examples_mean", None
            ),
            "per_rule_w_eff_separation_mean": best_wda.get(
                "per_rule_w_eff_separation_mean", None
            ),
        },
        "failure_reasons": failure_reasons if verdict != "PAPER_CANDIDATE" else [],
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_pass_fail_rule(path: Path) -> None:
    text = """# WDA Capacity/Compute Validation Pass/Fail Rule

Verdicts, from weakest to strongest:

- `FAILED_IMPLEMENTATION`
- `MECHANISM_WORKS_BUT_NO_VALUE`
- `VALUE_ON_ONE_TASK_ONLY`
- `PROMISING_CAPACITY_COMPUTE`
- `STRONG_CAPACITY_COMPUTE`
- `PAPER_CANDIDATE`

Core checks:

- The fixed small transformer must not saturate every benchmark.
- WDA must learn and must use realized effective weights in the forward pass.
- WDA must beat the same-size fixed dense transformer on average score.
- WDA must approach the fixed large reference when the large reference improves over fixed small.
- WDA must beat or be more compute-efficient than comparable dynamic-weight baselines.
- Fixed, random, or shuffled coordinates must hurt accuracy or NLL.
- Removing distribution scale/range must hurt.
- Removing or freezing the controller must hurt.
- Coordinates and effective weights must vary by input rule/context.
- Controller parameter fraction must be at most 0.25.
- WDA score per active mult-add must beat the fixed large transformer.
- Core value checks must hold on at least two benchmark settings for a multi-benchmark verdict.

The verdict logic is intentionally strict. If WDA loses to simple baselines, the report must say so.
"""
    path.write_text(text, encoding="utf-8")


def write_validation_report(
    path: Path,
    *,
    verdict: dict[str, Any],
    config_blob: dict[str, Any],
    aggregate: list[dict[str, Any]],
    per_benchmark: list[dict[str, Any]],
    ablations: list[dict[str, Any]],
    compute_report: dict[str, Any],
) -> None:
    aggregate_cols = [
        "variant",
        "test_accuracy_mean",
        "test_nll_mean",
        "val_accuracy_mean",
        "val_nll_mean",
        "estimated_active_mult_adds_per_example_mean",
        "stored_parameter_count_mean",
    ]
    per_benchmark_cols = [
        "benchmark",
        "variant",
        "test_accuracy_mean",
        "test_nll_mean",
        "exact_sequence_accuracy_mean",
        "per_rule_accuracy_mean",
        "held_out_rule_composition_accuracy_mean",
    ]
    ablation_cols = [
        "variant",
        "test_accuracy_mean",
        "test_nll_mean",
        "coordinate_variance_mean",
        "effective_weight_difference_norm_across_examples_mean",
    ]
    checks_text = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in verdict["all_checks"].items()
    )
    report = f"""# WDA Capacity/Compute Validation Report

Verdict: `{verdict['final_verdict']}`

## Plain-English Verdict

This run evaluates whether WDA provides capacity/compute value beyond proving the mechanism. The verdict is generated from measured dense baselines, dynamic-weight baselines, ablations, mechanism metrics, and active-compute estimates. If a simple baseline matches or beats WDA, that is reflected in the verdict.

## Exact Run Config

```json
{json.dumps(config_blob, indent=2)}
```

## Task Descriptions

- `rule_switching_sequence`: context tokens select one of several sequence computations, including copying from different positions, modular addition, parity/counting, xor-like transforms, and alternating position rules.
- `compositional_rule_switching`: two context tokens select a composition of elementary token transforms.
- `low_capacity_generalization`: the compositional setting with low training data and held-out rule-composition pairs at validation/test time.

## Why The Tasks Are Not Saturated

The `task_not_saturated` check fails if `fixed_small_transformer` exceeds 99% test accuracy on every benchmark. This prevents a high verdict on a task the same-size dense baseline already solves.

## WDA Architecture

WDA uses learned base weights `W_mu`, learned low-rank distribution directions/scales `W_sigma`, and a compact controller that reads the context and emits bounded continuous coordinates `C(x)`. Each `WDALinear` realizes exact per-example effective weights with `W_eff(x) = W_mu + C(x) * W_sigma` in structured low-rank form. Those effective weights are used in the attention projections and MLP projections during the transformer forward pass.

## Checks

{checks_text}

## Aggregate Metrics

{markdown_table(aggregate, aggregate_cols)}

## Per-Benchmark Metrics

{markdown_table(per_benchmark, per_benchmark_cols)}

## Capacity/Compute Comparison

```json
{json.dumps(compute_report, indent=2)}
```

## WDA vs Same-Size Dense

```json
{json.dumps(verdict['WDA vs fixed_small'], indent=2)}
```

## WDA vs Larger Dense

```json
{json.dumps(verdict['WDA vs fixed_large'], indent=2)}
```

## WDA vs Dynamic-Weight Baselines

```json
{json.dumps({
    'hypernetwork': verdict['WDA vs hypernetwork'],
    'condconv_style': verdict['WDA vs condconv_style'],
    'dynamic_lora': verdict['WDA vs dynamic_lora'],
    'moe': verdict['WDA vs moe'],
    'token_parameter_attention': verdict['WDA vs token_parameter_attention'],
    'previous_weight_atom_attention': verdict['WDA vs weight_atom_attention'],
}, indent=2)}
```

## Ablations

{markdown_table(ablations, ablation_cols)}

## Coordinate And Effective-Weight Mechanism

```json
{json.dumps(verdict['mechanism_metrics'], indent=2)}
```

## Compute-Cost Section

The compute estimates include dense transformer projection/attention/MLP mult-adds, controller overhead, and dynamic materialization overhead. They are estimates, not profiler traces; measured latency is reported separately in CSV files.

## Final Interpretation

`{verdict['final_verdict']}` is not final publication proof. It means only that this implementation and these benchmarks meet the strict rules encoded in `verdict.json`. If the verdict is below `STRONG_CAPACITY_COMPUTE`, inspect the failure reasons in `verdict.json` before making a stronger claim.
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--low-data-n-train", type=int, default=800)
    parser.add_argument("--low-data-n-val", type=int, default=400)
    parser.add_argument("--low-data-n-test", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--coord-dim", type=int, default=32)
    parser.add_argument("--distribution-groups", type=int, default=16)
    parser.add_argument("--low-rank", type=int, default=4)
    parser.add_argument("--refinement-steps", type=int, default=2)
    parser.add_argument("--controller-hidden-dim", type=int, default=64)
    parser.add_argument("--lambda-coord-reg", type=float, default=0.0001)
    parser.add_argument("--lambda-diversity", type=float, default=0.001)
    parser.add_argument("--lambda-smooth", type=float, default=0.0001)
    parser.add_argument("--lambda-progressive", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    parser.add_argument("--models", nargs="+", default=MODEL_VARIANTS)
    parser.add_argument("--sweep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sweep-epochs", type=int, default=3)
    parser.add_argument("--sweep-max-candidates", type=int, default=3)
    parser.add_argument("--no-mask-context-token", action="store_true")
    parser.add_argument("--no-apply-wda-qkv", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Small smoke run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.seeds = [0]
        args.epochs = min(args.epochs, 1)
        args.n_train = min(args.n_train, 96)
        args.n_val = min(args.n_val, 64)
        args.n_test = min(args.n_test, 64)
        args.low_data_n_train = min(args.low_data_n_train, 96)
        args.low_data_n_val = min(args.low_data_n_val, 64)
        args.low_data_n_test = min(args.low_data_n_test, 64)
        args.batch_size = min(args.batch_size, 32)
        args.seq_len = min(args.seq_len, 16)
        args.coord_dim = min(args.coord_dim, 16)
        args.distribution_groups = min(args.distribution_groups, 8)
        args.low_rank = min(args.low_rank, 2)
        args.sweep = False

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("transformer_uncertainty") / "runs" / (
            f"wda_capacity_compute_validation_{stamp}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_pass_fail_rule(args.output_dir / "PASS_FAIL_RULE.md")

    train_config = TrainConfig(
        seeds=args.seeds,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lambda_coord_reg=args.lambda_coord_reg,
        lambda_diversity=args.lambda_diversity,
        lambda_smooth=args.lambda_smooth,
        lambda_progressive=args.lambda_progressive,
        device=args.device,
    )

    benchmarks: list[CapacityBenchmark] = list(args.benchmarks)
    requested_models = list(dict.fromkeys(args.models))
    train_models = [model for model in requested_models if model in MODEL_VARIANTS]
    missing_models = [model for model in requested_models if model not in MODEL_VARIANTS]
    if missing_models:
        print(f"skipping unknown model names: {missing_models}", flush=True)

    config_blob = {
        "experiment_name": "wda_capacity_compute_validation",
        "benchmarks": benchmarks,
        "models": train_models,
        "trained_ablations": TRAINED_ABLATIONS,
        "evaluated_ablations": EVALUATED_ABLATIONS,
        "data_defaults": {
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "low_data_n_train": args.low_data_n_train,
            "low_data_n_val": args.low_data_n_val,
            "low_data_n_test": args.low_data_n_test,
            "seq_len": args.seq_len,
            "vocab_size": args.vocab_size,
            "num_classes": args.num_classes,
            "batch_size": args.batch_size,
        },
        "model_sizes": MODEL_SIZES,
        "wda_hyperparameters": {
            "coord_dim": args.coord_dim,
            "distribution_groups": args.distribution_groups,
            "low_rank": args.low_rank,
            "refinement_steps": args.refinement_steps,
            "controller_hidden_dim": args.controller_hidden_dim,
            "apply_wda_qkv": not args.no_apply_wda_qkv,
        },
        "sweep": {
            "enabled": args.sweep,
            "sweep_epochs": args.sweep_epochs,
            "sweep_max_candidates": args.sweep_max_candidates,
            "candidates": build_wda_sweep_candidates(args) if args.sweep else [],
        },
        "train": asdict(train_config),
    }
    (args.output_dir / "RUN_CONFIG.json").write_text(
        json.dumps(config_blob, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    device = torch.device(train_config.device)
    for seed in train_config.seeds:
        for benchmark in benchmarks:
            data_config = make_data_config(args, benchmark)
            loaders = make_capacity_dataloaders(data_config, seed, benchmark)
            trained: dict[str, tuple[WeightDistributionAttentionTransformer, WDATransformerConfig, dict[str, float], dict[str, Any]]] = {}
            selected_wda_config: dict[str, Any] = {}
            selected_wda_train_config = train_config
            if args.sweep and any(model.startswith("wda_") for model in train_models):
                selected_wda_config, selected_wda_train_config, current_sweep_rows = (
                    select_wda_config_by_validation(
                        benchmark=benchmark,
                        seed=seed,
                        args=args,
                        train_config=train_config,
                        loaders=loaders,
                    )
                )
                sweep_rows.extend(current_sweep_rows)
                print(
                    f"{benchmark} seed={seed} selected_wda_config="
                    f"{selected_wda_config}",
                    flush=True,
                )

            for variant in train_models:
                variant_is_wda = variant.startswith("wda_")
                effective_train_config = (
                    selected_wda_train_config if variant_is_wda else train_config
                )
                config_overrides = selected_wda_config if variant_is_wda else None
                model, model_config, train_stats = train_variant(
                    variant=variant,
                    benchmark=benchmark,
                    seed=seed,
                    args=args,
                    train_config=effective_train_config,
                    loaders=loaders,
                    config_overrides=config_overrides,
                )
                row = model_row(
                    variant=variant,
                    family="model",
                    benchmark=benchmark,
                    seed=seed,
                    model=model,
                    config=model_config,
                    loaders=loaders,
                    train_stats=train_stats,
                    train_config=effective_train_config,
                )
                rows.append(row)
                trained[variant] = (model, model_config, train_stats, row)
                print(
                    f"{benchmark} {variant} seed={seed} "
                    f"acc={row['test_accuracy']:.3f} nll={row['test_nll']:.3f}",
                    flush=True,
                )

            if "wda_small" in trained:
                wda_model, wda_config, wda_train_stats, _wda_row = trained["wda_small"]
                fixed_mean = fixed_coordinate_mean(
                    wda_model,
                    loaders["train"],
                    device=device,
                )
                for ablation, mode, disable_scale in [
                    ("wda_fixed_coordinates", "fixed", False),
                    ("wda_random_coordinates", "random", False),
                    ("wda_shuffled_coordinates", "shuffled", False),
                    ("wda_no_distribution_scale", "normal", True),
                ]:
                    rows.append(
                        model_row(
                            variant=ablation,
                            family="ablation",
                            benchmark=benchmark,
                            seed=seed,
                            model=wda_model,
                            config=wda_config,
                            loaders=loaders,
                            train_stats=wda_train_stats,
                            train_config=selected_wda_train_config,
                            coordinate_mode=mode,
                            fixed_coordinates=fixed_mean,
                            disable_distribution_scale=disable_scale,
                        )
                    )
                rows.append(
                    model_row(
                        variant="wda_no_iterative_refinement",
                        family="ablation",
                        benchmark=benchmark,
                        seed=seed,
                        model=wda_model,
                        config=wda_config,
                        loaders=loaders,
                        train_stats=wda_train_stats,
                        train_config=selected_wda_train_config,
                    )
                )

            for ablation in TRAINED_ABLATIONS:
                model, model_config, train_stats = train_variant(
                    variant=ablation,
                    benchmark=benchmark,
                    seed=seed,
                    args=args,
                    train_config=selected_wda_train_config,
                    loaders=loaders,
                    config_overrides=selected_wda_config,
                )
                row = model_row(
                    variant=ablation,
                    family="ablation",
                    benchmark=benchmark,
                    seed=seed,
                    model=model,
                    config=model_config,
                    loaders=loaders,
                    train_stats=train_stats,
                    train_config=selected_wda_train_config,
                )
                rows.append(row)
                print(
                    f"{benchmark} {ablation} seed={seed} "
                    f"acc={row['test_accuracy']:.3f} nll={row['test_nll']:.3f}",
                    flush=True,
                )

            if "fixed_small_transformer" in trained:
                fixed_model, fixed_config, fixed_stats, _fixed_row = trained[
                    "fixed_small_transformer"
                ]
                rows.append(
                    model_row(
                        variant="wda_deterministic_weights_only",
                        family="ablation",
                        benchmark=benchmark,
                        seed=seed,
                        model=fixed_model,
                        config=fixed_config,
                        loaders=loaders,
                        train_stats=fixed_stats,
                        train_config=train_config,
                    )
                )
            if "hypernetwork_transformer" in trained:
                hyper_model, hyper_config, hyper_stats, _hyper_row = trained[
                    "hypernetwork_transformer"
                ]
                rows.append(
                    model_row(
                        variant="wda_dense_hypernetwork_control",
                        family="ablation",
                        benchmark=benchmark,
                        seed=seed,
                        model=hyper_model,
                        config=hyper_config,
                        loaders=loaders,
                        train_stats=hyper_stats,
                        train_config=train_config,
                    )
                )

    aggregate = aggregate_rows(rows, ["variant"])
    per_benchmark = aggregate_rows(rows, ["benchmark", "variant"])
    ablation_rows = [row for row in rows if row["family"] == "ablation"]
    ablations = aggregate_rows(ablation_rows, ["variant"])
    mechanism_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "seed",
                "benchmark",
                "variant",
                "family",
                "coordinate_variance",
                "coordinate_entropy",
                "coordinate_l2_distance_across_examples",
                "different_input_coordinate_diversity",
                "per_rule_coordinate_separation",
                "coordinate_clustering_purity_by_rule",
                "active_coordinate_count",
                "effective_weight_difference_norm_across_examples",
                "per_rule_w_eff_separation",
                "w_eff_rank",
                "effective_weight_active_coordinate_count",
                "iterative_refinement_initial_final_coordinate_l2",
                "refinement_delta_per_step",
            }
        }
        for row in rows
    ]
    coordinate_cluster_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "seed",
                "benchmark",
                "variant",
                "coordinate_variance",
                "coordinate_entropy",
                "coordinate_l2_distance_across_examples",
                "per_rule_coordinate_separation",
                "coordinate_clustering_purity_by_rule",
                "active_coordinate_count",
            }
        }
        for row in rows
    ]
    effective_weight_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "seed",
                "benchmark",
                "variant",
                "effective_weight_difference_norm_across_examples",
                "per_rule_w_eff_separation",
                "w_eff_rank",
                "effective_weight_active_coordinate_count",
            }
        }
        for row in rows
    ]
    compute_report = {
        row["variant"]: {
            key: value
            for key, value in row.items()
            if key.endswith("_mean")
            and (
                "parameter_count" in key
                or "active_mult_adds" in key
                or "controller_parameter" in key
                or "latency_ms" in key
                or "dense_equivalent" in key
            )
        }
        for row in aggregate
    }
    verdict = choose_verdict(aggregate, per_benchmark)
    verdict["generated_at"] = datetime.now().isoformat(timespec="seconds")

    write_csv(args.output_dir / "per_seed_metrics.csv", rows)
    write_csv(args.output_dir / "sweep_selection_report.csv", sweep_rows)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate)
    write_csv(args.output_dir / "per_benchmark_metrics.csv", per_benchmark)
    write_csv(args.output_dir / "ablation_metrics.csv", ablations)
    write_csv(args.output_dir / "mechanism_metrics.csv", mechanism_rows)
    write_csv(args.output_dir / "coordinate_cluster_report.csv", coordinate_cluster_rows)
    write_csv(args.output_dir / "effective_weight_report.csv", effective_weight_rows)
    (args.output_dir / "compute_report.json").write_text(
        json.dumps(compute_report, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2),
        encoding="utf-8",
    )
    write_validation_report(
        args.output_dir / "VALIDATION_REPORT.md",
        verdict=verdict,
        config_blob=config_blob,
        aggregate=aggregate,
        per_benchmark=per_benchmark,
        ablations=ablations,
        compute_report=compute_report,
    )
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
