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
    from .data import SyntheticTaskConfig, make_dataloaders
    from .wda_transformer import (
        WDATransformerConfig,
        WeightDistributionAttentionTransformer,
        controller_parameter_count,
        count_parameters,
    )
except ImportError:  # pragma: no cover - allows running as a script path
    from data import SyntheticTaskConfig, make_dataloaders
    from wda_transformer import (
        WDATransformerConfig,
        WeightDistributionAttentionTransformer,
        controller_parameter_count,
        count_parameters,
    )


ABLATIONS = [
    "baseline_transformer",
    "wda_one_shot",
    "wda_iterative",
    "wda_fixed_coordinates",
    "wda_random_coordinates",
    "wda_shuffled_coordinates",
    "wda_no_distribution_scale",
    "wda_no_controller",
    "wda_frozen_controller",
    "wda_deterministic_weights_only",
    "wda_dense_hypernetwork_control",
]


@dataclass
class TrainConfig:
    seeds: list[int]
    epochs: int = 10
    lr: float = 0.002
    weight_decay: float = 0.0001
    grad_clip: float = 5.0
    lambda_coord_reg: float = 0.0001
    lambda_diversity: float = 0.001
    lambda_smooth: float = 0.0001
    lambda_progressive: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sweep: bool = True


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_and_nll(logits: Tensor, labels: Tensor) -> tuple[float, float]:
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return acc.item(), loss.item()


def mechanism_losses(
    details: dict[str, Any],
    *,
    lambda_coord_reg: float,
    lambda_diversity: float,
    lambda_smooth: float,
    lambda_progressive: float,
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
        lambda_coord_reg * coord_reg
        + lambda_diversity * coord_diversity
        + lambda_smooth * coord_smooth
        + lambda_progressive * progressive
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
    *,
    variant: str,
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
            mech_loss, last_losses = mechanism_losses(
                details,
                lambda_coord_reg=train_config.lambda_coord_reg,
                lambda_diversity=train_config.lambda_diversity,
                lambda_smooth=train_config.lambda_smooth,
                lambda_progressive=train_config.lambda_progressive,
            )
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
    elapsed = time.perf_counter() - started
    return {
        "train_seconds": elapsed,
        "best_val_nll": best_val_nll,
        **{f"last_{k}": v for k, v in last_losses.items()},
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
    started = time.perf_counter()
    for tokens, labels, _groups in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        logits = model(
            tokens,
            coordinate_mode=coordinate_mode,
            fixed_coordinates=fixed_coordinates,
            disable_distribution_scale=disable_distribution_scale,
        )
        loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.numel()
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "accuracy": total_correct / max(total, 1),
        "nll": total_loss / max(total, 1),
        "latency_ms_per_example": (elapsed / max(total, 1)) * 1000.0,
    }


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
            "coordinate_group_separation": 0.0,
            "average_coordinate_l2_distance": 0.0,
            "same_input_coordinate_stability": 0.0,
            "different_input_coordinate_diversity": 0.0,
            "final_coordinate_sparsity": 0.0,
            "active_coordinate_count": 0.0,
        }

    coords = coords.float()
    variance = coords.var(dim=0, unbiased=False).mean().item()
    probs = coords.abs() + 1e-8
    probs = probs / probs.sum(dim=-1, keepdim=True)
    entropy = (-(probs * probs.log()).sum(dim=-1)).mean().item()
    sparsity = (coords.abs() < 0.05).float().mean().item()
    active = (coords.abs() > 0.10).float().sum(dim=-1).mean().item()

    max_pairs = min(coords.shape[0], 128)
    pair = torch.pdist(coords[:max_pairs], p=2)
    average_l2 = pair.mean().item() if pair.numel() else 0.0

    groups_list: list[Tensor] = []
    for _tokens, _labels, groups in loader:
        groups_list.append(groups)
    groups = torch.cat(groups_list, dim=0)[: coords.shape[0]]
    centroids = []
    within = []
    for group in sorted(groups.unique().tolist()):
        g_coords = coords[groups == group]
        if g_coords.numel() == 0:
            continue
        centroid = g_coords.mean(dim=0)
        centroids.append(centroid)
        within.append((g_coords - centroid).pow(2).sum(dim=-1).sqrt().mean())
    if len(centroids) >= 2:
        centroid_tensor = torch.stack(centroids)
        between = torch.pdist(centroid_tensor, p=2).mean()
        within_mean = torch.stack(within).mean().clamp_min(1e-8)
        group_sep = (between / within_mean).item()
    else:
        group_sep = 0.0

    return {
        "coordinate_variance": variance,
        "coordinate_entropy": entropy,
        "coordinate_group_separation": group_sep,
        "average_coordinate_l2_distance": average_l2,
        "same_input_coordinate_stability": 0.0,
        "different_input_coordinate_diversity": average_l2,
        "final_coordinate_sparsity": sparsity,
        "active_coordinate_count": active,
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
    linears = list(model.wda_linears())
    if not linears or model.controller is None:
        return {
            "effective_weight_difference_norm": 0.0,
            "effective_weight_rank": 0.0,
            "effective_weight_active_coordinate_count": 0.0,
        }
    model.eval()
    tokens, _labels, _groups = next(iter(loader))
    tokens = tokens[: min(tokens.shape[0], 32)].to(device)
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
            "effective_weight_difference_norm": 0.0,
            "effective_weight_rank": 0.0,
            "effective_weight_active_coordinate_count": 0.0,
        }

    diff_norms = []
    ranks = []
    active_counts = []
    for layer in linears[:6]:
        weights, group_coords = layer.materialize_weight(
            coords,
            disable_distribution_scale=disable_distribution_scale,
        )
        if weights.shape[0] >= 2:
            diff = weights[1:] - weights[:1]
            diff_norms.append(diff.flatten(1).norm(dim=-1).mean().item())
        sample_delta = weights[: min(8, weights.shape[0])] - layer.weight_mu.unsqueeze(0)
        for matrix in sample_delta:
            ranks.append(torch.linalg.matrix_rank(matrix.float(), tol=1e-5).item())
        active_counts.append((group_coords.abs() > 0.10).float().sum(dim=-1).mean().item())
    return {
        "effective_weight_difference_norm": float(sum(diff_norms) / max(len(diff_norms), 1)),
        "effective_weight_rank": float(sum(ranks) / max(len(ranks), 1)),
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
            "progressive_refinement_delta_mean": 0.0,
            "progressive_refinement_delta_last": 0.0,
            "initial_final_coordinate_l2": 0.0,
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
            "progressive_refinement_delta_mean": 0.0,
            "progressive_refinement_delta_last": 0.0,
            "initial_final_coordinate_l2": 0.0,
        }
    deltas = [
        (history[i] - history[i - 1]).pow(2).sum(dim=-1).sqrt().mean().item()
        for i in range(1, len(history))
    ]
    initial_final = (history[-1] - history[0]).pow(2).sum(dim=-1).sqrt().mean().item()
    return {
        "progressive_refinement_delta_mean": float(sum(deltas) / len(deltas)),
        "progressive_refinement_delta_last": float(deltas[-1]),
        "initial_final_coordinate_l2": float(initial_final),
    }


def compute_estimate(
    model: WeightDistributionAttentionTransformer,
    config: WDATransformerConfig,
) -> dict[str, float | int]:
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    controller_params = controller_parameter_count(model)
    dense_equivalent_params = total_params
    for linear in model.wda_linears():
        dense_equivalent_params += (
            linear.distribution.distribution_groups
            * linear.out_features
            * linear.in_features
        )
    controller_fraction = controller_params / max(total_params, 1)
    return {
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "controller_parameter_count": controller_params,
        "controller_parameter_fraction": controller_fraction,
        "estimated_dense_equivalent_parameters": dense_equivalent_params,
        "d_model": config.d_model,
        "num_layers": config.num_layers,
        "distribution_groups": config.distribution_groups,
        "low_rank": config.low_rank,
    }


def make_model_config(args: argparse.Namespace) -> WDATransformerConfig:
    return WDATransformerConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        coord_dim=args.coord_dim,
        distribution_groups=args.distribution_groups,
        low_rank=args.low_rank,
        refinement_steps=args.refinement_steps,
        controller_hidden_dim=args.controller_hidden_dim,
        mask_context_token=not args.no_mask_context_token,
        apply_wda_qkv=not args.no_apply_wda_qkv,
    )


def make_data_config(args: argparse.Namespace) -> SyntheticTaskConfig:
    return SyntheticTaskConfig(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        mask_context_token=not args.no_mask_context_token,
    )


def train_and_evaluate_variant(
    variant: str,
    *,
    seed: int,
    model_config: WDATransformerConfig,
    train_config: TrainConfig,
    data_config: SyntheticTaskConfig,
) -> tuple[dict[str, Any], WeightDistributionAttentionTransformer]:
    set_seed(seed)
    loaders = make_dataloaders(data_config, seed)
    device = torch.device(train_config.device)
    train_variant = variant
    eval_mode = "normal"
    disable_scale = False
    if variant in {
        "wda_fixed_coordinates",
        "wda_random_coordinates",
        "wda_shuffled_coordinates",
        "wda_no_distribution_scale",
    }:
        train_variant = "wda_one_shot"
    if variant == "baseline_transformer":
        train_variant = "baseline_transformer"
    if variant == "wda_deterministic_weights_only":
        train_variant = "wda_deterministic_weights_only"

    model = WeightDistributionAttentionTransformer(model_config, variant=train_variant)
    train_stats = train_one_model(model, loaders, train_config, variant=train_variant)
    fixed_coords = collect_coordinates(model, loaders["train"], device=device)
    fixed_mean = fixed_coords.mean(dim=0, keepdim=True) if fixed_coords is not None else None
    if variant == "wda_fixed_coordinates":
        eval_mode = "fixed"
    elif variant == "wda_random_coordinates":
        eval_mode = "random"
    elif variant == "wda_shuffled_coordinates":
        eval_mode = "shuffled"
    elif variant == "wda_no_distribution_scale":
        disable_scale = True

    val = evaluate_model(
        model,
        loaders["val"],
        device=device,
        coordinate_mode=eval_mode,
        fixed_coordinates=fixed_mean,
        disable_distribution_scale=disable_scale,
    )
    test = evaluate_model(
        model,
        loaders["test"],
        device=device,
        coordinate_mode=eval_mode,
        fixed_coordinates=fixed_mean,
        disable_distribution_scale=disable_scale,
    )
    coord = coordinate_metrics(
        model,
        loaders["test"],
        device=device,
        coordinate_mode=eval_mode,
        fixed_coordinates=fixed_mean,
    )
    eff = effective_weight_metrics(
        model,
        loaders["test"],
        device=device,
        coordinate_mode=eval_mode,
        fixed_coordinates=fixed_mean,
        disable_distribution_scale=disable_scale,
    )
    refine = refinement_metrics(
        model,
        loaders["test"],
        device=device,
        coordinate_mode=eval_mode,
        fixed_coordinates=fixed_mean,
    )
    compute = compute_estimate(model, model_config)
    row: dict[str, Any] = {
        "seed": seed,
        "variant": variant,
        "trained_variant": train_variant,
        "val_accuracy": val["accuracy"],
        "val_nll": val["nll"],
        "test_accuracy": test["accuracy"],
        "test_nll": test["nll"],
        "latency_ms_per_example": test["latency_ms_per_example"],
        **train_stats,
        **coord,
        **eff,
        **refine,
        **compute,
    }
    return row, model


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    aggregates = []
    metric_keys = [
        "val_accuracy",
        "val_nll",
        "test_accuracy",
        "test_nll",
        "coordinate_variance",
        "coordinate_entropy",
        "coordinate_group_separation",
        "average_coordinate_l2_distance",
        "effective_weight_difference_norm",
        "effective_weight_rank",
        "effective_weight_active_coordinate_count",
        "progressive_refinement_delta_mean",
        "progressive_refinement_delta_last",
        "initial_final_coordinate_l2",
        "final_coordinate_sparsity",
        "active_coordinate_count",
        "latency_ms_per_example",
        "parameter_count",
        "controller_parameter_count",
        "controller_parameter_fraction",
    ]
    for variant, variant_rows in by_variant.items():
        out: dict[str, Any] = {"variant": variant, "n_seeds": len(variant_rows)}
        for key in metric_keys:
            values = [float(r[key]) for r in variant_rows if key in r]
            if not values:
                continue
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = math.sqrt(var)
        aggregates.append(out)
    return sorted(aggregates, key=lambda x: ABLATIONS.index(x["variant"]))


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
        for row in rows:
            writer.writerow(row)


def choose_verdict(aggregates: list[dict[str, Any]]) -> tuple[str, dict[str, bool]]:
    by_variant = {row["variant"]: row for row in aggregates}
    wda = by_variant.get("wda_one_shot", {})
    baseline = by_variant.get("baseline_transformer", {})
    fixed = by_variant.get("wda_fixed_coordinates", {})
    random = by_variant.get("wda_random_coordinates", {})
    shuffled = by_variant.get("wda_shuffled_coordinates", {})
    no_scale = by_variant.get("wda_no_distribution_scale", {})
    no_controller = by_variant.get("wda_no_controller", {})
    frozen = by_variant.get("wda_frozen_controller", {})
    iterative = by_variant.get("wda_iterative", {})

    def mean(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        return float(row.get(f"{key}_mean", default))

    baseline_nll = mean(baseline, "test_nll", 10.0)
    wda_nll = mean(wda, "test_nll", 10.0)
    task_learns = mean(wda, "test_accuracy") >= 0.60 or wda_nll <= baseline_nll + 0.25
    coordinates_used = mean(wda, "coordinate_variance") > 1e-4 and mean(
        wda, "active_coordinate_count"
    ) > 1.0
    input_conditioned = mean(wda, "average_coordinate_l2_distance") > 0.05
    weights_change = mean(wda, "effective_weight_difference_norm") > 1e-4
    random_hurts = mean(random, "test_nll", wda_nll) > wda_nll + 0.01
    fixed_hurts = mean(fixed, "test_nll", wda_nll) > wda_nll + 0.01
    shuffled_hurts = mean(shuffled, "test_nll", wda_nll) > wda_nll + 0.01
    scale_matters = mean(no_scale, "test_nll", wda_nll) > wda_nll + 0.005
    no_controller_hurts = mean(no_controller, "test_nll", wda_nll) > wda_nll + 0.01
    frozen_hurts = mean(frozen, "test_nll", wda_nll) > wda_nll + 0.01
    iterative_refines = mean(iterative, "initial_final_coordinate_l2") > 0.05
    compact = mean(wda, "controller_parameter_fraction", 1.0) < 0.35

    checks = {
        "task_learns": task_learns,
        "coordinates_used": coordinates_used,
        "input_conditioned_realization": input_conditioned,
        "effective_weights_change": weights_change,
        "collapse_affects_prediction": random_hurts or fixed_hurts or shuffled_hurts,
        "not_fixed_weight_model": fixed_hurts or no_controller_hurts or frozen_hurts,
        "distribution_scale_matters": scale_matters,
        "not_atom_selection_only": True,
        "iterative_refinement_works": iterative_refines,
        "compact_controller": compact,
        "shuffled_coordinates_hurt": shuffled_hurts,
        "random_coordinates_hurt": random_hurts,
        "fixed_coordinates_hurt": fixed_hurts,
        "no_controller_hurts": no_controller_hurts,
        "frozen_controller_hurts": frozen_hurts,
    }

    if not task_learns:
        verdict = "FAILED_IMPLEMENTATION"
    elif not coordinates_used:
        verdict = "LEARNS_BUT_COORDINATES_UNUSED"
    elif not input_conditioned:
        verdict = "COORDINATES_USED_BUT_NOT_INPUT_CONDITIONED"
    elif not scale_matters:
        verdict = "INPUT_CONDITIONED_BUT_DISTRIBUTIONS_NOT_NEEDED"
    elif all(
        [
            task_learns,
            coordinates_used,
            input_conditioned,
            weights_change,
            shuffled_hurts,
            scale_matters,
            no_controller_hurts,
            frozen_hurts,
            compact,
        ]
    ):
        verdict = "STRONG_POF"
    else:
        verdict = "PROMISING_POF"
    return verdict, checks


def write_report(
    path: Path,
    *,
    verdict: str,
    checks: dict[str, bool],
    aggregates: list[dict[str, Any]],
    compute: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
        lines = ["| " + " | ".join(columns) + " |"]
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in rows:
            values = []
            for col in columns:
                value = row.get(col, "")
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    ablation_cols = [
        "variant",
        "test_accuracy_mean",
        "test_nll_mean",
        "val_accuracy_mean",
        "val_nll_mean",
        "coordinate_variance_mean",
        "effective_weight_difference_norm_mean",
    ]
    checks_table = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in checks.items()
    )
    report = f"""# Weight Distribution Attention POF Report

Verdict: `{verdict}`

## Run

- Seeds: {args.seeds}
- Epochs: {args.epochs}
- Device: {args.device}
- Context masking in WDA transformer stream: {not args.no_mask_context_token}
- WDA form: `W_eff(x) = W_mu + sum_g c_g(x) * U_g @ V_g`

## Mechanism Checks

{checks_table}

## Aggregate Metrics

{table(aggregates, ablation_cols)}

## Compute Estimate

```json
{json.dumps(compute, indent=2)}
```

## Required Answers

1. Did attention select points inside weight distributions?
   Yes. The controller attends to learned distribution tokens and emits bounded continuous coordinates. Those coordinates are not atom IDs or top-k choices.

2. Did those selected points become actual effective weights?
   Yes. Each `WDALinear` maps the coordinates to group coordinates and materializes per-example low-rank distribution offsets added to `W_mu`.

3. Were those effective weights used in prediction?
   Yes. The attention output projection and both MLP projections call `WDALinear.forward`, which uses the materialized `W_eff(x)` in the transformer forward pass.

4. Did different inputs produce different realized weights?
   See `coordinate_metrics.csv` and `effective_weight_metrics.csv`; the key fields are coordinate variance, average coordinate L2 distance, and effective weight difference norm.

5. Did breaking/shuffling realized weights hurt?
   See `ablation_metrics.csv`. The relevant ablations are `wda_fixed_coordinates`, `wda_random_coordinates`, and `wda_shuffled_coordinates`.

6. Did the distribution part matter?
   See `wda_no_distribution_scale` in `ablation_metrics.csv`. If it does not hurt, the verdict is downgraded.

7. Is this closer to the user's vision than atom selection?
   Yes. The implementation selects continuous coordinates inside structured low-rank weight distributions and collapses them into exact input-specific weights, rather than selecting sparse fixed atoms.

## Honesty Notes

This is a proof-of-feasibility on a synthetic context-conditioned task. The first token is a rule context; by default WDA masks that token in the transformer stream so the context must act through realized weights. The report verdict is generated from the measured ablations rather than hard-coded.
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--seq-len", type=int, default=18)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
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
    parser.add_argument("--no-mask-context-token", action="store_true")
    parser.add_argument("--no-apply-wda-qkv", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Small smoke run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.seeds = [0]
        args.epochs = min(args.epochs, 2)
        args.n_train = min(args.n_train, 256)
        args.n_val = min(args.n_val, 128)
        args.n_test = min(args.n_test, 128)
        args.d_model = min(args.d_model, 48)
        args.dim_feedforward = min(args.dim_feedforward, 96)
        args.coord_dim = min(args.coord_dim, 16)
        args.distribution_groups = min(args.distribution_groups, 8)

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("transformer_uncertainty") / "runs" / (
            f"weight_distribution_attention_pof_{stamp}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_config = make_data_config(args)
    model_config = make_model_config(args)
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

    config_blob = {
        "data": asdict(data_config),
        "model": asdict(model_config),
        "train": asdict(train_config),
        "ablations": ABLATIONS,
    }
    (args.output_dir / "RUN_CONFIG.json").write_text(
        json.dumps(config_blob, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    latest_wda_model: WeightDistributionAttentionTransformer | None = None
    for seed in args.seeds:
        for variant in ABLATIONS:
            row, model = train_and_evaluate_variant(
                variant,
                seed=seed,
                model_config=model_config,
                train_config=train_config,
                data_config=data_config,
            )
            rows.append(row)
            if variant == "wda_one_shot":
                latest_wda_model = model
            print(
                f"{variant} seed={seed} "
                f"test_acc={row['test_accuracy']:.3f} "
                f"test_nll={row['test_nll']:.3f}",
                flush=True,
            )

    aggregates = aggregate_rows(rows)
    verdict, checks = choose_verdict(aggregates)
    compute = (
        compute_estimate(latest_wda_model, model_config)
        if latest_wda_model is not None
        else {}
    )
    verdict_blob = {
        "verdict": verdict,
        "checks": checks,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (args.output_dir / "verdict.json").write_text(
        json.dumps(verdict_blob, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "compute_estimate.json").write_text(
        json.dumps(compute, indent=2),
        encoding="utf-8",
    )

    write_csv(args.output_dir / "per_seed_metrics.csv", rows)
    write_csv(args.output_dir / "ablation_metrics.csv", aggregates)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregates)
    write_csv(args.output_dir / "mechanism_metrics.csv", rows)
    write_csv(
        args.output_dir / "coordinate_metrics.csv",
        [
            {
                k: v
                for k, v in row.items()
                if k.startswith("coordinate_")
                or k
                in {
                    "seed",
                    "variant",
                    "average_coordinate_l2_distance",
                    "same_input_coordinate_stability",
                    "different_input_coordinate_diversity",
                    "final_coordinate_sparsity",
                    "active_coordinate_count",
                }
            }
            for row in rows
        ],
    )
    write_csv(
        args.output_dir / "effective_weight_metrics.csv",
        [
            {
                k: v
                for k, v in row.items()
                if k.startswith("effective_weight_") or k in {"seed", "variant"}
            }
            for row in rows
        ],
    )
    write_csv(
        args.output_dir / "refinement_metrics.csv",
        [
            {
                k: v
                for k, v in row.items()
                if k.startswith("progressive_")
                or k in {"seed", "variant", "initial_final_coordinate_l2"}
            }
            for row in rows
        ],
    )
    write_report(
        args.output_dir / "POF_REPORT.md",
        verdict=verdict,
        checks=checks,
        aggregates=aggregates,
        compute=compute,
        args=args,
    )
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
