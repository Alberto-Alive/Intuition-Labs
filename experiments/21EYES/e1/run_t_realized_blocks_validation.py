from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from .t_realized_attention import TRAblation, TRealizedAttention
    from .t_realized_blocks import (
        TRealizedBlockTransformer,
        TRealizedBlockTransformerConfig,
        count_parameters,
        estimate_active_mult_adds,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from t_realized_attention import TRAblation, TRealizedAttention
    from t_realized_blocks import (
        TRealizedBlockTransformer,
        TRealizedBlockTransformerConfig,
        count_parameters,
        estimate_active_mult_adds,
    )


BenchmarkName = Literal["rule_switching_sequence", "compositional_rule_switching_easy"]

BENCHMARKS: list[BenchmarkName] = [
    "rule_switching_sequence",
    "compositional_rule_switching_easy",
]

MODEL_SIZES: dict[str, dict[str, int]] = {
    "small": {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
    "medium": {"d_model": 96, "nhead": 4, "num_layers": 3, "dim_feedforward": 192},
    "large": {"d_model": 128, "nhead": 8, "num_layers": 4, "dim_feedforward": 256},
    "small_4block": {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 4,
        "dim_feedforward": 128,
    },
}

MODEL_VARIANTS = [
    "fixed_small_transformer",
    "fixed_medium_transformer",
    "fixed_large_transformer",
    "tblock_1",
    "tblock_2",
    "tblock_every_other",
    "tblock_all",
    "v_only_tblock_1",
    "v_only_tblock_2",
    "k_only_tblock_1",
    "fixed_small_4block",
    "tblock_4block_1",
    "tblock_4block_2",
    "tblock_4block_all",
]

BASE_FULL_TBLOCK_VARIANTS = [
    "tblock_1",
    "tblock_2",
    "tblock_every_other",
    "tblock_all",
]
DEPTH_FULL_TBLOCK_VARIANTS = [
    "tblock_4block_1",
    "tblock_4block_2",
    "tblock_4block_all",
]
FULL_TBLOCK_VARIANTS = BASE_FULL_TBLOCK_VARIANTS + DEPTH_FULL_TBLOCK_VARIANTS

T_BLOCK_ABLATIONS: dict[str, TRAblation] = {
    "t_zero": "t_zero",
    "sigma_zero": "sigma_zero",
    "shuffle_T": "shuffle_T",
    "random_T": "random_T",
    "fixed_normal_attention": "fixed_normal_attention",
    "v_only": "v_only",
    "k_only": "k_only",
}

PLACEMENT_CANDIDATES = {
    "tblock_1": ["first_layer", "middle_layer", "last_layer"],
    "tblock_2": ["first_and_last", "first_two", "last_two"],
}


@dataclass(frozen=True)
class TBlockTaskConfig:
    n_train: int = 5000
    n_val: int = 1000
    n_test: int = 1000
    seq_len: int = 16
    vocab_size: int = 64
    num_rules: int = 6
    num_classes: int = 10
    batch_size: int = 128


@dataclass(frozen=True)
class TrainConfig:
    seeds: list[int]
    epochs: int = 20
    lr: float = 0.002
    weight_decay: float = 0.0001
    grad_clip: float = 5.0
    dropout: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _labels_by_rule(values: Tensor, num_classes: int) -> Tensor:
    payload_len = values.shape[1]
    reverse_position = max(0, payload_len - 2)
    return torch.stack(
        [
            values[:, 0],
            values[:, -1],
            (values[:, 0] + values[:, 1]) % num_classes,
            values[:, : min(8, payload_len)].sum(dim=1) % num_classes,
            values[:, reverse_position],
            (values[:, min(2, payload_len - 1)] + 3) % num_classes,
        ],
        dim=1,
    )


def make_rule_switching_split(
    n_examples: int,
    *,
    config: TBlockTaskConfig,
    seed: int,
) -> TensorDataset:
    if config.seq_len < 8:
        raise ValueError("seq_len must be at least 8")
    if config.num_rules < 1 or config.num_rules > 6:
        raise ValueError("num_rules must be between 1 and 6")
    payload_offset = 8
    if payload_offset + config.num_classes > config.vocab_size:
        raise ValueError("vocab_size must fit payload_offset + num_classes")

    generator = torch.Generator().manual_seed(seed)
    rules = torch.randint(0, config.num_rules, (n_examples,), generator=generator)
    values = torch.randint(
        0,
        config.num_classes,
        (n_examples, config.seq_len - 1),
        generator=generator,
    )
    tokens = torch.empty(n_examples, config.seq_len, dtype=torch.long)
    tokens[:, 0] = rules
    tokens[:, 1:] = values + payload_offset
    all_labels = _labels_by_rule(values, config.num_classes)[:, : config.num_rules]
    labels = all_labels.gather(1, rules.view(-1, 1)).squeeze(1)
    return TensorDataset(tokens, labels.long(), rules.long())


def _composition_features(values: Tensor, num_classes: int) -> Tensor:
    payload_len = values.shape[1]
    return torch.stack(
        [
            values[:, 0],
            values[:, -1],
            (values[:, 0] + values[:, 1]) % num_classes,
            values[:, : min(6, payload_len)].sum(dim=1) % num_classes,
            (2 * values[:, min(2, payload_len - 1)] + values[:, min(3, payload_len - 1)])
            % num_classes,
            (values[:, min(4, payload_len - 1)] + 3) % num_classes,
        ],
        dim=1,
    )


def make_compositional_rule_switching_easy_split(
    n_examples: int,
    *,
    config: TBlockTaskConfig,
    seed: int,
) -> TensorDataset:
    if config.seq_len < 8:
        raise ValueError("seq_len must be at least 8")
    if config.num_rules != 6:
        raise ValueError("compositional_rule_switching_easy expects num_rules=6")
    payload_offset = 16
    if payload_offset + config.num_classes > config.vocab_size:
        raise ValueError("vocab_size must fit payload_offset + num_classes")

    generator = torch.Generator().manual_seed(seed)
    first_rules = torch.randint(0, config.num_rules, (n_examples,), generator=generator)
    second_rules = torch.randint(0, config.num_rules, (n_examples,), generator=generator)
    values = torch.randint(
        0,
        config.num_classes,
        (n_examples, config.seq_len - 2),
        generator=generator,
    )
    tokens = torch.empty(n_examples, config.seq_len, dtype=torch.long)
    tokens[:, 0] = first_rules
    tokens[:, 1] = 8 + second_rules
    tokens[:, 2:] = values + payload_offset
    features = _composition_features(values, config.num_classes)
    first_values = features.gather(1, first_rules.view(-1, 1)).squeeze(1)
    second_values = features.gather(1, second_rules.view(-1, 1)).squeeze(1)
    labels = (first_values + 2 * second_values) % config.num_classes
    groups = first_rules * config.num_rules + second_rules
    return TensorDataset(tokens.long(), labels.long(), groups.long())


def make_dataloaders(
    config: TBlockTaskConfig,
    *,
    seed: int,
    benchmark: BenchmarkName,
) -> dict[str, DataLoader]:
    split_fn = (
        make_rule_switching_split
        if benchmark == "rule_switching_sequence"
        else make_compositional_rule_switching_easy_split
    )
    return {
        "train": DataLoader(
            split_fn(config.n_train, config=config, seed=seed * 1009 + 17),
            batch_size=config.batch_size,
            shuffle=True,
        ),
        "val": DataLoader(
            split_fn(config.n_val, config=config, seed=seed * 1009 + 29),
            batch_size=config.batch_size,
            shuffle=False,
        ),
        "test": DataLoader(
            split_fn(config.n_test, config=config, seed=seed * 1009 + 43),
            batch_size=config.batch_size,
            shuffle=False,
        ),
    }


def num_groups_for_benchmark(config: TBlockTaskConfig, benchmark: BenchmarkName) -> int:
    if benchmark == "compositional_rule_switching_easy":
        return config.num_rules * config.num_rules
    return config.num_rules


def size_name_for_variant(variant: str) -> str:
    if variant == "fixed_medium_transformer":
        return "medium"
    if variant == "fixed_large_transformer":
        return "large"
    if "4block" in variant:
        return "small_4block"
    return "small"


def t_mode_for_variant(variant: str) -> Literal["full", "v_only", "k_only"]:
    if variant.startswith("v_only"):
        return "v_only"
    if variant.startswith("k_only"):
        return "k_only"
    return "full"


def _unique_layers(indices: list[int], num_layers: int) -> tuple[int, ...]:
    return tuple(sorted({max(0, min(num_layers - 1, idx)) for idx in indices}))


def single_layer_indices(num_layers: int, placement: str) -> tuple[int, ...]:
    if placement == "first_layer":
        return (0,)
    if placement == "middle_layer":
        return (num_layers // 2,)
    if placement == "last_layer":
        return (num_layers - 1,)
    raise ValueError(f"unknown single-layer placement: {placement}")


def two_layer_indices(num_layers: int, placement: str) -> tuple[int, ...]:
    if placement == "first_and_last":
        return _unique_layers([0, num_layers - 1], num_layers)
    if placement == "first_two":
        return _unique_layers([0, 1], num_layers)
    if placement == "last_two":
        return _unique_layers([num_layers - 2, num_layers - 1], num_layers)
    raise ValueError(f"unknown two-layer placement: {placement}")


def t_layers_for_variant(
    variant: str,
    num_layers: int,
    *,
    selected_placements: dict[str, str] | None = None,
    explicit_placement: str | None = None,
) -> tuple[int, ...]:
    selected_placements = selected_placements or {}
    if variant.startswith("fixed"):
        return ()
    if variant in {"tblock_1", "v_only_tblock_1", "k_only_tblock_1"}:
        placement = explicit_placement or selected_placements.get("tblock_1", "middle_layer")
        return single_layer_indices(num_layers, placement)
    if variant in {"tblock_2", "v_only_tblock_2"}:
        placement = explicit_placement or selected_placements.get("tblock_2", "first_and_last")
        return two_layer_indices(num_layers, placement)
    if variant == "tblock_every_other":
        return tuple(range(0, num_layers, 2))
    if variant == "tblock_all":
        return tuple(range(num_layers))
    if variant == "tblock_4block_1":
        return single_layer_indices(num_layers, "middle_layer")
    if variant == "tblock_4block_2":
        return two_layer_indices(num_layers, "first_and_last")
    if variant == "tblock_4block_all":
        return tuple(range(num_layers))
    raise ValueError(f"unknown model variant: {variant}")


def build_model_config(
    args: argparse.Namespace,
    *,
    variant: str,
    selected_placements: dict[str, str] | None = None,
    explicit_placement: str | None = None,
) -> TRealizedBlockTransformerConfig:
    size = MODEL_SIZES[size_name_for_variant(variant)]
    t_layers = t_layers_for_variant(
        variant,
        size["num_layers"],
        selected_placements=selected_placements,
        explicit_placement=explicit_placement,
    )
    return TRealizedBlockTransformerConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        d_model=size["d_model"],
        nhead=size["nhead"],
        num_layers=size["num_layers"],
        dim_feedforward=size["dim_feedforward"],
        dropout=args.dropout,
        readout_position=args.readout_position,
        sigma_init_gain=args.sigma_init_gain,
        t_layers=t_layers,
        t_mode=t_mode_for_variant(variant),
    )


def build_model(
    args: argparse.Namespace,
    *,
    variant: str,
    selected_placements: dict[str, str] | None = None,
    explicit_placement: str | None = None,
) -> TRealizedBlockTransformer:
    return TRealizedBlockTransformer(
        build_model_config(
            args,
            variant=variant,
            selected_placements=selected_placements,
            explicit_placement=explicit_placement,
        )
    )


def logits_from_output(output: Tensor | dict[str, Any]) -> Tensor:
    if isinstance(output, dict):
        return output["logits"]
    return output


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    ablation: TRAblation = "none",
    num_groups: int = 0,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    correct_by_group = torch.zeros(num_groups, dtype=torch.long)
    count_by_group = torch.zeros(num_groups, dtype=torch.long)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for tokens, labels, groups in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        output = model(tokens, ablation=ablation, return_details=False)
        logits = logits_from_output(output)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        predictions = logits.argmax(dim=-1)
        correct = predictions.eq(labels)

        total_loss += float(loss.item())
        total_correct += int(correct.sum().item())
        total_examples += labels.numel()

        if num_groups:
            correct_cpu = correct.cpu()
            for group in range(num_groups):
                mask = groups.eq(group)
                count = int(mask.sum().item())
                if count:
                    count_by_group[group] += count
                    correct_by_group[group] += int(correct_cpu[mask].sum().item())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    per_rule_accuracy = {}
    for group in range(num_groups):
        if count_by_group[group].item():
            per_rule_accuracy[f"group_{group}"] = (
                correct_by_group[group].item() / count_by_group[group].item()
            )
        else:
            per_rule_accuracy[f"group_{group}"] = float("nan")

    return {
        "accuracy": total_correct / max(1, total_examples),
        "nll": total_loss / max(1, total_examples),
        "examples": total_examples,
        "latency_ms_per_example": 1000.0 * elapsed / max(1, total_examples),
        "per_rule_accuracy": per_rule_accuracy,
    }


def train_one_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    train_config: TrainConfig,
    *,
    num_groups: int,
) -> dict[str, float | int | None]:
    device = torch.device(train_config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    best_val_nll = float("inf")
    best_state: dict[str, Tensor] | None = None
    best_val_epoch = 0
    epoch_seconds: list[float] = []
    epochs_to_70: int | None = None
    epochs_to_75: int | None = None
    wall_to_70: float | None = None
    wall_to_75: float | None = None
    started = time.perf_counter()

    for epoch in range(1, train_config.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        for tokens, labels, _groups in loaders["train"]:
            tokens = tokens.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(tokens, return_details=False)
            logits = logits_from_output(output)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()

        val_metrics = evaluate_model(
            model,
            loaders["val"],
            device=device,
            num_groups=num_groups,
        )
        elapsed_total = time.perf_counter() - started
        epoch_seconds.append(time.perf_counter() - epoch_started)
        if epochs_to_70 is None and val_metrics["accuracy"] >= 0.70:
            epochs_to_70 = epoch
            wall_to_70 = elapsed_total
        if epochs_to_75 is None and val_metrics["accuracy"] >= 0.75:
            epochs_to_75 = epoch
            wall_to_75 = elapsed_total
        if val_metrics["nll"] < best_val_nll:
            best_val_nll = float(val_metrics["nll"])
            best_val_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    total_seconds = time.perf_counter() - started
    return {
        "training_seconds_total": total_seconds,
        "seconds_per_epoch": sum(epoch_seconds) / max(1, len(epoch_seconds)),
        "best_val_epoch": best_val_epoch,
        "best_val_nll": best_val_nll,
        "epochs_to_reach_70_acc": epochs_to_70,
        "epochs_to_reach_75_acc": epochs_to_75,
        "wall_clock_to_reach_70_acc": wall_to_70,
        "wall_clock_to_reach_75_acc": wall_to_75,
    }


def _mean_tensor(values: list[Tensor]) -> float:
    if not values:
        return 0.0
    return float(torch.stack([value.detach().float().cpu() for value in values]).mean().item())


def _mean_float(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@torch.no_grad()
def collect_mechanism_metrics(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int = 2,
) -> dict[str, Any]:
    model.eval()
    t_means: list[Tensor] = []
    t_stds: list[Tensor] = []
    t_variances: list[Tensor] = []
    k_temp_diffs: list[Tensor] = []
    v_temp_diffs: list[Tensor] = []
    attention_diffs: list[Tensor] = []
    t_zero_output_diffs: list[Tensor] = []
    sigma_zero_output_diffs: list[Tensor] = []
    per_layer_t_variance: dict[int, list[float]] = {}
    per_layer_zero_diff: dict[int, list[float]] = {}

    for batch_idx, (tokens, _labels, _groups) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        tokens = tokens.to(device)
        base_output = model(tokens, return_details=True)
        t_zero_output = model(tokens, ablation="t_zero", return_details=True)
        sigma_zero_output = model(tokens, ablation="sigma_zero", return_details=True)
        if not isinstance(base_output, dict):
            continue
        base_diagnostics = base_output.get("tra_diagnostics", [])
        t_zero_diagnostics = (
            t_zero_output.get("tra_diagnostics", [])
            if isinstance(t_zero_output, dict)
            else []
        )
        sigma_zero_diagnostics = (
            sigma_zero_output.get("tra_diagnostics", [])
            if isinstance(sigma_zero_output, dict)
            else []
        )

        for idx, diagnostic in enumerate(base_diagnostics):
            T = diagnostic["T"].float()
            K_temp = diagnostic["K_temp"].float()
            V_temp = diagnostic["V_temp"].float()
            K_mu = diagnostic["K_mu"].float().unsqueeze(2)
            V_mu = diagnostic["V_mu"].float().unsqueeze(2)
            attn = diagnostic["attn"].float()
            normal_attn = diagnostic["normal_attn"].float()
            output_heads = diagnostic["output_heads"].float()

            t_means.append(T.mean())
            t_stds.append(T.std(unbiased=False))
            t_variances.append(T.var(unbiased=False))
            k_temp_diffs.append((K_temp - K_mu).norm(dim=-1).mean())
            v_temp_diffs.append((V_temp - V_mu).norm(dim=-1).mean())
            attention_diffs.append((attn - normal_attn).abs().mean())
            per_layer_t_variance.setdefault(idx, []).append(float(T.var(unbiased=False).item()))

            if idx < len(t_zero_diagnostics):
                zero_heads = t_zero_diagnostics[idx]["output_heads"].float()
                diff = (output_heads - zero_heads).norm(dim=-1).mean()
                t_zero_output_diffs.append(diff)
                per_layer_zero_diff.setdefault(idx, []).append(float(diff.item()))
            if idx < len(sigma_zero_diagnostics):
                sigma_zero_heads = sigma_zero_diagnostics[idx]["output_heads"].float()
                sigma_zero_output_diffs.append(
                    (output_heads - sigma_zero_heads).norm(dim=-1).mean()
                )

    return {
        "T_mean": _mean_tensor(t_means),
        "T_std": _mean_tensor(t_stds),
        "T_variance": _mean_tensor(t_variances),
        "K_temp_difference_norm": _mean_tensor(k_temp_diffs),
        "V_temp_difference_norm": _mean_tensor(v_temp_diffs),
        "attention_difference_vs_normal": _mean_tensor(attention_diffs),
        "output_difference_when_T_zero": _mean_tensor(t_zero_output_diffs),
        "output_difference_when_sigma_zero": _mean_tensor(sigma_zero_output_diffs),
        "per_layer_T_variance": json.dumps(
            [_mean_float(per_layer_t_variance[idx]) for idx in sorted(per_layer_t_variance)]
        ),
        "per_layer_output_difference_when_T_zero": json.dumps(
            [_mean_float(per_layer_zero_diff[idx]) for idx in sorted(per_layer_zero_diff)]
        ),
    }


def placement_study(
    args: argparse.Namespace,
    *,
    seed: int,
    benchmark: BenchmarkName,
    loaders: dict[str, DataLoader],
    train_config: TrainConfig,
    num_groups: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, str] = {}
    placement_train_config = TrainConfig(
        seeds=[seed],
        epochs=args.placement_epochs,
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        grad_clip=train_config.grad_clip,
        dropout=train_config.dropout,
        device=train_config.device,
    )
    device = torch.device(train_config.device)
    for variant, placements in PLACEMENT_CANDIDATES.items():
        variant_rows: list[dict[str, Any]] = []
        for placement in placements:
            set_seed(seed * 2003 + len(rows))
            model = build_model(args, variant=variant, explicit_placement=placement)
            model_config = build_model_config(args, variant=variant, explicit_placement=placement)
            train_stats = train_one_model(
                model,
                loaders,
                placement_train_config,
                num_groups=num_groups,
            )
            val_metrics = evaluate_model(
                model,
                loaders["val"],
                device=device,
                num_groups=num_groups,
            )
            row = {
                "seed": seed,
                "benchmark": benchmark,
                "model": variant,
                "placement": placement,
                "t_layers": json.dumps(model_config.t_layers),
                "validation_accuracy": val_metrics["accuracy"],
                "validation_nll": val_metrics["nll"],
                "training_seconds_total": train_stats["training_seconds_total"],
                "selected": False,
            }
            rows.append(row)
            variant_rows.append(row)

        best = max(
            variant_rows,
            key=lambda item: (float(item["validation_accuracy"]), -float(item["validation_nll"])),
        )
        selected[variant] = str(best["placement"])
        for row in variant_rows:
            row["selected"] = row["placement"] == best["placement"]
    return rows, selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    numeric_keys = [
        "validation_accuracy",
        "validation_nll",
        "test_accuracy",
        "test_nll",
        "parameter_count",
        "active_mult_adds_estimate",
        "latency_ms_per_example",
        "training_seconds_total",
        "seconds_per_epoch",
        "best_val_epoch",
        "epochs_to_reach_70_acc",
        "epochs_to_reach_75_acc",
        "wall_clock_to_reach_70_acc",
        "wall_clock_to_reach_75_acc",
        "accuracy_gain_vs_fixed_small",
        "nll_gain_vs_fixed_small",
        "parameter_ratio_vs_fixed_small",
        "compute_ratio_vs_fixed_small",
        "latency_ratio_vs_fixed_small",
        "accuracy_per_million_multadds",
        "nll_per_latency",
        "improvement_per_extra_latency",
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[item] for item in group_keys)
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key, subset in sorted(groups.items(), key=lambda item: item[0]):
        aggregate = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        aggregate["n_rows"] = len(subset)
        for metric in numeric_keys:
            values = [
                value
                for value in (_as_float(row.get(metric)) for row in subset)
                if value is not None
            ]
            aggregate[f"mean_{metric}"] = sum(values) / len(values) if values else ""
        output.append(aggregate)
    return output


def enrich_value_metrics(rows: list[dict[str, Any]]) -> None:
    fixed_by_seed_benchmark = {
        (row["seed"], row["benchmark"]): row
        for row in rows
        if row["model"] == "fixed_small_transformer"
    }
    for row in rows:
        fixed = fixed_by_seed_benchmark.get((row["seed"], row["benchmark"]))
        if fixed is None:
            continue
        fixed_acc = float(fixed["test_accuracy"])
        fixed_nll = float(fixed["test_nll"])
        fixed_params = max(1.0, float(fixed["parameter_count"]))
        fixed_compute = max(1.0, float(fixed["active_mult_adds_estimate"]))
        fixed_latency = max(1e-9, float(fixed["latency_ms_per_example"]))
        row["accuracy_gain_vs_fixed_small"] = float(row["test_accuracy"]) - fixed_acc
        row["nll_gain_vs_fixed_small"] = fixed_nll - float(row["test_nll"])
        row["parameter_ratio_vs_fixed_small"] = float(row["parameter_count"]) / fixed_params
        row["compute_ratio_vs_fixed_small"] = (
            float(row["active_mult_adds_estimate"]) / fixed_compute
        )
        row["latency_ratio_vs_fixed_small"] = (
            float(row["latency_ms_per_example"]) / fixed_latency
        )
        row["accuracy_per_million_multadds"] = float(row["test_accuracy"]) / max(
            1e-9,
            float(row["active_mult_adds_estimate"]) / 1_000_000.0,
        )
        row["nll_per_latency"] = float(row["test_nll"]) / max(
            1e-9,
            float(row["latency_ms_per_example"]),
        )
        extra_latency = float(row["latency_ms_per_example"]) - fixed_latency
        row["improvement_per_extra_latency"] = (
            row["accuracy_gain_vs_fixed_small"] / max(1e-9, extra_latency)
            if extra_latency > 0
            else row["accuracy_gain_vs_fixed_small"]
        )


def aggregate_ablation_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name in sorted({row["ablation"] for row in rows}):
        subset = [row for row in rows if row["ablation"] == name]
        summary[name] = {
            "accuracy": sum(float(row["accuracy"]) for row in subset) / len(subset),
            "nll": sum(float(row["nll"]) for row in subset) / len(subset),
            "accuracy_drop": sum(float(row["accuracy_drop"]) for row in subset)
            / len(subset),
            "nll_increase": sum(float(row["nll_increase"]) for row in subset)
            / len(subset),
        }
    return summary


def aggregate_mechanism_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    numeric_keys = [
        "T_mean",
        "T_std",
        "T_variance",
        "K_temp_difference_norm",
        "V_temp_difference_norm",
        "attention_difference_vs_normal",
        "output_difference_when_T_zero",
        "output_difference_when_sigma_zero",
    ]
    result: dict[str, float] = {}
    for key in numeric_keys:
        values = [
            value
            for value in (_as_float(row.get(key)) for row in rows)
            if value is not None
        ]
        result[key] = sum(values) / len(values) if values else 0.0
    return result


def compute_metrics_summary(
    aggregate: list[dict[str, Any]],
    per_benchmark: list[dict[str, Any]],
) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    for row in aggregate:
        models[row["model"]] = {
            "test_accuracy": row.get("mean_test_accuracy", ""),
            "test_nll": row.get("mean_test_nll", ""),
            "parameter_count": row.get("mean_parameter_count", ""),
            "active_mult_adds_estimate": row.get("mean_active_mult_adds_estimate", ""),
            "latency_ms_per_example": row.get("mean_latency_ms_per_example", ""),
            "accuracy_gain_vs_fixed_small": row.get(
                "mean_accuracy_gain_vs_fixed_small",
                "",
            ),
            "nll_gain_vs_fixed_small": row.get("mean_nll_gain_vs_fixed_small", ""),
            "parameter_ratio_vs_fixed_small": row.get(
                "mean_parameter_ratio_vs_fixed_small",
                "",
            ),
            "compute_ratio_vs_fixed_small": row.get(
                "mean_compute_ratio_vs_fixed_small",
                "",
            ),
            "latency_ratio_vs_fixed_small": row.get(
                "mean_latency_ratio_vs_fixed_small",
                "",
            ),
        }
    return {
        "models": models,
        "per_benchmark": per_benchmark,
    }


def _row_for_model(aggregate: list[dict[str, Any]], model: str) -> dict[str, Any]:
    for row in aggregate:
        if row.get("model") == model:
            return row
    return {}


def _metric(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = _as_float(row.get(f"mean_{key}"))
    return default if value is None else value


def _model_metric(
    aggregate: list[dict[str, Any]],
    model: str,
    key: str,
    default: float = float("nan"),
) -> float:
    return _metric(_row_for_model(aggregate, model), key, default)


def _beats(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    acc_delta: float = 0.0,
    nll_delta: float = 0.0,
) -> bool:
    cand_acc = _metric(candidate, "test_accuracy")
    base_acc = _metric(baseline, "test_accuracy")
    cand_nll = _metric(candidate, "test_nll")
    base_nll = _metric(baseline, "test_nll")
    return cand_acc >= base_acc + acc_delta or cand_nll <= base_nll - nll_delta


def best_model_row(aggregate: list[dict[str, Any]], models: list[str]) -> dict[str, Any]:
    existing = [_row_for_model(aggregate, model) for model in models]
    existing = [row for row in existing if row]
    if not existing:
        return {}
    return max(
        existing,
        key=lambda row: (
            _metric(row, "test_accuracy", -1.0),
            -_metric(row, "test_nll", float("inf")),
        ),
    )


def ablation_hurts(
    ablation_summary: dict[str, dict[str, float]],
    name: str,
    *,
    accuracy_threshold: float,
    nll_threshold: float,
) -> bool:
    metrics = ablation_summary.get(name, {})
    return (
        metrics.get("accuracy_drop", 0.0) >= accuracy_threshold
        or metrics.get("nll_increase", 0.0) >= nll_threshold
    )


def implementation_path_clean() -> bool:
    source = inspect.getsource(TRealizedAttention.forward)
    required = [
        'T = torch.tanh(torch.einsum("bhtd,bhjd->bhtj", Q, K_point) / scale)',
        "K_temp = K_mu.unsqueeze(2) + T.unsqueeze(-1) * K_sigma.unsqueeze(2)",
        "V_temp = V_mu.unsqueeze(2) + T.unsqueeze(-1) * V_sigma.unsqueeze(2)",
        'score = torch.einsum("bhtd,bhtjd->bhtj", Q, K_temp) / scale',
        "attn = torch.softmax(score, dim=-1)",
        'output = torch.einsum("bhtj,bhtjd->bhtd", attn, V_temp)',
    ]
    return all(line in source for line in required)


def no_controller_or_hypernetwork(args: argparse.Namespace) -> tuple[bool, bool]:
    sample = TRealizedBlockTransformer(
        TRealizedBlockTransformerConfig(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            num_classes=args.num_classes,
            t_layers=(0,),
        )
    )
    no_controller = not any(
        "controller" in type(module).__name__.lower() for module in sample.modules()
    ) and not hasattr(sample, "controller")
    no_hypernetwork = not any(
        "hyper" in type(module).__name__.lower() for module in sample.modules()
    )
    return no_controller, no_hypernetwork


def make_verdict(
    *,
    args: argparse.Namespace,
    aggregate: list[dict[str, Any]],
    ablation_summary: dict[str, dict[str, float]],
) -> dict[str, Any]:
    fixed_small = _row_for_model(aggregate, "fixed_small_transformer")
    fixed_medium = _row_for_model(aggregate, "fixed_medium_transformer")
    fixed_4block = _row_for_model(aggregate, "fixed_small_4block")
    best_base = best_model_row(aggregate, BASE_FULL_TBLOCK_VARIANTS)
    best_depth = best_model_row(aggregate, DEPTH_FULL_TBLOCK_VARIANTS)
    random_accuracy = 1.0 / float(args.num_classes)

    fixed_acc = _metric(fixed_small, "test_accuracy")
    fixed_nll = _metric(fixed_small, "test_nll")
    medium_acc = _metric(fixed_medium, "test_accuracy")
    best_acc = _metric(best_base, "test_accuracy")
    best_nll = _metric(best_base, "test_nll")
    best_latency_ratio = _metric(best_base, "latency_ratio_vs_fixed_small", float("inf"))
    best_seconds = _metric(best_base, "seconds_per_epoch", float("inf"))
    fixed_seconds = _metric(fixed_small, "seconds_per_epoch", float("inf"))

    tblock_beats_fixed_small = (
        best_acc >= fixed_acc + 0.01 or best_nll <= fixed_nll - 0.03
    )
    tblock_improves_nll = best_nll < fixed_nll
    if medium_acc > fixed_acc:
        tblock_approaches_medium = best_acc >= fixed_acc + 0.5 * (medium_acc - fixed_acc)
    else:
        tblock_approaches_medium = True

    tblock_1 = _row_for_model(aggregate, "tblock_1")
    tblock_2 = _row_for_model(aggregate, "tblock_2")
    tblock_4block_1 = _row_for_model(aggregate, "tblock_4block_1")
    tblock_4block_2 = _row_for_model(aggregate, "tblock_4block_2")

    stacking_helps = _beats(tblock_2, tblock_1) or _beats(tblock_4block_2, tblock_4block_1)
    stacking_not_just_more_depth = _beats(
        tblock_4block_2,
        fixed_4block,
        acc_delta=0.0,
        nll_delta=0.03,
    )

    v_only_is_efficient = False
    for full_name, v_name in [
        ("tblock_1", "v_only_tblock_1"),
        ("tblock_2", "v_only_tblock_2"),
    ]:
        full = _row_for_model(aggregate, full_name)
        v_only = _row_for_model(aggregate, v_name)
        full_acc_gain = max(0.0, _metric(full, "test_accuracy") - fixed_acc)
        v_acc_gain = max(0.0, _metric(v_only, "test_accuracy") - fixed_acc)
        full_nll_gain = max(0.0, fixed_nll - _metric(full, "test_nll"))
        v_nll_gain = max(0.0, fixed_nll - _metric(v_only, "test_nll"))
        preserves_acc = full_acc_gain > 0 and v_acc_gain >= 0.70 * full_acc_gain
        preserves_nll = full_nll_gain > 0 and v_nll_gain >= 0.70 * full_nll_gain
        cheaper = (
            _metric(v_only, "compute_ratio_vs_fixed_small", float("inf"))
            < _metric(full, "compute_ratio_vs_fixed_small", float("inf"))
            or _metric(v_only, "latency_ms_per_example", float("inf"))
            < _metric(full, "latency_ms_per_example", float("inf"))
        )
        if (preserves_acc or preserves_nll) and cheaper:
            v_only_is_efficient = True

    target_speed_ok = False
    for key in ["wall_clock_to_reach_70_acc", "wall_clock_to_reach_75_acc"]:
        best_wall = _metric(best_base, key)
        fixed_wall = _metric(fixed_small, key)
        if not math.isnan(best_wall) and (math.isnan(fixed_wall) or best_wall <= fixed_wall):
            target_speed_ok = True

    no_controller_present, no_hypernetwork_present = no_controller_or_hypernetwork(args)
    checks = {
        "implementation_path_clean": implementation_path_clean(),
        "no_controller_present": no_controller_present,
        "no_hypernetwork_present": no_hypernetwork_present,
        "task_learns": best_acc >= random_accuracy + 0.08
        or best_nll <= math.log(args.num_classes) - 0.08,
        "baseline_not_saturated": fixed_acc < 0.95,
        "tblock_beats_fixed_small": tblock_beats_fixed_small,
        "tblock_improves_nll": tblock_improves_nll,
        "tblock_ablations_hurt": all(
            ablation_hurts(
                ablation_summary,
                name,
                accuracy_threshold=args.effect_threshold,
                nll_threshold=args.ablation_nll_threshold,
            )
            for name in ["t_zero", "sigma_zero", "shuffle_T"]
        ),
        "tblock_approaches_medium": tblock_approaches_medium,
        "v_only_is_efficient": v_only_is_efficient,
        "stacking_helps": stacking_helps,
        "stacking_not_just_more_depth": stacking_not_just_more_depth,
        "compute_reasonable": best_latency_ratio <= 1.25
        or (best_acc - fixed_acc) >= 0.03
        or (fixed_nll - best_nll) >= 0.05,
        "training_speed_not_catastrophic": best_seconds <= 1.5 * fixed_seconds
        or target_speed_ok,
    }

    if not (
        checks["implementation_path_clean"]
        and checks["no_controller_present"]
        and checks["no_hypernetwork_present"]
    ):
        final_verdict = "FAILED_IMPLEMENTATION"
    elif not checks["task_learns"]:
        final_verdict = "MECHANISM_WORKS_BUT_NO_VALUE"
    elif checks["tblock_ablations_hurt"] and not checks["tblock_beats_fixed_small"]:
        final_verdict = "MECHANISM_WORKS_BUT_NO_VALUE"
    elif (
        checks["tblock_beats_fixed_small"]
        and checks["tblock_approaches_medium"]
        and checks["tblock_ablations_hurt"]
        and checks["v_only_is_efficient"]
        and checks["stacking_helps"]
        and checks["stacking_not_just_more_depth"]
        and checks["compute_reasonable"]
        and checks["training_speed_not_catastrophic"]
        and checks["baseline_not_saturated"]
    ):
        final_verdict = "STRONG_T_BLOCK_RESULT"
    elif (
        checks["tblock_beats_fixed_small"]
        and checks["tblock_ablations_hurt"]
        and (checks["stacking_helps"] or checks["stacking_not_just_more_depth"])
    ):
        final_verdict = "STACKING_PROMISING"
    elif checks["tblock_beats_fixed_small"] and checks["tblock_ablations_hurt"]:
        final_verdict = "SINGLE_BLOCK_PROMISING"
    else:
        final_verdict = "MECHANISM_WORKS_BUT_NO_VALUE"

    return {
        "checks": checks,
        "final_verdict": final_verdict,
        "selected_best_base_tblock": best_base.get("model", ""),
        "selected_best_depth_tblock": best_depth.get("model", ""),
        "thresholds": {
            "random_accuracy": random_accuracy,
            "effect_threshold": args.effect_threshold,
            "ablation_nll_threshold": args.ablation_nll_threshold,
        },
        "key_metrics": {
            "fixed_small_accuracy": fixed_acc,
            "fixed_small_nll": fixed_nll,
            "fixed_medium_accuracy": medium_acc,
            "best_base_tblock_accuracy": best_acc,
            "best_base_tblock_nll": best_nll,
            "best_base_tblock_latency_ratio": best_latency_ratio,
        },
        "ablation_summary": ablation_summary,
    }


def collect_code_audit_lines() -> list[str]:
    source_lines = inspect.getsource(TRealizedAttention.forward).splitlines()
    prefixes = [
        "T = torch.tanh",
        "T = T.masked_fill",
        "K_temp =",
        "V_temp =",
        "score = torch.einsum",
        "score = score.masked_fill",
        "attn = torch.softmax",
        "output = torch.einsum",
    ]
    audit_lines: list[str] = []
    for prefix in prefixes:
        for line in source_lines:
            stripped = line.strip()
            if stripped.startswith(prefix):
                audit_lines.append(stripped)
                break
    return audit_lines


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    numeric = _as_float(value)
    if numeric is None:
        return str(value)
    if abs(numeric) >= 1000:
        return f"{numeric:.0f}"
    return f"{numeric:.4f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    table = [
        "| " + " | ".join(label for label, _key in columns) + " |",
        "| " + " | ".join("---" for _label, _key in columns) + " |",
    ]
    for row in rows:
        table.append(
            "| "
            + " | ".join(_fmt(row.get(key, "")) for _label, key in columns)
            + " |"
        )
    return table


def make_report(
    *,
    args: argparse.Namespace,
    task_config: TBlockTaskConfig,
    train_config: TrainConfig,
    aggregate: list[dict[str, Any]],
    per_benchmark: list[dict[str, Any]],
    placement_rows: list[dict[str, Any]],
    ablation_summary: dict[str, dict[str, float]],
    mechanism_summary: dict[str, float],
    compute_summary: dict[str, Any],
    verdict: dict[str, Any],
) -> str:
    model_rows = [row for row in aggregate if row.get("model") in MODEL_VARIANTS]
    model_rows.sort(key=lambda row: MODEL_VARIANTS.index(row["model"]))
    model_table = markdown_table(
        model_rows,
        [
            ("model", "model"),
            ("acc", "mean_test_accuracy"),
            ("nll", "mean_test_nll"),
            ("params", "mean_parameter_count"),
            ("multadds", "mean_active_mult_adds_estimate"),
            ("lat ms/ex", "mean_latency_ms_per_example"),
        ],
    )

    placement_table = markdown_table(
        placement_rows[: min(len(placement_rows), 24)],
        [
            ("benchmark", "benchmark"),
            ("model", "model"),
            ("placement", "placement"),
            ("layers", "t_layers"),
            ("val acc", "validation_accuracy"),
            ("selected", "selected"),
        ],
    )

    stacking_rows = [
        row
        for row in model_rows
        if row.get("model")
        in {
            "fixed_small_transformer",
            "tblock_1",
            "tblock_2",
            "tblock_all",
            "fixed_small_4block",
            "tblock_4block_1",
            "tblock_4block_2",
            "tblock_4block_all",
        }
    ]
    stacking_table = markdown_table(
        stacking_rows,
        [
            ("model", "model"),
            ("acc", "mean_test_accuracy"),
            ("nll", "mean_test_nll"),
            ("lat ratio", "mean_latency_ratio_vs_fixed_small"),
        ],
    )

    v_only_rows = [
        row
        for row in model_rows
        if row.get("model")
        in {"tblock_1", "v_only_tblock_1", "k_only_tblock_1", "tblock_2", "v_only_tblock_2"}
    ]
    v_only_table = markdown_table(
        v_only_rows,
        [
            ("model", "model"),
            ("acc gain", "mean_accuracy_gain_vs_fixed_small"),
            ("nll gain", "mean_nll_gain_vs_fixed_small"),
            ("compute ratio", "mean_compute_ratio_vs_fixed_small"),
            ("lat ratio", "mean_latency_ratio_vs_fixed_small"),
        ],
    )

    ablation_rows = [
        {"ablation": name, **metrics} for name, metrics in sorted(ablation_summary.items())
    ]
    ablation_table = markdown_table(
        ablation_rows,
        [
            ("ablation", "ablation"),
            ("acc", "accuracy"),
            ("nll", "nll"),
            ("acc drop", "accuracy_drop"),
            ("nll inc", "nll_increase"),
        ],
    )

    mechanism_rows = [
        {"metric": key, "value": value} for key, value in mechanism_summary.items()
    ]
    mechanism_table = markdown_table(mechanism_rows, [("metric", "metric"), ("value", "value")])

    speed_rows = markdown_table(
        model_rows,
        [
            ("model", "model"),
            ("sec/epoch", "mean_seconds_per_epoch"),
            ("best epoch", "mean_best_val_epoch"),
            ("epoch to 70", "mean_epochs_to_reach_70_acc"),
            ("wall to 70", "mean_wall_clock_to_reach_70_acc"),
        ],
    )

    checks = verdict["checks"]
    answers = [
        (
            "Did adding T-realized blocks improve over fixed small?",
            checks["tblock_beats_fixed_small"],
        ),
        ("Did stacking T-realized blocks help?", checks["stacking_helps"]),
        (
            "Did T-blocks beat depth-matched normal blocks?",
            checks["stacking_not_just_more_depth"],
        ),
        ("Did V-only keep most of the gain?", checks["v_only_is_efficient"]),
        ("Did T ablations still hurt?", checks["tblock_ablations_hurt"]),
        ("Was compute/latency reasonable?", checks["compute_reasonable"]),
        (
            "Does this support adjustable attention blocks compressing more behavior?",
            checks["tblock_beats_fixed_small"]
            and checks["tblock_ablations_hurt"]
            and checks["baseline_not_saturated"],
        ),
    ]
    answer_lines = [f"- {question} {yes_no(answer)}." for question, answer in answers]

    return "\n".join(
        [
            "# T-Realized Attention Blocks Validation Report",
            "",
            "## 1. Plain-English Verdict",
            "",
            f"Final verdict: `{verdict['final_verdict']}`.",
            f"Best base-size T-block model: `{verdict['selected_best_base_tblock']}`.",
            f"Best depth-matched T-block model: `{verdict['selected_best_depth_tblock']}`.",
            "",
            "## 2. Exact Architecture Description",
            "",
            "`TRealizedBlockTransformer` is a causal sequence classifier with token and "
            "position embeddings, a stack of pre-norm transformer blocks, final LayerNorm, "
            "and a class readout. `StandardBlock` uses normal causal self-attention. "
            "`TRealizedBlock` replaces only self-attention with `TRealizedAttention` and "
            "keeps the MLP normal. There is no controller, no hypernetwork, and no dynamic "
            "effective-weight matrix construction.",
            "",
            "## 3. Code Audit Of T-Realized Attention Path",
            "",
            "```python",
            *collect_code_audit_lines(),
            "```",
            "",
            "## 4. Model Comparison Table",
            "",
            *model_table,
            "",
            "## 5. Placement Study",
            "",
            "Placement candidates were selected using validation metrics only.",
            "",
            *placement_table,
            "",
            "## 6. Stacking Study",
            "",
            *stacking_table,
            "",
            "## 7. V-Only Vs Full T Comparison",
            "",
            *v_only_table,
            "",
            "## 8. Ablation Table",
            "",
            *ablation_table,
            "",
            "## 9. Mechanism Metrics",
            "",
            *mechanism_table,
            "",
            "## 10. Compute And Latency",
            "",
            "```json",
            json.dumps(compute_summary, indent=2),
            "```",
            "",
            "## 11. Training Speed",
            "",
            *speed_rows,
            "",
            "## 12. Final Honest Interpretation",
            "",
            "`STRONG_T_BLOCK_RESULT` is only emitted when the T-blocks beat fixed small, "
            "approach medium, ablations hurt, stacking helps beyond depth alone, V-only "
            "is efficient, and latency/training cost are reasonable. Otherwise the lower "
            "verdicts identify whether the mechanism works without practical value or is "
            "only a single/stacking promise.",
            "",
            "## Required Answers",
            "",
            *answer_lines,
            "",
            "## Verdict JSON",
            "",
            "```json",
            json.dumps(verdict, indent=2),
            "```",
            "",
            "## Run Config",
            "",
            "```json",
            json.dumps(
                {
                    "task": asdict(task_config),
                    "train": asdict(train_config),
                    "model_sizes": MODEL_SIZES,
                    "benchmarks": BENCHMARKS,
                    "models": MODEL_VARIANTS,
                    "per_benchmark_rows": per_benchmark,
                },
                indent=2,
            ),
            "```",
        ]
    )


def run_experiment(args: argparse.Namespace) -> Path:
    if args.quick:
        args.seeds = "0"
        args.epochs = min(args.epochs, 1)
        args.placement_epochs = min(args.placement_epochs, 1)
        args.n_train = min(args.n_train, 128)
        args.n_val = min(args.n_val, 64)
        args.n_test = min(args.n_test, 64)
        args.batch_size = min(args.batch_size, 32)
        args.mechanism_batches = min(args.mechanism_batches, 1)

    seeds = parse_seeds(args.seeds)
    task_config = TBlockTaskConfig(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_rules=args.num_rules,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
    )
    train_config = TrainConfig(
        seeds=seeds,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        dropout=args.dropout,
        device=args.device,
    )
    benchmarks: list[BenchmarkName] = list(args.benchmarks)
    model_variants = list(dict.fromkeys(args.models))
    unknown_models = [model for model in model_variants if model not in MODEL_VARIANTS]
    if unknown_models:
        raise ValueError(f"unknown model variants: {unknown_models}")

    run_dir = (
        Path(args.output_root)
        / f"t_realized_blocks_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_blob = {
        "experiment_name": "t_realized_blocks_validation",
        "quick": args.quick,
        "task": asdict(task_config),
        "train": asdict(train_config),
        "benchmarks": benchmarks,
        "models": model_variants,
        "model_sizes": MODEL_SIZES,
        "placement_candidates": PLACEMENT_CANDIDATES,
        "ablations": T_BLOCK_ABLATIONS,
    }
    (run_dir / "RUN_CONFIG.json").write_text(
        json.dumps(config_blob, indent=2),
        encoding="utf-8",
    )

    device = torch.device(train_config.device)
    per_seed_rows: list[dict[str, Any]] = []
    placement_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []

    for seed in seeds:
        for benchmark_idx, benchmark in enumerate(benchmarks):
            loaders = make_dataloaders(task_config, seed=seed, benchmark=benchmark)
            num_groups = num_groups_for_benchmark(task_config, benchmark)
            current_placement_rows, selected_placements = placement_study(
                args,
                seed=seed,
                benchmark=benchmark,
                loaders=loaders,
                train_config=train_config,
                num_groups=num_groups,
            )
            placement_rows.extend(current_placement_rows)

            trained_full_tblocks: dict[str, tuple[nn.Module, dict[str, Any]]] = {}
            for variant_idx, variant in enumerate(model_variants):
                set_seed(seed * 1009 + benchmark_idx * 97 + variant_idx)
                model = build_model(
                    args,
                    variant=variant,
                    selected_placements=selected_placements,
                )
                model_config = build_model_config(
                    args,
                    variant=variant,
                    selected_placements=selected_placements,
                )
                train_stats = train_one_model(
                    model,
                    loaders,
                    train_config,
                    num_groups=num_groups,
                )
                val_metrics = evaluate_model(
                    model,
                    loaders["val"],
                    device=device,
                    num_groups=num_groups,
                )
                test_metrics = evaluate_model(
                    model,
                    loaders["test"],
                    device=device,
                    num_groups=num_groups,
                )
                row = {
                    "seed": seed,
                    "benchmark": benchmark,
                    "model": variant,
                    "selected_tblock_1_placement": selected_placements.get("tblock_1", ""),
                    "selected_tblock_2_placement": selected_placements.get("tblock_2", ""),
                    "t_layers": json.dumps(model_config.t_layers),
                    "t_mode": model_config.t_mode,
                    "validation_accuracy": val_metrics["accuracy"],
                    "validation_nll": val_metrics["nll"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_nll": test_metrics["nll"],
                    "random_accuracy": 1.0 / float(args.num_classes),
                    "per_rule_accuracy": json.dumps(
                        test_metrics["per_rule_accuracy"],
                        sort_keys=True,
                    ),
                    "parameter_count": count_parameters(model),
                    "active_mult_adds_estimate": estimate_active_mult_adds(model_config),
                    "latency_ms_per_example": test_metrics["latency_ms_per_example"],
                    **train_stats,
                }
                per_seed_rows.append(row)
                print(
                    f"{benchmark} seed={seed} model={variant} "
                    f"acc={test_metrics['accuracy']:.4f} "
                    f"nll={test_metrics['nll']:.4f}",
                    flush=True,
                )

                if model_config.t_layers:
                    mechanism = collect_mechanism_metrics(
                        model,
                        loaders["test"],
                        device=device,
                        max_batches=args.mechanism_batches,
                    )
                    mechanism_rows.append(
                        {
                            "seed": seed,
                            "benchmark": benchmark,
                            "model": variant,
                            **mechanism,
                        }
                    )
                if variant in FULL_TBLOCK_VARIANTS:
                    trained_full_tblocks[variant] = (model, row)

            for candidates in [BASE_FULL_TBLOCK_VARIANTS, DEPTH_FULL_TBLOCK_VARIANTS]:
                available = [
                    (variant, trained_full_tblocks[variant])
                    for variant in candidates
                    if variant in trained_full_tblocks
                ]
                if not available:
                    continue
                best_variant, (best_model, best_row) = max(
                    available,
                    key=lambda item: (
                        float(item[1][1]["validation_accuracy"]),
                        -float(item[1][1]["validation_nll"]),
                    ),
                )
                for ablation_name, ablation in T_BLOCK_ABLATIONS.items():
                    ablated = evaluate_model(
                        best_model,
                        loaders["test"],
                        device=device,
                        ablation=ablation,
                        num_groups=num_groups,
                    )
                    ablation_rows.append(
                        {
                            "seed": seed,
                            "benchmark": benchmark,
                            "base_model": best_variant,
                            "ablation": ablation_name,
                            "base_accuracy": best_row["test_accuracy"],
                            "base_nll": best_row["test_nll"],
                            "accuracy": ablated["accuracy"],
                            "nll": ablated["nll"],
                            "accuracy_drop": float(best_row["test_accuracy"])
                            - float(ablated["accuracy"]),
                            "nll_increase": float(ablated["nll"])
                            - float(best_row["test_nll"]),
                            "latency_ms_per_example": ablated["latency_ms_per_example"],
                        }
                    )

    enrich_value_metrics(per_seed_rows)
    aggregate = aggregate_rows(per_seed_rows, ["model"])
    per_benchmark = aggregate_rows(per_seed_rows, ["benchmark", "model"])
    ablation_summary = aggregate_ablation_rows(ablation_rows)
    mechanism_summary = aggregate_mechanism_rows(mechanism_rows)
    compute_summary = compute_metrics_summary(aggregate, per_benchmark)
    verdict = make_verdict(
        args=args,
        aggregate=aggregate,
        ablation_summary=ablation_summary,
    )
    report = make_report(
        args=args,
        task_config=task_config,
        train_config=train_config,
        aggregate=aggregate,
        per_benchmark=per_benchmark,
        placement_rows=placement_rows,
        ablation_summary=ablation_summary,
        mechanism_summary=mechanism_summary,
        compute_summary=compute_summary,
        verdict=verdict,
    )

    training_speed_rows = [
        {
            key: row.get(key)
            for key in [
                "seed",
                "benchmark",
                "model",
                "training_seconds_total",
                "seconds_per_epoch",
                "best_val_epoch",
                "epochs_to_reach_70_acc",
                "epochs_to_reach_75_acc",
                "wall_clock_to_reach_70_acc",
                "wall_clock_to_reach_75_acc",
            ]
        }
        for row in per_seed_rows
    ]

    write_csv(run_dir / "per_seed_metrics.csv", per_seed_rows)
    write_csv(run_dir / "aggregate_metrics.csv", aggregate)
    write_csv(run_dir / "per_benchmark_metrics.csv", per_benchmark)
    write_csv(run_dir / "ablation_metrics.csv", ablation_rows)
    write_csv(run_dir / "mechanism_metrics.csv", mechanism_rows)
    write_csv(run_dir / "training_speed_metrics.csv", training_speed_rows)
    write_csv(run_dir / "placement_study.csv", placement_rows)
    (run_dir / "compute_metrics.json").write_text(
        json.dumps(compute_summary, indent=2),
        encoding="utf-8",
    )
    (run_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2),
        encoding="utf-8",
    )
    (run_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-root", default="transformer_uncertainty/runs")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--placement-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--num-rules", type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--readout-position", type=int, default=-1)
    parser.add_argument("--sigma-init-gain", type=float, default=0.35)
    parser.add_argument("--effect-threshold", type=float, default=0.02)
    parser.add_argument("--ablation-nll-threshold", type=float, default=0.05)
    parser.add_argument("--mechanism-batches", type=int, default=2)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=BENCHMARKS,
        default=BENCHMARKS,
    )
    parser.add_argument("--models", nargs="+", default=MODEL_VARIANTS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = run_experiment(args)
    print(f"Wrote T-Realized Blocks validation artifacts to {run_dir}")


if __name__ == "__main__":
    main()
