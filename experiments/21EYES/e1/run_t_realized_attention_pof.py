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
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from .t_realized_attention import (
        TRAblation,
        FixedNormalTransformerClassifier,
        TRealizedAttention,
        TRealizedAttentionTransformer,
        TRealizedTransformerConfig,
        count_parameters,
        estimate_active_mult_adds,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from t_realized_attention import (
        TRAblation,
        FixedNormalTransformerClassifier,
        TRealizedAttention,
        TRealizedAttentionTransformer,
        TRealizedTransformerConfig,
        count_parameters,
        estimate_active_mult_adds,
    )


MODEL_VARIANTS = ["fixed_small_transformer", "t_realized_transformer"]

ABLATIONS: dict[str, TRAblation] = {
    "t_zero": "t_zero",
    "k_temp_only": "k_temp_only",
    "v_temp_only": "v_temp_only",
    "shuffle_T": "shuffle_T",
    "random_T": "random_T",
    "sigma_zero": "sigma_zero",
    "fixed_normal_attention": "fixed_normal_attention",
}


@dataclass(frozen=True)
class RuleSwitchingTaskConfig:
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _labels_by_rule(values: Tensor, num_classes: int) -> Tensor:
    payload_len = values.shape[1]
    reverse_position = max(0, payload_len - 2)
    rules = torch.stack(
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
    return rules


def make_rule_switching_split(
    n_examples: int,
    *,
    config: RuleSwitchingTaskConfig,
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


def make_dataloaders(
    config: RuleSwitchingTaskConfig,
    *,
    seed: int,
) -> dict[str, DataLoader]:
    train = make_rule_switching_split(
        config.n_train,
        config=config,
        seed=seed * 1009 + 17,
    )
    val = make_rule_switching_split(
        config.n_val,
        config=config,
        seed=seed * 1009 + 29,
    )
    test = make_rule_switching_split(
        config.n_test,
        config=config,
        seed=seed * 1009 + 43,
    )
    return {
        "train": DataLoader(train, batch_size=config.batch_size, shuffle=True),
        "val": DataLoader(val, batch_size=config.batch_size, shuffle=False),
        "test": DataLoader(test, batch_size=config.batch_size, shuffle=False),
    }


def build_model_config(args: argparse.Namespace) -> TRealizedTransformerConfig:
    return TRealizedTransformerConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        readout_position=args.readout_position,
        sigma_init_gain=args.sigma_init_gain,
    )


def build_model(args: argparse.Namespace, *, variant: str) -> nn.Module:
    config = build_model_config(args)
    if variant == "t_realized_transformer":
        return TRealizedAttentionTransformer(config)
    if variant == "fixed_small_transformer":
        return FixedNormalTransformerClassifier(config)
    raise ValueError(f"unknown model variant: {variant}")


def model_kind_for_variant(variant: str) -> str:
    if variant == "t_realized_transformer":
        return "t_realized"
    return "fixed"


def logits_from_output(output: Tensor | dict[str, Any]) -> Tensor:
    if isinstance(output, dict):
        return output["logits"]
    return output


def train_one_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    train_config: TrainConfig,
) -> dict[str, float]:
    device = torch.device(train_config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    best_val_nll = float("inf")
    best_state: dict[str, Tensor] | None = None
    started = time.perf_counter()

    for _epoch in range(train_config.epochs):
        model.train()
        for tokens, labels, _rules in loaders["train"]:
            tokens = tokens.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(tokens, return_details=False)
            logits = logits_from_output(output)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()

        val_metrics = evaluate_model(model, loaders["val"], device=device)
        if val_metrics["nll"] < best_val_nll:
            best_val_nll = val_metrics["nll"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "train_seconds": time.perf_counter() - started,
        "best_val_nll": best_val_nll,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    ablation: TRAblation = "none",
    num_rules: int | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    if num_rules is None:
        num_rules = 0
    correct_by_rule = torch.zeros(num_rules, dtype=torch.long)
    count_by_rule = torch.zeros(num_rules, dtype=torch.long)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for tokens, labels, rules in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        output = model(tokens, ablation=ablation, return_details=False)
        logits = logits_from_output(output)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        predictions = logits.argmax(dim=-1)
        correct = predictions.eq(labels)

        batch_size = labels.numel()
        total_loss += float(loss.item())
        total_correct += int(correct.sum().item())
        total_examples += batch_size

        if num_rules:
            for rule in range(num_rules):
                mask = rules.eq(rule)
                count = int(mask.sum().item())
                if count:
                    count_by_rule[rule] += count
                    correct_by_rule[rule] += int(correct.cpu()[mask].sum().item())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    per_rule_accuracy = {}
    for rule in range(num_rules):
        if count_by_rule[rule].item():
            per_rule_accuracy[f"rule_{rule}"] = (
                correct_by_rule[rule].item() / count_by_rule[rule].item()
            )
        else:
            per_rule_accuracy[f"rule_{rule}"] = float("nan")

    return {
        "accuracy": total_correct / max(1, total_examples),
        "nll": total_loss / max(1, total_examples),
        "examples": total_examples,
        "latency_ms_per_example": 1000.0 * elapsed / max(1, total_examples),
        "per_rule_accuracy": per_rule_accuracy,
    }


def _mean(values: list[Tensor]) -> float:
    if not values:
        return 0.0
    return float(torch.stack([value.detach().float().cpu() for value in values]).mean().item())


@torch.no_grad()
def collect_mechanism_metrics(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int = 2,
) -> dict[str, float]:
    model.eval()
    t_means: list[Tensor] = []
    t_stds: list[Tensor] = []
    t_variances: list[Tensor] = []
    k_temp_diffs: list[Tensor] = []
    v_temp_diffs: list[Tensor] = []
    attention_diffs: list[Tensor] = []
    t_zero_output_diffs: list[Tensor] = []
    sigma_zero_output_diffs: list[Tensor] = []

    for batch_idx, (tokens, _labels, _rules) in enumerate(loader):
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

            if idx < len(t_zero_diagnostics):
                zero_heads = t_zero_diagnostics[idx]["output_heads"].float()
                t_zero_output_diffs.append((output_heads - zero_heads).norm(dim=-1).mean())
            if idx < len(sigma_zero_diagnostics):
                sigma_zero_heads = sigma_zero_diagnostics[idx]["output_heads"].float()
                sigma_zero_output_diffs.append(
                    (output_heads - sigma_zero_heads).norm(dim=-1).mean()
                )

    return {
        "T_mean": _mean(t_means),
        "T_std": _mean(t_stds),
        "T_variance": _mean(t_variances),
        "K_temp_difference_norm": _mean(k_temp_diffs),
        "V_temp_difference_norm": _mean(v_temp_diffs),
        "attention_difference_vs_normal": _mean(attention_diffs),
        "output_difference_when_T_zero": _mean(t_zero_output_diffs),
        "output_difference_when_sigma_zero": _mean(sigma_zero_output_diffs),
    }


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


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate_rows = []
    numeric_keys = [
        "validation_accuracy",
        "validation_nll",
        "test_accuracy",
        "test_nll",
        "train_seconds",
        "latency_ms_per_example",
        "parameter_count",
        "active_mult_adds_estimate",
    ]
    for variant in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == variant]
        aggregate = {"model": variant, "seeds": len(subset)}
        for key in numeric_keys:
            values = [float(row[key]) for row in subset if key in row]
            aggregate[f"mean_{key}"] = sum(values) / len(values) if values else 0.0
        aggregate_rows.append(aggregate)
    return aggregate_rows


def aggregate_ablation_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in sorted({row["ablation"] for row in rows}):
        subset = [row for row in rows if row["ablation"] == name]
        result[name] = {
            "accuracy": sum(float(row["accuracy"]) for row in subset) / len(subset),
            "nll": sum(float(row["nll"]) for row in subset) / len(subset),
            "accuracy_drop": sum(float(row["accuracy_drop"]) for row in subset)
            / len(subset),
            "nll_increase": sum(float(row["nll_increase"]) for row in subset)
            / len(subset),
        }
    return result


def aggregate_mechanism_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [key for key in rows[0] if key not in {"seed", "model"}]
    return {
        key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
        for key in keys
    }


def compute_metrics_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_metric_rows(rows)
    models: dict[str, dict[str, float]] = {}
    for row in aggregate:
        models[row["model"]] = {
            "parameter_count": row["mean_parameter_count"],
            "active_mult_adds_estimate": row["mean_active_mult_adds_estimate"],
            "latency_ms_per_example": row["mean_latency_ms_per_example"],
        }
    fixed_small = models.get("fixed_small_transformer", {})
    comparisons = {}
    for model, values in models.items():
        comparisons[model] = {
            "parameter_ratio_to_fixed_small": values["parameter_count"]
            / max(1.0, fixed_small.get("parameter_count", 1.0)),
            "compute_ratio_to_fixed_small": values["active_mult_adds_estimate"]
            / max(1.0, fixed_small.get("active_mult_adds_estimate", 1.0)),
            "latency_ratio_to_fixed_small": values["latency_ms_per_example"]
            / max(1e-9, fixed_small.get("latency_ms_per_example", 1e-9)),
        }
    return {
        "models": models,
        "compare_to_fixed_small": comparisons,
    }


def _mean_for_model(
    aggregate_rows: list[dict[str, Any]],
    model: str,
    key: str,
) -> float:
    for row in aggregate_rows:
        if row["model"] == model:
            return float(row[key])
    return float("nan")


def _ablation_metric(
    ablation_summary: dict[str, dict[str, float]],
    name: str,
    key: str,
) -> float:
    return float(ablation_summary.get(name, {}).get(key, float("nan")))


def collect_code_audit_lines() -> list[str]:
    source_lines = inspect.getsource(TRealizedAttention.forward).splitlines()
    prefixes = [
        "T = torch.tanh",
        "K_temp =",
        "V_temp =",
        "score = torch.einsum",
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


def make_verdict(
    *,
    args: argparse.Namespace,
    aggregate_rows: list[dict[str, Any]],
    ablation_summary: dict[str, dict[str, float]],
    mechanism_summary: dict[str, float],
) -> dict[str, Any]:
    random_accuracy = 1.0 / float(args.num_classes)
    t_accuracy = _mean_for_model(
        aggregate_rows,
        "t_realized_transformer",
        "mean_test_accuracy",
    )
    t_nll = _mean_for_model(
        aggregate_rows,
        "t_realized_transformer",
        "mean_test_nll",
    )
    fixed_small_accuracy = _mean_for_model(
        aggregate_rows,
        "fixed_small_transformer",
        "mean_test_accuracy",
    )
    fixed_small_nll = _mean_for_model(
        aggregate_rows,
        "fixed_small_transformer",
        "mean_test_nll",
    )

    def hurts(name: str) -> bool:
        metrics = ablation_summary.get(name, {})
        return (
            metrics.get("accuracy_drop", 0.0) >= args.effect_threshold
            or metrics.get("nll_increase", 0.0) >= args.nll_threshold
        )

    t_zero_accuracy = _ablation_metric(ablation_summary, "t_zero", "accuracy")
    sigma_zero_accuracy = _ablation_metric(ablation_summary, "sigma_zero", "accuracy")
    shuffle_t_accuracy = _ablation_metric(ablation_summary, "shuffle_T", "accuracy")
    random_t_accuracy = _ablation_metric(ablation_summary, "random_T", "accuracy")
    k_temp_only_accuracy = _ablation_metric(ablation_summary, "k_temp_only", "accuracy")
    v_temp_only_accuracy = _ablation_metric(ablation_summary, "v_temp_only", "accuracy")
    fixed_normal_accuracy = _ablation_metric(
        ablation_summary,
        "fixed_normal_attention",
        "accuracy",
    )
    fixed_normal_nll = _ablation_metric(
        ablation_summary,
        "fixed_normal_attention",
        "nll",
    )

    task_learns = t_accuracy >= random_accuracy + 0.08 or t_nll <= (
        math.log(args.num_classes) - 0.08
    )
    fixed_normal_attention_worse_or_equal = (
        fixed_normal_accuracy <= t_accuracy + args.equal_accuracy_tolerance
        and fixed_normal_nll >= t_nll - args.equal_nll_tolerance
    )

    sample_model = TRealizedAttentionTransformer(build_model_config(args))
    no_controller_present = not any(
        "controller" in type(module).__name__.lower()
        for module in sample_model.modules()
    ) and not hasattr(sample_model, "controller")
    no_hypernetwork_present = not any(
        "hyper" in type(module).__name__.lower()
        for module in sample_model.modules()
    )

    checks = {
        "task_learns": bool(task_learns),
        "t_zero_hurts": bool(hurts("t_zero")),
        "sigma_zero_hurts": bool(hurts("sigma_zero")),
        "shuffle_T_hurts": bool(hurts("shuffle_T")),
        "fixed_normal_attention_worse_or_equal": bool(
            fixed_normal_attention_worse_or_equal
        ),
        "T_affects_K": bool(
            mechanism_summary.get("K_temp_difference_norm", 0.0)
            >= args.mechanism_diff_threshold
        ),
        "T_affects_V": bool(
            mechanism_summary.get("V_temp_difference_norm", 0.0)
            >= args.mechanism_diff_threshold
        ),
        "no_controller_present": bool(no_controller_present),
        "no_hypernetwork_present": bool(no_hypernetwork_present),
        "implementation_path_clean": bool(implementation_path_clean()),
    }

    required_pass = [
        "task_learns",
        "t_zero_hurts",
        "sigma_zero_hurts",
        "shuffle_T_hurts",
        "fixed_normal_attention_worse_or_equal",
        "T_affects_K",
        "T_affects_V",
        "no_controller_present",
        "no_hypernetwork_present",
        "implementation_path_clean",
    ]
    if not (
        checks["implementation_path_clean"]
        and checks["no_controller_present"]
        and checks["no_hypernetwork_present"]
    ):
        final_verdict = "FAILED_IMPLEMENTATION"
    elif all(checks[name] for name in required_pass):
        final_verdict = "PASSED_T_REALIZED_ATTENTION_POF"
    elif task_learns:
        final_verdict = "LEARNS_BUT_MECHANISM_UNUSED"
    else:
        final_verdict = "FAILED_POF"

    return {
        "checks": checks,
        "final_verdict": final_verdict,
        "thresholds": {
            "random_accuracy": random_accuracy,
            "accuracy_effect_threshold": args.effect_threshold,
            "nll_effect_threshold": args.nll_threshold,
            "mechanism_diff_threshold": args.mechanism_diff_threshold,
            "equal_accuracy_tolerance": args.equal_accuracy_tolerance,
            "equal_nll_tolerance": args.equal_nll_tolerance,
        },
        "key_metrics": {
            "test_accuracy": t_accuracy,
            "test_nll": t_nll,
            "random_accuracy": random_accuracy,
            "fixed_small_accuracy": fixed_small_accuracy,
            "fixed_small_nll": fixed_small_nll,
            "t_realized_accuracy": t_accuracy,
            "t_realized_nll": t_nll,
            "t_zero_accuracy": t_zero_accuracy,
            "sigma_zero_accuracy": sigma_zero_accuracy,
            "shuffle_T_accuracy": shuffle_t_accuracy,
            "random_T_accuracy": random_t_accuracy,
            "k_temp_only_accuracy": k_temp_only_accuracy,
            "v_temp_only_accuracy": v_temp_only_accuracy,
            "fixed_normal_attention_accuracy": fixed_normal_accuracy,
        },
        "ablation_summary": ablation_summary,
        "mechanism_summary": mechanism_summary,
    }


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def make_report(
    *,
    args: argparse.Namespace,
    task_config: RuleSwitchingTaskConfig,
    train_config: TrainConfig,
    aggregate_rows: list[dict[str, Any]],
    ablation_summary: dict[str, dict[str, float]],
    mechanism_summary: dict[str, float],
    compute_summary: dict[str, Any],
    verdict: dict[str, Any],
) -> str:
    model_table = [
        "| model | test_accuracy | test_nll | parameters | active_mult_adds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(aggregate_rows, key=lambda item: item["model"]):
        model_table.append(
            "| {model} | {acc:.4f} | {nll:.4f} | {params:.0f} | {flops:.0f} |".format(
                model=row["model"],
                acc=row["mean_test_accuracy"],
                nll=row["mean_test_nll"],
                params=row["mean_parameter_count"],
                flops=row["mean_active_mult_adds_estimate"],
            )
        )

    ablation_table = [
        "| ablation | accuracy | nll | accuracy_drop | nll_increase |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in sorted(ablation_summary.items()):
        ablation_table.append(
            "| {name} | {accuracy:.4f} | {nll:.4f} | {accuracy_drop:.4f} | {nll_increase:.4f} |".format(
                name=name,
                accuracy=metrics["accuracy"],
                nll=metrics["nll"],
                accuracy_drop=metrics["accuracy_drop"],
                nll_increase=metrics["nll_increase"],
            )
        )

    mechanism_table = [
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in mechanism_summary.items():
        mechanism_table.append(f"| {key} | {value:.6f} |")

    checks_table = [
        "| required pass condition | passed |",
        "|---|---:|",
    ]
    for key, value in verdict["checks"].items():
        checks_table.append(f"| {key} | {yes_no(bool(value))} |")

    audit_lines = collect_code_audit_lines()
    audit_block = ["```python", *audit_lines, "```"]

    return "\n".join(
        [
            "# T-Realized Attention POF Report",
            "",
            f"Final verdict: `{verdict['final_verdict']}`",
            "",
            "## Run Config",
            "",
            "```json",
            json.dumps(
                {
                    "task": asdict(task_config),
                    "train": asdict(train_config),
                    "model": asdict(build_model_config(args)),
                    "ablations": ABLATIONS,
                },
                indent=2,
            ),
            "```",
            "",
            f"Random accuracy: {verdict['thresholds']['random_accuracy']:.4f}",
            "",
            "## Model Metrics",
            "",
            *model_table,
            "",
            "## Ablation Metrics",
            "",
            *ablation_table,
            "",
            "## Mechanism Metrics",
            "",
            *mechanism_table,
            "",
            "## Required Pass Conditions",
            "",
            *checks_table,
            "",
            "## Compute Metrics",
            "",
            "```json",
            json.dumps(compute_summary, indent=2),
            "```",
            "",
            "## Mechanism Code Audit",
            "",
            "The implementation in `t_realized_attention.py` uses the required path: "
            "`T = tanh(Q @ K_point.T)`, then realized `K_temp` and `V_temp`, "
            "then final attention over `K_temp` and weighted output from `V_temp`.",
            "",
            *audit_block,
            "",
            "There is no separate controller module, no hypernetwork, and no full "
            "dynamic effective-weight matrix materialization in `TRealizedAttention`.",
            "",
            "## Verdict JSON",
            "",
            "```json",
            json.dumps(verdict, indent=2),
            "```",
        ]
    )


def run_experiment(args: argparse.Namespace) -> Path:
    if args.quick:
        args.seeds = "0"
        args.epochs = min(args.epochs, 2)
        args.n_train = min(args.n_train, 768)
        args.n_val = min(args.n_val, 256)
        args.n_test = min(args.n_test, 256)

    seeds = parse_seeds(args.seeds)
    task_config = RuleSwitchingTaskConfig(
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
        device=args.device,
    )

    run_dir = (
        Path(args.output_root)
        / f"t_realized_attention_pof_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "RUN_CONFIG.json").write_text(
        json.dumps(
            {
                "task": asdict(task_config),
                "train": asdict(train_config),
                "model_variants": MODEL_VARIANTS,
                "model": asdict(build_model_config(args)),
                "ablations": ABLATIONS,
                "quick": args.quick,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    per_seed_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    device = torch.device(train_config.device)

    for seed in seeds:
        loaders = make_dataloaders(task_config, seed=seed)
        for variant in MODEL_VARIANTS:
            set_seed(seed * 1009 + MODEL_VARIANTS.index(variant))
            model = build_model(args, variant=variant)
            model_config = build_model_config(args)
            model_kind = model_kind_for_variant(variant)
            train_metrics = train_one_model(model, loaders, train_config)
            val_metrics = evaluate_model(
                model,
                loaders["val"],
                device=device,
                num_rules=args.num_rules,
            )
            test_metrics = evaluate_model(
                model,
                loaders["test"],
                device=device,
                num_rules=args.num_rules,
            )
            parameter_count = count_parameters(model)
            active_mult_adds = estimate_active_mult_adds(
                model_config,
                model_kind=model_kind,  # type: ignore[arg-type]
            )
            row = {
                "seed": seed,
                "model": variant,
                "validation_accuracy": val_metrics["accuracy"],
                "validation_nll": val_metrics["nll"],
                "test_accuracy": test_metrics["accuracy"],
                "test_nll": test_metrics["nll"],
                "random_accuracy": 1.0 / float(args.num_classes),
                "per_rule_accuracy_json": json.dumps(
                    test_metrics["per_rule_accuracy"],
                    sort_keys=True,
                ),
                "parameter_count": parameter_count,
                "active_mult_adds_estimate": active_mult_adds,
                "latency_ms_per_example": test_metrics["latency_ms_per_example"],
                **train_metrics,
            }
            per_seed_rows.append(row)
            print(
                f"seed={seed} model={variant} "
                f"test_acc={test_metrics['accuracy']:.4f} "
                f"test_nll={test_metrics['nll']:.4f}",
                flush=True,
            )

            if variant == "t_realized_transformer":
                mechanism = collect_mechanism_metrics(
                    model,
                    loaders["test"],
                    device=device,
                    max_batches=args.mechanism_batches,
                )
                mechanism_rows.append({"seed": seed, "model": variant, **mechanism})
                base_accuracy = test_metrics["accuracy"]
                base_nll = test_metrics["nll"]
                for ablation_name, ablation in ABLATIONS.items():
                    ablated = evaluate_model(
                        model,
                        loaders["test"],
                        device=device,
                        ablation=ablation,
                        num_rules=args.num_rules,
                    )
                    ablation_rows.append(
                        {
                            "seed": seed,
                            "ablation": ablation_name,
                            "accuracy": ablated["accuracy"],
                            "nll": ablated["nll"],
                            "accuracy_drop": base_accuracy - ablated["accuracy"],
                            "nll_increase": ablated["nll"] - base_nll,
                            "latency_ms_per_example": ablated[
                                "latency_ms_per_example"
                            ],
                        }
                    )

    aggregate_rows = aggregate_metric_rows(per_seed_rows)
    ablation_summary = aggregate_ablation_rows(ablation_rows)
    mechanism_summary = aggregate_mechanism_rows(mechanism_rows)
    compute_summary = compute_metrics_summary(per_seed_rows)
    verdict = make_verdict(
        args=args,
        aggregate_rows=aggregate_rows,
        ablation_summary=ablation_summary,
        mechanism_summary=mechanism_summary,
    )
    report = make_report(
        args=args,
        task_config=task_config,
        train_config=train_config,
        aggregate_rows=aggregate_rows,
        ablation_summary=ablation_summary,
        mechanism_summary=mechanism_summary,
        compute_summary=compute_summary,
        verdict=verdict,
    )

    write_csv(run_dir / "per_seed_metrics.csv", per_seed_rows)
    write_csv(run_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(run_dir / "ablation_metrics.csv", ablation_rows)
    write_csv(run_dir / "mechanism_metrics.csv", mechanism_rows)
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
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--readout-position", type=int, default=-1)
    parser.add_argument("--sigma-init-gain", type=float, default=0.35)
    parser.add_argument("--effect-threshold", type=float, default=0.02)
    parser.add_argument("--nll-threshold", type=float, default=0.03)
    parser.add_argument("--mechanism-diff-threshold", type=float, default=1e-4)
    parser.add_argument("--equal-accuracy-tolerance", type=float, default=0.005)
    parser.add_argument("--equal-nll-tolerance", type=float, default=0.01)
    parser.add_argument("--mechanism-batches", type=int, default=2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = run_experiment(args)
    print(f"Wrote T-Realized Attention POF artifacts to {run_dir}")


if __name__ == "__main__":
    main()

