"""Train and evaluate the E29 strict uncertainty-geometry experiment."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[5]
ROOT_DIR = Path(__file__).resolve().parents[2]
VARIANT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(VARIANT_DIR))

from experiments.DIGIT.Extrapolation.e29.extrapolation.data.trace_dataset import (
    TraceEntropyDataset,
    load_trace_records,
    trace_collate_fn,
)
from experiments.DIGIT.Extrapolation.e29.extrapolation.data.vocabulary import Vocabulary
from experiments.DIGIT.Extrapolation.e29.extrapolation.trace_pipeline import (
    ensure_trace_corpus,
    evaluate_trace_baselines,
    set_global_determinism,
)
from experiments.DIGIT.Extrapolation.e29.extrapolation.config import E29Config
from experiments.DIGIT.Extrapolation.e29.extrapolation.losses import E29Loss
from experiments.DIGIT.Extrapolation.e29.extrapolation.metrics import compute_outcome_metrics, summarize_tensor
from experiments.DIGIT.Extrapolation.e29.extrapolation.models.digit import E29Model


def _move_batch(batch, device: torch.device):
    queries, trace_inputs, evidence_targets, prim_targets, target_ids = batch
    del target_ids
    return (
        queries.to(device),
        trace_inputs.to(device),
        evidence_targets.to(device),
        prim_targets.to(device),
    )


def _limit_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or len(records) <= limit:
        return records
    return records[:limit]


def _config_field_names() -> set[str]:
    return {field.name for field in fields(E29Config)}


def _parse_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _apply_config_overrides(config: E29Config, overrides: list[str] | None) -> None:
    if not overrides:
        return
    valid_fields = _config_field_names()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected NAME=VALUE")
        name, raw_value = item.split("=", 1)
        if name not in valid_fields:
            raise ValueError(f"Unknown config field {name!r}")
        setattr(config, name, _parse_value(raw_value))


def _resolve_trace_dir(config: E29Config) -> Path:
    trace_dir = Path(config.trace_dir)
    if trace_dir.is_absolute():
        return trace_dir
    return (VARIANT_DIR / trace_dir).resolve()


def _make_loader(records: list[dict[str, Any]], config: E29Config, shuffle: bool) -> DataLoader:
    dataset = TraceEntropyDataset(records, vocab=Vocabulary(), config=config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=trace_collate_fn,
    )


def _evaluate(
    model: E29Model,
    criterion: E29Loss,
    loader: DataLoader,
    device: torch.device,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    logits = []
    targets = []
    margins = []
    uncertainties = []
    radii = []
    mean_vars = []
    agreements = []
    probe_order = []
    probe_boundary = []
    probe_support = []

    with torch.no_grad():
        for batch in loader:
            queries, trace_inputs, evidence_targets, prim_targets = _move_batch(batch, device)
            outputs = model(queries, trace_inputs=trace_inputs)
            losses = criterion(outputs, prim_targets, evidence_targets)
            batch_size = queries.size(0)
            total_loss += float(losses["total"].item()) * batch_size
            total_items += batch_size

            logits.append(outputs.outcome_logits.cpu())
            targets.append(prim_targets[:, 3].cpu())
            margins.append(outputs.margin.cpu())
            uncertainties.append(outputs.uncertainty.cpu())
            radii.append(outputs.cloud_stats.radius.cpu())
            mean_vars.append(outputs.cloud_stats.mean_var.cpu())
            agreements.append(outputs.cloud_stats.agreement.cpu())

            if outputs.diagnostic_probes is not None:
                probe_order.append(outputs.diagnostic_probes.order.cpu())
                probe_boundary.append(outputs.diagnostic_probes.boundary.cpu())
                probe_support.append(outputs.diagnostic_probes.support.cpu())

    logits_t = torch.cat(logits, dim=0)
    targets_t = torch.cat(targets, dim=0)
    bundle = {
        "outcome_logits": logits_t,
        "outcome_targets": targets_t,
        "margin": torch.cat(margins, dim=0),
        "uncertainty": torch.cat(uncertainties, dim=0),
        "radius": torch.cat(radii, dim=0),
        "mean_var": torch.cat(mean_vars, dim=0),
        "agreement": torch.cat(agreements, dim=0),
    }
    metrics = compute_outcome_metrics(logits_t, targets_t, records=records)
    metrics["loss"] = total_loss / max(total_items, 1)
    metrics["geometry"] = {
        "margin": summarize_tensor(bundle["margin"]),
        "uncertainty": summarize_tensor(bundle["uncertainty"]),
        "radius": summarize_tensor(bundle["radius"]),
        "mean_var": summarize_tensor(bundle["mean_var"]),
        "agreement": summarize_tensor(bundle["agreement"]),
    }
    if probe_order:
        bundle["probe_order"] = torch.cat(probe_order, dim=0)
        bundle["probe_boundary"] = torch.cat(probe_boundary, dim=0)
        bundle["probe_support"] = torch.cat(probe_support, dim=0)
        metrics["probes"] = {
            "order": summarize_tensor(bundle["probe_order"]),
            "boundary": summarize_tensor(bundle["probe_boundary"]),
            "support": summarize_tensor(bundle["probe_support"]),
        }
    return metrics, bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--rebuild-traces", action="store_true")
    parser.add_argument("--strict-existing-traces", action="store_true")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-val-records", type=int, default=None)
    parser.add_argument("--max-test-records", type=int, default=None)
    parser.add_argument("--cloud-samples", type=int, default=None)
    parser.add_argument("--enable-diagnostic-probes", action="store_true")
    parser.add_argument("--disable-radius-term", action="store_true")
    parser.add_argument("--disable-variance-term", action="store_true")
    parser.add_argument("--disable-margin-term", action="store_true")
    parser.add_argument("--config-override", action="append", default=None)
    args = parser.parse_args()

    config = E29Config()
    _apply_config_overrides(config, args.config_override)
    config.train_loop_seed = int(args.seed)
    if args.cloud_samples is not None:
        config.cloud_num_samples = int(args.cloud_samples)
    if args.enable_diagnostic_probes:
        config.enable_diagnostic_probes = True
    if args.disable_radius_term:
        config.geometry_use_radius_term = False
    if args.disable_variance_term:
        config.geometry_use_variance_term = False
    if args.disable_margin_term:
        config.geometry_use_margin_term = False

    output_dir = Path(args.output_dir) if args.output_dir else Path(config.output_root) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    set_global_determinism(config.train_loop_seed)

    if args.rebuild_traces:
        raise NotImplementedError(
            "--rebuild-traces is not supported in standalone e29. Use the existing local e29 trace_cache."
        )

    trace_dir = _resolve_trace_dir(config)
    trace_info = ensure_trace_corpus(
        config,
        root_dir=trace_dir,
        device=device,
        rebuild=args.rebuild_traces,
        strict_existing=args.strict_existing_traces,
    )

    train_records = _limit_records(load_trace_records(trace_dir / "train.jsonl"), args.max_train_records)
    val_records = _limit_records(load_trace_records(trace_dir / "val.jsonl"), args.max_val_records)
    test_records = _limit_records(load_trace_records(trace_dir / "test.jsonl"), args.max_test_records)

    train_loader = _make_loader(train_records, config, shuffle=True)
    val_loader = _make_loader(val_records, config, shuffle=False)
    test_loader = _make_loader(test_records, config, shuffle=False)

    train_dataset = TraceEntropyDataset(train_records, vocab=Vocabulary(), config=config)
    outcome_weights = train_dataset.get_class_weights(
        power=config.trace_class_weight_power,
        clip=config.trace_class_weight_clip,
    )["outcome"]

    model = E29Model(config).to(device)
    criterion = E29Loss(config, outcome_class_weights=outcome_weights).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val_metric = float("-inf")
    best_val_metrics: dict[str, Any] = {}
    patience = 0
    epoch_history = []

    for epoch in range(1, config.train_loop_epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        for batch in train_loader:
            queries, trace_inputs, evidence_targets, prim_targets = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(queries, trace_inputs=trace_inputs)
            losses = criterion(outputs, prim_targets, evidence_targets)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
            optimizer.step()

            batch_size = queries.size(0)
            total_loss += float(losses["total"].item()) * batch_size
            total_items += batch_size

        train_metrics, _ = _evaluate(model, criterion, train_loader, device, train_records)
        val_metrics, _ = _evaluate(model, criterion, val_loader, device, val_records)
        train_metrics["loss"] = total_loss / max(total_items, 1)
        epoch_history.append(
            {
                "epoch": epoch,
                "train_outcome_acc": train_metrics["outcome_acc"],
                "train_outcome_macro_f1": train_metrics["outcome_macro_f1"],
                "val_outcome_acc": val_metrics["outcome_acc"],
                "val_outcome_macro_f1": val_metrics["outcome_macro_f1"],
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
            }
        )
        print(
            f"epoch={epoch:02d} "
            f"train_acc={train_metrics['outcome_acc']:.3f} "
            f"train_f1={train_metrics['outcome_macro_f1']:.3f} "
            f"val_acc={val_metrics['outcome_acc']:.3f} "
            f"val_f1={val_metrics['outcome_macro_f1']:.3f}"
        )

        selection_metric = float(val_metrics["outcome_macro_f1"])
        if selection_metric > best_val_metric:
            best_val_metric = selection_metric
            best_val_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config.train_loop_patience:
                break

    model.load_state_dict(best_state)
    test_metrics, test_bundle = _evaluate(model, criterion, test_loader, device, test_records)
    baselines = evaluate_trace_baselines(train_records, test_records, config=config)

    metrics = {
        "experiment": config.experiment_name,
        "trace_info": trace_info,
        "config": asdict(config),
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "baselines": baselines,
        "epoch_history": epoch_history,
    }
    summary = {
        "experiment": config.experiment_name,
        "decision_rule": {
            "margin": "m = linear(mu)",
            "uncertainty": "u = softplus(alpha * radius + beta * mean(var) - gamma * abs(m) + b)",
            "logits": {
                "success": "m - u",
                "uncertain": "u",
                "failure": "-m - u",
            },
        },
        "ablation_toggles": {
            "geometry_use_radius_term": config.geometry_use_radius_term,
            "geometry_use_variance_term": config.geometry_use_variance_term,
            "geometry_use_margin_term": config.geometry_use_margin_term,
            "enable_diagnostic_probes": config.enable_diagnostic_probes,
            "cloud_num_samples": config.cloud_num_samples,
        },
        "test_metrics": test_metrics,
        "cloud_geometry": {
            "margin": summarize_tensor(test_bundle["margin"]),
            "uncertainty": summarize_tensor(test_bundle["uncertainty"]),
            "radius": summarize_tensor(test_bundle["radius"]),
            "mean_var": summarize_tensor(test_bundle["mean_var"]),
            "agreement": summarize_tensor(test_bundle["agreement"]),
        },
        "baselines": baselines,
    }

    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(config)}, output_dir / "model.pt")

    print(f"Wrote {output_dir / 'metrics.json'}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
