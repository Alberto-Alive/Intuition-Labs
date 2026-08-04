"""Trace-driven staged training and evaluation for DIGIT Extrapolation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
VARIANT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(VARIANT_DIR))

from eval_common import (
    audit_monotonicity,
    build_prediction_records,
    compute_label_audit,
    compute_success_safety_metrics,
    compute_threshold_occupancy,
    compute_threshold_sensitivity,
    evaluate_correctness_baselines,
    write_json,
    write_jsonl,
)
from extrapolation.config import Config
from extrapolation.data.ground_truth import (
    ATTENTION_PATTERN_LABELS,
    CONFIDENCE_LABELS,
    OUTCOME_LABELS,
    TRAJECTORY_SHAPE_LABELS,
)
from extrapolation.data.trace_dataset import TraceEntropyDataset, load_trace_records, trace_collate_fn
from extrapolation.data.vocabulary import Vocabulary
from extrapolation.losses import DIGITLoss
from extrapolation.models.digit import DIGITModel
from extrapolation.trace_pipeline import (
    TRACE_LABEL_VERSION,
    ensure_trace_corpus,
    evaluate_trace_baselines,
    set_global_determinism,
)


def _move_batch(batch, device: torch.device):
    queries, trace_inputs, prim_targets, target_ids = batch
    return (
        queries.to(device),
        trace_inputs.to(device),
        prim_targets.to(device),
        target_ids.to(device),
    )


def _one_hot_primitives(prim_targets: torch.Tensor, config: Config) -> tuple[torch.Tensor, ...]:
    return (
        F.one_hot(prim_targets[:, 0], config.num_trajectory_classes).float(),
        F.one_hot(prim_targets[:, 1], config.num_pattern_classes).float(),
        F.one_hot(prim_targets[:, 2], config.num_confidence_classes).float(),
        F.one_hot(prim_targets[:, 3], config.num_outcome_classes).float(),
    )


def _decoder_logits_from_ground_truth(
    model: DIGITModel,
    queries: torch.Tensor,
    prim_targets: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    trajectory_disc, pattern_disc, confidence_disc, outcome_disc = _one_hot_primitives(
        prim_targets,
        model.config,
    )
    z_q = model.encoder(queries)
    return model.decoder(
        z_q=z_q,
        trajectory_shape_disc=trajectory_disc.to(queries.device),
        attention_pattern_disc=pattern_disc.to(queries.device),
        confidence_disc=confidence_disc.to(queries.device),
        outcome_disc=outcome_disc.to(queries.device),
        target_ids=target_ids[:, :-1],
    )


def _generation_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_idx: int,
) -> torch.Tensor:
    targets = target_ids[:, 1:]
    batch_size, target_len, vocab_size = logits.shape
    return F.cross_entropy(
        logits.reshape(batch_size * target_len, vocab_size),
        targets.reshape(batch_size * target_len),
        ignore_index=pad_idx,
        reduction="mean",
    )


def _compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float | list[list[int]]]:
    metrics: dict[str, float | list[list[int]]] = {}
    head_names = ["trajectory", "pattern", "confidence", "outcome"]
    head_label_sizes = [
        len(TRAJECTORY_SHAPE_LABELS),
        len(ATTENTION_PATTERN_LABELS),
        len(CONFIDENCE_LABELS),
        len(OUTCOME_LABELS),
    ]
    for idx, head_name in enumerate(head_names):
        head_pred = predictions[:, idx]
        head_target = targets[:, idx]
        metrics[f"{head_name}_acc"] = (head_pred == head_target).float().mean().item()
        metrics[f"{head_name}_macro_f1"] = float(
            f1_score(head_target.numpy(), head_pred.numpy(), average="macro", zero_division=0)
        )
        metrics[f"{head_name}_confusion"] = confusion_matrix(
            head_target.numpy(),
            head_pred.numpy(),
            labels=list(range(head_label_sizes[idx])),
        ).tolist()

    metrics["joint_acc"] = (predictions == targets).all(dim=1).float().mean().item()
    outcome_precision, outcome_recall, outcome_f1, _ = precision_recall_fscore_support(
        targets[:, 3].numpy(),
        predictions[:, 3].numpy(),
        labels=list(range(len(OUTCOME_LABELS))),
        zero_division=0,
    )
    for idx, label in enumerate(OUTCOME_LABELS):
        label_key = label.lower()
        metrics[f"{label_key}_precision"] = float(outcome_precision[idx])
        metrics[f"{label_key}_recall"] = float(outcome_recall[idx])
        metrics[f"{label_key}_f1"] = float(outcome_f1[idx])

    metrics["failure_recall"] = float(
        outcome_recall[OUTCOME_LABELS.index("FAILURE_LIKELY")]
    )

    pattern_precision, pattern_recall, pattern_f1, _ = precision_recall_fscore_support(
        targets[:, 1].numpy(),
        predictions[:, 1].numpy(),
        labels=list(range(len(ATTENTION_PATTERN_LABELS))),
        zero_division=0,
    )
    for idx, label in enumerate(ATTENTION_PATTERN_LABELS):
        label_key = label.lower()
        metrics[f"pattern_{label_key}_precision"] = float(pattern_precision[idx])
        metrics[f"pattern_{label_key}_recall"] = float(pattern_recall[idx])
        metrics[f"pattern_{label_key}_f1"] = float(pattern_f1[idx])
    return metrics


def _evaluate_stage1(
    model: DIGITModel,
    criterion: DIGITLoss,
    loader: DataLoader,
    device: torch.device,
    records: list[dict] | None = None,
    collect_bundle: bool = False,
) -> dict[str, float | list[list[int]]] | tuple[dict[str, float | list[list[int]]], dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_predictions = []
    all_targets = []
    all_logits: dict[str, list[torch.Tensor]] = {
        "trajectory_shape": [],
        "attention_pattern": [],
        "confidence": [],
        "outcome": [],
    }
    all_probs: dict[str, list[torch.Tensor]] = {
        "trajectory_shape": [],
        "attention_pattern": [],
        "confidence": [],
        "outcome": [],
    }

    with torch.no_grad():
        for batch in loader:
            queries, trace_inputs, prim_targets, target_ids = _move_batch(batch, device)
            out = model(
                queries,
                trace_inputs,
                target_ids,
                bottleneck_mode="hard",
                tau=1.0,
                skip_decoder=True,
            )
            losses = criterion(
                out["decoder_logits"],
                out["primitives"],
                target_ids,
                prim_targets,
                out["executor_features"],
            )

            batch_size = queries.size(0)
            total_loss += losses["total"].item() * batch_size
            total_items += batch_size

            predictions = torch.stack(
                [
                    out["primitives"].trajectory_shape_logits.argmax(dim=-1),
                    out["primitives"].attention_pattern_logits.argmax(dim=-1),
                    out["primitives"].confidence_logits.argmax(dim=-1),
                    out["primitives"].outcome_logits.argmax(dim=-1),
                ],
                dim=1,
            )
            all_predictions.append(predictions.cpu())
            all_targets.append(prim_targets.cpu())
            all_logits["trajectory_shape"].append(out["primitives"].trajectory_shape_logits.cpu())
            all_logits["attention_pattern"].append(out["primitives"].attention_pattern_logits.cpu())
            all_logits["confidence"].append(out["primitives"].confidence_logits.cpu())
            all_logits["outcome"].append(out["primitives"].outcome_logits.cpu())
            all_probs["trajectory_shape"].append(F.softmax(out["primitives"].trajectory_shape_logits, dim=-1).cpu())
            all_probs["attention_pattern"].append(F.softmax(out["primitives"].attention_pattern_logits, dim=-1).cpu())
            all_probs["confidence"].append(F.softmax(out["primitives"].confidence_logits, dim=-1).cpu())
            all_probs["outcome"].append(F.softmax(out["primitives"].outcome_logits, dim=-1).cpu())

    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = _compute_metrics(predictions, targets)
    if records is not None:
        if len(records) != predictions.size(0):
            raise ValueError(f"Expected {predictions.size(0)} records, got {len(records)}")
        outcome_probs = torch.cat(all_probs["outcome"], dim=0)
        is_correct = [int(bool(record.get("is_correct", False))) for record in records]
        metrics.update(
            compute_success_safety_metrics(
                outcome_probabilities=outcome_probs.numpy(),
                outcome_predictions=predictions[:, 3].numpy(),
                outcome_targets=targets[:, 3].numpy(),
                is_correct=is_correct,
            )
        )
    metrics["loss"] = total_loss / max(total_items, 1)
    if not collect_bundle:
        return metrics
    return metrics, {
        "predictions": predictions,
        "targets": targets,
        "logits": {key: torch.cat(values, dim=0) for key, values in all_logits.items()},
        "probs": {key: torch.cat(values, dim=0) for key, values in all_probs.items()},
    }


def _evaluate_stage2(
    model: DIGITModel,
    loader: DataLoader,
    device: torch.device,
    pad_idx: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in loader:
            queries, _, prim_targets, target_ids = _move_batch(batch, device)
            logits = _decoder_logits_from_ground_truth(model, queries, prim_targets, target_ids)
            loss = _generation_loss(logits, target_ids, pad_idx=pad_idx)
            batch_size = queries.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

    return total_loss / max(total_items, 1)


def _metric_value(metrics: dict[str, float | list[list[int]]], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, list):
        raise TypeError(f"Metric '{key}' is not scalar")
    if value is None:
        raise KeyError(f"Metric '{key}' not found")
    return float(value)


def _is_better_stage1(
    metrics: dict[str, float | list[list[int]]],
    best_metrics: dict[str, float | list[list[int]]] | None,
    tolerance: float = 0.01,
) -> bool:
    if best_metrics is None:
        return True

    candidate_outcome = _metric_value(metrics, "outcome_macro_f1")
    best_outcome = _metric_value(best_metrics, "outcome_macro_f1")
    if candidate_outcome > best_outcome + tolerance:
        return True
    if abs(candidate_outcome - best_outcome) <= tolerance:
        candidate_pattern = _metric_value(metrics, "pattern_macro_f1")
        best_pattern = _metric_value(best_metrics, "pattern_macro_f1")
        if candidate_pattern > best_pattern + tolerance:
            return True
        if abs(candidate_pattern - best_pattern) <= tolerance:
            candidate_failure = _metric_value(metrics, "failure_recall")
            best_failure = _metric_value(best_metrics, "failure_recall")
            if candidate_failure > best_failure + tolerance:
                return True
            if abs(candidate_failure - best_failure) <= tolerance:
                candidate_trajectory = _metric_value(metrics, "trajectory_macro_f1")
                best_trajectory = _metric_value(best_metrics, "trajectory_macro_f1")
                if candidate_trajectory > best_trajectory + tolerance:
                    return True
                if abs(candidate_trajectory - best_trajectory) <= tolerance:
                    return _metric_value(metrics, "joint_acc") > _metric_value(best_metrics, "joint_acc") + 1e-8
    return False


def _format_confusion(matrix: list[list[int]], labels: list[str]) -> str:
    rows = []
    for label, row in zip(labels, matrix):
        rows.append(f"{label}: {row}")
    return " | ".join(rows)


def _collect_primitive_predictions(
    model: DIGITModel,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in loader:
            queries, trace_inputs, _, target_ids = _move_batch(batch, device)
            out = model(
                queries,
                trace_inputs,
                target_ids,
                bottleneck_mode="hard",
                tau=1.0,
                skip_decoder=True,
            )
            outputs.append(
                torch.stack(
                    [
                        out["primitives"].trajectory_shape_logits.argmax(dim=-1),
                        out["primitives"].attention_pattern_logits.argmax(dim=-1),
                        out["primitives"].confidence_logits.argmax(dim=-1),
                        out["primitives"].outcome_logits.argmax(dim=-1),
                    ],
                    dim=1,
                ).cpu()
            )
    return torch.cat(outputs, dim=0)


def _majority_baseline(train_records: list[dict], test_records: list[dict]) -> dict[str, float | int]:
    majority_class = Counter(record["outcome"] for record in train_records).most_common(1)[0][0]
    targets = [record["outcome"] for record in test_records]
    predictions = [majority_class] * len(test_records)
    accuracy = sum(int(pred == target) for pred, target in zip(predictions, targets)) / max(len(targets), 1)
    macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))
    return {
        "class_idx": int(majority_class),
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
    }


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def _reference_metrics_path(trace_dir: Path) -> Path:
    return trace_dir / "reference_metrics.e2.json"


def _default_output_dir(variant_name: str, seed: int) -> Path:
    return ROOT_DIR / "results" / variant_name / f"seed_{seed}"


def _load_reference_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _reference_metrics_need_refresh(reference_metrics: dict | None) -> bool:
    if reference_metrics is None:
        return True
    test_metrics = reference_metrics.get("test_metrics", {})
    required_keys = [
        "pattern_focused_precision",
        "pattern_focused_recall",
        "pattern_focused_f1",
        "pattern_mixed_precision",
        "pattern_mixed_recall",
        "pattern_mixed_f1",
        "pattern_diffuse_precision",
        "pattern_diffuse_recall",
        "pattern_diffuse_f1",
    ]
    return any(key not in test_metrics for key in required_keys)


def _maybe_write_reference_metrics(
    path: Path,
    trace_info: dict,
    test_metrics: dict,
    baseline_results: dict,
) -> bool:
    if trace_info.get("label_version") != TRACE_LABEL_VERSION:
        return False
    if trace_info.get("pattern_threshold_mode") != "auto":
        return False
    existing_reference = _load_reference_metrics(path)
    if existing_reference is not None and not _reference_metrics_need_refresh(existing_reference):
        return False
    payload = {
        "label_version": trace_info.get("label_version"),
        "pattern_z_threshold": trace_info.get("pattern_z_threshold"),
        "pattern_threshold_mode": trace_info.get("pattern_threshold_mode"),
        "corpus_fingerprint": trace_info.get("corpus_fingerprint"),
        "test_metrics": test_metrics,
        "baselines": baseline_results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-traces", action="store_true")
    parser.add_argument("--pattern-threshold-override", type=float, default=None)
    parser.add_argument("--train-loop-seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-reference-write", action="store_true")
    args = parser.parse_args()

    config = Config()
    if args.train_loop_seed is not None:
        config.train_loop_seed = int(args.train_loop_seed)
    config.trace_pattern_threshold_override = args.pattern_threshold_override
    variant_name = VARIANT_DIR.name
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(variant_name, config.train_loop_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = Vocabulary()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    set_global_determinism(config.train_loop_seed)

    trace_dir = Path(__file__).parent.parent / config.trace_dir
    trace_info = ensure_trace_corpus(
        config=config,
        root_dir=trace_dir,
        device=device,
        rebuild=args.rebuild_traces,
        strict_existing=not args.rebuild_traces,
    )

    with open(trace_dir / "metadata.json") as f:
        trace_metadata = json.load(f)

    train_dataset = TraceEntropyDataset.from_jsonl(trace_dir / "train.jsonl", vocab=vocab, config=config)
    val_dataset = TraceEntropyDataset.from_jsonl(trace_dir / "val.jsonl", vocab=vocab, config=config)
    test_dataset = TraceEntropyDataset.from_jsonl(trace_dir / "test.jsonl", vocab=vocab, config=config)

    if config.trace_use_weighted_sampler:
        raise ValueError("trace_use_weighted_sampler=True is not supported in the recovery plan")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=trace_collate_fn,
        generator=torch.Generator().manual_seed(config.train_loop_seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=trace_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=trace_collate_fn,
    )

    model = DIGITModel(config, vocab).to(device)
    model.set_trace_metadata(trace_metadata)
    class_weights = {
        key: value.to(device)
        for key, value in train_dataset.get_class_weights(
            power=config.trace_class_weight_power,
            clip=config.trace_class_weight_clip,
        ).items()
    }
    class_weights["outcome"] = torch.tensor(
        [1.0, 1.0, config.trace_outcome_failure_weight],
        dtype=torch.float32,
        device=device,
    )

    stage1_config = copy.deepcopy(config)
    stage1_config.lambda_primitive = 1.0
    stage1_config.lambda_generation = 0.0
    stage1_config.lambda_policy = 0.0
    stage1_config.lambda_leakage = 0.0
    stage1_config.lambda_abstention = 0.0
    stage1_config.primitive_pattern_loss_weight = 1.5
    stage1_config.primitive_confidence_loss_weight = 0.75
    stage1_criterion = DIGITLoss(
        stage1_config,
        pad_idx=vocab.pad_idx,
        class_weights=class_weights,
    ).to(device)
    stage1_optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_stage1_state = copy.deepcopy(model.state_dict())
    best_stage1_metrics: dict[str, float | list[list[int]]] | None = None
    patience_counter = 0

    print("DIGIT Extrapolation E2 uncertainty-tower run")
    print(f"Device: {device}")
    print(f"Results dir: {output_dir}")
    print(f"Trace dir: {trace_info['trace_dir']}")
    print(f"Trace label version: {trace_info.get('label_version', 'unknown')}")
    print(f"Corpus fingerprint: {trace_info.get('corpus_fingerprint')}")
    print(f"Pattern z-threshold: {trace_info.get('pattern_z_threshold')}")
    print(f"Pattern threshold mode: {trace_info.get('pattern_threshold_mode')}")
    print(
        f"Trace records train/val/test: "
        f"{len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}"
    )
    print(
        "Failure shares train/val/test: "
        f"{trace_info.get('train_failure_share', 0.0):.3f}/"
        f"{trace_info.get('val_failure_share', 0.0):.3f}/"
        f"{trace_info.get('test_failure_share', 0.0):.3f}"
    )
    print(f"Train hard-mined cases: {trace_info.get('train_hard_case_count', 0)}")
    print(f"Train label distribution: {train_dataset.get_label_distribution()}")
    print(
        f"Train attention-pattern distribution: "
        f"{trace_metadata.get('attention_pattern_distribution', {})}"
    )
    print(
        "Stage 1: joint primitive training "
        "(selection: outcome_macro_f1 -> pattern_macro_f1 -> failure_recall -> trajectory_macro_f1 -> joint_acc)"
    )

    for epoch in range(config.trace_stage1_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_items = 0

        for batch in train_loader:
            queries, trace_inputs, prim_targets, target_ids = _move_batch(batch, device)
            out = model(
                queries,
                trace_inputs,
                target_ids,
                bottleneck_mode="gumbel",
                tau=1.0,
                skip_decoder=True,
            )
            losses = stage1_criterion(
                out["decoder_logits"],
                out["primitives"],
                target_ids,
                prim_targets,
                out["executor_features"],
            )

            stage1_optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            stage1_optimizer.step()

            batch_size = queries.size(0)
            epoch_loss += losses["total"].item() * batch_size
            epoch_items += batch_size

        train_loss = epoch_loss / max(epoch_items, 1)
        val_metrics = _evaluate_stage1(model, stage1_criterion, val_loader, device)
        print(
            f"Stage1 Epoch {epoch + 1:02d}/{config.trace_stage1_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={_metric_value(val_metrics, 'loss'):.4f} | "
            f"val_outcome_acc={_metric_value(val_metrics, 'outcome_acc'):.3f} | "
            f"val_outcome_macro_f1={_metric_value(val_metrics, 'outcome_macro_f1'):.3f} | "
            f"val_failure_recall={_metric_value(val_metrics, 'failure_recall'):.3f} | "
            f"val_trajectory_macro_f1={_metric_value(val_metrics, 'trajectory_macro_f1'):.3f} | "
            f"val_pattern_macro_f1={_metric_value(val_metrics, 'pattern_macro_f1'):.3f} | "
            f"val_joint_acc={_metric_value(val_metrics, 'joint_acc'):.3f}"
        )

        if _is_better_stage1(
            val_metrics,
            best_stage1_metrics,
            tolerance=config.trace_stage1_selection_tolerance,
        ):
            best_stage1_state = copy.deepcopy(model.state_dict())
            best_stage1_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.trace_stage1_patience:
                print(f"Early stopping stage 1 at epoch {epoch + 1}")
                break

    model.load_state_dict(best_stage1_state)
    pre_stage2_predictions = _collect_primitive_predictions(model, test_loader, device)

    print("Stage 2: generation-only training")
    _freeze_module(model.bottleneck)
    _freeze_module(model.encoder)
    _unfreeze_module(model.decoder)
    stage2_optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_stage2_state = copy.deepcopy(model.state_dict())
    best_stage2_loss = float("inf")
    for epoch in range(config.trace_stage2_epochs):
        model.train()
        model.encoder.eval()
        model.bottleneck.eval()
        epoch_loss = 0.0
        epoch_items = 0

        for batch in train_loader:
            queries, _, prim_targets, target_ids = _move_batch(batch, device)
            logits = _decoder_logits_from_ground_truth(model, queries, prim_targets, target_ids)
            loss = _generation_loss(logits, target_ids, pad_idx=vocab.pad_idx)

            stage2_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            stage2_optimizer.step()

            batch_size = queries.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_items += batch_size

        train_gen_loss = epoch_loss / max(epoch_items, 1)
        val_gen_loss = _evaluate_stage2(model, val_loader, device, pad_idx=vocab.pad_idx)
        print(
            f"Stage2 Epoch {epoch + 1:02d}/{config.trace_stage2_epochs} | "
            f"train_gen_loss={train_gen_loss:.4f} | "
            f"val_gen_loss={val_gen_loss:.4f}"
        )
        if val_gen_loss < best_stage2_loss:
            best_stage2_loss = val_gen_loss
            best_stage2_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_stage2_state)
    post_stage2_predictions = _collect_primitive_predictions(model, test_loader, device)
    primitive_predictions_preserved = torch.equal(pre_stage2_predictions, post_stage2_predictions)

    test_metrics, test_bundle = _evaluate_stage1(
        model,
        stage1_criterion,
        test_loader,
        device,
        records=test_dataset.records,
        collect_bundle=True,
    )
    print(
        "Test metrics: "
        f"trajectory_acc={_metric_value(test_metrics, 'trajectory_acc'):.3f}, "
        f"trajectory_macro_f1={_metric_value(test_metrics, 'trajectory_macro_f1'):.3f}, "
        f"pattern_acc={_metric_value(test_metrics, 'pattern_acc'):.3f}, "
        f"pattern_macro_f1={_metric_value(test_metrics, 'pattern_macro_f1'):.3f}, "
        f"confidence_acc={_metric_value(test_metrics, 'confidence_acc'):.3f}, "
        f"confidence_macro_f1={_metric_value(test_metrics, 'confidence_macro_f1'):.3f}, "
        f"outcome_acc={_metric_value(test_metrics, 'outcome_acc'):.3f}, "
        f"outcome_macro_f1={_metric_value(test_metrics, 'outcome_macro_f1'):.3f}, "
        f"failure_recall={_metric_value(test_metrics, 'failure_recall'):.3f}, "
        f"joint_acc={_metric_value(test_metrics, 'joint_acc'):.3f}"
    )
    print(
        f"Trajectory confusion: "
        f"{_format_confusion(test_metrics['trajectory_confusion'], TRAJECTORY_SHAPE_LABELS)}"
    )
    print(
        f"Pattern confusion: "
        f"{_format_confusion(test_metrics['pattern_confusion'], ATTENTION_PATTERN_LABELS)}"
    )
    print(
        "Pattern per-class: "
        f"FOCUSED(p/r/f1)={_metric_value(test_metrics, 'pattern_focused_precision'):.3f}/"
        f"{_metric_value(test_metrics, 'pattern_focused_recall'):.3f}/"
        f"{_metric_value(test_metrics, 'pattern_focused_f1'):.3f} | "
        f"MIXED(p/r/f1)={_metric_value(test_metrics, 'pattern_mixed_precision'):.3f}/"
        f"{_metric_value(test_metrics, 'pattern_mixed_recall'):.3f}/"
        f"{_metric_value(test_metrics, 'pattern_mixed_f1'):.3f} | "
        f"DIFFUSE(p/r/f1)={_metric_value(test_metrics, 'pattern_diffuse_precision'):.3f}/"
        f"{_metric_value(test_metrics, 'pattern_diffuse_recall'):.3f}/"
        f"{_metric_value(test_metrics, 'pattern_diffuse_f1'):.3f}"
    )
    print(f"Outcome confusion: {_format_confusion(test_metrics['outcome_confusion'], OUTCOME_LABELS)}")
    print(
        "Outcome per-class: "
        f"SUCCESS(p/r/f1)={_metric_value(test_metrics, 'success_likely_precision'):.3f}/"
        f"{_metric_value(test_metrics, 'success_likely_recall'):.3f}/"
        f"{_metric_value(test_metrics, 'success_likely_f1'):.3f} | "
        f"UNCERTAIN(p/r/f1)={_metric_value(test_metrics, 'uncertain_precision'):.3f}/"
        f"{_metric_value(test_metrics, 'uncertain_recall'):.3f}/"
        f"{_metric_value(test_metrics, 'uncertain_f1'):.3f} | "
        f"FAILURE(p/r/f1)={_metric_value(test_metrics, 'failure_likely_precision'):.3f}/"
        f"{_metric_value(test_metrics, 'failure_likely_recall'):.3f}/"
        f"{_metric_value(test_metrics, 'failure_likely_f1'):.3f}"
    )
    print(f"Primitive predictions preserved through stage 2: {primitive_predictions_preserved}")

    train_records = load_trace_records(trace_dir / "train.jsonl")
    val_records = load_trace_records(trace_dir / "val.jsonl")
    test_records = load_trace_records(trace_dir / "test.jsonl")
    corpus_records = train_records + val_records + test_records
    majority = _majority_baseline(train_records, test_records)
    baseline_results = evaluate_trace_baselines(train_records, test_records, config=config)
    correctness_baselines = evaluate_correctness_baselines(
        train_records,
        test_records,
        mlp_max_iter=config.baseline_mlp_max_iter,
        mlp_early_stopping=config.baseline_mlp_early_stopping,
    )
    label_audit = compute_label_audit(corpus_records)
    threshold_occupancy = compute_threshold_occupancy(corpus_records, trace_metadata)
    threshold_sensitivity = compute_threshold_sensitivity(corpus_records, trace_metadata)
    monotonicity = audit_monotonicity(model, test_loader, device)
    metadata_path = trace_dir / "metadata.json"
    if metadata_path.exists() and not args.skip_reference_write:
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["mlp_converged"] = baseline_results["mlp"]["outcome"]["converged"]
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    reference_path = _reference_metrics_path(trace_dir)
    created_reference = False
    if not args.skip_reference_write:
        created_reference = _maybe_write_reference_metrics(
            reference_path,
            trace_info=trace_info,
            test_metrics=test_metrics,
            baseline_results=baseline_results,
        )
    reference_metrics = _load_reference_metrics(reference_path)
    print(
        "Baselines: "
        f"majority_outcome={OUTCOME_LABELS[int(majority['class_idx'])]}, "
        f"majority_acc={majority['accuracy']:.3f}, "
        f"majority_macro_f1={majority['macro_f1']:.3f}, "
        f"entropy_threshold_acc={baseline_results['entropy_threshold']['outcome_accuracy']:.3f}, "
        f"entropy_threshold_macro_f1={baseline_results['entropy_threshold']['outcome_macro_f1']:.3f}, "
        f"logreg_outcome_acc={baseline_results['logistic_regression']['outcome']['accuracy']:.3f}, "
        f"logreg_outcome_macro_f1={baseline_results['logistic_regression']['outcome']['macro_f1']:.3f}, "
        f"mlp_outcome_acc={baseline_results['mlp']['outcome']['accuracy']:.3f}, "
        f"mlp_outcome_macro_f1={baseline_results['mlp']['outcome']['macro_f1']:.3f}, "
        f"mlp_converged={baseline_results['mlp']['outcome']['converged']}, "
        f"mlp_n_iter={baseline_results['mlp']['outcome']['n_iter']}"
    )
    print(
        "Correctness baselines: "
        f"logreg_acc={correctness_baselines['logistic_regression']['accuracy']:.3f}, "
        f"logreg_auroc={correctness_baselines['logistic_regression']['auroc']:.3f}, "
        f"mlp_acc={correctness_baselines['mlp']['accuracy']:.3f}, "
        f"mlp_auroc={correctness_baselines['mlp']['auroc']:.3f}, "
        f"mlp_converged={correctness_baselines['mlp']['converged']}"
    )
    print(
        "Safety metrics: "
        f"unsafe_success_rate_label={_metric_value(test_metrics, 'unsafe_success_rate_label'):.3f}, "
        f"unsafe_success_rate_correctness={_metric_value(test_metrics, 'unsafe_success_rate_correctness'):.3f}, "
        f"success_precision_vs_is_correct={_metric_value(test_metrics, 'success_precision_vs_is_correct'):.3f}, "
        f"success_auroc={_metric_value(test_metrics, 'success_auroc'):.3f}, "
        f"success_auprc={_metric_value(test_metrics, 'success_auprc'):.3f}, "
        f"success_brier={_metric_value(test_metrics, 'success_brier'):.3f}, "
        f"success_ece_10bin={_metric_value(test_metrics, 'success_ece_10bin'):.3f}"
    )
    print(
        "Label audit: "
        f"P(correct|SUCCESS)={label_audit['p_is_correct_given_outcome']['SUCCESS_LIKELY']:.3f}, "
        f"P(correct|UNCERTAIN)={label_audit['p_is_correct_given_outcome']['UNCERTAIN']:.3f}, "
        f"P(correct|FAILURE)={label_audit['p_is_correct_given_outcome']['FAILURE_LIKELY']:.3f}, "
        f"correctness_separation={label_audit['correctness_separation_success_minus_uncertain']:.3f}, "
        f"max_outcome_share_swing={threshold_sensitivity['max_outcome_share_swing']:.3f}, "
        f"label_revision_trigger={threshold_sensitivity['label_revision_trigger']}"
    )
    print(
        "Monotonicity: "
        f"violation_rate={monotonicity['monotonicity_violation_rate']:.3f}, "
        f"all_supported_violation_rate={monotonicity['all_supported_monotonicity_violation_rate']:.3f}"
    )
    if created_reference:
        print(f"Reference metrics saved: {reference_path.name}")
    elif reference_metrics is not None:
        if reference_metrics.get("corpus_fingerprint") == trace_info.get("corpus_fingerprint"):
            ref_test = reference_metrics["test_metrics"]
            ref_mlp_macro_f1 = reference_metrics["baselines"]["mlp"]["outcome"]["macro_f1"]
            current_gap = baseline_results["mlp"]["outcome"]["macro_f1"] - _metric_value(test_metrics, "outcome_macro_f1")
            ref_gap = ref_mlp_macro_f1 - float(ref_test["outcome_macro_f1"])
            gap_shrink = 0.0 if ref_gap <= 1e-8 else max(0.0, (ref_gap - current_gap) / ref_gap)
            print(
                "Reference deltas: "
                f"outcome_macro_f1_delta={_metric_value(test_metrics, 'outcome_macro_f1') - float(ref_test['outcome_macro_f1']):+.3f}, "
                f"failure_recall_delta={_metric_value(test_metrics, 'failure_recall') - float(ref_test['failure_recall']):+.3f}, "
                f"trajectory_macro_f1_delta={_metric_value(test_metrics, 'trajectory_macro_f1') - float(ref_test['trajectory_macro_f1']):+.3f}, "
                f"pattern_macro_f1_delta={_metric_value(test_metrics, 'pattern_macro_f1') - float(ref_test['pattern_macro_f1']):+.3f}, "
                f"pattern_focused_recall_delta={_metric_value(test_metrics, 'pattern_focused_recall') - float(ref_test['pattern_focused_recall']):+.3f}, "
                f"pattern_diffuse_recall_delta={_metric_value(test_metrics, 'pattern_diffuse_recall') - float(ref_test['pattern_diffuse_recall']):+.3f}, "
                f"confidence_macro_f1_delta={_metric_value(test_metrics, 'confidence_macro_f1') - float(ref_test['confidence_macro_f1']):+.3f}, "
                f"mlp_gap_shrink={gap_shrink:.3f}"
            )
            outcome_gain_ok = (
                _metric_value(test_metrics, "outcome_macro_f1") >= float(ref_test["outcome_macro_f1"]) + 0.02
                or (
                    _metric_value(test_metrics, "outcome_macro_f1") >= float(ref_test["outcome_macro_f1"]) - 0.01
                    and _metric_value(test_metrics, "failure_recall") >= float(ref_test["failure_recall"]) + 0.05
                )
            )
            print(
                "Reference acceptance: "
                f"outcome_gain_ok={outcome_gain_ok}, "
                f"trajectory_gain_ok={_metric_value(test_metrics, 'trajectory_macro_f1') >= float(ref_test['trajectory_macro_f1']) + 0.08}, "
                f"pattern_gain_ok={_metric_value(test_metrics, 'pattern_macro_f1') >= float(ref_test['pattern_macro_f1']) + 0.05}, "
                f"focused_recall_gain_ok={_metric_value(test_metrics, 'pattern_focused_recall') >= float(ref_test['pattern_focused_recall']) + 0.10}, "
                f"diffuse_recall_gain_ok={_metric_value(test_metrics, 'pattern_diffuse_recall') >= float(ref_test['pattern_diffuse_recall']) + 0.10}, "
                f"pattern_not_collapsed_ok={_metric_value(test_metrics, 'pattern_focused_recall') >= 0.10 and _metric_value(test_metrics, 'pattern_diffuse_recall') >= 0.10}, "
                f"confidence_drop_ok={_metric_value(test_metrics, 'confidence_macro_f1') >= float(ref_test['confidence_macro_f1']) - 0.02}, "
                f"mlp_gap_shrink_ok={gap_shrink >= 0.30}"
            )
        else:
            print(
                "Reference metrics fingerprint mismatch: "
                f"{reference_metrics.get('corpus_fingerprint')} != {trace_info.get('corpus_fingerprint')}"
            )
    print(
        "Acceptance: "
        f"failure_recall_ok={_metric_value(test_metrics, 'failure_recall') >= 0.30}, "
        f"outcome_macro_f1_ok={_metric_value(test_metrics, 'outcome_macro_f1') >= 0.62}, "
        f"outcome_acc_ok={_metric_value(test_metrics, 'outcome_acc') >= 0.82}, "
        f"beats_majority={_metric_value(test_metrics, 'outcome_acc') > majority['accuracy']}, "
        f"beats_entropy_threshold={_metric_value(test_metrics, 'outcome_acc') > baseline_results['entropy_threshold']['outcome_accuracy']}, "
        f"trajectory_ok={_metric_value(test_metrics, 'trajectory_acc') >= 0.75}, "
        f"monotonicity_ok={monotonicity['monotonicity_violation_rate'] < 0.05}"
    )

    query, trace_inputs, prim_targets, target_ids = test_dataset[0]
    batch_query = query.unsqueeze(0).to(device)
    batch_trace = trace_inputs.unsqueeze(0).to(device)
    batch_targets = target_ids.unsqueeze(0).to(device)

    out = model(batch_query, batch_trace, batch_targets, bottleneck_mode="hard", tau=1.0)
    pred = model.bottleneck.get_primitive_indices(out["primitives"])
    text = vocab.decode(model.generate(batch_query, batch_trace)["token_ids"][0].cpu().tolist())
    ground_truth_text = vocab.decode(target_ids.tolist())

    print(
        "Ground truth: "
        f"trajectory={TRAJECTORY_SHAPE_LABELS[prim_targets[0].item()]}, "
        f"pattern={ATTENTION_PATTERN_LABELS[prim_targets[1].item()]}, "
        f"confidence={CONFIDENCE_LABELS[prim_targets[2].item()]}, "
        f"outcome={OUTCOME_LABELS[prim_targets[3].item()]}"
    )
    print(
        "Prediction: "
        f"trajectory={TRAJECTORY_SHAPE_LABELS[pred['trajectory_shape'][0].item()]}, "
        f"pattern={ATTENTION_PATTERN_LABELS[pred['attention_pattern'][0].item()]}, "
        f"confidence={CONFIDENCE_LABELS[pred['confidence'][0].item()]}, "
        f"outcome={OUTCOME_LABELS[pred['outcome'][0].item()]}"
    )
    print(f"Decoder logits shape: {tuple(out['decoder_logits'].shape)}")
    print(f"Executor features shape: {tuple(out['executor_features'].shape)}")
    print(f"Reference text: {ground_truth_text}")
    print(f"Generated text: {text}")

    label_names = {
        "trajectory_shape": TRAJECTORY_SHAPE_LABELS,
        "attention_pattern": ATTENTION_PATTERN_LABELS,
        "confidence": CONFIDENCE_LABELS,
        "outcome": OUTCOME_LABELS,
    }
    prediction_rows = build_prediction_records(
        test_dataset.records,
        test_bundle,
        run_seed=config.train_loop_seed,
        label_names=label_names,
    )
    acceptance = {
        "failure_recall_ok": bool(_metric_value(test_metrics, "failure_recall") >= 0.30),
        "outcome_macro_f1_ok": bool(_metric_value(test_metrics, "outcome_macro_f1") >= 0.62),
        "outcome_acc_ok": bool(_metric_value(test_metrics, "outcome_acc") >= 0.82),
        "trajectory_ok": bool(_metric_value(test_metrics, "trajectory_acc") >= 0.75),
        "monotonicity_ok": bool(monotonicity["monotonicity_violation_rate"] < 0.05),
        "label_revision_trigger": bool(threshold_sensitivity["label_revision_trigger"]),
    }
    sample_prediction = {
        "ground_truth": {
            "trajectory": TRAJECTORY_SHAPE_LABELS[prim_targets[0].item()],
            "pattern": ATTENTION_PATTERN_LABELS[prim_targets[1].item()],
            "confidence": CONFIDENCE_LABELS[prim_targets[2].item()],
            "outcome": OUTCOME_LABELS[prim_targets[3].item()],
        },
        "prediction": {
            "trajectory": TRAJECTORY_SHAPE_LABELS[pred["trajectory_shape"][0].item()],
            "pattern": ATTENTION_PATTERN_LABELS[pred["attention_pattern"][0].item()],
            "confidence": CONFIDENCE_LABELS[pred["confidence"][0].item()],
            "outcome": OUTCOME_LABELS[pred["outcome"][0].item()],
        },
        "reference_text": ground_truth_text,
        "generated_text": text,
    }
    metrics_payload = {
        "variant": variant_name,
        "train_loop_seed": int(config.train_loop_seed),
        "corpus_fingerprint": trace_info.get("corpus_fingerprint"),
        "trace_label_version": trace_info.get("label_version"),
        "result_dir": str(output_dir),
        "trace_dir": str(trace_dir),
        "skip_reference_write": bool(args.skip_reference_write),
        "best_val_metrics": best_stage1_metrics,
        "stage2": {
            "best_val_generation_loss": float(best_stage2_loss),
            "primitive_predictions_preserved": bool(primitive_predictions_preserved),
        },
        "test_metrics": test_metrics,
        "label_audit": label_audit,
        "threshold_occupancy": threshold_occupancy,
        "threshold_sensitivity": threshold_sensitivity,
        "baselines": {
            "majority": majority,
            "label_prediction": baseline_results,
            "correctness": correctness_baselines,
        },
        "monotonicity": monotonicity,
        "reference": {
            "path": str(reference_path),
            "created_reference": bool(created_reference),
            "available": bool(reference_metrics is not None),
            "fingerprint_match": bool(
                reference_metrics is not None
                and reference_metrics.get("corpus_fingerprint") == trace_info.get("corpus_fingerprint")
            ),
        },
        "acceptance": acceptance,
        "sample_prediction": sample_prediction,
    }
    summary_payload = {
        "variant": variant_name,
        "train_loop_seed": int(config.train_loop_seed),
        "corpus_fingerprint": trace_info.get("corpus_fingerprint"),
        "result_dir": str(output_dir),
        "test_metrics": test_metrics,
        "label_audit": {
            "p_is_correct_given_outcome": label_audit["p_is_correct_given_outcome"],
            "correctness_separation_success_minus_uncertain": label_audit[
                "correctness_separation_success_minus_uncertain"
            ],
            "max_outcome_share_swing": threshold_sensitivity["max_outcome_share_swing"],
            "label_revision_trigger": threshold_sensitivity["label_revision_trigger"],
        },
        "correctness_baselines": correctness_baselines,
        "monotonicity": monotonicity,
        "acceptance": acceptance,
        "artifacts": {
            "metrics": str(output_dir / "metrics.json"),
            "predictions": str(output_dir / "predictions.jsonl"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    write_json(output_dir / "metrics.json", metrics_payload)
    write_json(output_dir / "summary.json", summary_payload)


if __name__ == "__main__":
    main()
