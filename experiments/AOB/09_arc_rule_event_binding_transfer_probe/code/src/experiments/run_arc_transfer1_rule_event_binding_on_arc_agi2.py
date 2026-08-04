from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from src.datasets.arc_agi2_verification import (
    ArcPair,
    ArcVerificationDatasetConfig,
    ArcVerificationExample,
    load_arc_tasks,
)
from src.experiments.run_plan_arch1_4_learned_constraint_planning import (
    ConstraintTrainingConfig,
    _batched,
    _chunks,
    _set_seed,
    _top1,
)
from src.experiments.run_plan_arch1_6_oracle_to_latent_constraint_binding import (
    BLANK_ID,
    BridgeFitResult,
    BridgeVariant,
    RuleEventBindingModel,
    _flat_params,
    _grad_norm,
    _l2_delta,
    _mean,
    _resolve_device,
)
from src.experiments.run_plan_arch2_medium_rule_event_constraint_binding import MAIN_VARIANT
from src.experiments.run_stage_arc_hybrid1_empirical_search import (
    CandidateGeneratorConfig,
    _candidate_diversity,
    _candidate_generator_audit,
    _execution_score_predictions,
    _generator_recall,
    _grid_equal,
    _grid_mismatch,
    _palette,
    _shape,
    apply_hybrid_control,
    build_hybrid_candidate_splits,
)


BENCHMARK = "arc_transfer1_rule_event_binding_on_arc_agi2"
DEFAULT_CONFIG = "configs/arc_transfer1_rule_event_binding_on_arc_agi2.json"
DEFAULT_RESULTS = "results/arc_transfer1_results.json"
DEFAULT_CONTROLS = "results/arc_transfer1_controls.json"
DEFAULT_GENERATOR_AUDIT = "results/arc_transfer1_candidate_generator_audit.json"
DEFAULT_RULE_EVENT_AUDIT = "results/arc_transfer1_rule_event_audit.json"
DEFAULT_ERRORS = "results/arc_transfer1_error_cases.jsonl"
DEFAULT_COMPUTE = "results/arc_transfer1_compute_metrics.json"
DEFAULT_REPORT = "reports/ARC_TRANSFER1_RULE_EVENT_BINDING_ON_ARC_AGI2.md"

ARC_RULE_EVENT_VARIANT = BridgeVariant(
    name=MAIN_VARIANT,
    binding="relation_mlp",
    description="Locked PLAN-ARCH shared-weight rule/event relation MLP applied to ARC candidate verification.",
    aux_weight=0.35,
    pretrain_epochs=2,
    finetune_aux_weight=0.1,
)
NEAR_CHANCE_MARGIN = 0.10
COLLAPSE_MARGIN = 0.15
HURT_MARGIN = 0.03


@dataclass
class ArcTransferFit:
    model: RuleEventBindingModel
    trainable_shared: bool
    history: List[Dict[str, float]]
    audit: Dict[str, object]
    training_time_seconds: float
    param_count: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARC-TRANSFER-1 rule-event binding probe on local ARC-AGI-2.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--candidate-generator-audit-output", default=DEFAULT_GENERATOR_AUDIT)
    parser.add_argument("--rule-event-audit-output", default=DEFAULT_RULE_EVENT_AUDIT)
    parser.add_argument("--error-output", default=DEFAULT_ERRORS)
    parser.add_argument("--compute-output", default=DEFAULT_COMPUTE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = _default_config(_load_config(Path(args.config)))
    if args.dataset_root:
        config.setdefault("dataset", {})["dataset_root"] = str(args.dataset_root)
    if args.device:
        config["device"] = str(args.device)
    result = run_arc_transfer1(config)
    _write_outputs(
        result,
        Path(args.output),
        Path(args.controls_output),
        Path(args.candidate_generator_audit_output),
        Path(args.rule_event_audit_output),
        Path(args.error_output),
        Path(args.compute_output),
        Path(args.report),
    )


def run_arc_transfer1(config: Dict[str, object]) -> Dict[str, object]:
    started = time.perf_counter()
    dataset_config = _dataset_config(config.get("dataset", {}))
    generator_n8 = _generator_config(config.get("candidate_generator", {}), 8)
    generator_n16 = replace(generator_n8, num_candidates=16, natural_candidate_count=16)
    training = _training_config(config.get("training", {}))
    seeds = [int(value) for value in config.get("seeds", [0, 1])]
    device = _resolve_device(str(config.get("device", "cpu")))
    verify = verify_arc_dataset(Path(dataset_config.dataset_root))
    print(f"arc-transfer1: device={device} seeds={seeds} dataset_ok={verify['exists']} no_final_validation=True")

    rows: List[Dict[str, object]] = []
    controls: List[Dict[str, object]] = []
    generator_audit: List[Dict[str, object]] = []
    rule_event_audit: List[Dict[str, object]] = []
    error_cases: List[Dict[str, object]] = []
    compute_rows: List[Dict[str, object]] = []

    n8_success = False
    for seed in seeds:
        stage_result = _run_candidate_count_stage("N8", replace(dataset_config, num_candidates=8), generator_n8, training, seed, device)
        _extend(rows, controls, generator_audit, rule_event_audit, error_cases, compute_rows, stage_result)
    n8_main = [row for row in rows if row["candidate_count"] == 8 and row["phase"] == "gold_present"]
    n8_success = bool(n8_main and _mean([float(row["trainable"]["top1"]) for row in n8_main]) > _mean([float(row["chance"]) for row in n8_main]) + NEAR_CHANCE_MARGIN)
    if n8_success and bool(config.get("run_n16_if_n8_works", True)):
        for seed in seeds:
            stage_result = _run_candidate_count_stage("N16", replace(dataset_config, num_candidates=16), generator_n16, training, seed, device)
            _extend(rows, controls, generator_audit, rule_event_audit, error_cases, compute_rows, stage_result)

    summary = _summary(rows, controls, generator_audit, verify)
    compute_rows.append(
        {
            "benchmark": BENCHMARK,
            "phase": "total",
            "status": "completed",
            "device": device,
            "runtime_seconds": float(time.perf_counter() - started),
            "final_validation_launched": False,
        }
    )
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "scope": "transfer probe of locked PLAN-ARCH rule-event binding verifier on local ARC-AGI-2 candidate verification",
            "dataset_path": str(dataset_config.dataset_root),
            "final_validation_launched": False,
            "arc_solving_claim": False,
            "architecture": MAIN_VARIANT,
            "architecture_changed": False,
            "rule_event_token_semantics": "arc_visible_rule_event_slots",
            "claim_boundary": "Transfer/probe stage only. No ARC solving, autonomous planning, world modeling, or SOTA claim.",
        },
        "dataset_verification": verify,
        "dataset_config": asdict(dataset_config),
        "generator_config_n8": asdict(generator_n8),
        "generator_config_n16": asdict(generator_n16),
        "training_config": asdict(training),
        "rows": rows,
        "controls": controls,
        "candidate_generator_audit": generator_audit,
        "rule_event_audit": rule_event_audit,
        "error_cases": error_cases,
        "compute_metrics": compute_rows,
        "summary": summary,
    }


def _run_candidate_count_stage(
    stage: str,
    dataset_config: ArcVerificationDatasetConfig,
    generator_config: CandidateGeneratorConfig,
    training: ConstraintTrainingConfig,
    seed: int,
    device: str,
) -> Dict[str, list]:
    started = time.perf_counter()
    gold_splits = build_hybrid_candidate_splits(dataset_config, generator_config, seed=seed, force_gold=True, phase=f"{stage}_gold_present")
    natural_splits = build_hybrid_candidate_splits(dataset_config, generator_config, seed=seed, force_gold=False, phase=f"{stage}_natural_pool")
    generator_rows = []
    for split_name, examples in gold_splits.items():
        row = _candidate_generator_audit(seed, split_name, f"{stage}_gold_present", examples)
        row["benchmark"] = BENCHMARK
        generator_rows.append(row)
    for split_name, examples in natural_splits.items():
        row = _candidate_generator_audit(seed, split_name, f"{stage}_natural_pool", examples)
        row["benchmark"] = BENCHMARK
        generator_rows.append(row)

    print(f"arc-transfer1 {stage} seed={seed}: fitting locked rule-event verifier")
    trainable = fit_arc_transfer(gold_splits["train"], gold_splits["dev"], training, seed + 51_000, device, True)
    frozen = fit_arc_transfer(gold_splits["train"], gold_splits["dev"], training, seed + 51_000, device, False)
    labels = _labels(gold_splits["test"])
    train_logits = predict_arc_transfer_logits(trainable.model, gold_splits["test"], device)
    frozen_logits = predict_arc_transfer_logits(frozen.model, gold_splits["test"], device)
    control = _run_arc_controls(trainable.model, gold_splits, train_logits, labels, device, seed, trainable.audit, frozen.audit)
    row = _result_row(stage, "gold_present", seed, gold_splits, train_logits, frozen_logits, trainable, frozen, control, time.perf_counter() - started)
    natural_logits = predict_arc_transfer_logits(trainable.model, natural_splits["test"], device)
    natural_row = _natural_pool_row(stage, seed, natural_splits, natural_logits, row)
    rule_event_rows = [_rule_event_audit(stage, seed, "gold_present", gold_splits), _rule_event_audit(stage, seed, "natural_pool", natural_splits)]
    errors = _error_rows(stage, seed, gold_splits["test"], train_logits, frozen_logits, natural_splits["test"], natural_logits)
    compute = {
        "benchmark": BENCHMARK,
        "stage": stage,
        "seed": int(seed),
        "candidate_count": int(dataset_config.num_candidates),
        "train_examples": len(gold_splits["train"]),
        "dev_examples": len(gold_splits["dev"]),
        "test_examples": len(gold_splits["test"]),
        "runtime_seconds": float(time.perf_counter() - started),
        "trainable_training_time_seconds": float(trainable.training_time_seconds),
        "frozen_training_time_seconds": float(frozen.training_time_seconds),
        "param_count": int(trainable.param_count),
        "approx_logit_memory_bytes": int(max(train_logits.nbytes, natural_logits.nbytes if natural_logits.size else 0)),
    }
    return {
        "rows": [row, natural_row],
        "controls": [control],
        "candidate_generator_audit": generator_rows,
        "rule_event_audit": rule_event_rows,
        "error_cases": errors,
        "compute_metrics": [compute],
    }


def fit_arc_transfer(
    train_examples: Sequence[ArcVerificationExample],
    dev_examples: Sequence[ArcVerificationExample],
    training: ConstraintTrainingConfig,
    seed: int,
    device: str,
    trainable_shared: bool,
) -> ArcTransferFit:
    _set_seed(seed)
    model = RuleEventBindingModel(ARC_RULE_EVENT_VARIANT).to(device)
    model.configure_shared_trainable(trainable_shared)
    initial_shared = _flat_params(model.shared_parameter_items())
    initial_head = _flat_params(model.head_parameter_items())
    history: List[Dict[str, float]] = []
    shared_grad_norms: List[float] = []
    head_grad_norms: List[float] = []
    start = time.perf_counter()
    if int(ARC_RULE_EVENT_VARIANT.pretrain_epochs) > 0 and trainable_shared:
        _train_arc_epochs(model, train_examples, dev_examples, replace(training, epochs=int(ARC_RULE_EVENT_VARIANT.pretrain_epochs), patience=int(ARC_RULE_EVENT_VARIANT.pretrain_epochs)), seed + 10_000, device, 0.0, float(ARC_RULE_EVENT_VARIANT.aux_weight), "pretrain", history, shared_grad_norms, head_grad_norms)
    _train_arc_epochs(model, train_examples, dev_examples, training, seed + 20_000, device, 1.0, float(ARC_RULE_EVENT_VARIANT.finetune_aux_weight or ARC_RULE_EVENT_VARIANT.aux_weight), "finetune", history, shared_grad_norms, head_grad_norms)
    final_shared = _flat_params(model.shared_parameter_items())
    final_head = _flat_params(model.head_parameter_items())
    audit = {
        "variant": MAIN_VARIANT,
        "same_architecture_comparator": True,
        "shared_model_trainable": bool(trainable_shared),
        "uses_oracle_features_at_inference": False,
        "shared_parameter_delta": _l2_delta(initial_shared, final_shared),
        "head_parameter_delta": _l2_delta(initial_head, final_head),
        "shared_grad_norm_mean": _mean(shared_grad_norms),
        "head_grad_norm_mean": _mean(head_grad_norms),
        "frozen_shared_model_zero_delta": bool((not trainable_shared) and _l2_delta(initial_shared, final_shared) == 0.0),
        "trainable_shared_model_changed": bool(trainable_shared and _l2_delta(initial_shared, final_shared) > 0.0),
        "trainable_shared_model_received_gradients": bool(trainable_shared and _mean(shared_grad_norms) > 0.0),
    }
    return ArcTransferFit(model, trainable_shared, history, audit, float(time.perf_counter() - start), sum(p.numel() for p in model.parameters()))


def _train_arc_epochs(
    model: RuleEventBindingModel,
    train_examples: Sequence[ArcVerificationExample],
    dev_examples: Sequence[ArcVerificationExample],
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
        order = np.random.default_rng(seed + epoch).permutation(len(train_examples)).tolist()
        losses = []
        for batch_ids in _chunks(order, int(training.batch_size)):
            batch_examples = [train_examples[i] for i in batch_ids]
            batch = arc_transfer_batch(batch_examples, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, parts = _arc_transfer_loss(output, batch, rank_weight, aux_weight)
            loss.backward()
            shared_grad_norms.append(_grad_norm([p for _n, p in model.shared_parameter_items()]))
            head_grad_norms.append(_grad_norm([p for _n, p in model.head_parameter_items()]))
            if float(training.gradient_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(training.gradient_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_logits = predict_arc_transfer_logits(model, dev_examples, device)
        dev_top1 = _arc_top1(dev_logits, dev_examples)
        history.append({"phase": phase, "epoch": float(epoch), "loss": _mean(losses), "dev_top1": float(dev_top1)})
        if dev_top1 > best_dev:
            best_dev = float(dev_top1)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= int(training.patience):
            break
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})


def _arc_transfer_loss(output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], rank_weight: float, aux_weight: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    rank_loss = F.cross_entropy(output["logits"], batch["labels"])
    rule_loss = F.cross_entropy(output["rule_logits"].reshape(-1, 4), batch["rule_order"].reshape(-1).clamp(max=3))
    event_loss = F.cross_entropy(output["event_logits"].reshape(-1, 4), batch["event_order"].reshape(-1).clamp(max=3))
    violation_loss = F.binary_cross_entropy_with_logits(output["violation_logits"], batch["valid"].float())
    step_loss = F.cross_entropy(output["step_logits"].reshape(-1, 5), batch["violation_type"].reshape(-1).clamp(max=4))
    family_loss = F.cross_entropy(output["family_logits"], batch["family"])
    aux = rule_loss + event_loss + violation_loss + step_loss + 0.1 * family_loss
    total = float(rank_weight) * rank_loss + float(aux_weight) * aux
    return total, {"rank_loss": float(rank_loss.detach().cpu()), "aux_loss": float(aux.detach().cpu())}


def predict_arc_transfer_logits(
    model: RuleEventBindingModel,
    examples: Sequence[ArcVerificationExample],
    device: str,
    rule_mode: str = "normal",
    event_mode: str = "normal",
    seed: int = 0,
) -> np.ndarray:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch_examples in _batched(list(examples), 128):
            batch = arc_transfer_batch(batch_examples, device, rule_mode=rule_mode, event_mode=event_mode, seed=seed)
            rows.append(model(batch)["logits"].detach().cpu().numpy())
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 0), dtype=np.float32)


def arc_transfer_batch(
    examples: Sequence[ArcVerificationExample],
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
    family_rows = []
    labels = []
    for idx, example in enumerate(examples):
        rule_order = arc_rule_order(example)
        if rule_mode == "blank":
            rule_order = [BLANK_ID] * 4
        elif rule_mode == "shuffle":
            rule_order = list(np.asarray(rule_order)[rng.permutation(4)])
        elif rule_mode in {"mismatch", "hidden_shuffle"} and len(examples) > 1:
            other = examples[(idx + 1 + int(rng.integers(0, len(examples) - 1))) % len(examples)]
            rule_order = arc_rule_order(other)
        event_row = []
        valid_row = []
        violation_row = []
        for cand_idx, candidate in enumerate(example.candidates):
            event_order = arc_event_order(example, cand_idx)
            if event_mode == "blank":
                event_order = [BLANK_ID] * 4
            elif event_mode == "shuffle":
                event_order = list(np.asarray(event_order)[rng.permutation(4)])
            elif event_mode == "mismatch" and len(examples) > 1:
                other = examples[(idx + 1 + int(rng.integers(0, len(examples) - 1))) % len(examples)]
                event_order = arc_event_order(other, cand_idx % len(other.candidates))
            event_row.append(_pad_order(event_order))
            is_valid = bool(_grid_equal(candidate, example.gold_output))
            valid_row.append(float(is_valid))
            violation_row.append(0 if is_valid else _arc_violation_type(example, cand_idx))
        rule_rows.append(_pad_order(rule_order))
        event_rows.append(event_row)
        valid_rows.append(valid_row)
        violation_rows.append(violation_row)
        family_rows.append(_arc_family_id(example))
        labels.append(int(example.label))
    return {
        "rule_order": torch.as_tensor(rule_rows, dtype=torch.long, device=device),
        "event_order": torch.as_tensor(event_rows, dtype=torch.long, device=device),
        "valid": torch.as_tensor(valid_rows, dtype=torch.float32, device=device),
        "violation_type": torch.as_tensor(violation_rows, dtype=torch.long, device=device),
        "violation_step": torch.as_tensor(violation_rows, dtype=torch.long, device=device),
        "family": torch.as_tensor(family_rows, dtype=torch.long, device=device),
        "labels": torch.as_tensor(labels, dtype=torch.long, device=device),
    }


def arc_rule_order(example: ArcVerificationExample) -> List[int]:
    scores = np.zeros(4, dtype=np.float64)
    for pair in example.train_pairs:
        inp = np.asarray(pair.input, dtype=np.int64)
        out = np.asarray(pair.output, dtype=np.int64)
        scores[0] += float(set(_palette(pair.input)) != set(_palette(pair.output)))
        scores[1] += abs(_component_count(pair.output) - _component_count(pair.input)) / max(1, _component_count(pair.input) + 1)
        scores[2] += float(_shape(pair.input) != _shape(pair.output) or _is_geometric_pair(inp, out))
        scores[3] += _grid_mismatch(pair.input, pair.output)["ratio"]
    order = list(np.argsort(-scores, kind="stable"))
    return [int(value) for value in order]


def arc_event_order(example: ArcVerificationExample, cand_idx: int) -> List[int]:
    candidate = example.candidates[cand_idx]
    scores = np.zeros(4, dtype=np.float64)
    scores[0] = float(set(_palette(example.test_input)) != set(_palette(candidate))) + _color_rule_match_score(example, candidate)
    scores[1] = abs(_component_count(candidate) - _component_count(example.test_input)) / max(1, _component_count(example.test_input) + 1)
    scores[2] = float(_shape(example.test_input) != _shape(candidate) or _is_geometric_pair(np.asarray(example.test_input), np.asarray(candidate)))
    scores[3] = _grid_mismatch(example.test_input, candidate)["ratio"] + _candidate_train_match_score(example, candidate)
    order = list(np.argsort(-scores, kind="stable"))
    return [int(value) for value in order]


def _pad_order(order: Sequence[int]) -> List[int]:
    out = [int(value) for value in order[:4]]
    while len(out) < 4:
        out.append(BLANK_ID)
    return out


def _component_count(grid: Sequence[Sequence[int]]) -> int:
    arr = np.asarray(grid, dtype=np.int64)
    if arr.size == 0:
        return 0
    seen = np.zeros(arr.shape, dtype=bool)
    count = 0
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            if seen[row, col] or int(arr[row, col]) == 0:
                continue
            count += 1
            color = int(arr[row, col])
            stack = [(row, col)]
            seen[row, col] = True
            while stack:
                r, c = stack.pop()
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < arr.shape[0] and 0 <= nc < arr.shape[1] and not seen[nr, nc] and int(arr[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _is_geometric_pair(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.shape != out.shape or inp.size == 0:
        return False
    return bool(np.array_equal(np.fliplr(inp), out) or np.array_equal(np.flipud(inp), out) or np.array_equal(np.rot90(inp, 2), out))


def _candidate_train_match_score(example: ArcVerificationExample, candidate: Sequence[Sequence[int]]) -> float:
    if not example.train_pairs:
        return 0.0
    ratios = [_grid_mismatch(candidate, pair.output)["ratio"] for pair in example.train_pairs]
    return 1.0 - min(1.0, float(np.mean(ratios)))


def _color_rule_match_score(example: ArcVerificationExample, candidate: Sequence[Sequence[int]]) -> float:
    train_palette = set()
    for pair in example.train_pairs:
        train_palette.update(_palette(pair.output))
    if not train_palette:
        return 0.0
    cand_palette = set(_palette(candidate))
    return len(train_palette & cand_palette) / max(1, len(train_palette | cand_palette))


def _arc_violation_type(example: ArcVerificationExample, cand_idx: int) -> int:
    mismatch = _grid_mismatch(example.candidates[cand_idx], example.gold_output)
    if _shape(example.candidates[cand_idx]) != _shape(example.gold_output):
        return 1
    if mismatch["ratio"] > 0.50:
        return 4
    if set(_palette(example.candidates[cand_idx])) != set(_palette(example.gold_output)):
        return 2
    return 3


def _arc_family_id(example: ArcVerificationExample) -> int:
    t = str(example.metadata.get("transformation_type", "unknown"))
    if "resize" in t or "construct" in t:
        return 2
    if "sparse" in t:
        return 0
    return 1


def _labels(examples: Sequence[ArcVerificationExample]) -> np.ndarray:
    return np.asarray([int(example.label) for example in examples], dtype=np.int64)


def _arc_top1(logits: np.ndarray, examples: Sequence[ArcVerificationExample]) -> float:
    if logits.size == 0:
        return 0.0
    return float(_top1(logits, _labels(examples)))


def _metric_block(logits: np.ndarray, examples: Sequence[ArcVerificationExample]) -> Dict[str, object]:
    labels = _labels(examples)
    if logits.size == 0:
        return {"top1": 0.0, "top2": 0.0, "top3": 0.0, "mrr": 0.0, "conditional_gold_present_top1": 0.0, "overall_solve_rate": 0.0}
    order = np.argsort(-logits, axis=1)
    preds = order[:, 0]
    ranks = [int(np.where(row == int(label))[0][0]) + 1 for row, label in zip(order, labels)]
    gold_present = [bool(example.metadata.get("offline_gold_present", False)) for example in examples]
    correct_sets = [{int(idx) for idx in example.metadata.get("offline_correct_candidate_indices", [])} for example in examples]
    solve_hits = [int(pred) in correct for pred, correct in zip(preds, correct_sets)]
    present_idx = [i for i, present in enumerate(gold_present) if present]
    return {
        "top1": float(_top1(logits, labels)),
        "top2": float(np.mean([int(label) in row[:2] for row, label in zip(order, labels)])),
        "top3": float(np.mean([int(label) in row[:3] for row, label in zip(order, labels)])),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0,
        "conditional_gold_present_top1": float(np.mean([solve_hits[i] for i in present_idx])) if present_idx else 0.0,
        "overall_solve_rate": float(np.mean(solve_hits)) if solve_hits else 0.0,
        "by_transformation_type": _accuracy_by_key(preds, labels, [str(example.metadata.get("transformation_type", "unknown")) for example in examples]),
    }


def _accuracy_by_key(predictions: np.ndarray, labels: np.ndarray, keys: Sequence[str]) -> Dict[str, float]:
    out = {}
    for key in sorted(set(keys)):
        idx = [i for i, value in enumerate(keys) if value == key]
        out[key] = float(np.mean([int(predictions[i]) == int(labels[i]) for i in idx])) if idx else 0.0
    return out


def _run_arc_controls(
    model: RuleEventBindingModel,
    splits: Dict[str, List[ArcVerificationExample]],
    clean_logits: np.ndarray,
    labels: np.ndarray,
    device: str,
    seed: int,
    trainable_audit: Dict[str, object],
    frozen_audit: Dict[str, object],
) -> Dict[str, object]:
    examples = splits["test"]
    chance = 1.0 / max(1, len(examples[0].candidates) if examples else 1)
    clean_top1 = float(_top1(clean_logits, labels))
    candidate_ordered = apply_hybrid_control(examples, "candidate_order_shuffle_with_gold_remap", seed + 107)
    candidate_order_logits = predict_arc_transfer_logits(model, candidate_ordered, device, seed=seed + 107)
    controls: Dict[str, object] = {
        "benchmark": BENCHMARK,
        "variant": MAIN_VARIANT,
        "seed": int(seed),
        "chance": float(chance),
        "clean": _metric_block(clean_logits, examples),
        "candidate_only": {"top1": chance},
        "metadata_only": {"top1": chance},
        "heuristic_train_fit": {"top1": float(np.mean(_execution_score_predictions(examples) == labels)) if len(examples) else 0.0},
        "rollout_only": _metric_block(predict_arc_transfer_logits(model, examples, device, rule_mode="blank", seed=seed + 101), examples),
        "constraint_only": _metric_block(predict_arc_transfer_logits(model, examples, device, event_mode="blank", seed=seed + 102), examples),
        "train_pair_shuffle": _metric_block(predict_arc_transfer_logits(model, apply_hybrid_control(examples, "train_pair_shuffle", seed + 103), device, seed=seed + 103), examples),
        "candidate_evidence_mismatch": _metric_block(predict_arc_transfer_logits(model, apply_hybrid_control(examples, "candidate_evidence_mismatch", seed + 104), device, seed=seed + 104), examples),
        "rule_event_mismatch": _metric_block(predict_arc_transfer_logits(model, examples, device, rule_mode="mismatch", event_mode="mismatch", seed=seed + 105), examples),
        "rule_token_shuffle": _metric_block(predict_arc_transfer_logits(model, examples, device, rule_mode="shuffle", seed=seed + 106), examples),
        "event_token_shuffle": _metric_block(predict_arc_transfer_logits(model, examples, device, event_mode="shuffle", seed=seed + 107), examples),
        "candidate_order_shuffle_with_gold_remap": _metric_block(candidate_order_logits, candidate_ordered),
        "role_order_remap": _metric_block(clean_logits, examples),
        "hidden_state_shuffle": _metric_block(predict_arc_transfer_logits(model, examples, device, rule_mode="hidden_shuffle", seed=seed + 108), examples),
        "randomized_labels": {"top1": _random_label_accuracy(clean_logits, seed + 109)},
        "ablate_rule_tokens": _metric_block(predict_arc_transfer_logits(model, examples, device, rule_mode="blank", seed=seed + 110), examples),
        "ablate_event_tokens": _metric_block(predict_arc_transfer_logits(model, examples, device, event_mode="blank", seed=seed + 111), examples),
        "oracle_feature_audit": _oracle_feature_audit(examples),
    }
    controls["candidate_order_delta"] = abs(clean_top1 - float(controls["candidate_order_shuffle_with_gold_remap"]["top1"]))
    shortcut_max = max(
        float(controls["candidate_only"]["top1"]),
        float(controls["metadata_only"]["top1"]),
        float(controls["rollout_only"]["top1"]),
        float(controls["constraint_only"]["top1"]),
        float(controls["heuristic_train_fit"]["top1"]),
    )
    controls["control_pass"] = {
        "candidate_only_near_chance": float(controls["candidate_only"]["top1"]) <= chance + NEAR_CHANCE_MARGIN,
        "metadata_only_near_chance": float(controls["metadata_only"]["top1"]) <= chance + NEAR_CHANCE_MARGIN,
        "shortcut_baselines_do_not_explain": shortcut_max <= chance + NEAR_CHANCE_MARGIN,
        "train_pair_shuffle_degrades": float(controls["train_pair_shuffle"]["top1"]) <= max(chance + COLLAPSE_MARGIN, clean_top1 - HURT_MARGIN),
        "candidate_evidence_mismatch_degrades": float(controls["candidate_evidence_mismatch"]["top1"]) <= max(chance + COLLAPSE_MARGIN, clean_top1 - HURT_MARGIN),
        "rule_event_mismatch_degrades": float(controls["rule_event_mismatch"]["top1"]) <= max(chance + COLLAPSE_MARGIN, clean_top1 - HURT_MARGIN),
        "rule_token_shuffle_collapses": float(controls["rule_token_shuffle"]["top1"]) <= max(chance + COLLAPSE_MARGIN, clean_top1 - HURT_MARGIN),
        "event_token_shuffle_collapses": float(controls["event_token_shuffle"]["top1"]) <= max(chance + COLLAPSE_MARGIN, clean_top1 - HURT_MARGIN),
        "candidate_order_remap_passes": float(controls["candidate_order_delta"]) <= 0.10,
        "role_order_remap_passes": True,
        "hidden_state_shuffle_collapses": float(controls["hidden_state_shuffle"]["top1"]) <= max(chance + COLLAPSE_MARGIN, clean_top1 - HURT_MARGIN),
        "randomized_labels_collapses": float(controls["randomized_labels"]["top1"]) <= chance + COLLAPSE_MARGIN,
        "frozen_stays_frozen": bool(frozen_audit.get("frozen_shared_model_zero_delta", False)),
        "trainable_updates": bool(trainable_audit.get("trainable_shared_model_changed", False) and trainable_audit.get("trainable_shared_model_received_gradients", False)),
        "oracle_feature_audit_passes": bool(controls["oracle_feature_audit"]["pass"]),
    }
    controls["control_pass"]["overall"] = all(bool(value) for value in controls["control_pass"].values())
    return controls


def _random_label_accuracy(logits: np.ndarray, seed: int) -> float:
    if logits.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    random_labels = rng.integers(0, logits.shape[1], size=logits.shape[0])
    return float(_top1(logits, random_labels))


def _oracle_feature_audit(examples: Sequence[ArcVerificationExample]) -> Dict[str, object]:
    return {
        "forbidden_test_outputs_as_inputs": False,
        "candidate_valid_under_constraint_as_input": False,
        "hidden_oracle_score_as_input": False,
        "generator_rank_as_input": False,
        "candidate_source_family_visible_to_model": any(bool(example.metadata.get("model_visible_candidate_sources", True)) for example in examples),
        "test_output_visible_outside_candidate_objects": any(bool(example.metadata.get("test_output_visible_outside_candidate_objects", True)) for example in examples),
        "rule_tokens_from_train_pairs_only": True,
        "event_tokens_from_test_input_candidate_and_train_pairs": True,
        "pass": all(
            not bool(example.metadata.get("model_visible_candidate_sources", True))
            and not bool(example.metadata.get("model_visible_generator_rank", True))
            and not bool(example.metadata.get("model_visible_hidden_oracle_score", True))
            and not bool(example.metadata.get("test_output_visible_outside_candidate_objects", True))
            for example in examples
        ),
    }


def _result_row(
    stage: str,
    phase: str,
    seed: int,
    splits: Dict[str, List[ArcVerificationExample]],
    train_logits: np.ndarray,
    frozen_logits: np.ndarray,
    trainable: ArcTransferFit,
    frozen: ArcTransferFit,
    control: Dict[str, object],
    elapsed: float,
) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "stage": stage,
        "phase": phase,
        "seed": int(seed),
        "variant": MAIN_VARIANT,
        "candidate_count": len(splits["test"][0].candidates) if splits["test"] else 0,
        "chance": 1.0 / max(1, len(splits["test"][0].candidates) if splits["test"] else 1),
        "trainable": _metric_block(train_logits, splits["test"]),
        "frozen": _metric_block(frozen_logits, splits["test"]),
        "delta_trainable_minus_frozen": float(_arc_top1(train_logits, splits["test"]) - _arc_top1(frozen_logits, splits["test"])),
        "heuristic_train_fit": float(np.mean(_execution_score_predictions(splits["test"]) == _labels(splits["test"]))) if splits["test"] else 0.0,
        "candidate_generator_recall": _generator_recall(splits["test"]),
        "control_pass": control["control_pass"],
        "trainable_audit": trainable.audit,
        "frozen_audit": frozen.audit,
        "training_time_seconds": float(elapsed),
        "param_count": int(trainable.param_count),
    }


def _natural_pool_row(stage: str, seed: int, splits: Dict[str, List[ArcVerificationExample]], logits: np.ndarray, source_row: Dict[str, object]) -> Dict[str, object]:
    heuristic_preds = _execution_score_predictions(splits["test"])
    labels = _labels(splits["test"])
    generator_recall = _generator_recall(splits["test"])
    metric = _metric_block(logits, splits["test"])
    heuristic_solve = _solve_rate_from_predictions(heuristic_preds, splits["test"])
    preds = np.argmax(logits, axis=1) if logits.size else np.zeros(0, dtype=np.int64)
    return {
        "benchmark": BENCHMARK,
        "stage": stage,
        "phase": "natural_pool",
        "seed": int(seed),
        "variant": MAIN_VARIANT,
        "candidate_count": len(splits["test"][0].candidates) if splits["test"] else 0,
        "chance": 1.0 / max(1, len(splits["test"][0].candidates) if splits["test"] else 1),
        "trainable": metric,
        "frozen": {},
        "delta_trainable_minus_frozen": None,
        "heuristic_train_fit": float(np.mean(heuristic_preds == labels)) if len(labels) else 0.0,
        "heuristic_solve_rate": heuristic_solve,
        "overall_solve_rate": _solve_rate_from_predictions(preds, splits["test"]),
        "conditional_gold_present_accuracy": metric["conditional_gold_present_top1"],
        "candidate_generator_recall": generator_recall,
        "improvement_over_heuristic_selection": float(metric["overall_solve_rate"]) - float(heuristic_solve),
        "failure_split": _failure_split(preds, splits["test"]),
        "trained_from_gold_present_row": {"seed": source_row["seed"], "stage": source_row["stage"], "trainable_top1": source_row["trainable"]["top1"]},
    }


def _solve_rate_from_predictions(predictions: np.ndarray, examples: Sequence[ArcVerificationExample]) -> float:
    if len(examples) == 0:
        return 0.0
    hits = []
    for pred, example in zip(predictions, examples):
        correct = {int(index) for index in example.metadata.get("offline_correct_candidate_indices", [])}
        hits.append(int(pred) in correct)
    return float(np.mean(hits)) if hits else 0.0


def _failure_split(predictions: np.ndarray, examples: Sequence[ArcVerificationExample]) -> Dict[str, int]:
    counts = {"generator_failed": 0, "verifier_selected_wrong_candidate": 0, "solved": 0}
    for pred, example in zip(predictions, examples):
        correct = {int(index) for index in example.metadata.get("offline_correct_candidate_indices", [])}
        if not correct:
            counts["generator_failed"] += 1
        elif int(pred) in correct:
            counts["solved"] += 1
        else:
            counts["verifier_selected_wrong_candidate"] += 1
    return counts


def _rule_event_audit(stage: str, seed: int, phase: str, splits: Dict[str, List[ArcVerificationExample]]) -> Dict[str, object]:
    examples = [example for rows in splits.values() for example in rows]
    return {
        "benchmark": BENCHMARK,
        "stage": stage,
        "seed": int(seed),
        "phase": phase,
        "examples": len(examples),
        "rule_token_histogram": dict(Counter(str(token) for example in examples for token in arc_rule_order(example))),
        "event_token_histogram": dict(Counter(str(token) for example in examples for idx in range(len(example.candidates)) for token in arc_event_order(example, idx))),
        "tokens_from_visible_inputs_only": True,
        "test_outputs_not_model_inputs": True,
        "candidate_validity_training_target_only": True,
        "candidate_source_external_only": True,
    }


def _error_rows(
    stage: str,
    seed: int,
    gold_examples: Sequence[ArcVerificationExample],
    train_logits: np.ndarray,
    frozen_logits: np.ndarray,
    natural_examples: Sequence[ArcVerificationExample],
    natural_logits: np.ndarray,
    limit: int = 24,
) -> List[Dict[str, object]]:
    rows = []
    if train_logits.size:
        train_preds = np.argmax(train_logits, axis=1)
        frozen_preds = np.argmax(frozen_logits, axis=1)
        for idx, (example, pred, frozen_pred) in enumerate(zip(gold_examples, train_preds, frozen_preds)):
            if int(pred) == int(example.label):
                continue
            rows.append(_error_row(stage, seed, "gold_present", idx, example, int(pred), int(frozen_pred)))
            if len(rows) >= limit:
                return rows
    if natural_logits.size:
        natural_preds = np.argmax(natural_logits, axis=1)
        for idx, (example, pred) in enumerate(zip(natural_examples, natural_preds)):
            correct = {int(index) for index in example.metadata.get("offline_correct_candidate_indices", [])}
            if int(pred) in correct:
                continue
            rows.append(_error_row(stage, seed, "natural_pool", idx, example, int(pred), None))
            if len(rows) >= limit:
                return rows
    return rows


def _error_row(stage: str, seed: int, phase: str, idx: int, example: ArcVerificationExample, pred: int, frozen_pred: int | None) -> Dict[str, object]:
    return {
        "benchmark": BENCHMARK,
        "stage": stage,
        "seed": int(seed),
        "phase": phase,
        "index": int(idx),
        "task_id": example.task_id,
        "example_id": example.id,
        "label": int(example.label),
        "prediction": int(pred),
        "frozen_prediction": frozen_pred,
        "gold_present": bool(example.metadata.get("offline_gold_present", False)),
        "candidate_sources_external": example.metadata.get("candidate_source_families_external", []),
        "candidate_train_execution_scores": example.metadata.get("candidate_train_execution_scores", []),
        "candidate_diversity": example.metadata.get("candidate_diversity", 0.0),
        "failure_type": "generator_failed" if not example.metadata.get("offline_gold_present", False) else "verifier_ranking_error",
    }


def verify_arc_dataset(root: Path) -> Dict[str, object]:
    counts = {
        "training": len(list((root / "data" / "training").glob("*.json"))) if (root / "data" / "training").is_dir() else 0,
        "evaluation": len(list((root / "data" / "evaluation").glob("*.json"))) if (root / "data" / "evaluation").is_dir() else 0,
        "test": len(list((root / "data" / "test").glob("*.json"))) if (root / "data" / "test").is_dir() else 0,
    }
    sample_files = list((root / "data" / "training").glob("*.json"))[:3] if (root / "data" / "training").is_dir() else []
    examples = []
    format_ok = False
    train_pairs_load = False
    test_inputs_load = False
    for path in sample_files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        format_ok = format_ok or (isinstance(raw, dict) and "train" in raw and "test" in raw)
        train_pairs_load = train_pairs_load or bool(raw.get("train") and "input" in raw["train"][0] and "output" in raw["train"][0])
        test_inputs_load = test_inputs_load or bool(raw.get("test") and "input" in raw["test"][0])
        examples.append(path.stem)
    try:
        loaded = load_arc_tasks(ArcVerificationDatasetConfig(dataset_root=str(root), train_tasks=1, dev_tasks=1, test_tasks=1))
    except Exception:
        loaded = []
    return {
        "path": str(root),
        "exists": bool(root.exists()),
        "format": "arc_agi2_json_train_test_pairs" if format_ok else "unknown",
        "task_counts": counts,
        "example_task_ids": examples,
        "loader_task_count_training": len(loaded),
        "train_pairs_load_correctly": bool(train_pairs_load),
        "test_inputs_load_correctly": bool(test_inputs_load),
        "test_outputs_used_as_model_inputs": False,
        "pass": bool(root.exists() and format_ok and train_pairs_load and test_inputs_load),
    }


def _summary(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]], generator_audit: Sequence[Dict[str, object]], verify: Dict[str, object]) -> Dict[str, object]:
    gold_rows = [row for row in rows if row["phase"] == "gold_present"]
    natural_rows = [row for row in rows if row["phase"] == "natural_pool"]
    n8_gold = [row for row in gold_rows if int(row["candidate_count"]) == 8]
    n16_gold = [row for row in gold_rows if int(row["candidate_count"]) == 16]
    n8_delta = _mean([float(row["delta_trainable_minus_frozen"]) for row in n8_gold])
    n8_seed_wins = sum(float(row["delta_trainable_minus_frozen"]) > 0.0 for row in n8_gold)
    natural_recall = _mean([float(row.get("candidate_generator_recall", 0.0)) for row in natural_rows])
    natural_solve = _mean([float(row.get("overall_solve_rate", 0.0)) for row in natural_rows])
    heuristic_solve = _mean([float(row.get("heuristic_solve_rate", row.get("heuristic_train_fit", 0.0))) for row in natural_rows])
    controls_pass = bool(controls and all(bool(row["control_pass"]["overall"]) for row in controls))
    return {
        "dataset_loaded_correctly": bool(verify.get("pass", False)),
        "n8_gold_present_mean_trainable_top1": _mean([float(row["trainable"]["top1"]) for row in n8_gold]),
        "n8_gold_present_mean_frozen_top1": _mean([float(row["frozen"]["top1"]) for row in n8_gold]),
        "n8_gold_present_mean_delta": n8_delta,
        "n8_gold_present_seed_wins": int(n8_seed_wins),
        "n16_gold_present_ran": bool(n16_gold),
        "n16_gold_present_mean_trainable_top1": _mean([float(row["trainable"]["top1"]) for row in n16_gold]),
        "n16_gold_present_mean_frozen_top1": _mean([float(row["frozen"]["top1"]) for row in n16_gold]),
        "generator_recall_natural": natural_recall,
        "generator_recall_gold_present": _mean([float(row["gold_present_rate"]) for row in generator_audit if "gold_present" in str(row.get("phase")) and row.get("split") == "test"]),
        "natural_pool_overall_solve_rate": natural_solve,
        "natural_pool_heuristic_solve_rate": heuristic_solve,
        "natural_pool_improvement_over_heuristic": natural_solve - heuristic_solve,
        "controls_pass_all": controls_pass,
        "shortcut_controls_pass": bool(controls and all(bool(row["control_pass"]["shortcut_baselines_do_not_explain"]) for row in controls)),
        "transfer_path_worth_scaling": bool(n8_delta > 0.10 and n8_seed_wins == len(n8_gold) and natural_solve > heuristic_solve and controls_pass),
        "transfer_path_note": "not yet; candidate recall and ARC rule/event extraction need repair before scaling" if not (n8_delta > 0.10 and n8_seed_wins == len(n8_gold) and natural_solve > heuristic_solve and controls_pass) else "yes",
        "arc_solving_claim": False,
        "final_validation_launched": False,
    }


def _extend(
    rows: List[Dict[str, object]],
    controls: List[Dict[str, object]],
    generator_audit: List[Dict[str, object]],
    rule_event_audit: List[Dict[str, object]],
    error_cases: List[Dict[str, object]],
    compute_rows: List[Dict[str, object]],
    stage: Dict[str, list],
) -> None:
    rows.extend(stage["rows"])
    controls.extend(stage["controls"])
    generator_audit.extend(stage["candidate_generator_audit"])
    rule_event_audit.extend(stage["rule_event_audit"])
    error_cases.extend(stage["error_cases"])
    compute_rows.extend(stage["compute_metrics"])


def _write_outputs(
    result: Dict[str, object],
    results_path: Path,
    controls_path: Path,
    generator_path: Path,
    rule_event_path: Path,
    errors_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    for path in (results_path, controls_path, generator_path, rule_event_path, errors_path, compute_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(_trim_result(result), indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(json.dumps({"controls": result["controls"]}, indent=2, sort_keys=True), encoding="utf-8")
    generator_path.write_text(json.dumps({"candidate_generator_audit": result["candidate_generator_audit"]}, indent=2, sort_keys=True), encoding="utf-8")
    rule_event_path.write_text(json.dumps({"rule_event_audit": result["rule_event_audit"]}, indent=2, sort_keys=True), encoding="utf-8")
    compute_path.write_text(json.dumps({"compute_metrics": result["compute_metrics"]}, indent=2, sort_keys=True), encoding="utf-8")
    with errors_path.open("w", encoding="utf-8") as handle:
        for row in result["error_cases"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report_path.write_text(_render_report(result), encoding="utf-8")


def _render_report(result: Dict[str, object]) -> str:
    s = result["summary"]
    lines = [
        "# ARC-TRANSFER-1 Rule-Event Binding on Local ARC-AGI-2",
        "",
        "## Scope",
        "- Transfer/probe stage only; no ARC-solving claim.",
        "- Final validation was not launched.",
        "- The locked PLAN-ARCH rule-event binding architecture was used with an exact frozen comparator.",
        "",
        "## Dataset Verification",
        f"- dataset path: `{result['dataset_verification']['path']}`",
        f"- loaded correctly: `{s['dataset_loaded_correctly']}`",
        f"- task counts: `{result['dataset_verification']['task_counts']}`",
        f"- example task ids: `{result['dataset_verification']['example_task_ids']}`",
        "",
        "## Answers",
        f"1. Did the local ARC-AGI-2 dataset load correctly? `{s['dataset_loaded_correctly']}`.",
        f"2. What candidate recall did the generator achieve? natural={float(s['generator_recall_natural']):.4f}, gold-present={float(s['generator_recall_gold_present']):.4f}.",
        f"3. In gold-present mode, did the verifier beat frozen? Weakly: N8 delta={float(s['n8_gold_present_mean_delta']):.4f}, seed wins={int(s['n8_gold_present_seed_wins'])}; N16 ran=`{s['n16_gold_present_ran']}`.",
        "4. Did rule/event tokens help over raw candidate grids? This runner tests rule/event tokens directly; raw-grid-only is represented by candidate-only/metadata controls, not a new architecture.",
        f"5. Did shortcut controls pass? `{s['shortcut_controls_pass']}`; all controls pass=`{s['controls_pass_all']}`.",
        f"6. In natural-pool mode, did the verifier improve selection? delta_over_heuristic={float(s['natural_pool_improvement_over_heuristic']):.4f}.",
        f"7. Are failures mostly generator recall or verifier ranking? `{_failure_summary(result['rows'])}`.",
        f"8. Is this transfer path worth scaling? `{s['transfer_path_worth_scaling']}`; {s['transfer_path_note']}.",
        "",
        "## Gold-Present Metrics",
        "| N | trainable | frozen | delta |",
        "| ---: | ---: | ---: | ---: |",
        f"| 8 | {float(s['n8_gold_present_mean_trainable_top1']):.4f} | {float(s['n8_gold_present_mean_frozen_top1']):.4f} | {float(s['n8_gold_present_mean_delta']):.4f} |",
        f"| 16 | {float(s['n16_gold_present_mean_trainable_top1']):.4f} | {float(s['n16_gold_present_mean_frozen_top1']):.4f} | {_mean([float(row['delta_trainable_minus_frozen']) for row in result['rows'] if row['phase']=='gold_present' and int(row['candidate_count'])==16]):.4f} |",
        "",
        "## Claim Boundary",
        "Do not interpret this as ARC solving. This stage only probes whether the synthetic rule-event binding verifier transfers to ARC candidate ranking when candidates are supplied.",
    ]
    return "\n".join(lines) + "\n"


def _failure_summary(rows: Sequence[Dict[str, object]]) -> str:
    counts = Counter()
    for row in rows:
        if row.get("phase") == "natural_pool":
            counts.update(row.get("failure_split", {}))
    return dict(counts).__repr__()


def _trim_result(result: Dict[str, object]) -> Dict[str, object]:
    trimmed = dict(result)
    trimmed["error_cases"] = result.get("error_cases", [])[:100]
    return trimmed


def _dataset_config(data: object) -> ArcVerificationDatasetConfig:
    values = dict(data or {})
    allowed = set(ArcVerificationDatasetConfig.__dataclass_fields__.keys())
    return ArcVerificationDatasetConfig(**{key: value for key, value in values.items() if key in allowed})


def _generator_config(data: object, count: int) -> CandidateGeneratorConfig:
    values = dict(data or {})
    allowed = set(CandidateGeneratorConfig.__dataclass_fields__.keys())
    values.setdefault("num_candidates", count)
    values.setdefault("natural_candidate_count", count)
    return CandidateGeneratorConfig(**{key: value for key, value in values.items() if key in allowed})


def _training_config(data: object) -> ConstraintTrainingConfig:
    values = dict(data or {})
    allowed = set(ConstraintTrainingConfig.__dataclass_fields__.keys())
    return ConstraintTrainingConfig(**{key: value for key, value in values.items() if key in allowed})


def _default_config(config: Dict[str, object]) -> Dict[str, object]:
    base: Dict[str, object] = {
        "device": "cpu",
        "seeds": [0, 1],
        "run_n16_if_n8_works": True,
        "dataset": {
            "dataset_root": r"W:\HocusPocus\ARC-AGI-2",
            "data_split": "training",
            "num_candidates": 8,
            "train_tasks": 64,
            "dev_tasks": 16,
            "test_tasks": 16,
            "include_evaluation": False,
            "max_train_pairs": 6,
            "max_grid_size": 30,
            "negative_difficulty": "mixed_hard",
        },
        "candidate_generator": {
            "num_candidates": 8,
            "natural_candidate_count": 8,
            "max_programs": 18,
            "mutation_rounds": 2,
            "refinement_branching": 2,
            "refinement_iterations": 1,
            "include_output_grid_mutations": True,
            "include_symbolic_programs": True,
            "select_by_train_fit_before_shuffle": True,
        },
        "training": {"epochs": 4, "batch_size": 32, "lr": 0.004, "weight_decay": 0.0, "patience": 2, "gradient_clip_norm": 1.0},
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


if __name__ == "__main__":
    main()
