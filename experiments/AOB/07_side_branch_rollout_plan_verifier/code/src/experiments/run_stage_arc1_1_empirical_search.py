from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.datasets.arc_agi2_verification import (
    ArcVerificationDatasetConfig,
    ArcVerificationExample,
    apply_arc_control,
    build_arc_verification_splits,
    candidate_metadata_only_accuracy,
    leakage_audit_rows,
)
from src.experiments.run_stage_arc1_latent_rule_clones import (
    ArcFitResult,
    ArcModelConfig,
    ArcTrainingConfig,
    collate_arc_batch,
    fit_arc_verifier,
    predict_logits,
    _accuracy,
    _baseline_block,
    _candidate_objective_loss,
    _clear_cuda,
    _metric_block,
    _resolve_device,
    _shape,
    _top1,
)


BENCHMARK = "arc1_1_empirical_search"
DEFAULT_CONFIG = "configs/stage_arc1_1_empirical_search_smoke.json"
DEFAULT_RESULTS = "results/arc1_1_empirical_search_results.json"
DEFAULT_CONTROLS = "results/arc1_1_empirical_search_controls.json"
DEFAULT_LEADERBOARD = "results/arc1_1_variant_leaderboard.csv"
DEFAULT_FAILURES = "results/arc1_1_failure_taxonomy.json"
DEFAULT_OVERFIT = "results/arc1_1_overfit_curves.json"
DEFAULT_ERRORS = "results/arc1_1_error_cases.jsonl"
DEFAULT_REPORT = "reports/STAGE_ARC1_1_EMPIRICAL_SEARCH.md"


@dataclass(frozen=True)
class SearchVariant:
    name: str
    family: str
    representation: str
    view_set: Tuple[str, ...]
    interaction: str
    objective: str
    negative_difficulty: str = "mixed_hard"
    model_kind: str = "rule_clone"
    candidate_count: int = 8
    coordination_blocks: int = 1
    learned_slots: bool = False
    repeated_avenues: int = 1
    diagnostic_only: bool = False


@dataclass(frozen=True)
class FeatureTrainingConfig:
    epochs: int = 80
    batch_size: int = 16
    lr: float = 0.003
    weight_decay: float = 0.0001
    patience: int = 10
    hidden_dim: int = 64


@dataclass
class FeatureFitResult:
    method: str
    model: "FeatureVerifier"
    trainable: bool
    audit: Dict[str, object]
    history: List[Dict[str, float]]
    training_time_seconds: float
    param_count: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage ARC-1.1 empirical ARC candidate verifier search.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--leaderboard-output", default=DEFAULT_LEADERBOARD)
    parser.add_argument("--failure-output", default=DEFAULT_FAILURES)
    parser.add_argument("--overfit-output", default=DEFAULT_OVERFIT)
    parser.add_argument("--error-output", default=DEFAULT_ERRORS)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-variants", type=int, default=0)
    args = parser.parse_args()
    config = _load_config(Path(args.config))
    if args.dataset_root:
        config.setdefault("dataset", {})["dataset_root"] = str(args.dataset_root)
    if args.device:
        config["device"] = str(args.device)
    result = run_arc1_1_search(config, max_variants=int(args.max_variants))
    _write_outputs(
        result,
        Path(args.output),
        Path(args.controls_output),
        Path(args.leaderboard_output),
        Path(args.failure_output),
        Path(args.overfit_output),
        Path(args.error_output),
        Path(args.report),
    )


def run_arc1_1_search(config: Dict[str, object], max_variants: int = 0) -> Dict[str, object]:
    dataset_config = _dataset_config(config.get("dataset", {}))
    base_training = _training_config(config.get("training", {}))
    feature_training = _feature_training_config(config.get("feature_training", {}))
    overfit_training = _training_config({**dict(config.get("training", {})), **dict(config.get("overfit_training", {}))})
    seeds = [int(seed) for seed in config.get("seeds", [0])]
    device = _resolve_device(str(config.get("device", "auto")))
    variants = _variant_plan()
    if max_variants:
        variants = variants[:max_variants]
    if int(config.get("max_variants", 0)):
        variants = variants[: int(config.get("max_variants", 0))]

    rows: List[Dict[str, object]] = []
    controls: List[Dict[str, object]] = []
    overfit_rows: List[Dict[str, object]] = []
    error_cases: List[Dict[str, object]] = []
    print(f"arc1.1: device={device} variants={len(variants)} seeds={seeds}")
    for variant in variants:
        for seed in seeds:
            print(f"arc1.1 variant={variant.name} seed={seed}: overfit gates")
            gate_rows = _run_overfit_gates(variant, dataset_config, overfit_training, feature_training, seed, device)
            overfit_rows.extend(gate_rows)
            gates_pass = all(bool(row["pass"]) for row in gate_rows if row.get("gate_required", True))
            row_base = {
                "benchmark": BENCHMARK,
                "seed": seed,
                "variant": variant.name,
                "family": variant.family,
                "model_kind": variant.model_kind,
                "representation": variant.representation,
                "view_set": list(variant.view_set),
                "interaction": variant.interaction,
                "objective": variant.objective,
                "negative_difficulty": variant.negative_difficulty,
                "candidate_count": int(variant.candidate_count),
                "diagnostic_only": bool(variant.diagnostic_only),
                "overfit_gates_pass": gates_pass,
            }
            if not gates_pass:
                rows.append({**row_base, "status": "failed_overfit_gate", "failure_reason": _first_failed_gate(gate_rows)})
                continue

            split_config = replace(
                dataset_config,
                num_candidates=int(variant.candidate_count),
                negative_difficulty=str(variant.negative_difficulty),
            )
            splits = build_arc_verification_splits(split_config, seed=seed)
            leak_pass = all(row.get("pass") for row in leakage_audit_rows([example for part in splits.values() for example in part]))
            print(f"arc1.1 variant={variant.name} seed={seed}: cheap held-out fit")
            start = time.perf_counter()
            if variant.model_kind == "feature_mlp":
                trainable = fit_feature_verifier(splits["train"], splits["dev"], feature_training, seed + 501, trainable=True, method=f"trainable__{variant.name}")
                frozen = fit_feature_verifier(splits["train"], splits["dev"], feature_training, seed + 501, trainable=False, method=f"frozen__{variant.name}")
                dev_logits = predict_feature_logits(trainable.model, splits["dev"])
                frozen_dev_logits = predict_feature_logits(frozen.model, splits["dev"])
                test_logits = predict_feature_logits(trainable.model, splits["test"])
                predict_fn = lambda examples, condition="none", s=seed: _predict_feature_control(trainable.model, examples, condition, s)
            else:
                model_config = _model_config_for_variant(variant)
                training = replace(base_training, objective=variant.objective)
                trainable = fit_arc_verifier(
                    splits["train"],
                    splits["dev"],
                    model_config,
                    training,
                    seed=seed + 701,
                    device=device,
                    trainable_shared=True,
                    method=f"trainable__{variant.name}",
                )
                frozen = fit_arc_verifier(
                    splits["train"],
                    splits["dev"],
                    model_config,
                    training,
                    seed=seed + 701,
                    device=device,
                    trainable_shared=False,
                    method=f"frozen__{variant.name}",
                )
                dev_logits = predict_logits(trainable.model, splits["dev"], model_config, training.batch_size, device)
                frozen_dev_logits = predict_logits(frozen.model, splits["dev"], model_config, training.batch_size, device)
                test_logits = predict_logits(trainable.model, splits["test"], model_config, training.batch_size, device)
                predict_fn = lambda examples, condition="none", s=seed, mc=model_config, bs=training.batch_size: predict_logits(
                    trainable.model, examples, mc, bs, device, condition=condition, seed=s
                )
            elapsed = time.perf_counter() - start
            dev_labels = np.asarray([example.label for example in splits["dev"]], dtype=np.int64)
            test_labels = np.asarray([example.label for example in splits["test"]], dtype=np.int64)
            control_row = _cheap_controls(variant, splits["dev"], predict_fn, dev_logits, dev_labels, seed)
            controls.append(control_row)
            baselines = _baseline_block(splits["train"], splits["dev"], splits["dev"])
            trainable_block = _metric_block(dev_logits, dev_labels, splits["dev"])
            frozen_block = _metric_block(frozen_dev_logits, dev_labels, splits["dev"])
            row = {
                **row_base,
                "status": "completed",
                "leakage_audit_pass": bool(leak_pass),
                "dev_trainable": trainable_block,
                "dev_frozen": frozen_block,
                "test_smoke_trainable": _metric_block(test_logits, test_labels, splits["test"]),
                "dev_delta_trainable_minus_frozen": float(trainable_block["top1"] - frozen_block["top1"]),
                "baselines": baselines,
                "control_pass": control_row["control_pass"],
                "trainable_audit": trainable.audit,
                "frozen_audit": frozen.audit,
                "training_time_seconds": float(elapsed),
                "param_count": int(trainable.param_count),
                "medium_validation_triggered": False,
            }
            rows.append(row)
            error_cases.extend(_error_rows(variant, seed, splits["dev"], dev_logits, frozen_dev_logits, limit=12))
            _clear_cuda()

    leaderboard = _leaderboard(rows, controls)
    failure_taxonomy = _failure_taxonomy(rows, overfit_rows, controls)
    summary = _summary(leaderboard)
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "created_at_utc": _now(),
            "device": device,
            "dataset_root": dataset_config.dataset_root,
            "scope": "bounded empirical architecture/search phase; no final validation launched",
            "central_rule": "mechanisms searchable; labels, leakage controls, frozen comparator, and final protocol fixed",
        },
        "dataset_config": asdict(dataset_config),
        "training_config": asdict(base_training),
        "feature_training_config": asdict(feature_training),
        "variant_plan": [asdict(variant) for variant in variants],
        "rows": rows,
        "controls": controls,
        "leaderboard": leaderboard,
        "failure_taxonomy": failure_taxonomy,
        "overfit_curves": overfit_rows,
        "error_cases": error_cases,
        "summary": summary,
    }


def _run_overfit_gates(
    variant: SearchVariant,
    dataset_config: ArcVerificationDatasetConfig,
    overfit_training: ArcTrainingConfig,
    feature_training: FeatureTrainingConfig,
    seed: int,
    device: str,
) -> List[Dict[str, object]]:
    gates = [
        ("4_examples_N2", 2, 4, 0.95),
        ("12_examples_N4", 4, 12, 0.90),
        ("35_examples_N8", 8, 35, 0.80),
    ]
    rows = []
    for gate_name, candidate_count, example_count, threshold in gates:
        gate_config = replace(
            dataset_config,
            num_candidates=candidate_count,
            train_tasks=max(example_count + 4, int(dataset_config.train_tasks)),
            dev_tasks=2,
            test_tasks=2,
            negative_difficulty=str(variant.negative_difficulty),
        )
        splits = build_arc_verification_splits(gate_config, seed=seed + candidate_count * 101)
        train_examples = splits["train"][:example_count]
        if len(train_examples) < example_count:
            rows.append(
                {
                    "variant": variant.name,
                    "seed": seed,
                    "gate": gate_name,
                    "candidate_count": candidate_count,
                    "examples": len(train_examples),
                    "train_accuracy": 0.0,
                    "threshold": threshold,
                    "pass": False,
                    "gate_required": True,
                    "reason": "not_enough_examples",
                }
            )
            break
        if variant.model_kind == "feature_mlp":
            fit = fit_feature_verifier(
                train_examples,
                train_examples,
                replace(feature_training, epochs=max(feature_training.epochs, 100), patience=max(feature_training.patience, 20)),
                seed=seed + 900 + candidate_count,
                trainable=True,
                method=f"overfit__{variant.name}__{gate_name}",
            )
            logits = predict_feature_logits(fit.model, train_examples)
        else:
            model_config = _model_config_for_variant(variant, candidate_count=candidate_count)
            training = replace(
                overfit_training,
                epochs=max(int(overfit_training.epochs), 20),
                patience=max(int(overfit_training.patience), 20),
                batch_size=min(max(1, example_count), int(overfit_training.batch_size)),
                objective=variant.objective,
            )
            fit = fit_arc_verifier(
                train_examples,
                train_examples,
                model_config,
                training,
                seed=seed + 900 + candidate_count,
                device=device,
                trainable_shared=True,
                method=f"overfit__{variant.name}__{gate_name}",
            )
            logits = predict_logits(fit.model, train_examples, model_config, training.batch_size, device)
        labels = np.asarray([example.label for example in train_examples], dtype=np.int64)
        acc = _top1(logits, labels)
        rows.append(
            {
                "variant": variant.name,
                "seed": seed,
                "gate": gate_name,
                "candidate_count": candidate_count,
                "examples": len(train_examples),
                "train_accuracy": float(acc),
                "threshold": float(threshold),
                "pass": bool(acc >= threshold),
                "gate_required": True,
                "history": fit.history,
                "audit": fit.audit,
            }
        )
        if acc < threshold:
            break
    return rows


class FeatureVerifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, candidates, dim = features.shape
        return self.net(features.reshape(batch * candidates, dim)).reshape(batch, candidates)


def fit_feature_verifier(
    train_examples: Sequence[ArcVerificationExample],
    dev_examples: Sequence[ArcVerificationExample],
    config: FeatureTrainingConfig,
    seed: int,
    trainable: bool,
    method: str,
) -> FeatureFitResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FeatureVerifier(input_dim=_feature_dim(), hidden_dim=int(config.hidden_dim))
    for parameter in model.parameters():
        parameter.requires_grad = bool(trainable)
    initial = _flat_feature_params(model)
    history: List[Dict[str, float]] = []
    start = time.perf_counter()
    if trainable:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        best_dev = -1.0
        bad = 0
        for epoch in range(int(config.epochs)):
            model.train()
            losses = []
            order = np.random.default_rng(seed + epoch).permutation(len(train_examples)).tolist()
            for batch_ids in _chunks(order, int(config.batch_size)):
                examples = [train_examples[index] for index in batch_ids]
                features, labels = _feature_batch(examples)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            dev_logits = predict_feature_logits(model, dev_examples)
            dev_labels = np.asarray([example.label for example in dev_examples], dtype=np.int64)
            dev_acc = _top1(dev_logits, dev_labels)
            history.append({"epoch": float(epoch), "loss": _mean(losses), "dev_top1": float(dev_acc)})
            if dev_acc > best_dev:
                best_dev = float(dev_acc)
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= int(config.patience):
                break
        model.load_state_dict(best_state)
    final = _flat_feature_params(model)
    audit = {
        "method": method,
        "diagnostic_feature_baseline": True,
        "trainable": bool(trainable),
        "parameter_delta": _l2(initial, final),
        "frozen_zero_delta": bool((not trainable) and _l2(initial, final) == 0.0),
    }
    return FeatureFitResult(method, model, trainable, audit, history, float(time.perf_counter() - start), sum(p.numel() for p in model.parameters()))


def predict_feature_logits(model: FeatureVerifier, examples: Sequence[ArcVerificationExample]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        features, _labels = _feature_batch(examples)
        return model(features).detach().cpu().numpy()


def _predict_feature_control(model: FeatureVerifier, examples: Sequence[ArcVerificationExample], condition: str, seed: int) -> np.ndarray:
    if condition == "none":
        return predict_feature_logits(model, examples)
    if condition == "physical_role_order_shuffle":
        return predict_feature_logits(model, examples)
    controlled = apply_arc_control(examples, condition, seed)
    return predict_feature_logits(model, controlled)


def _feature_batch(examples: Sequence[ArcVerificationExample]) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for example in examples:
        rows.append([_candidate_features(example, candidate) for candidate in example.candidates])
    return torch.as_tensor(rows, dtype=torch.float32), torch.as_tensor([example.label for example in examples], dtype=torch.long)


def _candidate_features(example: ArcVerificationExample, candidate: Sequence[Sequence[int]]) -> List[float]:
    cand = np.asarray(candidate, dtype=np.float32)
    test = np.asarray(example.test_input, dtype=np.float32)
    train_outs = [np.asarray(pair.output, dtype=np.float32) for pair in example.train_pairs]
    train_ins = [np.asarray(pair.input, dtype=np.float32) for pair in example.train_pairs]
    cand_palette = len(np.unique(cand)) if cand.size else 0
    cand_dom = _dominant(cand)
    train_palette = _mean([len(np.unique(out)) for out in train_outs])
    train_dom = _mean([_dominant(out) for out in train_outs])
    same_shape = float(test.shape == cand.shape)
    test_diff = float(np.mean(test != cand)) if test.shape == cand.shape and cand.size else 1.0
    train_diff = _mean([float(np.mean(inp != out)) for inp, out in zip(train_ins, train_outs) if inp.shape == out.shape and out.size])
    nearest_train = min([_resized_hamming(cand, out) for out in train_outs] or [1.0])
    h, w = cand.shape if cand.ndim == 2 else (0, 0)
    return [
        h / 30.0,
        w / 30.0,
        cand_palette / 10.0,
        cand_dom / 10.0,
        train_palette / 10.0,
        train_dom / 10.0,
        same_shape,
        test_diff,
        train_diff,
        abs(test_diff - train_diff),
        nearest_train,
        float(np.mean(cand == cand_dom)) if cand.size else 0.0,
        _entropy(cand),
        _object_count(cand) / 20.0,
        _mean([_object_count(out) for out in train_outs]) / 20.0,
        abs((_object_count(cand) / 20.0) - (_mean([_object_count(out) for out in train_outs]) / 20.0)),
    ]


def _feature_dim() -> int:
    return 16


def _cheap_controls(
    variant: SearchVariant,
    examples: Sequence[ArcVerificationExample],
    predict_fn,
    clean_logits: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> Dict[str, object]:
    chance = 1.0 / max(1, variant.candidate_count)
    controls: Dict[str, object] = {
        "benchmark": BENCHMARK,
        "variant": variant.name,
        "seed": seed,
        "chance": chance,
        "candidate_source_metadata_only_accuracy": candidate_metadata_only_accuracy(examples),
    }
    for name in ("candidate_only", "train_pair_shuffle", "candidate_evidence_mismatch", "candidate_order_shuffle_with_gold_remap"):
        controlled = apply_arc_control(examples, name, seed + 17_000 + len(controls))
        logits = predict_fn(controlled, "none", seed + 23_000)
        control_labels = np.asarray([example.label for example in controlled], dtype=np.int64)
        controls[name] = _metric_block(logits, control_labels, controlled)
    if variant.model_kind == "rule_clone":
        role_logits = predict_fn(examples, "physical_role_order_shuffle", seed + 31_000)
    else:
        role_logits = clean_logits
    controls["candidate_order_invariance_delta"] = abs(float(_top1(clean_logits, labels)) - float(controls["candidate_order_shuffle_with_gold_remap"]["top1"]))
    controls["role_order_invariance_delta"] = abs(float(_top1(clean_logits, labels)) - float(_top1(role_logits, labels)))
    controls["control_pass"] = {
        "candidate_only_near_chance": float(controls["candidate_only"]["top1"]) <= chance + 0.10,
        "candidate_metadata_near_chance": float(controls["candidate_source_metadata_only_accuracy"]) <= chance + 0.10,
        "train_pair_shuffle_collapses": float(controls["train_pair_shuffle"]["top1"]) <= chance + 0.15,
        "candidate_evidence_mismatch_collapses": float(controls["candidate_evidence_mismatch"]["top1"]) <= chance + 0.15,
        "candidate_order_invariance": float(controls["candidate_order_invariance_delta"]) <= 0.08,
        "role_order_invariance": float(controls["role_order_invariance_delta"]) <= 0.08,
    }
    controls["control_pass"]["overall"] = all(controls["control_pass"].values())
    return controls


def _model_config_for_variant(variant: SearchVariant, candidate_count: int | None = None) -> ArcModelConfig:
    del candidate_count
    views = tuple(name for name in variant.view_set for _ in range(max(1, int(variant.repeated_avenues))))
    return ArcModelConfig(
        variant_family=variant.family,
        view_names=views,
        learned_slots=bool(variant.learned_slots),
        representation_mode=str(variant.representation),
        interaction_style=str(variant.interaction),
        model_dim=32,
        num_heads=2,
        rule_layers=1,
        candidate_layers=1,
        coordination_blocks=int(variant.coordination_blocks),
        ff_dim=64,
        max_rule_tokens=96,
        max_candidate_tokens=80,
    )


def _variant_plan() -> List[SearchVariant]:
    return [
        SearchVariant("stats_mlp_hybrid_features", "architecture_simplification", "handcrafted_stats", ("none",), "none", "cross_entropy", model_kind="feature_mlp", diagnostic_only=True),
        SearchVariant("clone_raw_cell_ce", "representation", "cell", ("raw",), "late_candidate_query", "cross_entropy"),
        SearchVariant("clone_raw_diff_ce", "representation", "raw_diff", ("raw", "diff"), "late_candidate_query", "cross_entropy"),
        SearchVariant("clone_object_bbox_ce", "representation", "object", ("raw", "object"), "late_candidate_query", "cross_entropy"),
        SearchVariant("clone_color_hist_ce", "representation", "color_histogram", ("raw", "color"), "late_candidate_query", "cross_entropy"),
        SearchVariant("clone_hybrid_all_views_ce", "view_set", "hybrid_all", ("raw", "diff", "object", "color", "geometry"), "late_candidate_query", "cross_entropy"),
        SearchVariant("clone_learned_slots_hybrid_ce", "learned_slots", "hybrid_cell_object_diff", ("slot_0", "slot_1", "slot_2", "slot_3"), "late_candidate_query", "cross_entropy", learned_slots=True),
        SearchVariant("clone_repeated_avenues_hybrid_ce", "multi_avenue", "hybrid_cell_object_diff", ("raw", "diff", "object"), "late_candidate_query", "cross_entropy", repeated_avenues=2),
        SearchVariant("clone_candidate_guided_hybrid_ce", "candidate_interaction", "hybrid_cell_object_diff", ("raw", "diff", "object", "color"), "candidate_guided_rule_induction", "cross_entropy"),
        SearchVariant("clone_bidirectional_hybrid_ce", "candidate_interaction", "hybrid_cell_object_diff", ("raw", "diff", "object", "color"), "bidirectional_candidate_rule", "cross_entropy"),
        SearchVariant("clone_stack2_hybrid_ce", "stacking", "hybrid_cell_object_diff", ("raw", "diff", "object", "color"), "late_candidate_query", "cross_entropy", coordination_blocks=2),
        SearchVariant("clone_stack3_hybrid_ce", "stacking", "hybrid_cell_object_diff", ("raw", "diff", "object", "color"), "late_candidate_query", "cross_entropy", coordination_blocks=3),
        SearchVariant("clone_hybrid_pairwise", "objective", "hybrid_cell_object_diff", ("raw", "diff", "object", "color"), "late_candidate_query", "pairwise_logistic"),
        SearchVariant("clone_hybrid_margin", "objective", "hybrid_cell_object_diff", ("raw", "diff", "object", "color"), "late_candidate_query", "margin_ranking"),
        SearchVariant("clone_palette_curriculum_ce", "negative_curriculum", "hybrid_cell_object_diff", ("raw", "diff", "color"), "late_candidate_query", "cross_entropy", negative_difficulty="palette_matched"),
        SearchVariant("clone_easy_random_ce", "negative_curriculum", "hybrid_cell_object_diff", ("raw", "diff"), "late_candidate_query", "cross_entropy", negative_difficulty="easy_random"),
        SearchVariant("clone_changed_cell_ce", "negative_curriculum", "hybrid_cell_object_diff", ("raw", "diff"), "late_candidate_query", "cross_entropy", negative_difficulty="changed_cell_count_matched"),
        SearchVariant("clone_adversarial_hard_ce", "negative_curriculum", "hybrid_cell_object_diff", ("raw", "diff", "object", "color"), "late_candidate_query", "cross_entropy", negative_difficulty="adversarial_hard"),
    ]


def _leaderboard(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    control_by_key = {(row["variant"], row["seed"]): row for row in controls}
    leaders = []
    for row in rows:
        if row.get("status") != "completed":
            leaders.append(
                {
                    "variant": row["variant"],
                    "seed": row["seed"],
                    "status": row["status"],
                    "trainable_top1": 0.0,
                    "frozen_top1": 0.0,
                    "delta": 0.0,
                    "candidate_only": 0.0,
                    "controls_pass": False,
                    "score": -999.0,
                }
            )
            continue
        control = control_by_key.get((row["variant"], row["seed"]), {})
        trainable = float(row["dev_trainable"]["top1"])
        frozen = float(row["dev_frozen"]["top1"])
        candidate_only = float(control.get("candidate_only", {}).get("top1", 0.0))
        controls_pass = bool(control.get("control_pass", {}).get("overall", False))
        score = trainable + (trainable - frozen) - 0.5 * candidate_only + (0.2 if controls_pass else -1.0)
        leaders.append(
            {
                "variant": row["variant"],
                "seed": row["seed"],
                "status": row["status"],
                "family": row.get("family"),
                "representation": row.get("representation"),
                "view_set": " ".join(row.get("view_set", [])),
                "interaction": row.get("interaction"),
                "objective": row.get("objective"),
                "negative_difficulty": row.get("negative_difficulty"),
                "candidate_count": int(row.get("candidate_count", 0)),
                "trainable_top1": trainable,
                "frozen_top1": frozen,
                "delta": trainable - frozen,
                "candidate_only": candidate_only,
                "heuristic_best": _best_non_oracle_baseline(row.get("baselines", {})),
                "controls_pass": controls_pass,
                "training_time_seconds": float(row.get("training_time_seconds", 0.0)),
                "param_count": int(row.get("param_count", 0)),
                "score": score,
            }
        )
    leaders.sort(key=lambda item: float(item["score"]), reverse=True)
    return leaders


def _failure_taxonomy(rows: Sequence[Dict[str, object]], overfit_rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]]) -> Dict[str, object]:
    failures: Dict[str, int] = {}
    for row in rows:
        if row.get("status") == "failed_overfit_gate":
            key = f"failed_overfit::{row.get('failure_reason')}"
            failures[key] = failures.get(key, 0) + 1
        elif row.get("status") == "completed":
            if float(row.get("dev_trainable", {}).get("top1", 0.0)) <= float(row.get("baselines", {}).get("random", 0.0)):
                failures["heldout_not_above_random"] = failures.get("heldout_not_above_random", 0) + 1
            if float(row.get("dev_delta_trainable_minus_frozen", 0.0)) <= 0.0:
                failures["did_not_beat_frozen"] = failures.get("did_not_beat_frozen", 0) + 1
    for row in controls:
        if not bool(row.get("control_pass", {}).get("overall", False)):
            failures["control_failure"] = failures.get("control_failure", 0) + 1
    return {
        "counts": dict(sorted(failures.items())),
        "failed_overfit_gates": [row for row in overfit_rows if not row.get("pass")],
        "interpretation": _failure_interpretation(failures),
    }


def _best_non_oracle_baseline(baselines: Dict[str, object]) -> float:
    values = [
        float(value)
        for key, value in baselines.items()
        if key != "oracle_candidate_present" and isinstance(value, (int, float))
    ]
    return max(values) if values else 0.0


def _summary(leaderboard: Sequence[Dict[str, object]]) -> Dict[str, object]:
    completed = [row for row in leaderboard if row.get("status") == "completed"]
    medium_ready = [
        row
        for row in completed
        if float(row.get("delta", 0.0)) > 0.10
        and float(row.get("trainable_top1", 0.0)) > float(row.get("candidate_only", 1.0))
        and float(row.get("trainable_top1", 0.0)) > float(row.get("heuristic_best", 1.0))
        and bool(row.get("controls_pass", False))
    ]
    return {
        "completed_variants": len(completed),
        "best_variant": completed[0]["variant"] if completed else None,
        "best_trainable_top1": float(completed[0]["trainable_top1"]) if completed else 0.0,
        "best_delta": float(completed[0]["delta"]) if completed else 0.0,
        "medium_validation_triggered": bool(medium_ready),
        "medium_ready_variants": [row["variant"] for row in medium_ready],
        "final_validation_launched": False,
    }


def _error_rows(
    variant: SearchVariant,
    seed: int,
    examples: Sequence[ArcVerificationExample],
    train_logits: np.ndarray,
    frozen_logits: np.ndarray,
    limit: int,
) -> List[Dict[str, object]]:
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    train_pred = np.argmax(train_logits, axis=1)
    frozen_pred = np.argmax(frozen_logits, axis=1)
    rows = []
    for index, example in enumerate(examples):
        category = None
        if train_pred[index] == labels[index] and frozen_pred[index] != labels[index]:
            category = "trainable_right_frozen_wrong"
        elif frozen_pred[index] == labels[index] and train_pred[index] != labels[index]:
            category = "frozen_right_trainable_wrong"
        elif train_pred[index] != labels[index] and frozen_pred[index] != labels[index]:
            category = "all_models_wrong"
        if category is None:
            continue
        rows.append(
            {
                "type": "arc1_1_error_case",
                "category": category,
                "variant": variant.name,
                "seed": seed,
                "task_id": example.task_id,
                "example_id": example.id,
                "gold_candidate_index": int(example.label),
                "predicted_candidate_index": int(train_pred[index]),
                "frozen_candidate_index": int(frozen_pred[index]),
                "candidate_probabilities": _softmax(train_logits[index]).tolist(),
                "negative_type_labels": list(example.negative_types),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _write_outputs(
    result: Dict[str, object],
    results_path: Path,
    controls_path: Path,
    leaderboard_path: Path,
    failure_path: Path,
    overfit_path: Path,
    error_path: Path,
    report_path: Path,
) -> None:
    for path in (results_path, controls_path, leaderboard_path, failure_path, overfit_path, error_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({key: value for key, value in result.items() if key not in {"error_cases"}}, indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(json.dumps({"controls": result["controls"]}, indent=2, sort_keys=True), encoding="utf-8")
    failure_path.write_text(json.dumps(result["failure_taxonomy"], indent=2, sort_keys=True), encoding="utf-8")
    overfit_path.write_text(json.dumps({"overfit_curves": result["overfit_curves"]}, indent=2, sort_keys=True), encoding="utf-8")
    error_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in result["error_cases"]) + ("\n" if result["error_cases"] else ""), encoding="utf-8")
    with leaderboard_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in result["leaderboard"] for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["leaderboard"])
    report_path.write_text(_render_report(result), encoding="utf-8")


def _render_report(result: Dict[str, object]) -> str:
    leaderboard = result.get("leaderboard", [])
    summary = result.get("summary", {})
    lines = [
        "# Stage ARC-1.1 Empirical Search",
        "",
        "## Scope",
        "",
        "- Bounded empirical search over representation, rule-clone views, candidate interaction, objectives, negative difficulty, and simple baselines.",
        "- No final validation was launched.",
        "- Labels, candidate order remapping, source metadata hiding, leakage audits, controls, and frozen-comparator protocol remain fixed.",
        "",
        "## Leaderboard",
        "",
        "| rank | variant | status | trainable | frozen | delta | candidate-only | controls | family |",
        "|---:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(leaderboard[:20], start=1):
        lines.append(
            f"| {rank} | {row.get('variant')} | {row.get('status')} | {float(row.get('trainable_top1', 0.0)):.4f} | {float(row.get('frozen_top1', 0.0)):.4f} | {float(row.get('delta', 0.0)):.4f} | {float(row.get('candidate_only', 0.0)):.4f} | `{bool(row.get('controls_pass', False))}` | {row.get('family', '')} |"
        )
    lines.extend(["", "## Failure Taxonomy", "", "```json", json.dumps(result.get("failure_taxonomy", {}).get("counts", {}), indent=2, sort_keys=True), "```", ""])
    lines.extend(_overfit_section(result))
    lines.extend(_control_section(result))
    lines.extend(_comparison_section(result, "representation", "Representation Comparison"))
    lines.extend(_comparison_section(result, "objective", "Objective Comparison"))
    lines.extend(_comparison_section(result, "negative_difficulty", "Negative Difficulty Comparison"))
    lines.extend(_compute_section(result))
    lines.extend(
        [
            "## Required Answers",
            "",
            f"1. Which representation made ARC candidate verification learnable? `micro_overfit={_best_overfit_by(result, 'representation')}`; `clean_heldout={_best_clean_heldout_by(result, 'representation')}`; `uncontrolled_best={_best_uncontrolled_by(result, 'representation')}`.",
            f"2. Which rule-clone view set worked best? `{_best_completed(result).get('view_set', '')}` among control-clean completed variants.",
            f"3. Did candidate-guided processing help? `{_family_delta(result, 'candidate_interaction')}`.",
            f"4. Did stacking help? `{_family_delta(result, 'stacking')}`.",
            f"5. Did curriculum negatives help? `{_family_delta(result, 'negative_curriculum')}`.",
            f"6. Did pairwise/ranking losses help? `{_family_delta(result, 'objective')}`.",
            f"7. Did any variant beat frozen and candidate-only baselines? `clean_valid={_any_clean_beats_frozen_and_candidate_only(leaderboard)}`; `uncontrolled={any(float(row.get('delta', 0.0)) > 0 and float(row.get('trainable_top1', 0.0)) > float(row.get('candidate_only', 1.0)) for row in leaderboard)}`.",
            f"8. Did any variant keep controls clean? `{any(bool(row.get('controls_pass', False)) for row in leaderboard)}`.",
            f"9. Is the ARC latent-rule-clone path worth Phase 2? `{bool(summary.get('medium_validation_triggered', False))}` by the configured medium trigger.",
            f"10. What is the smallest working architecture? `{_smallest_working(result)}`.",
            "",
            "## Decision",
            "",
            f"- Medium validation triggered: `{bool(summary.get('medium_validation_triggered', False))}`.",
            f"- Medium-ready variants: `{summary.get('medium_ready_variants', [])}`.",
            "- Final validation remains blocked until the Stage ARC-1.1 medium trigger passes.",
        ]
    )
    return "\n".join(lines) + "\n"


def _overfit_section(result: Dict[str, object]) -> List[str]:
    lines = ["## Overfit Results", "", "| variant | gate | train acc | threshold | pass |", "|---|---|---:|---:|---|"]
    for row in result.get("overfit_curves", []):
        lines.append(
            f"| {row.get('variant')} | {row.get('gate')} | {float(row.get('train_accuracy', 0.0)):.4f} | {float(row.get('threshold', 0.0)):.4f} | `{bool(row.get('pass', False))}` |"
        )
    lines.append("")
    return lines


def _control_section(result: Dict[str, object]) -> List[str]:
    lines = [
        "## Control Pass/Fail",
        "",
        "| variant | candidate-only | metadata-only | train shuffle | mismatch | order delta | role delta | overall |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result.get("controls", []):
        lines.append(
            f"| {row.get('variant')} | "
            f"{float(row.get('candidate_only', {}).get('top1', 0.0)):.4f} | "
            f"{float(row.get('candidate_source_metadata_only_accuracy', 0.0)):.4f} | "
            f"{float(row.get('train_pair_shuffle', {}).get('top1', 0.0)):.4f} | "
            f"{float(row.get('candidate_evidence_mismatch', {}).get('top1', 0.0)):.4f} | "
            f"{float(row.get('candidate_order_invariance_delta', 0.0)):.4f} | "
            f"{float(row.get('role_order_invariance_delta', 0.0)):.4f} | "
            f"`{bool(row.get('control_pass', {}).get('overall', False))}` |"
        )
    if not result.get("controls"):
        lines.append("| none | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |")
    lines.append("")
    return lines


def _comparison_section(result: Dict[str, object], key: str, title: str) -> List[str]:
    groups: Dict[str, List[float]] = {}
    for row in result.get("leaderboard", []):
        if row.get("status") != "completed":
            continue
        groups.setdefault(str(row.get(key, "unknown")), []).append(float(row.get("trainable_top1", 0.0)))
    lines = [f"## {title}", "", "| group | variants | mean top1 |", "|---|---:|---:|"]
    for group, values in sorted(groups.items(), key=lambda item: _mean(item[1]), reverse=True):
        lines.append(f"| {group} | {len(values)} | {_mean(values):.4f} |")
    lines.append("")
    return lines


def _compute_section(result: Dict[str, object]) -> List[str]:
    lines = ["## Compute Table", "", "| variant | params | train seconds | candidate count |", "|---|---:|---:|---:|"]
    for row in result.get("leaderboard", []):
        if row.get("status") != "completed":
            continue
        lines.append(
            f"| {row.get('variant')} | {int(row.get('param_count', 0))} | {float(row.get('training_time_seconds', 0.0)):.2f} | {int(row.get('candidate_count', 0)) if row.get('candidate_count') is not None else 0} |"
        )
    lines.append("")
    return lines


def _best_by(result: Dict[str, object], key: str) -> str:
    best = _best_completed(result)
    return str(best.get(key, "none"))


def _best_clean_heldout_by(result: Dict[str, object], key: str) -> str:
    rows = [
        row
        for row in result.get("leaderboard", [])
        if row.get("status") == "completed"
        and bool(row.get("controls_pass", False))
        and float(row.get("delta", 0.0)) > 0.0
        and float(row.get("trainable_top1", 0.0)) > float(row.get("candidate_only", 1.0))
        and float(row.get("trainable_top1", 0.0)) > float(row.get("heuristic_best", 1.0))
    ]
    if not rows:
        return "none"
    best = max(rows, key=lambda row: float(row.get("trainable_top1", 0.0)))
    return str(best.get(key, "none"))


def _best_overfit_by(result: Dict[str, object], key: str) -> str:
    by_variant = {row.get("variant"): row for row in result.get("leaderboard", [])}
    complete_gate_rows = []
    for row in result.get("overfit_curves", []):
        if row.get("gate") == "35_examples_N8" and bool(row.get("pass", False)):
            complete_gate_rows.append(row)
    if not complete_gate_rows:
        return "none"
    best_gate = max(complete_gate_rows, key=lambda row: float(row.get("train_accuracy", 0.0)))
    variant_row = by_variant.get(best_gate.get("variant"), {})
    return str(variant_row.get(key, "none"))


def _best_completed(result: Dict[str, object]) -> Dict[str, object]:
    for row in result.get("leaderboard", []):
        if row.get("status") == "completed" and bool(row.get("controls_pass", False)):
            return row
    for row in result.get("leaderboard", []):
        if row.get("status") == "completed":
            return row
    return {}


def _best_uncontrolled_by(result: Dict[str, object], key: str) -> str:
    rows = [row for row in result.get("leaderboard", []) if row.get("status") == "completed"]
    if not rows:
        return "none"
    best = max(rows, key=lambda row: float(row.get("trainable_top1", 0.0)))
    return str(best.get(key, "none"))


def _family_delta(result: Dict[str, object], family: str) -> str:
    rows = [row for row in result.get("leaderboard", []) if row.get("family") == family and row.get("status") == "completed"]
    if not rows:
        return "not established"
    best = max(rows, key=lambda row: float(row.get("trainable_top1", 0.0)))
    return f"best {best.get('variant')} top1={float(best.get('trainable_top1', 0.0)):.4f} delta={float(best.get('delta', 0.0)):.4f}"


def _smallest_working(result: Dict[str, object]) -> str:
    rows = [
        row
        for row in result.get("leaderboard", [])
        if row.get("status") == "completed"
        and float(row.get("delta", 0.0)) > 0.0
        and float(row.get("trainable_top1", 0.0)) > float(row.get("candidate_only", 0.0))
        and float(row.get("trainable_top1", 0.0)) > float(row.get("heuristic_best", 0.0))
        and bool(row.get("controls_pass", False))
    ]
    if not rows:
        return "none"
    rows.sort(key=lambda row: int(row.get("param_count", 10**12)))
    return str(rows[0].get("variant"))


def _any_clean_beats_frozen_and_candidate_only(leaderboard: Sequence[Dict[str, object]]) -> bool:
    return any(
        row.get("status") == "completed"
        and bool(row.get("controls_pass", False))
        and float(row.get("delta", 0.0)) > 0.0
        and float(row.get("trainable_top1", 0.0)) > float(row.get("candidate_only", 1.0))
        for row in leaderboard
    )


def _first_failed_gate(rows: Sequence[Dict[str, object]]) -> str:
    for row in rows:
        if not bool(row.get("pass", False)):
            return str(row.get("gate", "unknown"))
    return "none"


def _failure_interpretation(failures: Dict[str, int]) -> str:
    if any(key.startswith("failed_overfit") for key in failures):
        return "representation_or_objective_does_not_fit_small_candidate_sets"
    if failures.get("did_not_beat_frozen"):
        return "trainable_shared_encoder_not_improving_over_exact_frozen_comparator"
    if failures.get("control_failure"):
        return "shortcut_or_invariance_control_failed"
    return "no_dominant_failure"


def _dataset_config(data: object) -> ArcVerificationDatasetConfig:
    values = dict(data or {})
    allowed = set(ArcVerificationDatasetConfig.__dataclass_fields__.keys())
    return ArcVerificationDatasetConfig(**{key: value for key, value in values.items() if key in allowed})


def _training_config(data: object) -> ArcTrainingConfig:
    values = dict(data or {})
    allowed = set(ArcTrainingConfig.__dataclass_fields__.keys())
    return ArcTrainingConfig(**{key: value for key, value in values.items() if key in allowed})


def _feature_training_config(data: object) -> FeatureTrainingConfig:
    values = dict(data or {})
    allowed = set(FeatureTrainingConfig.__dataclass_fields__.keys())
    return FeatureTrainingConfig(**{key: value for key, value in values.items() if key in allowed})


def _load_config(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _chunks(values: Sequence[int], size: int) -> Iterable[List[int]]:
    for index in range(0, len(values), max(1, size)):
        yield list(values[index : index + max(1, size)])


def _flat_feature_params(model: nn.Module) -> torch.Tensor:
    params = [parameter.detach().float().reshape(-1) for parameter in model.parameters()]
    return torch.cat(params) if params else torch.zeros(0)


def _l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b).item()) if a.numel() == b.numel() and a.numel() else 0.0


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _dominant(grid: np.ndarray) -> int:
    if grid.size == 0:
        return 0
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _entropy(grid: np.ndarray) -> float:
    if grid.size == 0:
        return 0.0
    _values, counts = np.unique(grid, return_counts=True)
    probs = counts.astype(np.float64) / float(grid.size)
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))


def _resized_hamming(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 1.0
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return float(np.mean(a[:h, :w] != b[:h, :w])) + 0.05 * (abs(a.shape[0] - b.shape[0]) + abs(a.shape[1] - b.shape[1]))


def _object_count(grid: np.ndarray) -> int:
    if grid.ndim != 2 or grid.size == 0:
        return 0
    bg = _dominant(grid)
    seen = np.zeros(grid.shape, dtype=bool)
    count = 0
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if seen[row, col] or int(grid[row, col]) == bg:
                continue
            count += 1
            color = int(grid[row, col])
            stack = [(row, col)]
            seen[row, col] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64) - float(np.max(values))
    exp = np.exp(values)
    return exp / max(1e-12, float(exp.sum()))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
