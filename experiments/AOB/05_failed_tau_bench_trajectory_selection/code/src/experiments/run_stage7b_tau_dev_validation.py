from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from src.coordinators.latent_coordination import LatentCoordinatorConfig
from src.coordinators.mlp import MLPTrainingConfig
from src.datasets.taubench_trajectory_selection_dataset import (
    STAGE7_NUM_CANDIDATES,
    TauBenchTrajectorySelectionExample,
    apply_stage7_control,
    build_taubench_selection_examples,
    duplicate_trajectory_hash_audit,
    load_taubench_candidates_jsonl,
    near_duplicate_trajectory_audit,
    randomized_label_matrix,
    stage7_output_leakage_audit,
    stage7_split_leakage_audit,
    taubench_label_matrix,
    taubench_to_multiview_many,
    trajectory_candidate_text,
    trajectory_static_features,
    validate_taubench_selection_examples,
)
from src.experiments.real_shared_weight_latent_coordination import (
    MessageChannelConfig,
    RealSharedWeightTrainingConfig,
    SharedTransformerAgentConfig,
    _text_tokens,
)
from src.experiments.run_stage7_taubench_trajectory_selector import (
    Stage7CandidatewiseReranker,
    _audit_subset,
    _cost_metadata,
    _embedding_reranker_scores,
    _first_candidate_scores,
    _hashed_candidate_text_features,
    _llm_judge_scores,
    _paired_tests_for_split,
    _random_scores,
    _rank_candidate_scores,
    evaluate_scores,
    fit_stage7_latent_selector,
    predict_stage7_latent_logits,
)


BENCHMARK = "stage7b_tau_dev_validation"
DEFAULT_POOL = Path("results/stage7b0_tau_candidate_pool_balanced.jsonl")
DEFAULT_LABEL_AUDIT = Path("results/stage7b0_tau_label_audit.json")
DEFAULT_SPLITS = Path("results/stage7b_tau_dev_splits.json")
DEFAULT_RESULTS = Path("results/stage7b_tau_dev_selector_results.json")
DEFAULT_AUDIT = Path("results/stage7b_tau_dev_selector_audit.jsonl")
DEFAULT_VARIANT_COMPARISON = Path("results/stage7b_tau_dev_variant_comparison.json")
DEFAULT_REPORT = Path("reports/STAGE7B_TAU_DEV_VALIDATION.md")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/stage7b")
STAGE7B_SEEDS = (0, 1, 2)
SPLIT_SEED = 707_300


@dataclass(frozen=True)
class Stage7BVariant:
    name: str
    description: str
    num_avenues: int
    lr: float
    gradient_clip_norm: float
    avenue_dropout: float = 0.0
    avenue_prompt_mode: str = "stage7_taubench_trajectory_selection"
    coordinator_family: str = "candidate_token_cross_attention"


@dataclass(frozen=True)
class Stage7BConfig:
    domain: str = "retail"
    device: str = "cpu"
    hidden_dim: int = 32
    tiny_layers: int = 1
    tiny_heads: int = 1
    tiny_ff_dim: int = 64
    max_length: int = 192
    epochs: int = 3
    patience: int = 2
    batch_size: int = 8
    weight_decay: float = 0.0001
    mixed_precision: str = "none"
    bootstrap_samples: int = 1000


class BestGeneratorOnTrainAuditBaseline:
    def __init__(self, candidate_source_map: Dict[str, Dict[str, object]]) -> None:
        self.method = "best_generator_on_train_baseline"
        self.candidate_source_map = candidate_source_map
        self.generator_scores: Dict[str, float] = {}
        self.best_generator = ""

    def fit(self, train_examples: Sequence[TauBenchTrajectorySelectionExample]) -> None:
        grouped: Dict[str, List[int]] = defaultdict(list)
        for example in train_examples:
            for candidate, label in zip(example.candidates, example.labels_pass_fail):
                grouped[self._generator(candidate.candidate_id)].append(int(label))
        self.generator_scores = {
            generator: float(np.mean(labels)) if labels else 0.0
            for generator, labels in sorted(grouped.items())
        }
        self.best_generator = max(sorted(self.generator_scores), key=lambda key: self.generator_scores[key]) if self.generator_scores else ""

    def predict_scores(self, examples: Sequence[TauBenchTrajectorySelectionExample]) -> np.ndarray:
        rows = []
        for example in examples:
            rows.append([self.generator_scores.get(self._generator(candidate.candidate_id), 0.0) for candidate in example.candidates])
        return np.asarray(rows, dtype=np.float32)

    def _generator(self, candidate_id: str) -> str:
        row = self.candidate_source_map.get(str(candidate_id), {})
        return str(row.get("generator_name_audit_only", "unknown_audit_generator"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 7B dev-only tau-bench latent trajectory selector validation.")
    parser.add_argument("--candidate-pool", default=str(DEFAULT_POOL))
    parser.add_argument("--label-audit", default=str(DEFAULT_LABEL_AUDIT))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--variant-comparison", default=str(DEFAULT_VARIANT_COMPARISON))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    args = parser.parse_args()

    config = Stage7BConfig(
        device=str(args.device),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
    )
    result = run_stage7b_dev_validation(
        candidate_pool_path=Path(args.candidate_pool),
        label_audit_path=Path(args.label_audit),
        splits_path=Path(args.splits),
        results_path=Path(args.results),
        audit_path=Path(args.audit),
        variant_comparison_path=Path(args.variant_comparison),
        report_path=Path(args.report),
        checkpoint_dir=Path(args.checkpoint_dir),
        config=config,
    )
    print(json.dumps({"decision": result["decision"], "selected_config": result["selected_config"]["name"]}, sort_keys=True))


def run_stage7b_dev_validation(
    candidate_pool_path: Path,
    label_audit_path: Path,
    splits_path: Path,
    results_path: Path,
    audit_path: Path,
    variant_comparison_path: Path,
    report_path: Path,
    checkpoint_dir: Path,
    config: Stage7BConfig,
) -> Dict[str, object]:
    candidates = load_taubench_candidates_jsonl(candidate_pool_path)
    if not candidates:
        raise FileNotFoundError(f"Stage 7B candidate pool not found or empty: {candidate_pool_path}")
    examples = build_taubench_selection_examples(candidates)
    validation = validate_taubench_selection_examples(examples, require_labels=True)
    if not bool(validation.get("passes", False)):
        raise RuntimeError(json.dumps(validation, indent=2, sort_keys=True))
    pool_name = _pool_name_from_path(candidate_pool_path)
    candidate_source_map = load_candidate_source_map(label_audit_path, pool_name)
    splits = make_stage7b_split(examples, candidate_source_map=candidate_source_map, seed=SPLIT_SEED, train_fraction=0.75)
    write_splits(splits_path, splits, candidate_pool_path, pool_name, candidate_source_map, config)

    audit_rows: List[Dict[str, object]] = []
    progress_path = results_path.with_name(f"{results_path.stem}_rows_progress.jsonl")
    completed_rows = load_progress_rows(progress_path, config, candidate_pool_path, pool_name)
    completed_by_key = {(str(row.get("variant")), int(row.get("seed", -1))): row for row in completed_rows}
    all_examples = list(splits["train"]) + list(splits["dev"])
    split_leakage = stage7_split_leakage_audit("stage7_taubench_trajectory_selector", SPLIT_SEED, {"train": splits["train"], "dev": splits["dev"], "test": []})
    output_leakage = stage7_output_leakage_audit("stage7_taubench_trajectory_selector", SPLIT_SEED, all_examples)
    duplicate_audit = duplicate_trajectory_hash_audit(all_examples)
    near_duplicate_audit = near_duplicate_trajectory_audit(all_examples)
    audit_rows.extend(
        [
            {"benchmark": BENCHMARK, "type": "split_leakage", **split_leakage},
            {"benchmark": BENCHMARK, "type": "output_leakage", **output_leakage},
            {"benchmark": BENCHMARK, "type": "duplicate_trajectory_hash", **duplicate_audit},
            {"benchmark": BENCHMARK, "type": "near_duplicate_trajectory", **near_duplicate_audit},
        ]
    )

    rows = []
    variants = stage7b_variants()
    for variant in variants:
        for seed in STAGE7B_SEEDS:
            row = completed_by_key.get((variant.name, int(seed)))
            if row is None:
                row = run_variant_seed(
                    variant=variant,
                    seed=seed,
                    splits=splits,
                    candidate_source_map=candidate_source_map,
                    config=config,
                    checkpoint_dir=checkpoint_dir,
                    split_leakage=split_leakage,
                    output_leakage=output_leakage,
                    duplicate_audit=duplicate_audit,
                    near_duplicate_audit=near_duplicate_audit,
                )
                append_progress_row(progress_path, row, config, candidate_pool_path, pool_name)
            rows.append(row)
            audit_rows.extend(audit_rows_from_result_row(row))

    comparison = build_variant_comparison(rows)
    selected_name = str(comparison["selected_config"])
    selected_config = next(variant for variant in variants if variant.name == selected_name)
    decision_info = stage7b_decision(rows, comparison, selected_name, split_leakage, output_leakage, duplicate_audit, near_duplicate_audit)
    result = {
        "artifact": "stage7b_tau_dev_selector_results",
        "created_at_utc": _now(),
        "candidate_pool_path": str(candidate_pool_path),
        "pool_name": pool_name,
        "label_audit_path": str(label_audit_path),
        "splits_path": str(splits_path),
        "domain": config.domain,
        "k": STAGE7_NUM_CANDIDATES,
        "seeds": list(STAGE7B_SEEDS),
        "config": asdict(config),
        "variants": [asdict(variant) for variant in variants],
        "selected_config": asdict(selected_config),
        "selection_rule": comparison["selection_rule"],
        "dataset_validation": validation,
        "split_summary": split_summary(splits, candidate_source_map),
        "stage7b_only_no_final_claim": True,
        "stage7c_not_run": True,
        "rows": rows,
        "variant_comparison_path": str(variant_comparison_path),
        "report_path": str(report_path),
        **decision_info,
    }
    comparison_record = {
        "artifact": "stage7b_tau_dev_variant_comparison",
        "created_at_utc": _now(),
        "candidate_pool_path": str(candidate_pool_path),
        "pool_name": pool_name,
        "selection_rule": comparison["selection_rule"],
        "selected_config": selected_name,
        "variant_metrics": comparison["variant_metrics"],
        "decision": decision_info["decision"],
        "stage7b_only_no_final_claim": True,
    }
    write_outputs(result, comparison_record, audit_rows, results_path, audit_path, variant_comparison_path, report_path)
    return result


def stage7b_variants() -> Tuple[Stage7BVariant, ...]:
    return (
        Stage7BVariant(
            name="variant_a_4x4_lr3e4_clip1",
            description="Stage 5/6 unchanged: 4 roles x 4 avenues, candidate-token-direct, lr=3e-4, clip=1.0.",
            num_avenues=4,
            lr=3e-4,
            gradient_clip_norm=1.0,
        ),
        Stage7BVariant(
            name="variant_b_4x4_lr1e4_clip1",
            description="Same 4x4 candidate-token-direct architecture with lr=1e-4, clip=1.0.",
            num_avenues=4,
            lr=1e-4,
            gradient_clip_norm=1.0,
        ),
        Stage7BVariant(
            name="variant_c_4x4_dropout005_lr3e4_clip1",
            description="Same 4x4 candidate-token-direct architecture with avenue dropout=0.05, lr=3e-4, clip=1.0.",
            num_avenues=4,
            lr=3e-4,
            gradient_clip_norm=1.0,
            avenue_dropout=0.05,
        ),
        Stage7BVariant(
            name="variant_d_4x1_lr3e4_clip1",
            description="4 roles x 1 avenue candidate-token-direct baseline, lr=3e-4, clip=1.0.",
            num_avenues=1,
            lr=3e-4,
            gradient_clip_norm=1.0,
            avenue_prompt_mode="single",
        ),
    )


def run_variant_seed(
    variant: Stage7BVariant,
    seed: int,
    splits: Dict[str, List[TauBenchTrajectorySelectionExample]],
    candidate_source_map: Dict[str, Dict[str, object]],
    config: Stage7BConfig,
    checkpoint_dir: Path,
    split_leakage: Dict[str, object],
    output_leakage: Dict[str, object],
    duplicate_audit: Dict[str, object],
    near_duplicate_audit: Dict[str, object],
) -> Dict[str, object]:
    started = time.perf_counter()
    device = resolve_device(config.device)
    agent_config = agent_config_for(config)
    training = training_config_for(config, variant)
    message_config = message_config_for(variant)
    coordinator_config = coordinator_config_for(config, variant)
    models = fit_models(
        splits=splits,
        seed=seed,
        device=device,
        agent_config=agent_config,
        training=training,
        message_config=message_config,
        coordinator_config=coordinator_config,
        variant=variant,
        candidate_source_map=candidate_source_map,
    )
    metrics = {
        split: evaluate_methods(models, examples, split, seed, candidate_source_map)
        for split, examples in splits.items()
    }
    controls = evaluate_controls(models["trainable"], splits["dev"], seed)
    invariance = invariance_audit(models["trainable"], splits["dev"], seed)
    paired = _paired_tests_for_split(
        metrics["dev"],
        splits["dev"],
        seed,
        bootstrap_samples=int(config.bootstrap_samples),
    )
    checkpoint_paths = save_checkpoints(checkpoint_dir, variant, seed, models, config)
    row = {
        "stage": BENCHMARK,
        "phase": "7B_dev",
        "domain": config.domain,
        "seed": int(seed),
        "variant": variant.name,
        "variant_config": asdict(variant),
        "status": "completed",
        "completed_at_utc": _now(),
        "device": device,
        "metrics": metrics,
        "controls": controls,
        "invariance_audit": invariance,
        "paired_tests_vs_best_non_oracle_baseline": paired,
        "checkpoint_paths": checkpoint_paths,
        "elapsed_seconds": float(time.perf_counter() - started),
        "split_leakage_audit": split_leakage,
        "split_leakage_audit_passes": bool(split_leakage.get("passes", False)),
        "output_leakage_audit": output_leakage,
        "output_leakage_audit_passes": bool(output_leakage.get("passes", False)),
        "duplicate_trajectory_hash_audit": duplicate_audit,
        "near_duplicate_trajectory_audit": near_duplicate_audit,
        "trainable_audit": _audit_subset(models["trainable"].audit),
        "frozen_audit": _audit_subset(models["frozen"].audit),
        "raw_latent_audit": _audit_subset(models["raw_latent"].audit),
        "randomized_labels_audit": _audit_subset(models["randomized_labels"].audit),
        "generator_identity_only_diagnostic": generator_identity_only_diagnostic(models["best_generator_on_train_baseline"], splits["dev"]),
    }
    row["success_gates"] = seed_success_gates(row)
    return row


def fit_models(
    splits: Dict[str, List[TauBenchTrajectorySelectionExample]],
    seed: int,
    device: str,
    agent_config: SharedTransformerAgentConfig,
    training: RealSharedWeightTrainingConfig,
    message_config: MessageChannelConfig,
    coordinator_config: LatentCoordinatorConfig,
    variant: Stage7BVariant,
    candidate_source_map: Dict[str, Dict[str, object]],
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
    models: Dict[str, object] = {}
    models["raw_latent"] = fit_stage7_latent_selector(
        train,
        dev,
        agent_config=agent_config,
        coordinator_config=raw_coord,
        training_config=training,
        seed=seed + 71_000,
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
        seed=seed + 72_000,
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
        seed=seed + 73_000,
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
        seed=seed + 74_000,
        device=device,
        trainable_agent=True,
        method="randomized_labels_trainable_latent_selector",
        message_config=message_config,
        train_label_masks=randomized_label_matrix(train, seed + 74_001),
        dev_label_masks=randomized_label_matrix(dev, seed + 74_002),
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
        static_trajectory_features_with_entity_consistency,
        baseline_training,
        seed + 75_000,
        device,
        hidden_dims=(32,),
    )
    static.fit(train, dev)
    single = Stage7CandidatewiseReranker(
        "text_only_single_reviewer",
        lambda rows: _hashed_candidate_text_features(rows, feature_dim=256, mode="single_full_context"),
        baseline_training,
        seed + 76_000,
        device,
        hidden_dims=(64,),
    )
    single.fit(train, dev)
    multi = Stage7CandidatewiseReranker(
        "text_only_multi_agent_reviewer",
        lambda rows: _hashed_candidate_text_features(rows, feature_dim=256, mode="multi_role_text"),
        baseline_training,
        seed + 77_000,
        device,
        hidden_dims=(64,),
    )
    multi.fit(train, dev)
    best_generator = BestGeneratorOnTrainAuditBaseline(candidate_source_map)
    best_generator.fit(train)
    models.update(
        {
            "variant": variant,
            "static_trajectory_feature_reranker": static,
            "text_only_single_reviewer": single,
            "text_only_multi_agent_reviewer": multi,
            "best_generator_on_train_baseline": best_generator,
        }
    )
    return models


def evaluate_methods(
    models: Dict[str, object],
    examples: Sequence[TauBenchTrajectorySelectionExample],
    split: str,
    seed: int,
    candidate_source_map: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    llm_scores, llm_unavailable, llm_meta = _llm_judge_scores(examples)
    score_rows: Dict[str, Tuple[np.ndarray, bool, Dict[str, object]]] = {
        "oracle_pass_at_8": (taubench_label_matrix(examples), False, {"oracle": True}),
        "random_trajectory": (_random_scores(examples, seed), False, _cost_metadata(examples, "random_trajectory", 0.0)),
        "first_trajectory_order_baseline": (_first_candidate_scores(examples), False, _cost_metadata(examples, "first_trajectory", 0.0)),
        "embedding_reranker": (_embedding_reranker_scores(examples), False, _cost_metadata(examples, "embedding_reranker", 0.0)),
        "llm_judge_baseline": (llm_scores, llm_unavailable, llm_meta),
    }
    timed = [
        ("best_generator_on_train_baseline", lambda: models["best_generator_on_train_baseline"].predict_scores(examples)),
        ("static_trajectory_feature_reranker", lambda: models["static_trajectory_feature_reranker"].predict_scores(examples)),
        ("text_only_single_reviewer", lambda: models["text_only_single_reviewer"].predict_scores(examples)),
        ("text_only_multi_agent_reviewer", lambda: models["text_only_multi_agent_reviewer"].predict_scores(examples)),
        ("raw_latent_selector", lambda: predict_stage7_latent_logits(models["raw_latent"], examples, "none", seed)),
        ("frozen_same_architecture_latent_selector", lambda: predict_stage7_latent_logits(models["frozen"], examples, "none", seed)),
        ("trainable_shared_weight_latent_selector", lambda: predict_stage7_latent_logits(models["trainable"], examples, "none", seed)),
        ("randomized_labels_trainable_latent_selector", lambda: predict_stage7_latent_logits(models["randomized_labels"], examples, "none", seed)),
    ]
    for name, fn in timed:
        started = time.perf_counter()
        scores = fn()
        score_rows[name] = (scores, False, _cost_metadata(examples, name, time.perf_counter() - started))
    metrics = {
        name: evaluate_scores(name, scores, examples, split=split, unavailable=unavailable, metadata=metadata)
        for name, (scores, unavailable, metadata) in score_rows.items()
    }
    for name, (scores, unavailable, _metadata) in score_rows.items():
        if unavailable:
            continue
        metrics[name]["per_success_count_breakdown"] = per_success_count_breakdown(scores, examples)
        metrics[name]["per_generator_source_audit_breakdown"] = per_generator_source_breakdown(scores, examples, candidate_source_map)
    return metrics


def evaluate_controls(trainable, examples: Sequence[TauBenchTrajectorySelectionExample], seed: int) -> Dict[str, object]:
    controls = (
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
    out = {}
    for control in controls:
        if control == "hidden_states_shuffled_across_examples":
            controlled = list(examples)
            condition = "hidden_states_shuffled_across_examples"
        elif control == "physical_order_shuffled_roles_avenues_preserved":
            controlled = list(examples)
            condition = "physical_order_shuffled_roles_preserved"
        elif control == "candidate_order_shuffled_with_label_remap":
            controlled = apply_stage7_control(examples, control, seed + 78_000)
            condition = "none"
        else:
            controlled = apply_stage7_control(examples, control, seed + 78_000)
            condition = "none"
        scores = predict_stage7_latent_logits(trainable, controlled, condition, seed + 78_001)
        out[control] = evaluate_scores(control, scores, controlled, split="control")
    return out


def invariance_audit(trainable, examples: Sequence[TauBenchTrajectorySelectionExample], seed: int) -> Dict[str, object]:
    base_scores = predict_stage7_latent_logits(trainable, examples, "none", seed + 79_000)
    base_pred = _rank_candidate_scores(base_scores, examples)[:, 0]
    physical_scores = predict_stage7_latent_logits(trainable, examples, "physical_order_shuffled_roles_preserved", seed + 79_001)
    physical_pred = _rank_candidate_scores(physical_scores, examples)[:, 0]
    shuffled = apply_stage7_control(examples, "candidate_order_shuffled_with_label_remap", seed + 79_002)
    shuffled_scores = predict_stage7_latent_logits(trainable, shuffled, "none", seed + 79_003)
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
        "physical_order_invariance_passes": bool(physical_pass),
        "candidate_order_invariance_passes": bool(candidate_pass),
        "passes": bool(physical_pass and candidate_pass),
    }


def static_trajectory_features_with_entity_consistency(examples: Sequence[TauBenchTrajectorySelectionExample]) -> np.ndarray:
    rows = []
    for example in examples:
        goal_text = "\n".join(candidate.selector_visible_trajectory.user_messages[0] for candidate in example.candidates[:1])
        goal_entities = set(entity_tokens(goal_text))
        per_candidate = []
        for candidate in example.candidates:
            features = trajectory_static_features(candidate, user_goal=goal_text)
            candidate_text = trajectory_candidate_text(candidate)
            candidate_entities = set(entity_tokens(candidate_text))
            entity_consistency = len(goal_entities & candidate_entities) / float(max(1, len(goal_entities))) if goal_entities else 0.0
            values = [
                np.log1p(max(0.0, float(features.get("num_tool_calls", 0.0)))),
                float(features.get("invalid_tool_calls", 0.0)),
                float(features.get("repeated_tool_calls", 0.0)),
                float(features.get("tool_error_count", 0.0)),
                np.log1p(max(0.0, float(features.get("final_response_length", 0.0)))),
                np.log1p(max(0.0, float(features.get("conversation_length", 0.0)))),
                float(features.get("final_response_claims_completion", 0.0)),
                float(features.get("policy_mentions", 0.0)),
                float(features.get("user_goal_entity_overlap", 0.0)),
                float(entity_consistency),
            ]
            per_candidate.append(values)
        rows.append(per_candidate)
    return np.asarray(rows, dtype=np.float32)


def entity_tokens(text: str) -> List[str]:
    tokens = []
    for token in _text_tokens(text):
        if any(ch.isdigit() for ch in token) or token.startswith("#"):
            tokens.append(token.strip("#"))
    return tokens


def make_stage7b_split(
    examples: Sequence[TauBenchTrajectorySelectionExample],
    candidate_source_map: Dict[str, Dict[str, object]],
    seed: int,
    train_fraction: float,
) -> Dict[str, List[TauBenchTrajectorySelectionExample]]:
    strata: Dict[Tuple[int, str], List[TauBenchTrajectorySelectionExample]] = defaultdict(list)
    for example in examples:
        success_count = sum(example.labels_pass_fail)
        oracle_bin = "oracle_positive" if success_count > 0 else "oracle_empty"
        strata[(success_count, oracle_bin)].append(example)
    train: List[TauBenchTrajectorySelectionExample] = []
    dev: List[TauBenchTrajectorySelectionExample] = []
    for key, rows in sorted(strata.items()):
        ordered = sorted(rows, key=lambda example: stable_hash(f"{seed}:{example.domain}:{example.task_id}:{key}"))
        if len(ordered) == 1:
            target_train = 1
        else:
            target_train = int(round(len(ordered) * train_fraction))
            target_train = min(len(ordered) - 1, max(1, target_train))
        train.extend(ordered[:target_train])
        dev.extend(ordered[target_train:])
    train = sorted(train, key=lambda example: task_sort_key(example.task_id))
    dev = sorted(dev, key=lambda example: task_sort_key(example.task_id))
    if not train or not dev:
        raise RuntimeError("Stage 7B split construction produced an empty train or dev split")
    train_ids = {example.task_id for example in train}
    dev_ids = {example.task_id for example in dev}
    if train_ids & dev_ids:
        raise RuntimeError("Stage 7B split has overlapping task ids")
    return {"train": train, "dev": dev}


def load_candidate_source_map(label_audit_path: Path, pool_name: str) -> Dict[str, Dict[str, object]]:
    if not label_audit_path.exists():
        return {}
    audit = json.loads(label_audit_path.read_text(encoding="utf-8"))
    pool_rows = (audit.get("candidate_audit_map") or {}).get(pool_name, [])
    out = {}
    for row in pool_rows:
        if isinstance(row, dict):
            out[str(row.get("candidate_id", ""))] = dict(row)
    return out


def write_splits(
    path: Path,
    splits: Dict[str, List[TauBenchTrajectorySelectionExample]],
    candidate_pool_path: Path,
    pool_name: str,
    candidate_source_map: Dict[str, Dict[str, object]],
    config: Stage7BConfig,
) -> None:
    record = {
        "artifact": "stage7b_tau_dev_splits",
        "created_at_utc": _now(),
        "candidate_pool_path": str(candidate_pool_path),
        "pool_name": pool_name,
        "domain": config.domain,
        "seed": SPLIT_SEED,
        "split_strategy": "task_id disjoint stratified by per-task official success count and oracle-positive/oracle-empty status",
        "no_final_heldout_tasks_used": True,
        "splits": {name: [f"{example.domain}:{example.task_id}" for example in rows] for name, rows in splits.items()},
        "summary": split_summary(splits, candidate_source_map),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def split_summary(
    splits: Dict[str, List[TauBenchTrajectorySelectionExample]],
    candidate_source_map: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    out = {}
    for split, rows in splits.items():
        success_counts = [sum(example.labels_pass_fail) for example in rows]
        generators = Counter()
        for example in rows:
            for candidate in example.candidates:
                generators[str(candidate_source_map.get(candidate.candidate_id, {}).get("generator_name_audit_only", "unknown"))] += 1
        out[split] = {
            "tasks": len(rows),
            "candidates": sum(len(example.candidates) for example in rows),
            "oracle_pass_at_8": sum(1 for count in success_counts if count > 0) / float(max(1, len(success_counts))),
            "oracle_empty_tasks": sum(1 for count in success_counts if count == 0),
            "success_count_distribution": dict(sorted(Counter(success_counts).items())),
            "generator_source_distribution_audit_only": dict(sorted(generators.items())),
        }
    out["task_overlap_train_dev"] = len({example.task_id for example in splits.get("train", [])} & {example.task_id for example in splits.get("dev", [])})
    return out


def per_success_count_breakdown(scores: np.ndarray, examples: Sequence[TauBenchTrajectorySelectionExample]) -> Dict[str, Dict[str, float]]:
    labels = taubench_label_matrix(examples)
    ranking = _rank_candidate_scores(scores, examples)
    grouped: Dict[str, List[bool]] = defaultdict(list)
    for row, example in enumerate(examples):
        count = int(labels[row].sum())
        choice = int(ranking[row, 0])
        grouped[str(count)].append(bool(labels[row, choice] > 0))
    return {
        key: {"n": len(values), "pass_at_1": float(np.mean(values)) if values else 0.0}
        for key, values in sorted(grouped.items(), key=lambda item: int(item[0]))
    }


def per_generator_source_breakdown(
    scores: np.ndarray,
    examples: Sequence[TauBenchTrajectorySelectionExample],
    candidate_source_map: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    labels = taubench_label_matrix(examples)
    ranking = _rank_candidate_scores(scores, examples)
    grouped: Dict[str, List[bool]] = defaultdict(list)
    for row, example in enumerate(examples):
        choice = int(ranking[row, 0])
        candidate = example.candidates[choice]
        generator = str(candidate_source_map.get(candidate.candidate_id, {}).get("generator_name_audit_only", "unknown"))
        grouped[generator].append(bool(labels[row, choice] > 0))
    return {
        key: {"selected_count": len(values), "selected_success_rate": float(np.mean(values)) if values else 0.0}
        for key, values in sorted(grouped.items())
    }


def build_variant_comparison(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    variant_metrics = {}
    for variant, variant_rows in sorted(grouped.items()):
        dev_metrics = [row["metrics"]["dev"] for row in variant_rows]
        trainable = [metrics["trainable_shared_weight_latent_selector"] for metrics in dev_metrics]
        frozen = [metrics["frozen_same_architecture_latent_selector"] for metrics in dev_metrics]
        random_base = [metrics["random_trajectory"] for metrics in dev_metrics]
        first = [metrics["first_trajectory_order_baseline"] for metrics in dev_metrics]
        best_baseline = [best_non_oracle_baseline(metrics)[1] for metrics in dev_metrics]
        control_scores = [control_health(row) for row in variant_rows]
        variant_metrics[variant] = {
            "completed_seeds": sorted({int(row["seed"]) for row in variant_rows}),
            "trainable_pass_at_1_mean": mean_metric(trainable, "pass_at_1"),
            "trainable_conditional_accuracy_mean": mean_metric(trainable, "conditional_selector_accuracy"),
            "trainable_mrr_mean": mean_metric(trainable, "mrr"),
            "trainable_top_2_accuracy_mean": mean_metric(trainable, "top_2_accuracy"),
            "oracle_pass_at_8_mean": mean_metric(trainable, "oracle_pass_at_8"),
            "selection_efficiency_mean": mean_metric(trainable, "selection_efficiency"),
            "frozen_pass_at_1_mean": mean_metric(frozen, "pass_at_1"),
            "trainable_minus_frozen_pass_at_1": mean_metric(trainable, "pass_at_1") - mean_metric(frozen, "pass_at_1"),
            "random_pass_at_1_mean": mean_metric(random_base, "pass_at_1"),
            "first_pass_at_1_mean": mean_metric(first, "pass_at_1"),
            "best_non_oracle_baseline_pass_at_1_mean": float(np.mean(best_baseline)) if best_baseline else 0.0,
            "trainable_minus_best_non_oracle_baseline": mean_metric(trainable, "pass_at_1") - (float(np.mean(best_baseline)) if best_baseline else 0.0),
            "control_health_mean": float(np.mean(control_scores)) if control_scores else 0.0,
            "all_seed_gates": aggregate_seed_gates(variant_rows),
        }
    selected = max(
        sorted(variant_metrics),
        key=lambda name: (
            variant_metrics[name]["trainable_pass_at_1_mean"],
            variant_metrics[name]["trainable_minus_frozen_pass_at_1"],
            variant_metrics[name]["trainable_minus_best_non_oracle_baseline"],
            variant_metrics[name]["selection_efficiency_mean"],
            variant_metrics[name]["control_health_mean"],
        ),
    )
    return {
        "selection_rule": [
            "1. mean dev trainable latent pass@1",
            "2. mean trainable-minus-frozen pass@1 delta",
            "3. mean trainable-minus-best-non-oracle-baseline pass@1 delta",
            "4. mean selection efficiency",
            "5. mean control health",
        ],
        "selected_config": selected,
        "variant_metrics": variant_metrics,
    }


def stage7b_decision(
    rows: Sequence[Dict[str, object]],
    comparison: Dict[str, object],
    selected_name: str,
    split_leakage: Dict[str, object],
    output_leakage: Dict[str, object],
    duplicate_audit: Dict[str, object],
    near_duplicate_audit: Dict[str, object],
) -> Dict[str, object]:
    selected_rows = [row for row in rows if row["variant"] == selected_name]
    selected_metrics = comparison["variant_metrics"][selected_name]
    gates = {
        "completed_seeds_gte_3": len(selected_metrics["completed_seeds"]) >= 3,
        "oracle_pass_at_8_between_0_25_and_0_85": 0.25 <= float(selected_metrics["oracle_pass_at_8_mean"]) <= 0.85,
        "trainable_beats_frozen_mean_pass_at_1": float(selected_metrics["trainable_minus_frozen_pass_at_1"]) > 0.0,
        "trainable_beats_random_and_first": (
            float(selected_metrics["trainable_pass_at_1_mean"]) > float(selected_metrics["random_pass_at_1_mean"])
            and float(selected_metrics["trainable_pass_at_1_mean"]) > float(selected_metrics["first_pass_at_1_mean"])
        ),
        "trainable_competitive_with_embedding_static_text": competitive_with_embedding_static_text(selected_rows),
        "randomized_labels_collapse": randomized_label_trainable_delta(selected_rows) >= 0.05,
        "mismatch_or_cross_task_shuffle_degrades_by_0_05": max(
            mean_control_delta(selected_rows, "candidate_evidence_mismatch"),
            mean_control_delta(selected_rows, "cross_task_user_goal_shuffle"),
            mean_control_delta(selected_rows, "cross_task_policy_shuffle"),
            mean_control_delta(selected_rows, "cross_task_tool_observation_shuffle"),
        )
        >= 0.05,
        "candidate_order_invariance_passes": all(bool(row["invariance_audit"].get("candidate_order_invariance_passes", False)) for row in selected_rows),
        "physical_order_invariance_passes": all(bool(row["invariance_audit"].get("physical_order_invariance_passes", False)) for row in selected_rows),
        "leakage_audits_pass": bool(split_leakage.get("passes", False)) and bool(output_leakage.get("passes", False)),
        "duplicate_and_near_duplicate_audits_pass": bool(duplicate_audit.get("passes", False)) and bool(near_duplicate_audit.get("passes", False)),
        "gradient_update_audits_pass": all(gradient_update_audit_passes(row) for row in selected_rows),
        "one_final_config_selected_for_stage7c": bool(selected_name),
        "no_final_claim_made": True,
    }
    if all(gates.values()):
        decision = "A. READY_FOR_STAGE7C_FINAL"
    elif not gates["trainable_beats_frozen_mean_pass_at_1"] or not gates["trainable_beats_random_and_first"]:
        decision = "C. STAGE7B_NO_LATENT_SIGNAL"
    else:
        decision = "B. NEED_STAGE7B_VIEW_OR_BASELINE_FIX"
    return {
        "decision": decision,
        "stage7b_success_gates": gates,
        "failure_diagnosis": failure_diagnosis(gates, selected_metrics, selected_rows),
    }


def seed_success_gates(row: Dict[str, object]) -> Dict[str, bool]:
    dev = row["metrics"]["dev"]
    trainable = dev["trainable_shared_weight_latent_selector"]
    frozen = dev["frozen_same_architecture_latent_selector"]
    randomized = dev["randomized_labels_trainable_latent_selector"]
    random_base = dev["random_trajectory"]
    first = dev["first_trajectory_order_baseline"]
    base_pass = float(trainable["pass_at_1"])
    return {
        "oracle_pass_at_8_between_0_25_and_0_85": 0.25 <= float(trainable["oracle_pass_at_8"]) <= 0.85,
        "trainable_beats_frozen": base_pass > float(frozen["pass_at_1"]),
        "trainable_beats_random": base_pass > float(random_base["pass_at_1"]),
        "trainable_beats_first": base_pass > float(first["pass_at_1"]),
        "randomized_labels_trainable_collapse": float(randomized["pass_at_1"]) <= max(0.0, base_pass - 0.05),
        "candidate_order_invariance_passes": bool(row["invariance_audit"].get("candidate_order_invariance_passes", False)),
        "physical_order_invariance_passes": bool(row["invariance_audit"].get("physical_order_invariance_passes", False)),
        "gradient_update_audits_pass": gradient_update_audit_passes(row),
    }


def gradient_update_audit_passes(row: Dict[str, object]) -> bool:
    trainable = row.get("trainable_audit", {})
    frozen = row.get("frozen_audit", {})
    return bool(
        trainable.get("shared_parameter_identity", False)
        and float(trainable.get("agent_grad_norm_mean", 0.0)) > 0.0
        and float(trainable.get("agent_parameter_delta", 0.0)) > 0.0
        and float(trainable.get("coordinator_grad_norm_mean", 0.0)) > 0.0
        and float(trainable.get("coordinator_parameter_delta", 0.0)) > 0.0
        and bool(trainable.get("activation_requires_grad_before_coordinator", False))
        and bool(trainable.get("no_detach_between_clone_activations_and_loss", False))
        and float(frozen.get("agent_grad_norm_mean", 1.0)) == 0.0
        and float(frozen.get("agent_parameter_delta", 1.0)) == 0.0
        and float(frozen.get("coordinator_grad_norm_mean", 0.0)) > 0.0
        and float(frozen.get("coordinator_parameter_delta", 0.0)) > 0.0
    )


def competitive_with_embedding_static_text(selected_rows: Sequence[Dict[str, object]]) -> bool:
    margins = []
    for row in selected_rows:
        dev = row["metrics"]["dev"]
        trainable = float(dev["trainable_shared_weight_latent_selector"]["pass_at_1"])
        baseline = max(
            float(dev["embedding_reranker"]["pass_at_1"]),
            float(dev["static_trajectory_feature_reranker"]["pass_at_1"]),
            float(dev["text_only_single_reviewer"]["pass_at_1"]),
            float(dev["text_only_multi_agent_reviewer"]["pass_at_1"]),
        )
        margins.append(trainable - baseline)
    return bool(margins and float(np.mean(margins)) >= -0.02)


def mean_control_delta(rows: Sequence[Dict[str, object]], control: str) -> float:
    deltas = []
    for row in rows:
        base = float(row["metrics"]["dev"]["trainable_shared_weight_latent_selector"]["pass_at_1"])
        value = float((row.get("controls", {}).get(control) or {}).get("pass_at_1", base))
        deltas.append(base - value)
    return float(np.mean(deltas)) if deltas else 0.0


def control_health(row: Dict[str, object]) -> float:
    gates = [
        randomized_label_trainable_delta([row]) >= 0.05,
        max(
            mean_single_control_delta(row, "candidate_evidence_mismatch"),
            mean_single_control_delta(row, "cross_task_user_goal_shuffle"),
            mean_single_control_delta(row, "cross_task_policy_shuffle"),
            mean_single_control_delta(row, "cross_task_tool_observation_shuffle"),
        )
        >= 0.05,
        bool(row["invariance_audit"].get("candidate_order_invariance_passes", False)),
        bool(row["invariance_audit"].get("physical_order_invariance_passes", False)),
    ]
    return sum(1 for value in gates if value) / float(len(gates))


def mean_single_control_delta(row: Dict[str, object], control: str) -> float:
    base = float(row["metrics"]["dev"]["trainable_shared_weight_latent_selector"]["pass_at_1"])
    value = float((row.get("controls", {}).get(control) or {}).get("pass_at_1", base))
    return base - value


def randomized_label_trainable_delta(rows: Sequence[Dict[str, object]]) -> float:
    deltas = []
    for row in rows:
        dev = row["metrics"]["dev"]
        base = float(dev["trainable_shared_weight_latent_selector"]["pass_at_1"])
        randomized = float(dev["randomized_labels_trainable_latent_selector"]["pass_at_1"])
        deltas.append(base - randomized)
    return float(np.mean(deltas)) if deltas else 0.0


def failure_diagnosis(gates: Dict[str, bool], selected_metrics: Dict[str, object], selected_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "oracle_pass_at_8_mean": selected_metrics.get("oracle_pass_at_8_mean"),
        "candidate_pool_too_easy_or_hard": not gates.get("oracle_pass_at_8_between_0_25_and_0_85", False),
        "embedding_static_text_saturation": not gates.get("trainable_competitive_with_embedding_static_text", False),
        "evidence_views_not_useful": not gates.get("mismatch_or_cross_task_shuffle_degrades_by_0_05", False),
        "candidate_only_shortcut": mean_control_delta(selected_rows, "candidate_only") < 0.05,
        "possible_generator_identity_leakage": not gates.get("leakage_audits_pass", False),
        "insufficient_training_tasks": False,
        "architecture_mismatch": not gates.get("trainable_beats_frozen_mean_pass_at_1", False),
    }


def best_non_oracle_baseline(metrics: Dict[str, Dict[str, object]]) -> Tuple[str, float]:
    excluded = {
        "oracle_pass_at_8",
        "trainable_shared_weight_latent_selector",
        "randomized_labels_trainable_latent_selector",
        "llm_judge_baseline",
    }
    candidates = {
        name: float(row.get("pass_at_1", 0.0))
        for name, row in metrics.items()
        if name not in excluded and not bool(row.get("unavailable", False))
    }
    if not candidates:
        return ("", 0.0)
    name = max(sorted(candidates), key=lambda key: candidates[key])
    return name, candidates[name]


def mean_metric(rows: Sequence[Dict[str, object]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(np.mean(values)) if values else 0.0


def aggregate_seed_gates(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    keys = sorted({key for row in rows for key in row.get("success_gates", {})})
    return {
        key: {
            "pass_count": sum(1 for row in rows if bool(row.get("success_gates", {}).get(key, False))),
            "total": len(rows),
        }
        for key in keys
    }


def generator_identity_only_diagnostic(
    baseline: BestGeneratorOnTrainAuditBaseline,
    examples: Sequence[TauBenchTrajectorySelectionExample],
) -> Dict[str, object]:
    scores = baseline.predict_scores(examples)
    metrics = evaluate_scores("generator_identity_only_diagnostic_audit_only", scores, examples, split="audit")
    return {
        "audit_only_not_selector_visible": True,
        "best_generator_on_train": baseline.best_generator,
        "generator_scores_on_train": baseline.generator_scores,
        "metrics": {key: value for key, value in metrics.items() if key not in {"predictions", "selected_candidate_ids", "correct_by_example"}},
    }


def save_checkpoints(
    checkpoint_dir: Path,
    variant: Stage7BVariant,
    seed: int,
    models: Dict[str, object],
    config: Stage7BConfig,
) -> Dict[str, str]:
    root = checkpoint_dir / variant.name / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for name in ("trainable", "frozen", "raw_latent", "randomized_labels"):
        fit = models.get(name)
        path = root / f"{fit.method}.pt"
        torch.save(
            {
                "benchmark": BENCHMARK,
                "seed": int(seed),
                "variant": asdict(variant),
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
        path = root / f"{name}.pt"
        torch.save(
            {
                "benchmark": BENCHMARK,
                "seed": int(seed),
                "variant": asdict(variant),
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


def audit_rows_from_result_row(row: Dict[str, object]) -> List[Dict[str, object]]:
    out = []
    for key in ("trainable_audit", "frozen_audit", "raw_latent_audit", "randomized_labels_audit"):
        audit = row.get(key)
        if isinstance(audit, dict) and audit:
            out.append({"benchmark": BENCHMARK, "seed": row.get("seed"), "variant": row.get("variant"), "type": key, **audit})
    out.append({"benchmark": BENCHMARK, "seed": row.get("seed"), "variant": row.get("variant"), "type": "controls", **row.get("controls", {})})
    out.append({"benchmark": BENCHMARK, "seed": row.get("seed"), "variant": row.get("variant"), "type": "invariance_audit", **row.get("invariance_audit", {})})
    out.append({"benchmark": BENCHMARK, "seed": row.get("seed"), "variant": row.get("variant"), "type": "success_gates", **row.get("success_gates", {})})
    return out


def agent_config_for(config: Stage7BConfig) -> SharedTransformerAgentConfig:
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


def training_config_for(config: Stage7BConfig, variant: Stage7BVariant) -> RealSharedWeightTrainingConfig:
    return RealSharedWeightTrainingConfig(
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        lr=float(variant.lr),
        weight_decay=float(config.weight_decay),
        patience=int(config.patience),
        gradient_accumulation_steps=1,
        mixed_precision=str(config.mixed_precision),
        gradient_clip_norm=float(variant.gradient_clip_norm),
    )


def message_config_for(variant: Stage7BVariant) -> MessageChannelConfig:
    return MessageChannelConfig(
        use_msg_token=False,
        readout_source="pooled",
        use_message_head=False,
        coordinator_family=str(variant.coordinator_family),
        use_private_cue_aux=False,
        num_avenues=int(variant.num_avenues),
        avenue_prompt_mode=str(variant.avenue_prompt_mode),
        avenue_dropout=float(variant.avenue_dropout),
        canonicalize_role_order=True,
        canonicalize_avenue_order=True,
    )


def coordinator_config_for(config: Stage7BConfig, variant: Stage7BVariant) -> LatentCoordinatorConfig:
    return LatentCoordinatorConfig(
        family=str(variant.coordinator_family),
        input_dim=int(config.hidden_dim),
        model_dim=int(config.hidden_dim),
        num_heads=1,
        num_layers=1,
        ff_dim=max(16, int(config.hidden_dim) * 2),
        dropout=0.0,
        num_avenues=int(variant.num_avenues),
        avenue_topk=0,
    )


def resolve_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(device)


def write_outputs(
    result: Dict[str, object],
    comparison: Dict[str, object],
    audit_rows: Sequence[Dict[str, object]],
    results_path: Path,
    audit_path: Path,
    variant_comparison_path: Path,
    report_path: Path,
) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    variant_comparison_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    variant_comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    audit_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in audit_rows), encoding="utf-8")
    report_path.write_text(render_report(result, comparison), encoding="utf-8")


def append_progress_row(
    path: Path,
    row: Dict[str, object],
    config: Stage7BConfig,
    candidate_pool_path: Path,
    pool_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "artifact": "stage7b_tau_dev_selector_row_progress",
        "created_at_utc": _now(),
        "candidate_pool_path": str(candidate_pool_path),
        "pool_name": pool_name,
        "config": asdict(config),
        "row": row,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def load_progress_rows(
    path: Path,
    config: Stage7BConfig,
    candidate_pool_path: Path,
    pool_name: str,
) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    wanted_config = asdict(config)
    wanted_pool = str(candidate_pool_path)
    rows: Dict[Tuple[str, int], Dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("config") != wanted_config:
            continue
        if str(record.get("candidate_pool_path", "")) != wanted_pool:
            continue
        if str(record.get("pool_name", "")) != str(pool_name):
            continue
        row = record.get("row")
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")) != "completed":
            continue
        rows[(str(row.get("variant")), int(row.get("seed", -1)))] = row
    return [rows[key] for key in sorted(rows)]


def render_report(result: Dict[str, object], comparison: Dict[str, object]) -> str:
    lines = [
        "# Stage 7B tau-bench Dev Validation",
        "",
        "## Decision",
        f"- Decision: `{result['decision']}`",
        f"- Selected config for Stage 7C: `{comparison['selected_config']}`",
        "- Stage 7C final validation run: `False`.",
        "- Final benchmark claim made: `False`.",
        "",
        "## Dataset",
        f"- Candidate pool: `{result['candidate_pool_path']}`",
        f"- Pool name: `{result['pool_name']}`",
        f"- Domain: `{result['domain']}`",
        f"- K: `{result['k']}`",
        f"- Split summary: `{json.dumps(result['split_summary'], sort_keys=True)}`",
        "",
        "## Variant Comparison",
    ]
    for name, metrics in comparison["variant_metrics"].items():
        lines.extend(
            [
                f"### {name}",
                f"- Trainable pass@1 mean: `{metrics['trainable_pass_at_1_mean']:.4f}`",
                f"- Frozen pass@1 mean: `{metrics['frozen_pass_at_1_mean']:.4f}`",
                f"- Trainable-frozen delta: `{metrics['trainable_minus_frozen_pass_at_1']:.4f}`",
                f"- Best non-oracle baseline mean: `{metrics['best_non_oracle_baseline_pass_at_1_mean']:.4f}`",
                f"- Trainable-best baseline delta: `{metrics['trainable_minus_best_non_oracle_baseline']:.4f}`",
                f"- Oracle pass@8 mean: `{metrics['oracle_pass_at_8_mean']:.4f}`",
                f"- Selection efficiency mean: `{metrics['selection_efficiency_mean']:.4f}`",
                f"- Control health mean: `{metrics['control_health_mean']:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Gates",
            f"`{json.dumps(result['stage7b_success_gates'], sort_keys=True)}`",
            "",
            "## Failure Diagnosis",
            f"`{json.dumps(result['failure_diagnosis'], sort_keys=True)}`",
            "",
            "## Notes",
            "- Labels are official tau2 evaluator success/failure labels from Stage 7B.0.",
            "- Generator/source identity is used only in audit-only baselines and breakdowns.",
            "- Selector-visible fields remain free of official labels, evaluator verdicts, hidden goal state, and generator identity.",
            "- This is dev-only architecture/config selection and signal detection.",
            "",
        ]
    )
    return "\n".join(lines)


def _pool_name_from_path(path: Path) -> str:
    name = path.name
    if "balanced" in name:
        return "balanced_pool"
    if "nontrivial" in name:
        return "nontrivial_pool"
    if "all" in name:
        return "all_available_pool"
    return path.stem


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def task_sort_key(task_id: str) -> Tuple[int, str]:
    text = str(task_id)
    return (int(text), text) if text.isdigit() else (10**9, text)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
