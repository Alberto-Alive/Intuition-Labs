from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.coordinators.latent_coordination import LatentCoordinatorConfig
from src.coordinators.mlp import MLPTrainingConfig
from src.datasets.multiview_code_patch_selection import (
    MultiViewCodePatchDatasetConfig,
    MultiViewTaskExample,
    View,
    apply_example_control,
    build_multiview_code_patch_splits,
    format_clone_prompt,
    randomized_labels_for_examples,
    split_leakage_audit,
    example_oracle_metadata,
)
from src.experiments.real_shared_weight_latent_coordination import (
    BENCHMARK,
    BagOfWordsDiagnosticBaseline,
    EVIDENCE_TOKEN_UPPER_BOUND_METHOD,
    FROZEN_EVIDENCE_TOKEN_UPPER_BOUND_METHOD,
    MessageChannelConfig,
    RealSharedWeightTrainingConfig,
    SharedTransformerAgentConfig,
    TextOutputOnlyCoordinator,
    fit_context_baseline,
    fit_latent_system,
    predict_context_baseline,
    predict_latent_system,
    _message_mechanism_diagnostic_rows,
)


REPORT_PATH = Path("reports/ARCHITECTURE_SEARCH.md")
RESULTS_PATH = Path("results/architecture_search_results.json")
PROPOSED_RAW = "trainable_shared_agent_latent_coordinator"
TEXT_ONLY = "text_output_only_multi_agent_coordinator"
ORACLE = "explicit_evidence_oracle"
LOCKED_CONTROL_CONDITIONS = (
    "randomized_labels",
    "hidden_states_shuffled_across_examples",
    "view_masked",
    "view_shuffled",
    "role_labels_shuffled",
)
INVARIANCE_CONDITIONS = (
    "physical_order_shuffled_roles_preserved",
    "candidate_order_shuffled",
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    description: str
    message_config: MessageChannelConfig
    coordinator_config: LatentCoordinatorConfig


@dataclass(frozen=True)
class StageConfig:
    name: str
    n_train: int
    n_dev: int
    n_test: int
    seeds: tuple[int, ...]
    epochs: int
    patience: int
    hidden_dim: int
    tiny_layers: int
    tiny_ff_dim: int
    batch_size: int = 32
    lr: float = 0.002
    max_candidates: int = 999
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Empirical architecture search for semantic active-message coordination.")
    parser.add_argument("--max-stage", choices=("stage1", "stage2", "stage3"), default="stage1")
    parser.add_argument("--fixed-topk-stage3", action="store_true", help="Run hard validation only for the frozen topk_attention_no_head architecture.")
    parser.add_argument("--output", default=str(RESULTS_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.fixed_topk_stage3:
        results = run_fixed_topk_stage3(device=args.device)
    else:
        results = run_search(max_stage=args.max_stage, device=args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(results), encoding="utf-8")


def run_search(max_stage: str, device: str) -> Dict[str, object]:
    stages = _stage_configs()
    candidates = _candidate_specs()
    completed: Dict[str, List[Dict[str, object]]] = {}
    selected = candidates
    stage_order = ["stage1", "stage2", "stage3"]
    max_index = stage_order.index(max_stage)
    for stage in stages[: max_index + 1]:
        stage_candidates = selected[: stage.max_candidates]
        stage_rows = _run_stage(stage, stage_candidates, device=device)
        completed[stage.name] = stage_rows
        valid = [row for row in stage_rows if row.get("selection_valid")]
        valid.sort(key=lambda row: float(row.get("dev_delta", -999.0)), reverse=True)
        keep = 5 if stage.name == "stage1" else 2
        selected_names = {str(row["candidate"]) for row in valid[:keep]}
        selected = [candidate for candidate in stage_candidates if candidate.name in selected_names]
        if not selected:
            break
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "task": "semantic_no_literal_cue_architecture_search",
            "selection_rule": "Architecture selection uses dev metrics and dev controls only; test metrics are logged after candidate evaluation.",
            "max_stage_requested": max_stage,
            "report_path": str(REPORT_PATH),
        },
        "search_space": _search_space_rows(),
        "candidate_configs": {candidate.name: _candidate_config(candidate) for candidate in candidates},
        "stages": completed,
    }


def run_fixed_topk_stage3(device: str) -> Dict[str, object]:
    stage = _stage_configs()[2]
    candidate = next(candidate for candidate in _candidate_specs() if candidate.name == "topk_attention_no_head")
    rows = _run_stage(stage, [candidate], device=device, run_evidence_diagnostic=False)
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "task": "semantic_no_literal_cue_fixed_topk_attention_no_head_stage3",
            "selection_rule": "Architecture frozen before Stage 3; no architecture selection or modification occurs in this run.",
            "max_stage_requested": "stage3_fixed_topk",
            "frozen_architecture": candidate.name,
            "report_path": str(REPORT_PATH),
        },
        "search_space": _search_space_rows(),
        "candidate_configs": {candidate.name: _candidate_config(candidate)},
        "stages": {"stage3": rows},
    }


def _run_stage(
    stage: StageConfig,
    candidates: Sequence[CandidateSpec],
    device: str,
    run_evidence_diagnostic: bool = True,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seed in stage.seeds:
        print(f"{stage.name} seed={seed}: building real program-analysis splits")
        splits = build_multiview_code_patch_splits(_dataset_config(stage), seed=seed, repo_root=Path("."))
        baselines = _fit_baselines(stage, splits, seed=seed, device=device)
        leakage = split_leakage_audit(BENCHMARK, seed, splits)
        for candidate_index, candidate in enumerate(candidates):
            print(f"{stage.name} seed={seed}: fitting {candidate.name}")
            row = _evaluate_candidate(
                candidate,
                stage,
                splits,
                seed=seed + candidate_index * 1000,
                displayed_seed=seed,
                device=device,
                baselines=baselines,
                leakage=leakage,
                run_evidence_diagnostic=run_evidence_diagnostic,
            )
            rows.append(row)
    return rows


def _fit_baselines(
    stage: StageConfig,
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    seed: int,
    device: str,
) -> Dict[str, object]:
    agent_config = _agent_config(stage)
    training = _training_config(stage)
    coordinator_config = _coordinator_config(stage, family="cross_attention", num_layers=1, dropout=0.0)
    raw = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=replace(coordinator_config, input_dim=agent_config.hidden_dim),
        training_config=training,
        num_classes=8,
        seed=seed + 11_000,
        device=device,
        trainable_agent=True,
        method=PROPOSED_RAW,
    )
    text = TextOutputOnlyCoordinator(
        num_classes=8,
        training=MLPTrainingConfig(
            epochs=max(3, stage.epochs),
            batch_size=stage.batch_size,
            lr=stage.lr,
            weight_decay=0.0001,
            patience=max(2, stage.patience),
            hidden_dims=(32,),
        ),
        seed=seed + 12_000,
        device=device,
        feature_dim=128,
    )
    text.fit(splits["train"], splits["dev"])
    train_labels = np.asarray([example.label for example in splits["train"]], dtype=np.int64)
    majority = int(np.argmax(np.bincount(train_labels, minlength=8)))
    single_view = {}
    for role_index in range(len(splits["train"][0].views)):
        baseline = BagOfWordsDiagnosticBaseline(
            method=f"single_view_text_role_{role_index}",
            num_classes=8,
            training=MLPTrainingConfig(
                epochs=max(3, stage.epochs),
                batch_size=stage.batch_size,
                lr=stage.lr,
                weight_decay=0.0001,
                patience=max(2, stage.patience),
                hidden_dims=(32,),
            ),
            seed=seed + 12_500 + role_index,
            device=device,
            text_builder=lambda example, role_index=role_index: format_clone_prompt(example, role_index),
            feature_dim=128,
        )
        baseline.fit(splits["train"], splits["dev"])
        single_view[f"single_view_text_role_{role_index}"] = baseline
    full_context = fit_context_baseline(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        training_config=training,
        num_classes=8,
        seed=seed + 13_000,
        device=device,
        method="single_agent_full_context",
        prompt_mode="full",
    )
    return {
        "raw": raw,
        "text": text,
        "majority_prediction": majority,
        "candidate_order_prediction": 0,
        "single_view": single_view,
        "full_context": full_context,
    }


def _evaluate_candidate(
    candidate: CandidateSpec,
    stage: StageConfig,
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    seed: int,
    displayed_seed: int,
    device: str,
    baselines: Dict[str, object],
    leakage: Dict[str, object],
    run_evidence_diagnostic: bool = True,
) -> Dict[str, object]:
    agent_config = _agent_config(stage)
    training = _training_config(stage)
    trainable = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=candidate.coordinator_config,
        training_config=training,
        num_classes=8,
        seed=seed + 101,
        device=device,
        trainable_agent=True,
        method=f"search_trainable__{candidate.name}",
        message_config=candidate.message_config,
    )
    frozen = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=candidate.coordinator_config,
        training_config=training,
        num_classes=8,
        seed=seed + 201,
        device=device,
        trainable_agent=False,
        method=f"search_frozen__{candidate.name}",
        message_config=candidate.message_config,
    )
    randomized = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=candidate.coordinator_config,
        training_config=training,
        num_classes=8,
        seed=seed + 301,
        device=device,
        trainable_agent=True,
        method=f"search_randomized__{candidate.name}",
        condition="randomized_labels",
        train_labels=randomized_labels_for_examples(splits["train"], seed + 401, 8),
        dev_labels=randomized_labels_for_examples(splits["dev"], seed + 501, 8),
        message_config=candidate.message_config,
    )
    evidence = None
    frozen_evidence = None
    if run_evidence_diagnostic:
        evidence = fit_latent_system(
            train_examples=splits["train"],
            dev_examples=splits["dev"],
            agent_config=agent_config,
            coordinator_config=candidate.coordinator_config,
            training_config=training,
            num_classes=8,
            seed=seed + 601,
            device=device,
            trainable_agent=True,
            method=EVIDENCE_TOKEN_UPPER_BOUND_METHOD,
            message_config=replace(
                candidate.message_config,
                use_msg_token=True,
                readout_source="evidence_tokens",
                use_message_head=True,
                coordinator_family="candidate_query_cross_attention",
            ),
        )
        frozen_evidence = fit_latent_system(
            train_examples=splits["train"],
            dev_examples=splits["dev"],
            agent_config=agent_config,
            coordinator_config=candidate.coordinator_config,
            training_config=training,
            num_classes=8,
            seed=seed + 701,
            device=device,
            trainable_agent=False,
            method=FROZEN_EVIDENCE_TOKEN_UPPER_BOUND_METHOD,
            message_config=replace(
                candidate.message_config,
                use_msg_token=True,
                readout_source="evidence_tokens",
                use_message_head=True,
                coordinator_family="candidate_query_cross_attention",
            ),
        )

    metrics = _candidate_metrics(trainable, frozen, randomized, evidence, frozen_evidence, baselines, splits, seed)
    diagnostics = _message_mechanism_diagnostic_rows(trainable, splits, seed=seed, num_classes=8, device=device)
    collapse = _collapse_summary(diagnostics)
    private_cue = _private_cue_summary(diagnostics)
    combined_probe = _combined_probe_summary(diagnostics)
    per_family = _per_family_accuracy(trainable, splits["test"], seed)
    per_role = _per_role_ablation(trainable, splits["test"], seed)
    rejection_reasons = _rejection_reasons(metrics, collapse, per_role, leakage, chance=0.125)
    return {
        "stage": stage.name,
        "seed": displayed_seed,
        "candidate": candidate.name,
        "description": candidate.description,
        "architecture_config": _candidate_config(candidate),
        "dev_accuracy": metrics["dev"],
        "test_accuracy": metrics["test"],
        "dev_delta": metrics["dev"]["trainable"] - metrics["dev"]["frozen"],
        "test_delta": metrics["test"]["trainable"] - metrics["test"]["frozen"],
        "selection_valid": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "trainable_audit": _audit_subset(trainable.audit),
        "frozen_audit": _audit_subset(frozen.audit),
        "message_collapse": collapse,
        "private_cue_probes": private_cue,
        "combined_message_final_label_probe": combined_probe,
        "per_family_accuracy": per_family,
        "per_role_ablation": per_role,
        "leakage_audit_passes": _leakage_passes(leakage),
    }


def _candidate_metrics(
    trainable,
    frozen,
    randomized,
    evidence,
    frozen_evidence,
    baselines: Dict[str, object],
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    seed: int,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    raw = baselines["raw"]
    text = baselines["text"]
    majority = int(baselines["majority_prediction"])
    candidate_order = int(baselines["candidate_order_prediction"])
    single_view = dict(baselines.get("single_view", {}))
    full_context = baselines.get("full_context")
    for split in ("dev", "test"):
        examples = list(splits[split])
        y = np.asarray([example.label for example in examples], dtype=np.int64)
        values = {
            "trainable": _acc(predict_latent_system(trainable, examples, "none", seed), y),
            "frozen": _acc(predict_latent_system(frozen, examples, "none", seed), y),
            "text_only": _acc(text.predict(examples), y),
            "raw_latent": _acc(predict_latent_system(raw, examples, "none", seed), y),
            "explicit_evidence_oracle": 1.0,
            "evidence_token_diagnostic_trainable": (
                _acc(predict_latent_system(evidence, examples, "none", seed), y) if evidence is not None else None
            ),
            "evidence_token_diagnostic_frozen": (
                _acc(predict_latent_system(frozen_evidence, examples, "none", seed), y) if frozen_evidence is not None else None
            ),
            "majority_baseline": float(np.mean(np.full(len(y), majority, dtype=np.int64) == y)),
            "candidate_order_baseline": float(np.mean(np.full(len(y), candidate_order, dtype=np.int64) == y)),
        }
        for method, baseline in single_view.items():
            values[str(method)] = _acc(baseline.predict(examples), y)
        if full_context is not None:
            values["single_agent_full_context"] = _acc(predict_context_baseline(full_context, examples), y)
        if split == "dev":
            values["randomized_labels"] = _acc(predict_latent_system(randomized, examples, "none", seed), y)
        for condition in LOCKED_CONTROL_CONDITIONS[1:] + INVARIANCE_CONDITIONS:
            controlled = apply_example_control(examples, condition=condition, seed=seed)
            cy = np.asarray([example.label for example in controlled], dtype=np.int64)
            values[condition] = _acc(predict_latent_system(trainable, controlled, condition, seed), cy)
        if split == "test":
            values["randomized_labels"] = _acc(predict_latent_system(randomized, examples, "none", seed), y)
        out[split] = values
    return out


def _rejection_reasons(
    metrics: Dict[str, Dict[str, float]],
    collapse: Dict[str, Dict[str, float]],
    per_role: Dict[str, object],
    leakage: Dict[str, object],
    chance: float,
) -> List[str]:
    dev = metrics["dev"]
    reasons: List[str] = []
    if dev["trainable"] <= dev["frozen"]:
        reasons.append("trainable <= frozen on dev")
    if dev["trainable"] <= dev["text_only"]:
        reasons.append("trainable <= text-only on dev")
    if dev["trainable"] <= dev["raw_latent"]:
        reasons.append("trainable <= raw-latent on dev")
    single_view_values = [float(dev.get(f"single_view_text_role_{index}", 0.0)) for index in range(4)]
    if single_view_values and dev["trainable"] <= max(single_view_values):
        reasons.append("trainable <= a single-view baseline on dev")
    if dev.get("majority_baseline", 1.0) > chance + 0.10:
        reasons.append("majority baseline above threshold on dev")
    for condition in LOCKED_CONTROL_CONDITIONS:
        if dev.get(condition, 1.0) > chance + 0.10:
            reasons.append(f"{condition} control above threshold on dev")
    if dev.get("candidate_order_baseline", 1.0) > chance + 0.10:
        reasons.append("candidate-order baseline above threshold on dev")
    for source in ("active_message_readout", "message_head_output"):
        stats = collapse.get(source)
        if stats and (float(stats.get("variance_mean", 0.0)) <= 1e-5 or float(stats.get("mean_cosine_similarity", 0.0)) >= 0.98):
            reasons.append(f"{source} collapsed")
    if bool(per_role.get("one_role_only_failure", False)):
        reasons.append("improvement explained by one role only")
    if not _leakage_passes(leakage):
        reasons.append("train/dev/test leakage audit failed")
    return reasons


def _collapse_summary(diagnostics: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for row in diagnostics:
        if row.get("split") != "dev" or row.get("probe") != "message_collapse_statistics":
            continue
        source = str(row.get("feature_source"))
        if source in {"active_message_readout", "message_head_output"}:
            out[source] = {
                "variance_mean": float(row.get("variance_mean", 0.0)),
                "norm_mean": float(row.get("norm_mean", 0.0)),
                "norm_std": float(row.get("norm_std", 0.0)),
                "mean_cosine_similarity": float(row.get("mean_cosine_similarity", 0.0)),
                "within_label_cosine": float(row.get("within_label_cosine", 0.0)),
                "between_label_cosine": float(row.get("between_label_cosine", 0.0)),
            }
    return out


def _private_cue_summary(diagnostics: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    grouped: Dict[str, List[float]] = {}
    per_role: Dict[str, Dict[str, float]] = {}
    for row in diagnostics:
        if row.get("split") != "test" or row.get("probe") != "private_cue_probe":
            continue
        source = str(row.get("feature_source"))
        grouped.setdefault(source, []).append(float(row.get("accuracy", 0.0)))
        per_role.setdefault(source, {})[str(row.get("role_index"))] = float(row.get("accuracy", 0.0))
    for source, values in grouped.items():
        out[source] = {"mean": mean(values), "per_role": per_role.get(source, {})}
    return out


def _combined_probe_summary(diagnostics: Sequence[Dict[str, object]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in diagnostics:
        if row.get("split") == "test" and row.get("probe") == "combined_representation_probe":
            out[str(row.get("feature_source"))] = float(row.get("accuracy", 0.0))
    return out


def _per_family_accuracy(result, examples: Sequence[MultiViewTaskExample], seed: int) -> Dict[str, float]:
    predictions = predict_latent_system(result, examples, "none", seed)
    grouped: Dict[str, List[bool]] = {}
    for example, prediction in zip(examples, predictions):
        family = str(example_oracle_metadata(example).get("problem_family"))
        grouped.setdefault(family, []).append(int(prediction) == int(example.label))
    return {family: float(np.mean(values)) for family, values in sorted(grouped.items())}


def _per_role_ablation(result, examples: Sequence[MultiViewTaskExample], seed: int) -> Dict[str, object]:
    base_y = np.asarray([example.label for example in examples], dtype=np.int64)
    base_acc = _acc(predict_latent_system(result, examples, "none", seed), base_y)
    role_acc: Dict[str, float] = {}
    drops: Dict[str, float] = {}
    for role_id in range(len(examples[0].views)):
        ablated = _mask_single_role(examples, role_id)
        y = np.asarray([example.label for example in ablated], dtype=np.int64)
        acc = _acc(predict_latent_system(result, ablated, "none", seed + role_id + 1), y)
        role_acc[str(role_id)] = acc
        drops[str(role_id)] = base_acc - acc
    large_drops = [drop for drop in drops.values() if drop > max(0.10, base_acc - 0.225)]
    return {
        "base_accuracy": base_acc,
        "accuracy_when_role_masked": role_acc,
        "accuracy_drop": drops,
        "one_role_only_failure": bool(len(large_drops) == 1 and max(drops.values(), default=0.0) > 0.12),
    }


def _mask_single_role(examples: Sequence[MultiViewTaskExample], role_id: int) -> List[MultiViewTaskExample]:
    out = []
    for example in examples:
        views = list(example.views)
        old = views[role_id]
        views[role_id] = View(
            role=old.role,
            text="This role is masked for per-role ablation. No private structural evidence is available.",
            source_path=old.source_path,
            source_type=f"{old.source_type}:role_ablation",
            allowed_visibility=old.allowed_visibility,
        )
        out.append(replace(example, views=tuple(views)))
    return out


def _audit_subset(audit: Dict[str, object]) -> Dict[str, object]:
    keys = (
        "agent_grad_norm_mean",
        "agent_parameter_delta",
        "active_message_readout_grad_norm_mean",
        "active_message_readout_parameter_delta",
        "message_head_grad_norm_mean",
        "message_head_parameter_delta",
        "avenue_bottleneck_grad_norm_mean",
        "avenue_bottleneck_parameter_delta",
        "single_avenue_adversary_grad_norm_mean",
        "single_avenue_adversary_parameter_delta",
        "coordinator_grad_norm_mean",
        "coordinator_parameter_delta",
        "shared_parameter_identity",
        "activation_requires_grad_before_coordinator",
        "per_clone_gradient_contribution",
        "loss_backward_reaches_shared_agent",
        "agent_trainable",
        "device",
        "batch_size",
        "gradient_accumulation_steps",
        "mixed_precision",
        "cuda_max_memory_allocated",
    )
    return {key: audit.get(key) for key in keys}


def _candidate_specs() -> List[CandidateSpec]:
    base_coord = _coordinator_config(_stage_configs()[0], family="cross_attention", num_layers=1, dropout=0.0)

    def msg(**updates) -> MessageChannelConfig:
        values = dict(
            use_msg_token=False,
            readout_source="active_message",
            use_message_head=True,
            message_head_type="mlp",
            message_dim=64,
            message_head_dropout=0.0,
            coordinator_family="candidate_query_cross_attention",
            use_private_cue_aux=False,
            aux_loss_weight=0.0,
            active_message_layers=(-1,),
            active_message_readout_type="active_query",
            active_message_num_queries=1,
            active_message_heads=2,
            active_message_ff_dim=96,
            active_message_topk=8,
            active_message_dropout=0.0,
        )
        values.update(updates)
        return MessageChannelConfig(**values)

    specs = [
        ("active_no_aux_dim64", "No-aux active query baseline.", msg()),
        ("active_no_aux_dim32", "No-aux active query with 32-dim bottleneck.", msg(message_dim=32)),
        ("active_no_aux_dim128", "No-aux active query with 128-dim bottleneck.", msg(message_dim=128)),
        ("active_last2_layers", "Active query over last two layer token states.", msg(active_message_layers=(-2, -1))),
        ("active_multiquery4", "Four learned message queries averaged per clone.", msg(active_message_readout_type="multi_query", active_message_num_queries=4)),
        ("attention_pool_dim64", "Learned attention pooling over tokens.", msg(active_message_readout_type="attention_pool")),
        ("topk_attention_dim64", "Learned top-k token attention readout.", msg(active_message_readout_type="topk_attention")),
        ("topk_attention_k4_no_head", "Top-k token attention with k=4 and no message head.", msg(use_message_head=False, active_message_readout_type="topk_attention", active_message_topk=4)),
        ("topk_attention_k2_no_head", "Top-k token attention with k=2 and no message head.", msg(use_message_head=False, active_message_readout_type="topk_attention", active_message_topk=2)),
        ("multi_query_topk4_no_head", "Four learned queries with per-query top-k token attention and no message head.", msg(use_message_head=False, active_message_readout_type="multi_query_topk", active_message_num_queries=4, active_message_topk=4)),
        ("residual_active_dim64", "Residual active readout plus pooled state.", msg(residual_message_readout=True)),
        ("active_no_head", "Active query sent directly to coordinator without message head.", msg(use_message_head=False)),
        ("attention_pool_no_head", "Learned attention pooling sent directly to coordinator without message head.", msg(use_message_head=False, active_message_readout_type="attention_pool")),
        ("topk_attention_no_head", "Top-k token attention sent directly to coordinator without message head.", msg(use_message_head=False, active_message_readout_type="topk_attention")),
        ("residual_active_no_head", "Residual active plus pooled state without message head.", msg(use_message_head=False, residual_message_readout=True)),
        ("linear_head_dim64", "LayerNorm plus linear message head.", msg(message_head_type="linear")),
        ("gated_head_dim64", "Gated MLP message head.", msg(message_head_type="gated_mlp")),
        ("residual_head_dim64", "Residual MLP message head.", msg(message_head_type="residual_mlp")),
        ("dropout005_head", "No-aux active query with 0.05 dropout.", msg(message_head_dropout=0.05, active_message_dropout=0.05)),
        ("variance_reg", "No-aux active query with anti-collapse variance regularizer.", msg(message_variance_regularizer_weight=0.02, message_variance_regularizer_target=0.05)),
        ("tiny_aux005", "Very small private-cue auxiliary loss.", msg(use_private_cue_aux=True, aux_loss_weight=0.005, aux_warmup_epochs=1, aux_decay_epochs=2)),
        ("aux005_no_head", "Very small private-cue auxiliary loss without message head.", msg(use_message_head=False, use_private_cue_aux=True, aux_loss_weight=0.005, aux_warmup_epochs=1, aux_decay_epochs=2)),
        ("aux05_mlp", "Original-strength auxiliary loss with active readout and MLP head.", msg(use_private_cue_aux=True, aux_loss_weight=0.05, aux_warmup_epochs=2, aux_decay_epochs=4)),
        ("aux05_no_head", "Original-strength auxiliary loss without message head.", msg(use_message_head=False, use_private_cue_aux=True, aux_loss_weight=0.05, aux_warmup_epochs=2, aux_decay_epochs=4)),
        ("candidate_query_2layer", "Two-layer candidate-query cross-attention coordinator.", msg()),
        ("candidate_token_cross_attention", "Candidate queries attend directly over role token states.", msg(readout_source="pooled", coordinator_family="candidate_token_cross_attention")),
        ("bilinear_candidate", "Bilinear candidate-message scoring coordinator.", msg(coordinator_family="bilinear_candidate")),
        ("contrastive_candidate", "Contrastive normalized candidate-message scoring.", msg(coordinator_family="contrastive_candidate")),
        ("global_then_candidate", "Global latent then candidate scoring.", msg(coordinator_family="global_then_candidate")),
        ("two_round_message_passing", "Two-round messages via global latent then candidate scoring.", msg(coordinator_family="two_round_message_passing")),
    ]
    out = []
    for name, description, message_config in specs:
        coordinator = base_coord
        if name == "candidate_query_2layer":
            coordinator = replace(base_coord, num_layers=2)
        if message_config.coordinator_family in {"candidate_token_cross_attention", "bilinear_candidate", "contrastive_candidate", "global_then_candidate", "two_round_message_passing"}:
            coordinator = replace(base_coord, family=message_config.coordinator_family)
        out.append(CandidateSpec(name=name, description=description, message_config=message_config, coordinator_config=coordinator))
    return out


def _stage_configs() -> List[StageConfig]:
    return [
        StageConfig("stage1", n_train=128, n_dev=48, n_test=64, seeds=(0,), epochs=4, patience=2, hidden_dim=32, tiny_layers=2, tiny_ff_dim=64),
        StageConfig("stage2", n_train=512, n_dev=160, n_test=256, seeds=(0, 1, 2), epochs=8, patience=4, hidden_dim=48, tiny_layers=2, tiny_ff_dim=96),
        StageConfig("stage3", n_train=2048, n_dev=512, n_test=1024, seeds=tuple(range(10)), epochs=12, patience=5, hidden_dim=64, tiny_layers=3, tiny_ff_dim=128),
    ]


def _dataset_config(stage: StageConfig) -> MultiViewCodePatchDatasetConfig:
    return MultiViewCodePatchDatasetConfig(
        n_train=stage.n_train,
        n_dev=stage.n_dev,
        n_test=stage.n_test,
        num_candidates=8,
        n_views=4,
        source_roots=("src", "tests"),
        max_files=40,
        snippet_radius=2,
        include_private_signal_tokens=False,
        dataset_source="real_program_analysis_import_restoration",
        generator_version=f"real_import_restore_stable_categories_balanced_{stage.name}",
    )


def _agent_config(stage: StageConfig) -> SharedTransformerAgentConfig:
    return SharedTransformerAgentConfig(
        agent_mode="tiny_transformer",
        max_length=128,
        hidden_dim=stage.hidden_dim,
        tiny_vocab_size=4096,
        tiny_layers=stage.tiny_layers,
        tiny_heads=2,
        tiny_ff_dim=stage.tiny_ff_dim,
        adapter_hidden_dim=stage.hidden_dim,
    )


def _coordinator_config(stage: StageConfig, family: str, num_layers: int, dropout: float) -> LatentCoordinatorConfig:
    return LatentCoordinatorConfig(
        family=family,
        input_dim=stage.hidden_dim,
        model_dim=max(32, stage.hidden_dim),
        num_heads=2,
        num_layers=num_layers,
        ff_dim=max(64, stage.tiny_ff_dim),
        dropout=dropout,
    )


def _training_config(stage: StageConfig) -> RealSharedWeightTrainingConfig:
    return RealSharedWeightTrainingConfig(
        epochs=stage.epochs,
        batch_size=stage.batch_size,
        lr=stage.lr,
        weight_decay=0.0001,
        patience=stage.patience,
        gradient_accumulation_steps=stage.gradient_accumulation_steps,
        mixed_precision=stage.mixed_precision,
    )


def _search_space_rows() -> List[Dict[str, object]]:
    return [
        {"family": "message_readout", "values": ["active_query", "attention_pool", "topk_attention", "multi_query", "last1/last2 layer token states", "residual active+pooled"]},
        {"family": "message_head", "values": ["linear", "mlp", "gated_mlp", "residual_mlp", "dims 32/64/128", "dropout 0/0.05"]},
        {"family": "coordinator", "values": ["candidate_query_cross_attention", "bilinear_candidate", "contrastive_candidate", "global_then_candidate", "two_round_message_passing"]},
        {"family": "loss", "values": ["main CE only", "very small private-cue aux", "message variance anti-collapse", "norm regularizer support"]},
    ]


def _candidate_config(candidate: CandidateSpec) -> Dict[str, object]:
    return {
        "description": candidate.description,
        "message_config": asdict(candidate.message_config),
        "coordinator_config": asdict(candidate.coordinator_config),
    }


def _leakage_passes(leakage: Dict[str, object]) -> bool:
    return bool(
        int(leakage.get("train_test_id_overlap", 1)) == 0
        and int(leakage.get("train_test_candidate_hash_overlap", 1)) == 0
        and int(leakage.get("train_test_file_patch_overlap", 1)) == 0
        and int(leakage.get("train_test_problem_family_overlap", 1)) == 0
        and int(leakage.get("train_test_source_import_key_overlap", 1)) == 0
        and int(leakage.get("train_test_target_file_overlap", 1)) == 0
        and bool(leakage.get("candidate_patch_hash_leakage_audit_passes", False))
    )


def _acc(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions.astype(np.int64) == labels.astype(np.int64))) if len(labels) else 0.0


def render_report(results: Dict[str, object]) -> str:
    lines = [
        "# Architecture Search",
        "",
        "This report uses the real program-analysis import-restoration benchmark. Architecture selection is based on dev metrics and dev controls; test metrics are logged after evaluation and are not used to choose candidates.",
        "",
        "## Search Space",
    ]
    for row in results.get("search_space", []):
        lines.append(f"- **{row['family']}**: {', '.join(row['values'])}")
    stages = results.get("stages", {})
    for stage_name, rows in stages.items():
        lines.extend(["", f"## {stage_name.title()} Candidate Table", ""])
        lines.append("| candidate | seed | valid | dev train | dev frozen | dev delta | test train | test frozen | test delta | text | raw | majority | max single-view | full context | random | hidden shuffle | view masked | view shuffled | role shuffle | collapse cosine | reasons |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in rows:
            dev = row["dev_accuracy"]
            test = row["test_accuracy"]
            collapse = row.get("message_collapse", {}).get("message_head_output") or row.get("message_collapse", {}).get("active_message_readout") or {}
            reasons = "; ".join(row.get("rejection_reasons", [])) or "kept"
            max_single = max(float(test.get(f"single_view_text_role_{index}", 0.0)) for index in range(4))
            lines.append(
                "| {candidate} | {seed} | {valid} | {dt:.4f} | {df:.4f} | {dd:.4f} | {tt:.4f} | {tf:.4f} | {td:.4f} | {text:.4f} | {raw:.4f} | {maj:.4f} | {single:.4f} | {full:.4f} | {rand:.4f} | {hidden:.4f} | {masked:.4f} | {shuffled:.4f} | {role:.4f} | {cos:.4f} | {reasons} |".format(
                    candidate=row["candidate"],
                    seed=row["seed"],
                    valid="yes" if row.get("selection_valid") else "no",
                    dt=dev["trainable"],
                    df=dev["frozen"],
                    dd=row["dev_delta"],
                    tt=test["trainable"],
                    tf=test["frozen"],
                    td=row["test_delta"],
                    text=test["text_only"],
                    raw=test["raw_latent"],
                    maj=test.get("majority_baseline", 0.0),
                    single=max_single,
                    full=test.get("single_agent_full_context", 0.0),
                    rand=test["randomized_labels"],
                    hidden=test["hidden_states_shuffled_across_examples"],
                    masked=test["view_masked"],
                    shuffled=test["view_shuffled"],
                    role=test["role_labels_shuffled"],
                    cos=float(collapse.get("mean_cosine_similarity", 0.0)),
                    reasons=reasons,
                )
            )
        valid = [row for row in rows if row.get("selection_valid")]
        valid.sort(key=lambda row: float(row.get("dev_delta", -999.0)), reverse=True)
        lines.extend(["", "Top variants by valid dev delta:"])
        if valid:
            for row in valid[:5]:
                lines.append(f"- `{row['candidate']}`: dev delta {row['dev_delta']:.4f}, test delta {row['test_delta']:.4f}")
        else:
            lines.append("- None passed the locked dev rejection rules.")
        lines.extend(["", "Full controls and diagnostics for kept rows:"])
        kept = valid[:5] if stage_name == "stage1" else valid
        if kept:
            lines.append("| candidate | seed | explicit evidence oracle | evidence diag train/frozen | candidate-order baseline | physical-order shuffled | candidate-order shuffled | active variance | active norm | active cosine | private cue mean | combined-message probe | shared grad | shared delta | coordinator grad | coordinator delta |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for row in kept:
                test = row["test_accuracy"]
                collapse = row.get("message_collapse", {}).get("active_message_readout", {})
                private = row.get("private_cue_probes", {}).get("active_message_readout", {})
                combined = row.get("combined_message_final_label_probe", {}).get("combined_message")
                audit = row.get("trainable_audit", {})
                ev_t = test.get("evidence_token_diagnostic_trainable")
                ev_f = test.get("evidence_token_diagnostic_frozen")
                ev_pair = f"{ev_t:.4f}/{ev_f:.4f}" if ev_t is not None and ev_f is not None else "not_run"
                lines.append(
                    f"| {row['candidate']} | {row['seed']} | {test['explicit_evidence_oracle']:.4f} | {ev_pair} | "
                    f"{test['candidate_order_baseline']:.4f} | {test['physical_order_shuffled_roles_preserved']:.4f} | "
                    f"{test['candidate_order_shuffled']:.4f} | {float(collapse.get('variance_mean', 0.0)):.4f} | "
                    f"{float(collapse.get('norm_mean', 0.0)):.4f} | {float(collapse.get('mean_cosine_similarity', 0.0)):.4f} | "
                    f"{float(private.get('mean', 0.0)):.4f} | {float(combined if combined is not None else 0.0):.4f} | "
                    f"{float(audit.get('agent_grad_norm_mean') or 0.0):.4f} | {float(audit.get('agent_parameter_delta') or 0.0):.4f} | "
                    f"{float(audit.get('coordinator_grad_norm_mean') or 0.0):.4f} | {float(audit.get('coordinator_parameter_delta') or 0.0):.4f} |"
                )
            lines.extend(["", "Per-family accuracy for kept rows:"])
            for row in kept:
                family = ", ".join(f"{name}={value:.4f}" for name, value in row.get("per_family_accuracy", {}).items())
                lines.append(f"- `{row['candidate']}` seed {row['seed']}: {family}")
            lines.extend(["", "Per-role ablation for kept rows:"])
            for row in kept:
                ablation = row.get("per_role_ablation", {})
                drops = ", ".join(f"role {role}: drop {drop:.4f}" for role, drop in ablation.get("accuracy_drop", {}).items())
                lines.append(f"- `{row['candidate']}` seed {row['seed']}: base {float(ablation.get('base_accuracy', 0.0)):.4f}; {drops}")
        else:
            lines.append("- No kept rows in this stage.")
    lines.extend(["", "## Robustness Results", ""])
    lines.append("Stage 1 is a fast screen. Stage 2 and Stage 3 rows appear here only after the corresponding run is executed. No hard-validation success claim is made from Stage 1 alone.")
    stage2 = list(stages.get("stage2", [])) if isinstance(stages, dict) else []
    if stage2:
        by_candidate: Dict[str, List[Dict[str, object]]] = {}
        for row in stage2:
            by_candidate.setdefault(str(row["candidate"]), []).append(row)
        for candidate, candidate_rows in sorted(by_candidate.items()):
            deltas = [float(row["test_delta"]) for row in candidate_rows]
            trainable = [float(row["test_accuracy"]["trainable"]) for row in candidate_rows]
            frozen = [float(row["test_accuracy"]["frozen"]) for row in candidate_rows]
            ci_low, ci_high = _bootstrap_ci(deltas)
            lines.append(
                f"- Stage 2 `{candidate}`: mean trainable {mean(trainable):.4f}, mean frozen {mean(frozen):.4f}, "
                f"mean delta {mean(deltas):.4f}, std delta {_std(deltas):.4f}, bootstrap 95% CI [{ci_low:.4f}, {ci_high:.4f}], seeds {len(candidate_rows)}."
            )
            lines.append("  Per-seed deltas: " + ", ".join(f"seed {row['seed']}={row['test_delta']:.4f}" for row in candidate_rows))
    else:
        lines.append("- Stage 2 was not completed.")
    stage3 = list(stages.get("stage3", [])) if isinstance(stages, dict) else []
    if stage3:
        lines.append(f"- Stage 3 completed with {len(stage3)} rows.")
    else:
        lines.append("- Stage 3 hard validation was not run. Required hard-validation budget remains: n_train >= 2048, n_dev >= 512, n_test >= 1024, seeds >= 10.")
    lines.extend(
        [
            "",
            "## Conservative Interpretation",
            "",
            "The Stage 2 top-k no-head variant is a promising medium-validation result if present above, because it compares against an exact frozen same-architecture comparator on real program-analysis patch selection and keeps the locked controls near chance. It is not a hard-validation success claim until Stage 3 is run.",
            "",
            "No result should be interpreted from train accuracy. A candidate is rejected when trainable does not beat the exact frozen comparator, when text/raw baselines are not beaten, when locked controls are above threshold, when collapse statistics show near-zero variance with cosine near one, or when leakage checks fail. Do not claim open-ended agent collaboration or general LLM self-organization from this benchmark.",
        ]
    )
    return "\n".join(lines) + "\n"


def _std(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _bootstrap_ci(values: Sequence[float], samples: int = 2000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(17_331)
    arr = np.asarray(values, dtype=np.float64)
    means = []
    for _ in range(samples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


if __name__ == "__main__":
    main()
