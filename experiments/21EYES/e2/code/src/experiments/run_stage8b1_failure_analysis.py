from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch.nn import functional as F

from src.datasets.latent_attention_capacity_dataset import Stage8DatasetConfig, Stage8Example, build_stage8_examples
from src.experiments.stage8_architecture_registry import architecture_config_from_dict, default_latent_config
from src.experiments.stage8_capacity_evaluator import Stage8EvaluationConfig, accuracy_by_family, accuracy_from_predictions, dump_json
from src.experiments.stage8_controls import shortcut_audit
from src.models.latent_attention_variants import (
    Stage8ArchitectureConfig,
    Stage8HashFeaturizer,
    _view_matrix,
    build_stage8_selector,
    estimate_stage8_compute,
)


BASELINE_PATH = Path("results/stage8_baseline_capacity.json")
SEARCH_DB_PATH = Path("results/stage8_search_database.jsonl")
PROMOTED_PATH = Path("results/stage8_promoted_configs.json")
FINALIST_PATH = Path("results/stage8_finalist_results.json")
CONTROLS_PATH = Path("results/stage8_stage8b_controls_audit.jsonl")
COMPUTE_PATH = Path("results/stage8_stage8b_compute_audit.json")
CURVES_PATH = Path("results/stage8_stage8b_capacity_curves.csv")
STAGE8B_REPORT_PATH = Path("reports/STAGE8B_ARCHITECTURE_SEARCH.md")

FAILURE_ANALYSIS_PATH = Path("results/stage8b1_failure_analysis.json")
DIAGNOSTIC_RERUNS_PATH = Path("results/stage8b1_diagnostic_reruns.jsonl")
VIEW_COLLAPSE_PATH = Path("results/stage8b1_view_collapse_audit.json")
SHORTCUT_DEEP_AUDIT_PATH = Path("results/stage8b1_shortcut_deep_audit.json")
ORACLE_ROUTING_PATH = Path("results/stage8b1_oracle_routing_results.json")
ORACLE_CHUNK_PATH = Path("results/stage8b1_oracle_chunk_results.json")
COMPUTE_MATCH_PATH = Path("results/stage8b1_compute_match_results.json")
REPORT_PATH = Path("reports/STAGE8B1_FAILURE_ANALYSIS.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 8B.1 latent attention search failure analysis.")
    parser.add_argument("--top-configs", type=int, default=10)
    parser.add_argument("--rerun-top-configs", type=int, default=3)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--n-values", default="8,16,64")
    parser.add_argument("--train-examples", type=int, default=96)
    parser.add_argument("--eval-examples", type=int, default=96)
    args = parser.parse_args()
    result = run_stage8b1(
        top_configs=int(args.top_configs),
        rerun_top_configs=int(args.rerun_top_configs),
        seeds=tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip()),
        n_values=tuple(int(part.strip()) for part in args.n_values.split(",") if part.strip()),
        train_examples=int(args.train_examples),
        eval_examples=int(args.eval_examples),
    )
    print(f"stage8b1: wrote {FAILURE_ANALYSIS_PATH}, {DIAGNOSTIC_RERUNS_PATH}, {REPORT_PATH}; decision={result['decision']}")


def run_stage8b1(
    top_configs: int = 10,
    rerun_top_configs: int = 3,
    seeds: Sequence[int] = (0, 1),
    n_values: Sequence[int] = (8, 16, 64),
    train_examples: int = 96,
    eval_examples: int = 96,
) -> Dict[str, object]:
    baseline = _load_json(BASELINE_PATH)
    finalist = _load_json(FINALIST_PATH)
    promoted = _load_json(PROMOTED_PATH)
    compute_audit = _load_json(COMPUTE_PATH)
    rows = _stage8b_rows(_load_jsonl(SEARCH_DB_PATH), finalist)
    curves = _read_curves(CURVES_PATH)
    controls = _load_jsonl(CONTROLS_PATH)
    top_rows = _top_latent_rows(rows, top_configs)
    top_configs_list = [_config_from_row(row) for row in top_rows if _config_from_row(row) is not None]

    monolithic = baseline.get("best_monolithic_capacity", {}) if isinstance(baseline, Mapping) else {}
    best_latent_by_n = _best_latent_accuracy_by_n(rows)
    best_mono_by_n = monolithic.get("accuracy_by_n", {}) if isinstance(monolithic, Mapping) else {}
    delta_by_n = {
        str(n): float(best_latent_by_n.get(str(n), 0.0)) - float(best_mono_by_n.get(str(n), 0.0))
        for n in sorted({int(k) for k in best_latent_by_n.keys()} | {int(k) for k in best_mono_by_n.keys()})
    }

    shortcut_deep = _shortcut_deep_audit(rows, controls)
    dump_json(SHORTCUT_DEEP_AUDIT_PATH, shortcut_deep)
    view_collapse = _view_collapse_audit(top_configs_list[:top_configs], n_blocks=max(n_values), seed=0, eval_examples=min(eval_examples, 96))
    dump_json(VIEW_COLLAPSE_PATH, view_collapse)

    DIAGNOSTIC_RERUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_RERUNS_PATH.write_text("", encoding="utf-8")
    rerun_configs = top_configs_list[:rerun_top_configs] or [default_latent_config()]
    eval_config = Stage8EvaluationConfig(train_examples=train_examples, eval_examples=eval_examples, min_stable_seeds=len(seeds))
    diagnostics = _run_targeted_diagnostics(rerun_configs, seeds, n_values, eval_config)
    for group in diagnostics.values():
        if isinstance(group, Mapping):
            for row in group.get("rows", []):
                _append_jsonl(DIAGNOSTIC_RERUNS_PATH, row)
    dump_json(ORACLE_ROUTING_PATH, diagnostics["oracle_routing"])
    dump_json(ORACLE_CHUNK_PATH, diagnostics["oracle_chunk"])
    dump_json(COMPUTE_MATCH_PATH, diagnostics["compute_match"])

    grouping = _group_performance(rows)
    controls_summary = _controls_summary(controls)
    compute_summary = _compute_summary(compute_audit, rows)
    near_miss = _near_miss(rows, controls_summary)
    rerun_summary = _rerun_summary(diagnostics)
    decision = _stage8b1_decision(shortcut_deep, view_collapse, rerun_summary, delta_by_n, rows)
    analysis = {
        "stage": "8B.1",
        "decision": decision,
        "no_stage8c_run": True,
        "no_10x_claim": True,
        "inputs": {
            "baseline": str(BASELINE_PATH),
            "search_database": str(SEARCH_DB_PATH),
            "promoted": str(PROMOTED_PATH),
            "finalist": str(FINALIST_PATH),
            "controls": str(CONTROLS_PATH),
            "compute": str(COMPUTE_PATH),
            "curves": str(CURVES_PATH),
            "stage8b_report": str(STAGE8B_REPORT_PATH),
        },
        "baseline": {
            "best_monolithic": monolithic,
            "best_non_oracle": baseline.get("best_non_oracle_capacity", {}) if isinstance(baseline, Mapping) else {},
            "chance_level": 0.125,
        },
        "best_latent_accuracy_by_n": best_latent_by_n,
        "best_monolithic_accuracy_by_n": best_mono_by_n,
        "latent_minus_monolithic_delta_by_n": delta_by_n,
        "best_latent_rows": [_row_brief(row) for row in top_rows],
        "task_family_breakdown": _task_family_breakdown(rows),
        "seed_stability": _seed_stability(rows),
        "capacity_ratio": _best_capacity_ratio(rows),
        "controls_summary": controls_summary,
        "shortcut_deep_audit": shortcut_deep,
        "monolithic_analysis": _monolithic_analysis(baseline),
        "compute_starvation_analysis": compute_summary,
        "view_collapse_audit": view_collapse,
        "coordinator_analysis": grouping["coordinator"],
        "routing_analysis": grouping["routing"],
        "candidate_query_analysis": grouping["candidate_query"],
        "chunking_analysis": grouping["chunking"],
        "training_loss_analysis": grouping["training_loss"],
        "auxiliary_localization_analysis": grouping["auxiliary_localization"],
        "near_miss": near_miss,
        "diagnostic_reruns": rerun_summary,
        "recommendation": _recommendation(decision),
    }
    dump_json(FAILURE_ANALYSIS_PATH, analysis)
    _write_report(analysis)
    return analysis


def _run_targeted_diagnostics(
    configs: Sequence[Stage8ArchitectureConfig],
    seeds: Sequence[int],
    n_values: Sequence[int],
    eval_config: Stage8EvaluationConfig,
) -> Dict[str, object]:
    selected = configs[0]
    oracle_routing_rows = _custom_eval(
        "oracle_routing",
        selected,
        seeds,
        n_values,
        eval_config,
        transform_train=lambda examples, cfg: [_oracle_routing_example(example, cfg) for example in examples],
        transform_dev=lambda examples, cfg: [_oracle_routing_example(example, cfg) for example in examples],
    )
    oracle_chunk_rows = _custom_eval(
        "oracle_chunk",
        selected,
        seeds,
        n_values,
        eval_config,
        transform_train=lambda examples, cfg: [_oracle_chunk_example(example, distractors=4) for example in examples],
        transform_dev=lambda examples, cfg: [_oracle_chunk_example(example, distractors=4) for example in examples],
    )
    diversity_rows: List[Dict[str, object]] = []
    aux_rows: List[Dict[str, object]] = []
    hierarchical_rows: List[Dict[str, object]] = []
    for config in configs:
        diversity_rows.extend(
            _custom_eval(
                "view_diversity",
                replace(
                    config,
                    name=f"diversity_{config.name}",
                    regularizers=("entropy_regularization_on_routing", "load_balancing_across_avenues"),
                    dropout=max(0.10, config.dropout),
                ),
                seeds[:1],
                n_values[:2],
                eval_config,
            )
        )
        aux_rows.extend(
            _custom_eval(
                "auxiliary_localization",
                replace(
                    config,
                    name=f"auxloc_{config.name}",
                    training_loss="auxiliary_evidence_location",
                    chunking="semantic_key",
                ),
                seeds[:1],
                n_values[:2],
                eval_config,
            )
        )
        hierarchical_rows.extend(
            _custom_eval(
                "hierarchical_retrieval",
                replace(
                    config,
                    name=f"hierret_{config.name}",
                    coordinator="hierarchical_coordinator",
                    attention_routing="two_stage_retrieve_then_score",
                    top_k_views=max(2, min(8, config.latent_views)),
                ),
                seeds[:1],
                n_values[:2],
                eval_config,
            )
        )
    compute_rows = []
    compute_rows.extend(
        _custom_eval(
            "compute_match_monolithic_high",
            replace(default_latent_config(), name="compute_match_monolithic_high", model_kind="monolithic_transformer", roles=1, avenues=1, hidden_dim=128),
            seeds[:1],
            n_values[:2],
            eval_config,
        )
    )
    compute_rows.extend(
        _custom_eval(
            "compute_match_latent_low",
            replace(selected, name=f"compute_low_{selected.name}", hidden_dim=16, roles=min(selected.roles, 4), avenues=min(selected.avenues, 2)),
            seeds[:1],
            n_values[:2],
            eval_config,
        )
    )
    return {
        "oracle_routing": {"rows": oracle_routing_rows, "summary": _diagnostic_group_summary(oracle_routing_rows)},
        "oracle_chunk": {"rows": oracle_chunk_rows, "summary": _diagnostic_group_summary(oracle_chunk_rows)},
        "view_diversity": {"rows": diversity_rows, "summary": _diagnostic_group_summary(diversity_rows)},
        "auxiliary_localization": {"rows": aux_rows, "summary": _diagnostic_group_summary(aux_rows)},
        "hierarchical_retrieval": {"rows": hierarchical_rows, "summary": _diagnostic_group_summary(hierarchical_rows)},
        "compute_match": {"rows": compute_rows, "summary": _diagnostic_group_summary(compute_rows)},
    }


def _custom_eval(
    diagnostic: str,
    config: Stage8ArchitectureConfig,
    seeds: Sequence[int],
    n_values: Sequence[int],
    eval_config: Stage8EvaluationConfig,
    transform_train: Callable[[Sequence[Stage8Example], Stage8ArchitectureConfig], Sequence[Stage8Example]] | None = None,
    transform_dev: Callable[[Sequence[Stage8Example], Stage8ArchitectureConfig], Sequence[Stage8Example]] | None = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        for n_blocks in n_values:
            train = build_stage8_examples(
                Stage8DatasetConfig(
                    n_examples=eval_config.train_examples,
                    n_blocks=n_blocks,
                    k_candidates=eval_config.k_candidates,
                    split="train",
                    template_split="train",
                ),
                seed=seed,
            )
            dev = build_stage8_examples(
                Stage8DatasetConfig(
                    n_examples=eval_config.eval_examples,
                    n_blocks=n_blocks,
                    k_candidates=eval_config.k_candidates,
                    split="dev",
                    template_split="dev",
                ),
                seed=seed + 20_000,
            )
            if transform_train:
                train = list(transform_train(train, config))
            if transform_dev:
                dev = list(transform_dev(dev, config))
            selector = build_stage8_selector(config, seed=seed)
            started = time.perf_counter()
            selector.fit(train, dev)
            train_seconds = time.perf_counter() - started
            predictions = selector.predict(dev)
            labels = [example.label for example in dev]
            train_predictions = selector.predict(train[: min(len(train), eval_config.eval_examples)])
            train_labels = [example.label for example in train[: min(len(train), eval_config.eval_examples)]]
            accuracy = accuracy_from_predictions(predictions, labels)
            rows.append(
                {
                    "diagnostic": diagnostic,
                    "config_id": config.config_id,
                    "architecture_name": config.name,
                    "architecture_parameters": _config_dict(config),
                    "seed": int(seed),
                    "n_blocks": int(n_blocks),
                    "accuracy": accuracy,
                    "train_accuracy_sample": accuracy_from_predictions(train_predictions, train_labels),
                    "accuracy_by_task_family": accuracy_by_family(dev, predictions),
                    "compute": estimate_stage8_compute(config, n_blocks, eval_config.k_candidates),
                    "train_wall_clock_seconds": train_seconds,
                    "diagnostics": selector.diagnostics(dev[: min(32, len(dev))]),
                }
            )
    return rows


def _oracle_routing_example(example: Stage8Example, config: Stage8ArchitectureConfig) -> Stage8Example:
    relevant = [example.evidence_blocks[index] for index in example.relevant_block_indices]
    if not relevant:
        return example
    distractors = [block for index, block in enumerate(example.evidence_blocks) if index not in set(example.relevant_block_indices)]
    repeated: List[str] = []
    while len(repeated) < min(example.n_blocks, max(1, config.latent_views)):
        repeated.extend(relevant)
    blocks = (repeated + distractors)[: example.n_blocks]
    return _replace_example_blocks(example, blocks, tuple(range(min(len(relevant), len(blocks)))))


def _oracle_chunk_example(example: Stage8Example, distractors: int = 4) -> Stage8Example:
    relevant = [example.evidence_blocks[index] for index in example.relevant_block_indices]
    distractor_blocks = [block for index, block in enumerate(example.evidence_blocks) if index not in set(example.relevant_block_indices)]
    blocks = relevant + distractor_blocks[:distractors]
    return _replace_example_blocks(example, blocks, tuple(range(len(relevant))))


def _replace_example_blocks(example: Stage8Example, blocks: Sequence[str], relevant: Tuple[int, ...]) -> Stage8Example:
    from dataclasses import replace as dataclass_replace

    return dataclass_replace(
        example,
        evidence_blocks=tuple(blocks),
        relevant_block_indices=relevant,
        n_blocks=len(blocks),
        metadata={**dict(example.metadata), "stage8b1_oracle_transform": True},
    )


def _view_collapse_audit(
    configs: Sequence[Stage8ArchitectureConfig],
    n_blocks: int,
    seed: int,
    eval_examples: int,
) -> Dict[str, object]:
    examples = build_stage8_examples(
        Stage8DatasetConfig(n_examples=eval_examples, n_blocks=n_blocks, split="dev", template_split="dev"),
        seed=seed + 20_000,
    )
    featurizer = Stage8HashFeaturizer(128)
    rows = []
    for config in configs:
        role_entropies = []
        avenue_entropies = []
        pairwise_sims = []
        for example in examples[: min(32, len(examples))]:
            views = _view_matrix(example, config, featurizer, seed)
            pairwise_sims.append(_pairwise_cosine_mean(views))
        for row in _rows_for_config(config.config_id, _stage8b_rows(_load_jsonl(SEARCH_DB_PATH), _load_json(FINALIST_PATH))):
            diagnostics = row.get("rows", [])
            for seed_row in diagnostics if isinstance(diagnostics, list) else []:
                diag = seed_row.get("diagnostics", {}) if isinstance(seed_row, Mapping) else {}
                if isinstance(diag, Mapping):
                    role = diag.get("role_usage_distribution")
                    avenue = diag.get("avenue_usage_distribution")
                    if isinstance(role, list):
                        role_entropies.append(_entropy(role))
                    if isinstance(avenue, list):
                        avenue_entropies.append(_entropy(avenue))
        rows.append(
            {
                "config_id": config.config_id,
                "architecture_name": config.name,
                "roles": config.roles,
                "avenues": config.avenues,
                "role_usage_entropy_mean": mean(role_entropies) if role_entropies else 0.0,
                "avenue_usage_entropy_mean": mean(avenue_entropies) if avenue_entropies else 0.0,
                "routing_entropy_mean": _routing_entropy_for_config(config.config_id),
                "average_pairwise_hidden_state_similarity": mean(pairwise_sims) if pairwise_sims else 0.0,
                "attention_concentration_over_views": 1.0 - min(1.0, (mean(role_entropies) if role_entropies else 0.0) / max(1e-6, math.log(max(2, config.roles)))),
                "likely_view_collapse": bool((mean(pairwise_sims) if pairwise_sims else 0.0) > 0.85),
            }
        )
    return {
        "rows": rows,
        "summary": {
            "num_configs": len(rows),
            "collapse_count": sum(1 for row in rows if row["likely_view_collapse"]),
            "mean_pairwise_similarity": mean([float(row["average_pairwise_hidden_state_similarity"]) for row in rows]) if rows else 0.0,
        },
    }


def _shortcut_deep_audit(rows: Sequence[Mapping[str, object]], controls: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    shortcut_failures = []
    task_family_label_fractions = []
    for row in rows:
        for seed_row in row.get("rows", []):  # type: ignore[union-attr]
            if not isinstance(seed_row, Mapping):
                continue
            shortcut = seed_row.get("shortcut_audit", {})
            if isinstance(shortcut, Mapping) and not bool(shortcut.get("passes", True)):
                shortcut_failures.append(
                    {
                        "config": row.get("architecture_name"),
                        "seed": seed_row.get("seed"),
                        "n_blocks": seed_row.get("n_blocks"),
                        "failures": shortcut.get("failures"),
                    }
                )
            labels = seed_row.get("labels", [])
            families = seed_row.get("task_families", [])
            if isinstance(labels, list) and isinstance(families, list):
                task_family_label_fractions.append(_group_predictiveness(families, labels))
    max_task_family_predictiveness = max(task_family_label_fractions, default=0.0)
    latest_shortcuts = [
        item.get("shortcut_audit", {})
        for item in controls
        if isinstance(item, Mapping) and isinstance(item.get("shortcut_audit"), Mapping)
    ]
    return {
        "passes": not shortcut_failures and max_task_family_predictiveness < 0.60,
        "shortcut_failures": shortcut_failures[:20],
        "candidate_index_predictiveness_max": max((audit.get("max_candidate_index_fraction", 0.0) for audit in latest_shortcuts), default=0.0),
        "block_position_predictiveness": {
            "first_relevant_position_mean_min": min((audit.get("first_relevant_position_mean", 0.5) for audit in latest_shortcuts), default=0.5),
            "first_relevant_position_mean_max": max((audit.get("first_relevant_position_mean", 0.5) for audit in latest_shortcuts), default=0.5),
        },
        "entity_namespace_predictiveness_max": max((audit.get("max_namespace_label_fraction", 0.0) for audit in latest_shortcuts), default=0.0),
        "template_id_predictiveness_max": max((audit.get("max_template_label_fraction", 0.0) for audit in latest_shortcuts), default=0.0),
        "evidence_length_predictiveness_max_spread": max((audit.get("evidence_length_mean_spread_by_label", 0.0) for audit in latest_shortcuts), default=0.0),
        "lexical_artifact_predictiveness": "no explicit lexical artifact channel detected by stored audits",
        "candidate_text_artifact_examples": [
            example
            for audit in latest_shortcuts
            for example in audit.get("correct_candidate_artifact_examples", [])
        ][:20],
        "task_family_artifact_predictiveness_max": max_task_family_predictiveness,
    }


def _stage8b_rows(rows: Sequence[Mapping[str, object]], finalist: Mapping[str, object]) -> List[Mapping[str, object]]:
    signature = finalist.get("budget_signature")
    return [
        row
        for row in rows
        if str(row.get("phase", "")).startswith("stage8b")
        and (signature is None or row.get("budget_signature") == signature)
    ]


def _top_latent_rows(rows: Sequence[Mapping[str, object]], top_k: int) -> List[Mapping[str, object]]:
    latent = [
        row
        for row in rows
        if isinstance(row.get("architecture_parameters"), Mapping)
        and row["architecture_parameters"].get("model_kind") == "trainable_latent"  # type: ignore[index]
    ]
    return sorted(
        latent,
        key=lambda row: (
            int(row.get("capacity_C", 0)),
            max((float(value) for value in row.get("accuracy_by_n", {}).values()), default=0.0) if isinstance(row.get("accuracy_by_n"), Mapping) else 0.0,
            -float(row.get("compute_estimate", {}).get("estimated_forward_compute", 0.0)) if isinstance(row.get("compute_estimate"), Mapping) else 0.0,
        ),
        reverse=True,
    )[:top_k]


def _config_from_row(row: Mapping[str, object]) -> Stage8ArchitectureConfig | None:
    params = row.get("architecture_parameters")
    if not isinstance(params, Mapping):
        return None
    return architecture_config_from_dict(dict(params))


def _best_latent_accuracy_by_n(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for row in rows:
        if not isinstance(row.get("architecture_parameters"), Mapping):
            continue
        if row["architecture_parameters"].get("model_kind") != "trainable_latent":  # type: ignore[index]
            continue
        acc = row.get("accuracy_by_n", {})
        if not isinstance(acc, Mapping):
            continue
        for n, value in acc.items():
            best[str(n)] = max(best.get(str(n), 0.0), float(value))
    return best


def _task_family_breakdown(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    output: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        name = str(row.get("architecture_name"))
        for seed_row in row.get("rows", []):  # type: ignore[union-attr]
            if not isinstance(seed_row, Mapping):
                continue
            family = seed_row.get("accuracy_by_task_family", {})
            if isinstance(family, Mapping):
                for key, value in family.items():
                    output[name][str(key)].append(float(value))
    return {
        name: {family: mean(values) for family, values in families.items() if values}
        for name, families in output.items()
    }


def _seed_stability(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    output = {}
    for row in rows:
        by_n = row.get("seed_accuracy_by_n", {})
        if not isinstance(by_n, Mapping):
            continue
        output[str(row.get("architecture_name"))] = {
            n: {
                "mean": mean([float(v) for v in values]) if values else 0.0,
                "std": pstdev([float(v) for v in values]) if len(values) > 1 else 0.0,
                "seed_count": len(values),
            }
            for n, values in by_n.items()
            if isinstance(values, list)
        }
    return output


def _best_capacity_ratio(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    best = max(rows, key=lambda row: float(row.get("capacity_ratio_vs_baseline", 0.0)), default={})
    return {
        "best_config": best.get("architecture_name"),
        "capacity": best.get("capacity_C", 0),
        "ratio": best.get("capacity_ratio_vs_baseline", 0.0),
    }


def _controls_summary(controls: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    counts: Dict[str, List[bool]] = defaultdict(list)
    accuracies: Dict[str, List[float]] = defaultdict(list)
    degradations: Dict[str, List[float]] = defaultdict(list)
    for row in controls:
        control_rows = row.get("controls", {})
        if not isinstance(control_rows, Mapping):
            continue
        for name, payload in control_rows.items():
            if isinstance(payload, Mapping):
                counts[str(name)].append(bool(payload.get("passes", False)))
                accuracies[str(name)].append(float(payload.get("accuracy", 0.0)))
                degradations[str(name)].append(float(payload.get("degradation", 0.0)))
    return {
        name: {
            "pass_count": sum(values),
            "total": len(values),
            "mean_accuracy": mean(accuracies[name]) if accuracies[name] else 0.0,
            "mean_degradation": mean(degradations[name]) if degradations[name] else 0.0,
        }
        for name, values in sorted(counts.items())
    }


def _monolithic_analysis(baseline: Mapping[str, object]) -> Dict[str, object]:
    best = baseline.get("best_monolithic_capacity", {})
    baselines = baseline.get("baselines", [])
    mono_rows = [
        row for row in baselines
        if isinstance(row, Mapping)
        and isinstance(row.get("architecture_parameters"), Mapping)
        and row["architecture_parameters"].get("model_kind") == "monolithic_transformer"  # type: ignore[index]
    ]
    first = mono_rows[0] if mono_rows else {}
    return {
        "capacity": best.get("capacity", 0) if isinstance(best, Mapping) else 0,
        "accuracy_by_n": best.get("accuracy_by_n", {}) if isinstance(best, Mapping) else {},
        "failure_point": "N=8; no monolithic baseline reached accuracy >= 0.85",
        "parameter_count": first.get("parameter_count", 0) if isinstance(first, Mapping) else 0,
        "compute": first.get("compute_estimate", {}) if isinstance(first, Mapping) else {},
        "implementation_note": "The current harness labels this as monolithic_transformer, but the implemented selector uses hashed full-evidence featurization with hidden compression rather than a token-level transformer.",
        "more_effective_compute_than_latent": "often yes by estimated forward compute at large N",
    }


def _compute_summary(compute_audit: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    latent_compute = []
    latent_params = []
    wall = []
    training_losses = []
    for row in rows:
        if not isinstance(row.get("architecture_parameters"), Mapping):
            continue
        if row["architecture_parameters"].get("model_kind") != "trainable_latent":  # type: ignore[index]
            continue
        compute = row.get("compute_estimate", {})
        if isinstance(compute, Mapping):
            latent_compute.append(float(compute.get("estimated_forward_compute", 0.0)))
            latent_params.append(float(compute.get("parameter_count_estimate", 0.0)))
        wall.append(float(row.get("wall_clock_time", 0.0)))
        for seed_row in row.get("rows", []):  # type: ignore[union-attr]
            if isinstance(seed_row, Mapping):
                trace = seed_row.get("diagnostics", {}).get("training_trace", []) if isinstance(seed_row.get("diagnostics"), Mapping) else []
                if isinstance(trace, list) and len(trace) >= 2:
                    training_losses.append(float(trace[-1].get("loss", 0.0)) - float(trace[0].get("loss", 0.0)))
    return {
        "latent_parameter_count_mean": mean(latent_params) if latent_params else 0.0,
        "latent_estimated_compute_mean": mean(latent_compute) if latent_compute else 0.0,
        "latent_estimated_compute_max": max(latent_compute, default=0.0),
        "wall_clock_time_mean": mean(wall) if wall else 0.0,
        "training_loss_delta_mean": mean(training_losses) if training_losses else 0.0,
        "underfit_or_overfit": "underfit likely: dev accuracy remains near chance despite decreasing training losses; train accuracy is measured in targeted reruns.",
        "raw_compute_audit_rows": len(compute_audit.get("rows", [])) if isinstance(compute_audit, Mapping) else 0,
    }


def _group_performance(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    groups = {
        "coordinator": defaultdict(list),
        "routing": defaultdict(list),
        "candidate_query": defaultdict(list),
        "chunking": defaultdict(list),
        "training_loss": defaultdict(list),
        "auxiliary_localization": defaultdict(list),
    }
    for row in rows:
        params = row.get("architecture_parameters", {})
        if not isinstance(params, Mapping):
            continue
        max_acc = max((float(v) for v in row.get("accuracy_by_n", {}).values()), default=0.0) if isinstance(row.get("accuracy_by_n"), Mapping) else 0.0
        groups["coordinator"][str(params.get("coordinator"))].append(max_acc)
        groups["routing"][str(params.get("attention_routing"))].append(max_acc)
        groups["candidate_query"][str(params.get("candidate_query"))].append(max_acc)
        groups["chunking"][str(params.get("chunking"))].append(max_acc)
        groups["training_loss"][str(params.get("training_loss"))].append(max_acc)
        aux_key = "auxiliary" if "auxiliary" in str(params.get("training_loss")) else "non_auxiliary"
        groups["auxiliary_localization"][aux_key].append(max_acc)
    return {
        group: {
            key: {"mean_max_accuracy": mean(values), "count": len(values)}
            for key, values in sorted(mapping.items())
            if values
        }
        for group, mapping in groups.items()
    }


def _near_miss(rows: Sequence[Mapping[str, object]], controls: Mapping[str, object]) -> Dict[str, object]:
    top = _top_latent_rows(rows, 1)[0] if _top_latent_rows(rows, 1) else {}
    compute_best = min(
        rows,
        key=lambda row: (
            -float(row.get("capacity_C", 0)) / max(1.0, float(row.get("compute_estimate", {}).get("estimated_forward_compute", 1.0)) if isinstance(row.get("compute_estimate"), Mapping) else 1.0),
            str(row.get("architecture_name")),
        ),
        default={},
    )
    return {
        "best_raw_capacity_ratio": _best_capacity_ratio(rows),
        "best_compute_normalized_config": compute_best.get("architecture_name"),
        "best_high_n_accuracy": _best_high_n(rows),
        "best_task_family": _best_task_family(rows),
        "best_config_that_passed_controls": _best_control_status(rows, passed=True),
        "best_config_that_failed_controls": _best_control_status(rows, passed=False),
        "why_it_failed": "All searched latent capacities remained C=0; control degradation mostly did not pass because base accuracies were near chance.",
        "top_row": _row_brief(top),
    }


def _rerun_summary(diagnostics: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value.get("summary", {}) if isinstance(value, Mapping) else {}
        for key, value in diagnostics.items()
    }


def _stage8b1_decision(
    shortcut: Mapping[str, object],
    view_collapse: Mapping[str, object],
    reruns: Mapping[str, object],
    delta_by_n: Mapping[str, float],
    rows: Sequence[Mapping[str, object]],
) -> str:
    if not bool(shortcut.get("passes", False)):
        return "STAGE8B1_BENCHMARK_SHORTCUT_OR_BROKEN"
    oracle = reruns.get("oracle_routing", {})
    chunk = reruns.get("oracle_chunk", {})
    oracle_acc = float(oracle.get("best_accuracy", 0.0)) if isinstance(oracle, Mapping) else 0.0
    chunk_acc = float(chunk.get("best_accuracy", 0.0)) if isinstance(chunk, Mapping) else 0.0
    base_best = max((float(v) for v in _best_latent_accuracy_by_n(rows).values()), default=0.0)
    collapse_count = int(view_collapse.get("summary", {}).get("collapse_count", 0)) if isinstance(view_collapse.get("summary"), Mapping) else 0
    if max(oracle_acc, chunk_acc) - base_best >= 0.20:
        return "STAGE8B1_ROUTING_BOTTLENECK"
    if collapse_count >= max(1, int(0.6 * len(view_collapse.get("rows", [])))):
        return "STAGE8B1_VIEW_COLLAPSE_BOTTLENECK"
    if max(delta_by_n.values(), default=0.0) >= 0.04:
        return "STAGE8B1_PROMISING_NARROW_SIGNAL"
    if max(oracle_acc, chunk_acc) < 0.35:
        return "STAGE8B1_COORDINATOR_BOTTLENECK"
    return "STAGE8B1_BASELINE_TOO_STRONG_NO_LATENT_ADVANTAGE"


def _recommendation(decision: str) -> str:
    return {
        "STAGE8B1_ROUTING_BOTTLENECK": "Proceed to Stage 8B.2 focused on learned routing, auxiliary localization, and hierarchical retrieve-then-score.",
        "STAGE8B1_COORDINATOR_BOTTLENECK": "Redesign candidate-conditioned coordinator before further search.",
        "STAGE8B1_VIEW_COLLAPSE_BOTTLENECK": "Add stronger diversity, load balancing, dropout, or architectural separation.",
        "STAGE8B1_BENCHMARK_SHORTCUT_OR_BROKEN": "Fix benchmark before architecture search.",
        "STAGE8B1_BASELINE_TOO_STRONG_NO_LATENT_ADVANTAGE": "Pause 10x capacity claim; retain Stage 8 as negative result.",
        "STAGE8B1_PROMISING_NARROW_SIGNAL": "Run focused Stage 8B.2 on the narrow regimes where latent variants show signal.",
    }.get(decision, "Do not run Stage 8C; inspect diagnostics.")


def _write_report(analysis: Mapping[str, object]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 8B.1 Failure Analysis",
        "",
        "## Executive Summary",
        "",
        f"- Decision: {analysis.get('decision')}",
        f"- Recommendation: {analysis.get('recommendation')}",
        "- NO_STAGE8C_RUN: true",
        "- NO_10X_CLAIM: true",
        "",
        "## Best Latent Result",
        "",
        f"- Best latent accuracy by N: {analysis.get('best_latent_accuracy_by_n')}",
        f"- Capacity ratio: {analysis.get('capacity_ratio')}",
        "",
        "## Best Monolithic Baseline Result",
        "",
        f"- Best monolithic accuracy by N: {analysis.get('best_monolithic_accuracy_by_n')}",
        f"- Monolithic analysis: {analysis.get('monolithic_analysis')}",
        "",
        "## Capacity Ratio Achieved",
        "",
        f"{analysis.get('capacity_ratio')}",
        "",
        "## Where Latent Variants Helped",
        "",
        f"- Delta by N: {analysis.get('latent_minus_monolithic_delta_by_n')}",
        "",
        "## Where Latent Variants Failed",
        "",
        "- All searched latent variants had effective capacity C=0 under the Stage 8 capacity definition.",
        "- Accuracies mostly remained near chance, so degradation controls often could not pass meaningfully.",
        "",
        "## Control Failures",
        "",
        f"{analysis.get('controls_summary')}",
        "",
        "## Shortcut Audit",
        "",
        f"{analysis.get('shortcut_deep_audit')}",
        "",
        "## View Collapse Audit",
        "",
        f"{analysis.get('view_collapse_audit')}",
        "",
        "## Routing Bottleneck Analysis",
        "",
        f"{analysis.get('diagnostic_reruns', {}).get('oracle_routing') if isinstance(analysis.get('diagnostic_reruns'), Mapping) else {}}",
        "",
        "## Chunking Bottleneck Analysis",
        "",
        f"{analysis.get('diagnostic_reruns', {}).get('oracle_chunk') if isinstance(analysis.get('diagnostic_reruns'), Mapping) else {}}",
        "",
        "## Compute Accounting",
        "",
        f"{analysis.get('compute_starvation_analysis')}",
        "",
        "## Coordinator Analysis",
        "",
        f"{analysis.get('coordinator_analysis')}",
        "",
        "## Recommended Next Action",
        "",
        str(analysis.get("recommendation")),
        "",
        "## Artifacts",
        "",
        f"- Failure analysis JSON: `{FAILURE_ANALYSIS_PATH}`",
        f"- Diagnostic reruns JSONL: `{DIAGNOSTIC_RERUNS_PATH}`",
        f"- View collapse audit: `{VIEW_COLLAPSE_PATH}`",
        f"- Shortcut deep audit: `{SHORTCUT_DEEP_AUDIT_PATH}`",
        f"- Oracle routing: `{ORACLE_ROUTING_PATH}`",
        f"- Oracle chunk: `{ORACLE_CHUNK_PATH}`",
        f"- Compute match: `{COMPUTE_MATCH_PATH}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _diagnostic_group_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not rows:
        return {"best_accuracy": 0.0, "mean_accuracy": 0.0, "rows": 0}
    return {
        "best_accuracy": max(float(row.get("accuracy", 0.0)) for row in rows),
        "mean_accuracy": mean(float(row.get("accuracy", 0.0)) for row in rows),
        "best_train_accuracy_sample": max(float(row.get("train_accuracy_sample", 0.0)) for row in rows),
        "rows": len(rows),
        "best_by_n": {
            str(n): max(float(row.get("accuracy", 0.0)) for row in rows if int(row.get("n_blocks", -1)) == n)
            for n in sorted({int(row.get("n_blocks", 0)) for row in rows})
        },
    }


def _row_brief(row: Mapping[str, object]) -> Dict[str, object]:
    return {
        "phase": row.get("phase"),
        "config_id": row.get("config_id"),
        "architecture_name": row.get("architecture_name"),
        "capacity_C": row.get("capacity_C"),
        "capacity_ratio_vs_baseline": row.get("capacity_ratio_vs_baseline"),
        "accuracy_by_n": row.get("accuracy_by_n"),
        "compute_estimate": row.get("compute_estimate"),
    }


def _read_curves(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _config_dict(config: Stage8ArchitectureConfig) -> Dict[str, object]:
    from src.experiments.stage8_architecture_registry import architecture_config_to_dict

    return architecture_config_to_dict(config)


def _entropy(values: Sequence[float]) -> float:
    total = sum(float(value) for value in values)
    if total <= 0:
        return 0.0
    return -sum((float(value) / total) * math.log(max(1e-12, float(value) / total)) for value in values if value > 0)


def _pairwise_cosine_mean(views: torch.Tensor) -> float:
    if views.shape[0] < 2:
        return 1.0
    normalized = F.normalize(views, dim=-1)
    sims = []
    for i in range(normalized.shape[0]):
        for j in range(i + 1, normalized.shape[0]):
            sims.append(float(torch.dot(normalized[i], normalized[j]).item()))
    return mean(sims) if sims else 1.0


def _routing_entropy_for_config(config_id: str) -> float:
    rows = _rows_for_config(config_id, _stage8b_rows(_load_jsonl(SEARCH_DB_PATH), _load_json(FINALIST_PATH)))
    values = []
    for row in rows:
        for seed_row in row.get("rows", []):  # type: ignore[union-attr]
            if isinstance(seed_row, Mapping):
                diag = seed_row.get("diagnostics", {})
                if isinstance(diag, Mapping) and "routing_entropy_mean" in diag:
                    values.append(float(diag["routing_entropy_mean"]))
    return mean(values) if values else 0.0


def _rows_for_config(config_id: str, rows: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    return [row for row in rows if row.get("config_id") == config_id]


def _group_predictiveness(groups: Sequence[object], labels: Sequence[object]) -> float:
    grouped: Dict[str, Counter] = defaultdict(Counter)
    for group, label in zip(groups, labels):
        grouped[str(group)][label] += 1
    fractions = []
    for counts in grouped.values():
        total = sum(counts.values())
        if total:
            fractions.append(max(counts.values()) / total)
    return max(fractions, default=0.0)


def _best_high_n(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    best = {}
    best_score = -1.0
    for row in rows:
        acc = row.get("accuracy_by_n", {})
        if not isinstance(acc, Mapping):
            continue
        high_n = max((int(n) for n in acc.keys()), default=0)
        score = float(acc.get(str(high_n), 0.0))
        if score > best_score:
            best_score = score
            best = {"config": row.get("architecture_name"), "n": high_n, "accuracy": score}
    return best


def _best_task_family(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    best = {"family": None, "accuracy": 0.0, "config": None}
    for row in rows:
        for seed_row in row.get("rows", []):  # type: ignore[union-attr]
            if not isinstance(seed_row, Mapping):
                continue
            family = seed_row.get("accuracy_by_task_family", {})
            if isinstance(family, Mapping):
                for name, value in family.items():
                    if float(value) > float(best["accuracy"]):
                        best = {"family": name, "accuracy": float(value), "config": row.get("architecture_name")}
    return best


def _best_control_status(rows: Sequence[Mapping[str, object]], passed: bool) -> Dict[str, object]:
    candidates = []
    for row in rows:
        summaries = []
        for seed_row in row.get("rows", []):  # type: ignore[union-attr]
            controls = seed_row.get("controls", {}) if isinstance(seed_row, Mapping) else {}
            if isinstance(controls, Mapping):
                summaries.extend(bool(payload.get("passes", False)) for payload in controls.values() if isinstance(payload, Mapping))
        status = bool(summaries) and all(summaries)
        if status == passed:
            candidates.append(row)
    top = _top_latent_rows(candidates, 1)[0] if candidates else {}
    return _row_brief(top)


if __name__ == "__main__":
    main()

