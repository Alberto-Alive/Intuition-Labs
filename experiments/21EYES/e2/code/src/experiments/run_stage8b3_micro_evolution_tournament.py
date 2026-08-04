from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from src.datasets.latent_attention_capacity_dataset import (
    TASK_FAMILIES,
    Stage8DatasetConfig,
    Stage8Example,
    build_stage8_examples,
    validate_stage8_examples,
)
from src.experiments.stage8_architecture_registry import architecture_config_to_dict
from src.experiments.stage8_capacity_evaluator import (
    Stage8EvaluationConfig,
    accuracy_by_family,
    accuracy_from_predictions,
    run_stage8_control_suite,
)
from src.models.latent_attention_variants import (
    Stage8ArchitectureConfig,
    build_stage8_selector,
    estimate_stage8_compute,
)


DATABASE_PATH = Path("results/stage8b3_tournament_database.jsonl")
ROUND1_PATH = Path("results/stage8b3_round1_screen.json")
ROUND2_PATH = Path("results/stage8b3_round2_promotions.json")
ROUND3_PATH = Path("results/stage8b3_round3_mutations.json")
ROUND4_PATH = Path("results/stage8b3_round4_finalists.json")
TOP3_CONFIGS_PATH = Path("results/stage8b3_top3_configs.yaml")
TOP3_FREEZE_MANIFEST_PATH = Path("results/stage8b3_top3_freeze_manifest.json")
CONTROLS_AUDIT_PATH = Path("results/stage8b3_controls_audit.jsonl")
VIEW_DIVERSITY_AUDIT_PATH = Path("results/stage8b3_view_diversity_audit.json")
COMPUTE_AUDIT_PATH = Path("results/stage8b3_compute_audit.json")
CAPACITY_CURVES_PATH = Path("results/stage8b3_capacity_curves.csv")
REPORT_PATH = Path("reports/STAGE8B3_PARALLEL_MICRO_EVOLUTION_TOURNAMENT.md")
BASELINE_CAPACITY_PATH = Path("results/stage8_baseline_capacity.json")

FULL_CONTROLS: Tuple[str, ...] = (
    "randomized_labels",
    "randomized_candidate_order_with_label_remap",
    "randomized_evidence_block_order",
    "evidence_candidate_mismatch",
    "cross_task_evidence_shuffle",
    "cross_task_query_shuffle",
    "candidate_only",
    "query_only",
    "evidence_only",
    "distractor_only",
    "schema_template_only",
    "hidden_state_shuffle",
)

ROUND1_CONTROLS: Tuple[str, ...] = (
    "randomized_labels",
    "evidence_candidate_mismatch",
    "cross_task_evidence_shuffle",
    "candidate_only",
    "query_only",
    "hidden_state_shuffle",
)

FAMILY_NAMES = {
    "A": "Candidate-to-View Routing",
    "B": "Support vs Contradiction Attention",
    "C": "Hierarchical Coarse-to-Fine Attention",
    "D": "Evidence-Location Supervision",
    "E": "Anti-Collapse View Specialization",
    "F": "Memory Slot Attention",
    "G": "Recurrent Latent Search",
    "H": "Cross-Candidate Comparison",
    "I": "Retrieval-Then-Latent-Reasoning Hybrid",
    "J": "Explicit View Objective Assignment",
}


@dataclass(frozen=True)
class TournamentVariant:
    config: Stage8ArchitectureConfig
    family_code: str
    family_name: str
    parent_config_id: str | None = None
    mutation_description: str = "initial"


@dataclass(frozen=True)
class TournamentBudget:
    round1_variants: int = 60
    round2_promoted: int = 20
    round3_mutations_per_parent: int = 3
    round4_finalists: int = 8
    train_examples: int = 96
    eval_examples: int = 96
    round1_n: Tuple[int, ...] = (2, 4, 8)
    round2_n: Tuple[int, ...] = (4, 8, 16)
    round3_n: Tuple[int, ...] = (8, 16)
    round4_n: Tuple[int, ...] = (8, 16, 32)
    round1_seeds: Tuple[int, ...] = (0,)
    round2_seeds: Tuple[int, ...] = (0, 1, 2)
    round3_seeds: Tuple[int, ...] = (0, 1, 2)
    round4_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    workers: int = 1
    include_n64: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 8B.3 parallel latent attention micro-evolution tournament.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--round1-variants", type=int, default=60)
    parser.add_argument("--round2-promoted", type=int, default=20)
    parser.add_argument("--round4-finalists", type=int, default=8)
    parser.add_argument("--train-examples", type=int, default=96)
    parser.add_argument("--eval-examples", type=int, default=96)
    parser.add_argument("--include-n64", action="store_true")
    args = parser.parse_args()
    budget = TournamentBudget(
        round1_variants=max(60, int(args.round1_variants)),
        round2_promoted=int(args.round2_promoted),
        round4_finalists=int(args.round4_finalists),
        train_examples=int(args.train_examples),
        eval_examples=int(args.eval_examples),
        workers=max(1, int(args.workers)),
        include_n64=bool(args.include_n64),
    )
    payload = run_stage8b3(budget)
    print(f"stage8b3: wrote {DATABASE_PATH}, {TOP3_CONFIGS_PATH}, {REPORT_PATH}; decision={payload['decision']}")


def run_stage8b3(budget: TournamentBudget | None = None) -> Dict[str, object]:
    budget = budget or TournamentBudget()
    _reset_outputs()
    started = time.perf_counter()
    baseline_capacity = _load_baseline_capacity()
    sanity = _round0_sanity(budget)
    if not sanity["passes"]:
        payload = _broken_payload(budget, sanity, started)
        _write_all_terminal_artifacts(payload, [], [], [], [])
        return payload

    round1_variants = generate_stage8b3_variants(max_variants=budget.round1_variants)
    round1_rows = _evaluate_variants(
        round_name="round1",
        variants=round1_variants,
        n_values=budget.round1_n,
        seeds=budget.round1_seeds,
        controls=ROUND1_CONTROLS,
        budget=budget,
        baseline_capacity=baseline_capacity,
    )
    round1_baselines = _run_required_baselines("round1", budget.round1_n, budget.round1_seeds, ROUND1_CONTROLS, budget)
    round1_survivors = [row for row in round1_rows if row["screen_checks"]["round1_pass"]]
    round1_killed = [row for row in round1_rows if not row["screen_checks"]["round1_pass"]]
    round1_payload = {
        "stage": "8B.3",
        "round": 1,
        "screen": "wide_micro_screen",
        "required_initial_variants": 60,
        "evaluated_variants": len(round1_rows),
        "survivor_count": len(round1_survivors),
        "killed_count": len(round1_killed),
        "survivors": _brief_rows(round1_survivors),
        "killed_variants": _brief_rows(round1_killed),
        "baselines": round1_baselines,
    }
    _dump_json(ROUND1_PATH, round1_payload)

    promoted_variants = _variants_from_rows(round1_survivors[: budget.round2_promoted], round1_variants)
    round2_rows = _evaluate_variants(
        round_name="round2",
        variants=promoted_variants,
        n_values=budget.round2_n,
        seeds=budget.round2_seeds,
        controls=FULL_CONTROLS,
        budget=budget,
        baseline_capacity=baseline_capacity,
    )
    round2_baselines = _run_required_baselines("round2", budget.round2_n, budget.round2_seeds, FULL_CONTROLS, budget)
    round2_survivors = [row for row in round2_rows if row["screen_checks"]["round2_pass"]]
    round2_payload = {
        "stage": "8B.3",
        "round": 2,
        "screen": "promotion_screen",
        "promoted_from_round1": _brief_rows(round1_survivors[: budget.round2_promoted]),
        "evaluated_count": len(round2_rows),
        "survivor_count": len(round2_survivors),
        "survivors": _brief_rows(round2_survivors),
        "killed_variants": _brief_rows([row for row in round2_rows if not row["screen_checks"]["round2_pass"]]),
        "baselines": round2_baselines,
    }
    _dump_json(ROUND2_PATH, round2_payload)

    mutation_variants = []
    parent_by_id = {row["config_id"]: row for row in round2_survivors}
    for parent in _variants_from_rows(round2_survivors, promoted_variants):
        mutation_variants.extend(mutate_stage8b3_variant(parent, budget.round3_mutations_per_parent))
    round3_rows = _evaluate_variants(
        round_name="round3",
        variants=mutation_variants,
        n_values=budget.round3_n,
        seeds=budget.round3_seeds,
        controls=FULL_CONTROLS,
        budget=budget,
        baseline_capacity=baseline_capacity,
    )
    round3_baselines = _run_required_baselines("round3", budget.round3_n, budget.round3_seeds, FULL_CONTROLS, budget)
    round3_kept = [row for row in round3_rows if _mutation_improved(row, parent_by_id)]
    round3_payload = {
        "stage": "8B.3",
        "round": 3,
        "screen": "mutation_evolution",
        "parent_count": len(round2_survivors),
        "mutation_count": len(round3_rows),
        "kept_count": len(round3_kept),
        "kept_mutations": _brief_rows(round3_kept),
        "discarded_mutations": _brief_rows([row for row in round3_rows if row not in round3_kept]),
        "baselines": round3_baselines,
    }
    _dump_json(ROUND3_PATH, round3_payload)

    candidate_pool_rows = _rank_rows(round2_survivors + round3_kept)
    finalist_seed_rows = candidate_pool_rows[: budget.round4_finalists]
    finalist_variants = _variants_from_rows(finalist_seed_rows, promoted_variants + mutation_variants)
    round4_n = tuple(list(budget.round4_n) + ([64] if budget.include_n64 else []))
    round4_rows = _evaluate_variants(
        round_name="round4",
        variants=finalist_variants,
        n_values=round4_n,
        seeds=budget.round4_seeds,
        controls=FULL_CONTROLS,
        budget=budget,
        baseline_capacity=baseline_capacity,
    )
    round4_baselines = _run_required_baselines("round4", round4_n, budget.round4_seeds, FULL_CONTROLS, budget)
    ranked_finalists = _rank_rows(round4_rows)
    top3 = _select_top3(ranked_finalists)
    decision = _decision(top3, ranked_finalists)
    round4_payload = {
        "stage": "8B.3",
        "round": 4,
        "screen": "finalist_screen",
        "evaluated_count": len(round4_rows),
        "finalists": _brief_rows(ranked_finalists),
        "top3": _brief_rows(top3),
        "baselines": round4_baselines,
        "decision": decision,
    }
    _dump_json(ROUND4_PATH, round4_payload)

    all_rows = round1_rows + round2_rows + round3_rows + round4_rows
    payload = {
        "stage": "8B.3",
        "status": "completed",
        "decision": decision,
        "no_stage8c_run": True,
        "no_10x_attention_capacity_claim": True,
        "wall_clock_seconds": time.perf_counter() - started,
        "budget": asdict(budget),
        "round0_sanity": sanity,
        "round1": round1_payload,
        "round2": round2_payload,
        "round3": round3_payload,
        "round4": round4_payload,
        "top3": top3,
        "baseline_capacity": baseline_capacity,
        "baseline_note": "Baseline capacity is zero in the prior Stage 8 artifacts; early capacity ratio is recorded as 0.0 and is not a 10x claim.",
        "recommendation": _recommendation(decision, top3),
    }
    _write_all_terminal_artifacts(payload, all_rows, top3, ranked_finalists, round4_baselines)
    return payload


def generate_stage8b3_variants(max_variants: int = 60) -> List[TournamentVariant]:
    per_family = max(5, math.ceil(max_variants / len(FAMILY_NAMES)))
    variants: List[TournamentVariant] = []
    for family_code in FAMILY_NAMES:
        for index in range(per_family):
            variants.append(_family_variant(family_code, index))
    return variants[: max(max_variants, 60)]


def mutate_stage8b3_variant(parent: TournamentVariant, count: int = 3) -> List[TournamentVariant]:
    mutations = [
        ("add support/contradiction split", {"coordinator": "support_contradiction_attention", "training_loss": "margin_loss"}),
        ("add auxiliary evidence-location loss", {"training_loss": "auxiliary_evidence_location", "chunking": "semantic_key"}),
        ("change top-k routing", {"attention_routing": "top_k_view_attention", "top_k_views": max(2, min(8, parent.config.latent_views))}),
        ("add role+avenue load balancing", {"regularizers": tuple(sorted(set(parent.config.regularizers + ("load_balancing_across_avenues",))))}),
        ("switch to memory slots", {"memory_compression": "slot_attention_compression", "roles": max(parent.config.roles, 6)}),
    ]
    output = []
    for index, (description, values) in enumerate(mutations[: max(3, count)]):
        config = replace(parent.config, name=f"{parent.config.name}_mut{index}")
        config = replace(config, **values)
        output.append(
            TournamentVariant(
                config=config,
                family_code=parent.family_code,
                family_name=parent.family_name,
                parent_config_id=parent.config.config_id,
                mutation_description=description,
            )
        )
    return output


def _family_variant(family_code: str, index: int) -> TournamentVariant:
    base = Stage8ArchitectureConfig(
        name=f"stage8b3_{family_code.lower()}_{index:02d}",
        model_kind="evidence_feature_latent",
        roles=4 + (index % 3) * 2,
        avenues=2 + (index % 2),
        chunking="semantic_key",
        candidate_query="candidate_task_query",
        coordinator="mixture_of_views",
        view_sharing="shared_encoder_role_avenue_embeddings",
        memory_compression="mean_pooling",
        attention_routing="dense_all_view_attention",
        training_loss="multi_positive_cross_entropy",
        regularizers=("entropy_regularization_on_routing",),
        curriculum="mixed_n",
        vector_dim=128,
        hidden_dim=32,
        coord_hops=1,
        top_k_views=4,
        epochs=4,
        lr=0.05,
        dropout=0.05 if index % 2 else 0.0,
        max_train_examples=128,
    )
    values_by_family = {
        "A": {
            "coordinator": ("top_k_latent_view_selection", "gated_latent_view_selection")[index % 2],
            "attention_routing": ("top_k_view_attention", "sparsemax_routing", "coarse_to_fine_routing")[index % 3],
            "top_k_views": (2, 4, 6, 8, 3, 5)[index % 6],
            "regularizers": ("entropy_regularization_on_routing", "load_balancing_across_avenues"),
        },
        "B": {
            "coordinator": "support_contradiction_attention",
            "training_loss": ("margin_loss", "pairwise_ranking_loss", "multi_positive_cross_entropy")[index % 3],
            "regularizers": ("load_balancing_across_avenues",),
        },
        "C": {
            "chunking": "hierarchical",
            "coordinator": "hierarchical_coordinator",
            "attention_routing": "coarse_to_fine_routing",
            "coord_hops": (1, 2, 3, 4, 2, 1)[index % 6],
        },
        "D": {
            "training_loss": ("auxiliary_evidence_location", "contrastive_candidate_evidence_alignment")[index % 2],
            "chunking": ("semantic_key", "hierarchical", "overlapping")[index % 3],
            "regularizers": ("entropy_regularization_on_routing", "load_balancing_across_avenues"),
        },
        "E": {
            "view_sharing": ("partially_shared_role_adapters", "shared_encoder_low_rank_adapters", "shared_encoder_role_avenue_embeddings")[index % 3],
            "regularizers": ("role_diversity_penalty", "avenue_diversity_penalty", "load_balancing_across_avenues", "chunk_dropout"),
            "roles": (6, 8, 10, 12, 8, 6)[index % 6],
            "avenues": (2, 3, 4, 2, 4, 3)[index % 6],
            "dropout": 0.10,
        },
        "F": {
            "memory_compression": ("slot_attention_compression", "recurrent_memory_slots")[index % 2],
            "coordinator": "memory_slot_attention",
            "roles": (5, 6, 8, 10, 6, 5)[index % 6],
            "avenues": 2,
        },
        "G": {
            "coordinator": ("recurrent_refinement_1", "recurrent_refinement_2", "recurrent_refinement_4", "recurrent_refinement_8")[index % 4],
            "coord_hops": (1, 2, 4, 8, 2, 4)[index % 6],
            "memory_compression": "recurrent_memory_slots",
        },
        "H": {
            "coordinator": "cross_candidate_comparison",
            "training_loss": ("pairwise_ranking_loss", "margin_loss")[index % 2],
            "candidate_query": ("multi_query_per_candidate", "candidate_role_query")[index % 2],
        },
        "I": {
            "coordinator": "retrieval_then_latent_reasoning",
            "attention_routing": "two_stage_retrieve_then_score",
            "memory_compression": ("top_k_token_retention", "candidate_conditioned_token_retention")[index % 2],
            "top_k_views": (2, 4, 8, 6, 3, 5)[index % 6],
        },
        "J": {
            "coordinator": "explicit_view_objective_assignment",
            "memory_compression": "slot_attention_compression",
            "view_sharing": ("partially_shared_role_adapters", "shared_encoder_role_avenue_embeddings")[index % 2],
            "roles": (6, 7, 8, 9, 10, 6)[index % 6],
            "avenues": 2,
        },
    }
    config = replace(base, **values_by_family[family_code])
    config = replace(config, name=f"stage8b3_{family_code.lower()}_{_slug(FAMILY_NAMES[family_code])}_{index:02d}")
    return TournamentVariant(config=config, family_code=family_code, family_name=FAMILY_NAMES[family_code])


def _evaluate_variants(
    round_name: str,
    variants: Sequence[TournamentVariant],
    n_values: Sequence[int],
    seeds: Sequence[int],
    controls: Sequence[str],
    budget: TournamentBudget,
    baseline_capacity: int,
) -> List[Dict[str, object]]:
    if not variants:
        return []
    tasks = [(variant, round_name, tuple(n_values), tuple(seeds), tuple(controls), budget, baseline_capacity) for variant in variants]
    if budget.workers <= 1:
        rows = [_evaluate_variant_task(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=budget.workers) as executor:
            rows = list(executor.map(_evaluate_variant_task, tasks))
    ranked = _rank_rows(rows)
    for row in ranked:
        _append_jsonl(DATABASE_PATH, row)
        _append_jsonl(CONTROLS_AUDIT_PATH, _controls_audit_row(row))
    _append_capacity_curves(ranked)
    return ranked


def _evaluate_variant_task(task) -> Dict[str, object]:
    variant, round_name, n_values, seeds, controls, budget, baseline_capacity = task
    return _evaluate_variant(variant, round_name, n_values, seeds, controls, budget, baseline_capacity)


def _evaluate_variant(
    variant: TournamentVariant,
    round_name: str,
    n_values: Sequence[int],
    seeds: Sequence[int],
    controls: Sequence[str],
    budget: TournamentBudget,
    baseline_capacity: int,
) -> Dict[str, object]:
    started = time.perf_counter()
    rows: List[Dict[str, object]] = []
    eval_config = Stage8EvaluationConfig(
        train_examples=budget.train_examples,
        eval_examples=budget.eval_examples,
        k_candidates=8,
        min_stable_seeds=max(1, len(seeds)),
        controls=tuple(controls),
    )
    for seed in seeds:
        for n_blocks in n_values:
            rows.append(_evaluate_seed_n(variant.config, int(n_blocks), int(seed), eval_config))
    return _summarize_variant(variant, round_name, rows, baseline_capacity, time.perf_counter() - started)


def _evaluate_seed_n(
    config: Stage8ArchitectureConfig,
    n_blocks: int,
    seed: int,
    eval_config: Stage8EvaluationConfig,
) -> Dict[str, object]:
    train = _examples("train", "train", n_blocks, eval_config.train_examples, seed, eval_config.k_candidates)
    same_dev = _examples("dev", "train", n_blocks, eval_config.eval_examples, seed + 10_000, eval_config.k_candidates)
    heldout_dev = _examples("dev", "dev", n_blocks, eval_config.eval_examples, seed + 20_000, eval_config.k_candidates)
    audit = validate_stage8_examples(train + same_dev + heldout_dev)
    selector = build_stage8_selector(config, seed=seed)
    fit_started = time.perf_counter()
    selector.fit(train, same_dev)
    train_seconds = time.perf_counter() - fit_started
    train_sample = train[: min(len(train), eval_config.eval_examples)]
    train_predictions = selector.predict(train_sample)
    same_predictions = selector.predict(same_dev)
    heldout_predictions = selector.predict(heldout_dev)
    heldout_labels = [example.label for example in heldout_dev]
    same_labels = [example.label for example in same_dev]
    controls = run_stage8_control_suite(selector, heldout_dev, heldout_predictions, heldout_labels, eval_config, seed)
    comparator = _comparator_accuracies(config, train, heldout_dev, seed)
    diagnostics = selector.diagnostics(heldout_dev[: min(32, len(heldout_dev))])
    return {
        "seed": int(seed),
        "n_blocks": int(n_blocks),
        "train_accuracy": accuracy_from_predictions(train_predictions, [example.label for example in train_sample]),
        "dev_accuracy": accuracy_from_predictions(same_predictions, same_labels),
        "heldout_template_dev_accuracy": accuracy_from_predictions(heldout_predictions, heldout_labels),
        "accuracy": accuracy_from_predictions(heldout_predictions, heldout_labels),
        "accuracy_by_task_family": accuracy_by_family(heldout_dev, heldout_predictions),
        "same_template_accuracy_by_task_family": accuracy_by_family(same_dev, same_predictions),
        "controls": controls,
        "comparators": comparator,
        "diagnostics": diagnostics,
        "dataset_audit_passes": bool(audit["passes"]),
        "dataset_audit_failures": audit["failures"],
        "num_train_examples": len(train),
        "num_eval_examples": len(heldout_dev),
        "train_wall_clock_seconds": train_seconds,
        "compute": estimate_stage8_compute(config, n_blocks, eval_config.k_candidates),
    }


def _examples(split: str, template_split: str, n_blocks: int, n_examples: int, seed: int, k_candidates: int) -> List[Stage8Example]:
    return build_stage8_examples(
        Stage8DatasetConfig(
            n_examples=n_examples,
            n_blocks=n_blocks,
            k_candidates=k_candidates,
            split=split,
            template_split=template_split,
        ),
        seed=seed,
    )


def _comparator_accuracies(
    config: Stage8ArchitectureConfig,
    train: Sequence[Stage8Example],
    dev: Sequence[Stage8Example],
    seed: int,
) -> Dict[str, float]:
    labels = [example.label for example in dev]
    comparators = {
        "frozen_same_architecture": replace(config, name=f"frozen_{config.name}", frozen=True, epochs=0),
        "candidate_only": replace(config, name=f"candidate_only_{config.name}", model_kind="candidate_only"),
        "query_only": replace(config, name=f"query_only_{config.name}", model_kind="query_only"),
        "evidence_only": replace(config, name=f"evidence_only_{config.name}", model_kind="evidence_only"),
        "raw_latent": replace(config, name=f"raw_latent_{config.name}", model_kind="raw_latent_selector"),
        "retrieval_topk": replace(config, name=f"retrieval_{config.name}", model_kind="retrieval_topk", top_k_views=max(4, config.top_k_views)),
    }
    output: Dict[str, float] = {}
    for name, comp_config in comparators.items():
        selector = build_stage8_selector(comp_config, seed=seed)
        selector.fit(train, dev)
        output[f"{name}_accuracy"] = accuracy_from_predictions(selector.predict(dev), labels)
    return output


def _summarize_variant(
    variant: TournamentVariant,
    round_name: str,
    rows: Sequence[Mapping[str, object]],
    baseline_capacity: int,
    wall_clock_seconds: float,
) -> Dict[str, object]:
    accuracy_by_n = _mean_by_n(rows, "heldout_template_dev_accuracy")
    dev_accuracy_by_n = _mean_by_n(rows, "dev_accuracy")
    train_accuracy_by_n = _mean_by_n(rows, "train_accuracy")
    accuracy_std_by_n = _std_by_n(rows, "heldout_template_dev_accuracy")
    controls = _control_summary(rows)
    comparators = _comparator_summary(rows)
    diversity = _diversity_summary(rows, variant.config)
    capacity = _capacity_from_rows(rows, controls)
    compute = estimate_stage8_compute(variant.config, max([int(row.get("n_blocks", 0)) for row in rows] or [0]), 8)
    task_family = _task_family_summary(rows)
    summary: Dict[str, object] = {
        "phase": f"stage8b3_{round_name}",
        "config_id": variant.config.config_id,
        "architecture_name": variant.config.name,
        "architecture_family": variant.family_name,
        "architecture_family_code": variant.family_code,
        "parent_config_id": variant.parent_config_id,
        "mutation_description": variant.mutation_description,
        "architecture_parameters": architecture_config_to_dict(variant.config),
        "N_schedule": sorted({int(row.get("n_blocks", 0)) for row in rows}),
        "task_families": list(TASK_FAMILIES),
        "seeds": sorted({int(row.get("seed", 0)) for row in rows}),
        "train_accuracy": _mean([float(row.get("train_accuracy", 0.0)) for row in rows]),
        "dev_accuracy": _mean([float(row.get("dev_accuracy", 0.0)) for row in rows]),
        "held_out_template_dev_accuracy": _mean([float(row.get("heldout_template_dev_accuracy", 0.0)) for row in rows]),
        "capacity_C": capacity,
        "early_capacity_ratio": float(capacity / baseline_capacity) if baseline_capacity > 0 else 0.0,
        "capacity_ratio_note": "baseline_capacity_zero_ratio_not_used" if baseline_capacity <= 0 else "ratio_vs_stage8a_monolithic",
        "accuracy_by_N": accuracy_by_n,
        "dev_accuracy_by_N": dev_accuracy_by_n,
        "train_accuracy_by_N": train_accuracy_by_n,
        "accuracy_std_by_N": accuracy_std_by_n,
        "accuracy_by_task_family": task_family,
        "randomized_label_accuracy": _control_metric(controls, "randomized_labels", "mean_accuracy"),
        "candidate_evidence_mismatch_accuracy": _control_metric(controls, "evidence_candidate_mismatch", "mean_accuracy"),
        "candidate_evidence_mismatch_degradation": _control_metric(controls, "evidence_candidate_mismatch", "mean_degradation"),
        "cross_task_evidence_shuffle_accuracy": _control_metric(controls, "cross_task_evidence_shuffle", "mean_accuracy"),
        "cross_task_evidence_shuffle_degradation": _control_metric(controls, "cross_task_evidence_shuffle", "mean_degradation"),
        "candidate_only_accuracy": comparators.get("candidate_only_accuracy", 0.0),
        "query_only_accuracy": comparators.get("query_only_accuracy", 0.0),
        "evidence_only_accuracy": comparators.get("evidence_only_accuracy", 0.0),
        "frozen_comparator_accuracy": comparators.get("frozen_same_architecture_accuracy", 0.0),
        "raw_latent_comparator_accuracy": comparators.get("raw_latent_accuracy", 0.0),
        "retrieval_topk_baseline_accuracy": comparators.get("retrieval_topk_accuracy", 0.0),
        "hidden_state_shuffle_accuracy": _control_metric(controls, "hidden_state_shuffle", "mean_accuracy"),
        "hidden_state_shuffle_degradation": _control_metric(controls, "hidden_state_shuffle", "mean_degradation"),
        "role_entropy": diversity["role_entropy"],
        "avenue_entropy": diversity["avenue_entropy"],
        "routing_entropy": diversity["routing_entropy"],
        "attention_concentration": diversity["attention_concentration"],
        "pairwise_hidden_state_similarity": diversity["pairwise_hidden_state_similarity"],
        "parameter_count": int(compute.get("parameter_count_estimate", 0)),
        "estimated_compute": compute,
        "compute_normalized_capacity": capacity / max(1.0, float(compute.get("estimated_forward_compute", 1.0))),
        "wall_clock_time": wall_clock_seconds,
        "memory_usage": {"estimated_peak_bytes": 0, "source": "not_measured_cpu_micro_tournament"},
        "controls_summary": controls,
        "comparator_summary": comparators,
        "view_diversity": diversity,
        "rows": list(rows),
        "status": "completed",
    }
    summary["screen_checks"] = _screen_checks(summary)
    summary["failure_reason"] = _failure_reason(summary)
    summary["selection_components"] = _selection_components(summary)
    summary["selection_score"] = 0.0
    return summary


def _screen_checks(row: Mapping[str, object]) -> Dict[str, object]:
    acc8 = _accuracy_at(row, 8)
    mismatch = float(row.get("candidate_evidence_mismatch_degradation", 0.0))
    cross = float(row.get("cross_task_evidence_shuffle_degradation", 0.0))
    hidden = float(row.get("hidden_state_shuffle_degradation", 0.0))
    full = float(row.get("held_out_template_dev_accuracy", 0.0))
    frozen = float(row.get("frozen_comparator_accuracy", 0.0))
    candidate = float(row.get("candidate_only_accuracy", 0.0))
    query = float(row.get("query_only_accuracy", 0.0))
    randomized = float(row.get("randomized_label_accuracy", 1.0))
    pairwise = float(row.get("pairwise_hidden_state_similarity", 1.0))
    routing = float(row.get("routing_entropy", 0.0))
    same_minus_heldout = float(row.get("dev_accuracy", 0.0)) - full
    families85 = _families_at_threshold(row, threshold=0.85, n_values=(8, 16))
    common = {
        "randomized_labels_collapse": randomized <= 0.35,
        "candidate_only_not_explanatory": candidate <= max(0.35, full - 0.10),
        "query_only_not_explanatory": query <= max(0.35, full - 0.10),
        "view_not_severely_collapsed": pairwise < 0.85 and routing > 0.10,
        "hidden_state_shuffle_degrades": hidden >= 0.10,
        "heldout_template_not_failed": full >= 0.70 and same_minus_heldout <= 0.20,
        "families_ge_085_at_N8_or_N16": families85,
    }
    round1_pass = bool(
        acc8 >= 0.40
        and common["randomized_labels_collapse"]
        and common["candidate_only_not_explanatory"]
        and common["query_only_not_explanatory"]
        and mismatch > 0.0
        and common["view_not_severely_collapsed"]
    )
    round2_pass = bool(
        acc8 >= 0.70
        and mismatch >= 0.05
        and cross >= 0.05
        and full > frozen + 0.05
        and common["heldout_template_not_failed"]
    )
    finalist_pass = bool(
        round2_pass
        and mismatch >= 0.10
        and cross >= 0.10
        and hidden >= 0.10
        and common["randomized_labels_collapse"]
        and common["candidate_only_not_explanatory"]
        and common["query_only_not_explanatory"]
        and common["view_not_severely_collapsed"]
        and len(families85) >= 3
    )
    return {**common, "round1_pass": round1_pass, "round2_pass": round2_pass, "finalist_pass": finalist_pass}


def _failure_reason(row: Mapping[str, object]) -> str | None:
    checks = row.get("screen_checks", {})
    if not isinstance(checks, Mapping):
        return "missing_screen_checks"
    if checks.get("finalist_pass") or checks.get("round2_pass") or checks.get("round1_pass"):
        return None
    failures = [name for name, value in checks.items() if isinstance(value, bool) and not value]
    return ";".join(failures[:6]) or None


def _selection_components(row: Mapping[str, object]) -> Dict[str, float]:
    highest_n = max((int(n) for n in row.get("accuracy_by_N", {}).keys()), default=0) if isinstance(row.get("accuracy_by_N"), Mapping) else 0
    highest_acc = _accuracy_at(row, highest_n) if highest_n else 0.0
    evidence_degradation = _mean(
        [
            float(row.get("candidate_evidence_mismatch_degradation", 0.0)),
            float(row.get("cross_task_evidence_shuffle_degradation", 0.0)),
            float(row.get("hidden_state_shuffle_degradation", 0.0)),
        ]
    )
    frozen_gap = max(0.0, float(row.get("held_out_template_dev_accuracy", 0.0)) - float(row.get("frozen_comparator_accuracy", 0.0)))
    diversity = max(0.0, min(1.0, 1.0 - float(row.get("pairwise_hidden_state_similarity", 1.0))))
    stds = [float(value) for value in row.get("accuracy_std_by_N", {}).values()] if isinstance(row.get("accuracy_std_by_N"), Mapping) else []
    stability = max(0.0, 1.0 - _mean(stds))
    return {
        "normalized_dev_accuracy_highest_N": min(1.0, highest_acc),
        "evidence_use_control_degradation": min(1.0, evidence_degradation),
        "trainable_vs_frozen_gap": min(1.0, frozen_gap),
        "view_diversity_anti_collapse": diversity,
        "stability_across_seeds": stability,
        "compute_normalized_capacity": float(row.get("compute_normalized_capacity", 0.0)),
    }


def _select_top3(finalists: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    passers = [dict(row) for row in finalists if isinstance(row.get("screen_checks"), Mapping) and row["screen_checks"].get("finalist_pass")]
    if not passers:
        return []
    max_compute_norm = max(float(row.get("compute_normalized_capacity", 0.0)) for row in passers) or 1.0
    scored = []
    for row in passers:
        components = dict(row["selection_components"])
        components["compute_normalized_capacity"] = min(1.0, float(row.get("compute_normalized_capacity", 0.0)) / max_compute_norm)
        score = (
            0.30 * components["normalized_dev_accuracy_highest_N"]
            + 0.20 * components["evidence_use_control_degradation"]
            + 0.15 * components["trainable_vs_frozen_gap"]
            + 0.15 * components["view_diversity_anti_collapse"]
            + 0.10 * components["stability_across_seeds"]
            + 0.10 * components["compute_normalized_capacity"]
        )
        row["selection_components"] = components
        row["selection_score"] = score
        scored.append(row)
    scored.sort(key=lambda row: (-float(row["selection_score"]), -int(row.get("capacity_C", 0)), str(row.get("architecture_name"))))
    return scored[:3]


def _decision(top3: Sequence[Mapping[str, object]], finalists: Sequence[Mapping[str, object]]) -> str:
    if len(top3) >= 3 and any(_accuracy_at(row, 16) >= 0.85 or _accuracy_at(row, 32) >= 0.85 for row in top3):
        return "STAGE8B3_TOP3_SELECTED"
    if top3 or any(row.get("screen_checks", {}).get("round2_pass") for row in finalists if isinstance(row.get("screen_checks"), Mapping)):
        return "STAGE8B3_PROMISING_BUT_NO_TOP3"
    if not finalists:
        return "STAGE8B3_BENCHMARK_OR_REPRESENTATION_BROKEN"
    return "STAGE8B3_NO_PROMISING_ARCHITECTURES"


def _round0_sanity(budget: TournamentBudget) -> Dict[str, object]:
    n_blocks = 8
    examples = _examples("dev", "dev", n_blocks, budget.eval_examples, 83030, 8)
    oracle_predictions = []
    for example in examples:
        answer = str(example.metadata.get("answer_token") or example.metadata.get("answer_item") or example.metadata.get("answer_value") or "")
        candidates = [candidate.lower() for candidate in example.candidates]
        oracle_predictions.append(max(range(len(candidates)), key=lambda index: 1 if answer and answer.lower() in candidates[index] else 0))
    oracle_accuracy = accuracy_from_predictions(oracle_predictions, [example.label for example in examples])
    easy_config = Stage8ArchitectureConfig(
        name="stage8b3_round0_evidence_feature_oracle_routing",
        model_kind="evidence_feature_latent",
        roles=6,
        avenues=2,
        chunking="semantic_key",
        coordinator="support_contradiction_attention",
        memory_compression="slot_attention_compression",
        training_loss="margin_loss",
        regularizers=("load_balancing_across_avenues",),
        epochs=4,
        lr=0.05,
        max_train_examples=budget.train_examples,
    )
    row = _evaluate_seed_n(
        easy_config,
        n_blocks,
        0,
        Stage8EvaluationConfig(
            train_examples=budget.train_examples,
            eval_examples=budget.eval_examples,
            min_stable_seeds=1,
            controls=("randomized_labels", "evidence_candidate_mismatch"),
        ),
    )
    easy_family_learned = max(row["accuracy_by_task_family"].values()) >= 0.90 if isinstance(row.get("accuracy_by_task_family"), Mapping) else False
    passes = bool(oracle_accuracy >= 0.90 and easy_family_learned and row["heldout_template_dev_accuracy"] >= 0.85)
    return {
        "passes": passes,
        "oracle_evidence_accuracy_N8": oracle_accuracy,
        "oracle_evidence_threshold": 0.90,
        "evidence_feature_accuracy_N8": row["heldout_template_dev_accuracy"],
        "evidence_feature_accuracy_by_task_family": row["accuracy_by_task_family"],
        "easy_family_learned": easy_family_learned,
        "decision_if_failed": "STAGE8B3_BENCHMARK_OR_REPRESENTATION_BROKEN",
    }


def _run_required_baselines(
    round_name: str,
    n_values: Sequence[int],
    seeds: Sequence[int],
    controls: Sequence[str],
    budget: TournamentBudget,
) -> Dict[str, object]:
    del controls
    base = Stage8ArchitectureConfig(name=f"stage8b3_{round_name}_baseline", epochs=2, lr=0.01, max_train_examples=budget.train_examples)
    configs = [
        replace(base, name=f"{round_name}_random_candidate", model_kind="random_candidate"),
        replace(base, name=f"{round_name}_candidate_only", model_kind="candidate_only"),
        replace(base, name=f"{round_name}_query_only", model_kind="query_only"),
        replace(base, name=f"{round_name}_evidence_only", model_kind="evidence_only"),
        replace(base, name=f"{round_name}_retrieval_topk", model_kind="retrieval_topk", top_k_views=4),
        replace(base, name=f"{round_name}_true_monolithic_transformer", model_kind="monolithic_transformer", roles=1, avenues=1, hidden_dim=32),
        replace(base, name=f"{round_name}_legacy_hashed_feature_baseline", model_kind="retrieval_topk", top_k_views=1),
        replace(base, name=f"{round_name}_raw_latent_comparator", model_kind="raw_latent_selector"),
    ]
    output = []
    eval_config = Stage8EvaluationConfig(train_examples=budget.train_examples, eval_examples=budget.eval_examples, controls=())
    for config in configs:
        rows = [_evaluate_seed_n(config, int(n), int(seed), eval_config) for seed in seeds for n in n_values]
        output.append(
            {
                "architecture_name": config.name,
                "model_kind": config.model_kind,
                "accuracy_by_N": _mean_by_n(rows, "heldout_template_dev_accuracy"),
                "held_out_template_dev_accuracy": _mean([float(row.get("heldout_template_dev_accuracy", 0.0)) for row in rows]),
                "parameter_count": estimate_stage8_compute(config, max(n_values), 8).get("parameter_count_estimate", 0),
            }
        )
    return {"round": round_name, "n_values": list(n_values), "seeds": list(seeds), "rows": output}


def _capacity_from_rows(rows: Sequence[Mapping[str, object]], controls: Mapping[str, object]) -> int:
    capacity = 0
    by_n: Dict[int, List[float]] = {}
    for row in rows:
        by_n.setdefault(int(row.get("n_blocks", 0)), []).append(float(row.get("heldout_template_dev_accuracy", 0.0)))
    mismatch = _control_metric(controls, "evidence_candidate_mismatch", "mean_degradation")
    cross = _control_metric(controls, "cross_task_evidence_shuffle", "mean_degradation")
    randomized = _control_metric(controls, "randomized_labels", "mean_accuracy")
    for n_blocks, values in sorted(by_n.items()):
        if _mean(values) >= 0.85 and min(values) >= 0.75 and (pstdev(values) if len(values) > 1 else 0.0) <= 0.08:
            if mismatch >= 0.10 and cross >= 0.10 and randomized <= 0.35:
                capacity = max(capacity, n_blocks)
    return capacity


def _control_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        controls = row.get("controls", {})
        if not isinstance(controls, Mapping):
            continue
        for name, payload in controls.items():
            if isinstance(payload, Mapping):
                grouped.setdefault(str(name), []).append(payload)
    return {
        name: {
            "mean_accuracy": _mean([float(row.get("accuracy", 0.0)) for row in values]),
            "mean_degradation": _mean([float(row.get("degradation", 0.0)) for row in values]),
            "pass_rate": _mean([1.0 if row.get("passes") else 0.0 for row in values]),
            "count": len(values),
        }
        for name, values in sorted(grouped.items())
    }


def _comparator_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        comparators = row.get("comparators", {})
        if isinstance(comparators, Mapping):
            for name, value in comparators.items():
                grouped.setdefault(str(name), []).append(float(value))
    return {name: _mean(values) for name, values in sorted(grouped.items())}


def _diversity_summary(rows: Sequence[Mapping[str, object]], config: Stage8ArchitectureConfig) -> Dict[str, float]:
    role_entropies = []
    avenue_entropies = []
    routing = []
    pairwise = []
    for row in rows:
        diag = row.get("diagnostics", {})
        if not isinstance(diag, Mapping):
            continue
        role_dist = diag.get("role_usage_distribution", [])
        avenue_dist = diag.get("avenue_usage_distribution", [])
        if isinstance(role_dist, list):
            role_entropies.append(_entropy(role_dist) / math.log(max(2, len(role_dist))))
        if isinstance(avenue_dist, list):
            avenue_entropies.append(_entropy(avenue_dist) / math.log(max(2, len(avenue_dist))))
        if "routing_entropy_mean" in diag:
            routing.append(float(diag["routing_entropy_mean"]))
        if "pairwise_hidden_state_similarity" in diag:
            pairwise.append(float(diag["pairwise_hidden_state_similarity"]))
    max_routing = math.log(max(2, config.latent_views))
    routing_mean = _mean(routing)
    return {
        "role_entropy": _mean(role_entropies),
        "avenue_entropy": _mean(avenue_entropies),
        "routing_entropy": routing_mean,
        "attention_concentration": 1.0 - min(1.0, routing_mean / max(1e-6, max_routing)),
        "pairwise_hidden_state_similarity": _mean(pairwise) if pairwise else 1.0,
    }


def _task_family_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        values = row.get("accuracy_by_task_family", {})
        if isinstance(values, Mapping):
            for family, accuracy in values.items():
                grouped.setdefault(str(family), []).append(float(accuracy))
    return {family: _mean(values) for family, values in sorted(grouped.items())}


def _families_at_threshold(row: Mapping[str, object], threshold: float, n_values: Sequence[int]) -> List[str]:
    output = []
    by_family = row.get("accuracy_by_task_family", {})
    if not isinstance(by_family, Mapping):
        return output
    if max((_accuracy_at(row, n) for n in n_values), default=0.0) < threshold:
        return output
    for family, accuracy in by_family.items():
        if float(accuracy) >= threshold:
            output.append(str(family))
    return output


def _mutation_improved(row: Mapping[str, object], parent_by_id: Mapping[str, Mapping[str, object]]) -> bool:
    parent_id = str(row.get("parent_config_id") or "")
    parent = parent_by_id.get(parent_id)
    if not parent:
        return bool(row.get("screen_checks", {}).get("round2_pass")) if isinstance(row.get("screen_checks"), Mapping) else False
    return bool(
        float(row.get("held_out_template_dev_accuracy", 0.0)) >= float(parent.get("held_out_template_dev_accuracy", 0.0)) + 0.005
        or float(row.get("candidate_evidence_mismatch_degradation", 0.0)) >= float(parent.get("candidate_evidence_mismatch_degradation", 0.0)) + 0.025
    ) and bool(row.get("screen_checks", {}).get("round2_pass")) if isinstance(row.get("screen_checks"), Mapping) else False


def _rank_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    ranked = []
    for row in rows:
        copy = dict(row)
        components = _selection_components(copy)
        copy["selection_components"] = components
        copy["selection_score"] = (
            0.30 * components["normalized_dev_accuracy_highest_N"]
            + 0.20 * components["evidence_use_control_degradation"]
            + 0.15 * components["trainable_vs_frozen_gap"]
            + 0.15 * components["view_diversity_anti_collapse"]
            + 0.10 * components["stability_across_seeds"]
            + 0.10 * min(1.0, components["compute_normalized_capacity"] * 100000.0)
        )
        ranked.append(copy)
    ranked.sort(key=lambda row: (-float(row.get("selection_score", 0.0)), -int(row.get("capacity_C", 0)), str(row.get("architecture_name"))))
    return ranked


def _variants_from_rows(rows: Sequence[Mapping[str, object]], variants: Sequence[TournamentVariant]) -> List[TournamentVariant]:
    by_id = {variant.config.config_id: variant for variant in variants}
    output = []
    for row in rows:
        variant = by_id.get(str(row.get("config_id")))
        if variant is not None:
            output.append(variant)
    return output


def _write_all_terminal_artifacts(
    payload: Mapping[str, object],
    all_rows: Sequence[Mapping[str, object]],
    top3: Sequence[Mapping[str, object]],
    finalists: Sequence[Mapping[str, object]],
    round4_baselines: Mapping[str, object],
) -> None:
    _write_top3_yaml(top3, str(payload.get("decision")))
    _dump_json(TOP3_FREEZE_MANIFEST_PATH, _freeze_manifest(payload, top3))
    _dump_json(VIEW_DIVERSITY_AUDIT_PATH, _view_diversity_audit(all_rows))
    _dump_json(COMPUTE_AUDIT_PATH, _compute_audit(all_rows))
    if not CAPACITY_CURVES_PATH.exists():
        _append_capacity_curves([])
    if not ROUND1_PATH.exists():
        _dump_json(ROUND1_PATH, {"stage": "8B.3", "status": "not_run", "reason": payload.get("decision")})
    if not ROUND2_PATH.exists():
        _dump_json(ROUND2_PATH, {"stage": "8B.3", "status": "not_run", "reason": payload.get("decision")})
    if not ROUND3_PATH.exists():
        _dump_json(ROUND3_PATH, {"stage": "8B.3", "status": "not_run", "reason": payload.get("decision")})
    if not ROUND4_PATH.exists():
        _dump_json(ROUND4_PATH, {"stage": "8B.3", "status": "not_run", "reason": payload.get("decision")})
    _write_report(payload, all_rows, finalists, round4_baselines)


def _write_top3_yaml(top3: Sequence[Mapping[str, object]], decision: str) -> None:
    TOP3_CONFIGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "stage: \"8B.3\"",
        f"decision: {json.dumps(decision)}",
        "no_stage8c_run: true",
        "no_10x_attention_capacity_claim: true",
        "top3:",
    ]
    if not top3:
        lines.append("  []")
    for rank, row in enumerate(top3, start=1):
        lines.append(f"  - rank: {rank}")
        lines.append(f"    config_id: {json.dumps(row.get('config_id'))}")
        lines.append(f"    architecture_family: {json.dumps(row.get('architecture_family'))}")
        lines.append(f"    selection_score: {float(row.get('selection_score', 0.0)):.6f}")
        lines.append(f"    capacity_C: {json.dumps(row.get('capacity_C', 0))}")
        lines.append("    architecture:")
        params = row.get("architecture_parameters", {})
        if isinstance(params, Mapping):
            for key, value in params.items():
                lines.append(f"      {key}: {json.dumps(value)}")
    TOP3_CONFIGS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _freeze_manifest(payload: Mapping[str, object], top3: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "stage": "8B.3",
        "decision": payload.get("decision"),
        "freeze_statement": "Top-3 Stage 8B.3 architecture configs are frozen for later larger-scale validation. Stage 8C was not run.",
        "no_stage8c_run": True,
        "no_10x_attention_capacity_claim": True,
        "selection_score_formula": {
            "normalized_dev_accuracy_highest_N": 0.30,
            "evidence_use_control_degradation": 0.20,
            "trainable_vs_frozen_gap": 0.15,
            "view_diversity_anti_collapse": 0.15,
            "stability_across_seeds": 0.10,
            "compute_normalized_capacity": 0.10,
        },
        "top3": [_brief_row(row) for row in top3],
        "top3_full_configs": [row.get("architecture_parameters", {}) for row in top3],
        "artifacts": _artifact_map(),
    }


def _write_report(
    payload: Mapping[str, object],
    all_rows: Sequence[Mapping[str, object]],
    finalists: Sequence[Mapping[str, object]],
    round4_baselines: Mapping[str, object],
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    top3 = payload.get("top3", [])
    top3_rows = top3 if isinstance(top3, list) else []
    lines = [
        "# Stage 8B.3 Parallel Latent Attention Micro-Evolution Tournament",
        "",
        "## 1. Executive summary",
        "",
        f"- Decision: {payload.get('decision')}",
        "- Stage 8C was not run.",
        "- No 10x attention-capacity claim is made.",
        f"- Top-3 selected: {len(top3_rows)}",
        f"- Recommendation: {payload.get('recommendation')}",
        "",
        "## 2. Why Stage 8B failed",
        "",
        "Stage 8B and Stage 8B.1 showed zero latent and monolithic capacity, near-chance latent accuracy, weak degradation under evidence controls, and substantial role/avenue/view collapse. Stage 8B.3 therefore treats the problem as architecture discovery rather than final validation.",
        "",
        "## 3. Tournament design",
        "",
        f"- Round 0 sanity: {payload.get('round0_sanity')}",
        f"- Budget: {payload.get('budget')}",
        "- Same-template dev uses dev namespace with train templates. Held-out-template dev uses dev namespace with dev templates. Final templates are not used.",
        "",
        "## 4. Architecture families tested",
        "",
    ]
    for code, name in FAMILY_NAMES.items():
        count = sum(1 for row in all_rows if row.get("architecture_family_code") == code and row.get("phase") == "stage8b3_round1")
        lines.append(f"- {code}. {name}: {count} initial variants")
    round1 = payload.get("round1", {}) if isinstance(payload.get("round1"), Mapping) else {}
    round2 = payload.get("round2", {}) if isinstance(payload.get("round2"), Mapping) else {}
    round3 = payload.get("round3", {}) if isinstance(payload.get("round3"), Mapping) else {}
    round4 = payload.get("round4", {}) if isinstance(payload.get("round4"), Mapping) else {}
    lines.extend(
        [
            "",
            "## 5. Round 1 killed variants",
            "",
            f"- Evaluated: {round1.get('evaluated_variants', 0)}",
            f"- Killed: {round1.get('killed_count', 0)}",
            f"- Survivors: {round1.get('survivor_count', 0)}",
            "",
            "## 6. Round 2 promoted variants",
            "",
            f"- Evaluated: {round2.get('evaluated_count', 0)}",
            f"- Survivors: {round2.get('survivor_count', 0)}",
            "",
            "## 7. Round 3 mutations and improvements",
            "",
            f"- Parents: {round3.get('parent_count', 0)}",
            f"- Mutations: {round3.get('mutation_count', 0)}",
            f"- Kept mutations: {round3.get('kept_count', 0)}",
            "",
            "## 8. Round 4 finalists",
            "",
            f"- Evaluated finalists: {round4.get('evaluated_count', 0)}",
        ]
    )
    for row in _brief_rows(finalists[:8]):
        lines.append(f"- {row.get('architecture_name')}: score={float(row.get('selection_score', 0.0)):.4f}, C={row.get('capacity_C')}, heldout={row.get('held_out_template_dev_accuracy')}")
    lines.extend(["", "## 9. Top 3 selected architectures", ""])
    if top3_rows:
        for rank, row in enumerate(top3_rows, start=1):
            lines.append(
                f"{rank}. {row.get('architecture_name')} ({row.get('architecture_family')}), "
                f"score={float(row.get('selection_score', 0.0)):.4f}, C={row.get('capacity_C')}"
            )
    else:
        lines.append("No top-3 set passed all hard disqualifiers.")
    lines.extend(
        [
            "",
            "## 10. Evidence-use controls",
            "",
            _top_control_text(top3_rows),
            "",
            "## 11. View diversity / collapse analysis",
            "",
            _top_diversity_text(top3_rows),
            "",
            "## 12. Compute accounting",
            "",
            f"- Compute audit: `{COMPUTE_AUDIT_PATH}`",
            f"- Capacity curves: `{CAPACITY_CURVES_PATH}`",
            "",
            "## 13. Baseline comparison",
            "",
            json.dumps(round4_baselines, indent=2, sort_keys=True),
            "",
            "## 14. Failure modes",
            "",
            "- Failed variants remain in the tournament database with failure reasons.",
            "- Variants are disqualified when randomized labels do not collapse, evidence controls fail, candidate/query-only comparators explain performance, trainable does not beat frozen, or view collapse remains severe.",
            "",
            "## 15. Recommendation for Stage 8B.4",
            "",
            str(payload.get("recommendation")),
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, path in _artifact_map().items():
        lines.append(f"- {name}: `{path}`")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _top_control_text(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return "No passing top-3 controls to report."
    lines = []
    for row in rows:
        lines.append(
            f"- {row.get('architecture_name')}: mismatch degradation={float(row.get('candidate_evidence_mismatch_degradation', 0.0)):.3f}, "
            f"cross-task evidence degradation={float(row.get('cross_task_evidence_shuffle_degradation', 0.0)):.3f}, "
            f"randomized-label accuracy={float(row.get('randomized_label_accuracy', 0.0)):.3f}"
        )
    return "\n".join(lines)


def _top_diversity_text(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return "No passing top-3 diversity audit to report."
    lines = []
    for row in rows:
        lines.append(
            f"- {row.get('architecture_name')}: pairwise similarity={float(row.get('pairwise_hidden_state_similarity', 0.0)):.3f}, "
            f"role entropy={float(row.get('role_entropy', 0.0)):.3f}, routing entropy={float(row.get('routing_entropy', 0.0)):.3f}"
        )
    return "\n".join(lines)


def _view_diversity_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    audit_rows = [
        {
            "config_id": row.get("config_id"),
            "architecture_name": row.get("architecture_name"),
            "architecture_family": row.get("architecture_family"),
            "role_entropy": row.get("role_entropy"),
            "avenue_entropy": row.get("avenue_entropy"),
            "routing_entropy": row.get("routing_entropy"),
            "attention_concentration": row.get("attention_concentration"),
            "pairwise_hidden_state_similarity": row.get("pairwise_hidden_state_similarity"),
            "view_collapse_severe": float(row.get("pairwise_hidden_state_similarity", 1.0)) >= 0.85,
        }
        for row in rows
    ]
    return {
        "rows": audit_rows,
        "summary": {
            "count": len(audit_rows),
            "severe_collapse_count": sum(1 for row in audit_rows if row["view_collapse_severe"]),
            "mean_pairwise_hidden_state_similarity": _mean([float(row["pairwise_hidden_state_similarity"]) for row in audit_rows]) if audit_rows else 0.0,
        },
        "stage8b1_reference_mean_pairwise_similarity": 0.7086284708390153,
    }


def _compute_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "rows": [
            {
                "phase": row.get("phase"),
                "config_id": row.get("config_id"),
                "architecture_name": row.get("architecture_name"),
                "capacity_C": row.get("capacity_C"),
                "early_capacity_ratio": row.get("early_capacity_ratio"),
                "parameter_count": row.get("parameter_count"),
                "estimated_compute": row.get("estimated_compute"),
                "compute_normalized_capacity": row.get("compute_normalized_capacity"),
                "wall_clock_time": row.get("wall_clock_time"),
                "memory_usage": row.get("memory_usage"),
            }
            for row in rows
        ]
    }


def _append_capacity_curves(rows: Sequence[Mapping[str, object]]) -> None:
    exists = CAPACITY_CURVES_PATH.exists()
    CAPACITY_CURVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CAPACITY_CURVES_PATH.open("a", encoding="utf-8") as handle:
        if not exists:
            handle.write("phase,config_id,architecture_name,architecture_family,n_blocks,seed,dev_accuracy,heldout_template_dev_accuracy,capacity_C,estimated_compute\n")
        for row in rows:
            for seed_row in row.get("rows", []):  # type: ignore[union-attr]
                if not isinstance(seed_row, Mapping):
                    continue
                compute = seed_row.get("compute", {})
                estimated = float(compute.get("estimated_forward_compute", 0.0)) if isinstance(compute, Mapping) else 0.0
                handle.write(
                    "{phase},{config_id},{name},{family},{n},{seed},{dev:.6f},{heldout:.6f},{capacity},{compute:.3f}\n".format(
                        phase=row.get("phase", ""),
                        config_id=row.get("config_id", ""),
                        name=str(row.get("architecture_name", "")).replace(",", "_"),
                        family=str(row.get("architecture_family", "")).replace(",", "_"),
                        n=int(seed_row.get("n_blocks", 0)),
                        seed=int(seed_row.get("seed", 0)),
                        dev=float(seed_row.get("dev_accuracy", 0.0)),
                        heldout=float(seed_row.get("heldout_template_dev_accuracy", 0.0)),
                        capacity=int(row.get("capacity_C", 0)),
                        compute=estimated,
                    )
                )


def _controls_audit_row(row: Mapping[str, object]) -> Dict[str, object]:
    return {
        "phase": row.get("phase"),
        "config_id": row.get("config_id"),
        "architecture_name": row.get("architecture_name"),
        "architecture_family": row.get("architecture_family"),
        "controls_summary": row.get("controls_summary"),
        "candidate_only_accuracy": row.get("candidate_only_accuracy"),
        "query_only_accuracy": row.get("query_only_accuracy"),
        "evidence_only_accuracy": row.get("evidence_only_accuracy"),
        "frozen_comparator_accuracy": row.get("frozen_comparator_accuracy"),
        "screen_checks": row.get("screen_checks"),
    }


def _broken_payload(budget: TournamentBudget, sanity: Mapping[str, object], started: float) -> Dict[str, object]:
    return {
        "stage": "8B.3",
        "status": "stopped_after_round0",
        "decision": "STAGE8B3_BENCHMARK_OR_REPRESENTATION_BROKEN",
        "no_stage8c_run": True,
        "no_10x_attention_capacity_claim": True,
        "budget": asdict(budget),
        "round0_sanity": sanity,
        "top3": [],
        "wall_clock_seconds": time.perf_counter() - started,
        "recommendation": "Fix the Stage 8 benchmark or representation before running further architecture evolution.",
    }


def _recommendation(decision: str, top3: Sequence[Mapping[str, object]]) -> str:
    if decision == "STAGE8B3_TOP3_SELECTED":
        names = ", ".join(str(row.get("architecture_name")) for row in top3)
        return f"Proceed to Stage 8B.4 larger-scale validation of the frozen top-3 configs only: {names}. Do not run Stage 8C yet."
    if decision == "STAGE8B3_PROMISING_BUT_NO_TOP3":
        return "Run a narrower Stage 8B.4 micro-tournament around the promising variants and missing controls."
    if decision == "STAGE8B3_BENCHMARK_OR_REPRESENTATION_BROKEN":
        return "Stop architecture evolution and repair sanity/oracle learnability first."
    return "Stop this line of architecture search unless the representation or evidence-use mechanism is redesigned."


def _load_baseline_capacity() -> int:
    if not BASELINE_CAPACITY_PATH.exists():
        return 0
    try:
        payload = json.loads(BASELINE_CAPACITY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    best = payload.get("best_monolithic_capacity", {})
    if isinstance(best, Mapping):
        return int(best.get("capacity", 0))
    return 0


def _reset_outputs() -> None:
    for path in (
        DATABASE_PATH,
        CONTROLS_AUDIT_PATH,
        CAPACITY_CURVES_PATH,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def _artifact_map() -> Dict[str, str]:
    return {
        "database": str(DATABASE_PATH),
        "round1": str(ROUND1_PATH),
        "round2": str(ROUND2_PATH),
        "round3": str(ROUND3_PATH),
        "round4": str(ROUND4_PATH),
        "top3_configs": str(TOP3_CONFIGS_PATH),
        "freeze_manifest": str(TOP3_FREEZE_MANIFEST_PATH),
        "controls_audit": str(CONTROLS_AUDIT_PATH),
        "view_diversity_audit": str(VIEW_DIVERSITY_AUDIT_PATH),
        "compute_audit": str(COMPUTE_AUDIT_PATH),
        "capacity_curves": str(CAPACITY_CURVES_PATH),
        "report": str(REPORT_PATH),
    }


def _brief_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    return [_brief_row(row) for row in rows]


def _brief_row(row: Mapping[str, object]) -> Dict[str, object]:
    return {
        "phase": row.get("phase"),
        "config_id": row.get("config_id"),
        "architecture_name": row.get("architecture_name"),
        "architecture_family": row.get("architecture_family"),
        "parent_config_id": row.get("parent_config_id"),
        "mutation_description": row.get("mutation_description"),
        "capacity_C": row.get("capacity_C"),
        "early_capacity_ratio": row.get("early_capacity_ratio"),
        "held_out_template_dev_accuracy": row.get("held_out_template_dev_accuracy"),
        "accuracy_by_N": row.get("accuracy_by_N"),
        "candidate_evidence_mismatch_degradation": row.get("candidate_evidence_mismatch_degradation"),
        "cross_task_evidence_shuffle_degradation": row.get("cross_task_evidence_shuffle_degradation"),
        "frozen_comparator_accuracy": row.get("frozen_comparator_accuracy"),
        "pairwise_hidden_state_similarity": row.get("pairwise_hidden_state_similarity"),
        "selection_score": row.get("selection_score"),
        "screen_checks": row.get("screen_checks"),
        "failure_reason": row.get("failure_reason"),
    }


def _accuracy_at(row: Mapping[str, object], n_blocks: int) -> float:
    values = row.get("accuracy_by_N", {})
    if isinstance(values, Mapping):
        return float(values.get(str(n_blocks), values.get(n_blocks, 0.0)))
    return 0.0


def _mean_by_n(rows: Sequence[Mapping[str, object]], key: str) -> Dict[str, float]:
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        grouped.setdefault(int(row.get("n_blocks", 0)), []).append(float(row.get(key, 0.0)))
    return {str(n): _mean(values) for n, values in sorted(grouped.items())}


def _std_by_n(rows: Sequence[Mapping[str, object]], key: str) -> Dict[str, float]:
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        grouped.setdefault(int(row.get("n_blocks", 0)), []).append(float(row.get(key, 0.0)))
    return {str(n): (pstdev(values) if len(values) > 1 else 0.0) for n, values in sorted(grouped.items())}


def _control_metric(controls: Mapping[str, object], control: str, key: str) -> float:
    row = controls.get(control, {})
    if isinstance(row, Mapping):
        return float(row.get(key, 0.0))
    return 0.0


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return mean(rows) if rows else 0.0


def _entropy(values: Sequence[float]) -> float:
    total = sum(float(value) for value in values)
    if total <= 0:
        return 0.0
    return -sum((float(value) / total) * math.log(max(1e-12, float(value) / total)) for value in values if value > 0)


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
