from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.datasets.arc_agi2_verification import (
    ArcPair,
    ArcVerificationDatasetConfig,
    ArcVerificationExample,
    apply_arc_control,
    candidate_metadata_only_accuracy,
    infer_transformation_type,
    load_arc_tasks,
)
from src.experiments.run_stage_arc1_1_empirical_search import (
    FeatureTrainingConfig,
    fit_feature_verifier,
    predict_feature_logits,
)
from src.experiments.run_stage_arc1_latent_rule_clones import (
    ArcModelConfig,
    ArcTrainingConfig,
    _accuracy,
    _bootstrap_ci,
    _candidate_stat_predictions,
    _changed_cell_count_predictions,
    _clear_cuda,
    _mean,
    _metric_block,
    _resolve_device,
    _shape,
    _softmax_np,
    _std,
    _top1,
    fit_arc_verifier,
    predict_logits,
)


BENCHMARK = "arc_hybrid1_empirical_latent_verifier_for_refinement_loop"
STAGE_NAME = "ARC-HYBRID-1_EMPIRICAL_LATENT_VERIFIER_FOR_REFINEMENT_LOOP"
DEFAULT_CONFIG = "configs/stage_arc_hybrid1_empirical_search_smoke.json"
DEFAULT_RESULTS = "results/arc_hybrid1_empirical_results.json"
DEFAULT_CONTROLS = "results/arc_hybrid1_controls.json"
DEFAULT_GENERATOR_AUDIT = "results/arc_hybrid1_candidate_generator_audit.json"
DEFAULT_LEAKAGE = "results/arc_hybrid1_leakage_audit.jsonl"
DEFAULT_LEADERBOARD = "results/arc_hybrid1_variant_leaderboard.csv"
DEFAULT_FAILURES = "results/arc_hybrid1_failure_taxonomy.json"
DEFAULT_OVERFIT = "results/arc_hybrid1_overfit_curves.json"
DEFAULT_ERRORS = "results/arc_hybrid1_error_cases.jsonl"
DEFAULT_ATTENTION = "results/arc_hybrid1_attention_summaries.jsonl"
DEFAULT_COMPUTE = "results/arc_hybrid1_compute_metrics.json"
DEFAULT_REPORT = "reports/STAGE_ARC_HYBRID1_EMPIRICAL_SEARCH.md"
SIMPLE_CROSS_ATTENTION_VARIANT = "A_output_grid_only_latent_cross_attention"


Grid = List[List[int]]


@dataclass(frozen=True)
class CandidateGeneratorConfig:
    num_candidates: int = 8
    natural_candidate_count: int = 8
    max_programs: int = 18
    mutation_rounds: int = 2
    refinement_branching: int = 2
    refinement_iterations: int = 1
    include_output_grid_mutations: bool = True
    include_symbolic_programs: bool = True
    select_by_train_fit_before_shuffle: bool = True


@dataclass(frozen=True)
class HybridVariant:
    name: str
    family: str
    representation: str
    view_set: Tuple[str, ...]
    interaction: str
    objective: str = "cross_entropy"
    model_kind: str = "rule_clone"
    candidate_count: int = 8
    coordination_blocks: int = 1
    uses_execution_evidence: bool = False
    uses_mismatch_evidence: bool = False
    uses_program_evidence: bool = False
    diagnostic_only: bool = False


@dataclass(frozen=True)
class ProgramSpec:
    name: str
    source_family: str
    program_length: int
    trace_depth: int
    apply: Callable[[Grid], Grid]


@dataclass(frozen=True)
class CandidateProposal:
    output: Grid
    source_family: str
    program_name: str
    train_execution_score: float
    mismatch_count: float
    mismatch_ratio: float
    program_length: int
    trace_depth: int
    refinement_depth: int = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage ARC-HYBRID-1 bounded empirical candidate/refinement verifier search.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--candidate-generator-audit-output", default=DEFAULT_GENERATOR_AUDIT)
    parser.add_argument("--leakage-output", default=DEFAULT_LEAKAGE)
    parser.add_argument("--leaderboard-output", default=DEFAULT_LEADERBOARD)
    parser.add_argument("--failure-output", default=DEFAULT_FAILURES)
    parser.add_argument("--overfit-output", default=DEFAULT_OVERFIT)
    parser.add_argument("--error-output", default=DEFAULT_ERRORS)
    parser.add_argument("--attention-output", default=DEFAULT_ATTENTION)
    parser.add_argument("--compute-output", default=DEFAULT_COMPUTE)
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
    result = run_arc_hybrid1_search(config, max_variants=int(args.max_variants))
    _write_outputs(
        result,
        Path(args.output),
        Path(args.controls_output),
        Path(args.candidate_generator_audit_output),
        Path(args.leakage_output),
        Path(args.leaderboard_output),
        Path(args.failure_output),
        Path(args.overfit_output),
        Path(args.error_output),
        Path(args.attention_output),
        Path(args.compute_output),
        Path(args.report),
    )


def run_arc_hybrid1_search(config: Dict[str, object], max_variants: int = 0) -> Dict[str, object]:
    dataset_config = _dataset_config(config.get("dataset", {}))
    generator_config = _generator_config(config.get("candidate_generator", {}), dataset_config.num_candidates)
    training = _training_config(config.get("training", {}))
    overfit_training = _training_config({**dict(config.get("training", {})), **dict(config.get("overfit_training", {}))})
    feature_training = _feature_training_config(config.get("feature_training", {}))
    seeds = [int(seed) for seed in config.get("seeds", [0])]
    device = _resolve_device(str(config.get("device", "auto")))
    variants = _variant_plan(config.get("variants"))
    if max_variants:
        variants = variants[:max_variants]
    if int(config.get("max_variants", 0)):
        variants = variants[: int(config.get("max_variants", 0))]
    run_overfit = bool(config.get("run_overfit_gates", True))
    run_phase2 = bool(config.get("run_phase2_natural", True))
    run_phase3 = bool(config.get("run_phase3_refinement", True))
    run_feature_mlp = bool(config.get("run_simple_stats_mlp_baseline", True))
    micro_gates = _micro_gates(config.get("micro_overfit_gates"))

    rows: List[Dict[str, object]] = []
    controls: List[Dict[str, object]] = []
    generator_audit_rows: List[Dict[str, object]] = []
    leakage_rows: List[Dict[str, object]] = []
    overfit_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []
    attention_rows: List[Dict[str, object]] = []
    compute_rows: List[Dict[str, object]] = []
    refinement_rows: List[Dict[str, object]] = []

    print(f"arc-hybrid1: device={device} variants={len(variants)} seeds={seeds}")
    for seed in seeds:
        phase1_config = replace(dataset_config, num_candidates=int(generator_config.num_candidates))
        phase1_splits = build_hybrid_candidate_splits(
            phase1_config,
            generator_config,
            seed=seed,
            force_gold=True,
            phase="phase1_gold_present",
        )
        natural_splits = (
            build_hybrid_candidate_splits(
                phase1_config,
                replace(generator_config, num_candidates=int(generator_config.natural_candidate_count)),
                seed=seed,
                force_gold=False,
                phase="phase2_natural_pool",
            )
            if run_phase2 or run_phase3
            else {"train": [], "dev": [], "test": []}
        )
        for split_name, examples in phase1_splits.items():
            generator_audit_rows.append(_candidate_generator_audit(seed, split_name, "phase1_gold_present", examples))
            leakage_rows.extend(_hybrid_leakage_audit_rows(examples, expect_exactly_one_gold=True, seed=seed))
        for split_name, examples in natural_splits.items():
            if examples:
                generator_audit_rows.append(_candidate_generator_audit(seed, split_name, "phase2_natural_pool", examples))
                leakage_rows.extend(_hybrid_leakage_audit_rows(examples, expect_exactly_one_gold=False, seed=seed))

        simple_cross_attention_cache: Dict[str, np.ndarray] = {}
        simple_stats_mlp_cache: Dict[str, np.ndarray] = {}
        if run_feature_mlp:
            feature_fit = fit_feature_verifier(
                phase1_splits["train"],
                phase1_splits["dev"],
                feature_training,
                seed=seed + 811,
                trainable=True,
                method="simple_stats_mlp_baseline",
            )
            simple_stats_mlp_cache["phase1_dev"] = predict_feature_logits(feature_fit.model, phase1_splits["dev"])
            if natural_splits["test"]:
                simple_stats_mlp_cache["phase2_test"] = predict_feature_logits(feature_fit.model, natural_splits["test"])

        for variant in variants:
            print(f"arc-hybrid1 variant={variant.name} seed={seed}: overfit gates")
            gate_rows = (
                _run_overfit_gates(variant, dataset_config, generator_config, overfit_training, seed, device, micro_gates)
                if run_overfit and not variant.diagnostic_only
                else [
                    {
                        "variant": variant.name,
                        "seed": seed,
                        "gate": "skipped",
                        "pass": True,
                        "gate_required": False,
                        "reason": "diagnostic_or_config_skip",
                    }
                ]
            )
            overfit_rows.extend(gate_rows)
            gates_pass = all(bool(row.get("pass")) for row in gate_rows if row.get("gate_required", True))
            row_base = {
                "benchmark": BENCHMARK,
                "stage_name": STAGE_NAME,
                "seed": seed,
                "variant": variant.name,
                "family": variant.family,
                "representation": variant.representation,
                "view_set": list(variant.view_set),
                "interaction": variant.interaction,
                "objective": variant.objective,
                "candidate_count": int(variant.candidate_count),
                "uses_execution_evidence": bool(variant.uses_execution_evidence),
                "uses_mismatch_evidence": bool(variant.uses_mismatch_evidence),
                "uses_program_evidence": bool(variant.uses_program_evidence),
                "diagnostic_only": bool(variant.diagnostic_only),
                "overfit_gates_pass": bool(gates_pass),
            }
            if not gates_pass:
                rows.append({**row_base, "status": "failed_overfit_gate", "failure_reason": _first_failed_gate(gate_rows)})
                continue

            model_config = _model_config_for_variant(variant)
            variant_training = replace(training, objective=variant.objective)
            print(f"arc-hybrid1 variant={variant.name} seed={seed}: phase1 trainable/frozen")
            start = time.perf_counter()
            trainable = fit_arc_verifier(
                phase1_splits["train"],
                phase1_splits["dev"],
                model_config,
                variant_training,
                seed=seed + 10_901,
                device=device,
                trainable_shared=True,
                method=f"trainable__{variant.name}",
            )
            frozen = fit_arc_verifier(
                phase1_splits["train"],
                phase1_splits["dev"],
                model_config,
                variant_training,
                seed=seed + 10_901,
                device=device,
                trainable_shared=False,
                method=f"frozen__{variant.name}",
            )
            phase1_dev_logits = predict_logits(trainable.model, phase1_splits["dev"], model_config, variant_training.batch_size, device)
            frozen_phase1_dev_logits = predict_logits(frozen.model, phase1_splits["dev"], model_config, variant_training.batch_size, device)
            phase1_labels = np.asarray([example.label for example in phase1_splits["dev"]], dtype=np.int64)
            elapsed = time.perf_counter() - start

            if variant.name == SIMPLE_CROSS_ATTENTION_VARIANT:
                simple_cross_attention_cache["phase1_dev"] = phase1_dev_logits

            control_row = _run_hybrid_controls(
                trainable.model,
                model_config,
                phase1_splits["dev"],
                phase1_dev_logits,
                phase1_labels,
                variant,
                variant_training.batch_size,
                device,
                seed,
            )
            controls.append(control_row)

            phase2_trainable = {}
            phase2_frozen = {}
            phase2_baselines: Dict[str, object] = {}
            phase2_logits = np.zeros((0, 0), dtype=np.float32)
            frozen_phase2_logits = np.zeros((0, 0), dtype=np.float32)
            if run_phase2 and natural_splits["test"]:
                phase2_logits = predict_logits(trainable.model, natural_splits["test"], model_config, variant_training.batch_size, device)
                frozen_phase2_logits = predict_logits(frozen.model, natural_splits["test"], model_config, variant_training.batch_size, device)
                phase2_trainable = _natural_metric_block(phase2_logits, natural_splits["test"])
                phase2_frozen = _natural_metric_block(frozen_phase2_logits, natural_splits["test"])
                phase2_baselines = _hybrid_baseline_block(
                    phase1_splits["train"],
                    phase1_splits["dev"],
                    natural_splits["test"],
                    simple_stats_logits=simple_stats_mlp_cache.get("phase2_test"),
                    simple_cross_attention_logits=simple_cross_attention_cache.get("phase2_test"),
                )
            if run_phase3 and natural_splits["test"]:
                refinement_rows.extend(
                    _run_refinement_comparison(
                        variant,
                        natural_splits["test"],
                        trainable.model,
                        frozen.model,
                        model_config,
                        variant_training.batch_size,
                        device,
                        generator_config,
                        seed,
                    )
                )

            phase1_baselines = _hybrid_baseline_block(
                phase1_splits["train"],
                phase1_splits["dev"],
                phase1_splits["dev"],
                labels=phase1_labels,
                simple_stats_logits=simple_stats_mlp_cache.get("phase1_dev"),
                simple_cross_attention_logits=simple_cross_attention_cache.get("phase1_dev"),
            )
            trainable_block = _metric_block(phase1_dev_logits, phase1_labels, phase1_splits["dev"])
            frozen_block = _metric_block(frozen_phase1_dev_logits, phase1_labels, phase1_splits["dev"])
            row = {
                **row_base,
                "status": "completed",
                "phase1_gold_present": {
                    "trainable": trainable_block,
                    "frozen": frozen_block,
                    "delta_trainable_minus_frozen": float(trainable_block["top1"] - frozen_block["top1"]),
                    "baselines": phase1_baselines,
                },
                "phase2_natural_pool": {
                    "trainable": phase2_trainable,
                    "frozen": phase2_frozen,
                    "delta_conditional_trainable_minus_frozen": float(
                        phase2_trainable.get("conditional_top1_when_gold_present", 0.0)
                        - phase2_frozen.get("conditional_top1_when_gold_present", 0.0)
                    )
                    if phase2_trainable
                    else 0.0,
                    "baselines": phase2_baselines,
                    "generator_recall": float(_generator_recall(natural_splits["test"])),
                    "overall_solve_rate": float(phase2_trainable.get("overall_solve_rate", 0.0)) if phase2_trainable else 0.0,
                },
                "controls_key": {"variant": variant.name, "seed": seed},
                "control_pass": control_row["control_pass"],
                "leakage_audit_pass": all(bool(row.get("pass")) for row in leakage_rows if row.get("seed") == seed),
                "trainable_audit": trainable.audit,
                "frozen_audit": frozen.audit,
                "training_time_seconds": float(elapsed),
                "param_count": int(trainable.param_count),
                "medium_validation_triggered": False,
                "final_validation_launched": False,
            }
            rows.append(row)
            error_rows.extend(_error_rows(variant, seed, phase1_splits["dev"], phase1_dev_logits, frozen_phase1_dev_logits, limit=12))
            if phase2_logits.size:
                error_rows.extend(_natural_error_rows(variant, seed, natural_splits["test"], phase2_logits, frozen_phase2_logits, limit=12))
            attention_rows.extend(
                _attention_summaries(
                    trainable.model,
                    model_config,
                    phase1_splits["dev"][: min(4, len(phase1_splits["dev"]))],
                    variant_training.batch_size,
                    device,
                    seed,
                    variant.name,
                )
            )
            compute_rows.append(
                {
                    "benchmark": BENCHMARK,
                    "seed": seed,
                    "variant": variant.name,
                    "candidate_count": int(variant.candidate_count),
                    "rule_clones": len(model_config.view_names),
                    "coordination_blocks": int(model_config.coordination_blocks),
                    "training_time_seconds": float(elapsed),
                    "trainable_param_count": int(trainable.param_count),
                    "frozen_param_count": int(frozen.param_count),
                    "approx_candidate_evaluations_phase1": int(len(phase1_splits["train"]) + len(phase1_splits["dev"]))
                    * int(variant.candidate_count),
                    "approx_candidate_evaluations_phase2": int(len(natural_splits.get("test", []))) * int(generator_config.natural_candidate_count),
                }
            )
            del trainable, frozen
            _clear_cuda()

    leaderboard = _leaderboard(rows, controls)
    failure_taxonomy = _failure_taxonomy(rows, overfit_rows, controls, generator_audit_rows)
    summary = _summary(rows, controls, leaderboard, generator_audit_rows, refinement_rows)
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "stage_name": STAGE_NAME,
            "created_at_utc": _now(),
            "device": device,
            "dataset_root": dataset_config.dataset_root,
            "scope": "bounded empirical candidate/refinement verifier search; final validation is gated and not launched",
            "central_hypothesis": [
                "generate/refine many candidates",
                "execute/test candidates on visible train pairs",
                "expose rich non-oracle evidence traces",
                "use shared-weight latent clone coordination to rank/refine candidates",
            ],
            "claim_boundary": "No ARC-AGI-2 solving, SOTA, autonomous reasoning, or general abstraction claim is made.",
            "final_validation_launched": False,
        },
        "dataset_config": asdict(dataset_config),
        "candidate_generator_config": asdict(generator_config),
        "training_config": asdict(training),
        "feature_training_config": asdict(feature_training),
        "variant_plan": [asdict(variant) for variant in variants],
        "rows": rows,
        "controls": controls,
        "candidate_generator_audit": generator_audit_rows,
        "leakage_audit": leakage_rows,
        "leaderboard": leaderboard,
        "failure_taxonomy": failure_taxonomy,
        "overfit_curves": overfit_rows,
        "error_cases": error_rows,
        "attention_summaries": attention_rows,
        "compute_metrics": compute_rows,
        "refinement_results": refinement_rows,
        "summary": summary,
    }


def build_hybrid_candidate_splits(
    config: ArcVerificationDatasetConfig,
    generator_config: CandidateGeneratorConfig,
    seed: int,
    force_gold: bool,
    phase: str,
) -> Dict[str, List[ArcVerificationExample]]:
    tasks = load_arc_tasks(config)
    usable = [task for task in tasks if task["test_pairs"]]
    rng = np.random.default_rng(seed)
    order = np.array([task["task_id"] for task in usable], dtype=object)
    rng.shuffle(order)
    by_id = {task["task_id"]: task for task in usable}
    n_train = min(int(config.train_tasks), len(order))
    n_dev = min(int(config.dev_tasks), max(0, len(order) - n_train))
    n_test = min(int(config.test_tasks), max(0, len(order) - n_train - n_dev))
    split_ids = {
        "train": list(order[:n_train]),
        "dev": list(order[n_train : n_train + n_dev]),
        "test": list(order[n_train + n_dev : n_train + n_dev + n_test]),
    }
    splits: Dict[str, List[ArcVerificationExample]] = {"train": [], "dev": [], "test": []}
    for split, ids in split_ids.items():
        for task_id in ids:
            task = by_id[str(task_id)]
            train_pairs = tuple(task["train_pairs"][: int(config.max_train_pairs)])
            for test_index, test_pair in enumerate(task["test_pairs"]):
                example_seed = seed + _stable_int(f"{phase}:{task_id}:{test_index}") % 1_000_000_000
                example_rng = np.random.default_rng(example_seed)
                proposals = generate_candidate_pool(train_pairs, test_pair.input, generator_config, example_rng)
                example = materialize_candidate_example(
                    task_id=str(task_id),
                    split=split,
                    test_index=test_index,
                    train_pairs=train_pairs,
                    test_input=test_pair.input,
                    gold_output=test_pair.output,
                    proposals=proposals,
                    num_candidates=int(generator_config.num_candidates),
                    rng=example_rng,
                    force_gold=force_gold,
                    phase=phase,
                    generator_version="hybrid1_symbolic_programs_plus_output_mutations_v1",
                )
                splits[split].append(example)
    return splits


def generate_candidate_pool(
    train_pairs: Sequence[ArcPair],
    test_input: Grid,
    config: CandidateGeneratorConfig,
    rng: np.random.Generator,
) -> List[CandidateProposal]:
    proposals: List[CandidateProposal] = []
    seen: set[str] = set()
    if config.include_symbolic_programs:
        for program in _program_specs(train_pairs, rng)[: int(config.max_programs)]:
            proposal = _proposal_from_program(program, train_pairs, test_input)
            _append_unique_proposal(proposals, seen, proposal)
    if config.include_output_grid_mutations:
        seeds = list(proposals[: max(1, min(len(proposals), int(config.max_programs)))])
        if not seeds:
            identity = ProgramSpec("identity", "symbolic_identity", 1, 1, lambda grid: _grid_copy(grid))
            seeds = [_proposal_from_program(identity, train_pairs, test_input)]
        for round_index in range(int(config.mutation_rounds)):
            for proposal in list(seeds):
                for mutation_index in range(3):
                    mutated = _mutate_candidate_output(proposal.output, rng, round_index + mutation_index)
                    evidence = _output_only_evidence(train_pairs, test_input, mutated)
                    _append_unique_proposal(
                        proposals,
                        seen,
                        CandidateProposal(
                            output=mutated,
                            source_family="output_grid_mutation",
                            program_name=f"mutate:{proposal.program_name}",
                            train_execution_score=evidence["train_execution_score"],
                            mismatch_count=evidence["mismatch_count"],
                            mismatch_ratio=evidence["mismatch_ratio"],
                            program_length=proposal.program_length + 1,
                            trace_depth=proposal.trace_depth + 1,
                            refinement_depth=proposal.refinement_depth + 1,
                        ),
                    )
    while len(proposals) < max(int(config.num_candidates), 2):
        base = test_input if not proposals else proposals[int(rng.integers(0, len(proposals)))].output
        random_grid = _random_palette_grid_like(base, train_pairs, rng)
        evidence = _output_only_evidence(train_pairs, test_input, random_grid)
        _append_unique_proposal(
            proposals,
            seen,
            CandidateProposal(
                output=random_grid,
                source_family="random_palette_fill",
                program_name="random_palette_fill",
                train_execution_score=evidence["train_execution_score"],
                mismatch_count=evidence["mismatch_count"],
                mismatch_ratio=evidence["mismatch_ratio"],
                program_length=1,
                trace_depth=1,
            ),
        )
    return proposals


def materialize_candidate_example(
    task_id: str,
    split: str,
    test_index: int,
    train_pairs: Sequence[ArcPair],
    test_input: Grid,
    gold_output: Grid,
    proposals: Sequence[CandidateProposal],
    num_candidates: int,
    rng: np.random.Generator,
    force_gold: bool,
    phase: str,
    generator_version: str,
) -> ArcVerificationExample:
    selected = _select_candidate_proposals(proposals, max(1, int(num_candidates) - (1 if force_gold else 0)))
    if force_gold:
        selected = [proposal for proposal in selected if not _grid_equal(proposal.output, gold_output)]
        selected = selected[: max(0, int(num_candidates) - 1)]
        gold_evidence = _output_only_evidence(train_pairs, test_input, gold_output)
        selected.append(
            CandidateProposal(
                output=_grid_copy(gold_output),
                source_family="forced_gold_control",
                program_name="forced_gold_candidate_object",
                train_execution_score=gold_evidence["train_execution_score"],
                mismatch_count=gold_evidence["mismatch_count"],
                mismatch_ratio=gold_evidence["mismatch_ratio"],
                program_length=0,
                trace_depth=0,
            )
        )
    while len(selected) < int(num_candidates):
        fallback = _random_palette_grid_like(test_input, train_pairs, rng)
        evidence = _output_only_evidence(train_pairs, test_input, fallback)
        selected.append(
            CandidateProposal(
                output=fallback,
                source_family="fallback_random_palette_fill",
                program_name="fallback_random_palette_fill",
                train_execution_score=evidence["train_execution_score"],
                mismatch_count=evidence["mismatch_count"],
                mismatch_ratio=evidence["mismatch_ratio"],
                program_length=1,
                trace_depth=1,
            )
        )
    order = rng.permutation(len(selected))
    shuffled = [selected[int(index)] for index in order[: int(num_candidates)]]
    candidates = tuple(_grid_copy(proposal.output) for proposal in shuffled)
    correct_indices = [index for index, candidate in enumerate(candidates) if _grid_equal(candidate, gold_output)]
    label = int(correct_indices[0]) if correct_indices else 0
    metadata = _metadata_for_candidates(
        train_pairs=train_pairs,
        test_input=test_input,
        gold_output=gold_output,
        candidates=candidates,
        proposals=shuffled,
        force_gold=force_gold,
        phase=phase,
        generator_version=generator_version,
    )
    return ArcVerificationExample(
        id=f"{task_id}__test{test_index}",
        task_id=str(task_id),
        split=split,
        train_pairs=tuple(train_pairs),
        test_input=_grid_copy(test_input),
        gold_output=_grid_copy(gold_output),
        candidates=candidates,
        label=label,
        negative_types=tuple("candidate" for _ in candidates),
        public_source_tags=tuple("candidate_grid" for _ in candidates),
        metadata=metadata,
    )


def _metadata_for_candidates(
    train_pairs: Sequence[ArcPair],
    test_input: Grid,
    gold_output: Grid,
    candidates: Sequence[Grid],
    proposals: Sequence[CandidateProposal],
    force_gold: bool,
    phase: str,
    generator_version: str,
) -> Dict[str, object]:
    correct = [bool(_grid_equal(candidate, gold_output)) for candidate in candidates]
    mismatch_stats = [_candidate_mismatch_stats(train_pairs, test_input, proposal.output) for proposal in proposals]
    return {
        "stage": STAGE_NAME,
        "phase": phase,
        "candidate_count": len(candidates),
        "forced_gold_inserted": bool(force_gold),
        "offline_gold_present": any(correct),
        "offline_correct_candidate_indices": [index for index, value in enumerate(correct) if value],
        "offline_candidate_test_correct": correct,
        "candidate_generator_version": generator_version,
        "candidate_order_randomized": True,
        "model_visible_candidate_sources": False,
        "model_visible_generator_rank": False,
        "model_visible_hidden_oracle_score": False,
        "test_output_visible_outside_candidate_objects": False,
        "candidate_source_families_external": [proposal.source_family for proposal in proposals],
        "candidate_program_names_external": [proposal.program_name for proposal in proposals],
        "candidate_train_execution_scores": [float(proposal.train_execution_score) for proposal in proposals],
        "candidate_mismatch_counts": [float(proposal.mismatch_count) for proposal in proposals],
        "candidate_mismatch_ratios": [float(proposal.mismatch_ratio) for proposal in proposals],
        "candidate_program_lengths": [int(proposal.program_length) for proposal in proposals],
        "candidate_trace_depths": [int(proposal.trace_depth) for proposal in proposals],
        "candidate_refinement_depths": [int(proposal.refinement_depth) for proposal in proposals],
        "candidate_mismatch_stats_external": mismatch_stats,
        "candidate_diversity": _candidate_diversity(candidates),
        "num_train_pairs": len(train_pairs),
        "palette_size": len(_palette(gold_output)),
        "transformation_type": infer_transformation_type(train_pairs, test_input, gold_output),
    }


def apply_hybrid_control(
    examples: Sequence[ArcVerificationExample],
    control: str,
    seed: int,
) -> List[ArcVerificationExample]:
    rng = np.random.default_rng(seed)
    if control == "candidate_order_shuffle_with_gold_remap":
        return [_candidate_order_shuffle_with_metadata(example, rng) for example in examples]
    if control in {"candidate_only", "examples_only_no_candidates"}:
        return [_strip_visible_candidate_evidence(example) for example in apply_arc_control(examples, control, seed)]
    if control in {"train_pair_shuffle", "cross_task_train_pair_shuffle", "candidate_evidence_mismatch"}:
        return [_recompute_visible_evidence(example) for example in apply_arc_control(examples, control, seed)]
    if control == "execution_trace_mismatch":
        return _shuffle_candidate_metadata_arrays(examples, rng, ("candidate_train_execution_scores", "candidate_trace_depths"))
    if control == "mismatch_map_mismatch":
        return _shuffle_candidate_metadata_arrays(examples, rng, ("candidate_mismatch_counts", "candidate_mismatch_ratios", "candidate_mismatch_stats_external"))
    if control == "program_output_mismatch":
        return _shuffle_candidate_metadata_arrays(examples, rng, ("candidate_program_lengths", "candidate_trace_depths", "candidate_program_names_external"))
    return apply_arc_control(examples, control, seed)


def _candidate_order_shuffle_with_metadata(example: ArcVerificationExample, rng: np.random.Generator) -> ArcVerificationExample:
    order = rng.permutation(len(example.candidates))
    if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
        order = np.roll(order, 1)
    metadata = dict(example.metadata)
    for key in (
        "offline_candidate_test_correct",
        "candidate_source_families_external",
        "candidate_program_names_external",
        "candidate_train_execution_scores",
        "candidate_mismatch_counts",
        "candidate_mismatch_ratios",
        "candidate_program_lengths",
        "candidate_trace_depths",
        "candidate_refinement_depths",
        "candidate_mismatch_stats_external",
    ):
        values = metadata.get(key)
        if isinstance(values, list) and len(values) == len(order):
            metadata[key] = [values[int(index)] for index in order]
    label = int(np.where(order == int(example.label))[0][0])
    metadata["offline_correct_candidate_indices"] = [
        index for index, is_correct in enumerate(metadata.get("offline_candidate_test_correct", [])) if bool(is_correct)
    ]
    metadata["control"] = "candidate_order_shuffle_with_gold_remap"
    return replace(
        example,
        candidates=tuple(example.candidates[int(index)] for index in order),
        label=label,
        negative_types=tuple(example.negative_types[int(index)] for index in order),
        public_source_tags=tuple(example.public_source_tags[int(index)] for index in order),
        metadata=metadata,
    )


def _strip_visible_candidate_evidence(example: ArcVerificationExample) -> ArcVerificationExample:
    metadata = dict(example.metadata)
    candidate_count = len(example.candidates)
    metadata.update(
        {
            "candidate_visible_evidence_stripped": True,
            "candidate_train_execution_scores": [0.0] * candidate_count,
            "candidate_mismatch_counts": [31.0] * candidate_count,
            "candidate_mismatch_ratios": [1.0] * candidate_count,
            "candidate_program_lengths": [0] * candidate_count,
            "candidate_trace_depths": [0] * candidate_count,
        }
    )
    return replace(example, metadata=metadata)


def _recompute_visible_evidence(example: ArcVerificationExample) -> ArcVerificationExample:
    metadata = dict(example.metadata)
    scores = []
    counts = []
    ratios = []
    stats = []
    for candidate in example.candidates:
        evidence = _output_only_evidence(example.train_pairs, example.test_input, candidate)
        scores.append(float(evidence["train_execution_score"]))
        counts.append(float(evidence["mismatch_count"]))
        ratios.append(float(evidence["mismatch_ratio"]))
        stats.append(_candidate_mismatch_stats(example.train_pairs, example.test_input, candidate))
    metadata.update(
        {
            "candidate_train_execution_scores": scores,
            "candidate_mismatch_counts": counts,
            "candidate_mismatch_ratios": ratios,
            "candidate_mismatch_stats_external": stats,
            "visible_evidence_recomputed_for_control": True,
        }
    )
    return replace(example, metadata=metadata)


def _shuffle_candidate_metadata_arrays(
    examples: Sequence[ArcVerificationExample],
    rng: np.random.Generator,
    keys: Sequence[str],
) -> List[ArcVerificationExample]:
    if len(examples) < 2:
        return list(examples)
    pools: Dict[str, List[object]] = {}
    for key in keys:
        pools[key] = [example.metadata.get(key) for example in examples]
        rng.shuffle(pools[key])
    out = []
    for index, example in enumerate(examples):
        metadata = dict(example.metadata)
        for key in keys:
            values = pools[key][index]
            if isinstance(values, list):
                metadata[key] = list(values)
        metadata["control"] = "metadata_evidence_mismatch"
        out.append(replace(example, metadata=metadata))
    return out


def _run_overfit_gates(
    variant: HybridVariant,
    dataset_config: ArcVerificationDatasetConfig,
    generator_config: CandidateGeneratorConfig,
    overfit_training: ArcTrainingConfig,
    seed: int,
    device: str,
    gates: Sequence[Tuple[str, int, int, float]],
) -> List[Dict[str, object]]:
    rows = []
    for gate_name, candidate_count, example_count, threshold in gates:
        gate_dataset = replace(
            dataset_config,
            num_candidates=int(candidate_count),
            train_tasks=max(int(dataset_config.train_tasks), int(example_count) + 4),
            dev_tasks=2,
            test_tasks=2,
        )
        gate_generator = replace(generator_config, num_candidates=int(candidate_count))
        splits = build_hybrid_candidate_splits(
            gate_dataset,
            gate_generator,
            seed=seed + candidate_count * 1009,
            force_gold=True,
            phase=f"overfit_{gate_name}",
        )
        train_examples = splits["train"][: int(example_count)]
        if len(train_examples) < int(example_count):
            rows.append(
                {
                    "variant": variant.name,
                    "seed": seed,
                    "gate": gate_name,
                    "candidate_count": int(candidate_count),
                    "examples": len(train_examples),
                    "threshold": float(threshold),
                    "train_accuracy": 0.0,
                    "pass": False,
                    "gate_required": True,
                    "reason": "not_enough_examples",
                }
            )
            break
        model_config = _model_config_for_variant(replace(variant, candidate_count=int(candidate_count)))
        training = replace(
            overfit_training,
            batch_size=min(max(1, int(example_count)), int(overfit_training.batch_size)),
            objective=variant.objective,
        )
        fit = fit_arc_verifier(
            train_examples,
            train_examples,
            model_config,
            training,
            seed=seed + 91_000 + int(candidate_count),
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
                "candidate_count": int(candidate_count),
                "examples": len(train_examples),
                "threshold": float(threshold),
                "train_accuracy": float(acc),
                "pass": bool(acc >= float(threshold)),
                "gate_required": True,
                "history": fit.history,
                "audit": fit.audit,
            }
        )
        if acc < float(threshold):
            break
    return rows


def _run_hybrid_controls(
    model,
    model_config: ArcModelConfig,
    examples: Sequence[ArcVerificationExample],
    clean_logits: np.ndarray,
    labels: np.ndarray,
    variant: HybridVariant,
    batch_size: int,
    device: str,
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
    for name in (
        "candidate_only",
        "examples_only_no_candidates",
        "train_pair_shuffle",
        "cross_task_train_pair_shuffle",
        "candidate_evidence_mismatch",
        "execution_trace_mismatch",
        "mismatch_map_mismatch",
        "program_output_mismatch",
        "candidate_order_shuffle_with_gold_remap",
        "randomized_labels",
    ):
        controlled = apply_hybrid_control(examples, name, seed + 17_000 + len(controls))
        controlled_labels = np.asarray([example.label for example in controlled], dtype=np.int64)
        logits = predict_logits(model, controlled, model_config, batch_size, device, condition="none", seed=seed)
        controls[name] = _metric_block(logits, controlled_labels, controlled)
    role_logits = predict_logits(model, examples, model_config, batch_size, device, condition="physical_role_order_shuffle", seed=seed + 44_000)
    hidden_logits = predict_logits(model, examples, model_config, batch_size, device, condition="hidden_state_shuffle", seed=seed + 55_000)
    controls["physical_role_order_shuffle"] = _metric_block(role_logits, labels, examples)
    controls["hidden_state_shuffle"] = _metric_block(hidden_logits, labels, examples)
    controls["candidate_order_invariance_delta"] = abs(
        float(_top1(clean_logits, labels)) - float(controls["candidate_order_shuffle_with_gold_remap"]["top1"])
    )
    controls["role_order_invariance_delta"] = abs(float(_top1(clean_logits, labels)) - float(_top1(role_logits, labels)))
    controls["control_pass"] = {
        "candidate_only_near_chance": float(controls["candidate_only"]["top1"]) <= chance + 0.10,
        "examples_only_near_chance": float(controls["examples_only_no_candidates"]["top1"]) <= chance + 0.10,
        "metadata_only_near_chance": float(controls["candidate_source_metadata_only_accuracy"]) <= chance + 0.10,
        "train_pair_shuffle_degrades": float(controls["train_pair_shuffle"]["top1"]) <= chance + 0.15,
        "cross_task_train_pair_shuffle_degrades": float(controls["cross_task_train_pair_shuffle"]["top1"]) <= chance + 0.15,
        "candidate_evidence_mismatch_degrades": float(controls["candidate_evidence_mismatch"]["top1"]) <= chance + 0.15,
        "execution_trace_mismatch_degrades": (
            float(controls["execution_trace_mismatch"]["top1"]) <= chance + 0.15 if variant.uses_execution_evidence else True
        ),
        "mismatch_map_mismatch_degrades": (
            float(controls["mismatch_map_mismatch"]["top1"]) <= chance + 0.15 if variant.uses_mismatch_evidence else True
        ),
        "program_output_mismatch_degrades": (
            float(controls["program_output_mismatch"]["top1"]) <= chance + 0.15 if variant.uses_program_evidence else True
        ),
        "candidate_order_remap_invariance": float(controls["candidate_order_invariance_delta"]) <= 0.08,
        "role_order_invariance": float(controls["role_order_invariance_delta"]) <= 0.08,
        "hidden_state_shuffle_collapses": float(controls["hidden_state_shuffle"]["top1"]) <= max(chance + 0.20, float(_top1(clean_logits, labels)) - 0.05),
        "randomized_labels_collapses": float(controls["randomized_labels"]["top1"]) <= chance + 0.15,
    }
    controls["control_pass"]["overall"] = all(bool(value) for value in controls["control_pass"].values())
    return controls


def _run_refinement_comparison(
    variant: HybridVariant,
    examples: Sequence[ArcVerificationExample],
    trainable_model,
    frozen_model,
    model_config: ArcModelConfig,
    batch_size: int,
    device: str,
    generator_config: CandidateGeneratorConfig,
    seed: int,
) -> List[Dict[str, object]]:
    rows = []
    rng = np.random.default_rng(seed + 70_000)
    trainable_logits = predict_logits(trainable_model, examples, model_config, batch_size, device)
    frozen_logits = predict_logits(frozen_model, examples, model_config, batch_size, device)
    strategies = {
        "random_refinement": None,
        "heuristic_train_fit_refinement": None,
        "frozen_verifier_guided_refinement": frozen_logits,
        "trainable_latent_verifier_guided_refinement": trainable_logits,
    }
    for strategy, logits in strategies.items():
        refined = []
        for index, example in enumerate(examples):
            refined.append(
                refine_candidate_example(
                    example,
                    strategy=strategy,
                    logits=logits[index] if logits is not None and logits.size else None,
                    generator_config=generator_config,
                    rng=rng,
                )
            )
        refined_logits = predict_logits(trainable_model, refined, model_config, batch_size, device)
        before = _natural_metric_block(trainable_logits, examples)
        after = _natural_metric_block(refined_logits, refined)
        rows.append(
            {
                "benchmark": BENCHMARK,
                "variant": variant.name,
                "seed": seed,
                "strategy": strategy,
                "iterations": int(generator_config.refinement_iterations),
                "candidate_count": int(generator_config.num_candidates),
                "generator_recall_before": float(before["generator_recall"]),
                "generator_recall_after": float(after["generator_recall"]),
                "overall_solve_rate_before": float(before["overall_solve_rate"]),
                "overall_solve_rate_after": float(after["overall_solve_rate"]),
                "candidate_quality_before": _mean([_max_train_execution_score(example) for example in examples]),
                "candidate_quality_after": _mean([_max_train_execution_score(example) for example in refined]),
                "compute_cost_candidate_evaluations": int(len(examples)) * int(generator_config.num_candidates),
            }
        )
    return rows


def refine_candidate_example(
    example: ArcVerificationExample,
    strategy: str,
    logits: np.ndarray | None,
    generator_config: CandidateGeneratorConfig,
    rng: np.random.Generator,
) -> ArcVerificationExample:
    proposals = _proposals_from_example(example)
    selected_indices = _refinement_seed_indices(example, strategy, logits, rng, int(generator_config.refinement_branching))
    for index in selected_indices:
        if not (0 <= index < len(example.candidates)):
            continue
        for branch in range(int(generator_config.refinement_branching)):
            output = _mutate_candidate_output(example.candidates[index], rng, branch)
            evidence = _output_only_evidence(example.train_pairs, example.test_input, output)
            proposals.append(
                CandidateProposal(
                    output=output,
                    source_family="verifier_guided_refinement_mutation",
                    program_name=f"{strategy}:mutate_candidate_{index}",
                    train_execution_score=evidence["train_execution_score"],
                    mismatch_count=evidence["mismatch_count"],
                    mismatch_ratio=evidence["mismatch_ratio"],
                    program_length=int(_metadata_index(example, "candidate_program_lengths", index, 1)) + 1,
                    trace_depth=int(_metadata_index(example, "candidate_trace_depths", index, 1)) + 1,
                    refinement_depth=int(_metadata_index(example, "candidate_refinement_depths", index, 0)) + 1,
                )
            )
    selected = _select_candidate_proposals(proposals, int(generator_config.num_candidates))
    return materialize_candidate_example(
        task_id=example.task_id,
        split=example.split,
        test_index=0,
        train_pairs=example.train_pairs,
        test_input=example.test_input,
        gold_output=example.gold_output,
        proposals=selected,
        num_candidates=int(generator_config.num_candidates),
        rng=rng,
        force_gold=False,
        phase="phase3_refined_pool",
        generator_version="hybrid1_verifier_guided_refinement_v1",
    )


def _refinement_seed_indices(
    example: ArcVerificationExample,
    strategy: str,
    logits: np.ndarray | None,
    rng: np.random.Generator,
    count: int,
) -> List[int]:
    n = len(example.candidates)
    if n == 0:
        return []
    if strategy == "random_refinement" or logits is None:
        return [int(index) for index in rng.choice(n, size=min(count, n), replace=False)]
    if strategy == "heuristic_train_fit_refinement":
        scores = np.asarray(example.metadata.get("candidate_train_execution_scores", [0.0] * n), dtype=np.float64)
    else:
        scores = np.asarray(logits, dtype=np.float64)
    return [int(index) for index in np.argsort(-scores)[: min(count, n)]]


def _hybrid_baseline_block(
    train_examples: Sequence[ArcVerificationExample],
    dev_examples: Sequence[ArcVerificationExample],
    examples: Sequence[ArcVerificationExample],
    labels: np.ndarray | None = None,
    simple_stats_logits: np.ndarray | None = None,
    simple_cross_attention_logits: np.ndarray | None = None,
) -> Dict[str, object]:
    del train_examples, dev_examples
    if not examples:
        return {}
    if labels is None:
        block_fn = lambda preds: _natural_prediction_metric(preds, examples)
    else:
        block_fn = lambda preds: _accuracy(np.asarray(preds, dtype=np.int64), labels)
    n = len(examples[0].candidates)
    execution_pred = _execution_score_predictions(examples)
    mismatch_pred = _mismatch_count_predictions(examples)
    train_fit_pred = execution_pred
    candidate_stat_pred = _candidate_stat_predictions(examples)
    changed_count_pred = _changed_cell_count_predictions(examples)
    out: Dict[str, object] = {
        "random": 1.0 / max(1, n),
        "candidate_order_index0": block_fn(np.zeros(len(examples), dtype=np.int64)),
        "candidate_only_stat_heuristic": block_fn(candidate_stat_pred),
        "metadata_source_only": candidate_metadata_only_accuracy(examples),
        "examples_only_no_candidate": 1.0 / max(1, n),
        "execution_score_heuristic": block_fn(execution_pred),
        "mismatch_count_heuristic": block_fn(mismatch_pred),
        "train_fit_heuristic": block_fn(train_fit_pred),
        "changed_cell_count_heuristic": block_fn(changed_count_pred),
        "text_verbal_verifier": "not_run_no_local_text_verifier_configured",
    }
    if simple_stats_logits is not None and simple_stats_logits.size:
        out["simple_stats_mlp"] = block_fn(np.argmax(simple_stats_logits, axis=1))
    else:
        out["simple_stats_mlp"] = "not_run"
    if simple_cross_attention_logits is not None and simple_cross_attention_logits.size:
        out["simple_cross_attention_verifier"] = block_fn(np.argmax(simple_cross_attention_logits, axis=1))
    else:
        out["simple_cross_attention_verifier"] = "not_run"
    return out


def _natural_metric_block(logits: np.ndarray, examples: Sequence[ArcVerificationExample]) -> Dict[str, object]:
    if logits.size == 0:
        return {
            "top1": 0.0,
            "top2": 0.0,
            "top3": 0.0,
            "mrr": 0.0,
            "mean_gold_rank": 0.0,
            "gold_ranks": [],
            "generator_recall": 0.0,
            "conditional_top1_when_gold_present": 0.0,
            "overall_solve_rate": 0.0,
        }
    order = np.argsort(-logits, axis=1)
    ranks = []
    top1_hits = []
    top2_hits = []
    top3_hits = []
    conditional_hits = []
    for row, example in zip(order, examples):
        correct_indices = [int(index) for index in example.metadata.get("offline_correct_candidate_indices", [])]
        if not correct_indices:
            ranks.append(len(example.candidates) + 1)
            top1_hits.append(False)
            top2_hits.append(False)
            top3_hits.append(False)
            continue
        rank = min(int(np.where(row == correct)[0][0]) + 1 for correct in correct_indices)
        ranks.append(rank)
        hit1 = int(row[0]) in correct_indices
        top1_hits.append(hit1)
        top2_hits.append(any(int(index) in correct_indices for index in row[:2]))
        top3_hits.append(any(int(index) in correct_indices for index in row[:3]))
        conditional_hits.append(hit1)
    recall = _generator_recall(examples)
    return {
        "top1": float(np.mean(top1_hits)) if top1_hits else 0.0,
        "top2": float(np.mean(top2_hits)) if top2_hits else 0.0,
        "top3": float(np.mean(top3_hits)) if top3_hits else 0.0,
        "mrr": float(np.mean([0.0 if rank > len(example.candidates) else 1.0 / rank for rank, example in zip(ranks, examples)])) if ranks else 0.0,
        "mean_gold_rank": float(np.mean(ranks)) if ranks else 0.0,
        "gold_ranks": ranks,
        "generator_recall": float(recall),
        "conditional_top1_when_gold_present": float(np.mean(conditional_hits)) if conditional_hits else 0.0,
        "overall_solve_rate": float(np.mean(top1_hits)) if top1_hits else 0.0,
        "by_grid_size": _natural_accuracy_by_key(order[:, 0], examples, [f"{_shape(e.gold_output)[0]}x{_shape(e.gold_output)[1]}" for e in examples]),
        "by_train_examples": _natural_accuracy_by_key(order[:, 0], examples, [str(len(e.train_pairs)) for e in examples]),
        "by_palette_size": _natural_accuracy_by_key(order[:, 0], examples, [str(len(_palette(e.gold_output))) for e in examples]),
        "by_candidate_count": _natural_accuracy_by_key(order[:, 0], examples, [str(len(e.candidates)) for e in examples]),
        "by_candidate_source_family_external": _natural_accuracy_by_predicted_source(order[:, 0], examples),
        "by_program_length": _natural_accuracy_by_candidate_metadata(order[:, 0], examples, "candidate_program_lengths"),
        "by_trace_depth": _natural_accuracy_by_candidate_metadata(order[:, 0], examples, "candidate_trace_depths"),
    }


def _natural_prediction_metric(predictions: Sequence[int], examples: Sequence[ArcVerificationExample]) -> float:
    hits = []
    for prediction, example in zip(predictions, examples):
        correct = {int(index) for index in example.metadata.get("offline_correct_candidate_indices", [])}
        hits.append(int(prediction) in correct)
    return float(np.mean(hits)) if hits else 0.0


def _execution_score_predictions(examples: Sequence[ArcVerificationExample]) -> np.ndarray:
    preds = []
    for example in examples:
        scores = np.asarray(example.metadata.get("candidate_train_execution_scores", [0.0] * len(example.candidates)), dtype=np.float64)
        preds.append(int(np.argmax(scores)) if scores.size else 0)
    return np.asarray(preds, dtype=np.int64)


def _mismatch_count_predictions(examples: Sequence[ArcVerificationExample]) -> np.ndarray:
    preds = []
    for example in examples:
        scores = np.asarray(example.metadata.get("candidate_mismatch_counts", [31.0] * len(example.candidates)), dtype=np.float64)
        preds.append(int(np.argmin(scores)) if scores.size else 0)
    return np.asarray(preds, dtype=np.int64)


def _candidate_generator_audit(seed: int, split: str, phase: str, examples: Sequence[ArcVerificationExample]) -> Dict[str, object]:
    source_counts: Dict[str, int] = {}
    for example in examples:
        for source in example.metadata.get("candidate_source_families_external", []):
            source_counts[str(source)] = source_counts.get(str(source), 0) + 1
    return {
        "benchmark": BENCHMARK,
        "seed": seed,
        "phase": phase,
        "split": split,
        "examples": len(examples),
        "candidate_count_mean": _mean([float(len(example.candidates)) for example in examples]),
        "gold_present_rate": _generator_recall(examples),
        "candidate_generator_recall": _generator_recall(examples),
        "source_family_counts_external": dict(sorted(source_counts.items())),
        "mean_train_execution_score": _mean(
            [float(score) for example in examples for score in example.metadata.get("candidate_train_execution_scores", [])]
        ),
        "mean_candidate_diversity": _mean([float(example.metadata.get("candidate_diversity", 0.0)) for example in examples]),
        "mean_program_length": _mean(
            [float(value) for example in examples for value in example.metadata.get("candidate_program_lengths", [])]
        ),
        "mean_trace_depth": _mean([float(value) for example in examples for value in example.metadata.get("candidate_trace_depths", [])]),
        "model_visible_candidate_sources": False,
        "model_visible_generator_rank": False,
    }


def _hybrid_leakage_audit_rows(
    examples: Sequence[ArcVerificationExample],
    expect_exactly_one_gold: bool,
    seed: int | None = None,
) -> List[Dict[str, object]]:
    rows = []
    for example in examples:
        gold_count = sum(int(_grid_equal(candidate, example.gold_output)) for candidate in example.candidates)
        metadata = example.metadata or {}
        source_visible = bool(metadata.get("model_visible_candidate_sources", True))
        rank_visible = bool(metadata.get("model_visible_generator_rank", True))
        oracle_visible = bool(metadata.get("model_visible_hidden_oracle_score", True))
        test_output_outside_candidates = bool(metadata.get("test_output_visible_outside_candidate_objects", True))
        public_tags_safe = all(tag == "candidate_grid" for tag in example.public_source_tags)
        source_keys_external = "candidate_source_families_external" in metadata
        exact_gold_ok = gold_count == 1 if expect_exactly_one_gold else gold_count >= 0
        rows.append(
            {
                "type": "arc_hybrid1_leakage_audit",
                "benchmark": BENCHMARK,
                "seed": seed,
                "phase": metadata.get("phase"),
                "id": example.id,
                "task_id": example.task_id,
                "split": example.split,
                "gold_candidate_count": int(gold_count),
                "expect_exactly_one_gold": bool(expect_exactly_one_gold),
                "candidate_order_randomized": bool(metadata.get("candidate_order_randomized", False)),
                "candidate_source_metadata_visible_to_model": source_visible,
                "generator_rank_visible_to_model": rank_visible,
                "hidden_oracle_score_visible_to_model": oracle_visible,
                "test_output_visible_outside_candidate_objects": test_output_outside_candidates,
                "candidate_public_tags_safe": public_tags_safe,
                "source_family_logged_external_only": source_keys_external and not source_visible,
                "pass": bool(
                    exact_gold_ok
                    and not source_visible
                    and not rank_visible
                    and not oracle_visible
                    and not test_output_outside_candidates
                    and public_tags_safe
                    and bool(metadata.get("candidate_order_randomized", False))
                ),
            }
        )
    return rows


def _program_specs(train_pairs: Sequence[ArcPair], rng: np.random.Generator) -> List[ProgramSpec]:
    del rng
    programs: List[ProgramSpec] = [
        ProgramSpec("identity", "symbolic_identity", 1, 1, lambda grid: _grid_copy(grid)),
        ProgramSpec("flip_horizontal", "geometric_transform", 2, 1, lambda grid: np.fliplr(np.asarray(grid, dtype=np.int64)).tolist()),
        ProgramSpec("flip_vertical", "geometric_transform", 2, 1, lambda grid: np.flipud(np.asarray(grid, dtype=np.int64)).tolist()),
        ProgramSpec("rotate_180", "geometric_transform", 2, 1, lambda grid: np.rot90(np.asarray(grid, dtype=np.int64), 2).tolist()),
    ]
    color_map = _infer_color_map(train_pairs)
    if color_map:
        programs.append(
            ProgramSpec(
                "cellwise_majority_color_map",
                "symbolic_color_map",
                max(2, len(color_map)),
                2,
                lambda grid, mapping=color_map: _map_colors(grid, mapping),
            )
        )
    diff_patch = _infer_diff_patch(train_pairs)
    if diff_patch:
        programs.append(
            ProgramSpec(
                "majority_diff_patch",
                "procedural_diff_patch",
                max(2, len(diff_patch)),
                2,
                lambda grid, patch=diff_patch: _apply_diff_patch(grid, patch),
            )
        )
    shift = _infer_translation(train_pairs)
    if shift is not None:
        dy, dx = shift
        programs.append(
            ProgramSpec(
                f"translate_{dy}_{dx}",
                "geometric_translation",
                3,
                2,
                lambda grid, dy=dy, dx=dx: _shift_grid(grid, dy, dx),
            )
        )
    for index, pair in enumerate(train_pairs[:4]):
        output = _grid_copy(pair.output)
        programs.append(
            ProgramSpec(
                f"train_output_template_{index}",
                "train_output_template",
                max(1, _grid_size(output) // 10),
                1,
                lambda _grid, output=output: _grid_copy(output),
            )
        )
    return programs


def _proposal_from_program(program: ProgramSpec, train_pairs: Sequence[ArcPair], test_input: Grid) -> CandidateProposal:
    output = _clean_grid(program.apply(test_input))
    score, mismatch_count, mismatch_ratio = _program_train_fit(program, train_pairs)
    return CandidateProposal(
        output=output,
        source_family=program.source_family,
        program_name=program.name,
        train_execution_score=score,
        mismatch_count=mismatch_count,
        mismatch_ratio=mismatch_ratio,
        program_length=program.program_length,
        trace_depth=program.trace_depth,
    )


def _program_train_fit(program: ProgramSpec, train_pairs: Sequence[ArcPair]) -> Tuple[float, float, float]:
    if not train_pairs:
        return 0.0, 31.0, 1.0
    similarities = []
    counts = []
    ratios = []
    for pair in train_pairs:
        pred = _clean_grid(program.apply(pair.input))
        stat = _grid_mismatch(pred, pair.output)
        similarities.append(1.0 - stat["ratio"])
        counts.append(stat["count"])
        ratios.append(stat["ratio"])
    return _mean(similarities), _mean(counts), _mean(ratios)


def _output_only_evidence(train_pairs: Sequence[ArcPair], test_input: Grid, output: Grid) -> Dict[str, float]:
    del test_input
    if not train_pairs:
        return {"train_execution_score": 0.0, "mismatch_count": 31.0, "mismatch_ratio": 1.0}
    distances = []
    counts = []
    ratios = []
    for pair in train_pairs:
        stat = _grid_mismatch(output, pair.output)
        distances.append(1.0 - stat["ratio"])
        counts.append(stat["count"])
        ratios.append(stat["ratio"])
    return {
        "train_execution_score": max(0.0, min(1.0, _mean(distances))),
        "mismatch_count": _mean(counts),
        "mismatch_ratio": max(0.0, min(1.0, _mean(ratios))),
    }


def _candidate_mismatch_stats(train_pairs: Sequence[ArcPair], test_input: Grid, output: Grid) -> Dict[str, object]:
    evidence = _output_only_evidence(train_pairs, test_input, output)
    test_stat = _grid_mismatch(test_input, output)
    return {
        "mean_train_output_mismatch_count": float(evidence["mismatch_count"]),
        "mean_train_output_mismatch_ratio": float(evidence["mismatch_ratio"]),
        "test_input_candidate_mismatch_count": float(test_stat["count"]),
        "test_input_candidate_mismatch_ratio": float(test_stat["ratio"]),
    }


def _select_candidate_proposals(proposals: Sequence[CandidateProposal], count: int) -> List[CandidateProposal]:
    unique: List[CandidateProposal] = []
    seen: set[str] = set()
    for proposal in sorted(
        proposals,
        key=lambda item: (float(item.train_execution_score), -float(item.mismatch_ratio), -int(item.refinement_depth)),
        reverse=True,
    ):
        key = _grid_key(proposal.output)
        if key in seen:
            continue
        seen.add(key)
        unique.append(proposal)
        if len(unique) >= int(count):
            break
    return unique


def _append_unique_proposal(proposals: List[CandidateProposal], seen: set[str], proposal: CandidateProposal) -> None:
    key = _grid_key(proposal.output)
    if key in seen:
        return
    seen.add(key)
    proposals.append(proposal)


def _mutate_candidate_output(grid: Grid, rng: np.random.Generator, salt: int) -> Grid:
    arr = np.asarray(grid, dtype=np.int64)
    if arr.size == 0:
        return _grid_copy(grid)
    choice = int((salt + rng.integers(0, 5)) % 5)
    out = arr.copy()
    if choice == 0:
        row = int(rng.integers(0, out.shape[0]))
        col = int(rng.integers(0, out.shape[1]))
        out[row, col] = int((int(out[row, col]) + 1 + salt) % 10)
    elif choice == 1:
        out = np.fliplr(out)
    elif choice == 2:
        out = np.flipud(out)
    elif choice == 3:
        colors = _palette_array(out)
        if len(colors) >= 2:
            a, b = rng.choice(colors, size=2, replace=False)
            mask_a = out == int(a)
            mask_b = out == int(b)
            out[mask_a] = int(b)
            out[mask_b] = int(a)
    else:
        out = np.asarray(_shift_grid(out.tolist(), int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))), dtype=np.int64)
    return out.astype(np.int64).tolist()


def _random_palette_grid_like(grid: Grid, train_pairs: Sequence[ArcPair], rng: np.random.Generator) -> Grid:
    arr = np.asarray(grid, dtype=np.int64)
    h, w = arr.shape if arr.ndim == 2 and arr.size else (1, 1)
    palette = sorted({int(value) for pair in train_pairs for side in (pair.input, pair.output) for row in side for value in row})
    if not palette:
        palette = list(range(3))
    probs = np.ones(len(palette), dtype=np.float64) / max(1, len(palette))
    return rng.choice(np.asarray(palette, dtype=np.int64), size=(h, w), p=probs).astype(np.int64).tolist()


def _infer_color_map(train_pairs: Sequence[ArcPair]) -> Dict[int, int]:
    counts: Dict[int, Dict[int, int]] = {}
    for pair in train_pairs:
        if _shape(pair.input) != _shape(pair.output):
            continue
        inp = np.asarray(pair.input, dtype=np.int64)
        out = np.asarray(pair.output, dtype=np.int64)
        for src, dst in zip(inp.reshape(-1), out.reshape(-1)):
            src_i = int(src)
            dst_i = int(dst)
            counts.setdefault(src_i, {})[dst_i] = counts.setdefault(src_i, {}).get(dst_i, 0) + 1
    mapping = {}
    for src, dst_counts in counts.items():
        if dst_counts:
            mapping[src] = max(dst_counts.items(), key=lambda item: item[1])[0]
    return mapping


def _infer_diff_patch(train_pairs: Sequence[ArcPair]) -> Dict[Tuple[int, int], int]:
    votes: Dict[Tuple[int, int], Dict[int, int]] = {}
    for pair in train_pairs:
        if _shape(pair.input) != _shape(pair.output):
            continue
        inp = np.asarray(pair.input, dtype=np.int64)
        out = np.asarray(pair.output, dtype=np.int64)
        for row, col in np.argwhere(inp != out):
            key = (int(row), int(col))
            color = int(out[row, col])
            votes.setdefault(key, {})[color] = votes.setdefault(key, {}).get(color, 0) + 1
    return {key: max(values.items(), key=lambda item: item[1])[0] for key, values in votes.items()}


def _infer_translation(train_pairs: Sequence[ArcPair]) -> Tuple[int, int] | None:
    votes: Dict[Tuple[int, int], int] = {}
    for pair in train_pairs:
        if _shape(pair.input) != _shape(pair.output):
            continue
        inp = np.asarray(pair.input, dtype=np.int64)
        out = np.asarray(pair.output, dtype=np.int64)
        bg = _dominant(inp)
        in_pos = np.argwhere(inp != bg)
        out_pos = np.argwhere(out != _dominant(out))
        if len(in_pos) == 0 or len(out_pos) == 0:
            continue
        dy = int(round(float(np.mean(out_pos[:, 0]) - np.mean(in_pos[:, 0]))))
        dx = int(round(float(np.mean(out_pos[:, 1]) - np.mean(in_pos[:, 1]))))
        if abs(dy) <= 5 and abs(dx) <= 5:
            votes[(dy, dx)] = votes.get((dy, dx), 0) + 1
    if not votes:
        return None
    return max(votes.items(), key=lambda item: item[1])[0]


def _map_colors(grid: Grid, mapping: Dict[int, int]) -> Grid:
    arr = np.asarray(grid, dtype=np.int64).copy()
    out = arr.copy()
    for src, dst in mapping.items():
        out[arr == int(src)] = int(dst)
    return out.tolist()


def _apply_diff_patch(grid: Grid, patch: Dict[Tuple[int, int], int]) -> Grid:
    out = np.asarray(grid, dtype=np.int64).copy()
    if out.ndim != 2:
        return _grid_copy(grid)
    for (row, col), color in patch.items():
        if 0 <= row < out.shape[0] and 0 <= col < out.shape[1]:
            out[row, col] = int(color)
    return out.tolist()


def _shift_grid(grid: Grid, dy: int, dx: int) -> Grid:
    arr = np.asarray(grid, dtype=np.int64)
    if arr.ndim != 2 or arr.size == 0:
        return _grid_copy(grid)
    fill = _dominant(arr)
    out = np.full_like(arr, fill)
    src_r0 = max(0, -int(dy))
    src_r1 = min(arr.shape[0], arr.shape[0] - int(dy))
    src_c0 = max(0, -int(dx))
    src_c1 = min(arr.shape[1], arr.shape[1] - int(dx))
    dst_r0 = max(0, int(dy))
    dst_c0 = max(0, int(dx))
    dst_r1 = dst_r0 + max(0, src_r1 - src_r0)
    dst_c1 = dst_c0 + max(0, src_c1 - src_c0)
    if dst_r1 > dst_r0 and dst_c1 > dst_c0:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]
    return out.tolist()


def _model_config_for_variant(variant: HybridVariant) -> ArcModelConfig:
    return ArcModelConfig(
        variant_family=variant.family,
        view_names=tuple(variant.view_set),
        representation_mode=variant.representation,
        interaction_style=variant.interaction,
        model_dim=64,
        num_heads=4,
        rule_layers=1,
        candidate_layers=1,
        coordination_blocks=int(variant.coordination_blocks),
        ff_dim=128,
        max_rule_tokens=112,
        max_candidate_tokens=96,
    )


def _variant_plan(configured: object | None = None) -> List[HybridVariant]:
    default = [
        HybridVariant(
            name=SIMPLE_CROSS_ATTENTION_VARIANT,
            family="baseline_cross_attention",
            representation="hybrid_cell_object_diff",
            view_set=("raw", "diff", "object", "color"),
            interaction="late_candidate_query",
            uses_execution_evidence=False,
            uses_mismatch_evidence=False,
            uses_program_evidence=False,
        ),
        HybridVariant(
            name="B_execution_mismatch_latent_verifier",
            family="evidence_representation",
            representation="hybrid_execution",
            view_set=("raw", "diff", "object", "color", "candidate_execution"),
            interaction="late_candidate_query",
            uses_execution_evidence=True,
            uses_mismatch_evidence=True,
            uses_program_evidence=True,
        ),
        HybridVariant(
            name="C_bidirectional_all_evidence",
            family="candidate_query_mechanism",
            representation="hybrid_all_execution",
            view_set=("raw", "diff", "object", "color", "geometry", "candidate_execution"),
            interaction="bidirectional_candidate_rule",
            uses_execution_evidence=True,
            uses_mismatch_evidence=True,
            uses_program_evidence=True,
        ),
        HybridVariant(
            name="D_stack2_all_evidence_pairwise",
            family="objective_and_depth",
            representation="hybrid_all_execution",
            view_set=("raw", "diff", "object", "color", "geometry", "candidate_execution"),
            interaction="late_candidate_query",
            objective="pairwise_logistic",
            coordination_blocks=2,
            uses_execution_evidence=True,
            uses_mismatch_evidence=True,
            uses_program_evidence=True,
        ),
    ]
    if not configured:
        return default
    wanted = {str(name) for name in configured if isinstance(name, str)}
    return [variant for variant in default if variant.name in wanted] or default


def _leaderboard(rows: Sequence[Dict[str, object]], controls: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    control_by_key = {(row["variant"], row["seed"]): row for row in controls}
    leaders = []
    for row in rows:
        if row.get("status") != "completed":
            leaders.append(
                {
                    "variant": row.get("variant"),
                    "seed": row.get("seed"),
                    "status": row.get("status"),
                    "trainable_top1": 0.0,
                    "frozen_top1": 0.0,
                    "delta": 0.0,
                    "natural_solve_rate": 0.0,
                    "generator_recall": 0.0,
                    "controls_pass": False,
                    "score": -999.0,
                }
            )
            continue
        phase1 = row.get("phase1_gold_present", {})
        phase2 = row.get("phase2_natural_pool", {})
        trainable = float(phase1.get("trainable", {}).get("top1", 0.0))
        frozen = float(phase1.get("frozen", {}).get("top1", 0.0))
        delta = float(phase1.get("delta_trainable_minus_frozen", 0.0))
        natural = float(phase2.get("overall_solve_rate", 0.0))
        recall = float(phase2.get("generator_recall", 0.0))
        controls_pass = bool(control_by_key.get((row["variant"], row["seed"]), {}).get("control_pass", {}).get("overall", False))
        candidate_only = float(control_by_key.get((row["variant"], row["seed"]), {}).get("candidate_only", {}).get("top1", 0.0))
        heuristic_best = _best_baseline(phase1.get("baselines", {}))
        score = trainable + delta + natural - 0.5 * candidate_only + (0.2 if controls_pass else -1.0)
        leaders.append(
            {
                "variant": row["variant"],
                "seed": row["seed"],
                "status": row["status"],
                "family": row.get("family"),
                "representation": row.get("representation"),
                "view_set": " ".join(row.get("view_set", [])),
                "trainable_top1": trainable,
                "frozen_top1": frozen,
                "delta": delta,
                "candidate_only": candidate_only,
                "heuristic_best": heuristic_best,
                "natural_solve_rate": natural,
                "generator_recall": recall,
                "controls_pass": controls_pass,
                "training_time_seconds": float(row.get("training_time_seconds", 0.0)),
                "param_count": int(row.get("param_count", 0)),
                "score": score,
            }
        )
    leaders.sort(key=lambda item: float(item["score"]), reverse=True)
    return leaders


def _failure_taxonomy(
    rows: Sequence[Dict[str, object]],
    overfit_rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    generator_audit: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    counts: Dict[str, int] = {}
    for row in rows:
        if row.get("status") == "failed_overfit_gate":
            key = f"failed_overfit::{row.get('failure_reason')}"
            counts[key] = counts.get(key, 0) + 1
            continue
        if row.get("status") == "completed":
            if float(row.get("phase1_gold_present", {}).get("delta_trainable_minus_frozen", 0.0)) <= 0.0:
                counts["verifier_ranking_failure_did_not_beat_frozen"] = counts.get("verifier_ranking_failure_did_not_beat_frozen", 0) + 1
            if float(row.get("phase2_natural_pool", {}).get("generator_recall", 1.0)) <= 0.0:
                counts["generator_recall_failure"] = counts.get("generator_recall_failure", 0) + 1
    for row in controls:
        if not bool(row.get("control_pass", {}).get("overall", False)):
            counts["control_failure"] = counts.get("control_failure", 0) + 1
    if any(float(row.get("candidate_generator_recall", 0.0)) < 0.05 and row.get("phase") == "phase2_natural_pool" for row in generator_audit):
        counts["natural_pool_low_recall"] = counts.get("natural_pool_low_recall", 0) + 1
    return {
        "counts": dict(sorted(counts.items())),
        "failed_overfit_gates": [row for row in overfit_rows if not bool(row.get("pass"))],
        "failure_split_definitions": {
            "generator_recall_failure": "gold candidate absent from natural pool",
            "verifier_ranking_failure": "gold present but trainable verifier does not rank it top1",
            "train_fit_false_positive": "execution/train-fit heuristic ranks a wrong candidate above gold",
            "ambiguity_among_train_solving_candidates": "multiple high-train-fit candidates compete and test correctness differs",
            "representation_failure": "variant cannot overfit or ablation/control diagnostics show no useful evidence use",
        },
    }


def _summary(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    leaderboard: Sequence[Dict[str, object]],
    generator_audit: Sequence[Dict[str, object]],
    refinement_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    completed = [row for row in rows if row.get("status") == "completed"]
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in completed:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    variant_summaries = []
    for variant, variant_rows in by_variant.items():
        trainable = [float(row["phase1_gold_present"]["trainable"]["top1"]) for row in variant_rows]
        frozen = [float(row["phase1_gold_present"]["frozen"]["top1"]) for row in variant_rows]
        deltas = [float(row["phase1_gold_present"]["delta_trainable_minus_frozen"]) for row in variant_rows]
        natural = [float(row["phase2_natural_pool"].get("overall_solve_rate", 0.0)) for row in variant_rows]
        variant_controls = [row for row in controls if row.get("variant") == variant]
        variant_summaries.append(
            {
                "variant": variant,
                "seeds": [int(row["seed"]) for row in variant_rows],
                "mean_trainable_top1": _mean(trainable),
                "std_trainable_top1": _std(trainable),
                "min_trainable_top1": min(trainable) if trainable else 0.0,
                "max_trainable_top1": max(trainable) if trainable else 0.0,
                "mean_frozen_top1": _mean(frozen),
                "mean_delta": _mean(deltas),
                "bootstrap_95ci_delta": _bootstrap_ci(deltas),
                "seeds_trainable_beats_frozen": int(sum(delta > 0.0 for delta in deltas)),
                "mean_natural_solve_rate": _mean(natural),
                "controls_pass_all": all(bool(row.get("control_pass", {}).get("overall", False)) for row in variant_controls),
            }
        )
    variant_summaries.sort(key=lambda item: float(item["mean_delta"]), reverse=True)
    best = variant_summaries[0] if variant_summaries else {}
    gates = _medium_gates(best, controls, rows)
    return {
        "variant_summaries": variant_summaries,
        "best_variant_by_delta": best.get("variant"),
        "leaderboard_best": leaderboard[0]["variant"] if leaderboard else None,
        "phase2_generator_recall_mean": _mean(
            [float(row.get("candidate_generator_recall", 0.0)) for row in generator_audit if row.get("phase") == "phase2_natural_pool"]
        ),
        "refinement_summary": _refinement_summary(refinement_rows),
        "medium_validation_ready": all(bool(value) for key, value in gates.items() if key != "final_validation_not_launched"),
        "medium_gates": gates,
        "final_validation_launched": False,
        "allowed_claim_if_gates_pass": (
            "On controlled ARC-AGI-2 candidate/refinement pools, shared-weight latent clone coordination improves "
            "candidate verification/ranking over exact frozen, candidate-only, metadata-only, and heuristic baselines "
            "while passing leakage, mismatch, shuffle, and invariance controls."
        ),
    }


def _medium_gates(best: Dict[str, object], controls: Sequence[Dict[str, object]], rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    best_variant = best.get("variant")
    best_controls = [row for row in controls if row.get("variant") == best_variant]
    best_rows = [row for row in rows if row.get("variant") == best_variant and row.get("status") == "completed"]
    frozen_ok = all(
        bool(row.get("frozen_audit", {}).get("frozen_shared_model_zero_delta"))
        and bool(row.get("frozen_audit", {}).get("frozen_shared_model_zero_grad"))
        for row in best_rows
    )
    trainable_ok = all(bool(row.get("trainable_audit", {}).get("trainable_shared_model_changed")) for row in best_rows)
    seeds = len(best.get("seeds", [])) if best else 0
    beats = int(best.get("seeds_trainable_beats_frozen", 0)) if best else 0
    ci = best.get("bootstrap_95ci_delta", [0.0, 0.0]) if best else [0.0, 0.0]
    return {
        "trainable_beats_frozen_on_at_least_4_of_5_seeds": seeds >= 5 and beats >= 4,
        "mean_trainable_frozen_delta_at_least_0_15": float(best.get("mean_delta", 0.0)) >= 0.15 if best else False,
        "bootstrap_ci_lower_bound_gt_0": float(ci[0]) > 0.0 if ci else False,
        "candidate_only_near_chance": all(bool(row.get("control_pass", {}).get("candidate_only_near_chance", False)) for row in best_controls),
        "metadata_only_near_chance": all(bool(row.get("control_pass", {}).get("metadata_only_near_chance", False)) for row in best_controls),
        "execution_score_heuristic_does_not_explain_result": _best_beats_execution_heuristic(best_variant, rows),
        "train_pair_shuffle_degrades": all(bool(row.get("control_pass", {}).get("train_pair_shuffle_degrades", False)) for row in best_controls),
        "candidate_evidence_mismatch_degrades": all(bool(row.get("control_pass", {}).get("candidate_evidence_mismatch_degrades", False)) for row in best_controls),
        "trace_mismatch_degrades_if_used": all(bool(row.get("control_pass", {}).get("execution_trace_mismatch_degrades", False)) for row in best_controls),
        "mismatch_map_mismatch_degrades_if_used": all(bool(row.get("control_pass", {}).get("mismatch_map_mismatch_degrades", False)) for row in best_controls),
        "candidate_order_remap_invariance_passes": all(bool(row.get("control_pass", {}).get("candidate_order_remap_invariance", False)) for row in best_controls),
        "role_order_invariance_passes": all(bool(row.get("control_pass", {}).get("role_order_invariance", False)) for row in best_controls),
        "hidden_state_shuffle_collapses": all(bool(row.get("control_pass", {}).get("hidden_state_shuffle_collapses", False)) for row in best_controls),
        "frozen_audit_passes": frozen_ok,
        "trainable_update_audit_passes": trainable_ok,
        "final_validation_not_launched": True,
    }


def _best_beats_execution_heuristic(best_variant: object, rows: Sequence[Dict[str, object]]) -> bool:
    best_rows = [row for row in rows if row.get("variant") == best_variant and row.get("status") == "completed"]
    if not best_rows:
        return False
    return all(
        float(row.get("phase1_gold_present", {}).get("trainable", {}).get("top1", 0.0))
        > float(row.get("phase1_gold_present", {}).get("baselines", {}).get("execution_score_heuristic", 1.0))
        for row in best_rows
    )


def _write_outputs(
    result: Dict[str, object],
    results_path: Path,
    controls_path: Path,
    generator_audit_path: Path,
    leakage_path: Path,
    leaderboard_path: Path,
    failure_path: Path,
    overfit_path: Path,
    error_path: Path,
    attention_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    for path in (
        results_path,
        controls_path,
        generator_audit_path,
        leakage_path,
        leaderboard_path,
        failure_path,
        overfit_path,
        error_path,
        attention_path,
        compute_path,
        report_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(_without_large_logs(result), indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(json.dumps({"controls": result["controls"], "medium_gates": result["summary"]["medium_gates"]}, indent=2, sort_keys=True), encoding="utf-8")
    generator_audit_path.write_text(json.dumps({"candidate_generator_audit": result["candidate_generator_audit"]}, indent=2, sort_keys=True), encoding="utf-8")
    leakage_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in result["leakage_audit"]) + ("\n" if result["leakage_audit"] else ""), encoding="utf-8")
    failure_path.write_text(json.dumps(result["failure_taxonomy"], indent=2, sort_keys=True), encoding="utf-8")
    overfit_path.write_text(json.dumps({"overfit_curves": result["overfit_curves"]}, indent=2, sort_keys=True), encoding="utf-8")
    error_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in result["error_cases"]) + ("\n" if result["error_cases"] else ""), encoding="utf-8")
    attention_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in result["attention_summaries"]) + ("\n" if result["attention_summaries"] else ""), encoding="utf-8")
    compute_path.write_text(json.dumps({"compute_metrics": result["compute_metrics"]}, indent=2, sort_keys=True), encoding="utf-8")
    with leaderboard_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in result["leaderboard"] for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["leaderboard"])
    report_path.write_text(_render_report(result), encoding="utf-8")


def _render_report(result: Dict[str, object]) -> str:
    summary = result.get("summary", {})
    leaderboard = result.get("leaderboard", [])
    generator_audit = result.get("candidate_generator_audit", [])
    refinement = result.get("refinement_results", [])
    lines = [
        "# Stage ARC-HYBRID-1 Empirical Search",
        "",
        "## Scope",
        "",
        f"- Stage: `{STAGE_NAME}`.",
        "- Bounded empirical search over ARC candidate generation, candidate evidence, latent verification, and verifier-guided refinement.",
        "- The latent architecture is a verifier/ranker/controller inside candidate pools; it is not claimed to solve ARC from scratch.",
        "- Candidate source families, generator rank, and oracle correctness are logged externally only and are not encoded into model inputs.",
        "- Final validation was not launched.",
        "",
        "## Leaderboard",
        "",
        "| rank | variant | status | trainable | frozen | delta | natural solve | recall | controls |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(leaderboard[:20], start=1):
        lines.append(
            f"| {rank} | {row.get('variant')} | {row.get('status')} | {float(row.get('trainable_top1', 0.0)):.4f} | "
            f"{float(row.get('frozen_top1', 0.0)):.4f} | {float(row.get('delta', 0.0)):.4f} | "
            f"{float(row.get('natural_solve_rate', 0.0)):.4f} | {float(row.get('generator_recall', 0.0)):.4f} | "
            f"`{bool(row.get('controls_pass', False))}` |"
        )
    lines.extend(["", "## Candidate Generator Audit", "", "| phase | split | examples | recall | diversity | top source families |", "|---|---|---:|---:|---:|---|"])
    for row in generator_audit:
        sources = ", ".join(f"{key}:{value}" for key, value in list(row.get("source_family_counts_external", {}).items())[:4])
        lines.append(
            f"| {row.get('phase')} | {row.get('split')} | {int(row.get('examples', 0))} | "
            f"{float(row.get('candidate_generator_recall', 0.0)):.4f} | {float(row.get('mean_candidate_diversity', 0.0)):.4f} | {sources} |"
        )
    lines.extend(["", "## Controls And Gates", "", "| gate | pass |", "|---|---|"])
    for key, value in summary.get("medium_gates", {}).items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Failure Taxonomy", "", "```json", json.dumps(result.get("failure_taxonomy", {}).get("counts", {}), indent=2, sort_keys=True), "```", ""])
    lines.extend(["## Refinement", "", "| variant | strategy | recall before | recall after | solve before | solve after | quality before | quality after |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in refinement:
        lines.append(
            f"| {row.get('variant')} | {row.get('strategy')} | {float(row.get('generator_recall_before', 0.0)):.4f} | "
            f"{float(row.get('generator_recall_after', 0.0)):.4f} | {float(row.get('overall_solve_rate_before', 0.0)):.4f} | "
            f"{float(row.get('overall_solve_rate_after', 0.0)):.4f} | {float(row.get('candidate_quality_before', 0.0)):.4f} | "
            f"{float(row.get('candidate_quality_after', 0.0)):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            f"1. Which candidate generator/refiner produced useful candidate pools? `{_best_generator_source(generator_audit)}`; see generator audit for external source-family counts.",
            f"2. What was generator recall? Phase-2 mean recall `{float(summary.get('phase2_generator_recall_mean', 0.0)):.4f}`.",
            f"3. Did latent verification beat frozen on gold-present sets? `{_answer_beats_frozen(result)}`.",
            f"4. Did latent verification beat execution-score heuristics? `{_answer_beats_execution(result)}`.",
            f"5. Did latent verification improve natural candidate-pool selection? `{_answer_natural_improvement(result)}`.",
            f"6. Which evidence views mattered most? `{_answer_views(result)}`.",
            f"7. Did traces or mismatch maps help? `{_answer_trace_mismatch(result)}`.",
            f"8. Did program evidence help beyond output evidence? `{_answer_program_evidence(result)}`.",
            "9. Did pyramidal/compositional verification help after flat verifier worked? `not_run_in_this_bounded_smoke`; this stage keeps the interface open and gates it behind a flat-verifier pass.",
            f"10. Did verifier-guided refinement improve solve rate? `{_answer_refinement(refinement)}`.",
            f"11. Were failures mostly generator recall or verifier ranking? `{_answer_failure_mode(result)}`.",
            f"12. Which variants failed and why? `{result.get('failure_taxonomy', {}).get('counts', {})}`.",
            "13. Is this hybrid stronger than pure ARC latent-rule-clone verification? `not established unless medium gates pass against the frozen and heuristic baselines`.",
            f"14. Is it ready for medium validation? `{bool(summary.get('medium_validation_ready', False))}`.",
            "",
            "## Claim Boundary",
            "",
            "- Do not claim ARC-AGI-2 solved.",
            "- Do not claim SOTA, general abstraction, or autonomous reasoning.",
            f"- Allowed claim only after gates pass: {summary.get('allowed_claim_if_gates_pass')}",
        ]
    )
    return "\n".join(lines) + "\n"


def _attention_summaries(model, model_config: ArcModelConfig, examples: Sequence[ArcVerificationExample], batch_size: int, device: str, seed: int, variant: str) -> List[Dict[str, object]]:
    if not examples:
        return []
    output = predict_logits(model, examples, model_config, batch_size, device, seed=seed, return_attention=True)
    if not isinstance(output, tuple):
        return []
    _logits, rows = output
    for row in rows:
        row["benchmark"] = BENCHMARK
        row["variant"] = variant
        row["seed"] = seed
        row["diagnostic"] = "attention_mass_by_clone_view"
    return rows


def _error_rows(
    variant: HybridVariant,
    seed: int,
    examples: Sequence[ArcVerificationExample],
    train_logits: np.ndarray,
    frozen_logits: np.ndarray,
    limit: int,
) -> List[Dict[str, object]]:
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    train_pred = np.argmax(train_logits, axis=1) if train_logits.size else np.zeros(len(examples), dtype=np.int64)
    frozen_pred = np.argmax(frozen_logits, axis=1) if frozen_logits.size else np.zeros(len(examples), dtype=np.int64)
    rows = []
    for index, example in enumerate(examples):
        if len(rows) >= limit:
            break
        category = None
        if train_pred[index] == labels[index] and frozen_pred[index] != labels[index]:
            category = "latent_verifier_beats_frozen"
        elif train_pred[index] != labels[index] and frozen_pred[index] == labels[index]:
            category = "frozen_beats_latent_verifier"
        elif train_pred[index] != labels[index]:
            category = "verifier_ranking_failure"
        if category is None:
            continue
        rows.append(
            {
                "type": "arc_hybrid1_phase1_error_case",
                "category": category,
                "variant": variant.name,
                "seed": seed,
                "task_id": example.task_id,
                "example_id": example.id,
                "gold_candidate_index": int(example.label),
                "predicted_candidate_index": int(train_pred[index]),
                "frozen_candidate_index": int(frozen_pred[index]),
                "candidate_probabilities": _softmax_np(train_logits[index]).tolist(),
                "candidate_source_families_external": example.metadata.get("candidate_source_families_external", []),
                "candidate_train_execution_scores": example.metadata.get("candidate_train_execution_scores", []),
            }
        )
    return rows


def _natural_error_rows(
    variant: HybridVariant,
    seed: int,
    examples: Sequence[ArcVerificationExample],
    train_logits: np.ndarray,
    frozen_logits: np.ndarray,
    limit: int,
) -> List[Dict[str, object]]:
    train_pred = np.argmax(train_logits, axis=1) if train_logits.size else np.zeros(len(examples), dtype=np.int64)
    frozen_pred = np.argmax(frozen_logits, axis=1) if frozen_logits.size else np.zeros(len(examples), dtype=np.int64)
    rows = []
    for index, example in enumerate(examples):
        if len(rows) >= limit:
            break
        correct = {int(value) for value in example.metadata.get("offline_correct_candidate_indices", [])}
        if not correct:
            category = "generator_recall_failure"
        elif int(train_pred[index]) not in correct:
            category = "natural_pool_verifier_ranking_failure"
        else:
            continue
        rows.append(
            {
                "type": "arc_hybrid1_phase2_error_case",
                "category": category,
                "variant": variant.name,
                "seed": seed,
                "task_id": example.task_id,
                "example_id": example.id,
                "predicted_candidate_index": int(train_pred[index]),
                "frozen_candidate_index": int(frozen_pred[index]),
                "correct_candidate_indices": sorted(correct),
                "candidate_probabilities": _softmax_np(train_logits[index]).tolist(),
                "candidate_source_families_external": example.metadata.get("candidate_source_families_external", []),
            }
        )
    return rows


def _without_large_logs(result: Dict[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key
        not in {
            "leakage_audit",
            "error_cases",
            "attention_summaries",
            "compute_metrics",
        }
    }


def _dataset_config(data: object) -> ArcVerificationDatasetConfig:
    values = dict(data or {})
    allowed = set(ArcVerificationDatasetConfig.__dataclass_fields__.keys())
    return ArcVerificationDatasetConfig(**{key: value for key, value in values.items() if key in allowed})


def _generator_config(data: object, default_candidates: int) -> CandidateGeneratorConfig:
    values = dict(data or {})
    values.setdefault("num_candidates", default_candidates)
    values.setdefault("natural_candidate_count", values.get("num_candidates", default_candidates))
    allowed = set(CandidateGeneratorConfig.__dataclass_fields__.keys())
    return CandidateGeneratorConfig(**{key: value for key, value in values.items() if key in allowed})


def _training_config(data: object) -> ArcTrainingConfig:
    values = dict(data or {})
    allowed = set(ArcTrainingConfig.__dataclass_fields__.keys())
    return ArcTrainingConfig(**{key: value for key, value in values.items() if key in allowed})


def _feature_training_config(data: object) -> FeatureTrainingConfig:
    values = dict(data or {})
    allowed = set(FeatureTrainingConfig.__dataclass_fields__.keys())
    return FeatureTrainingConfig(**{key: value for key, value in values.items() if key in allowed})


def _micro_gates(data: object | None) -> List[Tuple[str, int, int, float]]:
    if not data:
        return [("4_tasks_N2", 2, 4, 0.95), ("16_tasks_N4", 4, 16, 0.90), ("64_tasks_N8", 8, 64, 0.80)]
    gates = []
    for row in data:
        if isinstance(row, dict):
            gates.append((str(row.get("name", f"{row.get('examples')}_N{row.get('candidate_count')}")), int(row["candidate_count"]), int(row["examples"]), float(row["threshold"])))
    return gates or [("4_tasks_N2", 2, 4, 0.95), ("16_tasks_N4", 4, 16, 0.90), ("64_tasks_N8", 8, 64, 0.80)]


def _load_config(path: Path) -> Dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _first_failed_gate(rows: Sequence[Dict[str, object]]) -> str:
    for row in rows:
        if row.get("gate_required", True) and not row.get("pass"):
            return str(row.get("gate"))
    return "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_int(text: str) -> int:
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % 2_147_483_647
    return value


def _grid_copy(grid: Sequence[Sequence[int]]) -> Grid:
    return [[int(value) for value in row] for row in grid]


def _clean_grid(grid: object) -> Grid:
    if not isinstance(grid, list):
        grid = np.asarray(grid, dtype=np.int64).tolist()
    rows = []
    for row in grid:
        if isinstance(row, (list, tuple)):
            rows.append([max(0, min(9, int(value))) for value in row])
    return rows if rows else [[0]]


def _grid_equal(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> bool:
    aa = np.asarray(a, dtype=np.int64)
    bb = np.asarray(b, dtype=np.int64)
    return aa.shape == bb.shape and bool(np.array_equal(aa, bb))


def _grid_key(grid: Sequence[Sequence[int]]) -> str:
    arr = np.asarray(grid, dtype=np.int64)
    return f"{arr.shape}:{arr.tolist()}"


def _grid_size(grid: Sequence[Sequence[int]]) -> int:
    shape = _shape(grid)
    return int(shape[0] * shape[1])


def _grid_mismatch(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> Dict[str, float]:
    aa = np.asarray(a, dtype=np.int64)
    bb = np.asarray(b, dtype=np.int64)
    if aa.size == 0 or bb.size == 0:
        return {"count": 31.0, "ratio": 1.0}
    h = min(aa.shape[0], bb.shape[0])
    w = min(aa.shape[1], bb.shape[1])
    overlap_count = int(np.sum(aa[:h, :w] != bb[:h, :w]))
    size_penalty = abs(aa.shape[0] - bb.shape[0]) * max(aa.shape[1], bb.shape[1], 1) + abs(aa.shape[1] - bb.shape[1]) * h
    count = float(overlap_count + size_penalty)
    denom = float(max(1, max(aa.shape[0], bb.shape[0]) * max(aa.shape[1], bb.shape[1])))
    return {"count": count, "ratio": max(0.0, min(1.0, count / denom))}


def _palette(grid: Sequence[Sequence[int]]) -> List[int]:
    arr = np.asarray(grid, dtype=np.int64)
    return [int(value) for value in np.unique(arr)] if arr.size else []


def _palette_array(grid: np.ndarray) -> List[int]:
    return [int(value) for value in np.unique(grid)] if grid.size else []


def _dominant(grid: np.ndarray) -> int:
    if grid.size == 0:
        return 0
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _candidate_diversity(candidates: Sequence[Grid]) -> float:
    if len(candidates) < 2:
        return 0.0
    values = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            values.append(_grid_mismatch(candidates[i], candidates[j])["ratio"])
    return _mean(values)


def _generator_recall(examples: Sequence[ArcVerificationExample]) -> float:
    if not examples:
        return 0.0
    return float(np.mean([bool(example.metadata.get("offline_gold_present", False)) for example in examples]))


def _proposals_from_example(example: ArcVerificationExample) -> List[CandidateProposal]:
    proposals = []
    for index, candidate in enumerate(example.candidates):
        proposals.append(
            CandidateProposal(
                output=_grid_copy(candidate),
                source_family=str(_metadata_index(example, "candidate_source_families_external", index, "existing_candidate")),
                program_name=str(_metadata_index(example, "candidate_program_names_external", index, "existing_candidate")),
                train_execution_score=float(_metadata_index(example, "candidate_train_execution_scores", index, 0.0)),
                mismatch_count=float(_metadata_index(example, "candidate_mismatch_counts", index, 31.0)),
                mismatch_ratio=float(_metadata_index(example, "candidate_mismatch_ratios", index, 1.0)),
                program_length=int(_metadata_index(example, "candidate_program_lengths", index, 1)),
                trace_depth=int(_metadata_index(example, "candidate_trace_depths", index, 1)),
                refinement_depth=int(_metadata_index(example, "candidate_refinement_depths", index, 0)),
            )
        )
    return proposals


def _metadata_index(example: ArcVerificationExample, key: str, index: int, default: object) -> object:
    values = example.metadata.get(key, [])
    if isinstance(values, list) and 0 <= int(index) < len(values):
        return values[int(index)]
    return default


def _max_train_execution_score(example: ArcVerificationExample) -> float:
    values = [float(value) for value in example.metadata.get("candidate_train_execution_scores", [])]
    return max(values) if values else 0.0


def _natural_accuracy_by_key(predictions: Sequence[int], examples: Sequence[ArcVerificationExample], keys: Sequence[str]) -> Dict[str, float]:
    out = {}
    for key in sorted(set(keys)):
        idx = [i for i, value in enumerate(keys) if value == key]
        out[str(key)] = _natural_prediction_metric([predictions[i] for i in idx], [examples[i] for i in idx])
    return out


def _natural_accuracy_by_predicted_source(predictions: Sequence[int], examples: Sequence[ArcVerificationExample]) -> Dict[str, float]:
    keys = []
    for prediction, example in zip(predictions, examples):
        keys.append(str(_metadata_index(example, "candidate_source_families_external", int(prediction), "unknown")))
    return _natural_accuracy_by_key(predictions, examples, keys)


def _natural_accuracy_by_candidate_metadata(predictions: Sequence[int], examples: Sequence[ArcVerificationExample], key: str) -> Dict[str, float]:
    keys = [str(_metadata_index(example, key, int(prediction), "unknown")) for prediction, example in zip(predictions, examples)]
    return _natural_accuracy_by_key(predictions, examples, keys)


def _best_baseline(baselines: Dict[str, object]) -> float:
    values = [float(value) for value in baselines.values() if isinstance(value, (int, float))]
    return max(values) if values else 0.0


def _refinement_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {"run": False}
    by_strategy: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_strategy.setdefault(str(row["strategy"]), []).append(row)
    return {
        strategy: {
            "mean_solve_delta": _mean([float(row["overall_solve_rate_after"]) - float(row["overall_solve_rate_before"]) for row in values]),
            "mean_recall_delta": _mean([float(row["generator_recall_after"]) - float(row["generator_recall_before"]) for row in values]),
            "mean_quality_delta": _mean([float(row["candidate_quality_after"]) - float(row["candidate_quality_before"]) for row in values]),
        }
        for strategy, values in sorted(by_strategy.items())
    }


def _best_generator_source(generator_audit: Sequence[Dict[str, object]]) -> str:
    counts: Dict[str, int] = {}
    for row in generator_audit:
        for key, value in row.get("source_family_counts_external", {}).items():
            counts[str(key)] = counts.get(str(key), 0) + int(value)
    if not counts:
        return "not_established"
    return max(counts.items(), key=lambda item: item[1])[0]


def _answer_beats_frozen(result: Dict[str, object]) -> str:
    leaders = [row for row in result.get("leaderboard", []) if row.get("status") == "completed"]
    if not leaders:
        return "not_run"
    best = leaders[0]
    return f"{float(best.get('delta', 0.0)) > 0.0} (best delta {float(best.get('delta', 0.0)):.4f})"


def _answer_beats_execution(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("rows", []) if row.get("status") == "completed"]
    if not rows:
        return "not_run"
    best = max(rows, key=lambda row: float(row.get("phase1_gold_present", {}).get("trainable", {}).get("top1", 0.0)))
    trainable = float(best.get("phase1_gold_present", {}).get("trainable", {}).get("top1", 0.0))
    heuristic = float(best.get("phase1_gold_present", {}).get("baselines", {}).get("execution_score_heuristic", 0.0))
    return f"{trainable > heuristic} (trainable {trainable:.4f}, execution heuristic {heuristic:.4f})"


def _answer_natural_improvement(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("rows", []) if row.get("status") == "completed"]
    if not rows:
        return "not_run"
    best = max(rows, key=lambda row: float(row.get("phase2_natural_pool", {}).get("overall_solve_rate", 0.0)))
    trainable = float(best.get("phase2_natural_pool", {}).get("overall_solve_rate", 0.0))
    frozen = float(best.get("phase2_natural_pool", {}).get("frozen", {}).get("overall_solve_rate", 0.0))
    return f"{trainable > frozen} (trainable {trainable:.4f}, frozen {frozen:.4f})"


def _answer_views(result: Dict[str, object]) -> str:
    completed = [row for row in result.get("leaderboard", []) if row.get("status") == "completed" and bool(row.get("controls_pass", False))]
    if not completed:
        return "not_established_controls_not_passed"
    return str(completed[0].get("view_set", "unknown"))


def _answer_trace_mismatch(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("rows", []) if row.get("uses_execution_evidence") or row.get("uses_mismatch_evidence")]
    if not rows:
        return "not_run"
    best = max(rows, key=lambda row: float(row.get("phase1_gold_present", {}).get("delta_trainable_minus_frozen", 0.0)))
    return f"best evidence variant {best.get('variant')} delta {float(best.get('phase1_gold_present', {}).get('delta_trainable_minus_frozen', 0.0)):.4f}"


def _answer_program_evidence(result: Dict[str, object]) -> str:
    rows = [row for row in result.get("rows", []) if row.get("uses_program_evidence")]
    if not rows:
        return "not_run"
    return "not_established_without_positive controlled delta" if all(float(row.get("phase1_gold_present", {}).get("delta_trainable_minus_frozen", 0.0)) <= 0.0 for row in rows) else "positive_delta_seen_but_gate_dependent"


def _answer_refinement(rows: Sequence[Dict[str, object]]) -> str:
    if not rows:
        return "not_run"
    best = max(rows, key=lambda row: float(row.get("overall_solve_rate_after", 0.0)) - float(row.get("overall_solve_rate_before", 0.0)))
    delta = float(best.get("overall_solve_rate_after", 0.0)) - float(best.get("overall_solve_rate_before", 0.0))
    return f"{delta > 0.0} (best {best.get('strategy')} solve delta {delta:.4f})"


def _answer_failure_mode(result: Dict[str, object]) -> str:
    counts = result.get("failure_taxonomy", {}).get("counts", {})
    if not counts:
        return "no_failures_logged"
    return max(counts.items(), key=lambda item: int(item[1]))[0]


if __name__ == "__main__":
    main()
