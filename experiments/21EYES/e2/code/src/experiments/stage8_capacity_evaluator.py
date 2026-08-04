from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from src.datasets.latent_attention_capacity_dataset import (
    SCALE_SCHEDULE,
    Stage8Example,
    build_stage8_examples,
    Stage8DatasetConfig,
    validate_stage8_examples,
)
from src.experiments.stage8_controls import (
    DEGRADATION_CONTROLS,
    INVARIANCE_CONTROLS,
    apply_stage8_control,
    control_expectation,
    shortcut_audit,
)
from src.models.latent_attention_variants import (
    Stage8ArchitectureConfig,
    build_stage8_selector,
    estimate_stage8_compute,
)


@dataclass(frozen=True)
class Stage8EvaluationConfig:
    train_examples: int = 192
    eval_examples: int = 96
    final_examples: int = 128
    k_candidates: int = 8
    accuracy_threshold: float = 0.85
    chance_margin: float = 0.20
    min_stable_seeds: int = 5
    max_seed_std: float = 0.08
    min_seed_accuracy: float = 0.75
    degradation_threshold: float = 0.10
    invariance_tolerance: float = 0.05
    controls: Tuple[str, ...] = (
        "randomized_labels",
        "randomized_candidate_order_with_label_remap",
        "randomized_evidence_block_order",
        "evidence_candidate_mismatch",
        "cross_task_evidence_shuffle",
        "cross_task_query_shuffle",
        "distractor_only",
        "schema_template_only",
    )


def evaluate_stage8_architecture(
    config: Stage8ArchitectureConfig,
    n_values: Sequence[int],
    seeds: Sequence[int],
    eval_config: Stage8EvaluationConfig,
    eval_split: str = "dev",
    run_controls: bool = True,
) -> Dict[str, object]:
    started = time.perf_counter()
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        for n_blocks in n_values:
            rows.append(
                evaluate_stage8_seed_n(
                    config=config,
                    n_blocks=int(n_blocks),
                    seed=int(seed),
                    eval_config=eval_config,
                    eval_split=eval_split,
                    run_controls=run_controls,
                )
            )
    capacity = compute_effective_capacity(
        rows,
        eval_config=eval_config,
        require_controls=run_controls and len(seeds) >= eval_config.min_stable_seeds,
    )
    compute = estimate_stage8_compute(config, max([int(value) for value in n_values] or [0]), eval_config.k_candidates)
    capacity_per_compute = 0.0
    if float(compute["estimated_forward_compute"]) > 0:
        capacity_per_compute = float(capacity["capacity"]) / float(compute["estimated_forward_compute"])
    return {
        "config_id": config.config_id,
        "architecture_name": config.name,
        "model_kind": config.model_kind,
        "rows": rows,
        "capacity": capacity,
        "compute": compute,
        "capacity_per_compute": capacity_per_compute,
        "wall_clock_seconds": time.perf_counter() - started,
        "status": "completed",
    }


def evaluate_stage8_seed_n(
    config: Stage8ArchitectureConfig,
    n_blocks: int,
    seed: int,
    eval_config: Stage8EvaluationConfig,
    eval_split: str = "dev",
    run_controls: bool = True,
) -> Dict[str, object]:
    train = _training_examples(config, n_blocks, seed, eval_config)
    split_name = "final" if eval_split in {"final", "stage8c", "heldout"} else "dev"
    eval_examples = build_stage8_examples(
        Stage8DatasetConfig(
            n_examples=eval_config.final_examples if split_name == "final" else eval_config.eval_examples,
            n_blocks=n_blocks,
            k_candidates=eval_config.k_candidates,
            split=split_name,
            template_split=split_name,
        ),
        seed=seed + 20_000,
    )
    audit = validate_stage8_examples(train + eval_examples)
    shortcut = shortcut_audit(eval_examples)
    selector = build_stage8_selector(config, seed=seed)
    fit_started = time.perf_counter()
    selector.fit(train, eval_examples)
    train_seconds = time.perf_counter() - fit_started
    predictions = selector.predict(eval_examples)
    labels = [example.label for example in eval_examples]
    accuracy = accuracy_from_predictions(predictions, labels)
    accuracy_by_task_family = accuracy_by_family(eval_examples, predictions)
    controls: Dict[str, object] = {}
    if run_controls:
        controls = run_stage8_control_suite(selector, eval_examples, predictions, labels, eval_config, seed)
    return {
        "config_id": config.config_id,
        "architecture_name": config.name,
        "model_kind": config.model_kind,
        "seed": int(seed),
        "n_blocks": int(n_blocks),
        "eval_split": eval_split,
        "accuracy": accuracy,
        "chance_accuracy": 1.0 / max(1, eval_config.k_candidates),
        "num_train_examples": len(train),
        "num_eval_examples": len(eval_examples),
        "predictions": predictions,
        "labels": labels,
        "task_families": [example.task_family for example in eval_examples],
        "accuracy_by_task_family": accuracy_by_task_family,
        "controls": controls,
        "dataset_audit_passes": bool(audit["passes"]),
        "dataset_audit_failures": audit["failures"],
        "shortcut_audit": shortcut,
        "diagnostics": selector.diagnostics(eval_examples[: min(32, len(eval_examples))]),
        "compute": estimate_stage8_compute(config, n_blocks, eval_config.k_candidates),
        "train_wall_clock_seconds": train_seconds,
    }


def run_stage8_control_suite(
    selector,
    examples: Sequence[Stage8Example],
    base_predictions: Sequence[int],
    labels: Sequence[int],
    eval_config: Stage8EvaluationConfig,
    seed: int,
) -> Dict[str, object]:
    base_accuracy = accuracy_from_predictions(base_predictions, labels)
    rows: Dict[str, object] = {}
    for control in eval_config.controls:
        controlled = apply_stage8_control(examples, control, seed + _control_offset(control))
        predictions = selector.predict(controlled)
        control_labels = [example.label for example in controlled]
        accuracy = accuracy_from_predictions(predictions, control_labels)
        expectation = control_expectation(control)
        if expectation == "degrade":
            passes = (base_accuracy - accuracy) >= eval_config.degradation_threshold
        elif expectation == "invariant":
            passes = abs(base_accuracy - accuracy) <= eval_config.invariance_tolerance
        else:
            passes = True
        rows[control] = {
            "accuracy": accuracy,
            "delta_from_base": accuracy - base_accuracy,
            "degradation": base_accuracy - accuracy,
            "expectation": expectation,
            "passes": bool(passes),
        }
    return rows


def compute_effective_capacity(
    rows: Sequence[Mapping[str, object]],
    eval_config: Stage8EvaluationConfig,
    require_controls: bool = True,
) -> Dict[str, object]:
    by_n: Dict[int, List[Mapping[str, object]]] = {}
    for row in rows:
        by_n.setdefault(int(row["n_blocks"]), []).append(row)
    capacity = 0
    n_summaries: Dict[str, object] = {}
    for n_blocks in sorted(by_n):
        values = by_n[n_blocks]
        accuracies = [float(row["accuracy"]) for row in values]
        chance = mean(float(row.get("chance_accuracy", 0.125)) for row in values)
        seed_count = len({int(row["seed"]) for row in values if "seed" in row})
        acc_mean = mean(accuracies) if accuracies else 0.0
        acc_std = pstdev(accuracies) if len(accuracies) > 1 else 0.0
        min_acc = min(accuracies) if accuracies else 0.0
        control_summary = _capacity_control_summary(values, eval_config)
        stable = (
            seed_count >= eval_config.min_stable_seeds
            and acc_std <= eval_config.max_seed_std
            and min_acc >= eval_config.min_seed_accuracy
        )
        accuracy_gate = acc_mean >= eval_config.accuracy_threshold and (acc_mean - chance) >= eval_config.chance_margin
        controls_gate = bool(control_summary["passes"]) if require_controls else True
        eligible = bool(stable and accuracy_gate and controls_gate)
        if eligible:
            capacity = max(capacity, n_blocks)
        n_summaries[str(n_blocks)] = {
            "mean_accuracy": acc_mean,
            "std_accuracy": acc_std,
            "min_accuracy": min_acc,
            "seed_count": seed_count,
            "chance_accuracy": chance,
            "stable": stable,
            "accuracy_gate": accuracy_gate,
            "controls_gate": controls_gate,
            "eligible": eligible,
            "controls": control_summary,
        }
    return {
        "capacity": int(capacity),
        "n_summaries": n_summaries,
        "criteria": {
            "accuracy_threshold": eval_config.accuracy_threshold,
            "chance_margin": eval_config.chance_margin,
            "min_stable_seeds": eval_config.min_stable_seeds,
            "max_seed_std": eval_config.max_seed_std,
            "min_seed_accuracy": eval_config.min_seed_accuracy,
            "require_controls": require_controls,
        },
    }


def capacity_ratio(latent_capacity: int, baseline_capacity: int) -> float:
    if baseline_capacity <= 0:
        return float("inf") if latent_capacity > 0 else 0.0
    return float(latent_capacity) / float(baseline_capacity)


def paired_bootstrap_ci(
    predictions_a: Sequence[int],
    predictions_b: Sequence[int],
    labels: Sequence[int],
    seed: int = 0,
    samples: int = 2000,
    alpha: float = 0.05,
) -> Dict[str, float]:
    if not labels:
        return {"mean_delta": 0.0, "lower": 0.0, "upper": 0.0}
    rng = random.Random(seed)
    deltas: List[float] = []
    n = len(labels)
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        acc_a = sum(1 for index in indices if predictions_a[index] == labels[index]) / n
        acc_b = sum(1 for index in indices if predictions_b[index] == labels[index]) / n
        deltas.append(acc_a - acc_b)
    deltas.sort()
    lower = deltas[int((alpha / 2.0) * (len(deltas) - 1))]
    upper = deltas[int((1.0 - alpha / 2.0) * (len(deltas) - 1))]
    mean_delta = accuracy_from_predictions(predictions_a, labels) - accuracy_from_predictions(predictions_b, labels)
    return {"mean_delta": mean_delta, "lower": lower, "upper": upper}


def accuracy_from_predictions(predictions: Sequence[int], labels: Sequence[int]) -> float:
    valid = [(prediction, label) for prediction, label in zip(predictions, labels) if label >= 0]
    if not valid:
        return 0.0
    return sum(1 for prediction, label in valid if int(prediction) == int(label)) / float(len(valid))


def accuracy_by_family(examples: Sequence[Stage8Example], predictions: Sequence[int]) -> Dict[str, float]:
    counts: Dict[str, List[int]] = {}
    for example, prediction in zip(examples, predictions):
        if example.label < 0:
            continue
        counts.setdefault(example.task_family, []).append(1 if int(prediction) == int(example.label) else 0)
    return {family: sum(values) / max(1, len(values)) for family, values in counts.items()}


def write_capacity_curves_csv(path: Path, results: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("config_id,architecture_name,model_kind,n_blocks,seed,accuracy,estimated_forward_compute,capacity\n")
        for result in results:
            capacity = int(result.get("capacity", {}).get("capacity", 0)) if isinstance(result.get("capacity"), Mapping) else 0
            for row in result.get("rows", []):  # type: ignore[union-attr]
                compute = row.get("compute", {}) if isinstance(row, Mapping) else {}
                handle.write(
                    "{config_id},{name},{kind},{n},{seed},{accuracy:.6f},{compute:.3f},{capacity}\n".format(
                        config_id=result.get("config_id", ""),
                        name=str(result.get("architecture_name", "")).replace(",", "_"),
                        kind=result.get("model_kind")
                        or (
                            result.get("architecture_parameters", {}).get("model_kind", "")
                            if isinstance(result.get("architecture_parameters"), Mapping)
                            else ""
                        ),
                        n=int(row.get("n_blocks", 0)),
                        seed=int(row.get("seed", 0)),
                        accuracy=float(row.get("accuracy", 0.0)),
                        compute=float(compute.get("estimated_forward_compute", 0.0)) if isinstance(compute, Mapping) else 0.0,
                        capacity=capacity,
                    )
                )


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _training_examples(
    config: Stage8ArchitectureConfig,
    n_blocks: int,
    seed: int,
    eval_config: Stage8EvaluationConfig,
) -> List[Stage8Example]:
    if config.curriculum in {"curriculum_8_to_target", "mixed_n"} and n_blocks > 8:
        schedule = [value for value in SCALE_SCHEDULE if value <= n_blocks]
        per_n = max(8, eval_config.train_examples // max(1, len(schedule)))
        rows: List[Stage8Example] = []
        for offset, schedule_n in enumerate(schedule):
            rows.extend(
                build_stage8_examples(
                    Stage8DatasetConfig(
                        n_examples=per_n,
                        n_blocks=schedule_n,
                        k_candidates=eval_config.k_candidates,
                        split="train",
                        template_split="train",
                    ),
                    seed=seed + offset * 1000,
                )
            )
        return rows[: eval_config.train_examples]
    return build_stage8_examples(
        Stage8DatasetConfig(
            n_examples=eval_config.train_examples,
            n_blocks=n_blocks,
            k_candidates=eval_config.k_candidates,
            split="train",
            template_split="train",
        ),
        seed=seed,
    )


def _capacity_control_summary(values: Sequence[Mapping[str, object]], eval_config: Stage8EvaluationConfig) -> Dict[str, object]:
    if not values:
        return {"passes": False, "controls": {}}
    by_control: Dict[str, List[Mapping[str, object]]] = {}
    for row in values:
        controls = row.get("controls", {})
        if not isinstance(controls, Mapping):
            continue
        for name, control_row in controls.items():
            if isinstance(control_row, Mapping):
                by_control.setdefault(str(name), []).append(control_row)
    summaries: Dict[str, object] = {}
    failures: List[str] = []
    for control, rows in by_control.items():
        accuracy = mean(float(row.get("accuracy", 0.0)) for row in rows)
        degradation = mean(float(row.get("degradation", 0.0)) for row in rows)
        expectation = str(rows[0].get("expectation", control_expectation(control)))
        if expectation == "degrade":
            passes = degradation >= eval_config.degradation_threshold
        elif expectation == "invariant":
            passes = abs(mean(float(row.get("delta_from_base", 0.0)) for row in rows)) <= eval_config.invariance_tolerance
        else:
            passes = all(bool(row.get("passes", True)) for row in rows)
        if not passes:
            failures.append(control)
        summaries[control] = {
            "mean_accuracy": accuracy,
            "mean_degradation": degradation,
            "expectation": expectation,
            "passes": passes,
        }
    required = set(eval_config.controls)
    missing = sorted(required - set(by_control))
    failures.extend(f"missing:{name}" for name in missing)
    return {"passes": not failures, "failures": failures, "controls": summaries}


def _control_offset(control: str) -> int:
    return sum(ord(ch) for ch in control) * 17
