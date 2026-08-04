from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from src.datasets.multiview_code_patch_selection import (
    MultiViewTaskExample,
    build_multiview_code_patch_splits,
    example_oracle_metadata,
)
from src.experiments.real_shared_weight_latent_coordination import predict_latent_system
from src.experiments.run_latent_vs_text_efficiency import _load_latent_checkpoint
from src.experiments.run_stage34_candidate_sanitization_diagnostics import _accuracy
from src.experiments.run_stage38_full_hard_validation import _labels
from src.experiments.run_stage5_multi_avenue_search import _stage5_config, _stage_for_phase


STAGE5_RESULTS = Path("results/stage5_multi_avenue_search_results.json")
STAGE4_RESULTS = Path("results/stage4_final_candidate_token_direct_validation_results.json")
RESULTS_PATH = Path("results/stage51_multi_avenue_portfolio_analysis.json")
REPORT_PATH = Path("reports/STAGE51_MULTI_AVENUE_PORTFOLIO_ANALYSIS.md")
VARIANT = "avenue4_goal_dropout05_lr3e4_clip1"
N_AVENUES = 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint-only Stage 5.1 multi-avenue portfolio analysis.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = run_analysis(device=str(args.device))
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_report(result), encoding="utf-8")
    print(f"stage5.1: wrote {RESULTS_PATH} and {REPORT_PATH}")


def run_analysis(device: str) -> Dict[str, object]:
    stage5 = json.loads(STAGE5_RESULTS.read_text(encoding="utf-8"))
    stage4 = json.loads(STAGE4_RESULTS.read_text(encoding="utf-8"))
    config = _stage5_config(stage5.get("config", {}))
    stage = _stage_for_phase(config, "final")
    final_rows = [
        row
        for row in stage5.get("phases", {}).get("final", [])
        if row.get("status") == "completed" and row.get("candidate") == VARIANT
    ]
    seed_rows = []
    for row in final_rows:
        seed = int(row["seed"])
        print(f"stage5.1 seed={seed}: loading checkpoints and fixed final split")
        splits = build_multiview_code_patch_splits(_dataset_config_from_row(row, stage), seed=seed, repo_root=Path("."))
        examples = list(splits["test"])
        labels = _labels(examples)
        trainable = _load_latent_checkpoint(Path(str(row["checkpoint_paths"]["trainable"])), device=device)
        frozen = _load_latent_checkpoint(Path(str(row["checkpoint_paths"]["frozen"])), device=device)
        train_preds = _portfolio_predictions(trainable, examples, seed, N_AVENUES)
        frozen_preds = _portfolio_predictions(frozen, examples, seed + 33_000, N_AVENUES)
        seed_rows.append(
            _seed_row(
                seed=seed,
                examples=examples,
                labels=labels,
                train_preds=train_preds,
                frozen_preds=frozen_preds,
                stage5_row=row,
                attention=_attention_usage(trainable, examples, device=device),
            )
        )
    result = {
        "metadata": {
            "stage": "stage5.1_multi_avenue_portfolio_analysis",
            "created_at_utc": _now(),
            "checkpoint_only": True,
            "trained_model_changed": False,
            "variant": VARIANT,
            "num_avenues": N_AVENUES,
            "device": device,
            "stage5_results": str(STAGE5_RESULTS),
            "stage4_results": str(STAGE4_RESULTS),
        },
        "stage4_baseline": _stage4_summary(stage4),
        "stage5_final_reference": _stage5_final_summary(stage5),
        "seed_rows": seed_rows,
        "summary": {},
    }
    result["summary"] = _summary(result)
    return result


def _dataset_config_from_row(row: Dict[str, object], stage):
    from src.datasets.multiview_code_patch_selection import MultiViewCodePatchDatasetConfig

    cfg = dict(row.get("dataset_config", {}))
    return MultiViewCodePatchDatasetConfig(
        n_train=int(cfg.get("n_train", stage.n_train)),
        n_dev=int(cfg.get("n_dev", stage.n_dev)),
        n_test=int(cfg.get("n_test", stage.n_test)),
        num_candidates=int(cfg.get("num_candidates", 8)),
        n_views=int(cfg.get("n_views", 4)),
        source_roots=tuple(cfg.get("source_roots", ("src", "tests"))),
        max_files=int(cfg.get("max_files", 40)),
        snippet_radius=int(cfg.get("snippet_radius", 2)),
        include_private_signal_tokens=bool(cfg.get("include_private_signal_tokens", False)),
        dataset_source=str(cfg.get("dataset_source", "real_import_restore_candidate_balanced_34b")),
        generator_version=str(cfg.get("generator_version", "real_import_restore_candidate_balanced_34b_stage5_multi_avenue")),
        candidate_representation=str(cfg.get("candidate_representation", "balanced_categories_v3")),
    )


def _portfolio_predictions(result, examples: Sequence[MultiViewTaskExample], seed: int, n_avenues: int) -> Dict[str, np.ndarray]:
    out = {"full": predict_latent_system(result, examples, "none", seed)}
    singles = []
    for avenue_index in range(n_avenues):
        pred = predict_latent_system(result, examples, f"avenue_only_{avenue_index}", seed + 1_000 + avenue_index)
        out[f"avenue_{avenue_index}"] = pred
        singles.append(pred)
    stacked = np.stack(singles, axis=0)
    rng = np.random.default_rng(seed + 77_777)
    choices = rng.integers(0, n_avenues, size=len(examples))
    out["random_single_avenue_per_example"] = stacked[choices, np.arange(len(examples))]
    return out


def _seed_row(
    seed: int,
    examples: Sequence[MultiViewTaskExample],
    labels: np.ndarray,
    train_preds: Dict[str, np.ndarray],
    frozen_preds: Dict[str, np.ndarray],
    stage5_row: Dict[str, object],
    attention: Dict[str, object],
) -> Dict[str, object]:
    train_acc = {name: _accuracy(pred, labels) for name, pred in train_preds.items()}
    frozen_acc = {name: _accuracy(pred, labels) for name, pred in frozen_preds.items()}
    avenue_names = [f"avenue_{idx}" for idx in range(N_AVENUES)]
    best_train_name = max(avenue_names, key=lambda name: train_acc[name])
    best_frozen_name = max(avenue_names, key=lambda name: frozen_acc[name])
    train_stack = np.stack([train_preds[name] for name in avenue_names], axis=0)
    frozen_stack = np.stack([frozen_preds[name] for name in avenue_names], axis=0)
    train_oracle = _oracle_best_accuracy(train_stack, labels)
    frozen_oracle = _oracle_best_accuracy(frozen_stack, labels)
    train_best_per_example = _oracle_best_avenue_indices(train_stack, labels)
    return {
        "seed": int(seed),
        "accuracy": {
            "full_4_avenue_trainable": train_acc["full"],
            "full_4_avenue_frozen": frozen_acc["full"],
            "full_4_avenue_delta": train_acc["full"] - frozen_acc["full"],
            **{f"single_avenue_{idx}_trainable": train_acc[f"avenue_{idx}"] for idx in range(N_AVENUES)},
            **{f"single_avenue_{idx}_frozen": frozen_acc[f"avenue_{idx}"] for idx in range(N_AVENUES)},
            "random_single_avenue_per_example_trainable": train_acc["random_single_avenue_per_example"],
            "random_single_avenue_per_example_frozen": frozen_acc["random_single_avenue_per_example"],
            "random_single_avenue_per_example_delta": train_acc["random_single_avenue_per_example"] - frozen_acc["random_single_avenue_per_example"],
            "best_fixed_single_avenue_trainable": train_acc[best_train_name],
            "best_fixed_single_avenue_frozen": frozen_acc[best_frozen_name],
            "best_fixed_single_avenue_delta": train_acc[best_train_name] - frozen_acc[best_frozen_name],
            "oracle_best_avenue_per_example_trainable": train_oracle,
            "oracle_best_avenue_per_example_frozen": frozen_oracle,
            "oracle_best_avenue_per_example_delta": train_oracle - frozen_oracle,
        },
        "best_fixed_single_avenue": {
            "trainable": best_train_name,
            "frozen": best_frozen_name,
        },
        "per_family_accuracy": _per_family_accuracy(examples, labels, train_preds),
        "family_dominant_avenues": _family_dominant_avenues(examples, labels, train_preds),
        "oracle_best_avenue_distribution": _distribution(train_best_per_example),
        "attention_usage": attention,
        "stage5_control_accuracy": {
            name: float(stage5_row.get("test_accuracy", {}).get(name, 0.0))
            for name in (
                "candidate_only",
                "candidate_metadata_only",
                "view_masked_candidates_visible",
                "schema_only",
                "null_evidence_values",
                "evidence_only_no_candidates",
                "value_shuffle_within_schema",
                "cross_example_view_bundle_shuffle",
                "candidate_evidence_mismatch",
                "schema_preserved_role_value_shuffle",
                "randomized_labels",
                "hidden_states_shuffled_across_examples",
            )
        },
    }


def _oracle_best_accuracy(stacked: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.any(stacked == labels.reshape(1, -1), axis=0)))


def _oracle_best_avenue_indices(stacked: np.ndarray, labels: np.ndarray) -> np.ndarray:
    correct = stacked == labels.reshape(1, -1)
    has_correct = correct.any(axis=0)
    indices = np.argmax(correct, axis=0).astype(np.int64)
    indices[~has_correct] = -1
    return indices


def _per_family_accuracy(examples: Sequence[MultiViewTaskExample], labels: np.ndarray, preds: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    families = _families(examples)
    out: Dict[str, Dict[str, float]] = {}
    for family in sorted(set(families)):
        mask = np.asarray([value == family for value in families], dtype=bool)
        if not mask.any():
            continue
        row = {"n": int(mask.sum())}
        for name, pred in preds.items():
            if name == "random_single_avenue_per_example" or name.startswith("avenue_") or name == "full":
                row[name] = _accuracy(pred[mask], labels[mask])
        out[family] = row
    return out


def _family_dominant_avenues(examples: Sequence[MultiViewTaskExample], labels: np.ndarray, preds: Dict[str, np.ndarray]) -> Dict[str, object]:
    families = _families(examples)
    out = {}
    for family in sorted(set(families)):
        mask = np.asarray([value == family for value in families], dtype=bool)
        scores = {
            f"avenue_{idx}": _accuracy(preds[f"avenue_{idx}"][mask], labels[mask])
            for idx in range(N_AVENUES)
            if mask.any()
        }
        best_score = max(scores.values()) if scores else 0.0
        out[family] = {
            "n": int(mask.sum()),
            "accuracy": scores,
            "dominant_avenues": [name for name, value in scores.items() if abs(value - best_score) <= 1e-12],
        }
    return out


def _families(examples: Sequence[MultiViewTaskExample]) -> List[str]:
    return [str(example_oracle_metadata(example).get("problem_family", "unknown")) for example in examples]


def _attention_usage(result, examples: Sequence[MultiViewTaskExample], device: str) -> Dict[str, object]:
    system = result.system
    coordinator = system.coordinator
    if not hasattr(coordinator, "layers") or not hasattr(coordinator, "token_projection"):
        return {"available": False, "reason": "coordinator does not expose candidate-token attention layers"}
    totals = np.zeros(N_AVENUES, dtype=np.float64)
    gold_totals = np.zeros(N_AVENUES, dtype=np.float64)
    n_batches = 0
    with torch.no_grad():
        for start in range(0, len(examples), 64):
            batch = list(examples[start : start + 64])
            readouts = system.collect_clone_representations(batch, condition="none", seed=91_000)
            token_states = readouts["token_states"]
            token_mask = readouts["token_mask"].bool()
            role_ids = torch.arange(system.n_roles, dtype=torch.long, device=token_states.device).view(1, system.n_roles)
            role_ids = role_ids.expand(token_states.shape[0], system.n_roles)
            candidate_features = _candidate_feature_tensor(batch, token_states.device)
            weights = _first_layer_avenue_attention(coordinator, role_ids, candidate_features, token_states, token_mask, readouts.get("avenue_ids"))
            totals += weights["all_candidates"]
            labels = torch.as_tensor([example.label for example in batch], dtype=torch.long, device=token_states.device)
            label_index = labels.detach().cpu().numpy()
            gold_totals += weights["gold_candidates"][np.arange(len(batch)), label_index].sum(axis=0)
            n_batches += 1
    if totals.sum() > 0:
        totals = totals / totals.sum()
    if gold_totals.sum() > 0:
        gold_totals = gold_totals / gold_totals.sum()
    return {
        "available": True,
        "source": "first_cross_attention_layer_weights_grouped_by_avenue",
        "all_candidate_attention_by_avenue": {str(idx): float(value) for idx, value in enumerate(totals)},
        "gold_candidate_attention_by_avenue": {str(idx): float(value) for idx, value in enumerate(gold_totals)},
        "batches": int(n_batches),
    }


def _first_layer_avenue_attention(coordinator, role_ids, candidate_features, token_states, token_mask, avenue_ids):
    batch, roles, avenues, tokens, _dim = token_states.shape
    projected = coordinator.token_projection(token_states)
    role = coordinator.role_embedding(role_ids.clamp(min=0, max=coordinator.role_embedding.num_embeddings - 1)).view(batch, roles, 1, 1, -1)
    if avenue_ids is None:
        ids = torch.arange(avenues, dtype=torch.long, device=token_states.device).view(1, 1, avenues)
        avenue_ids = ids.expand(batch, roles, avenues)
    avenue = coordinator.avenue_embedding(avenue_ids.clamp(min=0, max=coordinator.avenue_embedding.num_embeddings - 1)).unsqueeze(3)
    projected = projected + role + avenue
    flat_tokens = projected.reshape(batch, roles * avenues * tokens, projected.shape[-1])
    flat_mask = token_mask.reshape(batch, roles * avenues * tokens)
    queries = coordinator.candidate_projection(candidate_features)
    layer = coordinator.layers[0]
    _attended, weights = layer.attention(
        query=layer.norm1(queries),
        key=flat_tokens,
        value=flat_tokens,
        key_padding_mask=~flat_mask,
        need_weights=True,
        average_attn_weights=False,
    )
    # [batch, heads, candidates, roles * avenues * tokens]
    weights = weights.reshape(batch, weights.shape[1], weights.shape[2], roles, avenues, tokens)
    by_avenue = weights.sum(dim=(1, 3, 5))  # [batch, candidates, avenues]
    all_candidates = by_avenue.sum(dim=(0, 1)).detach().cpu().numpy()
    gold_candidates = by_avenue.detach().cpu().numpy()  # [batch, candidates, avenues]
    return {"all_candidates": all_candidates, "gold_candidates": gold_candidates}


def _candidate_feature_tensor(examples: Sequence[MultiViewTaskExample], device) -> torch.Tensor:
    from src.experiments.real_shared_weight_latent_coordination import _candidate_feature_tensor as feature_tensor

    return feature_tensor(examples, device)


def _distribution(values: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values.tolist():
        out[str(int(value))] = out.get(str(int(value)), 0) + 1
    return out


def _summary(result: Dict[str, object]) -> Dict[str, object]:
    rows = result["seed_rows"]
    metric_names = [
        "full_4_avenue",
        *[f"single_avenue_{idx}" for idx in range(N_AVENUES)],
        "random_single_avenue_per_example",
        "best_fixed_single_avenue",
        "oracle_best_avenue_per_example",
    ]
    metrics = {}
    for name in metric_names:
        train_key = f"{name}_trainable"
        frozen_key = f"{name}_frozen"
        train_values = [float(row["accuracy"][train_key]) for row in rows]
        frozen_values = [float(row["accuracy"][frozen_key]) for row in rows]
        metrics[name] = {
            "mean_accuracy": _mean(train_values),
            "min_seed_accuracy": min(train_values) if train_values else 0.0,
            "weak_seeds_below_060": sum(value < 0.60 for value in train_values),
            "mean_frozen_accuracy": _mean(frozen_values),
            "mean_trainable_frozen_delta": _mean([t - f for t, f in zip(train_values, frozen_values)]),
        }
    control_means = {}
    for name in rows[0]["stage5_control_accuracy"].keys() if rows else []:
        control_means[name] = _mean([float(row["stage5_control_accuracy"].get(name, 0.0)) for row in rows])
    family_dominance = _family_dominance_summary(rows)
    attention = _attention_summary(rows)
    stage4 = result["stage4_baseline"]
    full = metrics["full_4_avenue"]
    best_fixed = metrics["best_fixed_single_avenue"]
    portfolio_benefit = {
        "full_beats_stage4_mean_accuracy": full["mean_accuracy"] > float(stage4["mean_trainable_accuracy"]),
        "full_has_fewer_weak_seeds_than_stage4": int(full["weak_seeds_below_060"]) < int(stage4["weak_seeds_below_060"]),
        "full_beats_best_fixed_single_avenue_mean": full["mean_accuracy"] > best_fixed["mean_accuracy"],
        "different_avenues_dominate_families_or_seeds": bool(family_dominance["unique_dominant_avenues"] > 1 or len(_seed_best_fixed_set(rows)) > 1),
        "stage4_controls_pass": all(value <= 0.18 for value in control_means.values()),
    }
    portfolio_benefit["supports_portfolio_benefit"] = bool(
        portfolio_benefit["full_beats_stage4_mean_accuracy"]
        and portfolio_benefit["full_has_fewer_weak_seeds_than_stage4"]
        and (
            portfolio_benefit["full_beats_best_fixed_single_avenue_mean"]
            or portfolio_benefit["different_avenues_dominate_families_or_seeds"]
        )
        and portfolio_benefit["stage4_controls_pass"]
    )
    best_single_mean = max(metrics[f"single_avenue_{idx}"]["mean_accuracy"] for idx in range(N_AVENUES))
    seed_single_recovers = sum(
        max(float(row["accuracy"][f"single_avenue_{idx}_trainable"]) for idx in range(N_AVENUES))
        >= float(row["accuracy"]["full_4_avenue_trainable"]) - 0.02
        for row in rows
    )
    joint_fusion = {
        "full_substantially_beats_every_single_avenue_mean": full["mean_accuracy"] > best_single_mean + 0.05,
        "best_fixed_single_avenue_does_not_recover_most_mean": best_single_mean < 0.90 * full["mean_accuracy"],
        "no_seed_has_single_avenue_within_0.02_of_full": seed_single_recovers == 0,
        "seed_count_where_single_avenue_recovers_full_within_0.02": int(seed_single_recovers),
    }
    joint_fusion["supports_joint_multi_avenue_fusion"] = all(joint_fusion.values())
    return {
        "metrics": metrics,
        "control_means": control_means,
        "per_family_accuracy_summary": _per_family_accuracy_summary(rows),
        "family_dominance": family_dominance,
        "attention": attention,
        "same_compute_duplicated_single_avenue_baseline_available": False,
        "same_compute_duplicated_single_avenue_baseline_note": "No trained duplicated-single-avenue Stage 5 checkpoint was available; no new model was trained for Stage 5.1.",
        "portfolio_benefit": portfolio_benefit,
        "joint_fusion": joint_fusion,
    }


def _stage4_summary(stage4: Dict[str, object]) -> Dict[str, object]:
    rows = [row for row in stage4.get("stage4_final_rows", []) if row.get("status") == "completed"]
    train = [float(row["test_accuracy"]["trainable"]) for row in rows]
    frozen = [float(row["test_accuracy"]["frozen"]) for row in rows]
    return {
        "source": str(STAGE4_RESULTS),
        "n_completed": len(rows),
        "mean_trainable_accuracy": _mean(train),
        "mean_frozen_accuracy": _mean(frozen),
        "mean_delta": _mean([t - f for t, f in zip(train, frozen)]),
        "min_seed_accuracy": min(train) if train else 0.0,
        "weak_seeds_below_060": sum(value < 0.60 for value in train),
    }


def _stage5_final_summary(stage5: Dict[str, object]) -> Dict[str, object]:
    summary = stage5.get("summary", {})
    criteria = summary.get("final_success_criteria", [])
    return {
        "source": str(STAGE5_RESULTS),
        "passed_stage5_final_gates": bool(summary.get("passed_stage5_final_gates", False)),
        "failed_gates": [item.get("criterion") for item in criteria if not bool(item.get("pass"))],
    }


def _family_dominance_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    counts: Dict[str, int] = {}
    per_family: Dict[str, Dict[str, int]] = {}
    for row in rows:
        for family, info in row.get("family_dominant_avenues", {}).items():
            fam_counts = per_family.setdefault(family, {})
            for avenue in info.get("dominant_avenues", []):
                counts[avenue] = counts.get(avenue, 0) + 1
                fam_counts[avenue] = fam_counts.get(avenue, 0) + 1
    return {
        "dominant_avenue_counts": counts,
        "per_family_dominant_counts": per_family,
        "unique_dominant_avenues": len(counts),
    }


def _per_family_accuracy_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    values: Dict[str, Dict[str, List[float]]] = {}
    metric_names = ["full", *[f"avenue_{idx}" for idx in range(N_AVENUES)], "random_single_avenue_per_example"]
    for row in rows:
        for family, metrics in row.get("per_family_accuracy", {}).items():
            fam = values.setdefault(family, {name: [] for name in metric_names})
            for name in metric_names:
                if name in metrics:
                    fam[name].append(float(metrics[name]))
    out = {}
    for family, metrics in values.items():
        row = {name: _mean(vals) for name, vals in metrics.items()}
        single_vals = {name: row[name] for name in row if name.startswith("avenue_")}
        best_name = max(single_vals, key=lambda name: single_vals[name]) if single_vals else ""
        row["best_single_avenue"] = best_name
        row["best_single_accuracy"] = float(single_vals.get(best_name, 0.0))
        row["full_minus_best_single"] = float(row.get("full", 0.0) - row["best_single_accuracy"])
        out[family] = row
    return out


def _attention_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    all_values = []
    gold_values = []
    for row in rows:
        attn = row.get("attention_usage", {})
        if not attn.get("available"):
            continue
        all_values.append([float(attn["all_candidate_attention_by_avenue"].get(str(idx), 0.0)) for idx in range(N_AVENUES)])
        gold_values.append([float(attn["gold_candidate_attention_by_avenue"].get(str(idx), 0.0)) for idx in range(N_AVENUES)])
    if not all_values:
        return {"available": False}
    all_arr = np.asarray(all_values, dtype=np.float64)
    gold_arr = np.asarray(gold_values, dtype=np.float64)
    return {
        "available": True,
        "mean_all_candidate_attention_by_avenue": {str(idx): float(all_arr[:, idx].mean()) for idx in range(N_AVENUES)},
        "mean_gold_candidate_attention_by_avenue": {str(idx): float(gold_arr[:, idx].mean()) for idx in range(N_AVENUES)},
    }


def _seed_best_fixed_set(rows: Sequence[Dict[str, object]]) -> set[str]:
    return {str(row.get("best_fixed_single_avenue", {}).get("trainable")) for row in rows}


def _render_report(result: Dict[str, object]) -> str:
    summary = result["summary"]
    metrics = summary["metrics"]
    lines = [
        "# Stage 5.1 Multi-Avenue Portfolio Analysis",
        "",
        "## Scope",
        "",
        "- Checkpoint-only analysis using saved Stage 5 final checkpoints.",
        "- No retraining, architecture change, dataset change, or control change was performed.",
        "- Framing: avenue portfolio benefit, not mandatory joint avenue fusion.",
        "",
        "## Portfolio Metrics",
        "",
        "| mechanism | mean acc | min seed | weak <0.60 | frozen mean | delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    order = [
        "full_4_avenue",
        "single_avenue_0",
        "single_avenue_1",
        "single_avenue_2",
        "single_avenue_3",
        "random_single_avenue_per_example",
        "best_fixed_single_avenue",
        "oracle_best_avenue_per_example",
    ]
    for name in order:
        row = metrics[name]
        lines.append(
            f"| {name} | {row['mean_accuracy']:.4f} | {row['min_seed_accuracy']:.4f} | {int(row['weak_seeds_below_060'])} | {row['mean_frozen_accuracy']:.4f} | {row['mean_trainable_frozen_delta']:.4f} |"
        )
    stage4 = result["stage4_baseline"]
    lines.append(
        f"| Stage 4 4x1 baseline | {float(stage4['mean_trainable_accuracy']):.4f} | {float(stage4['min_seed_accuracy']):.4f} | {int(stage4['weak_seeds_below_060'])} | {float(stage4['mean_frozen_accuracy']):.4f} | {float(stage4['mean_delta']):.4f} |"
    )
    lines.extend(["", "## Controls", "", "| control | mean | pass <= 0.18 |", "|---|---:|---|"])
    for name, value in summary["control_means"].items():
        lines.append(f"| {name} | {float(value):.4f} | `{bool(float(value) <= 0.18)}` |")
    lines.extend(["", "## Avenue Dominance", ""])
    lines.append(f"- Seed-level best fixed avenues: `{sorted(_seed_best_fixed_set(result['seed_rows']))}`")
    lines.append(f"- Dominant avenue counts across families/seeds: `{summary['family_dominance']['dominant_avenue_counts']}`")
    lines.append(f"- Same-compute duplicated single-avenue baseline available: `{summary['same_compute_duplicated_single_avenue_baseline_available']}`")
    lines.append(f"- Note: {summary['same_compute_duplicated_single_avenue_baseline_note']}")
    lines.extend(["", "## Attention Usage", ""])
    attention = summary["attention"]
    if attention.get("available"):
        lines.append(f"- Mean all-candidate attention by avenue: `{attention['mean_all_candidate_attention_by_avenue']}`")
        lines.append(f"- Mean gold-candidate attention by avenue: `{attention['mean_gold_candidate_attention_by_avenue']}`")
    else:
        lines.append("- Attention usage was not available.")
    lines.extend(
        [
            "",
            "## Per-Family Accuracy",
            "",
            "| family | full | best single | best avenue | full - best | random single |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for family, row in summary["per_family_accuracy_summary"].items():
        lines.append(
            f"| {family} | {float(row.get('full', 0.0)):.4f} | {float(row.get('best_single_accuracy', 0.0)):.4f} | {row.get('best_single_avenue', '')} | {float(row.get('full_minus_best_single', 0.0)):.4f} | {float(row.get('random_single_avenue_per_example', 0.0)):.4f} |"
        )
    lines.extend(["", "## Per-Family Dominance", "", "| family | dominant avenue counts |", "|---|---|"])
    for family, counts in summary["family_dominance"]["per_family_dominant_counts"].items():
        lines.append(f"| {family} | `{counts}` |")
    portfolio = summary["portfolio_benefit"]
    joint = summary["joint_fusion"]
    lines.extend(
        [
            "",
            "## Interpretation Gates",
            "",
            f"- Full beats Stage 4 mean accuracy: `{portfolio['full_beats_stage4_mean_accuracy']}`.",
            f"- Full has fewer weak seeds than Stage 4: `{portfolio['full_has_fewer_weak_seeds_than_stage4']}`.",
            f"- Full beats best fixed single avenue on average: `{portfolio['full_beats_best_fixed_single_avenue_mean']}`.",
            f"- Different avenues dominate families/seeds: `{portfolio['different_avenues_dominate_families_or_seeds']}`.",
            f"- All Stage 4 controls pass: `{portfolio['stage4_controls_pass']}`.",
            f"- Supports portfolio benefit: `{portfolio['supports_portfolio_benefit']}`.",
            f"- Full substantially beats every single avenue by mean: `{joint['full_substantially_beats_every_single_avenue_mean']}`.",
            f"- Best fixed single avenue does not recover most of mean full result: `{joint['best_fixed_single_avenue_does_not_recover_most_mean']}`.",
            f"- Seeds where one single avenue is within 0.02 of full: `{joint['seed_count_where_single_avenue_recovers_full_within_0.02']}`.",
            f"- Supports joint multi-avenue fusion: `{joint['supports_joint_multi_avenue_fusion']}`.",
            "",
            "## Conservative Conclusion",
            "",
        ]
    )
    if portfolio["supports_portfolio_benefit"]:
        lines.append("- Stage 5.1 supports a portfolio-benefit interpretation under the requested criteria.")
    else:
        lines.append("- Stage 5.1 does not support the requested portfolio-benefit criteria.")
    if not joint["supports_joint_multi_avenue_fusion"]:
        lines.append("- Do not claim joint multi-avenue fusion: single-avenue ablations recover much of the result.")
    return "\n".join(lines) + "\n"


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
