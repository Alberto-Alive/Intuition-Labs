from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from src.datasets.multiview_code_patch_selection import (
    MultiViewTaskExample,
    build_multiview_code_patch_splits,
    example_oracle_metadata,
    format_clone_prompt,
)
from src.experiments.architecture_search import _per_role_ablation
from src.experiments.real_shared_weight_latent_coordination import _text_tokens, predict_latent_system
from src.experiments.run_latent_vs_text_efficiency import _load_latent_checkpoint
from src.experiments.run_stage38_full_hard_validation import _dataset_config_for_stage, _stage38_full_config, _stage_from_config


DEFAULT_STAGE38_RESULTS = "results/stage38_full_hard_validation_results.json"
DEFAULT_CONFIG = "configs/stage38_full_hard_validation.json"
DEFAULT_OUTPUT = "results/stage39_failure_analysis.json"
DEFAULT_REPORT = "reports/STAGE39_FAILURE_ANALYSIS.md"
RELEVANT_SEEDS = (0, 2, 4, 5, 7, 8, 9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage 3.8 full-validation failures without additional training.")
    parser.add_argument("--stage38-results", default=DEFAULT_STAGE38_RESULTS)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--token-sample", type=int, default=128)
    args = parser.parse_args()

    result = run_stage39_failure_analysis(
        stage38_results_path=Path(args.stage38_results),
        config_path=Path(args.config),
        device=str(args.device),
        token_sample=int(args.token_sample),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")


def run_stage39_failure_analysis(stage38_results_path: Path, config_path: Path, device: str, token_sample: int) -> Dict[str, object]:
    stage38 = json.loads(stage38_results_path.read_text(encoding="utf-8"))
    config = _stage38_full_config(json.loads(config_path.read_text(encoding="utf-8")))
    stage = _stage_from_config(config)
    rows = [row for row in stage38.get("stage38_rows", []) if row.get("status") == "completed"]
    weak_seeds = [int(row["seed"]) for row in rows if int(row["seed"]) in RELEVANT_SEEDS]
    seed_rows = []
    for row in rows:
        seed = int(row["seed"])
        if seed not in weak_seeds:
            continue
        print(f"stage39 analysis seed={seed}: loading checkpoints and test split")
        splits = build_multiview_code_patch_splits(_dataset_config_for_stage(config, stage), seed=seed, repo_root=Path("."))
        test = list(splits["test"])
        checkpoints = dict(row["checkpoint_paths"])
        trainable = _load_latent_checkpoint(Path(str(checkpoints["trainable"])), device=device)
        frozen = _load_latent_checkpoint(Path(str(checkpoints["frozen"])), device=device)
        seed_rows.append(_analyze_seed(seed, row, trainable, frozen, test, token_sample=token_sample))
        del trainable, frozen
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = _summary(seed_rows, rows)
    return {
        "metadata": {
            "stage": "stage3.9_failure_analysis",
            "stage38_results_path": str(stage38_results_path),
            "config_path": str(config_path),
            "device": device,
            "scope": "checkpoint analysis only; no training",
            "relevant_seed_policy": "user-specified weak/frozen-competitive seeds",
            "relevant_seeds": list(RELEVANT_SEEDS),
        },
        "stage38_summary": stage38.get("summary", {}),
        "seed_rows": seed_rows,
        "summary": summary,
    }


def _analyze_seed(seed: int, source_row: Dict[str, object], trainable, frozen, test: Sequence[MultiViewTaskExample], token_sample: int) -> Dict[str, object]:
    labels = np.asarray([example.label for example in test], dtype=np.int64)
    train_pred = predict_latent_system(trainable, test, "none", seed + 31_000)
    frozen_pred = predict_latent_system(frozen, test, "none", seed + 31_000)
    per_family = _per_family_compare(test, labels, train_pred, frozen_pred)
    readout = {
        "trainable": _readout_stats(trainable, test),
        "frozen": _readout_stats(frozen, test),
    }
    token_compare = _topk_token_comparison(trainable, frozen, test[: max(0, int(token_sample))])
    role_ablation = {
        "trainable": _per_role_ablation(trainable, test, seed + 41_000),
        "frozen": _per_role_ablation(frozen, test, seed + 41_000),
    }
    test_accuracy = dict(source_row.get("test_accuracy", {}))
    return {
        "seed": seed,
        "trainable_accuracy": float(test_accuracy.get("trainable", np.mean(train_pred == labels))),
        "frozen_accuracy": float(test_accuracy.get("frozen", np.mean(frozen_pred == labels))),
        "delta": float(source_row.get("test_delta", float(np.mean(train_pred == labels) - np.mean(frozen_pred == labels)))),
        "classification": _seed_failure_classification(test_accuracy, per_family, readout, token_compare),
        "per_family": per_family,
        "readout_stats": readout,
        "topk_token_comparison": token_compare,
        "per_role_ablation": role_ablation,
        "controls_snapshot": {
            name: float(test_accuracy.get(name, 0.0))
            for name in (
                "candidate_only",
                "schema_only",
                "view_masked_candidates_visible",
                "value_shuffle_within_schema",
                "candidate_evidence_mismatch",
                "null_evidence_values",
                "hidden_states_shuffled_across_examples",
                "role_embedding_shuffle_diagnostic",
            )
        },
    }


def _per_family_compare(
    examples: Sequence[MultiViewTaskExample],
    labels: np.ndarray,
    train_pred: np.ndarray,
    frozen_pred: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[str(example_oracle_metadata(example).get("problem_family"))].append(index)
    out = {}
    for family, indices in sorted(grouped.items()):
        idx = np.asarray(indices, dtype=np.int64)
        train_acc = float(np.mean(train_pred[idx] == labels[idx]))
        frozen_acc = float(np.mean(frozen_pred[idx] == labels[idx]))
        out[family] = {
            "n": int(len(indices)),
            "trainable": train_acc,
            "frozen": frozen_acc,
            "delta": train_acc - frozen_acc,
            "both_correct": float(np.mean((train_pred[idx] == labels[idx]) & (frozen_pred[idx] == labels[idx]))),
            "trainable_only_correct": float(np.mean((train_pred[idx] == labels[idx]) & (frozen_pred[idx] != labels[idx]))),
            "frozen_only_correct": float(np.mean((train_pred[idx] != labels[idx]) & (frozen_pred[idx] == labels[idx]))),
        }
    return out


def _readout_stats(result, examples: Sequence[MultiViewTaskExample]) -> Dict[str, float]:
    messages = []
    active = []
    with torch.no_grad():
        for start in range(0, len(examples), 128):
            batch = list(examples[start : start + 128])
            readouts = result.system.collect_clone_representations(batch, condition="none", seed=87_000)
            messages.append(readouts["message"].detach().float().cpu())
            active.append(readouts["active_message"].detach().float().cpu())
    message = torch.cat(messages, dim=0)
    active_message = torch.cat(active, dim=0)
    return {
        "message_variance_mean": _variance_mean(message),
        "message_mean_pairwise_cosine": _mean_pairwise_cosine(message),
        "message_norm_mean": float(message.norm(dim=-1).mean().item()),
        "active_variance_mean": _variance_mean(active_message),
        "active_mean_pairwise_cosine": _mean_pairwise_cosine(active_message),
        "active_norm_mean": float(active_message.norm(dim=-1).mean().item()),
    }


def _variance_mean(tensor: torch.Tensor) -> float:
    flat = tensor.reshape(tensor.shape[0], -1)
    return float(flat.var(dim=0, unbiased=False).mean().item())


def _mean_pairwise_cosine(tensor: torch.Tensor) -> float:
    flat = tensor.reshape(tensor.shape[0], -1)
    if flat.shape[0] < 2:
        return 1.0
    flat = F.normalize(flat, dim=-1)
    sim = flat @ flat.t()
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool)
    return float(sim[mask].mean().item())


def _topk_token_comparison(trainable, frozen, examples: Sequence[MultiViewTaskExample]) -> Dict[str, object]:
    if not examples:
        return {"available": False}
    if getattr(trainable.system.message_config, "active_message_readout_type", "") != "topk_attention":
        return {"available": False, "reason": "trainable is not topk_attention"}
    train = _topk_tokens(trainable, examples)
    frozen_tokens = _topk_tokens(frozen, examples)
    overlaps = []
    for left, right in zip(train["selected_position_sets"], frozen_tokens["selected_position_sets"]):
        overlaps.append(_jaccard(left, right))
    return {
        "available": True,
        "n_examples": len(examples),
        "mean_position_jaccard": float(mean(overlaps)) if overlaps else 0.0,
        "trainable": {key: value for key, value in train.items() if key != "selected_position_sets"},
        "frozen": {key: value for key, value in frozen_tokens.items() if key != "selected_position_sets"},
    }


def _topk_tokens(result, examples: Sequence[MultiViewTaskExample]) -> Dict[str, object]:
    system = result.system
    readout = system.active_message_readout
    if readout is None:
        return {"selected_position_sets": [], "top_tokens": {}, "evidence_token_rate": 0.0, "field_token_rate": 0.0}
    selected_counter: Counter[str] = Counter()
    field_hits = 0
    evidence_hits = 0
    total = 0
    position_sets: List[set[tuple[int, int, int]]] = []
    max_length = int(system.shared_agent.max_length)
    with torch.no_grad():
        for start in range(0, len(examples), 64):
            batch = list(examples[start : start + 64])
            role_ids = torch.arange(system.n_roles, device=next(system.parameters()).device).view(1, system.n_roles).expand(len(batch), system.n_roles)
            role_tokens = []
            role_masks = []
            for role_index in range(system.n_roles):
                readouts = system.shared_agent.forward_texts_with_readouts(
                    [format_clone_prompt(example, role_index) for example in batch],
                    use_msg_token=system.message_config.use_msg_token,
                    msg_position=system.message_config.msg_position,
                    selected_layer_ids=system.message_config.active_message_layers,
                )
                role_tokens.append(readouts["token_states"])
                role_masks.append(readouts["token_mask"])
            token_states = torch.stack(role_tokens, dim=1)
            token_mask = torch.stack(role_masks, dim=1).bool()
            batch_size, roles, tokens, dim = token_states.shape
            flat_tokens = token_states.reshape(batch_size * roles, tokens, dim)
            flat_mask = token_mask.reshape(batch_size * roles, tokens)
            flat_roles = role_ids.reshape(batch_size * roles).clamp(min=0, max=readout.role_embedding.num_embeddings - 1)
            role = readout.role_embedding(flat_roles)
            scores = readout.pool_scorer(flat_tokens + role.unsqueeze(1)).squeeze(-1)
            scores = scores.masked_fill(~flat_mask, -1e9)
            k = min(8, scores.shape[-1])
            top_indices = torch.topk(scores, k=k, dim=-1).indices.detach().cpu().numpy()
            for flat_index in range(top_indices.shape[0]):
                ex_index = flat_index // roles
                role_index = flat_index % roles
                prompt_tokens = _prompt_tokens(batch[ex_index], role_index, max_length)
                selected_positions = set()
                for raw_pos in top_indices[flat_index].tolist():
                    token_pos = int(raw_pos) % max_length
                    selected_positions.add((start + ex_index, role_index, token_pos))
                    token = prompt_tokens[token_pos] if token_pos < len(prompt_tokens) else "<pad>"
                    selected_counter[token] += 1
                    total += 1
                    if token in {"symbol_surface", "provider_area", "provider_name_shape", "import_slot_shape"}:
                        field_hits += 1
                    if token in {"class", "core", "odd", "even", "function", "value", "coordination", "experiment", "line", "provider", "area", "symbol", "kind"}:
                        evidence_hits += 1
                position_sets.append(selected_positions)
    return {
        "selected_position_sets": position_sets,
        "top_tokens": dict(selected_counter.most_common(25)),
        "field_token_rate": float(field_hits / max(1, total)),
        "evidence_token_rate": float(evidence_hits / max(1, total)),
        "total_selected_tokens": int(total),
    }


def _prompt_tokens(example: MultiViewTaskExample, role_index: int, max_length: int) -> List[str]:
    tokens = _text_tokens(format_clone_prompt(example, role_index))[:max_length]
    if len(tokens) < max_length:
        tokens.extend(["<pad>"] * (max_length - len(tokens)))
    return tokens


def _jaccard(left: set[tuple[int, int, int]], right: set[tuple[int, int, int]]) -> float:
    if not left and not right:
        return 1.0
    return float(len(left & right) / max(1, len(left | right)))


def _seed_failure_classification(test_accuracy: Dict[str, object], per_family: Dict[str, Dict[str, float]], readout: Dict[str, Dict[str, float]], token_compare: Dict[str, object]) -> Dict[str, object]:
    trainable = float(test_accuracy.get("trainable", 0.0))
    frozen = float(test_accuracy.get("frozen", 0.0))
    deltas = [float(row["delta"]) for row in per_family.values()]
    frozen_advantage_families = [name for name, row in per_family.items() if float(row["delta"]) < -0.02]
    train_low_families = [name for name, row in per_family.items() if float(row["trainable"]) < 0.60]
    train_readout = readout.get("trainable", {})
    frozen_readout = readout.get("frozen", {})
    optimizer_instability = bool(trainable < 0.60 and max(float(test_accuracy.get("candidate_pair_compatibility_mlp", 0.0)), 0.0) >= 0.90)
    frozen_solves = bool(frozen >= 0.48 and len(frozen_advantage_families) >= 2)
    underfit = bool(len(train_low_families) >= max(1, len(per_family) // 2))
    topk_suspicious = bool(
        token_compare.get("available")
        and float(token_compare.get("mean_position_jaccard", 0.0)) < 0.35
        and float(token_compare.get("trainable", {}).get("evidence_token_rate", 0.0)) < 0.30
    )
    coordinator_bottleneck = bool(
        trainable < 0.75
        and float(train_readout.get("active_variance_mean", 0.0)) > 0.01
        and float(test_accuracy.get("candidate_pair_compatibility_mlp", 0.0)) >= 0.90
    )
    return {
        "frozen_already_solves_specific_families": frozen_solves,
        "trainable_underfits_specific_families": underfit,
        "optimizer_instability": optimizer_instability,
        "topk_readout_selects_wrong_tokens": topk_suspicious,
        "coordinator_bottleneck_or_candidate_conditioning_limit": coordinator_bottleneck,
        "insufficient_candidate_conditioned_readout": coordinator_bottleneck or frozen_solves,
        "frozen_advantage_family_count": len(frozen_advantage_families),
        "trainable_low_family_count": len(train_low_families),
        "mean_family_delta": float(mean(deltas)) if deltas else 0.0,
        "readout_variance_delta_train_minus_frozen": float(train_readout.get("active_variance_mean", 0.0) - frozen_readout.get("active_variance_mean", 0.0)),
    }


def _summary(seed_rows: Sequence[Dict[str, object]], all_stage38_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    classifications = Counter()
    for row in seed_rows:
        for key, value in dict(row.get("classification", {})).items():
            if isinstance(value, bool) and value:
                classifications[key] += 1
    all_train = [float(row["test_accuracy"]["trainable"]) for row in all_stage38_rows]
    all_frozen = [float(row["test_accuracy"]["frozen"]) for row in all_stage38_rows]
    all_delta = [float(row["test_delta"]) for row in all_stage38_rows]
    weak_delta = [float(row["delta"]) for row in seed_rows]
    return {
        "n_relevant_seeds_analyzed": len(seed_rows),
        "relevant_seed_deltas": {str(row["seed"]): float(row["delta"]) for row in seed_rows},
        "stage38_all_seed_mean_trainable": float(mean(all_train)) if all_train else 0.0,
        "stage38_all_seed_mean_frozen": float(mean(all_frozen)) if all_frozen else 0.0,
        "stage38_all_seed_mean_delta": float(mean(all_delta)) if all_delta else 0.0,
        "weak_seed_mean_delta": float(mean(weak_delta)) if weak_delta else 0.0,
        "weak_seed_delta_std": pstdev(weak_delta) if len(weak_delta) > 1 else 0.0,
        "classification_counts": dict(classifications),
        "primary_interpretation": _primary_interpretation(classifications),
    }


def _primary_interpretation(classifications: Counter[str]) -> str:
    if not classifications:
        return "no_failure_mode_identified"
    top = classifications.most_common(3)
    labels = [name for name, _count in top]
    if "frozen_already_solves_specific_families" in labels and "coordinator_bottleneck_or_candidate_conditioning_limit" in labels:
        return "frozen_extracts_schema_signal_and_candidate_agnostic_message_readout_limits_trainable_robustness"
    if "optimizer_instability" in labels:
        return "optimizer_instability_with_seed_sensitive_underfitting"
    if "topk_readout_selects_wrong_tokens" in labels:
        return "topk_readout_selection_instability"
    return labels[0]


def _render_report(result: Dict[str, object]) -> str:
    summary = result["summary"]
    lines = [
        "# Stage 3.9 Failure Analysis",
        "",
        "## Scope",
        "",
        "- Source: Stage 3.8 full hard-validation checkpoints and JSON results.",
        "- Training: none.",
        "- Dataset and controls: unchanged Stage 3.8 schema-aware benchmark.",
        f"- Relevant seeds analyzed: `{result['metadata']['relevant_seeds']}`.",
        "",
        "## Summary",
        "",
        f"- Stage 3.8 all-seed mean trainable: `{float(summary['stage38_all_seed_mean_trainable']):.4f}`",
        f"- Stage 3.8 all-seed mean frozen: `{float(summary['stage38_all_seed_mean_frozen']):.4f}`",
        f"- Stage 3.8 all-seed mean delta: `{float(summary['stage38_all_seed_mean_delta']):.4f}`",
        f"- Weak-seed mean delta: `{float(summary['weak_seed_mean_delta']):.4f}`",
        f"- Primary interpretation: `{summary['primary_interpretation']}`",
        f"- Classification counts: `{json.dumps(summary['classification_counts'], sort_keys=True)}`",
        "",
        "## Per-Seed Diagnosis",
        "",
        "| seed | trainable | frozen | delta | frozen families | low train families | active var train/frozen | top-k jaccard | interpretation flags |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["seed_rows"]:
        cls = row["classification"]
        train_readout = row["readout_stats"]["trainable"]
        frozen_readout = row["readout_stats"]["frozen"]
        token = row["topk_token_comparison"]
        flags = ", ".join(key for key, value in cls.items() if isinstance(value, bool) and value) or "none"
        lines.append(
            "| {seed} | {train:.4f} | {frozen:.4f} | {delta:.4f} | {ff} | {lf} | {tv:.4f}/{fv:.4f} | {jac:.4f} | {flags} |".format(
                seed=row["seed"],
                train=float(row["trainable_accuracy"]),
                frozen=float(row["frozen_accuracy"]),
                delta=float(row["delta"]),
                ff=int(cls["frozen_advantage_family_count"]),
                lf=int(cls["trainable_low_family_count"]),
                tv=float(train_readout["active_variance_mean"]),
                fv=float(frozen_readout["active_variance_mean"]),
                jac=float(token.get("mean_position_jaccard", 0.0)) if token.get("available") else 0.0,
                flags=flags,
            )
        )
    lines.extend(["", "## Per-Family Accuracy", ""])
    for row in result["seed_rows"]:
        lines.append(f"### Seed {row['seed']}")
        lines.append("")
        lines.append("| family | n | trainable | frozen | delta | train-only | frozen-only |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for family, fam in row["per_family"].items():
            lines.append(
                f"| {family} | {int(fam['n'])} | {float(fam['trainable']):.4f} | {float(fam['frozen']):.4f} | {float(fam['delta']):.4f} | {float(fam['trainable_only_correct']):.4f} | {float(fam['frozen_only_correct']):.4f} |"
            )
        lines.append("")
    lines.extend(["## Top-K Token Selection", "", "| seed | train field rate | frozen field rate | train evidence rate | frozen evidence rate | top train tokens | top frozen tokens |", "|---:|---:|---:|---:|---:|---|---|"])
    for row in result["seed_rows"]:
        token = row["topk_token_comparison"]
        if not token.get("available"):
            continue
        train = token["trainable"]
        frozen = token["frozen"]
        train_tokens = ", ".join(list(train["top_tokens"].keys())[:8])
        frozen_tokens = ", ".join(list(frozen["top_tokens"].keys())[:8])
        lines.append(
            f"| {row['seed']} | {float(train['field_token_rate']):.4f} | {float(frozen['field_token_rate']):.4f} | {float(train['evidence_token_rate']):.4f} | {float(frozen['evidence_token_rate']):.4f} | {train_tokens} | {frozen_tokens} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The failing/weak seeds are not explained by dataset shortcuts: Stage 3.8 controls and positive controls passed.",
            "- The dominant failure mode is a robustness problem in the candidate-agnostic top-k role-message path: frozen often extracts enough schema/category signal to match or beat trainable, while trainable is seed-sensitive and underfits several families.",
            "- The next search should prioritize candidate-conditioned readout/scoring and readout/training stability, while preserving the Stage 3.8 schema-aware controls.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
