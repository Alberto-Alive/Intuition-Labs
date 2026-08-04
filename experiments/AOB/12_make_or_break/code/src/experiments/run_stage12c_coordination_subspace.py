from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import run_stage12_make_or_break as stage12


FILE_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = FILE_PATH.parents[3]
RESULTS_PATH = EXPERIMENT_ROOT / "results" / "stage12c_coordination_subspace_results.json"
REPORT_PATH = EXPERIMENT_ROOT / "reports" / "STAGE12C_COORDINATION_SUBSPACE.md"
CACHE_DIR = EXPERIMENT_ROOT / "results" / "stage12c_coordination_cache"
FEATURES_PATH = CACHE_DIR / "feature_sites.npz"
METADATA_PATH = CACHE_DIR / "metadata.json"

POSITIVE_SEEDS = (31, 37)
SCALE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
TOKEN_SITE = "token_summary_best_avenue_flat"
INTERVENTION_MIN_EXAMPLES = 16


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 12c within-example coordination subspace search.")
    parser.add_argument("--preset", default="stage5_final", choices=("stage5_final",))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    result = run_analysis(
        preset_name=str(args.preset),
        device=str(args.device),
        batch_size=int(args.batch_size),
        force_rebuild_cache=bool(args.force_rebuild_cache),
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_report(result), encoding="utf-8")
    print(f"stage12c: wrote {RESULTS_PATH}, {REPORT_PATH}, {FEATURES_PATH}, and {METADATA_PATH}")


def run_analysis(
    preset_name: str,
    device: str,
    batch_size: int,
    force_rebuild_cache: bool,
) -> Dict[str, Any]:
    preset = stage12._artifact_presets()[preset_name]
    modules = stage12._bootstrap_modules(preset)
    payload = json.loads(preset.results_path.read_text(encoding="utf-8"))
    rows = stage12._completed_rows(payload, preset)
    if not rows:
        raise RuntimeError(f"no completed rows found for preset={preset_name}")
    cache = _load_or_build_cache(
        modules=modules,
        preset=preset,
        rows=rows,
        device=device,
        batch_size=batch_size,
        force_rebuild_cache=force_rebuild_cache,
    )
    metadata_rows = cache["metadata_rows"]
    features_by_site = cache["features_by_site"]
    residualized_sites = {
        site: _leave_one_out_residualize(features, metadata_rows)
        for site, features in features_by_site.items()
    }
    site_scan = _scan_gain_sites(residualized_sites, metadata_rows)
    causal = _run_causal_tests(
        modules=modules,
        preset=preset,
        rows=rows,
        residualized_sites=residualized_sites,
        metadata_rows=metadata_rows,
        device=device,
        batch_size=batch_size,
    )
    verdict = _overall_verdict(site_scan, causal)
    return {
        "metadata": {
            "stage": "stage12c_coordination_subspace",
            "created_at_utc": stage12._now(),
            "source_preset": preset_name,
            "device": device,
            "batch_size": int(batch_size),
            "cache_reused": bool(cache["cache_reused"]),
            "seed_count": len(sorted({int(row["seed"]) for row in metadata_rows})),
            "example_seed_rows": len(metadata_rows),
            "unique_examples": len(sorted({str(row["example_id"]) for row in metadata_rows})),
            "positive_seeds_requested": list(POSITIVE_SEEDS),
        },
        "cache": {
            "cache_dir": str(CACHE_DIR),
            "features_path": str(FEATURES_PATH),
            "metadata_path": str(METADATA_PATH),
            "site_shapes": {site: list(value.shape) for site, value in features_by_site.items()},
        },
        "gain_direction_site_scan": site_scan,
        "causal_tests": causal,
        "overall_verdict": verdict,
    }


def _load_or_build_cache(
    modules: Any,
    preset: stage12.ArtifactPreset,
    rows: Sequence[Mapping[str, Any]],
    device: str,
    batch_size: int,
    force_rebuild_cache: bool,
) -> Dict[str, Any]:
    if not force_rebuild_cache and FEATURES_PATH.exists() and METADATA_PATH.exists():
        features = np.load(FEATURES_PATH)
        metadata_payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        metadata_rows = metadata_payload.get("rows", [])
        feature_names = metadata_payload.get("feature_names", [])
        if _cache_is_valid(features, metadata_rows, feature_names):
            return {
                "features_by_site": {name: features[name].astype(np.float32, copy=False) for name in feature_names},
                "metadata_rows": metadata_rows,
                "cache_reused": True,
            }
    features_by_site, metadata_rows = _build_cache(
        modules=modules,
        preset=preset,
        rows=rows,
        device=device,
        batch_size=batch_size,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURES_PATH, **features_by_site)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "feature_names": list(features_by_site.keys()),
                "rows": metadata_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "features_by_site": features_by_site,
        "metadata_rows": metadata_rows,
        "cache_reused": False,
    }


def _cache_is_valid(features: Any, metadata_rows: Any, feature_names: Sequence[str]) -> bool:
    if not isinstance(metadata_rows, list) or not feature_names:
        return False
    if any(name not in features for name in feature_names):
        return False
    row_count = None
    for name in feature_names:
        value = features[name]
        if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
            return False
        if row_count is None:
            row_count = int(value.shape[0])
        elif row_count != int(value.shape[0]):
            return False
    if row_count is None or row_count != len(metadata_rows):
        return False
    required = {
        "example_id",
        "seed",
        "family",
        "label",
        "role_count",
        "token_hidden_dim",
        "full_acc",
        "best_single_acc",
        "best_single_avenue",
        "gain_positive",
        "full_margin",
        "best_single_margin",
    }
    return all(required.issubset(set(row.keys())) for row in metadata_rows)


def _build_cache(
    modules: Any,
    preset: stage12.ArtifactPreset,
    rows: Sequence[Mapping[str, Any]],
    device: str,
    batch_size: int,
) -> tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    feature_lists: Dict[str, List[np.ndarray]] = defaultdict(list)
    metadata_rows: List[Dict[str, Any]] = []
    for row in rows:
        checkpoint_path = stage12._resolve_checkpoint_path(preset.experiment_root, row)
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"missing checkpoint for seed={row.get('seed')} at {checkpoint_path}")
        result = stage12._load_checkpoint(modules, checkpoint_path, device=device)
        system = result.system
        dataset_config = stage12._make_dataset_config(modules.dataset, row)
        seed = int(row.get("seed", 0))
        splits = modules.dataset.build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=stage12.REPO_ROOT)
        test_examples = list(splits["test"])
        labels = stage12._labels(test_examples)
        full_logits = stage12._logits_for_condition(system, test_examples, condition="none", seed=seed, batch_size=batch_size)
        full_preds = np.argmax(full_logits, axis=1).astype(np.int64)
        full_margin = _label_margin(full_logits, labels)
        avenue_count = int(system.message_config.num_avenues)
        single_logits = np.stack(
            [
                stage12._logits_for_condition(
                    system,
                    test_examples,
                    condition=f"avenue_only_{avenue_index}",
                    seed=seed,
                    batch_size=batch_size,
                )
                for avenue_index in range(avenue_count)
            ],
            axis=1,
        )
        single_preds = np.argmax(single_logits, axis=2).astype(np.int64)
        single_correct = single_preds == labels[:, None]
        single_margins = np.stack(
            [_label_margin(single_logits[:, avenue_index, :], labels) for avenue_index in range(avenue_count)],
            axis=1,
        )
        best_single_avenue = np.argmax(single_margins, axis=1).astype(np.int64)
        best_single_margin = single_margins[np.arange(len(test_examples)), best_single_avenue].astype(np.float32, copy=False)
        best_single_acc = single_correct.any(axis=1).astype(np.int64)
        full_acc = (full_preds == labels).astype(np.int64)
        gain_positive = ((full_acc == 1) & (best_single_acc == 0)).astype(np.int64)
        families = stage12._families(modules.dataset, test_examples)

        offset = 0
        for batch in stage12._example_batches(test_examples, batch_size):
            readouts = system.collect_clone_representations(batch, condition="none", seed=seed)
            batch_size_now = len(batch)
            slice_obj = slice(offset, offset + batch_size_now)
            batch_best = best_single_avenue[slice_obj]
            batch_sites = _extract_feature_sites(readouts, batch_best)
            for site_name, site_values in batch_sites.items():
                feature_lists[site_name].append(site_values.astype(np.float32, copy=False))
            for batch_index, example in enumerate(batch):
                global_index = offset + batch_index
                metadata_rows.append(
                    {
                        "row_index": len(metadata_rows),
                        "seed": seed,
                        "example_id": str(example.id),
                        "family": str(families[global_index]),
                        "label": int(labels[global_index]),
                        "full_acc": int(full_acc[global_index]),
                        "best_single_acc": int(best_single_acc[global_index]),
                        "gain_positive": bool(gain_positive[global_index]),
                        "best_single_avenue": int(best_single_avenue[global_index]),
                        "role_count": int(system.n_roles),
                        "token_hidden_dim": int(batch_sites["token_summary_best_avenue_flat"].shape[1] // max(1, int(system.n_roles))),
                        "full_margin": float(full_margin[global_index]),
                        "best_single_margin": float(best_single_margin[global_index]),
                        "margin_gap": float(full_margin[global_index] - best_single_margin[global_index]),
                        "full_pred": int(full_preds[global_index]),
                    }
                )
            offset += batch_size_now
        del result
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    features_by_site = {
        site_name: np.concatenate(site_values, axis=0).astype(np.float32, copy=False)
        for site_name, site_values in sorted(feature_lists.items())
    }
    return features_by_site, metadata_rows


def _extract_feature_sites(readouts: Mapping[str, torch.Tensor], best_single_avenue: np.ndarray) -> Dict[str, np.ndarray]:
    message = readouts["message"].detach().cpu()
    raw_pooled = readouts["raw_pooled"].detach().cpu()
    raw_pooled_avenues = readouts["raw_pooled_avenues"].detach().cpu()
    msg_avenues = readouts["msg_avenues"].detach().cpu()
    token_summary = _summarize_token_states(readouts["token_states"], readouts["token_mask"]).detach().cpu()
    best_index = torch.as_tensor(best_single_avenue, dtype=torch.long).view(-1, 1, 1, 1)
    gather_index = best_index.expand(-1, raw_pooled_avenues.shape[1], 1, raw_pooled_avenues.shape[3])
    best_raw = raw_pooled_avenues.gather(2, gather_index).squeeze(2)
    best_msg = msg_avenues.gather(2, gather_index).squeeze(2)
    best_token = token_summary.gather(2, gather_index).squeeze(2)
    return {
        "message_roles_flat": message.reshape(message.shape[0], -1).numpy(),
        "raw_pooled_roles_flat": raw_pooled.reshape(raw_pooled.shape[0], -1).numpy(),
        "raw_pooled_avenues_mean_roles_flat": raw_pooled_avenues.mean(dim=1).reshape(raw_pooled_avenues.shape[0], -1).numpy(),
        "raw_pooled_best_avenue_flat": best_raw.reshape(best_raw.shape[0], -1).numpy(),
        "msg_best_avenue_flat": best_msg.reshape(best_msg.shape[0], -1).numpy(),
        "token_summary_best_avenue_flat": best_token.reshape(best_token.shape[0], -1).numpy(),
        "token_summary_avenues_mean_roles_flat": token_summary.mean(dim=1).reshape(token_summary.shape[0], -1).numpy(),
    }


def _summarize_token_states(token_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    if token_states.dim() != 5:
        raise ValueError(f"expected [batch, roles, avenues, tokens, hidden], got {tuple(token_states.shape)}")
    mask = token_mask.bool().unsqueeze(-1)
    denom = mask.sum(dim=3).clamp(min=1)
    return (token_states * mask.to(dtype=token_states.dtype)).sum(dim=3) / denom.to(dtype=token_states.dtype)


def _label_margin(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    correct = logits[np.arange(len(labels)), labels]
    masked = logits.copy()
    masked[np.arange(len(labels)), labels] = -1e9
    best_other = masked.max(axis=1)
    return (correct - best_other).astype(np.float32, copy=False)


def _leave_one_out_residualize(features: np.ndarray, metadata_rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    by_example: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(metadata_rows):
        by_example[str(row["example_id"])].append(index)
    result = np.zeros_like(features, dtype=np.float32)
    for indices in by_example.values():
        group = np.asarray(indices, dtype=np.int64)
        group_features = features[group]
        positive_mask = np.asarray([bool(metadata_rows[index]["gain_positive"]) for index in group.tolist()], dtype=bool)
        negative_mask = ~positive_mask
        group_sum = group_features.sum(axis=0)
        group_count = group_features.shape[0]
        negative_sum = group_features[negative_mask].sum(axis=0) if np.any(negative_mask) else np.zeros(features.shape[1], dtype=np.float32)
        negative_count = int(np.sum(negative_mask))
        for local_index, global_index in enumerate(group.tolist()):
            current = group_features[local_index]
            if negative_mask[local_index] and negative_count > 1:
                baseline = (negative_sum - current) / float(negative_count - 1)
            elif not negative_mask[local_index] and negative_count > 0:
                baseline = negative_sum / float(negative_count)
            elif group_count > 1:
                baseline = (group_sum - current) / float(group_count - 1)
            else:
                baseline = np.zeros(features.shape[1], dtype=np.float32)
            result[global_index] = current - baseline
    return result.astype(np.float32, copy=False)


def _scan_gain_sites(
    residualized_sites: Mapping[str, np.ndarray],
    metadata_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    available_positive_seeds = _available_positive_seeds(metadata_rows)
    metadata = _metadata_arrays(metadata_rows)
    difficulty_scores = -metadata["best_single_margin"]
    site_rows = []
    for site_name, features in sorted(residualized_sites.items()):
        pair_rows = []
        for train_seed in available_positive_seeds:
            for test_seed in available_positive_seeds:
                if train_seed == test_seed:
                    continue
                train_mask = metadata["seed"] == int(train_seed)
                test_mask = metadata["seed"] == int(test_seed)
                train_y = metadata["gain_positive"][train_mask]
                test_y = metadata["gain_positive"][test_mask]
                if np.unique(train_y).size < 2 or np.unique(test_y).size < 2:
                    continue
                direction = _mean_difference_direction(features[train_mask], train_y)
                train_scores = features[train_mask] @ direction["unit_direction"]
                threshold = _score_threshold(train_scores, train_y)
                test_scores = features[test_mask] @ direction["unit_direction"]
                pair_rows.append(
                    {
                        "train_seed": int(train_seed),
                        "test_seed": int(test_seed),
                        "train": _score_metrics(train_scores, train_y, threshold),
                        "test": _score_metrics(test_scores, test_y, threshold),
                        "difficulty_test": _score_metrics(difficulty_scores[test_mask], test_y, _score_threshold(difficulty_scores[train_mask], train_y)),
                        "direction_norm": float(direction["direction_norm"]),
                    }
                )
        if not pair_rows:
            continue
        reciprocal = {
            "mean_test_balanced_accuracy": stage12._mean([float(row["test"]["balanced_accuracy"]) for row in pair_rows]),
            "min_test_balanced_accuracy": min(float(row["test"]["balanced_accuracy"]) for row in pair_rows),
            "mean_test_roc_auc": stage12._mean([float(row["test"]["roc_auc"]) for row in pair_rows]),
            "difficulty_mean_test_balanced_accuracy": stage12._mean(
                [float(row["difficulty_test"]["balanced_accuracy"]) for row in pair_rows]
            ),
        }
        site_rows.append(
            {
                "site": site_name,
                "pair_rows": pair_rows,
                "reciprocal_summary": reciprocal,
                "intervention_capable": site_name == TOKEN_SITE,
            }
        )
    site_rows.sort(
        key=lambda row: (
            -float(row["reciprocal_summary"]["min_test_balanced_accuracy"]),
            -float(row["reciprocal_summary"]["mean_test_roc_auc"]),
            row["site"],
        )
    )
    best_site = site_rows[0]["site"] if site_rows else None
    token_site = next((row for row in site_rows if row["site"] == TOKEN_SITE), None)
    counts_by_seed = {
        str(seed): int(np.sum((_metadata_arrays(metadata_rows)["seed"] == seed) & _metadata_arrays(metadata_rows)["gain_positive"]))
        for seed in sorted(set(_metadata_arrays(metadata_rows)["seed"].tolist()))
    }
    return {
        "available_positive_seeds": available_positive_seeds,
        "gain_positive_counts_by_seed": counts_by_seed,
        "best_transfer_site": best_site,
        "best_intervention_site": token_site["site"] if token_site else None,
        "site_rankings": site_rows,
    }


def _metadata_arrays(metadata_rows: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    return {
        "seed": np.asarray([int(row["seed"]) for row in metadata_rows], dtype=np.int64),
        "gain_positive": np.asarray([1 if bool(row["gain_positive"]) else 0 for row in metadata_rows], dtype=np.int64),
        "full_acc": np.asarray([int(row["full_acc"]) for row in metadata_rows], dtype=np.int64),
        "best_single_acc": np.asarray([int(row["best_single_acc"]) for row in metadata_rows], dtype=np.int64),
        "best_single_margin": np.asarray([float(row["best_single_margin"]) for row in metadata_rows], dtype=np.float32),
    }


def _available_positive_seeds(metadata_rows: Sequence[Mapping[str, Any]]) -> List[int]:
    counts = Counter(int(row["seed"]) for row in metadata_rows if bool(row["gain_positive"]))
    return [int(seed) for seed, count in sorted(counts.items()) if int(count) > 0 and int(seed) in POSITIVE_SEEDS]


def _mean_difference_direction(train_x: np.ndarray, train_y: np.ndarray) -> Dict[str, Any]:
    pos = train_x[train_y.astype(bool)]
    neg = train_x[~train_y.astype(bool)]
    direction = pos.mean(axis=0) - neg.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        direction = np.zeros(train_x.shape[1], dtype=np.float32)
        unit = direction
    else:
        unit = (direction / norm).astype(np.float32, copy=False)
    return {
        "direction": direction.astype(np.float32, copy=False),
        "unit_direction": unit,
        "direction_norm": norm,
    }


def _score_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    pos_scores = scores[labels.astype(bool)]
    neg_scores = scores[~labels.astype(bool)]
    return float((pos_scores.mean() + neg_scores.mean()) / 2.0)


def _score_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (scores >= float(threshold)).astype(np.int64)
    roc_auc = 0.5
    if np.unique(labels).size >= 2:
        roc_auc = float(roc_auc_score(labels, scores))
    return {
        "accuracy": float(np.mean(preds == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "roc_auc": roc_auc,
        "positive_rate": float(np.mean(labels)),
    }


def _run_causal_tests(
    modules: Any,
    preset: stage12.ArtifactPreset,
    rows: Sequence[Mapping[str, Any]],
    residualized_sites: Mapping[str, np.ndarray],
    metadata_rows: Sequence[Mapping[str, Any]],
    device: str,
    batch_size: int,
) -> Dict[str, Any]:
    metadata_by_seed_and_example = {
        (int(row["seed"]), str(row["example_id"])): row for row in metadata_rows
    }
    metadata_by_seed: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in metadata_rows:
        metadata_by_seed[int(row["seed"])].append(row)
    available_positive_seeds = _available_positive_seeds(metadata_rows)
    row_by_seed = {int(row["seed"]): row for row in rows}
    token_features = residualized_sites[TOKEN_SITE]
    directions_by_seed = _token_directions_by_seed(token_features, metadata_rows, available_positive_seeds)
    source_rows = []
    for source_seed in available_positive_seeds:
        if source_seed not in directions_by_seed or source_seed not in row_by_seed:
            continue
        source_direction = directions_by_seed[source_seed]
        source_entry = stage12._collect_seed_entry(
            modules=modules,
            preset=preset,
            row=row_by_seed[source_seed],
            device=device,
            batch_size=batch_size,
        )
        source_meta_lookup = {str(row["example_id"]): row for row in metadata_by_seed[source_seed]}
        source_positive_ids = sorted(
            str(row["example_id"]) for row in metadata_by_seed[source_seed] if bool(row["gain_positive"])
        )
        ablation_examples = [example for example in source_entry["test_examples"] if str(example.id) in source_positive_ids]
        ablation_meta = [source_meta_lookup[str(example.id)] for example in ablation_examples]
        ablation = _evaluate_token_direction_on_seed(
            modules=modules,
            system_seed=source_seed,
            system_entry=source_entry,
            examples=ablation_examples,
            metadata_rows=ablation_meta,
            base_direction=source_direction["direction_matrix"],
            device=device,
            batch_size=batch_size,
            mode="ablate",
            mask_mode="full",
        )
        substitution = _evaluate_token_direction_on_seed(
            modules=modules,
            system_seed=source_seed,
            system_entry=source_entry,
            examples=ablation_examples,
            metadata_rows=ablation_meta,
            base_direction=source_direction["direction_matrix"],
            device=device,
            batch_size=batch_size,
            mode="inject",
            mask_mode="best_single_only",
        )
        target_rows = []
        for target_seed in sorted(set(int(row["seed"]) for row in metadata_rows)):
            if target_seed == source_seed or target_seed not in row_by_seed:
                continue
            target_entry = stage12._collect_seed_entry(
                modules=modules,
                preset=preset,
                row=row_by_seed[target_seed],
                device=device,
                batch_size=batch_size,
            )
            target_meta_lookup = {str(row["example_id"]): row for row in metadata_by_seed[target_seed]}
            matched_negative_ids = sorted(
                example_id
                for example_id in source_positive_ids
                if example_id in target_meta_lookup
                and int(target_meta_lookup[example_id]["full_acc"]) == 0
                and int(target_meta_lookup[example_id]["best_single_acc"]) == 0
            )
            if len(matched_negative_ids) < INTERVENTION_MIN_EXAMPLES:
                continue
            target_examples = [example for example in target_entry["test_examples"] if str(example.id) in matched_negative_ids]
            target_meta = [target_meta_lookup[str(example.id)] for example in target_examples]
            injection = _evaluate_token_direction_on_seed(
                modules=modules,
                system_seed=target_seed,
                system_entry=target_entry,
                examples=target_examples,
                metadata_rows=target_meta,
                base_direction=source_direction["direction_matrix"],
                device=device,
                batch_size=batch_size,
                mode="inject",
                mask_mode="best_single_only",
            )
            target_rows.append(
                {
                    "target_seed": int(target_seed),
                    "matched_negative_examples": len(target_examples),
                    "result": injection,
                }
            )
        source_rows.append(
            {
                "source_seed": int(source_seed),
                "gain_positive_examples": len(source_positive_ids),
                "direction_norm": float(source_direction["direction_norm"]),
                "ablation_on_source_positives": ablation,
                "single_avenue_substitution_on_source_positives": substitution,
                "injection_on_matched_negative_targets": target_rows,
            }
        )
    return {
        "token_site": TOKEN_SITE,
        "source_rows": source_rows,
        "aggregate_summary": _summarize_causal_rows(source_rows),
    }


def _token_directions_by_seed(
    token_features: np.ndarray,
    metadata_rows: Sequence[Mapping[str, Any]],
    available_positive_seeds: Sequence[int],
) -> Dict[int, Dict[str, Any]]:
    dims = token_features.shape[1]
    role_count = int(metadata_rows[0]["role_count"]) if metadata_rows else 1
    hidden = int(metadata_rows[0]["token_hidden_dim"]) if metadata_rows else dims
    if role_count * hidden != dims:
        raise ValueError(
            f"token site dimension {dims} does not match cached role_count={role_count} and token_hidden_dim={hidden}"
        )
    output: Dict[int, Dict[str, Any]] = {}
    for seed in available_positive_seeds:
        mask = np.asarray([int(row["seed"]) == int(seed) for row in metadata_rows], dtype=bool)
        labels = np.asarray([1 if bool(row["gain_positive"]) else 0 for row in metadata_rows], dtype=np.int64)[mask]
        if np.unique(labels).size < 2:
            continue
        direction = _mean_difference_direction(token_features[mask], labels)
        output[int(seed)] = {
            "direction": direction["direction"].astype(np.float32, copy=False),
            "direction_matrix": direction["direction"].reshape(role_count, hidden).astype(np.float32, copy=False),
            "unit_direction": direction["unit_direction"].astype(np.float32, copy=False),
            "direction_norm": float(direction["direction_norm"]),
        }
    return output


def _evaluate_token_direction_on_seed(
    modules: Any,
    system_seed: int,
    system_entry: Mapping[str, Any],
    examples: Sequence[Any],
    metadata_rows: Sequence[Mapping[str, Any]],
    base_direction: np.ndarray,
    device: str,
    batch_size: int,
    mode: str,
    mask_mode: str,
) -> Dict[str, Any]:
    if not examples:
        return {"available": False, "reason": "no examples available"}
    checkpoint_path = Path(str(system_entry["checkpoint_path"]))
    result = stage12._load_checkpoint(modules, checkpoint_path, device=device)
    system = result.system
    if not bool(getattr(system.coordinator, "requires_token_states", False)):
        return {"available": False, "reason": "coordinator does not expose token-state intervention path"}
    meta_lookup = {str(row["example_id"]): row for row in metadata_rows}
    ordered_examples = [example for example in system_entry["test_examples"] if str(example.id) in meta_lookup]
    ordered_meta = [meta_lookup[str(example.id)] for example in ordered_examples]
    labels = stage12._labels(ordered_examples)
    direction = torch.as_tensor(base_direction, dtype=torch.float32, device=next(system.parameters()).device)
    shuffled = _shuffled_direction(direction)
    orthogonal = _orthogonal_direction(direction, seed=system_seed + 7_001)
    conditions = {
        "direction": direction,
        "shuffled": shuffled,
        "orthogonal": orthogonal,
    }
    baseline_correct: List[np.ndarray] = []
    results_by_scale: Dict[str, Dict[str, float]] = {}
    system.eval()
    with torch.no_grad():
        for scale in SCALE_GRID:
            results_by_scale[f"{scale:.2f}"] = {}
        for batch_start in range(0, len(ordered_examples), batch_size):
            batch = ordered_examples[batch_start : batch_start + batch_size]
            batch_labels = labels[batch_start : batch_start + batch_size]
            batch_meta = ordered_meta[batch_start : batch_start + batch_size]
            batch_best = np.asarray([int(row["best_single_avenue"]) for row in batch_meta], dtype=np.int64)
            device_obj = next(system.parameters()).device
            role_ids = stage12._role_ids_for_batch(system, len(batch), device_obj)
            readouts = system.collect_clone_representations(batch, condition="none", seed=system_seed, role_ids=role_ids)
            token_states = readouts["token_states"]
            token_mask = readouts["token_mask"].bool()
            eval_mask = _evaluation_mask(token_mask, batch_best, mode=mask_mode)
            baseline_logits = stage12._candidate_token_logits_from_readouts(
                modules.latent,
                system,
                batch,
                readouts,
                role_ids,
                token_states,
                eval_mask,
            )
            baseline_preds = torch.argmax(baseline_logits, dim=1).detach().cpu().numpy().astype(np.int64)
            baseline_correct.append((baseline_preds == batch_labels).astype(np.int64))
            best_index = torch.as_tensor(batch_best, dtype=torch.long, device=device_obj)
            base_repeated = direction.unsqueeze(0).expand(len(batch), -1, -1)
            for scale in SCALE_GRID:
                key = f"{scale:.2f}"
                noise = stage12._noise_matched_to_signal(base_repeated, seed=system_seed + batch_start + int(scale * 1000))
                for name, vector in conditions.items():
                    repeated = vector.unsqueeze(0).expand(len(batch), -1, -1)
                    if name == "orthogonal":
                        repeated = repeated * max(1.0, float(direction.norm().detach().cpu().item()))
                    signed = repeated * float(scale if mode == "inject" else -scale)
                    if name == "direction":
                        active = signed
                    elif name == "shuffled":
                        active = signed
                    elif name == "orthogonal":
                        active = signed
                    else:
                        raise AssertionError(name)
                    modified = _inject_by_example_avenue(token_states, best_index, active)
                    logits = stage12._candidate_token_logits_from_readouts(
                        modules.latent,
                        system,
                        batch,
                        readouts,
                        role_ids,
                        modified,
                        eval_mask,
                    )
                    preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
                    results_by_scale[key].setdefault(name, []).append((preds == batch_labels).astype(np.int64))
                noise_modified = _inject_by_example_avenue(token_states, best_index, noise * float(scale if mode == "inject" else -scale))
                noise_logits = stage12._candidate_token_logits_from_readouts(
                    modules.latent,
                    system,
                    batch,
                    readouts,
                    role_ids,
                    noise_modified,
                    eval_mask,
                )
                noise_preds = torch.argmax(noise_logits, dim=1).detach().cpu().numpy().astype(np.int64)
                results_by_scale[key].setdefault("noise", []).append((noise_preds == batch_labels).astype(np.int64))
    baseline_accuracy = float(np.mean(np.concatenate(baseline_correct, axis=0)))
    summarized = {}
    for key, values in results_by_scale.items():
        summarized[key] = {name: float(np.mean(np.concatenate(chunks, axis=0))) for name, chunks in values.items()}
    return {
        "available": True,
        "mode": mode,
        "mask_mode": mask_mode,
        "example_count": len(ordered_examples),
        "baseline_accuracy": baseline_accuracy,
        "accuracy_by_scale_and_condition": summarized,
    }


def _inject_by_example_avenue(
    token_states: torch.Tensor,
    avenue_indices: torch.Tensor,
    injection: torch.Tensor,
) -> torch.Tensor:
    if token_states.dim() != 5:
        raise ValueError(f"expected [batch, roles, avenues, tokens, hidden], got {tuple(token_states.shape)}")
    out = token_states.clone()
    for avenue in torch.unique(avenue_indices).tolist():
        mask = avenue_indices == int(avenue)
        if not bool(mask.any()):
            continue
        out[mask, :, int(avenue), :, :] = out[mask, :, int(avenue), :, :] + injection[mask].unsqueeze(2)
    return out


def _shuffled_direction(direction: torch.Tensor) -> torch.Tensor:
    flat = direction.reshape(-1)
    order = torch.arange(flat.shape[0], device=flat.device)
    order = torch.roll(order, shifts=11)
    return flat[order].reshape_as(direction)


def _orthogonal_direction(direction: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=direction.device)
    generator.manual_seed(int(seed))
    noise = torch.randn(direction.shape, generator=generator, device=direction.device, dtype=direction.dtype)
    flat_direction = direction.reshape(-1)
    flat_noise = noise.reshape(-1)
    projection = torch.dot(flat_noise, flat_direction) / flat_direction.norm().pow(2).clamp_min(1e-12)
    orthogonal = flat_noise - projection * flat_direction
    norm = orthogonal.norm().clamp_min(1e-12)
    return (orthogonal / norm).reshape_as(direction)


def _evaluation_mask(token_mask: torch.Tensor, best_single_avenues: np.ndarray, mode: str) -> torch.Tensor:
    if mode == "full":
        return token_mask
    if mode != "best_single_only":
        raise ValueError(f"unknown mask mode: {mode}")
    out = torch.zeros_like(token_mask, dtype=torch.bool)
    best = torch.as_tensor(best_single_avenues, dtype=torch.long, device=token_mask.device)
    for avenue in torch.unique(best).tolist():
        mask = best == int(avenue)
        if not bool(mask.any()):
            continue
        out[mask, :, int(avenue), :] = token_mask[mask, :, int(avenue), :]
    return out


def _summarize_causal_rows(source_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    injection_effects: List[tuple[str, float, float, float, float]] = []
    substitution_effects: List[tuple[str, float, float, float, float]] = []
    ablation_effects: List[tuple[str, float, float, float, float]] = []
    for row in source_rows:
        ablation = row.get("ablation_on_source_positives", {})
        if ablation.get("available"):
            baseline = float(ablation["baseline_accuracy"])
            for scale, values in ablation["accuracy_by_scale_and_condition"].items():
                ablation_effects.append(
                    (
                        f"seed{int(row['source_seed'])}_scale{scale}",
                        baseline - float(values["direction"]),
                        baseline - float(values["shuffled"]),
                        baseline - float(values["orthogonal"]),
                        baseline - float(values["noise"]),
                    )
                )
        substitution = row.get("single_avenue_substitution_on_source_positives", {})
        if substitution.get("available"):
            baseline = float(substitution["baseline_accuracy"])
            for scale, values in substitution["accuracy_by_scale_and_condition"].items():
                substitution_effects.append(
                    (
                        f"seed{int(row['source_seed'])}_scale{scale}",
                        float(values["direction"]) - baseline,
                        float(values["shuffled"]) - baseline,
                        float(values["orthogonal"]) - baseline,
                        float(values["noise"]) - baseline,
                    )
                )
        for target_row in row.get("injection_on_matched_negative_targets", []):
            result = target_row.get("result", {})
            if not result.get("available"):
                continue
            baseline = float(result["baseline_accuracy"])
            for scale, values in result["accuracy_by_scale_and_condition"].items():
                injection_effects.append(
                    (
                        f"{int(row['source_seed'])}->{int(target_row['target_seed'])}_scale{scale}",
                        float(values["direction"]) - baseline,
                        float(values["shuffled"]) - baseline,
                        float(values["orthogonal"]) - baseline,
                        float(values["noise"]) - baseline,
                    )
                )
    best_injection = max(injection_effects, key=lambda item: item[1], default=None)
    best_substitution = max(substitution_effects, key=lambda item: item[1], default=None)
    best_ablation = max(ablation_effects, key=lambda item: item[1], default=None)
    return {
        "best_direction_substitution_gain": (
            {
                "label": best_substitution[0],
                "direction_gain": best_substitution[1],
                "shuffled_gain": best_substitution[2],
                "orthogonal_gain": best_substitution[3],
                "noise_gain": best_substitution[4],
            }
            if best_substitution is not None
            else None
        ),
        "best_direction_injection_gain": (
            {
                "label": best_injection[0],
                "direction_gain": best_injection[1],
                "shuffled_gain": best_injection[2],
                "orthogonal_gain": best_injection[3],
                "noise_gain": best_injection[4],
            }
            if best_injection is not None
            else None
        ),
        "best_direction_ablation_drop": (
            {
                "label": best_ablation[0],
                "direction_drop": best_ablation[1],
                "shuffled_drop": best_ablation[2],
                "orthogonal_drop": best_ablation[3],
                "noise_drop": best_ablation[4],
            }
            if best_ablation is not None
            else None
        ),
    }


def _overall_verdict(site_scan: Mapping[str, Any], causal: Mapping[str, Any]) -> Dict[str, Any]:
    token_row = next(
        (row for row in site_scan.get("site_rankings", []) if row.get("site") == TOKEN_SITE),
        None,
    )
    reciprocal = token_row.get("reciprocal_summary", {}) if token_row else {}
    transfer_mean = float(reciprocal.get("mean_test_balanced_accuracy", 0.0))
    transfer_vs_difficulty = transfer_mean - float(reciprocal.get("difficulty_mean_test_balanced_accuracy", 0.0))
    best_substitution = causal.get("aggregate_summary", {}).get("best_direction_substitution_gain")
    best_injection = causal.get("aggregate_summary", {}).get("best_direction_injection_gain")
    best_ablation = causal.get("aggregate_summary", {}).get("best_direction_ablation_drop")
    causal_hit = bool(
        best_substitution
        and float(best_substitution["direction_gain"]) > max(
            float(best_substitution["shuffled_gain"]),
            float(best_substitution["orthogonal_gain"]),
            float(best_substitution["noise_gain"]),
        )
        and best_injection
        and best_ablation
        and float(best_injection["direction_gain"]) > max(
            float(best_injection["shuffled_gain"]),
            float(best_injection["orthogonal_gain"]),
            float(best_injection["noise_gain"]),
        )
        and float(best_ablation["direction_drop"]) > max(
            float(best_ablation["shuffled_drop"]),
            float(best_ablation["orthogonal_drop"]),
            float(best_ablation["noise_drop"]),
        )
    )
    if causal_hit and transfer_mean >= 0.60 and transfer_vs_difficulty > 0.05:
        verdict = "YES"
        meaning = "A transferable within-example coordination direction exists and has causal support at the token-state intervention site."
        next_step = "Refine the direction with stronger regularization and redesign the Stage 12 make-or-break intervention around that token-state subspace."
    elif best_substitution and float(best_substitution["direction_gain"]) > max(
        float(best_substitution["shuffled_gain"]),
        float(best_substitution["orthogonal_gain"]),
        float(best_substitution["noise_gain"]),
    ) and best_ablation and float(best_ablation["direction_drop"]) > max(
        float(best_ablation["shuffled_drop"]),
        float(best_ablation["orthogonal_drop"]),
        float(best_ablation["noise_drop"]),
    ):
        verdict = "PARTIAL"
        meaning = "The token-site direction can both rescue the source seed's best single avenue and break the source positive full-model cases, but it does not transfer cleanly across seeds."
        next_step = "Shift the search to layer-specific token-state directions and require 31↔37 transfer before trusting the signal as a reusable coordination subspace."
    elif transfer_mean >= 0.55 and transfer_vs_difficulty > 0.0:
        verdict = "PARTIAL"
        meaning = "The within-example token site carries transferable gain signal, but the causal intervention did not cleanly separate from the controls."
        next_step = "Move one level deeper and search layer-specific token-state directions instead of a single pooled avenue-token summary."
    else:
        verdict = "NO"
        meaning = "The aggressive within-example direction search did not produce a clean causal coordination direction at the current summary site."
        next_step = "Go lower-level next: layerwise token-state or attention-pattern search rather than more pooled-vector analysis."
    return {
        "stage12c_supported": verdict,
        "mean_token_site_transfer_balanced_accuracy": transfer_mean,
        "token_site_minus_difficulty_balanced_accuracy": transfer_vs_difficulty,
        "causal_hit": causal_hit,
        "meaning": meaning,
        "recommended_next_step": next_step,
    }


def _render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 12c Coordination Subspace",
        "",
        "## Source",
        "",
        f"- Preset: `{result['metadata']['source_preset']}`",
        f"- Device: `{result['metadata']['device']}`",
        f"- Cache reused: `{'YES' if result['metadata']['cache_reused'] else 'NO'}`",
        f"- Example-seed rows: `{int(result['metadata']['example_seed_rows'])}`",
        f"- Unique examples: `{int(result['metadata']['unique_examples'])}`",
        "",
        "## Gain Direction Scan",
        "",
        f"- Available positive seeds: `{result['gain_direction_site_scan']['available_positive_seeds']}`",
        f"- Gain-positive counts by seed: `{result['gain_direction_site_scan']['gain_positive_counts_by_seed']}`",
        f"- Best transfer site: `{result['gain_direction_site_scan']['best_transfer_site']}`",
        f"- Best intervention-capable site: `{result['gain_direction_site_scan']['best_intervention_site']}`",
        "",
        "| site | mean test bacc | min test bacc | mean test AUC | difficulty mean test bacc | intervention |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in result["gain_direction_site_scan"]["site_rankings"]:
        summary = row["reciprocal_summary"]
        lines.append(
            f"| {row['site']} | {float(summary['mean_test_balanced_accuracy']):.4f} | "
            f"{float(summary['min_test_balanced_accuracy']):.4f} | {float(summary['mean_test_roc_auc']):.4f} | "
            f"{float(summary['difficulty_mean_test_balanced_accuracy']):.4f} | "
            f"{'YES' if row['intervention_capable'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Causal Tests",
            "",
            f"- Token intervention site: `{result['causal_tests']['token_site']}`",
            "",
        ]
    )
    for row in result["causal_tests"]["source_rows"]:
        lines.extend(
            [
                f"### Source Seed {int(row['source_seed'])}",
                "",
                f"- Gain-positive examples: `{int(row['gain_positive_examples'])}`",
                f"- Direction norm: `{float(row['direction_norm']):.4f}`",
            ]
        )
        ablation = row["ablation_on_source_positives"]
        if ablation.get("available"):
            lines.append(f"- Ablation baseline full accuracy on source positives: `{float(ablation['baseline_accuracy']):.4f}`")
            lines.append("")
            lines.append("| scale | direction | shuffled | orthogonal | noise |")
            lines.append("|---:|---:|---:|---:|---:|")
            for scale, values in ablation["accuracy_by_scale_and_condition"].items():
                lines.append(
                    f"| {scale} | {float(values['direction']):.4f} | {float(values['shuffled']):.4f} | "
                    f"{float(values['orthogonal']):.4f} | {float(values['noise']):.4f} |"
                )
        substitution = row["single_avenue_substitution_on_source_positives"]
        if substitution.get("available"):
            lines.append("")
            lines.append(
                f"- Source best-single substitution baseline accuracy: `{float(substitution['baseline_accuracy']):.4f}`"
            )
            lines.append("| scale | direction | shuffled | orthogonal | noise |")
            lines.append("|---:|---:|---:|---:|---:|")
            for scale, values in substitution["accuracy_by_scale_and_condition"].items():
                lines.append(
                    f"| {scale} | {float(values['direction']):.4f} | {float(values['shuffled']):.4f} | "
                    f"{float(values['orthogonal']):.4f} | {float(values['noise']):.4f} |"
                )
        if row["injection_on_matched_negative_targets"]:
            lines.append("")
            for target in row["injection_on_matched_negative_targets"]:
                result_row = target["result"]
                lines.extend(
                    [
                        f"- Target seed `{int(target['target_seed'])}` matched negatives: `{int(target['matched_negative_examples'])}`",
                        f"  Baseline full accuracy: `{float(result_row['baseline_accuracy']):.4f}`",
                    ]
                )
                lines.append("| scale | direction | shuffled | orthogonal | noise |")
                lines.append("|---:|---:|---:|---:|---:|")
                for scale, values in result_row["accuracy_by_scale_and_condition"].items():
                    lines.append(
                        f"| {scale} | {float(values['direction']):.4f} | {float(values['shuffled']):.4f} | "
                        f"{float(values['orthogonal']):.4f} | {float(values['noise']):.4f} |"
                    )
                lines.append("")
        else:
            lines.extend(["- No matched negative target sets met the minimum example count.", ""])
    verdict = result["overall_verdict"]
    lines.extend(
        [
            "## Verdict",
            "",
            f"- Stage 12c supported: `{verdict['stage12c_supported']}`",
            f"- Mean token-site transfer balanced accuracy: `{float(verdict['mean_token_site_transfer_balanced_accuracy']):.4f}`",
            f"- Token-site minus difficulty balanced accuracy: `{float(verdict['token_site_minus_difficulty_balanced_accuracy']):.4f}`",
            f"- Causal hit: `{'YES' if verdict['causal_hit'] else 'NO'}`",
            f"- Meaning: {verdict['meaning']}",
            f"- Recommended next step: {verdict['recommended_next_step']}",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
