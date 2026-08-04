from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

import run_stage12_make_or_break as stage12


FILE_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = FILE_PATH.parents[3]
CACHE_DIR = EXPERIMENT_ROOT / "results" / "stage12b_internal_structure_cache"
REPRESENTATIONS_PATH = CACHE_DIR / "representations.npy"
METADATA_PATH = CACHE_DIR / "metadata.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 12b internal structure discovery.")
    parser.add_argument("--preset", default="stage5_final", choices=("stage5_final",))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    result = run_internal_structure_discovery(
        preset_name=str(args.preset),
        device=str(args.device),
        batch_size=int(args.batch_size),
        force_rebuild_cache=bool(args.force_rebuild_cache),
    )
    payload = _load_existing_stage12_payload()
    payload["internal_structure_discovery"] = result
    stage12.RESULTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    stage12.REPORT_PATH.write_text(stage12._render_report(payload), encoding="utf-8")
    print(
        f"stage12b: wrote {stage12.RESULTS_PATH}, {stage12.REPORT_PATH}, "
        f"{REPRESENTATIONS_PATH}, and {METADATA_PATH}"
    )


def run_internal_structure_discovery(
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
    analysis = _analyze_cached_representations(
        representations=cache["representations"],
        metadata_rows=cache["metadata_rows"],
    )
    return {
        "metadata": {
            "stage": "stage12b_internal_structure_discovery",
            "created_at_utc": stage12._now(),
            "source_preset": preset_name,
            "device": device,
            "batch_size": int(batch_size),
            "cache_reused": bool(cache["cache_reused"]),
            "representation_rows": int(cache["representations"].shape[0]),
            "hidden_dim": int(cache["representations"].shape[1]),
            "seed_count": len(sorted({int(row["seed"]) for row in cache["metadata_rows"]})),
            "unique_example_count": len(sorted({str(row["example_id"]) for row in cache["metadata_rows"]})),
            "unique_example_seed_count": len(sorted({str(row["example_seed_id"]) for row in cache["metadata_rows"]})),
        },
        "cache": {
            "cache_dir": str(CACHE_DIR),
            "representations_path": str(REPRESENTATIONS_PATH),
            "metadata_path": str(METADATA_PATH),
        },
        **analysis,
    }


def _load_existing_stage12_payload() -> Dict[str, Any]:
    if not stage12.RESULTS_PATH.exists():
        raise FileNotFoundError(f"missing prior Stage 12 results at {stage12.RESULTS_PATH}")
    payload = json.loads(stage12.RESULTS_PATH.read_text(encoding="utf-8"))
    if "family_hypothesis" not in payload or "prior_hypothesis_avenue" not in payload:
        raise RuntimeError("existing Stage 12 results do not contain the required strict sections to preserve")
    return dict(payload)


def _load_or_build_cache(
    modules: Any,
    preset: stage12.ArtifactPreset,
    rows: Sequence[Mapping[str, Any]],
    device: str,
    batch_size: int,
    force_rebuild_cache: bool,
) -> Dict[str, Any]:
    if not force_rebuild_cache and REPRESENTATIONS_PATH.exists() and METADATA_PATH.exists():
        representations = np.load(REPRESENTATIONS_PATH)
        metadata_rows = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if _cache_is_valid(representations, metadata_rows):
            return {
                "representations": representations.astype(np.float32, copy=False),
                "metadata_rows": metadata_rows,
                "cache_reused": True,
            }
    representations, metadata_rows = _build_representation_cache(
        modules=modules,
        preset=preset,
        rows=rows,
        device=device,
        batch_size=batch_size,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(REPRESENTATIONS_PATH, representations.astype(np.float32, copy=False))
    METADATA_PATH.write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")
    return {
        "representations": representations.astype(np.float32, copy=False),
        "metadata_rows": metadata_rows,
        "cache_reused": False,
    }


def _cache_is_valid(representations: np.ndarray, metadata_rows: Any) -> bool:
    if representations.ndim != 2 or representations.shape[0] <= 0 or representations.shape[1] <= 0:
        return False
    if not isinstance(metadata_rows, list) or len(metadata_rows) != int(representations.shape[0]):
        return False
    required = {"example_id", "family", "seed", "avenue", "full_acc", "best_single_acc", "example_seed_id"}
    return all(required.issubset(set(row.keys())) for row in metadata_rows)


def _build_representation_cache(
    modules: Any,
    preset: stage12.ArtifactPreset,
    rows: Sequence[Mapping[str, Any]],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    vectors: List[np.ndarray] = []
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
        pooled, families, _keys = stage12._collect_raw_pooled_avenues(
            modules.dataset, system, test_examples, seed=seed, batch_size=batch_size
        )
        pooled = pooled.mean(dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
        labels = stage12._labels(test_examples)
        full_logits = stage12._logits_for_condition(system, test_examples, condition="none", seed=seed, batch_size=batch_size)
        full_preds = np.argmax(full_logits, axis=1).astype(np.int64)
        avenue_count = int(pooled.shape[1])
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
        best_single_acc = single_correct.any(axis=1).astype(np.int64)
        full_acc = (full_preds == labels).astype(np.int64)
        for example_index, example in enumerate(test_examples):
            example_id = str(example.id)
            family = str(families[example_index])
            example_seed_id = f"{seed}:{example_id}"
            for avenue_index in range(avenue_count):
                vectors.append(pooled[example_index, avenue_index])
                metadata_rows.append(
                    {
                        "row_index": len(metadata_rows),
                        "example_id": example_id,
                        "example_seed_id": example_seed_id,
                        "family": family,
                        "seed": seed,
                        "avenue": int(avenue_index),
                        "label": int(labels[example_index]),
                        "full_acc": int(full_acc[example_index]),
                        "best_single_acc": int(best_single_acc[example_index]),
                        "full_minus_best_single": int(full_acc[example_index] - best_single_acc[example_index]),
                        "target_set_member": bool(full_acc[example_index] == 1 and best_single_acc[example_index] == 0),
                    }
                )
        del result
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return np.asarray(vectors, dtype=np.float32), metadata_rows


def _analyze_cached_representations(
    representations: np.ndarray,
    metadata_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    raw_grid = _run_kmeans_grid("kmeans_raw", representations, metadata_rows)
    pca_components = max(1, min(10, representations.shape[0], representations.shape[1]))
    pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=0)
    reduced = pca.fit_transform(representations)
    pca_grid = _run_kmeans_grid("pca10_kmeans", reduced.astype(np.float32, copy=False), metadata_rows)
    agreement_with_raw = {
        str(k): float(adjusted_rand_score(raw_grid["labels_by_k"][k], pca_grid["labels_by_k"][k]))
        for k in raw_grid["labels_by_k"].keys()
    }
    pca_grid["summary"]["agreement_with_raw_by_k"] = agreement_with_raw
    pca_grid["summary"]["pca_explained_variance_ratio"] = [float(value) for value in pca.explained_variance_ratio_.tolist()]
    pca_grid["summary"]["pca_explained_variance_total"] = float(np.sum(pca.explained_variance_ratio_))

    hierarchical = _hierarchical_external_family_summary(representations, metadata_rows)
    chosen_method, chosen_k = _choose_selected_clustering(raw_grid["summary"], pca_grid["summary"])
    most_stable_result = _most_stable_result(raw_grid["summary"], pca_grid["summary"])
    selected_grid = raw_grid if chosen_method == "kmeans_raw" else pca_grid
    selected_labels = selected_grid["labels_by_k"][chosen_k]
    modal_entries = selected_grid["modal_entries_by_k"][chosen_k]
    cluster_characterization = _characterize_clusters(
        labels=selected_labels,
        representations=representations,
        metadata_rows=metadata_rows,
        modal_entries=modal_entries,
        k=chosen_k,
    )
    stability = selected_grid["summary"]["results_by_k"][str(chosen_k)]["stability"]
    difficulty = _difficulty_confound_check(modal_entries)
    target_alignment = _target_set_alignment(modal_entries, chosen_k)
    interpretation = _interpretation_summary(
        stability=stability,
        difficulty=difficulty,
        target_alignment=target_alignment,
        cluster_characterization=cluster_characterization,
        selected_agreement_with_pca=float(agreement_with_raw.get(str(chosen_k), 1.0)),
    )
    return {
        "clustering_summary": {
            "optimal_k": int(chosen_k),
            "optimal_method": chosen_method,
            "method_that_produced_most_stable_clusters": most_stable_result["label"],
            "selected_split_agreement_with_pca": float(agreement_with_raw.get(str(chosen_k), 1.0)),
            "raw_vs_pca_agreement_by_k": agreement_with_raw,
            "cluster_sizes": cluster_characterization["cluster_sizes"],
            "pairwise_cluster_mean_cosine_summary": cluster_characterization["pairwise_cluster_mean_cosine"]["summary"],
        },
        "methods": {
            "kmeans_raw": raw_grid["summary"],
            "pca10_kmeans": pca_grid["summary"],
            "hierarchical_external_family_means": hierarchical,
        },
        "selected_clustering": {
            "method": chosen_method,
            "k": int(chosen_k),
            "cluster_characterization": cluster_characterization["clusters"],
            "pairwise_cluster_mean_cosine": cluster_characterization["pairwise_cluster_mean_cosine"],
        },
        "stability": stability,
        "difficulty_confound_check": difficulty,
        "target_set_alignment": target_alignment,
        "interpretation": interpretation,
    }


def _run_kmeans_grid(
    method_name: str,
    features: np.ndarray,
    metadata_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    max_k = min(8, max(2, features.shape[0] - 1))
    labels_by_k: Dict[int, np.ndarray] = {}
    modal_entries_by_k: Dict[int, List[Dict[str, Any]]] = {}
    results_by_k: Dict[str, Dict[str, Any]] = {}
    inertias: Dict[int, float] = {}
    silhouettes: Dict[int, float] = {}
    stabilities: Dict[int, float] = {}
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=0, max_iter=200)
        labels = model.fit_predict(features)
        labels_by_k[k] = labels.astype(np.int64, copy=False)
        modal_entries = _modal_cluster_entries(labels, metadata_rows)
        modal_entries_by_k[k] = modal_entries
        cluster_sizes = {str(index): int(count) for index, count in sorted(Counter(labels.tolist()).items())}
        silhouette = _safe_silhouette_score(features, labels)
        stability = _cluster_stability_summary(modal_entries)
        labels_modal = np.asarray([int(entry["cluster"]) for entry in modal_entries], dtype=np.int64)
        gap_values = np.asarray([float(entry["full_minus_best_single"]) for entry in modal_entries], dtype=np.float64)
        results_by_k[str(k)] = {
            "k": int(k),
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette),
            "cluster_sizes": cluster_sizes,
            "stability": stability,
            "correlation_ratio_gap": float(_correlation_ratio(labels_modal, gap_values)),
        }
        inertias[k] = float(model.inertia_)
        silhouettes[k] = float(silhouette)
        stabilities[k] = float(stability["mean_per_example_cluster_stability"])
    elbow_k = _select_elbow_k(inertias)
    best_silhouette_k = max(silhouettes, key=lambda item: (silhouettes[item], -item))
    most_stable_k = max(stabilities, key=lambda item: (stabilities[item], silhouettes[item], -item))
    return {
        "summary": {
            "method": method_name,
            "elbow_k": int(elbow_k),
            "best_silhouette_k": int(best_silhouette_k),
            "most_stable_k": int(most_stable_k),
            "results_by_k": results_by_k,
        },
        "labels_by_k": labels_by_k,
        "modal_entries_by_k": modal_entries_by_k,
    }


def _safe_silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if unique.size <= 1:
        return 0.0
    sample_size = min(5000, int(features.shape[0]))
    try:
        return float(
            silhouette_score(
                features,
                labels,
                metric="euclidean",
                sample_size=sample_size,
                random_state=0,
            )
        )
    except Exception:
        return 0.0


def _select_elbow_k(inertias: Mapping[int, float]) -> int:
    items = sorted((int(k), float(v)) for k, v in inertias.items())
    if len(items) <= 2:
        return items[0][0]
    xs = np.asarray([item[0] for item in items], dtype=np.float64)
    ys = np.asarray([item[1] for item in items], dtype=np.float64)
    start = np.asarray([xs[0], ys[0]], dtype=np.float64)
    end = np.asarray([xs[-1], ys[-1]], dtype=np.float64)
    baseline = end - start
    baseline_norm = np.linalg.norm(baseline)
    if baseline_norm <= 0.0:
        return int(xs[0])
    distances = []
    for x, y in zip(xs, ys):
        point = np.asarray([x, y], dtype=np.float64)
        vector = point - start
        area = float(np.abs((baseline[0] * vector[1]) - (baseline[1] * vector[0])))
        distance = area / baseline_norm
        distances.append(distance)
    best_index = int(np.argmax(np.asarray(distances)))
    return int(xs[best_index])


def _choose_selected_clustering(
    raw_summary: Mapping[str, Any],
    pca_summary: Mapping[str, Any],
) -> tuple[str, int]:
    candidates = []
    for method_name, summary in (("kmeans_raw", raw_summary), ("pca10_kmeans", pca_summary)):
        elbow_k = int(summary["elbow_k"])
        result = summary["results_by_k"][str(elbow_k)]
        candidates.append(
            (
                float(result["stability"]["mean_per_example_cluster_stability"]),
                float(result["silhouette"]),
                method_name,
                elbow_k,
            )
        )
    candidates.sort(reverse=True)
    _stability, _silhouette, method_name, k = candidates[0]
    return str(method_name), int(k)


def _most_stable_result(
    raw_summary: Mapping[str, Any],
    pca_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    candidates = []
    for method_name, summary in (("kmeans_raw", raw_summary), ("pca10_kmeans", pca_summary)):
        for key, result in summary["results_by_k"].items():
            candidates.append(
                (
                    float(result["stability"]["mean_per_example_cluster_stability"]),
                    float(result["silhouette"]),
                    method_name,
                    int(key),
                )
            )
    candidates.sort(reverse=True)
    stability, silhouette, method_name, k = candidates[0]
    return {
        "label": f"{method_name}@K={k}",
        "method": method_name,
        "k": int(k),
        "stability": float(stability),
        "silhouette": float(silhouette),
    }


def _modal_cluster_entries(
    labels: Sequence[int] | np.ndarray,
    metadata_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for label, metadata in zip(labels, metadata_rows):
        key = str(metadata["example_seed_id"])
        entry = grouped.setdefault(
            key,
            {
                "example_id": str(metadata["example_id"]),
                "example_seed_id": key,
                "seed": int(metadata["seed"]),
                "family": str(metadata["family"]),
                "full_acc": int(metadata["full_acc"]),
                "best_single_acc": int(metadata["best_single_acc"]),
                "full_minus_best_single": int(metadata["full_minus_best_single"]),
                "target_set_member": bool(metadata["target_set_member"]),
                "cluster_votes": [],
            },
        )
        entry["cluster_votes"].append(int(label))
    rows: List[Dict[str, Any]] = []
    for entry in sorted(grouped.values(), key=lambda item: (int(item["seed"]), str(item["example_id"]))):
        counts = Counter(entry["cluster_votes"])
        cluster, votes = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        rows.append(
            {
                "example_id": str(entry["example_id"]),
                "example_seed_id": str(entry["example_seed_id"]),
                "seed": int(entry["seed"]),
                "family": str(entry["family"]),
                "cluster": int(cluster),
                "cluster_vote_consistency": float(votes / max(1, len(entry["cluster_votes"]))),
                "full_acc": int(entry["full_acc"]),
                "best_single_acc": int(entry["best_single_acc"]),
                "full_minus_best_single": int(entry["full_minus_best_single"]),
                "target_set_member": bool(entry["target_set_member"]),
            }
        )
    return rows


def _cluster_stability_summary(modal_entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_example: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for entry in modal_entries:
        by_example[str(entry["example_id"])].append(entry)
    per_example: Dict[str, Dict[str, Any]] = {}
    stable_values: List[float] = []
    single_seed_only = 0
    for example_id, rows in sorted(by_example.items()):
        cluster_ids = [int(row["cluster"]) for row in rows]
        seeds = [int(row["seed"]) for row in rows]
        counts = Counter(cluster_ids)
        dominant_cluster, dominant_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        stability = float(dominant_count / max(1, len(cluster_ids)))
        per_example[example_id] = {
            "n_seeds_present": len(cluster_ids),
            "seeds": seeds,
            "dominant_cluster": int(dominant_cluster),
            "stability": stability,
        }
        if len(cluster_ids) >= 2:
            stable_values.append(stability)
        else:
            single_seed_only += 1
    mean_stability = stage12._mean(stable_values)
    return {
        "mean_per_example_cluster_stability": mean_stability,
        "stable_above_080": bool(mean_stability >= 0.80),
        "eligible_example_count": int(len(stable_values)),
        "single_seed_only_example_count": int(single_seed_only),
        "per_example_cluster_stability": per_example,
    }


def _characterize_clusters(
    labels: np.ndarray,
    representations: np.ndarray,
    metadata_rows: Sequence[Mapping[str, Any]],
    modal_entries: Sequence[Mapping[str, Any]],
    k: int,
) -> Dict[str, Any]:
    cluster_means: Dict[int, np.ndarray] = {}
    cluster_rows: Dict[str, Dict[str, Any]] = {}
    rows_by_cluster: Dict[int, List[int]] = defaultdict(list)
    for index, cluster in enumerate(labels.tolist()):
        rows_by_cluster[int(cluster)].append(index)
    modal_by_cluster: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for entry in modal_entries:
        modal_by_cluster[int(entry["cluster"])].append(entry)

    for cluster in range(k):
        indices = rows_by_cluster.get(cluster, [])
        if indices:
            cluster_means[cluster] = representations[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        else:
            cluster_means[cluster] = np.zeros(representations.shape[1], dtype=np.float32)
    cosine = _pairwise_mean_cosine(cluster_means)

    for cluster in range(k):
        indices = rows_by_cluster.get(cluster, [])
        row_subset = [metadata_rows[index] for index in indices]
        family_distribution = _distribution([str(row["family"]) for row in row_subset])
        avenue_distribution = _distribution([str(row["avenue"]) for row in row_subset])
        modal_subset = modal_by_cluster.get(cluster, [])
        mean_full = stage12._mean([float(row["full_acc"]) for row in modal_subset])
        mean_best_single = stage12._mean([float(row["best_single_acc"]) for row in modal_subset])
        mean_gap = stage12._mean([float(row["full_minus_best_single"]) for row in modal_subset])
        cluster_rows[str(cluster)] = {
            "row_count": int(len(indices)),
            "example_seed_count": int(len(modal_subset)),
            "unique_example_count": int(len({str(row["example_id"]) for row in modal_subset})),
            "external_family_distribution": family_distribution,
            "avenue_distribution": avenue_distribution,
            "accuracy_profile": {
                "mean_full_model_accuracy": mean_full,
                "mean_best_single_avenue_accuracy": mean_best_single,
                "mean_gap": mean_gap,
            },
            "difficulty_profile": _difficulty_profile(mean_full, mean_best_single),
            "cluster_mean_cosine_to_others": cosine["matrix"].get(str(cluster), {}),
        }
    return {
        "cluster_sizes": {str(cluster): int(len(rows_by_cluster.get(cluster, []))) for cluster in range(k)},
        "clusters": cluster_rows,
        "pairwise_cluster_mean_cosine": cosine,
    }


def _distribution(values: Sequence[str]) -> Dict[str, Any]:
    counts = Counter(str(value) for value in values)
    total = sum(counts.values())
    shares = {
        str(key): float(count / total) if total > 0 else 0.0
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
    }
    dominant_name = None
    dominant_share = 0.0
    if shares:
        dominant_name, dominant_share = next(iter(shares.items()))
    return {
        "counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "shares": shares,
        "dominant": dominant_name,
        "dominant_share": float(dominant_share),
    }


def _difficulty_profile(mean_full: float, mean_best_single: float) -> str:
    if mean_full >= 0.75 and mean_best_single >= 0.75:
        return "uniformly easy"
    if mean_full <= 0.25 and mean_best_single <= 0.25:
        return "uniformly hard"
    return "mixed"


def _pairwise_mean_cosine(cluster_means: Mapping[int, np.ndarray]) -> Dict[str, Any]:
    matrix: Dict[str, Dict[str, float]] = {}
    off_diag: List[float] = []
    normalized = {}
    for cluster, vector in cluster_means.items():
        norm = float(np.linalg.norm(vector))
        normalized[int(cluster)] = vector / max(norm, 1e-12)
    for cluster_a in sorted(normalized.keys()):
        row: Dict[str, float] = {}
        for cluster_b in sorted(normalized.keys()):
            cosine = float(np.dot(normalized[cluster_a], normalized[cluster_b]))
            row[str(cluster_b)] = cosine
            if cluster_a != cluster_b:
                off_diag.append(cosine)
        matrix[str(cluster_a)] = row
    return {
        "matrix": matrix,
        "summary": {
            "mean_off_diagonal": stage12._mean(off_diag),
            "max_off_diagonal": max(off_diag) if off_diag else 1.0,
            "min_off_diagonal": min(off_diag) if off_diag else 1.0,
        },
    }


def _correlation_ratio(categories: np.ndarray, values: np.ndarray) -> float:
    if categories.size == 0 or values.size == 0 or categories.size != values.size:
        return 0.0
    values = values.astype(np.float64, copy=False)
    grand_mean = float(np.mean(values))
    total_ss = float(np.sum((values - grand_mean) ** 2))
    if total_ss <= 0.0:
        return 0.0
    between_ss = 0.0
    for category in np.unique(categories):
        mask = categories == category
        subset = values[mask]
        if subset.size == 0:
            continue
        diff = float(np.mean(subset) - grand_mean)
        between_ss += float(subset.size) * diff * diff
    return float(np.sqrt(max(0.0, between_ss / total_ss)))


def _difficulty_confound_check(modal_entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    clusters = np.asarray([int(entry["cluster"]) for entry in modal_entries], dtype=np.int64)
    full_acc = np.asarray([float(entry["full_acc"]) for entry in modal_entries], dtype=np.float64)
    best_single = np.asarray([float(entry["best_single_acc"]) for entry in modal_entries], dtype=np.float64)
    gap = np.asarray([float(entry["full_minus_best_single"]) for entry in modal_entries], dtype=np.float64)
    full_eta = _correlation_ratio(clusters, full_acc)
    best_single_eta = _correlation_ratio(clusters, best_single)
    gap_eta = _correlation_ratio(clusters, gap)
    return {
        "correlation_ratio_full_accuracy": float(full_eta),
        "correlation_ratio_best_single_accuracy": float(best_single_eta),
        "correlation_ratio_gap": float(gap_eta),
        "is_purely_difficulty_stratified": bool(max(full_eta, best_single_eta, gap_eta) >= 0.80),
    }


def _target_set_alignment(modal_entries: Sequence[Mapping[str, Any]], k: int) -> Dict[str, Any]:
    target_entries = [entry for entry in modal_entries if bool(entry["target_set_member"])]
    overall_counts = Counter(int(entry["cluster"]) for entry in modal_entries)
    target_counts = Counter(int(entry["cluster"]) for entry in target_entries)
    total_all = max(1, len(modal_entries))
    total_target = max(1, len(target_entries))
    overall_distribution = {
        str(cluster): float(overall_counts.get(cluster, 0) / total_all) for cluster in range(k)
    }
    target_distribution = {
        str(cluster): float(target_counts.get(cluster, 0) / total_target) for cluster in range(k)
    }
    enrichment = {}
    concentrated_clusters: List[int] = []
    for cluster in range(k):
        baseline = overall_distribution[str(cluster)]
        target_share = target_distribution[str(cluster)]
        ratio = float(target_share / baseline) if baseline > 0 else 0.0
        enrichment[str(cluster)] = ratio
        if ratio >= 1.5 and target_counts.get(cluster, 0) >= 20:
            concentrated_clusters.append(cluster)
    contingency = np.asarray(
        [
            [int(target_counts.get(cluster, 0)), int(overall_counts.get(cluster, 0) - target_counts.get(cluster, 0))]
            for cluster in range(k)
        ],
        dtype=np.int64,
    )
    cramers_v = 0.0
    p_value = 1.0
    if contingency.size > 0 and contingency.sum() > 0 and contingency.shape[0] > 1:
        chi2, p_value, _dof, _expected = chi2_contingency(contingency, correction=False)
        n = float(contingency.sum())
        cramers_v = float(np.sqrt(max(0.0, chi2 / max(n, 1.0))))
    concentrates = bool(cramers_v >= 0.20 and concentrated_clusters)
    return {
        "target_example_seed_count": int(len(target_entries)),
        "overall_cluster_distribution": overall_distribution,
        "target_cluster_distribution": target_distribution,
        "enrichment_by_cluster": enrichment,
        "cramers_v": cramers_v,
        "chi2_p_value": float(p_value),
        "concentrates_in_specific_clusters": concentrates,
        "concentrated_clusters": concentrated_clusters,
    }


def _hierarchical_external_family_summary(
    representations: np.ndarray,
    metadata_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    family_to_indices: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(metadata_rows):
        family_to_indices[str(row["family"])].append(index)
    family_labels = sorted(family_to_indices.keys())
    if len(family_labels) <= 1:
        return {
            "family_labels": family_labels,
            "merge_distances": [],
            "largest_gap": 0.0,
            "suggested_cluster_count": len(family_labels),
            "suggested_cut_distance": 0.0,
            "linkage_rows": [],
        }
    family_means = np.stack(
        [
            representations[np.asarray(family_to_indices[family], dtype=np.int64)].mean(axis=0)
            for family in family_labels
        ],
        axis=0,
    )
    condensed = pdist(family_means, metric="cosine")
    linkage_matrix = linkage(condensed, method="average")
    merge_distances = [float(value) for value in linkage_matrix[:, 2].tolist()]
    largest_gap = 0.0
    suggested_clusters = len(family_labels)
    suggested_cut = merge_distances[0] if merge_distances else 0.0
    if len(merge_distances) >= 2:
        gaps = np.diff(np.asarray(merge_distances, dtype=np.float64))
        gap_index = int(np.argmax(gaps))
        largest_gap = float(gaps[gap_index])
        suggested_clusters = len(family_labels) - (gap_index + 1)
        suggested_cut = float((merge_distances[gap_index] + merge_distances[gap_index + 1]) / 2.0)
    linkage_rows = [
        {
            "left": int(row[0]),
            "right": int(row[1]),
            "distance": float(row[2]),
            "count": int(row[3]),
        }
        for row in linkage_matrix.tolist()
    ]
    return {
        "family_labels": family_labels,
        "merge_distances": merge_distances,
        "largest_gap": largest_gap,
        "suggested_cluster_count": int(max(1, suggested_clusters)),
        "suggested_cut_distance": suggested_cut,
        "linkage_rows": linkage_rows,
    }


def _interpretation_summary(
    stability: Mapping[str, Any],
    difficulty: Mapping[str, Any],
    target_alignment: Mapping[str, Any],
    cluster_characterization: Mapping[str, Any],
    selected_agreement_with_pca: float,
) -> Dict[str, Any]:
    stable = bool(stability.get("stable_above_080"))
    difficulty_confounded = bool(difficulty.get("is_purely_difficulty_stratified"))
    concentrates = bool(target_alignment.get("concentrates_in_specific_clusters"))
    cosine_summary = cluster_characterization["pairwise_cluster_mean_cosine"]["summary"]
    mean_cosine = float(cosine_summary["mean_off_diagonal"])
    if difficulty_confounded:
        representation = "The discovered groups look primarily difficulty-stratified rather than semantic."
        viable = "NO"
        next_step = (
            "Reframe the communication hypothesis around difficulty-aware routing if you want to pursue the shared-signal story."
        )
    elif not stable:
        representation = "The clustered pooled representations are not stable enough across seeds to support a robust latent grouping claim."
        viable = "NO"
        next_step = (
            "Inspect individual layer representations next; pooled representations appear to wash out the stable structure."
        )
    elif mean_cosine >= 0.90:
        representation = "The groups are stable but their mean directions remain too similar to support a clean cluster primitive."
        viable = "UNCLEAR"
        next_step = (
            "Probe deeper layers or alternative normalizations before redesigning Sub-experiments A and B around these clusters."
        )
    elif selected_agreement_with_pca < 0.50 and not concentrates:
        representation = (
            "A coarse latent grouping is real, but the elbow-selected refinement is method-sensitive and it does not isolate the target-set cases."
        )
        viable = "UNCLEAR"
        next_step = (
            "Use the stable 2-cluster split as a coarse probe or move to layerwise representations before redesigning Sub-experiments A and B."
        )
    elif selected_agreement_with_pca < 0.50:
        representation = "The clusters are stable at a coarse level, but the selected K refinement is method-sensitive."
        viable = "UNCLEAR"
        next_step = (
            "Confirm the discovered grouping at the layer level before treating the finer split as the new family primitive."
        )
    elif not concentrates:
        representation = "The clusters are stable and not purely difficulty-based, but the target-set cases do not concentrate in a distinctive cluster."
        viable = "UNCLEAR"
        next_step = (
            "Redesign the next probe around the stable coarse split only, or search for a decomposition more directly tied to the full-vs-single-avenue gain."
        )
    else:
        representation = "The clustered pooled representations define a stable latent grouping that is not dominated by difficulty."
        viable = "YES"
        next_step = (
            "Redesign Sub-experiments A and B to use the discovered cluster labels instead of the external restore-bucket families."
        )
    if concentrates:
        representation += " Target-set examples are enriched in a subset of the discovered clusters."
    return {
        "what_clusters_appear_to_represent": representation,
        "viable_unit_of_analysis_for_communication_hypothesis": viable,
        "recommended_next_step": next_step,
    }


if __name__ == "__main__":
    main()
