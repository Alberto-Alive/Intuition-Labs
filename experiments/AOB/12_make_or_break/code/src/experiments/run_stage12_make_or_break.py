from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch


FILE_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = FILE_PATH.parents[3]
REPO_ROOT = FILE_PATH.parents[6]
RESULTS_PATH = EXPERIMENT_ROOT / "results" / "stage12_make_or_break_results.json"
REPORT_PATH = EXPERIMENT_ROOT / "reports" / "STAGE12_MAKE_OR_BREAK.md"


@dataclass(frozen=True)
class ArtifactPreset:
    name: str
    experiment_root: Path
    results_path: Path
    phase: str
    candidate: str
    base_code_root: Path
    overlay_src_root: Path | None
    note: str


@dataclass(frozen=True)
class FamilyThresholds:
    family_shared_ratio_min: float = 0.60
    family_probe_min: float = 0.70
    residual_family_probe_max: float = 0.35
    duplicate_prompt_residual_norm_max: float = 1e-7
    same_family_alignment_tolerance: float = 0.05
    min_target_seed_count: int = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 12 family-shared checkpoint analysis.")
    parser.add_argument(
        "--preset",
        default="stage5_final",
        choices=("stage5_final", "stage6_reference_cheap"),
        help="Artifact source preset to analyze.",
    )
    parser.add_argument(
        "--allow-fallback",
        default="",
        choices=("", "stage6_reference_cheap"),
        help="Fallback preset to use when the requested preset has no available checkpoints.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-epochs", type=int, default=120)
    parser.add_argument("--probe-lr", type=float, default=0.05)
    parser.add_argument("--target-min-examples", type=int, default=16)
    args = parser.parse_args()

    requested = _artifact_presets()[str(args.preset)]
    fallback = _artifact_presets().get(str(args.allow_fallback)) if str(args.allow_fallback) else None
    result = run_analysis(
        requested_preset=requested,
        fallback_preset=fallback,
        device=str(args.device),
        batch_size=int(args.batch_size),
        probe_epochs=int(args.probe_epochs),
        probe_lr=float(args.probe_lr),
        target_min_examples=int(args.target_min_examples),
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_report(result), encoding="utf-8")
    print(f"stage12: wrote {RESULTS_PATH} and {REPORT_PATH}")


def run_analysis(
    requested_preset: ArtifactPreset,
    fallback_preset: ArtifactPreset | None,
    device: str,
    batch_size: int,
    probe_epochs: int,
    probe_lr: float,
    target_min_examples: int,
) -> Dict[str, Any]:
    prior_avenue = _load_prior_avenue_hypothesis()
    active_preset, fallback_reason = _select_preset(requested_preset, fallback_preset)
    modules = _bootstrap_modules(active_preset)
    payload = json.loads(active_preset.results_path.read_text(encoding="utf-8"))
    rows = _completed_rows(payload, active_preset)
    if not rows:
        raise RuntimeError(f"no completed rows found for preset={active_preset.name}")

    thresholds = FamilyThresholds()
    seed_entries = [
        _collect_seed_entry(modules=modules, preset=active_preset, row=row, device=device, batch_size=batch_size)
        for row in rows
    ]
    aggregate = _aggregate_seed_entries(seed_entries)
    family_means = compute_family_means(aggregate["combined_pooled"], aggregate["combined_families"])
    family_analysis = _run_family_hypothesis(
        modules=modules,
        seed_entries=seed_entries,
        aggregate=aggregate,
        family_means=family_means,
        thresholds=thresholds,
        device=device,
        batch_size=batch_size,
        probe_epochs=probe_epochs,
        probe_lr=probe_lr,
        target_min_examples=target_min_examples,
    )

    return {
        "metadata": {
            "stage": "stage12_make_or_break_family_redesign",
            "created_at_utc": _now(),
            "repo_root": str(REPO_ROOT),
            "requested_preset": requested_preset.name,
            "actual_preset": active_preset.name,
            "fallback_reason": fallback_reason,
            "source_note": active_preset.note,
            "device": device,
            "batch_size": int(batch_size),
            "probe_epochs": int(probe_epochs),
            "probe_lr": float(probe_lr),
            "target_min_examples": int(target_min_examples),
            "strict_stage5_final_available": active_preset.name == "stage5_final",
        },
        "artifact_source": {
            "results_path": str(active_preset.results_path),
            "phase": active_preset.phase,
            "candidate": active_preset.candidate,
            "row_count": len(rows),
            "checkpoint_paths": [entry["checkpoint_path"] for entry in seed_entries],
        },
        "prior_hypothesis_avenue": prior_avenue,
        "family_hypothesis": family_analysis,
    }


def _artifact_presets() -> Dict[str, ArtifactPreset]:
    stage5_root = REPO_ROOT / "experiments" / "AOB" / "02_selected_multi_avenue_portfolio"
    stage0_root = REPO_ROOT / "experiments" / "AOB" / "00_validated_candidate_token_direct_coordination"
    stage6_root = REPO_ROOT / "experiments" / "AOB" / "10_complementarity_regularized_candidate_token_direct_coordination"
    return {
        "stage5_final": ArtifactPreset(
            name="stage5_final",
            experiment_root=stage5_root,
            results_path=stage5_root / "results" / "stage5_multi_avenue_search_results.json",
            phase="final",
            candidate="avenue4_goal_dropout05_lr3e4_clip1",
            base_code_root=stage0_root / "code",
            overlay_src_root=stage5_root / "code" / "src",
            note="Strict source requested by the Stage 12 todo: final Stage 5 selected checkpoints.",
        ),
        "stage6_reference_cheap": ArtifactPreset(
            name="stage6_reference_cheap",
            experiment_root=stage6_root,
            results_path=stage6_root / "results" / "stage6_complementarity_search_results.json",
            phase="cheap",
            candidate="stage5_reference_dropout05",
            base_code_root=stage6_root / "code",
            overlay_src_root=None,
            note="Fallback smoke source: preserved cheap-stage reference checkpoints for the Stage 5 selected architecture.",
        ),
    }


def _load_prior_avenue_hypothesis() -> Dict[str, Any] | None:
    if not RESULTS_PATH.exists():
        return None
    try:
        payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if "prior_hypothesis_avenue" in payload:
        return dict(payload["prior_hypothesis_avenue"])
    if "subexperiment_a" in payload and "subexperiment_b" in payload:
        return {
            "source": "pre_redesign_stage12_results",
            "metadata": payload.get("metadata", {}),
            "artifact_source": payload.get("artifact_source", {}),
            "subexperiment_a": payload.get("subexperiment_a", {}),
            "subexperiment_b": payload.get("subexperiment_b", {}),
        }
    return None


def _select_preset(
    requested_preset: ArtifactPreset,
    fallback_preset: ArtifactPreset | None,
) -> tuple[ArtifactPreset, str | None]:
    if _preset_has_checkpoints(requested_preset):
        return requested_preset, None
    if fallback_preset is not None and _preset_has_checkpoints(fallback_preset):
        return fallback_preset, (
            f"requested preset {requested_preset.name} has no available checkpoint files in this workspace; "
            f"used {fallback_preset.name} instead"
        )
    raise FileNotFoundError(
        f"requested preset {requested_preset.name} is unavailable; checked {requested_preset.results_path}"
    )


def _preset_has_checkpoints(preset: ArtifactPreset) -> bool:
    if not preset.results_path.exists():
        return False
    try:
        payload = json.loads(preset.results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    for row in _completed_rows(payload, preset):
        path = _resolve_checkpoint_path(preset.experiment_root, row)
        if path is not None and path.exists():
            return True
    return False


def _bootstrap_modules(preset: ArtifactPreset) -> SimpleNamespace:
    base_code_root = str(preset.base_code_root.resolve())
    if base_code_root not in sys.path:
        sys.path.insert(0, base_code_root)
    importlib.invalidate_caches()
    import src  # type: ignore
    import src.coordinators  # type: ignore
    import src.experiments  # type: ignore

    if preset.overlay_src_root is not None:
        overlay_src = preset.overlay_src_root.resolve()
        overlay_src_str = str(overlay_src)
        if overlay_src_str not in src.__path__:
            src.__path__.insert(0, overlay_src_str)
        overlay_experiments = overlay_src / "experiments"
        if overlay_experiments.exists():
            overlay_experiments_str = str(overlay_experiments)
            if overlay_experiments_str not in src.experiments.__path__:
                src.experiments.__path__.insert(0, overlay_experiments_str)
        overlay_coordinators = overlay_src / "coordinators"
        if overlay_coordinators.exists():
            overlay_coordinators_str = str(overlay_coordinators)
            if overlay_coordinators_str not in src.coordinators.__path__:
                src.coordinators.__path__.insert(0, overlay_coordinators_str)
        importlib.invalidate_caches()

    dataset_mod = importlib.import_module("src.datasets.multiview_code_patch_selection")
    latent_mod = importlib.import_module("src.experiments.real_shared_weight_latent_coordination")
    loader_mod = importlib.import_module("src.experiments.run_latent_vs_text_efficiency")
    return SimpleNamespace(dataset=dataset_mod, latent=latent_mod, loader=loader_mod)


def _completed_rows(payload: Mapping[str, Any], preset: ArtifactPreset) -> List[Dict[str, Any]]:
    rows = [
        dict(row)
        for row in payload.get("phases", {}).get(preset.phase, [])
        if row.get("status") == "completed" and str(row.get("candidate")) == preset.candidate
    ]
    rows.sort(key=lambda row: int(row.get("seed", 0)))
    return rows


def _resolve_checkpoint_path(experiment_root: Path, row: Mapping[str, Any]) -> Path | None:
    raw = row.get("checkpoint_paths", {}).get("trainable")
    if not raw:
        return None
    return experiment_root / Path(str(raw))


def _load_checkpoint(modules: SimpleNamespace, path: Path, device: str):
    return modules.loader._load_latent_checkpoint(path, device=device)  # type: ignore[attr-defined]


def _make_dataset_config(dataset_mod, row: Mapping[str, Any]):
    cfg_cls = dataset_mod.MultiViewCodePatchDatasetConfig
    raw = dict(row.get("dataset_config", {}))
    allowed = {field.name for field in fields(cfg_cls)}
    values = {key: value for key, value in raw.items() if key in allowed}
    if "source_roots" in values:
        values["source_roots"] = tuple(values["source_roots"])
    return cfg_cls(**values)


def _example_batches(examples: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(examples), batch_size):
        yield examples[start : start + batch_size]


def _labels(examples: Sequence[Any]) -> np.ndarray:
    return np.asarray([int(example.label) for example in examples], dtype=np.int64)


def _example_keys(seed: int, split_name: str, examples: Sequence[Any]) -> List[str]:
    return [f"{seed}:{split_name}:{str(example.id)}" for example in examples]


def _families(dataset_mod, examples: Sequence[Any]) -> List[str]:
    return [str(dataset_mod.example_oracle_metadata(example).get("problem_family", "unknown")) for example in examples]


def _collect_raw_pooled_avenues(
    dataset_mod,
    system,
    examples: Sequence[Any],
    seed: int,
    batch_size: int,
) -> tuple[torch.Tensor, List[str], List[str]]:
    pooled_batches: List[torch.Tensor] = []
    families: List[str] = []
    system.eval()
    with torch.no_grad():
        for batch in _example_batches(examples, batch_size):
            readouts = system.collect_clone_representations(batch, condition="none", seed=seed)
            pooled_batches.append(readouts["raw_pooled_avenues"].detach().cpu())
            families.extend(_families(dataset_mod, batch))
    return torch.cat(pooled_batches, dim=0), families, _example_keys(seed, "unknown", examples)


def _collect_seed_entry(
    modules: SimpleNamespace,
    preset: ArtifactPreset,
    row: Mapping[str, Any],
    device: str,
    batch_size: int,
) -> Dict[str, Any]:
    checkpoint_path = _resolve_checkpoint_path(preset.experiment_root, row)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint for seed={row.get('seed')} at {checkpoint_path}")
    result = _load_checkpoint(modules, checkpoint_path, device=device)
    dataset_config = _make_dataset_config(modules.dataset, row)
    seed = int(row.get("seed", 0))
    splits = modules.dataset.build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=REPO_ROOT)
    train_examples = list(splits["train"])
    test_examples = list(splits["test"])
    train_pooled, train_families, _ = _collect_raw_pooled_avenues(
        modules.dataset, result.system, train_examples, seed=seed, batch_size=batch_size
    )
    test_pooled, test_families, _ = _collect_raw_pooled_avenues(
        modules.dataset, result.system, test_examples, seed=seed, batch_size=batch_size
    )
    return {
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "dataset_config": dict(row.get("dataset_config", {})),
        "train_examples": train_examples,
        "test_examples": test_examples,
        "train_pooled": train_pooled,
        "test_pooled": test_pooled,
        "train_families": train_families,
        "test_families": test_families,
        "train_keys": _example_keys(seed, "train", train_examples),
        "test_keys": _example_keys(seed, "test", test_examples),
    }


def _aggregate_seed_entries(seed_entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    train_pooled = torch.cat([entry["train_pooled"] for entry in seed_entries], dim=0)
    test_pooled = torch.cat([entry["test_pooled"] for entry in seed_entries], dim=0)
    combined_pooled = torch.cat([train_pooled, test_pooled], dim=0)
    train_families = [family for entry in seed_entries for family in entry["train_families"]]
    test_families = [family for entry in seed_entries for family in entry["test_families"]]
    train_keys = [key for entry in seed_entries for key in entry["train_keys"]]
    test_keys = [key for entry in seed_entries for key in entry["test_keys"]]
    combined_families = train_families + test_families
    combined_keys = train_keys + test_keys
    return {
        "train_pooled": train_pooled,
        "test_pooled": test_pooled,
        "combined_pooled": combined_pooled,
        "train_families": train_families,
        "test_families": test_families,
        "combined_families": combined_families,
        "train_keys": train_keys,
        "test_keys": test_keys,
        "combined_keys": combined_keys,
        "seed_contributions": {
            int(entry["seed"]): {
                "n_train": int(entry["train_pooled"].shape[0]),
                "n_test": int(entry["test_pooled"].shape[0]),
                "test_family_labels": sorted(set(entry["test_families"])),
            }
            for entry in seed_entries
        },
    }


def compute_family_means(representations: torch.Tensor, families: Sequence[str]) -> Dict[str, torch.Tensor]:
    if representations.dim() != 4:
        raise ValueError(f"expected [examples, roles, avenues, hidden], got {tuple(representations.shape)}")
    family_to_indices = _group_indices(families)
    return {
        family: representations[torch.as_tensor(indices, dtype=torch.long)].mean(dim=0)
        for family, indices in family_to_indices.items()
    }


def decompose_by_family(
    representations: torch.Tensor,
    families: Sequence[str],
    family_means: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if representations.dim() != 4:
        raise ValueError(f"expected [examples, roles, avenues, hidden], got {tuple(representations.shape)}")
    shared = torch.stack([family_means[str(family)] for family in families], dim=0)
    residual = representations - shared
    total_energy = representations.square().sum(dim=(1, 2, 3)).clamp_min(1e-12)
    shared_energy = shared.square().sum(dim=(1, 2, 3))
    residual_energy = residual.square().sum(dim=(1, 2, 3))
    return {
        "shared": shared,
        "residual": residual,
        "shared_ratio": shared_energy / total_energy,
        "residual_ratio": residual_energy / total_energy,
        "residual_norm": residual.reshape(residual.shape[0], -1).norm(dim=1),
    }


def _group_indices(labels: Sequence[str]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(str(label), []).append(index)
    return grouped


def _split_indices_within_groups(
    labels: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    grouped = _group_indices(labels)
    rng = np.random.default_rng(int(seed))
    train_indices: List[int] = []
    test_indices: List[int] = []
    excluded_groups: List[str] = []
    included_groups: List[str] = []
    for group, indices in sorted(grouped.items()):
        if len(indices) <= 1:
            excluded_groups.append(group)
            continue
        permuted = list(rng.permutation(np.asarray(indices, dtype=np.int64)).tolist())
        split_at = max(1, len(permuted) // 2)
        if split_at >= len(permuted):
            split_at = len(permuted) - 1
        train_indices.extend(int(index) for index in permuted[:split_at])
        test_indices.extend(int(index) for index in permuted[split_at:])
        included_groups.append(group)
    return {
        "train_indices": np.asarray(sorted(train_indices), dtype=np.int64),
        "test_indices": np.asarray(sorted(test_indices), dtype=np.int64),
        "included_groups": included_groups,
        "excluded_groups": excluded_groups,
    }


def _split_tensor_and_labels_within_groups(
    tensor: torch.Tensor,
    labels: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    split = _split_indices_within_groups(labels, seed=seed)
    train_indices = split["train_indices"]
    test_indices = split["test_indices"]
    return {
        **split,
        "train_tensor": tensor[torch.as_tensor(train_indices, dtype=torch.long)],
        "test_tensor": tensor[torch.as_tensor(test_indices, dtype=torch.long)],
        "train_labels": [str(labels[int(index)]) for index in train_indices.tolist()],
        "test_labels": [str(labels[int(index)]) for index in test_indices.tolist()],
    }


def _family_size_inventory(families: Sequence[str]) -> Dict[str, Any]:
    grouped = _group_indices(families)
    counts = {family: len(indices) for family, indices in sorted(grouped.items())}
    small = [family for family, count in counts.items() if count < 10]
    single = [family for family, count in counts.items() if count == 1]
    return {
        "counts": counts,
        "families_below_10": small,
        "single_example_families": single,
    }


def _family_ratio_rows(shared_ratio: torch.Tensor, families: Sequence[str], eligible_families: set[str]) -> Dict[str, Dict[str, float]]:
    values = shared_ratio.detach().cpu().numpy()
    grouped: Dict[str, List[float]] = {}
    for family, value in zip(families, values.tolist()):
        if family not in eligible_families:
            continue
        grouped.setdefault(str(family), []).append(float(value))
    return {
        family: {
            "n_examples": len(items),
            "mean_family_shared_ratio": float(np.mean(items)),
            "std_family_shared_ratio": float(np.std(items)) if len(items) > 1 else 0.0,
        }
        for family, items in sorted(grouped.items())
    }


def _flatten_example_features(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().reshape(tensor.shape[0], -1).numpy().astype(np.float32, copy=False)


def fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    num_classes: int,
    device: str,
    epochs: int,
    lr: float,
) -> Dict[str, float]:
    if train_x.size == 0 or test_x.size == 0:
        return {"train_accuracy": 0.0, "test_accuracy": 0.0}
    model = torch.nn.Linear(train_x.shape[1], int(num_classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.0001)
    x_train = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train_y, dtype=torch.long, device=device)
    x_test = torch.as_tensor(test_x, dtype=torch.float32, device=device)
    y_test = torch.as_tensor(test_y, dtype=torch.long, device=device)
    best_state = None
    best_train = -1.0
    for _epoch in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = torch.nn.functional.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_acc = float((torch.argmax(logits, dim=1) == y_train).float().mean().detach().cpu())
        if train_acc >= best_train:
            best_train = train_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_acc = float((torch.argmax(model(x_train), dim=1) == y_train).float().mean().detach().cpu())
        test_acc = float((torch.argmax(model(x_test), dim=1) == y_test).float().mean().detach().cpu())
    return {"train_accuracy": train_acc, "test_accuracy": test_acc}


def _label_map(labels: Sequence[str]) -> tuple[np.ndarray, Dict[str, int]]:
    unique = sorted(set(str(label) for label in labels))
    mapping = {label: index for index, label in enumerate(unique)}
    return np.asarray([mapping[str(label)] for label in labels], dtype=np.int64), mapping


def _fit_family_probe(
    train_tensor: torch.Tensor,
    train_families: Sequence[str],
    test_tensor: torch.Tensor,
    test_families: Sequence[str],
    device: str,
    epochs: int,
    lr: float,
) -> Dict[str, Any]:
    train_x = _flatten_example_features(train_tensor)
    test_x = _flatten_example_features(test_tensor)
    train_y, mapping = _label_map(train_families)
    test_y = np.asarray([mapping[str(label)] for label in test_families], dtype=np.int64)
    result = fit_linear_probe(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        num_classes=len(mapping),
        device=device,
        epochs=epochs,
        lr=lr,
    )
    result["chance"] = 1.0 / max(1, len(mapping))
    result["num_families"] = len(mapping)
    return result


def _family_probe_with_cross_family_shuffle(
    train_families: Sequence[str],
    test_families: Sequence[str],
    family_means: Mapping[str, torch.Tensor],
    device: str,
    epochs: int,
    lr: float,
) -> Dict[str, Any]:
    families = sorted(family_means.keys())
    if len(families) <= 1:
        return {"train_accuracy": 0.0, "test_accuracy": 0.0, "chance": 1.0}
    train_perm = _non_identity_permutation(len(families), seed=17)
    test_perm = _non_identity_permutation(len(families), seed=29)
    train_map = {family: families[int(train_perm[index])] for index, family in enumerate(families)}
    test_map = {family: families[int(test_perm[index])] for index, family in enumerate(families)}
    train_shared = torch.stack([family_means[train_map[str(family)]] for family in train_families], dim=0)
    test_shared = torch.stack([family_means[test_map[str(family)]] for family in test_families], dim=0)
    return _fit_family_probe(train_shared, train_families, test_shared, test_families, device=device, epochs=epochs, lr=lr)


def _fit_example_probe_within_family(
    residual_tensor: torch.Tensor,
    families: Sequence[str],
    device: str,
    epochs: int,
    lr: float,
    shuffled_test_assignments: bool,
) -> Dict[str, Any]:
    grouped = _group_indices(families)
    slot_count = residual_tensor.shape[1] * residual_tensor.shape[2]
    train_slots = np.asarray([slot for slot in range(slot_count) if slot % 2 == 0], dtype=np.int64)
    test_slots = np.asarray([slot for slot in range(slot_count) if slot % 2 == 1], dtype=np.int64)
    if train_slots.size == 0 or test_slots.size == 0:
        midpoint = max(1, slot_count // 2)
        train_slots = np.arange(0, midpoint, dtype=np.int64)
        test_slots = np.arange(midpoint, slot_count, dtype=np.int64)
    family_rows: Dict[str, Dict[str, float]] = {}
    weighted_acc: List[tuple[float, int]] = []
    weighted_chance: List[tuple[float, int]] = []
    residual_np = residual_tensor.detach().cpu().numpy()
    for family, indices in sorted(grouped.items()):
        if len(indices) <= 1:
            continue
        family_residual = residual_np[np.asarray(indices, dtype=np.int64)].reshape(len(indices), slot_count, residual_np.shape[-1])
        labels = np.arange(len(indices), dtype=np.int64)
        train_x = family_residual[:, train_slots, :].reshape(len(indices) * len(train_slots), residual_np.shape[-1])
        train_y = np.repeat(labels, len(train_slots))
        test_view = family_residual.copy()
        if shuffled_test_assignments:
            perm = _non_identity_permutation(len(indices), seed=101 + len(indices))
            test_view = test_view[perm]
        test_x = test_view[:, test_slots, :].reshape(len(indices) * len(test_slots), residual_np.shape[-1])
        test_y = np.repeat(labels, len(test_slots))
        probe = fit_linear_probe(
            train_x=train_x.astype(np.float32, copy=False),
            train_y=train_y,
            test_x=test_x.astype(np.float32, copy=False),
            test_y=test_y,
            num_classes=len(indices),
            device=device,
            epochs=epochs,
            lr=lr,
        )
        chance = 1.0 / max(1, len(indices))
        family_rows[family] = {
            "n_examples": len(indices),
            "accuracy": float(probe["test_accuracy"]),
            "chance": chance,
        }
        weighted_acc.append((float(probe["test_accuracy"]), len(indices)))
        weighted_chance.append((chance, len(indices)))
    return {
        "per_family": family_rows,
        "weighted_accuracy": _weighted_mean(weighted_acc),
        "weighted_chance": _weighted_mean(weighted_chance),
    }


def _collect_duplicate_prompt_pooled(
    latent_mod,
    system,
    examples: Sequence[Any],
    avenue_index: int,
    batch_size: int,
) -> torch.Tensor:
    pooled_batches: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in _example_batches(examples, batch_size):
            role_pooled: List[torch.Tensor] = []
            for role_index in range(system.n_roles):
                texts = [
                    latent_mod._format_avenue_clone_prompt(example, role_index, avenue_index, system.message_config)  # type: ignore[attr-defined]
                    for example in batch
                ]
                readouts = system.shared_agent.forward_texts_with_readouts(
                    texts,
                    use_msg_token=system.message_config.use_msg_token,
                    msg_position=system.message_config.msg_position,
                    selected_layer_ids=system.message_config.active_message_layers,
                )
                pooled = readouts["pooled"].detach().cpu()
                role_pooled.append(torch.stack([pooled, pooled.clone()], dim=1))
            pooled_batches.append(torch.stack(role_pooled, dim=1))
    return torch.cat(pooled_batches, dim=0)


def _duplicate_prompt_residual_norm(
    latent_mod,
    modules: SimpleNamespace,
    seed_entries: Sequence[Mapping[str, Any]],
    preset: ArtifactPreset,
    device: str,
    batch_size: int,
) -> float:
    values: List[float] = []
    for entry in seed_entries:
        checkpoint_path = Path(str(entry["checkpoint_path"]))
        result = _load_checkpoint(modules, checkpoint_path, device=device)
        examples = entry["test_examples"][: min(len(entry["test_examples"]), batch_size)]
        duplicate_pooled = _collect_duplicate_prompt_pooled(
            latent_mod=modules.latent,
            system=result.system,
            examples=examples,
            avenue_index=0,
            batch_size=batch_size,
        )
        duplicate_mean = duplicate_pooled.mean(dim=2, keepdim=True)
        duplicate_residual = duplicate_pooled - duplicate_mean
        values.append(float(duplicate_residual.norm(dim=-1).mean().item()))
    return _mean(values)


def _pairwise_family_mean_cosine(family_means: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    families = sorted(family_means.keys())
    flat = {family: family_means[family].reshape(-1).float() for family in families}
    matrix: Dict[str, Dict[str, float]] = {}
    off_diag: List[float] = []
    for family_a in families:
        row: Dict[str, float] = {}
        for family_b in families:
            cosine = torch.nn.functional.cosine_similarity(
                flat[family_a].view(1, -1), flat[family_b].view(1, -1), dim=1
            ).item()
            row[family_b] = float(cosine)
            if family_a != family_b:
                off_diag.append(float(cosine))
        matrix[family_a] = row
    return {
        "families": families,
        "matrix": matrix,
        "summary": {
            "mean_off_diagonal": _mean(off_diag),
            "max_off_diagonal": max(off_diag) if off_diag else 1.0,
            "min_off_diagonal": min(off_diag) if off_diag else 1.0,
        },
    }


def _same_family_different_family_map(family_means: Mapping[str, torch.Tensor]) -> Dict[str, str]:
    families = sorted(family_means.keys())
    flat = {family: family_means[family].reshape(-1).float() for family in families}
    mapping: Dict[str, str] = {}
    for family in families:
        best_other = None
        best_score = -1e9
        for other in families:
            if other == family:
                continue
            score = float(
                torch.nn.functional.cosine_similarity(flat[family].view(1, -1), flat[other].view(1, -1), dim=1).item()
            )
            if score > best_score:
                best_score = score
                best_other = other
        mapping[family] = str(best_other) if best_other is not None else family
    return mapping


def _family_accuracy_and_correlation(
    family_ratio_rows: Mapping[str, Mapping[str, float]],
    full_accuracy_by_family: Mapping[str, float],
) -> Dict[str, Any]:
    shared_values: List[float] = []
    accuracy_values: List[float] = []
    rows: Dict[str, Dict[str, float]] = {}
    for family, ratio_info in family_ratio_rows.items():
        if family not in full_accuracy_by_family:
            continue
        shared = float(ratio_info["mean_family_shared_ratio"])
        acc = float(full_accuracy_by_family[family])
        rows[family] = {
            "mean_family_shared_ratio": shared,
            "full_accuracy": acc,
        }
        shared_values.append(shared)
        accuracy_values.append(acc)
    correlation = None
    if len(shared_values) >= 2 and np.std(shared_values) > 0.0 and np.std(accuracy_values) > 0.0:
        correlation = float(np.corrcoef(np.asarray(shared_values), np.asarray(accuracy_values))[0, 1])
    return {
        "per_family": rows,
        "pearson_correlation": correlation,
    }


def inject_vectors_into_avenue(
    token_states: torch.Tensor,
    avenue_index: int,
    injection: torch.Tensor,
) -> torch.Tensor:
    if token_states.dim() != 5:
        raise ValueError(f"expected [batch, roles, avenues, tokens, hidden], got {tuple(token_states.shape)}")
    if injection.dim() != 3:
        raise ValueError(f"expected [batch, roles, hidden], got {tuple(injection.shape)}")
    out = token_states.clone()
    out[:, :, avenue_index, :, :] = out[:, :, avenue_index, :, :] + injection.unsqueeze(2)
    return out


def _non_identity_permutation(length: int, seed: int) -> np.ndarray:
    order = np.arange(length, dtype=np.int64)
    if length <= 1:
        return order
    rng = np.random.default_rng(seed)
    perm = rng.permutation(length)
    if np.all(perm == order):
        perm = np.roll(perm, 1)
    return perm.astype(np.int64, copy=False)


def _accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(preds == labels)) if len(labels) else 0.0


def _logits_for_condition(system, examples: Sequence[Any], condition: str, seed: int, batch_size: int) -> np.ndarray:
    chunks: List[np.ndarray] = []
    system.eval()
    with torch.no_grad():
        for batch in _example_batches(examples, batch_size):
            logits, _audit, _activations = system(batch, condition=condition, seed=seed)
            chunks.append(logits.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _role_ids_for_batch(system, batch_size: int, device: torch.device) -> torch.Tensor:
    role_ids = torch.arange(system.n_roles, dtype=torch.long, device=device).view(1, system.n_roles)
    return role_ids.expand(batch_size, system.n_roles)


def _candidate_token_logits_from_readouts(
    latent_mod,
    system,
    examples: Sequence[Any],
    readouts: Mapping[str, torch.Tensor],
    role_ids: torch.Tensor,
    token_states: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    candidate_features = latent_mod._candidate_feature_tensor(examples, token_states.device)  # type: ignore[attr-defined]
    return system.coordinator(
        readouts["message"],
        role_ids,
        candidate_features,
        token_states,
        token_mask,
        readouts.get("avenue_ids"),
    )


def _mean_by_family_vector(
    families: Sequence[str],
    family_values: Mapping[str, torch.Tensor],
    avenue_index: int,
) -> torch.Tensor:
    return torch.stack([family_values[str(family)][:, avenue_index, :] for family in families], dim=0)


def _noise_matched_to_signal(signal: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=signal.device)
    generator.manual_seed(int(seed))
    noise = torch.randn(signal.shape, generator=generator, device=signal.device, dtype=signal.dtype)
    signal_norm = signal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    noise_norm = noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return noise * (signal_norm / noise_norm)


def _best_other_avenue(single_acc: Sequence[float], baseline_avenue: int) -> int:
    candidates = [(index, value) for index, value in enumerate(single_acc) if index != baseline_avenue]
    if not candidates:
        return baseline_avenue
    candidates.sort(key=lambda item: float(item[1]), reverse=True)
    return int(candidates[0][0])


def _select_alternate_family_key(
    family: str,
    example_key: str,
    family_to_keys: Mapping[str, Sequence[str]],
) -> str | None:
    keys = [key for key in family_to_keys.get(family, []) if key != example_key]
    if not keys:
        return None
    return str(keys[0])


def _run_family_hypothesis(
    modules: SimpleNamespace,
    seed_entries: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    family_means: Mapping[str, torch.Tensor],
    thresholds: FamilyThresholds,
    device: str,
    batch_size: int,
    probe_epochs: int,
    probe_lr: float,
    target_min_examples: int,
) -> Dict[str, Any]:
    combined_inventory = _family_size_inventory(aggregate["combined_families"])
    eligible_families = {
        family for family, count in combined_inventory["counts"].items() if int(count) > 1
    }

    combined_decomp = decompose_by_family(aggregate["combined_pooled"], aggregate["combined_families"], family_means)
    probe_split = _split_tensor_and_labels_within_groups(
        combined_decomp["shared"],
        aggregate["combined_families"],
        seed=11,
    )
    residual_probe_split = _split_tensor_and_labels_within_groups(
        combined_decomp["residual"],
        aggregate["combined_families"],
        seed=23,
    )

    family_probe = _fit_family_probe(
        probe_split["train_tensor"],
        probe_split["train_labels"],
        probe_split["test_tensor"],
        probe_split["test_labels"],
        device=device,
        epochs=probe_epochs,
        lr=probe_lr,
    )
    residual_family_probe = _fit_family_probe(
        residual_probe_split["train_tensor"],
        residual_probe_split["train_labels"],
        residual_probe_split["test_tensor"],
        residual_probe_split["test_labels"],
        device=device,
        epochs=probe_epochs,
        lr=probe_lr,
    )
    example_probe = _fit_example_probe_within_family(
        combined_decomp["residual"],
        aggregate["combined_families"],
        device=device,
        epochs=probe_epochs,
        lr=probe_lr,
        shuffled_test_assignments=False,
    )
    shuffled_example_probe = _fit_example_probe_within_family(
        combined_decomp["residual"],
        aggregate["combined_families"],
        device=device,
        epochs=probe_epochs,
        lr=probe_lr,
        shuffled_test_assignments=True,
    )
    cross_family_shuffle_probe = _family_probe_with_cross_family_shuffle(
        probe_split["train_labels"],
        probe_split["test_labels"],
        family_means,
        device=device,
        epochs=probe_epochs,
        lr=probe_lr,
    )
    family_ratio_rows = _family_ratio_rows(
        combined_decomp["shared_ratio"],
        aggregate["combined_families"],
        eligible_families,
    )
    mean_family_shared_ratio = _mean(
        [float(row["mean_family_shared_ratio"]) for row in family_ratio_rows.values()]
    )
    duplicate_prompt_residual_norm = _duplicate_prompt_residual_norm(
        latent_mod=modules.latent,
        modules=modules,
        seed_entries=seed_entries,
        preset=_artifact_presets()["stage5_final"] if seed_entries else _artifact_presets()["stage5_final"],
        device=device,
        batch_size=batch_size,
    )
    pairwise_cosine = _pairwise_family_mean_cosine(family_means)

    full_accuracy_by_family, subexperiment_b_seed_rows = _run_family_subexperiment_b(
        modules=modules,
        seed_entries=seed_entries,
        family_means=family_means,
        combined_decomp=combined_decomp,
        aggregate=aggregate,
        device=device,
        batch_size=batch_size,
        target_min_examples=target_min_examples,
    )
    family_accuracy_correlation = _family_accuracy_and_correlation(family_ratio_rows, full_accuracy_by_family)
    subexperiment_b_summary = _summarize_family_subexperiment_b(
        rows=subexperiment_b_seed_rows,
        thresholds=thresholds,
    )

    family_component_exists = bool(mean_family_shared_ratio > thresholds.family_shared_ratio_min)
    family_probe_above = bool(float(family_probe["test_accuracy"]) > thresholds.family_probe_min)
    residual_probe_near_chance = bool(float(residual_family_probe["test_accuracy"]) < thresholds.residual_family_probe_max)
    duplicate_norm_ok = bool(duplicate_prompt_residual_norm <= thresholds.duplicate_prompt_residual_norm_max)
    decomposition_clean = bool(
        family_component_exists and family_probe_above and residual_probe_near_chance and duplicate_norm_ok
    )
    decomposition_status = "YES" if decomposition_clean else "CONFOUNDED"

    subexperiment_a = {
        "thresholds": {
            "family_shared_ratio_min": thresholds.family_shared_ratio_min,
            "family_probe_min": thresholds.family_probe_min,
            "residual_family_probe_max": thresholds.residual_family_probe_max,
            "duplicate_prompt_residual_norm_max": thresholds.duplicate_prompt_residual_norm_max,
        },
        "family_size_inventory": combined_inventory,
        "pairwise_family_mean_cosine": pairwise_cosine,
        "family_shared_ratio_by_family": family_ratio_rows,
        "family_accuracy_correlation": family_accuracy_correlation,
        "summary": {
            "family_shared_component_exists": family_component_exists,
            "family_probe_accuracy_above_070": family_probe_above,
            "cross_family_probe_on_residuals_near_chance": residual_probe_near_chance,
            "decomposition_clean": decomposition_clean,
            "decomposition_status": decomposition_status,
            "mean_family_shared_ratio": mean_family_shared_ratio,
            "family_probe_accuracy": float(family_probe["test_accuracy"]),
            "family_probe_train_accuracy": float(family_probe["train_accuracy"]),
            "family_probe_chance": float(family_probe["chance"]),
            "cross_family_probe_on_residuals": float(residual_family_probe["test_accuracy"]),
            "cross_family_probe_on_residuals_train_accuracy": float(residual_family_probe["train_accuracy"]),
            "cross_family_probe_on_residuals_chance": float(residual_family_probe["chance"]),
            "example_probe_within_family": float(example_probe["weighted_accuracy"]),
            "example_probe_within_family_chance": float(example_probe["weighted_chance"]),
            "within_family_shuffle_example_probe": float(shuffled_example_probe["weighted_accuracy"]),
            "cross_family_shuffle_probe_accuracy": float(cross_family_shuffle_probe["test_accuracy"]),
            "duplicate_prompt_residual_norm": float(duplicate_prompt_residual_norm),
            "probe_split_groups": {
                "included_families": probe_split["included_groups"],
                "excluded_singletons": probe_split["excluded_groups"],
            },
            "seed_coverage": aggregate["seed_contributions"],
        },
        "controls": {
            "within_family_shuffle_example_probe": shuffled_example_probe,
            "cross_family_shuffle_probe": cross_family_shuffle_probe,
        },
    }

    overall_verdict = "YES" if decomposition_clean and bool(subexperiment_b_summary["success"]) else ("PARTIAL" if family_component_exists else "NO")
    meaning = (
        "The representation is organized around reusable family structure rather than avenue-specific residuals."
        if family_component_exists
        else "The redesigned family-shared decomposition did not clear the structural gate."
    )
    next_step = (
        "Design a minimal training experiment that explicitly distills family-shared summaries across clones."
        if decomposition_clean and bool(subexperiment_b_summary["success"])
        else "Resolve the family confound or strengthen the family-mean intervention before making a communication-primitive claim."
    )

    return {
        "subexperiment_a": subexperiment_a,
        "subexperiment_b": {
            "seed_rows": subexperiment_b_seed_rows,
            "summary": subexperiment_b_summary,
        },
        "overall_verdict": {
            "redesigned_hypothesis_supported": overall_verdict,
            "meaning_one_sentence": meaning,
            "recommended_next_step": next_step,
        },
    }


def _run_family_subexperiment_b(
    modules: SimpleNamespace,
    seed_entries: Sequence[Mapping[str, Any]],
    family_means: Mapping[str, torch.Tensor],
    combined_decomp: Mapping[str, torch.Tensor],
    aggregate: Mapping[str, Any],
    device: str,
    batch_size: int,
    target_min_examples: int,
) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    residual_lookup = {
        key: combined_decomp["residual"][index].detach().cpu()
        for index, key in enumerate(aggregate["combined_keys"])
    }
    family_lookup = {key: aggregate["combined_families"][index] for index, key in enumerate(aggregate["combined_keys"])}
    family_to_keys = _group_keys_by_family(aggregate["combined_keys"], aggregate["combined_families"])
    different_family_map = _same_family_different_family_map(family_means)
    full_accuracy_by_family_values: Dict[str, List[float]] = {}
    rows: List[Dict[str, Any]] = []

    for entry in seed_entries:
        checkpoint_path = Path(str(entry["checkpoint_path"]))
        result = _load_checkpoint(modules, checkpoint_path, device=device)
        system = result.system
        if not bool(getattr(system.coordinator, "requires_token_states", False)):
            rows.append(
                {
                    "seed": int(entry["seed"]),
                    "available": False,
                    "reason": "selected checkpoint does not use candidate-token coordinator; no token-state intervention path available",
                }
            )
            continue

        seed = int(entry["seed"])
        test_examples = list(entry["test_examples"])
        labels = _labels(test_examples)
        families = list(entry["test_families"])
        keys = list(entry["test_keys"])
        full_logits = _logits_for_condition(system, test_examples, condition="none", seed=seed, batch_size=batch_size)
        full_preds = np.argmax(full_logits, axis=1).astype(np.int64)
        for family, label, pred in zip(families, labels.tolist(), full_preds.tolist()):
            full_accuracy_by_family_values.setdefault(str(family), []).append(float(int(pred == label)))

        avenue_count = int(system.message_config.num_avenues)
        single_logits = np.stack(
            [
                _logits_for_condition(
                    system, test_examples, condition=f"avenue_only_{avenue_index}", seed=seed, batch_size=batch_size
                )
                for avenue_index in range(avenue_count)
            ],
            axis=1,
        )
        single_preds = np.argmax(single_logits, axis=2).astype(np.int64)
        single_acc = [_accuracy(single_preds[:, avenue_index], labels) for avenue_index in range(avenue_count)]
        baseline_avenue = int(np.argmax(np.asarray(single_acc, dtype=np.float64)))
        baseline_preds = single_preds[:, baseline_avenue]
        target_mask = (full_preds == labels) & (baseline_preds != labels)
        target_indices = np.flatnonzero(target_mask)
        if target_indices.size < max(1, int(target_min_examples)):
            rows.append(
                {
                    "seed": seed,
                    "available": False,
                    "baseline_avenue": baseline_avenue,
                    "baseline_avenue_accuracy": float(single_acc[baseline_avenue]),
                    "single_avenue_accuracy": {str(index): float(value) for index, value in enumerate(single_acc)},
                    "target_examples": int(target_indices.size),
                    "target_fraction": float(target_indices.size / max(1, len(test_examples))),
                    "reason": f"target set too small for the requested intervention screen (need >= {target_min_examples})",
                }
            )
            continue

        alt_avenue = _best_other_avenue(single_acc, baseline_avenue)
        selected_examples = [test_examples[index] for index in target_indices.tolist()]
        selected_labels = labels[target_indices]
        selected_families = [families[index] for index in target_indices.tolist()]
        selected_keys = [keys[index] for index in target_indices.tolist()]

        condition_logits = {
            "baseline": [],
            "same_family_same_avenue": [],
            "same_family_different_avenue": [],
            "different_family": [],
            "example_residual_only": [],
            "within_family_shuffled": [],
            "noise": [],
        }
        system.eval()
        with torch.no_grad():
            for batch_start in range(0, len(selected_examples), batch_size):
                batch = selected_examples[batch_start : batch_start + batch_size]
                batch_labels = selected_labels[batch_start : batch_start + batch_size]
                batch_families = selected_families[batch_start : batch_start + batch_size]
                batch_keys = selected_keys[batch_start : batch_start + batch_size]
                device_obj = next(system.parameters()).device
                role_ids = _role_ids_for_batch(system, len(batch), device_obj)
                readouts = system.collect_clone_representations(batch, condition="none", seed=seed, role_ids=role_ids)
                token_states = readouts["token_states"]
                token_mask = readouts["token_mask"].bool()
                baseline_mask = torch.zeros_like(token_mask, dtype=torch.bool)
                baseline_mask[:, :, baseline_avenue, :] = token_mask[:, :, baseline_avenue, :]

                same_family_same_vec = _mean_by_family_vector(batch_families, family_means, baseline_avenue).to(device_obj)
                same_family_diff_vec = _mean_by_family_vector(batch_families, family_means, alt_avenue).to(device_obj)
                diff_families = [different_family_map[str(family)] for family in batch_families]
                different_family_vec = _mean_by_family_vector(diff_families, family_means, baseline_avenue).to(device_obj)
                example_residual_vec = torch.stack(
                    [residual_lookup[str(key)][:, baseline_avenue, :] for key in batch_keys], dim=0
                ).to(device_obj)

                shuffled_vectors: List[torch.Tensor] = []
                for family, key in zip(batch_families, batch_keys):
                    alternate_key = _select_alternate_family_key(str(family), str(key), family_to_keys)
                    if alternate_key is None:
                        shuffled_vectors.append(torch.zeros_like(example_residual_vec[0]).cpu())
                    else:
                        shuffled_vectors.append(residual_lookup[alternate_key][:, baseline_avenue, :])
                shuffled_residual_vec = torch.stack(shuffled_vectors, dim=0).to(device_obj)
                within_family_shuffled_vec = same_family_same_vec + shuffled_residual_vec
                noise_vec = _noise_matched_to_signal(same_family_same_vec, seed=seed + batch_start + 97)

                baseline_logits = _candidate_token_logits_from_readouts(
                    modules.latent, system, batch, readouts, role_ids, token_states, baseline_mask
                )
                same_family_same_logits = _candidate_token_logits_from_readouts(
                    modules.latent,
                    system,
                    batch,
                    readouts,
                    role_ids,
                    inject_vectors_into_avenue(token_states, baseline_avenue, same_family_same_vec),
                    baseline_mask,
                )
                same_family_diff_logits = _candidate_token_logits_from_readouts(
                    modules.latent,
                    system,
                    batch,
                    readouts,
                    role_ids,
                    inject_vectors_into_avenue(token_states, baseline_avenue, same_family_diff_vec),
                    baseline_mask,
                )
                different_family_logits = _candidate_token_logits_from_readouts(
                    modules.latent,
                    system,
                    batch,
                    readouts,
                    role_ids,
                    inject_vectors_into_avenue(token_states, baseline_avenue, different_family_vec),
                    baseline_mask,
                )
                example_residual_logits = _candidate_token_logits_from_readouts(
                    modules.latent,
                    system,
                    batch,
                    readouts,
                    role_ids,
                    inject_vectors_into_avenue(token_states, baseline_avenue, example_residual_vec),
                    baseline_mask,
                )
                within_family_shuffled_logits = _candidate_token_logits_from_readouts(
                    modules.latent,
                    system,
                    batch,
                    readouts,
                    role_ids,
                    inject_vectors_into_avenue(token_states, baseline_avenue, within_family_shuffled_vec),
                    baseline_mask,
                )
                noise_logits = _candidate_token_logits_from_readouts(
                    modules.latent,
                    system,
                    batch,
                    readouts,
                    role_ids,
                    inject_vectors_into_avenue(token_states, baseline_avenue, noise_vec),
                    baseline_mask,
                )

                condition_logits["baseline"].append(baseline_logits.detach().cpu().numpy())
                condition_logits["same_family_same_avenue"].append(same_family_same_logits.detach().cpu().numpy())
                condition_logits["same_family_different_avenue"].append(same_family_diff_logits.detach().cpu().numpy())
                condition_logits["different_family"].append(different_family_logits.detach().cpu().numpy())
                condition_logits["example_residual_only"].append(example_residual_logits.detach().cpu().numpy())
                condition_logits["within_family_shuffled"].append(within_family_shuffled_logits.detach().cpu().numpy())
                condition_logits["noise"].append(noise_logits.detach().cpu().numpy())

        accuracy_by_condition: Dict[str, float] = {}
        for name, chunks in condition_logits.items():
            logits = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
            preds = np.argmax(logits, axis=1).astype(np.int64)
            accuracy_by_condition[name] = _accuracy(preds, selected_labels)

        rows.append(
            {
                "seed": seed,
                "available": True,
                "baseline_avenue": baseline_avenue,
                "different_avenue": alt_avenue,
                "baseline_avenue_accuracy": float(single_acc[baseline_avenue]),
                "single_avenue_accuracy": {str(index): float(value) for index, value in enumerate(single_acc)},
                "target_examples": int(target_indices.size),
                "target_fraction": float(target_indices.size / max(1, len(test_examples))),
                "target_family_labels": sorted(set(selected_families)),
                "accuracy_by_condition": accuracy_by_condition,
                "comparisons": {
                    "same_family_beats_different_family": bool(
                        accuracy_by_condition["same_family_same_avenue"] > accuracy_by_condition["different_family"]
                    ),
                    "same_family_alignment_within_005": bool(
                        abs(
                            accuracy_by_condition["same_family_same_avenue"]
                            - accuracy_by_condition["same_family_different_avenue"]
                        )
                        <= 0.05
                    ),
                    "example_residual_weaker_than_family_mean": bool(
                        accuracy_by_condition["example_residual_only"] < accuracy_by_condition["same_family_same_avenue"]
                    ),
                    "within_family_shuffle_matches_family_mean": bool(
                        abs(
                            accuracy_by_condition["within_family_shuffled"]
                            - accuracy_by_condition["same_family_same_avenue"]
                        )
                        <= 0.05
                    ),
                    "noise_does_not_help": bool(
                        accuracy_by_condition["noise"] <= accuracy_by_condition["baseline"] + 1e-12
                    ),
                },
            }
        )

    full_accuracy_by_family = {
        family: _mean(values) for family, values in sorted(full_accuracy_by_family_values.items())
    }
    return full_accuracy_by_family, rows


def _group_keys_by_family(keys: Sequence[str], families: Sequence[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for key, family in zip(keys, families):
        grouped.setdefault(str(family), []).append(str(key))
    return grouped


def _summarize_family_subexperiment_b(
    rows: Sequence[Mapping[str, Any]],
    thresholds: FamilyThresholds,
) -> Dict[str, Any]:
    available_rows = [row for row in rows if row.get("available")]
    target_seed_count = len(available_rows)
    if not available_rows:
        return {
            "target_set_non_empty": False,
            "target_seed_count": 0,
            "success": False,
            "reason": "No seed had a large enough target set.",
        }
    keys = list(available_rows[0]["accuracy_by_condition"].keys())
    mean_accuracy = {key: _mean([float(row["accuracy_by_condition"][key]) for row in available_rows]) for key in keys}
    weighted_accuracy = {
        key: _weighted_mean([(float(row["accuracy_by_condition"][key]), int(row["target_examples"])) for row in available_rows])
        for key in keys
    }
    comparison_flags = {
        "same_family_beats_different_family": all(bool(row["comparisons"]["same_family_beats_different_family"]) for row in available_rows),
        "same_family_same_avenue_approx_same_family_different_avenue": all(
            bool(row["comparisons"]["same_family_alignment_within_005"]) for row in available_rows
        ),
        "example_residual_weaker_than_family_mean": all(
            bool(row["comparisons"]["example_residual_weaker_than_family_mean"]) for row in available_rows
        ),
        "within_family_shuffle_matches_family_mean": all(
            bool(row["comparisons"]["within_family_shuffle_matches_family_mean"]) for row in available_rows
        ),
        "noise_does_not_help": all(bool(row["comparisons"]["noise_does_not_help"]) for row in available_rows),
    }
    success = bool(
        target_seed_count >= thresholds.min_target_seed_count
        and comparison_flags["same_family_beats_different_family"]
        and comparison_flags["same_family_same_avenue_approx_same_family_different_avenue"]
        and comparison_flags["example_residual_weaker_than_family_mean"]
        and comparison_flags["within_family_shuffle_matches_family_mean"]
        and comparison_flags["noise_does_not_help"]
    )
    return {
        "target_set_non_empty": True,
        "target_seed_count": target_seed_count,
        "mean_target_examples": _mean([float(row["target_examples"]) for row in available_rows]),
        "mean_accuracy_by_condition": mean_accuracy,
        "weighted_accuracy_by_condition": weighted_accuracy,
        "same_family_beats_different_family": comparison_flags["same_family_beats_different_family"],
        "same_family_same_avenue_approx_same_family_different_avenue": comparison_flags[
            "same_family_same_avenue_approx_same_family_different_avenue"
        ],
        "example_residual_weaker_than_family_mean": comparison_flags["example_residual_weaker_than_family_mean"],
        "within_family_shuffle_matches_family_mean": comparison_flags["within_family_shuffle_matches_family_mean"],
        "noise_does_not_help": comparison_flags["noise_does_not_help"],
        "controls_passed": bool(
            comparison_flags["same_family_beats_different_family"]
            and comparison_flags["within_family_shuffle_matches_family_mean"]
            and comparison_flags["noise_does_not_help"]
        ),
        "success": success,
    }


def _render_prior_avenue_section(prior: Mapping[str, Any] | None) -> List[str]:
    if not prior:
        return ["## Prior Hypothesis: Avenue Residual", "", "- No prior avenue-residual result was available to preserve.", ""]
    sub_a = prior.get("subexperiment_a", {}).get("summary", {})
    sub_b = prior.get("subexperiment_b", {}).get("summary", {})
    return [
        "## Prior Hypothesis: Avenue Residual",
        "",
        f"- Mean shared ratio: `{float(sub_a.get('mean_shared_ratio', 0.0)):.4f}`",
        f"- Avenue probe accuracy: `{float(sub_a.get('mean_avenue_probe_accuracy', 0.0)):.4f}`",
        f"- Family-centred avenue probe accuracy: `{float(sub_a.get('mean_avenue_probe_accuracy_family_centered', 0.0)):.4f}`",
        f"- Shuffled avenue probe accuracy: `{float(sub_a.get('mean_shuffled_avenue_probe_accuracy', 0.0)):.4f}`",
        f"- Duplicate prompt residual norm: `{float(sub_a.get('mean_duplicate_prompt_residual_norm', 0.0)):.8f}`",
        (
            f"- Avenue Sub-experiment B weighted residual vs wrong avenue: "
            f"`{float(sub_b.get('weighted_accuracy_by_condition', {}).get('residual_injection', 0.0)):.4f}` vs "
            f"`{float(sub_b.get('weighted_accuracy_by_condition', {}).get('wrong_avenue_residual', 0.0)):.4f}`"
            if sub_b.get("weighted_accuracy_by_condition")
            else "- Avenue Sub-experiment B was not available."
        ),
        "- Contrast: the avenue-residual hypothesis showed a family confound and did not identify a clean avenue-specific communication primitive.",
        "",
    ]


def _render_distribution_summary(distribution: Mapping[str, Any], top_n: int = 4) -> str:
    counts = distribution.get("counts", {})
    shares = distribution.get("shares", {})
    ordered = sorted(
        counts.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )
    items = []
    for name, count in ordered[:top_n]:
        share = float(shares.get(str(name), shares.get(name, 0.0)))
        items.append(f"{name}:{int(count)} ({share:.2f})")
    return ", ".join(items) if items else "none"


def _render_internal_structure_section(section: Mapping[str, Any] | None) -> List[str]:
    if not section:
        return []
    clustering = section.get("clustering_summary", {})
    stability = section.get("stability", {})
    difficulty = section.get("difficulty_confound_check", {})
    target = section.get("target_set_alignment", {})
    selected = section.get("selected_clustering", {})
    cluster_rows = selected.get("cluster_characterization", {})
    lines = [
        "## INTERNAL STRUCTURE DISCOVERY",
        "",
        "Clustering summary:",
        "",
        f"- Optimal K: `{clustering.get('optimal_k')}`",
        f"- Method that produced most stable clusters: `{clustering.get('method_that_produced_most_stable_clusters')}`",
        f"- Selected split agreement with PCA K-means: `{float(clustering.get('selected_split_agreement_with_pca', 0.0)):.4f}`",
        f"- Raw vs PCA agreement by K: `{clustering.get('raw_vs_pca_agreement_by_k', {})}`",
        f"- Cluster sizes: `{clustering.get('cluster_sizes')}`",
        f"- Pairwise cluster mean cosine similarities: `{clustering.get('pairwise_cluster_mean_cosine_summary')}`",
        "",
        "Per-cluster characterisation:",
        "",
    ]
    for cluster_name, info in sorted(cluster_rows.items(), key=lambda item: int(item[0])):
        accuracy = info.get("accuracy_profile", {})
        lines.append(
            f"- Cluster {cluster_name}: size `{int(info.get('example_seed_count', 0))}` example-seed cases / "
            f"`{int(info.get('row_count', 0))}` rows, families `{_render_distribution_summary(info.get('external_family_distribution', {}))}`, "
            f"avenues `{_render_distribution_summary(info.get('avenue_distribution', {}))}`, "
            f"full `{float(accuracy.get('mean_full_model_accuracy', 0.0)):.4f}`, "
            f"best single `{float(accuracy.get('mean_best_single_avenue_accuracy', 0.0)):.4f}`, "
            f"gap `{float(accuracy.get('mean_gap', 0.0)):.4f}`, difficulty `{info.get('difficulty_profile', 'unknown')}`"
        )
    lines.extend(
        [
            "",
            "Stability:",
            "",
            f"- Mean per-example cluster stability across seeds: `{float(stability.get('mean_per_example_cluster_stability', 0.0)):.4f}`",
            f"- Stable (above 0.80): `{'YES' if stability.get('stable_above_080') else 'NO'}`",
            "",
            "Difficulty confound check:",
            "",
            f"- Correlation of cluster identity with accuracy: `full={float(difficulty.get('correlation_ratio_full_accuracy', 0.0)):.4f}`, "
            f"`best_single={float(difficulty.get('correlation_ratio_best_single_accuracy', 0.0)):.4f}`, "
            f"`gap={float(difficulty.get('correlation_ratio_gap', 0.0)):.4f}`",
            f"- Is clustering purely difficulty-stratified: `{'YES' if difficulty.get('is_purely_difficulty_stratified') else 'NO'}`",
            "",
            "Target set alignment:",
            "",
            f"- Target set examples concentrate in specific clusters: `{'YES' if target.get('concentrates_in_specific_clusters') else 'NO'}`",
            f"- If yes, which clusters: `{target.get('concentrated_clusters', [])}`",
            "",
            "Interpretation:",
            "",
            f"- What do the clusters appear to represent: {section.get('interpretation', {}).get('what_clusters_appear_to_represent', 'n/a')}",
            f"- Are they a viable unit of analysis for the communication hypothesis: `{section.get('interpretation', {}).get('viable_unit_of_analysis_for_communication_hypothesis', 'UNCLEAR')}`",
            "",
            f"Recommended next step: {section.get('interpretation', {}).get('recommended_next_step', 'n/a')}",
            "",
        ]
    )
    return lines


def _render_report(result: Mapping[str, Any]) -> str:
    metadata = result["metadata"]
    family = result["family_hypothesis"]
    sub_a = family["subexperiment_a"]
    sub_a_summary = sub_a["summary"]
    sub_b = family["subexperiment_b"]
    sub_b_summary = sub_b["summary"]
    lines = [
        "# Stage 12 Make-Or-Break",
        "",
        "## Source",
        "",
        f"- Requested preset: `{metadata['requested_preset']}`",
        f"- Actual preset: `{metadata['actual_preset']}`",
        f"- Device: `{metadata['device']}`",
        "",
    ]
    lines.extend(_render_prior_avenue_section(result.get("prior_hypothesis_avenue")))
    lines.extend(
        [
            "## SUB-EXPERIMENT A — FAMILY DECOMPOSITION",
            "",
            f"- Family shared component exists: `{'YES' if sub_a_summary['family_shared_component_exists'] else 'NO'}`",
            f"- Family probe accuracy above 0.70: `{'YES' if sub_a_summary['family_probe_accuracy_above_070'] else 'NO'}`",
            f"- Cross-family probe on residuals near chance: `{'YES' if sub_a_summary['cross_family_probe_on_residuals_near_chance'] else 'NO'}`",
            f"- Decomposition is clean: `{'YES' if sub_a_summary['decomposition_clean'] else 'NO / CONFOUNDED'}`",
            "",
            f"- Mean family shared ratio: `{float(sub_a_summary['mean_family_shared_ratio']):.4f}`",
            f"- Family probe accuracy: `{float(sub_a_summary['family_probe_accuracy']):.4f}`",
            f"- Cross-family probe on residuals: `{float(sub_a_summary['cross_family_probe_on_residuals']):.4f}`",
            f"- Example probe within family: `{float(sub_a_summary['example_probe_within_family']):.4f}`",
            f"- Example probe chance within family: `{float(sub_a_summary['example_probe_within_family_chance']):.4f}`",
            f"- Within-family shuffle example probe: `{float(sub_a_summary['within_family_shuffle_example_probe']):.4f}`",
            f"- Cross-family shuffle probe accuracy: `{float(sub_a_summary['cross_family_shuffle_probe_accuracy']):.4f}`",
            f"- Duplicate prompt residual norm: `{float(sub_a_summary['duplicate_prompt_residual_norm']):.8f}`",
            "",
            "Family size distribution:",
            "",
        ]
    )
    for family_name, count in sub_a["family_size_inventory"]["counts"].items():
        flag = " <10" if family_name in sub_a["family_size_inventory"]["families_below_10"] else ""
        lines.append(f"- {family_name}: `{int(count)}`{flag}")
    lines.extend(
        [
            "",
            f"- Families below 10 examples: `{sub_a['family_size_inventory']['families_below_10']}`",
            f"- Single-example families excluded from ratio calculation: `{sub_a['family_size_inventory']['single_example_families']}`",
            f"- Pairwise family-mean cosine mean/max/min off diagonal: `{sub_a['pairwise_family_mean_cosine']['summary']}`",
            f"- Shared-ratio vs full-accuracy correlation: `{sub_a['family_accuracy_correlation']['pearson_correlation']}`",
            "",
            "## SUB-EXPERIMENT B — FAMILY RESIDUAL INJECTION",
            "",
            f"- Target set non-empty: `{'YES' if sub_b_summary.get('target_set_non_empty') else 'NO'}`",
            f"- Same family injection beats different family injection: `{'YES' if sub_b_summary.get('same_family_beats_different_family') else 'NO'}`",
            f"- Same family same avenue ≈ same family different avenue: `{'YES' if sub_b_summary.get('same_family_same_avenue_approx_same_family_different_avenue') else 'NO'}`",
            f"- Example residual alone weaker than family mean: `{'YES' if sub_b_summary.get('example_residual_weaker_than_family_mean') else 'NO'}`",
            f"- Within-family shuffle ≈ same family injection: `{'YES' if sub_b_summary.get('within_family_shuffle_matches_family_mean') else 'NO'}`",
            f"- Controls passed: `{'YES' if sub_b_summary.get('controls_passed') else 'NO'}`",
            "",
            "Target set size per seed:",
            "",
        ]
    )
    for row in sub_b["seed_rows"]:
        lines.append(f"- Seed `{int(row['seed'])}`: `{int(row.get('target_examples', 0))}`")
    if sub_b_summary.get("mean_accuracy_by_condition"):
        lines.extend(["", "| condition | mean acc | weighted acc |", "|---|---:|---:|"])
        for name, value in sub_b_summary["mean_accuracy_by_condition"].items():
            lines.append(
                f"| {name} | {float(value):.4f} | {float(sub_b_summary['weighted_accuracy_by_condition'][name]):.4f} |"
            )
        lines.extend(["", "| seed | baseline | same family same avenue | same family different avenue | different family | example residual only | within-family shuffled | noise |", "|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for row in [seed_row for seed_row in sub_b["seed_rows"] if seed_row.get("available")]:
            acc = row["accuracy_by_condition"]
            lines.append(
                f"| {int(row['seed'])} | {float(acc['baseline']):.4f} | {float(acc['same_family_same_avenue']):.4f} | {float(acc['same_family_different_avenue']):.4f} | {float(acc['different_family']):.4f} | {float(acc['example_residual_only']):.4f} | {float(acc['within_family_shuffled']):.4f} | {float(acc['noise']):.4f} |"
            )
    lines.extend(
        [
            "",
            "## OVERALL VERDICT",
            "",
            f"- Redesigned hypothesis supported: `{family['overall_verdict']['redesigned_hypothesis_supported']}`",
            f"- What the result means in one sentence: {family['overall_verdict']['meaning_one_sentence']}",
            f"- Recommended next step: {family['overall_verdict']['recommended_next_step']}",
            "",
        ]
    )
    lines.extend(_render_internal_structure_section(result.get("internal_structure_discovery")))
    return "\n".join(lines)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _weighted_mean(pairs: Sequence[tuple[float, int]]) -> float:
    total_weight = sum(weight for _value, weight in pairs)
    if total_weight <= 0:
        return 0.0
    return float(sum(value * weight for value, weight in pairs) / total_weight)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
