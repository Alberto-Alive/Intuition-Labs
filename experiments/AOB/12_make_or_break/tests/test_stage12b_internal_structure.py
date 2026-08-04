from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "src"
    / "experiments"
    / "run_stage12b_internal_structure.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("stage12b_internal_structure", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_select_elbow_k_prefers_visible_knee() -> None:
    inertias = {2: 100.0, 3: 60.0, 4: 45.0, 5: 40.0, 6: 38.0}
    assert MODULE._select_elbow_k(inertias) == 3


def test_modal_cluster_entries_and_stability_aggregate_by_example_seed() -> None:
    labels = np.asarray([0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    metadata = [
        {"example_id": "ex1", "example_seed_id": "30:ex1", "seed": 30, "family": "f0", "full_acc": 1, "best_single_acc": 0, "full_minus_best_single": 1, "target_set_member": True},
        {"example_id": "ex1", "example_seed_id": "30:ex1", "seed": 30, "family": "f0", "full_acc": 1, "best_single_acc": 0, "full_minus_best_single": 1, "target_set_member": True},
        {"example_id": "ex1", "example_seed_id": "31:ex1", "seed": 31, "family": "f0", "full_acc": 1, "best_single_acc": 0, "full_minus_best_single": 1, "target_set_member": True},
        {"example_id": "ex1", "example_seed_id": "31:ex1", "seed": 31, "family": "f0", "full_acc": 1, "best_single_acc": 0, "full_minus_best_single": 1, "target_set_member": True},
        {"example_id": "ex2", "example_seed_id": "30:ex2", "seed": 30, "family": "f1", "full_acc": 0, "best_single_acc": 0, "full_minus_best_single": 0, "target_set_member": False},
        {"example_id": "ex2", "example_seed_id": "30:ex2", "seed": 30, "family": "f1", "full_acc": 0, "best_single_acc": 0, "full_minus_best_single": 0, "target_set_member": False},
        {"example_id": "ex2", "example_seed_id": "31:ex2", "seed": 31, "family": "f1", "full_acc": 0, "best_single_acc": 0, "full_minus_best_single": 0, "target_set_member": False},
        {"example_id": "ex2", "example_seed_id": "31:ex2", "seed": 31, "family": "f1", "full_acc": 0, "best_single_acc": 0, "full_minus_best_single": 0, "target_set_member": False},
    ]
    modal = MODULE._modal_cluster_entries(labels, metadata)
    assert [row["cluster"] for row in modal] == [0, 1, 1, 1]
    stability = MODULE._cluster_stability_summary(modal)
    assert stability["eligible_example_count"] == 2
    assert stability["per_example_cluster_stability"]["ex1"]["stability"] == 0.5
    assert stability["per_example_cluster_stability"]["ex2"]["stability"] == 1.0


def test_correlation_ratio_detects_cluster_accuracy_association() -> None:
    categories = np.asarray([0, 0, 1, 1], dtype=np.int64)
    values = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    eta = MODULE._correlation_ratio(categories, values)
    assert eta > 0.99
