from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "src"
    / "experiments"
    / "run_stage12_make_or_break.py"
)
SPEC = importlib.util.spec_from_file_location("stage12_make_or_break", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_compute_family_means_and_decomposition_match_manual_values() -> None:
    representations = torch.tensor(
        [
            [[[2.0, 0.0], [0.0, 2.0]]],
            [[[4.0, 0.0], [2.0, 2.0]]],
            [[[10.0, 0.0], [6.0, 2.0]]],
        ],
        dtype=torch.float32,
    )
    families = ["alpha", "alpha", "beta"]
    family_means = MODULE.compute_family_means(representations, families)
    assert torch.allclose(family_means["alpha"], torch.tensor([[[3.0, 0.0], [1.0, 2.0]]]))
    decomp = MODULE.decompose_by_family(representations, families, family_means)
    assert torch.allclose(decomp["shared"][0], family_means["alpha"])
    assert torch.allclose(decomp["residual"][0], torch.tensor([[[-1.0, 0.0], [-1.0, 0.0]]]))
    assert decomp["shared_ratio"].shape == (3,)
    assert torch.all(decomp["shared_ratio"] > 0.0)


def test_inject_vectors_into_avenue_only_changes_target_avenue() -> None:
    token_states = torch.zeros(2, 3, 4, 5, 6, dtype=torch.float32)
    injection = torch.arange(2 * 3 * 6, dtype=torch.float32).reshape(2, 3, 6)
    out = MODULE.inject_vectors_into_avenue(token_states, avenue_index=2, injection=injection)
    assert torch.allclose(out[:, :, 1], torch.zeros_like(out[:, :, 1]))
    expected = injection.unsqueeze(2).expand(-1, -1, 5, -1)
    assert torch.allclose(out[:, :, 2], expected)


def test_family_size_inventory_flags_small_and_singleton_families() -> None:
    inventory = MODULE._family_size_inventory(["a", "a", "b", "c", "c", "c"])
    assert inventory["counts"] == {"a": 2, "b": 1, "c": 3}
    assert inventory["single_example_families"] == ["b"]
    assert inventory["families_below_10"] == ["a", "b", "c"]


def test_group_split_keeps_each_non_singleton_family_in_both_probe_sides() -> None:
    labels = ["alpha", "alpha", "beta", "beta", "beta", "gamma"]
    split = MODULE._split_indices_within_groups(labels, seed=7)
    train_labels = [labels[index] for index in split["train_indices"].tolist()]
    test_labels = [labels[index] for index in split["test_indices"].tolist()]
    assert set(train_labels) == {"alpha", "beta"}
    assert set(test_labels) == {"alpha", "beta"}
    assert split["excluded_groups"] == ["gamma"]
