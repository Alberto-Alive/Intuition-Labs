"""Invariant tests for the partitioned inference-time bank in E20."""

from __future__ import annotations

import math

import pytest
import torch

from experiments.DIGIT.Extrapolation.e19.extrapolation.data import LEVEL_NOVEL
from experiments.DIGIT.Extrapolation.e19.extrapolation.experiment import (
    E19Config,
    train_single_seed_artifacts,
)
from experiments.DIGIT.Extrapolation.e20.extrapolation.partitioned_bank import PartitionedActivationBank


def _make_partitioned_bank(*, inference_capacity: int = 1024) -> PartitionedActivationBank:
    training_entries = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    training_valid_mask = torch.tensor([True, True, False], dtype=torch.bool)
    return PartitionedActivationBank(
        training_entries=training_entries,
        training_valid_mask=training_valid_mask,
        inference_capacity=inference_capacity,
    )


def test_training_partition_is_frozen_and_rejects_writes() -> None:
    """Inference-time writes must never reach the frozen training partition."""
    bank = _make_partitioned_bank()
    before_entries = bank.training_entries.clone()
    before_mask = bank.training_valid_mask.clone()

    with pytest.raises(RuntimeError, match="training partition is frozen"):
        bank.update_training_partition(torch.tensor([[9.0, 9.0, 9.0]], dtype=torch.float32))

    assert torch.equal(bank.training_entries, before_entries)
    assert torch.equal(bank.training_valid_mask, before_mask)


def test_inference_writes_only_change_the_inference_partition() -> None:
    """A write during inference must leave the training partition unchanged."""
    bank = _make_partitioned_bank()
    before_entries = bank.training_entries.clone()
    before_mask = bank.training_valid_mask.clone()

    bank.update_inference_partition(torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32))

    assert torch.equal(bank.training_entries, before_entries)
    assert torch.equal(bank.training_valid_mask, before_mask)
    assert bank.inference_occupancy == 1
    assert bank.inference_write_pointer == 1


def test_inference_partition_does_not_overwrite_during_protocol() -> None:
    """With capacity above 512, 512 writes must not wrap or overwrite."""
    bank = _make_partitioned_bank(inference_capacity=1024)
    for step in range(512):
        bank.update_inference_partition(torch.tensor([[float(step), 1.0, -1.0]], dtype=torch.float32))

    assert bank.inference_occupancy == 512
    assert bank.inference_write_pointer == 512
    assert bool(bank.inference_partition.valid_mask[:512].all())
    assert not bool(bank.inference_partition.valid_mask[512:].any())


def test_probe_queries_do_not_write_into_the_bank() -> None:
    """Query-only probe passes must not change inference occupancy."""
    bank = _make_partitioned_bank()
    bank.update_inference_partition(torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32))
    before_occupancy = bank.inference_occupancy
    before_pointer = bank.inference_write_pointer

    certainty = bank.query(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32))

    assert certainty.shape == (1,)
    assert bank.inference_occupancy == before_occupancy
    assert bank.inference_write_pointer == before_pointer


@pytest.fixture(scope="module")
def trained_seed0_e19_artifacts():
    """Train the approved E19 seed-0 model once for the checkpoint-0 baseline check."""
    return train_single_seed_artifacts(E19Config(seed=0))


def test_checkpoint0_certainty_matches_e19_novel_baseline(trained_seed0_e19_artifacts) -> None:
    """Checkpoint-0 certainty from the frozen training partition should match archived E19 seed 0."""
    artifacts = trained_seed0_e19_artifacts
    model = artifacts.model
    split = artifacts.dataset.test
    device = torch.device(artifacts.config.device)

    partitioned_bank = PartitionedActivationBank.from_live_bank(
        model.layer1_bank,
        inference_capacity=1024,
    ).to(device)

    novel_mask = split.levels == LEVEL_NOVEL
    with torch.no_grad():
        input_embedding = model.encode_inputs_with_jitter(
            split.part_a[novel_mask].to(device),
            split.part_b[novel_mask].to(device),
            part_a_noise=split.part_a_noise[novel_mask].to(device),
            part_b_noise=split.part_b_noise[novel_mask].to(device),
        )
        layer1_pre_gate = torch.relu(model.hidden1(input_embedding))
        checkpoint0_certainty = partitioned_bank.query(layer1_pre_gate).mean().item()

    archived_seed0_novel_baseline = 0.8510462045669556
    assert math.isfinite(checkpoint0_certainty)
    assert abs(checkpoint0_certainty - archived_seed0_novel_baseline) < 0.01
