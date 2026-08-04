"""Frozen inference-time protocol configuration for DIGIT Extrapolation E20."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class E20Config:
    inference_partition_capacity: int = 1024
    exposure_rounds: int = 32
    probe_repeats_per_combo: int = 32
    checkpoint_rounds: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)
    update_noise_seed_offset: int = 30_000
    probe_noise_seed_offset: int = 40_000
