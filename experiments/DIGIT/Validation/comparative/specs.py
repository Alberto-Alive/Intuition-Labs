"""Shared types for the comparative DIGIT-vs-DP suite."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class AttackMode(str, Enum):
    """Comparative evaluation mode."""

    MECHANISM = "mechanism"
    SYSTEM = "system"

    @property
    def min_group_size(self) -> int:
        return 1 if self is AttackMode.MECHANISM else 10

    @property
    def label(self) -> str:
        return self.value


@dataclass(frozen=True)
class SystemSpec:
    """Concrete system instantiation parameters."""

    family: str
    epsilon: Optional[float] = None
    digit_variant: str = "a6_s4_c3_r3"

    @property
    def system_key(self) -> str:
        if self.family == "dp_laplace":
            if self.epsilon is None:
                raise ValueError("DP system requires epsilon.")
            return f"dp_laplace_e{self.epsilon}".replace(".", "p")
        if self.family == "digit_stability_gate":
            return "digit_stability_gate"
        if self.family == "digit_specificity_gate":
            return "digit_specificity_gate"
        return self.family


@dataclass
class AttackRunConfig:
    """Shared run-level settings for privacy attacks."""

    dataset: str
    phase: str
    task: str
    mode: AttackMode
    variant: str
    seeds: List[int]
    query_budget: int
    results_root: Path
    seed_mode: str = "full"
    harness_version: str = "1.1"
    pass_fail_threshold: float = 0.60
    attacker_knowledge: str = "black-box query access under declared threat model"
    notes: str = ""


@dataclass(frozen=True)
class DatasetAttackSpec:
    """Dataset-specific attack metadata."""

    dataset_name: str
    hidden_attribute_name: str
    hidden_attribute_type: str = "binary"
    reconstruction_target_name: str = "label"
    canary_label: int = 1
    linkage_overlap_fraction: float = 0.50
    utility_task: str = "binary_classification"
    utility_metrics: List[str] = field(default_factory=list)
    canary_feature_values: Optional[List[int]] = None
    notes: Dict[str, str] = field(default_factory=dict)
